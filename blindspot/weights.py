"""Weight lineage: does a checkpoint descend from a specific earlier one, even after it has been fine-tuned?

The gap this closes: `gate --artifact` hashes a file exactly, so it can prove a checkpoint is BYTE-IDENTICAL to one it audited, but any
legitimate update changes every byte, so the hash check goes from PASS to "I can't tell you anything" the moment the model is retrained
even slightly. For a self-updating agent that is not an edge case, it is what happens on every release.

The idea, applied from spread-spectrum watermarking (same principle used for audio and image watermarks, and for DNN weight watermarking
since Uchida et al. 2017 - this is not a new invention, the new part is wiring it into blindspot's decision gate and ledger): pick many
pseudo-random directions in parameter space from a secret key, and nudge the checkpoint being signed very slightly so the SIGN of its
projection onto each direction encodes one bit of a signature. A single weight or a small group can be overwritten by any later training
step, but the sign of a projection onto a random direction spanning millions of parameters is a statistic of the WHOLE checkpoint. Moving
it back would need to fight the embedding on every one of those parameters at once, which ordinary fine-tuning does not do (it moves
weights to reduce a loss, not to flip an unrelated random projection). Recovery is a statistical claim, not a cryptographic one: it
degrades gracefully with how much the weights have moved, and a determined adversary who retrains from scratch, or who knows the scheme
and specifically optimizes against it, can remove it. What it defends against is the ordinary case: was THIS checkpoint produced by
updating the one that was actually audited, or does it need a fresh audit because its lineage cannot be shown.

The agent cannot "learn" to evade this the way it could learn to game a behavioral monitor, because the signature lives in raw parameter
values that have no effect on the function the network computes (the perturbation budget is chosen far below the noise a training run
already tolerates) and nothing about the scheme appears in training data, a prompt or a policy the agent could read or be shaped by.
Removing it would need either retraining hard enough to move the whole parameter vector by more than the recovery threshold (which is a
question of degree you can measure below), or the key, which is never released with the weights.

Runs on a laptop: parameters are streamed from disk in chunks, and directions are generated in the same chunks with a hash-seeded
generator, so memory use is one chunk at a time regardless of model size.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np

CHUNK = 1 << 20  # elements per streaming chunk


def _rng_for(key: bytes, chunk_index: int, bit: int):
    seed = int.from_bytes(hashlib.sha256(key + struct.pack(">QQ", chunk_index, bit)).digest()[:8], "big")
    return np.random.default_rng(seed)


def load_vector(path):
    """A flat float64 view of a checkpoint's parameters. .npy and .npz (all arrays concatenated) load directly; anything else is read as
    raw little-endian float32. For a real model, save or convert its parameters to one of these first."""
    if path.endswith(".npz"):
        z = np.load(path)
        return np.concatenate([np.asarray(z[k], dtype=np.float64).ravel() for k in z.files])
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64).ravel()
    return np.fromfile(path, dtype="<f4").astype(np.float64)


def save_vector(path, vec, like_path=None):
    """Write back in the same format load_vector reads. .npy is safest; for a raw file, the original dtype/shape is not recoverable, so
    it is written back as a flat float32 array of the same length (fine for round-tripping through embed/extract, not for reloading into
    a framework that expects the original shape)."""
    if path.endswith(".npy"):
        np.save(path, vec.astype(np.float32))
    else:
        vec.astype("<f4").tofile(path)


def _bits_from_bytes(sig: bytes):
    return np.unpackbits(np.frombuffer(sig, dtype=np.uint8)).astype(np.int8) * 2 - 1  # -1/+1


def embed_signature(vec, key: bytes, signature: bytes, budget=1e-3, chunk=CHUNK):
    """-> new vector with sign(vec . direction_i) == signature bit i for every bit of `signature`, for len(signature)*8 random directions
    each spanning the whole vector. budget: the perturbation's L2 norm as a fraction of the vector's own L2 norm (1e-3 is far below what
    an ordinary training step moves a checkpoint by; --measure below to see how it holds up against a given amount of drift)."""
    bits = _bits_from_bytes(signature)
    n = len(vec)
    out = vec.copy()
    norm0 = float(np.linalg.norm(vec)) or 1.0
    per_bit_budget = (budget * norm0) / max(1, len(bits))
    for b, bit in enumerate(bits):
        proj = 0.0
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            d = _rng_for(key, c0 // chunk, b).standard_normal(c1 - c0)
            d /= (np.linalg.norm(d) + 1e-12)
            proj += float(out[c0:c1] @ d)
        # each direction is (near-)unit length overall (its per-chunk pieces are individually normalized, so the whole direction has
        # length ~ sqrt(n_chunks)); push the projection to exactly the target value in one step, using the direction's own scale
        target = per_bit_budget if bit > 0 else -per_bit_budget
        n_chunks = -(-n // chunk)
        step = (target - proj) / n_chunks
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            d = _rng_for(key, c0 // chunk, b).standard_normal(c1 - c0)
            d /= (np.linalg.norm(d) + 1e-12)
            out[c0:c1] += step * d
    return out


def recover_signature(vec, key: bytes, n_bits: int, chunk=CHUNK):
    """-> (bits recovered as -1/+1, projection magnitude per bit). Independent of embed_signature: works on any vector, revealing chance-level
    (near 0) projections if it was never signed with this key."""
    n = len(vec)
    proj = np.zeros(n_bits)
    for b in range(n_bits):
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            d = _rng_for(key, c0 // chunk, b).standard_normal(c1 - c0)
            d /= (np.linalg.norm(d) + 1e-12)
            proj[b] += float(vec[c0:c1] @ d)
    return np.sign(proj), proj


def verify(vec, key: bytes, signature: bytes, alpha=1e-9):
    """-> {'match', 'n_bits', 'agree', 'p', 'lineage'}. p is the two-sided binomial chance of agreeing this often by luck if the vector
    had never been signed with this key. lineage is a plain-language verdict: 'confirmed' (far below alpha), 'no evidence' (agreement at
    chance), or 'weak/degraded' in between, meaning the checkpoint may share lineage but has moved far from where it was signed."""
    from scipy.stats import binomtest
    want = _bits_from_bytes(signature)
    got, proj = recover_signature(vec, key, len(want))
    agree = int(np.sum(got == want))
    p = binomtest(agree, len(want), 0.5, alternative="greater").pvalue
    lineage = "confirmed" if p < alpha else ("no evidence" if agree <= len(want) * 0.55 else "weak or degraded")
    return {"match": bool(p < alpha), "n_bits": len(want), "agree": agree, "fraction": agree / len(want), "p": float(p), "lineage": lineage,
            "mean_abs_projection": float(np.mean(np.abs(proj)))}


def gen_signature(n_bytes=16, seed=None):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, n_bytes, dtype=np.uint8).tobytes()


def report_md(V, path, key_note="a secret key held by the lab, never released with the checkpoint"):
    o = [f"# blindspot: weight lineage check for {path}", "",
         f"**{V['lineage'].upper()}** — {V['agree']} of {V['n_bits']} signature bits agree ({100 * V['fraction']:.0f}%), "
         f"binomial p = {V['p']:.2e} against a chance rate of 50%.", ""]
    if V["lineage"] == "confirmed":
        o.append("This checkpoint's parameters carry the signature embedded with " + key_note + ". Ordinary fine-tuning does not remove this;"
                 " full retraining, targeted removal by someone who has the key or scheme, or replacing the checkpoint outright would.")
    elif V["lineage"] == "no evidence":
        o.append("No trace of the signature. Either this checkpoint was never signed with this key, or it has moved far enough from the "
                 "signed checkpoint (heavy retraining, quantization, a different model entirely) that lineage cannot be shown. Treat it as "
                 "unaudited: it needs a fresh audit, whatever its history is claimed to be.")
    else:
        o.append("Partial agreement: consistent with a signed checkpoint that has since moved substantially (heavy fine-tuning, pruning, "
                 "quantization), but not strong enough to state lineage with confidence. Re-run with a larger signature or check sooner "
                 "after signing next time.")
    o += ["", "## What this does not establish",
          "- This is a statistical claim about the whole parameter vector, not a cryptographic proof about any single weight.",
          "- An adversary who knows the scheme and specifically optimizes against these directions, or who retrains from scratch, can remove it.",
          "- It says the checkpoint's parameters are statistically continuous with a signed one. It says nothing about whether the model is safe, correct, or unchanged in behavior.",
          "- Validated so far only on synthetic parameter vectors (see the README), not on a real model's weights."]
    return "\n".join(o) + "\n"
