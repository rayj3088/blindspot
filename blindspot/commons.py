"""Decentralised pieces: linear sketches (share aggregates, not logs), tamper-evident ledger, holographic code."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from .core import K, N_CTRL, Trace

P61 = (1 << 61) - 1


class Sketch:
    """Count-Min sketch. Linear: sketches from different machines ADD, so a commons can see global spikes
    without ever holding raw logs. Every cell carries a little of everything (a hologram, loosely)."""

    def __init__(self, depth=5, width=2048, seed=7):
        r = np.random.default_rng(seed)
        self.d, self.w = depth, width
        self.ab = [(int(r.integers(1, P61)), int(r.integers(0, P61))) for _ in range(depth)]
        self.tab = np.zeros((depth, width))

    def _idx(self, key):
        return [((a * key + b) % P61) % self.w for a, b in self.ab]

    def add(self, key, c=1.0):
        for r, i in enumerate(self._idx(key)):
            self.tab[r, i] += c

    def est(self, key):
        return float(min(self.tab[r, i] for r, i in enumerate(self._idx(key))))

    def merge(self, other):
        s = Sketch(self.d, self.w)
        s.ab, s.tab = self.ab, self.tab + other.tab
        return s

    def privatize(self, eps, rng):
        """Laplace noise so a contributor's sketch leaks little about any single event."""
        self.tab = self.tab + rng.laplace(0, self.d / eps, self.tab.shape)
        return self


def sketch_trace(tr: Trace, bin_s=60.0, seed=7):
    """Sketch of (control-file class, time bin) write counts: the shape of a lockstep campaign."""
    s = Sketch(seed=seed)
    r = tr.rep
    m = (r.kind == K["write"]) & r.ctrl
    for t, c in zip(r.t[m], r.target[m]):
        s.add(int(c) * 10_000_000 + int(t // bin_s))
    return s


def spike_scan(sk: Sketch, t_range, bin_s=60.0, z=5.0):
    """Flag (class, time-bin) cells far above the robust background. Returns [(class, bin_start, estimate)]."""
    b0, b1 = int(t_range[0] // bin_s), int(t_range[1] // bin_s) + 1
    grid = np.array([[sk.est(c * 10_000_000 + b) for b in range(b0, b1)] for c in range(N_CTRL)])
    med = np.median(grid)
    mad = max(np.median(np.abs(grid - med)) * 1.4826, 1.0)
    cls, b = np.where(grid - med > z * mad)
    return [(int(c), (b0 + int(i)) * bin_s, float(grid[c, i])) for c, i in zip(cls, b)]


# ------------------------------------------------------------------ ledger
def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Ledger:
    """Append-only hash chain. Publish `head()` to several independent places (git commits, mirrors):
    any edit to any earlier record changes every later head, so a single honest witness exposes it."""

    def __init__(self):
        self.recs = []

    def append(self, rec: dict):
        s = json.dumps(rec, sort_keys=True, separators=(",", ":"))
        prev = self.recs[-1][1] if self.recs else "0" * 64
        self.recs.append((s, _h((prev + s).encode())))

    def head(self):
        return self.recs[-1][1] if self.recs else "0" * 64

    def verify(self, witness_head=None):
        prev = "0" * 64
        for i, (s, h) in enumerate(self.recs):
            if _h((prev + s).encode()) != h:
                return False, i
            prev = h
        return (witness_head is None or witness_head == prev), None

    def merkle_root(self):
        lv = [_h(s.encode()) for s, _ in self.recs] or [_h(b"")]
        while len(lv) > 1:
            lv = [_h((lv[i] + lv[min(i + 1, len(lv) - 1)]).encode()) for i in range(0, len(lv), 2)]
        return lv[0]


def spotcheck_detect_prob(delta: float, k: int) -> float:
    """P(at least one of k independent random checks lands on a corrupted fraction delta)."""
    return 1 - (1 - delta) ** k


def commit(secret: bytes, salt: bytes | None = None):
    """Publish the digest now, reveal (secret, salt) later: a hold-out set nobody can tune against."""
    salt = salt or os.urandom(16)
    return _h(salt + secret), salt.hex()


def reveal_ok(secret: bytes, salt_hex: str, digest: str) -> bool:
    return _h(bytes.fromhex(salt_hex) + secret) == digest


# ------------------------------------------------------------------ holographic (Hadamard) code
def hadamard_encode(bits):
    """n data bits -> 2^n codeword bits; every codeword bit is a parity of a random-looking subset of the data."""
    n = len(bits)
    mask = sum(int(b) << i for i, b in enumerate(bits))
    v = np.arange(1 << n, dtype=np.int64) & mask
    for s in (16, 8, 4, 2, 1):
        v ^= v >> s
    return (v & 1).astype(np.uint8)


def corrupt(f, frac, rng):
    g = f.copy()
    g[rng.choice(len(f), int(frac * len(f)), replace=False)] ^= 1
    return g


def blr_reject_rate(f, trials, rng):
    """Blum-Luby-Rubinfeld linearity test: 3 queries per trial, rejects with probability >= the corrupted fraction."""
    x, y = rng.integers(0, len(f), trials), rng.integers(0, len(f), trials)
    return float(np.mean((f[x] ^ f[y]) != f[x ^ y]))


def local_decode(f, i, votes, rng):
    """Recover data bit i from a corrupted codeword with 2 queries per vote; survives ~10% damage."""
    r = rng.integers(0, len(f), votes)
    return int(np.mean(f[r] ^ f[r ^ (1 << i)]) > 0.5)
