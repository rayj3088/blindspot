"""Attack shapes: known library, incident replays, random 'unknown shape' fuzzer, adaptive evasion search.

Attacks are *specs* (plain dicts) rendered into events, so every escape the fuzzer finds is a replayable,
shareable test case. Harm is measured in arbitrary units (see harm_score) so 'how much harm gets through
undetected' is a number, not a feeling.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from .core import DEFAULT_PROFILE, K, N_CTRL, N_FILES, ZIPF, Trace, synth_baseline

PRIMS: dict = {}


def prim(name, **defaults):
    def deco(f):
        f.defaults = defaults
        PRIMS[name] = f
        return f
    return deco


def _pair(rng, t, kind, target, size=0.0, value=np.nan, agent=0):
    """Reported (chan 0) events. Independent-channel twins are added later by _twin, after any time stretching."""
    return Trace.make(np.asarray(t, float), kind, target, size, value, agent, 0)


def _twin(rng, tr):
    """Independent-record copies of every reported event, a fraction of a second later."""
    r = tr.rep
    return Trace.make(r.t + rng.uniform(0, .5, len(r)), r.kind, r.target, r.size, r.value, r.agent, 1)


def _rf(rng, n):
    return N_CTRL + rng.integers(0, N_FILES, n)


@prim("burst", n=250, dur=20.0)
def p_burst(rng, t0, prof, n, dur):
    t = t0 + np.sort(rng.uniform(0, dur, int(max(1, n))))
    return _pair(rng, t, K["read"], _rf(rng, len(t))), {"burst": len(t)}


@prim("drip", n=60, dur=1800.0, jitter=0.05, size=3000.0)
def p_drip(rng, t0, prof, n, dur, jitter, size):
    n = int(max(2, n))
    t = np.sort(t0 + np.linspace(0, dur, n) + rng.normal(0, jitter * dur / n, n))
    return _pair(rng, t, K["net"], _rf(rng, n), size), {"exfil": n * size}


@prim("sweep", n=250, dur=300.0)
def p_sweep(rng, t0, prof, n, dur):
    n = int(max(1, n))
    t = t0 + np.sort(rng.uniform(0, dur, n))
    return _pair(rng, t, K["read"], N_CTRL + rng.permutation(N_FILES)[np.arange(n) % N_FILES]), {"recon": n}


@prim("ctrl", n=3, dur=60.0)
def p_ctrl(rng, t0, prof, n, dur):
    n = int(max(1, n))
    t = t0 + np.sort(rng.uniform(0, dur, n))
    return _pair(rng, t, K["write"], rng.integers(0, N_CTRL, n)), {"ctrl": n}


@prim("lockstep", n_agents=6, n=3, jitter=0.05)
def p_lockstep(rng, t0, prof, n_agents, n, jitter):
    na, n = int(max(2, n_agents)), int(max(1, n))
    t = (t0 + np.arange(n)[None, :] * 7.0 + rng.normal(0, jitter, (na, n))).ravel()
    ag = np.repeat(10 + np.arange(na), n)
    tg = np.tile(rng.integers(0, N_CTRL, n), na)
    o = np.argsort(t)
    return _pair(rng, t[o], K["write"], tg[o], agent=ag[o]), {"ctrl": na * n}


@prim("fabricate", n=90, dur=600.0, mode=0.0)
def p_fab(rng, t0, prof, n, dur, mode):
    n = int(max(2, n))
    t = t0 + np.sort(rng.uniform(0, dur, n))
    v = np.full(n, 100.0) if mode < 0.5 else np.round(np.round(rng.normal(prof.value_mu, prof.value_sd * .15, n) * 2) / 2, 1)
    return _pair(rng, t, K["test"], _rf(rng, n), value=v), {"fabricated": n}


@prim("unlogged", n=20, dur=300.0)
def p_unlogged(rng, t0, prof, n, dur):
    n = int(max(1, n))
    t = t0 + np.sort(rng.uniform(0, dur, n))
    return Trace.make(t, rng.choice([K["bash"], K["net"]], n), _rf(rng, n), 500.0, np.nan, 0, 1), {"unlogged": n}


@prim("loop", n=120, dur=240.0)
def p_loop(rng, t0, prof, n, dur):
    n = int(max(3, n))
    t = np.sort(t0 + np.linspace(0, dur, n) + rng.normal(0, .01 * dur / n, n))
    return _pair(rng, t, K["bash"], int(_rf(rng, 1)[0])), {"loop": n}


@prim("chain", pace=1.0, exfil=120000.0)
def p_chain(rng, t0, prof, pace, exfil):
    phases = [("search", 5, 60), ("read", 20, 120), ("bash", 5, 60), ("ctrl", 2, 30), ("net", 6, 60)]
    t, parts = t0, []
    for name, c, d in phases:
        ts = t + np.sort(rng.uniform(0, d * pace, c))
        if name == "ctrl":
            parts.append(_pair(rng, ts, K["write"], rng.integers(0, N_CTRL, c)))
        elif name == "net":
            parts.append(_pair(rng, ts, K["net"], _rf(rng, c), exfil / c))
        else:
            parts.append(_pair(rng, ts, K[name], _rf(rng, c)))
        t += d * pace
    return Trace.cat(*parts), {"exfil": exfil, "ctrl": 2}


def harm_score(h: dict) -> float:
    return (h.get("exfil", 0) / 1e5 + 5 * h.get("ctrl", 0) + h.get("unlogged", 0) + h.get("fabricated", 0) / 20
            + h.get("recon", 0) / 100 + h.get("loop", 0) / 50 + h.get("burst", 0) / 100)


def _filler(rng, prof, t):
    n = len(t)
    k = rng.choice(6, size=n, p=prof.p_kind)
    val = np.where(k == K["test"], np.round(np.clip(rng.normal(prof.value_mu, prof.value_sd, n), 0, 100), 1), np.nan)
    return _pair(rng, t, k, N_CTRL + rng.choice(N_FILES, n, p=ZIPF), np.exp(rng.normal(prof.size_mu, prof.size_sd, n)), val)


def render(spec, rng, prof=DEFAULT_PROFILE):
    """spec {"t0","prims":[[name,params]..],"camo","dilate"} -> (attack Trace, harm dict, (start, end))."""
    t0, parts, harm = float(spec["t0"] or 0.0), [], {}
    for name, params in spec["prims"]:
        tr, h = PRIMS[name](rng, t0, prof, **{**PRIMS[name].defaults, **params})
        parts.append(tr)
        for k, v in h.items():
            harm[k] = harm.get(k, 0.0) + v
    tr = Trace.cat(*parts)
    d = spec.get("dilate", 1.0)
    if len(tr) and d != 1.0:
        tr.t = t0 + (tr.t - t0) * d
    c = spec.get("camo", 0.0)
    if len(tr) and c > 0:
        n = int(c * len(tr.rep))
        if n:
            tr = Trace.cat(tr, _filler(rng, prof, rng.uniform(tr.t.min(), tr.t.max() + 1e-6, n)))
    tr = Trace.cat(tr, _twin(rng, tr))
    return tr, harm, ((float(tr.t.min()), float(tr.t.max())) if len(tr) else (t0, t0))


def _s(*prims, **kw):
    return {"t0": None, "prims": [list(p) for p in prims], "camo": 0.0, "dilate": 1.0, **kw}


LIBRARY = {"burst_read": _s(("burst", {})), "recon_sweep": _s(("sweep", {})),
           "rulebook_tamper": _s(("ctrl", {"n": 4})), "lockstep_fleet": _s(("lockstep", {})),
           "fabricated_metrics": _s(("fabricate", {})), "unlogged_actions": _s(("unlogged", {})),
           "stuck_loop": _s(("loop", {"n": 400, "dur": 200.0})), "slow_exfil": _s(("drip", {"n": 60, "dur": 5400.0})),
           "kill_chain": _s(("chain", {}))}

# ------------------------------------------------------------------ incident replays
_INC = os.path.join(os.path.dirname(__file__), "incidents")


def list_incidents():
    return sorted(f[:-5] for f in os.listdir(_INC) if f.endswith(".json"))


def load_incident(name):
    p = name if os.path.exists(name) else os.path.join(_INC, name + ".json")
    with open(p) as fh:
        return json.load(fh)


def render_incident(inc, rng, t0=0.0, compress=1.0):
    """Approximate replay of a published incident from its phase structure (NOT a reconstruction)."""
    D, parts, harm = float(inc["duration_s"]) * compress, [], {"exfil": 0.0, "ctrl": 0.0}
    for ph in inc["phases"]:
        n = int(ph["actions"])
        t = t0 + D * (ph["start"] + (ph["end"] - ph["start"]) * np.sort(rng.random(n)))
        mix = ph["mix"]
        toks = rng.choice(list(mix), n, p=np.array(list(mix.values())) / sum(mix.values()))
        kind = np.array([K["write"] if x == "ctrl" else K[x] for x in toks])
        tg = np.where(toks == "ctrl", rng.integers(0, N_CTRL, n), _rf(rng, n))
        sz = np.where(toks == "net", ph.get("net_bytes_each", 4000.0), 300.0)
        parts.append(_pair(rng, t, kind, tg, sz))
        harm["exfil"] += float(sz[toks == "net"].sum()); harm["ctrl"] += float((toks == "ctrl").sum())
    tr = Trace.cat(*parts)
    tr = Trace.cat(tr, _twin(rng, tr))
    return tr, harm, (float(tr.t.min()), float(tr.t.max()))


def run_incident(stack, prof, inc, seed=0, compress=1.0):
    rng = np.random.default_rng(seed)
    hours = inc["duration_s"] * compress / 3600.0 + 24.0
    base = synth_baseline(rng, hours, prof)
    atk, h, span = render_incident(inc, rng, 6 * 3600.0, compress)
    ev = stack.evaluate(Trace.cat(base, atk), prof, span, clean=stack.series(base, prof))
    ev["frac_to_alarm"] = None if ev["first"] is None else float(np.clip((ev["first"] - span[0]) / max(span[1] - span[0], 1e-9), 0, 1))
    return ev


# ------------------------------------------------------------------ bench, fuzzer, adaptive attacker
class Bench:
    """Pre-generated clean baselines; attacks are injected at random times and judged inside their span."""

    def __init__(self, stack, prof, hours=3.0, n_base=6, seed=100):
        rng = np.random.default_rng(seed)
        self.stack, self.prof, self.hours = stack, prof, hours
        self.bases = [synth_baseline(rng, hours, prof) for _ in range(n_base)]
        self.clean = [stack.series(b, prof) for b in self.bases]

    def run(self, spec, rng):
        i = int(rng.integers(len(self.bases)))
        t0 = float(rng.uniform(300, 0.15 * self.hours * 3600 + 300))
        atk, h, span = render({**spec, "t0": t0}, rng, self.prof)
        ev = self.stack.evaluate(Trace.cat(self.bases[i], atk), self.prof, span, clean=self.clean[i])
        ev["t0"] = t0
        return ev, harm_score(h), span


BOUNDS = {"n": (1, 600), "dur": (5, 3000), "jitter": (0.0, 1.0), "size": (100, 50000), "n_agents": (2, 30),
          "pace": (0.05, 10), "exfil": (1e3, 5e5), "mode": (0.0, 1.0)}
_LINEAR = ("mode", "jitter")


def random_spec(rng, max_prims=3):
    """Sample an unknown attack shape from the grammar: 1-3 primitives, random parameters, camouflage, pace."""
    names = list(PRIMS)
    prims = []
    for name in [names[i] for i in rng.integers(0, len(names), int(rng.integers(1, max_prims + 1)))]:
        params = {}
        for p in PRIMS[name].defaults:
            lo, hi = BOUNDS[p]
            params[p] = float(rng.uniform(lo, hi)) if p in _LINEAR else float(np.exp(rng.uniform(np.log(max(lo, 1e-3)), np.log(hi))))
        prims.append([name, params])
    return {"t0": None, "prims": prims, "camo": float(rng.choice([0, 0, 0.5, 1, 2, 4])),
            "dilate": float(np.exp(rng.uniform(np.log(.3), np.log(4))))}


def mutate(spec, rng):
    s = copy.deepcopy(spec)
    r = rng.random()
    if r < 0.6 and s["prims"]:
        name, params = s["prims"][int(rng.integers(len(s["prims"])))]
        full = {**PRIMS[name].defaults, **params}
        p = list(full)[int(rng.integers(len(full)))]
        lo, hi = BOUNDS[p]
        v = full[p] + rng.normal(0, .2) if p in _LINEAR else full[p] * np.exp(rng.normal(0, .5))
        params[p] = float(np.clip(v, lo, hi))
    elif r < 0.8:
        s["camo"] = float(np.clip(s.get("camo", 0) + rng.normal(0, 1.0), 0, 6))
    elif r < 0.95:
        s["dilate"] = float(np.clip(s.get("dilate", 1) * np.exp(rng.normal(0, .4)), .3, 4))
    elif len(s["prims"]) < 3:
        n = list(PRIMS)[int(rng.integers(len(PRIMS)))]
        s["prims"].append([n, {}])
    return s


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0, c - h)), float(min(1, c + h))


def fuzz(bench, n=200, seed=1, h_min=1.0):
    """Unknown-shape coverage: detection rate over n randomly generated attacks with harm >= h_min."""
    rng = np.random.default_rng(seed)
    rows, tries = [], 0
    while len(rows) < n and tries < 10 * n:
        spec = random_spec(rng)
        run_seed = int(rng.integers(1 << 30))
        ev, h, _ = bench.run(spec, np.random.default_rng(run_seed))
        tries += 1
        if h >= h_min:
            rows.append({"spec": spec, "seed": run_seed, "harm": h, **ev})
    det = np.array([r["detected"] for r in rows], bool)
    lo, hi = wilson(int(det.sum()), len(det))
    return {"n": len(rows), "coverage": float(det.mean()) if len(det) else 0.0, "ci": (lo, hi), "rows": rows}


def evade(bench, spec0, seed=0, budget=80, constrain=True):
    """Adaptive attacker: (1+4) evolution over the spec, maximising harm while staying under every threshold."""
    rng = np.random.default_rng(seed)

    def fit(spec):
        ev, h, _ = bench.run(spec, np.random.default_rng(seed + 1))
        m = max(ev["norm"].values())
        return (h * np.exp(-3 * max(0.0, m - 0.9)) if constrain else h), h, ev["detected"]
    cur, (f, h, det) = copy.deepcopy(spec0), fit(spec0)
    best_un = (h, copy.deepcopy(cur)) if not det else (0.0, None)
    hist = [(h, det)]
    for _ in range(max(1, budget // 4)):
        cands = [mutate(cur, rng) for _ in range(4)]
        res = [(fit(c), c) for c in cands]
        for (fc, hc, dc), c in res:
            hist.append((hc, dc))
            if not dc and hc > best_un[0]:
                best_un = (hc, copy.deepcopy(c))
        (fb, hb, db), cb = max(res, key=lambda x: x[0][0])
        if fb >= f:
            cur, f = cb, fb
    return {"undetected_harm": float(best_un[0]), "spec": best_un[1], "history": hist}
