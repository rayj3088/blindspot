"""Core: Trace model (numpy arrays), synthetic baselines, profile fitting, log loaders."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np

KINDS = ["read", "write", "bash", "search", "test", "net"]
K = {k: i for i, k in enumerate(KINDS)}
CTRL_NAMES = ["AGENTS.md", "CLAUDE.md", ".claude/settings.json", ".cursor/rules",
              ".gemini/settings.json", ".vscode/tasks.json", ".github/workflows"]
N_CTRL, N_FILES = len(CTRL_NAMES), 300
_HINTS = [("agents.md", 0), ("claude.md", 1), (".claude/", 2), ("skills/", 2), ("hooks", 2),
          (".cursor/", 3), ("cursorrules", 3), (".gemini/", 4), ("gemini.md", 4),
          (".vscode/tasks", 5), (".github/workflows", 6)]
ZIPF = 1.0 / np.arange(1, N_FILES + 1) ** 1.1
ZIPF = ZIPF / ZIPF.sum()


def ctrl_id(path: str) -> int:
    """Index of the agent-control file class a path belongs to, or -1."""
    p = path.lower()
    for h, i in _HINTS:
        if h in p:
            return i
    return -1


def file_id(key: str) -> int:
    return N_CTRL + int(hashlib.md5(key.encode()).hexdigest(), 16) % N_FILES


_F = ("t", "agent", "kind", "target", "size", "value", "chan")


@dataclass
class Trace:
    """Event stream as parallel arrays. chan 0 = reported by the agent, 1 = independent record."""
    t: np.ndarray
    agent: np.ndarray
    kind: np.ndarray
    target: np.ndarray
    size: np.ndarray
    value: np.ndarray
    chan: np.ndarray

    @staticmethod
    def make(t, kind, target, size=0.0, value=np.nan, agent=0, chan=0):
        t = np.atleast_1d(np.asarray(t, float))
        n = len(t)

        def b(x, d):
            return np.broadcast_to(np.asarray(x, d), (n,)).copy()
        return Trace(t, b(agent, np.int16), b(kind, np.int8), b(target, np.int32),
                     b(size, float), b(value, float), b(chan, np.int8))

    @staticmethod
    def empty():
        return Trace.make([], 0, 0)

    def __len__(self):
        return len(self.t)

    def sel(self, m):
        return Trace(*(getattr(self, f)[m] for f in _F))

    @staticmethod
    def cat(*trs):
        trs = [x for x in trs if len(x)]
        if not trs:
            return Trace.empty()
        c = Trace(*(np.concatenate([getattr(x, f) for x in trs]) for f in _F))
        return c.sel(np.argsort(c.t, kind="stable"))

    @property
    def ctrl(self):
        return self.target < N_CTRL

    @property
    def rep(self):
        return self.sel(self.chan == 0)


@dataclass
class Profile:
    p_kind: np.ndarray
    trans: np.ndarray
    rate: float = 0.4          # events/sec while active
    active_frac: float = 0.35
    ctrl_p: float = 0.002      # chance a write hits an agent-control file
    size_mu: float = 6.0
    size_sd: float = 1.0
    net_mu: float = 8.0
    value_mu: float = 85.0
    value_sd: float = 6.0
    drop: float = 0.005        # independent channel misses this fraction
    win: float = 300.0         # analysis window (seconds)


_P = np.array([.45, .12, .18, .10, .10, .05])
_T = np.array([[.55, .12, .10, .10, .08, .05], [.25, .20, .15, .05, .30, .05],
               [.30, .15, .30, .05, .15, .05], [.50, .05, .10, .25, .05, .05],
               [.20, .35, .15, .10, .15, .05], [.40, .10, .20, .10, .10, .10]])
DEFAULT_PROFILE = Profile(_P / _P.sum(), _T / _T.sum(1, keepdims=True))


def fit_profile(tr: Trace) -> Profile:
    """Estimate a Profile from a real trace (reported channel)."""
    r = tr.rep
    if len(r) < 50:
        return DEFAULT_PROFILE
    k = r.kind.astype(int)
    p = np.bincount(k, minlength=6) + 1.0
    T = np.ones((6, 6))
    np.add.at(T, (k[:-1], k[1:]), 1)
    gaps = np.diff(r.t)
    act = gaps[gaps < 300].sum()
    span = max(r.t[-1] - r.t[0], 1.0)
    rate = len(r) / max(act, 1.0)
    w = r.kind == K["write"]
    pos = r.size > 0
    sz, nz = r.size[pos & (r.kind != K["net"])], r.size[pos & (r.kind == K["net"])]
    v = r.value[np.isfinite(r.value)]
    return Profile(p / p.sum(), T / T.sum(1, keepdims=True), rate=float(rate),
                   active_frac=float(np.clip(act / span, 0.02, 0.95)),
                   ctrl_p=float(max(r.ctrl[w].mean() if w.any() else 0.0, 1e-4)),
                   size_mu=float(np.log(sz).mean()) if len(sz) > 5 else 6.0,
                   size_sd=float(np.log(sz).std()) if len(sz) > 5 else 1.0,
                   net_mu=float(np.log(nz).mean()) if len(nz) > 5 else 8.0,
                   value_mu=float(v.mean()) if len(v) > 10 else 85.0,
                   value_sd=float(max(v.std(), 1e-3)) if len(v) > 10 else 6.0,
                   win=float(np.clip(40.0 / rate, 120.0, 1800.0)))


def synth_baseline(rng, hours=3.0, prof: Profile | None = None, agents=2, both=True) -> Trace:
    """Plausible clean agent activity: sessions, Markov tool sequences, Zipf file targets."""
    P = prof or DEFAULT_PROFILE
    T = hours * 3600.0
    act_mean = 2400.0
    idle_mean = act_mean * (1 - P.active_frac) / P.active_frac
    cur, state = 0.0, rng.random() < P.active_frac
    ts, ag = [], []
    while cur < T:
        d = rng.exponential(act_mean if state else idle_mean)
        if state:
            span = min(d, T - cur)
            n = rng.poisson(P.rate * span)
            if n:
                ts.append(cur + np.sort(rng.uniform(0, span, n)))
                ag.append(np.full(n, rng.integers(agents)))
        cur += d
        state = not state
    if not ts:
        return Trace.empty()
    t, agent = np.concatenate(ts), np.concatenate(ag)
    o = np.argsort(t, kind="stable")
    t, agent = t[o], agent[o]
    n = len(t)
    cum, u = np.cumsum(P.trans, axis=1), rng.random(n)
    k = np.empty(n, np.int8)
    k[0] = rng.choice(6, p=P.p_kind)
    for i in range(1, n):
        k[i] = min(np.searchsorted(cum[k[i - 1]], u[i]), 5)
    tgt = N_CTRL + rng.choice(N_FILES, n, p=ZIPF)
    hit = (k == K["write"]) & (rng.random(n) < P.ctrl_p)
    tgt[hit] = rng.integers(N_CTRL, size=int(hit.sum()))
    size = np.where(k == K["net"], np.exp(rng.normal(P.net_mu, 1.0, n)), np.exp(rng.normal(P.size_mu, P.size_sd, n)))
    val = np.where(k == K["test"], np.round(np.clip(rng.normal(P.value_mu, P.value_sd, n), 0, 100), 1), np.nan)
    rep = Trace.make(t, k, tgt, size, val, agent, 0)
    if not both:
        return rep
    keep = rng.random(n) > P.drop
    obs = Trace.make(t[keep] + rng.uniform(0, 0.5, int(keep.sum())), k[keep], tgt[keep], size[keep],
                     val[keep], agent[keep], 1)
    return Trace.cat(rep, obs)


# ---------------------------------------------------------------- loaders
_TOOL = {"Read": "read", "Glob": "search", "Grep": "search", "LS": "search", "Write": "write",
         "Edit": "write", "MultiEdit": "write", "NotebookEdit": "write", "Bash": "bash",
         "WebFetch": "net", "WebSearch": "net"}
_TESTS = ("pytest", "unittest", "npm test", "cargo test", "go test")


def _iso(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _files(path):
    if os.path.isfile(path):
        return [path]
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs if f.endswith(".jsonl"))


def load_claude_code(path: str) -> Trace:
    """Claude Code session transcripts (~/.claude/projects/**.jsonl): one event per tool_use block."""
    T, A, Kd, Tg, Sz = [], [], [], [], []
    for fp in _files(path):
        with open(fp, errors="ignore") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                m, ts = o.get("message"), o.get("timestamp")
                if o.get("type") != "assistant" or not ts or not isinstance(m, dict) or not isinstance(m.get("content"), list):
                    continue
                for b in m["content"]:
                    if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                        continue
                    inp, name = b.get("input") or {}, b.get("name", "")
                    p = inp.get("file_path") or inp.get("path") or inp.get("notebook_path") or ""
                    cmd = inp.get("command", "") or ""
                    kind = _TOOL.get(name, "bash")
                    if kind == "bash" and any(w in cmd for w in _TESTS):
                        kind = "test"
                    cid = ctrl_id(p) if p else -1
                    T.append(_iso(ts)); A.append(1 if o.get("isSidechain") else 0); Kd.append(K[kind])
                    Tg.append(cid if (cid >= 0 and kind == "write") else file_id(p or cmd or inp.get("url", "") or name))
                    Sz.append(float(len(json.dumps(inp))))
    if not T:
        return Trace.empty()
    tr = Trace.make(T, Kd, Tg, Sz, np.nan, A, 0)
    return tr.sel(np.argsort(tr.t, kind="stable"))


def load_jsonl(path: str, chan: int = 0) -> Trace:
    """Generic events: {"t":..,"kind"|"tool":..,"target":..,"size":..,"value":..,"agent":..,"chan":..}."""
    rows = []
    for fp in _files(path):
        with open(fp, errors="ignore") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if "t" in o and ("kind" in o or "tool" in o):
                    rows.append(o)
    if not rows:
        return Trace.empty()
    Kd, Tg = [], []
    for o in rows:
        kind = o.get("kind") or _TOOL.get(o.get("tool", ""), "bash")
        kind = kind if kind in K else "bash"
        tg = str(o.get("target", ""))
        cid = ctrl_id(tg)
        Kd.append(K[kind]); Tg.append(cid if (cid >= 0 and kind == "write") else file_id(tg or kind))
    tr = Trace.make([float(o["t"]) for o in rows], Kd, Tg, [float(o.get("size", 0.0)) for o in rows],
                    [float(o.get("value", np.nan)) for o in rows], [int(o.get("agent", 0)) for o in rows],
                    [int(o.get("chan", chan)) for o in rows])
    return tr.sel(np.argsort(tr.t, kind="stable"))


def load(path: str) -> Trace:
    tr = load_claude_code(path)
    return tr if len(tr) else load_jsonl(path)
