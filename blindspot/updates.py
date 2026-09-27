"""Self-update accounting and the verification backlog.

An agent that updates itself (weights, prompts, tools, configs, code) makes two demands on oversight:
  1. ACCOUNTING  every change to the agent must be declared, and the declared changes must match an independent record of what
                 actually changed. Anything else is an alarm. (This is the `books` idea applied to the agent itself.)
  2. PACE        changes have to be reviewed as fast as they arrive. If they arrive faster than they are reviewed, an unreviewed
                 backlog grows, and it is the backlog, not any single change, that is the pacing problem.

Input is JSONL, one record per line, four kinds (t in seconds or an ISO timestamp):
  {"type":"change",   "t":..., "artifact":"model/ckpt", "old":"<hash>", "new":"<hash>", "actor":"agent", "kind":"weights"}
  {"type":"snapshot", "t":..., "artifact":"model/ckpt", "hash":"<hash>"}         independent observation of the current state
  {"type":"review",   "t":..., "artifact":"model/ckpt", "new":"<hash>", "by":"alice", "result":"pass"}   a human or system verified it
"""
from __future__ import annotations

import json
import os
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from datetime import datetime

import numpy as np

from .commons import Ledger

FINDINGS = {
    "UNEXPLAINED_CHANGE": "the independent snapshot shows a state that no declared change produced",
    "PHANTOM_CHANGE": "a change was declared but the independent snapshot still shows the old state",
    "BROKEN_CHAIN": "a declared change starts from a state that does not match the last known state",
    "FAILED_REVIEW_STILL_LIVE": "a change failed review but is still the live state",
    "UNKNOWN_REVIEW": "a review refers to a change that was never declared",
    "UNREVIEWED_PAST_SLA": "a change was not reviewed within the allowed time",
}


def _t(x):
    if isinstance(x, (int, float)):
        return float(x)
    return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()


def load_records(path):
    rows = []
    for fp in ([path] if os.path.isfile(path) else sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jsonl"))):
        with open(fp, errors="ignore") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                    if o.get("type") in ("change", "snapshot", "review") and "t" in o and "artifact" in o:
                        o = dict(o); o["t"] = _t(o["t"]); rows.append(o)
                except Exception:
                    continue
    return sorted(rows, key=lambda r: r["t"])


def seal(records):
    """Hash-chain the declared records so later edits are detectable. Returns (ledger head, count). Keep the head somewhere else too."""
    L = Ledger()
    for r in records:
        L.append(r)
    return L.head(), len(records)


def reconcile(records, sla_hours=24.0, tol=60.0, end_t=None, seed=0):
    """Return {'findings': [...], 'pacing': {...}, 'artifacts': n}. tol = seconds of slack between a change and the snapshot that sees it.
    Runs in O(n log n): records are indexed per artifact, so hundreds of thousands of records are fine."""
    by = defaultdict(list)
    for r in records:
        by[r["artifact"]].append(r)
    findings, changes = [], []
    horizon = end_t if end_t is not None else (max(r["t"] for r in records) if records else 0.0)
    for art, recs in by.items():
        recs = sorted(recs, key=lambda r: r["t"])
        decl_t = [r["t"] for r in recs if r["type"] == "change"]
        new_times = defaultdict(list)
        for r in recs:
            if r["type"] == "change":
                new_times[r["new"]].append(r["t"])
        known = prev = None
        pending, art_changes = defaultdict(deque), []
        for r in recs:
            if r["type"] == "change":
                if known is not None and r.get("old") != known:
                    findings.append({"kind": "BROKEN_CHAIN", "artifact": art, "t": r["t"],
                                     "detail": f"declared old={r.get('old')} but last known state was {known}"})
                prev, known = known, r["new"]
                c = {"artifact": art, "t": r["t"], "new": r["new"], "review_t": None, "result": None}
                pending[r["new"]].append(c); art_changes.append(c); changes.append(c)
            elif r["type"] == "snapshot":
                h = r["hash"]
                if known is None:
                    known = h
                    continue
                if h == known:
                    continue
                ts = new_times.get(h)
                if ts:                                               # snapshot taken just before the change was logged
                    j = bisect_left(ts, r["t"])
                    if j < len(ts) and ts[j] - r["t"] <= tol:
                        continue
                if prev is not None and h == prev:
                    j = bisect_right(decl_t, r["t"]) - 1
                    if j >= 0 and r["t"] - decl_t[j] <= tol:         # snapshot lagging the change
                        continue
                    findings.append({"kind": "PHANTOM_CHANGE", "artifact": art, "t": r["t"],
                                     "detail": "declared new state never appeared; snapshot still shows the old one"})
                else:
                    findings.append({"kind": "UNEXPLAINED_CHANGE", "artifact": art, "t": r["t"],
                                     "detail": f"snapshot {h} matches no declared change (expected {known})"})
                prev, known = known, h                               # resync so one gap does not cascade
            elif r["type"] == "review":
                dq = pending.get(r.get("new"))
                if not dq:
                    findings.append({"kind": "UNKNOWN_REVIEW", "artifact": art, "t": r["t"],
                                     "detail": f"review of {r.get('new')} matches no declared change"})
                    continue
                c = dq.popleft()
                c["review_t"], c["result"] = r["t"], r.get("result", "pass")
        for c in art_changes:
            if c["result"] == "fail" and known == c["new"]:
                findings.append({"kind": "FAILED_REVIEW_STILL_LIVE", "artifact": art, "t": c["review_t"],
                                 "detail": f"{c['new']} failed review but is the current state"})
    sla = sla_hours * 3600.0
    for c in changes:
        if c["review_t"] is None and horizon - c["t"] > sla:
            findings.append({"kind": "UNREVIEWED_PAST_SLA", "artifact": c["artifact"], "t": c["t"], "detail": f"no review after {(horizon - c['t']) / 3600:.1f} h"})
        elif c["review_t"] is not None and c["review_t"] - c["t"] > sla:
            findings.append({"kind": "UNREVIEWED_PAST_SLA", "artifact": c["artifact"], "t": c["t"], "detail": f"reviewed after {(c['review_t'] - c['t']) / 3600:.1f} h"})
    return {"findings": sorted(findings, key=lambda f: f["t"]), "pacing": pacing(changes, horizon, sla, seed), "artifacts": len(by),
            "records": len(records), "sla_hours": sla_hours}


def _slope_ci(y, seed=0, boots=300):
    """Trend of an ordered series (per step) with a 95% interval from a moving-block bootstrap, since neighbouring points are correlated."""
    y = np.asarray(y, float)
    n = len(y)
    x = np.arange(n, dtype=float)
    fit = lambda xs, ys: float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 2 else 0.0      # noqa: E731
    if n < 8:
        return fit(x, y), -np.inf, np.inf
    rng, blk = np.random.default_rng(seed), max(2, n // 8)
    out = []
    for _ in range(boots):
        idx = np.concatenate([np.arange(s, min(s + blk, n)) for s in rng.integers(0, n - 1, int(np.ceil(n / blk)))])[:n]
        out.append(fit(x[idx], y[idx]))
    return fit(x, y), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def pacing(changes, horizon, sla, seed=0):
    if not changes:
        return {"changes": 0}
    ts = np.array(sorted(c["t"] for c in changes))
    rev = np.array(sorted(c["review_t"] for c in changes if c["review_t"] is not None))
    lag = np.array([c["review_t"] - c["t"] for c in changes if c["review_t"] is not None])
    span_days = max((horizon - ts[0]) / 86400.0, 1e-9)
    if span_days < 2.0:      # too little history for rates or a trend to mean anything
        n_rev = int(len(rev))
        return {"changes": len(ts), "reviewed": n_rev, "unreviewed_now": int(len(ts) - n_rev), "span_days": span_days, "too_short": True,
                "lag_median_h": float(np.median(lag) / 3600) if len(lag) else None, "lag_p95_h": float(np.percentile(lag, 95) / 3600) if len(lag) else None,
                "within_sla": float(np.mean(lag <= sla)) if len(lag) else 0.0, "falling_behind": False}
    npts = int(np.clip(span_days * 2, 20, 200))
    grid = np.linspace(ts[0], max(horizon, ts[-1]), npts)
    backlog = np.searchsorted(ts, grid, side="right") - np.searchsorted(rev, grid, side="right")
    step_days = (grid[1] - grid[0]) / 86400.0
    sl, lo, hi = _slope_ci(backlog, seed)
    n_rev = int(len(rev))
    return {"changes": len(ts), "reviewed": n_rev, "unreviewed_now": int(len(ts) - n_rev), "span_days": span_days,
            "changes_per_day": len(ts) / span_days, "reviews_per_day": n_rev / span_days,
            "peak_backlog": int(backlog.max()), "backlog_growth_per_day": sl / step_days, "growth_ci_per_day": [lo / step_days, hi / step_days],
            "lag_median_h": float(np.median(lag) / 3600) if len(lag) else None, "lag_p95_h": float(np.percentile(lag, 95) / 3600) if len(lag) else None,
            "within_sla": float(np.mean(lag <= sla)) if len(lag) else 0.0,
            "falling_behind": bool(lo / step_days > 0.0)}


def _p(x):
    return "n/a" if x is None else f"{x:.1f}"


def report_md(R, seal_info=None):
    p, f = R["pacing"], R["findings"]
    o = ["# blindspot: self-update accounting", "", f"{R['records']} records over {R['artifacts']} artifacts. SLA for review: {R['sla_hours']:g} h.", "",
         "## Accounting", ("**Clean.** Every observed state matches a declared change." if not [x for x in f if x["kind"] != "UNREVIEWED_PAST_SLA"] else
                           f"**{len([x for x in f if x['kind'] != 'UNREVIEWED_PAST_SLA'])} accounting problems.**")]
    if seal_info:
        o.append(f"Declared records are hash-chained. Ledger head `{seal_info[0][:16]}...` over {seal_info[1]} records. Store the head somewhere the agent cannot write.")
    o.append("")
    kinds = {}
    for x in f:
        kinds.setdefault(x["kind"], []).append(x)
    for k, xs in kinds.items():
        o += [f"### {k} ({len(xs)})", f"_{FINDINGS[k]}_", ""]
        o += [f"- `{x['artifact']}` at t={x['t']:.0f}: {x['detail']}" for x in xs[:8]] + ([f"- ...and {len(xs) - 8} more"] if len(xs) > 8 else []) + [""]
    o += ["## Pace"]
    if not p.get("changes"):
        o += ["No declared changes.", ""]
    else:
        if p.get("too_short"):
            o += [f"- Only {p['span_days']:.1f} days of history, so per-day rates and a backlog trend would be meaningless. Need at least 2 days.",
                  f"- Changes: {p['changes']}. Reviewed: {p['reviewed']}. Unreviewed right now: **{p['unreviewed_now']}**.", ""]
    if p.get("changes") and not p.get("too_short"):
        lo, hi = p["growth_ci_per_day"]
        o += [f"- Changes: **{p['changes_per_day']:.1f} per day**. Reviews completed: **{p['reviews_per_day']:.1f} per day**.",
              f"- Unreviewed right now: **{p['unreviewed_now']}** (peak backlog {p['peak_backlog']}).",
              f"- Backlog trend: **{p['backlog_growth_per_day']:+.2f} per day** (95% interval {lo:+.2f} to {hi:+.2f}).",
              f"- Review lag: median {_p(p['lag_median_h'])} h, 95th percentile {_p(p['lag_p95_h'])} h. Reviewed within the SLA: {100 * p['within_sla']:.0f}%.",
              "- **Verdict: " + ("the backlog is growing, so changes are arriving faster than they are being reviewed.**" if p["falling_behind"]
                                else "no sign the backlog is growing.**"), ""]
    o += ["## Limits",
          "- This checks the paperwork against an independent record. It cannot tell whether a change is safe, only that it was declared, seen, and reviewed.",
          "- The independent snapshot must come from somewhere the agent cannot write. If the agent controls both records, this passes.",
          "- A review here means someone or something recorded a result. It does not judge how good the review was."]
    return "\n".join(o) + "\n"


def write_outputs(R, out, seal_info=None):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "updates.md"), "w") as fh:
        fh.write(report_md(R, seal_info))
    with open(os.path.join(out, "updates.json"), "w") as fh:
        json.dump({**R, "ledger_head": seal_info[0] if seal_info else None}, fh, indent=1, default=str)


def simulate(days=30, changes_per_day=6.0, reviews_per_day=4.0, seed=0, unexplained=0, phantom=0):
    """Synthetic self-update history for demos and tests. Returns records. Review capacity below the change rate makes the backlog grow."""
    rng = np.random.default_rng(seed)
    recs, state = [], {"weights": "w0", "prompt": "p0", "tools": "t0"}
    t = 0.0
    n_change = int(days * changes_per_day)
    times = np.sort(rng.uniform(0, days * 86400, n_change))
    for a, h in state.items():
        recs.append({"type": "snapshot", "t": 0.0, "artifact": a, "hash": h})
    review_free = 0.0
    for i, tc in enumerate(times):
        art = list(state)[int(rng.integers(len(state)))]
        new = f"{art[0]}{i + 1}"
        recs.append({"type": "change", "t": float(tc), "artifact": art, "old": state[art], "new": new, "actor": "agent"})
        recs.append({"type": "snapshot", "t": float(tc + 30), "artifact": art, "hash": new})
        state[art] = new
        review_free = max(review_free, tc) + rng.exponential(86400.0 / reviews_per_day)     # one reviewer queue
        recs.append({"type": "review", "t": float(review_free), "artifact": art, "new": new, "by": "reviewer", "result": "pass"})
    horizon = days * 86400.0
    recs = [r for r in recs if r["t"] <= horizon]
    for j in range(unexplained):      # a state change nobody declared, on its own artifact so the cases stay independent
        recs += [{"type": "snapshot", "t": 0.0, "artifact": f"undeclared{j}", "hash": "u0"},
                 {"type": "snapshot", "t": float(rng.uniform(0.2, 0.9) * horizon), "artifact": f"undeclared{j}", "hash": "SNEAKY"}]
    for j in range(phantom):          # a change declared but never applied
        tc = float(rng.uniform(0.2, 0.8) * horizon)
        recs += [{"type": "snapshot", "t": 0.0, "artifact": f"ghost{j}", "hash": "g0"},
                 {"type": "change", "t": tc, "artifact": f"ghost{j}", "old": "g0", "new": "g1", "actor": "agent"},
                 {"type": "snapshot", "t": tc + 3600.0, "artifact": f"ghost{j}", "hash": "g0"}]
    return sorted(recs, key=lambda r: r["t"])


def _hash_file(path, chunk=16 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:                      # streamed, so multi-gigabyte checkpoints are fine
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hash_path(path):
    """sha256 of a file, or of a folder (sorted relative paths + file hashes). Skips .git."""
    import hashlib
    if os.path.isfile(path):
        return _hash_file(path)
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        for f in sorted(files):
            fp = os.path.join(root, f)
            h.update(os.path.relpath(fp, path).encode()); h.update(_hash_file(fp).encode())
    return h.hexdigest()


def snapshot(paths, t=None):
    """Independent observation records for the current state of files/folders (weights, prompts, tool configs).
    Run this from a system the agent cannot write to, on a schedule."""
    import time
    t = time.time() if t is None else t
    return [{"type": "snapshot", "t": t, "artifact": p, "hash": hash_path(p)} for p in paths]
