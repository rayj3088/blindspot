"""Load the public TraceLab coding-agent traces (CC BY 4.0, https://github.com/uw-syfi/TraceLab) as blindspot Traces.

Written from TraceLab's published schema: JSONL, one row per LLM invocation, with `session_id`, `provider`, `model` and a
`tools[]` list whose entries have `tool_name`, `emitted_at`, `input_chars`, `result_chars`, `is_error`. Not yet run against the
real release file. What the schema strips (paths and tool inputs) limits what blindspot can see: the target-based detectors
(sweep, ctrl) are blind on this data. Timing, volume, tool mix and sequence are all there, which is what a staleness study needs.
"""
from __future__ import annotations

import gzip
import json
import re
from datetime import datetime

import numpy as np

from .core import K, Trace, file_id


def _ts(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        x = float(x)
        return x / 1000.0 if x > 1e11 else x
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


_RULES = [("net", r"web|fetch|http|curl|browser|mcp|url"), ("test", r"test|pytest"), ("write", r"edit|write|patch|apply|create|notebook|replace"),
          ("read", r"read|cat|view|open|head|tail"), ("search", r"grep|glob|search|find|list|^ls$")]


def kind_of(tool_name: str) -> str:
    n = (tool_name or "").lower()
    for kind, pat in _RULES:
        if re.search(pat, n):
            return kind
    return "bash"


def load_tracelab(path, group_by="model", gap_s=3600.0, max_rows=None):
    """-> {group label: Trace}. Sessions in a group are laid end to end (gap_s apart) so profiles reflect activity, not calendar overlap."""
    opener = gzip.open if str(path).endswith(".gz") else open
    sess = {}
    with opener(path, "rt", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if max_rows and i >= max_rows:
                break
            try:
                row = json.loads(line)
            except Exception:
                continue
            label = str(row.get(group_by) or "unknown")
            key = (label, str(row.get("session_id") or "s"))
            for tl in row.get("tools") or []:
                t = _ts(tl.get("emitted_at"))
                if t is None:
                    continue
                name = tl.get("tool_name") or ""
                sess.setdefault(key, []).append((t, K[kind_of(name)], file_id(name), float(tl.get("input_chars") or 0)))
    by_group = {}
    for (label, _), ev in sorted(sess.items(), key=lambda kv: min(e[0] for e in kv[1])):
        by_group.setdefault(label, []).append(sorted(ev))
    out = {}
    for label, sessions in by_group.items():
        T, Kd, Tg, Sz, offset = [], [], [], [], 0.0
        for ev in sessions:
            t0 = ev[0][0]
            for t, k, tg, sz in ev:
                T.append(offset + (t - t0)); Kd.append(k); Tg.append(tg); Sz.append(sz)
            offset += (ev[-1][0] - t0) + gap_s
        tr = Trace.make(T, Kd, Tg, Sz)
        out[label] = tr.sel(np.argsort(tr.t, kind="stable"))
    return out
