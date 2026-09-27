"""Audit monitors that read transcripts, on labeled transcripts you already have.

Frontier oversight leans on monitors that read what an agent says and does, usually language models. blindspot audits them
on a labeled set you supply (your red-team transcripts, or a public benchmark) and measures what a lab needs to know
about the monitors as a SET:

  detection       each monitor's catch rate at a false-alarm rate calibrated on clean transcripts, with intervals
  independence    how many monitors are independent once near-copies are merged, disjoint groups that each catch 90% of what all
                  catch, and the correlation between their misses (a set of monitors that all miss the same transcripts is one opinion)
  shared misses   the transcripts no monitor catches, grouped by any label you attach (attack family, task, source)
  transfer        monitors calibrated on clean transcripts of one model and tested on another (the tool-call result, for transcripts)
  regression      a saved baseline, so a new monitor version or a new model fails the check only if oversight got worse

Input: JSONL, one transcript per line: {"id", "label": "attack" | "clean", "text" or "messages", optional "model", "family"}.
Monitors: a command reading one transcript as JSON on stdin and printing a score or true/false, or a Python function(dict) -> score.
Scores are cached per monitor and transcript, so an expensive model-based monitor is called once per transcript.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys

import numpy as np

from .attacks import wilson
from .audit import collapse_duplicates, failure_correlation, redundancy_groups


def load(path):
    rows = []
    with open(path, errors="ignore") as fh:
        for i, line in enumerate(fh):
            try:
                o = json.loads(line)
            except Exception:
                continue
            lab = str(o.get("label", "")).lower()
            if lab not in ("attack", "clean") or not (o.get("text") or o.get("messages")):
                continue
            o.setdefault("id", str(i)); o.setdefault("model", "unknown"); o.setdefault("family", "unlabeled")
            rows.append(o)
    return rows


class CmdScorer:
    def __init__(self, cmd, timeout=600):
        self.argv = shlex.split(cmd)
        if self.argv and self.argv[0] in ("python", "python3"):
            self.argv[0] = sys.executable
        self.timeout, self.name = timeout, cmd

    def __call__(self, row):
        p = subprocess.run(self.argv, input=json.dumps(row), capture_output=True, text=True, timeout=self.timeout)
        if p.returncode != 0:
            raise RuntimeError(f"monitor failed ({p.returncode}): {p.stderr.strip()[:300]}")
        from .external import parse_result
        return parse_result(p.stdout)


def py_scorer(spec):
    import importlib
    mod, _, fn = spec.rpartition(":")
    if mod.endswith(".py") or os.path.exists(mod):
        sys.path.insert(0, os.path.dirname(os.path.abspath(mod)) or ".")
        mod = os.path.splitext(os.path.basename(mod))[0]
    return getattr(importlib.import_module(mod), fn)


def score_all(rows, monitors, cache_dir=None, log=None):
    """-> {monitor: np.array of float scores (bools become 0/1)}; cached on disk by monitor name and transcript content."""
    say = log or (lambda *_: None)
    out = {}
    for name, fn in monitors.items():
        cache = {}
        cp = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cp = os.path.join(cache_dir, hashlib.sha256(name.encode()).hexdigest()[:16] + ".json")
            if os.path.exists(cp):
                cache = json.load(open(cp))
        vals = []
        
