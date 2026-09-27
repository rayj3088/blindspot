"""Audit transcript-reading monitors (the language-model monitors labs rely on most), the same way the rest of blindspot audits
behavioral ones: coverage, independence, and stability, measured on labels YOU supply.

blindspot does not generate attacks or evasive text here. It takes a corpus you already have (flagged transcripts and clean ones,
each labeled) and your monitor as a black box (a command or a function that reads a transcript and returns flag/score), and reports:

  coverage        share of your flagged transcripts each monitor catches, and its false-positive rate on your clean ones, at a
                  threshold set on the clean set so the monitors are compared at equal false-positive cost.
  independence    across several monitors, how many are really independent (near-duplicate verdicts merged), how many disjoint groups
                  each catch most of what any of them catch, and which transcripts only one monitor catches. Same redundancy idea as the
                  behavioral gate, applied to your LM monitors.
  stability       whether a monitor's verdict survives label-preserving reformatting of the SAME transcript (whitespace, capitalization,
                  Unicode normalization, blank lines). A verdict that flips on cosmetic changes is a brittle threshold, not a caught
                  attack. This is a robustness check on the monitor, not an attack: the transforms cannot change what a transcript means.

A transcript is a list of turns [{"role": ..., "text": ...}] or a plain string. Labels are 1 for a transcript that should be flagged
and 0 for one that should not. What counts as flaggable is your definition, carried entirely by your labels.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import unicodedata

import numpy as np

from .audit import collapse_duplicates, obstruction_rate, redundancy_groups
from .attacks import wilson


def render(transcript):
    if isinstance(transcript, str):
        return transcript
    return "\n".join(f"{t.get('role', '')}: {t.get('text', '')}".strip() for t in transcript)


def load_labeled(path):
    """JSONL rows: {"text": str | [turns], "label": 0|1, "id"?}. Accepts "transcript"/"messages" for the text and "flag"/"y" for the label."""
    rows = []
    with open(path, errors="ignore") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            text = o.get("text") if o.get("text") is not None else o.get("transcript") if o.get("transcript") is not None else o.get("messages")
            lab = o.get("label") if o.get("label") is not None else o.get("flag") if o.get("flag") is not None else o.get("y")
            if text is None or lab is None:
                continue
            rows.append({"id": o.get("id", i), "text": text, "label": int(lab)})
    return rows


# ---- label-preserving reformattings: cosmetic only, cannot change meaning ----
def _t_whitespace(s):
    return "\n".join(" ".join(line.split()) for line in s.splitlines())


def _t_case(s):
    return s.upper()


def _t_nfkc(s):
    return unicodedata.normalize("NFKC", s)


def _t_blank_lines(s):
    return "\n\n".join(s.splitlines())


def _t_trailing(s):
    return "\n".join(line + "   " for line in s.splitlines())


TRANSFORMS = {"whitespace collapsed": _t_whitespace, "uppercased": _t_case, "Unicode NFKC": _t_nfkc,
              "blank lines added": _t_blank_lines, "trailing spaces": _t_trailing}


class PyTranscriptMonitor:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, transcript):
        return self.fn(render(transcript))


class CmdTranscriptMonitor:
    def __init__(self, cmd):
        self.argv = shlex.split(cmd)
        if self.argv and self.argv[0] in ("python", "python3"):
            self.argv[0] = sys.executable
        self.cmd = cmd

    def __call__(self, transcript):
        p = subprocess.run(self.argv, input=render(transcript), capture_output=True, text=True, timeout=120)
        out = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        try:
            o = json.loads(out)
            v = o.get("flag", o.get("score", o.get("alarm")))
            return float(v) if not isinstance(v, bool) else v
        except (ValueError, AttributeError):
            low = out.lower()
            if low in ("true", "flag", "1", "yes"):
                return True
            if low in ("false", "0", "no", ""):
                return False
            try:
                return float(out)
            except ValueError:
                return False


def load_py(spec):
    import importlib
    import importlib.util
    if ":" not in spec:
        raise SystemExit("use module:function or file.py:function")
    mod, fn = spec.rsplit(":", 1)
    if mod.endswith(".py"):
        s = importlib.util.spec_from_file_location("m", mod); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return PyTranscriptMonitor(getattr(m, fn))


def _is_bool(x):
    return isinstance(x, (bool, np.bool_))


def _calibrate(mon, cleans, far):
    raw = [mon(c) for c in cleans]
    scored = any(not _is_bool(x) for x in raw)
    thr = float(np.quantile(np.asarray(raw, float), 1 - far, method="higher")) if scored else None
    alarm = (lambda t, thr=thr: bool(mon(t)) if thr is None else float(mon(t)) > thr)
    return alarm, scored, thr


def audit(monitors: dict, examples, far=0.05, stability_n=60, seed=0, log=None):
    say = log or (lambda *_: None)
    pos = [e for e in examples if e["label"] == 1]
    neg = [e for e in examples if e["label"] == 0]
    if not pos or not neg:
        raise SystemExit(f"need both flagged and clean examples; got {len(pos)} flagged, {len(neg)} clean")
    names = list(monitors)
    alarms = {}
    R = {"n_flagged": len(pos), "n_clean": len(neg), "monitors": names, "far_target": far, "per_monitor": {}}
    det = {}
    for n in names:
        say(f"[{n}] calibrating threshold on {len(neg)} clean transcripts")
        alarm, scored, thr = _calibrate(monitors[n], [e["text"] for e in neg], far)
        alarms[n] = alarm
        fp = int(sum(alarm(e["text"]) for e in neg))
        caught = [bool(alarm(e["text"])) for e in pos]
        det[n] = caught
        cov = sum(caught)
        R["per_monitor"][n] = {"returns": "score" if scored else "flag", "threshold": thr,
                               "coverage": cov / len(pos), "coverage_ci": list(wilson(cov, len(pos))),
                               "false_positive": fp / len(neg), "false_positive_ci": list(wilson(fp, len(neg)))}
    # independence across monitors, over the flagged transcripts any monitor catches
    if len(names) >= 2:
        vec = {n: [int(x) for x in det[n]] for n in names}
        live = [n for n in names if any(vec[n])]
        rep, merged = collapse_duplicates({n: vec[n] for n in live}) if live else ({}, [])
        reps = sorted(set(rep.values()), key=names.index)
        caught_sets = [f for f in ({rep[n] for n in live if det[n][i]} for i in range(len(pos))) if f]
        R["independence"] = {"named": len(names), "ever_fire": len(live), "independent": len(set(rep.values())),
                             "near_copies": merged, "groups": redundancy_groups(caught_sets, reps),
                             "obstruction": obstruction_rate(caught_sets, reps),
                             "only_one_catches": float(np.mean([len(f) == 1 for f in caught_sets])) if caught_sets else 0.0,
                             "caught_by_none": float(1 - len(caught_sets) / len(pos))}
    # stability: does a verdict survive cosmetic reformatting of the same transcript?
    rng = np.random.default_rng(seed)
    sample = [pos[i] for i in rng.permutation(len(pos))[:min(stability_n, len(pos))]]
    R["stability"] = {}
    for n in names:
        flips = {k: 0 for k in TRANSFORMS}
        base = {id(e): alarms[n](e["text"]) for e in sample}
        for e in sample:
            for k, fn in TRANSFORMS.items():
                t = e["text"] if isinstance(e["text"], str) else render(e["text"])
                if bool(alarms[n](fn(t))) != bool(base[id(e)]):
                    flips[k] += 1
        R["stability"][n] = {"n": len(sample), "flip_rate": {k: flips[k] / max(1, len(sample)) for k in TRANSFORMS},
                             "worst": max(flips.values()) / max(1, len(sample))}
    return R


def _p(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def report_md(R):
    o = ["# blindspot: transcript-monitor audit", "",
         f"Your labels: {R['n_flagged']} transcripts that should be flagged, {R['n_clean']} that should not. Thresholds set on the clean set "
         f"at a {_p(R['far_target'])} false-positive target, so monitors compare at equal cost.", "",
         "## Coverage", "", "| monitor | catches flagged | false positives |", "|---|---|---|"]
    for n in R["monitors"]:
        m = R["per_monitor"][n]
        o.append(f"| {n} | {_p(m['coverage'])} ({_p(m['coverage_ci'][0])}-{_p(m['coverage_ci'][1])}) | {_p(m['false_positive'])} ({_p(m['false_positive_ci'][0])}-{_p(m['false_positive_ci'][1])}) |")
    ind = R.get("independence")
    if ind:
        o += ["", "## Independence",
              f"{ind['named']} monitors named, {ind['ever_fire']} catch anything, {ind['independent']} independent after merging near-duplicate verdicts. "
              f"{len(ind['groups'])} disjoint group(s) each catch 90%+ of what any monitor catches. "
              f"{_p(ind['only_one_catches'])} of caught transcripts are caught by only one monitor, and {_p(ind['caught_by_none'])} of flagged transcripts are caught by none."]
        if ind["near_copies"]:
            o.append("Near-duplicate monitors (verdicts correlated 0.95+), counted as one: " + "; ".join("{" + ", ".join(g) + "}" for g in ind["near_copies"]) + ".")
        if ind["independent"] < 2:
            o.append("**One independent monitor: no real second opinion at the transcript layer.**")
    o += ["", "## Stability under cosmetic reformatting",
          "Share of flagged transcripts whose verdict flips when the same transcript is only reformatted (meaning unchanged). Lower is better; a high number is a brittle threshold.", "",
          "| monitor | " + " | ".join(TRANSFORMS) + " |", "|---|" + "---|" * len(TRANSFORMS)]
    for n in R["monitors"]:
        s = R["stability"][n]
        o.append(f"| {n} | " + " | ".join(_p(s["flip_rate"][k]) for k in TRANSFORMS) + " |")
    o += ["", "A monitor that flags an attack but drops the flag when the transcript is only reindented was keying on formatting, not on what happened."]
    return "\n".join(o) + "\n"
