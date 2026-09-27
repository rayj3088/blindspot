"""Ablation: do the added pieces earn their place against plain baselines, on YOUR real data?

A reviewer's first objection to borrowed ideas (proper time, redundancy counting) is that they are decoration unless they beat what a
plain engineer would build. This runs the same scale-matched attacks and the same evasion search against:
  plain     the built-in detectors without `clock`
  +clock    the same detectors plus `clock`
and reports, per attack shape, the smallest strength caught in 80% of runs, the strongest attack the search could hide, and the clean
alert rate, so the comparison is at equal false-alarm cost. It also reports how many monitors a naive count sees against how many are
independent after near-copies are merged.
"""
from __future__ import annotations

import numpy as np

from . import detect_real, external, lenses
from .attacks import wilson
from .audit import collapse_duplicates


def _run(chunks, prof, with_clock, seed, reps, budget, n_cal, n_val):
    had = "clock" in lenses.DEFAULT
    if with_clock and not had:
        lenses.DEFAULT.insert(5, "clock")
    if not with_clock and had:
        lenses.DEFAULT.remove("clock")
    try:
        R, st = external.audit_monitor({"s": external.BuiltinLenses(split=False)}, seed=seed, n_cal=n_cal, n_val=n_val, n_base=8, reps=reps,
                                       prof=prof, real_chunks=chunks, mode="scaled", chronological=True, return_state=True)
        alarm = st["alarm"]["built-in"]
        clean = [b for b in st["val"] if not alarm(b)][:20]
        pm = R["per_monitor"]["built-in"]
        fa = pm["false_alarm"]
        out = {"false_alarm": fa["rate"], "false_alarm_ci": fa["ci"], "families": {}}
        for fam in detect_real.FAMILIES:
            a = detect_real.evade_blackbox(alarm, clean, st["S"], fam, budget=budget, seed=seed) if len(clean) >= 4 else None
            hidden = a["k_evade"] if a and (a["holds_up"] or 0) >= 0.5 else None
            out["families"][fam] = {"k80": pm["scaled"][fam]["k80"], "hidden_k": hidden}
        return out
    finally:
        if had and "clock" not in lenses.DEFAULT:
            lenses.DEFAULT.insert(5, "clock")
        if not had and "clock" in lenses.DEFAULT:
            lenses.DEFAULT.remove("clock")


def ablate(chunks, prof, seed=0, reps=5, budget=160, n_cal=150, n_val=100):
    plain = _run(chunks, prof, False, seed, reps, budget, n_cal, n_val)
    clock = _run(chunks, prof, True, seed, reps, budget, n_cal, n_val)
    R = external.audit_monitor({"l": external.BuiltinLenses(split=True)}, seed=seed, n_cal=n_cal, n_val=n_val, n_base=8, reps=reps, prof=prof,
                               real_chunks=chunks, mode="scaled", chronological=True)
    D = R["detections"]["monitors"]; names = list(D)
    ok = [i for i in range(len(R["detections"]["items"])) if all(D[n][i] is not None for n in names)]
    live = [n for n in names if any(D[n][i] for i in ok)]
    rep, merged = collapse_duplicates({n: [D[n][i] for i in ok] for n in live}) if live else ({}, [])
    return {"plain": plain, "clock": clock, "monitors_named": len(names), "monitors_that_ever_fire": len(live),
            "independent_after_merging": len(set(rep.values())), "near_copies": merged, "chunks": len(chunks)}


def _k(x):
    return "not reached" if x is None else f"{x:g}"


def report_md(A, label="your data"):
    p, c = A["plain"], A["clock"]
    o = [f"# blindspot ablation on {label}", "",
         f"{A['chunks']} real chunks, calibrated on the earliest and tested on the latest. Lower is better in every strength column.", "",
         f"Clean alert rate: plain {100 * p['false_alarm']:.0f}% ({100 * p['false_alarm_ci'][0]:.0f}%-{100 * p['false_alarm_ci'][1]:.0f}%), "
         f"+clock {100 * c['false_alarm']:.0f}% ({100 * c['false_alarm_ci'][0]:.0f}%-{100 * c['false_alarm_ci'][1]:.0f}%).", "",
         "| attack shape | caught at 80% (plain) | caught at 80% (+clock) | strongest hidden (plain) | strongest hidden (+clock) |", "|---|---|---|---|---|"]
    for fam in p["families"]:
        a, b = p["families"][fam], c["families"][fam]
        o.append(f"| {fam} | {_k(a['k80'])} | {_k(b['k80'])} | {_k(a['hidden_k'])} | {_k(b['hidden_k'])} |")
    o += ["", "\"Strongest hidden\" counts only attacks the search hid on at least half of fresh tries; \"not reached\" there means it found none.", "",
          "## How many independent monitors there really are",
          f"{A['monitors_named']} detectors are named; {A['monitors_that_ever_fire']} ever fired on an attack in this data; after merging near-copies "
          f"(detections correlated at 0.95 or more) {A['independent_after_merging']} are independent."]
    if A["near_copies"]:
        o.append("Merged: " + "; ".join("{" + ", ".join(g) + "}" for g in A["near_copies"]) + ".")
    o += ["", "A naive count of monitors overstates how many second opinions you have. The gate's redundancy criterion uses the independent count.",
          "", "## Limits", "- One seed. Rerun with --seed to see how much the numbers move.",
          "- The attacks are scale-matched shapes and the search is small; a better attacker would hide more in both columns."]
    return "\n".join(o) + "\n"
