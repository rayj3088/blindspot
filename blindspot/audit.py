"""Audit engine: calibrate -> known attacks -> incident replays -> unknown-shape fuzz -> blind-spot map -> adaptive attacker."""
from __future__ import annotations

import json
import os

import numpy as np

from .attacks import LIBRARY, PRIMS, Bench, evade, fuzz, list_incidents, load_incident, run_incident
from .core import DEFAULT_PROFILE, Trace, synth_baseline
from .lenses import Stack

SCOPE = ("Measures coverage against known attack shapes and machine-generated ones. It is NOT a measure of "
         "safety against unknown attacks, and a clean report is not a clean bill of health.")


def build_stack(prof, seed=0, n_cal=300, n_val=100, far=0.05, hours=3.0, names=None):
    rng = np.random.default_rng(seed)
    cal = [synth_baseline(rng, hours, prof) for _ in range(n_cal)]
    val = [synth_baseline(rng, hours, prof) for _ in range(n_val)]
    st = Stack(names).calibrate(cal, prof, far)
    return st, st.false_alarm_rate(val, prof)


def effective_rank(R):
    """Participation ratio of the singular values of the (column-normalised) lens-response matrix.
    ~ number of genuinely independent directions the lens set covers."""
    R = np.asarray(R, float)
    if R.size == 0:
        return 0.0
    R = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)
    p = np.linalg.svd(R, compute_uv=False) ** 2
    return float(p.sum() ** 2 / (p ** 2).sum()) if p.sum() > 0 else 0.0


def greedy_order(fired, names):
    covered, order, left = set(), [], list(names)
    while left:
        gain = {n: sum(1 for i, f in enumerate(fired) if i not in covered and n in f) for n in left}
        n = max(left, key=lambda x: gain[x])
        order.append((n, gain[n]))
        covered |= {i for i, f in enumerate(fired) if n in f}
        left.remove(n)
    return order


def _count(evs):
    c = {}
    for e in evs:
        for l in e["fired"]:
            c[l] = c.get(l, 0) + 1
    return c


def run_audit(prof=DEFAULT_PROFILE, seed=0, n_cal=300, n_val=100, far=0.05, n_fuzz=150, evade_budget=60,
              reps=10, inc_reps=3, compress=1.0, real=None, incidents=True):
    st, far_val = build_stack(prof, seed, n_cal, n_val, far)
    bench = Bench(st, prof, seed=seed + 100)
    L = st.names
    R = {"scope": SCOPE, "lenses": L,
         "calibration": {"far_target": far, "far_validation": far_val, "n_cal": n_cal, "n_val": n_val, "tau": st.tau}}

    lib = {}
    for name, spec in LIBRARY.items():
        rows = [bench.run(spec, np.random.default_rng(seed * 1000 + i)) for i in range(reps)]
        ev = [r[0] for r in rows]
        ttd = [e["first"] - e["t0"] for e in ev if e["detected"]]
        lib[name] = {"rate": float(np.mean([e["detected"] for e in ev])), "ttd_median": float(np.median(ttd)) if ttd else None,
                     "lenses": _count(ev), "harm": float(np.mean([r[1] for r in rows]))}
    R["library"] = lib

    R["pace_kill_chain"] = {str(d): float(np.mean([bench.run({**LIBRARY["kill_chain"], "dilate": d}, np.random.default_rng(seed * 1000 + i))[0]["detected"]
                                                  for i in range(reps)])) for d in (0.25, 1.0, 4.0, 16.0)}

    R["incidents"] = {}
    if incidents:
        for nm in list_incidents():
            inc = load_incident(nm)
            evs = [run_incident(st, prof, inc, seed + i, compress) for i in range(inc_reps)]
            fr = [e["frac_to_alarm"] for e in evs if e["frac_to_alarm"] is not None]
            R["incidents"][nm] = {"rate": float(np.mean([e["detected"] for e in evs])), "frac_to_alarm": float(np.median(fr)) if fr else None,
                                  "lenses": _count(evs), "compress": compress, "note": inc["note"]}

    fz = fuzz(bench, n_fuzz, seed + 1)
    rows = fz["rows"]
    Rm = np.array([[r["norm"][n] for n in L] for r in rows]) if rows else np.zeros((0, len(L)))
    det = np.array([r["detected"] for r in rows], bool)
    hv = np.array([r["harm"] for r in rows])
    fired = [set(r["fired"]) for r in rows if r["detected"]]
    sv = np.linalg.svd(Rm / (np.linalg.norm(Rm, axis=0, keepdims=True) + 1e-12), compute_uv=False) if len(rows) else []
    R["fuzz"] = {"n": fz["n"], "coverage": fz["coverage"], "ci": fz["ci"],
                 "discovery": {str(k): float(det[:k].mean()) for k in (25, 50, 100, 200, 400) if k <= len(det)},
                 "per_primitive": {p: {"n": int(sum(any(x[0] == p for x in r["spec"]["prims"]) for r in rows)),
                                       "rate": float(np.mean([r["detected"] for r in rows if any(x[0] == p for x in r["spec"]["prims"])] or [0]))}
                                   for p in PRIMS},
                 "by_harm": {lab: {"n": int(m.sum()), "rate": float(det[m].mean()) if m.any() else None}
                             for lab, m in (("1-3", (hv >= 1) & (hv < 3)), ("3-10", (hv >= 3) & (hv < 10)), ("10+", hv >= 10))},
                 "escapes": int((~det).sum()), "k_eff": effective_rank(Rm), "singular": [float(x) for x in sv],
                 "single_lens_fraction": float(np.mean([len(f) == 1 for f in fired])) if fired else 0.0,
                 "irreplaceable": {n: int(sum(1 for f in fired if f == {n})) for n in L},
                 "greedy": greedy_order(fired, L)}
    R["escapes"] = [{"spec": r["spec"], "seed": r["seed"], "harm": r["harm"]} for r in rows if not r["detected"]]

    adv = {}
    for name in ("kill_chain", "recon_sweep", "slow_exfil", "rulebook_tamper"):
        con, unc = evade(bench, LIBRARY[name], seed, evade_budget, True), evade(bench, LIBRARY[name], seed, evade_budget, False)
        hm = max(h for h, _ in unc["history"])
        tr = float(np.mean([not bench.run(con["spec"], np.random.default_rng(900 + i))[0]["detected"] for i in range(8)])) if con["spec"] else None
        adv[name] = {"undetected_harm": con["undetected_harm"], "max_harm": float(hm),
                     "cost_of_evasion": float(np.clip(1 - con["undetected_harm"] / hm, 0, 1)) if hm > 0 else None, "transfer": tr}
    R["adaptive"] = adv

    if real is not None and len(real):
        t0, seg, res = float(real.t.min()), 10800.0, []
        for i in range(int((real.t.max() - t0) // seg) + 1):
            c = real.sel((real.t >= t0 + i * seg) & (real.t < t0 + (i + 1) * seg))
            if len(c.rep) >= 100:
                res.append(st.evaluate(c, prof))
        R["real"] = {"chunks": len(res), "alarmed": int(sum(e["detected"] for e in res)), "lenses": _count(res),
                     "books_available": bool(len(real.sel(real.chan == 1)))}
    return R


def _p(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def report_md(R):
    c, f = R["calibration"], R["fuzz"]
    o = ["# blindspot audit", "", f"> **Scope.** {R['scope']}", "",
         "## 1. Calibration gate", f"Target: a clean 3-hour trace alarms with probability <= {_p(c['far_target'])}. "
         f"Measured on {c['n_val']} unseen clean traces: **{_p(c['far_validation'])}**.", "",
         "## 2. Known attack shapes", "| attack | caught | median time to alarm | lenses that fired |", "|---|---|---|---|"]
    for k, v in R["library"].items():
        t = "n/a" if v["ttd_median"] is None else f"{v['ttd_median']:.0f}s"
        o.append(f"| {k} | {_p(v['rate'])} | {t} | {', '.join(f'{a}x{b}' for a, b in sorted(v['lenses'].items(), key=lambda x: -x[1])) or '-'} |")
    o += ["", "## 3. Pace sensitivity (kill chain stretched in time)", "| stretch | caught |", "|---|---|"]
    o += [f"| {k}x | {_p(v)} |" for k, v in R["pace_kill_chain"].items()]
    if R["incidents"]:
        o += ["", "## 4. Incident replays (approximate shapes from public summaries, not reconstructions)",
              "| incident | caught | first alarm (fraction of incident) | lenses |", "|---|---|---|---|"]
        for k, v in R["incidents"].items():
            o.append(f"| {k} | {_p(v['rate'])} | {_p(v['frac_to_alarm'])} | {', '.join(v['lenses']) or '-'} |")
    o += ["", "## 5. Unknown shapes (machine-generated attacks)",
          f"**{f['n']}** random attacks with real harm. Coverage: **{_p(f['coverage'])}** (95% CI {_p(f['ci'][0])}-{_p(f['ci'][1])}). "
          f"**{f['escapes']}** escaped and are saved to `escapes.jsonl` as replayable test cases.", "",
          "Coverage as the sample grows: " + ", ".join(f"{k}: {_p(v)}" for k, v in f["discovery"].items()), "",
          "", "By harm delivered (small harm = stealthier): " + ", ".join(f"{k}: {_p(v['rate'])} (n={v['n']})" for k, v in f["by_harm"].items()), "",
          "| ingredient | attacks containing it | caught |", "|---|---|---|"]
    o += [f"| {p} | {v['n']} | {_p(v['rate'])} |" for p, v in f["per_primitive"].items()]
    o += ["", "## 6. Blind-spot map",
          f"Effective independent lenses: **{f['k_eff']:.1f}** of {len(R['lenses'])}. "
          f"Attacks caught by exactly one lens: **{_p(f['single_lens_fraction'])}** (fragile coverage).", "",
          "Irreplaceable lenses (attacks only they catch): " + (", ".join(f"{k} ({v})" for k, v in f["irreplaceable"].items() if v) or "none"), "",
          "Greedy value order (lens: new attacks it adds): " + ", ".join(f"{n} +{g}" for n, g in f["greedy"] if g), "",
          "## 7. Adaptive attacker (evolves attacks to slip under every threshold)",
          "| starting attack | harm delivered undetected | max harm | cost of evasion | still evades on new noise |", "|---|---|---|---|---|"]
    o += [f"| {k} | {v['undetected_harm']:.1f} | {v['max_harm']:.1f} | {_p(v['cost_of_evasion'])} | {_p(v['transfer'])} |" for k, v in R["adaptive"].items()]
    if "real" in R:
        r = R["real"]
        o += ["", "## 8. Your real logs", f"{r['alarmed']} of {r['chunks']} three-hour chunks of clean-as-far-as-you-know activity alarmed "
              f"({', '.join(r['lenses']) or 'no lens'}). Frequent alarms mean the calibration does not transfer to your data yet. "
              + ("" if r["books_available"] else "The `books` lens is off: no independent record was supplied.")]
    o += ["", "## Limits", "- Coverage of known and generated shapes, not safety against unknown ones.",
          "- The attack library and fuzz grammar are open. Attackers can read them. Keep private hold-outs.",
          "- Baselines are synthetic unless calibrated on your logs. Synthetic attackers are not frontier models.",
          "- Lenses are language-free by design: they cannot judge whether a change is right, only whether it looks accounted for and ordinary.",
          "- An attacker who controls every record, or who camouflages perfectly at low harm, passes."]
    return "\n".join(o) + "\n"


def write_outputs(R, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write(report_md(R))
    with open(os.path.join(out, "report.json"), "w") as fh:
        json.dump(R, fh, indent=1, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    with open(os.path.join(out, "escapes.jsonl"), "w") as fh:
        for e in R["escapes"]:
            fh.write(json.dumps(e) + "\n")
