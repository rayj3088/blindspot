"""Command line: blindspot demo | audit PATH | replay INCIDENT | commit FILE | reveal FILE SALT DIGEST."""
from __future__ import annotations

import argparse

import numpy as np

from .attacks import list_incidents, load_incident, run_incident
from .audit import build_stack, report_md, run_audit, write_outputs
from .commons import commit, reveal_ok
from .core import DEFAULT_PROFILE, Trace, fit_profile, load, load_jsonl


def _run(a, prof, real=None, quick=False):
    R = run_audit(prof, seed=a.seed, n_cal=150 if quick else 300, n_val=60 if quick else 100, n_fuzz=60 if quick else a.fuzz,
                  evade_budget=24 if quick else a.evade, reps=5 if quick else 10, inc_reps=1 if quick else 3, real=real)
    write_outputs(R, a.out)
    print(report_md(R))
    print(f"[written to {a.out}/report.md, report.json, escapes.jsonl]")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="blindspot", description="Blind-spot audit for AI-agent oversight stacks.")
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("demo", help="audit the default synthetic stack (no data needed)")
    d.add_argument("--quick", action="store_true"); d.add_argument("--out", default="blindspot-out")
    d.add_argument("--seed", type=int, default=0); d.add_argument("--fuzz", type=int, default=150); d.add_argument("--evade", type=int, default=60)
    a = sp.add_parser("audit", help="audit against YOUR agent logs (Claude Code transcripts or generic jsonl); local only")
    a.add_argument("path"); a.add_argument("--observed", help="independent record (jsonl) for the books-balance lens")
    a.add_argument("--quick", action="store_true"); a.add_argument("--out", default="blindspot-out")
    a.add_argument("--seed", type=int, default=0); a.add_argument("--fuzz", type=int, default=150); a.add_argument("--evade", type=int, default=60)
    r = sp.add_parser("replay", help="replay a public incident shape against the stack")
    r.add_argument("incident", nargs="?", default="hf_2026_07"); r.add_argument("--compress", type=float, default=1.0)
    c = sp.add_parser("commit", help="fingerprint a private hold-out file; publish the digest, keep the salt")
    c.add_argument("file")
    v = sp.add_parser("reveal", help="verify a revealed hold-out against its published digest")
    v.add_argument("file"); v.add_argument("salt"); v.add_argument("digest")
    x = ap.parse_args(argv)

    if x.cmd == "demo":
        _run(x, DEFAULT_PROFILE, quick=x.quick)
    elif x.cmd == "audit":
        tr = load(x.path)
        if not len(tr):
            raise SystemExit(f"no agent events found in {x.path}")
        if x.observed:
            tr = Trace.cat(tr, load_jsonl(x.observed, chan=1))
        print(f"loaded {len(tr.rep)} events; fitting profile to your data")
        _run(x, fit_profile(tr), real=tr, quick=x.quick)
    elif x.cmd == "replay":
        st, far = build_stack(DEFAULT_PROFILE)
        print(f"available: {', '.join(list_incidents())} | clean false-alarm rate {far:.0%}")
        inc = load_incident(x.incident)
        for s in range(3):
            e = run_incident(st, DEFAULT_PROFILE, inc, s, x.compress)
            print(f"seed {s}: detected={e['detected']} first-alarm={e['frac_to_alarm']} lenses={e['fired']}")
        print("note:", inc["note"])
    elif x.cmd == "commit":
        dig, salt = commit(open(x.file, "rb").read())
        print(f"digest (publish): {dig}\nsalt   (keep secret until reveal): {salt}")
    elif x.cmd == "reveal":
        print("VALID" if reveal_ok(open(x.file, "rb").read(), x.salt, x.digest) else "MISMATCH")
