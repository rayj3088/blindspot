"""Command line: blindspot demo | audit PATH | replay INCIDENT | commit FILE | reveal FILE SALT DIGEST."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .attacks import list_incidents, load_incident, run_incident
from .audit import build_stack, report_md, run_audit, write_outputs
from .commons import commit, reveal_ok
from . import channel as chan_mod, detect_real, drift, external, gate as gatemod, gitupdates, tracelab, updates, weights as wt_mod, ablate as ab_mod
from .core import DEFAULT_PROFILE, Trace, fit_profile, load, load_jsonl


def _run(a, prof, real=None, quick=False):
    R = run_audit(prof, seed=a.seed, n_cal=300 if quick else 600, n_val=100 if quick else 200, n_fuzz=60 if quick else a.fuzz,
                  evade_budget=24 if quick else a.evade, reps=5 if quick else 10, inc_reps=10 if quick else 20, real=real)
    write_outputs(R, a.out)
    print(report_md(R))
    print(f"[written to {a.out}/report.md, report.json, escapes.jsonl]")


def _monitor_args(m, gate=False):
    m.add_argument("--cmd", dest="mon_cmd", action="append", default=[], help='command that reads JSONL events on stdin and prints a result, e.g. "python my_monitor.py"; repeatable')
    m.add_argument("--py", dest="mon_py", action="append", default=[], help="Python function as module:function or file.py:function; repeatable")
    m.add_argument("--with-builtin", action="store_true", help="also include the built-in detectors as one combined monitor")
    m.add_argument("--builtin-split", action="store_true", help="include the built-in detectors as separate monitors" + (" (the default when no monitor is given)" if gate else " (lets redundancy among them be measured)"))
    m.add_argument("--agentnorm", action="store_true", help="also audit agentnorm (install it first; see its README)")
    m.add_argument("--independent", action="store_true", help="also pass the independent record (chan=1) to your monitor")
    m.add_argument("--quick", action="store_true"); m.add_argument("--out", default="blindspot-out")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--real", help="your real agent logs (Claude Code folder or jsonl): use real 3-hour chunks as normal activity")
    m.add_argument("--tracelab", help="TraceLab jsonl(.gz): use the busiest model's real chunks as normal activity")
    m.add_argument("--tracelab-model", help="which model in the TraceLab file (default: the one with the most tool calls)")
    m.add_argument("--plugin", action="append", default=[], help="Python file with register(api) that adds attack shapes, gate criteria, log readers or monitors; repeatable")
    m.add_argument("--format", help="name of a log reader registered by a plugin, used with --real")
    m.add_argument("--chunk-minutes", type=float, default=180.0, help="length of each real chunk of normal activity; shorter chunks need less log history (40 chunks are needed)")
    if not gate:
        m.add_argument("--chronological", action="store_true", help="calibrate on the earliest chunks and test on the latest (always on in the gate)")


def _prepare_monitors(x, default_split=False, chronological=False):
    """-> monitors dict, real chunks (or None), profile, stripped flag"""
    from . import plugins
    for pl in x.plugin:
        plugins.load_plugin(pl)
    mons = dict(plugins.CUSTOM_MONITORS)
    many = len(x.mon_cmd) + len(x.mon_py) > 1
    for s_ in x.mon_cmd:
        mons[s_ if many else "your-monitor"] = external.CmdMonitor(s_, x.independent)
    for s_ in x.mon_py:
        mons[s_ if many else "your-monitor"] = external.load_py_monitor(s_, x.independent)
    if x.with_builtin:
        mons["built-in"] = external.BuiltinLenses(split=False)
    if x.builtin_split or (default_split and not mons and not x.agentnorm):
        mons["lenses"] = external.BuiltinLenses(split=True)
    if not mons and not x.agentnorm:
        raise SystemExit('give at least one monitor: --cmd "python my_monitor.py", --py my_file.py:my_function, --agentnorm, --with-builtin or --builtin-split')
    real_chunks, prof, stripped = None, DEFAULT_PROFILE, False
    if x.real or x.tracelab:
        if x.tracelab:
            g = tracelab.load_tracelab(x.tracelab)
            label = x.tracelab_model if x.tracelab_model in g else max(g, key=lambda k: len(g[k]))
            rtr = g[label]; stripped = True
            print(f"normal activity: TraceLab {label} ({len(rtr)} tool calls)", file=sys.stderr)
        else:
            rtr = load(x.real, x.format)
            if len(rtr):
                span_h = (rtr.t.max() - rtr.t.min()) / 3600
                rtr = drift.compress_gaps(rtr)
                print(f"your logs: {len(rtr)} tool calls over {span_h:.0f} hours of calendar time, {(rtr.t.max() - rtr.t.min()) / 3600:.1f} hours of active time "
                      "(idle gaps over 10 minutes shortened)", file=sys.stderr)
        if not len(rtr):
            raise SystemExit("no agent events found in the real logs")
        prof = fit_profile(rtr)
        seg = x.chunk_minutes * 60.0
        real_chunks = drift._chunks(rtr, seg, 100 if seg >= 10800 else max(20, int(100 * seg / 10800)))
        print(f"real chunks of {x.chunk_minutes:g} minutes available: {len(real_chunks)} (40 needed)", file=sys.stderr)
    if x.agentnorm:
        from .agentnorm_adapter import AgentnormMonitor
        fit_chunks = None
        if real_chunks and len(real_chunks) >= 40:
            k = int(0.3 * len(real_chunks))
            if chronological:
                fit_chunks, real_chunks = list(real_chunks[:k]), list(real_chunks[k:])
            else:
                order = np.random.default_rng(x.seed + 5).permutation(len(real_chunks))
                fit_chunks = [real_chunks[i] for i in order[:k]]                       # agentnorm learns from these
                real_chunks = [real_chunks[i] for i in order[k:]]                      # and is tested on different ones
        mons["agentnorm"] = AgentnormMonitor(seed=x.seed, fit_chunks=fit_chunks)
        print(f"agentnorm fitted on {mons['agentnorm'].fitted_on} activity", file=sys.stderr)
    return mons, real_chunks, prof, stripped


def main(argv=None):
    ap = argparse.ArgumentParser(prog="blindspot", description="Blind-spot audit for AI-agent oversight stacks.")
    sp = ap.add_subparsers(dest="subcmd", required=True)
    d = sp.add_parser("demo", help="audit the default synthetic stack (no data needed)")
    d.add_argument("--quick", action="store_true"); d.add_argument("--out", default="blindspot-out")
    d.add_argument("--seed", type=int, default=0); d.add_argument("--fuzz", type=int, default=150); d.add_argument("--evade", type=int, default=60)
    a = sp.add_parser("audit", help="audit against YOUR agent logs (Claude Code transcripts or generic jsonl); local only")
    a.add_argument("path"); a.add_argument("--observed", help="independent record (jsonl) for the books-balance lens")
    a.add_argument("--quick", action="store_true"); a.add_argument("--out", default="blindspot-out")
    a.add_argument("--seed", type=int, default=0); a.add_argument("--fuzz", type=int, default=150); a.add_argument("--evade", type=int, default=60)
    m = sp.add_parser("audit-monitor", help="attack YOUR monitor (a command or Python function) and report what it misses")
    _monitor_args(m)
    m.add_argument("--min-coverage", type=float, help="exit 2 if any monitor catches less than this share of machine-generated attacks (0-1)")
    m.add_argument("--max-false-alarm", type=float, help="exit 2 if any monitor false-alarms on more than this share of clean traces (0-1)")
    gt = sp.add_parser("gate", help="oversight-sufficiency gate: are the necessary conditions for trusting oversight of the next change met? exit 0 GO, 1 NO-GO, 3 insufficient evidence")
    _monitor_args(gt, gate=True)
    gt.add_argument("--step", default="next capability step", help="what this decision is about (recorded in the ledger)")
    gt.add_argument("--policy", help="JSON file overriding the default thresholds (see --print-default-policy)")
    gt.add_argument("--save-baseline", metavar="FILE", help="write a regression policy from this run: later runs with --policy FILE fail only if oversight got worse")
    gt.add_argument("--print-default-policy", action="store_true", help="print the default (placeholder) thresholds and exit")
    gt.add_argument("--adversary-budget", type=int, default=160, help="monitor queries the evasion search may use per attack shape")
    gt.add_argument("--updates", help="JSONL of change / snapshot / review records")
    gt.add_argument("--updates-git", help="git repo to build change records from")
    gt.add_argument("--sla-hours", type=float, default=24.0)
    gt.add_argument("--ledger", help="append-only decision ledger (JSONL)"); gt.add_argument("--witness", help="ledger head you stored somewhere the agent cannot write")
    gt.add_argument("--artifact", action="append", default=[], help="file or folder this decision covers (weights, configs, prompts); its sha256 is recorded in the ledger; repeatable")
    gt.add_argument("--lineage-key", help="secret key file (see 'weights sign') to check the artifact's lineage instead of requiring an exact hash match")
    gt.add_argument("--lineage-signature", help="signature file from 'weights sign', used with --lineage-key")
    gt.add_argument("--record", action="store_true", help="append this decision to the ledger and print the new head to store elsewhere")
    ch = sp.add_parser("channel", help="what do agents say to each other? decode a private language by what it makes the receiver do, and find agents coordinating beyond shared context")
    ch.add_argument("messages", help='JSONL: time (t/time/timestamp; seconds, ms or ISO), from/sender, to/receiver (missing = broadcast), text/content/message')
    ch.add_argument("--actions", help="what the agents did: JSONL with a time, an agent and a tool/name/kind (tool names are used as labels), or a Claude Code folder")
    ch.add_argument("--reference", help="a text sample of your agents' ordinary language, to measure how far the channel has moved from it")
    ch.add_argument("--windows", default="30,300,1800", help="delays (seconds) after a message in which its effect is looked for")
    ch.add_argument("--min-count", type=int, default=8); ch.add_argument("--perms", type=int, default=500, help="lower this on a slow machine")
    ch.add_argument("--max-messages", type=int, default=100000, help="use a random sample of this many messages if there are more")
    ch.add_argument("--out", default="blindspot-out"); ch.add_argument("--seed", type=int, default=0)
    ws = sp.add_parser("weights", help="prove a checkpoint's lineage even after it has been fine-tuned (survives ordinary updates; a hash match does not)")
    wsub = ws.add_subparsers(dest="wcmd", required=True)
    sgn = wsub.add_parser("sign", help="embed a signature in a checkpoint's parameters, keyed to a secret you keep")
    sgn.add_argument("path"); sgn.add_argument("out"); sgn.add_argument("--key-file", required=True, help="file holding the secret key bytes; never ship this with the checkpoint")
    sgn.add_argument("--bits", type=int, default=128); sgn.add_argument("--budget", type=float, default=1e-3, help="perturbation as a fraction of the checkpoint's own L2 norm")
    sgn.add_argument("--signature-out", required=True, help="where to save the signature bytes; keep this, it is what you check against later")
    vfy = wsub.add_parser("verify", help="check whether a checkpoint's parameters still carry a signature")
    vfy.add_argument("path"); vfy.add_argument("--key-file", required=True); vfy.add_argument("--signature", required=True)
    vfy.add_argument("--alpha", type=float, default=1e-9); vfy.add_argument("--out", default="blindspot-out")
    ab = sp.add_parser("ablate", help="do the added detectors beat plain baselines on YOUR real data? same attacks and evasion search, with and without them")
    ab.add_argument("--real"); ab.add_argument("--tracelab"); ab.add_argument("--tracelab-model"); ab.add_argument("--format")
    ab.add_argument("--chunk-minutes", type=float, default=180.0); ab.add_argument("--quick", action="store_true")
    ab.add_argument("--seed", type=int, default=0); ab.add_argument("--out", default="blindspot-out")
    ab.add_argument("--budgets", help="also show how the strongest hidden attack changes with query budget, e.g. 40,160,640")
    rp = sp.add_parser("reproduce", help="rerun the README's real-data table on TraceLab: ablation across models and seeds, one report")
    rp.add_argument("tracelab"); rp.add_argument("--models", help="comma-separated (default: the four busiest)"); rp.add_argument("--seeds", default="0,1")
    rp.add_argument("--quick", action="store_true"); rp.add_argument("--out", default="blindspot-out")
    gv = sp.add_parser("gate-verify", help="verify a decision ledger's hash chain, optionally against a witnessed head")
    gv.add_argument("ledger"); gv.add_argument("--witness")
    g = sp.add_parser("drift-sim", help="synthetic study: how big can a self-update be before un-recalibrated monitors go stale?")
    g.add_argument("--quick", action="store_true"); g.add_argument("--out", default="blindspot-out"); g.add_argument("--seed", type=int, default=0)
    g.add_argument("--reps", type=int, default=2); g.add_argument("--fail-if-stale", action="store_true", help="exit 2 if any tested update left the monitors stale")
    dd = sp.add_parser("drift", help="measure monitor staleness across REAL versions of your agent (one log file/folder per version, oldest first)")
    dd.add_argument("paths", nargs="+"); dd.add_argument("--quick", action="store_true"); dd.add_argument("--out", default="blindspot-out")
    dd.add_argument("--seed", type=int, default=0); dd.add_argument("--fail-if-stale", action="store_true")
    up = sp.add_parser("updates", help="check that every self-change is declared, matches an independent record, and is reviewed in time")
    up.add_argument("path", nargs="?", help="JSONL of change / snapshot / review records (see blindspot/updates.py)")
    up.add_argument("--simulate", type=float, metavar="DAYS", help="use a synthetic history of this many days instead of a file")
    up.add_argument("--changes-per-day", type=float, default=6.0); up.add_argument("--reviews-per-day", type=float, default=4.0)
    up.add_argument("--sla-hours", type=float, default=24.0); up.add_argument("--tol", type=float, default=60.0, help="seconds of slack between a change and the snapshot that sees it")
    up.add_argument("--out", default="blindspot-out")
    up.add_argument("--fail", action="store_true", help="exit 1 if any accounting problem is found")
    up.add_argument("--fail-if-behind", action="store_true", help="exit 1 if the review backlog is growing")
    tl = sp.add_parser("tracelab", help="real-data map of how monitor calibration transfers across models (or users) in the public TraceLab dataset, and whether it drifts over time")
    tl.add_argument("file", help="TraceLab JSONL (.jsonl or .jsonl.gz)"); tl.add_argument("--by", default="model", help="field that defines a group: model, provider or user")
    tl.add_argument("--min-events", type=int, default=3000); tl.add_argument("--top", type=int, default=5, help="compare at most this many groups")
    tl.add_argument("--max-rows", type=int); tl.add_argument("--quick", action="store_true"); tl.add_argument("--out", default="blindspot-out"); tl.add_argument("--seed", type=int, default=0)
    cv = sp.add_parser("curves", help="how strong must an attack be before your monitors notice it? Scale-matched attacks on your REAL logs (works when paths/sizes are stripped)")
    cv.add_argument("path", help="your real agent logs (Claude Code folder or jsonl)"); cv.add_argument("--quick", action="store_true")
    cv.add_argument("--out", default="blindspot-out"); cv.add_argument("--seed", type=int, default=0)
    ug = sp.add_parser("updates-from-git", help="self-update accounting from a git repo: agent edits to CLAUDE.md, .claude/, hooks, CI vs the working tree")
    ug.add_argument("repo"); ug.add_argument("--paths", nargs="+", help="files/folders to watch (default: the usual agent control files)")
    ug.add_argument("--agent", default=gitupdates.DEFAULT_AGENT, help="regex marking an agent-authored or agent-assisted commit (author or Co-Authored-By)")
    ug.add_argument("--review-trailer", default="Reviewed-by", help="commit trailer that marks an agent commit as reviewed")
    ug.add_argument("--since"); ug.add_argument("--sla-hours", type=float, default=24.0); ug.add_argument("--out", default="blindspot-out")
    ug.add_argument("--records-out", help="also write the change/snapshot/review records here (JSONL)")
    ug.add_argument("--fail", action="store_true"); ug.add_argument("--fail-if-behind", action="store_true")
    sn = sp.add_parser("snapshot", help="print independent snapshot records (sha256) for files or folders; run it where the agent cannot write")
    sn.add_argument("paths", nargs="+")
    r = sp.add_parser("replay", help="replay a public incident shape against the stack")
    r.add_argument("incident", nargs="?", default="hf_2026_07"); r.add_argument("--compress", type=float, default=1.0)
    c = sp.add_parser("commit", help="fingerprint a private hold-out file; publish the digest, keep the salt")
    c.add_argument("file")
    v = sp.add_parser("reveal", help="verify a revealed hold-out against its published digest")
    v.add_argument("file"); v.add_argument("salt"); v.add_argument("digest")
    x = ap.parse_args(argv)

    if x.subcmd == "demo":
        _run(x, DEFAULT_PROFILE, quick=x.quick)
    elif x.subcmd == "audit":
        tr = load(x.path)
        if not len(tr):
            raise SystemExit(f"no agent events found in {x.path}")
        if x.observed:
            tr = Trace.cat(tr, load_jsonl(x.observed, chan=1))
        print(f"loaded {len(tr.rep)} events; fitting profile to your data")
        _run(x, fit_profile(tr), real=tr, quick=x.quick)
    elif x.subcmd == "audit-monitor":
        mons, real_chunks, prof, stripped = _prepare_monitors(x, chronological=x.chronological)
        q = x.quick
        if stripped and not (real_chunks and len(real_chunks) >= 40):
            raise SystemExit("this data has too few real 3-hour chunks (40 needed)")
        R = external.audit_monitor(mons, seed=x.seed, n_cal=60 if q else 150, n_val=60 if q else 100, n_base=6 if q else 8,
                                   reps=3 if q else 5, n_fuzz=40 if q else 100, log=lambda s: print(s, file=sys.stderr),
                                   prof=prof, real_chunks=real_chunks, mode="scaled" if stripped else "library", chronological=x.chronological)
        external.write_outputs(R, x.out)
        print(external.report_md(R))
        print(f"[written to {x.out}/report.md, report.json, escapes.jsonl]")
        bad = []
        for n, pm in R["per_monitor"].items():
            if x.max_false_alarm is not None and pm["false_alarm"]["rate"] > x.max_false_alarm:
                bad.append(f"{n}: false-alarm rate {pm['false_alarm']['rate']:.0%} > {x.max_false_alarm:.0%}")
            if x.min_coverage is not None and (not pm["usable"] or (pm["fuzz"]["coverage"] or 0.0) < x.min_coverage):
                bad.append(f"{n}: coverage {(pm['fuzz']['coverage'] or 0.0):.0%} < {x.min_coverage:.0%}")
        if bad:
            print("GATE FAILED: " + "; ".join(bad), file=sys.stderr)
            raise SystemExit(2)
    elif x.subcmd == "channel":
        import os as _os
        msgs = chan_mod.load_messages(x.messages)
        if not msgs:
            raise SystemExit("no messages found: each line needs a time and text (see --help)")
        ref = open(x.reference, errors="ignore").read() if x.reference else None
        P = chan_mod.profile(msgs, ref, x.seed)
        H = chan_mod.hidden_characters(msgs)
        G = C = I = None
        if x.actions:
            act = chan_mod.load_actions(x.actions)
            if not len(act):
                tr = load(x.actions)
                act = chan_mod.actions_from_trace(tr) if len(tr) else act
            if not len(act):
                raise SystemExit(f"no actions found in {x.actions}: each line needs a time, an agent and a tool or kind")
            wins = tuple(float(w) for w in x.windows.split(","))
            say = lambda s_: print(s_, file=sys.stderr)          # noqa: E731
            say(f"decoding by consequence ({len(msgs)} messages, {len(act)} actions, delays {x.windows} s)")
            G = chan_mod.ground(msgs, act, wins, x.min_count, x.perms, seed=x.seed, max_messages=x.max_messages)
            say("testing whether the whole message predicts the next action")
            I = chan_mod.informativeness(msgs, act, window=wins[min(1, len(wins) - 1)], perms=max(30, x.perms // 2), seed=x.seed, max_messages=min(x.max_messages, 60000))
            say("testing coordination between agents")
            C = chan_mod.coordination(act, seed=x.seed, perms=x.perms)
        md = chan_mod.report_md(P, G, C, I, H)
        _os.makedirs(x.out, exist_ok=True)
        with open(_os.path.join(x.out, "channel.md"), "w") as fh:
            fh.write(md)
        with open(_os.path.join(x.out, "channel.json"), "w") as fh:
            json.dump({"profile": P, "hidden_characters": H, "grounding": G, "informativeness": I, "coordination": C}, fh, indent=1, default=str)
        print(md)
        print(f"[written to {x.out}/channel.md, channel.json]")
    elif x.subcmd == "weights":
        import os as _os
        if x.wcmd == "sign":
            key = open(x.key_file, "rb").read()
            sig = wt_mod.gen_signature(x.bits // 8)
            vec = wt_mod.load_vector(x.path)
            print(f"loaded {len(vec)} parameters from {x.path}", file=sys.stderr)
            signed = wt_mod.embed_signature(vec, key, sig, x.budget)
            wt_mod.save_vector(x.out, signed, x.path)
            open(x.signature_out, "wb").write(sig)
            rel = float(__import__("numpy").linalg.norm(signed - vec) / (__import__("numpy").linalg.norm(vec) or 1))
            print(f"signed {x.out} with {x.bits} bits; relative L2 change {rel:.2e}. Signature saved to {x.signature_out}: keep it and the key.")
        else:
            key = open(x.key_file, "rb").read(); sig = open(x.signature, "rb").read()
            vec = wt_mod.load_vector(x.path)
            V = wt_mod.verify(vec, key, sig, x.alpha)
            md = wt_mod.report_md(V, x.path)
            _os.makedirs(x.out, exist_ok=True)
            with open(_os.path.join(x.out, "lineage.md"), "w") as fh:
                fh.write(md)
            print(md)
            raise SystemExit(0 if V["lineage"] == "confirmed" else (2 if V["lineage"] == "no evidence" else 1))
    elif x.subcmd == "ablate":
        import os as _os
        if x.tracelab:
            g = tracelab.load_tracelab(x.tracelab)
            label = x.tracelab_model if x.tracelab_model in g else max(g, key=lambda k: len(g[k]))
            rtr = g[label]; label = f"TraceLab {label}"
        elif x.real:
            rtr = load(x.real, x.format); label = x.real
            if not len(rtr):
                raise SystemExit(f"no tool calls found in {x.real}")
            span_h = (rtr.t.max() - rtr.t.min()) / 3600
            rtr = drift.compress_gaps(rtr)
            print(f"your logs: {len(rtr)} tool calls over {span_h:.0f} hours of calendar time, {(rtr.t.max() - rtr.t.min()) / 3600:.1f} hours of active time", file=sys.stderr)
        else:
            raise SystemExit("give --real PATH or --tracelab FILE")
        seg = x.chunk_minutes * 60.0
        ch = drift._chunks(rtr, seg, 100 if seg >= 10800 else max(20, int(100 * seg / 10800)))
        if len(ch) < 40:
            need = 40 * max(20, int(100 * seg / 10800)) if seg < 10800 else 4000
            raise SystemExit(f"only {len(ch)} real chunks of {x.chunk_minutes:g} minutes with enough activity: 40 are needed. "
                             f"You have {len(rtr)} tool calls; about {need} are needed at this chunk length. Try --chunk-minutes 10, or use the tool more and rerun later.")
        q = x.quick
        A = ab_mod.ablate(ch, fit_profile(rtr), seed=x.seed, reps=3 if q else 5, budget=60 if q else 160, n_cal=60 if q else 150, n_val=60 if q else 100,
                          budgets=tuple(int(b) for b in x.budgets.split(",")) if x.budgets else None)
        md = ab_mod.report_md(A, label)
        _os.makedirs(x.out, exist_ok=True)
        with open(_os.path.join(x.out, "ablation.md"), "w") as fh:
            fh.write(md)
        print(md)
    elif x.subcmd == "reproduce":
        import os as _os
        g = tracelab.load_tracelab(x.tracelab)
        models = x.models.split(",") if x.models else sorted(g, key=lambda k: -len(g[k]))[:4]
        seeds = [int(s) for s in x.seeds.split(",")]
        q = x.quick; rows = []
        for mdl in models:
            if mdl not in g:
                print(f"skipping {mdl}: not in the file", file=sys.stderr); continue
            ch = drift._chunks(g[mdl]); prof = fit_profile(g[mdl])
            if len(ch) < 40:
                print(f"skipping {mdl}: {len(ch)} chunks", file=sys.stderr); continue
            for sd in seeds:
                print(f"[{mdl} seed {sd}]", file=sys.stderr)
                A = ab_mod.ablate(ch, prof, seed=sd, reps=3 if q else 5, budget=60 if q else 160, n_cal=60 if q else 150, n_val=60 if q else 100)
                rows.append((mdl, sd, A))
        k = lambda v: "none found" if v is None else f"{v:.1f}"          # noqa: E731
        md = ["# blindspot: real-data reproduction on TraceLab", "", "Strongest `loop` attack the evasion search hid, with and without `clock`, and the clean alert rate. "
              "Strength is in multiples of the data's own 99th percentile.", "",
              "| model | seed | without clock | with clock | clean alert rate | clock threshold vs clean median |", "|---|---|---|---|---|---|"]
        for mdl, sd, A in rows:
            d = A["clock_diagnostic"]
            md.append(f"| {mdl} | {sd} | {k(A['plain']['families']['loop']['hidden_k'])} | {k(A['clock']['families']['loop']['hidden_k'])} | "
                      f"{100 * A['plain']['false_alarm']:.0f}% → {100 * A['clock']['false_alarm']:.0f}% | {d['threshold']:.3f} vs {d['clean_median']:.3f} |")
        md += ["", "Full per-run tables follow.", ""] + [ab_mod.report_md(A, f"TraceLab {mdl}, seed {sd}") for mdl, sd, A in rows]
        md = "\n".join(md) + "\n"
        _os.makedirs(x.out, exist_ok=True)
        with open(_os.path.join(x.out, "reproduce.md"), "w") as fh:
            fh.write(md)
        print(md)
    elif x.subcmd == "gate-verify":
        info = gatemod.ledger_verify(x.ledger, x.witness)
        if not info.get("exists"):
            raise SystemExit(f"no ledger at {x.ledger}")
        print(f"{'VALID' if info['ok'] else 'BROKEN'}: {info['n']} decision(s), head {info['head']}" + (f" ({info['problem']})" if info["problem"] else ""))
        if x.witness:
            print("witness head found in the chain" if info["witness_found"] else "witness head NOT found in the chain")
        stale = []
        if info["ok"] and info.get("last") and info["last"].get("artifacts"):
            for a_, h_ in info["last"]["artifacts"].items():
                now = updates.hash_path(a_) if __import__("os").path.exists(a_) else None
                if now != h_:
                    stale.append(a_)
            print(f"last decision: {info['last']['decision']} for '{info['last']['step']}'; " +
                  (f"these files changed or vanished since it was made, so the decision does not cover them: {', '.join(stale)}" if stale else "every recorded file still matches"))
        raise SystemExit(0 if info["ok"] and (info["witness_found"] is not False) and not stale else 1)
    elif x.subcmd == "gate":
        if x.print_default_policy:
            print(json.dumps(gatemod.DEFAULT_POLICY, indent=2))
            return
        if not (x.real or x.tracelab):
            raise SystemExit("the gate needs real activity to calibrate and test on: --real PATH or --tracelab FILE")
        policy = json.load(open(x.policy)) if x.policy else None
        mons, real_chunks, prof, _ = _prepare_monitors(x, default_split=True, chronological=True)
        if not real_chunks or len(real_chunks) < 40:
            n_have = len(real_chunks or [])
            raise SystemExit(f"only {n_have} real chunks of {x.chunk_minutes:g} minutes: 40 are needed. Try a shorter chunk, for example --chunk-minutes {max(5, int(x.chunk_minutes * max(n_have, 1) / 45))}")
        q = x.quick
        say = lambda s_: print(s_, file=sys.stderr)          # noqa: E731
        R, st = external.audit_monitor(mons, seed=x.seed, n_cal=60 if q else 150, n_val=60 if q else 100, n_base=6 if q else 8, reps=3 if q else 5,
                                       log=say, prof=prof, real_chunks=real_chunks, mode="scaled", chronological=True, return_state=True)
        alarms = st["alarm"]
        ens = lambda tr: any(a(tr) for a in alarms.values())          # noqa: E731
        clean = [b for b in st["val"] if not ens(b)][:20]
        adversary = {}
        if len(clean) >= 4:
            for fam in detect_real.FAMILIES:
                say(f"[adversary] searching for a {fam} attack the monitors miss")
                adversary[fam] = detect_real.evade_blackbox(ens, clean, st["S"], fam, budget=60 if q else x.adversary_budget, seed=x.seed)
        else:
            adversary = None
        upd = None
        if x.updates or x.updates_git:
            if x.updates_git:
                recs, _ = gitupdates.records_from_git(x.updates_git)
            else:
                recs = updates.load_records(x.updates)
            if recs:
                upd = updates.reconcile(recs, x.sla_hours, 60.0)
        lineage = None
        if x.lineage_key and x.lineage_signature and x.artifact:
            key = open(x.lineage_key, "rb").read(); sig = open(x.lineage_signature, "rb").read()
            lineage = wt_mod.verify(wt_mod.load_vector(x.artifact[0]), key, sig)
        G = gatemod.evaluate_gate(R, adversary, upd, policy, x.ledger, x.witness, x.step, lineage)
        import os as _os
        _os.makedirs(x.out, exist_ok=True)
        G["chunk_minutes"] = x.chunk_minutes
        if x.save_baseline:
            with open(x.save_baseline, "w") as fh:
                json.dump(gatemod.baseline_policy(G), fh, indent=2)
            print(f"regression baseline written to {x.save_baseline}; use it with --policy {x.save_baseline}", file=sys.stderr)
        md = gatemod.gate_md(G, adversary)
        with open(_os.path.join(x.out, "gate.md"), "w") as fh:
            fh.write(md)
        with open(_os.path.join(x.out, "gate.json"), "w") as fh:
            json.dump({"gate": G, "adversary": adversary, "audit": R}, fh, indent=1, default=str)
        print(md)
        print(f"[written to {x.out}/gate.md, gate.json]")
        if x.record:
            if not x.ledger:
                raise SystemExit("--record needs --ledger PATH")
            import hashlib as _h
            head = gatemod.ledger_append(x.ledger, {"step": x.step, "decision": G["decision"], "time": G["time"],
                                                    "criteria": {c["id"]: c["status"] for c in G["criteria"]},
                                                    "report_sha256": _h.sha256(md.encode()).hexdigest(),
                                                    "policy_sha256": _h.sha256(json.dumps(G["policy"], sort_keys=True).encode()).hexdigest(),
                                                    "artifacts": {a_: updates.hash_path(a_) for a_ in x.artifact}})
            print(f"recorded. ledger head: {head}\nStore this head somewhere the agent cannot write, and pass it as --witness next time.")
        raise SystemExit(0 if G["decision"].startswith("GO") else (1 if G["decision"] == "NO-GO" else 3))
    elif x.subcmd == "drift-sim":
        q = x.quick
        R = drift.staleness(levels=(0.0, 0.2, 0.6) if q else (0.0, 0.1, 0.2, 0.4, 0.8), reps=1 if q else x.reps, seed=x.seed,
                            n_cal=100 if q else 150, n_val=40 if q else 60, lib_reps=2 if q else 3, n_fuzz=15 if q else 25,
                            log=lambda s: print(s, file=sys.stderr))
        drift.write_outputs(R, x.out)
        print(drift.report_md(R))
        print(f"[written to {x.out}/staleness.md, staleness.json]")
        if x.fail_if_stale and R["stale_count"]:
            print(f"GATE FAILED: {R['stale_count']} of {R['n_versions']} tested updates left the old monitors stale", file=sys.stderr)
            raise SystemExit(2)
    elif x.subcmd == "drift":
        if len(x.paths) < 2:
            raise SystemExit("give at least two paths: the version whose monitors you calibrated on, then later versions")
        q = x.quick
        R = drift.staleness_from_logs(x.paths, seed=x.seed, n_cal=100 if q else 150, n_val=40 if q else 60,
                                      lib_reps=2 if q else 3, n_fuzz=15 if q else 25, log=lambda s: print(s, file=sys.stderr))
        drift.write_outputs(R, x.out, "blindspot: monitor staleness on your agent versions")
        print(drift.report_md(R, "blindspot: monitor staleness on your agent versions"))
        print(f"[written to {x.out}/staleness.md, staleness.json]")
        if x.fail_if_stale and R["stale_count"]:
            print(f"GATE FAILED: {R['stale_count']} of {R['n_versions']} versions left the old monitors stale", file=sys.stderr)
            raise SystemExit(2)
    elif x.subcmd == "updates":
        if x.simulate:
            recs = updates.simulate(x.simulate, x.changes_per_day, x.reviews_per_day, unexplained=1, phantom=1)
            print(f"(synthetic history: {x.simulate:g} days, {x.changes_per_day:g} changes/day, review capacity {x.reviews_per_day:g}/day, one undeclared change and one phantom change planted)\n", file=sys.stderr)
        elif x.path:
            recs = updates.load_records(x.path)
        else:
            raise SystemExit("give a records file, or --simulate DAYS")
        if not recs:
            raise SystemExit("no change / snapshot / review records found")
        R = updates.reconcile(recs, x.sla_hours, x.tol)
        sealed = updates.seal([r for r in recs if r["type"] == "change"])
        updates.write_outputs(R, x.out, sealed)
        print(updates.report_md(R, sealed))
        print(f"[written to {x.out}/updates.md, updates.json]")
        acct = [f for f in R["findings"] if f["kind"] != "UNREVIEWED_PAST_SLA"]
        if (x.fail and acct) or (x.fail_if_behind and R["pacing"].get("falling_behind")):
            print("GATE FAILED: " + (f"{len(acct)} accounting problems" if x.fail and acct else "review backlog is growing"), file=sys.stderr)
            raise SystemExit(1)
    elif x.subcmd == "tracelab":
        groups = tracelab.load_tracelab(x.file, x.by, max_rows=x.max_rows)
        groups = {k: v for k, v in groups.items() if len(v) >= x.min_events}
        order = sorted(groups, key=lambda k: -len(groups[k]))[: x.top]
        groups = {k: groups[k] for k in order}
        if len(groups) < 2:
            raise SystemExit(f"need at least two {x.by} groups with {x.min_events}+ tool calls; found {len(groups)}. Lower --min-events or change --by.")
        print("groups (tool calls): " + ", ".join(f"{k} ({len(groups[k])})" for k in groups), file=sys.stderr)
        q = x.quick
        say = lambda s: print(s, file=sys.stderr)          # noqa: E731
        try:
            T = drift.transfer_matrix(groups, seed=x.seed, max_cal=120 if q else None, max_val=60 if q else None, log=say)
        except ValueError as e:
            raise SystemExit(str(e))
        D = drift.time_drift(groups, seed=x.seed, max_cal=120 if q else None, log=say)
        title = f"blindspot: how monitor calibration transfers across real {x.by}s (TraceLab)"
        note = ("The public TraceLab data strips file paths and tool inputs and records sizes in characters, so attack detection cannot be judged from it, "
                "and the target-based detectors (sweep, ctrl) have nothing to see.")
        drift.write_transfer(T, D, x.out, x.by, title, note)
        curves = {}
        for lab in T["labels"]:
            ch = drift._chunks(groups[lab])
            curves[lab] = detect_real.detection_curves(ch, fit_profile(groups[lab]), reps=10 if q else 20, ks=(0.5, 1, 2, 4, 8) if q else (0.25, 0.5, 1, 2, 4, 8),
                                                       seed=x.seed, max_cal=120 if q else None, log=say)
        md = drift.transfer_report_md(T, D, x.by, title, note)
        md += "\n## How strong must an attack be before the monitors notice it?\n" + \
              "Attacks are built only from fields this data has, copy its own target and tool conventions, and are sized as multiples k of the data's own statistics. " + \
              "They measure sensitivity. They are not recorded intrusions.\n\n" + "\n".join(detect_real.curves_md(v, k) for k, v in curves.items())
        import os as _os
        _os.makedirs(x.out, exist_ok=True)
        with open(_os.path.join(x.out, "transfer.md"), "w") as fh:
            fh.write(md)
        print(md)
        print(f"[written to {x.out}/transfer.md, transfer.json]")
    elif x.subcmd == "curves":
        rtr = load(x.path)
        ch = drift._chunks(rtr)
        if len(ch) < 40:
            raise SystemExit(f"only {len(ch)} real 3-hour chunks found; 40 are needed to calibrate on real activity (about 120 hours of logs)")
        q = x.quick
        C = detect_real.detection_curves(ch, fit_profile(rtr), reps=10 if q else 20, ks=(0.5, 1, 2, 4, 8) if q else (0.25, 0.5, 1, 2, 4, 8), seed=x.seed,
                                         log=lambda s: print(s, file=sys.stderr))
        md = "# blindspot: how strong must an attack be before your monitors notice it?\n\n" + detect_real.curves_md(C, "your logs")
        import os as _os
        _os.makedirs(x.out, exist_ok=True)
        with open(_os.path.join(x.out, "curves.md"), "w") as fh:
            fh.write(md)
        print(md)
    elif x.subcmd == "updates-from-git":
        recs, info = gitupdates.records_from_git(x.repo, x.paths, x.agent, x.review_trailer, x.since)
        if not any(r["type"] == "change" for r in recs):
            raise SystemExit("no commits touched the watched paths; use --paths to point at your agent's files")
        if x.records_out:
            with open(x.records_out, "w") as fh:
                fh.write("\n".join(json.dumps(r) for r in recs) + "\n")
        R = updates.reconcile(recs, x.sla_hours, 60.0)
        sealed = updates.seal([r for r in recs if r["type"] == "change"])
        updates.write_outputs(R, x.out, sealed)
        print(f"(git history of {x.repo}: {sum(r['type'] == 'change' for r in recs)} changes to watched files, {info['agent_commits']} agent-assisted commits, "
              f"watching {', '.join(sorted({r['artifact'] for r in recs}))})\n", file=sys.stderr)
        print(updates.report_md(R, sealed))
        print(f"[written to {x.out}/updates.md, updates.json]")
        acct = [f for f in R["findings"] if f["kind"] != "UNREVIEWED_PAST_SLA"]
        if (x.fail and acct) or (x.fail_if_behind and R["pacing"].get("falling_behind")):
            print("GATE FAILED: " + (f"{len(acct)} accounting problems" if x.fail and acct else "review backlog is growing"), file=sys.stderr)
            raise SystemExit(1)
    elif x.subcmd == "snapshot":
        for rec in updates.snapshot(x.paths):
            print(json.dumps(rec))
    elif x.subcmd == "replay":
        st, far = build_stack(DEFAULT_PROFILE)
        print(f"available: {', '.join(list_incidents())} | clean false-alarm rate {far:.0%}")
        inc = load_incident(x.incident)
        for s in range(3):
            e = run_incident(st, DEFAULT_PROFILE, inc, s, x.compress)
            print(f"seed {s}: detected={e['detected']} first-alarm={e['frac_to_alarm']} lenses={e['fired']}")
        print("note:", inc["note"])
    elif x.subcmd == "commit":
        dig, salt = commit(open(x.file, "rb").read())
        print(f"digest (publish): {dig}\nsalt   (keep secret until reveal): {salt}")
    elif x.subcmd == "reveal":
        print("VALID" if reveal_ok(open(x.file, "rb").read(), x.salt, x.digest) else "MISMATCH")
