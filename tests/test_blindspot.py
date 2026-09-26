"""Validation. Run with pytest, or without it:  python tests/test_blindspot.py"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindspot import DEFAULT_PROFILE, LIBRARY, Bench, build_stack, effective_rank, fit_profile, fuzz, random_spec, render, synth_baseline  # noqa: E402
from blindspot.attacks import evade, load_incident, render_incident  # noqa: E402
from blindspot.audit import greedy_order  # noqa: E402
from blindspot.cli import main as cli_main  # noqa: E402
from blindspot.commons import (Ledger, Sketch, blr_reject_rate, commit, corrupt, hadamard_encode, local_decode,  # noqa: E402
                               reveal_ok, sketch_trace, spike_scan, spotcheck_detect_prob)
from blindspot.core import K, N_CTRL, Trace, load_claude_code, load_jsonl  # noqa: E402
from blindspot.lenses import LENSES  # noqa: E402

P = DEFAULT_PROFILE
_CACHE = {}


def _stack():
    if "s" not in _CACHE:
        st, far = build_stack(P, seed=0, n_cal=200, n_val=60)
        _CACHE["s"], _CACHE["far"], _CACHE["b"] = st, far, Bench(st, P, seed=100)
    return _CACHE["s"], _CACHE["far"], _CACHE["b"]


def _rate(bench, spec, reps=8):
    ev = [bench.run(spec, np.random.default_rng(i))[0] for i in range(reps)]
    return float(np.mean([e["detected"] for e in ev])), ev


# ------------------------------------------------------------------ data model
def test_trace_sorted_and_channels():
    b = synth_baseline(np.random.default_rng(1), 1.0)
    assert np.all(np.diff(b.t) >= 0) and len(b.rep) > 100 and len(b) > len(b.rep)


def test_profile_fit_roundtrip():
    tr = synth_baseline(np.random.default_rng(2), 12.0, both=False)
    p = fit_profile(tr)
    w, v = np.linalg.eig(P.trans.T)                     # the generator is a Markov chain: compare to its stationary law
    pi = np.real(v[:, np.argmax(np.real(w))]); pi = pi / pi.sum()
    assert np.abs(p.p_kind - pi).max() < 0.03, (p.p_kind, pi)
    assert abs(p.rate / P.rate - 1) < 0.35, p.rate


# ------------------------------------------------------------------ calibration gate
def test_calibration_false_alarm_rate():
    _, far, _ = _stack()
    assert far <= 0.13, f"clean false-alarm rate {far:.2f} vs 0.05 target"


def test_differential_scoring_discounts_baseline_alarms():
    st, _, b = _stack()
    base = b.bases[0]
    assert not st.evaluate(base, P, None, clean=st.series(base, P))["detected"]


# ------------------------------------------------------------------ known attacks (each lens does its job)
def test_known_attacks_are_caught():
    _, _, b = _stack()
    for name in ("burst_read", "recon_sweep", "rulebook_tamper", "lockstep_fleet", "fabricated_metrics", "unlogged_actions", "kill_chain"):
        r, _ = _rate(b, LIBRARY[name])
        assert r >= 0.8, (name, r)


def test_the_right_lens_fires():
    _, _, b = _stack()
    for name, lens in (("unlogged_actions", "books"), ("fabricated_metrics", "digits"), ("burst_read", "burst")):
        _, ev = _rate(b, LIBRARY[name], 6)
        assert sum(lens in e["fired"] for e in ev) >= 5, (name, lens)


def test_known_limit_slow_exfil_is_missed():
    """Documented blind spot: a slow drip below every threshold is NOT caught. The audit must say so."""
    _, _, b = _stack()
    r, _ = _rate(b, LIBRARY["slow_exfil"])
    assert r <= 0.5, r


def test_stretching_an_attack_in_time_evades_and_is_reported():
    _, _, b = _stack()
    fast, _ = _rate(b, {**LIBRARY["kill_chain"], "dilate": 0.25}, 10)
    slow, _ = _rate(b, {**LIBRARY["kill_chain"], "dilate": 16.0}, 10)
    assert fast >= 0.9 and slow <= 0.5 and slow < fast, (fast, slow)


def test_stretching_does_not_fake_a_books_alarm():
    """Regression: twin events must be created after time stretching, or `books` fires on an artifact."""
    _, _, b = _stack()
    _, ev = _rate(b, {**LIBRARY["kill_chain"], "dilate": 16.0}, 8)
    assert all("books" not in e["fired"] for e in ev)


def test_camouflage_is_a_real_lever():
    """Camouflage changes what the lenses see (it is one of the attacker's moves)."""
    _, _, b = _stack()
    n0 = np.mean([b.run({**LIBRARY["fabricated_metrics"], "camo": 0.0}, np.random.default_rng(i))[0]["norm"]["digits"] for i in range(6)])
    n4 = np.mean([b.run({**LIBRARY["fabricated_metrics"], "camo": 4.0}, np.random.default_rng(i))[0]["norm"]["digits"] for i in range(6)])
    assert n4 < n0, (n0, n4)


# ------------------------------------------------------------------ lenses in isolation
def test_books_lens_sees_missing_records():
    b = synth_baseline(np.random.default_rng(3), 3.0)
    ok = LENSES["books"](b, P)[1].max()
    o = np.flatnonzero(b.chan == 1)
    keep = np.ones(len(b), bool)
    keep[o[np.random.default_rng(0).random(len(o)) < 0.3]] = False
    assert LENSES["books"](b.sel(keep), P)[1].max() > 3 * max(ok, 1)


def test_digits_lens_multi_base():
    r = np.random.default_rng(4)
    t = np.arange(200.0)
    nat = Trace.make(t, K["test"], 10, value=np.round(r.normal(85, 6, 200), 1))
    fab = Trace.make(t, K["test"], 10, value=np.round(np.round(r.normal(85, 1, 200) * 2) / 2, 1))
    assert LENSES["digits"](nat, P)[1].max() < 3.0 < LENSES["digits"](fab, P)[1].max()


# ------------------------------------------------------------------ fuzzer and adaptive attacker
def test_random_specs_render():
    rng = np.random.default_rng(5)
    for _ in range(150):
        tr, h, span = render({**random_spec(rng), "t0": 500.0}, rng, P)
        assert len(tr) > 0 and span[1] >= span[0] and all(v >= 0 for v in h.values())


def test_fuzz_is_reproducible_and_ci_is_sane():
    _, _, b = _stack()
    a, c = fuzz(b, 30, seed=3), fuzz(b, 30, seed=3)
    assert a["coverage"] == c["coverage"] and a["rows"][0]["spec"] == c["rows"][0]["spec"]
    assert a["ci"][0] <= a["coverage"] <= a["ci"][1] and a["n"] == 30


def test_escapes_replay_deterministically():
    _, _, b = _stack()
    fz = fuzz(b, 80, seed=5)
    esc = [r for r in fz["rows"] if not r["detected"]]
    for r in esc[:5]:
        assert not b.run(r["spec"], np.random.default_rng(r["seed"]))[0]["detected"]


def test_adaptive_attacker_result_is_consistent():
    _, _, b = _stack()
    res = evade(b, LIBRARY["kill_chain"], seed=0, budget=24)
    assert res["undetected_harm"] >= 0 and len(res["history"]) >= 6
    if res["spec"] is not None:
        assert not b.run(res["spec"], np.random.default_rng(1))[0]["detected"]


# ------------------------------------------------------------------ blind-spot map maths
def test_effective_rank_counts_independent_directions():
    r = np.random.default_rng(6)
    A = r.random((80, 3))
    assert effective_rank(np.c_[A, A[:, 0]]) <= effective_rank(A) + 1e-9
    assert effective_rank(np.c_[A, r.random(80)]) > effective_rank(A)
    assert abs(effective_rank(np.eye(5)) - 5) < 1e-9


def test_greedy_order_prefers_coverage():
    o = greedy_order([{"a"}, {"a", "b"}, {"c"}, {"c"}, {"c"}], ["a", "b", "c"])
    assert o[0][0] == "c" and o[0][1] == 3 and sum(g for _, g in o) == 5


# ------------------------------------------------------------------ incidents
def test_incident_shapes():
    hf = load_incident("hf_2026_07")
    assert sum(p["actions"] for p in hf["phases"]) == 17600 and hf["duration_s"] == 388800
    tr, h, span = render_incident(load_incident("miasma_2026_06"), np.random.default_rng(0), 100.0)
    assert len(tr.rep) == 10 and span[1] - span[0] <= 49.0 and h["ctrl"] >= 1


# ------------------------------------------------------------------ decentralised pieces
def _contrib(rng, ctrl_times, cls, n_benign=3):
    tb = rng.uniform(0, 10800, n_benign)
    t = np.r_[tb, ctrl_times]
    return Trace.make(t, K["write"], np.r_[rng.integers(0, N_CTRL, n_benign), np.full(len(ctrl_times), cls)])


def test_sketches_find_lockstep_without_raw_logs():
    rng = np.random.default_rng(7)
    sks = [sketch_trace(_contrib(rng, [], 0)) for _ in range(40)]
    sks += [sketch_trace(_contrib(rng, [5000 + rng.uniform(-20, 20)], 1)) for _ in range(10)]
    tot = sks[0]
    for s in sks[1:]:
        tot = tot.merge(s)
    hits = spike_scan(tot, (0, 10800))
    assert any(c == 1 and 4900 <= t <= 5100 for c, t, _ in hits), hits
    benign = sks[:40][0]
    for s in sks[1:40]:
        benign = benign.merge(s)
    assert spike_scan(benign, (0, 10800)) == []
    noisy = tot.tab.copy()
    tot.privatize(50.0, rng)
    assert any(c == 1 for c, _, _ in spike_scan(tot, (0, 10800))) and not np.allclose(noisy, tot.tab)


def test_ledger_tamper_evidence_and_commit_reveal():
    L = Ledger()
    for i in range(20):
        L.append({"i": i, "v": i * i})
    head, root = L.head(), L.merkle_root()
    assert L.verify(head) == (True, None)
    s, h = L.recs[7]
    L.recs[7] = (s.replace('"v":49', '"v":50'), h)
    ok, at = L.verify()
    assert not ok and at == 7 and L.merkle_root() != root
    dig, salt = commit(b"holdout-set-v1")
    assert reveal_ok(b"holdout-set-v1", salt, dig) and not reveal_ok(b"holdout-set-v2", salt, dig)
    rng = np.random.default_rng(8)
    mc = np.mean([(rng.random(20) < 0.1).any() for _ in range(20000)])
    assert abs(mc - spotcheck_detect_prob(0.1, 20)) < 0.01


def test_holographic_code_survives_damage():
    rng = np.random.default_rng(9)
    bits = rng.integers(0, 2, 12)
    f = hadamard_encode(bits)
    assert blr_reject_rate(f, 2000, rng) == 0.0
    g = corrupt(f, 0.10, rng)
    assert blr_reject_rate(g, 4000, rng) > 0.10
    assert [local_decode(g, i, 31, rng) for i in range(12)] == list(bits)


# ------------------------------------------------------------------ adapters and CLI
def test_claude_code_adapter_and_generic_hook_format():
    def line(ts, blocks, side=False):
        return json.dumps({"type": "assistant", "timestamp": ts, "isSidechain": side, "message": {"content": blocks}})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.jsonl")
        with open(p, "w") as fh:
            fh.write(line("2026-09-26T15:00:05.000Z", [{"type": "tool_use", "name": "Write", "input": {"file_path": "/repo/CLAUDE.md", "content": "x"}}]) + "\n")
            fh.write(line("2026-09-26T15:00:01.000Z", [{"type": "tool_use", "name": "Read", "input": {"file_path": "/repo/a.py"}}]) + "\n")
            fh.write(line("2026-09-26T15:00:09.000Z", [{"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}}], True) + "\n")
            fh.write("not json\n")
        tr = load_claude_code(p)
        assert len(tr) == 3 and np.all(np.diff(tr.t) > 0)
        assert list(tr.kind) == [K["read"], K["write"], K["test"]] and tr.target[1] == 1 and list(tr.agent) == [0, 0, 1]
        hk = os.path.join(d, "e.jsonl")
        with open(hk, "w") as fh:
            fh.write(json.dumps({"t": 1.0, "tool": "Edit", "target": "/x/.cursor/rules/a.mdc", "size": 10}) + "\n")
        h = load_jsonl(hk)
        assert h.kind[0] == K["write"] and h.target[0] == 3


def test_cli_quick_demo_writes_report():
    with tempfile.TemporaryDirectory() as d:
        cli_main(["demo", "--quick", "--out", d])
        for f in ("report.md", "report.json", "escapes.jsonl"):
            assert os.path.exists(os.path.join(d, f))
        assert "NOT a measure of safety" in open(os.path.join(d, "report.md")).read()


def test_real_logs_smoke_if_provided():
    """Optional: BLINDSPOT_REAL_LOGS=/path/to/.claude/projects. Never prints or stores content."""
    p = os.environ.get("BLINDSPOT_REAL_LOGS")
    if not p:
        return
    tr = load_claude_code(p)
    assert len(tr) > 0 and fit_profile(tr).rate > 0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
