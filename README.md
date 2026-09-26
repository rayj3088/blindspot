# blindspot

**A 60-second audit that tells you what your AI-agent monitoring can't see.**

Agents now change themselves faster than people can check. Most tools ask "did this action look bad?"
`blindspot` asks a different question: **"which attacks would my monitors miss?"** It attacks your monitor stack
with known attack shapes, replayed incident shapes, and thousands of *randomly generated* attacks, then shows you the holes.

> **Scope.** This measures *coverage against known and machine-generated attack shapes*.
> It does **not** measure safety against unknown attacks. A clean report is not a clean bill of health.

It does not slow agents down or block self-improvement. It only watches and audits.

## Quick start

```bash
pip install -e .            # needs numpy, scipy
python -m blindspot demo --quick      # ~12 s, no data needed
python -m blindspot demo              # fuller run
python -m blindspot audit ~/.claude/projects     # your own Claude Code logs (stay local)
python -m blindspot audit my_events.jsonl --observed independent.jsonl
python -m blindspot replay hf_2026_07 --compress 0.1
python tests/test_blindspot.py
```

Output goes to `blindspot-out/`: `report.md`, `report.json`, `escapes.jsonl` (attacks that got through, replayable).

## What it does

1. **Calibration gate.** Each lens gets a threshold so a clean trace false-alarms about 5%. Measured on unseen traces, reported up front.
2. **Known attacks.** Burst reads, recon sweeps, rulebook tampering, lockstep fleets, fabricated metrics, unlogged actions, loops, slow exfil, kill chains.
3. **Pace sweep.** The same attack stretched 0.25x to 16x. Slow attacks are harder; the report says how much.
4. **Incident replays.** Approximate shapes from public summaries (HF July 2026, Miasma June 2026). Illustrative, not reconstructions.
5. **Unknown-shape fuzzer.** Random attacks built from primitives, with Wilson confidence intervals, a coverage-vs-sample curve, coverage by harm level, and an escape corpus.
6. **Blind-spot map.** Effective number of independent lenses (SVD participation ratio), attacks caught by only one lens, irreplaceable lenses, greedy value order.
7. **Adaptive attacker.** A simple evolutionary search that shapes attacks to slip under every threshold. Reports how much harm it can deliver undetected and whether that transfers to fresh noise.

## The lenses (none read text, so none can be argued with)

| lens | idea |
|---|---|
| burst, sweep | sudden volume; many distinct targets |
| ctrl | writes to agent control files |
| netvol | bytes leaving |
| rhythm | machine-regular timing (scale-free) |
| surprisal | unusual tool-to-tool transitions |
| coupling | reads phase-locked with network sends |
| digits | last-digit forensics in bases 10, 7, 12, plus too-consistent variance |
| books | conservation check: reported vs independent record must balance |
| drift_* | SETI-style ordered-pattern scan at any pace |

## Decentralized pieces (`blindspot/commons.py`)

- Count-Min sketches that merge by addition, so groups can compare telemetry without sharing raw logs.
- Hash-chain ledger with witness heads.
- Commit-reveal for private hold-out attacks (`blindspot commit FILE`, `blindspot reveal FILE SALT DIGEST`).
- Hadamard code with a linearity test. Real local-decoding property, but exponential size, so small n only.

## Honest limits

- Coverage of known and generated shapes only.
- The attack library and fuzz grammar are open; attackers can read them. Keep private hold-outs.
- Baselines are synthetic unless you audit your own logs. Synthetic attackers are not frontier models.
- Lenses can't judge whether a change is *correct*, only whether it looks accounted for and ordinary.
- Perfect camouflage at low harm, or control of every record, passes.
- `books` needs an independent record. Without one it is off.
- Hidden-channel / steganography detection is not claimed.
- Incident replays are approximations.
- Novelty: I found no existing tool that audits agent *monitors* this way, but that is unverified. Please tell me if it exists.
- With few real events, the real-log calibration doesn't mean much yet; watch the "alarmed chunks" line.

## Claude Code hook (local-first, never blocks)

`hooks/claude_code_hook.py` appends events to `.blindspot/events.jsonl`. Example `.claude/settings.json`
(verify field names against current Claude Code docs; I have not):

```json
{"hooks": {"PostToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": "python3 /path/to/hooks/claude_code_hook.py"}]}]}}
```

See [RESOURCES.md](RESOURCES.md) for related work. License: MIT.
