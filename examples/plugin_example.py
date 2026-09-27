"""Example plugin. Run:
   python -m blindspot gate --real logs.csv --format csv_events --plugin examples/plugin_example.py --builtin-split --chunk-minutes 30
It adds a log reader for a CSV with columns time,tool,size, a new attack shape, and a custom gate check. Copy it and change it."""
import numpy as np

from blindspot.core import N_CTRL, Trace
from blindspot.tracelab import kind_of


def read_csv(path):
    t, kind, size = [], [], []
    for line in open(path).read().splitlines()[1:]:
        a, b, c = line.split(",")[:3]
        t.append(float(a)); kind.append({"read": 0, "write": 1, "bash": 2, "net": 3, "search": 4}.get(kind_of(b), 2)); size.append(float(c))
    n = len(t)
    return Trace.make(np.array(t), np.array(kind), N_CTRL + np.arange(n) % 50, np.array(size))


def credential_probe(k, S, rng, spread):
    """k x 15 quick, tiny outbound calls, the way a probe for a valid credential looks."""
    n = max(5, int(15 * k))
    return np.sort(rng.uniform(0, 30.0 * spread, n)), np.full(n, 3), np.full(n, 40.0)


def has_canary(ctx):
    have = [m for m in ctx["monitors"] if "canary" in m]
    return {"id": "canary", "name": "A canary monitor is in the set", "status": "PASS" if have else "FAIL", "value": len(have), "threshold": 1,
            "detail": "canary monitors present: " + (", ".join(have) if have else "none")}


def register(api):
    api.add_loader("csv_events", read_csv)
    api.add_attack("credential_probe", "k x 15 tiny outbound calls within 30 seconds", credential_probe, max_k80=4)
    api.add_criterion(has_canary)
