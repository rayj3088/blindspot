"""Plugins: make blindspot fit YOUR system without editing it.

A plugin is one Python file with a function `register(api)`. Pass it with `--plugin my_plugin.py` (repeatable) to `gate` or `audit-monitor`.
Through `api` it can add:

  api.add_attack(name, description, shape_fn, max_k80=None)
      A new attack shape. shape_fn(k, S, rng, spread) -> (times, kinds, sizes): relative times in seconds, kind codes (0-5, see blindspot.core.K),
      and sizes, for ONE attack of strength k. S holds the data's own yardsticks (p99_burst, p99_net, gap_med, kinds, targets, rare_pair).
      max_k80 is the strongest attack you are willing to leave unnoticed; without it the gate reports the shape but cannot pass or fail it.
  api.add_criterion(fn)
      A new gate criterion. fn(context) -> one criterion dict or a list of them: {"id", "name", "status": "PASS"|"FAIL"|"UNKNOWN", "value", "threshold", "detail"}.
      context has: audit (the black-box audit report), adversary, updates, policy, monitors. A plugin that raises or returns an invalid status is
      recorded as UNKNOWN, so a broken check can never turn into a pass.
  api.add_loader(name, fn)
      A reader for your log format. fn(path) -> blindspot.core.Trace. Use it with `--format name`.
  api.add_monitor(name, monitor)
      A monitor (a callable trace -> bool or score) included in every audit alongside --cmd and --py monitors.
"""
from __future__ import annotations

import importlib.util
import os

CUSTOM_SHAPES = {}
CUSTOM_CRITERIA = []
CUSTOM_LOADERS = {}
CUSTOM_MONITORS = {}
_LOADED = set()


class _API:
    def add_attack(self, name, description, shape_fn, max_k80=None):
        from . import detect_real, gate
        if not callable(shape_fn):
            raise TypeError("shape_fn must be callable")
        CUSTOM_SHAPES[name] = shape_fn
        detect_real.FAMILIES[name] = description
        if max_k80 is not None:
            gate.DEFAULT_POLICY["max_k80"][name] = max_k80

    def add_criterion(self, fn):
        if not callable(fn):
            raise TypeError("criterion must be callable")
        CUSTOM_CRITERIA.append(fn)

    def add_loader(self, name, fn):
        CUSTOM_LOADERS[name] = fn

    def add_monitor(self, name, monitor):
        CUSTOM_MONITORS[name] = monitor


def load_plugin(path):
    path = os.path.abspath(os.path.expanduser(path))
    if path in _LOADED:
        return
    if not os.path.exists(path):
        raise SystemExit(f"plugin not found: {path}")
    spec = importlib.util.spec_from_file_location(f"blindspot_plugin_{len(_LOADED)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "register"):
        raise SystemExit(f"{path} has no register(api) function")
    mod.register(_API())
    _LOADED.add(path)


def run_criteria(context):
    """Run every plugin criterion. Failures become UNKNOWN, never PASS."""
    out = []
    for fn in CUSTOM_CRITERIA:
        name = getattr(fn, "__name__", "plugin")
        try:
            res = fn(context)
            res = [res] if isinstance(res, dict) else list(res)
            for c in res:
                if c.get("status") not in ("PASS", "FAIL", "UNKNOWN") or not c.get("id") or not c.get("name"):
                    raise ValueError(f"invalid criterion {c!r}")
                out.append({"id": "plugin:" + str(c["id"]), "name": c["name"], "status": c["status"], "value": c.get("value"),
                            "threshold": c.get("threshold"), "detail": str(c.get("detail", ""))})
        except Exception as e:  # noqa: BLE001
            out.append({"id": f"plugin:{name}", "name": f"Plugin check {name}", "status": "UNKNOWN", "value": None, "threshold": None,
                        "detail": f"the plugin check failed ({type(e).__name__}: {e}) and is not counted as a pass"})
    return out
