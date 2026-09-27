"""Audit agentnorm (https://github.com/kaustubhspatil/agentnorm) with blindspot.

Written from agentnorm's documented API: `Monitor.fit(history)` on benign `Run`s, then `monitor.score(run).flagged`;
`Run(agent, version, principal, calls=[...])` with `ToolCall(tool, step, args, result_size, output_bytes, resource, scope, started_at)`.
Not yet run against the real package. Install agentnorm yourself first (see its README), then:
    python -m blindspot audit-monitor --agentnorm --quick
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .core import DEFAULT_PROFILE, synth_baseline
from .external import to_events


class AgentnormMonitor:
    def __init__(self, n_fit=60, seed=0, hours=3.0, prof=DEFAULT_PROFILE, fit_chunks=None):
        """fit_chunks: clean REAL chunks to fit agentnorm on (recommended when you have them). Keep them separate from the chunks it is tested on."""
        try:
            from agentnorm import Monitor
            from agentnorm.trace import Run, ToolCall
        except ImportError as e:
            raise ImportError("agentnorm is not installed. Install it as its README says, then rerun.") from e
        self._Run, self._ToolCall = Run, ToolCall
        rng = np.random.default_rng(seed + 31)
        history = [self._run(c) for c in fit_chunks[:n_fit]] if fit_chunks else [self._run(synth_baseline(rng, hours, prof)) for _ in range(n_fit)]
        self.monitor = Monitor.fit(history)
        self.fitted_on = "real" if fit_chunks else "synthetic"

    def _run(self, tr):
        calls = [self._ToolCall(tool=e["tool"], step=i, args={"target": e["target"]}, result_size=int(e["size"]), output_bytes=int(e["size"]),
                                resource=e["target"], started_at=datetime.fromtimestamp(e["t"], timezone.utc))
                 for i, e in enumerate(to_events(tr))]
        return self._Run(agent="audited", version="v0", calls=calls)

    def __call__(self, tr):
        return bool(self.monitor.score(self._run(tr)).flagged)
