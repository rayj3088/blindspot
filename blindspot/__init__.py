"""blindspot: a coverage audit for AI-agent oversight stacks."""
from .attacks import LIBRARY, Bench, evade, fuzz, random_spec, render
from .audit import build_stack, effective_rank, run_audit
from .core import DEFAULT_PROFILE, Trace, fit_profile, load, synth_baseline
from .external import CmdMonitor, PyMonitor, audit_monitor
from .lenses import LENSES, Stack

__version__ = "0.1.0"
