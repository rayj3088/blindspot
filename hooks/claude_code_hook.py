#!/usr/bin/env python3
"""Local-first Claude Code hook: append one line per tool call to .blindspot/events.jsonl. Never blocks the agent.

Nothing leaves your machine. `blindspot audit .blindspot/events.jsonl` reads it back. Field names (tool_name,
tool_input, session_id) follow the PostToolUse JSON Claude Code sends on stdin; checked against the Claude Code
hooks documentation on 2026-09-26. PostToolUse runs after the tool, and this script always exits 0.
"""
import json
import os
import sys
import time

try:
    d = json.load(sys.stdin)
    ti = d.get("tool_input") or {}
    target = ti.get("file_path") or ti.get("path") or (ti.get("command") or "")[:200] or ti.get("url") or ""
    rec = {"t": time.time(), "tool": d.get("tool_name", ""), "target": target,
           "size": len(json.dumps(ti)), "agent": 0, "session": d.get("session_id", "")}
    os.makedirs(".blindspot", exist_ok=True)
    with open(os.path.join(".blindspot", "events.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
except Exception:
    pass
sys.exit(0)
