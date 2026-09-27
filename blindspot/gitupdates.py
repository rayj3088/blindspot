"""Self-update accounting from a git repository, so it works on real data today.

Agents like Claude Code edit their own instruction and control files (CLAUDE.md, AGENTS.md, .claude/, .cursor/, hooks, CI
workflows). Git already keeps the declared history of those edits. The working tree on disk is an independent observation of
what the files are RIGHT NOW. This module turns both into the records `blindspot updates` understands:

  change    one per commit that touched a watched path (old and new state of that path, actor = agent or human)
  snapshot  the current working-tree state of each watched path
  review    a human commit counts as reviewed by its author; an agent-involved commit counts as reviewed only if its message
            carries a `Reviewed-by:` trailer (or your --review-trailer)

Then the usual checks apply: an edit made on disk that is in no commit shows up as an UNEXPLAINED_CHANGE, and agent commits
nobody has reviewed pile up as a backlog.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from datetime import datetime

DEFAULT_PATHS = ["AGENTS.md", "CLAUDE.md", ".claude", ".cursor", ".gemini", ".vscode/tasks.json", ".github/workflows"]
DEFAULT_AGENT = r"claude|anthropic|copilot|codex|cursor|gemini|\bbot\b|noreply@anthropic"


def _git(repo, *args, check=True):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, errors="ignore")
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {p.stderr.strip()[:200]}")
    return p.stdout


def _state(pairs):
    """pairs: iterable of (path, blob id). One stable hash for a file or folder."""
    h = hashlib.sha256()
    n = 0
    for name, bid in sorted(pairs):
        h.update(name.encode()); h.update(b"\0"); h.update(bid.encode()); h.update(b"\n")
        n += 1
    return h.hexdigest()[:16] if n else "absent"


def _state_at(repo, rev, path):
    if rev is None:
        return "absent"
    out = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "-z", rev, "--", path], capture_output=True, text=True, errors="ignore")
    pairs = []
    for rec in out.stdout.split("\0"):
        if not rec:
            continue
        meta, _, name = rec.partition("\t")
        parts = meta.split()
        if len(parts) >= 3:
            pairs.append((name, parts[2]))
    return _state(pairs)


def _state_now(repo, path):
    names = [n for n in _git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", path).split("\0") if n]
    pairs = []
    for n in names:
        fp = os.path.join(repo, n)
        if os.path.isfile(fp):
            pairs.append((n, _git(repo, "hash-object", "--", n).strip()))
    return _state(pairs)


def records_from_git(repo, paths=None, agent_regex=DEFAULT_AGENT, review_trailer="Reviewed-by", since=None):
    paths = paths or DEFAULT_PATHS
    if not os.path.isdir(os.path.join(repo, ".git")) and _git(repo, "rev-parse", "--is-inside-work-tree", check=False).strip() != "true":
        raise SystemExit(f"{repo} is not a git repository")
    is_agent = re.compile(agent_regex, re.I)
    has_review = re.compile(rf"^{re.escape(review_trailer)}:", re.I | re.M)
    fmt = "%H%x1f%aI%x1f%an <%ae>%x1f%B%x1e"
    args = ["log", "--first-parent", "--reverse", f"--format={fmt}"]
    if since:
        args.append(f"--since={since}")
    raw = _git(repo, *args, "--", *paths, check=False)
    recs, agent_commits = [], 0
    for chunk in raw.split("\x1e"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        sha, when, author, msg = chunk.split("\x1f", 3)
        t = datetime.fromisoformat(when).timestamp()
        parent = _git(repo, "rev-parse", "--verify", "--quiet", f"{sha}^1", check=False).strip() or None
        if parent is None:
            touched = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", sha).splitlines()
        else:
            touched = _git(repo, "diff", "--name-only", parent, sha).splitlines()
        agentic = bool(is_agent.search(author)) or bool(re.search(r"^co-authored-by:.*", msg, re.I | re.M) and is_agent.search(msg))
        agent_commits += agentic
        for p in paths:
            if not any(f == p or f.startswith(p.rstrip("/") + "/") for f in touched):
                continue
            old, new = _state_at(repo, parent, p), _state_at(repo, sha, p)
            if old == new:
                continue
            recs.append({"type": "change", "t": t, "artifact": p, "old": old, "new": new, "actor": "agent" if agentic else "human",
                         "commit": sha[:10], "author": author})
            if (not agentic) or has_review.search(msg):
                recs.append({"type": "review", "t": t, "artifact": p, "new": new, "by": "author" if not agentic else "trailer", "result": "pass"})
    now = datetime.now().timestamp()
    for p in paths:
        if any(r["artifact"] == p for r in recs) or _state_now(repo, p) != "absent":
            recs.append({"type": "snapshot", "t": now, "artifact": p, "hash": _state_now(repo, p)})
    return sorted(recs, key=lambda r: (r["t"], {"snapshot": 2, "change": 0, "review": 1}[r["type"]])), {"agent_commits": agent_commits}
