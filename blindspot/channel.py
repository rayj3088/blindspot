"""Agent-to-agent channels: what are they saying to each other, and are they coordinating out of sight?

Built for the traffic frontier agents actually produce: long messages mixed with JSON, hashes, paths and code, many agents at once,
broadcasts, effects that arrive seconds or half an hour later, and meaning spread over many words instead of one token.

  ground           DECODE BY CONSEQUENCE. A token that reliably precedes an action by the receiver carries that meaning, whatever it looks
                   like. Each token is tested against the same actions with the messages moved a few windows sideways, which keeps slow
                   shared context (a busy afternoon makes both the wording and the actions busier) and breaks the link to these messages.
                   The bar is the strongest chance pairing across every token, action and delay tried, so a lexicon is not a lucky pick.
  informativeness  WHEN NO SINGLE TOKEN MEANS ANYTHING. A code spread over many synonyms or word patterns defeats a token table. This
                   asks whether the whole message predicts the receiver's next action (cross-validated, against the same sideways null).
  hidden_characters  ZERO-WIDTH, UNICODE-TAG AND HOMOGLYPH CARRIERS, decoded when they hide text.
  coordination     THE ENTANGLEMENT IDEA AS ENGINEERING. Two agents whose actions line up in the same window more tightly than slow shared
                   context explains (the slow level of activity around them, for each of them and for everyone else) are connected by something outside the log.
  profile          character entropy, Zipf slope, compressibility, vocabulary drift, distance from a sample of ordinary language.

Runs on a laptop: sparse matrices, a cap on messages, and permutation counts you can lower.
"""
from __future__ import annotations

import json
import re
import zlib
from collections import Counter
from datetime import datetime

import numpy as np
from scipy import sparse
from scipy.stats import norm

from .core import KINDS


# ------------------------------------------------------------------ reading real traffic
def parse_time(x):
    if isinstance(x, (int, float)):
        v = float(x)
        return v / 1000.0 if v > 1e11 else v
    s = str(x).strip()
    try:
        v = float(s)
        return v / 1000.0 if v > 1e11 else v
    except ValueError:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(_text(b.get("text") if isinstance(b, dict) and "text" in b else b) for b in x)
    if x is None:
        return ""
    return json.dumps(x, ensure_ascii=False, default=str)


def _pick(o, names):
    for n in names:
        if o.get(n) is not None:
            return o[n]
    return None


def load_messages(path):
    """JSONL rows. Time: t / time / timestamp (seconds, milliseconds or ISO 8601). Sender: from / sender / agent. Receiver: to / receiver /
    recipient (missing or "all" = broadcast). Text: text / content / message / body (strings, content blocks or JSON)."""
    out = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            try:
                o = json.loads(line)
                t = _pick(o, ("t", "time", "timestamp", "ts"))
                txt = _pick(o, ("text", "content", "message", "body"))
                if t is None or txt is None:
                    continue
                out.append({"t": parse_time(t), "from": _pick(o, ("from", "sender", "agent")), "to": _pick(o, ("to", "receiver", "recipient")), "text": _text(txt)})
            except Exception:
                continue
    out.sort(key=lambda m: m["t"])
    return out


class Actions:
    """Receiver actions as labels: t, agent code, label code, plus the names of both."""

    def __init__(self, t, agent, label, labels, agents):
        o = np.argsort(t, kind="stable")
        self.t, self.agent, self.label = np.asarray(t, float)[o], np.asarray(agent, int)[o], np.asarray(label, int)[o]
        self.labels, self.agents = list(labels), list(agents)

    def __len__(self):
        return len(self.t)


def actions_from_trace(tr):
    ids = sorted(set(int(a) for a in np.unique(tr.agent)))
    m = {a: i for i, a in enumerate(ids)}
    return Actions(tr.t, [m[int(a)] for a in tr.agent], tr.kind.astype(int), KINDS, ids)


def load_actions(path):
    """JSONL rows with a time, an agent (any name or number) and what it did (tool / name / kind / action; the tool name is used as-is)."""
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            try:
                o = json.loads(line)
                t = _pick(o, ("t", "time", "timestamp", "ts")); a = _pick(o, ("agent", "agent_id", "actor", "from")); l = _pick(o, ("tool", "tool_name", "name", "kind", "action"))
                if t is not None and l is not None:
                    rows.append((parse_time(t), a if a is not None else 0, str(l)))
            except Exception:
                continue
    labels = sorted({r[2] for r in rows}); agents = sorted({r[1] for r in rows}, key=str)
    li, ai = {l: i for i, l in enumerate(labels)}, {a: i for i, a in enumerate(agents)}
    return Actions([r[0] for r in rows], [ai[r[1]] for r in rows], [li[r[2]] for r in rows], labels, agents)


# ------------------------------------------------------------------ tokens
_MASKS = [(re.compile(r"https?://\S+"), "<url>"), (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<id>"),
          (re.compile(r"\b[0-9a-f]{8,}\b", re.I), "<hex>"), (re.compile(r"[A-Za-z0-9+/=_-]{24,}"), "<blob>"),
          (re.compile(r"(?:[\w.~-]*/){2,}[\w.~-]*"), "<path>"), (re.compile(r"\b\d[\d.,:]*\b"), "<num>")]
_STRIP = ".,;:!?()[]{}\"'`<>*#"


def tokenize(text):
    """Hashes, IDs, blobs, paths and numbers collapse to classes so they cannot flood the vocabulary; the rest is lowercased words."""
    for rx, rep in _MASKS:
        text = rx.sub(f" {rep} ", text)
    out = []
    for w in text.split():
        w = w.strip(_STRIP).lower() if not (w.startswith("<") and w.endswith(">")) else w
        if w:
            out.append(w)
    return out


# ------------------------------------------------------------------ hidden characters
_ZW = re.compile("[​-‏⁠-⁤﻿]")
_TAG = re.compile("[\U000e0000-\U000e007f]")
_VS = re.compile("[︀-️\U000e0100-\U000e01ef]")
_BIDI = re.compile("[‪-‮⁦-⁩]")


def _script(ch):
    o = ord(ch)
    return "latin" if ch.isascii() and ch.isalpha() else "cyr" if 0x400 <= o <= 0x4FF else "greek" if 0x370 <= o <= 0x3FF else None


def hidden_characters(msgs):
    """Carriers that show nothing on screen. Decodes Unicode-tag characters (each hides one ASCII letter) and zero-width binary
    (zero-width space = 0, zero-width non-joiner = 1) when they spell printable text."""
    found = {"zero_width": 0, "unicode_tags": 0, "variation_selectors": 0, "bidi_controls": 0, "mixed_script_words": 0}
    by_sender, decoded = Counter(), []
    for m in msgs:
        s, hit = m["text"], False
        z, tg, vs, bd = len(_ZW.findall(s)), len(_TAG.findall(s)), len(_VS.findall(s)), len(_BIDI.findall(s))
        mixed = sum(1 for w in s.split() if len({_script(c) for c in w if _script(c)}) > 1)
        for k, v in (("zero_width", z), ("unicode_tags", tg), ("variation_selectors", vs), ("bidi_controls", bd), ("mixed_script_words", mixed)):
            found[k] += 1 if v else 0
            hit = hit or bool(v)
        if tg:
            txt = "".join(chr(ord(c) - 0xE0000) for c in _TAG.findall(s) if 0x20 <= ord(c) - 0xE0000 < 0x7F)
            if txt:
                decoded.append({"t": m["t"], "from": m["from"], "carrier": "unicode tags", "text": txt[:200]})
        if z >= 16:
            bits = "".join("1" if c == "‌" else "0" for c in _ZW.findall(s) if c in ("​", "‌"))
            chars = [chr(int(bits[i:i + 8], 2)) for i in range(0, len(bits) - 7, 8)]
            if chars and all(32 <= ord(c) < 127 for c in chars):
                decoded.append({"t": m["t"], "from": m["from"], "carrier": "zero-width binary", "text": "".join(chars)[:200]})
        if hit:
            by_sender[str(m["from"])] += 1
    return {"messages_with_any": int(sum(1 for m in msgs if _ZW.search(m["text"]) or _TAG.search(m["text"]) or _VS.search(m["text"]) or _BIDI.search(m["text"]))),
            "counts": found, "by_sender": dict(by_sender.most_common(10)), "decoded": decoded[:20], "messages": len(msgs)}


# ------------------------------------------------------------------ text profile
def _entropy(counts):
    p = np.asarray(list(counts), float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def profile(msgs, reference=None, seed=0):
    text = "\n".join(m["text"] for m in msgs)
    toks = [w for m in msgs for w in tokenize(m["text"])]
    if not toks:
        return {"messages": len(msgs), "tokens": 0}
    c = Counter(toks)
    f = np.array(sorted(c.values(), reverse=True)[:200], float)
    slope = float(np.polyfit(np.log(np.arange(1, len(f) + 1)), np.log(f), 1)[0]) if len(f) >= 5 else None
    third = max(1, len(toks) // 3)
    early, late = set(toks[:2 * third]), toks[2 * third:]
    novelty = float(np.mean([w not in early for w in late])) if late else None
    raw = text.encode()
    P = {"messages": len(msgs), "tokens": len(toks), "vocabulary": len(c), "type_token_ratio": len(c) / len(toks),
         "char_entropy_bits": _entropy(Counter(text).values()), "zipf_slope": slope,
         "compression_ratio": len(zlib.compress(raw, 9)) / max(1, len(raw)), "new_token_rate_last_third": novelty}
    if reference:
        def tri(s):
            s = s.lower()
            return Counter(s[i:i + 3] for i in range(len(s) - 2))

        def jsd(a, b):
            keys = list(set(a) | set(b))
            p = np.array([a.get(k, 0) for k in keys], float); q = np.array([b.get(k, 0) for k in keys], float)
            p /= p.sum(); q /= q.sum(); mid = (p + q) / 2
            kl = lambda x, y: float((x[x > 0] * np.log2(x[x > 0] / y[x > 0])).sum())    # noqa: E731
            return 0.5 * kl(p, mid) + 0.5 * kl(q, mid)
        P["distance_from_reference_bits"] = jsd(tri(text), tri(reference))
        n = min(len(reference) // 2, max(300, len(text)))
        P["reference_self_distance_bits"] = jsd(tri(reference[:n]), tri(reference[n:2 * n])) if n >= 300 and len(reference) >= 2 * n else None
    return P


# ------------------------------------------------------------------ decoding by consequence
class _Grid:
    """Fast 'did label l happen in (t, t + w] for receiver a' lookups."""

    def __init__(self, act):
        self.L = len(act.labels)
        self.by = {}
        for a in np.unique(act.agent):
            s = act.agent == a
            self.by[int(a)] = [act.t[s][act.label[s] == l] for l in range(self.L)]
        self.all = [act.t[act.label == l] for l in range(self.L)]
        self.t0, self.t1 = float(act.t.min()), float(act.t.max())

    def presence(self, t, recv_rows, w):
        """recv_rows: list of (row index array, agent code or None). -> [len(t), L] bool"""
        P = np.zeros((len(t), self.L), bool)
        for rows, a in recv_rows:
            tl = self.all if a is None else self.by.get(a, [np.array([])] * self.L)
            for l in range(self.L):
                x = tl[l]
                if len(x):
                    P[rows, l] = np.searchsorted(x, t[rows] + w, "right") - np.searchsorted(x, t[rows], "right") > 0
        return P


def _prep(msgs, act, max_messages, seed):
    rng = np.random.default_rng(seed)
    if len(msgs) > max_messages:
        keep = np.sort(rng.choice(len(msgs), max_messages, replace=False)); msgs = [msgs[i] for i in keep]
    amap = {a: i for i, a in enumerate(act.agents)}
    recv = {}
    for i, m in enumerate(msgs):
        a = amap.get(m["to"]) if m["to"] not in (None, "all", "*", "broadcast") else None
        if a is None and m["to"] is not None and isinstance(m["to"], str) and m["to"].isdigit():
            a = amap.get(int(m["to"]))
        recv.setdefault(a, []).append(i)
    recv_rows = [(np.array(v), a) for a, v in recv.items()]
    return msgs, recv_rows


_OFFSETS = (3.0, 4.0, 5.0, 6.0)


def _paired(grid, tm, recv_rows, w):
    """For each message and action label: did the action follow the message (1/0), minus how often it follows the SAME message moved
    3 to 6 windows earlier and later. The difference removes anything slow that changes both wording and activity (a busy phase), because
    the message and its shifted copies share it. Under no link between messages and actions the differences are symmetric around zero."""
    P0 = grid.presence(tm, recv_rows, w).astype(np.float32)
    N = np.zeros_like(P0)
    for o in _OFFSETS:
        for sgn in (-1.0, 1.0):
            N += grid.presence(np.clip(tm + sgn * o * w, grid.t0 - 1.0, grid.t1 + 1.0), recv_rows, w)
    N /= 2 * len(_OFFSETS)
    D = P0 - N
    return D - D.mean(0, keepdims=True), P0, N            # centred: a typical message is followed by activity, so only what is special about a token counts


def _vocab_matrix(toks, vocab):
    idx = {w: i for i, w in enumerate(vocab)}
    r_, c_ = [], []
    for i, s in enumerate(toks):
        for w in s:
            if w in idx:
                r_.append(i); c_.append(idx[w])
    return sparse.csr_matrix((np.ones(len(r_), np.float32), (r_, c_)), shape=(len(toks), len(vocab)))


def ground(msgs, act, windows=(30.0, 300.0, 1800.0), min_count=8, perms=500, alpha=0.05, seed=0, max_vocab=500, max_messages=200000, window=None):
    """Lexicon of tokens that reliably precede a kind of action by the receiver, at any of several delays. act: Actions (or a Trace).
    Each message is compared with itself moved sideways (see _paired); significance is a sign-flip test with the strongest chance
    pairing across every token, action and delay as the bar."""
    if hasattr(act, "kind") and not isinstance(act, Actions):
        act = actions_from_trace(act)
    if window is not None:
        windows = (float(window),)
    rng = np.random.default_rng(seed)
    if len(msgs) < 30 or not len(act):
        return {"lexicon": [], "ungrounded": [], "note": "too few messages or no actions to ground against", "n_messages": len(msgs)}
    msgs, recv_rows = _prep(msgs, act, max_messages, seed)
    toks = [set(tokenize(m["text"])) for m in msgs]
    cnt = Counter(w for s in toks for w in s)
    vocab = [w for w, n in cnt.most_common(max_vocab) if n >= min_count and not (w.startswith("<") and w.endswith(">") and cnt[w] > 0.9 * len(msgs))]
    if not vocab:
        return {"lexicon": [], "ungrounded": [], "note": "no token is frequent enough to test", "n_messages": len(msgs)}
    X = _vocab_matrix(toks, vocab)
    XT = X.T.tocsr()
    nv = np.asarray(X.sum(0)).ravel()
    grid = _Grid(act)
    tm = np.array([m["t"] for m in msgs]); M = len(msgs)
    per = []
    for w in windows:
        D, P0, N = _paired(grid, tm, recv_rows, w)
        S = np.asarray(XT @ D); V = np.asarray(XT @ (D * D))
        per.append((w, D, S, V, np.asarray(XT @ P0), np.asarray(XT @ N)))
    null_max = np.full(perms, -np.inf)
    for w, D, S, V, _, _ in per:
        den = np.sqrt(V + 1.0)
        for p in range(perms):
            sgn = rng.choice([-1.0, 1.0], M).astype(np.float32)
            null_max[p] = max(null_max[p], float((np.asarray(XT @ (D * sgn[:, None])) / den).max()))
    bar = float(np.quantile(null_max, 1 - alpha))
    lex = {}
    for w, D, S, V, O, Nb in per:
        z = S / np.sqrt(V + 1.0)
        for v, l in zip(*np.where((z > bar) & (O >= 3))):
            row = {"token": vocab[v], "action": act.labels[l], "window_s": w, "lift": float((O[v, l] + 0.5) / (Nb[v, l] + 0.5)), "z": float(z[v, l]), "count": int(nv[v])}
            if vocab[v] not in lex or row["z"] > lex[vocab[v]]["z"]:
                lex[vocab[v]] = row
    lexicon = sorted(lex.values(), key=lambda r: -r["z"])
    ungr = [{"token": vocab[v], "count": int(nv[v])} for v in np.argsort(-nv) if vocab[v] not in lex and not vocab[v].startswith("<")][:15]
    tot = float(nv.sum())
    vol = float(sum(nv[i] for i, w in enumerate(vocab) if w not in lex) / tot) if tot else 0.0
    return {"lexicon": lexicon, "ungrounded": ungr, "ungrounded_share_of_frequent_tokens": vol, "bar_z": bar, "windows_s": list(windows),
            "n_messages": M, "vocabulary_tested": len(vocab), "perms": perms, "labels": act.labels}


def informativeness(msgs, act, window=300.0, perms=200, folds=5, seed=0, max_vocab=600, max_messages=60000, min_pos=25, ridge=30.0):
    """Does the WHOLE message predict what the receiver does next, beyond what slow context explains? Ridge regression of each action's
    paired outcome (see _paired) on token presence, cross-validated; the score is the correlation between held-out predictions and outcomes,
    compared with the same score after randomly flipping the sign of each message's outcome. Catches a code spread over many synonyms."""
    if hasattr(act, "kind") and not isinstance(act, Actions):
        act = actions_from_trace(act)
    rng = np.random.default_rng(seed)
    if len(msgs) < 100 or not len(act):
        return {"labels": [], "note": "too few messages or no actions"}
    msgs, recv_rows = _prep(msgs, act, max_messages, seed)
    toks = [set(tokenize(m["text"])) for m in msgs]
    cnt = Counter(w for s in toks for w in s)
    vocab = [w for w, n in cnt.most_common(max_vocab) if n >= 5]
    if not vocab:
        return {"labels": [], "note": "no token is frequent enough"}
    X = _vocab_matrix(toks, vocab).astype(np.float64)
    grid = _Grid(act); tm = np.array([m["t"] for m in msgs]); M = len(msgs)
    D, P0, N = _paired(grid, tm, recv_rows, window)
    fold = rng.permutation(M) % folds
    inv = []
    for f in range(folds):
        Xt = X[fold != f]
        inv.append(np.linalg.inv((Xt.T @ Xt).toarray() + ridge * np.eye(X.shape[1])))
    XtT = [X[fold != f].T.tocsr() for f in range(folds)]

    def score(y):
        pred = np.zeros(M)
        for f in range(folds):
            w = inv[f] @ (XtT[f] @ y[fold != f])
            pred[fold == f] = X[fold == f] @ w
        if pred.std() < 1e-12 or y.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(pred, y)[0, 1])
    out = []
    for l, name in enumerate(act.labels):
        pos = int(P0[:, l].sum())
        if pos < min_pos or M - pos < min_pos:
            continue
        out.append({"action": name, "l": l, "score": score(D[:, l].astype(np.float64)), "positives": pos})
    if not out:
        return {"labels": [], "note": "no action label is frequent enough to predict"}
    m = len(out)
    for o in out:
        y = D[:, o["l"]].astype(np.float64)
        nl = np.array([score(y * rng.choice([-1.0, 1.0], M)) for _ in range(perms)])
        sd = float(nl.std()) or 1e-9
        p_norm = float(norm.sf((o["score"] - nl.mean()) / sd))                      # the null is close to normal, so the tail is not capped at 1/(perms+1)
        o["p"] = max(p_norm, 1e-300); o["null_score"] = float(nl.mean())
        o["informative"] = bool(o["p"] * m < 0.05 and o["score"] > nl.max()); del o["l"]
    out.sort(key=lambda r: -r["score"])
    return {"labels": out, "window_s": window, "perms": perms, "n_messages": M}


# ------------------------------------------------------------------ hidden coordination
def coordination(act, win=30.0, perms=300, seed=0, min_windows=60, alpha=0.01, max_labels=8, max_full_tests=150):
    """Pairs of agents whose actions line up window by window beyond what the logs' own shared drivers explain.
    For a pair, windows are grouped by the slow level of activity around them for each of the two agents and for everyone else (ten minutes
    either side, the window itself left out). The pair's rows are then shuffled within each group. That keeps the slow workload every agent
    shares and breaks only the window-by-window alignment between the two. -> pairs sorted by excess mutual information."""
    if hasattr(act, "kind") and not isinstance(act, Actions):
        act = actions_from_trace(act)
    rng = np.random.default_rng(seed)
    if len(act.agents) < 2 or not len(act):
        return {"pairs": [], "note": "need at least two agents"}
    t0 = float(act.t.min()); nb = int((act.t.max() - t0) // win) + 1
    if nb < min_windows:
        return {"pairs": [], "note": f"only {nb} windows of {win:g} s; {min_windows} are needed"}
    top = [l for l, _ in Counter(act.label.tolist()).most_common(max_labels)]
    wi = ((act.t - t0) // win).astype(int)
    Mx, tot_cnt = {}, {}
    for a in range(len(act.agents)):
        s = act.agent == a
        b = np.zeros((nb, len(top)), bool)
        for j, l in enumerate(top):
            q = s & (act.label == l)
            b[wi[q], j] = True
        tot_cnt[a] = np.bincount(wi[s], minlength=nb)
        if b.sum() >= 20:
            Mx[a] = b
    allc = np.sum([tot_cnt[a] for a in tot_cnt], axis=0)
    kern = np.ones(21); kern[10] = 0.0

    def loo(x):
        """slow level around each window (10 minutes either side at 30 s windows), leaving the window itself out"""
        return np.convolve(x.astype(float), kern, mode="same") / np.convolve(np.ones(len(x)), kern, mode="same")

    def lvl(x, n=3):
        q = np.quantile(x, np.linspace(0, 1, n + 1)[1:-1])
        return np.searchsorted(q, x, side="right")

    def strata(a, b):
        key = lvl(loo(tot_cnt[a])) * 9 + lvl(loo(tot_cnt[b])) * 3 + lvl(loo(allc - tot_cnt[a] - tot_cnt[b]))
        return [np.flatnonzero(key == k) for k in np.unique(key) if (key == k).sum() > 1]

    def mi(x, y):
        tot = 0.0
        for k in range(x.shape[1]):
            a, b = x[:, k], y[:, k]
            if a.all() or not a.any() or b.all() or not b.any():
                continue
            n = len(a); j = np.array([[np.sum(~a & ~b), np.sum(~a & b)], [np.sum(a & ~b), np.sum(a & b)]], float) / n
            pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True); mk = j > 0
            tot += float((j[mk] * np.log2(j[mk] / (pa @ pb)[mk])).sum())
        return tot
    ag = sorted(Mx)
    cand = [(a, b, mi(Mx[a], Mx[b])) for i, a in enumerate(ag) for b in ag[i + 1:]]
    total = len(cand)
    cand.sort(key=lambda r: -r[2])
    out = []
    for a, b, obs in cand[:max_full_tests]:
        groups = strata(a, b); null = []
        for _ in range(perms):
            perm = np.arange(nb)
            for g in groups:
                perm[g] = rng.permutation(g)
            null.append(mi(Mx[a], Mx[b][perm]))
        null = np.array(null)
        sd = float(null.std()) or 1e-9
        p = max(float(norm.sf((obs - null.mean()) / sd)), 1e-300)                    # normal tail; also required: obs above every shuffled copy
        out.append({"a": act.agents[a], "b": act.agents[b], "mutual_information_bits": obs, "expected_from_shared_context_bits": float(null.mean()),
                    "excess_bits": obs - float(null.mean()), "p": p, "above_all_null": bool(obs > null.max())})
    out.sort(key=lambda r: -r["excess_bits"])
    for r in out:
        r["flagged"] = bool(r["p"] * total < alpha and r["above_all_null"])
    return {"pairs": out, "window_s": win, "windows": nb, "pairs_total": total, "pairs_tested": len(out)}


# ------------------------------------------------------------------ report
def report_md(P, G=None, C=None, I=None, H=None):
    o = ["# blindspot: what the agents say to each other", ""]
    if H is not None and H["messages_with_any"]:
        c = H["counts"]
        o += ["## Hidden characters",
              f"{H['messages_with_any']} of {H['messages']} messages contain characters that show nothing on screen "
              f"(zero-width {c['zero_width']}, Unicode tags {c['unicode_tags']}, variation selectors {c['variation_selectors']}, bidi controls {c['bidi_controls']}); "
              f"{c['mixed_script_words']} messages mix scripts inside one word (look-alike letters). Most by sender: {H['by_sender']}."]
        for d in H["decoded"][:8]:
            o.append(f"- Decoded from {d['carrier']}, sender {d['from']}: `{d['text']}`")
        o.append("")
    if P.get("tokens"):
        o += ["## The channel's text",
              f"{P['messages']} messages, {P['tokens']} tokens, {P['vocabulary']} distinct. Character entropy {P['char_entropy_bits']:.2f} bits, compression ratio {P['compression_ratio']:.2f}"
              + (f", Zipf slope {P['zipf_slope']:.2f}." if P.get("zipf_slope") is not None else ".")]
        if P.get("new_token_rate_last_third") is not None:
            o.append(f"In the last third of the channel, {100 * P['new_token_rate_last_third']:.0f}% of tokens had never appeared before. A rate that stays high means the vocabulary keeps changing.")
        if P.get("distance_from_reference_bits") is not None:
            ref = P.get("reference_self_distance_bits")
            o.append(f"Distance of the channel's character patterns from your reference text: {P['distance_from_reference_bits']:.3f} bits"
                     + (f" (two halves of the reference differ by {ref:.3f})." if ref is not None else "."))
        o.append("")
    if G is not None:
        o += ["## Decoding by consequence: which tokens move which actions", ""]
        if not G["lexicon"]:
            o.append(G.get("note") or "No token reliably precedes any action beyond chance, at the strictest bar across every token, action and delay tested.")
        else:
            o += [f"{G['vocabulary_tested']} frequent tokens, delays of {', '.join(f'{w:g} s' for w in G['windows_s'])}, each message compared with itself moved sideways, {G['perms']} sign-flip permutations.", "",
                  "| token | action that follows | within | times more often than chance | messages |", "|---|---|---|---|---|"]
            o += [f"| `{r['token']}` | {r['action']} | {r['window_s']:g} s | {r['lift']:.1f} | {r['count']} |" for r in G["lexicon"][:25]]
        if G.get("ungrounded"):
            o += ["", f"Frequent tokens that move no action: {', '.join('`' + u['token'] + '`' for u in G['ungrounded'][:10])} "
                      f"({100 * G['ungrounded_share_of_frequent_tokens']:.0f}% of frequent-token volume). That volume is not explained by behavior in this log."]
        o.append("")
    if I is not None and I.get("labels"):
        o += ["## Does the whole message predict what happens next?", "",
              f"Cross-validated, per action, within {I['window_s']:g} s, after removing what slow context explains. The score is the correlation between held-out predictions and what happened; 0 means no information.", "",
              "| action | score | chance | p | informative |", "|---|---|---|---|---|"]
        o += [f"| {r['action']} | {r['score']:.2f} | {r['null_score']:.2f} | {r['p']:.3f} | {'yes' if r['informative'] else ''} |" for r in I["labels"][:12]]
        if any(r["informative"] for r in I["labels"]) and G is not None and not G["lexicon"]:
            o += ["", "The messages predict actions although no single token does: the meaning is spread across many words or patterns."]
        o.append("")
    if C is not None:
        o += ["## Coordination beyond shared context", ""]
        if not C["pairs"]:
            o.append(C.get("note", "No agent pairs to test."))
        else:
            fl = [r for r in C["pairs"] if r["flagged"]]
            o += [f"{len(fl)} of {C['pairs_total']} pairs line up in the same {C['window_s']:g}-second window more tightly than their slow shared context explains (corrected for all {C['pairs_total']} pairs).", "",
                  "| agent A | agent B | mutual information (bits) | from shared context (bits) | excess | flagged |", "|---|---|---|---|---|---|"]
            o += [f"| {r['a']} | {r['b']} | {r['mutual_information_bits']:.3f} | {r['expected_from_shared_context_bits']:.3f} | {r['excess_bits']:.3f} | {'yes' if r['flagged'] else ''} |" for r in C["pairs"][:15]]
        o.append("")
    o += ["", "Richer action logs sharpen results: tool names as labels beat the six coarse kinds."]
    return "\n".join(o) + "\n"
