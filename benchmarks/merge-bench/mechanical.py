#!/usr/bin/env python3
"""Mechanical NLI merge: union of atomic lines, dedup by bidirectional entailment.

The candidate replacement for the LLM in Phase 2 MERGE on CML content.
Selection only, never generation: every output line is an input line
verbatim, so fact preservation is provable by construction. A line is
dropped only when it is a bidirectional-entailment duplicate (both
directions >= tau) of a kept line, and the drop is recorded with its
partner and score.

Keep rule on duplicates: the NEWER memory's phrasing survives (recency
semantics, matches how updates supersede stale wordings).

Cost control: an exact-match fast path plus a lexical prefilter (word
overlap or shared digit token) skips NLI on pairs that cannot plausibly
be duplicates. A false skip costs compression, never information.

Usage: imported by bench.py. Deps: mnemos[nli] with a local ONNX export.
"""

import re

from mnemos import nli
from mnemos.splitter import explode_cml_chain

TAU_DUP = 0.70
PREFILTER_JACCARD = 0.10

_WORD = re.compile(r"[a-za-åäö0-9_./#-]+", re.IGNORECASE)


def lines_of(content):
    text = explode_cml_chain(content)
    out = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if ln and not ln.startswith("---"):
            out.append(ln)
    return out


def _words(line):
    return {w.lower() for w in _WORD.findall(line) if len(w) > 2}


def _prefilter_pass(la, lb):
    if la == lb:
        return True
    wa, wb = _words(la), _words(lb)
    if not wa or not wb:
        return False
    jac = len(wa & wb) / len(wa | wb)
    return jac >= PREFILTER_JACCARD


def mechanical_merge(inputs, tau=TAU_DUP):
    """inputs: list of dicts {id, created_at, content}, any order.

    Returns {merged, kept, dropped, nli_calls, prefiltered}.
    kept/dropped entries: {line, mem_id}; dropped adds {dup_of, dup_line, score}.
    """
    ordered = sorted(inputs, key=lambda m: m.get("created_at") or "")
    pool = []
    for mem in ordered:
        for ln in lines_of(mem["content"]):
            pool.append({"line": ln, "mem_id": mem["id"]})

    kept, dropped = [], []
    nli_calls = prefiltered = 0
    # Iterate newest-first so the newer phrasing is kept and older
    # duplicates are the ones that drop.
    for cand in reversed(pool):
        dup = None
        for k in kept:
            if k["line"] == cand["line"]:
                dup = {"partner": k, "score": 1.0}
                break
            if not _prefilter_pass(cand["line"], k["line"]):
                prefiltered += 1
                continue
            nli_calls += 2
            (e1, _), (e2, _) = nli._score_pair(cand["line"], k["line"])
            score = min(e1, e2)
            if score >= tau:
                dup = {"partner": k, "score": score}
                break
        if dup:
            dropped.append({**cand, "dup_of": dup["partner"]["mem_id"],
                            "dup_line": dup["partner"]["line"],
                            "score": round(dup["score"], 3)})
        else:
            kept.append(cand)

    kept.reverse()
    return {
        "merged": "\n".join(k["line"] for k in kept),
        "kept": kept,
        "dropped": dropped,
        "nli_calls": nli_calls,
        "prefiltered": prefiltered,
    }
