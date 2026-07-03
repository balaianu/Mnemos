#!/usr/bin/env python3
"""Replay Phase 2 tight clusters through the proposed NLI membership gate.

Validates the 10.17 candidate fix: before merging a tight cluster, require
each member to share at least one line-level duplicate fact (bidirectional
entailment) with another member; eject members that share none. A cosine
cluster of topically-unrelated multi-fact memories should mostly dissolve.

Default clusters are the two from the 2026-07-03 03:00 Nyx run. Cluster 1
grouped five unrelated server memories at tight threshold 0.88 (secrets
convention, rustdesk version, torch policy, nyx-cloud decision, prod install
fix); the Sonnet merge + splitter recovered atomicity downstream, but the
cluster itself was noise and the gate should show that upstream.

Usage: /root/venvs/ai/bin/python gate_replay.py [--db /root/work/memory.db]
Deps: mnemos[nli] with a local ONNX export (venv ai).
"""

import argparse
import json
import sqlite3

from mnemos import nli

CLUSTERS = {
    "cluster1-20260703": [5815, 5834, 5835, 6126, 6135],
    "cluster2-20260703": [5859, 5942, 5944, 6034, 6041, 6042, 6063, 6127],
}
TH_DUP = 0.70    # line-level bidirectional entailment = shared fact
MAX_LINES = 8


def lines_of(text):
    out = [ln.strip() for ln in text.split("\n") if ln.strip() and ln.strip() != "---"]
    return out[:MAX_LINES]


def max_line_duplicate(a_text, b_text):
    """Max over line pairs of min-direction P(entailment)."""
    best = 0.0
    for la in lines_of(a_text):
        for lb in lines_of(b_text):
            e = nli.bidirectional_entailment(la, lb)
            if e is not None and e > best:
                best = e
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/root/work/memory.db")
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    report = {}
    for name, ids in CLUSTERS.items():
        contents = {}
        for mid in ids:
            row = conn.execute(
                "SELECT content FROM memories WHERE id=?", (mid,)).fetchone()
            if row:
                contents[mid] = row[0]
        pair_scores = {}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if a in contents and b in contents:
                    s = max_line_duplicate(contents[a], contents[b])
                    pair_scores[f"{a}-{b}"] = round(s, 3)
        keeps, ejects = [], []
        for mid in ids:
            linked = any(
                s >= TH_DUP for k, s in pair_scores.items()
                if str(mid) in k.split("-"))
            (keeps if linked else ejects).append(mid)
        report[name] = {
            "members": ids,
            "pair_max_line_duplicate": pair_scores,
            "gate_keeps": keeps,
            "gate_ejects": ejects,
            "verdict": ("cluster dissolves" if len(keeps) < 2 else
                        f"merge shrinks to {len(keeps)} members"),
        }
        print(f"{name}: keeps={keeps} ejects={ejects}", flush=True)

    with open("gate_replay_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("wrote gate_replay_results.json")


if __name__ == "__main__":
    main()
