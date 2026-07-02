#!/usr/bin/env python3
"""Harvest labeled-candidate memory pairs from prod memory.db (read-only).

Sources:
  1. All memory_links of types contradicts/relates/evolves (few, high value)
  2. A sample of 'related' links (expected label: related-distinct; these are
     the false-positive test set for both dedup and contradiction)
  3. Known dedup false-block: #6127 vs #5822 (2026-07-02, vec 73% block)
  4. High-cosine (>=0.80) active pairs with NO link (dedup FP candidates)

Output: pairs.jsonl with one candidate pair per line. Labels are assigned
separately (labels.json) by a human-grade reader, not by this script.

Usage: /root/venvs/ai/bin/python harvest.py [--db /root/work/memory.db]
Deps: numpy (via fastembed), sqlite3 stdlib. DB opened mode=ro.
"""

import argparse
import json
import sqlite3
import struct
import sys

KNOWN_CASES = [(6127, 5822, "known-false-block")]
RELATED_SAMPLE = 25
HIGHSIM_THRESHOLD = 0.80
HIGHSIM_SAMPLE = 20
MAX_CHARS = 4000


def row_content(conn, mid):
    r = conn.execute(
        "SELECT id, content, project, status, created_at FROM memories WHERE id=?",
        (mid,),
    ).fetchone()
    return dict(r) if r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/root/work/memory.db")
    ap.add_argument("--out", default="pairs.jsonl")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception as e:
        print(f"sqlite_vec not loaded ({e}); highsim will be skipped", file=sys.stderr)

    pairs = []
    seen = set()

    def add(a_id, b_id, source):
        key = (min(a_id, b_id), max(a_id, b_id))
        if key in seen:
            return
        a, b = row_content(conn, a_id), row_content(conn, b_id)
        if not a or not b or not a["content"] or not b["content"]:
            return
        seen.add(key)
        pairs.append({
            "pair_id": f"p{len(pairs):03d}",
            "a_id": a_id, "b_id": b_id, "source": source,
            "a_project": a["project"], "b_project": b["project"],
            "a_status": a["status"], "b_status": b["status"],
            "a_content": a["content"][:MAX_CHARS],
            "b_content": b["content"][:MAX_CHARS],
        })

    # 1. High-value link types (all of them)
    for lt in ("contradicts", "relates", "evolves", "informs"):
        for r in conn.execute(
            "SELECT source_id, target_id FROM memory_links WHERE relation_type=?", (lt,)
        ):
            add(r["source_id"], r["target_id"], f"link:{lt}")

    # 2. Sample of 'related' links, deterministic spread
    rel = conn.execute(
        "SELECT source_id, target_id FROM memory_links WHERE relation_type='related' "
        "ORDER BY id"
    ).fetchall()
    step = max(1, len(rel) // RELATED_SAMPLE)
    for r in rel[::step][:RELATED_SAMPLE]:
        add(r["source_id"], r["target_id"], "link:related")

    # 3. Known cases
    for a, b, src in KNOWN_CASES:
        add(a, b, src)

    # 4. High-cosine unlinked active pairs
    try:
        import numpy as np
        try:
            conn.execute("SELECT id FROM embed_vec LIMIT 0")
            join_col = "id"
        except sqlite3.OperationalError:
            join_col = "rowid"
        rows = conn.execute(
            f"SELECT em.source_id AS mid, ev.embedding AS emb FROM embed_meta em "
            f"JOIN embed_vec ev ON ev.{join_col} = em.id "
            "JOIN memories m ON m.id = em.source_id "
            "WHERE em.source_db='memory' AND m.status='active'"
        ).fetchall()
        ids = [r["mid"] for r in rows]
        mat = np.array([struct.unpack(f"{len(r['emb'])//4}f", r["emb"]) for r in rows],
                       dtype=np.float32)
        mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = mat @ mat.T
        np.fill_diagonal(sims, 0)
        linked = set()
        for r in conn.execute("SELECT source_id, target_id FROM memory_links"):
            linked.add((min(r["source_id"], r["target_id"]),
                        max(r["source_id"], r["target_id"])))
        cand = []
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                if sims[i, j] >= HIGHSIM_THRESHOLD:
                    key = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                    if key not in linked:
                        cand.append((float(sims[i, j]), ids[i], ids[j]))
        cand.sort(reverse=True)
        step = max(1, len(cand) // HIGHSIM_SAMPLE)
        for sim, a, b in cand[::step][:HIGHSIM_SAMPLE]:
            add(a, b, f"highsim:{sim:.3f}")
        print(f"highsim candidates total: {len(cand)}", file=sys.stderr)
    except Exception as e:
        print(f"highsim harvest skipped: {e}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
