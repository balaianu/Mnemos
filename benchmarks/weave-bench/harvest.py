#!/usr/bin/env python3
"""Harvest weave-labeled memory pairs from prod memory.db (read-only).

Eval set for the question: can the NLI layer replace the LLM in Phase 3
weave link decisions (link/no-link gate + relation typing)?

Sources:
  1. All typed weave links: relation_type in {evolves, informs, contradicts,
     enables, reflects}. These carry silver labels assigned by the weave LLM
     (Haiku as of 2026-07). Silver, not gold: disagreement is not
     automatically an NLI error.
  2. Negative pairs: random unlinked active pairs inside the weave candidate
     cosine band [WEAVE_MIN_SIMILARITY, NEG_MAX_SIM], both non-nyx-generated,
     labeled "none". Approximates pairs weave would evaluate but reject
     (rejected pairs are not persisted, so this is the closest observable
     population).

Output: pairs.jsonl, one pair per line with contents, created_at, silver
label and silver strength.

Usage: /root/venvs/ai/bin/python harvest.py [--db /root/work/memory.db]
Deps: numpy, sqlite_vec (venv ai). DB opened mode=ro.
"""

import argparse
import json
import random
import sqlite3
import struct

WEAVE_TYPES = ("evolves", "informs", "contradicts", "enables", "reflects")
NEG_TARGET = 150
NEG_MIN_SIM = 0.55   # WEAVE_MIN_SIMILARITY default
NEG_MAX_SIM = 0.85
MAX_CHARS = 4000
SEED = 42


def open_ro(path):
    import sqlite_vec
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    return conn


def is_nyx_generated(tags):
    tags = (tags or "").lower()
    return any(t in tags for t in ("synthesized", "nyx-cycle", "bridge"))


def load_memory(conn, mid):
    r = conn.execute(
        "SELECT id, content, tags, status, created_at FROM memories WHERE id=?",
        (mid,),
    ).fetchone()
    return dict(r) if r else None


def harvest_positives(conn):
    rows = conn.execute(
        "SELECT source_id, target_id, relation_type, strength, created_at "
        "FROM memory_links WHERE relation_type IN ({})".format(
            ",".join("?" * len(WEAVE_TYPES))),
        WEAVE_TYPES,
    ).fetchall()
    pairs, skipped = [], 0
    for r in rows:
        a = load_memory(conn, r["source_id"])
        b = load_memory(conn, r["target_id"])
        if not a or not b or not a["content"] or not b["content"]:
            skipped += 1
            continue
        pairs.append({
            "a_id": a["id"], "b_id": b["id"],
            "a_content": a["content"][:MAX_CHARS],
            "b_content": b["content"][:MAX_CHARS],
            "a_created": a["created_at"], "b_created": b["created_at"],
            "label": r["relation_type"],
            "silver_strength": r["strength"],
            "source": "weave-link",
        })
    return pairs, skipped


def load_vectors(conn):
    """Active, non-nyx memories with vectors. Returns {id: normalized vec}."""
    import numpy as np
    metas = conn.execute(
        "SELECT m.id AS meta_id, m.source_id, mem.tags FROM embed_meta m "
        "JOIN memories mem ON mem.id = m.source_id "
        "WHERE m.source_db='memory' AND mem.status='active'"
    ).fetchall()
    # embed_vec join col detection, same dual-schema reality as load_embeddings
    cols = {r[1] for r in conn.execute("PRAGMA table_info(embed_vec)")}
    join_col = "id" if "id" in cols else "rowid"
    vecs = {}
    for r in metas:
        if is_nyx_generated(r["tags"]):
            continue
        row = conn.execute(
            f"SELECT embedding FROM embed_vec WHERE {join_col}=?",
            (r["meta_id"],),
        ).fetchone()
        if not row or not row[0]:
            continue
        n = len(row[0]) // 4
        v = np.array(struct.unpack(f"{n}f", row[0]), dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            vecs[r["source_id"]] = v / norm
    return vecs


def harvest_negatives(conn, exclude_pairs):
    import numpy as np
    vecs = load_vectors(conn)
    linked = set()
    for r in conn.execute("SELECT source_id, target_id FROM memory_links"):
        linked.add((r[0], r[1]))
        linked.add((r[1], r[0]))
    ids = sorted(vecs.keys())
    rng = random.Random(SEED)
    negatives, seen, attempts = [], set(), 0
    while len(negatives) < NEG_TARGET and attempts < 200_000:
        attempts += 1
        a_id, b_id = rng.sample(ids, 2)
        key = (min(a_id, b_id), max(a_id, b_id))
        if key in seen or (a_id, b_id) in linked or key in exclude_pairs:
            continue
        cos = float(np.dot(vecs[a_id], vecs[b_id]))
        if not (NEG_MIN_SIM <= cos <= NEG_MAX_SIM):
            continue
        seen.add(key)
        a = load_memory(conn, a_id)
        b = load_memory(conn, b_id)
        if not a or not b:
            continue
        negatives.append({
            "a_id": a_id, "b_id": b_id,
            "a_content": a["content"][:MAX_CHARS],
            "b_content": b["content"][:MAX_CHARS],
            "a_created": a["created_at"], "b_created": b["created_at"],
            "label": "none",
            "silver_strength": None,
            "cosine": round(cos, 4),
            "source": f"unlinked-band-{NEG_MIN_SIM}-{NEG_MAX_SIM}",
        })
    return negatives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/root/work/memory.db")
    ap.add_argument("--out", default="pairs.jsonl")
    args = ap.parse_args()

    conn = open_ro(args.db)
    positives, skipped = harvest_positives(conn)
    exclude = {(min(p["a_id"], p["b_id"]), max(p["a_id"], p["b_id"]))
               for p in positives}
    negatives = harvest_negatives(conn, exclude)

    with open(args.out, "w") as f:
        for p in positives + negatives:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    by_label = {}
    for p in positives + negatives:
        by_label[p["label"]] = by_label.get(p["label"], 0) + 1
    print(f"positives: {len(positives)} (skipped {skipped} with missing content)")
    print(f"negatives: {len(negatives)}")
    print(f"labels: {json.dumps(by_label, sort_keys=True)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
