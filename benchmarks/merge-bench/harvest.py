#!/usr/bin/env python3
"""Harvest eval data for the mechanical-merge bench (pass 2 after weave-bench).

Three arms:
  A. Synthetic overlap splits: real multi-line CML memories split into two
     halves sharing ~1/3 of their lines. Ground truth is exact (the original
     line set), so Arm A proves the union algorithm and catches
     over-aggressive tau (a dropped unique line = NLI false duplicate).
  B. Production merge candidates: high-cosine active non-nyx pairs. The
     bench gates them (line-level shared-fact >= 0.70, the validated
     phase-2 gate) and mechanically merges the passers.
  C. The two real clusters from the 2026-07-03 03:00 Nyx run, with the
     production Sonnet+splitter outputs they produced, for a head-to-head
     fact-coverage comparison on identical inputs.

Output: arms.json. Usage: /root/venvs/ai/bin/python harvest.py
Deps: numpy, sqlite_vec (venv ai). DB opened mode=ro.
"""

import argparse
import json
import random
import sqlite3
import struct

from mnemos.splitter import explode_cml_chain

SYNTH_TARGET = 25
SYNTH_MIN_LINES = 6
HIGHSIM_MIN = 0.78
HIGHSIM_CAP = 120
MAX_CHARS = 4000
SEED = 7

# 2026-07-03 03:00 run provenance (nyx log + created_at window 03:02-03:07)
CLUSTER_CASES = {
    "cluster1": {"inputs": [5815, 5834, 5835, 6126, 6135],
                 "prod_outputs": [6137, 6138, 6139, 6140, 6141]},
    "cluster2": {"inputs": [5859, 5942, 5944, 6034, 6041, 6042, 6063, 6127],
                 "prod_outputs": [6142, 6143, 6144, 6145, 6146]},
}


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


def get_mem(conn, mid):
    r = conn.execute(
        "SELECT id, content, tags, created_at FROM memories WHERE id=?",
        (mid,)).fetchone()
    return dict(r) if r else None


def lines_of(content):
    text = explode_cml_chain(content)
    return [ln.strip() for ln in text.split("\n")
            if ln.strip() and not ln.strip().startswith("---")]


def harvest_synthetic(conn):
    rng = random.Random(SEED)
    rows = conn.execute(
        "SELECT id, content, tags, created_at FROM memories "
        "WHERE status='active' AND length(content) > 400 ORDER BY id").fetchall()
    cases = []
    for r in rows:
        if is_nyx_generated(r["tags"]):
            continue
        lines = lines_of(r["content"])
        if len(lines) < SYNTH_MIN_LINES:
            continue
        n = len(lines)
        overlap = max(2, n // 3)
        cut = (n + overlap) // 2
        a_lines = lines[:cut]
        b_lines = lines[cut - overlap:]
        cases.append({
            "source_id": r["id"],
            "truth_lines": lines,
            "a": {"id": r["id"] * 10 + 1, "created_at": r["created_at"],
                  "content": "\n".join(a_lines)},
            "b": {"id": r["id"] * 10 + 2, "created_at": r["created_at"],
                  "content": "\n".join(b_lines)},
            "overlap": overlap,
        })
        if len(cases) >= SYNTH_TARGET:
            break
    return cases


def load_vectors(conn):
    import numpy as np
    metas = conn.execute(
        "SELECT m.id AS meta_id, m.source_id, mem.tags FROM embed_meta m "
        "JOIN memories mem ON mem.id = m.source_id "
        "WHERE m.source_db='memory' AND mem.status='active'").fetchall()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(embed_vec)")}
    join_col = "id" if "id" in cols else "rowid"
    vecs = {}
    for r in metas:
        if is_nyx_generated(r["tags"]):
            continue
        row = conn.execute(
            f"SELECT embedding FROM embed_vec WHERE {join_col}=?",
            (r["meta_id"],)).fetchone()
        if not row or not row[0]:
            continue
        n = len(row[0]) // 4
        v = np.array(struct.unpack(f"{n}f", row[0]), dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            vecs[r["source_id"]] = v / norm
    return vecs


def harvest_highsim(conn):
    import numpy as np
    vecs = load_vectors(conn)
    ids = sorted(vecs.keys())
    mat = np.stack([vecs[i] for i in ids])
    sims = mat @ mat.T
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sims[i, j] >= HIGHSIM_MIN:
                pairs.append((float(sims[i, j]), ids[i], ids[j]))
    pairs.sort(reverse=True)
    out = []
    for cos, a_id, b_id in pairs[:HIGHSIM_CAP]:
        a, b = get_mem(conn, a_id), get_mem(conn, b_id)
        if not a or not b:
            continue
        out.append({"cosine": round(cos, 4),
                    "a": {"id": a["id"], "created_at": a["created_at"],
                          "content": a["content"][:MAX_CHARS]},
                    "b": {"id": b["id"], "created_at": b["created_at"],
                          "content": b["content"][:MAX_CHARS]}})
    return out, len(pairs)


def harvest_clusters(conn):
    cases = {}
    for name, spec in CLUSTER_CASES.items():
        inputs = [get_mem(conn, mid) for mid in spec["inputs"]]
        outputs = [get_mem(conn, mid) for mid in spec["prod_outputs"]]
        cases[name] = {
            "inputs": [{"id": m["id"], "created_at": m["created_at"],
                        "content": m["content"]} for m in inputs if m],
            "prod_output_text": "\n".join(m["content"] for m in outputs if m),
            "prod_output_ids": spec["prod_outputs"],
        }
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/root/work/memory.db")
    ap.add_argument("--out", default="arms.json")
    args = ap.parse_args()
    conn = open_ro(args.db)

    synth = harvest_synthetic(conn)
    highsim, total_band = harvest_highsim(conn)
    clusters = harvest_clusters(conn)

    with open(args.out, "w") as f:
        json.dump({"synthetic": synth, "highsim": highsim,
                   "clusters": clusters}, f, ensure_ascii=False, indent=1)
    print(f"arm A synthetic: {len(synth)} split pairs")
    print(f"arm B highsim: {len(highsim)} pairs (of {total_band} >= {HIGHSIM_MIN})")
    print(f"arm C clusters: {list(clusters)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
