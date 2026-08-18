#!/usr/bin/env python3
"""
LongMemEval matrix runner: several retrieval modes per question, one index.

The published single-mode runner (longmemeval_bench.py) re-indexes the whole
haystack for every run, so an N-mode comparison costs N full runs and the
indexing work (which dominates wall clock) is repeated identically each time.
This runner indexes once per question and evaluates every requested mode
against that index, which makes an embedder x reranker matrix affordable.

The published runner is left untouched on purpose: it produced the numbers in
docs/, and those stay reproducible.

Usage:
    MNEMOS_EMBED_MODEL=BAAI/bge-small-en-v1.5 MNEMOS_EMBED_DIMS=384 \
    python longmemeval_matrix.py --modes hybrid,hybrid+rerank --out bge-small.json
"""

import argparse, json, os, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import longmemeval_bench as lmb
from mnemos.constants import FASTEMBED_MODEL, FASTEMBED_DIMS

MODES = ("fts", "vec", "hybrid", "hybrid+rerank")


def retrieve(conn, mode, question, query_emb):
    if mode == "fts":
        return lmb.search_fts(conn, question, limit=50)
    if not query_emb:
        return lmb.search_fts(conn, question, limit=50)
    if mode == "vec":
        return lmb.search_vec(conn, query_emb, limit=50)
    if mode == "hybrid":
        return lmb.search_hybrid(conn, question, query_emb, limit=50)
    if mode == "hybrid+rerank":
        return lmb.search_hybrid_rerank(conn, question, query_emb, limit=50)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--modes", default="hybrid,hybrid+rerank")
    ap.add_argument("--granularity", default="session")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tempdb", default=None, help="override temp index path (for concurrent runs)")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            sys.exit(f"unknown mode {m}; pick from {MODES}")
    if args.tempdb:
        lmb.TEMP_DB = args.tempdb

    with open(lmb.DATA_PATH) as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]

    print(f"model={FASTEMBED_MODEL} dims={FASTEMBED_DIMS} modes={modes} questions={len(data)}", flush=True)
    lmb.fastembed_embed(["warmup"], prefix="search_query")
    conn = lmb.create_temp_db()

    ks = [1, 3, 5, 10]
    recall = {m: {k: [] for k in ks} for m in modes}
    by_type = {m: defaultdict(lambda: {k: [] for k in ks}) for m in modes}
    t_start = time.time()
    scored = 0

    for i, entry in enumerate(data):
        if "_abs" in entry["question_id"]:
            continue
        lmb.reset_db(conn)
        corpus = lmb.build_corpus(entry, args.granularity)
        if not corpus:
            continue
        embs = lmb.fastembed_embed([d["text"] for d in corpus], prefix="passage")
        lmb.ingest_corpus(conn, corpus, embs)
        qe = lmb.fastembed_embed([entry["question"]], prefix="search_query")
        qe = qe[0] if qe else None

        for m in modes:
            sessions = lmb.id_to_session(conn, retrieve(conn, m, entry["question"], qe))
            for k in ks:
                r = lmb.recall_at_k(sessions, entry["answer_session_ids"], k)
                recall[m][k].append(r)
                by_type[m][entry["question_type"]][k].append(r)
        scored += 1

        if scored % 25 == 0:
            el = time.time() - t_start
            print(f"  {scored} scored | {el/scored:.1f}s/q | "
                  + " | ".join(f"{m} R@5={sum(recall[m][5])/len(recall[m][5]):.1%}" for m in modes),
                  flush=True)

    out = {"model": FASTEMBED_MODEL, "dims": FASTEMBED_DIMS, "questions": scored,
           "elapsed_s": round(time.time() - t_start, 1), "modes": {}}
    print(f"\n{'='*72}\nmodel={FASTEMBED_MODEL} dims={FASTEMBED_DIMS} n={scored}")
    for m in modes:
        agg = {f"R@{k}": round(100 * sum(recall[m][k]) / max(len(recall[m][k]), 1), 2) for k in ks}
        types = {t: {f"R@{k}": round(100 * sum(v[k]) / max(len(v[k]), 1), 2) for k in ks}
                 for t, v in by_type[m].items()}
        out["modes"][m] = {"overall": agg, "by_type": types,
                           "n_by_type": {t: len(v[1]) for t, v in by_type[m].items()}}
        print(f"\n  {m:<16} " + "  ".join(f"{key}={val:.2f}%" for key, val in agg.items()))
        for t in sorted(types):
            print(f"    {t:<28} " + "  ".join(f"{key}={val:6.2f}%" for key, val in types[t].items()))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
