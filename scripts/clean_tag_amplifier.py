#!/usr/bin/env python3
"""Repair tag-amplifier damage in a Mnemos store written before v10.35.2.

The bug (fixed in 10.35.2): Nyx merge unioned every source's tags unbounded,
and the size-guard split copied that full union to every sibling -- a merge of
N memories split into k pieces produced k memories each claiming all N topics
while holding 1/k of the content, compounding nightly as unions merged with
unions. Field-measured worst case: a 275-char sibling carrying 6KB of tags.
Because prep_memory_text embeds retrieval-relevant tags, the unions also
poisoned the affected vectors.

This script applies the 10.35.2 rule retroactively to rows that carry Nyx
bookkeeping markers (only those -- hand-authored tags are deliberate and are
never touched): a tag survives when the row's own content supports it, when
it is bookkeeping/provenance (split-part, merged-into, src:, ...), or when it
is a protected marker (STRICTLY-PRIVATE, authoritative). Cluster-universality
cannot be recomputed post-hoc, so protected markers stand in for it.

Diagnose first (plain sqlite3, any machine):
    SELECT COUNT(*) FROM memories
    WHERE status='active' AND length(tags) > 1000;

Usage:
    python scripts/clean_tag_amplifier.py --db ~/work/memory.db          # dry run
    python scripts/clean_tag_amplifier.py --db ~/work/memory.db --apply

--apply writes a timestamped .bak sibling of the DB first. Tags are part of
the embedded text, so RE-EMBED after applying: `mnemos reembed`, or your
deployment's embed-sync if it detects prep-text hash drift.
"""
import argparse
import os
import shutil
import sqlite3
import sys
import time

try:
    from mnemos.consolidation.phases import _tag_supported
except ImportError:
    sys.exit("mnemos >= 10.35.2 must be importable (pip install -e . in the repo)")

BOOKKEEPING_PREFIX = ("merged-into", "split-from", "split-part", "nyx-",
                      "topic:", "src:", "giant-sort", "topic-sort", "imported-")
BOOKKEEPING_EXACT = {"consolidated", "nyx-split", "nyx-cycle", "synthesized",
                     "bridge"}
PROTECTED = {"strictly-private", "authoritative"}


def is_bookkeeping(tag):
    low = tag.lower()
    return low in BOOKKEEPING_EXACT or any(low.startswith(p) for p in BOOKKEEPING_PREFIX)


def is_nyx_product(tags):
    """Provenance tags (src:, imported-) also appear on hand-ingested rows;
    only actual consolidation markers qualify a row for filtering."""
    return any(is_bookkeeping(t) and not t.lower().startswith(("src:", "imported-"))
               for t in tags)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True, help="path to the store (memory.db)")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default: dry run, print only)")
    args = ap.parse_args()

    db = os.path.expanduser(args.db)
    if not os.path.exists(db):
        sys.exit(f"no such file: {db}")

    if args.apply:
        bak = f"{db}.bak-tagclean-{time.strftime('%Y%m%d-%H%M%S')}"
        src = sqlite3.connect(db)
        src.execute(f"VACUUM INTO ?", (bak,))
        src.close()
        print(f"backup: {bak}")

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, content, tags FROM memories "
        "WHERE status='active' AND tags IS NOT NULL AND tags != ''"
    ).fetchall()

    changed, saved = 0, 0
    worst = []
    for r in rows:
        tags = [t.strip() for t in r["tags"].split(",") if t.strip()]
        if not is_nyx_product(tags):
            continue
        cl = r["content"].lower()
        kept = [t for t in tags
                if is_bookkeeping(t) or t.lower() in PROTECTED
                or _tag_supported(t, cl)]
        if kept == tags:
            continue
        changed += 1
        delta = len(r["tags"]) - len(",".join(kept))
        saved += delta
        worst.append((delta, r["id"], len(tags), len(kept)))
        if args.apply:
            conn.execute("UPDATE memories SET tags=? WHERE id=?",
                         (",".join(kept), r["id"]))

    if args.apply:
        conn.commit()
    worst.sort(reverse=True)
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"{mode}: {len(rows)} active rows scanned, {changed} rows "
          f"{'cleaned' if args.apply else 'would change'}, "
          f"{saved} tag chars removed")
    for delta, mid, before, after in worst[:10]:
        print(f"  id {mid}: {before} tags -> {after} (-{delta} chars)")
    if args.apply and changed:
        print("now re-embed: tags are part of the embedded text "
              "(`mnemos reembed`, or your embed-sync)")


if __name__ == "__main__":
    main()
