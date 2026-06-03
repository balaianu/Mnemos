# Roadmap

Loose, opinionated, subject to change. Not a release schedule. Items move
to CHANGELOG.md when shipped.

---

## v10.5 candidates

### Parallelize Phase 0.5 (Cemelify)

**Status**: identified 2026-04-18, not started.

Phase 0.5 currently iterates candidates serially: one Haiku call per
memory, ~6-8s wall time each. On a 750-candidate Nyx run that's 75-100
minutes for cemelify alone, blocking Phase 2A dedup behind it.

The standalone `cml_convert_batch` helper used during the v10.4.0 data
fix processed 457 memories in 5 min 41 s using 8 parallel workers - same
LLM, same prompt, same content shape. Order-of-magnitude speedup.

Implementation sketch:
- Wrap the Phase 0.5 loop in `concurrent.futures.ThreadPoolExecutor` with
  configurable `max_workers` (env: `MNEMOS_CEMELIFY_WORKERS`, default 8)
- Preserve the in-memory `mem_by_id[mid]["content"]` update pattern so
  downstream phases see cemelified content within the same run
- Keep the per-100 progress log; add a per-worker concurrency note
- Stay within rate limits: 8 workers is comfortable for OpenAI/Gradient
  defaults; users on tighter limits can dial down via env

Risk: thread safety of the SQLite write path inside `store.update_memory`
needs verification - SQLite handles concurrent writes via WAL but the
SQLAlchemy-style connection pool in `SQLiteStore` may not. Worst case:
parallelize the LLM calls only, batch the writes serially after.

---

## Other candidates (lower priority)

- **Per-phase fast/slow budget split.** `MNEMOS_LLM_BUDGET` env to cap
  total Haiku/gpt-4o-mini spend per cycle, fail loud at limit instead of
  silently hammering the API.
- **Phase 0.5 sample preview** in dry-run mode: show diff of 5 random
  candidates' cemelified output before applying, so users can eyeball
  quality without committing.
- **Cemelify-on-import in batch.** Currently `MNEMOS_CEMELIFY_ON_IMPORT`
  is per-memory; `mnemos ingest` could batch-cemelify chunks in parallel
  for the same speedup story as Phase 0.5.
- **PyPI publication.** README still installs via `git clone`. Once API
  surface stabilizes, ship to pypi.org for `pip install mnemos`.

---

Last updated: 2026-04-18
