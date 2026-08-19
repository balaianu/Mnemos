# Roadmap

Loose, opinionated, subject to change. Not a release schedule. Items move
to CHANGELOG.md when shipped.

---

## Near-term candidates

- **Per-symbol CML operator mapping, resolved from the tokenizer probe.**
  The v2 map (10.33.0) is uniform across models, chosen to be safe for the
  worst tokenizer benchmarked; blindness is per-model (WordPiece drops nine
  operators to [UNK], byte-BPE shreds a different set into byte pieces,
  SentencePiece carries all of them weakly). The load-time probe already
  answers "does this model represent X" per symbol and today aggregates it
  to one boolean. Resolving the map per symbol would give each model exactly
  the substitutions it needs, no more, and would let a future model keep an
  operator native where measurement says native wins. Precondition, and the
  reason this is not built yet: per-model embed text means the RESOLVED map
  must be hashed into embed provenance, since a version number cannot
  distinguish two models' differing per-symbol verdicts. Build it the first
  time a real model shows up that the uniform map mis-serves; until then the
  uniform map measurably lands every shipped tier in the "knows it cold"
  band. (Design rule either way: membership is decided by probing
  tokenizers, never by assuming. The arrows lesson, 2026-08-19.)


### Fix the order-dependent tier-2 test flake

**Status**: identified 2026-07-02.

`test_v107_tier2::test_tier2_recall_surfaces_archived` fails in a full
suite run and passes in isolation, on trees before and after v10.15, so
it is state leakage between tests (likely env or module-level cache),
not a product bug. Track it down and pin it.

### Weave phase signal quality

**Status**: identified 2026-06-30 during the consolidation bench.

Phase 3 (Weave) scores ~50% against gold link classifications for every
model tested, strong and weak alike; models rarely MISS links but
over-link and mislabel link types. The phase is net-positive but
low-signal. Candidates: tighten `WEAVE_MIN_SIMILARITY`, reduce
`WEAVE_TOP_K`, or fold link-type classification into the NLI layer
(entailment direction distinguishes supports/refines better than a
generative label).

### Postgres backend

**Status**: stub exists (`PostgresStore`), not implemented.

Postgres + pgvector with the same single-transaction guarantee as
SQLite. Adds ACID multi-tenancy and MVCC. Contributions welcome.

### PyPI publication

README still installs via `git clone`. The API surface has been stable
across v10.14-10.16; publish to pypi.org for `pip install mnemos` once
the NLI layer has a few weeks of production soak.

---

## Explicitly rejected (do not resurrect without new evidence)

- **int8 quantization of the NLI models.** Validated 2026-07-03 against
  the 114-pair nli-bench and rejected: dynamic int8 collapses DeBERTa-v3
  scoring to chance (contradiction AUC 0.94 -> 0.51 English, 0.84 -> 0.48
  multilingual) and was not reliably faster on CPU. fp32 ONNX is
  score-identical to torch and ships instead. If someone wants the speed,
  the research path is QDQ/per-channel static quantization WITH the
  parity gate re-run; anything that flips threshold decisions on the
  bench pairs is dead on arrival.
- **Jina v3 reranker.** Decided against; v2 serves the multilingual tier for search
  ranking. The store-decision roles the reranker used to hold moved to
  the NLI layer in v10.15.

---

## Older candidates (lower priority, still plausible)

- **Parallelize Phase 0.5 (Cemelify).** Identified 2026-04-18. Serial
  Haiku calls make large cemelify runs slow (75-100 min at 750
  candidates); a ThreadPoolExecutor with `MNEMOS_CEMELIFY_WORKERS` would
  give an order-of-magnitude speedup (the standalone batch helper did
  457 memories in under 6 min at 8 workers). Priority dropped: the
  reference deployment runs cemelify OFF by policy (document-shaped
  content must not be machine-rewritten) and cloud Haiku made the runs
  it does do fast enough. Verify SQLite write-path thread safety before
  attempting; worst case parallelize the LLM calls and serialize writes.
- **Per-cycle LLM spend cap.** `NORMAL_MAX_CALLS` / `SURGE_MAX_CALLS`
  became env-tunable in v10.15.1, which covers call-count budgeting; a
  money-denominated cap (fail loud at N dollars) is still open.
- **Phase 0.5 sample preview** in dry-run mode: show diffs of a few
  random candidates' cemelified output before applying.
- **Cemelify-on-import in batch** for `mnemos ingest`.

---

Last updated: 2026-07-03
