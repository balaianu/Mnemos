"""Tests for v10.32.0: language-coverage detection for English-only tiers.

An English-only model pair on a non-English store fails silently: nothing
errors, retrieval is just quietly mediocre, and nothing attributes it. Same
failure shape as a dimension mismatch, same treatment: detect and say so.

Calibration is against a real store, not intuition: an English-primary CML
store whose Swedish terms of art are its retrieval keys measures 0.5-2%
non-ASCII letters in the affected memories, so the per-memory threshold is
0.5%. Pure-English stores measure 0 and never trip it.
"""

import os

from mnemos.language import is_english_only, non_english_share, scan_contents
from mnemos.constants import FASTEMBED_DIMS
from mnemos.core import Mnemos
from mnemos.storage.sqlite_store import SQLiteStore


# --- model classification ---------------------------------------------------

def test_english_only_models_are_flagged():
    for m in ("BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5",
              "Alibaba-NLP/gte-reranker-modernbert-base",
              "Xenova/ms-marco-MiniLM-L-6-v2",
              "jinaai/jina-reranker-v1-turbo-en"):
        assert is_english_only(m), m


def test_multilingual_models_are_not():
    for m in ("intfloat/multilingual-e5-large",
              "jinaai/jina-reranker-v2-base-multilingual",
              "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        assert not is_english_only(m), m


def test_unknown_models_are_assumed_capable():
    """A false 'your setup is fine' is worse than a false warning, but an
    unknown id gives no evidence either way; assume capable so new
    multilingual models do not produce spurious warnings."""
    assert not is_english_only("acme/future-reranker-9000")


# --- the share heuristic ----------------------------------------------------

def test_pure_english_is_zero():
    assert non_english_share("the quick brown fox, 42 times") == 0.0


def test_swedish_terms_register():
    assert non_english_share("BRF kallelse till styrelsemöte") > 0.005


def test_cml_operators_do_not_count_as_language():
    assert non_english_share("D: migration ✓ → deployed ∵ tests") == 0.0


def test_emoji_do_not_count():
    assert non_english_share("shipped it \U0001F525\U0001F389 nice") == 0.0


def test_cjk_registers_strongly():
    assert non_english_share("記憶システム memory system") > 0.3


def test_empty_and_numeric_are_safe():
    assert non_english_share("") == 0.0
    assert non_english_share("12345 !!") == 0.0


def test_scan_counts_affected_memories():
    rows = ["plain english", "styrelsemöte i föreningen", "more english",
            "ärende hos hyresnämnden"]
    affected, total = scan_contents(rows)
    assert (affected, total) == (2, 4)


# --- doctor integration ------------------------------------------------------

def _mnemos(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")
    return Mnemos(store=store, namespace="t",
                  enable_contradiction_detection=False, enable_rerank=True)


def _seed(m, contents):
    for i, c in enumerate(contents, 1):
        conn = m.store._get_conn()
        conn.execute(
            "INSERT INTO memories (id, namespace, project, content, status, layer, type) "
            "VALUES (?, 't', 'dev', ?, 'active', 'semantic', 'fact')", (i, c))
        conn.commit()


def test_doctor_flags_english_pair_on_mixed_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("MNEMOS_EMBED_DIMS", "384")
    monkeypatch.setenv("MNEMOS_RERANKER_MODEL",
                       "Alibaba-NLP/gte-reranker-modernbert-base")
    import importlib
    import mnemos.constants, mnemos.embed
    importlib.reload(mnemos.constants); importlib.reload(mnemos.embed)
    m = _mnemos(tmp_path)
    _seed(m, ["styrelsemöte ärende förening"] * 3 + ["english"] * 2)
    report = m.doctor()
    assert any("Language coverage" in i and "English-only" in i
               for i in report["issues"])
    monkeypatch.delenv("MNEMOS_EMBED_MODEL"); monkeypatch.delenv("MNEMOS_EMBED_DIMS")
    monkeypatch.delenv("MNEMOS_RERANKER_MODEL")
    importlib.reload(mnemos.constants); importlib.reload(mnemos.embed)


def test_doctor_quiet_on_english_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("MNEMOS_EMBED_DIMS", "384")
    import importlib
    import mnemos.constants, mnemos.embed
    importlib.reload(mnemos.constants); importlib.reload(mnemos.embed)
    m = _mnemos(tmp_path)
    _seed(m, ["perfectly ordinary english content"] * 5)
    report = m.doctor()
    assert not any("Language coverage" in i for i in report["issues"])
    monkeypatch.delenv("MNEMOS_EMBED_MODEL"); monkeypatch.delenv("MNEMOS_EMBED_DIMS")
    importlib.reload(mnemos.constants); importlib.reload(mnemos.embed)


def test_doctor_quiet_with_multilingual_pair(tmp_path, monkeypatch):
    _multilingual_models(monkeypatch)
    m = _mnemos(tmp_path)
    _seed(m, ["styrelsemöte ärende förening"] * 5)
    import mnemos.constants as _c
    from mnemos.embed import effective_model
    report = m.doctor()
    offending = [i for i in report["issues"] if "Language coverage" in i]
    assert not offending, (offending, _c.RERANKER_MODEL, effective_model(),
                           os.environ.get("MNEMOS_RERANKER_MODEL"))
    _reset_models(monkeypatch)


# --- store-time early warning ------------------------------------------------

def _english_models(monkeypatch):
    import importlib
    monkeypatch.setenv("MNEMOS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("MNEMOS_EMBED_DIMS", "384")
    monkeypatch.setenv("MNEMOS_RERANKER_MODEL",
                       "Alibaba-NLP/gte-reranker-modernbert-base")
    import mnemos.constants, mnemos.embed
    importlib.reload(mnemos.constants); importlib.reload(mnemos.embed)
    # These tests exercise the WARNING, not the encoder. Stub the embed call
    # core.py imported, so no model loads and the vector width matches
    # whatever the temp store was created with.
    import mnemos.core
    monkeypatch.setattr(mnemos.core, "embed",
                        lambda texts, prefix="passage": [[0.1] * FASTEMBED_DIMS
                                                         for _ in texts])


def _reset_models(monkeypatch):
    import importlib
    for k in ("MNEMOS_EMBED_MODEL", "MNEMOS_EMBED_DIMS", "MNEMOS_RERANKER_MODEL"):
        monkeypatch.delenv(k, raising=False)
    import mnemos.constants, mnemos.embed
    importlib.reload(mnemos.constants); importlib.reload(mnemos.embed)


def _multilingual_models(monkeypatch):
    """The multilingual TIER, explicitly: since v10.33.0 the defaults are the
    English tier, so tests about the multilingual pair must ask for it."""
    import importlib
    monkeypatch.setenv("MNEMOS_EMBED_MODEL", "intfloat/multilingual-e5-large")
    monkeypatch.setenv("MNEMOS_EMBED_DIMS", "1024")
    monkeypatch.setenv("MNEMOS_RERANKER_MODEL",
                       "jinaai/jina-reranker-v2-base-multilingual")
    import mnemos.constants, mnemos.embed
    importlib.reload(mnemos.constants); importlib.reload(mnemos.embed)
    # Some earlier test in a full-suite run leaves embed._effective pointing
    # at the English default even after the reload above (reload rebuilds
    # _effective from constants, but only if constants were reloaded FIRST in
    # this exact order on every path). Pin the pair positively instead of
    # trusting reload ordering.
    import mnemos.embed as _e
    _e._effective.update({"model": "intfloat/multilingual-e5-large",
                          "dims": 1024})


def test_store_warns_early_on_foreign_content(tmp_path, monkeypatch):
    """The wrong moment to learn your tier is wrong is after importing a
    thousand memories. Store #1 is the right moment."""
    _english_models(monkeypatch)
    Mnemos._lang_warned = False
    m = _mnemos(tmp_path)
    r = m.store_memory(project="brf", content="kallelse till styrelsemöte i föreningen",
                       skip_dedup=True)
    assert "warning" in r and "English-only" in r["warning"]
    _reset_models(monkeypatch)


def test_store_warning_fires_once_per_process(tmp_path, monkeypatch):
    _english_models(monkeypatch)
    Mnemos._lang_warned = False
    m = _mnemos(tmp_path)
    r1 = m.store_memory(project="brf", content="ärende hos hyresnämnden", skip_dedup=True)
    r2 = m.store_memory(project="brf", content="beslut om föreningsstämma", skip_dedup=True)
    assert "English-only" in r1.get("warning", "")
    assert "English-only" not in r2.get("warning", "")
    _reset_models(monkeypatch)


def test_store_silent_on_english_content(tmp_path, monkeypatch):
    _english_models(monkeypatch)
    Mnemos._lang_warned = False
    m = _mnemos(tmp_path)
    r = m.store_memory(project="dev", content="plain english content here",
                       skip_dedup=True)
    assert "English-only" not in r.get("warning", "")
    _reset_models(monkeypatch)


def test_store_silent_with_multilingual_models(tmp_path, monkeypatch):
    _multilingual_models(monkeypatch)
    import mnemos.core
    monkeypatch.setattr(mnemos.core, "embed",
                        lambda texts, prefix="passage": [[0.1] * FASTEMBED_DIMS
                                                         for _ in texts])
    Mnemos._lang_warned = False
    m = _mnemos(tmp_path)
    r = m.store_memory(project="brf", content="kallelse till styrelsemöte",
                       skip_dedup=True)
    assert "English-only" not in r.get("warning", "")
    _reset_models(monkeypatch)


def test_http_startup_check_respects_store_pinning(tmp_path, monkeypatch):
    """First 10.33.0 boot regression: the startup notice judged the raw
    default because per-store pinning had not run yet; it must touch the
    store first and judge what the store will actually use."""
    import importlib
    _reset_models(monkeypatch)   # English defaults
    from mnemos.storage.sqlite_store import SQLiteStore
    # Build a store whose provenance pins a multilingual embedder.
    store = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")
    conn = store._get_conn()
    conn.execute("INSERT INTO embed_meta (source_db, source_id, text_hash, model) "
                 "VALUES ('memory', 1, 'h', 'intfloat/multilingual-e5-large+cmlnorm')")
    for i in range(1, 6):
        conn.execute(
            "INSERT INTO memories (id, namespace, project, content, status, layer, type) "
            "VALUES (?, 't', 'brf', 'kallelse till styrelsemöte i föreningen', "
            "'active', 'semantic', 'fact')", (i,))
    conn.commit()
    store.close()
    # Fresh objects, pinning not yet run, exactly like a server boot.
    import mnemos.embed as e
    importlib.reload(e)
    from mnemos.core import Mnemos
    m = Mnemos(store=SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t"),
               namespace="t", enable_contradiction_detection=False,
               enable_rerank=False)
    lines = []
    import mnemos.http_server as h
    import mnemos.language, mnemos.constants
    # Reproduce the startup block's logic directly.
    try:
        rows = m.store.sample_contents(m.namespace)
        enc_en = mnemos.language.is_english_only(e.effective_model())
        assert enc_en is False, "store touch must have pinned e5 before judging"
    finally:
        # The pin mutated module state; put the defaults back or every later
        # test that embeds for real loads e5 against a 384-wide index. Same
        # leak class as the http fixture bug this file just chronicled.
        importlib.reload(e)


# --- mojibake detection and repair (v10.34.0, field-reported) ---------------

def test_mojibake_detected_and_repaired():
    from mnemos.language import has_mojibake, repair_mojibake
    bad = "beslut â€” taget, navajoâ†’anthropic"
    assert has_mojibake(bad)
    fixed, changed = repair_mojibake(bad)
    assert changed and "—" in fixed and "→" in fixed


def test_clean_text_is_never_touched():
    from mnemos.language import has_mojibake, repair_mojibake
    ok = "normal → text — fine, åtgärd klar"
    assert not has_mojibake(ok)
    assert repair_mojibake(ok) == (ok, False)


def test_doctor_reports_mojibake_without_migrate(tmp_path):
    m = _mnemos(tmp_path)
    _seed(m, ["ok content", "trasig â€” minne â†’ fel"])
    report = m.doctor(migrate=False)
    assert any("mojibake" in i for i in report["issues"])


def test_doctor_migrate_repairs_and_reembeds(tmp_path, monkeypatch):
    import mnemos.core
    monkeypatch.setattr(mnemos.core, "embed",
                        lambda texts, prefix="passage": [[0.1] * FASTEMBED_DIMS
                                                         for _ in texts])
    m = _mnemos(tmp_path)
    _seed(m, ["trasig â€” minne"])
    report = m.doctor(migrate=True)
    assert any("repaired mojibake in 1" in x
               for x in report["migrations_applied"])
    fixed = m.store._get_conn().execute(
        "SELECT content FROM memories WHERE id=1").fetchone()[0]
    assert "—" in fixed and "â€" not in fixed
