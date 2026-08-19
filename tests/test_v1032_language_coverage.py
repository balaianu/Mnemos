"""Tests for v10.32.0: language-coverage detection for English-only tiers.

An English-only model pair on a non-English store fails silently: nothing
errors, retrieval is just quietly mediocre, and nothing attributes it. Same
failure shape as a dimension mismatch, same treatment: detect and say so.

Calibration is against a real store, not intuition: an English-primary CML
store whose Swedish terms of art are its retrieval keys measures 0.5-2%
non-ASCII letters in the affected memories, so the per-memory threshold is
0.5%. Pure-English stores measure 0 and never trip it.
"""

from mnemos.language import is_english_only, non_english_share, scan_contents
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


def test_doctor_quiet_with_multilingual_pair(tmp_path):
    m = _mnemos(tmp_path)
    _seed(m, ["styrelsemöte ärende förening"] * 5)
    report = m.doctor()
    assert not any("Language coverage" in i for i in report["issues"])
