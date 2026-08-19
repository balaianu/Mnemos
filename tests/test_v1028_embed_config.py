"""Tests for v10.28.0: configurable embedder, and repair when it changes.

Three defects motivated this. (1) `MNEMOS_EMBED_DIMS` was documented in
docs/features.md as the knob for switching models, but constants.py hardcoded
1024, so following the docs did nothing. (2) embed() hardcoded e5's
"passage: "/"query: " prefixes, so a user who did switch to BGE silently
embedded every document and query with a meaningless prefix glued on.
(3) doctor's provenance check pointed at `mnemos embed-fill` to repair a model
swap, which cannot work: embed-fill only fills rows with NO vector, and after a
swap every row has one from the wrong encoder.
"""

import importlib
import pytest

from mnemos.core import Mnemos
from mnemos.storage.sqlite_store import SQLiteStore


def _mnemos(tmp_path, namespace="t"):
    store = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace=namespace)
    return Mnemos(store=store, namespace=namespace,
                  enable_contradiction_detection=False, enable_rerank=False)


def _reload_with(monkeypatch, **env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import mnemos.constants as c
    return importlib.reload(c)


# --- dims are configurable ------------------------------------------------

def test_dims_default_is_1024(monkeypatch):
    c = _reload_with(monkeypatch, MNEMOS_EMBED_DIMS=None)
    assert c.FASTEMBED_DIMS == 1024


def test_dims_follow_env(monkeypatch):
    c = _reload_with(monkeypatch, MNEMOS_EMBED_DIMS="384")
    assert c.FASTEMBED_DIMS == 384
    _reload_with(monkeypatch, MNEMOS_EMBED_DIMS=None)


# --- prefixes follow the model family -------------------------------------

@pytest.mark.parametrize("model,expected", [
    ("intfloat/multilingual-e5-large", ("passage: ", "query: ")),
    ("intfloat/multilingual-e5-small", ("passage: ", "query: ")),
    ("BAAI/bge-small-en-v1.5",
     ("", "Represent this sentence for searching relevant passages: ")),
    ("BAAI/bge-base-en-v1.5",
     ("", "Represent this sentence for searching relevant passages: ")),
    # Unknown families get no prefix: a wrong prefix is worse than none.
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ("", "")),
    ("BAAI/bge-small-zh-v1.5", ("", "")),
])
def test_prefixes_per_family(monkeypatch, model, expected):
    _reload_with(monkeypatch, MNEMOS_EMBED_MODEL=model)
    import mnemos.embed as e
    importlib.reload(e)
    assert e._prefixes() == expected
    _reload_with(monkeypatch, MNEMOS_EMBED_MODEL=None)
    importlib.reload(e)


# --- a mismatch names the knob, not just the numbers ----------------------

def test_check_dims_raises_with_actionable_message(monkeypatch):
    import mnemos.embed as e
    importlib.reload(e)
    e._dims_checked = False
    with pytest.raises(ValueError) as excinfo:
        e._check_dims(384)
    msg = str(excinfo.value)
    assert "MNEMOS_EMBED_DIMS=384" in msg
    assert "reembed" in msg or "embed-fill" in msg


def test_check_dims_runs_once(monkeypatch):
    import mnemos.embed as e
    importlib.reload(e)
    e._dims_checked = False
    e._check_dims(e.FASTEMBED_DIMS)
    # Already satisfied once; a later bad value is not re-litigated per call.
    e._check_dims(384)


# --- the index width is readable and rebuildable --------------------------

def test_get_vec_dims_reads_declared_width(tmp_path):
    m = _mnemos(tmp_path)
    from mnemos.constants import FASTEMBED_DIMS
    assert m.store.get_vec_dims() == FASTEMBED_DIMS


def test_reset_vec_index_changes_width(tmp_path):
    m = _mnemos(tmp_path)
    m.store.reset_vec_index(384)
    assert m.store.get_vec_dims() == 384


def test_reset_vec_index_clears_active_meta_only(tmp_path):
    m = _mnemos(tmp_path)
    conn = m.store._get_conn()
    conn.execute("INSERT INTO embed_meta (source_db, source_id, text_hash, model) "
                 "VALUES ('memory', 1, 'h', 'old-model')")
    conn.execute("INSERT INTO embed_meta_arch (source_db, source_id, text_hash, model) "
                 "VALUES ('memory', 1, 'h', 'old-model')")
    conn.commit()
    m.store.reset_vec_index(384)
    assert conn.execute("SELECT count(*) FROM embed_meta WHERE source_db='memory'"
                        ).fetchone()[0] == 0
    # The tier-2 archived index has its own rebuild command and is untouched.
    assert conn.execute("SELECT count(*) FROM embed_meta_arch").fetchone()[0] == 1


# --- doctor surfaces the mismatch -----------------------------------------

def test_doctor_flags_dimension_mismatch(tmp_path):
    m = _mnemos(tmp_path)
    m.store.reset_vec_index(384)
    report = m.doctor()
    assert any("dimension mismatch" in i.lower() for i in report["issues"])
    assert any("reembed" in i for i in report["issues"])


def test_doctor_passes_on_matching_dims(tmp_path):
    m = _mnemos(tmp_path)
    report = m.doctor()
    assert not any("dimension mismatch" in i.lower() for i in report["issues"])
    assert any("Vector dimensions" in c for c in report["checks"])


# --- reembed reports before it acts ---------------------------------------

def test_reembed_dry_run_touches_nothing(tmp_path):
    m = _mnemos(tmp_path)
    before = m.store.get_vec_dims()
    result = m.reembed(dry_run=True)
    assert result["dry_run"] is True
    assert result["previous_dims"] == before
    assert "backup" not in result
    assert m.store.get_vec_dims() == before


# --- tier-2 index follows a tier switch (v10.32.2, field-reported) ---------

def test_arch_dims_readable(tmp_path):
    m = _mnemos(tmp_path)
    from mnemos.embed import effective_dims
    assert m.store.get_arch_vec_dims() == effective_dims()


def test_reset_arch_index_changes_width_and_clears_meta(tmp_path):
    m = _mnemos(tmp_path)
    conn = m.store._get_conn()
    conn.execute("INSERT INTO embed_meta_arch (source_db, source_id, text_hash, model) "
                 "VALUES ('memory', 1, 'h', 'old')")
    conn.commit()
    m.store.reset_arch_vec_index(384)
    assert m.store.get_arch_vec_dims() == 384
    # Meta must clear too, or the backfill skips rows whose stale meta says
    # they are covered; that is exactly why plain backfill cannot repair a
    # switch.
    assert conn.execute("SELECT count(*) FROM embed_meta_arch").fetchone()[0] == 0


def test_doctor_flags_stranded_archived_index(tmp_path):
    """The field case: active index rebuilt at 384, archived left at 1024,
    doctor reported complete because it only counted missing rows."""
    m = _mnemos(tmp_path)
    m.store.reset_vec_index(384)
    import os
    os.environ["MNEMOS_EMBED_DIMS"] = "384"
    import importlib
    import mnemos.constants, mnemos.embed
    importlib.reload(mnemos.constants); importlib.reload(mnemos.embed)
    try:
        report = m.doctor()
        assert any("Archived vector index" in i and "reindex-archived" in i
                   for i in report["issues"])
    finally:
        del os.environ["MNEMOS_EMBED_DIMS"]
        importlib.reload(mnemos.constants); importlib.reload(mnemos.embed)


def test_doctor_dim_check_respects_pinning(tmp_path):
    """Gap 2 from the field: the check compared against the raw constants, so
    a store correctly pinned to a non-default model reported a false
    'vectors are being rejected' issue while inserts succeeded."""
    m = _mnemos(tmp_path)
    from mnemos import embed as e
    # Simulate a store pinned to bge/384 while the default remains e5/1024.
    m.store.reset_vec_index(384)
    changed = e.adopt_store_config("BAAI/bge-small-en-v1.5", 384)
    assert changed, "pin must take effect for the scenario to be real"
    try:
        report = m.doctor()
        assert not any("dimension mismatch" in i.lower()
                       for i in report["issues"])
    finally:
        e.adopt_store_config("intfloat/multilingual-e5-large+cmlnorm", 1024)
