"""Tests for v10.29.0: CML operator normalization for the embedder.

CML's relational operators are unicode. SentencePiece vocabularies (e5, the
Jina reranker: 250k) carry them; English WordPiece vocabularies (bge, gte,
mxbai, nomic, MiniLM: 30522) map every one of them to a single shared [UNK].
Measured on a 720-memory production store: 901 UNK tokens, and since [UNK] has
one embedding, "migration confirmed" and "migration rejected" collapse to the
same vector. That is a precision failure, not a recall one, which matches the
published CML ablation: R@5 barely moves while R@1 on single-session-preference
falls to 53.33%.

Normalization substitutes the words the operators stand for, before
tokenization, for the embedder only. Stored content, FTS5, the reranker and
what the agent reads back are untouched.
"""

import importlib
import pytest


def _reload(monkeypatch, enabled):
    monkeypatch.setenv("MNEMOS_EMBED_NORMALIZE_CML", "1" if enabled else "0")
    import mnemos.constants as c
    importlib.reload(c)
    import mnemos.embed as e
    return importlib.reload(e)


def test_on_by_default(monkeypatch):
    """Default ON since v10.30.0; safe only because populated stores pin."""
    monkeypatch.delenv("MNEMOS_EMBED_NORMALIZE_CML", raising=False)
    import mnemos.constants as c
    importlib.reload(c)
    assert c.EMBED_NORMALIZE_CML is True


def test_disabled_is_identity(monkeypatch):
    e = _reload(monkeypatch, False)
    src = "D: migration ✓ confirmed ∵ tests green"
    assert e.normalize_cml(src) == src


@pytest.mark.parametrize("sym,word", [
    ("∵", "because"), ("∴", "therefore"), ("△", "changed from"),
    ("⚠", "warning"), ("✓", "confirmed"), ("✗", "rejected"),
])
def test_each_lost_operator_becomes_a_word(monkeypatch, sym, word):
    e = _reload(monkeypatch, True)
    out = e.normalize_cml(f"F: thing {sym} other")
    assert word in out
    assert sym not in out


def test_opposites_stay_distinguishable(monkeypatch):
    """The failure this exists to prevent: ✓ and ✗ are the same [UNK]."""
    e = _reload(monkeypatch, True)
    yes = e.normalize_cml("D: migration ✓")
    no = e.normalize_cml("W: migration ✗")
    assert yes != no
    assert "confirmed" in yes and "rejected" in no


def test_ascii_operators_are_left_alone(monkeypatch):
    """v2 map: only ASCII operators survive untouched. The arrows and null
    map to ASCII forms because the default reranker's byte-BPE vocabulary
    shreds the unicode originals, and the arrows are the two most common
    operators in a real store."""
    e = _reload(monkeypatch, True)
    out = e.normalize_cml("F: a → b ↔ c @ d ~ e ∅ f > g")
    for sym in ("@", "~", ">"):
        assert sym in out
    assert "->" in out and "<->" in out and " none " in out
    for sym in ("→", "↔", "∅"):
        assert sym not in out


def test_snippet_markers_and_rules_are_dropped(monkeypatch):
    e = _reload(monkeypatch, True)
    out = e.normalize_cml("F: ⟪match⟫ ═══ done")
    for sym in ("⟪", "⟫", "═"):
        assert sym not in out
    assert "match" in out and "done" in out


def test_no_double_spaces_left(monkeypatch):
    e = _reload(monkeypatch, True)
    assert "  " not in e.normalize_cml("F: a ⟪ ⟫ ═ ═ b")


def test_plain_text_is_untouched(monkeypatch):
    e = _reload(monkeypatch, True)
    src = "the board approved the budget on Tuesday"
    assert e.normalize_cml(src) == src


# --- provenance: a flag flip must be visible, not silent -------------------

def test_model_id_marks_normalization(monkeypatch):
    e = _reload(monkeypatch, False)
    plain = e.embed_model_id()
    e = _reload(monkeypatch, True)
    assert e.embed_model_id() == plain + f"+cmlnorm{e.CML_MAP_VERSION}"


def test_model_id_is_plain_when_off(monkeypatch):
    e = _reload(monkeypatch, False)
    from mnemos.constants import FASTEMBED_MODEL
    assert e.embed_model_id() == FASTEMBED_MODEL


# --- per-store pinning (v10.30.0) -----------------------------------------

def _fresh_embed(monkeypatch, **env):
    """Reload constants+embed under a given env, returning the embed module."""
    for k in ("MNEMOS_EMBED_MODEL", "MNEMOS_EMBED_DIMS",
              "MNEMOS_EMBED_NORMALIZE_CML"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import mnemos.constants as c
    importlib.reload(c)
    import mnemos.embed as e
    return importlib.reload(e)


def test_adopts_model_dims_and_normalization_from_store(monkeypatch):
    e = _fresh_embed(monkeypatch)
    changed = e.adopt_store_config("BAAI/bge-base-en-v1.5+cmlnorm", 768)
    assert changed["model"][1] == "BAAI/bge-base-en-v1.5"
    assert changed["dims"][1] == 768
    # normalize already matches the v10.30.0 default, so it is not a change
    assert "normalize" not in changed
    assert e.effective_model() == "BAAI/bge-base-en-v1.5"
    assert e.effective_dims() == 768
    assert e.effective_normalize() is True


def test_plain_model_id_adopts_without_normalization(monkeypatch):
    e = _fresh_embed(monkeypatch)
    e.adopt_store_config("BAAI/bge-small-en-v1.5", 384)
    assert e.effective_normalize() is False
    assert e.embed_model_id() == "BAAI/bge-small-en-v1.5"


def test_explicit_env_outranks_the_store(monkeypatch):
    """An exported knob is an instruction; a default is only a seed."""
    e = _fresh_embed(monkeypatch, MNEMOS_EMBED_MODEL="BAAI/bge-small-en-v1.5",
                     MNEMOS_EMBED_DIMS="384",
                     MNEMOS_EMBED_NORMALIZE_CML="1")
    changed = e.adopt_store_config("intfloat/multilingual-e5-large", 1024)
    assert changed == {}
    assert e.effective_model() == "BAAI/bge-small-en-v1.5"
    assert e.effective_dims() == 384
    assert e.effective_normalize() is True


def test_partial_explicit_only_pins_the_rest(monkeypatch):
    e = _fresh_embed(monkeypatch, MNEMOS_EMBED_MODEL="BAAI/bge-large-en-v1.5")
    changed = e.adopt_store_config("BAAI/bge-base-en-v1.5", 768)
    assert "model" not in changed        # explicitly set, untouched
    assert changed["dims"][1] == 768     # not set, adopted from the store
    assert e.effective_model() == "BAAI/bge-large-en-v1.5"
    assert e.effective_dims() == 768


def test_matching_config_reports_no_change(monkeypatch):
    e = _fresh_embed(monkeypatch)
    # Default-agnostic on purpose: build the provenance string from the
    # CURRENT default so the next default flip does not break this test.
    assert e.adopt_store_config(e.embed_model_id(), e.effective_dims()) == {}


def test_null_provenance_pins_dims_only(monkeypatch):
    """Changed contract in v10.33.0 (field finding): pre-v10.6 stores have
    vectors but NULL embed_meta.model. The model cannot be guessed, but the
    index width is authoritative, so dims pin even without a model name;
    otherwise a default flip leaves every new insert rejected against the
    old-geometry index."""
    e = _fresh_embed(monkeypatch)
    changed = e.adopt_store_config(None, 1024)
    assert changed == {"dims": (e.FASTEMBED_DIMS, 1024)} or "dims" in changed
    assert e.effective_dims() == 1024
    from mnemos.constants import FASTEMBED_MODEL
    assert e.effective_model() == FASTEMBED_MODEL   # model untouched
    # dims matching the effective value is a true no-op
    assert e.adopt_store_config("", e.effective_dims()) == {}


def test_populated_store_pins_a_moved_default(monkeypatch, tmp_path):
    """The upgrade case: default moved, store did not."""
    e = _fresh_embed(monkeypatch)
    from mnemos.storage.sqlite_store import SQLiteStore
    store = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")
    conn = store._get_conn()
    conn.execute("INSERT INTO embed_meta (source_db, source_id, text_hash, model) "
                 "VALUES ('memory', 1, 'h', 'BAAI/bge-base-en-v1.5+cmlnorm')")
    conn.commit()
    store2 = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")
    store2._get_conn()
    assert store2._embed_adoption["model"][1] == "BAAI/bge-base-en-v1.5"
    assert e.effective_normalize() is True


def test_empty_store_does_not_pin_anything(monkeypatch, tmp_path):
    e = _fresh_embed(monkeypatch)
    from mnemos.storage.sqlite_store import SQLiteStore
    store = SQLiteStore(db_path=str(tmp_path / "empty.db"), namespace="t")
    store._get_conn()
    assert store._embed_adoption == {}
    from mnemos.constants import FASTEMBED_MODEL
    assert e.effective_model() == FASTEMBED_MODEL


def test_existing_unnormalized_store_survives_the_new_default(monkeypatch, tmp_path):
    """The upgrade case for v10.30.0.

    Someone on 10.29 has a store full of un-normalized vectors. They upgrade,
    the default flips on, and nothing in their store may change meaning: the
    provenance recorded on those vectors carries no +cmlnorm suffix, so the
    store pins normalization back off for itself.
    """
    e = _fresh_embed(monkeypatch)
    assert e.effective_normalize() is True          # the new default
    from mnemos.storage.sqlite_store import SQLiteStore
    store = SQLiteStore(db_path=str(tmp_path / "legacy.db"), namespace="t")
    conn = store._get_conn()
    conn.execute("INSERT INTO embed_meta (source_db, source_id, text_hash, model) "
                 "VALUES ('memory', 1, 'h', 'intfloat/multilingual-e5-large')")
    conn.commit()
    reopened = SQLiteStore(db_path=str(tmp_path / "legacy.db"), namespace="t")
    reopened._get_conn()
    assert reopened._embed_adoption["normalize"] == (True, False)
    assert e.effective_normalize() is False
