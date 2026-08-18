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


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("MNEMOS_EMBED_NORMALIZE_CML", raising=False)
    import mnemos.constants as c
    importlib.reload(c)
    assert c.EMBED_NORMALIZE_CML is False


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


def test_surviving_operators_are_left_alone(monkeypatch):
    e = _reload(monkeypatch, True)
    out = e.normalize_cml("F: a → b ↔ c @ d ~ e ∅ f > g")
    for sym in ("→", "↔", "@", "~", "∅", ">"):
        assert sym in out


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
    assert e.embed_model_id() == plain + "+cmlnorm"


def test_model_id_is_plain_when_off(monkeypatch):
    e = _reload(monkeypatch, False)
    from mnemos.constants import FASTEMBED_MODEL
    assert e.embed_model_id() == FASTEMBED_MODEL
