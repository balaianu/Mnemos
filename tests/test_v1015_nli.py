"""Tests for v10.15.0: the NLI decision layer (bench-backed, 2026-07-02).

Covers mnemos/nli.py (language routing, direction aggregation, graceful
degradation), the store-path integrations (MNEMOS_DEDUP_CONFIRM=nli
bidirectional-entailment confirm tier, MNEMOS_CONTRADICT_MODE=nli
max-direction P(contradiction) gate), and the Nyx phase-4 candidate
selection (floor-only cosine gate in nli mode vs the legacy band).

Model-dependent scoring is faked via monkeypatch; no NLI checkpoint is
downloaded by this suite.
"""

import mnemos.core as core_mod
import mnemos.nli as nli
from mnemos.core import Mnemos
from mnemos.storage.sqlite_store import SQLiteStore

DIMS = 1024


def _store(tmp_path, name="m.db"):
    return SQLiteStore(db_path=str(tmp_path / name), namespace="t")


def _mnemos(tmp_path, **kw):
    kw.setdefault("enable_contradiction_detection", False)
    kw.setdefault("enable_rerank", False)
    return Mnemos(store=_store(tmp_path), namespace="t", **kw)


def _fake_embed(texts, prefix="passage"):
    return [[0.001] * DIMS for _ in texts]


class _StubScorer:
    """Fake NLI scorer with per-direction (entail, contra) scores."""

    def __init__(self, scores):
        self.scores = scores  # {(premise, hypothesis): (entail, contra)}
        self.calls = []

    def score(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        return self.scores.get((premise, hypothesis), (0.0, 0.0))


class TestIsEnglish:
    def test_english_cml_is_english(self):
        assert nli.is_english(
            "F: the server has 64GB of RAM and is used for the media stack")

    def test_swedish_prose_is_not_english(self):
        assert not nli.is_english(
            "F: servern har 64GB minne och används för mediastacken")

    def test_german_prose_is_not_english(self):
        assert not nli.is_english(
            "F: der Server hat 64GB Speicher und wird für Medien genutzt")

    def test_no_prose_signal_defaults_to_english(self):
        assert nli.is_english("F:epsilon 64gb ddr5 /root/work v10.15.0")


class TestDirectionAggregation:
    def test_p_contradiction_takes_max_of_both_directions(self, monkeypatch):
        stub = _StubScorer({("a", "b"): (0.1, 0.2), ("b", "a"): (0.05, 0.9)})
        monkeypatch.setattr(nli, "_get_scorer", lambda multilingual=False: stub)
        monkeypatch.setattr(nli, "is_available", lambda: True)
        assert nli.p_contradiction("a", "b") == 0.9

    def test_bidirectional_entailment_takes_min_of_both_directions(self, monkeypatch):
        stub = _StubScorer({("a", "b"): (0.9, 0.0), ("b", "a"): (0.4, 0.0)})
        monkeypatch.setattr(nli, "_get_scorer", lambda multilingual=False: stub)
        monkeypatch.setattr(nli, "is_available", lambda: True)
        assert nli.bidirectional_entailment("a", "b") == 0.4

    def test_non_english_pair_routes_to_multilingual_scorer(self, monkeypatch):
        seen = []

        def fake_get_scorer(multilingual=False):
            seen.append(multilingual)
            return _StubScorer({})

        monkeypatch.setattr(nli, "_get_scorer", fake_get_scorer)
        monkeypatch.setattr(nli, "is_available", lambda: True)
        nli.p_contradiction("servern har 64GB minne och den är snabb",
                            "the server is fast and has memory")
        assert seen and seen[0] is True

    def test_english_pair_routes_to_english_scorer(self, monkeypatch):
        seen = []

        def fake_get_scorer(multilingual=False):
            seen.append(multilingual)
            return _StubScorer({})

        monkeypatch.setattr(nli, "_get_scorer", fake_get_scorer)
        monkeypatch.setattr(nli, "is_available", lambda: True)
        nli.p_contradiction("the server is fast and has memory",
                            "the server is slow and has no memory")
        assert seen and seen[0] is False

    def test_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr(nli, "is_available", lambda: False)
        assert nli.p_contradiction("a", "b") is None
        assert nli.bidirectional_entailment("a", "b") is None


class TestNliDedupConfirm:
    def _prime(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _fake_embed)
        monkeypatch.setenv("MNEMOS_DEDUP_CONFIRM", "nli")
        m = _mnemos(tmp_path)
        m.store_memory("dev", "F: the deploy pipeline uses staging first",
                       skip_dedup=True)
        return m

    def test_blocks_on_high_bidirectional_entailment(self, tmp_path, monkeypatch):
        m = self._prime(tmp_path, monkeypatch)
        monkeypatch.setattr(core_mod.nli, "is_available", lambda: True)
        monkeypatch.setattr(core_mod.nli, "bidirectional_entailment",
                            lambda a, b: 0.95)
        result = m.store_memory(
            "dev", "F: the deploy pipeline always goes through staging first")
        assert "existing_id" in result
        assert "nli" in result["methods"]

    def test_allows_distinct_below_threshold(self, tmp_path, monkeypatch):
        m = self._prime(tmp_path, monkeypatch)
        monkeypatch.setattr(core_mod.nli, "is_available", lambda: True)
        monkeypatch.setattr(core_mod.nli, "bidirectional_entailment",
                            lambda a, b: 0.30)
        result = m.store_memory(
            "dev", "F: the deploy pipeline has a rollback stage")
        assert "id" in result and "existing_id" not in result

    def test_falls_back_to_vec_path_when_nli_unavailable(self, tmp_path, monkeypatch):
        m = self._prime(tmp_path, monkeypatch)
        monkeypatch.setattr(core_mod.nli, "is_available", lambda: False)
        # identical fake vectors: vec distance 0 -> legacy fallback blocks
        result = m.store_memory(
            "dev", "F: the deploy pipeline uses staging first really")
        assert "existing_id" in result
        assert "nli" not in result["methods"]


class TestNliContradictionMode:
    def _prime(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _fake_embed)
        monkeypatch.setenv("MNEMOS_CONTRADICT_MODE", "nli")
        monkeypatch.setenv("MNEMOS_DEDUP_CONFIRM", "nli")
        m = _mnemos(tmp_path, enable_contradiction_detection=True)
        monkeypatch.setattr(core_mod.nli, "is_available", lambda: True)
        monkeypatch.setattr(core_mod.nli, "bidirectional_entailment",
                            lambda a, b: 0.0)  # never dedup-block in these tests
        m.store_memory("dev", "F: the API listens on port 8080", skip_dedup=True)
        return m

    def test_warns_and_links_above_threshold(self, tmp_path, monkeypatch):
        m = self._prime(tmp_path, monkeypatch)
        monkeypatch.setattr(core_mod.nli, "p_contradiction", lambda a, b: 0.99)
        result = m.store_memory("dev", "F: the API listens on port 9090")
        assert result.get("contradictions"), result
        links = m.store._get_conn().execute(
            "SELECT relation_type FROM memory_links WHERE source_id = ?",
            (result["id"],)).fetchall()
        assert any(r[0] == "contradicts" for r in links)

    def test_silent_below_threshold(self, tmp_path, monkeypatch):
        m = self._prime(tmp_path, monkeypatch)
        monkeypatch.setattr(core_mod.nli, "p_contradiction", lambda a, b: 0.50)
        result = m.store_memory("dev", "F: the API supports websocket upgrades")
        assert not result.get("contradictions")
        links = m.store._get_conn().execute(
            "SELECT relation_type FROM memory_links WHERE source_id = ?",
            (result["id"],)).fetchall()
        assert not any(r[0] == "contradicts" for r in links)

    def test_unavailable_yields_no_contradictions(self, tmp_path, monkeypatch):
        m = self._prime(tmp_path, monkeypatch)
        monkeypatch.setattr(core_mod.nli, "is_available", lambda: False)
        result = m.store_memory("dev", "F: the API listens on port 9090")
        assert not result.get("contradictions")


class TestLineLevelFinder:
    def test_line_max_contradiction_returns_max_over_line_pairs(self, monkeypatch):
        a = "F: the API listens on port 8080\nF: the DB is postgres"
        b = "F: the API listens on port 9090\nF: the cache is redis"
        contra_pair = ("F: the API listens on port 8080",
                       "F: the API listens on port 9090")

        def fake_line_embed(texts, prefix="passage"):
            # identical vectors: every line pair passes cosine preselect
            return [[0.001] * DIMS for _ in texts]

        stub = _StubScorer({})

        def fake_score(premise, hypothesis):
            if {premise, hypothesis} == set(contra_pair):
                return (0.01, 0.97)
            return (0.5, 0.05)

        stub.score = fake_score
        monkeypatch.setattr(nli, "_get_scorer", lambda multilingual=False: stub)
        monkeypatch.setattr(nli, "is_available", lambda: True)
        monkeypatch.setattr(nli, "embed", fake_line_embed)
        assert nli.line_max_contradiction(a, b) == 0.97

    def test_line_max_contradiction_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr(nli, "is_available", lambda: False)
        assert nli.line_max_contradiction("a", "b") is None


class TestPhase4CandidateSelection:
    def _fixtures(self):
        mem_by_id = {
            1: {"project": "dev", "type": "fact"},
            2: {"project": "dev", "type": "fact"},
            3: {"project": "dev", "type": "fact"},
        }
        # pair (1,2) cos 0.95 (above legacy band), pair (1,3) cos 0.70 (in band),
        # pair (2,3) cos 0.40 (below floor)
        sim = [[1.0, 0.95, 0.70], [0.95, 1.0, 0.40], [0.70, 0.40, 1.0]]
        return [1, 2, 3], sim, mem_by_id

    def test_cosine_mode_keeps_legacy_band(self):
        from mnemos.consolidation.phases import select_contradict_candidates
        ids, sim, mem_by_id = self._fixtures()
        pairs = select_contradict_candidates(ids, sim, mem_by_id, mode="cosine")
        assert (1, 3) in [(a, b) for a, b, _ in pairs]
        assert (1, 2) not in [(a, b) for a, b, _ in pairs]

    def test_nli_mode_has_no_upper_band(self):
        from mnemos.consolidation.phases import select_contradict_candidates
        ids, sim, mem_by_id = self._fixtures()
        pairs = select_contradict_candidates(ids, sim, mem_by_id, mode="nli")
        got = [(a, b) for a, b, _ in pairs]
        assert (1, 2) in got and (1, 3) in got
        assert (2, 3) not in got
