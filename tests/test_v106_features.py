"""Regression tests for v10.6.0: the fresh-eyes review fixes.

Covers: text_hash recorded at store/update time (the staleness column was
previously declared but never written), embed_status stale/unverified
reporting, update() surfacing re-embed failure, hard delete pruning
memory_links, linked-expansion status filtering, and the contradiction
pipeline bailing on an unscored rerank result instead of mass-writing
relates links at sigmoid(0)=0.5.
"""

import mnemos.core as core_mod
from mnemos.core import Mnemos
from mnemos.embed import prep_memory_text, text_hash
from mnemos.storage.base import Memory
from mnemos.storage.sqlite_store import SQLiteStore

DIMS = 1024


def _store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")


def _vec(seed=0.0):
    return [seed] * DIMS


def _fake_embed(texts, prefix="passage"):
    return [_vec(0.001) for _ in texts]


def _failing_embed(texts, prefix="passage"):
    return []


class TestTextHashWiring:
    def test_store_memory_records_text_hash(self, tmp_path):
        store = _store(tmp_path)
        mid = store.store_memory(
            Memory(namespace="t", project="dev", content="alpha"),
            embedding=_vec(), text_hash="hash-a",
        )
        conn = store._get_conn()
        row = conn.execute(
            "SELECT text_hash FROM embed_meta WHERE source_id=?", (mid,)
        ).fetchone()
        assert row["text_hash"] == "hash-a"

    def test_core_store_hash_matches_canonical_text(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _fake_embed)
        m = Mnemos(store=_store(tmp_path), namespace="t",
                   enable_contradiction_detection=False, enable_rerank=False)
        res = m.store_memory("dev", "beta content", tags="x", skip_dedup=True)
        conn = m.store._get_conn()
        row = conn.execute(
            "SELECT text_hash FROM embed_meta WHERE source_id=?", (res["id"],)
        ).fetchone()
        expected = text_hash(prep_memory_text("dev", "beta content", "x",
                                              mem_type="fact", layer="semantic"))
        assert row["text_hash"] == expected

    def test_update_refreshes_hash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _fake_embed)
        m = Mnemos(store=_store(tmp_path), namespace="t",
                   enable_contradiction_detection=False, enable_rerank=False)
        mid = m.store_memory("dev", "gamma", skip_dedup=True)["id"]
        res = m.update(mid, content="gamma revised")
        assert res["embedded"] is True
        conn = m.store._get_conn()
        row = conn.execute(
            "SELECT text_hash FROM embed_meta WHERE source_id=?", (mid,)
        ).fetchone()
        assert row["text_hash"] == text_hash(prep_memory_text(
            "dev", "gamma revised", "", mem_type="fact", layer="semantic"))


class TestStaleDetection:
    def test_failed_reembed_is_reported_and_detectable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _fake_embed)
        m = Mnemos(store=_store(tmp_path), namespace="t",
                   enable_contradiction_detection=False, enable_rerank=False)
        mid = m.store_memory("dev", "delta original", skip_dedup=True)["id"]
        assert m.embed_status()["stale"] == 0

        monkeypatch.setattr(core_mod, "embed", _failing_embed)
        res = m.update(mid, content="delta changed")
        assert res["embedded"] is False
        assert "stale" in res["warning"]

        status = m.embed_status()
        assert status["stale"] == 1  # old vector, new content, now visible

    def test_legacy_null_hash_counts_unverified(self, tmp_path):
        store = _store(tmp_path)
        store.store_memory(Memory(namespace="t", project="dev", content="eps"),
                           embedding=_vec(), text_hash=None)
        m = Mnemos(store=store, namespace="t",
                   enable_contradiction_detection=False, enable_rerank=False)
        status = m.embed_status()
        assert status["unverified"] == 1
        assert status["stale"] == 0


class TestHardDeletePrunesLinks:
    def test_links_removed_with_memory(self, tmp_path):
        store = _store(tmp_path)
        a = store.store_memory(Memory(namespace="t", project="dev", content="a"))
        b = store.store_memory(Memory(namespace="t", project="dev", content="b"))
        store.store_link(a, b, "relates", 0.5)
        assert store.get_links([b])
        store.delete_memory(a, hard=True)
        assert store.get_links([b]) == {}


class TestLinkedExpansionStatusFilter:
    def test_archived_content_not_resurfaced(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _fake_embed)
        m = Mnemos(store=_store(tmp_path), namespace="t",
                   enable_contradiction_detection=False, enable_rerank=False)
        live = m.store_memory("dev", "zeta anchor fact", skip_dedup=True)["id"]
        dead = m.store_memory("dev", "secret archived detail", skip_dedup=True)["id"]
        m.store.store_link(live, dead, "relates", 0.9)
        m.delete(dead)  # soft delete -> archived
        res = m.search("zeta anchor", search_mode="fts", include_linked=True)
        assert res["count"] == 1
        assert "linked_memories" not in res["results"][0]


class TestContradictionRerankGuard:
    def test_unscored_rerank_writes_no_links(self, tmp_path, monkeypatch):
        store = _store(tmp_path)
        existing = store.store_memory(
            Memory(namespace="t", project="dev", content="the sky is blue"),
            embedding=_vec(0.001),
        )
        m = Mnemos(store=store, namespace="t",
                   enable_contradiction_detection=True, enable_rerank=True)
        # rerank degrades by returning docs untouched (no _rerank_score)
        monkeypatch.setattr(core_mod, "rerank", lambda q, docs: docs)
        new_id = store.store_memory(
            Memory(namespace="t", project="dev", content="the sky is green"),
            embedding=_vec(0.001),
        )
        warnings = m._detect_contradictions(
            new_id, "the sky is green", "dev", _vec(0.001))
        assert warnings == []
        assert store.get_links([existing]) == {}


class TestPrefixAwareStaleness:
    def test_legacy_truncated_hash_verifies_fresh(self, tmp_path):
        store = _store(tmp_path)
        canonical = prep_memory_text("dev", "eta stable", "",
                                     mem_type="fact", layer="semantic")
        legacy = text_hash(canonical)[:16]  # pre-v10.6 script wrote prefixes
        store.store_memory(Memory(namespace="t", project="dev", content="eta stable"),
                           embedding=_vec(), text_hash=legacy)
        m = Mnemos(store=store, namespace="t",
                   enable_contradiction_detection=False, enable_rerank=False)
        status = m.embed_status()
        assert status["stale"] == 0
        assert status["unverified"] == 0


class TestNyxNamespaceInheritance:
    def test_apply_merge_inherits_active_namespace(self, tmp_path, monkeypatch):
        import mnemos.consolidation.phases as phases
        monkeypatch.setenv("MNEMOS_NAMESPACE", "t2")
        monkeypatch.setattr(phases, "fastembed_embed", lambda texts, prefix="passage": [_vec()])
        store = _store(tmp_path)
        ids = [
            store.store_memory(Memory(namespace="t2", project="dev", content=c))
            for c in ("one fact", "two fact")
        ]
        mem_by_id = {
            mid: {"project": "dev", "tags": "x", "importance": 5,
                  "consolidation_lock": 0, "verified": 0, "type": "fact",
                  "last_confirmed": None}
            for mid in ids
        }
        conn = store._get_conn()
        new_id = phases.apply_merge(conn, ids, "merged fact", mem_by_id)
        assert new_id is not None
        row = conn.execute("SELECT namespace, status FROM memories WHERE id=?",
                           (new_id,)).fetchone()
        assert row["namespace"] == "t2"  # previously landed in 'default',
        # invisible to every namespace-filtered search while its sources
        # were archived: silent memory attrition
        archived = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE id IN (?, ?) AND status='archived'",
            ids).fetchone()[0]
        assert archived == 2
