"""Memory correctness regressions using only synthetic, temporary stores."""

from datetime import datetime

import pytest

import mnemos.core as core_mod
from mnemos.core import Mnemos
from mnemos.embed import prep_memory_text, text_hash
from mnemos.mcp_server import tool_update
from mnemos.storage.base import Memory
from mnemos.storage.sqlite_store import SQLiteStore


@pytest.fixture
def stores(tmp_path):
    path = str(tmp_path / "memory.db")
    alice = SQLiteStore(db_path=path, namespace="alice")
    bob = SQLiteStore(db_path=path, namespace="bob")
    alice.init_schema()
    bob.init_schema()
    yield alice, bob
    alice.close()
    bob.close()


def put(store, content="fixture", project="test", vector=None, **fields):
    return store.store_memory(
        Memory(namespace=store.namespace, project=project, content=content, **fields),
        embedding=vector,
    )


def vector(store, offset=0.0):
    value = [0.0] * store.get_vec_dims()
    value[0], value[1] = 1.0, offset
    return value


def agent(store):
    return Mnemos(store=store, namespace=store.namespace, enable_rerank=False,
                  enable_contradiction_detection=False)


def test_foreign_get_and_bulk_get_do_not_read_or_touch(stores):
    alice, bob = stores
    own = put(alice)
    foreign = put(bob)
    assert alice.get_memory(foreign) is None
    assert "error" in agent(alice).get(foreign)
    assert set(alice.get_memories_by_ids([own, foreign])) == {own}
    untouched = bob.get_memory(foreign, increment_access=False)
    assert untouched.access_count == 0
    assert untouched.last_accessed is None
    snippets = alice.get_snippets([own, foreign], "fixture")
    assert own in snippets
    assert foreign not in snippets


@pytest.mark.parametrize("fields", [{"importance": 9}, {}])
def test_foreign_update_cannot_overwrite_vector(stores, fields):
    alice, bob = stores
    original = vector(bob)
    foreign = put(bob, vector=original)
    assert not alice.update_memory(foreign, fields, embedding=vector(alice, 9))
    assert bob.get_memory(foreign, False).importance == 5
    assert bob.search_vec(original, limit=1) == [(foreign, 0.0)]


@pytest.mark.parametrize("hard", [False, True])
def test_foreign_delete_has_no_side_effects(stores, hard):
    alice, bob = stores
    foreign = put(bob, vector=vector(bob))
    assert not alice.delete_memory(foreign, hard=hard)
    assert not alice.move_embedding_to_archive(foreign)
    assert bob.get_memory(foreign, False).status == "active"
    assert bob.search_vec(vector(bob), limit=1) == [(foreign, 0.0)]
    assert not alice.delete_memory(999999, hard=hard)


def test_cross_namespace_links_and_lineage_are_not_exposed(stores):
    alice, bob = stores
    own, foreign = put(alice), put(bob)
    assert alice.store_link(own, foreign, "relates") is False
    # Existing bad links/lineage must also be contained on read.
    conn = alice.raw_connection()
    conn.execute("INSERT INTO memory_links (source_id, target_id, relation_type, strength) VALUES (?, ?, ?, ?)",
                 (own, foreign, "relates", 0.9))
    conn.commit()
    assert alice.get_links([own, foreign]) == {}
    assert bob.get_links([foreign]) == {}
    alice.store_nyx_insight(own, [foreign], "merge")
    assert alice.get_merged_sources(own) == []
    assert bob.get_merged_sources(own) == []


def test_direction_is_preserved_at_both_endpoints_and_in_expansion(stores):
    store, _ = stores
    older = put(store, "historical instruction")
    newer = put(store, "unique anchor")
    store.store_link(older, newer, "superseded_by", 0.9)
    links = store.get_links([older, newer])
    for mid, other, direction in [(older, newer, "outgoing"), (newer, older, "incoming")]:
        assert links[mid] == [{
            "linked_id": other, "relation": "superseded_by", "strength": 0.9,
            "source_id": older, "target_id": newer, "direction": direction,
        }]
    result = agent(store).search("unique anchor", search_mode="fts", include_linked=True)
    summary = result["results"][0]["linked_memories"][0]
    assert summary["direction"] == "incoming"
    assert summary["source_id"] == older
    assert summary["target_id"] == newer


@pytest.mark.parametrize("boundary", ["expired", "future", "today"])
def test_valid_only_blocks_invalid_link_and_transitive_bridge(stores, boundary):
    store, _ = stores
    fields = {
        "expired": {"valid_until": "2000-01-01"},
        "future": {"valid_from": "2999-01-01"},
        "today": {"valid_until": datetime.now().date().isoformat()},
    }[boundary]
    root = put(store, "unique anchor")
    invalid = put(store, "old instruction", **fields)
    tail = put(store, "valid but only reachable through old instruction")
    store.store_link(root, invalid, "relates")
    store.store_link(invalid, tail, "relates")
    m = agent(store)
    result = m.search("unique anchor", search_mode="fts", include_linked=True,
                      linked_depth=2, valid_only=True)
    assert "linked_memories" not in result["results"][0]
    historical = m.search("unique anchor", search_mode="fts", include_linked=True,
                          linked_depth=2, valid_only=False)
    assert {r["id"] for r in historical["results"][0]["linked_memories"]} == {invalid, tail}


@pytest.mark.parametrize("archived", [False, True])
@pytest.mark.parametrize("distractor_kind", ["namespace", "project", "expired", "future"])
def test_filtered_knn_keeps_eligible_hits(stores, archived, distractor_kind):
    store, bob = stores
    query = vector(store)
    for _ in range(8):
        owner = bob if distractor_kind == "namespace" else store
        fields = {"project": "wanted"}
        if distractor_kind == "project":
            fields["project"] = "other"
        elif distractor_kind == "expired":
            fields["valid_until"] = "2000-01-01"
        elif distractor_kind == "future":
            fields["valid_from"] = "2999-01-01"
        mid = put(owner, vector=query, **fields)
        if archived:
            owner.delete_memory(mid)
    target = put(store, project="wanted", vector=vector(store, 0.1))
    if archived:
        store.delete_memory(target)
    search = store.search_vec_archived if archived else store.search_vec
    results = search(query, project="wanted", valid_only=True, limit=1)
    assert [mid for mid, _ in results] == [target]
    assert results[0][1] == pytest.approx(0.1)
    assert search(query, project="absent", valid_only=True, limit=1) == []


def test_reads_do_not_confirm_but_explicit_confirmation_does(stores):
    store, _ = stores
    mid = put(store)
    m = agent(store)
    assert m.get(mid).get("last_confirmed") is None
    assert m.get(mid)["access_count"] == 2
    store.update_memory(mid, {"last_confirmed": "2000-01-01 00:00:00"})
    assert m.get(mid)["last_confirmed"] == "2000-01-01 00:00:00"
    result = tool_update(m, {"id": mid, "confirmed": True})
    assert result["status"] == "updated"
    confirmed = store.get_memory(mid, False).last_confirmed
    assert confirmed.startswith(datetime.now().date().isoformat())
    assert m.get(mid)["last_confirmed"] == confirmed


def test_project_update_reembeds_and_updates_provenance(stores, monkeypatch):
    store, _ = stores
    mid = put(store, content="fact", project="before", vector=vector(store))
    calls = []

    def embed(texts, prefix):
        calls.extend(texts)
        return [vector(store, 0.2)]

    monkeypatch.setattr(core_mod, "embed", embed)
    result = agent(store).update(mid, project="after")
    expected = prep_memory_text("after", "fact")
    assert result["embedded"] is True
    assert calls == [expected]
    row = store.raw_connection().execute(
        "SELECT text_hash FROM embed_meta WHERE source_id = ?", (mid,),
    ).fetchone()
    assert row[0] == text_hash(expected)


def test_project_update_reports_failed_embedding(stores, monkeypatch):
    store, _ = stores
    mid = put(store, vector=vector(store))
    monkeypatch.setattr(core_mod, "embed", lambda *a, **kw: [])
    result = agent(store).update(mid, project="new")
    assert result["embedded"] is False
    assert "stale" in result["warning"]


def test_remediation_preserves_incoming_and_outgoing_direction(stores, monkeypatch):
    store, _ = stores
    blob = "\n".join(f"F:unique item {i} " + "detail " * 12 for i in range(60))
    parent = put(store, blob)
    earlier, later = put(store, "earlier"), put(store, "later")
    store.store_link(earlier, parent, "superseded_by", 0.9)
    store.store_link(parent, later, "informs", 0.7)
    monkeypatch.setattr(core_mod, "embed",
                        lambda texts, **kw: [vector(store) for _ in texts])
    result = agent(store).remediate_oversized(min_size=4000)
    assert result["errors"] == 0
    assert result["split"] == 1
    conn = store.raw_connection()
    child = conn.execute(
        "SELECT id FROM memories WHERE tags LIKE ? ORDER BY id LIMIT 1",
        (f"%split-from:#{parent}%",),
    ).fetchone()[0]
    links = store.get_links([child])[child]
    assert any(l["source_id"] == earlier and l["target_id"] == child
               and l["relation"] == "superseded_by" for l in links)
    assert any(l["source_id"] == child and l["target_id"] == later
               and l["relation"] == "informs" for l in links)


def test_cli_confirmation(stores, capsys):
    import argparse
    import json
    from mnemos.cli import cmd_update

    store, _ = stores
    mid = put(store)
    cmd_update(agent(store), argparse.Namespace(id=mid, confirm=True))
    assert json.loads(capsys.readouterr().out)["status"] == "updated"
    assert store.get_memory(mid, False).last_confirmed is not None
