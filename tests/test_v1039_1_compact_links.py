"""Compact link output: Nyx audit relations stay out of results unless asked for."""

import argparse
import json

import pytest

import mnemos.core as core_mod
from mnemos.core import Mnemos
from mnemos.mcp_server import TOOL_DEFINITIONS, tool_search
from mnemos.storage.base import Memory
from mnemos.storage.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(db_path=str(tmp_path / "memory.db"), namespace="alice")
    s.init_schema()
    yield s
    s.close()


def put(store, content="fixture", **fields):
    return store.store_memory(
        Memory(namespace=store.namespace, project="test", content=content, **fields)
    )


def agent(store):
    return Mnemos(store=store, namespace=store.namespace, enable_rerank=False,
                  enable_contradiction_detection=False)


@pytest.fixture
def graph(store):
    """root <-cleared- cleared ; root -related-> related -cleared-> tail"""
    root = put(store, "unique anchor")
    cleared = put(store, "judged compatible by nyx")
    related = put(store, "genuinely related")
    tail = put(store, "only reachable through an audit row")
    store.store_link(cleared, root, "contradiction-cleared", 0.1)
    store.store_link(root, related, "related", 0.6)
    store.store_link(related, tail, "contradiction-cleared", 0.1)
    return root, cleared, related, tail


def test_store_get_links_hides_audit_relations_by_default(store, graph):
    root, cleared, related, _ = graph
    assert [l["linked_id"] for l in store.get_links([root])[root]] == [related]
    assert store.get_links([cleared]) == {}


def test_store_get_links_include_audit_returns_everything(store, graph):
    root, cleared, related, _ = graph
    links = store.get_links([root], include_audit=True)[root]
    assert {l["linked_id"] for l in links} == {cleared, related}
    audit = store.get_links([cleared], include_audit=True)[cleared]
    assert audit[0]["relation"] == "contradiction-cleared"


def test_search_result_links_omit_audit_rows_by_default(store, graph):
    root, cleared, related, _ = graph
    hit = agent(store).search("unique anchor", search_mode="fts")["results"][0]
    assert [l["linked_id"] for l in hit["links"]] == [related]


def test_search_include_audit_links_restores_them(store, graph):
    root, cleared, related, _ = graph
    hit = agent(store).search("unique anchor", search_mode="fts",
                              include_audit_links=True)["results"][0]
    assert {l["linked_id"] for l in hit["links"]} == {cleared, related}


def test_search_result_with_only_audit_links_has_no_links_key(store):
    root = put(store, "unique anchor")
    other = put(store, "other")
    store.store_link(other, root, "contradiction-cleared", 0.1)
    hit = agent(store).search("unique anchor", search_mode="fts")["results"][0]
    assert "links" not in hit


def test_linked_expansion_does_not_traverse_audit_rows(store, graph):
    root, cleared, related, tail = graph
    m = agent(store)
    hit = m.search("unique anchor", search_mode="fts", include_linked=True,
                   linked_depth=2)["results"][0]
    assert [s["id"] for s in hit["linked_memories"]] == [related]
    hit = m.search("unique anchor", search_mode="fts", include_linked=True,
                   linked_depth=2, include_audit_links=True)["results"][0]
    assert {s["id"] for s in hit["linked_memories"]} == {cleared, related, tail}


def test_remediation_still_copies_audit_links_to_children(store, monkeypatch):
    blob = "\n".join(f"F:unique item {i} " + "detail " * 12 for i in range(60))
    parent = put(store, blob)
    judged = put(store, "judged compatible with the parent")
    store.store_link(judged, parent, "contradiction-cleared", 0.1)
    dims = store.get_vec_dims()
    monkeypatch.setattr(core_mod, "embed",
                        lambda texts, **kw: [[1.0] + [0.0] * (dims - 1) for _ in texts])
    result = agent(store).remediate_oversized(min_size=4000)
    assert result["errors"] == 0
    child = store.raw_connection().execute(
        "SELECT id FROM memories WHERE tags LIKE ? ORDER BY id LIMIT 1",
        (f"%split-from:#{parent}%",),
    ).fetchone()[0]
    links = store.get_links([child], include_audit=True)[child]
    assert any(l["relation"] == "contradiction-cleared" and l["source_id"] == judged
               for l in links)


def test_mcp_search_exposes_include_audit_links(store, graph):
    root, cleared, related, _ = graph
    hit = tool_search(agent(store), {"query": "unique anchor", "search_mode": "fts",
                                     "include_audit_links": True})["results"][0]
    assert {l["linked_id"] for l in hit["links"]} == {cleared, related}
    schema = next(t for t in TOOL_DEFINITIONS if t["name"] == "memory_search")
    assert schema["inputSchema"]["properties"]["include_audit_links"]["default"] is False


def cli_args(**overrides):
    base = dict(
        query="unique anchor", project=None, subcategory=None, type=None, layer=None,
        valid_only=False, mode="fts", limit=20, expand_merged=False, snippet_chars=None,
        include_linked=False, audit_links=False, json=True,
    )
    return argparse.Namespace(**{**base, **overrides})


def test_cli_search_hides_audit_links_by_default(store, graph, capsys):
    from mnemos.cli import cmd_search

    root, cleared, related, _ = graph
    cmd_search(agent(store), cli_args())
    hit = json.loads(capsys.readouterr().out)["results"][0]
    assert [l["linked_id"] for l in hit["links"]] == [related]


def test_cli_search_audit_links_flag(store, graph, capsys):
    from mnemos.cli import cmd_search

    root, cleared, related, _ = graph
    cmd_search(agent(store), cli_args(audit_links=True))
    hit = json.loads(capsys.readouterr().out)["results"][0]
    assert {l["linked_id"] for l in hit["links"]} == {cleared, related}
