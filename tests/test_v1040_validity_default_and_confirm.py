"""10.40.0: current validity is the default at every search surface, and a
content correction or a verified flag through update counts as confirmation."""

import argparse
import json
from datetime import datetime

import pytest

from mnemos.core import Mnemos
from mnemos.mcp_server import TOOL_DEFINITIONS, tool_search, tool_update
from mnemos.storage.base import Memory
from mnemos.storage.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(db_path=str(tmp_path / "memory.db"), namespace="alice")
    s.init_schema()
    yield s
    s.close()


def put(store, content="fixture", vector=None, **fields):
    return store.store_memory(
        Memory(namespace=store.namespace, project="test", content=content, **fields),
        embedding=vector,
    )


def agent(store):
    return Mnemos(store=store, namespace=store.namespace, enable_rerank=False,
                  enable_contradiction_detection=False)


def vector(store, offset=0.0):
    value = [0.0] * store.get_vec_dims()
    value[0], value[1] = 1.0, offset
    return value


@pytest.fixture
def expired_and_current(store):
    expired = put(store, "unique anchor, old instruction", valid_until="2000-01-01")
    current = put(store, "unique anchor, current instruction")
    return expired, current


def ids(result):
    return {r["id"] for r in result["results"]}


# --- validity default ---

def test_search_hides_expired_by_default(store, expired_and_current):
    expired, current = expired_and_current
    assert ids(agent(store).search("unique anchor", search_mode="fts")) == {current}


def test_search_valid_only_false_returns_history(store, expired_and_current):
    expired, current = expired_and_current
    result = agent(store).search("unique anchor", search_mode="fts", valid_only=False)
    assert ids(result) == {expired, current}


def test_mcp_search_default_and_schema(store, expired_and_current):
    expired, current = expired_and_current
    m = agent(store)
    assert ids(tool_search(m, {"query": "unique anchor", "search_mode": "fts"})) == {current}
    assert ids(tool_search(m, {"query": "unique anchor", "search_mode": "fts",
                               "valid_only": False})) == {expired, current}
    schema = next(t for t in TOOL_DEFINITIONS if t["name"] == "memory_search")
    assert schema["inputSchema"]["properties"]["valid_only"]["default"] is True


def cli_args(**overrides):
    base = dict(
        query="unique anchor", project=None, subcategory=None, type=None, layer=None,
        include_expired=False, mode="fts", limit=20, expand_merged=False,
        snippet_chars=None, include_linked=False, audit_links=False, json=True,
    )
    return argparse.Namespace(**{**base, **overrides})


def test_cli_search_default_and_include_expired(store, expired_and_current, capsys):
    from mnemos.cli import cmd_search

    expired, current = expired_and_current
    m = agent(store)
    cmd_search(m, cli_args())
    assert ids(json.loads(capsys.readouterr().out)) == {current}
    cmd_search(m, cli_args(include_expired=True))
    assert ids(json.loads(capsys.readouterr().out)) == {expired, current}


def test_store_level_defaults_agree(store):
    expired = put(store, "unique anchor", vector=vector(store), valid_until="2000-01-01")
    assert store.search_fts("unique anchor") == []
    assert store.search_vec(vector(store)) == []
    assert [mid for mid, _ in store.search_vec(vector(store), valid_only=False)] == [expired]
    store.delete_memory(expired)
    assert store.search_vec_archived(vector(store)) == []
    assert [mid for mid, _ in store.search_vec_archived(vector(store), valid_only=False)] == [expired]


# --- confirmation producer ---

def confirmed_at(store, mid):
    return store.get_memory(mid, increment_access=False).last_confirmed


def test_content_update_confirms(store):
    mid = put(store)
    agent(store).update(mid, content="corrected after checking the source")
    assert confirmed_at(store, mid).startswith(datetime.now().date().isoformat())


def test_metadata_update_does_not_confirm(store):
    mid = put(store)
    agent(store).update(mid, tags="retagged", importance=8, project="moved")
    assert confirmed_at(store, mid) is None


def test_content_update_with_confirmed_false_stays_silent(store):
    mid = put(store)
    agent(store).update(mid, content="mechanical rewrite", confirmed=False)
    assert confirmed_at(store, mid) is None


def test_verified_true_confirms(store):
    mid = put(store)
    agent(store).update(mid, verified=True)
    assert confirmed_at(store, mid) is not None
    assert store.get_memory(mid, False).verified == 1


def test_mcp_update_accepts_verified_and_confirms(store):
    mid = put(store)
    result = tool_update(agent(store), {"id": mid, "verified": True})
    assert result["status"] == "updated"
    assert confirmed_at(store, mid) is not None
    schema = next(t for t in TOOL_DEFINITIONS if t["name"] == "memory_update")
    assert "verified" in schema["inputSchema"]["properties"]


def test_bulk_rewrite_does_not_confirm(store):
    mid = put(store, "old-term appears here")
    m = agent(store)
    m.bulk_rewrite(pattern="old-term", replacement="new-term", dry_run=False)
    assert "new-term" in store.get_memory(mid, False).content
    assert confirmed_at(store, mid) is None
