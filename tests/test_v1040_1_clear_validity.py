"""10.40.1: validity can be cleared again. With valid_only defaulting to True,
a memory whose valid_until was set by mistake, or by a Phase 4 verdict later
judged wrong, silently vanished from every default search, and neither the
MCP tool nor the CLI offered a way to null the column: both dropped None."""

import argparse
import json

import pytest

from mnemos.core import Mnemos
from mnemos.mcp_server import TOOL_DEFINITIONS, tool_update
from mnemos.storage.base import Memory
from mnemos.storage.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(db_path=str(tmp_path / "memory.db"), namespace="alice")
    s.init_schema()
    yield s
    s.close()


def put(store, **fields):
    return store.store_memory(Memory(namespace=store.namespace, project="test",
                                     content="unique anchor", **fields))


def agent(store):
    return Mnemos(store=store, namespace=store.namespace, enable_rerank=False,
                  enable_contradiction_detection=False)


def row(store, mid):
    return store.get_memory(mid, increment_access=False)


@pytest.mark.parametrize("field", ["valid_until", "valid_from", "subcategory"])
@pytest.mark.parametrize("empty", [None, ""])
def test_mcp_update_null_or_empty_clears_a_nullable_field(store, field, empty):
    mid = put(store, valid_until="2000-01-01", valid_from="1999-01-01", subcategory="x")
    result = tool_update(agent(store), {"id": mid, field: empty})
    assert result["status"] == "updated"
    assert getattr(row(store, mid), field) is None


def test_mcp_update_omitted_keys_are_left_alone(store):
    mid = put(store, valid_until="2000-01-01")
    tool_update(agent(store), {"id": mid, "importance": 8})
    assert row(store, mid).valid_until == "2000-01-01"


def test_clearing_valid_until_brings_the_memory_back_into_default_search(store):
    mid = put(store, valid_until="2000-01-01")
    m = agent(store)
    assert m.search("unique anchor", search_mode="fts")["results"] == []
    tool_update(m, {"id": mid, "valid_until": None})
    assert [r["id"] for r in m.search("unique anchor", search_mode="fts")["results"]] == [mid]


def test_mcp_schema_admits_null_for_the_nullable_fields():
    schema = next(t for t in TOOL_DEFINITIONS if t["name"] == "memory_update")
    for field in ("valid_until", "valid_from", "subcategory"):
        assert "null" in schema["inputSchema"]["properties"][field]["type"]


def test_cli_clear_flag(store, capsys):
    from mnemos.cli import cmd_update

    mid = put(store, valid_until="2000-01-01", subcategory="x")
    cmd_update(agent(store), argparse.Namespace(id=mid, clear=["valid_until", "subcategory"]))
    assert json.loads(capsys.readouterr().out)["status"] == "updated"
    assert row(store, mid).valid_until is None
    assert row(store, mid).subcategory is None


def test_cli_clear_and_set_in_one_call(store, capsys):
    from mnemos.cli import cmd_update

    mid = put(store, valid_until="2000-01-01")
    cmd_update(agent(store), argparse.Namespace(id=mid, clear=["valid_until"], importance=9))
    assert row(store, mid).valid_until is None
    assert row(store, mid).importance == 9
