"""Follow-ups from the 10.39.0 review: link failures never abort a store,
namespace scoping lives in the statements, the implicit-rowid archive
schema has coverage."""

import pytest

import mnemos.embed as embed_mod
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


def link_rows(store):
    return store.raw_connection().execute("SELECT COUNT(*) FROM memory_links").fetchone()[0]


@pytest.mark.parametrize("endpoint", ["missing", "foreign"])
def test_store_link_on_bad_endpoint_returns_false_and_writes_nothing(stores, endpoint):
    alice, bob = stores
    own = put(alice)
    other = 999999 if endpoint == "missing" else put(bob)
    assert alice.store_link(own, other, "related", 0.6) is False
    assert link_rows(alice) == 0


def test_store_link_on_valid_endpoints_returns_true(stores):
    alice, _ = stores
    a, b = put(alice), put(alice)
    assert alice.store_link(a, b, "related", 0.6) is True
    assert link_rows(alice) == 1


def test_reembed_mismatched_skips_foreign_ids(stores, monkeypatch):
    alice, bob = stores
    foreign = put(bob, content="theirs", vector=vector(bob))
    before = bob.raw_connection().execute(
        "SELECT text_hash FROM embed_meta WHERE source_id = ?", (foreign,)).fetchone()[0]
    calls = []
    monkeypatch.setattr(embed_mod, "embed",
                        lambda texts, **kw: calls.extend(texts) or [vector(bob, 0.5)])
    assert alice.reembed_mismatched("alice", [foreign]) == 0
    assert calls == []
    after = bob.raw_connection().execute(
        "SELECT text_hash FROM embed_meta WHERE source_id = ?", (foreign,)).fetchone()[0]
    assert after == before


@pytest.mark.parametrize("hard", [False, True])
def test_foreign_delete_leaves_links_and_row(stores, hard):
    alice, bob = stores
    a, b = put(bob), put(bob)
    bob.store_link(a, b, "related", 0.6)
    assert alice.delete_memory(a, hard=hard) is False
    assert link_rows(bob) == 1
    assert bob.get_memory(a, False).status == "active"


def test_archived_knn_prefilters_on_implicit_rowid_schema(stores):
    store, _ = stores
    dims = store.get_vec_dims()
    conn = store.raw_connection()
    conn.execute("DROP TABLE embed_vec_arch")
    conn.execute(f"CREATE VIRTUAL TABLE embed_vec_arch USING vec0(embedding float[{dims}])")
    conn.commit()
    query = vector(store)
    for _ in range(8):
        store.delete_memory(put(store, project="other", vector=query))
    target = put(store, project="wanted", vector=vector(store, 0.1))
    store.delete_memory(target)
    assert store.archived_embed_count() == 9
    results = store.search_vec_archived(query, project="wanted", limit=1)
    assert [mid for mid, _ in results] == [target]
    assert results[0][1] == pytest.approx(0.1)
