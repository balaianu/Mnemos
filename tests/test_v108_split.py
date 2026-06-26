"""v10.8.0: store-path size-guard splits oversized memories into atomic siblings."""

import pytest

from mnemos.core import Mnemos
from mnemos.storage.sqlite_store import SQLiteStore
from mnemos.splitter import _nonblank_lines


@pytest.fixture
def m(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")
    return Mnemos(store=store, namespace="t", enable_rerank=False,
                  enable_contradiction_detection=False)


def _blob(n_blocks):
    blocks = []
    for b in range(n_blocks):
        lines = [f"## Topic {b}"]
        for i in range(5):
            lines.append(f"F:fact {b}.{i}; detail about subject {b} item {i} with text to add length")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def test_small_content_single_store(m):
    res = m.store_memory("test", "F:one small fact; nothing to split")
    assert res["status"] == "stored"
    assert "id" in res


def test_oversized_splits_into_siblings(m):
    blob = _blob(60)
    assert len(blob) > 4000
    res = m.store_memory("test", blob, tags="janne,care")
    assert res["status"] == "stored-split"
    assert res["parts"] > 1
    ids = res["ids"]
    assert len(ids) == res["parts"]

    # Each child is within the target band (or a single un-splittable line).
    by_id = m.store.get_memories_by_ids(ids)
    for cid in ids:
        c = by_id[cid].content
        assert len(c) <= 2800 or len(_nonblank_lines(c)) == 1

    # Lossless: every fact line preserved exactly, in order, across children.
    rebuilt = [ln for cid in ids for ln in _nonblank_lines(by_id[cid].content)]
    assert rebuilt == _nonblank_lines(blob)

    # Children carry the original tags plus the split markers.
    assert "janne" in by_id[ids[0]].tags
    assert "nyx-split" in by_id[ids[0]].tags

    # Siblings are chained with 'related' links (walkable cluster).
    links = m.store.get_links(ids)
    linked_pairs = {
        (mid, l["linked_id"]) for mid, ls in links.items() for l in ls
    }
    assert any((ids[i], ids[i + 1]) in linked_pairs or
               (ids[i + 1], ids[i]) in linked_pairs for i in range(len(ids) - 1))


def test_consolidation_lock_not_split(m):
    blob = _blob(60)
    res = m.store_memory("test", blob, consolidation_lock=True)
    # Locked prose is intentionally kept whole.
    assert res["status"] == "stored"
    assert "id" in res


def _insert_legacy_oversized(m, blob, tags="legacy"):
    """Insert an oversized memory raw, bypassing the size-guarded store path,
    to simulate pre-v10.8.0 bloat for the remediation backfill."""
    conn = m.store._get_conn()
    cur = conn.execute(
        "INSERT INTO memories (namespace, project, content, tags, importance, type, layer) "
        "VALUES (?, 'test', ?, ?, 5, 'fact', 'semantic')",
        (m.namespace, blob, tags),
    )
    conn.commit()
    return cur.lastrowid


def test_remediate_oversized_backfill(m):
    blob = _blob(60)
    oid = _insert_legacy_oversized(m, blob)

    res = m.remediate_oversized(min_size=4000)
    assert res["split"] >= 1
    assert res["children_created"] > res["split"]
    assert res["archived"] >= 1
    assert res["errors"] == 0

    # Original archived, not deleted.
    assert m.store.get_memory(oid, increment_access=False).status == "archived"

    # Children exist (active), in order, lossless, tagged with provenance.
    conn = m.store._get_conn()
    children = conn.execute(
        "SELECT content, tags FROM memories WHERE status='active' AND tags LIKE ? ORDER BY id",
        (f"%split-from:#{oid}%",),
    ).fetchall()
    assert len(children) > 1
    rebuilt = [ln for c in children for ln in _nonblank_lines(c["content"])]
    assert rebuilt == _nonblank_lines(blob)
    for c in children:
        assert len(c["content"]) <= 2800 or len(_nonblank_lines(c["content"])) == 1


def test_remediate_dry_run_changes_nothing(m):
    blob = _blob(60)
    oid = _insert_legacy_oversized(m, blob)
    res = m.remediate_oversized(min_size=4000, dry_run=True)
    assert res["split"] >= 1
    assert res["archived"] == 0
    # Original untouched.
    assert m.store.get_memory(oid, increment_access=False).status == "active"
