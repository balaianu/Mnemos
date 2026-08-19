"""Regression: the Nyx tag amplifier (v10.35.1).

Merge unioned tags unbounded, and the size-guard split copied the full union
to every sibling: a merge of N memories split into k pieces produced k
memories each claiming all N topics while holding 1/k of the content --
compounding nightly as unions merged with unions (field-measured: a 275-char
sibling carrying 6KB of tags; 77-tag strings byte-identical across sibling
groups). The v10.8.0 size guard measured content only, so it never saw them.

The rule now: a chunk keeps a tag only when its own content supports it or
every cluster source carried it (universal). Universal matters: a pure
content-support filter strips privacy markers like STRICTLY-PRIVATE that
legitimately ride on every source without appearing in any content.
"""
import pytest

import mnemos.consolidation.phases as phases
from mnemos.constants import FASTEMBED_DIMS as DIMS
from mnemos.storage.base import Memory
from mnemos.storage.sqlite_store import SQLiteStore


def _store(tmp_path, name="m.db"):
    return SQLiteStore(db_path=str(tmp_path / name), namespace="t")


def _vec(seed=0.001):
    return [seed] * DIMS


# --- unit: the filter itself -------------------------------------------------

def test_supported_tag_survives_unsupported_dies():
    kept = phases.filter_chunk_tags(
        "F:postgres chosen over mysql for the api",
        {"postgres", "chunk-3-of-479", "kubernetes"},
        universal=set(),
    )
    assert "postgres" in kept
    assert "chunk-3-of-479" not in kept
    assert "kubernetes" not in kept


def test_universal_tag_survives_without_content_support():
    kept = phases.filter_chunk_tags(
        "F:she prefers the morning window",
        {"STRICTLY-PRIVATE", "layla", "spa-weekend-postmortem"},
        universal={"STRICTLY-PRIVATE", "layla"},
    )
    assert "STRICTLY-PRIVATE" in kept
    assert "layla" in kept
    assert "spa-weekend-postmortem" not in kept


def test_multiword_tag_needs_all_terms():
    kept = phases.filter_chunk_tags(
        "F:the migration finished and was confirmed",
        {"migration-confirmed", "migration-rejected"},
        universal=set(),
    )
    assert kept == ["migration-confirmed"]


def test_budget_bounds_tag_string_deterministically():
    tags = {f"verylongtagnumber-{i:04d}" for i in range(400)}
    # make every tag supported so only the budget can stop them
    content = " ".join(t.replace("-", " ") for t in sorted(tags))
    kept = phases.filter_chunk_tags(content, tags, universal=set())
    joined = ",".join(kept)
    assert len(joined) <= phases.TAG_BUDGET
    # deterministic: same inputs, same output
    assert kept == phases.filter_chunk_tags(content, tags, universal=set())


def test_universal_tags_win_the_budget():
    tags = {f"filler-tag-{i:04d}" for i in range(400)} | {"STRICTLY-PRIVATE"}
    content = " ".join(t.replace("-", " ") for t in sorted(tags))
    kept = phases.filter_chunk_tags(content, tags,
                                    universal={"STRICTLY-PRIVATE"})
    assert "STRICTLY-PRIVATE" in kept


# --- integration: apply_merge no longer amplifies ----------------------------

def test_split_siblings_carry_only_their_own_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOS_NAMESPACE", "t")
    monkeypatch.setattr(phases, "fastembed_embed",
                        lambda texts, prefix="passage": [_vec() for _ in texts])
    # force a split: tiny threshold
    monkeypatch.setattr(phases, "needs_split", lambda c: True)
    monkeypatch.setattr(phases, "split_content", lambda c: c.split("\n"))
    store = _store(tmp_path)
    ids = [
        store.store_memory(Memory(namespace="t", project="dev", content=c))
        for c in ("F:alpha subsystem uses postgres",
                  "F:beta subsystem uses redis")
    ]
    mem_by_id = {
        ids[0]: {"project": "dev", "tags": "STRICTLY-PRIVATE,postgres,alpha",
                 "importance": 5, "consolidation_lock": 0, "verified": 0,
                 "type": "fact", "last_confirmed": None},
        ids[1]: {"project": "dev", "tags": "STRICTLY-PRIVATE,redis,beta",
                 "importance": 5, "consolidation_lock": 0, "verified": 0,
                 "type": "fact", "last_confirmed": None},
    }
    conn = store._get_conn()
    merged = "F:alpha subsystem uses postgres\nF:beta subsystem uses redis"
    new_id = phases.apply_merge(conn, ids, merged, mem_by_id)
    assert new_id is not None

    rows = conn.execute(
        "SELECT content, tags FROM memories WHERE tags LIKE '%split-part%'"
    ).fetchall()
    assert len(rows) >= 2
    for content, tags in rows:
        tagset = {t.strip() for t in tags.split(",")}
        assert "STRICTLY-PRIVATE" in tagset          # universal survives
        assert "consolidated" in tagset
        if "postgres" in content:
            assert "postgres" in tagset
            assert "redis" not in tagset             # no sibling union
        if "redis" in content:
            assert "redis" in tagset
            assert "postgres" not in tagset
