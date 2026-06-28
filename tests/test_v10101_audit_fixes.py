"""Regression tests for v10.10.1: the post-v10.10.0 audit fixes.

Four findings from the 2026-06-28 read-through audit of the WAL-safe-backup work:
  1. [MED] no busy_timeout -> cross-process write contention raises SQLITE_BUSY
     instead of waiting (MCP store + Nyx consolidation + the now-prod-default CLI).
  2. [MED] backup() removed the destination BEFORE writing the new snapshot, so a
     failed VACUUM left you with nothing (the one thing a backup tool must never do).
  3. [LOW] doctor read only quick_check.fetchone()[0], hiding the extent of corruption.
  4. [LOW] backup() didn't abspath its dest or create a missing parent dir; no
     runtime "this looks corrupt, run doctor" hint on the search hot path.
"""
import os
import sqlite3

import pytest

import mnemos.core as core_mod
from mnemos.core import Mnemos, _summarize_quick_check, _corruption_hint
from mnemos.storage.sqlite_store import SQLiteStore

DIMS = 1024


def _fake_embed(texts, prefix="passage"):
    return [[0.001] * DIMS for _ in texts]


def _m(tmp_path, monkeypatch):
    monkeypatch.setattr(core_mod, "embed", _fake_embed)
    store = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")
    return Mnemos(store=store, namespace="t", enable_rerank=False)


# --- 1. busy_timeout ---

class TestBusyTimeout:
    def test_busy_timeout_is_set(self, tmp_path, monkeypatch):
        m = _m(tmp_path, monkeypatch)
        conn = m.store._get_conn()
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


# --- 2 + 4. atomic, dir-creating, abspath backup ---

class TestBackupHardening:
    def test_overwrites_existing_dest_no_tmp_left(self, tmp_path, monkeypatch):
        m = _m(tmp_path, monkeypatch)
        m.store_memory(project="dev", content="F: one", skip_dedup=True)
        dest = str(tmp_path / "snap.db")
        with open(dest, "w") as f:
            f.write("STALE")  # a pre-existing backup we are replacing
        m.store.backup(dest)
        snap = sqlite3.connect(dest)
        try:
            assert snap.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        finally:
            snap.close()
        assert not os.path.exists(dest + ".tmp")  # no leftover temp

    def test_preserves_prior_backup_on_failure(self, tmp_path, monkeypatch):
        m = _m(tmp_path, monkeypatch)
        m.store_memory(project="dev", content="F: one", skip_dedup=True)
        dest = str(tmp_path / "good.db")
        with open(dest, "w") as f:
            f.write("PRIOR-GOOD-BACKUP")

        real = m.store._get_conn()

        class FailingVacuum:
            def commit(self):
                return real.commit()
            def execute(self, sql, *a):
                if sql.strip().upper().startswith("VACUUM"):
                    raise sqlite3.OperationalError("disk I/O error")
                return real.execute(sql, *a)

        monkeypatch.setattr(m.store, "_get_conn", lambda: FailingVacuum())
        with pytest.raises(sqlite3.OperationalError):
            m.store.backup(dest)
        # The prior backup must survive a failed run, untouched.
        with open(dest) as f:
            assert f.read() == "PRIOR-GOOD-BACKUP"
        assert not os.path.exists(dest + ".tmp")

    def test_creates_missing_parent_dir(self, tmp_path, monkeypatch):
        m = _m(tmp_path, monkeypatch)
        m.store_memory(project="dev", content="F: one", skip_dedup=True)
        dest = str(tmp_path / "nested" / "deeper" / "snap.db")
        m.store.backup(dest)
        assert os.path.exists(dest)

    def test_returns_absolute_path_for_relative_dest(self, tmp_path, monkeypatch):
        m = _m(tmp_path, monkeypatch)
        m.store_memory(project="dev", content="F: one", skip_dedup=True)
        monkeypatch.chdir(tmp_path)
        out = m.store.backup("rel-snap.db")
        assert os.path.isabs(out)
        assert out == str(tmp_path / "rel-snap.db")
        assert os.path.exists(out)


# --- 3. quick_check summary ---

class TestQuickCheckSummary:
    def test_ok_single_row(self):
        assert _summarize_quick_check([("ok",)]) == (True, "ok")

    def test_empty_treated_as_ok(self):
        assert _summarize_quick_check([]) == (True, "ok")

    def test_multiple_errors_summarized_with_count(self):
        rows = [("err a",), ("err b",), ("err c",), ("err d",), ("err e",)]
        ok, summary = _summarize_quick_check(rows)
        assert ok is False
        assert "err a" in summary
        assert "+2 more" in summary  # 5 rows, 3 shown


# --- 4c. corruption hint ---

class TestCorruptionHint:
    def test_malformed_yields_doctor_hint(self):
        hint = _corruption_hint(sqlite3.DatabaseError("database disk image is malformed"))
        assert hint and "mnemos doctor" in hint

    def test_non_corruption_returns_none(self):
        assert _corruption_hint(sqlite3.OperationalError("no such column: foo")) is None

    def test_search_augments_corruption_error_with_hint(self, tmp_path, monkeypatch):
        m = _m(tmp_path, monkeypatch)
        m.store_memory(project="dev", content="F: alpha", skip_dedup=True)

        def boom(*a, **k):
            raise sqlite3.DatabaseError("database disk image is malformed")

        monkeypatch.setattr(m.store, "search_fts", boom)
        with pytest.raises(sqlite3.DatabaseError) as ei:
            m.search("alpha")
        assert "mnemos doctor" in str(ei.value)
