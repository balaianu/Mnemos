"""v10.26.2: consolidation_log audit writes survive a pre-v10.2.1 schema.

CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so a DB whose
consolidation_log was created before phase_details was added (v10.2.1) never
gained the column. Every log_consolidation_run() INSERT then failed against
it, the failure was swallowed, MAX(run_at) stayed NULL, and each cycle read
"Last run: never" and re-triaged the whole store with surge mode latched on.

Diagnosed on the angssatra deployment, 2026-07-25.
"""

import sqlite3

from mnemos.consolidation.orchestrator import _migrate_nyx_schema


def _pre_v1021_db(tmp_path):
    """A consolidation_log exactly as it existed before phase_details."""
    conn = sqlite3.connect(str(tmp_path / "old.db"))
    conn.execute(
        "CREATE TABLE memory_links (id INTEGER PRIMARY KEY, source_id "
        "INTEGER, target_id INTEGER, relation_type TEXT, strength REAL, "
        "created_at TEXT)")
    conn.execute("""
        CREATE TABLE consolidation_log (
            id INTEGER PRIMARY KEY,
            run_at TEXT DEFAULT (datetime('now','localtime')),
            clusters_found INTEGER DEFAULT 0,
            clusters_merged INTEGER DEFAULT 0,
            memories_archived INTEGER DEFAULT 0,
            memories_created INTEGER DEFAULT 0,
            details TEXT DEFAULT ''
        )
    """)
    conn.commit()
    return conn


class TestPreV1021ConsolidationLog:
    def test_migration_backfills_phase_details(self, tmp_path):
        conn = _pre_v1021_db(tmp_path)
        _migrate_nyx_schema(conn)
        cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(consolidation_log)")]
        assert "phase_details" in cols

    def test_migration_still_backfills_the_v1050_counters(self, tmp_path):
        conn = _pre_v1021_db(tmp_path)
        _migrate_nyx_schema(conn)
        cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(consolidation_log)")]
        assert "access_decayed" in cols and "importance_demoted" in cols

    def test_audit_insert_lands_after_migration(self, tmp_path):
        # The whole point: MAX(run_at) must stop being NULL, because that is
        # what the next cycle reads back as "Last run" to decide surge mode.
        conn = _pre_v1021_db(tmp_path)
        _migrate_nyx_schema(conn)
        conn.execute(
            "INSERT INTO consolidation_log (clusters_found, clusters_merged, "
            "memories_archived, memories_created, access_decayed, "
            "importance_demoted, details, phase_details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (1, 0, 0, 0, 0, 0, "t", "{}"))
        conn.commit()
        assert conn.execute(
            "SELECT MAX(run_at) FROM consolidation_log").fetchone()[0]

    def test_migration_is_idempotent(self, tmp_path):
        conn = _pre_v1021_db(tmp_path)
        _migrate_nyx_schema(conn)
        _migrate_nyx_schema(conn)
        cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(consolidation_log)")]
        assert cols.count("phase_details") == 1


class TestAuditWriteFailureIsAudible:
    def test_failed_write_reports_the_reason_on_stderr(self, tmp_path,
                                                       capsys):
        # A swallowed failure is indistinguishable from a cron that never
        # fired, which is what made the original bug survive for months.
        from mnemos.storage.sqlite_store import SQLiteStore

        store = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")
        store._get_conn().execute("DROP TABLE IF EXISTS consolidation_log")
        store.log_consolidation_run(clusters_found=1)
        assert "consolidation_log write failed" in capsys.readouterr().err
