"""Tests for v10.41.0: `mnemos move`, relocating a store without losing writes.

Moving a live SQLite file by hand is how stores get lost: a process that still
holds the old file keeps writing to it after the rename, a raw copy of a
WAL-mode file can miss WAL-resident rows, and a config that still names the
old path makes SQLite create a fresh empty database there. These tests pin the
contract of mnemos.storage.move.move_database():

  - it refuses while another connection has the store open (fail closed),
  - the destination holds every row, including rows still in the WAL,
  - the source is kept, renamed, as the rollback copy,
  - a symlink at the old path keeps a stale config on the real store,
  - a failed move leaves the source untouched and no half-written destination.
"""
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import time

import pytest

from mnemos.storage.move import StoreBusyError, move_database


def _make_db(path, rows=3, wal=True):
    conn = sqlite3.connect(path)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT)")
    conn.executemany("INSERT INTO memories (content) VALUES (?)",
                     [(f"row {i}",) for i in range(rows)])
    conn.commit()
    return conn


def _count(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    finally:
        conn.close()


def test_move_copies_every_row_to_the_destination(tmp_path):
    src = tmp_path / "memory.db"
    _make_db(str(src), rows=5).close()
    dest = tmp_path / "db" / "memory.db"

    result = move_database(str(src), str(dest))

    assert result["status"] == "moved"
    assert result["dest"] == str(dest)
    assert _count(str(dest)) == 5


def test_move_includes_rows_still_resident_in_the_wal(tmp_path):
    live = tmp_path / "live" / "memory.db"
    live.parent.mkdir()
    conn = _make_db(str(live), rows=2)
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("INSERT INTO memories (content) VALUES ('wal only')")
    conn.commit()
    # Image of a writer that died before checkpointing: the third row exists
    # only in the -wal file. Copying memory.db alone would lose it.
    src = tmp_path / "memory.db"
    shutil.copy(str(live), str(src))
    shutil.copy(str(live) + "-wal", str(src) + "-wal")
    conn.close()
    assert os.path.getsize(str(src) + "-wal") > 0
    dest = tmp_path / "db" / "memory.db"

    move_database(str(src), str(dest))

    assert _count(str(dest)) == 3


def test_move_refuses_while_another_connection_holds_the_store(tmp_path):
    src = tmp_path / "memory.db"
    holder = _make_db(str(src))
    dest = tmp_path / "db" / "memory.db"

    with pytest.raises(StoreBusyError):
        move_database(str(src), str(dest))

    assert not dest.exists()
    assert not os.path.islink(str(src))
    holder.execute("INSERT INTO memories (content) VALUES ('still writable')")
    holder.commit()
    holder.close()
    assert _count(str(src)) == 4


def test_move_keeps_the_source_as_a_renamed_rollback_copy(tmp_path):
    src = tmp_path / "memory.db"
    _make_db(str(src), rows=4).close()
    dest = tmp_path / "db" / "memory.db"

    result = move_database(str(src), str(dest))

    kept = result["kept"]
    assert kept != str(src) and os.path.dirname(kept) == str(tmp_path)
    assert os.path.isfile(kept) and not os.path.islink(kept)
    assert _count(kept) == 4


def test_move_leaves_a_symlink_so_a_stale_config_reaches_the_real_store(tmp_path):
    src = tmp_path / "memory.db"
    _make_db(str(src), rows=2).close()
    dest = tmp_path / "db" / "memory.db"

    result = move_database(str(src), str(dest))

    assert result["link"] == str(src)
    assert os.path.realpath(str(src)) == os.path.realpath(str(dest))
    # A writer that still uses the old path lands in the moved store.
    conn = sqlite3.connect(str(src))
    conn.execute("INSERT INTO memories (content) VALUES ('via old path')")
    conn.commit()
    conn.close()
    assert _count(str(dest)) == 3


def test_move_without_link_frees_the_old_path(tmp_path):
    src = tmp_path / "memory.db"
    _make_db(str(src)).close()
    dest = tmp_path / "db" / "memory.db"

    result = move_database(str(src), str(dest), link=False)

    assert result["link"] is None
    assert not os.path.lexists(str(src))


def test_move_into_a_directory_keeps_the_file_name(tmp_path):
    src = tmp_path / "memory.db"
    _make_db(str(src)).close()
    dest_dir = tmp_path / "db"
    dest_dir.mkdir()

    result = move_database(str(src), str(dest_dir))

    assert result["dest"] == str(dest_dir / "memory.db")
    assert _count(str(dest_dir / "memory.db")) == 3


def test_move_refuses_to_overwrite_an_existing_destination(tmp_path):
    src = tmp_path / "memory.db"
    _make_db(str(src), rows=2).close()
    dest = tmp_path / "db" / "memory.db"
    dest.parent.mkdir()
    _make_db(str(dest), rows=7).close()

    with pytest.raises(FileExistsError):
        move_database(str(src), str(dest))

    assert _count(str(dest)) == 7
    assert _count(str(src)) == 2


def test_move_is_a_no_op_when_the_store_is_already_at_the_destination(tmp_path):
    src = tmp_path / "memory.db"
    _make_db(str(src), rows=2).close()
    dest = tmp_path / "db" / "memory.db"
    move_database(str(src), str(dest))

    # Second run through the symlink the first run left behind.
    result = move_database(str(src), str(dest))

    assert result["status"] == "already-there"
    assert _count(str(dest)) == 2


def test_move_removes_the_sources_wal_and_shm_sidecars(tmp_path):
    src = tmp_path / "memory.db"
    _make_db(str(src)).close()
    dest = tmp_path / "db" / "memory.db"

    move_database(str(src), str(dest))

    leftovers = [n for n in os.listdir(str(tmp_path))
                 if n.endswith("-wal") or n.endswith("-shm")]
    assert leftovers == []


def test_move_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        move_database(str(tmp_path / "nope.db"), str(tmp_path / "db" / "nope.db"))


def test_cli_move_does_not_initialise_a_store_at_the_old_path(tmp_path, monkeypatch, capsys):
    """`mnemos move` must act on the file, not open it as a Mnemos store first:
    the CLI's own connection would hold the lock the move needs."""
    src = tmp_path / "memory.db"
    _make_db(str(src), rows=2).close()
    dest = tmp_path / "db" / "memory.db"
    import mnemos.cli as cli
    monkeypatch.setattr(cli, "DEFAULT_DB_PATH", str(src), raising=False)

    cli.main(["move", str(dest)])

    out = capsys.readouterr().out
    assert _count(str(dest)) == 2
    assert str(dest) in out and "MNEMOS_DB" in out


def test_cli_move_exits_nonzero_when_the_store_is_in_use(tmp_path, monkeypatch, capsys):
    src = tmp_path / "memory.db"
    holder = _make_db(str(src))
    dest = tmp_path / "db" / "memory.db"
    import mnemos.cli as cli
    monkeypatch.setattr(cli, "DEFAULT_DB_PATH", str(src), raising=False)

    with pytest.raises(SystemExit) as exc:
        cli.main(["move", str(dest)])

    holder.close()
    assert exc.value.code == 1
    assert not dest.exists()


def test_move_of_a_real_mnemos_store_keeps_memories_and_vectors(tmp_path, monkeypatch):
    """The real schema carries FTS5 and vec0 virtual tables; the mover has no
    extension loaded and must still copy and verify them."""
    import mnemos.core as core_mod
    from mnemos.constants import FASTEMBED_DIMS
    from mnemos.core import Mnemos
    from mnemos.storage.sqlite_store import SQLiteStore

    monkeypatch.setattr(core_mod, "embed",
                        lambda texts, prefix="passage": [[0.001] * FASTEMBED_DIMS for _ in texts])
    src = tmp_path / "memory.db"
    m = Mnemos(store=SQLiteStore(db_path=str(src), namespace="t"),
               namespace="t", enable_rerank=False)
    for i in range(6):
        m.store_memory(project="dev", content=f"F: movable fact number {i}", skip_dedup=True)
    m.close()
    dest = tmp_path / "db" / "memory.db"

    result = move_database(str(src), str(dest))

    assert result["status"] == "moved"
    moved = Mnemos(store=SQLiteStore(db_path=str(dest), namespace="t"),
                   namespace="t", enable_rerank=False)
    try:
        assert moved.stats()["total"] == 6
        conn = moved.store._get_conn()
        assert conn.execute("SELECT COUNT(*) FROM embed_meta").fetchone()[0] == 6
    finally:
        moved.close()


# A rollback-journal database is the SQLite default, and what a Mnemos store
# is unless someone switched it to WAL. There an idle connection holds no file
# lock, so the exclusive-lock probe alone cannot see it; a server that kept
# such a connection would go on writing into the renamed old file.

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="open-handle scan reads /proc")


@linux_only
def test_move_refuses_when_an_idle_connection_holds_a_rollback_journal_store(tmp_path):
    src = tmp_path / "memory.db"
    holder = _make_db(str(src), wal=False)
    holder.execute("SELECT COUNT(*) FROM memories").fetchall()
    dest = tmp_path / "db" / "memory.db"

    with pytest.raises(StoreBusyError):
        move_database(str(src), str(dest))

    assert not dest.exists()
    assert os.path.isfile(str(src)) and not os.path.islink(str(src))
    holder.close()


@linux_only
def test_move_refuses_when_another_process_idles_on_a_rollback_journal_store(tmp_path):
    src = tmp_path / "memory.db"
    _make_db(str(src), wal=False).close()
    dest = tmp_path / "db" / "memory.db"
    code = ("import sqlite3, sys, time; c = sqlite3.connect(sys.argv[1]); "
            "c.execute('SELECT COUNT(*) FROM memories').fetchall(); "
            "print('ready', flush=True); time.sleep(30)")
    proc = subprocess.Popen([sys.executable, "-c", code, str(src)],
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"

        with pytest.raises(StoreBusyError) as exc:
            move_database(str(src), str(dest))

        assert str(proc.pid) in str(exc.value)
        assert not dest.exists()
    finally:
        proc.kill()
        proc.wait()

    # With the holder gone the same move goes through.
    assert move_database(str(src), str(dest))["status"] == "moved"


# --- Review findings, 2026-10-06 ---------------------------------------------

def test_a_writer_that_slipped_past_the_checks_fails_instead_of_writing_into_the_old_file(
        tmp_path, monkeypatch):
    """A WAL connection that is open but has not touched the file holds no lock.
    If it also escapes the handle scan (another user's process, a platform
    without /proc, or an open() between scan and lock), it must not be able to
    commit into the abandoned file afterwards."""
    import mnemos.storage.move as move_mod
    src = tmp_path / "memory.db"
    _make_db(str(src), rows=3).close()
    stale = sqlite3.connect(str(src))
    monkeypatch.setattr(move_mod, "_open_handles", lambda path: [])
    dest = tmp_path / "db" / "memory.db"

    result = move_database(str(src), str(dest))

    with pytest.raises(sqlite3.OperationalError):
        stale.execute("INSERT INTO memories (content) VALUES ('lost write')")
        stale.commit()
    stale.close()
    assert _count(str(dest)) == 3
    assert _count(result["kept"]) == 3


def test_a_refused_move_leaves_no_directories_behind(tmp_path):
    src = tmp_path / "memory.db"
    holder = _make_db(str(src))
    dest = tmp_path / "new" / "sub" / "memory.db"

    with pytest.raises(StoreBusyError):
        move_database(str(src), str(dest))

    holder.close()
    assert not (tmp_path / "new").exists()


def test_destination_with_a_trailing_slash_is_a_directory_even_if_it_does_not_exist(tmp_path):
    src = tmp_path / "memory.db"
    _make_db(str(src)).close()

    result = move_database(str(src), str(tmp_path / "db") + os.sep)

    assert result["dest"] == str(tmp_path / "db" / "memory.db")
    assert (tmp_path / "db").is_dir()
    assert _count(str(tmp_path / "db" / "memory.db")) == 3


def test_an_existing_file_with_the_rollback_name_is_not_overwritten(tmp_path, monkeypatch):
    import mnemos.storage.move as move_mod
    monkeypatch.setattr(move_mod.time, "strftime", lambda fmt: "20260101-000000")
    src = tmp_path / "memory.db"
    _make_db(str(src)).close()
    taken = tmp_path / "memory.db.moved-20260101-000000"
    taken.write_text("an earlier rollback copy")

    result = move_database(str(src), str(tmp_path / "db" / "memory.db"))

    assert taken.read_text() == "an earlier rollback copy"
    assert result["kept"] != str(taken) and _count(result["kept"]) == 3


def test_the_destination_keeps_the_sources_permissions(tmp_path):
    src = tmp_path / "memory.db"
    _make_db(str(src)).close()
    os.chmod(str(src), 0o600)
    dest = tmp_path / "db" / "memory.db"

    move_database(str(src), str(dest))

    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o600


def test_a_failed_symlink_is_reported_and_the_move_still_completes(tmp_path, monkeypatch):
    src = tmp_path / "memory.db"
    _make_db(str(src)).close()
    dest = tmp_path / "db" / "memory.db"

    def no_symlinks(*args, **kwargs):
        raise OSError("symlinks not permitted here")
    monkeypatch.setattr(os, "symlink", no_symlinks)

    result = move_database(str(src), str(dest))

    assert result["status"] == "moved"
    assert result["link"] is None and "not permitted" in result["link_error"]
    assert not os.path.lexists(str(src))
    assert _count(str(dest)) == 3 and _count(result["kept"]) == 3


def test_cli_move_on_a_file_that_is_not_sqlite_exits_1_without_a_traceback(
        tmp_path, monkeypatch, capsys):
    src = tmp_path / "notes.txt"
    src.write_text("this is not a database, " * 200)
    import mnemos.cli as cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["move", "--db", str(src), str(tmp_path / "db" / "notes.txt")])

    assert exc.value.code == 1
    assert "move failed" in capsys.readouterr().err
    assert src.read_text().startswith("this is not a database")


def test_a_writer_blocked_on_the_lock_during_the_move_cannot_commit_into_the_old_file(
        tmp_path, monkeypatch):
    """The lost-write race from review: a second process opens the store while
    the move holds the lock, waits out its busy timeout, and commits right
    after the rename. Its commit must fail, not vanish into the old file."""
    import mnemos.storage.move as move_mod
    src = tmp_path / "memory.db"
    _make_db(str(src), rows=3).close()
    dest = tmp_path / "db" / "memory.db"
    code = ("import sqlite3, sys\n"
            "c = sqlite3.connect(sys.argv[1], timeout=10)\n"
            "print('opened', flush=True)\n"
            "try:\n"
            "    c.execute(\"INSERT INTO memories (content) VALUES ('late')\")\n"
            "    c.commit()\n"
            "    print('committed', flush=True)\n"
            "except sqlite3.Error as e:\n"
            "    print('error', e, flush=True)\n")
    started = {}
    real_counts = move_mod._table_counts

    def counts_then_let_a_writer_in(conn):
        # First call happens under the exclusive lock, after the handle scan.
        if "proc" not in started:
            started["proc"] = subprocess.Popen(
                [sys.executable, "-c", code, str(src)],
                stdout=subprocess.PIPE, text=True)
            assert started["proc"].stdout.readline().strip() == "opened"
            time.sleep(0.3)  # let it reach the lock and start waiting
        return real_counts(conn)
    monkeypatch.setattr(move_mod, "_table_counts", counts_then_let_a_writer_in)

    result = move_database(str(src), str(dest))

    outcome = started["proc"].stdout.readline().strip()
    started["proc"].wait(timeout=15)
    in_dest = _count(str(dest))
    assert outcome.startswith("error") or in_dest == 4, outcome
    assert in_dest in (3, 4)
    assert _count(result["kept"]) == 3
    assert not os.path.exists(str(src) + "-wal")
