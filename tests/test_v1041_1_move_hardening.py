"""Tests for v10.41.1: `mnemos move` hardening after the security review of 10.41.0.

10.41.0 let root move a store owned by another user and then chown the copy to
that user. Every path it touched sat in a directory that user controls, so the
user could swap a path for a symlink between two steps and have root write,
chown or chmod a file of their choosing. The copy in progress was also created
with the default umask, readable by others until the final chmod.

  - The caller must own the store. Root is not an exception.
  - Root, moving its own store, refuses directories another user can write.
  - The temporary copy is private from its first byte.
"""
import os
import sqlite3
import stat

import pytest

import mnemos.storage.move as move_mod
from mnemos.storage.move import move_database

as_root = pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() != 0,
                             reason="needs root to change ownership")


def _make_db(path, rows=3):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT)")
    conn.executemany("INSERT INTO memories (content) VALUES (?)",
                     [(f"row {i}",) for i in range(rows)])
    conn.commit()
    conn.close()


def _count(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    finally:
        conn.close()


@as_root
def test_move_refuses_a_store_the_caller_does_not_own(tmp_path):
    src = tmp_path / "memory.db"
    _make_db(str(src))
    os.chown(str(src), 65534, 65534)
    dest = tmp_path / "db" / "memory.db"

    with pytest.raises(PermissionError) as exc:
        move_database(str(src), str(dest))

    assert "65534" in str(exc.value)
    assert not (tmp_path / "db").exists()
    assert os.stat(str(src)).st_uid == 65534 and _count(str(src)) == 3


@as_root
def test_root_refuses_a_destination_directory_another_user_can_write(tmp_path):
    src = tmp_path / "memory.db"
    _make_db(str(src))
    shared = tmp_path / "shared"
    shared.mkdir()
    os.chmod(str(shared), 0o777)

    with pytest.raises(PermissionError):
        move_database(str(src), str(shared / "memory.db"))

    assert os.listdir(str(shared)) == []
    assert _count(str(src)) == 3


@as_root
def test_root_refuses_a_source_directory_owned_by_another_user(tmp_path):
    theirs = tmp_path / "theirs"
    theirs.mkdir()
    src = theirs / "memory.db"
    _make_db(str(src))
    os.chown(str(theirs), 65534, 65534)

    with pytest.raises(PermissionError):
        move_database(str(src), str(tmp_path / "db" / "memory.db"))

    assert not (tmp_path / "db").exists()
    assert os.path.isfile(str(src)) and not os.path.islink(str(src))


def test_the_copy_in_progress_is_private_and_the_result_gets_the_sources_mode(
        tmp_path, monkeypatch):
    src = tmp_path / "memory.db"
    _make_db(str(src))
    os.chmod(str(src), 0o644)
    dest = tmp_path / "db" / "memory.db"
    seen = []
    real_counts = move_mod._table_counts

    def counts_and_look(conn):
        tmp = str(dest) + ".moving"
        if os.path.exists(tmp):
            seen.append(stat.S_IMODE(os.stat(tmp).st_mode))
        return real_counts(conn)
    monkeypatch.setattr(move_mod, "_table_counts", counts_and_look)

    move_database(str(src), str(dest))

    assert seen and all(mode == 0o600 for mode in seen)
    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o644
