"""Relocate a SQLite store without losing writes (v10.41.0).

Moving a live SQLite file with `mv` is how stores get lost:

  - A process that still has the old file open keeps writing to it after the
    rename, and those writes never reach the new location.
  - A raw copy of a WAL-mode file omits rows that still live in the -wal.
  - A config that still names the old path makes SQLite create a fresh, empty
    database there, and the system carries on with amnesia.

move_database() closes all three. It refuses while it can see anyone else on
the store, copies through the backup API under SQLite's exclusive lock,
verifies the copy, gives it the source's permissions, keeps the source as a
rollback file, and swaps a symlink onto the old path in one atomic
step so a stale config still reaches the real store.

The caller must own the store, root included (v10.41.1). Every path the move
touches sits in a directory the store's owner controls, so a privileged mover
could be made to write, chown or chmod a file of that owner's choosing by
swapping a path for a symlink between two steps. Run as the owner, the kernel
enforces the owner's permissions on every step and there is nothing to
escalate to. Root moving its own store refuses any directory another user can
write, for the same reason.

The mover is schema-agnostic. It never loads an extension and never opens the
file as a Mnemos store, so it relocates any SQLite database, not only memory.db.

Two in-use checks run before anything is touched:

  1. On Linux, /proc is scanned for any process that has the file open. This
     sees idle connections in every journal mode, which matters because a
     default Mnemos store is a rollback-journal database, where an idle
     connection holds no file lock at all.
  2. SQLite's exclusive lock, on every platform. It sees every connection that
     has read or written a WAL database and every active transaction on a
     rollback-journal one.

A process neither check can see (another user's process when not root, any
idle one where /proc is missing, an open() that lands between scan and lock)
is handled by the last line of defence: the kept file is left in
rollback-journal mode. SQLite refuses to write a rollback-journal database
whose file has been renamed under an open handle (SQLITE_READONLY_DBMOVED), so
such a process gets an error on its next write instead of committing into the
abandoned file. WAL mode has no such guard, which is why the kept file is
switched out of it. The destination keeps the source's journal mode.

Ceiling: an unseen process can still READ the kept file and see stale data
until it is restarted. Stop MCP and HTTP servers before moving; the checks are
there for the ones you forgot. Upgrade path: an advisory lock file taken by
every opener.
"""
import os
import sqlite3
import stat
import time
from typing import Dict, List, Optional


class StoreBusyError(RuntimeError):
    """Another process holds the store; the move was refused."""


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_counts(conn) -> Dict[str, int]:
    # Ordinary tables only. The shadow tables of FTS5 and vec0 are ordinary
    # tables too, so their contents are counted without loading either module.
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND sql NOT LIKE 'CREATE VIRTUAL%' ORDER BY name")]
    return {n: conn.execute(f"SELECT COUNT(*) FROM {_quote(n)}").fetchone()[0]
            for n in names}


def _open_handles(path: str) -> List[int]:
    """PIDs that have `path` open, from /proc. Empty where /proc is missing
    and for processes this user may not inspect."""
    try:
        target = os.stat(path)
        entries = os.listdir("/proc")
    except OSError:
        return []
    pids = []
    for entry in entries:
        if not entry.isdigit():
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                st = os.stat(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if (st.st_ino, st.st_dev) == (target.st_ino, target.st_dev):
                pids.append(int(entry))
                break
    return sorted(pids)


def _remove_quiet(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _missing_dirs(path: str) -> List[str]:
    """`path` and those of its ancestors that do not exist yet, outermost first."""
    missing = []
    while path and not os.path.isdir(path):
        missing.append(path)
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return missing[::-1]


def _free_name(base: str) -> str:
    name, n = base, 1
    while os.path.lexists(name):
        name, n = f"{base}.{n}", n + 1
    return name


def _others_can_write(directory: str) -> bool:
    st = os.stat(directory)
    if st.st_uid != 0 or st.st_mode & 0o002:
        return True
    if st.st_mode & 0o020 and st.st_gid != 0:
        return True
    try:
        # A POSIX ACL can grant write access the mode bits do not show.
        return "system.posix_acl_access" in os.listxattr(directory)
    except (AttributeError, OSError):
        return False


def _check_caller(real_src: str, src_stat: os.stat_result, dest: str) -> None:
    """Refuse a move that would run with more privilege than the store's owner
    has over the directories involved. See the module docstring."""
    if not hasattr(os, "geteuid"):
        return
    euid = os.geteuid()
    if src_stat.st_uid != euid:
        raise PermissionError(
            f"{real_src} belongs to uid {src_stat.st_uid}; run the move as that "
            f"user (this process is uid {euid}).")
    if euid != 0:
        return
    anchor = os.path.dirname(dest)
    while not os.path.isdir(anchor):
        anchor = os.path.dirname(anchor)
    for directory in (os.path.dirname(real_src), anchor):
        if _others_can_write(directory):
            raise PermissionError(
                f"refusing to move a root-owned store through {directory}: "
                "another user can write there.")


def move_database(src: str, dest: str, link: bool = True,
                  timeout: float = 2.0) -> dict:
    """Move the SQLite database at `src` to `dest`.

    `dest` is a file path, an existing directory, or a path ending in a
    separator (a directory, created if missing); for a directory the file name
    is kept. Returns a dict with status ("moved" or "already-there"), source,
    dest, kept (the rollback copy), link (the symlink at the old path, or
    None), link_error and journal_mode.

    Raises FileNotFoundError (no source), FileExistsError (dest taken),
    PermissionError (the caller does not own the store, or root would work in
    a directory another user can write), StoreBusyError (store in use), sqlite3.Error (not a database) or
    RuntimeError (the copy failed verification). On any failure the source is
    unchanged and nothing is left at dest.
    """
    src_given = os.path.abspath(os.path.expanduser(src))
    if not os.path.exists(src_given):
        raise FileNotFoundError(f"no database at {src_given}")
    real_src = os.path.realpath(src_given)

    into_dir = dest.endswith(os.sep) or bool(os.altsep and dest.endswith(os.altsep))
    dest = os.path.abspath(os.path.expanduser(dest))
    if into_dir or os.path.isdir(dest):
        dest = os.path.join(dest, os.path.basename(real_src))
    if os.path.realpath(dest) == real_src:
        return {"status": "already-there", "source": real_src, "dest": dest,
                "kept": None, "link": None, "link_error": None}
    if os.path.lexists(dest):
        raise FileExistsError(f"destination already exists: {dest}")

    # Scanned before our own connection exists, so every hit is someone else,
    # including another connection inside this process.
    holders = _open_handles(real_src)
    if holders:
        raise StoreBusyError(
            f"{real_src} is open in process {', '.join(map(str, holders))} "
            "(an MCP or HTTP server, a running session, a cron job). "
            "Stop it and retry.")

    src_stat = os.stat(real_src)
    _check_caller(real_src, src_stat, dest)
    conn = sqlite3.connect(real_src, timeout=timeout, isolation_level=None)
    tmp = dest + ".moving"
    created_dirs: List[str] = []
    left_wal = False
    tmp_fd = None
    try:
        try:
            # In exclusive locking mode the file lock is taken on first use and
            # held until close, so nobody can read or write during the move.
            conn.execute("PRAGMA locking_mode=EXCLUSIVE")
            conn.execute("BEGIN EXCLUSIVE")
            conn.execute("COMMIT")
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                raise StoreBusyError(
                    f"{real_src} is in use by another process (an MCP or HTTP "
                    "server, a running session, a cron job). Stop it and retry."
                ) from e
            raise

        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
        expected = _table_counts(conn)

        created_dirs = _missing_dirs(os.path.dirname(dest))
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        # Created private and exclusively, before SQLite opens it: the copy is
        # never readable by others, and a symlink planted at this name is not
        # followed. Mode and group go on through the descriptor at the end.
        _remove_quiet(tmp)
        tmp_fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
        out = sqlite3.connect(tmp)
        try:
            conn.backup(out)
        finally:
            out.close()

        check = sqlite3.connect(tmp)
        try:
            verdict = check.execute("PRAGMA quick_check").fetchone()[0]
            actual = _table_counts(check)
        finally:
            check.close()
        if verdict != "ok":
            raise RuntimeError(f"copy failed quick_check: {verdict}")
        if actual != expected:
            diff = sorted(t for t in set(expected) | set(actual)
                          if expected.get(t) != actual.get(t))
            raise RuntimeError(f"copy differs from source in tables: {diff}")
        for ext in ("-wal", "-shm"):
            _remove_quiet(tmp + ext)
        if hasattr(os, "fchmod"):
            try:
                os.fchown(tmp_fd, -1, src_stat.st_gid)
            except OSError:
                pass  # not a member of the source's group; the copy keeps ours
            os.fchmod(tmp_fd, stat.S_IMODE(src_stat.st_mode))
        else:
            os.chmod(tmp, stat.S_IMODE(src_stat.st_mode))
        os.close(tmp_fd)
        tmp_fd = None

        if os.path.lexists(dest):
            raise FileExistsError(f"destination already exists: {dest}")
        if journal_mode == "wal":
            # See the module docstring: only a rollback-journal file refuses
            # writes from a handle that outlived the rename.
            now = conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0].lower()
            if now != "delete":
                raise RuntimeError("could not take the source out of WAL mode")
            left_wal = True
        os.rename(tmp, dest)
    except BaseException:
        if left_wal:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass
        conn.close()
        if tmp_fd is not None:
            os.close(tmp_fd)
        for leftover in (tmp, tmp + "-wal", tmp + "-shm"):
            _remove_quiet(leftover)
        for d in reversed(created_dirs):
            try:
                os.rmdir(d)
            except OSError:
                pass
        raise

    kept = _free_name(f"{real_src}.moved-{time.strftime('%Y%m%d-%H%M%S')}")
    link_path: Optional[str] = None
    link_error: Optional[str] = None
    swapped = False
    if link and os.name != "nt":
        # Hard-link the old file to its rollback name, then replace the old
        # path with the symlink in one rename: the path never goes missing, so
        # nothing can create a fresh empty database there in between.
        tmp_link = real_src + ".moving-link"
        try:
            os.link(real_src, kept)
            _remove_quiet(tmp_link)
            os.symlink(dest, tmp_link)
            os.replace(tmp_link, real_src)
            link_path, swapped = real_src, True
        except OSError as e:
            link_error = str(e)
            _remove_quiet(tmp_link)
    if not swapped:
        if os.name == "nt":
            # Windows cannot rename an open file; accept the short unlocked window.
            conn.close()
        if os.path.lexists(kept):
            os.unlink(real_src)  # kept already is a hard link to it
        else:
            os.rename(real_src, kept)
        if link and link_error is None:
            try:
                os.symlink(dest, real_src)
                link_path = real_src
            except OSError as e:
                link_error = str(e)
    conn.close()
    for ext in ("-wal", "-shm"):
        _remove_quiet(real_src + ext)

    return {"status": "moved", "source": real_src, "dest": dest, "kept": kept,
            "link": link_path, "link_error": link_error,
            "journal_mode": journal_mode, "tables": len(expected),
            "rows": sum(expected.values())}
