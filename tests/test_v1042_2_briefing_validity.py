"""10.42.2: the briefing honours validity like every search surface.

A memory retired with valid_until (how a self trait that stopped being true is
recorded) or scheduled with a future valid_from stays out of the session
briefing unless the caller asks for history.
"""

from mnemos.core import Mnemos
from mnemos.storage.base import Memory
from mnemos.storage.sqlite_store import SQLiteStore

import pytest


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(db_path=str(tmp_path / "memory.db"), namespace="alice")
    s.init_schema()
    yield s
    s.close()


def put(store, content, **fields):
    return store.store_memory(
        Memory(namespace=store.namespace, project="self", content=content,
               importance=8, **fields),
        embedding=None,
    )


def agent(store):
    return Mnemos(store=store, namespace=store.namespace, enable_rerank=False,
                  enable_contradiction_detection=False)


def test_briefing_hides_expired_and_not_yet_valid_by_default(store):
    put(store, "L:retired trait zz-expired", valid_until="2000-01-01")
    put(store, "L:future trait zz-future", valid_from="2999-01-01")
    put(store, "L:current trait zz-current")
    out = agent(store).briefing()
    assert "zz-current" in out
    assert "zz-expired" not in out
    assert "zz-future" not in out


def test_briefing_valid_only_false_returns_history(store):
    put(store, "L:retired trait zz-expired", valid_until="2000-01-01")
    put(store, "L:current trait zz-current")
    out = agent(store).briefing(valid_only=False)
    assert "zz-expired" in out and "zz-current" in out


def test_cli_briefing_include_expired(tmp_path, monkeypatch, capsys):
    from tests.test_v1042_self import _reload
    from mnemos import cli
    _reload(monkeypatch, MNEMOS_DB=str(tmp_path / "memory.db"), MNEMOS_NAMESPACE="alice")
    cli.main(["add", "-p", "self", "L:retired trait zz-expired", "--valid-until", "2000-01-01"])
    cli.main(["add", "-p", "self", "L:current trait zz-current"])
    capsys.readouterr()
    cli.main(["briefing"])
    out = capsys.readouterr().out
    assert "zz-current" in out and "zz-expired" not in out
    cli.main(["briefing", "--include-expired"])
    out = capsys.readouterr().out
    assert "zz-expired" in out
