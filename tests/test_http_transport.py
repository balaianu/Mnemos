"""
Tests for the streamable-HTTP transport (v10.27.0).

The point of the transport is N harnesses sharing ONE process so models load
once; the tests exercise the protocol subset and the shared-state property
against a throwaway SQLite DB. Search uses search_mode=fts so no embedding
model is required.
"""

import json
import os
import tempfile
import threading
import urllib.request

import pytest


def _post(url, payload, timeout=10):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, dict(resp.headers), json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        body = e.read()
        return e.code, dict(e.headers), json.loads(body) if body else None


@pytest.fixture
def http_server():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["MNEMOS_DB"] = db_path
    os.environ["MNEMOS_EAGER_WARMUP"] = "0"
    os.environ["MNEMOS_ENABLE_RERANK"] = "0"

    from mnemos.core import Mnemos
    from mnemos.storage.sqlite_store import SQLiteStore
    from mnemos.http_server import MnemosHTTPServer

    mnemos = Mnemos(store=SQLiteStore(db_path=db_path))
    server = MnemosHTTPServer(("127.0.0.1", 0), mnemos)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}/"
    finally:
        server.shutdown()
        server.server_close()
        os.environ.pop("MNEMOS_DB", None)
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass


def _rpc(method, id_=1, params=None):
    msg = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


class TestProtocol:
    def test_initialize_issues_session_id(self, http_server):
        status, headers, body = _post(http_server, _rpc("initialize"))
        assert status == 200
        assert body["result"]["serverInfo"]["name"] == "mnemos"
        assert headers.get("Mcp-Session-Id")

    def test_tools_list_has_six_tools(self, http_server):
        status, _, body = _post(http_server, _rpc("tools/list"))
        assert status == 200
        names = {t["name"] for t in body["result"]["tools"]}
        assert names == {
            "memory_store", "memory_search", "memory_get",
            "memory_update", "memory_list_tags", "memory_bulk_rewrite",
        }

    def test_notification_gets_202(self, http_server):
        status, _, body = _post(
            http_server,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        assert status == 202
        assert body is None

    def test_unknown_method_is_json_rpc_error(self, http_server):
        status, _, body = _post(http_server, _rpc("nonsense/method"))
        assert status == 200
        assert body["error"]["code"] == -32601

    def test_parse_error_is_400(self, http_server):
        req = urllib.request.Request(
            http_server, data=b"{not json", method="POST",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 400

    def test_get_is_405(self, http_server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(http_server, timeout=10)
        assert exc.value.code == 405

    def test_batch_request(self, http_server):
        status, _, body = _post(
            http_server, [_rpc("initialize", id_=1), _rpc("tools/list", id_=2)]
        )
        assert status == 200
        assert isinstance(body, list)
        assert {m["id"] for m in body} == {1, 2}


class TestSharedState:
    def test_two_clients_share_one_store(self, http_server):
        """Client A initializes and stores; client B initializes separately and
        finds A's memory. One process, one DB, one model set: the acceptance
        criterion of the shared-server design."""
        _post(http_server, _rpc("initialize", id_=1))
        status, _, body = _post(http_server, _rpc("tools/call", id_=2, params={
            "name": "memory_store",
            "arguments": {"project": "test", "content": "F:shared-http canary zqx11"},
        }))
        assert status == 200
        stored = json.loads(body["result"]["content"][0]["text"])
        assert stored.get("id")

        _post(http_server, _rpc("initialize", id_=3))
        status, _, body = _post(http_server, _rpc("tools/call", id_=4, params={
            "name": "memory_search",
            "arguments": {"query": "zqx11", "search_mode": "fts", "project": "test"},
        }))
        assert status == 200
        found = json.loads(body["result"]["content"][0]["text"])
        assert found["count"] >= 1
        assert any("zqx11" in r["content"] for r in found["results"])


class TestBindGuard:
    def test_nonlocal_bind_refused(self):
        from mnemos.http_server import serve

        os.environ.pop("MNEMOS_HTTP_ALLOW_NONLOCAL", None)
        with pytest.raises(ValueError, match="non-local"):
            serve(http="0.0.0.0:9", mnemos=object())

    def test_hostport_parsing(self):
        from mnemos.http_server import _parse_hostport

        assert _parse_hostport("127.0.0.1:8377") == ("127.0.0.1", 8377)
        assert _parse_hostport(":8377") == ("127.0.0.1", 8377)
        with pytest.raises(ValueError):
            _parse_hostport("8377")


class TestUnixSocket:
    def test_unix_socket_roundtrip(self, tmp_path):
        import http.client
        import socket

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.environ["MNEMOS_EAGER_WARMUP"] = "0"

        from mnemos.core import Mnemos
        from mnemos.storage.sqlite_store import SQLiteStore
        from mnemos.http_server import MnemosUnixHTTPServer

        sock_path = str(tmp_path / "mnemos.sock")
        mnemos = Mnemos(store=SQLiteStore(db_path=db_path))
        server = MnemosUnixHTTPServer(sock_path, mnemos)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            assert oct(os.stat(sock_path).st_mode & 0o777) == "0o600"

            class UnixConn(http.client.HTTPConnection):
                def connect(self):
                    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    self.sock.connect(sock_path)

            conn = UnixConn("localhost")
            conn.request(
                "POST", "/", body=json.dumps(_rpc("tools/list")),
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            assert resp.status == 200
            body = json.loads(resp.read())
            assert len(body["result"]["tools"]) == 6
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            os.unlink(db_path)
