"""
Dual-era MCP protocol tests (v10.36.0).

2026-07-28 removed the initialize handshake: modern clients declare their
protocol version per-request in _meta, servers MUST implement server/discover,
and every result carries resultType. Legacy clients still open with initialize
and never announce a version per request. These tests pin both eras plus the
HTTP-only MCP-Protocol-Version header check, through the real HTTP transport.
"""

import json
import urllib.request

from tests.test_http_transport import _post, http_server  # noqa: F401 (fixture)

MODERN = "2026-07-28"
LEGACY = "2024-11-05"
PROTOCOL_KEY = "io.modelcontextprotocol/protocolVersion"
SERVER_INFO_KEY = "io.modelcontextprotocol/serverInfo"


def _modern_params(extra=None):
    params = dict(extra or {})
    params["_meta"] = {PROTOCOL_KEY: MODERN}
    return params


def test_discover_advertises_both_eras(http_server):
    status, _headers, body = _post(http_server, {
        "jsonrpc": "2.0", "id": 1, "method": "server/discover",
        "params": _modern_params(),
    })
    assert status == 200
    result = body["result"]
    assert MODERN in result["supportedVersions"]
    assert LEGACY in result["supportedVersions"], "legacy clients must stay served"
    assert result["capabilities"]["tools"] == {}
    assert result["instructions"]
    assert result["ttlMs"] > 0
    assert result["cacheScope"] in ("public", "private")
    assert result["resultType"] == "complete"
    assert result["_meta"][SERVER_INFO_KEY]["name"] == "mnemos"


def test_initialize_negotiates_and_stays_legacy_shaped(http_server):
    status, _headers, body = _post(http_server, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": LEGACY, "capabilities": {}},
    })
    assert status == 200
    result = body["result"]
    assert result["protocolVersion"] == LEGACY
    assert "resultType" not in result
    assert "_meta" not in result


def test_initialize_echoes_a_supported_modern_proposal(http_server):
    _status, _headers, body = _post(http_server, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": MODERN, "capabilities": {}},
    })
    assert body["result"]["protocolVersion"] == MODERN


def test_initialize_falls_back_on_unknown_proposal(http_server):
    _status, _headers, body = _post(http_server, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {}},
    })
    assert body["result"]["protocolVersion"] == LEGACY


def test_tools_list_is_cacheable_and_sorted(http_server):
    _status, _headers, body = _post(http_server, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })
    result = body["result"]
    names = [t["name"] for t in result["tools"]]
    assert names and names == sorted(names)
    assert result["ttlMs"] > 0
    assert result["cacheScope"] == "public"
    assert result["resultType"] == "complete"


def test_unsupported_meta_version_is_refused_with_retry_data(http_server):
    _status, _headers, body = _post(http_server, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list",
        "params": {"_meta": {PROTOCOL_KEY: "1900-01-01"}},
    })
    error = body["error"]
    assert error["code"] == -32022
    assert error["data"]["requested"] == "1900-01-01"
    assert MODERN in error["data"]["supported"]


def test_http_header_unsupported_version_is_400(http_server):
    data = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode()
    req = urllib.request.Request(
        http_server, data=data, method="POST",
        headers={"Content-Type": "application/json", "MCP-Protocol-Version": "1900-01-01"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status, body = resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        status, body = e.code, json.loads(e.read())
    assert status == 400
    assert body["error"]["code"] == -32022


def test_http_header_supported_version_is_accepted(http_server):
    data = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode()
    req = urllib.request.Request(
        http_server, data=data, method="POST",
        headers={"Content-Type": "application/json", "MCP-Protocol-Version": MODERN},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        assert json.loads(resp.read())["result"]["tools"]


def test_tool_call_works_under_the_modern_envelope(http_server):
    _status, _headers, body = _post(http_server, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": _modern_params({"name": "memory_list_tags", "arguments": {}}),
    })
    result = body["result"]
    assert result["resultType"] == "complete"
    assert result["_meta"][SERVER_INFO_KEY]["name"] == "mnemos"
    assert not result.get("isError")
    assert "content" in result
