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
INTERMEDIATE = ["2025-11-25", "2025-06-18", "2025-03-26"]
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
        "params": {"protocolVersion": "1900-01-01", "capabilities": {}},
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


# The intermediate published revisions (2025-03-26, 2025-06-18) sit between the
# two eras. They are served by the legacy path, but they must not be refused:
# before the version gate existed the header was ignored, so rejecting them
# would silently break clients that used to work.


def _post_with_header(url, version):
    data = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "MCP-Protocol-Version": version},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_discover_advertises_every_supported_revision(http_server):
    _status, _headers, body = _post(http_server, {
        "jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": {},
    })
    advertised = body["result"]["supportedVersions"]
    assert advertised == [MODERN, *INTERMEDIATE, LEGACY]


def test_intermediate_versions_pass_the_http_header_gate(http_server):
    for version in INTERMEDIATE:
        status, body = _post_with_header(http_server, version)
        assert status == 200, f"{version} was refused with {status}"
        assert body["result"]["tools"]


def test_intermediate_versions_pass_the_meta_gate(http_server):
    for version in INTERMEDIATE:
        _status, _headers, body = _post(http_server, {
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
            "params": {"_meta": {PROTOCOL_KEY: version}},
        })
        assert "error" not in body, f"{version} was refused: {body.get('error')}"
        assert body["result"]["tools"]


def test_initialize_echoes_an_intermediate_proposal(http_server):
    for version in INTERMEDIATE:
        _status, _headers, body = _post(http_server, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": version, "capabilities": {},
                       "clientInfo": {"name": "probe", "version": "0"}},
        })
        assert body["result"]["protocolVersion"] == version


# The gate is a range, not a list. Refusing every revision we had not heard of
# broke working clients twice; the wire format is stable across the handshake
# era, so an unnamed revision inside the range is served rather than rejected.


def test_unnamed_in_range_revision_is_served(http_server):
    """A revision nobody has told this server about, between legacy and modern."""
    status, body = _post_with_header(http_server, "2025-09-09")
    assert status == 200
    assert body["result"]["tools"]


def test_revision_newer_than_modern_is_refused(http_server):
    """Beyond PROTOCOL_MODERN we cannot know what changed -- let them downgrade."""
    status, body = _post_with_header(http_server, "2027-01-01")
    assert status == 400
    assert body["error"]["code"] == -32022
    assert MODERN in body["error"]["data"]["supported"]


def test_predates_legacy_and_malformed_are_refused(http_server):
    for bogus in ["1900-01-01", "not-a-version", "2025-6-18", "", "2025-06-18-extra"]:
        if not bogus:
            continue
        status, _body = _post_with_header(http_server, bogus)
        assert status == 400, f"{bogus!r} should be refused, got {status}"


def test_predicate_boundaries():
    from mnemos.mcp_server import protocol_supported, PROTOCOL_MODERN, PROTOCOL_LEGACY
    assert protocol_supported(PROTOCOL_MODERN)
    assert protocol_supported(PROTOCOL_LEGACY)
    assert protocol_supported("2025-11-25")
    assert protocol_supported("2025-09-09")
    assert not protocol_supported("2024-11-04")   # one day before the floor
    assert not protocol_supported("2026-07-29")   # one day past the ceiling
    assert not protocol_supported("20250618")
    assert not protocol_supported(None)
    assert not protocol_supported("")


def test_rejected_request_does_not_poison_the_next_on_one_connection(http_server):
    """Regression: the 400 path used to return before reading the body, so the
    next request on a keep-alive connection was parsed as a continuation of it.
    """
    import http.client
    from urllib.parse import urlparse
    parsed = urlparse(http_server)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    # 1: refused on version, body deliberately left for the server to drain.
    conn.request("POST", "/", body=payload, headers={
        "Content-Type": "application/json", "MCP-Protocol-Version": "2027-01-01"})
    first = conn.getresponse()
    first.read()
    assert first.status == 400
    # 2: same connection, valid. Must be answered on its own merits.
    conn.request("POST", "/", body=payload, headers={
        "Content-Type": "application/json", "MCP-Protocol-Version": MODERN})
    second = conn.getresponse()
    body = json.loads(second.read())
    conn.close()
    assert second.status == 200, f"second request desynced: {second.status}"
    assert body["result"]["tools"]
