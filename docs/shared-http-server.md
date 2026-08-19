# Shared HTTP server: one Mnemos, many harnesses

## Why

Every MCP client that spawns `mnemos-mcp` over stdio gets a private process,
and each private process loads its own copy of the ONNX models (embedder,
reranker, optional NLI). The model files on disk are shared
page cache, but ONNX Runtime copies weights into anonymous private heap per
process, and the ORT CPU arena grows with use. Three harnesses on one machine
means three multi-GB copies of the same weights.

`mnemos serve --http` starts ONE long-lived process that any number of
harnesses attach to over localhost HTTP. Models load once, quality settings
stay identical (rerank and NLI on), and all clients read and write the same
database and namespace.

## Start the server

```
mnemos serve --http 127.0.0.1:8377
# or a user-only unix socket:
mnemos serve --unix "$XDG_RUNTIME_DIR/mnemos.sock"
```

Binding non-local addresses is refused unless `MNEMOS_HTTP_ALLOW_NONLOCAL=1`
is set. There is no auth: keep it on localhost or a 0600 unix socket.

Concurrent requests from different harnesses are serialized through one
process-wide dispatch lock, so two busy clients queue briefly instead of
running unbounded concurrent ONNX sessions.

## systemd user unit

`~/.config/systemd/user/mnemos-http.service`:

```ini
[Unit]
Description=Mnemos shared MCP server (HTTP, models load once)
After=default.target

[Service]
Type=simple
ExecStart=%h/.mnemos/venv/bin/mnemos serve --http 127.0.0.1:8377
Restart=on-failure
RestartSec=5
Environment=MNEMOS_DB=%h/.mnemos/memory.db
Environment=MNEMOS_NAMESPACE=mikael
Environment=MNEMOS_ENABLE_RERANK=1
Environment=MNEMOS_RETRIEVAL_LOG=1
Environment=MNEMOS_TOOL_USAGE_LOG=1
Environment=MNEMOS_DEDUP_CONFIRM=nli
Environment=MNEMOS_CONTRADICT_MODE=nli

[Install]
WantedBy=default.target
```

```
systemctl --user daemon-reload
systemctl --user enable --now mnemos-http.service
loginctl enable-linger "$USER"   # keep it alive without an open session
```

Adjust `ExecStart` and the env block to the machine (paths above match a
`~/.mnemos/venv` install). Do not add `MNEMOS_MODEL_IDLE_TTL` here unless you
want the shared copy to unload too; with one process the copy is cheap to keep.

## Point the harnesses at it

Remove the `command` entry (that is what forks private copies) and replace it
with the URL.

Claude Code (`~/.claude.json`):

```json
"agent-memory": {
  "type": "http",
  "url": "http://127.0.0.1:8377/"
}
```

Codex CLI (`~/.codex/config.toml`):

```toml
[mcp_servers.agent-memory]
url = "http://127.0.0.1:8377/"
```

Grok CLI: same idea, `url` instead of `command` in its MCP server entry.

Restart the harnesses after rewiring (MCP connections are made at startup).

## Verify

```
pgrep -af "mnemos serve --http"        # exactly one process
curl -s http://127.0.0.1:8377/ -X POST \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | head -c 200
```

Then search from two harnesses at once and watch RSS: one process in the
single-copy band (a few GB with e5 + jina + NLI), not N times that.

## stdio is still the default

`mnemos-mcp` and plain `mnemos serve` speak newline JSON-RPC on stdio exactly
as before. Single-client installs (laptops, CI) need none of this.
