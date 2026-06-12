## Why

The two MCP tools (`codebase-retrieval` and `file-retrieval`) exposed by the engine today are only reachable through the MCP protocol or the existing REST proxy endpoints (`POST /api/mcp-tool` and `POST /api/mcp-tool/file-retrieval`). Agents and scripts that cannot speak MCP — including shell pipelines, terminal-only assistants, and CI jobs — currently have no first-class way to invoke these tools. Adding terminal-native subcommands lets any agent or operator query the running engine without an MCP client.

## What Changes

- Add a `search` subcommand to the existing `context-engine-rs` binary that invokes the `codebase-retrieval` skill via the running server's REST proxy and prints the plain-text result to stdout.
- Add a `read` subcommand that invokes the `file-retrieval` skill via the running server's REST proxy and prints the plain-text result to stdout.
- Subcommands act as a thin HTTP client: they MUST NOT boot the index engine, MUST NOT open RocksDB, and MUST NOT load `settings.json`.
- Default server endpoint is `http://127.0.0.1:6699`, overridable per-invocation with `--server <url>` or environment variable `CONTEXT_ENGINE_URL`.
- Connection failures and non-2xx HTTP responses produce dedicated stderr messages and non-zero exit codes; stdout is reserved for the tool's plain-text result.
- The existing serve mode (running the binary with no subcommand and the previous `--port` / `--bind` / `--data-dir` / `--embeddings-dir` flags) is preserved unchanged. No breaking changes for existing `npx` usage.

## Capabilities

### New Capabilities

- `cli-skill-invocation`: Provides terminal subcommands that call the engine's existing skill REST proxies (`codebase-retrieval`, `file-retrieval`) and emit their plain-text results to stdout, enabling non-MCP agents to use the same skills.

### Modified Capabilities

<!-- None: serve mode behavior, MCP tools, and existing REST endpoints are unchanged. -->

## Impact

- Affected code: `src/main.rs` (CLI parser refactor to support optional subcommand while preserving current flags), new module `src/cli.rs` (HTTP client logic for the two subcommands), `src/lib.rs` (export the new module).
- Dependencies: no new crates required — `clap`, `reqwest`, `serde_json`, and `tokio` are already in `Cargo.toml`.
- APIs: consumes existing endpoints `POST /api/mcp-tool` and `POST /api/mcp-tool/file-retrieval`; no server-side changes.
- Distribution: no impact on the npm wrapper — the binary entry point is unchanged.
