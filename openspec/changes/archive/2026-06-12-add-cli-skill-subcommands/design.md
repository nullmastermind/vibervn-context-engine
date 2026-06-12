## Context

The engine already exposes its semantic-retrieval capabilities through three transports: an MCP `Streamable HTTP` endpoint at `/mcp`, a per-repo MCP endpoint at `/mcp-repo/:repo_name`, and two REST proxy endpoints (`POST /api/mcp-tool` and `POST /api/mcp-tool/file-retrieval`) that share the exact same funnel functions (`run_codebase_retrieval`, `run_file_retrieval`) as the MCP tools. The REST proxies wrap the funnel result in `{"result": "<plain text>"}` so output is byte-identical to MCP responses.

Today, the binary entry point in `src/main.rs` parses only serve-mode flags via clap derive (`Cli` struct with `port`, `bind`, `data_dir`, `embeddings_dir`) and unconditionally boots the index engine, opens RocksDB, and starts the axum server. There is no terminal-native way to invoke the skills; non-MCP agents must use a separate HTTP client and craft JSON manually.

This change adds a thin subcommand layer so the same binary can act as either the server or a stateless HTTP client of a running server.

## Goals / Non-Goals

**Goals:**

- Provide `search` and `read` subcommands that map 1:1 to the existing REST proxy endpoints.
- Keep client-mode startup near-zero overhead: no RocksDB open, no settings load, no IndexEngine boot.
- Preserve every existing serve-mode flag and behavior (zero regression for `npx vibervn-context-engine@latest ...`).
- Use only crates already in `Cargo.toml` (clap, reqwest, serde_json, tokio).

**Non-Goals:**

- Auto-starting the server when it is not running.
- Streaming progress, SSE, or long-running CLI operations.
- Adding subcommands for any other endpoints (`/api/repos/...`, `/api/index-all`, `/api/query`, etc.).
- Shell completion scripts, man pages, or interactive TUIs.
- Changing the MCP protocol, REST endpoints, or any server-side behavior.
- Local in-process invocation (the user explicitly chose HTTP-only).

## Decisions

### Decision: Optional subcommand on the existing Cli struct

Use clap's `Option<Commands>` enum on the existing `Cli` struct. When `Cli.command` is `None`, run the legacy serve path; when `Some(Commands::Search { ... })` or `Some(Commands::Read { ... })`, run the new client path and `return` from `main` before any IndexEngine code executes.

**Why over alternatives:**
- Adding a separate binary in `src/bin/` would require a second npm distribution path; rejected.
- Renaming the current behavior into a `serve` subcommand would break `npx vibervn-context-engine@latest --port 8080` invocations documented in README.md; rejected.

### Decision: New module `src/cli.rs` with two async entry functions

Place the HTTP client logic in a new module `src/cli.rs` exposing `pub async fn run_search(args: SearchArgs) -> ExitCode` and `pub async fn run_read(args: ReadArgs) -> ExitCode`. `main.rs` matches on the subcommand and awaits the corresponding function, returning its `ExitCode`.

**Why:** keeps `main.rs` small and lets the client functions be unit-tested directly. The functions are async because `reqwest` is async and `tokio::main` is already in place.

### Decision: Reuse existing `reqwest` client with rustls-tls

Build a per-invocation `reqwest::Client` with a connect timeout of 5 seconds and a request timeout of 60 seconds. This matches the engine's behavior under MCP (queries can take tens of seconds when indexing/embedding/reranking is on the path) but caps connect failures at a snappy 5 s.

**Why:** reusing the in-tree `reqwest` (already feature-gated for `rustls-tls`) avoids new dependencies and matches the dev-dependency setup used in integration tests.

### Decision: Server URL precedence — flag > env > default

Resolve the base URL once at the start of each subcommand:
1. `--server <url>` if present
2. `CONTEXT_ENGINE_URL` env var if set and non-empty
3. Default `http://127.0.0.1:6699`

Trim trailing slashes before joining with `/api/mcp-tool` or `/api/mcp-tool/file-retrieval` to avoid double slashes.

**Why:** mirrors the precedence patterns already used by the existing `--port` (CLI > env > default) and is the convention the operator already knows.

### Decision: Output rules — stdout for result, stderr for diagnostics

- On success (`status: 200`, body parses to `{"result": <string>}`), print exactly the `result` value to stdout — no added newline beyond what the value already contains.
- On any failure path, write a single line to stderr and exit non-zero. Stdout MUST stay empty so callers can `cmd 2>/dev/null` and either see the result or see nothing.

**Exit codes:**
- `0` — success
- `2` — connect/network failure (cannot reach server)
- `3` — server returned non-2xx
- Other failures (response parse, JSON shape) → exit `3` as well, since they originate from the server contract

**Why:** this is the conventional Unix-y split (`grep`, `curl`-with-`--fail`) and lets agents pipe the result without contaminating stdout with chatter.

### Decision: Read subcommand omits top_k when not given

When the user does not pass `--top-k`, the request body omits the field entirely (serde `skip_serializing_if = "Option::is_none"`), letting the server-side default in `post_file_retrieval` (`unwrap_or(5)`) apply. This keeps a single source of truth for the default.

## Risks / Trade-offs

- **[Risk] Server lock collision** → not possible: client mode never opens RocksDB. Only serve mode does.
- **[Risk] Long-running queries hit timeout** → mitigation: 60 s request timeout matches typical MCP wait budget (`mcp_index_wait_secs` defaults to 60). If the operator sets a longer wait, they can re-run the query after the index settles.
- **[Risk] User confuses `--server` (CLI) with `--bind` / `--port` (serve)** → mitigation: `--server` is only valid under the `search`/`read` subcommands; clap will reject it at the top level. `--port` and `--bind` remain on the top level.
- **[Trade-off] Two new subcommand names hard-code current skill set** → acceptable: only two MCP tools exist today and adding more is a separate decision. New skills in the future would either (a) reuse `search`/`read` semantics or (b) get their own subcommand under the same pattern.
- **[Risk] Output budget mismatch** → not relevant: the REST proxy already applies the same budget (`MAX_TOOL_OUTPUT_CHARS`) inside `run_codebase_retrieval`, so CLI output is byte-identical to MCP output.

## Migration Plan

This is purely additive. No data migration, no settings migration, no rollback risk. The npm wrapper script in `npm/` does not need changes — it continues to spawn the binary with whatever flags the user passed, including the new subcommands.
