## ADDED Requirements

### Requirement: Search subcommand invokes codebase-retrieval over HTTP

The `context-engine-rs` binary SHALL accept a `search` subcommand that calls the running engine's `POST /api/mcp-tool` endpoint and prints the returned plain-text result on stdout. The subcommand MUST NOT initialize the index engine, open RocksDB, or load `settings.json`.

#### Scenario: Successful search against running server

- **WHEN** the user runs `context-engine-rs search --query "where is auth handled" --repo /path/to/repo` and an engine server is reachable at `http://127.0.0.1:6699`
- **THEN** the binary sends `POST /api/mcp-tool` with body `{"information_request": "where is auth handled", "workspace_full_path": "/path/to/repo"}`, prints the response's `result` string verbatim to stdout (no added prefix or suffix), writes nothing to stderr, and exits with status code `0`

#### Scenario: Custom server URL via flag

- **WHEN** the user runs `context-engine-rs search --query "Q" --repo /R --server http://192.168.1.10:8080`
- **THEN** the binary issues the request against `http://192.168.1.10:8080/api/mcp-tool` instead of the default endpoint

#### Scenario: Custom server URL via environment

- **WHEN** environment variable `CONTEXT_ENGINE_URL=http://192.168.1.10:8080` is set and the user runs `context-engine-rs search --query "Q" --repo /R` without `--server`
- **THEN** the binary issues the request against `http://192.168.1.10:8080/api/mcp-tool`

#### Scenario: --server flag overrides environment

- **WHEN** `CONTEXT_ENGINE_URL=http://a.example` is set and the user runs `context-engine-rs search --query "Q" --repo /R --server http://b.example`
- **THEN** the binary uses `http://b.example/api/mcp-tool`

#### Scenario: Missing required arguments

- **WHEN** the user runs `context-engine-rs search` without `--query` or without `--repo`
- **THEN** the binary writes a clap-generated usage error to stderr and exits with a non-zero status code

### Requirement: Read subcommand invokes file-retrieval over HTTP

The `context-engine-rs` binary SHALL accept a `read` subcommand that calls the running engine's `POST /api/mcp-tool/file-retrieval` endpoint and prints the returned plain-text result on stdout. The subcommand MUST NOT initialize the index engine, open RocksDB, or load `settings.json`.

#### Scenario: Successful read against running server

- **WHEN** the user runs `context-engine-rs read --repo /path/to/repo --file src/main.rs --query "tracing setup"` with an engine server reachable at the default URL
- **THEN** the binary sends `POST /api/mcp-tool/file-retrieval` with body `{"workspace_full_path": "/path/to/repo", "file_path": "src/main.rs", "information_request": "tracing setup"}`, prints the response's `result` string verbatim to stdout, writes nothing to stderr, and exits with status code `0`

#### Scenario: Read with explicit top-k

- **WHEN** the user runs `context-engine-rs read --repo /R --file src/x.rs --query "Q" --top-k 10`
- **THEN** the request body includes `"top_k": 10`

#### Scenario: Read without top-k uses server default

- **WHEN** the user runs `context-engine-rs read --repo /R --file src/x.rs --query "Q"` without `--top-k`
- **THEN** the request body omits the `top_k` field (or sends `null`), allowing the server-side default of 5 to apply

### Requirement: Server resolution rules

The CLI subcommands SHALL resolve the server base URL with the following precedence: command-line flag `--server`, environment variable `CONTEXT_ENGINE_URL`, then default `http://127.0.0.1:6699`.

#### Scenario: Default base URL

- **WHEN** neither `--server` nor `CONTEXT_ENGINE_URL` is set
- **THEN** the binary uses `http://127.0.0.1:6699` as the base URL

#### Scenario: Trailing slash tolerated

- **WHEN** the user passes `--server http://127.0.0.1:6699/`
- **THEN** the binary still composes valid endpoint URLs (no double slash) and the request succeeds against the same server

### Requirement: Connection failure produces clear error

When the CLI subcommands cannot reach the configured server (DNS failure, connection refused, network unreachable, request timeout), the binary SHALL write a single human-readable error line to stderr and exit with status code `2`. The error MUST name the URL that failed and suggest starting the server.

#### Scenario: Server is not running

- **WHEN** no process listens on the configured server URL and the user runs `context-engine-rs search --query "Q" --repo /R`
- **THEN** stdout is empty, stderr contains a message identifying the unreachable URL and instructing the user to start the server, and the exit code is `2`

### Requirement: Non-2xx HTTP response produces clear error

When the engine responds with an HTTP status outside the 2xx range, the binary SHALL write the status code and response body to stderr and exit with status code `3`. Stdout MUST remain empty.

#### Scenario: Server returns 4xx for invalid input

- **WHEN** the engine responds with HTTP 400 and a JSON error body
- **THEN** stdout is empty, stderr contains the status code and body content, and the exit code is `3`

### Requirement: Serve mode preserved when no subcommand is provided

When the binary is invoked without `search` or `read`, it SHALL behave exactly as today: parse `--port`, `--bind`, `--data-dir`, `--embeddings-dir` flags (and their environment-variable equivalents) and start the HTTP + MCP server. No existing flag or behavior is removed.

#### Scenario: Backwards-compatible serve invocation

- **WHEN** the user runs `context-engine-rs --port 8080 --bind 0.0.0.0`
- **THEN** the binary boots the IndexEngine and HTTP server on `0.0.0.0:8080` exactly as in versions prior to this change

#### Scenario: Bare invocation defaults preserved

- **WHEN** the user runs `context-engine-rs` with no arguments
- **THEN** the binary boots the server on `127.0.0.1:6699` using the documented data directory precedence
