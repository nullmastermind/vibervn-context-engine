## 1. CLI parser refactor

- [x] 1.1 Add `Commands` enum (`Search { query, repo, server }`, `Read { repo, file, query, top_k, server }`) in `src/main.rs`
- [x] 1.2 Add `command: Option<Commands>` field to the `Cli` struct, leaving the existing `port`, `bind`, `data_dir`, `embeddings_dir` fields and their `env` attributes untouched
- [x] 1.3 Verify `cargo build` succeeds and `context-engine-rs --help` lists both subcommands while `context-engine-rs` (no args) still parses the legacy flags ← (verify: legacy invocations like `--port 8080 --bind 0.0.0.0` still parse; `search --help` and `read --help` show their flags)

## 2. CLI client module

- [x] 2.1 Create `src/cli.rs` with `pub async fn run_search(...)` and `pub async fn run_read(...)` returning `std::process::ExitCode`
- [x] 2.2 Implement `resolve_base_url(flag: Option<&str>)` that applies precedence flag > `CONTEXT_ENGINE_URL` env > default `http://127.0.0.1:6699`, trimming any trailing slash
- [x] 2.3 Implement a shared `reqwest::Client` builder with connect timeout 5 s and request timeout 60 s using `rustls-tls` (already enabled in `Cargo.toml`)
- [x] 2.4 In `run_search`, POST `{"information_request": query, "workspace_full_path": repo}` to `<base>/api/mcp-tool` and print response field `result` on stdout
- [x] 2.5 In `run_read`, POST `{"workspace_full_path": repo, "file_path": file, "information_request": query, "top_k": top_k?}` (skip `top_k` when None) to `<base>/api/mcp-tool/file-retrieval` and print response field `result` on stdout
- [x] 2.6 Map errors to exit codes: connect/timeout/DNS → exit `2` with stderr message naming the URL; non-2xx → exit `3` with stderr containing status + body; success → exit `0`
- [x] 2.7 Register `pub mod cli;` in `src/lib.rs`
- [x] 2.8 Wire `main.rs` to early-return into `cli::run_search` / `cli::run_read` when a subcommand is present, before the home-dir probe and IndexEngine boot ← (verify: subcommand path never opens RocksDB, never reads settings.json, never starts IndexEngine — confirm by running search with serve mode also active and observe no lock collision)

## 3. Tests

- [x] 3.1 Add unit test in `src/cli.rs` for `resolve_base_url`: covers default, env-only, flag-only, flag-overrides-env, and trailing-slash trimming
- [x] 3.2 Add unit test for request body serialization: `read` with `top_k = None` MUST omit the field; `read` with `top_k = Some(10)` MUST include `"top_k": 10`
- [x] 3.3 Add integration test under `tests/cli_subcommand.rs` that boots the server (using existing test helpers if available, or a `tokio::spawn`-ed `server::build_router`) and runs `run_search` / `run_read` against it, asserting stdout matches the same `run_codebase_retrieval` / `run_file_retrieval` output ← (verify: end-to-end CLI → REST proxy → funnel emits byte-identical text to direct funnel call)

## 4. Verification

- [x] 4.1 Run `cargo build --release` and confirm the binary emits subcommands in `--help`
- [ ] 4.2 Run `cargo test` and confirm all existing + new tests pass
- [ ] 4.3 Manual smoke: start server in one terminal, run `target/release/context-engine-rs search --query "MCP handler" --repo $(pwd)` in another, confirm plain-text result on stdout
- [ ] 4.4 Manual smoke: stop the server, rerun the same command, confirm exit code 2 and stderr message naming the URL ← (verify: stdout empty, stderr has actionable message, exit code is 2)
- [ ] 4.5 Manual smoke: run `context-engine-rs --port 8080` (no subcommand) and confirm it boots the server exactly as before this change
