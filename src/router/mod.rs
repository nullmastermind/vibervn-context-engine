//! ROUTER mode: the lightweight front-end of the process-per-project design.
//!
//! The router holds NO per-repo index — no `IndexEngine`, no open RocksDB
//! handle, no resident vector shard. Its job is:
//! - serve the UI (`/`) and global config endpoints (`/api/config`) natively,
//! - render the repo list + cold `/index-stats` / `/graph` from per-repo
//!   [`sidecar`] files (no DB open, no worker spawn),
//! - reverse-proxy every per-repo operation (query, index, MCP, chunks, chat,
//!   SSE events) to an on-demand worker process via [`proxy`], spawning and
//!   reaping workers through the [`registry`] + [`spawn`] + [`jobobject`].
//!
//! This is what bounds resident memory to the set of repos ACTIVE within the
//! worker idle window, instead of every configured repo.

pub mod jobobject;
pub mod mcp_proxy;
pub mod plan;
pub mod proxy;
pub mod registry;
pub mod sidecar;
pub mod spawn;
pub mod worker;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::Json;
use axum::extract::{Path as AxumPath, Query as AxumQuery, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, delete, get, post};
use axum::{Router, extract::Request};
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use serde_json::json;

use crate::config::{Settings, default_data_dir, default_embeddings_dir, ensure_dir_and_load};
use crate::engine_boot::set_rocksdb_memory_bounds;
use crate::store::normalize_repo_path;

use self::jobobject::JobObject;
use self::proxy::{ProxyCtx, acquire_and_proxy};

/// Boot inputs for router mode (mirrors the CLI flags the router accepts).
pub struct RouterBootOptions {
    pub data_dir: Option<PathBuf>,
    pub embeddings_dir: Option<PathBuf>,
    pub bind: String,
    /// Override for the home directory (settings.json location). `None` uses the
    /// real `dirs::home_dir()` (production). Tests inject a TempDir so the router
    /// reads a hermetic settings.json instead of the developer's real config —
    /// the same explicit-home-dir pattern `build_router` / `boot_engine` use.
    pub home_dir: Option<PathBuf>,
    /// Override for the worker executable path. `None` uses
    /// `std::env::current_exe()` (production: the router IS the same binary it
    /// spawns as `--worker`). Tests inject `CARGO_BIN_EXE_context-engine-rs`
    /// because the test harness's own `current_exe()` is the test binary, not
    /// the context-engine binary.
    pub worker_exe: Option<PathBuf>,
}

/// Router-mode shared state. Deliberately small: the proxy context (client +
/// worker registry + spawn inputs), the resolved data dir (for sidecar reads),
/// and the home dir (for settings.json).
#[derive(Clone)]
pub struct RouterState {
    pub home_dir: PathBuf,
    pub data_dir: PathBuf,
    /// Embedding-cache root — for the native `/api/embedding-cache` purge (a
    /// host-level op on the shared content-addressed cache; no repo/worker
    /// involved).
    pub embeddings_dir: PathBuf,
    pub proxy: ProxyCtx,
}

/// Build the router-mode axum app: load settings + resolve dirs (NO IndexEngine,
/// NO repo DB opens), create the Job Object, and wire global + proxy routes.
pub async fn build_router_app(opts: RouterBootOptions) -> Result<Router> {
    // The router does not open RocksDB, but a worker it spawns will — and the
    // worker reads these same env-derived bounds at its own boot. Setting them
    // here is harmless and keeps parity if the router is ever extended.
    set_rocksdb_memory_bounds();

    let home_dir = match opts.home_dir.clone() {
        Some(h) => h,
        None => dirs::home_dir().context("could not determine home directory")?,
    };
    let settings = ensure_dir_and_load(&home_dir).context("could not load settings")?;

    let data_dir = opts
        .data_dir
        .clone()
        .or_else(|| settings.data_dir.clone())
        .unwrap_or_else(|| default_data_dir(&home_dir));
    let embeddings_dir = opts
        .embeddings_dir
        .clone()
        .or_else(|| settings.embeddings_dir.clone())
        .unwrap_or_else(|| default_embeddings_dir(&home_dir));

    // The Job Object guarantees workers die if the router dies (kill-on-close).
    let job = Arc::new(JobObject::new().unwrap_or_else(|| {
        // new() already logged; this branch is unreachable on the no-op shim and
        // the Windows impl returns None only after logging. Build a shim-equivalent
        // by retrying (the non-windows shim always returns Some).
        JobObject::new().expect("job object shim")
    }));

    // Args every worker inherits so it resolves the SAME dirs + bind as the
    // router. Port is added per-spawn (`--port 0`).
    let exe = match opts.worker_exe.clone() {
        Some(e) => e,
        None => std::env::current_exe().context("resolve current exe for worker spawn")?,
    };
    let mut worker_args = vec!["--bind".to_string(), opts.bind.clone()];
    if let Some(d) = &opts.data_dir {
        worker_args.push("--data-dir".to_string());
        worker_args.push(d.to_string_lossy().to_string());
    }
    if let Some(e) = &opts.embeddings_dir {
        worker_args.push("--embeddings-dir".to_string());
        worker_args.push(e.to_string_lossy().to_string());
    }

    let state = RouterState {
        home_dir,
        data_dir: data_dir.clone(),
        embeddings_dir,
        proxy: ProxyCtx::new(job, exe, worker_args),
    };

    // Global `/mcp` PROXYING service. Repo-addressed per call: each tool call's
    // `workspace_full_path` selects the worker the call is forwarded to (see
    // `mcp_proxy::ProxyMcpHandler`). Built with the same rmcp StreamableHttp
    // plumbing the monolith used, so MCP clients connecting to `/mcp` see an
    // identical surface. Enabled-tool gating reads the same setting.
    let mcp_proxy_ctx = state.proxy.clone();
    let enabled_tools = settings.enabled_mcp_tools.clone();
    let bind_host = opts.bind.clone();
    let mcp_config = {
        let is_loopback = matches!(bind_host.as_str(), "127.0.0.1" | "localhost" | "::1");
        if is_loopback {
            StreamableHttpServerConfig::default()
        } else {
            StreamableHttpServerConfig::default().with_allowed_hosts(vec![
                bind_host.clone(),
                "localhost".to_string(),
                "127.0.0.1".to_string(),
                "::1".to_string(),
            ])
        }
    };
    let mcp_service = StreamableHttpService::new(
        move || {
            Ok(mcp_proxy::ProxyMcpHandler::new(
                mcp_proxy_ctx.clone(),
                &enabled_tools,
            ))
        },
        Arc::new(LocalSessionManager::default()),
        mcp_config,
    );

    // One-time migration backfill: write sidecars for any already-indexed repo
    // that lacks one (pre-router monolith indexes, or indexes built before
    // sidecars existed). Runs ONCE in the background at boot — it opens each
    // sidecar-less repo's DB exactly once to read the real file count, writes the
    // sidecar only when count>0 (so a phantom/empty dir never becomes a false
    // "indexed"), then drops the handle. Best-effort + lock-safe: a repo whose
    // worker is already mid-open is skipped (that worker writes its own sidecar
    // at boot). After the first boot every real repo has a sidecar, so this is a
    // no-op thereafter. Backgrounded so it never delays the listener binding.
    {
        let data_dir_bf = data_dir.clone();
        let home_bf = state.home_dir.clone();
        tokio::spawn(async move {
            backfill_missing_sidecars(&home_bf, &data_dir_bf).await;
        });
    }

    let app = Router::new()
        // ── Global endpoints served NATIVELY (no worker, no DB) ──────────────
        .route("/", get(serve_index))
        .route("/api/config", get(get_config).put(put_config))
        .route("/api/repos", get(list_repos))
        // Host-level ops (no repo index): served natively by the router.
        .route("/api/embedding-cache", delete(delete_embedding_cache))
        .route("/api/defender-status", get(get_defender_status))
        .route("/api/defender-exclude", post(post_defender_exclude))
        // Plan endpoints proxy the admin gateway (no engine needed) — same as
        // the monolith, just hosted by the router.
        .route("/api/plan/packages", get(plan::packages))
        .route("/api/plan/payment-methods", get(plan::payment_methods))
        .route("/api/plan/checkout", post(plan::checkout))
        .route("/api/plan/orders/:invoice/status", get(plan::order_status))
        .route("/api/plan/usage", get(plan::usage))
        .route("/api/plan/free-trial", get(plan::free_trial))
        .route("/api/plan/free-trial/claim", post(plan::free_trial_claim))
        // Fan-out / aggregate across repos.
        .route("/api/index-all", post(post_index_all))
        .route("/api/index-status", get(get_index_status))
        // Body-addressed-by-repo → extract repo from JSON body, then proxy.
        .route("/api/query", post(proxy_query_by_body))
        .route("/api/mcp-tool", post(proxy_mcp_tool_by_body))
        .route(
            "/api/mcp-tool/file-retrieval",
            post(proxy_file_retrieval_by_body),
        )
        // ── PER-REPO ROUTES — split by spawn policy (audit each line) ────────
        // SPAWN-BLOCKED (view): explicit routes BEFORE the catch-all. Each either
        // proxies to an ALREADY-LIVE worker (fresh data) or serves cold from a
        // sidecar/native — NEVER `acquire_and_proxy` (the spawn path). Opening a
        // repo detail hits only these, so it starts 0 workers.
        .route("/api/repos/:repo_id/status", get(repo_status_cold_or_proxy))
        .route(
            "/api/repos/:repo_id/index-stats",
            get(repo_index_stats_cold_or_proxy),
        )
        .route("/api/repos/:repo_id/graph", get(repo_graph_cold_or_proxy))
        .route("/api/repos/:repo_id/files", get(repo_files_cold_or_proxy))
        .route(
            "/api/repos/:repo_id/ignored-files",
            get(repo_ignored_files_cold_or_proxy),
        )
        .route(
            "/api/repos/:repo_id/index-events",
            get(repo_index_events_cold_or_proxy),
        )
        .route(
            "/api/repos/:repo_id/cancel-index",
            post(repo_cancel_index_cold_or_proxy),
        )
        .route(
            "/api/repos/:repo_id/chat/:conversation_id",
            delete(repo_chat_delete_cold_or_proxy),
        )
        // NATIVE-KILL (no spawn) on DELETE; SPAWN-ALLOWED on POST. The same path
        // `/index` serves both, so they MUST share one explicit route — a
        // DELETE-only route would 405 the POST instead of letting it reach the
        // catch-all. DELETE → router-native (kill worker, remove on-disk state);
        // POST → proxy to an on-demand worker that runs the index.
        .route(
            "/api/repos/:repo_id/index",
            delete(repo_delete_index_native).post(proxy_repo_index_post),
        )
        // SPAWN-ALLOWED (action): everything else per-repo → on-demand worker.
        // (index POST, rebuild, chunks, ignore-file*, chat POST, mcp-setup, …)
        .route("/api/repos/:repo_id", any(proxy_repo_root))
        .route("/api/repos/:repo_id/*rest", any(proxy_repo_subpath))
        // ── MCP per-repo endpoint also proxies (action) ──────────────────────
        .route("/mcp-repo/:repo_id", any(proxy_mcp_repo))
        .with_state(state)
        // Global multi-repo `/mcp`: proxying handler (repo per call).
        .merge(Router::new().nest_service("/mcp", mcp_service));

    Ok(app)
}

// ── Native global handlers ──────────────────────────────────────────────────

async fn serve_index() -> impl IntoResponse {
    // Same single-page UI the standalone server serves. Embedded at compile time.
    let html = include_str!("../assets/index.html");
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], html)
}

async fn get_config(State(state): State<RouterState>) -> Response {
    let home = state.home_dir.clone();
    match tokio::task::spawn_blocking(move || ensure_dir_and_load(&home)).await {
        Ok(Ok(settings)) => {
            Json(serde_json::to_value(&settings).unwrap_or_default()).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("{e}") })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("internal error: {e}") })),
        )
            .into_response(),
    }
}

async fn put_config(State(state): State<RouterState>, body: axum::body::Bytes) -> Response {
    // The router owns settings.json (PUT writes disk FIRST). Workers re-read the
    // file on a mtime gate (index + query paths), so a key/model change here
    // reaches a live worker on its next operation — no IPC, no worker restart.
    let value: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("invalid JSON body: {e}") })),
            )
                .into_response();
        }
    };
    let mut settings: Settings = match serde_json::from_value(value) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("invalid settings: {e}") })),
            )
                .into_response();
        }
    };
    settings.version = crate::config::CURRENT_VERSION;
    settings.repos = settings
        .repos
        .iter()
        .map(|r| normalize_repo_path(r))
        .collect();

    // FIRST-ADD auto-index: capture the repo set BEFORE the write so we can diff
    // out repos that are genuinely NEW in this PUT. Read old from disk now —
    // after the write the old set is gone. A read failure (settings.json doesn't
    // exist yet / unreadable) → treat old as empty, so on a first-ever PUT every
    // configured repo counts as new and gets its initial index (correct: "added
    // → it indexes itself"). Keyed by normalize_repo_path to match how repos are
    // stored + how the registry/worker key them.
    let old_repos: std::collections::HashSet<String> = ensure_dir_and_load(&state.home_dir)
        .map(|s| {
            s.repos
                .into_iter()
                .map(|r| normalize_repo_path(&r))
                .collect()
        })
        .unwrap_or_default();
    let new_repos_snapshot = settings.repos.clone();

    let target = crate::config::config_path(&state.home_dir);
    let written = match tokio::task::spawn_blocking(move || {
        crate::config::write_settings_atomic(&target, &settings)?;
        Ok::<Settings, crate::config::ConfigError>(settings)
    })
    .await
    {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("{e}") })),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("internal error: {e}") })),
            )
                .into_response();
        }
    };

    // Disk write succeeded. Now fire-and-forget an initial index for each repo
    // that is NEW in this PUT (added since the prior config). This is what makes
    // "add a repo" auto-build its first index without the user clicking Index.
    // It is SPAWN-ALLOWED (an explicit user action: adding the repo), routed
    // through the same proxy path a manual index uses (spawns a worker on demand,
    // runs the index async). Detached so the PUT response is not blocked by the
    // spawn/index. A PUT that changes only keys/model/enabled-tools adds no repo
    // → the diff is empty → NOTHING spawns (the common case must stay spawn-free).
    for repo in &new_repos_snapshot {
        if !old_repos.contains(repo) {
            let proxy = state.proxy.clone();
            let repo = repo.clone();
            let path = format!("/api/repos/{}/index", urlencode_segment(&repo));
            tokio::spawn(async move {
                // Best-effort: a spawn/index failure here must not affect the
                // already-committed config write. The worker logs its own errors;
                // the user can re-trigger Index manually if needed.
                let _ = proxy::forward_json_to_worker(&proxy, &repo, &path, json!({})).await;
            });
        }
    }

    Json(serde_json::to_value(&written).unwrap_or_default()).into_response()
}

/// Repo list with light per-repo metadata read from sidecars — NO DB open, NO
/// worker spawn. A repo with no sidecar (never indexed) renders a "not indexed"
/// placeholder. Live worker presence is annotated so the UI can show "active".
async fn list_repos(State(state): State<RouterState>) -> Response {
    let settings = match ensure_dir_and_load(&state.home_dir) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("{e}") })),
            )
                .into_response();
        }
    };
    let ready = state.proxy.registry.ready_repos().await;
    let repos: Vec<_> = settings
        .repos
        .iter()
        .map(|repo| {
            let normalized = normalize_repo_path(repo);
            let side = sidecar::read_sidecar(&state.data_dir, &normalized);
            let active = ready.contains(&normalized);
            match side {
                Some(s) => json!({
                    "repo": normalized,
                    "file_count": s.file_count,
                    "last_indexed_at": s.last_indexed_at,
                    "state": if active { "active" } else { s.state.as_str() },
                    "embedding_model": s.embedding_model,
                    "embedding_dim": s.embedding_dim,
                    "worker_active": active,
                }),
                None => json!({
                    "repo": normalized,
                    "file_count": 0,
                    "last_indexed_at": null,
                    "state": "not_indexed",
                    "worker_active": active,
                }),
            }
        })
        .collect();
    Json(json!({ "repos": repos })).into_response()
}

/// Resolve the worker key (normalized repo path) for an `:repo_id` path segment.
///
/// The `/api/repos/:repo_id/*` family encodes the repo as URL_SAFE_NO_PAD
/// base64 (worker side: `decode_repo_id`). The router must resolve the SAME
/// normalized path to pick the right worker key + spawn argument. If the segment
/// isn't valid base64 (a caller passed a raw path), fall back to normalizing it
/// directly so both encodings work. NOTE: the forwarded HTTP path keeps the
/// ORIGINAL `:repo_id` segment untouched, so the worker re-decodes it itself —
/// we only decode here to choose the worker.
fn resolve_repo_id(repo_id: &str) -> String {
    use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
    URL_SAFE_NO_PAD
        .decode(repo_id)
        .ok()
        .and_then(|b| String::from_utf8(b).ok())
        .map(|s| normalize_repo_path(&s))
        .unwrap_or_else(|| normalize_repo_path(repo_id))
}

/// Resolve the worker key for a `/mcp-repo/:repo_name` segment, which uses the
/// SANITIZED-name scheme (worker side: scan settings.repos for a matching
/// `sanitize_repo_name`). The router loads settings and does the same scan so it
/// spawns/selects the worker with the real path. Returns `None` if no
/// configured repo sanitizes to `repo_name`.
fn resolve_mcp_repo_name(home_dir: &std::path::Path, repo_name: &str) -> Option<String> {
    let settings = ensure_dir_and_load(home_dir).ok()?;
    settings
        .repos
        .iter()
        .find(|r| crate::store::sanitize_repo_name(r) == repo_name)
        .map(|r| normalize_repo_path(r))
}

/// `/index-stats`: if a worker is live, proxy for fresh numbers; else serve the
/// sidecar's light counts (cold), or a not-indexed placeholder. Uses
/// `proxy_if_live` (race-safe: never spawns — if the worker died between the
/// residency check and the proxy, it falls through to cold instead of spawning).
async fn repo_index_stats_cold_or_proxy(
    State(state): State<RouterState>,
    AxumPath(repo_id): AxumPath<String>,
    req: Request,
) -> Response {
    let repo = resolve_repo_id(&repo_id);
    // Proxy ONLY if a worker is already live; otherwise serve cold (no spawn).
    if let Some(resp) = proxy::proxy_if_live(&state.proxy, &repo, req).await {
        return resp;
    }
    // Cold: sidecar or placeholder. Do NOT spawn a worker just to show stats.
    match sidecar::read_sidecar(&state.data_dir, &repo) {
        Some(s) => Json(json!({
            "repo": repo,
            "files": s.file_count,
            "chunks": null,
            "symbols": null,
            "embedding_model": s.embedding_model,
            "embedding_dim": s.embedding_dim,
            "state": "indexed_cold",
            "last_indexed_at": s.last_indexed_at,
            "note": "served from sidecar; open the repo to load full stats",
        }))
        .into_response(),
        None => Json(json!({
            "repo": repo,
            "files": 0, "chunks": 0, "symbols": 0,
            "embedding_dim": null,
            "state": "not_indexed",
            "last_indexed_at": null,
        }))
        .into_response(),
    }
}

/// `/graph`: if a worker is live, proxy for the fresh bounded call-graph; else
/// serve the cached `graph.json` sidecar (stale but real — accepted), or an empty
/// `cold:true` placeholder if no cache yet. NEVER spawns (uses `proxy_if_live`).
async fn repo_graph_cold_or_proxy(
    State(state): State<RouterState>,
    AxumPath(repo_id): AxumPath<String>,
    req: Request,
) -> Response {
    let repo = resolve_repo_id(&repo_id);
    if let Some(resp) = proxy::proxy_if_live(&state.proxy, &repo, req).await {
        return resp;
    }
    // Cold: serve the cached graph payload written at index-complete. Stale is
    // acceptable (per the plan) — far better UX than a blank box or a spawn.
    match sidecar::read_aux_json::<crate::store::ops::CallGraph>(&state.data_dir, &repo, "graph") {
        Some(g) => Json(json!({
            "nodes": g.nodes,
            "edges": g.edges,
            "truncated": g.truncated,
            "cold": true,
        }))
        .into_response(),
        // No cached graph (never indexed, or pre-cache index) → empty placeholder.
        None => Json(json!({
            "nodes": [], "edges": [], "truncated": false,
            "cold": true,
            "note": "graph loads once the repo is indexed",
        }))
        .into_response(),
    }
}

/// `/status`: live worker → proxy real status; else synthesize a cold status
/// from the sidecar (indexed count) or not_indexed. No spawn.
async fn repo_status_cold_or_proxy(
    State(state): State<RouterState>,
    AxumPath(repo_id): AxumPath<String>,
    req: Request,
) -> Response {
    let repo = resolve_repo_id(&repo_id);
    if let Some(resp) = proxy::proxy_if_live(&state.proxy, &repo, req).await {
        return resp;
    }
    match sidecar::read_sidecar(&state.data_dir, &repo) {
        Some(s) => Json(json!({
            "repo": repo,
            "state": "idle",
            "indexed_files": s.file_count,
            "total_files": s.file_count,
            "last_indexed_at": s.last_indexed_at,
        }))
        .into_response(),
        None => Json(json!({
            "repo": repo,
            "state": "idle",
            "indexed_files": 0,
            "total_files": 0,
            "last_indexed_at": null,
        }))
        .into_response(),
    }
}

/// `/files`: live worker → proxy fresh file list; else serve the cached
/// `files.json` sidecar (stale accepted), or an empty list. No spawn.
async fn repo_files_cold_or_proxy(
    State(state): State<RouterState>,
    AxumPath(repo_id): AxumPath<String>,
    req: Request,
) -> Response {
    let repo = resolve_repo_id(&repo_id);
    if let Some(resp) = proxy::proxy_if_live(&state.proxy, &repo, req).await {
        return resp;
    }
    match sidecar::read_aux_json::<serde_json::Value>(&state.data_dir, &repo, "files") {
        Some(files) => {
            Json(json!({ "files": files, "truncated": false, "cold": true })).into_response()
        }
        None => Json(json!({ "files": [], "truncated": false, "cold": true })).into_response(),
    }
}

/// `/ignored-files`: this is a function of settings (ignore list), not the index
/// — but the worker computes it against the live tree. Cold: live worker →
/// proxy; else empty (the UI shows none until the repo is active). No spawn.
async fn repo_ignored_files_cold_or_proxy(
    State(state): State<RouterState>,
    AxumPath(repo_id): AxumPath<String>,
    req: Request,
) -> Response {
    let repo = resolve_repo_id(&repo_id);
    if let Some(resp) = proxy::proxy_if_live(&state.proxy, &repo, req).await {
        return resp;
    }
    Json(json!({ "ignored": [], "cold": true })).into_response()
}

/// `/index-events` (SSE progress): only meaningful while indexing, which always
/// follows an action that spawned a worker. Cold (no worker) → close the stream
/// immediately so the browser's EventSource gets a clean end, NOT a spawn. When
/// the user triggers Index, a worker comes up and the UI re-opens this stream.
async fn repo_index_events_cold_or_proxy(
    State(state): State<RouterState>,
    AxumPath(repo_id): AxumPath<String>,
    req: Request,
) -> Response {
    let repo = resolve_repo_id(&repo_id);
    if let Some(resp) = proxy::proxy_if_live(&state.proxy, &repo, req).await {
        return resp;
    }
    // Cold: 204 No Content — EventSource sees the connection close and stops;
    // there is no progress to stream when nothing is indexing.
    StatusCode::NO_CONTENT.into_response()
}

/// `/cancel-index`: cancels an in-flight run. Cold (no worker) → no run exists →
/// no-op 200. NEVER spawns a worker just to "cancel" nothing.
async fn repo_cancel_index_cold_or_proxy(
    State(state): State<RouterState>,
    AxumPath(repo_id): AxumPath<String>,
    req: Request,
) -> Response {
    let repo = resolve_repo_id(&repo_id);
    if let Some(resp) = proxy::proxy_if_live(&state.proxy, &repo, req).await {
        return resp;
    }
    Json(json!({ "status": "ok", "note": "no active worker; nothing to cancel" })).into_response()
}

/// `DELETE /chat/:cid`: the chat ConversationStore is IN-MEMORY PER-WORKER
/// (chat.rs ConversationStore = Mutex<HashMap>, lives in each worker's AppState).
/// So a conversation only exists in a live worker's RAM. Cold (no worker) → there
/// is nothing to delete anywhere → no-op 200. Live worker → proxy the delete.
/// MUST NOT spawn — spawning a fresh worker would create an empty store with no
/// such conversation, making the delete pointless AND violating the view-no-spawn
/// rule.
async fn repo_chat_delete_cold_or_proxy(
    State(state): State<RouterState>,
    AxumPath((repo_id, _cid)): AxumPath<(String, String)>,
    req: Request,
) -> Response {
    let repo = resolve_repo_id(&repo_id);
    if let Some(resp) = proxy::proxy_if_live(&state.proxy, &repo, req).await {
        return resp;
    }
    Json(json!({ "status": "ok", "note": "no active worker; conversation not resident" }))
        .into_response()
}

// ── Proxy handlers ──────────────────────────────────────────────────────────

async fn proxy_repo_root(
    State(state): State<RouterState>,
    AxumPath(repo_id): AxumPath<String>,
    req: Request,
) -> Response {
    let repo = resolve_repo_id(&repo_id);
    acquire_and_proxy(&state.proxy, &repo, req).await
}

async fn proxy_repo_subpath(
    State(state): State<RouterState>,
    AxumPath((repo_id, _rest)): AxumPath<(String, String)>,
    req: Request,
) -> Response {
    let repo = resolve_repo_id(&repo_id);
    acquire_and_proxy(&state.proxy, &repo, req).await
}

async fn proxy_mcp_repo(
    State(state): State<RouterState>,
    AxumPath(repo_id): AxumPath<String>,
    req: Request,
) -> Response {
    // `/mcp-repo/:repo_id` uses the SANITIZED-name scheme (not base64). Resolve
    // it to the real path so we spawn/select the correct worker; the forwarded
    // path keeps the original sanitized segment so the worker's own resolver
    // runs unchanged.
    match resolve_mcp_repo_name(&state.home_dir, &repo_id) {
        Some(repo) => acquire_and_proxy(&state.proxy, &repo, req).await,
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("unknown repo: {repo_id}") })),
        )
            .into_response(),
    }
}

/// `POST /api/repos/:id/index` — SPAWN-ALLOWED: an index is a user action that
/// needs a worker. Proxy to an on-demand worker (spawning if cold).
async fn proxy_repo_index_post(
    State(state): State<RouterState>,
    AxumPath(repo_id): AxumPath<String>,
    req: Request,
) -> Response {
    let repo = resolve_repo_id(&repo_id);
    acquire_and_proxy(&state.proxy, &repo, req).await
}

/// `DELETE /api/repos/:id/index[?remove_repo=true]` — NATIVE-KILL, no spawn.
///
/// Router-native because removal needs NO engine and MUST NOT spawn a worker.
/// Order is load-bearing for the RocksDB exclusive-LOCK race (see store::
/// remove_index_dir + Registry::kill):
///   1. `registry.kill(repo)` — `child.kill()` then `child.wait()` BLOCKS until
///      the worker process is reaped, so the LOCK is being released before we
///      touch the directory (in-process drop would NOT free it; process exit
///      does). Idempotent: no live worker → no-op.
///   2. Bump `repo_generations` (and drop the repo from `settings.repos` when
///      `remove_repo=true`) and persist to disk atomically — so any later open
///      targets a FRESH generation path the just-killed handle never held, and a
///      reload mid-teardown can't resurrect the repo.
///   3. `store::remove_index_dir` on the OLD generation — its 30s backoff rides
///      out the async LOCK drain the killed process is still completing.
///   4. Remove ALL sidecars (meta+graph+files) so the cold view stops serving
///      stale data for a just-deleted index.
///
/// A drain outliving the 30s budget is reclaimed by `sweep_stale_generations` at
/// next boot (existing mechanism).
async fn repo_delete_index_native(
    State(state): State<RouterState>,
    AxumPath(repo_id): AxumPath<String>,
    AxumQuery(params): AxumQuery<std::collections::HashMap<String, String>>,
) -> Response {
    let repo = resolve_repo_id(&repo_id);
    let remove_repo = params
        .get("remove_repo")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    // 1. Kill the worker (sync wait → process gone → LOCK release begins).
    state.proxy.registry.kill(&repo).await;

    // Resolve the CURRENT generation (the on-disk dir to remove) before the bump.
    let settings = match ensure_dir_and_load(&state.home_dir) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("could not load settings: {e}") })),
            )
                .into_response();
        }
    };
    let current_generation = settings.repo_generation(&repo);

    // 2. Bump generation (+ drop repo if remove_repo) and persist atomically. The
    //    router has no in-memory settings cache (it reads disk per request), so a
    //    single durable disk write is the whole commit.
    let next_generation = current_generation.saturating_add(1);
    let to_write = {
        let mut s = settings.clone();
        s.repo_generations.insert(repo.clone(), next_generation);
        if remove_repo {
            s.repos.retain(|r| r != &repo);
        }
        s
    };
    let target = crate::config::config_path(&state.home_dir);
    if let Err(e) = tokio::task::spawn_blocking(move || {
        crate::config::write_settings_atomic(&target, &to_write)
    })
    .await
    .map_err(|e| format!("join: {e}"))
    .and_then(|r| r.map_err(|e| format!("{e}")))
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("failed to persist generation bump: {e}") })),
        )
            .into_response();
    }

    // 3. Remove the OLD generation directory (30s LOCK-drain retry). Ungated:
    //    the durable bump means nothing targets the old path anymore.
    let removed =
        crate::store::remove_old_generation_dir(&state.data_dir, &repo, current_generation).await;

    // 4. Drop all sidecars so cold view doesn't serve stale data post-removal.
    sidecar::remove_all_sidecars(&state.data_dir, &repo);

    if removed {
        Json(json!({ "status": "ok" })).into_response()
    } else {
        Json(json!({
            "status": "pending",
            "message": "old index directory not fully removed yet; it will be reclaimed on next restart"
        }))
        .into_response()
    }
}

// ── Body-addressed-by-repo proxies ───────────────────────────────────────────
// `/api/query` and `/api/mcp-tool[/file-retrieval]` carry the target repo INSIDE
// the JSON body (`repo` / `workspace_full_path`), not in the path. The router
// must peek that field to pick the worker, then forward the ORIGINAL body
// unchanged (peek, don't reserialize — so a field the router doesn't model
// survives to the worker). We buffer the body once, extract the key, and rebuild
// a Request the streaming proxy can forward.

/// Pull a string field out of a buffered JSON body without consuming it.
fn peek_body_field(bytes: &[u8], field: &str) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()?
        .get(field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Shared core: buffer body, peek `repo_field`, normalize, rebuild Request, proxy.
async fn proxy_by_body_field(state: &RouterState, repo_field: &str, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let bytes = match axum::body::to_bytes(body, 16 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("could not read request body: {e}") })),
            )
                .into_response();
        }
    };
    let repo = match peek_body_field(&bytes, repo_field).map(|r| normalize_repo_path(r.trim())) {
        Some(r) if !r.is_empty() => r,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": format!("`{repo_field}` is required to route the request to a repository worker"),
                })),
            )
                .into_response();
        }
    };
    // Rebuild the Request with the SAME parts + buffered body for the proxy.
    let rebuilt = Request::from_parts(parts, axum::body::Body::from(bytes));
    acquire_and_proxy(&state.proxy, &repo, rebuilt).await
}

async fn proxy_query_by_body(State(state): State<RouterState>, req: Request) -> Response {
    proxy_by_body_field(&state, "repo", req).await
}

async fn proxy_mcp_tool_by_body(State(state): State<RouterState>, req: Request) -> Response {
    proxy_by_body_field(&state, "workspace_full_path", req).await
}

async fn proxy_file_retrieval_by_body(State(state): State<RouterState>, req: Request) -> Response {
    proxy_by_body_field(&state, "workspace_full_path", req).await
}

// ── Host-native global handlers (no repo/worker) ─────────────────────────────
// These operate on host-level resources (Defender exclusions, the shared
// content-addressed embedding cache), not any repo index — so the router runs
// them directly, reusing the same public ops the monolith server calls.

async fn delete_embedding_cache(
    State(state): State<RouterState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let older_than = match params.get("older_than").map(|s| s.as_str()) {
        Some("all") | None => None,
        Some("30d") => Some(std::time::Duration::from_secs(30 * 24 * 3600)),
        Some(other) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("invalid older_than value: {other}; use 'all' or '30d'") })),
            )
                .into_response();
        }
    };
    let embeddings_dir = state.embeddings_dir.clone();
    match tokio::task::spawn_blocking(move || {
        crate::embedding::cache::EmbeddingCache::purge_global(&embeddings_dir, older_than)
    })
    .await
    {
        Ok(pr) => Json(json!({ "deleted": pr.deleted, "errors": pr.errors })).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("purge task failed: {e}") })),
        )
            .into_response(),
    }
}

async fn get_defender_status(State(state): State<RouterState>) -> Response {
    let data_dir = state.data_dir.to_string_lossy().to_string();
    match tokio::task::spawn_blocking(move || crate::defender::check_status(&data_dir)).await {
        Ok(s) => Json(json!(s)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("defender check failed: {e}") })),
        )
            .into_response(),
    }
}

async fn post_defender_exclude(State(state): State<RouterState>) -> Response {
    let data_dir = state.data_dir.to_string_lossy().to_string();
    match tokio::task::spawn_blocking(move || crate::defender::add_exclusions(&data_dir)).await {
        Ok(r) => {
            let code = if r.success {
                StatusCode::OK
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (code, Json(json!(r))).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("defender exclude failed: {e}") })),
        )
            .into_response(),
    }
}

// ── Fan-out / aggregate across repos ─────────────────────────────────────────

/// POST /api/index-all — trigger an index on EVERY configured repo. In the
/// process-per-project model this means: for each repo, ensure a worker is up
/// (spawn if cold) and forward a `POST /api/repos/<id>/index` to it, so the
/// worker's own `IndexEngine` runs the index. Best-effort per repo: a repo whose
/// worker can't be spawned is reported in `failed` rather than failing the whole
/// call. Returns 202 Accepted with per-repo outcomes (indexing is async on each
/// worker, exactly as the monolith's trigger_index_all was fire-and-forget).
async fn post_index_all(State(state): State<RouterState>) -> Response {
    let repos = match ensure_dir_and_load(&state.home_dir) {
        Ok(s) => s.repos,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("{e}") })),
            )
                .into_response();
        }
    };
    let mut triggered = Vec::new();
    let mut failed = Vec::new();
    for repo in repos {
        let normalized = normalize_repo_path(&repo);
        // Forward a POST index trigger to the (spawned-on-demand) worker. We use
        // the JSON helper because we only need to know it was accepted, not
        // stream a body. An empty JSON object is a fine trigger body.
        match proxy::forward_json_to_worker(
            &state.proxy,
            &normalized,
            &format!("/api/repos/{}/index", urlencode_segment(&normalized)),
            json!({}),
        )
        .await
        {
            Ok(_) => triggered.push(normalized),
            Err(e) => failed.push(json!({ "repo": normalized, "error": e })),
        }
    }
    (
        StatusCode::ACCEPTED,
        Json(json!({ "status": "accepted", "triggered": triggered, "failed": failed })),
    )
        .into_response()
}

/// GET /api/index-status — aggregate status across all repos WITHOUT forcing a
/// worker spawn. A repo with a live worker is reported as its live state
/// (proxied status); a cold repo is reported from its sidecar (durable
/// last-known) or "not_indexed". This keeps the dashboard responsive without
/// waking every project just to render status — matching the scale-to-zero goal.
async fn get_index_status(State(state): State<RouterState>) -> Response {
    let settings = match ensure_dir_and_load(&state.home_dir) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("{e}") })),
            )
                .into_response();
        }
    };
    let repos = settings.repos.clone();
    let ready = state.proxy.registry.ready_repos().await;
    let spawning = state.proxy.registry.spawning_repos().await;
    let mut out = Vec::new();
    for repo in repos {
        let normalized = normalize_repo_path(&repo);
        if ready.contains(&normalized) {
            // Live worker: GET its authoritative status (a GET endpoint that
            // returns the status object directly — NOT via forward_json_to_worker,
            // which POSTs + unwraps an MCP `{result}` envelope and would 405 here).
            if let Some(mut v) = proxy::get_json_if_live(
                &state.proxy,
                &normalized,
                &format!("/api/repos/{}/status", urlencode_segment(&normalized)),
            )
            .await
            {
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("repo".to_string(), json!(normalized));
                    obj.insert("worker_active".to_string(), json!(true));
                }
                out.push(v);
                continue;
            }
            // No live status (raced to death / transient) → fall through to cold.
        }
        // Worker mid-SPAWN (an action just triggered it, not bound yet): report
        // `indexing` so the badge shows progress during the ~1-2s spawn window,
        // NOT the cold sidecar's "not_indexed" (the exact "shows Chưa index right
        // after I triggered it" symptom). `phase:"starting"` marks it a live-ish
        // state so the UI's live-vs-cold distinction treats it as active. We do
        // NOT GET /status here — the worker isn't accepting yet; the next poll
        // (≤3s) flips to the real worker status once it's Ready.
        if spawning.contains(&normalized) {
            out.push(json!({
                "repo": normalized,
                "state": "indexing",
                "phase": "starting",
                "indexed_files": 0,
                "total_files": 0,
                "worker_active": true,
            }));
            continue;
        }
        // Cold view: the SIDECAR is the single source of truth for whether a
        // repo is indexed. We deliberately do NOT fall back to "the RocksDB dir
        // exists on disk" — `open_db` runs create_dir_all + SCHEMA_DDL on EVERY
        // open, so a dir exists for any repo a worker ever booted for, even one
        // that was never actually indexed (0 chunks) or whose first index
        // crashed mid-way. A dir-exists check would report those as "indexed"
        // (false positive) AND disagree with a live worker's /status (which reads
        // the real count). The only correct "is there real data" signal is
        // count_indexed_files, which needs a DB open — forbidden on a status
        // poll. So: sidecar present → its (real, count>0) numbers; sidecar absent
        // → not_indexed. Migrated repos (data on disk, no sidecar yet) are
        // backfilled by the one-time router-boot pass (see backfill_missing_
        // sidecars) and by each worker at its own boot, so "absent" converges to
        // the truth without ever guessing from a bare directory.
        match sidecar::read_sidecar(&state.data_dir, &normalized) {
            Some(s) => out.push(json!({
                "repo": normalized,
                "state": s.state,
                "indexed_files": s.file_count,
                "last_indexed_at": s.last_indexed_at,
                "worker_active": false,
            })),
            None => out.push(json!({
                "repo": normalized,
                "state": "not_indexed",
                "indexed_files": 0,
                "last_indexed_at": null,
                "worker_active": false,
            })),
        }
    }
    Json(out).into_response()
}

/// Percent-encode a repo path for use as a single `:repo_id` path segment when
/// the router forwards to a worker route. The worker's `decode_repo_id` expects
/// URL_SAFE_NO_PAD base64 OR a normalized path; the worker's own resolver
/// handles both. We base64 it to match the worker's `decode_repo_id` contract.
fn urlencode_segment(repo: &str) -> String {
    use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
    URL_SAFE_NO_PAD.encode(repo.as_bytes())
}

/// One-time router-boot migration: write a sidecar for every configured repo
/// that is genuinely indexed (count>0) but has no sidecar yet (pre-router
/// monolith indexes / indexes built before sidecars). Opens each such repo's DB
/// ONCE, reads the real `count_indexed_files`, and writes the sidecar only when
/// count>0 — so a phantom/empty/crashed dir (count 0) never becomes a false
/// "indexed". The handle is dropped at the end of each iteration (transient, one
/// at a time — no resident growth). Best-effort throughout: an open that fails
/// (e.g. a worker already holds the LOCK) is skipped and logged; that worker
/// writes its own sidecar at boot, so the result converges either way. Idempotent
/// — once a sidecar exists this skips the repo, so after the first boot it is a
/// pure no-op.
async fn backfill_missing_sidecars(home_dir: &std::path::Path, data_dir: &std::path::Path) {
    let settings = match ensure_dir_and_load(home_dir) {
        Ok(s) => s,
        Err(_) => return, // can't read config → nothing to backfill
    };
    for repo in &settings.repos {
        let normalized = normalize_repo_path(repo);
        // Already has a sidecar → nothing to do (the common steady-state path).
        if sidecar::read_sidecar(data_dir, &normalized).is_some() {
            continue;
        }
        let generation = settings.repo_generation(&normalized);
        // Skip if there is no on-disk dir at all — genuinely never indexed, and
        // we must NOT create one as a side effect (open_db would create_dir_all).
        if !crate::store::db_path(data_dir, &normalized, generation).exists() {
            continue;
        }
        // Open transiently to read the REAL count. A fresh handle (not the cached
        // map) so it drops at end of scope. NOTE on the LOCK: dropping this handle
        // releases the RocksDB LOCK ASYNCHRONOUSLY (a background flush, not instant
        // on Windows). That is fine here: backfill runs once at boot, well before
        // any worker is spawned, and a worker request that DOES race a just-dropped
        // backfill handle is a SEPARATE PROCESS whose `open_db` 30s retry rides out
        // the router's async drain (the router keeps draining while the worker
        // retries — verified: a worker opened a just-backfilled repo in 0s). If a
        // worker already holds the LOCK when backfill runs, the open below fails
        // fast-ish into its own retry and we skip — that worker writes its own
        // sidecar at its boot, so the result converges either way.
        let db = match crate::store::open_db(data_dir, &normalized, generation).await {
            Ok(db) => db,
            Err(e) => {
                tracing::warn!(repo = %normalized, error = %format!("{e:#}"),
                    "sidecar backfill: could not open DB (worker may hold it); skipping");
                continue;
            }
        };
        let count = crate::store::ops::count_indexed_files(&db, &normalized)
            .await
            .unwrap_or(0);
        if count == 0 {
            // Phantom/empty/crashed dir — NOT indexed. Leave no sidecar so the
            // cold view honestly reports not_indexed. Drop handle, move on.
            continue;
        }
        let last_indexed_at = crate::store::ops::get_meta(&db, "last_indexed_at")
            .await
            .ok()
            .flatten();
        let (embedding_model, embedding_dim) = (
            settings.embedding.model.clone(),
            settings.embedding.dimensions.unwrap_or(0) as u64,
        );
        let meta = sidecar::RepoSidecar {
            file_count: count,
            last_indexed_at,
            state: "indexed".to_string(),
            embedding_model,
            embedding_dim,
            schema: sidecar::SIDECAR_SCHEMA,
        };
        if let Err(e) = sidecar::write_sidecar(data_dir, &normalized, &meta) {
            tracing::warn!(repo = %normalized, error = %format!("{e:#}"),
                "sidecar backfill: write failed");
        } else {
            tracing::info!(repo = %normalized, count, "sidecar backfilled at router boot");
        }
        // Also backfill the AUX sidecars (graph + files) while the DB is open, so
        // a detail-open right after boot reads the cache instead of spawning.
        // Best-effort; the worker also self-heals these at its own boot.
        if sidecar::read_aux_json::<crate::store::ops::CallGraph>(data_dir, &normalized, "graph")
            .is_none()
            && let Ok(graph) = crate::store::ops::compute_and_cache_graph(&db).await
        {
            let _ = sidecar::write_aux_json(data_dir, &normalized, "graph", &graph);
        }
        if sidecar::read_aux_json::<serde_json::Value>(data_dir, &normalized, "files").is_none()
            && let Ok(files) = crate::store::ops::files_page(&db, &normalized, 2000, None).await
        {
            let _ = sidecar::write_aux_json(data_dir, &normalized, "files", &files);
        }
        // `db` drops here → handle released before the next iteration.
    }
}
