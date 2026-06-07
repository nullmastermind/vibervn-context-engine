use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::convert::Infallible;
use std::time::Duration;

use axum::{
    extract::{Json, Path, Query, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    response::sse::{Event, Sse},
    routing::{delete, get, post, put},
    Router,
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::Deserialize;
use serde_json::{Value, json};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tokio::sync::RwLock;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService,
    session::local::LocalSessionManager,
};

use crate::config::{
    ConfigError, CURRENT_VERSION, Settings, ensure_dir_and_load, write_settings_atomic,
    config_path,
};
use crate::defender;
use crate::embedding::voyage::VoyageClient;
use crate::indexing::IndexEngine;
use crate::llm::LlmClient;
use crate::mcp::{McpHandler, run_codebase_retrieval};
use crate::path_in_repo;
use crate::store;
use crate::query;

// ─── IntoResponse for ConfigError ─────────────────────────────────────────

impl IntoResponse for ConfigError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ConfigError::Io { op, source } => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to {op} settings: {source}"),
            ),
            ConfigError::Parse(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("settings.json is corrupt: {e}; fix or delete the file"),
            ),
            ConfigError::VersionTooNew { found } => (
                StatusCode::CONFLICT,
                format!(
                    "settings.json was written by a newer version of context-engine \
                     (version {found}); upgrade the binary or restore an older settings.json"
                ),
            ),
            ConfigError::MigrationFailed { from, to, detail } => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("migration from v{from} to v{to} failed: {detail}"),
            ),
        };

        let body = json!({ "error": message });
        (status, Json(body)).into_response()
    }
}

// ─── App state ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    /// Resolved home directory, used to locate settings.json.
    pub home_dir: PathBuf,
    /// Shared index engine.
    pub index_engine: Arc<IndexEngine>,
    /// Per-repo SurrealDB handles, keyed by repo path.
    pub repo_dbs: Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    /// Shared live settings — the single source of truth.
    /// All mutations go through this handle AND are written to disk first.
    pub settings: Arc<RwLock<crate::config::Settings>>,
}

// ─── Router ────────────────────────────────────────────────────────────────

pub fn build_router(
    home_dir: PathBuf,
    index_engine: Arc<IndexEngine>,
    repo_dbs: Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    settings: Arc<RwLock<crate::config::Settings>>,
    bind_host: &str,
) -> Router {
    let state = AppState { home_dir: home_dir.clone(), index_engine: index_engine.clone(), repo_dbs: repo_dbs.clone(), settings: settings.clone() };

    // Build the StreamableHttpService for the /mcp endpoint.
    // The factory closure must return a fresh McpHandler per session.
    let mcp_home = home_dir.clone();
    let mcp_engine = index_engine.clone();
    let mcp_dbs = repo_dbs.clone();
    let mcp_settings = settings.clone();

    let mcp_config = {
        // DNS-rebinding protection: if bind is non-loopback, add it to allowed_hosts.
        let is_loopback = matches!(bind_host, "127.0.0.1" | "localhost" | "::1");
        if is_loopback {
            StreamableHttpServerConfig::default()
        } else {
            StreamableHttpServerConfig::default().with_allowed_hosts(vec![
                bind_host.to_string(),
                "localhost".to_string(),
                "127.0.0.1".to_string(),
                "::1".to_string(),
            ])
        }
    };

    let session_manager = Arc::new(LocalSessionManager::default());
    let mcp_service = StreamableHttpService::new(
        move || {
            Ok(McpHandler::new(
                mcp_home.clone(),
                mcp_engine.clone(),
                mcp_dbs.clone(),
                mcp_settings.clone(),
            ))
        },
        session_manager,
        mcp_config,
    );

    Router::new()
        .route("/", get(serve_index))
        .route("/api/config", get(get_config))
        .route("/api/config", put(put_config))
        .route("/api/repos/:repo_id/index", post(post_index_repo).delete(delete_repo_index))
        .route("/api/repos/:repo_id/rebuild", post(post_rebuild_repo))
        .route("/api/repos/:repo_id/cancel-index", post(post_cancel_index))
        .route("/api/repos/:repo_id/status", get(get_repo_status))
        .route("/api/repos/:repo_id/index-stats", get(get_index_stats))
        .route("/api/repos/:repo_id/files", get(get_repo_files))
        .route("/api/repos/:repo_id/graph", get(get_repo_graph))
        .route("/api/repos/:repo_id/chunks", get(get_repo_chunks))
        .route("/api/repos/:repo_id/index-events", get(get_index_events))
        .route("/api/index-all", post(post_index_all))
        .route("/api/index-status", get(get_index_status))
        .route("/api/query", post(post_query))
        .route("/api/mcp-tool", post(post_mcp_tool))
        .route("/api/embedding-cache", delete(delete_embedding_cache))
        .route("/api/defender-status", get(get_defender_status))
        .route("/api/defender-exclude", post(post_defender_exclude))
        .merge(Router::new().nest_service("/mcp", mcp_service))
        .with_state(state)
}

// ─── Helpers ──────────────────────────────────────────────────────────────

#[allow(clippy::result_large_err)]
fn decode_repo_id(repo_id: &str) -> Result<String, Response> {
    URL_SAFE_NO_PAD
        .decode(repo_id)
        .ok()
        .and_then(|b| String::from_utf8(b).ok())
        .ok_or_else(|| {
            let body = json!({ "error": "invalid repo_id encoding" });
            (StatusCode::BAD_REQUEST, Json(body)).into_response()
        })
}

/// Acquire the shared SurrealDB handle for `repo`, or `None` if the repo has no
/// index on disk yet. Read-only browse endpoints use this so an unindexed repo
/// renders an empty state instead of erroring (and without materializing a
/// phantom DB directory as a side effect of a read). Delegates to
/// [`store::open_if_indexed`].
async fn acquire_repo_db_if_indexed(
    state: &AppState,
    repo: &str,
) -> Result<Option<Surreal<Db>>, Response> {
    store::open_if_indexed(&state.repo_dbs, &state.home_dir, repo)
        .await
        .map_err(|e| {
            let body = json!({ "error": format!("failed to open index DB: {e}") });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        })
}

/// Map a `store::ops` error to a 500 JSON response.
fn db_error(context: &str, e: anyhow::Error) -> Response {
    let body = json!({ "error": format!("{context}: {e}") });
    (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
}

// ─── Handlers ──────────────────────────────────────────────────────────────

async fn serve_index() -> impl IntoResponse {
    let html = include_str!("assets/index.html");
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        "text/html; charset=utf-8".parse().unwrap(),
    );
    (headers, html)
}

async fn get_config(State(state): State<AppState>) -> Response {
    match tokio::task::spawn_blocking(move || ensure_dir_and_load(&state.home_dir)).await {
        Ok(Ok(settings)) => Json(settings).into_response(),
        Ok(Err(e)) => e.into_response(),
        Err(join_err) => {
            let body = json!({ "error": format!("internal error: {join_err}") });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

async fn put_config(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    // Parse body as generic Value first so we can return a 400 with a clear message.
    let value: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            let body = json!({ "error": format!("invalid JSON body: {e}") });
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    };

    // Validate into Settings.
    let mut settings: Settings = match serde_json::from_value(value) {
        Ok(s) => s,
        Err(e) => {
            let body = json!({ "error": format!("invalid settings: {e}") });
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    };

    // Server always stamps the current version regardless of what the client sent.
    settings.version = CURRENT_VERSION;

    let target = config_path(&state.home_dir);

    // (2) Persist to disk FIRST. Memory is only updated on success (step 3).
    let saved = match tokio::task::spawn_blocking(move || {
        write_settings_atomic(&target, &settings)?;
        Ok::<Settings, ConfigError>(settings)
    })
    .await
    {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return e.into_response(),
        Err(join_err) => {
            let body = json!({ "error": format!("internal error: {join_err}") });
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response();
        }
    };

    // (3) Under a SINGLE write-lock critical section: diff the new repo list
    // against the value we are about to replace, then swap in the new settings.
    // Computing newly-added atomically against the replaced value (instead of a
    // separate earlier read snapshot) closes the PUT/PUT race where two concurrent
    // adds of the same repo could both trigger an initial index. The guard is
    // dropped at the end of this block — it is NOT held across any await below.
    let newly_added: Vec<String> = {
        let mut guard = state.settings.write().await;
        let added: Vec<String> = saved
            .repos
            .iter()
            .filter(|r| !guard.repos.contains(*r))
            .cloned()
            .collect();
        *guard = saved.clone();
        added
    };

    // (4) Register + trigger each newly-added repo that is an existing directory.
    // Runs on the locally-owned `newly_added` set — no settings guard is held here.
    for repo in &newly_added {
        if std::path::Path::new(repo).is_dir() {
            state.index_engine.register_repo(repo).await;
            if let Err(e) = state.index_engine.trigger_index(repo).await {
                tracing::warn!(repo = %repo, error = %e, "put_config: trigger_index failed for new repo");
            }
        }
    }

    // (5) Return the saved settings JSON — same as before.
    Json(saved).into_response()
}

/// POST /api/repos/:repo_id/index — trigger index for one repo.
async fn post_index_repo(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };

    match state.index_engine.trigger_index(&repo).await {
        Ok(()) => (StatusCode::ACCEPTED, Json(json!({ "status": "accepted" }))).into_response(),
        Err(e) => {
            let body = json!({ "error": format!("failed to trigger index: {e}") });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

/// POST /api/repos/:repo_id/rebuild — force a full rebuild for one repo.
async fn post_rebuild_repo(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };

    match state.index_engine.trigger_rebuild(&repo).await {
        Ok(()) => (StatusCode::ACCEPTED, Json(json!({ "status": "accepted" }))).into_response(),
        Err(e) => {
            let body = json!({ "error": format!("failed to trigger rebuild: {e}") });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

/// POST /api/repos/:repo_id/cancel-index — cancel an in-progress index run.
async fn post_cancel_index(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };

    let cancelled = state.index_engine.cancel_index(&repo).await;
    Json(json!({ "cancelled": cancelled })).into_response()
}

/// DELETE /api/repos/:repo_id/index — remove the index DB folder for a repo.
async fn delete_repo_index(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };

    // Cancel any in-progress indexing, wait for it to finish, then drop the
    // cached DB handle. This guarantees no pipeline holds a RocksDB lock on
    // the directory when we attempt to remove it.
    state.index_engine.close_repo_db(&repo).await;

    // Clear in-memory state immediately — the index is functionally gone from
    // the user's perspective regardless of whether the directory removal succeeds.
    state.index_engine.clear_repo_index(&repo).await;

    // Remove the on-disk directory, serialized against `open_db` via the per-repo
    // open gate (see store::remove_index_dir). Holding that gate across the
    // retry loop is what prevents a concurrent/immediate re-index from racing the
    // cleanup: the re-index's get_or_open blocks on the gate until the directory
    // is fully gone, then opens a fresh DB on a clean path. Without it, the old
    // detached cleaner could delete files out from under a freshly opened RocksDB
    // (or collide with the still-draining async LOCK release), producing the
    // repeating `open surrealdb` errors seen on re-index after "Remove Indexes".
    let removed = store::remove_index_dir(&state.home_dir, &repo).await;

    if removed {
        Json(json!({ "status": "ok" })).into_response()
    } else {
        // The directory is logically removed (handle closed, status cleared,
        // vector shard evicted) but the OS hasn't released the files yet. Report
        // it so the UI can surface that a retry may be needed; the gate is no
        // longer held, so a re-index would race — the user should retry removal.
        Json(json!({
            "status": "pending",
            "message": "index directory could not be fully removed yet; retry shortly"
        }))
        .into_response()
    }
}

/// POST /api/index-all — trigger index for all repos.
async fn post_index_all(State(state): State<AppState>) -> Response {
    // Read the current repo list from the shared live handle for single-source-of-truth
    // consistency. Guard is dropped immediately after the clone — not held across awaits.
    let repos = state.settings.read().await.repos.clone();

    match state.index_engine.trigger_index_all(&repos).await {
        Ok(()) => (StatusCode::ACCEPTED, Json(json!({ "status": "accepted" }))).into_response(),
        Err(e) => {
            let body = json!({ "error": format!("failed to trigger index-all: {e}") });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

/// GET /api/repos/:repo_id/status
async fn get_repo_status(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };

    match state.index_engine.repo_status(&repo).await {
        Some(status) => Json(status).into_response(),
        None => {
            let body = json!({ "error": "repo not found" });
            (StatusCode::NOT_FOUND, Json(body)).into_response()
        }
    }
}

/// GET /api/repos/:repo_id/index-stats — summary counts + config for the explorer.
async fn get_index_stats(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let db = match acquire_repo_db_if_indexed(&state, &repo).await {
        Ok(Some(d)) => d,
        // Never indexed → zeroed stats with a "not_indexed" state instead of an
        // error. The explorer renders this cleanly (counts show 0 / Status reads
        // from index status), and no phantom DB is created by a read.
        Ok(None) => {
            let embedding_model = state.settings.read().await.embedding.model.clone();
            let db_dir = store::db_path(&state.home_dir, &repo);
            return Json(json!({
                "repo": repo,
                "files": 0,
                "chunks": 0,
                "symbols": 0,
                "embedding_model": embedding_model,
                "embedding_dim": null,
                "db_path": db_dir.to_string_lossy(),
                "state": "not_indexed",
                "last_indexed_at": null,
            }))
            .into_response();
        }
        Err(r) => return r,
    };

    let files = match store::ops::count_indexed_files(&db, &repo).await {
        Ok(v) => v,
        Err(e) => return db_error("count files", e),
    };
    let chunks = match store::ops::count_chunks(&db).await {
        Ok(v) => v,
        Err(e) => return db_error("count chunks", e),
    };
    let symbols = match store::ops::count_symbols(&db).await {
        Ok(v) => v,
        Err(e) => return db_error("count symbols", e),
    };
    let embedding_dim = match store::ops::sample_embedding_dim(&db).await {
        Ok(v) => v,
        Err(e) => return db_error("sample embedding dim", e),
    };

    let status = state.index_engine.repo_status(&repo).await;
    let (state_str, last_indexed_at) = match &status {
        Some(s) => (
            serde_json::to_value(&s.state).ok().and_then(|v| v.as_str().map(str::to_string)),
            s.last_indexed_at,
        ),
        None => (None, None),
    };

    let db_dir = store::db_path(&state.home_dir, &repo);

    // Take an owned snapshot of only what's needed — guard dropped before the Json call.
    let embedding_model = state.settings.read().await.embedding.model.clone();

    Json(json!({
        "repo": repo,
        "files": files,
        "chunks": chunks,
        "symbols": symbols,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "db_path": db_dir.to_string_lossy(),
        "state": state_str,
        "last_indexed_at": last_indexed_at,
    }))
    .into_response()
}

/// GET /api/repos/:repo_id/files — bounded file browser rows.
async fn get_repo_files(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let db = match acquire_repo_db_if_indexed(&state, &repo).await {
        Ok(Some(d)) => d,
        // Never indexed → empty file list, not an error. The UI shows its
        // "no files / not indexed" empty state.
        Ok(None) => {
            return Json(json!({ "files": [], "truncated": false })).into_response();
        }
        Err(r) => return r,
    };

    const FILE_LIMIT: usize = 2000;
    match store::ops::files_page(&db, &repo, FILE_LIMIT).await {
        Ok(rows) => {
            let truncated = rows.len() >= FILE_LIMIT;
            Json(json!({ "files": rows, "truncated": truncated })).into_response()
        }
        Err(e) => db_error("list files", e),
    }
}

/// GET /api/repos/:repo_id/graph — bounded call-graph node-link payload.
async fn get_repo_graph(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let db = match acquire_repo_db_if_indexed(&state, &repo).await {
        Ok(Some(d)) => d,
        // Never indexed → empty graph, not an error.
        Ok(None) => {
            return Json(json!({ "nodes": [], "edges": [], "truncated": false })).into_response();
        }
        Err(r) => return r,
    };

    const EDGE_LIMIT: usize = 600;
    const NODE_LIMIT: usize = 250;
    match store::ops::call_graph(&db, EDGE_LIMIT, NODE_LIMIT).await {
        Ok(graph) => Json(graph).into_response(),
        Err(e) => db_error("build graph", e),
    }
}

/// GET /api/repos/:repo_id/chunks?file=... — chunk detail for one file.
async fn get_repo_chunks(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
    Query(params): Query<ChunksQuery>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };

    let file = params.file.trim().to_string();
    if file.is_empty() {
        let body = json!({ "error": "missing 'file' query parameter" });
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }
    // Path traversal guard: the requested file must live under the repo root.
    if !path_in_repo(&file, &repo) {
        let body = json!({ "error": "file is not within the requested repo" });
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }

    let db = match acquire_repo_db_if_indexed(&state, &repo).await {
        Ok(Some(d)) => d,
        // Never indexed → no chunks, not an error.
        Ok(None) => {
            return Json(json!({ "file": file, "chunks": [] })).into_response();
        }
        Err(r) => return r,
    };

    const CHUNK_LIMIT: usize = 500;
    match store::ops::chunks_for_file(&db, &file, CHUNK_LIMIT).await {
        Ok(chunks) => Json(json!({ "file": file, "chunks": chunks })).into_response(),
        Err(e) => db_error("list chunks", e),
    }
}

/// GET /api/index-status — array of all repo statuses.
async fn get_index_status(State(state): State<AppState>) -> Response {
    let all = state.index_engine.all_statuses().await;
    let as_vec: Vec<serde_json::Value> = all
        .iter()
        .map(|(repo, status)| {
            let mut v = serde_json::to_value(status).unwrap_or(json!({}));
            v.as_object_mut()
                .unwrap()
                .insert("repo".to_string(), json!(repo));
            v
        })
        .collect();
    Json(as_vec).into_response()
}

// ─── Query request / response ──────────────────────────────────────────────

#[derive(Deserialize)]
struct ChunksQuery {
    file: String,
}

#[derive(Deserialize)]
struct QueryRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    repo: Option<String>,
    #[serde(default = "default_rerank")]
    rerank: bool,
}

fn default_top_k() -> usize {
    30
}

fn default_rerank() -> bool {
    true
}

/// POST /api/query — run the query pipeline and return results.
async fn post_query(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Response {
    // Take ONE owned snapshot of settings at the top of the handler.
    // The guard is dropped as soon as `.clone()` completes — NOT held across any
    // subsequent .await calls (vector_index.read(), query::run_query, etc.).
    let settings = state.settings.read().await.clone();

    // Pre-flight checks.
    if settings.repos.is_empty() {
        let body = json!({ "error": "No repositories configured. Add repos in Settings first." });
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }

    // NOTE: we intentionally do NOT reject on an empty resident vector index here.
    // Under per-repo sharding with lazy warming, a repo that IS indexed on disk can
    // be momentarily cold (not yet warmed into RAM) — its shard reads empty. The
    // query path handles this correctly: it returns partial (possibly empty) results
    // and spawns a background warm, so the next query hits the now-resident shard.
    // A hard "index is empty" rejection here would falsely block queries to
    // populated-but-cold repos. Truly-unindexed setups simply return no results.

    if settings.embedding.api_keys.is_empty() {
        let body = json!({ "error": "No embedding API keys configured." });
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }

    // Build voyage client.
    let voyage_client = match VoyageClient::new(
        settings.embedding.model.clone(),
        settings.embedding.api_keys.clone(),
    ) {
        Ok(c) => c,
        Err(e) => {
            let body = json!({ "error": format!("failed to create embedding client: {e}") });
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response();
        }
    };

    // Build LLM client for reranking (None if no keys configured or rerank disabled).
    let llm_client = if req.rerank {
        LlmClient::new(&settings.llm)
    } else {
        None
    };

    let top_k = req.top_k.max(1);

    // A repo is mandatory: queries are always scoped to one repository. Reject a
    // repo-less query rather than silently searching across every configured repo.
    let repo_filter = match req.repo.as_deref().map(str::trim) {
        Some(r) if !r.is_empty() => r,
        _ => {
            let body = json!({ "error": "A repository is required. Pass `repo` with the workspace path to scope the query." });
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    };

    match query::run_query(
        &req.query,
        top_k,
        Some(repo_filter),
        &voyage_client,
        &state.index_engine,
        &state.repo_dbs,
        settings.llm.rerank_min_prune_lines,
        llm_client.as_ref(),
        std::time::Duration::from_secs(settings.mcp_index_wait_secs),
    )
    .await
    {
        Ok(result) => Json(result).into_response(),
        Err(e) => {
            let body = json!({ "error": format!("query failed: {e}") });
            (StatusCode::BAD_GATEWAY, Json(body)).into_response()
        }
    }
}

// ─── MCP tool REST proxy ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct McpToolRequest {
    information_request: String,
    workspace_full_path: String,
}

/// POST /api/mcp-tool — call the shared MCP tool funnel over HTTP.
///
/// The response JSON contains `{ "result": "<plain text>" }`. The text is
/// byte-identical to what the MCP tool returns for the same inputs, so the
/// "Test" sub-tab in the web UI exercises the exact same code path.
async fn post_mcp_tool(
    State(state): State<AppState>,
    Json(req): Json<McpToolRequest>,
) -> Response {
    // Take an owned snapshot — guard dropped before the .await below.
    let settings = state.settings.read().await.clone();
    let result = run_codebase_retrieval(
        &state.home_dir,
        &state.index_engine,
        &state.repo_dbs,
        &settings,
        &req.information_request,
        &req.workspace_full_path,
    )
    .await;
    Json(json!({ "result": result })).into_response()
}

// ─── Index events SSE stream ─────────────────────────────────────────────

/// GET /api/repos/:repo_id/index-events — SSE stream of indexing progress events.
///
/// Subscribes to the IndexEngine's broadcast channel and filters events for the
/// requested repo. Sends a keepalive comment every 15s to prevent proxy timeouts.
async fn get_index_events(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };

    let rx = state.index_engine.event_bus.subscribe();
    let stream = BroadcastStream::new(rx);
    let repo_filter = repo.clone();

    let event_stream = stream
        .filter_map(move |result| {
            match result {
                Ok(event) => {
                    let matches = match &event {
                        crate::indexing::events::IndexEvent::Started { repo, .. } => *repo == repo_filter,
                        crate::indexing::events::IndexEvent::FileParsed { .. } => true,
                        crate::indexing::events::IndexEvent::FileSkipped { .. } => true,
                        crate::indexing::events::IndexEvent::FileEmbedded { .. } => true,
                        crate::indexing::events::IndexEvent::FileStored { .. } => true,
                        crate::indexing::events::IndexEvent::FileIndexed { .. } => true,
                        crate::indexing::events::IndexEvent::Phase2Start { repo } => *repo == repo_filter,
                        crate::indexing::events::IndexEvent::Phase2Done { repo, .. } => *repo == repo_filter,
                        crate::indexing::events::IndexEvent::Completed { repo, .. } => *repo == repo_filter,
                        crate::indexing::events::IndexEvent::Failed { repo, .. } => *repo == repo_filter,
                        crate::indexing::events::IndexEvent::Cancelled { repo } => *repo == repo_filter,
                    };
                    if matches {
                        let data = serde_json::to_string(&event).unwrap_or_default();
                        Some(Ok::<_, Infallible>(Event::default().data(data)))
                    } else {
                        None
                    }
                }
                Err(_) => None,
            }
        })
        .map(|e| e);

    let keepalive_stream = async_stream::stream! {
        let mut event_stream = Box::pin(event_stream);
        let mut keepalive = tokio::time::interval(Duration::from_secs(15));
        keepalive.tick().await; // skip first immediate tick

        loop {
            tokio::select! {
                Some(event) = event_stream.next() => {
                    yield event;
                }
                _ = keepalive.tick() => {
                    yield Ok(Event::default().comment("keepalive"));
                }
            }
        }
    };

    Sse::new(keepalive_stream).into_response()
}

// ─── Embedding cache purge ────────────────────────────────────────────────

/// DELETE /api/embedding-cache?older_than=all|30d
///
/// Purges embedding cache entries across all model subdirectories.
/// `older_than=all` (default) deletes everything; `older_than=30d` deletes
/// files not accessed in the last 30 days.
async fn delete_embedding_cache(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let older_than = match params.get("older_than").map(|s| s.as_str()) {
        Some("all") | None => None,
        Some("30d") => Some(std::time::Duration::from_secs(30 * 24 * 3600)),
        Some(other) => {
            let body = json!({ "error": format!("invalid older_than value: {other}; use 'all' or '30d'") });
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    };

    let home_dir = state.home_dir.clone();
    let result = tokio::task::spawn_blocking(move || {
        crate::embedding::cache::EmbeddingCache::purge_global(&home_dir, older_than)
    })
    .await;

    match result {
        Ok(pr) => Json(json!({ "deleted": pr.deleted, "errors": pr.errors })).into_response(),
        Err(e) => {
            let body = json!({ "error": format!("purge task failed: {e}") });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

// ─── Windows Defender exclusion management ────────────────────────────────

async fn get_defender_status(State(state): State<AppState>) -> Response {
    let data_dir = state
        .home_dir
        .join(".vibervn")
        .join("context-engine")
        .to_string_lossy()
        .to_string();

    let status =
        tokio::task::spawn_blocking(move || defender::check_status(&data_dir)).await;

    match status {
        Ok(s) => Json(json!(s)).into_response(),
        Err(e) => {
            let body = json!({ "error": format!("defender check failed: {e}") });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

async fn post_defender_exclude(State(state): State<AppState>) -> Response {
    let data_dir = state
        .home_dir
        .join(".vibervn")
        .join("context-engine")
        .to_string_lossy()
        .to_string();

    let result =
        tokio::task::spawn_blocking(move || defender::add_exclusions(&data_dir)).await;

    match result {
        Ok(r) => {
            let code = if r.success {
                StatusCode::OK
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (code, Json(json!(r))).into_response()
        }
        Err(e) => {
            let body = json!({ "error": format!("defender exclude failed: {e}") });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}
