use std::collections::HashMap;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    Router,
    extract::{Json, Path, Query, Request, State},
    http::{HeaderMap, StatusCode, header},
    response::sse::{Event, Sse},
    response::{IntoResponse, Response},
    routing::{any, delete, get, post, put},
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::Deserialize;
use serde_json::{Value, json};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tokio::sync::RwLock;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};

use crate::mcp_session_store::{BoundedSessionStore, SharedSessionStore};

type RepoMcpService = StreamableHttpService<RepoMcpHandler, LocalSessionManager>;

/// Build the streamable-HTTP config with our bounded session store attached.
///
/// ## The bug this fixes
///
/// rmcp's `LocalSessionManager` keeps live sessions in memory with a
/// `keep_alive` idle timeout (default 5 min). After that idle window the
/// session worker exits and `close_session` drops the entry. *Without a store*,
/// the next request bearing the now-stale `mcp-session-id` finds no session and
/// the client gets `404 "Session not found"`. Clients (Claude Code) hold a
/// session id across long idle gaps, so for an always-on server that runs for
/// weeks this surfaces intermittently — and MCP must never break that way.
///
/// ## Why a store, not a longer timeout
///
/// Bumping `keep_alive` only widens the window; a long-enough idle gap (or the
/// process being restarted as a service) still 404s, and pinning workers alive
/// for hours keeps O(clients) live channels/tasks resident. The correct fix is
/// rmcp's restore path: with a store configured, an idle timeout drops only the
/// *live worker* (keeping resident workers bounded under the short default
/// timeout), the store entry survives, and the next stale request transparently
/// restores the session via `try_restore_from_store` — no client-visible error.
/// The store is LRU-bounded ([`BoundedSessionStore`]) so memory stays bounded
/// no matter how many clients connect over the server's lifetime, and session
/// ids are globally unique so every client/connection is independent.
fn mcp_config_with_store(
    mut base: StreamableHttpServerConfig,
    store: SharedSessionStore,
) -> StreamableHttpServerConfig {
    base.session_store = Some(store);
    base
}

use crate::config::{
    CURRENT_VERSION, ConfigError, Settings, config_path, ensure_dir_and_load, write_settings_atomic,
};
use crate::defender;
use crate::indexing::IndexEngine;
use crate::mcp::{McpHandler, RepoMcpHandler, run_codebase_retrieval};
use crate::path_in_repo;
use crate::store;

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
    /// Resolved home directory. Used ONLY for `settings.json` access (its
    /// location is fixed at `~/.vibervn/context-engine/settings.json` — see
    /// the bootstrap notes on `Settings.data_dir`).
    pub home_dir: PathBuf,
    /// Boot-resolved data directory (CLI > env > `Settings.data_dir` > builtin
    /// default). Used for store + defender paths. Captured once at startup;
    /// **never re-read from `Settings` at runtime** so already-open RocksDB
    /// handles in `repo_dbs` and resident vector shards stay consistent. PUT
    /// /api/config that changes `data_dir` only affects the next launch.
    pub data_dir: PathBuf,
    /// Boot-resolved embedding-cache root (precedence: CLI, env
    /// `CONTEXT_ENGINE_EMBEDDINGS_DIR`, `Settings.embeddings_dir`, then
    /// `<data_dir>/embeddings`). Used for the cache-purge endpoint. Separate
    /// from `data_dir` so the content-addressed cache can be shared across
    /// instances. Boot-frozen, like `data_dir`.
    pub embeddings_dir: PathBuf,
    /// Shared index engine.
    pub index_engine: Arc<IndexEngine>,
    /// Per-repo SurrealDB handles, keyed by repo path.
    pub repo_dbs: Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    /// Shared live settings — the single source of truth.
    /// All mutations go through this handle AND are written to disk first.
    pub settings: Arc<RwLock<crate::config::Settings>>,
    /// Per-repo MCP services, lazily created on first access.
    pub repo_mcp_services: Arc<RwLock<HashMap<String, RepoMcpService>>>,
    /// Shared bounded session store backing every MCP service (the global
    /// `/mcp` endpoint and each lazily-created per-repo service). Session ids
    /// are globally unique, so one store safely serves all clients; it enables
    /// transparent session restore after idle timeout (see `mcp_config_with_store`).
    pub mcp_session_store: SharedSessionStore,
    /// In-memory, LRU-bounded chat conversation store (repo detail chat).
    pub conversations: Arc<crate::chat::ConversationStore>,
}

// ─── Router ────────────────────────────────────────────────────────────────

pub fn build_router(
    home_dir: PathBuf,
    data_dir: PathBuf,
    embeddings_dir: PathBuf,
    index_engine: Arc<IndexEngine>,
    repo_dbs: Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    settings: Arc<RwLock<crate::config::Settings>>,
    bind_host: &str,
) -> Router {
    let state = AppState {
        home_dir: home_dir.clone(),
        data_dir: data_dir.clone(),
        embeddings_dir,
        index_engine: index_engine.clone(),
        repo_dbs: repo_dbs.clone(),
        settings: settings.clone(),
        repo_mcp_services: Arc::new(RwLock::new(HashMap::new())),
        mcp_session_store: Arc::new(BoundedSessionStore::new()),
        conversations: Arc::new(crate::chat::ConversationStore::new()),
    };

    // Build the StreamableHttpService for the /mcp endpoint.
    // The factory closure must return a fresh McpHandler per session.
    let mcp_home = home_dir.clone();
    let mcp_data = data_dir.clone();
    let mcp_engine = index_engine.clone();
    let mcp_dbs = repo_dbs.clone();
    let mcp_settings = settings.clone();

    let mcp_config = {
        // DNS-rebinding protection: if bind is non-loopback, add it to allowed_hosts.
        let is_loopback = matches!(bind_host, "127.0.0.1" | "localhost" | "::1");
        let base = if is_loopback {
            StreamableHttpServerConfig::default()
        } else {
            StreamableHttpServerConfig::default().with_allowed_hosts(vec![
                bind_host.to_string(),
                "localhost".to_string(),
                "127.0.0.1".to_string(),
                "::1".to_string(),
            ])
        };
        // Attach the shared session store so idle-dropped sessions self-heal
        // via rmcp's restore path instead of 404-ing "Session not found".
        mcp_config_with_store(base, state.mcp_session_store.clone())
    };

    let session_manager = Arc::new(LocalSessionManager::default());
    let mcp_service = StreamableHttpService::new(
        move || {
            let enabled = mcp_settings
                .try_read()
                .map(|g| g.enabled_mcp_tools.clone())
                .unwrap_or_else(|_| crate::config::Settings::default().enabled_mcp_tools);
            Ok(McpHandler::new(
                mcp_home.clone(),
                mcp_data.clone(),
                mcp_engine.clone(),
                mcp_dbs.clone(),
                mcp_settings.clone(),
                &enabled,
            ))
        },
        session_manager,
        mcp_config,
    );

    Router::new()
        .route("/", get(serve_index))
        .route("/api/config", get(get_config))
        .route("/api/config", put(put_config))
        .route(
            "/api/repos/:repo_id/index",
            post(post_index_repo).delete(delete_repo_index),
        )
        .route("/api/repos/:repo_id/rebuild", post(post_rebuild_repo))
        .route("/api/repos/:repo_id/cancel-index", post(post_cancel_index))
        .route("/api/repos/:repo_id/status", get(get_repo_status))
        .route("/api/repos/:repo_id/index-stats", get(get_index_stats))
        .route("/api/repos/:repo_id/files", get(get_repo_files))
        .route("/api/repos/:repo_id/ignore-file", post(post_ignore_file))
        .route("/api/repos/:repo_id/ignore-files", post(post_ignore_files))
        .route(
            "/api/repos/:repo_id/unignore-file",
            post(post_unignore_file),
        )
        .route("/api/repos/:repo_id/ignored-files", get(get_ignored_files))
        .route("/api/repos/:repo_id/graph", get(get_repo_graph))
        .route("/api/repos/:repo_id/chunks", get(get_repo_chunks))
        .route("/api/repos/:repo_id/index-events", get(get_index_events))
        .route("/api/repos/:repo_id/mcp-setup", post(post_mcp_setup))
        .route("/api/repos/:repo_id/chat", post(post_repo_chat))
        .route(
            "/api/repos/:repo_id/chat/:conversation_id",
            delete(delete_repo_chat),
        )
        .route("/api/index-all", post(post_index_all))
        .route("/api/index-status", get(get_index_status))
        .route("/api/query", post(post_query))
        .route("/api/mcp-tool", post(post_mcp_tool))
        .route("/api/mcp-tool/file-retrieval", post(post_file_retrieval))
        .route("/api/embedding-cache", delete(delete_embedding_cache))
        .route("/api/defender-status", get(get_defender_status))
        .route("/api/defender-exclude", post(post_defender_exclude))
        .route("/api/plan/packages", get(plan_get_packages))
        .route("/api/plan/payment-methods", get(plan_get_payment_methods))
        .route("/api/plan/checkout", post(plan_post_checkout))
        .route(
            "/api/plan/orders/:invoice/status",
            get(plan_get_order_status),
        )
        .route("/api/plan/usage", get(plan_get_usage))
        .route("/api/plan/free-trial", get(plan_get_free_trial))
        .route(
            "/api/plan/free-trial/claim",
            post(plan_post_free_trial_claim),
        )
        .route("/mcp-repo/:repo_id", any(handle_repo_mcp))
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
        .map(|s| crate::store::normalize_repo_path(&s))
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
    let generation = state.settings.read().await.repo_generation(repo);
    store::open_if_indexed(&state.repo_dbs, &state.data_dir, repo, generation)
        .await
        .map_err(|e| {
            let body = json!({ "error": format!("failed to open index DB: {e}") });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        })
}

/// Current on-disk index generation for `repo` from the live settings handle.
/// The read guard is dropped before returning, so it never spans a DB `.await`.
async fn repo_generation(state: &AppState, repo: &str) -> u32 {
    state.settings.read().await.repo_generation(repo)
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
        Ok(Ok(settings)) => {
            Json(serde_json::to_value(&settings).unwrap_or_default()).into_response()
        }
        Ok(Err(e)) => e.into_response(),
        Err(join_err) => {
            let body = json!({ "error": format!("internal error: {join_err}") });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

async fn put_config(State(state): State<AppState>, body: axum::body::Bytes) -> Response {
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

    // `repo_generations` is SERVER-OWNED bookkeeping (bumped only by the delete
    // handler). Discard whatever the client sent and preserve the live in-memory
    // map — otherwise the UI's "Xóa repo" flow (DELETE bumps the counter, then PUT
    // /api/config with the config it loaded *before* the bump) would silently
    // clobber the bump, and the re-added repo would reuse the just-deleted (and
    // possibly still-draining) directory. The live handle already reflects the
    // bump because the delete handler persisted to disk AND memory before
    // responding, and the UI awaits the DELETE before PUTting.
    settings.repo_generations = state.settings.read().await.repo_generations.clone();

    // Validate voyage_base_url if provided.
    if let Some(ref url) = settings.embedding.voyage_base_url {
        let trimmed = url.trim();
        if !trimmed.is_empty() && reqwest::Url::parse(trimmed).is_err() {
            let body = json!({ "error": "embedding.voyage_base_url is not a valid URL" });
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    }

    // Validate embedding.dimensions if provided. Negative values are already
    // rejected by serde (the field is u32); the only invalid in-range value is
    // 0. Reject it with a 400 BEFORE the disk write so settings stay unmutated
    // on rejection (the API rejects values the model can't honor at request time).
    if settings.embedding.dimensions == Some(0) {
        let body = json!({ "error": "embedding.dimensions must be a positive integer" });
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }

    // Normalize repo paths to OS-native separators so D:/foo and D:\foo unify.
    settings.repos = settings
        .repos
        .iter()
        .map(|r| crate::store::normalize_repo_path(r))
        .collect();

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

    // (4a) If the request changed `data_dir` to a value different from the
    // boot-resolved path the running process is using, log a warning. The
    // running process is INTENTIONALLY pinned to its boot path: open RocksDB
    // handles in `repo_dbs` and resident vector shards are bound to it, so
    // re-pointing mid-run would split-brain reads against writes (writes land
    // in OLD DBs while a fresh `get_or_open` would land in NEW DBs). The new
    // value is persisted for the next launch's boot resolution.
    let configured = saved
        .data_dir
        .clone()
        .unwrap_or_else(|| crate::config::default_data_dir(&state.home_dir));
    if configured != state.data_dir {
        tracing::warn!(
            requested = %configured.display(),
            active = %state.data_dir.display(),
            "data_dir change persisted to settings.json; takes effect on next launch \
             (current process continues using the boot-resolved path)"
        );
    }

    // (4b) Same boot-frozen treatment for embeddings_dir. The default is
    // anchored to home (`~/.vibervn/context-engine/embeddings`), matching how
    // boot resolution computes it — NOT derived from the configured data_dir.
    // A mismatch is lower-risk than data_dir (a cache-root switch only causes
    // cache misses, not split-brain), but the running process still keeps its
    // boot path, so we log it for parity and operator clarity.
    let configured_emb = saved
        .embeddings_dir
        .clone()
        .unwrap_or_else(|| crate::config::default_embeddings_dir(&state.home_dir));
    if configured_emb != state.embeddings_dir {
        tracing::warn!(
            requested = %configured_emb.display(),
            active = %state.embeddings_dir.display(),
            "embeddings_dir change persisted to settings.json; takes effect on next launch \
             (current process continues using the boot-resolved cache root)"
        );
    }

    // (5) Return the saved settings JSON — same as before.
    Json(saved).into_response()
}

/// POST /api/repos/:repo_id/index — trigger index for one repo.
async fn post_index_repo(State(state): State<AppState>, Path(repo_id): Path<String>) -> Response {
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
async fn post_rebuild_repo(State(state): State<AppState>, Path(repo_id): Path<String>) -> Response {
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
async fn post_cancel_index(State(state): State<AppState>, Path(repo_id): Path<String>) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };

    let cancelled = state.index_engine.cancel_index(&repo).await;
    Json(json!({ "cancelled": cancelled })).into_response()
}

/// DELETE /api/repos/:repo_id/index — remove the index DB folder for a repo.
///
/// `?remove_repo=true` additionally drops the repo from `settings.repos` in the
/// SAME durable write that bumps the generation, so "Remove Repo" is committed
/// server-side and survives a reload even if the client never sends a follow-up
/// PUT /api/config (or reloads mid-teardown — the on-disk lock-drain below can
/// take many seconds on Windows). Without the flag the repo stays configured and
/// only its index is torn down ("Remove Index").
async fn delete_repo_index(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };

    let remove_repo = params
        .get("remove_repo")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    // The entire remove core (close handle, clear in-memory state, atomic
    // generation-bump persist disk-then-memory, ungated old-dir removal) lives in
    // the shared `engine_ops::remove_index` so the CLI runs the EXACT same logic.
    // `also_drop_repo = remove_repo` folds the "Remove Repo" repos.retain(...) into
    // the SAME durable write as the bump (see the fn doc for the atomicity
    // rationale). The two response shapes below are unchanged.
    //
    // NOTE: even for a full "Remove Repo" we deliberately do NOT drop the
    // in-memory status entry — `clear_repo_index` inside the shared fn reset it in
    // place. The detached watcher has no stop handle, so the status entry is the
    // only thing that keeps a future `register_repo` (re-add) from spawning a
    // second watcher (unbounded growth on repeated remove/re-add). The leftover
    // idle entry is harmless: GET /api/config (disk) no longer lists the repo, so
    // the UI renders it gone; the poll may re-sync once per tick, the same bounded
    // path already used to surface MCP-auto-registered repos.
    match crate::engine_ops::remove_index(
        &state.home_dir,
        &state.data_dir,
        &state.index_engine,
        &state.settings,
        &repo,
        remove_repo,
    )
    .await
    {
        Ok(crate::engine_ops::RemoveOutcome::Removed) => {
            Json(json!({ "status": "ok" })).into_response()
        }
        Ok(crate::engine_ops::RemoveOutcome::Pending) => {
            // The directory wasn't fully removed yet (OS still holds the files), but the
            // generation bump already redirected future indexing to a fresh path — so the
            // repo is fully usable now and the orphan will be swept on next boot. Report
            // "pending" for transparency (the UI can note the leftover), not as a blocker.
            Json(json!({
                "status": "pending",
                "message": "old index directory not fully removed yet; it will be reclaimed on next restart"
            }))
            .into_response()
        }
        Err(e) => {
            // Persisting the bump failed (or an internal join error). Without a durable
            // bump a re-index could reuse the old generation path — and the old index is
            // intact (we abort before removal). Surface the error so the user can retry
            // rather than silently degrade.
            let body = json!({ "error": format!("failed to persist index generation: {e}") });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
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
async fn get_repo_status(State(state): State<AppState>, Path(repo_id): Path<String>) -> Response {
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
async fn get_index_stats(State(state): State<AppState>, Path(repo_id): Path<String>) -> Response {
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
            let db_dir =
                store::db_path(&state.data_dir, &repo, repo_generation(&state, &repo).await);
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

    // The four summary counts are a pure function of the index content, so they
    // are computed once at the end of each successful index run and cached in
    // `index_meta` (key `stats_cache`). Fast path: serve the cached counts — no
    // three full-table `count() GROUP ALL` scans (measured p50 ≈ 9.7s at kernel
    // scale). Cold miss (DB indexed before this key existed, or a first index not
    // yet finished): compute live ONCE, then persist the cache BEST-EFFORT — so
    // the slow path happens at most once per repo after upgrade, then warm
    // forever. The persist is best-effort on purpose: the old direct-count serve
    // path never wrote, so it had no write-failure mode; a transient cache-write
    // hiccup must NOT 500 a request whose counts are already correct (mirrors how
    // /graph's cold miss recomputes). A genuine COUNT failure still errors, as the
    // old path did.
    let stats = match store::ops::get_cached_stats(&db).await {
        Ok(Some(s)) => s,
        _ => match store::ops::compute_stats(&db, &repo).await {
            Ok(s) => {
                if let Err(e) = store::ops::persist_stats(&db, &s).await {
                    tracing::warn!(
                        repo = %repo,
                        error = %format!("{e:#}"),
                        "failed to persist stats_cache on /index-stats cold miss; serving computed counts anyway"
                    );
                }
                s
            }
            Err(e) => return db_error("compute index stats", e),
        },
    };

    let status = state.index_engine.repo_status(&repo).await;
    let (state_str, last_indexed_at) = match &status {
        Some(s) => (
            serde_json::to_value(&s.state)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string)),
            s.last_indexed_at,
        ),
        None => (None, None),
    };

    let db_dir = store::db_path(&state.data_dir, &repo, repo_generation(&state, &repo).await);

    // Take an owned snapshot of only what's needed — guard dropped before the Json call.
    let embedding_model = state.settings.read().await.embedding.model.clone();

    Json(json!({
        "repo": repo,
        "files": stats.files,
        "chunks": stats.chunks,
        "symbols": stats.symbols,
        "embedding_model": embedding_model,
        "embedding_dim": stats.embedding_dim,
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
    Query(params): Query<HashMap<String, String>>,
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
    let filter = params.get("filter").map(|s| s.as_str());
    match store::ops::files_page(&db, &repo, FILE_LIMIT, filter).await {
        Ok(rows) => {
            let truncated = rows.len() >= FILE_LIMIT;
            Json(json!({ "files": rows, "truncated": truncated })).into_response()
        }
        Err(e) => db_error("list files", e),
    }
}

/// POST /api/repos/:repo_id/ignore-file — ignore a file (delete from index + add to ignore list).
async fn post_ignore_file(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let file_path = match body.get("path").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "missing 'path' field" })),
            )
                .into_response();
        }
    };

    // Derive relative path via strip_prefix, then normalize to forward slashes.
    let relative = {
        let abs = std::path::Path::new(&file_path);
        let root = std::path::Path::new(&repo);
        match abs.strip_prefix(root) {
            Ok(rel) => rel.to_str().unwrap_or("").replace('\\', "/"),
            Err(_) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": "path is not inside repo" })),
                )
                    .into_response();
            }
        }
    };
    if relative.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "empty relative path" })),
        )
            .into_response();
    }

    // Acquire per-repo lock (same lock the pipeline consumer uses).
    let lock = state.index_engine.get_repo_lock_public(&repo).await;
    let _guard = lock.lock().await;

    // Open DB handle.
    let generation = repo_generation(&state, &repo).await;
    let db = match store::get_or_open(&state.repo_dbs, &state.data_dir, &repo, generation).await {
        Ok(d) => d,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("db: {e}") })),
            )
                .into_response();
        }
    };

    // 1. Delete file data from DB (includes file_meta, chunks, symbols, edges, raw_edge).
    if let Err(e) = store::ops::delete_files_data_bulk(&db, std::slice::from_ref(&file_path)).await
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("delete: {e}") })),
        )
            .into_response();
    }

    // 2. Evict vectors from the in-memory shard.
    {
        let mut vi = state.index_engine.vector_index.write().await;
        vi.apply_incremental(&repo, &[file_path], &[], &[]);
    }
    // The shard changed → invalidate the persisted file so the next warm rebuilds it.
    {
        let root = crate::vector::shard_file::repo_shard_root(&state.index_engine.data_dir, &repo);
        let _ = std::fs::remove_file(root.join("CURRENT"));
    }

    // 3. Append relative path to per-repo ignored_paths.
    let mut ignored = store::ops::get_ignored_paths(&db).await.unwrap_or_default();
    if !ignored.contains(&relative) {
        ignored.push(relative.clone());
        if let Err(e) = store::ops::set_ignored_paths(&db, &ignored).await {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("set ignored: {e}") })),
            )
                .into_response();
        }
    }

    Json(json!({ "status": "ok", "ignored": relative })).into_response()
}

/// POST /api/repos/:repo_id/ignore-files — batch-ignore every file whose path
/// matches a name filter (the Index Explorer search box).
///
/// Processes the match set in keyset PAGES of `BATCH` (never materializes the
/// whole set), so peak RAM and the size of every `IN $paths` array stay O(BATCH)
/// regardless of how many files match — a broad filter at kernel scale can match
/// tens of thousands. Returns the count ignored.
///
/// Crash-safety / consistency invariant — per page, in this exact order:
/// first persist the page's relative paths into `ignored_paths` (the DURABLE
/// marker), then delete the page's DB data, then evict the page's vectors from
/// the in-memory shard.
///
/// If the process dies after the marker write but before the delete/evict, the
/// file is already in the ignore list, so the walker skips it on the next index
/// (O(1) HashSet lookup in walker.rs `allows`/`walk_repo_with`) — the stale
/// chunks left behind are never re-indexed and are dropped on the next full
/// rebuild. The marker therefore always dominates the delete (ignored ⊇ deleted),
/// never the reverse, so there is no silent-corruption window where a file_meta
/// row survives without its chunks yet still looks freshly indexed. If a mid-loop
/// page fails, every page already processed is in the consistent ignored+deleted
/// state and the unhandled remainder is untouched (still indexed, not ignored);
/// we surface the error with the count committed so far.
/// the count committed so far.
async fn post_ignore_files(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    /// Page size for the keyset loop. Bounds peak RAM and every `IN $paths`
    /// array to this many paths, independent of total match count.
    const BATCH: usize = 1000;

    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let filter = match body.get("filter").and_then(|v| v.as_str()) {
        Some(f) if !f.trim().is_empty() => f.trim().to_string(),
        // Refuse an empty filter: it would match (and ignore) the whole repo.
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "missing or empty 'filter' field" })),
            )
                .into_response();
        }
    };

    // Acquire per-repo lock (same lock the pipeline consumer uses).
    let lock = state.index_engine.get_repo_lock_public(&repo).await;
    let _guard = lock.lock().await;

    let generation = repo_generation(&state, &repo).await;
    let db = match store::get_or_open(&state.repo_dbs, &state.data_dir, &repo, generation).await {
        Ok(d) => d,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("db: {e}") })),
            )
                .into_response();
        }
    };

    let root = std::path::Path::new(&repo);

    // Load the existing ignore list once and dedup with a HashSet — `Vec::contains`
    // in a loop over the match set is O(N²) (the project forbids it). We mutate the
    // set in memory and re-persist after each page so the durable marker advances
    // with the deletes.
    let mut ignored_vec = store::ops::get_ignored_paths(&db).await.unwrap_or_default();
    let mut ignored_set: std::collections::HashSet<String> = ignored_vec.iter().cloned().collect();

    let mut cursor = String::new();
    let mut total: usize = 0;

    loop {
        let abs_paths =
            match store::ops::paths_matching_filter_page(&db, &repo, &filter, &cursor, BATCH).await
            {
                Ok(p) => p,
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": format!("match: {e}"), "count": total })),
                    )
                        .into_response();
                }
            };
        if abs_paths.is_empty() {
            break;
        }
        // Advance the keyset cursor to the last path of this page (paths are
        // ORDER BY path; the raw string compares identically to the SQL `>`).
        cursor = abs_paths[abs_paths.len() - 1].clone();
        let page_len = abs_paths.len();

        // Derive forward-slash relative paths for the ignore list (mirrors post_ignore_file).
        let relatives: Vec<String> = abs_paths
            .iter()
            .filter_map(|abs| {
                std::path::Path::new(abs)
                    .strip_prefix(root)
                    .ok()
                    .and_then(|rel| rel.to_str())
                    .map(|s| s.replace('\\', "/"))
                    .filter(|s| !s.is_empty())
            })
            .collect();

        // 1. DURABLE MARKER FIRST — merge this page into ignored_paths and persist,
        //    BEFORE deleting any data (see crash-safety invariant above).
        let mut added = false;
        for rel in relatives {
            if ignored_set.insert(rel.clone()) {
                ignored_vec.push(rel);
                added = true;
            }
        }
        if added && let Err(e) = store::ops::set_ignored_paths(&db, &ignored_vec).await {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("set ignored: {e}"), "count": total })),
            )
                .into_response();
        }

        // 2. Delete this page's data from the DB.
        if let Err(e) = store::ops::delete_files_data_bulk(&db, &abs_paths).await {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("delete: {e}"), "count": total })),
            )
                .into_response();
        }

        // 3. Evict this page's vectors from the in-memory shard.
        {
            let mut vi = state.index_engine.vector_index.write().await;
            vi.apply_incremental(&repo, &abs_paths, &[], &[]);
        }

        total += page_len;
        // A short page means we've reached the end of the match set.
        if page_len < BATCH {
            break;
        }
    }

    if total == 0 {
        return Json(json!({ "status": "ok", "count": 0 })).into_response();
    }

    // The shard changed → invalidate the persisted file so the next warm rebuilds it.
    {
        let shard_root =
            crate::vector::shard_file::repo_shard_root(&state.index_engine.data_dir, &repo);
        let _ = std::fs::remove_file(shard_root.join("CURRENT"));
    }

    Json(json!({ "status": "ok", "count": total })).into_response()
}

/// POST /api/repos/:repo_id/unignore-file — restore a file (remove from ignore list).
async fn post_unignore_file(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let relative = match body.get("path").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "missing 'path' field" })),
            )
                .into_response();
        }
    };

    // Acquire per-repo lock (same as ignore handler and pipeline consumer).
    let lock = state.index_engine.get_repo_lock_public(&repo).await;
    let _guard = lock.lock().await;

    let generation = repo_generation(&state, &repo).await;
    let db = match store::get_or_open(&state.repo_dbs, &state.data_dir, &repo, generation).await {
        Ok(d) => d,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("db: {e}") })),
            )
                .into_response();
        }
    };

    let mut ignored = store::ops::get_ignored_paths(&db).await.unwrap_or_default();
    ignored.retain(|p| p != &relative);
    if let Err(e) = store::ops::set_ignored_paths(&db, &ignored).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("set ignored: {e}") })),
        )
            .into_response();
    }

    Json(json!({ "status": "ok" })).into_response()
}

/// GET /api/repos/:repo_id/ignored-files — list per-repo ignored paths.
async fn get_ignored_files(State(state): State<AppState>, Path(repo_id): Path<String>) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let generation = repo_generation(&state, &repo).await;
    let db = match store::get_or_open(&state.repo_dbs, &state.data_dir, &repo, generation).await {
        Ok(d) => d,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("db: {e}") })),
            )
                .into_response();
        }
    };

    let ignored = store::ops::get_ignored_paths(&db).await.unwrap_or_default();
    Json(json!({ "paths": ignored })).into_response()
}

/// GET /api/repos/:repo_id/graph — bounded call-graph node-link payload.
async fn get_repo_graph(State(state): State<AppState>, Path(repo_id): Path<String>) -> Response {
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

    // The bounded call-graph payload is a pure function of the `calls` table,
    // so it is computed once at the end of each successful index run and cached
    // in `index_meta` (key `graph_cache`). Fast path: serve the cached payload
    // directly — no `call_graph` recompute (two full-table GROUP BY aggregations,
    // ~80s at kernel scale). Cold miss (DB indexed before this key existed, or a
    // first index not yet finished): compute live ONCE, store it, and return —
    // so the slow path happens at most once per repo after upgrade, then warm
    // forever. Note: the canonical node/edge limits live in `store::ops`
    // (GRAPH_NODE_LIMIT / GRAPH_EDGE_LIMIT) and are shared with the index-time
    // cache refresh so the cached payload matches this endpoint's contract.
    match store::ops::get_cached_graph(&db).await {
        Ok(Some(graph)) => Json(graph).into_response(),
        _ => match store::ops::compute_and_cache_graph(&db).await {
            Ok(graph) => Json(graph).into_response(),
            Err(e) => db_error("build graph", e),
        },
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
async fn post_query(State(state): State<AppState>, Json(req): Json<QueryRequest>) -> Response {
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

    // A repo is mandatory: queries are always scoped to one repository. Reject a
    // repo-less query rather than silently searching across every configured repo.
    let repo_filter = match req.repo.as_deref().map(str::trim) {
        Some(r) if !r.is_empty() => r,
        _ => {
            let body = json!({ "error": "A repository is required. Pass `repo` with the workspace path to scope the query." });
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    };

    // Delegate to the shared query op (VoyageClient build, optional LlmClient,
    // repo normalization, query::run_query with all settings-derived args) so the
    // CLI and server produce byte-identical retrieval. The `settings` snapshot was
    // cloned above — no settings guard is held across the await below.
    match crate::engine_ops::run_query_op(
        &settings,
        &state.index_engine,
        &state.repo_dbs,
        repo_filter,
        &req.query,
        req.top_k,
        req.rerank,
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
async fn post_mcp_tool(State(state): State<AppState>, Json(req): Json<McpToolRequest>) -> Response {
    // Take an owned snapshot — guard dropped before the .await below.
    let settings = state.settings.read().await.clone();
    let result = run_codebase_retrieval(
        &state.home_dir,
        &state.data_dir,
        &state.index_engine,
        &state.repo_dbs,
        &settings,
        &req.information_request,
        &req.workspace_full_path,
    )
    .await;
    Json(json!({ "result": result })).into_response()
}

// ─── File-retrieval REST proxy ────────────────────────────────────────────

#[derive(Deserialize)]
struct FileRetrievalRequest {
    workspace_full_path: String,
    file_path: String,
    information_request: String,
    top_k: Option<usize>,
}

/// POST /api/mcp-tool/file-retrieval — call the file-retrieval funnel over HTTP.
async fn post_file_retrieval(
    State(state): State<AppState>,
    Json(req): Json<FileRetrievalRequest>,
) -> Response {
    let settings = state.settings.read().await.clone();
    let result = crate::mcp::run_file_retrieval(
        &state.data_dir,
        &state.repo_dbs,
        &settings,
        &req.workspace_full_path,
        &req.file_path,
        &req.information_request,
        req.top_k.unwrap_or(5),
    )
    .await;
    Json(json!({ "result": result })).into_response()
}

// ─── MCP auto-setup ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct McpSetupRequest {
    /// "claude" | "codex" | "opencode".
    target: String,
    /// Browser-built MCP URL: `<origin>/mcp-repo/<sanitized>`. The origin is how
    /// the live listen port reaches us (it lives only in `main.rs`, not state).
    endpoint_url: String,
}

/// POST /api/repos/:repo_id/mcp-setup — write a tool's MCP config + prompt
/// files directly into the repo on disk.
///
/// Validates: (1) repo_id decodes and is present in `settings.repos`; (2) the
/// target is a known tool; (3) `endpoint_url` is the repo's own MCP endpoint
/// (path `/mcp-repo/<sanitize_repo_name(repo)>`) so a caller can't write an
/// arbitrary URL into the user's config.
async fn post_mcp_setup(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
    Json(req): Json<McpSetupRequest>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };

    let target = match crate::mcp_setup::Target::parse(&req.target) {
        Some(t) => t,
        None => {
            let body = json!({ "error": format!("unknown target: {}", req.target) });
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    };

    // Repo must be a configured repo — never write into an arbitrary path.
    {
        let settings = state.settings.read().await;
        if !settings.repos.contains(&repo) {
            let body = json!({ "error": "repo not found" });
            return (StatusCode::NOT_FOUND, Json(body)).into_response();
        }
    }

    // The URL must be THIS repo's own MCP endpoint. We accept any scheme/host
    // (reverse proxies are valid) but the path must end with the repo's
    // sanitized-name route, so a caller can't smuggle a foreign URL into the
    // user's config file.
    let expected_suffix = format!("/mcp-repo/{}", store::sanitize_repo_name(&repo));
    let url_path = req
        .endpoint_url
        .split(['?', '#'])
        .next()
        .unwrap_or(&req.endpoint_url)
        .trim_end_matches('/');
    if !url_path.ends_with(&expected_suffix) {
        let body = json!({
            "error": "endpoint_url does not match this repo's MCP endpoint",
        });
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }

    // File IO is blocking — run off the async runtime.
    let repo_root = PathBuf::from(&repo);
    let endpoint_url = req.endpoint_url.clone();
    let actions = match tokio::task::spawn_blocking(move || {
        crate::mcp_setup::run_setup(&repo_root, target, &endpoint_url)
    })
    .await
    {
        Ok(a) => a,
        Err(e) => {
            let body = json!({ "error": format!("setup task failed: {e}") });
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response();
        }
    };

    let any_error = actions
        .iter()
        .any(|a| a.status == crate::mcp_setup::FileStatus::Error);
    let body = json!({ "actions": actions.iter().map(|a| a.to_json()).collect::<Vec<_>>() });
    // 207 (Multi-Status) when some files errored but others succeeded — the UI
    // renders per-file status either way.
    let code = if any_error {
        StatusCode::MULTI_STATUS
    } else {
        StatusCode::OK
    };
    (code, Json(body)).into_response()
}

// ─── Per-repo MCP endpoint ──────────────────────────────────────────────
/// ANY /mcp-repo/:repo_name — per-repo MCP endpoint (no workspace_full_path needed).
/// `repo_name` is the sanitized repo name (e.g. `D__projects_Python_foo`).
async fn handle_repo_mcp(
    State(state): State<AppState>,
    Path(repo_name): Path<String>,
    req: Request,
) -> Response {
    // Resolve sanitized name back to the full repo path.
    let settings = state.settings.read().await;
    let repo = settings
        .repos
        .iter()
        .find(|r| store::sanitize_repo_name(r) == repo_name)
        .cloned();
    drop(settings);

    let repo = match repo {
        Some(r) => r,
        None => {
            let body = json!({ "error": format!("unknown repo: {}", repo_name) });
            return (StatusCode::NOT_FOUND, Json(body)).into_response();
        }
    };

    // Get or create the per-repo MCP service.
    let service = {
        let cache = state.repo_mcp_services.read().await;
        cache.get(&repo).cloned()
    };
    let mut service = match service {
        Some(s) => s,
        None => {
            let home = state.home_dir.clone();
            let data = state.data_dir.clone();
            let engine = state.index_engine.clone();
            let dbs = state.repo_dbs.clone();
            let settings = state.settings.clone();
            let repo_clone = repo.clone();
            let new_service = StreamableHttpService::new(
                move || {
                    let enabled = settings
                        .try_read()
                        .map(|g| g.enabled_mcp_tools.clone())
                        .unwrap_or_else(|_| crate::config::Settings::default().enabled_mcp_tools);
                    Ok(RepoMcpHandler::new(
                        home.clone(),
                        data.clone(),
                        repo_clone.clone(),
                        engine.clone(),
                        dbs.clone(),
                        settings.clone(),
                        &enabled,
                    ))
                },
                Arc::new(LocalSessionManager::default()),
                mcp_config_with_store(
                    StreamableHttpServerConfig::default(),
                    state.mcp_session_store.clone(),
                ),
            );
            state
                .repo_mcp_services
                .write()
                .await
                .insert(repo.clone(), new_service.clone());
            new_service
        }
    };

    use tower_service::Service;
    match service.call(req).await {
        Ok(resp) => resp.into_response(),
        Err(e) => match e {},
    }
}

// ─── Index events SSE stream ─────────────────────────────────────────────

/// GET /api/repos/:repo_id/index-events — SSE stream of indexing progress events.
///
/// Subscribes to the IndexEngine's broadcast channel and filters events for the
/// requested repo. Sends a keepalive comment every 15s to prevent proxy timeouts.
async fn get_index_events(State(state): State<AppState>, Path(repo_id): Path<String>) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };

    let rx = state.index_engine.event_bus.subscribe();
    let stream = BroadcastStream::new(rx);
    let repo_filter = repo.clone();

    let event_stream = stream
        .filter_map(move |result| match result {
            Ok(event) => {
                let matches = match &event {
                    crate::indexing::events::IndexEvent::Started { repo, .. } => {
                        *repo == repo_filter
                    }
                    crate::indexing::events::IndexEvent::FileParsed { .. } => true,
                    crate::indexing::events::IndexEvent::FileSkipped { .. } => true,
                    crate::indexing::events::IndexEvent::FileEmbedded { .. } => true,
                    crate::indexing::events::IndexEvent::FileStored { .. } => true,
                    crate::indexing::events::IndexEvent::FileIndexed { .. } => true,
                    crate::indexing::events::IndexEvent::Phase2Start { repo } => {
                        *repo == repo_filter
                    }
                    crate::indexing::events::IndexEvent::Phase2Done { repo, .. } => {
                        *repo == repo_filter
                    }
                    crate::indexing::events::IndexEvent::SymbolIndexStart { repo } => {
                        *repo == repo_filter
                    }
                    crate::indexing::events::IndexEvent::SymbolIndexDone { repo, .. } => {
                        *repo == repo_filter
                    }
                    crate::indexing::events::IndexEvent::Completed { repo, .. } => {
                        *repo == repo_filter
                    }
                    crate::indexing::events::IndexEvent::Failed { repo, .. } => {
                        *repo == repo_filter
                    }
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

// ─── Repo chat (streaming, tool-calling agent) ────────────────────────────

#[derive(Deserialize)]
struct ChatRequest {
    conversation_id: String,
    message: String,
    /// Optional custom-model selection. When both are present and resolve to a
    /// configured `llm.chat_custom_endpoints` entry, the chat turn uses that
    /// OpenAI-compatible endpoint instead of the Settings rerank/index model.
    /// Absent / blank / unrecognized → the default `settings.llm` client.
    /// Both are sent (not an index) so the selection survives endpoint edits.
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

/// Build the chat `LlmClient` for a request. When `base_url` + `model` name a
/// configured custom endpoint that serves that model, return a client pinned to
/// it. The endpoint's own `provider` decides the path: `"google"` goes through
/// the official Gemini API (base URL ignored — matched by `(provider, model)`
/// with an empty request base_url); `"openai"` uses the OpenAI-compatible path
/// with its base URL (matched by `(base_url, model)`). Otherwise fall back to
/// the default `settings.llm` client. The rerank/index path is never affected.
fn build_chat_llm(
    settings: &crate::config::Settings,
    base_url: Option<&str>,
    model: Option<&str>,
) -> Option<crate::llm::LlmClient> {
    if let Some(model) = model {
        let model = model.trim();
        // The request base_url is empty for google endpoints, set for openai.
        let req_base = base_url.map(str::trim).unwrap_or("");
        if !model.is_empty() {
            // Match on the same identity the UI sends: google → base_url is
            // empty, so compare provider+model; openai → compare base_url+model.
            if let Some(ep) = settings.llm.chat_custom_endpoints.iter().find(|e| {
                let is_google = e.provider == "google";
                let model_ok = e.models.iter().any(|m| m.trim() == model);
                if is_google {
                    // google: request carries no base_url; match by provider+model.
                    req_base.is_empty() && model_ok
                } else {
                    // openai: match by exact base_url + model.
                    e.base_url.trim() == req_base && !req_base.is_empty() && model_ok
                }
            }) {
                if ep.api_key.trim().is_empty() {
                    return None; // configured but keyless — caller surfaces the error
                }
                // Reuse the rerank LlmConfig shape, overriding only what the
                // custom chat endpoint defines, per its provider.
                let cfg = if ep.provider == "google" {
                    crate::config::LlmConfig {
                        provider: "google".to_owned(),
                        rerank_model: model.to_owned(),
                        api_keys: vec![ep.api_key.clone()],
                        // google.rs ignores base_url; leave it None.
                        openai_base_url: None,
                        ..settings.llm.clone()
                    }
                } else {
                    crate::config::LlmConfig {
                        provider: "openai".to_owned(),
                        rerank_model: model.to_owned(),
                        api_keys: vec![ep.api_key.clone()],
                        openai_base_url: Some(ep.base_url.clone()),
                        openai_force_tool_use: ep.force_tool_use,
                        ..settings.llm.clone()
                    }
                };
                return crate::llm::LlmClient::new(&cfg);
            }
        }
    }
    crate::llm::LlmClient::new(&settings.llm)
}

/// POST /api/repos/:repo_id/chat — stream an answer for one chat message.
///
/// Returns an SSE stream of `ChatEvent` JSON frames (tool_call, tool_result,
/// token, done, error). The conversation transcript is held in memory keyed by
/// `conversation_id`; closing the dialog (DELETE below) drops it.
async fn post_repo_chat(
    State(state): State<AppState>,
    Path(repo_id): Path<String>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let repo = match decode_repo_id(&repo_id) {
        Ok(r) => r,
        Err(r) => return r,
    };

    let conversation_id = req.conversation_id.trim().to_owned();
    let message = req.message.trim().to_owned();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<crate::chat::ChatEvent>();

    // Validate inputs up front; emit an Error frame (not an HTTP error) so the
    // dialog always renders a message instead of a dead stream.
    if conversation_id.is_empty() || message.is_empty() {
        let _ = tx.send(crate::chat::ChatEvent::Error {
            message: "conversation_id and message are required.".to_owned(),
        });
        return chat_sse_response(rx);
    }

    // Build the LLM client from settings.llm, or from a selected custom chat
    // endpoint when the request names one. Missing/keyless → concise error frame.
    let settings = state.settings.read().await.clone();
    let llm = match build_chat_llm(&settings, req.base_url.as_deref(), req.model.as_deref()) {
        Some(c) => c,
        None => {
            let _ = tx.send(crate::chat::ChatEvent::Error {
                message: "No LLM API key configured. Add a provider key in Settings to use chat."
                    .to_owned(),
            });
            return chat_sse_response(rx);
        }
    };

    let deps = crate::chat::ChatTurnDeps {
        home_dir: state.home_dir.clone(),
        data_dir: state.data_dir.clone(),
        index_engine: state.index_engine.clone(),
        repo_dbs: state.repo_dbs.clone(),
        settings,
        conversations: state.conversations.clone(),
    };

    // Drive the agent on a background task; the SSE stream drains `rx`.
    //
    // Client-disconnect handling: when the dialog is closed (or the connection
    // drops), axum drops the SSE response body → drops the stream → drops `rx`.
    // We race the agent against `tx.closed()` (resolves once every receiver is
    // gone) so a disconnect cancels `run_chat_turn` at its next await point —
    // cutting an in-flight LLM stream or tool call instead of burning tokens to
    // completion for a client that already left.
    tokio::spawn(async move {
        tokio::select! {
            _ = crate::chat::run_chat_turn(&deps, &llm, &repo, &conversation_id, &message, &tx) => {}
            _ = tx.closed() => {
                tracing::debug!(%repo, "chat client disconnected — agent loop cancelled");
            }
        }
    });

    chat_sse_response(rx)
}

/// Wrap a `ChatEvent` receiver into an SSE response. Each event is one
/// `data:` frame of JSON; the stream ends after a `done`/`error` event when the
/// sender is dropped.
fn chat_sse_response(rx: tokio::sync::mpsc::UnboundedReceiver<crate::chat::ChatEvent>) -> Response {
    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(|ev| {
        let data = serde_json::to_string(&ev).unwrap_or_default();
        Ok::<_, Infallible>(Event::default().data(data))
    });
    Sse::new(stream).into_response()
}

/// DELETE /api/repos/:repo_id/chat/:conversation_id — drop a conversation when
/// the dialog is closed. `repo_id` is accepted for route symmetry but the
/// conversation id is globally unique, so only the id is needed.
async fn delete_repo_chat(
    State(state): State<AppState>,
    Path((_repo_id, conversation_id)): Path<(String, String)>,
) -> Response {
    state
        .conversations
        .drop_conversation(&conversation_id)
        .await;
    Json(json!({ "ok": true })).into_response()
}

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

    let embeddings_dir = state.embeddings_dir.clone();
    let result = tokio::task::spawn_blocking(move || {
        crate::embedding::cache::EmbeddingCache::purge_global(&embeddings_dir, older_than)
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
    let data_dir = state.data_dir.to_string_lossy().to_string();

    let status = tokio::task::spawn_blocking(move || defender::check_status(&data_dir)).await;

    match status {
        Ok(s) => Json(json!(s)).into_response(),
        Err(e) => {
            let body = json!({ "error": format!("defender check failed: {e}") });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

async fn post_defender_exclude(State(state): State<AppState>) -> Response {
    let data_dir = state.data_dir.to_string_lossy().to_string();

    let result = tokio::task::spawn_blocking(move || defender::add_exclusions(&data_dir)).await;

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

// ─── Plan proxy (admin gateway) ──────────────────────────────────────────

const PLAN_DEFAULT_ADMIN: &str = "https://context-engine.viber.vn";
const PLAN_PROXY_TIMEOUT: Duration = Duration::from_secs(15);

fn plan_admin_base() -> String {
    std::env::var("CONTEXT_ENGINE_ADMIN_URL")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| PLAN_DEFAULT_ADMIN.to_string())
        .trim_end_matches('/')
        .to_string()
}

fn plan_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(PLAN_PROXY_TIMEOUT)
        .build()
        .unwrap_or_default()
}

// Salt mixed into the machine-id hash. Now lives in `crate::config` since the
// id is computed once at boot and persisted to settings.json. See
// `config::ensure_machine_id` and `config::MACHINE_ID_SALT`.

/// Read the persisted machine_id from the live settings handle. Boot guarantees
/// `Some(...)` after `ensure_machine_id`; `None`/empty would only occur if a
/// caller mutated the field at runtime. Returns the value or a 500 response.
async fn machine_id_from_settings(state: &AppState) -> Result<String, Response> {
    let id = state
        .settings
        .read()
        .await
        .machine_id
        .clone()
        .filter(|s| !s.is_empty());
    id.ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "machine id unavailable: settings not initialized" })),
        )
            .into_response()
    })
}

async fn plan_get_free_trial(State(_): State<AppState>) -> Response {
    let base = plan_admin_base();
    let url = format!("{base}/api/free-trial");

    let res = match plan_http_client().get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("admin gateway unreachable: {e}") })),
            )
                .into_response();
        }
    };

    let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let body = res.bytes().await.unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body))
        .unwrap()
}

async fn plan_post_free_trial_claim(State(state): State<AppState>) -> Response {
    // The machine id is read from persisted settings (populated once at boot
    // by `ensure_machine_id`). The browser never sees it. The previous
    // implementation derived it on the fly via machine_uid::get(); now both
    // free-trial and paid checkout share the same persisted source so a
    // hardware-uid hiccup at runtime can never re-roll the id.
    let machine_id = match machine_id_from_settings(&state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };

    let base = plan_admin_base();
    let url = format!("{base}/api/free-trial/claim");

    let res = match plan_http_client()
        .post(&url)
        .header("content-type", "application/json")
        .json(&json!({ "machine_id": machine_id }))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("admin gateway unreachable: {e}") })),
            )
                .into_response();
        }
    };

    let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let body_bytes = res.bytes().await.unwrap_or_default();

    // Inject base_url on success so the frontend knows where the key points to
    // (mirrors plan_post_checkout). Applies to both 201 Claimed and 200
    // Recovered responses.
    if status.is_success()
        && let Ok(mut obj) = serde_json::from_slice::<Value>(&body_bytes)
    {
        let admin_url = plan_admin_base();
        obj["base_url"] = Value::String(format!("{admin_url}/v1"));
        return (status, Json(obj)).into_response();
    }

    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body_bytes))
        .unwrap()
}

async fn plan_get_packages(State(_): State<AppState>) -> Response {
    let base = plan_admin_base();
    let url = format!("{base}/api/packages");

    let res = match plan_http_client().get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("admin gateway unreachable: {e}") })),
            )
                .into_response();
        }
    };

    let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let body = res.bytes().await.unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body))
        .unwrap()
}

/// Proxy the admin gateway's enabled-payment-methods so the engine UI popup can
/// show only the methods the operator turned on (and nothing when both are off).
async fn plan_get_payment_methods(State(_): State<AppState>) -> Response {
    let base = plan_admin_base();
    let url = format!("{base}/api/payment-methods");

    let res = match plan_http_client().get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("admin gateway unreachable: {e}") })),
            )
                .into_response();
        }
    };

    let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let body = res.bytes().await.unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body))
        .unwrap()
}

async fn plan_post_checkout(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    // Inject the persisted machine_id into the request so the admin gateway
    // can dedup paid purchases per machine (one machine = one user, with
    // accumulated budgets/expiry on repeat purchase). The browser never sees
    // or controls this — it only sends `package_id`.
    let machine_id = match machine_id_from_settings(&state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    let mut body = body;
    if let Value::Object(ref mut obj) = body {
        obj.insert("machine_id".to_string(), Value::String(machine_id));
    } else {
        // Frontend always sends a JSON object; if it doesn't, build one from
        // scratch so the admin gateway never sees a missing machine_id.
        body = json!({ "machine_id": machine_id });
    }

    let base = plan_admin_base();
    let url = format!("{base}/api/checkout");

    let res = match plan_http_client()
        .post(&url)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("admin gateway unreachable: {e}") })),
            )
                .into_response();
        }
    };

    let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let body_bytes = res.bytes().await.unwrap_or_default();

    // Inject base_url into the response so frontend knows where the proxy key points to.
    if status.is_success()
        && let Ok(mut obj) = serde_json::from_slice::<Value>(&body_bytes)
    {
        let admin_url = plan_admin_base();
        obj["base_url"] = Value::String(format!("{admin_url}/v1"));
        return (status, Json(obj)).into_response();
    }

    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body_bytes))
        .unwrap()
}

async fn plan_get_order_status(State(_): State<AppState>, Path(invoice): Path<String>) -> Response {
    let base = plan_admin_base();
    let url = format!("{base}/api/orders/{invoice}/status");

    let res = match plan_http_client().get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("admin gateway unreachable: {e}") })),
            )
                .into_response();
        }
    };

    let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let body_bytes = res.bytes().await.unwrap_or_default();

    // Inject base_url when order is completed so frontend can show it
    if status.is_success()
        && let Ok(mut obj) = serde_json::from_slice::<Value>(&body_bytes)
    {
        if obj.get("status").and_then(|s| s.as_str()) == Some("COMPLETED") {
            let admin_url = plan_admin_base();
            obj["base_url"] = Value::String(format!("{admin_url}/v1"));
        }
        return (status, Json(obj)).into_response();
    }

    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body_bytes))
        .unwrap()
}

async fn plan_get_usage(headers: HeaderMap) -> Response {
    let base = plan_admin_base();
    let url = format!("{base}/api/usage");

    let auth = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let res = match plan_http_client()
        .get(&url)
        .header(header::AUTHORIZATION, &auth)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("admin gateway unreachable: {e}") })),
            )
                .into_response();
        }
    };

    let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let body = res.bytes().await.unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body))
        .unwrap()
}
