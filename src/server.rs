use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::{Json, Path, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post, put},
    Router,
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::Deserialize;
use serde_json::{Value, json};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tokio::sync::RwLock;

use crate::config::{
    ConfigError, CURRENT_VERSION, Settings, ensure_dir_and_load, write_settings_atomic,
    config_path,
};
use crate::embedding::voyage::VoyageClient;
use crate::indexing::IndexEngine;
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
    /// Current settings (needed for VoyageClient creation at query time).
    pub settings: Settings,
}

// ─── Router ────────────────────────────────────────────────────────────────

pub fn build_router(
    home_dir: PathBuf,
    index_engine: Arc<IndexEngine>,
    repo_dbs: Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    settings: Settings,
) -> Router {
    let state = AppState { home_dir, index_engine, repo_dbs, settings };
    Router::new()
        .route("/", get(serve_index))
        .route("/api/config", get(get_config))
        .route("/api/config", put(put_config))
        .route("/api/repos/:repo_id/index", post(post_index_repo))
        .route("/api/repos/:repo_id/status", get(get_repo_status))
        .route("/api/index-all", post(post_index_all))
        .route("/api/index-status", get(get_index_status))
        .route("/api/query", post(post_query))
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

    match tokio::task::spawn_blocking(move || {
        write_settings_atomic(&target, &settings)?;
        Ok::<Settings, ConfigError>(settings)
    })
    .await
    {
        Ok(Ok(saved)) => Json(saved).into_response(),
        Ok(Err(e)) => e.into_response(),
        Err(join_err) => {
            let body = json!({ "error": format!("internal error: {join_err}") });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
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

/// POST /api/index-all — trigger index for all repos.
async fn post_index_all(State(state): State<AppState>) -> Response {
    // Load current settings to get repo list.
    let settings = match tokio::task::spawn_blocking({
        let hd = state.home_dir.clone();
        move || ensure_dir_and_load(&hd)
    })
    .await
    {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return e.into_response(),
        Err(e) => {
            let body = json!({ "error": format!("internal error: {e}") });
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response();
        }
    };

    match state.index_engine.trigger_index_all(&settings.repos).await {
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
struct QueryRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    repo: Option<String>,
}

fn default_top_k() -> usize {
    30
}

/// POST /api/query — run the query pipeline and return results.
async fn post_query(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Response {
    // Pre-flight checks.
    if state.settings.repos.is_empty() {
        let body = json!({ "error": "No repositories configured. Add repos in Settings first." });
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }

    {
        let vi = state.index_engine.vector_index.read().await;
        if vi.is_empty() {
            let body = json!({ "error": "Index is empty. Trigger indexing first." });
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    }

    if state.settings.embedding.api_keys.is_empty() {
        let body = json!({ "error": "No embedding API keys configured." });
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }

    // Build voyage client.
    let voyage_client = match VoyageClient::new(
        state.settings.embedding.model.clone(),
        state.settings.embedding.api_keys.clone(),
    ) {
        Ok(c) => c,
        Err(e) => {
            let body = json!({ "error": format!("failed to create embedding client: {e}") });
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response();
        }
    };

    let top_k = req.top_k.max(1);
    let repo_filter = req.repo.as_deref();

    match query::run_query(
        &req.query,
        top_k,
        repo_filter,
        &voyage_client,
        &state.index_engine,
        &state.repo_dbs,
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
