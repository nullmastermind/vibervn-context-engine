// UI (`/`) + ADE fonts (`/assets/fonts/:name`) are served by `crate::assets`
// (shared with the standalone server) — see routes.rs.

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
    // Same normalize + bound as the monolith `put_config`, rejected before the
    // disk write so settings.json is untouched on a bad value.
    match crate::config::normalize_docs_extensions(&settings.docs_extensions) {
        Ok(list) => settings.docs_extensions = list,
        Err(msg) => {
            return (StatusCode::BAD_REQUEST, Json(json!({ "error": msg }))).into_response();
        }
    }
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

