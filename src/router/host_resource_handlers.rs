async fn delete_embedding_cache(
    State(state): State<RouterState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let older_than = match params.get("older_than").map(String::as_str) {
        Some("all") | None => None,
        Some("30d") => Some(std::time::Duration::from_secs(30 * 24 * 3600)),
        Some(value) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("invalid older_than value: {value}; use 'all' or '30d'")})),
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
        Ok(result) => {
            Json(json!({"deleted": result.deleted, "errors": result.errors})).into_response()
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("purge task failed: {error}")})),
        )
            .into_response(),
    }
}

/// GET /api/server-info — the router's real listen port (`null` when unknown).
/// The UI builds MCP endpoint URLs as `http://localhost:<port>/…` from this; the
/// browser's own origin can't be trusted (iframe embed, reverse proxy, remote
/// browser), while MCP clients always reach the engine on the host it runs on.
async fn get_server_info(State(state): State<RouterState>) -> Response {
    Json(json!({ "port": state.listen_port })).into_response()
}

async fn get_defender_status(State(state): State<RouterState>) -> Response {
    let data_dir = state.data_dir.to_string_lossy().to_string();
    match tokio::task::spawn_blocking(move || crate::defender::check_status(&data_dir)).await {
        Ok(status) => Json(json!(status)).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("defender check failed: {error}")})),
        )
            .into_response(),
    }
}

async fn post_defender_exclude(State(state): State<RouterState>) -> Response {
    let data_dir = state.data_dir.to_string_lossy().to_string();
    match tokio::task::spawn_blocking(move || crate::defender::add_exclusions(&data_dir)).await {
        Ok(result) => {
            let status = if result.success {
                StatusCode::OK
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(json!(result))).into_response()
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("defender exclude failed: {error}")})),
        )
            .into_response(),
    }
}
