fn build_http_routes() -> Router<RouterState> {
    Router::new()
        .route("/", get(crate::assets::serve_index))
        .route("/assets/fonts/:name", get(crate::assets::serve_font))
        .route("/api/config", get(get_config).put(put_config))
        .route("/api/server-info", get(get_server_info))
        .route("/api/repos", get(list_repos))
        .route("/api/embedding-cache", delete(delete_embedding_cache))
        .route("/api/defender-status", get(get_defender_status))
        .route("/api/defender-exclude", post(post_defender_exclude))
        .route("/api/plan/packages", get(plan::packages))
        .route("/api/plan/payment-methods", get(plan::payment_methods))
        .route("/api/plan/checkout", post(plan::checkout))
        .route("/api/plan/orders/:invoice/status", get(plan::order_status))
        .route("/api/plan/usage", get(plan::usage))
        .route("/api/plan/free-trial", get(plan::free_trial))
        .route("/api/plan/free-trial/claim", post(plan::free_trial_claim))
        .route("/api/index-all", post(post_index_all))
        .route("/api/index-status", get(get_index_status))
        .route("/api/query", post(proxy_query_by_body))
        .route("/api/mcp-tool", post(proxy_mcp_tool_by_body))
        .route(
            "/api/mcp-tool/file-retrieval",
            post(proxy_file_retrieval_by_body),
        )
        // Read routes never acquire or spawn a worker.
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
        .route(
            "/api/repos/:repo_id/index",
            delete(repo_delete_index_native).post(proxy_repo_index_post),
        )
        .route("/api/repos/:repo_id", any(proxy_repo_root))
        .route("/api/repos/:repo_id/*rest", any(proxy_repo_subpath))
        .route("/mcp-repo/:repo_id", any(proxy_mcp_repo))
}
