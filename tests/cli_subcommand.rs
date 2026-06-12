use std::collections::HashMap;
use std::sync::Arc;

use context_engine_rs::cli::{ReadArgs, SearchArgs, run_read_with_writer, run_search_with_writer};
use context_engine_rs::config::Settings;
use context_engine_rs::indexing::IndexEngine;
use context_engine_rs::mcp::{run_codebase_retrieval, run_file_retrieval};
use context_engine_rs::server::build_router;
use context_engine_rs::store::RepoDbMap;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::RwLock;

async fn start_server() -> (TempDir, TempDir, String, Arc<IndexEngine>, RepoDbMap, Settings) {
    let home = TempDir::new().expect("home tempdir");
    let repo = TempDir::new().expect("repo tempdir");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let settings = Settings::default();
    let settings_handle = Arc::new(RwLock::new(settings.clone()));
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    let data_dir = home.path().join("data");
    let embeddings_dir = home.path().join("embeddings");
    let index_engine = IndexEngine::start(
        data_dir.clone(),
        embeddings_dir.clone(),
        &settings,
        repo_dbs.clone(),
        settings_handle.clone(),
    )
    .await;
    let app = build_router(
        home.path().to_path_buf(),
        data_dir,
        embeddings_dir,
        index_engine.clone(),
        repo_dbs.clone(),
        settings_handle,
        "127.0.0.1",
    );

    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("server error");
    });

    (home, repo, format!("http://{addr}"), index_engine, repo_dbs, settings)
}

#[tokio::test]
async fn search_subcommand_matches_rest_proxy_funnel() {
    let (home, repo, server, index_engine, repo_dbs, settings) = start_server().await;
    let repo_path = repo.path().to_string_lossy().to_string();
    let query = "where is auth handled";
    let mut stdout = Vec::new();

    let exit_code = run_search_with_writer(
        SearchArgs {
            query: query.to_owned(),
            repo: repo_path.clone(),
            server: Some(server),
        },
        &mut stdout,
    )
    .await;

    let expected = run_codebase_retrieval(
        home.path(),
        &home.path().join("data"),
        &index_engine,
        &repo_dbs,
        &settings,
        query,
        &repo_path,
    )
    .await;

    assert_eq!(exit_code, std::process::ExitCode::SUCCESS);
    assert_eq!(String::from_utf8(stdout).expect("utf8 stdout"), expected);
}

#[tokio::test]
async fn read_subcommand_matches_rest_proxy_funnel() {
    let (home, repo, server, _index_engine, repo_dbs, settings) = start_server().await;
    let repo_path = repo.path().to_string_lossy().to_string();
    let file_path = "src/main.rs";
    let query = "tracing setup";
    let top_k = 10;
    let mut stdout = Vec::new();

    let exit_code = run_read_with_writer(
        ReadArgs {
            repo: repo_path.clone(),
            file: file_path.to_owned(),
            query: query.to_owned(),
            top_k: Some(top_k),
            server: Some(server),
        },
        &mut stdout,
    )
    .await;

    let expected = run_file_retrieval(
        &home.path().join("data"),
        &repo_dbs,
        &settings,
        &repo_path,
        file_path,
        query,
        top_k,
    )
    .await;

    assert_eq!(exit_code, std::process::ExitCode::SUCCESS);
    assert_eq!(String::from_utf8(stdout).expect("utf8 stdout"), expected);
}
