#![allow(dead_code)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use reqwest::Client;
use tempfile::TempDir;
use tokio::net::TcpListener;

use context_engine_rs::config::{Settings, config_path, write_settings_atomic};
use context_engine_rs::router::{RouterBootOptions, build_router_app};

pub fn worker_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_context-engine-rs"))
}

pub fn seed_settings(home: &TempDir, repo: &str, idle_secs: u64) {
    let mut settings = Settings {
        machine_id: Some("e2e-machine".to_string()),
        ..Settings::default()
    };
    settings.repos = vec![repo.to_string()];
    settings.worker_idle_secs = idle_secs;
    write_settings_atomic(&config_path(home.path()), &settings).expect("seed settings");
}

pub async fn start_router(home: &TempDir) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let home_path = home.path().to_path_buf();
    let app = build_router_app(RouterBootOptions {
        data_dir: Some(home_path.clone()),
        embeddings_dir: Some(home_path.join("embeddings")),
        bind: "127.0.0.1".to_string(),
        home_dir: Some(home_path),
        worker_exe: Some(worker_exe()),
        listen_port: None,
    })
    .await
    .expect("router app")
    .0;
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    addr
}

pub fn repo_id_b64(repo: &str) -> String {
    use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
    URL_SAFE_NO_PAD.encode(repo.as_bytes())
}

pub async fn poke_action(client: &Client, addr: SocketAddr, repo: &str) -> reqwest::StatusCode {
    let url = format!("http://{addr}/api/repos/{}/index", repo_id_b64(repo));
    client
        .post(&url)
        .timeout(Duration::from_secs(40))
        .send()
        .await
        .map(|response| response.status())
        .unwrap_or(reqwest::StatusCode::BAD_GATEWAY)
}

pub async fn worker_active(client: &Client, addr: SocketAddr, repo: &str) -> bool {
    let repos: serde_json::Value = client
        .get(format!("http://{addr}/api/index-status"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    repos
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| {
            entry["repo"].as_str() == Some(&context_engine_rs::store::normalize_repo_path(repo))
        })
        .map(|entry| entry["worker_active"].as_bool().unwrap_or(false))
        .unwrap_or(false)
}
