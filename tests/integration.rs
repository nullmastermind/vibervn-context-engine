use std::net::SocketAddr;

use reqwest::Client;
use tempfile::TempDir;
use tokio::net::TcpListener;

use context_engine_rs::config::{Settings, config_path};
use context_engine_rs::server::build_router;

/// Boot the axum server bound to port 0 (OS assigns a free port).
/// Returns the bound address and a join handle for the server task.
async fn start_server(home: &TempDir) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind ephemeral port");
    let addr = listener.local_addr().expect("no local addr");
    let app = build_router(home.path().to_path_buf());
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("server error");
    });
    addr
}

// ─── Test 1: GET creates default settings ────────────────────────────────
#[tokio::test]
async fn test_get_creates_default() {
    let home = TempDir::new().expect("tempdir");
    let addr = start_server(&home).await;

    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .expect("client");

    let res = client
        .get(format!("http://{addr}/api/config"))
        .send()
        .await
        .expect("request failed");

    assert_eq!(res.status().as_u16(), 200);

    let body: serde_json::Value = res.json().await.expect("parse json");

    // version should be 1
    assert_eq!(body["version"], 1);

    // repos should be an empty array
    assert!(body["repos"].as_array().map(|a| a.is_empty()).unwrap_or(false));

    // settings.json should have been created on disk
    let settings_path = config_path(home.path());
    assert!(settings_path.exists(), "settings.json was not created");

    // Deserialize from disk and verify it matches defaults
    let on_disk: Settings =
        serde_json::from_str(&std::fs::read_to_string(&settings_path).expect("read"))
            .expect("deserialize");
    assert_eq!(on_disk, Settings::default());
}

// ─── Test 2: PUT round-trip ────────────────────────────────────────────────
#[tokio::test]
async fn test_put_round_trips() {
    let home = TempDir::new().expect("tempdir");
    let addr = start_server(&home).await;

    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .expect("client");

    // Seed with a GET so the file is created.
    client
        .get(format!("http://{addr}/api/config"))
        .send()
        .await
        .expect("initial GET");

    let payload = serde_json::json!({
        "version": 1,
        "repos": ["/home/user/myproject", "/home/user/other"],
        "embedding": {
            "provider": "voyage",
            "model": "voyage-code-3",
            "api_keys": ["key-abc-1234", "key-xyz-5678"]
        },
        "llm": {
            "provider": "google",
            "rerank_model": "gemini-2.0-flash",
            "api_keys": ["google-api-key-9999"]
        }
    });

    let put_res = client
        .put(format!("http://{addr}/api/config"))
        .json(&payload)
        .send()
        .await
        .expect("PUT request");

    assert_eq!(put_res.status().as_u16(), 200, "PUT should return 200");

    let put_body: Settings = put_res.json().await.expect("parse PUT response");

    // GET to verify round-trip
    let get_res = client
        .get(format!("http://{addr}/api/config"))
        .send()
        .await
        .expect("GET after PUT");

    assert_eq!(get_res.status().as_u16(), 200);

    let get_body: Settings = get_res.json().await.expect("parse GET response");

    assert_eq!(put_body, get_body, "PUT response and subsequent GET should be equal");
    assert_eq!(get_body.repos, vec!["/home/user/myproject", "/home/user/other"]);
    assert_eq!(get_body.embedding.model, "voyage-code-3");
    assert_eq!(get_body.llm.rerank_model, "gemini-2.0-flash");
    assert_eq!(get_body.version, 1);
}

// ─── Test 3 (Unix only): file mode bits should be 0o600 ───────────────────
#[cfg(unix)]
#[tokio::test]
async fn test_unix_file_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let home = TempDir::new().expect("tempdir");
    let addr = start_server(&home).await;

    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .expect("client");

    // GET creates the default file.
    let res = client
        .get(format!("http://{addr}/api/config"))
        .send()
        .await
        .expect("GET");

    assert_eq!(res.status().as_u16(), 200);

    let settings_path = config_path(home.path());
    let meta = std::fs::metadata(&settings_path).expect("stat settings.json");
    let mode = meta.permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "settings.json should have mode 0o600, got 0o{mode:o}");
}
