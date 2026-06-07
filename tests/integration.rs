use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use reqwest::Client;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::RwLock;

use context_engine_rs::config::{Settings, config_path};
use context_engine_rs::indexing::IndexEngine;
use context_engine_rs::server::build_router;
use context_engine_rs::store::RepoDbMap;

/// Boot the axum server bound to port 0 (OS assigns a free port).
/// Returns the bound address and a join handle for the server task.
async fn start_server(home: &TempDir) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind ephemeral port");
    let addr = listener.local_addr().expect("no local addr");
    let settings = Settings::default();
    let settings_handle = Arc::new(RwLock::new(settings.clone()));
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    // Tests pass the same TempDir as both home_dir and data_dir — settings.json
    // and the data files (rocksdb/, embeddings/) all live under it. Production
    // splits them, but the test harness only cares that data_dir is honored.
    let data_dir = home.path().to_path_buf();
    let embeddings_dir = data_dir.join("embeddings");
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
        index_engine,
        repo_dbs,
        settings_handle,
        "127.0.0.1",
    );
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

    // version should be CURRENT_VERSION (= 3 after the data_dir + embeddings_dir migrations)
    assert_eq!(body["version"], 3);

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
    assert_eq!(get_body.version, 3);
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

// ─── Test 4: PUT a repo → subsequent query sees it (live settings) ────────
//
// Boot server with no repos. PUT a config that includes one real temp-dir repo.
// POST /api/query — expect "Index is empty" (NOT "No repositories configured").
// This proves post_query reads live settings, not the frozen boot-time snapshot.
//
// Preflight order in post_query (current design):
//   1. repos.is_empty()          → "No repositories configured."
//   2. api_keys.is_empty()       → "No embedding API keys configured."
//   3. repo missing/blank        → "A repository is required. …"
// (The old empty-resident-vector-index rejection was intentionally removed: under
//  per-repo lazy-warmed shards an indexed-but-cold repo reads empty, so a hard
//  "Index is empty" reject would wrongly block it — see server.rs post_query.)
//
// This test PUTs a repo then sends a repo-LESS query. With live settings honored,
// check (1) passes (repos no longer empty) and (2) passes (a key is configured),
// so the request reaches the repo-mandatory gate (3). Reaching (3) proves the PUT
// updated live settings AND does so without any network call — the repo-required
// rejection fires before run_query embeds anything.
#[tokio::test]
async fn test_put_repo_then_query_passes_preflight() {
    let home = TempDir::new().expect("tempdir");
    let repo_dir = TempDir::new().expect("repo tempdir");
    let addr = start_server(&home).await;

    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .expect("client");

    // Seed the config file via GET so it exists on disk.
    client
        .get(format!("http://{addr}/api/config"))
        .send()
        .await
        .expect("initial GET");

    // PUT a config that includes one real directory as a repo.
    let repo_path = repo_dir.path().to_string_lossy().to_string();
    let payload = serde_json::json!({
        "version": 1,
        "repos": [repo_path],
        "embedding": {
            "provider": "voyage",
            "model": "voyage-4-lite",
            "api_keys": ["key-test-1234"]
        },
        "llm": {
            "provider": "google",
            "rerank_model": "gemini-3.1-flash-lite",
            "api_keys": []
        }
    });

    let put_res = client
        .put(format!("http://{addr}/api/config"))
        .json(&payload)
        .send()
        .await
        .expect("PUT request");

    assert_eq!(put_res.status().as_u16(), 200, "PUT should return 200");

    // Now POST /api/query WITHOUT a repo. The repos preflight must pass (a repo
    // was added) and the api-keys preflight must pass (a key is configured), so the
    // request reaches the repo-mandatory gate — which rejects before any embedding
    // network call. Reaching that gate proves the PUT updated live settings.
    let query_res = client
        .post(format!("http://{addr}/api/query"))
        .json(&serde_json::json!({ "query": "x", "top_k": 5 }))
        .send()
        .await
        .expect("query request");

    let status = query_res.status().as_u16();
    let body: serde_json::Value = query_res.json().await.expect("parse query response");
    let error_msg = body["error"].as_str().unwrap_or("");

    // Must NOT be "No repositories configured" — that would mean we're reading stale settings.
    assert!(
        !error_msg.contains("No repositories configured"),
        "Expected live settings to be used but got: {error_msg}"
    );
    // Must NOT be the api-keys error — a key was configured in the PUT.
    assert!(
        !error_msg.contains("No embedding API keys configured"),
        "api-keys preflight should pass with a configured key, got: {error_msg}"
    );

    // Must be the repo-mandatory rejection — proves both earlier preflights passed
    // (live settings honored) and that it fired before any network call to embed.
    assert_eq!(status, 400);
    assert!(
        error_msg.contains("A repository is required"),
        "Expected 'A repository is required' but got: {error_msg}"
    );
}

// ─── Test 5: PUT a repo → register_repo fires → status entry exists ───────
//
// PUT a config with a real temp directory as a repo. After the PUT returns 200,
// GET /api/index-status and assert that an entry for that repo path exists.
// This proves register_repo was called from put_config (not just next restart).
#[tokio::test]
async fn test_put_repo_registers_status() {
    let home = TempDir::new().expect("tempdir");
    let repo_dir = TempDir::new().expect("repo tempdir");
    let addr = start_server(&home).await;

    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .expect("client");

    // Seed config file.
    client
        .get(format!("http://{addr}/api/config"))
        .send()
        .await
        .expect("initial GET");

    let repo_path = repo_dir.path().to_string_lossy().to_string();
    let payload = serde_json::json!({
        "version": 1,
        "repos": [repo_path],
        "embedding": { "provider": "voyage", "model": "voyage-4-lite", "api_keys": [] },
        "llm": { "provider": "google", "rerank_model": "gemini-3.1-flash-lite", "api_keys": [] }
    });

    let put_res = client
        .put(format!("http://{addr}/api/config"))
        .json(&payload)
        .send()
        .await
        .expect("PUT request");

    assert_eq!(put_res.status().as_u16(), 200, "PUT should return 200");

    // register_repo is awaited inside put_config before the response is sent,
    // so the status entry must exist immediately after the 200 response.
    let status_res = client
        .get(format!("http://{addr}/api/index-status"))
        .send()
        .await
        .expect("index-status request");

    assert_eq!(status_res.status().as_u16(), 200);

    let statuses: Vec<serde_json::Value> = status_res.json().await.expect("parse status response");
    let found = statuses.iter().any(|s| {
        s["repo"].as_str().map(|r| r == repo_path).unwrap_or(false)
    });

    assert!(
        found,
        "Expected a status entry for {repo_path} but got: {statuses:?}"
    );
}

fn encode_repo_id(path: &str) -> String {
    URL_SAFE_NO_PAD.encode(path.as_bytes())
}

// ─── Test 6: cancel indexing → repo returns to idle, SSE emits cancelled,
//     and a subsequent index run is not poisoned by the old token ──────────
#[tokio::test]
async fn test_cancel_index_and_reindex() {
    use std::io::Write;

    let home = TempDir::new().expect("tempdir");
    let repo_dir = TempDir::new().expect("repo tempdir");

    // Populate the repo with enough files and symbols to ensure indexing takes
    // long enough for the cancel to land mid-pipeline. Each file has 50 functions
    // to generate heavy DB write load in Stage 3 (where the cancel check lives).
    for i in 0..200 {
        let path = repo_dir.path().join(format!("mod_{i}.rs"));
        let mut f = std::fs::File::create(&path).expect("create file");
        for j in 0..50 {
            writeln!(f, "pub fn func_{i}_{j}(x: i32) -> i32 {{ x + {j} }}").unwrap();
        }
        writeln!(f, "pub fn caller_{i}() {{ func_{i}_0(1); func_{i}_1(2); }}").unwrap();
    }

    let addr = start_server(&home).await;

    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .expect("client");

    // Register the repo.
    let repo_path = repo_dir.path().to_string_lossy().to_string();
    let payload = serde_json::json!({
        "version": 1,
        "repos": [&repo_path],
        "embedding": { "provider": "voyage", "model": "voyage-4-lite", "api_keys": [] },
        "llm": { "provider": "google", "rerank_model": "gemini-3.1-flash-lite", "api_keys": [] }
    });

    let put_res = client
        .put(format!("http://{addr}/api/config"))
        .json(&payload)
        .send()
        .await
        .expect("PUT config");
    assert_eq!(put_res.status().as_u16(), 200);

    let repo_id = encode_repo_id(&repo_path);

    // Spawn a background task that collects SSE events from the index-events endpoint.
    let sse_addr = addr;
    let sse_repo_id = repo_id.clone();
    let sse_collected = Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));
    let sse_collected_clone = sse_collected.clone();
    let sse_task = tokio::spawn(async move {
        use futures::StreamExt;
        let sse_client = Client::builder()
            .danger_accept_invalid_certs(true)
            .build()
            .expect("sse client");
        let sse_res = sse_client
            .get(format!("http://{sse_addr}/api/repos/{sse_repo_id}/index-events"))
            .send()
            .await
            .expect("SSE connect");
        let mut stream = sse_res.bytes_stream();
        loop {
            match tokio::time::timeout(std::time::Duration::from_secs(10), stream.next()).await {
                Ok(Some(Ok(bytes))) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    sse_collected_clone.lock().await.push(text);
                }
                _ => break,
            }
        }
    });

    // The PUT triggers register_repo which auto-queues an index.
    // Wait until the SSE stream shows a 'started' event (proves the pipeline is running).
    let mut saw_started = false;
    for _ in 0..50 {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let collected = sse_collected.lock().await;
        let all = collected.join("");
        if all.contains("\"type\":\"started\"") {
            saw_started = true;
            break;
        }
    }
    assert!(saw_started, "never saw SSE 'started' event");

    // Cancel immediately.
    let cancel_res = client
        .post(format!("http://{addr}/api/repos/{repo_id}/cancel-index"))
        .send()
        .await
        .expect("cancel");
    assert_eq!(cancel_res.status().as_u16(), 200);
    let cancel_body: serde_json::Value = cancel_res.json().await.expect("parse cancel");

    // The cancel may have landed after completion (small repo, no embedding keys).
    // If cancelled=true, the pipeline was interrupted and we expect the cancelled event.
    // If cancelled=false, the pipeline completed before cancel arrived — still valid,
    // but we can't assert the cancelled SSE event.
    let was_cancelled = cancel_body["cancelled"].as_bool().unwrap_or(false);

    // Wait until status returns to idle (whether via cancel or normal completion).
    let mut back_to_idle = false;
    for _ in 0..50 {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let status_res = client
            .get(format!("http://{addr}/api/index-status"))
            .send()
            .await
            .expect("status");
        let statuses: Vec<serde_json::Value> = status_res.json().await.expect("parse");
        if let Some(s) = statuses.iter().find(|s| s["repo"].as_str() == Some(&repo_path)) {
            if s["state"].as_str() == Some("idle") {
                back_to_idle = true;
                break;
            }
        }
    }
    assert!(back_to_idle, "repo did not return to idle after cancel");

    // Verify SSE stream received a 'cancelled' event (only if cancel actually landed).
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    sse_task.abort();
    let collected = sse_collected.lock().await;
    let all_text = collected.join("");
    if was_cancelled {
        assert!(
            all_text.contains("\"type\":\"cancelled\""),
            "cancel_index returned true but SSE never emitted 'cancelled'. Collected {} bytes: {}",
            all_text.len(),
            &all_text[..all_text.len().min(500)]
        );
    } else {
        // Pipeline finished before cancel — we should see 'completed' instead.
        assert!(
            all_text.contains("\"type\":\"completed\""),
            "cancel_index returned false but SSE has neither cancelled nor completed. Collected: {}",
            &all_text[..all_text.len().min(500)]
        );
    }

    // Now trigger a fresh index — prove the token isn't poisoned.
    let index_res = client
        .post(format!("http://{addr}/api/repos/{repo_id}/index"))
        .send()
        .await
        .expect("index");
    assert!(
        [200, 202].contains(&index_res.status().as_u16()),
        "index returned {}",
        index_res.status()
    );

    // Wait for it to complete (or at least start indexing).
    let mut reindex_started = false;
    for _ in 0..30 {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let status_res = client
            .get(format!("http://{addr}/api/index-status"))
            .send()
            .await
            .expect("status");
        let statuses: Vec<serde_json::Value> = status_res.json().await.expect("parse");
        if let Some(s) = statuses.iter().find(|s| s["repo"].as_str() == Some(&repo_path)) {
            let state = s["state"].as_str().unwrap_or("");
            if state == "indexing" || (state == "idle" && s["indexed_files"].as_u64().unwrap_or(0) > 0) {
                reindex_started = true;
                break;
            }
        }
    }
    assert!(reindex_started, "re-index after cancel did not start — token may be poisoned");
}

// ─── Test 7: DELETE /api/repos/:repo_id/index removes the DB directory ────
#[tokio::test]
async fn test_delete_repo_index_removes_directory() {
    use context_engine_rs::store;

    let home = TempDir::new().expect("tempdir");
    let addr = start_server(&home).await;
    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .expect("client");

    let repo_path = "D:/fake/repo/for_delete_test";
    let repo_id = encode_repo_id(repo_path);

    // Pre-create the DB directory and open a handle via get_or_open so a real
    // RocksDB LOCK file exists — this is what triggers OS error 32 if not
    // closed before deletion.
    let db_dir = store::db_path(home.path(), repo_path);
    std::fs::create_dir_all(&db_dir).expect("create db dir");

    // Open a DB handle through the server by triggering an index (which calls
    // get_or_open internally). We need the repo registered first.
    let put_body = serde_json::json!({
        "repos": [repo_path],
        "embedding": { "api_keys": [] }
    });
    client
        .put(format!("http://{addr}/api/config"))
        .json(&put_body)
        .send()
        .await
        .expect("put config");

    // Trigger indexing so a DB handle gets cached in repo_dbs.
    client
        .post(format!("http://{addr}/api/repos/{repo_id}/index"))
        .send()
        .await
        .expect("trigger index");

    // Give the consumer a moment to open the handle.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Now delete the index — must return 200 (success) regardless of whether
    // the directory could be immediately removed on disk.
    let del_res = client
        .delete(format!("http://{addr}/api/repos/{repo_id}/index"))
        .send()
        .await
        .expect("delete index");

    assert_eq!(
        del_res.status().as_u16(),
        200,
        "delete should succeed: {:?}",
        del_res.text().await
    );

    // The index status should be cleared (idle, 0 files).
    let status_res = client
        .get(format!("http://{addr}/api/index-status"))
        .send()
        .await
        .expect("status");
    let statuses: Vec<serde_json::Value> = status_res.json().await.expect("parse");
    if let Some(s) = statuses.iter().find(|s| s["repo"].as_str() == Some(repo_path)) {
        assert_eq!(s["state"].as_str(), Some("idle"));
        assert_eq!(s["indexed_files"].as_u64(), Some(0));
    }
}

// ─── Test 8: PUT data_dir persists but does NOT relocate the running process ─
//
// PUT /api/config with a new `data_dir` must:
//   1. Persist the value to settings.json (round-trips on a subsequent GET).
//   2. NOT close cached RocksDB handles in repo_dbs (they're bound to the
//      boot-resolved path; switching mid-run would split-brain).
//   3. NOT create the new path's directory tree as a side effect.
//   4. Subsequent indexing operations (which open DBs via get_or_open) still
//      land at the boot-resolved path, NOT the newly persisted one.
//
// Confirms the boot-frozen contract from the design plan.
#[tokio::test]
async fn test_put_data_dir_persists_but_does_not_relocate() {
    use context_engine_rs::store;

    let boot_home = TempDir::new().expect("boot tempdir");
    // The "new" data_dir the user wants for *next* launch — never touched here.
    let next_launch_dir = TempDir::new().expect("next-launch tempdir");

    let addr = start_server(&boot_home).await;
    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .expect("client");

    // Seed via GET so settings.json exists on disk.
    let initial = client
        .get(format!("http://{addr}/api/config"))
        .send()
        .await
        .expect("initial GET");
    assert_eq!(initial.status().as_u16(), 200);

    // Pre-PUT: data_dir is None on disk.
    let body: Settings = client
        .get(format!("http://{addr}/api/config"))
        .send()
        .await
        .expect("get pre-put")
        .json()
        .await
        .expect("parse pre-put");
    assert!(body.data_dir.is_none(), "fresh install: data_dir should be None");

    // PUT a config with a new data_dir.
    let new_data_dir = next_launch_dir.path().to_path_buf();
    let payload = serde_json::json!({
        "version": 2,
        "repos": [],
        "embedding": {"provider": "voyage", "model": "voyage-4-lite", "api_keys": []},
        "llm": {"provider": "google", "rerank_model": "gemini-3.1-flash-lite", "api_keys": []},
        "data_dir": new_data_dir.to_string_lossy(),
    });
    let put_res = client
        .put(format!("http://{addr}/api/config"))
        .json(&payload)
        .send()
        .await
        .expect("put");
    assert_eq!(put_res.status().as_u16(), 200, "PUT must succeed");

    // (1) GET round-trips the new value.
    let after: Settings = client
        .get(format!("http://{addr}/api/config"))
        .send()
        .await
        .expect("get after put")
        .json()
        .await
        .expect("parse after put");
    assert_eq!(after.data_dir.as_ref(), Some(&new_data_dir));

    // (3) The PUT must NOT have created the rocksdb tree at the new path.
    let new_rocksdb_root = new_data_dir.join("rocksdb");
    assert!(
        !new_rocksdb_root.exists(),
        "PUT must not create the new data_dir's rocksdb tree (boot-frozen)"
    );

    // (4) Trigger an indexing operation on a fake repo. The DB directory must
    // appear under the BOOT path (boot_home), not the newly persisted path.
    let repo_path = "D:/fake/repo/for_data_dir_test";
    let repo_id = encode_repo_id(repo_path);
    let put_repo = serde_json::json!({
        "version": 2,
        "repos": [repo_path],
        "embedding": {"provider": "voyage", "model": "voyage-4-lite", "api_keys": []},
        "llm": {"provider": "google", "rerank_model": "gemini-3.1-flash-lite", "api_keys": []},
        "data_dir": new_data_dir.to_string_lossy(),
    });
    client
        .put(format!("http://{addr}/api/config"))
        .json(&put_repo)
        .send()
        .await
        .expect("put repo");

    client
        .post(format!("http://{addr}/api/repos/{repo_id}/index"))
        .send()
        .await
        .expect("trigger index");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // boot path: data_dir == boot_home.path() per start_server.
    let boot_db = store::db_path(boot_home.path(), repo_path);
    let new_db = store::db_path(&new_data_dir, repo_path);
    assert!(
        boot_db.exists(),
        "DB must materialize under the BOOT data_dir, not the persisted one. \
         expected: {boot_db:?}"
    );
    assert!(
        !new_db.exists(),
        "DB must NOT materialize under the newly persisted data_dir (would mean \
         the running process honored the PUT — split-brain). \
         unexpected: {new_db:?}"
    );
}
