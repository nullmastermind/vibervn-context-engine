//! Integration tests for ROUTER mode (process-per-project).
//!
//! These boot the real router axum app on an ephemeral port with a HERMETIC
//! home dir (a TempDir, so no developer settings.json is read), and exercise the
//! parts that do NOT require a real worker subprocess:
//! - the router serves global endpoints (`/api/config`, `/api/repos`, `/`)
//!   natively, with NO IndexEngine and NO repo DB open;
//! - cold per-repo display (`/index-stats`, `/graph`) reads the sidecar / returns
//!   a placeholder WITHOUT spawning a worker;
//! - a corrupt sidecar degrades to the cold placeholder (never 500);
//! - the repo list reflects sidecar contents.
//!
//! A full spawn→proxy→idle-exit→respawn test needs the compiled `context-engine`
//! binary on PATH and a real on-disk index; that is covered by the worker-side
//! unit tests (readiness parse, single-flight election) plus the machine-local
//! MEASURE-gate (`measure_cold_open.rs`). Here we pin the router's OWN contract.

use std::net::SocketAddr;

use reqwest::Client;
use tempfile::TempDir;
use tokio::net::TcpListener;

use context_engine_rs::config::{Settings, config_path, write_settings_atomic};
use context_engine_rs::router::sidecar::{
    RepoSidecar, SIDECAR_SCHEMA, sidecar_path, write_sidecar,
};
use context_engine_rs::router::{RouterBootOptions, build_router_app};

/// Boot the router app on an ephemeral port with `home`/`data_dir` = the given
/// TempDir. Returns the bound address.
async fn start_router(home: &TempDir) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let home_path = home.path().to_path_buf();
    let app = build_router_app(RouterBootOptions {
        data_dir: Some(home_path.clone()),
        embeddings_dir: Some(home_path.join("embeddings")),
        bind: "127.0.0.1".to_string(),
        home_dir: Some(home_path),
        worker_exe: None,
    })
    .await
    .expect("router app builds");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("router serve");
    });
    addr
}

/// Seed a settings.json under `home` with the given repos.
fn seed_settings(home: &TempDir, repos: &[&str]) {
    let mut s = Settings {
        machine_id: Some("test-machine".to_string()),
        ..Settings::default()
    };
    s.repos = repos.iter().map(|r| r.to_string()).collect();
    write_settings_atomic(&config_path(home.path()), &s).expect("seed settings");
}

#[tokio::test]
async fn router_serves_config_natively_without_engine() {
    let home = TempDir::new().unwrap();
    seed_settings(&home, &[]);
    let addr = start_router(&home).await;
    let resp = Client::new()
        .get(format!("http://{addr}/api/config"))
        .send()
        .await
        .expect("GET /api/config");
    assert!(resp.status().is_success(), "router serves config natively");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body.get("repos").is_some(), "config has repos field");
}

#[tokio::test]
async fn repo_list_reflects_sidecar() {
    let home = TempDir::new().unwrap();
    let repo = r"d:\projects\rust\demo";
    seed_settings(&home, &[repo]);

    // Write a sidecar as a worker would after indexing.
    write_sidecar(
        home.path(),
        repo,
        &RepoSidecar {
            file_count: 123,
            last_indexed_at: Some("2026-06-25T00:00:00+00:00".to_string()),
            state: "indexed".to_string(),
            embedding_model: "voyage-code-3".to_string(),
            embedding_dim: 1024,
            schema: SIDECAR_SCHEMA,
        },
    )
    .unwrap();

    let addr = start_router(&home).await;
    let body: serde_json::Value = Client::new()
        .get(format!("http://{addr}/api/repos"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let repos = body["repos"].as_array().expect("repos array");
    assert_eq!(repos.len(), 1);
    assert_eq!(repos[0]["file_count"], 123);
    assert_eq!(repos[0]["state"], "indexed");
    assert_eq!(repos[0]["worker_active"], false);
}

#[tokio::test]
async fn cold_index_stats_served_from_sidecar_no_worker() {
    let home = TempDir::new().unwrap();
    let repo = r"d:\projects\rust\demo2";
    seed_settings(&home, &[repo]);
    write_sidecar(
        home.path(),
        repo,
        &RepoSidecar {
            file_count: 7,
            last_indexed_at: Some("2026-06-25T00:00:00+00:00".to_string()),
            state: "indexed".to_string(),
            embedding_model: "voyage-code-3".to_string(),
            embedding_dim: 1024,
            schema: SIDECAR_SCHEMA,
        },
    )
    .unwrap();

    let addr = start_router(&home).await;
    // The repo_id in the path is the normalized repo path. We URL-encode it.
    let encoded = urlencoding_encode(repo);
    let body: serde_json::Value = Client::new()
        .get(format!("http://{addr}/api/repos/{encoded}/index-stats"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["files"], 7, "cold stats served from sidecar");
    assert_eq!(body["state"], "indexed_cold");
}

#[tokio::test]
async fn cold_index_stats_corrupt_sidecar_degrades_not_500() {
    let home = TempDir::new().unwrap();
    let repo = r"d:\projects\rust\demo3";
    seed_settings(&home, &[repo]);
    // Corrupt sidecar (truncated JSON) — reader must treat as cold placeholder.
    std::fs::create_dir_all(home.path().join("sidecar")).unwrap();
    std::fs::write(sidecar_path(home.path(), repo), b"{ truncated").unwrap();

    let addr = start_router(&home).await;
    let encoded = urlencoding_encode(repo);
    let resp = Client::new()
        .get(format!("http://{addr}/api/repos/{encoded}/index-stats"))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "corrupt sidecar must NOT 500 — degrade to placeholder"
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["state"], "not_indexed",
        "corrupt sidecar reads as not_indexed placeholder"
    );
}

#[tokio::test]
async fn cold_graph_returns_empty_placeholder_no_worker() {
    let home = TempDir::new().unwrap();
    let repo = r"d:\projects\rust\demo4";
    seed_settings(&home, &[repo]);
    let addr = start_router(&home).await;
    let encoded = urlencoding_encode(repo);
    let body: serde_json::Value = Client::new()
        .get(format!("http://{addr}/api/repos/{encoded}/graph"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["cold"], true, "cold graph flagged");
    assert!(body["nodes"].as_array().unwrap().is_empty());
}

/// EDGE CASE (the phantom-dir false-positive guard): `/api/index-status` must
/// report a repo with NO sidecar as `not_indexed` — EVEN IF a RocksDB directory
/// exists on disk for it. `open_db` creates the dir + runs SCHEMA_DDL on every
/// open, so a dir alone is NOT proof of real data (a never-indexed repo a worker
/// merely opened, or a crashed first index, leaves an empty dir). The sidecar —
/// written only when count>0 — is the single source of truth. A bare dir must
/// never flip the badge to "indexed".
#[tokio::test]
async fn index_status_empty_dir_without_sidecar_is_not_indexed() {
    let home = TempDir::new().unwrap();
    let repo = r"d:\projects\rust\phantom";
    seed_settings(&home, &[repo]);

    // Materialize an EMPTY RocksDB-style dir for the repo (no sidecar, no data) —
    // mimicking a dir that open_db's create_dir_all left behind for a repo that
    // was never actually indexed. Path mirrors store::db_path(gen 0).
    let sane = repo
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect::<String>();
    let trimmed = sane.trim_matches('_');
    std::fs::create_dir_all(home.path().join("rocksdb").join(trimmed)).unwrap();

    let addr = start_router(&home).await;
    // Give the boot backfill a beat; with count==0 it must write NO sidecar.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    let arr: serde_json::Value = Client::new()
        .get(format!("http://{addr}/api/index-status"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let entry = arr
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["repo"].as_str() == Some(&normalize(repo)))
        .expect("repo present in index-status");
    assert_eq!(
        entry["state"], "not_indexed",
        "an empty/phantom dir with no sidecar must report not_indexed, never a false 'indexed'"
    );
}

/// `/api/index-status` reflects a real sidecar's count for a cold repo.
#[tokio::test]
async fn index_status_reads_real_count_from_sidecar() {
    let home = TempDir::new().unwrap();
    let repo = r"d:\projects\rust\hassidecar";
    seed_settings(&home, &[repo]);
    write_sidecar(
        home.path(),
        repo,
        &RepoSidecar {
            file_count: 88,
            last_indexed_at: Some("2026-06-25T00:00:00+00:00".to_string()),
            state: "indexed".to_string(),
            embedding_model: "voyage-code-3".to_string(),
            embedding_dim: 1024,
            schema: SIDECAR_SCHEMA,
        },
    )
    .unwrap();

    let addr = start_router(&home).await;
    let arr: serde_json::Value = Client::new()
        .get(format!("http://{addr}/api/index-status"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let entry = arr
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["repo"].as_str() == Some(&normalize(repo)))
        .expect("repo present");
    assert_eq!(entry["state"], "indexed");
    assert_eq!(
        entry["indexed_files"], 88,
        "real count surfaces from sidecar"
    );
}

/// Normalize a repo path the same way the router keys it (the test only needs
/// the lowercase + separator-normalized form to match index-status' `repo`).
fn normalize(repo: &str) -> String {
    context_engine_rs::store::normalize_repo_path(repo)
}

/// Encode the repo path as the `:repo_id` path segment the router expects:
/// URL_SAFE_NO_PAD base64 (the worker's `decode_repo_id` contract, which the
/// router's `resolve_repo_id` decodes to pick the worker).
fn urlencoding_encode(s: &str) -> String {
    use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
    URL_SAFE_NO_PAD.encode(s.as_bytes())
}

// ── SPAWN-POLICY: opening a repo detail must spawn ZERO workers ───────────────
//
// The core regression this whole change targets: GET status/graph/files/
// index-stats (what loadExplorer fires on detail-open) must serve cold and NEVER
// start a worker. The router has no live worker for a repo whose detail is merely
// viewed; these assertions hit each view endpoint and confirm the GET succeeds
// without a worker subprocess having been spawned. (We can't observe child PIDs
// from here without the worker binary, but a cold serve returning 200 with
// `cold:true`/sidecar data — not a 503 warming — proves no spawn was attempted:
// acquire_and_proxy would have 503'd while spawning, proxy_if_live serves cold.)

/// Seed settings + a sidecar (meta+graph+files) for a repo so the cold view has
/// data to serve.
fn seed_indexed_repo(home: &TempDir, repo: &str) {
    seed_settings(home, &[repo]);
    write_sidecar(
        home.path(),
        repo,
        &RepoSidecar {
            file_count: 12,
            last_indexed_at: Some("2026-06-25T00:00:00+00:00".to_string()),
            state: "indexed".to_string(),
            embedding_model: "voyage-code-3".to_string(),
            embedding_dim: 1024,
            schema: SIDECAR_SCHEMA,
        },
    )
    .unwrap();
    // Aux graph + files sidecars (what the worker writes at index-complete).
    context_engine_rs::router::sidecar::write_aux_json(
        home.path(),
        repo,
        "graph",
        &serde_json::json!({ "nodes": [], "edges": [], "truncated": false }),
    )
    .unwrap();
    context_engine_rs::router::sidecar::write_aux_json(
        home.path(),
        repo,
        "files",
        &serde_json::json!([{ "path": "src/a.rs", "size": 10, "mtime": 0, "chunk_count": 1 }]),
    )
    .unwrap();
}

#[tokio::test]
async fn detail_view_endpoints_serve_cold_without_spawn() {
    let home = TempDir::new().unwrap();
    let repo = r"d:\projects\rust\viewonly";
    seed_indexed_repo(&home, repo);
    let addr = start_router(&home).await;
    let id = urlencoding_encode(repo);
    let client = Client::new();

    // Each view endpoint must return 200 (cold serve), NOT 503 (which would mean
    // a spawn was attempted). status / index-stats / graph / files / index-events
    // / ignored-files.
    for (path, _label) in [
        (format!("/api/repos/{id}/status"), "status"),
        (format!("/api/repos/{id}/index-stats"), "index-stats"),
        (format!("/api/repos/{id}/graph"), "graph"),
        (format!("/api/repos/{id}/files"), "files"),
        (format!("/api/repos/{id}/ignored-files"), "ignored-files"),
    ] {
        let res = client
            .get(format!("http://{addr}{path}"))
            .send()
            .await
            .unwrap();
        assert!(
            res.status().is_success(),
            "{path} must serve cold (200), got {} — a 503 would mean a worker spawn was attempted",
            res.status()
        );
    }

    // index-events cold → 204 No Content (stream closes, no spawn).
    let ev = client
        .get(format!("http://{addr}/api/repos/{id}/index-events"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        ev.status().as_u16(),
        204,
        "cold index-events must be 204 (closed stream), never a spawn"
    );
}

#[tokio::test]
async fn graph_cold_serves_cached_sidecar() {
    let home = TempDir::new().unwrap();
    let repo = r"d:\projects\rust\graphcache";
    seed_settings(&home, &[repo]);
    context_engine_rs::router::sidecar::write_aux_json(
        home.path(),
        repo,
        "graph",
        &serde_json::json!({
            "nodes": [{
                "id": "symbol:a", "name": "a", "kind": "function",
                "file": "src/a.rs", "line_start": 1, "line_end": 5
            }],
            "edges": [],
            "truncated": false
        }),
    )
    .unwrap();
    let addr = start_router(&home).await;
    let id = urlencoding_encode(repo);
    let body: serde_json::Value = Client::new()
        .get(format!("http://{addr}/api/repos/{id}/graph"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["cold"], true, "cold graph flagged");
    assert_eq!(
        body["nodes"].as_array().unwrap().len(),
        1,
        "cold graph serves the cached node from the sidecar, not an empty placeholder"
    );
}

#[tokio::test]
async fn cancel_index_cold_is_noop_200() {
    let home = TempDir::new().unwrap();
    let repo = r"d:\projects\rust\cancelcold";
    seed_settings(&home, &[repo]);
    let addr = start_router(&home).await;
    let id = urlencoding_encode(repo);
    let res = Client::new()
        .post(format!("http://{addr}/api/repos/{id}/cancel-index"))
        .send()
        .await
        .unwrap();
    assert!(
        res.status().is_success(),
        "cold cancel-index must be a no-op 200 (no run, no spawn)"
    );
}

#[tokio::test]
async fn chat_delete_cold_is_noop_200() {
    let home = TempDir::new().unwrap();
    let repo = r"d:\projects\rust\chatcold";
    seed_settings(&home, &[repo]);
    let addr = start_router(&home).await;
    let id = urlencoding_encode(repo);
    let res = Client::new()
        .delete(format!("http://{addr}/api/repos/{id}/chat/some-convo-id"))
        .send()
        .await
        .unwrap();
    assert!(
        res.status().is_success(),
        "cold chat-DELETE must be a no-op 200 (conversation store is per-worker; nothing resident)"
    );
}

#[tokio::test]
async fn remove_repo_drops_it_and_clears_sidecars() {
    let home = TempDir::new().unwrap();
    let repo = r"d:\projects\rust\toremove";
    seed_indexed_repo(&home, repo);
    // Confirm sidecars exist pre-remove.
    assert!(
        sidecar_path(home.path(), repo).exists(),
        "meta sidecar should exist before remove"
    );
    let addr = start_router(&home).await;
    let id = urlencoding_encode(repo);

    let res = Client::new()
        .delete(format!(
            "http://{addr}/api/repos/{id}/index?remove_repo=true"
        ))
        .send()
        .await
        .unwrap();
    assert!(
        res.status().is_success(),
        "remove repo should succeed (no worker live → kill is a no-op, then native remove)"
    );

    // Sidecars cleared.
    assert!(
        !sidecar_path(home.path(), repo).exists(),
        "meta sidecar must be removed on repo removal"
    );
    // Repo dropped from settings (durable).
    let cfg = context_engine_rs::config::ensure_dir_and_load(home.path()).unwrap();
    assert!(
        !cfg.repos.iter().any(|r| r == &normalize(repo)),
        "repo must be dropped from settings.repos on remove_repo=true"
    );
}

/// FIRST-ADD policy negative case: a PUT /api/config that changes a NON-repo
/// field (here: an embedding API key) with the SAME repo set must NOT spawn any
/// worker — only a genuinely-new repo triggers an auto-index. We assert the PUT
/// succeeds, the field persisted, and `/index-status` still reports
/// `worker_active:false` for the existing repo (no spawn attempted). The common
/// "edit settings" path must stay spawn-free.
#[tokio::test]
async fn put_config_without_new_repo_does_not_spawn() {
    let home = TempDir::new().unwrap();
    let repo = r"d:\projects\rust\existing";
    seed_settings(&home, &[repo]);
    let addr = start_router(&home).await;
    let client = Client::new();

    // PUT the SAME repo set but add an embedding API key (a non-repo change).
    let mut cfg = context_engine_rs::config::ensure_dir_and_load(home.path()).unwrap();
    cfg.embedding.api_keys = vec!["test-key".to_string()];
    let body = serde_json::to_value(&cfg).unwrap();
    let res = client
        .put(format!("http://{addr}/api/config"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert!(res.status().is_success(), "PUT config must succeed");

    // Field persisted.
    let reloaded = context_engine_rs::config::ensure_dir_and_load(home.path()).unwrap();
    assert_eq!(
        reloaded.embedding.api_keys,
        vec!["test-key".to_string()],
        "non-repo field must persist"
    );

    // Give any (erroneous) spawn a beat to register, then assert none happened.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let arr: serde_json::Value = client
        .get(format!("http://{addr}/api/index-status"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let active = arr
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["repo"].as_str() == Some(&normalize(repo)))
        .map(|e| e["worker_active"].as_bool().unwrap_or(false))
        .unwrap_or(false);
    assert!(
        !active,
        "a PUT that adds NO new repo must not spawn a worker (worker_active stayed false)"
    );
}
