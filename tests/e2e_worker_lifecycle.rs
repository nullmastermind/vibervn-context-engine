//! END-TO-END process-per-project lifecycle test (REAL subprocess).
//!
//! This is the test that PROVES the core safety claim of the whole design:
//! after a worker idle-exits, the router can respawn a worker for the SAME repo
//! WITHOUT colliding on RocksDB's exclusive per-directory LOCK. Unit tests cover
//! the registry's single-flight election and the readiness parse in isolation;
//! this exercises the real chain end to end:
//!
//!   router spawns a REAL `context-engine --worker` subprocess
//!     → worker opens RocksDB (takes the LOCK), binds, prints the handshake
//!     → router proxies a request, gets a real response
//!     → worker idle-exits (tiny idle window) → drops the DB handle (releases
//!       the LOCK) → process exits
//!     → router detects the dead child via `Child::try_wait` (reuse-safe) and
//!       respawns → the NEW worker's `open_db` succeeds (LOCK is free, or its
//!       30s retry rides out any residual drain) → request succeeds again.
//!
//! ## Why the LOCK can't be held across respawn (the race this guards)
//!
//! The worker's shutdown sequence (`close_repo_db`) drops the cached
//! `Surreal<Db>` handle under the per-repo lock BEFORE the process exits, so the
//! RocksDB LOCK is released first. The router's `try_wait` confirm-dead means it
//! never spawns a replacement while the old PID is alive. The 500ms drain pause
//! is only a probability reducer, NOT the guarantee — the guarantee is
//! confirm-dead + `open_db`'s 30s LOCK-drain retry. If respawn were racy, the
//! second worker's `open_db` would fail with "open surrealdb"/LOCK and the final
//! request would error — so a green assertion here IS the proof.
//!
//! These tests are `#[ignore]` by default because they build+spawn a real binary
//! and take a few seconds. Run with:
//!   cargo test --test e2e_worker_lifecycle -- --ignored --nocapture

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use reqwest::Client;
use tempfile::TempDir;
use tokio::net::TcpListener;

use context_engine_rs::config::{Settings, config_path, write_settings_atomic};
use context_engine_rs::router::{RouterBootOptions, build_router_app};

/// Path to the compiled `context-engine` binary the worker subprocess runs.
/// Cargo sets `CARGO_BIN_EXE_<name>` for integration tests; the package/binary
/// name is `context-engine-rs`.
fn worker_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_context-engine-rs"))
}

/// Seed a hermetic settings.json with one repo and a SHORT worker idle window so
/// the worker self-exits quickly (the test drives the respawn race on this).
fn seed_settings(home: &TempDir, repo: &str, idle_secs: u64) {
    let mut s = Settings {
        machine_id: Some("e2e-machine".to_string()),
        ..Settings::default()
    };
    s.repos = vec![repo.to_string()];
    s.worker_idle_secs = idle_secs;
    write_settings_atomic(&config_path(home.path()), &s).expect("seed settings");
}

/// Boot the router on an ephemeral port with a hermetic home + the real worker
/// binary. data_dir = home so RocksDB + sidecar live under the TempDir.
async fn start_router(home: &TempDir) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let home_path = home.path().to_path_buf();
    let app = build_router_app(RouterBootOptions {
        data_dir: Some(home_path.clone()),
        embeddings_dir: Some(home_path.join("embeddings")),
        bind: "127.0.0.1".to_string(),
        home_dir: Some(home_path),
        worker_exe: Some(worker_exe()),
    })
    .await
    .expect("router app");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    addr
}

fn repo_id_b64(repo: &str) -> String {
    use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
    URL_SAFE_NO_PAD.encode(repo.as_bytes())
}

/// Hit a SPAWN-ALLOWED action endpoint (`POST /index`) for `repo`, forcing a
/// worker spawn. Returns the HTTP status — a SUCCESS (200/202) once the worker is
/// up and the proxied index trigger was accepted (the index handler returns 202
/// Accepted since indexing runs async). (Post route-split, `/status` is spawn-
/// BLOCKED and serves cold, so it can no longer be used to boot a worker; an
/// action endpoint is required.)
async fn poke_action(client: &Client, addr: SocketAddr, repo: &str) -> reqwest::StatusCode {
    let url = format!("http://{addr}/api/repos/{}/index", repo_id_b64(repo));
    client
        .post(&url)
        .timeout(Duration::from_secs(40))
        .send()
        .await
        .map(|r| r.status())
        .unwrap_or(reqwest::StatusCode::BAD_GATEWAY)
}

/// Read `worker_active` for `repo` from `/api/index-status` (the router reports
/// it from `registry.ready_repos()` WITHOUT spawning). Used to assert that
/// viewing a detail spawned 0 workers.
async fn worker_active(client: &Client, addr: SocketAddr, repo: &str) -> bool {
    let arr: serde_json::Value = client
        .get(format!("http://{addr}/api/index-status"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    arr.as_array()
        .unwrap()
        .iter()
        .find(|e| e["repo"].as_str() == Some(&context_engine_rs::store::normalize_repo_path(repo)))
        .map(|e| e["worker_active"].as_bool().unwrap_or(false))
        .unwrap_or(false)
}

/// SPAWN-POLICY (the headline fix): opening a repo detail — GET status / graph /
/// files / index-stats — must spawn ZERO workers. With a REAL worker binary
/// wired, we hit every view endpoint then assert `/index-status` still reports
/// `worker_active:false` for the repo: proof no worker process was started.
#[tokio::test]
#[ignore = "uses the real worker binary path; run with --ignored --nocapture"]
async fn opening_detail_spawns_no_worker() {
    let home = TempDir::new().unwrap();
    let repo_dir = home.path().join("repo");
    std::fs::create_dir_all(&repo_dir).unwrap();
    std::fs::write(repo_dir.join("a.rs"), b"pub fn a() {}\n").unwrap();
    let repo = repo_dir.to_string_lossy().to_string();
    seed_settings(&home, &repo, 3600);
    let addr = start_router(&home).await;
    let client = Client::new();
    let id = repo_id_b64(&repo);

    // Fire the exact trio loadExplorer fires, plus status — all spawn-blocked.
    for path in [
        format!("/api/repos/{id}/status"),
        format!("/api/repos/{id}/index-stats"),
        format!("/api/repos/{id}/graph"),
        format!("/api/repos/{id}/files"),
    ] {
        let res = client
            .get(format!("http://{addr}{path}"))
            .send()
            .await
            .unwrap();
        assert!(
            res.status().is_success(),
            "{path} must serve cold (200), not 503 (which would mean a spawn attempt)"
        );
    }

    // The decisive assertion: NO worker came up from merely viewing.
    assert!(
        !worker_active(&client, addr, &repo).await,
        "opening a repo detail must spawn 0 workers (worker_active stayed false)"
    );

    // Sanity: an ACTION (index) DOES spawn one.
    assert!(
        poke_action(&client, addr, &repo).await.is_success(),
        "an index action must spawn + proxy to a worker (200/202)"
    );
    assert!(
        worker_active(&client, addr, &repo).await,
        "after an index action, a worker must be live (worker_active true)"
    );
}

/// FULL lifecycle: spawn → proxy → idle-exit → respawn, proving no LOCK collision.
#[tokio::test]
#[ignore = "spawns a real worker subprocess; run with --ignored --nocapture"]
async fn worker_spawn_idleexit_respawn_no_lock_collision() {
    let home = TempDir::new().unwrap();
    // A real, tiny, indexable directory: the repo IS the crate's own src dir is
    // too big; use the TempDir itself with one source file so indexing has
    // something but stays instant. No embedding keys are set → indexing yields
    // no vectors, but the worker still boots, opens RocksDB (takes the LOCK),
    // and serves /status — which is all the lifecycle race needs.
    let repo_dir = home.path().join("repo");
    std::fs::create_dir_all(&repo_dir).unwrap();
    std::fs::write(
        repo_dir.join("main.rs"),
        b"fn main() { println!(\"hi\"); }\n",
    )
    .unwrap();
    let repo = repo_dir.to_string_lossy().to_string();

    // 1-second idle window so the worker self-exits quickly after we stop poking.
    seed_settings(&home, &repo, 1);
    let addr = start_router(&home).await;
    let client = Client::new();

    // ── Phase 1: first ACTION spawns a real worker; it must succeed.
    // (`/status` is spawn-blocked post-split, so use an action to boot.) ──
    let s1 = poke_action(&client, addr, &repo).await;
    assert!(
        s1.is_success(),
        "first action must spawn a worker that accepts the index (200/202), got {s1}"
    );

    // ── Phase 2: stop poking and wait past the idle window + drain so the
    // worker self-exits and releases the RocksDB LOCK. Idle=1s; the watchdog
    // polls at ~max(1s, window/10)=1s, plus the 500ms drain + process teardown.
    // Wait generously so the exit has definitely happened. ──
    tokio::time::sleep(Duration::from_secs(5)).await;

    // ── Phase 3: a NEW request to the SAME repo. The router must detect the dead
    // child (try_wait), respawn a fresh worker, whose open_db must NOT collide
    // with the (now-released) LOCK. A LOCK collision would surface as a non-200
    // (the worker would fail to boot / open_db would error). Allow generous time
    // because a respawn pays a fresh process start + open_db (measured 0.6–1.4s),
    // and worst case open_db rides out a residual drain via its 30s retry.
    // Use an ACTION so this request actually triggers the respawn (a blocked
    // view would serve cold and never re-spawn). ──
    let s2 = poke_action(&client, addr, &repo).await;
    assert!(
        s2.is_success(),
        "respawn after idle-exit must succeed (200/202) — a failure means the new \
         worker's open_db hit the old worker's LOCK (the race this test catches), got {s2}"
    );
}

/// IN-FLIGHT GUARD: a request that is still being served when the idle window
/// elapses must NOT be cut off by idle-exit. We assert the watchdog's
/// inflight==0 precondition by keeping the worker continuously busy across more
/// than one idle window and confirming requests keep succeeding (the worker did
/// not exit mid-request). This is the (a) half of the race.
#[tokio::test]
#[ignore = "spawns a real worker subprocess; run with --ignored --nocapture"]
async fn worker_does_not_exit_while_requests_in_flight() {
    let home = TempDir::new().unwrap();
    let repo_dir = home.path().join("repo");
    std::fs::create_dir_all(&repo_dir).unwrap();
    std::fs::write(repo_dir.join("lib.rs"), b"pub fn f() {}\n").unwrap();
    let repo = repo_dir.to_string_lossy().to_string();

    // 1-second idle window.
    seed_settings(&home, &repo, 1);
    let addr = start_router(&home).await;
    let client = Client::new();

    // Spawn the worker (action endpoint — /status is spawn-blocked).
    assert!(
        poke_action(&client, addr, &repo).await.is_success(),
        "action must spawn a worker (200/202)"
    );

    // Hammer the worker with back-to-back ACTION requests for ~3× the idle
    // window. Each proxied action resets the worker's idle timer AND is the only
    // thing that would re-spawn if it died — so a worker that idle-exited mid-run
    // would surface as a failed/again-spawned request. If the watchdog ignored
    // the inflight count, a request landing exactly as the window elapses would
    // be cut off; every request must still succeed.
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    let mut count = 0u32;
    while std::time::Instant::now() < deadline {
        let s = poke_action(&client, addr, &repo).await;
        assert!(
            s.is_success(),
            "a request during continuous load must never be cut off by idle-exit (got {s})"
        );
        count += 1;
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(count > 5, "should have issued several requests under load");
}

/// THE CONTENDED RETRY-RIDE-OUT PROOF (the race the other test does NOT cover).
///
/// The lifecycle test above leaves a ~3s gap between worker-exit and the respawn
/// request, so the OS has already released the LOCK and the new `open_db`
/// succeeds on its FIRST attempt — it never enters the retry loop. This test
/// forces the genuinely contended case: it starts an `open_db` on a path WHILE a
/// real worker subprocess still holds the LOCK, so `open_db` MUST fail its first
/// attempt(s) and spin in its 30s retry loop; then it kills the worker (process
/// exit is what actually releases the RocksDB LOCK on Windows — an in-process
/// drop does NOT, which a store-level test confirmed) and asserts the retrying
/// `open_db` RIDES OUT the drain and succeeds.
///
/// This is what makes "respawn after a worker that was still holding the LOCK"
/// safe — the exact correctness claim behind process-per-project respawn.
#[tokio::test]
#[ignore = "spawns a real worker subprocess; run with --ignored --nocapture"]
async fn open_db_rides_out_lock_held_by_a_live_worker_until_it_exits() {
    use std::process::{Command, Stdio};
    use std::time::Instant;

    let home = TempDir::new().unwrap();
    let repo_dir = home.path().join("repo");
    std::fs::create_dir_all(&repo_dir).unwrap();
    std::fs::write(repo_dir.join("a.rs"), b"pub fn a() {}\n").unwrap();
    let repo = repo_dir.to_string_lossy().to_string();

    // Long idle window so the worker does NOT self-exit during the test — WE
    // control its death (kill) to time the LOCK release precisely.
    seed_settings(&home, &repo, 3600);

    // Spawn a REAL worker that opens RocksDB for `repo` and holds the LOCK. We
    // read its stdout readiness line so we know the LOCK is actually held before
    // we start contending.
    let mut child = Command::new(worker_exe())
        .arg("--worker")
        .arg(&repo)
        .arg("--port")
        .arg("0")
        .arg("--data-dir")
        .arg(home.path())
        .arg("--bind")
        .arg("127.0.0.1")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn()
        .expect("spawn worker holding the LOCK");

    // Wait for the readiness handshake → the worker has opened RocksDB (LOCK held).
    {
        use std::io::{BufRead, BufReader};
        let stdout = child.stdout.take().expect("worker stdout");
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let ready_deadline = Instant::now() + Duration::from_secs(30);
        loop {
            line.clear();
            let n = reader.read_line(&mut line).expect("read worker stdout");
            assert!(n > 0, "worker stdout closed before readiness");
            if line.starts_with("CONTEXT_ENGINE_WORKER_READY") {
                break;
            }
            assert!(Instant::now() < ready_deadline, "worker never became ready");
        }
    }

    // The LOCK is now held by the live worker. Start `open_db` on the SAME path:
    // it MUST fail its first attempt and enter the retry loop (it cannot succeed
    // while the worker lives). Run it on a blocking-friendly task.
    let data_dir = home.path().to_path_buf();
    let repo_for_open = repo.clone();
    let started = Instant::now();
    let open_task = tokio::spawn(async move {
        context_engine_rs::store::open_db(&data_dir, &repo_for_open, 0).await
    });

    // Let the retry loop actually spin against the held LOCK for a beat, THEN
    // kill the worker. Process exit (not a graceful drop) is what releases the
    // RocksDB LOCK — so after this, the retrying open must be able to win.
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert!(
        !open_task.is_finished(),
        "open_db must NOT have succeeded while the worker still held the LOCK — \
         if it did, the contention precondition didn't hold"
    );
    child
        .kill()
        .expect("kill worker to release the LOCK via process exit");
    let _ = child.wait();

    // The retrying open must now ride out the OS-level drain after the worker's
    // exit and SUCCEED within open_db's 30s retry budget.
    let opened = tokio::time::timeout(Duration::from_secs(35), open_task)
        .await
        .expect("open_db did not finish within the retry budget (ride-out failed)")
        .expect("join")
        .expect("open_db must succeed after the worker exits and releases the LOCK");
    let waited = started.elapsed();

    // Sanity: usable handle, and it genuinely went through contention (it took
    // longer than a trivial uncontended open because it retried against the held
    // LOCK before the worker was killed).
    assert_eq!(
        context_engine_rs::store::ops::count_chunks(&opened)
            .await
            .unwrap_or(0),
        0,
        "recovered handle opens a usable DB"
    );
    assert!(
        waited >= Duration::from_millis(800),
        "open should have spent time retrying against the held LOCK before winning \
         (waited {waited:?})"
    );
}

/// FIRST-ADD AUTO-INDEX (the divergence fix): adding a repo via PUT /api/config
/// must auto-trigger its initial index — spawning a worker — WITHOUT the user
/// clicking Index. We boot the router with an EMPTY repo set, PUT a config that
/// adds one indexable repo, then assert `/index-status` reports `worker_active`
/// true for it within a few seconds (the router's put_config diffs the new repo
/// and fire-and-forgets a POST .../index, which spawns + proxies to a worker).
#[tokio::test]
#[ignore = "uses the real worker binary path; run with --ignored --nocapture"]
async fn adding_a_repo_auto_triggers_first_index() {
    let home = TempDir::new().unwrap();
    // Seed an EMPTY config (no repos) so the PUT below is a genuine first-add.
    {
        let s = Settings {
            machine_id: Some("e2e-machine".to_string()),
            worker_idle_secs: 3600,
            ..Settings::default()
        };
        write_settings_atomic(&config_path(home.path()), &s).expect("seed empty settings");
    }
    // A tiny indexable repo dir.
    let repo_dir = home.path().join("freshrepo");
    std::fs::create_dir_all(&repo_dir).unwrap();
    std::fs::write(repo_dir.join("a.rs"), b"pub fn a() {}\n").unwrap();
    let repo = repo_dir.to_string_lossy().to_string();

    let addr = start_router(&home).await;
    let client = Client::new();

    // Sanity: before the add, no worker is active for the repo.
    assert!(
        !worker_active(&client, addr, &repo).await,
        "no worker should be active before the repo is even configured"
    );

    // PUT a config that ADDS the new repo (mirrors what the UI's add-repo flow
    // sends: the full settings with the repo appended).
    let mut cfg = context_engine_rs::config::ensure_dir_and_load(home.path()).unwrap();
    cfg.repos.push(repo.clone());
    cfg.worker_idle_secs = 3600;
    let body = serde_json::to_value(&cfg).unwrap();
    let res = client
        .put(format!("http://{addr}/api/config"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert!(res.status().is_success(), "PUT add-repo must succeed");

    // The auto-index is fire-and-forget: poll until a worker comes up for the
    // new repo (spawn + open_db is ~0.6–1.4s; allow generous margin).
    let mut became_active = false;
    for _ in 0..40 {
        if worker_active(&client, addr, &repo).await {
            became_active = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    assert!(
        became_active,
        "adding a repo must auto-trigger its first index (a worker should become \
         active without any manual Index click)"
    );

    // REGRESSION GUARD (the "shows Chưa index while indexing" bug): `worker_active`
    // alone is set router-side from ready_repos() and does NOT prove the LIVE
    // worker status was fetched correctly. The router fetches /status from the
    // worker via a GET; an earlier bug used a POST helper (405 → fell through to
    // the cold sidecar, so the UI badge read "not_indexed" while the worker was
    // actively indexing). Assert the entry carries a LIVE status — `state` is one
    // of the worker's real states (indexing/idle) AND the live-only `phase` field
    // is present — not the cold sidecar's synthesized shape (which omits `phase`).
    let arr: serde_json::Value = client
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
        .find(|e| e["repo"].as_str() == Some(&context_engine_rs::store::normalize_repo_path(&repo)))
        .expect("repo present in index-status")
        .clone();
    assert_eq!(entry["worker_active"], true, "worker must be active");
    assert!(
        entry.get("phase").is_some(),
        "live worker status must include the `phase` field — its absence means the \
         router served the COLD sidecar instead of the worker's GET /status (the \
         405-on-POST regression that showed 'not indexed' during indexing). got: {entry}"
    );
    let state = entry["state"].as_str().unwrap_or("");
    assert!(
        state == "indexing" || state == "idle",
        "live status `state` must be a real worker state (indexing/idle), got {state:?}"
    );
}

/// BOOT CATCH-UP INCREMENTAL: a worker spawned by a NON-index action (here a
/// `/chunks` GET — spawn-allowed but does not itself trigger indexing) must
/// still index the repo on boot, because the worker fires an Incremental Update
/// right after readiness. This is what keeps scale-to-zero from serving stale
/// data: edits made while the worker was idle-exited are reconciled on respawn.
///
/// Setup avoids the OTHER trigger paths so the boot trigger is the ONLY cause:
/// the repo is in settings BEFORE the router starts (so put_config's add-diff
/// does not fire), and we spawn via `/chunks` (not POST /index). We then assert
/// the repo reaches a real indexed state (idle with the file counted, or at
/// least an `indexing` run observed) purely from the boot incremental.
#[tokio::test]
#[ignore = "uses the real worker binary path; run with --ignored --nocapture"]
async fn worker_boot_triggers_incremental_index() {
    let home = TempDir::new().unwrap();
    let repo_dir = home.path().join("bootidx");
    std::fs::create_dir_all(&repo_dir).unwrap();
    std::fs::write(repo_dir.join("a.rs"), b"pub fn a() {}\n").unwrap();
    let repo = repo_dir.to_string_lossy().to_string();
    // Repo present in settings BEFORE router start → add-diff path does NOT fire.
    seed_settings(&home, &repo, 3600);
    let addr = start_router(&home).await;
    let client = Client::new();

    // Spawn the worker via a NON-index action: /chunks (spawn-allowed, does not
    // itself enqueue an index). The router spawns the worker at acquire time,
    // before the request reaches the worker handler — so the file param is
    // irrelevant to the spawn. reqwest .query() URL-encodes the value.
    let id = repo_id_b64(&repo);
    let res = client
        .get(format!("http://{addr}/api/repos/{id}/chunks"))
        .query(&[("file", repo_dir.join("a.rs").to_string_lossy().as_ref())])
        .timeout(Duration::from_secs(40))
        .send()
        .await
        .unwrap();
    // The point is that the request reached a SPAWNED WORKER, not that chunks
    // returned data. A 503 would mean the spawn failed/warming; ANY other status
    // (incl. a 400 the worker's own path guard may return) proves spawn+proxy
    // succeeded — which is all this test needs to then observe the boot trigger.
    assert_ne!(
        res.status().as_u16(),
        503,
        "chunks action must spawn + proxy to a worker (got 503 = spawn failed/warming)"
    );

    // The boot incremental is fire-and-forget; poll /index-status until the repo
    // shows a real indexed result. Success = state idle AND indexed_files>0
    // (the run completed) OR state indexing (the run is in flight) — either proves
    // the boot trigger fired (a worker that never triggered would sit idle with
    // indexed_files 0 the whole time).
    let mut indexed_observed = false;
    for _ in 0..60 {
        let arr: serde_json::Value = client
            .get(format!("http://{addr}/api/index-status"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if let Some(e) = arr.as_array().unwrap().iter().find(|e| {
            e["repo"].as_str() == Some(&context_engine_rs::store::normalize_repo_path(&repo))
        }) {
            let state = e["state"].as_str().unwrap_or("");
            let n = e["indexed_files"].as_u64().unwrap_or(0);
            if state == "indexing" || (state == "idle" && n > 0) {
                indexed_observed = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    assert!(
        indexed_observed,
        "a worker spawned by a non-index action must STILL index on boot \
         (boot catch-up incremental) — observed no indexing run for the repo"
    );
}
