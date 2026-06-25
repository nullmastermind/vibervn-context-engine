//! MEASURE-gate (process-per-project plan, task #1 — DATA DEPENDENCY, not a phase).
//!
//! Before the router's readiness-wait budget can be written, we must know the
//! real cold `open_db` cost at kernel scale vs a small repo. The router owns
//! "window A" (spawn → worker accepts), and that window is dominated by
//! `boot_engine` → `store::open_db` (which itself carries a 30s LOCK-drain retry
//! loop + `ensure_secondary_indexes`). A guessed budget would reintroduce the
//! exact connection-refused/hang failure mode we hardened against, so this is
//! measured first.
//!
//! These tests are `#[ignore]` because they depend on machine-local on-disk
//! indexes under `~/.vibervn/context-engine/rocksdb/`. Run explicitly:
//!
//! ```text
//! cargo test --test measure_cold_open -- --ignored --nocapture
//! ```
//!
//! Each case measures a COLD open: a fresh process has no warm RocksDB handle
//! and the OS page cache for the index is cold-ish after idle, which approximates
//! the post-idle-exit respawn path the plan must budget for. `open_db` is the
//! dominant unknown; warm (mmap shard) is already near-zero per `shard_file.rs`.

use std::path::PathBuf;
use std::time::Instant;

use context_engine_rs::{engine_boot, store};

/// Resolve `~/.vibervn/context-engine` the same way the engine does at boot.
fn data_dir() -> PathBuf {
    dirs::home_dir()
        .expect("home dir")
        .join(".vibervn")
        .join("context-engine")
}

/// Time a single cold `open_db` for `repo_path` at `generation`, returning the
/// elapsed milliseconds. Pins the production RocksDB memory bounds first so the
/// measurement reflects the real worker boot path (not RAM-derived defaults).
async fn time_open(repo_path: &str, generation: u32) -> u128 {
    // Mirror `boot_engine`: bound RocksDB memory before any datastore opens.
    engine_boot::set_rocksdb_memory_bounds();

    let dir = data_dir();
    let started = Instant::now();
    let db = store::open_db(&dir, repo_path, generation)
        .await
        .expect("open_db should succeed against an existing on-disk index");
    let elapsed = started.elapsed().as_millis();

    // Drop the handle so the LOCK releases before the process exits (mirrors the
    // worker idle-exit contract). Explicit drop for intent.
    drop(db);
    elapsed
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "machine-local: depends on on-disk kernel index; run with --ignored --nocapture"]
async fn measure_kernel_scale_cold_open() {
    // Linux kernel index, ~6.8 GB on disk, generation 26.
    let repo = r"c:\users\0x317\downloads\linux";
    let ms = time_open(repo, 26).await;
    println!("MEASURE kernel-scale cold open_db: {ms} ms (repo={repo}, gen=26)");
    // No hard assertion on absolute time — this run EXISTS to produce the number
    // that sets `mcp_index_wait_secs` / the router window-A budget. We only guard
    // that it returned within the existing 30s LOCK-drain ceiling of open_db; if
    // it ever exceeds that, the open itself failed its own retry budget.
    assert!(
        ms < 60_000,
        "kernel cold open exceeded 60s ({ms} ms) — budget assumptions invalid"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "machine-local: depends on on-disk small repo index; run with --ignored --nocapture"]
async fn measure_small_repo_cold_open() {
    // A small repo for the low-end of the budget range. Must be a repo NOT held
    // by a live server (seed_statuses_from_db opens every settings.repo at boot
    // and holds the exclusive LOCK), else open_db races that handle. `prompts`
    // (~2.7 MB, gen 0) is not in the default settings.repos set.
    let repo = r"d:\projects\prompts";
    let ms = time_open(repo, 0).await;
    println!("MEASURE small-repo cold open_db: {ms} ms (repo={repo}, gen=0)");
    assert!(
        ms < 60_000,
        "small cold open exceeded 60s ({ms} ms) — unexpected"
    );
}
