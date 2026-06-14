//! Shared engine boot sequence.
//!
//! WHY this module exists: both the always-on HTTP server (`main.rs`) and the
//! self-contained `bench-query` CLI must boot the engine through the EXACT same
//! steps so they can never drift (same settings load, same data_dir/embeddings_dir
//! precedence, same RocksDB memory bounds, same stale-generation sweep, same
//! `IndexEngine::start`). Any divergence between the two boot paths would mean the
//! CLI exercises a different engine than production — defeating its purpose as a
//! real-index bench/query tool. So the entire boot block lives here ONCE.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::RwLock;
use tracing::info;

use crate::config::{
    default_data_dir, default_embeddings_dir, ensure_dir_and_load, ensure_machine_id, Settings,
};
use crate::indexing::IndexEngine;
use crate::store;

/// All the long-lived shared handles produced by booting the engine. Threaded
/// into `server::build_router` (HTTP server) or used directly by the CLI.
pub struct BootedEngine {
    /// Resolved home directory. Used ONLY for `settings.json` access (its
    /// location is fixed at `~/.vibervn/context-engine/settings.json`).
    pub home_dir: PathBuf,
    /// Boot-resolved data directory (CLI > env > `Settings.data_dir` > builtin
    /// default). Captured once; never re-read from `Settings` at runtime.
    pub data_dir: PathBuf,
    /// Boot-resolved embedding-cache root.
    pub embeddings_dir: PathBuf,
    /// Shared index engine — owns watchers, status, the sharded vector index.
    pub index_engine: Arc<IndexEngine>,
    /// Per-repo SurrealDB handles, keyed by repo path. Starts empty.
    pub repo_dbs: store::RepoDbMap,
    /// Shared live settings — the single source of truth.
    pub settings: Arc<RwLock<Settings>>,
}

/// CLI-flag overrides for the two boot-frozen directories. `None` means "fall
/// back to the documented precedence (env > settings > builtin default)".
#[derive(Default)]
pub struct BootOptions {
    /// `--data-dir` flag override (clap collapses CLI > env into this).
    pub data_dir: Option<PathBuf>,
    /// `--embeddings-dir` flag override.
    pub embeddings_dir: Option<PathBuf>,
}

/// Pin bounded RocksDB memory settings unless the operator has overridden them.
///
/// SurrealDB's RocksDB layer reads these `SURREAL_ROCKSDB_*` env vars once (via
/// `LazyLock`) the first time a datastore opens, so they MUST be set before any
/// `Surreal::new::<RocksDb>` call. `boot_engine` calls this at its very top,
/// before `IndexEngine::start` (and well before any per-repo DB open), so the
/// guarantee holds for every binary that boots through here. The defaults below
/// are repo-count-stable:
/// - BLOCK_CACHE_SIZE: a single modest shared LRU cache (128 MiB) instead of the
///   RAM-derived ~31 GiB default. The cache is lazy (fills as blocks are read),
///   but the default ceiling is far too high for an always-on local server, and
///   the hot query path reads vectors from the in-memory shards, not RocksDB.
/// - WRITE_BUFFER_SIZE × MAX_WRITE_BUFFER_NUMBER: small per-DB write buffers
///   (32 MiB × 2 = 64 MiB/DB) instead of up to 1 GiB/DB. This is the dominant
///   per-repo term; pinning it keeps total RAM bounded as repo count grows.
/// - ENABLE_BLOB_FILES off-default blob sizes are left alone; embeddings live in
///   the vector shards, not re-read from RocksDB on the hot path.
pub fn set_rocksdb_memory_bounds() {
    // (var, default) — only applied when unset, so explicit overrides win.
    let bounds = [
        ("SURREAL_ROCKSDB_BLOCK_CACHE_SIZE", "134217728"), // 128 MiB shared LRU
        ("SURREAL_ROCKSDB_WRITE_BUFFER_SIZE", "33554432"), // 32 MiB per buffer
        ("SURREAL_ROCKSDB_MAX_WRITE_BUFFER_NUMBER", "2"),  // 2 buffers/DB → 64 MiB/DB
    ];
    for (key, default) in bounds {
        if std::env::var_os(key).is_none() {
            // SAFETY: set at the very top of boot_engine, before the datastore
            // layer opens any RocksDB handle (IndexEngine::start and the lazy
            // per-repo opens all run after this). The vars are read once by the
            // RocksDB layer on first open.
            unsafe {
                std::env::set_var(key, default);
            }
        }
    }
}

/// Boot the engine: bound RocksDB memory, load + migrate settings, resolve the
/// boot-frozen directories, sweep stale generations, and start `IndexEngine`.
///
/// Returns the shared handles. Does NOT init tracing (each binary owns its own
/// `EnvFilter` default) and does NOT build the HTTP router or bind a socket —
/// the caller decides what to do with the booted engine.
pub async fn boot_engine(opts: BootOptions) -> Result<BootedEngine> {
    // Bound RocksDB memory BEFORE any datastore opens. SurrealDB derives its
    // RocksDB defaults from total system RAM and applies the write buffers PER
    // database: on a 64 GiB host the block cache defaults to ~31 GiB and each DB
    // gets up to 128 MiB × 8 = 1 GiB of write buffers. With one DB per repo this
    // reintroduces unbounded per-repo growth (axis 1). We pin small, repo-count-
    // stable values so total RocksDB RAM stays bounded no matter how many repos
    // are configured. Each var is only set if the operator has NOT overridden it,
    // so power users keep full control.
    set_rocksdb_memory_bounds();

    // Home-dir probe: fail early if we can't determine the home directory.
    let home_dir = dirs::home_dir().context(
        "could not determine user home directory; set HOME (Unix) or USERPROFILE (Windows)",
    )?;

    // Load settings (needed to know which repos to watch).
    let mut settings = ensure_dir_and_load(&home_dir).context("could not load settings")?;

    // Populate + persist `machine_id` if missing. First boot (or upgrade from a
    // settings file written before this field existed) computes the seed and
    // writes it back; subsequent boots are a no-op. Done HERE — before
    // settings_handle is built — so every downstream reader sees `Some`.
    ensure_machine_id(&home_dir, &mut settings).context("could not initialize machine_id")?;

    // Resolve data_dir with the documented precedence:
    //   CLI flag > env CONTEXT_ENGINE_DATA_DIR > Settings.data_dir > builtin default.
    // (clap collapses CLI > env into `opts.data_dir`.) The resolved value is
    // captured ONCE here and threaded into IndexEngine + AppState; it is never
    // re-read from `Settings` at runtime, so a PUT /api/config that changes
    // data_dir mid-run does NOT close existing RocksDB handles or evict warmed
    // shards — that change takes effect on the next launch.
    let data_dir = opts
        .data_dir
        .clone()
        .or_else(|| settings.data_dir.clone())
        .unwrap_or_else(|| default_data_dir(&home_dir));

    std::fs::create_dir_all(&data_dir)
        .with_context(|| format!("could not create data directory {}", data_dir.display()))?;
    info!(data_dir = %data_dir.display(), "data_dir resolved");

    // Resolve embeddings_dir with its own precedence:
    //   CLI flag > env CONTEXT_ENGINE_EMBEDDINGS_DIR > Settings.embeddings_dir
    //   > ~/.vibervn/context-engine/embeddings (anchored to HOME, not data_dir).
    // Anchoring the default to home — not the resolved data_dir — means multiple
    // instances launched with different --data-dir values share ONE cache by
    // default: the content-addressed cache is concurrency-safe, so sharing it
    // avoids re-embedding identical chunks (only RocksDB needs per-instance
    // isolation). A pure default install still lands at
    // ~/.vibervn/context-engine/embeddings, byte-identical to before.
    // Also boot-frozen. We do NOT fail-fast on a create_dir_all error here: the
    // cache degrades gracefully (EmbeddingCache::new returns None and the
    // pipeline runs without a cache), unlike RocksDB which must open — so a
    // non-writable cache root should not block startup.
    let embeddings_dir = opts
        .embeddings_dir
        .clone()
        .or_else(|| settings.embeddings_dir.clone())
        .unwrap_or_else(|| default_embeddings_dir(&home_dir));
    if let Err(e) = std::fs::create_dir_all(&embeddings_dir) {
        // Non-fatal: log and continue. The cache will retry create on first use
        // and disable itself if it still cannot create the directory.
        tracing::warn!(
            embeddings_dir = %embeddings_dir.display(),
            error = %e,
            "could not pre-create embeddings dir; cache will retry lazily / degrade"
        );
    }
    info!(embeddings_dir = %embeddings_dir.display(), "embeddings_dir resolved");

    // Wrap the loaded settings in a shared live handle so IndexEngine and the
    // HTTP server share a single source of truth that mutates on every PUT /api/config.
    let settings_handle = Arc::new(RwLock::new(settings));

    // Shared per-repo DB map — starts empty; `get_or_open` populates lazily.
    // The eager open loop has been removed: SurrealDB handles are expensive to
    // open and the consumer + query paths already cache them via `get_or_open`.
    let repo_dbs: store::RepoDbMap = Arc::new(RwLock::new(std::collections::HashMap::new()));

    // Take one owned boot snapshot — the read guard drops at the end of this
    // statement, so it is NOT held across the IndexEngine::start(...).await below.
    let boot_settings = settings_handle.read().await.clone();
    let repo_count = boot_settings.repos.len();

    // Reclaim stale per-repo index generations BEFORE any RocksDB handle opens.
    // Each repo/index delete bumps a repo's generation and moves the next index to
    // a fresh directory; when the old directory's removal previously failed (Windows
    // held the LOCK past the retry budget) it was left on disk. This is the one
    // moment guaranteed lock-free — no handle in `repo_dbs`, no warmed shard — so a
    // blocking removal here can't race a live datastore. Runs synchronously before
    // IndexEngine::start so the first index never collides with a leftover. Skips
    // (doesn't error) any directory it still can't remove; the next boot retries.
    store::sweep_stale_generations(&data_dir, &boot_settings.repos, &boot_settings.repo_generations);

    // Start IndexEngine — spawns watchers for all configured repos.
    // It shares `repo_dbs` so indexer writes land in the handles the server reads.
    // It receives the shared settings handle so the consumer picks up post-boot changes.
    let index_engine = IndexEngine::start(
        data_dir.clone(),
        embeddings_dir.clone(),
        &boot_settings,
        repo_dbs.clone(),
        settings_handle.clone(),
    )
    .await;
    info!("IndexEngine started ({} repos)", repo_count);

    Ok(BootedEngine {
        home_dir,
        data_dir,
        embeddings_dir,
        index_engine,
        repo_dbs,
        settings: settings_handle,
    })
}
