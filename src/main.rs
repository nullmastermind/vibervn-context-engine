use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use tokio::sync::RwLock;
use tracing::info;
use tracing_subscriber::EnvFilter;

use context_engine_rs::{
    config::{default_data_dir, ensure_dir_and_load},
    indexing::IndexEngine,
    server, store,
};

#[derive(Parser, Debug)]
#[command(name = "context-engine", about = "Context Engine settings server")]
struct Cli {
    /// Port to listen on [env: CONTEXT_ENGINE_PORT]
    #[arg(long, env = "CONTEXT_ENGINE_PORT")]
    port: Option<u16>,

    /// Bind address [env: CONTEXT_ENGINE_BIND]
    #[arg(long, env = "CONTEXT_ENGINE_BIND")]
    bind: Option<String>,

    /// Data directory base. RocksDB lives at `<data_dir>/rocksdb/`, embedding
    /// cache at `<data_dir>/embeddings/`. settings.json itself stays at
    /// `~/.vibervn/context-engine/settings.json` regardless of this value.
    ///
    /// Boot precedence: this flag > env `CONTEXT_ENGINE_DATA_DIR` >
    /// `Settings.data_dir` (in settings.json) > builtin default
    /// (`~/.vibervn/context-engine`).
    ///
    /// Use this to run multiple isolated instances simultaneously: RocksDB
    /// takes an exclusive per-directory lock, so two instances sharing one
    /// data dir will fail to open. Pointing each at its own dir avoids the
    /// collision.
    #[arg(long, env = "CONTEXT_ENGINE_DATA_DIR")]
    data_dir: Option<PathBuf>,
}

/// Pin bounded RocksDB memory settings unless the operator has overridden them.
///
/// SurrealDB's RocksDB layer reads these `SURREAL_ROCKSDB_*` env vars once (via
/// `LazyLock`) the first time a datastore opens, so they MUST be set before any
/// `Surreal::new::<RocksDb>` call. The defaults below are repo-count-stable:
/// - BLOCK_CACHE_SIZE: a single modest shared LRU cache (128 MiB) instead of the
///   RAM-derived ~31 GiB default. The cache is lazy (fills as blocks are read),
///   but the default ceiling is far too high for an always-on local server, and
///   the hot query path reads vectors from the in-memory shards, not RocksDB.
/// - WRITE_BUFFER_SIZE × MAX_WRITE_BUFFER_NUMBER: small per-DB write buffers
///   (32 MiB × 2 = 64 MiB/DB) instead of up to 1 GiB/DB. This is the dominant
///   per-repo term; pinning it keeps total RAM bounded as repo count grows.
/// - ENABLE_BLOB_FILES off-default blob sizes are left alone; embeddings live in
///   the vector shards, not re-read from RocksDB on the hot path.
fn set_rocksdb_memory_bounds() {
    // (var, default) — only applied when unset, so explicit overrides win.
    let bounds = [
        ("SURREAL_ROCKSDB_BLOCK_CACHE_SIZE", "134217728"), // 128 MiB shared LRU
        ("SURREAL_ROCKSDB_WRITE_BUFFER_SIZE", "33554432"), // 32 MiB per buffer
        ("SURREAL_ROCKSDB_MAX_WRITE_BUFFER_NUMBER", "2"),  // 2 buffers/DB → 64 MiB/DB
    ];
    for (key, default) in bounds {
        if std::env::var_os(key).is_none() {
            // SAFETY: set at the very top of main(), before any other thread is
            // spawned (tokio worker threads and the DB layer start after this).
            unsafe {
                std::env::set_var(key, default);
            }
        }
    }
}

#[tokio::main]
async fn main() {
    // Bound RocksDB memory BEFORE any datastore opens. SurrealDB derives its
    // RocksDB defaults from total system RAM and applies the write buffers PER
    // database: on a 64 GiB host the block cache defaults to ~31 GiB and each DB
    // gets up to 128 MiB × 8 = 1 GiB of write buffers. With one DB per repo this
    // reintroduces unbounded per-repo growth (axis 1). We pin small, repo-count-
    // stable values so total RocksDB RAM stays bounded no matter how many repos
    // are configured. Each var is only set if the operator has NOT overridden it,
    // so power users keep full control.
    set_rocksdb_memory_bounds();

    // Initialise tracing subscriber — reads RUST_LOG env var for filtering.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("context_engine_rs=info,warn")),
        )
        .init();

    let cli = Cli::parse();

    // Resolve port: CLI flag → env (handled by clap) → default 6699.
    let port = cli.port.unwrap_or(6699);

    // Resolve bind address: CLI flag → env (handled by clap) → default 127.0.0.1.
    let bind = cli.bind.as_deref().unwrap_or("127.0.0.1").to_owned();

    // Home-dir probe: exit early if we can't determine the home directory.
    let home_dir = match dirs::home_dir() {
        Some(h) => h,
        None => {
            eprintln!(
                "error: could not determine user home directory; \
                 set HOME (Unix) or USERPROFILE (Windows)"
            );
            std::process::exit(2);
        }
    };

    // Load settings (needed to know which repos to watch).
    let settings = match ensure_dir_and_load(&home_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not load settings: {e}");
            std::process::exit(2);
        }
    };

    // Resolve data_dir with the documented precedence:
    //   CLI flag > env CONTEXT_ENGINE_DATA_DIR > Settings.data_dir > builtin default.
    // (clap collapses CLI > env into `cli.data_dir`.) The resolved value is
    // captured ONCE here and threaded into IndexEngine + AppState; it is never
    // re-read from `Settings` at runtime, so a PUT /api/config that changes
    // data_dir mid-run does NOT close existing RocksDB handles or evict warmed
    // shards — that change takes effect on the next launch.
    let data_dir = cli
        .data_dir
        .clone()
        .or_else(|| settings.data_dir.clone())
        .unwrap_or_else(|| default_data_dir(&home_dir));

    if let Err(e) = std::fs::create_dir_all(&data_dir) {
        eprintln!(
            "error: could not create data directory {}: {e}",
            data_dir.display()
        );
        std::process::exit(2);
    }
    info!(data_dir = %data_dir.display(), "data_dir resolved");

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

    // Start IndexEngine — spawns watchers for all configured repos.
    // It shares `repo_dbs` so indexer writes land in the handles the server reads.
    // It receives the shared settings handle so the consumer picks up post-boot changes.
    let index_engine = IndexEngine::start(
        data_dir.clone(),
        &boot_settings,
        repo_dbs.clone(),
        settings_handle.clone(),
    )
    .await;
    info!("IndexEngine started ({} repos)", repo_count);

    let addr: std::net::SocketAddr = format!("{bind}:{port}")
        .parse()
        .unwrap_or_else(|e| {
            eprintln!("error: invalid bind address '{bind}:{port}': {e}");
            std::process::exit(2);
        });

    let app = server::build_router(
        home_dir,
        data_dir,
        index_engine,
        repo_dbs,
        settings_handle,
        &bind,
    );

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| {
            eprintln!("error: could not bind to {addr}: {e}");
            std::process::exit(2);
        });

    info!("Context Engine listening on http://{addr}");
    axum::serve(listener, app).await.unwrap_or_else(|e| {
        eprintln!("server error: {e}");
        std::process::exit(1);
    });
}
