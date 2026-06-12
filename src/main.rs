use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use tokio::sync::RwLock;
use tracing::info;
use tracing_subscriber::EnvFilter;

use context_engine_rs::{
    cli as cli_client,
    cli::{ReadArgs, SearchArgs},
    config::{default_data_dir, default_embeddings_dir, ensure_dir_and_load},
    indexing::IndexEngine,
    server, store,
};

#[derive(Parser, Debug)]
#[command(name = "context-engine", about = "Context Engine settings server")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

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

    /// Embedding-cache root. The content-addressed cache (keyed by chunk text +
    /// model, NOT by repo) is concurrency-safe, so multiple instances can SHARE
    /// one cache and avoid re-embedding identical chunks — only RocksDB needs
    /// per-instance isolation.
    ///
    /// Boot precedence: this flag > env `CONTEXT_ENGINE_EMBEDDINGS_DIR` >
    /// `Settings.embeddings_dir` > `~/.vibervn/context-engine/embeddings`
    /// (anchored to home, so instances with different `--data-dir` share one
    /// cache by default).
    #[arg(long, env = "CONTEXT_ENGINE_EMBEDDINGS_DIR")]
    embeddings_dir: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Invoke codebase-retrieval against a running Context Engine server.
    Search {
        #[arg(long)]
        query: String,
        #[arg(long)]
        repo: String,
        #[arg(long)]
        server: Option<String>,
    },

    /// Invoke file-retrieval against a running Context Engine server.
    Read {
        #[arg(long)]
        repo: String,
        #[arg(long)]
        file: String,
        #[arg(long)]
        query: String,
        #[arg(long = "top-k")]
        top_k: Option<usize>,
        #[arg(long)]
        server: Option<String>,
    },
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
async fn main() -> ExitCode {
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

    if let Some(command) = cli.command {
        let exit_code = match command {
            Commands::Search {
                query,
                repo,
                server,
            } => {
                cli_client::run_search(SearchArgs {
                    query,
                    repo,
                    server,
                })
                .await
            }
            Commands::Read {
                repo,
                file,
                query,
                top_k,
                server,
            } => {
                cli_client::run_read(ReadArgs {
                    repo,
                    file,
                    query,
                    top_k,
                    server,
                })
                .await
            }
        };
        return exit_code;
    }

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
    let embeddings_dir = cli
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

    let addr: std::net::SocketAddr = format!("{bind}:{port}")
        .parse()
        .unwrap_or_else(|e| {
            eprintln!("error: invalid bind address '{bind}:{port}': {e}");
            std::process::exit(2);
        });

    let app = server::build_router(
        home_dir,
        data_dir,
        embeddings_dir,
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

    ExitCode::SUCCESS
}
