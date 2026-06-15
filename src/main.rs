use std::path::PathBuf;

use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use context_engine_rs::engine_boot::{boot_engine, BootOptions, BootedEngine};
use context_engine_rs::server;

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

#[tokio::main]
async fn main() {
    // Initialise tracing subscriber — reads RUST_LOG env var for filtering.
    // Stays in each binary's main() (not in boot_engine) so each bin owns its
    // own EnvFilter default and tracing is never initialised twice.
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

    // Boot the engine through the SHARED path so the server and the bench-query
    // CLI never drift (RocksDB memory bounds, settings load + machine_id,
    // data_dir/embeddings_dir precedence, stale-generation sweep, IndexEngine).
    // On failure we map to the same `eprintln! + exit(2)` behavior as before.
    let BootedEngine {
        home_dir,
        data_dir,
        embeddings_dir,
        index_engine,
        repo_dbs,
        settings,
    } = match boot_engine(BootOptions {
        data_dir: cli.data_dir.clone(),
        embeddings_dir: cli.embeddings_dir.clone(),
        // Server keeps watchers ON — auto-reindex on edits is the whole point.
        no_watchers: false,
    })
    .await
    {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(2);
        }
    };

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
        settings,
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
