use std::path::PathBuf;

use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use context_engine_rs::engine_boot::{BootOptions, BootedEngine, boot_engine};
use context_engine_rs::router;
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

    /// PROCESS-PER-PROJECT WORKER MODE. When set, this process serves exactly
    /// ONE repo instead of acting as the router. Spawned by the router on demand
    /// (`context-engine --worker <repo> --port 0`); not normally launched by a
    /// user. The worker boots a single-repo `IndexEngine`, binds an ephemeral
    /// port (use `--port 0`), and prints a one-line readiness handshake to stdout
    /// AFTER the listener is bound (so the router never proxies to a port that
    /// would refuse the connection), then serves the same HTTP surface the router
    /// reverse-proxies into. It self-exits after an idle window with no requests.
    #[arg(long, value_name = "REPO")]
    worker: Option<String>,

    /// Idle window (seconds) after which a `--worker` process self-exits to
    /// release its RocksDB handle, watcher, and resident shard. Ignored outside
    /// worker mode. When unset, falls back to `settings.worker_idle_secs`
    /// (default 300s). The router respawns on the next request. Lower = tighter
    /// resource reclaim, more cold starts.
    #[arg(long, env = "CONTEXT_ENGINE_WORKER_IDLE_SECS")]
    worker_idle_secs: Option<u64>,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Initialise tracing subscriber — reads RUST_LOG env var for filtering.
    // Stays in each binary's main() (not in boot_engine) so each bin owns its
    // own EnvFilter default and tracing is never initialised twice.
    //
    // CRITICAL (worker mode): a worker's STDOUT carries exactly ONE machine-read
    // line — the readiness handshake (`run_worker`). The router pipes that stdout,
    // reads the handshake, then DROPS the read end. If tracing also wrote to
    // stdout, every worker log after the handshake would hit a closed pipe
    // ("failed to write ... os error 232 / broken pipe"). So a worker logs to
    // STDERR (which the router inherits and forwards to its own console),
    // leaving stdout for the handshake alone. The router itself keeps stdout.
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("context_engine_rs=info,warn"));
    if cli.worker.is_some() {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_writer(std::io::stderr)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    }

    // Resolve bind address: CLI flag → env (handled by clap) → default 127.0.0.1.
    let bind = cli.bind.as_deref().unwrap_or("127.0.0.1").to_owned();

    // ── Mode dispatch ──────────────────────────────────────────────────────
    // Worker mode: serve one repo + readiness handshake + idle self-exit.
    // Router mode (default): spawn/proxy/manage per-repo workers.
    match cli.worker.clone() {
        Some(repo) => run_worker(&cli, &bind, repo).await,
        None => run_router(&cli, &bind).await,
    }
}

/// PROCESS-PER-PROJECT WORKER: boot a single-repo engine, bind an ephemeral
/// port, emit the readiness handshake to stdout, then serve until idle.
async fn run_worker(cli: &Cli, bind: &str, repo: String) {
    // Worker requested port: `--port 0` (router default) yields an OS-assigned
    // ephemeral port, reported back via the handshake. An explicit port is
    // honored (useful for manual debugging).
    let requested_port = cli.port.unwrap_or(0);

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
        // Worker keeps its single repo's watcher ON while alive: edits during a
        // live session still auto-reindex. Scale-to-zero comes from idle-exit
        // tearing the whole process (and thus the watcher) down, not from
        // suppressing the watcher.
        no_watchers: false,
        only_repo: Some(repo.clone()),
    })
    .await
    {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(2);
        }
    };

    let addr: std::net::SocketAddr =
        format!("{bind}:{requested_port}")
            .parse()
            .unwrap_or_else(|e| {
                eprintln!("error: invalid bind address '{bind}:{requested_port}': {e}");
                std::process::exit(2);
            });

    // Idle-exit tracker: the router proxies real requests through this worker's
    // HTTP surface; an access-time middleware bumps the tracker, and a watchdog
    // task self-exits after the idle window of no requests. The shutdown
    // sequence (drain → drop+flush DB → exit) runs inside the tracker so the
    // RocksDB LOCK is released before the process dies (avoids a respawn race).
    //
    // Idle window precedence: CLI flag / env override > persisted
    // `settings.worker_idle_secs` > default (300s). CLI/env is for launch-time
    // tuning (the router passes it per-spawn); settings is the durable default.
    let idle_secs = match cli.worker_idle_secs {
        Some(v) => v,
        None => settings.read().await.worker_idle_secs,
    };
    let idle = router::worker::IdleTracker::new(std::time::Duration::from_secs(idle_secs));

    let app = server::build_router(
        home_dir.clone(),
        data_dir,
        embeddings_dir,
        index_engine.clone(),
        repo_dbs.clone(),
        settings.clone(),
        bind,
    );
    // Wrap with the access-time middleware so every proxied request resets idle.
    let app = router::worker::with_idle_tracking(app, idle.clone());
    // Wrap with mtime-gated config reload so a router PUT /api/config (rotated
    // Voyage key, changed rerank model) reaches THIS live worker on its next
    // request — covering both the index and query paths — without a restart.
    let app = router::worker::with_config_reload(
        app,
        router::worker::ConfigReloader::new(home_dir, settings),
    );

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| {
            eprintln!("error: could not bind to {addr}: {e}");
            std::process::exit(2);
        });

    // Resolve the ACTUAL bound port (ephemeral when requested_port == 0).
    let actual = listener.local_addr().unwrap_or_else(|e| {
        eprintln!("error: could not read local_addr: {e}");
        std::process::exit(2);
    });

    // ── Readiness handshake ──────────────────────────────────────────────
    // Emitted AFTER bind succeeds: the port is now connectable (connects queue
    // in the listen backlog rather than being refused), so the router may begin
    // proxying the instant it reads this line. The router parses stdout for this
    // exact prefix; keep it stable and machine-readable.
    println!(
        "{} port={} pid={} repo={}",
        router::worker::READY_PREFIX,
        actual.port(),
        std::process::id(),
        repo
    );
    // Flush so the router sees the line immediately (stdout is line-buffered to a
    // pipe, but be explicit — a delayed flush would stall the router's window-A).
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    info!(repo = %repo, addr = %actual, "worker ready");

    // CATCH-UP INCREMENTAL on boot: a worker is spawned on demand and idle-exits
    // under scale-to-zero, so the filesystem watcher (which only runs while the
    // worker is alive) MISSES any edits made while the worker was down. Kick one
    // Incremental Update right after readiness so a freshly-(re)spawned worker
    // reconciles its index with the current on-disk state. Fire-and-forget on
    // purpose: `trigger_index` only enqueues onto the consumer's mpsc channel and
    // returns immediately (the actual scan runs async + serialized per-repo), so
    // this does NOT delay the handshake/serving. Benign if the spawning action
    // was itself an index/rebuild — the per-repo consumer serializes the runs and
    // a no-change incremental is a cheap disk scan. `changes:None, rebuild:false`
    // == the "Incremental Update" button (blast-radius-gated re-resolve).
    if let Err(e) = index_engine.trigger_index(&repo).await {
        // Non-fatal: the watcher + any explicit action still trigger indexing.
        info!(repo = %repo, error = %e, "boot catch-up incremental trigger failed (non-fatal)");
    }

    // Spawn the idle watchdog: when the idle window elapses with no requests, it
    // runs the graceful shutdown (drop the index engine's DB handles so the LOCK
    // releases) and exits the process. The router will respawn on the next
    // request.
    router::worker::spawn_idle_watchdog(idle, index_engine, repo_dbs, repo.clone());

    axum::serve(listener, app).await.unwrap_or_else(|e| {
        eprintln!("worker server error: {e}");
        std::process::exit(1);
    });
}

/// ROUTER (default mode): the lightweight front-end. Holds no per-repo index;
/// serves global endpoints + the UI directly and reverse-proxies per-repo
/// requests to on-demand worker processes (spawning + reaping them).
async fn run_router(cli: &Cli, bind: &str) {
    let port = cli.port.unwrap_or(6699);

    let addr: std::net::SocketAddr = format!("{bind}:{port}").parse().unwrap_or_else(|e| {
        eprintln!("error: invalid bind address '{bind}:{port}': {e}");
        std::process::exit(2);
    });

    // The router still needs settings (for the UI, config endpoints, repo list)
    // and the embedding/data dirs, but it does NOT start an IndexEngine or open
    // any repo DB — that is the whole point. `boot_router` loads settings and
    // resolves dirs only.
    let app = match router::build_router_app(router::RouterBootOptions {
        data_dir: cli.data_dir.clone(),
        embeddings_dir: cli.embeddings_dir.clone(),
        bind: bind.to_owned(),
        home_dir: None,
        worker_exe: None,
    })
    .await
    {
        Ok(app) => app,
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(2);
        }
    };

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| {
            eprintln!("error: could not bind to {addr}: {e}");
            std::process::exit(2);
        });

    info!("Context Engine router listening on http://{addr}");
    axum::serve(listener, app).await.unwrap_or_else(|e| {
        eprintln!("router server error: {e}");
        std::process::exit(1);
    });
}
