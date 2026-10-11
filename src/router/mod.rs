//! ROUTER mode: the lightweight front-end of the process-per-project design.
//!
//! The router holds NO per-repo index — no `IndexEngine`, no open RocksDB
//! handle, no resident vector shard. Its job is:
//! - serve the UI (`/`) and global config endpoints (`/api/config`) natively,
//! - render the repo list + cold `/index-stats` / `/graph` from per-repo
//!   [`sidecar`] files (no DB open, no worker spawn),
//! - reverse-proxy every per-repo operation (query, index, MCP, chunks, chat,
//!   SSE events) to an on-demand worker process via [`proxy`], spawning and
//!   reaping workers through the [`registry`] + [`spawn`] + [`jobobject`].
//!
//! This is what bounds resident memory to the set of repos ACTIVE within the
//! worker idle window, instead of every configured repo.

pub mod jobobject;
pub mod mcp_proxy;
pub mod plan;
pub mod proxy;
pub mod registry;
pub mod sidecar;
pub mod spawn;
pub mod worker;
mod worker_args;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::Json;
use axum::extract::{Path as AxumPath, Query as AxumQuery, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{any, delete, get, post};
use axum::{Router, extract::Request};
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use serde_json::json;

use crate::config::{
    Settings, default_data_dir, default_embeddings_dir, ensure_dir_and_load, ensure_machine_id,
};
use crate::engine_boot::set_rocksdb_memory_bounds;
use crate::mcp_session_store::{BoundedSessionStore, mcp_sessions_dir};
use crate::store::normalize_repo_path;

use self::jobobject::JobObject;
use self::proxy::{ProxyCtx, acquire_and_proxy};

/// Boot inputs for router mode (mirrors the CLI flags the router accepts).
pub struct RouterBootOptions {
    pub data_dir: Option<PathBuf>,
    pub embeddings_dir: Option<PathBuf>,
    pub bind: String,
    /// Override for the home directory (settings.json location). `None` uses the
    /// real `dirs::home_dir()` (production). Tests inject a TempDir so the router
    /// reads a hermetic settings.json instead of the developer's real config —
    /// the same explicit-home-dir pattern `build_router` / `boot_engine` use.
    pub home_dir: Option<PathBuf>,
    /// Override for the worker executable path. `None` uses
    /// `std::env::current_exe()` (production: the router IS the same binary it
    /// spawns as `--worker`). Tests inject `CARGO_BIN_EXE_context-engine-rs`
    /// because the test harness's own `current_exe()` is the test binary, not
    /// the context-engine binary.
    pub worker_exe: Option<PathBuf>,
    /// The port the router's listener is bound to, surfaced via
    /// `GET /api/server-info` so the UI can build MCP endpoint URLs from the real
    /// listen port instead of `window.location` (which is wrong behind an iframe,
    /// reverse proxy, or port-forward). `None` when the caller binds an ephemeral
    /// port only after building the app (tests).
    pub listen_port: Option<u16>,
}

/// Router-mode shared state. Deliberately small: the proxy context (client +
/// worker registry + spawn inputs), the resolved data dir (for sidecar reads),
/// and the home dir (for settings.json).
#[derive(Clone)]
pub struct RouterState {
    pub home_dir: PathBuf,
    pub data_dir: PathBuf,
    /// Embedding-cache root — for the native `/api/embedding-cache` purge (a
    /// host-level op on the shared content-addressed cache; no repo/worker
    /// involved).
    pub embeddings_dir: PathBuf,
    /// See [`RouterBootOptions::listen_port`].
    pub listen_port: Option<u16>,
    pub proxy: ProxyCtx,
}

/// Build the router-mode axum app: load settings + resolve dirs (NO IndexEngine,
/// NO repo DB opens), create the Job Object, and wire global + proxy routes.
///
/// Returns the wired `Router` plus a clone of the [`ProxyCtx`] so the caller's
/// shutdown handler can reach the worker [`registry::Registry`] (to kill live
/// workers on Ctrl+C). The router's own `RouterState` keeps the original; the
/// returned clone shares the same `Arc`-backed registry + Job Object.
pub async fn build_router_app(opts: RouterBootOptions) -> Result<(Router, ProxyCtx)> {
    // The router does not open RocksDB, but a worker it spawns will — and the
    // worker reads these same env-derived bounds at its own boot. Setting them
    // here is harmless and keeps parity if the router is ever extended.
    set_rocksdb_memory_bounds();

    let home_dir = match opts.home_dir.clone() {
        Some(h) => h,
        None => dirs::home_dir().context("could not determine home directory")?,
    };
    let mut settings = ensure_dir_and_load(&home_dir).context("could not load settings")?;

    // Populate + persist `machine_id` if missing — the same first-boot seeding
    // `boot_engine` does, but the ROUTER must do it too: in process-per-project
    // mode the router (not a worker) is the always-on process that serves the UI
    // and the `/api/plan/*` checkout/free-trial routes, and those read
    // `machine_id` fresh from settings.json (per-machine dedup). A worker only
    // boots on demand, so a fresh machine that has never indexed a repo would
    // have an empty `machine_id` and every checkout would 500 with
    // "machine_id not initialized" until some worker happened to boot.
    //
    // NON-FATAL on purpose: unlike `boot_engine` (which `?`-aborts), a failure to
    // compute/persist the id here must NOT take down the router — that would kill
    // the whole UI for an issue that only degrades one feature. The checkout path
    // already degrades gracefully (it surfaces the same 500 the user would see
    // anyway), so we log and keep serving rather than failing boot harder than
    // the feature it gates.
    if let Err(e) = ensure_machine_id(&home_dir, &mut settings) {
        tracing::warn!(
            error = %e,
            "could not initialize machine_id at router boot; checkout/free-trial will report it \
             uninitialized until it can be persisted, but the UI stays up"
        );
    }

    let data_dir = opts
        .data_dir
        .clone()
        .or_else(|| settings.data_dir.clone())
        .unwrap_or_else(|| default_data_dir(&home_dir));
    let embeddings_dir = opts
        .embeddings_dir
        .clone()
        .or_else(|| settings.embeddings_dir.clone())
        .unwrap_or_else(|| default_embeddings_dir(&home_dir));

    // The Job Object guarantees workers die if the router dies (kill-on-close).
    let job = Arc::new(JobObject::new().unwrap_or_else(|| {
        // new() already logged; this branch is unreachable on the no-op shim and
        // the Windows impl returns None only after logging. Build a shim-equivalent
        // by retrying (the non-windows shim always returns Some).
        JobObject::new().expect("job object shim")
    }));

    // Args every worker inherits so it resolves the SAME dirs + bind as the
    // router. Port is added per-spawn (`--port 0`).
    let exe = match opts.worker_exe.clone() {
        Some(e) => e,
        None => std::env::current_exe().context("resolve current exe for worker spawn")?,
    };
    let worker_args = worker_args::build(&opts, &home_dir);

    let state = RouterState {
        home_dir,
        data_dir: data_dir.clone(),
        embeddings_dir,
        listen_port: opts.listen_port,
        proxy: ProxyCtx::new(job, exe, worker_args),
    };

    // Global `/mcp` PROXYING service. Repo-addressed per call: each tool call's
    // `workspace_full_path` selects the worker the call is forwarded to (see
    // `mcp_proxy::ProxyMcpHandler`). Built with the same rmcp StreamableHttp
    // plumbing the monolith used, so MCP clients connecting to `/mcp` see an
    // identical surface. Enabled-tool gating reads the same setting.
    let mcp_proxy_ctx = state.proxy.clone();
    let enabled_tools = settings.enabled_mcp_tools.clone();
    let bind_host = opts.bind.clone();
    let mcp_session_store = Arc::new(BoundedSessionStore::with_persist(mcp_sessions_dir(
        &data_dir,
    )));
    let mcp_config = {
        let is_loopback = matches!(bind_host.as_str(), "127.0.0.1" | "localhost" | "::1");
        let mut base = if is_loopback {
            StreamableHttpServerConfig::default()
        } else {
            StreamableHttpServerConfig::default().with_allowed_hosts(vec![
                bind_host.clone(),
                "localhost".to_string(),
                "127.0.0.1".to_string(),
                "::1".to_string(),
            ])
        };
        // Same restore path as the worker: idle-drop the live worker, keep the
        // store entry (now on disk too) so the next stale request restores
        // instead of 404-ing. Without this, router `/mcp` is store-less and
        // Claude Code sees "session expired" after rmcp's 5 min keep_alive.
        base.session_store = Some(mcp_session_store);
        base
    };
    let mcp_service = StreamableHttpService::new(
        move || {
            Ok(mcp_proxy::ProxyMcpHandler::new(
                mcp_proxy_ctx.clone(),
                &enabled_tools,
            ))
        },
        Arc::new(LocalSessionManager::default()),
        mcp_config,
    );

    // One-time migration backfill: write sidecars for any already-indexed repo
    // that lacks one (pre-router monolith indexes, or indexes built before
    // sidecars existed). Runs ONCE in the background at boot — it opens each
    // sidecar-less repo's DB exactly once to read the real file count, writes the
    // sidecar only when count>0 (so a phantom/empty dir never becomes a false
    // "indexed"), then drops the handle. Best-effort + lock-safe: a repo whose
    // worker is already mid-open is skipped (that worker writes its own sidecar
    // at boot). After the first boot every real repo has a sidecar, so this is a
    // no-op thereafter. Backgrounded so it never delays the listener binding.
    {
        let data_dir_bf = data_dir.clone();
        let home_bf = state.home_dir.clone();
        tokio::spawn(async move {
            backfill_missing_sidecars(&home_bf, &data_dir_bf).await;
        });
    }

    // Registry SELF-HEAL watchdog. The router learns a worker died only when a
    // request next probes its entry; under scale-to-zero a repo can sit idle for
    // a long time after its worker idle-exits, leaving a stale `Ready` entry —
    // and a `Spawning` entry can orphan if a spawn driver is dropped mid-flight.
    // This periodic sweep drops both (dead `Ready` → absent, over-age `Spawning`
    // → absent + wake awaiters) so the registry converges to the truth without
    // waiting for a request. Cheap: a single write-locked `retain` over a map
    // that holds at most one entry per repo, on a coarse timer. (The orphan-age
    // cutoff `SPAWN_ORPHAN_AFTER` can never reap a live in-flight spawn — see its
    // docs.)
    {
        let registry = state.proxy.registry.clone();
        tokio::spawn(async move {
            // Coarse cadence: orphans are rare and self-heal is not latency-
            // sensitive (read routes already reap dead `Ready` on access via
            // `peek_ready`; this is the backstop). Tick well under the orphan
            // cutoff so a true orphan is cleared within ~one cutoff window.
            let tick = std::time::Duration::from_secs(15);
            loop {
                tokio::time::sleep(tick).await;
                registry.reap(spawn::SPAWN_ORPHAN_AFTER).await;
            }
        });
    }

    // Clone the proxy context BEFORE `state` is moved into `.with_state(state)`
    // below — the caller's shutdown handler needs it to reach the worker registry.
    // Shares the same Arc-backed registry + Job Object as the router's state.
    let proxy = state.proxy.clone();

    let app = build_http_routes()
        .with_state(state)
        // Global multi-repo `/mcp`: proxying handler (repo per call).
        .merge(Router::new().nest_service("/mcp", mcp_service));

    Ok((app, proxy))
}

// ── Native global handlers ──────────────────────────────────────────────────

include!("config_handlers.rs");
include!("repo_read_handlers.rs");
include!("repo_action_handlers.rs");
include!("host_resource_handlers.rs");
include!("aggregate_handlers.rs");
include!("sidecar_backfill.rs");
include!("routes.rs");
