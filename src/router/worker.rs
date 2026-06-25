//! Process-per-project WORKER-side helpers: idle tracking + graceful self-exit.
//!
//! A worker process serves exactly one repo (see `main::run_worker`). To make
//! "scale-to-zero" real, the worker must release everything — RocksDB handle,
//! watcher, resident shard — when it stops being used. It does that by EXITING
//! after an idle window, which lets the OS reclaim all of it at once. The router
//! respawns on the next request.
//!
//! Two pieces live here:
//! - [`IdleTracker`] + [`with_idle_tracking`]: an axum middleware layer that
//!   stamps "last activity = now" on every request, so the watchdog can tell
//!   when the worker has been quiet long enough to exit.
//! - [`spawn_idle_watchdog`]: the background task that performs the graceful
//!   shutdown SEQUENCE when the idle window elapses.
//!
//! ## Why the shutdown SEQUENCE matters (LOCK-race correctness)
//!
//! RocksDB holds an EXCLUSIVE per-directory LOCK. The crucial, measured fact:
//! on Windows that LOCK is released by **process exit** (the OS reclaims the
//! file handle), NOT by dropping the `Surreal<Db>` handle in-process — a
//! store-level test confirmed an in-process drop leaves the LOCK held for >30s.
//! So the watchdog's `close_repo_db` + `clear_repo_index` are about an ORDERLY
//! teardown (cancel the run, abort any migration, drain the pipeline under the
//! per-repo lock so nothing is half-written), while the actual LOCK release
//! comes from the `std::process::exit(0)` that follows.
//!
//! The respawn-safety guarantee is therefore NOT "drop releases the LOCK before
//! exit". It is the pair: (1) the router's `Child::try_wait` confirm-dead never
//! spawns a replacement while this pid is alive — so the new worker's `open_db`
//! only begins after this process has exited and the OS has begun releasing the
//! handle; and (2) `open_db`'s 30s LOCK-drain retry rides out any residual
//! OS-level drain after exit (measured 7s+ under Defender). The 500ms pause
//! before exit is a probability-reducer, not a correctness requirement.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use axum::Router;
use axum::extract::Request;
use axum::middleware::{self, Next};
use axum::response::Response;
use tokio::sync::{Mutex as AsyncMutex, RwLock};
use tracing::info;

use crate::config::{Settings, maybe_reload_settings};
use crate::indexing::IndexEngine;
use crate::store::RepoDbMap;

/// The exact stdout line prefix a worker prints once its listener is bound and
/// ready to accept. The router parses child stdout for a line starting with
/// this token. Stable + machine-readable — do not change without updating
/// `router::spawn::parse_ready_line`.
pub const READY_PREFIX: &str = "CONTEXT_ENGINE_WORKER_READY";

/// Monotonic last-activity clock for a worker. Stores the millisecond stamp of
/// the most recent request (relative to a process-start `Instant`). An atomic so
/// the request hot path never takes a lock; the watchdog reads it on a coarse
/// timer.
#[derive(Clone)]
pub struct IdleTracker {
    /// Milliseconds since `start` of the last observed request.
    last_activity_ms: Arc<AtomicU64>,
    /// Process-start anchor for the millisecond clock.
    start: Arc<Instant>,
    /// Idle window after which the worker self-exits.
    idle_window: Duration,
    /// Count of in-flight requests. The watchdog must NOT exit while > 0 even if
    /// the idle window elapsed (a long-running index/query keeps the worker
    /// alive). Incremented on request entry, decremented on response.
    inflight: Arc<AtomicU64>,
}

impl IdleTracker {
    pub fn new(idle_window: Duration) -> Self {
        let start = Instant::now();
        Self {
            // Start "active now" so a worker isn't eligible to exit before it has
            // served anything (the readiness window counts as activity).
            last_activity_ms: Arc::new(AtomicU64::new(0)),
            start: Arc::new(start),
            idle_window,
            inflight: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Stamp "active now".
    fn touch(&self) {
        let ms = self.start.elapsed().as_millis() as u64;
        self.last_activity_ms.store(ms, Ordering::Relaxed);
    }

    /// Duration since the last request (or since start if none yet).
    fn idle_for(&self) -> Duration {
        let now_ms = self.start.elapsed().as_millis() as u64;
        let last = self.last_activity_ms.load(Ordering::Relaxed);
        Duration::from_millis(now_ms.saturating_sub(last))
    }

    fn inflight_count(&self) -> u64 {
        self.inflight.load(Ordering::Relaxed)
    }
}

/// Wrap an axum `Router` so every request stamps the idle tracker on entry and
/// tracks in-flight count across the whole request lifetime. This is what makes
/// the worker's idle watchdog accurate: ANY proxied request (query, index,
/// stats, MCP) counts as activity.
pub fn with_idle_tracking(app: Router, tracker: IdleTracker) -> Router {
    app.layer(middleware::from_fn_with_state(tracker, idle_tracking_mw))
}

async fn idle_tracking_mw(
    axum::extract::State(tracker): axum::extract::State<IdleTracker>,
    req: Request,
    next: Next,
) -> Response {
    tracker.touch();
    tracker.inflight.fetch_add(1, Ordering::Relaxed);
    let resp = next.run(req).await;
    tracker.inflight.fetch_sub(1, Ordering::Relaxed);
    // Stamp again on completion so a long request's END also counts — otherwise
    // a 40s index that started right before the window could let the worker
    // exit the instant it finishes.
    tracker.touch();
    resp
}

/// Shared state for the worker's mtime-gated config reload. Holds the home dir
/// (to find settings.json), the shared `Settings` handle (swapped on change),
/// and the last-seen mtime behind an async mutex so concurrent requests don't
/// re-parse redundantly.
#[derive(Clone)]
pub struct ConfigReloader {
    home_dir: PathBuf,
    settings: Arc<RwLock<Settings>>,
    last_seen: Arc<AsyncMutex<Option<SystemTime>>>,
}

impl ConfigReloader {
    pub fn new(home_dir: PathBuf, settings: Arc<RwLock<Settings>>) -> Self {
        // Seed with the current mtime so the first request doesn't trigger a
        // redundant reload (boot already loaded the freshest settings).
        let seed = crate::config::settings_mtime(&home_dir);
        Self {
            home_dir,
            settings,
            last_seen: Arc::new(AsyncMutex::new(seed)),
        }
    }
}

/// Wrap an app so every request first mtime-checks `settings.json` and, if the
/// router rewrote it (PUT /api/config), reloads the shared `Settings` handle
/// BEFORE the handler builds its per-request Voyage/LLM clients. This is what
/// makes a rotated key / changed rerank model reach a LIVE worker on its next
/// request — covering BOTH the query path and any index trigger — with no IPC
/// and no worker restart.
pub fn with_config_reload(app: Router, reloader: ConfigReloader) -> Router {
    app.layer(middleware::from_fn_with_state(reloader, config_reload_mw))
}

async fn config_reload_mw(
    axum::extract::State(reloader): axum::extract::State<ConfigReloader>,
    req: Request,
    next: Next,
) -> Response {
    {
        let mut last = reloader.last_seen.lock().await;
        *last = maybe_reload_settings(&reloader.home_dir, &reloader.settings, *last).await;
    }
    next.run(req).await
}

/// Spawn the background watchdog that self-exits the worker after the idle
/// window. Polls on a coarse timer (1/10th of the window, clamped to [1s, 30s]).
///
/// SHUTDOWN SEQUENCE (see module docs for the LOCK-race rationale):
/// 1. Confirm idle: idle_for >= window AND inflight == 0.
/// 2. `close_repo_db`: cancel run + abort migration + drop the cached DB handle
///    under the per-repo lock (drains in-flight pipeline, releases the LOCK).
/// 3. Give RocksDB's async LOCK drain a brief head start, then `exit(0)`.
///
/// We `exit(0)` rather than returning so the watchdog does not need to thread a
/// shutdown signal into `axum::serve`; the router treats a clean child exit as
/// "reap + respawn on next request".
pub fn spawn_idle_watchdog(
    tracker: IdleTracker,
    index_engine: Arc<IndexEngine>,
    repo_dbs: RepoDbMap,
    repo: String,
) {
    let window = tracker.idle_window;
    // Poll cadence: responsive enough to reclaim promptly, coarse enough to be
    // free. 1/10th the window, clamped.
    let tick = (window / 10).clamp(Duration::from_secs(1), Duration::from_secs(30));

    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tick).await;

            if tracker.inflight_count() > 0 {
                // A request is in flight (e.g. a long index/query). Never exit
                // mid-request — that would drop a client mid-stream and could
                // leave the index in a partially-committed state (the pipeline's
                // file_meta crash-safety would self-heal, but we avoid the churn).
                continue;
            }
            if tracker.idle_for() < window {
                continue;
            }

            // Idle long enough with nothing in flight → graceful self-exit.
            info!(
                repo = %repo,
                idle_secs = window.as_secs(),
                "worker idle window elapsed; releasing DB handle and exiting"
            );

            // STEP 1+2: tear down the in-process index state. `close_repo_db`
            // cancels the run, aborts any migration, drains the pipeline under the
            // per-repo lock, and removes the cached handle; `clear_repo_index`
            // drops the resident shard + status. This makes the shutdown ORDERLY
            // (no half-written state, no mid-flight pipeline), but note the LOCK
            // caveat below: on Windows, dropping the Surreal<Db> handle does NOT
            // actually release the RocksDB OS-level LOCK — only PROCESS EXIT does.
            // (Measured: an in-process drop leaves the LOCK held for >30s; a
            // store-level test confirmed the LOCK persists until the process
            // terminates.) So these steps are about clean teardown, not LOCK
            // release.
            index_engine.close_repo_db(&repo).await;
            index_engine.clear_repo_index(&repo).await;
            // Defensive: ensure the map entry is gone even if normalization
            // differed (close_repo_db already removes the normalized key).
            repo_dbs
                .write()
                .await
                .remove(&crate::store::normalize_repo_path(&repo));

            // STEP 3: EXIT THE PROCESS — this is what actually releases the
            // RocksDB LOCK (the OS reclaims the file handle on termination). The
            // brief pause before exit lets `close_repo_db`'s flush settle; it is
            // NOT what frees the LOCK. The respawn-safety guarantee is the
            // combination of (a) the router's `try_wait` confirm-dead — it never
            // spawns a replacement while THIS pid is alive, so the new worker's
            // open begins only after this process has exited and the OS has begun
            // releasing the handle — and (b) `open_db`'s 30s LOCK-drain retry,
            // which rides out any residual OS-level drain after exit.
            tokio::time::sleep(Duration::from_millis(500)).await;

            info!(repo = %repo, "worker exiting (scale-to-zero)");
            std::process::exit(0);
        }
    });
}
