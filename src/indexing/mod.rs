pub mod events;
pub mod frameworks;
pub mod import_resolver;
pub mod pipeline;
pub mod rebuild_guard;
pub mod retry_backoff;
pub mod tracker;
pub mod walker;
pub mod watcher;

#[cfg(test)]
mod load_repos_tests;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::config::Settings;
use crate::embedding::identity::{EMBEDDING_IDENTITY_KEY, EmbeddingIdentity};
use crate::embedding::voyage::VoyageClient;
use crate::indexing::events::{IndexEvent, IndexEventBus};
use crate::indexing::pipeline::IndexPipeline;
use crate::indexing::rebuild_guard::IdentityRebuildGuard;
use crate::indexing::tracker::FileChange;
use crate::indexing::watcher::start_watcher;
use crate::store::ops::set_meta;
use crate::store::{self, RepoDbMap};
use crate::vector::{ScopedAll, ScopedSearch, SearchResult, ShardedVectorIndex, VectorIndex};

use surrealdb::Surreal;
use surrealdb::engine::local::Db;

// ─── Repo indexing status ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum IndexState {
    Idle,
    Indexing,
    Error,
}

/// Which stage of an in-flight run the pipeline is currently in.
///
/// The file-count progress bar (`indexed_files / total_files`) only describes
/// the `Embedding` stage. After every file is embedded the bar pins at 100% but
/// the run is NOT done: `SymbolIndex` (concurrent index rebuild) and
/// `ResolveEdges` (Phase 2) run for many minutes at kernel scale with no
/// file-count motion. This enum lets the UI show what's happening past 100%.
///
/// Only meaningful while `state == Indexing`; `Idle` otherwise.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexPhase {
    /// Not in an indexing run (state is Idle/Error), or run hasn't started a stage.
    Idle,
    /// Parse → embed → store loop. Progress = `indexed_files / total_files`.
    Embedding,
    /// Rebuilding symbol indexes (concurrent `DEFINE INDEX`). Indeterminate —
    /// a single blocking DB op with no sub-progress to report.
    SymbolIndex,
    /// Phase 2 edge resolution. Progress = `phase_done / phase_total` (edges)
    /// when `phase_total > 0`; indeterminate while `phase_total == 0`.
    ResolveEdges,
}

/// Per-repo status snapshot returned by `GET /api/index-status`.
///
/// Dual-meaning fields (state-gated contract):
/// - `state == Indexing`: `total_files` = workset size for this run (denominator);
///   `indexed_files` = files whose chunks have been embedded so far (numerator).
///   Progress % = indexed_files / total_files (guard against total_files == 0).
/// - `state == Idle` / `Error`: `indexed_files` = files in the index;
///   `total_files` = total files in the repo on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoStatus {
    pub state: IndexState,
    pub indexed_files: u64,
    pub total_files: u64,
    pub last_indexed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
    /// Current pipeline stage. `Idle` unless `state == Indexing`.
    #[serde(default = "default_phase")]
    pub phase: IndexPhase,
    /// Sub-progress numerator for `phase == ResolveEdges` (edges resolved so far).
    /// `0` for all other phases. Paired with `phase_total`.
    #[serde(default)]
    pub phase_done: u64,
    /// Sub-progress denominator for `phase == ResolveEdges` (total edges to
    /// resolve). `0` means indeterminate (show a pulsing bar, no percentage).
    #[serde(default)]
    pub phase_total: u64,
}

fn default_phase() -> IndexPhase {
    IndexPhase::Idle
}

impl Default for RepoStatus {
    fn default() -> Self {
        Self {
            state: IndexState::Idle,
            indexed_files: 0,
            total_files: 0,
            last_indexed_at: None,
            error: None,
            phase: IndexPhase::Idle,
            phase_done: 0,
            phase_total: 0,
        }
    }
}

// ─── ProgressHandle ───────────────────────────────────────────────────────

/// Lightweight handle passed into `IndexPipeline::run` so the pipeline can
/// write live progress into the shared status map without knowing the full engine.
#[derive(Clone)]
pub struct ProgressHandle {
    statuses: Arc<RwLock<HashMap<String, RepoStatus>>>,
    repo: String,
}

impl ProgressHandle {
    /// Test-only constructor over a bare status map, so pipeline tests can
    /// observe phase/sub-progress transitions without booting the full engine.
    #[cfg(test)]
    pub fn new_for_test(statuses: Arc<RwLock<HashMap<String, RepoStatus>>>, repo: String) -> Self {
        Self { statuses, repo }
    }

    /// Set the denominator once the parsed file set is known.
    pub async fn set_run_total(&self, total: u64) {
        let mut map = self.statuses.write().await;
        let s = map.entry(self.repo.clone()).or_default();
        s.total_files = total;
    }

    /// Advance the numerator. Monotonic — never decreases.
    pub async fn set_processed(&self, processed: u64) {
        let mut map = self.statuses.write().await;
        let s = map.entry(self.repo.clone()).or_default();
        if processed > s.indexed_files {
            s.indexed_files = processed;
        }
    }

    /// Move the run to a new pipeline stage. Resets the edge sub-progress
    /// counters (only `ResolveEdges` populates them).
    pub async fn set_phase(&self, phase: IndexPhase) {
        let mut map = self.statuses.write().await;
        let s = map.entry(self.repo.clone()).or_default();
        s.phase = phase;
        s.phase_done = 0;
        s.phase_total = 0;
    }

    /// Set the edge-resolution denominator at the start of Phase 2.
    pub async fn set_phase_total(&self, total: u64) {
        let mut map = self.statuses.write().await;
        let s = map.entry(self.repo.clone()).or_default();
        s.phase_total = total;
    }

    /// Advance the edge-resolution numerator. Monotonic — never decreases.
    /// Called throttled (every N batches) to bound RwLock churn at scale.
    pub async fn set_phase_done(&self, done: u64) {
        let mut map = self.statuses.write().await;
        let s = map.entry(self.repo.clone()).or_default();
        if done > s.phase_done {
            s.phase_done = done;
        }
    }
}

// ─── IndexEngine ──────────────────────────────────────────────────────────

/// Debounce window for the per-repo graph/stats cache recompute.
///
/// WHY this exists (connection contention): there is exactly ONE RocksDB handle
/// per repo (exclusive per-dir lock — see `store::get_or_open`), and the
/// graph/stats cache recompute is an O(repo) full-table GROUP BY that does NOT
/// cooperatively yield (~90s at kernel scale). If it fires on EVERY index
/// completion — as it used to — an edit burst (user saves N files → N
/// completions) runs it back-to-back, permanently pinning the one connection so
/// every incremental's own queries stall behind it. Detaching it onto a
/// `tokio::spawn` frees the consumer TASK but NOT the CONNECTION.
///
/// The fix is to DEFER the recompute until the repo has gone quiet: we only run
/// it after no new completion has arrived for this window. Long enough to absorb
/// a save burst, short enough the cache isn't very stale. The `/graph` and
/// `/index-stats` serve paths each have a cold-miss fallback that recomputes
/// on-demand, so a request landing in the gap before the deferred recompute is
/// still correct (just pays the one-off aggregation itself).
///
/// CONSIDERED-AND-REJECTED: *cancelling* an in-flight recompute on a new trigger
/// does NOT reliably free the connection — the ~90s GROUP BY is a single
/// non-yielding RocksDB call, so a `CancellationToken` can only take effect
/// BETWEEN the graph query and the stats query, never mid-query. Debounce (defer
/// until quiet) sidesteps the contention entirely instead of racing it. Making
/// the aggregation itself incremental (O(changed)) is the long-term ideal but is
/// a much larger change (the graph cache is a global degree-ranked hub subgraph,
/// not trivially incremental) and is explicitly a FUTURE optimization, NOT this
/// task — debounce already keeps the recompute off the incremental's critical
/// path so the locked ≤10s wall criterion holds without it.
const CACHE_RECOMPUTE_DEBOUNCE: Duration = Duration::from_millis(4000);

/// Per-repo recompute scheduler state. Tracks a single-flight debounced cache
/// recompute. Guarded by `IndexEngine::recompute_slots` (a `std::sync::Mutex` —
/// every critical section is tiny and never spans an `.await`).
#[derive(Default)]
struct RecomputeSlot {
    /// Instant of the most recent index completion. The debounce timer waits
    /// until `last_completion.elapsed() >= CACHE_RECOMPUTE_DEBOUNCE` before
    /// running, so each new completion pushes the run further out (leading edge
    /// absorbed, trailing edge fires once the burst settles).
    last_completion: Option<Instant>,
    /// True while a scheduler task exists for this repo (either sleeping in the
    /// debounce window or mid-recompute). Single-flight: at most one task per
    /// repo — a completion arriving while one exists never spawns a second.
    scheduled: bool,
    /// True while the spawned task is actually running the (~90s) aggregation
    /// rather than sleeping. A completion landing in this phase sets `rearm` so
    /// exactly ONE more pass runs after the current one, guaranteeing the final
    /// cache reflects the last completion.
    recomputing: bool,
    /// Trailing-edge re-arm flag: set by a completion that arrived while the task
    /// was mid-recompute. Checked after the recompute finishes to decide whether
    /// to loop once more (rearm) or retire (clear `scheduled`).
    rearm: bool,
}

/// Central orchestrator for all indexing operations.
/// Stored in `AppState` and shared via `Arc`.
pub struct IndexEngine {
    /// Boot-resolved data directory base (CLI > env > `Settings.data_dir` >
    /// builtin default). Captured ONCE at startup; the indexer/store paths are
    /// derived from this. **Never re-read from `Settings` mid-run** —
    /// already-open RocksDB handles in `repo_dbs` and resident vector shards
    /// are bound to this path; switching would split-brain reads against writes.
    pub data_dir: PathBuf,
    /// Boot-resolved embedding-cache root (precedence: CLI, env
    /// `CONTEXT_ENGINE_EMBEDDINGS_DIR`, `Settings.embeddings_dir`, then
    /// `<data_dir>/embeddings`). Separate from `data_dir` because the cache is
    /// content-addressed and concurrency-safe, so it can be SHARED across
    /// instances (only RocksDB needs per-instance isolation). Boot-frozen.
    pub embeddings_dir: PathBuf,
    /// Per-repo status map, keyed by repo path string.
    /// Wrapped in Arc so `ProgressHandle` can hold a reference without borrowing self.
    pub statuses: Arc<RwLock<HashMap<String, RepoStatus>>>,
    /// Serialises concurrent pipeline runs per repo.
    repo_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    /// Channel sender for triggering index runs (manual or watcher-driven).
    trigger_tx: tokio::sync::mpsc::Sender<IndexTrigger>,
    /// Per-repo sharded vector index with a resident-byte cap and repo-keyed LRU.
    /// Bounds memory axis 2: only the active working set of repos stays resident;
    /// cold repos are evicted and background-warmed on demand.
    pub vector_index: Arc<RwLock<ShardedVectorIndex>>,
    /// Per-repo async serialisation lock for warming. Gives single-flight semantics
    /// for the blocking warm-on-query path: when N queries hit the same cold repo,
    /// the first acquires the lock and runs the warm; the rest block on the lock,
    /// then re-check residency under it and return WITHOUT re-running `load_from_db`
    /// (a 0.4–1.1 GB scan for a large repo) or contending for the write lock.
    ///
    /// A `tokio::sync::Mutex` (not std) because the critical section spans the
    /// `.await` on the DB scan + shard install. Mirrors `repo_locks`. The lock is
    /// dropped on every exit path (normal, error, cancellation) by RAII guard drop.
    warm_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    /// Shared per-repo DB handles — the same map the server reads through, so
    /// indexer writes are visible to explorer/query reads (one instance per repo).
    repo_dbs: RepoDbMap,
    /// Broadcast channel for streaming indexing events to SSE clients.
    pub event_bus: IndexEventBus,
    /// Per-repo cancellation tokens. The consumer checks this before each file in
    /// the pipeline. A cancelled token triggers graceful early exit.
    cancel_tokens: Mutex<HashMap<String, CancellationToken>>,
    /// Shared live settings handle — the single source of truth for per-repo index
    /// generation. Read (never written) by the lazy warm-on-query path so it can
    /// resolve a repo's current generation without the caller threading it through
    /// the query layer. Only the generation counter is read mid-run; `data_dir`
    /// stays boot-frozen.
    settings_handle: Arc<RwLock<Settings>>,
    /// Per-repo debounced single-flight graph/stats cache recompute state.
    /// See [`CACHE_RECOMPUTE_DEBOUNCE`] and [`RecomputeSlot`] for the WHY
    /// (connection contention) and the coalescing mechanism. A `std::sync::Mutex`
    /// because every critical section is a few field reads/writes and never spans
    /// an `.await`. Wrapped in `Arc` so the detached scheduler task can hold a
    /// reference without borrowing the engine.
    recompute_slots: Arc<std::sync::Mutex<HashMap<String, RecomputeSlot>>>,
    /// Coalesces automatic identity-mismatch rebuild triggers per repo.
    identity_rebuild: IdentityRebuildGuard,
    /// Per-repo automatic retry after a failed run or one that left files in
    /// `pending_retry` (see [`retry_backoff`]).
    retry_backoff: Arc<retry_backoff::RetryBackoff>,
}

#[derive(Debug)]
pub struct IndexTrigger {
    pub repo: String,
    pub changes: Option<Vec<FileChange>>, // None = full incremental scan
    /// Force a full rebuild (clear + re-embed everything) regardless of staleness.
    pub rebuild: bool,
}

/// Outcome of warming one repo's resident vector shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmResult {
    Installed(usize),
    IdentityMismatch,
    MutationInProgress,
}

/// Warm one repo's vector shard: load its embeddings from DB into a temp
/// `VectorIndex` (NO lock held during the scan), then install the finished shard
/// under a short write lock. Returns the number of vectors loaded (0 on failure
/// or empty repo). `active` shards are protected from eviction during install.
///
/// This is the single warm path used by both boot warming and lazy on-query warm.
/// The lock-order contract (repo_dbs → vector_index) holds: `get_or_open` touches
/// only repo_dbs; the write lock on vector_index is taken AFTER repo_dbs work is
/// done and the DB scan has completed.
pub(crate) async fn warm_repo_shard(
    vector_index: &Arc<RwLock<ShardedVectorIndex>>,
    repo_dbs: &RepoDbMap,
    data_dir: &std::path::Path,
    repo: &str,
    generation: u32,
    current: &EmbeddingIdentity,
    active: &[String],
) -> anyhow::Result<WarmResult> {
    // Capture the mutation epoch BEFORE any DB read. The guard is dropped at the
    // end of the statement (never held across an await, never nested inside
    // repo_dbs). Every install below requires it unchanged: an update that begins
    // and publishes while the scan runs leaves `is_mutating` false again, so only
    // the epoch reveals that this snapshot predates it.
    let scan_epoch = vector_index.read().await.mutation_epoch(repo);

    let db = store::get_or_open(repo_dbs, data_dir, repo, generation).await?;

    // DB reads happen with NO vector-index lock held. Fail closed: inability to
    // establish either content or identity never falls through to installing or
    // searching an unverified shard.
    let chunk_count = crate::store::ops::count_chunks(&db).await?;
    if chunk_count == 0 {
        return Ok(WarmResult::Installed(0));
    }

    let current_key = current.as_key_string();
    let stored_key = crate::store::ops::get_meta(&db, EMBEDDING_IDENTITY_KEY).await?;
    if stored_key.as_deref() != Some(current_key.as_str()) {
        // Durable marker is a cross-restart safety net. Correctness does not depend
        // on this write succeeding: warm_repo_blocking also queues rebuild:true.
        if let Err(e) = set_meta(&db, "needs_rebuild", "1").await {
            warn!(repo = %repo, error = %e, "warm: failed to persist needs_rebuild after embedding identity mismatch");
        }
        warn!(repo = %repo, stored_identity = ?stored_key, current_identity = %current_key,
              "warm: embedding identity mismatch; blocking stale shard and forcing rebuild");
        return Ok(WarmResult::IdentityMismatch);
    }

    let index_epoch = crate::store::ops::get_index_epoch(&db).await?;
    let stamp = current.content_stamp(chunk_count, index_epoch);

    // Fast path: a valid, current persisted shard file → mmap it (near-instant,
    // no SELECT + decode). The identity is already validated against DB above;
    // content_stamp independently invalidates same-count shards from other models.
    match crate::vector::shard_file::open_current(data_dir, repo, 0, stamp) {
        Ok(Some((shard, generation_loaded))) => {
            let count = shard.len();
            if count > 0 {
                let mut vi = vector_index.write().await;
                if !vi.install_shard_if_not_mutating(
                    repo,
                    shard,
                    current_key.clone(),
                    active,
                    scan_epoch,
                ) {
                    return Ok(WarmResult::MutationInProgress);
                }
                crate::vector::shard_file::reap_stale_generations(
                    data_dir,
                    repo,
                    &[generation_loaded],
                );
                info!(repo = %repo, count, generation = generation_loaded, "warm: mmap'd persisted shard (no DB scan)");
                return Ok(WarmResult::Installed(count));
            }
        }
        Ok(None) => {} // no usable file — build from DB below
        Err(e) => {
            warn!(repo = %repo, error = %e, "warm: shard file open failed; rebuilding from DB")
        }
    }

    // Slow path: build from the chunk table, then persist a new generation so
    // subsequent warms mmap it. DB decode errors propagate (fail closed).
    let shard = VectorIndex::load_from_db(&db).await?;
    let count = shard.len();
    if count == 0 {
        return Ok(WarmResult::Installed(0));
    }
    {
        // Never persist a snapshot that an update overlapped: it would become the
        // CURRENT file other warms mmap.
        let vi = vector_index.read().await;
        if vi.is_mutating(repo) || vi.mutation_epoch(repo) != scan_epoch {
            return Ok(WarmResult::MutationInProgress);
        }
    }
    let persisted = match crate::vector::shard_file::write_new_generation(
        data_dir, repo, &shard, stamp,
    ) {
        Ok(g) => Some(g),
        Err(e) => {
            warn!(repo = %repo, error = %e, "warm: failed to persist shard file (will rebuild next warm)");
            None
        }
    };

    let mut vi = vector_index.write().await;
    if let Some(g) = persisted
        && let Ok(Some((mmap_shard, _))) =
            crate::vector::shard_file::open_current(data_dir, repo, shard.dim(), stamp)
    {
        if !vi.install_shard_if_not_mutating(repo, mmap_shard, current_key, active, scan_epoch) {
            drop(vi);
            crate::vector::shard_file::invalidate_current_if_generation(data_dir, repo, Some(g));
            return Ok(WarmResult::MutationInProgress);
        }
        crate::vector::shard_file::reap_stale_generations(data_dir, repo, &[g]);
        info!(repo = %repo, count, generation = g, "warm: built from DB, persisted + mmap'd shard");
    } else {
        if !vi.install_shard_if_not_mutating(repo, shard, current_key, active, scan_epoch) {
            return Ok(WarmResult::MutationInProgress);
        }
        info!(repo = %repo, count, "warm: installed in-RAM shard (persist/mmap unavailable)");
    }
    Ok(WarmResult::Installed(count))
}

/// Restore persisted per-repo status (file count + last-indexed timestamp) from
/// each repo's DB at boot. Without this, a repo indexed in a prior session reads
/// `indexed_files: 0` until it is re-indexed, so the UI shows a blank count.
///
/// Only entries still at the untouched `Idle` default with a zero count are
/// updated — if a watcher- or user-triggered run has already advanced a repo's
/// status by the time this runs, that live state is left intact.
pub(crate) async fn seed_statuses_from_db(
    statuses: &Arc<RwLock<HashMap<String, RepoStatus>>>,
    repo_dbs: &RepoDbMap,
    data_dir: &std::path::Path,
    repos: &[String],
    generations: &HashMap<String, u32>,
    settings_handle: &Arc<RwLock<crate::config::Settings>>,
) {
    for repo in repos {
        let repo = crate::store::normalize_repo_path(repo);
        let generation = generations.get(&repo).copied().unwrap_or(0);
        let db = match store::get_or_open(repo_dbs, data_dir, &repo, generation).await {
            Ok(db) => db,
            Err(e) => {
                warn!(repo = %repo, error = %format!("{e:#}"), "failed to open DB for status seed; skipping repo");
                continue;
            }
        };
        let indexed = match store::ops::count_indexed_files(&db, &repo).await {
            Ok(n) => n,
            Err(e) => {
                warn!(repo = %repo, error = %e, "failed to count indexed files for status seed; skipping repo");
                continue;
            }
        };
        if indexed == 0 {
            continue; // never indexed — leave the default so the UI can show a placeholder
        }
        let last_indexed_at = store::ops::get_meta(&db, "last_indexed_at")
            .await
            .ok()
            .flatten()
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        // Backfill the router-readable sidecar if missing. A repo indexed by the
        // pre-router monolith (or before sidecars existed) has data on disk but
        // no sidecar, so the router's cold view can't show its file count. Here —
        // where a worker has the DB open at boot and already paid the count — we
        // write the sidecar so the next router status poll shows the real count
        // instead of just "indexed (count pending)". Best-effort; only write when
        // absent so we never clobber a fresher sidecar from a completed run.
        if crate::router::sidecar::read_sidecar(data_dir, &repo).is_none() {
            let (embedding_model, embedding_dim) = {
                let cfg = settings_handle.read().await;
                (
                    cfg.embedding.model.clone(),
                    cfg.embedding.dimensions.unwrap_or(0) as u64,
                )
            };
            let meta = crate::router::sidecar::RepoSidecar {
                file_count: indexed,
                last_indexed_at: last_indexed_at.map(|dt| dt.to_rfc3339()),
                state: "indexed".to_string(),
                embedding_model,
                embedding_dim,
                schema: crate::router::sidecar::SIDECAR_SCHEMA,
            };
            if let Err(e) = crate::router::sidecar::write_sidecar(data_dir, &repo, &meta) {
                warn!(repo = %repo, error = %format!("{e:#}"), "failed to backfill sidecar at boot");
            }
        }

        // Backfill the AUX sidecars (graph + files) too if missing, so a repo
        // indexed before these caches existed shows its detail view (graph +
        // file list) on detail-open without a spawn. The worker already has the
        // DB open here. Guarded by `indexed > 0` (we're past the count==0 early
        // continue above). Best-effort.
        if crate::router::sidecar::read_aux_json::<crate::store::ops::CallGraph>(
            data_dir, &repo, "graph",
        )
        .is_none()
            && let Ok(graph) = crate::store::ops::compute_and_cache_graph(&db).await
        {
            let _ = crate::router::sidecar::write_aux_json(data_dir, &repo, "graph", &graph);
        }
        if crate::router::sidecar::read_aux_json::<serde_json::Value>(data_dir, &repo, "files")
            .is_none()
            && let Ok(files) = crate::store::ops::files_page(&db, &repo, 2000, None).await
        {
            let _ = crate::router::sidecar::write_aux_json(data_dir, &repo, "files", &files);
        }

        let mut map = statuses.write().await;
        let s = map.entry(repo.clone()).or_default();
        // Don't clobber a run that started after boot.
        if s.state == IndexState::Idle && s.indexed_files == 0 {
            s.indexed_files = indexed;
            s.last_indexed_at = last_indexed_at;
        }
    }
}

/// Outcome of a `vector_search` call.
///
/// `warming` distinguishes a transient cold/warming shard from a genuine empty:
/// it is true only when a single-repo query found the target shard NOT resident
/// after the bounded warm-wait expired. An empty `results` with `warming=true`
/// means "retry shortly", NOT "the index contains nothing". A resident shard that
/// matches nothing yields `warming=false` (a real empty).
pub struct VectorSearchOutcome {
    pub results: Vec<SearchResult>,
    pub warming: bool,
}

impl IndexEngine {
    /// Create the engine and spawn the watcher background task.
    ///
    /// `repo_dbs` is the shared handle map (also held by the server); the
    /// indexer writes through these same handles so reads observe its commits.
    /// `settings_handle` is the shared live settings source of truth; the consumer
    /// task takes a fresh snapshot at the top of each trigger iteration so API keys
    /// and other config added after boot are picked up on the next run.
    pub async fn start(
        data_dir: PathBuf,
        embeddings_dir: PathBuf,
        settings: &Settings,
        repo_dbs: RepoDbMap,
        settings_handle: Arc<RwLock<Settings>>,
        no_watchers: bool,
    ) -> Arc<Self> {
        let (trigger_tx, trigger_rx) = tokio::sync::mpsc::channel::<IndexTrigger>(256);

        // Derive the resident-byte cap from settings (MB → bytes). 0 disables the
        // cap (unbounded — not recommended; kept as an escape hatch).
        let cap_bytes = settings.vector_resident_cap_mb.saturating_mul(1024 * 1024);

        // Start with an empty sharded index so the server can bind immediately.
        // Shards are warmed in a background task below (bounded by the cap), and
        // any not warmed at boot are loaded on first query (background warm).
        let vector_index = Arc::new(RwLock::new(ShardedVectorIndex::new(cap_bytes)));

        // Clone handles needed by the background vector-load task BEFORE they
        // are moved into the engine struct.
        let repo_dbs_bg = repo_dbs.clone();
        let data_dir_bg = data_dir.clone();
        let repos_bg = settings.repos.clone();
        let generations_bg = settings.repo_generations.clone();

        let engine = Arc::new(IndexEngine {
            data_dir: data_dir.clone(),
            embeddings_dir,
            statuses: Arc::new(RwLock::new(HashMap::new())),
            repo_locks: Mutex::new(HashMap::new()),
            trigger_tx: trigger_tx.clone(),
            vector_index,
            warm_locks: Mutex::new(HashMap::new()),
            repo_dbs,
            event_bus: IndexEventBus::new(),
            cancel_tokens: Mutex::new(HashMap::new()),
            settings_handle: settings_handle.clone(),
            recompute_slots: Arc::new(std::sync::Mutex::new(HashMap::new())),
            identity_rebuild: IdentityRebuildGuard::default(),
            retry_backoff: Arc::new(retry_backoff::RetryBackoff::default()),
        });

        // Initialise status entries.
        {
            let mut statuses = engine.statuses.write().await;
            for repo in &settings.repos {
                statuses.insert(
                    crate::store::normalize_repo_path(repo),
                    RepoStatus::default(),
                );
            }
        }

        // Spawn a background task that restores each repo's persisted file count /
        // last-indexed timestamp from its DB so previously-indexed repos show their
        // count after a restart. Vector shards are NOT warmed here: warming loads
        // every repo's embeddings from RocksDB into RAM (a transient multi-GB spike
        // on the SurrealDB deserialize path), which is wasteful when most repos are
        // never queried in a session. Shards are instead warmed lazily and blockingly
        // on first query to a repo (see `vector_search`), bounding boot RAM to the
        // open DB handles regardless of how many repos are configured.
        {
            let statuses_bg = Arc::clone(&engine.statuses);
            let settings_bg = settings_handle.clone();
            tokio::spawn(async move {
                seed_statuses_from_db(
                    &statuses_bg,
                    &repo_dbs_bg,
                    &data_dir_bg,
                    &repos_bg,
                    &generations_bg,
                    &settings_bg,
                )
                .await;
            });
        }

        // Startup sweep: reap stale vector-shard generations. No mmap handles
        // survive a restart, so every non-CURRENT generation dir is reapable. This
        // bounds disk use across restarts (a crashed/aborted rewrite can leave an
        // extra gen dir behind). Cheap (directory listing per repo); run in the
        // background so it never delays boot.
        {
            let data_dir_sweep = engine.data_dir.clone();
            let repos_sweep = settings.repos.clone();
            tokio::spawn(async move {
                for repo in repos_sweep {
                    // keep=[] → reap everything except CURRENT (reap_stale_generations
                    // always preserves the CURRENT gen internally).
                    crate::vector::shard_file::reap_stale_generations(&data_dir_sweep, &repo, &[]);
                }
            });
        }

        // Start watcher for each repo — UNLESS the caller suppressed watchers
        // (the bench oracle does, so a boot watcher on a repo already in
        // settings.repos can't fire its own incremental on the bench's on-disk
        // edits and contaminate the measured run; see BootOptions::no_watchers).
        if !no_watchers {
            for repo in settings.repos.clone() {
                let tx = trigger_tx.clone();
                let repo_path = crate::store::normalize_repo_path(&repo);
                tokio::spawn(async move {
                    start_watcher(repo_path, tx).await;
                });
            }
        }

        // Spawn the single consumer task — passes the SHARED handle so the consumer
        // can take a fresh snapshot on each trigger (picks up post-boot config changes).
        let engine_clone = engine.clone();
        let settings_handle_clone = settings_handle.clone();
        tokio::spawn(async move {
            run_consumer(engine_clone, trigger_rx, settings_handle_clone).await;
        });

        engine
    }

    /// Register a repository at runtime (not known at boot): seed a status entry
    /// and spawn its filesystem watcher, mirroring what `start` does per-repo at
    /// boot. Idempotent: if the repo already has a status entry, this is a no-op
    /// (avoids spawning a duplicate watcher).
    pub async fn register_repo(&self, repo: &str) {
        let repo = crate::store::normalize_repo_path(repo);
        {
            let mut statuses = self.statuses.write().await;
            if statuses.contains_key(&repo) {
                return; // already registered — don't spawn a second watcher
            }
            statuses.insert(repo.clone(), RepoStatus::default());
        }
        let tx = self.trigger_tx.clone();
        let repo_path = repo;
        tokio::spawn(async move {
            start_watcher(repo_path, tx).await;
        });
    }

    /// Register a repo's status entry WITHOUT spawning a filesystem watcher.
    ///
    /// For measurement tools (e.g. `bench-incremental`) that mutate files on disk
    /// to drive a controlled incremental run and must guarantee the ONLY trigger
    /// in flight is the one they explicitly send. A live watcher would fire its
    /// own debounced trigger on those same edits (and on the restore), polluting
    /// the measured window with extra runs racing for the single per-repo
    /// connection. Idempotent: a no-op if the repo already has a status entry.
    pub async fn register_repo_no_watcher(&self, repo: &str) {
        let repo = crate::store::normalize_repo_path(repo);
        let mut statuses = self.statuses.write().await;
        statuses.entry(repo).or_default();
    }

    /// True if a debounced graph/stats cache recompute is still pending or running
    /// for `repo` (a scheduler task exists — sleeping in the debounce window or
    /// mid-aggregation). False once it has retired and the shared connection is
    /// idle. Lets a measurement tool wait for the post-rebuild recompute to fully
    /// drain before timing an incremental, so the recompute never overlaps the
    /// measured window on the single per-repo connection.
    pub fn recompute_pending(&self, repo: &str) -> bool {
        let repo = crate::store::normalize_repo_path(repo);
        let slots = match self.recompute_slots.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        slots.get(&repo).map(|s| s.scheduled).unwrap_or(false)
    }

    /// Send a manual trigger to index a single repo.
    pub async fn trigger_index(&self, repo: &str) -> Result<()> {
        self.trigger_tx
            .send(IndexTrigger {
                repo: crate::store::normalize_repo_path(repo),
                changes: None,
                rebuild: false,
            })
            .await
            .map_err(|e| anyhow::anyhow!("trigger channel closed: {e}"))?;
        Ok(())
    }

    /// Send a manual trigger to fully rebuild a single repo's index.
    pub async fn trigger_rebuild(&self, repo: &str) -> Result<()> {
        self.trigger_tx
            .send(IndexTrigger {
                repo: crate::store::normalize_repo_path(repo),
                changes: None,
                rebuild: true,
            })
            .await
            .map_err(|e| anyhow::anyhow!("trigger channel closed: {e}"))?;
        Ok(())
    }

    /// Send a manual trigger to index all repos.
    pub async fn trigger_index_all(&self, repos: &[String]) -> Result<()> {
        for repo in repos {
            self.trigger_index(repo).await?;
        }
        Ok(())
    }

    /// Return per-repo status snapshot.
    pub async fn all_statuses(&self) -> HashMap<String, RepoStatus> {
        self.statuses.read().await.clone()
    }

    /// Return status for a single repo.
    pub async fn repo_status(&self, repo: &str) -> Option<RepoStatus> {
        let repo = crate::store::normalize_repo_path(repo);
        self.statuses.read().await.get(&repo).cloned()
    }

    /// True only during the destructive DB/vector publication window for `repo`.
    /// Unlike `RepoStatus::Indexing`, this does not include queued/early work that
    /// can safely continue serving the previous complete index.
    pub async fn repo_is_mutating(&self, repo: &str) -> bool {
        let repo = crate::store::normalize_repo_path(repo);
        self.vector_index.read().await.is_mutating(&repo)
    }

    /// Clear all in-memory index state for a repo after its on-disk index has
    /// been removed: reset the status counters to default (Idle, 0 files) and
    /// evict the resident vector shard. The status entry is reset in place — not
    /// removed — so the existing filesystem watcher registration is preserved and
    /// a later `register_repo` can't spawn a duplicate watcher.
    pub async fn clear_repo_index(&self, repo: &str) {
        let repo = crate::store::normalize_repo_path(repo);
        {
            let mut statuses = self.statuses.write().await;
            if let Some(s) = statuses.get_mut(&repo) {
                *s = RepoStatus::default();
            }
        }
        self.vector_index.write().await.remove_repo(&repo);
    }

    /// Cancel indexing, abort any in-flight schema migration, then remove the
    /// cached DB handle from `repo_dbs`. The per-repo serialisation lock guarantees
    /// that no pipeline iteration is mid-flight when the handle is removed — the
    /// consumer holds this lock for the entire iteration (open → run → emit).
    ///
    /// Returns once the handle has been removed and no pipeline holds it.
    pub async fn close_repo_db(&self, repo: &str) {
        let repo = crate::store::normalize_repo_path(repo);
        // 1. Cancel any in-progress run so it exits early.
        self.cancel_index(&repo).await;

        // Abort any in-flight schema migration for this repo. A running migration
        // holds a live `Surreal<Db>` clone that pins the RocksDB exclusive LOCK
        // (see store::maybe_spawn_migration). Aborting + awaiting it here drops that
        // clone deterministically so `remove_index_dir` can delete the directory.
        // Safe: migrations are idempotent + crash-resumable, so an aborted migration
        // self-heals on the next open.
        crate::store::abort_migration(&repo).await;

        // 2. Acquire the per-repo lock — blocks until the consumer's current
        //    iteration for this repo finishes (including dropping its `db` clone).
        let lock = self.get_repo_lock(&repo).await;
        let _guard = lock.lock().await;

        // 3. Under the lock, remove the cached handle. No one else can open a
        //    new one for this repo while we hold the lock (the consumer is the
        //    only other caller of get_or_open, and it needs this same lock).
        let mut map = self.repo_dbs.write().await;
        map.remove(&repo);
    }

    /// Cancel an in-progress indexing run for a repo. Returns true if the repo
    /// was actively indexing and the cancellation signal was sent.
    pub async fn cancel_index(&self, repo: &str) -> bool {
        let repo = crate::store::normalize_repo_path(repo);
        // Only cancel if the repo is actually in 'indexing' state — avoids a
        // TOCTOU race where the pipeline finishes between Ok(stats) and
        // clear_cancel_token, making the token still present but the run done.
        let is_indexing = {
            let statuses = self.statuses.read().await;
            statuses
                .get(&repo)
                .map(|s| s.state == IndexState::Indexing)
                .unwrap_or(false)
        };
        if !is_indexing {
            return false;
        }
        let mut tokens = self.cancel_tokens.lock().await;
        if let Some(token) = tokens.remove(&repo) {
            token.cancel();
            true
        } else {
            false
        }
    }

    /// Get or create a fresh cancellation token for a repo run.
    /// Called at the start of each consumer iteration.
    async fn new_cancel_token(&self, repo: &str) -> CancellationToken {
        let token = CancellationToken::new();
        let mut tokens = self.cancel_tokens.lock().await;
        tokens.insert(repo.to_string(), token.clone());
        token
    }

    /// Remove the cancellation token for a repo (run finished).
    async fn clear_cancel_token(&self, repo: &str) {
        let mut tokens = self.cancel_tokens.lock().await;
        tokens.remove(repo);
    }

    async fn get_repo_lock(&self, repo: &str) -> Arc<Mutex<()>> {
        let mut locks = self.repo_locks.lock().await;
        locks
            .entry(repo.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Public accessor for the per-repo serialisation lock (used by server handlers
    /// that need to mutate per-repo state without racing the consumer).
    pub async fn get_repo_lock_public(&self, repo: &str) -> Arc<Mutex<()>> {
        self.get_repo_lock(repo).await
    }

    /// True while an index run holds `repo`'s run lock or its status reads
    /// Indexing. The worker idle watchdog treats this as activity: it used to
    /// count only HTTP requests, so a watcher- or catch-up-driven run with no
    /// client attached was cancelled when the idle window elapsed (forcing a
    /// full rebuild on the next spawn).
    pub async fn is_index_busy(&self, repo: &str) -> bool {
        let repo = crate::store::normalize_repo_path(repo);
        if self.get_repo_lock(&repo).await.try_lock().is_err() {
            return true;
        }
        self.statuses
            .read()
            .await
            .get(&repo)
            .is_some_and(|s| s.state == IndexState::Indexing)
    }

    /// Search the resident vector shards for the top-k most similar chunks.
    ///
    /// Fan-out over resident shards + bounded global top-k merge (scores are
    /// comparable across shards: all L2-normalized cosine).
    ///
    /// Cold-repo handling depends on scope:
    /// - `repo_filter = Some(repo)` (the only scope used by the query layer — a
    ///   repo is always required): if the repo's shard is not resident, BLOCK on a
    ///   single-flight warm (bounded by `warm_wait`) so the FIRST query to a repo
    ///   this session returns complete results, then search the now-resident shard.
    ///   On warm timeout, fall through and return whatever is resident (partial).
    /// - `repo_filter = None`: search every resident shard and background-warm any
    ///   cold in-scope repo (non-blocking, partial results). Retained for safety;
    ///   the query handlers reject repo-less queries before reaching here.
    ///
    /// Lock order holds: this method takes only `vector_index` (read) for the
    /// search; the warm path takes `repo_dbs` then `vector_index` (write), never the
    /// reverse. The blocking warm runs OUTSIDE the search read guard.
    pub async fn vector_search(
        self: &Arc<Self>,
        query_embedding: &[f32],
        top_k: usize,
        repo_filter: Option<&str>,
        warm_wait: std::time::Duration,
        current: &EmbeddingIdentity,
    ) -> VectorSearchOutcome {
        let identity_key = current.as_key_string();

        if let Some(repo) = repo_filter {
            // Atomic validation + cosine search under ONE read guard. A mismatch
            // returns the observed stale identity without searching any vector.
            let first = {
                self.vector_index.read().await.search_scoped(
                    query_embedding,
                    top_k,
                    repo,
                    &identity_key,
                )
            };
            match first {
                ScopedSearch::Hit(found) => {
                    return VectorSearchOutcome {
                        results: found.results,
                        warming: false,
                    };
                }
                ScopedSearch::IdentityMismatch(stale_identity) => {
                    // Drop read guard before the write lock; identity fence prevents
                    // deleting a newer shard installed in this handoff window.
                    self.vector_index
                        .write()
                        .await
                        .evict_if_identity(repo, &stale_identity);
                }
                ScopedSearch::Cold => {}
            }

            // The ONLY warm path. Pass the query client's exact identity snapshot;
            // warm never re-derives it from mutable settings.
            let _ = tokio::time::timeout(
                warm_wait,
                self.warm_repo_blocking(repo.to_string(), current.clone()),
            )
            .await;

            // Cold re-check is identity-fenced too — never fall back to plain search.
            return match self.vector_index.read().await.search_scoped(
                query_embedding,
                top_k,
                repo,
                &identity_key,
            ) {
                ScopedSearch::Hit(found) => VectorSearchOutcome {
                    results: found.results,
                    warming: false,
                },
                ScopedSearch::IdentityMismatch(_) | ScopedSearch::Cold => VectorSearchOutcome {
                    results: Vec::new(),
                    warming: true,
                },
            };
        }

        // Multi-repo safety path (query handlers currently require a repo, but this
        // retained branch must still be correct): only identity-matching shards can
        // contribute results. `search_all_scoped` and `search_scoped` share the ONE
        // `shard_match` primitive in vector/sharded.rs.
        let scope: Vec<String> = {
            let statuses = self.statuses.read().await;
            statuses.keys().cloned().collect()
        };
        let ScopedAll {
            results,
            cold_repos,
            mismatch_repos,
        } = {
            self.vector_index.read().await.search_all_scoped(
                query_embedding,
                top_k,
                &scope,
                &identity_key,
            )
        };
        let warming = !cold_repos.is_empty() || !mismatch_repos.is_empty();

        // Evict mismatches with the identity observed atomically during validation.
        for (repo, stale_identity) in &mismatch_repos {
            self.vector_index
                .write()
                .await
                .evict_if_identity(repo, stale_identity);
        }

        // Background-warm every cold/mismatched repo with the same query identity.
        // Warm is the sole authority that persists needs_rebuild and queues one
        // `rebuild: true` trigger when the DB marker itself mismatches.
        for repo in cold_repos
            .into_iter()
            .chain(mismatch_repos.into_iter().map(|(repo, _)| repo))
        {
            let engine = Arc::clone(self);
            let identity = current.clone();
            tokio::spawn(async move {
                engine.warm_repo_blocking(repo, identity).await;
            });
        }

        VectorSearchOutcome { results, warming }
    }

    /// Per-repo async warm lock, lazily created. Mirrors `get_repo_lock`.
    async fn get_warm_lock(&self, repo: &str) -> Arc<Mutex<()>> {
        let mut locks = self.warm_locks.lock().await;
        locks
            .entry(repo.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Warm one repo's shard with SINGLE-FLIGHT semantics: at most one
    /// `load_from_db` scan runs per repo at a time, and concurrent callers coalesce
    /// onto it instead of each launching their own.
    ///
    /// N queries hitting the same cold repo would otherwise each run a full
    /// `load_from_db` scan (0.4–1.1 GB for a large repo) and contend for the RocksDB
    /// handle and the vector write lock. Here the first caller acquires the per-repo
    /// warm lock and runs the warm; the rest block on the lock, then re-check
    /// residency under it and return immediately WITHOUT re-scanning — the work the
    /// first caller did is already visible.
    ///
    /// The lock guard is released on every exit path (normal return, the inner
    /// `warm_repo_shard` error path, panic unwind, AND tokio task cancellation — the
    /// future being dropped mid-`.await`, e.g. when the `vector_search` timeout
    /// fires), so a dropped/aborted warm never strands the repo's warm lock.
    async fn warm_repo_blocking(self: &Arc<Self>, repo: String, current: EmbeddingIdentity) {
        let lock = self.get_warm_lock(&repo).await;
        let _guard = lock.lock().await;

        // Coalesce only when the resident shard already matches THIS query identity.
        // A resident shard from another embedding space must still be repaired.
        let current_key = current.as_key_string();
        if self
            .vector_index
            .read()
            .await
            .resident_identity(&repo)
            .as_deref()
            == Some(current_key.as_str())
        {
            return;
        }

        // Resolve the repo generation from live settings, then drop the guard before
        // DB warm work. Embedding identity is NOT read from settings: it came from
        // the real query client and is passed by value above.
        let generation = self.settings_handle.read().await.repo_generation(&repo);
        match warm_repo_shard(
            &self.vector_index,
            &self.repo_dbs,
            &self.data_dir,
            &repo,
            generation,
            &current,
            &[],
        )
        .await
        {
            Ok(WarmResult::IdentityMismatch) => self.request_identity_rebuild(&repo).await,
            Ok(WarmResult::Installed(_) | WarmResult::MutationInProgress) => {}
            Err(e) => {
                warn!(repo = %repo, error = %format!("{e:#}"), "warm: failed closed; shard remains unavailable")
            }
        }
    }

    /// Queue at most one automatic full rebuild per repo until the consumer
    /// completes or fails. `rebuild: true` makes correctness independent of the
    /// best-effort durable needs_rebuild write in the warm mismatch path.
    async fn request_identity_rebuild(&self, repo: &str) {
        if !self.identity_rebuild.reserve(repo) {
            return;
        }
        let trigger = IndexTrigger {
            repo: repo.to_string(),
            changes: None,
            rebuild: true,
        };
        if let Err(e) = self.trigger_tx.send(trigger).await {
            self.identity_rebuild.release(repo);
            warn!(repo = %repo, error = %e, "failed to queue embedding-identity rebuild");
        }
    }

    /// Record an index completion for `repo` and (re)arm the debounced
    /// single-flight cache recompute. Replaces the old fire-on-every-completion
    /// `tokio::spawn` of `compute_and_cache_graph` + `compute_and_cache_stats`.
    ///
    /// Mechanism (see [`CACHE_RECOMPUTE_DEBOUNCE`] / [`RecomputeSlot`]):
    /// - Always stamp `last_completion = now` so the debounce timer is pushed out.
    /// - If a scheduler task already exists for this repo (`scheduled`), do NOT
    ///   spawn a second (single-flight). If that existing task is mid-recompute,
    ///   set `rearm` so exactly one more pass runs afterward (trailing edge); if
    ///   it is still sleeping, the bumped `last_completion` extends its wait — no
    ///   flag needed.
    /// - Otherwise spawn ONE scheduler task that sleeps until the repo is quiet,
    ///   then runs the recompute once (looping only if re-armed mid-recompute).
    ///
    /// Lifecycle: the task holds only `Arc` clones (the `recompute_slots` map, the
    /// repo string, and the Arc-backed `Surreal<Db>` handle), never the engine, so
    /// it cannot keep the engine alive beyond a bounded window. It always retires
    /// (clears `scheduled`) once the burst settles — at most one in-flight task
    /// plus at most one queued pass per repo, so no unbounded timer-task growth.
    /// Schedule one delayed full-scan retry for `repo` (exponential backoff,
    /// at most one pending per repo). See [`retry_backoff`].
    fn schedule_retry(&self, repo: &str, reason: &'static str) {
        let Some(delay) = self.retry_backoff.schedule(repo) else {
            return;
        };
        info!(repo = %repo, reason, delay_secs = delay.as_secs(), "scheduling automatic index retry");
        let tx = self.trigger_tx.clone();
        let backoff = self.retry_backoff.clone();
        let repo = repo.to_owned();
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            backoff.fired(&repo);
            let _ = tx
                .send(IndexTrigger {
                    repo,
                    changes: None,
                    rebuild: false,
                })
                .await;
        });
    }

    fn note_index_completion(&self, repo: &str, db: Surreal<Db>) {
        let mut slots = match self.recompute_slots.lock() {
            Ok(g) => g,
            // A panicked holder can only have poisoned a tiny non-await critical
            // section; recover the map and carry on (best-effort cache refresh).
            Err(poisoned) => poisoned.into_inner(),
        };
        let slot = slots.entry(repo.to_string()).or_default();
        slot.last_completion = Some(Instant::now());
        if slot.scheduled {
            // A task already owns this repo. If it is currently running the
            // aggregation, re-arm so one more pass picks up THIS completion.
            // If it is still in the debounce sleep, the bumped timestamp above
            // already defers it — nothing else to do.
            if slot.recomputing {
                slot.rearm = true;
            }
            return;
        }
        slot.scheduled = true;
        drop(slots); // release before spawning — keep the critical section tiny

        let slots_arc = Arc::clone(&self.recompute_slots);
        let repo_owned = repo.to_string();
        tokio::spawn(async move {
            run_debounced_recompute(slots_arc, repo_owned, db).await;
        });
    }
}

/// The detached per-repo debounced single-flight recompute task. Sleeps until the
/// repo has been quiet for [`CACHE_RECOMPUTE_DEBOUNCE`], runs the graph+stats
/// recompute once (best-effort), then either re-arms (if a completion arrived
/// mid-recompute) or retires. Exactly one of these exists per repo at a time.
async fn run_debounced_recompute(
    slots: Arc<std::sync::Mutex<HashMap<String, RecomputeSlot>>>,
    repo: String,
    db: Surreal<Db>,
) {
    loop {
        // ── Debounce phase: sleep until the repo is quiet. Each iteration
        // re-reads `last_completion`; a completion arriving during the sleep
        // bumps it (via note_index_completion) and we wait again. ──
        loop {
            let wait = {
                let slots_g = match slots.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                match slots_g.get(&repo).and_then(|s| s.last_completion) {
                    Some(last) => CACHE_RECOMPUTE_DEBOUNCE.checked_sub(last.elapsed()),
                    None => None, // shouldn't happen; treat as quiet
                }
            };
            match wait {
                Some(remaining) if !remaining.is_zero() => {
                    tokio::time::sleep(remaining).await;
                }
                _ => break, // quiet long enough (or no stamp) — proceed
            }
        }

        // ── Mark recomputing so a completion now sets `rearm` (trailing edge)
        // instead of being silently absorbed by the debounce timer. ──
        {
            let mut slots_g = match slots.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if let Some(s) = slots_g.get_mut(&repo) {
                s.recomputing = true;
                s.rearm = false;
            }
        }

        // ── Recompute phase: O(repo) full-table aggregation on the SHARED
        // Arc-backed handle (one handle per repo is mandatory — never open a
        // second connection). Best-effort: a failure here just means the serve
        // path's cold-miss fallback recomputes on the next request. ──
        let t = Instant::now();
        if let Err(e) = store::ops::compute_and_cache_graph(&db).await {
            warn!(repo = %repo, error = %format!("{e:#}"), "failed to refresh graph_cache after index");
        }
        if let Err(e) = store::ops::compute_and_cache_stats(&db, &repo).await {
            warn!(repo = %repo, error = %format!("{e:#}"), "failed to refresh stats_cache after index");
        }
        info!(repo = %repo, cache_recompute_ms = t.elapsed().as_millis() as u64,
              "background graph/stats cache recompute complete");

        // ── Decide: re-arm (a completion landed mid-recompute) or retire. This
        // block takes the lock and makes the decision atomically with no `.await`,
        // so it cannot race a concurrent completion. ──
        {
            let mut slots_g = match slots.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if let Some(s) = slots_g.get_mut(&repo) {
                s.recomputing = false;
                if s.rearm {
                    s.rearm = false;
                    continue; // one more debounced pass for the late completion
                }
                s.scheduled = false; // retire: no task owns this repo now
            }
        }
        return;
    }
}

async fn run_consumer(
    engine: Arc<IndexEngine>,
    mut rx: tokio::sync::mpsc::Receiver<IndexTrigger>,
    settings_handle: Arc<RwLock<Settings>>,
) {
    while let Some(trigger) = rx.recv().await {
        // Take a fresh owned snapshot at the top of each iteration so post-boot
        // changes to API keys and other config are reflected immediately.
        // The guard is dropped as soon as `.clone()` completes — it is NOT held
        // across any of the subsequent .await calls below.
        let settings_ref = settings_handle.read().await.clone();

        let repo = trigger.repo.clone();
        let force_rebuild = trigger.rebuild;
        let engine_ref = engine.clone();
        let run_start = Instant::now();

        // Acquire per-repo serialisation lock.
        let lock = engine_ref.get_repo_lock(&repo).await;
        let _guard = lock.lock().await;

        // Create a fresh cancellation token for this run.
        let cancel_token = engine_ref.new_cancel_token(&repo).await;

        // Mark indexing. Reset progress counters so the UI shows the
        // indeterminate pulse (total_files == 0) until the pipeline reports a total.
        {
            let mut statuses = engine_ref.statuses.write().await;
            let status = statuses.entry(repo.clone()).or_default();
            status.state = IndexState::Indexing;
            status.error = None;
            status.indexed_files = 0;
            status.total_files = 0;
            status.phase = IndexPhase::Embedding;
            status.phase_done = 0;
            status.phase_total = 0;
        }

        // Build embedding client — reject if no keys configured.
        let voyage_client = if settings_ref.embedding.api_keys.is_empty() {
            let msg =
                "no embedding API keys configured — cannot index without embeddings".to_string();
            error!(repo = %repo, "{}", msg);
            let mut statuses = engine_ref.statuses.write().await;
            let s = statuses.entry(repo.clone()).or_default();
            s.state = IndexState::Error;
            s.error = Some(msg.clone());
            s.phase = IndexPhase::Idle;
            s.phase_done = 0;
            s.phase_total = 0;
            engine_ref.event_bus.emit(IndexEvent::Failed {
                repo: repo.clone(),
                error: msg,
            });
            engine_ref.identity_rebuild.release(&repo);
            engine_ref.clear_cancel_token(&repo).await;
            continue;
        } else {
            match VoyageClient::new_for_provider(
                crate::embedding::voyage::Provider::parse(&settings_ref.embedding.provider),
                settings_ref.embedding.model.clone(),
                settings_ref.embedding.api_keys.clone(),
                settings_ref.embedding.voyage_base_url.as_deref(),
                settings_ref.embedding.dimensions,
            ) {
                Ok(c) => Some(c),
                Err(e) => {
                    error!(repo = %repo, error = %e, "failed to create voyage client");
                    let mut statuses = engine_ref.statuses.write().await;
                    let s = statuses.entry(repo.clone()).or_default();
                    s.state = IndexState::Error;
                    s.error = Some(e.to_string());
                    s.phase = IndexPhase::Idle;
                    s.phase_done = 0;
                    s.phase_total = 0;
                    engine_ref.event_bus.emit(IndexEvent::Failed {
                        repo: repo.clone(),
                        error: e.to_string(),
                    });
                    engine_ref.clear_cancel_token(&repo).await;
                    continue;
                }
            }
        };

        // Build a progress handle that lets the pipeline write live counts.
        let progress = ProgressHandle {
            statuses: Arc::clone(&engine_ref.statuses),
            repo: repo.clone(),
        };

        // Acquire the SHARED repo DB handle — the same instance the server reads
        // through, so the explorer/query layer sees these writes immediately.
        //
        // `open_or_reset_index` self-heals a corrupt or orphaned-LOCK index dir:
        // `open_db` already rode out transient locks for ~30s, so a failure here
        // means the data is unrecoverable. It deletes + reopens (a full rebuild,
        // API-free via the embedding cache) rather than wedging the repo in Error
        // forever. A live OS handle still holding the LOCK blocks the delete and
        // surfaces the original error — the safety valve against destroying a
        // healthy index. We must NOT call `close_repo_db` from here (it re-acquires
        // the per-repo index lock we already hold → deadlock); the heal removes the
        // cached handle directly.
        let generation = settings_ref.repo_generation(&repo);
        let db = match store::open_or_reset_index(
            &engine_ref.repo_dbs,
            &engine_ref.data_dir,
            &repo,
            generation,
        )
        .await
        {
            Ok((db, was_reset)) => {
                if was_reset {
                    warn!(repo = %repo, "index directory failed to open and was reset; rebuilding from scratch");
                }
                db
            }
            Err(e) => {
                error!(repo = %repo, error = %e, "failed to open repo DB");
                let mut statuses = engine_ref.statuses.write().await;
                let s = statuses.entry(repo.clone()).or_default();
                s.state = IndexState::Error;
                s.error = Some(e.to_string());
                engine_ref.event_bus.emit(IndexEvent::Failed {
                    repo: repo.clone(),
                    error: e.to_string(),
                });
                engine_ref.clear_cancel_token(&repo).await;
                continue;
            }
        };

        // Never run the pipeline concurrently with a background schema migration
        // for this repo (see `store::wait_migration`).
        store::wait_migration(&repo).await;

        // Read per-repo ignored paths from index_meta (fresh each run).
        let per_repo_ignored_paths = store::ops::get_ignored_paths(&db).await.unwrap_or_default();

        // Mask API keys for event display.
        let key_hints: Vec<String> = settings_ref
            .embedding
            .api_keys
            .iter()
            .map(|k| {
                if k.len() > 8 {
                    format!("{}...{}", &k[..4], &k[k.len() - 4..])
                } else {
                    "****".to_string()
                }
            })
            .collect();

        // Check durable needs_rebuild flag (set by v2→v3 migration if gating readback failed).
        let force_rebuild = force_rebuild || {
            match store::ops::get_meta(&db, "needs_rebuild").await {
                Ok(Some(v)) if v == "1" => {
                    info!(repo = %repo, "needs_rebuild flag set — forcing full rebuild");
                    true
                }
                _ => false,
            }
        };

        let pipeline = {
            // total in-flight batches = per-key concurrency × number of keys.
            let n_keys = settings_ref.embedding.api_keys.len().max(1);
            let configured = settings_ref.embedding.embed_concurrency;
            let embed_concurrency = configured * n_keys;

            // Build the embedding cache — uses the model name (and any non-default
            // output dimension) from the client so different model/dimension
            // configurations get isolated cache directories.
            let embed_cache = if let Some(ref client) = voyage_client {
                crate::embedding::cache::EmbeddingCache::new(
                    &engine_ref.embeddings_dir,
                    client.model(),
                    client.dimensions(),
                )
            } else {
                None
            };

            IndexPipeline::new_with_concurrency(
                repo.clone(),
                voyage_client,
                embed_concurrency,
                embed_cache,
            )
            .with_extra_extensions(settings_ref.custom_extensions.clone())
            .with_ignore_filenames(settings_ref.index_ignore_filenames.clone())
            .with_ignore_paths(per_repo_ignored_paths)
            .with_data_dir(engine_ref.data_dir.clone())
        };

        let pipeline_run_start = Instant::now();
        match pipeline
            .run(
                &db,
                trigger.changes,
                force_rebuild,
                Some(&engine_ref.vector_index),
                Some(progress),
                Some(&engine_ref.event_bus),
                &key_hints,
                Some(cancel_token),
            )
            .await
        {
            Ok(stats) => {
                let elapsed_ms = run_start.elapsed().as_millis() as u64;
                info!(repo = %repo, indexed = stats.indexed_files,
                      pipeline_run_wall_ms = pipeline_run_start.elapsed().as_millis() as u64,
                      "indexing complete");
                // Set the observable "done" status and persist the durable
                // timestamp INSIDE a tight scope so the `statuses` write guard is
                // RELEASED before the O(repo) cache recompute below. A status
                // reader (e.g. the UI poll / MCP freshness check / bench
                // wait_for_index) takes `statuses.read()`; if we held the write
                // guard across the ~90s graph/stats aggregation, no reader could
                // OBSERVE state=Idle until that finished — the user would keep
                // seeing "Indexing..." for ~90s after the work was actually done.
                // This honors the project rule: read/write guards on shared state
                // must be dropped before any `.await` on DB or heavy work.
                {
                    let mut statuses = engine_ref.statuses.write().await;
                    let s = statuses.entry(repo.clone()).or_default();
                    s.state = IndexState::Idle;
                    s.indexed_files = stats.indexed_files;
                    s.total_files = stats.total_files;
                    s.last_indexed_at = Some(Utc::now());
                    s.error = None;
                    s.phase = IndexPhase::Idle;
                    s.phase_done = 0;
                    s.phase_total = 0;
                } // <-- statuses write guard dropped here: "done" is now observable.
                // Persist durable timestamp so the MCP tool can check freshness
                // without relying on in-memory state. Runs OFF the statuses lock.
                let _ = set_meta(&db, "last_indexed_at", &chrono::Utc::now().to_rfc3339()).await;
                // Emit Completed BEFORE the cache recompute so subscribers learn
                // the run finished immediately, not after the aggregation.
                engine_ref.event_bus.emit(IndexEvent::Completed {
                    repo: repo.clone(),
                    indexed_files: stats.indexed_files,
                    total_files: stats.total_files,
                    elapsed_ms,
                });
                // Files skipped for a transient reason are queued in pending_retry;
                // retry them automatically instead of waiting for an unrelated
                // trigger. A clean run resets the backoff.
                match store::ops::get_pending_retry(&db).await {
                    Ok(pending) if !pending.is_empty() => {
                        engine_ref.schedule_retry(&repo, "files queued in pending_retry");
                    }
                    _ => engine_ref.retry_backoff.reset(&repo),
                }
                // Clear needs_rebuild flag after successful rebuild.
                if force_rebuild {
                    let _ = db
                        .query("DELETE FROM index_meta WHERE key = 'needs_rebuild'")
                        .await;
                }
                // Refresh the cached call-graph + /index-stats payloads. Both are
                // pure functions of the `calls` / chunk / symbol tables (which
                // change on full AND incremental runs), persisted (keys
                // `graph_cache` / `stats_cache`) so the `/graph` and
                // `/index-stats` endpoints don't each pay full-table aggregation
                // (~80s + ~10s at kernel scale) per request.
                //
                // CRITICAL — this recompute is O(repo) full-table aggregation
                // (~90s at kernel scale) on the SINGLE per-repo RocksDB handle
                // (exclusive per-dir lock — one handle per repo is mandatory).
                // The aggregation does not cooperatively yield, so for its whole
                // duration it PINS that one connection. Firing it inline on every
                // completion — even detached onto a `tokio::spawn` — means an edit
                // burst (N saved files → N completions) runs it back-to-back, and
                // the NEXT incremental's own queries stall behind it on the shared
                // connection. Detaching frees the consumer TASK but NOT the
                // CONNECTION, which is the real contention the ≤10s wall criterion
                // trips over during active editing.
                //
                // So instead of spawning here, we DEBOUNCE + SINGLE-FLIGHT it:
                // mark the repo dirty and (re)arm a per-repo timer that runs the
                // recompute ONCE, only after the repo has been quiet for
                // CACHE_RECOMPUTE_DEBOUNCE. Burst completions coalesce into at
                // most one in-flight recompute plus one queued trailing pass, so
                // the recompute never sits between a trigger and its observable
                // done during a save burst. The /graph and /index-stats serve
                // paths each have a cold-miss fallback that recomputes on-demand,
                // covering the gap before the deferred recompute lands. The
                // recompute STILL fires (the first-ever full-rebuild cache
                // population just lands one debounce window later). See
                // `note_index_completion` / `run_debounced_recompute` and the
                // CONSIDERED-AND-REJECTED note on CACHE_RECOMPUTE_DEBOUNCE.
                //
                // Best-effort throughout: a failure to build/store the cache must
                // NOT affect the run. The handle's `Surreal<Db>` clone is
                // Arc-backed (shares the one cached per-repo connection).
                engine_ref.note_index_completion(&repo, db.clone());

                // PROCESS-PER-PROJECT: write the router-readable sidecar so the
                // router can render this repo's count / last-indexed / model in
                // its list and cold `/index-stats` WITHOUT opening this RocksDB
                // (which would defeat scale-to-zero) or spawning a worker just to
                // display it. Light + fixed-shape (no graph/stats copy). Atomic
                // write; best-effort — a failure must not fail the index run, the
                // router just falls back to its cold placeholder.
                {
                    let (embedding_model, embedding_dim) = {
                        let s = engine_ref.settings_handle.read().await;
                        (
                            s.embedding.model.clone(),
                            s.embedding.dimensions.unwrap_or(0) as u64,
                        )
                    };
                    let meta = crate::router::sidecar::RepoSidecar {
                        file_count: stats.indexed_files,
                        last_indexed_at: Some(Utc::now().to_rfc3339()),
                        state: "indexed".to_string(),
                        embedding_model,
                        embedding_dim,
                        schema: crate::router::sidecar::SIDECAR_SCHEMA,
                    };
                    if let Err(e) =
                        crate::router::sidecar::write_sidecar(&engine_ref.data_dir, &repo, &meta)
                    {
                        warn!(repo = %repo, error = %format!("{e:#}"), "failed to write sidecar (router will use cold placeholder)");
                    }

                    // Also persist the AUX sidecars (graph + files list) so the
                    // router can serve the Index-Explorer detail view (graph +
                    // file list) on detail-open WITHOUT spawning a worker. Stale
                    // between indexes is accepted (router prefers a live worker
                    // when one exists). Best-effort; failure just means the cold
                    // view falls back to empty until the next index. The graph is
                    // the SAME bounded payload `/graph` serves (compute_and_cache_
                    // graph reads it from the just-refreshed cache or recomputes).
                    if let Ok(graph) = crate::store::ops::compute_and_cache_graph(&db).await
                        && let Err(e) = crate::router::sidecar::write_aux_json(
                            &engine_ref.data_dir,
                            &repo,
                            "graph",
                            &graph,
                        )
                    {
                        warn!(repo = %repo, error = %format!("{e:#}"), "failed to write graph sidecar");
                    }
                    // Files list: same bound the /files endpoint uses (2000).
                    if let Ok(files) = crate::store::ops::files_page(&db, &repo, 2000, None).await
                        && let Err(e) = crate::router::sidecar::write_aux_json(
                            &engine_ref.data_dir,
                            &repo,
                            "files",
                            &files,
                        )
                    {
                        warn!(repo = %repo, error = %format!("{e:#}"), "failed to write files sidecar");
                    }
                }
            }
            Err(e) => {
                let is_cancelled = e
                    .downcast_ref::<pipeline::PipelineAbort>()
                    .is_some_and(|a| matches!(a, pipeline::PipelineAbort::Cancelled));
                if is_cancelled {
                    info!(repo = %repo, "indexing cancelled by user");
                    let mut statuses = engine_ref.statuses.write().await;
                    let s = statuses.entry(repo.clone()).or_default();
                    s.state = IndexState::Idle;
                    s.error = None;
                    s.phase = IndexPhase::Idle;
                    s.phase_done = 0;
                    s.phase_total = 0;
                    engine_ref
                        .event_bus
                        .emit(IndexEvent::Cancelled { repo: repo.clone() });
                } else {
                    let err_str = format!("{e:#}");
                    error!(repo = %repo, error = %err_str, "indexing failed");
                    // Mark for full rebuild on next attempt so the index is consistent.
                    let _ = set_meta(&db, "needs_rebuild", "1").await;
                    let mut statuses = engine_ref.statuses.write().await;
                    let s = statuses.entry(repo.clone()).or_default();
                    s.state = IndexState::Error;
                    s.error = Some(err_str.clone());
                    s.phase = IndexPhase::Idle;
                    s.phase_done = 0;
                    s.phase_total = 0;
                    engine_ref.event_bus.emit(IndexEvent::Failed {
                        repo: repo.clone(),
                        error: err_str,
                    });
                    drop(statuses);
                    // Retry transient failures automatically (the repo is
                    // fail-closed until a run succeeds). Fatal embedding errors
                    // (auth/config) wait for a natural trigger instead.
                    if !retry_backoff::is_fatal_index_error(&e) {
                        engine_ref.schedule_retry(&repo, "index run failed");
                    }
                }
            }
        }
        engine_ref.identity_rebuild.release(&repo);
        engine_ref.clear_cancel_token(&repo).await;
    }
}
