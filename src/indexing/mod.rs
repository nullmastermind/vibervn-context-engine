pub mod events;
pub mod frameworks;
pub mod import_resolver;
pub mod pipeline;
pub mod tracker;
pub mod walker;
pub mod watcher;

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
use crate::embedding::voyage::VoyageClient;
use crate::indexing::events::{IndexEvent, IndexEventBus};
use crate::indexing::pipeline::IndexPipeline;
use crate::indexing::tracker::FileChange;
use crate::indexing::watcher::start_watcher;
use crate::store::{self, RepoDbMap};
use crate::store::ops::set_meta;
use crate::vector::{SearchResult, ShardedSearch, ShardedVectorIndex, VectorIndex};

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
}

impl Default for RepoStatus {
    fn default() -> Self {
        Self {
            state: IndexState::Idle,
            indexed_files: 0,
            total_files: 0,
            last_indexed_at: None,
            error: None,
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
}

#[derive(Debug)]
pub struct IndexTrigger {
    pub repo: String,
    pub changes: Option<Vec<FileChange>>, // None = full incremental scan
    /// Force a full rebuild (clear + re-embed everything) regardless of staleness.
    pub rebuild: bool,
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
    active: &[String],
) -> usize {
    let db = match store::get_or_open(repo_dbs, data_dir, repo, generation).await {
        Ok(db) => db,
        Err(e) => {
            warn!(repo = %repo, error = %e, "warm: failed to open DB; skipping repo");
            return 0;
        }
    };
    // DB scan happens here with NO vector_index lock held.
    // First, the staleness stamp = current chunk-row count. Cheap; also tells us
    // whether a persisted shard file is current.
    let stamp = crate::store::ops::count_chunks(&db).await.unwrap_or(0);

    // Fast path: a valid, current persisted shard file → mmap it (near-instant,
    // no SELECT + decode). Its f32 payload is OS-page-cache-resident, off our heap.
    // dim is unknown until we have a shard; probe the model dim from any one chunk
    // via the file header's own dim (open_current validates it against expected).
    // We pass the model dim by reading it from the file header indirectly: try the
    // common dims is brittle, so we instead trust the header's dim and only reject
    // on a mismatch with a known dim. Here we accept the file's own dim by passing
    // it through a two-step: peek is folded into open_current (expected_dim=0 means
    // "accept the header dim"). See shard_file::open_current.
    match crate::vector::shard_file::open_current(data_dir, repo, 0, stamp) {
        Ok(Some((shard, generation_loaded))) => {
            let count = shard.len();
            if count > 0 {
                let mut vi = vector_index.write().await;
                vi.install_shard(repo, shard, active);
                // Reap stale generations now that the new one is installed; keep
                // only the generation we just mapped (under the same write lock
                // that governs CURRENT, so no reader/reaper race).
                crate::vector::shard_file::reap_stale_generations(
                    data_dir, repo, &[generation_loaded],
                );
                info!(repo = %repo, count, generation = generation_loaded, "warm: mmap'd persisted shard (no DB scan)");
                return count;
            }
        }
        Ok(None) => {} // no usable file — build from DB below
        Err(e) => warn!(repo = %repo, error = %e, "warm: shard file open failed; rebuilding from DB"),
    }

    // Slow path: build from the chunk table (the existing SELECT + decode), then
    // persist a new generation so subsequent warms mmap it.
    let shard = match VectorIndex::load_from_db(&db).await {
        Ok(vi) => vi,
        Err(e) => {
            warn!(repo = %repo, error = %e, "warm: failed to load shard from DB; skipping repo");
            return 0;
        }
    };
    let count = shard.len();
    if count == 0 {
        return 0;
    }
    // Persist the built shard to a fresh generation + flip CURRENT (win32-safe:
    // no existing mapped file is touched). Best-effort — a write failure just
    // means the next warm rebuilds from DB again.
    let persisted = match crate::vector::shard_file::write_new_generation(data_dir, repo, &shard, stamp) {
        Ok(g) => Some(g),
        Err(e) => {
            warn!(repo = %repo, error = %e, "warm: failed to persist shard file (will rebuild next warm)");
            None
        }
    };
    // Re-open the just-written file as an mmap so the resident shard is page-cache
    // backed (not the heap copy we just built). Falls back to the heap shard if the
    // re-open fails for any reason.
    let mut vi = vector_index.write().await;
    if let Some(g) = persisted
        && let Ok(Some((mmap_shard, _))) = crate::vector::shard_file::open_current(data_dir, repo, shard.dim(), stamp)
    {
        vi.install_shard(repo, mmap_shard, active);
        crate::vector::shard_file::reap_stale_generations(data_dir, repo, &[g]);
        info!(repo = %repo, count, generation = g, "warm: built from DB, persisted + mmap'd shard");
    } else {
        // Heap fallback: install the in-RAM shard we built.
        vi.install_shard(repo, shard, active);
        info!(repo = %repo, count, "warm: installed in-RAM shard (persist/mmap unavailable)");
    }
    count
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
        let cap_bytes = settings
            .vector_resident_cap_mb
            .saturating_mul(1024 * 1024);

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
        });

        // Initialise status entries.
        {
            let mut statuses = engine.statuses.write().await;
            for repo in &settings.repos {
                statuses.insert(crate::store::normalize_repo_path(repo), RepoStatus::default());
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
            tokio::spawn(async move {
                seed_statuses_from_db(&statuses_bg, &repo_dbs_bg, &data_dir_bg, &repos_bg, &generations_bg).await;
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
            statuses.get(&repo).map(|s| s.state == IndexState::Indexing).unwrap_or(false)
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
    ) -> VectorSearchOutcome {
        // Single-repo scope: block-warm a cold repo before searching so the first
        // query of the session returns complete results instead of empty.
        let mut warming = false;
        if let Some(repo) = repo_filter {
            let resident = self.vector_index.read().await.is_resident(repo);
            if !resident {
                // Bound the wait so a huge/slow repo never hangs the request. On
                // timeout the warm future is dropped (releasing its warm lock via
                // guard drop) and we proceed with whatever is resident — a later
                // query re-attempts the warm.
                let _ = tokio::time::timeout(warm_wait, self.warm_repo_blocking(repo.to_string()))
                    .await;
                // Re-check residency AFTER the bounded warm. If the shard is STILL
                // not resident, the warm-wait expired before load_from_db finished
                // (e.g. a multi-GB shard, or it was evicted under memory pressure).
                // The search below will return empty — but that empty means "still
                // warming", NOT "the index contains nothing". Surface that distinction
                // so callers retry instead of concluding the codebase is empty.
                warming = !self.vector_index.read().await.is_resident(repo);
            }
        }

        let scope: Vec<String> = match repo_filter {
            Some(repo) => vec![repo.to_string()],
            None => {
                let statuses = self.statuses.read().await;
                statuses.keys().cloned().collect()
            }
        };

        let ShardedSearch { results, cold_repos } = {
            // READ lock — concurrent searches run in parallel. `search` bumps
            // per-shard atomic recency stamps under this shared guard.
            let index = self.vector_index.read().await;
            index.search(query_embedding, top_k, repo_filter, &scope)
        }; // read lock dropped HERE — before spawning any warm task

        // Only reached for the `None` (search-all) path now, since a `Some` cold
        // repo was already block-warmed above. Background-warm any cold in-scope
        // repo so subsequent queries hit it; non-blocking (results already returned).
        for repo in cold_repos {
            let engine = Arc::clone(self);
            tokio::spawn(async move {
                engine.warm_repo_blocking(repo).await;
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
    async fn warm_repo_blocking(self: &Arc<Self>, repo: String) {
        let lock = self.get_warm_lock(&repo).await;
        let _guard = lock.lock().await;

        // Coalesce: a prior holder may have already warmed this repo while we waited.
        if self.vector_index.read().await.is_resident(&repo) {
            return;
        }

        // Warm with an empty active set: the cap may freely evict LRU shards to make
        // room. In-flight searches are NOT at risk — they hold a read guard and
        // return owned results (cloned ChunkIds); install/evict take the write guard
        // afterwards. The warmed repo is protected internally by install_shard.
        //
        // Resolve the repo's current generation from live settings (read guard
        // dropped before the await on warm work). Safe to read mid-run: generation
        // only advances after a delete dropped the cached handle + evicted the shard,
        // so a warm here always targets the live directory.
        let generation = self.settings_handle.read().await.repo_generation(&repo);
        warm_repo_shard(
            &self.vector_index,
            &self.repo_dbs,
            &self.data_dir,
            &repo,
            generation,
            &[],
        )
        .await;
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
        }

        // Build embedding client — reject if no keys configured.
        let voyage_client = if settings_ref.embedding.api_keys.is_empty() {
            let msg = "no embedding API keys configured — cannot index without embeddings".to_string();
            error!(repo = %repo, "{}", msg);
            let mut statuses = engine_ref.statuses.write().await;
            let s = statuses.entry(repo.clone()).or_default();
            s.state = IndexState::Error;
            s.error = Some(msg.clone());
            engine_ref.event_bus.emit(IndexEvent::Failed {
                repo: repo.clone(),
                error: msg,
            });
            engine_ref.clear_cancel_token(&repo).await;
            continue;
        } else {
            match VoyageClient::new(
                settings_ref.embedding.model.clone(),
                settings_ref.embedding.api_keys.clone(),
                settings_ref.embedding.voyage_base_url.as_deref(),
            ) {
                Ok(c) => Some(c),
                Err(e) => {
                    error!(repo = %repo, error = %e, "failed to create voyage client");
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
        let db = match store::open_or_reset_index(&engine_ref.repo_dbs, &engine_ref.data_dir, &repo, generation).await {
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

        // Read per-repo ignored paths from index_meta (fresh each run).
        let per_repo_ignored_paths = store::ops::get_ignored_paths(&db)
            .await
            .unwrap_or_default();

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

            // Build the embedding cache — uses the model name from settings so
            // different model configurations get isolated cache directories.
            let embed_cache = if let Some(ref client) = voyage_client {
                crate::embedding::cache::EmbeddingCache::new(
                    &engine_ref.embeddings_dir,
                    client.model(),
                )
            } else {
                None
            };

            IndexPipeline::new_with_concurrency(repo.clone(), voyage_client, embed_concurrency, embed_cache)
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
                // Clear needs_rebuild flag after successful rebuild.
                if force_rebuild {
                    let _ = db.query("DELETE FROM index_meta WHERE key = 'needs_rebuild'").await;
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
            }
            Err(e) => {
                let is_cancelled = e.downcast_ref::<pipeline::PipelineAbort>()
                    .is_some_and(|a| matches!(a, pipeline::PipelineAbort::Cancelled));
                if is_cancelled {
                    info!(repo = %repo, "indexing cancelled by user");
                    let mut statuses = engine_ref.statuses.write().await;
                    let s = statuses.entry(repo.clone()).or_default();
                    s.state = IndexState::Idle;
                    s.error = None;
                    engine_ref.event_bus.emit(IndexEvent::Cancelled {
                        repo: repo.clone(),
                    });
                } else {
                    let err_str = format!("{e:#}");
                    error!(repo = %repo, error = %err_str, "indexing failed");
                    // Mark for full rebuild on next attempt so the index is consistent.
                    let _ = set_meta(&db, "needs_rebuild", "1").await;
                    let mut statuses = engine_ref.statuses.write().await;
                    let s = statuses.entry(repo.clone()).or_default();
                    s.state = IndexState::Error;
                    s.error = Some(err_str.clone());
                    engine_ref.event_bus.emit(IndexEvent::Failed {
                        repo: repo.clone(),
                        error: err_str,
                    });
                }
            }
        }
        engine_ref.clear_cancel_token(&repo).await;
    }
}

#[cfg(test)]
mod load_repos_tests {
    use super::*;
    use tempfile::TempDir;

    /// Seed `n` chunk rows (each with a non-empty 4-d embedding) into `repo`'s DB,
    /// writing THROUGH the shared `repo_dbs` map (one cached handle per repo, like
    /// production). RocksDB holds an exclusive per-directory lock, so a second
    /// uncached `open_db` on the same path would deadlock on the lock file —
    /// seeding through `get_or_open` keeps a single handle, mirroring real usage.
    async fn seed_repo(repo_dbs: &RepoDbMap, home: &std::path::Path, repo: &str, n: usize) {
        let db = store::get_or_open(repo_dbs, home, repo, 0).await.expect("get_or_open");
        for i in 0..n {
            let q = format!(
                "CREATE chunk SET file = '{repo}/f{i}.rs', line_start = 1, line_end = 2, \
                 content = 'x', embedding = [0.1, 0.2, 0.3, 0.4], symbol_ref = NONE;"
            );
            db.query(&q).await.expect("seed chunk");
        }
    }

    /// Warming EACH configured repo into its own resident shard (when the cap
    /// allows) installs an independent shard per repo, not just the first. Two
    /// repos seeded with 1 and 2 chunks → both shards resident, total 3 vectors
    /// searchable across shards. This is the per-repo lazy-warm path now used on
    /// first query (boot no longer eagerly warms).
    #[tokio::test]
    async fn loads_all_repos_not_just_first() {
        let home = TempDir::new().expect("tempdir");
        let repo_one = "/proj/repo_one".to_string();
        let repo_two = "/proj/repo_two".to_string();

        // Shared map used for BOTH seeding and warming — exactly one handle per repo.
        let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
        seed_repo(&repo_dbs, home.path(), &repo_one, 1).await;
        seed_repo(&repo_dbs, home.path(), &repo_two, 2).await;

        // Large cap so both repos stay resident after warming.
        let vector_index = Arc::new(RwLock::new(ShardedVectorIndex::new(1024 * 1024 * 1024)));
        warm_repo_shard(&vector_index, &repo_dbs, home.path(), &repo_one, 0, &[]).await;
        warm_repo_shard(&vector_index, &repo_dbs, home.path(), &repo_two, 0, &[]).await;

        let vi = vector_index.read().await;
        assert!(vi.is_resident(&repo_one), "repo_one shard must be resident");
        assert!(vi.is_resident(&repo_two), "repo_two shard must be resident");
        assert_eq!(
            vi.resident_repo_count(),
            2,
            "expected both repos warmed into shards, not just the first"
        );
    }

    /// Seed `n` file_meta rows for `repo`, writing through the shared `repo_dbs`
    /// map (single cached handle per repo — see [`seed_repo`] for why RocksDB
    /// requires this).
    async fn seed_file_meta(repo_dbs: &RepoDbMap, home: &std::path::Path, repo: &str, n: usize) {
        let db = store::get_or_open(repo_dbs, home, repo, 0).await.expect("get_or_open");
        for i in 0..n {
            let path = format!("{repo}/f{i}.rs");
            db.query("CREATE file_meta SET path = $path, mtime = 0, size = 1, repo = $repo, chunk_count = 1;")
                .bind(("path", path))
                .bind(("repo", repo.to_string()))
                .await
                .expect("seed file_meta");
        }
    }

    /// After a restart, a repo indexed in a prior session must show its persisted
    /// file count — not the zeroed default. A never-indexed repo must stay at 0
    /// so the UI can render a "Not indexed" placeholder.
    #[tokio::test]
    async fn seeds_status_from_persisted_file_meta() {
        let home = TempDir::new().expect("tempdir");
        let indexed_raw = "/proj/indexed".to_string();
        let empty_raw = "/proj/empty".to_string();
        let indexed = store::normalize_repo_path(&indexed_raw);
        let empty = store::normalize_repo_path(&empty_raw);

        // Shared map for seeding AND the seed-status call — one handle per repo.
        let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
        seed_file_meta(&repo_dbs, home.path(), &indexed, 5).await;
        // `empty` gets a DB (cached) but no file_meta rows.
        let _ = store::get_or_open(&repo_dbs, home.path(), &empty, 0).await.expect("get_or_open");

        let statuses: Arc<RwLock<HashMap<String, RepoStatus>>> =
            Arc::new(RwLock::new(HashMap::new()));
        {
            let mut m = statuses.write().await;
            m.insert(indexed.clone(), RepoStatus::default());
            m.insert(empty.clone(), RepoStatus::default());
        }

        seed_statuses_from_db(&statuses, &repo_dbs, home.path(), &[indexed_raw, empty_raw], &HashMap::new())
            .await;

        let m = statuses.read().await;
        assert_eq!(m[&indexed].indexed_files, 5, "indexed repo must restore its file count");
        assert_eq!(m[&empty].indexed_files, 0, "never-indexed repo must stay at 0");
    }

    /// A run that has already advanced a repo's status by the time the seed task
    /// runs must not be clobbered back to the persisted (possibly stale) count.
    #[tokio::test]
    async fn seed_does_not_clobber_live_run() {
        let home = TempDir::new().expect("tempdir");
        let repo_raw = "/proj/live".to_string();
        let repo = store::normalize_repo_path(&repo_raw);
        let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
        seed_file_meta(&repo_dbs, home.path(), &repo, 5).await;

        let statuses: Arc<RwLock<HashMap<String, RepoStatus>>> =
            Arc::new(RwLock::new(HashMap::new()));
        {
            let mut m = statuses.write().await;
            m.insert(
                repo.clone(),
                RepoStatus { state: IndexState::Indexing, ..Default::default() },
            );
        }

        seed_statuses_from_db(&statuses, &repo_dbs, home.path(), &[repo_raw], &HashMap::new()).await;

        let m = statuses.read().await;
        assert_eq!(m[&repo].state, IndexState::Indexing, "in-flight run must survive the seed");
        assert_eq!(m[&repo].indexed_files, 0, "seed must not overwrite a live run's numerator");
    }

    /// Single-flight coalescing: concurrent `warm_repo_blocking` calls for the same
    /// cold repo must result in exactly ONE `load_from_db` scan + install — the
    /// later callers block on the per-repo warm lock, then observe the repo already
    /// resident and return without re-scanning. We assert the end state (resident,
    /// correct vector count) and that the shard was installed once (no duplicate /
    /// doubled vectors), which is the observable guarantee of coalescing.
    #[tokio::test]
    async fn warm_blocking_is_single_flight() {
        let home = TempDir::new().expect("tempdir");
        let repo = "/proj/warm".to_string();

        let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
        seed_repo(&repo_dbs, home.path(), &repo, 3).await;

        // Minimal engine sharing the SAME repo_dbs map used for seeding (one handle).
        let settings = crate::config::Settings {
            repos: vec![repo.clone()],
            ..Default::default()
        };
        let settings_handle = Arc::new(RwLock::new(settings.clone()));
        let engine = IndexEngine::start(
            home.path().to_path_buf(),
            home.path().join("embeddings"),
            &settings,
            repo_dbs.clone(),
            settings_handle,
            false,
        )
        .await;

        // Fire several concurrent warms for the same cold repo.
        let mut handles = Vec::new();
        for _ in 0..5 {
            let e = engine.clone();
            let r = repo.clone();
            handles.push(tokio::spawn(async move {
                e.warm_repo_blocking(r).await;
            }));
        }
        for h in handles {
            h.await.expect("warm task");
        }

        let vi = engine.vector_index.read().await;
        assert!(vi.is_resident(&repo), "repo must be resident after warm");
        // Exactly the seeded 3 vectors — coalescing means no doubled inserts.
        let mut q = vec![0.0f32; 4];
        q[0] = 1.0;
        let out = vi.search(&q, 100, Some(&repo), std::slice::from_ref(&repo));
        assert_eq!(
            out.results.len(),
            3,
            "single-flight warm must install the shard once (3 seeded vectors, not doubled)"
        );
    }

    /// A shard that is NOT resident after the warm attempt yields warming=true with
    /// empty results — the "retry, not empty" signal. Forced deterministically with a
    /// 0-chunk repo: warm_repo_shard loads nothing (count==0) and never installs a
    /// shard, so it stays non-resident regardless of timing — exactly the condition
    /// `warming = !is_resident(repo)` detects after the bounded warm.
    #[tokio::test(flavor = "multi_thread")]
    async fn vector_search_signals_warming_when_shard_not_resident() {
        let home = TempDir::new().expect("tempdir");
        let repo = "/proj/cold".to_string();
        let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
        // 0 chunks → warm installs no shard → non-resident after the warm attempt.
        seed_repo(&repo_dbs, home.path(), &repo, 0).await;
        let settings = crate::config::Settings { repos: vec![repo.clone()], ..Default::default() };
        let settings_handle = Arc::new(RwLock::new(settings.clone()));
        let engine = IndexEngine::start(
            home.path().to_path_buf(), home.path().join("embeddings"),
            &settings, repo_dbs.clone(), settings_handle, false,
        ).await;

        let q = vec![1.0f32, 0.0, 0.0, 0.0];
        let outcome = engine
            .vector_search(&q, 10, Some(&repo), std::time::Duration::from_secs(5))
            .await;
        assert!(outcome.results.is_empty(), "non-resident shard search returns empty");
        assert!(outcome.warming, "non-resident shard after warm attempt must signal warming=true");
    }

    /// A resident shard that genuinely matches nothing yields warming=false — a real
    /// empty, NOT a warming state. (Here the shard is warmed first via a generous wait.)
    #[tokio::test(flavor = "multi_thread")]
    async fn vector_search_resident_empty_is_not_warming() {
        let home = TempDir::new().expect("tempdir");
        let repo = "/proj/resident".to_string();
        let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
        seed_repo(&repo_dbs, home.path(), &repo, 3).await;
        let settings = crate::config::Settings { repos: vec![repo.clone()], ..Default::default() };
        let settings_handle = Arc::new(RwLock::new(settings.clone()));
        let engine = IndexEngine::start(
            home.path().to_path_buf(), home.path().join("embeddings"),
            &settings, repo_dbs.clone(), settings_handle, false,
        ).await;

        // Warm the shard explicitly so it is resident before the search.
        engine.warm_repo_blocking(repo.clone()).await;
        assert!(engine.vector_index.read().await.is_resident(&repo), "precondition: resident");

        // Query with a generous wait; the shard is resident so warming must be false.
        // top_k=0 forces an empty result set on a resident shard (genuine empty).
        let q = vec![1.0f32, 0.0, 0.0, 0.0];
        let outcome = engine
            .vector_search(&q, 0, Some(&repo), std::time::Duration::from_secs(10))
            .await;
        assert!(outcome.results.is_empty(), "top_k=0 yields empty on a resident shard");
        assert!(!outcome.warming, "resident shard must NOT signal warming, even when empty");
    }
}
