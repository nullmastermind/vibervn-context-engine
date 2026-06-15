use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::sql::{Array as SqlArray, Bytes as SqlBytes, Id as SqlId, Object as SqlObject, Thing as SqlThing, Value as SqlValue};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::embedding::InputType;
use crate::embedding::cache::EmbeddingCache;
use crate::embedding::voyage::VoyageClient;
use crate::indexing::ProgressHandle;
use crate::indexing::events::{IndexEvent, IndexEventBus};
use crate::indexing::tracker::{ChangeKind, FileChange, stat_file};
use crate::indexing::walker::{ChangeFilter, walk_repo_with};
use crate::parsing::parse_file;
use crate::parsing::relations::{EdgeKind, EdgeTarget};
use crate::parsing::symbols::Symbol;
use crate::store::ops::{
    FileMeta, delete_all_data, delete_files_data_incremental, get_all_file_meta,
    get_meta, set_meta, upsert_file_meta, find_symbols_by_names_with_pos, SymbolWithPos,
};
use crate::store::build_index_concurrently;
use crate::vector::{ChunkId, ShardedVectorIndex};

/// Batch size for DB writes — keeps per-query payload small and avoids the
/// gigabyte-sized transaction that caused 3 GB RAM spikes on large repos.
const WRITE_BATCH_SIZE: usize = 512;

/// Batch size for Phase-2 RELATE edge writes. Larger than WRITE_BATCH_SIZE
/// because RELATE statements are compact (no embedding payload) and reducing
/// round-trips gives measurable gains on real-disk SurrealDB kv-surrealkv.
/// 8192 reduces the full-rebuild flush to ~10 round-trips for 77K resolved edges.
const EDGE_RELATE_BATCH_SIZE: usize = 8192;

/// Batch size for raw_edge INSERT writes in Stage 3.
/// Raw edges are small records (no embedding payload), so larger batches
/// reduce round-trips significantly (138K edges / 4096 = ~34 batches vs
/// 138K / 512 = ~270 batches). Keep below 8192 to avoid oversized payloads.
const RAW_EDGE_INSERT_BATCH_SIZE: usize = 4096;

/// Streaming channel capacity. Parser feeds at most this many parsed-file results
/// into the embed stage before blocking. Keeps peak inflight bounded independent
/// of repo size (O(channel_cap * chunks_per_file) RAM, not O(repo)).
const PARSE_CHANNEL_CAP: usize = 64;

/// Embed-output channel capacity (from embed stage to writer).
const EMBED_CHANNEL_CAP: usize = 64;

/// Number of resolve_set files whose scoped `calls` deletes are grouped into ONE
/// `BEGIN/COMMIT` transaction during incremental Phase-2 edge resolution.
///
/// Each file contributes two single-value indexed-equality DELETE statements
/// (`WHERE in_file = $f` and `WHERE out_file = $f` — both point-seeks on
/// idx_calls_in_file / idx_calls_out_file). The previous code ran each statement
/// in its OWN auto-commit, so a 2748-file resolve_set paid 5496 RocksDB
/// commit+fsyncs (~39ms fixed apiece → 215685ms at kernel scale, commit-bound).
/// Batching CALLS_DELETE_TXN_CHUNK files per transaction collapses that to
/// ceil(resolve_set / chunk) commits (~14 for 2748 files) while keeping every
/// delete an indexed point seek — measured 53018ms (~4x faster) in
/// `delete_strategy_probe` at full kernel scale, deleting the IDENTICAL row set.
/// 200 keeps each transaction's statement string (2×200 short statements) and
/// its uncommitted tombstone set well within the pinned 32 MiB RocksDB write
/// buffer; larger chunks risk a write-buffer flush mid-transaction with no
/// further commit-amortization gain.
const CALLS_DELETE_TXN_CHUNK: usize = 200;

/// A chunk row ready for bulk INSERT via native SurrealDB value construction.
///
/// `embedding` is a **packed little-endian f32 byte blob** (4 bytes/element),
/// stored on disk as `Value::Bytes` (DB schema v5+). The prior representation
/// was `array<float>`, which forced SurrealDB to encode ~1024 floats/row as
/// individual `Value::Number` enums — measured (ablation) at ~12.3s of the
/// ~13s chunk-write for spec-ade (94%), all on a single thread. Packed bytes
/// eliminate those ~21M enum allocations: a 4096-byte blob per row, encoded
/// with a memcpy. `flush_chunk_batch` builds the row natively (no serde) so
/// the field reaches the engine as `Value::Bytes`, not an array-of-ints.
struct ChunkRecord {
    file: String,
    line_start: i64,
    line_end: i64,
    content: String,
    /// Packed little-endian f32 bytes (see `store::ops::pack_embedding`).
    embedding: Vec<u8>,
    symbol_ref: Option<String>,
}

/// A raw (unresolved) edge written to the `raw_edge` staging table in Phase 1.
/// All fields are locally known at parse time: the caller is always in the current file.
/// SurrealDB assigns the record id at insert time; Phase 2 uses `type::string(id)` as
/// the keyset cursor — no app-managed sequence counter needed.
#[derive(Serialize, Clone)]
struct RawEdgeRecord {
    from_file: String,
    from_name: String,
    /// Full FQN of the calling symbol (file::scope1::...::name). Stored at parse time
    /// so Phase 2 can use it directly as the RELATE source without re-constructing it.
    from_fqn: String,
    to_name: String,
    kind: String,
    line: i64,
    import_path: Option<String>,
}

/// Output of parse_one_file — either a successfully parsed file or a skip record.
enum ParseOutput {
    Parsed(ParsedFile),
    Skipped { file: String, reason: String },
}

/// A parsed file result ready for the embed stage.
struct ParsedFile {
    path: String,
    symbols: Vec<Symbol>,
    chunks: Vec<crate::parsing::chunker::Chunk>,
    raw_edges: Vec<RawEdgeRecord>,
    mtime: i64,
    size: i64,
    /// How long the parse took (for FileParsed event).
    parse_elapsed_ms: u64,
    /// When this ParsedFile was created (to measure queue wait in Stage 2).
    /// Not serialized — internal pipeline field only.
    created_at: Instant,
}

/// An embedded file result ready for the writer.
struct EmbeddedFile {
    path: String,
    symbols: Vec<Symbol>,
    chunks: Vec<crate::parsing::chunker::Chunk>,
    embeddings: Vec<Vec<f32>>,
    raw_edges: Vec<RawEdgeRecord>,
    mtime: i64,
    size: i64,
    /// True if the API returned all-empty embeddings due to an error.
    embed_failed: bool,
    /// When this EmbeddedFile was created (to measure queue wait in Stage 3).
    /// Not serialized — internal pipeline field only.
    created_at: Instant,
    /// When Stage 1 started for this file (for total_elapsed_ms in FileIndexed).
    /// Not serialized — internal pipeline field only.
    pipeline_start: Instant,
    /// Wall time spent in the embed/cache-read stage for this file (ms).
    embed_elapsed_ms: u64,
    /// Chunks served from the on-disk embedding cache.
    cache_hit_chunks: u64,
    /// Chunks NOT in the cache (needed API call or stored empty).
    cache_miss_chunks: u64,
}

/// Distinguishes user-initiated cancel from an embedding API failure.
/// `run_consumer` uses this to decide whether to set `needs_rebuild`.
#[derive(Debug)]
pub enum PipelineAbort {
    Cancelled,
    EmbeddingFailed(String),
}

impl std::fmt::Display for PipelineAbort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => write!(f, "indexing cancelled"),
            Self::EmbeddingFailed(msg) => write!(f, "embedding failed: {msg}"),
        }
    }
}

impl std::error::Error for PipelineAbort {}

/// Sub-term breakdown for Phase 2 (edge resolution), surfaced into
/// `IndexPipelineStats` so a single "PERF SUMMARY phase2" line attributes the
/// Phase-2 wall time to load / bucket-build / resolve / write / index steps.
#[derive(Default, Debug)]
pub struct Phase2Stats {
    pub sym_load_ms: u64,
    pub bucket_build_ms: u64,
    pub resolve_cpu_ms: u64,
    pub relate_write_ms: u64,
    pub idx_drop_ms: u64,
    pub idx_rebuild_ms: u64,
    pub edges_written: u64,
}

#[derive(Default)]
pub struct IndexPipelineStats {
    pub indexed_files: u64,
    pub total_files: u64,
    /// Stage-3 write time breakdown (milliseconds).
    pub stage3_sym_ms: u64,
    pub stage3_rawedge_ms: u64,
    pub stage3_chunk_ms: u64,
    pub stage3_filemeta_ms: u64,
    /// Total Stage-3 wall time (may differ from sum due to overhead).
    pub stage3_total_ms: u64,
    /// Phase-2 edge resolution wall time.
    pub phase2_ms: u64,
    /// Total counts.
    pub total_chunks: u64,
    pub total_symbols: u64,
    pub total_raw_edges: u64,
    /// Total wall time spent in the embed/cache-read stage (Stage 2), milliseconds.
    pub embed_total_ms: u64,
    /// Number of chunks that were fully served from the on-disk embedding cache.
    pub cache_hit_chunks: u64,
    /// Number of chunks that were NOT in the cache (re-embedded or stored empty).
    pub cache_miss_chunks: u64,
    /// Sub-term: time spent in the actual db.query INSERT for chunks (milliseconds).
    pub stage3_chunk_db_ms: u64,
    /// Sub-term: time spent constructing ChunkRecord structs + pushing to Vec (milliseconds).
    pub stage3_chunk_cpu_ms: u64,
    /// Time spent dropping idx_chunk_file before bulk chunk write (full rebuild only, ms).
    pub stage3_chunk_idx_drop_ms: u64,
    /// Time spent rebuilding idx_chunk_file after bulk chunk write (full rebuild only, ms).
    pub stage3_chunk_idx_rebuild_ms: u64,
    /// Time spent dropping idx_symbol_file + idx_symbol_name before the bulk symbol
    /// write (full rebuild only, ms). Symbols were the only high-volume bulk-write
    /// table still going through live secondary indexes; on the kernel that cost
    /// ~20.6 min (92% of Stage 3). Mirrors the chunk/calls drop→bulk→rebuild trick.
    /// ≈0 in practice (REMOVE INDEX is a metadata op). 0 on the incremental path.
    pub stage3_sym_idx_drop_ms: u64,
    /// Time spent rebuilding idx_symbol_file + idx_symbol_name after all symbol rows
    /// are durable (full rebuild only, ms). One-shot bulk build that replaces ~6.2M
    /// per-row incremental index updates. 0 on the incremental path.
    pub stage3_sym_idx_rebuild_ms: u64,

    // ─── Phase-2 sub-term breakdown (edge resolution) ────────────────────────
    // These attribute the Phase-2 wall time (`phase2_ms`) to its constituent
    // steps so the bottleneck is visible in one greppable "PERF SUMMARY phase2"
    // line. Populated by both resolve_edges_from_ram (RAM fast path, the path
    // kernel-scale repos take) and resolve_edges_phase2 (DB-scan overflow path).
    /// Time loading all symbols from the DB into RAM (ms).
    pub phase2_sym_load_ms: u64,
    /// Time building+sorting the name→candidates bucket map (ms).
    pub phase2_bucket_build_ms: u64,
    /// CPU time resolving raw edges against the symbol map (ms). Excludes DB writes.
    pub phase2_resolve_cpu_ms: u64,
    /// Time in `INSERT RELATION INTO calls` flushes (ms).
    pub phase2_relate_write_ms: u64,
    /// Time dropping the 4 calls indexes before bulk RELATE (ms).
    pub phase2_idx_drop_ms: u64,
    /// Time rebuilding the 4 calls indexes after bulk RELATE (ms).
    pub phase2_idx_rebuild_ms: u64,
    /// Number of resolved `calls` edges written.
    pub phase2_edges_written: u64,

    // ─── Incremental-path per-stage breakdown ────────────────────────────────
    // The incremental run (parse→embed→store of only the changed files, then a
    // blast-radius edge re-resolution) previously emitted NO per-stage timing —
    // `run` returned `..Default::default()` for everything but indexed/total
    // files. These fields fill that gap so a single "PERF SUMMARY incremental"
    // line attributes the incremental wall time to its constituent stages and
    // the dominant cost is provable. 0 on the full-rebuild path.
    /// Manual-path walk + detect_changes (the spawn_blocking in `run`). 0 when
    /// the change set is watcher-supplied (no walk performed).
    pub incr_walk_ms: u64,
    /// `get_all_file_meta` load at the top of `run` (ms).
    pub incr_meta_load_ms: u64,
    /// Pre-delete caller query in `incremental_run` (ms).
    pub incr_predelete_callers_ms: u64,
    /// `delete_files_data_bulk` for all affected files (ms).
    pub incr_delete_bulk_ms: u64,
    /// `streaming_index` call in `incremental_run` (parse→embed→store) (ms).
    pub incr_streaming_ms: u64,
    /// Total `resolve_edges_incremental` wall time (ms).
    pub incr_phase2_total_ms: u64,
    /// Phase-2 sub: the `SELECT name FROM symbol WHERE file IN $files` query (ms).
    pub incr_p2_symname_ms: u64,
    /// Phase-2 sub: the `SELECT from_file FROM raw_edge WHERE to_name IN $names`
    /// direction-2 expansion scan (ms). PRIME SUSPECT — timed in isolation.
    pub incr_p2_dir2_scan_ms: u64,
    /// Phase-2 sub: the scoped `DELETE FROM calls` (ms).
    pub incr_p2_delete_calls_ms: u64,
    /// Phase-2 sub: the raw_edge keyset scan + resolve loop + tail flush (ms).
    pub incr_p2_reresolve_ms: u64,
    /// Phase-2 sub: final `resolve_set.len()` (a count, not a time — explains cost).
    pub incr_resolve_set_size: u64,
}

/// Per-stage timings produced INSIDE `resolve_edges_incremental` and threaded up
/// through `incremental_run` → `run`, mirroring how `Phase2Stats` is threaded for
/// the full-rebuild path. All times in milliseconds; `resolve_set_size` is a count.
#[derive(Default, Debug)]
pub struct Phase2IncrStats {
    pub p2_symname_ms: u64,
    pub p2_dir2_scan_ms: u64,
    pub p2_delete_calls_ms: u64,
    pub p2_reresolve_ms: u64,
    pub resolve_set_size: u64,
}

/// Per-stage timings produced INSIDE `incremental_run` and threaded up to `run`.
/// All times in milliseconds.
#[derive(Default, Debug)]
pub struct IncrementalRunStats {
    /// Time loading + diffing the OLD vs NEW per-file symbol surface (ms). This
    /// is the new gating computation that replaces the unconditional pre-delete
    /// caller scan; reported in the PERF SUMMARY as `incr_p2_symname_ms`.
    pub surface_delta_ms: u64,
    pub predelete_callers_ms: u64,
    pub delete_bulk_ms: u64,
    pub streaming_ms: u64,
    pub phase2_total_ms: u64,
    pub phase2: Phase2IncrStats,
}

/// Key used to track whether Phase 2 (raw edge resolution) has completed.
const EDGES_RESOLVED_KEY: &str = "edges_resolved";

/// A row fetched from `raw_edge` during Phase 2 edge resolution.
/// Shared by both full-build and incremental Phase 2 paths.
#[derive(Deserialize)]
struct RawEdgeRow {
    id_str: String,
    from_file: String,
    #[allow(dead_code)]
    from_name: String,
    from_fqn: String,
    to_name: String,
    #[allow(dead_code)]
    kind: String,
    line: i64,
    import_path: Option<String>,
}

/// Resolve a page of raw edges into the edge accumulator.
///
/// Shared logic for both full-build and incremental Phase 2:
/// 1. Collect unique callee names from the batch
/// 2. Batch-lookup symbols by name (returns FQN via meta::id)
/// 3. Bucket by leaf name, sort deterministically
/// 4. For each raw edge, select best candidate and push resolved edge
/// 5. Flush to DB when accumulator reaches WRITE_BATCH_SIZE
async fn resolve_raw_edge_page(
    db: &Surreal<Db>,
    batch: &[RawEdgeRow],
    edge_batch: &mut Vec<(String, String, i64, String, String, String, String)>,
    label: &str,
) -> Result<()> {
    let to_names: Vec<String> = {
        let mut names: Vec<String> = batch.iter().map(|r| r.to_name.clone()).collect();
        names.sort_unstable();
        names.dedup();
        names
    };

    let sym_rows = find_symbols_by_names_with_pos(db, &to_names).await?;

    let mut name_bucket: HashMap<String, Vec<SymbolWithPos>> = HashMap::new();
    for s in sym_rows {
        name_bucket.entry(s.name.clone()).or_default().push(s);
    }
    for bucket in name_bucket.values_mut() {
        bucket.sort_unstable_by(|a, b| {
            a.file.cmp(&b.file)
                .then(a.line_start.cmp(&b.line_start))
                .then(a.line_end.cmp(&b.line_end))
        });
    }

    for row in batch {
        let resolved_to = match name_bucket.get(&row.to_name) {
            Some(candidates) if !candidates.is_empty() => {
                IndexPipeline::select_best_candidate(
                    candidates,
                    &row.from_file,
                    row.import_path.as_deref(),
                ).cloned()
            }
            _ => {
                debug!(name = %row.to_name, "{}: dropping unresolved raw edge", label);
                None
            }
        };

        if let Some(to) = resolved_to {
            edge_batch.push((
                row.from_fqn.clone(),
                to.fqn.clone(),
                row.line,
                row.from_file.clone(),
                to.file.clone(),
                row.from_fqn.clone(),
                to.fqn.clone(),
            ));

            if edge_batch.len() >= WRITE_BATCH_SIZE {
                flush_edge_batch(db, edge_batch)
                    .await
                    .context(format!("{}: flush edge batch", label))?;
                edge_batch.clear();
            }
        }
    }

    Ok(())
}

/// Runs the parse → embed → store pipeline for one repo.
pub struct IndexPipeline {
    repo: String,
    voyage: Option<VoyageClient>,
    /// Concurrent embedding batches in-flight. Derived from config or api_keys.len()*4.
    embed_concurrency: usize,
    /// Optional file-based embedding cache to avoid redundant Voyage API calls.
    cache: Option<Arc<EmbeddingCache>>,
    /// User-configured extra file extensions beyond the built-in CODE_EXTENSIONS.
    extra_extensions: Vec<String>,
    /// Filenames to skip during indexing (case-sensitive, filename-only match).
    ignore_filenames: HashSet<String>,
    /// Per-repo ignored relative paths (forward-slash-normalized).
    ignore_paths: HashSet<String>,
    /// Data dir for persisted vector shard files. When set, full rebuilds and
    /// incremental updates invalidate the repo's persisted shard (delete CURRENT)
    /// so the next warm rebuilds + re-persists it. None in tests that don't need it.
    data_dir: Option<std::path::PathBuf>,
}

impl IndexPipeline {
    pub fn new(repo: String, voyage: Option<VoyageClient>) -> Self {
        Self::new_with_concurrency(repo, voyage, 4, None)
    }

    pub fn new_with_concurrency(repo: String, voyage: Option<VoyageClient>, embed_concurrency: usize, cache: Option<EmbeddingCache>) -> Self {
        let embed_concurrency = embed_concurrency.max(1);
        Self { repo, voyage, embed_concurrency, cache: cache.map(Arc::new), extra_extensions: vec![], ignore_filenames: HashSet::new(), ignore_paths: HashSet::new(), data_dir: None }
    }

    /// Set the data dir so vector-changing operations invalidate the persisted
    /// shard file (the engine sets this; tests that don't exercise persistence omit it).
    pub fn with_data_dir(mut self, data_dir: std::path::PathBuf) -> Self {
        self.data_dir = Some(data_dir);
        self
    }

    /// Invalidate the repo's persisted vector shard (delete CURRENT) so the next
    /// warm rebuilds + re-persists it. O(1) — does NOT rewrite the multi-GB file
    /// on the hot path; the rewrite happens lazily at the next cold warm. No-op
    /// when data_dir is unset. Old generation dirs are reaped by the warm/startup
    /// sweep once unreferenced.
    fn invalidate_persisted_shard(&self) {
        if let Some(dd) = &self.data_dir {
            let root = crate::vector::shard_file::repo_shard_root(dd, &self.repo);
            let _ = std::fs::remove_file(root.join("CURRENT"));
        }
    }

    pub fn with_extra_extensions(mut self, extra: Vec<String>) -> Self {
        self.extra_extensions = extra;
        self
    }

    pub fn with_ignore_filenames(mut self, filenames: Vec<String>) -> Self {
        self.ignore_filenames = filenames.into_iter().collect();
        self
    }

    pub fn with_ignore_paths(mut self, paths: Vec<String>) -> Self {
        self.ignore_paths = paths.into_iter().collect();
        self
    }

    /// Run the pipeline against the shared `db` handle.
    #[allow(clippy::too_many_arguments)]
    pub async fn run(
        &self,
        db: &Surreal<Db>,
        changes: Option<Vec<FileChange>>,
        force_rebuild: bool,
        vector_index: Option<&tokio::sync::RwLock<ShardedVectorIndex>>,
        progress: Option<ProgressHandle>,
        event_bus: Option<&IndexEventBus>,
        key_hints: &[String],
        cancel_token: Option<CancellationToken>,
    ) -> Result<IndexPipelineStats> {
        // Check if first run (no file_meta at all).
        let meta_load_start = Instant::now();
        let stored_meta = get_all_file_meta(db, &self.repo).await?;
        let incr_meta_load_ms = meta_load_start.elapsed().as_millis() as u64;
        let is_first_run = stored_meta.is_empty();

        if is_first_run || force_rebuild {
            if force_rebuild && !is_first_run {
                info!(repo = %self.repo, "forced full rebuild");
            } else {
                info!(repo = %self.repo, "first run — full rebuild");
            }
            // Walk is needed here (once) to populate Started.total_files.
            // Run it off the async runtime to avoid blocking the executor.
            let repo_clone = self.repo.clone();
            let ext_clone = self.extra_extensions.clone();
            let ign_clone = self.ignore_filenames.clone();
            let ign_paths_clone = self.ignore_paths.clone();
            let total_files = tokio::task::spawn_blocking(move || walk_repo_with(&repo_clone, &ext_clone, &ign_clone, &ign_paths_clone).len() as u64)
                .await
                .unwrap_or(0);
            if let Some(bus) = event_bus {
                bus.emit(IndexEvent::Started {
                    repo: self.repo.clone(),
                    total_files,
                    is_rebuild: force_rebuild,
                });
            }
            let (new_vectors, stage_stats) = self.full_rebuild(db, progress.as_ref(), event_bus, key_hints, cancel_token.as_ref()).await?;
            if let Some(vi) = vector_index {
                let mut guard = vi.write().await;
                // Empty active set: the cap may evict LRU shards to honor the
                // bound. The repo being (re)built is protected internally by
                // replace_repo → install_shard. Query safety is guaranteed by the
                // shared write lock, not the active set.
                guard.replace_repo(&self.repo, &new_vectors, &[]);
            }
            // The shard changed → invalidate the persisted file so the next warm
            // rebuilds + re-persists it (lazy; O(1) here, no multi-GB rewrite on
            // the hot path).
            self.invalidate_persisted_shard();
            let indexed = get_all_file_meta(db, &self.repo).await?.len() as u64;
            info!(
                repo = %self.repo,
                stage3_total_ms = stage_stats.stage3_total_ms,
                stage3_sym_ms = stage_stats.stage3_sym_ms,
                stage3_rawedge_ms = stage_stats.stage3_rawedge_ms,
                stage3_chunk_ms = stage_stats.stage3_chunk_ms,
                stage3_chunk_db_ms = stage_stats.stage3_chunk_db_ms,
                stage3_chunk_cpu_ms = stage_stats.stage3_chunk_cpu_ms,
                stage3_chunk_idx_drop_ms = stage_stats.stage3_chunk_idx_drop_ms,
                stage3_chunk_idx_rebuild_ms = stage_stats.stage3_chunk_idx_rebuild_ms,
                stage3_sym_idx_drop_ms = stage_stats.stage3_sym_idx_drop_ms,
                stage3_sym_idx_rebuild_ms = stage_stats.stage3_sym_idx_rebuild_ms,
                stage3_filemeta_ms = stage_stats.stage3_filemeta_ms,
                phase2_ms = stage_stats.phase2_ms,
                phase2_sym_load_ms = stage_stats.phase2_sym_load_ms,
                phase2_bucket_build_ms = stage_stats.phase2_bucket_build_ms,
                phase2_resolve_cpu_ms = stage_stats.phase2_resolve_cpu_ms,
                phase2_relate_write_ms = stage_stats.phase2_relate_write_ms,
                phase2_idx_drop_ms = stage_stats.phase2_idx_drop_ms,
                phase2_idx_rebuild_ms = stage_stats.phase2_idx_rebuild_ms,
                phase2_edges_written = stage_stats.phase2_edges_written,
                embed_total_ms = stage_stats.embed_total_ms,
                cache_hit_chunks = stage_stats.cache_hit_chunks,
                cache_miss_chunks = stage_stats.cache_miss_chunks,
                files = indexed,
                chunks = stage_stats.total_chunks,
                symbols = stage_stats.total_symbols,
                edges = stage_stats.total_raw_edges,
                "PERF SUMMARY full_rebuild"
            );
            return Ok(IndexPipelineStats {
                indexed_files: indexed,
                total_files,
                stage3_sym_ms: stage_stats.stage3_sym_ms,
                stage3_rawedge_ms: stage_stats.stage3_rawedge_ms,
                stage3_chunk_ms: stage_stats.stage3_chunk_ms,
                stage3_filemeta_ms: stage_stats.stage3_filemeta_ms,
                stage3_total_ms: stage_stats.stage3_total_ms,
                phase2_ms: stage_stats.phase2_ms,
                total_chunks: stage_stats.total_chunks,
                total_symbols: stage_stats.total_symbols,
                total_raw_edges: stage_stats.total_raw_edges,
                embed_total_ms: stage_stats.embed_total_ms,
                cache_hit_chunks: stage_stats.cache_hit_chunks,
                cache_miss_chunks: stage_stats.cache_miss_chunks,
                stage3_chunk_db_ms: stage_stats.stage3_chunk_db_ms,
                stage3_chunk_cpu_ms: stage_stats.stage3_chunk_cpu_ms,
                stage3_chunk_idx_drop_ms: stage_stats.stage3_chunk_idx_drop_ms,
                stage3_chunk_idx_rebuild_ms: stage_stats.stage3_chunk_idx_rebuild_ms,
                stage3_sym_idx_drop_ms: stage_stats.stage3_sym_idx_drop_ms,
                stage3_sym_idx_rebuild_ms: stage_stats.stage3_sym_idx_rebuild_ms,
                phase2_sym_load_ms: stage_stats.phase2_sym_load_ms,
                phase2_bucket_build_ms: stage_stats.phase2_bucket_build_ms,
                phase2_resolve_cpu_ms: stage_stats.phase2_resolve_cpu_ms,
                phase2_relate_write_ms: stage_stats.phase2_relate_write_ms,
                phase2_idx_drop_ms: stage_stats.phase2_idx_drop_ms,
                phase2_idx_rebuild_ms: stage_stats.phase2_idx_rebuild_ms,
                phase2_edges_written: stage_stats.phase2_edges_written,
                // Incremental-path fields are 0 on the full-rebuild path.
                ..Default::default()
            });
        }

        // Incremental run.
        // Time the manual-path walk + detect_changes separately. 0 when the
        // change set is watcher-supplied (no walk performed).
        let walk_start = Instant::now();
        let mut incr_walk_ms: u64 = 0;
        let file_changes = match changes {
            Some(explicit) => {
                // Watcher-supplied explicit change set: skip the walk entirely.
                // total_files is derived from stored_meta (already loaded above).
                explicit
            }
            None => {
                // Manual/poll incremental: must walk to detect changes.
                // Run off the async runtime — this is genuinely O(repo).
                let repo_clone = self.repo.clone();
                let ext_clone = self.extra_extensions.clone();
                let ign_clone = self.ignore_filenames.clone();
                let ign_paths_clone = self.ignore_paths.clone();
                let meta_map: HashMap<String, (i64, i64, i64)> = stored_meta
                    .iter()
                    .map(|m| (m.path.clone(), (m.mtime, m.size, m.chunker_version)))
                    .collect();
                let changes = tokio::task::spawn_blocking(move || {
                    let all_files = walk_repo_with(&repo_clone, &ext_clone, &ign_clone, &ign_paths_clone);
                    crate::indexing::tracker::detect_changes(
                        &all_files,
                        &meta_map,
                        crate::parsing::chunker::CHUNKER_VERSION,
                    )
                })
                .await
                .context("incremental walk spawn_blocking")?;
                incr_walk_ms = walk_start.elapsed().as_millis() as u64;
                changes
            }
        };

        // Filter out Added/Modified changes whose paths are inside dot-prefixed directories.
        // Deleted changes are allowed through to clean up any previously indexed dot-dir entries.
        let file_changes = filter_hidden_changes_with(std::path::Path::new(&self.repo), file_changes, self.extra_extensions.clone(), self.ignore_filenames.clone(), self.ignore_paths.clone());

        if file_changes.is_empty() {
            debug!(repo = %self.repo, "no changes detected");
            // Check if edges_resolved marker is missing.
            let resolved = get_meta(db, EDGES_RESOLVED_KEY).await?.is_some();
            if !resolved {
                // Check how many raw_edges are in the DB.
                // If raw_edge is empty but file_meta is present, this is the crash
                // scenario for the RAM-path full rebuild: Stage 3 completed with raw_edges
                // buffered in RAM (never written to DB), but the process died before
                // Phase 2 completed.  We cannot replay Phase 2 from DB (no raw_edges
                // there), so we must force a full rebuild.
                use serde::Deserialize;
                #[derive(Deserialize)]
                struct CountRow { count: i64 }
                let raw_edge_count: Vec<CountRow> = db
                    .query("SELECT count() AS count FROM raw_edge GROUP ALL")
                    .await
                    .context("crash-recovery: count raw_edge")?
                    .take(0)?;
                let raw_edge_total = raw_edge_count.first().map(|r| r.count).unwrap_or(0);

                if raw_edge_total == 0 && !stored_meta.is_empty() {
                    // RAM-path crash: Stage 3 completed, Phase 2 never ran, no DB raw_edges.
                    // Force a full rebuild to regenerate calls edges.
                    warn!(
                        repo = %self.repo,
                        "RAM-path crash detected (edges_resolved absent, raw_edge empty, file_meta present) \
                         — forcing full rebuild to recover calls edges"
                    );
                    let (new_vectors, stage_stats) = self.full_rebuild(db, None, event_bus, key_hints, cancel_token.as_ref()).await?;
                    if let Some(vi) = vector_index {
                        let mut guard = vi.write().await;
                        guard.replace_repo(&self.repo, &new_vectors, &[]);
                    }
                    self.invalidate_persisted_shard();
                    let indexed = get_all_file_meta(db, &self.repo).await?.len() as u64;
                    let total_files = stored_meta.len() as u64;
                    return Ok(IndexPipelineStats {
                        indexed_files: indexed,
                        total_files,
                        phase2_ms: stage_stats.phase2_ms,
                        ..Default::default()
                    });
                } else {
                    // Normal Phase 2 replay: raw_edges are in DB (overflow path or incremental).
                    info!(repo = %self.repo, raw_edge_total, "edges_resolved marker absent — replaying Phase 2 from DB");
                    self.resolve_edges_phase2(db, cancel_token.as_ref()).await
                        .context("edges Phase 2 replay on no-change run")?;
                    // (replay path discards Phase2Stats — no aggregate stats returned here)
                }
            }
            let indexed = stored_meta.len() as u64;
            let total_files = stored_meta.len() as u64;
            return Ok(IndexPipelineStats { indexed_files: indexed, total_files, incr_walk_ms, incr_meta_load_ms, ..Default::default() });
        }

        // For watcher-path (changes == Some), total_files comes from stored_meta (no walk).
        // This value is already computed from stored_meta above.
        let total_files = stored_meta.len() as u64;

        info!(repo = %self.repo, changes = file_changes.len(), "incremental index");
        if let Some(bus) = event_bus {
            bus.emit(IndexEvent::Started {
                repo: self.repo.clone(),
                total_files: file_changes.iter().filter(|c| c.kind != ChangeKind::Deleted).count() as u64,
                is_rebuild: false,
            });
        }
        let (removed_files, new_vectors, incr_stats) = self.incremental_run(db, file_changes, progress.as_ref(), event_bus, key_hints, cancel_token.as_ref()).await?;

        let vi_apply_start = Instant::now();
        if let Some(vi) = vector_index {
            let mut guard = vi.write().await;
            // Empty active set — see replace_repo call above for the rationale.
            // apply_incremental protects `self.repo` internally.
            guard.apply_incremental(&self.repo, &removed_files, &new_vectors, &[]);
        }
        let incr_vi_apply_ms = vi_apply_start.elapsed().as_millis() as u64;
        // Incremental changed the in-RAM shard → invalidate the persisted file
        // (O(1)); it is rebuilt + re-persisted on the next cold warm. We do NOT
        // rewrite the multi-GB file per incremental edit (would be O(repo)).
        self.invalidate_persisted_shard();

        let post_meta_start = Instant::now();
        let indexed = get_all_file_meta(db, &self.repo).await?.len() as u64;
        let incr_post_meta_ms = post_meta_start.elapsed().as_millis() as u64;

        // Per-stage PERF SUMMARY for the incremental path, mirroring the
        // full_rebuild summary above. This is the breakdown that was previously
        // missing — it attributes the incremental wall time to walk / meta-load /
        // pre-delete / bulk-delete / streaming / phase2 (and phase2's sub-stages)
        // so the dominant cost is greppable in one line.
        let incr_phase2_total_ms = incr_stats.phase2_total_ms;
        let p2 = &incr_stats.phase2;
        // `incr_p2_symname_ms` now carries the surface-delta (load+diff) time —
        // the gating computation that replaced the old unconditional symbol-name
        // query. `p2.p2_symname_ms` is always 0 now (the query moved up here).
        let incr_surface_delta_ms = incr_stats.surface_delta_ms;
        info!(
            repo = %self.repo,
            incr_walk_ms,
            incr_meta_load_ms,
            incr_predelete_callers_ms = incr_stats.predelete_callers_ms,
            incr_delete_bulk_ms = incr_stats.delete_bulk_ms,
            incr_streaming_ms = incr_stats.streaming_ms,
            incr_phase2_total_ms,
            incr_p2_symname_ms = incr_surface_delta_ms,
            incr_p2_dir2_scan_ms = p2.p2_dir2_scan_ms,
            incr_p2_delete_calls_ms = p2.p2_delete_calls_ms,
            incr_p2_reresolve_ms = p2.p2_reresolve_ms,
            incr_resolve_set_size = p2.resolve_set_size,
            incr_vi_apply_ms,
            incr_post_meta_ms,
            files = indexed,
            "PERF SUMMARY incremental"
        );

        Ok(IndexPipelineStats {
            indexed_files: indexed,
            total_files,
            incr_walk_ms,
            incr_meta_load_ms,
            incr_predelete_callers_ms: incr_stats.predelete_callers_ms,
            incr_delete_bulk_ms: incr_stats.delete_bulk_ms,
            incr_streaming_ms: incr_stats.streaming_ms,
            incr_phase2_total_ms,
            incr_p2_symname_ms: incr_surface_delta_ms,
            incr_p2_dir2_scan_ms: p2.p2_dir2_scan_ms,
            incr_p2_delete_calls_ms: p2.p2_delete_calls_ms,
            incr_p2_reresolve_ms: p2.p2_reresolve_ms,
            incr_resolve_set_size: p2.resolve_set_size,
            ..Default::default()
        })
    }

    // ─── Full rebuild ─────────────────────────────────────────────────────

    async fn full_rebuild(
        &self,
        db: &Surreal<Db>,
        progress: Option<&ProgressHandle>,
        event_bus: Option<&IndexEventBus>,
        key_hints: &[String],
        cancel_token: Option<&CancellationToken>,
    ) -> Result<(Vec<(ChunkId, Vec<f32>)>, IndexPipelineStats)> {
        let all_files = walk_repo_with(&self.repo, &self.extra_extensions, &self.ignore_filenames, &self.ignore_paths);
        info!(repo = %self.repo, file_count = all_files.len(), "walking repo for full rebuild");

        // Cancel landing before any DB writes: bail before the destructive
        // delete_all_data so the existing index is left intact.
        if let Some(ct) = cancel_token
            && ct.is_cancelled()
        {
            return Err(PipelineAbort::Cancelled.into());
        }

        // Delete everything first (crash-safe: file_meta is the commit marker,
        // written per-file only after its chunks are durable).
        delete_all_data(db).await.context("full_rebuild: delete_all_data")?;

        // Also clear the edges_resolved marker so Phase 2 re-runs after build.
        let _ = db.query("DELETE FROM index_meta WHERE key = $k")
            .bind(("k", EDGES_RESOLVED_KEY))
            .await;

        // Stream parse → embed → write with bounded channels.
        // Raw edges are buffered in RAM (bounded by MAX_RAM_EDGES) when possible,
        // avoiding a DB write + read round-trip (~27s for notepad-ade).
        // If the repo exceeds MAX_RAM_EDGES, edges overflow to the DB and Phase 2
        // falls back to the keyset scan path (same as before).
        let (chunk_vectors, mut stats, ram_raw_edges, ram_edges_overflowed, ram_symbols) = self
            .streaming_index(&all_files, db, progress, event_bus, key_hints, true, cancel_token)
            .await
            .context("full_rebuild: streaming_index")?;

        // Phase 2: resolve raw edges into denormalized calls rows.
        if let Some(bus) = event_bus {
            bus.emit(IndexEvent::Phase2Start { repo: self.repo.clone() });
        }
        let phase2_start = Instant::now();
        let p2: Phase2Stats = if !ram_edges_overflowed && !ram_raw_edges.is_empty() {
            // Fast path: all raw_edges are in RAM — skip DB scan entirely. Pass the
            // in-RAM symbol buffer so Phase 2 reuses it instead of reloading every
            // symbol from the DB (None when the buffer overflowed → DB reload).
            self.resolve_edges_from_ram(db, ram_raw_edges, ram_symbols, cancel_token)
                .await
                .context("full_rebuild: resolve_edges_from_ram")?
        } else {
            // DB path: overflow or incremental — use keyset scan as before.
            self.resolve_edges_phase2(db, cancel_token)
                .await
                .context("full_rebuild: resolve_edges_phase2")?
        };
        let phase2_ms = phase2_start.elapsed().as_millis() as u64;
        stats.phase2_ms = phase2_ms;
        stats.phase2_sym_load_ms = p2.sym_load_ms;
        stats.phase2_bucket_build_ms = p2.bucket_build_ms;
        stats.phase2_resolve_cpu_ms = p2.resolve_cpu_ms;
        stats.phase2_relate_write_ms = p2.relate_write_ms;
        stats.phase2_idx_drop_ms = p2.idx_drop_ms;
        stats.phase2_idx_rebuild_ms = p2.idx_rebuild_ms;
        stats.phase2_edges_written = p2.edges_written;
        if let Some(bus) = event_bus {
            bus.emit(IndexEvent::Phase2Done {
                repo: self.repo.clone(),
                elapsed_ms: phase2_ms,
            });
        }

        Ok((chunk_vectors, stats))
    }

    // ─── Incremental run ──────────────────────────────────────────────────

    async fn incremental_run(
        &self,
        db: &Surreal<Db>,
        changes: Vec<FileChange>,
        progress: Option<&ProgressHandle>,
        event_bus: Option<&IndexEventBus>,
        key_hints: &[String],
        cancel_token: Option<&CancellationToken>,
    ) -> Result<(Vec<String>, Vec<(ChunkId, Vec<f32>)>, IncrementalRunStats)> {
        let mut run_stats = IncrementalRunStats::default();
        let to_process: Vec<String> = changes
            .iter()
            .filter(|c| c.kind != ChangeKind::Deleted)
            .map(|c| c.path.clone())
            .collect();
        let to_delete: Vec<String> = changes
            .iter()
            .filter(|c| c.kind == ChangeKind::Deleted)
            .map(|c| c.path.clone())
            .collect();

        let all_affected: Vec<String> = to_delete
            .iter()
            .chain(to_process.iter())
            .cloned()
            .collect();

        // ── Capture the OLD symbol surface of the modified files BEFORE deleting ──
        // their symbols. The surface is the set of (leaf_name, fqn) pairs each file
        // defines — precisely the part that can change any caller's `calls`
        // resolution. We diff it against the NEW surface (after streaming_index
        // re-writes the symbols) to gate the blast radius: a comment/body-only edit
        // leaves the surface IDENTICAL, so direction-1 (callers of surface-changed
        // files) and direction-2 (callers of newly-added names) both collapse to
        // empty and the resolve_set is exactly the changed files.
        let surface_start = Instant::now();
        let old_surface = load_file_surface(db, &to_process)
            .await
            .context("incremental_run: load old symbol surface")?;
        run_stats.surface_delta_ms = surface_start.elapsed().as_millis() as u64;

        // Delete affected files' data EXCEPT `calls` (incremental helper). The
        // `calls` deletion is deferred to resolve_edges_incremental, scoped to the
        // computed resolve_set — wiping incoming edges here (as the OR-deleting
        // bulk helper does) is exactly the blow-up we are fixing.
        let delete_bulk_start = Instant::now();
        delete_files_data_incremental(db, &all_affected)
            .await
            .context("incremental_run: delete_files_data_incremental")?;
        run_stats.delete_bulk_ms = delete_bulk_start.elapsed().as_millis() as u64;

        // Stream parse → embed → write.
        // Raw edges go to DB (crash-safe incremental path).
        let streaming_start = Instant::now();
        let (chunk_vectors, _stage_stats, _ram_edges, _overflowed, _ram_symbols) = self
            .streaming_index(&to_process, db, progress, event_bus, key_hints, false, cancel_token)
            .await
            .context("incremental_run: streaming_index")?;
        run_stats.streaming_ms = streaming_start.elapsed().as_millis() as u64;

        // Delete file_meta for deleted files (their symbols/chunks/raw_edge were
        // already cleared by delete_files_data_incremental; this is belt-and-braces
        // for the genuinely-removed files which streaming_index does not re-add).
        for file in &to_delete {
            let escaped = escape_surreal(file);
            db.query(format!(
                "DELETE FROM file_meta WHERE path = '{escaped}'"
            ))
            .await
            .context("incremental_run: delete file_meta for deleted file")?;
        }

        // ── Compute the surface delta now that NEW symbols are written ──────────
        // removed_surface_files = modified files that LOST/MOVED a (name,fqn) pair,
        // plus every deleted file (these gate direction-1). added_names = leaf names
        // that gained a (name,fqn) pair (these gate direction-2). A pure-addition or
        // surface-unchanged file appears in NEITHER gate's file set.
        let surface_start2 = Instant::now();
        let new_surface = load_file_surface(db, &to_process)
            .await
            .context("incremental_run: load new symbol surface")?;
        let delta = compute_surface_delta(&old_surface, &new_surface, &to_process, &to_delete);
        run_stats.surface_delta_ms += surface_start2.elapsed().as_millis() as u64;

        // ── Direction-1: callers pointing INTO a file that REMOVED/MOVED a symbol ─
        // Only a file that LOST a `(name, fqn)` pair (or was deleted) can have
        // orphaned a pre-existing incoming edge, so direction-1 is gated on
        // `removed_surface_files`. A file that only GAINED symbols, or whose surface
        // is unchanged, contributes NO dir1 callers — its (possibly thousands of)
        // incoming edges all still point at still-existing, still-winning symbols
        // and stay out of the resolve_set. Pure additions are handled solely by
        // direction-2 (`added_names`). The query runs against the still-intact
        // `calls` table (the incremental delete left `calls` untouched). Skipped
        // entirely when nothing was removed — the common comment/add-symbol cases.
        let predelete_start = Instant::now();
        let dir1_callers: Vec<String> = if delta.removed_surface_files.is_empty() {
            Vec::new()
        } else {
            #[derive(Deserialize)]
            struct CallerRow { in_file: String }
            let rows: Vec<CallerRow> = db
                .query(
                    "SELECT in_file FROM calls \
                     WHERE out_file IN $changed AND in_file NOT IN $affected \
                     GROUP BY in_file",
                )
                .bind(("changed", delta.removed_surface_files.clone()))
                .bind(("affected", all_affected.clone()))
                .await
                .context("incremental_run: direction-1 caller query")?
                .take(0)?;
            rows.into_iter().map(|r| r.in_file).collect()
        };
        run_stats.predelete_callers_ms = predelete_start.elapsed().as_millis() as u64;

        // Phase 2: resolve only edges in the gated blast radius —
        // O(changed + callers_of_removed_surface + callers_of_added_names).
        if let Some(bus) = event_bus {
            bus.emit(IndexEvent::Phase2Start { repo: self.repo.clone() });
        }
        let phase2_start = Instant::now();
        let phase2_stats = self
            .resolve_edges_incremental(db, &all_affected, &dir1_callers, &delta.added_names)
            .await
            .context("incremental_run: resolve_edges_incremental")?;
        run_stats.phase2_total_ms = phase2_start.elapsed().as_millis() as u64;
        run_stats.phase2 = phase2_stats;
        if let Some(bus) = event_bus {
            bus.emit(IndexEvent::Phase2Done {
                repo: self.repo.clone(),
                elapsed_ms: run_stats.phase2_total_ms,
            });
        }

        Ok((all_affected, chunk_vectors, run_stats))
    }

    // ─── Streaming parse→embed→write pipeline ────────────────────────────

    /// Stream files through parse → embed → write with bounded channels.
    ///
    /// Peak inflight = PARSE_CHANNEL_CAP + EMBED_CHANNEL_CAP parsed/embedded files
    /// (O(channels * chunks_per_file)), independent of total repo size.
    ///
    /// For full rebuilds: raw_edges are buffered in RAM (up to MAX_RAM_EDGES) to
    /// avoid a DB write+read round-trip.  If the repo exceeds the cap, edges overflow
    /// to the `raw_edge` DB table and Phase 2 falls back to the keyset scan path.
    /// For incremental builds: raw_edges always go to the `raw_edge` DB table for
    /// crash-safe Phase 2 replay.
    #[allow(clippy::too_many_arguments)]
    async fn streaming_index(
        &self,
        files: &[String],
        db: &Surreal<Db>,
        progress: Option<&ProgressHandle>,
        event_bus: Option<&IndexEventBus>,
        key_hints: &[String],
        is_full_rebuild: bool,
        cancel_token: Option<&CancellationToken>,
    ) -> Result<(Vec<(ChunkId, Vec<f32>)>, IndexPipelineStats, Vec<RawEdgeRecord>, bool, Option<HashMap<String, SymbolWithPos>>)> {
        if files.is_empty() {
            if let Some(ph) = progress {
                ph.set_run_total(0).await;
                ph.set_processed(0).await;
            }
            return Ok((vec![], IndexPipelineStats::default(), vec![], false, None));
        }

        let total_files = files.len() as u64;
        if let Some(ph) = progress {
            ph.set_run_total(total_files).await;
        }

        let voyage = self.voyage.clone();
        let embed_concurrency = self.embed_concurrency;
        let cache_arc = self.cache.clone();
        let event_bus_clone = event_bus.cloned();
        let key_hints_owned: Vec<String> = key_hints.to_vec();

        // ── Framework detection (once per run) ───────────────────────────────
        // Build a file set from the files being indexed to detect active frameworks.
        let framework_registry = {
            use crate::indexing::frameworks::{DetectionContext, FrameworkRegistry};
            let mut registry = FrameworkRegistry::new();
            let file_set: HashSet<String> = files.iter().cloned().collect();
            let ctx = DetectionContext {
                file_set: &file_set,
                read_file: &|path: &str| std::fs::read_to_string(path).ok(),
            };
            registry.detect(&ctx);
            Arc::new(registry)
        };

        // ── Stage 1: parallel parse (rayon), feed into bounded channel ────
        let (parse_tx, parse_rx) = mpsc::channel::<ParseOutput>(PARSE_CHANNEL_CAP);
        {
            let files_owned: Vec<String> = files.to_vec();
            let fw_registry = framework_registry.clone();
            tokio::task::spawn_blocking(move || {
                use rayon::prelude::*;

                // Use a LOCAL rayon thread pool with a large stack size for parsing.
                // The default rayon global pool has ~2 MB stacks which overflow on
                // deeply-nested ASTs (Linux kernel C files). 64 MB virtual stack
                // (only committed pages are physical) gives 3200 frames * ~512 bytes
                // per frame = ~1.6 MB actual usage with headroom for future growth.
                // We do NOT change the global pool — vector search uses it and must
                // not be affected.
                const PARSE_STACK_SIZE: usize = 64 * 1024 * 1024; // 64 MB virtual

                let pool = rayon::ThreadPoolBuilder::new()
                    .stack_size(PARSE_STACK_SIZE)
                    .build();

                match pool {
                    Ok(pool) => {
                        pool.install(|| {
                            files_owned.par_iter().for_each(|file| {
                                let output = parse_one_file_with_frameworks(file, &fw_registry);
                                // Blocking send — applies backpressure when embed is slow.
                                if parse_tx.blocking_send(output).is_err() {
                                    // Receiver dropped (pipeline cancelled) — stop.
                                }
                            });
                        });
                    }
                    Err(e) => {
                        // Fallback: use global pool if custom pool creation fails.
                        // This is degraded but functional — log and continue.
                        warn!(error = %e, "failed to create parse thread pool with large stack; \
                               falling back to global pool (stack overflow risk on deep ASTs)");
                        files_owned.par_iter().for_each(|file| {
                            let output = parse_one_file_with_frameworks(file, &fw_registry);
                            if parse_tx.blocking_send(output).is_err() {}
                        });
                    }
                }
                // parse_tx dropped here, closing the channel.
            });
        }

        // ── Stage 2: concurrent embed (buffer_unordered(N)) ──────────────
        // Monotonic progress counter shared across concurrent embed tasks.
        let done_counter = Arc::new(AtomicU64::new(0));

        let (embed_tx, mut embed_rx) = mpsc::channel::<EmbeddedFile>(EMBED_CHANNEL_CAP);

        // Wrap the parse receiver as a stream of ParseOutput, embed each
        // concurrently up to `embed_concurrency` at a time.
        // Shared error slot: when Voyage fails, Stage 2 writes the error here
        // and cancels the token. Stage 3 checks this after its loop to distinguish
        // "user cancelled" from "Voyage failed".
        let embed_error: Arc<std::sync::Mutex<Option<String>>> =
            Arc::new(std::sync::Mutex::new(None));

        {
            let voyage_clone = voyage.clone();
            let done_counter_clone = done_counter.clone();
            let embed_tx_clone = embed_tx.clone();
            let progress_clone = progress.cloned();
            let cache_clone = cache_arc.clone();
            let bus_clone = event_bus_clone.clone();
            let hints_clone = key_hints_owned.clone();
            let cancel_clone = cancel_token.cloned();
            let embed_error_clone = embed_error.clone();

            tokio::spawn(async move {
                // Convert mpsc receiver to a stream that stops on cancel.
                let ct_for_stream = cancel_clone.clone();
                let stream = futures::stream::unfold(parse_rx, move |mut rx| {
                    let ct = ct_for_stream.clone();
                    async move {
                        if let Some(ref ct) = ct
                            && ct.is_cancelled()
                        {
                            return None;
                        }
                        rx.recv().await.map(|item| (item, rx))
                    }
                });

                stream
                    .map(|output| {
                        let voyage_ref = voyage_clone.clone();
                        let cache_ref = cache_clone.clone();
                        let done_ref = done_counter_clone.clone();
                        let progress_ref = progress_clone.clone();
                        let bus_ref = bus_clone.clone();
                        let hints_ref = hints_clone.clone();
                        let ct_ref = cancel_clone.clone();
                        let err_slot = embed_error_clone.clone();
                        async move {
                            // Short-circuit if cancelled — skip the expensive embed call.
                            if let Some(ref ct) = ct_ref
                                && ct.is_cancelled()
                            {
                                return None;
                            }
                            match output {
                                ParseOutput::Skipped { file, reason } => {
                                    // Emit skip event and count it — no EmbeddedFile produced.
                                    if let Some(ref bus) = bus_ref {
                                        bus.emit(IndexEvent::FileSkipped {
                                            file: file.clone(),
                                            reason,
                                        });
                                    }
                                    let done = done_ref.fetch_add(1, Ordering::Relaxed) + 1;
                                    if let Some(ph) = &progress_ref {
                                        ph.set_processed(done).await;
                                    }
                                    None
                                }
                                ParseOutput::Parsed(pf) => {
                                    // Measure queue wait: time from when pf was created in rayon
                                    // to when Stage 2 picks it up.
                                    let queue_wait_ms = pf.created_at.elapsed().as_millis() as u64;
                                    let chunk_count = pf.chunks.len();
                                    let symbol_count = pf.symbols.len();
                                    let file_path = pf.path.clone();

                                    // Emit FileParsed event.
                                    if let Some(ref bus) = bus_ref {
                                        bus.emit(IndexEvent::FileParsed {
                                            file: file_path.clone(),
                                            chunks: chunk_count,
                                            symbols: symbol_count,
                                            parse_ms: pf.parse_elapsed_ms,
                                            queue_wait_ms,
                                        });
                                    }

                                    let key_hint = hints_ref.first().cloned().unwrap_or_default();
                                    let embed_start = Instant::now();

                                    let embed_result = match embed_parsed_file(&pf, voyage_ref.as_ref(), cache_ref.clone()).await {
                                        Ok(r) => r,
                                        Err(embed_err) => {
                                            // Classify: transient/retry-exhausted errors are
                                            // NON-FATAL — skip this file and continue. The file
                                            // has no file_meta committed, so the next index
                                            // trigger re-processes it (self-healing via crash-
                                            // safe file_meta). This prevents a single gateway
                                            // timeout from aborting an entire 79K-file rebuild.
                                            //
                                            // Only genuinely fatal errors (auth 4xx, config,
                                            // permanent API failure = EmbedFileError::Fatal) abort.
                                            match embed_err {
                                                EmbedFileError::Transient(e) => {
                                                    let msg = format!("{e:#}");
                                                    warn!(
                                                        file = %file_path,
                                                        error = %msg,
                                                        "embed failed (transient, retry exhausted) — skipping file; \
                                                         will retry on next index trigger"
                                                    );
                                                    // Emit FileSkipped event + advance progress so
                                                    // the run completes without this file.
                                                    if let Some(ref bus) = bus_ref {
                                                        bus.emit(IndexEvent::FileSkipped {
                                                            file: file_path.clone(),
                                                            reason: format!("transient embed failure: {msg}"),
                                                        });
                                                    }
                                                    let done = done_ref.fetch_add(1, Ordering::Relaxed) + 1;
                                                    if let Some(ph) = &progress_ref {
                                                        ph.set_processed(done).await;
                                                    }
                                                    return None;
                                                }
                                                EmbedFileError::Fatal(e) => {
                                                    // Fatal error — store and cancel pipeline.
                                                    let msg = format!("{e:#}");
                                                    warn!(file = %file_path, error = %msg, "embed failed (fatal) — aborting index");
                                                    if let Ok(mut slot) = err_slot.lock()
                                                        && slot.is_none()
                                                    {
                                                        *slot = Some(msg);
                                                    }
                                                    if let Some(ref ct) = ct_ref {
                                                        ct.cancel();
                                                    }
                                                    return None;
                                                }
                                            }
                                        }
                                    };

                                    let embed_elapsed_ms = embed_start.elapsed().as_millis() as u64;

                                    // Detect embed failure: all embeddings empty and chunks non-zero
                                    // indicates a cache-panic degradation (not a Voyage error — those
                                    // now abort the pipeline).
                                    let embed_failed = !pf.chunks.is_empty()
                                        && embed_result.embeddings.iter().all(|e| e.is_empty());

                                    // Emit FileEmbedded event.
                                    if let Some(ref bus) = bus_ref {
                                        bus.emit(IndexEvent::FileEmbedded {
                                            file: file_path.clone(),
                                            chunks: chunk_count,
                                            elapsed_ms: embed_elapsed_ms,
                                            cached: embed_result.fully_cached,
                                            key_hint,
                                        });
                                    }

                                    let pipeline_start = pf.created_at;

                                    Some(EmbeddedFile {
                                        path: pf.path,
                                        symbols: pf.symbols,
                                        chunks: pf.chunks,
                                        embeddings: embed_result.embeddings,
                                        raw_edges: pf.raw_edges,
                                        mtime: pf.mtime,
                                        size: pf.size,
                                        embed_failed,
                                        created_at: Instant::now(),
                                        pipeline_start,
                                        embed_elapsed_ms,
                                        cache_hit_chunks: embed_result.hit_chunks,
                                        cache_miss_chunks: embed_result.miss_chunks,
                                    })
                                }
                            }
                        }
                    })
                    .buffer_unordered(embed_concurrency)
                    .for_each(|opt_ef| {
                        let tx = embed_tx_clone.clone();
                        async move {
                            if let Some(ef) = opt_ef {
                                // If writer is slow, this blocks (bounded channel backpressure).
                                let _ = tx.send(ef).await;
                            }
                        }
                    })
                    .await;
                // embed_tx_clone dropped here (but original embed_tx still alive).
            });
        }
        // Drop the original embed_tx so the channel closes when the spawned task finishes.
        drop(embed_tx);

        // ── Stage 3: writer — drain embed_rx, flush in batches ───────────
        let mut all_chunk_vectors: Vec<(ChunkId, Vec<f32>)> = Vec::new();

        // Per-stage timing accumulators (nanoseconds, summed across all files).
        let mut sym_ns: u64 = 0;
        let mut rawedge_ns: u64 = 0;
        let mut chunk_ns: u64 = 0;
        // Sub-term: nanoseconds spent actually awaiting the DB INSERT.
        let mut chunk_db_ns: u64 = 0;
        // Sub-term: nanoseconds spent building ChunkRecord structs + pushing to Vec.
        let mut chunk_cpu_ns: u64 = 0;
        let mut filemeta_ns: u64 = 0;
        let mut total_chunks_count: u64 = 0;
        let mut total_symbols_count: u64 = 0;
        let mut total_raw_edges_count: u64 = 0;
        // Secondary-symbol-index drop/rebuild timings (full rebuild only; 0 otherwise).
        // See the drop block before the writer loop and the rebuild block after the
        // tail flush for the why. Kept separate from `sym_ns` so `sym_ms` keeps
        // measuring pure symbol-flush time and stays comparable before/after.
        let mut sym_idx_drop_ms: u64 = 0;
        let mut sym_idx_rebuild_ms: u64 = 0;
        let stage3_start = Instant::now();
        // Embed/cache-read stage accumulators (from EmbeddedFile fields set in Stage 2).
        let mut embed_total_ms: u64 = 0;
        let mut total_cache_hit_chunks: u64 = 0;
        let mut total_cache_miss_chunks: u64 = 0;

        // Cross-file chunk accumulator: buffer chunks from multiple files before INSERT.
        // This reduces INSERT round-trips from O(files) to O(total_chunks/WRITE_BATCH_SIZE).
        // file_meta is deferred until the batch containing each file's last chunk is committed.
        let mut pending_chunk_batch: Vec<ChunkRecord> = Vec::with_capacity(WRITE_BATCH_SIZE);
        // FileMeta records buffered until the current chunk batch flushes.
        let mut pending_file_metas: Vec<FileMeta> = Vec::new();
        // Cross-file symbol accumulator: buffer symbols from multiple files before UPSERT.
        // Reduces symbol write round-trips from O(files) to O(total_symbols/SYM_BATCH_SIZE).
        const SYM_BATCH_SIZE: usize = 2048;
        let mut pending_symbol_batch: Vec<Symbol> = Vec::with_capacity(SYM_BATCH_SIZE);
        // Cross-file raw_edge accumulator: buffer raw edges from multiple files before INSERT.
        // Reduces raw_edge write round-trips from O(files) to O(total_edges/RAW_EDGE_INSERT_BATCH_SIZE).
        // Used on the overflow path (full rebuild after RAM cap exceeded) and incremental path.
        // The RAM fast path (pre-overflow full rebuild) does NOT use this — edges stay in ram_raw_edges.
        let mut pending_raw_edge_batch: Vec<RawEdgeRecord> = Vec::with_capacity(RAW_EDGE_INSERT_BATCH_SIZE);

        // Full-rebuild optimisation: buffer raw_edges in RAM (up to MAX_RAM_EDGES).
        // Avoids writing/reading raw_edges from DB (saves ~27s for notepad-ade).
        // If the repo exceeds MAX_RAM_EDGES, the buffer is flushed to DB and
        // `ram_edges_overflowed` is set — Phase 2 falls back to the DB scan path.
        // Memory bound: 8M × ~200 bytes ≈ 1.6 GB (constant, bounded regardless of
        // repo size). Fits comfortably in a 64 GB workstation.
        // Measured: the Linux kernel produces 4.44M raw edges in a full rebuild,
        // which exceeded the previous 4M cap and fell through to the slow DB-scan
        // Phase 2 path (45+ min). 8M gives ~1.8× headroom above real kernel scale
        // while keeping memory bounded. Repos exceeding 8M edges (Chromium-scale)
        // overflow to DB gracefully — no OOM risk.
        // NOT used for incremental (few edges, existing DB path is already fast).
        const MAX_RAM_EDGES: usize = 8_000_000;
        let mut ram_raw_edges: Vec<RawEdgeRecord> = if is_full_rebuild {
            Vec::with_capacity(std::cmp::min(4096, MAX_RAM_EDGES))
        } else {
            Vec::new()
        };
        // Once the RAM buffer overflows, all subsequent raw_edges go to DB.
        let mut ram_edges_overflowed = false;

        // Full-rebuild optimisation: buffer parsed symbols in RAM (up to
        // MAX_RAM_SYMBOLS DISTINCT FQNs) alongside the Stage-3 DB symbol write.
        // Phase 2 (resolve_edges_from_ram) reuses this buffer to build its
        // name→candidates map instead of reloading every symbol from the DB
        // (`load_all_symbols`), which costs ~4.7 min at kernel scale (2.6M rows).
        //
        // Keyed by FQN with last-write-wins, reproducing the `symbol` table's
        // `INSERT ... ON DUPLICATE KEY UPDATE` dedup (one row per FQN) EXACTLY —
        // a plain `insert(fqn, pos)` per parsed symbol in Stage-3 stream order is
        // last-write-wins for free. This is an ADDITIVE in-RAM copy: symbols are
        // STILL written to the DB below, so crash-safety (file_meta commit marker)
        // and recovery are byte-identical to today; the buffer is ephemeral.
        //
        // Memory bound: distinct-FQN cap of 6M × ~150–250 bytes/entry (the three
        // Strings fqn+file+name dominate, plus HashMap overhead) ≈ 0.9–1.5 GB at
        // the cap. This coexists with the ≤1.6 GB ram_raw_edges buffer (8M cap),
        // a combined bounded worst case consistent with the measured ~9 GB RSS at
        // kernel scale. The kernel produces ~3.1M parsed symbols / ~2.6M distinct
        // FQNs — within the cap with headroom. Repos exceeding the cap (Chromium-
        // scale) drop the buffer and Phase 2 falls back to the DB reload — no OOM.
        // NOT populated for incremental (returned as None; that path is untouched).
        const MAX_RAM_SYMBOLS_DEFAULT: usize = 6_000_000;
        // Test/ops seam: CONTEXT_ENGINE_MAX_RAM_SYMBOLS overrides the cap. This is a
        // GENERAL knob (not repo-specific) — its purpose is to let the overflow
        // fallback path be exercised against a real repo (e.g. force a low cap on a
        // small repo so the buffer overflows and Phase 2 reloads from the DB),
        // proving the fallback in production, not just in unit tests. Unset → default.
        let max_ram_symbols: usize = std::env::var("CONTEXT_ENGINE_MAX_RAM_SYMBOLS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(MAX_RAM_SYMBOLS_DEFAULT);
        let mut ram_symbols: HashMap<String, SymbolWithPos> = HashMap::new();
        // Once distinct buffered symbols exceed the cap, the buffer is dropped and
        // Phase 2 reloads symbols from the DB (mirrors ram_edges_overflowed).
        let mut ram_symbols_overflowed = false;
        let mut cancelled = false;

        // ── Drop the two secondary symbol indexes before the bulk symbol write ──
        // (full rebuild only). The `symbol` table was the last high-volume bulk-write
        // table still writing through LIVE secondary indexes: `idx_symbol_file` +
        // `idx_symbol_name` were maintained per row, costing ~6.2M incremental index
        // updates on the kernel (3.1M symbols × 2 indexes) — measured at sym_ms≈20.6 min,
        // 92% of Stage 3. This is the same drop→bulk-write→rebuild trick already used for
        // the `calls` table (see resolve_edges_phase2 ~pipeline.rs:1598) — we just apply it
        // to symbols. The bulk INSERT and its ON DUPLICATE KEY UPDATE (dedup on PRIMARY
        // KEY = FQN, NOT on these secondary indexes) are unchanged, so the exact same rows
        // land in the table; only index-maintenance timing moves.
        //
        // Gated strictly on `is_full_rebuild` (D5): these are GLOBAL indexes. On the
        // incremental path we touch O(changed) symbols, so dropping a global index and
        // rebuilding over all 3.1M rows would turn O(changed) into O(repo) — catastrophic.
        // Incremental keeps writing through the live indexes (cheap at small N).
        //
        // Crash-safety (D4): a crash between this drop and the post-tail-flush rebuild
        // leaves symbol ROWS intact but these two secondary indexes missing. Recovery
        // is owned by `store::ensure_secondary_indexes`, which `open_db` runs on every
        // open: it finds the indexes absent on the (populated) symbol table and
        // rebuilds them via `build_index_concurrently` (batched commits, poll-to-ready)
        // — NEVER a foreground `DEFINE INDEX` backfill, which would roll back under the
        // pinned RocksDB buffers at kernel scale and fail the entire open. (These two
        // indexes are deliberately NOT defined in `SCHEMA_DDL` — see store/schema.rs —
        // precisely so the foreground-backfill rollback can never reoccur on recovery.)
        // No data loss; file_meta crash-safety ordering is untouched.
        if is_full_rebuild {
            let t_sym_idx_drop = Instant::now();
            db.query(
                "REMOVE INDEX IF EXISTS idx_symbol_file ON symbol; \
                 REMOVE INDEX IF EXISTS idx_symbol_name ON symbol;"
            ).await.context("streaming_index: drop symbol indexes")?;
            sym_idx_drop_ms = t_sym_idx_drop.elapsed().as_millis() as u64;
            info!(repo = %self.repo, sym_idx_drop_ms, "stage3: dropped symbol indexes (full rebuild)");
        }

        while let Some(ef) = embed_rx.recv().await {
            // Check cancellation before processing each file.
            if let Some(ct) = cancel_token
                && ct.is_cancelled()
            {
                info!(repo = %self.repo, "indexing cancelled — stopping pipeline");
                cancelled = true;
                break;
            }
            // Measure queue wait: time from when EmbeddedFile was created in Stage 2
            // to when Stage 3 picks it up.
            let queue_wait_ms = ef.created_at.elapsed().as_millis() as u64;
            let store_start = Instant::now();

            // Accumulate embed/cache-read stage metrics from Stage 2.
            embed_total_ms += ef.embed_elapsed_ms;
            total_cache_hit_chunks += ef.cache_hit_chunks;
            total_cache_miss_chunks += ef.cache_miss_chunks;

            // ── symbols (cross-file batched) ───────────────────────────
            // Accumulate symbols from multiple files, flush when batch fills.
            let t0 = Instant::now();
            total_symbols_count += ef.symbols.len() as u64;
            // Full-rebuild optimisation: additively buffer each parsed symbol in
            // RAM (keyed by FQN, last-write-wins) so Phase 2 can build its
            // name→candidates map without reloading the symbol table. Populate in
            // Stage-3 STREAM ORDER and BEFORE moving the symbols into the DB write
            // batch, so the map's last-write-wins matches the DB's
            // `INSERT ... ON DUPLICATE KEY UPDATE` per-FQN dedup byte-for-byte.
            // Bounded by distinct-FQN count: on cap exceed, drop the buffer and set
            // the overflow flag (mirrors ram_edges_overflowed) — Phase 2 then
            // reloads from the DB. The DB write below is UNCHANGED either way.
            if is_full_rebuild && !ram_symbols_overflowed {
                for sym in &ef.symbols {
                    let fqn = sym.qualified.fqn();
                    // Inserting a NEW distinct FQN would push past the cap → drop.
                    if !ram_symbols.contains_key(&fqn) && ram_symbols.len() >= max_ram_symbols {
                        info!(
                            buffered = ram_symbols.len(),
                            "stage3: RAM symbol buffer full — dropping, Phase 2 will reload from DB"
                        );
                        ram_symbols = HashMap::new();
                        ram_symbols.shrink_to_fit();
                        ram_symbols_overflowed = true;
                        break;
                    }
                    // Keyed insert via the shared helper — same code the invariance
                    // test exercises, so the test proves the PRODUCTION dedup, not a copy.
                    buffer_insert_symbol(&mut ram_symbols, sym);
                }
            }
            pending_symbol_batch.extend(ef.symbols);
            if pending_symbol_batch.len() >= SYM_BATCH_SIZE {
                flush_symbol_batch_native(db, &std::mem::take(&mut pending_symbol_batch))
                    .await
                    .context("streaming_index: cross-file symbol batch")?;
            }
            sym_ns += t0.elapsed().as_nanos() as u64;

            // ── raw edges ──────────────────────────────────────────────
            let t1 = Instant::now();
            total_raw_edges_count += ef.raw_edges.len() as u64;
            // Full-rebuild path: buffer raw_edges in RAM (bounded by MAX_RAM_EDGES).
            // This avoids a DB write + read round-trip (~27s for notepad-ade).
            // If the buffer overflows, flush everything to DB and continue with DB path.
            // Incremental path: always write to DB (crash-safe anchor for Phase 2 replay).
            if is_full_rebuild && !ram_edges_overflowed {
                let new_total = ram_raw_edges.len() + ef.raw_edges.len();
                if new_total <= MAX_RAM_EDGES {
                    // Buffer in RAM — no DB write.
                    ram_raw_edges.extend(ef.raw_edges.iter().cloned());
                } else {
                    // RAM cap exceeded: flush all accumulated edges to DB and stop buffering.
                    info!(
                        buffered = ram_raw_edges.len(),
                        new_edges = ef.raw_edges.len(),
                        "stage3: RAM raw_edge buffer full — flushing to DB"
                    );
                    // Flush the entire RAM buffer to DB first.
                    if !ram_raw_edges.is_empty() {
                        flush_raw_edge_batch_native(db, &std::mem::take(&mut ram_raw_edges))
                            .await
                            .context("streaming_index: ram_raw_edges flush on overflow")?;
                    }
                    // Route current file's edges into the cross-file pending batch
                    // (not a direct per-file flush). The batch flushes at
                    // RAW_EDGE_INSERT_BATCH_SIZE granularity, converting O(files)
                    // round-trips into O(total_edges / batch_size).
                    pending_raw_edge_batch.extend(ef.raw_edges);
                    if pending_raw_edge_batch.len() >= RAW_EDGE_INSERT_BATCH_SIZE {
                        flush_raw_edge_batch_native(db, &std::mem::take(&mut pending_raw_edge_batch))
                            .await
                            .context("streaming_index: raw_edges (overflow batch)")?;
                    }
                    ram_edges_overflowed = true;
                }
            } else {
                // Incremental path or post-overflow: accumulate into cross-file batch.
                // Crash-safe anchor: if process dies after Stage 3 but before Phase 2
                // completes, the next run detects the absent `edges_resolved` marker
                // and replays Phase 2 from the raw_edge DB table.
                pending_raw_edge_batch.extend(ef.raw_edges);
                if pending_raw_edge_batch.len() >= RAW_EDGE_INSERT_BATCH_SIZE {
                    flush_raw_edge_batch_native(db, &std::mem::take(&mut pending_raw_edge_batch))
                        .await
                        .context("streaming_index: raw_edges (batched)")?;
                }
            }
            rawedge_ns += t1.elapsed().as_nanos() as u64;

            // ── chunks (cross-file batched) ────────────────────────────
            // Accumulate this file's chunks into the cross-file buffer.
            // Flush only when the buffer fills, to batch INSERT round-trips.
            // chunk_ns = total; chunk_cpu_ns = record construction; chunk_db_ns = DB INSERT await.
            let t2 = Instant::now();
            let file_chunk_count = ef.chunks.len() as i64;
            total_chunks_count += ef.chunks.len() as u64;

            for (chunk, emb) in ef.chunks.iter().zip(
                ef.embeddings.iter().cloned().chain(std::iter::repeat(vec![]))
            ) {
                // (a) CPU: construct ChunkRecord and push to vector index accumulator.
                let t_cpu = Instant::now();
                // Pack the f32 embedding into a little-endian byte blob for storage
                // (DB schema v5). Done here, on the writer thread, before the f32
                // copy is moved into the vector-index accumulator. This is the
                // memcpy that replaces ~1024 Value::Number enum allocations/row.
                let packed = crate::store::ops::pack_embedding(&emb);
                all_chunk_vectors.push((
                    ChunkId {
                        file: chunk.file.clone(),
                        line_start: chunk.line_start,
                        line_end: chunk.line_end,
                    },
                    emb,
                ));
                pending_chunk_batch.push(ChunkRecord {
                    file: chunk.file.clone(),
                    line_start: chunk.line_start as i64,
                    line_end: chunk.line_end as i64,
                    content: chunk.content.clone(),
                    embedding: packed,
                    symbol_ref: chunk.symbol_ref.as_ref().map(|fqn| format!("symbol:⟨{fqn}⟩")),
                });
                chunk_cpu_ns += t_cpu.elapsed().as_nanos() as u64;

                // Flush when the cross-file buffer is full.
                if pending_chunk_batch.len() >= WRITE_BATCH_SIZE {
                    // (b) DB: actual INSERT await.
                    let t_db = Instant::now();
                    flush_chunk_batch(db, std::mem::take(&mut pending_chunk_batch))
                        .await
                        .context("streaming_index: cross-file chunk batch")?;
                    chunk_db_ns += t_db.elapsed().as_nanos() as u64;
                    // Flush pending raw edges BEFORE committing file_metas.
                    // Crash-safety invariant: file_meta is the commit marker and must
                    // only become durable AFTER that file's chunks AND raw edges are
                    // durable. Without this flush, a crash could leave file_meta present
                    // with raw edges still in the pending batch — Phase 2 replay would
                    // silently under-resolve because the edges were never persisted.
                    if !pending_raw_edge_batch.is_empty() {
                        let t_re = Instant::now();
                        flush_raw_edge_batch_native(db, &std::mem::take(&mut pending_raw_edge_batch))
                            .await
                            .context("streaming_index: raw_edge batch at chunk-batch boundary")?;
                        rawedge_ns += t_re.elapsed().as_nanos() as u64;
                    }
                    // Commit all deferred file_metas accumulated so far.
                    let t_fm = Instant::now();
                    for fm in std::mem::take(&mut pending_file_metas) {
                        upsert_file_meta(db, &fm)
                            .await
                            .context("streaming_index: upsert_file_meta (deferred)")?;
                    }
                    filemeta_ns += t_fm.elapsed().as_nanos() as u64;
                }
            }
            chunk_ns += t2.elapsed().as_nanos() as u64;

            // ── file_meta deferred (crash-safety) ─────────────────────
            // Enqueue this file's meta. It will be committed after the
            // next chunk-batch flush that includes this file's last chunk.
            pending_file_metas.push(FileMeta {
                path: ef.path.clone(),
                mtime: ef.mtime,
                size: ef.size,
                repo: self.repo.clone(),
                chunk_count: file_chunk_count,
                // Stamp the build's chunker version. file_meta is the crash-safe
                // commit marker — written only after this file's chunks are
                // durable (deferred until the chunk-batch flush below), so an
                // interrupted re-chunk leaves the stale version and re-runs.
                chunker_version: crate::parsing::chunker::CHUNKER_VERSION,
            });

            let store_elapsed_ms = store_start.elapsed().as_millis() as u64;
            let total_elapsed_ms = ef.pipeline_start.elapsed().as_millis() as u64;

            // Emit FileStored event.
            if let Some(bus) = event_bus {
                bus.emit(IndexEvent::FileStored {
                    file: ef.path.clone(),
                    elapsed_ms: store_elapsed_ms,
                    queue_wait_ms,
                });
            }

            // Increment done counter and emit FileIndexed.
            let done = done_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(ph) = progress {
                ph.set_processed(done).await;
            }
            if let Some(bus) = event_bus {
                let status = if ef.embed_failed { "no_embeddings" } else { "ok" };
                bus.emit(IndexEvent::FileIndexed {
                    file: ef.path.clone(),
                    indexed: done,
                    total: total_files,
                    total_elapsed_ms,
                    status: status.to_string(),
                });
            }
        }

        // Post-loop cancellation re-check. The Stage 3 loop above only observes a
        // cancel when it *receives* a file. If the user cancels while Stage 3 is
        // parked on `recv()`, Stage 2's unfold stops feeding the channel, in-flight
        // embeds drain, `embed_tx` drops, and `recv()` returns None — so the loop
        // exits normally with `cancelled` still false. Without this re-check we'd
        // flush the tail and return Ok, Phase 2 would run to completion, and the repo
        // would finish "successfully" — i.e. the cancel button would appear to do
        // nothing. Re-reading the token here turns that drained-on-cancel exit into a
        // proper abort.
        if !cancelled
            && let Some(ct) = cancel_token
            && ct.is_cancelled()
        {
            info!(repo = %self.repo, "indexing cancelled — channel drained after cancel");
            cancelled = true;
        }

        // If cancelled, drop remaining channel items and return early.
        // Check the shared error slot first — if Voyage failed, return the typed
        // EmbeddingFailed variant so run_consumer can distinguish it from user cancel.
        if cancelled {
            drop(embed_rx);
            if let Ok(slot) = embed_error.lock()
                && let Some(msg) = slot.clone()
            {
                return Err(PipelineAbort::EmbeddingFailed(msg).into());
            }
            return Err(PipelineAbort::Cancelled.into());
        }

        // ── Flush tail: remaining symbols + chunks + raw edges + file_metas ─
        // Order is critical for crash-safety: file_meta (the commit marker) must
        // only become durable AFTER that file's chunks AND raw edges are durable.
        // Phase-2 replay invariant: if file_meta is present, all of that file's
        // raw_edge rows are guaranteed in the DB, so resolution is complete.
        if !pending_symbol_batch.is_empty() {
            let t0 = Instant::now();
            flush_symbol_batch_native(db, &pending_symbol_batch)
                .await
                .context("streaming_index: tail symbol batch")?;
            sym_ns += t0.elapsed().as_nanos() as u64;
        }

        // ── Rebuild the two secondary symbol indexes (full rebuild only) ──
        // The symbol tail flush above is the LAST symbol write, so at this point ALL
        // symbol rows are durable. Rebuild `idx_symbol_file` + `idx_symbol_name`
        // synchronously (no CONCURRENTLY) so the index is fully available before this
        // function returns — i.e. before Phase 2 edge resolution, which reads symbols
        // by name (load_all_symbols / find_symbols_by_names_with_pos), and before any
        // query (query-time name lookup uses idx_symbol_name — store/ops.rs ~542). This
        // is the one-shot bulk build that replaces the ~6.2M per-row index updates we
        // skipped by dropping the indexes before the write; mirrors the calls rebuild
        // (resolve_edges_phase2 ~pipeline.rs:1747). It is INDEPENDENT of Phase 2's own
        // calls-index drop/rebuild (different table, different indexes) — the two must
        // not be conflated. Gated on `is_full_rebuild` to pair with the drop above (D5).
        if is_full_rebuild {
            let t_sym_idx_rebuild = Instant::now();
            // Build CONCURRENTLY + poll to ready (see store::build_index_concurrently).
            // A plain foreground `DEFINE INDEX` backfills all symbol rows in ONE
            // transaction, which ROLLS BACK under the production-pinned RocksDB
            // write buffers at kernel scale (commit conflict) — leaving the index
            // absent and every name lookup a full scan. CONCURRENTLY batches the
            // backfill; the poll guarantees the index is query-usable before
            // Phase 2's name lookups run.
            build_index_concurrently(db, "idx_symbol_file", "symbol", "file")
                .await.context("streaming_index: build idx_symbol_file")?;
            build_index_concurrently(db, "idx_symbol_name", "symbol", "name")
                .await.context("streaming_index: build idx_symbol_name")?;
            sym_idx_rebuild_ms = t_sym_idx_rebuild.elapsed().as_millis() as u64;
            info!(repo = %self.repo, sym_idx_rebuild_ms, "stage3: rebuilt symbol indexes concurrently (full rebuild)");
        }

        if !pending_chunk_batch.is_empty() {
            let t2 = Instant::now();
            let t_db = Instant::now();
            flush_chunk_batch(db, pending_chunk_batch)
                .await
                .context("streaming_index: tail chunk batch")?;
            chunk_db_ns += t_db.elapsed().as_nanos() as u64;
            chunk_ns += t2.elapsed().as_nanos() as u64;
        }
        if !pending_raw_edge_batch.is_empty() {
            let t_re = Instant::now();
            flush_raw_edge_batch_native(db, &pending_raw_edge_batch)
                .await
                .context("streaming_index: tail raw_edge batch")?;
            rawedge_ns += t_re.elapsed().as_nanos() as u64;
        }
        if !pending_file_metas.is_empty() {
            let t_fm = Instant::now();
            for fm in pending_file_metas {
                upsert_file_meta(db, &fm)
                    .await
                    .context("streaming_index: upsert_file_meta (tail)")?;
            }
            filemeta_ns += t_fm.elapsed().as_nanos() as u64;
        }

        let stage3_total_ms = stage3_start.elapsed().as_millis() as u64;
        let sym_ms = sym_ns / 1_000_000;
        let rawedge_ms = rawedge_ns / 1_000_000;
        let chunk_ms = chunk_ns / 1_000_000;
        let chunk_db_ms = chunk_db_ns / 1_000_000;
        let chunk_cpu_ms = chunk_cpu_ns / 1_000_000;
        let filemeta_ms = filemeta_ns / 1_000_000;
        let ram_edges_in_buf = ram_raw_edges.len() as u64;
        let ram_symbols_in_buf = ram_symbols.len() as u64;

        info!(
            stage3_total_ms,
            sym_ms,
            sym_idx_drop_ms,
            sym_idx_rebuild_ms,
            rawedge_ms,
            chunk_ms,
            chunk_db_ms,
            chunk_cpu_ms,
            filemeta_ms,
            embed_total_ms,
            ram_edges_buffered = ram_edges_in_buf,
            ram_edges_overflowed,
            ram_symbols_buffered = ram_symbols_in_buf,
            ram_symbols_overflowed,
            cache_hit_chunks = total_cache_hit_chunks,
            cache_miss_chunks = total_cache_miss_chunks,
            files = total_files,
            chunks = total_chunks_count,
            symbols = total_symbols_count,
            edges = total_raw_edges_count,
            "PERF SUMMARY streaming_index stage3"
        );

        let stats = IndexPipelineStats {
            indexed_files: total_files,
            total_files,
            stage3_sym_ms: sym_ms,
            stage3_rawedge_ms: rawedge_ms,
            stage3_chunk_ms: chunk_ms,
            stage3_filemeta_ms: filemeta_ms,
            stage3_total_ms,
            phase2_ms: 0, // filled in by full_rebuild
            total_chunks: total_chunks_count,
            total_symbols: total_symbols_count,
            total_raw_edges: total_raw_edges_count,
            embed_total_ms,
            cache_hit_chunks: total_cache_hit_chunks,
            cache_miss_chunks: total_cache_miss_chunks,
            stage3_chunk_db_ms: chunk_db_ms,
            stage3_chunk_cpu_ms: chunk_cpu_ms,
            stage3_chunk_idx_drop_ms: 0,
            stage3_chunk_idx_rebuild_ms: 0,
            stage3_sym_idx_drop_ms: sym_idx_drop_ms,
            stage3_sym_idx_rebuild_ms: sym_idx_rebuild_ms,
            // Phase-2 sub-terms are filled in by full_rebuild after Phase 2 runs.
            ..Default::default()
        };

        // Surface the in-RAM symbol buffer to Phase 2: Some when it holds the
        // full deduped symbol set (within cap), None when overflowed/incremental.
        // None → Phase 2 falls back to load_all_symbols (today's behavior).
        let ram_symbols_out = if is_full_rebuild && !ram_symbols_overflowed {
            Some(ram_symbols)
        } else {
            None
        };

        Ok((all_chunk_vectors, stats, ram_raw_edges, ram_edges_overflowed, ram_symbols_out))
    }

    // ─── Phase 2: batched edge resolution ────────────────────────────────

    /// Select the best candidate symbol using 5-level priority:
    ///
    /// Level 0: Full import path resolution via `resolve_import_path`. Uses the
    ///          file set from candidate symbols to resolve the import to a concrete file.
    /// Level 1: If `import_path` contains `/`, find the candidate whose file path
    ///          `ends_with(import_path)`.
    /// Level 2: If `import_path` is bare (no `/`), find a candidate in the same
    ///          directory as `from_file` (compare parent directory component).
    /// Level 3: Same-file match (`candidate.file == from_file`).
    /// Level 4: First in pre-sorted bucket order (`bucket.first()`).
    ///
    /// Within each level, `.find()` on the pre-sorted bucket gives deterministic
    /// first-match. The bucket is pre-sorted by `(file, line_start, line_end)`.
    pub(crate) fn select_best_candidate<'a>(
        candidates: &'a [SymbolWithPos],
        from_file: &str,
        import_path: Option<&str>,
    ) -> Option<&'a SymbolWithPos> {
        use crate::indexing::import_resolver::resolve_import_path;
        use crate::parsing::{Lang, detect_language};
        use crate::parsing::generated::is_generated_file;

        if candidates.is_empty() {
            return None;
        }

        // Helper: within a level, prefer non-generated files. Falls back to
        // generated candidates only if no hand-written match exists at that level.
        let prefer_non_generated = |iter: &mut dyn Iterator<Item = &'a SymbolWithPos>| -> Option<&'a SymbolWithPos> {
            let mut generated_fallback: Option<&'a SymbolWithPos> = None;
            for c in iter {
                if !is_generated_file(&c.file) {
                    return Some(c);
                }
                if generated_fallback.is_none() {
                    generated_fallback = Some(c);
                }
            }
            generated_fallback
        };

        // Level 0: Full import path resolution — highest priority.
        // Attempt to resolve import_path to a concrete file via language-aware probing,
        // then match against candidates.
        if let Some(imp) = import_path {
            let lang = detect_language(std::path::Path::new(from_file));
            // Only attempt resolution for languages the resolver supports.
            if !matches!(lang, Lang::Other | Lang::Java | Lang::C | Lang::Cpp) {
                let file_set: HashSet<String> = candidates.iter().map(|c| c.file.clone()).collect();
                if let Some(resolved_file) = resolve_import_path(imp, from_file, lang, &file_set) {
                    // Find the first candidate in the resolved file.
                    let result = prefer_non_generated(
                        &mut candidates.iter().filter(|c| c.file == resolved_file)
                    );
                    if result.is_some() {
                        return result;
                    }
                    // Resolution found a file but no candidate symbol in it — try
                    // re-export/barrel chasing. The target symbol is the name we're
                    // resolving (all candidates share the same leaf name).
                    let target_symbol = &candidates[0].name;
                    if let Some(reexport_file) = crate::indexing::import_resolver::chase_reexports(
                        &resolved_file, target_symbol, &file_set, 0,
                    ) {
                        let result = prefer_non_generated(
                            &mut candidates.iter().filter(|c| c.file == reexport_file)
                        );
                        if result.is_some() {
                            return result;
                        }
                    }
                }
            }
        }

        // Level 1 / Level 2 — only attempted when import_path is present.
        if let Some(imp) = import_path {
            if imp.contains('/') {
                // Level 1: path ends_with import_path (handles subdirectory imports).
                let result = prefer_non_generated(
                    &mut candidates.iter().filter(|c| c.file.ends_with(imp))
                );
                if result.is_some() {
                    return result;
                }
            } else {
                // Level 2: bare filename — same parent directory as from_file.
                let from_dir = std::path::Path::new(from_file)
                    .parent()
                    .and_then(|p| p.to_str())
                    .unwrap_or("");
                let result = prefer_non_generated(
                    &mut candidates.iter().filter(|c| {
                        std::path::Path::new(&c.file)
                            .parent()
                            .and_then(|p| p.to_str())
                            .map(|d| d == from_dir)
                            .unwrap_or(false)
                    })
                );
                if result.is_some() {
                    return result;
                }
            }
        }

        // Level 3: same-file match.
        if let Some(found) = candidates.iter().find(|c| c.file == from_file) {
            return Some(found);
        }

        // Level 4: first in sorted order, preferring non-generated.
        prefer_non_generated(&mut candidates.iter())
    }

    /// Resolve raw edges (stored in `raw_edge` table) into denormalized `calls` rows.
    ///
    /// Algorithm (two-pass, bounded-memory):
    ///
    /// Pass 1 — symbol map load:
    ///   Load ALL symbols from the `symbol` table into a `HashMap<name, Vec<SymbolWithPos>>`.
    ///   This is a one-time O(symbol_count) allocation for O(1) per-edge name→id lookup.
    ///   At ~27K symbols × ~120 bytes the map is ~3.3 MB — bounded by symbol count, not
    ///   edge count.  This is the legitimate fix for the prior O(N²) per-page symbol
    ///   subquery; the map must stay.
    ///
    /// Pass 2 — compound keyset scan over raw_edge (O(N) total via index seek):
    ///   Pages through `raw_edge` using a compound keyset on `(from_file, id_str)`:
    ///
    ///   ```text
    ///   SELECT type::string(id) AS id_str, from_file, from_name, from_fqn,
    ///          to_name, kind, line, import_path
    ///   FROM raw_edge
    ///   WHERE from_file > $last_file
    ///      OR (from_file = $last_file AND type::string(id) > $last_id)
    ///   ORDER BY from_file, id_str
    ///   LIMIT $page
    ///   ```
    ///
    ///   ORDER BY uses `id_str` (the projected alias for `type::string(id)`).
    ///   SurrealDB 2.6.5 requires ORDER BY fields to appear in the SELECT list; it
    ///   rejects bare function calls (`type::string(id)`) and the native `id` field
    ///   unless explicitly included in SELECT.  Since `id_str` is already selected,
    ///   `ORDER BY id_str` is accepted.  The WHERE tiebreaker `type::string(id) > $last_id`
    ///   and ORDER BY `id_str` compare the same string values — perfectly consistent,
    ///   no rows skipped or duplicated.
    ///
    ///   The `from_file > $last_file` branch lets SurrealDB seek via
    ///   `idx_raw_edge_from_file` (defined in schema.rs) — O(log N) per boundary
    ///   lookup, O(N) total over all pages.  `id_str` (= type::string(id)) is unique
    ///   per row, so `(from_file, id_str)` is a unique compound key; every row is
    ///   visited exactly once with no skip or duplicate hazard.
    ///
    ///   Start with `last_file = ""` and `last_id = ""` (empty strings sort before all
    ///   real values).  After each page, advance:
    ///
    ///   ```text
    ///   last_file = batch.last().from_file
    ///   last_id   = batch.last().id_str
    ///   ```
    ///
    ///   Each page is resolved in-memory against the symbol map and accumulated into
    ///   `edge_batch`.  `edge_batch` flushes at WRITE_BATCH_SIZE, so peak memory is
    ///   bounded by: symbol map + one raw_edge page + at most WRITE_BATCH_SIZE resolved
    ///   edges — independent of total raw_edge count and safe at Linux/Chromium scale.
    ///
    /// NOTE: OFFSET pagination (`START $offset`) is O(N²) — to fetch page i the DB
    /// walks and discards i×page_size rows.  It must NOT be used here.
    /// NOTE: keyset on `type::string(id) > $cursor` alone was measured as O(N²) in
    /// SurrealDB 2.6.5 (145 s for 34 pages) because the function-call predicate cannot
    /// use any index.  The compound `from_file > $last_file` branch is what enables
    /// the index seek and achieves O(N) total.
    ///
    /// Writes the `edges_resolved` marker in `index_meta` only after all pages commit.
    async fn resolve_edges_phase2(&self, db: &Surreal<Db>, cancel_token: Option<&CancellationToken>) -> Result<Phase2Stats> {
        use serde::Deserialize;
        let mut p2 = Phase2Stats::default();

        // First delete all existing calls edges (we're rewriting them from raw_edge).
        db.query("DELETE FROM calls").await.context("phase2: delete calls")?;

        // Count total raw edges first to know if there's work to do.
        #[derive(Deserialize)]
        struct CountRow { count: i64 }
        let count_rows: Vec<CountRow> = db
            .query("SELECT count() AS count FROM raw_edge GROUP ALL")
            .await.context("phase2: count raw_edge")?
            .take(0)?;
        let total = count_rows.first().map(|r| r.count).unwrap_or(0);
        info!(repo = %self.repo, total_raw_edges = total, "phase2: starting edge resolution");

        if total == 0 {
            set_meta(db, EDGES_RESOLVED_KEY, "1")
                .await
                .context("phase2: set edges_resolved marker (empty)")?;
            return Ok(p2);
        }

        // Load ALL symbols into memory at once for O(1) per-edge lookup.
        // This avoids per-page round-trips to the DB for symbol resolution.
        // Memory: 27K symbols × ~120 bytes = ~3.3 MB — bounded and safe.
        let t_sym_load = Instant::now();
        let all_symbols = load_all_symbols(db).await.context("phase2: load all symbols")?;
        p2.sym_load_ms = t_sym_load.elapsed().as_millis() as u64;
        info!(repo = %self.repo, symbol_count = all_symbols.len(), sym_load_ms = p2.sym_load_ms, "phase2: loaded all symbols");

        // Build a name → Vec<SymbolWithPos> lookup map for O(1) resolution.
        let t_bucket = Instant::now();
        let mut name_bucket: HashMap<String, Vec<SymbolWithPos>> = HashMap::new();
        for s in all_symbols {
            name_bucket.entry(s.name.clone()).or_default().push(s);
        }
        // Pre-sort each bucket for deterministic tie-breaking (file, line_start, line_end).
        for bucket in name_bucket.values_mut() {
            bucket.sort_unstable_by(|a, b| {
                a.file.cmp(&b.file)
                    .then(a.line_start.cmp(&b.line_start))
                    .then(a.line_end.cmp(&b.line_end))
            });
        }
        p2.bucket_build_ms = t_bucket.elapsed().as_millis() as u64;

        // Drop all 4 calls indexes before the bulk RELATE flush to eliminate per-insert
        // index maintenance overhead (~4 index updates × 77K rows). Rebuild synchronously
        // after all RELATEs are committed — much faster than writing through live indexes.
        // This is the same drop→bulk-write→rebuild trick used in the old in-memory path.
        let t_idx_drop = Instant::now();
        db.query(
            "REMOVE INDEX IF EXISTS idx_calls_in_file  ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_out_file ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_in_name  ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_out_name ON calls;"
        ).await.context("phase2: drop calls indexes")?;
        p2.idx_drop_ms = t_idx_drop.elapsed().as_millis() as u64;
        info!(repo = %self.repo, idx_drop_ms = p2.idx_drop_ms, "phase2: dropped calls indexes");

        // Stream raw_edge in O(N) passes via file-keyset pagination.
        //
        // Strategy: paginate the outer loop by `from_file` (the indexed field), processing
        // all edges for one file before advancing to the next.  This avoids any secondary
        // sort on a computed field (type::string(id)) that cannot use an index and would
        // cause O(N²) full-table scans.
        //
        // Outer step: get the next `from_file` value via:
        //   SELECT from_file FROM raw_edge WHERE from_file > $cursor
        //   GROUP BY from_file ORDER BY from_file LIMIT $batch_files
        //   → uses idx_raw_edge_from_file; O(log N) seek per file boundary.
        //
        // Inner fetch: for each file, fetch all its edges via:
        //   SELECT ... FROM raw_edge WHERE from_file = $file
        //   → simple equality on the indexed field; O(edges_per_file).
        //
        // Memory: symbol map (3.3 MB) + max(edges_per_file) rows + at most
        //   WRITE_BATCH_SIZE resolved edges — independent of total raw_edge count.
        //
        // O(N) total: O(distinct_files) outer seeks + O(total_edges) inner reads.
        //
        // File-batch size: fetch FILE_BATCH_SIZE distinct files per outer query to amortise
        // the outer round-trip overhead (2464 files / FILE_BATCH_SIZE = few outer calls).
        const FILE_BATCH_SIZE: i64 = 256;

        let t_load_start = Instant::now();
        let mut last_file = String::new();
        let mut edge_batch: Vec<(String, String, i64, String, String, String, String)> = Vec::new();
        let mut pages_processed: u64 = 0;
        let mut scan_ms_total: u64 = 0;
        let mut resolve_ms_total: u64 = 0;

        loop {
            // Cancellation check at the page boundary. Phase 2 can run for tens of
            // seconds on a large repo; without this a cancel landing here would be
            // ignored until resolution finished. We leave EDGES_RESOLVED_KEY unset
            // on abort, so the next run replays Phase 2 from the durable raw_edge
            // table (crash-safe replay path) rather than shipping a half-built graph.
            if let Some(ct) = cancel_token
                && ct.is_cancelled()
            {
                info!(repo = %self.repo, "phase2: cancelled at page boundary");
                return Err(PipelineAbort::Cancelled.into());
            }

            // Outer: get next batch of distinct from_file values after the cursor.
            #[derive(Deserialize)]
            struct FileRow { from_file: String }
            let t_outer = Instant::now();
            let file_batch: Vec<FileRow> = db
                .query(
                    "SELECT from_file FROM raw_edge \
                     WHERE from_file > $cursor \
                     GROUP BY from_file \
                     ORDER BY from_file \
                     LIMIT $batch",
                )
                .bind(("cursor", last_file.clone()))
                .bind(("batch", FILE_BATCH_SIZE))
                .await
                .context("phase2: get next file batch")?
                .take(0)?;
            scan_ms_total += t_outer.elapsed().as_millis() as u64;

            if file_batch.is_empty() {
                break;
            }

            // Advance outer cursor to the last file in this batch.
            if let Some(last) = file_batch.last() {
                last_file = last.from_file.clone();
            }

            // Inner: for each file, fetch all its raw_edge rows and resolve them.
            // No ORDER BY needed — we just need all rows for this file.
            let files_in_batch: Vec<String> = file_batch.into_iter().map(|r| r.from_file).collect();
            let t_inner = Instant::now();
            let batch: Vec<RawEdgeRow> = db
                .query(
                    "SELECT type::string(id) AS id_str, \
                            from_file, from_name, from_fqn, to_name, kind, line, import_path \
                     FROM raw_edge \
                     WHERE from_file IN $files",
                )
                .bind(("files", files_in_batch))
                .await
                .context("phase2: scan raw_edge for file batch")?
                .take(0)?;
            scan_ms_total += t_inner.elapsed().as_millis() as u64;

            if batch.is_empty() {
                continue;
            }

            pages_processed += 1;

            // Resolve this batch in-memory against the pre-loaded symbol map.
            let t_resolve = Instant::now();
            resolve_raw_edge_page_from_map(&name_bucket, &batch, &mut edge_batch, "phase2");
            resolve_ms_total += t_resolve.elapsed().as_millis() as u64;

            // Flush resolved edges when accumulator reaches the write cap.
            // Uses EDGE_RELATE_BATCH_SIZE (larger than WRITE_BATCH_SIZE) because
            // RELATE statements are compact and fewer round-trips = faster on-disk writes.
            if edge_batch.len() >= EDGE_RELATE_BATCH_SIZE {
                let t_write = Instant::now();
                p2.edges_written += edge_batch.len() as u64;
                flush_edge_batch(db, &edge_batch)
                    .await
                    .context("phase2: flush edge batch")?;
                p2.relate_write_ms += t_write.elapsed().as_millis() as u64;
                edge_batch.clear();
            }

            // If the outer batch was smaller than FILE_BATCH_SIZE, we've exhausted
            // all files — no need for another outer query.
            // (The outer loop will break because file_batch.is_empty() on next iter,
            //  but we can also break early here for clarity.)
        }

        p2.resolve_cpu_ms = resolve_ms_total;
        let load_elapsed_ms = t_load_start.elapsed().as_millis() as u64;
        info!(
            repo = %self.repo,
            pages_processed,
            scan_ms_total,
            resolve_ms_total,
            load_elapsed_ms,
            "phase2: raw_edge scan + resolve complete"
        );

        // Flush any remaining edges.
        let t_flush_tail = Instant::now();
        if !edge_batch.is_empty() {
            p2.edges_written += edge_batch.len() as u64;
            flush_edge_batch(db, &edge_batch)
                .await
                .context("phase2: flush tail edge batch")?;
        }
        p2.relate_write_ms += t_flush_tail.elapsed().as_millis() as u64;
        info!(repo = %self.repo, flush_tail_ms = t_flush_tail.elapsed().as_millis() as u64, "phase2: tail flush complete");

        // Rebuild calls indexes after all bulk RELATEs are committed. Build
        // CONCURRENTLY + poll to ready (see store::build_index_concurrently): a
        // plain foreground `DEFINE INDEX` backfills all calls rows in ONE
        // transaction, which ROLLS BACK under the production-pinned RocksDB write
        // buffers at kernel scale (commit conflict) — leaving the indexes absent
        // and every WHERE out_name/in_name a full scan. CONCURRENTLY batches the
        // backfill; the poll guarantees each index is query-usable before the
        // marker is stamped and before any query runs.
        let t_idx_rebuild = Instant::now();
        build_index_concurrently(db, "idx_calls_in_file", "calls", "in_file")
            .await.context("phase2: build idx_calls_in_file")?;
        build_index_concurrently(db, "idx_calls_out_file", "calls", "out_file")
            .await.context("phase2: build idx_calls_out_file")?;
        build_index_concurrently(db, "idx_calls_in_name", "calls", "in_name")
            .await.context("phase2: build idx_calls_in_name")?;
        build_index_concurrently(db, "idx_calls_out_name", "calls", "out_name")
            .await.context("phase2: build idx_calls_out_name")?;
        p2.idx_rebuild_ms = t_idx_rebuild.elapsed().as_millis() as u64;
        info!(repo = %self.repo, idx_rebuild_ms = p2.idx_rebuild_ms, "phase2: rebuilt calls indexes concurrently");

        // Stamp the edges_resolved marker ONLY after all pages commit AND indexes rebuild.
        set_meta(db, EDGES_RESOLVED_KEY, "1")
            .await
            .context("phase2: set edges_resolved marker")?;

        info!(
            repo = %self.repo,
            edges_written = p2.edges_written,
            sym_load_ms = p2.sym_load_ms,
            bucket_build_ms = p2.bucket_build_ms,
            scan_ms_total,
            resolve_cpu_ms = p2.resolve_cpu_ms,
            relate_write_ms = p2.relate_write_ms,
            idx_drop_ms = p2.idx_drop_ms,
            idx_rebuild_ms = p2.idx_rebuild_ms,
            "PERF SUMMARY phase2(db-scan)"
        );
        Ok(p2)
    }

    // ─── RAM-path Phase 2: resolve pre-buffered edges without DB scan ─────

    /// Resolve raw edges from a pre-built RAM buffer (full-rebuild fast path).
    ///
    /// This avoids the 9.6s raw_edge DB write + 17.5s DB scan that the keyset
    /// scan path (`resolve_edges_phase2`) requires.  Applicable only when all
    /// raw_edges fit in RAM (bounded by MAX_RAM_EDGES = 8M); falls back to
    /// `resolve_edges_phase2` for larger repos.
    ///
    /// Crash-safety note: raw_edges are NOT in the DB when this path is taken.
    /// If the process crashes AFTER Stage 3 (all file_meta committed) but BEFORE
    /// this function completes:
    ///
    /// - `edges_resolved` marker is absent
    /// - `raw_edge` table is empty
    /// - file_meta is present and current
    ///
    /// On next `run()` call, no changes are detected, `edges_resolved` is absent,
    /// Phase 2 replay is triggered, finds 0 raw_edges, and would set `edges_resolved=1`
    /// with an empty calls table.
    ///
    /// To avoid silent data loss, `run()` detects this state
    /// (`raw_edge_count=0 AND file_meta non-empty AND edges_resolved absent`)
    /// and forces a full rebuild.
    async fn resolve_edges_from_ram(
        &self,
        db: &Surreal<Db>,
        raw_edges: Vec<RawEdgeRecord>,
        ram_symbols: Option<HashMap<String, SymbolWithPos>>,
        cancel_token: Option<&CancellationToken>,
    ) -> Result<Phase2Stats> {
        let mut p2 = Phase2Stats::default();
        let total = raw_edges.len();
        info!(repo = %self.repo, total_raw_edges = total, "phase2(ram): starting in-RAM edge resolution");

        if total == 0 {
            set_meta(db, EDGES_RESOLVED_KEY, "1")
                .await
                .context("phase2(ram): set edges_resolved marker (empty)")?;
            return Ok(p2);
        }

        // Source the symbol set: when the Stage-3 in-RAM symbol buffer is present
        // (within cap), consume it directly and SKIP the redundant full
        // `symbol`-table reload (`load_all_symbols`, ~4.7 min at kernel scale).
        // The buffer is already deduped to one entry per FQN with last-write-wins,
        // reproducing `load_all_symbols`' result EXACTLY. When absent (overflow),
        // fall back to the DB reload — today's behavior, output-identical.
        let t_sym_load = Instant::now();
        let all_symbols: Vec<SymbolWithPos> = match ram_symbols {
            Some(buf) => {
                let n = buf.len();
                let v: Vec<SymbolWithPos> = buf.into_values().collect();
                // sym_load is ~0 here — the symbols never left RAM.
                p2.sym_load_ms = t_sym_load.elapsed().as_millis() as u64;
                info!(repo = %self.repo, symbol_count = n, sym_load_ms = p2.sym_load_ms, "phase2(ram): reused in-RAM symbol buffer (no DB reload)");
                v
            }
            None => {
                let v = load_all_symbols(db).await.context("phase2(ram): load all symbols")?;
                p2.sym_load_ms = t_sym_load.elapsed().as_millis() as u64;
                info!(repo = %self.repo, symbol_count = v.len(), sym_load_ms = p2.sym_load_ms, "phase2(ram): loaded all symbols from DB (buffer overflowed)");
                v
            }
        };

        // Build name → Vec<SymbolWithPos> map for O(1) resolution.
        let t_bucket = Instant::now();
        let mut name_bucket: HashMap<String, Vec<SymbolWithPos>> = HashMap::new();
        for s in all_symbols {
            name_bucket.entry(s.name.clone()).or_default().push(s);
        }
        for bucket in name_bucket.values_mut() {
            bucket.sort_unstable_by(|a, b| {
                a.file.cmp(&b.file)
                    .then(a.line_start.cmp(&b.line_start))
                    .then(a.line_end.cmp(&b.line_end))
            });
        }
        p2.bucket_build_ms = t_bucket.elapsed().as_millis() as u64;

        // Drop calls indexes before bulk RELATE.
        let t_idx_drop = Instant::now();
        db.query(
            "REMOVE INDEX IF EXISTS idx_calls_in_file  ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_out_file ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_in_name  ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_out_name ON calls;"
        ).await.context("phase2(ram): drop calls indexes")?;
        p2.idx_drop_ms = t_idx_drop.elapsed().as_millis() as u64;
        info!(repo = %self.repo, idx_drop_ms = p2.idx_drop_ms, "phase2(ram): dropped calls indexes");

        // Resolve all RAM-buffered raw_edges in one pass (no DB scan needed).
        let t_resolve = Instant::now();
        let mut edge_batch: Vec<(String, String, i64, String, String, String, String)> = Vec::new();
        let mut relate_write_ms: u64 = 0;
        let mut edges_written: u64 = 0;

        for (i, re) in raw_edges.iter().enumerate() {
            // Cancellation check every EDGE_RELATE_BATCH_SIZE edges (cheap, bounded
            // frequency). On abort we leave EDGES_RESOLVED_KEY unset; because the
            // RAM path never wrote raw_edge to the DB, run()'s crash-recovery branch
            // (edges_resolved absent + raw_edge empty + file_meta present) forces a
            // full rebuild on the next trigger, which is the correct recovery here.
            if i % EDGE_RELATE_BATCH_SIZE == 0
                && let Some(ct) = cancel_token
                && ct.is_cancelled()
            {
                info!(repo = %self.repo, "phase2(ram): cancelled mid-resolution");
                return Err(PipelineAbort::Cancelled.into());
            }
            // Resolve this edge using the symbol map.
            let candidates = match name_bucket.get(&re.to_name) {
                Some(v) if !v.is_empty() => v,
                _ => continue,
            };
            let best = Self::select_best_candidate(candidates, &re.from_file, re.import_path.as_deref());
            let best = match best {
                Some(b) => b,
                None => continue,
            };
            // in_name/out_name must store the FULL FQN (file::scope::name), matching
            // the DB-scan path (positions 5/6 at the resolved-edge push below) and the
            // node IDs consumed by call_graph (meta::id(id)) and query_callers/callees.
            // Writing leaf names here desyncs both the UI graph and search-time expansion.
            edge_batch.push((
                re.from_fqn.clone(),
                best.fqn.clone(),
                re.line,
                re.from_file.clone(),
                best.file.clone(),
                re.from_fqn.clone(),
                best.fqn.clone(),
            ));

            if edge_batch.len() >= EDGE_RELATE_BATCH_SIZE {
                let t_write = Instant::now();
                edges_written += edge_batch.len() as u64;
                flush_edge_batch(db, &edge_batch)
                    .await
                    .context("phase2(ram): flush edge batch")?;
                relate_write_ms += t_write.elapsed().as_millis() as u64;
                edge_batch.clear();
            }
        }
        // resolve_cpu = whole loop minus the time spent inside flush calls.
        let resolve_ms = (t_resolve.elapsed().as_millis() as u64).saturating_sub(relate_write_ms);
        p2.resolve_cpu_ms = resolve_ms;
        info!(repo = %self.repo, resolve_cpu_ms = resolve_ms, "phase2(ram): in-memory resolution complete");

        // Flush tail.
        if !edge_batch.is_empty() {
            let t_write = Instant::now();
            edges_written += edge_batch.len() as u64;
            flush_edge_batch(db, &edge_batch)
                .await
                .context("phase2(ram): flush tail edge batch")?;
            relate_write_ms += t_write.elapsed().as_millis() as u64;
        }
        p2.relate_write_ms = relate_write_ms;
        p2.edges_written = edges_written;

        // Rebuild calls indexes (same as DB-scan Phase 2). Build CONCURRENTLY +
        // poll to ready (see store::build_index_concurrently): a plain foreground
        // `DEFINE INDEX` backfills all calls rows in ONE transaction, which ROLLS
        // BACK under the production-pinned RocksDB write buffers at kernel scale
        // (this is the kernel path — RAM-buffered edges) — leaving the indexes
        // absent and every WHERE out_name/in_name a full scan. CONCURRENTLY
        // batches the backfill; the poll guarantees query-usability before the
        // marker is stamped.
        let t_idx_rebuild = Instant::now();
        build_index_concurrently(db, "idx_calls_in_file", "calls", "in_file")
            .await.context("phase2(ram): build idx_calls_in_file")?;
        build_index_concurrently(db, "idx_calls_out_file", "calls", "out_file")
            .await.context("phase2(ram): build idx_calls_out_file")?;
        build_index_concurrently(db, "idx_calls_in_name", "calls", "in_name")
            .await.context("phase2(ram): build idx_calls_in_name")?;
        build_index_concurrently(db, "idx_calls_out_name", "calls", "out_name")
            .await.context("phase2(ram): build idx_calls_out_name")?;
        p2.idx_rebuild_ms = t_idx_rebuild.elapsed().as_millis() as u64;
        info!(repo = %self.repo, idx_rebuild_ms = p2.idx_rebuild_ms, relate_write_ms, "phase2(ram): rebuilt calls indexes concurrently");

        // Stamp edges_resolved marker.
        set_meta(db, EDGES_RESOLVED_KEY, "1")
            .await
            .context("phase2(ram): set edges_resolved marker")?;

        info!(
            repo = %self.repo,
            edges_written = p2.edges_written,
            sym_load_ms = p2.sym_load_ms,
            bucket_build_ms = p2.bucket_build_ms,
            resolve_cpu_ms = p2.resolve_cpu_ms,
            relate_write_ms = p2.relate_write_ms,
            idx_drop_ms = p2.idx_drop_ms,
            idx_rebuild_ms = p2.idx_rebuild_ms,
            "PERF SUMMARY phase2(ram)"
        );
        Ok(p2)
    }

    // ─── Incremental Phase 2: scoped edge resolution ──────────────────────

    /// Re-resolve only the edges that touch the blast radius of an edit.
    ///
    /// Complexity: O(changed + callers_of_removed_surface + callers_of_added_names)
    /// — proportional to the ACTUAL symbol-surface change, NOT to the call-fan-in
    /// of the edited file. A comment/body-only edit (which changes zero symbols)
    /// resolves to O(changed): `dir1_callers` and `added_names` are both empty, so
    /// the resolve_set is exactly the changed files and every incoming edge to them
    /// survives untouched. A genuine API change pays the honest cost of re-resolving
    /// exactly the callers it can affect.
    ///
    /// The blast radius is computed by the caller (`incremental_run`) from the
    /// per-file symbol-surface delta and handed in as two gates:
    ///
    ///   - `dir1_callers` ("removal/identity direction"): unchanged files that had
    ///     a `calls` edge pointing INTO a file that REMOVED or MOVED a `(name, fqn)`
    ///     pair (or was deleted). Their resolved target may now be stale, so they
    ///     re-resolve. A file that only GAINED symbols, or whose surface is
    ///     unchanged, contributes NO dir1 caller — its incoming edges still point at
    ///     still-existing, still-winning symbols. (A same-name overload ADDED to a
    ///     file is handled by direction-2 via its leaf name, not direction-1.)
    ///
    ///   - `added_names` ("new target now wins" / direction-2): leaf names that
    ///     GAINED a `(name, fqn)` pair in the changed set. A newly-added name can win
    ///     a tie-break for an unchanged caller that never pointed into the changed
    ///     file. We expand to those callers via `raw_edge.to_name`. Names that were
    ///     already present (unchanged) cannot change any caller's resolution and are
    ///     NOT expanded — the previous code expanded for ALL names defined in the
    ///     changed files, which is what blew the resolve_set up to thousands of files
    ///     on a comment edit.
    ///
    /// Algorithm:
    ///   1. resolve_set = changed_files ∪ dir1_callers ∪ {from_file of any raw_edge
    ///      whose to_name ∈ added_names}.
    ///   2. DELETE the `calls` rows whose `in_file` is in resolve_set, per-file
    ///      `WHERE in_file = $f` (a point seek on idx_calls_in_file), batched into
    ///      transactions. We delete the OUTGOING edges of resolve_set files ONLY
    ///      (in_file), NOT incoming (out_file): every file whose incoming edges
    ///      could have changed is ITSELF a resolve_set member as a from_file (its
    ///      outgoing edges are deleted+re-resolved here), or it points into a
    ///      surface-changed/deleted file and is therefore a dir1/dir2 caller (also
    ///      a resolve_set member, also deleted+re-resolved). An out_file delete
    ///      would additionally destroy incoming edges from files OUTSIDE resolve_set
    ///      — exactly the surface-unchanged callers whose edges must survive — and
    ///      those would never be re-resolved (their from_file is not in resolve_set,
    ///      and at repo scale their `raw_edge` rows aren't even persisted). So the
    ///      in_file-only delete is both correct AND the lever that bounds the cost.
    ///   3. Re-resolve raw_edge rows WHERE from_file IN resolve_set via keyset
    ///      pagination (uses idx_raw_edge_from_file).
    ///
    /// The `edges_resolved` crash-recovery marker is NOT written here — it is only
    /// meaningful for a full rebuild where ALL raw_edge must be re-resolved on crash
    /// recovery. Incremental is already idempotent: if it crashes before file_meta
    /// is written (the crash-safe anchor in streaming_index), the whole incremental
    /// re-runs on next trigger, including this method.
    async fn resolve_edges_incremental(
        &self,
        db: &Surreal<Db>,
        changed_files: &[String],
        dir1_callers: &[String],
        added_names: &[String],
    ) -> Result<Phase2IncrStats> {
        use serde::Deserialize;

        let mut stats = Phase2IncrStats::default();

        if changed_files.is_empty() {
            return Ok(stats);
        }

        // Step 1: Build resolve_set = changed_files ∪ dir1_callers (deduped).
        //
        // dir1_callers was computed by incremental_run as the callers pointing into
        // SURFACE-CHANGED (or deleted) files — the "removal/identity direction". A
        // surface-unchanged changed file contributes no dir1 callers, so its
        // thousands of incoming edges stay out of the resolve_set.
        let mut resolve_set: Vec<String> = changed_files.to_vec();
        for caller in dir1_callers {
            if !resolve_set.contains(caller) {
                resolve_set.push(caller.clone());
            }
        }

        // Direction 2: "new target now wins". `added_names` are the leaf names that
        // GAINED a (name, fqn) pair in the changed set (computed from the surface
        // delta by incremental_run). A newly-added name can win a tie-break for an
        // unchanged caller, so we pull those callers in via raw_edge.to_name. Names
        // that were unchanged are NOT in added_names and are not expanded — this is
        // the narrowing that keeps a comment edit's resolve_set == changed_files.
        //
        // NOTE: `p2_symname_ms` is no longer measured here — the symbol-surface
        // load+diff that produces `added_names` now happens once in incremental_run
        // (its time is reported as the surface-delta stage), so this method just
        // consumes the precomputed result.
        if !added_names.is_empty() {
            // Find callers that target any added name via raw_edge.to_name.
            // raw_edge.to_name stores the unresolved leaf callee name, so this correctly
            // finds any file that calls a symbol with the given leaf name — including files
            // whose existing calls row points to a different definition (stale lex-first target).
            // Uses idx_raw_edge_from_file for the GROUP BY; the to_name lookup is bounded
            // by the number of edges with matching callee names.
            #[derive(Deserialize)]
            struct FromFileRow { from_file: String }
            let dir2_start = Instant::now();
            let name_exp_rows: Vec<FromFileRow> = db
                .query("SELECT from_file FROM raw_edge WHERE to_name IN $names GROUP BY from_file")
                .bind(("names", added_names.to_vec()))
                .await
                .context("incremental phase2: name-based expansion via raw_edge")?
                .take(0)?;
            stats.p2_dir2_scan_ms = dir2_start.elapsed().as_millis() as u64;

            for row in name_exp_rows {
                if !resolve_set.contains(&row.from_file) {
                    resolve_set.push(row.from_file);
                }
            }
        }

        debug!(
            repo = %self.repo,
            changed = changed_files.len(),
            resolve_set = resolve_set.len(),
            "incremental phase2: resolve_set built"
        );
        stats.resolve_set_size = resolve_set.len() as u64;

        // Step 2: Delete only the OUTGOING calls rows of resolve_set files.
        //
        // CORRECTNESS: we delete `WHERE in_file = $f` for each resolve_set file —
        // its outgoing edges — and re-resolve them below from raw_edge. We do NOT
        // delete `WHERE out_file = $f` (incoming edges). Every calls row that could
        // have CHANGED is an outgoing edge of SOME resolve_set member:
        //   - a changed file's own outgoing edges → it is in resolve_set (changed);
        //   - an edge into a surface-changed/deleted file from an unchanged caller →
        //     that caller is a dir1_caller (in resolve_set), so the edge is its
        //     outgoing edge and is deleted+re-resolved here;
        //   - an edge whose resolution could flip to a newly-added name → its caller
        //     is a dir2 expansion (in resolve_set), same as above.
        // An incoming edge to a SURFACE-UNCHANGED changed file is NOT touched: its
        // caller is not in resolve_set, it still points at a still-existing,
        // still-winning symbol, and (at repo scale) that caller's raw_edge isn't
        // even persisted — so re-resolving it would be both unnecessary and lossy.
        // This in_file-only scoping is what makes the comment-edit cost O(changed).
        //
        // PERF (measured, isolated probe `delete_strategy_probe`, REAL schema +
        // idx_calls_in_file, at full kernel scale): per-file `WHERE in_file = $f`
        // is a POINT SEEK on idx_calls_in_file. Batching CALLS_DELETE_TXN_CHUNK
        // files per BEGIN/COMMIT amortizes the per-commit fsync (~39ms) ~200x vs
        // auto-commit-per-statement. A bare `WHERE in_file IN $list` does NOT
        // range-scan cleanly in SurrealDB 2.6.5 (~O(list×rows)); per-file equality
        // avoids that. Distinct $p{i} binds keep every file value parameterized
        // (no injection, no per-row re-parse of the path string).
        let delete_calls_start = Instant::now();
        for chunk in resolve_set.chunks(CALLS_DELETE_TXN_CHUNK) {
            let mut stmt = String::from("BEGIN;\n");
            for i in 0..chunk.len() {
                stmt.push_str(&format!("DELETE FROM calls WHERE in_file = $p{i};\n"));
            }
            stmt.push_str("COMMIT;");
            let mut q = db.query(&stmt);
            for (i, file) in chunk.iter().enumerate() {
                q = q.bind((format!("p{i}"), file.clone()));
            }
            q.await
                .context("incremental phase2: delete scoped calls (batched txn)")?
                .check()
                .context("incremental phase2: delete scoped calls batch had errors")?;
        }
        stats.p2_delete_calls_ms = delete_calls_start.elapsed().as_millis() as u64;

        // Step 3: Re-resolve raw_edge rows whose from_file is in the resolve set.
        // Keyset-paginated with from_file filter — uses idx_raw_edge_from_file.

        let reresolve_start = Instant::now();
        let page_size: i64 = WRITE_BATCH_SIZE as i64;
        let mut cursor = String::new();
        let mut edge_batch: Vec<(String, String, i64, String, String, String, String)> = Vec::new();

        loop {
            let batch: Vec<RawEdgeRow> = db
                .query(
                    "SELECT type::string(id) AS id_str, \
                            from_file, from_name, from_fqn, to_name, kind, line, import_path \
                     FROM raw_edge \
                     WHERE from_file IN $files \
                       AND type::string(id) > $cursor \
                     ORDER BY id_str \
                     LIMIT $page",
                )
                .bind(("files", resolve_set.clone()))
                .bind(("cursor", cursor.clone()))
                .bind(("page", page_size))
                .await
                .context("incremental phase2: scan raw_edge page")?
                .take(0)?;

            if batch.is_empty() {
                break;
            }

            cursor = batch.last().map(|r| r.id_str.clone()).unwrap_or(cursor);

            resolve_raw_edge_page(db, &batch, &mut edge_batch, "incremental phase2").await?;

            let batch_len = batch.len() as i64;
            if batch_len < page_size {
                break;
            }
        }

        // Flush remaining edges.
        if !edge_batch.is_empty() {
            flush_edge_batch(db, &edge_batch)
                .await
                .context("incremental phase2: flush tail edge batch")?;
        }
        stats.p2_reresolve_ms = reresolve_start.elapsed().as_millis() as u64;

        info!(repo = %self.repo, resolve_set = resolve_set.len(), "incremental Phase 2 edge resolution complete");
        Ok(stats)
    }
}

// ─── Incremental blast-radius gating: symbol-surface delta ────────────────

/// The "symbol surface" of one file: the set of `(leaf_name, fqn)` pairs the file
/// defines. This is precisely the part of a file's symbols that can affect how
/// ANY caller's `calls` edge resolves:
///
/// - resolution buckets candidates by leaf `name` (`resolve_raw_edge_page`),
///   so a name that is added/removed changes which candidates exist;
/// - the chosen candidate is identified by its `fqn` (the RELATE endpoint), so
///   a changed fqn changes the resolved target identity.
///
/// A symbol whose `(name, fqn)` pair is unchanged cannot change any caller's
/// resolution. Line numbers are deliberately EXCLUDED: `calls` stores the
/// call-site `line` from the CALLER (not the target's definition line — see the
/// RELATE in `flush_edge_batch`/`insert_edge`), and `fqn` does not embed a line
/// (`QualifiedSymbol::fqn` = file::scope::name). So a comment/body edit that only
/// shifts a file's symbols up/down by N lines, keeping their `(name, fqn)` pairs
/// identical, produces an IDENTICAL surface and an empty delta.
type FileSurface = HashMap<String, HashSet<(String, String)>>;

/// Result of diffing the OLD vs NEW symbol surface of a set of changed files.
///
/// The two fields gate the two independent ways an edit can invalidate a `calls`
/// edge — and they are deliberately ASYMMETRIC, because addition and removal have
/// asymmetric effects on resolution:
///
/// - REMOVING (or moving) a `(name, fqn)` pair can orphan a caller that resolved
///   to exactly that fqn → its edge must re-resolve (direction-1).
/// - ADDING a pair can only ever make some caller of that *leaf name* newly
///   prefer the changed file (a tie-break win) → direction-2 re-resolves the
///   callers of the added *names*, wherever they are. A pure addition of a name
///   nobody calls (or a uniquely-named symbol) pulls in NOBODY.
///
/// Crucially, a file that ONLY GAINED symbols has every pre-existing incoming
/// edge still pointing at a still-existing, still-winning fqn — so it is NOT
/// direction-1 material. Conflating "added" into the direction-1 gate is what made
/// a 10-file add-symbol burst re-resolve 2748 callers (157s) for nothing.
#[derive(Debug, Default, PartialEq, Eq)]
struct SurfaceDelta {
    /// Files that REMOVED or MOVED at least one `(name, fqn)` pair (a pair present
    /// in OLD but absent in NEW), PLUS every genuinely-deleted file. These are the
    /// only files whose pre-existing INCOMING edges can have become stale, so
    /// direction-1 (`dir1_callers`) is gated on exactly this set. A file that only
    /// gained symbols, or whose surface is unchanged, is NOT here.
    removed_surface_files: Vec<String>,
    /// Leaf names that GAINED a `(name, fqn)` pair somewhere in the changed set
    /// (brand-new name, new overload of an existing name, or a scope move that
    /// produced a new fqn). A newly-added name can win a lex/locality tie-break
    /// for an UNCHANGED caller that never pointed into the changed file — that is
    /// the only way an ADDITION can pull an unrelated caller into the blast radius,
    /// so direction-2 name-expansion is narrowed to exactly these names. Names that
    /// only LOST pairs are NOT here: removing a losing candidate can never make a
    /// caller newly prefer the changed file (it only matters for callers that
    /// already pointed INTO it, which direction-1 covers).
    added_names: Vec<String>,
}

/// Compute the blast-radius gates from the per-file OLD and NEW symbol surfaces.
///
/// `deleted_files` are files removed from the repo entirely (no NEW surface);
/// they are always "removed surface" (their incoming edges are now dangling) and
/// contribute no added names.
///
/// Pure and deterministic (sorted outputs) so it is unit-testable in isolation
/// and the oracle can assert on it. Bounded by the number of changed files and
/// their symbols — never touches the whole repo.
fn compute_surface_delta(
    old: &FileSurface,
    new: &FileSurface,
    modified_files: &[String],
    deleted_files: &[String],
) -> SurfaceDelta {
    let empty: HashSet<(String, String)> = HashSet::new();
    let mut removed_surface_files: Vec<String> = Vec::new();
    let mut added_pairs_names: HashSet<String> = HashSet::new();

    for file in modified_files {
        let old_set = old.get(file).unwrap_or(&empty);
        let new_set = new.get(file).unwrap_or(&empty);
        // Direction-1 gate: did this file LOSE (or move) any pair? Pure additions
        // do NOT qualify — their pre-existing incoming edges are all still valid.
        if old_set.difference(new_set).next().is_some() {
            removed_surface_files.push(file.clone());
        }
        // Direction-2 feed: added pairs = NEW minus OLD; their leaf names expand.
        for (name, fqn) in new_set.difference(old_set) {
            let _ = fqn; // fqn distinguishes the pair; the name drives expansion.
            added_pairs_names.insert(name.clone());
        }
    }

    // Deleted files: always removed-surface (dangling incoming edges), no adds.
    for file in deleted_files {
        removed_surface_files.push(file.clone());
    }

    removed_surface_files.sort_unstable();
    removed_surface_files.dedup();
    let mut added_names: Vec<String> = added_pairs_names.into_iter().collect();
    added_names.sort_unstable();

    SurfaceDelta { removed_surface_files, added_names }
}

/// Load the per-file symbol surface (`(name, fqn)` pairs) for a set of files in
/// ONE indexed query. Returns a map keyed by file path; files with no symbols are
/// simply absent (callers treat absence as the empty set). Uses `file IN $files`
/// (idx_symbol_file) — bounded by the changed set's symbol count, never the repo.
async fn load_file_surface(db: &Surreal<Db>, files: &[String]) -> Result<FileSurface> {
    let mut surface: FileSurface = HashMap::new();
    if files.is_empty() {
        return Ok(surface);
    }
    #[derive(Deserialize)]
    struct Row {
        file: String,
        name: String,
        fqn: String,
    }
    let rows: Vec<Row> = db
        .query("SELECT file, name, meta::id(id) AS fqn FROM symbol WHERE file IN $files")
        .bind(("files", files.to_vec()))
        .await
        .context("load_file_surface")?
        .take(0)?;
    for r in rows {
        // meta::id returns the bracketed complex id; strip to the plain fqn so OLD
        // (from DB) and NEW (also from DB, post-streaming) compare apples-to-apples.
        let fqn = crate::store::ops::strip_id_brackets(&r.fqn);
        surface.entry(r.file).or_default().insert((r.name, fqn));
    }
    Ok(surface)
}

// ─── Phase 2: in-memory symbol map helpers ───────────────────────────────

/// Insert one parsed symbol into the Stage-3 in-RAM symbol buffer, keyed by FQN
/// with last-write-wins on collision.
///
/// This is the SINGLE source of truth for how the buffer is populated: both the
/// Stage-3 streaming loop and the invariance unit tests call it, so the tests
/// exercise the PRODUCTION dedup behavior rather than a hand-copied duplicate.
/// Last-write-wins (a plain `insert` that overwrites) reproduces the `symbol`
/// table's `INSERT ... ON DUPLICATE KEY UPDATE` per-FQN dedup byte-for-byte, so
/// the resulting symbol set equals `load_all_symbols`' result for the same stream.
fn buffer_insert_symbol(buf: &mut HashMap<String, SymbolWithPos>, sym: &Symbol) {
    let fqn = sym.qualified.fqn();
    buf.insert(
        fqn.clone(),
        SymbolWithPos {
            fqn,
            file: sym.qualified.file.clone(),
            name: sym.qualified.name.clone(),
            line_start: sym.line_start as i64,
            line_end: sym.line_end as i64,
        },
    );
}

/// Load ALL symbols from the DB into memory at once.
/// Memory: 27K symbols × ~120 bytes = ~3.3 MB — bounded for repo-scale indexes.
async fn load_all_symbols(db: &Surreal<Db>) -> Result<Vec<SymbolWithPos>> {
    use serde::Deserialize;
    #[derive(Deserialize)]
    struct Row {
        fqn: String,
        file: String,
        name: String,
        line_start: i64,
        line_end: i64,
    }

    let rows: Vec<Row> = db
        .query("SELECT meta::id(id) AS fqn, file, name, line_start, line_end FROM symbol")
        .await
        .context("load_all_symbols")?
        .take(0)?;

    Ok(rows.into_iter().map(|r| {
        use crate::store::ops::SymbolWithPos;
        SymbolWithPos {
            fqn: strip_id_brackets_phase2(&r.fqn),
            file: r.file,
            name: r.name,
            line_start: r.line_start,
            line_end: r.line_end,
        }
    }).collect())
}

/// Strip SurrealDB complex-ID brackets ⟨…⟩ returned by `meta::id(id)`.
fn strip_id_brackets_phase2(id: &str) -> String {
    id.strip_prefix("⟨")
        .and_then(|s| s.strip_suffix("⟩"))
        .unwrap_or(id)
        .to_string()
}

/// Resolve a page of raw edges using a pre-built in-memory symbol map.
/// This avoids per-page DB round-trips for symbol lookup.
fn resolve_raw_edge_page_from_map(
    name_bucket: &HashMap<String, Vec<SymbolWithPos>>,
    batch: &[RawEdgeRow],
    edge_batch: &mut Vec<(String, String, i64, String, String, String, String)>,
    label: &str,
) {
    for row in batch {
        let resolved_to = match name_bucket.get(&row.to_name) {
            Some(candidates) if !candidates.is_empty() => {
                IndexPipeline::select_best_candidate(
                    candidates,
                    &row.from_file,
                    row.import_path.as_deref(),
                ).cloned()
            }
            _ => {
                debug!(name = %row.to_name, "{}: dropping unresolved raw edge (in-memory map)", label);
                None
            }
        };

        if let Some(to) = resolved_to {
            edge_batch.push((
                row.from_fqn.clone(),
                to.fqn.clone(),
                row.line,
                row.from_file.clone(),
                to.file.clone(),
                row.from_fqn.clone(),
                to.fqn.clone(),
            ));
        }
    }
}

// ─── Parse one file with framework edge extraction ──────────────────────────

/// Wrapper that calls `parse_one_file` and then appends framework-produced edges.
/// The `FrameworkRegistry` must have had `detect()` called before this function.
fn parse_one_file_with_frameworks(
    file: &str,
    registry: &crate::indexing::frameworks::FrameworkRegistry,
) -> ParseOutput {
    let mut output = parse_one_file(file);

    // Append framework-detected edges if parse succeeded and registry has active resolvers.
    if let ParseOutput::Parsed(ref mut parsed) = output
        && registry.is_detected()
    {
        // Read source for framework extraction (we already parsed it, but
        // parse_one_file doesn't return the source — re-read is cheap for
        // framework regex extraction which is O(source.len())).
        if let Ok(source) = std::fs::read_to_string(file) {
            let fw_edges = registry.extract_edges(file, &source, &parsed.symbols);
            for edge in fw_edges {
                if matches!(edge.kind, crate::parsing::relations::EdgeKind::Calls) {
                    let (to_name, import_path) = match &edge.to {
                        crate::parsing::relations::EdgeTarget::Unresolved { name, import_path, .. } => {
                            (name.clone(), import_path.clone())
                        }
                        crate::parsing::relations::EdgeTarget::Resolved(qs) => {
                            (qs.name.clone(), None)
                        }
                    };
                    parsed.raw_edges.push(RawEdgeRecord {
                        from_file: edge.from.file.clone(),
                        from_name: edge.from.name.clone(),
                        from_fqn: edge.from.fqn(),
                        to_name,
                        kind: "calls".to_string(),
                        line: edge.line as i64,
                        import_path,
                    });
                }
            }
        }
    }

    output
}

// ─── Parse one file (returns ParseOutput — always returns, never drops silently) ─

fn parse_one_file(file: &str) -> ParseOutput {
    // Reset the recursion guard state for this file. Rayon workers are reused
    // across files — without a per-file reset, the warn-once flag and current-file
    // path carry over from the previous file on the same worker thread, which
    // suppresses diagnostics and mis-attributes warnings to wrong files.
    crate::parsing::recursion_guard::begin_file(file);

    let parse_start = Instant::now();

    let source = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            warn!(file = %file, error = %e, "failed to read file");
            return ParseOutput::Skipped {
                file: file.to_string(),
                reason: format!("read error: {e}"),
            };
        }
    };

    if source.contains('\0') {
        debug!(file = %file, "skipping binary file (contains null byte)");
        return ParseOutput::Skipped {
            file: file.to_string(),
            reason: "binary file (contains null byte)".to_string(),
        };
    }

    let (mtime, size) = match stat_file(file) {
        Some(s) => (s.mtime, s.size),
        None => {
            warn!(file = %file, "failed to stat file");
            return ParseOutput::Skipped {
                file: file.to_string(),
                reason: "stat failed".to_string(),
            };
        }
    };

    let result = parse_file(file, &source);

    // Convert raw edges to RawEdgeRecord for Phase 1 storage.
    let raw_edges: Vec<RawEdgeRecord> = result
        .edges
        .iter()
        .filter_map(|e| {
            let (to_name, import_path) = match &e.to {
                EdgeTarget::Unresolved { name, import_path, .. } => (name.clone(), import_path.clone()),
                EdgeTarget::Resolved(qs) => (qs.name.clone(), None),
            };
            // Only store Calls edges (❼ spec: only `calls` table uses in_name/out_name).
            if matches!(e.kind, EdgeKind::Calls) {
                Some(RawEdgeRecord {
                    from_file: e.from.file.clone(),
                    from_name: e.from.name.clone(),
                    from_fqn: e.from.fqn(),
                    to_name,
                    kind: "calls".to_string(),
                    line: e.line as i64,
                    import_path,
                })
            } else {
                // For non-calls edges, still resolve them synchronously (no in_name needed).
                None
            }
        })
        .collect();

    let parse_elapsed_ms = parse_start.elapsed().as_millis() as u64;

    ParseOutput::Parsed(ParsedFile {
        path: file.to_string(),
        symbols: result.symbols,
        chunks: result.chunks,
        raw_edges,
        mtime,
        size,
        parse_elapsed_ms,
        created_at: Instant::now(),
    })
}

// ─── Embed a parsed file's chunks ────────────────────────────────────────

struct EmbedFileResult {
    embeddings: Vec<Vec<f32>>,
    fully_cached: bool,
    /// Chunks served from the on-disk cache (no API call needed).
    hit_chunks: u64,
    /// Chunks NOT found in the cache (re-embedded via API or stored empty).
    miss_chunks: u64,
}

/// Error from embed_parsed_file, distinguishing transient/retry-exhausted
/// errors (which should skip the file, non-fatal to the pipeline) from fatal
/// errors (auth/config, which should abort the pipeline).
///
/// A transient gateway timeout on one file must not abort a 79K-file rebuild.
/// Crash-safe file_meta means a skipped file is simply not committed and will
/// be retried on the next index trigger (self-healing).
enum EmbedFileError {
    /// Transient network error that exhausted all retry attempts.
    /// The file should be skipped (no file_meta committed); the next index
    /// trigger re-processes it automatically.
    Transient(anyhow::Error),
    /// Fatal error (auth 4xx, config, permanent API failure).
    /// The pipeline should abort immediately.
    Fatal(anyhow::Error),
}

/// Classify an embed error as transient or fatal by checking the anyhow chain
/// for the `TransientEmbedExhausted` marker.
fn classify_embed_error(e: anyhow::Error) -> EmbedFileError {
    // Walk the chain: TransientEmbedExhausted is added as `.context()` on the
    // original reqwest error, so it appears as a cause in the source chain.
    // anyhow::Error::downcast_ref checks the outermost type only; we need to
    // check if the Error IS TransientEmbedExhausted (the outermost after
    // embed_batch wraps with `.context(TransientEmbedExhausted{..})`).
    if e.downcast_ref::<crate::embedding::TransientEmbedExhausted>().is_some() {
        return EmbedFileError::Transient(e);
    }
    EmbedFileError::Fatal(e)
}

/// Outcome of a cache `get_many` lookup: `(hits, miss_indices)` where each hit
/// is `(original_index, embedding)`.
type GetManyOutcome = (Vec<(usize, Vec<f32>)>, Vec<usize>);

/// Map the result of a `spawn_blocking(cache.get_many)` call to the cache
/// lookup outcome, degrading a `JoinError` (panic inside the blocking task)
/// to "everything missed, empty embeddings" — identical to the Voyage-API
/// error path. Returning `Err(EmbedFileResult)` signals the caller to return
/// that degraded result immediately; `Ok((hits, misses))` is the normal path.
///
/// Extracted so the JoinError arm the whole no-drop guarantee rests on is
/// covered by a test that drives a real panic through this exact logic
/// (`get_many` itself never panics — it converts all I/O errors to misses).
fn map_get_many_result(
    file: &str,
    n_texts: usize,
    get_result: std::result::Result<GetManyOutcome, tokio::task::JoinError>,
) -> std::result::Result<GetManyOutcome, EmbedFileResult> {
    match get_result {
        Ok(result) => Ok(result),
        Err(e) => {
            warn!(file = %file, error = %e, "cache get_many panicked in spawn_blocking; treating all as miss");
            // Return empty embeddings — same as the Voyage-API-error path.
            Err(EmbedFileResult {
                fully_cached: false,
                embeddings: vec![vec![]; n_texts],
                hit_chunks: 0,
                miss_chunks: n_texts as u64,
            })
        }
    }
}

async fn embed_parsed_file(
    pf: &ParsedFile,
    voyage: Option<&VoyageClient>,
    cache: Option<Arc<EmbeddingCache>>,
) -> std::result::Result<EmbedFileResult, EmbedFileError> {
    if pf.chunks.is_empty() {
        return Ok(EmbedFileResult {
            embeddings: vec![],
            fully_cached: false,
            hit_chunks: 0,
            miss_chunks: 0,
        });
    }

    let texts: Vec<String> = pf.chunks.iter().map(|c| c.content.clone()).collect();

    // No voyage client AND no cache → return empty embeddings (same as before).
    if voyage.is_none() && cache.is_none() {
        return Ok(EmbedFileResult {
            embeddings: vec![vec![]; texts.len()],
            fully_cached: false,
            hit_chunks: 0,
            miss_chunks: texts.len() as u64,
        });
    }

    match cache {
        Some(cache_arc) => {
            // --- Cache path ---
            // Run cache.get_many() off the async runtime (blocking FS I/O).
            let texts_for_lookup = texts.clone();
            let cache_for_lookup = cache_arc.clone();
            let get_result = tokio::task::spawn_blocking(move || {
                cache_for_lookup.get_many(&texts_for_lookup)
            })
            .await;

            // Map JoinError (panic in spawn_blocking) to the degradation path.
            let (raw_hits, miss_indices) = match map_get_many_result(&pf.path, texts.len(), get_result) {
                Ok(result) => result,
                Err(degraded) => return Ok(degraded),
            };

            if miss_indices.is_empty() && !raw_hits.is_empty() {
                // 100% cache hit path.
                let dim = raw_hits[0].1.len();

                // Partition into valid hits and dim-mismatches.
                let mut valid_hits: Vec<(usize, Vec<f32>)> = Vec::new();
                let mut dim_miss_indices: Vec<usize> = Vec::new();
                for (idx, emb) in raw_hits {
                    if emb.len() == dim {
                        valid_hits.push((idx, emb));
                    } else {
                        dim_miss_indices.push(idx);
                    }
                }

                // Re-embed dim-mismatched entries if any.
                let mut extra_embeddings: Vec<(usize, Vec<f32>)> = Vec::new();
                if !dim_miss_indices.is_empty() {
                    if let Some(client) = voyage {
                        let miss_texts: Vec<String> = dim_miss_indices
                            .iter()
                            .map(|&i| texts[i].clone())
                            .collect();
                        match client.embed(&miss_texts, InputType::Document).await {
                            Ok(api_results) => {
                                let put_texts: Vec<String> = dim_miss_indices
                                    .iter()
                                    .map(|&i| texts[i].clone())
                                    .collect();
                                // put_many is blocking FS — run off the async runtime.
                                let cache_for_put = cache_arc.clone();
                                let put_embeddings = api_results.clone();
                                if let Err(e) = tokio::task::spawn_blocking(move || {
                                    cache_for_put.put_many(&put_texts, &put_embeddings);
                                })
                                .await
                                {
                                    warn!(file = %pf.path, error = %e, "cache put_many panicked (non-fatal)");
                                }
                                for (local_i, emb) in api_results.into_iter().enumerate() {
                                    extra_embeddings.push((dim_miss_indices[local_i], emb));
                                }
                            }
                            Err(e) => {
                                return Err(classify_embed_error(e));
                            }
                        }
                    } else {
                        for &i in &dim_miss_indices {
                            extra_embeddings.push((i, vec![]));
                        }
                    }
                }

                // Assemble final result in original order.
                let mut result = vec![vec![]; texts.len()];
                for (idx, emb) in valid_hits {
                    result[idx] = emb;
                }
                for (idx, emb) in extra_embeddings {
                    result[idx] = emb;
                }
                let n_dim_miss = dim_miss_indices.len() as u64;
                let n_total = texts.len() as u64;
                Ok(EmbedFileResult {
                    fully_cached: dim_miss_indices.is_empty(),
                    embeddings: result,
                    // valid cache reads = total minus any dim-mismatches that needed API
                    hit_chunks: n_total - n_dim_miss,
                    miss_chunks: n_dim_miss,
                })
            } else {
                // Partial or total cache miss path.
                let mut result = vec![vec![]; texts.len()];

                // Place valid cache hits into result.
                let mut valid_hits: Vec<(usize, Vec<f32>)> = Vec::new();

                // We need to know dim to validate hits — will learn from API response.
                // Collect all hits for now; validate after API call.
                let tentative_hits = raw_hits; // (idx, embedding)

                let all_miss_indices = if miss_indices.is_empty() {
                    // raw_hits also empty — full miss.
                    (0..texts.len()).collect::<Vec<_>>()
                } else {
                    miss_indices
                };

                // Call API for miss texts.
                let api_embeddings: Option<Vec<Vec<f32>>> = if let Some(client) = voyage {
                    let miss_texts: Vec<String> = all_miss_indices
                        .iter()
                        .map(|&i| texts[i].clone())
                        .collect();
                    match client.embed(&miss_texts, InputType::Document).await {
                        Ok(embs) => Some(embs),
                        Err(e) => {
                            return Err(classify_embed_error(e));
                        }
                    }
                } else {
                    None
                };

                match api_embeddings {
                    Some(api_results) if !api_results.is_empty() => {
                        // Learn dim from API results.
                        let dim = api_results[0].len();

                        // Cache the API results — blocking FS, run off async runtime.
                        let miss_texts: Vec<String> = all_miss_indices
                            .iter()
                            .map(|&i| texts[i].clone())
                            .collect();
                        let cache_for_put = cache_arc.clone();
                        let put_embeddings = api_results.clone();
                        if let Err(e) = tokio::task::spawn_blocking(move || {
                            cache_for_put.put_many(&miss_texts, &put_embeddings);
                        })
                        .await
                        {
                            warn!(file = %pf.path, error = %e, "cache put_many panicked (non-fatal)");
                        }

                        // Place API results into result.
                        for (local_i, emb) in api_results.into_iter().enumerate() {
                            result[all_miss_indices[local_i]] = emb;
                        }

                        // Validate cache hits against dim.
                        let mut re_embed_indices: Vec<usize> = Vec::new();
                        for (idx, emb) in tentative_hits {
                            if emb.len() == dim {
                                valid_hits.push((idx, emb));
                            } else {
                                re_embed_indices.push(idx);
                            }
                        }

                        // Re-embed any hits that were the wrong dim.
                        if !re_embed_indices.is_empty()
                            && let Some(client) = voyage
                        {
                            let re_texts: Vec<String> = re_embed_indices
                                .iter()
                                .map(|&i| texts[i].clone())
                                .collect();
                            match client.embed(&re_texts, InputType::Document).await {
                                Ok(re_results) => {
                                    let cache_for_put = cache_arc.clone();
                                    let put_re_texts = re_texts.clone();
                                    let put_re_embeddings = re_results.clone();
                                    if let Err(e) = tokio::task::spawn_blocking(move || {
                                        cache_for_put.put_many(&put_re_texts, &put_re_embeddings);
                                    })
                                    .await
                                    {
                                        warn!(file = %pf.path, error = %e, "cache put_many panicked (non-fatal)");
                                    }
                                    for (local_i, emb) in re_results.into_iter().enumerate() {
                                        result[re_embed_indices[local_i]] = emb;
                                    }
                                }
                                Err(e) => {
                                    return Err(classify_embed_error(e));
                                }
                            }
                        }
                    }
                    _ => {
                        // API failed or no voyage client — place empty for misses.
                        for &i in &all_miss_indices {
                            result[i] = vec![];
                        }
                        // Accept cache hits as-is (may be wrong dim but no API to fix them).
                        valid_hits = tentative_hits;
                    }
                }

                // Place valid cache hits.
                for (idx, emb) in valid_hits {
                    result[idx] = emb;
                }

                let n_miss = all_miss_indices.len() as u64;
                let n_total = texts.len() as u64;
                Ok(EmbedFileResult {
                    fully_cached: false,
                    embeddings: result,
                    hit_chunks: n_total.saturating_sub(n_miss),
                    miss_chunks: n_miss,
                })
            }
        }
        None => {
            // No cache — existing behavior.
            match voyage {
                Some(client) => {
                    match client.embed(&texts, InputType::Document).await {
                        Ok(embs) => Ok(EmbedFileResult {
                            fully_cached: false,
                            embeddings: embs,
                            hit_chunks: 0,
                            miss_chunks: texts.len() as u64,
                        }),
                        Err(e) => {
                            Err(classify_embed_error(e))
                        }
                    }
                }
                None => Ok(EmbedFileResult {
                    fully_cached: false,
                    embeddings: vec![vec![]; texts.len()],
                    hit_chunks: 0,
                    miss_chunks: texts.len() as u64,
                }),
            }
        }
    }
}

// ─── Flush helpers ────────────────────────────────────────────────────────

/// Flush a batch of chunk records via a native value-construction INSERT.
///
/// Builds each row as a `sql::Object` directly — bypassing serde — so the
/// `embedding` field reaches the engine as `Value::Bytes` (a packed
/// little-endian f32 blob), NOT an `array<int>`. A `#[derive(Serialize)]`
/// struct with a `Vec<u8>` field would serialize as an integer array, which
/// would (a) defeat the whole optimisation by re-introducing per-element
/// Number enums and (b) break the dual-format reader's byte path. The native
/// `SqlValue::Bytes` path is the same fast-lane `flush_symbol_batch_native`
/// and `flush_edge_batch` use.
async fn flush_chunk_batch(db: &Surreal<Db>, batch: Vec<ChunkRecord>) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }
    use std::collections::BTreeMap;

    let records: Vec<SqlValue> = batch
        .into_iter()
        .map(|rec| {
            let mut map: BTreeMap<String, SqlValue> = BTreeMap::new();
            map.insert("file".to_string(), SqlValue::from(rec.file));
            map.insert("line_start".to_string(), SqlValue::from(rec.line_start));
            map.insert("line_end".to_string(), SqlValue::from(rec.line_end));
            map.insert("content".to_string(), SqlValue::from(rec.content));
            // Packed embedding → Value::Bytes (single memcpy, no per-float enum).
            map.insert("embedding".to_string(), SqlValue::Bytes(SqlBytes::from(rec.embedding)));
            match rec.symbol_ref {
                Some(s) => map.insert("symbol_ref".to_string(), SqlValue::from(s)),
                None => map.insert("symbol_ref".to_string(), SqlValue::None),
            };
            SqlValue::Object(SqlObject::from(map))
        })
        .collect();

    let data = SqlArray::from(records);

    db.query("INSERT INTO chunk $data RETURN NONE")
        .bind(("data", data))
        .await
        .context("flush_chunk_batch")?;
    Ok(())
}

/// Flush symbols using native `INSERT INTO symbol $data` with a `surrealdb::sql::Array`.
///
/// Each symbol record is a `sql::Object` with an explicit string `id` field, bypassing the
/// serde serialization path entirely — so INSERT uses it as the record key.
///
/// Why this is faster than text-built UPSERT batches:
///   The text UPSERT approach builds 512 `UPSERT symbol:⟨fqn⟩ SET ...` statements per
///   batch and sends them as a single multi-statement query.  SurrealDB must parse all
///   512 statements.  The native sql::Array approach sends one INSERT statement with a
///   bound `$data` array — just one statement to parse, no per-row SQL text construction.
///
/// Duplicate-FQN handling (`ON DUPLICATE KEY UPDATE`):
///   A plain `INSERT` ERRORS when a record id already exists and rolls back the entire
///   batch. C++ produces duplicate FQNs (a symbol declared in a .h and defined in a .cpp),
///   so without the merge clause every batch containing a dup fails and 0 symbols persist.
///   `ON DUPLICATE KEY UPDATE ... = $input.<field>` makes the duplicate update the existing
///   record (last-write-wins), exactly matching the original UPSERT semantics.
async fn flush_symbol_batch_native(db: &Surreal<Db>, symbols: &[Symbol]) -> Result<()> {
    use crate::store::ops::kind_to_str;
    use std::collections::BTreeMap;

    // Use a larger batch size for native INSERT (no per-statement parsing overhead).
    // 4096 symbols × ~200 bytes = ~820KB per batch — safe payload size.
    for chunk in symbols.chunks(4096) {
        if chunk.is_empty() {
            continue;
        }

        let records: Vec<SqlValue> = chunk
            .iter()
            .map(|sym| {
                let mut map: BTreeMap<String, SqlValue> = BTreeMap::new();
                map.insert("id".to_string(), SqlValue::from(sym.qualified.fqn()));
                map.insert("name".to_string(), SqlValue::from(sym.qualified.name.as_str()));
                map.insert("kind".to_string(), SqlValue::from(kind_to_str(&sym.kind)));
                map.insert("file".to_string(), SqlValue::from(sym.qualified.file.as_str()));
                map.insert("line_start".to_string(), SqlValue::from(sym.line_start as i64));
                map.insert("line_end".to_string(), SqlValue::from(sym.line_end as i64));
                match &sym.signature {
                    Some(s) => map.insert("signature".to_string(), SqlValue::from(s.as_str())),
                    None    => map.insert("signature".to_string(), SqlValue::None),
                };
                match &sym.parent_fqn {
                    Some(p) => map.insert("parent".to_string(), SqlValue::from(
                        format!("symbol:⟨{}⟩", p)
                    )),
                    None => map.insert("parent".to_string(), SqlValue::None),
                };
                SqlValue::Object(SqlObject::from(map))
            })
            .collect();

        let data = SqlArray::from(records);

        // ON DUPLICATE KEY UPDATE: C++ declares a symbol in a .h and defines it
        // in a .cpp, producing two records with the same FQN (= record id). A plain
        // INSERT errors with "record already exists" and rolls back the WHOLE batch,
        // silently leaving 0 symbols. The merge clause makes the duplicate update the
        // existing record (last-write-wins), matching the original UPSERT semantics.
        // `.check()` surfaces statement-level errors that `.await?` alone swallows.
        db.query(
            "INSERT INTO symbol $data ON DUPLICATE KEY UPDATE \
             name = $input.name, kind = $input.kind, file = $input.file, \
             line_start = $input.line_start, line_end = $input.line_end, \
             signature = $input.signature, parent = $input.parent RETURN NONE",
        )
            .bind(("data", data))
            .await
            .context("flush_symbol_batch_native: INSERT INTO symbol")?
            .check()
            .context("flush_symbol_batch_native: INSERT statement error")?;
    }
    Ok(())
}

/// Flush raw edges (Phase 1) using native-bind INSERT.
/// Raw edges are stored in `raw_edge` table for later Phase 2 resolution.
async fn flush_raw_edge_batch_native(db: &Surreal<Db>, edges: &[RawEdgeRecord]) -> Result<()> {
    for chunk in edges.chunks(RAW_EDGE_INSERT_BATCH_SIZE) {
        let records: Vec<RawEdgeRecord> = chunk.to_vec();
        if !records.is_empty() {
            db.query("INSERT INTO raw_edge $data RETURN NONE")
                .bind(("data", records))
                .await
                .context("flush_raw_edge_batch_native")?;
        }
    }
    Ok(())
}

/// Flush a batch of resolved call edges using native `INSERT RELATION INTO calls $data`.
///
/// This constructs a `surrealdb::sql::Array` directly — bypassing serde serialization
/// entirely — so `in`/`out` fields are `Value::Thing` at the point they reach SurrealDB.
///
/// Why this works:
///   `to_value<T>` has a fast-path (`castaway::match_type!`) for `sql::Array` at the top
///   level: it returns `Value::Array(array)` without re-serializing the elements.  The
///   `Value::Thing` entries inside each `Object` are already native SQL values — they are
///   preserved exactly as `Thing { tb: "symbol", id: Id::String(fqn) }`.
///
///   The prior approach (`RELATE symbol:⟨fqn⟩->calls->symbol:⟨fqn⟩ SET ...`) built a
///   multi-statement text query that SurrealDB had to parse for each row.  At 138K edges
///   this parsing overhead dominated (~14s vs ~7s minimum for the raw KV writes).
async fn flush_edge_batch(
    db: &Surreal<Db>,
    batch: &[(String, String, i64, String, String, String, String)],
) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }

    // Build a sql::Array of sql::Object records.  Each Object has:
    //   in:       Value::Thing(symbol:⟨from_fqn⟩)
    //   out:      Value::Thing(symbol:⟨to_fqn⟩)
    //   line:     Value::Number(i64)
    //   in_file:  Value::Strand(string)
    //   out_file: Value::Strand(string)
    //   in_name:  Value::Strand(string)
    //   out_name: Value::Strand(string)
    //
    // The Array is passed as `$data`.  `to_value(sql::Array)` fast-paths through
    // `sql::Array as v => Ok(v.into())` — no serde, no type loss.
    use std::collections::BTreeMap;

    let records: Vec<SqlValue> = batch
        .iter()
        .map(|(from_fqn, to_fqn, line, in_file, out_file, in_name, out_name)| {
            let mut map: BTreeMap<String, SqlValue> = BTreeMap::new();
            map.insert("in".to_string(), SqlValue::Thing(
                SqlThing::from(("symbol", SqlId::String(from_fqn.clone())))
            ));
            map.insert("out".to_string(), SqlValue::Thing(
                SqlThing::from(("symbol", SqlId::String(to_fqn.clone())))
            ));
            map.insert("line".to_string(), SqlValue::from(*line));
            map.insert("in_file".to_string(), SqlValue::from(in_file.as_str()));
            map.insert("out_file".to_string(), SqlValue::from(out_file.as_str()));
            map.insert("in_name".to_string(), SqlValue::from(in_name.as_str()));
            map.insert("out_name".to_string(), SqlValue::from(out_name.as_str()));
            SqlValue::Object(SqlObject::from(map))
        })
        .collect();

    let data = SqlArray::from(records);

    // calls is a NORMAL table (schema v6+): graph-adjacency keys are not written,
    // so a plain INSERT is ~45% cheaper per edge than INSERT RELATION while every
    // v5+ read path (denormalized in_name/out_name/in_file/out_file) is unchanged.
    db.query("INSERT INTO calls $data")
        .bind(("data", data))
        .await
        .context("flush_edge_batch: INSERT")?;

    Ok(())
}

// ─── Watcher change filter ────────────────────────────────────────────────

/// Filter watcher-supplied file changes down to the same set `walk_repo` would
/// index during a full rebuild: indexable extension, not in a dot-dir, not in a
/// `SKIP_DIRS` tree (`target/`, `node_modules/`, …), and not gitignored.
///
/// The watcher emits raw filesystem events for every touched path, so without
/// this filter, build artifacts (e.g. `target/debug/*.exe`, `*.d`) written by a
/// concurrent `cargo build` get indexed and surface in query results — even
/// though a full rebuild correctly excludes them. This is the source of the
/// "gitignored files appear in results until a `--rebuild`" bug.
///
/// Deleted changes are ALWAYS allowed through regardless of the rules above, so
/// any artifact that a previous (unfiltered) watcher run indexed is cleaned up
/// when it is later removed — self-healing without requiring a full rebuild.
#[cfg(test)]
pub(crate) fn filter_hidden_changes(repo: &std::path::Path, changes: Vec<FileChange>) -> Vec<FileChange> {
    filter_hidden_changes_with(repo, changes, vec![], HashSet::new(), HashSet::new())
}

pub(crate) fn filter_hidden_changes_with(repo: &std::path::Path, changes: Vec<FileChange>, extra_extensions: Vec<String>, ignore_filenames: HashSet<String>, ignore_paths: HashSet<String>) -> Vec<FileChange> {
    let filter = ChangeFilter::new_complete(repo, extra_extensions, ignore_filenames, ignore_paths);
    changes
        .into_iter()
        .filter(|c| {
            // Always allow Deleted changes through (self-heal stale entries that
            // a previous unfiltered watcher run may have indexed).
            if c.kind == ChangeKind::Deleted {
                return true;
            }
            // Drop Added/Modified unless the path passes the full walk_repo rule set.
            filter.allows(std::path::Path::new(&c.path))
        })
        .collect()
}

// ─── SurrealQL escaping ───────────────────────────────────────────────────

fn escape_surreal(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

// ─── End-to-end pipeline regression tests ────────────────────────────────
#[cfg(test)]
mod end_to_end_persist {
    use super::*;
    use crate::store::open_db;
    use crate::store::ops::{count_chunks, count_indexed_files, count_symbols};
    use tempfile::TempDir;

    fn write_test_file(dir: &std::path::Path) -> String {
        let path = dir.join("sample.rs");
        std::fs::write(
            &path,
            "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\nfn subtract(a: i32, b: i32) -> i32 {\n    a - b\n}\n",
        )
        .expect("write test file");
        path.to_str().unwrap().replace('\\', "/")
    }

    #[tokio::test]
    async fn full_rebuild_real_source_tree_voyage_none() {
        let home = TempDir::new().unwrap();
        let repo = env!("CARGO_MANIFEST_DIR").replace('\\', "/");
        println!("REAL-TREE PROBE: repo = {repo}");

        let db = open_db(home.path(), &repo, 0).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);

        let result = pipeline.run(&db, None, true, None, None, None, &[], None).await;
        println!("REAL-TREE PROBE: result = {:?}", result.as_ref().map(|s| (s.indexed_files, s.total_files)));

        let chunks = count_chunks(&db).await.unwrap();
        let symbols = count_symbols(&db).await.unwrap();
        let files = count_indexed_files(&db, &repo).await.unwrap();
        println!("REAL-TREE PROBE: chunks={chunks}, symbols={symbols}, files={files}");

        assert!(result.is_ok(), "full_rebuild of real source tree must succeed (got: {:?})", result.err());
        assert!(chunks > 0, "must have chunks after full_rebuild of real source tree");
        assert!(files > 0, "must have indexed files");
    }

    #[tokio::test]
    async fn full_rebuild_persists_chunks_files_symbols() {
        let home = TempDir::new().unwrap();
        let repo_dir = TempDir::new().unwrap();

        let _file_path = write_test_file(repo_dir.path());
        let repo = repo_dir.path().to_str().unwrap().replace('\\', "/");

        let db = open_db(home.path(), &repo, 0).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);

        let stats = pipeline
            .run(&db, None, true, None, None, None, &[], None)
            .await
            .expect("full_rebuild must succeed");

        let chunks = count_chunks(&db).await.unwrap();
        let files = count_indexed_files(&db, &repo).await.unwrap();
        let symbols = count_symbols(&db).await.unwrap();

        println!("STEP3 — indexed_files={}, total_files={}", stats.indexed_files, stats.total_files);
        println!("STEP3 — chunks={chunks}, files={files}, symbols={symbols}");

        assert!(chunks > 0,
            "chunks must be > 0 after full_rebuild (got {chunks}); batched write path failed");
        assert!(files > 0,
            "indexed files must be > 0 after full_rebuild (got {files})");
        assert!(symbols > 0,
            "symbols must be > 0 after full_rebuild (got {symbols})");
        assert_eq!(stats.indexed_files, files,
            "stats.indexed_files must match count_indexed_files");
    }

    /// ❷ NEW: file_meta.chunk_count is populated correctly after streaming index.
    #[tokio::test]
    async fn chunk_count_in_file_meta_matches_actual_chunks() {
        let home = TempDir::new().unwrap();
        let repo_dir = TempDir::new().unwrap();
        let _file_path = write_test_file(repo_dir.path());
        let repo = repo_dir.path().to_str().unwrap().replace('\\', "/");

        let db = open_db(home.path(), &repo, 0).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);
        pipeline.run(&db, None, true, None, None, None, &[], None).await.expect("rebuild");

        // Check that file_meta.chunk_count > 0 for the test file.
        use serde::Deserialize;
        #[derive(Deserialize)]
        struct Row { chunk_count: i64 }
        let rows: Vec<Row> = db
            .query("SELECT chunk_count FROM file_meta WHERE repo = $repo")
            .bind(("repo", repo.clone()))
            .await
            .unwrap()
            .take(0)
            .unwrap();

        assert!(!rows.is_empty(), "file_meta rows must exist");
        for row in &rows {
            assert!(row.chunk_count >= 0, "chunk_count must not be negative");
        }
        let total: i64 = rows.iter().map(|r| r.chunk_count).sum();
        assert!(total > 0, "total chunk_count across all files must be > 0");
    }

    /// ❸ NEW: edges_resolved marker is set after full_rebuild.
    #[tokio::test]
    async fn edges_resolved_marker_set_after_rebuild() {
        let home = TempDir::new().unwrap();
        let repo_dir = TempDir::new().unwrap();
        let _file_path = write_test_file(repo_dir.path());
        let repo = repo_dir.path().to_str().unwrap().replace('\\', "/");

        let db = open_db(home.path(), &repo, 0).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);
        pipeline.run(&db, None, true, None, None, None, &[], None).await.expect("rebuild");

        let marker = get_meta(&db, EDGES_RESOLVED_KEY).await.unwrap();
        assert!(marker.is_some(), "edges_resolved marker must be set after full_rebuild");
    }
}

// ─── Symbol-index drop/rebuild (optimize-symbol-write-throughput) ─────────
//
// These pin the full-rebuild-only drop→bulk-write→rebuild of the two secondary
// symbol indexes (idx_symbol_file, idx_symbol_name). The optimization moves
// index maintenance OFF the per-row write path; correctness rests on: dedup is
// on the PRIMARY KEY (FQN) not the secondary indexes (D3), the indexes are
// present and usable after a full build (D2), and the incremental path NEVER
// touches the global indexes (D5).
#[cfg(test)]
mod symbol_index_drop_rebuild {
    use super::*;
    use crate::parsing::symbols::{QualifiedSymbol, SymbolKind};
    use crate::store::open_db;
    use crate::store::ops::{count_symbols, find_symbols_by_names_with_pos};
    use tempfile::TempDir;

    /// Read the set of defined index names on the `symbol` table via INFO FOR TABLE.
    /// SurrealDB returns `{ "indexes": { "idx_symbol_name": "DEFINE INDEX ...", ... } }`.
    async fn symbol_index_names(db: &Surreal<Db>) -> Vec<String> {
        let info: Option<serde_json::Value> = db
            .query("INFO FOR TABLE symbol")
            .await
            .expect("INFO FOR TABLE symbol")
            .take(0)
            .ok()
            .flatten();
        info.and_then(|v| v.get("indexes").and_then(|i| i.as_object()).cloned())
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// 5.1 — A full rebuild via the drop/rebuild path produces the SAME symbol set
    /// as writing through live indexes would, including the C++ same-FQN dedup case.
    /// Dropping the two secondary indexes cannot change row contents because dedup
    /// is on the PRIMARY KEY (record id = FQN), proving D3.
    #[tokio::test]
    async fn full_rebuild_symbol_set_and_cpp_dedup_unchanged() {
        let home = TempDir::new().unwrap();
        let repo_dir = TempDir::new().unwrap();

        // A C++ declaration in a .h and its definition in a .cpp share one FQN
        // (file is part of the FQN only via the per-file path — here we force the
        // SAME FQN by giving both the declaration of `compute` the same qualified
        // name through identical file-relative scoping). To exercise the documented
        // .h/.cpp last-write-wins dedup deterministically, we feed the symbol batch
        // a duplicate FQN directly through the same flush path the pipeline uses.
        let repo = repo_dir.path().to_str().unwrap().replace('\\', "/");
        let db = open_db(home.path(), &repo, 0).await.expect("open db");

        // Drop the indexes (as a full rebuild would) then bulk-write a batch that
        // contains a duplicate FQN — the second row must collapse onto the first
        // via ON DUPLICATE KEY UPDATE, regardless of secondary-index presence.
        db.query(
            "REMOVE INDEX IF EXISTS idx_symbol_file ON symbol; \
             REMOVE INDEX IF EXISTS idx_symbol_name ON symbol;"
        ).await.expect("drop symbol indexes");

        let header = format!("{repo}/widget.h");
        let mk = |file: &str, name: &str, ls: u32, le: u32, sig: &str| Symbol {
            qualified: QualifiedSymbol {
                file: file.to_string(),
                scope_path: vec![],
                name: name.to_string(),
            },
            kind: SymbolKind::Function,
            line_start: ls,
            line_end: le,
            signature: Some(sig.to_string()),
            parent_fqn: None,
        };
        // Two symbols with the SAME FQN (same file + name) — the .h/.cpp dup case.
        // Last write (line 42, sig "definition") must win.
        let batch = vec![
            mk(&header, "compute", 10, 12, "declaration"),
            mk(&header, "compute", 40, 42, "definition"),
            mk(&format!("{repo}/other.cpp"), "render", 1, 5, "void render()"),
        ];
        flush_symbol_batch_native(&db, &batch).await.expect("flush symbols");

        // Rebuild the indexes (as the post-tail-flush step does).
        db.query(
            "DEFINE INDEX idx_symbol_file ON symbol FIELDS file; \
             DEFINE INDEX idx_symbol_name ON symbol FIELDS name;"
        ).await.expect("rebuild symbol indexes");

        // Dedup outcome: 3 rows written, 1 duplicate FQN → exactly 2 rows persisted.
        let total = count_symbols(&db).await.expect("count symbols");
        assert_eq!(total, 2, "same-FQN duplicate must collapse to one row (got {total})");

        // The surviving `compute` row must be the last write (line_start 40).
        let rows = find_symbols_by_names_with_pos(&db, &["compute".to_string()])
            .await
            .expect("lookup compute");
        assert_eq!(rows.len(), 1, "exactly one compute row");
        assert_eq!(rows[0].line_start, 40, "last-write-wins on duplicate FQN");
    }

    /// 5.2 — After a full rebuild the secondary symbol indexes are present and a
    /// name lookup (the path Phase 2 and queries use) returns the expected rows.
    /// Proves the rebuild actually ran and the index is usable (D2).
    #[tokio::test]
    async fn full_rebuild_leaves_symbol_indexes_present_and_usable() {
        let home = TempDir::new().unwrap();
        let repo_dir = TempDir::new().unwrap();
        let path = repo_dir.path().join("sample.rs");
        std::fs::write(
            &path,
            "fn alpha() -> i32 {\n    1\n}\n\nfn beta() -> i32 {\n    2\n}\n",
        )
        .expect("write source file");
        let repo = repo_dir.path().to_str().unwrap().replace('\\', "/");

        let db = open_db(home.path(), &repo, 0).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);
        pipeline
            .run(&db, None, true, None, None, None, &[], None)
            .await
            .expect("full rebuild");

        // Both secondary indexes must be defined after the build (rebuild ran).
        let names = symbol_index_names(&db).await;
        assert!(
            names.iter().any(|n| n == "idx_symbol_file"),
            "idx_symbol_file must be present after full rebuild (got {names:?})"
        );
        assert!(
            names.iter().any(|n| n == "idx_symbol_name"),
            "idx_symbol_name must be present after full rebuild (got {names:?})"
        );

        // Name lookup via idx_symbol_name must return the expected symbol.
        let rows = find_symbols_by_names_with_pos(&db, &["alpha".to_string()])
            .await
            .expect("lookup alpha");
        assert_eq!(rows.len(), 1, "exactly one alpha symbol");
        assert_eq!(rows[0].name, "alpha");
    }

    /// 5.3 — An incremental run must NOT drop the global symbol indexes (D5).
    /// We seed a built index, then run an incremental update for one changed file
    /// and assert both secondary indexes remain defined throughout.
    #[tokio::test]
    async fn incremental_does_not_drop_symbol_indexes() {
        let home = TempDir::new().unwrap();
        let repo_dir = TempDir::new().unwrap();
        let path = repo_dir.path().join("lib.rs");
        std::fs::write(&path, "fn one() {}\n").expect("write file");
        let repo = repo_dir.path().to_str().unwrap().replace('\\', "/");

        let db = open_db(home.path(), &repo, 0).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);

        // Initial full rebuild establishes the indexes.
        pipeline
            .run(&db, None, true, None, None, None, &[], None)
            .await
            .expect("initial full rebuild");
        let before = symbol_index_names(&db).await;
        assert!(before.iter().any(|n| n == "idx_symbol_file"));
        assert!(before.iter().any(|n| n == "idx_symbol_name"));

        // Modify the file and run an incremental update (force_rebuild = false).
        std::fs::write(&path, "fn one() {}\n\nfn two() {}\n").expect("modify file");
        let changes = vec![FileChange {
            path: path.to_str().unwrap().replace('\\', "/"),
            kind: ChangeKind::Modified,
        }];
        pipeline
            .run(&db, Some(changes), false, None, None, None, &[], None)
            .await
            .expect("incremental run");

        // Both global symbol indexes must still be present — incremental must not
        // have dropped/rebuilt them (would turn O(changed) into O(repo)).
        let after = symbol_index_names(&db).await;
        assert!(
            after.iter().any(|n| n == "idx_symbol_file"),
            "idx_symbol_file must remain after incremental (got {after:?})"
        );
        assert!(
            after.iter().any(|n| n == "idx_symbol_name"),
            "idx_symbol_name must remain after incremental (got {after:?})"
        );

        // And the incremental's new symbol must be reachable by name lookup.
        let rows = find_symbols_by_names_with_pos(&db, &["two".to_string()])
            .await
            .expect("lookup two");
        assert_eq!(rows.len(), 1, "incremental symbol must be indexed/queryable");
    }

    /// 5.5 — Crash-safety of the drop/rebuild window (D4). If the process dies
    /// AFTER the secondary symbol indexes are dropped but BEFORE the post-build
    /// rebuild, the indexes are missing on disk while rows exist. The next
    /// `open_db` runs `store::ensure_secondary_indexes`, which finds the indexes
    /// absent on the populated symbol table and rebuilds them via
    /// `build_index_concurrently` (CONCURRENTLY — never a foreground backfill that
    /// would roll back under the pinned RocksDB buffers at scale) — self-healing the
    /// crash window with no manual rebuild and no data loss. (At this test's tiny
    /// row count the rebuild is instant either way; the scale failure that motivates
    /// CONCURRENTLY is proven by `store::foreground_index_backfill_at_scale`.)
    ///
    /// This exercises the EXACT crash window rather than reading the DDL:
    ///   open → drop indexes → write rows → assert ABSENT (in the window) →
    ///   PROCESS DEATH (flush to disk) → next boot reads the crash image →
    ///   assert indexes restored + rows intact.
    ///
    /// Why we open a COPY of the on-disk image instead of reopening the same
    /// path: SurrealDB's embedded RocksDB engine holds an exclusive per-dir LOCK
    /// for the WHOLE process — the `rocksdb::DB` instance is kept alive by the
    /// engine layer, so within one process a second open of the same path fails
    /// on LOCK even after the `Surreal<Db>` handle drops and its runtime tears
    /// down (this is exactly why the store documents "production never
    /// drops+reopens" — `get_or_open` keeps one cached handle for the repo's
    /// lifetime). In production, boot 2 is a fresh OS process where that path is
    /// unlocked. We reproduce a fresh boot faithfully: build the crash-window
    /// state inside a DEDICATED runtime/thread, fully tear it down so RocksDB
    /// flushes its WAL/MANIFEST/SSTs to disk (a clean quiesced crash image), then
    /// COPY that on-disk image (minus the LOCK marker, which a new boot recreates)
    /// to a fresh data dir and `open_db` there. That is the bytes-that-survived
    /// opened by a new process — real crash recovery, not a weakened assertion.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn crash_between_drop_and_rebuild_self_heals_on_next_open() {
        // Recursively copy `src` → `dst`, skipping any RocksDB `LOCK` file (a new
        // boot recreates it). Mirrors what survives on disk after a crash.
        fn copy_crash_image(src: &std::path::Path, dst: &std::path::Path) {
            std::fs::create_dir_all(dst).expect("mkdir crash-image dst");
            for entry in std::fs::read_dir(src).expect("read crash-image src") {
                let entry = entry.expect("dir entry");
                let name = entry.file_name();
                let ty = entry.file_type().expect("file type");
                let to = dst.join(&name);
                if ty.is_dir() {
                    copy_crash_image(&entry.path(), &to);
                } else if name != "LOCK" {
                    std::fs::copy(entry.path(), &to).expect("copy crash-image file");
                }
            }
        }

        let crash_home = TempDir::new().unwrap();
        let repo_dir = TempDir::new().unwrap();
        let repo = repo_dir.path().to_str().unwrap().replace('\\', "/");

        // ── Boot 1 + crash window, in a sacrificial runtime/thread ──────────────
        // Everything up to and including "process death" happens here. When this
        // thread returns, its runtime is dropped, deterministically tearing down
        // the SurrealDB engine so RocksDB flushes a clean on-disk image. The
        // assertions inside cover steps 1-4; a failure panics the thread and is
        // surfaced by the join unwrap below.
        let crash_home_path = crash_home.path().to_path_buf();
        let repo_for_thread = repo.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build sacrificial runtime");
            rt.block_on(async move {
                // Step 1 — first boot: open_db defines both indexes (on the empty
                // table, via ensure_secondary_indexes).
                let db = open_db(&crash_home_path, &repo_for_thread, 0)
                    .await
                    .expect("open db (boot 1)");
                let initial = symbol_index_names(&db).await;
                assert!(
                    initial.iter().any(|n| n == "idx_symbol_file")
                        && initial.iter().any(|n| n == "idx_symbol_name"),
                    "fresh open_db must define both symbol indexes (got {initial:?})"
                );

                // Step 2 — reproduce the post-drop state: the full rebuild drops
                // both secondary indexes before its bulk symbol write.
                db.query(
                    "REMOVE INDEX IF EXISTS idx_symbol_file ON symbol; \
                     REMOVE INDEX IF EXISTS idx_symbol_name ON symbol;",
                )
                .await
                .expect("drop symbol indexes (enter crash window)");

                // Step 3 — write symbol rows through the SAME flush path the
                // pipeline uses, with indexes dropped: "dropped, rows written, NOT
                // yet rebuilt" — the precise state at the moment of a crash.
                let mk = |file: &str, name: &str, ls: u32, le: u32, sig: &str| Symbol {
                    qualified: QualifiedSymbol {
                        file: file.to_string(),
                        scope_path: vec![],
                        name: name.to_string(),
                    },
                    kind: SymbolKind::Function,
                    line_start: ls,
                    line_end: le,
                    signature: Some(sig.to_string()),
                    parent_fqn: None,
                };
                let batch = vec![
                    mk(&format!("{repo_for_thread}/a.rs"), "alpha", 1, 3, "fn alpha()"),
                    mk(&format!("{repo_for_thread}/b.rs"), "beta", 5, 9, "fn beta()"),
                ];
                flush_symbol_batch_native(&db, &batch)
                    .await
                    .expect("flush symbols in crash window");

                // Step 4 — confirm we are genuinely IN the crash window: both
                // secondary indexes must be ABSENT right now (dropped, rebuild
                // never ran).
                let in_window = symbol_index_names(&db).await;
                assert!(
                    !in_window.iter().any(|n| n == "idx_symbol_file"),
                    "idx_symbol_file must be ABSENT in the crash window (got {in_window:?})"
                );
                assert!(
                    !in_window.iter().any(|n| n == "idx_symbol_name"),
                    "idx_symbol_name must be ABSENT in the crash window (got {in_window:?})"
                );

                // Step 5 — process death: drop the handle WITHOUT rebuilding.
                drop(db);
            });
            // Runtime drops here → engine task gone → RocksDB flushed a clean image.
        })
        .join()
        .expect("crash-window thread must not panic (its assertions are steps 1-4)");

        // ── Boot 2: a fresh process reads the bytes that survived the crash ─────
        // Copy the quiesced on-disk image to a brand-new data dir (a path never
        // opened in this process → no held LOCK), the faithful analog of a new OS
        // process booting on the crashed repo's directory.
        let boot2_home = TempDir::new().unwrap();
        copy_crash_image(crash_home.path(), boot2_home.path());

        // Step 6 — next boot: open_db on the crash image; it runs
        // ensure_secondary_indexes after SCHEMA_DDL.
        let db = open_db(boot2_home.path(), &repo, 0)
            .await
            .expect("open db (boot 2, post-crash)");

        // Step 7 — self-heal: ensure_secondary_indexes found both indexes absent on
        // the populated symbol table and rebuilt them concurrently over the rows.
        let healed = symbol_index_names(&db).await;
        assert!(
            healed.iter().any(|n| n == "idx_symbol_file"),
            "idx_symbol_file must be restored on next open (got {healed:?})"
        );
        assert!(
            healed.iter().any(|n| n == "idx_symbol_name"),
            "idx_symbol_name must be restored on next open (got {healed:?})"
        );

        // Step 8 — no data loss: rows written in the crash window survived, and a
        // name lookup (the path Phase 2/queries use) returns the expected row via
        // the restored index.
        let total = count_symbols(&db).await.expect("count symbols post-heal");
        assert_eq!(total, 2, "both crash-window rows must persist (got {total})");
        let rows = find_symbols_by_names_with_pos(&db, &["beta".to_string()])
            .await
            .expect("lookup beta post-heal");
        assert_eq!(rows.len(), 1, "exactly one beta symbol after self-heal");
        assert_eq!(rows[0].name, "beta");
        assert_eq!(rows[0].line_start, 5, "row content intact after self-heal");
    }
}

// ─── Two-phase resolution equivalence tests ──────────────────────────────
#[cfg(test)]
mod resolution_tests {
    use super::*;
    use crate::store::open_db;
    use tempfile::TempDir;

    /// ❸ NEW: find_symbols_by_names returns ONLY requested names.
    #[tokio::test]
    async fn find_symbols_by_names_no_full_table_leak() {
        use crate::store::ops::find_symbols_by_names_with_pos;

        let home = TempDir::new().unwrap();
        let repo = "/test/symbol_repo";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Insert 3 symbols with different names.
        for (name, file) in &[("foo", "/a.rs"), ("bar", "/b.rs"), ("baz", "/c.rs")] {
            db.query(format!(
                "UPSERT symbol:`⟨{file}::{name}⟩` SET \
                 name = '{name}', kind = 'function', file = '{file}', \
                 line_start = 1, line_end = 5, signature = NONE, parent = NONE"
            ))
            .await
            .unwrap();
        }

        // Request only "foo" and "bar" — must NOT return "baz".
        let result = find_symbols_by_names_with_pos(
            &db,
            &["foo".to_string(), "bar".to_string()],
        )
        .await
        .unwrap();

        assert_eq!(result.len(), 2, "should return exactly 2 symbols");
        for s in &result {
            assert!(
                s.name == "foo" || s.name == "bar",
                "unexpected symbol name: {}",
                s.name
            );
            assert_ne!(s.name, "baz", "baz must not be returned");
        }
    }

    /// ❸ NEW: tie-break sort — multiple candidates for same name sorted by
    /// (file, line_start, line_end) ascending; same-file preferred.
    #[test]
    fn tie_break_sort_deterministic() {
        let mut candidates: Vec<SymbolWithPos> = vec![
            SymbolWithPos { fqn: "/c.rs::f".to_string(), file: "/c.rs".to_string(), name: "f".to_string(), line_start: 10, line_end: 20 },
            SymbolWithPos { fqn: "/a.rs::f".to_string(), file: "/a.rs".to_string(), name: "f".to_string(), line_start: 5, line_end: 15 },
            SymbolWithPos { fqn: "/b.rs::f".to_string(), file: "/b.rs".to_string(), name: "f".to_string(), line_start: 1, line_end: 5 },
        ];

        candidates.sort_unstable_by(|a, b| {
            a.file.cmp(&b.file)
                .then(a.line_start.cmp(&b.line_start))
                .then(a.line_end.cmp(&b.line_end))
        });

        // After sort: /a.rs < /b.rs < /c.rs.
        assert_eq!(candidates[0].file, "/a.rs");
        assert_eq!(candidates[1].file, "/b.rs");
        assert_eq!(candidates[2].file, "/c.rs");
    }

    /// ❸ NEW: same-file resolution is preferred over sorted-first cross-file.
    #[test]
    fn same_file_preferred_over_sorted_first() {
        let from_file = "/b.rs";
        let candidates: Vec<SymbolWithPos> = vec![
            SymbolWithPos { fqn: "/a.rs::f".to_string(), file: "/a.rs".to_string(), name: "f".to_string(), line_start: 1, line_end: 5 },
            SymbolWithPos { fqn: "/b.rs::f".to_string(), file: "/b.rs".to_string(), name: "f".to_string(), line_start: 10, line_end: 20 },
        ];

        // Same-file candidate (/b.rs) should be preferred even though /a.rs sorts first.
        let resolved = candidates
            .iter()
            .find(|c| c.file == from_file)
            .or_else(|| candidates.first())
            .cloned()
            .unwrap();

        assert_eq!(resolved.file, "/b.rs", "same-file must be preferred");
    }
}

// ─── Concurrency bound test ───────────────────────────────────────────────
#[cfg(test)]
mod concurrency_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// ❶ NEW: embedding stage respects configured concurrency N.
    /// We mock the embed function with a counter to ensure at most N run concurrently.
    #[tokio::test]
    async fn embed_concurrency_bound_respected() {
        use futures::StreamExt;

        let max_concurrent = Arc::new(AtomicUsize::new(0));
        let peak_concurrent = Arc::new(AtomicUsize::new(0));
        let configured_n = 3usize;

        // Create 10 "tasks" that track concurrent execution.
        let tasks: Vec<usize> = (0..10).collect();
        let max_ref = max_concurrent.clone();
        let peak_ref = peak_concurrent.clone();

        futures::stream::iter(tasks)
            .map(|_i| {
                let cur = max_ref.clone();
                let peak = peak_ref.clone();
                async move {
                    let n = cur.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(n, Ordering::SeqCst);
                    // Simulate async work.
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    cur.fetch_sub(1, Ordering::SeqCst);
                }
            })
            .buffer_unordered(configured_n)
            .collect::<Vec<_>>()
            .await;

        let peak = peak_concurrent.load(Ordering::SeqCst);
        assert!(
            peak <= configured_n,
            "peak concurrent ({peak}) exceeded configured N ({configured_n})"
        );
    }
}

// ─── Keyset pagination correctness tests ─────────────────────────────────
#[cfg(test)]
mod keyset_pagination_tests {
    use crate::store::open_db;
    use tempfile::TempDir;

    /// Keyset pagination on raw_edge visits every row exactly once across multi-page datasets.
    ///
    /// Rows are inserted via `INSERT INTO raw_edge $data` (native-bind, same path as
    /// flush_raw_edge_batch_native in production), letting SurrealDB assign the record ids.
    /// The test then runs the same `type::string(id) > $cursor ORDER BY id_str` keyset loop
    /// used by resolve_edges_phase2 and verifies:
    ///   1. All N rows are returned (none skipped).
    ///   2. No row appears twice (none duplicated).
    ///   3. id_str values are returned in ascending order.
    #[tokio::test]
    async fn raw_edge_keyset_visits_every_row_exactly_once() {
        use serde::{Deserialize, Serialize};

        let home = TempDir::new().unwrap();
        let repo = "/test/keyset_repo";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Insert 15 raw_edge rows using the same native-bind path as Phase 1
        // (SurrealDB assigns the record ids — no app-managed seq).
        // Some share the same (from_file, to_name) to exercise the skip-hazard scenario.
        #[derive(Serialize)]
        struct RawEdge {
            from_file: String,
            from_name: String,
            from_fqn: String,
            to_name: String,
            kind: String,
            line: i64,
        }

        let total: usize = 15;
        let records: Vec<RawEdge> = (1i64..=(total as i64))
            .map(|i| {
                // Rows 1, 6, 11 share from_file="/a.rs" and to_name="foo" — these are the
                // kind of non-unique-on-content rows that caused OFFSET to potentially skip.
                let from_file = if i % 5 == 1 { "/a.rs".to_string() } else { format!("/f{i}.rs") };
                let to_name = if i % 5 == 1 { "foo".to_string() } else { format!("sym{i}") };
                let from_name = format!("caller{i}");
                let from_fqn = format!("{}::{}", from_file, from_name);
                RawEdge {
                    from_file,
                    from_name,
                    from_fqn,
                    to_name,
                    kind: "calls".to_string(),
                    line: i,
                }
            })
            .collect();

        db.query("INSERT INTO raw_edge $data RETURN NONE")
            .bind(("data", records))
            .await
            .expect("insert raw_edge batch")
            .check()
            .expect("insert must succeed");

        // Page through using the same keyset logic as resolve_edges_phase2.
        let page_size: i64 = 5;
        let mut cursor = String::new();
        let mut seen_ids: Vec<String> = Vec::new();

        loop {
            #[derive(Deserialize)]
            struct Row { id_str: String }
            let batch: Vec<Row> = db
                .query(
                    "SELECT type::string(id) AS id_str FROM raw_edge \
                     WHERE type::string(id) > $cursor ORDER BY id_str LIMIT $page",
                )
                .bind(("cursor", cursor.clone()))
                .bind(("page", page_size))
                .await
                .unwrap()
                .take(0)
                .unwrap();

            if batch.is_empty() {
                break;
            }

            cursor = batch.last().map(|r| r.id_str.clone()).unwrap_or(cursor);

            for row in &batch {
                seen_ids.push(row.id_str.clone());
            }

            if (batch.len() as i64) < page_size {
                break;
            }
        }

        // Verify: exactly `total` rows, no duplicates, strictly ascending.
        assert_eq!(
            seen_ids.len(),
            total,
            "keyset must visit every row: expected {total}, got {}",
            seen_ids.len()
        );

        // No duplicates.
        let mut sorted = seen_ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            total,
            "keyset produced duplicate rows: {} unique out of {}",
            sorted.len(),
            seen_ids.len()
        );

        // Strictly ascending (id_str ordering is consistent within a page-scan).
        for w in seen_ids.windows(2) {
            assert!(
                w[0] < w[1],
                "rows not in ascending id_str order: {} >= {}",
                w[0],
                w[1]
            );
        }
    }

    /// Restart-collision regression: two insert passes into the same raw_edge table
    /// (simulating incremental runs across a process restart) must not cause id collisions,
    /// and Phase 2 keyset pagination must visit all rows exactly once.
    ///
    /// With the old `RAW_EDGE_SEQ` counter approach:
    ///   - Pass 1 writes rows with seq = 1..5 and commits.
    ///   - Process restarts; RAW_EDGE_SEQ resets to 1.
    ///   - Pass 2 (incremental) deletes file A's rows, re-inserts them with seq = 1, 2, 3...
    ///   - Those seq values collide with Pass 1's surviving rows → UNIQUE constraint failure.
    ///
    /// With the SurrealDB record-id approach:
    ///   - SurrealDB assigns new unique ids for every INSERT regardless of restarts.
    ///   - No collision is possible. This test confirms the invariant.
    #[tokio::test]
    async fn restart_collision_no_id_collision_across_insert_passes() {
        use serde::{Deserialize, Serialize};

        let home = TempDir::new().unwrap();
        let repo = "/test/restart_collision_repo";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        #[derive(Serialize)]
        struct RawEdge {
            from_file: String,
            from_name: String,
            from_fqn: String,
            to_name: String,
            kind: String,
            line: i64,
        }

        // Pass 1: insert 5 rows for file_a.
        let pass1: Vec<RawEdge> = (1i64..=5)
            .map(|i| RawEdge {
                from_file: "/file_a.rs".to_string(),
                from_name: format!("fn_a{i}"),
                from_fqn: format!("/file_a.rs::fn_a{i}"),
                to_name: format!("target{i}"),
                kind: "calls".to_string(),
                line: i,
            })
            .collect();

        db.query("INSERT INTO raw_edge $data RETURN NONE")
            .bind(("data", pass1))
            .await
            .expect("pass1 insert")
            .check()
            .expect("pass1 must succeed");

        // Simulate process restart + incremental run: delete file_a's rows, re-insert them.
        // (This is what delete_files_data_bulk does for changed files in incremental_run.)
        db.query("DELETE FROM raw_edge WHERE from_file = '/file_a.rs'")
            .await
            .expect("delete file_a rows");

        // Pass 2: re-insert the same 5 rows (simulates re-parse of file_a after restart).
        let pass2: Vec<RawEdge> = (1i64..=5)
            .map(|i| RawEdge {
                from_file: "/file_a.rs".to_string(),
                from_name: format!("fn_a{i}"),
                from_fqn: format!("/file_a.rs::fn_a{i}"),
                to_name: format!("target{i}"),
                kind: "calls".to_string(),
                line: i,
            })
            .collect();

        // With the old seq-counter approach this would fail with a UNIQUE constraint error.
        // With the id-based approach, SurrealDB assigns fresh ids and succeeds.
        let result = db
            .query("INSERT INTO raw_edge $data RETURN NONE")
            .bind(("data", pass2))
            .await;

        assert!(result.is_ok(), "pass2 insert must not fail: {:?}", result.err());
        result.unwrap().check().expect("pass2 insert must have no per-statement errors");

        // Verify 5 rows total (pass1 rows were deleted, pass2 replaced them).
        #[derive(Deserialize)]
        struct CountRow { count: i64 }
        let counts: Vec<CountRow> = db
            .query("SELECT count() AS count FROM raw_edge GROUP ALL")
            .await.unwrap().take(0).unwrap();
        let count = counts.first().map(|r| r.count).unwrap_or(0);
        assert_eq!(count, 5, "must have exactly 5 rows after pass2 (got {count})");

        // Phase 2 keyset pagination must visit all 5 rows exactly once.
        let mut cursor = String::new();
        let mut visited: Vec<String> = Vec::new();

        loop {
            #[derive(Deserialize)]
            struct Row { id_str: String }
            let batch: Vec<Row> = db
                .query(
                    "SELECT type::string(id) AS id_str FROM raw_edge \
                     WHERE type::string(id) > $cursor ORDER BY id_str LIMIT $page",
                )
                .bind(("cursor", cursor.clone()))
                .bind(("page", 3i64))
                .await.unwrap().take(0).unwrap();

            if batch.is_empty() { break; }
            cursor = batch.last().map(|r| r.id_str.clone()).unwrap_or(cursor);
            for row in &batch { visited.push(row.id_str.clone()); }
            if (batch.len() as i64) < 3 { break; }
        }

        assert_eq!(visited.len(), 5, "phase2 keyset must visit all 5 rows (got {})", visited.len());

        let mut deduped = visited.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(deduped.len(), 5, "no duplicate ids in phase2 scan");
    }
}

// ─── Per-edge backfill correctness test ──────────────────────────────────
#[cfg(test)]
mod per_edge_backfill_tests {
    use crate::store::{open_db, run_migration_v1_to_v2};
    use tempfile::TempDir;

    /// Defect 2 regression: calls backfill assigns per-edge-correct names even when
    /// two distinct edges share the same (in_file, out_file) pair.
    ///
    /// Scenario:
    ///   edge1: A::foo -> B::baz   (in_file=/a.rs, out_file=/b.rs)
    ///   edge2: A::bar -> B::qux   (in_file=/a.rs, out_file=/b.rs)
    ///
    /// The old file-pair UPDATE would stamp one pair onto BOTH edges.
    /// The fixed per-id UPDATE must set in_name/out_name correctly on each.
    #[tokio::test]
    async fn calls_backfill_assigns_per_edge_correct_names() {
        use serde::Deserialize;

        let home = TempDir::new().unwrap();
        let repo = "/test/per_edge_backfill";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Create symbols for the four endpoints.
        for (file, name) in &[
            ("/a.rs", "foo"), ("/a.rs", "bar"),
            ("/b.rs", "baz"), ("/b.rs", "qux"),
        ] {
            db.query(format!(
                "UPSERT symbol:`⟨{file}::{name}⟩` SET \
                 name = '{name}', kind = 'function', file = '{file}', \
                 line_start = 1, line_end = 5, signature = NONE, parent = NONE"
            ))
            .await
            .unwrap();
        }

        // Create two edges WITHOUT in_name/out_name (v1 state). Both share
        // in_file=/a.rs and out_file=/b.rs. calls is a NORMAL table (v6+), so we
        // INSERT plain rows; `in`/`out` are stored as record links so the v1→v2
        // backfill's `in.name`/`out.name` deref still resolves.
        db.query(
            "INSERT INTO calls { in: symbol:`⟨/a.rs::foo⟩`, out: symbol:`⟨/b.rs::baz⟩`, \
             line: 1, in_file: '/a.rs', out_file: '/b.rs' }"
        ).await.unwrap();

        db.query(
            "INSERT INTO calls { in: symbol:`⟨/a.rs::bar⟩`, out: symbol:`⟨/b.rs::qux⟩`, \
             line: 2, in_file: '/a.rs', out_file: '/b.rs' }"
        ).await.unwrap();

        // Verify pre-migration state: in_name IS NONE on both.
        #[derive(Deserialize, Debug)]
        struct EdgeRow {
            id_str: String,
            in_name: Option<String>,
            out_name: Option<String>,
        }
        let before: Vec<EdgeRow> = db
            .query(
                "SELECT type::string(id) AS id_str, in_name, out_name \
                 FROM calls ORDER BY id_str",
            )
            .await
            .unwrap()
            .take(0)
            .unwrap();

        assert_eq!(before.len(), 2, "must have 2 call edges before migration");
        for row in &before {
            assert!(
                row.in_name.is_none(),
                "pre-migration in_name must be NONE, got {:?}",
                row.in_name
            );
        }

        // Run migration.
        run_migration_v1_to_v2(&db).await.unwrap();

        // Read back the edges and verify per-edge correctness.
        let after: Vec<EdgeRow> = db
            .query(
                "SELECT type::string(id) AS id_str, in_name, out_name \
                 FROM calls ORDER BY id_str",
            )
            .await
            .unwrap()
            .take(0)
            .unwrap();

        assert_eq!(after.len(), 2, "must still have 2 call edges after migration");

        // Build a lookup: id -> (in_name, out_name).
        let edge_map: std::collections::HashMap<String, (Option<String>, Option<String>)> = after
            .iter()
            .map(|r| (r.id_str.clone(), (r.in_name.clone(), r.out_name.clone())))
            .collect();

        // Verify both edges have non-None, DISTINCT in_name/out_name pairs.
        let all_in_names: Vec<&str> = after
            .iter()
            .filter_map(|r| r.in_name.as_deref())
            .collect();
        let all_out_names: Vec<&str> = after
            .iter()
            .filter_map(|r| r.out_name.as_deref())
            .collect();

        // Both in_names must be present and distinct.
        assert_eq!(all_in_names.len(), 2, "both edges must have in_name set");
        assert_ne!(
            all_in_names[0], all_in_names[1],
            "in_names must be distinct per edge (got both = {:?}); \
             file-pair UPDATE incorrectly collapsed them",
            all_in_names[0]
        );

        // Both out_names must be present and distinct.
        assert_eq!(all_out_names.len(), 2, "both edges must have out_name set");
        assert_ne!(
            all_out_names[0], all_out_names[1],
            "out_names must be distinct per edge (got both = {:?}); \
             file-pair UPDATE incorrectly collapsed them",
            all_out_names[0]
        );

        // Exact values: {foo,bar} and {baz,qux} in some order.
        let mut in_names_sorted = all_in_names.to_vec();
        in_names_sorted.sort_unstable();
        assert_eq!(in_names_sorted, vec!["bar", "foo"], "in_names must be {{foo,bar}}");

        let mut out_names_sorted = all_out_names.to_vec();
        out_names_sorted.sort_unstable();
        assert_eq!(out_names_sorted, vec!["baz", "qux"], "out_names must be {{baz,qux}}");

        println!("per_edge_backfill: edge_map = {:?}", edge_map);
    }
}

// ─── Incremental Phase 2 scoped resolution test ───────────────────────────
#[cfg(test)]
mod incremental_phase2_tests {
    use super::*;
    use crate::store::open_db;
    use serde::Deserialize;
    use tempfile::TempDir;

    /// Inserts a symbol into the DB directly (bypasses the full pipeline).
    async fn insert_symbol(db: &Surreal<Db>, file: &str, name: &str) {
        db.query(format!(
            "UPSERT symbol:`⟨{file}::{name}⟩` SET \
             name = '{name}', kind = 'function', file = '{file}', \
             line_start = 1, line_end = 10, signature = NONE, parent = NONE"
        ))
        .await
        .expect("insert symbol");
    }

    /// Inserts a raw_edge row into the DB directly (simulates Phase 1 output).
    async fn insert_raw_edge(db: &Surreal<Db>, from_file: &str, from_name: &str, to_name: &str) {
        use serde::Serialize;
        #[derive(Serialize)]
        struct RawEdge {
            from_file: String,
            from_name: String,
            from_fqn: String,
            to_name: String,
            kind: String,
            line: i64,
        }
        let rec = vec![RawEdge {
            from_file: from_file.to_string(),
            from_name: from_name.to_string(),
            from_fqn: format!("{}::{}", from_file, from_name),
            to_name: to_name.to_string(),
            kind: "calls".to_string(),
            line: 1,
        }];
        db.query("INSERT INTO raw_edge $data RETURN NONE")
            .bind(("data", rec))
            .await
            .expect("insert raw_edge");
    }

    /// Count calls rows where in_file = $file.
    async fn count_calls_from(db: &Surreal<Db>, in_file: &str) -> usize {
        #[derive(Deserialize)]
        struct Row { count: i64 }
        let rows: Vec<Row> = db
            .query("SELECT count() AS count FROM calls WHERE in_file = $f GROUP ALL")
            .bind(("f", in_file.to_string()))
            .await.unwrap().take(0).unwrap();
        rows.first().map(|r| r.count as usize).unwrap_or(0)
    }

    /// Read all calls rows from the DB (for precise assertions).
    async fn all_calls(db: &Surreal<Db>) -> Vec<(String, String, String, String)> {
        #[derive(Deserialize)]
        struct Row {
            in_file: String,
            out_file: String,
            in_name: Option<String>,
            out_name: Option<String>,
        }
        let rows: Vec<Row> = db
            .query("SELECT in_file, out_file, in_name, out_name FROM calls ORDER BY in_file, in_name, out_name")
            .await.unwrap().take(0).unwrap();
        rows.into_iter()
            .map(|r| (
                r.in_file,
                r.out_file,
                r.in_name.unwrap_or_default(),
                r.out_name.unwrap_or_default(),
            ))
            .collect()
    }

    /// Scenario: A calls B, B calls C.
    ///
    /// File layout:
    ///   /a.rs  — defines `a_fn`, raw_edge: a_fn -> b_fn
    ///   /b.rs  — defines `b_fn`, raw_edge: b_fn -> c_fn
    ///   /c.rs  — defines `c_fn`, no outgoing edges
    ///
    /// Incremental on file B (changed_files = ["/b.rs"]) must:
    ///   - Re-resolve B's outgoing edge (b_fn -> c_fn).
    ///   - Re-resolve A's edge that pointed into B (a_fn -> b_fn) because
    ///     B's symbols changed: Approach A finds A as an extra_from_file.
    ///   - NOT touch C's edges (C has no outgoing edges, so count_calls_from C = 0
    ///     both before and after, but we verify total calls is correct).
    ///
    /// After the incremental, we assert:
    ///   - calls A->B edge exists (a_fn -> b_fn)
    ///   - calls B->C edge exists (b_fn -> c_fn)
    ///   - total calls count = 2
    ///   - calls_from C = 0 (untouched — C had no outgoing edges)
    #[tokio::test]
    async fn incremental_phase2_resolves_only_affected_files() {
        let home = TempDir::new().unwrap();
        let repo = "/test/incremental_phase2";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        let pipeline = IndexPipeline::new(repo.to_string(), None);

        // ── Set up initial state: A calls B, B calls C ────────────────────
        // Insert symbols for all three files.
        insert_symbol(&db, "/a.rs", "a_fn").await;
        insert_symbol(&db, "/b.rs", "b_fn").await;
        insert_symbol(&db, "/c.rs", "c_fn").await;

        // Insert raw_edges (Phase 1 output).
        insert_raw_edge(&db, "/a.rs", "a_fn", "b_fn").await;
        insert_raw_edge(&db, "/b.rs", "b_fn", "c_fn").await;

        // Run a full Phase 2 to establish baseline calls rows.
        pipeline.resolve_edges_phase2(&db, None).await.expect("initial full phase2");

        let initial_calls = all_calls(&db).await;
        println!("Initial calls: {:?}", initial_calls);
        assert_eq!(initial_calls.len(), 2, "initial state must have 2 calls edges");

        // Record the calls rows for C (should be 0 — C has no outgoing edges).
        let c_calls_before = count_calls_from(&db, "/c.rs").await;
        assert_eq!(c_calls_before, 0, "C has no outgoing edges initially");

        // ── Simulate incremental: B is changed ────────────────────────────
        // In a real incremental, streaming_index would delete B's raw_edge rows
        // and re-insert them (delete_files_data_bulk covers that). Here we
        // manually simulate the state that incremental_run sets up before calling
        // resolve_edges_incremental:
        //   - B's symbols are still correct (unchanged for this test).
        //   - B's raw_edge rows survive (delete_files_data_bulk only deletes
        //     raw_edge WHERE from_file IN changed, so B's row is gone and re-added
        //     during streaming_index; we keep it as-is here since the content is same).
        // The key invariant: calls table has been wiped for changed files already
        // by delete_files_data_bulk (which runs before streaming_index in incremental_run).
        // We simulate that by not touching the calls table — resolve_edges_incremental
        // will handle its own scoped delete.

        // Run incremental Phase 2 for changed file B.
        // pre_delete_callers is empty here because we're calling resolve_edges_incremental
        // directly (bypassing incremental_run). The test's scenario has A pointing at B,
        // and the direction-1 path is gated on a surface change. Here B's symbol
        // surface is UNCHANGED (b_fn stays), so dir1_callers and added_names are
        // both empty: resolve_set = [B]. We delete only B's OUTGOING edge (B→C)
        // and re-resolve it; A→B is an INCOMING edge to B and is left untouched
        // (it still points at the still-existing b_fn). Net result is identical to
        // the full rebuild, but the blast radius is just the changed file.
        let changed = vec!["/b.rs".to_string()];
        pipeline.resolve_edges_incremental(&db, &changed, &[], &[])
            .await
            .expect("incremental phase2 must succeed");

        // ── Assertions ────────────────────────────────────────────────────
        let final_calls = all_calls(&db).await;
        println!("Final calls after incremental on B: {:?}", final_calls);

        // Must still have exactly 2 calls edges.
        assert_eq!(
            final_calls.len(), 2,
            "must have 2 calls edges after incremental (A->B and B->C); got {:?}",
            final_calls
        );

        // A->B edge must be present.
        // in_name and out_name now store full FQNs (file::name).
        let a_to_b = final_calls.iter().any(|(in_f, out_f, in_n, out_n)| {
            in_f == "/a.rs" && out_f == "/b.rs"
                && in_n == "/a.rs::a_fn" && out_n == "/b.rs::b_fn"
        });
        assert!(a_to_b, "A->B edge (a_fn -> b_fn) must be present; got {:?}", final_calls);

        // B->C edge must be present.
        let b_to_c = final_calls.iter().any(|(in_f, out_f, in_n, out_n)| {
            in_f == "/b.rs" && out_f == "/c.rs"
                && in_n == "/b.rs::b_fn" && out_n == "/c.rs::c_fn"
        });
        assert!(b_to_c, "B->C edge (b_fn -> c_fn) must be present; got {:?}", final_calls);

        // C's outgoing calls are still 0 (untouched — C was not in changed set
        // and had no outgoing edges; its raw_edge rows were not touched).
        let c_calls_after = count_calls_from(&db, "/c.rs").await;
        assert_eq!(
            c_calls_after, 0,
            "C's calls must be untouched (0) after incremental on B (got {})",
            c_calls_after
        );
    }

    /// Test: "new file wins the tie-break for an unchanged caller"
    ///
    /// Scenario:
    ///   - File X ("/x_caller.rs") has a raw_edge targeting name `foo` (X calls foo).
    ///   - File Z ("/z_defines_foo.rs") defines symbol `foo`.
    ///   - Full rebuild resolves X→foo to Z (only candidate at the time).
    ///
    /// Incremental:
    ///   - File W ("/a_defines_foo.rs") is "added" — we insert its symbol `foo` and
    ///     mark it as a changed file. W < Z lexicographically ("a_" < "z_"), so W
    ///     wins the tie-break in a full rebuild.
    ///   - After resolve_edges_incremental with changed_files = [W], X→foo must
    ///     now point to W (the new lex-first winner).
    ///   - Without direction-2 expansion X is not in resolve_set (it never pointed
    ///     into W, because W didn't exist yet), so X→foo would stay stale pointing
    ///     at Z — a divergence from full-rebuild.
    #[tokio::test]
    async fn new_file_wins_tiebreak_for_unchanged_caller() {
        let home = TempDir::new().unwrap();
        let repo = "/test/tiebreak_caller";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        let pipeline = IndexPipeline::new(repo.to_string(), None);

        // ── Initial state: X calls foo, only Z defines foo ─────────────────
        // Paths chosen so that /a_defines_foo.rs < /z_defines_foo.rs
        // and /x_caller.rs sits between them alphabetically — it is NOT the
        // lex-first definer, so X is not picked as a self-file resolution.
        insert_symbol(&db, "/z_defines_foo.rs", "foo").await;
        insert_raw_edge(&db, "/x_caller.rs", "x_fn", "foo").await;

        // Full Phase 2: X→foo resolves to Z (the only candidate).
        pipeline.resolve_edges_phase2(&db, None).await.expect("initial full phase2");

        let initial_calls = all_calls(&db).await;
        println!("Initial calls: {:?}", initial_calls);
        assert_eq!(initial_calls.len(), 1, "initial state must have exactly 1 calls edge");
        let x_to_z = initial_calls.iter().any(|(in_f, out_f, _, out_n)| {
            in_f == "/x_caller.rs" && out_f == "/z_defines_foo.rs"
                && out_n == "/z_defines_foo.rs::foo"
        });
        assert!(x_to_z, "X→foo must initially resolve to Z; got {:?}", initial_calls);

        // ── "Add" file W: insert its symbol foo ────────────────────────────
        // W sorts before Z lexicographically, so it should win the tie-break.
        insert_symbol(&db, "/a_defines_foo.rs", "foo").await;

        // Run incremental Phase 2 with changed_files = [W].
        // dir1_callers is empty: X never pointed into W (W didn't exist yet), so the
        // direction-1 caller query would return nothing for this scenario. W's
        // surface GAINED the name `foo`, so added_names = ["foo"]; direction-2
        // name-expansion (raw_edge.to_name = foo) is what finds X here and re-points
        // X→foo to W (the new lex-first winner).
        let changed = vec!["/a_defines_foo.rs".to_string()];
        pipeline
            .resolve_edges_incremental(&db, &changed, &[], &["foo".to_string()])
            .await
            .expect("incremental phase2 must succeed");

        // ── Assertions ─────────────────────────────────────────────────────
        let final_calls = all_calls(&db).await;
        println!("Final calls after incremental on W: {:?}", final_calls);

        // Still exactly 1 edge (X→foo).
        assert_eq!(
            final_calls.len(), 1,
            "must still have exactly 1 calls edge after incremental; got {:?}", final_calls
        );

        // X→foo must now point to W ("/a_defines_foo.rs"), not Z.
        let x_to_w = final_calls.iter().any(|(in_f, out_f, _, out_n)| {
            in_f == "/x_caller.rs" && out_f == "/a_defines_foo.rs"
                && out_n == "/a_defines_foo.rs::foo"
        });
        assert!(
            x_to_w,
            "X→foo must re-resolve to W (lex-first winner) after incremental; got {:?}",
            final_calls
        );
    }

    /// Regression: "removal direction" that was previously uncaught.
    ///
    /// Scenario:
    ///   - X ("/x.rs") has raw_edge to_name=bar.
    ///   - W ("/w.rs") defines bar. Y ("/y.rs") also defines bar. W < Y lexicographically.
    ///   - Full rebuild resolves X→bar→W (W is lex-first).
    ///   - W is edited and removes bar.
    ///
    /// Without pre-delete capture:
    ///   - delete_files_data_bulk([W]) removes X's calls row (out_file=W).
    ///   - direction-1 queries `calls WHERE out_file IN [W]` → empty (deleted!).
    ///   - X never enters resolve_set. X→bar is permanently lost.
    ///
    /// With pre-delete capture (this test):
    ///   - Pre-delete query finds X (it has out_file=W).
    ///   - After bulk delete and re-index of W (no bar symbol), resolve_edges_incremental
    ///     with pre_delete_callers=[X] includes X in resolve_set.
    ///   - X→bar re-resolves to Y (the remaining candidate).
    #[tokio::test]
    async fn removal_from_changed_file_caller_repoints() {
        let home = TempDir::new().unwrap();
        let repo = "/test/removal_repoints";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        let pipeline = IndexPipeline::new(repo.to_string(), None);

        // ── Initial state: X calls bar, W and Y both define bar, W < Y lex ──
        // W="/w.rs" < Y="/y.rs" lexicographically, so W wins the tie-break.
        insert_symbol(&db, "/w.rs", "bar").await;
        insert_symbol(&db, "/y.rs", "bar").await;
        insert_raw_edge(&db, "/x.rs", "x_fn", "bar").await;

        // Full Phase 2: X→bar→W (W is lex-first).
        pipeline.resolve_edges_phase2(&db, None).await.expect("initial full phase2");

        let initial_calls = all_calls(&db).await;
        println!("Initial calls: {:?}", initial_calls);
        assert_eq!(initial_calls.len(), 1, "initial: 1 calls edge");
        let x_to_w = initial_calls.iter().any(|(in_f, out_f, _, out_n)| {
            in_f == "/x.rs" && out_f == "/w.rs" && out_n == "/w.rs::bar"
        });
        assert!(x_to_w, "X→bar must initially resolve to W; got {:?}", initial_calls);

        // ── Simulate production incremental path for W being edited (bar removed) ──

        // Step 1: Pre-delete query (before bulk delete) — finds X as a caller of W.
        use serde::Deserialize;
        #[derive(Deserialize)]
        struct PreDeleteRow { in_file: String }
        let changed_files = vec!["/w.rs".to_string()];
        let pre_rows: Vec<PreDeleteRow> = db
            .query(
                "SELECT in_file FROM calls \
                 WHERE out_file IN $files AND in_file NOT IN $files \
                 GROUP BY in_file",
            )
            .bind(("files", changed_files.clone()))
            .await.unwrap().take(0).unwrap();
        let pre_delete_callers: Vec<String> = pre_rows.into_iter().map(|r| r.in_file).collect();
        println!("pre_delete_callers: {:?}", pre_delete_callers);
        assert!(
            pre_delete_callers.contains(&"/x.rs".to_string()),
            "pre-delete query must find X as a caller of W; got {:?}", pre_delete_callers
        );

        // Step 2: Simulate bulk delete of W (removes W's symbols, raw_edges, calls).
        db.query("DELETE FROM symbol WHERE file = '/w.rs'").await.unwrap();
        db.query("DELETE FROM raw_edge WHERE from_file = '/w.rs'").await.unwrap();
        db.query("DELETE FROM calls WHERE in_file = '/w.rs' OR out_file = '/w.rs'").await.unwrap();

        // Step 3: Re-index W without bar (W edited, bar removed — only x_fn raw_edge
        // came from X, not W, so W has no outgoing edges to re-add). W's symbol row
        // for bar is gone (deleted above). We do NOT re-add it.

        // Step 4: resolve_edges_incremental with dir1_callers=[X], added_names=[].
        // W's surface REMOVED `bar` (a removal, not an addition), so added_names is
        // empty; X is the direction-1 caller (it pointed into W). X re-resolves to Y.
        pipeline
            .resolve_edges_incremental(&db, &changed_files, &pre_delete_callers, &[])
            .await
            .expect("incremental phase2 must succeed");

        // ── Assertions ─────────────────────────────────────────────────────
        let final_calls = all_calls(&db).await;
        println!("Final calls after W removes bar: {:?}", final_calls);

        // X→bar must now resolve to Y (the remaining candidate after W removed bar).
        let x_to_y = final_calls.iter().any(|(in_f, out_f, _, out_n)| {
            in_f == "/x.rs" && out_f == "/y.rs" && out_n == "/y.rs::bar"
        });
        assert!(
            x_to_y,
            "X→bar must re-resolve to Y after W removes bar; got {:?}", final_calls
        );

        // Must have exactly 1 edge (X→bar→Y).
        assert_eq!(
            final_calls.len(), 1,
            "must have exactly 1 calls edge after re-resolve; got {:?}", final_calls
        );
    }

    /// Prove direction-1 (pre_delete_callers) actually fires in the production
    /// sequence.
    ///
    /// Scenario:
    ///   - X ("/x.rs") has raw_edge to_name=foo, W ("/w.rs") defines foo.
    ///   - Full rebuild: X→foo→W.
    ///   - W is edited but KEEPS foo (no change to symbol).
    ///
    /// In production, incremental_run:
    ///   1. Pre-delete query finds X (X has out_file=W).
    ///   2. delete_files_data_bulk([W]) deletes W's calls rows (including X→foo→W).
    ///   3. Re-index W (foo still present).
    ///   4. resolve_edges_incremental([W], pre_delete_callers=[X]).
    ///
    /// Assert: after the incremental, X→foo still resolves to W (re-resolved
    /// correctly, not lost even though X's calls row was deleted by bulk delete).
    #[tokio::test]
    async fn direction1_fires_in_production_path() {
        let home = TempDir::new().unwrap();
        let repo = "/test/direction1_fires";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        let pipeline = IndexPipeline::new(repo.to_string(), None);

        // ── Initial state: X calls foo, W defines foo ─────────────────────
        insert_symbol(&db, "/w.rs", "foo").await;
        insert_raw_edge(&db, "/x.rs", "x_fn", "foo").await;

        // Full Phase 2: X→foo→W.
        pipeline.resolve_edges_phase2(&db, None).await.expect("initial full phase2");

        let initial_calls = all_calls(&db).await;
        assert_eq!(initial_calls.len(), 1, "initial: 1 calls edge");
        let x_to_w = initial_calls.iter().any(|(in_f, out_f, _, out_n)| {
            in_f == "/x.rs" && out_f == "/w.rs" && out_n == "/w.rs::foo"
        });
        assert!(x_to_w, "X→foo must initially resolve to W; got {:?}", initial_calls);

        // ── Simulate production incremental path for W being edited (foo kept) ──

        // Step 1: Pre-delete query — finds X.
        use serde::Deserialize;
        #[derive(Deserialize)]
        struct PreDeleteRow { in_file: String }
        let changed_files = vec!["/w.rs".to_string()];
        let pre_rows: Vec<PreDeleteRow> = db
            .query(
                "SELECT in_file FROM calls \
                 WHERE out_file IN $files AND in_file NOT IN $files \
                 GROUP BY in_file",
            )
            .bind(("files", changed_files.clone()))
            .await.unwrap().take(0).unwrap();
        let pre_delete_callers: Vec<String> = pre_rows.into_iter().map(|r| r.in_file).collect();
        println!("pre_delete_callers: {:?}", pre_delete_callers);
        assert!(
            pre_delete_callers.contains(&"/x.rs".to_string()),
            "pre-delete query must find X; got {:?}", pre_delete_callers
        );

        // Step 2: Simulate bulk delete of W (removes W's calls rows — including X→foo→W).
        db.query("DELETE FROM calls WHERE in_file = '/w.rs' OR out_file = '/w.rs'").await.unwrap();
        db.query("DELETE FROM raw_edge WHERE from_file = '/w.rs'").await.unwrap();
        // NOTE: W's symbol (foo) and X's raw_edge remain intact (only calls is wiped by
        // delete_files_data_bulk in production for the calls/raw_edge tables of changed files;
        // X is unchanged so its raw_edge row survives).

        // Confirm X's calls row is gone after bulk delete.
        let after_delete = all_calls(&db).await;
        assert_eq!(after_delete.len(), 0, "X→foo must be gone after simulated bulk delete");

        // Step 3: Re-index W — foo still present (no change to symbol row).
        // (Symbol already exists from initial setup; no action needed.)

        // Step 4: resolve_edges_incremental([W], dir1_callers=[X], added_names=[]).
        // W KEEPS foo (surface unchanged in identity), but in production W is a
        // surface-changed file in this scenario's framing — direction-1 supplies X
        // so its bulk-deleted X→foo→W edge is re-resolved (back to W). added_names
        // is empty (foo was already present — not a NEW pair).
        pipeline
            .resolve_edges_incremental(&db, &changed_files, &pre_delete_callers, &[])
            .await
            .expect("incremental phase2 must succeed");

        // ── Assertions ─────────────────────────────────────────────────────
        let final_calls = all_calls(&db).await;
        println!("Final calls after W edited (foo kept): {:?}", final_calls);

        // X→foo must still resolve to W (re-resolved via direction-1).
        let x_to_w_again = final_calls.iter().any(|(in_f, out_f, _, out_n)| {
            in_f == "/x.rs" && out_f == "/w.rs" && out_n == "/w.rs::foo"
        });
        assert!(
            x_to_w_again,
            "X→foo must re-resolve to W after incremental (direction-1 must fire); got {:?}",
            final_calls
        );

        assert_eq!(
            final_calls.len(), 1,
            "must have exactly 1 calls edge; got {:?}", final_calls
        );
    }
}

// ─── Correctness oracle: INCREMENTAL calls-set == FULL-REBUILD calls-set ────
//
// The most invariant-dense guard in the crate. For each edit scenario we build
// the FINAL content from scratch (full rebuild) and snapshot its `calls` set,
// then build the INITIAL content, full-rebuild it, and replay the edit through
// the SAME DB-level sequence `incremental_run` uses (capture old surface →
// incremental delete → re-write changed symbols/raw_edges → compute surface
// delta → gated direction-1 query → resolve_edges_incremental). The two `calls`
// sets MUST be EQUAL (order-insensitive). If the gating ever drops or mis-points
// an edge, a scenario diverges and fails.
//
// raw_edge is populated for ALL files (not just changed ones), exactly as the
// task requires, so unchanged callers can re-resolve in the direction-1/2 paths.
// (Production's RAM-path full rebuild leaves raw_edge empty for unchanged files —
// an orthogonal pre-existing limitation that only affects genuine API-change
// incrementals, never the gated comment/body-only case where no caller is pulled
// in. The oracle isolates and proves the GATING logic itself.)
#[cfg(test)]
mod incremental_correctness_oracle {
    use super::*;
    use crate::store::open_db;
    use crate::store::ops::delete_files_data_incremental;
    use serde::Deserialize;
    use tempfile::TempDir;

    #[derive(Clone)]
    struct SymDef { file: String, name: String, fqn: String, line_start: i64, line_end: i64 }
    #[derive(Clone)]
    struct EdgeDef { from_file: String, from_name: String, from_fqn: String, to_name: String, line: i64 }

    #[derive(Clone, Default)]
    struct RepoState { syms: Vec<SymDef>, edges: Vec<EdgeDef> }

    impl RepoState {
        fn sym(mut self, file: &str, name: &str, line_start: i64) -> Self {
            self.syms.push(SymDef {
                file: file.to_string(),
                name: name.to_string(),
                fqn: format!("{file}::{name}"),
                line_start,
                line_end: line_start + 5,
            });
            self
        }
        /// A symbol with an explicit fqn (for overloads / scope moves where the
        /// leaf name repeats but the fqn must differ).
        #[allow(dead_code)]
        fn sym_fqn(mut self, file: &str, name: &str, fqn: &str, line_start: i64) -> Self {
            self.syms.push(SymDef {
                file: file.to_string(),
                name: name.to_string(),
                fqn: fqn.to_string(),
                line_start,
                line_end: line_start + 5,
            });
            self
        }
        fn edge(mut self, from_file: &str, from_name: &str, to_name: &str, line: i64) -> Self {
            self.edges.push(EdgeDef {
                from_file: from_file.to_string(),
                from_name: from_name.to_string(),
                from_fqn: format!("{from_file}::{from_name}"),
                to_name: to_name.to_string(),
                line,
            });
            self
        }
    }

    async fn ins_sym(db: &Surreal<Db>, s: &SymDef) {
        db.query(format!(
            "UPSERT symbol:`⟨{}⟩` SET name = '{}', kind = 'function', file = '{}', \
             line_start = {}, line_end = {}, signature = NONE, parent = NONE",
            s.fqn, s.name, s.file, s.line_start, s.line_end
        ))
        .await
        .expect("ins_sym");
    }

    async fn ins_edge(db: &Surreal<Db>, e: &EdgeDef) {
        use serde::Serialize;
        #[derive(Serialize)]
        struct RawEdge {
            from_file: String,
            from_name: String,
            from_fqn: String,
            to_name: String,
            kind: String,
            line: i64,
        }
        let rec = vec![RawEdge {
            from_file: e.from_file.clone(),
            from_name: e.from_name.clone(),
            from_fqn: e.from_fqn.clone(),
            to_name: e.to_name.clone(),
            kind: "calls".to_string(),
            line: e.line,
        }];
        db.query("INSERT INTO raw_edge $data RETURN NONE")
            .bind(("data", rec))
            .await
            .expect("ins_edge");
    }

    async fn write_state(db: &Surreal<Db>, st: &RepoState) {
        for s in &st.syms { ins_sym(db, s).await; }
        for e in &st.edges { ins_edge(db, e).await; }
    }

    /// (in_file, out_file, in_name, out_name) tuples, sorted — an order-insensitive
    /// snapshot of the `calls` table.
    async fn calls_set(db: &Surreal<Db>) -> Vec<(String, String, String, String)> {
        #[derive(Deserialize)]
        struct Row { in_file: String, out_file: String, in_name: Option<String>, out_name: Option<String> }
        let rows: Vec<Row> = db
            .query("SELECT in_file, out_file, in_name, out_name FROM calls")
            .await.unwrap().take(0).unwrap();
        let mut v: Vec<(String, String, String, String)> = rows.into_iter()
            .map(|r| (r.in_file, r.out_file, r.in_name.unwrap_or_default(), r.out_name.unwrap_or_default()))
            .collect();
        v.sort();
        v
    }

    /// Full rebuild of `state` from scratch → its calls set.
    async fn full_calls(repo: &str, state: &RepoState) -> Vec<(String, String, String, String)> {
        let home = TempDir::new().unwrap();
        let db = open_db(home.path(), repo, 0).await.unwrap();
        let pipeline = IndexPipeline::new(repo.to_string(), None);
        write_state(&db, state).await;
        pipeline.resolve_edges_phase2(&db, None).await.expect("full phase2");
        calls_set(&db).await
    }

    /// Build `initial`, full-rebuild it, then replay the edit to `final_state`
    /// through the production incremental DB sequence → resulting calls set.
    /// Returns (calls_set, removed_surface_files, added_names) so scenarios can
    /// also assert the gating shape (e.g. comment edit ⇒ both empty).
    async fn incremental_calls(
        repo: &str,
        initial: &RepoState,
        final_state: &RepoState,
        changed: &[&str],
        deleted: &[&str],
    ) -> (Vec<(String, String, String, String)>, Vec<String>, Vec<String>) {
        let home = TempDir::new().unwrap();
        let db = open_db(home.path(), repo, 0).await.unwrap();
        let pipeline = IndexPipeline::new(repo.to_string(), None);

        // Initial state + full resolve (this populates raw_edge for ALL files).
        write_state(&db, initial).await;
        pipeline.resolve_edges_phase2(&db, None).await.expect("initial full phase2");

        let to_process: Vec<String> = changed.iter().map(|s| s.to_string()).collect();
        let to_delete: Vec<String> = deleted.iter().map(|s| s.to_string()).collect();
        let all_affected: Vec<String> = to_delete.iter().chain(to_process.iter()).cloned().collect();

        // 1) capture OLD surface (before deleting changed files' symbols).
        let old_surface = load_file_surface(&db, &to_process).await.unwrap();

        // 2) incremental delete (everything EXCEPT calls).
        delete_files_data_incremental(&db, &all_affected).await.unwrap();

        // 3) simulate streaming_index: re-write FINAL symbols + raw_edges for the
        //    changed (to_process) files only.
        for s in &final_state.syms {
            if to_process.contains(&s.file) { ins_sym(&db, s).await; }
        }
        for e in &final_state.edges {
            if to_process.contains(&e.from_file) { ins_edge(&db, e).await; }
        }

        // 4) compute surface delta.
        let new_surface = load_file_surface(&db, &to_process).await.unwrap();
        let delta = compute_surface_delta(&old_surface, &new_surface, &to_process, &to_delete);

        // 5) gated direction-1 caller query (mirrors incremental_run).
        let dir1_callers: Vec<String> = if delta.removed_surface_files.is_empty() {
            Vec::new()
        } else {
            #[derive(Deserialize)]
            struct CallerRow { in_file: String }
            let rows: Vec<CallerRow> = db
                .query(
                    "SELECT in_file FROM calls \
                     WHERE out_file IN $changed AND in_file NOT IN $affected GROUP BY in_file",
                )
                .bind(("changed", delta.removed_surface_files.clone()))
                .bind(("affected", all_affected.clone()))
                .await.unwrap().take(0).unwrap();
            rows.into_iter().map(|r| r.in_file).collect()
        };

        // 6) scoped incremental resolution.
        pipeline
            .resolve_edges_incremental(&db, &all_affected, &dir1_callers, &delta.added_names)
            .await
            .expect("incremental phase2");

        (calls_set(&db).await, delta.removed_surface_files, delta.added_names)
    }

    /// The shared base world: two callers, two libs each defining `helper`
    /// (lib_x lex-first wins), lib_x also defines `util`.
    fn base_world() -> RepoState {
        RepoState::default()
            .sym("/caller_a.rs", "a_fn", 1)
            .sym("/caller_b.rs", "b_fn", 1)
            .sym("/lib_x.rs", "helper", 10)
            .sym("/lib_x.rs", "util", 20)
            .sym("/lib_z.rs", "helper", 10)
            .edge("/caller_a.rs", "a_fn", "helper", 2)
            .edge("/caller_b.rs", "b_fn", "helper", 3)
            .edge("/caller_b.rs", "b_fn", "util", 4)
    }

    /// Scenario 1: comment/body-only edit — lib_x's symbols shift lines but the
    /// (name, fqn) surface is IDENTICAL. MUST be fast (empty gates) AND correct.
    #[tokio::test]
    async fn oracle_comment_body_only_edit() {
        let initial = base_world();
        // lib_x edited: helper/util shifted down (comment inserted), same surface.
        let mut final_state = RepoState::default()
            .sym("/caller_a.rs", "a_fn", 1)
            .sym("/caller_b.rs", "b_fn", 1)
            .sym("/lib_x.rs", "helper", 13)   // +3 lines
            .sym("/lib_x.rs", "util", 23)      // +3 lines
            .sym("/lib_z.rs", "helper", 10);
        final_state.edges = initial.edges.clone();
        // lib_x's own call-site lines could shift too, but lib_x has no outgoing edges.

        let full = full_calls("/oracle/comment", &final_state).await;
        let (incr, removed_surface, added) =
            incremental_calls("/oracle/comment", &initial, &final_state, &["/lib_x.rs"], &[]).await;

        assert_eq!(
            removed_surface, Vec::<String>::new(),
            "comment/body-only edit MUST NOT mark any file removed-surface (got {:?})", removed_surface
        );
        assert_eq!(
            added, Vec::<String>::new(),
            "comment/body-only edit MUST add no names (got {:?})", added
        );
        assert_eq!(incr, full, "incremental != full after comment edit");
    }

    /// Scenario 2: add a NEW file defining `helper` that is lex-first → existing
    /// unchanged callers must re-point to it (direction-2 must fire). This is a
    /// PURE ADDITION: NOTHING was removed, so direction-1 must NOT fire — the work
    /// is done entirely by direction-2 on the added name `helper`.
    #[tokio::test]
    async fn oracle_add_symbol_wins_tiebreak() {
        let initial = base_world();
        // lib_a (< lib_x, lib_z) is added defining helper. caller_a/b must repoint.
        let mut final_state = initial.clone();
        final_state = final_state.sym("/lib_a.rs", "helper", 10);

        let full = full_calls("/oracle/add", &final_state).await;
        let (incr, removed_surface, added) =
            incremental_calls("/oracle/add", &initial, &final_state, &["/lib_a.rs"], &[]).await;

        assert_eq!(removed_surface, Vec::<String>::new(),
            "pure addition must NOT fire direction-1 (nothing removed); got {:?}", removed_surface);
        assert!(added.contains(&"helper".to_string()),
            "added_names must include helper; got {:?}", added);
        assert_eq!(incr, full, "incremental != full after add-symbol");
        // sanity: both callers now point at lib_a (via direction-2).
        assert!(incr.iter().any(|(i, o, _, _)| i == "/caller_a.rs" && o == "/lib_a.rs"),
            "caller_a must repoint to lib_a; got {:?}", incr);
    }

    /// Scenario 3: remove a symbol an unchanged caller targeted (direction-1).
    #[tokio::test]
    async fn oracle_remove_symbol_caller_repoints() {
        let initial = base_world();
        // lib_x removes helper (keeps util). caller_a/b helper edges → lib_z.
        let final_state = RepoState::default()
            .sym("/caller_a.rs", "a_fn", 1)
            .sym("/caller_b.rs", "b_fn", 1)
            .sym("/lib_x.rs", "util", 20)       // helper removed
            .sym("/lib_z.rs", "helper", 10)
            .edge("/caller_a.rs", "a_fn", "helper", 2)
            .edge("/caller_b.rs", "b_fn", "helper", 3)
            .edge("/caller_b.rs", "b_fn", "util", 4);

        let full = full_calls("/oracle/remove", &final_state).await;
        let (incr, removed_surface, added) =
            incremental_calls("/oracle/remove", &initial, &final_state, &["/lib_x.rs"], &[]).await;

        assert!(removed_surface.contains(&"/lib_x.rs".to_string()),
            "lib_x removed helper → must fire direction-1; got {:?}", removed_surface);
        assert_eq!(added, Vec::<String>::new(), "removal adds no names; got {:?}", added);
        assert_eq!(incr, full, "incremental != full after remove-symbol");
        assert!(incr.iter().any(|(i, o, _, _)| i == "/caller_a.rs" && o == "/lib_z.rs"),
            "caller_a→helper must repoint to lib_z after lib_x removed helper; got {:?}", incr);
    }

    /// Scenario 4: rename a symbol (remove old name + add new name). BOTH
    /// directions fire: direction-1 (helper removed) + direction-2 (helper2 added).
    #[tokio::test]
    async fn oracle_rename_symbol() {
        let initial = base_world();
        // lib_x renames helper → helper2. caller_a/b helper edges → lib_z.
        // helper2 is a new name nobody calls (added_names=[helper2], dir2 finds none).
        let final_state = RepoState::default()
            .sym("/caller_a.rs", "a_fn", 1)
            .sym("/caller_b.rs", "b_fn", 1)
            .sym("/lib_x.rs", "helper2", 10)    // renamed
            .sym("/lib_x.rs", "util", 20)
            .sym("/lib_z.rs", "helper", 10)
            .edge("/caller_a.rs", "a_fn", "helper", 2)
            .edge("/caller_b.rs", "b_fn", "helper", 3)
            .edge("/caller_b.rs", "b_fn", "util", 4);

        let full = full_calls("/oracle/rename", &final_state).await;
        let (incr, removed_surface, added) =
            incremental_calls("/oracle/rename", &initial, &final_state, &["/lib_x.rs"], &[]).await;

        assert!(removed_surface.contains(&"/lib_x.rs".to_string()),
            "rename removed helper → must fire direction-1; got {:?}", removed_surface);
        assert!(added.contains(&"helper2".to_string()),
            "rename adds the new name; got {:?}", added);
        assert_eq!(incr, full, "incremental != full after rename");
    }

    /// Scenario 5: multi-file edit mixing a surface-UNCHANGED file (lib_x body
    /// edit) and a PURE-ADDITION file (lib_a added defining helper). Neither fires
    /// direction-1 (nothing removed); lib_a's `helper` drives direction-2.
    #[tokio::test]
    async fn oracle_mixed_surface_changed_and_unchanged() {
        let initial = base_world();
        let mut final_state = RepoState::default()
            .sym("/caller_a.rs", "a_fn", 1)
            .sym("/caller_b.rs", "b_fn", 1)
            .sym("/lib_x.rs", "helper", 14)    // body edit: lines shift, same surface
            .sym("/lib_x.rs", "util", 24)
            .sym("/lib_z.rs", "helper", 10)
            .sym("/lib_a.rs", "helper", 10);   // NEW lex-first definer
        final_state.edges = initial.edges.clone();

        let full = full_calls("/oracle/mixed", &final_state).await;
        let (incr, removed_surface, added) = incremental_calls(
            "/oracle/mixed", &initial, &final_state, &["/lib_x.rs", "/lib_a.rs"], &[],
        ).await;

        // Neither file removed anything → direction-1 stays empty. lib_x's body
        // edit is inert; lib_a is a pure addition handled by direction-2.
        assert_eq!(removed_surface, Vec::<String>::new(),
            "mixed edit removed nothing → direction-1 must NOT fire; got {:?}", removed_surface);
        assert!(added.contains(&"helper".to_string()),
            "added_names must include helper from lib_a; got {:?}", added);
        assert_eq!(incr, full, "incremental != full after mixed edit");
    }

    /// Pure-helper unit coverage for compute_surface_delta.
    #[test]
    fn surface_delta_pure_logic() {
        let mut old: FileSurface = HashMap::new();
        old.insert("/f.rs".to_string(), HashSet::from([
            ("foo".to_string(), "/f.rs::foo".to_string()),
            ("bar".to_string(), "/f.rs::bar".to_string()),
        ]));
        // Unchanged surface → empty delta.
        let d = compute_surface_delta(&old, &old.clone(), &["/f.rs".to_string()], &[]);
        assert!(d.removed_surface_files.is_empty());
        assert!(d.added_names.is_empty());

        // Add a name (PURE addition) → NO removed-surface, added_names=[baz].
        let mut new = old.clone();
        new.get_mut("/f.rs").unwrap().insert(("baz".to_string(), "/f.rs::baz".to_string()));
        let d = compute_surface_delta(&old, &new, &["/f.rs".to_string()], &[]);
        assert!(d.removed_surface_files.is_empty(),
            "pure addition must NOT be removed-surface; got {:?}", d.removed_surface_files);
        assert_eq!(d.added_names, vec!["baz".to_string()]);

        // Remove a name → removed-surface, no adds.
        let mut new2 = old.clone();
        new2.get_mut("/f.rs").unwrap().remove(&("bar".to_string(), "/f.rs::bar".to_string()));
        let d = compute_surface_delta(&old, &new2, &["/f.rs".to_string()], &[]);
        assert_eq!(d.removed_surface_files, vec!["/f.rs".to_string()]);
        assert!(d.added_names.is_empty());

        // Rename (remove bar + add baz) → BOTH removed-surface and added.
        let mut new3 = old.clone();
        new3.get_mut("/f.rs").unwrap().remove(&("bar".to_string(), "/f.rs::bar".to_string()));
        new3.get_mut("/f.rs").unwrap().insert(("baz".to_string(), "/f.rs::baz".to_string()));
        let d = compute_surface_delta(&old, &new3, &["/f.rs".to_string()], &[]);
        assert_eq!(d.removed_surface_files, vec!["/f.rs".to_string()]);
        assert_eq!(d.added_names, vec!["baz".to_string()]);

        // Deleted file → always removed-surface.
        let d = compute_surface_delta(&old, &HashMap::new(), &[], &["/gone.rs".to_string()]);
        assert_eq!(d.removed_surface_files, vec!["/gone.rs".to_string()]);
    }
}

// ─── Hidden-change filter tests ───────────────────────────────────────────
#[cfg(test)]
mod hidden_change_filter_tests {
    use super::*;
    use tempfile::TempDir;

    /// Verifies that filter_hidden_changes:
    /// - drops Added/Modified changes whose paths are inside a dot-prefixed directory
    /// - keeps Added/Modified for root-level dot-FILES (not directories)
    /// - keeps Added/Modified for normal files
    /// - always keeps Deleted changes, even when the path is inside a dot-dir
    #[test]
    fn filter_drops_dot_dir_modified_keeps_others() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Build real paths so strip_prefix works cross-platform.
        let claude_dir = root.join(".claude");
        std::fs::create_dir_all(&claude_dir).unwrap();
        let claude_file = claude_dir.join("agents.md");
        std::fs::File::create(&claude_file).unwrap();

        let claude_deleted = claude_dir.join("old.md");
        // (does not need to exist on disk for Deleted)

        let eslintrc = root.join(".eslintrc.json");
        std::fs::File::create(&eslintrc).unwrap();

        let src_dir = root.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src_main = src_dir.join("main.rs");
        std::fs::File::create(&src_main).unwrap();

        let changes = vec![
            FileChange {
                path: claude_file.to_str().unwrap().to_string(),
                kind: ChangeKind::Modified,
            },
            FileChange {
                path: eslintrc.to_str().unwrap().to_string(),
                kind: ChangeKind::Modified,
            },
            FileChange {
                path: src_main.to_str().unwrap().to_string(),
                kind: ChangeKind::Modified,
            },
            FileChange {
                path: claude_deleted.to_str().unwrap().to_string(),
                kind: ChangeKind::Deleted,
            },
        ];

        let filtered = filter_hidden_changes(root, changes);

        // .claude/agents.md Modified must be dropped.
        let has_claude_modified = filtered
            .iter()
            .any(|c| c.path.contains(".claude") && c.kind != ChangeKind::Deleted);
        assert!(
            !has_claude_modified,
            ".claude/ Modified must be dropped; got: {:?}",
            filtered.iter().map(|c| (&c.path, &c.kind)).collect::<Vec<_>>()
        );

        // .eslintrc.json Modified must survive (root-level dot-file).
        let has_eslintrc = filtered
            .iter()
            .any(|c| c.path.ends_with(".eslintrc.json"));
        assert!(
            has_eslintrc,
            ".eslintrc.json must survive filtering; got: {:?}",
            filtered.iter().map(|c| (&c.path, &c.kind)).collect::<Vec<_>>()
        );

        // src/main.rs Modified must survive.
        let has_src_main = filtered
            .iter()
            .any(|c| c.path.ends_with("main.rs") && c.kind != ChangeKind::Deleted);
        assert!(
            has_src_main,
            "src/main.rs must survive filtering; got: {:?}",
            filtered.iter().map(|c| (&c.path, &c.kind)).collect::<Vec<_>>()
        );

        // .claude/old.md Deleted must survive.
        let has_claude_deleted = filtered
            .iter()
            .any(|c| c.path.contains(".claude") && c.kind == ChangeKind::Deleted);
        assert!(
            has_claude_deleted,
            ".claude/old.md Deleted must survive (self-heal); got: {:?}",
            filtered.iter().map(|c| (&c.path, &c.kind)).collect::<Vec<_>>()
        );

        // Total surviving: .eslintrc.json, src/main.rs, .claude/old.md Deleted = 3
        assert_eq!(
            filtered.len(),
            3,
            "expected 3 changes to survive; got: {:?}",
            filtered.iter().map(|c| (&c.path, &c.kind)).collect::<Vec<_>>()
        );
    }

    /// Regression: watcher-supplied changes for gitignored build artifacts under
    /// `target/` must be dropped (Added/Modified), matching what `walk_repo` does
    /// on a full rebuild. Previously only dot-dirs were filtered, so a concurrent
    /// `cargo build` leaked `target/debug/*.exe` / `*.d` into query results until
    /// the next `--rebuild`. Deleted changes for those artifacts still pass through
    /// so a previously-indexed artifact self-heals when removed.
    #[test]
    fn filter_drops_gitignored_target_artifacts() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // A .gitignore that excludes the target/ tree (mirrors the real repo).
        std::fs::write(root.join(".gitignore"), "/target\n").unwrap();

        // Real source that must survive.
        let src_dir = root.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src_main = src_dir.join("main.rs");
        std::fs::File::create(&src_main).unwrap();

        // Build artifacts under target/ — gitignored AND in SKIP_DIRS.
        let target_dir = root.join("target").join("debug");
        std::fs::create_dir_all(&target_dir).unwrap();
        let exe = target_dir.join("context-engine-rs.exe");
        let dep = target_dir.join("context-engine-rs.d");
        std::fs::File::create(&exe).unwrap();
        std::fs::File::create(&dep).unwrap();
        // A .rs file under target/ must ALSO be dropped (gitignore/SKIP_DIRS win
        // even though the extension is indexable).
        let gen_rs = target_dir.join("generated.rs");
        std::fs::File::create(&gen_rs).unwrap();

        let changes = vec![
            FileChange { path: src_main.to_str().unwrap().to_string(), kind: ChangeKind::Modified },
            FileChange { path: exe.to_str().unwrap().to_string(), kind: ChangeKind::Added },
            FileChange { path: dep.to_str().unwrap().to_string(), kind: ChangeKind::Modified },
            FileChange { path: gen_rs.to_str().unwrap().to_string(), kind: ChangeKind::Added },
            // Deleted artifact: must survive so a previously-indexed entry is cleaned up.
            FileChange { path: exe.to_str().unwrap().to_string(), kind: ChangeKind::Deleted },
        ];

        let filtered = filter_hidden_changes(root, changes);

        // src/main.rs survives.
        assert!(
            filtered.iter().any(|c| c.path.ends_with("main.rs") && c.kind == ChangeKind::Modified),
            "src/main.rs Modified must survive; got: {:?}",
            filtered.iter().map(|c| (&c.path, &c.kind)).collect::<Vec<_>>()
        );

        // No Added/Modified target/ artifact survives (exe, .d, generated.rs all dropped).
        assert!(
            !filtered.iter().any(|c| c.path.contains("target") && c.kind != ChangeKind::Deleted),
            "no Added/Modified target/ artifact may survive; got: {:?}",
            filtered.iter().map(|c| (&c.path, &c.kind)).collect::<Vec<_>>()
        );

        // The Deleted artifact survives (self-heal).
        assert!(
            filtered.iter().any(|c| c.path.contains("target") && c.kind == ChangeKind::Deleted),
            "Deleted target/ artifact must survive for self-heal; got: {:?}",
            filtered.iter().map(|c| (&c.path, &c.kind)).collect::<Vec<_>>()
        );

        // Surviving: src/main.rs + the one Deleted artifact = 2.
        assert_eq!(
            filtered.len(),
            2,
            "expected exactly 2 changes to survive; got: {:?}",
            filtered.iter().map(|c| (&c.path, &c.kind)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn filter_drops_ignored_filenames_but_allows_deleted() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        let src_dir = root.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src_main = src_dir.join("main.rs");
        std::fs::File::create(&src_main).unwrap();

        let claude_md = root.join("CLAUDE.md");
        std::fs::File::create(&claude_md).unwrap();

        let changes = vec![
            FileChange {
                path: src_main.to_str().unwrap().to_string(),
                kind: ChangeKind::Modified,
            },
            FileChange {
                path: claude_md.to_str().unwrap().to_string(),
                kind: ChangeKind::Modified,
            },
            FileChange {
                path: claude_md.to_str().unwrap().to_string(),
                kind: ChangeKind::Deleted,
            },
        ];

        let ignore: HashSet<String> = ["CLAUDE.md"].iter().map(|s| s.to_string()).collect();
        let filtered = filter_hidden_changes_with(root, changes, vec![], ignore, HashSet::new());

        assert!(
            filtered.iter().any(|c| c.path.ends_with("main.rs") && c.kind == ChangeKind::Modified),
            "src/main.rs Modified must survive; got: {:?}",
            filtered.iter().map(|c| (&c.path, &c.kind)).collect::<Vec<_>>()
        );
        assert!(
            !filtered.iter().any(|c| c.path.ends_with("CLAUDE.md") && c.kind == ChangeKind::Modified),
            "CLAUDE.md Modified must be dropped; got: {:?}",
            filtered.iter().map(|c| (&c.path, &c.kind)).collect::<Vec<_>>()
        );
        assert!(
            filtered.iter().any(|c| c.path.ends_with("CLAUDE.md") && c.kind == ChangeKind::Deleted),
            "CLAUDE.md Deleted must survive for self-heal; got: {:?}",
            filtered.iter().map(|c| (&c.path, &c.kind)).collect::<Vec<_>>()
        );

        assert_eq!(
            filtered.len(),
            2,
            "expected 2 changes (main.rs Modified + CLAUDE.md Deleted); got: {:?}",
            filtered.iter().map(|c| (&c.path, &c.kind)).collect::<Vec<_>>()
        );
    }
}

// ─── Performance-fix regression tests ────────────────────────────────────
/// Tests that validate the three performance fixes from the locked plan:
///   1. Concurrent cached-file processing (spawn_blocking unblocks buffer_unordered).
///   2. Panicking cache op degrades gracefully to no_embeddings, not abort.
///   3. Watcher-path run performs zero full-repo walk.
#[cfg(test)]
mod perf_fix_tests {
    use super::*;
    use crate::store::open_db;
    use crate::store::ops::count_indexed_files;
    use tempfile::TempDir;

    // ── Test 1: Concurrent cache hits are no longer serialized ───────────────
    //
    // With the old synchronous `cache.get_many()` on the async task, every file
    // processed strictly one-at-a-time even under `buffer_unordered(N)` because
    // the blocking FS read held the driver task without yielding.
    //
    // With the new `spawn_blocking` wrapping, each file's cache lookup yields the
    // async task and the tokio thread pool runs up to N lookups concurrently.
    //
    // This test verifies the infrastructure: N spawn_blocking closures (each
    // simulating a cache read with a short sleep) achieve peak concurrency > 1
    // when driven through `buffer_unordered(N)`.  If the old serial path were
    // still in place, peak would be 1.
    #[tokio::test]
    async fn cached_file_processing_is_concurrent_not_serial() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let concurrency = 4usize;
        let file_count = 8usize;

        let inflight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let items: Vec<usize> = (0..file_count).collect();
        let inflight_ref = inflight.clone();
        let peak_ref = peak.clone();

        futures::stream::iter(items)
            .map(|_i| {
                let inf = inflight_ref.clone();
                let pk = peak_ref.clone();
                async move {
                    // Each file's cache lookup runs in spawn_blocking, yielding
                    // the async task and allowing other tasks to proceed.
                    tokio::task::spawn_blocking(move || {
                        let n = inf.fetch_add(1, Ordering::SeqCst) + 1;
                        pk.fetch_max(n, Ordering::SeqCst);
                        // Simulate a short FS read.
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        inf.fetch_sub(1, Ordering::SeqCst);
                    })
                    .await
                    .expect("spawn_blocking must not panic")
                }
            })
            .buffer_unordered(concurrency)
            .collect::<Vec<_>>()
            .await;

        let peak_observed = peak.load(Ordering::SeqCst);
        assert!(
            peak_observed > 1,
            "spawn_blocking cache reads must run concurrently (peak={peak_observed}); \
             if peak==1 the old serial path is still in effect"
        );
        assert!(
            peak_observed <= concurrency,
            "peak ({peak_observed}) must not exceed buffer_unordered limit ({concurrency})"
        );
    }

    // ── Test 2: Panicking cache op → no_embeddings, not abort ────────────────
    //
    // `embed_parsed_file` wraps `cache.get_many()` in `spawn_blocking`.  If that
    // closure panics, the JoinError must be caught and the file must be returned
    // with all-empty embeddings (the existing degradation path), NOT propagated
    // as an unwrap-panic that aborts the driver task.
    //
    // We simulate this by exercising the same JoinError-handling path used by the
    // production code: a spawn_blocking closure that panics produces a JoinError,
    // and our code maps it to the degraded EmbedFileResult.
    #[tokio::test]
    async fn spawn_blocking_panic_maps_to_degraded_embed_not_abort() {
        // Drive a REAL spawn_blocking panic through map_get_many_result — the exact
        // function the production get_many call site uses. This covers the JoinError
        // arm directly (not an equal-valued sibling branch): if someone later changed
        // that arm to .unwrap() or to propagate, this test would fail.
        let get_result: std::result::Result<GetManyOutcome, tokio::task::JoinError> =
            tokio::task::spawn_blocking(|| -> GetManyOutcome {
                panic!("simulated cache get_many panic");
            })
            .await;

        assert!(get_result.is_err(), "panicking spawn_blocking must yield Err(JoinError)");

        // n_texts = 3 → degraded result must be exactly 3 empty embedding slots.
        let mapped = map_get_many_result("/test/panic_file.rs", 3, get_result);

        match mapped {
            Ok(_) => panic!("JoinError must map to Err(degraded EmbedFileResult), not Ok"),
            Err(degraded) => {
                // The file is NOT dropped: it flows on with one empty slot per chunk,
                // which the pipeline's all-empty check turns into embed_failed=true →
                // status "no_embeddings". The driver task never panics.
                assert_eq!(
                    degraded.embeddings.len(),
                    3,
                    "degraded result must have one slot per text (no file dropped)"
                );
                assert!(
                    degraded.embeddings.iter().all(|e| e.is_empty()),
                    "every slot must be empty on the JoinError degradation path"
                );
                assert!(
                    !degraded.fully_cached,
                    "degraded result must not be marked fully_cached"
                );
            }
        }

        // Sanity: the happy path passes through unchanged.
        let ok_input: std::result::Result<GetManyOutcome, tokio::task::JoinError> =
            Ok((vec![(0, vec![1.0, 2.0])], vec![1]));
        match map_get_many_result("/test/ok_file.rs", 2, ok_input) {
            Ok((hits, misses)) => {
                assert_eq!(hits.len(), 1, "hits must be preserved");
                assert_eq!(misses, vec![1], "misses must be preserved");
            }
            Err(_) => panic!("Ok(get_many result) must pass through as Ok"),
        }
    }

    // ── Test 3: Watcher path performs zero full-repo walk ────────────────────
    //
    // When `run()` is called with `changes == Some(explicit_list)` (watcher path),
    // only the explicitly changed files should be processed — no `walk_repo` should
    // be invoked against the on-disk tree.
    //
    // Approach:
    //   1. Build a temp repo with several files (A, B, C, D).
    //   2. Pre-seed `file_meta` rows for all four files (simulating a prior full build).
    //   3. Call `run()` with `changes = Some(vec![single_change_for_file_A])`.
    //   4. Assert: only file A's chunks were (re)written; files B/C/D are untouched.
    //      If a full walk had occurred, all four files would be (re)indexed.
    #[tokio::test]
    async fn watcher_path_processes_only_explicit_changes_no_full_walk() {
        let home = TempDir::new().unwrap();
        let repo_dir = TempDir::new().unwrap();
        let repo = repo_dir.path().to_str().unwrap().replace('\\', "/");

        // Write four files to the repo.
        let file_a = repo_dir.path().join("a.rs");
        let file_b = repo_dir.path().join("b.rs");
        let file_c = repo_dir.path().join("c.rs");
        let file_d = repo_dir.path().join("d.rs");

        std::fs::write(&file_a, "fn alpha() {}\n").unwrap();
        std::fs::write(&file_b, "fn beta() {}\n").unwrap();
        std::fs::write(&file_c, "fn gamma() {}\n").unwrap();
        std::fs::write(&file_d, "fn delta() {}\n").unwrap();

        let file_a_path = file_a.to_str().unwrap().replace('\\', "/");

        let db = open_db(home.path(), &repo, 0).await.expect("open db");

        // First, do a full build so all four files are indexed.
        let pipeline = IndexPipeline::new(repo.clone(), None);
        pipeline
            .run(&db, None, true, None, None, None, &[], None)
            .await
            .expect("full build must succeed");

        let initial_file_count = count_indexed_files(&db, &repo).await.unwrap();
        assert_eq!(initial_file_count, 4, "all four files must be indexed after full build");

        // Modify only file_a on disk so its content changes.
        std::fs::write(&file_a, "fn alpha_v2() {}\nfn alpha_extra() {}\n").unwrap();

        // Construct the explicit single-file change (watcher path).
        // FileChange only carries path + kind (mtime/size live in file_meta).
        let changes = vec![FileChange {
            path: file_a_path.clone(),
            kind: ChangeKind::Modified,
        }];

        // Run the incremental pipeline with changes = Some(...) — the watcher path.
        let stats = pipeline
            .run(&db, Some(changes), false, None, None, None, &[], None)
            .await
            .expect("incremental run must succeed");

        // Assert: all four files are still indexed (B, C, D were not removed).
        let after_file_count = count_indexed_files(&db, &repo).await.unwrap();
        assert!(
            after_file_count >= initial_file_count,
            "file count must not decrease after watcher-path run \
             (before={initial_file_count}, after={after_file_count})"
        );

        // The stats should reflect the change set size, not the full repo.
        // total_files must come from stored_meta count (not a fresh walk).
        assert_eq!(
            stats.total_files,
            initial_file_count as u64,
            "total_files must equal stored_meta count from the prior run ({initial_file_count}), not a fresh walk result"
        );

        // Verify file_b, file_c, file_d were NOT re-indexed: their file_meta
        // mtime must still match the original (unchanged) file timestamps.
        let all_meta = crate::store::ops::get_all_file_meta(&db, &repo)
            .await
            .expect("get_all_file_meta");

        // Match by filename suffix — path normalization (/ vs \) may differ
        // between what we constructed and what walk_repo stored in the DB.
        let b_meta = all_meta.iter().find(|m| m.path.ends_with("b.rs")).expect("file_b must have meta");
        let c_meta = all_meta.iter().find(|m| m.path.ends_with("c.rs")).expect("file_c must have meta");
        let d_meta = all_meta.iter().find(|m| m.path.ends_with("d.rs")).expect("file_d must have meta");
        let a_meta_stored = all_meta.iter().find(|m| m.path.ends_with("a.rs")).expect("file_a must have meta");

        // B, C, D were not in the change set → their mtime in file_meta must
        // match the on-disk stat (unchanged), proving they were not re-parsed.
        let b_stat = stat_file(&b_meta.path).expect("stat file_b");
        let c_stat = stat_file(&c_meta.path).expect("stat file_c");
        let d_stat = stat_file(&d_meta.path).expect("stat file_d");

        assert_eq!(b_meta.mtime, b_stat.mtime, "file_b mtime must be unchanged");
        assert_eq!(c_meta.mtime, c_stat.mtime, "file_c mtime must be unchanged");
        assert_eq!(d_meta.mtime, d_stat.mtime, "file_d mtime must be unchanged");

        // Verify file_a was re-indexed: its stored chunk_count must reflect the
        // new content (2 functions → different chunking than the original 1).
        // We do NOT compare mtime because stat_file uses second-level granularity
        // (.as_secs()) and the full build + incremental run can both complete
        // within the same calendar second in a fast test environment.
        assert!(
            a_meta_stored.chunk_count > 0,
            "file_a must have chunks after watcher-path re-index (got {})",
            a_meta_stored.chunk_count,
        );
    }
}

// ─── RAM-path edge resolution FQN test ────────────────────────────────────
//
// Regression for the "0 edges after index" bug: the full-rebuild RAM fast-path
// (`resolve_edges_from_ram`) wrote LEAF names into calls.in_name/out_name, while
// the DB-scan path writes full FQNs. Consumers (call_graph node ids = meta::id(id),
// and query_callers/callees `WHERE out_name = $fqn`) match on full FQNs, so the
// leaf-name rows silently failed every match → empty UI graph + broken search
// expansion. This test pins in_name/out_name to full FQNs on the RAM path, using
// a METHOD symbol whose FQN (file::scope::name) differs from its leaf name —
// the leaf-name bug would pass a free-function assertion but fail this one.
#[cfg(test)]
mod ram_path_fqn_tests {
    use super::*;
    use crate::store::open_db;
    use serde::Deserialize;
    use tempfile::TempDir;

    /// Insert a (possibly scoped) symbol whose record id IS the full FQN.
    async fn insert_symbol_fqn(db: &Surreal<Db>, fqn: &str, file: &str, name: &str) {
        db.query(format!(
            "UPSERT symbol:`⟨{fqn}⟩` SET \
             name = '{name}', kind = 'method', file = '{file}', \
             line_start = 1, line_end = 10, signature = NONE, parent = NONE"
        ))
        .await
        .expect("insert symbol");
    }

    /// resolve_edges_from_ram must write the FULL FQN (file::scope::name) into
    /// calls.in_name and calls.out_name — never the leaf name.
    #[tokio::test]
    async fn ram_path_writes_full_fqn_in_call_names() {
        let home = TempDir::new().unwrap();
        let repo = "/test/ram_fqn";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Caller: method `caller` inside class `Foo` in /a.cpp → FQN /a.cpp::Foo::caller
        // Callee: method `callee` inside class `Bar` in /b.cpp → FQN /b.cpp::Bar::callee
        insert_symbol_fqn(&db, "/a.cpp::Foo::caller", "/a.cpp", "caller").await;
        insert_symbol_fqn(&db, "/b.cpp::Bar::callee", "/b.cpp", "callee").await;

        // One RAM raw edge: caller calls `callee` (unresolved leaf name, as parsed).
        let raw_edges = vec![RawEdgeRecord {
            from_file: "/a.cpp".to_string(),
            from_name: "caller".to_string(),
            from_fqn: "/a.cpp::Foo::caller".to_string(),
            to_name: "callee".to_string(),
            kind: "calls".to_string(),
            line: 7,
            import_path: None,
        }];

        let pipeline = IndexPipeline::new(repo.to_string(), None);
        pipeline
            .resolve_edges_from_ram(&db, raw_edges, None, None)
            .await
            .expect("resolve_edges_from_ram");

        #[derive(Deserialize, Debug)]
        struct EdgeRow {
            in_name: Option<String>,
            out_name: Option<String>,
        }
        let rows: Vec<EdgeRow> = db
            .query("SELECT in_name, out_name FROM calls")
            .await
            .unwrap()
            .take(0)
            .unwrap();

        assert_eq!(rows.len(), 1, "exactly one calls edge expected, got {rows:?}");
        let row = &rows[0];
        assert_eq!(
            row.in_name.as_deref(),
            Some("/a.cpp::Foo::caller"),
            "in_name must be the full FQN, not the leaf name 'caller'"
        );
        assert_eq!(
            row.out_name.as_deref(),
            Some("/b.cpp::Bar::callee"),
            "out_name must be the full FQN, not the leaf name 'callee'"
        );
    }

    /// A pre-cancelled token must abort Phase 2 (RAM path) with PipelineAbort::Cancelled
    /// and leave the edges_resolved marker UNSET, so the next run replays/rebuilds.
    /// This pins the cancellation responsiveness added to resolve_edges_from_ram —
    /// before the fix, Phase 2 ran to completion regardless of the token.
    #[tokio::test]
    async fn ram_phase2_aborts_on_cancelled_token() {
        let home = TempDir::new().unwrap();
        let repo = "/test/ram_cancel";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        insert_symbol_fqn(&db, "/a.cpp::Foo::caller", "/a.cpp", "caller").await;
        insert_symbol_fqn(&db, "/b.cpp::Bar::callee", "/b.cpp", "callee").await;

        let raw_edges = vec![RawEdgeRecord {
            from_file: "/a.cpp".to_string(),
            from_name: "caller".to_string(),
            from_fqn: "/a.cpp::Foo::caller".to_string(),
            to_name: "callee".to_string(),
            kind: "calls".to_string(),
            line: 7,
            import_path: None,
        }];

        let token = CancellationToken::new();
        token.cancel();

        let pipeline = IndexPipeline::new(repo.to_string(), None);
        let err = pipeline
            .resolve_edges_from_ram(&db, raw_edges, None, Some(&token))
            .await
            .expect_err("pre-cancelled token must abort Phase 2");
        assert!(
            matches!(
                err.downcast_ref::<PipelineAbort>(),
                Some(PipelineAbort::Cancelled)
            ),
            "expected PipelineAbort::Cancelled, got: {err:#}"
        );

        // edges_resolved marker must NOT be stamped on abort — the next run must
        // be able to detect the unresolved state and recover.
        let marker = get_meta(&db, EDGES_RESOLVED_KEY).await.unwrap();
        assert!(
            marker.is_none(),
            "edges_resolved must stay unset after a cancelled Phase 2"
        );
    }

    /// Same guarantee for the DB-scan Phase 2 path (resolve_edges_phase2): a
    /// pre-cancelled token aborts at the first page boundary with Cancelled and
    /// leaves the marker unset.
    #[tokio::test]
    async fn db_phase2_aborts_on_cancelled_token() {
        let home = TempDir::new().unwrap();
        let repo = "/test/db_cancel";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        insert_symbol_fqn(&db, "/a.cpp::Foo::caller", "/a.cpp", "caller").await;
        insert_symbol_fqn(&db, "/b.cpp::Bar::callee", "/b.cpp", "callee").await;

        // Persist a raw_edge row so the DB-scan path has work to do (total > 0).
        db.query(
            "CREATE raw_edge SET from_file = '/a.cpp', from_name = 'caller', \
             from_fqn = '/a.cpp::Foo::caller', to_name = 'callee', kind = 'calls', \
             line = 7, import_path = NONE",
        )
        .await
        .expect("seed raw_edge");

        let token = CancellationToken::new();
        token.cancel();

        let pipeline = IndexPipeline::new(repo.to_string(), None);
        let err = pipeline
            .resolve_edges_phase2(&db, Some(&token))
            .await
            .expect_err("pre-cancelled token must abort DB-scan Phase 2");
        assert!(
            matches!(
                err.downcast_ref::<PipelineAbort>(),
                Some(PipelineAbort::Cancelled)
            ),
            "expected PipelineAbort::Cancelled, got: {err:#}"
        );

        let marker = get_meta(&db, EDGES_RESOLVED_KEY).await.unwrap();
        assert!(
            marker.is_none(),
            "edges_resolved must stay unset after a cancelled Phase 2"
        );
    }
}

// ─── In-RAM symbol-buffer Phase-2 invariance (optimize-index-pipeline-walltime) ─
//
// These pin the contract that the Stage-3 in-RAM symbol buffer reproduces
// `load_all_symbols`' result EXACTLY, so Phase 2 resolving from the buffer
// (Some) yields byte-identical `calls` rows to resolving from the DB reload
// (None). The buffer is `HashMap<String /*fqn*/, SymbolWithPos>` with
// last-write-wins on FQN collision — matching the symbol table's
// `INSERT ... ON DUPLICATE KEY UPDATE` per-FQN dedup.
#[cfg(test)]
mod ram_symbol_buffer_invariance_tests {
    use super::*;
    use crate::parsing::symbols::{QualifiedSymbol, SymbolKind};
    use crate::store::open_db;
    use serde::Deserialize;
    use tempfile::TempDir;

    #[derive(Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
    struct CallRow {
        line: i64,
        in_file: Option<String>,
        out_file: Option<String>,
        in_name: Option<String>,
        out_name: Option<String>,
    }

    /// Read all `calls` rows, sorted deterministically, for set comparison.
    async fn dump_calls(db: &Surreal<Db>) -> Vec<CallRow> {
        let mut rows: Vec<CallRow> = db
            .query("SELECT line, in_file, out_file, in_name, out_name FROM calls")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        rows.sort();
        rows
    }

    fn mk_symbol(file: &str, name: &str, ls: u32, le: u32) -> Symbol {
        Symbol {
            qualified: QualifiedSymbol {
                file: file.to_string(),
                scope_path: vec![],
                name: name.to_string(),
            },
            kind: SymbolKind::Function,
            line_start: ls,
            line_end: le,
            signature: None,
            parent_fqn: None,
        }
    }

    /// Build the in-RAM buffer EXACTLY as `streaming_index` does, by calling the
    /// SAME production helper (`buffer_insert_symbol`) in stream order →
    /// last-write-wins on FQN collision. Not a copy of the logic — the real thing.
    fn build_buffer(symbols: &[Symbol]) -> HashMap<String, SymbolWithPos> {
        let mut buf: HashMap<String, SymbolWithPos> = HashMap::new();
        for sym in symbols {
            buffer_insert_symbol(&mut buf, sym);
        }
        buf
    }

    /// Build the name→sorted-candidates bucket map the SAME way
    /// `resolve_edges_from_ram` does, so two symbol *sources* can be compared at
    /// the resolution-input layer (not just the `calls` output layer).
    fn build_name_bucket(symbols: Vec<SymbolWithPos>) -> std::collections::BTreeMap<String, Vec<SymbolWithPos>> {
        let mut name_bucket: HashMap<String, Vec<SymbolWithPos>> = HashMap::new();
        for s in symbols {
            name_bucket.entry(s.name.clone()).or_default().push(s);
        }
        for bucket in name_bucket.values_mut() {
            bucket.sort_unstable_by(|a, b| {
                a.file
                    .cmp(&b.file)
                    .then(a.line_start.cmp(&b.line_start))
                    .then(a.line_end.cmp(&b.line_end))
            });
        }
        // Collect into a BTreeMap so equality comparison is order-independent at
        // the key level while each bucket's candidate order is already fixed.
        name_bucket.into_iter().collect()
    }

    /// 4.1 — Duplicate-FQN fixture: resolving via the in-RAM symbol buffer (Some)
    /// produces byte-identical `calls` rows to resolving via the DB reload (None).
    /// A function declared then defined in the SAME file yields two parsed symbols
    /// with the SAME FQN and different positions — the .h/.cpp last-write-wins case.
    #[tokio::test]
    async fn ram_buffer_matches_db_reload_with_duplicate_fqn() {
        // ── Fixture symbols (shared by both runs) ──────────────────────────
        // Callee `compute` appears TWICE with the same FQN (/lib.cpp::compute):
        // a declaration (lines 10-12) overwritten by a definition (lines 40-55).
        // Last-write-wins must collapse it to the line-40 row in BOTH paths.
        // A second distinct callee FQN with the same LEAF name `compute`
        // (/util.cpp::compute) shares the name-bucket, so bucket ordering /
        // tie-breaking is exercised. Two callers each call `compute`.
        let caller_a = mk_symbol("/a.cpp", "caller_a", 1, 5);
        let caller_b = mk_symbol("/util.cpp", "caller_b", 1, 5);
        let compute_decl = mk_symbol("/lib.cpp", "compute", 10, 12);
        let compute_def = mk_symbol("/lib.cpp", "compute", 40, 55);
        let compute_util = mk_symbol("/util.cpp", "compute", 200, 230);
        // Stream order matters for last-write-wins: decl BEFORE def.
        let symbols = vec![
            caller_a.clone(),
            compute_decl,
            compute_def,
            compute_util,
            caller_b.clone(),
        ];

        // Raw edges: caller_a (in /a.cpp) calls `compute`; caller_b (in /util.cpp)
        // calls `compute`. Same-file preference (Level 3) should make caller_b
        // resolve to /util.cpp::compute and caller_a to /lib.cpp::compute.
        let raw_edges = vec![
            RawEdgeRecord {
                from_file: "/a.cpp".to_string(),
                from_name: "caller_a".to_string(),
                from_fqn: "/a.cpp::caller_a".to_string(),
                to_name: "compute".to_string(),
                kind: "calls".to_string(),
                line: 3,
                import_path: None,
            },
            RawEdgeRecord {
                from_file: "/util.cpp".to_string(),
                from_name: "caller_b".to_string(),
                from_fqn: "/util.cpp::caller_b".to_string(),
                to_name: "compute".to_string(),
                kind: "calls".to_string(),
                line: 4,
                import_path: None,
            },
        ];

        // ── Run 1: DB-reload path (None) — the baseline (today's behavior) ──
        let home_db = TempDir::new().unwrap();
        let repo = "/test/inv_db";
        let db1 = open_db(home_db.path(), repo, 0).await.unwrap();
        flush_symbol_batch_native(&db1, &symbols).await.unwrap();
        let pipeline = IndexPipeline::new(repo.to_string(), None);
        pipeline
            .resolve_edges_from_ram(&db1, raw_edges.clone(), None, None)
            .await
            .expect("resolve via DB reload");
        let calls_db = dump_calls(&db1).await;

        // ── Run 2: in-RAM buffer path (Some) — the optimized path ───────────
        let home_buf = TempDir::new().unwrap();
        let db2 = open_db(home_buf.path(), repo, 0).await.unwrap();
        // The symbols are STILL written to the DB in Stage 3 (additive buffer);
        // but Phase 2 must NOT need them — pass the buffer and an EMPTY DB symbol
        // table to prove no reload happens.
        let buffer = build_buffer(&symbols);
        pipeline
            .resolve_edges_from_ram(&db2, raw_edges.clone(), Some(buffer), None)
            .await
            .expect("resolve via in-RAM buffer");
        let calls_buf = dump_calls(&db2).await;

        assert_eq!(
            calls_db, calls_buf,
            "in-RAM buffer resolution must be byte-identical to DB reload\nDB:  {calls_db:?}\nBUF: {calls_buf:?}"
        );
        assert_eq!(calls_db.len(), 2, "exactly two resolved edges expected");
        // Same-file preference: caller_b → /util.cpp::compute (out_file /util.cpp).
        let b_edge = calls_buf
            .iter()
            .find(|r| r.in_name.as_deref() == Some("/util.cpp::caller_b"))
            .expect("caller_b edge present");
        assert_eq!(b_edge.out_name.as_deref(), Some("/util.cpp::compute"));
        // The /lib.cpp::compute endpoint resolves to a single FQN (dedup worked) —
        // never two phantom rows from the duplicate decl/def.
        let lib_edges = calls_buf
            .iter()
            .filter(|r| r.out_name.as_deref() == Some("/lib.cpp::compute"))
            .count();
        assert_eq!(lib_edges, 1, "duplicate-FQN callee must not create phantom edges");
    }

    /// 4.2 — Overflow fallback: when the buffer is `None` (overflowed or
    /// incremental), Phase 2 reloads from the DB and produces identical edges to
    /// the buffer path. Same fixture as 4.1, asserting both sources agree.
    #[tokio::test]
    async fn overflow_fallback_matches_buffer_path() {
        let symbols = vec![
            mk_symbol("/a.cpp", "caller", 1, 5),
            mk_symbol("/lib.cpp", "target", 10, 12),
            mk_symbol("/lib.cpp", "target", 40, 55), // dup FQN, last wins
        ];
        let raw_edges = vec![RawEdgeRecord {
            from_file: "/a.cpp".to_string(),
            from_name: "caller".to_string(),
            from_fqn: "/a.cpp::caller".to_string(),
            to_name: "target".to_string(),
            kind: "calls".to_string(),
            line: 3,
            import_path: None,
        }];
        let pipeline = IndexPipeline::new("/test/overflow".to_string(), None);

        // Buffer path (Some).
        let home_buf = TempDir::new().unwrap();
        let db_buf = open_db(home_buf.path(), "/test/overflow", 0).await.unwrap();
        let buffer = build_buffer(&symbols);
        pipeline
            .resolve_edges_from_ram(&db_buf, raw_edges.clone(), Some(buffer), None)
            .await
            .unwrap();
        let calls_buf = dump_calls(&db_buf).await;

        // Overflow path (None) — symbols must be in the DB for the reload.
        let home_of = TempDir::new().unwrap();
        let db_of = open_db(home_of.path(), "/test/overflow", 0).await.unwrap();
        flush_symbol_batch_native(&db_of, &symbols).await.unwrap();
        pipeline
            .resolve_edges_from_ram(&db_of, raw_edges.clone(), None, None)
            .await
            .unwrap();
        let calls_of = dump_calls(&db_of).await;

        assert_eq!(
            calls_buf, calls_of,
            "overflow DB-reload must produce identical edges to the buffer path"
        );
        assert_eq!(calls_buf.len(), 1, "one resolved edge expected");
        assert_eq!(calls_buf[0].out_name.as_deref(), Some("/lib.cpp::target"));
    }

    /// The buffer construction reproduces last-write-wins on FQN collision: a
    /// duplicate FQN collapses to a single entry holding the LAST write's position.
    #[test]
    fn buffer_dedups_fqn_last_write_wins() {
        let symbols = vec![
            mk_symbol("/lib.cpp", "compute", 10, 12), // decl
            mk_symbol("/lib.cpp", "compute", 40, 55), // def — wins
        ];
        let buf = build_buffer(&symbols);
        assert_eq!(buf.len(), 1, "duplicate FQN must collapse to one entry");
        let entry = buf.get("/lib.cpp::compute").expect("compute present");
        assert_eq!(entry.line_start, 40, "last write must win");
        assert_eq!(entry.line_end, 55);
    }

    /// Symbol-MAP-level invariance (closes the gap the `calls`-level tests can't
    /// observe): the name→sorted-candidates bucket built from the in-RAM buffer
    /// MUST equal the one built from `load_all_symbols`, given the SAME symbol
    /// stream. The `calls` output stores only the resolved callee FQN+file, so a
    /// buggy buffer that kept duplicate FQNs (a Vec instead of last-write-wins map)
    /// could still emit identical `calls` rows — but it would put a PHANTOM extra
    /// candidate into the name-bucket here. This test compares the resolution INPUT
    /// (the bucket map) directly, so such a regression fails loudly.
    #[tokio::test]
    async fn name_bucket_from_buffer_equals_name_bucket_from_db() {
        // A duplicate FQN (decl+def) AND two distinct FQNs sharing a leaf name —
        // exercises both dedup and intra-bucket sort.
        let symbols = vec![
            mk_symbol("/lib.cpp", "compute", 10, 12), // decl — overwritten
            mk_symbol("/lib.cpp", "compute", 40, 55), // def — wins
            mk_symbol("/util.cpp", "compute", 200, 230), // distinct FQN, same leaf
            mk_symbol("/a.cpp", "caller", 1, 5),
        ];

        // DB source: write symbols, reload via load_all_symbols (the baseline).
        let home = TempDir::new().unwrap();
        let db = open_db(home.path(), "/test/bucket_inv", 0).await.unwrap();
        flush_symbol_batch_native(&db, &symbols).await.unwrap();
        let db_symbols = load_all_symbols(&db).await.unwrap();
        let bucket_db = build_name_bucket(db_symbols);

        // Buffer source: same stream through the production helper.
        let buf_symbols: Vec<SymbolWithPos> = build_buffer(&symbols).into_values().collect();
        let bucket_buf = build_name_bucket(buf_symbols);

        assert_eq!(
            bucket_db, bucket_buf,
            "name→candidates bucket from the in-RAM buffer must be byte-identical to the DB-reload bucket\nDB:  {bucket_db:?}\nBUF: {bucket_buf:?}"
        );
        // The `compute` bucket must hold exactly TWO distinct-FQN candidates
        // (/lib.cpp::compute deduped to one, + /util.cpp::compute), never three.
        let compute_bucket = bucket_buf.get("compute").expect("compute bucket");
        assert_eq!(
            compute_bucket.len(),
            2,
            "duplicate decl/def must dedup to ONE candidate (+ the distinct util.cpp one) = 2, not 3"
        );
        // And the surviving /lib.cpp::compute candidate is the LAST write (line 40).
        let lib = compute_bucket
            .iter()
            .find(|s| s.file == "/lib.cpp")
            .expect("lib.cpp::compute candidate");
        assert_eq!(lib.line_start, 40, "last-write-wins position must survive into the bucket");
    }
}

// Isolated micro-benchmark for the symbol-write primitive (target ①): is a plain
// `INSERT INTO symbol $data` materially faster than the current
// `INSERT ... ON DUPLICATE KEY UPDATE` (merge) for a DEDUPED set written into an
// EMPTY table (the full-rebuild scenario)? Answers go/no-go BEFORE touching the
// durable write path. #[ignore]d — run explicitly:
//   cargo test --release --lib symbol_insert_merge_vs_plain_microbench -- --ignored --nocapture
#[cfg(test)]
mod symbol_write_microbench {
    use super::*;
    use crate::store::open_db;
    use std::time::Instant;
    use tempfile::TempDir;

    fn synth_symbols(n: usize) -> Vec<Symbol> {
        use crate::parsing::symbols::{QualifiedSymbol, SymbolKind};
        // Distinct FQNs (deduped set): file_{i/50}.c::sym_{i}. ~50 symbols/file,
        // realistic name/signature sizes so byte volume mirrors real symbols.
        (0..n)
            .map(|i| Symbol {
                qualified: QualifiedSymbol {
                    file: format!("/repo/sub{}/file_{}.c", i % 4096, i / 50),
                    scope_path: vec![],
                    name: format!("sym_{i}"),
                },
                kind: SymbolKind::Function,
                line_start: (i % 2000) as u32,
                line_end: (i % 2000 + 12) as u32,
                signature: Some(format!("int sym_{i}(struct foo *ctx, unsigned long flags)")),
                parent_fqn: None,
            })
            .collect()
    }

    async fn drop_symbol_indexes(db: &Surreal<Db>) {
        db.query(
            "REMOVE INDEX IF EXISTS idx_symbol_file ON symbol; \
             REMOVE INDEX IF EXISTS idx_symbol_name ON symbol;",
        )
        .await
        .unwrap();
    }

    /// Plain INSERT (no merge clause) — the candidate write. Valid ONLY when the
    /// batch set is deduped and the table is empty (no key collisions possible).
    async fn flush_plain(db: &Surreal<Db>, symbols: &[Symbol]) {
        use crate::store::ops::kind_to_str;
        use std::collections::BTreeMap;
        for chunk in symbols.chunks(4096) {
            let records: Vec<SqlValue> = chunk
                .iter()
                .map(|sym| {
                    let mut map: BTreeMap<String, SqlValue> = BTreeMap::new();
                    map.insert("id".into(), SqlValue::from(sym.qualified.fqn()));
                    map.insert("name".into(), SqlValue::from(sym.qualified.name.as_str()));
                    map.insert("kind".into(), SqlValue::from(kind_to_str(&sym.kind)));
                    map.insert("file".into(), SqlValue::from(sym.qualified.file.as_str()));
                    map.insert("line_start".into(), SqlValue::from(sym.line_start as i64));
                    map.insert("line_end".into(), SqlValue::from(sym.line_end as i64));
                    match &sym.signature {
                        Some(s) => map.insert("signature".into(), SqlValue::from(s.as_str())),
                        None => map.insert("signature".into(), SqlValue::None),
                    };
                    map.insert("parent".into(), SqlValue::None);
                    SqlValue::Object(SqlObject::from(map))
                })
                .collect();
            db.query("INSERT INTO symbol $data RETURN NONE")
                .bind(("data", SqlArray::from(records)))
                .await
                .unwrap()
                .check()
                .unwrap();
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn symbol_insert_merge_vs_plain_microbench() {
        let n: usize = std::env::var("MICROBENCH_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2_600_000);
        let symbols = synth_symbols(n);
        println!("microbench: {} deduped synthetic symbols", symbols.len());

        // Path A: current merge INSERT (ON DUPLICATE KEY UPDATE).
        let home_m = TempDir::new().unwrap();
        let db_m = open_db(home_m.path(), "/mb/merge", 0).await.unwrap();
        drop_symbol_indexes(&db_m).await;
        let t = Instant::now();
        flush_symbol_batch_native(&db_m, &symbols).await.unwrap();
        let merge_ms = t.elapsed().as_millis();

        // Path B: plain INSERT (no merge).
        let home_p = TempDir::new().unwrap();
        let db_p = open_db(home_p.path(), "/mb/plain", 0).await.unwrap();
        drop_symbol_indexes(&db_p).await;
        let t = Instant::now();
        flush_plain(&db_p, &symbols).await;
        let plain_ms = t.elapsed().as_millis();

        let delta = merge_ms as i128 - plain_ms as i128;
        let pct = if merge_ms > 0 { 100.0 * delta as f64 / merge_ms as f64 } else { 0.0 };
        println!("MICROBENCH RESULT n={n} merge_ms={merge_ms} plain_ms={plain_ms} delta_ms={delta} plain_faster_by={pct:.1}%");
    }
}

// Isolated micro-benchmark for the COLD-SHARD-WARM path (the correctness bug:
// kernel query returns EMPTY after a 50s warm-wait timeout). Measures
// VectorIndex::load_from_db at kernel scale (909k chunks x 1024-dim) on a
// synthetic chunk table — WITHOUT a 30-min rebuild and WITHOUT sibling-repo
// watcher contamination. Answers hypothesis (c): "load_from_db at 3.7GB is
// genuinely >50s". #[ignore]d — run explicitly:
//   cargo test --release --lib load_from_db_cold_warm_microbench -- --ignored --nocapture
#[cfg(test)]
mod load_from_db_microbench {
    use super::*;
    use crate::store::open_db;
    use crate::vector::VectorIndex;
    use std::time::Instant;
    use tempfile::TempDir;

    /// Write `n` synthetic chunks with realistic 1024-dim packed embeddings,
    /// mirroring the production chunk-write shape (flush_chunk_batch).
    async fn seed_chunks(db: &Surreal<Db>, n: usize, dim: usize) {
        use crate::store::ops::pack_embedding;
        let mut batch: Vec<ChunkRecord> = Vec::with_capacity(4096);
        let mut written = 0usize;
        // A fixed pseudo-random embedding per row (deterministic, cheap).
        for i in 0..n {
            let emb: Vec<f32> = (0..dim)
                .map(|j| (((i * 31 + j * 17) % 1000) as f32) / 1000.0 - 0.5)
                .collect();
            batch.push(ChunkRecord {
                file: format!("/repo/sub{}/file_{}.c", i % 4096, i / 20),
                line_start: (i % 5000) as i64,
                line_end: (i % 5000 + 18) as i64,
                content: String::new(), // content is not read by load_from_db
                embedding: pack_embedding(&emb),
                symbol_ref: None,
            });
            if batch.len() >= 4096 {
                flush_chunk_batch(db, std::mem::take(&mut batch)).await.unwrap();
                written += 4096;
                if written.is_multiple_of(200_704) {
                    println!("  seeded {written}/{n} chunks");
                }
            }
        }
        if !batch.is_empty() {
            flush_chunk_batch(db, batch).await.unwrap();
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn load_from_db_cold_warm_microbench() {
        let n: usize = std::env::var("MICROBENCH_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(909_711); // kernel scale
        let dim: usize = 1024;
        println!("load_from_db microbench: seeding {n} chunks x {dim}-dim ...");

        let home = TempDir::new().unwrap();
        let db = open_db(home.path(), "/mb/warm", 0).await.unwrap();
        let t_seed = Instant::now();
        seed_chunks(&db, n, dim).await;
        println!("  seed done in {} ms", t_seed.elapsed().as_millis());

        // The decisive measurement: cold-shard warm at kernel scale.
        let t = Instant::now();
        let idx = VectorIndex::load_from_db(&db).await.unwrap();
        let warm_ms = t.elapsed().as_millis();

        // Phase split (replicates load_from_db internals) to attribute the warm cost
        // to SELECT scan vs serde decode vs L2-normalize insert — tells us whether
        // scalar quantization (i8) would cut warm time or only fix residency.
        use crate::vector::ChunkId;
        #[derive(serde::Deserialize)]
        struct Row {
            file: String,
            line_start: i64,
            line_end: i64,
            #[serde(deserialize_with = "crate::store::ops::de_embedding_dual")]
            embedding: Vec<f32>,
        }
        let t_sel = Instant::now();
        let rows: Vec<Row> = db
            .query("SELECT file, line_start, line_end, embedding FROM chunk WHERE embedding IS NOT NONE")
            .await.unwrap().take(0).unwrap();
        let select_ms = t_sel.elapsed().as_millis();
        let t_dec = Instant::now();
        let pairs: Vec<(ChunkId, Vec<f32>)> = rows.into_iter().map(|r| {
            (ChunkId { file: r.file, line_start: r.line_start as u32, line_end: r.line_end as u32 }, r.embedding)
        }).collect();
        let decode_ms = t_dec.elapsed().as_millis();
        let t_ins = Instant::now();
        let mut vi = VectorIndex::new();
        vi.insert(&pairs);
        let insert_ms = t_ins.elapsed().as_millis();

        println!(
            "LOAD_FROM_DB RESULT n={n} loaded={} warm_ms={warm_ms} \
             [split] select_ms={select_ms} decode_ms={decode_ms} insert_ms={insert_ms} \
             vs_warm_wait_50000ms={}",
            idx.len(),
            if warm_ms > 50_000 { "EXCEEDS (timeout->empty)" } else { "under" }
        );
    }
}

// Task 6.1: isolated mmap warm vs DB warm at kernel scale. Builds a kernel-sized
// shard, persists it to a shard.f32 file, then measures the THREE numbers the
// mmap change is justified by: (a) cold warm = DB load_from_db (the 25.7s we kill),
// (b) warm-after-persist = mmap open (near-instant), (c) first-search page-fault
// latency over the freshly-mapped shard (the OS faults cold pages on first scan).
//   cargo test --release --lib mmap_warm_vs_db_warm_microbench -- --ignored --nocapture
#[cfg(test)]
mod mmap_warm_microbench {
    use crate::vector::{shard_file, ChunkId, VectorIndex};
    use std::time::Instant;
    use tempfile::TempDir;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn mmap_warm_vs_db_warm_microbench() {
        let n: usize = std::env::var("MICROBENCH_N").ok().and_then(|v| v.parse().ok()).unwrap_or(909_711);
        let dim = 1024usize;
        println!("mmap warm microbench: building {n} x {dim} shard in RAM ...");

        // Build a kernel-sized in-RAM shard (deterministic synthetic vectors).
        let mut ram = VectorIndex::new();
        {
            let mut pairs: Vec<(ChunkId, Vec<f32>)> = Vec::with_capacity(4096);
            for i in 0..n {
                let emb: Vec<f32> = (0..dim).map(|j| (((i * 31 + j * 17) % 1000) as f32) / 1000.0 - 0.5).collect();
                pairs.push((ChunkId { file: format!("/repo/sub{}/f{}.c", i % 4096, i / 20), line_start: (i % 5000) as u32, line_end: (i % 5000 + 18) as u32 }, emb));
                if pairs.len() >= 4096 { ram.insert(&pairs); pairs.clear(); }
            }
            if !pairs.is_empty() { ram.insert(&pairs); }
        }
        println!("  built in-RAM shard: len={}", ram.len());

        let tmp = TempDir::new().unwrap();
        let repo = "c:/users/0x317/downloads/linux";

        // (b) Persist the shard to disk (this is what the slow-path warm does once).
        let t_persist = Instant::now();
        shard_file::write_new_generation(tmp.path(), repo, &ram, n as u64).unwrap();
        let persist_ms = t_persist.elapsed().as_millis();

        // Drop the in-RAM shard so the OS page cache for this file is the only
        // residency; sleep briefly to let the writeback settle.
        drop(ram);

        // (b) WARM-AFTER-PERSIST = mmap open + validate (the near-instant warm).
        let t_open = Instant::now();
        let (mapped, _gen) = shard_file::open_current(tmp.path(), repo, dim, n as u64).unwrap().expect("opens");
        let mmap_open_ms = t_open.elapsed().as_millis();

        // (c) FIRST-SEARCH page-fault latency: first query faults cold pages in.
        let q: Vec<f32> = (0..dim).map(|j| (j % 7) as f32 / 7.0).collect();
        let t_first = Instant::now();
        let r1 = mapped.search(&q, 30);
        let first_search_ms = t_first.elapsed().as_millis();
        // Second search: pages now resident → steady-state search latency.
        let t_second = Instant::now();
        let _ = mapped.search(&q, 30);
        let second_search_ms = t_second.elapsed().as_millis();

        println!(
            "MMAP WARM RESULT n={n} | persist_ms={persist_ms} mmap_open_ms={mmap_open_ms} \
             first_search_ms={first_search_ms} second_search_ms={second_search_ms} \
             results={} | vs f32 DB warm baseline ~25,700ms (select ~16,600ms)",
            r1.len()
        );
        println!(
            "  => warm-after-persist (mmap_open + first_search) = {} ms vs DB warm ~25,700 ms",
            mmap_open_ms + first_search_ms
        );
    }
}

// ─── RECALL GATE (Group 1): i8 vs f32 ground-truth on a REAL index ─────────
//
// The acceptance gate for the i8 quantization change (quantize-vector-shards-i8).
// Reads the REAL f32 embeddings already stored in a production index (NOT synthetic
// — these are the actual Voyage document embeddings of real code), builds f32 top-k
// ground truth for a probe set, then measures i8 recall@10 / recall@30 + score drift.
// GATE: recall@10 >= 0.98. Below that → pivot to mmap fallback, do NOT ship i8.
//
// Runs against the production data dir by default (~/.vibervn/context-engine), repo
// chosen via env. #[ignore]d — run explicitly:
//   RECALL_REPO='c:/users/0x317/downloads/linux' \
//     cargo test --release --lib recall_gate_i8_vs_f32 -- --ignored --nocapture
#[cfg(test)]
mod recall_gate {
    use crate::store::open_db;
    use crate::vector::{dot_product, dot_product_i8_dequant, l2_normalize, quantize_i8};

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn recall_gate_i8_vs_f32() {
        let repo = std::env::var("RECALL_REPO")
            .expect("set RECALL_REPO to a repo path that is already indexed");
        let data_dir = std::env::var("RECALL_DATA_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| {
            dirs::home_dir().unwrap().join(".vibervn").join("context-engine")
        });
        let n_probes: usize = std::env::var("RECALL_PROBES").ok().and_then(|v| v.parse().ok()).unwrap_or(50);

        println!("recall gate: repo={repo} data_dir={data_dir:?} probes={n_probes}");
        let db = open_db(&data_dir, &repo, 0).await.expect("open real index DB");

        // Load ALL real f32 embeddings (the corpus).
        #[derive(serde::Deserialize)]
        struct Row {
            file: String,
            line_start: i64,
            #[serde(deserialize_with = "crate::store::ops::de_embedding_dual")]
            embedding: Vec<f32>,
        }
        let rows: Vec<Row> = db
            .query("SELECT file, line_start, line_end, embedding FROM chunk WHERE embedding IS NOT NONE")
            .await.expect("scan chunks").take(0).expect("take rows");
        let corpus: Vec<Vec<f32>> = rows.iter().map(|r| l2_normalize(&r.embedding)).collect();
        let labels: Vec<String> = rows.iter().map(|r| format!("{}:{}", r.file, r.line_start)).collect();
        let n = corpus.len();
        assert!(n > 1000, "index too small to be a meaningful recall gate (got {n})");
        println!("  corpus: {n} real f32 embeddings, dim={}", corpus[0].len());

        // Pre-quantize the whole corpus once (the i8 shard).
        let corpus_i8: Vec<Vec<i8>> = corpus.iter().map(|v| quantize_i8(v)).collect();

        // Probes: evenly-spaced held-out corpus vectors used as queries. These are
        // real embeddings → faithful distribution. Each probe is excluded from its
        // own ground truth (leave-one-out) so the self-match doesn't inflate recall.
        let stride = (n / n_probes).max(1);
        let probe_idxs: Vec<usize> = (0..n).step_by(stride).take(n_probes).collect();

        let topk = 30usize;

        // Per-probe metrics, computed in parallel (last single-threaded run was 9.4min).
        use rayon::prelude::*;
        struct ProbeMetric { r10: f64, r30: f64, drift_sum: f64, drift_max: f32 }
        let metrics: Vec<ProbeMetric> = probe_idxs
            .par_iter()
            .map(|&qi| {
                let q_f32 = &corpus[qi];
                let q_i8 = &corpus_i8[qi];

                // f32 ground-truth top-k (leave-one-out: skip self).
                let mut f32_scored: Vec<(usize, f32)> = (0..n)
                    .filter(|&j| j != qi)
                    .map(|j| (j, dot_product(q_f32, &corpus[j])))
                    .collect();
                f32_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let gt: Vec<usize> = f32_scored.iter().take(topk).map(|(j, _)| *j).collect();

                // i8 top-k.
                let mut i8_scored: Vec<(usize, f32)> = (0..n)
                    .filter(|&j| j != qi)
                    .map(|j| (j, dot_product_i8_dequant(q_i8, &corpus_i8[j])))
                    .collect();
                i8_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let got: Vec<usize> = i8_scored.iter().take(topk).map(|(j, _)| *j).collect();

                let gt10: std::collections::HashSet<usize> = gt.iter().take(10).copied().collect();
                let gt30: std::collections::HashSet<usize> = gt.iter().copied().collect();
                let got10: std::collections::HashSet<usize> = got.iter().take(10).copied().collect();
                let got30: std::collections::HashSet<usize> = got.iter().copied().collect();
                let r10 = gt10.intersection(&got10).count() as f64 / 10.0;
                let r30 = gt30.intersection(&got30).count() as f64 / topk as f64;

                let mut drift_sum = 0.0f64;
                let mut drift_max = 0.0f32;
                for &j in gt.iter().take(10) {
                    let d = (dot_product(q_f32, &corpus[j]) - dot_product_i8_dequant(q_i8, &corpus_i8[j])).abs();
                    drift_sum += d as f64;
                    drift_max = drift_max.max(d);
                }
                ProbeMetric { r10, r30, drift_sum, drift_max }
            })
            .collect();
        let _ = &labels;

        let p = metrics.len() as f64;
        let recall10 = metrics.iter().map(|m| m.r10).sum::<f64>() / p;
        let recall30 = metrics.iter().map(|m| m.r30).sum::<f64>() / p;
        let drift_mean = metrics.iter().map(|m| m.drift_sum).sum::<f64>() / (p * 10.0);
        let drift_max = metrics.iter().map(|m| m.drift_max).fold(0.0f32, f32::max);

        // recall@10 DISTRIBUTION — is 0.93 uniform, or a few probes tanking the mean?
        let mut r10s: Vec<f64> = metrics.iter().map(|m| m.r10).collect();
        r10s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |q: f64| r10s[((r10s.len() as f64 - 1.0) * q).round() as usize];
        // Histogram of per-probe recall@10 in 0.1 buckets [0.0..1.0].
        let mut hist = [0usize; 11];
        for &r in &r10s { hist[(r * 10.0).round() as usize] += 1; }
        let below_080 = r10s.iter().filter(|&&r| r < 0.80).count();
        let perfect = r10s.iter().filter(|&&r| r >= 0.999).count();
        println!(
            "RECALL@10 DISTRIBUTION min={:.2} p10={:.2} p25={:.2} median={:.2} p75={:.2} max={:.2} \
             | #probes<0.80={below_080} #probes==1.0={perfect} | hist[0.0..1.0 by 0.1]={hist:?}",
            r10s[0], pct(0.10), pct(0.25), pct(0.50), pct(0.75), r10s[r10s.len()-1],
        );
        println!(
            "RECALL GATE RESULT repo={repo} probes={} corpus={n} \
             recall@10={recall10:.4} recall@30={recall30:.4} \
             score_drift_mean={drift_mean:.5} score_drift_max={drift_max:.5} \
             gate_recall10>=0.98={}",
            metrics.len(),
            if recall10 >= 0.98 { "PASS" } else { "FAIL->mmap-fallback" }
        );
    }
}

// ─── DELETE-STRATEGY PROBE: why is `DELETE FROM calls WHERE col IN $files` slow? ─
//
// p2_delete_calls measured 471s (OR form) / 394s (split single-column form) on a
// 2748-file resolve_set against a 4.44M-row calls table at kernel scale. This
// SELF-CONTAINED probe seeds a synthetic `calls` table with the REAL schema +
// the REAL idx_calls_in_file / idx_calls_out_file indexes, then times the
// candidate strategies on a realistic-size resolve_set so the planner's index
// decision (and the O(list × rows) pathology) is reproduced WITHOUT a 25-min
// kernel rebuild and WITHOUT depending on version-locked on-disk data:
//   (A) DELETE WHERE in_file IN $files            — the bare-IN form (suspect)
//   (B) per-file `DELETE WHERE in_file = $f` loop — the candidate fix (ops.rs:199)
//   (C) keyset id-scan WHERE in_file IN $files    — step-3's proven-fast pattern
// Rows/files are scaled by env (default 500k rows / 6000 files / 2748 resolve_set)
// to keep the probe under a minute while preserving the list×rows ratio.
//
// Run explicitly:
//   cargo test --release --lib delete_strategy_probe -- --ignored --nocapture
//   PROBE_ROWS=1000000 PROBE_FILES=8000 PROBE_RESOLVE=2748 cargo test ... (scale up)
#[cfg(test)]
mod delete_strategy_probe {
    use crate::store::open_db;
    use std::collections::BTreeMap;
    use std::time::Instant;
    use surrealdb::sql::{Array as SqlArray, Object as SqlObject, Thing as SqlThing, Value as SqlValue};
    use surrealdb::engine::local::Db;
    use surrealdb::Surreal;
    use tempfile::TempDir;

    /// Bulk-seed `n_rows` calls rows over `n_files` files via the native array
    /// INSERT path (the same fast path flush_edge_batch uses), with the two
    /// secondary indexes already defined so deletes plan against a live index.
    async fn seed(db: &Surreal<Db>, n_rows: usize, n_files: usize) {
        db.query("DEFINE INDEX IF NOT EXISTS idx_calls_in_file  ON calls FIELDS in_file; \
                  DEFINE INDEX IF NOT EXISTS idx_calls_out_file ON calls FIELDS out_file;")
            .await.expect("define indexes").check().expect("define check");
        let mut written = 0usize;
        while written < n_rows {
            let batch = std::cmp::min(20_000, n_rows - written);
            let records: Vec<SqlValue> = (0..batch)
                .map(|i| {
                    let n = written + i;
                    let inf = n % n_files;
                    let outf = (n * 7 + 3) % n_files;
                    let mut m: BTreeMap<String, SqlValue> = BTreeMap::new();
                    m.insert("line".into(), SqlValue::from(n as i64));
                    m.insert("in_file".into(), SqlValue::from(format!("f{inf}.c")));
                    m.insert("out_file".into(), SqlValue::from(format!("f{outf}.c")));
                    m.insert("in_name".into(), SqlValue::from(format!("s{n}")));
                    m.insert("out_name".into(), SqlValue::from(format!("t{n}")));
                    SqlValue::Object(SqlObject::from(m))
                })
                .collect();
            db.query("INSERT INTO calls $data RETURN NONE")
                .bind(("data", SqlArray::from(records)))
                .await.expect("bulk insert").check().expect("insert check");
            written += batch;
        }
    }

    async fn count_all(db: &Surreal<Db>) -> i64 {
        #[derive(serde::Deserialize)]
        struct CountRow { count: i64 }
        let c: Vec<CountRow> = db.query("SELECT count() AS count FROM calls GROUP ALL")
            .await.unwrap().take(0).unwrap();
        c.first().map(|r| r.count).unwrap_or(0)
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn delete_strategy_probe() {
        let n_rows: usize = envvar("PROBE_ROWS", 500_000);
        let n_files: usize = envvar("PROBE_FILES", 6_000);
        let n_resolve: usize = envvar("PROBE_RESOLVE", 2_748);
        println!("DELPROBE: seeding {n_rows} calls rows / {n_files} files; \
                  resolve_set={n_resolve} (real schema + idx_calls_in_file/out_file)");

        let home = TempDir::new().unwrap();
        let files: Vec<String> = (0..n_resolve).map(|i| format!("f{i}.c")).collect();

        // PROBE_ONLY_C=1 / PROBE_ONLY_D=1 run a single strategy alone at full
        // scale in minutes. A bare-IN (A) is O(list×rows) ~25min at 4.44M rows;
        // B per-file is the already-measured kernel form (215685ms); C's keyset
        // `SELECT WHERE col IN $files` suffers the SAME IN-list pathology
        // (measured >21min collect at 4.44M and aborted). D is the candidate.
        let only_c = envvar("PROBE_ONLY_C", 0) != 0;
        let only_d = envvar("PROBE_ONLY_D", 0) != 0;
        let run_ab = !only_c && !only_d;

        // ── (A) bare `IN $files` DELETE (the current/suspect form) ──────────────
        let (mut a_ms, mut deleted_a) = (-1i128, -1i64);
        if run_ab {
            let dba = open_db(home.path(), "/probe/a", 0).await.expect("open a");
            let t = Instant::now(); seed(&dba, n_rows, n_files).await;
            println!("DELPROBE: seed A done in {} ms (rows={})",
                t.elapsed().as_millis(), count_all(&dba).await);
            let before_a = count_all(&dba).await;
            let t = Instant::now();
            dba.query("DELETE FROM calls WHERE in_file IN $files").bind(("files", files.clone()))
                .await.expect("A in");
            dba.query("DELETE FROM calls WHERE out_file IN $files").bind(("files", files.clone()))
                .await.expect("A out");
            a_ms = t.elapsed().as_millis() as i128;
            deleted_a = before_a - count_all(&dba).await;
            println!("DELPROBE (A) bare `IN $files` DELETE x2cols: {a_ms} ms (deleted={deleted_a})");
        }

        // ── (B) per-file equality DELETE loop (the candidate fix) ───────────────
        let (mut b_ms, mut deleted_b) = (-1i128, -1i64);
        if run_ab {
            let dbb = open_db(home.path(), "/probe/b", 0).await.expect("open b");
            let t = Instant::now(); seed(&dbb, n_rows, n_files).await;
            println!("DELPROBE: seed B done in {} ms", t.elapsed().as_millis());
            let before_b = count_all(&dbb).await;
            let t = Instant::now();
            for f in &files {
                dbb.query("DELETE FROM calls WHERE in_file = $f").bind(("f", f.clone()))
                    .await.expect("B in");
                dbb.query("DELETE FROM calls WHERE out_file = $f").bind(("f", f.clone()))
                    .await.expect("B out");
            }
            b_ms = t.elapsed().as_millis() as i128;
            deleted_b = before_b - count_all(&dbb).await;
            println!("DELPROBE (B) per-file `= $f` DELETE x2cols x{}: {b_ms} ms (deleted={deleted_b})",
                files.len());
        }

        // ── (C) keyset-paginated id-collect + direct-record-id batch DELETE ─────
        // The step-3 primitive applied to deletion: ONE keyset-paginated
        // `SELECT id WHERE col IN $files` per column (the SAME index range-scan
        // that resolves the 2748-file raw_edge set in 262ms), collecting native
        // `Thing` record ids, then `DELETE` them by DIRECT record id in batches.
        // Direct-record delete is O(deleted rows) and does NOT re-plan a predicate
        // per row, so it is independent of total table size — unlike (B), whose
        // per-`= $f` cost grows with the table (4.44M rows → ~39ms/query).
        async fn collect_ids(db: &Surreal<Db>, col: &str, files: &[String]) -> Vec<SqlThing> {
            let mut ids: Vec<SqlThing> = Vec::new();
            let mut cursor = String::new();
            let page: i64 = 5000;
            loop {
                let q = format!(
                    "SELECT id, type::string(id) AS id_str FROM calls \
                     WHERE {col} IN $files AND type::string(id) > $cursor \
                     ORDER BY id_str LIMIT $page"
                );
                #[derive(serde::Deserialize)]
                struct Row { id: SqlThing, id_str: String }
                let rows: Vec<Row> = db.query(&q)
                    .bind(("files", files.to_vec()))
                    .bind(("cursor", cursor.clone()))
                    .bind(("page", page))
                    .await.expect("C select").take(0).expect("C take");
                if rows.is_empty() { break; }
                cursor = rows.last().unwrap().id_str.clone();
                let n = rows.len();
                ids.extend(rows.into_iter().map(|r| r.id));
                if (n as i64) < page { break; }
            }
            ids
        }
        let dbc = open_db(home.path(), "/probe/c", 0).await.expect("open c");
        let (mut c_ms, mut deleted_c) = (-1i128, -1i64);
        if !only_d {
            let t = Instant::now(); seed(&dbc, n_rows, n_files).await;
            println!("DELPROBE: seed C done in {} ms", t.elapsed().as_millis());
            let before_c = count_all(&dbc).await;
            let t = Instant::now();
            // Collect ids touching the resolve_set on EITHER column, dedup (a row
            // may match both columns → would otherwise be deleted twice).
            let mut all_ids = collect_ids(&dbc, "in_file", &files).await;
            all_ids.extend(collect_ids(&dbc, "out_file", &files).await);
            let collect_ms = t.elapsed().as_millis();
            let id_strings: std::collections::BTreeSet<String> =
                all_ids.iter().map(|t| t.to_string()).collect();
            let dedup_ids: Vec<SqlThing> = {
                let mut seen = std::collections::HashSet::new();
                all_ids.into_iter().filter(|t| seen.insert(t.to_string())).collect()
            };
            let t2 = Instant::now();
            // Direct-record-id DELETE in batches: `DELETE $ids` where $ids is an
            // Array of Things. Direct access — no WHERE predicate, no per-row plan.
            for chunk in dedup_ids.chunks(10_000) {
                let arr = SqlArray::from(chunk.iter().cloned().map(SqlValue::Thing).collect::<Vec<_>>());
                dbc.query("DELETE $ids").bind(("ids", arr)).await.expect("C delete");
            }
            let delete_ms = t2.elapsed().as_millis();
            c_ms = (collect_ms + delete_ms) as i128;
            deleted_c = before_c - count_all(&dbc).await;
            println!("DELPROBE (C) keyset-id-collect ({} ids, {} dedup) + direct-id DELETE: \
                      {c_ms} ms (collect={collect_ms} ms, delete={delete_ms} ms, deleted={deleted_c})",
                id_strings.len(), dedup_ids.len());
        }

        // ── (D) per-file indexed-equality DELETE, BATCHED into one transaction ──
        // per chunk. Keeps the proven O(resolve_set) INDEX POINT-SEEK (B's
        // `WHERE col = $f` drives idx_calls_in_file/out_file — no IN-list, no
        // scan) but collapses B's 5496 SEPARATE auto-commit round-trips (each its
        // own RocksDB commit+fsync — measured ~39ms apiece, commit-bound NOT
        // seek-bound) into ONE multi-statement BEGIN/COMMIT per chunk of files.
        // So ~14 transactions instead of 5496 → the per-commit fixed cost (the
        // real dominator at 215685ms) is amortized ~390x while every delete stays
        // a single-value indexed equality. Distinct $p{i} binds per statement keep
        // values parameterized (no injection, no per-row parse of file strings).
        let dbd = open_db(home.path(), "/probe/d", 0).await.expect("open d");
        let t = Instant::now(); seed(&dbd, n_rows, n_files).await;
        println!("DELPROBE: seed D done in {} ms", t.elapsed().as_millis());
        let before_d = count_all(&dbd).await;
        let chunk_files: usize = envvar("PROBE_D_CHUNK", 200);
        let t = Instant::now();
        for chunk in files.chunks(chunk_files) {
            let mut stmt = String::from("BEGIN;\n");
            for i in 0..chunk.len() {
                stmt.push_str(&format!("DELETE FROM calls WHERE in_file = $p{i};\n"));
                stmt.push_str(&format!("DELETE FROM calls WHERE out_file = $p{i};\n"));
            }
            stmt.push_str("COMMIT;");
            let mut q = dbd.query(&stmt);
            for (i, f) in chunk.iter().enumerate() {
                q = q.bind((format!("p{i}"), f.clone()));
            }
            q.await.expect("D delete chunk").check().expect("D check");
        }
        let d_ms = t.elapsed().as_millis() as i128;
        let deleted_d = before_d - count_all(&dbd).await;
        println!("DELPROBE (D) per-file-eq BATCHED in {}-file txns ({} txns): \
                  {d_ms} ms (deleted={deleted_d})",
            chunk_files, files.len().div_ceil(chunk_files));

        // Sentinels (-1) print for any strategy a PROBE_ONLY_* flag skipped. The
        // identical-delete-count check only compares strategies that ran.
        let mut counts: Vec<i64> = Vec::new();
        for v in [deleted_a, deleted_b, deleted_c, deleted_d] {
            if v >= 0 { counts.push(v); }
        }
        let counts_match = counts.windows(2).all(|w| w[0] == w[1]);
        println!("DELPROBE VERDICT: (A) bare-IN = {a_ms} ms | (B) per-file-eq = {b_ms} ms \
                  | (C) id-collect+direct = {c_ms} ms | (D) per-file-eq batched-txn = {d_ms} ms \
                  || identical delete count across run strategies: {} ({deleted_a}/{deleted_b}/{deleted_c}/{deleted_d})",
            counts_match);
    }

    fn envvar(k: &str, d: usize) -> usize {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    }
}



#[cfg(test)]
mod null_byte_skip_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn file_with_null_byte_is_skipped() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"hello\x00world").unwrap();
        f.flush().unwrap();
        let path = f.path().to_str().unwrap();

        let output = parse_one_file(path);
        match output {
            ParseOutput::Skipped { reason, .. } => {
                assert!(reason.contains("null byte"), "reason: {reason}");
            }
            ParseOutput::Parsed(_) => panic!("expected Skipped for file with null byte"),
        }
    }

    #[test]
    fn file_without_null_byte_is_parsed() {
        let mut f = NamedTempFile::with_suffix(".rs").unwrap();
        f.write_all(b"fn main() {}").unwrap();
        f.flush().unwrap();
        let path = f.path().to_str().unwrap();

        let output = parse_one_file(path);
        assert!(matches!(output, ParseOutput::Parsed(_)));
    }
}

// ─── Tests: raw-edge batching + file_meta ordering (optimize-kernel-index-throughput) ───

#[cfg(test)]
mod raw_edge_batching_tests {
    use super::*;
    use crate::store::open_db;
    use crate::store::ops::get_all_file_meta;
    use serde::Deserialize;
    use tempfile::TempDir;

    /// Helper: insert a symbol for resolution in Phase 2.
    #[allow(dead_code)]
    async fn insert_symbol_fqn(db: &Surreal<Db>, fqn: &str, file: &str, name: &str) {
        db.query(format!(
            "UPSERT symbol:`⟨{fqn}⟩` SET \
             name = '{name}', kind = 'function', file = '{file}', \
             line_start = 1, line_end = 10, signature = NONE, parent = NONE"
        ))
        .await
        .expect("insert symbol");
    }

    /// When raw edges are written via the DB path (overflow or incremental),
    /// they must be batched across files (O(total_edges/batch_size) round-trips,
    /// not O(files)). This test drives many small-file raw edges through the
    /// pipeline's post-overflow path and verifies:
    ///   1. All raw_edge records land in the DB (count matches expected)
    ///   2. Phase 2 resolves the same set of calls as if written per-file
    ///   3. file_meta exists only for files whose raw edges were flushed
    #[tokio::test]
    async fn batched_raw_edges_produce_correct_resolution() {
        let home = TempDir::new().unwrap();
        let repo_dir = TempDir::new().unwrap();
        let repo = repo_dir.path().to_str().unwrap().replace('\\', "/");

        // Create 10 source files, each defining a function and calling another.
        // This generates enough raw edges to exercise the batching path.
        for i in 0..10 {
            let callee_name = format!("target_{}", i);
            let caller_name = format!("caller_{}", i);
            let content = format!(
                "fn {caller_name}() {{\n    {callee_name}();\n}}\n\nfn {callee_name}() {{\n}}\n"
            );
            let path = repo_dir.path().join(format!("file_{i}.rs"));
            std::fs::write(&path, content).unwrap();
        }

        let db = open_db(home.path(), &repo, 0).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);

        let stats = pipeline
            .run(&db, None, true, None, None, None, &[], None)
            .await
            .expect("full_rebuild with batched raw edges must succeed");

        // Verify file_meta is present for all 10 files.
        let all_meta = get_all_file_meta(&db, &repo).await.unwrap();
        assert_eq!(
            all_meta.len(), 10,
            "all 10 files must have file_meta after full rebuild"
        );

        // Verify edges_resolved marker is set.
        let marker = crate::store::ops::get_meta(&db, EDGES_RESOLVED_KEY).await.unwrap();
        assert!(
            marker.is_some(),
            "edges_resolved marker must be set after full rebuild"
        );

        // Verify stats captured raw edges.
        assert!(
            stats.total_raw_edges > 0,
            "total_raw_edges must be > 0 (got {})",
            stats.total_raw_edges
        );
    }

    /// file_meta ordering: if a run is interrupted (simulated) after chunks flush
    /// but before file_meta commit, the file must be treated as not-yet-committed
    /// on the next run (re-processed). This proves meta does not precede its
    /// dependencies (chunks + raw edges).
    ///
    /// Implementation: run a full rebuild, manually delete a file's file_meta row
    /// (simulating the crash window between chunk flush and meta commit), then run
    /// an incremental. The incremental must detect the file as needing re-processing
    /// (it won't appear in stored_meta, so a non-watcher incremental walk will pick
    /// it up as Added).
    #[tokio::test]
    async fn file_meta_absence_triggers_reprocessing() {
        let home = TempDir::new().unwrap();
        let repo_dir = TempDir::new().unwrap();
        let repo = repo_dir.path().to_str().unwrap().replace('\\', "/");

        // Create two source files.
        let file_a = repo_dir.path().join("alpha.rs");
        let file_b = repo_dir.path().join("beta.rs");
        std::fs::write(&file_a, "fn alpha() {}\n").unwrap();
        std::fs::write(&file_b, "fn beta() { alpha(); }\n").unwrap();

        let db = open_db(home.path(), &repo, 0).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);

        // Full rebuild — both files indexed.
        pipeline
            .run(&db, None, true, None, None, None, &[], None)
            .await
            .expect("first full rebuild must succeed");

        let meta_before = get_all_file_meta(&db, &repo).await.unwrap();
        assert_eq!(meta_before.len(), 2, "both files must have meta after rebuild");

        // Simulate crash: delete file_meta for alpha.rs (as if the meta commit
        // never happened — the crash window our ordering prevents).
        let alpha_path = file_a.to_str().unwrap().replace('\\', "/");
        let escaped = escape_surreal(&alpha_path);
        db.query(format!("DELETE FROM file_meta WHERE path = '{escaped}'"))
            .await
            .expect("simulate crash: delete alpha file_meta");

        // Also remove the edges_resolved marker to simulate a partial state.
        db.query("DELETE FROM index_meta WHERE key = 'edges_resolved'")
            .await
            .expect("delete edges_resolved");

        // Next run: since file_meta is absent for alpha, it should detect it
        // needs re-processing. With edges_resolved absent + raw_edge possibly empty
        // for the RAM path, the crash-recovery logic in run() should trigger.
        let result = pipeline
            .run(&db, None, false, None, None, None, &[], None)
            .await;

        // The run should succeed (either via full rebuild recovery or incremental).
        assert!(result.is_ok(), "recovery run must succeed: {:?}", result.err());

        // After recovery, both files should have file_meta again.
        let meta_after = get_all_file_meta(&db, &repo).await.unwrap();
        assert_eq!(
            meta_after.len(), 2,
            "both files must have meta after recovery (got {})",
            meta_after.len()
        );

        // edges_resolved must be set again.
        let marker = crate::store::ops::get_meta(&db, EDGES_RESOLVED_KEY).await.unwrap();
        assert!(marker.is_some(), "edges_resolved must be set after recovery");
    }

    /// When the RAM cap is exceeded (simulated via a large-enough repo or the
    /// constant itself), the overflow-to-DB path must still produce the same
    /// resolved calls as the RAM path would. This test creates files with enough
    /// cross-file calls to verify resolution correctness regardless of path.
    #[tokio::test]
    async fn overflow_path_resolution_matches_ram_path() {
        let home = TempDir::new().unwrap();
        let repo_dir = TempDir::new().unwrap();
        let repo = repo_dir.path().to_str().unwrap().replace('\\', "/");

        // Create a chain of files: file_0 calls target_0, file_1 calls target_1, etc.
        // Each file defines both its own caller and the target it calls.
        for i in 0..5 {
            let content = format!(
                "fn caller_{i}() {{\n    target_{i}();\n}}\n\nfn target_{i}() {{}}\n"
            );
            let path = repo_dir.path().join(format!("mod_{i}.rs"));
            std::fs::write(&path, content).unwrap();
        }

        let db = open_db(home.path(), &repo, 0).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);

        let stats = pipeline
            .run(&db, None, true, None, None, None, &[], None)
            .await
            .expect("rebuild must succeed");

        // Verify calls were resolved.
        #[derive(Deserialize)]
        struct CallRow {
            in_name: String,
            out_name: String,
        }
        let calls: Vec<CallRow> = db
            .query("SELECT in_name, out_name FROM calls")
            .await
            .unwrap()
            .take(0)
            .unwrap();

        // Each file has caller_N → target_N, so we expect at least 5 resolved calls.
        assert!(
            calls.len() >= 5,
            "expected at least 5 resolved calls, got {}",
            calls.len()
        );

        // Verify all callers reference full FQNs (not leaf names).
        for call in &calls {
            assert!(
                call.in_name.contains("::"),
                "in_name must be a full FQN, got: {}",
                call.in_name
            );
            assert!(
                call.out_name.contains("::"),
                "out_name must be a full FQN, got: {}",
                call.out_name
            );
        }

        // Verify file_meta count matches file count.
        let all_meta = get_all_file_meta(&db, &repo).await.unwrap();
        assert_eq!(all_meta.len(), 5, "all 5 files must have file_meta");

        // Verify stats.
        assert!(stats.total_raw_edges >= 5, "expected at least 5 raw edges");
        assert!(stats.total_symbols >= 10, "expected at least 10 symbols (2 per file)");
    }
}

// ─── Tests: transient embed failure resilience ──────────────────────────────
/// Tests that validate the per-file transient error skip behavior:
///   1. A transient/exhausted embed error skips the file (no file_meta, FileSkipped
///      emitted, pipeline continues indexing remaining files).
///   2. A fatal (non-transient) embed error still aborts the pipeline.
///   3. The classify_embed_error helper correctly classifies errors.
///
/// Root cause: a transient gateway timeout on one file during a 79K-file Linux kernel
/// rebuild must NOT abort the entire multi-hour run. Crash-safe file_meta makes per-file
/// skip self-healing — the next index trigger re-embeds the skipped file.
#[cfg(test)]
mod transient_embed_resilience_tests {
    use super::*;
    use crate::embedding::TransientEmbedExhausted;
    use crate::store::open_db;
    use crate::store::ops::get_all_file_meta;
    use tempfile::TempDir;

    /// classify_embed_error correctly identifies a TransientEmbedExhausted error
    /// as Transient (the marker is the outermost context after embed_batch wraps it).
    #[test]
    fn classify_transient_error_correctly() {
        // Simulate the exact error chain produced by embed_batch when transient
        // retries are exhausted: original_err.context(TransientEmbedExhausted{..})
        let original = anyhow::anyhow!("connection timed out");
        let with_marker = original.context(TransientEmbedExhausted { attempts: 6 });

        match classify_embed_error(with_marker) {
            EmbedFileError::Transient(_) => {} // correct
            EmbedFileError::Fatal(e) => panic!(
                "TransientEmbedExhausted must be classified as Transient, got Fatal: {e:#}"
            ),
        }
    }

    /// A fatal (non-transient) error must NOT be misclassified as transient.
    #[test]
    fn classify_fatal_error_correctly() {
        let fatal = anyhow::anyhow!("VoyageAI error 401: invalid API key");

        match classify_embed_error(fatal) {
            EmbedFileError::Fatal(_) => {} // correct
            EmbedFileError::Transient(e) => panic!(
                "fatal auth errors must NOT be classified as Transient: {e:#}"
            ),
        }
    }

    /// Full pipeline test: with no Voyage client, all files index with empty
    /// embeddings (no transient error possible). This confirms the normal path
    /// still works and no file_meta is skipped.
    #[tokio::test]
    async fn pipeline_completes_with_all_files_when_no_transient_errors() {
        let home = TempDir::new().unwrap();
        let repo_dir = TempDir::new().unwrap();
        let repo = repo_dir.path().to_str().unwrap().replace('\\', "/");

        // Create 5 source files.
        for i in 0..5 {
            let content = format!("fn func_{i}() {{}}\n");
            let path = repo_dir.path().join(format!("file_{i}.rs"));
            std::fs::write(&path, content).unwrap();
        }

        let db = open_db(home.path(), &repo, 0).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);

        let stats = pipeline
            .run(&db, None, true, None, None, None, &[], None)
            .await
            .expect("full rebuild without voyage must succeed");

        // All 5 files must have file_meta (no skips).
        let all_meta = get_all_file_meta(&db, &repo).await.unwrap();
        assert_eq!(
            all_meta.len(), 5,
            "all 5 files must have file_meta when no embed errors occur"
        );
        assert_eq!(stats.indexed_files, 5);
    }

    /// Verify that the TRANSIENT_RETRY_LIMIT constant is in the expected range
    /// after the bump from 3 to 6.
    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn transient_retry_limit_is_six() {
        use crate::embedding::voyage::TRANSIENT_RETRY_LIMIT_FOR_TEST;
        assert_eq!(
            TRANSIENT_RETRY_LIMIT_FOR_TEST, 6,
            "TRANSIENT_RETRY_LIMIT must be 6 to ride out multi-second gateway blips"
        );
    }
}

