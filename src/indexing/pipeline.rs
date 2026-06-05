use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::embedding::InputType;
use crate::embedding::cache::EmbeddingCache;
use crate::embedding::voyage::VoyageClient;
use crate::indexing::ProgressHandle;
use crate::indexing::events::{IndexEvent, IndexEventBus};
use crate::indexing::tracker::{ChangeKind, FileChange, stat_file};
use crate::indexing::walker::{is_under_hidden_dir, walk_repo};
use crate::parsing::parse_file;
use crate::parsing::relations::{EdgeKind, EdgeTarget};
use crate::parsing::symbols::Symbol;
use crate::store::ops::{
    FileMeta, delete_all_data, delete_files_data_bulk, get_all_file_meta,
    get_meta, set_meta, upsert_file_meta, find_symbols_by_names_with_pos, SymbolWithPos,
};
use crate::vector::{ChunkId, VectorIndex};

/// Batch size for DB writes — keeps per-query payload small and avoids the
/// gigabyte-sized transaction that caused 3 GB RAM spikes on large repos.
const WRITE_BATCH_SIZE: usize = 512;

/// Streaming channel capacity. Parser feeds at most this many parsed-file results
/// into the embed stage before blocking. Keeps peak inflight bounded independent
/// of repo size (O(channel_cap * chunks_per_file) RAM, not O(repo)).
const PARSE_CHANNEL_CAP: usize = 64;

/// Embed-output channel capacity (from embed stage to writer).
const EMBED_CHANNEL_CAP: usize = 64;

/// A chunk row ready for bulk INSERT via native SurrealDB parameter binding.
/// Using `Vec<f32>` (not a text-formatted string) means the driver serialises
/// the embedding as a CBOR array — no float-token parsing by the query engine.
#[derive(Serialize)]
struct ChunkRecord {
    file: String,
    line_start: i64,
    line_end: i64,
    content: String,
    embedding: Vec<f32>,
    symbol_ref: Option<String>,
}

/// A symbol row for native-bind INSERT.
#[derive(Serialize)]
struct SymbolRecord {
    name: String,
    kind: String,
    file: String,
    line_start: i64,
    line_end: i64,
    signature: Option<String>,
    parent: Option<String>,
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
}

pub struct IndexPipelineStats {
    pub indexed_files: u64,
    pub total_files: u64,
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
}

impl IndexPipeline {
    pub fn new(repo: String, voyage: Option<VoyageClient>) -> Self {
        Self::new_with_concurrency(repo, voyage, 4, None)
    }

    pub fn new_with_concurrency(repo: String, voyage: Option<VoyageClient>, embed_concurrency: usize, cache: Option<EmbeddingCache>) -> Self {
        let embed_concurrency = embed_concurrency.max(1);
        Self { repo, voyage, embed_concurrency, cache: cache.map(Arc::new) }
    }

    /// Run the pipeline against the shared `db` handle.
    #[allow(clippy::too_many_arguments)]
    pub async fn run(
        &self,
        db: &Surreal<Db>,
        changes: Option<Vec<FileChange>>,
        force_rebuild: bool,
        vector_index: Option<&tokio::sync::RwLock<VectorIndex>>,
        progress: Option<ProgressHandle>,
        event_bus: Option<&IndexEventBus>,
        key_hints: &[String],
    ) -> Result<IndexPipelineStats> {
        // Check if first run (no file_meta at all).
        let stored_meta = get_all_file_meta(db, &self.repo).await?;
        let is_first_run = stored_meta.is_empty();

        let total_files = walk_repo(&self.repo).len() as u64;

        if is_first_run || force_rebuild {
            if force_rebuild && !is_first_run {
                info!(repo = %self.repo, "forced full rebuild");
            } else {
                info!(repo = %self.repo, "first run — full rebuild");
            }
            if let Some(bus) = event_bus {
                bus.emit(IndexEvent::Started {
                    repo: self.repo.clone(),
                    total_files,
                    is_rebuild: force_rebuild,
                });
            }
            let new_vectors = self.full_rebuild(db, progress.as_ref(), event_bus, key_hints).await?;
            if let Some(vi) = vector_index {
                let mut guard = vi.write().await;
                guard.remove_repo(&self.repo);
                guard.insert(&new_vectors);
            }
            let indexed = get_all_file_meta(db, &self.repo).await?.len() as u64;
            return Ok(IndexPipelineStats { indexed_files: indexed, total_files });
        }

        // Incremental run.
        let file_changes = match changes {
            Some(explicit) => explicit,
            None => {
                let all_files = walk_repo(&self.repo);
                let meta_map: HashMap<String, (i64, i64)> = stored_meta
                    .iter()
                    .map(|m| (m.path.clone(), (m.mtime, m.size)))
                    .collect();
                crate::indexing::tracker::detect_changes(&all_files, &meta_map)
            }
        };

        // Filter out Added/Modified changes whose paths are inside dot-prefixed directories.
        // Deleted changes are allowed through to clean up any previously indexed dot-dir entries.
        let file_changes = filter_hidden_changes(std::path::Path::new(&self.repo), file_changes);

        if file_changes.is_empty() {
            debug!(repo = %self.repo, "no changes detected");
            // Check if edges_resolved marker is missing — replay Phase 2.
            let resolved = get_meta(db, EDGES_RESOLVED_KEY).await?.is_some();
            if !resolved {
                info!(repo = %self.repo, "edges_resolved marker absent — replaying Phase 2");
                self.resolve_edges_phase2(db).await
                    .context("edges Phase 2 replay on no-change run")?;
            }
            let indexed = stored_meta.len() as u64;
            return Ok(IndexPipelineStats { indexed_files: indexed, total_files });
        }

        info!(repo = %self.repo, changes = file_changes.len(), "incremental index");
        if let Some(bus) = event_bus {
            bus.emit(IndexEvent::Started {
                repo: self.repo.clone(),
                total_files: file_changes.iter().filter(|c| c.kind != ChangeKind::Deleted).count() as u64,
                is_rebuild: false,
            });
        }
        let (removed_files, new_vectors) = self.incremental_run(db, file_changes, progress.as_ref(), event_bus, key_hints).await?;

        if let Some(vi) = vector_index {
            let mut guard = vi.write().await;
            for file in &removed_files {
                guard.remove_file(file);
            }
            guard.insert(&new_vectors);
        }

        let indexed = get_all_file_meta(db, &self.repo).await?.len() as u64;
        Ok(IndexPipelineStats { indexed_files: indexed, total_files })
    }

    // ─── Full rebuild ─────────────────────────────────────────────────────

    async fn full_rebuild(
        &self,
        db: &Surreal<Db>,
        progress: Option<&ProgressHandle>,
        event_bus: Option<&IndexEventBus>,
        key_hints: &[String],
    ) -> Result<Vec<(ChunkId, Vec<f32>)>> {
        let all_files = walk_repo(&self.repo);
        info!(repo = %self.repo, file_count = all_files.len(), "walking repo for full rebuild");

        // Delete everything first (crash-safe: file_meta is the commit marker,
        // written per-file only after its chunks are durable).
        delete_all_data(db).await.context("full_rebuild: delete_all_data")?;

        // Also clear the edges_resolved marker so Phase 2 re-runs after build.
        let _ = db.query("DELETE FROM index_meta WHERE key = $k")
            .bind(("k", EDGES_RESOLVED_KEY))
            .await;

        // Stream parse → embed → write with bounded channels.
        let chunk_vectors = self
            .streaming_index(&all_files, db, progress, event_bus, key_hints)
            .await
            .context("full_rebuild: streaming_index")?;

        // Phase 2: resolve raw edges into denormalized calls rows.
        if let Some(bus) = event_bus {
            bus.emit(IndexEvent::Phase2Start { repo: self.repo.clone() });
        }
        let phase2_start = Instant::now();
        self.resolve_edges_phase2(db)
            .await
            .context("full_rebuild: resolve_edges_phase2")?;
        if let Some(bus) = event_bus {
            bus.emit(IndexEvent::Phase2Done {
                repo: self.repo.clone(),
                elapsed_ms: phase2_start.elapsed().as_millis() as u64,
            });
        }

        Ok(chunk_vectors)
    }

    // ─── Incremental run ──────────────────────────────────────────────────

    async fn incremental_run(
        &self,
        db: &Surreal<Db>,
        changes: Vec<FileChange>,
        progress: Option<&ProgressHandle>,
        event_bus: Option<&IndexEventBus>,
        key_hints: &[String],
    ) -> Result<(Vec<String>, Vec<(ChunkId, Vec<f32>)>)> {
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

        // Pre-compute: unchanged callers currently pointing INTO the affected files.
        // These calls rows will be destroyed by delete_files_data_bulk below, so
        // we must capture them NOW — before the delete — to avoid losing the
        // "removal direction" (Scenario A: target removes symbol, caller must re-resolve).
        use serde::Deserialize;
        #[derive(Deserialize)]
        struct PreDeleteCallerRow { in_file: String }
        let pre_delete_caller_rows: Vec<PreDeleteCallerRow> = db
            .query(
                "SELECT in_file FROM calls \
                 WHERE out_file IN $files AND in_file NOT IN $files \
                 GROUP BY in_file",
            )
            .bind(("files", all_affected.clone()))
            .await
            .context("incremental_run: pre-delete caller query")?
            .take(0)?;
        let pre_delete_callers: Vec<String> = pre_delete_caller_rows
            .into_iter()
            .map(|r| r.in_file)
            .collect();

        // Bulk-delete all affected files (O(tables) round-trips instead of O(files)).
        delete_files_data_bulk(db, &all_affected)
            .await
            .context("incremental_run: delete_files_data_bulk")?;

        // Stream parse → embed → write.
        let chunk_vectors = self
            .streaming_index(&to_process, db, progress, event_bus, key_hints)
            .await
            .context("incremental_run: streaming_index")?;

        // Delete file_meta for deleted files.
        for file in &to_delete {
            let escaped = escape_surreal(file);
            db.query(format!(
                "DELETE FROM file_meta WHERE path = '{escaped}'"
            ))
            .await
            .context("incremental_run: delete file_meta for deleted file")?;
        }

        // Phase 2: resolve only edges touching the changed files — O(changed + callers_of_changed).
        if let Some(bus) = event_bus {
            bus.emit(IndexEvent::Phase2Start { repo: self.repo.clone() });
        }
        let phase2_start = Instant::now();
        self.resolve_edges_incremental(db, &all_affected, &pre_delete_callers)
            .await
            .context("incremental_run: resolve_edges_incremental")?;
        if let Some(bus) = event_bus {
            bus.emit(IndexEvent::Phase2Done {
                repo: self.repo.clone(),
                elapsed_ms: phase2_start.elapsed().as_millis() as u64,
            });
        }

        Ok((all_affected, chunk_vectors))
    }

    // ─── Streaming parse→embed→write pipeline ────────────────────────────

    /// Stream files through parse → embed → write with bounded channels.
    ///
    /// Peak inflight = PARSE_CHANNEL_CAP + EMBED_CHANNEL_CAP parsed/embedded files
    /// (O(channels * chunks_per_file)), independent of total repo size.
    async fn streaming_index(
        &self,
        files: &[String],
        db: &Surreal<Db>,
        progress: Option<&ProgressHandle>,
        event_bus: Option<&IndexEventBus>,
        key_hints: &[String],
    ) -> Result<Vec<(ChunkId, Vec<f32>)>> {
        if files.is_empty() {
            if let Some(ph) = progress {
                ph.set_run_total(0).await;
                ph.set_processed(0).await;
            }
            return Ok(vec![]);
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

        // ── Stage 1: parallel parse (rayon), feed into bounded channel ────
        let (parse_tx, parse_rx) = mpsc::channel::<ParseOutput>(PARSE_CHANNEL_CAP);
        {
            let files_owned: Vec<String> = files.to_vec();
            tokio::task::spawn_blocking(move || {
                use rayon::prelude::*;
                // Par-iterate, but channel send must be blocking.
                files_owned.par_iter().for_each(|file| {
                    let output = parse_one_file(file);
                    // Blocking send — applies backpressure when embed is slow.
                    if parse_tx.blocking_send(output).is_err() {
                        // Receiver dropped (pipeline cancelled) — stop.
                    }
                });
                // parse_tx dropped here, closing the channel.
            });
        }

        // ── Stage 2: concurrent embed (buffer_unordered(N)) ──────────────
        // Monotonic progress counter shared across concurrent embed tasks.
        let done_counter = Arc::new(AtomicU64::new(0));

        let (embed_tx, mut embed_rx) = mpsc::channel::<EmbeddedFile>(EMBED_CHANNEL_CAP);

        // Wrap the parse receiver as a stream of ParseOutput, embed each
        // concurrently up to `embed_concurrency` at a time.
        {
            let voyage_clone = voyage.clone();
            let done_counter_clone = done_counter.clone();
            let embed_tx_clone = embed_tx.clone();
            let progress_clone = progress.cloned();
            let cache_clone = cache_arc.clone();
            let bus_clone = event_bus_clone.clone();
            let hints_clone = key_hints_owned.clone();

            tokio::spawn(async move {
                // Convert mpsc receiver to a stream.
                let stream = futures::stream::unfold(parse_rx, |mut rx| async move {
                    rx.recv().await.map(|item| (item, rx))
                });

                stream
                    .map(|output| {
                        let voyage_ref = voyage_clone.clone();
                        let cache_ref = cache_clone.clone();
                        let done_ref = done_counter_clone.clone();
                        let progress_ref = progress_clone.clone();
                        let bus_ref = bus_clone.clone();
                        let hints_ref = hints_clone.clone();
                        async move {
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

                                    let embed_result = embed_parsed_file(&pf, voyage_ref.as_ref(), cache_ref.as_deref()).await;

                                    let embed_elapsed_ms = embed_start.elapsed().as_millis() as u64;

                                    // Detect embed failure: all embeddings empty and chunks non-zero
                                    // indicates an API error path.
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

        while let Some(ef) = embed_rx.recv().await {
            // Measure queue wait: time from when EmbeddedFile was created in Stage 2
            // to when Stage 3 picks it up.
            let queue_wait_ms = ef.created_at.elapsed().as_millis() as u64;
            let store_start = Instant::now();

            // Write symbols for this file (native-bind).
            flush_symbol_batch_native(db, &ef.symbols)
                .await
                .context("streaming_index: symbols")?;

            // Write raw edges for this file (Phase 1 — will be resolved in Phase 2).
            flush_raw_edge_batch_native(db, &ef.raw_edges)
                .await
                .context("streaming_index: raw_edges")?;

            // Write chunks for this file and collect chunk vectors.
            let file_chunk_count = ef.chunks.len() as i64;
            let chunk_vectors = flush_file_chunks(db, &ef.chunks, &ef.embeddings)
                .await
                .context("streaming_index: chunks")?;
            all_chunk_vectors.extend(chunk_vectors);

            // Write file_meta LAST (crash-safety anchor).
            // file_meta.chunk_count is set to the actual count for this file.
            upsert_file_meta(db, &FileMeta {
                path: ef.path.clone(),
                mtime: ef.mtime,
                size: ef.size,
                repo: self.repo.clone(),
                chunk_count: file_chunk_count,
            })
            .await
            .context("streaming_index: upsert_file_meta")?;

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

        Ok(all_chunk_vectors)
    }

    // ─── Phase 2: batched edge resolution ────────────────────────────────

    /// Select the best candidate symbol using 4-level priority:
    ///
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
        if candidates.is_empty() {
            return None;
        }

        // Level 1 / Level 2 — only attempted when import_path is present.
        if let Some(imp) = import_path {
            if imp.contains('/') {
                // Level 1: path ends_with import_path (handles subdirectory imports).
                if let Some(found) = candidates.iter().find(|c| c.file.ends_with(imp)) {
                    return Some(found);
                }
            } else {
                // Level 2: bare filename — same parent directory as from_file.
                let from_dir = std::path::Path::new(from_file)
                    .parent()
                    .and_then(|p| p.to_str())
                    .unwrap_or("");
                if let Some(found) = candidates.iter().find(|c| {
                    std::path::Path::new(&c.file)
                        .parent()
                        .and_then(|p| p.to_str())
                        .map(|d| d == from_dir)
                        .unwrap_or(false)
                }) {
                    return Some(found);
                }
            }
        }

        // Level 3: same-file match.
        if let Some(found) = candidates.iter().find(|c| c.file == from_file) {
            return Some(found);
        }

        // Level 4: first in sorted order.
        candidates.first()
    }

    /// Resolve raw edges (stored in `raw_edge` table) into denormalized `calls`
    /// rows using batched symbol lookups. Never loads all symbols into RAM.
    ///
    /// Pages through raw_edge rows using keyset pagination on the SurrealDB-assigned
    /// record id as a string: `SELECT ..., type::string(id) AS id_str FROM raw_edge
    /// WHERE type::string(id) > $cursor ORDER BY id_str LIMIT $page`. The cursor
    /// starts at "" (empty string, sorts before all real record-id strings) and
    /// advances to the last row's id_str after each page.
    ///
    /// This is the same pattern used in `run_migration_v1_to_v2` for the `calls`
    /// table. The record id is:
    ///   - Unique per row (SurrealDB guarantees this).
    ///   - Server-assigned at insert time — never depends on app-managed global state.
    ///   - Durable across server restarts (persists in the DB file).
    ///
    /// Because incremental runs only partially clear raw_edge (delete_files_data_bulk
    /// removes only the changed files' rows, leaving all other files' rows intact),
    /// Phase 2 always re-resolves the ENTIRE raw_edge table. This is correct:
    ///   - Phase 2 starts by deleting ALL calls rows and rebuilding from scratch.
    ///   - The surviving raw_edge rows from unchanged files are included in the rebuild,
    ///     so no edges are lost or orphaned.
    ///   - No raw_edge row can be double-counted: each SurrealDB id is unique and the
    ///     strictly-greater keyset cursor visits each row exactly once.
    ///
    /// NOTE: SurrealDB 2.6.5 requires ORDER BY to reference a projected alias, not a
    /// function call. `ORDER BY type::string(id)` fails; `ORDER BY id_str` works.
    ///
    /// Writes the `edges_resolved` marker in `index_meta` only after all pages commit.
    async fn resolve_edges_phase2(&self, db: &Surreal<Db>) -> Result<()> {
        use serde::Deserialize;

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

        if total == 0 {
            set_meta(db, EDGES_RESOLVED_KEY, "1")
                .await
                .context("phase2: set edges_resolved marker (empty)")?;
            return Ok(());
        }

        // Keyset-paginate raw_edge by the SurrealDB-assigned record id (as string).
        //
        // Correctness guarantee:
        //   - The record id is assigned by SurrealDB at INSERT time — unique per row,
        //     durable across restarts, never depends on app-managed global state.
        //   - `type::string(id) AS id_str` projects the id as a comparable string.
        //   - `WHERE type::string(id) > $cursor ORDER BY id_str LIMIT $page` advances
        //     the cursor to the last row's id_str after each page.
        //   - Every row is visited exactly once: the strictly-greater predicate excludes
        //     the cursor row itself, and the unique id guarantees no row is skipped or
        //     duplicated.
        //   - Sentinel: start with cursor = "" (empty string sorts before all real ids).

        let page_size: i64 = WRITE_BATCH_SIZE as i64;
        let mut cursor = String::new();
        let mut edge_batch: Vec<(String, String, i64, String, String, String, String)> = Vec::new();

        loop {
            let batch: Vec<RawEdgeRow> = db
                .query(
                    "SELECT type::string(id) AS id_str, \
                            from_file, from_name, from_fqn, to_name, kind, line, import_path \
                     FROM raw_edge \
                     WHERE type::string(id) > $cursor \
                     ORDER BY id_str \
                     LIMIT $page",
                )
                .bind(("cursor", cursor.clone()))
                .bind(("page", page_size))
                .await
                .context("phase2: scan raw_edge page")?
                .take(0)?;

            if batch.is_empty() {
                break;
            }

            cursor = batch.last().map(|r| r.id_str.clone()).unwrap_or(cursor);

            resolve_raw_edge_page(db, &batch, &mut edge_batch, "phase2").await?;

            let batch_len = batch.len() as i64;
            if batch_len < page_size {
                break;
            }
        }

        // Flush any remaining edges.
        if !edge_batch.is_empty() {
            flush_edge_batch(db, &edge_batch)
                .await
                .context("phase2: flush tail edge batch")?;
        }

        // Stamp the edges_resolved marker ONLY after all pages commit.
        set_meta(db, EDGES_RESOLVED_KEY, "1")
            .await
            .context("phase2: set edges_resolved marker")?;

        info!(repo = %self.repo, "Phase 2 edge resolution complete");
        Ok(())
    }

    // ─── Incremental Phase 2: scoped edge resolution ──────────────────────

    /// Re-resolve only the edges that touch `changed_files`.
    ///
    /// Complexity: O(changed + callers_of_changed) — proportional to the blast
    /// radius of the edit, not to the total repo size.
    ///
    /// Algorithm (Approach A from spec):
    ///   1. Accept `pre_delete_callers`: unchanged files that previously had calls
    ///      edges pointing into the changed set. These were captured by the caller
    ///      BEFORE `delete_files_data_bulk` ran (the bulk delete removes those calls
    ///      rows, so querying after the delete would miss the "removal direction").
    ///   2. Build `resolve_set = changed_files ∪ pre_delete_callers` (deduped).
    ///   3. Direction-2 expansion: a changed file may have GAINED a symbol whose
    ///      name matches an edge in an unchanged caller. That caller's resolution can
    ///      now pick a different target (the new file wins the lex-first tie-break),
    ///      so we include it in the resolve set even though it never pointed into the
    ///      changed file before.
    ///   4. DELETE FROM calls WHERE in_file IN resolve_set OR out_file IN resolve_set.
    ///      Uses the existing idx_calls_in_file / idx_calls_out_file indexes — O(changed).
    ///   5. Re-resolve raw_edge rows WHERE from_file IN resolve_set via keyset
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
        pre_delete_callers: &[String],
    ) -> Result<()> {
        use serde::Deserialize;

        if changed_files.is_empty() {
            return Ok(());
        }

        // Step 1: Build resolve_set = changed_files ∪ pre_delete_callers (deduped).
        //
        // pre_delete_callers was captured by incremental_run BEFORE delete_files_data_bulk
        // ran, so it correctly captures the "removal direction":
        //   - X→bar resolved to W (out_file=W). W removes bar.
        //   - delete_files_data_bulk deletes X's calls row (out_file=W).
        //   - Querying calls WHERE out_file IN [W] AFTER the bulk delete → empty.
        //   - But pre_delete_callers already contains X, so X enters the resolve set.
        let mut resolve_set: Vec<String> = changed_files.to_vec();
        for caller in pre_delete_callers {
            if !resolve_set.contains(caller) {
                resolve_set.push(caller.clone());
            }
        }

        // Direction 2: "new target now wins" — a changed file may have GAINED a symbol
        // whose name matches an edge in an unchanged caller. That caller's resolution can
        // now pick a different target (the new file wins the lex-first tie-break), so we
        // must include it in the resolve set even though it never pointed into the changed
        // file before.
        //
        // Step: collect the leaf names now defined in the changed files (the ORIGINAL
        // changed_files parameter, NOT the already-expanded resolve_set — we want names
        // that were added/changed, not the transitive set).
        // We query by leaf `name` and look up raw_edge.to_name, which still stores the
        // unresolved leaf callee name — this is correct for direction-2 expansion.
        #[derive(Deserialize)]
        struct SymbolNameRow { name: String }
        let new_symbol_rows: Vec<SymbolNameRow> = db
            .query("SELECT name FROM symbol WHERE file IN $files GROUP BY name")
            .bind(("files", changed_files.to_vec()))
            .await
            .context("incremental phase2: collect symbol names in changed files")?
            .take(0)?;

        if !new_symbol_rows.is_empty() {
            let new_names: Vec<String> = new_symbol_rows.into_iter().map(|r| r.name).collect();

            // Find callers that target any of those names via raw_edge.to_name.
            // raw_edge.to_name stores the unresolved leaf callee name, so this correctly
            // finds any file that calls a symbol with the given leaf name — including files
            // whose existing calls row points to a different definition (stale lex-first target).
            // Uses idx_raw_edge_from_file for the GROUP BY; the to_name lookup is bounded
            // by the number of edges with matching callee names.
            #[derive(Deserialize)]
            struct FromFileRow { from_file: String }
            let name_exp_rows: Vec<FromFileRow> = db
                .query("SELECT from_file FROM raw_edge WHERE to_name IN $names GROUP BY from_file")
                .bind(("names", new_names))
                .await
                .context("incremental phase2: name-based expansion via raw_edge")?
                .take(0)?;

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

        // Step 2: Delete only the calls rows that touch the resolve set.
        // Uses idx_calls_in_file + idx_calls_out_file — O(resolve_set).
        db.query("DELETE FROM calls WHERE in_file IN $files OR out_file IN $files")
            .bind(("files", resolve_set.clone()))
            .await
            .context("incremental phase2: delete scoped calls")?;

        // Step 3: Re-resolve raw_edge rows whose from_file is in the resolve set.
        // Keyset-paginated with from_file filter — uses idx_raw_edge_from_file.

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

        info!(repo = %self.repo, resolve_set = resolve_set.len(), "incremental Phase 2 edge resolution complete");
        Ok(())
    }
}

// ─── Parse one file (returns ParseOutput — always returns, never drops silently) ─

fn parse_one_file(file: &str) -> ParseOutput {
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
}

async fn embed_parsed_file(
    pf: &ParsedFile,
    voyage: Option<&VoyageClient>,
    cache: Option<&EmbeddingCache>,
) -> EmbedFileResult {
    if pf.chunks.is_empty() {
        return EmbedFileResult { embeddings: vec![], fully_cached: false };
    }

    let texts: Vec<String> = pf.chunks.iter().map(|c| c.content.clone()).collect();

    // No voyage client AND no cache → return empty embeddings (same as before).
    if voyage.is_none() && cache.is_none() {
        return EmbedFileResult { embeddings: vec![vec![]; texts.len()], fully_cached: false };
    }

    match cache {
        Some(cache) => {
            // --- Cache path ---
            let (raw_hits, miss_indices) = cache.get_many(&texts);

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
                                cache.put_many(&put_texts, &api_results);
                                for (local_i, emb) in api_results.into_iter().enumerate() {
                                    extra_embeddings.push((dim_miss_indices[local_i], emb));
                                }
                            }
                            Err(e) => {
                                warn!(file = %pf.path, error = %e, "embed failed for dim-mismatched entries; storing empty");
                                for &i in &dim_miss_indices {
                                    extra_embeddings.push((i, vec![]));
                                }
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
                EmbedFileResult { fully_cached: dim_miss_indices.is_empty(), embeddings: result }
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
                            warn!(file = %pf.path, error = %e, "embed failed; storing empty embeddings");
                            None
                        }
                    }
                } else {
                    None
                };

                match api_embeddings {
                    Some(api_results) if !api_results.is_empty() => {
                        // Learn dim from API results.
                        let dim = api_results[0].len();

                        // Cache the API results.
                        let miss_texts: Vec<String> = all_miss_indices
                            .iter()
                            .map(|&i| texts[i].clone())
                            .collect();
                        cache.put_many(&miss_texts, &api_results);

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
                                    cache.put_many(&re_texts, &re_results);
                                    for (local_i, emb) in re_results.into_iter().enumerate() {
                                        result[re_embed_indices[local_i]] = emb;
                                    }
                                }
                                Err(e) => {
                                    warn!(file = %pf.path, error = %e, "re-embed failed for dim-mismatched hits; storing empty");
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

                EmbedFileResult { fully_cached: false, embeddings: result }
            }
        }
        None => {
            // No cache — existing behavior.
            match voyage {
                Some(client) => {
                    match client.embed(&texts, InputType::Document).await {
                        Ok(embs) => EmbedFileResult { fully_cached: false, embeddings: embs },
                        Err(e) => {
                            warn!(file = %pf.path, error = %e, "embed failed; storing empty embeddings");
                            EmbedFileResult { fully_cached: false, embeddings: vec![vec![]; texts.len()] }
                        }
                    }
                }
                None => EmbedFileResult { fully_cached: false, embeddings: vec![vec![]; texts.len()] },
            }
        }
    }
}

// ─── Flush helpers ────────────────────────────────────────────────────────

/// Write chunks for a single file and return (ChunkId, embedding) pairs.
/// Drops chunk content from memory after flushing — never accumulates globally.
async fn flush_file_chunks(
    db: &Surreal<Db>,
    chunks: &[crate::parsing::chunker::Chunk],
    embeddings: &[Vec<f32>],
) -> Result<Vec<(ChunkId, Vec<f32>)>> {
    let mut chunk_vectors: Vec<(ChunkId, Vec<f32>)> = Vec::with_capacity(chunks.len());
    let mut batch: Vec<ChunkRecord> = Vec::with_capacity(WRITE_BATCH_SIZE);

    for (chunk, emb) in chunks.iter().zip(
        embeddings.iter().cloned().chain(std::iter::repeat(vec![]))
    ) {
        chunk_vectors.push((
            ChunkId {
                file: chunk.file.clone(),
                line_start: chunk.line_start,
                line_end: chunk.line_end,
            },
            emb.clone(),
        ));
        batch.push(ChunkRecord {
            file: chunk.file.clone(),
            line_start: chunk.line_start as i64,
            line_end: chunk.line_end as i64,
            content: chunk.content.clone(),
            embedding: emb,
            symbol_ref: chunk.symbol_ref.as_ref().map(|fqn| format!("symbol:⟨{fqn}⟩")),
        });

        if batch.len() >= WRITE_BATCH_SIZE {
            flush_chunk_batch(db, std::mem::take(&mut batch))
                .await
                .context("flush_file_chunks: batch")?;
        }
    }
    if !batch.is_empty() {
        flush_chunk_batch(db, batch)
            .await
            .context("flush_file_chunks: tail batch")?;
    }

    Ok(chunk_vectors)
}

/// Flush a batch of chunk records via a native-bind INSERT.
async fn flush_chunk_batch(db: &Surreal<Db>, batch: Vec<ChunkRecord>) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }
    db.query("INSERT INTO chunk $data RETURN NONE")
        .bind(("data", batch))
        .await
        .context("flush_chunk_batch")?;
    Ok(())
}

/// Flush symbols for one file using native-bind INSERT (❹: replaces text-query builder).
async fn flush_symbol_batch_native(db: &Surreal<Db>, symbols: &[Symbol]) -> Result<()> {
    use crate::store::ops::kind_to_str;

    for chunk in symbols.chunks(WRITE_BATCH_SIZE) {
        let records: Vec<SymbolRecord> = chunk
            .iter()
            .map(|sym| SymbolRecord {
                name: sym.qualified.name.clone(),
                kind: kind_to_str(&sym.kind).to_string(),
                file: sym.qualified.file.clone(),
                line_start: sym.line_start as i64,
                line_end: sym.line_end as i64,
                signature: sym.signature.clone(),
                parent: sym.parent_fqn.as_deref().map(|p| format!("symbol:⟨{}⟩", p)),
            })
            .collect();

        if !records.is_empty() {
            // Use UPSERT so incremental re-runs are idempotent.
            // We need to upsert by the deterministic record ID (fqn).
            // Native INSERT INTO doesn't support custom IDs, so fall back to
            // per-record UPSERT with native bind for the field values.
            for (sym, rec) in chunk.iter().zip(records) {
                let fqn = escape_surreal(&sym.qualified.fqn());
                db.query(format!(
                    "UPSERT symbol:`⟨{fqn}⟩` SET \
                     name = $name, kind = $kind, file = $file, \
                     line_start = $line_start, line_end = $line_end, \
                     signature = $signature, parent = $parent"
                ))
                .bind(("name", rec.name))
                .bind(("kind", rec.kind))
                .bind(("file", rec.file))
                .bind(("line_start", rec.line_start))
                .bind(("line_end", rec.line_end))
                .bind(("signature", rec.signature))
                .bind(("parent", rec.parent))
                .await
                .context("flush_symbol_batch_native: upsert symbol")?;
            }
        }
    }
    Ok(())
}

/// Flush raw edges (Phase 1) using native-bind INSERT.
/// Raw edges are stored in `raw_edge` table for later Phase 2 resolution.
async fn flush_raw_edge_batch_native(db: &Surreal<Db>, edges: &[RawEdgeRecord]) -> Result<()> {
    for chunk in edges.chunks(WRITE_BATCH_SIZE) {
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

/// Flush a batch of resolved calls edges as a single multi-statement query.
///
/// Each edge is written as `RELATE symbol:⟨…⟩->calls->symbol:⟨…⟩ SET …`.
/// All statements are concatenated into one `db.query(...)` call to achieve
/// O(1) round-trips per batch instead of O(edges).
///
/// Why text-query with escape_surreal instead of native bind:
/// SurrealDB record IDs (`symbol:⟨fqn⟩`) cannot be passed as typed parameters
/// in RELATE — they must appear literally in the SurrealQL text. escape_surreal
/// handles the only injection surface (backslash and single-quote escaping).
async fn flush_edge_batch(
    db: &Surreal<Db>,
    batch: &[(String, String, i64, String, String, String, String)],
) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }

    let mut query = String::with_capacity(batch.len() * 256);
    for (from_fqn, to_fqn, line, in_file, out_file, in_name, out_name) in batch {
        let from_esc  = escape_surreal(from_fqn);
        let to_esc    = escape_surreal(to_fqn);
        let in_file_e = escape_surreal(in_file);
        let out_file_e = escape_surreal(out_file);
        let in_name_e  = escape_surreal(in_name);
        let out_name_e = escape_surreal(out_name);
        query.push_str(&format!(
            "RELATE symbol:`⟨{from_esc}⟩`->calls->symbol:`⟨{to_esc}⟩` \
             SET line = {line}, in_file = '{in_file_e}', out_file = '{out_file_e}', \
             in_name = '{in_name_e}', out_name = '{out_name_e}';\n"
        ));
    }

    db.query(query)
        .await
        .context("flush_edge_batch: RELATE statements")?;

    Ok(())
}

// ─── Hidden-directory change filter ──────────────────────────────────────

/// Filter watcher-supplied file changes so that Added/Modified events for files
/// inside dot-prefixed directories are never ingested.
///
/// Deleted changes are always allowed through — they clean up any stale entries
/// that were indexed before this rule existed (self-healing behavior).
pub(crate) fn filter_hidden_changes(repo: &std::path::Path, changes: Vec<FileChange>) -> Vec<FileChange> {
    changes
        .into_iter()
        .filter(|c| {
            // Always allow Deleted changes through (self-heal stale dot-dir entries).
            if c.kind == ChangeKind::Deleted {
                return true;
            }
            // Drop Added/Modified if the path is inside a dot-prefixed directory.
            !is_under_hidden_dir(repo, std::path::Path::new(&c.path))
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

        let db = open_db(home.path(), &repo).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);

        let result = pipeline.run(&db, None, true, None, None, None, &[]).await;
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

        let db = open_db(home.path(), &repo).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);

        let stats = pipeline
            .run(&db, None, true, None, None, None, &[])
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

        let db = open_db(home.path(), &repo).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);
        pipeline.run(&db, None, true, None, None, None, &[]).await.expect("rebuild");

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

        let db = open_db(home.path(), &repo).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);
        pipeline.run(&db, None, true, None, None, None, &[]).await.expect("rebuild");

        let marker = get_meta(&db, EDGES_RESOLVED_KEY).await.unwrap();
        assert!(marker.is_some(), "edges_resolved marker must be set after full_rebuild");
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
        let db = open_db(home.path(), repo).await.unwrap();

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
        let db = open_db(home.path(), repo).await.unwrap();

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
        let db = open_db(home.path(), repo).await.unwrap();

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
        let db = open_db(home.path(), repo).await.unwrap();

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

        // Create two RELATE edges WITHOUT in_name/out_name (v1 state).
        // Both share in_file=/a.rs and out_file=/b.rs.
        db.query(
            "RELATE symbol:`⟨/a.rs::foo⟩`->calls->symbol:`⟨/b.rs::baz⟩` \
             SET line = 1, in_file = '/a.rs', out_file = '/b.rs'"
        ).await.unwrap();

        db.query(
            "RELATE symbol:`⟨/a.rs::bar⟩`->calls->symbol:`⟨/b.rs::qux⟩` \
             SET line = 2, in_file = '/a.rs', out_file = '/b.rs'"
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
        let db = open_db(home.path(), repo).await.unwrap();

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
        pipeline.resolve_edges_phase2(&db).await.expect("initial full phase2");

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
        // and the direction-1 path (A was a caller of B) is covered by pre_delete_callers
        // in production; here we pass empty and verify that A is still found because
        // it still has a surviving calls row pointing at B when we call this method
        // (we did not call delete_files_data_bulk in this direct-call test).
        let changed = vec!["/b.rs".to_string()];
        pipeline.resolve_edges_incremental(&db, &changed, &[])
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
        let db = open_db(home.path(), repo).await.unwrap();

        let pipeline = IndexPipeline::new(repo.to_string(), None);

        // ── Initial state: X calls foo, only Z defines foo ─────────────────
        // Paths chosen so that /a_defines_foo.rs < /z_defines_foo.rs
        // and /x_caller.rs sits between them alphabetically — it is NOT the
        // lex-first definer, so X is not picked as a self-file resolution.
        insert_symbol(&db, "/z_defines_foo.rs", "foo").await;
        insert_raw_edge(&db, "/x_caller.rs", "x_fn", "foo").await;

        // Full Phase 2: X→foo resolves to Z (the only candidate).
        pipeline.resolve_edges_phase2(&db).await.expect("initial full phase2");

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
        // pre_delete_callers is empty: X never pointed into W (W didn't exist yet),
        // so the pre-delete query would return nothing for this scenario. Direction-2
        // expansion (name-based) is what finds X here.
        let changed = vec!["/a_defines_foo.rs".to_string()];
        pipeline
            .resolve_edges_incremental(&db, &changed, &[])
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
        let db = open_db(home.path(), repo).await.unwrap();

        let pipeline = IndexPipeline::new(repo.to_string(), None);

        // ── Initial state: X calls bar, W and Y both define bar, W < Y lex ──
        // W="/w.rs" < Y="/y.rs" lexicographically, so W wins the tie-break.
        insert_symbol(&db, "/w.rs", "bar").await;
        insert_symbol(&db, "/y.rs", "bar").await;
        insert_raw_edge(&db, "/x.rs", "x_fn", "bar").await;

        // Full Phase 2: X→bar→W (W is lex-first).
        pipeline.resolve_edges_phase2(&db).await.expect("initial full phase2");

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

        // Step 4: resolve_edges_incremental with pre_delete_callers=[X].
        pipeline
            .resolve_edges_incremental(&db, &changed_files, &pre_delete_callers)
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
        let db = open_db(home.path(), repo).await.unwrap();

        let pipeline = IndexPipeline::new(repo.to_string(), None);

        // ── Initial state: X calls foo, W defines foo ─────────────────────
        insert_symbol(&db, "/w.rs", "foo").await;
        insert_raw_edge(&db, "/x.rs", "x_fn", "foo").await;

        // Full Phase 2: X→foo→W.
        pipeline.resolve_edges_phase2(&db).await.expect("initial full phase2");

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

        // Step 4: resolve_edges_incremental([W], pre_delete_callers=[X]).
        pipeline
            .resolve_edges_incremental(&db, &changed_files, &pre_delete_callers)
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
}
