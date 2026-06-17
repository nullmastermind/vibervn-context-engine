use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tokio::sync::RwLock;
use tracing::warn;

use crate::embedding::voyage::VoyageClient;
use crate::indexing::IndexEngine;
use crate::llm::LlmClient;
use crate::path_in_repo;
use crate::query::find_db_for_file;
use crate::query::graph_expand::graph_expand;
use crate::query::merger::{MergeChunk, merge_chunks};
use crate::query::reranker;

// ─── Output types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct CodeResult {
    pub file: String,
    pub line_start: u32,
    pub line_end: u32,
    pub score: f32,
    /// Numbered lines from filesystem, or stored content on fallback.
    pub content: String,
    pub symbol: Option<String>,
    /// Number of direct callers (depth-1) from the call graph. Lower bound.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callers: Option<u32>,
    /// Number of distinct files containing direct callers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caller_files: Option<u32>,
    /// Short names of callers, sorted by proximity (max 3 for display).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub caller_names: Vec<String>,
    /// Short names of callees, sorted by proximity (max 3 for display).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub callee_names: Vec<String>,
    /// Number of callees.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callees: Option<u32>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct QueryTiming {
    pub embed_ms: u64,
    pub search_ms: u64,
    pub graph_ms: u64,
    pub merge_ms: u64,
    pub rerank_ms: u64,
    pub total_ms: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RerankInfo {
    pub raw_request: String,
    pub raw_response: String,
    pub fallback_used: bool,
    pub skip_reason: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct QueryResult {
    pub results: Vec<CodeResult>,
    pub pre_rerank_results: Vec<CodeResult>,
    pub timing: QueryTiming,
    pub rerank: Option<RerankInfo>,
    /// True when the target repo's vector shard was not resident after the bounded
    /// warm-wait expired — an empty `results` here means "still warming, retry",
    /// NOT "the index contains nothing". False for a genuine empty or a hit.
    #[serde(default)]
    pub warming: bool,
}

// ─── DB row types ─────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct ChunkContentRow {
    content: String,
    symbol_ref: Option<String>,
}

// ─── Pipeline ─────────────────────────────────────────────────────────────

/// Execute the full query pipeline:
/// embed → vector search → graph expand → merge → rerank → format.
///
/// `repo_filter`: if Some, only return results from that repo path prefix.
/// `llm_client`: if None, rerank step is skipped.
/// `warm_wait`: max time to block warming a cold single-repo shard before search.
#[allow(clippy::too_many_arguments)]
pub async fn run_query(
    query: &str,
    top_k: usize,
    repo_filter: Option<&str>,
    voyage_client: &VoyageClient,
    index_engine: &Arc<IndexEngine>,
    repo_dbs: &Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    min_prune_lines: u32,
    llm_client: Option<&LlmClient>,
    warm_wait: std::time::Duration,
    agentic_rag: bool,
    agentic_rag_max_turns: u32,
    agentic_rag_max_chunk_chars: u32,
    agentic_rag_grep_read: bool,
) -> Result<QueryResult> {
    run_query_with_filters(
        query, top_k, repo_filter, voyage_client, index_engine, repo_dbs,
        min_prune_lines, llm_client, warm_wait, agentic_rag,
        agentic_rag_max_turns, agentic_rag_max_chunk_chars, agentic_rag_grep_read, None,
    ).await
}

/// Extended query entry point that accepts pre-parsed structured filters
/// (from MCP tool params). Merged with any in-query filter prefixes.
#[allow(clippy::too_many_arguments)]
pub async fn run_query_with_filters(
    query: &str,
    top_k: usize,
    repo_filter: Option<&str>,
    voyage_client: &VoyageClient,
    index_engine: &Arc<IndexEngine>,
    repo_dbs: &Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    min_prune_lines: u32,
    llm_client: Option<&LlmClient>,
    warm_wait: std::time::Duration,
    agentic_rag: bool,
    agentic_rag_max_turns: u32,
    agentic_rag_max_chunk_chars: u32,
    agentic_rag_grep_read: bool,
    external_filters: Option<crate::query::filters::QueryFilters>,
) -> Result<QueryResult> {
    let total_start = Instant::now();

    // ── Step 0: Parse query filters ──────────────────────────────────────────
    let (clean_query, mut filters) = crate::query::filters::parse_query_filters(query);
    if let Some(ext) = external_filters {
        filters.merge(ext);
    }
    // Use clean query for embedding (filters stripped), or original if clean is empty
    let embed_query = if clean_query.is_empty() { query } else { &clean_query };

    // ── Step 1: Embed query ───────────────────────────────────────────────
    let embed_start = Instant::now();
    let embedding = voyage_client.embed_query(embed_query).await?;
    let embed_ms = embed_start.elapsed().as_millis() as u64;

    if embedding.is_empty() {
        bail!("embed_query returned an empty vector");
    }

    // ── Step 2: Vector search ─────────────────────────────────────────────
    let search_start = Instant::now();
    // Search for 2× top_k so graph expansion has candidates to work with.
    let crate::indexing::VectorSearchOutcome { results: raw_results, warming } =
        index_engine.vector_search(&embedding, top_k * 2, repo_filter, warm_wait).await;
    let search_ms = search_start.elapsed().as_millis() as u64;

    if raw_results.is_empty() {
        return Ok(QueryResult {
            results: vec![],
            pre_rerank_results: vec![],
            timing: QueryTiming {
                embed_ms,
                search_ms,
                graph_ms: 0,
                merge_ms: 0,
                rerank_ms: 0,
                total_ms: total_start.elapsed().as_millis() as u64,
            },
            rerank: None,
            warming,
        });
    }

    // Apply repo filter.
    let filtered: Vec<_> = if let Some(repo) = repo_filter {
        raw_results
            .into_iter()
            .filter(|r| path_in_repo(&r.chunk_id.file, repo))
            .collect()
    } else {
        raw_results
    };

    // ── Step 3: Fetch stored content for base chunks ──────────────────────
    // Clone DB handles (Arc-wrapped, cheap) and release the RwLock immediately.
    // This prevents holding the lock across graph expansion await points.
    let db_map: HashMap<String, Surreal<Db>> = {
        let guard = repo_dbs.read().await;
        guard.clone()
    }; // read lock dropped HERE — before any async DB queries

    let mut base_chunks: Vec<MergeChunk> = Vec::with_capacity(filtered.len());
    for sr in &filtered {
        let (content, symbol, symbol_fqn, symbol_kind) =
            fetch_chunk_content(&db_map, &sr.chunk_id.file, sr.chunk_id.line_start, sr.chunk_id.line_end).await;
        base_chunks.push(MergeChunk {
            file: sr.chunk_id.file.clone(),
            line_start: sr.chunk_id.line_start,
            line_end: sr.chunk_id.line_end,
            score: sr.score,
            content,
            symbol,
            symbol_fqn,
            symbol_kind,
        });
    }

    // ── Step 3.5: Apply query filters ────────────────────────────────────
    if !filters.is_empty() {
        base_chunks = apply_query_filters(base_chunks, &filters);
    }

    // ── Step 4: Graph expansion ───────────────────────────────────────────
    let graph_start = Instant::now();

    // Read db_schema_version from any available DB (cached after migration).
    // If no DB available, default to 1 (safe: uses unindexed but correct path).
    let schema_version = {
        let db_map_guard = repo_dbs.read().await;
        if let Some(db) = db_map_guard.values().next() {
            crate::store::read_db_schema_version(db).await
        } else {
            1
        }
    };

    let expanded = graph_expand(&base_chunks, &db_map, schema_version).await;
    let graph_ms = graph_start.elapsed().as_millis() as u64;

    // Merge base + expanded into a single pool.
    let mut all_chunks = base_chunks;
    for e in expanded {
        all_chunks.push(MergeChunk {
            file: e.file,
            line_start: e.line_start,
            line_end: e.line_end,
            score: e.score,
            content: e.content,
            symbol: e.symbol,
            symbol_fqn: e.symbol_fqn,
            symbol_kind: e.symbol_kind,
        });
    }

    // ── Step 5: Merge ─────────────────────────────────────────────────────
    let merge_start = Instant::now();
    let merged = merge_chunks(all_chunks, top_k);
    let merge_ms = merge_start.elapsed().as_millis() as u64;

    // Read numbered content from disk ONCE per candidate. Bounded by top_k
    // (merge caps the candidate set), so this is never an unbounded read storm.
    // Reused for BOTH the rerank LLM payload and the final output — no double
    // read. `None` means the file could not be read (deleted/moved since index);
    // that chunk degrades to stored DB content and is never line-pruned.
    let numbered: Vec<Option<String>> = merged
        .iter()
        .map(|c| read_lines_from_fs(&c.file, c.line_start, c.line_end).ok())
        .collect();

    // ── Step 5.5: Caller stats (parallel with numbered read, bounded by top_k)
    let caller_stats = fetch_caller_stats_batch(&merged, &db_map, schema_version).await;

    // Convert enriched stats to legacy tuple format for the reranker interface.
    let legacy_stats: Vec<Option<(u32, u32)>> = caller_stats
        .iter()
        .map(|s| s.as_ref().map(|cs| (cs.caller_count, cs.caller_file_count)))
        .collect();

    // ── Step 6: Rerank ────────────────────────────────────────────────────
    let rerank_start = Instant::now();
    // `extended_pool` is Some only on the agentic path: its chunks/numbered are
    // the base candidates followed by any `query`-tool results, and the returned
    // indices address THAT pool. On the single-shot path it's None and indices
    // address the base `merged`/`numbered`.
    let (rerank_output, extended_pool) = match (agentic_rag, llm_client, repo_filter) {
        (true, Some(client), Some(repo)) => {
            let (out, pool) = reranker::rerank_agentic(
                query, &merged, &numbered, &legacy_stats, min_prune_lines,
                client, agentic_rag_max_turns, agentic_rag_max_chunk_chars, agentic_rag_grep_read,
                repo, voyage_client, index_engine, repo_dbs, warm_wait,
            ).await;
            (out, Some(pool))
        }
        _ => {
            let out = reranker::rerank(query, &merged, &numbered, &legacy_stats, min_prune_lines, llm_client).await;
            (out, None)
        }
    };
    let rerank_ms = rerank_start.elapsed().as_millis() as u64;

    // ── Step 7: Format ────────────────────────────────────────────────────
    // Resolve reranked indices against the extended pool when present (agentic),
    // else the base merged set. The pool's first `merged.len()` entries ARE the
    // base chunks, so base-index selections still resolve correctly; only
    // sub-query chunks (index >= base len) live exclusively in the pool.
    let (res_chunks, res_numbered): (&[MergeChunk], &[Option<String>]) = match &extended_pool {
        Some(pool) => (&pool.chunks, &pool.numbered),
        None => (&merged, &numbered),
    };

    // Build final results in reranked order. When the LLM selected line ranges
    // for a chunk, emit one block per range (sliced from the already-read
    // numbered text — no re-read); otherwise emit the whole chunk.
    let mut results: Vec<CodeResult> = Vec::new();
    for (k, &idx) in rerank_output.reranked_indices.iter().enumerate() {
        let Some(chunk) = res_chunks.get(idx) else { continue };
        // Caller stats are computed only for the base candidate set; chunks
        // pulled in by the `query` tool (index >= merged.len()) have none.
        let stats = caller_stats.get(idx).and_then(|s| s.as_ref());
        let (callers, caller_files) = stats.map_or((None, None), |s| (Some(s.caller_count), Some(s.caller_file_count)));
        let caller_names = stats.map(|s| s.caller_names.clone()).unwrap_or_default();
        let callee_names = stats.map(|s| s.callee_names.clone()).unwrap_or_default();
        let callees = stats.map(|s| s.callee_count);
        let numbered_text = res_numbered.get(idx).and_then(|n| n.as_deref());
        let selection = rerank_output.line_selections.get(k).and_then(|s| s.as_ref());
        match (numbered_text, selection) {
            (Some(text), Some(ranges)) if !ranges.is_empty() => {
                for &(s, e) in ranges {
                    results.push(CodeResult {
                        file: chunk.file.clone(),
                        line_start: s,
                        line_end: e,
                        score: chunk.score,
                        content: slice_numbered(text, chunk.line_start, s, e),
                        symbol: chunk.symbol.clone(),
                        callers,
                        caller_files,
                        caller_names: caller_names.clone(),
                        callee_names: callee_names.clone(),
                        callees,
                    });
                }
            }
            (Some(text), _) => results.push(CodeResult {
                file: chunk.file.clone(),
                line_start: chunk.line_start,
                line_end: chunk.line_end,
                score: chunk.score,
                content: text.to_string(),
                symbol: chunk.symbol.clone(),
                callers,
                caller_files,
                caller_names: caller_names.clone(),
                callee_names: callee_names.clone(),
                callees,
            }),
            (None, _) => results.push(CodeResult {
                file: chunk.file.clone(),
                line_start: chunk.line_start,
                line_end: chunk.line_end,
                score: chunk.score,
                content: chunk.content.clone(),
                symbol: chunk.symbol.clone(),
                callers,
                caller_files,
                caller_names: caller_names.clone(),
                callee_names: callee_names.clone(),
                callees,
            }),
        }
    }

    // Pre-rerank diagnostic output, reusing the same numbered reads.
    let mut pre_rerank_results: Vec<CodeResult> = Vec::with_capacity(merged.len());
    for (i, chunk) in merged.iter().enumerate() {
        let content = numbered
            .get(i)
            .and_then(|n| n.clone())
            .unwrap_or_else(|| chunk.content.clone());
        let stats = caller_stats.get(i).and_then(|s| s.as_ref());
        let (callers, caller_files) = stats.map_or((None, None), |s| (Some(s.caller_count), Some(s.caller_file_count)));
        pre_rerank_results.push(CodeResult {
            file: chunk.file.clone(),
            line_start: chunk.line_start,
            line_end: chunk.line_end,
            score: chunk.score,
            content,
            symbol: chunk.symbol.clone(),
            callers,
            caller_files,
            caller_names: stats.map(|s| s.caller_names.clone()).unwrap_or_default(),
            callee_names: stats.map(|s| s.callee_names.clone()).unwrap_or_default(),
            callees: stats.map(|s| s.callee_count),
        });
    }

    let total_ms = total_start.elapsed().as_millis() as u64;

    let rerank_info = RerankInfo {
        raw_request: rerank_output.raw_request,
        raw_response: rerank_output.raw_response,
        fallback_used: rerank_output.fallback_used,
        skip_reason: rerank_output.skip_reason,
    };

    Ok(QueryResult {
        results,
        pre_rerank_results,
        timing: QueryTiming {
            embed_ms,
            search_ms,
            graph_ms,
            merge_ms,
            rerank_ms,
            total_ms,
        },
        rerank: Some(rerank_info),
        warming,
    })
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Sub-query: embed → vector search → graph expand → merge. NO rerank stage.
/// Used exclusively by the agentic rerank loop's `query` tool. Cannot recurse
/// into rerank because it has no `llm_client` and no rerank call.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_sub_query(
    query: &str,
    top_k: usize,
    repo_filter: &str,
    voyage_client: &VoyageClient,
    index_engine: &Arc<IndexEngine>,
    repo_dbs: &Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    warm_wait: std::time::Duration,
) -> Result<Vec<MergeChunk>> {
    let embedding = voyage_client.embed_query(query).await?;
    if embedding.is_empty() {
        bail!("embed_query returned an empty vector");
    }

    let crate::indexing::VectorSearchOutcome { results: raw_results, .. } =
        index_engine.vector_search(&embedding, top_k * 2, Some(repo_filter), warm_wait).await;
    if raw_results.is_empty() {
        return Ok(vec![]);
    }

    let filtered: Vec<_> = raw_results
        .into_iter()
        .filter(|r| path_in_repo(&r.chunk_id.file, repo_filter))
        .collect();

    // Clone DB handles then drop the read lock — no guard spans the await below.
    let db_map: HashMap<String, Surreal<Db>> = {
        let guard = repo_dbs.read().await;
        guard.clone()
    };

    let mut base_chunks: Vec<MergeChunk> = Vec::with_capacity(filtered.len());
    for sr in &filtered {
        let (content, symbol, symbol_fqn, symbol_kind) =
            fetch_chunk_content(&db_map, &sr.chunk_id.file, sr.chunk_id.line_start, sr.chunk_id.line_end).await;
        base_chunks.push(MergeChunk {
            file: sr.chunk_id.file.clone(),
            line_start: sr.chunk_id.line_start,
            line_end: sr.chunk_id.line_end,
            score: sr.score,
            content,
            symbol,
            symbol_fqn,
            symbol_kind,
        });
    }

    let schema_version = {
        let db_map_guard = repo_dbs.read().await;
        if let Some(db) = db_map_guard.values().next() {
            crate::store::read_db_schema_version(db).await
        } else {
            1
        }
    };

    let expanded = graph_expand(&base_chunks, &db_map, schema_version).await;

    let mut all_chunks = base_chunks;
    for e in expanded {
        all_chunks.push(MergeChunk {
            file: e.file,
            line_start: e.line_start,
            line_end: e.line_end,
            score: e.score,
            content: e.content,
            symbol: e.symbol,
            symbol_fqn: e.symbol_fqn,
            symbol_kind: e.symbol_kind,
        });
    }

    Ok(merge_chunks(all_chunks, top_k))
}

// ─── Query filter application ────────────────────────────────────────────

/// Apply parsed query filters to the candidate chunks. Filters by path, language
/// (inferred from file extension), symbol kind (from symbol name heuristics), and
/// symbol name (exact match, with fuzzy fallback on zero results).
fn apply_query_filters(
    chunks: Vec<MergeChunk>,
    filters: &crate::query::filters::QueryFilters,
) -> Vec<MergeChunk> {
    use crate::parsing::detect_language;
    use crate::query::filters::bounded_edit_distance;
    use std::path::Path;

    // First pass: apply path and language filters (not name).
    let base_set: Vec<MergeChunk> = chunks
        .into_iter()
        .filter(|chunk| {
            // Path filter: chunk file path must contain the filter substring
            if !filters.path_filters.is_empty()
                && !filters.path_filters.iter().any(|p| {
                    let norm_file = chunk.file.replace('\\', "/");
                    let norm_filter = p.replace('\\', "/");
                    norm_file.contains(&norm_filter)
                })
            {
                return false;
            }

            // Language filter: infer language from file extension
            if !filters.languages.is_empty() {
                let lang = detect_language(Path::new(&chunk.file));
                let lang_str = format!("{:?}", lang).to_lowercase();
                if !filters.languages.iter().any(|l| {
                    lang_str.contains(l) || lang_matches_alias(l, &lang_str)
                }) {
                    return false;
                }
            }

            // Kind filter: match against symbol_kind if populated on MergeChunk.
            if !filters.kinds.is_empty()
                && let Some(ref kind) = chunk.symbol_kind
            {
                let kind_lower = kind.to_lowercase();
                if !filters.kinds.iter().any(|k| kind_lower.contains(&k.to_lowercase())) {
                    return false;
                }
            }

            true
        })
        .collect();

    // If no name filters, we're done.
    if filters.name_filters.is_empty() {
        return base_set;
    }

    // Second pass: exact name filtering.
    let exact_results: Vec<MergeChunk> = base_set
        .iter()
        .filter(|chunk| {
            if let Some(ref sym) = chunk.symbol {
                let sym_lower = sym.to_lowercase();
                filters.name_filters.iter().any(|n| sym_lower.contains(&n.to_lowercase()))
            } else {
                // No symbol info — keep the chunk (conservative)
                true
            }
        })
        .cloned()
        .collect();

    if !exact_results.is_empty() {
        return exact_results;
    }

    // Fuzzy name fallback: exact match yielded zero results — retry with
    // bounded edit distance <= 2 on the base set.
    base_set
        .into_iter()
        .filter(|chunk| {
            if let Some(ref sym) = chunk.symbol {
                let sym_lower = sym.to_lowercase();
                filters.name_filters.iter().any(|n| {
                    bounded_edit_distance(&sym_lower, &n.to_lowercase(), 2) <= 2
                })
            } else {
                false
            }
        })
        .collect()
}

/// Check if a filter language name matches a detected language string with common aliases.
fn lang_matches_alias(filter: &str, detected: &str) -> bool {
    match filter {
        "ts" | "typescript" => detected == "typescript" || detected == "tsx",
        "js" | "javascript" => detected == "javascript",
        "tsx" => detected == "tsx",
        "py" | "python" => detected == "python",
        "rs" | "rust" => detected == "rust",
        "go" | "golang" => detected == "go",
        "java" => detected == "java",
        "c" => detected == "c",
        "cpp" | "c++" | "cxx" => detected == "cpp",
        "cs" | "csharp" | "c#" => detected == "csharp",
        "rb" | "ruby" => detected == "ruby",
        "php" => detected == "php",
        "swift" => detected == "swift",
        "kotlin" | "kt" => detected == "kotlin",
        "dart" => detected == "dart",
        _ => false,
    }
}

/// Enriched caller/callee stats for a single chunk's symbol.
/// Contains counts, file counts, and up to 3 short names for display.
#[derive(Debug, Clone)]
pub struct CallerCalleeStats {
    pub caller_count: u32,
    pub caller_file_count: u32,
    /// Short names of callers, sorted by proximity (same-file first).
    pub caller_names: Vec<String>,
    pub callee_count: u32,
    pub callee_file_count: u32,
    /// Short names of callees, sorted by proximity.
    pub callee_names: Vec<String>,
}

/// Legacy type alias for backward compatibility with reranker which uses (count, file_count).
type CallerStats = CallerCalleeStats;

/// Batch-query caller stats for merged chunks. Returns a Vec aligned 1:1 with
/// `merged`: `Some(stats)` when the chunk has a symbol_fqn and the
/// DB query succeeds, `None` otherwise. Bounded by merged.len() (≤ top_k).
async fn fetch_caller_stats_batch(
    merged: &[MergeChunk],
    db_map: &HashMap<String, Surreal<Db>>,
    schema_version: u32,
) -> Vec<Option<CallerStats>> {
    let mut stats: Vec<Option<CallerStats>> = vec![None; merged.len()];

    for (i, chunk) in merged.iter().enumerate() {
        let fqn = match &chunk.symbol_fqn {
            Some(f) => f,
            None => continue,
        };
        let db = match find_db_for_file(db_map, &chunk.file) {
            Some(db) => db,
            None => continue,
        };

        if let Some(s) = query_caller_callee_stats(db, fqn, &chunk.file, schema_version).await {
            stats[i] = Some(s);
        }
    }
    stats
}

#[derive(serde::Deserialize)]
struct CallerRow {
    in_file: String,
    #[serde(default)]
    in_name: Option<String>,
}

#[derive(serde::Deserialize)]
struct CalleeRow {
    out_file: String,
    #[serde(default)]
    out_name: Option<String>,
}

async fn query_caller_callee_stats(
    db: &Surreal<Db>,
    fqn: &str,
    target_file: &str,
    schema_version: u32,
) -> Option<CallerCalleeStats> {
    // Fetch callers (who calls this symbol)
    let caller_rows: Vec<CallerRow> = if schema_version >= 2 {
        db.query("SELECT in_file, in_name FROM calls WHERE out_name = $fqn")
            .bind(("fqn", fqn.to_string()))
            .await
            .ok()?
            .take(0)
            .ok()?
    } else {
        let name = fqn.rsplit("::").next().unwrap_or(fqn);
        db.query("SELECT in_file, in_name FROM calls WHERE out_name = $name")
            .bind(("name", name.to_string()))
            .await
            .ok()?
            .take(0)
            .ok()?
    };

    // Only count edges whose caller symbol name is known. Rows with a NULL/empty
    // in_name (v1→v2 migrated DBs where link-deref failed) would otherwise inflate
    // the count while contributing no displayable name → bare "[callers:N]".
    // Filtering at the source keeps count and names consistent.
    let caller_rows: Vec<CallerRow> = caller_rows
        .into_iter()
        .filter(|r| r.in_name.as_deref().is_some_and(|n| !n.is_empty()))
        .collect();

    let caller_count = caller_rows.len() as u32;
    let distinct_caller_files: HashSet<&str> = caller_rows.iter().map(|r| r.in_file.as_str()).collect();
    let caller_file_count = distinct_caller_files.len() as u32;

    // Extract and sort caller names by proximity
    let caller_names = proximity_sorted_names(
        &caller_rows.iter()
            .filter_map(|r| r.in_name.as_ref().map(|n| (n.clone(), r.in_file.clone())))
            .collect::<Vec<_>>(),
        target_file,
    );

    // Fetch callees (what this symbol calls)
    let callee_rows: Vec<CalleeRow> = if schema_version >= 2 {
        db.query("SELECT out_file, out_name FROM calls WHERE in_name = $fqn")
            .bind(("fqn", fqn.to_string()))
            .await
            .ok()?
            .take(0)
            .ok()?
    } else {
        // Pre-schema-2 fallback: use short name
        let name = fqn.rsplit("::").next().unwrap_or(fqn);
        db.query("SELECT out_file, out_name FROM calls WHERE in_name = $name")
            .bind(("name", name.to_string()))
            .await
            .ok()?
            .take(0)
            .ok()?
    };

    // Same consistency guarantee as callers: drop edges with a NULL/empty
    // out_name so callee_count never exceeds the number of displayable names.
    let callee_rows: Vec<CalleeRow> = callee_rows
        .into_iter()
        .filter(|r| r.out_name.as_deref().is_some_and(|n| !n.is_empty()))
        .collect();

    let callee_count = callee_rows.len() as u32;
    let distinct_callee_files: HashSet<&str> = callee_rows.iter().map(|r| r.out_file.as_str()).collect();
    let callee_file_count = distinct_callee_files.len() as u32;

    let callee_names = proximity_sorted_names(
        &callee_rows.iter()
            .filter_map(|r| r.out_name.as_ref().map(|n| (n.clone(), r.out_file.clone())))
            .collect::<Vec<_>>(),
        target_file,
    );

    Some(CallerCalleeStats {
        caller_count,
        caller_file_count,
        caller_names,
        callee_count,
        callee_file_count,
        callee_names,
    })
}

/// Sort names by file proximity to target: same-file first, same-directory second,
/// then alphabetical. Dedup and cap at the first unique entries.
fn proximity_sorted_names(entries: &[(String, String)], target_file: &str) -> Vec<String> {
    use std::path::Path;

    let target_dir = Path::new(target_file)
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or("");

    // Deduplicate names
    let mut seen = HashSet::new();
    let mut scored: Vec<(u8, &str)> = Vec::new();
    for (name, file) in entries {
        // Extract short name (last :: segment)
        let short = name.rsplit("::").next().unwrap_or(name);
        if !seen.insert(short.to_string()) {
            continue;
        }
        let priority = if file == target_file {
            0 // same file
        } else {
            let file_dir = Path::new(file.as_str())
                .parent()
                .and_then(|p| p.to_str())
                .unwrap_or("");
            if file_dir == target_dir { 1 } else { 2 } // same dir vs. other
        };
        scored.push((priority, short));
    }
    scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored.into_iter().map(|(_, name)| name.to_string()).collect()
}

/// Format a caller tag string for MCP output.
/// `[callers: fn_a, fn_b, fn_c +N more]` when callers > 3
/// `[callers: fn_a, fn_b]` when callers <= 3
pub fn format_caller_tag(stats: &CallerCalleeStats) -> String {
    if stats.caller_count == 0 {
        return String::new();
    }
    let max_display = 30;
    let names = &stats.caller_names;
    let display_names: Vec<&str> = names.iter().take(max_display).map(|s| s.as_str()).collect();
    let remaining = stats.caller_count.saturating_sub(display_names.len() as u32);
    if remaining > 0 {
        format!(" [callers: {} +{} more]", display_names.join(", "), remaining)
    } else {
        format!(" [callers: {}]", display_names.join(", "))
    }
}

/// Format a callee tag string for MCP output.
/// `[calls: fn_x, fn_y +N more]` when callees > 3
pub fn format_callee_tag(stats: &CallerCalleeStats) -> String {
    if stats.callee_count == 0 {
        return String::new();
    }
    let max_display = 30;
    let names = &stats.callee_names;
    let display_names: Vec<&str> = names.iter().take(max_display).map(|s| s.as_str()).collect();
    let remaining = stats.callee_count.saturating_sub(display_names.len() as u32);
    if remaining > 0 {
        format!(" [calls: {} +{} more]", display_names.join(", "), remaining)
    } else {
        format!(" [calls: {}]", display_names.join(", "))
    }
}

/// Fetch stored chunk content, symbol short name, full FQN, and symbol kind
/// from whichever DB contains the file.
#[allow(clippy::result_large_err)]
async fn fetch_chunk_content(
    db_map: &HashMap<String, Surreal<Db>>,
    file: &str,
    line_start: u32,
    line_end: u32,
) -> (String, Option<String>, Option<String>, Option<String>) {
    let db = match find_db_for_file(db_map, file) {
        Some(db) => db,
        None => return (String::new(), None, None, None),
    };

    let rows: Result<Vec<ChunkContentRow>, _> = db
        .query(
            "SELECT content, symbol_ref FROM chunk \
             WHERE file = $file AND line_start = $ls AND line_end = $le LIMIT 1",
        )
        .bind(("file", file.to_string()))
        .bind(("ls", line_start as i64))
        .bind(("le", line_end as i64))
        .await
        .and_then(|mut r| r.take(0));

    match rows {
        Ok(rows) => {
            if let Some(row) = rows.into_iter().next() {
                // symbol_ref is stored as "symbol:⟨fqn⟩" — extract both full FQN and short name.
                let (symbol, fqn) = match row.symbol_ref.as_deref() {
                    Some(s) => {
                        let full_fqn = s.strip_prefix("symbol:⟨")
                            .and_then(|s| s.strip_suffix("⟩"))
                            .map(|f| f.to_string());
                        let short_name = full_fqn.as_deref()
                            .map(|fqn| fqn.rsplit("::").next().unwrap_or(fqn).to_string());
                        (short_name, full_fqn)
                    }
                    None => (None, None),
                };
                // Fetch symbol kind from the symbol table if we have an FQN.
                let symbol_kind = if let Some(ref fqn_str) = fqn {
                    fetch_symbol_kind(db, fqn_str).await
                } else {
                    None
                };
                (row.content, symbol, fqn, symbol_kind)
            } else {
                (String::new(), None, None, None)
            }
        }
        Err(e) => {
            warn!(error = %e, file = %file, "failed to fetch chunk content");
            (String::new(), None, None, None)
        }
    }
}

/// Fetch the `kind` field from the symbol table for a given FQN.
/// Returns None on any failure or if the symbol doesn't exist.
async fn fetch_symbol_kind(db: &Surreal<Db>, fqn: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct KindRow {
        kind: Option<String>,
    }
    let thing = surrealdb::sql::Thing::from((
        "symbol",
        surrealdb::sql::Id::String(fqn.to_string()),
    ));
    let rows: Vec<KindRow> = db
        .query("SELECT kind FROM $t")
        .bind(("t", thing))
        .await
        .ok()?
        .take(0)
        .ok()?;
    rows.into_iter().next().and_then(|r| r.kind)
}

/// Read lines [line_start, line_end] (1-based, inclusive) from the filesystem.
/// Returns formatted numbered lines: "10: fn main() {\n11: ..."
pub(crate) fn read_lines_from_fs(file: &str, line_start: u32, line_end: u32) -> Result<String> {
    let content = std::fs::read_to_string(file)?;
    let lines: Vec<&str> = content.lines().collect();
    let start_idx = (line_start.saturating_sub(1)) as usize;
    let end_idx = (line_end as usize).min(lines.len());
    if start_idx >= lines.len() {
        bail!("line_start {} out of range for file {}", line_start, file);
    }
    let numbered: String = lines[start_idx..end_idx]
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{}: {}", start_idx + i + 1, line))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(numbered)
}

/// Slice an already-numbered chunk text (produced by `read_lines_from_fs`,
/// first line == `chunk_start`) down to the absolute line range [s, e].
/// Both bounds are 1-based inclusive and assumed already clamped to the chunk.
pub(crate) fn slice_numbered(numbered: &str, chunk_start: u32, s: u32, e: u32) -> String {
    let lines: Vec<&str> = numbered.lines().collect();
    let from = s.saturating_sub(chunk_start) as usize;
    let to = (e.saturating_sub(chunk_start) as usize + 1).min(lines.len());
    if from >= lines.len() || from >= to {
        return numbered.to_string();
    }
    lines[from..to].join("\n")
}

#[cfg(test)]
mod tests {
    use super::slice_numbered;

    // chunk_start=100, text holds absolute lines 100..=104 (5 lines).
    fn sample() -> String {
        "100: a\n101: b\n102: c\n103: d\n104: e".to_owned()
    }

    #[test]
    fn slice_normal_in_bounds() {
        // Keep absolute 101..=103 → middle three lines.
        let out = slice_numbered(&sample(), 100, 101, 103);
        assert_eq!(out, "101: b\n102: c\n103: d");
    }

    #[test]
    fn slice_end_past_text_truncates_to_len() {
        // File shrank: text only has 5 lines but range asks up to 110.
        let out = slice_numbered(&sample(), 100, 102, 110);
        assert_eq!(out, "102: c\n103: d\n104: e");
    }

    #[test]
    fn slice_from_past_text_returns_whole() {
        // Start beyond the available text → bail to whole numbered text.
        let out = slice_numbered(&sample(), 100, 130, 140);
        assert_eq!(out, sample());
    }

    // --- proximity_sorted_names tests ---

    #[test]
    fn proximity_same_file_first() {
        use super::proximity_sorted_names;
        let entries = vec![
            ("module::other_fn".to_string(), "/src/other.rs".to_string()),
            ("module::same_fn".to_string(), "/src/main.rs".to_string()),
            ("module::alpha".to_string(), "/src/lib.rs".to_string()),
        ];
        let result = proximity_sorted_names(&entries, "/src/main.rs");
        // same-file first, then others sorted alphabetically
        assert_eq!(result[0], "same_fn");
    }

    #[test]
    fn proximity_same_dir_second() {
        use super::proximity_sorted_names;
        let entries = vec![
            ("pkg::far_fn".to_string(), "/other/far.rs".to_string()),
            ("pkg::near_fn".to_string(), "/src/sibling.rs".to_string()),
        ];
        let result = proximity_sorted_names(&entries, "/src/main.rs");
        // same-dir (priority 1) before other (priority 2)
        assert_eq!(result[0], "near_fn");
        assert_eq!(result[1], "far_fn");
    }

    #[test]
    fn proximity_deduplicates_names() {
        use super::proximity_sorted_names;
        let entries = vec![
            ("mod::dup".to_string(), "/src/a.rs".to_string()),
            ("mod2::dup".to_string(), "/src/b.rs".to_string()),
        ];
        let result = proximity_sorted_names(&entries, "/src/main.rs");
        // "dup" appears only once (deduped by short name)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "dup");
    }

    #[test]
    fn proximity_alphabetical_within_tier() {
        use super::proximity_sorted_names;
        let entries = vec![
            ("m::zebra".to_string(), "/other/z.rs".to_string()),
            ("m::alpha".to_string(), "/other/a.rs".to_string()),
            ("m::mid".to_string(), "/other/m.rs".to_string()),
        ];
        let result = proximity_sorted_names(&entries, "/src/main.rs");
        // All in tier 2 (different dir), sorted alphabetically
        assert_eq!(result, vec!["alpha", "mid", "zebra"]);
    }

    // --- format_caller_tag / format_callee_tag tests ---

    #[test]
    fn caller_tag_zero_callers_empty() {
        use super::{CallerCalleeStats, format_caller_tag};
        let stats = CallerCalleeStats {
            caller_count: 0,
            caller_file_count: 0,
            caller_names: vec![],
            callee_count: 0,
            callee_file_count: 0,
            callee_names: vec![],
        };
        assert_eq!(format_caller_tag(&stats), "");
    }

    #[test]
    fn caller_tag_one_to_three_all_shown() {
        use super::{CallerCalleeStats, format_caller_tag};
        let stats = CallerCalleeStats {
            caller_count: 2,
            caller_file_count: 2,
            caller_names: vec!["fn_a".into(), "fn_b".into()],
            callee_count: 0,
            callee_file_count: 0,
            callee_names: vec![],
        };
        assert_eq!(format_caller_tag(&stats), " [callers: fn_a, fn_b]");
    }

    #[test]
    fn caller_tag_four_plus_shows_more() {
        use super::{CallerCalleeStats, format_caller_tag};
        let stats = CallerCalleeStats {
            caller_count: 35,
            caller_file_count: 10,
            caller_names: (1..=35).map(|i| format!("fn_{i}")).collect(),
            callee_count: 0,
            callee_file_count: 0,
            callee_names: vec![],
        };
        // Shows first 30, then "+5 more"
        let tag = format_caller_tag(&stats);
        assert!(tag.starts_with(" [callers: fn_1, fn_2, fn_3"));
        assert!(tag.ends_with("+5 more]"));
    }

    #[test]
    fn callee_tag_zero_empty() {
        use super::{CallerCalleeStats, format_callee_tag};
        let stats = CallerCalleeStats {
            caller_count: 0,
            caller_file_count: 0,
            caller_names: vec![],
            callee_count: 0,
            callee_file_count: 0,
            callee_names: vec![],
        };
        assert_eq!(format_callee_tag(&stats), "");
    }

    #[test]
    fn callee_tag_four_plus_shows_more() {
        use super::{CallerCalleeStats, format_callee_tag};
        let stats = CallerCalleeStats {
            caller_count: 0,
            caller_file_count: 0,
            caller_names: vec![],
            callee_count: 35,
            callee_file_count: 8,
            callee_names: (1..=35).map(|i| format!("call_{i}")).collect(),
        };
        let tag = format_callee_tag(&stats);
        assert!(tag.starts_with(" [calls: call_1, call_2, call_3"));
        assert!(tag.ends_with("+5 more]"));
    }

    #[test]
    fn callee_tag_three_all_shown() {
        use super::{CallerCalleeStats, format_callee_tag};
        let stats = CallerCalleeStats {
            caller_count: 0,
            caller_file_count: 0,
            caller_names: vec![],
            callee_count: 3,
            callee_file_count: 2,
            callee_names: vec!["foo".into(), "bar".into(), "baz".into()],
        };
        assert_eq!(format_callee_tag(&stats), " [calls: foo, bar, baz]");
    }
}
