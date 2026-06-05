use std::collections::HashMap;
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
pub async fn run_query(
    query: &str,
    top_k: usize,
    repo_filter: Option<&str>,
    voyage_client: &VoyageClient,
    index_engine: &Arc<IndexEngine>,
    repo_dbs: &Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    llm_client: Option<&LlmClient>,
) -> Result<QueryResult> {
    let total_start = Instant::now();

    // ── Step 1: Embed query ───────────────────────────────────────────────
    let embed_start = Instant::now();
    let embedding = voyage_client.embed_query(query).await?;
    let embed_ms = embed_start.elapsed().as_millis() as u64;

    if embedding.is_empty() {
        bail!("embed_query returned an empty vector");
    }

    // ── Step 2: Vector search ─────────────────────────────────────────────
    let search_start = Instant::now();
    // Search for 2× top_k so graph expansion has candidates to work with.
    let raw_results = index_engine.vector_search(&embedding, top_k * 2).await;
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
        let (content, symbol) =
            fetch_chunk_content(&db_map, &sr.chunk_id.file, sr.chunk_id.line_start, sr.chunk_id.line_end).await;
        base_chunks.push(MergeChunk {
            file: sr.chunk_id.file.clone(),
            line_start: sr.chunk_id.line_start,
            line_end: sr.chunk_id.line_end,
            score: sr.score,
            content,
            symbol,
        });
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
        });
    }

    // ── Step 5: Merge ─────────────────────────────────────────────────────
    let merge_start = Instant::now();
    let merged = merge_chunks(all_chunks, top_k);
    let merge_ms = merge_start.elapsed().as_millis() as u64;

    // ── Step 6: Rerank ────────────────────────────────────────────────────
    // Clone merged so we can preserve pre-rerank order for the response.
    let pre_rerank_merged = merged.clone();

    let rerank_start = Instant::now();
    let rerank_output = reranker::rerank(query, &merged, llm_client).await;
    let rerank_ms = rerank_start.elapsed().as_millis() as u64;

    // Reorder merged according to reranked_indices.
    let reranked_merged: Vec<MergeChunk> = rerank_output
        .reranked_indices
        .iter()
        .filter_map(|&idx| merged.get(idx).cloned())
        .collect();

    // ── Step 7: Format — re-read from filesystem ──────────────────────────
    let mut results: Vec<CodeResult> = Vec::with_capacity(reranked_merged.len());
    for chunk in &reranked_merged {
        let content = read_lines_from_fs(&chunk.file, chunk.line_start, chunk.line_end)
            .unwrap_or_else(|_| chunk.content.clone());
        results.push(CodeResult {
            file: chunk.file.clone(),
            line_start: chunk.line_start,
            line_end: chunk.line_end,
            score: chunk.score,
            content,
            symbol: chunk.symbol.clone(),
        });
    }

    let mut pre_rerank_results: Vec<CodeResult> = Vec::with_capacity(pre_rerank_merged.len());
    for chunk in &pre_rerank_merged {
        let content = read_lines_from_fs(&chunk.file, chunk.line_start, chunk.line_end)
            .unwrap_or_else(|_| chunk.content.clone());
        pre_rerank_results.push(CodeResult {
            file: chunk.file.clone(),
            line_start: chunk.line_start,
            line_end: chunk.line_end,
            score: chunk.score,
            content,
            symbol: chunk.symbol.clone(),
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
    })
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Fetch stored chunk content and symbol_ref from whichever DB contains the file.
#[allow(clippy::result_large_err)]
async fn fetch_chunk_content(
    db_map: &HashMap<String, Surreal<Db>>,
    file: &str,
    line_start: u32,
    line_end: u32,
) -> (String, Option<String>) {
    let db = match find_db_for_file(db_map, file) {
        Some(db) => db,
        None => return (String::new(), None),
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
                // symbol_ref is stored as "symbol:⟨fqn⟩" — extract the name part.
                let symbol = row.symbol_ref.as_deref().and_then(|s| {
                    s.strip_prefix("symbol:⟨")
                        .and_then(|s| s.strip_suffix("⟩"))
                        .map(|fqn| fqn.rsplit("::").next().unwrap_or(fqn).to_string())
                });
                (row.content, symbol)
            } else {
                (String::new(), None)
            }
        }
        Err(e) => {
            warn!(error = %e, file = %file, "failed to fetch chunk content");
            (String::new(), None)
        }
    }
}

/// Read lines [line_start, line_end] (1-based, inclusive) from the filesystem.
/// Returns formatted numbered lines: "10: fn main() {\n11: ..."
fn read_lines_from_fs(file: &str, line_start: u32, line_end: u32) -> Result<String> {
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
