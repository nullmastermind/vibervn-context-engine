use std::collections::{HashMap, HashSet};

use anyhow::Result;
use serde::Deserialize;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tracing::warn;

use crate::query::find_db_for_file;
use crate::query::merger::MergeChunk;

/// An expanded chunk produced by BFS graph traversal.
pub struct ExpandedChunk {
    pub file: String,
    pub line_start: u32,
    pub line_end: u32,
    pub score: f32,
    pub content: String,
    pub symbol: Option<String>,
}

// ─── DB row types ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SymbolRow {
    /// Full FQN from `meta::id(id)` (file::scope::name), used as the BFS seed key.
    /// Matches the stored `calls.in_name` / `out_name` form so methods resolve.
    #[serde(default)]
    fqn: String,
    file: String,
    name: String,
    line_start: i64,
    line_end: i64,
}

#[derive(Deserialize)]
struct ChunkRow {
    file: String,
    line_start: i64,
    line_end: i64,
    content: String,
}

// ─── Graph expansion ──────────────────────────────────────────────────────

const CALLER_SCORE_FACTOR: f32 = 0.6;
const CALLEE_SCORE_FACTOR: f32 = 0.5;
const SCORE_FLOOR: f32 = 0.15;
const MAX_DEPTH: usize = 2;
const MAX_BONUS_CHUNKS: usize = 30;

/// Expand base search results via BFS over the call graph.
///
/// For each chunk in `base_chunks`, finds overlapping symbols, then BFS-expands
/// callers (score × 0.6) and callees (score × 0.5) up to 2 levels deep.
/// Returns up to MAX_BONUS_CHUNKS additional chunks.
///
/// `schema_version`: when >= 2, uses indexed `WHERE out_name=$name` queries.
/// When < 2 (migration in progress), falls back to the old link-deref query.
pub async fn graph_expand(
    base_chunks: &[MergeChunk],
    db_map: &HashMap<String, Surreal<Db>>,
    schema_version: u32,
) -> Vec<ExpandedChunk> {
    if db_map.is_empty() {
        return vec![];
    }

    let mut all_expanded: Vec<ExpandedChunk> = Vec::new();
    let mut global_seen: HashSet<String> = HashSet::new();

    let base_keys: HashSet<(String, u32, u32)> = base_chunks
        .iter()
        .map(|c| (c.file.clone(), c.line_start, c.line_end))
        .collect();

    'outer: for base_chunk in base_chunks {
        let db = match find_db_for_file(db_map, &base_chunk.file) {
            Some(db) => db,
            None => continue,
        };

        let overlapping = match query_overlapping_symbols(
            db,
            &base_chunk.file,
            base_chunk.line_start,
            base_chunk.line_end,
        )
        .await
        {
            Ok(syms) => syms,
            Err(e) => {
                warn!(error = %e, file = %base_chunk.file, "failed to query overlapping symbols");
                continue;
            }
        };

        if overlapping.is_empty() {
            continue;
        }

        let mut queue: Vec<(String, f32, usize)> = overlapping
            .iter()
            .map(|s| (strip_id_brackets(&s.fqn), base_chunk.score, 0))
            .collect();

        while let Some((fqn, score, depth)) = queue.pop() {
            if depth >= MAX_DEPTH {
                continue;
            }
            if all_expanded.len() >= MAX_BONUS_CHUNKS {
                break 'outer;
            }

            // Expand callers.
            let caller_score = score * CALLER_SCORE_FACTOR;
            if caller_score >= SCORE_FLOOR {
                let callers = query_callers(db, &fqn, schema_version).await.unwrap_or_default();
                for caller_fqn in callers {
                    if global_seen.contains(&caller_fqn) {
                        continue;
                    }
                    global_seen.insert(caller_fqn.clone());
                    if let Some(chunk) =
                        fetch_chunk_for_fqn(db, &caller_fqn, caller_score, &base_keys).await
                    {
                        if all_expanded.len() < MAX_BONUS_CHUNKS {
                            all_expanded.push(chunk);
                        }
                        queue.push((caller_fqn, caller_score, depth + 1));
                    }
                }
            }

            // Expand callees.
            let callee_score = score * CALLEE_SCORE_FACTOR;
            if callee_score >= SCORE_FLOOR {
                let callees = query_callees(db, &fqn, schema_version).await.unwrap_or_default();
                for callee_fqn in callees {
                    if global_seen.contains(&callee_fqn) {
                        continue;
                    }
                    global_seen.insert(callee_fqn.clone());
                    if let Some(chunk) =
                        fetch_chunk_for_fqn(db, &callee_fqn, callee_score, &base_keys).await
                    {
                        if all_expanded.len() < MAX_BONUS_CHUNKS {
                            all_expanded.push(chunk);
                        }
                        queue.push((callee_fqn, callee_score, depth + 1));
                    }
                }
            }
        }
    }

    all_expanded
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Strip the SurrealDB complex-ID wrapper `⟨…⟩` returned by `meta::id(id)`.
/// A record id `symbol:⟨/foo.cpp::Bar::baz⟩` projects as `⟨/foo.cpp::Bar::baz⟩`;
/// this recovers the plain FQN that `calls.in_name` / `out_name` store.
fn strip_id_brackets(id: &str) -> String {
    id.strip_prefix("⟨")
        .and_then(|s| s.strip_suffix("⟩"))
        .unwrap_or(id)
        .to_string()
}

async fn query_overlapping_symbols(
    db: &Surreal<Db>,
    file: &str,
    chunk_start: u32,
    chunk_end: u32,
) -> Result<Vec<SymbolRow>> {
    let rows: Vec<SymbolRow> = db
        .query(
            "SELECT meta::id(id) AS fqn, file, name, line_start, line_end FROM symbol \
             WHERE file = $file AND line_start <= $chunk_end AND line_end >= $chunk_start",
        )
        .bind(("file", file.to_string()))
        .bind(("chunk_end", chunk_end as i64))
        .bind(("chunk_start", chunk_start as i64))
        .await?
        .take(0)?;
    Ok(rows)
}

/// Query callers of the symbol identified by `fqn`.
///
/// Uses indexed `in_name`/`out_name` columns which now store full FQNs.
/// The `schema_version` parameter is retained for API compatibility but
/// the v1 link-deref fallback is no longer accurate since in_name/out_name
/// now store FQNs (v2+ schema). For v1 DBs the fallback path is kept
/// for graceful degradation.
async fn query_callers(db: &Surreal<Db>, fqn: &str, schema_version: u32) -> Result<Vec<String>> {
    #[derive(Deserialize)]
    struct Row {
        in_name: String,
    }

    let rows: Vec<Row> = if schema_version >= 2 {
        // Fast path: query by full FQN — in_name now stores FQN, indexed by idx_calls_in_name.
        db.query("SELECT in_name FROM calls WHERE out_name = $fqn LIMIT 20")
            .bind(("fqn", fqn.to_string()))
            .await?
            .take(0)?
    } else {
        // Slow fallback for v1 DBs (link-deref on the `in` record).
        let name = fqn.rsplit("::").next().unwrap_or(fqn);
        #[derive(Deserialize)]
        struct V1Row { in_file: String }
        let v1_rows: Vec<V1Row> = db
            .query("SELECT in_file FROM calls WHERE out.name = $name LIMIT 20")
            .bind(("name", name.to_string()))
            .await?
            .take(0)?;
        return Ok(v1_rows.into_iter().map(|r| format!("{}::{}", r.in_file, name)).collect());
    };

    let callers: Vec<String> = rows
        .into_iter()
        .map(|r| r.in_name)
        .collect();
    Ok(callers)
}

/// Query callees of the symbol identified by `fqn`.
///
/// Uses indexed `in_name`/`out_name` columns which now store full FQNs.
async fn query_callees(db: &Surreal<Db>, fqn: &str, schema_version: u32) -> Result<Vec<String>> {
    #[derive(Deserialize)]
    struct Row {
        out_name: String,
    }

    let rows: Vec<Row> = if schema_version >= 2 {
        // Fast path: query by full FQN — out_name now stores FQN, indexed by idx_calls_out_name.
        db.query("SELECT out_name FROM calls WHERE in_name = $fqn LIMIT 20")
            .bind(("fqn", fqn.to_string()))
            .await?
            .take(0)?
    } else {
        // Slow fallback for v1 DBs (link-deref on the `out` record).
        let name = fqn.rsplit("::").next().unwrap_or(fqn);
        #[derive(Deserialize)]
        struct V1Row { out_file: String }
        let v1_rows: Vec<V1Row> = db
            .query("SELECT out_file FROM calls WHERE in.name = $name LIMIT 20")
            .bind(("name", name.to_string()))
            .await?
            .take(0)?;
        return Ok(v1_rows.into_iter().map(|r| format!("{}::{}", r.out_file, name)).collect());
    };

    let callees: Vec<String> = rows
        .into_iter()
        .map(|r| r.out_name)
        .collect();
    Ok(callees)
}

async fn fetch_chunk_for_fqn(
    db: &Surreal<Db>,
    fqn: &str,
    score: f32,
    base_keys: &HashSet<(String, u32, u32)>,
) -> Option<ExpandedChunk> {
    // Resolve by full record id (the symbol id IS the FQN). This avoids the old
    // `rfind("::")` split, which mis-derived file_prefix for methods/namespaced
    // symbols (e.g. "x.cpp::Foo::bar" → file "x.cpp::Foo", matching no file) and
    // silently dropped every method-target expansion.
    let thing = surrealdb::sql::Thing::from((
        "symbol",
        surrealdb::sql::Id::String(fqn.to_string()),
    ));

    let sym_rows: Vec<SymbolRow> = db
        .query(
            "SELECT meta::id(id) AS fqn, file, name, line_start, line_end FROM symbol \
             WHERE id = $thing LIMIT 1",
        )
        .bind(("thing", thing))
        .await
        .ok()?
        .take(0)
        .ok()?;

    let sym = sym_rows.into_iter().next()?;

    let chunk_rows: Vec<ChunkRow> = db
        .query(
            "SELECT file, line_start, line_end, content FROM chunk \
             WHERE file = $file AND line_start <= $sym_end AND line_end >= $sym_start \
             ORDER BY line_start LIMIT 1",
        )
        .bind(("file", sym.file.clone()))
        .bind(("sym_end", sym.line_end))
        .bind(("sym_start", sym.line_start))
        .await
        .ok()?
        .take(0)
        .ok()?;

    let row = chunk_rows.into_iter().next()?;
    let ls = row.line_start as u32;
    let le = row.line_end as u32;

    if base_keys.contains(&(row.file.clone(), ls, le)) {
        return None;
    }

    Some(ExpandedChunk {
        file: row.file,
        line_start: ls,
        line_end: le,
        score,
        content: row.content,
        symbol: Some(sym.name),
    })
}
