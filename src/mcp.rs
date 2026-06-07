use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tokio::sync::RwLock;

use rmcp::{
    ServerHandler, tool, tool_handler, tool_router,
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    schemars, ErrorData,
};

use crate::config::Settings;
use crate::embedding::voyage::VoyageClient;
use crate::indexing::{IndexEngine, IndexState};
use crate::llm::LlmClient;
use crate::store;

// ─── Tool argument schema ─────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CodebaseRetrievalArgs {
    /// Natural-language description of the code or information you are looking for.
    pub information_request: String,
    /// Absolute path to the repository root. Must be a configured and indexed repository.
    pub workspace_full_path: String,
}

// ─── MCP handler ─────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct McpHandler {
    /// Used ONLY for `settings.json` access (config_path / ensure_dir_and_load).
    /// settings.json's location is fixed at `~/.vibervn/context-engine/settings.json`.
    home_dir: PathBuf,
    /// Boot-resolved data directory (CLI > env > `Settings.data_dir` > builtin
    /// default). Used for store/embedding paths. Captured once at startup —
    /// MUST NOT be re-read from `Settings` mid-run.
    data_dir: PathBuf,
    index_engine: Arc<IndexEngine>,
    repo_dbs: Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    settings: Arc<RwLock<crate::config::Settings>>,
    // Required by the #[tool_router] macro; suppress the dead_code lint.
    #[allow(dead_code)]
    tool_router: ToolRouter<McpHandler>,
}

#[tool_router]
impl McpHandler {
    pub fn new(
        home_dir: PathBuf,
        data_dir: PathBuf,
        index_engine: Arc<IndexEngine>,
        repo_dbs: Arc<RwLock<HashMap<String, Surreal<Db>>>>,
        settings: Arc<RwLock<crate::config::Settings>>,
    ) -> Self {
        Self {
            home_dir,
            data_dir,
            index_engine,
            repo_dbs,
            settings,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "codebase-retrieval",
        description = "\
Primary tool for semantic codebase search. Consider it the FIRST CHOICE when you need to \
find code by meaning rather than exact text. It:
1. Takes a natural-language description of the code you are looking for.
2. Embeds the query and retrieves the most relevant code snippets across the repository, \
then expands via the call graph and reranks the candidates.
3. Maintains a real-time index (updated by a file watcher), so results reflect the current \
state of files on disk.
4. Works across multiple programming languages.
5. Reflects only the current on-disk state — it has no knowledge of version control or history.

Use codebase-retrieval when:
* You don't know which files contain what you need.
* You want a high-level understanding of an area before editing.
* Before editing a file, to gather the symbols/classes/methods involved.

Good queries: \"Where is user authentication handled?\", \"What tests cover login?\", \
\"How is the database connected?\".
Prefer grep/exact-search when you already know the identifier and want ALL occurrences, \
or when searching within specific files.

Parameters:
* information_request: natural-language description of the code/information you need.
* workspace_full_path: absolute path to the repository root (must be a configured & indexed repository)."
    )]
    async fn codebase_retrieval(
        &self,
        Parameters(args): Parameters<CodebaseRetrievalArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        // Take an owned snapshot of settings — the guard is dropped before the .await below.
        let settings = self.settings.read().await.clone();
        let text = run_codebase_retrieval(
            &self.home_dir,
            &self.data_dir,
            &self.index_engine,
            &self.repo_dbs,
            &settings,
            &args.information_request,
            &args.workspace_full_path,
        )
        .await;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

#[tool_handler]
impl ServerHandler for McpHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(rmcp::model::Implementation::new(
                "context-engine-rs",
                env!("CARGO_PKG_VERSION"),
            ))
    }
}

// ─── Shared query funnel ──────────────────────────────────────────────────

/// Run the codebase retrieval tool logic.
///
/// Returns plain-text results or an error/guidance string. Never panics, never
/// returns `Err` — all failure paths produce a human-readable string.
///
/// `home_dir` locates the fixed `settings.json` file. `data_dir` is the
/// boot-resolved data directory used for the per-repo RocksDB / embedding cache
/// paths. They are intentionally NOT collapsed into a single parameter — see
/// `Settings.data_dir` for the bootstrap rationale (Shape C).
///
/// This is the single shared funnel used by both the MCP tool and the REST
/// endpoint (`POST /api/mcp-tool`), so their outputs are byte-identical.
pub async fn run_codebase_retrieval(
    home_dir: &Path,
    data_dir: &Path,
    index_engine: &Arc<IndexEngine>,
    repo_dbs: &Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    settings: &Settings,
    information_request: &str,
    workspace_full_path: &str,
) -> String {
    // 1. Validate workspace_full_path.
    let repo = workspace_full_path.trim();
    if repo.is_empty() {
        return "Error: workspace_full_path is required. Pass the full path to the workspace \
                (repository) root directory."
            .to_string();
    }

    // 2. Auto-register the repo if it is not yet configured.
    if !settings.repos.iter().any(|r| r == repo) {
        // Guard: path must exist and be a directory before we accept it.
        if !std::path::Path::new(repo).is_dir() {
            return format!(
                "Error: workspace '{}' does not exist or is not a directory.",
                repo
            );
        }

        // Best-effort: append to settings.json on disk so the repo survives restart.
        match crate::config::ensure_dir_and_load(home_dir) {
            Ok(mut disk) => {
                if !disk.repos.iter().any(|r| r == repo) {
                    disk.repos.push(repo.to_string());
                    disk.version = crate::config::CURRENT_VERSION;
                    let target = crate::config::config_path(home_dir);
                    if let Err(e) = crate::config::write_settings_atomic(&target, &disk) {
                        tracing::warn!(repo = %repo, error = %e, "failed to persist auto-added repo to settings.json");
                    }
                }
            }
            Err(e) => {
                tracing::warn!(repo = %repo, error = %e, "failed to read settings.json for auto-add");
            }
        }

        // Register at runtime: seed status entry + spawn watcher.
        // Falls through to the existing freshness/trigger/wait/query flow below.
        index_engine.register_repo(repo).await;
    }

    // 3. Confirm embedding keys are present.
    if settings.embedding.api_keys.is_empty() {
        return "Error: no embedding API keys configured. \
                Add a Voyage AI key in the Context Engine UI first."
            .to_string();
    }

    // 4. Open the repo DB and determine freshness from durable state.
    let db = match store::get_or_open(repo_dbs, data_dir, repo).await {
        Ok(d) => d,
        Err(e) => {
            return format!("Error: could not open index database: {e}");
        }
    };

    let chunk_count = store::ops::count_chunks(&db).await.unwrap_or(0);
    let last_indexed_ts = store::ops::get_meta(&db, "last_indexed_at").await.unwrap_or(None);

    let stale_threshold = chrono::Duration::days(settings.mcp_stale_after_days as i64);
    let is_usable = check_usable(chunk_count, &last_indexed_ts, stale_threshold);

    // 5. Check in-flight indexing state and trigger if needed.
    let current_status = index_engine.repo_status(repo).await;
    let currently_indexing = current_status
        .as_ref()
        .map(|s| s.state == IndexState::Indexing)
        .unwrap_or(false);

    let need_wait = if currently_indexing {
        // Already in flight — join the wait loop without triggering again.
        true
    } else if !is_usable {
        // Not usable and not currently indexing — trigger incremental.
        let _ = index_engine.trigger_index(repo).await;
        true
    } else {
        // Usable. If the durable stamp is missing or unparseable (legacy pre-timestamp
        // index, or corrupt stamp), kick a NON-BLOCKING refresh so a real timestamp gets
        // written for next time — but don't wait; serve current results immediately.
        let has_valid_stamp = last_indexed_ts
            .as_deref()
            .and_then(|ts| ts.parse::<chrono::DateTime<chrono::Utc>>().ok())
            .is_some();
        if !has_valid_stamp {
            let _ = index_engine.trigger_index(repo).await;
        }
        false
    };

    if need_wait {
        let deadline = Instant::now() + Duration::from_secs(settings.mcp_index_wait_secs);
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;

            let status = index_engine.repo_status(repo).await;
            let state = status.as_ref().map(|s| s.state.clone());
            let err_msg = status.as_ref().and_then(|s| s.error.clone());

            match state {
                Some(IndexState::Idle) => {
                    // Success — proceed to query with fresh results.
                    break;
                }
                Some(IndexState::Error) => {
                    // Indexing failed — return immediately without burning the budget.
                    let err = err_msg.unwrap_or_else(|| "unknown error".to_string());
                    if is_usable {
                        // Had usable data before — run query with stale data + note.
                        let prefix = format!(
                            "(index refresh failed: {}; showing previous results)\n\n",
                            err
                        );
                        return format!("{}{}", prefix, do_query(
                            index_engine, repo_dbs, settings,
                            information_request, repo,
                        ).await);
                    } else {
                        return format!(
                            "Error: indexing failed ({}). Use grep to search the codebase directly.",
                            err
                        );
                    }
                }
                _ => {
                    // Still indexing.
                    if Instant::now() >= deadline {
                        if is_usable {
                            let prefix = "(still indexing; results may be incomplete)\n\n";
                            return format!("{}{}", prefix, do_query(
                                index_engine, repo_dbs, settings,
                                information_request, repo,
                            ).await);
                        } else {
                            return "Codebase is indexing, use grep instead.".to_string();
                        }
                    }
                }
            }
        }
    }

    do_query(index_engine, repo_dbs, settings, information_request, repo).await
}

/// Returns true if the DB has chunks AND the durable timestamp is within the
/// staleness threshold. The durable timestamp is the source of truth — in-memory
/// `RepoStatus.last_indexed_at` is intentionally NOT consulted here.
///
/// Staleness rules:
/// * chunk_count == 0 → false (never indexed)
/// * chunk_count > 0, timestamp missing → true (pre-timestamp legacy index; chunks exist so usable)
/// * chunk_count > 0, timestamp unparseable → true (corrupt stamp but chunks exist; don't punish user)
/// * chunk_count > 0, age <= threshold → true (fresh)
/// * chunk_count > 0, age > threshold → false (genuinely stale)
fn check_usable(
    chunk_count: u64,
    last_indexed_ts: &Option<String>,
    threshold: chrono::Duration,
) -> bool {
    if chunk_count == 0 {
        return false;
    }
    match last_indexed_ts {
        // Legacy index (pre-timestamp upgrade) or missing stamp: chunks exist, assume usable.
        None => true,
        Some(ts) => match ts.parse::<chrono::DateTime<chrono::Utc>>() {
            // Unparseable/corrupt stamp: chunks exist, assume usable rather than punishing user.
            Err(_) => true,
            Ok(dt) => (chrono::Utc::now() - dt) <= threshold,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLD: fn() -> chrono::Duration = || chrono::Duration::days(7);

    fn ts_days_ago(n: i64) -> String {
        (chrono::Utc::now() - chrono::Duration::days(n)).to_rfc3339()
    }

    // 1. chunk_count == 0 → always false, regardless of timestamp.
    #[test]
    fn no_chunks_is_not_usable() {
        assert!(!check_usable(0, &None, THRESHOLD()));
        assert!(!check_usable(0, &Some(ts_days_ago(1)), THRESHOLD()));
    }

    // 2. chunk_count > 0, timestamp None → true (legacy backfill / pre-timestamp regression guard).
    #[test]
    fn legacy_index_no_timestamp_is_usable() {
        assert!(check_usable(1, &None, THRESHOLD()));
    }

    // 3. chunk_count > 0, timestamp 1 day ago (≤ 7d threshold) → true.
    #[test]
    fn fresh_timestamp_is_usable() {
        assert!(check_usable(100, &Some(ts_days_ago(1)), THRESHOLD()));
    }

    // 4. chunk_count > 0, timestamp 30 days ago (> 7d threshold) → false.
    #[test]
    fn old_timestamp_is_not_usable() {
        assert!(!check_usable(100, &Some(ts_days_ago(30)), THRESHOLD()));
    }

    // 5. chunk_count > 0, unparseable timestamp → true (corrupt stamp, chunks exist).
    #[test]
    fn unparseable_timestamp_is_usable() {
        assert!(check_usable(50, &Some("not-a-date".to_string()), THRESHOLD()));
    }

    // 6a. Boundary: just inside threshold (6 days ago ≤ 7d) → true.
    #[test]
    fn just_inside_threshold_is_usable() {
        assert!(check_usable(10, &Some(ts_days_ago(6)), THRESHOLD()));
    }

    // 6b. Boundary: just outside threshold (8 days ago > 7d) → false.
    #[test]
    fn just_outside_threshold_is_not_usable() {
        assert!(!check_usable(10, &Some(ts_days_ago(8)), THRESHOLD()));
    }
}

/// Execute the query pipeline and format the results as plain text.
/// Returns a string — never panics, never returns Err.
///
/// Note: neither `home_dir` nor `data_dir` is needed here — both DB opens and
/// vector access go through `index_engine` / `repo_dbs`, which were constructed
/// with the boot-resolved `data_dir`. Keeping the signature path-free
/// documents that this function never re-derives a base directory mid-run.
async fn do_query(
    index_engine: &Arc<IndexEngine>,
    repo_dbs: &Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    settings: &Settings,
    information_request: &str,
    repo: &str,
) -> String {
    let voyage_client = match VoyageClient::new(
        settings.embedding.model.clone(),
        settings.embedding.api_keys.clone(),
    ) {
        Ok(c) => c,
        Err(e) => return format!("Error: failed to create embedding client: {e}"),
    };

    let llm_client: Option<LlmClient> = LlmClient::new(&settings.llm);

    match crate::query::run_query(
        information_request,
        30,
        Some(repo),
        &voyage_client,
        index_engine,
        repo_dbs,
        settings.llm.rerank_min_prune_lines,
        llm_client.as_ref(),
        Duration::from_secs(settings.mcp_index_wait_secs),
    )
    .await
    {
        Err(e) => format!("Error: query failed: {e}"),
        Ok(result) => {
            if result.results.is_empty() {
                let rerank_rejected = result.rerank.as_ref().is_some_and(|r| {
                    !r.fallback_used && r.skip_reason.is_none() && !r.raw_response.is_empty()
                });
                if rerank_rejected {
                    return "No relevant code found. The indexed codebase does not appear to \
                            contain information related to this query. Please verify the query \
                            is relevant to this project, or try alternative tools such as Grep \
                            for exact-match searches."
                        .to_string();
                }
                return format!("No results found for: {information_request}");
            }
            let mut out = String::new();
            for r in &result.results {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&format!(
                    "{}#L{}-{}\n{}",
                    r.file, r.line_start, r.line_end, r.content
                ));
            }
            out
        }
    }
}
