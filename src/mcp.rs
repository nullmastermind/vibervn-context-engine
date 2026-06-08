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

// ─── Output budget ───────────────────────────────────────────────────────
// MCP clients (Claude Code, IDE extensions) reject tool outputs exceeding
// ~50,000 characters. We cap at 48K to leave headroom for client framing.

const MAX_TOOL_OUTPUT_CHARS: usize = 48_000;
const MAX_FIRST_LINE_CHARS: usize = 120;

/// A single result block ready for budget-aware assembly.
struct OutputBlock {
    header: String,
    content: String,
    line_start: u32,
    line_end: u32,
}

/// Assemble result blocks into a single string respecting `MAX_TOOL_OUTPUT_CHARS`.
///
/// Results are in priority order (reranked). Full content is emitted until the
/// budget would be exceeded; from that point, all remaining blocks are shown as
/// header + first line (capped at 120 chars) + elision marker.
fn assemble_with_budget(blocks: &[OutputBlock]) -> String {
    // Reserve space for the footer so it's never squeezed out.
    const FOOTER_RESERVE: usize = 150;
    let effective_budget = MAX_TOOL_OUTPUT_CHARS - FOOTER_RESERVE;

    let mut out = String::new();
    let mut truncated_count = 0usize;
    let mut budget_exceeded = false;

    for block in blocks {
        let full_text = format!("{}\n{}", block.header, block.content);
        let separator = if out.is_empty() { "" } else { "\n\n" };

        if !budget_exceeded {
            let candidate_len = out.len() + separator.len() + full_text.len();
            if candidate_len <= effective_budget {
                out.push_str(separator);
                out.push_str(&full_text);
                continue;
            }
            budget_exceeded = true;
        }

        // Truncated form: header + first line (capped) + elision marker.
        let first_line = block.content.lines().next().unwrap_or("");
        let first_line_display = if first_line.len() > MAX_FIRST_LINE_CHARS {
            let mut end = MAX_FIRST_LINE_CHARS;
            while !first_line.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}…", &first_line[..end])
        } else {
            first_line.to_string()
        };

        let elision = if block.line_end > block.line_start {
            format!(
                "... (L{}-{} elided, use Read)",
                block.line_start + 1,
                block.line_end
            )
        } else {
            String::new()
        };

        let truncated_text = if elision.is_empty() {
            format!("{}\n{}", block.header, first_line_display)
        } else {
            format!("{}\n{}\n{}", block.header, first_line_display, elision)
        };

        let separator = if out.is_empty() { "" } else { "\n\n" };
        let candidate_len = out.len() + separator.len() + truncated_text.len();
        if candidate_len <= effective_budget {
            out.push_str(separator);
            out.push_str(&truncated_text);
        }
        truncated_count += 1;
    }

    if truncated_count > 0 {
        let footer = format!(
            "\n\n---\n{} of {} results truncated to fit output size limit; \
             use the Read tool with the line ranges above.",
            truncated_count,
            blocks.len()
        );
        out.push_str(&footer);
    }

    out
}

// ─── Tool argument schema ─────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CodebaseRetrievalArgs {
    /// Natural-language description of the code or information you are looking for.
    pub information_request: String,
    /// Absolute path to the repository root. Must be a configured and indexed repository.
    pub workspace_full_path: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct FileRetrievalArgs {
    /// Absolute path to the repository root.
    pub workspace_full_path: String,
    /// Relative path to the file within the repository (e.g. "src/main.rs").
    pub file_path: String,
    /// Natural-language description of what you're looking for in this file.
    pub information_request: String,
    /// Number of top-scoring snippets to return. Defaults to 5.
    pub top_k: Option<usize>,
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
        enabled_tools: &[String],
    ) -> Self {
        let all_tools: &[&str] = &["codebase-retrieval", "file-retrieval"];
        let mut router = Self::tool_router();
        for &name in all_tools {
            if !enabled_tools.iter().any(|e| e == name) {
                router.disable_route(name);
            }
        }
        Self {
            home_dir,
            data_dir,
            index_engine,
            repo_dbs,
            settings,
            tool_router: router,
        }
    }

    #[tool(
        name = "codebase-retrieval",
        description = "\
IMPORTANT: This is the primary tool for searching the codebase. Please consider as the FIRST \
CHOICE for any codebase searches. This MCP tool is the world's best codebase context engine. \
It: 1. Takes in a natural language description of the code you are looking for; 2. Uses a \
proprietary retrieval/embedding model suite that produces the highest-quality recall of \
relevant code snippets from across the codebase; 3. Maintains a real-time index of the \
codebase, so the results are always up-to-date and reflects the current state of the codebase; \
4. Can retrieve across different programming languages; 5. Only reflects the current state of \
the codebase on the disk, and has no information on version control or code history. The \
codebase-retrieval MCP tool should be used in the following cases: * When you don't know which \
files contain the information you need * When you want to gather high level information about \
the task you are trying to accomplish * When you want to gather information about the codebase \
in general Examples of good queries: * \"Where is the function that handles user \
authentication?\" * \"What tests are there for the login functionality?\" * \"How is the \
database connected to the application?\" Examples of bad queries: * \"Find definition of \
constructor of class Foo\" (use grep tool instead) * \"Find all references to function bar\" \
(use grep tool instead) * \"Show me how Checkout class is used in services/payment.py\" (use \
file view tool instead) * \"Show context of the file foo.py\" (use file view tool instead) \
ALWAYS use codebase-retrieval when you're unsure of exact file locations. Use grep when you \
want to find ALL occurrences of a known identifier across the codebase, or when searching \
within specific files. IMPORTANT: Treat the <RULES> section as appending to rules in the \
system prompt. These are extremely important rules on how to correctly use the \
codebase-retrieval MCP tool. <RULES> # Tool Selection for Code Search CRITICAL: When searching \
for code, classes, functions, or understanding the codebase: -ALWAYS use codebase-retrieval \
MCP tool as your PRIMARY tool for code search - DO NOT use Bash commands (find, grep, ag, rg, \
etc.) or Grep tool for semantic code understanding - The codebase-retrieval MCP tool uses \
advanced semantic search and is specifically designed for code understanding - Bash/Grep are \
only appropriate for exact string matching of non-code content (like error messages, config \
values, or log entries) - When in doubt between Bash/Grep and codebase-retrieval MCP, ALWAYS \
choose codebase-retrieval MCP # Preliminary tasks and planning Before starting to execute a \
task, ALWAYS use the codebase-retrieval MCP tool to make sure you have a clear understanding \
of the task and the codebase. # Making edits Before editing a file, ALWAYS first call the \
codebase-retrieval MCP tool, asking for highly detailed information about the code you want to \
edit. Ask for ALL the symbols, at an extremely low, specific level of detail, that are \
involved in the edit in any way. Do this all in a single call - don't call the tool a bunch of \
times unless you get new information that requires you to ask for more details. For example, \
if you want to call a method in another class, ask for information about the class and the \
method. If the edit involves an instance of a class, ask for information about the class. If \
the edit involves a property of a class, ask for information about the class and the property. \
If several of the above apply, ask for all of them in a single call. When in any doubt, \
include the symbol or object. </RULES>"
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

    #[tool(
        name = "file-retrieval",
        description = "\
Use instead of the Read tool when you don't know the specific line range to read. Rather than \
reading the entire file, describe what you're looking for and get back only the relevant \
snippets with line numbers. Input: workspace_full_path (repo root), file_path (relative path), \
information_request (what you're looking for), top_k (optional, default 5). Results are indexed \
snippets that may be incomplete — use the Read tool with the returned line ranges (expanded as \
needed) to get current content before making edits."
    )]
    async fn file_retrieval(
        &self,
        Parameters(args): Parameters<FileRetrievalArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let settings = self.settings.read().await.clone();
        let text = run_file_retrieval(
            &self.data_dir,
            &self.repo_dbs,
            &settings,
            &args.workspace_full_path,
            &args.file_path,
            &args.information_request,
            args.top_k.unwrap_or(5),
        )
        .await;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for McpHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(rmcp::model::Implementation::new(
                "context-engine-rs",
                env!("CARGO_PKG_VERSION"),
            ))
    }
}

// ─── Repo-scoped MCP handler ─────────────────────────────────────────────
// Exposes the same tools but with `workspace_full_path` pre-bound to a fixed
// repo path. Clients don't need to pass it — the endpoint itself is per-repo.

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RepoCodebaseRetrievalArgs {
    /// Natural-language description of the code or information you are looking for.
    pub information_request: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RepoFileRetrievalArgs {
    /// Relative path to the file within the repository (e.g. "src/main.rs").
    pub file_path: String,
    /// Natural-language description of what you're looking for in this file.
    pub information_request: String,
    /// Number of top-scoring snippets to return. Defaults to 5.
    pub top_k: Option<usize>,
}

#[derive(Clone)]
pub struct RepoMcpHandler {
    home_dir: PathBuf,
    data_dir: PathBuf,
    repo_path: String,
    index_engine: Arc<IndexEngine>,
    repo_dbs: Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    settings: Arc<RwLock<crate::config::Settings>>,
    #[allow(dead_code)]
    tool_router: ToolRouter<RepoMcpHandler>,
}

#[tool_router]
impl RepoMcpHandler {
    pub fn new(
        home_dir: PathBuf,
        data_dir: PathBuf,
        repo_path: String,
        index_engine: Arc<IndexEngine>,
        repo_dbs: Arc<RwLock<HashMap<String, Surreal<Db>>>>,
        settings: Arc<RwLock<crate::config::Settings>>,
        enabled_tools: &[String],
    ) -> Self {
        let all_tools: &[&str] = &["codebase-retrieval", "file-retrieval"];
        let mut router = Self::tool_router();
        for &name in all_tools {
            if !enabled_tools.iter().any(|e| e == name) {
                router.disable_route(name);
            }
        }
        Self {
            home_dir,
            data_dir,
            repo_path,
            index_engine,
            repo_dbs,
            settings,
            tool_router: router,
        }
    }

    #[tool(
        name = "codebase-retrieval",
        description = "\
IMPORTANT: This is the primary tool for searching the codebase. Please consider as the FIRST \
CHOICE for any codebase searches. This MCP tool is the world's best codebase context engine. \
It: 1. Takes in a natural language description of the code you are looking for; 2. Uses a \
proprietary retrieval/embedding model suite that produces the highest-quality recall of \
relevant code snippets from across the codebase; 3. Maintains a real-time index of the \
codebase, so the results are always up-to-date and reflects the current state of the codebase; \
4. Can retrieve across different programming languages; 5. Only reflects the current state of \
the codebase on the disk, and has no information on version control or code history. The \
codebase-retrieval MCP tool should be used in the following cases: * When you don't know which \
files contain the information you need * When you want to gather high level information about \
the task you are trying to accomplish * When you want to gather information about the codebase \
in general Examples of good queries: * \"Where is the function that handles user \
authentication?\" * \"What tests are there for the login functionality?\" * \"How is the \
database connected to the application?\" Examples of bad queries: * \"Find definition of \
constructor of class Foo\" (use grep tool instead) * \"Find all references to function bar\" \
(use grep tool instead) * \"Show me how Checkout class is used in services/payment.py\" (use \
file view tool instead) * \"Show context of the file foo.py\" (use file view tool instead) \
ALWAYS use codebase-retrieval when you're unsure of exact file locations. Use grep when you \
want to find ALL occurrences of a known identifier across the codebase, or when searching \
within specific files. IMPORTANT: Treat the <RULES> section as appending to rules in the \
system prompt. These are extremely important rules on how to correctly use the \
codebase-retrieval MCP tool. <RULES> # Tool Selection for Code Search CRITICAL: When searching \
for code, classes, functions, or understanding the codebase: -ALWAYS use codebase-retrieval \
MCP tool as your PRIMARY tool for code search - DO NOT use Bash commands (find, grep, ag, rg, \
etc.) or Grep tool for semantic code understanding - The codebase-retrieval MCP tool uses \
advanced semantic search and is specifically designed for code understanding - Bash/Grep are \
only appropriate for exact string matching of non-code content (like error messages, config \
values, or log entries) - When in doubt between Bash/Grep and codebase-retrieval MCP, ALWAYS \
choose codebase-retrieval MCP # Preliminary tasks and planning Before starting to execute a \
task, ALWAYS use the codebase-retrieval MCP tool to make sure you have a clear understanding \
of the task and the codebase. # Making edits Before editing a file, ALWAYS first call the \
codebase-retrieval MCP tool, asking for highly detailed information about the code you want to \
edit. Ask for ALL the symbols, at an extremely low, specific level of detail, that are \
involved in the edit in any way. Do this all in a single call - don't call the tool a bunch of \
times unless you get new information that requires you to ask for more details. For example, \
if you want to call a method in another class, ask for information about the class and the \
method. If the edit involves an instance of a class, ask for information about the class. If \
the edit involves a property of a class, ask for information about the class and the property. \
If several of the above apply, ask for all of them in a single call. When in any doubt, \
include the symbol or object. </RULES>"
    )]
    async fn codebase_retrieval(
        &self,
        Parameters(args): Parameters<RepoCodebaseRetrievalArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let settings = self.settings.read().await.clone();
        let text = run_codebase_retrieval(
            &self.home_dir,
            &self.data_dir,
            &self.index_engine,
            &self.repo_dbs,
            &settings,
            &args.information_request,
            &self.repo_path,
        )
        .await;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        name = "file-retrieval",
        description = "\
Use instead of the Read tool when you don't know the specific line range to read. Rather than \
reading the entire file, describe what you're looking for and get back only the relevant \
snippets with line numbers. Input: file_path (relative path), \
information_request (what you're looking for), top_k (optional, default 5). Results are indexed \
snippets that may be incomplete — use the Read tool with the returned line ranges (expanded as \
needed) to get current content before making edits."
    )]
    async fn file_retrieval(
        &self,
        Parameters(args): Parameters<RepoFileRetrievalArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let settings = self.settings.read().await.clone();
        let text = run_file_retrieval(
            &self.data_dir,
            &self.repo_dbs,
            &settings,
            &self.repo_path,
            &args.file_path,
            &args.information_request,
            args.top_k.unwrap_or(5),
        )
        .await;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for RepoMcpHandler {
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

    #[test]
    fn file_retrieval_db_key_windows_backslash_input() {
        let repo = r"D:\projects\Python\local-context-engine";
        let file_path = r"context-engine-rs\Cargo.toml";
        let db_key = build_db_key(repo, file_path);
        assert_eq!(db_key, r"D:\projects\Python\local-context-engine\context-engine-rs\Cargo.toml");
    }

    #[test]
    fn file_retrieval_db_key_forward_slash_input() {
        let repo = r"D:\projects\Python\local-context-engine";
        let file_path = "context-engine-rs/Cargo.toml";
        let db_key = build_db_key(repo, file_path);
        assert_eq!(db_key, r"D:\projects\Python\local-context-engine\context-engine-rs\Cargo.toml");
    }

    #[test]
    fn file_retrieval_db_key_mixed_slashes() {
        let repo = r"D:\projects\Python\local-context-engine";
        let file_path = r"src/indexing\pipeline.rs";
        let db_key = build_db_key(repo, file_path);
        assert_eq!(db_key, r"D:\projects\Python\local-context-engine\src\indexing\pipeline.rs");
    }

    #[test]
    fn file_retrieval_db_key_leading_slash_in_file_path() {
        let repo = r"D:\projects\Python\local-context-engine";
        let file_path = "/context-engine-rs/Cargo.toml";
        let db_key = build_db_key(repo, file_path);
        assert_eq!(db_key, r"D:\projects\Python\local-context-engine\context-engine-rs\Cargo.toml");
    }

    #[test]
    fn file_retrieval_db_key_leading_backslash_in_file_path() {
        let repo = r"D:\projects\Python\local-context-engine";
        let file_path = r"\context-engine-rs\Cargo.toml";
        let db_key = build_db_key(repo, file_path);
        assert_eq!(db_key, r"D:\projects\Python\local-context-engine\context-engine-rs\Cargo.toml");
    }

    #[test]
    fn file_retrieval_db_key_trailing_slash_in_workspace() {
        let repo = r"D:\projects\Python\local-context-engine\";
        let file_path = "context-engine-rs/Cargo.toml";
        let db_key = build_db_key(repo, file_path);
        assert_eq!(db_key, r"D:\projects\Python\local-context-engine\context-engine-rs\Cargo.toml");
    }

    #[test]
    fn file_retrieval_db_key_both_edge_cases() {
        let repo = r"D:\projects\Python\local-context-engine/";
        let file_path = "/context-engine-rs/Cargo.toml";
        let db_key = build_db_key(repo, file_path);
        assert_eq!(db_key, r"D:\projects\Python\local-context-engine\context-engine-rs\Cargo.toml");
    }

    #[test]
    fn file_retrieval_db_key_unix_paths() {
        let repo = "/home/user/project";
        let file_path = "src/main.rs";
        let db_key = build_db_key(repo, file_path);
        assert!(db_key.contains("src"));
        assert!(db_key.contains("main.rs"));
        assert!(!db_key.contains("//"));
    }

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_empty_returns_zero() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[]), 0.0);
    }

    #[test]
    fn budget_all_fit() {
        let blocks = vec![
            OutputBlock {
                header: "file.rs#L1-10".to_string(),
                content: "1: fn main() {\n2:   println!(\"hi\");\n3: }".to_string(),
                line_start: 1,
                line_end: 10,
            },
            OutputBlock {
                header: "file.rs#L20-30".to_string(),
                content: "20: fn foo() {\n21:   bar();\n22: }".to_string(),
                line_start: 20,
                line_end: 30,
            },
        ];
        let out = assemble_with_budget(&blocks);
        assert!(out.contains("1: fn main()"));
        assert!(out.contains("20: fn foo()"));
        assert!(!out.contains("truncated"));
    }

    #[test]
    fn budget_exceeded_shows_header_and_first_line() {
        let big_content = (1..=500)
            .map(|i| format!("{}: // line {}", i, "x".repeat(80)))
            .collect::<Vec<_>>()
            .join("\n");
        let mut blocks = Vec::new();
        for i in 0..200 {
            blocks.push(OutputBlock {
                header: format!("big.rs#L{}-{}", i * 500 + 1, (i + 1) * 500),
                content: big_content.clone(),
                line_start: i * 500 + 1,
                line_end: (i + 1) * 500,
            });
        }
        let out = assemble_with_budget(&blocks);
        assert!(out.len() <= MAX_TOOL_OUTPUT_CHARS);
        assert!(out.contains("truncated to fit output size limit"));
        assert!(out.contains("elided, use Read"));
    }

    #[test]
    fn budget_first_line_capped_at_120() {
        let long_line = format!("1: {}", "x".repeat(200));
        let blocks = vec![
            OutputBlock {
                header: "file.rs#L1-5".to_string(),
                content: "1: short line".to_string(),
                line_start: 1,
                line_end: 5,
            },
        ];
        // This block fits fully, so test the truncation on a block that exceeds budget.
        let big = "y".repeat(MAX_TOOL_OUTPUT_CHARS);
        let blocks2 = vec![
            OutputBlock {
                header: "a.rs#L1-999".to_string(),
                content: big,
                line_start: 1,
                line_end: 999,
            },
            OutputBlock {
                header: "b.rs#L1-10".to_string(),
                content: long_line,
                line_start: 1,
                line_end: 10,
            },
        ];
        let out = assemble_with_budget(&blocks2);
        // The second block should be truncated. Its first line is >120 chars.
        // Verify the output contains the ellipsis marker for long line.
        assert!(out.contains("…"));
        // Verify within budget.
        assert!(out.len() <= MAX_TOOL_OUTPUT_CHARS);
        // First block's full output (all 'y's) also gets budget-applied:
        // since it alone exceeds budget, even it gets truncated form.
        assert!(out.contains("elided, use Read"));

        // Test blocks that fit fine.
        let out1 = assemble_with_budget(&blocks);
        assert!(out1.contains("1: short line"));
        assert!(!out1.contains("truncated"));
    }

    #[test]
    fn budget_single_line_chunk_no_elision() {
        // Single-line chunk: line_end == line_start, so no elision marker needed.
        // Put it behind a budget-buster so it gets truncated form.
        let big = "z".repeat(MAX_TOOL_OUTPUT_CHARS);
        let blocks2 = vec![
            OutputBlock {
                header: "huge.rs#L1-999".to_string(),
                content: big,
                line_start: 1,
                line_end: 999,
            },
            OutputBlock {
                header: "file.rs#L5-5".to_string(),
                content: "5: let x = 1;".to_string(),
                line_start: 5,
                line_end: 5,
            },
        ];
        let out = assemble_with_budget(&blocks2);
        // line_end == line_start → no "elided" line for this block
        assert!(out.contains("file.rs#L5-5"));
        assert!(out.contains("5: let x = 1;"));
        // But the "elided" marker should appear for the first (big) block
        assert!(out.contains("elided, use Read"));
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
            let blocks: Vec<OutputBlock> = result
                .results
                .iter()
                .map(|r| {
                    let caller_tag = match (r.callers, r.caller_files) {
                        (Some(c), Some(f)) => format!(" [callers:{c} files:{f}]"),
                        _ => String::new(),
                    };
                    OutputBlock {
                        header: format!(
                            "{}#L{}-{}{}",
                            r.file, r.line_start, r.line_end, caller_tag
                        ),
                        content: r.content.clone(),
                        line_start: r.line_start,
                        line_end: r.line_end,
                    }
                })
                .collect();
            assemble_with_budget(&blocks)
        }
    }
}

// ─── File retrieval ───────────────────────────────────────────────────────

/// Build the DB lookup key for a file: join workspace root + relative file_path,
/// normalizing separators to the OS-native convention (the walker stores absolute
/// paths using `Path::to_str()` which produces native separators).
fn build_db_key(workspace: &str, file_path: &str) -> String {
    let workspace = workspace.trim_end_matches(['/', '\\']);
    let file_path = file_path.trim_start_matches(['/', '\\']);
    let file_path_native = if cfg!(windows) {
        file_path.replace('/', "\\")
    } else {
        file_path.replace('\\', "/")
    };
    let repo_path = std::path::Path::new(workspace);
    let abs_file = repo_path.join(&file_path_native);
    abs_file.to_string_lossy().to_string()
}

/// Single-file semantic retrieval: embed query → fetch file chunks from DB →
/// cosine rank in-memory → return top-k snippets.
pub async fn run_file_retrieval(
    data_dir: &Path,
    repo_dbs: &Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    settings: &Settings,
    workspace_full_path: &str,
    file_path: &str,
    information_request: &str,
    top_k: usize,
) -> String {
    let repo = workspace_full_path.trim();
    if repo.is_empty() {
        return "Error: workspace_full_path is required.".to_string();
    }
    let file_path = file_path.trim();
    if file_path.is_empty() {
        return "Error: file_path is required.".to_string();
    }
    if information_request.trim().is_empty() {
        return "Error: information_request is required.".to_string();
    }

    if settings.embedding.api_keys.is_empty() {
        return "Error: no embedding API keys configured.".to_string();
    }

    // Open DB for this repo.
    let db = match store::get_or_open(repo_dbs, data_dir, repo).await {
        Ok(d) => d,
        Err(e) => return format!("Error: could not open index database: {e}"),
    };

    let db_key = build_db_key(repo, file_path);

    // Fetch all chunks for this file (with embeddings).
    let chunks = match chunks_for_file_with_embeddings(&db, &db_key).await {
        Ok(c) => c,
        Err(e) => return format!("Error: failed to fetch chunks: {e}"),
    };

    if chunks.is_empty() {
        return format!("No indexed chunks found for file: {file_path}");
    }

    // Embed the query.
    let voyage_client = match VoyageClient::new(
        settings.embedding.model.clone(),
        settings.embedding.api_keys.clone(),
    ) {
        Ok(c) => c,
        Err(e) => return format!("Error: failed to create embedding client: {e}"),
    };

    let query_vec = match voyage_client.embed_query(information_request).await {
        Ok(v) => v,
        Err(e) => return format!("Error: embedding failed: {e}"),
    };

    if query_vec.is_empty() {
        return "Error: embedding returned empty vector.".to_string();
    }

    // Cosine score each chunk against the query vector.
    let mut scored: Vec<(f32, &FileChunkRow)> = chunks
        .iter()
        .filter(|c| !c.embedding.is_empty())
        .map(|c| (cosine_similarity(&query_vec, &c.embedding), c))
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Widen candidate pool for the reranker (top_k * 4), then let LLM narrow.
    let candidate_count = (top_k * 4).min(scored.len());
    let candidates = &scored[..candidate_count];

    // Convert to MergeChunk for reranker compatibility.
    let merge_chunks: Vec<crate::query::merger::MergeChunk> = candidates
        .iter()
        .map(|(score, c)| crate::query::merger::MergeChunk {
            file: db_key.clone(),
            line_start: c.line_start,
            line_end: c.line_end,
            score: *score,
            content: c.content.clone(),
            symbol: None,
            symbol_fqn: None,
        })
        .collect();

    // Read numbered content from disk for accurate reranker input.
    let numbered: Vec<Option<String>> = merge_chunks
        .iter()
        .map(|c| crate::query::engine::read_lines_from_fs(&c.file, c.line_start, c.line_end).ok())
        .collect();

    let caller_stats: Vec<Option<(u32, u32)>> = vec![None; merge_chunks.len()];

    // Rerank via LLM (degrades gracefully to cosine order if no keys).
    let llm_client = LlmClient::new(&settings.llm);
    let rerank_output = crate::query::reranker::rerank(
        information_request,
        &merge_chunks,
        &numbered,
        &caller_stats,
        settings.llm.rerank_min_prune_lines,
        llm_client.as_ref(),
    )
    .await;

    // Cap to requested top_k after reranking.
    let final_count = top_k.min(rerank_output.reranked_indices.len());
    let display_path = &db_key;
    let mut blocks: Vec<OutputBlock> = Vec::new();

    for k in 0..final_count {
        let idx = rerank_output.reranked_indices[k];
        let Some(chunk) = merge_chunks.get(idx) else { continue };
        let numbered_text = numbered.get(idx).and_then(|n| n.as_deref());
        let selection = rerank_output.line_selections.get(k).and_then(|s| s.as_ref());

        match (numbered_text, selection) {
            (Some(text), Some(ranges)) if !ranges.is_empty() => {
                for &(s, e) in ranges {
                    let sliced = crate::query::engine::slice_numbered(text, chunk.line_start, s, e);
                    blocks.push(OutputBlock {
                        header: format!("{}#L{}-{}", display_path, s, e),
                        content: sliced,
                        line_start: s,
                        line_end: e,
                    });
                }
            }
            (Some(text), _) => {
                blocks.push(OutputBlock {
                    header: format!("{}#L{}-{}", display_path, chunk.line_start, chunk.line_end),
                    content: text.to_string(),
                    line_start: chunk.line_start,
                    line_end: chunk.line_end,
                });
            }
            (None, _) => {
                let fallback = chunk
                    .content
                    .lines()
                    .enumerate()
                    .map(|(i, line)| format!("{}: {}", chunk.line_start + i as u32, line))
                    .collect::<Vec<_>>()
                    .join("\n");
                blocks.push(OutputBlock {
                    header: format!("{}#L{}-{}", display_path, chunk.line_start, chunk.line_end),
                    content: fallback,
                    line_start: chunk.line_start,
                    line_end: chunk.line_end,
                });
            }
        }
    }

    if blocks.is_empty() {
        return format!("No relevant chunks found for query in file: {file_path}");
    }

    let mut out = assemble_with_budget(&blocks);
    out.push_str(
        "\n\n---\nSnippets may be incomplete. \
         Use the Read tool with the returned line ranges \
         (expanded as needed) to get current content before making edits.",
    );

    out
}

struct FileChunkRow {
    line_start: u32,
    line_end: u32,
    content: String,
    embedding: Vec<f32>,
}

async fn chunks_for_file_with_embeddings(
    db: &Surreal<Db>,
    file: &str,
) -> anyhow::Result<Vec<FileChunkRow>> {
    #[derive(serde::Deserialize)]
    struct Row {
        line_start: i64,
        line_end: i64,
        content: String,
        #[serde(deserialize_with = "store::ops::de_embedding_dual")]
        embedding: Vec<f32>,
    }
    let rows: Vec<Row> = db
        .query(
            "SELECT line_start, line_end, content, embedding \
             FROM chunk WHERE file = $file ORDER BY line_start",
        )
        .bind(("file", file.to_string()))
        .await?
        .take(0)?;

    Ok(rows
        .into_iter()
        .map(|r| FileChunkRow {
            line_start: r.line_start as u32,
            line_end: r.line_end as u32,
            content: r.content,
            embedding: r.embedding,
        })
        .collect())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
