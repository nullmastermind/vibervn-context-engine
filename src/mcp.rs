// Pre-existing layout: a few helpers live AFTER the inline #[cfg(test)]
// `tests` module rather than before it. Clippy flags this as
// `items_after_test_module`. Reordering the file is out of scope for the
// current change; suppress the lint at module level.
#![allow(clippy::items_after_test_module)]

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
#[derive(Default)]
struct OutputBlock {
    header: String,
    content: String,
    file: String,
    line_start: u32,
    line_end: u32,
    callers: Option<u32>,
    caller_files: Option<u32>,
    caller_names: Vec<String>,
    callee_names: Vec<String>,
    callees: Option<u32>,
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

/// Merge output blocks from the same file whose line ranges overlap or are
/// adjacent (next.line_start <= current.line_end + 1). Merged content is
/// re-read from the filesystem; if the read fails, original content strings
/// are concatenated with line-number dedup.
///
/// Preserves first-occurrence position: the merged block occupies the slot of
/// the earliest block in its file group. Blocks from different files pass
/// through unchanged.
fn merge_overlapping_blocks(blocks: Vec<OutputBlock>) -> Vec<OutputBlock> {
    if blocks.len() <= 1 {
        return blocks;
    }

    // Group by file. Normalize path separators for grouping on Windows
    // (the index stores native `\` but sub-query paths may use `/`).
    let normalize_key = |file: &str| -> String {
        if cfg!(windows) {
            file.replace('/', "\\")
        } else {
            file.to_string()
        }
    };

    // Group by normalized file key. Collect (original_index, block).
    let mut by_file: std::collections::HashMap<String, Vec<(usize, OutputBlock)>> =
        std::collections::HashMap::new();
    for (i, block) in blocks.into_iter().enumerate() {
        let key = normalize_key(&block.file);
        by_file.entry(key).or_default().push((i, block));
    }

    // Merge within each file group.
    let mut positioned: Vec<(usize, OutputBlock)> = Vec::new();

    for (_file, mut group) in by_file {
        if group.len() == 1 {
            let (idx, block) = group.remove(0);
            positioned.push((idx, block));
            continue;
        }

        // Sort by line_start within file.
        group.sort_unstable_by_key(|(_, b)| b.line_start);

        // Merge pass: accumulate (min_orig_idx, block, original_contents).
        // min_orig_idx tracks the earliest original position of any block
        // that was merged into this entry — used for output ordering.
        let mut merged: Vec<(usize, OutputBlock, Vec<String>)> = Vec::new();

        for (orig_idx, mut next) in group {
            if let Some((min_idx, current, originals)) = merged.last_mut() {
                if next.line_start <= current.line_end + 1 {
                    current.line_end = current.line_end.max(next.line_end);
                    *min_idx = (*min_idx).min(orig_idx);
                    // Combine caller/callee stats: the two merged blocks usually
                    // belong to DIFFERENT symbols (e.g. an import region vs. a
                    // function), so the count and its names MUST travel together —
                    // adopt them as an atomic triple/pair from whichever block has
                    // the higher count. If we bumped only the count (old behavior),
                    // a merged block could carry callers=Some(N) with empty names,
                    // tripping the `names.is_empty()` fallback in
                    // format_enriched_caller_tag and emitting a bare "[callers:N]".
                    if next.callers.unwrap_or(0) > current.callers.unwrap_or(0) {
                        current.callers = next.callers;
                        current.caller_files = next.caller_files;
                        current.caller_names = std::mem::take(&mut next.caller_names);
                    }
                    if next.callees.unwrap_or(0) > current.callees.unwrap_or(0) {
                        current.callees = next.callees;
                        current.callee_names = std::mem::take(&mut next.callee_names);
                    }
                    originals.push(next.content);
                } else {
                    let content_snapshot = next.content.clone();
                    merged.push((orig_idx, next, vec![content_snapshot]));
                }
            } else {
                let content_snapshot = next.content.clone();
                merged.push((orig_idx, next, vec![content_snapshot]));
            }
        }

        // Rebuild content and header for merged blocks.
        for (_, block, originals) in &mut merged {
            if originals.len() > 1 {
                // Multiple blocks were merged — try FS re-read for the full range.
                match crate::query::engine::read_lines_from_fs(
                    &block.file,
                    block.line_start,
                    block.line_end,
                ) {
                    Ok(text) => block.content = text,
                    Err(_) => {
                        // Fallback: union original content lines, dedup by
                        // line-number prefix, sort by line number.
                        block.content = merge_content_fallback(originals);
                    }
                }
            }
            // Rebuild header with updated range + enriched caller/callee tags.
            let caller_tag = format_enriched_caller_tag(
                block.callers, &block.caller_names, block.caller_files,
            );
            let callee_tag = format_enriched_callee_tag(
                block.callees, &block.callee_names,
            );
            block.header = format!(
                "{}#L{}-{}{}{}",
                block.file, block.line_start, block.line_end, caller_tag, callee_tag
            );
        }

        // Use the tracked min_orig_idx for output ordering.
        for (min_idx, block, _) in merged {
            positioned.push((min_idx, block));
        }
    }

    // Sort by the position index to restore original priority order.
    positioned.sort_by_key(|(idx, _)| *idx);
    positioned.into_iter().map(|(_, b)| b).collect()
}

/// Fallback content merge: union all numbered lines from the original content
/// strings, dedup by line number, sort ascending. Only used when the FS re-read
/// fails (file moved/deleted mid-query).
fn merge_content_fallback(originals: &[String]) -> String {
    let mut by_lineno: std::collections::BTreeMap<u32, &str> = std::collections::BTreeMap::new();
    for content in originals {
        for line in content.lines() {
            if let Some(colon_pos) = line.find(':')
                && let Ok(num) = line[..colon_pos].trim().parse::<u32>()
            {
                by_lineno.entry(num).or_insert(line);
            }
        }
    }
    if by_lineno.is_empty() {
        return originals.join("\n");
    }
    by_lineno.values().copied().collect::<Vec<_>>().join("\n")
}

// ─── Tool argument schema ─────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CodebaseRetrievalArgs {
    /// Natural-language description of the code or information you are looking for.
    pub information_request: String,
    /// Absolute path to the repository root. Must be a configured and indexed repository.
    pub workspace_full_path: String,
    /// Optional: filter results to specific symbol kinds (e.g. ["function", "class"]).
    #[serde(default)]
    pub filter_kind: Option<Vec<String>>,
    /// Optional: filter results to specific languages (e.g. ["rust", "typescript"]).
    #[serde(default)]
    pub filter_lang: Option<Vec<String>>,
    /// Optional: filter results to files matching this path substring.
    #[serde(default)]
    pub filter_path: Option<String>,
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
CHOICE for any codebase searches. It performs semantic code search: 1. Takes in a natural \
language description of the code you are looking for; 2. Retrieves the most relevant code \
snippets across the codebase using embedding-based vector search, call-graph expansion of the \
matched symbols, and optional LLM reranking; 3. Maintains a real-time index of the codebase, \
so results reflect the current state of the code on disk; 4. Can retrieve across different \
programming languages; 5. Only reflects the current state of the codebase on the disk, and \
has no information on version control or code history. The \
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
embedding-based semantic search and is specifically designed for code understanding - Bash/Grep are \
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
        // Build augmented query with structured filter params as inline prefixes
        let augmented_query = build_augmented_query(
            &args.information_request,
            args.filter_kind.as_deref(),
            args.filter_lang.as_deref(),
            args.filter_path.as_deref(),
        );
        let text = run_codebase_retrieval(
            &self.home_dir,
            &self.data_dir,
            &self.index_engine,
            &self.repo_dbs,
            &settings,
            &augmented_query,
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
CHOICE for any codebase searches. It performs semantic code search: 1. Takes in a natural \
language description of the code you are looking for; 2. Retrieves the most relevant code \
snippets across the codebase using embedding-based vector search, call-graph expansion of the \
matched symbols, and optional LLM reranking; 3. Maintains a real-time index of the codebase, \
so results reflect the current state of the code on disk; 4. Can retrieve across different \
programming languages; 5. Only reflects the current state of the codebase on the disk, and \
has no information on version control or code history. The \
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
embedding-based semantic search and is specifically designed for code understanding - Bash/Grep are \
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
/// Choose the message for a query that produced no result blocks, distinguishing a
/// transient *warming* shard (retry) from a genuine empty ("no results"). Pure
/// function of the three signals so it is unit-testable without a live query.
///
/// Precedence: `warming` wins — an empty result while the shard is still loading
/// must NOT be reported as "no results" (the index is complete on disk). Only when
/// the shard is resident (`warming=false`) do we report a genuine empty, with the
/// rerank-rejected wording when the reranker actively rejected all candidates.
fn select_empty_or_warming_message(
    warming: bool,
    rerank_rejected: bool,
    information_request: &str,
) -> String {
    if warming {
        return "The index for this workspace is still warming (loading into memory). \
                It is complete on disk — retry the same request in a few seconds."
            .to_string();
    }
    if rerank_rejected {
        return "No relevant code found. The indexed codebase does not appear to \
                contain information related to this query. Please verify the query \
                is relevant to this project, or try alternative tools such as Grep \
                for exact-match searches."
            .to_string();
    }
    format!("No results found for: {information_request}")
}

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
    let repo = &crate::store::normalize_repo_path(repo);

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
    let db = match store::get_or_open(repo_dbs, data_dir, repo, settings.repo_generation(repo)).await {
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
                file: "file.rs".to_string(),
                line_start: 1,
                line_end: 10,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
            OutputBlock {
                header: "file.rs#L20-30".to_string(),
                content: "20: fn foo() {\n21:   bar();\n22: }".to_string(),
                file: "file.rs".to_string(),
                line_start: 20,
                line_end: 30,
            callers: None,
            caller_files: None,
                    ..Default::default()
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
                file: "big.rs".to_string(),
                line_start: i * 500 + 1,
                line_end: (i + 1) * 500,
            callers: None,
            caller_files: None,
                    ..Default::default()
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
                file: "file.rs".to_string(),
                line_start: 1,
                line_end: 5,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
        ];
        // This block fits fully, so test the truncation on a block that exceeds budget.
        let big = "y".repeat(MAX_TOOL_OUTPUT_CHARS);
        let blocks2 = vec![
            OutputBlock {
                header: "a.rs#L1-999".to_string(),
                content: big,
                file: "a.rs".to_string(),
                line_start: 1,
                line_end: 999,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
            OutputBlock {
                header: "b.rs#L1-10".to_string(),
                content: long_line,
                file: "b.rs".to_string(),
                line_start: 1,
                line_end: 10,
            callers: None,
            caller_files: None,
                    ..Default::default()
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
                file: "huge.rs".to_string(),
                line_start: 1,
                line_end: 999,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
            OutputBlock {
                header: "file.rs#L5-5".to_string(),
                content: "5: let x = 1;".to_string(),
                file: "file.rs".to_string(),
                line_start: 5,
                line_end: 5,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
        ];
        let out = assemble_with_budget(&blocks2);
        // line_end == line_start → no "elided" line for this block
        assert!(out.contains("file.rs#L5-5"));
        assert!(out.contains("5: let x = 1;"));
        // But the "elided" marker should appear for the first (big) block
        assert!(out.contains("elided, use Read"));
    }

    #[test]
    fn merge_blocks_no_overlap() {
        let blocks = vec![
            OutputBlock {
                header: "a.rs#L1-10".into(),
                content: "1: aaa".into(),
                file: "a.rs".into(),
                line_start: 1,
                line_end: 10,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
            OutputBlock {
                header: "a.rs#L20-30".into(),
                content: "20: bbb".into(),
                file: "a.rs".into(),
                line_start: 20,
                line_end: 30,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
        ];
        let merged = merge_overlapping_blocks(blocks);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].line_start, 1);
        assert_eq!(merged[1].line_start, 20);
    }

    #[test]
    fn merge_blocks_overlap_same_file() {
        let blocks = vec![
            OutputBlock {
                header: "a.rs#L1-50".into(),
                content: "1: aaa\n2: bbb".into(),
                file: "a.rs".into(),
                line_start: 1,
                line_end: 50,
                callers: None,
                caller_files: None,
                    ..Default::default()
            },
            OutputBlock {
                header: "a.rs#L26-75".into(),
                content: "26: ccc\n27: ddd".into(),
                file: "a.rs".into(),
                line_start: 26,
                line_end: 75,
                callers: None,
                caller_files: None,
                    ..Default::default()
            },
        ];
        let merged = merge_overlapping_blocks(blocks);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].line_start, 1);
        assert_eq!(merged[0].line_end, 75);
        // Header uses the same file path as inputs + merged range, no caller tag.
        assert_eq!(merged[0].header, "a.rs#L1-75");
    }

    #[test]
    fn merge_blocks_combines_caller_tags() {
        let blocks = vec![
            OutputBlock {
                header: "a.rs#L1-50 [callers:3 files:2]".into(),
                content: "1: aaa".into(),
                file: "a.rs".into(),
                line_start: 1,
                line_end: 50,
                callers: Some(3),
                caller_files: Some(2),
                    ..Default::default()
            },
            OutputBlock {
                header: "a.rs#L26-75 [callers:7 files:4]".into(),
                content: "26: bbb".into(),
                file: "a.rs".into(),
                line_start: 26,
                line_end: 75,
                callers: Some(7),
                caller_files: Some(4),
                    ..Default::default()
            },
        ];
        let merged = merge_overlapping_blocks(blocks);
        assert_eq!(merged.len(), 1);
        // Caller stats: max(3,7)=7, max(2,4)=4
        assert_eq!(merged[0].callers, Some(7));
        assert_eq!(merged[0].caller_files, Some(4));
        // Header includes the combined caller tag (count-only format for merged blocks).
        assert_eq!(merged[0].header, "a.rs#L1-75 [callers:7]");
    }

    #[test]
    fn merge_blocks_carries_caller_and_callee_names() {
        // Regression: the names MUST travel with the count through a merge.
        // Block A is name-less (the import region), Block B carries the real
        // symbol's counts AND names. Old merge logic bumped only the counts,
        // leaving A's empty names → bare "[callers:N]". Now the higher-count
        // block's names are adopted atomically with its count.
        let blocks = vec![
            OutputBlock {
                file: "a.rs".into(),
                content: "1: use foo;".into(),
                line_start: 1,
                line_end: 50,
                callers: None,
                caller_names: vec![],
                callees: None,
                callee_names: vec![],
                ..Default::default()
            },
            OutputBlock {
                file: "a.rs".into(),
                content: "26: fn x".into(),
                line_start: 26,
                line_end: 75,
                callers: Some(2),
                caller_files: Some(1),
                caller_names: vec!["foo".into(), "bar".into()],
                callees: Some(1),
                callee_names: vec!["baz".into()],
                ..Default::default()
            },
        ];
        let merged = merge_overlapping_blocks(blocks);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].callers, Some(2));
        assert_eq!(merged[0].caller_names, vec!["foo".to_string(), "bar".to_string()]);
        assert_eq!(merged[0].callees, Some(1));
        assert_eq!(merged[0].callee_names, vec!["baz".to_string()]);
        // Header renders the NAMED form, not the bare count fallback.
        // (a.rs doesn't exist on disk, so the multi-block merge falls back to a
        // content union — only the header tags matter for this assertion.)
        assert!(
            merged[0].header.contains("[callers: foo, bar]"),
            "header missing named callers: {}",
            merged[0].header
        );
        assert!(
            merged[0].header.contains("[calls: baz]"),
            "header missing named callees: {}",
            merged[0].header
        );
        assert!(
            !merged[0].header.contains("[callers:2]"),
            "header fell back to bare count: {}",
            merged[0].header
        );
    }

    #[test]
    fn merge_blocks_adjacent() {
        let blocks = vec![
            OutputBlock {
                header: "a.rs#L1-10".into(),
                content: "1: x".into(),
                file: "a.rs".into(),
                line_start: 1,
                line_end: 10,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
            OutputBlock {
                header: "a.rs#L11-20".into(),
                content: "11: y".into(),
                file: "a.rs".into(),
                line_start: 11,
                line_end: 20,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
        ];
        let merged = merge_overlapping_blocks(blocks);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].line_start, 1);
        assert_eq!(merged[0].line_end, 20);
    }

    #[test]
    fn merge_blocks_different_files_no_merge() {
        let blocks = vec![
            OutputBlock {
                header: "a.rs#L1-50".into(),
                content: "1: aaa".into(),
                file: "a.rs".into(),
                line_start: 1,
                line_end: 50,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
            OutputBlock {
                header: "b.rs#L1-50".into(),
                content: "1: bbb".into(),
                file: "b.rs".into(),
                line_start: 1,
                line_end: 50,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
        ];
        let merged = merge_overlapping_blocks(blocks);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn merge_blocks_preserves_priority_order() {
        let blocks = vec![
            OutputBlock {
                header: "b.rs#L1-10".into(),
                content: "1: first".into(),
                file: "b.rs".into(),
                line_start: 1,
                line_end: 10,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
            OutputBlock {
                header: "a.rs#L1-50".into(),
                content: "1: second".into(),
                file: "a.rs".into(),
                line_start: 1,
                line_end: 50,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
            OutputBlock {
                header: "a.rs#L26-75".into(),
                content: "26: third".into(),
                file: "a.rs".into(),
                line_start: 26,
                line_end: 75,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
            OutputBlock {
                header: "b.rs#L20-30".into(),
                content: "20: fourth".into(),
                file: "b.rs".into(),
                line_start: 20,
                line_end: 30,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
        ];
        let merged = merge_overlapping_blocks(blocks);
        // b.rs: L1-10 and L20-30 not overlapping → 2 blocks
        // a.rs: L1-50 and L26-75 overlap → 1 merged block
        assert_eq!(merged.len(), 3);
        // b.rs appeared first (index 0), so its blocks come first
        assert_eq!(merged[0].file, "b.rs");
        assert_eq!(merged[0].line_start, 1);
        // a.rs appeared at index 1
        assert_eq!(merged[1].file, "a.rs");
        assert_eq!(merged[1].line_start, 1);
        assert_eq!(merged[1].line_end, 75);
        // b.rs second block at index 3 → comes after a.rs
        assert_eq!(merged[2].file, "b.rs");
        assert_eq!(merged[2].line_start, 20);
    }

    #[test]
    fn merge_blocks_fallback_preserves_content_on_fs_failure() {
        // Use a non-existent file path so read_lines_from_fs will fail,
        // exercising the fallback content-merge path.
        let blocks = vec![
            OutputBlock {
                header: "/nonexistent/z.rs#L1-50".into(),
                content: "1: aaa\n2: bbb\n3: ccc".into(),
                file: "/nonexistent/z.rs".into(),
                line_start: 1,
                line_end: 50,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
            OutputBlock {
                header: "/nonexistent/z.rs#L26-75".into(),
                content: "2: bbb\n26: ddd\n27: eee".into(),
                file: "/nonexistent/z.rs".into(),
                line_start: 26,
                line_end: 75,
            callers: None,
            caller_files: None,
                    ..Default::default()
            },
        ];
        let merged = merge_overlapping_blocks(blocks);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].line_start, 1);
        assert_eq!(merged[0].line_end, 75);
        assert!(merged[0].header.contains("L1-75"));
        // Fallback should preserve original lines, deduped by line number.
        assert!(merged[0].content.contains("1: aaa"));
        assert!(merged[0].content.contains("2: bbb"));
        assert!(merged[0].content.contains("3: ccc"));
        assert!(merged[0].content.contains("26: ddd"));
        assert!(merged[0].content.contains("27: eee"));
        // Line "2: bbb" appeared in both blocks but should only appear once.
        assert_eq!(merged[0].content.matches("2: bbb").count(), 1);
    }

    // ─── select_empty_or_warming_message (warming-vs-empty signal) ───────────

    #[test]
    fn warming_message_takes_precedence_and_says_retry() {
        // warming=true → retry message, regardless of rerank_rejected.
        for rr in [false, true] {
            let msg = select_empty_or_warming_message(true, rr, "find the parser");
            assert!(msg.contains("warming"), "warming msg must mention warming: {msg}");
            assert!(msg.to_lowercase().contains("retry"), "must tell caller to retry: {msg}");
            // MUST NOT use the genuine-empty wording.
            assert!(!msg.contains("No results found"), "warming must not say 'No results found'");
            assert!(!msg.contains("No relevant code found"), "warming must not say 'No relevant code found'");
        }
    }

    #[test]
    fn genuine_empty_resident_shard_keeps_existing_wording() {
        // warming=false, not rejected → the unchanged "No results found for: <q>".
        let msg = select_empty_or_warming_message(false, false, "find the parser");
        assert_eq!(msg, "No results found for: find the parser");

        // warming=false, rerank actively rejected → the unchanged rerank-rejected wording.
        let msg = select_empty_or_warming_message(false, true, "find the parser");
        assert!(msg.starts_with("No relevant code found."), "rerank-rejected wording preserved: {msg}");
        assert!(!msg.contains("warming"), "genuine empty must not mention warming");
    }
}

/// Build an augmented query string that prepends structured filter params as inline
/// filter prefixes (e.g. `kind:function lang:rust path:src/ <original query>`).
/// The `run_query` filter parser will strip these back out before embedding.
fn build_augmented_query(
    information_request: &str,
    filter_kind: Option<&[String]>,
    filter_lang: Option<&[String]>,
    filter_path: Option<&str>,
) -> String {
    let mut prefixes = Vec::new();
    if let Some(kinds) = filter_kind {
        for k in kinds {
            prefixes.push(format!("kind:{}", k));
        }
    }
    if let Some(langs) = filter_lang {
        for l in langs {
            prefixes.push(format!("lang:{}", l));
        }
    }
    if let Some(path) = filter_path
        && !path.is_empty()
    {
        prefixes.push(format!("path:{}", path));
    }
    if prefixes.is_empty() {
        information_request.to_string()
    } else {
        format!("{} {}", prefixes.join(" "), information_request)
    }
}

/// Format an enriched caller tag: `[callers: fn_a, fn_b, fn_c +N more]`
/// When callers > 3, shows first 3 names + count of remaining.
/// Returns empty string when no callers.
fn format_enriched_caller_tag(
    count: Option<u32>,
    names: &[String],
    _file_count: Option<u32>,
) -> String {
    let c = match count {
        Some(c) if c > 0 => c,
        _ => return String::new(),
    };
    if names.is_empty() {
        // Fallback to count-only format if names weren't fetched
        return format!(" [callers:{c}]");
    }
    let max_display = 30;
    let display_names: Vec<&str> = names.iter().take(max_display).map(|s| s.as_str()).collect();
    let remaining = c.saturating_sub(display_names.len() as u32);
    if remaining > 0 {
        format!(" [callers: {} +{} more]", display_names.join(", "), remaining)
    } else {
        format!(" [callers: {}]", display_names.join(", "))
    }
}

/// Format an enriched callee tag: `[calls: fn_x, fn_y +N more]`
/// Returns empty string when no callees.
fn format_enriched_callee_tag(
    count: Option<u32>,
    names: &[String],
) -> String {
    let c = match count {
        Some(c) if c > 0 => c,
        _ => return String::new(),
    };
    if names.is_empty() {
        return format!(" [calls:{c}]");
    }
    let max_display = 30;
    let display_names: Vec<&str> = names.iter().take(max_display).map(|s| s.as_str()).collect();
    let remaining = c.saturating_sub(display_names.len() as u32);
    if remaining > 0 {
        format!(" [calls: {} +{} more]", display_names.join(", "), remaining)
    } else {
        format!(" [calls: {}]", display_names.join(", "))
    }
}

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
        settings.embedding.voyage_base_url.as_deref(),
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
        settings.llm.agentic_rag,
        settings.llm.agentic_rag_max_turns,
        settings.llm.agentic_rag_max_chunk_chars,
        settings.llm.agentic_rag_grep_read,
    )
    .await
    {
        Err(e) => format!("Error: query failed: {e}"),
        Ok(result) => {
            // Warming takes precedence over the empty-handling: an empty result with
            // `warming` set means the repo's vector shard was not resident after the
            // bounded warm-wait expired — the index IS complete, it just hasn't loaded
            // into memory yet. Returning "No results found" here would falsely tell the
            // caller the codebase has nothing relevant; instead signal a retry. The
            // decision is a pure function of (warming, empty, rerank_rejected) so it is
            // unit-tested directly (see select_empty_or_warming_message).
            if result.warming || result.results.is_empty() {
                let rerank_rejected = result.rerank.as_ref().is_some_and(|r| {
                    !r.fallback_used && r.skip_reason.is_none() && !r.raw_response.is_empty()
                });
                return select_empty_or_warming_message(
                    result.warming, rerank_rejected, information_request,
                );
            }
            let blocks: Vec<OutputBlock> = result
                .results
                .iter()
                .map(|r| {
                    let caller_tag = format_enriched_caller_tag(
                        r.callers, &r.caller_names, r.caller_files,
                    );
                    let callee_tag = format_enriched_callee_tag(
                        r.callees, &r.callee_names,
                    );
                    OutputBlock {
                        header: format!(
                            "{}#L{}-{}{}{}",
                            r.file, r.line_start, r.line_end, caller_tag, callee_tag
                        ),
                        content: r.content.clone(),
                        file: r.file.clone(),
                        line_start: r.line_start,
                        line_end: r.line_end,
                        callers: r.callers,
                        caller_files: r.caller_files,
                        caller_names: r.caller_names.clone(),
                        callee_names: r.callee_names.clone(),
                        callees: r.callees,
                    }
                })
                .collect();
            let blocks = merge_overlapping_blocks(blocks);
            // Sort generated-file blocks after hand-written ones, preserving
            // relative order within each group (stable partition).
            let (hand_written, generated): (Vec<_>, Vec<_>) = blocks
                .into_iter()
                .partition(|b| !crate::parsing::generated::is_generated_file(&b.file));
            let mut blocks = hand_written;
            blocks.extend(generated);
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
    let repo = &crate::store::normalize_repo_path(repo);
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
    let db = match store::get_or_open(repo_dbs, data_dir, repo, settings.repo_generation(repo)).await {
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
        settings.embedding.voyage_base_url.as_deref(),
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
            symbol_kind: None,
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
                        file: display_path.clone(),
                        line_start: s,
                        line_end: e,
                        callers: None,
                        caller_files: None,
                    ..Default::default()
                    });
                }
            }
            (Some(text), _) => {
                blocks.push(OutputBlock {
                    header: format!("{}#L{}-{}", display_path, chunk.line_start, chunk.line_end),
                    content: text.to_string(),
                    file: display_path.clone(),
                    line_start: chunk.line_start,
                    line_end: chunk.line_end,
                    callers: None,
                    caller_files: None,
                    ..Default::default()
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
                    file: display_path.clone(),
                    line_start: chunk.line_start,
                    line_end: chunk.line_end,
                    callers: None,
                    caller_files: None,
                    ..Default::default()
                });
            }
        }
    }

    if blocks.is_empty() {
        return format!("No relevant chunks found for query in file: {file_path}");
    }

    let blocks = merge_overlapping_blocks(blocks);
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
