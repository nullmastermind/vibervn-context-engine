// Pre-existing layout: a few helpers live AFTER the inline #[cfg(test)]
// `tests` module rather than before it. Clippy flags this as
// `items_after_test_module`. Reordering the file is out of scope for the
// current change; suppress the lint at module level.
#![allow(clippy::items_after_test_module)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tokio::sync::RwLock;

use rmcp::{
    ErrorData, RoleServer, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    schemars,
    service::RequestContext,
    tool, tool_handler, tool_router,
};

mod progress;
pub(crate) mod query_gate;
pub(crate) mod readiness;
#[cfg(test)]
pub(crate) mod tests;

pub use progress::{MCP_PROGRESS_HEARTBEAT, with_progress_heartbeat};

use crate::config::Settings;
use crate::embedding::voyage::VoyageClient;
use crate::indexing::IndexEngine;
use crate::llm::LlmClient;
use crate::query::engine::QueryGraphMode;
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
    symbols: Vec<String>,
}

// ─── Grep hint footer ────────────────────────────────────────────────────
// Retrieval is top-k, not exhaustive. Handing the agent a ready-to-run literal
// pattern for the symbols it just received beats a static "use grep" line, which
// gives it nothing to run and tends to be ignored.

const GREP_HINT_MAX_SYMBOLS: usize = 5;
/// Shorter names (`new`, `get`, `run`) would make the grep all noise.
const GREP_HINT_MIN_NAME_LEN: usize = 4;
const GREP_HINT_MAX_NAME_LEN: usize = 40;
/// Total length of the `a|b|c` pattern. It appears TWICE in the hint (once for
/// `rg`, once for `grep`), so this bounds the hint size, not just its pattern.
const GREP_HINT_MAX_PATTERN_CHARS: usize = 100;
/// Upper bound on the whole rendered hint: template text (~135 chars) plus the
/// pattern twice. Reserved out of the output budget so the hint can never push the
/// response past `MAX_TOOL_OUTPUT_CHARS` or squeeze out the truncation footer.
const GREP_HINT_MAX_CHARS: usize = 400;

/// Identifiers only. That keeps the pattern free of regex and shell
/// metacharacters, so quoting it in the hint is always safe (`operator==` and
/// the like are dropped rather than escaped).
fn is_greppable_symbol(name: &str) -> bool {
    (GREP_HINT_MIN_NAME_LEN..=GREP_HINT_MAX_NAME_LEN).contains(&name.len())
        && name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Repo-relative, `/`-separated view of `file`. Absolute paths MUST NOT reach the
/// test-path check: a repo that lives under a `test/` directory would otherwise lose
/// its hint on every query. When `file` is not under `repo` (should not happen), only
/// the file name is kept so no directory segment can produce a false positive.
fn repo_relative(file: &str, repo: &str) -> String {
    let file = file.replace('\\', "/");
    let repo = repo.replace('\\', "/");
    let repo = repo.trim_end_matches('/');
    if file.len() > repo.len()
        && file.is_char_boundary(repo.len())
        && file[..repo.len()].eq_ignore_ascii_case(repo)
        && file.as_bytes()[repo.len()] == b'/'
    {
        return file[repo.len() + 1..].to_string();
    }
    file.rsplit('/').next().unwrap_or("").to_string()
}

/// The own symbol of a result, if it is worth grepping for. Dropped: modules,
/// symbols from test files, and symbols from generated files. `impl_Foo` is
/// rewritten to the type name `Foo`. A test block's CALLEES are kept elsewhere: they are the
/// production API it exercises.
fn hint_symbol(file: &str, repo: &str, symbol: Option<&str>, kind: Option<&str>) -> Option<String> {
    let symbol = symbol?;
    if kind.is_some_and(|k| k.eq_ignore_ascii_case("module")) {
        return None;
    }
    // The Rust extractor names an impl block `impl_<Type>`: not itself grep-able,
    // but the type it names is. Dropping it outright left impl-only result sets
    // (e.g. a whole query answered by `impl Runner`/`impl Registry` chunks) with no
    // hint at all. Generics are stripped; anything non-identifier (`&Foo`) is
    // filtered later by `is_greppable_symbol`.
    let symbol = if kind.is_some_and(|k| k.eq_ignore_ascii_case("impl")) {
        symbol
            .strip_prefix("impl_")
            .and_then(|ty| ty.split('<').next())?
    } else {
        symbol
    };
    if crate::parsing::generated::is_generated_file(file)
        || crate::parsing::test_paths::is_test_path(&repo_relative(file, repo))
    {
        return None;
    }
    Some(symbol.to_string())
}

/// A row of the `symbol` table overlapping a result block.
#[derive(Debug, Clone)]
struct DefinedSymbol {
    name: String,
    kind: Option<String>,
    line_start: u32,
    line_end: u32,
}

/// One block must not fill the whole hint on its own.
const GREP_HINT_MAX_DEFINED_PER_BLOCK: usize = 3;

/// Symbols that name a block which carries no `symbol_ref` of its own. Measured on
/// real repos, MOST result chunks have none: `symbol_ref` is only set at index time
/// when a chunk maps onto exactly one symbol, so a chunk that includes a doc comment,
/// a `static` plus a `fn`, or the middle of a long function gets nothing — which is
/// how `open_gate` (defined inside such a chunk) never reached the hint.
///
/// Symbols that START inside the block come first, in line order. Only when none
/// does is the innermost enclosing symbol used (a slice from the middle of a long
/// function is named by that function, not by its `impl` or module).
fn pick_block_symbols(rows: &[DefinedSymbol], start: u32, end: u32) -> Vec<&DefinedSymbol> {
    let overlapping = rows
        .iter()
        .filter(|s| s.line_start <= end && s.line_end >= start);
    let mut inside: Vec<&DefinedSymbol> = overlapping
        .clone()
        .filter(|s| s.line_start >= start)
        .collect();
    if inside.is_empty() {
        return overlapping
            .min_by_key(|s| s.line_end.saturating_sub(s.line_start))
            .into_iter()
            .collect();
    }
    inside.sort_by_key(|s| s.line_start);
    inside
}

/// Rust keeps unit tests inline in production files, so the test-path rule cannot
/// see them. A block whose own text carries a test attribute must not contribute
/// the test functions it defines.
fn has_inline_test_marker(content: &str) -> bool {
    ["#[test]", "#[tokio::test]", "#[cfg(test)]"]
        .iter()
        .any(|marker| content.contains(marker))
}

/// Whether the definition starting at `line_start` (1-based) carries a test
/// attribute: tree-sitter starts a Rust item at its `fn`/`mod` keyword, so its
/// `#[test]` / `#[cfg(test)]` sits on the few lines just above it.
fn is_test_marked(source: &str, line_start: u32) -> bool {
    let end = line_start as usize;
    let begin = end.saturating_sub(3).max(1);
    let window: Vec<&str> = source
        .lines()
        .skip(begin - 1)
        .take(end.saturating_sub(begin) + 1)
        .collect();
    has_inline_test_marker(&window.join("\n"))
}

/// `symbol` rows overlapping `file:start-end`, from whichever repo DB owns the file.
/// Empty on any failure: the hint is best-effort and must never fail a query.
async fn defined_symbols_for(
    db_map: &HashMap<String, Surreal<Db>>,
    file: &str,
    start: u32,
    end: u32,
) -> Vec<DefinedSymbol> {
    let Some(db) = crate::query::find_db_for_file(db_map, file) else {
        return Vec::new();
    };
    crate::query::graph_expand::query_overlapping_symbols(db, file, start, end)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|row| DefinedSymbol {
            name: row.name,
            kind: row.kind,
            line_start: row.line_start.max(0) as u32,
            line_end: row.line_end.max(0) as u32,
        })
        .collect()
}

/// The hint symbols one result contributes: its own symbol when it has a usable
/// one, otherwise the symbols the block defines (see `pick_block_symbols`). An own
/// symbol that was FILTERED (e.g. a `module`) does not fall back: the block is that
/// module, and in Rust an inline module is almost always a test module.
async fn block_hint_symbols(
    db_map: &HashMap<String, Surreal<Db>>,
    repo: &str,
    r: &crate::query::engine::CodeResult,
) -> Vec<String> {
    if r.symbol.is_some() {
        return hint_symbol(&r.file, repo, r.symbol.as_deref(), r.symbol_kind.as_deref())
            .into_iter()
            .collect();
    }
    if has_inline_test_marker(&r.content) {
        return Vec::new();
    }
    let rows = defined_symbols_for(db_map, &r.file, r.line_start, r.line_end).await;
    if rows.is_empty() {
        return Vec::new();
    }
    // Rust is the one language here that keeps unit tests INSIDE production files,
    // so neither the path rule nor the block's own text (a reranker slice from the
    // middle of a test body has no attribute in it) can see them. Read the source
    // once and look above each definition instead. Other languages skip the read.
    let source = if r.file.ends_with(".rs") {
        std::fs::read_to_string(&r.file).unwrap_or_default()
    } else {
        String::new()
    };
    // Strictly enclosing (starts before the block): a symbol whose range IS the
    // block is a candidate, judged by the per-candidate check below.
    let in_test_scope = rows.iter().any(|s| {
        s.line_start < r.line_start
            && s.line_end >= r.line_end
            && is_test_marked(&source, s.line_start)
    });
    if in_test_scope {
        return Vec::new();
    }
    pick_block_symbols(&rows, r.line_start, r.line_end)
        .into_iter()
        .filter(|s| !is_test_marked(&source, s.line_start))
        .filter_map(|s| hint_symbol(&r.file, repo, Some(&s.name), s.kind.as_deref()))
        .take(GREP_HINT_MAX_DEFINED_PER_BLOCK)
        .collect()
}

/// Build the grep hint from the final, priority-ordered blocks, in two passes:
/// first the symbols every block DEFINES (block order), then the callees of every
/// block. Measured across five repos, defined symbols are on-topic while callees
/// often drag in logging/util helpers (`maskUserID`, `CreatePendingLog`), so callees
/// only fill what is left. They still matter: they come from the resolved call
/// graph, so they always name real in-repo definitions, and for test blocks they
/// are the production API under test.
/// `None` when no usable symbol exists: no static fallback, nothing to grep for.
fn build_grep_hint(blocks: &[OutputBlock]) -> Option<String> {
    let mut names: Vec<&str> = Vec::new();
    let mut pattern_len = 0usize;
    let defined = blocks.iter().flat_map(|b| &b.symbols);
    let callees = blocks.iter().flat_map(|b| &b.callee_names);
    for symbol in defined.chain(callees) {
        if names.len() >= GREP_HINT_MAX_SYMBOLS {
            break;
        }
        let name = symbol.as_str();
        if !is_greppable_symbol(name) || names.contains(&name) {
            continue;
        }
        let added = name.len() + usize::from(!names.is_empty());
        if pattern_len + added > GREP_HINT_MAX_PATTERN_CHARS {
            break;
        }
        pattern_len += added;
        names.push(name);
    }
    if names.is_empty() {
        return None;
    }
    Some(render_grep_hint(
        crate::prompts::MCP_GREP_HINT,
        &names.join("|"),
    ))
}

/// `core.autocrlf=true` checks the template out with CRLF on Windows and
/// `include_str!` embeds those bytes, so line endings are normalized here rather
/// than trusting the checkout.
fn render_grep_hint(template: &str, pattern: &str) -> String {
    let rendered = crate::prompts::render(template, &[("pattern", pattern)]);
    format!("\n\n{}", rendered.replace("\r\n", "\n").trim_end())
}

/// Append the grep hint after the assembled output (and its truncation footer).
fn with_grep_hint(mut assembled: String, blocks: &[OutputBlock]) -> String {
    if let Some(hint) = build_grep_hint(blocks) {
        assembled.push_str(&hint);
    }
    assembled
}

/// Assemble result blocks into a single string respecting `MAX_TOOL_OUTPUT_CHARS`.
///
/// Results are in priority order (reranked). Full content is emitted until the
/// budget would be exceeded; from that point, all remaining blocks are shown as
/// header + first line (capped at 120 chars) + elision marker.
fn assemble_with_budget(blocks: &[OutputBlock]) -> String {
    // Reserve space for the truncation footer so it's never squeezed out, plus the
    // grep hint that `with_grep_hint` appends after it: the hint is appended
    // unconditionally, and dropping it only when it didn't fit would lose it
    // exactly on the big, truncated responses where the agent most needs to search.
    const FOOTER_RESERVE: usize = 150 + GREP_HINT_MAX_CHARS;
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
                    // Carry the absorbed block's symbols so the grep hint can still
                    // name them; the block itself is gone after this merge.
                    current.symbols.append(&mut next.symbols);
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
            let caller_tag =
                format_enriched_caller_tag(block.callers, &block.caller_names, block.caller_files);
            let callee_tag = format_enriched_callee_tag(block.callees, &block.callee_names);
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
    #[doc = include_str!("prompts/mcp_scope_param.txt")]
    #[serde(default)]
    pub scope: Option<crate::vector::Scope>,
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

    #[doc = include_str!("prompts/mcp_codebase_retrieval.txt")]
    #[tool(name = "codebase-retrieval")]
    async fn codebase_retrieval(
        &self,
        Parameters(args): Parameters<CodebaseRetrievalArgs>,
        ctx: RequestContext<RoleServer>,
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
        let text = with_progress_heartbeat(
            ctx.peer,
            &ctx.meta,
            ctx.ct,
            MCP_PROGRESS_HEARTBEAT,
            run_codebase_retrieval(
                &self.home_dir,
                &self.data_dir,
                &self.index_engine,
                &self.repo_dbs,
                &settings,
                &augmented_query,
                &args.workspace_full_path,
                args.scope.unwrap_or_default(),
            ),
        )
        .await;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[doc = include_str!("prompts/mcp_file_retrieval.txt")]
    #[tool(name = "file-retrieval")]
    async fn file_retrieval(
        &self,
        Parameters(args): Parameters<FileRetrievalArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let settings = self.settings.read().await.clone();
        let text = with_progress_heartbeat(
            ctx.peer,
            &ctx.meta,
            ctx.ct,
            MCP_PROGRESS_HEARTBEAT,
            run_file_retrieval(
                &self.data_dir,
                &self.repo_dbs,
                &settings,
                &args.workspace_full_path,
                &args.file_path,
                &args.information_request,
                args.top_k.unwrap_or(5),
            ),
        )
        .await;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for McpHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_server_info(
            rmcp::model::Implementation::new("context-engine-rs", env!("CARGO_PKG_VERSION")),
        )
    }
}

// ─── Repo-scoped MCP handler ─────────────────────────────────────────────
// Exposes the same tools but with `workspace_full_path` pre-bound to a fixed
// repo path. Clients don't need to pass it — the endpoint itself is per-repo.

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RepoCodebaseRetrievalArgs {
    /// Natural-language description of the code or information you are looking for.
    pub information_request: String,
    #[doc = include_str!("prompts/mcp_scope_param.txt")]
    #[serde(default)]
    pub scope: Option<crate::vector::Scope>,
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

    #[doc = include_str!("prompts/mcp_codebase_retrieval.txt")]
    #[tool(name = "codebase-retrieval")]
    async fn codebase_retrieval(
        &self,
        Parameters(args): Parameters<RepoCodebaseRetrievalArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let settings = self.settings.read().await.clone();
        let text = with_progress_heartbeat(
            ctx.peer,
            &ctx.meta,
            ctx.ct,
            MCP_PROGRESS_HEARTBEAT,
            run_codebase_retrieval(
                &self.home_dir,
                &self.data_dir,
                &self.index_engine,
                &self.repo_dbs,
                &settings,
                &args.information_request,
                &self.repo_path,
                args.scope.unwrap_or_default(),
            ),
        )
        .await;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[doc = include_str!("prompts/mcp_file_retrieval_repo.txt")]
    #[tool(name = "file-retrieval")]
    async fn file_retrieval(
        &self,
        Parameters(args): Parameters<RepoFileRetrievalArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let settings = self.settings.read().await.clone();
        let text = with_progress_heartbeat(
            ctx.peer,
            &ctx.meta,
            ctx.ct,
            MCP_PROGRESS_HEARTBEAT,
            run_file_retrieval(
                &self.data_dir,
                &self.repo_dbs,
                &settings,
                &self.repo_path,
                &args.file_path,
                &args.information_request,
                args.top_k.unwrap_or(5),
            ),
        )
        .await;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for RepoMcpHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_server_info(
            rmcp::model::Implementation::new("context-engine-rs", env!("CARGO_PKG_VERSION")),
        )
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
const MCP_PARTIAL_RESULTS_PREFIX: &str =
    "(index update is still publishing; showing only content-verified partial results)\n\n";

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

/// `scope` is REQUIRED here (not an `Option`) so that every entry surface must
/// decide it explicitly — each passes `args.scope.unwrap_or_default()`, i.e.
/// `code` when the caller omitted it — and none can silently skip the default.
#[allow(clippy::too_many_arguments)]
pub async fn run_codebase_retrieval(
    home_dir: &Path,
    data_dir: &Path,
    index_engine: &Arc<IndexEngine>,
    repo_dbs: &Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    settings: &Settings,
    information_request: &str,
    workspace_full_path: &str,
    scope: crate::vector::Scope,
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

    // 4. One shared readiness decision selects both the wait budget and graph
    // mode. MCP and REST therefore cannot drift on the ResolveEdges fast path.
    let (query_graph_mode, query_warm_wait, output_prefix) = match readiness::await_index_ready(
        settings,
        index_engine,
        repo_dbs,
        data_dir,
        repo,
    )
    .await
    {
        readiness::IndexReadiness::Ready { warm_budget } => (QueryGraphMode::Full, warm_budget, ""),
        readiness::IndexReadiness::ReadyVectorOnly { warm_budget } => (
            QueryGraphMode::VectorOnly,
            warm_budget,
            crate::prompts::MCP_GRAPH_PENDING,
        ),
        readiness::IndexReadiness::Timeout => {
            return crate::prompts::MCP_DEGRADE_INDEXING.to_string();
        }
        readiness::IndexReadiness::Failed(error) => {
            let message = format!("{error:#}");
            return crate::prompts::render(
                crate::prompts::MCP_DEGRADE_INDEX_FAILED,
                &[("err", &message)],
            );
        }
    };

    let output = do_query(
        index_engine,
        repo_dbs,
        settings,
        &scoped_request(information_request, scope),
        repo,
        query_graph_mode,
        query_warm_wait,
    )
    .await;
    format!("{output_prefix}{output}")
}

/// Prefix the request with the structured scope as an inline `scope:` filter (the
/// same channel `filter_lang` uses). It goes FIRST and the parser keeps the first
/// valid `scope:`, so the tool parameter beats one typed inside the free text.
fn scoped_request(information_request: &str, scope: crate::vector::Scope) -> String {
    format!("scope:{} {}", scope.as_str(), information_request)
}

/// Banner for results served from the OTHER scope after the requested one was
/// empty — the agent must know it did not get what it asked for.
fn scope_fallback_notice(requested: crate::vector::Scope, served: crate::vector::Scope) -> String {
    crate::prompts::render(
        crate::prompts::MCP_SCOPE_FALLBACK,
        &[
            ("requested", requested.as_str()),
            ("served", served.as_str()),
        ],
    )
}

/// A genuine empty for a scoped query means BOTH scopes were searched (the
/// fallback ran and also found nothing). Say so, but only when that is true:
/// not while warming (retry wording must stay intact) and not when candidates
/// existed and the reranker rejected them.
fn note_both_scopes_searched(
    message: String,
    scoped: bool,
    no_candidates: bool,
    warming: bool,
) -> String {
    if scoped && no_candidates && !warming {
        format!("{message} (searched both code and docs)")
    } else {
        message
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
        format!(
            " [callers: {} +{} more]",
            display_names.join(", "),
            remaining
        )
    } else {
        format!(" [callers: {}]", display_names.join(", "))
    }
}

/// Format an enriched callee tag: `[calls: fn_x, fn_y +N more]`
/// Returns empty string when no callees.
fn format_enriched_callee_tag(count: Option<u32>, names: &[String]) -> String {
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
    graph_mode: QueryGraphMode,
    warm_wait: Duration,
) -> String {
    let voyage_client = match VoyageClient::new_for_provider(
        crate::embedding::voyage::Provider::parse(&settings.embedding.provider),
        settings.embedding.model.clone(),
        settings.embedding.api_keys.clone(),
        settings.embedding.voyage_base_url.as_deref(),
        settings.embedding.dimensions,
    ) {
        Ok(c) => c,
        Err(e) => return format!("Error: failed to create embedding client: {e}"),
    };

    let llm_client: Option<LlmClient> = LlmClient::new(&settings.llm);

    match crate::query::engine::run_query_with_filters_and_mode(
        information_request,
        30,
        Some(repo),
        &voyage_client,
        index_engine,
        repo_dbs,
        settings.llm.rerank_min_prune_lines,
        llm_client.as_ref(),
        warm_wait,
        settings.llm.agentic_rag,
        settings.llm.agentic_rag_max_turns,
        settings.llm.agentic_rag_max_chunk_chars,
        settings.llm.agentic_rag_grep_read,
        // The live docs-extensions setting rides in on the filters, read from the
        // per-request settings snapshot (workers reload settings.json per request).
        Some(crate::query::filters::QueryFilters {
            docs_extensions: Some(settings.docs_extensions.clone()),
            ..Default::default()
        }),
        graph_mode,
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
            if result.results.is_empty() {
                let rerank_rejected = result.rerank.as_ref().is_some_and(|r| {
                    !r.fallback_used && r.skip_reason.is_none() && !r.raw_response.is_empty()
                });
                return note_both_scopes_searched(
                    select_empty_or_warming_message(
                        result.warming,
                        rerank_rejected,
                        information_request,
                    ),
                    result.scope.is_some(),
                    // The engine's early-return paths (no candidates survived)
                    // are the only ones without rerank info.
                    result.rerank.is_none(),
                    result.warming,
                );
            }
            // Snapshot the DB handles for the grep-hint symbol lookup. The read
            // guard is dropped at the end of this block, BEFORE any DB await.
            let db_map: HashMap<String, Surreal<Db>> = {
                let guard = repo_dbs.read().await;
                guard.clone()
            };
            // One indexed seek (`idx_symbol_file`) per result that lacks its own
            // symbol — bounded by the result count, like graph expansion.
            let mut hint_symbols: Vec<Vec<String>> = Vec::with_capacity(result.results.len());
            for r in &result.results {
                hint_symbols.push(block_hint_symbols(&db_map, repo, r).await);
            }
            let blocks: Vec<OutputBlock> = result
                .results
                .iter()
                .zip(hint_symbols)
                .map(|(r, symbols)| {
                    let caller_tag =
                        format_enriched_caller_tag(r.callers, &r.caller_names, r.caller_files);
                    let callee_tag = format_enriched_callee_tag(r.callees, &r.callee_names);
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
                        symbols,
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
            let mut assembled = with_grep_hint(assemble_with_budget(&blocks), &blocks);
            if result.scope_fallback
                && let Some(served) = result
                    .scope
                    .as_deref()
                    .and_then(crate::vector::Scope::parse)
            {
                assembled = format!(
                    "{}{assembled}",
                    scope_fallback_notice(served.opposite(), served)
                );
            }
            if result.warming {
                format!("{MCP_PARTIAL_RESULTS_PREFIX}{assembled}")
            } else {
                assembled
            }
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
    let db =
        match store::get_or_open(repo_dbs, data_dir, repo, settings.repo_generation(repo)).await {
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
    let voyage_client = match VoyageClient::new_for_provider(
        crate::embedding::voyage::Provider::parse(&settings.embedding.provider),
        settings.embedding.model.clone(),
        settings.embedding.api_keys.clone(),
        settings.embedding.voyage_base_url.as_deref(),
        settings.embedding.dimensions,
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
        let Some(chunk) = merge_chunks.get(idx) else {
            continue;
        };
        let numbered_text = numbered.get(idx).and_then(|n| n.as_deref());
        let selection = rerank_output
            .line_selections
            .get(k)
            .and_then(|s| s.as_ref());

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
    out.push_str(crate::prompts::MCP_FILE_RETRIEVAL_HINT);

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
