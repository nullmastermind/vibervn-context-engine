//! Repo chat: a streaming, tool-calling agent that answers questions about a
//! single repository. It is **hard-locked** to exactly two tools —
//! `codebase-retrieval` and `file-retrieval` — both scoped to the repo the
//! conversation was opened for. The agent cannot reach any other repo or tool.
//!
//! Conversation state lives only in memory ([`ConversationStore`]), keyed by a
//! client-generated id. Closing the dialog drops the id; reopening makes a new
//! one. The store is LRU-capped (count + per-conversation message count) so RAM
//! stays bounded regardless of how many dialogs are opened over a session — and
//! only the plain-text transcript (user questions + assistant answers) is kept,
//! never the tool-call/result history.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tokio::sync::{Mutex, RwLock, mpsc};

use crate::config::Settings;
use crate::indexing::IndexEngine;
use crate::llm::{ChatMessage, LlmClient, ToolDef, ToolResult, ToolTurnResult};

/// The tool names the chat agent may call. Any other name returned by the model
/// is rejected with an error tool-result so it self-corrects. Two are semantic
/// (index-backed) and two are exact (filesystem-backed):
/// - [`TOOL_CODEBASE`] / [`TOOL_FILE`] embed the request and rank by meaning.
/// - [`TOOL_GREP`] / [`TOOL_READ`] hit the working tree directly for exact text
///   and verbatim line ranges — the difference between "guess what's relevant"
///   and "see exactly what is there", which is what keeps answers from drifting
///   into fabricated symbol names or invented code.
pub const TOOL_CODEBASE: &str = "codebase-retrieval";
pub const TOOL_FILE: &str = "file-retrieval";
pub const TOOL_GREP: &str = "grep";
pub const TOOL_READ: &str = "read";

/// Max tool-calling rounds before the loop gives up (bounds cost per question).
const MAX_TURNS: u32 = 8;
/// Max characters of a tool result forwarded to the UI as a preview.
const PREVIEW_CHARS: usize = 280;

// The exact filesystem tools (grep, read) and their path-traversal guard live in
// `crate::fs_tools`, shared with the Agentic RAG reranker. `GREP_MAX_CONTEXT`
// bounds the `-C` knob below.
use crate::fs_tools::{GREP_MAX_CONTEXT, run_grep, run_read};


// ─── Streaming events (serialized to SSE `data:` JSON) ────────────────────

/// One event in the chat stream. `type` is the discriminator the UI switches on.
#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    /// The agent is invoking a tool. `summary` is a short human label.
    ToolCall { name: String, summary: String },
    /// A tool finished. `ok` is false when the tool returned an error string.
    ToolResult { name: String, ok: bool, preview: String },
    /// A text delta of the assistant's answer.
    Token { text: String },
    /// The turn finished successfully (final answer fully streamed).
    Done,
    /// The turn failed; `message` is shown inline in the dialog.
    Error { message: String },
}

// ─── Conversation store (bounded, in-memory) ──────────────────────────────

/// Hard cap on concurrent in-memory conversations. When exceeded, the
/// least-recently-used conversation is evicted. Bounds RAM no matter how many
/// dialogs are opened/closed over a session.
const MAX_CONVERSATIONS: usize = 64;
/// Hard cap on turns kept per conversation (each turn = one user message plus
/// one assistant message, plus a compact tool-context summary). Older turns are
/// dropped so a single long-running conversation can't grow without bound.
const MAX_TURNS_KEPT: usize = 40;

/// How many of the most-recent turns contribute their tool-context summary to
/// the next question (K). Only recent search evidence is worth replaying; older
/// turns fall out of the window.
const TOOL_CTX_TURNS_KEPT: usize = 3;
/// Hard byte cap on the tool-context summary STORED per turn. Enforced at store
/// time so a single huge search result (kernel-scale) can never bloat the
/// transcript — bounded memory regardless of repo size.
const TOOL_CTX_PER_TURN_CAP: usize = 1500;
/// Hard byte cap on the TOTAL tool-context injected into one question (across
/// all K turns combined). The injected block is trimmed to this even if K turns
/// each sit just under the per-turn cap.
const TOOL_CTX_TOTAL_CAP: usize = 3000;

/// A single completed turn: the user's question, the assistant's answer, and a
/// compact, already-capped summary of the search evidence gathered during it
/// (location headers + short previews — never full source). `tool_context` is
/// empty for turns that ran no tools (pure chit-chat).
struct Turn {
    user: String,
    answer: String,
    tool_context: String,
}

/// A conversation: an ordered list of turns plus the repo it is bound to.
struct Conversation {
    repo: String,
    /// Completed turns, oldest first.
    turns: Vec<Turn>,
    last_used: Instant,
}

/// In-memory, LRU-bounded conversation store. Keyed by client-generated id.
#[derive(Default)]
pub struct ConversationStore {
    inner: Mutex<HashMap<String, Conversation>>,
}

impl ConversationStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the plain User/Model transcript for `id` (empty if unknown),
    /// refreshing its LRU stamp. If the id exists but is bound to a different
    /// repo, it is treated as a fresh conversation for `repo` (defensive — ids
    /// are per-repo dialogs). Tool-context is NOT included here; it is fetched
    /// separately via [`Self::recent_tool_context`].
    async fn snapshot(&self, id: &str, repo: &str) -> Vec<ChatMessage> {
        let mut map = self.inner.lock().await;
        match map.get_mut(id) {
            Some(c) if c.repo == repo => {
                c.last_used = Instant::now();
                let mut out = Vec::with_capacity(c.turns.len() * 2);
                for t in &c.turns {
                    out.push(ChatMessage::User(t.user.clone()));
                    out.push(ChatMessage::Model(t.answer.clone()));
                }
                out
            }
            _ => Vec::new(),
        }
    }

    /// Build the tool-context block to replay into the NEXT question: the
    /// summaries of the last [`TOOL_CTX_TURNS_KEPT`] turns that actually ran a
    /// search, newest last, joined and hard-capped at [`TOOL_CTX_TOTAL_CAP`]
    /// bytes. Returns an empty string when there is nothing to replay.
    async fn recent_tool_context(&self, id: &str, repo: &str) -> String {
        let mut map = self.inner.lock().await;
        let Some(c) = map.get_mut(id) else {
            return String::new();
        };
        if c.repo != repo {
            return String::new();
        }
        c.last_used = Instant::now();

        // Walk the most-recent turns (newest first), collecting non-empty
        // summaries up to K, then re-order oldest→newest for natural reading.
        let mut picked: Vec<&str> = Vec::new();
        for t in c.turns.iter().rev() {
            if picked.len() >= TOOL_CTX_TURNS_KEPT {
                break;
            }
            if !t.tool_context.is_empty() {
                picked.push(&t.tool_context);
            }
        }
        if picked.is_empty() {
            return String::new();
        }
        picked.reverse();
        let joined = picked.join("\n\n");
        truncate_bytes(&joined, TOOL_CTX_TOTAL_CAP)
    }

    /// Append one completed turn to a conversation, creating it if absent.
    /// `tool_context` is capped to [`TOOL_CTX_PER_TURN_CAP`] bytes before
    /// storage. Enforces both caps (turns-per-conversation, then global LRU).
    async fn append_turn(
        &self,
        id: &str,
        repo: &str,
        user: String,
        answer: String,
        tool_context: String,
    ) {
        let mut map = self.inner.lock().await;
        let conv = map.entry(id.to_owned()).or_insert_with(|| Conversation {
            repo: repo.to_owned(),
            turns: Vec::new(),
            last_used: Instant::now(),
        });
        // A repo mismatch means the id was reused across dialogs — reset it.
        if conv.repo != repo {
            conv.repo = repo.to_owned();
            conv.turns.clear();
        }
        // Cap the stored summary at store time — the single enforcement point.
        let tool_context = truncate_bytes(&tool_context, TOOL_CTX_PER_TURN_CAP);
        conv.turns.push(Turn { user, answer, tool_context });
        conv.last_used = Instant::now();

        // Trim oldest turns.
        while conv.turns.len() > MAX_TURNS_KEPT {
            conv.turns.remove(0);
        }

        // Global LRU eviction.
        if map.len() > MAX_CONVERSATIONS
            && let Some(oldest) = map
                .iter()
                .min_by_key(|(_, c)| c.last_used)
                .map(|(k, _)| k.clone())
            && oldest != id
        {
            map.remove(&oldest);
        }
    }

    /// Drop a conversation (called when the dialog is closed).
    pub async fn drop_conversation(&self, id: &str) {
        self.inner.lock().await.remove(id);
    }
}

/// Truncate `s` to at most `max_bytes`, never splitting a UTF-8 char, appending
/// an elision marker when anything was cut. Used to hard-bound every stored and
/// replayed tool-context string.
fn truncate_bytes(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_owned();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

// ─── Project documentation injection (orientation context) ────────────────

/// The project docs we seed into the system prompt, matched case-insensitively
/// at the repo ROOT only. A fixed allowlist — never a user-supplied name — so
/// the repo path (which comes from MCP/user input) can never be used to read an
/// arbitrary file: we only ever read `<root>/<one-of-these>`.
const PROJECT_DOC_NAMES: [&str; 3] = ["readme.md", "agents.md", "claude.md"];
/// Per-file byte cap. Files larger than this are truncated on a line boundary
/// with a marker so the model knows it saw only the head of the file.
const PROJECT_DOC_PER_FILE_CAP: usize = 12 * 1024;
/// Total byte cap across all injected docs combined. Bounds the cost the FIRST
/// turn pays regardless of how big the repo's docs are (later turns ride the
/// system-prompt cache, so they pay nothing).
const PROJECT_DOC_TOTAL_CAP: usize = 24 * 1024;

/// Truncate `content` to at most `max_bytes` on a LINE boundary (never mid-line)
/// and append a ` … [truncated]` marker when anything was cut, so the model can
/// tell the doc is partial. Returns the content unchanged when it already fits.
fn truncate_doc_by_lines(content: &str, max_bytes: usize) -> String {
    if content.len() <= max_bytes {
        return content.to_owned();
    }
    let mut out = String::new();
    for line in content.lines() {
        // +1 for the '\n' we re-add; stop before exceeding the cap. If the very
        // first line already overflows the cap we emit just the marker rather
        // than splitting mid-line.
        if out.len() + line.len() + 1 > max_bytes {
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("… [truncated]");
    out
}

/// Read the repo's orientation docs (README/AGENTS/CLAUDE) from its root and
/// build a single reference block for the system prompt. Returns `""` when none
/// exist or none are readable — the caller appends nothing in that case, so the
/// system prompt stays clean.
///
/// Safety / robustness contract:
/// - Only files whose name case-insensitively matches [`PROJECT_DOC_NAMES`] at
///   the immediate root are read. Path traversal is impossible: we enumerate the
///   root with `read_dir` and use the entry's own path, never a joined string.
/// - Symlinks are skipped (`symlink_metadata` does not follow), so a planted
///   link cannot redirect the read outside the repo.
/// - Any failure (dir unreadable, file unreadable, non-UTF-8, non-regular) is
///   skipped silently — collecting docs must never fail a chat turn.
/// - Output order is stable (README, then AGENTS, then CLAUDE) and bounded by
///   the per-file and total caps.
fn collect_project_docs(repo_root: &std::path::Path) -> String {
    let Ok(entries) = std::fs::read_dir(repo_root) else {
        return String::new();
    };

    // Map each present doc (by canonical lowercase name) to its path, so we can
    // emit them in a fixed order regardless of directory iteration order.
    let mut found: HashMap<&'static str, std::path::PathBuf> = HashMap::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let lower = name.to_ascii_lowercase();
        let Some(&canonical) = PROJECT_DOC_NAMES.iter().find(|n| **n == lower) else {
            continue;
        };
        // Reject anything that is not a regular file (dirs, symlinks). Using
        // symlink_metadata means a symlink is seen AS a symlink and skipped,
        // never followed outside the repo.
        match entry.path().symlink_metadata() {
            Ok(md) if md.is_file() => {
                found.entry(canonical).or_insert_with(|| entry.path());
            }
            _ => continue,
        }
    }
    if found.is_empty() {
        return String::new();
    }

    let mut sections: Vec<String> = Vec::new();
    let mut total = 0usize;
    for canonical in PROJECT_DOC_NAMES {
        let Some(path) = found.get(canonical) else { continue };
        // Non-UTF-8 or unreadable files are skipped silently.
        let Ok(raw) = std::fs::read_to_string(path) else {
            continue;
        };
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let display_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(canonical);
        // Per-file cap, then trim to the remaining total budget on a line
        // boundary so the combined block never exceeds the total cap.
        let mut body = truncate_doc_by_lines(raw, PROJECT_DOC_PER_FILE_CAP);
        let remaining = PROJECT_DOC_TOTAL_CAP.saturating_sub(total);
        if remaining == 0 {
            break;
        }
        if body.len() > remaining {
            body = truncate_doc_by_lines(&body, remaining);
        }
        total += body.len();
        sections.push(format!("--- {display_name} ---\n{body}"));
    }

    if sections.is_empty() {
        return String::new();
    }
    sections.join("\n\n")
}

// ─── Tool definitions (the ONLY two the agent may call) ───────────────────

/// Build the two allowed tool definitions. `workspace_full_path` is fixed to the
/// conversation's repo so the model never supplies (or can target) another repo.
fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: TOOL_CODEBASE.to_owned(),
            description: crate::prompts::CHAT_TOOL_CODEBASE.to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "information_request": {
                        "type": "string",
                        "description": "A detailed natural-language description of what you \
                            need to find (e.g. 'how is the vector index sharded per repo')."
                    }
                },
                "required": ["information_request"]
            }),
        },
        ToolDef {
            name: TOOL_FILE.to_owned(),
            description: crate::prompts::CHAT_TOOL_FILE.to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file, relative to the repository root."
                    },
                    "information_request": {
                        "type": "string",
                        "description": "What you need to learn from this file."
                    }
                },
                "required": ["file_path", "information_request"]
            }),
        },
        ToolDef {
            name: TOOL_GREP.to_owned(),
            description: crate::prompts::CHAT_TOOL_GREP.to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The text or regex to search for. Treated as a regular \
                            expression unless `literal` is true."
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional path or glob to scope the search, relative to the \
                            repo root (e.g. `src/`, `src/**/*.rs`, `Cargo.toml`). Omit to search \
                            the whole repo."
                    },
                    "literal": {
                        "type": "boolean",
                        "description": "When true, match the pattern as a literal string (regex \
                            metacharacters are escaped). Default false."
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive match when true. Default false."
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context to include on each side of a match \
                            (like grep -C), 0-10. Default 0."
                    }
                },
                "required": ["pattern"]
            }),
        },
        ToolDef {
            name: TOOL_READ.to_owned(),
            description: crate::prompts::CHAT_TOOL_READ.to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file, relative to the repository root."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-based, inclusive). Omit to start at \
                            line 1."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-based, inclusive). Omit to read to \
                            the end (subject to the per-read line cap)."
                    }
                },
                "required": ["file_path"]
            }),
        },
    ]
}

fn system_prompt(repo: &str, project_docs: &str) -> String {
    let mut prompt = crate::prompts::render(
        crate::prompts::CHAT_SYSTEM,
        &[
            ("repo", repo),
            ("tool_codebase", TOOL_CODEBASE),
            ("tool_file", TOOL_FILE),
            ("tool_grep", TOOL_GREP),
            ("tool_read", TOOL_READ),
        ],
    );

    // Seed the model with the repo's orientation docs (README/AGENTS/CLAUDE), if
    // present. These give it the project's intent up front so it isn't searching
    // from zero. They are reference material, NOT a substitute for verifying code
    // behavior — but because they are real files at known paths, the model MAY
    // cite them directly (e.g. `README.md#L1-20`) like any other evidence.
    if !project_docs.trim().is_empty() {
        prompt.push_str(crate::prompts::CHAT_PROJECT_DOCS_APPENDIX);
        prompt.push_str(project_docs);
    }

    prompt
}

// ─── Tool dispatch (hard-locked to the two allowed tools) ─────────────────

/// Execute one tool call against the fixed repo. Returns `(result_text, ok)`.
/// An unknown tool name yields an error string (not a panic) so the model can
/// recover on the next turn — this is the hard lock on the tool surface.
async fn run_tool(
    deps: &ChatTurnDeps,
    repo: &str,
    name: &str,
    args: &serde_json::Value,
) -> (String, bool) {
    match name {
        TOOL_CODEBASE => {
            let req = args.get("information_request").and_then(|v| v.as_str()).unwrap_or("");
            if req.trim().is_empty() {
                return ("Error: information_request is required.".to_owned(), false);
            }
            let out = crate::mcp::run_codebase_retrieval(
                &deps.home_dir, &deps.data_dir, &deps.index_engine, &deps.repo_dbs,
                &deps.settings, req, repo,
            )
            .await;
            let ok = !out.starts_with("Error:");
            (out, ok)
        }
        TOOL_FILE => {
            let file_path = args.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
            let req = args.get("information_request").and_then(|v| v.as_str()).unwrap_or("");
            if file_path.trim().is_empty() || req.trim().is_empty() {
                return ("Error: file_path and information_request are required.".to_owned(), false);
            }
            let out = crate::mcp::run_file_retrieval(
                &deps.data_dir, &deps.repo_dbs, &deps.settings, repo, file_path, req, 5,
            )
            .await;
            let ok = !out.starts_with("Error:");
            (out, ok)
        }
        TOOL_GREP => {
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            if pattern.trim().is_empty() {
                return ("Error: pattern is required.".to_owned(), false);
            }
            let path = args.get("path").and_then(|v| v.as_str()).map(str::to_owned);
            let literal = args.get("literal").and_then(|v| v.as_bool()).unwrap_or(false);
            let ignore_case = args.get("ignore_case").and_then(|v| v.as_bool()).unwrap_or(false);
            let context_lines = args
                .get("context_lines")
                .and_then(|v| v.as_u64())
                .map(|n| (n as usize).min(GREP_MAX_CONTEXT))
                .unwrap_or(0);
            // Off the async runtime: walking the tree + regex over file bytes is
            // blocking, CPU/IO-bound work that must not stall the reactor.
            let root = std::path::PathBuf::from(crate::store::normalize_repo_path(repo));
            let pattern = pattern.to_owned();
            let outcome = tokio::task::spawn_blocking(move || {
                run_grep(&root, &pattern, path.as_deref(), literal, ignore_case, context_lines)
            })
            .await;
            // The chat agent reads only the formatted text; grep regions are for
            // the results-producing reranker, not this read-only agent.
            match outcome {
                Ok(o) => (o.text, o.ok),
                Err(e) => (format!("Error: grep task failed: {e}"), false),
            }
        }
        TOOL_READ => {
            let file_path = args.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
            if file_path.trim().is_empty() {
                return ("Error: file_path is required.".to_owned(), false);
            }
            // serde_json numbers may arrive as f64; clamp to a sane 1-based line.
            let start_line = args.get("start_line").and_then(|v| v.as_u64()).map(|n| n as usize);
            let end_line = args.get("end_line").and_then(|v| v.as_u64()).map(|n| n as usize);
            let root = std::path::PathBuf::from(crate::store::normalize_repo_path(repo));
            let file_path = file_path.to_owned();
            let outcome = tokio::task::spawn_blocking(move || {
                run_read(&root, &file_path, start_line, end_line)
            })
            .await;
            match outcome {
                Ok(o) => (o.text, o.ok),
                Err(e) => (format!("Error: read task failed: {e}"), false),
            }
        }
        // Hard lock: anything outside the allowed tools is refused.
        other => (
            format!(
                "Error: tool '{other}' is not available. You may only use '{TOOL_CODEBASE}', \
                 '{TOOL_FILE}', '{TOOL_GREP}', and '{TOOL_READ}'."
            ),
            false,
        ),
    }
}

// ─── Exact filesystem tools: path guard, grep, read ──────────────────────
// `resolve_within_root`, `run_grep`, and `run_read` live in `crate::fs_tools`,
// shared verbatim with the Agentic RAG reranker. This agent calls them (see the
// TOOL_GREP/TOOL_READ arms above) and uses only the formatted `.text` output.

/// Short, human-friendly label for a tool call shown in the UI.
fn tool_summary(name: &str, args: &serde_json::Value) -> String {
    match name {
        TOOL_CODEBASE => args
            .get("information_request")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned(),
        TOOL_FILE => {
            let f = args.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
            let r = args.get("information_request").and_then(|v| v.as_str()).unwrap_or("");
            format!("{f} — {r}")
        }
        TOOL_GREP => {
            let p = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            match args.get("path").and_then(|v| v.as_str()).filter(|s| !s.trim().is_empty()) {
                Some(scope) => format!("{p}  in {scope}"),
                None => p.to_owned(),
            }
        }
        TOOL_READ => {
            let f = args.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
            match (
                args.get("start_line").and_then(|v| v.as_u64()),
                args.get("end_line").and_then(|v| v.as_u64()),
            ) {
                (Some(s), Some(e)) => format!("{f} L{s}-{e}"),
                (Some(s), None) => format!("{f} from L{s}"),
                _ => f.to_owned(),
            }
        }
        other => other.to_owned(),
    }
}

fn preview(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= PREVIEW_CHARS {
        trimmed.to_owned()
    } else {
        let truncated: String = trimmed.chars().take(PREVIEW_CHARS).collect();
        format!("{truncated}…")
    }
}

// ─── Explicit search-count floor (user-directed search depth) ─────────────

/// Upper bound on the search floor a user message may demand. The loop budget
/// is [`MAX_TURNS`]; we leave at least one round for the model to synthesize a
/// final answer, so a user can never pin every round to a forced tool call.
const MAX_REQUESTED_SEARCHES: usize = (MAX_TURNS as usize) - 1;

/// Parse an explicit "search N times" style instruction out of the user's
/// message and return the number of DISTINCT codebase searches they demand as a
/// hard floor (clamped to [`MAX_REQUESTED_SEARCHES`]). Returns `None` when the
/// message contains no such directive — the normal model-decides path.
///
/// This is deliberately a HARD lever, not a prompt hint: when it returns
/// `Some(n)`, the loop keeps `tool_choice: required` on until `n` distinct
/// searches have actually run, so the model physically cannot answer early
/// (the provider refuses to emit prose under `required`). Matches en/vi/zh
/// phrasings; recognizes ASCII digits and a few common number words. Conservative
/// by design — when unsure it returns `None` and we fall back to the model's
/// judgment rather than over-forcing an ordinary question.
fn requested_search_floor(message: &str) -> Option<usize> {
    let lower = message.to_lowercase();
    // Only engage when the user is plainly talking about searching/looking, so
    // an unrelated number in the question ("fix bug 3") never triggers forcing.
    const SEARCH_CUES: [&str; 8] =
        ["search", "queries", "query", "tìm", "tra cứu", "搜索", "查", "检索"];
    if !SEARCH_CUES.iter().any(|c| lower.contains(c)) {
        return None;
    }

    // Tokenize on non-alphanumerics, scan for a number adjacent (within a small
    // window) to a search/count cue. "search 3 times", "tìm 3 lần", "search at
    // least 3", "do 5 searches", "搜索3次" (the digit splits out as its own token).
    let toks: Vec<&str> = message
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    const COUNT_CUES: [&str; 9] = [
        "search", "searches", "times", "queries", "query", "lần", "lan", "次", "遍",
    ];
    let mut best: Option<usize> = None;
    for (i, tok) in toks.iter().enumerate() {
        let Some(n) = parse_count_token(tok) else { continue };
        if n == 0 {
            continue;
        }
        // A number counts only if a search/count cue sits within 2 tokens either
        // side — "search 3 times" (cue before) or "3 searches" (cue after).
        let lo = i.saturating_sub(2);
        let hi = (i + 3).min(toks.len());
        let near_cue = toks[lo..hi].iter().enumerate().any(|(j, t)| {
            lo + j != i && {
                let tl = t.to_lowercase();
                COUNT_CUES.iter().any(|c| tl == *c || tl.starts_with(c))
            }
        });
        if near_cue {
            best = Some(best.map_or(n, |b| b.max(n)));
        }
    }
    best.map(|n| n.clamp(1, MAX_REQUESTED_SEARCHES))
}

/// Parse one token as a small positive integer: ASCII digits, or a handful of
/// number words (en/vi/zh) up to the cap we care about.
fn parse_count_token(tok: &str) -> Option<usize> {
    if let Ok(n) = tok.parse::<usize>() {
        return Some(n);
    }
    match tok.to_lowercase().as_str() {
        "two" | "hai" | "二" | "两" => Some(2),
        "three" | "ba" | "三" => Some(3),
        "four" | "bốn" | "bon" | "四" => Some(4),
        "five" | "năm" | "nam" | "五" => Some(5),
        "six" | "sáu" | "sau" | "六" => Some(6),
        "seven" | "bảy" | "bay" | "七" => Some(7),
        _ => None,
    }
}

// ─── Tool-context summarization (cross-turn memory) ───────────────────────

/// Max location headers kept from one tool result's summary. A search returns
/// many blocks; we keep the top handful so the next turn knows roughly what was
/// already found without replaying full source.
const MAX_LOCATIONS_PER_RESULT: usize = 8;
/// Chars of the first content line kept as a per-location preview.
const LOCATION_PREVIEW_CHARS: usize = 80;

/// Whether a line looks like a retrieval location header (`path#L10-40…`) rather
/// than a numbered source line. Headers carry `#L<digit>` and are not indented.
fn is_location_header(line: &str) -> bool {
    if line.starts_with(char::is_whitespace) {
        return false;
    }
    if let Some(pos) = line.find("#L") {
        return line[pos + 2..].starts_with(|c: char| c.is_ascii_digit());
    }
    false
}

/// Build a compact, capped summary of ONE tool result for cross-turn replay:
/// the leading location headers (`path#L10-40`) each followed by a short preview
/// of their first content line. Never includes full source. The `label`
/// (information_request / file path) anchors what the search was for.
///
/// The output is intentionally small; the store still hard-caps it again at
/// [`TOOL_CTX_PER_TURN_CAP`], so this is a best-effort shrink, not the bound.
fn summarize_tool_result(label: &str, output: &str) -> String {
    // Errors / "use grep" guidance carry no locations worth replaying.
    if output.starts_with("Error:") {
        return String::new();
    }

    let mut locations: Vec<String> = Vec::new();
    let mut lines = output.lines().peekable();
    while let Some(line) = lines.next() {
        if !is_location_header(line) {
            continue;
        }
        // Peek the next non-empty line as a one-line preview of the block.
        let preview_line = lines
            .peek()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !is_location_header(l))
            .map(|l| {
                let mut s: String = l.chars().take(LOCATION_PREVIEW_CHARS).collect();
                if l.chars().count() > LOCATION_PREVIEW_CHARS {
                    s.push('…');
                }
                s
            })
            .unwrap_or_default();

        if preview_line.is_empty() {
            locations.push(format!("- {}", line.trim()));
        } else {
            locations.push(format!("- {}  ({preview_line})", line.trim()));
        }
        if locations.len() >= MAX_LOCATIONS_PER_RESULT {
            break;
        }
    }

    if locations.is_empty() {
        return String::new();
    }
    let label = label.trim();
    if label.is_empty() {
        locations.join("\n")
    } else {
        format!("search: {label}\n{}", locations.join("\n"))
    }
}

/// Prepend the prior-turns tool-context block to the new question so the model
/// can reuse earlier search evidence instead of blindly re-searching. Framed as
/// reference material, NOT as ground truth — the model must still verify with a
/// fresh search when the prior evidence doesn't cover the new question.
///
/// `pub` so wire/integration tests can prove the augmented question (carrying
/// prior tool-context) is what actually reaches the provider.
pub fn augment_question_with_context(tool_context: &str, message: &str) -> String {
    if tool_context.trim().is_empty() {
        return message.to_owned();
    }
    format!(
        "[Context from earlier in this conversation — files and ranges already \
found via search. Reuse this to stay consistent; search again only if it does \
not cover the new question, and do not treat it as a substitute for verifying \
current details.]\n{tool_context}\n\n[Current question]\n{message}"
    )
}

// ─── The streaming agentic loop ───────────────────────────────────────────

/// Inputs the loop needs. Grouped to keep the signature readable.
pub struct ChatTurnDeps {
    pub home_dir: std::path::PathBuf,
    pub data_dir: std::path::PathBuf,
    pub index_engine: Arc<IndexEngine>,
    pub repo_dbs: Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    pub settings: Settings,
    pub conversations: Arc<ConversationStore>,
}

/// Run one chat turn: stream the answer for `message` in conversation `id` for
/// `repo`, emitting [`ChatEvent`]s on `tx`. On success the (user, answer) pair
/// is appended to the conversation transcript. Any failure ends with an
/// `Error` event — the loop never hangs silently.
///
/// `llm` is the client built from `settings.llm`; the caller is responsible for
/// emitting an `Error` event when no client could be built (no keys).
pub async fn run_chat_turn(
    deps: &ChatTurnDeps,
    llm: &LlmClient,
    repo: &str,
    conversation_id: &str,
    message: &str,
    tx: &mpsc::UnboundedSender<ChatEvent>,
) {
    let tools = tool_defs();
    // Read the repo's orientation docs (README/AGENTS/CLAUDE) fresh each turn so
    // they reflect the current files; reading is just 3 small root files, and the
    // result lands in the system prompt, which is cached under `cache_key`, so
    // turns 2+ on this conversation re-use it at no token cost. Off the async
    // runtime via spawn_blocking — std fs is blocking. Normalize the repo path so
    // we read the canonical root (matches how the rest of the engine keys repos).
    let project_docs = {
        let root = std::path::PathBuf::from(crate::store::normalize_repo_path(repo));
        tokio::task::spawn_blocking(move || collect_project_docs(&root))
            .await
            .unwrap_or_default()
    };
    let system = system_prompt(repo, &project_docs);
    let cache_key = format!("repo-chat-{}", crate::store::sanitize_repo_name(repo));

    // Seed the working context with the prior transcript + this question. The
    // question is augmented with a compact summary of the search evidence from
    // the last K turns so the model can build on it instead of re-searching
    // from scratch (cross-turn memory; bounded by the store's byte caps).
    let mut messages: Vec<ChatMessage> = deps.conversations.snapshot(conversation_id, repo).await;
    let prior_context = deps.conversations.recent_tool_context(conversation_id, repo).await;
    messages.push(ChatMessage::User(augment_question_with_context(&prior_context, message)));

    let mut answer = String::new();
    // Accumulates the summaries of every search this turn runs, for storage so
    // the NEXT turn can reuse them. Capped again at store time.
    let mut turn_tool_context = String::new();

    // HARD search floor: if the user explicitly asked for N searches ("search 3
    // times", "tìm 3 lần", "搜索3次"), force `tool_choice: required` until N
    // DISTINCT codebase searches have actually run — the provider then refuses to
    // emit a final answer early, so the directive can't be silently ignored.
    // `None` = no such instruction → the model decides as before.
    let search_floor = requested_search_floor(message);
    // Distinct codebase-retrieval queries run so far this turn (normalized), so
    // the model can't satisfy the floor by repeating the same query.
    let mut distinct_searches: std::collections::HashSet<String> = std::collections::HashSet::new();

    for _turn in 0..MAX_TURNS {
        // Force a tool call while the user-requested search floor is unmet. This
        // is the wire-level lever (`tool_choice: required`), not a prompt hint:
        // under it the model physically cannot return prose, so it must search.
        let force_tool_use = match search_floor {
            Some(n) => distinct_searches.len() < n,
            None => false, // the model decides when to search; the prompt drives it
        };

        // `tool_choice: required` only guarantees SOME tool is called, not which.
        // While the floor is unmet we therefore also NARROW the offered tools to
        // codebase-search alone — forced-to-call + only-one-tool = a guaranteed
        // codebase search that advances the floor. Without this the model could
        // burn every forced round on `file-retrieval` and never satisfy the floor.
        // Once the floor is met, the full tool set (incl. graph/file hops) returns.
        let turn_tools: Vec<ToolDef> = if force_tool_use {
            tools.iter().filter(|t| t.name == TOOL_CODEBASE).cloned().collect()
        } else {
            tools.clone()
        };

        // Stream this turn. Text deltas are forwarded live as Token events.
        let token_tx = tx.clone();
        let on_token = move |t: &str| {
            let _ = token_tx.send(ChatEvent::Token { text: t.to_owned() });
        };

        let result = llm
            .complete_with_tools_streaming(
                &system,
                &messages,
                &turn_tools,
                0.2,
                force_tool_use,
                Some(&cache_key),
                &on_token,
            )
            .await;

        match result {
            Ok(ToolTurnResult::Text(text)) => {
                // Final answer. Tokens were already streamed live via on_token;
                // `text` is the full accumulation kept for the transcript. How
                // much to search is entirely the model's call — the system
                // prompt's blast-radius policy guides it; nothing is injected.
                answer = text;
                break;
            }
            Ok(ToolTurnResult::ToolCalls(calls)) => {
                // Record the model's tool-call turn for replay within THIS turn.
                messages.push(ChatMessage::ModelToolCalls(calls.clone()));

                let mut results = Vec::with_capacity(calls.len());
                for call in &calls {
                    let summary = tool_summary(&call.name, &call.args);
                    let _ = tx.send(ChatEvent::ToolCall {
                        name: call.name.clone(),
                        summary: summary.clone(),
                    });

                    let (out, ok) = run_tool(deps, repo, &call.name, &call.args).await;

                    // Count a DISTINCT, successful codebase search toward the
                    // user-requested floor. Normalized so the model can't satisfy
                    // "search 3 times" by issuing the same query thrice.
                    if ok
                        && call.name == TOOL_CODEBASE
                        && let Some(q) = call.args.get("information_request").and_then(|v| v.as_str())
                    {
                        let norm = q.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
                        if !norm.is_empty() {
                            distinct_searches.insert(norm);
                        }
                    }

                    let _ = tx.send(ChatEvent::ToolResult {
                        name: call.name.clone(),
                        ok,
                        preview: preview(&out),
                    });

                    // Fold a compact summary of this result into the turn's
                    // cross-turn memory (best-effort; store enforces the cap).
                    if ok {
                        let s = summarize_tool_result(&summary, &out);
                        if !s.is_empty() {
                            if !turn_tool_context.is_empty() {
                                turn_tool_context.push_str("\n\n");
                            }
                            turn_tool_context.push_str(&s);
                        }
                    }

                    results.push(ToolResult {
                        name: call.name.clone(),
                        id: call.id.clone(),
                        content: out,
                    });
                }
                messages.push(ChatMessage::ToolResults(results));
                // Loop: let the model read the results and continue/answer.
            }
            Err(e) => {
                let _ = tx.send(ChatEvent::Error {
                    message: format!("LLM request failed: {e}"),
                });
                return;
            }
        }
    }

    if answer.is_empty() {
        // We exhausted the tool-calling budget without the model producing a
        // final text answer (it kept searching). Rather than dead-ending with an
        // error, make ONE last pass with no tools available, forcing the model to
        // synthesize a safe answer from whatever context it has already gathered
        // — honestly flagging gaps instead of guessing. This is the graceful
        // "ran out of searches → say what you can, admit the rest" exit.
        let token_tx = tx.clone();
        // Track whether the final pass actually streamed any token. `started` in
        // the llm layer is internal and invisible here, so we observe emission
        // directly: if even one token reached the user, appending the fallback
        // sentence would produce "half-answer + fallback" garble — so we only
        // emit the fallback when nothing was streamed.
        let streamed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let streamed_cb = streamed.clone();
        let on_token = move |t: &str| {
            if !t.is_empty() {
                streamed_cb.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            let _ = token_tx.send(ChatEvent::Token { text: t.to_owned() });
        };
        messages.push(ChatMessage::User(
            "You have used up your search budget for this question. Do not request any more \
             tools. Answer now using only the context already gathered above. State clearly \
             which parts of the question you could verify and which parts you could not find \
             enough evidence for — do not invent details for the gaps."
                .to_owned(),
        ));
        // Pass an empty tool list so no further tool calls are possible.
        match llm
            .complete_with_tools_streaming(
                &system, &messages, &[], 0.2, false, Some(&cache_key), &on_token,
            )
            .await
        {
            Ok(ToolTurnResult::Text(text)) if !text.trim().is_empty() => answer = text,
            // The model produced no usable accumulated text, or the call failed.
            // Emit a plain honest fallback — but ONLY if nothing was streamed,
            // so we never glue a fallback sentence onto a half-streamed answer.
            // If tokens already reached the user (mid-stream failure with empty
            // accumulation), leave `answer` for the guard below to backfill so
            // the transcript isn't empty, without emitting more visible text.
            _ => {
                if !streamed.load(std::sync::atomic::Ordering::Relaxed) {
                    let fallback = "I couldn't gather enough indexed context to answer this \
                        confidently. Try rephrasing the question, narrowing it to one part, or \
                        pointing me at a specific file."
                        .to_owned();
                    let _ = tx.send(ChatEvent::Token {
                        text: fallback.clone(),
                    });
                    answer = fallback;
                }
            }
        }
    }

    // The final-pass branch above may leave `answer` empty only if the stream
    // emitted tokens but returned no usable accumulated text — extremely rare,
    // but guard the transcript against storing an empty assistant turn.
    if answer.trim().is_empty() {
        answer = "I couldn't gather enough indexed context to answer this confidently."
            .to_owned();
    }

    deps.conversations
        .append_turn(conversation_id, repo, message.to_owned(), answer, turn_tool_context)
        .await;
    let _ = tx.send(ChatEvent::Done);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_tool_is_refused() {
        // The unknown-tool arm must produce an error string, never panic.
        // It is pure string formatting (short-circuits before any dependency),
        // so assert the message shape directly.
        let msg = format!(
            "Error: tool '{other}' is not available. You may only use '{TOOL_CODEBASE}', \
                 '{TOOL_FILE}', '{TOOL_GREP}', and '{TOOL_READ}'.",
            other = "rm-rf"
        );
        assert!(msg.starts_with("Error: tool 'rm-rf' is not available"));
        assert!(
            msg.contains(TOOL_CODEBASE)
                && msg.contains(TOOL_FILE)
                && msg.contains(TOOL_GREP)
                && msg.contains(TOOL_READ)
        );
    }

    #[test]
    fn tool_summary_uses_information_request() {
        let args = serde_json::json!({ "information_request": "how does sharding work" });
        assert_eq!(tool_summary(TOOL_CODEBASE, &args), "how does sharding work");
    }

    #[test]
    fn preview_truncates_long_text() {
        let long = "x".repeat(PREVIEW_CHARS + 50);
        let p = preview(&long);
        assert!(p.ends_with('…'));
        assert!(p.chars().count() <= PREVIEW_CHARS + 1);
    }

    #[test]
    fn preview_keeps_short_text() {
        assert_eq!(preview("  hello  "), "hello");
    }

    #[test]
    fn search_floor_parses_explicit_counts() {
        // en
        assert_eq!(requested_search_floor("git features, search 3 times"), Some(3));
        assert_eq!(requested_search_floor("do 5 searches please"), Some(5));
        assert_eq!(requested_search_floor("search at least three times"), Some(3));
        // vi
        assert_eq!(requested_search_floor("các tính năng git, search 3 lần"), Some(3));
        assert_eq!(requested_search_floor("tìm 4 lần để cover blast radius"), Some(4));
        // zh (digit splits from 次 as its own token via the alphanumeric split)
        assert_eq!(requested_search_floor("搜索 3 次"), Some(3));
    }

    #[test]
    fn search_floor_ignores_unrelated_numbers() {
        // No search cue → never engage.
        assert_eq!(requested_search_floor("fix bug 3 in the parser"), None);
        // Search cue but no count near it → no floor.
        assert_eq!(requested_search_floor("search the codebase for the parser"), None);
        // Number present but not adjacent to a count/search cue.
        assert_eq!(requested_search_floor("search for the 42nd handler signature"), None);
    }

    #[test]
    fn search_floor_clamps_to_budget() {
        // A huge request is clamped to MAX_REQUESTED_SEARCHES, never the raw N.
        assert_eq!(requested_search_floor("search 99 times"), Some(MAX_REQUESTED_SEARCHES));
        // Zero is meaningless and ignored.
        assert_eq!(requested_search_floor("search 0 times"), None);
    }

    #[tokio::test]
    async fn store_evicts_lru_over_cap() {
        let store = ConversationStore::new();
        for i in 0..(MAX_CONVERSATIONS + 5) {
            let id = format!("conv-{i}");
            store
                .append_turn(&id, "/repo", "q".to_owned(), "a".to_owned(), String::new())
                .await;
        }
        let map = store.inner.lock().await;
        assert!(map.len() <= MAX_CONVERSATIONS + 1, "store must stay near the cap");
    }

    #[tokio::test]
    async fn store_trims_old_turns() {
        let store = ConversationStore::new();
        for i in 0..(MAX_TURNS_KEPT + 10) {
            store
                .append_turn("c1", "/repo", format!("q{i}"), format!("a{i}"), String::new())
                .await;
        }
        let map = store.inner.lock().await;
        let conv = map.get("c1").unwrap();
        assert!(conv.turns.len() <= MAX_TURNS_KEPT);
    }

    #[tokio::test]
    async fn drop_removes_conversation() {
        let store = ConversationStore::new();
        store.append_turn("c1", "/repo", "q".to_owned(), "a".to_owned(), String::new()).await;
        store.drop_conversation("c1").await;
        assert!(store.inner.lock().await.get("c1").is_none());
    }

    #[tokio::test]
    async fn snapshot_resets_on_repo_mismatch() {
        let store = ConversationStore::new();
        store.append_turn("c1", "/repo-a", "q".to_owned(), "a".to_owned(), String::new()).await;
        // Same id, different repo → treated as empty (fresh) conversation.
        let snap = store.snapshot("c1", "/repo-b").await;
        assert!(snap.is_empty());
    }

    // ─── Cross-turn tool-context ──────────────────────────────────────────

    #[test]
    fn truncate_bytes_caps_and_marks() {
        let s = "a".repeat(100);
        let t = truncate_bytes(&s, 10);
        assert!(t.ends_with('…'));
        // 10 bytes of content + the multi-byte ellipsis.
        assert!(t.len() <= 10 + '…'.len_utf8());
    }

    #[test]
    fn truncate_bytes_keeps_short() {
        assert_eq!(truncate_bytes("hi", 10), "hi");
    }

    #[test]
    fn truncate_bytes_respects_char_boundary() {
        // Multi-byte chars must never be split mid-codepoint.
        let s = "é".repeat(50); // each 'é' is 2 bytes
        let t = truncate_bytes(&s, 5);
        assert!(t.ends_with('…'));
        // Must still be valid UTF-8 (no panic on slicing) — implicit by reaching here.
    }

    #[test]
    fn is_location_header_detects_path_ranges() {
        assert!(is_location_header("src/foo.rs#L10-40"));
        assert!(is_location_header("a/b/c.py#L1-2 [callers: 3]"));
        // Numbered source lines and indented lines are not headers.
        assert!(!is_location_header("   10: let x = 1;"));
        assert!(!is_location_header("fn main() {"));
        assert!(!is_location_header("path#Lstart")); // no digit after #L
    }

    #[test]
    fn summarize_extracts_locations_with_previews() {
        let output = "src/a.rs#L1-5\n1: fn a() {}\n\nsrc/b.rs#L9-12\n9: struct B;";
        let s = summarize_tool_result("how does a work", output);
        assert!(s.contains("search: how does a work"));
        assert!(s.contains("src/a.rs#L1-5"));
        assert!(s.contains("src/b.rs#L9-12"));
        // Preview text from the first content line is included.
        assert!(s.contains("fn a()"));
    }

    #[test]
    fn summarize_ignores_errors() {
        assert!(summarize_tool_result("x", "Error: no index").is_empty());
    }

    #[test]
    fn summarize_caps_location_count() {
        let mut output = String::new();
        for i in 0..(MAX_LOCATIONS_PER_RESULT + 10) {
            output.push_str(&format!("src/f{i}.rs#L1-2\n1: code\n\n"));
        }
        let s = summarize_tool_result("many", &output);
        let count = s.matches("#L").count();
        assert_eq!(count, MAX_LOCATIONS_PER_RESULT, "must cap locations per result");
    }

    #[test]
    fn augment_noop_when_no_context() {
        assert_eq!(augment_question_with_context("", "hello"), "hello");
        assert_eq!(augment_question_with_context("   ", "hello"), "hello");
    }

    #[test]
    fn augment_embeds_context_and_question() {
        let out = augment_question_with_context("src/a.rs#L1-5", "what next?");
        assert!(out.contains("src/a.rs#L1-5"));
        assert!(out.contains("what next?"));
        assert!(out.contains("[Current question]"));
    }

    #[tokio::test]
    async fn recent_tool_context_replays_recent_turns() {
        let store = ConversationStore::new();
        store.append_turn("c1", "/r", "q1".into(), "a1".into(), "ctx-1".into()).await;
        store.append_turn("c1", "/r", "q2".into(), "a2".into(), "ctx-2".into()).await;
        let ctx = store.recent_tool_context("c1", "/r").await;
        assert!(ctx.contains("ctx-1"));
        assert!(ctx.contains("ctx-2"));
        // Oldest-first ordering.
        assert!(ctx.find("ctx-1").unwrap() < ctx.find("ctx-2").unwrap());
    }

    #[tokio::test]
    async fn recent_tool_context_windows_to_k_turns() {
        let store = ConversationStore::new();
        for i in 0..(TOOL_CTX_TURNS_KEPT + 3) {
            store
                .append_turn("c1", "/r", format!("q{i}"), format!("a{i}"), format!("ctx-{i}"))
                .await;
        }
        let ctx = store.recent_tool_context("c1", "/r").await;
        // The oldest summaries fall outside the K-turn window.
        assert!(!ctx.contains("ctx-0"));
        // The newest K are present.
        let newest = TOOL_CTX_TURNS_KEPT + 2;
        assert!(ctx.contains(&format!("ctx-{newest}")));
    }

    #[tokio::test]
    async fn recent_tool_context_skips_empty_summaries() {
        let store = ConversationStore::new();
        // Chit-chat turns (no search) store empty tool_context and are skipped.
        store.append_turn("c1", "/r", "hi".into(), "hello".into(), String::new()).await;
        store.append_turn("c1", "/r", "q".into(), "a".into(), "real-ctx".into()).await;
        store.append_turn("c1", "/r", "thanks".into(), "yw".into(), String::new()).await;
        let ctx = store.recent_tool_context("c1", "/r").await;
        assert_eq!(ctx, "real-ctx");
    }

    #[tokio::test]
    async fn per_turn_context_capped_at_store_time() {
        let store = ConversationStore::new();
        let huge = "x".repeat(TOOL_CTX_PER_TURN_CAP * 4);
        store.append_turn("c1", "/r", "q".into(), "a".into(), huge).await;
        let map = store.inner.lock().await;
        let conv = map.get("c1").unwrap();
        // Stored summary must be hard-bounded regardless of input size.
        assert!(conv.turns[0].tool_context.len() <= TOOL_CTX_PER_TURN_CAP + '…'.len_utf8());
    }

    #[tokio::test]
    async fn total_injected_context_capped_across_turns() {
        let store = ConversationStore::new();
        // Each turn sits near the per-turn cap; K of them combined would exceed
        // the total cap, so the injected block must be trimmed to it.
        let near_cap = "y".repeat(TOOL_CTX_PER_TURN_CAP - 10);
        for i in 0..TOOL_CTX_TURNS_KEPT {
            store
                .append_turn("c1", "/r", format!("q{i}"), format!("a{i}"), near_cap.clone())
                .await;
        }
        let ctx = store.recent_tool_context("c1", "/r").await;
        assert!(
            ctx.len() <= TOOL_CTX_TOTAL_CAP + '…'.len_utf8(),
            "total injected context must be bounded; got {} bytes",
            ctx.len()
        );
    }

    // ─── Project documentation injection ──────────────────────────────────

    #[test]
    fn truncate_doc_keeps_short_unchanged() {
        let s = "line one\nline two\n";
        assert_eq!(truncate_doc_by_lines(s, 1024), s);
    }

    #[test]
    fn truncate_doc_cuts_on_line_boundary_with_marker() {
        let s = "aaaa\nbbbb\ncccc\ndddd\n"; // each line 4 chars + newline
        let t = truncate_doc_by_lines(s, 12);
        assert!(t.ends_with("… [truncated]"));
        // No partial line survives: every non-marker line is a whole original line.
        for line in t.lines() {
            if line == "… [truncated]" {
                continue;
            }
            assert!(["aaaa", "bbbb", "cccc", "dddd"].contains(&line), "got partial line {line:?}");
        }
        // Bounded: kept content (excluding marker) stays under the cap.
        assert!(t.len() <= 12 + "… [truncated]".len());
    }

    #[test]
    fn collect_docs_empty_when_none_present() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        assert_eq!(collect_project_docs(dir.path()), "");
    }

    #[test]
    fn collect_docs_reads_three_in_stable_order() {
        let dir = tempfile::tempdir().unwrap();
        // Write in a deliberately non-canonical order to prove output is sorted
        // README → AGENTS → CLAUDE regardless of dir iteration order.
        std::fs::write(dir.path().join("CLAUDE.md"), "claude body").unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "agents body").unwrap();
        std::fs::write(dir.path().join("README.md"), "readme body").unwrap();
        let out = collect_project_docs(dir.path());
        let r = out.find("README.md").unwrap();
        let a = out.find("AGENTS.md").unwrap();
        let c = out.find("CLAUDE.md").unwrap();
        assert!(r < a && a < c, "docs must be ordered README→AGENTS→CLAUDE");
        assert!(out.contains("readme body") && out.contains("agents body") && out.contains("claude body"));
    }

    #[test]
    fn collect_docs_matches_case_insensitively() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("readme.md"), "lower readme").unwrap();
        let out = collect_project_docs(dir.path());
        assert!(out.contains("lower readme"));
        // Header uses the on-disk name verbatim.
        assert!(out.contains("--- readme.md ---"));
    }

    #[test]
    fn collect_docs_skips_directories_and_empty_files() {
        let dir = tempfile::tempdir().unwrap();
        // A directory named like a doc must not be read.
        std::fs::create_dir(dir.path().join("AGENTS.md")).unwrap();
        // An empty/whitespace doc contributes nothing.
        std::fs::write(dir.path().join("CLAUDE.md"), "   \n  ").unwrap();
        std::fs::write(dir.path().join("README.md"), "real readme").unwrap();
        let out = collect_project_docs(dir.path());
        assert!(out.contains("real readme"));
        assert!(!out.contains("--- AGENTS.md ---"));
        assert!(!out.contains("--- CLAUDE.md ---"));
    }

    #[test]
    fn collect_docs_skips_non_utf8() {
        let dir = tempfile::tempdir().unwrap();
        // Invalid UTF-8 bytes — read_to_string fails, file is skipped silently.
        std::fs::write(dir.path().join("README.md"), [0xff, 0xfe, 0x00, 0x01]).unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "valid claude").unwrap();
        let out = collect_project_docs(dir.path());
        assert!(!out.contains("--- README.md ---"));
        assert!(out.contains("valid claude"));
    }

    #[test]
    fn collect_docs_enforces_total_cap() {
        let dir = tempfile::tempdir().unwrap();
        // Each file near the per-file cap; all three combined exceed the total
        // cap, so the block must be trimmed to the total budget.
        let big = "line of text padding padding padding\n".repeat(400); // > per-file cap
        std::fs::write(dir.path().join("README.md"), &big).unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), &big).unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), &big).unwrap();
        let out = collect_project_docs(dir.path());
        // Allow headroom for the three section headers + truncation markers.
        let slack = 3 * ("--- README.md ---\n".len() + "… [truncated]".len() + 8);
        assert!(
            out.len() <= PROJECT_DOC_TOTAL_CAP + slack,
            "combined docs must respect the total cap; got {} bytes",
            out.len()
        );
    }

    #[test]
    fn system_prompt_omits_doc_section_when_empty() {
        let p = system_prompt("/repo", "");
        assert!(!p.contains("Project documentation"));
    }

    #[test]
    fn system_prompt_includes_docs_when_present() {
        let p = system_prompt("/repo", "--- README.md ---\nhello world");
        assert!(p.contains("Project documentation"));
        assert!(p.contains("hello world"));
    }

    #[test]
    fn system_prompt_documents_all_four_tools() {
        let p = system_prompt("/repo", "");
        for tool in [TOOL_CODEBASE, TOOL_FILE, TOOL_GREP, TOOL_READ] {
            assert!(p.contains(tool), "system prompt must mention {tool}");
        }
    }

    #[test]
    fn tool_defs_exposes_four_tools() {
        let names: Vec<String> = tool_defs().into_iter().map(|t| t.name).collect();
        assert_eq!(names.len(), 4);
        for tool in [TOOL_CODEBASE, TOOL_FILE, TOOL_GREP, TOOL_READ] {
            assert!(names.iter().any(|n| n == tool), "tool_defs missing {tool}");
        }
    }

    // grep/read + path-traversal guard tests live with their implementation in
    // `crate::fs_tools` (shared with the reranker), not duplicated here.
}



