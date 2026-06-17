use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use regex::Regex;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tokio::sync::RwLock;
use tracing::warn;
use crate::embedding::voyage::VoyageClient;
use crate::indexing::IndexEngine;
use crate::llm::{ChatMessage, LlmClient, ToolDef, ToolResult, ToolTurnResult};
use crate::query::merger::MergeChunk;
use crate::query::engine::{run_sub_query, read_lines_from_fs, slice_numbered};

/// A chunk selection: (chunk_index, optional line ranges to keep).
type ChunkSelection = (usize, Option<Vec<(u32, u32)>>);

#[derive(Debug, Clone, serde::Serialize)]
pub struct RerankOutput {
    pub reranked_indices: Vec<usize>,
    /// Per-position line selections, aligned 1:1 with `reranked_indices`.
    /// `Some(ranges)` = LLM chose absolute line ranges to keep for that chunk
    /// (sorted, merged, clamped to chunk bounds); `None` = keep whole chunk.
    pub line_selections: Vec<Option<Vec<(u32, u32)>>>,
    pub raw_request: String,
    pub raw_response: String,
    pub elapsed_ms: u64,
    pub fallback_used: bool,
    pub skip_reason: Option<String>,
}

/// The agentic loop's addressable chunk pool: base candidates followed by any
/// chunks pulled in by the `query` tool. `reranked_indices` from an agentic
/// rerank index into THIS pool (not the caller's base `merged`), so the query
/// engine must resolve against it or sub-query selections are lost. Returned
/// only by the agentic path; the single-shot reranker addresses `merged`.
#[derive(Debug, Clone)]
pub struct ExtendedPool {
    pub chunks: Vec<MergeChunk>,
    /// Numbered-from-disk text aligned 1:1 with `chunks` (`None` = unreadable).
    pub numbered: Vec<Option<String>>,
}

/// Padding added on each side of an LLM-selected range before clamping to the
/// chunk bounds — absorbs off-by-a-line selections.
const RANGE_PAD: u32 = 2;

pub async fn rerank(
    query: &str,
    chunks: &[MergeChunk],
    numbered: &[Option<String>],
    caller_stats: &[Option<(u32, u32)>],
    min_prune_lines: u32,
    llm_client: Option<&LlmClient>,
) -> RerankOutput {
    let n = chunks.len();
    let all_indices: Vec<usize> = (0..n).collect();

    let client = match llm_client {
        Some(c) => c,
        None => return RerankOutput {
            reranked_indices: all_indices,
            line_selections: vec![None; n],
            raw_request: String::new(),
            raw_response: String::new(),
            elapsed_ms: 0,
            fallback_used: false,
            skip_reason: Some("no LLM API key configured".to_owned()),
        },
    };

    if chunks.is_empty() {
        return RerankOutput {
            reranked_indices: vec![],
            line_selections: vec![],
            raw_request: String::new(),
            raw_response: String::new(),
            elapsed_ms: 0,
            fallback_used: false,
            skip_reason: None,
        };
    }

    let structured = client.structured_output_active();

    let common_intro = "You are a code search relevance ranker. \
        Given a query and numbered code chunks with metadata (relevance score, callers count, \
        callees count, flow membership), your job is to rank the chunks by relevance to the query. \
        OMIT chunks that are not relevant to the query. \
        Higher score, more callers, and flow membership indicate higher structural importance. \
        When both source code and documentation chunks are relevant to the query, \
        prefer source code over documentation because code is the source of truth. \
        Documentation can be outdated or inaccurate, but the code always reflects actual behavior. \
        Each code line is prefixed with its absolute line number (\"123: code\"). ";

    let element_spec = "Each element MUST be an object identified by `chunk_index` (the chunk index), in ONE of two forms: \
        to narrow a large chunk to the relevant parts, use \
        {\"chunk_index\": <index>, \"lines\": [[start, end], ...]} where `lines` are absolute line-number \
        ranges to keep from that chunk; \
        to keep an entire chunk (small chunks, or chunks that are wholly relevant), use \
        {\"chunk_index\": <index>, \"keep\": \"full\"}. \
        Only include chunks that are actually relevant to the query.";

    let system = if structured {
        format!(
            "{common_intro}\
            Respond with a single JSON object with exactly one key, `ranked_indices`, whose value \
            is a JSON array of objects ordered from most relevant to least relevant. \
            {element_spec} \
            Output only the JSON object — no prose, no code fences."
        )
    } else {
        format!(
            "{common_intro}\
            Your output MUST contain a pair of XML tags called ranked_indices. \
            Between the opening <ranked_indices> tag and the closing </ranked_indices> tag, \
            place a JSON array of objects, ordered from most relevant to least relevant. \
            {element_spec} \
            Do not include any other text between the tags, only the JSON array."
        )
    };
    let system = system.as_str();

    // Build user prompt with chunk entries. Use the disk-numbered content
    // (same text the server will slice) so the line numbers the LLM selects map
    // exactly to what is returned. Fall back to stored content when the file
    // could not be read — such chunks are NOT line-prunable downstream.
    let mut entries = Vec::with_capacity(n);
    for (i, chunk) in chunks.iter().enumerate() {
        let stats = caller_stats.get(i).copied().flatten();
        let meta_str = match stats {
            Some((callers, files)) => format!("score={:.2} callers={callers} files={files}", chunk.score),
            None => format!("score={:.2}", chunk.score),
        };
        let symbol_display = chunk.symbol.as_deref().unwrap_or("no symbol");

        let raw = numbered
            .get(i)
            .and_then(|c| c.as_deref())
            .unwrap_or(&chunk.content);
        // Truncate content at 100 lines
        let content = truncate_content(raw, 100);

        let entry = format!(
            "[{i}] {meta_str} | {}:{}-{} ({symbol_display})\n<content chunk-index=\"{i}\">\n{content}\n</content>",
            chunk.file, chunk.line_start, chunk.line_end
        );
        entries.push(entry);
    }

    let chunks_text = entries.join("\n---\n");
    let query_json = serde_json::to_string(query).unwrap_or_else(|_| format!("\"{}\"", query));
    let user_prompt = if structured {
        format!(
            "Query: {query_json}\n\nChunks:\n{chunks_text}\n\n\
             Now rank the chunks by relevance. Respond with a JSON object \
             {{\"ranked_indices\": [ ... ]}} whose array holds objects — \
             {{\"chunk_index\":index,\"lines\":[[start,end]]}} to narrow, or {{\"chunk_index\":index,\"keep\":\"full\"}} \
             to keep the whole chunk — from most to least relevant."
        )
    } else {
        format!(
            "Query: {query_json}\n\nChunks:\n{chunks_text}\n\n\
             Now rank the chunks by relevance. \
             Write the opening tag <ranked_indices>, then a JSON array of objects — \
             {{\"chunk_index\":index,\"lines\":[[start,end]]}} to narrow, or {{\"chunk_index\":index,\"keep\":\"full\"}} \
             to keep the whole chunk — from most to least relevant, then the closing tag \
             </ranked_indices>."
        )
    };

    let raw_request = format!("[System]\n{system}\n\n[User]\n{user_prompt}");

    let start = Instant::now();
    let result = client.complete(system, &user_prompt, 0.0, structured).await;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(response) => {
            let mut output = parse_rerank_response(&response, chunks, min_prune_lines, elapsed_ms, structured);
            output.raw_request = raw_request;
            output
        }
        Err(e) => {
            warn!(error = %e, "LLM rerank call failed, using fallback order");
            RerankOutput {
                reranked_indices: all_indices,
                line_selections: vec![None; n],
                raw_request,
                raw_response: String::new(),
                elapsed_ms,
                fallback_used: true,
                skip_reason: Some(format!("LLM request failed: {e}")),
            }
        }
    }
}

// ─── Agentic RAG rerank ──────────────────────────────────────────────────

const AGENTIC_HISTORY_BYTE_CAP: usize = 1_000_000;

/// Max times one `add_chunks` phase may be rejected for a malformed `lines`
/// shape (a flat integer list instead of nested `[[start,end],...]` pairs)
/// before we stop nudging and accept the call as-is (keeping whole chunks for
/// the malformed entries rather than losing them). Bounds the reject→retry loop
/// so a model that simply cannot produce the nested form can't spin forever.
const MAX_FORMAT_RETRIES: u32 = 2;

/// Tool surface offered to the agentic loop. `add_chunks` + `query` are always
/// present; when `grep_read` is true the exact filesystem tools `grep` and
/// `read` are appended. Their results become addressable chunks (Design B) the
/// agent can commit via `add_chunks`; they don't consume the query turn budget.
fn agentic_tool_definitions(grep_read: bool) -> Vec<ToolDef> {
    let mut tools = vec![
        ToolDef {
            name: "add_chunks".to_owned(),
            description: "Add one or more relevant code chunks to the final results, ordered most-relevant first. \
                Each call may include many chunks at once. There is a total character budget for added \
                chunk content; once it is reached the agent stops.".to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "chunks": {
                        "type": "array",
                        "description": "Chunks to add to the final results, most relevant first.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "chunk_index": {
                                    "type": "integer",
                                    "description": "The index of the chunk to add (from the numbered chunk list)"
                                },
                                "lines": {
                                    "type": "array",
                                    "description": "Line ranges to keep from this chunk (prune mode). Each element is [start, end] using the absolute line numbers shown in the chunk. Mutually exclusive with keep.",
                                    "items": {
                                        "type": "array",
                                        "items": { "type": "integer" },
                                        "minItems": 2,
                                        "maxItems": 2
                                    }
                                },
                                "keep": {
                                    "type": "string",
                                    "enum": ["all"],
                                    "description": "Set to \"all\" to keep the entire chunk. Mutually exclusive with lines."
                                }
                            },
                            "required": ["chunk_index"]
                        }
                    }
                },
                "required": ["chunks"]
            }),
        },
        ToolDef {
            name: "query".to_owned(),
            description: "Search for additional code context. Use when the current chunks are insufficient to answer the original query. Returns new chunks you can then add via add_chunks. Each query call counts against the turn budget.".to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "information_request": {
                        "type": "string",
                        "description": "What to search for in the codebase"
                    }
                },
                "required": ["information_request"]
            }),
        },
    ];

    if grep_read {
        tools.push(ToolDef {
            name: "grep".to_owned(),
            description: "Exact text/regex search over the repository's working tree (like \
                ripgrep). Unlike query (semantic, ranked by meaning), this finds the LITERAL \
                pattern and returns matching lines as `path:line: text`, plus addressable chunk \
                indices you can pass to add_chunks. Use it to find every call site of a symbol, \
                where a constant is defined, or all uses of an identifier. Does NOT consume the \
                query turn budget. Respects .gitignore; skips binary files."
                .to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex to search for. Regex unless `literal` is true."
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional path or glob to scope the search, relative to the \
                            repo root (e.g. `src/`, `src/**/*.rs`). Omit to search the whole repo."
                    },
                    "literal": {
                        "type": "boolean",
                        "description": "Match the pattern as a literal string (escape regex \
                            metacharacters). Default false."
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive match when true. Default false."
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context on each side of a match (grep -C), 0-10. \
                            Default 0."
                    }
                },
                "required": ["pattern"]
            }),
        });
        tools.push(ToolDef {
            name: "read".to_owned(),
            description: "Read the verbatim contents of ONE file in the repository as numbered \
                lines, exactly as on disk. Unlike query (semantic chunks), this returns the raw \
                lines with no ranking, plus an addressable chunk index for the range you read so \
                you can pass it to add_chunks. Use it to see the actual code of a file you found \
                via query or grep. Optionally restrict to a line range. Does NOT consume the query \
                turn budget."
                .to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file, relative to the repository root."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-based, inclusive). Omit to start at 1."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-based, inclusive). Omit to read to the \
                            end (subject to the per-read line cap)."
                    }
                },
                "required": ["file_path"]
            }),
        });
    }

    tools
}

/// Estimate byte size of conversation history for cap enforcement.
fn estimate_history_bytes(messages: &[ChatMessage]) -> usize {
    messages.iter().map(|m| match m {
        ChatMessage::User(t) => t.len(),
        ChatMessage::Model(t) => t.len(),
        ChatMessage::ModelToolCalls(calls) => calls.iter().map(|c| c.name.len() + c.args.to_string().len() + 50).sum(),
        ChatMessage::ToolResults(results) => results.iter().map(|r| r.name.len() + r.content.len() + 50).sum(),
    }).sum()
}

/// Why the agentic loop stopped. Captured explicitly (not inferred from log
/// strings) so the final-output decision is a pure function of structured state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoopExit {
    /// LLM returned a text response (agent signalled it is done).
    AgentDone,
    /// An LLM HTTP/transport call failed mid-loop.
    LlmError,
    /// Agent used its query turn budget (`max_turns` `query` calls).
    QueryBudget,
    /// Accumulated chunk content reached the configured character budget
    /// (`max_chunk_chars`). Implies at least one add_chunks succeeded.
    ChunkCharBudget,
    /// Hard safety ceiling on TOTAL loop iterations was hit. Bounds an agent
    /// that neither finishes, queries to its budget, nor adds enough chars to
    /// trip the char budget. Memory stays bounded by this + the byte cap.
    IterationCap,
    /// Conversation history exceeded `AGENTIC_HISTORY_BYTE_CAP`.
    ByteCap,
}

/// The action the post-loop logic must take, derived purely from
/// `(exit_reason, accumulated_count)`. This is the failure-semantics table,
/// isolated so every row is unit-testable without a live LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinalAction {
    /// Emit the accumulated chunks as the reranked result.
    UseAccumulated,
    /// Emit an empty result set — the agent judged nothing relevant.
    /// NOT a fallback (`fallback_used = false`).
    EmptyRelevant,
    /// Degrade to original candidate order (`fallback_used = true`).
    Fallback,
}

/// Pure decision: given how the loop exited and how many chunks the agent
/// committed via `add_chunks`, what should the final output be?
///
/// Failure-semantics table (committed contract):
/// | exit            | accumulated == 0 | accumulated >= 1 |
/// |-----------------|------------------|------------------|
/// | AgentDone       | EmptyRelevant    | UseAccumulated   |
/// | LlmError        | Fallback         | UseAccumulated   |
/// | QueryBudget     | Fallback         | UseAccumulated   |
/// | ChunkCharBudget | Fallback*        | UseAccumulated   |
/// | IterationCap    | Fallback         | UseAccumulated   |
/// | ByteCap         | Fallback         | UseAccumulated   |
///
/// *ChunkCharBudget with accumulated == 0 is unreachable in the live loop (the
/// char total only grows when add_chunks commits a chunk), but the table is
/// total: if it ever occurs, fallback is the safe choice.
fn decide_final_action(exit: LoopExit, accumulated_count: usize) -> FinalAction {
    if accumulated_count >= 1 {
        // Once the agent has committed chunks, they are durable regardless of
        // how the loop ended. Partial success is still success.
        return FinalAction::UseAccumulated;
    }
    match exit {
        // Agent explicitly finished with zero selections → it judged nothing
        // relevant. Honor that as an empty result, not a fallback.
        LoopExit::AgentDone => FinalAction::EmptyRelevant,
        // Any non-graceful termination with zero selections → fallback to
        // original order so the caller still gets candidates.
        LoopExit::LlmError
        | LoopExit::QueryBudget
        | LoopExit::ChunkCharBudget
        | LoopExit::IterationCap
        | LoopExit::ByteCap => FinalAction::Fallback,
    }
}

/// Pure post-response budget decision. A char-budget stop (a HARD mid-turn
/// cap, already detected during dispatch and passed as `char_stop`) is the
/// only thing checked here. Query-budget enforcement lives in the dispatch
/// loop: after add_chunks when `awaiting_query && query_calls >= budget`, and
/// as a rejection on the next `query` call. This ensures the agent always gets
/// one more turn to call add_chunks after exhausting the query budget.
///
/// Isolated so the char stop is unit-testable without driving a full loop.
fn boundary_exit(char_stop: Option<LoopExit>) -> Option<LoopExit> {
    char_stop
}

/// External IO the agentic loop depends on. The real impl (`LiveBackend`)
/// wraps the LLM client + retrieval deps; tests provide a scripted mock so the
/// loop body, decision logic, AND output assembly run for real without any
/// network/DB. This is what closes the drift gap — there is no parallel copy of
/// the field-assembly code; `run_agentic_loop` is the single source of truth.
trait AgenticBackend {
    /// Execute one LLM tool-calling turn over the current conversation.
    /// `force_tool_use` forbids a prose reply (the model MUST call a tool) — set
    /// while no chunk is committed yet so the agent can't answer the query
    /// directly instead of selecting chunks.
    fn next_turn(
        &self,
        system: &str,
        messages: &[ChatMessage],
        tools: &[ToolDef],
        force_tool_use: bool,
    ) -> impl std::future::Future<Output = anyhow::Result<ToolTurnResult>> + Send;

    /// Retrieve additional chunks for the `query` tool
    /// (embed → search → expand → merge; NO rerank — cannot recurse).
    fn sub_query(
        &self,
        information_request: &str,
    ) -> impl std::future::Future<Output = anyhow::Result<Vec<MergeChunk>>> + Send;

    /// Absolute repo root for the exact filesystem tools (`grep`, `read`). The
    /// path-traversal guard in `crate::fs_tools` is anchored here, so the agent
    /// can never read outside it. `None` disables grep/read (mock backends that
    /// don't exercise them return `None`).
    fn repo_root(&self) -> Option<std::path::PathBuf>;
}

/// Production backend: forwards to the real LLM client and `run_sub_query`.
struct LiveBackend<'a> {
    llm_client: &'a LlmClient,
    prompt_cache_key: Option<String>,
    repo_filter: &'a str,
    voyage_client: &'a VoyageClient,
    index_engine: &'a Arc<IndexEngine>,
    repo_dbs: &'a Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    warm_wait: std::time::Duration,
}

impl AgenticBackend for LiveBackend<'_> {
    fn next_turn(
        &self,
        system: &str,
        messages: &[ChatMessage],
        tools: &[ToolDef],
        force_tool_use: bool,
    ) -> impl std::future::Future<Output = anyhow::Result<ToolTurnResult>> + Send {
        self.llm_client.complete_with_tools(system, messages, tools, 0.0, force_tool_use, self.prompt_cache_key.as_deref())
    }

    fn sub_query(
        &self,
        information_request: &str,
    ) -> impl std::future::Future<Output = anyhow::Result<Vec<MergeChunk>>> + Send {
        run_sub_query(
            information_request,
            30,
            self.repo_filter,
            self.voyage_client,
            self.index_engine,
            self.repo_dbs,
            self.warm_wait,
        )
    }

    fn repo_root(&self) -> Option<std::path::PathBuf> {
        // `repo_filter` is the absolute repo root (same value `path_in_repo`
        // gates sub-query results against). Normalize so the fs_tools guard
        // canonicalizes against the canonical root.
        Some(std::path::PathBuf::from(crate::store::normalize_repo_path(self.repo_filter)))
    }
}

/// Agentic rerank: multi-turn tool-calling loop. The agent selects chunks via
/// `add_chunks` and can expand search via `query`. Structurally cannot recurse
/// into rerank — `run_sub_query` has no LLM client and no rerank step.
///
/// Thin wrapper: builds the live backend and delegates to `run_agentic_loop`,
/// which holds the actual loop + decision + output-assembly logic so tests can
/// drive it with a mock backend.
#[allow(clippy::too_many_arguments)]
pub async fn rerank_agentic(
    query: &str,
    chunks: &[MergeChunk],
    numbered: &[Option<String>],
    caller_stats: &[Option<(u32, u32)>],
    min_prune_lines: u32,
    llm_client: &LlmClient,
    max_turns: u32,
    max_chunk_chars: u32,
    grep_read: bool,
    // Sub-query dependencies (for the `query` tool)
    repo_filter: &str,
    voyage_client: &VoyageClient,
    index_engine: &Arc<IndexEngine>,
    repo_dbs: &Arc<RwLock<HashMap<String, Surreal<Db>>>>,
    warm_wait: std::time::Duration,
) -> (RerankOutput, ExtendedPool) {
    use std::hash::{Hash, Hasher};
    let cache_key = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        query.hash(&mut hasher);
        repo_filter.hash(&mut hasher);
        format!("agentic-rerank-{:x}", hasher.finish())
    };
    let backend = LiveBackend {
        llm_client,
        prompt_cache_key: Some(cache_key),
        repo_filter,
        voyage_client,
        index_engine,
        repo_dbs,
        warm_wait,
    };
    run_agentic_loop(&backend, query, chunks, numbered, caller_stats, min_prune_lines, max_turns, max_chunk_chars, grep_read).await
}

/// The real agentic loop: prompt build → turn loop → `decide_final_action` →
/// output assembly. Generic over the backend so unit tests exercise THIS code
/// (not a copy) with a scripted mock.
#[allow(clippy::too_many_arguments)]
async fn run_agentic_loop<B: AgenticBackend>(
    backend: &B,
    query: &str,
    chunks: &[MergeChunk],
    numbered: &[Option<String>],
    caller_stats: &[Option<(u32, u32)>],
    min_prune_lines: u32,
    max_turns: u32,
    max_chunk_chars: u32,
    grep_read: bool,
) -> (RerankOutput, ExtendedPool) {
    let n = chunks.len();
    let all_indices: Vec<usize> = (0..n).collect();
    let start = Instant::now();

    if chunks.is_empty() {
        return (
            RerankOutput {
                reranked_indices: vec![],
                line_selections: vec![],
                raw_request: String::new(),
                raw_response: String::new(),
                elapsed_ms: 0,
                fallback_used: false,
                skip_reason: None,
            },
            ExtendedPool { chunks: vec![], numbered: vec![] },
        );
    }

    let system = "You are a code search relevance agent. \
        You are given a query and numbered code chunks with metadata. \
        Your ONLY job is to select relevant chunks via tools. NEVER answer the query. NEVER explain or summarize.\n\
        YOUR FIRST RESPONSE MUST BE a call to the `add_chunks` tool. No exceptions. No text. Call `add_chunks` immediately.\n\
        WORKFLOW — strict alternating cadence:\n\
        1. Call add_chunks ONCE with ALL relevant initial chunks (most relevant first). OMIT irrelevant chunks.\n\
        2. Call query to search for RELATED code that is NOT in the initial chunks (callers, callees, dependencies, implementations, tests, types). You MUST do this at least once to cover the full blast radius.\n\
        3. Call add_chunks ONCE with ONLY the chunks from those query results that are DIRECTLY relevant to the ORIGINAL user query (shown at the top). Do NOT add chunks just because they appeared in query results — most results will be tangential. Be highly selective.\n\
        4. Repeat steps 2-3 with a DIFFERENT query string each time.\n\
        The pattern is always: add_chunks → query → add_chunks → query → ...\n\
        You CANNOT call add_chunks twice in a row. After each add_chunks you MUST either call query or stop.\n\
        CRITICAL: Each add_chunks call is FINAL for that set of chunks. Include EVERY relevant chunk in that single call. You will NOT get another chance to add from the same set.\n\
        IMPORTANT: The initial chunks are only a starting point. They rarely cover the full picture. \
        You MUST use query to find related code: callers of key functions, type definitions, trait implementations, \
        sibling modules, test files, configuration, and anything else needed to fully understand the query topic. \
        Stopping after just the initial chunks gives an incomplete answer.\n\
        RELEVANCE FILTER: Every chunk you add MUST directly help answer the ORIGINAL user query. \
        Query results often contain tangentially related code — do NOT blindly add all results. \
        Ask yourself: \"Does this chunk contain information the user needs to answer their query?\" If not, skip it.\n\
        RULES:\n\
        - For each chunk, you MUST specify either `lines` (to prune to specific line ranges) or `keep: \"all\"` (to keep the entire chunk). Never omit both.\n\
        - LINE PRECISION IS MANDATORY: Default to using `lines` to select ONLY the specific line ranges that answer the query. \
        `keep: \"all\"` is ONLY for chunks where literally EVERY line is relevant (rare — typically only for short chunks <20 lines). \
        For most chunks, only a portion matters — use `lines: [[start, end], ...]` with the absolute line numbers shown in the chunk. \
        Adding entire large chunks wastes your character budget and dilutes the results with irrelevant code.\n\
        - DO NOT FAKE PRUNING. A range that spans the whole chunk (e.g. `[[1, 42]]` for a 42-line chunk, or `[[1, <last line>]]`) is NOT pruning — it keeps everything. \
        If you genuinely need the entire chunk, use `keep: \"all\"` instead; otherwise pick the actual sub-ranges that matter. \
        Each chunk's ranges must be chosen from ITS OWN content — never copy the same range (like `[[15, 52]]`) onto multiple different chunks. \
        If you find yourself selecting almost every line of most chunks, you are being too greedy: re-read each chunk and cut to the lines that actually answer the query.\n\
        - `lines` FORMAT — REQUIRED ON THE FIRST TRY: it MUST be an array of [start, end] PAIRS (an array OF arrays). \
        Correct: \"lines\": [[7, 11], [20, 28]]. WRONG: \"lines\": [7, 11, 20, 28] (flat list) and WRONG: \"lines\": [7, 11] (single un-nested pair). \
        A non-nested `lines` is REJECTED and you must redo the entire call — get it right the first time by always wrapping every pair in its own brackets. \
        Group consecutive lines into one [start, end] pair (lines 7,8,9 → [7, 9]); use one bracketed pair per contiguous range.\n\
        - You have a limited query budget and a character budget for total added content.\n\
        - Each query MUST use a different information_request string. Repeating the same query is not allowed.\n\
        - You may respond with ONLY the text \"[DONE]\" (nothing else) ONLY after you have called query at least once and are confident the results fully cover the query topic.";

    // When the exact filesystem tools are enabled, teach them as additional
    // EXPLORATION tools (they slot into the same add→explore cadence as `query`
    // and don't consume the query budget). Owned String so the extra block can
    // be appended conditionally; `&system` is passed to the backend below.
    let system: String = if grep_read {
        format!(
            "{system}\n\
        EXACT TOOLS (grep, read) — use ALONGSIDE query, they do NOT consume the query budget:\n\
        - `grep` does an EXACT text/regex search over the working tree. Use it to find every \
        occurrence of a symbol, where a constant/identifier is defined, or all call sites — things \
        semantic `query` ranks fuzzily. grep returns matching lines AND addressable chunk indices.\n\
        - `read` returns the verbatim numbered lines of ONE file (optionally a range), plus an \
        addressable chunk index for what it read. Use it to pull the exact code of a function you \
        found via query or grep.\n\
        - grep/read results join the same numbered pool as query results: add their reported chunk \
        indices via `add_chunks` (with `lines`) exactly like any other chunk. They count as an \
        exploration step (they satisfy the \"explore before [DONE]\" rule), so the cadence is \
        add_chunks → (query | grep | read) → add_chunks → ...\n\
        - Prefer grep/read when you need EXACT facts (a specific symbol, an exact line range); \
        prefer query when you need to discover related code by meaning."
        )
    } else {
        system.to_owned()
    };

    // Build initial user prompt with chunk entries
    let mut entries = Vec::with_capacity(n);
    for (i, chunk) in chunks.iter().enumerate() {
        let stats = caller_stats.get(i).copied().flatten();
        let meta_str = match stats {
            Some((callers, files)) => format!("score={:.2} callers={callers} files={files}", chunk.score),
            None => format!("score={:.2}", chunk.score),
        };
        let symbol_display = chunk.symbol.as_deref().unwrap_or("no symbol");
        let raw = numbered.get(i).and_then(|c| c.as_deref()).unwrap_or(&chunk.content);
        let content = truncate_content(raw, 100);
        let entry = format!(
            "[{i}] {meta_str} | {}:{}-{} ({symbol_display})\n{content}",
            chunk.file, chunk.line_start, chunk.line_end
        );
        entries.push(entry);
    }

    let query_json = serde_json::to_string(query).unwrap_or_else(|_| format!("\"{}\"", query));

    let chunks_text = entries.join("\n---\n");
    let user_prompt = format!(
        "Query: {query_json}\n\nChunks:\n{chunks_text}\n\n\
         <system-reminder>\n\
         You MUST call `add_chunks` now to select ALL relevant chunks from the list above (most relevant first). \
         Do NOT respond with text. Use the add_chunks tool.\n\
         </system-reminder>"
    );

    let tools = agentic_tool_definitions(grep_read);
    let mut messages: Vec<ChatMessage> = vec![ChatMessage::User(user_prompt.clone())];

    // Accumulated results from add_chunks calls.
    let mut accumulated: Vec<ChunkSelection> = Vec::new();
    // Cache for sub-query results (query tool). Maps information_request → formatted chunks.
    let mut query_cache: HashMap<String, Vec<MergeChunk>> = HashMap::new();
    // Track all sub-query chunks so add_chunks can reference them by extended index.
    let mut extended_chunks: Vec<MergeChunk> = chunks.to_vec();
    let mut extended_numbered: Vec<Option<String>> = numbered.to_vec();

    let mut raw_response_log = String::new();
    // Set on every loop-exit path so the final decision is driven by structured
    // state, never by scraping `raw_response_log`.
    let mut exit = LoopExit::IterationCap;

    // Turn budget: counts `query` calls. When the agent has issued this many
    // queries, the loop stops (per product spec).
    let query_budget = max_turns.max(1);
    let mut query_calls: u32 = 0;

    // Char budget: total emitted characters of added chunks. 0 disables. When
    // the running total reaches this, the agent stops. Bounds output size and,
    // together with the iteration + byte caps, keeps memory bounded.
    let char_budget = max_chunk_chars as usize;
    let mut accumulated_chars: usize = 0;

    // Enforces the add_chunks → query → add_chunks cadence. After each
    // successful add_chunks, this flips true; only a query call resets it.
    // A second add_chunks while this is true is rejected with an error nudging
    // the model toward query.
    let mut awaiting_query = false;

    // True when the most recent `query` fetched results the model has NOT yet
    // had a turn to harvest via add_chunks. If a resource cap (byte/iteration)
    // then cuts the loop, those results would be silently discarded and the run
    // would end on a wasted `query` — so we grant one final forced harvest turn
    // after the loop (see the finalization block below). Set true after a query
    // runs; cleared once add_chunks is dispatched (whether or not it picks from
    // that query's results — the model HAD its chance).
    let mut unharvested_query = false;

    // Counts how many times the CURRENT add_chunks phase has been rejected for a
    // malformed `lines` shape (flat list instead of nested pairs). Reset to 0
    // whenever a phase is accepted or the model moves on (query/grep/read). Caps
    // the reject→retry loop at MAX_FORMAT_RETRIES so a model that can't produce
    // pairs eventually has its call accepted as-is (whole chunks, no data loss).
    let mut format_retries: u32 = 0;

    // Hard safety ceiling on TOTAL iterations. Bounds an agent that neither
    // finishes, queries to its budget, nor trips the char budget. Derived from
    // the query budget — generous for interleaved query+add_chunks, capped.
    let max_iterations = (query_budget as usize).saturating_mul(4).max(12);

    for iteration in 0..max_iterations {
        // Enforce byte cap — but always allow at least the first turn so the
        // agent can select from the initial chunks even when the prompt is large.
        if iteration > 0 && estimate_history_bytes(&messages) > AGENTIC_HISTORY_BYTE_CAP {
            tracing::info!(iteration, "agentic rerank: history byte cap reached, stopping");
            exit = LoopExit::ByteCap;
            break;
        }

        // Force tool use unless the model is allowed to respond [DONE].
        // After add_chunks with at least one query done, the model may stop.
        // After query results, model MUST call add_chunks (always forced).
        // After add_chunks with zero queries done, model MUST call query (forced).
        let force_tool_use = !awaiting_query || query_calls == 0;
        let result = backend.next_turn(&system, &messages, &tools, force_tool_use).await;

        match result {
            Err(e) => {
                warn!(iteration, error = %e, "agentic rerank: LLM call failed");
                raw_response_log.push_str(&format!("[Iter {iteration}] ERROR: {e}\n"));
                exit = LoopExit::LlmError;
                break;
            }
            Ok(ToolTurnResult::Text(text)) => {
                raw_response_log.push_str(&format!("[Iter {iteration}] TEXT: {text}\n"));
                exit = LoopExit::AgentDone;
                break;
            }
            Ok(ToolTurnResult::ToolCalls(calls)) => {
                raw_response_log.push_str(&format!("[Iter {iteration}] TOOL_CALLS: {}\n",
                    calls.iter().map(|c| format!("{}({})", c.name, c.args)).collect::<Vec<_>>().join(", ")));

                // Record model's tool calls in history
                messages.push(ChatMessage::ModelToolCalls(calls.clone()));

                let mut tool_results: Vec<ToolResult> = Vec::new();
                // Mid-turn hard stop: set when the char budget is crossed BETWEEN
                // add_chunks calls within a single response, so one response with
                // many add_chunks cannot overshoot the budget unbounded.
                let mut stop_mid_turn: Option<LoopExit> = None;

                for call in &calls {
                    match call.name.as_str() {
                        "add_chunks" => {
                            if awaiting_query {
                                tool_results.push(ToolResult {
                                    name: "add_chunks".to_owned(),
                                    id: call.id.clone(),
                                    content: "Error: add_chunks already called this phase. \
                                        Call query to search for more context, or stop."
                                        .to_owned(),
                                });
                                continue;
                            }
                            // Reject a malformed `lines` shape (flat integer list
                            // instead of nested [[start,end],...] pairs) and nudge
                            // the model to re-emit — but only up to
                            // MAX_FORMAT_RETRIES, after which we accept as-is so a
                            // model that can't produce pairs never blocks the loop.
                            // No commit, no cadence change: the model retries.
                            let bad = malformed_line_chunks(&call.args);
                            if !bad.is_empty() && format_retries < MAX_FORMAT_RETRIES {
                                format_retries += 1;
                                let bad_list = bad
                                    .iter()
                                    .map(|i| i.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                tool_results.push(ToolResult {
                                    name: "add_chunks".to_owned(),
                                    id: call.id.clone(),
                                    content: format!(
                                        "Error: the `lines` field for chunk(s) [{bad_list}] is a \
                                         FLAT list of line numbers. It MUST be an array of \
                                         [start, end] pairs. Example: \"lines\": [[7, 11], [20, 28]] \
                                         (NOT [7, 8, 9, 20, 21]). Re-send this add_chunks call with \
                                         every `lines` as nested [start, end] pairs."
                                    ),
                                });
                                continue;
                            }
                            let (result_content, added_chars) = handle_add_chunks(
                                &call.args,
                                &extended_chunks,
                                &extended_numbered,
                                min_prune_lines,
                                &mut accumulated,
                                accumulated_chars,
                                char_budget,
                            );
                            accumulated_chars = accumulated_chars.saturating_add(added_chars);
                            format_retries = 0;
                            awaiting_query = true;
                            // The model got its turn to harvest — clear the flag
                            // regardless of which set it actually pulled from.
                            unharvested_query = false;
                            tool_results.push(ToolResult {
                                name: "add_chunks".to_owned(),
                                id: call.id.clone(),
                                content: result_content,
                            });
                            // Hard stop the moment the budget is crossed — do NOT
                            // process the rest of this response's calls.
                            if char_budget > 0 && accumulated_chars >= char_budget {
                                stop_mid_turn = Some(LoopExit::ChunkCharBudget);
                                break;
                            }
                            // If query budget is already spent, no point continuing —
                            // the model can't add_chunks again (awaiting_query) and
                            // can't query (budget exhausted). End cleanly.
                            if query_calls >= query_budget {
                                stop_mid_turn = Some(LoopExit::QueryBudget);
                                break;
                            }
                        }
                        "query" => {
                            awaiting_query = false;
                            // Reject if query budget exhausted — agent should
                            // add_chunks from existing results or stop.
                            if query_calls >= query_budget {
                                tool_results.push(ToolResult {
                                    name: "query".to_owned(),
                                    id: call.id.clone(),
                                    content: "Error: query budget exhausted. \
                                        Use add_chunks to select from existing results, or stop."
                                        .to_owned(),
                                });
                                continue;
                            }
                            let info_req = call.args.get("information_request")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            let result_content = run_query_tool(
                                backend,
                                info_req,
                                &mut query_cache,
                                &mut extended_chunks,
                                &mut extended_numbered,
                            ).await;
                            // Only count against budget if the query actually ran
                            // (not rejected as duplicate/empty).
                            if !result_content.starts_with("Error:") {
                                query_calls += 1;
                                // Fetched fresh results the model hasn't harvested
                                // yet — guard against a resource cap discarding them.
                                unharvested_query = true;
                            }
                            // When query budget is now exhausted, append a nudge
                            // so the model calls add_chunks next instead of stopping.
                            let content = result_content;
                            tool_results.push(ToolResult {
                                name: "query".to_owned(),
                                id: call.id.clone(),
                                content,
                            });
                        }
                        "grep" | "read" if backend.repo_root().is_none() => {
                            // grep_read disabled or no root → tool not offered.
                            // Defensive: refuse like any unknown tool.
                            warn!(tool = %call.name, "agentic rerank: fs tool called without repo root");
                            tool_results.push(ToolResult {
                                name: call.name.clone(),
                                id: call.id.clone(),
                                content: format!("Error: tool '{}' is not available.", call.name),
                            });
                        }
                        "grep" => {
                            // Exact search is an EXPLORATION tool: it satisfies the
                            // add→explore cadence (reset awaiting_query) but does NOT
                            // consume the query turn budget. Bounded only by the
                            // shared char budget + iteration cap.
                            awaiting_query = false;
                            let root = backend.repo_root().expect("checked above");
                            let content = run_grep_tool(
                                &root, &call.args, &mut extended_chunks, &mut extended_numbered,
                            );
                            tool_results.push(ToolResult {
                                name: "grep".to_owned(),
                                id: call.id.clone(),
                                content,
                            });
                        }
                        "read" => {
                            awaiting_query = false;
                            let root = backend.repo_root().expect("checked above");
                            let content = run_read_tool(
                                &root, &call.args, &mut extended_chunks, &mut extended_numbered,
                            );
                            tool_results.push(ToolResult {
                                name: "read".to_owned(),
                                id: call.id.clone(),
                                content,
                            });
                        }
                        other => {
                            warn!(tool = %other, "agentic rerank: unknown tool call, skipping");
                            tool_results.push(ToolResult {
                                name: other.to_owned(),
                                id: call.id.clone(),
                                content: format!("Error: unknown tool '{other}'"),
                            });
                        }
                    }
                }

                messages.push(ChatMessage::ToolResults(tool_results));

                // Inject a user message telling the model which tool to call next.
                // After add_chunks → nudge toward query (or stop if budget spent).
                // After query → nudge toward add_chunks.
                if stop_mid_turn.is_none() {
                    let nudge = if awaiting_query {
                        if query_calls >= query_budget {
                            // No more queries allowed — stop cleanly.
                            None
                        } else if query_calls == 0 {
                            // No queries yet — force the model to query.
                            Some(
                                "<system-reminder>\n\
                                 You MUST call `query` now. The initial chunks alone do NOT cover the full blast radius. \
                                 Search for related code: callers, callees, type definitions, implementations, tests, or sibling modules relevant to the query. \
                                 Use a specific information_request string describing what you need.\n\
                                 </system-reminder>".to_owned()
                            )
                        } else {
                            // Already queried at least once — allow [DONE].
                            Some(
                                "<system-reminder>\n\
                                 Call `query` to search for more related context (use a NEW information_request string different from previous queries). \
                                 If you are confident the results fully cover the query topic, respond with ONLY the text \"[DONE]\".\n\
                                 </system-reminder>".to_owned()
                            )
                        }
                    } else {
                        Some(
                            format!(
                                "<system-reminder>\n\
                                 You MUST call `add_chunks` now. ONLY add chunks that are DIRECTLY relevant to the ORIGINAL query: {query_json}\n\
                                 Do NOT add all query results — most will be tangential. Be highly selective. \
                                 Use `lines` to prune chunks to relevant ranges. `keep: \"all\"` only for short chunks where every line matters.\n\
                                 </system-reminder>"
                            )
                        )
                    };
                    match nudge {
                        Some(msg) => messages.push(ChatMessage::User(msg)),
                        None => {
                            exit = LoopExit::QueryBudget;
                            break;
                        }
                    }
                }

                // Post-response budget decision: only the char-budget hard cap
                // stops here. Query budget is enforced earlier (mid-turn after
                // add_chunks, or as a rejection on the next query call).
                if let Some(reason) = boundary_exit(stop_mid_turn) {
                    tracing::info!(
                        ?reason, accumulated_chars, char_budget, query_calls, query_budget,
                        "agentic rerank: budget reached, stopping"
                    );
                    exit = reason;
                    break;
                }
            }
        }
    }

    // Finalization harvest: a resource cap (byte/iteration) can cut the loop the
    // turn AFTER a `query` ran but BEFORE the model harvested its results — the
    // run would then end on a wasted `query` and discard fetched context. When
    // that happens, grant exactly ONE more forced turn so the model can call
    // add_chunks on the pending results. Scoped to ByteCap/IterationCap only:
    // those mean a HEALTHY backend hit a resource ceiling mid-exploration.
    // LlmError (broken backend) and AgentDone/budget exits are NOT retried.
    // This is a single bounded extra turn — memory stays bounded.
    if unharvested_query && matches!(exit, LoopExit::ByteCap | LoopExit::IterationCap) {
        messages.push(ChatMessage::User(
            "Search budget is spent and no more queries are allowed. \
             Call add_chunks NOW with every relevant chunk from the results so far. \
             This is your final action — do not call query.".to_owned(),
        ));
        match backend.next_turn(&system, &messages, &tools, /*force_tool_use*/ true).await {
            Ok(ToolTurnResult::ToolCalls(calls)) => {
                raw_response_log.push_str(&format!("[Final harvest] TOOL_CALLS: {}\n",
                    calls.iter().map(|c| format!("{}({})", c.name, c.args)).collect::<Vec<_>>().join(", ")));
                for call in &calls {
                    // Only add_chunks is honored here; query is explicitly refused
                    // so the harvest turn cannot spend more budget or recurse.
                    if call.name == "add_chunks" && !awaiting_query {
                        let (_summary, added_chars) = handle_add_chunks(
                            &call.args,
                            &extended_chunks,
                            &extended_numbered,
                            min_prune_lines,
                            &mut accumulated,
                            accumulated_chars,
                            char_budget,
                        );
                        accumulated_chars = accumulated_chars.saturating_add(added_chars);
                        awaiting_query = true;
                    }
                }
            }
            // Text, error, or transport failure on the harvest turn: nothing to
            // add — fall through with whatever was already accumulated.
            other => {
                if let Ok(ToolTurnResult::Text(t)) = &other {
                    raw_response_log.push_str(&format!("[Final harvest] TEXT: {t}\n"));
                } else if let Err(e) = &other {
                    raw_response_log.push_str(&format!("[Final harvest] ERROR: {e}\n"));
                }
            }
        }
    }

    let elapsed_ms = start.elapsed().as_millis() as u64;
    let raw_request_log = format!("[System]\n{system}\n\n[User]\n{user_prompt}");

    // Build final output from the failure-semantics table (pure decision).
    // The pool the agent could address: base candidates + any `query` results.
    // `reranked_indices` (including the Fallback `all_indices`, since the first
    // `n` entries ARE the base chunks) resolve against this uniformly.
    let pool = ExtendedPool { chunks: extended_chunks, numbered: extended_numbered };

    match decide_final_action(exit, accumulated.len()) {
        FinalAction::EmptyRelevant => {
            // Agent judged nothing relevant — empty result, not fallback.
            return (
                RerankOutput {
                    reranked_indices: vec![],
                    line_selections: vec![],
                    raw_request: raw_request_log,
                    raw_response: raw_response_log,
                    elapsed_ms,
                    fallback_used: false,
                    skip_reason: None,
                },
                pool,
            );
        }
        FinalAction::Fallback => {
            let reason = match exit {
                LoopExit::LlmError => "agentic LLM request failed with no chunks selected",
                LoopExit::ByteCap => "agentic loop hit history byte cap with no chunks selected",
                LoopExit::IterationCap => "agentic loop hit its iteration ceiling without selecting chunks",
                LoopExit::QueryBudget => "agentic loop used its query budget without selecting chunks",
                _ => "agentic loop ended without selecting chunks",
            };
            return (
                RerankOutput {
                    reranked_indices: all_indices,
                    line_selections: vec![None; n],
                    raw_request: raw_request_log,
                    raw_response: raw_response_log,
                    elapsed_ms,
                    fallback_used: true,
                    skip_reason: Some(reason.to_owned()),
                },
                pool,
            );
        }
        FinalAction::UseAccumulated => { /* fall through to build below */ }
    }

    // Deduplicate: keep first occurrence of each chunk_index. Bound-check against
    // the EXTENDED pool length so chunks pulled in by the `query` tool (index
    // >= n) survive — bounding by `n` here would silently drop them.
    let pool_len = pool.chunks.len();
    let mut seen = std::collections::HashSet::new();
    let mut deduped: Vec<ChunkSelection> = Vec::new();
    for (idx, lines) in accumulated {
        if idx < pool_len && seen.insert(idx) {
            deduped.push((idx, lines));
        }
    }

    let reranked_indices: Vec<usize> = deduped.iter().map(|(idx, _)| *idx).collect();
    let line_selections: Vec<Option<Vec<(u32, u32)>>> = deduped.into_iter().map(|(_, lines)| lines).collect();

    (
        RerankOutput {
            reranked_indices,
            line_selections,
            raw_request: raw_request_log,
            raw_response: raw_response_log,
            elapsed_ms,
            fallback_used: false,
            skip_reason: None,
        },
        pool,
    )
}

/// Handle one `add_chunks` call: add every chunk in the `chunks` array to the
/// accumulator, in order. Returns `(summary, added_chars)` where `added_chars`
/// is the total emitted character count of the chunks committed by THIS call —
/// the same content engine.rs step-7 will emit, so the loop's char budget
/// measures real output size. `numbered` is aligned 1:1 with `chunks`.
fn handle_add_chunks(
    args: &serde_json::Value,
    chunks: &[MergeChunk],
    numbered: &[Option<String>],
    min_prune_lines: u32,
    accumulated: &mut Vec<ChunkSelection>,
    accumulated_chars: usize,
    char_budget: usize,
) -> (String, usize) {
    let Some(arr) = args.get("chunks").and_then(|v| v.as_array()) else {
        return ("Error: `chunks` is required and must be an array of {chunk_index, lines?} objects".to_owned(), 0);
    };
    if arr.is_empty() {
        return ("Error: `chunks` array is empty — include at least one {chunk_index} object".to_owned(), 0);
    }

    let mut lines_out: Vec<String> = Vec::with_capacity(arr.len());
    let mut added_chars = 0usize;
    for item in arr {
        let (line, chars) = add_one_chunk(item, chunks, numbered, min_prune_lines, accumulated);
        lines_out.push(line);
        added_chars = added_chars.saturating_add(chars);
    }

    let total_chars = accumulated_chars.saturating_add(added_chars);
    let remaining = if char_budget > 0 { char_budget.saturating_sub(total_chars) } else { 0 };
    lines_out.push(format!(
        "--- {}/{} chunks selected, {total_chars}/{char_budget} chars used, {remaining} remaining.",
        accumulated.len(), chunks.len()
    ));
    (lines_out.join("\n"), added_chars)
}

/// Scan an `add_chunks` call's args for chunks whose `lines` is in the wrong
/// (flattened) shape. Returns the list of offending `chunk_index` values (empty
/// when every `lines` field is well-formed or absent). Used by the loop to
/// reject-and-nudge BEFORE committing, so the model re-emits nested pairs.
fn malformed_line_chunks(args: &serde_json::Value) -> Vec<u64> {
    let Some(arr) = args.get("chunks").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let mut bad = Vec::new();
    for item in arr {
        if let Some(lines) = item.get("lines").and_then(|v| v.as_array())
            && lines_shape_is_malformed(lines)
        {
            if let Some(idx) = item.get("chunk_index").and_then(|v| v.as_u64()) {
                bad.push(idx);
            } else {
                bad.push(u64::MAX); // index missing too; still flag the entry
            }
        }
    }
    bad
}

/// Validate and accumulate a single `{chunk_index, lines?}` entry. Returns
/// `(status_line, emitted_chars)`. `emitted_chars` is 0 for an error entry, and
/// otherwise the character count of the content engine.rs will emit for this
/// chunk (sliced ranges, or whole numbered text, or stored content fallback).
fn add_one_chunk(
    item: &serde_json::Value,
    chunks: &[MergeChunk],
    numbered: &[Option<String>],
    min_prune_lines: u32,
    accumulated: &mut Vec<ChunkSelection>,
) -> (String, usize) {
    let idx = match item.get("chunk_index").and_then(|v| v.as_u64()) {
        Some(i) => i as usize,
        None => return ("Error: chunk_index is required and must be an integer".to_owned(), 0),
    };

    if idx >= chunks.len() {
        return (format!("Error: chunk_index {idx} out of range (0..{})", chunks.len()), 0);
    }

    if accumulated.iter().any(|(i, _)| *i == idx) {
        return (format!("Skipped: chunk {idx} already added"), 0);
    }

    let chunk = &chunks[idx];
    let raw_lines = item.get("lines").and_then(|v| v.as_array());

    let selection = match raw_lines {
        None => None,
        Some(arr) if arr.is_empty() => None,
        Some(arr) => {
            if chunk.line_end.saturating_sub(chunk.line_start) < min_prune_lines {
                None
            } else {
                sanitize_ranges(arr, chunk.line_start, chunk.line_end)
            }
        }
    };

    let emitted = emitted_chars(chunk, numbered.get(idx).and_then(|n| n.as_deref()), &selection);
    accumulated.push((idx, selection));
    (
        format!("OK: added chunk {idx} ({}:{}-{})", chunk.file, chunk.line_start, chunk.line_end),
        emitted,
    )
}

/// Character count of the content engine.rs step-7 will emit for one selected
/// chunk. Mirrors that formatting exactly: per-range slices when a line
/// selection is present and the numbered text is readable; the whole numbered
/// text otherwise; the stored content as the final fallback.
fn emitted_chars(chunk: &MergeChunk, numbered_text: Option<&str>, selection: &Option<Vec<(u32, u32)>>) -> usize {
    match (numbered_text, selection) {
        (Some(text), Some(ranges)) if !ranges.is_empty() => ranges
            .iter()
            .map(|&(s, e)| slice_numbered(text, chunk.line_start, s, e).len())
            .sum(),
        (Some(text), _) => text.len(),
        (None, _) => chunk.content.len(),
    }
}

/// Handle one `query` tool call: cache lookup → `backend.sub_query` → extend
/// the addressable chunk pool → format for the agent. Generic over the backend
/// so tests drive it with mock retrieval.
async fn run_query_tool<B: AgenticBackend>(
    backend: &B,
    information_request: &str,
    query_cache: &mut HashMap<String, Vec<MergeChunk>>,
    extended_chunks: &mut Vec<MergeChunk>,
    extended_numbered: &mut Vec<Option<String>>,
) -> String {
    if information_request.is_empty() {
        return "Error: information_request is required".to_owned();
    }

    // Reject duplicate queries — force the model to vary its search terms.
    if query_cache.contains_key(information_request) {
        return "Error: this exact query was already used. Try a different information_request.".to_owned();
    }

    match backend.sub_query(information_request).await {
        Err(e) => format!("Error: query failed: {e}"),
        Ok(results) => {
            if results.is_empty() {
                return "No results found for this query.".to_owned();
            }

            let start_idx = extended_chunks.len();
            // Add results to extended pool so add_chunk can reference them
            for chunk in &results {
                let numbered_text = read_lines_from_fs(&chunk.file, chunk.line_start, chunk.line_end).ok();
                extended_numbered.push(numbered_text);
                extended_chunks.push(chunk.clone());
            }

            let response = format_sub_query_results(&results, start_idx);
            query_cache.insert(information_request.to_owned(), results);
            response
        }
    }
}

fn format_sub_query_results(results: &[MergeChunk], start_idx: usize) -> String {
    let mut output = format!("Found {} results (chunk indices {}-{}):\n\n",
        results.len(), start_idx, start_idx + results.len() - 1);

    for (i, chunk) in results.iter().enumerate() {
        let idx = start_idx + i;
        let symbol_display = chunk.symbol.as_deref().unwrap_or("no symbol");
        let content = truncate_content(&chunk.content, 60);
        output.push_str(&format!(
            "[{idx}] score={:.2} | {}:{}-{} ({symbol_display})\n{content}\n---\n",
            chunk.score, chunk.file, chunk.line_start, chunk.line_end
        ));
    }
    output
}

/// Build a pool chunk from a repo-relative file + 1-based line range discovered
/// by `grep`/`read`. The chunk's `file` is the ABSOLUTE path (the pool's
/// convention — `read_lines_from_fs` and the final formatter read from it), and
/// its content/numbered text is read fresh from disk so `add_chunks` can prune
/// it like any other chunk. Score 0.0 / no symbol metadata — these are exact-FS
/// chunks, not embedding hits. Returns `None` if the lines can't be read.
fn synth_pool_chunk(
    root: &std::path::Path,
    rel_path: &str,
    line_start: u32,
    line_end: u32,
    extended_chunks: &mut Vec<MergeChunk>,
    extended_numbered: &mut Vec<Option<String>>,
) -> Option<usize> {
    let abs = root.join(rel_path);
    let abs_str = abs.to_string_lossy().replace('\\', "/");
    let numbered_text = read_lines_from_fs(&abs_str, line_start, line_end).ok()?;
    let idx = extended_chunks.len();
    extended_chunks.push(MergeChunk {
        file: abs_str,
        line_start,
        line_end,
        score: 0.0,
        content: numbered_text.clone(),
        symbol: None,
        symbol_fqn: None,
        symbol_kind: None,
    });
    extended_numbered.push(Some(numbered_text));
    Some(idx)
}

/// `grep` tool: exact text/regex search over the working tree. Each match-region
/// becomes an addressable pool chunk (Design B); returns the raw grep output
/// followed by the `[idx] path:start-end` chunk index map so the agent can
/// `add_chunks` them. Does not consume the query budget.
fn run_grep_tool(
    root: &std::path::Path,
    args: &serde_json::Value,
    extended_chunks: &mut Vec<MergeChunk>,
    extended_numbered: &mut Vec<Option<String>>,
) -> String {
    let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
    if pattern.trim().is_empty() {
        return "Error: pattern is required.".to_owned();
    }
    let path = args.get("path").and_then(|v| v.as_str());
    let literal = args.get("literal").and_then(|v| v.as_bool()).unwrap_or(false);
    let ignore_case = args.get("ignore_case").and_then(|v| v.as_bool()).unwrap_or(false);
    let context_lines = args
        .get("context_lines")
        .and_then(|v| v.as_u64())
        .map(|n| (n as usize).min(crate::fs_tools::GREP_MAX_CONTEXT))
        .unwrap_or(0);

    let outcome = crate::fs_tools::run_grep(root, pattern, path, literal, ignore_case, context_lines);
    if !outcome.ok || outcome.regions.is_empty() {
        // Error or no matches — return the message as-is, nothing addressable.
        return outcome.text;
    }

    let mut out = outcome.text;
    out.push_str("\nAddressable chunks from this grep (pass chunk_index to add_chunks):\n");
    for region in &outcome.regions {
        match synth_pool_chunk(
            root, &region.rel_path, region.line_start, region.line_end,
            extended_chunks, extended_numbered,
        ) {
            Some(idx) => out.push_str(&format!(
                "[{idx}] {}:{}-{}\n", region.rel_path, region.line_start, region.line_end
            )),
            None => continue,
        }
    }
    out
}

/// `read` tool: verbatim numbered lines of one file. The emitted range becomes a
/// single addressable pool chunk (Design B). Does not consume the query budget.
fn run_read_tool(
    root: &std::path::Path,
    args: &serde_json::Value,
    extended_chunks: &mut Vec<MergeChunk>,
    extended_numbered: &mut Vec<Option<String>>,
) -> String {
    let file_path = args.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
    if file_path.trim().is_empty() {
        return "Error: file_path is required.".to_owned();
    }
    let start_line = args.get("start_line").and_then(|v| v.as_u64()).map(|n| n as usize);
    let end_line = args.get("end_line").and_then(|v| v.as_u64()).map(|n| n as usize);

    let outcome = crate::fs_tools::run_read(root, file_path, start_line, end_line);
    let Some((rel, s, e)) = outcome.range.clone() else {
        // Error or empty file — return the message, nothing addressable.
        return outcome.text;
    };

    let mut out = outcome.text;
    if let Some(idx) = synth_pool_chunk(root, &rel, s, e, extended_chunks, extended_numbered) {
        out.push_str(&format!(
            "\nAddressable chunk from this read (pass chunk_index to add_chunks):\n[{idx}] {rel}:{s}-{e}\n"
        ));
    }
    out
}

fn parse_rerank_response(
    response: &str,
    chunks: &[MergeChunk],
    min_prune_lines: u32,
    elapsed_ms: u64,
    structured: bool,
) -> RerankOutput {
    let n = chunks.len();
    let all_indices: Vec<usize> = (0..n).collect();

    // Unwrap the JSON array of ranking entries from the response.
    //
    // Structured (native JSON) mode: the whole response is a JSON object
    // `{"ranked_indices": [ ... ]}`. Parse it and pull the array out.
    // XML mode: the array lives between <ranked_indices> tags (with a
    // markdown-fence fallback), then is parsed as a bare JSON array.
    //
    // Either way, a missing/non-array/unparseable payload converges on the SAME
    // fallback below: original order with `fallback_used: true`.
    let parsed: Vec<serde_json::Value> = if structured {
        match serde_json::from_str::<serde_json::Value>(response.trim()) {
            Ok(serde_json::Value::Object(map)) => match map.get("ranked_indices") {
                Some(serde_json::Value::Array(arr)) => arr.clone(),
                _ => {
                    warn!(raw = %response, "structured rerank response missing `ranked_indices` array");
                    return RerankOutput {
                        reranked_indices: all_indices,
                        line_selections: vec![None; n],
                        raw_request: String::new(),
                        raw_response: response.to_owned(),
                        elapsed_ms,
                        fallback_used: true,
                        skip_reason: Some("structured response missing ranked_indices".to_owned()),
                    };
                }
            },
            _ => {
                warn!(raw = %response, "failed to parse structured rerank response as a JSON object");
                return RerankOutput {
                    reranked_indices: all_indices,
                    line_selections: vec![None; n],
                    raw_request: String::new(),
                    raw_response: response.to_owned(),
                    elapsed_ms,
                    fallback_used: true,
                    skip_reason: Some("failed to parse structured response".to_owned()),
                };
            }
        }
    } else {
        // Try XML tags first
        let re = Regex::new(r"(?s)<ranked_indices>\s*(.*?)\s*</ranked_indices>").unwrap();
        let text = if let Some(caps) = re.captures(response) {
            caps.get(1).unwrap().as_str().to_owned()
        } else {
            // Fallback: try raw response, strip markdown fences
            let trimmed = response.trim();
            if trimmed.starts_with("```") {
                trimmed.lines()
                    .filter(|line| !line.starts_with("```"))
                    .collect::<Vec<_>>()
                    .join("\n")
                    .trim()
                    .to_owned()
            } else {
                trimmed.to_owned()
            }
        };

        match serde_json::from_str(&text) {
            Ok(arr) => arr,
            Err(_) => {
                warn!(raw = %response, "failed to parse rerank response as JSON array");
                return RerankOutput {
                    reranked_indices: all_indices,
                    line_selections: vec![None; n],
                    raw_request: String::new(),
                    raw_response: response.to_owned(),
                    elapsed_ms,
                    fallback_used: true,
                    skip_reason: Some("failed to parse LLM response".to_owned()),
                };
            }
        }
    };

    let mut reranked_indices: Vec<usize> = Vec::new();
    let mut line_selections: Vec<Option<Vec<(u32, u32)>>> = Vec::new();

    for entry in &parsed {
        // Element is a bare integer index (back-compat safety net), or an object:
        //   {"chunk_index": idx, "lines": [[s,e],...]}  → narrow to ranges
        //   {"chunk_index": idx, "keep": "full"}        → whole chunk (no `lines` field)
        //   "i" accepted as legacy alias for "chunk_index"
        let (idx, raw_lines) = if let Some(i) = entry.as_u64() {
            (i as usize, None)
        } else if let Some(obj) = entry.as_object() {
            let Some(i) = obj.get("chunk_index").or_else(|| obj.get("i")).and_then(|v| v.as_u64()) else { continue };
            (i as usize, obj.get("lines").and_then(|v| v.as_array()))
        } else {
            continue;
        };

        // Drop indices outside the candidate set.
        if idx >= n {
            continue;
        }
        let chunk = &chunks[idx];

        let selection = match raw_lines {
            // Whole chunk: bare int, {"keep":"full"} (no `lines`), or empty `lines`.
            None => None,
            Some(arr) if arr.is_empty() => None,
            Some(arr) => {
                // Small chunks are never line-pruned (1C policy).
                if chunk.line_end.saturating_sub(chunk.line_start) < min_prune_lines {
                    None
                } else {
                    sanitize_ranges(arr, chunk.line_start, chunk.line_end)
                }
            }
        };

        reranked_indices.push(idx);
        line_selections.push(selection);
    }

    if reranked_indices.is_empty() {
        // LLM legitimately judged nothing relevant — honor that (empty result),
        // matching prior behavior. Not a fallback.
        return RerankOutput {
            reranked_indices: vec![],
            line_selections: vec![],
            raw_request: String::new(),
            raw_response: response.to_owned(),
            elapsed_ms,
            fallback_used: false,
            skip_reason: None,
        };
    }

    RerankOutput {
        reranked_indices,
        line_selections,
        raw_request: String::new(),
        raw_response: response.to_owned(),
        elapsed_ms,
        fallback_used: false,
        skip_reason: None,
    }
}

/// Detect a `lines` array the model emitted in the WRONG (flattened) shape: a
/// flat list of bare integers like `[7, 8, 10, 11]` instead of the required
/// nested pairs `[[7, 8], [10, 11]]`. Returns true iff the array is non-empty
/// and any element is NOT a 2-element array — i.e. the call should be rejected
/// and the model nudged to re-emit nested pairs (rather than the parser guessing
/// the ambiguous meaning). An empty/absent `lines` is fine (means "whole chunk")
/// and is never flagged here.
fn lines_shape_is_malformed(arr: &[serde_json::Value]) -> bool {
    if arr.is_empty() {
        return false;
    }
    arr.iter().any(|v| !matches!(v.as_array(), Some(p) if p.len() == 2))
}

/// Validate and normalize the LLM's `lines` array for one chunk:
/// - parse each `[start, end]` pair (skip malformed / start>end),
/// - pad by RANGE_PAD and clamp to [chunk_start, chunk_end],
/// - sort by start and merge overlapping/adjacent ranges (gap <= 1).
///
/// STRICT: only nested `[start, end]` pairs are accepted — that is the schema,
/// the prompt, and the format the model is nudged toward at runtime (see
/// [`lines_shape_is_malformed`] and the reject-and-retry in the dispatch loop).
/// A flattened integer list (`[7, 8, 10, 11]`) is intentionally NOT reinterpreted
/// here: its meaning (ranges vs individual lines) is genuinely ambiguous, so the
/// loop rejects it and asks the model to re-emit pairs instead of this parser
/// guessing. Anything non-pair is skipped; if nothing valid survives we return
/// `None` and the caller keeps the whole chunk (safe fallback, never data loss).
///
/// Returns `None` if no valid range survives.
fn sanitize_ranges(
    arr: &[serde_json::Value],
    chunk_start: u32,
    chunk_end: u32,
) -> Option<Vec<(u32, u32)>> {
    let mut ranges: Vec<(u32, u32)> = Vec::new();
    for pair in arr {
        let Some(p) = pair.as_array() else { continue };
        if p.len() != 2 {
            continue;
        }
        let (Some(s), Some(e)) = (p[0].as_u64(), p[1].as_u64()) else { continue };
        let (s, e) = (s as u32, e as u32);
        if s > e {
            continue;
        }
        // Pad then clamp to chunk bounds.
        let s = s.saturating_sub(RANGE_PAD).max(chunk_start);
        let e = e.saturating_add(RANGE_PAD).min(chunk_end);
        if s > e {
            continue;
        }
        ranges.push((s, e));
    }

    if ranges.is_empty() {
        return None;
    }

    // Sort by start, then merge overlapping/adjacent ranges (gap <= 1).
    ranges.sort_unstable_by_key(|&(s, _)| s);
    let mut merged: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
    for (s, e) in ranges {
        if let Some(last) = merged.last_mut()
            && s <= last.1 + 1
        {
            last.1 = last.1.max(e);
            continue;
        }
        merged.push((s, e));
    }
    Some(merged)
}

fn truncate_content(content: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= max_lines {
        return content.to_owned();
    }
    let half = max_lines / 2;
    let truncated_count = lines.len() - max_lines;
    let mut result = lines[..half].join("\n");
    result.push_str(&format!("\n... ({truncated_count} lines truncated) ...\n"));
    result.push_str(&lines[lines.len() - half..].join("\n"));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chunk(line_start: u32, line_end: u32) -> MergeChunk {
        MergeChunk {
            file: "/x.rs".to_owned(),
            line_start,
            line_end,
            score: 1.0,
            content: "stored".to_owned(),
            symbol: None,
            symbol_fqn: None,
            symbol_kind: None,
        }
    }

    // ── sanitize_ranges ──────────────────────────────────────────────────

    #[test]
    fn sanitize_merges_overlapping_and_adjacent() {
        // Chunk 100..200. Ranges [110,120] and [121,130] are gap==1 → merge.
        // With pad ±2: [108,122] and [119,132] overlap → one [108,132].
        let arr = vec![json!([110, 120]), json!([121, 130])];
        let out = sanitize_ranges(&arr, 100, 200).expect("some");
        assert_eq!(out, vec![(108, 132)]);
    }

    #[test]
    fn sanitize_drops_start_gt_end() {
        let arr = vec![json!([150, 140])];
        assert_eq!(sanitize_ranges(&arr, 100, 200), None);
    }

    #[test]
    fn sanitize_range_fully_outside_chunk_is_none() {
        // Chunk 100..120, range 300..310 → after pad+clamp s=max(298,100)=100... but
        // e=min(312,120)=120, s=298 padded -> 296.. wait: clamp pins s to chunk_start.
        // To truly land outside: range below the chunk, e < chunk_start.
        let arr = vec![json!([10, 20])]; // entirely below chunk_start=100
        assert_eq!(sanitize_ranges(&arr, 100, 200), None);
    }

    #[test]
    fn sanitize_pads_and_clamps_to_bounds() {
        // Range [101,199] padded ±2 → [99,201], clamped to [100,200].
        let arr = vec![json!([101, 199])];
        let out = sanitize_ranges(&arr, 100, 200).expect("some");
        assert_eq!(out, vec![(100, 200)]);
    }

    #[test]
    fn sanitize_all_malformed_is_none() {
        // Strict nested parsing: [1,2,3] wrong arity; ["a","b"] non-numeric; bare
        // 5 not a pair; [9,4] start>end. Nothing survives → None.
        let arr = vec![json!([1, 2, 3]), json!(["a", "b"]), json!(5), json!([9, 4])];
        assert_eq!(sanitize_ranges(&arr, 100, 200), None);
    }

    #[test]
    fn sanitize_strict_rejects_flat_list_to_none() {
        // A flattened integer list is NOT reinterpreted by the parser (that's the
        // dispatch loop's reject-and-nudge job). With no valid nested pair, the
        // parser returns None → caller keeps the whole chunk (safe, no guessing).
        let arr = vec![json!(7), json!(8), json!(10), json!(11)];
        assert_eq!(sanitize_ranges(&arr, 1, 100), None);
        // The wide-pair ambiguous case likewise yields None, never slivers.
        assert_eq!(sanitize_ranges(&[json!(15), json!(52)], 1, 100), None);
    }

    #[test]
    fn sanitize_accepts_nested_pairs() {
        // The one true form: nested [start,end] pairs parse exactly (pad+merge).
        let arr = vec![json!([7, 11]), json!([20, 28])];
        let out = sanitize_ranges(&arr, 1, 100).expect("nested pairs must parse");
        // [7,11]→[5,13]; [20,28]→[18,30]. Disjoint (gap 13→18 > 1) → two ranges.
        assert_eq!(out, vec![(5, 13), (18, 30)]);
        // A wide nested pair stays one contiguous range.
        assert_eq!(sanitize_ranges(&[json!([15, 52])], 1, 100), Some(vec![(13, 54)]));
    }

    // ── lines_shape_is_malformed / malformed_line_chunks (reject-and-nudge) ──

    #[test]
    fn malformed_detects_flat_list_but_not_nested() {
        // Flat integer lists are malformed; nested pairs and empty/absent are not.
        assert!(lines_shape_is_malformed(&[json!(7), json!(8), json!(10)]));
        assert!(lines_shape_is_malformed(&[json!(15), json!(52)]));
        // A single stray bare int among pairs is still malformed (one bad elem).
        assert!(lines_shape_is_malformed(&[json!([7, 11]), json!(20)]));
        // Well-formed nested pairs: not malformed.
        assert!(!lines_shape_is_malformed(&[json!([7, 11]), json!([20, 28])]));
        // Empty array → "whole chunk", never flagged.
        assert!(!lines_shape_is_malformed(&[]));
    }

    #[test]
    fn malformed_line_chunks_reports_offending_indices() {
        // chunk 2 has a flat list (bad); chunk 5 has nested pairs (ok); chunk 9
        // has keep:"all" / no lines (ok). Only [2] is reported.
        let args = json!({ "chunks": [
            { "chunk_index": 2, "lines": [7, 8, 9, 20, 21] },
            { "chunk_index": 5, "lines": [[7, 11], [20, 28]] },
            { "chunk_index": 9, "keep": "all" },
        ]});
        assert_eq!(malformed_line_chunks(&args), vec![2]);

        // All well-formed → empty.
        let ok = json!({ "chunks": [{ "chunk_index": 1, "lines": [[1, 5]] }] });
        assert!(malformed_line_chunks(&ok).is_empty());
    }

    // ── parse_rerank_response ────────────────────────────────────────────

    fn parse(resp: &str, chunks: &[MergeChunk]) -> RerankOutput {
        // 16 = default rerank_min_prune_lines. XML mode (structured = false).
        parse_rerank_response(resp, chunks, 16, 0, false)
    }

    fn parse_structured(resp: &str, chunks: &[MergeChunk]) -> RerankOutput {
        // 16 = default rerank_min_prune_lines. JSON object-root mode.
        parse_rerank_response(resp, chunks, 16, 0, true)
    }

    #[test]
    fn parse_bare_int_is_whole_chunk() {
        let chunks = vec![chunk(1, 10), chunk(1, 10)];
        let out = parse("<ranked_indices>[1, 0]</ranked_indices>", &chunks);
        assert_eq!(out.reranked_indices, vec![1, 0]);
        assert_eq!(out.line_selections, vec![None, None]);
        assert!(!out.fallback_used);
    }

    #[test]
    fn parse_object_with_lines_yields_ranges() {
        // Chunk 100..200 is large (>30 lines) so pruning applies.
        let chunks = vec![chunk(100, 200)];
        let out = parse(
            "<ranked_indices>[{\"chunk_index\":0,\"lines\":[[110,120]]}]</ranked_indices>",
            &chunks,
        );
        assert_eq!(out.reranked_indices, vec![0]);
        // [110,120] padded ±2 → [108,122], within bounds.
        assert_eq!(out.line_selections, vec![Some(vec![(108, 122)])]);
    }

    #[test]
    fn parse_object_without_chunk_index_is_skipped() {
        let chunks = vec![chunk(1, 10)];
        let out = parse(
            "<ranked_indices>[{\"lines\":[[1,2]]}, 0]</ranked_indices>",
            &chunks,
        );
        assert_eq!(out.reranked_indices, vec![0]);
        assert_eq!(out.line_selections, vec![None]);
    }

    #[test]
    fn parse_out_of_range_index_dropped() {
        let chunks = vec![chunk(1, 10)];
        let out = parse("<ranked_indices>[5, 0]</ranked_indices>", &chunks);
        assert_eq!(out.reranked_indices, vec![0]);
    }

    #[test]
    fn parse_legacy_i_alias_still_accepted() {
        let chunks = vec![chunk(100, 200)];
        let out = parse(
            "<ranked_indices>[{\"i\":0,\"lines\":[[110,120]]}]</ranked_indices>",
            &chunks,
        );
        assert_eq!(out.reranked_indices, vec![0]);
        assert_eq!(out.line_selections, vec![Some(vec![(108, 122)])]);
    }

    #[test]
    fn parse_keep_full_is_whole_chunk() {
        // Canonical "keep whole chunk" form: {"chunk_index":idx,"keep":"full"} — no `lines` field.
        let chunks = vec![chunk(100, 200)];
        let out = parse(
            "<ranked_indices>[{\"chunk_index\":0,\"keep\":\"full\"}]</ranked_indices>",
            &chunks,
        );
        assert_eq!(out.reranked_indices, vec![0]);
        assert_eq!(out.line_selections, vec![None]);
    }

    #[test]
    fn parse_empty_lines_is_whole_chunk() {
        // Safety net: a stray empty `lines` array still degrades to whole chunk.
        let chunks = vec![chunk(100, 200)];
        let out = parse(
            "<ranked_indices>[{\"chunk_index\":0,\"lines\":[]}]</ranked_indices>",
            &chunks,
        );
        assert_eq!(out.reranked_indices, vec![0]);
        assert_eq!(out.line_selections, vec![None]);
    }

    #[test]
    fn parse_small_chunk_never_pruned() {
        // Chunk span 9 < min_prune_lines (16) → selection forced to None.
        let chunks = vec![chunk(1, 10)];
        let out = parse(
            "<ranked_indices>[{\"chunk_index\":0,\"lines\":[[3,5]]}]</ranked_indices>",
            &chunks,
        );
        assert_eq!(out.line_selections, vec![None]);
    }

    #[test]
    fn parse_broken_json_falls_back_to_all_indices() {
        let chunks = vec![chunk(1, 10), chunk(1, 10), chunk(1, 10)];
        let out = parse("<ranked_indices>not json</ranked_indices>", &chunks);
        assert!(out.fallback_used);
        assert_eq!(out.reranked_indices, vec![0, 1, 2]);
        assert_eq!(out.line_selections, vec![None, None, None]);
    }

    // ── parse_rerank_response (structured object-root mode) ──────────────

    #[test]
    fn parse_structured_object_root_ranks_chunks() {
        // {"ranked_indices":[...]} object root, reusing the same element forms.
        let chunks = vec![chunk(1, 10), chunk(100, 200)];
        let out = parse_structured(
            "{\"ranked_indices\":[{\"chunk_index\":1,\"lines\":[[110,120]]},{\"chunk_index\":0,\"keep\":\"full\"}]}",
            &chunks,
        );
        assert!(!out.fallback_used);
        assert_eq!(out.reranked_indices, vec![1, 0]);
        // Chunk 1 (100..200) is large → [110,120] padded ±2 → [108,122].
        // Chunk 0 keep:full → None.
        assert_eq!(out.line_selections, vec![Some(vec![(108, 122)]), None]);
    }

    #[test]
    fn parse_structured_missing_key_falls_back_to_all_indices() {
        // Valid JSON object, but no `ranked_indices` key → original-order fallback.
        let chunks = vec![chunk(1, 10), chunk(1, 10)];
        let out = parse_structured("{\"something_else\":[0,1]}", &chunks);
        assert!(out.fallback_used);
        assert_eq!(out.reranked_indices, vec![0, 1]);
        assert_eq!(out.line_selections, vec![None, None]);
    }

    #[test]
    fn parse_structured_key_not_array_falls_back() {
        // `ranked_indices` present but not an array → original-order fallback.
        let chunks = vec![chunk(1, 10), chunk(1, 10)];
        let out = parse_structured("{\"ranked_indices\":\"oops\"}", &chunks);
        assert!(out.fallback_used);
        assert_eq!(out.reranked_indices, vec![0, 1]);
    }

    #[test]
    fn parse_structured_non_object_root_falls_back() {
        // A bare array (not the expected object root) in structured mode → fallback.
        let chunks = vec![chunk(1, 10), chunk(1, 10)];
        let out = parse_structured("[0, 1]", &chunks);
        assert!(out.fallback_used);
        assert_eq!(out.reranked_indices, vec![0, 1]);
    }

    #[test]
    fn parse_structured_unknown_entry_keys_skipped() {
        // Element-level tolerance: an object without `chunk_index` or `i` is skipped,
        // the valid one is kept. (Same loop as XML mode.)
        let chunks = vec![chunk(1, 10), chunk(1, 10)];
        let out = parse_structured(
            "{\"ranked_indices\":[{\"lines\":[[1,2]]},{\"chunk_index\":1,\"keep\":\"full\"}]}",
            &chunks,
        );
        assert!(!out.fallback_used);
        assert_eq!(out.reranked_indices, vec![1]);
        assert_eq!(out.line_selections, vec![None]);
    }

    // ── Agentic rerank helper tests ─────────────────────────────────────────

    #[test]
    fn add_chunks_valid_full() {
        let chunks = vec![chunk(100, 200), chunk(1, 10)];
        let numbered = vec![None, None];
        let mut accumulated = Vec::new();
        let (result, chars) = handle_add_chunks(
            &json!({"chunks": [{"chunk_index": 0}]}),
            &chunks,
            &numbered,
            16,
            &mut accumulated,
            0,
            50_000,
        );
        assert!(result.starts_with("OK:"));
        assert_eq!(accumulated.len(), 1);
        assert_eq!(accumulated[0], (0, None));
        // No numbered text → emitted chars = stored content len ("stored" = 6).
        assert_eq!(chars, "stored".len());
    }

    #[test]
    fn add_chunks_multiple_at_once() {
        // The core new behavior: one call adds many chunks, in order.
        let chunks = vec![chunk(100, 200), chunk(1, 10), chunk(300, 400)];
        let numbered = vec![None, None, None];
        let mut accumulated = Vec::new();
        let (result, _chars) = handle_add_chunks(
            &json!({"chunks": [
                {"chunk_index": 2},
                {"chunk_index": 0, "lines": [[110, 120]]},
                {"chunk_index": 1}
            ]}),
            &chunks,
            &numbered,
            16,
            &mut accumulated,
            0,
            50_000,
        );
        // Three OK lines, one per chunk.
        assert_eq!(result.lines().filter(|l| l.starts_with("OK:")).count(), 3);
        assert_eq!(accumulated.len(), 3);
        assert_eq!(accumulated[0], (2, None));
        assert_eq!(accumulated[1], (0, Some(vec![(108, 122)]))); // big chunk → ranges kept
        assert_eq!(accumulated[2], (1, None));                    // small chunk → whole
    }

    #[test]
    fn add_chunks_chars_count_uses_sliced_ranges() {
        // With numbered text present and a line selection, emitted chars must be
        // the SLICED length (what engine.rs emits), not the whole chunk.
        let chunks = vec![chunk(100, 200)]; // span 100 ≥ min_prune_lines
        // 5 numbered lines of the chunk; selection [110,120] pads→[108,122] then
        // slices within this text. Build a realistic numbered block.
        let numbered_text: String = (100..=200)
            .map(|ln| format!("{ln}: code line {ln}"))
            .collect::<Vec<_>>()
            .join("\n");
        let numbered = vec![Some(numbered_text.clone())];
        let mut accumulated = Vec::new();
        let (_result, chars) = handle_add_chunks(
            &json!({"chunks": [{"chunk_index": 0, "lines": [[110, 120]]}]}),
            &chunks,
            &numbered,
            16,
            &mut accumulated,
            0,
            50_000,
        );
        // Expected = len of the sliced range [108,122] (pad ±2), computed via the
        // same slice_numbered the engine uses — so the budget measures real output.
        let expected = slice_numbered(&numbered_text, 100, 108, 122).len();
        assert_eq!(chars, expected);
        assert!(chars < numbered_text.len(), "sliced is smaller than whole chunk");
    }

    #[test]
    fn add_chunks_with_lines() {
        let chunks = vec![chunk(100, 200)];
        let numbered = vec![None];
        let mut accumulated = Vec::new();
        let (result, _chars) = handle_add_chunks(
            &json!({"chunks": [{"chunk_index": 0, "lines": [[110, 120]]}]}),
            &chunks,
            &numbered,
            16,
            &mut accumulated,
            0,
            50_000,
        );
        assert!(result.starts_with("OK:"));
        assert_eq!(accumulated.len(), 1);
        assert_eq!(accumulated[0].0, 0);
        assert_eq!(accumulated[0].1, Some(vec![(108, 122)]));
    }

    #[test]
    fn add_chunks_small_chunk_ignores_lines() {
        let chunks = vec![chunk(1, 10)]; // span 9 < min_prune_lines 16
        let numbered = vec![None];
        let mut accumulated = Vec::new();
        handle_add_chunks(
            &json!({"chunks": [{"chunk_index": 0, "lines": [[3, 5]]}]}),
            &chunks,
            &numbered,
            16,
            &mut accumulated,
            0,
            50_000,
        );
        assert_eq!(accumulated[0].1, None);
    }

    #[test]
    fn add_chunks_out_of_range_entry_errors_but_others_kept() {
        // A bad index in the array is reported but does not abort the whole call.
        let chunks = vec![chunk(1, 10)];
        let numbered = vec![None];
        let mut accumulated = Vec::new();
        let (result, _chars) = handle_add_chunks(
            &json!({"chunks": [{"chunk_index": 5}, {"chunk_index": 0}]}),
            &chunks,
            &numbered,
            16,
            &mut accumulated,
            0,
            50_000,
        );
        assert!(result.contains("Error:"), "bad index reported");
        assert!(result.contains("OK:"), "valid index still added");
        assert_eq!(accumulated, vec![(0, None)]);
    }

    #[test]
    fn add_chunks_missing_chunks_array_errors() {
        let chunks = vec![chunk(1, 10)];
        let numbered = vec![None];
        let mut accumulated = Vec::new();
        let (result, chars) = handle_add_chunks(
            &json!({"chunk_index": 0}), // wrong shape: no `chunks` array
            &chunks,
            &numbered,
            16,
            &mut accumulated,
            0,
            50_000,
        );
        assert!(result.starts_with("Error:"));
        assert!(accumulated.is_empty());
        assert_eq!(chars, 0, "error path adds no chars");
    }

    #[test]
    fn add_chunks_empty_array_errors() {
        let chunks = vec![chunk(1, 10)];
        let numbered = vec![None];
        let mut accumulated = Vec::new();
        let (result, _chars) = handle_add_chunks(
            &json!({"chunks": []}),
            &chunks,
            &numbered,
            16,
            &mut accumulated,
            0,
            50_000,
        );
        assert!(result.starts_with("Error:"));
        assert!(accumulated.is_empty());
    }

    #[test]
    fn add_chunks_missing_index_in_entry() {
        let chunks = vec![chunk(1, 10)];
        let numbered = vec![None];
        let mut accumulated = Vec::new();
        let (result, _chars) = handle_add_chunks(
            &json!({"chunks": [{"lines": [[1, 2]]}]}),
            &chunks,
            &numbered,
            16,
            &mut accumulated,
            0,
            50_000,
        );
        assert!(result.contains("Error:"));
        assert!(accumulated.is_empty());
    }

    #[test]
    fn estimate_history_bytes_basic() {
        let messages = vec![
            ChatMessage::User("hello world".to_owned()),
            ChatMessage::ToolResults(vec![ToolResult {
                name: "add_chunk".to_owned(),
                id: None,
                content: "OK: added chunk 0".to_owned(),
            }]),
        ];
        let bytes = estimate_history_bytes(&messages);
        assert!(bytes > 20);
        assert!(bytes < 200);
    }

    #[test]
    fn agentic_tool_definitions_are_valid() {
        // Without grep/read: the original two tools.
        let tools = agentic_tool_definitions(false);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "add_chunks");
        assert_eq!(tools[1].name, "query");
        assert!(tools[0].parameters.is_object());
        assert!(tools[1].parameters.is_object());

        // With grep/read: four tools, grep + read appended.
        let tools = agentic_tool_definitions(true);
        assert_eq!(tools.len(), 4);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["add_chunks", "query", "grep", "read"]);
        assert!(tools.iter().all(|t| t.parameters.is_object()));
    }

    // ── Failure-semantics matrix (decide_final_action) ──────────────────────
    // Every row of the committed table is asserted here. This is the contract
    // that distinguishes empty-relevant from fallback — the silent-break risk.

    #[test]
    fn matrix_agent_done_zero_is_empty_relevant_not_fallback() {
        // Agent finished with text and selected nothing → honor as empty result.
        assert_eq!(decide_final_action(LoopExit::AgentDone, 0), FinalAction::EmptyRelevant);
    }

    #[test]
    fn matrix_agent_done_nonzero_uses_accumulated() {
        assert_eq!(decide_final_action(LoopExit::AgentDone, 3), FinalAction::UseAccumulated);
    }

    #[test]
    fn matrix_llm_error_zero_is_fallback() {
        // HTTP error with nothing committed → fallback to original order.
        assert_eq!(decide_final_action(LoopExit::LlmError, 0), FinalAction::Fallback);
    }

    #[test]
    fn matrix_llm_error_nonzero_keeps_accumulated() {
        // HTTP error AFTER the agent committed chunks → keep them (break-and-keep).
        assert_eq!(decide_final_action(LoopExit::LlmError, 2), FinalAction::UseAccumulated);
    }

    #[test]
    fn matrix_query_budget_nonzero_uses_accumulated() {
        // Spent the query budget but had committed chunks → keep them.
        assert_eq!(decide_final_action(LoopExit::QueryBudget, 5), FinalAction::UseAccumulated);
    }

    #[test]
    fn matrix_query_budget_zero_is_fallback() {
        // Used all queries but never added a chunk → fallback to original order.
        assert_eq!(decide_final_action(LoopExit::QueryBudget, 0), FinalAction::Fallback);
    }

    #[test]
    fn matrix_chunk_char_budget_nonzero_uses_accumulated() {
        // Hit the char budget → by construction chunks were committed → keep them.
        assert_eq!(decide_final_action(LoopExit::ChunkCharBudget, 4), FinalAction::UseAccumulated);
    }

    #[test]
    fn matrix_chunk_char_budget_zero_is_fallback() {
        // Unreachable in the live loop (chars only grow when add_chunks commits),
        // but the table is total — fallback is the safe choice.
        assert_eq!(decide_final_action(LoopExit::ChunkCharBudget, 0), FinalAction::Fallback);
    }

    #[test]
    fn matrix_iteration_cap_zero_is_fallback() {
        assert_eq!(decide_final_action(LoopExit::IterationCap, 0), FinalAction::Fallback);
    }

    #[test]
    fn matrix_iteration_cap_nonzero_uses_accumulated() {
        assert_eq!(decide_final_action(LoopExit::IterationCap, 2), FinalAction::UseAccumulated);
    }

    #[test]
    fn matrix_byte_cap_zero_is_fallback() {
        assert_eq!(decide_final_action(LoopExit::ByteCap, 0), FinalAction::Fallback);
    }

    #[test]
    fn matrix_byte_cap_nonzero_uses_accumulated() {
        assert_eq!(decide_final_action(LoopExit::ByteCap, 1), FinalAction::UseAccumulated);
    }

    #[test]
    fn matrix_empty_relevant_is_distinct_from_fallback() {
        // The critical distinction: AgentDone/0 and IterationCap/0 both have zero
        // accumulated chunks but MUST produce different actions, or the
        // empty-vs-fallback contract collapses silently.
        let agent_done = decide_final_action(LoopExit::AgentDone, 0);
        let iter_cap = decide_final_action(LoopExit::IterationCap, 0);
        assert_ne!(agent_done, iter_cap);
        assert_eq!(agent_done, FinalAction::EmptyRelevant);
        assert_eq!(iter_cap, FinalAction::Fallback);
    }

    // ── End-to-end agentic loop via a scripted mock backend ─────────────────
    // These drive the REAL `run_agentic_loop` (same code path as production),
    // proving each failure-matrix row maps to the correct observable
    // RerankOutput. No parallel copy of the field-assembly logic exists.

    use std::cell::RefCell;
    use crate::llm::ToolCall;

    /// One scripted LLM turn outcome the mock will return, in order.
    enum MockTurn {
        /// Emit tool calls (e.g. add_chunk / query).
        Calls(Vec<ToolCall>),
        /// Emit a text response (agent signals done).
        Text(String),
        /// Simulate an HTTP/transport error on this turn.
        Err(String),
    }

    /// Scripted backend: pops one `MockTurn` per `next_turn` call. `sub_query`
    /// returns a fixed canned chunk set so the `query` tool path is exercised.
    struct MockBackend {
        turns: RefCell<std::collections::VecDeque<MockTurn>>,
        sub_query_chunks: Vec<MergeChunk>,
        repo_root: Option<std::path::PathBuf>,
    }

    impl MockBackend {
        fn new(turns: Vec<MockTurn>) -> Self {
            Self { turns: RefCell::new(turns.into()), sub_query_chunks: vec![], repo_root: None }
        }
        fn with_sub_query(turns: Vec<MockTurn>, sub: Vec<MergeChunk>) -> Self {
            Self { turns: RefCell::new(turns.into()), sub_query_chunks: sub, repo_root: None }
        }
        /// Mock that exposes a real on-disk repo root so grep/read run for real.
        fn with_root(turns: Vec<MockTurn>, root: std::path::PathBuf) -> Self {
            Self { turns: RefCell::new(turns.into()), sub_query_chunks: vec![], repo_root: Some(root) }
        }
    }

    impl AgenticBackend for MockBackend {
        fn next_turn(
            &self,
            _system: &str,
            _messages: &[ChatMessage],
            _tools: &[ToolDef],
            _force_tool_use: bool,
        ) -> impl std::future::Future<Output = anyhow::Result<ToolTurnResult>> + Send {
            // Pop synchronously (no guard held across the returned future's await).
            let next = self.turns.borrow_mut().pop_front();
            async move {
                match next {
                    Some(MockTurn::Calls(c)) => Ok(ToolTurnResult::ToolCalls(c)),
                    Some(MockTurn::Text(t)) => Ok(ToolTurnResult::Text(t)),
                    Some(MockTurn::Err(e)) => Err(anyhow::anyhow!(e)),
                    // Script exhausted unexpectedly → behave like an error.
                    None => Err(anyhow::anyhow!("mock: no more scripted turns")),
                }
            }
        }

        fn sub_query(
            &self,
            _information_request: &str,
        ) -> impl std::future::Future<Output = anyhow::Result<Vec<MergeChunk>>> + Send {
            let chunks = self.sub_query_chunks.clone();
            async move { Ok(chunks) }
        }

        fn repo_root(&self) -> Option<std::path::PathBuf> {
            self.repo_root.clone()
        }
    }

    /// An `add_chunks` call carrying a single chunk (one turn, one chunk).
    fn add_chunk_call(idx: usize) -> ToolCall {
        ToolCall {
            name: "add_chunks".to_owned(),
            id: Some(format!("c{idx}")),
            args: json!({ "chunks": [{ "chunk_index": idx }] }),
            ..Default::default()
        }
    }

    /// An `add_chunks` call carrying a single chunk with line ranges.
    fn add_chunk_call_lines(idx: usize, ranges: serde_json::Value) -> ToolCall {
        ToolCall {
            name: "add_chunks".to_owned(),
            id: Some(format!("c{idx}")),
            args: json!({ "chunks": [{ "chunk_index": idx, "lines": ranges }] }),
            ..Default::default()
        }
    }

    /// An `add_chunks` call carrying MANY chunks at once (one turn, many chunks).
    fn add_chunks_call(indices: &[usize]) -> ToolCall {
        let chunks: Vec<_> = indices.iter().map(|i| json!({ "chunk_index": i })).collect();
        ToolCall {
            name: "add_chunks".to_owned(),
            id: Some("batch".to_owned()),
            args: json!({ "chunks": chunks }),
            ..Default::default()
        }
    }

    fn run_loop(backend: &MockBackend, n: usize, max_turns: u32) -> (RerankOutput, ExtendedPool) {
        // char budget disabled (0) for query-budget / general loop tests.
        run_loop_full(backend, n, max_turns, 0)
    }

    /// Like `run_loop` but with an explicit char budget (for char-budget tests).
    fn run_loop_full(backend: &MockBackend, n: usize, max_turns: u32, max_chunk_chars: u32) -> (RerankOutput, ExtendedPool) {
        // Base chunks get DISTINCT line ranges (100+i .. 110+i) so a test can
        // tell a base chunk from a sub-query chunk by its coordinates — this is
        // what proves base indices resolve to base chunks, not shifted ones.
        let chunks: Vec<MergeChunk> = (0..n).map(|i| chunk(100 + i as u32, 110 + i as u32)).collect();
        let numbered = vec![None; n];
        let caller_stats = vec![None; n];
        // Drive the REAL loop on the current-thread runtime.
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(run_agentic_loop(backend, "q", &chunks, &numbered, &caller_stats, 16, max_turns, max_chunk_chars, false))
    }

    /// Mirror of engine.rs step-7 index resolution: for each reranked index,
    /// resolve against the pool (when present) exactly as production does —
    /// `res_chunks.get(idx)` with `else continue`. Returns the (line_start,
    /// line_end) of each successfully-resolved chunk, in output order. If a base
    /// index failed to resolve (the silent-drop bug), it would be MISSING here.
    fn resolve_like_engine(out: &RerankOutput, pool: &ExtendedPool) -> Vec<(u32, u32)> {
        let mut resolved = Vec::new();
        for &idx in &out.reranked_indices {
            let Some(chunk) = pool.chunks.get(idx) else { continue };
            resolved.push((chunk.line_start, chunk.line_end));
        }
        resolved
    }

    #[test]
    fn loop_agent_done_zero_chunks_is_empty_not_fallback() {
        // Turn 0: agent immediately returns text, selecting nothing.
        let backend = MockBackend::new(vec![MockTurn::Text("nothing relevant".into())]);
        let (out, _pool) = run_loop(&backend, 4, 3);
        assert!(out.reranked_indices.is_empty(), "empty result set");
        assert!(!out.fallback_used, "must NOT be flagged as fallback");
        assert!(out.skip_reason.is_none());
    }

    #[test]
    fn loop_iteration_cap_zero_chunks_is_fallback_all_indices() {
        // Every iteration emits an unknown tool (never add_chunks, never finishes,
        // never queries) → hits the hard iteration ceiling with zero accumulated.
        // Script enough no-ops to exceed max_iterations = max(budget*4, 12).
        let noop = || MockTurn::Calls(vec![ToolCall {
            name: "noop".to_owned(), id: Some("x".into()), args: json!({}),
            ..Default::default()
        }]);
        let backend = MockBackend::new((0..20).map(|_| noop()).collect());
        let (out, _pool) = run_loop(&backend, 4, 2);
        assert_eq!(out.reranked_indices, vec![0, 1, 2, 3], "fallback = original order");
        assert!(out.fallback_used, "must be flagged as fallback");
        assert!(out.skip_reason.as_deref().unwrap().contains("iteration ceiling"));
    }

    #[test]
    fn loop_byte_cap_after_query_grants_final_harvest() {
        // Bug: query fetches results, then byte cap trips at the top of the
        // next iteration — before the model can harvest. Without finalization
        // the run ends on a wasted query (0 accumulated → fallback). With it,
        // the model gets one forced add_chunks turn and the chunk is committed.
        let big = MergeChunk {
            file: "/big.rs".to_owned(),
            line_start: 1,
            line_end: 1,
            score: 1.0,
            content: "x".repeat(250_000),
            symbol: None,
            symbol_fqn: None,
            symbol_kind: None,
        };
        let backend = MockBackend::with_sub_query(
            vec![
                // Turn 0: query loads the oversized chunk (base_n=2 → idx 2).
                MockTurn::Calls(vec![ToolCall {
                    name: "query".to_owned(), id: Some("q".into()),
                    args: json!({ "information_request": "x" }), ..Default::default()
                }]),
                // Turn 1 never starts: byte cap trips → ByteCap exit.
                // Final harvest turn: model commits the sub-query chunk.
                MockTurn::Calls(vec![add_chunk_call(2)]),
            ],
            vec![big],
        );
        let (out, _pool) = run_loop(&backend, 2, 9);
        assert_eq!(out.reranked_indices, vec![2], "sub-query chunk harvested in final turn");
        assert!(!out.fallback_used, "harvested → UseAccumulated, not fallback");
    }

    #[test]
    fn loop_llm_error_with_accumulated_keeps_chunks() {
        // Turn 0: add_chunks([2, 0]) in one call. Turn 1: HTTP error.
        // Expect [2, 0] kept, NOT fallback.
        let backend = MockBackend::new(vec![
            MockTurn::Calls(vec![add_chunks_call(&[2, 0])]),
            MockTurn::Err("503 upstream".into()),
        ]);
        let (out, _pool) = run_loop(&backend, 4, 5);
        assert_eq!(out.reranked_indices, vec![2, 0], "accumulated order preserved");
        assert!(!out.fallback_used, "accumulated chunks are not a fallback");
    }

    #[test]
    fn loop_llm_error_zero_chunks_is_fallback() {
        // Turn 0: immediate HTTP error before any add_chunk.
        let backend = MockBackend::new(vec![MockTurn::Err("connection reset".into())]);
        let (out, _pool) = run_loop(&backend, 3, 5);
        assert_eq!(out.reranked_indices, vec![0, 1, 2]);
        assert!(out.fallback_used);
        assert!(out.skip_reason.as_deref().unwrap().contains("failed"));
    }

    #[test]
    fn loop_agent_done_with_accumulated_uses_them() {
        // Turn 0: add_chunks([1, 0]) in one call. Turn 1: text done.
        let backend = MockBackend::new(vec![
            MockTurn::Calls(vec![add_chunks_call(&[1, 0])]),
            MockTurn::Text("done".into()),
        ]);
        let (out, _pool) = run_loop(&backend, 3, 5);
        assert_eq!(out.reranked_indices, vec![1, 0], "selection order preserved");
        assert!(!out.fallback_used);
        // chunk(1,10) span 9 < min_prune_lines 16 → both kept whole.
        assert_eq!(out.line_selections, vec![None, None]);
    }

    #[test]
    fn loop_dedups_repeated_add_chunk() {
        // Agent adds chunks [1, 0] including a duplicate of 1 in the same call.
        // Dedup keeps first occurrence of each index.
        let backend = MockBackend::new(vec![
            MockTurn::Calls(vec![ToolCall {
                name: "add_chunks".to_owned(),
                id: Some("c0".into()),
                args: json!({ "chunks": [
                    { "chunk_index": 1 },
                    { "chunk_index": 1 },
                    { "chunk_index": 0 }
                ]}),
                ..Default::default()
            }]),
            MockTurn::Text("done".into()),
        ]);
        let (out, _pool) = run_loop(&backend, 3, 5);
        assert_eq!(out.reranked_indices, vec![1, 0], "dedup keeps first occurrence");
        assert!(!out.fallback_used);
    }

    #[test]
    fn loop_query_tool_then_add_extended_chunk() {
        // Turn 0: agent calls query → mock returns 1 extra chunk (index n=2).
        // Turn 1: agent add_chunk(2) (the new extended chunk). Turn 2: done.
        // Proves the query path extends the addressable pool and add_chunk can
        // reference sub-query results.
        let sub = vec![chunk(50, 60)];
        let backend = MockBackend::with_sub_query(
            vec![
                MockTurn::Calls(vec![ToolCall {
                    name: "query".to_owned(),
                    id: Some("q0".into()),
                    args: json!({ "information_request": "more context" }),
                    ..Default::default()
                }]),
                MockTurn::Calls(vec![add_chunk_call(2)]), // index 2 = first sub-query chunk
                MockTurn::Text("done".into()),
            ],
            sub,
        );
        // n=2 base chunks; sub-query adds index 2.
        let (out, pool) = run_loop(&backend, 2, 5);
        assert_eq!(out.reranked_indices, vec![2], "added the extended sub-query chunk");
        assert!(!out.fallback_used);
        // The extended pool must carry the sub-query chunk at index 2, and the
        // engine resolves the reranked index against THIS pool (not base merged).
        assert_eq!(pool.chunks.len(), 3, "base 2 + 1 sub-query chunk");
        assert_eq!(pool.chunks[2].line_start, 50, "index 2 is the sub-query chunk");
        assert_eq!(pool.chunks[2].line_end, 60);
        assert!(pool.chunks.get(out.reranked_indices[0]).is_some(),
            "reranked index resolves within the extended pool");
    }

    #[test]
    fn loop_fallback_after_query_resolves_base_indices_not_subquery() {
        // The trap: a `query` runs (appending sub-query chunks to the pool),
        // THEN the loop falls back (LLM error, zero accumulated). Fallback emits
        // base indices 0..base_n — they MUST resolve to the base chunks, not be
        // shifted onto the appended sub-query chunks, and must NOT silently drop.
        let sub = vec![chunk(50, 60), chunk(70, 80)]; // distinct from base 100+ ranges
        let backend = MockBackend::with_sub_query(
            vec![
                MockTurn::Calls(vec![ToolCall {
                    name: "query".to_owned(),
                    id: Some("q0".into()),
                    args: json!({ "information_request": "expand" }),
                    ..Default::default()
                }]),
                MockTurn::Err("503 after query".into()), // fallback with 0 accumulated
            ],
            sub,
        );
        // 3 base chunks (ranges 100-110, 101-111, 102-112) + 2 sub-query appended.
        let (out, pool) = run_loop(&backend, 3, 5);

        assert!(out.fallback_used, "must be a fallback");
        assert_eq!(out.reranked_indices, vec![0, 1, 2], "fallback = base order 0..base_n");
        assert_eq!(pool.chunks.len(), 5, "pool = 3 base + 2 sub-query");

        // Resolve exactly as engine.rs does. All three base indices must resolve
        // (no silent drop) AND land on the BASE chunks (100-range), proving the
        // appended sub-query chunks did not shift base positions.
        let resolved = resolve_like_engine(&out, &pool);
        assert_eq!(
            resolved,
            vec![(100, 110), (101, 111), (102, 112)],
            "base indices resolve to base chunks, not the appended sub-query chunks"
        );
        assert_eq!(resolved.len(), 3, "no base index silently dropped via else-continue");
    }

    #[test]
    fn loop_empty_relevant_with_populated_pool_resolves_to_nothing() {
        // EmptyRelevant: agent ran a query (pool populated) then finished with
        // text and zero selections. reranked_indices is empty, so engine.rs
        // emits no blocks — confirm a Some(pool) with chunks doesn't choke and
        // produces an empty result (not the base chunks, not a panic).
        let sub = vec![chunk(50, 60)];
        let backend = MockBackend::with_sub_query(
            vec![
                MockTurn::Calls(vec![ToolCall {
                    name: "query".to_owned(),
                    id: Some("q0".into()),
                    args: json!({ "information_request": "expand" }),
                    ..Default::default()
                }]),
                MockTurn::Text("nothing relevant after all".into()),
            ],
            sub,
        );
        let (out, pool) = run_loop(&backend, 3, 5);

        assert!(!out.fallback_used, "EmptyRelevant is NOT a fallback");
        assert!(out.reranked_indices.is_empty(), "agent selected nothing");
        assert_eq!(pool.chunks.len(), 4, "pool still populated: 3 base + 1 sub-query");
        // Engine-style resolution over zero indices yields zero blocks — no panic
        // on the non-empty pool.
        assert!(resolve_like_engine(&out, &pool).is_empty());
    }

    // ── Turn-budget semantics (turns = query calls) ─────────────────────────

    #[test]
    fn budget_counts_query_calls_and_allows_final_add_chunks() {
        // max_turns = 2 (query budget). Two query turns spend the budget, but
        // the agent gets one more turn to add_chunks before the loop stops.
        let sub = vec![chunk(50, 60)];
        let backend = MockBackend::with_sub_query(
            vec![
                MockTurn::Calls(vec![ToolCall {
                    name: "query".to_owned(), id: Some("q1".into()),
                    args: json!({ "information_request": "first query" }),
                    ..Default::default()
                }]),                                            // query 1/2
                MockTurn::Calls(vec![ToolCall {
                    name: "query".to_owned(), id: Some("q2".into()),
                    args: json!({ "information_request": "second query" }),
                    ..Default::default()
                }]),                                            // query 2/2
                MockTurn::Calls(vec![add_chunk_call(0)]),       // agent gets to add_chunks
            ],
            sub,
        );
        let (out, _pool) = run_loop(&backend, 2, 2);
        assert_eq!(out.reranked_indices, vec![0], "agent could add_chunks after query budget exhausted");
        assert!(!out.fallback_used);
    }

    #[test]
    fn budget_query_then_add_keeps_chunks() {
        // max_turns = 1. Turn 0: one query (spends the budget) AND, in the SAME
        // model turn, an add_chunks. The boundary check keeps the committed chunk
        // and exits via QueryBudget with accumulated >= 1 → UseAccumulated.
        let sub = vec![chunk(50, 60)];
        let backend = MockBackend::with_sub_query(
            vec![
                MockTurn::Calls(vec![
                    ToolCall { name: "query".to_owned(), id: Some("q".into()),
                               args: json!({ "information_request": "x" }), ..Default::default() },
                    add_chunk_call(0),
                ]),
                MockTurn::Text("done".into()), // not reached (budget hit)
            ],
            sub,
        );
        let (out, _pool) = run_loop(&backend, 2, 1);
        assert_eq!(out.reranked_indices, vec![0], "committed chunk kept despite query budget");
        assert!(!out.fallback_used);
    }

    #[test]
    fn budget_add_chunks_calls_do_not_consume_query_budget() {
        // max_turns = 1 (query budget). The agent adds all chunks in one call
        // then finishes with text. Proves add_chunks does not decrement the budget.
        let backend = MockBackend::new(vec![
            MockTurn::Calls(vec![add_chunks_call(&[0, 1, 2])]),
            MockTurn::Text("done".into()),
        ]);
        let (out, _pool) = run_loop(&backend, 3, 1);
        assert_eq!(out.reranked_indices, vec![0, 1, 2], "all add_chunks committed, query budget untouched");
        assert!(!out.fallback_used);
    }

    #[test]
    fn budget_zero_max_turns_clamped_to_one() {
        // max_turns = 0 must clamp to 1 (query_budget.max(1)). One query is
        // allowed, then the budget stops the loop.
        let sub = vec![chunk(50, 60)];
        let backend = MockBackend::with_sub_query(
            vec![
                MockTurn::Calls(vec![
                    ToolCall { name: "query".to_owned(), id: Some("q".into()),
                               args: json!({ "information_request": "x" }), ..Default::default() },
                    add_chunk_call(0),
                ]),
                MockTurn::Calls(vec![add_chunk_call(1)]), // not reached
            ],
            sub,
        );
        let (out, _pool) = run_loop(&backend, 2, 0);
        assert_eq!(out.reranked_indices, vec![0], "clamped query budget of 1 allows one query turn");
        assert!(!out.fallback_used);
    }

    // ── Char-budget semantics (stop when added chunk content reaches cap) ────

    #[test]
    fn char_budget_stops_after_threshold() {
        // Base chunks have stored content "stored" (6 chars) and no numbered text,
        // so each added chunk emits 6 chars. Budget = 7: one add_chunks([0, 1])
        // adds both (6 + 6 = 12 >= 7) → loop stops. Chunk 2 is never reached.
        let backend = MockBackend::new(vec![
            MockTurn::Calls(vec![add_chunks_call(&[0, 1])]), // total 12 (>= 7) → STOP
            MockTurn::Calls(vec![add_chunk_call(2)]),        // not reached
        ]);
        // query budget high (10) so it cannot interfere; char budget = 7.
        let (out, _pool) = run_loop_full(&backend, 3, 10, 7);
        assert_eq!(out.reranked_indices, vec![0, 1], "stopped after char budget, both kept");
        assert!(!out.fallback_used, "char budget with chunks → UseAccumulated");
        assert!(out.skip_reason.is_none());
    }

    #[test]
    fn char_budget_zero_disables_limit() {
        // char budget = 0 → no char cap. Agent adds 3 chunks then finishes; all
        // kept, never tripped a char-budget stop.
        let backend = MockBackend::new(vec![
            MockTurn::Calls(vec![add_chunks_call(&[0, 1, 2])]),
            MockTurn::Text("done".into()),
        ]);
        let (out, _pool) = run_loop_full(&backend, 3, 10, 0);
        assert_eq!(out.reranked_indices, vec![0, 1, 2], "no char cap → all kept");
        assert!(!out.fallback_used);
    }

    #[test]
    fn char_budget_single_call_many_chunks_counts_all() {
        // ONE add_chunks call commits 3 chunks = 18 chars (3 × "stored"). With
        // budget = 10, the boundary check after that single turn sees 18 >= 10 and
        // stops — but all 3 chunks committed in that call are kept.
        let backend = MockBackend::new(vec![
            MockTurn::Calls(vec![add_chunks_call(&[0, 1, 2])]),
            MockTurn::Text("done".into()), // not reached
        ]);
        let (out, _pool) = run_loop_full(&backend, 3, 10, 10);
        assert_eq!(out.reranked_indices, vec![0, 1, 2], "all chunks from the over-budget call kept");
        assert!(!out.fallback_used);
    }

    #[test]
    fn consecutive_add_chunks_rejected_without_query() {
        // The cadence gate: a second add_chunks in a row (without an interleaved
        // query) is rejected. Only the first phase's chunks are committed.
        // Turn 0: add_chunks([0]) → OK, awaiting_query = true.
        // Turn 1: add_chunks([1]) → rejected (must query first).
        // Turn 2: add_chunks([2]) → rejected again.
        // Loop hits iteration cap with only chunk 0 committed.
        let backend = MockBackend::new(vec![
            MockTurn::Calls(vec![add_chunk_call(0)]),
            MockTurn::Calls(vec![add_chunk_call(1)]),
            MockTurn::Calls(vec![add_chunk_call(2)]),
            MockTurn::Text("done".into()),
        ]);
        let (out, _pool) = run_loop_full(&backend, 3, 10, 50_000);
        assert_eq!(out.reranked_indices, vec![0], "only first add_chunks committed");
        assert!(!out.fallback_used);
    }

    #[test]
    fn boundary_exit_only_checks_char_stop() {
        // boundary_exit now only gates on the char-budget hard cap.
        // Query budget is enforced in the dispatch loop, not here.
        assert_eq!(boundary_exit(Some(LoopExit::ChunkCharBudget)), Some(LoopExit::ChunkCharBudget));
        assert_eq!(boundary_exit(None), None);
    }

    #[test]
    fn char_budget_priority_end_to_end_keeps_chunk() {
        // End-to-end through the real loop: one response with query (exhausts
        // query_budget=1) THEN add_chunk (crosses char_budget=6). The committed
        // chunk must be kept (UseAccumulated), confirming the char stop did not
        // discard work and the loop terminated cleanly.
        let sub = vec![chunk(50, 60)];
        let backend = MockBackend::with_sub_query(
            vec![
                MockTurn::Calls(vec![
                    ToolCall { name: "query".to_owned(), id: Some("q".into()),
                               args: json!({ "information_request": "x" }), ..Default::default() },
                    add_chunk_call(0), // +6 == char_budget 6 → mid-turn ChunkCharBudget
                ]),
                MockTurn::Text("done".into()), // not reached
            ],
            sub,
        );
        let (out, _pool) = run_loop_full(&backend, 2, 1, 6);
        assert_eq!(out.reranked_indices, vec![0], "committed chunk kept");
        assert!(!out.fallback_used);
        assert!(out.skip_reason.is_none(), "UseAccumulated, not a fallback");
    }

    // ── grep/read tools (Design B: results become addressable chunks) ───────

    /// Drive the real loop with grep/read ENABLED and a real on-disk repo root,
    /// so grep/read run for real against `root`. Base chunks use the same
    /// distinct line coordinates as `run_loop_full` (100+i..110+i).
    fn run_loop_grep_read(
        backend: &MockBackend, n: usize, max_turns: u32, max_chunk_chars: u32,
    ) -> (RerankOutput, ExtendedPool) {
        let chunks: Vec<MergeChunk> = (0..n).map(|i| chunk(100 + i as u32, 110 + i as u32)).collect();
        let numbered = vec![None; n];
        let caller_stats = vec![None; n];
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(run_agentic_loop(
            backend, "q", &chunks, &numbered, &caller_stats, 16, max_turns, max_chunk_chars, true,
        ))
    }

    /// A `grep` tool call.
    fn grep_call(pattern: &str) -> ToolCall {
        ToolCall {
            name: "grep".to_owned(),
            id: Some("g".into()),
            args: json!({ "pattern": pattern }),
            ..Default::default()
        }
    }

    /// A `read` tool call for a file + range.
    fn read_call(file: &str, start: u32, end: u32) -> ToolCall {
        ToolCall {
            name: "read".to_owned(),
            id: Some("r".into()),
            args: json!({ "file_path": file, "start_line": start, "end_line": end }),
            ..Default::default()
        }
    }

    #[test]
    fn grep_results_are_addressable_and_committable() {
        // A real temp repo with a file the grep will hit. The agent: add nothing
        // initially is not allowed (first must be add_chunks) — so add base [0],
        // then grep, then add_chunks the grepped chunk by its reported index, done.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hit.rs"), "fn needle() {}\nother\n").unwrap();

        // Base pool has 1 chunk (index 0). grep appends index 1 (the match region).
        let backend = MockBackend::with_root(
            vec![
                MockTurn::Calls(vec![add_chunk_call(0)]),
                MockTurn::Calls(vec![grep_call("needle")]),
                MockTurn::Calls(vec![add_chunk_call(1)]), // commit the grepped chunk
                MockTurn::Text("[DONE]".into()),
            ],
            dir.path().to_path_buf(),
        );
        let (out, pool) = run_loop_grep_read(&backend, 1, 5, 0);
        // The grep chunk (index 1) was synthesized into the pool and committed.
        assert!(out.reranked_indices.contains(&1), "grepped chunk must be committable");
        assert!(pool.chunks.len() >= 2, "grep appended a pool chunk");
        assert!(!out.fallback_used);
        // The appended chunk points at the matched file.
        assert!(pool.chunks[1].file.ends_with("hit.rs"));
    }

    #[test]
    fn read_result_is_addressable_and_committable() {
        let dir = tempfile::tempdir().unwrap();
        let body: String = (1..=20).map(|i| format!("line {i}\n")).collect();
        std::fs::write(dir.path().join("f.rs"), body).unwrap();

        let backend = MockBackend::with_root(
            vec![
                MockTurn::Calls(vec![add_chunk_call(0)]),
                MockTurn::Calls(vec![read_call("f.rs", 5, 10)]),
                MockTurn::Calls(vec![add_chunk_call(1)]),
                MockTurn::Text("[DONE]".into()),
            ],
            dir.path().to_path_buf(),
        );
        let (out, pool) = run_loop_grep_read(&backend, 1, 5, 0);
        assert!(out.reranked_indices.contains(&1), "read chunk must be committable");
        assert_eq!(pool.chunks[1].line_start, 5);
        assert_eq!(pool.chunks[1].line_end, 10);
        assert!(pool.chunks[1].file.ends_with("f.rs"));
    }

    #[test]
    fn grep_does_not_consume_query_budget() {
        // query_budget = 1. The agent greps 3 times (exploration), never calling
        // `query`, then [DONE]. grep must NOT decrement the query budget, so the
        // loop runs to the agent's own [DONE] rather than a QueryBudget exit.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "alpha\nbeta\ngamma\n").unwrap();

        let backend = MockBackend::with_root(
            vec![
                MockTurn::Calls(vec![add_chunk_call(0)]),
                MockTurn::Calls(vec![grep_call("alpha")]),
                MockTurn::Calls(vec![grep_call("beta")]),
                MockTurn::Calls(vec![grep_call("gamma")]),
                MockTurn::Text("[DONE]".into()),
            ],
            dir.path().to_path_buf(),
        );
        // max_turns (query budget) = 1, but 3 greps run fine because they don't count.
        let (out, _pool) = run_loop_grep_read(&backend, 1, 1, 0);
        assert_eq!(out.reranked_indices, vec![0], "base chunk kept; greps explored freely");
        assert!(!out.fallback_used, "agent reached [DONE] cleanly, not a budget fallback");
    }

    #[test]
    fn grep_read_disabled_refuses_the_tool() {
        // No repo root (grep_read effectively off) → a grep call is refused like
        // an unknown tool, and the loop still completes on accumulated chunks.
        let backend = MockBackend::new(vec![
            MockTurn::Calls(vec![add_chunk_call(0)]),
            MockTurn::Calls(vec![grep_call("x")]), // refused (repo_root None)
            MockTurn::Text("[DONE]".into()),
        ]);
        // run_loop drives with grep_read=false AND repo_root None.
        let (out, _pool) = run_loop(&backend, 1, 5);
        assert_eq!(out.reranked_indices, vec![0]);
        assert!(!out.fallback_used);
    }

    // ── reject-and-nudge for malformed `lines` shape (Path A) ───────────────

    /// An add_chunks call with a FLAT (malformed) lines list for one chunk.
    fn add_chunk_flat_lines(idx: usize, flat: serde_json::Value) -> ToolCall {
        ToolCall {
            name: "add_chunks".to_owned(),
            id: Some(format!("c{idx}")),
            args: json!({ "chunks": [{ "chunk_index": idx, "lines": flat }] }),
            ..Default::default()
        }
    }

    #[test]
    fn malformed_lines_rejected_then_corrected_commits() {
        // Turn 0: add_chunks with a FLAT list → rejected, NOT committed, cadence
        // unchanged (still awaiting add_chunks). Turn 1: model re-emits nested
        // pairs → committed. Turn 2: query (cadence ok). Turn 3: [DONE].
        // Chunk 0 spans 100..200 (>min_prune_lines) so pruning actually applies.
        let backend = MockBackend::with_sub_query(
            vec![
                MockTurn::Calls(vec![add_chunk_flat_lines(0, json!([110, 111, 120, 121]))]),
                MockTurn::Calls(vec![add_chunk_call_lines(0, json!([[110, 121]]))]),
                MockTurn::Calls(vec![ToolCall {
                    name: "query".to_owned(), id: Some("q".into()),
                    args: json!({ "information_request": "more" }), ..Default::default()
                }]),
                MockTurn::Text("[DONE]".into()),
            ],
            vec![],
        );
        // One WIDE chunk (100..200, span 100 > min_prune_lines 16) so pruning
        // actually applies and the committed selection is observable.
        let chunks = vec![chunk(100, 200)];
        let numbered = vec![None];
        let caller_stats = vec![None];
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let (out, _pool) = rt.block_on(run_agentic_loop(
            &backend, "q", &chunks, &numbered, &caller_stats, 16, 5, 0, false,
        ));
        // The corrected call committed chunk 0 with a real PRUNED line selection
        // ([[110,121]] padded ±2 → [(108,123)]). If the flat call had been
        // accepted instead, the selection would be None (whole chunk) — so this
        // proves the flat call was rejected and only the nested retry committed.
        assert_eq!(out.reranked_indices, vec![0]);
        assert!(!out.fallback_used);
        assert_eq!(out.line_selections, vec![Some(vec![(108, 123)])]);
    }

    #[test]
    fn malformed_lines_accepted_after_retry_cap() {
        // Model NEVER produces nested pairs. The first MAX_FORMAT_RETRIES flat
        // calls are rejected; the next is accepted as-is (whole chunk kept, no
        // data loss). Provide enough flat turns to exhaust the cap + 1.
        let mut turns: Vec<MockTurn> = (0..(MAX_FORMAT_RETRIES + 1))
            .map(|_| MockTurn::Calls(vec![add_chunk_flat_lines(0, json!([110, 120]))]))
            .collect();
        turns.push(MockTurn::Text("[DONE]".into()));
        let backend = MockBackend::new(turns);
        let (out, _pool) = run_loop(&backend, 1, 5);
        // Accepted on the (cap+1)th try → chunk 0 committed (whole chunk: the flat
        // lines yield no valid nested range, so selection is None = keep all).
        assert_eq!(out.reranked_indices, vec![0]);
        assert!(!out.fallback_used, "accepted-as-is is a real selection, not a fallback");
    }
}
