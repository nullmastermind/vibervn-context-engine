//! Centralized, compile-time-embedded LLM prompt templates.
//!
//! Every natural-language string the engine sends to an LLM (system prompts,
//! user-prompt templates, tool descriptions, agent guidance) lives as a `.txt`
//! file in this directory and is embedded into the binary via `include_str!`.
//! This keeps the single-binary deployment model intact (no runtime file
//! loading) while gathering all prompt copy in one place for maintenance.
//!
//! Templates carry `{{token}}` placeholders filled at runtime by [`render`].
//! The double-brace delimiter is deliberate: several prompts embed literal
//! single-brace `{...}` JSON examples, and double braces never collide with
//! those.
//!
//! NOTE: MCP `#[tool(description = ...)]` attributes cannot route through a
//! `const` (the attribute needs the macro inline), so those sites use
//! `#[doc = include_str!("prompts/<name>.txt")]` directly in `mcp.rs` —
//! rmcp-macros preserves the `include_str!` expression as the tool description.

// ─── Chat agent (chat.rs) ────────────────────────────────────────────────
pub const CHAT_SYSTEM: &str = include_str!("chat_system.txt");
pub const CHAT_PROJECT_DOCS_APPENDIX: &str = include_str!("chat_project_docs_appendix.txt");
pub const CHAT_TOOL_CODEBASE: &str = include_str!("chat_tool_codebase.txt");
pub const CHAT_TOOL_FILE: &str = include_str!("chat_tool_file.txt");
pub const CHAT_TOOL_GREP: &str = include_str!("chat_tool_grep.txt");
pub const CHAT_TOOL_READ: &str = include_str!("chat_tool_read.txt");

// ─── One-shot reranker (query/reranker.rs::rerank) ───────────────────────
pub const RERANK_INTRO: &str = include_str!("rerank_intro.txt");
pub const RERANK_ELEMENT_SPEC: &str = include_str!("rerank_element_spec.txt");
pub const RERANK_SYSTEM_STRUCTURED: &str = include_str!("rerank_system_structured.txt");
pub const RERANK_SYSTEM_XML: &str = include_str!("rerank_system_xml.txt");
pub const RERANK_USER_STRUCTURED: &str = include_str!("rerank_user_structured.txt");
pub const RERANK_USER_XML: &str = include_str!("rerank_user_xml.txt");

// ─── Agentic reranker (query/reranker.rs::run_agentic_loop) ──────────────
pub const RERANK_AGENTIC_SYSTEM: &str = include_str!("rerank_agentic_system.txt");
pub const RERANK_AGENTIC_SYSTEM_EXACT_TOOLS: &str =
    include_str!("rerank_agentic_system_exact_tools.txt");
pub const RERANK_AGENTIC_USER: &str = include_str!("rerank_agentic_user.txt");
pub const RERANK_AGENTIC_TOOL_ADD_CHUNKS: &str = include_str!("rerank_agentic_tool_add_chunks.txt");
pub const RERANK_AGENTIC_TOOL_QUERY: &str = include_str!("rerank_agentic_tool_query.txt");
pub const RERANK_AGENTIC_TOOL_GREP: &str = include_str!("rerank_agentic_tool_grep.txt");
pub const RERANK_AGENTIC_TOOL_READ: &str = include_str!("rerank_agentic_tool_read.txt");

// ─── MCP freshness/degradation guidance (mcp.rs) ─────────────────────────
pub const MCP_DEGRADE_INDEX_FAILED: &str = include_str!("mcp_degrade_index_failed.txt");
pub const MCP_DEGRADE_INDEXING: &str = include_str!("mcp_degrade_indexing.txt");
pub const MCP_FILE_RETRIEVAL_HINT: &str = include_str!("mcp_file_retrieval_hint.txt");

// ─── LLM transport (llm/openai.rs) ───────────────────────────────────────
pub const LLM_RESPOND_IN_JSON: &str = include_str!("llm_respond_in_json.txt");

/// Substitute `{{token}}` placeholders in a compile-time prompt template.
///
/// Each `(key, value)` replaces every occurrence of `{{key}}` with `value`.
/// Double-brace delimiters are used so the literal single-brace `{...}` JSON
/// examples embedded in some prompts never collide with a placeholder.
pub fn render(template: &str, vars: &[(&str, &str)]) -> String {
    let mut out = template.to_owned();
    for (key, value) in vars {
        out = out.replace(&format!("{{{{{}}}}}", key), value);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_fills_placeholders() {
        let t = "Repo: {{repo}}, tool: {{tool}}.";
        assert_eq!(
            render(t, &[("repo", "/x"), ("tool", "grep")]),
            "Repo: /x, tool: grep."
        );
    }

    #[test]
    fn render_leaves_literal_json_braces_untouched() {
        // A template may embed literal single-brace JSON next to placeholders;
        // only the double-brace token is substituted.
        let t = r#"Query: {{query}} -> {"keep":"full"}"#;
        assert_eq!(
            render(t, &[("query", "find x")]),
            r#"Query: find x -> {"keep":"full"}"#
        );
    }

    #[test]
    fn render_replaces_all_occurrences() {
        let t = "{{a}} and {{a}} again";
        assert_eq!(render(t, &[("a", "X")]), "X and X again");
    }

    #[test]
    fn embedded_prompts_are_non_empty() {
        // Guard against an empty/missing .txt slipping in via include_str!.
        for p in [
            CHAT_SYSTEM,
            CHAT_TOOL_CODEBASE,
            RERANK_INTRO,
            RERANK_SYSTEM_STRUCTURED,
            RERANK_AGENTIC_SYSTEM,
            MCP_DEGRADE_INDEXING,
            LLM_RESPOND_IN_JSON,
        ] {
            assert!(!p.trim().is_empty());
        }
    }
}
