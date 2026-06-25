//! Router-side GLOBAL `/mcp` handler: a PROXYING MCP server.
//!
//! ## Why this exists (the repo-addressing problem)
//!
//! The global `/mcp` endpoint exposes the `codebase-retrieval` / `file-retrieval`
//! tools, but each tool call carries its OWN `workspace_full_path` — so a single
//! `/mcp` session is NOT bound to one repo. In the monolith the handler held the
//! `IndexEngine` + `repo_dbs` and ran the query directly. The router holds
//! NEITHER (that is the whole point of process-per-project). So the router's
//! global `/mcp` cannot run the query itself.
//!
//! The correct, complete behavior — not a "use /mcp-repo instead" punt — is to
//! PROXY each tool call to the worker that owns the call's `workspace_full_path`:
//! resolve the repo, acquire/spawn its worker via [`ProxyCtx`], and forward the
//! call to that worker's `/api/mcp-tool` REST endpoint. That REST endpoint runs
//! the SAME `run_codebase_retrieval` funnel the MCP tool would, so the output is
//! byte-identical to the monolith — the client cannot tell it was proxied.
//!
//! Per-repo `/mcp-repo/:id` (pre-bound workspace) is proxied separately as a raw
//! HTTP passthrough in `router::mod`; this module is ONLY the global, multi-repo
//! `/mcp` surface.

use rmcp::{
    ErrorData, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
};
use serde_json::json;

use super::proxy::{ProxyCtx, forward_json_to_worker};
use crate::store::normalize_repo_path;

/// Args for the proxied global `codebase-retrieval` tool. Mirrors
/// `mcp::CodebaseRetrievalArgs` (the fields the funnel reads): the free-form
/// request plus the workspace path that selects the repo/worker.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ProxyCodebaseRetrievalArgs {
    /// Natural-language description of the code or information you are looking for.
    pub information_request: String,
    /// Full path to the workspace/repository to search. Selects which worker
    /// handles the call.
    pub workspace_full_path: String,
}

/// Args for the proxied global `file-retrieval` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ProxyFileRetrievalArgs {
    /// Full path to the workspace/repository. Selects which worker handles the call.
    pub workspace_full_path: String,
    /// Relative path to the file within the repository (e.g. "src/main.rs").
    pub file_path: String,
    /// Natural-language description of what you're looking for in this file.
    pub information_request: String,
    /// Number of top-scoring snippets to return. Defaults to 5.
    pub top_k: Option<usize>,
}

/// Proxying global MCP handler. Holds the [`ProxyCtx`] so each tool call can
/// acquire/spawn the right worker and forward to it.
#[derive(Clone)]
pub struct ProxyMcpHandler {
    proxy: ProxyCtx,
    #[allow(dead_code)]
    tool_router: ToolRouter<ProxyMcpHandler>,
}

#[tool_router]
impl ProxyMcpHandler {
    pub fn new(proxy: ProxyCtx, enabled_tools: &[String]) -> Self {
        let all_tools: &[&str] = &["codebase-retrieval", "file-retrieval"];
        let mut router = Self::tool_router();
        for &name in all_tools {
            if !enabled_tools.iter().any(|e| e == name) {
                router.disable_route(name);
            }
        }
        Self {
            proxy,
            tool_router: router,
        }
    }

    #[doc = include_str!("../prompts/mcp_codebase_retrieval.txt")]
    #[tool(name = "codebase-retrieval")]
    async fn codebase_retrieval(
        &self,
        Parameters(args): Parameters<ProxyCodebaseRetrievalArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let repo = normalize_repo_path(args.workspace_full_path.trim());
        if repo.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "Error: workspace_full_path is required.".to_string(),
            )]));
        }
        // Forward to the worker's /api/mcp-tool — the SAME funnel run_codebase_
        // retrieval the monolith MCP tool uses, so output is byte-identical.
        let body = json!({
            "information_request": args.information_request,
            "workspace_full_path": args.workspace_full_path,
        });
        let text = forward_json_to_worker(&self.proxy, &repo, "/api/mcp-tool", body)
            .await
            .unwrap_or_else(|e| format!("Error: {e}"));
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[doc = include_str!("../prompts/mcp_file_retrieval.txt")]
    #[tool(name = "file-retrieval")]
    async fn file_retrieval(
        &self,
        Parameters(args): Parameters<ProxyFileRetrievalArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let repo = normalize_repo_path(args.workspace_full_path.trim());
        if repo.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "Error: workspace_full_path is required.".to_string(),
            )]));
        }
        let body = json!({
            "workspace_full_path": args.workspace_full_path,
            "file_path": args.file_path,
            "information_request": args.information_request,
            "top_k": args.top_k,
        });
        let text = forward_json_to_worker(&self.proxy, &repo, "/api/mcp-tool/file-retrieval", body)
            .await
            .unwrap_or_else(|e| format!("Error: {e}"));
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ProxyMcpHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_server_info(
            rmcp::model::Implementation::new("context-engine-rs", env!("CARGO_PKG_VERSION")),
        )
    }
}

/// Extract the `result` string the worker's `/api/mcp-tool` REST handlers wrap
/// their funnel output in (`{"result": "..."}`). Falls back to the raw body for
/// any other shape so an error surface is never swallowed.
pub fn unwrap_mcp_tool_result(raw: &str) -> String {
    serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .and_then(|v| v.get("result").and_then(|r| r.as_str()).map(String::from))
        .unwrap_or_else(|| raw.to_string())
}
