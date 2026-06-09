pub mod google;
pub mod openai;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use anyhow::{Result, bail};
use reqwest::Client;
use tracing::warn;
use crate::config::LlmConfig;

// ─── Shared types for tool-calling ───────────────────────────────────────

/// A tool definition passed to the LLM.
#[derive(Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// A single tool call the model wants to make.
#[derive(Clone, Debug, Default)]
pub struct ToolCall {
    pub name: String,
    pub id: Option<String>,
    pub args: serde_json::Value,
    /// Gemini 2.5/3.x: an opaque `thoughtSignature` attached (at the part level,
    /// as a sibling of `functionCall`) when the model emits a tool call while
    /// thinking is active. It MUST be echoed verbatim when this model turn is
    /// replayed in the conversation history, or the next request 400s with
    /// "Function call is missing a thought_signature". `None` for providers /
    /// models that don't produce one (OpenAI, non-thinking Gemini).
    pub thought_signature: Option<String>,
}

/// A tool result to send back to the model.
#[derive(Clone)]
pub struct ToolResult {
    pub name: String,
    pub id: Option<String>,
    pub content: String,
}

/// Conversation messages for multi-turn tool-calling.
pub enum ChatMessage {
    User(String),
    ModelToolCalls(Vec<ToolCall>),
    ToolResults(Vec<ToolResult>),
}

/// Unified result from a tool-calling turn.
pub enum ToolTurnResult {
    Text(String),
    ToolCalls(Vec<ToolCall>),
}

// ─── LlmClient ───────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct LlmClient {
    provider: String,
    model: String,
    api_keys: Vec<String>,
    http: Client,
    key_cursor: std::sync::Arc<AtomicUsize>,
    use_structured_output: bool,
    /// Custom OpenAI-compatible endpoint. Honored only when `provider ==
    /// "openai"`; ignored for other providers. `None` / blank → the OpenAI
    /// client falls back to `api.openai.com`. Normalization (base form vs
    /// full URL) happens centrally in `openai::chat_url`.
    openai_base_url: Option<String>,
}

/// Whether `provider` has a native JSON output mode the reranker can request.
fn provider_supports_structured_output(provider: &str) -> bool {
    matches!(provider, "google" | "openai")
}

impl LlmClient {
    /// Create a new client. Returns None if api_keys is empty.
    pub fn new(config: &LlmConfig) -> Option<Self> {
        if config.api_keys.is_empty() {
            return None;
        }
        let http = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .ok()?;
        Some(Self {
            provider: config.provider.clone(),
            model: config.rerank_model.clone(),
            api_keys: config.api_keys.clone(),
            http,
            key_cursor: std::sync::Arc::new(AtomicUsize::new(0)),
            use_structured_output: config.use_structured_output,
            openai_base_url: config.openai_base_url.clone(),
        })
    }

    /// Whether this client will request native JSON output for reranking.
    pub fn structured_output_active(&self) -> bool {
        if !self.use_structured_output {
            return false;
        }
        if provider_supports_structured_output(&self.provider) {
            true
        } else {
            tracing::warn!(
                provider = %self.provider,
                "use_structured_output is enabled but provider has no native JSON mode; \
                 falling back to XML rerank path"
            );
            false
        }
    }

    /// Dispatch to the provider-specific completion function.
    async fn call_provider(&self, system: &str, user: &str, temperature: f32, structured: bool, key: &str) -> Result<String> {
        match self.provider.as_str() {
            "google" => google::complete(&self.http, &self.model, key, system, user, temperature, structured).await,
            "openai" => openai::complete(&self.http, &self.model, key, system, user, temperature, structured, self.openai_base_url.as_deref()).await,
            other => bail!("unsupported LLM provider: {other}"),
        }
    }

    /// Send a completion request to the configured LLM provider.
    /// Rotates through all keys on failure; backs off 2s and retries once more
    /// before returning the last error.
    pub async fn complete(&self, system: &str, user: &str, temperature: f32, structured: bool) -> Result<String> {
        let n_keys = self.api_keys.len();
        let start_cursor = self.key_cursor.fetch_add(1, Ordering::Relaxed) % n_keys;

        let mut last_err = None;

        // First pass — try each key once.
        for offset in 0..n_keys {
            let key_idx = (start_cursor + offset) % n_keys;
            let key = &self.api_keys[key_idx];
            match self.call_provider(system, user, temperature, structured, key).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    warn!(key_index = key_idx, error = %e, "LLM call failed — trying next key");
                    last_err = Some(e);
                }
            }
        }

        // All keys failed — backoff 2s and retry once more.
        tokio::time::sleep(Duration::from_secs(2)).await;

        for offset in 0..n_keys {
            let key_idx = (start_cursor + offset) % n_keys;
            let key = &self.api_keys[key_idx];
            match self.call_provider(system, user, temperature, structured, key).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap())
    }

    /// Dispatch to the provider-specific tool-calling function.
    #[allow(clippy::too_many_arguments)]
    async fn call_provider_with_tools(
        &self,
        system: &str,
        contents: &[ChatMessage],
        tools: &[ToolDef],
        temperature: f32,
        force_tool_use: bool,
        key: &str,
        prompt_cache_key: Option<&str>,
    ) -> Result<ToolTurnResult> {
        match self.provider.as_str() {
            "google" => {
                let r = google::complete_with_tools(&self.http, &self.model, key, system, contents, tools, temperature, force_tool_use).await?;
                match r {
                    google::ToolTurnResult::Text(t) => Ok(ToolTurnResult::Text(t)),
                    google::ToolTurnResult::ToolCalls(calls) => Ok(ToolTurnResult::ToolCalls(
                        calls.into_iter().map(|c| ToolCall {
                            name: c.call.name,
                            id: c.call.id,
                            args: c.call.args,
                            thought_signature: c.thought_signature,
                        }).collect()
                    )),
                }
            }
            "openai" => {
                let r = openai::complete_with_tools(&self.http, &self.model, key, system, contents, tools, temperature, force_tool_use, prompt_cache_key, self.openai_base_url.as_deref()).await?;
                match r {
                    openai::ToolTurnResult::Text(t) => Ok(ToolTurnResult::Text(t)),
                    openai::ToolTurnResult::ToolCalls(calls) => Ok(ToolTurnResult::ToolCalls(
                        calls.into_iter().map(|c| {
                            let args = serde_json::from_str(&c.function.arguments).unwrap_or(serde_json::Value::Object(Default::default()));
                            ToolCall { name: c.function.name, id: Some(c.id), args, thought_signature: None }
                        }).collect()
                    )),
                }
            }
            other => bail!("unsupported LLM provider for tool-calling: {other}"),
        }
    }

    /// Send a tool-calling request with key rotation + retry.
    ///
    /// `force_tool_use`: when true, the provider is told the model MUST emit a
    /// tool call and may NOT reply with prose (Gemini `mode:ANY`, OpenAI
    /// `tool_choice:required`). The agentic loop sets this while no chunk has
    /// been committed yet, so the model cannot answer the question directly
    /// instead of selecting chunks. Once a chunk is added it flips to false so
    /// the agent can finish with a text summary.
    pub async fn complete_with_tools(
        &self,
        system: &str,
        contents: &[ChatMessage],
        tools: &[ToolDef],
        temperature: f32,
        force_tool_use: bool,
        prompt_cache_key: Option<&str>,
    ) -> Result<ToolTurnResult> {
        let n_keys = self.api_keys.len();
        let start_cursor = self.key_cursor.fetch_add(1, Ordering::Relaxed) % n_keys;

        let mut last_err = None;

        for offset in 0..n_keys {
            let key_idx = (start_cursor + offset) % n_keys;
            let key = &self.api_keys[key_idx];
            match self.call_provider_with_tools(system, contents, tools, temperature, force_tool_use, key, prompt_cache_key).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    warn!(key_index = key_idx, error = %e, "LLM tool-call failed — trying next key");
                    last_err = Some(e);
                }
            }
        }

        tokio::time::sleep(Duration::from_secs(2)).await;

        for offset in 0..n_keys {
            let key_idx = (start_cursor + offset) % n_keys;
            let key = &self.api_keys[key_idx];
            match self.call_provider_with_tools(system, contents, tools, temperature, force_tool_use, key, prompt_cache_key).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap())
    }
}
