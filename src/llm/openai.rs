use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::info;

use super::ToolDef;

// ─── Single-turn types ───────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    /// "required" forces a tool call (no prose); omitted = model's choice (auto).
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Message {
    Standard { role: String, content: String },
    Assistant { role: String, content: Option<String>, tool_calls: Vec<ToolCallMessage> },
    ToolResult { role: String, tool_call_id: String, content: String },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCallMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    kind: String,
    function: OpenAIFunction,
}

#[derive(Serialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

// ─── Response types ──────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ChatResponse {
    choices: Option<Vec<Choice>>,
    error: Option<OpenAIError>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Deserialize)]
struct PromptTokensDetails {
    cached_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCallMessage>>,
}

#[derive(Deserialize)]
struct OpenAIError {
    message: String,
}

// ─── Public API ──────────────────────────────────────────────────────────

/// Default OpenAI chat-completions URL (used when `base_url` is unset or blank).
const DEFAULT_OPENAI_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Resolve the chat-completions URL from an optional user-supplied base.
///
/// Normalization rules (single source of truth — both `complete()` and
/// `complete_with_tools()` go through this helper):
///   * `None`, empty, or whitespace-only → `DEFAULT_OPENAI_URL`.
///   * Trim whitespace, then strip a trailing `/`.
///   * If the path already ends in `/chat/completions`, keep it as-is.
///   * Otherwise append `/chat/completions`.
///
/// This means users can paste either the base form (`http://localhost:11434/v1`)
/// or the full form (`http://localhost:11434/v1/chat/completions`) and either
/// works — they hit the same endpoint with no double `/chat/completions`
/// appended and no double slashes.
fn chat_url(base: Option<&str>) -> String {
    let raw = match base {
        Some(s) => s.trim(),
        None => "",
    };
    if raw.is_empty() {
        return DEFAULT_OPENAI_URL.to_owned();
    }
    let trimmed = raw.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_owned()
    } else {
        format!("{trimmed}/chat/completions")
    }
}

fn log_cache_metrics(usage: &Option<Usage>) {
    if let Some(u) = usage {
        let prompt = u.prompt_tokens.unwrap_or(0);
        let completion = u.completion_tokens.unwrap_or(0);
        let cached = u.prompt_tokens_details.as_ref()
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0);
        info!(
            prompt_tokens = prompt,
            completion_tokens = completion,
            cached_tokens = cached,
            cache_hit_pct = if prompt > 0 { (cached as f64 / prompt as f64 * 100.0) as u32 } else { 0 },
            "openai cache metrics"
        );
    }
}

/// Guarantee the literal token "json" is present in the messages when
/// `structured` json_object mode is on (OpenAI rejects the request otherwise).
fn ensure_json_token(structured: bool, system: &str, user: &str) -> String {
    let missing = structured
        && !system.to_lowercase().contains("json")
        && !user.to_lowercase().contains("json");
    if missing {
        format!("{system}\nRespond in JSON.")
    } else {
        system.to_owned()
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn complete(
    http: &Client,
    model: &str,
    api_key: &str,
    system: &str,
    user: &str,
    temperature: f32,
    structured: bool,
    base_url: Option<&str>,
) -> Result<String> {
    let url = chat_url(base_url);

    let system_owned = ensure_json_token(structured, system, user);

    let body = ChatRequest {
        model: model.to_owned(),
        messages: vec![
            Message::Standard { role: "system".to_owned(), content: system_owned },
            Message::Standard { role: "user".to_owned(), content: user.to_owned() },
        ],
        temperature,
        response_format: structured.then(|| ResponseFormat { kind: "json_object".to_owned() }),
        tools: None,
        tool_choice: None,
        prompt_cache_key: None,
    };

    info!(
        url = %url,
        model = %model,
        structured = structured,
        "openai completion request"
    );

    let resp = match http
        .post(&url)
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(
                url = %url,
                model = %model,
                error = %e,
                error_debug = ?e,
                is_timeout = e.is_timeout(),
                is_connect = e.is_connect(),
                "openai completion: connection-level failure"
            );
            return Err(e).context("OpenAI HTTP request failed");
        }
    };

    let status = resp.status();
    let text = resp.text().await.context("failed to read OpenAI response body")?;

    if !status.is_success() {
        tracing::error!(
            url = %url,
            model = %model,
            status = %status,
            response_body = %text,
            "openai completion HTTP error"
        );
        bail!("OpenAI API returned HTTP {status}: {text}");
    }

    let parsed: ChatResponse = serde_json::from_str(&text)
        .context("failed to parse OpenAI response JSON")?;

    if let Some(err) = parsed.error {
        bail!("OpenAI API error: {}", err.message);
    }

    log_cache_metrics(&parsed.usage);

    let result_text = parsed.choices
        .and_then(|c| c.into_iter().next())
        .and_then(|c| c.message.content)
        .unwrap_or_default();

    Ok(result_text)
}

/// Result of a single turn in the tool-calling loop.
pub enum ToolTurnResult {
    /// Model returned text (done).
    Text(String),
    /// Model requested tool calls.
    ToolCalls(Vec<ToolCallMessage>),
}

/// Send a multi-turn tool-calling request to OpenAI.
#[allow(clippy::too_many_arguments)]
pub async fn complete_with_tools(
    http: &Client,
    model: &str,
    api_key: &str,
    system: &str,
    contents: &[super::ChatMessage],
    tools: &[ToolDef],
    temperature: f32,
    force_tool_use: bool,
    prompt_cache_key: Option<&str>,
    base_url: Option<&str>,
) -> Result<ToolTurnResult> {
    let url = chat_url(base_url);

    let mut messages: Vec<Message> = Vec::with_capacity(contents.len() + 1);
    messages.push(Message::Standard { role: "system".to_owned(), content: system.to_owned() });

    for msg in contents {
        match msg {
            super::ChatMessage::User(text) => {
                messages.push(Message::Standard { role: "user".to_owned(), content: text.clone() });
            }
            super::ChatMessage::ModelToolCalls(calls) => {
                let tool_calls: Vec<ToolCallMessage> = calls.iter().map(|c| ToolCallMessage {
                    id: c.id.clone().unwrap_or_default(),
                    kind: "function".to_owned(),
                    function: ToolCallFunction {
                        name: c.name.clone(),
                        arguments: c.args.to_string(),
                    },
                }).collect();
                messages.push(Message::Assistant {
                    role: "assistant".to_owned(),
                    content: None,
                    tool_calls,
                });
            }
            super::ChatMessage::ToolResults(results) => {
                for r in results {
                    messages.push(Message::ToolResult {
                        role: "tool".to_owned(),
                        tool_call_id: r.id.clone().unwrap_or_default(),
                        content: r.content.clone(),
                    });
                }
            }
        }
    }

    let openai_tools: Vec<OpenAITool> = tools.iter().map(|t| OpenAITool {
        kind: "function".to_owned(),
        function: OpenAIFunction {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.parameters.clone(),
        },
    }).collect();

    let body = ChatRequest {
        model: model.to_owned(),
        messages,
        temperature,
        response_format: None,
        tools: Some(openai_tools),
        // "required" forces a tool call while no chunk is committed; otherwise
        // omit (model may finish with a text summary). Non-official endpoints
        // (custom base_url) often only support "auto"/"none", so downgrade.
        tool_choice: if force_tool_use {
            let is_custom = base_url.is_some_and(|u| !u.trim().is_empty());
            Some(if is_custom { "auto".to_owned() } else { "required".to_owned() })
        } else {
            None
        },
        prompt_cache_key: prompt_cache_key.map(|s| s.to_owned()),
    };

    let request_json = serde_json::to_string(&body).unwrap_or_default();
    info!(
        url = %url,
        model = %model,
        tool_count = tools.len(),
        force_tool_use = force_tool_use,
        request_bytes = request_json.len(),
        "openai tool-call request"
    );

    let resp = match http
        .post(&url)
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(
                url = %url,
                model = %model,
                error = %e,
                error_debug = ?e,
                is_timeout = e.is_timeout(),
                is_connect = e.is_connect(),
                "openai tool-call: connection-level failure"
            );
            return Err(e).context("OpenAI tool-calling HTTP request failed");
        }
    };

    let status = resp.status();
    let text = resp.text().await.context("failed to read OpenAI response body")?;

    if !status.is_success() {
        tracing::error!(
            url = %url,
            model = %model,
            status = %status,
            response_body = %text,
            "openai tool-call HTTP error"
        );
        bail!("OpenAI API returned HTTP {status}: {text}");
    }

    let parsed: ChatResponse = serde_json::from_str(&text)
        .context("failed to parse OpenAI response JSON")?;

    if let Some(err) = parsed.error {
        bail!("OpenAI API error: {}", err.message);
    }

    log_cache_metrics(&parsed.usage);

    let choice = parsed.choices
        .and_then(|c| c.into_iter().next());

    match choice {
        Some(c) => {
            if let Some(tool_calls) = c.message.tool_calls
                && !tool_calls.is_empty()
            {
                return Ok(ToolTurnResult::ToolCalls(tool_calls));
            }
            Ok(ToolTurnResult::Text(c.message.content.unwrap_or_default()))
        }
        None => Ok(ToolTurnResult::Text(String::new())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_present_in_system_is_unchanged() {
        let s = ensure_json_token(true, "Respond with a JSON object.", "rank these");
        assert_eq!(s, "Respond with a JSON object.");
    }

    #[test]
    fn token_present_in_user_leaves_system_unchanged() {
        let s = ensure_json_token(true, "You are a ranker.", "reply as json please");
        assert_eq!(s, "You are a ranker.");
    }

    #[test]
    fn token_absent_appends_directive() {
        let s = ensure_json_token(true, "You are a ranker.", "rank these chunks");
        assert!(s.to_lowercase().contains("json"), "must inject the json token");
    }

    #[test]
    fn not_structured_never_modifies() {
        let s = ensure_json_token(false, "You are a ranker.", "rank these chunks");
        assert_eq!(s, "You are a ranker.");
    }

    #[test]
    fn usage_with_cached_tokens_deserializes() {
        let json = r#"{
            "choices": [{"message": {"content": "hello"}}],
            "usage": {
                "prompt_tokens": 1500,
                "completion_tokens": 200,
                "prompt_tokens_details": {
                    "cached_tokens": 1024
                }
            }
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).expect("parse");
        let usage = resp.usage.expect("has usage");
        assert_eq!(usage.prompt_tokens, Some(1500));
        assert_eq!(usage.completion_tokens, Some(200));
        let details = usage.prompt_tokens_details.expect("has details");
        assert_eq!(details.cached_tokens, Some(1024));
    }

    #[test]
    fn usage_without_cache_details_deserializes() {
        let json = r#"{
            "choices": [{"message": {"content": "hi"}}],
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 100
            }
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).expect("parse");
        let usage = resp.usage.expect("has usage");
        assert_eq!(usage.prompt_tokens, Some(500));
        assert!(usage.prompt_tokens_details.is_none());
    }

    #[test]
    fn response_without_usage_deserializes() {
        let json = r#"{"choices": [{"message": {"content": "ok"}}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).expect("parse");
        assert!(resp.usage.is_none());
        let text = resp.choices.unwrap().into_iter().next().unwrap().message.content;
        assert_eq!(text.as_deref(), Some("ok"));
    }

    #[test]
    fn prompt_cache_key_serializes_when_present() {
        let body = ChatRequest {
            model: "gpt-4o".to_owned(),
            messages: vec![],
            temperature: 0.0,
            response_format: None,
            tools: None,
            tool_choice: None,
            prompt_cache_key: Some("test-key-123".to_owned()),
        };
        let json = serde_json::to_value(&body).expect("serialize");
        assert_eq!(json.get("prompt_cache_key").and_then(|v| v.as_str()), Some("test-key-123"));
    }

    #[test]
    fn prompt_cache_key_omitted_when_none() {
        let body = ChatRequest {
            model: "gpt-4o".to_owned(),
            messages: vec![],
            temperature: 0.0,
            response_format: None,
            tools: None,
            tool_choice: None,
            prompt_cache_key: None,
        };
        let json = serde_json::to_value(&body).expect("serialize");
        assert!(json.get("prompt_cache_key").is_none());
    }

    // ─── chat_url normalization ──────────────────────────────────────────

    #[test]
    fn chat_url_default_when_none() {
        assert_eq!(chat_url(None), DEFAULT_OPENAI_URL);
    }

    #[test]
    fn chat_url_default_when_blank() {
        assert_eq!(chat_url(Some("")), DEFAULT_OPENAI_URL);
        assert_eq!(chat_url(Some("   ")), DEFAULT_OPENAI_URL);
        assert_eq!(chat_url(Some("\t\n")), DEFAULT_OPENAI_URL);
    }

    #[test]
    fn chat_url_appends_to_base() {
        assert_eq!(
            chat_url(Some("http://localhost:11434/v1")),
            "http://localhost:11434/v1/chat/completions"
        );
        assert_eq!(
            chat_url(Some("https://openrouter.ai/api/v1")),
            "https://openrouter.ai/api/v1/chat/completions"
        );
    }

    #[test]
    fn chat_url_strips_trailing_slash() {
        // Trailing slash on the base form must not produce a double slash.
        assert_eq!(
            chat_url(Some("http://localhost:11434/v1/")),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    #[test]
    fn chat_url_accepts_full_form() {
        // User pastes the full URL — return it unchanged (no double append).
        assert_eq!(
            chat_url(Some("http://localhost:11434/v1/chat/completions")),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    #[test]
    fn chat_url_accepts_full_form_trailing_slash() {
        // Full URL with a trailing slash is also a no-op (after strip).
        assert_eq!(
            chat_url(Some("http://localhost:11434/v1/chat/completions/")),
            "http://localhost:11434/v1/chat/completions"
        );
    }
}
