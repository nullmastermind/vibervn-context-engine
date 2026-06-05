use std::time::Instant;
use regex::Regex;
use tracing::warn;
use crate::llm::LlmClient;
use crate::query::merger::MergeChunk;

#[derive(Debug, Clone, serde::Serialize)]
pub struct RerankOutput {
    pub reranked_indices: Vec<usize>,
    pub raw_request: String,
    pub raw_response: String,
    pub elapsed_ms: u64,
    pub fallback_used: bool,
    pub skip_reason: Option<String>,
}

pub async fn rerank(
    query: &str,
    chunks: &[MergeChunk],
    llm_client: Option<&LlmClient>,
) -> RerankOutput {
    let n = chunks.len();
    let all_indices: Vec<usize> = (0..n).collect();

    let client = match llm_client {
        Some(c) => c,
        None => return RerankOutput {
            reranked_indices: all_indices,
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
            raw_request: String::new(),
            raw_response: String::new(),
            elapsed_ms: 0,
            fallback_used: false,
            skip_reason: None,
        };
    }

    let system = "You are a code search relevance ranker. \
        Given a query and numbered code chunks with metadata (relevance score, callers count, \
        callees count, flow membership), your job is to rank the chunks by relevance to the query. \
        OMIT chunks that are not relevant to the query. \
        Higher score, more callers, and flow membership indicate higher structural importance. \
        When both source code and documentation chunks are relevant to the query, \
        prefer source code over documentation because code is the source of truth. \
        Documentation can be outdated or inaccurate, but the code always reflects actual behavior. \
        Your output MUST contain a pair of XML tags called ranked_indices. \
        Between the opening <ranked_indices> tag and the closing </ranked_indices> tag, \
        place a JSON array of integer chunk indices sorted from most relevant to least relevant. \
        Only include indices of chunks that are actually relevant to the query. \
        Do not include any other text between the tags, only the JSON array.";

    // Build user prompt with chunk entries
    let mut entries = Vec::with_capacity(n);
    for (i, chunk) in chunks.iter().enumerate() {
        let meta_str = format!("score={:.2}", chunk.score);
        let symbol_display = chunk.symbol.as_deref().unwrap_or("no symbol");

        // Truncate content at 100 lines
        let content = truncate_content(&chunk.content, 100);

        let entry = format!(
            "[{i}] {meta_str} | {}:{}-{} ({symbol_display})\n<content chunk-index=\"{i}\">\n{content}\n</content>",
            chunk.file, chunk.line_start, chunk.line_end
        );
        entries.push(entry);
    }

    let chunks_text = entries.join("\n---\n");
    let user_prompt = format!(
        "Query: {query}\n\nChunks:\n{chunks_text}\n\n\
         Now rank the chunks by relevance. \
         Write the opening tag <ranked_indices>, then a JSON array of the relevant chunk indices \
         from most to least relevant, then the closing tag </ranked_indices>."
    );

    let raw_request = format!("[System]\n{system}\n\n[User]\n{user_prompt}");

    let start = Instant::now();
    let result = client.complete(system, &user_prompt, 0.0).await;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(response) => {
            let mut output = parse_rerank_response(&response, n, elapsed_ms);
            output.raw_request = raw_request;
            output
        }
        Err(e) => {
            warn!(error = %e, "LLM rerank call failed, using fallback order");
            RerankOutput {
                reranked_indices: all_indices,
                raw_request,
                raw_response: String::new(),
                elapsed_ms,
                fallback_used: true,
                skip_reason: Some(format!("LLM request failed: {e}")),
            }
        }
    }
}

fn parse_rerank_response(response: &str, n: usize, elapsed_ms: u64) -> RerankOutput {
    let all_indices: Vec<usize> = (0..n).collect();

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

    // Parse JSON array
    let indices: Vec<usize> = match serde_json::from_str::<Vec<serde_json::Value>>(&text) {
        Ok(arr) => arr.iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .filter(|&idx| idx < n)
            .collect(),
        Err(_) => {
            warn!(raw = %response, "failed to parse rerank response as JSON array");
            return RerankOutput {
                reranked_indices: all_indices,
                raw_request: String::new(),
                raw_response: response.to_owned(),
                elapsed_ms,
                fallback_used: true,
                skip_reason: Some("failed to parse LLM response".to_owned()),
            };
        }
    };

    if indices.is_empty() {
        return RerankOutput {
            reranked_indices: all_indices,
            raw_request: String::new(),
            raw_response: response.to_owned(),
            elapsed_ms,
            fallback_used: true,
            skip_reason: Some("LLM returned empty ranking".to_owned()),
        };
    }

    RerankOutput {
        reranked_indices: indices,
        raw_request: String::new(),
        raw_response: response.to_owned(),
        elapsed_ms,
        fallback_used: false,
        skip_reason: None,
    }
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
