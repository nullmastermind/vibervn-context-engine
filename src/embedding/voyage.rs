use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::embedding::InputType;

const VOYAGE_ENDPOINT: &str = "https://api.voyageai.com/v1/embeddings";
pub const MAX_BATCH_SIZE: usize = 128;

/// Resolve the embeddings URL from an optional user-supplied base.
///
/// Normalization rules (mirrors `llm::openai::chat_url`):
///   * `None`, empty, or whitespace-only → `VOYAGE_ENDPOINT`.
///   * Trim whitespace, then strip a trailing `/`.
///   * If the path already ends in `/embeddings`, keep it as-is.
///   * Otherwise append `/embeddings`.
pub fn voyage_url(base: Option<&str>) -> String {
    let raw = match base {
        Some(s) => s.trim(),
        None => "",
    };
    if raw.is_empty() {
        return VOYAGE_ENDPOINT.to_owned();
    }
    let trimmed = raw.trim_end_matches('/');
    if trimmed.ends_with("/embeddings") {
        trimmed.to_owned()
    } else {
        format!("{trimmed}/embeddings")
    }
}

// ─── Request / response shapes ────────────────────────────────────────────

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    input_type: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

// ─── Client ───────────────────────────────────────────────────────────────

/// VoyageAI embedding client with round-robin key rotation and retry on 429.
#[derive(Clone)]
pub struct VoyageClient {
    inner: Arc<VoyageInner>,
}

struct VoyageInner {
    http: Client,
    /// Tighter-timeout client for user-facing query embedding (30s vs 120s).
    query_http: Client,
    model: String,
    api_keys: Vec<String>,
    /// Resolved embeddings endpoint URL.
    endpoint: String,
    /// Round-robin cursor — atomically advanced on each batch call.
    key_cursor: AtomicUsize,
}

impl VoyageClient {
    /// Create a new client. Returns `Err` if `api_keys` is empty.
    pub fn new(model: String, api_keys: Vec<String>, base_url: Option<&str>) -> Result<Self> {
        if api_keys.is_empty() {
            bail!("VoyageAI client requires at least one API key");
        }
        let http = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("build reqwest client")?;
        let query_http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("build query reqwest client")?;
        let endpoint = voyage_url(base_url);
        Ok(Self {
            inner: Arc::new(VoyageInner {
                http,
                query_http,
                model,
                api_keys,
                endpoint,
                key_cursor: AtomicUsize::new(0),
            }),
        })
    }

    /// Return the configured embedding model name.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// Embed a single query string with bounded retry.
    ///
    /// Uses `input_type: "query"`. On 429 from all keys, waits 2 s and retries
    /// once. A second 429 wave returns `Err`. Non-429 errors return `Err` immediately.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let n_keys = self.inner.api_keys.len();
        let start_cursor = self.inner.key_cursor.fetch_add(1, Ordering::Relaxed) % n_keys;

        // First pass — try each key once (30s timeout per attempt).
        for offset in 0..n_keys {
            let key_idx = (start_cursor + offset) % n_keys;
            let key = &self.inner.api_keys[key_idx];
            match self.try_embed_query_with_key(key, &texts, InputType::Query).await {
                Ok(mut embeddings) => {
                    return embeddings
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("VoyageAI returned empty embeddings"));
                }
                Err(EmbedError::RateLimited) => {
                    warn!(key_index = key_idx, "VoyageAI 429 on query embed — trying next key");
                }
                Err(EmbedError::Other(e)) => return Err(e),
            }
        }

        // All keys 429 — one backoff attempt (2 s), then return Err.
        warn!("all VoyageAI keys rate-limited on query embed; backing off 2s");
        tokio::time::sleep(Duration::from_secs(2)).await;

        for offset in 0..n_keys {
            let key_idx = (start_cursor + offset) % n_keys;
            let key = &self.inner.api_keys[key_idx];
            match self.try_embed_query_with_key(key, &texts, InputType::Query).await {
                Ok(mut embeddings) => {
                    return embeddings
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("VoyageAI returned empty embeddings"));
                }
                Err(EmbedError::RateLimited) => continue,
                Err(EmbedError::Other(e)) => return Err(e),
            }
        }

        anyhow::bail!("VoyageAI query embed still rate-limited after backoff")
    }

    /// Embed texts in batches of up to 128. Returns one Vec<f32> per input.
    pub async fn embed(&self, texts: &[String], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());
        for batch in texts.chunks(MAX_BATCH_SIZE) {
            let embeddings = self.embed_batch(batch, input_type).await?;
            all_embeddings.extend(embeddings);
        }
        Ok(all_embeddings)
    }

    /// Embed a batch of up to `MAX_BATCH_SIZE` texts. Public so the pipeline can
    /// drive batching manually and report per-batch progress between awaits.
    pub async fn embed_batch(&self, texts: &[String], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let n_keys = self.inner.api_keys.len();
        let start_cursor = self.inner.key_cursor.fetch_add(1, Ordering::Relaxed) % n_keys;

        // Try each key once before falling back to exponential backoff.
        for offset in 0..n_keys {
            let key_idx = (start_cursor + offset) % n_keys;
            let key = &self.inner.api_keys[key_idx];

            match self.try_embed_with_key(key, texts, input_type).await {
                Ok(embeddings) => return Ok(embeddings),
                Err(EmbedError::RateLimited) => {
                    warn!(key_index = key_idx, "VoyageAI 429 — trying next key");
                }
                // Non-429 error: abort immediately, old data untouched.
                Err(EmbedError::Other(e)) => return Err(e),
            }
        }

        // All keys returned 429 — exponential backoff, retry indefinitely.
        let mut delay_secs: u64 = 2;
        loop {
            warn!(
                delay_secs = delay_secs,
                "all VoyageAI keys rate-limited; backing off"
            );
            tokio::time::sleep(Duration::from_secs(delay_secs)).await;

            for offset in 0..n_keys {
                let key_idx = (start_cursor + offset) % n_keys;
                let key = &self.inner.api_keys[key_idx];
                match self.try_embed_with_key(key, texts, input_type).await {
                    Ok(embeddings) => {
                        info!("VoyageAI embed succeeded after backoff");
                        return Ok(embeddings);
                    }
                    Err(EmbedError::RateLimited) => continue,
                    Err(EmbedError::Other(e)) => return Err(e),
                }
            }

            delay_secs = (delay_secs * 2).min(60);
        }
    }

    async fn try_embed_with_key(
        &self,
        key: &str,
        texts: &[String],
        input_type: InputType,
    ) -> std::result::Result<Vec<Vec<f32>>, EmbedError> {
        self.try_embed_with_key_using(&self.inner.http, key, texts, input_type).await
    }

    async fn try_embed_query_with_key(
        &self,
        key: &str,
        texts: &[String],
        input_type: InputType,
    ) -> std::result::Result<Vec<Vec<f32>>, EmbedError> {
        self.try_embed_with_key_using(&self.inner.query_http, key, texts, input_type).await
    }

    async fn try_embed_with_key_using(
        &self,
        client: &Client,
        key: &str,
        texts: &[String],
        input_type: InputType,
    ) -> std::result::Result<Vec<Vec<f32>>, EmbedError> {
        let body = EmbedRequest {
            model: &self.inner.model,
            input: texts,
            input_type: input_type.as_str(),
        };

        let response = client
            .post(&self.inner.endpoint)
            .bearer_auth(key)
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbedError::Other(e.into()))?;

        let status = response.status();

        if status.as_u16() == 429 {
            return Err(EmbedError::RateLimited);
        }

        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(EmbedError::Other(anyhow::anyhow!(
                "VoyageAI error {}: {}",
                status,
                text
            )));
        }

        let resp: EmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbedError::Other(e.into()))?;

        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }
}

enum EmbedError {
    RateLimited,
    Other(anyhow::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voyage_url_default_when_none() {
        assert_eq!(voyage_url(None), VOYAGE_ENDPOINT);
    }

    #[test]
    fn voyage_url_default_when_blank() {
        assert_eq!(voyage_url(Some("")), VOYAGE_ENDPOINT);
        assert_eq!(voyage_url(Some("   ")), VOYAGE_ENDPOINT);
        assert_eq!(voyage_url(Some("\t\n")), VOYAGE_ENDPOINT);
    }

    #[test]
    fn voyage_url_appends_to_base() {
        assert_eq!(
            voyage_url(Some("https://my-proxy.com/v1")),
            "https://my-proxy.com/v1/embeddings"
        );
        assert_eq!(
            voyage_url(Some("http://localhost:8080/api/v1")),
            "http://localhost:8080/api/v1/embeddings"
        );
    }

    #[test]
    fn voyage_url_strips_trailing_slash() {
        assert_eq!(
            voyage_url(Some("https://my-proxy.com/v1/")),
            "https://my-proxy.com/v1/embeddings"
        );
    }

    #[test]
    fn voyage_url_accepts_full_form() {
        assert_eq!(
            voyage_url(Some("https://my-proxy.com/v1/embeddings")),
            "https://my-proxy.com/v1/embeddings"
        );
    }

    #[test]
    fn voyage_url_accepts_full_form_trailing_slash() {
        assert_eq!(
            voyage_url(Some("https://my-proxy.com/v1/embeddings/")),
            "https://my-proxy.com/v1/embeddings"
        );
    }
}
