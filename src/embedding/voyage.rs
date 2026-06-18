use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::embedding::InputType;

const VOYAGE_ENDPOINT: &str = "https://api.voyageai.com/v1/embeddings";
const OPENAI_ENDPOINT: &str = "https://api.openai.com/v1/embeddings";
pub const MAX_BATCH_SIZE: usize = 128;
/// Byte-size cap for the sum of input texts in a single batch. VoyageAI's
/// per-batch token limit is 1M for voyage-4-lite. Worst-case for minified code
/// is ~2 bytes/token; 1.5 MB / 2 = 750K tokens — 25% headroom under the 1M limit.
const MAX_BATCH_BYTES: usize = 1_500_000;

/// Embedding provider. Selected by the `embedding.provider` config field and
/// the only thing that branches request building and the default endpoint —
/// all retry/batching/parsing logic below is provider-agnostic and shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Voyage,
    OpenAI,
}

impl Provider {
    /// Parse from the config `provider` string. Unknown values fall back to
    /// `Voyage` (the historical default) so a typo never silently breaks
    /// indexing — it just keeps the prior behavior.
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "openai" => Provider::OpenAI,
            _ => Provider::Voyage,
        }
    }

    /// Default embeddings endpoint when no base URL is configured.
    fn default_endpoint(self) -> &'static str {
        match self {
            Provider::Voyage => VOYAGE_ENDPOINT,
            Provider::OpenAI => OPENAI_ENDPOINT,
        }
    }
}

/// Resolve the embeddings URL for a provider from an optional user-supplied base.
///
/// Normalization rules (mirrors `llm::openai::chat_url`):
///   * `None`, empty, or whitespace-only → the provider's default endpoint.
///   * Trim whitespace, then strip a trailing `/`.
///   * If the path already ends in `/embeddings`, keep it as-is.
///   * Otherwise append `/embeddings`.
///
/// The `voyage_base_url` config field is reused as a generic embedding base URL
/// (only one provider is active at a time), so this normalization is identical
/// across providers — only the blank-fallback default differs.
pub fn embedding_url(provider: Provider, base: Option<&str>) -> String {
    let raw = match base {
        Some(s) => s.trim(),
        None => "",
    };
    if raw.is_empty() {
        return provider.default_endpoint().to_owned();
    }
    let trimmed = raw.trim_end_matches('/');
    if trimmed.ends_with("/embeddings") {
        trimmed.to_owned()
    } else {
        format!("{trimmed}/embeddings")
    }
}

/// Backward-compatible alias resolving against the Voyage default endpoint.
/// Retained so existing call sites/tests keep working; new code should use
/// `embedding_url(provider, base)`.
pub fn voyage_url(base: Option<&str>) -> String {
    embedding_url(Provider::Voyage, base)
}

// ─── Request / response shapes ────────────────────────────────────────────

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    /// Voyage-only: `document`/`query` hint. Omitted entirely for OpenAI, which
    /// rejects unknown fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<&'a str>,
    /// OpenAI-only: optional Matryoshka output-dimension truncation. Omitted for
    /// Voyage and when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
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
    /// Selected embedding provider — branches request building and the default
    /// endpoint only; all retry/batching/parsing below is provider-agnostic.
    provider: Provider,
    model: String,
    api_keys: Vec<String>,
    /// Resolved embeddings endpoint URL.
    endpoint: String,
    /// Optional output dimension (OpenAI Matryoshka). `None` → native dimension.
    /// Only sent when `provider == OpenAI`.
    dimensions: Option<u32>,
    /// Round-robin cursor — atomically advanced on each batch call.
    key_cursor: AtomicUsize,
}

impl VoyageClient {
    /// Create a Voyage embedding client (backward-compatible shim).
    ///
    /// Equivalent to `new_for_provider(Provider::Voyage, model, api_keys,
    /// base_url, None)`. Retained for tests and any direct Voyage construction;
    /// production call sites go through `new_for_provider` so the configured
    /// provider is honored everywhere.
    pub fn new(model: String, api_keys: Vec<String>, base_url: Option<&str>) -> Result<Self> {
        Self::new_for_provider(Provider::Voyage, model, api_keys, base_url, None)
    }

    /// Provider-aware factory: the single entry point every call site uses so the
    /// `embedding.provider` dropdown is honored at indexing, query, and MCP.
    ///
    /// `dimensions` is the optional OpenAI output-dimension truncation; it is
    /// stored but only emitted in the request body when `provider == OpenAI`.
    /// Returns `Err` if `api_keys` is empty.
    pub fn new_for_provider(
        provider: Provider,
        model: String,
        api_keys: Vec<String>,
        base_url: Option<&str>,
        dimensions: Option<u32>,
    ) -> Result<Self> {
        if api_keys.is_empty() {
            bail!("embedding client requires at least one API key");
        }
        let http = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("build reqwest client")?;
        let query_http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("build query reqwest client")?;
        let endpoint = embedding_url(provider, base_url);
        Ok(Self {
            inner: Arc::new(VoyageInner {
                http,
                query_http,
                provider,
                model,
                api_keys,
                endpoint,
                dimensions,
                key_cursor: AtomicUsize::new(0),
            }),
        })
    }

    /// Return the configured embedding model name.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// Return the configured output dimension override, if any. `None` means the
    /// model's native dimension. Used to isolate the on-disk cache directory.
    pub fn dimensions(&self) -> Option<u32> {
        self.inner.dimensions
    }

    /// Embed a single query string with bounded retry.
    ///
    /// Uses `input_type: "query"`. On 429 from all keys, waits 2 s and retries
    /// once. A second 429 wave returns `Err`. Transient/non-429 errors return
    /// `Err` immediately (query path is user-facing with tight timeout).
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
                Err(EmbedError::Transient(e)) => return Err(e.context("VoyageAI transient error on query embed")),
                Err(EmbedError::Other(e)) => return Err(e),
            }
        }

        // All keys 429 — one backoff attempt (2 s with jitter), then return Err.
        let cursor_val = self.inner.key_cursor.load(Ordering::Relaxed);
        let delay = backoff_with_jitter(2, cursor_val);
        warn!(delay_ms = delay.as_millis() as u64, "all VoyageAI keys rate-limited on query embed; backing off");
        tokio::time::sleep(delay).await;

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
                Err(EmbedError::Transient(e)) => return Err(e.context("VoyageAI transient error on query embed")),
                Err(EmbedError::Other(e)) => return Err(e),
            }
        }

        anyhow::bail!("VoyageAI query embed still rate-limited after backoff")
    }

    /// Embed texts in batches respecting both count (128) and byte-size limits
    /// per request. Returns one Vec<f32> per input.
    pub async fn embed(&self, texts: &[String], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());
        for batch in byte_aware_batches(texts) {
            let embeddings = self.embed_batch(batch, input_type).await?;
            all_embeddings.extend(embeddings);
        }
        Ok(all_embeddings)
    }

    /// Embed a batch of up to `MAX_BATCH_SIZE` texts. Public so the pipeline can
    /// drive batching manually and report per-batch progress between awaits.
    ///
    /// Retry policy:
    /// - 429 (rate-limited): infinite retry with exponential backoff + jitter
    ///   (rate limits clear eventually).
    /// - Transient (timeout/connect): bounded retry (TRANSIENT_RETRY_LIMIT attempts)
    ///   then propagate error — the file is left un-embedded and will be retried on
    ///   next index trigger via file_meta crash-safety.
    /// - Other errors: abort immediately.
    pub async fn embed_batch(&self, texts: &[String], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let n_keys = self.inner.api_keys.len();
        let mut transient_attempts: usize = 0;

        loop {
            let start_cursor = self.inner.key_cursor.fetch_add(1, Ordering::Relaxed) % n_keys;

            // Try each key once before falling back to backoff.
            for offset in 0..n_keys {
                let key_idx = (start_cursor + offset) % n_keys;
                let key = &self.inner.api_keys[key_idx];

                match self.try_embed_with_key(key, texts, input_type).await {
                    Ok(embeddings) => return Ok(embeddings),
                    Err(EmbedError::RateLimited) => {
                        warn!(key_index = key_idx, "VoyageAI 429 — trying next key");
                    }
                    Err(EmbedError::Transient(e)) => {
                        transient_attempts += 1;
                        if transient_attempts >= TRANSIENT_RETRY_LIMIT {
                            // Wrap with TransientEmbedExhausted so callers can
                            // distinguish "transient/retry-exhausted" from a fatal
                            // config/auth error via downcast_ref on the error chain.
                            // This lets the pipeline skip one file without aborting
                            // the whole run.
                            return Err(e.context(TransientEmbedExhausted {
                                attempts: transient_attempts,
                            }));
                        }
                        let cursor_val = self.inner.key_cursor.load(Ordering::Relaxed);
                        let delay = backoff_with_jitter(
                            2u64.pow(transient_attempts as u32).min(16),
                            cursor_val,
                        );
                        warn!(
                            attempt = transient_attempts,
                            max_attempts = TRANSIENT_RETRY_LIMIT,
                            delay_ms = delay.as_millis() as u64,
                            error = %e,
                            "VoyageAI transient error — retrying after backoff"
                        );
                        tokio::time::sleep(delay).await;
                        // Break inner key loop to retry from scratch with fresh cursor.
                        break;
                    }
                    // Non-transient, non-429 error: abort immediately, old data untouched.
                    Err(EmbedError::Other(e)) => return Err(e),
                }
            }

            // If we exhausted all keys without a transient break, they were all 429.
            // Exponential backoff with jitter, retry indefinitely (rate limits clear).
            if transient_attempts == 0 || transient_attempts >= TRANSIENT_RETRY_LIMIT {
                // Only enter 429 backoff if the inner loop completed (no transient break).
                let mut delay_secs: u64 = 2;
                loop {
                    let cursor_val = self.inner.key_cursor.load(Ordering::Relaxed);
                    let delay = backoff_with_jitter(delay_secs, cursor_val);
                    warn!(
                        delay_ms = delay.as_millis() as u64,
                        "all VoyageAI keys rate-limited; backing off with jitter"
                    );
                    tokio::time::sleep(delay).await;

                    let retry_cursor = self.inner.key_cursor.fetch_add(1, Ordering::Relaxed) % n_keys;
                    for offset in 0..n_keys {
                        let key_idx = (retry_cursor + offset) % n_keys;
                        let key = &self.inner.api_keys[key_idx];
                        match self.try_embed_with_key(key, texts, input_type).await {
                            Ok(embeddings) => {
                                info!("VoyageAI embed succeeded after backoff");
                                return Ok(embeddings);
                            }
                            Err(EmbedError::RateLimited) => continue,
                            Err(EmbedError::Transient(e)) => {
                                transient_attempts += 1;
                                if transient_attempts >= TRANSIENT_RETRY_LIMIT {
                                    return Err(e.context(TransientEmbedExhausted {
                                        attempts: transient_attempts,
                                    }));
                                }
                                // Break out of the 429 loop to retry via outer loop.
                                break;
                            }
                            Err(EmbedError::Other(e)) => return Err(e),
                        }
                    }

                    delay_secs = (delay_secs * 2).min(60);
                }
            }
            // If we broke out of inner loop due to transient, the outer loop retries.
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
        // Per-provider request body. Voyage carries `input_type`; OpenAI rejects
        // unknown fields, so it omits `input_type` and instead carries an optional
        // `dimensions`. Both are `Option` with skip_serializing_if so each provider
        // emits exactly its own field set.
        let (input_type_field, dimensions_field) = match self.inner.provider {
            Provider::Voyage => (Some(input_type.as_str()), None),
            Provider::OpenAI => (None, self.inner.dimensions),
        };
        let body = EmbedRequest {
            model: &self.inner.model,
            input: texts,
            input_type: input_type_field,
            dimensions: dimensions_field,
        };

        let response = client
            .post(&self.inner.endpoint)
            .bearer_auth(key)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                // Any error from .send() means no valid HTTP response was received —
                // this is a transport-layer failure (timeout, connection refused,
                // mid-flight connection reset/os error 10054, DNS, TLS handshake, etc.)
                // and is inherently transient. Only is_builder() (a programming error
                // constructing the request) is truly fatal here.
                //
                // Incident: a Linux kernel 79K-file rebuild got to 86% then ABORTED
                // because a mid-flight connection reset (reqwest SendRequest / os error
                // 10054) was not covered by the old `is_timeout() || is_connect()` check.
                // The correct rule: no HTTP response arrived → Transient. HTTP error
                // statuses (401, 400, 5xx) are handled separately below and remain fatal.
                if is_send_error_transient(&e) {
                    EmbedError::Transient(e.into())
                } else {
                    EmbedError::Other(e.into())
                }
            })?;

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
            .map_err(|e| {
                // At this point the HTTP status was verified as 2xx, so any error here
                // is a body-read/decode failure: stream interrupted mid-transfer, partial
                // JSON, connection reset during chunked encoding, decode timeout, etc.
                // ALL of these are transient — the server accepted the request and began
                // a valid response, but transport broke before we got the full body.
                // Retrying will get the embeddings fresh.
                //
                // History: a body-read timeout and a mid-stream connection reset both
                // aborted a 79K-file Linux kernel rebuild. The correct rule: if no
                // complete, parseable response body was received from a 2xx response,
                // it's transient. Fatal errors are exclusively HTTP error statuses
                // (handled above) where the server explicitly rejected the request.
                EmbedError::Transient(e.into())
            })?;

        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }
}

/// Split texts into sub-slices where each batch has at most `MAX_BATCH_SIZE`
/// texts AND the sum of `text.len()` stays under `MAX_BATCH_BYTES`. A single
/// text exceeding the byte cap is sent alone (VoyageAI will truncate or reject
/// at the token level, but it won't poison the whole batch).
fn byte_aware_batches(texts: &[String]) -> Vec<&[String]> {
    let mut batches = Vec::new();
    let mut start = 0;
    while start < texts.len() {
        let mut end = start;
        let mut batch_bytes = 0usize;
        while end < texts.len()
            && end - start < MAX_BATCH_SIZE
            && (batch_bytes + texts[end].len() <= MAX_BATCH_BYTES || end == start)
        {
            batch_bytes += texts[end].len();
            end += 1;
        }
        batches.push(&texts[start..end]);
        start = end;
    }
    batches
}

/// Classify whether a `reqwest::Error` from the `.send()` phase is transient.
///
/// The rule is simple: if no valid HTTP response was received, the failure is
/// transport-level and inherently transient (timeout, connection refused,
/// mid-flight reset/os error 10054, DNS resolution, TLS handshake error, proxy
/// error, etc.). The ONLY non-transient send error is `is_builder()`, which
/// indicates a programming mistake constructing the request (invalid URL, bad
/// header value) — that will never self-heal on retry.
///
/// This function is factored out as a pure predicate so it can be unit-tested
/// without requiring a live socket or specific OS error. The downstream retry
/// loop (embed_batch) retries Transient up to TRANSIENT_RETRY_LIMIT times with
/// exponential backoff; if exhausted, the file is skipped (not the whole run).
#[inline]
fn is_send_error_transient(e: &reqwest::Error) -> bool {
    // is_builder() = config/programming error building the Request struct itself.
    // Everything else from .send() is a transport failure: network unreachable,
    // connection refused, connection reset (10054), timeout, DNS failure,
    // TLS error, proxy errors, redirect loops, etc.
    !e.is_builder()
}

enum EmbedError {
    RateLimited,
    /// Transient network error (timeout, connection refused/reset, mid-flight
    /// connection close, body-read failure) — retryable with bounded attempts.
    Transient(anyhow::Error),
    Other(anyhow::Error),
}

/// Marker error type wrapped around a transient embed failure that exhausted
/// all retry attempts. Carried inside the `anyhow::Error` chain so callers
/// can distinguish "transient/exhausted-retry" from "fatal/config" errors via
/// `err.downcast_ref::<TransientEmbedExhausted>()`.
///
/// A transient-exhausted failure for a single file is NON-FATAL to the pipeline:
/// crash-safe `file_meta` means the file is simply not committed and will be
/// retried on the next index trigger (self-healing). This distinction prevents
/// a single gateway timeout from aborting an entire 79K-file Linux kernel rebuild.
#[derive(Debug)]
pub struct TransientEmbedExhausted {
    pub attempts: usize,
}

impl std::fmt::Display for TransientEmbedExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "transient embed error exhausted after {} attempts",
            self.attempts
        )
    }
}

impl std::error::Error for TransientEmbedExhausted {}

/// Maximum number of retry attempts for transient network errors (timeout,
/// connection failures). After this many attempts, the error propagates and
/// the file is left un-embedded (resumable via file_meta on next trigger).
///
/// Set to 6 to ride out multi-second gateway blips on large-repo runs (e.g.
/// Linux kernel 79K files through a shared gateway). With exponential backoff
/// capped at 16s, 6 attempts ≈ 2+4+8+16+16 = 46s worst-case per file — long
/// enough for most transient outages without stalling the pipeline indefinitely.
const TRANSIENT_RETRY_LIMIT: usize = 6;

/// Test-visible alias for TRANSIENT_RETRY_LIMIT so pipeline tests can assert
/// the value without duplicating the constant.
#[cfg(test)]
pub const TRANSIENT_RETRY_LIMIT_FOR_TEST: usize = TRANSIENT_RETRY_LIMIT;

/// Compute a backoff duration with jitter for retry loops. Uses a lightweight
/// deterministic jitter derived from the atomic key cursor to de-correlate
/// concurrent embed tasks without pulling in the `rand` crate.
///
/// `base_secs`: the base delay (doubles each iteration in the caller).
/// `cursor_val`: a monotonically increasing value (e.g. key_cursor) used as
/// entropy source for jitter.
///
/// Returns `base_secs + jitter` where jitter is 0..base_secs/4.
fn backoff_with_jitter(base_secs: u64, cursor_val: usize) -> Duration {
    // Use low bits of the cursor + current time nanos to derive cheap jitter.
    // This is NOT cryptographic — it just needs to de-correlate concurrent tasks.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0) as u64;
    let entropy = (cursor_val as u64).wrapping_add(nanos);
    // Jitter range: 0 .. base_secs/4 seconds (in millis for finer granularity).
    let max_jitter_ms = (base_secs * 250).max(100); // at least 100ms jitter
    let jitter_ms = entropy % max_jitter_ms;
    Duration::from_secs(base_secs) + Duration::from_millis(jitter_ms)
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

    // ── OpenAI provider endpoint resolution ─────────────────────────────────

    #[test]
    fn openai_url_default_when_none_or_blank() {
        assert_eq!(embedding_url(Provider::OpenAI, None), OPENAI_ENDPOINT);
        assert_eq!(embedding_url(Provider::OpenAI, Some("")), OPENAI_ENDPOINT);
        assert_eq!(embedding_url(Provider::OpenAI, Some("   ")), OPENAI_ENDPOINT);
    }

    #[test]
    fn voyage_url_default_via_embedding_url() {
        assert_eq!(embedding_url(Provider::Voyage, None), VOYAGE_ENDPOINT);
    }

    #[test]
    fn openai_url_appends_and_normalizes_base() {
        // Non-blank base → identical normalization across providers (only the
        // blank-fallback default differs).
        assert_eq!(
            embedding_url(Provider::OpenAI, Some("https://gateway.local/v1")),
            "https://gateway.local/v1/embeddings"
        );
        assert_eq!(
            embedding_url(Provider::OpenAI, Some("https://gateway.local/v1/")),
            "https://gateway.local/v1/embeddings"
        );
        assert_eq!(
            embedding_url(Provider::OpenAI, Some("https://gateway.local/v1/embeddings")),
            "https://gateway.local/v1/embeddings"
        );
    }

    #[test]
    fn provider_from_str_parses_case_insensitively() {
        assert_eq!(Provider::parse("openai"), Provider::OpenAI);
        assert_eq!(Provider::parse("OpenAI"), Provider::OpenAI);
        assert_eq!(Provider::parse(" openai "), Provider::OpenAI);
        assert_eq!(Provider::parse("voyage"), Provider::Voyage);
        // Unknown values fall back to Voyage (historical default).
        assert_eq!(Provider::parse("something-else"), Provider::Voyage);
        assert_eq!(Provider::parse(""), Provider::Voyage);
    }

    // ── Per-provider request body field set ─────────────────────────────────

    fn serialize_body(provider: Provider, input_type: InputType, dimensions: Option<u32>) -> serde_json::Value {
        let texts = vec!["hi".to_string()];
        let (input_type_field, dimensions_field) = match provider {
            Provider::Voyage => (Some(input_type.as_str()), None),
            Provider::OpenAI => (None, dimensions),
        };
        let body = EmbedRequest {
            model: "m",
            input: &texts,
            input_type: input_type_field,
            dimensions: dimensions_field,
        };
        serde_json::to_value(&body).expect("serialize body")
    }

    #[test]
    fn voyage_body_includes_input_type_omits_dimensions() {
        let v = serialize_body(Provider::Voyage, InputType::Document, Some(512));
        let obj = v.as_object().expect("object");
        assert_eq!(obj.get("input_type").and_then(|x| x.as_str()), Some("document"));
        assert!(!obj.contains_key("dimensions"), "Voyage must never send dimensions");
        // Exact field set: model, input, input_type.
        assert_eq!(obj.len(), 3, "Voyage body fields: {:?}", obj.keys().collect::<Vec<_>>());
    }

    #[test]
    fn openai_body_omits_input_type_and_dimensions_when_unset() {
        let v = serialize_body(Provider::OpenAI, InputType::Document, None);
        let obj = v.as_object().expect("object");
        assert!(!obj.contains_key("input_type"), "OpenAI must omit input_type");
        assert!(!obj.contains_key("dimensions"), "OpenAI must omit dimensions when unset");
        // Exact field set: model, input.
        assert_eq!(obj.len(), 2, "OpenAI body fields: {:?}", obj.keys().collect::<Vec<_>>());
    }

    #[test]
    fn openai_body_includes_dimensions_when_set() {
        let v = serialize_body(Provider::OpenAI, InputType::Query, Some(256));
        let obj = v.as_object().expect("object");
        assert!(!obj.contains_key("input_type"), "OpenAI must omit input_type");
        assert_eq!(obj.get("dimensions").and_then(|x| x.as_u64()), Some(256));
        // Exact field set: model, input, dimensions.
        assert_eq!(obj.len(), 3, "OpenAI body fields: {:?}", obj.keys().collect::<Vec<_>>());
    }

    #[test]
    fn byte_aware_batches_splits_by_size() {
        // 600 KB each → only 2 fit in 1.5 MB cap (1.2 MB < 1.5 MB, 1.8 MB > 1.5 MB)
        let texts: Vec<String> = (0..5).map(|_| "x".repeat(600_000)).collect();
        let batches = byte_aware_batches(&texts);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 2);
        assert_eq!(batches[2].len(), 1);
    }

    #[test]
    fn byte_aware_batches_respects_count_limit() {
        let texts: Vec<String> = (0..200).map(|_| "short".to_string()).collect();
        let batches = byte_aware_batches(&texts);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 128);
        assert_eq!(batches[1].len(), 72);
    }

    #[test]
    fn byte_aware_batches_oversized_single_text_sent_alone() {
        let texts: Vec<String> = vec!["x".repeat(3_000_000), "small".to_string()];
        let batches = byte_aware_batches(&texts);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 1);
        assert_eq!(batches[1].len(), 1);
    }

    #[test]
    fn backoff_with_jitter_produces_bounded_duration() {
        // backoff_with_jitter(base=2, cursor) should produce 2s + [0, 500ms)
        for cursor in 0..100 {
            let d = backoff_with_jitter(2, cursor);
            assert!(d >= Duration::from_secs(2), "duration should be >= base");
            // Max jitter for base=2: 2*250 = 500ms
            assert!(
                d < Duration::from_secs(2) + Duration::from_millis(500),
                "duration should be < base + max_jitter; got {:?}",
                d
            );
        }
    }

    #[test]
    fn backoff_with_jitter_has_minimum_jitter_range() {
        // When base_secs=0, max_jitter_ms should be at least 100.
        let d = backoff_with_jitter(0, 42);
        // Duration should be [0ms, 100ms)
        assert!(d < Duration::from_millis(100));
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn transient_retry_limit_is_bounded() {
        // Unit test: verify the constant is a reasonable small number.
        assert!(
            TRANSIENT_RETRY_LIMIT >= 2 && TRANSIENT_RETRY_LIMIT <= 10,
            "TRANSIENT_RETRY_LIMIT should be a small bounded number, got {}",
            TRANSIENT_RETRY_LIMIT
        );
    }

    /// Test that transient errors from reqwest are classified correctly.
    /// With the broadened rule (!is_builder → Transient), ANY transport failure
    /// from .send() must land in the Transient bucket — timeout, connection
    /// refused, connection reset, DNS failure, etc.
    #[tokio::test]
    async fn transient_error_classification() {
        // Build a client with 1ms timeout — any request will timeout.
        let client = Client::builder()
            .timeout(Duration::from_millis(1))
            .build()
            .unwrap();

        // Create a VoyageClient pointing to localhost (no server running).
        let vc = VoyageClient::new(
            "test-model".to_string(),
            vec!["fake-key".to_string()],
            Some("http://127.0.0.1:1"), // port 1 — will timeout or connection-refuse
        )
        .unwrap();

        // Call the internal method directly to check classification.
        let texts = vec!["test".to_string()];
        let result = vc
            .try_embed_with_key_using(&client, "fake-key", &texts, InputType::Document)
            .await;

        match result {
            Err(EmbedError::Transient(_)) => {
                // Correct: any transport error from .send() is transient.
            }
            Err(EmbedError::Other(e)) => {
                panic!(
                    "transport error from .send() should be Transient, not Other: {}",
                    e
                );
            }
            Err(EmbedError::RateLimited) => {
                panic!("should not get RateLimited from a transport error");
            }
            Ok(_) => {
                panic!("should not succeed connecting to port 1");
            }
        }
    }

    /// Test the `is_send_error_transient` classification predicate directly.
    ///
    /// We construct reqwest errors via actual failed requests rather than trying
    /// to construct them directly (reqwest::Error has no public constructor).
    /// - Timeout error (1ms timeout to port 1) → transient
    /// - Connection refused (port 1) → transient
    /// Both must be `!is_builder()` → transient under our rule.
    ///
    /// This test validates that a connection-reset style error (which is
    /// `is_request()` in reqwest — the same category as OS error 10054) would
    /// also classify as transient, because `is_request()` implies `!is_builder()`.
    #[tokio::test]
    async fn is_send_error_transient_predicate() {
        // Test 1: Timeout error — is_timeout() && !is_builder() → transient
        let timeout_client = Client::builder()
            .timeout(Duration::from_millis(1))
            .build()
            .unwrap();
        let timeout_err = timeout_client
            .get("http://127.0.0.1:1/embeddings")
            .send()
            .await
            .unwrap_err();
        assert!(
            is_send_error_transient(&timeout_err),
            "timeout error must be transient: {:?}",
            timeout_err
        );
        // Verify it's truly a timeout or at least not a builder error
        assert!(!timeout_err.is_builder());

        // Test 2: Connection-refused error (no timeout, port 1 rejects)
        let connect_client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        let connect_err = connect_client
            .get("http://127.0.0.1:1/embeddings")
            .send()
            .await
            .unwrap_err();
        assert!(
            is_send_error_transient(&connect_err),
            "connection-refused error must be transient: {:?}",
            connect_err
        );
        assert!(!connect_err.is_builder());

        // Verify the logical invariant: for any reqwest error from .send(),
        // is_builder() is the ONLY case that is NOT transient under our rule.
        // Both errors above should NOT be builder errors:
        assert!(
            !timeout_err.is_builder() && !connect_err.is_builder(),
            "send errors from actual network attempts are never builder errors"
        );
    }

    /// Verify that the is_send_error_transient predicate would classify a
    /// connection-reset (is_request) error as transient. We can't easily
    /// synthesize OS error 10054 in a unit test, but we can verify the logical
    /// contract: is_request() implies !is_builder(), which implies transient.
    ///
    /// reqwest categorizes errors into: is_builder, is_redirect, is_status,
    /// is_timeout, is_connect, is_request, is_body, is_decode.
    /// For .send() failures: is_timeout, is_connect, and is_request are the
    /// three transport categories. None of these are is_builder, so all three
    /// pass our `!is_builder()` check → classified transient.
    #[tokio::test]
    async fn connection_reset_is_transient_by_contract() {
        // Get any .send() error to verify the contract
        let client = Client::builder()
            .timeout(Duration::from_millis(1))
            .build()
            .unwrap();
        let err = client
            .get("http://127.0.0.1:1/embeddings")
            .send()
            .await
            .unwrap_err();

        // Key assertions about the classification contract:
        // 1. The error is not a builder error (it got past request construction)
        assert!(!err.is_builder());
        // 2. Therefore it's transient under our rule
        assert!(is_send_error_transient(&err));
        // 3. The error IS one of the transport categories
        assert!(
            err.is_timeout() || err.is_connect() || err.is_request(),
            "send error should be timeout/connect/request, got: {:?}",
            err
        );
        // 4. Verify inverse: is_builder would be fatal
        // (We can't construct a builder error without invalid input, but we
        // verify the predicate logic: is_builder → !transient)
        // This is guaranteed by the implementation: !e.is_builder()
    }
}
