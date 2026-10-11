use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::embedding::InputType;

use super::retry::{EmbedError, is_send_error_transient};
use super::{Provider, VoyageClient};

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
pub(super) struct EmbedData {
    pub(super) embedding: Vec<f32>,
    /// Position of the input this vector belongs to. Both Voyage and OpenAI send
    /// it; absent → the list order is trusted.
    #[serde(default)]
    pub(super) index: Option<usize>,
}

/// Put a response's vectors back into input order and refuse anything that does
/// not map one-to-one onto the `n_inputs` inputs.
///
/// WHY: the vectors used to be taken in list order with no count check. A short
/// reply left the trailing chunks with empty embeddings — the pipeline padded
/// them, committed file_meta, and `VectorIndex::insert` silently skipped them, so
/// those chunks were permanently unsearchable — and a reordered reply attached
/// vectors to the wrong chunks. Treated as transient (retried like a decode
/// error); a persistent mismatch ends up skipped and retried later rather than
/// stored wrong.
pub(super) fn order_embeddings(
    mut data: Vec<EmbedData>,
    n_inputs: usize,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    if data.len() != n_inputs {
        return Err(EmbedError::Transient(anyhow::anyhow!(
            "embedding response has {} vectors for {} inputs",
            data.len(),
            n_inputs
        )));
    }
    if data.iter().all(|d| d.index.is_some()) {
        data.sort_by_key(|d| d.index);
        if data.iter().enumerate().any(|(pos, d)| d.index != Some(pos)) {
            return Err(EmbedError::Transient(anyhow::anyhow!(
                "embedding response indexes are not a permutation of 0..{n_inputs}"
            )));
        }
    }
    if data.iter().any(|d| d.embedding.is_empty()) {
        return Err(EmbedError::Transient(anyhow::anyhow!(
            "embedding response contains an empty vector"
        )));
    }
    Ok(data.into_iter().map(|d| d.embedding).collect())
}

impl VoyageClient {
    pub(super) async fn try_embed_with_key(
        &self,
        key: &str,
        texts: &[String],
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.try_embed_with_key_using(&self.inner.http, key, texts, input_type)
            .await
    }

    pub(super) async fn try_embed_query_with_key(
        &self,
        key: &str,
        texts: &[String],
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.try_embed_with_key_using(&self.inner.query_http, key, texts, input_type)
            .await
    }

    pub(super) async fn try_embed_with_key_using(
        &self,
        client: &Client,
        key: &str,
        texts: &[String],
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let (input_type, dimensions) = match self.inner.provider {
            Provider::Voyage => (Some(input_type.as_str()), None),
            Provider::OpenAI => (None, self.inner.dimensions),
        };
        let body = EmbedRequest {
            model: &self.inner.model,
            input: texts,
            input_type,
            dimensions,
        };
        let response = client
            .post(&self.inner.endpoint)
            .bearer_auth(key)
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                if is_send_error_transient(&error) {
                    EmbedError::Transient(error.into())
                } else {
                    EmbedError::Other(error.into())
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
        let response: EmbedResponse = response
            .json()
            .await
            .map_err(|error| EmbedError::Transient(error.into()))?;
        order_embeddings(response.data, texts.len())
    }
}

#[cfg(test)]
pub(super) fn request_json(
    provider: Provider,
    input_type: InputType,
    dimensions: Option<u32>,
) -> serde_json::Value {
    let texts = vec!["hi".to_string()];
    let (input_type, dimensions) = match provider {
        Provider::Voyage => (Some(input_type.as_str()), None),
        Provider::OpenAI => (None, dimensions),
    };
    serde_json::to_value(EmbedRequest {
        model: "m",
        input: &texts,
        input_type,
        dimensions,
    })
    .expect("serialize body")
}
