use std::sync::atomic::Ordering;

use axum::http::StatusCode;

use super::fixture::rejecting_client;

#[tokio::test]
async fn query_does_not_retry_auth_or_payment_rejections() {
    for status in [
        StatusCode::UNAUTHORIZED,
        StatusCode::PAYMENT_REQUIRED,
        StatusCode::FORBIDDEN,
    ] {
        let (client, requests) = rejecting_client(status).await;

        let error = client.embed_query("query").await.unwrap_err();
        assert!(error.to_string().contains(status.as_str()));
        assert_eq!(requests.load(Ordering::SeqCst), 1, "{status} was retried");
    }
}

/// S11/D16: a short response, an empty vector, or a non-permutation of indexes
/// is an error — never silently padded with empty vectors (which file_meta then
/// committed as if the file were indexed) — and a reordered response is put back
/// in input order. Pure validator: over HTTP the transient-retry backoff would
/// make this test sit for ~45 s.
#[test]
fn malformed_embedding_responses_are_rejected_and_order_restored() {
    use super::super::http_transport::{EmbedData, order_embeddings};
    let d = |index: Option<usize>, v: Vec<f32>| EmbedData {
        embedding: v,
        index,
    };
    assert!(order_embeddings(vec![d(Some(0), vec![1.0]), d(Some(1), vec![2.0])], 3).is_err());
    assert!(order_embeddings(vec![d(Some(0), vec![1.0]), d(Some(1), vec![])], 2).is_err());
    assert!(order_embeddings(vec![d(Some(0), vec![1.0]), d(Some(0), vec![2.0])], 2).is_err());
    let ok = order_embeddings(vec![d(Some(1), vec![2.0]), d(Some(0), vec![1.0])], 2)
        .ok()
        .unwrap();
    assert_eq!(ok, vec![vec![1.0], vec![2.0]]);
    let no_index = order_embeddings(vec![d(None, vec![1.0]), d(None, vec![2.0])], 2)
        .ok()
        .unwrap();
    assert_eq!(
        no_index,
        vec![vec![1.0], vec![2.0]],
        "list order kept without indexes"
    );
}
