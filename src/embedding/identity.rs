//! Embedding identity — the single source of truth describing which embedding
//! model/space a set of stored vectors belongs to.
//!
//! ## Why this exists
//! The vector store is a cache of embeddings produced by ONE embedding model.
//! If a version upgrade changes the embedding model (or its output dimension or
//! provider) while keeping the same vector length, cosine similarity between a
//! query embedded with the NEW model and vectors embedded with the OLD model is
//! mathematically meaningless — the store returns confident-but-wrong results.
//!
//! Nothing in the freshness path (`tracker::detect_changes` on mtime/size/
//! chunker_version, or `shard_file::open_current` on dim+stamp) knows which model
//! produced a vector, so a same-dimension model swap slips through undetected and
//! retrieval silently degrades until a manual rebuild.
//!
//! `EmbeddingIdentity` closes that hole: it is stamped into the repo DB
//! (`index_meta[EMBEDDING_IDENTITY_KEY]`) at index-commit time, mixed into the
//! persisted shard's `content_stamp`, and carried on every resident shard so the
//! query path can validate the space before searching.
//!
//! ## Source of truth
//! An identity is ALWAYS derived from the real embedding client used for the
//! operation (`from_client`) — the same `VoyageClient` that embeds a query is the
//! one whose identity describes that query's vector space; the same `self.voyage`
//! that re-embeds during an index run is the one whose identity is committed.
//! There is deliberately no config-derived constructor: a second normalization
//! path would risk diverging from the client actually used.

use crate::embedding::voyage::{Provider, VoyageClient};
use sha2::{Digest, Sha256};

/// `index_meta` key under which a repo's committed embedding identity is stored.
/// Written only after a full index run's data (chunks + edges) is durable.
pub const EMBEDDING_IDENTITY_KEY: &str = "embedding_identity";

/// Describes the embedding vector space: `(provider, model, dimensions)`.
///
/// Two stores are query-compatible iff their identities are equal. This is a
/// read-only projection of the three fields a [`VoyageClient`] stores verbatim
/// (`new_for_provider` applies no normalization), so `from_client` faithfully
/// captures the space of whatever client is passed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EmbeddingIdentity {
    provider: Provider,
    model: String,
    dimensions: Option<u32>,
}

impl EmbeddingIdentity {
    /// The ONLY constructor: snapshot the identity of a real embedding client.
    pub fn from_client(client: &VoyageClient) -> Self {
        Self {
            provider: client.provider(),
            model: client.model().to_owned(),
            dimensions: client.dimensions(),
        }
    }

    /// Canonical string form persisted to `index_meta` and stored on shards.
    /// Shape: `"{provider}|{model}|{dimensions}"` where `dimensions` is the
    /// number or the literal `native` when the model's native dimension is used.
    pub fn as_key_string(&self) -> String {
        format!(
            "{}|{}|{}",
            self.provider.as_str(),
            self.model,
            match self.dimensions {
                Some(d) => d.to_string(),
                None => "native".to_owned(),
            }
        )
    }

    /// Fold this identity and the durable index epoch into a chunk-count stamp so
    /// the persisted shard file's `content_stamp` changes whenever the identity
    /// changes, even at an unchanged chunk count — a second-line defense behind the
    /// DB identity check — and whenever ANY index run changed chunk rows since the
    /// file was written (`index_epoch`, see `store::ops::bump_index_epoch`).
    /// Count + identity alone let a same-count edit revalidate a stale file.
    /// Hashed rather than linearly mixed so distinct (count, epoch) pairs cannot
    /// cancel out to the same stamp.
    pub fn content_stamp(&self, chunk_count: u64, index_epoch: u64) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(self.as_key_string().as_bytes());
        hasher.update(chunk_count.to_le_bytes());
        hasher.update(index_epoch.to_le_bytes());
        let digest = hasher.finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[0..8]);
        u64::from_le_bytes(bytes)
    }
}
