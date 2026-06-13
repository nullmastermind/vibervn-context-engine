use anyhow::{Context, Result};
use rayon::prelude::*;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tracing::info;

use crate::path_in_repo;

pub mod sharded;
pub use sharded::{ShardedSearch, ShardedVectorIndex};

// ─── Public types ─────────────────────────────────────────────────────────

/// Identifies a chunk by its location in the source tree.
/// Used to map VectorIndex results back to SurrealDB rows.
#[derive(Debug, Clone)]
pub struct ChunkId {
    pub file: String,
    pub line_start: u32,
    pub line_end: u32,
}

/// A single result returned by [`VectorIndex::search`].
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk_id: ChunkId,
    /// Cosine similarity in [0, 1] (vectors are pre-normalized).
    pub score: f32,
}

// ─── VectorIndex ─────────────────────────────────────────────────────────

/// In-memory flat cosine-similarity index.
///
/// All vectors are L2-normalized at insert time, so cosine similarity reduces
/// to a plain dot product at query time (no division per candidate).
///
/// Storage is a single contiguous `Vec<f32>` in row-major order: row i occupies
/// `embeddings[i*dim .. (i+1)*dim]`. This gives better cache locality than
/// `Vec<Vec<f32>>` and enables rayon parallel iteration over fixed-size rows.
///
/// At 500 K chunks × 1024 dims, rayon + SIMD give sub-100 ms per query.
pub struct VectorIndex {
    /// Contiguous row-major flat storage. Length = len * dim.
    embeddings: Vec<f32>,
    /// Parallel array: chunk_ids[i] corresponds to row i.
    chunk_ids: Vec<ChunkId>,
    /// Dimensionality of every vector. `None` until the first insert.
    dim: usize,
}

impl VectorIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            embeddings: Vec::new(),
            chunk_ids: Vec::new(),
            dim: 0,
        }
    }

    /// Number of indexed vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.chunk_ids.len()
    }

    /// Returns `true` if the index contains no vectors.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.chunk_ids.is_empty()
    }

    /// Insert a batch of (ChunkId, embedding) pairs.
    ///
    /// Each embedding is L2-normalized before storage. Zero-length or
    /// zero-magnitude vectors are skipped (they carry no information).
    pub fn insert(&mut self, chunks: &[(ChunkId, Vec<f32>)]) {
        for (id, raw_emb) in chunks {
            if raw_emb.is_empty() {
                continue;
            }
            // Record dimension on first insert; skip mismatches.
            if self.dim == 0 {
                self.dim = raw_emb.len();
            } else if self.dim != raw_emb.len() {
                tracing::warn!(
                    expected = self.dim,
                    got = raw_emb.len(),
                    file = %id.file,
                    "embedding dimension mismatch — skipping chunk"
                );
                continue;
            }

            let normalized = l2_normalize(raw_emb);
            self.embeddings.extend_from_slice(&normalized);
            self.chunk_ids.push(id.clone());
        }
    }

    /// Remove all embeddings whose `file` field matches `file`.
    ///
    /// Uses swap-remove to avoid O(n) shifts; moves whole `dim`-sized rows.
    pub fn remove_file(&mut self, file: &str) {
        let mut i = 0;
        while i < self.chunk_ids.len() {
            if self.chunk_ids[i].file == file {
                self.chunk_ids.swap_remove(i);
                // Swap the flat rows: row i ↔ last row.
                let last = self.chunk_ids.len(); // after swap_remove, len is already decremented
                if i < last {
                    // Copy last row into slot i.
                    let src_start = last * self.dim;
                    let dst_start = i * self.dim;
                    // Safety: src and dst are non-overlapping (i < last).
                    let src: Vec<f32> = self.embeddings[src_start..src_start + self.dim].to_vec();
                    self.embeddings[dst_start..dst_start + self.dim].copy_from_slice(&src);
                }
                self.embeddings.truncate(last * self.dim);
                // Don't advance i — the swapped element now lives at i.
            } else {
                i += 1;
            }
        }
        if self.chunk_ids.is_empty() {
            self.dim = 0;
        }
    }

    /// Remove all embeddings belonging to a repo.
    ///
    /// Uses [`path_in_repo`] for boundary-safe matching. O(n) swap-remove pass.
    pub fn remove_repo(&mut self, repo: &str) {
        let mut i = 0;
        while i < self.chunk_ids.len() {
            if path_in_repo(&self.chunk_ids[i].file, repo) {
                self.chunk_ids.swap_remove(i);
                let last = self.chunk_ids.len();
                if i < last {
                    let src_start = last * self.dim;
                    let dst_start = i * self.dim;
                    let src: Vec<f32> = self.embeddings[src_start..src_start + self.dim].to_vec();
                    self.embeddings[dst_start..dst_start + self.dim].copy_from_slice(&src);
                }
                self.embeddings.truncate(last * self.dim);
            } else {
                i += 1;
            }
        }
        if self.chunk_ids.is_empty() {
            self.dim = 0;
        }
    }

    /// Merge another `VectorIndex` into this one, consuming it.
    ///
    /// Vectors from `other` are already normalized; they are moved directly into
    /// the flat storage. Dimension compatibility: if `self` has no dimension yet,
    /// adopts `other`'s. If both have a dimension and they differ, logs a warning
    /// and skips the merge.
    pub fn merge(&mut self, other: VectorIndex) {
        if other.is_empty() {
            return;
        }
        if self.dim == 0 {
            self.dim = other.dim;
        } else if self.dim != other.dim {
            tracing::warn!(
                self_dim = self.dim,
                other_dim = other.dim,
                "VectorIndex merge dimension mismatch — skipping"
            );
            return;
        }
        self.embeddings.extend_from_slice(&other.embeddings);
        self.chunk_ids.extend(other.chunk_ids);
    }

    /// Search for the top-k most similar chunks to `query`.
    ///
    /// `query` is normalized internally so the caller need not pre-normalize.
    /// Returns results sorted by descending score, capped at `top_k`.
    /// Uses rayon for parallel dot-product computation.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        if self.is_empty() || query.is_empty() || top_k == 0 || self.dim == 0 {
            return vec![];
        }

        let q_norm = l2_normalize(query);

        // Parallel dot product over rows. Each row is exactly `self.dim` floats.
        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .par_chunks(self.dim)
            .enumerate()
            .map(|(i, row)| (i, dot_product(&q_norm, row)))
            .collect();

        // Partial sort: bring the top-k largest scores to the front.
        let k = top_k.min(scored.len());
        scored.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        scored
            .into_iter()
            .map(|(i, score)| SearchResult {
                chunk_id: self.chunk_ids[i].clone(),
                score,
            })
            .collect()
    }

    /// Load all embeddings from SurrealDB on startup.
    ///
    /// Only loads rows that have a non-empty embedding vector.
    pub async fn load_from_db(db: &Surreal<Db>) -> Result<Self> {
        use serde::Deserialize;

        #[derive(Deserialize)]
        struct Row {
            file: String,
            line_start: i64,
            line_end: i64,
            // Dual-format reader: tolerates BOTH the old `array<float>` rows
            // (≤ schema v4) and the new packed `bytes` rows (≥ v5). This is the
            // keystone that decouples query correctness from migration
            // completion — a half-migrated DB loads every shard correctly,
            // mixing old and new rows, because both decode to Vec<f32> here.
            #[serde(deserialize_with = "crate::store::ops::de_embedding_dual")]
            embedding: Vec<f32>,
        }

        let t_select = std::time::Instant::now();
        let rows: Vec<Row> = db
            .query("SELECT file, line_start, line_end, embedding FROM chunk WHERE embedding IS NOT NONE")
            .await
            .context("load embeddings from chunk table")?
            .take(0)?;
        let select_ms = t_select.elapsed().as_millis() as u64;

        let t_decode = std::time::Instant::now();
        let mut index = VectorIndex::new();
        let pairs: Vec<(ChunkId, Vec<f32>)> = rows
            .into_iter()
            .map(|r| {
                (
                    ChunkId {
                        file: r.file,
                        line_start: r.line_start as u32,
                        line_end: r.line_end as u32,
                    },
                    r.embedding,
                )
            })
            .collect();
        let decode_ms = t_decode.elapsed().as_millis() as u64;

        let count = pairs.len();
        let t_insert = std::time::Instant::now();
        index.insert(&pairs);
        let insert_ms = t_insert.elapsed().as_millis() as u64;
        // PERF SUMMARY load_from_db: the cold-shard-warm cost a user's first query
        // blocks on. select_ms = DB scan of all chunk rows; decode_ms = bytes->Vec<f32>
        // deserialize (serde, single-threaded); insert_ms = L2-normalize + flat copy.
        info!(
            count, select_ms, decode_ms, insert_ms,
            total_ms = select_ms + decode_ms + insert_ms,
            "PERF SUMMARY load_from_db (cold shard warm breakdown)"
        );

        Ok(index)
    }

    /// Remove all entries from the index.
    pub fn clear(&mut self) {
        self.embeddings.clear();
        self.chunk_ids.clear();
        self.dim = 0;
    }

    /// Total bytes occupied by the stored vector data.
    ///
    /// This is the resident-cap accounting metric: it counts ONLY the flat f32
    /// embedding storage (`len * 4`), which is the dominant and predictable term
    /// and matches the contract's "total resident vector bytes" definition. The
    /// `chunk_ids` bookkeeping is comparatively negligible and intentionally
    /// excluded so the cap maps cleanly to embedding payload.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.embeddings.len() * std::mem::size_of::<f32>()
    }
}

impl Default for VectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Math helpers ─────────────────────────────────────────────────────────

/// Compute the dot product of two equal-length slices.
#[inline]
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Return a copy of `v` normalized to unit L2 length.
/// Returns `v` unchanged if its magnitude is zero (avoids NaN).
pub(crate) fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / mag).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a ChunkId for a given file and line range.
    fn chunk(file: &str, start: u32, end: u32) -> ChunkId {
        ChunkId {
            file: file.to_string(),
            line_start: start,
            line_end: end,
        }
    }

    /// Helper: create a simple non-zero embedding of the given dimension.
    fn emb(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| seed + i as f32 * 0.1).collect()
    }

    #[test]
    fn remove_repo_no_prefix_collision() {
        let mut index = VectorIndex::new();

        let foo_chunks: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk("/foo/a.rs", 1, 10), emb(4, 1.0)),
            (chunk("/foo/b.rs", 1, 5), emb(4, 2.0)),
        ];
        let foobar_chunks: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk("/foobar/c.rs", 1, 20), emb(4, 3.0)),
            (chunk("/foobar/d.rs", 5, 15), emb(4, 4.0)),
        ];

        index.insert(&foo_chunks);
        index.insert(&foobar_chunks);
        assert_eq!(index.len(), 4);

        index.remove_repo("/foo");
        assert_eq!(index.len(), 2);

        for cid in &index.chunk_ids {
            assert!(
                cid.file.starts_with("/foobar/"),
                "unexpected file after remove_repo: {}",
                cid.file
            );
        }
    }

    #[test]
    fn remove_repo_windows_paths_no_collision() {
        let mut index = VectorIndex::new();

        let foo_chunks: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk(r"D:\proj\foo\x.rs", 1, 10), emb(4, 1.0)),
        ];
        let foobar_chunks: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk(r"D:\proj\foobar\y.rs", 1, 10), emb(4, 2.0)),
        ];

        index.insert(&foo_chunks);
        index.insert(&foobar_chunks);
        assert_eq!(index.len(), 2);

        index.remove_repo(r"D:\proj\foo");
        assert_eq!(index.len(), 1);
        assert_eq!(index.chunk_ids[0].file, r"D:\proj\foobar\y.rs");
    }

    #[test]
    fn full_rebuild_one_repo_preserves_other() {
        let mut index = VectorIndex::new();

        let repo_a_old: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk("/repo_a/old1.rs", 1, 10), emb(4, 1.0)),
            (chunk("/repo_a/old2.rs", 5, 20), emb(4, 2.0)),
        ];
        let repo_b: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk("/repo_b/file1.rs", 1, 5), emb(4, 3.0)),
            (chunk("/repo_b/file2.rs", 10, 30), emb(4, 4.0)),
            (chunk("/repo_b/file3.rs", 1, 100), emb(4, 5.0)),
        ];

        index.insert(&repo_a_old);
        index.insert(&repo_b);
        assert_eq!(index.len(), 5);

        index.remove_repo("/repo_a");
        assert_eq!(index.len(), 3);

        let repo_a_new: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk("/repo_a/new1.rs", 1, 15), emb(4, 6.0)),
            (chunk("/repo_a/new2.rs", 1, 8), emb(4, 7.0)),
            (chunk("/repo_a/new3.rs", 1, 50), emb(4, 8.0)),
        ];
        index.insert(&repo_a_new);
        assert_eq!(index.len(), 6);

        let b_files: Vec<&str> = index
            .chunk_ids
            .iter()
            .filter(|c| path_in_repo(&c.file, "/repo_b"))
            .map(|c| c.file.as_str())
            .collect();
        assert_eq!(b_files.len(), 3);
        assert!(b_files.contains(&"/repo_b/file1.rs"));
        assert!(b_files.contains(&"/repo_b/file2.rs"));
        assert!(b_files.contains(&"/repo_b/file3.rs"));

        let a_files: Vec<&str> = index
            .chunk_ids
            .iter()
            .filter(|c| path_in_repo(&c.file, "/repo_a"))
            .map(|c| c.file.as_str())
            .collect();
        assert_eq!(a_files.len(), 3);
        assert!(a_files.contains(&"/repo_a/new1.rs"));
        assert!(a_files.contains(&"/repo_a/new2.rs"));
        assert!(a_files.contains(&"/repo_a/new3.rs"));
    }

    #[test]
    fn merge_combines_two_indexes() {
        let mut a = VectorIndex::new();
        a.insert(&[(chunk("/a/f.rs", 1, 10), emb(4, 1.0))]);

        let mut b = VectorIndex::new();
        b.insert(&[(chunk("/b/g.rs", 1, 5), emb(4, 2.0))]);

        a.merge(b);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn merge_dimension_mismatch_skips() {
        let mut a = VectorIndex::new();
        a.insert(&[(chunk("/a/f.rs", 1, 10), emb(4, 1.0))]);

        let mut b = VectorIndex::new();
        b.insert(&[(chunk("/b/g.rs", 1, 5), emb(8, 2.0))]);

        a.merge(b);
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn remove_repo_with_trailing_sep() {
        let mut index = VectorIndex::new();
        index.insert(&[(chunk("/repo/file.rs", 1, 10), emb(4, 1.0))]);
        assert_eq!(index.len(), 1);

        index.remove_repo("/repo/");
        assert_eq!(index.len(), 0);
    }

    /// ❻ NEW: flat storage parity — search results must be identical to a
    /// reference dot-product computation on the same normalized inputs.
    #[test]
    fn flat_search_parity_with_reference() {
        let dim = 8;
        let pairs: Vec<(ChunkId, Vec<f32>)> = (0..10_usize)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|j| i as f32 * 0.1 + j as f32 * 0.01).collect();
                (chunk(&format!("/f{i}.rs", ), 1, 10), v)
            })
            .collect();

        let mut index = VectorIndex::new();
        index.insert(&pairs);

        let query: Vec<f32> = (0..dim).map(|j| 0.5 + j as f32 * 0.05).collect();
        let results = index.search(&query, 5);

        // Reference: manually normalize all inserted vecs + query, dot product.
        let q_norm = l2_normalize(&query);
        let mut ref_scored: Vec<(usize, f32)> = pairs
            .iter()
            .enumerate()
            .map(|(i, (_, v))| {
                let n = l2_normalize(v);
                (i, dot_product(&q_norm, &n))
            })
            .collect();
        ref_scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ref_scored.truncate(5);

        assert_eq!(results.len(), ref_scored.len());
        for (r, (ref_i, ref_score)) in results.iter().zip(ref_scored.iter()) {
            assert_eq!(
                r.chunk_id.file,
                format!("/f{ref_i}.rs"),
                "chunk_id order must match reference"
            );
            let diff = (r.score - ref_score).abs();
            assert!(diff < 1e-5, "score diff {diff} exceeds tolerance");
        }
    }

    /// ❻ NEW: remove_file with flat storage — after removal, remaining rows must
    /// give the same search results as if the removed file was never inserted.
    #[test]
    fn remove_file_flat_consistency() {
        let dim = 4;
        let mut index = VectorIndex::new();
        let c1 = (chunk("/a/f1.rs", 1, 10), emb(dim, 1.0));
        let c2 = (chunk("/a/f2.rs", 1, 10), emb(dim, 2.0));
        let c3 = (chunk("/b/f3.rs", 1, 10), emb(dim, 3.0));
        index.insert(&[c1, c2, c3]);
        assert_eq!(index.len(), 3);
        assert_eq!(index.embeddings.len(), 3 * dim);

        index.remove_file("/a/f1.rs");
        assert_eq!(index.len(), 2);
        assert_eq!(index.embeddings.len(), 2 * dim);

        // All remaining files should NOT be /a/f1.rs
        for cid in &index.chunk_ids {
            assert_ne!(cid.file, "/a/f1.rs");
        }

        // Search should still return results from remaining files.
        let query = emb(dim, 2.0);
        let results = index.search(&query, 2);
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_ne!(r.chunk_id.file, "/a/f1.rs");
        }
    }
}
