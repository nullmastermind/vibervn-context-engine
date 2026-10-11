use anyhow::{Context, Result};
use rayon::prelude::*;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tracing::info;

use crate::path_in_repo;

pub mod scope;
pub mod shard_file;
pub mod sharded;
pub use scope::{Scope, ScopeFilter, ext_key};

#[cfg(test)]
mod scope_tests;
pub use sharded::{ScopedAll, ScopedSearch, ShardedSearch, ShardedVectorIndex};

// ─── Public types ─────────────────────────────────────────────────────────

/// Identifies a chunk by its location in the source tree.
/// Used to map VectorIndex results back to SurrealDB rows.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Row-major embedding storage: either an owned heap `Vec<f32>` (the mutable
    /// build path) or a memory-mapped read-only f32 region (the mmap path, where
    /// the OS page cache owns physical residency — not our heap). Both expose a
    /// contiguous `&[f32]` of length `len * dim` via `embeddings_slice()`.
    store: EmbeddingStore,
    /// Parallel array: chunk_ids[i] corresponds to row i. Always heap-resident
    /// (small: ~tens of bytes/row), even for an mmap shard.
    chunk_ids: Vec<ChunkId>,
    /// Parallel array: ext_ids[i] is row i's interned extension id (see
    /// `scope.rs`). Derived from `chunk_ids[i].file`, never persisted, so legacy
    /// and mmap shards get it for free. INVARIANT: same length as `chunk_ids`,
    /// mutated in lockstep by every method that touches `chunk_ids`.
    ext_ids: Vec<u16>,
    /// Interns extension keys for `ext_ids`. Ids are local to this shard.
    exts: scope::ExtTable,
    /// Dimensionality of every vector. 0 until the first insert / mmap load.
    dim: usize,
}

/// Backing storage for a shard's flat row-major f32 embeddings.
enum EmbeddingStore {
    /// Owned heap storage — the mutable build path (insert/remove/merge).
    Ram(Vec<f32>),
    /// Memory-mapped read-only f32 region. `map` holds the file mapping alive;
    /// `f32_off`/`f32_len` delimit the f32 payload within it (after the header).
    /// Physical residency is OS-page-cache-owned, NOT counted against the heap cap.
    Mmap {
        map: memmap2::Mmap,
        f32_off: usize,
        f32_len: usize,
    },
}

impl EmbeddingStore {
    /// The flat row-major f32 slice, regardless of backing.
    #[inline]
    fn as_slice(&self) -> &[f32] {
        match self {
            EmbeddingStore::Ram(v) => v.as_slice(),
            EmbeddingStore::Mmap {
                map,
                f32_off,
                f32_len,
            } => {
                let bytes = &map[*f32_off..*f32_off + *f32_len * std::mem::size_of::<f32>()];
                // The writer 4-byte-aligns the f32 region (validated on open), so
                // this cast is sound. bytemuck checks alignment + size.
                bytemuck::cast_slice::<u8, f32>(bytes)
            }
        }
    }

    /// Mutable heap Vec for the build/mutation path.
    ///
    /// If the store is currently mmap-backed (read-only), it is MATERIALIZED first:
    /// the mmap'd f32 region is copied into an owned `Vec` once and the mapping is
    /// dropped, so the shard becomes mutable. This happens on the FIRST incremental
    /// edit after a repo was cold-warmed from its persisted shard file — a one-time
    /// ~payload-sized copy; subsequent edits are O(changed) on the now-in-RAM shard.
    /// The shard stays resident and correct throughout (no re-warm, no DB round-trip).
    #[inline]
    fn ram_mut(&mut self) -> &mut Vec<f32> {
        if let EmbeddingStore::Mmap { .. } = self {
            // Materialize: copy the mmap'd payload to an owned Vec, drop the mapping.
            let owned: Vec<f32> = self.as_slice().to_vec();
            *self = EmbeddingStore::Ram(owned);
        }
        match self {
            EmbeddingStore::Ram(v) => v,
            EmbeddingStore::Mmap { .. } => unreachable!("materialized to Ram above"),
        }
    }
}

impl VectorIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            store: EmbeddingStore::Ram(Vec::new()),
            chunk_ids: Vec::new(),
            ext_ids: Vec::new(),
            exts: scope::ExtTable::default(),
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
            self.store.ram_mut().extend_from_slice(&normalized);
            self.chunk_ids.push(id.clone());
            let ext_id = self.exts.intern_path(&id.file);
            self.ext_ids.push(ext_id);
        }
    }

    /// Remove every embedding whose `file` is in `files`, in ONE pass over the
    /// shard: O(shard) per call instead of O(shard × files) for a per-file loop,
    /// which a branch switch touching thousands of files turned into a multi-minute
    /// stall at kernel scale. Stable compaction (row order is irrelevant to
    /// search). An mmap-backed shard is only materialized when something actually
    /// matches.
    pub fn remove_files(&mut self, files: &std::collections::HashSet<&str>) {
        if files.is_empty()
            || !self
                .chunk_ids
                .iter()
                .any(|c| files.contains(c.file.as_str()))
        {
            return;
        }
        let dim = self.dim;
        let emb = self.store.ram_mut();
        let mut write = 0usize;
        for read in 0..self.chunk_ids.len() {
            if files.contains(self.chunk_ids[read].file.as_str()) {
                continue;
            }
            if write != read {
                emb.copy_within(read * dim..(read + 1) * dim, write * dim);
                self.chunk_ids.swap(write, read);
                self.ext_ids.swap(write, read);
            }
            write += 1;
        }
        self.chunk_ids.truncate(write);
        self.ext_ids.truncate(write);
        emb.truncate(write * dim);
        if self.chunk_ids.is_empty() {
            self.dim = 0;
        }
    }

    /// Remove all embeddings whose `file` field matches `file`.
    ///
    /// Uses swap-remove to avoid O(n) shifts; moves whole `dim`-sized rows.
    pub fn remove_file(&mut self, file: &str) {
        let emb = self.store.ram_mut();
        let mut i = 0;
        while i < self.chunk_ids.len() {
            if self.chunk_ids[i].file == file {
                self.chunk_ids.swap_remove(i);
                self.ext_ids.swap_remove(i);
                // Swap the flat rows: row i ↔ last row.
                let last = self.chunk_ids.len(); // after swap_remove, len is already decremented
                if i < last {
                    // Copy last row into slot i.
                    let src_start = last * self.dim;
                    let dst_start = i * self.dim;
                    // Safety: src and dst are non-overlapping (i < last).
                    let src: Vec<f32> = emb[src_start..src_start + self.dim].to_vec();
                    emb[dst_start..dst_start + self.dim].copy_from_slice(&src);
                }
                emb.truncate(last * self.dim);
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
        let emb = self.store.ram_mut();
        let mut i = 0;
        while i < self.chunk_ids.len() {
            if path_in_repo(&self.chunk_ids[i].file, repo) {
                self.chunk_ids.swap_remove(i);
                self.ext_ids.swap_remove(i);
                let last = self.chunk_ids.len();
                if i < last {
                    let src_start = last * self.dim;
                    let dst_start = i * self.dim;
                    let src: Vec<f32> = emb[src_start..src_start + self.dim].to_vec();
                    emb[dst_start..dst_start + self.dim].copy_from_slice(&src);
                }
                emb.truncate(last * self.dim);
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
        self.store
            .ram_mut()
            .extend_from_slice(other.store.as_slice());
        // `other`'s ext ids index ITS table, not ours: remap through the names.
        let remap: Vec<u16> = other
            .exts
            .names()
            .iter()
            .map(|n| self.exts.intern(n))
            .collect();
        self.ext_ids.extend(other.ext_ids.iter().map(|&id| {
            remap
                .get(id as usize)
                .copied()
                .unwrap_or(scope::EXT_SENTINEL)
        }));
        self.chunk_ids.extend(other.chunk_ids);
    }

    /// Search for the top-k most similar chunks to `query`.
    ///
    /// `query` is normalized internally so the caller need not pre-normalize.
    /// Returns results sorted by descending score, capped at `top_k`.
    /// Uses rayon for parallel dot-product computation.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        self.search_filtered(query, top_k, None)
    }

    /// [`search`](Self::search) restricted to one scope (code or docs).
    ///
    /// The restriction is applied INSIDE the scan, before top-k selection: rows of
    /// the other class are skipped without computing their dot product, so a class
    /// that is crowded out by higher-scoring rows of the other class (docs vs
    /// code) is never starved the way a post-filter over top-k would be. Scores
    /// are plain cosine, so cross-shard merging stays exact.
    pub fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        scope: Option<&ScopeFilter>,
    ) -> Vec<SearchResult> {
        if self.is_empty() || query.is_empty() || top_k == 0 || self.dim == 0 {
            return vec![];
        }

        let q_norm = l2_normalize(query);

        // Parallel dot product over rows. Each row is exactly `self.dim` floats.
        // `embeddings_slice()` borrows the f32 region from either heap or mmap —
        // the dot-product kernel is identical, so scores are bit-identical to the
        // in-RAM path (exact, no quantization).
        let rows = self.store.as_slice().par_chunks(self.dim).enumerate();
        let mut scored: Vec<(usize, f32)> = match scope {
            None => rows
                .map(|(i, row)| (i, dot_product(&q_norm, row)))
                .collect(),
            Some(sf) => {
                let keep = sf.keep_table(&self.exts);
                let keep_unknown = sf.keeps_unknown();
                rows.filter_map(|(i, row)| {
                    let kept = keep
                        .get(self.ext_ids[i] as usize)
                        .copied()
                        .unwrap_or(keep_unknown);
                    kept.then(|| (i, dot_product(&q_norm, row)))
                })
                .collect()
            }
        };

        // Nothing survived the scope filter: `k - 1` below would underflow.
        if scored.is_empty() {
            return vec![];
        }

        // Partial sort: bring the top-k largest scores to the front.
        let k = top_k.min(scored.len());
        scored.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
            count,
            select_ms,
            decode_ms,
            insert_ms,
            total_ms = select_ms + decode_ms + insert_ms,
            "PERF SUMMARY load_from_db (cold shard warm breakdown)"
        );

        Ok(index)
    }

    /// Remove all entries from the index.
    pub fn clear(&mut self) {
        self.store = EmbeddingStore::Ram(Vec::new());
        self.chunk_ids.clear();
        self.ext_ids.clear();
        self.exts = scope::ExtTable::default();
        self.dim = 0;
    }

    /// HEAP bytes occupied by this shard — the resident-cap accounting metric.
    /// In-RAM: the flat f32 storage (`len * 4`) — payload-only, as before.
    /// MMAP: the f32 payload is page-cache-resident (NOT heap → 0); we instead
    /// charge the heap-resident chunk-id sidecar so the cap stays meaningful for
    /// mmap shards (bounds heap AND indirectly bounds open mmap handles — evicting
    /// to honor the cap drops both the sidecar and the `Mmap` handle). The OS page
    /// cache independently bounds the mmap payload's physical residency.
    #[inline]
    pub fn byte_size(&self) -> usize {
        match &self.store {
            EmbeddingStore::Ram(v) => v.len() * std::mem::size_of::<f32>(),
            EmbeddingStore::Mmap { .. } => {
                self.chunk_ids
                    .iter()
                    .map(|c| c.file.len() + 32)
                    .sum::<usize>()
                    + self.ext_ids.len() * std::mem::size_of::<u16>()
            }
        }
    }

    /// Dimensionality (0 if empty).
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Borrow the flat row-major f32 slice + chunk ids — used by the shard-file
    /// writer to serialize an in-RAM shard to disk.
    pub(crate) fn raw_parts(&self) -> (&[f32], &[ChunkId], usize) {
        (self.store.as_slice(), &self.chunk_ids, self.dim)
    }

    /// Construct an MMAP-backed shard from a validated mapping + its chunk ids.
    /// The caller (shard_file::open) has already validated header/length/alignment.
    pub(crate) fn from_mmap(
        map: memmap2::Mmap,
        f32_off: usize,
        f32_len: usize,
        dim: usize,
        chunk_ids: Vec<ChunkId>,
    ) -> Self {
        // Not persisted in the shard file: re-derived from the paths on open.
        let mut exts = scope::ExtTable::default();
        let ext_ids = chunk_ids
            .iter()
            .map(|c| exts.intern_path(&c.file))
            .collect();
        Self {
            store: EmbeddingStore::Mmap {
                map,
                f32_off,
                f32_len,
            },
            chunk_ids,
            ext_ids,
            exts,
            dim,
        }
    }

    /// True if this shard is mmap-backed (its f32 payload is page-cache-resident,
    /// not heap). Used by the engine to skip heap-cap eviction of the payload.
    #[inline]
    pub fn is_mmap(&self) -> bool {
        matches!(self.store, EmbeddingStore::Mmap { .. })
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

// ─── Scalar quantization (i8) ─────────────────────────────────────────────
//
// Shrinks a stored embedding ~4x (f32 -> i8) so a kernel-scale shard (3.73 GB)
// fits under the 2 GB resident cap and warms ~4x faster. The scale is a SINGLE
// GLOBAL constant (×127), valid because embeddings are L2-normalized before
// quantization so every component is in [-1, 1]. A global (not per-shard, not
// per-vector) scale is what keeps cross-shard scores comparable: the dequantized
// dot-product is an integer dot-product times the SAME constant `1/QUANT_SCALE^2`
// for every shard, a positive monotonic factor that cannot reorder candidates
// across shards — so a cross-shard top-k merge stays exact w.r.t. the quantized
// vectors. A per-shard scale would break that and is therefore forbidden.

/// Global quantization scale. i8 range is [-127, 127] (127, not 128, so +1.0 and
/// -1.0 are symmetric and |q| never exceeds 127).
///
/// NOTE: the i8 quantization path is currently TEST-ONLY. The recall gate
/// (recall_gate_i8_vs_f32) measured recall@10=0.93 on the real Linux-kernel index
/// — below the 0.98 acceptance gate — because kernel embeddings are packed denser
/// than the i8 step can resolve. Per the change's pre-committed decision, i8 is NOT
/// adopted for storage; the mmap-f32 fallback is the path. These helpers are
/// retained, gated under cfg(test), as the reproducible gate machinery.
#[cfg(test)]
pub(crate) const QUANT_SCALE: f32 = 127.0;

/// Quantize an ALREADY-L2-NORMALIZED vector to i8 with the global scale.
/// Components are clamped to [-127, 127] defensively (a normalized component is
/// already in [-1, 1], but floating error at the boundary could round to ±128).
#[cfg(test)]
#[inline]
pub(crate) fn quantize_i8(normalized: &[f32]) -> Vec<i8> {
    normalized
        .iter()
        .map(|&c| {
            let q = (c * QUANT_SCALE).round();
            q.clamp(-127.0, 127.0) as i8
        })
        .collect()
}

/// Integer dot product of two i8 rows, accumulated in i32 (no overflow: at
/// dim=1024, max |sum| = 1024 × 127 × 127 ≈ 16.5M, well within i32).
#[cfg(test)]
#[inline]
pub(crate) fn dot_product_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// Dequantized dot product: the comparable cosine-equivalent score between two
/// i8 rows. The `1/QUANT_SCALE^2` factor is global, so scores from different
/// shards are directly comparable.
#[cfg(test)]
#[inline]
pub(crate) fn dot_product_i8_dequant(a: &[i8], b: &[i8]) -> f32 {
    dot_product_i8(a, b) as f32 / (QUANT_SCALE * QUANT_SCALE)
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

    // ─── i8 quantization core ────────────────────────────────────────────

    /// Round-trip error is bounded by the quantization step (1/127 per component).
    /// A normalized vector quantized then dequantized stays within step/2 per dim.
    #[test]
    fn quantize_i8_roundtrip_error_bounded() {
        let v = l2_normalize(&emb(1024, 0.3));
        let q = quantize_i8(&v);
        assert_eq!(q.len(), v.len());
        let step = 1.0 / QUANT_SCALE;
        for (orig, &qi) in v.iter().zip(&q) {
            let deq = qi as f32 / QUANT_SCALE;
            assert!(
                (orig - deq).abs() <= step / 2.0 + 1e-6,
                "component error {} exceeds half-step {}",
                (orig - deq).abs(),
                step / 2.0
            );
            assert!((-127..=127).contains(&(qi as i32)), "i8 in [-127,127]");
        }
    }

    /// Boundary components ±1.0 quantize to ±127 (never ±128 — no overflow).
    #[test]
    fn quantize_i8_clamps_at_boundary() {
        let v = vec![1.0_f32, -1.0, 0.0, 0.999, -0.999];
        let q = quantize_i8(&v);
        assert_eq!(q, vec![127, -127, 0, 127, -127]);
    }

    /// The dequantized dot product equals the integer dot product times the global
    /// constant — and no overflow at dim=1024 with extreme values.
    #[test]
    fn dot_product_i8_matches_dequant_and_no_overflow() {
        let a = vec![127_i8; 1024];
        let b = vec![127_i8; 1024];
        let int_dot = dot_product_i8(&a, &b);
        assert_eq!(int_dot, 1024 * 127 * 127); // 16,516,096 — fits i32
        let deq = dot_product_i8_dequant(&a, &b);
        assert!((deq - int_dot as f32 / (QUANT_SCALE * QUANT_SCALE)).abs() < 1e-3);
    }

    /// Quantized cosine approximates true cosine for normalized vectors.
    #[test]
    fn quantized_cosine_approximates_true_cosine() {
        let a = l2_normalize(&emb(1024, 0.2));
        let b = l2_normalize(&emb(1024, 0.7));
        let true_cos = dot_product(&a, &b);
        let q_cos = dot_product_i8_dequant(&quantize_i8(&a), &quantize_i8(&b));
        assert!(
            (true_cos - q_cos).abs() < 0.01,
            "quantized cosine {q_cos} too far from true {true_cos}"
        );
    }

    /// CROSS-SHARD EXACTNESS (the load-bearing invariant): with a single GLOBAL
    /// scale, ranking candidates from two different shards by the dequantized score
    /// yields the SAME order as ranking the same quantized vectors in one combined
    /// pool. A per-shard scale would break this. We build two disjoint vector sets
    /// ("shards"), score a query against each with the i8 path, merge by score, and
    /// assert the merged order equals a single-pool i8 ranking.
    #[test]
    fn cross_shard_merge_is_exact_under_global_scale() {
        let query = l2_normalize(&emb(64, 0.5));
        let q_i8 = quantize_i8(&query);

        // Two shards of normalized vectors with distinct seeds.
        let shard_a: Vec<Vec<i8>> = (0..10)
            .map(|i| quantize_i8(&l2_normalize(&emb(64, 0.1 * i as f32 + 0.05))))
            .collect();
        let shard_b: Vec<Vec<i8>> = (0..10)
            .map(|i| quantize_i8(&l2_normalize(&emb(64, 0.1 * i as f32 + 0.5))))
            .collect();

        // Per-shard scored, then merged by the comparable dequant score.
        let mut merged: Vec<(String, f32)> = Vec::new();
        for (i, v) in shard_a.iter().enumerate() {
            merged.push((format!("a{i}"), dot_product_i8_dequant(&q_i8, v)));
        }
        for (i, v) in shard_b.iter().enumerate() {
            merged.push((format!("b{i}"), dot_product_i8_dequant(&q_i8, v)));
        }
        merged.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        let merged_order: Vec<String> = merged.iter().map(|(k, _)| k.clone()).collect();

        // Single combined pool, same i8 vectors, same scoring.
        let mut combined: Vec<(String, f32)> = Vec::new();
        for (i, v) in shard_a.iter().enumerate() {
            combined.push((format!("a{i}"), dot_product_i8_dequant(&q_i8, v)));
        }
        for (i, v) in shard_b.iter().enumerate() {
            combined.push((format!("b{i}"), dot_product_i8_dequant(&q_i8, v)));
        }
        combined.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        let combined_order: Vec<String> = combined.iter().map(|(k, _)| k.clone()).collect();

        assert_eq!(
            merged_order, combined_order,
            "global-scale cross-shard merge must equal single-pool ranking"
        );
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

        let foo_chunks: Vec<(ChunkId, Vec<f32>)> =
            vec![(chunk(r"D:\proj\foo\x.rs", 1, 10), emb(4, 1.0))];
        let foobar_chunks: Vec<(ChunkId, Vec<f32>)> =
            vec![(chunk(r"D:\proj\foobar\y.rs", 1, 10), emb(4, 2.0))];

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
                (chunk(&format!("/f{i}.rs",), 1, 10), v)
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
        assert_eq!(index.store.as_slice().len(), 3 * dim);

        index.remove_file("/a/f1.rs");
        assert_eq!(index.len(), 2);
        assert_eq!(index.store.as_slice().len(), 2 * dim);

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

    /// D10: batched removal keeps exactly the rows a per-file loop keeps, with
    /// each surviving id still paired with its own vector.
    #[test]
    fn remove_files_matches_per_file_removal() {
        let dim = 4;
        let mut batched = VectorIndex::new();
        let rows: Vec<(ChunkId, Vec<f32>)> = (0..12)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i % dim] = 1.0 + i as f32;
                (
                    chunk(&format!("/r/f{}.rs", i % 5), i as u32, i as u32 + 1),
                    v,
                )
            })
            .collect();
        batched.insert(&rows);
        let mut looped = VectorIndex::new();
        looped.insert(&rows);

        let gone = ["/r/f0.rs", "/r/f3.rs", "/r/missing.rs"];
        batched.remove_files(&gone.iter().copied().collect());
        for f in gone {
            looped.remove_file(f);
        }

        let key = |ix: &VectorIndex| {
            let mut v: Vec<(String, u32, Vec<u32>)> = ix
                .chunk_ids
                .iter()
                .enumerate()
                .map(|(row, c)| {
                    let vec = ix.store.as_slice()[row * dim..(row + 1) * dim]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect();
                    (c.file.clone(), c.line_start, vec)
                })
                .collect();
            v.sort();
            v
        };
        assert_eq!(key(&batched), key(&looped));
        assert!(
            batched
                .chunk_ids
                .iter()
                .all(|c| !gone.contains(&c.file.as_str()))
        );
        assert_eq!(batched.store.as_slice().len(), batched.len() * dim);
    }
}
