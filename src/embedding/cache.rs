use std::path::{Path, PathBuf};
use std::time::SystemTime;

use tracing::{trace, warn};

/// Result of a cache purge operation.
pub struct PurgeResult {
    pub deleted: u64,
    pub errors: u64,
}

/// File-based embedding cache.
///
/// Layout: `{cache_dir}/{md5[..2]}/{md5}.bin`
/// Each `.bin` file stores the embedding as little-endian IEEE 754 f32 values,
/// 4 bytes each, with no header.
pub struct EmbeddingCache {
    cache_dir: PathBuf,
}

impl EmbeddingCache {
    /// Construct a new cache rooted at `embeddings_dir/{model}/`.
    ///
    /// `embeddings_dir` is the boot-resolved embedding-cache root (CLI > env >
    /// `Settings.embeddings_dir` > `<data_dir>/embeddings`) — the FULL root, not
    /// a base to append `embeddings` to. Captured once at startup; MUST NOT be
    /// re-derived from `Settings` mid-run.
    ///
    /// Returns `None` if the directory cannot be created, so callers can
    /// degrade gracefully (pipeline still works, just with no cache).
    pub fn new(embeddings_dir: &Path, model: &str) -> Option<Self> {
        let sanitized: String = model
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        let cache_dir = embeddings_dir.join(&sanitized);

        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            warn!(path = %cache_dir.display(), error = %e, "embedding cache: failed to create cache dir; disabling cache");
            return None;
        }

        Some(Self { cache_dir })
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    fn compute_md5(text: &str) -> String {
        format!("{:x}", md5::compute(text.as_bytes()))
    }

    fn entry_path(&self, md5_hex: &str) -> PathBuf {
        self.cache_dir
            .join(&md5_hex[..2])
            .join(format!("{md5_hex}.bin"))
    }

    fn encode_embedding(embedding: &[f32]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(embedding.len() * 4);
        for &f in embedding {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf
    }

    fn decode_embedding(bytes: &[u8]) -> Option<Vec<f32>> {
        if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
            return None;
        }
        Some(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    // ── Public API ───────────────────────────────────────────────────────

    /// Look up embeddings for many texts in the cache.
    ///
    /// Returns `(hits, miss_indices)` where:
    /// - `hits`: `(original_index, embedding)` for every text that was cached.
    /// - `miss_indices`: indices of texts not found or with corrupt cache entries.
    ///
    /// Mtime is touched on each hit so LRU-style purging works correctly.
    /// Corrupt files (non-empty but decode-fail) are deleted on discovery.
    pub fn get_many(&self, texts: &[String]) -> (Vec<(usize, Vec<f32>)>, Vec<usize>) {
        let mut hits: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut misses: Vec<usize> = Vec::new();

        for (idx, text) in texts.iter().enumerate() {
            let md5 = Self::compute_md5(text);
            let path = self.entry_path(&md5);

            match std::fs::read(&path) {
                Ok(bytes) => {
                    match Self::decode_embedding(&bytes) {
                        Some(embedding) => {
                            // Touch mtime so LRU-style purges keep this entry.
                            if let Err(e) = filetime::set_file_mtime(
                                &path,
                                filetime::FileTime::now(),
                            ) {
                                trace!(path = %path.display(), error = %e, "cache hit: mtime touch failed (non-fatal)");
                            }
                            hits.push((idx, embedding));
                        }
                        None => {
                            // Corrupt file — delete it so it doesn't block future
                            // writes.
                            warn!(path = %path.display(), "embedding cache: corrupt entry (decode failed); deleting");
                            let _ = std::fs::remove_file(&path);
                            misses.push(idx);
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    misses.push(idx);
                }
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "embedding cache: read error; treating as miss");
                    misses.push(idx);
                }
            }
        }

        (hits, misses)
    }

    /// Write embeddings for many texts to the cache atomically.
    ///
    /// Each entry is written via a `NamedTempFile` then persisted (renamed) to
    /// its final path, so a crash during write leaves no partial file behind.
    /// Empty embeddings are skipped — they must not be cached.
    pub fn put_many(&self, texts: &[String], embeddings: &[Vec<f32>]) {
        for (text, embedding) in texts.iter().zip(embeddings.iter()) {
            if embedding.is_empty() {
                continue;
            }

            let md5 = Self::compute_md5(text);
            let final_path = self.entry_path(&md5);
            let shard_dir = match final_path.parent() {
                Some(d) => d.to_path_buf(),
                None => {
                    warn!("embedding cache: could not determine shard dir; skipping");
                    continue;
                }
            };

            if let Err(e) = std::fs::create_dir_all(&shard_dir) {
                warn!(path = %shard_dir.display(), error = %e, "embedding cache: failed to create shard dir; skipping");
                continue;
            }

            let bytes = Self::encode_embedding(embedding);

            // Atomic write: write to a tempfile in the same shard dir, then rename.
            match tempfile::NamedTempFile::new_in(&shard_dir) {
                Ok(tmp) => {
                    use std::io::Write;
                    let mut tmp = tmp;
                    if let Err(e) = tmp.write_all(&bytes) {
                        warn!(path = %final_path.display(), error = %e, "embedding cache: tempfile write failed; skipping");
                        continue;
                    }
                    if let Err(e) = tmp.persist(&final_path) {
                        warn!(path = %final_path.display(), error = %e, "embedding cache: persist (rename) failed; skipping");
                    }
                }
                Err(e) => {
                    warn!(shard = %shard_dir.display(), error = %e, "embedding cache: NamedTempFile::new_in failed; skipping");
                }
            }
        }
    }

    /// Purge cache entries across ALL model subdirectories under
    /// `embeddings_dir/`.
    ///
    /// `embeddings_dir` is the boot-resolved embedding-cache root (CLI > env >
    /// `Settings.embeddings_dir` > `<data_dir>/embeddings`) — the FULL root.
    ///
    /// - `older_than = None`  → delete all `.bin` files.
    /// - `older_than = Some(d)` → delete `.bin` files whose mtime is older than
    ///   `SystemTime::now() - d`.
    ///
    /// Empty shard directories are removed after file deletion (best-effort).
    /// Returns the count of deleted files and the count of errors.
    pub fn purge_global(embeddings_dir: &Path, older_than: Option<std::time::Duration>) -> PurgeResult {
        let root = embeddings_dir.to_path_buf();

        if !root.exists() {
            return PurgeResult { deleted: 0, errors: 0 };
        }

        let cutoff: Option<SystemTime> =
            older_than.map(|d| SystemTime::now() - d);

        let mut deleted: u64 = 0;
        let mut errors: u64 = 0;

        // Walk the tree, collecting shard dirs as we go so we can try to
        // remove them afterwards.
        let mut shard_dirs: Vec<PathBuf> = Vec::new();

        // Recursive walk via a manual stack (avoids a walkdir dependency).
        let mut stack: Vec<PathBuf> = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            let entries = match std::fs::read_dir(&dir) {
                Ok(e) => e,
                Err(e) => {
                    warn!(dir = %dir.display(), error = %e, "purge_global: read_dir failed");
                    errors += 1;
                    continue;
                }
            };

            for entry in entries {
                let entry = match entry {
                    Ok(e) => e,
                    Err(e) => {
                        warn!(error = %e, "purge_global: dir entry error");
                        errors += 1;
                        continue;
                    }
                };
                let path = entry.path();
                if path.is_dir() {
                    shard_dirs.push(path.clone());
                    stack.push(path);
                } else if path.extension().map(|e| e == "bin").unwrap_or(false) {
                    // Decide whether to delete based on mtime / cutoff.
                    let should_delete = match cutoff {
                        None => true,
                        Some(cutoff_time) => {
                            match std::fs::metadata(&path)
                                .and_then(|m| m.modified())
                            {
                                Ok(mtime) => mtime < cutoff_time,
                                Err(e) => {
                                    warn!(path = %path.display(), error = %e, "purge_global: metadata failed; skipping");
                                    errors += 1;
                                    false
                                }
                            }
                        }
                    };

                    if should_delete {
                        match std::fs::remove_file(&path) {
                            Ok(()) => deleted += 1,
                            Err(e) => {
                                warn!(path = %path.display(), error = %e, "purge_global: remove_file failed");
                                errors += 1;
                            }
                        }
                    }
                }
            }
        }

        // Best-effort: remove now-empty shard dirs (deepest first).
        // Reverse so children are tried before parents.
        shard_dirs.reverse();
        for dir in shard_dirs {
            // remove_dir only succeeds on empty directories — correct behaviour here.
            let _ = std::fs::remove_dir(&dir);
        }

        PurgeResult { deleted, errors }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};
    use tempfile::TempDir;

    // ── (a) encode→decode round-trip ────────────────────────────────────────

    #[test]
    fn encode_decode_round_trip() {
        let original: Vec<f32> = vec![-1.5, 0.0, 2.5, -0.001, 1e30, 255.0, -128.0];
        let bytes = EmbeddingCache::encode_embedding(&original);
        let decoded = EmbeddingCache::decode_embedding(&bytes).expect("decode must succeed");
        assert_eq!(decoded, original);
    }

    // ── (b) decode rejects non-multiple-of-4 lengths ────────────────────────

    #[test]
    fn decode_rejects_bad_length() {
        // Not a multiple of 4
        assert!(EmbeddingCache::decode_embedding(&[1, 2, 3, 4, 5]).is_none());
        assert!(EmbeddingCache::decode_embedding(&[1, 2, 3]).is_none());
        assert!(EmbeddingCache::decode_embedding(&[1, 2, 3, 4, 5, 6, 7]).is_none());
        // Empty slice
        assert!(EmbeddingCache::decode_embedding(&[]).is_none());
    }

    // ── (c) corrupt entry is deleted and returned as miss ───────────────────

    #[test]
    fn corrupt_entry_deleted_on_get() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = EmbeddingCache::new(tmp.path(), "test-model").expect("cache");

        let text = "corrupt-text".to_string();
        let md5 = EmbeddingCache::compute_md5(&text);
        let path = cache.entry_path(&md5);

        // Create the shard directory and write a non-multiple-of-4 byte file.
        std::fs::create_dir_all(path.parent().unwrap()).expect("shard dir");
        std::fs::write(&path, [1u8, 2, 3, 4, 5]).expect("write corrupt file");
        assert!(path.exists(), "corrupt file should exist before get_many");

        let (hits, misses) = cache.get_many(&[text]);

        assert!(hits.is_empty(), "corrupt entry must not be a hit");
        assert_eq!(misses, vec![0], "corrupt entry must be a miss at index 0");
        assert!(!path.exists(), "corrupt file must be deleted by get_many");
    }

    // ── (d) put then get round-trips through the filesystem ─────────────────

    #[test]
    fn put_then_get_round_trip() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = EmbeddingCache::new(tmp.path(), "test-model").expect("cache");

        let texts = vec!["hello".to_string(), "world".to_string()];
        let embeddings = vec![vec![1.0f32, 2.0, 3.0], vec![4.0f32, 5.0, 6.0]];

        cache.put_many(&texts, &embeddings);

        let (hits, misses) = cache.get_many(&texts);

        assert!(misses.is_empty(), "both entries should be cache hits");
        assert_eq!(hits.len(), 2);

        // Build index → embedding map from hits
        let mut result = std::collections::HashMap::new();
        for (idx, emb) in hits {
            result.insert(idx, emb);
        }

        assert_eq!(result[&0], vec![1.0f32, 2.0, 3.0]);
        assert_eq!(result[&1], vec![4.0f32, 5.0, 6.0]);
    }

    #[test]
    fn empty_embedding_not_cached() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = EmbeddingCache::new(tmp.path(), "test-model").expect("cache");

        cache.put_many(&["empty".to_string()], &[vec![]]);

        let (hits, misses) = cache.get_many(&["empty".to_string()]);

        assert!(hits.is_empty(), "empty embedding must not be cached");
        assert_eq!(misses, vec![0], "empty embedding must be a miss");
    }

    // ── (e) purge_global with older_than deletes only stale entries ──────────

    #[test]
    fn purge_global_deletes_only_stale() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = EmbeddingCache::new(tmp.path(), "test-model").expect("cache");

        let texts = vec!["fresh-text".to_string(), "stale-text".to_string()];
        let embeddings = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]];
        cache.put_many(&texts, &embeddings);

        // Backdate the "stale-text" entry to 40 days ago.
        let stale_md5 = EmbeddingCache::compute_md5("stale-text");
        let stale_path = cache.entry_path(&stale_md5);
        let forty_days_ago = SystemTime::now() - Duration::from_secs(40 * 24 * 3600);
        filetime::set_file_mtime(
            &stale_path,
            filetime::FileTime::from_system_time(forty_days_ago),
        )
        .expect("set mtime");

        // Purge entries older than 30 days.
        let result =
            EmbeddingCache::purge_global(tmp.path(), Some(Duration::from_secs(30 * 24 * 3600)));

        assert_eq!(result.deleted, 1, "only the stale entry should be deleted");
        assert_eq!(result.errors, 0);

        // Fresh entry still exists; stale is gone.
        let fresh_md5 = EmbeddingCache::compute_md5("fresh-text");
        let fresh_path = cache.entry_path(&fresh_md5);
        assert!(fresh_path.exists(), "fresh entry must survive the purge");
        assert!(!stale_path.exists(), "stale entry must be deleted");

        // Purge with no cutoff should delete all remaining entries.
        let result2 = EmbeddingCache::purge_global(tmp.path(), None);
        assert_eq!(result2.deleted, 1, "one remaining entry should be deleted");
        assert_eq!(result2.errors, 0);
        assert!(!fresh_path.exists(), "fresh entry must be gone after unconditional purge");
    }
}
