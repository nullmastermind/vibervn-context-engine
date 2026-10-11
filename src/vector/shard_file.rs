//! On-disk persisted vector shards: a flat, row-major little-endian f32 file per
//! repo that is memory-mapped for search. The file is a DERIVED CACHE of the
//! `chunk` table embeddings — it is never the source of truth, so it can always
//! be rebuilt, which is what makes crash-safety trivial (validate-or-rebuild, no
//! commit marker) and the migration zero-cost (lazy: built on first warm).
//!
//! Physical residency of the mmap'd f32 payload is owned by the OS page cache,
//! NOT the process heap — so a 3.73 GB kernel shard does not sit on our heap or
//! count against `vector_resident_cap_mb`.
//!
//! ## Layout
//!
//! ```text
//! <data_dir>/vector_shards/<sanitized-repo>/CURRENT        (ascii generation number)
//! <data_dir>/vector_shards/<sanitized-repo>/<generation>/shard.f32  (header + f32 rows)
//! <data_dir>/vector_shards/<sanitized-repo>/<generation>/shard.ids  (chunk-id sidecar)
//! ```
//!
//! ## Windows-safe swap (concurrent mmap during a full-rebuild rewrite)
//!
//! On win32, `rename`-over a file that a query has mmap'd raises a sharing
//! violation (unlike POSIX inode-swap). So a rewrite NEVER replaces a mapped
//! file: it writes a NEW generation dir, then flips `CURRENT`. In-flight readers
//! keep their old mapping (its file is untouched); new warms read `CURRENT`. A
//! stale generation is reaped only when no live handle references it (tracked by the
//! engine) or at startup (no handles survive a restart). Within a single generation the
//! first write still uses tmp→fsync→rename for crash-safety of that one file.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::vector::{ChunkId, VectorIndex};

/// `b"VSF1"` — Vector Shard File, format v1.
const MAGIC: u32 = 0x31_46_53_56;
const FORMAT_VERSION: u16 = 1;
/// Header is padded to this many bytes so the f32 region that follows is
/// 4-byte aligned (required for the `&[u8]`→`&[f32]` cast on mmap).
const HEADER_BYTES: usize = 64;

/// Per-repo root: `<data_dir>/vector_shards/<sanitized-repo>/`.
pub fn repo_shard_root(data_dir: &Path, repo: &str) -> PathBuf {
    data_dir
        .join("vector_shards")
        .join(crate::store::sanitize_repo_name(repo))
}

/// Read the CURRENT generation number for a repo, if any.
pub fn read_current_gen(root: &Path) -> Option<u64> {
    std::fs::read_to_string(root.join("CURRENT"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
}

/// Remove a repo's `CURRENT` pointer only if it still points at `expected`.
///
/// A failed index run captures the generation visible before it publishes. A
/// concurrent warm may later write and flip to a newer generation; comparing the
/// pointer before removal prevents the old rollback from invalidating that newer
/// persisted shard. `expected = None` fences the no-CURRENT state in the same way.
pub fn invalidate_current_if_generation(
    data_dir: &Path,
    repo: &str,
    expected: Option<u64>,
) -> bool {
    let root = repo_shard_root(data_dir, repo);
    if read_current_gen(&root) != expected {
        return false;
    }
    std::fs::remove_file(root.join("CURRENT")).is_ok()
}

/// Atomically write the CURRENT pointer (tmp + rename — a tiny file, always safe
/// to replace since nobody mmaps CURRENT itself).
fn write_current_gen(root: &Path, generation: u64) -> Result<()> {
    std::fs::create_dir_all(root).ok();
    let tmp = root.join(format!("CURRENT.tmp.{}", std::process::id()));
    std::fs::write(&tmp, generation.to_string()).context("write CURRENT tmp")?;
    std::fs::rename(&tmp, root.join("CURRENT")).context("rename CURRENT")?;
    Ok(())
}

/// Serialize a built in-RAM shard to a NEW generation dir and flip CURRENT to it.
/// Returns the new generation number. Never overwrites a file an existing reader
/// may have mapped (win32-safe): the new generation dir is fresh.
///
/// `content_stamp` is an opaque value (e.g. chunk-row count) stored in the header
/// and checked on open for staleness.
pub fn write_new_generation(
    data_dir: &Path,
    repo: &str,
    index: &VectorIndex,
    content_stamp: u64,
) -> Result<u64> {
    let root = repo_shard_root(data_dir, repo);
    let next = read_current_gen(&root).map(|g| g + 1).unwrap_or(0);
    let generation_dir = root.join(next.to_string());
    std::fs::create_dir_all(&generation_dir)
        .with_context(|| format!("create generation dir {generation_dir:?}"))?;

    let (emb, ids, dim) = index.raw_parts();
    let count = ids.len() as u64;

    // ── shard.f32 : header + row-major f32 ──────────────────────────────────
    let pid = std::process::id();
    let f32_tmp = generation_dir.join(format!("shard.f32.tmp.{pid}"));
    {
        let mut f = std::fs::File::create(&f32_tmp).context("create shard.f32 tmp")?;
        let mut header = [0u8; HEADER_BYTES];
        header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        header[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        header[6..8].copy_from_slice(&(dim as u16).to_le_bytes());
        header[8..16].copy_from_slice(&count.to_le_bytes());
        header[16] = 1; // normalized
        header[24..32].copy_from_slice(&content_stamp.to_le_bytes());
        f.write_all(&header).context("write header")?;
        // f32 payload (already L2-normalized, row-major).
        f.write_all(bytemuck::cast_slice(emb))
            .context("write f32 payload")?;
        f.flush().ok();
        f.sync_all().context("fsync shard.f32")?;
    }
    std::fs::rename(&f32_tmp, generation_dir.join("shard.f32")).context("rename shard.f32")?;

    // ── shard.ids : length-prefixed chunk ids in row order ──────────────────
    let ids_tmp = generation_dir.join(format!("shard.ids.tmp.{pid}"));
    {
        let mut buf: Vec<u8> = Vec::with_capacity(ids.len() * 24);
        buf.extend_from_slice(&count.to_le_bytes());
        for id in ids {
            let fb = id.file.as_bytes();
            buf.extend_from_slice(&(fb.len() as u32).to_le_bytes());
            buf.extend_from_slice(fb);
            buf.extend_from_slice(&id.line_start.to_le_bytes());
            buf.extend_from_slice(&id.line_end.to_le_bytes());
        }
        let mut f = std::fs::File::create(&ids_tmp).context("create shard.ids tmp")?;
        f.write_all(&buf).context("write shard.ids")?;
        f.flush().ok();
        f.sync_all().context("fsync shard.ids")?;
    }
    std::fs::rename(&ids_tmp, generation_dir.join("shard.ids")).context("rename shard.ids")?;

    // Flip CURRENT to the new generation — readers see the new shard from here on; any
    // in-flight reader of the old generation keeps its mapping (old files untouched).
    write_current_gen(&root, next)?;
    Ok(next)
}

/// Open + validate the CURRENT shard for a repo and return an mmap-backed
/// VectorIndex. Returns Ok(None) if there is no usable file (missing / corrupt /
/// dim-mismatch / stale) — the caller then rebuilds from the DB. `expected_stamp`
/// is compared against the header's content stamp for staleness.
pub fn open_current(
    data_dir: &Path,
    repo: &str,
    expected_dim: usize,
    expected_stamp: u64,
) -> Result<Option<(VectorIndex, u64)>> {
    let root = repo_shard_root(data_dir, repo);
    let Some(generation) = read_current_gen(&root) else {
        return Ok(None);
    };
    let generation_dir = root.join(generation.to_string());
    let f32_path = generation_dir.join("shard.f32");
    let ids_path = generation_dir.join("shard.ids");
    if !f32_path.exists() || !ids_path.exists() {
        return Ok(None);
    }

    let file = std::fs::File::open(&f32_path).context("open shard.f32")?;
    // SAFETY: the file is a private cache we wrote; we only ever map it read-only.
    let map = unsafe { memmap2::Mmap::map(&file).context("mmap shard.f32")? };
    if map.len() < HEADER_BYTES {
        return Ok(None);
    }
    let magic = u32::from_le_bytes(map[0..4].try_into().unwrap());
    let version = u16::from_le_bytes(map[4..6].try_into().unwrap());
    let dim = u16::from_le_bytes(map[6..8].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(map[8..16].try_into().unwrap()) as usize;
    let stamp = u64::from_le_bytes(map[24..32].try_into().unwrap());

    // Validate: magic, version, dim, length, alignment, staleness.
    // expected_dim == 0 means "accept the file's own header dim" (fast-path warm,
    // where the model dim isn't known yet); any nonzero value must match exactly.
    if magic != MAGIC || version != FORMAT_VERSION {
        return Ok(None);
    }
    if expected_dim != 0 && dim != expected_dim {
        return Ok(None);
    }
    if dim == 0 {
        return Ok(None);
    }
    let f32_len = count * dim;
    let expect_bytes = HEADER_BYTES + f32_len * std::mem::size_of::<f32>();
    if map.len() != expect_bytes {
        return Ok(None);
    }
    if stamp != expected_stamp {
        return Ok(None); // stale — index changed since this file was built
    }
    // Alignment: the f32 region starts at HEADER_BYTES (64, multiple of 4) and the
    // mmap base is page-aligned, so the region is 4-aligned. bytemuck re-checks.
    let bytes = &map[HEADER_BYTES..HEADER_BYTES + f32_len * std::mem::size_of::<f32>()];
    if bytemuck::try_cast_slice::<u8, f32>(bytes).is_err() {
        return Ok(None);
    }

    // Sidecar: decode chunk ids.
    let ids_bytes = std::fs::read(&ids_path).context("read shard.ids")?;
    let Some(chunk_ids) = decode_ids(&ids_bytes, count) else {
        return Ok(None);
    };

    let index = VectorIndex::from_mmap(map, HEADER_BYTES, f32_len, dim, chunk_ids);
    Ok(Some((index, generation)))
}

fn decode_ids(buf: &[u8], expect_count: usize) -> Option<Vec<ChunkId>> {
    if buf.len() < 8 {
        return None;
    }
    let count = u64::from_le_bytes(buf[0..8].try_into().ok()?) as usize;
    if count != expect_count {
        return None;
    }
    let mut ids = Vec::with_capacity(count);
    let mut p = 8usize;
    for _ in 0..count {
        if p + 4 > buf.len() {
            return None;
        }
        let flen = u32::from_le_bytes(buf[p..p + 4].try_into().ok()?) as usize;
        p += 4;
        if p + flen + 8 > buf.len() {
            return None;
        }
        let file = String::from_utf8(buf[p..p + flen].to_vec()).ok()?;
        p += flen;
        let line_start = u32::from_le_bytes(buf[p..p + 4].try_into().ok()?);
        p += 4;
        let line_end = u32::from_le_bytes(buf[p..p + 4].try_into().ok()?);
        p += 4;
        ids.push(ChunkId {
            file,
            line_start,
            line_end,
        });
    }
    Some(ids)
}

/// Reap stale generation directories for a repo: delete every `<generation>` dir that is
/// NOT in `keep` (the set of gens with live handles, plus CURRENT). Called under
/// the engine's vector-index write lock so it cannot race a reader that is
/// resolving CURRENT + opening a handle. At startup `keep` is just {CURRENT}
/// (no handles survive a restart), reaping all older gens.
pub fn reap_stale_generations(data_dir: &Path, repo: &str, keep: &[u64]) {
    let root = repo_shard_root(data_dir, repo);
    let current = read_current_gen(&root);
    let Ok(entries) = std::fs::read_dir(&root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Ok(generation) = name.parse::<u64>() else {
            continue;
        }; // skip CURRENT, tmp, etc.
        if Some(generation) == current || keep.contains(&generation) {
            continue;
        }
        let _ = std::fs::remove_dir_all(entry.path());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::{ChunkId, VectorIndex};
    use tempfile::TempDir;

    fn build_index(n: usize, dim: usize) -> VectorIndex {
        let mut vi = VectorIndex::new();
        let pairs: Vec<(ChunkId, Vec<f32>)> = (0..n)
            .map(|i| {
                let emb: Vec<f32> = (0..dim).map(|j| ((i * 7 + j) % 13) as f32 - 6.0).collect();
                (
                    ChunkId {
                        file: format!("/r/f{}.rs", i),
                        line_start: i as u32,
                        line_end: i as u32 + 5,
                    },
                    emb,
                )
            })
            .collect();
        vi.insert(&pairs);
        vi
    }

    /// Round-trip: write → mmap-open → search returns BIT-IDENTICAL results to the
    /// in-RAM shard (exactness, no precision loss).
    #[test]
    fn write_then_mmap_roundtrip_is_exact() {
        let tmp = TempDir::new().unwrap();
        let repo = "/proj/round";
        let ram = build_index(200, 64);
        let g = write_new_generation(tmp.path(), repo, &ram, 200).unwrap();
        assert_eq!(g, 0);

        let (mapped, gen_read) = open_current(tmp.path(), repo, 64, 200)
            .unwrap()
            .expect("opens");
        assert_eq!(gen_read, 0);
        assert!(mapped.is_mmap(), "opened shard must be mmap-backed");
        assert_eq!(mapped.len(), ram.len());

        // Same query → identical scored results.
        let q: Vec<f32> = (0..64).map(|j| (j % 5) as f32).collect();
        let r_ram = ram.search(&q, 10);
        let r_map = mapped.search(&q, 10);
        assert_eq!(r_ram.len(), r_map.len());
        for (a, b) in r_ram.iter().zip(&r_map) {
            assert_eq!(a.chunk_id, b.chunk_id, "same chunk order");
            assert!((a.score - b.score).abs() < 1e-6, "bit-identical scores");
        }
    }

    /// The code/docs classification is NOT persisted in the shard file; a shard
    /// cold-warmed from disk (the production path at scale) must re-derive it from
    /// the chunk paths, and scoped search over the mmap-backed shard must honor it.
    #[test]
    fn mmap_opened_shard_classifies_rows_by_scope() {
        use crate::vector::{Scope, ScopeFilter};

        let tmp = TempDir::new().unwrap();
        let repo = "/proj/scoped";
        let mut ram = VectorIndex::new();
        let rows: Vec<(ChunkId, Vec<f32>)> = ["a.rs", "README.md", "b.py", "GUIDE.MD", "c.json"]
            .iter()
            .enumerate()
            .map(|(i, name)| {
                (
                    ChunkId {
                        file: format!("/r/{name}"),
                        line_start: i as u32,
                        line_end: i as u32 + 1,
                    },
                    vec![1.0, 0.1 * i as f32, 0.0, 0.0],
                )
            })
            .collect();
        ram.insert(&rows);
        write_new_generation(tmp.path(), repo, &ram, 5).unwrap();
        let (mapped, _) = open_current(tmp.path(), repo, 4, 5).unwrap().unwrap();
        assert!(mapped.is_mmap());

        let q = [1.0, 0.0, 0.0, 0.0];
        let files = |scope: Scope| -> Vec<String> {
            let f = ScopeFilter::new(scope, ["md"]);
            let mut v: Vec<String> = mapped
                .search_filtered(&q, 10, Some(&f))
                .into_iter()
                .map(|r| r.chunk_id.file)
                .collect();
            v.sort();
            v
        };
        assert_eq!(files(Scope::Docs), vec!["/r/GUIDE.MD", "/r/README.md"]);
        assert_eq!(files(Scope::Code), vec!["/r/a.rs", "/r/b.py", "/r/c.json"]);
    }

    /// mmap shard byte_size excludes the f32 payload (page-cache-resident); only
    /// the small chunk-id sidecar counts against the heap cap.
    #[test]
    fn mmap_shard_byte_size_excludes_payload() {
        let tmp = TempDir::new().unwrap();
        let repo = "/proj/heap";
        let ram = build_index(100, 64);
        let ram_bytes = ram.byte_size();
        let payload = 100 * 64 * std::mem::size_of::<f32>();
        assert!(ram_bytes >= payload, "in-RAM shard counts the f32 payload");
        write_new_generation(tmp.path(), repo, &ram, 100).unwrap();
        let (mapped, _) = open_current(tmp.path(), repo, 64, 100).unwrap().unwrap();
        let map_bytes = mapped.byte_size();
        assert!(
            map_bytes < payload,
            "mmap byte_size excludes the f32 payload ({map_bytes} < {payload})"
        );
        // Sidecar is the only heap cost — bounded by id count, not vector dim.
        assert!(
            map_bytes > 0,
            "sidecar still counts (bounds heap + open-handle pressure)"
        );
    }

    /// Stale stamp → open returns None (forces rebuild).
    #[test]
    fn stale_stamp_returns_none() {
        let tmp = TempDir::new().unwrap();
        let repo = "/proj/stale";
        let ram = build_index(50, 64);
        write_new_generation(tmp.path(), repo, &ram, 50).unwrap(); // built at stamp=50
        // Index now reports a different chunk count (stamp=51) → file is stale.
        assert!(open_current(tmp.path(), repo, 64, 51).unwrap().is_none());
        // Same stamp still opens.
        assert!(open_current(tmp.path(), repo, 64, 50).unwrap().is_some());
    }

    /// Dim mismatch → None.
    #[test]
    fn dim_mismatch_returns_none() {
        let tmp = TempDir::new().unwrap();
        let repo = "/proj/dim";
        let ram = build_index(50, 64);
        write_new_generation(tmp.path(), repo, &ram, 50).unwrap();
        assert!(
            open_current(tmp.path(), repo, 128, 50).unwrap().is_none(),
            "wrong dim rejected"
        );
        assert!(
            open_current(tmp.path(), repo, 0, 50).unwrap().is_some(),
            "dim=0 accepts header dim"
        );
    }

    /// Truncated/corrupt shard.f32 → None (self-heal: caller rebuilds).
    #[test]
    fn corrupt_file_returns_none() {
        let tmp = TempDir::new().unwrap();
        let repo = "/proj/corrupt";
        let ram = build_index(50, 64);
        write_new_generation(tmp.path(), repo, &ram, 50).unwrap();
        // Truncate shard.f32 to half its bytes.
        let root = repo_shard_root(tmp.path(), repo);
        let cur = read_current_gen(&root).unwrap();
        let f = root.join(cur.to_string()).join("shard.f32");
        let len = std::fs::metadata(&f).unwrap().len();
        let file = std::fs::OpenOptions::new().write(true).open(&f).unwrap();
        file.set_len(len / 2).unwrap();
        drop(file);
        assert!(
            open_current(tmp.path(), repo, 64, 50).unwrap().is_none(),
            "truncated file rejected"
        );
    }

    /// Incremental mutation of an MMAP-backed shard must materialize (copy to RAM)
    /// and apply the edit WITHOUT panicking, leaving a correct mutable in-RAM shard
    /// that stays resident (no re-warm). This is the live-editing path: a repo
    /// cold-warmed as mmap, then a watcher fires an incremental.
    #[test]
    fn incremental_on_mmap_shard_materializes_not_panics() {
        let tmp = TempDir::new().unwrap();
        let repo = "/proj/edit";
        let ram = build_index(100, 64);
        write_new_generation(tmp.path(), repo, &ram, 100).unwrap();
        let (mut mapped, _) = open_current(tmp.path(), repo, 64, 100).unwrap().unwrap();
        assert!(mapped.is_mmap(), "starts mmap-backed");
        let before = mapped.len();

        // Remove one file's vectors (an incremental delete) — must materialize, not panic.
        mapped.remove_file("/r/f0.rs");
        assert!(!mapped.is_mmap(), "materialized to in-RAM after mutation");
        assert_eq!(mapped.len(), before - 1, "one row removed");
        assert!(mapped.byte_size() > 0, "now heap-resident (mutable)");

        // Insert a new vector — still works on the materialized shard.
        mapped.insert(&[(
            ChunkId {
                file: "/r/new.rs".into(),
                line_start: 1,
                line_end: 2,
            },
            vec![0.5f32; 64],
        )]);
        assert_eq!(mapped.len(), before, "one removed, one added");

        // Search still returns correct results over the materialized shard.
        let q = vec![0.5f32; 64];
        assert!(!mapped.search(&q, 5).is_empty());
    }

    /// Missing CURRENT (or whole dir) → None (first warm / crash before any write).
    #[test]
    fn missing_current_returns_none() {
        let tmp = TempDir::new().unwrap();
        assert!(
            open_current(tmp.path(), "/proj/never", 64, 0)
                .unwrap()
                .is_none()
        );
    }

    /// A new generation is written to a FRESH dir and CURRENT flips — the old gen's
    /// file is left intact (win32-safe: a reader holding the old mmap is undisturbed).
    #[test]
    fn new_generation_preserves_old_gen_files() {
        let tmp = TempDir::new().unwrap();
        let repo = "/proj/gen";
        let g0 = write_new_generation(tmp.path(), repo, &build_index(30, 64), 30).unwrap();
        // Open + hold the gen-0 mmap (simulates an in-flight reader).
        let (held, _) = open_current(tmp.path(), repo, 64, 30).unwrap().unwrap();
        // Write a new generation (a "full rebuild").
        let g1 = write_new_generation(tmp.path(), repo, &build_index(40, 64), 40).unwrap();
        assert_eq!(g1, g0 + 1, "gen advances");
        let root = repo_shard_root(tmp.path(), repo);
        // CURRENT now points at g1; gen-0's files still exist (held mapping valid).
        assert_eq!(read_current_gen(&root), Some(g1));
        assert!(
            root.join(g0.to_string()).join("shard.f32").exists(),
            "old gen file intact for in-flight reader"
        );
        // The held reader still searches correctly over its (old) mapping.
        let q: Vec<f32> = (0..64).map(|j| (j % 5) as f32).collect();
        assert_eq!(held.search(&q, 5).len(), 5);
        // Reaping with keep={g0} preserves it; reaping with keep={} drops g0 (not CURRENT).
        reap_stale_generations(tmp.path(), repo, &[g0]);
        assert!(root.join(g0.to_string()).exists(), "kept gen survives reap");
        reap_stale_generations(tmp.path(), repo, &[]);
        assert!(
            !root.join(g0.to_string()).exists(),
            "unreferenced old gen reaped"
        );
        assert!(
            root.join(g1.to_string()).exists(),
            "CURRENT gen never reaped"
        );
    }
}
