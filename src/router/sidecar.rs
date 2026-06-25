//! Per-repo SIDECAR metadata: a tiny JSON file the worker writes after each
//! index completion, that the ROUTER reads to render the repo list / cold
//! `/index-stats` / `/graph` WITHOUT opening RocksDB (and without spawning a
//! worker just to display a count).
//!
//! ## Why a sidecar (not the DB)
//!
//! `graph_cache` / `stats_cache` live INSIDE each repo's RocksDB (`index_meta`).
//! Reading them needs an exclusive-LOCK DB open — exactly the eager-open the
//! process-per-project design exists to avoid. So the router cannot read them
//! for a cold repo. The sidecar is the router-readable channel: a few numbers +
//! a timestamp, written by the worker (which already has the DB open) at index
//! completion.
//!
//! ## Deliberately LIGHT
//!
//! It carries ONLY what the cold UI needs: file count, last-indexed time, state,
//! embedding model/dim. It does NOT copy `graph_cache` / `stats_cache` — those
//! can be large at kernel scale and the full graph/stats views are served by
//! proxying to a (now-spawned) worker. Keeping the sidecar to fixed-size scalars
//! means no scale concern and a trivially-validated shape.
//!
//! ## Crash-safety / corruption
//!
//! Written with the project's standard atomic tempfile+rename (`config.rs`
//! pattern), so the router never observes a half-written file. A
//! missing/corrupt/old-shape file is NOT an error: the reader returns `None` and
//! the router renders a "not indexed / unknown" placeholder, mirroring how
//! `store::ops::get_cached_graph` treats a cold/corrupt cache as `Ok(None)`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::store::sanitize_repo_name;

/// Light, fixed-shape per-repo metadata for cold (no-worker) display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoSidecar {
    /// Indexed file count (from `stats_cache.files` / `count_indexed_files`).
    pub file_count: u64,
    /// RFC3339 last successful index completion, or None if never indexed.
    pub last_indexed_at: Option<String>,
    /// Coarse state string for the UI: "indexed" once a successful run exists.
    /// (Live "indexing"/"error" state belongs to a running worker; the sidecar
    /// only records the durable last-known-good.)
    pub state: String,
    /// Embedding model name the index was built with.
    pub embedding_model: String,
    /// Embedding dimension (0 if unknown).
    pub embedding_dim: u64,
    /// Sidecar shape version — bump if the shape changes so an old file is
    /// treated as `None` (placeholder) rather than mis-parsed.
    pub schema: u32,
}

/// Current sidecar shape version. A file with a different `schema` is treated as
/// absent (cold placeholder), never mis-read.
pub const SIDECAR_SCHEMA: u32 = 1;

/// Directory holding sidecar files: `<data_dir>/sidecar/`.
fn sidecar_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("sidecar")
}

/// Path to a repo's sidecar file: `<data_dir>/sidecar/<sanitized-repo>.json`.
/// Keyed by the SAME `sanitize_repo_name` used for the RocksDB dir so the router
/// and worker agree on the filename.
pub fn sidecar_path(data_dir: &Path, repo: &str) -> PathBuf {
    sidecar_dir(data_dir).join(format!("{}.json", sanitize_repo_name(repo)))
}

/// Write a repo's sidecar atomically (tempfile + fsync + rename), matching the
/// crash-safe pattern used for `settings.json` and the vector shard files. A
/// half-written file never appears under the real name. Best-effort at the call
/// site: a write failure must not fail an index run (the worker logs + moves on;
/// the router falls back to its cold placeholder).
pub fn write_sidecar(data_dir: &Path, repo: &str, meta: &RepoSidecar) -> Result<()> {
    let dir = sidecar_dir(data_dir);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create sidecar dir {}", dir.display()))?;
    let target = sidecar_path(data_dir, repo);

    let json = serde_json::to_vec_pretty(meta).context("serialize sidecar")?;

    // Atomic write: tempfile in the SAME dir → write → fsync → persist (rename).
    let mut tmp = tempfile::NamedTempFile::new_in(&dir).context("create sidecar tempfile")?;
    {
        use std::io::Write as _;
        tmp.write_all(&json).context("write sidecar tempfile")?;
        tmp.as_file().sync_all().context("fsync sidecar tempfile")?;
    }
    tmp.persist(&target)
        .with_context(|| format!("persist sidecar {}", target.display()))?;
    Ok(())
}

/// Read a repo's sidecar. Returns `Ok(None)` when the file is absent, unreadable,
/// corrupt, or a different `schema` — in all those cases the router renders the
/// cold "not indexed / unknown" placeholder rather than erroring. Only an
/// unexpected IO error (other than not-found) propagates.
pub fn read_sidecar(data_dir: &Path, repo: &str) -> Option<RepoSidecar> {
    let path = sidecar_path(data_dir, repo);
    let bytes = std::fs::read(&path).ok()?;
    let meta: RepoSidecar = serde_json::from_slice(&bytes).ok()?;
    if meta.schema != SIDECAR_SCHEMA {
        return None; // old/unknown shape → treat as cold
    }
    Some(meta)
}

// ── Auxiliary sidecar payloads (graph + files list) ──────────────────────────
//
// The light `meta.json` (above) is polled frequently (every 3s by the UI's
// index-status loop), so it stays fixed-shape + tiny. The graph and files-list
// payloads can be larger (graph ~hundreds of KB at scale) and are read ONLY when
// a repo detail is opened — so they live in SEPARATE sibling files keyed by the
// same sanitized name: `<sanitized>.graph.json` / `<sanitized>.files.json`. This
// keeps the frequent meta poll from ever dragging the heavy graph payload, and
// lets the router serve a cold (possibly stale — accepted) graph/files view on
// detail-open WITHOUT spawning a worker.

/// Path to an auxiliary sidecar file: `<data_dir>/sidecar/<sanitized>.<kind>.json`.
fn aux_path(data_dir: &Path, repo: &str, kind: &str) -> PathBuf {
    sidecar_dir(data_dir).join(format!("{}.{}.json", sanitize_repo_name(repo), kind))
}

/// Write an arbitrary JSON-serializable payload to a repo's `<kind>` sidecar,
/// using the SAME atomic tempfile+fsync+rename pattern as `write_sidecar`. Used
/// for the `graph` and `files` payloads. Best-effort at the call site — a write
/// failure must not fail an index run.
pub fn write_aux_json<T: serde::Serialize>(
    data_dir: &Path,
    repo: &str,
    kind: &str,
    value: &T,
) -> Result<()> {
    let dir = sidecar_dir(data_dir);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create sidecar dir {}", dir.display()))?;
    let target = aux_path(data_dir, repo, kind);
    let json = serde_json::to_vec(value).context("serialize aux sidecar")?;
    let mut tmp = tempfile::NamedTempFile::new_in(&dir).context("create aux tempfile")?;
    {
        use std::io::Write as _;
        tmp.write_all(&json).context("write aux tempfile")?;
        tmp.as_file().sync_all().context("fsync aux tempfile")?;
    }
    tmp.persist(&target)
        .with_context(|| format!("persist aux sidecar {}", target.display()))?;
    Ok(())
}

/// Read + deserialize a repo's `<kind>` aux sidecar. Returns `None` when absent,
/// unreadable, or corrupt — the caller serves a cold/empty placeholder, never
/// errors (mirrors `read_sidecar` / `store::ops::get_cached_graph` degradation).
pub fn read_aux_json<T: serde::de::DeserializeOwned>(
    data_dir: &Path,
    repo: &str,
    kind: &str,
) -> Option<T> {
    let bytes = std::fs::read(aux_path(data_dir, repo, kind)).ok()?;
    serde_json::from_slice::<T>(&bytes).ok()
}

/// Remove ALL of a repo's sidecar files (meta + graph + files). Called on repo /
/// index removal so the router's cold view doesn't keep serving stale data for a
/// repo whose index was just deleted. Best-effort: a file that isn't there (or
/// can't be removed) is ignored — the next index rewrites them anyway.
pub fn remove_all_sidecars(data_dir: &Path, repo: &str) {
    let _ = std::fs::remove_file(sidecar_path(data_dir, repo));
    let _ = std::fs::remove_file(aux_path(data_dir, repo, "graph"));
    let _ = std::fs::remove_file(aux_path(data_dir, repo, "files"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample() -> RepoSidecar {
        RepoSidecar {
            file_count: 42,
            last_indexed_at: Some("2026-06-25T00:00:00+00:00".to_string()),
            state: "indexed".to_string(),
            embedding_model: "voyage-code-3".to_string(),
            embedding_dim: 1024,
            schema: SIDECAR_SCHEMA,
        }
    }

    #[test]
    fn roundtrip_write_then_read() {
        let dir = TempDir::new().unwrap();
        let repo = r"d:\projects\rust\foo";
        write_sidecar(dir.path(), repo, &sample()).unwrap();
        let got = read_sidecar(dir.path(), repo).expect("sidecar should read back");
        assert_eq!(got.file_count, 42);
        assert_eq!(got.embedding_dim, 1024);
        assert_eq!(got.state, "indexed");
    }

    #[test]
    fn missing_file_reads_as_none() {
        let dir = TempDir::new().unwrap();
        assert!(read_sidecar(dir.path(), r"d:\never\indexed").is_none());
    }

    #[test]
    fn corrupt_file_reads_as_none_not_error() {
        let dir = TempDir::new().unwrap();
        let repo = r"d:\projects\rust\bar";
        // Write a valid sidecar, then clobber it with garbage (simulating a
        // worker that died mid-write — though the atomic rename prevents that,
        // this proves the reader degrades on ANY corruption).
        write_sidecar(dir.path(), repo, &sample()).unwrap();
        std::fs::write(sidecar_path(dir.path(), repo), b"{ not valid json").unwrap();
        assert!(
            read_sidecar(dir.path(), repo).is_none(),
            "corrupt sidecar must read as None (cold placeholder), never panic/error"
        );
    }

    #[test]
    fn wrong_schema_reads_as_none() {
        let dir = TempDir::new().unwrap();
        let repo = r"d:\projects\rust\baz";
        let mut m = sample();
        m.schema = 999; // future/unknown shape
        write_sidecar(dir.path(), repo, &m).unwrap();
        assert!(
            read_sidecar(dir.path(), repo).is_none(),
            "unknown schema must read as None so an old shape is never mis-read"
        );
    }
}
