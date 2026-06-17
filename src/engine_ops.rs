//! Shared engine operations callable by BOTH the HTTP handlers (`server.rs`) and
//! the `bench-query` CLI, so the two never drift.
//!
//! WHY: "Remove Index" and the query pipeline have non-trivial, correctness-
//! critical bodies (atomic generation-bump persistence, optional LLM client
//! construction, repo normalization). The CLI must run the EXACT same logic the
//! server runs. These fns take the individual shared handles (not `AppState`) so
//! the CLI can call them without constructing an axum `State`. The server's
//! handlers are thin wrappers that translate the result into HTTP responses.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::RwLock;

use crate::config::{config_path, write_settings_atomic, Settings};
use crate::embedding::voyage::VoyageClient;
use crate::indexing::IndexEngine;
use crate::llm::LlmClient;
use crate::query::{self, QueryResult};
use crate::store::{self, RepoDbMap};

/// Result of [`remove_index`]: whether the old generation's directory was fully
/// removed before returning, or left draining (reclaimed on next boot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoveOutcome {
    /// The old generation directory is gone on return.
    Removed,
    /// The OS still held the files; the generation bump already redirected future
    /// indexing to a fresh path, so the repo is usable now and the orphan will be
    /// swept on the next restart.
    Pending,
}

/// The "Remove Index" core. Tears down a repo's on-disk index. When
/// `also_drop_repo` is false this is the index-only teardown (the DELETE
/// handler's `remove_repo=false` path and what the CLI uses); when true it ALSO
/// drops the repo from `settings.repos` in the SAME durable write as the
/// generation bump (the handler's `remove_repo=true` "Remove Repo" path).
///
/// This is the SINGLE source of truth for the remove logic — `delete_repo_index`
/// in `server.rs` calls this for both paths. Folding the optional repo-drop in
/// here (rather than as a second write in the handler) preserves the original
/// atomicity: the bump and the repo removal land in ONE `write_settings_atomic`,
/// so a reload mid-teardown can't observe a bumped-but-still-listed repo.
///
/// Steps (order matters — see the inline rationale, a fix for a real incident):
/// 1. `close_repo_db`: cancel in-flight indexing, abort migrations, drop the
///    cached handle so no pipeline holds the RocksDB LOCK.
/// 2. `clear_repo_index`: reset in-memory status + evict the resident shard.
/// 3. Bump the generation (and optionally drop the repo) and persist disk FIRST
///    then memory, BEFORE touching the old directory. Persisting the bump first
///    guarantees a concurrent re-index resolves the NEW (pristine) generation and
///    opens immediately instead of parking behind the slow lock-drain.
/// 4. Remove the OLD generation's directory WITHOUT the open gate (the durable
///    bump means nothing can race or recreate it).
pub async fn remove_index(
    home_dir: &std::path::Path,
    data_dir: &std::path::Path,
    index_engine: &Arc<IndexEngine>,
    settings: &Arc<RwLock<Settings>>,
    repo: &str,
    also_drop_repo: bool,
) -> Result<RemoveOutcome> {
    let normalized_repo = store::normalize_repo_path(repo);

    // Resolve the generation BEFORE the bump — this is the directory currently on
    // disk that we must remove. The read guard is dropped immediately.
    let current_generation = settings.read().await.repo_generation(&normalized_repo);

    // Cancel any in-progress indexing, wait for it to finish, then drop the
    // cached DB handle. This guarantees no pipeline holds a RocksDB lock on
    // the directory when we attempt to remove it.
    index_engine.close_repo_db(&normalized_repo).await;

    // Clear in-memory state immediately — the index is functionally gone from
    // the user's perspective regardless of whether the directory removal succeeds.
    index_engine.clear_repo_index(&normalized_repo).await;

    // Bump the generation and persist (disk FIRST, then memory — mirroring
    // put_config's ordering) BEFORE touching the old directory. Ordering is the fix
    // for a real incident: the old code removed the directory first (a gated, ~30s
    // Windows+Defender lock-drain retry loop) and bumped only afterwards. A re-index
    // triggered during that window read the *old* generation (bump not yet durable)
    // and parked behind the same per-repo open gate the removal held — wedging the UI
    // in an indeterminate "Indexing…" with a dead Cancel, then failing with "open
    // surrealdb" once it recreated the still-draining old path. Persisting the bump
    // first guarantees a concurrent re-index resolves the NEW generation (a pristine
    // path the draining handle never touched) and opens immediately. Persisting to
    // memory before responding also lets the UI's subsequent "Xóa repo" PUT
    // /api/config preserve the bump (it reads repo_generations from the live handle;
    // see put_config).
    let next_generation = current_generation.saturating_add(1);
    let target = config_path(home_dir);
    let to_write = {
        let mut s = settings.read().await.clone();
        s.repo_generations
            .insert(normalized_repo.clone(), next_generation);
        // "Remove Repo": drop it from the configured list in the SAME durable
        // write as the generation bump. Doing it here (not via a follow-up PUT
        // /api/config) makes the removal durable BEFORE the slow lock-drain below,
        // so a reload mid-teardown — or a lost PUT — can't resurrect the repo from
        // disk. The generation entry is intentionally KEPT (see repo_generations
        // doc) so a future re-add reuses the higher generation instead of racing
        // the still-draining old LOCK.
        if also_drop_repo {
            s.repos.retain(|r| r != &normalized_repo);
        }
        s
    };
    tokio::task::spawn_blocking({
        let to_write = to_write.clone();
        move || write_settings_atomic(&target, &to_write)
    })
    .await
    .context("internal error joining settings-persist task")?
    .context("failed to persist index generation")?;

    // Disk write succeeded — now swap memory under the write lock so the live
    // handle matches disk (GET /api/config reads disk; in-memory is the source of
    // truth for indexing triggers and the put_config diff).
    {
        let mut guard = settings.write().await;
        guard
            .repo_generations
            .insert(normalized_repo.clone(), next_generation);
        if also_drop_repo {
            guard.repos.retain(|r| r != &normalized_repo);
        }
    }

    // Now remove the OLD generation's directory WITHOUT the open gate. The bump above
    // is durable, so every future open targets `next_generation` — nothing can race
    // or recreate `current_generation`, and there is nothing to serialize against. A
    // gate-held removal here would block the fresh generation's open for the whole
    // ~30s drain (the gate is keyed by repo, not generation); the ungated removal lets
    // a re-index proceed on the clean path immediately while this drains in the
    // foreground. If it outlives the retry budget, the boot-time sweep reclaims it
    // (store::sweep_stale_generations).
    let removed =
        store::remove_old_generation_dir(data_dir, &normalized_repo, current_generation).await;

    Ok(if removed {
        RemoveOutcome::Removed
    } else {
        RemoveOutcome::Pending
    })
}

/// The query pipeline core: build the embedding client, build the LLM reranker
/// (only when `rerank` is set), normalize the repo, and run `query::run_query`
/// with every settings-derived argument. SINGLE source of truth for query
/// behavior — `post_query` in `server.rs` calls this, and so does the CLI, so
/// both produce byte-identical retrieval.
///
/// `settings` is an owned snapshot taken by the caller (the server drops its
/// read guard before calling; the CLI clones from its handle) so no settings
/// lock is held across the `.await`s here.
///
/// Pre-flight policy checks (no repos configured, no embedding keys, repo
/// required) stay in the HTTP handler so it can return the exact 400 bodies the
/// API contract specifies; this fn assumes `repo` is a non-empty workspace path.
#[allow(clippy::too_many_arguments)]
pub async fn run_query_op(
    settings: &Settings,
    index_engine: &Arc<IndexEngine>,
    repo_dbs: &RepoDbMap,
    repo: &str,
    query_text: &str,
    top_k: usize,
    rerank: bool,
) -> Result<QueryResult> {
    // Build voyage client.
    let voyage_client = VoyageClient::new(
        settings.embedding.model.clone(),
        settings.embedding.api_keys.clone(),
        settings.embedding.voyage_base_url.as_deref(),
    )
    .context("failed to create embedding client")?;

    // Build LLM client for reranking (None if no keys configured or rerank disabled).
    let llm_client = if rerank {
        LlmClient::new(&settings.llm)
    } else {
        None
    };

    let top_k = top_k.max(1);

    // Queries are always scoped to one repository.
    let repo_filter = store::normalize_repo_path(repo);

    query::run_query(
        query_text,
        top_k,
        Some(&repo_filter),
        &voyage_client,
        index_engine,
        repo_dbs,
        settings.llm.rerank_min_prune_lines,
        llm_client.as_ref(),
        Duration::from_secs(settings.mcp_index_wait_secs),
        settings.llm.agentic_rag,
        settings.llm.agentic_rag_max_turns,
        settings.llm.agentic_rag_max_chunk_chars,
        settings.llm.agentic_rag_grep_read,
    )
    .await
}
