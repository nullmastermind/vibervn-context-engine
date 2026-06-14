//! `bench-query` — a self-contained CLI that boots the engine through the SAME
//! shared path the HTTP server uses (`engine_boot::boot_engine`) and runs
//! remove/rebuild/incremental/query through the SAME shared logic
//! (`engine_ops::remove_index`, `IndexEngine::trigger_rebuild` /
//! `trigger_index`, `engine_ops::run_query_op`). No behavioral drift from the
//! server: same engine, same ops.
//!
//! WHY a CLI: it lets you rebuild + query the REAL on-disk index without
//! starting the always-on server — useful as a retrieval bench (clean rebuild +
//! raw, no-rerank query) and as a one-shot query tool for users.
//!
//! It defaults to the SHARED real data dir (env > settings > the builtin
//! `~/.vibervn/context-engine`), exactly like the server. RocksDB takes an
//! exclusive per-directory lock, so it CANNOT run against the same data dir as a
//! live server — that conflict surfaces on first DB open during index/query and
//! is reported with clear guidance (stop the server or pass `--data-dir`).
//!
//! Usage:
//!   bench-query --repo <PATH> --query <TEXT> [--top-k N] [--data-dir PATH]
//!               [--rebuild-index] [--rerank]

use std::time::{Duration, Instant};

use clap::Parser;
use tracing_subscriber::EnvFilter;

use context_engine_rs::engine_boot::{boot_engine, BootOptions, BootedEngine};
use context_engine_rs::engine_ops;
use context_engine_rs::indexing::IndexState;
use context_engine_rs::store;

#[derive(Parser, Debug)]
#[command(
    name = "bench-query",
    about = "Rebuild + query a repo's index through the same engine the server uses"
)]
struct Cli {
    /// Workspace path to query (required).
    #[arg(long)]
    repo: String,

    /// The query string (required).
    #[arg(long)]
    query: String,

    /// Number of results to return.
    #[arg(long, default_value_t = 30)]
    top_k: usize,

    /// Data directory base override. Defaults to the SAME precedence as the
    /// server (env CONTEXT_ENGINE_DATA_DIR > Settings.data_dir > builtin
    /// `~/.vibervn/context-engine`) — i.e. the REAL shared index, intentionally,
    /// so the CLI queries the index the server built. Pass a separate dir to run
    /// alongside a live server (RocksDB's exclusive per-dir lock forbids sharing).
    #[arg(long)]
    data_dir: Option<std::path::PathBuf>,

    /// Clean rebuild: remove the existing index, then full rebuild. When NOT set,
    /// runs an incremental update only (no remove).
    #[arg(long, default_value_t = false)]
    rebuild_index: bool,

    /// Run the query through the LLM rerank flow. Off by default (raw retrieval),
    /// which is what the bench uses; users pass --rerank for the reranked path.
    #[arg(long, default_value_t = false)]
    rerank: bool,
}

/// Overall cap on how long we wait for indexing to finish before giving up.
/// Generous — a clean rebuild of a large repo is network-embed-bound. The bench
/// target (notepad-ade) finishes in seconds; this only guards a wedged run.
const INDEX_WAIT_CAP: Duration = Duration::from_secs(30 * 60);

/// True if an error string looks like a RocksDB exclusive-lock / open conflict —
/// the signal that another process (the server?) holds the data dir.
fn looks_like_lock_conflict(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    m.contains("lock") || m.contains("could not open") || m.contains("open surreal")
}

/// Print the standard lock-conflict guidance and return the non-zero exit code.
fn lock_conflict_guidance(data_dir: &std::path::Path, detail: &str) -> i32 {
    eprintln!(
        "error: could not open index DB at {} — another process (the context-engine \
         server?) may be running on the same data dir. Stop the server or pass \
         --data-dir <other>.\n  detail: {detail}",
        data_dir.display()
    );
    3
}

#[tokio::main]
async fn main() {
    std::process::exit(run().await);
}

async fn run() -> i32 {
    // Tracing: keep CLI stderr readable (warnings only by default). Result output
    // goes to stdout via println!, NOT through tracing. Lives here (not in
    // boot_engine) so each binary owns its own filter and tracing inits once.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("context_engine_rs=warn,warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let repo = store::normalize_repo_path(&cli.repo);

    // Boot the engine through the shared path. set_rocksdb_memory_bounds runs at
    // the top of boot_engine, before any datastore opens.
    let BootedEngine {
        home_dir,
        data_dir,
        index_engine,
        repo_dbs,
        settings,
        ..
    } = match boot_engine(BootOptions {
        data_dir: cli.data_dir.clone(),
        embeddings_dir: None,
    })
    .await
    {
        Ok(b) => b,
        Err(e) => {
            let detail = format!("{e:#}");
            // boot_engine only starts IndexEngine; it doesn't open a per-repo DB,
            // so a lock conflict normally surfaces later. But classify here too in
            // case a future change makes boot touch a datastore. We don't have the
            // resolved data_dir on this path (destructure failed), so report the
            // CLI override if given, else a generic hint.
            if looks_like_lock_conflict(&detail) {
                let hint = cli
                    .data_dir
                    .clone()
                    .unwrap_or_else(|| std::path::PathBuf::from("<default data dir>"));
                return lock_conflict_guidance(&hint, &detail);
            }
            eprintln!("error: failed to boot engine: {detail}");
            return 2;
        }
    };

    eprintln!("booted engine (data_dir = {})", data_dir.display());

    // Register the repo so it has a status entry + filesystem watcher (no-op if
    // already known). This is the same primitive the server calls for a new repo.
    index_engine.register_repo(&repo).await;

    // NOTE: we deliberately do NOT add the repo to settings.repos. The CLI is a
    // transient tool and must not mutate the user's configured repo list — and on
    // the --rebuild-index path, engine_ops::remove_index clones the live settings
    // and bump-writes them to the home-anchored settings.json, so pushing here
    // would PERMANENTLY persist the repo into the user's real config. Nothing
    // between here and the query needs it: register_repo seeds the status entry +
    // watcher, trigger_rebuild/trigger_index carry the repo in the trigger message,
    // run_consumer reads the repo from the trigger and resolves its generation from
    // repo_generations (never from the .repos list), and run_query_op takes the
    // repo directly and does NOT run the server's "no repos configured" 400 guard
    // (that lives only in the post_query HTTP handler).

    // Capture the pre-trigger status so we can detect the Indexing→Idle transition
    // with a FRESH last_indexed_at (the same "done" definition scripts/ab_bench.sh
    // uses against /api/repos/:id/status).
    let pre = index_engine.repo_status(&repo).await;
    let pre_last_indexed = pre.as_ref().and_then(|s| s.last_indexed_at);

    // Trigger the index run.
    if cli.rebuild_index {
        eprintln!("rebuild-index: removing existing index, then full rebuild...");
        // Shared "Remove Index" (index-only teardown; does NOT drop the repo from
        // settings — also_drop_repo = false), identical to the server's
        // remove_repo=false DELETE path.
        match engine_ops::remove_index(&home_dir, &data_dir, &index_engine, &settings, &repo, false)
            .await
        {
            Ok(engine_ops::RemoveOutcome::Removed) => {}
            Ok(engine_ops::RemoveOutcome::Pending) => {
                eprintln!(
                    "note: old index directory not fully removed yet (OS lock drain); \
                     the generation bump already redirected indexing to a fresh path."
                );
            }
            Err(e) => {
                let detail = format!("{e:#}");
                if looks_like_lock_conflict(&detail) {
                    return lock_conflict_guidance(&data_dir, &detail);
                }
                eprintln!("error: remove-index failed: {detail}");
                return 2;
            }
        }
        if let Err(e) = index_engine.trigger_rebuild(&repo).await {
            eprintln!("error: failed to trigger rebuild: {e:#}");
            return 2;
        }
    } else {
        eprintln!("incremental: triggering index update...");
        if let Err(e) = index_engine.trigger_index(&repo).await {
            eprintln!("error: failed to trigger index: {e:#}");
            return 2;
        }
    }

    // Poll until done. "Done" = state == Idle AND last_indexed_at is fresh
    // (changed, or None→Some). An Error state aborts with the captured context.
    if let Some(code) = wait_for_index(&index_engine, &repo, pre_last_indexed, &data_dir).await {
        return code;
    }

    // Run the query through the SHARED op so retrieval is byte-identical to the
    // server's /api/query. Take an owned settings snapshot first (no guard held
    // across the await).
    let settings_snapshot = settings.read().await.clone();
    let result = match engine_ops::run_query_op(
        &settings_snapshot,
        &index_engine,
        &repo_dbs,
        &repo,
        &cli.query,
        cli.top_k,
        cli.rerank,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            let detail = format!("{e:#}");
            if looks_like_lock_conflict(&detail) {
                return lock_conflict_guidance(&data_dir, &detail);
            }
            eprintln!("error: query failed: {detail}");
            return 2;
        }
    };

    print_results(&cli, &result);

    // Oracle: non-empty results → success. Empty + warming → shard not resident
    // yet (shouldn't happen post-rebuild, but don't silently pass). Empty + not
    // warming → genuine miss = bench failure.
    if !result.results.is_empty() {
        0
    } else if result.warming {
        eprintln!(
            "FAIL: no results — the repo's vector shard is still warming into RAM. \
             Retry the query; this should not happen immediately after a rebuild."
        );
        4
    } else {
        eprintln!(
            "FAIL: no results for query {:?} against repo {} — the index returned nothing.",
            cli.query, repo
        );
        5
    }
}

/// Poll `repo_status` until the run is done or fails. Returns `None` on success,
/// or `Some(exit_code)` on failure/timeout (already printed). "Done" mirrors
/// scripts/ab_bench.sh: state Idle AND last_indexed_at fresh relative to `pre`.
async fn wait_for_index(
    index_engine: &context_engine_rs::indexing::IndexEngine,
    repo: &str,
    pre_last_indexed: Option<chrono::DateTime<chrono::Utc>>,
    data_dir: &std::path::Path,
) -> Option<i32> {
    let start = Instant::now();
    let mut last_progress_log = Instant::now();
    loop {
        let status = index_engine.repo_status(repo).await;
        match status {
            Some(s) => match s.state {
                IndexState::Idle => {
                    // Fresh completion: last_indexed_at changed (or went None→Some).
                    let fresh = match (pre_last_indexed, s.last_indexed_at) {
                        (_, None) => false, // never indexed yet — keep waiting
                        (None, Some(_)) => true,
                        (Some(prev), Some(now)) => now > prev,
                    };
                    if fresh {
                        eprintln!(
                            "index done in {:.1}s ({} files indexed)",
                            start.elapsed().as_secs_f64(),
                            s.indexed_files
                        );
                        return None;
                    }
                    // Idle but not yet fresh: the trigger may not have been picked
                    // up by the consumer yet. Keep polling within the cap.
                }
                IndexState::Indexing => { /* in progress — keep polling */ }
                IndexState::Error => {
                    let detail = s.error.unwrap_or_else(|| "unknown indexing error".to_string());
                    if looks_like_lock_conflict(&detail) {
                        return Some(lock_conflict_guidance(data_dir, &detail));
                    }
                    eprintln!("error: indexing failed for repo {repo}: {detail}");
                    return Some(2);
                }
            },
            None => {
                // No status entry at all — register_repo should have seeded one.
                eprintln!("error: no index status for repo {repo} (not registered?)");
                return Some(2);
            }
        }

        if start.elapsed() > INDEX_WAIT_CAP {
            eprintln!(
                "error: indexing did not complete within {}s for repo {repo}",
                INDEX_WAIT_CAP.as_secs()
            );
            return Some(2);
        }

        // Periodic progress so a long rebuild isn't silent.
        if last_progress_log.elapsed() >= Duration::from_secs(10) {
            eprintln!("indexing... ({}s)", start.elapsed().as_secs());
            last_progress_log = Instant::now();
        }

        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// Print each result as `path#Lstart-end  score=<score>` followed by the
/// numbered content, then the query timing breakdown.
fn print_results(cli: &Cli, result: &context_engine_rs::query::QueryResult) {
    println!(
        "\n=== {} result(s) for {:?} (top_k={}, rerank={}) ===",
        result.results.len(),
        cli.query,
        cli.top_k,
        cli.rerank
    );
    for (i, r) in result.results.iter().enumerate() {
        println!(
            "\n[{}] {}#L{}-{}  score={:.4}{}",
            i + 1,
            r.file,
            r.line_start,
            r.line_end,
            r.score,
            r.symbol
                .as_deref()
                .map(|s| format!("  symbol={s}"))
                .unwrap_or_default()
        );
        println!("{}", r.content);
    }

    let t = &result.timing;
    println!(
        "\n--- timing (ms): embed={} search={} graph={} merge={} rerank={} total={} ---",
        t.embed_ms, t.search_ms, t.graph_ms, t.merge_ms, t.rerank_ms, t.total_ms
    );
}
