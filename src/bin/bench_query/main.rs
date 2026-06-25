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
//!   bench-query --repo <PATH> --diagnose-graph [--data-dir PATH]   # oracle only

use std::time::{Duration, Instant};

use clap::Parser;
use tracing_subscriber::EnvFilter;

use context_engine_rs::engine_boot::{BootOptions, BootedEngine, boot_engine};
use context_engine_rs::engine_ops;
use context_engine_rs::indexing::IndexState;
use context_engine_rs::store;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;

#[derive(Parser, Debug)]
#[command(
    name = "bench-query",
    about = "Rebuild + query a repo's index through the same engine the server uses"
)]
struct Cli {
    /// Workspace path to query (required).
    #[arg(long)]
    repo: String,

    /// The query string. Required for a normal run; ignored (and optional) when
    /// `--diagnose-graph` is set, since the diagnostic does not run a query.
    #[arg(long)]
    query: Option<String>,

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

    /// Diagnostic mode (PHASE-2 oracle, no fix): instead of running the query,
    /// open the repo's REAL on-disk DB and print the EXPLAIN plan + index-usage
    /// verdict for the three hot predicates `graph_expand` issues (calls callers,
    /// calls callees, symbol overlap), plus table row counts and `INFO FOR TABLE`
    /// so the human can SEE whether `in_name`/`out_name` are served by an index or
    /// a full table scan. Skips the normal query/print path entirely. Does NOT
    /// change any query logic — it only measures. Exits non-zero if any verdict
    /// could not be determined.
    #[arg(long, default_value_t = false)]
    diagnose_graph: bool,
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

    // --query is required for a normal run, but meaningless under --diagnose-graph
    // (the diagnostic IS the whole job and runs no query). Validate here rather
    // than via clap's `required` so the two modes can share one struct.
    if !cli.diagnose_graph && cli.query.is_none() {
        eprintln!("error: --query <TEXT> is required (unless running --diagnose-graph)");
        return 2;
    }

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
        // bench-query is read-only — it never mutates files, so a boot watcher
        // has nothing to race. Keep the production default.
        no_watchers: false,
        only_repo: None,
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

    // --diagnose-graph: the index is ready, so open the REAL on-disk DB for this
    // repo and run the EXPLAIN-based oracle instead of a query. This is the whole
    // job for that invocation — we never touch the query/print path below.
    if cli.diagnose_graph {
        // Resolve the generation the SAME way the engine does (settings is the
        // source of truth) so we open the directory the index actually wrote.
        let generation = settings.read().await.repo_generation(&repo);
        // Mandatory one-handle-per-repo access: reuse the engine's cached handle
        // if present (the index run above opened it), else open through the same
        // gated store API. NEVER a second raw Surreal::new on the same path.
        let db = match store::get_or_open(&repo_dbs, &data_dir, &repo, generation).await {
            Ok(db) => db,
            Err(e) => {
                let detail = format!("{e:#}");
                if looks_like_lock_conflict(&detail) {
                    return lock_conflict_guidance(&data_dir, &detail);
                }
                eprintln!("error: could not open index DB for diagnostics: {detail}");
                return 2;
            }
        };
        return diagnose_graph(&db, &repo).await;
    }

    // Run the query through the SHARED op so retrieval is byte-identical to the
    // server's /api/query. Take an owned settings snapshot first (no guard held
    // across the await).
    let settings_snapshot = settings.read().await.clone();
    // Safe: the non-diagnose path required --query (validated at startup).
    let query_text = cli.query.as_deref().unwrap_or_default();
    let result = match engine_ops::run_query_op(
        &settings_snapshot,
        &index_engine,
        &repo_dbs,
        &repo,
        query_text,
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

    print_results(query_text, &cli, &result);

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
            "FAIL: no results for query {query_text:?} against repo {repo} — the index returned nothing."
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
                    let detail = s
                        .error
                        .unwrap_or_else(|| "unknown indexing error".to_string());
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
fn print_results(query_text: &str, cli: &Cli, result: &context_engine_rs::query::QueryResult) {
    println!(
        "\n=== {} result(s) for {:?} (top_k={}, rerank={}) ===",
        result.results.len(),
        query_text,
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

// ── PHASE-2 ORACLE: graph_expand index-usage diagnostic ────────────────────
//
// This is measurement-only. It issues the SAME three hot predicates that
// `query/graph_expand.rs` runs during a real query and prints, for each, the
// SurrealDB EXPLAIN plan plus a verdict on whether the plan is served by a
// secondary index or a full table scan. It also prints row counts and
// `INFO FOR TABLE` for `calls`/`symbol` so the human can SEE the indexes that
// are actually present on disk. It changes NO query/schema/graph_expand code.
//
// SurrealDB 2.6.5 takes the EXPLAIN as a *suffix* on the SELECT (this matches the
// in-crate tests in store/mod.rs); the plan deserializes as `Vec<serde_json::Value>`.

/// Walk an EXPLAIN plan (a JSON array of operation rows) and classify it.
/// Returns `(uses_index, detail)` where `detail` names the index when one is
/// used, or describes the scan otherwise. `found` is false only when neither an
/// index nor a table iteration could be located (verdict undetermined).
fn classify_plan(plan: &[serde_json::Value]) -> (bool, bool, String) {
    let plan_str = serde_json::to_string(plan).unwrap_or_default();
    if plan_str.contains("Iterate Index") {
        let idx = plan
            .iter()
            .find_map(find_index_name)
            .unwrap_or_else(|| "<unnamed>".to_string());
        (true, true, format!("Iterate Index {idx}"))
    } else if plan_str.contains("Iterate Table") {
        (false, true, "Iterate Table (FULL TABLE SCAN)".to_string())
    } else {
        (
            false,
            false,
            "no Iterate Index / Iterate Table operation found".to_string(),
        )
    }
}

/// Recursively search an EXPLAIN JSON tree for the `index` field that
/// SurrealDB attaches to an "Iterate Index" operation's detail block.
fn find_index_name(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::Object(map) => {
            if let Some(serde_json::Value::String(s)) = map.get("index") {
                return Some(s.clone());
            }
            map.values().find_map(find_index_name)
        }
        serde_json::Value::Array(arr) => arr.iter().find_map(find_index_name),
        _ => None,
    }
}

/// Print a section header to stdout.
fn section(title: &str) {
    println!("\n========== {title} ==========");
}

/// Actually EXECUTE a query (NOT EXPLAIN) once, timing it with a wall clock, and
/// print `EXEC: <n> rows in <ms> ms`. This is the proof that complements the
/// plan: a full table scan shows up here as seconds, a primary-key / index fetch
/// as sub-millisecond. Takes the already-bound query as an `IntoFuture` (a
/// `db.query(..).bind(..)` chain) so each caller supplies its own real binds.
/// Returns true on success, false if the execution itself errored.
async fn time_exec(
    label: &str,
    fut: impl std::future::IntoFuture<Output = surrealdb::Result<surrealdb::Response>>,
) -> bool {
    let start = Instant::now();
    match fut.await {
        Ok(mut resp) => {
            // Count rows from the first (only) statement. We don't care about the
            // row shape here — just that the query ran and how long it took.
            let rows: Vec<serde_json::Value> = resp.take(0).unwrap_or_default();
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            println!("EXEC: {} rows in {ms:.3} ms", rows.len());
            true
        }
        Err(e) => {
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("error: EXEC ({label}) failed after {ms:.3} ms: {e}");
            false
        }
    }
}

/// Run `INFO FOR TABLE <table>` and pretty-print it. Returns Err on failure so
/// the caller can record an undetermined verdict.
async fn print_table_info(db: &Surreal<Db>, table: &str) -> Result<(), String> {
    let info: Option<serde_json::Value> = db
        .query(format!("INFO FOR TABLE {table}"))
        .await
        .map_err(|e| format!("INFO FOR TABLE {table} failed: {e}"))?
        .take(0)
        .map_err(|e| format!("take INFO FOR TABLE {table} row failed: {e}"))?;
    match info {
        Some(v) => {
            println!(
                "INFO FOR TABLE {table}:\n{}",
                serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
            );
            Ok(())
        }
        None => Err(format!("INFO FOR TABLE {table} returned no row")),
    }
}

/// Run `SELECT count() FROM <table> GROUP ALL` and print the count. Returns the
/// count on success.
async fn print_row_count(db: &Surreal<Db>, table: &str) -> Result<i64, String> {
    let row: Option<serde_json::Value> = db
        .query(format!("SELECT count() FROM {table} GROUP ALL"))
        .await
        .map_err(|e| format!("count {table} failed: {e}"))?
        .take(0)
        .map_err(|e| format!("take count {table} failed: {e}"))?;
    let count = row
        .as_ref()
        .and_then(|v| v.get("count"))
        .and_then(|c| c.as_i64())
        .unwrap_or(0);
    println!("row count for `{table}`: {count}");
    Ok(count)
}

/// The oracle. Opens nothing (takes an already-opened handle), runs the hot
/// predicates with EXPLAIN against the REAL on-disk DB, prints plans + verdicts +
/// context (counts, INFO FOR TABLE), and — for each predicate — actually EXECUTES
/// the query once with a wall-clock timer (`EXEC: <n> rows in <ms> ms`) so plan
/// claims are backed by measured latency. Predicates 1-3 are the calls/symbol
/// predicates `graph_expand` issues; predicates 4-6 are the two `fetch_chunk_for_fqn`
/// queries (the symbol-by-id SUSPECT, its direct-record-access PROPOSED FIX, and the
/// chunk fetch). Returns the process exit code: 0 only when every verdict/exec was
/// determined; non-zero (6) if any single sub-step failed. Every sub-step is
/// independently fault-isolated: a failure prints and the run continues.
#[allow(clippy::result_large_err)] // surrealdb::Error is large; matches engine.rs/server.rs
async fn diagnose_graph(db: &Surreal<Db>, repo: &str) -> i32 {
    println!("\n##### graph_expand index-usage diagnostic (PHASE-2 oracle) #####");
    println!("repo: {repo}");
    println!(
        "NOTE: this is measurement-only — no query, schema, or graph_expand code is changed.\n\
         The three predicates below are exactly what query/graph_expand.rs issues."
    );

    // Track whether every verdict could be determined. Any failure → exit 6.
    let mut all_ok = true;

    // ── Context: row counts ────────────────────────────────────────────────
    section("table row counts");
    match print_row_count(db, "calls").await {
        Ok(_) => {}
        Err(e) => {
            eprintln!("error: {e}");
            all_ok = false;
        }
    }
    match print_row_count(db, "symbol").await {
        Ok(_) => {}
        Err(e) => {
            eprintln!("error: {e}");
            all_ok = false;
        }
    }

    // ── Context: INFO FOR TABLE (which indexes are actually on disk) ─────────
    section("INFO FOR TABLE calls / symbol (indexes present on disk)");
    if let Err(e) = print_table_info(db, "calls").await {
        eprintln!("error: {e}");
        all_ok = false;
    }
    if let Err(e) = print_table_info(db, "symbol").await {
        eprintln!("error: {e}");
        all_ok = false;
    }

    // ── Pick a REAL high-degree fqn from calls (avoids planner short-circuit on
    //    a value that isn't present). The PLAN does not depend on the specific
    //    value, but a real one is the honest test. ────────────────────────────
    let fqn: Option<String> = match db
        .query("SELECT out_name FROM calls WHERE out_name != NONE LIMIT 1")
        .await
        .and_then(|mut r| r.take::<Vec<serde_json::Value>>(0))
    {
        Ok(rows) => rows
            .first()
            .and_then(|v| v.get("out_name"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        Err(e) => {
            eprintln!("error: failed to select a real out_name from calls: {e}");
            all_ok = false;
            None
        }
    };

    // ── Pick a REAL symbol row for the overlap predicate. ────────────────────
    let symbol_probe: Option<(String, i64, i64)> = match db
        .query("SELECT file, line_start, line_end FROM symbol LIMIT 1")
        .await
        .and_then(|mut r| r.take::<Vec<serde_json::Value>>(0))
    {
        Ok(rows) => rows.first().and_then(|v| {
            let file = v.get("file")?.as_str()?.to_string();
            let s = v.get("line_start")?.as_i64()?;
            let e = v.get("line_end")?.as_i64()?;
            Some((file, s, e))
        }),
        Err(e) => {
            eprintln!("error: failed to select a real symbol row: {e}");
            all_ok = false;
            None
        }
    };

    // ── Predicate 1: calls callers — WHERE out_name = $fqn ───────────────────
    section(
        "PREDICATE 1: calls callers — SELECT in_name FROM calls WHERE out_name = $fqn LIMIT 20",
    );
    match &fqn {
        Some(fqn) => {
            println!("bound $fqn = {fqn:?}");
            match db
                .query("SELECT in_name FROM calls WHERE out_name = $fqn LIMIT 20 EXPLAIN")
                .bind(("fqn", fqn.clone()))
                .await
                .and_then(|mut r| r.take::<Vec<serde_json::Value>>(0))
            {
                Ok(plan) => {
                    println!(
                        "EXPLAIN:\n{}",
                        serde_json::to_string_pretty(&plan).unwrap_or_default()
                    );
                    let (uses_idx, found, detail) = classify_plan(&plan);
                    println!(
                        "VERDICT: {} — {detail}",
                        if uses_idx { "INDEX" } else { "SCAN" }
                    );
                    if !found {
                        all_ok = false;
                    }
                }
                Err(e) => {
                    eprintln!("error: EXPLAIN (callers) failed: {e}");
                    all_ok = false;
                }
            }
            if !time_exec(
                "callers",
                db.query("SELECT in_name FROM calls WHERE out_name = $fqn LIMIT 20")
                    .bind(("fqn", fqn.clone())),
            )
            .await
            {
                all_ok = false;
            }
        }
        None => {
            eprintln!(
                "error: no real $fqn available (calls table empty or out_name all NONE) — cannot run predicate 1"
            );
            all_ok = false;
        }
    }

    // ── Predicate 2: calls callees — WHERE in_name = $fqn ────────────────────
    section(
        "PREDICATE 2: calls callees — SELECT out_name FROM calls WHERE in_name = $fqn LIMIT 20",
    );
    match &fqn {
        Some(fqn) => {
            println!("bound $fqn = {fqn:?}");
            match db
                .query("SELECT out_name FROM calls WHERE in_name = $fqn LIMIT 20 EXPLAIN")
                .bind(("fqn", fqn.clone()))
                .await
                .and_then(|mut r| r.take::<Vec<serde_json::Value>>(0))
            {
                Ok(plan) => {
                    println!(
                        "EXPLAIN:\n{}",
                        serde_json::to_string_pretty(&plan).unwrap_or_default()
                    );
                    let (uses_idx, found, detail) = classify_plan(&plan);
                    println!(
                        "VERDICT: {} — {detail}",
                        if uses_idx { "INDEX" } else { "SCAN" }
                    );
                    if !found {
                        all_ok = false;
                    }
                }
                Err(e) => {
                    eprintln!("error: EXPLAIN (callees) failed: {e}");
                    all_ok = false;
                }
            }
            if !time_exec(
                "callees",
                db.query("SELECT out_name FROM calls WHERE in_name = $fqn LIMIT 20")
                    .bind(("fqn", fqn.clone())),
            )
            .await
            {
                all_ok = false;
            }
        }
        None => {
            eprintln!("error: no real $fqn available — cannot run predicate 2");
            all_ok = false;
        }
    }

    // ── Predicate 3: symbol overlap — WHERE file = $file AND line_start <= $e
    //    AND line_end >= $s (the exact shape symbols_overlapping issues). ──────
    section(
        "PREDICATE 3: symbol overlap — SELECT file FROM symbol WHERE file = $file AND line_start <= $e AND line_end >= $s",
    );
    match &symbol_probe {
        Some((file, s, e)) => {
            println!("bound $file = {file:?}, $s = {s}, $e = {e}");
            match db
                .query(
                    "SELECT file FROM symbol WHERE file = $file AND line_start <= $e AND line_end >= $s EXPLAIN",
                )
                .bind(("file", file.clone()))
                .bind(("e", *e))
                .bind(("s", *s))
                .await
                .and_then(|mut r| r.take::<Vec<serde_json::Value>>(0))
            {
                Ok(plan) => {
                    println!(
                        "EXPLAIN:\n{}",
                        serde_json::to_string_pretty(&plan).unwrap_or_default()
                    );
                    let (uses_idx, found, detail) = classify_plan(&plan);
                    println!("VERDICT: {} — {detail}", if uses_idx { "INDEX" } else { "SCAN" });
                    if !found {
                        all_ok = false;
                    }
                }
                Err(e) => {
                    eprintln!("error: EXPLAIN (symbol overlap) failed: {e}");
                    all_ok = false;
                }
            }
            if !time_exec(
                "symbol overlap",
                db.query(
                    "SELECT file FROM symbol WHERE file = $file AND line_start <= $e AND line_end >= $s",
                )
                .bind(("file", file.clone()))
                .bind(("e", *e))
                .bind(("s", *s)),
            )
            .await
            {
                all_ok = false;
            }
        }
        None => {
            eprintln!(
                "error: no real symbol row available (symbol table empty) — cannot run predicate 3"
            );
            all_ok = false;
        }
    }

    // ── Predicate 4 / 5 / 6 need a Thing built from a REAL symbol FQN, exactly as
    //    fetch_chunk_for_fqn does. The $fqn we already selected from calls.out_name
    //    IS a symbol record id (`<file>::<name>`), so reuse it. ─────────────────
    let thing: Option<surrealdb::sql::Thing> = fqn.as_ref().map(|fqn| {
        surrealdb::sql::Thing::from(("symbol", surrealdb::sql::Id::String(fqn.clone())))
    });

    // ── Predicate 4 (SUSPECT): symbol by id — WHERE id = $thing. Hypothesis: in
    //    SurrealDB 2.6.5 this is NOT optimized into a primary-key fetch and instead
    //    TABLE-SCANS the multi-million-row symbol table. The EXEC time is the proof
    //    (could be seconds — that's expected, let it run). ──────────────────────
    section(
        "PREDICATE 4 (suspect): symbol by id — SELECT ... FROM symbol WHERE id = $thing LIMIT 1",
    );
    match &thing {
        Some(thing) => {
            println!("bound $thing = {thing}");
            match db
                .query(
                    "SELECT meta::id(id) AS fqn, file, name, line_start, line_end, kind \
                     FROM symbol WHERE id = $thing LIMIT 1 EXPLAIN",
                )
                .bind(("thing", thing.clone()))
                .await
                .and_then(|mut r| r.take::<Vec<serde_json::Value>>(0))
            {
                Ok(plan) => {
                    println!(
                        "EXPLAIN:\n{}",
                        serde_json::to_string_pretty(&plan).unwrap_or_default()
                    );
                    let (uses_idx, found, detail) = classify_plan(&plan);
                    println!(
                        "VERDICT: {} — {detail}",
                        if uses_idx { "INDEX" } else { "SCAN" }
                    );
                    if !found {
                        all_ok = false;
                    }
                }
                Err(e) => {
                    eprintln!("error: EXPLAIN (symbol by id) failed: {e}");
                    all_ok = false;
                }
            }
            // Money shot: actually run it and time it.
            if !time_exec(
                "symbol by id",
                db.query(
                    "SELECT meta::id(id) AS fqn, file, name, line_start, line_end, kind \
                     FROM symbol WHERE id = $thing LIMIT 1",
                )
                .bind(("thing", thing.clone())),
            )
            .await
            {
                all_ok = false;
            }
        }
        None => {
            eprintln!("error: no real $fqn available — cannot run predicate 4");
            all_ok = false;
        }
    }

    // ── Predicate 5 (PROPOSED FIX): direct record access — FROM $thing. Expected
    //    to NOT scan; its EXEC time should be sub-millisecond. A direct `FROM
    //    $thing` plan may be neither "Iterate Index" nor "Iterate Table" (e.g. an
    //    "Iterate Value"/"Fetch" op) — in that case the verdict is DIRECT/OTHER and
    //    the EXEC time is what matters. ─────────────────────────────────────────
    section("PREDICATE 5 (proposed fix): direct record access — SELECT ... FROM $thing LIMIT 1");
    match &thing {
        Some(thing) => {
            println!("bound $thing = {thing}");
            match db
                .query(
                    "SELECT meta::id(id) AS fqn, file, name, line_start, line_end, kind \
                     FROM $thing LIMIT 1 EXPLAIN",
                )
                .bind(("thing", thing.clone()))
                .await
                .and_then(|mut r| r.take::<Vec<serde_json::Value>>(0))
            {
                Ok(plan) => {
                    println!(
                        "EXPLAIN:\n{}",
                        serde_json::to_string_pretty(&plan).unwrap_or_default()
                    );
                    let (uses_idx, found, detail) = classify_plan(&plan);
                    if found {
                        println!(
                            "VERDICT: {} — {detail}",
                            if uses_idx { "INDEX" } else { "SCAN" }
                        );
                    } else {
                        // Direct record access need not iterate an index or table —
                        // don't force the binary verdict. The EXEC time is the proof.
                        println!("VERDICT: DIRECT/OTHER — {detail}");
                    }
                }
                Err(e) => {
                    eprintln!("error: EXPLAIN (direct record access) failed: {e}");
                    all_ok = false;
                }
            }
            if !time_exec(
                "direct record access",
                db.query(
                    "SELECT meta::id(id) AS fqn, file, name, line_start, line_end, kind \
                     FROM $thing LIMIT 1",
                )
                .bind(("thing", thing.clone())),
            )
            .await
            {
                all_ok = false;
            }
        }
        None => {
            eprintln!("error: no real $fqn available — cannot run predicate 5");
            all_ok = false;
        }
    }

    // ── Predicate 6 (chunk fetch): the second query inside fetch_chunk_for_fqn —
    //    reuse the real $file/$s/$e from the symbol row probed earlier. Should use
    //    idx_chunk_file. ────────────────────────────────────────────────────────
    section(
        "PREDICATE 6: chunk fetch — SELECT ... FROM chunk WHERE file = $file AND line_start <= $e AND line_end >= $s ORDER BY line_start LIMIT 1",
    );
    match &symbol_probe {
        Some((file, s, e)) => {
            println!("bound $file = {file:?}, $s = {s}, $e = {e}");
            match db
                .query(
                    "SELECT file, line_start, line_end, content FROM chunk \
                     WHERE file = $file AND line_start <= $e AND line_end >= $s \
                     ORDER BY line_start LIMIT 1 EXPLAIN",
                )
                .bind(("file", file.clone()))
                .bind(("e", *e))
                .bind(("s", *s))
                .await
                .and_then(|mut r| r.take::<Vec<serde_json::Value>>(0))
            {
                Ok(plan) => {
                    println!(
                        "EXPLAIN:\n{}",
                        serde_json::to_string_pretty(&plan).unwrap_or_default()
                    );
                    let (uses_idx, found, detail) = classify_plan(&plan);
                    println!(
                        "VERDICT: {} — {detail}",
                        if uses_idx { "INDEX" } else { "SCAN" }
                    );
                    if !found {
                        all_ok = false;
                    }
                }
                Err(e) => {
                    eprintln!("error: EXPLAIN (chunk fetch) failed: {e}");
                    all_ok = false;
                }
            }
            if !time_exec(
                "chunk fetch",
                db.query(
                    "SELECT file, line_start, line_end, content FROM chunk \
                     WHERE file = $file AND line_start <= $e AND line_end >= $s \
                     ORDER BY line_start LIMIT 1",
                )
                .bind(("file", file.clone()))
                .bind(("e", *e))
                .bind(("s", *s)),
            )
            .await
            {
                all_ok = false;
            }
        }
        None => {
            eprintln!("error: no real symbol row available — cannot run predicate 6");
            all_ok = false;
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    section("SUMMARY");
    if all_ok {
        println!(
            "all verdicts determined — see VERDICT lines and EXEC timings above for index vs scan and per-query latency."
        );
        0
    } else {
        eprintln!(
            "FAIL: one or more diagnostic sub-steps could not be completed (see errors above). \
             Exiting non-zero."
        );
        6
    }
}
