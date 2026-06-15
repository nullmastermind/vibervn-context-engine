//! `bench-incremental` — the INCREMENTAL-path performance-benchmark ORACLE for
//! context-engine-rs. Measurement/verification tooling ONLY; it implements no
//! optimization. The optimization is built later and verified AGAINST this.
//!
//! It encodes the locked success criterion: a manual Incremental Update on a
//! large repo, for a SINGLE-FILE edit and for an N-FILE edit (both performed
//! AFTER a clean full rebuild that sets up realistic on-disk DB state), should
//! complete in <= 10_000 ms of incremental *pipeline* wall time (NOT counting
//! the one-time full rebuild). The harness PRINTS a per-stage breakdown so the
//! dominant cost is provable, and gates on the 10_000ms threshold.
//!
//! WHY a CLI (not a unit test): the gate must run at REAL kernel scale on the
//! REAL on-disk index, through the SAME engine the server uses — never synthetic
//! data. It boots via `engine_boot::boot_engine`, registers the repo (without
//! persisting it to settings.repos, exactly like bench-query), and drives
//! remove/rebuild/incremental through the SAME `engine_ops` / `IndexEngine` ops.
//!
//! The per-stage breakdown is emitted by the pipeline as a single
//! `PERF SUMMARY incremental` tracing event. We capture it IN-PROCESS via a
//! custom tracing Layer (no fragile stderr re-parsing) and re-print it to stdout
//! in a stable, greppable format for the harness script.
//!
//! Usage:
//!   bench-incremental --repo <PATH> --files <N> [--data-dir PATH]
//!
//! Step A: clean full rebuild (remove_index + trigger_rebuild, wait to done) —
//!         SETUP only, NOT part of the measured metric.
//! Step B: pick N real indexed source files and inject an edit of `--edit-kind`:
//!         `comment` appends a comment line (symbol surface UNCHANGED — the gated
//!         typical-edit case); `add-symbol` appends a uniquely-named function
//!         (surface CHANGED — reported informationally). Original byte length is
//!         recorded for restore.
//! Step C: ONE manual incremental index (trigger_index — NOT rebuild), wait to
//!         done, capture the `PERF SUMMARY incremental` per-stage breakdown.
//! Step D: ALWAYS restore the modified files to their original bytes (a defer
//!         guard runs even on error/panic) so the checkout is left byte-identical.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::Parser;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

use context_engine_rs::engine_boot::{boot_engine, BootOptions, BootedEngine};
use context_engine_rs::engine_ops;
use context_engine_rs::indexing::IndexState;
use context_engine_rs::store;

mod perf_layer;
use perf_layer::{PerfCapture, PerfLayer};

#[derive(Parser, Debug)]
#[command(
    name = "bench-incremental",
    about = "Measure the incremental-index per-stage breakdown on a real repo (oracle only)"
)]
struct Cli {
    /// Workspace path to benchmark.
    #[arg(long, default_value = "c:/users/0x317/downloads/linux")]
    repo: String,

    /// Number of real indexed files to modify (e.g. 1 or 10). Each gets a
    /// sentinel line appended so it classifies as Modified, then is restored.
    #[arg(long, default_value_t = 1)]
    files: usize,

    /// What KIND of edit to inject into each picked file:
    ///   - `comment` (default): append a comment line. The symbol surface is
    ///     UNCHANGED, so the blast-radius gating must collapse to O(changed) — this
    ///     is the typical save-while-coding edit and is the GATED criterion.
    ///   - `add-symbol`: append a uniquely-named function. The symbol surface
    ///     CHANGES (direction-1/2 can fire), so this legitimately pays more. Its
    ///     cost is reported informationally, NOT gated.
    #[arg(long, value_enum, default_value_t = EditKind::Comment)]
    edit_kind: EditKind,

    /// Data directory base override (same precedence as the server). Pass a
    /// separate dir to run alongside a live server (RocksDB exclusive per-dir lock).
    #[arg(long)]
    data_dir: Option<std::path::PathBuf>,
}

/// The edit to inject into each picked file (see `Cli::edit_kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum EditKind {
    /// Append a comment line — symbol surface UNCHANGED (gated criterion).
    Comment,
    /// Append a uniquely-named function — symbol surface CHANGED (informational).
    AddSymbol,
}

impl EditKind {
    /// The bytes to append to a file of the given path, for this edit kind. `ts`
    /// makes the addition unique per run (and the size always changes → Modified).
    /// For `add-symbol` the appended construct is a real, parseable top-level
    /// definition in the file's language so the parser extracts a NEW symbol whose
    /// leaf name did not exist before (surface change). C/C++/headers get a C
    /// function; Rust gets a Rust fn; anything else falls back to a C-style fn
    /// (the picked set is restricted to .c/.h/.rs, so this is exhaustive).
    fn appended_bytes(&self, path: &str, ts: u128) -> String {
        match self {
            EditKind::Comment => format!("\n// bench-incremental touch {ts}\n"),
            EditKind::AddSymbol => {
                let lp = path.to_ascii_lowercase();
                if lp.ends_with(".rs") {
                    format!("\npub fn bench_added_{ts}() {{}}\n")
                } else {
                    // .c / .h — a free function definition.
                    format!("\nvoid bench_added_{ts}(void) {{}}\n")
                }
            }
        }
    }
}

/// Overall cap on how long we wait for indexing to finish. Generous — the
/// one-time full rebuild of a large repo is network-embed-bound (kernel: tens of
/// minutes). The measured incremental run is what must be fast.
const INDEX_WAIT_CAP: Duration = Duration::from_secs(60 * 60);

/// True if an error string looks like a RocksDB exclusive-lock / open conflict.
fn looks_like_lock_conflict(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    m.contains("lock") || m.contains("could not open") || m.contains("open surreal")
}

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

/// A restore guard: on drop (normal return OR panic), truncate each modified
/// file back to its recorded original byte length. This is the mandatory "leave
/// the checkout byte-identical" discipline — it runs even if indexing errors.
struct RestoreGuard {
    /// (path, original_len_bytes)
    originals: Vec<(String, u64)>,
    restored: bool,
}

impl RestoreGuard {
    fn new(originals: Vec<(String, u64)>) -> Self {
        Self { originals, restored: false }
    }

    /// Truncate each file back to its original length. Idempotent; sets a flag so
    /// Drop doesn't redo the work. Returns the count restored and any failures.
    fn restore(&mut self) -> (usize, Vec<String>) {
        if self.restored {
            return (0, vec![]);
        }
        self.restored = true;
        let mut ok = 0usize;
        let mut failures = Vec::new();
        for (path, orig_len) in &self.originals {
            match std::fs::OpenOptions::new().write(true).open(path) {
                Ok(f) => match f.set_len(*orig_len) {
                    Ok(()) => ok += 1,
                    Err(e) => failures.push(format!("{path}: set_len failed: {e}")),
                },
                Err(e) => failures.push(format!("{path}: open-for-restore failed: {e}")),
            }
        }
        (ok, failures)
    }
}

impl Drop for RestoreGuard {
    fn drop(&mut self) {
        // Last-resort restore if the explicit path didn't run (e.g. early return
        // or panic). Best-effort: surface failures on stderr, never panic in Drop.
        let (ok, failures) = self.restore();
        if ok > 0 || !failures.is_empty() {
            eprintln!("[restore-guard] truncated {ok} file(s) back to original bytes on drop");
            for f in &failures {
                eprintln!("[restore-guard] WARNING: {f}");
            }
        }
    }
}

async fn run() -> i32 {
    // Install a tracing subscriber that (a) prints warnings to stderr for the
    // human, AND (b) captures the `PERF SUMMARY incremental` event in-process via
    // a custom layer. RUST_LOG can raise the stderr verbosity; the capture layer
    // sees the INFO-level event regardless of the fmt filter.
    let capture: PerfCapture = Arc::new(Mutex::new(None));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("context_engine_rs=warn,warn")),
        );
    // The capture layer must NOT be gated by a low RUST_LOG, or the PERF SUMMARY
    // event would be dropped before we see it. Gate it only at INFO (its level).
    let _ = tracing_subscriber::registry()
        .with(fmt_layer)
        .with(
            PerfLayer::new(capture.clone())
                .with_filter(tracing_subscriber::filter::LevelFilter::INFO),
        )
        .try_init();

    let cli = Cli::parse();
    if cli.files == 0 {
        eprintln!("error: --files must be >= 1 (got 0)");
        return 2;
    }
    let repo = store::normalize_repo_path(&cli.repo);

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
        // Suppress boot-time watchers for ALL configured repos. The kernel repo
        // under test is typically already in settings.repos (the user indexes it
        // via the server), so a boot watcher would fire its OWN debounced
        // incremental on the step-B sentinel appends / step-D restores and race
        // the ONE manual incremental we time for the single per-repo connection.
        // With watchers off, the only triggers in flight are the ones this
        // harness explicitly sends — the measured run is the sole timed work.
        no_watchers: true,
    })
    .await
    {
        Ok(b) => b,
        Err(e) => {
            let detail = format!("{e:#}");
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

    // Register the repo (status entry ONLY — NO filesystem watcher). We
    // deliberately do NOT add it to settings.repos (same transient-tool
    // discipline as bench-query: remove_index bump-writes settings, so pushing
    // here would permanently mutate the user's config). We ALSO skip the watcher:
    // step B appends sentinels and step D restores them, and a live watcher would
    // fire its own debounced incremental on those edits, racing the ONE manual
    // incremental we time for the single per-repo RocksDB connection. With no
    // watcher, the only triggers that ever fire are the ones this harness sends.
    index_engine.register_repo_no_watcher(&repo).await;

    // ── Step A: clean full rebuild (SETUP — not measured). ───────────────────
    eprintln!("[step A] clean full rebuild (remove_index + trigger_rebuild) — SETUP, not measured...");
    let pre_rebuild = index_engine.repo_status(&repo).await;
    let pre_rebuild_indexed = pre_rebuild.as_ref().and_then(|s| s.last_indexed_at);
    match engine_ops::remove_index(&home_dir, &data_dir, &index_engine, &settings, &repo, false).await {
        Ok(_) => {}
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
    let rebuild_start = Instant::now();
    if let Some(code) = wait_for_index(&index_engine, &repo, pre_rebuild_indexed, &data_dir).await {
        return code;
    }
    eprintln!("[step A] full rebuild complete in {:.1}s", rebuild_start.elapsed().as_secs_f64());

    // ── Step A.5: DRAIN the post-rebuild background cache recompute. ──────────
    // The full rebuild schedules a DEBOUNCED, single-flight graph/stats cache
    // recompute (O(repo) ~90s full-table GROUP BY) that pins the ONE per-repo
    // RocksDB connection while it runs. If we measured an incremental while that
    // recompute were still sleeping in its debounce window or mid-aggregation,
    // the incremental's own queries would stall behind it on the shared
    // connection and we'd be timing contention, not the incremental. So we
    // block here on a DETERMINISTIC idle signal — `recompute_pending` is true
    // while a scheduler task exists (debounce sleep OR aggregation) and flips to
    // false only once it retires — NOT a blind sleep. Once idle, the connection
    // is free and the incremental we time is the sole user of it.
    eprintln!("[step A.5] draining post-rebuild background cache recompute (waiting for idle)...");
    let drain_start = Instant::now();
    let drain_cap = Duration::from_secs(10 * 60);
    while index_engine.recompute_pending(&repo) {
        if drain_start.elapsed() > drain_cap {
            eprintln!(
                "error: post-rebuild cache recompute did not drain within {}s — refusing to \
                 measure against a busy connection",
                drain_cap.as_secs()
            );
            return 2;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    eprintln!("[step A.5] recompute idle after {:.1}s — connection is free", drain_start.elapsed().as_secs_f64());

    // ── Step B: pick N real indexed files, append a sentinel, record orig len. ─
    // Open the repo's REAL on-disk DB through the gated one-handle-per-repo API
    // (NEVER a second raw Surreal::new on the same path). Resolve the generation
    // the same way the engine does (settings is the source of truth).
    let generation = settings.read().await.repo_generation(&repo);
    let db = match store::get_or_open(&repo_dbs, &data_dir, &repo, generation).await {
        Ok(db) => db,
        Err(e) => {
            let detail = format!("{e:#}");
            if looks_like_lock_conflict(&detail) {
                return lock_conflict_guidance(&data_dir, &detail);
            }
            eprintln!("error: could not open index DB to pick files: {detail}");
            return 2;
        }
    };

    let mut all_meta = match store::ops::get_all_file_meta(&db, &repo).await {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: failed to read file_meta: {e:#}");
            return 2;
        }
    };
    // Deterministic selection: sort by path, then take the first N whose extension
    // is a source extension we know parses (.c/.h/.rs). They MUST already be in
    // file_meta so detect_changes classifies them as Modified (not Added).
    all_meta.sort_by(|a, b| a.path.cmp(&b.path));
    let is_pick = |p: &str| {
        let lp = p.to_ascii_lowercase();
        lp.ends_with(".c") || lp.ends_with(".h") || lp.ends_with(".rs")
    };
    let picked: Vec<String> = all_meta
        .iter()
        .map(|m| m.path.clone())
        .filter(|p| is_pick(p))
        .take(cli.files)
        .collect();

    if picked.len() < cli.files {
        eprintln!(
            "error: only found {} indexed .c/.h/.rs files (need {}). Is the repo indexed and large enough?",
            picked.len(),
            cli.files
        );
        return 2;
    }

    // Record original byte lengths, then append a UNIQUE sentinel line to each so
    // the size changes (guaranteeing Modified even at second-granularity mtime).
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut originals: Vec<(String, u64)> = Vec::with_capacity(picked.len());
    for p in &picked {
        match std::fs::metadata(p) {
            Ok(md) => originals.push((p.clone(), md.len())),
            Err(e) => {
                eprintln!("error: cannot stat picked file {p}: {e}");
                return 2;
            }
        }
    }
    // Install the restore guard NOW (before any mutation) so even an error or
    // panic between here and the explicit restore leaves the checkout clean.
    let mut guard = RestoreGuard::new(originals.clone());

    for (p, _) in &originals {
        let sentinel = cli.edit_kind.appended_bytes(p, ts);
        if let Err(e) = append_bytes(p, sentinel.as_bytes()) {
            eprintln!("error: failed to append edit to {p}: {e}");
            // guard drops → restores any already-touched files.
            return 2;
        }
    }
    eprintln!(
        "[step B] applied {:?} edit to {} file(s); recorded original bytes for restore",
        cli.edit_kind, picked.len()
    );

    // ── Step C: ONE manual incremental index (NOT rebuild). Capture stats. ────
    // Clear any stale capture, then trigger + wait. The pipeline emits exactly one
    // `PERF SUMMARY incremental` event for this run, captured by PerfLayer.
    if let Ok(mut slot) = capture.lock() {
        *slot = None;
    }
    let pre_incr = index_engine.repo_status(&repo).await;
    let pre_incr_indexed = pre_incr.as_ref().and_then(|s| s.last_indexed_at);
    eprintln!("[step C] triggering MANUAL incremental index...");
    if let Err(e) = index_engine.trigger_index(&repo).await {
        eprintln!("error: failed to trigger incremental index: {e:#}");
        return 2;
    }
    let incr_wall_start = Instant::now();
    if let Some(code) = wait_for_index(&index_engine, &repo, pre_incr_indexed, &data_dir).await {
        return code;
    }
    let incr_wall_ms = incr_wall_start.elapsed().as_millis() as u64;

    // ── Step D: restore the modified files (explicit; guard is the backstop). ──
    let (restored, failures) = guard.restore();
    eprintln!("[step D] restored {restored} file(s) to original bytes");
    let mut restore_failed = false;
    for f in &failures {
        eprintln!("error: restore failed: {f}");
        restore_failed = true;
    }

    // ── Read the captured per-stage breakdown. ────────────────────────────────
    let perf = capture.lock().ok().and_then(|s| s.clone());
    let perf = match perf {
        Some(p) => p,
        None => {
            eprintln!(
                "error: no `PERF SUMMARY incremental` event captured — the incremental run \
                 may not have detected any changes (a no-op incremental is a harness failure: \
                 the sentinel append did not register), or the pipeline did not take the \
                 incremental branch. files_changed=0"
            );
            return 5;
        }
    };

    let files_changed = picked.len();

    // Stable, parseable stdout line (the harness greps this). total = sum of the
    // six top-level stages (phase2_total already subsumes the p2_* sub-stages).
    // `p2_symname` now carries the surface-delta (load+diff) time — the gating
    // computation that replaced the old unconditional symbol-name query.
    let total_ms = perf.total_ms();
    println!(
        "--- incr-stages (ms): edit_kind={:?} walk={} meta_load={} surface_delta={} dir1_callers={} \
         delete_bulk={} streaming={} phase2_total={} p2_dir2_scan={} p2_delete_calls={} \
         p2_reresolve={} resolve_set={} total={} ---",
        cli.edit_kind,
        perf.incr_walk_ms,
        perf.incr_meta_load_ms,
        perf.incr_p2_symname_ms,
        perf.incr_predelete_callers_ms,
        perf.incr_delete_bulk_ms,
        perf.incr_streaming_ms,
        perf.incr_phase2_total_ms,
        perf.incr_p2_dir2_scan_ms,
        perf.incr_p2_delete_calls_ms,
        perf.incr_p2_reresolve_ms,
        perf.incr_resolve_set_size,
        total_ms,
    );
    println!("files_changed={files_changed}");
    // Cross-check wall time (trigger→done, includes ~1s poll granularity + engine
    // consumer pickup) for transparency; the GATE is on the stage-sum total above.
    println!("incr_wall_ms_observed={incr_wall_ms}");

    if restore_failed {
        eprintln!("FAIL: one or more files could not be restored — the checkout may be dirty.");
        return 4;
    }

    // The incremental MUST have done real work. If every stage is 0 AND the
    // resolve set is empty, the touch didn't register (no-op) — harness failure.
    if total_ms == 0 && perf.incr_resolve_set_size == 0 {
        eprintln!("FAIL: incremental run did no measurable work (no-op) — sentinel append did not register as a change.");
        return 5;
    }

    0
}

/// Append bytes to an existing file.
fn append_bytes(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().append(true).open(path)?;
    f.write_all(bytes)?;
    f.flush()?;
    Ok(())
}

/// Poll `repo_status` until the run is done or fails. Returns `None` on success,
/// `Some(exit_code)` on failure/timeout. "Done" = state Idle AND last_indexed_at
/// fresh relative to `pre`. Mirrors bench-query's wait_for_index.
async fn wait_for_index(
    index_engine: &context_engine_rs::indexing::IndexEngine,
    repo: &str,
    pre_last_indexed: Option<chrono::DateTime<chrono::Utc>>,
    data_dir: &std::path::Path,
) -> Option<i32> {
    let start = Instant::now();
    let mut last_progress_log = Instant::now();
    loop {
        match index_engine.repo_status(repo).await {
            Some(s) => match s.state {
                IndexState::Idle => {
                    let fresh = match (pre_last_indexed, s.last_indexed_at) {
                        (_, None) => false,
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
                }
                IndexState::Indexing => {}
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

        if last_progress_log.elapsed() >= Duration::from_secs(10) {
            eprintln!("indexing... ({}s)", start.elapsed().as_secs());
            last_progress_log = Instant::now();
        }

        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}
