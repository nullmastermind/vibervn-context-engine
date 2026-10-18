//! The filesystem watcher's memory must not scale with what it ignores.
//!
//! Regression for 15 GB on hcengineering/platform (~7k indexed files): the
//! debouncer's file-id cache walked the whole repo on `watch()`, through every
//! pnpm/rush junction in `node_modules`, and kept one entry per path reached.
//!
//! Its own test binary with a single test, so no other test's allocations land
//! in the process-wide measurement. Windows-only: junctions are the pnpm layout
//! there, and it is where the debouncer defaulted to that cache (Linux never did).
#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use context_engine_rs::indexing::IndexTrigger;
use context_engine_rs::indexing::watcher::start_watcher;

#[repr(C)]
#[derive(Default)]
struct ProcessMemoryCounters {
    cb: u32,
    page_fault_count: u32,
    peak_working_set_size: usize,
    working_set_size: usize,
    quota_peak_paged_pool_usage: usize,
    quota_paged_pool_usage: usize,
    quota_peak_non_paged_pool_usage: usize,
    quota_non_paged_pool_usage: usize,
    pagefile_usage: usize,
    peak_pagefile_usage: usize,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentProcess() -> isize;
    fn K32GetProcessMemoryInfo(
        process: isize,
        counters: *mut ProcessMemoryCounters,
        cb: u32,
    ) -> i32;
}

/// Committed private bytes of this process.
fn private_bytes() -> usize {
    let mut c = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        ..Default::default()
    };
    // SAFETY: `c` is a correctly sized PROCESS_MEMORY_COUNTERS; the pseudo-handle
    // from GetCurrentProcess needs no closing.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    assert_ne!(ok, 0, "K32GetProcessMemoryInfo failed");
    c.pagefile_usage
}

/// One real package of `files` files, linked `links` times from the dependents'
/// `node_modules` (pnpm's layout), next to one source file.
fn build_fixture(root: &Path, files: usize, links: usize) -> PathBuf {
    std::fs::write(root.join(".gitignore"), "node_modules/\n").unwrap();
    let src = root.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("index.ts"), "export const x = 0;\n").unwrap();

    let pkg = root.join("node_modules/.pnpm/pkg@1.0.0/node_modules/pkg");
    for f in 0..files {
        let dir = pkg.join(format!("dist/m{}", f % 16));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("f{f}.js")), "module.exports = 1;\n").unwrap();
    }
    let mut script = String::from("@echo off\r\n");
    for l in 0..links {
        let nm = root.join(format!("node_modules/.pnpm/dep{l}@1.0.0/node_modules"));
        std::fs::create_dir_all(&nm).unwrap();
        script.push_str(&format!(
            "mklink /J \"{}\" \"{}\" >nul || exit /b 1\r\n",
            nm.join("pkg").display(),
            pkg.display()
        ));
    }
    let script_path = root.join("links.cmd");
    std::fs::write(&script_path, script).unwrap();
    let status = std::process::Command::new("cmd")
        .arg("/c")
        .arg(&script_path)
        .status()
        .unwrap();
    assert!(status.success(), "creating junctions failed");
    std::fs::remove_file(&script_path).unwrap();
    src.join("index.ts")
}

#[tokio::test]
async fn watcher_memory_does_not_scale_with_ignored_trees() {
    // 2000 files reached through 21 paths = 42k paths a whole-tree cache would
    // hold (~16 MB measured); a watcher that ignores them holds ~nothing.
    const FILES: usize = 2000;
    const LINKS: usize = 20;
    const BUDGET: usize = 4 * 1024 * 1024;

    let dir = tempfile::tempdir().unwrap();
    let src_file = build_fixture(dir.path(), FILES, LINKS);
    let repo = context_engine_rs::store::normalize_repo_path(dir.path().to_str().unwrap());
    let (tx, mut rx) = tokio::sync::mpsc::channel::<IndexTrigger>(64);

    let before = private_bytes();
    tokio::spawn(start_watcher(repo, tx));

    // Edit the source file until its trigger arrives: that proves `watch()` has
    // returned (any cache is fully built) and the watcher is live.
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut n = 0;
    let trigger = loop {
        assert!(Instant::now() < deadline, "the source edit never arrived");
        n += 1;
        std::fs::write(&src_file, format!("export const x = {n};\n")).unwrap();
        if let Ok(Some(t)) = tokio::time::timeout(Duration::from_millis(500), rx.recv()).await {
            break t;
        }
    };
    let grown = private_bytes().saturating_sub(before);

    assert!(
        grown < BUDGET,
        "watching a repo with one source file grew private bytes by {:.1} MB \
         (budget {:.1} MB): the watcher is holding state for ignored paths",
        grown as f64 / 1048576.0,
        BUDGET as f64 / 1048576.0
    );
    let paths: Vec<String> = trigger
        .changes
        .expect("explicit change set")
        .into_iter()
        .map(|c| c.path.replace('\\', "/").to_lowercase())
        .collect();
    assert!(
        paths.iter().any(|p| p.ends_with("src/index.ts"))
            && !paths.iter().any(|p| p.contains("node_modules")),
        "the source edit, and nothing ignored, is a change: {paths:?}"
    );
}
