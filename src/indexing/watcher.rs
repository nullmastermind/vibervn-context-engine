use std::path::PathBuf;
use std::time::Duration;

use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_full::{DebounceEventResult, Debouncer, RecommendedCache, new_debouncer};
use tokio::sync::mpsc::Sender;
use tracing::{error, info, warn};

use crate::indexing::{IndexTrigger};
use crate::indexing::tracker::{ChangeKind, FileChange};

/// Start a filesystem watcher for `repo_path`.
/// Events are debounced with a 3-second window and sent through `tx` as `IndexTrigger`.
/// Falls back to 30-second polling if the watcher fails to initialise.
pub async fn start_watcher(repo_path: String, tx: Sender<IndexTrigger>) {
    let path = PathBuf::from(&repo_path);

    // Try to start a real watcher.
    let debounce_duration = Duration::from_secs(3);
    let tx_watcher = tx.clone();
    let repo_for_watcher = repo_path.clone();

    // `RecommendedCache` is a platform-conditional alias: `FileIdMap` on
    // macOS/Windows, `NoCache` on Linux. Annotating it (rather than `FileIdMap`)
    // is what `new_debouncer` actually returns, so this compiles on every target.
    let watcher_result: anyhow::Result<Debouncer<RecommendedWatcher, RecommendedCache>> =
        (|| -> anyhow::Result<_> {
            let tx_inner = tx_watcher.clone();
            let repo_inner = repo_for_watcher.clone();

            let debouncer = new_debouncer(
                debounce_duration,
                None,
                move |result: DebounceEventResult| {
                    let changes = match result {
                        Ok(events) => convert_events(events),
                        Err(errors) => {
                            for e in errors {
                                warn!(error = %e, "watcher error");
                            }
                            return;
                        }
                    };
                    if changes.is_empty() {
                        return;
                    }
                    let trigger = IndexTrigger {
                        repo: repo_inner.clone(),
                        changes: Some(changes),
                        rebuild: false,
                    };
                    // Non-blocking send; if channel is full, drop (will recover on next poll).
                    let _ = tx_inner.try_send(trigger);
                },
            )?;

            Ok(debouncer)
        })();

    match watcher_result {
        Ok(mut debouncer) => {
            if let Err(e) = debouncer.watch(&path, RecursiveMode::Recursive) {
                warn!(
                    repo = %repo_path,
                    error = %e,
                    "watcher watch() failed — falling back to polling"
                );
                run_polling_fallback(repo_path, tx).await;
            } else {
                info!(repo = %repo_path, "filesystem watcher started");
                // Keep the debouncer alive by parking this task.
                loop {
                    tokio::time::sleep(Duration::from_secs(60)).await;
                }
            }
        }
        Err(e) => {
            warn!(
                repo = %repo_path,
                error = %e,
                "failed to create filesystem watcher — falling back to polling"
            );
            run_polling_fallback(repo_path, tx).await;
        }
    }
}

/// 30-second polling fallback: send a full incremental trigger every 30 seconds.
async fn run_polling_fallback(repo_path: String, tx: Sender<IndexTrigger>) {
    info!(repo = %repo_path, "polling fallback active (30s interval)");
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;
        let trigger = IndexTrigger {
            repo: repo_path.clone(),
            changes: None, // full incremental scan
            rebuild: false,
        };
        if tx.send(trigger).await.is_err() {
            error!(repo = %repo_path, "trigger channel closed; stopping polling");
            break;
        }
    }
}

fn convert_events(events: Vec<notify_debouncer_full::DebouncedEvent>) -> Vec<FileChange> {
    let mut changes = Vec::new();
    for event in events {
        let kind = match &event.kind {
            notify::EventKind::Create(_) => Some(ChangeKind::Added),
            notify::EventKind::Modify(_) => Some(ChangeKind::Modified),
            notify::EventKind::Remove(_) => Some(ChangeKind::Deleted),
            _ => None,
        };
        if let Some(k) = kind {
            for path in &event.paths {
                if let Some(s) = path.to_str() {
                    changes.push(FileChange {
                        path: s.to_string(),
                        kind: k.clone(),
                    });
                }
            }
        }
    }
    changes
}
