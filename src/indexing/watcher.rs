use std::path::PathBuf;
use std::time::Duration;

use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_full::{DebounceEventResult, Debouncer, RecommendedCache, new_debouncer};
use tokio::sync::mpsc::Sender;
use tracing::{error, info, warn};

use crate::indexing::IndexTrigger;
use crate::indexing::tracker::{ChangeKind, FileChange};

/// How often a live watcher also enqueues a full (walk + stat) scan. notify 8.2
/// on Windows never reports a ReadDirectoryChangesW buffer overflow: a zero-byte
/// completion yields no events, and an unknown error unwatches the directory with
/// only a log line. Changes lost that way are picked up within this window. The
/// scan does work only for files whose mtime/size changed.
const SAFETY_NET_SCAN_INTERVAL: Duration = Duration::from_secs(15 * 60);

/// Coalesces "a full rescan is needed" requests: at most one rescan trigger is
/// in flight per watcher at a time.
#[derive(Clone)]
struct RescanGate {
    repo: String,
    tx: Sender<IndexTrigger>,
    pending: std::sync::Arc<std::sync::atomic::AtomicBool>,
    runtime: tokio::runtime::Handle,
}

impl RescanGate {
    fn new(repo: String, tx: Sender<IndexTrigger>, runtime: tokio::runtime::Handle) -> Self {
        Self {
            repo,
            tx,
            pending: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            runtime,
        }
    }

    /// Enqueue a full incremental scan (`changes = None`) with a blocking-capable
    /// async send, unless one is already on its way.
    fn request(&self, reason: &'static str) {
        use std::sync::atomic::Ordering;
        if self.pending.swap(true, Ordering::AcqRel) {
            return;
        }
        info!(repo = %self.repo, reason, "watcher: enqueueing full rescan");
        let gate = self.clone();
        self.runtime.spawn(async move {
            let trigger = IndexTrigger {
                repo: gate.repo.clone(),
                changes: None,
                rebuild: false,
            };
            if gate.tx.send(trigger).await.is_err() {
                error!(repo = %gate.repo, "trigger channel closed; rescan dropped");
            }
            gate.pending.store(false, Ordering::Release);
        });
    }
}

/// Turn one debouncer callback into a trigger. Nothing is ever dropped silently:
/// a watcher error, or a full trigger channel, becomes a coalesced full rescan.
fn dispatch(result: DebounceEventResult, gate: &RescanGate) {
    let changes = match result {
        Ok(events) => convert_events(events),
        Err(errors) => {
            for e in errors {
                warn!(repo = %gate.repo, error = %e, "watcher error");
            }
            gate.request("watcher error");
            return;
        }
    };
    if changes.is_empty() {
        return;
    }
    let trigger = IndexTrigger {
        repo: gate.repo.clone(),
        changes: Some(changes),
        rebuild: false,
    };
    if let Err(tokio::sync::mpsc::error::TrySendError::Full(_)) = gate.tx.try_send(trigger) {
        gate.request("trigger channel full");
    }
}

/// Start a filesystem watcher for `repo_path`.
/// Events are debounced with a 3-second window and sent through `tx` as `IndexTrigger`.
/// Falls back to 30-second polling if the watcher fails to initialise.
pub async fn start_watcher(repo_path: String, tx: Sender<IndexTrigger>) {
    let path = PathBuf::from(&repo_path);

    // Try to start a real watcher.
    let debounce_duration = Duration::from_secs(3);
    let gate = RescanGate::new(
        repo_path.clone(),
        tx.clone(),
        tokio::runtime::Handle::current(),
    );

    // `RecommendedCache` is a platform-conditional alias: `FileIdMap` on
    // macOS/Windows, `NoCache` on Linux. Annotating it (rather than `FileIdMap`)
    // is what `new_debouncer` actually returns, so this compiles on every target.
    let watcher_result: anyhow::Result<Debouncer<RecommendedWatcher, RecommendedCache>> =
        (|| -> anyhow::Result<_> {
            let gate_inner = gate.clone();
            let debouncer = new_debouncer(
                debounce_duration,
                None,
                move |result: DebounceEventResult| dispatch(result, &gate_inner),
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
                // Keep the debouncer alive by parking this task, enqueueing the
                // overflow safety-net scan on every tick.
                loop {
                    tokio::time::sleep(SAFETY_NET_SCAN_INTERVAL).await;
                    gate.request("safety-net interval");
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
    use notify::event::{ModifyKind, RenameMode};
    let mut changes = Vec::new();
    let mut push = |path: &std::path::Path, kind: ChangeKind| {
        if let Some(s) = path.to_str() {
            changes.push(FileChange {
                path: s.to_string(),
                kind,
            });
        }
    };
    for event in events {
        match &event.kind {
            // Renames by direction: a folder moved INTO the repo is a single
            // rename-to event for the folder, and must read as Added so the
            // pipeline expands it into its files; the source of a rename no longer
            // exists. (All of these used to collapse into Modified.)
            notify::EventKind::Modify(ModifyKind::Name(RenameMode::To)) => {
                for p in &event.paths {
                    push(p, ChangeKind::Added);
                }
            }
            notify::EventKind::Modify(ModifyKind::Name(RenameMode::From)) => {
                for p in &event.paths {
                    push(p, ChangeKind::Deleted);
                }
            }
            notify::EventKind::Modify(ModifyKind::Name(RenameMode::Both)) => {
                if let [from, to] = event.paths.as_slice() {
                    push(from, ChangeKind::Deleted);
                    push(to, ChangeKind::Added);
                } else {
                    for p in &event.paths {
                        push(p, ChangeKind::Modified);
                    }
                }
            }
            notify::EventKind::Create(_) => {
                for p in &event.paths {
                    push(p, ChangeKind::Added);
                }
            }
            notify::EventKind::Modify(_) => {
                for p in &event.paths {
                    push(p, ChangeKind::Modified);
                }
            }
            notify::EventKind::Remove(_) => {
                for p in &event.paths {
                    push(p, ChangeKind::Deleted);
                }
            }
            _ => {}
        }
    }
    changes
}

#[cfg(test)]
mod tests {
    use super::*;
    use notify::event::{EventKind, ModifyKind, RenameMode};
    use notify_debouncer_full::DebouncedEvent;
    use std::path::PathBuf;
    use std::time::Instant;

    fn ev(kind: EventKind, paths: &[&str]) -> DebouncedEvent {
        let mut e = notify::Event::new(kind);
        for p in paths {
            e = e.add_path(PathBuf::from(p));
        }
        DebouncedEvent::new(e, Instant::now())
    }

    /// S6/D11: renames are mapped by direction. A folder moved INTO the repo is a
    /// rename-to and must read as Added (so the pipeline expands it); the source
    /// of a rename is gone and must read as Deleted.
    #[test]
    fn renames_map_by_direction() {
        let changes = convert_events(vec![
            ev(
                EventKind::Modify(ModifyKind::Name(RenameMode::To)),
                &["r/new"],
            ),
            ev(
                EventKind::Modify(ModifyKind::Name(RenameMode::From)),
                &["r/old"],
            ),
            ev(
                EventKind::Modify(ModifyKind::Name(RenameMode::Both)),
                &["r/a.rs", "r/b.rs"],
            ),
        ]);
        let got: Vec<(String, ChangeKind)> =
            changes.into_iter().map(|c| (c.path, c.kind)).collect();
        assert_eq!(
            got,
            vec![
                ("r/new".to_string(), ChangeKind::Added),
                ("r/old".to_string(), ChangeKind::Deleted),
                ("r/a.rs".to_string(), ChangeKind::Deleted),
                ("r/b.rs".to_string(), ChangeKind::Added),
            ]
        );
    }

    async fn recv_soon(rx: &mut tokio::sync::mpsc::Receiver<IndexTrigger>) -> IndexTrigger {
        tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("trigger within 5s")
            .expect("channel open")
    }

    /// S5/D12: a full trigger channel used to drop the change set (the comment
    /// promised a poll that never runs while the watcher is alive). It must
    /// become a full rescan once the channel drains.
    #[tokio::test]
    async fn full_channel_becomes_a_rescan() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        tx.try_send(IndexTrigger {
            repo: "r".into(),
            changes: Some(vec![]),
            rebuild: false,
        })
        .unwrap();
        let gate = RescanGate::new("r".into(), tx, tokio::runtime::Handle::current());
        dispatch(
            Ok(vec![ev(
                EventKind::Create(notify::event::CreateKind::File),
                &["r/a.rs"],
            )]),
            &gate,
        );
        let first = recv_soon(&mut rx).await;
        assert!(first.changes.is_some(), "the queued trigger drains first");
        let second = recv_soon(&mut rx).await;
        assert!(
            second.changes.is_none(),
            "the dropped batch became a full rescan"
        );
    }

    /// S4/D12: a watcher error is a full rescan, not just a log line.
    #[tokio::test]
    async fn watcher_error_becomes_a_rescan() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let gate = RescanGate::new("r".into(), tx, tokio::runtime::Handle::current());
        dispatch(Err(vec![notify::Error::generic("overflow")]), &gate);
        let t = recv_soon(&mut rx).await;
        assert!(t.changes.is_none());
    }
}
