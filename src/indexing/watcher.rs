use std::path::{Path, PathBuf};
use std::time::Duration;

use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_full::{DebounceEventResult, Debouncer, NoCache, new_debouncer_opt};
use tokio::sync::mpsc::Sender;
use tracing::{error, info, warn};

use crate::indexing::IndexTrigger;
use crate::indexing::tracker::{ChangeKind, FileChange};
use crate::indexing::walker::{ChangeFilter, SKIP_DIRS};

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

/// False when `path` sits where `walk_repo` never indexes anything: under (or
/// itself being) a dot-dir, a `SKIP_DIRS` tree, or a gitignored directory.
///
/// WHY in the watcher and not only in the pipeline: on a pnpm/rush monorepo
/// (hcengineering/platform, ~7k indexed files) `node_modules` and build-output
/// churn dwarfs real edits. Each such batch used to become a trigger carrying
/// every path, queued behind runs that open the DB, stat every path, filter
/// everything out, then still refresh the O(repo) sidecars.
///
/// Conservative: per-repo ignored paths, configured ignore filenames and custom
/// extensions are not known here and stay the pipeline's job; this drops only
/// what every rule set excludes. A stale entry under an excluded directory (left
/// by an older, unfiltered watcher) is still reconciled by the safety-net scan.
fn may_affect_index(filter: &ChangeFilter, root: &Path, path: &Path) -> bool {
    let strictly_inside = path
        .strip_prefix(root)
        .is_ok_and(|rel| !rel.as_os_str().is_empty());
    if !strictly_inside {
        return true;
    }
    if path
        .parent()
        .is_some_and(|parent| !filter.could_contain_indexed(parent))
    {
        return false;
    }
    // No SKIP_DIRS name is an indexable file name (none has an extension), so the
    // event is for that directory itself, whether it still exists or not.
    if path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| SKIP_DIRS.contains(&n))
    {
        return false;
    }
    // Windows reports a Modify on a directory whenever a direct child changes, so
    // `.git` or a gitignored `lib/` would otherwise trigger on every write in it.
    !(path.is_dir() && !filter.could_contain_indexed(path))
}

/// Turn one debouncer callback into a trigger. Nothing is ever dropped silently:
/// a watcher error, or a full trigger channel, becomes a coalesced full rescan.
/// Changes that cannot affect the index are dropped (see [`may_affect_index`]).
fn dispatch(result: DebounceEventResult, gate: &RescanGate) {
    let changes = match result {
        Ok(events) => {
            let root = Path::new(&gate.repo);
            // Built per batch, like the pipeline's: an edited ignore file applies
            // to the next batch with no invalidation.
            let filter = ChangeFilter::new(root);
            let mut changes = convert_events(events);
            changes.retain(|c| may_affect_index(&filter, root, Path::new(&c.path)));
            changes
        }
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

    // `NoCache`, not `RecommendedCache` (`FileIdMap` on Windows/macOS). On
    // `watch()` that cache walks the WHOLE root — `node_modules`, `.git`, build
    // output — with `follow_links(true)`, and keeps one `PathBuf` → file-id entry
    // per path for the watcher's lifetime. pnpm/rush link every dependency as a
    // junction/symlink, so the walk re-enters shared packages once per link
    // (40k files behind 8 links each: 133 MB and 18 s in `watch()`; reported
    // 15 GB on a repo with ~7k indexed files). Every Remove event then runs a
    // `retain` over that whole map under the debouncer lock, so an install or
    // `rm -rf node_modules` stalls event intake for deletes × cached paths. The
    // cache only stitches rename halves into `RenameMode::Both`, which
    // `convert_events` splits back into Deleted + Added anyway; Linux has always
    // run with `NoCache`.
    //
    // `follow_symlinks(false)`: the indexer's walk never follows links, so
    // nothing behind one can be indexed. inotify/kqueue would otherwise add a
    // watch (and a map entry) per directory reached through every link.
    let watcher_result: anyhow::Result<Debouncer<RecommendedWatcher, NoCache>> =
        (|| -> anyhow::Result<_> {
            let gate_inner = gate.clone();
            let debouncer = new_debouncer_opt(
                debounce_duration,
                None,
                move |result: DebounceEventResult| dispatch(result, &gate_inner),
                NoCache::new(),
                notify::Config::default().with_follow_symlinks(false),
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

    /// 15 GB on a ~7k-file repo (hcengineering/platform): every `node_modules` /
    /// build-output event reached the trigger channel and ran the pipeline, only
    /// to be filtered there. A change in a place that can never hold indexed
    /// files is dropped in the watcher; a batch made only of those sends nothing.
    #[tokio::test]
    async fn dispatch_drops_changes_that_cannot_affect_the_index() {
        use notify::event::{CreateKind, RemoveKind};
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join(".gitignore"), "lib/\n").unwrap();
        for d in ["src", "app/lib", "node_modules/a", ".git"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        let p = |rel: &str| root.join(rel).to_str().unwrap().to_string();
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let gate = RescanGate::new(
            root.to_str().unwrap().to_string(),
            tx,
            tokio::runtime::Handle::current(),
        );
        let create = EventKind::Create(CreateKind::Any);
        let modify = EventKind::Modify(ModifyKind::Any);
        let remove = EventKind::Remove(RemoveKind::Any);

        dispatch(
            Ok(vec![
                ev(create, &[&p("node_modules/a/index.js")]),
                ev(modify, &[&p("node_modules/a")]),
                // Windows reports a Modify on a directory whenever a direct child
                // changes: `npm install` / any git command touch these constantly.
                ev(modify, &[&p("node_modules")]),
                ev(modify, &[&p(".git")]),
                ev(create, &[&p(".git/index.lock")]),
                ev(create, &[&p("pkg/node_modules/b.ts")]),
                ev(
                    create,
                    &[&p("common/temp/node_modules/.pnpm/x@1/node_modules/x/y.ts")],
                ),
                ev(modify, &[&p("app/lib")]),
                ev(create, &[&p("app/lib/out.ts")]),
                // `rm -rf node_modules`: the directory is gone, its name says enough.
                ev(remove, &[&p("dist")]),
            ]),
            &gate,
        );
        assert!(
            matches!(
                rx.try_recv(),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty)
            ),
            "a batch of ignored-only changes must not trigger an index run"
        );

        dispatch(
            Ok(vec![
                ev(modify, &[&p("src/a.ts")]),
                ev(create, &[&p("node_modules/c.js")]),
                ev(modify, &[&p("app/.gitignore")]),
                ev(modify, &[&p(".gitignore")]),
                // Deleted paths are kept: they may be indexed directories.
                ev(remove, &[&p("src/old")]),
                ev(
                    EventKind::Modify(ModifyKind::Name(RenameMode::Both)),
                    &[&p("src/b.ts"), &p("node_modules/b.ts")],
                ),
            ]),
            &gate,
        );
        let mut got: Vec<(String, String)> = recv_soon(&mut rx)
            .await
            .changes
            .expect("explicit change set")
            .into_iter()
            .map(|c| (c.path, format!("{:?}", c.kind)))
            .collect();
        got.sort();
        let mut want: Vec<(String, String)> = [
            (p("src/a.ts"), ChangeKind::Modified),
            (p("app/.gitignore"), ChangeKind::Modified),
            (p(".gitignore"), ChangeKind::Modified),
            (p("src/old"), ChangeKind::Deleted),
            (p("src/b.ts"), ChangeKind::Deleted),
        ]
        .into_iter()
        .map(|(path, kind)| (path, format!("{kind:?}")))
        .collect();
        want.sort();
        assert_eq!(got, want);
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
