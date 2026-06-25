//! Router-side WORKER REGISTRY: the source of truth for "which repos have a live
//! worker, on what port, and is it still alive?" plus single-flight spawn.
//!
//! ## State machine (per repo)
//!
//! ```text
//!   (absent) ──spawn──▶ Spawning ──ready line──▶ Ready{port,child}
//!        ▲                  │                          │
//!        │                  │ spawn/handshake failed   │ child exited / killed
//!        └──────────────────┴──────────────────────────┘
//! ```
//!
//! - **Spawning**: a worker process has been launched; we are awaiting its
//!   readiness handshake (stdout line). Concurrent requests for the same repo
//!   DO NOT spawn a second process — they await the same `ready` notify
//!   (single-flight, mirroring `IndexEngine::warm_locks` / `store::OPEN_GATES`
//!   lifted to the process level).
//! - **Ready**: the worker accepted on `port`; the router proxies to it. We hold
//!   the `Child` handle so we can `try_wait` (reuse-safe liveness — never PID
//!   lookup, which is vulnerable to PID reuse and the `STILL_ACTIVE`=259 trap)
//!   and so the OS parent/child relationship + Job Object keep it from orphaning.
//! - **absent/Dead**: no live worker; the next request spawns one.
//!
//! ## Why we keep the `Child`
//!
//! `Child::try_wait()` checks liveness via the retained OS handle, not the PID,
//! so it is immune to PID reuse and never misreads exit code 259. Before
//! respawning we confirm the prior child is truly dead with `try_wait`, so we
//! never race the old worker's still-draining RocksDB LOCK.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{Mutex, Notify, RwLock};

/// A live, ready worker: the port it accepted on + the owned child process
/// handle (for reuse-safe liveness checks and lifetime ownership).
pub struct ReadyWorker {
    pub port: u16,
    pub pid: u32,
    /// Owned child handle. `try_wait` on this is the reuse-safe liveness check.
    /// Behind a `Mutex` because `try_wait`/`kill` take `&mut Child` but the
    /// registry entry is shared.
    pub child: Arc<Mutex<std::process::Child>>,
}

/// Per-repo worker state.
pub enum WorkerState {
    /// A spawn is in flight; awaiters block on `ready` until it flips to Ready
    /// (or the entry is removed on failure).
    Spawning { ready: Arc<Notify> },
    /// Worker is accepting on `port`.
    Ready(Arc<ReadyWorker>),
}

/// The router's live worker table, keyed by NORMALIZED repo path.
#[derive(Clone, Default)]
pub struct Registry {
    inner: Arc<RwLock<HashMap<String, WorkerState>>>,
}

/// Outcome of [`Registry::acquire`]: either a ready worker to proxy to, or this
/// caller has been elected to perform the spawn (and must call
/// [`Registry::publish_ready`] / [`Registry::abandon_spawn`] when done).
pub enum Acquire {
    /// A live worker already exists (or another caller's spawn completed) —
    /// proxy here.
    Ready(Arc<ReadyWorker>),
    /// This caller won the single-flight election and must spawn the worker.
    /// `ready` is the notify to signal awaiters once published.
    SpawnElected { ready: Arc<Notify> },
    /// A spawn is already in flight by another caller; await `ready` then
    /// re-acquire.
    AwaitSpawn { ready: Arc<Notify> },
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Single-flight acquire. Returns:
    /// - `Ready` if a live worker exists,
    /// - `SpawnElected` if this caller should spawn (state set to `Spawning`),
    /// - `AwaitSpawn` if another caller is mid-spawn.
    ///
    /// Liveness: a `Ready` entry whose child has exited is treated as absent
    /// (the caller is elected to respawn). The dead-child check uses `try_wait`
    /// on the owned handle (reuse-safe).
    pub async fn acquire(&self, repo: &str) -> Acquire {
        // Fast path: read lock, return Ready if present + alive.
        {
            let map = self.inner.read().await;
            match map.get(repo) {
                Some(WorkerState::Ready(w)) => {
                    if is_child_alive(w).await {
                        return Acquire::Ready(w.clone());
                    }
                    // dead — fall through to write path to respawn
                }
                Some(WorkerState::Spawning { ready }) => {
                    return Acquire::AwaitSpawn {
                        ready: ready.clone(),
                    };
                }
                None => {}
            }
        }

        // Slow path: write lock, re-check (another task may have changed it),
        // then either elect-to-spawn or await.
        let mut map = self.inner.write().await;
        match map.get(repo) {
            Some(WorkerState::Ready(w)) => {
                if is_child_alive(w).await {
                    return Acquire::Ready(w.clone());
                }
                // Confirmed dead under the write lock → reap + elect to respawn.
                // (try_wait already reaped the zombie; just replace the entry.)
            }
            Some(WorkerState::Spawning { ready }) => {
                return Acquire::AwaitSpawn {
                    ready: ready.clone(),
                };
            }
            None => {}
        }
        let ready = Arc::new(Notify::new());
        map.insert(
            repo.to_string(),
            WorkerState::Spawning {
                ready: ready.clone(),
            },
        );
        Acquire::SpawnElected { ready }
    }

    /// Publish a freshly-spawned ready worker and wake all awaiters.
    pub async fn publish_ready(&self, repo: &str, worker: Arc<ReadyWorker>, ready: &Arc<Notify>) {
        {
            let mut map = self.inner.write().await;
            map.insert(repo.to_string(), WorkerState::Ready(worker));
        }
        ready.notify_waiters();
    }

    /// Abandon a failed spawn: remove the `Spawning` entry and wake awaiters so
    /// they re-acquire (and one of them re-elects to spawn).
    pub async fn abandon_spawn(&self, repo: &str, ready: &Arc<Notify>) {
        {
            let mut map = self.inner.write().await;
            // Only remove if it's still the Spawning we created (don't clobber a
            // Ready another path may have installed).
            if matches!(map.get(repo), Some(WorkerState::Spawning { .. })) {
                map.remove(repo);
            }
        }
        ready.notify_waiters();
    }

    /// Kill + drop a repo's worker (used on explicit teardown, e.g. repo
    /// removal). Idempotent.
    pub async fn kill(&self, repo: &str) {
        let entry = {
            let mut map = self.inner.write().await;
            map.remove(repo)
        };
        if let Some(WorkerState::Ready(w)) = entry {
            let mut child = w.child.lock().await;
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// Snapshot of repos that currently have a Ready worker (for diagnostics /
    /// the router's "which repos are warm" view).
    pub async fn ready_repos(&self) -> Vec<String> {
        self.inner
            .read()
            .await
            .iter()
            .filter(|(_, v)| matches!(v, WorkerState::Ready(_)))
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Snapshot of repos whose worker is mid-SPAWN (an action triggered a worker
    /// that hasn't finished binding yet). `ready_repos` excludes these — but the
    /// status view needs them so the badge shows "indexing/starting" during the
    /// ~1-2s spawn window instead of falling through to the cold sidecar and
    /// flashing "not indexed" right after the user triggered an action.
    pub async fn spawning_repos(&self) -> Vec<String> {
        self.inner
            .read()
            .await
            .iter()
            .filter(|(_, v)| matches!(v, WorkerState::Spawning { .. }))
            .map(|(k, _)| k.clone())
            .collect()
    }
}

/// Reuse-safe liveness check: `try_wait` on the OWNED child handle. Returns true
/// if the child is still running. A child that has exited is reaped here (so it
/// does not linger as a zombie on Unix) and reported dead.
async fn is_child_alive(w: &ReadyWorker) -> bool {
    let mut child = w.child.lock().await;
    match child.try_wait() {
        Ok(Some(_status)) => false, // exited — reaped
        Ok(None) => true,           // still running
        Err(_) => false,            // can't determine → treat as dead, respawn
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SINGLE-FLIGHT: the first acquire on an absent repo is elected to spawn;
    /// a concurrent acquire for the SAME repo must NOT also be elected (that
    /// would spawn a second worker racing the RocksDB LOCK). It awaits instead.
    #[tokio::test]
    async fn concurrent_acquire_elects_exactly_one_spawner() {
        let reg = Registry::new();
        let a = reg.acquire("repoX").await;
        let b = reg.acquire("repoX").await;

        let a_elected = matches!(a, Acquire::SpawnElected { .. });
        let b_awaits = matches!(b, Acquire::AwaitSpawn { .. });
        assert!(a_elected, "first caller must be elected to spawn");
        assert!(
            b_awaits,
            "second concurrent caller must AWAIT, never spawn a second worker (LOCK race)"
        );
    }

    /// A repo mid-SPAWN (acquire elected a spawner but it hasn't published Ready)
    /// must appear in `spawning_repos()` and NOT in `ready_repos()`. This is what
    /// lets `get_index_status` report "indexing/starting" during the ~1-2s spawn
    /// window instead of falling through to the cold sidecar and flashing
    /// "not_indexed" right after the user triggered an action (the reported bug).
    #[tokio::test]
    async fn spawning_repo_is_reported_spawning_not_ready() {
        let reg = Registry::new();
        // Elect a spawner → the repo's state is now `Spawning`.
        assert!(matches!(
            reg.acquire("repoS").await,
            Acquire::SpawnElected { .. }
        ));
        assert!(
            reg.spawning_repos().await.contains(&"repoS".to_string()),
            "a mid-spawn repo must be reported by spawning_repos()"
        );
        assert!(
            !reg.ready_repos().await.contains(&"repoS".to_string()),
            "a mid-spawn repo must NOT be in ready_repos() (it isn't accepting yet)"
        );
    }

    /// A different repo acquired concurrently IS independently elected (distinct
    /// repos open distinct DBs — no shared LOCK, so they spawn in parallel).
    #[tokio::test]
    async fn distinct_repos_each_elect() {
        let reg = Registry::new();
        let a = reg.acquire("repoA").await;
        let b = reg.acquire("repoB").await;
        assert!(matches!(a, Acquire::SpawnElected { .. }));
        assert!(matches!(b, Acquire::SpawnElected { .. }));
    }

    /// After a failed spawn is abandoned, the entry is cleared so the next
    /// acquire re-elects (retry), rather than getting stuck in Spawning forever.
    #[tokio::test]
    async fn abandon_allows_reelection() {
        let reg = Registry::new();
        let ready = match reg.acquire("repoY").await {
            Acquire::SpawnElected { ready } => ready,
            _ => panic!("first must be elected"),
        };
        reg.abandon_spawn("repoY", &ready).await;
        // Next acquire should be elected again (not stuck awaiting a dead spawn).
        assert!(
            matches!(reg.acquire("repoY").await, Acquire::SpawnElected { .. }),
            "after abandon, the repo must be re-electable for a fresh spawn"
        );
    }
}
