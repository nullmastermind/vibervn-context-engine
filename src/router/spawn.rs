//! Router-side worker SPAWN + readiness handshake.
//!
//! Spawns `context-engine --worker <repo> --port 0`, then reads the child's
//! stdout for the readiness line (`READY_PREFIX port=… pid=… repo=…`) that the
//! worker prints AFTER its listener is bound. Only once we have that line does
//! the router begin proxying — so the router NEVER connects to a port that would
//! refuse (window-A is closed cleanly, not papered over with connect-retries).
//!
//! ## Budget (window A)
//!
//! The MEASURE-gate showed cold `open_db` is 0.6–1.4s even at kernel scale, so
//! the dominant cost here is process spawn + bind, not the DB open. We still
//! bound the wait so a wedged worker can't hang a request forever: if the
//! readiness line doesn't arrive within [`SPAWN_READY_TIMEOUT`], we kill the
//! child and report failure, and the caller degrades to "warming, retry" (never
//! a hang, never connection-refused).

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;
use tracing::{info, warn};

use super::jobobject::JobObject;
use super::registry::ReadyWorker;
use super::worker::READY_PREFIX;

/// Upper bound on the spawn→ready handshake. Generous relative to the measured
/// 0.6–1.4s cold open: covers a cold process load + kernel-scale open + bind
/// with wide margin, while still failing fast on a wedged worker. The caller's
/// own request budget (`mcp_index_wait_secs`, default 50s) is far larger, so a
/// spawn that lands inside this window leaves ample budget for the actual query.
pub const SPAWN_READY_TIMEOUT: Duration = Duration::from_secs(20);

/// Parsed readiness handshake from a worker's stdout.
struct ReadyLine {
    port: u16,
    pid: u32,
}

/// Parse `CONTEXT_ENGINE_WORKER_READY port=NNN pid=NNN repo=...`. Returns None if
/// the line isn't the readiness line or is malformed.
fn parse_ready_line(line: &str) -> Option<ReadyLine> {
    let rest = line.strip_prefix(READY_PREFIX)?;
    // Require a separator (space) right after the prefix so a different token
    // that merely starts with the prefix (e.g. `..._READYISH`) is NOT matched.
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let mut port = None;
    let mut pid = None;
    for tok in rest.split_whitespace() {
        if let Some(v) = tok.strip_prefix("port=") {
            port = v.parse::<u16>().ok();
        } else if let Some(v) = tok.strip_prefix("pid=") {
            pid = v.parse::<u32>().ok();
        }
    }
    Some(ReadyLine {
        port: port?,
        pid: pid?,
    })
}

/// Spawn a worker for `repo` and await its readiness handshake. On success,
/// returns a `ReadyWorker` (port + owned child handle) and assigns the child to
/// the Job Object so it dies with the router. On timeout/failure, kills the
/// child (if any) and returns an error so the caller can degrade.
///
/// `exe`: path to the current executable (so a worker runs the SAME binary).
/// `extra_args`: data-dir / embeddings-dir / bind passthrough so the worker
/// resolves the same paths the router did.
pub async fn spawn_worker(
    exe: &std::path::Path,
    repo: &str,
    extra_args: &[String],
    job: &Arc<JobObject>,
) -> Result<Arc<ReadyWorker>> {
    let mut cmd = Command::new(exe);
    cmd.arg("--worker")
        .arg(repo)
        .arg("--port")
        .arg("0") // ephemeral; worker reports the real port back
        .args(extra_args)
        .stdout(Stdio::piped())
        // Inherit stderr so worker tracing/logs surface in the router's console.
        .stderr(Stdio::inherit())
        .stdin(Stdio::null());

    let mut child: Child = cmd
        .spawn()
        .with_context(|| format!("spawn worker for repo {repo}"))?;

    // Assign to the Job Object IMMEDIATELY after spawn (before we even await
    // readiness) so a router crash during the handshake still can't orphan it.
    job.assign(&child);

    // Take stdout to read the readiness line.
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("worker stdout not captured"))?;

    // Read lines until we see the readiness line or the timeout fires. We must
    // use tokio's async stdio over the std child's pipe: wrap the raw handle.
    let read_ready = async {
        let async_stdout = tokio::process::ChildStdout::from_std(stdout)
            .context("convert worker stdout to async")?;
        let mut lines = BufReader::new(async_stdout).lines();
        while let Some(line) = lines.next_line().await.context("read worker stdout")? {
            if let Some(parsed) = parse_ready_line(&line) {
                return Ok::<ReadyLine, anyhow::Error>(parsed);
            }
            // Non-readiness stdout lines (if any) are ignored; worker logs go to
            // stderr (inherited). Keep reading for the readiness line.
        }
        bail!("worker stdout closed before readiness handshake")
    };

    match tokio::time::timeout(SPAWN_READY_TIMEOUT, read_ready).await {
        Ok(Ok(ready)) => {
            info!(repo = %repo, port = ready.port, pid = ready.pid, "worker ready (handshake)");
            Ok(Arc::new(ReadyWorker {
                port: ready.port,
                pid: ready.pid,
                child: Arc::new(Mutex::new(child)),
            }))
        }
        Ok(Err(e)) => {
            // Handshake failed (stdout closed / parse error). Kill + reap.
            let _ = child.kill();
            let _ = child.wait();
            Err(e.context("worker readiness handshake failed"))
        }
        Err(_elapsed) => {
            // Timeout: wedged worker. Kill + reap so it can't hold the LOCK.
            warn!(repo = %repo, "worker spawn timed out before readiness; killing");
            let _ = child.kill();
            let _ = child.wait();
            bail!("worker spawn timed out after {:?}", SPAWN_READY_TIMEOUT)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_well_formed_ready_line() {
        let line = format!("{READY_PREFIX} port=54321 pid=12345 repo=d:\\projects\\foo");
        let parsed = parse_ready_line(&line).expect("should parse");
        assert_eq!(parsed.port, 54321);
        assert_eq!(parsed.pid, 12345);
    }

    #[test]
    fn ignores_non_ready_lines() {
        // Worker log lines / blank lines must not be mistaken for the handshake.
        assert!(parse_ready_line("INFO some log line").is_none());
        assert!(parse_ready_line("").is_none());
        // A token that merely STARTS with the prefix must NOT match (boundary).
        assert!(
            parse_ready_line("CONTEXT_ENGINE_WORKER_READYISH port=1 pid=2").is_none(),
            "prefix without a separator boundary must not be treated as the handshake"
        );
    }

    #[test]
    fn rejects_ready_line_missing_fields() {
        // Prefix present but no port/pid → None (don't proxy to a bogus port).
        assert!(parse_ready_line(READY_PREFIX).is_none());
        assert!(parse_ready_line(&format!("{READY_PREFIX} port=80")).is_none());
        assert!(parse_ready_line(&format!("{READY_PREFIX} pid=80")).is_none());
    }

    #[test]
    fn rejects_unparseable_port() {
        assert!(parse_ready_line(&format!("{READY_PREFIX} port=notanum pid=5")).is_none());
    }
}
