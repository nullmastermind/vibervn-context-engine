//! Per-repo automatic retry scheduling with exponential backoff.
//!
//! WHY: a failed run (or one that left files in `pending_retry`) used to wait for
//! the next natural trigger — a watcher event, a query, a manual Index. Until
//! then the repo sat fail-closed (no vector results) or silently missing the
//! skipped files. The consumer now schedules one delayed retry per repo, backing
//! off 1 m → 2 m → 4 m … capped at 30 m, reset by a clean run. Fatal errors
//! (auth/config, e.g. an expired key) are never scheduled: retrying cannot fix
//! them and would only hammer the endpoint.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

const BASE_DELAY: Duration = Duration::from_secs(60);
const MAX_DELAY: Duration = Duration::from_secs(30 * 60);

#[derive(Default, Clone, Copy)]
struct State {
    attempt: u32,
    pending: bool,
}

#[derive(Default)]
pub struct RetryBackoff {
    state: Mutex<HashMap<String, State>>,
}

impl RetryBackoff {
    /// Reserve the next retry for `repo`. Returns the delay to wait before
    /// sending it, or `None` when a retry is already scheduled (at most one per
    /// repo). Each reservation doubles the next delay, up to the cap.
    pub fn schedule(&self, repo: &str) -> Option<Duration> {
        let mut map = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let st = map.entry(repo.to_owned()).or_default();
        if st.pending {
            return None;
        }
        st.pending = true;
        let delay = BASE_DELAY
            .checked_mul(1u32 << st.attempt.min(16))
            .unwrap_or(MAX_DELAY)
            .min(MAX_DELAY);
        st.attempt = st.attempt.saturating_add(1);
        Some(delay)
    }

    /// The scheduled retry fired (its trigger was sent): a new one may be
    /// scheduled after the run it causes.
    pub fn fired(&self, repo: &str) {
        let mut map = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(st) = map.get_mut(repo) {
            st.pending = false;
        }
    }

    /// A clean run: the next failure starts from the base delay again. A retry
    /// already scheduled still fires (harmless: it finds nothing to do).
    pub fn reset(&self, repo: &str) {
        let mut map = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(st) = map.get_mut(repo) {
            st.attempt = 0;
            if !st.pending {
                map.remove(repo);
            }
        }
    }
}

/// True when a failed run's error is fatal for automatic retry: an embedding
/// auth/config failure (`PipelineAbort::EmbeddingFailed`, including the full
/// rebuild preflight). Cancellation is handled separately by the consumer.
pub fn is_fatal_index_error(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<crate::indexing::pipeline::PipelineAbort>(),
            Some(crate::indexing::pipeline::PipelineAbort::EmbeddingFailed(_))
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delays_double_up_to_cap_and_reset() {
        let b = RetryBackoff::default();
        let mut seen = Vec::new();
        for _ in 0..8 {
            seen.push(b.schedule("r").expect("none pending").as_secs());
            b.fired("r");
        }
        assert_eq!(seen, vec![60, 120, 240, 480, 960, 1800, 1800, 1800]);
        b.reset("r");
        assert_eq!(b.schedule("r").unwrap().as_secs(), 60, "reset starts over");
    }

    #[test]
    fn at_most_one_pending_retry_per_repo() {
        let b = RetryBackoff::default();
        assert!(b.schedule("r").is_some());
        assert!(b.schedule("r").is_none(), "second schedule coalesces");
        assert!(b.schedule("other").is_some(), "repos are independent");
        b.fired("r");
        assert!(b.schedule("r").is_some());
    }

    #[test]
    fn fatal_classification() {
        use crate::indexing::pipeline::PipelineAbort;
        let fatal: anyhow::Error = PipelineAbort::EmbeddingFailed("403".into()).into();
        assert!(is_fatal_index_error(&fatal.context("wrapped")));
        let transient = anyhow::anyhow!("connection reset").context("embed");
        assert!(!is_fatal_index_error(&transient));
        let cancelled: anyhow::Error = PipelineAbort::Cancelled.into();
        assert!(!is_fatal_index_error(&cancelled));
    }
}
