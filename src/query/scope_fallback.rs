//! One-shot code↔docs fallback for scoped retrieval.
//!
//! `codebase-retrieval` searches exactly one class (code or docs) per call. When
//! the requested class yields no candidates at all, the OTHER class is searched
//! once and returned, flagged so the caller can tell the agent what it actually
//! got. Kept as a small generic function over a "gather" closure so the control
//! flow (when to retry, when NOT to) is unit-tested without a live index.

use std::collections::HashMap;
use std::future::Future;

use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use crate::query::merger::MergeChunk;
use crate::vector::ScopeFilter;

/// Candidates for one scan of one scope: vector search → hydrate → post-filters.
pub(crate) struct Gathered {
    pub base_chunks: Vec<MergeChunk>,
    /// The shard was still loading, or a candidate lost the publication race. An
    /// empty `base_chunks` with this set means "retry shortly", never "no results".
    pub warming: bool,
    /// Wall time of the vector scan(s) that produced this, for `QueryTiming`.
    pub search_ms: u64,
    /// DB handles snapshot, reused by graph expansion later in the pipeline.
    pub db_map: HashMap<String, Surreal<Db>>,
}

/// What was actually served and under which scope.
pub(crate) struct Served {
    pub gathered: Gathered,
    /// The scope the returned candidates belong to (`None` = unscoped query).
    pub scope: Option<ScopeFilter>,
    /// True only when the requested scope was empty and the other was served.
    pub fallback: bool,
}

/// Gather candidates for `requested`; if that is empty (and not merely warming),
/// retry ONCE with the opposite scope.
///
/// Fallback is deliberately narrow:
/// - only for a scoped request (`None` = unscoped, nothing to fall back from);
/// - never when the first scan reported `warming` — that would mask a retryable
///   state as a different answer;
/// - at most one retry, so two empty classes cannot ping-pong;
/// - "empty" is "no candidates survived", not a relevance threshold: cosine
///   always returns the nearest rows, so a weak match is a result, not an empty.
pub(crate) async fn gather_with_fallback<F, Fut>(
    requested: Option<ScopeFilter>,
    mut gather: F,
) -> Served
where
    F: FnMut(Option<ScopeFilter>) -> Fut,
    Fut: Future<Output = Gathered>,
{
    let mut first = gather(requested.clone()).await;
    let Some(requested_filter) = requested else {
        return Served {
            gathered: first,
            scope: None,
            fallback: false,
        };
    };
    if !first.base_chunks.is_empty() || first.warming {
        return Served {
            gathered: first,
            scope: Some(requested_filter),
            fallback: false,
        };
    }

    let other = requested_filter.opposite();
    let mut second = gather(Some(other.clone())).await;
    if !second.base_chunks.is_empty() {
        second.search_ms += first.search_ms;
        return Served {
            gathered: second,
            scope: Some(other),
            fallback: true,
        };
    }

    // Both empty. Report the requested scope, but if the retry saw an incomplete
    // index surface that instead of a misleading "no results".
    first.search_ms += second.search_ms;
    first.warming = second.warming;
    Served {
        gathered: first,
        scope: Some(requested_filter),
        fallback: false,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;

    use super::*;
    use crate::query::merger::MergeChunk;
    use crate::vector::{Scope, ScopeFilter};

    fn chunk(file: &str) -> MergeChunk {
        MergeChunk {
            file: file.to_string(),
            line_start: 1,
            line_end: 2,
            score: 0.5,
            content: String::new(),
            symbol: None,
            symbol_fqn: None,
            symbol_kind: None,
        }
    }

    fn gathered(files: &[&str], warming: bool, search_ms: u64) -> Gathered {
        Gathered {
            base_chunks: files.iter().map(|f| chunk(f)).collect(),
            warming,
            search_ms,
            db_map: HashMap::new(),
        }
    }

    fn filter(scope: Scope) -> Option<ScopeFilter> {
        Some(ScopeFilter::new(scope, ["md"]))
    }

    /// Drive `gather_with_fallback` with canned per-scope data and record which
    /// scopes were actually searched, in order.
    async fn run(
        requested: Option<ScopeFilter>,
        code: Gathered,
        docs: Gathered,
    ) -> (Served, Vec<Option<Scope>>) {
        let calls: RefCell<Vec<Option<Scope>>> = RefCell::new(Vec::new());
        let code = RefCell::new(Some(code));
        let docs = RefCell::new(Some(docs));
        let served = gather_with_fallback(requested, |scope: Option<ScopeFilter>| {
            calls.borrow_mut().push(scope.as_ref().map(|s| s.scope()));
            let out = match scope.as_ref().map(|s| s.scope()) {
                Some(Scope::Docs) => docs.borrow_mut().take(),
                _ => code.borrow_mut().take(),
            }
            .expect("each scope is searched at most once");
            async move { out }
        })
        .await;
        (served, calls.into_inner())
    }

    fn served_files(s: &Served) -> Vec<&str> {
        s.gathered
            .base_chunks
            .iter()
            .map(|c| c.file.as_str())
            .collect()
    }

    #[tokio::test]
    async fn non_empty_requested_scope_is_served_without_a_second_search() {
        let (s, calls) = run(
            filter(Scope::Code),
            gathered(&["a.rs"], false, 5),
            gathered(&["README.md"], false, 7),
        )
        .await;
        assert_eq!(calls, vec![Some(Scope::Code)]);
        assert!(!s.fallback);
        assert_eq!(s.scope.as_ref().map(|f| f.scope()), Some(Scope::Code));
        assert_eq!(served_files(&s), vec!["a.rs"]);
    }

    #[tokio::test]
    async fn empty_code_falls_back_to_docs_once_and_says_so() {
        let (s, calls) = run(
            filter(Scope::Code),
            gathered(&[], false, 5),
            gathered(&["README.md"], false, 7),
        )
        .await;
        assert_eq!(calls, vec![Some(Scope::Code), Some(Scope::Docs)]);
        assert!(s.fallback);
        assert_eq!(s.scope.as_ref().map(|f| f.scope()), Some(Scope::Docs));
        assert_eq!(served_files(&s), vec!["README.md"]);
        assert_eq!(
            s.gathered.search_ms, 12,
            "search time covers both scans, not just the fallback"
        );
    }

    #[tokio::test]
    async fn empty_docs_falls_back_to_code() {
        let (s, calls) = run(
            filter(Scope::Docs),
            gathered(&["a.rs"], false, 1),
            gathered(&[], false, 1),
        )
        .await;
        assert_eq!(calls, vec![Some(Scope::Docs), Some(Scope::Code)]);
        assert!(s.fallback);
        assert_eq!(s.scope.as_ref().map(|f| f.scope()), Some(Scope::Code));
        assert_eq!(served_files(&s), vec!["a.rs"]);
    }

    #[tokio::test]
    async fn both_empty_reports_the_requested_scope_and_no_fallback() {
        let (s, calls) = run(
            filter(Scope::Code),
            gathered(&[], false, 1),
            gathered(&[], false, 1),
        )
        .await;
        assert_eq!(calls.len(), 2, "exactly one retry, never a loop");
        assert!(!s.fallback);
        assert_eq!(s.scope.as_ref().map(|f| f.scope()), Some(Scope::Code));
        assert!(s.gathered.base_chunks.is_empty());
        assert!(!s.gathered.warming);
    }

    /// A shard that is still loading is "retry shortly", not "this class is
    /// empty": falling back would mask the warming outcome.
    #[tokio::test]
    async fn warming_empty_result_never_triggers_fallback() {
        let (s, calls) = run(
            filter(Scope::Code),
            gathered(&[], true, 1),
            gathered(&["README.md"], false, 1),
        )
        .await;
        assert_eq!(calls, vec![Some(Scope::Code)]);
        assert!(!s.fallback);
        assert!(s.gathered.warming);
        assert!(s.gathered.base_chunks.is_empty());
    }

    /// If the fallback scan itself reports an incomplete index, surface the
    /// warming signal instead of a misleading "no results".
    #[tokio::test]
    async fn warming_during_fallback_scan_is_surfaced() {
        let (s, _) = run(
            filter(Scope::Code),
            gathered(&[], false, 1),
            gathered(&[], true, 1),
        )
        .await;
        assert!(!s.fallback);
        assert!(s.gathered.warming);
    }

    #[tokio::test]
    async fn no_requested_scope_means_no_fallback_even_when_empty() {
        let (s, calls) = run(
            None,
            gathered(&[], false, 1),
            gathered(&["README.md"], false, 1),
        )
        .await;
        assert_eq!(calls, vec![None]);
        assert!(!s.fallback);
        assert!(s.scope.is_none());
        assert!(s.gathered.base_chunks.is_empty());
    }
}
