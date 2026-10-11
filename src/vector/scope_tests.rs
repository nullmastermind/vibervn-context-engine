//! Tests for the code/docs scope pre-filter inside `VectorIndex::search_filtered`.
//!
//! Expected values are hand-derived literals (never computed by the code under
//! test). Every mutator of `chunk_ids` has a test proving the parallel `ext_ids`
//! array stays aligned — a silent misalignment would classify rows wrongly with
//! no panic, so it is asserted through observable search behavior, not only len.

use super::*;

fn chunk(file: &str, start: u32) -> ChunkId {
    ChunkId {
        file: file.to_string(),
        line_start: start,
        line_end: start + 1,
    }
}

/// 2-d embedding at angle `deg` — cosine against another angle is cos(Δ).
fn at(deg: f32) -> Vec<f32> {
    let r = deg.to_radians();
    vec![r.cos(), r.sin()]
}

fn default_docs() -> Vec<&'static str> {
    vec!["md", "mdx", "rst", "txt"]
}

fn files(results: &[SearchResult]) -> Vec<String> {
    let mut v: Vec<String> = results.iter().map(|r| r.chunk_id.file.clone()).collect();
    v.sort();
    v
}

#[test]
fn ext_key_table() {
    let cases = [
        ("src/a.RS", "rs"),
        (r"C:\proj\Readme.MD", "md"),
        ("a.test.ts", "ts"),
        ("/x/Dockerfile", "dockerfile"),
        ("x/.gitignore", ".gitignore"),
        ("noext", "noext"),
        ("dir.d/file", "file"),
        ("weird.", "weird."),
        ("", ""),
    ];
    for (path, want) in cases {
        assert_eq!(ext_key(path), want, "ext_key({path:?})");
    }
}

#[test]
fn scoped_search_returns_only_requested_class() {
    let mut idx = VectorIndex::new();
    idx.insert(&[
        (chunk("/r/a.rs", 1), at(0.0)),
        (chunk("/r/b.json", 1), at(10.0)),
        (chunk("/r/README.md", 1), at(5.0)),
        (chunk("/r/notes.TXT", 1), at(15.0)),
    ]);
    let q = at(0.0);

    let code = ScopeFilter::new(Scope::Code, default_docs());
    let docs = ScopeFilter::new(Scope::Docs, default_docs());

    assert_eq!(
        files(&idx.search_filtered(&q, 10, Some(&code))),
        vec!["/r/a.rs", "/r/b.json"]
    );
    assert_eq!(
        files(&idx.search_filtered(&q, 10, Some(&docs))),
        vec!["/r/README.md", "/r/notes.TXT"]
    );
    // No filter = both classes, unchanged behavior.
    assert_eq!(idx.search_filtered(&q, 10, None).len(), 4);
}

/// The reason the filter must run INSIDE the scan: 100 docs rows all outscore the
/// 5 code rows, so a post-filter over top-5 would return nothing.
#[test]
fn scoped_search_is_not_starved_by_higher_scoring_other_class() {
    let mut idx = VectorIndex::new();
    let mut rows = Vec::new();
    for i in 0..100 {
        rows.push((chunk(&format!("/r/doc{i}.md"), 1), at(0.0)));
    }
    for i in 0..5 {
        rows.push((chunk(&format!("/r/code{i}.rs"), 1), at(80.0)));
    }
    idx.insert(&rows);
    let q = at(0.0);

    // Sanity: the unfiltered top-5 is all docs, so the test is meaningful.
    assert!(
        idx.search(&q, 5)
            .iter()
            .all(|r| r.chunk_id.file.ends_with(".md"))
    );

    let code = ScopeFilter::new(Scope::Code, default_docs());
    let got = idx.search_filtered(&q, 5, Some(&code));
    assert_eq!(got.len(), 5);
    assert!(got.iter().all(|r| r.chunk_id.file.ends_with(".rs")));
}

/// `select_nth_unstable_by(k - 1, ..)` underflows when nothing survives the
/// filter. This is the exact state D8's fallback depends on (repo has no files of
/// the requested class).
#[test]
fn scoped_search_with_zero_matching_rows_returns_empty() {
    let mut idx = VectorIndex::new();
    idx.insert(&[(chunk("/r/README.md", 1), at(0.0))]);
    let code = ScopeFilter::new(Scope::Code, default_docs());
    assert!(idx.search_filtered(&at(0.0), 5, Some(&code)).is_empty());

    let mut only_code = VectorIndex::new();
    only_code.insert(&[(chunk("/r/a.rs", 1), at(0.0))]);
    let docs = ScopeFilter::new(Scope::Docs, default_docs());
    assert!(
        only_code
            .search_filtered(&at(0.0), 5, Some(&docs))
            .is_empty()
    );
}

#[test]
fn docs_set_is_taken_from_the_filter_not_baked_into_the_shard() {
    let mut idx = VectorIndex::new();
    idx.insert(&[
        (chunk("/r/a.rs", 1), at(0.0)),
        (chunk("/r/notes.txt", 1), at(5.0)),
        (chunk("/r/guide.md", 1), at(10.0)),
    ]);
    let q = at(0.0);

    let md_only = ScopeFilter::new(Scope::Docs, ["md"]);
    assert_eq!(
        files(&idx.search_filtered(&q, 10, Some(&md_only))),
        vec!["/r/guide.md"]
    );
    // Same warm index, setting changed: .txt moves from code to docs.
    let md_txt = ScopeFilter::new(Scope::Docs, ["md", "txt"]);
    assert_eq!(
        files(&idx.search_filtered(&q, 10, Some(&md_txt))),
        vec!["/r/guide.md", "/r/notes.txt"]
    );
    let code_md_only = ScopeFilter::new(Scope::Code, ["md"]);
    assert_eq!(
        files(&idx.search_filtered(&q, 10, Some(&code_md_only))),
        vec!["/r/a.rs", "/r/notes.txt"]
    );
    // Empty docs list is legal: nothing is docs.
    let none: [&str; 0] = [];
    assert!(
        idx.search_filtered(&q, 10, Some(&ScopeFilter::new(Scope::Docs, none)))
            .is_empty()
    );
}

#[test]
fn docs_extensions_are_matched_case_insensitively() {
    let mut idx = VectorIndex::new();
    idx.insert(&[(chunk("/r/GUIDE.MD", 1), at(0.0))]);
    let docs = ScopeFilter::new(Scope::Docs, ["MD", " .md "]);
    assert_eq!(idx.search_filtered(&at(0.0), 5, Some(&docs)).len(), 1);
}

/// Each shard interns extensions in its own order, so ids are NOT comparable
/// across shards. `merge` must remap `other`'s ids through its own table.
#[test]
fn merge_remaps_ext_ids_between_differently_ordered_tables() {
    let mut a = VectorIndex::new();
    a.insert(&[
        (chunk("/r/x.rs", 1), at(0.0)),
        (chunk("/r/y.md", 1), at(1.0)),
    ]); // a: rs=0, md=1
    let mut b = VectorIndex::new();
    b.insert(&[
        (chunk("/r/z.md", 1), at(2.0)),
        (chunk("/r/w.rs", 1), at(3.0)),
    ]); // b: md=0, rs=1
    a.merge(b);

    assert_eq!(a.ext_ids.len(), a.chunk_ids.len());
    let q = at(0.0);
    let docs = ScopeFilter::new(Scope::Docs, default_docs());
    let code = ScopeFilter::new(Scope::Code, default_docs());
    assert_eq!(
        files(&a.search_filtered(&q, 10, Some(&docs))),
        vec!["/r/y.md", "/r/z.md"]
    );
    assert_eq!(
        files(&a.search_filtered(&q, 10, Some(&code))),
        vec!["/r/w.rs", "/r/x.rs"]
    );
}

#[test]
fn merge_into_empty_adopts_other_classification() {
    let mut a = VectorIndex::new();
    let mut b = VectorIndex::new();
    b.insert(&[
        (chunk("/r/z.md", 1), at(0.0)),
        (chunk("/r/w.rs", 1), at(1.0)),
    ]);
    a.merge(b);
    let docs = ScopeFilter::new(Scope::Docs, default_docs());
    assert_eq!(
        files(&a.search_filtered(&at(0.0), 10, Some(&docs))),
        vec!["/r/z.md"]
    );
}

fn mixed_index() -> VectorIndex {
    let mut idx = VectorIndex::new();
    idx.insert(&[
        (chunk("/r/a.rs", 1), at(0.0)),
        (chunk("/r/b.md", 1), at(1.0)),
        (chunk("/r/c.rs", 1), at(2.0)),
        (chunk("/r/d.md", 1), at(3.0)),
        (chunk("/other/e.md", 1), at(4.0)),
    ]);
    idx
}

fn assert_classes(idx: &VectorIndex, docs: &[&str], code: &[&str]) {
    assert_eq!(idx.ext_ids.len(), idx.chunk_ids.len(), "ext_ids aligned");
    let q = at(0.0);
    let d = ScopeFilter::new(Scope::Docs, default_docs());
    let c = ScopeFilter::new(Scope::Code, default_docs());
    assert_eq!(files(&idx.search_filtered(&q, 20, Some(&d))), docs);
    assert_eq!(files(&idx.search_filtered(&q, 20, Some(&c))), code);
}

#[test]
fn remove_files_keeps_classes_aligned() {
    let mut idx = mixed_index();
    // Removing a code row from the front swaps later rows (mixed classes) down.
    idx.remove_files(&std::collections::HashSet::from(["/r/a.rs", "/r/d.md"]));
    assert_classes(&idx, &["/other/e.md", "/r/b.md"], &["/r/c.rs"]);
}

#[test]
fn remove_file_keeps_classes_aligned() {
    let mut idx = mixed_index();
    idx.remove_file("/r/a.rs"); // swap_remove pulls the LAST row (a docs row) into slot 0
    assert_classes(&idx, &["/other/e.md", "/r/b.md", "/r/d.md"], &["/r/c.rs"]);
}

#[test]
fn remove_repo_keeps_classes_aligned() {
    let mut idx = mixed_index();
    idx.remove_repo("/r");
    assert_classes(&idx, &["/other/e.md"], &[]);
}

#[test]
fn clear_resets_classification() {
    let mut idx = mixed_index();
    idx.clear();
    assert_eq!(idx.ext_ids.len(), 0);
    idx.insert(&[(chunk("/r/new.md", 1), at(0.0))]);
    assert_classes(&idx, &["/r/new.md"], &[]);
}

fn two_shard_index() -> (ShardedVectorIndex, Vec<String>) {
    let scope = vec!["/r/a".to_string(), "/r/b".to_string()];
    let mut sharded = ShardedVectorIndex::new(1 << 20);
    for (repo, rows) in [
        (
            "/r/a",
            vec![
                (chunk("/r/a/x.rs", 1), at(0.0)),
                (chunk("/r/a/README.md", 1), at(1.0)),
            ],
        ),
        (
            "/r/b",
            vec![
                (chunk("/r/b/y.py", 1), at(2.0)),
                (chunk("/r/b/GUIDE.md", 1), at(3.0)),
            ],
        ),
    ] {
        let mut vi = VectorIndex::new();
        vi.insert(&rows);
        sharded.install_shard(repo, vi, "id".to_owned(), &scope);
    }
    (sharded, scope)
}

#[test]
fn sharded_single_repo_search_honors_scope() {
    let (sharded, _) = two_shard_index();
    let docs = ScopeFilter::new(Scope::Docs, default_docs());
    let out = sharded.search_scoped_with(&at(0.0), 10, "/r/b", "id", Some(&docs));
    let ScopedSearch::Hit(hit) = out else {
        panic!("expected a hit");
    };
    assert_eq!(files(&hit.results), vec!["/r/b/GUIDE.md"]);
}

#[test]
fn sharded_fan_out_search_honors_scope_across_shards() {
    let (sharded, scope) = two_shard_index();
    let code = ScopeFilter::new(Scope::Code, default_docs());
    let out = sharded.search_all_scoped_with(&at(0.0), 10, &scope, "id", Some(&code));
    assert_eq!(files(&out.results), vec!["/r/a/x.rs", "/r/b/y.py"]);
    // Unfiltered entry points are unchanged.
    let all = sharded.search_all_scoped(&at(0.0), 10, &scope, "id");
    assert_eq!(all.results.len(), 4);
}

/// More distinct extensions than a u16 id can hold: must not wrap into a
/// misclassified id. Overflowed rows land on a sentinel that counts as code.
#[test]
fn ext_table_overflow_saturates_to_code_without_wrapping() {
    let mut idx = VectorIndex::new();
    let n = u16::MAX as usize + 10;
    let rows: Vec<(ChunkId, Vec<f32>)> = (0..n)
        .map(|i| (chunk(&format!("/r/f.e{i}"), 1), at(0.0)))
        .collect();
    idx.insert(&rows);
    assert_eq!(idx.ext_ids.len(), n);

    // "e0" interned first, so it is a real id and honors the docs list...
    let docs = ScopeFilter::new(Scope::Docs, ["e0"]);
    assert_eq!(
        files(&idx.search_filtered(&at(0.0), 5, Some(&docs))),
        vec!["/r/f.e0"]
    );
    // ...and the last (overflowed) row is code, not a wrapped docs id.
    let code = ScopeFilter::new(Scope::Code, ["e0"]);
    let last = format!("/r/f.e{}", n - 1);
    let got = idx.search_filtered(&at(0.0), n, Some(&code));
    assert!(got.iter().any(|r| r.chunk_id.file == last));
    assert!(got.iter().all(|r| r.chunk_id.file != "/r/f.e0"));
}
