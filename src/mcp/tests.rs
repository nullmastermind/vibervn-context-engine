//! MCP formatting, merging, and file-key tests. Readiness policy tests live in
//! the readiness tests beside the policy itself.

use super::*;

#[test]
fn vector_only_fast_path_only_for_resolve_edges() {
    let resolving = crate::indexing::RepoStatus {
        state: crate::indexing::IndexState::Indexing,
        phase: crate::indexing::IndexPhase::ResolveEdges,
        ..Default::default()
    };
    assert!(readiness::is_resolve_edges_status(Some(&resolving)));

    let embedding = crate::indexing::RepoStatus {
        state: crate::indexing::IndexState::Indexing,
        phase: crate::indexing::IndexPhase::Embedding,
        ..Default::default()
    };
    assert!(!readiness::is_resolve_edges_status(Some(&embedding)));

    let idle_resolve = crate::indexing::RepoStatus {
        state: crate::indexing::IndexState::Idle,
        phase: crate::indexing::IndexPhase::ResolveEdges,
        ..Default::default()
    };
    assert!(!readiness::is_resolve_edges_status(Some(&idle_resolve)));
    assert!(!readiness::is_resolve_edges_status(None));
}

#[test]
fn file_retrieval_db_key_windows_backslash_input() {
    let repo = r"D:\projects\Python\local-context-engine";
    let file_path = r"context-engine-rs\Cargo.toml";
    let db_key = build_db_key(repo, file_path);
    assert_eq!(
        db_key,
        r"D:\projects\Python\local-context-engine\context-engine-rs\Cargo.toml"
    );
}

#[test]
fn file_retrieval_db_key_forward_slash_input() {
    let repo = r"D:\projects\Python\local-context-engine";
    let file_path = "context-engine-rs/Cargo.toml";
    let db_key = build_db_key(repo, file_path);
    assert_eq!(
        db_key,
        r"D:\projects\Python\local-context-engine\context-engine-rs\Cargo.toml"
    );
}

#[test]
fn file_retrieval_db_key_mixed_slashes() {
    let repo = r"D:\projects\Python\local-context-engine";
    let file_path = r"src/indexing\pipeline.rs";
    let db_key = build_db_key(repo, file_path);
    assert_eq!(
        db_key,
        r"D:\projects\Python\local-context-engine\src\indexing\pipeline.rs"
    );
}

#[test]
fn file_retrieval_db_key_leading_slash_in_file_path() {
    let repo = r"D:\projects\Python\local-context-engine";
    let file_path = "/context-engine-rs/Cargo.toml";
    let db_key = build_db_key(repo, file_path);
    assert_eq!(
        db_key,
        r"D:\projects\Python\local-context-engine\context-engine-rs\Cargo.toml"
    );
}

#[test]
fn file_retrieval_db_key_leading_backslash_in_file_path() {
    let repo = r"D:\projects\Python\local-context-engine";
    let file_path = r"\context-engine-rs\Cargo.toml";
    let db_key = build_db_key(repo, file_path);
    assert_eq!(
        db_key,
        r"D:\projects\Python\local-context-engine\context-engine-rs\Cargo.toml"
    );
}

#[test]
fn file_retrieval_db_key_trailing_slash_in_workspace() {
    let repo = r"D:\projects\Python\local-context-engine\";
    let file_path = "context-engine-rs/Cargo.toml";
    let db_key = build_db_key(repo, file_path);
    assert_eq!(
        db_key,
        r"D:\projects\Python\local-context-engine\context-engine-rs\Cargo.toml"
    );
}

#[test]
fn file_retrieval_db_key_both_edge_cases() {
    let repo = r"D:\projects\Python\local-context-engine/";
    let file_path = "/context-engine-rs/Cargo.toml";
    let db_key = build_db_key(repo, file_path);
    assert_eq!(
        db_key,
        r"D:\projects\Python\local-context-engine\context-engine-rs\Cargo.toml"
    );
}

#[test]
fn file_retrieval_db_key_unix_paths() {
    let repo = "/home/user/project";
    let file_path = "src/main.rs";
    let db_key = build_db_key(repo, file_path);
    assert!(db_key.contains("src"));
    assert!(db_key.contains("main.rs"));
    assert!(!db_key.contains("//"));
}

#[test]
fn cosine_identical_vectors() {
    let v = vec![1.0, 0.0, 0.0];
    assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_orthogonal_vectors() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    assert!(cosine_similarity(&a, &b).abs() < 1e-6);
}

#[test]
fn cosine_empty_returns_zero() {
    assert_eq!(cosine_similarity(&[], &[]), 0.0);
    assert_eq!(cosine_similarity(&[1.0], &[]), 0.0);
}

#[test]
fn budget_all_fit() {
    let blocks = vec![
        OutputBlock {
            header: "file.rs#L1-10".to_string(),
            content: "1: fn main() {\n2:   println!(\"hi\");\n3: }".to_string(),
            file: "file.rs".to_string(),
            line_start: 1,
            line_end: 10,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
        OutputBlock {
            header: "file.rs#L20-30".to_string(),
            content: "20: fn foo() {\n21:   bar();\n22: }".to_string(),
            file: "file.rs".to_string(),
            line_start: 20,
            line_end: 30,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
    ];
    let out = assemble_with_budget(&blocks);
    assert!(out.contains("1: fn main()"));
    assert!(out.contains("20: fn foo()"));
    assert!(!out.contains("truncated"));
}

#[test]
fn budget_exceeded_shows_header_and_first_line() {
    let big_content = (1..=500)
        .map(|i| format!("{}: // line {}", i, "x".repeat(80)))
        .collect::<Vec<_>>()
        .join("\n");
    let mut blocks = Vec::new();
    for i in 0..200 {
        blocks.push(OutputBlock {
            header: format!("big.rs#L{}-{}", i * 500 + 1, (i + 1) * 500),
            content: big_content.clone(),
            file: "big.rs".to_string(),
            line_start: i * 500 + 1,
            line_end: (i + 1) * 500,
            callers: None,
            caller_files: None,
            ..Default::default()
        });
    }
    let out = assemble_with_budget(&blocks);
    assert!(out.len() <= MAX_TOOL_OUTPUT_CHARS);
    assert!(out.contains("truncated to fit output size limit"));
    assert!(out.contains("elided, use Read"));
}

#[test]
fn budget_first_line_capped_at_120() {
    let long_line = format!("1: {}", "x".repeat(200));
    let blocks = vec![OutputBlock {
        header: "file.rs#L1-5".to_string(),
        content: "1: short line".to_string(),
        file: "file.rs".to_string(),
        line_start: 1,
        line_end: 5,
        callers: None,
        caller_files: None,
        ..Default::default()
    }];
    // This block fits fully, so test the truncation on a block that exceeds budget.
    let big = "y".repeat(MAX_TOOL_OUTPUT_CHARS);
    let blocks2 = vec![
        OutputBlock {
            header: "a.rs#L1-999".to_string(),
            content: big,
            file: "a.rs".to_string(),
            line_start: 1,
            line_end: 999,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
        OutputBlock {
            header: "b.rs#L1-10".to_string(),
            content: long_line,
            file: "b.rs".to_string(),
            line_start: 1,
            line_end: 10,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
    ];
    let out = assemble_with_budget(&blocks2);
    // The second block should be truncated. Its first line is >120 chars.
    // Verify the output contains the ellipsis marker for long line.
    assert!(out.contains("…"));
    // Verify within budget.
    assert!(out.len() <= MAX_TOOL_OUTPUT_CHARS);
    // First block's full output (all 'y's) also gets budget-applied:
    // since it alone exceeds budget, even it gets truncated form.
    assert!(out.contains("elided, use Read"));

    // Test blocks that fit fine.
    let out1 = assemble_with_budget(&blocks);
    assert!(out1.contains("1: short line"));
    assert!(!out1.contains("truncated"));
}

#[test]
fn budget_single_line_chunk_no_elision() {
    // Single-line chunk: line_end == line_start, so no elision marker needed.
    // Put it behind a budget-buster so it gets truncated form.
    let big = "z".repeat(MAX_TOOL_OUTPUT_CHARS);
    let blocks2 = vec![
        OutputBlock {
            header: "huge.rs#L1-999".to_string(),
            content: big,
            file: "huge.rs".to_string(),
            line_start: 1,
            line_end: 999,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
        OutputBlock {
            header: "file.rs#L5-5".to_string(),
            content: "5: let x = 1;".to_string(),
            file: "file.rs".to_string(),
            line_start: 5,
            line_end: 5,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
    ];
    let out = assemble_with_budget(&blocks2);
    // line_end == line_start → no "elided" line for this block
    assert!(out.contains("file.rs#L5-5"));
    assert!(out.contains("5: let x = 1;"));
    // But the "elided" marker should appear for the first (big) block
    assert!(out.contains("elided, use Read"));
}

#[test]
fn merge_blocks_no_overlap() {
    let blocks = vec![
        OutputBlock {
            header: "a.rs#L1-10".into(),
            content: "1: aaa".into(),
            file: "a.rs".into(),
            line_start: 1,
            line_end: 10,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
        OutputBlock {
            header: "a.rs#L20-30".into(),
            content: "20: bbb".into(),
            file: "a.rs".into(),
            line_start: 20,
            line_end: 30,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
    ];
    let merged = merge_overlapping_blocks(blocks);
    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].line_start, 1);
    assert_eq!(merged[1].line_start, 20);
}

#[test]
fn merge_blocks_overlap_same_file() {
    let blocks = vec![
        OutputBlock {
            header: "a.rs#L1-50".into(),
            content: "1: aaa\n2: bbb".into(),
            file: "a.rs".into(),
            line_start: 1,
            line_end: 50,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
        OutputBlock {
            header: "a.rs#L26-75".into(),
            content: "26: ccc\n27: ddd".into(),
            file: "a.rs".into(),
            line_start: 26,
            line_end: 75,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
    ];
    let merged = merge_overlapping_blocks(blocks);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].line_start, 1);
    assert_eq!(merged[0].line_end, 75);
    // Header uses the same file path as inputs + merged range, no caller tag.
    assert_eq!(merged[0].header, "a.rs#L1-75");
}

#[test]
fn merge_blocks_combines_caller_tags() {
    let blocks = vec![
        OutputBlock {
            header: "a.rs#L1-50 [callers:3 files:2]".into(),
            content: "1: aaa".into(),
            file: "a.rs".into(),
            line_start: 1,
            line_end: 50,
            callers: Some(3),
            caller_files: Some(2),
            ..Default::default()
        },
        OutputBlock {
            header: "a.rs#L26-75 [callers:7 files:4]".into(),
            content: "26: bbb".into(),
            file: "a.rs".into(),
            line_start: 26,
            line_end: 75,
            callers: Some(7),
            caller_files: Some(4),
            ..Default::default()
        },
    ];
    let merged = merge_overlapping_blocks(blocks);
    assert_eq!(merged.len(), 1);
    // Caller stats: max(3,7)=7, max(2,4)=4
    assert_eq!(merged[0].callers, Some(7));
    assert_eq!(merged[0].caller_files, Some(4));
    // Header includes the combined caller tag (count-only format for merged blocks).
    assert_eq!(merged[0].header, "a.rs#L1-75 [callers:7]");
}

#[test]
fn merge_blocks_carries_caller_and_callee_names() {
    // Regression: the names MUST travel with the count through a merge.
    // Block A is name-less (the import region), Block B carries the real
    // symbol's counts AND names. Old merge logic bumped only the counts,
    // leaving A's empty names → bare "[callers:N]". Now the higher-count
    // block's names are adopted atomically with its count.
    let blocks = vec![
        OutputBlock {
            file: "a.rs".into(),
            content: "1: use foo;".into(),
            line_start: 1,
            line_end: 50,
            callers: None,
            caller_names: vec![],
            callees: None,
            callee_names: vec![],
            ..Default::default()
        },
        OutputBlock {
            file: "a.rs".into(),
            content: "26: fn x".into(),
            line_start: 26,
            line_end: 75,
            callers: Some(2),
            caller_files: Some(1),
            caller_names: vec!["foo".into(), "bar".into()],
            callees: Some(1),
            callee_names: vec!["baz".into()],
            ..Default::default()
        },
    ];
    let merged = merge_overlapping_blocks(blocks);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].callers, Some(2));
    assert_eq!(
        merged[0].caller_names,
        vec!["foo".to_string(), "bar".to_string()]
    );
    assert_eq!(merged[0].callees, Some(1));
    assert_eq!(merged[0].callee_names, vec!["baz".to_string()]);
    // Header renders the NAMED form, not the bare count fallback.
    // (a.rs doesn't exist on disk, so the multi-block merge falls back to a
    // content union — only the header tags matter for this assertion.)
    assert!(
        merged[0].header.contains("[callers: foo, bar]"),
        "header missing named callers: {}",
        merged[0].header
    );
    assert!(
        merged[0].header.contains("[calls: baz]"),
        "header missing named callees: {}",
        merged[0].header
    );
    assert!(
        !merged[0].header.contains("[callers:2]"),
        "header fell back to bare count: {}",
        merged[0].header
    );
}

#[test]
fn merge_blocks_adjacent() {
    let blocks = vec![
        OutputBlock {
            header: "a.rs#L1-10".into(),
            content: "1: x".into(),
            file: "a.rs".into(),
            line_start: 1,
            line_end: 10,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
        OutputBlock {
            header: "a.rs#L11-20".into(),
            content: "11: y".into(),
            file: "a.rs".into(),
            line_start: 11,
            line_end: 20,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
    ];
    let merged = merge_overlapping_blocks(blocks);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].line_start, 1);
    assert_eq!(merged[0].line_end, 20);
}

#[test]
fn merge_blocks_different_files_no_merge() {
    let blocks = vec![
        OutputBlock {
            header: "a.rs#L1-50".into(),
            content: "1: aaa".into(),
            file: "a.rs".into(),
            line_start: 1,
            line_end: 50,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
        OutputBlock {
            header: "b.rs#L1-50".into(),
            content: "1: bbb".into(),
            file: "b.rs".into(),
            line_start: 1,
            line_end: 50,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
    ];
    let merged = merge_overlapping_blocks(blocks);
    assert_eq!(merged.len(), 2);
}

#[test]
fn merge_blocks_preserves_priority_order() {
    let blocks = vec![
        OutputBlock {
            header: "b.rs#L1-10".into(),
            content: "1: first".into(),
            file: "b.rs".into(),
            line_start: 1,
            line_end: 10,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
        OutputBlock {
            header: "a.rs#L1-50".into(),
            content: "1: second".into(),
            file: "a.rs".into(),
            line_start: 1,
            line_end: 50,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
        OutputBlock {
            header: "a.rs#L26-75".into(),
            content: "26: third".into(),
            file: "a.rs".into(),
            line_start: 26,
            line_end: 75,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
        OutputBlock {
            header: "b.rs#L20-30".into(),
            content: "20: fourth".into(),
            file: "b.rs".into(),
            line_start: 20,
            line_end: 30,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
    ];
    let merged = merge_overlapping_blocks(blocks);
    // b.rs: L1-10 and L20-30 not overlapping → 2 blocks
    // a.rs: L1-50 and L26-75 overlap → 1 merged block
    assert_eq!(merged.len(), 3);
    // b.rs appeared first (index 0), so its blocks come first
    assert_eq!(merged[0].file, "b.rs");
    assert_eq!(merged[0].line_start, 1);
    // a.rs appeared at index 1
    assert_eq!(merged[1].file, "a.rs");
    assert_eq!(merged[1].line_start, 1);
    assert_eq!(merged[1].line_end, 75);
    // b.rs second block at index 3 → comes after a.rs
    assert_eq!(merged[2].file, "b.rs");
    assert_eq!(merged[2].line_start, 20);
}

#[test]
fn merge_blocks_fallback_preserves_content_on_fs_failure() {
    // Use a non-existent file path so read_lines_from_fs will fail,
    // exercising the fallback content-merge path.
    let blocks = vec![
        OutputBlock {
            header: "/nonexistent/z.rs#L1-50".into(),
            content: "1: aaa\n2: bbb\n3: ccc".into(),
            file: "/nonexistent/z.rs".into(),
            line_start: 1,
            line_end: 50,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
        OutputBlock {
            header: "/nonexistent/z.rs#L26-75".into(),
            content: "2: bbb\n26: ddd\n27: eee".into(),
            file: "/nonexistent/z.rs".into(),
            line_start: 26,
            line_end: 75,
            callers: None,
            caller_files: None,
            ..Default::default()
        },
    ];
    let merged = merge_overlapping_blocks(blocks);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].line_start, 1);
    assert_eq!(merged[0].line_end, 75);
    assert!(merged[0].header.contains("L1-75"));
    // Fallback should preserve original lines, deduped by line number.
    assert!(merged[0].content.contains("1: aaa"));
    assert!(merged[0].content.contains("2: bbb"));
    assert!(merged[0].content.contains("3: ccc"));
    assert!(merged[0].content.contains("26: ddd"));
    assert!(merged[0].content.contains("27: eee"));
    // Line "2: bbb" appeared in both blocks but should only appear once.
    assert_eq!(merged[0].content.matches("2: bbb").count(), 1);
}

// ─── select_empty_or_warming_message (warming-vs-empty signal) ───────────

#[test]
fn warming_message_takes_precedence_and_says_retry() {
    // warming=true → retry message, regardless of rerank_rejected.
    for rr in [false, true] {
        let msg = select_empty_or_warming_message(true, rr, "find the parser");
        assert!(
            msg.contains("warming"),
            "warming msg must mention warming: {msg}"
        );
        assert!(
            msg.to_lowercase().contains("retry"),
            "must tell caller to retry: {msg}"
        );
        // MUST NOT use the genuine-empty wording.
        assert!(
            !msg.contains("No results found"),
            "warming must not say 'No results found'"
        );
        assert!(
            !msg.contains("No relevant code found"),
            "warming must not say 'No relevant code found'"
        );
    }
}

#[test]
fn genuine_empty_resident_shard_keeps_existing_wording() {
    // warming=false, not rejected → the unchanged "No results found for: <q>".
    let msg = select_empty_or_warming_message(false, false, "find the parser");
    assert_eq!(msg, "No results found for: find the parser");

    // warming=false, rerank actively rejected → the unchanged rerank-rejected wording.
    let msg = select_empty_or_warming_message(false, true, "find the parser");
    assert!(
        msg.starts_with("No relevant code found."),
        "rerank-rejected wording preserved: {msg}"
    );
    assert!(
        !msg.contains("warming"),
        "genuine empty must not mention warming"
    );
}

// ─── scope parameter (code | docs) ───────────────────────────────────────

fn scope_property<T: schemars::JsonSchema>() -> serde_json::Value {
    let schema = serde_json::to_value(schemars::schema_for!(T)).expect("schema serializes");
    schema["properties"]["scope"].clone()
}

/// Agents read this schema, and some MCP clients reject `$ref` / `anyOf`, so the
/// `scope` property must be a flat string enum carrying the shared description —
/// identical on every surface that exposes `codebase-retrieval`.
#[test]
fn scope_param_schema_is_flat_and_identical_on_every_mcp_surface() {
    let surfaces = [
        (
            "CodebaseRetrievalArgs",
            scope_property::<CodebaseRetrievalArgs>(),
        ),
        (
            "RepoCodebaseRetrievalArgs",
            scope_property::<RepoCodebaseRetrievalArgs>(),
        ),
        (
            "ProxyCodebaseRetrievalArgs",
            scope_property::<crate::router::mcp_proxy::ProxyCodebaseRetrievalArgs>(),
        ),
    ];
    for (name, prop) in surfaces {
        assert!(!prop.is_null(), "{name} must expose a `scope` property");
        assert_eq!(
            prop["description"].as_str().map(str::trim),
            Some(crate::prompts::MCP_SCOPE_PARAM.trim()),
            "{name} scope description must be the shared text"
        );
        let text = prop.to_string();
        for banned in ["$ref", "anyOf", "oneOf", "allOf"] {
            assert!(
                !text.contains(banned),
                "{name} scope schema has {banned}: {text}"
            );
        }
        let values: Vec<&str> = prop["enum"]
            .as_array()
            .unwrap_or_else(|| panic!("{name} scope must be an enum: {text}"))
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert_eq!(values, ["code", "docs"], "{name}: no `all` value");
    }
}

#[test]
fn scope_param_is_optional_on_every_surface() {
    for (name, schema) in [
        (
            "CodebaseRetrievalArgs",
            serde_json::to_value(schemars::schema_for!(CodebaseRetrievalArgs)).unwrap(),
        ),
        (
            "RepoCodebaseRetrievalArgs",
            serde_json::to_value(schemars::schema_for!(RepoCodebaseRetrievalArgs)).unwrap(),
        ),
        (
            "ProxyCodebaseRetrievalArgs",
            serde_json::to_value(schemars::schema_for!(
                crate::router::mcp_proxy::ProxyCodebaseRetrievalArgs
            ))
            .unwrap(),
        ),
    ] {
        let required = schema["required"].as_array().cloned().unwrap_or_default();
        assert!(
            !required.iter().any(|r| r == "scope"),
            "{name}: scope must stay optional (defaults to code)"
        );
    }
}

/// The structured scope is prepended, so it beats a `scope:` typed in the text,
/// and the funnel's default is `code`.
#[test]
fn scoped_request_prepends_structured_scope_and_wins_over_text() {
    use crate::query::filters::parse_query_filters;
    use crate::vector::Scope;

    assert_eq!(
        scoped_request("how does sharding work", Scope::Docs),
        "scope:docs how does sharding work"
    );
    let (clean, filters) = parse_query_filters(&scoped_request("scope:docs how x", Scope::Code));
    assert_eq!(filters.scope, Some(Scope::Code));
    assert_eq!(clean, "how x");
}

#[test]
fn scope_fallback_notice_names_both_scopes() {
    use crate::vector::Scope;
    let note = scope_fallback_notice(Scope::Code, Scope::Docs);
    assert!(note.starts_with("[scope fallback]"));
    assert!(note.contains(r#"requested scope "code""#));
    assert!(note.contains(r#"scope "docs" instead"#));
    assert!(note.ends_with("\n\n"), "must separate from the results");
}

#[test]
fn both_scopes_empty_note_only_for_a_genuine_scoped_empty() {
    let base = "No results found for: q".to_string();
    // Scoped, nothing found anywhere, index ready → say both were searched.
    assert_eq!(
        note_both_scopes_searched(base.clone(), true, true, false),
        "No results found for: q (searched both code and docs)"
    );
    // Warming must keep its retry wording untouched.
    assert_eq!(
        note_both_scopes_searched(base.clone(), true, true, true),
        base
    );
    // Candidates existed (rerank rejected them): not an empty-scope situation.
    assert_eq!(
        note_both_scopes_searched(base.clone(), true, false, false),
        base
    );
    // Unscoped query: nothing to add.
    assert_eq!(
        note_both_scopes_searched(base.clone(), false, true, false),
        base
    );
}

/// The schema a client actually receives from `tools/list` for each
/// `codebase-retrieval` surface — rmcp builds it with its own settings, which can
/// differ from a bare `schemars::schema_for!`, so assert on THAT.
fn served_scope_property(router: &ToolRouter<impl Send + Sync + 'static>) -> serde_json::Value {
    let tool = router
        .list_all()
        .into_iter()
        .find(|t| t.name == "codebase-retrieval")
        .expect("codebase-retrieval is registered");
    serde_json::Value::Object((*tool.input_schema).clone())["properties"]["scope"].clone()
}

pub(crate) fn assert_served_scope_schema(surface: &str, prop: serde_json::Value) {
    assert!(!prop.is_null(), "{surface}: tools/list must expose `scope`");
    assert_eq!(
        prop["description"].as_str().map(str::trim),
        Some(crate::prompts::MCP_SCOPE_PARAM.trim()),
        "{surface}: served description must be the shared text"
    );
    let text = prop.to_string();
    for banned in ["$ref", "anyOf", "oneOf", "allOf"] {
        assert!(
            !text.contains(banned),
            "{surface}: served scope has {banned}: {text}"
        );
    }
    let values: Vec<&str> = prop["enum"]
        .as_array()
        .unwrap_or_else(|| panic!("{surface}: served scope must be an enum: {text}"))
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert_eq!(values, ["code", "docs"], "{surface}: no `all` value");
}

#[test]
fn served_tool_schema_exposes_flat_shared_scope_on_global_and_repo_mcp() {
    assert_served_scope_schema(
        "McpHandler",
        served_scope_property(&McpHandler::tool_router()),
    );
    assert_served_scope_schema(
        "RepoMcpHandler",
        served_scope_property(&RepoMcpHandler::tool_router()),
    );
}

/// `scope` must stay optional in the served schema too: an agent that omits it
/// gets the `code` default rather than a validation error.
fn served_scope_is_required(router: &ToolRouter<impl Send + Sync + 'static>) -> bool {
    let tool = router
        .list_all()
        .into_iter()
        .find(|t| t.name == "codebase-retrieval")
        .expect("registered");
    let schema = serde_json::Value::Object((*tool.input_schema).clone());
    let required = schema["required"].as_array().cloned().unwrap_or_default();
    required.iter().any(|r| r == "scope")
}

#[test]
fn served_tool_schema_keeps_scope_optional() {
    assert!(!served_scope_is_required(&McpHandler::tool_router()));
    assert!(!served_scope_is_required(&RepoMcpHandler::tool_router()));
}

// ─── grep hint footer ─────────────────────────────────────────────────────

fn sym_block(file: &str, line_start: u32, line_end: u32, symbols: &[&str]) -> OutputBlock {
    OutputBlock {
        header: format!("{file}#L{line_start}-{line_end}"),
        content: format!("{line_start}: body"),
        file: file.to_string(),
        line_start,
        line_end,
        symbols: symbols.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

#[test]
fn grep_hint_none_without_symbols() {
    let blocks = vec![sym_block("a.rs", 1, 5, &[]), sym_block("b.rs", 1, 5, &[])];
    assert_eq!(build_grep_hint(&blocks), None);
    assert_eq!(build_grep_hint(&[]), None);
}

#[test]
fn grep_hint_names_both_rg_and_grep_with_same_pattern() {
    let blocks = vec![
        sym_block("a.rs", 1, 5, &["open_gate"]),
        sym_block("b.rs", 1, 5, &["get_or_open"]),
    ];
    let hint = build_grep_hint(&blocks).expect("symbols present");
    assert!(hint.starts_with("\n\n---\n"));
    assert!(hint.contains("rg 'open_gate|get_or_open'"));
    assert!(hint.contains("grep -rE 'open_gate|get_or_open'"));
}

#[test]
fn grep_hint_dedupes_keeps_first_seen_order_and_caps_at_five() {
    let blocks = vec![
        sym_block("a.rs", 1, 5, &["alpha_fn", "beta_fn", "alpha_fn"]),
        sym_block("b.rs", 1, 5, &["gamma_fn", "beta_fn", "delta_fn"]),
        sym_block("c.rs", 1, 5, &["epsilon_fn", "zeta_fn", "eta_fn"]),
    ];
    let hint = build_grep_hint(&blocks).unwrap();
    assert!(hint.contains("rg 'alpha_fn|beta_fn|gamma_fn|delta_fn|epsilon_fn'"));
    assert!(!hint.contains("zeta_fn"));
}

#[test]
fn grep_hint_drops_non_identifier_short_and_overlong_names() {
    let long = "x".repeat(41);
    let blocks = vec![sym_block(
        "a.rs",
        1,
        5,
        &[
            "new",        // < 4 chars: grep noise
            "operator==", // regex/shell metacharacters
            "foo bar",    // whitespace
            "9lives",     // not an identifier
            "",           // empty
            &long,        // > 40 chars
            "keep_me",
        ],
    )];
    let hint = build_grep_hint(&blocks).unwrap();
    assert!(hint.contains("rg 'keep_me'"));
    assert!(!hint.contains("operator"));
    assert!(!hint.contains("new"));
    assert!(!hint.contains(&long));
}

#[test]
fn grep_hint_none_when_every_name_filtered() {
    let blocks = vec![sym_block("a.rs", 1, 5, &["new", "get", "operator+"])];
    assert_eq!(build_grep_hint(&blocks), None);
}

#[test]
fn grep_hint_never_exceeds_declared_max() {
    let names: Vec<String> = (0..5).map(|i| format!("{}{i}", "n".repeat(39))).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let hint = build_grep_hint(&[sym_block("a.rs", 1, 5, &refs)]).unwrap();
    assert!(
        hint.len() <= GREP_HINT_MAX_CHARS,
        "hint is {} chars, max {}",
        hint.len(),
        GREP_HINT_MAX_CHARS
    );
}

#[test]
fn merged_blocks_keep_absorbed_symbols() {
    // Overlapping ranges in one file collapse into one block; the absorbed
    // block's symbol must survive so the hint can still name it.
    let blocks = vec![
        sym_block("zz_missing_merge.rs", 1, 10, &["first_symbol"]),
        sym_block("zz_missing_merge.rs", 8, 20, &["second_symbol"]),
    ];
    let merged = merge_overlapping_blocks(blocks);
    assert_eq!(merged.len(), 1);
    assert!(merged[0].symbols.contains(&"first_symbol".to_string()));
    assert!(merged[0].symbols.contains(&"second_symbol".to_string()));
}

#[test]
fn hint_fits_inside_budget_reserve_even_when_output_is_truncated() {
    let big_content = (1..=500)
        .map(|i| format!("{}: // line {}", i, "x".repeat(80)))
        .collect::<Vec<_>>()
        .join("\n");
    let blocks: Vec<OutputBlock> = (0..200)
        .map(|i| OutputBlock {
            header: format!("big.rs#L{}-{}", i * 500 + 1, (i + 1) * 500),
            content: big_content.clone(),
            file: "big.rs".to_string(),
            line_start: i * 500 + 1,
            line_end: (i + 1) * 500,
            symbols: vec!["big_symbol".to_string()],
            ..Default::default()
        })
        .collect();
    let out = with_grep_hint(assemble_with_budget(&blocks), &blocks);
    assert!(out.contains("truncated to fit output size limit"));
    assert!(
        out.contains("rg 'big_symbol'"),
        "hint must survive truncation"
    );
    assert!(
        out.len() <= MAX_TOOL_OUTPUT_CHARS,
        "output {} exceeds cap {}",
        out.len(),
        MAX_TOOL_OUTPUT_CHARS
    );
    // The truncation footer must come BEFORE the hint, never squeezed out by it.
    assert!(out.find("truncated to fit").unwrap() < out.find("rg 'big_symbol'").unwrap());
}

#[test]
fn with_grep_hint_leaves_output_untouched_when_no_symbols() {
    let blocks = vec![sym_block("a.rs", 1, 5, &[])];
    assert_eq!(with_grep_hint("body".to_string(), &blocks), "body");
}

// ─── grep hint: symbol eligibility (test files, kinds, callees) ──────────

#[test]
fn repo_relative_strips_repo_prefix_across_separators_and_case() {
    assert_eq!(
        repo_relative(
            r"d:\projects\python\app\src\main.rs",
            r"D:\Projects\Python\App"
        ),
        "src/main.rs"
    );
    assert_eq!(
        repo_relative("/home/u/app/src/a.py", "/home/u/app/"),
        "src/a.py"
    );
}

#[test]
fn repo_relative_never_leaks_outer_directories() {
    // Not under the repo: keep only the file name, so a `test/` directory ABOVE
    // the repo can never make production code look like tests.
    assert_eq!(repo_relative("/elsewhere/test/a.py", "/home/u/app"), "a.py");
    // A sibling with a shared prefix is not "inside" the repo.
    assert_eq!(repo_relative("/home/u/app2/x.rs", "/home/u/app"), "x.rs");
}

#[test]
fn hint_symbol_keeps_real_symbols_and_drops_the_rest() {
    let repo = "/work/app";
    let keep =
        |file: &str, sym: Option<&str>, kind: Option<&str>| hint_symbol(file, repo, sym, kind);
    assert_eq!(
        keep("/work/app/src/a.rs", Some("open_db"), Some("function")),
        Some("open_db".into())
    );
    assert_eq!(
        keep("/work/app/src/a.rs", Some("open_db"), None),
        Some("open_db".into())
    );
    // No symbol on the chunk.
    assert_eq!(keep("/work/app/src/a.rs", None, Some("function")), None);
    // Module-like kinds are not grep targets.
    assert_eq!(
        keep(
            "/work/app/src/a.rs",
            Some("isolation_repro"),
            Some("module")
        ),
        None
    );
    // `impl_Type` is synthetic, but the TYPE it names is a good target: strip the
    // prefix (and any generics) rather than losing the symbol.
    assert_eq!(
        keep(
            "/work/app/src/a.rs",
            Some("impl_IndexPipeline"),
            Some("impl")
        ),
        Some("IndexPipeline".into())
    );
    assert_eq!(
        keep("/work/app/src/a.rs", Some("impl_Cache<K, V>"), Some("impl")),
        Some("Cache".into())
    );
    // Test files, any language.
    assert_eq!(
        keep(
            "/work/app/src/store/tests.rs",
            Some("some_helper"),
            Some("function")
        ),
        None
    );
    assert_eq!(
        keep(
            "/work/app/pkg/handler_test.go",
            Some("TestHandler"),
            Some("function")
        ),
        None
    );
    assert_eq!(
        keep(
            "/work/app/src/utils.test.ts",
            Some("helperFn"),
            Some("function")
        ),
        None
    );
    // Generated files.
    assert_eq!(
        keep(
            "/work/app/src/api/types.generated.ts",
            Some("GeneratedType"),
            Some("class")
        ),
        None
    );
}

#[test]
fn hint_symbol_ignores_test_dirs_above_the_repo_root() {
    // The repo itself lives under a `test/` dir; its production files still count.
    assert_eq!(
        hint_symbol(
            "/ci/test/app/src/a.rs",
            "/ci/test/app",
            Some("open_db"),
            None
        ),
        Some("open_db".into())
    );
    assert_eq!(
        hint_symbol(
            "/ci/test/app/tests/a.rs",
            "/ci/test/app",
            Some("open_db"),
            None
        ),
        None
    );
}

fn callee_block(symbols: &[&str], callees: &[&str]) -> OutputBlock {
    OutputBlock {
        symbols: symbols.iter().map(|s| s.to_string()).collect(),
        callee_names: callees.iter().map(|s| s.to_string()).collect(),
        ..sym_block("a.rs", 1, 5, &[])
    }
}

#[test]
fn grep_hint_lists_every_blocks_own_symbols_before_any_callees() {
    // Measured across 5 repos: symbols DEFINED in the results are on-topic, while
    // callees often drag in logging/util helpers (`maskUserID`, `CreatePendingLog`).
    // So all own symbols come first (block order), callees only fill what is left.
    let blocks = vec![
        callee_block(&["alpha_fn"], &["beta_fn", "gamma_fn"]),
        callee_block(&["delta_fn"], &[]),
    ];
    let hint = build_grep_hint(&blocks).unwrap();
    assert!(
        hint.contains("rg 'alpha_fn|delta_fn|beta_fn|gamma_fn'"),
        "{hint}"
    );
}

fn defined(name: &str, kind: &str, line_start: u32, line_end: u32) -> DefinedSymbol {
    DefinedSymbol {
        name: name.to_string(),
        kind: Some(kind.to_string()),
        line_start,
        line_end,
    }
}

fn picked(rows: &[DefinedSymbol], start: u32, end: u32) -> Vec<String> {
    pick_block_symbols(rows, start, end)
        .into_iter()
        .map(|s| s.name.clone())
        .collect()
}

#[test]
fn pick_block_symbols_prefers_symbols_defined_inside_the_block_in_line_order() {
    let rows = vec![
        defined("impl_Store", "impl", 1, 100),
        defined("second_fn", "function", 30, 40),
        defined("first_fn", "function", 5, 20),
        defined("after_block", "function", 50, 60),
    ];
    // Block 5-45: `first_fn` starts exactly at the block start and counts as inside;
    // the enclosing impl is NOT added when something is defined inside.
    assert_eq!(picked(&rows, 5, 45), vec!["first_fn", "second_fn"]);
}

#[test]
fn pick_block_symbols_falls_back_to_the_innermost_enclosing_symbol() {
    // A slice from the middle of `get_or_open` (real case: L1394-1403).
    let rows = vec![
        defined("store_mod", "module", 1, 3000),
        defined("impl_Store", "impl", 1, 2000),
        defined("get_or_open", "function", 1380, 1420),
    ];
    assert_eq!(picked(&rows, 1394, 1403), vec!["get_or_open"]);
}

#[test]
fn pick_block_symbols_ignores_rows_that_do_not_overlap_and_empty_input() {
    let rows = vec![defined("far_away", "function", 500, 600)];
    assert!(picked(&rows, 1, 50).is_empty());
    assert!(picked(&[], 1, 50).is_empty());
}

#[test]
fn inline_test_markers_are_detected_in_numbered_content() {
    assert!(has_inline_test_marker(
        "1626:     #[tokio::test]\n1627: async fn x() {}"
    ));
    assert!(has_inline_test_marker("10: #[test]\n11: fn t() {}"));
    assert!(has_inline_test_marker("1: #[cfg(test)]\n2: mod tests {"));
    assert!(!has_inline_test_marker("1: fn main() {}\n2: let test = 1;"));
}

#[tokio::test]
async fn defined_symbols_for_reads_overlapping_rows_from_the_repo_db() {
    let home = tempfile::TempDir::new().unwrap();
    let repo = "/test/hint_defined";
    let db = crate::store::open_db(home.path(), repo, 0).await.unwrap();
    let file = "/test/hint_defined/src/a.rs";
    for (fqn, name, kind, ls, le) in [
        (
            "src/a.rs::open_gate",
            "open_gate",
            "function",
            10_i64,
            20_i64,
        ),
        ("src/a.rs::later_fn", "later_fn", "function", 200, 220),
        ("src/b.rs::other_file", "other_file", "function", 10, 20),
    ] {
        let f = if name == "other_file" {
            "/test/hint_defined/src/b.rs"
        } else {
            file
        };
        let thing =
            surrealdb::sql::Thing::from(("symbol", surrealdb::sql::Id::String(fqn.to_string())));
        db.query(
            "CREATE $t SET name = $n, kind = $k, file = $f, \
             line_start = $ls, line_end = $le, signature = NONE, parent = NONE",
        )
        .bind(("t", thing))
        .bind(("n", name.to_string()))
        .bind(("k", kind.to_string()))
        .bind(("f", f.to_string()))
        .bind(("ls", ls))
        .bind(("le", le))
        .await
        .expect("insert symbol");
    }
    let mut db_map = HashMap::new();
    db_map.insert(repo.to_string(), db);

    let rows = defined_symbols_for(&db_map, file, 1, 50).await;
    let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, vec!["open_gate"]);
    assert_eq!(rows[0].kind.as_deref(), Some("function"));
    assert_eq!((rows[0].line_start, rows[0].line_end), (10, 20));

    // A file outside every known repo yields nothing rather than erroring.
    assert!(
        defined_symbols_for(&db_map, "/elsewhere/x.rs", 1, 50)
            .await
            .is_empty()
    );
}

#[test]
fn grep_hint_works_from_callees_alone_when_chunks_carry_no_symbol() {
    // Real data: most chunks have no own symbol, but the call graph still names
    // the production code they touch.
    let blocks = vec![callee_block(&[], &["normalize_repo_path", "open_gate"])];
    let hint = build_grep_hint(&blocks).unwrap();
    assert!(
        hint.contains("rg 'normalize_repo_path|open_gate'"),
        "{hint}"
    );
}

#[test]
fn grep_hint_filters_callee_names_like_symbols_and_dedupes_across_both() {
    let blocks = vec![callee_block(
        &["open_db"],
        &["get", "open_db", "operator==", "db_path"],
    )];
    let hint = build_grep_hint(&blocks).unwrap();
    assert!(hint.contains("rg 'open_db|db_path'"), "{hint}");
}

/// Real result shapes captured from `/api/query` on this repo (Windows-indexed, so
/// paths are lowercased): `(file, symbol, kind, callees)` in rank order. The hint
/// must name production code and never test modules or synthetic `impl_*` names.
fn hint_from_captured(rows: &[(&str, &str, &str, &[&str])]) -> String {
    let repo = r"D:\projects\Python\local-context-engine";
    let root = r"d:\projects\python\local-context-engine\context-engine-rs\src";
    let blocks: Vec<OutputBlock> = rows
        .iter()
        .map(|(file, symbol, kind, callees)| OutputBlock {
            symbols: hint_symbol(
                &format!(r"{root}\{file}"),
                repo,
                Some(symbol).filter(|s| !s.is_empty()).copied(),
                Some(kind).filter(|k| !k.is_empty()).copied(),
            )
            .into_iter()
            .collect(),
            callee_names: callees.iter().map(|c| c.to_string()).collect(),
            ..Default::default()
        })
        .collect();
    build_grep_hint(&blocks).expect("captured data yields a hint")
}

#[test]
fn captured_lock_query_hint_names_open_gate_and_no_test_modules() {
    let hint = hint_from_captured(&[
        (r"store\mod.rs", "", "", &[]),
        (r"store\mod.rs", "", "", &[]),
        (
            r"store\mod.rs",
            "remove_index_dir",
            "function",
            &["normalize_repo_path", "open_gate", "remove_dir_with_retry"],
        ),
        (
            r"store\mod.rs",
            "open_db",
            "function",
            &["db_path", "ensure_secondary_indexes", "sanitize_repo_name"],
        ),
        (r"router\worker.rs", "", "", &[]),
        (r"store\mod.rs", "isolation_repro", "module", &[]),
        (r"store\mod.rs", "open_concurrency", "module", &[]),
    ]);
    assert!(hint.contains("open_gate"), "{hint}");
    assert!(!hint.contains("isolation_repro"), "{hint}");
    assert!(!hint.contains("open_concurrency"), "{hint}");
}

#[test]
fn captured_staleness_query_hint_names_readiness_entry_points() {
    let hint = hint_from_captured(&[
        (
            r"mcp\readiness.rs",
            "await_index_ready",
            "function",
            &["evaluate", "is_resolve_edges_status"],
        ),
        (r"mcp\readiness.rs", "", "", &[]),
    ]);
    assert!(hint.contains("await_index_ready"), "{hint}");
    assert!(hint.contains("evaluate"), "{hint}");
}

#[test]
fn captured_edge_resolution_hint_skips_synthetic_impl_names() {
    let hint = hint_from_captured(&[
        (r"indexing\pipeline.rs", "", "", &[]),
        (
            r"indexing\pipeline.rs",
            "resolve_edges_from_ram",
            "function",
            &["flush_edge_batch", "load_all_symbols"],
        ),
        (r"indexing\pipeline.rs", "impl_IndexPipeline", "impl", &[]),
    ]);
    assert!(hint.contains("resolve_edges_from_ram"), "{hint}");
    assert!(!hint.contains("impl_IndexPipeline"), "{hint}");
    assert!(hint.contains("IndexPipeline"), "{hint}");
}

#[test]
fn captured_impl_only_chunks_still_yield_the_type_names() {
    // Real spec-ade data (scheduled tasks query): every chunk was an `impl` block
    // with no callees. Without stripping `impl_`, the hint vanished entirely.
    let hint = hint_from_captured(&[
        (
            r"spec-ade-api\src\scheduled_task\runner.rs",
            "impl_ScheduledTaskRunner",
            "impl",
            &[],
        ),
        (r"spec-ade-api\src\routes\scheduled_tasks.rs", "", "", &[]),
        (
            r"spec-ade-api\src\scheduled_task\registry.rs",
            "impl_ScheduledTaskRegistry",
            "impl",
            &[],
        ),
    ]);
    assert!(
        hint.contains("rg 'ScheduledTaskRunner|ScheduledTaskRegistry'"),
        "{hint}"
    );
}

fn code_result(
    file: &str,
    line_start: u32,
    line_end: u32,
    symbol: Option<&str>,
    kind: Option<&str>,
    content: &str,
) -> crate::query::engine::CodeResult {
    crate::query::engine::CodeResult {
        file: file.to_string(),
        line_start,
        line_end,
        score: 0.5,
        content: content.to_string(),
        symbol: symbol.map(str::to_string),
        symbol_kind: kind.map(str::to_string),
        callers: None,
        caller_files: None,
        caller_names: vec![],
        callee_names: vec![],
        callees: None,
    }
}

async fn db_with_symbols(
    home: &std::path::Path,
    repo: &str,
    rows: &[(&str, &str, &str, i64, i64)],
) -> HashMap<String, Surreal<Db>> {
    let db = crate::store::open_db(home, repo, 0).await.unwrap();
    for (file, name, kind, ls, le) in rows {
        let fqn = format!("{file}::{name}");
        let thing = surrealdb::sql::Thing::from(("symbol", surrealdb::sql::Id::String(fqn)));
        db.query(
            "CREATE $t SET name = $n, kind = $k, file = $f, \
             line_start = $ls, line_end = $le, signature = NONE, parent = NONE",
        )
        .bind(("t", thing))
        .bind(("n", name.to_string()))
        .bind(("k", kind.to_string()))
        .bind(("f", file.to_string()))
        .bind(("ls", *ls))
        .bind(("le", *le))
        .await
        .expect("insert symbol");
    }
    let mut map = HashMap::new();
    map.insert(repo.to_string(), db);
    map
}

#[tokio::test]
async fn block_hint_symbols_uses_own_symbol_or_falls_back_to_defined_ones() {
    let home = tempfile::TempDir::new().unwrap();
    let repo = "/test/hint_blocks";
    let lib = "/test/hint_blocks/src/lib.rs";
    let spec = "/test/hint_blocks/src/a.spec.ts";
    let map = db_with_symbols(
        home.path(),
        repo,
        &[
            (lib, "open_gate", "function", 10, 20),
            (lib, "get_or_open", "function", 100, 160),
            (lib, "tests", "module", 300, 400),
            (lib, "lock_race_test", "function", 310, 330),
            (spec, "describeHelper", "function", 1, 30),
        ],
    )
    .await;

    // Own symbol wins; no DB lookup needed.
    let r = code_result(
        lib,
        1,
        50,
        Some("remove_index_dir"),
        Some("function"),
        "1: fn x() {}",
    );
    assert_eq!(
        block_hint_symbols(&map, repo, &r).await,
        vec!["remove_index_dir"]
    );

    // No own symbol: the function defined inside the block names it.
    let r = code_result(lib, 1, 50, None, None, "10: fn open_gate() {}");
    assert_eq!(block_hint_symbols(&map, repo, &r).await, vec!["open_gate"]);

    // A slice from the middle of a long function is named by that function.
    let r = code_result(lib, 120, 130, None, None, "120: let x = 1;");
    assert_eq!(
        block_hint_symbols(&map, repo, &r).await,
        vec!["get_or_open"]
    );

    // Inline Rust test code: its functions are not hint material.
    let r = code_result(
        lib,
        305,
        335,
        None,
        None,
        "309: #[test]\n310: fn lock_race_test() {}",
    );
    assert!(block_hint_symbols(&map, repo, &r).await.is_empty());

    // Test files stay excluded on the fallback path too.
    let r = code_result(spec, 1, 30, None, None, "1: function describeHelper() {}");
    assert!(block_hint_symbols(&map, repo, &r).await.is_empty());

    // An own symbol that is filtered (module) does NOT trigger the fallback:
    // the block IS that module, and in Rust that is almost always a test module.
    let r = code_result(
        lib,
        300,
        400,
        Some("tests"),
        Some("module"),
        "300: mod tests {",
    );
    assert!(block_hint_symbols(&map, repo, &r).await.is_empty());
}

#[tokio::test]
async fn block_hint_symbols_caps_defined_symbols_per_block() {
    let home = tempfile::TempDir::new().unwrap();
    let repo = "/test/hint_cap";
    let file = "/test/hint_cap/src/many.go";
    let map = db_with_symbols(
        home.path(),
        repo,
        &[
            (file, "FirstFunc", "function", 1, 5),
            (file, "SecondFunc", "function", 6, 10),
            (file, "ThirdFunc", "function", 11, 15),
            (file, "FourthFunc", "function", 16, 20),
        ],
    )
    .await;
    let r = code_result(file, 1, 20, None, None, "1: func FirstFunc() {}");
    assert_eq!(
        block_hint_symbols(&map, repo, &r).await,
        vec!["FirstFunc", "SecondFunc", "ThirdFunc"]
    );
}

#[test]
fn test_marker_is_found_just_above_a_definition() {
    let src = "mod a {\n\
               #[cfg(test)]\n\
               mod isolation_repro {\n\
               \x20   /// doc\n\
               \x20   #[tokio::test]\n\
               \x20   async fn exclusive_lock_works() {\n\
               \x20       let x = 1;\n\
               \x20   }\n\
               }\n\
               fn production() {}\n";
    // `mod isolation_repro` (line 3) has `#[cfg(test)]` right above it.
    assert!(is_test_marked(src, 3));
    // The test fn (line 6) has `#[tokio::test]` right above it.
    assert!(is_test_marked(src, 6));
    // A production fn (line 10) and line 1 (nothing above) are not marked.
    assert!(!is_test_marked(src, 10));
    assert!(!is_test_marked(src, 1));
    // Out-of-range lines are simply not marked.
    assert!(!is_test_marked(src, 999));
}

#[tokio::test]
async fn block_hint_symbols_skips_slices_of_inline_rust_tests() {
    // Real leak: a reranker slice from inside an inline test fn's body carries no
    // test attribute in its own text, so only the source ABOVE the enclosing
    // definitions can tell it is test code.
    let home = tempfile::TempDir::new().unwrap();
    let repo_dir = home.path().join("repo");
    std::fs::create_dir_all(repo_dir.join("src")).unwrap();
    let file_path = repo_dir.join("src").join("store.rs");
    let source = "pub fn open_gate() {}\n\
                  \n\
                  #[cfg(test)]\n\
                  mod isolation_repro {\n\
                  \x20   #[tokio::test]\n\
                  \x20   async fn exclusive_lock_works() {\n\
                  \x20       let a = 1;\n\
                  \x20       let b = 2;\n\
                  \x20   }\n\
                  }\n";
    std::fs::write(&file_path, source).unwrap();
    let repo = repo_dir.to_string_lossy().to_string();
    let file = file_path.to_string_lossy().to_string();
    let map = db_with_symbols(
        home.path(),
        &repo,
        &[
            (file.as_str(), "open_gate", "function", 1, 1),
            (file.as_str(), "isolation_repro", "module", 4, 10),
            (file.as_str(), "exclusive_lock_works", "function", 6, 9),
        ],
    )
    .await;

    let slice = code_result(
        &file,
        7,
        8,
        None,
        None,
        "7:         let a = 1;\n8:         let b = 2;",
    );
    assert!(block_hint_symbols(&map, &repo, &slice).await.is_empty());

    let prod = code_result(&file, 1, 1, None, None, "1: pub fn open_gate() {}");
    assert_eq!(
        block_hint_symbols(&map, &repo, &prod).await,
        vec!["open_gate"]
    );
}

#[tokio::test]
async fn inline_test_guards_each_catch_their_own_case() {
    let home = tempfile::TempDir::new().unwrap();
    let repo_dir = home.path().join("repo2");
    std::fs::create_dir_all(repo_dir.join("src")).unwrap();
    let file_path = repo_dir.join("src").join("lib.rs");
    // The helper sits more than 3 lines below `#[cfg(test)]`, so the per-candidate
    // window cannot see the attribute; only the enclosing-scope guard can.
    let source = "#[cfg(test)]\n\
                  mod helpers_mod {\n\
                  \x20   use super::*;\n\
                  \x20   // setup\n\
                  \x20   // more setup\n\
                  \x20   fn build_fixture_repo() {}\n\
                  }\n\
                  #[test]\n\
                  fn top_level_check() {}\n";
    std::fs::write(&file_path, source).unwrap();
    let repo = repo_dir.to_string_lossy().to_string();
    let file = file_path.to_string_lossy().to_string();
    let map = db_with_symbols(
        home.path(),
        &repo,
        &[
            (file.as_str(), "helpers_mod", "module", 2, 7),
            (file.as_str(), "build_fixture_repo", "function", 6, 6),
            (file.as_str(), "top_level_check", "function", 9, 9),
        ],
    )
    .await;
    // Unattributed helper inside a `#[cfg(test)]` module: only the enclosing-scope
    // guard can see it.
    let helper = code_result(&file, 6, 6, None, None, "6:     fn build_fixture_repo() {}");
    assert!(block_hint_symbols(&map, &repo, &helper).await.is_empty());
    // Top-level `#[test]` fn with no enclosing module: only the per-candidate
    // guard can see it.
    let top = code_result(&file, 9, 9, None, None, "9: fn top_level_check() {}");
    assert!(block_hint_symbols(&map, &repo, &top).await.is_empty());
}

#[test]
fn grep_hint_render_normalizes_crlf_templates() {
    // `core.autocrlf=true` checks the .txt template out with CRLF on Windows
    // (`i/lf w/crlf`), and `include_str!` embeds those bytes. The footer must not
    // carry `\r` into the tool output either way.
    let crlf = "---\r\nResults above are top-k.\r\nrg '{{pattern}}'\r\n";
    let lf = "---\nResults above are top-k.\nrg '{{pattern}}'\n";
    let expected = "\n\n---\nResults above are top-k.\nrg 'a|b'";
    assert_eq!(render_grep_hint(crlf, "a|b"), expected);
    assert_eq!(render_grep_hint(lf, "a|b"), expected);
}
