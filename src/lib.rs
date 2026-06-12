/// Public library interface — exposes internal modules for integration tests.
pub mod cli;
pub mod config;
pub mod defender;
pub mod embedding;
pub mod indexing;
pub mod llm;
pub mod mcp;
pub mod parsing;
pub mod query;
pub mod server;
pub mod store;
pub mod vector;

/// Check whether `file` belongs to the repository rooted at `repo`.
///
/// Returns `true` iff `file` is exactly `repo` (after stripping one optional
/// trailing separator from `repo`), or `file` starts with `repo` and the very
/// next character is a path separator (`/` or `\`).
///
/// This avoids the classic prefix-collision bug where repo `/foo` falsely
/// matches file `/foobar/x.rs`.
pub(crate) fn path_in_repo(file: &str, repo: &str) -> bool {
    // Strip at most one trailing separator from repo.
    let r = repo
        .strip_suffix('/')
        .or_else(|| repo.strip_suffix('\\'))
        .unwrap_or(repo);

    if file == r {
        return true;
    }

    if let Some(rest) = file.strip_prefix(r) {
        let next_char = rest.chars().next();
        matches!(next_char, Some('/') | Some('\\'))
    } else {
        false
    }
}

#[cfg(test)]
mod path_in_repo_tests {
    use super::path_in_repo;

    #[test]
    fn rejects_prefix_collision() {
        // repo "D:\proj\foo", file "D:\proj\foobar\x.rs" -> false
        assert!(!path_in_repo(r"D:\proj\foobar\x.rs", r"D:\proj\foo"));
    }

    #[test]
    fn accepts_child_with_backslash() {
        // repo "D:\proj\foo", file "D:\proj\foo\x.rs" -> true
        assert!(path_in_repo(r"D:\proj\foo\x.rs", r"D:\proj\foo"));
    }

    #[test]
    fn accepts_child_after_trailing_sep() {
        // repo "D:\proj\foo\" (trailing sep), file "D:\proj\foo\x.rs" -> true
        assert!(path_in_repo(r"D:\proj\foo\x.rs", r"D:\proj\foo\"));
    }

    #[test]
    fn accepts_exact_match() {
        // repo "D:\proj\foo", file "D:\proj\foo" -> true
        assert!(path_in_repo(r"D:\proj\foo", r"D:\proj\foo"));
    }

    #[test]
    fn forward_slash_paths() {
        assert!(path_in_repo("/proj/foo/bar.rs", "/proj/foo"));
        assert!(!path_in_repo("/proj/foobar/bar.rs", "/proj/foo"));
        assert!(path_in_repo("/proj/foo/bar.rs", "/proj/foo/"));
    }

    #[test]
    fn mixed_separators() {
        assert!(path_in_repo(r"D:\proj\foo/bar.rs", r"D:\proj\foo"));
        assert!(path_in_repo("D:/proj/foo\\bar.rs", "D:/proj/foo"));
    }

    #[test]
    fn no_match_different_root() {
        assert!(!path_in_repo("/other/path/file.rs", "/proj/foo"));
    }
}
