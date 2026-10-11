use std::collections::HashSet;
use std::path::{Component, Path};

use ignore::gitignore::GitignoreBuilder;
use ignore::{Match, WalkBuilder};
use tracing::debug;

/// Extensions considered indexable code/config.
pub const CODE_EXTENSIONS: &[&str] = &[
    "py",
    "js",
    "ts",
    "tsx",
    "jsx",
    "rs",
    "go",
    "java",
    "cs",
    "cpp",
    "c",
    "h",
    "hpp",
    "cc",
    "cxx",
    "hxx",
    "hh",
    "rb",
    "php",
    "swift",
    "kt",
    "kts",
    "scala",
    "ex",
    "exs",
    "clj",
    "hs",
    "ml",
    "lua",
    "luau",
    "r",
    "sh",
    "bash",
    "zsh",
    "fish",
    "ps1",
    "yaml",
    "yml",
    "toml",
    "json",
    "xml",
    "html",
    "css",
    "scss",
    "sass",
    "less",
    "sql",
    "proto",
    "graphql",
    "md",
    "txt",
    "dockerfile",
    "tf",
    "hcl",
    "vue",
    "svelte",
    "astro",
    "mdx",
    "prisma",
    "dart",
    "zig",
    "nim",
    "sol",
    "elm",
    "jl",
    "erl",
    "hrl",
    "nix",
    "m",
    "mm",
    "groovy",
    "gradle",
    "pl",
    "pm",
    "rst",
    "wgsl",
    "glsl",
    "hlsl",
    "pas",
    "pp",
    "dpr",
    "lpr",
    "dpk",
    "liquid",
];

/// Non-dot directories to always skip (dot-prefixed directories are pruned by the
/// walk filter automatically, so they do not need to appear here).
pub const SKIP_DIRS: &[&str] = &[
    "node_modules",
    "target",
    "build",
    "dist",
    "__pycache__",
    "vendor",
];

/// Returns true if `path` is inside a dot-prefixed directory relative to `repo_root`.
/// The check looks ONLY at directory components between `repo_root` and the file's
/// parent — the file's own basename does not count, so a root-level file like
/// `.eslintrc.json` returns false. The repo root itself is never considered hidden.
/// Returns false if `path` is not inside `repo_root` (caller's responsibility).
pub fn is_under_hidden_dir(repo_root: &Path, path: &Path) -> bool {
    let relative = match path.strip_prefix(repo_root) {
        Ok(r) => r,
        Err(_) => return false,
    };
    // Iterate all components except the last one (the filename).
    let components: Vec<_> = relative.components().collect();
    let dir_components = if components.is_empty() {
        &[][..]
    } else {
        &components[..components.len() - 1]
    };
    for component in dir_components {
        if let Component::Normal(name) = component
            && name.to_str().map(|s| s.starts_with('.')).unwrap_or(false)
        {
            return true;
        }
    }
    false
}

/// Returns true if `path` has an extension (or extension-less filename, e.g.
/// `Dockerfile`/`Makefile`/`justfile`) considered indexable code/config.
///
/// This is the single source of truth for the file-type allowlist, shared by
/// the full-rebuild `walk_repo` and the watcher-driven incremental change filter
/// so both paths index exactly the same set of file types.
///
/// `extra_extensions` supplies user-configured extensions beyond the built-in
/// `CODE_EXTENSIONS` list (from `Settings.custom_extensions`).
pub fn has_indexable_extension(path: &Path) -> bool {
    has_indexable_extension_with(path, &[])
}

pub fn has_indexable_extension_with(path: &Path, extra_extensions: &[String]) -> bool {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        let lower = ext.to_lowercase();
        if CODE_EXTENSIONS.contains(&lower.as_str()) {
            return true;
        }
        return extra_extensions.iter().any(|e| e == &lower);
    }
    // No extension — allow a small set of well-known extension-less files.
    let fname = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();
    matches!(fname.as_str(), "dockerfile" | "makefile" | "justfile")
}

/// Returns true if any directory component of `path` relative to `repo_root` is
/// in `SKIP_DIRS` (e.g. `target`, `node_modules`, `build`). The file's own
/// basename is not treated as a directory. Mirrors the `filter_entry` directory
/// pruning that `walk_repo` performs, so the watcher path skips the same trees.
pub fn is_under_skip_dir(repo_root: &Path, path: &Path) -> bool {
    let relative = match path.strip_prefix(repo_root) {
        Ok(r) => r,
        Err(_) => return false,
    };
    let components: Vec<_> = relative.components().collect();
    let dir_components = if components.is_empty() {
        &[][..]
    } else {
        &components[..components.len() - 1]
    };
    for component in dir_components {
        if let Component::Normal(name) = component
            && let Some(s) = name.to_str()
            && SKIP_DIRS.contains(&s)
        {
            return true;
        }
    }
    false
}

/// Reusable filter that applies the full `walk_repo` rule set to many paths
/// while building the gitignore matcher exactly once.
///
/// The full-rebuild `walk_repo` gets dot-dir, SKIP_DIRS, and gitignore handling
/// for free from the `ignore` crate's `WalkBuilder`. The watcher, by contrast,
/// feeds raw filesystem paths that never went through that walk — so without
/// this filter, build artifacts under a gitignored dir (e.g. `target/*.exe`)
/// leak into the index on every incremental run. This re-applies the same rules
/// so the watcher and full-rebuild paths agree on which files exist.
///
/// The watcher debounces filesystem events into a batch that can contain
/// hundreds of paths (e.g. a `cargo build` rewriting `target/`). Rebuilding the
/// gitignore matcher per path would re-read `.gitignore` from disk O(batch)
/// times; this struct reads it once and reuses the compiled matcher, keeping the
/// batch filter O(batch) with a single disk read.
pub struct ChangeFilter {
    repo_root: std::path::PathBuf,
    gitignore: ignore::gitignore::Gitignore,
    extra_extensions: Vec<String>,
    ignore_filenames: HashSet<String>,
    /// Per-repo ignored relative paths (forward-slash-normalized).
    ignore_paths: HashSet<String>,
}

impl ChangeFilter {
    /// Build the filter for `repo_root`, compiling its `.gitignore` / `.ignore`
    /// rules once. A missing or malformed ignore file degrades to a matcher that
    /// excludes nothing (the other predicates still apply).
    pub fn new(repo_root: &Path) -> Self {
        Self::new_with_extensions(repo_root, vec![])
    }

    pub fn new_with_extensions(repo_root: &Path, extra_extensions: Vec<String>) -> Self {
        Self::new_full(repo_root, extra_extensions, HashSet::new())
    }

    pub fn new_full(
        repo_root: &Path,
        extra_extensions: Vec<String>,
        ignore_filenames: HashSet<String>,
    ) -> Self {
        Self::new_complete(
            repo_root,
            extra_extensions,
            ignore_filenames,
            HashSet::new(),
        )
    }

    pub fn new_complete(
        repo_root: &Path,
        extra_extensions: Vec<String>,
        ignore_filenames: HashSet<String>,
        ignore_paths: HashSet<String>,
    ) -> Self {
        let mut builder = GitignoreBuilder::new(repo_root);
        let _ = builder.add(repo_root.join(".gitignore"));
        let _ = builder.add(repo_root.join(".ignore"));
        // `.augmentignore` added LAST so it ranks above `.gitignore`/`.ignore`:
        // `GitignoreBuilder` applies patterns in add-order and a later match wins,
        // so its patterns (including `!pattern` negation) override earlier files,
        // letting a repo re-include something `.gitignore` excluded. ROOT-LEVEL
        // only here — matching how `.gitignore`/`.ignore` are handled on this
        // watcher/incremental path (the full-rebuild walker gets per-directory
        // nesting via `add_custom_ignore_filename`). The extension allowlist,
        // dot-dir, and SKIP_DIRS predicates still run first in `allows`, so `!`
        // cannot resurrect files those layers already excluded.
        let _ = builder.add(repo_root.join(".augmentignore"));
        let gitignore = builder.build().unwrap_or_else(|e| {
            debug!(error = %e, "failed to build gitignore matcher; nothing will be gitignore-excluded");
            ignore::gitignore::Gitignore::empty()
        });
        Self {
            repo_root: repo_root.to_path_buf(),
            gitignore,
            extra_extensions,
            ignore_filenames,
            ignore_paths,
        }
    }

    /// Returns true if `path` passes every indexability rule (extension, dot-dir,
    /// SKIP_DIRS, gitignore, ignore filenames, per-repo ignore paths) and should
    /// therefore be indexed.
    pub fn allows(&self, path: &Path) -> bool {
        if !has_indexable_extension_with(path, &self.extra_extensions)
            || is_under_hidden_dir(&self.repo_root, path)
            || is_under_skip_dir(&self.repo_root, path)
        {
            return false;
        }
        if !self.ignore_filenames.is_empty()
            && let Some(fname) = path.file_name().and_then(|n| n.to_str())
            && self.ignore_filenames.contains(fname)
        {
            return false;
        }
        if !self.ignore_paths.is_empty()
            && let Ok(rel) = path.strip_prefix(&self.repo_root)
            && let Some(rel_str) = rel.to_str()
            && self.ignore_paths.contains(&rel_str.replace('\\', "/"))
        {
            return false;
        }
        let is_dir = path.is_dir();
        !matches!(
            self.gitignore.matched_path_or_any_parents(path, is_dir),
            Match::Ignore(_)
        )
    }

    /// True if directory `dir` is a place indexed files can live: not itself (or
    /// under) a dot-dir or SKIP_DIRS entry, not gitignored, not a per-repo ignored
    /// path. Used to decide whether a vanished path could have been a directory
    /// holding indexed files — `.git/` lock churn and editor temp dirs must not
    /// trigger a scan of every stored path.
    pub fn could_contain_indexed(&self, dir: &Path) -> bool {
        // The helpers only inspect PARENT components; probe a child so `dir`
        // itself is checked too.
        let probe = dir.join("_");
        if is_under_hidden_dir(&self.repo_root, &probe)
            || is_under_skip_dir(&self.repo_root, &probe)
        {
            return false;
        }
        if !self.ignore_paths.is_empty()
            && let Ok(rel) = dir.strip_prefix(&self.repo_root)
            && let Some(rel_str) = rel.to_str()
            && self.ignore_paths.contains(&rel_str.replace('\\', "/"))
        {
            return false;
        }
        !matches!(
            self.gitignore.matched_path_or_any_parents(dir, true),
            Match::Ignore(_)
        )
    }
}

/// Walk only the subtree `subdir` of the repo at `repo_root`, returning exactly
/// the files the full `walk_repo_with` would return for that subtree.
///
/// Used when a whole directory appears in one watcher event (created, or moved
/// into the repo): the directory's files never get individual events. The inner
/// walk prunes like the full walker; the repo-root-relative rules (dot-dirs and
/// SKIP_DIRS above `subdir`, the root ignore files, per-repo ignored paths) are
/// re-applied with [`ChangeFilter`], since the inner walk is rooted at `subdir`.
/// Cost is O(subtree), never O(repo).
pub fn walk_subtree_with(
    repo_root: &str,
    subdir: &str,
    extra_extensions: &[String],
    ignore_filenames: &HashSet<String>,
    ignore_paths: &HashSet<String>,
) -> Vec<String> {
    let filter = ChangeFilter::new_complete(
        Path::new(repo_root),
        extra_extensions.to_vec(),
        ignore_filenames.clone(),
        ignore_paths.clone(),
    );
    if !filter.could_contain_indexed(Path::new(subdir)) {
        return Vec::new();
    }
    walk_repo_with(subdir, extra_extensions, ignore_filenames, &HashSet::new())
        .into_iter()
        .filter(|p| filter.allows(Path::new(p)))
        .collect()
}

/// Walk a repository directory and return all indexable file paths.
/// Respects .gitignore and .ignore files via the `ignore` crate.
pub fn walk_repo(repo_path: &str) -> Vec<String> {
    walk_repo_with(repo_path, &[], &HashSet::new(), &HashSet::new())
}

pub fn walk_repo_with(
    repo_path: &str,
    extra_extensions: &[String],
    ignore_filenames: &HashSet<String>,
    ignore_paths: &HashSet<String>,
) -> Vec<String> {
    let root = Path::new(repo_path);
    if !root.exists() {
        return vec![];
    }

    let mut files = Vec::new();

    let walker = WalkBuilder::new(root)
        .hidden(false) // include dot-files that aren't gitignored
        .git_ignore(true)
        .git_global(true)
        .ignore(true)
        // `.augmentignore` is a custom ignore file discovered at every directory
        // level. The `ignore` crate ranks custom ignore filenames ABOVE all other
        // ignore sources (`.gitignore`, `.ignore`, global git ignore), so its
        // patterns win. It uses full gitignore syntax including `!pattern`
        // negation, which lets a repo re-include something `.gitignore` excluded
        // (e.g. `!build_config.rs`). Note this only re-includes within the layers
        // below it: the extension allowlist, SKIP_DIRS pruning, and dot-dir
        // pruning still run first, so `!` cannot resurrect files under
        // `node_modules`/`target`/dot-dirs or with non-code extensions.
        .add_custom_ignore_filename(".augmentignore")
        .filter_entry(|entry| {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let name = entry.file_name().to_str().unwrap_or("");
                // Skip non-dot directories from the explicit list.
                if SKIP_DIRS.contains(&name) {
                    return false;
                }
                // Prune any dot-prefixed directory that is NOT the repo root itself.
                // depth() == 0 means this entry IS the root — never prune it, even if
                // the root folder's own name starts with '.'.
                if entry.depth() > 0 && name.starts_with('.') {
                    return false;
                }
            }
            true
        })
        .build();

    for result in walker {
        match result {
            Ok(entry) => {
                let ft = entry.file_type().unwrap_or_else(|| {
                    // DirEntry without file type — skip.
                    entry.file_type().unwrap()
                });
                if !ft.is_file() {
                    continue;
                }
                let path = entry.path();
                if has_indexable_extension_with(path, extra_extensions) {
                    if !ignore_filenames.is_empty()
                        && let Some(fname) = path.file_name().and_then(|n| n.to_str())
                        && ignore_filenames.contains(fname)
                    {
                        continue;
                    }
                    if !ignore_paths.is_empty()
                        && let Ok(rel) = path.strip_prefix(root)
                        && let Some(rel_str) = rel.to_str()
                        && ignore_paths.contains(&rel_str.replace('\\', "/"))
                    {
                        continue;
                    }
                    debug!(path = ?path, "discovered file");
                    if let Some(s) = path.to_str() {
                        files.push(s.to_string());
                    }
                }
            }
            Err(e) => {
                debug!(error = %e, "walk error (skipping)");
            }
        }
    }

    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Normalize a path string to use forward slashes (for cross-platform comparison).
    fn fwd(s: &str) -> String {
        s.replace('\\', "/")
    }

    /// Create a file at `dir/rel_path`, creating parent directories as needed.
    fn touch(dir: &std::path::Path, rel_path: &str) {
        let full = dir.join(rel_path);
        fs::create_dir_all(full.parent().unwrap()).unwrap();
        fs::File::create(&full).unwrap();
    }

    #[test]
    fn walk_repo_skips_dot_directories() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        touch(root, "src/main.rs");
        touch(root, ".eslintrc.json");
        touch(root, ".claude/agents/x.md");
        touch(root, ".agent/y.py");
        touch(root, ".github/workflows/ci.yml");

        let repo_str = root.to_str().unwrap();
        let result: Vec<String> = walk_repo(repo_str).into_iter().map(|p| fwd(&p)).collect();

        // src/main.rs must be indexed.
        let has_main = result.iter().any(|p| p.ends_with("src/main.rs"));
        assert!(has_main, "src/main.rs must be indexed; got: {:?}", result);

        // Root-level dot-file must be indexed.
        let has_eslintrc = result.iter().any(|p| p.ends_with(".eslintrc.json"));
        assert!(
            has_eslintrc,
            ".eslintrc.json must be indexed (root-level dot-file); got: {:?}",
            result
        );

        // Nothing under .claude, .agent, or .github must appear.
        for p in &result {
            assert!(
                !p.contains("/.claude/") && !p.contains("\\.claude\\"),
                ".claude/ content must not be indexed; got path: {p}"
            );
            assert!(
                !p.contains("/.agent/") && !p.contains("\\.agent\\"),
                ".agent/ content must not be indexed; got path: {p}"
            );
            assert!(
                !p.contains("/.github/") && !p.contains("\\.github\\"),
                ".github/ content must not be indexed; got path: {p}"
            );
        }
    }

    #[test]
    fn walk_repo_still_skips_node_modules() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        touch(root, "node_modules/foo/index.js");
        touch(root, "src/main.js");

        let repo_str = root.to_str().unwrap();
        let result: Vec<String> = walk_repo(repo_str).into_iter().map(|p| fwd(&p)).collect();

        let has_main = result.iter().any(|p| p.ends_with("src/main.js"));
        assert!(has_main, "src/main.js must be indexed; got: {:?}", result);

        let has_node_modules = result.iter().any(|p| p.contains("node_modules"));
        assert!(
            !has_node_modules,
            "node_modules content must not be indexed; got: {:?}",
            result
        );
    }

    #[test]
    fn walk_repo_works_when_root_name_is_dotted() {
        let tmp = TempDir::new().unwrap();
        // Create a subdirectory with a dot-prefixed name to use as the repo root.
        let dotted_root = tmp.path().join(".dotted-root");
        fs::create_dir_all(&dotted_root).unwrap();
        touch(&dotted_root, "src/main.rs");

        let repo_str = dotted_root.to_str().unwrap();
        let result: Vec<String> = walk_repo(repo_str).into_iter().map(|p| fwd(&p)).collect();

        let has_main = result.iter().any(|p| p.ends_with("src/main.rs"));
        assert!(
            has_main,
            "src/main.rs must be indexed even when repo root name starts with '.'; got: {:?}",
            result
        );
    }

    #[test]
    fn is_under_hidden_dir_helper() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Create actual paths so strip_prefix works correctly on all platforms.
        let claude_dir = root.join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        let x_md = claude_dir.join("x.md");
        fs::File::create(&x_md).unwrap();

        let eslintrc = root.join(".eslintrc.json");
        fs::File::create(&eslintrc).unwrap();

        let src_dir = root.join("src");
        fs::create_dir_all(&src_dir).unwrap();
        let foo_rs = src_dir.join("foo.rs");
        fs::File::create(&foo_rs).unwrap();

        let nested_dot_dir = root.join("a").join(".b");
        fs::create_dir_all(&nested_dot_dir).unwrap();
        let c_rs = nested_dot_dir.join("c.rs");
        fs::File::create(&c_rs).unwrap();

        // .claude/x.md — inside a dot-dir → true
        assert!(
            is_under_hidden_dir(root, &x_md),
            ".claude/x.md must be under a hidden dir"
        );

        // .eslintrc.json at repo root — root-level dot-FILE → false
        assert!(
            !is_under_hidden_dir(root, &eslintrc),
            ".eslintrc.json is a root-level dot-file, not under a hidden dir"
        );

        // src/foo.rs — normal file → false
        assert!(
            !is_under_hidden_dir(root, &foo_rs),
            "src/foo.rs must not be under a hidden dir"
        );

        // a/.b/c.rs — nested dot-dir → true
        assert!(
            is_under_hidden_dir(root, &c_rs),
            "a/.b/c.rs must be under a hidden dir"
        );
    }

    /// Isolates the GITIGNORE branch of `ChangeFilter::allows`. The candidate
    /// file has an indexable extension (.rs), is NOT in a dot-dir, and is NOT in
    /// any SKIP_DIRS tree — so the only thing that can exclude it is the custom
    /// `.gitignore` pattern. This proves the gitignore matcher is actually the
    /// deciding predicate, which the `target/*.exe` regression test does NOT
    /// (those are killed earlier by the extension allowlist / SKIP_DIRS).
    #[test]
    fn change_filter_gitignore_branch_is_decisive() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Custom ignore pattern that is NOT a SKIP_DIR and NOT a dot-dir:
        // a top-level "generated/" folder of .rs files.
        fs::write(root.join(".gitignore"), "/generated\nsecret.rs\n").unwrap();

        touch(root, "src/keep.rs");
        touch(root, "generated/leak.rs"); // excluded only by /generated
        touch(root, "secret.rs"); // excluded only by the bare pattern

        let filter = ChangeFilter::new(root);

        // Real source survives (no rule excludes it).
        assert!(
            filter.allows(&root.join("src").join("keep.rs")),
            "src/keep.rs must pass — no rule excludes it"
        );
        // gitignored dir: only the gitignore matcher can catch this (.rs ext is
        // indexable, "generated" is not in SKIP_DIRS, not a dot-dir).
        assert!(
            !filter.allows(&root.join("generated").join("leak.rs")),
            "generated/leak.rs must be dropped by the gitignore branch"
        );
        // gitignored bare filename pattern.
        assert!(
            !filter.allows(&root.join("secret.rs")),
            "secret.rs must be dropped by the gitignore branch"
        );
    }

    /// The watcher feeds OS-native paths (backslashes on Windows) while the repo
    /// root from config may use forward slashes. If `ChangeFilter` can't reconcile
    /// the two, the gitignore branch silently becomes a no-op in production and
    /// the whole fix regresses even though the test above (same separators) passes.
    ///
    /// This test builds the filter with a forward-slash root and queries it with
    /// a candidate path joined the OS-native way (what notify actually delivers),
    /// matching the real `incremental_run` path-form mismatch.
    #[test]
    fn change_filter_matches_across_separator_forms() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        fs::write(root.join(".gitignore"), "/generated\n").unwrap();
        touch(root, "generated/leak.rs");
        touch(root, "src/keep.rs");

        // Root in forward-slash form (as config / self.repo stores it).
        let root_fwd = root.to_str().unwrap().replace('\\', "/");
        let filter = ChangeFilter::new(Path::new(&root_fwd));

        // Candidate as the OS delivers it (PathBuf::join → native separators).
        let leak_native = root.join("generated").join("leak.rs");
        let keep_native = root.join("src").join("keep.rs");

        assert!(
            !filter.allows(&leak_native),
            "gitignored path must be dropped regardless of root/candidate separator form; \
             root={root_fwd:?} candidate={leak_native:?}"
        );
        assert!(
            filter.allows(&keep_native),
            "non-ignored path must pass regardless of separator form; candidate={keep_native:?}"
        );
    }

    /// The PRODUCTION bug — `target/*` artifacts — is caught by the SKIP_DIRS
    /// branch (`is_under_skip_dir`), NOT the gitignore branch: gitignore only runs
    /// after extension passes AND the path is not already in a skip dir. That
    /// branch uses `Path::strip_prefix`, a different mechanism than the `ignore`
    /// crate's matcher — so it needs its OWN proof that it survives the Windows
    /// separator mismatch (fwd-slash root from config vs native-separator path
    /// from notify). If `strip_prefix` fails to reconcile them it returns Err →
    /// `is_under_skip_dir` returns false → the artifact leaks. Prove it doesn't.
    ///
    /// Uses a `.rs` file under `target/` so the EXTENSION check (step 1) passes and
    /// SKIP_DIRS (step 3) is the predicate under test — a `.exe` would short-circuit
    /// at step 1 and never reach this branch, defeating the point.
    #[test]
    fn change_filter_skip_dir_survives_separator_mismatch() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        // No .gitignore at all — isolate the SKIP_DIRS branch from gitignore.
        touch(root, "target/debug/generated.rs");
        touch(root, "src/keep.rs");

        // Root in forward-slash form (as config / self.repo stores it).
        let root_fwd = root.to_str().unwrap().replace('\\', "/");
        let filter = ChangeFilter::new(Path::new(&root_fwd));

        // Candidate as notify delivers it (PathBuf::join → native separators).
        let artifact_native = root.join("target").join("debug").join("generated.rs");
        let keep_native = root.join("src").join("keep.rs");

        // Sanity: prove strip_prefix actually reconciles the two forms here.
        // If this fails, is_under_skip_dir's Err arm fires and the assert below
        // would silently pass for the WRONG reason — so check it explicitly.
        assert!(
            is_under_skip_dir(Path::new(&root_fwd), &artifact_native),
            "is_under_skip_dir must reconcile fwd-slash root vs native candidate; \
             root={root_fwd:?} candidate={artifact_native:?}"
        );

        assert!(
            !filter.allows(&artifact_native),
            "target/debug/generated.rs (.rs ext, no .gitignore) must be dropped by the \
             SKIP_DIRS branch across separator forms; root={root_fwd:?} candidate={artifact_native:?}"
        );
        assert!(
            filter.allows(&keep_native),
            "src/keep.rs must still pass; candidate={keep_native:?}"
        );
    }

    #[test]
    fn walk_repo_skips_ignored_filenames() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        touch(root, "src/main.rs");
        touch(root, "CLAUDE.md");
        touch(root, "AGENTS.md");
        touch(root, "src/lib.rs");

        let repo_str = root.to_str().unwrap();
        let ignore: HashSet<String> = ["CLAUDE.md", "AGENTS.md"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let result: Vec<String> = walk_repo_with(repo_str, &[], &ignore, &HashSet::new())
            .into_iter()
            .map(|p| fwd(&p))
            .collect();

        assert!(
            result.iter().any(|p| p.ends_with("src/main.rs")),
            "src/main.rs must be indexed; got: {:?}",
            result
        );
        assert!(
            result.iter().any(|p| p.ends_with("src/lib.rs")),
            "src/lib.rs must be indexed; got: {:?}",
            result
        );
        assert!(
            !result.iter().any(|p| p.ends_with("CLAUDE.md")),
            "CLAUDE.md must be skipped; got: {:?}",
            result
        );
        assert!(
            !result.iter().any(|p| p.ends_with("AGENTS.md")),
            "AGENTS.md must be skipped; got: {:?}",
            result
        );
    }

    #[test]
    fn change_filter_ignores_filenames_but_allows_deleted() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        touch(root, "CLAUDE.md");
        touch(root, "src/main.rs");

        let ignore: HashSet<String> = ["CLAUDE.md"].iter().map(|s| s.to_string()).collect();
        let filter = ChangeFilter::new_full(root, vec![], ignore);

        assert!(
            !filter.allows(&root.join("CLAUDE.md")),
            "CLAUDE.md Modified must be rejected by ignore_filenames"
        );
        assert!(
            filter.allows(&root.join("src").join("main.rs")),
            "src/main.rs must pass"
        );
    }

    #[test]
    fn walk_repo_skips_ignored_paths() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        touch(root, "src/main.rs");
        touch(root, "doc/Building.md");
        touch(root, "doc/README.md");

        let repo_str = root.to_str().unwrap();
        let ignore_paths: HashSet<String> =
            ["doc/Building.md"].iter().map(|s| s.to_string()).collect();
        let result: Vec<String> = walk_repo_with(repo_str, &[], &HashSet::new(), &ignore_paths)
            .into_iter()
            .map(|p| fwd(&p))
            .collect();

        assert!(
            result.iter().any(|p| p.ends_with("src/main.rs")),
            "src/main.rs must be indexed; got: {:?}",
            result
        );
        assert!(
            result.iter().any(|p| p.ends_with("doc/README.md")),
            "doc/README.md must be indexed; got: {:?}",
            result
        );
        assert!(
            !result.iter().any(|p| p.ends_with("doc/Building.md")),
            "doc/Building.md must be skipped by ignore_paths; got: {:?}",
            result
        );
    }

    #[test]
    fn change_filter_ignores_paths_across_separator_forms() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        touch(root, "doc/Building.md");
        touch(root, "src/main.rs");

        // Build filter with forward-slash root (as config may store it).
        let root_fwd = root.to_str().unwrap().replace('\\', "/");
        let ignore_paths: HashSet<String> =
            ["doc/Building.md"].iter().map(|s| s.to_string()).collect();
        let filter =
            ChangeFilter::new_complete(Path::new(&root_fwd), vec![], HashSet::new(), ignore_paths);

        // Candidate as native (PathBuf::join → backslash on Windows).
        let building_native = root.join("doc").join("Building.md");
        let main_native = root.join("src").join("main.rs");

        assert!(
            !filter.allows(&building_native),
            "doc/Building.md must be rejected by ignore_paths across separator forms; \
             root={root_fwd:?} candidate={building_native:?}"
        );
        assert!(
            filter.allows(&main_native),
            "src/main.rs must pass; candidate={main_native:?}"
        );
    }

    /// Full-rebuild path: a top-level `.augmentignore` with a plain exclude
    /// pattern must drop the matching file while a control file is still indexed.
    /// Uses indexable extensions (.md/.rs) and paths outside SKIP_DIRS/dot-dirs so
    /// the `.augmentignore` matcher is the deciding predicate.
    #[test]
    fn walk_repo_respects_augmentignore_exclude() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::write(root.join(".augmentignore"), "notes.md\n/generated2\n").unwrap();

        touch(root, "src/main.rs"); // control — indexed
        touch(root, "notes.md"); // excluded by bare pattern
        touch(root, "generated2/leak.rs"); // excluded by /generated2

        let repo_str = root.to_str().unwrap();
        let result: Vec<String> = walk_repo(repo_str).into_iter().map(|p| fwd(&p)).collect();

        assert!(
            result.iter().any(|p| p.ends_with("src/main.rs")),
            "src/main.rs must be indexed; got: {:?}",
            result
        );
        assert!(
            !result.iter().any(|p| p.ends_with("notes.md")),
            "notes.md must be excluded by .augmentignore; got: {:?}",
            result
        );
        assert!(
            !result.iter().any(|p| p.contains("generated2")),
            "generated2/leak.rs must be excluded by .augmentignore; got: {:?}",
            result
        );
    }

    /// Full-rebuild path: `.augmentignore` ranks ABOVE `.gitignore`, so a `!`
    /// negation re-includes a file `.gitignore` excluded. A second file excluded
    /// by `.gitignore` with no negation proves `.gitignore` still works.
    /// All files use the indexable `.rs` extension so the allowlist never masks
    /// the gitignore-layer result.
    #[test]
    fn walk_repo_augmentignore_negation_overrides_gitignore() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // `WalkBuilder.git_ignore(true)` only honors `.gitignore` when the tree is
        // a git repo (default `require_git`), so create an empty `.git` dir to make
        // the gitignore layer active — otherwise `.gitignore` is a no-op and this
        // test would not actually prove `.augmentignore` outranks it.
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::write(root.join(".gitignore"), "secret.rs\nalso_secret.rs\n").unwrap();
        fs::write(root.join(".augmentignore"), "!secret.rs\n").unwrap();

        touch(root, "secret.rs"); // re-included by .augmentignore negation
        touch(root, "also_secret.rs"); // excluded by .gitignore, no negation
        touch(root, "src/main.rs"); // control

        let repo_str = root.to_str().unwrap();
        let result: Vec<String> = walk_repo(repo_str).into_iter().map(|p| fwd(&p)).collect();

        assert!(
            result
                .iter()
                .any(|p| p.ends_with("secret.rs") && !p.ends_with("also_secret.rs")),
            "secret.rs must be re-included by .augmentignore !negation; got: {:?}",
            result
        );
        assert!(
            !result.iter().any(|p| p.ends_with("also_secret.rs")),
            "also_secret.rs must stay excluded by .gitignore (no negation); got: {:?}",
            result
        );
        assert!(
            result.iter().any(|p| p.ends_with("src/main.rs")),
            "src/main.rs control must be indexed; got: {:?}",
            result
        );
    }

    /// `ChangeFilter` path: a root-level `.augmentignore` exclude pattern must
    /// reject the matching file while a control file passes. Indexable extensions
    /// and non-SKIP_DIRS/non-dot paths so the gitignore-layer is decisive.
    #[test]
    fn change_filter_respects_augmentignore_exclude() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::write(root.join(".augmentignore"), "notes.md\n").unwrap();
        touch(root, "notes.md");
        touch(root, "src/main.rs");

        let filter = ChangeFilter::new(root);

        assert!(
            !filter.allows(&root.join("notes.md")),
            "notes.md must be excluded by .augmentignore"
        );
        assert!(
            filter.allows(&root.join("src").join("main.rs")),
            "src/main.rs must pass"
        );
    }

    /// `ChangeFilter` path: `.augmentignore` is added LAST in the builder so its
    /// `!secret.rs` negation outranks `.gitignore`'s `secret.rs` exclude and
    /// re-includes the file. A gitignore-only-excluded control still returns false.
    #[test]
    fn change_filter_augmentignore_negation_overrides_gitignore() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::write(root.join(".gitignore"), "secret.rs\nalso_secret.rs\n").unwrap();
        fs::write(root.join(".augmentignore"), "!secret.rs\n").unwrap();

        touch(root, "secret.rs");
        touch(root, "also_secret.rs");
        touch(root, "src/main.rs");

        let filter = ChangeFilter::new(root);

        assert!(
            filter.allows(&root.join("secret.rs")),
            "secret.rs must be re-included by .augmentignore !negation"
        );
        assert!(
            !filter.allows(&root.join("also_secret.rs")),
            "also_secret.rs must stay excluded by .gitignore (no negation)"
        );
        assert!(
            filter.allows(&root.join("src").join("main.rs")),
            "src/main.rs control must pass"
        );
    }
}
