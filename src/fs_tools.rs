//! Exact filesystem tools shared by the two tool-calling agents — the Chat
//! Agent (`chat.rs`) and the Agentic RAG reranker (`query/reranker.rs`).
//!
//! Both agents can search and read the working tree directly, in addition to
//! their semantic (index-backed) tools. The difference is "guess what's
//! relevant" vs "see exactly what is there", which is what keeps answers from
//! drifting into fabricated symbol names or invented code.
//!
//! Everything here is a single source of truth for the **path-traversal guard**
//! (`resolve_within_root`) — the security boundary that keeps an LLM-supplied
//! path from escaping the repo root — so neither agent carries its own copy.
//!
//! All functions are blocking (std fs + regex over file bytes); callers run them
//! under `spawn_blocking` so they never stall the async runtime.

// ─── grep/read bounds (exact filesystem tools) ────────────────────────────
// These keep the two exact tools cost-bounded at kernel scale: a common regex
// over a multi-million-file tree must never stream unbounded output into the
// model, and a single `read` must never pull a whole generated megafile.

/// Hard cap on total grep matches returned across all files in one call. A
/// common term in a huge repo can match millions of lines; we return the first
/// [`GREP_MAX_MATCHES`] (in deterministic walk order) and flag truncation.
const GREP_MAX_MATCHES: usize = 200;
/// Cap on matches returned from any single file, so one noisy file can't crowd
/// out matches the model needs to see from other files.
const GREP_MAX_PER_FILE: usize = 20;
/// Cap on files scanned in one grep, independent of match count — bounds the
/// walk itself on a Chromium/Linux-scale tree even when nothing matches.
const GREP_MAX_FILES_SCANNED: usize = 20_000;
/// Skip any file larger than this from grep (generated bundles, minified JS,
/// binary blobs that slipped past the binary check) — they blow the budget and
/// almost never hold the answer.
const GREP_MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;
/// Context lines allowed on either side of a grep match (the `-C` knob), capped
/// so a large value can't multiply output past the match budget.
pub const GREP_MAX_CONTEXT: usize = 10;

/// Max lines returned by a single `read` call. The model pages with
/// start_line/end_line if it needs more — bounds one read at kernel scale.
const READ_MAX_LINES: usize = 800;
/// Hard byte cap on a single `read` result, enforced alongside the line cap so
/// a file of very long lines can't blow the budget within [`READ_MAX_LINES`].
const READ_MAX_BYTES: usize = 64 * 1024;

// PLACEHOLDER_GUARD

/// Resolve a model-supplied repo-relative path to an absolute path that is
/// PROVABLY inside `root`, or return an error string. This is the hard
/// path-traversal guard for both `grep` and `read`: the model never gets to
/// read outside the conversation's repo, no matter what it passes
/// (`../../etc/passwd`, an absolute path, a symlink escaping the tree).
///
/// Strategy: reject absolute inputs up front, join under root, then canonicalize
/// BOTH root and the target and verify the canonical target is still prefixed by
/// the canonical root. Canonicalization resolves `..` and symlinks against the
/// real filesystem, so a symlink pointing outside the repo is caught too. The
/// file must exist (canonicalize requires it) — callers treat a missing file as
/// a normal "not found" error, which is the right outcome for the agent.
pub fn resolve_within_root(root: &std::path::Path, rel: &str) -> Result<std::path::PathBuf, String> {
    let rel = rel.trim();
    if rel.is_empty() {
        return Err("Error: file_path is required.".to_owned());
    }
    let candidate = std::path::Path::new(rel);
    // Absolute paths (incl. Windows `C:\..` and UNC) are never allowed — the
    // model addresses files relative to the repo root only.
    if candidate.is_absolute() {
        return Err(format!("Error: path must be relative to the repo root, got absolute: {rel}"));
    }
    // Reject Windows drive-relative / verbatim prefixes defensively; on unix this
    // is a no-op. `is_absolute` misses `C:foo` (drive-relative), so also bail if
    // any component looks like a drive/prefix.
    if rel.contains(':') {
        return Err(format!("Error: path must be relative to the repo root: {rel}"));
    }

    let joined = root.join(candidate);
    // Canonicalize the root once; if the repo root itself can't be resolved the
    // index could never have been built, so surface it plainly.
    let canon_root = root
        .canonicalize()
        .map_err(|e| format!("Error: cannot resolve repo root: {e}"))?;
    let canon_target = joined
        .canonicalize()
        .map_err(|_| format!("Error: file not found in repo: {rel}"))?;

    if !canon_target.starts_with(&canon_root) {
        // The path escaped the repo (via `..`, a symlink, or a junction).
        return Err(format!("Error: path escapes the repository root: {rel}"));
    }
    Ok(canon_target)
}

/// Heuristic binary-file detector: a NUL byte in the first 8 KB. ripgrep uses the
/// same signal. Keeps grep from dumping binary noise and from wasting budget.
fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8192).any(|&b| b == 0)
}

// PLACEHOLDER_GREP

/// One addressable match-region from a grep: a contiguous span of lines (the
/// match plus its context, with overlapping spans merged) in one file. The
/// Agentic RAG reranker turns each region into a selectable chunk; the Chat
/// Agent ignores these and uses [`GrepOutcome::text`] only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrepRegion {
    /// Repo-relative path, forward-slashed.
    pub rel_path: String,
    /// 1-based inclusive line range covering the match + its context band.
    pub line_start: u32,
    pub line_end: u32,
}

/// Result of a grep: the formatted `path:line: text` output the model reads,
/// an `ok` flag (false when `text` is an `Error:` string), and the structured
/// match-regions for results-producing callers (Design B).
#[derive(Debug, Clone)]
pub struct GrepOutcome {
    pub text: String,
    pub ok: bool,
    pub regions: Vec<GrepRegion>,
}

/// Exact text/regex search over the repo working tree. Returns matching lines as
/// `path:line: text` (with optional `-C` context), capped at [`GREP_MAX_MATCHES`]
/// total / [`GREP_MAX_PER_FILE`] per file, walking at most
/// [`GREP_MAX_FILES_SCANNED`] files. Respects `.gitignore` via the `ignore`
/// crate and skips binary/oversized files. Blocking — call under
/// `spawn_blocking`.
pub fn run_grep(
    root: &std::path::Path,
    pattern: &str,
    path_scope: Option<&str>,
    literal: bool,
    ignore_case: bool,
    context_lines: usize,
) -> GrepOutcome {
    // Build the regex. `literal` escapes metacharacters so the model can search
    // for `foo(bar)` without crafting a regex; `ignore_case` flips the flag.
    let effective = if literal { regex::escape(pattern) } else { pattern.to_owned() };
    let re = match regex::RegexBuilder::new(&effective).case_insensitive(ignore_case).build() {
        Ok(r) => r,
        Err(e) => return GrepOutcome::err(format!("Error: invalid regex pattern: {e}")),
    };

    // Scope the walk. A path scope is validated against the root so it can't
    // redirect the walk outside the repo; a glob is applied as an overlay filter.
    let canon_root = match root.canonicalize() {
        Ok(r) => r,
        Err(e) => return GrepOutcome::err(format!("Error: cannot resolve repo root: {e}")),
    };

    // Split the scope into a concrete start dir/file (if it names one) and an
    // optional glob. `src/` → walk src/; `src/**/*.rs` → walk src/ filtered by
    // the glob; `*.rs` → walk root filtered by the glob.
    let mut walk_start = canon_root.clone();
    let mut glob: Option<globset::GlobMatcher> = None;
    if let Some(scope) = path_scope.map(str::trim).filter(|s| !s.is_empty()) {
        if scope.contains('*') || scope.contains('?') || scope.contains('[') {
            match globset::Glob::new(scope) {
                Ok(g) => glob = Some(g.compile_matcher()),
                Err(e) => return GrepOutcome::err(format!("Error: invalid path glob: {e}")),
            }
            // Anchor the walk at the longest literal prefix dir of the glob so a
            // glob like `src/**/*.rs` doesn't rescan the whole tree.
            if let Some(prefix) = glob_literal_prefix(scope) {
                let p = canon_root.join(prefix);
                if p.starts_with(&canon_root) && p.exists() {
                    walk_start = p;
                }
            }
        } else {
            // A concrete relative path: validate it's inside the root.
            match resolve_within_root(root, scope) {
                Ok(p) => walk_start = p,
                Err(e) => return GrepOutcome::err(e),
            }
        }
    }

    let mut out = String::new();
    let mut regions: Vec<GrepRegion> = Vec::new();
    let mut total_matches = 0usize;
    let mut files_scanned = 0usize;
    let mut truncated = false;

    let walker = ignore::WalkBuilder::new(&walk_start)
        .hidden(false) // index dotfiles too; .gitignore still applies
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();

    'walk: for dent in walker {
        let dent = match dent {
            Ok(d) => d,
            Err(_) => continue,
        };
        if !dent.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let path = dent.path();
        // Defense in depth: never read a file that resolves outside the root,
        // even if the walker somehow yielded one (symlinked dir, junction).
        let Ok(canon) = path.canonicalize() else { continue };
        if !canon.starts_with(&canon_root) {
            continue;
        }
        // Apply the glob overlay against the repo-relative path.
        if let Some(g) = &glob {
            let rel = canon.strip_prefix(&canon_root).unwrap_or(&canon);
            if !g.is_match(rel) {
                continue;
            }
        }

        files_scanned += 1;
        if files_scanned > GREP_MAX_FILES_SCANNED {
            truncated = true;
            break;
        }

        // Skip oversized files outright (generated bundles, blobs).
        if std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) > GREP_MAX_FILE_BYTES {
            continue;
        }
        let Ok(bytes) = std::fs::read(path) else { continue };
        if looks_binary(&bytes) {
            continue;
        }
        let Ok(text) = String::from_utf8(bytes) else { continue };

        let rel_display = canon.strip_prefix(&canon_root).unwrap_or(&canon);
        let rel_str = rel_display.to_string_lossy().replace('\\', "/");
        let lines: Vec<&str> = text.lines().collect();
        let total_lines = lines.len();
        let mut file_matches = 0usize;

        for (idx, line) in lines.iter().enumerate() {
            if !re.is_match(line) {
                continue;
            }
            // Emit context-before (only the band immediately preceding), the
            // match line marked with `:`, then context-after. Context lines use
            // `-` as the separator so the model can tell them from matches.
            if context_lines > 0 {
                let from = idx.saturating_sub(context_lines);
                for (j, ctx) in lines[from..idx].iter().enumerate() {
                    out.push_str(&format!("{rel_str}-{}-{ctx}\n", from + j + 1));
                }
            }
            out.push_str(&format!("{rel_str}:{}: {line}\n", idx + 1));
            if context_lines > 0 {
                let to = (idx + 1 + context_lines).min(lines.len());
                for (j, ctx) in lines[idx + 1..to].iter().enumerate() {
                    out.push_str(&format!("{rel_str}-{}-{ctx}\n", idx + j + 2));
                }
            }

            // Record the addressable region for this match (1-based, inclusive),
            // spanning the context band. Merge into the previous region for the
            // same file when they touch or overlap, so adjacent matches collapse
            // into one selectable chunk rather than many slivers.
            let lo = (idx + 1).saturating_sub(context_lines).max(1) as u32;
            let hi = (idx + 1 + context_lines).min(total_lines.max(1)) as u32;
            push_region(&mut regions, &rel_str, lo, hi);

            total_matches += 1;
            file_matches += 1;
            if total_matches >= GREP_MAX_MATCHES {
                truncated = true;
                break 'walk;
            }
            if file_matches >= GREP_MAX_PER_FILE {
                out.push_str(&format!(
                    "{rel_str}: … more matches in this file omitted (per-file cap {GREP_MAX_PER_FILE})\n"
                ));
                break;
            }
        }
    }

    if total_matches == 0 {
        return GrepOutcome {
            text: format!(
                "No matches for pattern `{pattern}`{}. Note: grep matches exact text/regex — if you \
                 expected a match, try different wording with codebase-retrieval (semantic) or check \
                 the pattern.",
                path_scope.map(|s| format!(" in {s}")).unwrap_or_default()
            ),
            ok: true,
            regions: Vec::new(),
        };
    }
    if truncated {
        out.push_str(&format!(
            "\n[truncated: hit the {GREP_MAX_MATCHES}-match / {GREP_MAX_FILES_SCANNED}-file cap — \
             narrow with a more specific pattern or a `path` scope]\n"
        ));
    }
    GrepOutcome { text: out, ok: true, regions }
}

impl GrepOutcome {
    fn err(msg: String) -> Self {
        GrepOutcome { text: msg, ok: false, regions: Vec::new() }
    }
}

/// Append a region for `rel_path`, merging into the last region when it is the
/// same file and the spans touch or overlap (so adjacent matches collapse).
fn push_region(regions: &mut Vec<GrepRegion>, rel_path: &str, lo: u32, hi: u32) {
    if let Some(last) = regions.last_mut()
        && last.rel_path == rel_path
        && lo <= last.line_end.saturating_add(1)
    {
        last.line_end = last.line_end.max(hi);
        last.line_start = last.line_start.min(lo);
        return;
    }
    regions.push(GrepRegion { rel_path: rel_path.to_owned(), line_start: lo, line_end: hi });
}

/// Longest leading directory of a glob with no wildcard, used to anchor the walk
/// (e.g. `src/foo/**/*.rs` → `src/foo`). Returns `None` when the first component
/// already contains a wildcard (`*.rs`), so the caller walks from the root.
fn glob_literal_prefix(glob: &str) -> Option<String> {
    let mut prefix = Vec::new();
    for comp in glob.split('/') {
        if comp.contains('*') || comp.contains('?') || comp.contains('[') {
            break;
        }
        prefix.push(comp);
    }
    if prefix.is_empty() {
        None
    } else {
        Some(prefix.join("/"))
    }
}

// PLACEHOLDER_READ

/// Result of a read: the formatted numbered-line output, an `ok` flag, and —
/// when the read succeeded — the repo-relative path and the 1-based inclusive
/// line range actually emitted, so a results-producing caller can make it an
/// addressable chunk. `range` is `None` for errors and empty files.
#[derive(Debug, Clone)]
pub struct ReadOutcome {
    pub text: String,
    pub ok: bool,
    pub range: Option<(String, u32, u32)>,
}

/// Read one file's verbatim contents as numbered lines, scoped to `root` and
/// bounded by [`READ_MAX_LINES`] / [`READ_MAX_BYTES`]. `start_line`/`end_line`
/// are 1-based inclusive; out-of-range values clamp rather than error. Blocking —
/// call under `spawn_blocking`.
pub fn run_read(
    root: &std::path::Path,
    file_path: &str,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> ReadOutcome {
    let abs = match resolve_within_root(root, file_path) {
        Ok(p) => p,
        Err(e) => return ReadOutcome::err(e),
    };
    if !abs.is_file() {
        return ReadOutcome::err(format!("Error: not a regular file: {file_path}"));
    }
    let bytes = match std::fs::read(&abs) {
        Ok(b) => b,
        Err(e) => return ReadOutcome::err(format!("Error: could not read file: {e}")),
    };
    if looks_binary(&bytes) {
        return ReadOutcome::err(format!("Error: file appears to be binary, not reading: {file_path}"));
    }
    let text = match String::from_utf8(bytes) {
        Ok(t) => t,
        Err(_) => return ReadOutcome::err(format!("Error: file is not valid UTF-8: {file_path}")),
    };

    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    if total == 0 {
        return ReadOutcome { text: format!("{file_path} is empty (0 lines)."), ok: true, range: None };
    }

    // Clamp the 1-based range into [1, total]. Defaults: whole file (capped).
    let start = start_line.unwrap_or(1).max(1);
    if start > total {
        return ReadOutcome::err(format!(
            "Error: start_line {start} is past end of file ({total} lines): {file_path}"
        ));
    }
    let end = end_line.unwrap_or(total).min(total).max(start);

    // Enforce the line cap; if the requested window is bigger, serve the first
    // READ_MAX_LINES and tell the model to page from where we stopped. The byte
    // cap ([`READ_MAX_BYTES`]) may cut earlier still on files of very long lines.
    let line_cap_end = end.min(start + READ_MAX_LINES - 1);

    let mut body = String::new();
    let mut emitted_to = start - 1; // last line number actually emitted
    for (offset, line) in lines[start - 1..line_cap_end].iter().enumerate() {
        let n = start + offset;
        let rendered = format!("{n}: {line}\n");
        if !body.is_empty() && body.len() + rendered.len() > READ_MAX_BYTES {
            break;
        }
        body.push_str(&rendered);
        emitted_to = n;
    }

    let rel = file_path.trim().replace('\\', "/");
    let mut out = format!("{rel} (lines {start}-{emitted_to} of {total})\n");
    out.push_str(&body);
    // Truncated whenever we stopped short of the user's requested end line.
    if emitted_to < end {
        out.push_str(&format!(
            "\n[truncated at the per-read cap — call read again with start_line={} to continue]\n",
            emitted_to + 1
        ));
    }
    ReadOutcome { text: out, ok: true, range: Some((rel, start as u32, emitted_to as u32)) }
}

impl ReadOutcome {
    fn err(msg: String) -> Self {
        ReadOutcome { text: msg, ok: false, range: None }
    }
}




#[cfg(test)]
mod tests {
    use super::*;

    // ─── Path-traversal guard (the hard security boundary) ────────────────

    #[test]
    fn resolve_within_root_accepts_file_inside() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hi").unwrap();
        let got = resolve_within_root(dir.path(), "a.txt");
        assert!(got.is_ok(), "a file inside the root must resolve: {got:?}");
    }

    #[test]
    fn resolve_within_root_rejects_dotdot_escape() {
        let dir = tempfile::tempdir().unwrap();
        let outside = dir.path().parent().unwrap().join("secret.txt");
        std::fs::write(&outside, "secret").unwrap();
        let sub = dir.path().join("repo");
        std::fs::create_dir(&sub).unwrap();
        let r = resolve_within_root(&sub, "../secret.txt");
        assert!(r.is_err(), "../ escape must be rejected");
        let _ = std::fs::remove_file(&outside);
    }

    #[test]
    fn resolve_within_root_rejects_absolute() {
        let dir = tempfile::tempdir().unwrap();
        assert!(resolve_within_root(dir.path(), "/etc/passwd").is_err());
        assert!(resolve_within_root(dir.path(), "").is_err());
    }

    #[test]
    fn resolve_within_root_rejects_colon_paths() {
        let dir = tempfile::tempdir().unwrap();
        assert!(resolve_within_root(dir.path(), "C:windows").is_err());
        assert!(resolve_within_root(dir.path(), "file.txt:stream").is_err());
    }

    #[test]
    fn resolve_within_root_missing_file_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let r = resolve_within_root(dir.path(), "nope.txt");
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("not found"));
    }

    // PLACEHOLDER_READ_TESTS

    // ─── read: line-range logic ───────────────────────────────────────────

    fn write_lines(dir: &std::path::Path, name: &str, n: usize) -> String {
        let body: String = (1..=n).map(|i| format!("line {i}\n")).collect();
        std::fs::write(dir.join(name), &body).unwrap();
        name.to_owned()
    }

    #[test]
    fn read_whole_small_file() {
        let dir = tempfile::tempdir().unwrap();
        write_lines(dir.path(), "f.txt", 5);
        let o = run_read(dir.path(), "f.txt", None, None);
        assert!(o.ok);
        assert!(o.text.contains("lines 1-5 of 5"));
        assert!(o.text.contains("1: line 1"));
        assert!(o.text.contains("5: line 5"));
        assert!(!o.text.contains("[truncated"));
        assert_eq!(o.range, Some(("f.txt".to_owned(), 1, 5)));
    }

    #[test]
    fn read_respects_explicit_range() {
        let dir = tempfile::tempdir().unwrap();
        write_lines(dir.path(), "f.txt", 100);
        let o = run_read(dir.path(), "f.txt", Some(10), Some(12));
        assert!(o.text.contains("lines 10-12 of 100"));
        assert!(o.text.contains("10: line 10"));
        assert!(o.text.contains("12: line 12"));
        assert!(!o.text.contains("9: line 9"));
        assert!(!o.text.contains("13: line 13"));
        assert_eq!(o.range, Some(("f.txt".to_owned(), 10, 12)));
    }

    #[test]
    fn read_clamps_end_past_eof() {
        let dir = tempfile::tempdir().unwrap();
        write_lines(dir.path(), "f.txt", 5);
        let o = run_read(dir.path(), "f.txt", Some(3), Some(999));
        assert!(o.text.contains("lines 3-5 of 5"));
        assert!(o.text.contains("5: line 5"));
    }

    #[test]
    fn read_start_past_eof_errors() {
        let dir = tempfile::tempdir().unwrap();
        write_lines(dir.path(), "f.txt", 5);
        let o = run_read(dir.path(), "f.txt", Some(99), None);
        assert!(!o.ok);
        assert!(o.text.starts_with("Error:"));
        assert!(o.text.contains("past end of file"));
        assert!(o.range.is_none());
    }

    #[test]
    fn read_enforces_line_cap_and_paging_hint() {
        let dir = tempfile::tempdir().unwrap();
        write_lines(dir.path(), "big.txt", READ_MAX_LINES + 50);
        let o = run_read(dir.path(), "big.txt", None, None);
        assert!(o.text.contains(&format!("lines 1-{READ_MAX_LINES} of {}", READ_MAX_LINES + 50)));
        assert!(o.text.contains("[truncated"));
        assert!(o.text.contains(&format!("start_line={}", READ_MAX_LINES + 1)));
        assert_eq!(o.range, Some(("big.txt".to_owned(), 1, READ_MAX_LINES as u32)));
    }

    #[test]
    fn read_rejects_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let o = run_read(dir.path(), "../../etc/passwd", None, None);
        assert!(!o.ok);
        assert!(o.text.starts_with("Error:"));
    }

    #[test]
    fn read_rejects_binary() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("b.bin"), [0x00, 0x01, 0x02, b'a']).unwrap();
        let o = run_read(dir.path(), "b.bin", None, None);
        assert!(!o.ok);
        assert!(o.text.contains("binary"));
    }

    // PLACEHOLDER_GREP_TESTS

    // ─── grep: matching, caps, scope, context ─────────────────────────────

    #[test]
    fn grep_finds_literal_and_regex() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn alpha() {}\nlet x = 1;\nfn beta() {}\n").unwrap();
        let o = run_grep(dir.path(), r"fn \w+", None, false, false, 0);
        assert!(o.ok);
        assert!(o.text.contains("a.rs:1: fn alpha() {}"));
        assert!(o.text.contains("a.rs:3: fn beta() {}"));
        assert!(!o.text.contains("let x"));
        // Two matches in one file → at least one addressable region for it.
        assert!(o.regions.iter().all(|r| r.rel_path == "a.rs"));
        assert!(!o.regions.is_empty());
    }

    #[test]
    fn grep_literal_escapes_metacharacters() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "foo(bar)\nfooXbar\n").unwrap();
        let o = run_grep(dir.path(), "foo(bar)", None, true, false, 0);
        assert!(o.text.contains("a.rs:1: foo(bar)"));
        assert!(!o.text.contains("fooXbar"));
    }

    #[test]
    fn grep_ignore_case() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "Hello\nhELLo\nworld\n").unwrap();
        let o = run_grep(dir.path(), "hello", None, false, true, 0);
        assert!(o.text.contains("a.rs:1: Hello"));
        assert!(o.text.contains("a.rs:2: hELLo"));
    }

    #[test]
    fn grep_context_lines() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "one\ntwo\nMATCH\nfour\nfive\n").unwrap();
        let o = run_grep(dir.path(), "MATCH", None, false, false, 1);
        assert!(o.text.contains("a.rs-2-two"));
        assert!(o.text.contains("a.rs:3: MATCH"));
        assert!(o.text.contains("a.rs-4-four"));
        assert!(!o.text.contains("one"));
        assert!(!o.text.contains("five"));
        // Region spans the context band [2,4].
        assert_eq!(o.regions, vec![GrepRegion { rel_path: "a.rs".into(), line_start: 2, line_end: 4 }]);
    }

    #[test]
    fn grep_no_match_explains() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "nothing here\n").unwrap();
        let o = run_grep(dir.path(), "zzzznotfound", None, false, false, 0);
        assert!(o.ok);
        assert!(o.text.starts_with("No matches"));
        assert!(o.regions.is_empty());
    }

    #[test]
    fn grep_invalid_regex_errors() {
        let dir = tempfile::tempdir().unwrap();
        let o = run_grep(dir.path(), "(unclosed", None, false, false, 0);
        assert!(!o.ok);
        assert!(o.text.starts_with("Error: invalid regex"));
    }

    #[test]
    fn grep_per_file_cap() {
        let dir = tempfile::tempdir().unwrap();
        let body: String = (0..(GREP_MAX_PER_FILE + 30)).map(|_| "hit\n").collect();
        std::fs::write(dir.path().join("a.rs"), body).unwrap();
        let o = run_grep(dir.path(), "hit", None, false, false, 0);
        let hits = o.text.matches(":  hit").count() + o.text.matches(": hit").count();
        assert!(hits <= GREP_MAX_PER_FILE, "per-file cap must bound matches, got {hits}");
        assert!(o.text.contains("per-file cap"));
    }

    #[test]
    fn grep_skips_binary_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "needle\n").unwrap();
        std::fs::write(dir.path().join("b.bin"), [b'n', b'e', 0x00, b'e', b'd', b'l', b'e']).unwrap();
        let o = run_grep(dir.path(), "needle", None, false, false, 0);
        assert!(o.text.contains("a.rs:1: needle"));
        assert!(!o.text.contains("b.bin"));
    }

    #[test]
    fn grep_scopes_to_subdir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/in.rs"), "target\n").unwrap();
        std::fs::write(dir.path().join("out.rs"), "target\n").unwrap();
        let o = run_grep(dir.path(), "target", Some("src"), false, false, 0);
        assert!(o.text.contains("in.rs:1: target"));
        assert!(!o.text.contains("out.rs"));
    }

    #[test]
    fn grep_glob_filter() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "match\n").unwrap();
        std::fs::write(dir.path().join("a.txt"), "match\n").unwrap();
        let o = run_grep(dir.path(), "match", Some("**/*.rs"), false, false, 0);
        assert!(o.text.contains("a.rs:1: match"));
        assert!(!o.text.contains("a.txt"));
    }

    #[test]
    fn grep_rejects_traversal_scope() {
        let dir = tempfile::tempdir().unwrap();
        let o = run_grep(dir.path(), "x", Some("../.."), false, false, 0);
        assert!(!o.ok);
        assert!(o.text.starts_with("Error:"));
    }

    #[test]
    fn grep_merges_adjacent_regions() {
        // Matches on consecutive lines with context collapse into ONE region,
        // not one sliver per match — the addressable unit the reranker commits.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "hit\nhit\nhit\ngap\n").unwrap();
        let o = run_grep(dir.path(), "hit", None, false, false, 0);
        assert_eq!(o.regions, vec![GrepRegion { rel_path: "a.rs".into(), line_start: 1, line_end: 3 }]);
    }

    #[test]
    fn glob_literal_prefix_extracts_dir() {
        assert_eq!(glob_literal_prefix("src/foo/**/*.rs"), Some("src/foo".to_owned()));
        assert_eq!(glob_literal_prefix("src/*.rs"), Some("src".to_owned()));
        assert_eq!(glob_literal_prefix("*.rs"), None);
    }
}
