//! Cross-language "is this a test file?" detection by path.
//!
//! Used to keep test-only symbols out of retrieval hints. Callers MUST pass a
//! repo-relative path: an absolute path would make a repo that lives under a
//! `test/` directory look like all tests.
//!
//! Windows-indexed paths arrive fully lowercased, so CamelCase suffix rules
//! (`FooServiceTest.java`) can only be checked case-sensitively when the path kept
//! its case; an all-lowercase name falls back to the lowercase suffix, which
//! accepts rare false positives (`latest.java`). A dropped hint symbol is cheap.

/// Directory segments that mark everything below them as tests.
const TEST_DIRS: &[&str] = &["test", "tests", "__tests__", "spec"];

pub fn is_test_path(rel: &str) -> bool {
    let normalized = rel.replace('\\', "/");
    let mut segments: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();
    let Some(file) = segments.pop() else {
        return false;
    };
    let in_test_dir = segments.iter().any(|dir| {
        let dir = dir.to_ascii_lowercase();
        // `*.Tests` is the .NET test-project directory convention.
        TEST_DIRS.contains(&dir.as_str()) || dir.ends_with(".tests")
    });
    in_test_dir || is_test_file_name(file)
}

fn is_test_file_name(file: &str) -> bool {
    let Some((stem, ext)) = file.rsplit_once('.') else {
        return false;
    };
    let lower = stem.to_ascii_lowercase();
    let ends = |suffixes: &[&str]| suffixes.iter().any(|s| lower.ends_with(s));
    match ext.to_ascii_lowercase().as_str() {
        "go" => ends(&["_test"]),
        "py" => lower.starts_with("test_") || lower == "conftest" || ends(&["_test"]),
        "rs" => lower == "tests" || ends(&["_tests", "_test"]),
        "js" | "jsx" | "mjs" | "cjs" | "ts" | "tsx" | "mts" | "cts" | "svelte" => {
            ends(&[".test", ".spec"])
        }
        "rb" => ends(&["_spec", "_test"]),
        "dart" => ends(&["_test"]),
        "lua" | "luau" => ends(&["_spec", "_test", ".spec", ".test"]),
        "c" | "cc" | "cpp" | "cxx" => {
            lower.starts_with("test_")
                || ends(&["_test", "_unittest", "_unittests", "_browsertest"])
        }
        "java" | "kt" | "kts" | "scala" | "cs" | "php" | "swift" | "m" | "mm" => {
            camel_test_suffix(stem)
        }
        _ => false,
    }
}

/// `FooTest` / `FooTests`, without matching `Latest` or `Contest`. When the whole
/// name is lowercase the capital T was lost upstream, so fall back to the plain
/// lowercase suffix.
fn camel_test_suffix(stem: &str) -> bool {
    if stem.ends_with("Test") || stem.ends_with("Tests") {
        return true;
    }
    !stem.chars().any(|c| c.is_ascii_uppercase())
        && (stem.ends_with("test") || stem.ends_with("tests"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all(paths: &[&str], expected: bool) {
        for p in paths {
            assert_eq!(is_test_path(p), expected, "path: {p}");
        }
    }

    #[test]
    fn rust_test_files() {
        assert_all(
            &[
                "src/store/tests.rs",
                "src/mcp/tests.rs",
                "src/indexing/load_repos_tests.rs",
                "src/foo_test.rs",
                "tests/integration.rs",
            ],
            true,
        );
    }

    #[test]
    fn go_python_ruby_dart_lua_test_files() {
        assert_all(
            &[
                "pkg/api/handler_test.go",
                "app/test_models.py",
                "app/models_test.py",
                "app/conftest.py",
                "spec/models/user_spec.rb",
                "lib/user_test.rb",
                "lib/foo_test.dart",
                "test/foo_widget.dart",
                "spec/foo_spec.lua",
                "lib/foo_test.lua",
            ],
            true,
        );
    }

    #[test]
    fn js_ts_svelte_test_files() {
        assert_all(
            &[
                "src/utils.test.ts",
                "src/App.spec.tsx",
                "src/a.test.js",
                "src/Button.test.svelte",
                "src/__tests__/a.js",
                "src/__tests__/deep/b.ts",
            ],
            true,
        );
    }

    #[test]
    fn jvm_dotnet_php_swift_objc_test_files() {
        assert_all(
            &[
                "src/main/java/com/x/FooServiceTest.java",
                "src/main/java/com/x/FooServiceTests.java",
                "src/test/java/com/x/FooService.java",
                "app/src/FooTest.kt",
                "app/Models/UserTest.php",
                "Tests/FooTests.swift",
                "Sources/FooTests.m",
                "MyApp.Tests/UserTests.cs",
                "src/UserTests.cs",
            ],
            true,
        );
    }

    #[test]
    fn c_cpp_test_files() {
        assert_all(
            &[
                "base/foo_unittest.cc",
                "chrome/browser/foo_browsertest.cc",
                "src/foo_test.cpp",
                "src/foo_test.c",
                "src/test_foo.c",
                "src/test/foo.cc",
            ],
            true,
        );
    }

    #[test]
    fn separators_and_case_are_normalized() {
        assert_all(
            &[
                r"src\pkg\handler_test.go",
                r"SRC\TESTS\X.RS",
                // Windows-indexed paths arrive fully lowercased, so the CamelCase
                // `FooServiceTest.java` shows up as `fooservicetest.java`.
                "src/fooservicetest.java",
            ],
            true,
        );
    }

    #[test]
    fn test_directories_mark_any_file_below_them() {
        // File names here match no suffix rule, so only the directory can decide.
        assert_all(
            &[
                "spec/support/helpers.rb",
                "test/fixtures/data.py",
                "tests/common/mod.rs",
                "src/__tests__/setup.js",
                "MyApp.Tests/Program.cs",
                "src/test/resources/Config.java",
            ],
            true,
        );
    }

    #[test]
    fn camel_case_suffix_needs_a_capital_t_when_case_is_preserved() {
        // With case intact, `Latest`/`Contest` must not look like `…Test`.
        assert_all(
            &["src/Latest.java", "src/Contest.cs", "src/Attest.kt"],
            false,
        );
        assert_all(&["src/LatestTest.java", "src/ContestTests.cs"], true);
    }

    #[test]
    fn production_files_are_not_tests() {
        assert_all(
            &[
                "src/store/mod.rs",
                "src/main.py",
                "src/attest.py",
                "src/contest.rs",
                "src/inspect/mod.rs",
                "contest/main.go",
                "src/testing/util.go",
                "src/test_utils/mod.rs",
                "docs/spec_overview.md",
                "src/Latest.java",
                "src/Request.cs",
                "src/specification.ts",
                "src/protest.go",
            ],
            false,
        );
    }
}
