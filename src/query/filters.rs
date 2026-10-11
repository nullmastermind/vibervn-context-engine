//! Field-qualified query filter parsing and application.
//!
//! Extracts structured filters from natural-language queries before embedding:
//! `kind:function`, `lang:rust`, `path:src/`, `name:parse_file`.
//!
//! Filters are stripped from the query text so the embedding model receives only
//! the semantic content. After vector search, results are narrowed by the filter
//! predicates. Fuzzy name matching (bounded edit distance) triggers when exact
//! matches yield zero results.

use crate::vector::{Scope, ScopeFilter};

/// Parsed structured filters extracted from a query string.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QueryFilters {
    /// Symbol kind filters (e.g. "function", "class", "struct")
    pub kinds: Vec<String>,
    /// Language filters (e.g. "rust", "typescript", "python")
    pub languages: Vec<String>,
    /// Path prefix/substring filters
    pub path_filters: Vec<String>,
    /// Symbol name filters (exact first, fuzzy fallback)
    pub name_filters: Vec<String>,
    /// Code/docs class to search. NOT a post-filter: it is pushed down into the
    /// vector scan (pre-top-k) so the requested class is never starved by the
    /// other. Deliberately ignored by [`is_empty`](Self::is_empty).
    pub scope: Option<Scope>,
    /// Which extensions count as docs, from the live settings snapshot. `None`
    /// means "use the default list" (callers that have no settings at hand).
    pub docs_extensions: Option<Vec<String>>,
}

impl QueryFilters {
    /// Returns true if no POST-filters are active (`scope` / `docs_extensions`
    /// are applied inside the vector scan, not by `apply_query_filters`).
    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
            && self.languages.is_empty()
            && self.path_filters.is_empty()
            && self.name_filters.is_empty()
    }

    /// Merge another set of filters into this one: list filters are unioned;
    /// `scope` / `docs_extensions` are single-valued, so a value carried by
    /// `other` (the structured, tool-param side) overrides, and an absent one
    /// never erases what is already set.
    pub fn merge(&mut self, other: QueryFilters) {
        self.kinds.extend(other.kinds);
        self.languages.extend(other.languages);
        self.path_filters.extend(other.path_filters);
        self.name_filters.extend(other.name_filters);
        if other.scope.is_some() {
            self.scope = other.scope;
        }
        if other.docs_extensions.is_some() {
            self.docs_extensions = other.docs_extensions;
        }
    }

    /// The scope request resolved against the docs-extensions setting, ready for
    /// the vector scan. `None` when no scope was requested.
    pub fn scope_filter(&self) -> Option<ScopeFilter> {
        let scope = self.scope?;
        Some(match &self.docs_extensions {
            Some(list) => ScopeFilter::new(scope, list),
            None => ScopeFilter::new(scope, crate::config::default_docs_extensions()),
        })
    }
}

/// Parse a query string, extracting recognized filter prefixes.
///
/// Returns `(clean_query, filters)` where `clean_query` has filter tokens removed
/// and is suitable for embedding. Supports quoted values: `kind:"async function"`.
///
/// Recognized prefixes: `kind:`, `lang:`, `language:`, `path:`, `name:`, `scope:`
///
/// `scope:` is only a filter when its value is exactly `code` or `docs`; any other
/// value (`scope:foo`, `scope::resolve`) is ordinary query text and is left in the
/// returned clean query. If several valid `scope:` tokens appear the FIRST wins
/// (the tool prepends its structured value, so it beats one typed in the text).
pub fn parse_query_filters(query: &str) -> (String, QueryFilters) {
    let mut filters = QueryFilters::default();
    let mut clean_parts: Vec<&str> = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = query.chars().collect();
    let len = chars.len();

    while i < len {
        // Skip leading whitespace
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }

        // Try to match a filter prefix at position i
        let remaining = &query[byte_offset(query, i)..];
        if let Some((prefix_len, filter_type)) = match_filter_prefix(remaining) {
            let value_start = i + prefix_len;
            let (value, end_pos) = extract_value(query, &chars, value_start);
            if let FilterType::Scope = filter_type {
                match Scope::parse(&value) {
                    // First valid scope wins; later ones are still stripped.
                    Some(scope) => {
                        filters.scope.get_or_insert(scope);
                    }
                    // Not a scope filter after all: keep the raw token as query text.
                    None => {
                        clean_parts.push(&query[byte_offset(query, i)..byte_offset(query, end_pos)])
                    }
                }
            } else if !value.is_empty() {
                match filter_type {
                    FilterType::Kind => filters.kinds.push(value.to_lowercase()),
                    FilterType::Lang => filters.languages.push(value.to_lowercase()),
                    FilterType::Path => filters.path_filters.push(value),
                    FilterType::Name => filters.name_filters.push(value),
                    FilterType::Scope => unreachable!("handled above"),
                }
            }
            i = end_pos;
        } else {
            // Not a filter — find the end of this word/token
            let word_start = byte_offset(query, i);
            while i < len && !chars[i].is_whitespace() {
                i += 1;
            }
            let word_end = byte_offset(query, i);
            clean_parts.push(&query[word_start..word_end]);
        }
    }

    let clean_query = clean_parts.join(" ");
    (clean_query, filters)
}

/// Bounded Damerau-Levenshtein edit distance with early exit.
///
/// Uses a single-row DP approach with O(min(a.len(), b.len())) memory.
/// Returns the edit distance, or `max_dist + 1` if the actual distance
/// exceeds `max_dist` (early exit optimization).
pub fn bounded_edit_distance(a: &str, b: &str, max_dist: u32) -> u32 {
    // Case-insensitive comparison: normalize both inputs to lowercase.
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let a_chars: Vec<char> = a_lower.chars().collect();
    let b_chars: Vec<char> = b_lower.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    // Length shortcut: if length difference exceeds max_dist, answer is known.
    let len_diff = m.abs_diff(n);
    if len_diff as u32 > max_dist {
        return max_dist + 1;
    }

    // Ensure a is the shorter string for memory efficiency.
    let (short, long, s_len, l_len) = if m <= n {
        (&a_chars, &b_chars, m, n)
    } else {
        (&b_chars, &a_chars, n, m)
    };

    // Single-row DP (Levenshtein, not full Damerau for simplicity + correctness).
    let mut prev_row: Vec<u32> = (0..=(s_len as u32)).collect();
    let mut curr_row: Vec<u32> = vec![0; s_len + 1];

    for j in 1..=l_len {
        curr_row[0] = j as u32;
        let mut row_min = curr_row[0];

        for i in 1..=s_len {
            let cost = if short[i - 1] == long[j - 1] { 0 } else { 1 };
            curr_row[i] = (prev_row[i] + 1) // deletion
                .min(curr_row[i - 1] + 1) // insertion
                .min(prev_row[i - 1] + cost); // substitution
            row_min = row_min.min(curr_row[i]);
        }

        // Early exit: if the minimum value in the current row exceeds max_dist,
        // the final result cannot be within budget.
        if row_min > max_dist {
            return max_dist + 1;
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[s_len]
}

// ─── Internal helpers ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum FilterType {
    Kind,
    Lang,
    Path,
    Name,
    Scope,
}

/// Match a filter prefix at the start of `s`. Returns (char_count_consumed, type).
fn match_filter_prefix(s: &str) -> Option<(usize, FilterType)> {
    let lower = s.to_lowercase();
    if lower.starts_with("kind:") {
        Some((5, FilterType::Kind))
    } else if lower.starts_with("language:") {
        Some((9, FilterType::Lang))
    } else if lower.starts_with("lang:") {
        Some((5, FilterType::Lang))
    } else if lower.starts_with("path:") {
        Some((5, FilterType::Path))
    } else if lower.starts_with("name:") {
        Some((5, FilterType::Name))
    } else if lower.starts_with("scope:") {
        Some((6, FilterType::Scope))
    } else {
        None
    }
}

/// Extract a value after a filter prefix. Handles quoted values.
/// Returns (value_string, next_char_index).
fn extract_value(query: &str, chars: &[char], start: usize) -> (String, usize) {
    let len = chars.len();
    if start >= len {
        return (String::new(), start);
    }

    // Quoted value: consume until matching close quote
    if chars[start] == '"' || chars[start] == '\'' {
        let quote = chars[start];
        let mut end = start + 1;
        while end < len && chars[end] != quote {
            end += 1;
        }
        let value_start_byte = byte_offset(query, start + 1);
        let value_end_byte = byte_offset(query, end);
        let value = query[value_start_byte..value_end_byte].to_string();
        // Skip past closing quote
        let next = if end < len { end + 1 } else { end };
        (value, next)
    } else {
        // Unquoted: consume until whitespace
        let mut end = start;
        while end < len && !chars[end].is_whitespace() {
            end += 1;
        }
        let value_start_byte = byte_offset(query, start);
        let value_end_byte = byte_offset(query, end);
        let value = query[value_start_byte..value_end_byte].to_string();
        (value, end)
    }
}

/// Convert a char index to a byte offset in the string.
fn byte_offset(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(byte_pos, _)| byte_pos)
        .unwrap_or(s.len())
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_query_filters ---

    #[test]
    fn no_filters_passes_through() {
        let (clean, filters) = parse_query_filters("find the authentication handler");
        assert_eq!(clean, "find the authentication handler");
        assert!(filters.is_empty());
    }

    #[test]
    fn single_kind_filter() {
        let (clean, filters) = parse_query_filters("kind:function authentication handler");
        assert_eq!(clean, "authentication handler");
        assert_eq!(filters.kinds, vec!["function"]);
    }

    #[test]
    fn multiple_filters() {
        let (clean, filters) = parse_query_filters("kind:class lang:rust path:src/ user model");
        assert_eq!(clean, "user model");
        assert_eq!(filters.kinds, vec!["class"]);
        assert_eq!(filters.languages, vec!["rust"]);
        assert_eq!(filters.path_filters, vec!["src/"]);
    }

    #[test]
    fn language_alias() {
        let (_, filters) = parse_query_filters("language:typescript auth");
        assert_eq!(filters.languages, vec!["typescript"]);
    }

    #[test]
    fn quoted_value() {
        let (clean, filters) = parse_query_filters("kind:\"async function\" handler");
        assert_eq!(clean, "handler");
        assert_eq!(filters.kinds, vec!["async function"]);
    }

    #[test]
    fn single_quoted_value() {
        let (clean, filters) = parse_query_filters("name:'parse_file' details");
        assert_eq!(clean, "details");
        assert_eq!(filters.name_filters, vec!["parse_file"]);
    }

    #[test]
    fn name_filter() {
        let (clean, filters) = parse_query_filters("name:run_query how it works");
        assert_eq!(clean, "how it works");
        assert_eq!(filters.name_filters, vec!["run_query"]);
    }

    #[test]
    fn filters_at_end() {
        let (clean, filters) = parse_query_filters("authentication handler kind:function");
        assert_eq!(clean, "authentication handler");
        assert_eq!(filters.kinds, vec!["function"]);
    }

    #[test]
    fn case_insensitive_prefix() {
        let (_, filters) = parse_query_filters("Kind:Function LANG:Rust search");
        assert_eq!(filters.kinds, vec!["function"]);
        assert_eq!(filters.languages, vec!["rust"]);
    }

    #[test]
    fn unknown_prefix_passes_through() {
        let (clean, filters) = parse_query_filters("unknown:value search term");
        assert_eq!(clean, "unknown:value search term");
        assert!(filters.is_empty());
    }

    #[test]
    fn empty_input() {
        let (clean, filters) = parse_query_filters("");
        assert_eq!(clean, "");
        assert!(filters.is_empty());
    }

    #[test]
    fn filter_only_query() {
        let (clean, filters) = parse_query_filters("kind:function");
        assert_eq!(clean, "");
        assert_eq!(filters.kinds, vec!["function"]);
    }

    #[test]
    fn merge_filters() {
        let mut f1 = QueryFilters {
            kinds: vec!["function".into()],
            ..Default::default()
        };
        let f2 = QueryFilters {
            kinds: vec!["class".into()],
            languages: vec!["rust".into()],
            ..Default::default()
        };
        f1.merge(f2);
        assert_eq!(f1.kinds, vec!["function", "class"]);
        assert_eq!(f1.languages, vec!["rust"]);
    }

    // --- scope: ---

    use crate::vector::Scope;

    #[test]
    fn scope_is_extracted_and_stripped_from_the_embed_text() {
        let (clean, filters) = parse_query_filters("scope:docs how does sharding work");
        assert_eq!(clean, "how does sharding work");
        assert_eq!(filters.scope, Some(Scope::Docs));
    }

    #[test]
    fn scope_prefix_and_value_are_case_insensitive_and_may_be_quoted() {
        let (clean, filters) = parse_query_filters("SCOPE:Code foo");
        assert_eq!(clean, "foo");
        assert_eq!(filters.scope, Some(Scope::Code));
        let (clean, filters) = parse_query_filters("scope:\"docs\" foo");
        assert_eq!(clean, "foo");
        assert_eq!(filters.scope, Some(Scope::Docs));
    }

    /// The tool prepends its structured `scope:` to the free text, so when the
    /// text carries its own `scope:` the structured one (first) must win.
    #[test]
    fn first_scope_wins_and_later_ones_are_still_stripped() {
        let (clean, filters) = parse_query_filters("scope:code scope:docs foo");
        assert_eq!(clean, "foo");
        assert_eq!(filters.scope, Some(Scope::Code));
    }

    /// `scope:foo` may be real query text (`scope::resolve`, a CSS `scope:` rule),
    /// so an unrecognized value is NOT a filter and stays in the embedded text.
    #[test]
    fn unrecognized_scope_value_is_left_in_the_query() {
        let (clean, filters) = parse_query_filters("scope:foo bar");
        assert_eq!(clean, "scope:foo bar");
        assert_eq!(filters.scope, None);
        let (clean, filters) = parse_query_filters("scope: bar");
        assert_eq!(clean, "scope: bar");
        assert_eq!(filters.scope, None);
    }

    /// `scope` is handled inside the vector scan, not by `apply_query_filters`,
    /// so a scope-only query must still count as "no post-filters".
    #[test]
    fn scope_alone_does_not_activate_post_filters() {
        let (_, filters) = parse_query_filters("scope:docs foo");
        assert!(filters.is_empty());
    }

    #[test]
    fn merge_lets_the_structured_scope_override_the_parsed_one() {
        let mut parsed = QueryFilters {
            scope: Some(Scope::Docs),
            ..Default::default()
        };
        parsed.merge(QueryFilters {
            scope: Some(Scope::Code),
            docs_extensions: Some(vec!["md".into()]),
            ..Default::default()
        });
        assert_eq!(parsed.scope, Some(Scope::Code));
        assert_eq!(parsed.docs_extensions, Some(vec!["md".to_string()]));

        // A merge that carries no scope must not erase an existing one.
        parsed.merge(QueryFilters::default());
        assert_eq!(parsed.scope, Some(Scope::Code));
        assert_eq!(parsed.docs_extensions, Some(vec!["md".to_string()]));
    }

    // --- bounded_edit_distance ---

    #[test]
    fn identical_strings() {
        assert_eq!(bounded_edit_distance("hello", "hello", 3), 0);
    }

    #[test]
    fn single_substitution() {
        assert_eq!(bounded_edit_distance("hello", "hallo", 3), 1);
    }

    #[test]
    fn single_insertion() {
        assert_eq!(bounded_edit_distance("hello", "helloo", 3), 1);
    }

    #[test]
    fn single_deletion() {
        assert_eq!(bounded_edit_distance("hello", "helo", 3), 1);
    }

    #[test]
    fn distance_two() {
        assert_eq!(bounded_edit_distance("parse", "pars", 2), 1);
        assert_eq!(bounded_edit_distance("parse", "prse", 2), 1);
    }

    #[test]
    fn exceeds_max_dist() {
        // "abc" vs "xyz" = distance 3, max_dist = 2 → returns 3
        assert_eq!(bounded_edit_distance("abc", "xyz", 2), 3);
    }

    #[test]
    fn length_shortcut() {
        // Length difference of 5 exceeds max_dist of 2
        assert_eq!(bounded_edit_distance("a", "abcdef", 2), 3);
    }

    #[test]
    fn empty_strings() {
        assert_eq!(bounded_edit_distance("", "", 3), 0);
        assert_eq!(bounded_edit_distance("abc", "", 3), 3);
        assert_eq!(bounded_edit_distance("", "ab", 3), 2);
    }

    #[test]
    fn early_exit_optimization() {
        // Large distance, small budget — should exit early
        assert_eq!(bounded_edit_distance("completely", "different!", 1), 2);
    }

    #[test]
    fn case_insensitive() {
        // Case differences are ignored — "Hello" and "hello" are identical.
        assert_eq!(bounded_edit_distance("Hello", "hello", 3), 0);
    }
}
