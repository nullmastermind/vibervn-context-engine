//! Code/docs scope classification for the vector scan.
//!
//! Which extensions count as "docs" is a USER SETTING (`Settings.docs_extensions`),
//! so a shard must never bake "is docs" into its rows — a settings edit would
//! leave it stale. Instead each row stores a small interned *extension id*
//! (derived from its file path, never persisted) and the docs/code decision is
//! made per query by looking the id up in a tiny per-query table. A settings
//! change therefore applies on the very next query with nothing to recompute.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Which class of files a retrieval searches. There is deliberately no "both":
/// exactly one class is scanned per call. `Code` is the default everywhere a
/// caller omits it. Inlined in JSON schemas (no `$ref`) because some MCP clients
/// reject schemas with references.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
#[schemars(inline)]
pub enum Scope {
    #[default]
    Code,
    Docs,
}

impl Scope {
    pub fn as_str(self) -> &'static str {
        match self {
            Scope::Code => "code",
            Scope::Docs => "docs",
        }
    }

    /// Case-insensitive; `None` for anything that is not exactly `code`/`docs`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "code" => Some(Scope::Code),
            "docs" => Some(Scope::Docs),
            _ => None,
        }
    }

    pub fn opposite(self) -> Self {
        match self {
            Scope::Code => Scope::Docs,
            Scope::Docs => Scope::Code,
        }
    }
}

/// Normalize one user-supplied extension: trim, drop leading dots, lowercase.
/// Shared by the settings normalizer and [`ScopeFilter::new`] so a hand-edited
/// `settings.json` (`".MD"`) classifies the same as the canonical form (`"md"`).
pub fn normalize_ext(raw: &str) -> String {
    raw.trim().trim_start_matches('.').to_lowercase()
}

/// Classification key of a file path: its lowercase extension, or the whole
/// lowercase filename when there is none (`Dockerfile` → `dockerfile`, and a
/// dotfile such as `.gitignore` stays whole). Handles both `/` and `\`.
pub fn ext_key(path: &str) -> String {
    ext_key_cow(path).into_owned()
}

/// Allocation-free for the common all-lowercase-ASCII case: this runs once per
/// row while warming a shard (hundreds of thousands of rows at kernel scale).
fn ext_key_cow(path: &str) -> Cow<'_, str> {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let key = match name.rfind('.') {
        Some(i) if i > 0 && i + 1 < name.len() => &name[i + 1..],
        _ => name,
    };
    if key.bytes().any(|b| b.is_ascii_uppercase() || b >= 0x80) {
        Cow::Owned(key.to_lowercase())
    } else {
        Cow::Borrowed(key)
    }
}

/// Id used for any extension that did not fit the table. Counts as code (it is
/// never in a docs list), so overflow degrades safely instead of wrapping into a
/// valid id and misclassifying rows.
pub(crate) const EXT_SENTINEL: u16 = u16::MAX;

/// Per-shard interned extension table: id → lowercase extension key. Ids are
/// local to ONE shard (each interns in its own insertion order), so they are not
/// comparable across shards — `VectorIndex::merge` remaps through this table.
#[derive(Debug, Default, Clone)]
pub(crate) struct ExtTable {
    names: Vec<String>,
    ids: HashMap<String, u16>,
}

impl ExtTable {
    pub(crate) fn intern(&mut self, key: &str) -> u16 {
        if let Some(&id) = self.ids.get(key) {
            return id;
        }
        // Valid ids are 0..EXT_SENTINEL; a table that large means a pathological
        // repo — saturate rather than wrap.
        if self.names.len() >= EXT_SENTINEL as usize {
            return EXT_SENTINEL;
        }
        let id = self.names.len() as u16;
        self.names.push(key.to_owned());
        self.ids.insert(key.to_owned(), id);
        id
    }

    pub(crate) fn intern_path(&mut self, path: &str) -> u16 {
        self.intern(&ext_key_cow(path))
    }

    pub(crate) fn names(&self) -> &[String] {
        &self.names
    }
}

/// A scope request resolved against the CURRENT docs-extensions setting.
/// Cheap to clone (the set is shared).
#[derive(Debug, Clone)]
pub struct ScopeFilter {
    scope: Scope,
    docs: Arc<HashSet<String>>,
}

impl ScopeFilter {
    pub fn new<I, S>(scope: Scope, docs_extensions: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let docs = docs_extensions
            .into_iter()
            .map(|e| normalize_ext(e.as_ref()))
            .filter(|e| !e.is_empty())
            .collect();
        Self {
            scope,
            docs: Arc::new(docs),
        }
    }

    pub fn scope(&self) -> Scope {
        self.scope
    }

    /// Same docs set, opposite class — used by the one-shot fallback.
    pub fn opposite(&self) -> Self {
        Self {
            scope: self.scope.opposite(),
            docs: Arc::clone(&self.docs),
        }
    }

    pub(crate) fn keeps(&self, is_docs: bool) -> bool {
        is_docs == (self.scope == Scope::Docs)
    }

    /// Per-id keep table for one shard: O(#distinct extensions), built once per
    /// query per shard, so the scan does a single array lookup per row.
    pub(crate) fn keep_table(&self, table: &ExtTable) -> Vec<bool> {
        table
            .names()
            .iter()
            .map(|n| self.keeps(self.docs.contains(n)))
            .collect()
    }

    /// Verdict for ids outside the table (the overflow sentinel): never docs.
    pub(crate) fn keeps_unknown(&self) -> bool {
        self.keeps(false)
    }
}
