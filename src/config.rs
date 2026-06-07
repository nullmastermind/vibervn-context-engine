use std::{
    fs,
    io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempfile::NamedTempFile;

/// Bump this when a new migration is appended to MIGRATIONS.
pub const CURRENT_VERSION: u32 = 2;

/// Migration function type: transforms a JSON Value from version N to version N+1.
pub type MigrationFn = fn(Value) -> Result<Value, ConfigError>;

/// Ordered list of migration functions. Each entry migrates from version N to N+1,
/// where N is the index into this slice (0-based, so index 0 = v1→v2, etc.).
pub const MIGRATIONS: &[MigrationFn] = &[migrate_v1_to_v2];

/// v1→v2: introduce `data_dir` (Option<PathBuf>). The body is a no-op stamp —
/// `serde(default)` already handles missing fields on deserialize, but we
/// persist an explicit `null` and bump the file's `version` so that an older
/// v1 binary refuses to read this file (VersionTooNew) instead of silently
/// dropping the new field on the next save.
fn migrate_v1_to_v2(mut value: Value) -> Result<Value, ConfigError> {
    if let Value::Object(ref mut obj) = value {
        obj.entry("data_dir".to_string()).or_insert(Value::Null);
    }
    Ok(value)
}

// ─── Settings ──────────────────────────────────────────────────────────────

fn default_embed_concurrency() -> usize {
    // Per-key concurrency: each API key is allowed this many concurrent
    // embedding batches in-flight. Runtime total = this value × number of
    // keys. Default 16.
    16
}

fn default_vector_resident_cap_mb() -> usize {
    // Resident-byte cap for the per-repo sharded vector index, in megabytes.
    // Total resident embedding bytes across all repo shards are kept at or below
    // this; least-recently-used non-active repos are evicted when an insert/warm
    // would exceed it. Cold repos are warmed lazily on query. 0 disables the cap
    // (unbounded — not recommended). Default 2048 MB (~2 GB).
    2048
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub api_keys: Vec<String>,
    /// Per-key concurrency: number of embedding batches in-flight per API key.
    /// Runtime total in-flight batches = embed_concurrency × api_keys.len().
    /// Defaults to 16.
    #[serde(default = "default_embed_concurrency")]
    pub embed_concurrency: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "voyage".to_owned(),
            model: "voyage-4-lite".to_owned(),
            api_keys: Vec::new(),
            embed_concurrency: default_embed_concurrency(),
        }
    }
}

fn default_min_prune_lines() -> u32 {
    // Chunks whose line span is below this are never line-pruned by the reranker
    // (kept whole). Pruning a small chunk saves little and risks losing context.
    16
}

fn default_use_structured_output() -> bool {
    // When true, the reranker requests the provider's native JSON output mode
    // (Gemini responseMimeType / OpenAI response_format) instead of wrapping the
    // ranking in <ranked_indices> XML tags. Providers without a JSON mode fall
    // back to the XML path regardless of this flag.
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmConfig {
    pub provider: String,
    pub rerank_model: String,
    pub api_keys: Vec<String>,
    /// Minimum chunk line-span eligible for line-range pruning during rerank.
    /// Chunks smaller than this are returned whole. Defaults to 16.
    #[serde(default = "default_min_prune_lines")]
    pub rerank_min_prune_lines: u32,
    /// Use the provider's native JSON output mode for reranking instead of XML
    /// tag wrapping. Only honored for providers that support it (google, openai);
    /// others fall back to the XML path with a warning. Defaults to true.
    #[serde(default = "default_use_structured_output")]
    pub use_structured_output: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "google".to_owned(),
            rerank_model: "gemini-3.1-flash-lite".to_owned(),
            api_keys: Vec::new(),
            rerank_min_prune_lines: default_min_prune_lines(),
            use_structured_output: default_use_structured_output(),
        }
    }
}

fn default_mcp_index_wait_secs() -> u64 {
    50
}

fn default_mcp_stale_after_days() -> u64 {
    7
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Settings {
    /// Schema version. Server always stamps CURRENT_VERSION on write.
    pub version: u32,
    /// Absolute paths to indexed repositories.
    pub repos: Vec<String>,
    pub embedding: EmbeddingConfig,
    pub llm: LlmConfig,
    /// Maximum wall-clock seconds the MCP tool will wait for indexing to finish
    /// before returning a partial/error response.
    #[serde(default = "default_mcp_index_wait_secs")]
    pub mcp_index_wait_secs: u64,
    /// Number of days after which a durable last_indexed_at timestamp is
    /// considered stale for MCP freshness checks.
    #[serde(default = "default_mcp_stale_after_days")]
    pub mcp_stale_after_days: u64,
    /// Resident-byte cap for the per-repo sharded vector index, in megabytes.
    /// Bounds in-RAM embedding storage across all repos; LRU-evicts non-active
    /// repos when exceeded. 0 disables the cap. Defaults to 2048 (~2 GB).
    #[serde(default = "default_vector_resident_cap_mb")]
    pub vector_resident_cap_mb: usize,
    /// User's preferred data directory base. RocksDB lives at
    /// `<data_dir>/rocksdb/`, embedding cache at `<data_dir>/embeddings/`.
    /// `settings.json` itself ALWAYS lives at
    /// `~/.vibervn/context-engine/settings.json` regardless of this value.
    ///
    /// `None` means "use the builtin default" (`~/.vibervn/context-engine`),
    /// distinguishing an unset preference from an explicit choice.
    /// Boot precedence: CLI flag > env `CONTEXT_ENGINE_DATA_DIR` >
    /// `Settings.data_dir` > builtin default.
    /// Changes via PUT /api/config persist to disk and take effect on the
    /// NEXT launch only — the running process keeps using its boot-resolved
    /// path so already-open RocksDB handles and warmed vector shards stay
    /// consistent.
    #[serde(default)]
    pub data_dir: Option<PathBuf>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: CURRENT_VERSION,
            repos: Vec::new(),
            embedding: EmbeddingConfig::default(),
            llm: LlmConfig::default(),
            mcp_index_wait_secs: default_mcp_index_wait_secs(),
            mcp_stale_after_days: default_mcp_stale_after_days(),
            vector_resident_cap_mb: default_vector_resident_cap_mb(),
            data_dir: None,
        }
    }
}

// ─── ConfigError ───────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ConfigError {
    /// I/O failure (read / write / create_dir). `op` carries human-readable context.
    Io { op: &'static str, source: io::Error },
    /// settings.json could not be parsed as valid JSON or the schema didn't match.
    Parse(serde_json::Error),
    /// The file was written by a newer binary — this binary cannot read it safely.
    VersionTooNew { found: u32 },
    /// A migration step failed.
    MigrationFailed { from: u32, to: u32, detail: String },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io { op, source } => write!(f, "failed to {op} settings: {source}"),
            ConfigError::Parse(e) => {
                write!(f, "settings.json is corrupt: {e}; fix or delete the file")
            }
            ConfigError::VersionTooNew { found } => write!(
                f,
                "settings.json was written by a newer version of context-engine (version {found}); \
                 upgrade the binary or restore an older settings.json"
            ),
            ConfigError::MigrationFailed { from, to, detail } => {
                write!(f, "migration from v{from} to v{to} failed: {detail}")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

// ─── Path helpers ──────────────────────────────────────────────────────────

/// Return the path of `settings.json` under `home_dir`.
///
/// settings.json's location is intentionally fixed (NOT controlled by
/// `Settings.data_dir`): the data_dir field itself lives inside settings.json,
/// so deriving its location from the field would be circular. See the bootstrap
/// notes on `Settings.data_dir`.
pub fn config_path(home_dir: &Path) -> PathBuf {
    home_dir
        .join(".vibervn")
        .join("context-engine")
        .join("settings.json")
}

/// Return the builtin-default data directory under `home_dir`
/// (`~/.vibervn/context-engine`).
///
/// Used as the lowest-precedence fallback in boot resolution when no CLI flag,
/// env var, or persisted `Settings.data_dir` is set.
pub fn default_data_dir(home_dir: &Path) -> PathBuf {
    home_dir.join(".vibervn").join("context-engine")
}

// ─── Atomic write ──────────────────────────────────────────────────────────

/// Write `settings` atomically to `target`.
///
/// Sequence:
/// 1. `create_dir_all(parent)` — idempotent, race-safe.
/// 2. Create a `NamedTempFile` in `parent` (same-filesystem so rename is atomic).
/// 3. Serialize with `serde_json::to_string_pretty` and write to the tempfile.
/// 4. (Unix) Set 0o600 **before** persist — defensive against tempfile default changes.
/// 5. `temp.persist(target)` — atomic rename.
/// 6. (Unix) Reassert 0o600 **after** persist — closes the rename-onto-existing edge case
///    where the previous target's permissions might have been preserved by the kernel.
///
/// Windows: no permission manipulation. Files inherit the `%USERPROFILE%` NTFS ACLs
/// (owner + SYSTEM + Administrators by default), which is threat-model-equivalent to
/// Unix 0o600. This is intentional, not an oversight.
pub fn write_settings_atomic(target: &Path, settings: &Settings) -> Result<(), ConfigError> {
    let parent = target.parent().expect("settings path must have a parent");

    // 1. Ensure directory exists.
    fs::create_dir_all(parent).map_err(|e| ConfigError::Io {
        op: "create directory for",
        source: e,
    })?;

    // 2. Tempfile in same directory (same filesystem → atomic rename).
    let temp = NamedTempFile::new_in(parent).map_err(|e| ConfigError::Io {
        op: "create tempfile for",
        source: e,
    })?;

    // 3. Serialize and write.
    let json = serde_json::to_string_pretty(settings)
        .map_err(ConfigError::Parse)?;

    fs::write(temp.path(), json.as_bytes()).map_err(|e| ConfigError::Io {
        op: "write tempfile for",
        source: e,
    })?;

    // 4. (Unix only) Set 0o600 before persist.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o600)).map_err(|e| {
            ConfigError::Io {
                op: "set permissions on tempfile for",
                source: e,
            }
        })?;
    }

    // 5. Atomic rename.
    let target_path = target.to_path_buf();
    temp.persist(&target_path).map_err(|e| ConfigError::Io {
        op: "persist (rename) tempfile to",
        source: e.error,
    })?;

    // 6. (Unix only) Reassert 0o600 after persist.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&target_path, fs::Permissions::from_mode(0o600)).map_err(|e| {
            ConfigError::Io {
                op: "set permissions after persist for",
                source: e,
            }
        })?;
    }

    Ok(())
}

// ─── Load with migration ────────────────────────────────────────────────────

/// Ensure the config directory exists, bootstrap a default `settings.json` if absent,
/// run migrations if necessary, and return the current `Settings`.
pub fn ensure_dir_and_load(home_dir: &Path) -> Result<Settings, ConfigError> {
    let path = config_path(home_dir);
    let parent = path.parent().expect("settings path must have a parent");

    // 1. Ensure directory.
    fs::create_dir_all(parent).map_err(|e| ConfigError::Io {
        op: "create directory for",
        source: e,
    })?;

    // 2. Bootstrap default file if absent.
    if !path.exists() {
        write_settings_atomic(&path, &Settings::default())?;
    }

    // 3. Read file.
    let raw = fs::read_to_string(&path).map_err(|e| ConfigError::Io {
        op: "read",
        source: e,
    })?;

    // 4. Parse as generic Value first (needed for migration).
    let mut value: Value = serde_json::from_str(&raw).map_err(ConfigError::Parse)?;

    // 5. Migration logic.
    let file_version = value
        .get("version")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(1); // missing → treat as v1 (forward-compat for hand-written files)

    let settings = if file_version == 0 {
        return Err(ConfigError::MigrationFailed {
            from: 0,
            to: 1,
            detail: "version 0 is not a valid schema version".to_string(),
        });
    } else if file_version == CURRENT_VERSION {
        serde_json::from_value::<Settings>(value).map_err(ConfigError::Parse)?
    } else if file_version > CURRENT_VERSION {
        return Err(ConfigError::VersionTooNew { found: file_version });
    } else {
        // Run migrations from file_version to CURRENT_VERSION.
        for step in file_version..CURRENT_VERSION {
            let idx = (step - 1) as usize; // migration index 0 = v1→v2
            let migrate = MIGRATIONS.get(idx).ok_or_else(|| ConfigError::MigrationFailed {
                from: step,
                to: step + 1,
                detail: format!("no migration registered for v{step}→v{}", step + 1),
            })?;
            value = migrate(value).map_err(|e| match e {
                ConfigError::MigrationFailed { .. } => e,
                other => ConfigError::MigrationFailed {
                    from: step,
                    to: step + 1,
                    detail: other.to_string(),
                },
            })?;
        }

        let mut s = serde_json::from_value::<Settings>(value).map_err(ConfigError::Parse)?;
        // Stamp the migrated content with CURRENT_VERSION before persisting —
        // otherwise the file's `version` field still reads as the OLD version
        // and the next load would re-run the migration. Each migration
        // function focuses on field-shape changes only; the version bump is
        // applied here so it stays in lockstep with CURRENT_VERSION even when
        // a migration is a no-op stamp like v1→v2.
        s.version = CURRENT_VERSION;
        // Re-save with the migrated content.
        write_settings_atomic(&path, &s)?;
        s
    };

    Ok(settings)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// version 0 is invalid — ensure_dir_and_load must return MigrationFailed,
    /// not panic (debug) or silently wrap-around (release).
    #[test]
    fn test_version_zero_returns_migration_error() {
        let home = TempDir::new().expect("tempdir");
        let path = config_path(home.path());

        // Create parent dirs.
        fs::create_dir_all(path.parent().expect("has parent")).expect("create dirs");

        // Write a settings.json with version 0.
        let content = r#"{"version":0,"repos":[],"embedding":{"provider":"voyage","model":"","api_keys":[]},"llm":{"provider":"google","rerank_model":"","api_keys":[]}}"#;
        fs::write(&path, content).expect("write settings.json");

        let result = ensure_dir_and_load(home.path());

        match result {
            Err(ConfigError::MigrationFailed { from, .. }) => {
                assert_eq!(from, 0, "expected 'from' == 0");
            }
            Err(other) => panic!("expected MigrationFailed, got: {other}"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    /// Default Settings carries `data_dir == None`, signalling "use the builtin
    /// default at boot". An explicit `Some(path)` represents a frozen user
    /// choice and changes the lowest-precedence fallback in main.rs.
    #[test]
    fn test_data_dir_default_is_none() {
        let s = Settings::default();
        assert!(s.data_dir.is_none(), "default data_dir must be None");
        assert_eq!(s.version, CURRENT_VERSION);
    }

    /// v1 → v2 migration: stamps an explicit `null` for the new `data_dir`
    /// field and bumps the version, so an old v1 binary refuses the file
    /// (VersionTooNew) instead of silently dropping the field on the next save.
    #[test]
    fn test_v1_to_v2_migration_stamps_null_data_dir() {
        let home = TempDir::new().expect("tempdir");
        let path = config_path(home.path());
        fs::create_dir_all(path.parent().expect("has parent")).expect("create dirs");

        // Write a valid v1 settings.json (no data_dir field).
        let v1 = r#"{
            "version": 1,
            "repos": [],
            "embedding": {"provider":"voyage","model":"voyage-4-lite","api_keys":[]},
            "llm": {"provider":"google","rerank_model":"gemini-3.1-flash-lite","api_keys":[]}
        }"#;
        fs::write(&path, v1).expect("write v1 settings.json");

        // Load: should run the v1→v2 migration.
        let loaded = ensure_dir_and_load(home.path()).expect("load v1");
        assert_eq!(loaded.version, CURRENT_VERSION);
        assert!(loaded.data_dir.is_none(), "data_dir should be None after migration");

        // The on-disk file must now report version 2 with an explicit data_dir
        // field set to null — the tripwire that prevents an older binary from
        // silently re-reading and re-saving without the new field.
        let raw = fs::read_to_string(&path).expect("re-read");
        let v: Value = serde_json::from_str(&raw).expect("parse re-read");
        assert_eq!(v.get("version").and_then(|x| x.as_u64()), Some(2));
        assert!(
            v.get("data_dir").map(|x| x.is_null()).unwrap_or(false),
            "on-disk data_dir should be explicit null after migration, got: {:?}",
            v.get("data_dir")
        );
    }

    /// Round-trip: an explicit `data_dir` value survives serialize+deserialize
    /// and is preserved on subsequent loads (no spurious migration).
    #[test]
    fn test_data_dir_explicit_value_round_trips() {
        let home = TempDir::new().expect("tempdir");
        let path = config_path(home.path());
        fs::create_dir_all(path.parent().expect("has parent")).expect("create dirs");

        let custom = PathBuf::from("/var/data/instance-A");
        let s = Settings {
            data_dir: Some(custom.clone()),
            ..Settings::default()
        };
        write_settings_atomic(&path, &s).expect("write");

        let loaded = ensure_dir_and_load(home.path()).expect("load");
        assert_eq!(loaded.data_dir, Some(custom));
        assert_eq!(loaded.version, CURRENT_VERSION);
    }

    /// `default_data_dir` is the documented fallback used by boot resolution
    /// when no CLI/env/persisted value is set. Pinning it as a public helper
    /// guarantees the same path is used everywhere it's needed.
    #[test]
    fn test_default_data_dir_layout() {
        let home = TempDir::new().expect("tempdir");
        let dd = default_data_dir(home.path());
        assert_eq!(
            dd,
            home.path().join(".vibervn").join("context-engine"),
            "default data_dir must match historical layout for byte-identical default install"
        );
    }
}
