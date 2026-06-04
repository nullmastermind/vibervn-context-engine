use std::{
    fs,
    io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempfile::NamedTempFile;

/// Bump this when a new migration is appended to MIGRATIONS.
pub const CURRENT_VERSION: u32 = 1;

/// Migration function type: transforms a JSON Value from version N to version N+1.
pub type MigrationFn = fn(Value) -> Result<Value, ConfigError>;

/// Ordered list of migration functions. Each entry migrates from version N to N+1,
/// where N is the index into this slice (0-based, so index 0 = v1→v2, etc.).
/// Currently empty because CURRENT_VERSION == 1 and no prior versions exist.
pub const MIGRATIONS: &[MigrationFn] = &[];

// ─── Settings ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub api_keys: Vec<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "voyage".to_owned(),
            model: "voyage-4-lite".to_owned(),
            api_keys: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmConfig {
    pub provider: String,
    pub rerank_model: String,
    pub api_keys: Vec<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "google".to_owned(),
            rerank_model: "gemini-3.1-flash-lite".to_owned(),
            api_keys: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Settings {
    /// Schema version. Server always stamps CURRENT_VERSION on write.
    pub version: u32,
    /// Absolute paths to indexed repositories.
    pub repos: Vec<String>,
    pub embedding: EmbeddingConfig,
    pub llm: LlmConfig,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: CURRENT_VERSION,
            repos: Vec::new(),
            embedding: EmbeddingConfig::default(),
            llm: LlmConfig::default(),
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
pub fn config_path(home_dir: &Path) -> PathBuf {
    home_dir
        .join(".vibervn")
        .join("context-engine")
        .join("settings.json")
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

        let s = serde_json::from_value::<Settings>(value).map_err(ConfigError::Parse)?;
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
}
