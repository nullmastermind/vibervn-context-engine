use std::{
    collections::HashMap,
    fs,
    io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempfile::NamedTempFile;

/// Bump this when a new migration is appended to MIGRATIONS.
pub const CURRENT_VERSION: u32 = 8;

/// Migration function type: transforms a JSON Value from version N to version N+1.
pub type MigrationFn = fn(Value) -> Result<Value, ConfigError>;

/// Ordered list of migration functions. Each entry migrates from version N to N+1,
/// where N is the index into this slice (0-based, so index 0 = v1→v2, etc.).
pub const MIGRATIONS: &[MigrationFn] = &[migrate_v1_to_v2, migrate_v2_to_v3, migrate_v3_to_v4, migrate_v4_to_v5, migrate_v5_to_v6, migrate_v6_to_v7, migrate_v7_to_v8];

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

/// v2→v3: introduce `embeddings_dir` (Option<PathBuf>). Same no-op stamp +
/// forward-incompat tripwire rationale as v1→v2. `embeddings_dir` lets the
/// content-addressed embedding cache live at its own location — typically a
/// SHARED path across multiple instances, so identical code chunks are embedded
/// once (the cache is concurrency-safe; only RocksDB needs per-instance
/// isolation). `None` means the builtin default
/// `~/.vibervn/context-engine/embeddings` (anchored to home, not `data_dir`).
fn migrate_v2_to_v3(mut value: Value) -> Result<Value, ConfigError> {
    if let Value::Object(ref mut obj) = value {
        obj.entry("embeddings_dir".to_string()).or_insert(Value::Null);
    }
    Ok(value)
}

/// v3→v4: introduce `enabled_mcp_tools` (Vec<String>). Defaults to both tools
/// enabled so existing installations gain file-retrieval without manual opt-in.
fn migrate_v3_to_v4(mut value: Value) -> Result<Value, ConfigError> {
    if let Value::Object(ref mut obj) = value {
        obj.entry("enabled_mcp_tools".to_string()).or_insert_with(|| {
            Value::Array(vec![
                Value::String("codebase-retrieval".to_string()),
                Value::String("file-retrieval".to_string()),
            ])
        });
    }
    Ok(value)
}

/// v4→v5: introduce `custom_extensions` (Vec<String>). Defaults to empty —
/// users add their own extensions beyond the built-in CODE_EXTENSIONS list.
fn migrate_v4_to_v5(mut value: Value) -> Result<Value, ConfigError> {
    if let Value::Object(ref mut obj) = value {
        obj.entry("custom_extensions".to_string())
            .or_insert_with(|| Value::Array(vec![]));
    }
    Ok(value)
}

/// v5→v6: introduce `index_ignore_filenames` (Vec<String>). Defaults to
/// CLAUDE.md and AGENTS.md — files typically consumed by AI agents that add
/// noise to code search results.
fn migrate_v5_to_v6(mut value: Value) -> Result<Value, ConfigError> {
    if let Value::Object(ref mut obj) = value {
        obj.entry("index_ignore_filenames".to_string())
            .or_insert_with(|| {
                Value::Array(vec![
                    Value::String("CLAUDE.md".to_string()),
                    Value::String("AGENTS.md".to_string()),
                ])
            });
    }
    Ok(value)
}

/// v6→v7: introduce `embedding.voyage_base_url` (Option<String>). Allows
/// overriding the Voyage AI endpoint (proxy, self-hosted compatible API).
/// `None` / null means the default `https://api.voyageai.com/v1/embeddings`.
fn migrate_v6_to_v7(mut value: Value) -> Result<Value, ConfigError> {
    if let Value::Object(ref mut obj) = value
        && let Some(Value::Object(emb)) = obj.get_mut("embedding")
    {
        emb.entry("voyage_base_url".to_string())
            .or_insert(Value::Null);
    }
    Ok(value)
}

/// v7→v8: introduce `repo_generations` (map of repo path → generation counter).
/// Defaults to an empty object — every existing repo is implicitly generation 0,
/// which `db_path` maps to exactly today's `<data_dir>/rocksdb/<name>` layout, so
/// no on-disk index is orphaned by this migration. The counter is bumped on every
/// repo/index delete so the next index for that repo lands on a FRESH directory
/// (`<data_dir>/rocksdb/<gen>/<name>` for gen ≥ 1), side-stepping the async RocksDB
/// LOCK drain that otherwise makes an immediate re-index race the deleted handle.
fn migrate_v7_to_v8(mut value: Value) -> Result<Value, ConfigError> {
    if let Value::Object(ref mut obj) = value {
        obj.entry("repo_generations".to_string())
            .or_insert_with(|| Value::Object(serde_json::Map::new()));
    }
    Ok(value)
}

// ─── Settings ──────────────────────────────────────────────────────────────

fn default_index_ignore_filenames() -> Vec<String> {
    vec!["CLAUDE.md".to_string(), "AGENTS.md".to_string()]
}

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
    /// Custom Voyage AI-compatible endpoint. Honored only when
    /// `provider == "voyage"`. `None` / blank → the client falls back to
    /// `https://api.voyageai.com/v1/embeddings`. Accepts either the base form
    /// (`…/v1`) or the full `…/v1/embeddings` URL — normalization is
    /// centralized in `embedding::voyage::voyage_url`.
    #[serde(default)]
    pub voyage_base_url: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "voyage".to_owned(),
            model: "voyage-4-lite".to_owned(),
            api_keys: Vec::new(),
            embed_concurrency: default_embed_concurrency(),
            voyage_base_url: None,
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

fn default_agentic_rag_max_turns() -> u32 {
    9
}

fn default_agentic_rag_max_chunk_chars() -> u32 {
    50_000
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
    /// When true, the rerank step uses a tool-calling agent loop instead of a
    /// single-shot LLM call. The agent uses `query` (search for more context)
    /// and `add_chunks` (commit relevant chunks) to build the final result.
    #[serde(default)]
    pub agentic_rag: bool,
    /// Turn budget for the agentic loop, counted as the number of `query` tool
    /// calls. When the agent has issued this many queries, the loop stops.
    /// Defaults to 9.
    #[serde(default = "default_agentic_rag_max_turns")]
    pub agentic_rag_max_turns: u32,
    /// Character budget for accumulated chunk content in the agentic loop. Once
    /// the total emitted characters of added chunks reaches this, the agent
    /// stops. 0 disables the cap. Defaults to 50000.
    #[serde(default = "default_agentic_rag_max_chunk_chars")]
    pub agentic_rag_max_chunk_chars: u32,
    /// Custom OpenAI-compatible endpoint (Ollama, LM Studio, OpenRouter, Azure,
    /// vLLM, etc.). Honored only when `provider == "openai"`. `None` / blank →
    /// the OpenAI client falls back to `https://api.openai.com/v1/chat/completions`.
    /// Accepts either the base form (`…/v1`) or the full `…/v1/chat/completions`
    /// URL — normalization is centralized in `llm::openai::chat_url`.
    #[serde(default)]
    pub openai_base_url: Option<String>,
    /// When true, send `tool_choice: "required"` even for custom OpenAI base URLs.
    /// Official OpenAI always gets "required"; custom endpoints default to "auto"
    /// because some don't support it. Enable this if your custom endpoint supports
    /// forced tool use (OpenRouter, vLLM, Together, etc.). Defaults to false.
    #[serde(default)]
    pub openai_force_tool_use: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "google".to_owned(),
            rerank_model: "gemini-3.1-flash-lite".to_owned(),
            api_keys: Vec::new(),
            rerank_min_prune_lines: default_min_prune_lines(),
            use_structured_output: default_use_structured_output(),
            agentic_rag: false,
            agentic_rag_max_turns: default_agentic_rag_max_turns(),
            agentic_rag_max_chunk_chars: default_agentic_rag_max_chunk_chars(),
            openai_base_url: None,
            openai_force_tool_use: false,
        }
    }
}

fn default_mcp_index_wait_secs() -> u64 {
    50
}

fn default_mcp_stale_after_days() -> u64 {
    7
}

fn default_enabled_mcp_tools() -> Vec<String> {
    vec![
        "codebase-retrieval".to_string(),
        "file-retrieval".to_string(),
    ]
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
    /// `<data_dir>/rocksdb/`. The embedding cache defaults to
    /// `<data_dir>/embeddings/` but can be relocated independently via
    /// `embeddings_dir`. `settings.json` itself ALWAYS lives at
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
    /// Location of the content-addressed embedding cache root. The cache is
    /// keyed by `md5(text) + model` (NOT by repo), is concurrency-safe (atomic
    /// tempfile+rename writes), and therefore can be SHARED across multiple
    /// instances so identical chunks are embedded once — unlike RocksDB, which
    /// needs per-instance isolation.
    ///
    /// `None` means "use the builtin default" —
    /// `~/.vibervn/context-engine/embeddings`, anchored to home (NOT to
    /// `data_dir`) so multiple instances with different `--data-dir` values
    /// share ONE cache by default.
    /// Boot precedence: CLI flag > env `CONTEXT_ENGINE_EMBEDDINGS_DIR` >
    /// `Settings.embeddings_dir` > `~/.vibervn/context-engine/embeddings`.
    /// Like `data_dir`, this is boot-frozen: a PUT change persists for the next
    /// launch only.
    #[serde(default)]
    pub embeddings_dir: Option<PathBuf>,
    /// Which MCP tools are exposed to clients. Tools not in this list are hidden
    /// from `list_tools` and reject calls with a "tool disabled" error.
    #[serde(default = "default_enabled_mcp_tools")]
    pub enabled_mcp_tools: Vec<String>,
    /// Extra file extensions (without leading dot, lowercase) to index beyond the
    /// built-in `CODE_EXTENSIONS` list. E.g. `["prisma", "zig", "nim"]`.
    #[serde(default)]
    pub custom_extensions: Vec<String>,
    /// Filenames to skip during indexing (case-sensitive, filename-only match).
    #[serde(default = "default_index_ignore_filenames")]
    pub index_ignore_filenames: Vec<String>,
    /// Per-repo on-disk index generation counter, keyed by the **normalized** repo
    /// path (see `store::normalize_repo_path`). SERVER-OWNED: `put_config` ignores
    /// any client-sent value and preserves what is on disk, so the UI's
    /// delete-then-PUT-config flow ("Xóa repo") cannot clobber a bump made by the
    /// delete handler.
    ///
    /// Semantics: a repo absent from this map (or mapped to 0) uses the legacy path
    /// `<data_dir>/rocksdb/<name>`. After each repo/index delete the counter is
    /// incremented and persisted, and the next index lands on
    /// `<data_dir>/rocksdb/<gen>/<name>` — a fresh directory the just-deleted (and
    /// possibly still-draining) RocksDB handle never touched. Entries persist even
    /// after a repo is removed from `repos`, so re-adding it keeps the higher
    /// generation instead of resetting to 0 and racing the old LOCK again.
    #[serde(default)]
    pub repo_generations: HashMap<String, u32>,
    /// Stable per-machine identifier used to dedup payment + free-trial flows
    /// against a single user (one machine = one user). Computed once on first
    /// boot via `ensure_machine_id` and persisted; never recomputed at runtime.
    ///
    /// Seed value: `sha256(MACHINE_ID_SALT ‖ \0 ‖ hardware_uid)` as hex when
    /// `machine_uid::get()` succeeds. This intentionally matches the legacy
    /// formula used by the free-trial claim flow before persistence — old
    /// claims tied to a hardware-derived id keep matching after the upgrade.
    /// On the rare host where `machine_uid::get()` fails we fall back to a
    /// random UUIDv4. The fallback is only "safe" because the result is
    /// persisted: every subsequent run reads the same value, so idempotency
    /// (one machine → one user) holds across restarts.
    ///
    /// `None` on the in-memory struct only ever occurs *during boot* between
    /// `ensure_dir_and_load` and `ensure_machine_id`. After boot the field is
    /// always `Some` — handlers can `unwrap_or_default` defensively but should
    /// never see empty.
    #[serde(default)]
    pub machine_id: Option<String>,
}

/// Salt mixed into the machine-id hash. A fixed compile-in constant: it must
/// stay byte-identical across versions/restarts so the seed value computed by
/// `ensure_machine_id` matches what the legacy free-trial claim flow used to
/// compute on the fly. Changing this breaks every existing free-trial claim.
pub const MACHINE_ID_SALT: &str = "vibervn-context-engine::free-trial::v1";

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
            embeddings_dir: None,
            enabled_mcp_tools: default_enabled_mcp_tools(),
            custom_extensions: Vec::new(),
            index_ignore_filenames: default_index_ignore_filenames(),
            repo_generations: HashMap::new(),
            machine_id: None,
        }
    }
}

impl Settings {
    /// Generation counter for `repo` (0 if never deleted). The lookup key is the
    /// normalized repo path so it matches how `repo_dbs`, statuses, and the delete
    /// handler key the same repo. Generation 0 → legacy `<data_dir>/rocksdb/<name>`
    /// path; ≥ 1 → `<data_dir>/rocksdb/<gen>/<name>`.
    pub fn repo_generation(&self, repo: &str) -> u32 {
        self.repo_generations
            .get(&crate::store::normalize_repo_path(repo))
            .copied()
            .unwrap_or(0)
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

/// Return the default embedding-cache root under `home_dir`
/// (`~/.vibervn/context-engine/embeddings`).
///
/// Used as the lowest-precedence fallback in boot resolution when no CLI flag,
/// env var, or persisted `Settings.embeddings_dir` is set. Anchored to
/// `home_dir` (NOT the resolved `data_dir`) on purpose: the content-addressed
/// cache is concurrency-safe and meant to be shared, so multiple instances
/// running with different `--data-dir` values share ONE cache by default —
/// identical chunks are embedded once. A pure default install (no flags) still
/// lands at `~/.vibervn/context-engine/embeddings`, byte-identical to the
/// historical layout, because `default_data_dir(home)` is the same base.
pub fn default_embeddings_dir(home_dir: &Path) -> PathBuf {
    default_data_dir(home_dir).join("embeddings")
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

    // Normalize repo paths (OS-native separators + case-fold on Windows) and
    // deduplicate. Handles pre-existing mixed-case entries from before the
    // normalization fix — e.g. both "D:\Projects\Foo" and "d:\projects\foo" collapse
    // to a single entry. Persist the cleaned list so duplicates don't reappear.
    let mut settings = settings;
    let original_repos = settings.repos.clone();
    {
        let mut seen = std::collections::HashSet::new();
        settings.repos = settings
            .repos
            .iter()
            .map(|r| crate::store::normalize_repo_path(r))
            .filter(|r| seen.insert(r.clone()))
            .collect();
    }
    if settings.repos != original_repos {
        let _ = write_settings_atomic(&path, &settings);
    }

    Ok(settings)
}

/// Ensure `settings.machine_id` is `Some(...)` and persisted on disk. Called
/// once at boot, after `ensure_dir_and_load`. Mutates `settings` in place.
///
/// First boot (or upgrades from a settings file written before this field
/// existed): compute `sha256(MACHINE_ID_SALT ‖ \0 ‖ hardware_uid)` as hex,
/// matching the legacy free-trial claim formula so machines that already
/// claimed pick the SAME id and continue to dedup against their existing user.
/// If `machine_uid::get()` fails (rare hosts where no hardware uid is
/// reachable), fall back to a random UUIDv4. The fallback is only sound
/// BECAUSE we persist it immediately — the next boot reads the same value, so
/// "one machine = one user" still holds across restarts.
///
/// Subsequent boots: field already populated → no-op (don't recompute, don't
/// rewrite).
pub fn ensure_machine_id(home_dir: &Path, settings: &mut Settings) -> Result<(), ConfigError> {
    if settings
        .machine_id
        .as_deref()
        .map(|s| !s.is_empty())
        .unwrap_or(false)
    {
        return Ok(());
    }

    let id = compute_seed_machine_id();
    settings.machine_id = Some(id);
    let path = config_path(home_dir);
    write_settings_atomic(&path, settings)
}

/// Compute the machine-id seed used by `ensure_machine_id`. Public-in-crate so
/// tests can assert the legacy formula is preserved.
fn compute_seed_machine_id() -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(MACHINE_ID_SALT.as_bytes());
    hasher.update(b"\x00");
    match machine_uid::get() {
        Ok(uid) => {
            hasher.update(uid.as_bytes());
        }
        Err(_) => {
            // Fallback only — not the common path. Mix high-entropy host signals
            // (system time + process id + a fresh allocation address) into the
            // hash so the result is unique per first-boot. Acceptable here
            // because we persist it on the same call: the next boot reads the
            // same value, preserving idempotency.
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            hasher.update(b"fallback-v1\x00");
            hasher.update(nanos.to_le_bytes());
            hasher.update(std::process::id().to_le_bytes());
            let probe: Box<u8> = Box::new(0);
            hasher.update((Box::as_ref(&probe) as *const u8 as usize).to_le_bytes());
        }
    }
    let bytes = hasher.finalize();
    let mut s = String::with_capacity(bytes.len() * 2);
    use std::fmt::Write as _;
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
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

    /// Default Settings carries `data_dir == None` and `embeddings_dir == None`,
    /// signalling "use the builtin defaults at boot". An explicit `Some(path)`
    /// represents a frozen user choice and changes boot resolution in main.rs.
    #[test]
    fn test_data_dir_default_is_none() {
        let s = Settings::default();
        assert!(s.data_dir.is_none(), "default data_dir must be None");
        assert!(s.embeddings_dir.is_none(), "default embeddings_dir must be None");
        assert_eq!(s.version, CURRENT_VERSION);
    }

    /// v1 → … → CURRENT migration starting from a v1 file: stamps explicit
    /// `null` for every field added since v1 (`data_dir`, `embeddings_dir`) and
    /// advances the version, so an old binary refuses the file (VersionTooNew)
    /// instead of silently dropping fields on the next save.
    #[test]
    fn test_v1_to_v2_migration_stamps_null_data_dir() {
        let home = TempDir::new().expect("tempdir");
        let path = config_path(home.path());
        fs::create_dir_all(path.parent().expect("has parent")).expect("create dirs");

        // Write a valid v1 settings.json (no data_dir / embeddings_dir fields).
        let v1 = r#"{
            "version": 1,
            "repos": [],
            "embedding": {"provider":"voyage","model":"voyage-4-lite","api_keys":[]},
            "llm": {"provider":"google","rerank_model":"gemini-3.1-flash-lite","api_keys":[]}
        }"#;
        fs::write(&path, v1).expect("write v1 settings.json");

        // Load: should run the full migration chain (v1→v2→v3).
        let loaded = ensure_dir_and_load(home.path()).expect("load v1");
        assert_eq!(loaded.version, CURRENT_VERSION);
        assert!(loaded.data_dir.is_none(), "data_dir should be None after migration");
        assert!(loaded.embeddings_dir.is_none(), "embeddings_dir should be None after migration");

        // The on-disk file must now report CURRENT_VERSION with explicit null
        // fields — the tripwire that prevents an older binary from silently
        // re-reading and re-saving without the new fields.
        let raw = fs::read_to_string(&path).expect("re-read");
        let v: Value = serde_json::from_str(&raw).expect("parse re-read");
        assert_eq!(v.get("version").and_then(|x| x.as_u64()), Some(CURRENT_VERSION as u64));
        assert!(
            v.get("data_dir").map(|x| x.is_null()).unwrap_or(false),
            "on-disk data_dir should be explicit null after migration, got: {:?}",
            v.get("data_dir")
        );
        assert!(
            v.get("embeddings_dir").map(|x| x.is_null()).unwrap_or(false),
            "on-disk embeddings_dir should be explicit null after migration, got: {:?}",
            v.get("embeddings_dir")
        );
    }

    /// v2 → v3 migration in isolation: a v2 file (already has `data_dir`, lacks
    /// `embeddings_dir`) gains an explicit `embeddings_dir: null` and bumps to v3.
    #[test]
    fn test_v2_to_v3_migration_stamps_null_embeddings_dir() {
        let home = TempDir::new().expect("tempdir");
        let path = config_path(home.path());
        fs::create_dir_all(path.parent().expect("has parent")).expect("create dirs");

        // Valid v2 file: data_dir present (explicit value), no embeddings_dir.
        let v2 = r#"{
            "version": 2,
            "repos": [],
            "embedding": {"provider":"voyage","model":"voyage-4-lite","api_keys":[]},
            "llm": {"provider":"google","rerank_model":"gemini-3.1-flash-lite","api_keys":[]},
            "data_dir": "/var/data/instance-A"
        }"#;
        fs::write(&path, v2).expect("write v2 settings.json");

        let loaded = ensure_dir_and_load(home.path()).expect("load v2");
        assert_eq!(loaded.version, CURRENT_VERSION);
        // v2→v3 must NOT disturb the existing data_dir value.
        assert_eq!(loaded.data_dir, Some(PathBuf::from("/var/data/instance-A")));
        assert!(loaded.embeddings_dir.is_none(), "embeddings_dir should be None (null) after v2→v3");

        let raw = fs::read_to_string(&path).expect("re-read");
        let v: Value = serde_json::from_str(&raw).expect("parse re-read");
        assert_eq!(v.get("version").and_then(|x| x.as_u64()), Some(CURRENT_VERSION as u64));
        assert!(
            v.get("embeddings_dir").map(|x| x.is_null()).unwrap_or(false),
            "on-disk embeddings_dir should be explicit null after v2→v3, got: {:?}",
            v.get("embeddings_dir")
        );
    }

    /// Round-trip: explicit `data_dir` and `embeddings_dir` values survive
    /// serialize+deserialize and are preserved on subsequent loads.
    #[test]
    fn test_data_dir_explicit_value_round_trips() {
        let home = TempDir::new().expect("tempdir");
        let path = config_path(home.path());
        fs::create_dir_all(path.parent().expect("has parent")).expect("create dirs");

        let custom = PathBuf::from("/var/data/instance-A");
        let custom_emb = PathBuf::from("/shared/embeddings");
        let s = Settings {
            data_dir: Some(custom.clone()),
            embeddings_dir: Some(custom_emb.clone()),
            ..Settings::default()
        };
        write_settings_atomic(&path, &s).expect("write");

        let loaded = ensure_dir_and_load(home.path()).expect("load");
        assert_eq!(loaded.data_dir, Some(custom));
        assert_eq!(loaded.embeddings_dir, Some(custom_emb));
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

    /// `default_embeddings_dir` is anchored to `home_dir`
    /// (`~/.vibervn/context-engine/embeddings`), NOT to the resolved data_dir,
    /// so instances with different data dirs share one cache by default. A pure
    /// default install still matches the historical layout.
    #[test]
    fn test_default_embeddings_dir_layout() {
        let home = TempDir::new().expect("tempdir");
        let ed = default_embeddings_dir(home.path());
        assert_eq!(
            ed,
            home.path().join(".vibervn").join("context-engine").join("embeddings"),
            "default embeddings_dir must match historical layout for byte-identical default install"
        );
    }

    #[test]
    fn test_v5_to_v6_migration_injects_ignore_filenames() {
        let home = TempDir::new().expect("tempdir");
        let path = config_path(home.path());
        fs::create_dir_all(path.parent().expect("has parent")).expect("create dirs");

        let v5 = r#"{
            "version": 5,
            "repos": [],
            "embedding": {"provider":"voyage","model":"voyage-4-lite","api_keys":[],"embed_concurrency":16},
            "llm": {"provider":"google","rerank_model":"gemini-3.1-flash-lite","api_keys":[]},
            "data_dir": null,
            "embeddings_dir": null,
            "enabled_mcp_tools": ["codebase-retrieval","file-retrieval"],
            "custom_extensions": []
        }"#;
        fs::write(&path, v5).expect("write v5 settings.json");

        let loaded = ensure_dir_and_load(home.path()).expect("load v5");
        assert_eq!(loaded.version, CURRENT_VERSION);
        assert_eq!(
            loaded.index_ignore_filenames,
            vec!["CLAUDE.md".to_string(), "AGENTS.md".to_string()],
            "migration must inject default ignore filenames"
        );
    }

    #[test]
    fn test_v6_explicit_empty_ignore_not_clobbered() {
        let home = TempDir::new().expect("tempdir");
        let path = config_path(home.path());
        fs::create_dir_all(path.parent().expect("has parent")).expect("create dirs");

        let v6 = r#"{
            "version": 6,
            "repos": [],
            "embedding": {"provider":"voyage","model":"voyage-4-lite","api_keys":[],"embed_concurrency":16},
            "llm": {"provider":"google","rerank_model":"gemini-3.1-flash-lite","api_keys":[]},
            "data_dir": null,
            "embeddings_dir": null,
            "enabled_mcp_tools": ["codebase-retrieval","file-retrieval"],
            "custom_extensions": [],
            "index_ignore_filenames": []
        }"#;
        fs::write(&path, v6).expect("write v6 settings.json");

        let loaded = ensure_dir_and_load(home.path()).expect("load v6");
        assert_eq!(loaded.version, CURRENT_VERSION);
        assert!(
            loaded.index_ignore_filenames.is_empty(),
            "explicit empty list must not be overwritten by serde default; got: {:?}",
            loaded.index_ignore_filenames
        );
    }

    /// Backward-compat: an existing user's settings.json (already at v6, the
    /// current schema) whose `llm` block predates the Agentic RAG fields must
    /// load WITHOUT a deserialize error and fill both fields from serde defaults
    /// (agentic_rag=false, agentic_rag_max_turns=3). Adding these fields was an
    /// additive-with-defaults change, so no version bump/migration was needed —
    /// this test pins that the defaults actually apply on an old-shaped file.
    #[test]
    fn test_v6_missing_agentic_rag_fields_default_cleanly() {
        let home = TempDir::new().expect("tempdir");
        let path = config_path(home.path());
        fs::create_dir_all(path.parent().expect("has parent")).expect("create dirs");

        // Note: `llm` has provider/rerank_model/api_keys only — NO agentic_rag,
        // NO agentic_rag_max_turns, NO rerank_min_prune_lines, NO
        // use_structured_output. Exactly an upgraded-from-older-build file.
        let v6 = r#"{
            "version": 6,
            "repos": [],
            "embedding": {"provider":"voyage","model":"voyage-4-lite","api_keys":[],"embed_concurrency":16},
            "llm": {"provider":"google","rerank_model":"gemini-3.1-flash-lite","api_keys":[]},
            "data_dir": null,
            "embeddings_dir": null,
            "enabled_mcp_tools": ["codebase-retrieval","file-retrieval"],
            "custom_extensions": [],
            "index_ignore_filenames": []
        }"#;
        fs::write(&path, v6).expect("write v6 settings.json");

        // Must NOT error on the missing fields.
        let loaded = ensure_dir_and_load(home.path()).expect("load v6 missing agentic fields");
        assert_eq!(loaded.version, CURRENT_VERSION);
        assert!(!loaded.llm.agentic_rag, "agentic_rag must default to false on old files");
        assert_eq!(
            loaded.llm.agentic_rag_max_turns, 9,
            "agentic_rag_max_turns must default to 9 on old files"
        );
        assert_eq!(
            loaded.llm.agentic_rag_max_chunk_chars, 50_000,
            "agentic_rag_max_chunk_chars must default to 50000 on old files"
        );
    }

    /// Direct deserialization guard: parsing an `LlmConfig` from JSON with the
    /// agentic fields absent succeeds and yields the documented defaults. This
    /// is the narrowest possible proof, independent of the file-load path.
    #[test]
    fn test_llm_config_deserializes_without_agentic_fields() {
        let json = r#"{"provider":"google","rerank_model":"gemini-3.1-flash-lite","api_keys":["k"]}"#;
        let cfg: LlmConfig = serde_json::from_str(json).expect("deserialize old llm block");
        assert!(!cfg.agentic_rag);
        assert_eq!(cfg.agentic_rag_max_turns, 9);
        assert_eq!(cfg.agentic_rag_max_chunk_chars, 50_000);
    }

    /// Backward-compat for `openai_base_url`: an existing settings.json whose
    /// `llm` block predates the field must deserialize cleanly with the field
    /// defaulted to `None`. Mirrors `test_v6_missing_agentic_rag_fields_default_cleanly`.
    /// No version bump was needed (additive `Option<String>` with serde default),
    /// so this test pins that the additive contract still holds.
    #[test]
    fn test_llm_config_deserializes_without_openai_base_url() {
        let json = r#"{"provider":"openai","rerank_model":"gpt-4o-mini","api_keys":["k"]}"#;
        let cfg: LlmConfig = serde_json::from_str(json).expect("deserialize old llm block");
        assert!(cfg.openai_base_url.is_none(), "openai_base_url must default to None on old files");
    }

    /// Round-trip: an explicit `openai_base_url` survives serialize → atomic
    /// write → migration-aware reload. Uses the same write/load path as the
    /// running server so we exercise the real persistence logic, not just
    /// `serde_json::from_str`.
    #[test]
    fn test_llm_config_round_trips_openai_base_url() {
        let home = TempDir::new().expect("tempdir");
        let path = config_path(home.path());
        fs::create_dir_all(path.parent().expect("has parent")).expect("create dirs");

        let s = Settings {
            llm: LlmConfig {
                provider: "openai".to_owned(),
                rerank_model: "gpt-4o-mini".to_owned(),
                api_keys: vec!["k".to_owned()],
                openai_base_url: Some("http://localhost:11434/v1".to_owned()),
                ..LlmConfig::default()
            },
            ..Settings::default()
        };
        write_settings_atomic(&path, &s).expect("write");

        let loaded = ensure_dir_and_load(home.path()).expect("load");
        assert_eq!(
            loaded.llm.openai_base_url.as_deref(),
            Some("http://localhost:11434/v1"),
            "openai_base_url must round-trip through write+load"
        );
        assert_eq!(loaded.version, CURRENT_VERSION);
    }

    #[test]
    fn test_v6_to_v7_migration_stamps_null_voyage_base_url() {
        let home = TempDir::new().expect("tempdir");
        let path = config_path(home.path());
        fs::create_dir_all(path.parent().expect("has parent")).expect("create dirs");

        let v6 = r#"{
            "version": 6,
            "repos": [],
            "embedding": {"provider":"voyage","model":"voyage-4-lite","api_keys":[],"embed_concurrency":16},
            "llm": {"provider":"google","rerank_model":"gemini-3.1-flash-lite","api_keys":[]},
            "data_dir": null,
            "embeddings_dir": null,
            "enabled_mcp_tools": ["codebase-retrieval","file-retrieval"],
            "custom_extensions": [],
            "index_ignore_filenames": ["CLAUDE.md","AGENTS.md"]
        }"#;
        fs::write(&path, v6).expect("write v6 settings.json");

        let loaded = ensure_dir_and_load(home.path()).expect("load v6");
        assert_eq!(loaded.version, CURRENT_VERSION);
        assert!(
            loaded.embedding.voyage_base_url.is_none(),
            "voyage_base_url must default to None after v6→v7 migration"
        );

        let raw = fs::read_to_string(&path).expect("re-read");
        let v: Value = serde_json::from_str(&raw).expect("parse re-read");
        assert_eq!(v.get("version").and_then(|x| x.as_u64()), Some(CURRENT_VERSION as u64));
        let emb = v.get("embedding").expect("embedding key");
        assert!(
            emb.get("voyage_base_url").map(|x| x.is_null()).unwrap_or(false),
            "on-disk voyage_base_url should be explicit null after migration, got: {:?}",
            emb.get("voyage_base_url")
        );
    }

    #[test]
    fn test_v7_to_v8_migration_stamps_empty_repo_generations() {
        let home = TempDir::new().expect("tempdir");
        let path = config_path(home.path());
        fs::create_dir_all(path.parent().expect("has parent")).expect("create dirs");

        // A v7 file has no `repo_generations` key. After migration every existing
        // repo must read as generation 0 (legacy path preserved — no orphaning).
        let v7 = r#"{
            "version": 7,
            "repos": ["D:\\projects\\foo"],
            "embedding": {"provider":"voyage","model":"voyage-4-lite","api_keys":[],"embed_concurrency":16,"voyage_base_url":null},
            "llm": {"provider":"google","rerank_model":"gemini-3.1-flash-lite","api_keys":[]},
            "data_dir": null,
            "embeddings_dir": null,
            "enabled_mcp_tools": ["codebase-retrieval","file-retrieval"],
            "custom_extensions": [],
            "index_ignore_filenames": ["CLAUDE.md","AGENTS.md"]
        }"#;
        fs::write(&path, v7).expect("write v7 settings.json");

        let loaded = ensure_dir_and_load(home.path()).expect("load v7");
        assert_eq!(loaded.version, CURRENT_VERSION);
        assert!(
            loaded.repo_generations.is_empty(),
            "repo_generations must default to empty after v7→v8 migration"
        );
        // An unlisted repo (and every existing one) reads as generation 0.
        assert_eq!(
            loaded.repo_generation("D:\\projects\\foo"),
            0,
            "existing repo must be generation 0 so its on-disk index is not orphaned"
        );

        let raw = fs::read_to_string(&path).expect("re-read");
        let v: Value = serde_json::from_str(&raw).expect("parse re-read");
        assert_eq!(v.get("version").and_then(|x| x.as_u64()), Some(CURRENT_VERSION as u64));
        assert!(
            v.get("repo_generations").map(|x| x.is_object()).unwrap_or(false),
            "on-disk repo_generations should be an explicit object after migration, got: {:?}",
            v.get("repo_generations")
        );
    }

    #[test]
    fn test_embedding_config_deserializes_without_voyage_base_url() {
        let json = r#"{"provider":"voyage","model":"voyage-4-lite","api_keys":["k"],"embed_concurrency":16}"#;
        let cfg: EmbeddingConfig = serde_json::from_str(json).expect("deserialize old embedding block");
        assert!(cfg.voyage_base_url.is_none(), "voyage_base_url must default to None on old files");
    }

    #[test]
    fn test_embedding_config_round_trips_voyage_base_url() {
        let home = TempDir::new().expect("tempdir");
        let path = config_path(home.path());
        fs::create_dir_all(path.parent().expect("has parent")).expect("create dirs");

        let s = Settings {
            embedding: EmbeddingConfig {
                voyage_base_url: Some("https://my-proxy.com/v1".to_owned()),
                ..EmbeddingConfig::default()
            },
            ..Settings::default()
        };
        write_settings_atomic(&path, &s).expect("write");

        let loaded = ensure_dir_and_load(home.path()).expect("load");
        assert_eq!(
            loaded.embedding.voyage_base_url.as_deref(),
            Some("https://my-proxy.com/v1"),
            "voyage_base_url must round-trip through write+load"
        );
        assert_eq!(loaded.version, CURRENT_VERSION);
    }

    /// `ensure_machine_id` populates the field on first call, persists it to
    /// disk, and is a no-op on the next call (same value, no rewrite).
    #[test]
    fn ensure_machine_id_persists_and_is_idempotent() {
        let home = TempDir::new().expect("tempdir");
        let mut s = ensure_dir_and_load(home.path()).expect("load default");
        // Default settings have no machine_id yet (file just bootstrapped, but
        // the on-disk default does not include this field).
        assert!(
            s.machine_id.as_deref().map(str::is_empty).unwrap_or(true),
            "fresh settings should have no machine_id"
        );

        ensure_machine_id(home.path(), &mut s).expect("first ensure");
        let id = s.machine_id.clone().expect("populated");
        assert!(!id.is_empty());

        // Reload from disk — value persisted.
        let reloaded = ensure_dir_and_load(home.path()).expect("reload");
        assert_eq!(
            reloaded.machine_id.as_deref(),
            Some(id.as_str()),
            "machine_id must persist across reload"
        );

        // Second ensure on the same in-memory struct is a no-op.
        let mut s2 = reloaded;
        ensure_machine_id(home.path(), &mut s2).expect("second ensure");
        assert_eq!(s2.machine_id.as_deref(), Some(id.as_str()));
    }
}
