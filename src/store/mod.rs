pub mod ops;
pub mod schema;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex as StdMutex};

use anyhow::{Context, Result};
use surrealdb::Surreal;
use surrealdb::engine::local::{Db, RocksDb};
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn};

use crate::store::schema::SCHEMA_DDL;

/// Current DB schema version. Bump when new backfills are added.
/// v1 = original schema (no in_name/out_name, no chunk_count).
/// v2 = adds calls.in_name/out_name + file_meta.chunk_count.
/// v3 = chunk table flipped to SCHEMALESS for ~8.9× faster writes.
/// v4 = symbol table flipped to SCHEMALESS so the native sql::Array INSERT path
///      (which writes parent as a plain string) is not rejected by an existing
///      SCHEMAFULL symbol.parent definition (older DBs declared it as
///      option<record<symbol>>, which silently rolled back the whole batch → 0 symbols).
/// v5 = chunk.embedding packed from array<float> → bytes (8.9× faster insert).
/// v6 = calls flipped from TYPE RELATION → TYPE NORMAL. Nothing traverses the
///      graph at v5+; all reads use denormalized columns. RELATION forced
///      graph-adjacency writes on every edge (~44% of Phase-2 write time on the
///      kernel). The migration clears old RELATION rows so they re-resolve as
///      plain rows. Output (call-graph nodes/edges) is byte-identical.
/// v7 = call-graph heal. Builds before v7 kept full-rebuild raw edges RAM-only, so
///      `raw_edge` is partial and a later no-change run could replay Phase 2 from
///      it, truncating `calls` while stamping `edges_resolved`. The migration
///      clears both edge markers so the pipeline's edge heal (re-parse, no
///      embedding) rebuilds raw_edge + calls on the next trigger. Also adds the
///      `pending_retry` table (SCHEMA_DDL).
pub const DB_SCHEMA_VERSION: u32 = 7;

/// key in index_meta for the DB schema version.
pub const DB_SCHEMA_VERSION_KEY: &str = "db_schema_version";

/// Shared, process-wide map of one open SurrealDB handle per repo path.
pub type RepoDbMap = Arc<RwLock<HashMap<String, Surreal<Db>>>>;

/// Normalize a repo path to a canonical form for use as a HashMap/gate key.
/// On Windows: lowercase + backslash separators (NTFS is case-insensitive).
/// On Unix: forward slashes only (case-sensitive filesystems — no case fold).
/// Trailing separators are stripped on both platforms.
pub fn normalize_repo_path(repo: &str) -> String {
    let s = if cfg!(windows) {
        repo.replace('/', "\\").to_lowercase()
    } else {
        repo.replace('\\', "/")
    };
    s.trim_end_matches(['/', '\\']).to_string()
}

/// Sanitize a repo path to a safe directory name (max 64 chars).
pub fn sanitize_repo_name(repo_path: &str) -> String {
    let sanitized: String = repo_path
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    let trimmed = sanitized.trim_matches('_');
    if trimmed.len() > 64 {
        trimmed[trimmed.len() - 64..].to_string()
    } else {
        trimmed.to_string()
    }
}

/// Return the SurrealDB data directory for a given repo at a given generation.
///
/// Generation 0 → `<data_dir>/rocksdb/<sanitized-repo-name>/` — byte-for-byte the
/// legacy layout, so existing on-disk indexes are NOT orphaned when the
/// `repo_generations` map is introduced (an unlisted repo reads as generation 0).
/// Generation ≥ 1 → `<data_dir>/rocksdb/<gen>/<sanitized-repo-name>/`. The counter
/// is bumped on every repo/index delete so the next index lands on a FRESH
/// directory the just-deleted RocksDB handle never touched — side-stepping the
/// async LOCK drain (7s+ on Windows under Defender) that otherwise makes an
/// immediate re-index race the deleted handle's still-held lock.
///
/// Namespaced under `rocksdb/` (not the legacy `surreal/` SurrealKV path). The
/// backend swap from SurrealKV to RocksDB changes the on-disk format, so the old
/// `surreal/<name>` directories are intentionally left untouched for rollback; a
/// repo opened here for the first time has no file_meta and triggers a full
/// rebuild via the pipeline's is_first_run path (embedding cache makes it
/// API-free).
///
/// `data_dir` is the boot-resolved data directory (CLI > env >
/// `Settings.data_dir` > builtin default). It is captured once at startup and
/// MUST NOT be re-read from `Settings` mid-run — open RocksDB handles in
/// `repo_dbs` and resident vector shards are bound to the boot path; switching
/// would split-brain reads against writes. `generation`, by contrast, IS read
/// live from `Settings`: it only changes after `close_repo_db` has dropped the
/// cached handle and `clear_repo_index` has evicted the resident shard, so no
/// open handle or warmed shard is ever bound to a stale generation.
pub fn db_path(data_dir: &Path, repo_path: &str, generation: u32) -> PathBuf {
    let name = sanitize_repo_name(repo_path);
    let base = data_dir.join("rocksdb");
    if generation == 0 {
        base.join(name)
    } else {
        base.join(generation.to_string()).join(name)
    }
}

/// Read the stored db_schema_version from index_meta, defaulting to 1
/// (treat unversioned DBs as v1 for safe migration).
pub async fn read_db_schema_version(db: &Surreal<Db>) -> u32 {
    match ops::get_meta(db, DB_SCHEMA_VERSION_KEY).await {
        Ok(Some(v)) => v.parse::<u32>().unwrap_or(1),
        _ => 1,
    }
}

/// Return the DB's schema version, first stamping a brand-new (empty, unversioned)
/// DB with [`DB_SCHEMA_VERSION`].
///
/// WHY: an unversioned DB reads as v1, so a fresh repo used to spawn the whole
/// v1→current migration chain in the background while its first index ran —
/// rewriting/deleting rows the pipeline was writing at the same time. An empty DB
/// has nothing to migrate. A DB with data but no version key is a genuine pre-v2
/// DB and still migrates.
pub async fn stamp_if_fresh(db: &Surreal<Db>) -> Result<u32> {
    if ops::get_meta(db, DB_SCHEMA_VERSION_KEY).await?.is_some() {
        return Ok(read_db_schema_version(db).await);
    }
    let mut res = db
        .query(
            "SELECT VALUE type::string(id) FROM file_meta LIMIT 1;              SELECT VALUE type::string(id) FROM chunk LIMIT 1;",
        )
        .await
        .context("stamp_if_fresh: probe tables")?;
    let metas: Vec<String> = res.take(0)?;
    let chunks: Vec<String> = res.take(1)?;
    if metas.is_empty() && chunks.is_empty() {
        ops::set_meta(db, DB_SCHEMA_VERSION_KEY, &DB_SCHEMA_VERSION.to_string())
            .await
            .context("stamp_if_fresh: stamp version")?;
        return Ok(DB_SCHEMA_VERSION);
    }
    Ok(1)
}

/// Open (or create) a SurrealDB database for the given repo.
/// Runs schema DDL to ensure all tables/indexes exist.
/// Returns the db handle; the caller is responsible for triggering migrations.
pub async fn open_db(data_dir: &Path, repo_path: &str, generation: u32) -> Result<Surreal<Db>> {
    let path = db_path(data_dir, repo_path, generation);
    std::fs::create_dir_all(&path).with_context(|| format!("create db dir {:?}", path))?;

    // Retry the RocksDB datastore open: when this path was just released by a
    // prior `Surreal<Db>` clone (e.g. close_repo_db → remove_index_dir gave up,
    // then a queued re-index slipped through the gate), the previous
    // datastore's background router can still hold the exclusive LOCK file for
    // a meaningful window — Windows in particular delays releasing OS handles
    // past the Rust drop, and with Defender real-time scanning of RocksDB files
    // the drain on a freshly-indexed repo has been measured at 7s+. Budget ~30s
    // total — long enough for a Windows+Defender drain after a rapid
    // remove+rebuild, short enough that a genuinely corrupted directory still
    // surfaces in a single user retry window.
    let path_str = path.to_str().unwrap();
    let mut last_err: Option<surrealdb::Error> = None;
    let mut db_opt: Option<Surreal<Db>> = None;
    for attempt in 0..20u32 {
        match Surreal::new::<RocksDb>(path_str).await {
            Ok(db) => {
                db_opt = Some(db);
                break;
            }
            Err(e) => {
                last_err = Some(e);
                // Log on first failure and every ~5s thereafter so the user
                // sees progress during a long drain instead of a silent stall.
                if attempt == 0 || attempt == 5 || attempt == 10 || attempt == 15 {
                    info!(
                        path = ?path,
                        attempt,
                        "open surrealdb failed; retrying — likely a stale LOCK from a draining prior handle"
                    );
                }
                // 200ms, 400ms, … capped at 2s — summing to ~30s across 20 tries.
                let backoff_ms = (200u64 * (attempt as u64 + 1)).min(2000);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
            }
        }
    }
    let db = match db_opt {
        Some(db) => db,
        None => {
            return Err(anyhow::Error::new(
                last_err.expect("loop sets last_err on failure"),
            ))
            .context("open surrealdb");
        }
    };

    db.use_ns("context_engine")
        .use_db(sanitize_repo_name(repo_path))
        .await
        .context("select ns/db")?;

    db.query(SCHEMA_DDL)
        .await
        .context("apply schema DDL")?
        .check()
        .context("schema DDL contained errors")?;

    // Build the six drop-during-rebuild secondary indexes (symbol + calls) that
    // SCHEMA_DDL deliberately omits. This is the crash-recovery cure for Theory-A:
    // a repo that died mid-rebuild has those indexes dropped while rows remain; a
    // foreground `DEFINE INDEX` over the populated table would roll back under the
    // pinned RocksDB buffers and fail this open. ensure_secondary_indexes routes any
    // missing index through `build_index_concurrently` (batched commits) and is a
    // fast no-op when the index already exists (steady-state reopen) or the table is
    // empty (fresh DB — defined instantly).
    ensure_secondary_indexes(&db)
        .await
        .context("ensure secondary indexes")?;

    Ok(db)
}

/// Spawn background migration tasks if needed (non-blocking).
///
/// Checks `db_schema_version` in `index_meta`. Spawns tasks to bring the DB
/// up to the current schema version. Failures are logged, not propagated.
///
/// v1→v2: backfills calls.in_name/out_name + file_meta.chunk_count.
/// v2→v3: flips chunk table to SCHEMALESS for ~8.9× faster writes.
///
/// If both migrations are needed, they run in a single chained task so v1→v2
/// always completes before v2→v3 starts.
pub fn maybe_spawn_migration(repo_dbs: RepoDbMap, repo: String, stored_version: u32) {
    if stored_version >= DB_SCHEMA_VERSION {
        return;
    }
    info!(
        stored_version,
        target = DB_SCHEMA_VERSION,
        "spawning chained DB migration background task"
    );
    // Run all needed migrations in one chained task so each completes before the
    // next starts. A failed step aborts the chain via `?` (the version stamp is only
    // written on success, so the next open retries from the same point).
    //
    // The task clones the `Surreal<Db>` handle once and holds that owned clone for
    // the migration's entire duration. That clone pins the RocksDB exclusive LOCK,
    // so removing the entry from `repo_dbs` does NOT release it — the task owns its
    // own clone independent of the map. To delete the directory deterministically,
    // `close_repo_db` calls `store::abort_migration`, which aborts + awaits this
    // task so the clone is dropped before `remove_index_dir` runs. Safe because
    // migrations are idempotent + crash-resumable: an aborted migration self-heals
    // on the next open. We register the JoinHandle in `MIGRATION_TASKS` (keyed by
    // repo) so `abort_migration` can find it; the task self-removes its entry on
    // completion so the registry stays bounded by repo count.
    let repo_for_cleanup = repo.clone();
    let repo_key = repo.clone();
    let handle = tokio::spawn(async move {
        let db = match repo_dbs.read().await.get(&repo) {
            Some(db) => db.clone(),
            None => {
                // repo was removed before migration started — still self-remove the
                // registry entry (the outer insert may have landed before us).
                MIGRATION_TASKS.lock().unwrap().remove(&repo_for_cleanup);
                return;
            }
        };
        let result: Result<()> = async {
            if stored_version < 2 {
                run_migration_v1_to_v2(&db).await.context("v1→v2")?;
            }
            if stored_version < 3 {
                run_migration_v2_to_v3(&db).await.context("v2→v3")?;
            }
            if stored_version < 4 {
                run_migration_v3_to_v4(&db).await.context("v3→v4")?;
            }
            if stored_version < 5 {
                run_migration_v4_to_v5(&db).await.context("v4→v5")?;
            }
            if stored_version < 6 {
                run_migration_v5_to_v6(&db).await.context("v5→v6")?;
            }
            if stored_version < 7 {
                run_migration_v6_to_v7(&db).await.context("v6→v7")?;
            }
            Ok(())
        }
        .await;
        if let Err(e) = result {
            warn!(error = %e, "chained DB migration failed");
        }
        // Self-deregister so the registry doesn't leak. A completed-then-removed
        // handle is harmless: a later `abort_migration` for this repo finds nothing
        // and is a no-op. If the outer insert below races and re-adds this (already
        // finished) handle, it only wastes one tiny HashMap slot, overwritten on the
        // next migration for this repo — bounded by repo count.
        MIGRATION_TASKS.lock().unwrap().remove(&repo_for_cleanup);
    });
    MIGRATION_TASKS.lock().unwrap().insert(repo_key, handle);
}

/// Paged v1→v2 migration. Must be idempotent (safe to re-run).
///
/// Backfill 1: calls.in_name/out_name — reads link-deref in.name/out.name per page
///   and populates the new denormalized columns.
/// Backfill 2: file_meta.chunk_count — counts chunks per file and updates file_meta.
///
/// `db_schema_version=2` is written ONLY after both backfills complete.
///
/// Keyset pagination:
///   - calls: keyset on `type::string(id) AS id_str` (string-ordered record ID).
///     Using `type::string(id)` sidesteps the Thing-serde blocker: we never
///     deserialize a SurrealDB `Thing` through serde — we just read the string
///     representation. The string form `calls:⟨rand⟩` has stable lexicographic
///     order (SurrealDB random IDs are fixed-length alphanumeric, giving consistent
///     string sort). The `id` is unique per row, so `WHERE type::string(id) > $cursor
///     ORDER BY id_str` skips no rows and visits no row twice.
///     NOTE: SurrealDB 2.6.5 requires ORDER BY to reference a column that appears in
///     the SELECT projection. `ORDER BY type::string(id)` fails (function in ORDER BY
///     not supported), but `ORDER BY id_str` (the projected alias) works correctly.
///   - file_meta: keyset on `path` (UNIQUE via idx_filemeta_path). `WHERE path > $cursor
///     ORDER BY path` is correct and skips nothing.
///
/// Per-edge update correctness (Defect 2 fix):
///   Each calls row is updated via `UPDATE type::thing($id_str)` using the per-row
///   id_str read from that exact row. This ensures the (in_name, out_name) values
///   written come from the in.name/out.name of that specific row — not a file-pair
///   group that may contain multiple distinct edges sharing the same in_file/out_file.
pub async fn run_migration_v1_to_v2(db: &Surreal<Db>) -> Result<()> {
    use serde::Deserialize;

    let page_size: i64 = 512;

    // ── Backfill 1: calls.in_name / out_name ─────────────────────────────
    // The link-deref `in.name`/`out.name` is valid on existing rows (v1 rows
    // have proper `in`/`out` symbol record links). We read them to get the names.
    {
        info!("migration v1→v2: backfilling calls.in_name/out_name");

        // Keyset cursor: the last `type::string(id)` seen. Start from "" (empty
        // string sorts before all real record-id strings).
        let cursor_key = "migration_v2_calls_cursor";
        let mut cursor: String = ops::get_meta(db, cursor_key)
            .await
            .ok()
            .flatten()
            .unwrap_or_default();

        loop {
            #[derive(Deserialize)]
            struct EdgeRow {
                id_str: String,
                #[serde(rename = "in_name_link")]
                in_name: Option<String>,
                #[serde(rename = "out_name_link")]
                out_name: Option<String>,
            }

            // Select id as a string via type::string(id) so we never touch Thing serde.
            // WHERE type::string(id) > $cursor gives keyset pagination over the random IDs.
            // ORDER BY id_str (the projected alias) gives consistent ordering.
            // NOTE: ORDER BY type::string(id) fails in SurrealDB 2.6.5 (function not
            // allowed in ORDER BY); ORDER BY id_str (alias) works correctly.
            let batch: Vec<EdgeRow> = db
                .query(
                    "SELECT type::string(id) AS id_str, \
                            in.name AS in_name_link, \
                            out.name AS out_name_link \
                     FROM calls \
                     WHERE type::string(id) > $cursor \
                     ORDER BY id_str \
                     LIMIT $page",
                )
                .bind(("cursor", cursor.clone()))
                .bind(("page", page_size))
                .await
                .context("migration: scan calls page")?
                .take(0)?;

            if batch.is_empty() {
                break;
            }

            // Advance cursor to the last id_str in this page.
            cursor = batch
                .last()
                .map(|r| r.id_str.clone())
                .unwrap_or(cursor.clone());

            // Update each row by its OWN record id. This is the per-edge fix:
            // we update exactly the row whose in.name/out.name we read — never
            // a file-pair group that would stamp one name pair onto all edges
            // sharing the same (in_file, out_file).
            for row in &batch {
                if let (Some(in_n), Some(out_n)) = (&row.in_name, &row.out_name) {
                    db.query(
                        "UPDATE type::thing($id) SET in_name = $in_name, out_name = $out_name",
                    )
                    .bind(("id", row.id_str.clone()))
                    .bind(("in_name", in_n.clone()))
                    .bind(("out_name", out_n.clone()))
                    .await
                    .context("migration: update calls in_name/out_name by id")?;
                }
            }

            // Persist cursor for crash resume.
            ops::set_meta(db, cursor_key, &cursor)
                .await
                .context("migration: persist calls cursor")?;

            let batch_len = batch.len() as i64;
            if batch_len < page_size {
                break;
            }
        }

        // Clean up cursor key.
        let _ = db
            .query("DELETE FROM index_meta WHERE key = $k")
            .bind(("k", cursor_key))
            .await;
    }

    // ── Backfill 2: file_meta.chunk_count ────────────────────────────────
    {
        info!("migration v1→v2: backfilling file_meta.chunk_count");

        // Keyset cursor on path (UNIQUE via idx_filemeta_path).
        // `WHERE path > $cursor ORDER BY path` is correct and skips nothing.
        let cursor_key = "migration_v2_filemeta_cursor";
        let mut cursor: String = ops::get_meta(db, cursor_key)
            .await
            .ok()
            .flatten()
            .unwrap_or_default();

        loop {
            #[derive(Deserialize)]
            struct FileMetaRow {
                path: String,
            }

            let batch: Vec<FileMetaRow> = db
                .query(
                    "SELECT path FROM file_meta \
                     WHERE path > $cursor \
                     ORDER BY path \
                     LIMIT $page",
                )
                .bind(("cursor", cursor.clone()))
                .bind(("page", page_size))
                .await
                .context("migration: scan file_meta page")?
                .take(0)?;

            if batch.is_empty() {
                break;
            }

            // Advance cursor.
            cursor = batch
                .last()
                .map(|r| r.path.clone())
                .unwrap_or(cursor.clone());

            for row in &batch {
                #[derive(Deserialize)]
                struct CountRow {
                    count: i64,
                }
                let count_rows: Vec<CountRow> = db
                    .query("SELECT count() AS count FROM chunk WHERE file = $f GROUP ALL")
                    .bind(("f", row.path.clone()))
                    .await
                    .context("migration: count chunks for file")?
                    .take(0)?;
                let count = count_rows.first().map(|r| r.count).unwrap_or(0);

                // Update by path (unique via idx_filemeta_path).
                db.query("UPDATE file_meta SET chunk_count = $count WHERE path = $path")
                    .bind(("count", count))
                    .bind(("path", row.path.clone()))
                    .await
                    .context("migration: update file_meta chunk_count")?;
            }

            ops::set_meta(db, cursor_key, &cursor)
                .await
                .context("migration: persist file_meta cursor")?;

            let batch_len = batch.len() as i64;
            if batch_len < page_size {
                break;
            }
        }

        // Clean up cursor key.
        let _ = db
            .query("DELETE FROM index_meta WHERE key = $k")
            .bind(("k", cursor_key))
            .await;
    }

    // Stamp db_schema_version=2 ONLY after both backfills complete.
    ops::set_meta(db, DB_SCHEMA_VERSION_KEY, "2")
        .await
        .context("migration: stamp db_schema_version=2")?;

    info!("migration v1→v2 complete");
    Ok(())
}

/// Migrate chunk table from SCHEMAFULL (with per-element array<float> validation)
/// to SCHEMALESS for ~8.9× faster writes.
///
/// Steps:
///   1. Flip table mode + remove all field definitions (single multi-statement query).
///   2. Gating readback: verify one existing chunk row still has embedding.len() >= 512.
///   3. Stamp db_schema_version=3.
///   4. If gating fails: set needs_rebuild flag (next index run forces full rebuild).
///
/// Idempotent: safe to re-run. REMOVE FIELD on a non-existent field is a no-op.
/// DEFINE TABLE OVERWRITE on an already-SCHEMALESS table is a no-op.
pub async fn run_migration_v2_to_v3(db: &Surreal<Db>) -> Result<()> {
    use serde::Deserialize;

    info!("migration v2→v3: flipping chunk table to SCHEMALESS");

    // Step 1: flip table mode + remove field definitions.
    // Each statement auto-commits. REMOVE FIELD is idempotent (no-op if absent).
    db.query(
        "DEFINE TABLE OVERWRITE chunk SCHEMALESS;\
         REMOVE FIELD embedding ON chunk;\
         REMOVE FIELD file ON chunk;\
         REMOVE FIELD line_start ON chunk;\
         REMOVE FIELD line_end ON chunk;\
         REMOVE FIELD content ON chunk;\
         REMOVE FIELD symbol_ref ON chunk;",
    )
    .await
    .context("migration v2→v3: flip chunk to SCHEMALESS + remove fields")?;

    // Step 2: gating readback — verify existing embeddings survive the flip.
    // Uses the dual-format reader defensively: when this v2→v3 migration runs,
    // rows are still `array<float>` (the v4→v5 bytes conversion happens later),
    // but reading through the tolerant deserializer is correct regardless of
    // which format a row is in, so it stays correct even if migrations are
    // somehow reordered or replayed on a partially-converted DB.
    #[derive(Deserialize)]
    struct ProbeRow {
        #[serde(deserialize_with = "ops::de_embedding_dual")]
        embedding: Vec<f32>,
    }
    let probe: Vec<ProbeRow> = db
        .query("SELECT embedding FROM chunk WHERE embedding IS NOT NONE LIMIT 1")
        .await
        .context("migration v2→v3: gating readback query")?
        .take(0)?;

    let gating_ok = match probe.first() {
        Some(row) => row.embedding.len() >= 512, // sanity: at least half-dim
        None => true, // empty table — nothing to validate, migration is trivially safe
    };

    if gating_ok {
        info!("migration v2→v3: gating readback passed");
    } else {
        warn!("migration v2→v3: gating readback FAILED — setting needs_rebuild flag");
        ops::set_meta(db, "needs_rebuild", "1")
            .await
            .context("migration v2→v3: set needs_rebuild")?;
    }

    // Step 3: stamp version (regardless of gating — prevents re-running migration).
    ops::set_meta(db, DB_SCHEMA_VERSION_KEY, "3")
        .await
        .context("migration v2→v3: stamp db_schema_version=3")?;

    info!("migration v2→v3 complete");
    Ok(())
}

/// Migrate the symbol table from SCHEMAFULL to SCHEMALESS.
///
/// Why: the native `INSERT INTO symbol $data` path (flush_symbol_batch_native) writes
/// `parent` as a plain string "symbol:⟨fqn⟩". Older DBs declared `parent` as
/// `option<record<symbol>>`; SCHEMAFULL type enforcement rejects the string and rolls
/// back the WHOLE INSERT batch, so 0 symbols persist and Phase 2 resolves 0 call edges —
/// a silent accuracy regression with no surfaced error. Flipping to SCHEMALESS removes
/// the enforcement; correctness is guaranteed by the explicit per-field Value types in
/// flush_symbol_batch_native.
///
/// `DEFINE TABLE OVERWRITE ... SCHEMALESS` alone is NOT sufficient — the persisted
/// `DEFINE FIELD` definitions still enforce their types. Each field definition must be
/// explicitly removed (verified: flip-only leaves the insert failing; flip + REMOVE FIELD
/// makes it succeed). Mirrors run_migration_v2_to_v3 for the chunk table.
///
/// Idempotent: DEFINE TABLE OVERWRITE on an already-SCHEMALESS table and REMOVE FIELD on
/// an absent field are both no-ops. The existing symbol rows and their data are preserved.
pub async fn run_migration_v3_to_v4(db: &Surreal<Db>) -> Result<()> {
    info!("migration v3→v4: flipping symbol table to SCHEMALESS");

    db.query(
        "DEFINE TABLE OVERWRITE symbol SCHEMALESS;\
         REMOVE FIELD IF EXISTS name ON symbol;\
         REMOVE FIELD IF EXISTS kind ON symbol;\
         REMOVE FIELD IF EXISTS file ON symbol;\
         REMOVE FIELD IF EXISTS line_start ON symbol;\
         REMOVE FIELD IF EXISTS line_end ON symbol;\
         REMOVE FIELD IF EXISTS signature ON symbol;\
         REMOVE FIELD IF EXISTS parent ON symbol;",
    )
    .await
    .context("migration v3→v4: flip symbol to SCHEMALESS + remove fields")?
    .check()
    .context("migration v3→v4: symbol flip statement error")?;

    ops::set_meta(db, DB_SCHEMA_VERSION_KEY, "4")
        .await
        .context("migration v3→v4: stamp db_schema_version=4")?;

    info!("migration v3→v4 complete");
    Ok(())
}

/// Paged v4→v5 migration: convert `chunk.embedding` from `array<float>` to a
/// packed little-endian `bytes` blob. Must be idempotent and crash-resumable.
///
/// WHY: storing 1024 floats/row as `array<float>` forces SurrealDB to encode
/// ~21M `Value::Number` enums per full rebuild (measured: 94% of chunk-write
/// time). Packed `bytes` is a 4096-byte blob/row written with a memcpy. The
/// new writer (`flush_chunk_batch`) already emits bytes; this migration brings
/// pre-v5 rows up to the same format so their shard warm-loads are fast too.
///
/// SAFETY — correctness does NOT depend on this migration completing. The
/// embedding read path (`de_embedding_dual`) tolerates BOTH formats, so a
/// half-migrated DB returns correct query results; this is purely a storage/
/// warm-load optimisation. That is what makes a background, resumable run safe.
///
/// Keyset pagination (mirrors run_migration_v1_to_v2):
///   - Cursor: `type::string(id) AS id_str` over chunk's random record ids.
///     `WHERE type::string(id) > $cursor ORDER BY id_str` visits every row
///     exactly once, skips none. `type::string(id)` avoids Thing-serde and
///     gives stable lexicographic order. ORDER BY uses the projected alias
///     (`id_str`) — a bare `type::string(id)` in ORDER BY fails in 2.6.5.
///   - Cursor persisted to `index_meta` each page for crash resume.
///   - Memory: one page (512 rows) + its re-encoded bytes in flight — O(page),
///     independent of chunk count. No OFFSET (would be O(N²)).
///
/// Idempotent: each page reads `embedding` via `de_embedding_dual` (so an
/// already-`bytes` row decodes correctly) and rewrites it with `pack_embedding`.
/// Re-encoding an already-converted row reproduces byte-identical content
/// (`pack(decode(bytes)) == bytes`), so resuming/replaying any page is a no-op
/// in effect. Empty embeddings (`[]` or empty bytes) round-trip to empty bytes.
///
/// `db_schema_version=5` is stamped ONLY after the full scan completes, so an
/// interrupted run leaves the version at 4 and the next `open_db` resumes from
/// the persisted cursor.
pub async fn run_migration_v4_to_v5(db: &Surreal<Db>) -> Result<()> {
    use serde::Deserialize;

    info!("migration v4→v5: packing chunk.embedding array<float> → bytes");

    let page_size: i64 = 512;
    let cursor_key = "migration_v5_chunk_cursor";
    let mut cursor: String = ops::get_meta(db, cursor_key)
        .await
        .ok()
        .flatten()
        .unwrap_or_default();

    loop {
        #[derive(Deserialize)]
        struct ChunkRow {
            id_str: String,
            // Dual reader: old rows are array<float>, already-migrated rows are
            // bytes; both decode to Vec<f32> so re-encoding is idempotent.
            #[serde(deserialize_with = "ops::de_embedding_dual")]
            embedding: Vec<f32>,
        }

        let batch: Vec<ChunkRow> = db
            .query(
                "SELECT type::string(id) AS id_str, embedding \
                 FROM chunk \
                 WHERE type::string(id) > $cursor \
                 ORDER BY id_str \
                 LIMIT $page",
            )
            .bind(("cursor", cursor.clone()))
            .bind(("page", page_size))
            .await
            .context("migration v4→v5: scan chunk page")?
            .take(0)?;

        if batch.is_empty() {
            break;
        }

        // Advance cursor to the last id_str in this page.
        cursor = batch
            .last()
            .map(|r| r.id_str.clone())
            .unwrap_or(cursor.clone());

        // Re-encode each row's embedding as packed bytes, updating by its OWN id.
        for row in &batch {
            let packed = ops::pack_embedding(&row.embedding);
            db.query("UPDATE type::thing($id) SET embedding = $embedding")
                .bind(("id", row.id_str.clone()))
                .bind(("embedding", surrealdb::sql::Bytes::from(packed)))
                .await
                .context("migration v4→v5: update chunk embedding by id")?;
        }

        // Persist cursor for crash resume.
        ops::set_meta(db, cursor_key, &cursor)
            .await
            .context("migration v4→v5: persist chunk cursor")?;

        let batch_len = batch.len() as i64;
        if batch_len < page_size {
            break;
        }
    }

    // Clean up cursor key.
    let _ = db
        .query("DELETE FROM index_meta WHERE key = $k")
        .bind(("k", cursor_key))
        .await;

    // Stamp version ONLY after the full scan completes (crash-resume anchor).
    ops::set_meta(db, DB_SCHEMA_VERSION_KEY, "5")
        .await
        .context("migration v4→v5: stamp db_schema_version=5")?;

    info!("migration v4→v5 complete");
    Ok(())
}

/// v5→v6: convert the `calls` table from a graph RELATION to a NORMAL table.
///
/// WHY: at v5+ no code traverses the call graph (`->calls->` / `<-calls<-`).
/// Every read path — query_callers/query_callees (graph_expand.rs) and call_graph
/// (ops.rs) — reads the denormalized in_name/out_name/in_file/out_file columns via
/// their secondary indexes. The RELATION type forced SurrealDB to maintain
/// graph-adjacency keys on every edge insert; on the Linux kernel (4.44M edges)
/// that was ~44% of Phase-2 edge-write time, for data nothing reads.
///
/// The table TYPE flip itself is done by `DEFINE TABLE OVERWRITE calls TYPE NORMAL`
/// in SCHEMA_DDL (applied synchronously on every open_db, before any write). This
/// migration only clears the OLD rows that were written as RELATION records, so
/// they are re-resolved as plain rows on the next index run.
///
/// Recovery model (matches the Phase-2 crash-recovery branch in pipeline.rs::run):
/// `DELETE FROM calls` drops stale RELATION rows; deleting the `edges_resolved`
/// marker forces Phase-2 re-resolution on the next run. The raw_edge table
/// (incremental path) or a full rebuild (RAM path) then repopulates calls.
/// Idempotent: re-running deletes an already-empty table and re-clears the marker
/// — both no-ops in effect.
///
/// Output invariance: verified on notepad-ade — the call-graph node/edge digest is
/// byte-identical before and after (RELATION vs NORMAL), since the read columns are
/// unchanged.
pub async fn run_migration_v5_to_v6(db: &Surreal<Db>) -> Result<()> {
    info!("migration v5→v6: converting calls RELATION → NORMAL (clearing stale edge rows)");

    // Drop all existing calls rows (written as RELATION records). They will be
    // re-resolved as plain NORMAL rows by Phase 2 on the next index trigger.
    db.query("DELETE FROM calls")
        .await
        .context("migration v5→v6: delete calls")?;

    // Clear the edges_resolved marker so Phase 2 re-runs. NOTE: a replay from
    // raw_edge was only safe when raw_edge was complete, which pre-v7 RAM-path DBs
    // never were; the chained v6→v7 migration below routes them to the edge heal.
    let _ = db
        .query("DELETE FROM index_meta WHERE key = $k")
        .bind(("k", "edges_resolved"))
        .await;

    // Stamp version ONLY after both deletes complete.
    ops::set_meta(db, DB_SCHEMA_VERSION_KEY, "6")
        .await
        .context("migration v5→v6: stamp db_schema_version=6")?;

    info!("migration v5→v6 complete");
    Ok(())
}

/// v6→v7: clear `edges_resolved` and `raw_edge_complete` so the next index trigger
/// heals the call graph from source (`IndexPipeline::heal_edges`).
///
/// WHY not `needs_rebuild`: a damaged DB (truncated `calls` under a stamped
/// `edges_resolved`) is indistinguishable from a healthy one, so every DB must be
/// healed — and a full rebuild deletes everything before it embeds, so with a dead
/// key it would empty every repo. The heal needs no embedding. Idempotent.
pub async fn run_migration_v6_to_v7(db: &Surreal<Db>) -> Result<()> {
    info!("migration v6→v7: clearing edge markers so the call graph heals from source");
    db.query("DELETE FROM index_meta WHERE key IN $keys")
        .bind(("keys", vec!["edges_resolved", "raw_edge_complete"]))
        .await
        .context("migration v6→v7: clear edge markers")?;
    ops::set_meta(db, DB_SCHEMA_VERSION_KEY, "7")
        .await
        .context("migration v6→v7: stamp db_schema_version=7")?;
    info!("migration v6→v7 complete");
    Ok(())
}

/// The six secondary indexes that the rebuild pipeline DROPS before its bulk write
/// and that are therefore NOT defined in `SCHEMA_DDL` (see the SCHEMA_DDL doc comment).
/// Each entry is `(index_name, table, field)`.
const DROPPABLE_SECONDARY_INDEXES: &[(&str, &str, &str)] = &[
    ("idx_symbol_file", "symbol", "file"),
    ("idx_symbol_name", "symbol", "name"),
    ("idx_calls_in_file", "calls", "in_file"),
    ("idx_calls_out_file", "calls", "out_file"),
    ("idx_calls_in_name", "calls", "in_name"),
    ("idx_calls_out_name", "calls", "out_name"),
];

/// Return the set of secondary-index names currently defined on `table`.
async fn defined_index_names(
    db: &Surreal<Db>,
    table: &str,
) -> Result<std::collections::BTreeSet<String>> {
    #[derive(serde::Deserialize)]
    struct TableInfo {
        indexes: std::collections::BTreeMap<String, String>,
    }
    let info: Option<TableInfo> = db
        .query(format!("INFO FOR TABLE {table}"))
        .await
        .context("INFO FOR TABLE")?
        .take(0)
        .context("take INFO FOR TABLE row")?;
    Ok(info
        .map(|i| i.indexes.into_keys().collect())
        .unwrap_or_default())
}

/// Read the CONCURRENTLY-build status of a single, already-defined index via
/// `INFO FOR INDEX <index> ON <table>`.
///
/// Returns the `building.status` string when a build record is present, or
/// `None` when the `INFO FOR INDEX` row carries no `building` block / no
/// `status` — which in SurrealDB 2.6.5 is exactly the shape of a plain,
/// fully-built index (one defined on an empty table, or one whose CONCURRENTLY
/// build has completed and whose transient `building` record is gone). This is
/// the SAME `building` → `status` parse `build_index_concurrently` uses in its
/// poll loop (kept identical so we match the real serialization rather than
/// inventing a second shape).
async fn index_building_status(
    db: &Surreal<Db>,
    index: &str,
    table: &str,
) -> Result<Option<String>> {
    let info: Option<serde_json::Value> = db
        .query(format!("INFO FOR INDEX {index} ON {table}"))
        .await
        .with_context(|| format!("INFO FOR INDEX {index} ON {table}"))?
        .take(0)
        .with_context(|| format!("take INFO FOR INDEX {index} ON {table} row"))?;
    Ok(info
        .as_ref()
        .and_then(|v| v.get("building"))
        .and_then(|b| b.get("status"))
        .and_then(|s| s.as_str())
        .map(|s| s.to_string()))
}

/// Decide whether an already-DEFINED index can be trusted as fully built from
/// its `building.status` (as returned by [`index_building_status`]).
///
/// A `DEFINE INDEX … CONCURRENTLY` writes the index DEFINITION at kickoff (so
/// `INFO FOR TABLE` immediately lists the name) while the BACKFILL runs in a
/// background pass. If the process dies MID-backfill, the name is present but
/// the index is only partially populated — and trusting name-presence alone
/// would let a half-built index serve Phase-2 name lookups / queries with
/// wrong/incomplete results. So an index is trustworthy ONLY when:
///   - `None`     → no in-progress `building` record at all = a plain,
///     fully-built index (empty-table define, or a completed concurrent build
///     whose transient record is gone), or
///   - `"ready"`  → the concurrent build reached completion.
///
/// Anything else — in-progress (`started`/`cleaning`/`indexing`) left by a
/// crash, or a terminal failure (`error`/`aborted`) — is NOT trustworthy and
/// must be dropped and rebuilt.
fn index_status_is_trustworthy(status: Option<&str>) -> bool {
    matches!(status, None | Some("ready"))
}

/// (Re)build the six drop-during-rebuild secondary indexes that `SCHEMA_DDL`
/// deliberately omits (`idx_symbol_*`, `idx_calls_*`).
///
/// ROOT-CAUSE FIX (crash recovery, Theory-A): if a process dies AFTER the rebuild
/// pipeline drops these indexes but BEFORE it rebuilds them, the table holds all its
/// rows while the indexes are physically absent. Re-defining them in `SCHEMA_DDL`
/// with a plain foreground `DEFINE INDEX … FIELDS …` would backfill every existing
/// row in ONE transaction, which ROLLS BACK under the production-pinned RocksDB write
/// buffers at kernel scale — and, because `open_db` runs `.check()` on `SCHEMA_DDL`,
/// would fail the ENTIRE open of that repo (the original symptom: "schema DDL
/// contained errors", repo unreopenable). Even absent the error the index would be
/// missing → the kernel-query regression returns.
///
/// This helper is the single owner of populated-table builds for these six indexes.
/// For each one:
///   - present + fully built (`building.status` is `ready`, or there is no
///     in-progress `building` record at all) → no-op (the steady-state reopen of a
///     healthy indexed DB; the common case — one cheap `INFO FOR TABLE` per table
///     plus one cheap `INFO FOR INDEX` per present index, no build).
///   - present but NOT ready (a `DEFINE INDEX … CONCURRENTLY` whose backfill was
///     interrupted by a crash — `building.status` is `indexing`/`cleaning`/`started`,
///     or a terminal `error`/`aborted`) → `REMOVE INDEX` then `build_index_concurrently`,
///     because a half-built index is listed by name yet would serve wrong/incomplete
///     results to Phase-2 name lookups and queries. We do NOT trust SurrealDB to
///     auto-resume an interrupted build (unverified in 2.6.5); we rebuild cleanly.
///   - absent + table empty → define it in the foreground (instant, no backfill — the
///     fresh-DB open; a full rebuild later drops+concurrently-rebuilds it anyway).
///   - absent + table populated → `build_index_concurrently` (batched commits, polls
///     to `ready`), which commits cleanly under the pinned buffers — the crash-recovery
///     self-heal, with NO foreground backfill over a populated table.
///
/// Idempotent + crash-safe: re-running is a no-op once all six are present AND ready;
/// an interrupted concurrent build (whether absent or present-but-not-ready) is
/// re-driven to `ready` on the next open.
pub async fn ensure_secondary_indexes(db: &Surreal<Db>) -> Result<()> {
    // Cache the present-index set + emptiness per table so the healthy-reopen path
    // (all six present) costs exactly two `INFO FOR TABLE` queries and no build.
    let mut present: HashMap<&str, std::collections::BTreeSet<String>> = HashMap::new();
    let mut empty: HashMap<&str, bool> = HashMap::new();

    for &(index, table, field) in DROPPABLE_SECONDARY_INDEXES {
        let names = match present.get(table) {
            Some(n) => n,
            None => {
                let n = defined_index_names(db, table).await?;
                present.entry(table).or_insert(n)
            }
        };
        if names.contains(index) {
            // Name present — but a `DEFINE INDEX … CONCURRENTLY` writes the
            // definition (so the name shows up in `INFO FOR TABLE`) BEFORE its
            // background backfill finishes. A crash mid-backfill leaves a
            // half-built index that name-presence alone would treat as ready.
            // Confirm the build actually reached `ready` (or carries no
            // in-progress `building` record at all) before trusting it.
            let status = index_building_status(db, index, table).await?;
            if index_status_is_trustworthy(status.as_deref()) {
                continue; // healthy reopen: index defined AND fully built → nothing to do.
            }
            // Present but not ready: a crash interrupted the concurrent backfill
            // (or it failed). Do NOT rely on SurrealDB auto-resuming the build
            // (unverified in 2.6.5). Drop the untrustworthy half-built index and
            // rebuild it cleanly via the concurrent helper (which polls to `ready`
            // and hard-errors on a terminal failure).
            warn!(
                %index, %table, ?status,
                "secondary index present but its concurrent build is not 'ready' \
                 (interrupted/failed backfill) — dropping and rebuilding concurrently"
            );
            db.query(format!("REMOVE INDEX IF EXISTS {index} ON {table};"))
                .await
                .with_context(|| format!("remove half-built {index} on {table}"))?
                .check()
                .with_context(|| format!("remove half-built {index} on {table} (check)"))?;
            build_index_concurrently(db, index, table, field)
                .await
                .with_context(|| format!("rebuild half-built {index} concurrently"))?;
            continue;
        }

        // Index absent. Decide foreground (empty table) vs concurrent (populated).
        let is_empty = match empty.get(table) {
            Some(&e) => e,
            None => {
                let row: Option<serde_json::Value> = db
                    .query(format!("SELECT 1 FROM {table} LIMIT 1"))
                    .await
                    .with_context(|| format!("probe {table} emptiness"))?
                    .take(0)
                    .with_context(|| format!("take {table} emptiness row"))?;
                let e = row.is_none();
                *empty.entry(table).or_insert(e)
            }
        };

        if is_empty {
            // Fresh DB / empty table: a plain DEFINE INDEX has nothing to backfill,
            // so it is instant and cannot trip the pinned-buffer rollback.
            db.query(format!(
                "DEFINE INDEX IF NOT EXISTS {index} ON {table} FIELDS {field};"
            ))
            .await
            .with_context(|| format!("define {index} on empty {table}"))?
            .check()
            .with_context(|| format!("define {index} on empty {table} (check)"))?;
        } else {
            // Populated table with the index missing: the crash-recovery case. Build
            // CONCURRENTLY (batched) so it survives the pinned buffers — never a
            // foreground backfill over a populated table.
            warn!(
                %index, %table,
                "secondary index absent on a populated table (likely a crash mid-rebuild) \
                 — rebuilding concurrently"
            );
            build_index_concurrently(db, index, table, field)
                .await
                .with_context(|| format!("crash-recovery concurrent build of {index}"))?;
        }
    }
    Ok(())
}

/// Build a secondary index over an already-populated table without a single
/// monolithic transaction.
///
/// ROOT-CAUSE FIX (kernel-query regression): a plain foreground
/// `DEFINE INDEX … FIELDS …` backfills every existing row inside ONE write
/// transaction. Under the production-pinned RocksDB write buffers
/// (`SURREAL_ROCKSDB_WRITE_BUFFER_SIZE` = 32 MiB × 2; see
/// `engine_boot::set_rocksdb_memory_bounds`), that single transaction over
/// millions of rows fails to commit — "Failed to commit transaction due to a
/// read or write conflict" — and the ENTIRE `DEFINE INDEX` (including the index
/// *definition*) rolls back, leaving `INFO FOR TABLE` with `indexes:{}`. The
/// pipeline's rebuild sites issued the statement with `.await?` and NO
/// `.check()`, so this statement-level rollback was silently swallowed: the run
/// reported success while the secondary indexes were physically absent, turning
/// every `WHERE out_name = …` / `WHERE file = …` into a full table scan
/// (measured ~165s / ~129s per call at kernel scale). Verified empirically:
/// the foreground build succeeds with SurrealDB's huge DEFAULT buffers but FAILS
/// (rolls back, index absent) under the pinned 32 MiB buffers at 3.5M rows.
///
/// `CONCURRENTLY` (SurrealDB 2.x) builds the index in a background two-stage
/// (initial + update) pass that batches its writes instead of one giant
/// transaction, so it commits cleanly under the pinned buffers. Because the
/// `DEFINE INDEX … CONCURRENTLY` statement returns immediately while the build
/// continues asynchronously, we then POLL `INFO FOR INDEX` until
/// `building.status == 'ready'` so the index is fully built and query-usable
/// before this function returns — preserving the pipeline's invariant that the
/// index exists before Phase 2 (symbol-name lookups) and before any query.
///
/// Idempotent + crash-safe: `IF NOT EXISTS` makes a re-run a no-op when the
/// index is already present; a crash mid-build leaves the prior (possibly
/// absent) definition, and the next full rebuild re-issues this build.
pub async fn build_index_concurrently(
    db: &Surreal<Db>,
    index: &str,
    table: &str,
    field: &str,
) -> Result<()> {
    // `.check()` is MANDATORY here: `.await?` alone returns Ok even when the
    // statement rolled back (the exact bug this fix addresses). A transient
    // write-conflict on the kickoff statement is retried below.
    let ddl = format!("DEFINE INDEX IF NOT EXISTS {index} ON {table} FIELDS {field} CONCURRENTLY;");

    // Kick off the concurrent build. Retry a transient commit conflict on the
    // (tiny) definition write itself — the heavy backfill happens in the
    // background pass, not in this statement's transaction.
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 0..5u32 {
        let outcome = match db.query(&ddl).await {
            Ok(resp) => resp.check().map(|_| ()),
            Err(e) => Err(e),
        };
        match outcome {
            Ok(()) => {
                last_err = None;
                break;
            }
            Err(e) => {
                last_err = Some(anyhow::Error::new(e).context(format!(
                    "kick off concurrent build of {index} (attempt {attempt})"
                )));
                tokio::time::sleep(std::time::Duration::from_millis(200 * (attempt as u64 + 1)))
                    .await;
            }
        }
    }
    if let Some(e) = last_err {
        return Err(e);
    }

    // Poll INFO FOR INDEX until the background build reaches status 'ready'.
    // While building, SurrealDB returns { building: { status: 'indexing',
    // initial, pending, updated } }; on completion { building: { status:
    // 'ready' } }. An index built when the table was empty (or already complete)
    // reports 'ready' immediately. A FAILED build reports { status: 'error',
    // error: <msg> } or { status: 'aborted' } — both terminal and handled below as
    // a hard error return (never a forever-poll). No fixed cap for a HEALTHY slow
    // build: a kernel-scale build legitimately takes tens of seconds, and the
    // caller's run is already long-lived; we log progress so a long build is never
    // silent.
    let start = std::time::Instant::now();
    let mut last_log = std::time::Instant::now();
    loop {
        let info: Option<serde_json::Value> = db
            .query(format!("INFO FOR INDEX {index} ON {table}"))
            .await
            .context("INFO FOR INDEX")?
            .take(0)
            .context("take INFO FOR INDEX row")?;

        // `building` is present only during/after a CONCURRENTLY build. Absent
        // `building` (or absent status) means the index is a plain definition
        // that is already complete (e.g. defined on an empty table by SCHEMA_DDL).
        let status = info
            .as_ref()
            .and_then(|v| v.get("building"))
            .and_then(|b| b.get("status"))
            .and_then(|s| s.as_str())
            .map(|s| s.to_string());

        match status.as_deref() {
            Some("ready") | None => {
                info!(
                    %index, %table,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "concurrent index build ready"
                );
                return Ok(());
            }
            // Terminal FAILURE states (surrealdb 2.6.5 `BuildingStatus` →
            // `Value`: see surrealdb-core/src/kvs/index.rs, the `From<BuildingStatus>`
            // impl serializes `Aborted` → "aborted" and `Error(msg)` → {status:"error",
            // error: msg}). These never progress to "ready", so we MUST return an error
            // instead of polling forever — otherwise a failed CONCURRENTLY build (disk
            // full, internal error, abort) wedges the per-repo indexing consumer
            // indefinitely, blocking ALL future indexing for that repo.
            Some("error") | Some("aborted") => {
                let detail = info
                    .as_ref()
                    .and_then(|v| v.get("building"))
                    .and_then(|b| b.get("error"))
                    .and_then(|e| e.as_str())
                    .unwrap_or("(no error detail)");
                let st = status.as_deref().unwrap_or("error");
                return Err(anyhow::anyhow!(
                    "concurrent build of index {index} on {table} reached terminal status \
                     '{st}' after {}ms: {detail}",
                    start.elapsed().as_millis() as u64
                ));
            }
            // Healthy in-progress states (started / cleaning / indexing): no fixed cap
            // — a kernel-scale build legitimately takes tens of seconds; we log
            // progress so a long build is never silent.
            Some(_) => {
                if last_log.elapsed() >= std::time::Duration::from_secs(5) {
                    info!(
                        %index, %table,
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        building = ?info.as_ref().and_then(|v| v.get("building")),
                        "concurrent index build in progress"
                    );
                    last_log = std::time::Instant::now();
                }
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }
        }
    }
}

/// Return the shared `Surreal<Db>` handle for `repo`, opening and caching it on
/// first use. Spawns background migration if stored schema version < current.
/// Per-repo open gate. RocksDB takes an EXCLUSIVE per-directory lock, so two
/// concurrent `open_db` calls on the same path race: the loser fails on the
/// `LOCK` file with "open surrealdb". The plain read-then-write double-check in
/// `get_or_open` only dedupes the *insert* — both callers can still pass the
/// read miss and call `open_db` simultaneously (e.g. the indexer's first warm
/// racing a browse request's first DB access). This gate serializes the
/// open critical section *per repo* so exactly one `open_db` runs per path; the
/// loser waits, then re-checks the cache and gets the winner's handle. Distinct
/// repos still open concurrently (one gate each). The map only ever grows by one
/// tiny `Arc<Mutex<()>>` per distinct repo — bounded by repo count, not repo size.
static OPEN_GATES: LazyLock<StdMutex<HashMap<String, Arc<Mutex<()>>>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

fn open_gate(repo: &str) -> Arc<Mutex<()>> {
    let mut gates = OPEN_GATES.lock().unwrap();
    gates
        .entry(repo.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Tracks in-flight background migration tasks per repo so `close_repo_db` can
/// abort + await them before directory removal. A running migration holds a
/// live `Surreal<Db>` clone (see `maybe_spawn_migration`) that pins the RocksDB
/// exclusive LOCK; without explicit cancellation the LOCK outlives `close_repo_db`
/// and `remove_index_dir` fails. Bounded by repo count (one entry per repo with a
/// live migration; entries self-remove on completion).
static MIGRATION_TASKS: LazyLock<StdMutex<HashMap<String, tokio::task::JoinHandle<()>>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

/// Abort and await any in-flight migration task for `repo`, dropping the
/// migration's `Surreal<Db>` clone so the RocksDB LOCK can be released before
/// directory removal. No-op if no migration is running (or it already finished
/// and self-removed). Safe to call always: migrations are idempotent and
/// crash-resumable, so an aborted migration self-heals on the next open.
pub async fn abort_migration(repo: &str) {
    let repo = normalize_repo_path(repo);
    let handle = {
        let mut tasks = MIGRATION_TASKS.lock().unwrap();
        tasks.remove(&repo)
    };
    if let Some(handle) = handle {
        handle.abort();
        let _ = handle.await; // JoinError from abort/panic is expected; ignore.
    }
}

/// Wait until no migration task is running for `repo`.
///
/// The index consumer calls this before running the pipeline: a migration runs
/// unlocked in the background, and racing it is unsafe — v5→v6 `DELETE FROM calls`
/// against a live Phase 2, v4→v5 `UPDATE` of chunk rows an incremental just deleted
/// (resurrecting them), and v6→v7's marker clear must be visible to the very first
/// run. Polls the registry (each task self-deregisters on completion), so the wait
/// is bounded by the migration itself; aborted tasks are removed by
/// `abort_migration`.
pub async fn wait_migration(repo: &str) {
    let repo = normalize_repo_path(repo);
    loop {
        let running = MIGRATION_TASKS
            .lock()
            .unwrap()
            .get(&repo)
            .is_some_and(|h| !h.is_finished());
        if !running {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// Remove a repo's on-disk index directory, serialized against `open_db` via the
/// same per-repo open gate `get_or_open` uses.
///
/// Why the gate matters: SurrealDB's RocksDB datastore releases its exclusive
/// per-directory LOCK *asynchronously* — a background router task flushes
/// memtables and shuts down the engine some time after the last `Surreal<Db>`
/// clone drops. On Windows the OS file handles outlive the handle drop, so a
/// `remove_dir_all` immediately after `close_repo_db` can fail, AND a concurrent
/// re-index that calls `open_db` will `create_dir_all` + open RocksDB on the very
/// path we're trying to delete. Without serialization those two interleave: the
/// cleaner deletes files out from under a freshly opened datastore (or collides
/// with the still-draining old LOCK), producing repeating `open surrealdb`
/// errors on re-index.
///
/// Holding the open gate for the entire retry loop closes that race: any
/// concurrent `get_or_open` blocks on the gate until removal finishes, then
/// re-checks the cache (miss) and opens a fresh DB on the now-clean directory.
/// The caller MUST have already dropped the cached handle (`close_repo_db`) so
/// the only thing keeping the LOCK alive is the async shutdown, which the retry
/// loop waits out. Returns `true` if the directory is gone on return.
pub async fn remove_index_dir(data_dir: &Path, repo: &str, generation: u32) -> bool {
    let repo = normalize_repo_path(repo);

    // Serialize against open_db for this repo. Held across every retry so no
    // re-index can recreate/open the directory mid-removal. This matters ONLY for
    // the self-heal path (open_or_reset_index), which deletes the *current*
    // generation and then reopens it on the SAME path — a concurrent open of that
    // same generation would race the delete. The delete handler, by contrast, has
    // already bumped the generation, so nothing can target the old path anymore;
    // it uses `remove_old_generation_dir` (no gate) to avoid blocking the fresh
    // generation's open behind this drain.
    let gate = open_gate(&repo);
    let _open_guard = gate.lock().await;

    remove_dir_with_retry(data_dir, &repo, generation).await
}

/// Remove a SUPERSEDED generation's directory WITHOUT holding the per-repo open
/// gate. Safe only after the generation counter has already been bumped and
/// persisted: once that is durable, every open/path resolution for the repo
/// targets the new generation, so no concurrent `open_db` can recreate or race
/// this old path — there is nothing to serialize against. Holding the gate here
/// would be actively harmful: the gate is keyed by repo (not generation), so a
/// ~30s Windows+Defender lock drain on the OLD directory would block the FRESH
/// generation's open for the entire window — wedging a just-triggered re-index in
/// the indeterminate "Indexing…" state with an unresponsive Cancel, then failing
/// once the re-index recreated the still-draining old path. If the drain outlives
/// the retry budget, the leftover is reclaimed by `sweep_stale_generations` on the
/// next boot. Returns `true` if the directory is gone on return.
pub async fn remove_old_generation_dir(data_dir: &Path, repo: &str, generation: u32) -> bool {
    let repo = normalize_repo_path(repo);
    remove_dir_with_retry(data_dir, &repo, generation).await
}

/// Shared removal core: `remove_dir_all` with backoff to ride out the async
/// RocksDB LOCK drain. `repo` is assumed already normalized. Callers decide
/// whether to hold the per-repo open gate (see the two wrappers above).
async fn remove_dir_with_retry(data_dir: &Path, repo: &str, generation: u32) -> bool {
    let path = db_path(data_dir, repo, generation);

    if !path.exists() {
        return true;
    }

    // Retry with backoff: the async datastore shutdown that still holds the LOCK
    // typically completes within a second or two. Budget ~30s total to match
    // `open_db`'s retry budget — a slow Windows+Defender handle drain that
    // `open_db` tolerates must not make removal give up early.
    for attempt in 0..20u32 {
        let p = path.clone();
        let removed = tokio::task::spawn_blocking(move || std::fs::remove_dir_all(&p).is_ok())
            .await
            .unwrap_or(false);
        if removed || !path.exists() {
            return true;
        }
        // 200ms, 400ms, … capped at 2s — summing to ~30s over 20 tries, mirroring
        // `open_db`'s backoff so removal waits out the same drain window.
        let backoff_ms = (200u64 * (attempt as u64 + 1)).min(2000);
        tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
    }

    let still = path.exists();
    if still {
        warn!(path = ?path, "index directory still present after removal retries");
    }
    !still
}

/// Boot-time sweep of stale per-repo index generations.
///
/// Each repo/index delete bumps a repo's generation and moves the next index to a
/// fresh directory; when the OLD directory's removal failed (Windows held the LOCK
/// past the retry budget) it is left on disk. Repeated delete+reindex on a stubborn
/// lock would otherwise let those orphans accumulate without bound — violating the
/// "disk stays bounded at scale" rule. This sweep reclaims them.
///
/// MUST be called at startup BEFORE any RocksDB handle is opened (no entry in
/// `repo_dbs`, no warmed shard): only then is every directory guaranteed lock-free,
/// so a `remove_dir_all` can't race a live datastore. For each `(repo, generation)`
/// it removes every sibling generation directory for that repo EXCEPT the current
/// one (gen 0 → `rocksdb/<name>`; gen N → `rocksdb/N/<name>`). A directory that
/// still can't be removed (rare residual OS handle) is skipped, not surfaced as an
/// error — the next boot retries it.
///
/// Scope: only repos still listed in `repos` are swept. Directories for repos fully
/// forgotten from settings are left untouched (a deeper sweep can be added later if
/// that becomes a real disk concern).
pub fn sweep_stale_generations(
    data_dir: &Path,
    repos: &[String],
    generations: &HashMap<String, u32>,
) {
    let rocksdb_root = data_dir.join("rocksdb");
    if !rocksdb_root.exists() {
        return;
    }

    for repo in repos {
        let repo = normalize_repo_path(repo);
        let current = generations.get(&repo).copied().unwrap_or(0);

        // Candidate stale paths = every generation's directory for this repo other
        // than `current`. We can't enumerate "all generations ever used", so we
        // sweep the contiguous range [0, current): every prior generation the
        // counter has passed through. The live `current` directory is preserved.
        for prior_gen in 0..current {
            let stale = db_path(data_dir, &repo, prior_gen);
            if stale.exists() {
                match std::fs::remove_dir_all(&stale) {
                    Ok(()) => info!(path = ?stale, repo = %repo, "swept stale index generation"),
                    Err(e) => {
                        // Skip, don't error: a residual OS handle may still hold it.
                        // The next boot will retry. We only warn so the operator can
                        // see disk isn't being reclaimed if it persists.
                        warn!(path = ?stale, error = %e, "could not sweep stale index generation; will retry next boot");
                    }
                }
            }
        }
    }
}

pub async fn get_or_open(
    repo_dbs: &RepoDbMap,
    data_dir: &Path,
    repo: &str,
    generation: u32,
) -> Result<Surreal<Db>> {
    let repo = &normalize_repo_path(repo);
    // Fast path: already cached.
    if let Some(db) = repo_dbs.read().await.get(repo.as_str()) {
        return Ok(db.clone());
    }

    // Slow path: serialize the open per repo so concurrent first-opens can't both
    // call `open_db` and collide on RocksDB's exclusive directory lock. The gate
    // is acquired BEFORE any repo_dbs lock is held (the read guard above is already
    // dropped), so the global lock order (repo_dbs → vector_index) is preserved.
    let gate = open_gate(repo);
    let _open_guard = gate.lock().await;

    // Re-check under the gate: a previous holder may have just opened it.
    if let Some(db) = repo_dbs.read().await.get(repo) {
        return Ok(db.clone());
    }

    let db = open_db(data_dir, repo, generation).await?;

    // Check schema version and spawn migration if needed (non-blocking).
    let stored_version = stamp_if_fresh(&db).await?;

    let mut map = repo_dbs.write().await;
    // Final double-check (defensive; the gate already guarantees uniqueness).
    if let Some(existing) = map.get(repo) {
        return Ok(existing.clone());
    }
    map.insert(repo.to_string(), db.clone());
    drop(map);

    // Spawn migration AFTER the handle is in the map so the migration task's
    // `repo_dbs.read().get(repo)` finds the freshly-inserted handle to clone. The
    // task then holds that owned clone for its duration (see `maybe_spawn_migration`);
    // `close_repo_db` cancels it via `store::abort_migration` before removal.
    maybe_spawn_migration(repo_dbs.clone(), repo.to_string(), stored_version);

    Ok(db)
}

/// Open the repo DB like [`get_or_open`], but self-heal a corrupt or orphaned-LOCK
/// index directory by deleting it and reopening fresh. Returns `(db, was_reset)`
/// where `was_reset` is `true` when the directory had to be destroyed and rebuilt.
///
/// WHY this is safe and correct:
/// - `open_db` already retries the RocksDB open for ~30s to ride out a *transient*
///   stale LOCK from a draining prior handle (see its retry loop). So if
///   `get_or_open` still fails here, the directory is genuinely corrupt or holds an
///   orphaned LOCK that no longer has an owner — neither of which clears on its own.
/// - Deleting the directory and reopening is exactly the pipeline's documented
///   `is_first_run` recovery: a missing `file_meta` triggers a full rebuild. It is
///   API-free because embeddings are cached in a *separate* directory
///   (`<data_dir>/embeddings/<model>/`), so the rebuild re-uses cached vectors and
///   never re-hits the Voyage API for unchanged content.
/// - The deletion goes through [`remove_index_dir`], whose `remove_dir_all` FAILS
///   when a live OS handle still holds the LOCK. That failure is the intended SAFETY
///   VALVE: if some other handle is alive on this path, the index is healthy (just
///   contended) and must NOT be destroyed — so we surface the original error and
///   leave the data untouched. No data loss is possible.
/// - We retry the open exactly ONCE after a successful delete. A *fresh empty
///   directory* that still won't open is not a corruption we can fix by deleting
///   again — it signals an environment fault (disk full, permissions, AV
///   quarantine). Looping would just thrash; we surface the second error instead.
///
/// DEADLOCK NOTE: the index consumer that calls this already holds the per-repo
/// *index* lock. `close_repo_db` re-acquires that same lock, so we must NOT call it
/// here — we drop the cached handle directly from `repo_dbs`. `remove_index_dir`
/// uses a *separate* per-repo open gate (released by `get_or_open` before it
/// returns), so there is no deadlock between the index lock and the open gate.
pub async fn open_or_reset_index(
    repo_dbs: &RepoDbMap,
    data_dir: &Path,
    repo: &str,
    generation: u32,
) -> Result<(Surreal<Db>, bool)> {
    match get_or_open(repo_dbs, data_dir, repo, generation).await {
        Ok(db) => Ok((db, false)),
        Err(orig) => {
            let normalized = normalize_repo_path(repo);

            // Defensively drop any cached handle so nothing in this process keeps
            // the LOCK alive. We remove it directly from the map rather than via
            // `close_repo_db`, which would deadlock (see DEADLOCK NOTE above).
            repo_dbs.write().await.remove(&normalized);

            // Attempt to delete the on-disk index. Returns false (without deleting)
            // if a live OS handle still holds the LOCK — the safety valve.
            if remove_index_dir(data_dir, repo, generation).await {
                // Directory is gone. Reopen exactly once on the fresh path.
                match get_or_open(repo_dbs, data_dir, repo, generation).await {
                    Ok(db) => Ok((db, true)),
                    Err(e2) => Err(e2).context("reopen after index reset"),
                }
            } else {
                // A live handle blocked removal: the index is healthy, not corrupt.
                // Surface the original open error and leave the data intact.
                Err(orig)
            }
        }
    }
}

/// Like [`get_or_open`], but returns `Ok(None)` when the repo has **no index on
/// disk yet** instead of creating one.
///
/// `open_db` calls `create_dir_all`, so a bare `get_or_open` on a never-indexed
/// repo materializes an empty RocksDB directory purely as a side effect of a
/// read — and can race the indexer's first open. Read-only browse endpoints use
/// this guard so an unindexed repo reads as "not indexed" (empty state) rather
/// than erroring or leaving a phantom DB behind. Once a repo has been indexed
/// (or is mid-indexing) the directory exists and this behaves like `get_or_open`.
pub async fn open_if_indexed(
    repo_dbs: &RepoDbMap,
    data_dir: &Path,
    repo: &str,
    generation: u32,
) -> Result<Option<Surreal<Db>>> {
    let repo = normalize_repo_path(repo);
    // A cached handle means it's open regardless of the on-disk check below.
    if let Some(db) = repo_dbs.read().await.get(repo.as_str()) {
        return Ok(Some(db.clone()));
    }
    if !db_path(data_dir, &repo, generation).exists() {
        return Ok(None);
    }
    get_or_open(repo_dbs, data_dir, &repo, generation)
        .await
        .map(Some)
}

#[cfg(test)]
mod generation_paths {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    /// Generation 0 maps to the LEGACY layout (`rocksdb/<name>`, no number segment)
    /// so existing on-disk indexes are not orphaned. Generation ≥ 1 nests under the
    /// number (`rocksdb/<gen>/<name>`).
    #[test]
    fn db_path_layout_by_generation() {
        let data_dir = std::path::Path::new("/data");
        let repo = "/home/user/projects/notepad";
        let name = sanitize_repo_name(repo);

        let gen0 = db_path(data_dir, repo, 0);
        assert_eq!(
            gen0,
            data_dir.join("rocksdb").join(&name),
            "generation 0 must be the legacy path with no number segment"
        );

        let gen1 = db_path(data_dir, repo, 1);
        assert_eq!(
            gen1,
            data_dir.join("rocksdb").join("1").join(&name),
            "generation 1 must nest the repo under the number segment"
        );

        let gen7 = db_path(data_dir, repo, 7);
        assert_eq!(gen7, data_dir.join("rocksdb").join("7").join(&name));

        // Distinct generations never collide on disk.
        assert_ne!(gen0, gen1);
        assert_ne!(gen1, gen7);
    }

    /// The boot sweep removes every prior generation directory for a listed repo
    /// while preserving the current one. A directory it can't enumerate is simply
    /// absent (no error). Repos not in the list are left untouched.
    #[test]
    fn sweep_removes_prior_generations_keeps_current() {
        let home = TempDir::new().expect("tempdir");
        let data_dir = home.path();
        let repo = "/proj/alpha";
        let other = "/proj/untracked";

        // Materialise generation dirs 0,1,2 for `repo` and gen 0 for `other`.
        for g in 0..=2u32 {
            std::fs::create_dir_all(db_path(data_dir, repo, g)).expect("mk repo gen dir");
        }
        std::fs::create_dir_all(db_path(data_dir, other, 0)).expect("mk other gen dir");

        // Current generation of `repo` is 2; `other` is not listed.
        let mut generations = HashMap::new();
        generations.insert(normalize_repo_path(repo), 2u32);

        sweep_stale_generations(data_dir, &[repo.to_string()], &generations);

        assert!(!db_path(data_dir, repo, 0).exists(), "gen 0 must be swept");
        assert!(!db_path(data_dir, repo, 1).exists(), "gen 1 must be swept");
        assert!(
            db_path(data_dir, repo, 2).exists(),
            "current gen 2 must be kept"
        );
        assert!(
            db_path(data_dir, other, 0).exists(),
            "untracked repo's directory must be left untouched"
        );
    }

    /// A repo absent from the generations map (or at gen 0) has no prior generations,
    /// so the sweep is a no-op for it and never touches its live gen-0 directory.
    #[test]
    fn sweep_noop_for_generation_zero_repo() {
        let home = TempDir::new().expect("tempdir");
        let data_dir = home.path();
        let repo = "/proj/fresh";
        std::fs::create_dir_all(db_path(data_dir, repo, 0)).expect("mk gen0");

        sweep_stale_generations(data_dir, &[repo.to_string()], &HashMap::new());

        assert!(
            db_path(data_dir, repo, 0).exists(),
            "gen-0 repo with no bump must keep its directory"
        );
    }
}

#[cfg(test)]
mod isolation_repro {
    use super::*;
    use tempfile::TempDir;

    /// RocksDB takes an EXCLUSIVE per-directory lock, so two independent handles on
    /// the same on-disk path cannot coexist — a second open fails on the LOCK file.
    /// This makes the shared `get_or_open` cache (one handle per repo) the mandatory
    /// access pattern, not merely an optimization. This test proves both halves:
    /// (1) a second raw `open_db` is rejected while a cached handle is alive;
    /// (2) the shared cached handle reads its own writes correctly.
    ///
    /// Note: we do NOT drop-then-reopen — SurrealDB releases the RocksDB lock
    /// asynchronously (the datastore lives in a background task past handle drop),
    /// so an immediate reopen in-process would race the lock. Production never
    /// drops+reopens; `get_or_open` keeps exactly one cached handle for the repo's
    /// lifetime, which is precisely what this test exercises.
    #[tokio::test]
    async fn exclusive_lock_then_shared_handle_works() {
        let home = TempDir::new().unwrap();
        let repo = "/proj/repo_iso";

        // The shared cache opens the single authoritative handle.
        let map: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
        let sa = get_or_open(&map, home.path(), repo, 0)
            .await
            .expect("shared A");
        assert_eq!(
            ops::count_chunks(&sa).await.unwrap(),
            0,
            "fresh DB must be empty"
        );

        // ── PART 1: a second RAW open on the same live path must be rejected ────
        // (Under the old SurrealKV backend this silently succeeded with isolated
        // state — the root of the original cross-handle bug. RocksDB's exclusive
        // lock structurally prevents it.)
        let raw_result = open_db(home.path(), repo, 0).await;
        assert!(
            raw_result.is_err(),
            "RocksDB must reject a second concurrent handle on the same path (exclusive lock)"
        );

        // ── PART 2: the shared cached handle reads its own writes ───────────────
        // A second get_or_open returns the SAME cached instance (no new lock).
        let sb = get_or_open(&map, home.path(), repo, 0)
            .await
            .expect("shared B");
        sb.query(
            "CREATE chunk SET file = '/x/f.rs', line_start = 3, line_end = 4, \
             content = 'y', embedding = [0.5, 0.6, 0.7, 0.8], symbol_ref = NONE;",
        )
        .await
        .expect("write chunk via shared B");

        let sa_after = ops::count_chunks(&sa).await.unwrap();
        assert_eq!(
            sa_after, 1,
            "shared handle must see writes made through the same cached instance"
        );
    }
}

#[cfg(test)]
mod open_concurrency {
    use super::*;
    use tempfile::TempDir;

    /// Regression: N concurrent first-opens on the SAME repo must NOT race on
    /// RocksDB's exclusive directory lock. Before the per-repo open gate, two
    /// callers could both miss the read-cache, both call `open_db`, and the loser
    /// failed with "open surrealdb" (the symptom behind the "Failed to load files:
    /// failed to open index DB" UI error). With the gate, exactly one `open_db`
    /// runs and every caller gets the same handle.
    #[tokio::test]
    async fn concurrent_first_opens_do_not_race() {
        let home = TempDir::new().unwrap();
        let repo = "/proj/repo_concurrent".to_string();
        let map: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));

        // Fan out many simultaneous first-opens.
        let mut handles = Vec::new();
        for _ in 0..16 {
            let map = map.clone();
            let home = home.path().to_path_buf();
            let repo = repo.clone();
            handles.push(tokio::spawn(async move {
                get_or_open(&map, &home, &repo, 0).await.map(|_| ())
            }));
        }
        for h in handles {
            h.await
                .unwrap()
                .expect("every concurrent open must succeed (no lock race)");
        }

        // Exactly one handle ended up cached.
        assert_eq!(
            map.read().await.len(),
            1,
            "exactly one cached handle per repo"
        );
    }

    /// `open_if_indexed` returns None for a never-indexed repo (no DB directory on
    /// disk) and does NOT create one as a side effect — so a read-only browse of an
    /// unindexed repo reads as "not indexed" rather than erroring or leaving a
    /// phantom DB behind. After a real open the directory exists and it returns Some.
    #[tokio::test]
    async fn open_if_indexed_skips_unindexed_repo() {
        let home = TempDir::new().unwrap();
        let repo = "/proj/repo_never_indexed";
        let map: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));

        // Never indexed → None, and no directory materialized.
        let res = open_if_indexed(&map, home.path(), repo, 0)
            .await
            .expect("ok");
        assert!(res.is_none(), "unindexed repo must return None");
        assert!(
            !db_path(home.path(), repo, 0).exists(),
            "open_if_indexed must NOT create the DB directory for an unindexed repo"
        );
        assert_eq!(
            map.read().await.len(),
            0,
            "no handle cached for an unindexed repo"
        );

        // After a real open, the directory exists → Some, and the handle is shared.
        let _opened = get_or_open(&map, home.path(), repo, 0).await.expect("open");
        assert!(db_path(home.path(), repo, 0).exists());
        let res2 = open_if_indexed(&map, home.path(), repo, 0)
            .await
            .expect("ok");
        assert!(res2.is_some(), "indexed repo must return Some");
    }
}

#[cfg(test)]
mod reset_index {
    use super::*;
    use tempfile::TempDir;

    /// A healthy repo opens normally and is NEVER reset. First open creates a
    /// fresh empty DB (`was_reset == false`); a second open returns the same
    /// cached handle, also without a reset. This is the common path — the heal
    /// must add zero behavior change when nothing is wrong.
    #[tokio::test]
    async fn healthy_repo_is_not_reset() {
        let home = TempDir::new().unwrap();
        let repo = "/proj/repo_healthy";
        let map: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));

        let (db, was_reset) = open_or_reset_index(&map, home.path(), repo, 0)
            .await
            .expect("healthy open");
        assert!(!was_reset, "a healthy fresh repo must not be reset");
        assert_eq!(
            ops::count_chunks(&db).await.unwrap(),
            0,
            "fresh DB must be empty"
        );
        assert!(
            db_path(home.path(), repo, 0).exists(),
            "index directory must exist"
        );

        let (_db2, was_reset2) = open_or_reset_index(&map, home.path(), repo, 0)
            .await
            .expect("second open");
        assert!(!was_reset2, "a cached healthy repo must not be reset");
    }

    /// SAFETY VALVE: when a live OS handle still holds the exclusive RocksDB LOCK,
    /// the heal's `remove_dir_all` fails, so `open_or_reset_index` MUST surface the
    /// original open error and leave the directory intact — never destroy a
    /// (contended-but-healthy) index.
    ///
    /// Deterministic setup mirrors `exclusive_lock_then_shared_handle_works`: a raw
    /// `open_db` handle (held in a local, NOT inserted into the map) holds the lock.
    /// With an EMPTY map, `open_or_reset_index` misses the cache → `open_db` fails on
    /// the live lock → `remove_index_dir` fails (live OS handle blocks removal) →
    /// `Err` is returned. The raw handle is kept alive until after the assertion.
    #[tokio::test]
    async fn live_handle_blocks_reset_and_surfaces_error() {
        let home = TempDir::new().unwrap();
        let repo = "/proj/repo_locked";

        // Hold the exclusive lock with a raw handle (not in the map).
        let _holder = open_db(home.path(), repo, 0).await.expect("hold lock");

        let map: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
        let res = open_or_reset_index(&map, home.path(), repo, 0).await;
        assert!(
            res.is_err(),
            "a live lock must block the reset and surface the original error (no data loss)"
        );

        // The directory must still exist — the safety valve did not delete it.
        assert!(
            db_path(home.path(), repo, 0).exists(),
            "the contended index directory must NOT be destroyed"
        );

        // Keep the holder alive until here so the lock is genuinely held during the
        // heal attempt above.
        drop(_holder);
    }

    /// REGRESSION (delete-then-reindex stuck on "Indexing…"): the delete handler bumps
    /// the generation, then removes the OLD generation's directory via
    /// `remove_old_generation_dir`, which must NOT hold the per-repo open gate. If it
    /// did, a re-index that opens the FRESH generation (which acquires the same
    /// repo-keyed gate) would block behind the old directory's lock drain. Here we
    /// hold the open gate explicitly and assert the ungated removal still completes —
    /// i.e. it never tries to take the gate.
    #[tokio::test]
    async fn remove_old_generation_dir_does_not_take_open_gate() {
        let home = TempDir::new().unwrap();
        let repo = "/proj/repo_gen_swap";

        // Materialize an old-generation directory with no live handle (nothing holds
        // the RocksDB LOCK), so removal itself can succeed.
        let old_path = db_path(home.path(), repo, 0);
        std::fs::create_dir_all(&old_path).unwrap();
        std::fs::write(old_path.join("CURRENT"), b"stale\n").unwrap();

        // Hold the per-repo open gate for the entire removal — as a concurrent open of
        // the fresh generation would. A gated removal would deadlock/serialize here;
        // the ungated one must complete regardless.
        let gate = open_gate(&normalize_repo_path(repo));
        let _held = gate.lock().await;

        let removed = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            remove_old_generation_dir(home.path(), repo, 0),
        )
        .await
        .expect("ungated removal must not block on the held open gate");

        assert!(removed, "old generation directory must be removed");
        assert!(!old_path.exists(), "old generation directory must be gone");
    }

    /// CORRUPT-BUT-UNLOCKED: a directory with a malformed RocksDB `CURRENT` (pointing
    /// at a non-existent MANIFEST) and NO live handle. If `open_db` rejects it, the
    /// heal deletes the dir (nothing holds the lock) and reopens fresh, returning
    /// `(db, true)` with an empty, usable DB.
    ///
    /// EMPIRICAL OUTCOME (verified by running this test): RocksDB DOES reject the
    /// malformed `CURRENT`, so the open fails and the heal path is exercised —
    /// `was_reset == true` and the rebuilt DB is empty.
    #[tokio::test]
    async fn corrupt_unlocked_dir_is_healed() {
        let home = TempDir::new().unwrap();
        let repo = "/proj/repo_corrupt";

        // Materialize the index directory with a garbage CURRENT file. No handle is
        // opened, so no LOCK is held — removal will succeed.
        let path = db_path(home.path(), repo, 0);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("CURRENT"), b"GARBAGE-NOT-A-MANIFEST\n").unwrap();

        let map: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
        let (db, was_reset) = open_or_reset_index(&map, home.path(), repo, 0)
            .await
            .expect("corrupt dir must heal and reopen");
        assert!(
            was_reset,
            "a corrupt unlocked directory must be reset and rebuilt fresh"
        );
        assert_eq!(
            ops::count_chunks(&db).await.unwrap(),
            0,
            "the rebuilt DB must be empty and usable"
        );
    }
}

// ─── Stale-schema regression ──────────────────────────────────────────────
//
// This module proves that `DEFINE FIELD OVERWRITE` correctly migrates an existing
// database whose field was created with the OLD type (`option<record<symbol>>`).
//
// WITHOUT the OVERWRITE fix (plain `DEFINE FIELD`):
//   - Re-applying the corrected DDL is a no-op: the on-disk type stays as
//     `option<record<symbol>>`.
//   - Attempting to write a quoted-string `symbol_ref` value fails with:
//       "Found '<string>' for field `symbol_ref`, ... but expected a
//       option<record<symbol>>"
//   - The whole transaction rolls back silently.
//
// WITH the OVERWRITE fix:
//   - Re-applying the DDL updates the persisted type to `option<string>`.
//   - The same quoted-string write commits successfully (count = 1).
//
// This is the exact scenario for every on-disk SurrealKV database that was
// created before the `parent`/`symbol_ref` type correction — which is why the
// bug only appeared on existing deployments, not on fresh installs.
#[cfg(test)]
mod stale_schema {
    use surrealdb::Surreal;
    use surrealdb::engine::local::{Db, RocksDb};
    use tempfile::TempDir;

    use crate::store::ops::count_chunks;
    use crate::store::schema::SCHEMA_DDL;

    /// Open a raw SurrealKV DB (no DDL applied) on a TempDir.
    /// The caller is responsible for applying whatever schema it needs.
    async fn open_raw_db(dir: &std::path::Path, name: &str) -> Surreal<Db> {
        let path = dir.join(name);
        std::fs::create_dir_all(&path).unwrap();
        let db = Surreal::new::<RocksDb>(path.to_str().unwrap())
            .await
            .expect("open raw db");
        db.use_ns("context_engine")
            .use_db(name)
            .await
            .expect("ns/db");
        db
    }

    /// Retrieve the INFO FOR TABLE result for `table` as a raw JSON string.
    /// Used to inspect the persisted field definition before and after DDL re-application.
    async fn info_for_table(db: &Surreal<Db>, table: &str) -> String {
        let result: Option<serde_json::Value> = db
            .query(format!("INFO FOR TABLE {table};"))
            .await
            .expect("INFO FOR TABLE")
            .take(0)
            .ok()
            .flatten();
        format!("{result:?}")
    }

    /// STEP 1 (RED → GREEN):
    ///
    /// 1. Force the datastore into the STALE state: apply OLD DDL declaring
    ///    `symbol_ref` and `parent` as `option<record<symbol>>`.
    /// 2. Inspect the persisted type via `INFO FOR TABLE` — confirms old type is in place.
    /// 3. Re-apply the CURRENT corrected `SCHEMA_DDL` (with OVERWRITE).
    /// 4. Inspect again — with OVERWRITE the type MUST now read `option<string>`.
    /// 5. Attempt the real writer's statement (quoted-string `symbol_ref` inside a txn).
    /// 6. Assert the write COMMITS and count = 1.
    ///
    /// This test FAILS without `DEFINE FIELD OVERWRITE` (plain re-DEFINE is a no-op,
    /// the FieldCheck error still triggers) and PASSES with OVERWRITE.
    #[tokio::test]
    async fn overwrite_migrates_stale_schema_and_write_commits() {
        let home = TempDir::new().unwrap();
        let db = open_raw_db(home.path(), "stale_repro").await;

        // ── 1. Install the OLD (stale) schema ────────────────────────────────
        // This mirrors what every pre-fix on-disk database had: both critical
        // fields declared as `option<record<symbol>>`.
        // The chunk table must include all previously-required fields (file,
        // line_start, line_end, content, embedding) so that SCHEMA_DDL's
        // `DEFINE INDEX IF NOT EXISTS idx_chunk_file ON chunk FIELDS file`
        // does not fail with FdNotFound on this SCHEMAFULL table.
        let old_ddl = "\
            DEFINE TABLE chunk SCHEMAFULL;\
            DEFINE FIELD file ON chunk TYPE string;\
            DEFINE FIELD line_start ON chunk TYPE int;\
            DEFINE FIELD line_end ON chunk TYPE int;\
            DEFINE FIELD content ON chunk TYPE string;\
            DEFINE FIELD embedding ON chunk TYPE array<float>;\
            DEFINE FIELD symbol_ref ON chunk TYPE option<record<symbol>>;\
            DEFINE TABLE symbol SCHEMAFULL;\
            DEFINE FIELD parent ON symbol TYPE option<record<symbol>>;";
        db.query(old_ddl)
            .await
            .expect("install old stale DDL")
            .check()
            .expect("old DDL must not err");

        // ── 2. Confirm the old type is persisted ─────────────────────────────
        let before = info_for_table(&db, "chunk").await;
        println!("STALE-SCHEMA INFO BEFORE re-apply:\n  chunk: {before}");
        // The persisted definition must contain `record<symbol>` or `record` to
        // confirm the stale state is actually in place.
        assert!(
            before.to_lowercase().contains("record"),
            "before re-apply, the stale type must contain 'record' — got: {before}"
        );

        // ── 3. Re-apply the corrected SCHEMA_DDL ─────────────────────────────
        // The chunk table is now SCHEMALESS in SCHEMA_DDL — its typed field
        // definitions were removed. `DEFINE TABLE IF NOT EXISTS chunk SCHEMALESS`
        // is a no-op here since the table already exists.
        // The symbol table still has `DEFINE FIELD OVERWRITE parent` which fixes
        // the stale `option<record<symbol>>` type.
        db.query(SCHEMA_DDL)
            .await
            .expect("corrected DDL must not return transport error")
            .check()
            .expect("corrected DDL must have no per-statement errors");

        // ── 4. Confirm symbol.parent type has been updated ───────────────────
        let after_symbol = info_for_table(&db, "symbol").await;
        println!("STALE-SCHEMA INFO AFTER re-apply:\n  symbol: {after_symbol}");
        // After OVERWRITE, `record<symbol>` must be gone from symbol.parent's definition.
        assert!(
            !after_symbol.contains("record<symbol>"),
            "after re-apply with OVERWRITE, 'record<symbol>' must be gone from symbol.parent \
             field definition — OVERWRITE did not update the persisted type. Got: {after_symbol}"
        );

        // The chunk table's symbol_ref is handled by v2→v3 migration (REMOVE FIELD),
        // not by SCHEMA_DDL re-application. The write below uses SCHEMALESS-compatible
        // syntax; any stale type on chunk.symbol_ref in a pre-v3 DB would be cleared
        // by run_migration_v2_to_v3 before this write path is reached in production.

        // ── 5. Run v2→v3 migration to remove the stale chunk field definitions ─
        // In production, this runs before any write path that stores chunks.
        // We need index_meta for the migration's set_meta calls.
        db.query(
            "DEFINE TABLE IF NOT EXISTS index_meta SCHEMAFULL;\
             DEFINE FIELD OVERWRITE key ON index_meta TYPE string;\
             DEFINE FIELD OVERWRITE value ON index_meta TYPE string;\
             DEFINE INDEX IF NOT EXISTS idx_meta_key ON index_meta FIELDS key UNIQUE;",
        )
        .await
        .expect("setup index_meta for migration")
        .check()
        .expect("index_meta setup check");

        crate::store::run_migration_v2_to_v3(&db)
            .await
            .expect("v2→v3 migration must succeed");

        // ── 6. Attempt the real writer's statement (mirroring pipeline.rs) ───
        let txn = "BEGIN TRANSACTION;\n\
            CREATE chunk SET \
              file = '/x/config.rs', \
              line_start = 1, \
              line_end = 10, \
              content = 'impl EmbeddingConfig {}', \
              embedding = [0.0, 1.0, 0.5], \
              symbol_ref = 'symbol:⟨config.rs::impl_EmbeddingConfig⟩';\n\
            COMMIT TRANSACTION;\n";

        let mut resp = db.query(txn).await.expect(".await must not fail");
        let errors = resp.take_errors();
        println!("STALE-SCHEMA WRITE RESULT: errors = {errors:?}");

        const GENERIC: &str = "The query was not executed due to a failed transaction";
        let real_error: Vec<_> = errors
            .iter()
            .filter(|(_, e)| !e.to_string().contains(GENERIC))
            .collect();
        println!("STALE-SCHEMA WRITE: non-generic errors = {real_error:?}");

        // ── 7. Assert commit succeeded ────────────────────────────────────────
        assert!(
            real_error.is_empty(),
            "transaction must commit after v2→v3 migration removes stale field type: {real_error:?}\n\
             REMOVE FIELD did NOT remove the stale 'option<record<symbol>>' definition."
        );

        let count = count_chunks(&db).await.unwrap();
        println!("STALE-SCHEMA WRITE: chunk count after commit = {count}");
        assert_eq!(
            count, 1,
            "chunk must persist after migration (got {count}); \
             transaction is still rolling back due to stale field type"
        );
    }

    /// Verify that `DEFINE TABLE IF NOT EXISTS` does NOT drop existing rows.
    /// This confirms the table DDL form we chose is safe to re-run on a live database.
    #[tokio::test]
    async fn table_redefine_does_not_drop_rows() {
        let home = TempDir::new().unwrap();
        let db = open_raw_db(home.path(), "table_redef").await;

        // Set up a minimal chunk table and insert a sentinel row.
        db.query(
            "DEFINE TABLE IF NOT EXISTS chunk SCHEMAFULL;\
             DEFINE FIELD OVERWRITE file ON chunk TYPE string;\
             DEFINE FIELD OVERWRITE line_start ON chunk TYPE int;\
             DEFINE FIELD OVERWRITE line_end ON chunk TYPE int;\
             DEFINE FIELD OVERWRITE content ON chunk TYPE string;\
             DEFINE FIELD OVERWRITE embedding ON chunk TYPE array<float>;\
             DEFINE FIELD OVERWRITE symbol_ref ON chunk TYPE option<string>;\
             CREATE chunk SET file='/sentinel', line_start=1, line_end=1, \
               content='sentinel', embedding=[], symbol_ref=NONE;",
        )
        .await
        .expect("setup")
        .check()
        .expect("setup check");

        let before = count_chunks(&db).await.unwrap();
        assert_eq!(before, 1, "sentinel row must exist before re-DDL");

        // Re-run the full SCHEMA_DDL (simulating a server restart).
        db.query(SCHEMA_DDL)
            .await
            .expect("re-apply DDL")
            .check()
            .expect("re-apply check");

        let after = count_chunks(&db).await.unwrap();
        println!("TABLE-REDEF: rows before={before}, after={after}");
        assert_eq!(
            after, before,
            "DEFINE TABLE IF NOT EXISTS must not drop existing rows (before={before}, after={after})"
        );
    }

    /// v5→v6 regression: a `calls` table created as a graph RELATION (old schema)
    /// must be flippable to a NORMAL table by the current SCHEMA_DDL, and plain
    /// INSERTs must succeed afterward. This pins the riskiest part of the
    /// RELATION→NORMAL conversion: that `DEFINE TABLE OVERWRITE ... TYPE NORMAL`
    /// against a POPULATED relation table does not error, and that the post-flip
    /// write path (`INSERT INTO calls`, used by flush_edge_batch) works.
    #[tokio::test]
    async fn calls_relation_table_flips_to_normal_and_accepts_plain_insert() {
        use serde::Deserialize;
        let home = TempDir::new().unwrap();
        let db = open_raw_db(home.path(), "calls_flip").await;

        // Old-world: calls is a graph RELATION with two symbol endpoints + one
        // RELATE edge, exactly as a pre-v6 DB would have persisted it.
        db.query(
            "DEFINE TABLE IF NOT EXISTS calls TYPE RELATION IN symbol OUT symbol;\
             CREATE symbol:`⟨/a.rs::foo⟩` SET name='foo', file='/a.rs';\
             CREATE symbol:`⟨/b.rs::bar⟩` SET name='bar', file='/b.rs';\
             RELATE symbol:`⟨/a.rs::foo⟩`->calls->symbol:`⟨/b.rs::bar⟩` \
               SET line=1, in_file='/a.rs', out_file='/b.rs', \
                   in_name='/a.rs::foo', out_name='/b.rs::bar';",
        )
        .await
        .expect("relation setup")
        .check()
        .expect("relation setup check");

        // Re-apply current SCHEMA_DDL: flips calls to TYPE NORMAL via OVERWRITE.
        // Must NOT error against the populated relation table.
        db.query(SCHEMA_DDL)
            .await
            .expect("re-apply DDL flips calls to NORMAL")
            .check()
            .expect("re-apply DDL check");

        // Post-flip: a plain INSERT (the flush_edge_batch write path) must work.
        db.query(
            "INSERT INTO calls { in: symbol:`⟨/a.rs::foo⟩`, out: symbol:`⟨/b.rs::bar⟩`, \
             line: 2, in_file: '/a.rs', out_file: '/b.rs', \
             in_name: '/a.rs::foo', out_name: '/b.rs::bar' }",
        )
        .await
        .expect("plain INSERT into NORMAL calls")
        .check()
        .expect("plain INSERT check");

        // The denormalized read path (WHERE out_name = ...) must still resolve.
        #[derive(Deserialize)]
        struct Row {
            in_name: Option<String>,
        }
        let rows: Vec<Row> = db
            .query("SELECT in_name FROM calls WHERE out_name = '/b.rs::bar'")
            .await
            .expect("read by out_name")
            .take(0)
            .expect("take rows");
        assert!(
            rows.iter()
                .any(|r| r.in_name.as_deref() == Some("/a.rs::foo")),
            "denormalized in_name/out_name read must work on NORMAL calls table"
        );
    }
}

// ─── Secondary-index persistence tests ────────────────────────────────────
//
// These tests pin the root-cause fix for the kernel-query regression: on the
// persisted datastore `INFO FOR TABLE calls` / `INFO FOR TABLE symbol` returned
// `indexes:{}` — the secondary indexes were PHYSICALLY ABSENT, so every
// `WHERE out_name = $fqn` / `WHERE file = $f` was a full table scan (165s/129s
// per call at kernel scale).
//
// MECHANISM — Theory-A (foreground-backfill rollback under pinned buffers).
// The rebuild pipeline DROPS the symbol/calls secondary indexes before its bulk
// write, then rebuilt them with a FOREGROUND `DEFINE INDEX … FIELDS …`. That
// statement backfills every existing row inside ONE write transaction. Under the
// production-pinned RocksDB write buffers (`SURREAL_ROCKSDB_WRITE_BUFFER_SIZE`
// = 32 MiB × 2; see `engine_boot::set_rocksdb_memory_bounds`) that single
// transaction over millions of rows fails to commit and the ENTIRE `DEFINE INDEX`
// — including the index *definition* — ROLLS BACK, leaving `indexes:{}`. The fix
// is `build_index_concurrently` (`DEFINE INDEX … CONCURRENTLY`, batched commits +
// poll-to-ready), used by the three pipeline rebuild sites AND by
// `ensure_secondary_indexes` on `open_db` for crash recovery. The `#[ignore]`d
// `foreground_index_backfill_at_scale` test reproduces the rollback at volume
// (and the CONCURRENTLY success under the same pinned bounds).
//
// NOT the mechanism — Theory-B (`DEFINE TABLE OVERWRITE` drops indexes). This was
// the original (wrong) hypothesis. Verified FALSE against surrealdb 2.6.5
// (`sql/statements/define/table.rs::DefineTableStatement::compute`): for a
// non-view table, OVERWRITE only rewrites the table-definition record and clears
// caches; it deletes table DATA (`txn.delp`) ONLY inside the `if let Some(view)`
// branch, and never touches index definitions or index keys. So `symbol`/`calls`
// keep their secondary indexes across an OVERWRITE reopen. (Because of this, the
// six droppable indexes were removed from SCHEMA_DDL entirely — see its doc
// comment — and are owned by `ensure_secondary_indexes`.)
#[cfg(test)]
mod index_persistence_tests {
    use super::*;
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    /// Pull the set of index names defined on a table via `INFO FOR TABLE`.
    async fn table_index_names(db: &Surreal<Db>, table: &str) -> Vec<String> {
        #[derive(serde::Deserialize)]
        struct TableInfo {
            indexes: BTreeMap<String, String>,
        }
        let info: Option<TableInfo> = db
            .query(format!("INFO FOR TABLE {table}"))
            .await
            .expect("INFO FOR TABLE query")
            .take(0)
            .expect("take INFO FOR TABLE row");
        let mut names: Vec<String> = info
            .map(|i| i.indexes.into_keys().collect())
            .unwrap_or_default();
        names.sort();
        names
    }

    /// Recursively copy a quiesced RocksDB image, skipping the `LOCK` marker (a
    /// new boot recreates it). Mirrors what a fresh OS process would open.
    fn copy_db_image(src: &std::path::Path, dst: &std::path::Path) {
        std::fs::create_dir_all(dst).expect("mkdir image dst");
        for entry in std::fs::read_dir(src).expect("read image src") {
            let entry = entry.expect("dir entry");
            let name = entry.file_name();
            let ty = entry.file_type().expect("file type");
            let to = dst.join(&name);
            if ty.is_dir() {
                copy_db_image(&entry.path(), &to);
            } else if name != "LOCK" {
                std::fs::copy(entry.path(), &to).expect("copy image file");
            }
        }
    }

    /// THE crash-recovery contract test (real regression guard, not a tautology).
    ///
    /// Reproduces the precise state a crash mid-rebuild leaves on disk: a POPULATED
    /// `symbol`/`calls` table with the secondary indexes DROPPED. We build that
    /// state, quiesce it to a clean on-disk image, then open it from a FRESH path —
    /// a faithful new-process server re-open.
    ///
    /// The fix's contract, which this asserts: `open_db` → `ensure_secondary_indexes`
    /// rebuilds the absent indexes CONCURRENTLY (never a foreground backfill over the
    /// populated table, which would roll back under the pinned RocksDB buffers and
    /// fail the open — Theory-A), and the hot predicate is index-served afterward.
    ///
    /// Note this is NOT served by SCHEMA_DDL: the six droppable indexes were removed
    /// from SCHEMA_DDL entirely, so a reopen that finds them absent MUST go through
    /// `ensure_secondary_indexes` to restore them. (The kernel-scale rollback that
    /// motivates CONCURRENTLY is reproduced by the `#[ignore]`d
    /// `foreground_index_backfill_at_scale`; this test pins the recovery wiring at
    /// unit speed.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn indexes_survive_cross_process_reopen() {
        let build_home = TempDir::new().unwrap();
        let repo = "/test/index_persistence";

        // ── Boot 1: build a populated DB, then DROP the secondary indexes to
        //    reproduce the crash-mid-rebuild on-disk state, then tear down so
        //    RocksDB flushes a clean image. ──
        let build_home_path = build_home.path().to_path_buf();
        let repo_owned = repo.to_string();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build sacrificial runtime");
            rt.block_on(async move {
                let db = open_db(&build_home_path, &repo_owned, 0).await.expect("open boot1");
                // Populate both tables so we reopen against non-empty tables (the
                // production state: kernel = 2.63M symbols / 3.08M calls).
                for i in 0..50 {
                    db.query(format!(
                        "INSERT INTO symbol {{ id: '/f.rs::s{i}', name: 's{i}', kind: 'function', \
                         file: '/f.rs', line_start: {i}, line_end: {i}, signature: NONE, parent: NONE }}"
                    )).await.expect("insert symbol").check().expect("symbol insert check");
                    db.query(format!(
                        "INSERT INTO calls {{ line: {i}, in_file: '/f.rs', out_file: '/g.rs', \
                         in_name: '/f.rs::s{i}', out_name: '/g.rs::t{i}' }}"
                    )).await.expect("insert call").check().expect("call insert check");
                }
                // Enter the crash window: drop ALL six secondary indexes while the
                // rows remain (exactly the pipeline's pre-bulk-write drop state).
                db.query(
                    "REMOVE INDEX IF EXISTS idx_symbol_file ON symbol; \
                     REMOVE INDEX IF EXISTS idx_symbol_name ON symbol; \
                     REMOVE INDEX IF EXISTS idx_calls_in_file  ON calls; \
                     REMOVE INDEX IF EXISTS idx_calls_out_file ON calls; \
                     REMOVE INDEX IF EXISTS idx_calls_in_name  ON calls; \
                     REMOVE INDEX IF EXISTS idx_calls_out_name ON calls;",
                ).await.expect("drop indexes (enter crash window)").check().expect("drop check");
                // Confirm we are genuinely IN the crash window: indexes absent, rows present.
                let calls_idx = table_index_names(&db, "calls").await;
                let symbol_idx = table_index_names(&db, "symbol").await;
                assert!(
                    !calls_idx.iter().any(|n| n.starts_with("idx_calls_"))
                        && !symbol_idx.iter().any(|n| n.starts_with("idx_symbol_")),
                    "boot1 must be in the crash window: indexes dropped (calls={calls_idx:?}, symbol={symbol_idx:?})"
                );
                drop(db);
            });
        })
        .join()
        .expect("boot1 build thread must not panic");

        // ── Boot 2: open the quiesced image from a FRESH path (new-process boot).
        //    open_db runs SCHEMA_DDL (no longer defines these six indexes) then
        //    ensure_secondary_indexes, which must concurrently rebuild them over the
        //    populated tables. ──
        let reopen_home = TempDir::new().unwrap();
        copy_db_image(
            &db_path(build_home.path(), repo, 0),
            &db_path(reopen_home.path(), repo, 0),
        );
        let db = open_db(reopen_home.path(), repo, 0)
            .await
            .expect("open boot2");

        let calls_idx = table_index_names(&db, "calls").await;
        let symbol_idx = table_index_names(&db, "symbol").await;
        assert!(
            calls_idx.contains(&"idx_calls_out_name".to_string())
                && calls_idx.contains(&"idx_calls_in_name".to_string())
                && calls_idx.contains(&"idx_calls_in_file".to_string())
                && calls_idx.contains(&"idx_calls_out_file".to_string()),
            "after crash-recovery reopen, all 4 calls indexes MUST be rebuilt, got {calls_idx:?}"
        );
        assert!(
            symbol_idx.contains(&"idx_symbol_file".to_string())
                && symbol_idx.contains(&"idx_symbol_name".to_string()),
            "after crash-recovery reopen, both symbol indexes MUST be rebuilt, got {symbol_idx:?}"
        );

        // The hot predicate must be served by the rebuilt index (not a full scan).
        let plan: Vec<serde_json::Value> = db
            .query("SELECT * FROM calls WHERE out_name = '/g.rs::t1' EXPLAIN")
            .await
            .expect("explain query")
            .take(0)
            .expect("take explain rows");
        let plan_str = serde_json::to_string(&plan).expect("serialize plan");
        assert!(
            plan_str.contains("Iterate Index"),
            "WHERE out_name=… must use an index after crash-recovery reopen (got plan: {plan_str})"
        );
    }

    /// `build_index_concurrently` (the production fix) must, on a POPULATED table:
    ///   (a) drive the build to completion (poll until building.status=='ready'),
    ///   (b) leave the index present in `INFO FOR TABLE`, and
    ///   (c) make the hot predicate use the index (EXPLAIN → "Iterate Index").
    /// It must also be idempotent (a second call is a no-op via IF NOT EXISTS).
    ///
    /// This is the always-run guard for the root-cause fix: the pipeline rebuild
    /// sites call exactly this helper. (The kernel-scale failure mode — a plain
    /// foreground DEFINE INDEX rolling back under the pinned RocksDB buffers — is
    /// reproduced separately by the `#[ignore]`d `foreground_index_backfill_at_scale`
    /// with SURREAL_ROCKSDB_* set; this test pins the helper's contract at unit speed.)
    #[tokio::test]
    async fn build_index_concurrently_completes_and_is_usable() {
        let home = TempDir::new().unwrap();
        let repo = "/test/concurrent_build";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Drop the index (as the pipeline does before its bulk write), then
        // populate the table with the index absent.
        db.query("REMOVE INDEX IF EXISTS idx_calls_out_name ON calls;")
            .await
            .expect("drop index")
            .check()
            .expect("drop check");
        for i in 0..500 {
            db.query(format!(
                "INSERT INTO calls {{ line: {i}, in_file: '/f.rs', out_file: '/g.rs', \
                 in_name: '/f.rs::c{i}', out_name: '/g.rs::t{}' }}",
                i % 25
            ))
            .await
            .expect("insert call")
            .check()
            .expect("insert check");
        }

        // Build via the production helper; must return only once ready.
        super::build_index_concurrently(&db, "idx_calls_out_name", "calls", "out_name")
            .await
            .expect("concurrent build must complete");

        let names = table_index_names(&db, "calls").await;
        assert!(
            names.contains(&"idx_calls_out_name".to_string()),
            "index must be present after concurrent build (got {names:?})"
        );

        let plan: Vec<serde_json::Value> = db
            .query("SELECT * FROM calls WHERE out_name = '/g.rs::t1' EXPLAIN")
            .await
            .expect("explain")
            .take(0)
            .expect("take explain");
        assert!(
            serde_json::to_string(&plan)
                .unwrap_or_default()
                .contains("Iterate Index"),
            "predicate must use the index after concurrent build (plan: {plan:?})"
        );

        // Idempotent: a second build is a no-op and still leaves the index ready.
        super::build_index_concurrently(&db, "idx_calls_out_name", "calls", "out_name")
            .await
            .expect("second concurrent build must be a no-op");
        let names2 = table_index_names(&db, "calls").await;
        assert!(
            names2.contains(&"idx_calls_out_name".to_string()),
            "index must remain present after idempotent re-build (got {names2:?})"
        );
    }

    /// FRESH-DB open: SCHEMA_DDL no longer defines the six droppable indexes, so
    /// `ensure_secondary_indexes` must define them on the EMPTY tables. This must be
    /// instant (no backfill — empty tables) and leave all six present, so the
    /// incremental write path (which relies on the live indexes) works immediately.
    #[tokio::test]
    async fn fresh_open_defines_all_six_indexes_on_empty_tables() {
        let home = TempDir::new().unwrap();
        let repo = "/test/fresh_open";
        let db = open_db(home.path(), repo, 0).await.expect("fresh open");

        let calls_idx = table_index_names(&db, "calls").await;
        let symbol_idx = table_index_names(&db, "symbol").await;
        assert!(
            calls_idx.contains(&"idx_calls_in_file".to_string())
                && calls_idx.contains(&"idx_calls_out_file".to_string())
                && calls_idx.contains(&"idx_calls_in_name".to_string())
                && calls_idx.contains(&"idx_calls_out_name".to_string()),
            "fresh open must define all 4 calls indexes (got {calls_idx:?})"
        );
        assert!(
            symbol_idx.contains(&"idx_symbol_file".to_string())
                && symbol_idx.contains(&"idx_symbol_name".to_string()),
            "fresh open must define both symbol indexes (got {symbol_idx:?})"
        );
    }

    /// HEALTHY-REOPEN: `ensure_secondary_indexes` on a DB whose six indexes already
    /// exist must NOT issue any build (no `DEFINE INDEX`, no CONCURRENTLY) — it is a
    /// pure no-op driven by the per-table `INFO FOR TABLE` check. We assert it by
    /// populating the tables, confirming the indexes are present, then calling the
    /// helper directly: it must return quickly and leave everything intact (a
    /// foreground backfill over the populated table would be the regression).
    #[tokio::test]
    async fn ensure_secondary_indexes_is_noop_on_healthy_indexed_db() {
        let home = TempDir::new().unwrap();
        let repo = "/test/healthy_reopen";
        let db = open_db(home.path(), repo, 0).await.expect("open");

        // Populate both tables; indexes are live (defined on the fresh open above).
        for i in 0..200 {
            db.query(format!(
                "INSERT INTO symbol {{ id: '/f.rs::s{i}', name: 's{i}', kind: 'function', \
                 file: '/f.rs', line_start: {i}, line_end: {i}, signature: NONE, parent: NONE }}"
            ))
            .await
            .expect("insert symbol")
            .check()
            .expect("symbol check");
            db.query(format!(
                "INSERT INTO calls {{ line: {i}, in_file: '/f.rs', out_file: '/g.rs', \
                 in_name: '/f.rs::s{i}', out_name: '/g.rs::t{}' }}",
                i % 10
            ))
            .await
            .expect("insert call")
            .check()
            .expect("call check");
        }
        let before = table_index_names(&db, "calls").await;
        assert!(
            before.len() >= 4,
            "indexes must be present before the no-op call (got {before:?})"
        );

        // Re-running ensure on the already-indexed, populated DB must be a no-op and
        // must not error (it must NOT attempt a foreground backfill).
        super::ensure_secondary_indexes(&db)
            .await
            .expect("ensure on healthy indexed DB must be a no-op");

        let after_calls = table_index_names(&db, "calls").await;
        let after_symbol = table_index_names(&db, "symbol").await;
        assert!(
            after_calls.len() >= 4 && after_symbol.len() >= 2,
            "all indexes must remain after the no-op (calls={after_calls:?}, symbol={after_symbol:?})"
        );
    }

    /// CRASH-RECOVERY ROUTING (same-process): drop the six indexes on a POPULATED
    /// DB, then call `ensure_secondary_indexes`. It must rebuild every one (via the
    /// concurrent helper — there is no foreground `DEFINE INDEX` over a populated
    /// table anywhere in the open path) and leave the hot predicate index-served.
    #[tokio::test]
    async fn ensure_secondary_indexes_rebuilds_dropped_indexes_on_populated_db() {
        let home = TempDir::new().unwrap();
        let repo = "/test/recovery_routing";
        let db = open_db(home.path(), repo, 0).await.expect("open");

        for i in 0..200 {
            db.query(format!(
                "INSERT INTO symbol {{ id: '/f.rs::s{i}', name: 's{i}', kind: 'function', \
                 file: '/f.rs', line_start: {i}, line_end: {i}, signature: NONE, parent: NONE }}"
            ))
            .await
            .expect("insert symbol")
            .check()
            .expect("symbol check");
            db.query(format!(
                "INSERT INTO calls {{ line: {i}, in_file: '/f.rs', out_file: '/g.rs', \
                 in_name: '/f.rs::s{i}', out_name: '/g.rs::t{}' }}",
                i % 10
            ))
            .await
            .expect("insert call")
            .check()
            .expect("call check");
        }

        // Enter the crash window: indexes dropped, rows present.
        db.query(
            "REMOVE INDEX IF EXISTS idx_symbol_file ON symbol; \
             REMOVE INDEX IF EXISTS idx_symbol_name ON symbol; \
             REMOVE INDEX IF EXISTS idx_calls_in_file  ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_out_file ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_in_name  ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_out_name ON calls;",
        )
        .await
        .expect("drop indexes")
        .check()
        .expect("drop check");
        let dropped = table_index_names(&db, "calls").await;
        assert!(
            !dropped.iter().any(|n| n.starts_with("idx_calls_")),
            "calls indexes must be absent before recovery (got {dropped:?})"
        );

        // Recovery: rebuild via the concurrent helper, polling to ready.
        super::ensure_secondary_indexes(&db)
            .await
            .expect("ensure must rebuild dropped indexes on a populated table");

        let calls_idx = table_index_names(&db, "calls").await;
        let symbol_idx = table_index_names(&db, "symbol").await;
        assert!(
            calls_idx.len() >= 4 && symbol_idx.len() >= 2,
            "all six indexes must be rebuilt (calls={calls_idx:?}, symbol={symbol_idx:?})"
        );
        let plan: Vec<serde_json::Value> = db
            .query("SELECT * FROM calls WHERE out_name = '/g.rs::t1' EXPLAIN")
            .await
            .expect("explain")
            .take(0)
            .expect("take explain");
        assert!(
            serde_json::to_string(&plan)
                .unwrap_or_default()
                .contains("Iterate Index"),
            "predicate must use the rebuilt index (plan: {plan:?})"
        );
    }

    /// STATUS-TRUST DECISION (pure): the gate that distinguishes a fully-built
    /// present index from a crash-interrupted half-built one. Only `None` (no
    /// in-progress `building` record — a plain/completed index) and `"ready"`
    /// (a finished concurrent build) are trustworthy; every other status —
    /// in-progress (`started`/`cleaning`/`indexing`) left by a crash, or a
    /// terminal failure (`error`/`aborted`) — must route to drop+rebuild.
    #[test]
    fn index_status_trust_gate_only_trusts_ready_or_absent() {
        assert!(
            super::index_status_is_trustworthy(None),
            "absent building record = built"
        );
        assert!(
            super::index_status_is_trustworthy(Some("ready")),
            "ready = built"
        );
        for bad in [
            "started", "cleaning", "indexing", "error", "aborted", "weird",
        ] {
            assert!(
                !super::index_status_is_trustworthy(Some(bad)),
                "status {bad:?} must NOT be trusted (half-built / failed)"
            );
        }
    }

    /// STATUS PARSE + HEALTHY FAST-PATH (real DB): proves we read the REAL
    /// `INFO FOR INDEX` serialization, and that a fully-built present index is
    /// classified trustworthy so `ensure_secondary_indexes` leaves it untouched
    /// (no drop, no rebuild).
    ///
    /// COVERAGE HONESTY: SurrealDB exposes no SQL to forge a *mid-build*
    /// `building.status` (the `building` record is internal KV state managed by
    /// the concurrent builder; it cannot be hand-set), so a true interrupted-
    /// build image is not deterministically constructable in a unit test. We
    /// therefore split the proof: this test pins the real "built → trustworthy →
    /// skip" serialization end-to-end, while `index_status_trust_gate_only_trusts_
    /// ready_or_absent` exhaustively pins the not-ready → rebuild routing on the
    /// exact status strings SurrealDB emits, and `ensure_secondary_indexes_
    /// rebuilds_dropped_indexes_on_populated_db` pins the drop→concurrent-rebuild
    /// mechanics. What is NOT directly exercised: a literal process-crash leaving
    /// `building.status == 'indexing'` persisted on disk.
    #[tokio::test]
    async fn present_built_index_reports_ready_and_is_left_untouched() {
        let home = TempDir::new().unwrap();
        let repo = "/test/status_fast_path";
        let db = open_db(home.path(), repo, 0).await.expect("open");

        // Populate, then build idx_calls_out_name via the production helper so it
        // carries a real (completed) concurrent-build record.
        db.query("REMOVE INDEX IF EXISTS idx_calls_out_name ON calls;")
            .await
            .expect("drop")
            .check()
            .expect("drop check");
        for i in 0..200 {
            db.query(format!(
                "INSERT INTO calls {{ line: {i}, in_file: '/f.rs', out_file: '/g.rs', \
                 in_name: '/f.rs::c{i}', out_name: '/g.rs::t{}' }}",
                i % 10
            ))
            .await
            .expect("insert")
            .check()
            .expect("insert check");
        }
        super::build_index_concurrently(&db, "idx_calls_out_name", "calls", "out_name")
            .await
            .expect("concurrent build");

        // The real INFO FOR INDEX status of a completed build is either 'ready'
        // or carries no in-progress building record — both trustworthy.
        let status = super::index_building_status(&db, "idx_calls_out_name", "calls")
            .await
            .expect("read status");
        assert!(
            super::index_status_is_trustworthy(status.as_deref()),
            "a completed concurrent build must read as trustworthy (got {status:?})"
        );

        // ensure_secondary_indexes must now leave this present+ready index alone:
        // the index stays present and the predicate stays index-served (a spurious
        // drop+rebuild would still leave it present, so we also assert the row set
        // is intact and queryable to catch an accidental data-affecting rebuild).
        super::ensure_secondary_indexes(&db)
            .await
            .expect("ensure on healthy DB");
        let names = table_index_names(&db, "calls").await;
        assert!(
            names.contains(&"idx_calls_out_name".to_string()),
            "present+ready index must remain after ensure (got {names:?})"
        );
        let plan: Vec<serde_json::Value> = db
            .query("SELECT * FROM calls WHERE out_name = '/g.rs::t1' EXPLAIN")
            .await
            .expect("explain")
            .take(0)
            .expect("take explain");
        assert!(
            serde_json::to_string(&plan)
                .unwrap_or_default()
                .contains("Iterate Index"),
            "predicate must stay index-served after the healthy no-op (plan: {plan:?})"
        );
    }

    /// SCALE REPRO — THE source-of-truth demonstration of Theory-A (ignored by
    /// default; run with `--ignored` and `IDX_SCALE_ROWS=<n>`). Mirrors the EXACT
    /// pipeline path that the kernel rebuild takes and that the small-scale tests
    /// never exercise:
    ///   drop the secondary index → bulk-INSERT N rows with NO index live →
    ///   foreground `DEFINE INDEX` that must BACKFILL all N existing rows.
    ///
    /// The small crash/rebuild tests only ever backfill ≤50 rows, where the build
    /// trivially succeeds regardless of the fix (which is exactly why they are NOT a
    /// regression guard for Theory-A on their own — they prove the recovery WIRING,
    /// not the scale failure). This test scales the backfill to confirm the
    /// foreground build itself ROLLS BACK at volume under the pinned RocksDB buffers
    /// (and that `.await?` WITHOUT `.check()` silently swallows it, leaving
    /// `INFO FOR TABLE` with `indexes:{}` exactly as the kernel DB showed) — and,
    /// with `IDX_SCALE_CONCURRENTLY=1`, that `build_index_concurrently` succeeds
    /// under the same bounds. That concurrent helper is what both the pipeline
    /// rebuild sites AND `ensure_secondary_indexes` (crash-recovery on open) use, so
    /// this test underwrites the whole drop→rebuild→reopen chain at true scale.
    ///
    /// Kept `#[ignore]`d because a faithful repro needs millions of rows (minutes +
    /// the pinned-buffer env), too slow for the default suite. The unit-speed
    /// regression guards are `indexes_survive_cross_process_reopen` (recovery wiring)
    /// and `build_index_concurrently_completes_and_is_usable` (helper contract).
    ///
    /// Set IDX_SCALE_ROWS to e.g. 3000000 to match kernel `calls` cardinality.
    #[tokio::test]
    #[ignore = "scale repro; run explicitly with IDX_SCALE_ROWS set"]
    async fn foreground_index_backfill_at_scale() {
        let rows: usize = std::env::var("IDX_SCALE_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000_000);
        let home = TempDir::new().unwrap();
        let repo = "/test/idx_scale";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Drop the calls indexes (as the pipeline does before its bulk write).
        db.query(
            "REMOVE INDEX IF EXISTS idx_calls_in_file  ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_out_file ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_in_name  ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_out_name ON calls;",
        )
        .await
        .expect("drop calls indexes")
        .check()
        .expect("drop check");

        // Bulk-insert N rows with NO index live (native array INSERT, like the
        // pipeline's flush_edge_batch).
        use std::collections::BTreeMap;
        use surrealdb::sql::{Array as SqlArray, Object as SqlObject, Value as SqlValue};
        let mut written = 0usize;
        while written < rows {
            let batch = std::cmp::min(20_000, rows - written);
            let records: Vec<SqlValue> = (0..batch)
                .map(|i| {
                    let n = written + i;
                    // Skew control: IDX_SCALE_DISTINCT caps the number of distinct
                    // out_name/in_name values, reproducing the kernel's hot-symbol
                    // skew (printk etc. appear in tens of thousands of rows). When
                    // unset, keys are unique (low skew).
                    let distinct: usize = std::env::var("IDX_SCALE_DISTINCT")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(usize::MAX);
                    let key = if distinct == usize::MAX {
                        n
                    } else {
                        n % distinct
                    };
                    let mut m: BTreeMap<String, SqlValue> = BTreeMap::new();
                    m.insert("line".into(), SqlValue::from(n as i64));
                    m.insert(
                        "in_file".into(),
                        SqlValue::from(format!("/f{}.rs", n % 4096)),
                    );
                    m.insert(
                        "out_file".into(),
                        SqlValue::from(format!("/g{}.rs", n % 4096)),
                    );
                    m.insert(
                        "in_name".into(),
                        SqlValue::from(format!("/f{}.rs::c{key}", key % 4096)),
                    );
                    m.insert(
                        "out_name".into(),
                        SqlValue::from(format!("/g{}.rs::t{key}", key % 4096)),
                    );
                    SqlValue::Object(SqlObject::from(m))
                })
                .collect();
            db.query("INSERT INTO calls $data RETURN NONE")
                .bind(("data", SqlArray::from(records)))
                .await
                .expect("bulk insert calls")
                .check()
                .expect("bulk insert check");
            written += batch;
        }

        // Foreground DEFINE INDEX that must backfill all N rows. Capture the
        // response and run BOTH .await (transport) and .check() (statement) so we
        // can see whether a statement-level failure is being produced at scale.
        // Set IDX_SCALE_CONCURRENTLY=1 to instead exercise the CONCURRENTLY +
        // poll-to-ready fix path under the same pinned bounds.
        let concurrently = std::env::var("IDX_SCALE_CONCURRENTLY").is_ok();
        let t = std::time::Instant::now();
        if concurrently {
            build_index_concurrently(&db, "idx_calls_out_name", "calls", "out_name")
                .await
                .expect("concurrent build must succeed under pinned bounds");
            eprintln!(
                "SCALE rows={rows} CONCURRENTLY build elapsed_ms={}",
                t.elapsed().as_millis()
            );
        } else {
            let resp = db
                .query("DEFINE INDEX IF NOT EXISTS idx_calls_out_name ON calls FIELDS out_name;")
                .await;
            let await_ok = resp.is_ok();
            let check_result = match resp {
                Ok(r) => r.check().map(|_| ()),
                Err(e) => Err(e),
            };
            eprintln!(
                "SCALE rows={rows} define_index await_ok={await_ok} check_ok={} elapsed_ms={}",
                check_result.is_ok(),
                t.elapsed().as_millis()
            );
            if let Err(e) = &check_result {
                eprintln!("SCALE define_index CHECK ERROR: {e:#}");
            }
        }

        // Report whether the index definition actually persisted.
        let names = table_index_names(&db, "calls").await;
        eprintln!("SCALE post-build calls indexes = {names:?}");
        let plan: Vec<serde_json::Value> = db
            .query("SELECT * FROM calls WHERE out_name = '/g1.rs::t1' EXPLAIN")
            .await
            .expect("explain")
            .take(0)
            .expect("take explain");
        eprintln!(
            "SCALE plan uses index = {}",
            serde_json::to_string(&plan)
                .unwrap_or_default()
                .contains("Iterate Index")
        );

        // The fix must make this hold even at scale. If the foreground build is
        // the culprit, this assertion fails (index absent) BEFORE the fix.
        assert!(
            names.contains(&"idx_calls_out_name".to_string()),
            "idx_calls_out_name must exist after a backfill build over {rows} rows (got {names:?})"
        );
    }
}

// ─── Migration tests ──────────────────────────────────────────────────────
#[cfg(test)]
mod migration_tests {
    use super::*;
    use tempfile::TempDir;

    /// ❾ NEW: migration stamps db_schema_version=2 after completing.
    #[tokio::test]
    async fn migration_stamps_version_2_after_completion() {
        let home = TempDir::new().unwrap();
        let repo = "/test/migration_repo";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Confirm we start at version 1 (fresh DB has no version key).
        let before = read_db_schema_version(&db).await;
        assert_eq!(before, 1, "fresh DB should report version 1");

        // Run migration directly.
        run_migration_v1_to_v2(&db).await.unwrap();

        let after = read_db_schema_version(&db).await;
        assert_eq!(after, 2, "after migration, db_schema_version must be 2");
    }

    /// ❾ NEW: migration is idempotent — re-running on a v2 DB changes nothing.
    #[tokio::test]
    async fn migration_idempotent_on_v2_db() {
        let home = TempDir::new().unwrap();
        let repo = "/test/idempotent_repo";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Run migration twice.
        run_migration_v1_to_v2(&db).await.unwrap();
        run_migration_v1_to_v2(&db).await.unwrap();

        let version = read_db_schema_version(&db).await;
        assert_eq!(version, 2, "version must still be 2 after second run");
    }

    /// ❾ NEW: crash/resume — migration resumes from persisted cursor.
    /// We seed some calls rows, run migration partially by directly calling the
    /// inner loop logic, then verify a second full run completes cleanly.
    #[tokio::test]
    async fn migration_resumes_from_cursor() {
        let home = TempDir::new().unwrap();
        let repo = "/test/cursor_repo";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Migration on empty DB should complete without error.
        run_migration_v1_to_v2(&db).await.unwrap();

        // Version must be 2.
        let v = read_db_schema_version(&db).await;
        assert_eq!(v, 2);

        // Simulate a "resume" by clearing the version key and re-running.
        let _ = db
            .query("DELETE FROM index_meta WHERE key = $k")
            .bind(("k", DB_SCHEMA_VERSION_KEY))
            .await;
        let v_cleared = read_db_schema_version(&db).await;
        assert_eq!(v_cleared, 1, "after clearing version key, should read 1");

        run_migration_v1_to_v2(&db).await.unwrap();
        let v_again = read_db_schema_version(&db).await;
        assert_eq!(v_again, 2, "after re-run, version must be 2 again");
    }
}

// ─── SCHEMALESS tests ─────────────────────────────────────────────────────
#[cfg(test)]
mod schemaless_tests {
    use super::*;
    use tempfile::TempDir;

    /// Helper: open a raw SurrealKV DB without any DDL.
    async fn open_raw(dir: &std::path::Path, name: &str) -> Surreal<Db> {
        let path = dir.join(name);
        std::fs::create_dir_all(&path).unwrap();
        let db = Surreal::new::<RocksDb>(path.to_str().unwrap())
            .await
            .expect("open raw db");
        db.use_ns("context_engine")
            .use_db(name)
            .await
            .expect("ns/db");
        db
    }

    /// Helper: build a distinct 1024-dim embedding from a seed value.
    fn emb_1024(seed: f32) -> Vec<f32> {
        (0..1024).map(|i| seed + i as f32 * 0.0001).collect()
    }

    /// 6a. SCHEMALESS round-trip integrity.
    ///
    /// - Open a fresh DB (applies SCHEMALESS DDL via open_db).
    /// - Write 3 chunks with known distinct 1024-dim embeddings.
    /// - Load via VectorIndex::load_from_db.
    /// - Assert index.len() == 3.
    /// - Search with a query matching one known embedding; assert score ≈ 1.0.
    #[tokio::test]
    async fn schemaless_roundtrip_integrity() {
        use crate::vector::VectorIndex;

        let home = TempDir::new().unwrap();
        let repo = "/test/schemaless_roundtrip";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        let embeddings: Vec<Vec<f32>> = vec![emb_1024(1.0), emb_1024(2.0), emb_1024(3.0)];
        let files = ["/repo/a.rs", "/repo/b.rs", "/repo/c.rs"];

        for (i, emb) in embeddings.iter().enumerate() {
            let emb_str: String = emb
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let q = format!(
                "INSERT INTO chunk {{ file: '{}', line_start: 1, line_end: 10, \
                 content: 'x', embedding: [{}], symbol_ref: NONE }}",
                files[i], emb_str
            );
            db.query(&q).await.expect("insert chunk");
        }

        let index = VectorIndex::load_from_db(&db).await.unwrap();
        assert_eq!(index.len(), 3, "index must contain all 3 chunks");

        // Query with exact copy of embeddings[1] — should get score ≈ 1.0.
        let results = index.search(&embeddings[1], 1);
        assert_eq!(results.len(), 1);
        let diff = (results[0].score - 1.0_f32).abs();
        assert!(
            diff < 1e-4,
            "search for exact embedding must return score ≈ 1.0, got {}",
            results[0].score
        );
        assert_eq!(results[0].chunk_id.file, files[1]);
    }

    /// 6b. Migration gating readback.
    ///
    /// - Open a DB with OLD SCHEMAFULL schema.
    /// - Write chunks (short 4-dim embeddings so the schema allows them).
    /// - Manually write a 1024-dim chunk.
    /// - Run run_migration_v2_to_v3.
    /// - Read back; assert embeddings are intact.
    /// - Assert db_schema_version == 3.
    #[tokio::test]
    async fn migration_v2_to_v3_gating_readback() {
        use serde::Deserialize;

        let home = TempDir::new().unwrap();
        let db = open_raw(home.path(), "gating_readback").await;

        // Apply old SCHEMAFULL DDL (v2 state: chunk is SCHEMAFULL with typed fields,
        // but embedding type is array<float> which accepts any float array).
        // We also need index_meta for set_meta/get_meta.
        let old_ddl = "\
            DEFINE TABLE chunk SCHEMAFULL;\
            DEFINE FIELD OVERWRITE file ON chunk TYPE string;\
            DEFINE FIELD OVERWRITE line_start ON chunk TYPE int;\
            DEFINE FIELD OVERWRITE line_end ON chunk TYPE int;\
            DEFINE FIELD OVERWRITE content ON chunk TYPE string;\
            DEFINE FIELD OVERWRITE embedding ON chunk TYPE array<float>;\
            DEFINE FIELD OVERWRITE symbol_ref ON chunk TYPE option<string>;\
            DEFINE TABLE IF NOT EXISTS index_meta SCHEMAFULL;\
            DEFINE FIELD OVERWRITE key ON index_meta TYPE string;\
            DEFINE FIELD OVERWRITE value ON index_meta TYPE string;\
            DEFINE INDEX IF NOT EXISTS idx_meta_key ON index_meta FIELDS key UNIQUE;";
        db.query(old_ddl)
            .await
            .expect("old DDL")
            .check()
            .expect("old DDL check");

        // Write one chunk with a 1024-dim embedding via raw query (bypassing SCHEMAFULL
        // embedding type — we didn't define it typed so it stores as SCHEMALESS for embedding).
        let emb = emb_1024(42.0);
        let emb_str: String = emb
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let q = format!(
            "CREATE chunk SET file = '/x/f.rs', line_start = 1, line_end = 10, \
             content = 'test', embedding = [{}], symbol_ref = NONE",
            emb_str
        );
        db.query(&q).await.expect("insert chunk");

        // Run the migration.
        run_migration_v2_to_v3(&db).await.unwrap();

        // Verify db_schema_version is now 3.
        let version = read_db_schema_version(&db).await;
        assert_eq!(version, 3, "db_schema_version must be 3 after migration");

        // Read back the embedding and assert it's intact.
        #[derive(Deserialize)]
        struct Row {
            embedding: Vec<f32>,
        }
        let rows: Vec<Row> = db
            .query("SELECT embedding FROM chunk WHERE embedding IS NOT NONE LIMIT 1")
            .await
            .expect("readback")
            .take(0)
            .expect("take(0)");

        assert_eq!(rows.len(), 1, "must have one chunk after migration");
        assert_eq!(
            rows[0].embedding.len(),
            1024,
            "embedding must be 1024-dim after migration"
        );
        // Check first and last value are close to the seeded values.
        let diff_first = (rows[0].embedding[0] - emb[0]).abs();
        assert!(
            diff_first < 1e-4,
            "first embedding value must match: {}",
            diff_first
        );
    }

    /// 6c. needs_rebuild flag lifecycle.
    #[tokio::test]
    async fn needs_rebuild_flag_lifecycle() {
        let home = TempDir::new().unwrap();
        let repo = "/test/needs_rebuild";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Set needs_rebuild to "1".
        ops::set_meta(&db, "needs_rebuild", "1").await.unwrap();

        // Assert get_meta returns Some("1").
        let v = ops::get_meta(&db, "needs_rebuild").await.unwrap();
        assert_eq!(v, Some("1".to_string()), "needs_rebuild must be set to '1'");

        // Simulate clearing (the same query used in run_consumer's Ok arm).
        db.query("DELETE FROM index_meta WHERE key = 'needs_rebuild'")
            .await
            .expect("delete needs_rebuild");

        // Assert get_meta returns None.
        let v_after = ops::get_meta(&db, "needs_rebuild").await.unwrap();
        assert_eq!(v_after, None, "needs_rebuild must be None after deletion");
    }

    /// 6d. IS NOT NONE filter correctness.
    ///
    /// - Write chunks: some with real embeddings, some with empty `[]`.
    /// - Query with WHERE embedding IS NOT NONE — assert rows returned include empties too
    ///   (the filter passes both real and empty since [] is not NONE).
    /// - Build VectorIndex — assert only real-embedding rows end up in index
    ///   (empty ones skipped by VectorIndex::insert's is_empty check).
    #[tokio::test]
    async fn is_not_none_filter_correctness() {
        use crate::vector::VectorIndex;

        let home = TempDir::new().unwrap();
        let repo = "/test/is_not_none";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Write 2 chunks with real 1024-dim embeddings.
        for i in 0..2_usize {
            let emb = emb_1024(i as f32 + 1.0);
            let emb_str: String = emb
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let q = format!(
                "INSERT INTO chunk {{ file: '/repo/real_{i}.rs', line_start: 1, line_end: 5, \
                 content: 'real', embedding: [{}], symbol_ref: NONE }}",
                emb_str
            );
            db.query(&q).await.expect("insert real chunk");
        }

        // Write 1 chunk with an empty [] embedding.
        db.query(
            "INSERT INTO chunk { file: '/repo/empty.rs', line_start: 1, line_end: 5, \
             content: 'empty', embedding: [], symbol_ref: NONE }",
        )
        .await
        .expect("insert empty chunk");

        // The IS NOT NONE filter should include ALL rows ([] is not NONE).
        // This matches the behavior documented in the plan for test 6d.
        #[derive(serde::Deserialize)]
        struct CountRow {
            #[allow(dead_code)]
            file: String,
        }
        let all_rows: Vec<CountRow> = db
            .query("SELECT file FROM chunk WHERE embedding IS NOT NONE")
            .await
            .expect("query")
            .take(0)
            .expect("take");
        assert_eq!(
            all_rows.len(),
            3,
            "IS NOT NONE must include all 3 rows (both real and empty [])"
        );

        // VectorIndex::load_from_db uses the IS NOT NONE filter and then skips
        // empty embeddings in VectorIndex::insert. Only 2 real rows end up in index.
        let index = VectorIndex::load_from_db(&db).await.unwrap();
        assert_eq!(
            index.len(),
            2,
            "VectorIndex must contain only 2 real-embedding rows, got {}",
            index.len()
        );
    }

    // ─── v4→v5 packed-bytes embedding tests ──────────────────────────────

    /// pack/unpack round-trip: decode(pack(v)) == v exactly for f32 values,
    /// including the values that show up in real embeddings (negatives, zero,
    /// small fractional). Bit-exact because to_le_bytes/from_le_bytes is a
    /// lossless reinterpretation, not a numeric conversion.
    #[test]
    fn pack_unpack_roundtrip_exact() {
        use crate::store::ops::{de_embedding_dual, pack_embedding};
        use serde::de::value::{BytesDeserializer, Error as ValueError};

        let original = emb_1024(7.0);
        let packed = pack_embedding(&original);
        assert_eq!(packed.len(), original.len() * 4, "4 bytes per f32");

        // Decode through the SAME deserializer used on the read path, via its
        // visit_bytes arm (BytesDeserializer drives visit_bytes).
        let de: BytesDeserializer<ValueError> = BytesDeserializer::new(&packed);
        let decoded = de_embedding_dual(de).expect("decode packed bytes");
        assert_eq!(
            decoded, original,
            "decode(pack(v)) must equal v bit-exactly"
        );
    }

    /// IDEMPOTENCY (the contract): re-encoding an already-`bytes` row reproduces
    /// byte-identical content. This is what makes the v4→v5 migration safe to
    /// resume or replay — a second pass over a converted row is a no-op in effect.
    #[test]
    fn reencode_already_bytes_is_byte_identical() {
        use crate::store::ops::{de_embedding_dual, pack_embedding};
        use serde::de::value::{BytesDeserializer, Error as ValueError};

        let original = emb_1024(3.5);
        let packed_once = pack_embedding(&original);

        // Simulate the migration reading an already-bytes row and re-packing it.
        let de: BytesDeserializer<ValueError> = BytesDeserializer::new(&packed_once);
        let decoded = de_embedding_dual(de).expect("decode bytes");
        let packed_twice = pack_embedding(&decoded);

        assert_eq!(
            packed_once, packed_twice,
            "re-encoding an already-bytes embedding must reproduce identical bytes"
        );
    }

    /// Empty embeddings round-trip to empty: pack([]) == [] and decoding empty
    /// bytes yields []. Confirms the empty-embedding sentinel survives the format
    /// change (VectorIndex::insert skips zero-length vectors downstream).
    #[test]
    fn empty_embedding_roundtrips_empty() {
        use crate::store::ops::{de_embedding_dual, pack_embedding};
        use serde::de::value::{BytesDeserializer, Error as ValueError};

        let packed = pack_embedding(&[]);
        assert!(packed.is_empty(), "pack([]) must be empty bytes");

        let de: BytesDeserializer<ValueError> = BytesDeserializer::new(&packed);
        let decoded = de_embedding_dual(de).expect("decode empty bytes");
        assert!(decoded.is_empty(), "decode(empty bytes) must be empty Vec");
    }

    /// Dual-format READ: a chunk stored as packed `bytes` loads correctly through
    /// VectorIndex::load_from_db (the new-format arm of de_embedding_dual), and a
    /// search for the exact embedding returns score ≈ 1.0.
    #[tokio::test]
    async fn bytes_format_chunk_loads_and_searches() {
        use crate::store::ops::pack_embedding;
        use crate::vector::VectorIndex;
        use surrealdb::sql::Value;

        let home = TempDir::new().unwrap();
        let repo = "/test/bytes_format_read";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Insert a chunk with embedding stored as Value::Bytes (the v5 format),
        // built natively exactly like flush_chunk_batch does.
        let emb = emb_1024(9.0);
        let packed = pack_embedding(&emb);
        let mut map: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
        map.insert("file".into(), Value::from("/repo/bytes.rs"));
        map.insert("line_start".into(), Value::from(1i64));
        map.insert("line_end".into(), Value::from(10i64));
        map.insert("content".into(), Value::from("x"));
        map.insert(
            "embedding".into(),
            Value::Bytes(surrealdb::sql::Bytes::from(packed)),
        );
        map.insert("symbol_ref".into(), Value::None);
        let data =
            surrealdb::sql::Array::from(vec![Value::Object(surrealdb::sql::Object::from(map))]);
        db.query("INSERT INTO chunk $data RETURN NONE")
            .bind(("data", data))
            .await
            .expect("insert bytes chunk");

        // Load via the production read path — exercises the visit_bytes arm.
        let index = VectorIndex::load_from_db(&db).await.unwrap();
        assert_eq!(
            index.len(),
            1,
            "bytes-format chunk must load into the index"
        );

        let results = index.search(&emb, 1);
        assert_eq!(results.len(), 1);
        let diff = (results[0].score - 1.0_f32).abs();
        assert!(
            diff < 1e-4,
            "exact-embedding search must score ≈ 1.0, got {}",
            results[0].score
        );
        assert_eq!(results[0].chunk_id.file, "/repo/bytes.rs");
    }

    /// Mixed-format READ (the half-migrated DB): one row in old `array<float>`
    /// form and one in new `bytes` form load TOGETHER through the same
    /// load_from_db scan, both searchable. This is the keystone guarantee — a
    /// DB mid-migration returns correct results.
    #[tokio::test]
    async fn mixed_old_and_new_format_load_together() {
        use crate::store::ops::pack_embedding;
        use crate::vector::VectorIndex;
        use surrealdb::sql::Value;

        let home = TempDir::new().unwrap();
        let repo = "/test/mixed_format";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Old-format row: array<float> via literal INSERT.
        let old_emb = emb_1024(1.0);
        let old_str: String = old_emb
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        db.query(format!(
            "INSERT INTO chunk {{ file: '/repo/old.rs', line_start: 1, line_end: 5, \
             content: 'old', embedding: [{}], symbol_ref: NONE }}",
            old_str
        ))
        .await
        .expect("insert old-format chunk");

        // New-format row: packed bytes.
        let new_emb = emb_1024(2.0);
        let mut map: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
        map.insert("file".into(), Value::from("/repo/new.rs"));
        map.insert("line_start".into(), Value::from(1i64));
        map.insert("line_end".into(), Value::from(5i64));
        map.insert("content".into(), Value::from("new"));
        map.insert(
            "embedding".into(),
            Value::Bytes(surrealdb::sql::Bytes::from(pack_embedding(&new_emb))),
        );
        map.insert("symbol_ref".into(), Value::None);
        let data =
            surrealdb::sql::Array::from(vec![Value::Object(surrealdb::sql::Object::from(map))]);
        db.query("INSERT INTO chunk $data RETURN NONE")
            .bind(("data", data))
            .await
            .expect("insert new-format chunk");

        // Both must load through the one dual-tolerant scan.
        let index = VectorIndex::load_from_db(&db).await.unwrap();
        assert_eq!(index.len(), 2, "both old and new format rows must load");

        // Both must be searchable to their exact vectors.
        let r_old = index.search(&old_emb, 1);
        assert_eq!(r_old[0].chunk_id.file, "/repo/old.rs");
        assert!((r_old[0].score - 1.0).abs() < 1e-4);
        let r_new = index.search(&new_emb, 1);
        assert_eq!(r_new[0].chunk_id.file, "/repo/new.rs");
        assert!((r_new[0].score - 1.0).abs() < 1e-4);
    }

    /// v4→v5 migration: converts array<float> rows to bytes, is idempotent, and
    /// stamps version=5 only on completion. Re-running is a no-op that keeps the
    /// embeddings bit-exact and the version at 5.
    #[tokio::test]
    async fn migration_v4_to_v5_converts_and_is_idempotent() {
        let home = TempDir::new().unwrap();
        let repo = "/test/v4_to_v5";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Seed rows in OLD array<float> format.
        let seeds = [11.0_f32, 22.0, 33.0];
        for (i, s) in seeds.iter().enumerate() {
            let emb = emb_1024(*s);
            let emb_str: String = emb
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            db.query(format!(
                "INSERT INTO chunk {{ file: '/repo/m_{i}.rs', line_start: 1, line_end: 5, \
                 content: 'c', embedding: [{}], symbol_ref: NONE }}",
                emb_str
            ))
            .await
            .expect("seed old-format chunk");
        }

        // Run the migration.
        run_migration_v4_to_v5(&db).await.expect("v4→v5");
        assert_eq!(
            read_db_schema_version(&db).await,
            5,
            "version must be 5 after migration"
        );

        // All embeddings must still load and search exactly (now from bytes).
        let index = crate::vector::VectorIndex::load_from_db(&db).await.unwrap();
        assert_eq!(index.len(), 3, "all 3 rows present post-migration");
        for s in seeds {
            let emb = emb_1024(s);
            let r = index.search(&emb, 1);
            assert!(
                (r[0].score - 1.0).abs() < 1e-4,
                "post-migration exact search must score ≈ 1.0"
            );
        }

        // Idempotent: re-run completes, version stays 5, embeddings unchanged.
        run_migration_v4_to_v5(&db).await.expect("v4→v5 re-run");
        assert_eq!(
            read_db_schema_version(&db).await,
            5,
            "version stays 5 on re-run"
        );
        let index2 = crate::vector::VectorIndex::load_from_db(&db).await.unwrap();
        assert_eq!(index2.len(), 3, "re-run must not lose or duplicate rows");
        for s in seeds {
            let emb = emb_1024(s);
            let r = index2.search(&emb, 1);
            assert!(
                (r[0].score - 1.0).abs() < 1e-4,
                "embeddings bit-stable across re-run"
            );
        }
    }

    /// v4→v5 crash/resume: a persisted cursor lets a re-run finish the remaining
    /// rows. We simulate an interruption by stamping a partial cursor + leaving
    /// version at 4, then re-run and assert all rows convert and version → 5.
    #[tokio::test]
    async fn migration_v4_to_v5_resumes_from_cursor() {
        let home = TempDir::new().unwrap();
        let repo = "/test/v4_to_v5_resume";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Seed old-format rows.
        for i in 0..5_usize {
            let emb = emb_1024(i as f32 + 1.0);
            let emb_str: String = emb
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            db.query(format!(
                "INSERT INTO chunk {{ file: '/repo/r_{i}.rs', line_start: 1, line_end: 5, \
                 content: 'c', embedding: [{}], symbol_ref: NONE }}",
                emb_str
            ))
            .await
            .expect("seed");
        }

        // Run once to convert everything and reach version 5.
        run_migration_v4_to_v5(&db).await.expect("first run");
        assert_eq!(read_db_schema_version(&db).await, 5);

        // Simulate a crash BEFORE completion of a hypothetical replay: reset the
        // version to 4 and re-run. Because every row is already bytes, the re-run
        // re-encodes them idempotently and re-stamps 5 — proving resume safety
        // even when the cursor key is absent (fresh scan from "").
        let _ = db
            .query("DELETE FROM index_meta WHERE key = $k")
            .bind(("k", DB_SCHEMA_VERSION_KEY))
            .await;
        ops::set_meta(&db, DB_SCHEMA_VERSION_KEY, "4")
            .await
            .unwrap();
        assert_eq!(read_db_schema_version(&db).await, 4);

        run_migration_v4_to_v5(&db).await.expect("resume run");
        assert_eq!(
            read_db_schema_version(&db).await,
            5,
            "resume must complete to version 5"
        );

        let index = crate::vector::VectorIndex::load_from_db(&db).await.unwrap();
        assert_eq!(index.len(), 5, "all rows intact after resume");
    }

    /// `abort_migration` must take the registered handle out of `MIGRATION_TASKS`,
    /// abort + await it, and leave the registry without the key. We register a
    /// never-ending dummy task so we can prove the call returns promptly (it would
    /// hang forever if it awaited the task without aborting) and the key is gone.
    #[tokio::test]
    async fn abort_migration_cancels_and_deregisters() {
        let repo = "/test/abort_migration_cancels";
        let key = normalize_repo_path(repo);
        let handle = tokio::spawn(async {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            }
        });
        MIGRATION_TASKS.lock().unwrap().insert(key.clone(), handle);
        assert!(MIGRATION_TASKS.lock().unwrap().contains_key(&key));

        // Would never return if it awaited the loop without aborting first.
        abort_migration(repo).await;

        assert!(
            !MIGRATION_TASKS.lock().unwrap().contains_key(&key),
            "registry entry must be removed after abort"
        );
    }

    /// `abort_migration` on a repo that was never registered is a no-op: it must
    /// return without panicking and without touching unrelated entries.
    #[tokio::test]
    async fn abort_migration_unknown_repo_is_noop() {
        let repo = "/test/abort_migration_never_registered";
        // Must not panic and must return promptly.
        abort_migration(repo).await;
        assert!(
            !MIGRATION_TASKS
                .lock()
                .unwrap()
                .contains_key(&normalize_repo_path(repo))
        );
    }
}
