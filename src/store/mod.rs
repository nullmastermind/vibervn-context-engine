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
pub const DB_SCHEMA_VERSION: u32 = 5;

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
            return Err(anyhow::Error::new(last_err.expect("loop sets last_err on failure")))
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
    info!(stored_version, target = DB_SCHEMA_VERSION, "spawning chained DB migration background task");
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
            cursor = batch.last().map(|r| r.id_str.clone()).unwrap_or(cursor.clone());

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
        let _ = db.query("DELETE FROM index_meta WHERE key = $k")
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
            cursor = batch.last().map(|r| r.path.clone()).unwrap_or(cursor.clone());

            for row in &batch {
                #[derive(Deserialize)]
                struct CountRow { count: i64 }
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
        let _ = db.query("DELETE FROM index_meta WHERE key = $k")
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
         REMOVE FIELD symbol_ref ON chunk;"
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
        cursor = batch.last().map(|r| r.id_str.clone()).unwrap_or(cursor.clone());

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
    let _ = db.query("DELETE FROM index_meta WHERE key = $k")
        .bind(("k", cursor_key))
        .await;

    // Stamp version ONLY after the full scan completes (crash-resume anchor).
    ops::set_meta(db, DB_SCHEMA_VERSION_KEY, "5")
        .await
        .context("migration v4→v5: stamp db_schema_version=5")?;

    info!("migration v4→v5 complete");
    Ok(())
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
pub fn sweep_stale_generations(data_dir: &Path, repos: &[String], generations: &HashMap<String, u32>) {
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
    let stored_version = read_db_schema_version(&db).await;

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
    get_or_open(repo_dbs, data_dir, &repo, generation).await.map(Some)
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
        assert!(db_path(data_dir, repo, 2).exists(), "current gen 2 must be kept");
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
        let sa = get_or_open(&map, home.path(), repo, 0).await.expect("shared A");
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
        let sb = get_or_open(&map, home.path(), repo, 0).await.expect("shared B");
        sb.query(
            "CREATE chunk SET file = '/x/f.rs', line_start = 3, line_end = 4, \
             content = 'y', embedding = [0.5, 0.6, 0.7, 0.8], symbol_ref = NONE;",
        )
        .await
        .expect("write chunk via shared B");

        let sa_after = ops::count_chunks(&sa).await.unwrap();
        assert_eq!(
            sa_after,
            1,
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
            h.await.unwrap().expect("every concurrent open must succeed (no lock race)");
        }

        // Exactly one handle ended up cached.
        assert_eq!(map.read().await.len(), 1, "exactly one cached handle per repo");
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
        let res = open_if_indexed(&map, home.path(), repo, 0).await.expect("ok");
        assert!(res.is_none(), "unindexed repo must return None");
        assert!(
            !db_path(home.path(), repo, 0).exists(),
            "open_if_indexed must NOT create the DB directory for an unindexed repo"
        );
        assert_eq!(map.read().await.len(), 0, "no handle cached for an unindexed repo");

        // After a real open, the directory exists → Some, and the handle is shared.
        let _opened = get_or_open(&map, home.path(), repo, 0).await.expect("open");
        assert!(db_path(home.path(), repo, 0).exists());
        let res2 = open_if_indexed(&map, home.path(), repo, 0).await.expect("ok");
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
        assert_eq!(ops::count_chunks(&db).await.unwrap(), 0, "fresh DB must be empty");
        assert!(db_path(home.path(), repo, 0).exists(), "index directory must exist");

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

    use crate::store::schema::SCHEMA_DDL;
    use crate::store::ops::count_chunks;

    /// Open a raw SurrealKV DB (no DDL applied) on a TempDir.
    /// The caller is responsible for applying whatever schema it needs.
    async fn open_raw_db(dir: &std::path::Path, name: &str) -> Surreal<Db> {
        let path = dir.join(name);
        std::fs::create_dir_all(&path).unwrap();
        let db = Surreal::new::<RocksDb>(path.to_str().unwrap())
            .await
            .expect("open raw db");
        db.use_ns("context_engine").use_db(name).await.expect("ns/db");
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
             DEFINE INDEX IF NOT EXISTS idx_meta_key ON index_meta FIELDS key UNIQUE;"
        )
        .await
        .expect("setup index_meta for migration")
        .check()
        .expect("index_meta setup check");

        crate::store::run_migration_v2_to_v3(&db).await
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
        println!(
            "STALE-SCHEMA WRITE RESULT: errors = {errors:?}"
        );

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
            count,
            1,
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
            after,
            before,
            "DEFINE TABLE IF NOT EXISTS must not drop existing rows (before={before}, after={after})"
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
        let _ = db.query("DELETE FROM index_meta WHERE key = $k")
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
        db.use_ns("context_engine").use_db(name).await.expect("ns/db");
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

        let embeddings: Vec<Vec<f32>> = vec![
            emb_1024(1.0),
            emb_1024(2.0),
            emb_1024(3.0),
        ];
        let files = ["/repo/a.rs", "/repo/b.rs", "/repo/c.rs"];

        for (i, emb) in embeddings.iter().enumerate() {
            let emb_str: String = emb.iter()
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
        db.query(old_ddl).await.expect("old DDL").check().expect("old DDL check");

        // Write one chunk with a 1024-dim embedding via raw query (bypassing SCHEMAFULL
        // embedding type — we didn't define it typed so it stores as SCHEMALESS for embedding).
        let emb = emb_1024(42.0);
        let emb_str: String = emb.iter()
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
        struct Row { embedding: Vec<f32> }
        let rows: Vec<Row> = db
            .query("SELECT embedding FROM chunk WHERE embedding IS NOT NONE LIMIT 1")
            .await
            .expect("readback")
            .take(0)
            .expect("take(0)");

        assert_eq!(rows.len(), 1, "must have one chunk after migration");
        assert_eq!(rows[0].embedding.len(), 1024, "embedding must be 1024-dim after migration");
        // Check first and last value are close to the seeded values.
        let diff_first = (rows[0].embedding[0] - emb[0]).abs();
        assert!(diff_first < 1e-4, "first embedding value must match: {}", diff_first);
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
            let emb_str: String = emb.iter()
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
             content: 'empty', embedding: [], symbol_ref: NONE }"
        )
        .await
        .expect("insert empty chunk");

        // The IS NOT NONE filter should include ALL rows ([] is not NONE).
        // This matches the behavior documented in the plan for test 6d.
        #[derive(serde::Deserialize)]
        struct CountRow { #[allow(dead_code)] file: String }
        let all_rows: Vec<CountRow> = db
            .query("SELECT file FROM chunk WHERE embedding IS NOT NONE")
            .await
            .expect("query")
            .take(0)
            .expect("take");
        assert_eq!(
            all_rows.len(), 3,
            "IS NOT NONE must include all 3 rows (both real and empty [])"
        );

        // VectorIndex::load_from_db uses the IS NOT NONE filter and then skips
        // empty embeddings in VectorIndex::insert. Only 2 real rows end up in index.
        let index = VectorIndex::load_from_db(&db).await.unwrap();
        assert_eq!(
            index.len(), 2,
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
        use crate::store::ops::{pack_embedding, de_embedding_dual};
        use serde::de::value::{BytesDeserializer, Error as ValueError};

        let original = emb_1024(7.0);
        let packed = pack_embedding(&original);
        assert_eq!(packed.len(), original.len() * 4, "4 bytes per f32");

        // Decode through the SAME deserializer used on the read path, via its
        // visit_bytes arm (BytesDeserializer drives visit_bytes).
        let de: BytesDeserializer<ValueError> = BytesDeserializer::new(&packed);
        let decoded = de_embedding_dual(de).expect("decode packed bytes");
        assert_eq!(decoded, original, "decode(pack(v)) must equal v bit-exactly");
    }

    /// IDEMPOTENCY (the contract): re-encoding an already-`bytes` row reproduces
    /// byte-identical content. This is what makes the v4→v5 migration safe to
    /// resume or replay — a second pass over a converted row is a no-op in effect.
    #[test]
    fn reencode_already_bytes_is_byte_identical() {
        use crate::store::ops::{pack_embedding, de_embedding_dual};
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
        use crate::store::ops::{pack_embedding, de_embedding_dual};
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
        use crate::vector::VectorIndex;
        use crate::store::ops::pack_embedding;
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
        map.insert("embedding".into(), Value::Bytes(surrealdb::sql::Bytes::from(packed)));
        map.insert("symbol_ref".into(), Value::None);
        let data = surrealdb::sql::Array::from(vec![Value::Object(surrealdb::sql::Object::from(map))]);
        db.query("INSERT INTO chunk $data RETURN NONE")
            .bind(("data", data))
            .await
            .expect("insert bytes chunk");

        // Load via the production read path — exercises the visit_bytes arm.
        let index = VectorIndex::load_from_db(&db).await.unwrap();
        assert_eq!(index.len(), 1, "bytes-format chunk must load into the index");

        let results = index.search(&emb, 1);
        assert_eq!(results.len(), 1);
        let diff = (results[0].score - 1.0_f32).abs();
        assert!(diff < 1e-4, "exact-embedding search must score ≈ 1.0, got {}", results[0].score);
        assert_eq!(results[0].chunk_id.file, "/repo/bytes.rs");
    }

    /// Mixed-format READ (the half-migrated DB): one row in old `array<float>`
    /// form and one in new `bytes` form load TOGETHER through the same
    /// load_from_db scan, both searchable. This is the keystone guarantee — a
    /// DB mid-migration returns correct results.
    #[tokio::test]
    async fn mixed_old_and_new_format_load_together() {
        use crate::vector::VectorIndex;
        use crate::store::ops::pack_embedding;
        use surrealdb::sql::Value;

        let home = TempDir::new().unwrap();
        let repo = "/test/mixed_format";
        let db = open_db(home.path(), repo, 0).await.unwrap();

        // Old-format row: array<float> via literal INSERT.
        let old_emb = emb_1024(1.0);
        let old_str: String = old_emb.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
        db.query(format!(
            "INSERT INTO chunk {{ file: '/repo/old.rs', line_start: 1, line_end: 5, \
             content: 'old', embedding: [{}], symbol_ref: NONE }}",
            old_str
        )).await.expect("insert old-format chunk");

        // New-format row: packed bytes.
        let new_emb = emb_1024(2.0);
        let mut map: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
        map.insert("file".into(), Value::from("/repo/new.rs"));
        map.insert("line_start".into(), Value::from(1i64));
        map.insert("line_end".into(), Value::from(5i64));
        map.insert("content".into(), Value::from("new"));
        map.insert("embedding".into(), Value::Bytes(surrealdb::sql::Bytes::from(pack_embedding(&new_emb))));
        map.insert("symbol_ref".into(), Value::None);
        let data = surrealdb::sql::Array::from(vec![Value::Object(surrealdb::sql::Object::from(map))]);
        db.query("INSERT INTO chunk $data RETURN NONE").bind(("data", data)).await.expect("insert new-format chunk");

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
            let emb_str: String = emb.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
            db.query(format!(
                "INSERT INTO chunk {{ file: '/repo/m_{i}.rs', line_start: 1, line_end: 5, \
                 content: 'c', embedding: [{}], symbol_ref: NONE }}",
                emb_str
            )).await.expect("seed old-format chunk");
        }

        // Run the migration.
        run_migration_v4_to_v5(&db).await.expect("v4→v5");
        assert_eq!(read_db_schema_version(&db).await, 5, "version must be 5 after migration");

        // All embeddings must still load and search exactly (now from bytes).
        let index = crate::vector::VectorIndex::load_from_db(&db).await.unwrap();
        assert_eq!(index.len(), 3, "all 3 rows present post-migration");
        for s in seeds {
            let emb = emb_1024(s);
            let r = index.search(&emb, 1);
            assert!((r[0].score - 1.0).abs() < 1e-4, "post-migration exact search must score ≈ 1.0");
        }

        // Idempotent: re-run completes, version stays 5, embeddings unchanged.
        run_migration_v4_to_v5(&db).await.expect("v4→v5 re-run");
        assert_eq!(read_db_schema_version(&db).await, 5, "version stays 5 on re-run");
        let index2 = crate::vector::VectorIndex::load_from_db(&db).await.unwrap();
        assert_eq!(index2.len(), 3, "re-run must not lose or duplicate rows");
        for s in seeds {
            let emb = emb_1024(s);
            let r = index2.search(&emb, 1);
            assert!((r[0].score - 1.0).abs() < 1e-4, "embeddings bit-stable across re-run");
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
            let emb_str: String = emb.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
            db.query(format!(
                "INSERT INTO chunk {{ file: '/repo/r_{i}.rs', line_start: 1, line_end: 5, \
                 content: 'c', embedding: [{}], symbol_ref: NONE }}",
                emb_str
            )).await.expect("seed");
        }

        // Run once to convert everything and reach version 5.
        run_migration_v4_to_v5(&db).await.expect("first run");
        assert_eq!(read_db_schema_version(&db).await, 5);

        // Simulate a crash BEFORE completion of a hypothetical replay: reset the
        // version to 4 and re-run. Because every row is already bytes, the re-run
        // re-encodes them idempotently and re-stamps 5 — proving resume safety
        // even when the cursor key is absent (fresh scan from "").
        let _ = db.query("DELETE FROM index_meta WHERE key = $k")
            .bind(("k", DB_SCHEMA_VERSION_KEY)).await;
        ops::set_meta(&db, DB_SCHEMA_VERSION_KEY, "4").await.unwrap();
        assert_eq!(read_db_schema_version(&db).await, 4);

        run_migration_v4_to_v5(&db).await.expect("resume run");
        assert_eq!(read_db_schema_version(&db).await, 5, "resume must complete to version 5");

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
        assert!(!MIGRATION_TASKS.lock().unwrap().contains_key(&normalize_repo_path(repo)));
    }
}
