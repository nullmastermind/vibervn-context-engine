use context_engine_rs::indexing::pipeline::IndexPipeline;
use context_engine_rs::store::open_db;
use std::time::Instant;
use tempfile::TempDir;

/// Upgrade-path regression guard: applying SCHEMA_DDL over the OLDEST stale symbol
/// schema (parent = option<record<symbol>>) must flip it to SCHEMALESS so the native
/// symbol INSERT (which writes parent as a plain string) persists instead of silently
/// rolling back the whole batch.
///
/// Before the fix this left 0 symbols on upgrade. SCHEMA_DDL now runs
/// `DEFINE TABLE OVERWRITE symbol SCHEMALESS` + `REMOVE FIELD IF EXISTS …`
/// synchronously on every open_db, so the flip happens before any write — no race.
#[tokio::test]
async fn schema_ddl_flips_stale_symbol_so_native_insert_persists() {
    use context_engine_rs::store::schema::SCHEMA_DDL;
    use surrealdb::engine::local::RocksDb;
    use surrealdb::sql::{Array as SqlArray, Object as SqlObject, Value as SqlValue};
    use std::collections::BTreeMap;

    let dir = TempDir::new().unwrap();
    let db = surrealdb::Surreal::new::<RocksDb>(dir.path().to_str().unwrap()).await.unwrap();
    db.use_ns("t").use_db("t").await.unwrap();

    // Oldest stale prod schema: parent declared as a record link + a sentinel row,
    // proving the flip preserves existing data.
    db.query(
        "DEFINE TABLE symbol SCHEMAFULL;\
         DEFINE FIELD name   ON symbol TYPE string;\
         DEFINE FIELD parent ON symbol TYPE option<record<symbol>>;"
    ).await.unwrap().check().unwrap();
    db.query("CREATE symbol:keep SET name = 'sentinel', parent = NONE")
        .await.unwrap().check().unwrap();

    // Apply the real production DDL (idempotent flip happens here).
    db.query(SCHEMA_DDL).await.unwrap().check()
        .expect("SCHEMA_DDL must apply cleanly over a stale SCHEMAFULL symbol table");
    // Re-apply to prove idempotency (open_db runs it on every open).
    db.query(SCHEMA_DDL).await.unwrap().check()
        .expect("SCHEMA_DDL must be idempotent");

    // Native insert writing parent as a plain string — the exact pipeline path.
    let mut m: BTreeMap<String, SqlValue> = BTreeMap::new();
    m.insert("id".into(), SqlValue::from("a.cpp::foo"));
    m.insert("name".into(), SqlValue::from("foo"));
    m.insert("parent".into(), SqlValue::from("symbol:⟨a.cpp::Bar⟩"));
    let data = SqlArray::from(vec![SqlValue::Object(SqlObject::from(m))]);
    db.query("INSERT INTO symbol $data ON DUPLICATE KEY UPDATE name = $input.name RETURN NONE")
        .bind(("data", data))
        .await.unwrap().check()
        .expect("native symbol INSERT must not be rejected by a stale parent field type");

    #[derive(serde::Deserialize)]
    struct C { count: i64 }
    let rows: Vec<C> = db.query("SELECT count() AS count FROM symbol GROUP ALL")
        .await.unwrap().take(0).unwrap();
    assert_eq!(
        rows.first().map(|r| r.count).unwrap_or(0), 2,
        "sentinel row must survive the flip AND the new symbol must persist"
    );
}


/// Regression guard for the silent-empty-symbol bug.
///
/// `INSERT INTO symbol $data` errors with "record already exists" on a batch that
/// contains a duplicate record id (C++ declares a symbol in a .h and defines it in a
/// .cpp → same FQN). The error rolls back the WHOLE batch and is swallowed by
/// `.await` (it only surfaces via `.check()` or `.take()`), so the symbol table ends
/// up empty and Phase 2 resolves zero call edges — fast but broken.
///
/// The fix is `ON DUPLICATE KEY UPDATE ... = $input.<field>` (last-write-wins merge).
/// This test asserts that the merge clause persists all distinct ids and surfaces no
/// error, exactly as the symbol flush path relies on.
#[tokio::test]
async fn insert_with_duplicate_id_merges_instead_of_failing() {
    use surrealdb::engine::local::RocksDb;
    use surrealdb::sql::{Array as SqlArray, Object as SqlObject, Value as SqlValue};
    use std::collections::BTreeMap;

    let dir = TempDir::new().unwrap();
    let db = surrealdb::Surreal::new::<RocksDb>(dir.path().to_str().unwrap()).await.unwrap();
    db.use_ns("t").use_db("t").await.unwrap();
    db.query("DEFINE TABLE symbol SCHEMALESS").await.unwrap();

    fn rec(id: &str, name: &str) -> SqlValue {
        let mut m: BTreeMap<String, SqlValue> = BTreeMap::new();
        m.insert("id".into(), SqlValue::from(id));
        m.insert("name".into(), SqlValue::from(name));
        SqlValue::Object(SqlObject::from(m))
    }

    // Batch contains a duplicate id ("A" twice) — mimics C++ .h/.cpp dup FQNs.
    let data = SqlArray::from(vec![rec("A", "first"), rec("B", "b"), rec("A", "second")]);
    db.query("INSERT INTO symbol $data ON DUPLICATE KEY UPDATE name = $input.name RETURN NONE")
        .bind(("data", data))
        .await
        .expect("insert await")
        .check()
        .expect("INSERT with ON DUPLICATE KEY UPDATE must not error on duplicate id");

    #[derive(serde::Deserialize)]
    struct C { count: i64 }
    let rows: Vec<C> = db
        .query("SELECT count() AS count FROM symbol GROUP ALL")
        .await.unwrap().take(0).unwrap();
    assert_eq!(
        rows.first().map(|r| r.count).unwrap_or(0), 2,
        "both distinct ids (A, B) must persist — duplicate A merges, not rolls back the batch"
    );

    #[derive(serde::Deserialize)]
    struct N { name: String }
    let a: Vec<N> = db
        .query("SELECT name FROM symbol:A")
        .await.unwrap().take(0).unwrap();
    assert_eq!(
        a.first().map(|r| r.name.as_str()), Some("second"),
        "duplicate id must update to the last-written value (matches original UPSERT)"
    );
}

/// Accuracy guard for the real ~/.vibervn DB after a notepad-ade rebuild.
/// Asserts symbols + the calls graph actually persisted (this is exactly the state
/// the silent-empty-symbol bug left at 0). Run after a warm-cache rebuild:
///   cargo test --release --test repro_notepad count_real_db_rows -- --ignored --nocapture
#[ignore]
#[tokio::test]
async fn count_real_db_rows() {
    let repo = "D:/projects/Cpp/notepad-ade".to_string();
    let home_dir = dirs::home_dir().expect("home dir");
    // Diagnostic against the real on-disk index — its base is the builtin
    // default data dir (`~/.vibervn/context-engine`). Constructed locally so
    // the test continues to inspect the real index regardless of any
    // `Settings.data_dir` override (this binary is NOT booting the server).
    let data_dir = home_dir.join(".vibervn").join("context-engine");
    let db = open_db(&data_dir, &repo, 0).await.expect("open real db");

    #[derive(serde::Deserialize)]
    struct C { count: i64 }
    async fn count(db: &surrealdb::Surreal<surrealdb::engine::local::Db>, table: &str) -> i64 {
        let q = format!("SELECT count() AS count FROM {table} GROUP ALL");
        let rows: Vec<C> = db.query(q).await.unwrap().take(0).unwrap();
        rows.first().map(|r| r.count).unwrap_or(0)
    }

    let symbols = count(&db, "symbol").await;
    let chunks = count(&db, "chunk").await;
    let calls = count(&db, "calls").await;
    let raw_edges = count(&db, "raw_edge").await;
    eprintln!(
        "REAL DB ROWS: symbol={symbols} chunk={chunks} calls={calls} raw_edge={raw_edges}"
    );
    assert!(symbols > 0, "symbol table must not be empty (silent-empty-symbol regression)");
    assert!(chunks > 0, "chunk table must not be empty");
    assert!(calls > 0, "calls graph must not be empty — graph-expansion retrieval depends on it");
}

#[tokio::test]
async fn repro_full_rebuild_notepad_ade_fresh_db() {
    // Surface info-level logs (incl. the `PERF SUMMARY` line) on stderr.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("context_engine_rs=info")),
        )
        .with_test_writer()
        .try_init();

    let home = TempDir::new().unwrap();
    let repo = "D:/projects/Cpp/notepad-ade".to_string();
    if !std::path::Path::new(&repo).exists() {
        eprintln!("SKIP: source repo not present");
        return;
    }
    let db = open_db(home.path(), &repo, 0).await.expect("open fresh db");
    // voyage = None — exercises parse + all DB writes + Phase 2 with zero embedding
    // work (the all-cached / no-network floor used for every prior measurement).
    let pipeline = IndexPipeline::new(repo.clone(), None);

    let start = Instant::now();
    let res = pipeline.run(&db, None, true, None, None, None, &[], None).await;
    let wall = start.elapsed();

    match res {
        Ok(s) => {
            #[derive(serde::Deserialize)]
            struct C { count: i64 }
            async fn count(db: &surrealdb::Surreal<surrealdb::engine::local::Db>, table: &str) -> i64 {
                let q = format!("SELECT count() AS count FROM {table} GROUP ALL");
                let rows: Vec<C> = db.query(q).await.unwrap().take(0).unwrap();
                rows.first().map(|r| r.count).unwrap_or(0)
            }
            let symbols = count(&db, "symbol").await;
            let chunks = count(&db, "chunk").await;
            let calls = count(&db, "calls").await;
            eprintln!(
                "REPRO OK: indexed={} total={} wall={:.1}s | PERSISTED symbol={} chunk={} calls={}",
                s.indexed_files,
                s.total_files,
                wall.as_secs_f64(),
                symbols, chunks, calls,
            );
            // Accuracy guards: the silent-empty-symbol bug left symbol=0 and calls=0
            // while chunk stayed populated. Assert the call graph is actually built.
            assert!(symbols > 0, "symbol table must not be empty (silent-empty-symbol regression)");
            assert!(chunks > 0, "chunk table must not be empty");
            assert!(calls > 0, "calls graph must not be empty — graph-expansion retrieval depends on it");
        }
        Err(e) => panic!("REPRO FAILED at: {e:#}"),
    }
}

/// Diagnostic test: inspect the real ~/.vibervn calls indexes and repair any
/// incomplete/missing ones synchronously.
///
/// WRITES TO THE REAL ~/.vibervn INDEX — do NOT run as part of normal CI.
/// Run explicitly with:
///   cargo test --release --test repro_notepad inspect_real_calls_indexes -- --ignored --nocapture
///
/// This test:
///   1. Opens the real SurrealDB for D:/projects/Cpp/notepad-ade
///   2. Runs INFO FOR TABLE calls to inspect index definitions/states
///   3. If any of the 4 idx_calls_* indexes is missing or stuck "building",
///      repairs it synchronously: REMOVE + DEFINE without CONCURRENTLY
///   4. Verifies via a second INFO FOR TABLE calls that all 4 are complete
#[ignore]
#[tokio::test]
async fn inspect_real_calls_indexes() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("context_engine_rs=info")),
        )
        .with_test_writer()
        .try_init();

    let repo = "D:/projects/Cpp/notepad-ade".to_string();
    if !std::path::Path::new(&repo).exists() {
        eprintln!("SKIP: source repo not present at {repo}");
        return;
    }

    let home_dir = dirs::home_dir().expect("dirs::home_dir() must return a value on this platform");
    // Real-data-dir base for diagnostic open_db; matches builtin default.
    let data_dir = home_dir.join(".vibervn").join("context-engine");

    let surreal_dir = home_dir
        .join(".vibervn")
        .join("context-engine")
        .join("surreal")
        .join("D__projects_Cpp_notepad_ade");
    if !surreal_dir.exists() {
        eprintln!("SKIP: real surreal DB not found at {}", surreal_dir.display());
        return;
    }

    let db = open_db(&data_dir, &repo, 0)
        .await
        .expect("open real surreal db");

    // ── BEFORE: inspect calls indexes ───────────────────────────────────────
    let before: Option<serde_json::Value> = db
        .query("INFO FOR TABLE calls")
        .await
        .expect("INFO FOR TABLE calls")
        .take(0)
        .ok()
        .flatten();
    let before = before.unwrap_or(serde_json::Value::Null);
    eprintln!("=== BEFORE: INFO FOR TABLE calls ===");
    eprintln!("{}", serde_json::to_string_pretty(&before).unwrap_or_else(|_| format!("{before:?}")));

    // Extract index names from the before state.
    // SurrealDB returns { "indexes": { "idx_calls_in_file": "DEFINE INDEX ...", ... } }
    let expected_indexes = [
        "idx_calls_in_file",
        "idx_calls_out_file",
        "idx_calls_in_name",
        "idx_calls_out_name",
    ];

    let indexes_obj = before
        .get("indexes")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    // Check which indexes are missing or appear to be in a CONCURRENTLY/building state.
    let mut needs_repair = false;
    for name in &expected_indexes {
        match indexes_obj.get(*name) {
            None => {
                eprintln!("REPAIR NEEDED: index '{name}' is MISSING");
                needs_repair = true;
            }
            Some(def) => {
                let def_str = def.to_string();
                if def_str.contains("CONCURRENTLY") || def_str.contains("building") {
                    eprintln!("REPAIR NEEDED: index '{name}' appears incomplete/building: {def_str}");
                    needs_repair = true;
                } else {
                    eprintln!("OK: index '{name}' present: {def_str}");
                }
            }
        }
    }

    if needs_repair {
        eprintln!("Repairing calls indexes synchronously (REMOVE + DEFINE without CONCURRENTLY)...");
        db.query(
            "REMOVE INDEX IF EXISTS idx_calls_in_file  ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_out_file ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_in_name  ON calls; \
             REMOVE INDEX IF EXISTS idx_calls_out_name ON calls;"
        ).await.expect("remove calls indexes for repair");

        db.query(
            "DEFINE INDEX IF NOT EXISTS idx_calls_in_file  ON calls FIELDS in_file; \
             DEFINE INDEX IF NOT EXISTS idx_calls_out_file ON calls FIELDS out_file; \
             DEFINE INDEX IF NOT EXISTS idx_calls_in_name  ON calls FIELDS in_name; \
             DEFINE INDEX IF NOT EXISTS idx_calls_out_name ON calls FIELDS out_name;"
        ).await.expect("rebuild calls indexes synchronously");

        eprintln!("Repair complete.");
    } else {
        eprintln!("All 4 calls indexes look complete — no repair needed.");
    }

    // ── AFTER: re-inspect to confirm ────────────────────────────────────────
    let after: Option<serde_json::Value> = db
        .query("INFO FOR TABLE calls")
        .await
        .expect("INFO FOR TABLE calls after repair")
        .take(0)
        .ok()
        .flatten();
    let after = after.unwrap_or(serde_json::Value::Null);
    eprintln!("=== AFTER: INFO FOR TABLE calls ===");
    eprintln!("{}", serde_json::to_string_pretty(&after).unwrap_or_else(|_| format!("{after:?}")));

    // Verify all 4 indexes are present and not CONCURRENTLY after repair.
    let after_indexes = after
        .get("indexes")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    for name in &expected_indexes {
        match after_indexes.get(*name) {
            None => panic!("FAIL: index '{name}' still missing after repair"),
            Some(def) => {
                let def_str = def.to_string();
                assert!(
                    !def_str.contains("CONCURRENTLY"),
                    "index '{name}' still has CONCURRENTLY after repair: {def_str}"
                );
                eprintln!("VERIFIED: index '{name}' present and synchronous");
            }
        }
    }
}

/// Warm-cache full-rebuild benchmark.
///
/// WRITES TO THE REAL ~/.vibervn INDEX — do NOT run as part of normal CI.
/// Run explicitly with:
///   cargo test --release --test repro_notepad repro_full_rebuild_notepad_ade_warm_cache -- --ignored --nocapture
///
/// Prerequisites:
///   1. The source repo D:/projects/Cpp/notepad-ade must be present.
///   2. The embedding cache dir ~/.vibervn/context-engine/embeddings/voyage-4-lite must exist
///      (delete the surreal DB first to force a real full rebuild; the cache survives).
///
/// What it measures: the complete production path including cache-READ time for ~53K .bin files.
#[ignore]
#[tokio::test]
async fn repro_full_rebuild_notepad_ade_warm_cache() {
    // Surface info-level logs including the PERF SUMMARY line.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("context_engine_rs=info")),
        )
        .with_test_writer()
        .try_init();

    let repo = "D:/projects/Cpp/notepad-ade".to_string();
    if !std::path::Path::new(&repo).exists() {
        eprintln!("SKIP: source repo not present at {repo}");
        return;
    }

    let home_dir = dirs::home_dir().expect("dirs::home_dir() must return a value on this platform");
    // Real-data-dir base for diagnostic open_db / EmbeddingCache; matches builtin default.
    let data_dir = home_dir.join(".vibervn").join("context-engine");

    // Guard: confirm the embedding cache exists (otherwise the test would just benchmark
    // empty-embedding writes, not the warm-cache-read path the user cares about).
    let cache_dir = data_dir.join("embeddings").join("voyage-4-lite");
    if !cache_dir.exists() {
        eprintln!(
            "SKIP: embedding cache dir not found at {}",
            cache_dir.display()
        );
        return;
    }

    // Build EmbeddingCache pointed at the real on-disk cache.
    use context_engine_rs::embedding::cache::EmbeddingCache;
    let cache = match EmbeddingCache::new(&data_dir, "voyage-4-lite") {
        Some(c) => c,
        None => {
            eprintln!("SKIP: could not open EmbeddingCache (new returned None)");
            return;
        }
    };

    // Open (or create) the real SurrealDB at ~/.vibervn/context-engine/rocksdb/…
    // Note: the rocksdb dir should be deleted before running this test so the
    // rebuild is genuinely forced from scratch, but open_db handles both cases.
    let db = open_db(&data_dir, &repo, 0)
        .await
        .expect("open real surreal db");

    // voyage = None: on a ~100% warm cache the API is never needed.
    // The concurrency of 4 matches the default in IndexPipeline::new().
    let pipeline = IndexPipeline::new_with_concurrency(repo.clone(), None, 4, Some(cache));

    let start = Instant::now();
    let res = pipeline.run(&db, None, true, None, None, None, &[], None).await;
    let wall = start.elapsed();

    match res {
        Ok(s) => {
            eprintln!(
                "REPRO WARM-CACHE OK: indexed={} wall={:.1}s \
                 cache_hit_chunks={} cache_miss_chunks={} embed_total_ms={} \
                 stage3_chunk_ms={} chunk_db_ms={} chunk_cpu_ms={} \
                 chunk_idx_drop_ms={} chunk_idx_rebuild_ms={}",
                s.indexed_files,
                wall.as_secs_f64(),
                s.cache_hit_chunks,
                s.cache_miss_chunks,
                s.embed_total_ms,
                s.stage3_chunk_ms,
                s.stage3_chunk_db_ms,
                s.stage3_chunk_cpu_ms,
                s.stage3_chunk_idx_drop_ms,
                s.stage3_chunk_idx_rebuild_ms,
            );
        }
        Err(e) => panic!("REPRO WARM-CACHE FAILED: {e:#}"),
    }
}

