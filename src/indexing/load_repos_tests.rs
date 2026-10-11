//! Tests for the warm/lazy-load and identity-fenced query paths of
//! [`super::IndexEngine`]. Split out of `indexing/mod.rs` to keep that file
//! focused on production logic.

use super::*;
use tempfile::TempDir;

/// Seed `n` chunk rows (each with a non-empty 4-d embedding) into `repo`'s DB,
/// writing THROUGH the shared `repo_dbs` map (one cached handle per repo, like
/// production). RocksDB holds an exclusive per-directory lock, so a second
/// uncached `open_db` on the same path would deadlock on the lock file 鈥?seeding
/// through `get_or_open` keeps a single handle, mirroring real usage.
fn identity_for(model: &str) -> EmbeddingIdentity {
    let client = VoyageClient::new(model.to_owned(), vec!["test-key".to_owned()], None)
        .expect("test embedding client");
    EmbeddingIdentity::from_client(&client)
}

fn test_identity() -> EmbeddingIdentity {
    identity_for("test-model")
}

/// The 4-d embedding every seeded chunk carries; also used as the query probe so
/// a resident shard always yields a candidate.
const PROBE_EMBEDDING: [f32; 4] = [0.1, 0.2, 0.3, 0.4];

/// Build a real `IndexEngine` over `repo_dbs` with `repo` configured, sharing the
/// same handle map used for seeding (one RocksDB handle per repo, like production).
/// Watchers are disabled: these tests drive the fence directly.
async fn start_test_engine(
    home: &std::path::Path,
    repo_dbs: &RepoDbMap,
    repo: &str,
) -> Arc<IndexEngine> {
    let settings = crate::config::Settings {
        repos: vec![repo.to_owned()],
        ..Default::default()
    };
    IndexEngine::start(
        home.to_path_buf(),
        home.join("embeddings"),
        &settings,
        repo_dbs.clone(),
        Arc::new(RwLock::new(settings.clone())),
        true,
    )
    .await
}

/// One publishable vector for `repo`, matching what the pipeline hands to
/// `publish_full_update` / `publish_incremental_update`.
fn published_vector(repo: &str) -> (crate::vector::ChunkId, Vec<f32>) {
    (
        crate::vector::ChunkId {
            file: format!("{repo}/f0.rs"),
            line_start: 1,
            line_end: 2,
        },
        PROBE_EMBEDDING.to_vec(),
    )
}

async fn seed_repo(repo_dbs: &RepoDbMap, home: &std::path::Path, repo: &str, n: usize) {
    let db = store::get_or_open(repo_dbs, home, repo, 0)
        .await
        .expect("get_or_open");
    for i in 0..n {
        let q = format!(
            "CREATE chunk SET file = '{repo}/f{i}.rs', line_start = 1, line_end = 2, \
             content = 'x', embedding = [0.1, 0.2, 0.3, 0.4], symbol_ref = NONE;"
        );
        db.query(&q).await.expect("seed chunk");
    }
    let identity = test_identity();
    set_meta(&db, EMBEDDING_IDENTITY_KEY, &identity.as_key_string())
        .await
        .expect("seed embedding identity");
}

/// Warming EACH configured repo into its own resident shard (when the cap
/// allows) installs an independent shard per repo, not just the first. Two
/// repos seeded with 1 and 2 chunks 鈫?both shards resident, total 3 vectors
/// searchable across shards. This is the per-repo lazy-warm path now used on
/// first query (boot no longer eagerly warms).
#[tokio::test]
async fn loads_all_repos_not_just_first() {
    let home = TempDir::new().expect("tempdir");
    let repo_one = "/proj/repo_one".to_string();
    let repo_two = "/proj/repo_two".to_string();

    // Shared map used for BOTH seeding and warming 鈥?exactly one handle per repo.
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    seed_repo(&repo_dbs, home.path(), &repo_one, 1).await;
    seed_repo(&repo_dbs, home.path(), &repo_two, 2).await;

    // Large cap so both repos stay resident after warming.
    let vector_index = Arc::new(RwLock::new(ShardedVectorIndex::new(1024 * 1024 * 1024)));
    let identity = test_identity();
    warm_repo_shard(
        &vector_index,
        &repo_dbs,
        home.path(),
        &repo_one,
        0,
        &identity,
        &[],
    )
    .await
    .expect("warm repo one");
    warm_repo_shard(
        &vector_index,
        &repo_dbs,
        home.path(),
        &repo_two,
        0,
        &identity,
        &[],
    )
    .await
    .expect("warm repo two");

    let vi = vector_index.read().await;
    assert!(vi.is_resident(&repo_one), "repo_one shard must be resident");
    assert!(vi.is_resident(&repo_two), "repo_two shard must be resident");
    assert_eq!(
        vi.resident_repo_count(),
        2,
        "expected both repos warmed into shards, not just the first"
    );
}

/// Seed `n` file_meta rows for `repo`, writing through the shared `repo_dbs`
/// map (single cached handle per repo 鈥?see [`seed_repo`] for why RocksDB
/// requires this).
async fn seed_file_meta(repo_dbs: &RepoDbMap, home: &std::path::Path, repo: &str, n: usize) {
    let db = store::get_or_open(repo_dbs, home, repo, 0)
        .await
        .expect("get_or_open");
    for i in 0..n {
        let path = format!("{repo}/f{i}.rs");
        db.query("CREATE file_meta SET path = $path, mtime = 0, size = 1, repo = $repo, chunk_count = 1;")
            .bind(("path", path))
            .bind(("repo", repo.to_string()))
            .await
            .expect("seed file_meta");
    }
}

/// After a restart, a repo indexed in a prior session must show its persisted
/// file count 鈥?not the zeroed default. A never-indexed repo must stay at 0
/// so the UI can render a "Not indexed" placeholder.
#[tokio::test]
async fn seeds_status_from_persisted_file_meta() {
    let home = TempDir::new().expect("tempdir");
    let indexed_raw = "/proj/indexed".to_string();
    let empty_raw = "/proj/empty".to_string();
    let indexed = store::normalize_repo_path(&indexed_raw);
    let empty = store::normalize_repo_path(&empty_raw);

    // Shared map for seeding AND the seed-status call 鈥?one handle per repo.
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    seed_file_meta(&repo_dbs, home.path(), &indexed, 5).await;
    // `empty` gets a DB (cached) but no file_meta rows.
    let _ = store::get_or_open(&repo_dbs, home.path(), &empty, 0)
        .await
        .expect("get_or_open");

    let statuses: Arc<RwLock<HashMap<String, RepoStatus>>> = Arc::new(RwLock::new(HashMap::new()));
    {
        let mut m = statuses.write().await;
        m.insert(indexed.clone(), RepoStatus::default());
        m.insert(empty.clone(), RepoStatus::default());
    }

    seed_statuses_from_db(
        &statuses,
        &repo_dbs,
        home.path(),
        &[indexed_raw, empty_raw],
        &HashMap::new(),
        &Arc::new(RwLock::new(crate::config::Settings::default())),
    )
    .await;

    let m = statuses.read().await;
    assert_eq!(
        m[&indexed].indexed_files, 5,
        "indexed repo must restore its file count"
    );
    assert_eq!(
        m[&empty].indexed_files, 0,
        "never-indexed repo must stay at 0"
    );
}

/// A run that has already advanced a repo's status by the time the seed task
/// runs must not be clobbered back to the persisted (possibly stale) count.
#[tokio::test]
async fn seed_does_not_clobber_live_run() {
    let home = TempDir::new().expect("tempdir");
    let repo_raw = "/proj/live".to_string();
    let repo = store::normalize_repo_path(&repo_raw);
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    seed_file_meta(&repo_dbs, home.path(), &repo, 5).await;

    let statuses: Arc<RwLock<HashMap<String, RepoStatus>>> = Arc::new(RwLock::new(HashMap::new()));
    {
        let mut m = statuses.write().await;
        m.insert(
            repo.clone(),
            RepoStatus {
                state: IndexState::Indexing,
                ..Default::default()
            },
        );
    }

    seed_statuses_from_db(
        &statuses,
        &repo_dbs,
        home.path(),
        &[repo_raw],
        &HashMap::new(),
        &Arc::new(RwLock::new(crate::config::Settings::default())),
    )
    .await;

    let m = statuses.read().await;
    assert_eq!(
        m[&repo].state,
        IndexState::Indexing,
        "in-flight run must survive the seed"
    );
    assert_eq!(
        m[&repo].indexed_files, 0,
        "seed must not overwrite a live run's numerator"
    );
}

/// Single-flight coalescing: concurrent `warm_repo_blocking` calls for the same
/// cold repo must result in exactly ONE `load_from_db` scan + install 鈥?the
/// later callers block on the per-repo warm lock, then observe the repo already
/// resident and return without re-scanning. We assert the end state (resident,
/// correct vector count) and that the shard was installed once (no duplicate /
/// doubled vectors), which is the observable guarantee of coalescing.
#[tokio::test]
async fn warm_blocking_is_single_flight() {
    let home = TempDir::new().expect("tempdir");
    let repo = "/proj/warm".to_string();

    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    seed_repo(&repo_dbs, home.path(), &repo, 3).await;

    // Minimal engine sharing the SAME repo_dbs map used for seeding (one handle).
    let settings = crate::config::Settings {
        repos: vec![repo.clone()],
        ..Default::default()
    };
    let settings_handle = Arc::new(RwLock::new(settings.clone()));
    let engine = IndexEngine::start(
        home.path().to_path_buf(),
        home.path().join("embeddings"),
        &settings,
        repo_dbs.clone(),
        settings_handle,
        false,
    )
    .await;

    // Fire several concurrent warms for the same cold repo.
    let mut handles = Vec::new();
    let identity = test_identity();
    for _ in 0..5 {
        let e = engine.clone();
        let r = repo.clone();
        let id = identity.clone();
        handles.push(tokio::spawn(async move {
            e.warm_repo_blocking(r, id).await;
        }));
    }
    for h in handles {
        h.await.expect("warm task");
    }

    let vi = engine.vector_index.read().await;
    assert!(vi.is_resident(&repo), "repo must be resident after warm");
    // Exactly the seeded 3 vectors 鈥?coalescing means no doubled inserts.
    let mut q = vec![0.0f32; 4];
    q[0] = 1.0;
    let out = vi.search(&q, 100, Some(&repo), std::slice::from_ref(&repo));
    assert_eq!(
        out.results.len(),
        3,
        "single-flight warm must install the shard once (3 seeded vectors, not doubled)"
    );
}

/// A shard that is NOT resident after the warm attempt yields warming=true with
/// empty results 鈥?the "retry, not empty" signal. Forced deterministically with a
/// 0-chunk repo: warm_repo_shard loads nothing (count==0) and never installs a
/// shard, so it stays non-resident regardless of timing 鈥?exactly the condition
/// `warming = !is_resident(repo)` detects after the bounded warm.
#[tokio::test(flavor = "multi_thread")]
async fn vector_search_signals_warming_when_shard_not_resident() {
    let home = TempDir::new().expect("tempdir");
    let repo = "/proj/cold".to_string();
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    // 0 chunks 鈫?warm installs no shard 鈫?non-resident after the warm attempt.
    seed_repo(&repo_dbs, home.path(), &repo, 0).await;
    let settings = crate::config::Settings {
        repos: vec![repo.clone()],
        ..Default::default()
    };
    let settings_handle = Arc::new(RwLock::new(settings.clone()));
    let engine = IndexEngine::start(
        home.path().to_path_buf(),
        home.path().join("embeddings"),
        &settings,
        repo_dbs.clone(),
        settings_handle,
        false,
    )
    .await;

    let q = vec![1.0f32, 0.0, 0.0, 0.0];
    let identity = test_identity();
    let outcome = engine
        .vector_search(
            &q,
            10,
            Some(&repo),
            std::time::Duration::from_secs(5),
            &identity,
        )
        .await;
    assert!(
        outcome.results.is_empty(),
        "non-resident shard search returns empty"
    );
    assert!(
        outcome.warming,
        "non-resident shard after warm attempt must signal warming=true"
    );
}

/// Multi-repo (`repo_filter=None`) search must fail closed per shard identity:
/// matching shards contribute results, mismatched shards contribute NOTHING,
/// are identity-fenced evicted, and are background-warmed for repair.
#[tokio::test(flavor = "multi_thread")]
async fn vector_search_none_excludes_mismatched_shards_and_background_warms() {
    let home = TempDir::new().expect("tempdir");
    let matching_raw = "/proj/matching".to_string();
    let mismatch_raw = "/proj/mismatch".to_string();
    let matching_repo = store::normalize_repo_path(&matching_raw);
    let mismatch_repo = store::normalize_repo_path(&mismatch_raw);
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));

    // The mismatched repo has durable old-model data. Background warm proves it
    // ran by persisting needs_rebuild=1 after it sees DB identity != query identity.
    seed_repo(&repo_dbs, home.path(), &mismatch_raw, 1).await;

    let settings = crate::config::Settings {
        repos: vec![matching_raw, mismatch_raw],
        ..Default::default()
    };
    let settings_handle = Arc::new(RwLock::new(settings.clone()));
    let engine = IndexEngine::start(
        home.path().to_path_buf(),
        home.path().join("embeddings"),
        &settings,
        repo_dbs.clone(),
        settings_handle,
        true,
    )
    .await;

    let current = identity_for("current-model");
    let stale = test_identity();
    let make_shard = |repo: &str, value: f32| {
        let mut shard = VectorIndex::new();
        shard.insert(&[(
            crate::vector::ChunkId {
                file: format!("{repo}/result.rs"),
                line_start: 1,
                line_end: 2,
            },
            vec![value, 0.0, 0.0, 0.0],
        )]);
        shard
    };
    {
        let mut index = engine.vector_index.write().await;
        index.install_shard(
            &matching_repo,
            make_shard(&matching_repo, 1.0),
            current.as_key_string(),
            &[],
        );
        index.install_shard(
            &mismatch_repo,
            make_shard(&mismatch_repo, 1.0),
            stale.as_key_string(),
            &[],
        );
    }

    let outcome = engine
        .vector_search(
            &[1.0, 0.0, 0.0, 0.0],
            10,
            None,
            std::time::Duration::from_secs(1),
            &current,
        )
        .await;

    assert!(
        outcome
            .results
            .iter()
            .all(|r| r.chunk_id.file.starts_with(&matching_repo)),
        "identity-mismatched repo must contribute no result: {:?}",
        outcome
            .results
            .iter()
            .map(|r| &r.chunk_id.file)
            .collect::<Vec<_>>()
    );
    assert!(
        outcome
            .results
            .iter()
            .any(|r| r.chunk_id.file.starts_with(&matching_repo)),
        "matching shard must still contribute results"
    );
    assert!(
        outcome.warming,
        "mismatched in-scope repo must signal warming"
    );
    assert!(
        !engine.vector_index.read().await.is_resident(&mismatch_repo),
        "mismatched shard must be identity-fenced evicted"
    );

    let mismatch_db = store::get_or_open(&repo_dbs, home.path(), &mismatch_repo, 0)
        .await
        .expect("open mismatch DB");
    let mut warm_observed = false;
    for _ in 0..100 {
        if store::ops::get_meta(&mismatch_db, "needs_rebuild")
            .await
            .expect("read needs_rebuild")
            .as_deref()
            == Some("1")
        {
            warm_observed = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(
        warm_observed,
        "background warm must run and persist needs_rebuild on DB identity mismatch"
    );
}

/// A resident shard that genuinely matches nothing yields warming=false 鈥?a real
/// empty, NOT a warming state. (Here the shard is warmed first via a generous wait.)
#[tokio::test(flavor = "multi_thread")]
async fn vector_search_resident_empty_is_not_warming() {
    let home = TempDir::new().expect("tempdir");
    let repo = "/proj/resident".to_string();
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    seed_repo(&repo_dbs, home.path(), &repo, 3).await;
    let settings = crate::config::Settings {
        repos: vec![repo.clone()],
        ..Default::default()
    };
    let settings_handle = Arc::new(RwLock::new(settings.clone()));
    let engine = IndexEngine::start(
        home.path().to_path_buf(),
        home.path().join("embeddings"),
        &settings,
        repo_dbs.clone(),
        settings_handle,
        false,
    )
    .await;

    // Warm the shard explicitly so it is resident before the search.
    let identity = test_identity();
    engine
        .warm_repo_blocking(repo.clone(), identity.clone())
        .await;
    assert!(
        engine.vector_index.read().await.is_resident(&repo),
        "precondition: resident"
    );

    // Query with a generous wait; the shard is resident so warming must be false.
    // top_k=0 forces an empty result set on a resident shard (genuine empty).
    let q = vec![1.0f32, 0.0, 0.0, 0.0];
    let outcome = engine
        .vector_search(
            &q,
            0,
            Some(&repo),
            std::time::Duration::from_secs(10),
            &identity,
        )
        .await;
    assert!(
        outcome.results.is_empty(),
        "top_k=0 yields empty on a resident shard"
    );
    assert!(
        !outcome.warming,
        "resident shard must NOT signal warming, even when empty"
    );
}

/// Harness for the full-rebuild window: delete_all_data removes every chunk
/// while the old resident shard is still installed. This must be fail-closed:
/// no vector result may survive without its DB content.
#[tokio::test(flavor = "multi_thread")]
async fn full_rebuild_window_does_not_search_stale_resident_shard() {
    let home = TempDir::new().expect("tempdir");
    let repo = "/proj/full-window".to_string();
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    seed_repo(&repo_dbs, home.path(), &repo, 1).await;
    let db = store::get_or_open(&repo_dbs, home.path(), &repo, 0)
        .await
        .expect("open repo DB");
    store::ops::delete_all_data(&db)
        .await
        .expect("simulate full rebuild delete phase");
    assert_eq!(
        store::ops::count_chunks(&db).await.expect("count chunks"),
        0,
        "precondition: full rebuild deleted chunk rows"
    );

    let settings = crate::config::Settings {
        repos: vec![repo.clone()],
        ..Default::default()
    };
    let engine = IndexEngine::start(
        home.path().to_path_buf(),
        home.path().join("embeddings"),
        &settings,
        repo_dbs.clone(),
        Arc::new(RwLock::new(settings.clone())),
        true,
    )
    .await;
    let identity = test_identity();
    let mut stale_shard = VectorIndex::new();
    stale_shard.insert(&[(
        crate::vector::ChunkId {
            file: format!("{repo}/f0.rs"),
            line_start: 1,
            line_end: 2,
        },
        vec![0.1, 0.2, 0.3, 0.4],
    )]);
    engine.vector_index.write().await.install_shard(
        &repo,
        stale_shard,
        identity.as_key_string(),
        &[],
    );

    let outcome = engine
        .vector_search(
            &[0.1, 0.2, 0.3, 0.4],
            10,
            Some(&repo),
            std::time::Duration::ZERO,
            &identity,
        )
        .await;
    assert!(
        !outcome.results.is_empty(),
        "precondition: the stale resident shard still yields a vector candidate"
    );
    let fenced = crate::query::engine::hydrate_candidates(
        &HashMap::from([(repo.clone(), db.clone())]),
        &outcome.results,
    )
    .await;
    assert!(
        fenced.kept.is_empty(),
        "full rebuild window must emit no result block without durable content"
    );
    assert!(
        fenced.stale_shard_detected(),
        "all stale candidates must become retryable warming"
    );
}

/// Harness for the incremental window: delete_files_data_incremental removes
/// the affected file's chunks while its old vectors remain resident.
#[tokio::test(flavor = "multi_thread")]
async fn incremental_window_does_not_search_stale_resident_shard() {
    let home = TempDir::new().expect("tempdir");
    let repo = "/proj/incremental-window".to_string();
    let file = format!("{repo}/f0.rs");
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    seed_repo(&repo_dbs, home.path(), &repo, 1).await;
    let db = store::get_or_open(&repo_dbs, home.path(), &repo, 0)
        .await
        .expect("open repo DB");
    store::ops::delete_files_data_incremental(&db, std::slice::from_ref(&file))
        .await
        .expect("simulate incremental delete phase");
    assert_eq!(
        store::ops::count_chunks(&db).await.expect("count chunks"),
        0,
        "precondition: incremental rebuild deleted affected chunk rows"
    );

    let settings = crate::config::Settings {
        repos: vec![repo.clone()],
        ..Default::default()
    };
    let engine = IndexEngine::start(
        home.path().to_path_buf(),
        home.path().join("embeddings"),
        &settings,
        repo_dbs.clone(),
        Arc::new(RwLock::new(settings.clone())),
        true,
    )
    .await;
    let identity = test_identity();
    let mut stale_shard = VectorIndex::new();
    stale_shard.insert(&[(
        crate::vector::ChunkId {
            file,
            line_start: 1,
            line_end: 2,
        },
        vec![0.1, 0.2, 0.3, 0.4],
    )]);
    engine.vector_index.write().await.install_shard(
        &repo,
        stale_shard,
        identity.as_key_string(),
        &[],
    );

    let outcome = engine
        .vector_search(
            &[0.1, 0.2, 0.3, 0.4],
            10,
            Some(&repo),
            std::time::Duration::ZERO,
            &identity,
        )
        .await;
    assert!(
        !outcome.results.is_empty(),
        "precondition: the stale affected-file vector still yields a candidate"
    );
    let fenced = crate::query::engine::hydrate_candidates(
        &HashMap::from([(repo.clone(), db.clone())]),
        &outcome.results,
    )
    .await;
    assert!(
        fenced.kept.is_empty(),
        "incremental rebuild window must emit no result block without durable content"
    );
    assert!(
        fenced.stale_shard_detected(),
        "all stale candidates must become retryable warming"
    );
}

/// Produce a path spelling which canonicalises to the same fence key while
/// exercising separator conversion on every platform and case folding on Windows.
fn alternate_repo_spelling(repo: &str) -> String {
    let with_other_separators = repo.replace('/', "\\");
    let equivalent = if cfg!(windows) {
        with_other_separators.to_uppercase()
    } else {
        with_other_separators
    };
    format!("{equivalent}\\")
}

/// Exercise the production wiring, not just the MCP readiness policy:
/// `ShardedVectorIndex` mutation state must be observable through
/// `IndexEngine::repo_is_mutating`, including an equivalent unnormalised path.
/// A successful incremental publish releases the fence.
#[tokio::test]
async fn incremental_mutation_fence_is_visible_through_engine_until_publish() {
    const REPO_INPUT: &str = "C:/Proj/Fence-Wiring-Incremental";

    let home = TempDir::new().expect("tempdir");
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    let engine = start_test_engine(home.path(), &repo_dbs, REPO_INPUT).await;
    let fence_key = store::normalize_repo_path(REPO_INPUT);
    let alternate_spelling = alternate_repo_spelling(REPO_INPUT);

    engine
        .vector_index
        .write()
        .await
        .begin_incremental_update(&fence_key);
    assert!(engine.repo_is_mutating(REPO_INPUT).await);
    assert!(
        engine.repo_is_mutating(&alternate_spelling).await,
        "equivalent slash/case/trailing-separator spelling must hit the same fence key"
    );

    let new_vectors = [published_vector(&fence_key)];
    engine
        .vector_index
        .write()
        .await
        .publish_incremental_update(
            &fence_key,
            &[],
            &new_vectors,
            test_identity().as_key_string(),
            &[],
        );
    assert!(!engine.repo_is_mutating(REPO_INPUT).await);
    assert!(!engine.repo_is_mutating(&alternate_spelling).await);
}

/// A failed destructive update intentionally remains fail-closed after abort;
/// only a later successful publish releases it. This pins the actual
/// `abort_update_fail_closed` contract and the full-update wiring end to end.
#[tokio::test]
async fn full_mutation_fence_is_visible_through_engine_and_survives_abort() {
    const REPO_INPUT: &str = "C:/Proj/Fence-Wiring-Full";

    let home = TempDir::new().expect("tempdir");
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    let engine = start_test_engine(home.path(), &repo_dbs, REPO_INPUT).await;
    let fence_key = store::normalize_repo_path(REPO_INPUT);
    let alternate_spelling = alternate_repo_spelling(REPO_INPUT);

    engine
        .vector_index
        .write()
        .await
        .begin_full_update(&fence_key);
    assert!(engine.repo_is_mutating(REPO_INPUT).await);
    assert!(engine.repo_is_mutating(&alternate_spelling).await);

    engine
        .vector_index
        .write()
        .await
        .abort_update_fail_closed(&fence_key);
    assert!(
        engine.repo_is_mutating(REPO_INPUT).await,
        "failed destructive update must stay fenced until repaired"
    );
    assert!(engine.repo_is_mutating(&alternate_spelling).await);

    let replacement = [published_vector(&fence_key)];
    engine.vector_index.write().await.publish_full_update(
        &fence_key,
        &replacement,
        test_identity().as_key_string(),
        &[],
    );
    assert!(!engine.repo_is_mutating(REPO_INPUT).await);
    assert!(!engine.repo_is_mutating(&alternate_spelling).await);
}

/// REGRESSION: writer-side spelling must not create a fence key that the
/// canonical MCP reader cannot see. This deliberately passes the unnormalised
/// spelling to the real sharded writer APIs, then observes through
/// `IndexEngine::repo_is_mutating` using the canonical spelling.
#[tokio::test]
async fn mutation_fence_writer_boundary_normalizes_repo_spelling() {
    const REPO_INPUT: &str = "C:/Proj/Fence-Writer-Boundary";

    let home = TempDir::new().expect("tempdir");
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    let engine = start_test_engine(home.path(), &repo_dbs, REPO_INPUT).await;
    let canonical = store::normalize_repo_path(REPO_INPUT);
    let raw_writer_spelling = alternate_repo_spelling(REPO_INPUT);

    engine
        .vector_index
        .write()
        .await
        .begin_incremental_update(&raw_writer_spelling);
    assert!(
        engine.repo_is_mutating(&canonical).await,
        "raw incremental writer spelling must set the canonical fence key"
    );

    let new_vectors = [published_vector(&canonical)];
    engine
        .vector_index
        .write()
        .await
        .publish_incremental_update(
            &raw_writer_spelling,
            &[],
            &new_vectors,
            test_identity().as_key_string(),
            &[],
        );
    assert!(!engine.repo_is_mutating(&canonical).await);

    engine
        .vector_index
        .write()
        .await
        .begin_full_update(&raw_writer_spelling);
    assert!(
        engine.repo_is_mutating(&canonical).await,
        "raw full writer spelling must set the canonical fence key"
    );

    engine
        .vector_index
        .write()
        .await
        .abort_update_fail_closed(&raw_writer_spelling);
    assert!(
        engine.repo_is_mutating(&canonical).await,
        "raw abort spelling must preserve the canonical fail-closed fence"
    );

    let replacement = [published_vector(&canonical)];
    engine.vector_index.write().await.publish_full_update(
        &raw_writer_spelling,
        &replacement,
        test_identity().as_key_string(),
        &[],
    );
    assert!(!engine.repo_is_mutating(&canonical).await);
}

/// S10/D9: a persisted shard file whose CURRENT pointer survived a later run
/// (the best-effort delete failed — e.g. Defender holding the file on Windows) and
/// whose chunk count still matches must NOT be mmap'd: the index epoch bumped by
/// that run invalidates it.
#[tokio::test]
async fn stale_persisted_shard_with_same_chunk_count_is_not_mmapped() {
    let home = TempDir::new().expect("tempdir");
    let repo = "/proj/stale-same-count".to_string();
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    seed_repo(&repo_dbs, home.path(), &repo, 1).await;
    let identity = test_identity();
    let vector_index = Arc::new(RwLock::new(ShardedVectorIndex::new(1024 * 1024 * 1024)));

    // First warm builds from DB and persists generation 0 with f0.rs.
    warm_repo_shard(
        &vector_index,
        &repo_dbs,
        home.path(),
        &repo,
        0,
        &identity,
        &[],
    )
    .await
    .expect("first warm");
    vector_index.write().await.evict_repo(&repo);

    // A later run replaced f0.rs with f1.rs (same chunk count) and bumped the
    // epoch, but its CURRENT delete "failed": the old file is still CURRENT.
    let db = store::get_or_open(&repo_dbs, home.path(), &repo, 0)
        .await
        .expect("open");
    store::ops::bump_index_epoch(&db).await.expect("bump epoch");
    db.query(format!(
        "DELETE FROM chunk; CREATE chunk SET file = '{repo}/f1.rs', line_start = 1, \
         line_end = 2, content = 'y', embedding = [0.1, 0.2, 0.3, 0.4], symbol_ref = NONE;"
    ))
    .await
    .expect("replace chunk");

    warm_repo_shard(
        &vector_index,
        &repo_dbs,
        home.path(),
        &repo,
        0,
        &identity,
        &[],
    )
    .await
    .expect("second warm");
    let files: Vec<String> = vector_index
        .read()
        .await
        .search(
            &PROBE_EMBEDDING,
            10,
            Some(&repo),
            std::slice::from_ref(&repo),
        )
        .results
        .into_iter()
        .map(|r| r.chunk_id.file)
        .collect();
    assert_eq!(
        files,
        vec![format!("{repo}/f1.rs")],
        "warm must rebuild from DB, not mmap the stale same-count file"
    );
}

/// S16/D18: a run holding the repo lock (no HTTP request attached) is activity
/// for the idle watchdog.
#[tokio::test]
async fn index_run_counts_as_busy_for_idle_watchdog() {
    let home = TempDir::new().expect("tempdir");
    let repo = crate::store::normalize_repo_path("/proj/busy");
    let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
    let engine = start_test_engine(home.path(), &repo_dbs, &repo).await;
    assert!(!engine.is_index_busy(&repo).await);
    let lock = engine.get_repo_lock_public(&repo).await;
    let guard = lock.lock().await;
    assert!(engine.is_index_busy(&repo).await, "a held run lock is busy");
    drop(guard);
    assert!(!engine.is_index_busy(&repo).await);
}
