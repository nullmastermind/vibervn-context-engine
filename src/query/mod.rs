pub mod content_fence;
pub mod engine;
pub mod filters;
pub mod graph_expand;
pub mod merger;
pub mod reranker;
pub(crate) mod scope_fallback;

pub use engine::{CodeResult, QueryResult, QueryTiming, RerankInfo, run_query};

use std::collections::HashMap;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use crate::path_in_repo;

/// Find the DB handle whose repo key owns `file`, or the first DB as fallback.
pub(crate) fn find_db_for_file<'a>(
    db_map: &'a HashMap<String, Surreal<Db>>,
    file: &str,
) -> Option<&'a Surreal<Db>> {
    for (repo_path, db) in db_map {
        if path_in_repo(file, repo_path) {
            return Some(db);
        }
    }
    db_map.values().next()
}
