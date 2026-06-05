use serde::Serialize;
use tokio::sync::broadcast;

/// Events streamed to the frontend during indexing.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IndexEvent {
    /// Indexing run started for a repo.
    Started {
        repo: String,
        total_files: u64,
        is_rebuild: bool,
    },
    /// A file was parsed (stage 1 complete for this file).
    FileParsed {
        file: String,
        chunks: usize,
        symbols: usize,
        parse_ms: u64,
        queue_wait_ms: u64,
    },
    /// A file was skipped during parsing (read/stat failure).
    FileSkipped {
        file: String,
        reason: String,
    },
    /// A file's chunks were embedded (stage 2 complete for this file).
    FileEmbedded {
        file: String,
        chunks: usize,
        elapsed_ms: u64,
        cached: bool,
        key_hint: String,
    },
    /// A file was written to the DB (stage 3 complete for this file).
    FileStored {
        file: String,
        elapsed_ms: u64,
        queue_wait_ms: u64,
    },
    /// A file completed the full parse+embed+write cycle.
    FileIndexed {
        file: String,
        indexed: u64,
        total: u64,
        total_elapsed_ms: u64,
        status: String,
    },
    /// Phase 2 edge resolution started.
    Phase2Start { repo: String },
    /// Phase 2 edge resolution done.
    Phase2Done { repo: String, elapsed_ms: u64 },
    /// Indexing completed successfully.
    Completed {
        repo: String,
        indexed_files: u64,
        total_files: u64,
        elapsed_ms: u64,
    },
    /// Indexing failed.
    Failed { repo: String, error: String },
}

/// Shared event broadcaster for indexing progress.
#[derive(Clone)]
pub struct IndexEventBus {
    tx: broadcast::Sender<IndexEvent>,
}

impl Default for IndexEventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexEventBus {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1024);
        Self { tx }
    }

    pub fn emit(&self, event: IndexEvent) {
        let _ = self.tx.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<IndexEvent> {
        self.tx.subscribe()
    }
}
