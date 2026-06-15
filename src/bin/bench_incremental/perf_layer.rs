//! In-process capture of the pipeline's `PERF SUMMARY incremental` tracing event.
//!
//! The incremental per-stage breakdown is emitted by `IndexPipeline::run` as a
//! single `info!(... "PERF SUMMARY incremental")` event with all stage timings as
//! structured `u64` fields. Rather than re-parse stderr text (fragile), we attach
//! a custom tracing `Layer` that pattern-matches that event by message and records
//! its numeric fields into a shared slot. The bench binary reads the slot after the
//! incremental run completes and re-prints it to stdout in a stable format.

use std::sync::{Arc, Mutex};

use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

/// Captured per-stage timings (milliseconds) plus the resolve-set size (a count).
/// Field names mirror `IndexPipelineStats` / the `PERF SUMMARY incremental` event.
#[derive(Debug, Clone, Default)]
pub struct IncrPerf {
    pub incr_walk_ms: u64,
    pub incr_meta_load_ms: u64,
    pub incr_predelete_callers_ms: u64,
    pub incr_delete_bulk_ms: u64,
    pub incr_streaming_ms: u64,
    pub incr_phase2_total_ms: u64,
    pub incr_p2_symname_ms: u64,
    pub incr_p2_dir2_scan_ms: u64,
    pub incr_p2_delete_calls_ms: u64,
    pub incr_p2_reresolve_ms: u64,
    pub incr_resolve_set_size: u64,
}

impl IncrPerf {
    /// Sum of the six top-level pipeline stages — the incremental pipeline wall
    /// time we gate on. phase2_total already includes the p2_* sub-stages, so the
    /// sub-stages are NOT re-added (that would double-count).
    pub fn total_ms(&self) -> u64 {
        self.incr_walk_ms
            + self.incr_meta_load_ms
            + self.incr_predelete_callers_ms
            + self.incr_delete_bulk_ms
            + self.incr_streaming_ms
            + self.incr_phase2_total_ms
    }
}

/// Shared, thread-safe slot the layer writes into and the binary reads from.
pub type PerfCapture = Arc<Mutex<Option<IncrPerf>>>;

/// A tracing `Layer` that captures exactly the `PERF SUMMARY incremental` event.
pub struct PerfLayer {
    capture: PerfCapture,
}

impl PerfLayer {
    pub fn new(capture: PerfCapture) -> Self {
        Self { capture }
    }
}

/// Visitor that records the message + every recognized u64/i64 field of an event.
#[derive(Default)]
struct FieldVisitor {
    message: String,
    perf: IncrPerf,
}

impl FieldVisitor {
    fn set(&mut self, name: &str, value: u64) {
        match name {
            "incr_walk_ms" => self.perf.incr_walk_ms = value,
            "incr_meta_load_ms" => self.perf.incr_meta_load_ms = value,
            "incr_predelete_callers_ms" => self.perf.incr_predelete_callers_ms = value,
            "incr_delete_bulk_ms" => self.perf.incr_delete_bulk_ms = value,
            "incr_streaming_ms" => self.perf.incr_streaming_ms = value,
            "incr_phase2_total_ms" => self.perf.incr_phase2_total_ms = value,
            "incr_p2_symname_ms" => self.perf.incr_p2_symname_ms = value,
            "incr_p2_dir2_scan_ms" => self.perf.incr_p2_dir2_scan_ms = value,
            "incr_p2_delete_calls_ms" => self.perf.incr_p2_delete_calls_ms = value,
            "incr_p2_reresolve_ms" => self.perf.incr_p2_reresolve_ms = value,
            "incr_resolve_set_size" => self.perf.incr_resolve_set_size = value,
            _ => {}
        }
    }
}

impl Visit for FieldVisitor {
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.set(field.name(), value);
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        // tracing may record some integer fields as i64; clamp negatives to 0.
        self.set(field.name(), value.max(0) as u64);
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        // The `message` of an `info!(... "literal")` event arrives via Debug.
        if field.name() == "message" {
            self.message = format!("{value:?}");
        }
    }
}

impl<S: Subscriber> Layer<S> for PerfLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        // The message captured via Debug is wrapped in quotes (e.g. "\"PERF
        // SUMMARY incremental\"") — match on a contains() so both shapes work.
        if visitor.message.contains("PERF SUMMARY incremental")
            && let Ok(mut slot) = self.capture.lock()
        {
            *slot = Some(visitor.perf);
        }
    }
}
