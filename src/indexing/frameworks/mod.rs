//! Framework-aware resolution: pluggable trait + registry for detecting
//! frameworks and extracting additional call edges from framework-specific patterns.
//!
//! Each resolver implements `FrameworkResolver` — a `detect` method that checks
//! whether the framework is in use (based on manifest files / import patterns),
//! and an `extract_edges` method that produces additional `RawEdge`s from source
//! files belonging to that framework.

pub mod django;
pub mod express;
pub mod go_gin;
pub mod react;
pub mod spring;

use std::collections::HashSet;

use crate::parsing::relations::RawEdge;
use crate::parsing::symbols::Symbol;

/// Context provided to framework detection. Contains file listing and
/// an optional file reader for inspecting manifest contents.
pub struct DetectionContext<'a> {
    /// Set of all file paths in the repo (absolute or repo-relative).
    pub file_set: &'a HashSet<String>,
    /// Read a file's content by path. Returns None if file cannot be read.
    pub read_file: &'a dyn Fn(&str) -> Option<String>,
}

/// Trait for pluggable framework resolvers.
///
/// Each implementation detects one framework and extracts edges from its
/// convention-based patterns (routing, DI, component rendering, etc.).
pub trait FrameworkResolver: Send + Sync {
    /// Human-readable name of the framework.
    fn name(&self) -> &str;

    /// Check if this framework is active in the repository.
    /// Called once per full index run; result is cached for the session.
    fn detect(&self, ctx: &DetectionContext) -> bool;

    /// Extract additional edges from a single source file.
    /// Called on files matching the framework's language(s) when the
    /// framework is detected as active.
    fn extract_edges(&self, file_path: &str, source: &str, symbols: &[Symbol]) -> Vec<RawEdge>;
}

/// Registry of all known framework resolvers.
/// Detection runs once; only active resolvers participate in edge extraction.
pub struct FrameworkRegistry {
    resolvers: Vec<Box<dyn FrameworkResolver>>,
    active: Vec<usize>, // indices into `resolvers` that passed detection
    detected: bool,
}

impl FrameworkRegistry {
    /// Create a new registry with all built-in resolvers.
    pub fn new() -> Self {
        let resolvers: Vec<Box<dyn FrameworkResolver>> = vec![
            Box::new(react::ReactResolver),
            Box::new(express::ExpressResolver),
            Box::new(django::DjangoResolver),
            Box::new(spring::SpringResolver),
            Box::new(go_gin::GoGinResolver),
        ];
        Self {
            resolvers,
            active: Vec::new(),
            detected: false,
        }
    }

    /// Run detection on all resolvers. Should be called once per index session.
    pub fn detect(&mut self, ctx: &DetectionContext) {
        self.active.clear();
        for (i, resolver) in self.resolvers.iter().enumerate() {
            if resolver.detect(ctx) {
                self.active.push(i);
            }
        }
        self.detected = true;
    }

    /// Returns true if detection has been run.
    pub fn is_detected(&self) -> bool {
        self.detected
    }

    /// Extract edges from active frameworks for a given file.
    pub fn extract_edges(&self, file_path: &str, source: &str, symbols: &[Symbol]) -> Vec<RawEdge> {
        let mut edges = Vec::new();
        for &idx in &self.active {
            let resolver = &self.resolvers[idx];
            edges.extend(resolver.extract_edges(file_path, source, symbols));
        }
        edges
    }

    /// Additionally activate resolvers by name (a set persisted by an earlier
    /// whole-repo detection). Unknown names are ignored; marks detection as done.
    pub fn activate_names<S: AsRef<str>>(&mut self, names: &[S]) {
        for (i, resolver) in self.resolvers.iter().enumerate() {
            if !self.active.contains(&i) && names.iter().any(|n| n.as_ref() == resolver.name()) {
                self.active.push(i);
            }
        }
        self.detected = true;
    }

    /// Names of active frameworks (persisted so incremental runs keep them).
    pub fn active_names(&self) -> Vec<&str> {
        self.active
            .iter()
            .map(|&i| self.resolvers[i].name())
            .collect()
    }
}

impl Default for FrameworkRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_extracts_nothing() {
        let mut registry = FrameworkRegistry::new();
        let file_set = HashSet::new();
        let ctx = DetectionContext {
            file_set: &file_set,
            read_file: &|_| None,
        };
        registry.detect(&ctx);
        let edges = registry.extract_edges("test.ts", "const x = 1;", &[]);
        assert!(edges.is_empty());
    }

    #[test]
    fn detection_caches_active_resolvers() {
        let mut registry = FrameworkRegistry::new();
        let mut file_set = HashSet::new();
        file_set.insert("package.json".to_string());
        let ctx = DetectionContext {
            file_set: &file_set,
            read_file: &|path| {
                if path == "package.json" {
                    Some(r#"{"dependencies": {"react": "^18.0.0"}}"#.to_string())
                } else {
                    None
                }
            },
        };
        registry.detect(&ctx);
        assert!(registry.is_detected());
        // React should be detected
        assert!(registry.active_names().contains(&"react"));
    }
}
