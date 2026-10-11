//! Bounded in-memory session store for the MCP streamable-HTTP transport.
//!
//! ## Why this exists (the bug it fixes)
//!
//! rmcp's `LocalSessionManager` keeps live sessions in a `HashMap` with a
//! `keep_alive` idle timeout (default 5 min). When a client goes quiet for
//! longer than that, the session worker exits with `IdleTimeout` and
//! `close_session` removes the entry from that map. With a *store-less*
//! manager, the very next request carrying the now-stale `mcp-session-id`
//! finds `has_session == false`, there is nothing to restore from, and the
//! client receives `404 "Session not found"`. Clients (e.g. Claude Code) hold
//! a session id across long idle gaps, so the failure is intermittent — it
//! only fires when the gap exceeds the idle window. For a server that runs for
//! weeks as an always-on service, that is not acceptable: MCP must not break.
//!
//! ## How a store fixes it (root cause, not a longer timeout)
//!
//! rmcp's transport supports a pluggable [`SessionStore`]. When one is
//! configured, the lifecycle becomes self-healing:
//!   - on `initialize`, the transport persists the client's `initialize_params`
//!     here (keyed by the unique session id);
//!   - an idle timeout still drops the *live worker* (cheap — bounds the number
//!     of resident channels/tasks), but it does **not** delete the store entry
//!     (only an explicit client `DELETE` does);
//!   - the next request with the stale id misses the live map, then
//!     `try_restore_from_store` loads the params from here and transparently
//!     re-creates the worker and replays the handshake. The client never sees
//!     an error.
//!
//! This lets us keep rmcp's short default `keep_alive` (so the count of *live*
//! workers stays bounded) while still never returning "Session not found".
//!
//! ## Bounded memory (project invariant)
//!
//! This server indexes Linux-kernel / Chromium-scale repos and must keep memory
//! bounded regardless of load. Each entry is tiny (just `initialize_params`),
//! but an always-on server serving many short-lived clients over weeks would
//! grow this map without limit. So it is an LRU bounded at [`MAX_SESSIONS`],
//! mirroring the per-repo vector-shard LRU: on insert past the cap we evict the
//! least-recently-used entry. An evicted client simply re-initializes on its
//! next request (a fresh session id), which is correct, not an error.
//!
//! ## Multi-client safety
//!
//! Session ids are globally unique (rmcp's `session_id()`), so every client —
//! and every concurrent connection — has an independent entry. A single shared
//! store instance backs both the global `/mcp` endpoint and the per-repo
//! services; there is no cross-client interference. All state lives behind a
//! single `RwLock`, so concurrent `load`/`store`/`delete` are serialized.
//!
//! ## Disk persist (worker scale-to-zero)
//!
//! The in-memory map dies with the process. Process-per-project workers
//! self-exit after `worker_idle_secs` (default 5 min), so the next MCP call
//! hits a *new* process. [`with_persist`] writes each session as a JSON file
//! under a directory; `load` lazy-reads that file when memory misses. rmcp's
//! `try_restore_from_store` then replays `initialize` — the client keeps its
//! `mcp-session-id` and never sees 404. Corrupt/missing/unreadable files log
//! and return `None` (404), never panic.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::RwLock;

use rmcp::transport::streamable_http_server::session::store::{
    SessionState, SessionStore, SessionStoreError,
};

/// Upper bound on retained sessions. Each entry is only the client's
/// `initialize_params` (a few hundred bytes), so 8192 is generous yet keeps
/// worst-case memory in the low single-digit megabytes. Far more than the
/// number of *live* workers a single-user (or small-team) install will ever
/// hold concurrently; the cap exists purely to bound an always-on server.
pub const MAX_SESSIONS: usize = 8192;

/// On-disk session files live at `<data_dir>/mcp-sessions/<session-id>`.
/// Shared by the always-on router (`/mcp`) and each worker (`/mcp-repo`).
pub fn mcp_sessions_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("mcp-sessions")
}

/// A single stored session plus its recency stamp (for LRU eviction).
#[derive(Debug)]
struct Entry {
    state: SessionState,
    /// Monotonic stamp bumped on insert/access. Lowest = least recently used.
    last_used: u64,
}

/// Bounded, in-memory, LRU [`SessionStore`]. Optional disk dir for
/// cross-process restore (see [`BoundedSessionStore::with_persist`]).
#[derive(Debug, Default)]
pub struct BoundedSessionStore {
    inner: RwLock<Inner>,
    persist_dir: Option<PathBuf>,
}

#[derive(Debug, Default)]
struct Inner {
    sessions: HashMap<String, Entry>,
    /// Monotonic recency counter. Never reset; u64 won't wrap in any realistic
    /// server lifetime (would need ~10^19 store ops).
    clock: u64,
}

impl Inner {
    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    /// Evict least-recently-used entries until strictly below `MAX_SESSIONS`,
    /// leaving room for one new insert. Returns the evicted ids so the caller
    /// can drop their persist files. O(n) per eviction, but evictions only
    /// happen on `store` (session creation), which is rare relative to
    /// `load` (every restore) — never on the hot query path.
    fn evict_to_cap(&mut self) -> Vec<String> {
        let mut victims = Vec::new();
        while self.sessions.len() >= MAX_SESSIONS {
            let Some(victim) = self
                .sessions
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(id, _)| id.clone())
            else {
                break;
            };
            self.sessions.remove(&victim);
            victims.push(victim);
        }
        victims
    }
}

impl BoundedSessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Persist each session as a JSON file under `dir` so a new process
    /// (worker respawn) can restore it via [`SessionStore::load`].
    pub fn with_persist(dir: impl AsRef<Path>) -> Self {
        Self {
            inner: RwLock::new(Inner::default()),
            persist_dir: Some(dir.as_ref().to_path_buf()),
        }
    }

    /// Test/diagnostic helper: number of currently retained sessions.
    #[cfg(test)]
    pub async fn session_count(&self) -> usize {
        self.inner.read().await.sessions.len()
    }
}

/// rmcp session ids are UUID v4 (hex + hyphens). Reject anything else so a
/// hostile id cannot escape `persist_dir`.
fn session_file_path(dir: &Path, session_id: &str) -> Option<PathBuf> {
    if session_id.is_empty()
        || !session_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return None;
    }
    Some(dir.join(session_id))
}

fn persist_to_disk(dir: &Path, session_id: &str, state: &SessionState) {
    let Some(path) = session_file_path(dir, session_id) else {
        tracing::warn!(
            session_id,
            "MCP session id is not a safe filename; skip persist"
        );
        return;
    };
    if let Err(e) = write_session_atomic(&path, state) {
        tracing::warn!(
            error = %e,
            path = %path.display(),
            "MCP session persist failed; in-memory entry kept"
        );
    }
}

fn load_from_disk(dir: &Path, session_id: &str) -> Option<SessionState> {
    let path = session_file_path(dir, session_id)?;
    match fs::read(&path) {
        Ok(bytes) => match serde_json::from_slice::<SessionState>(&bytes) {
            Ok(state) => Some(state),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "MCP session store file corrupt; treating as missing"
                );
                None
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "MCP session store read failed; treating as missing"
            );
            None
        }
    }
}

fn delete_from_disk(dir: &Path, session_id: &str) {
    let Some(path) = session_file_path(dir, session_id) else {
        return;
    };
    match fs::remove_file(&path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "MCP session persist delete failed"
            );
        }
    }
}

/// Atomic write of one session file.
///
/// Unix: mode `0o600` set on the tempfile *and* re-asserted after persist
/// (rename can preserve a previous target's more-open mode).
/// Windows: replace the DACL with a *protected* current-user-only ACL
/// (no inherited Users/Admin/SYSTEM ACEs). Applied on the tempfile and
/// re-asserted after persist. Admin/SYSTEM can still take ownership; the
/// threat model is unprivileged peer users, same as 0o600.
fn write_session_atomic(target: &Path, state: &SessionState) -> std::io::Result<()> {
    let parent = target.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "session path has no parent",
        )
    })?;
    fs::create_dir_all(parent)?;
    let temp = tempfile::NamedTempFile::new_in(parent)?;
    let json = serde_json::to_vec_pretty(state)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    fs::write(temp.path(), json)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(windows)]
    restrict_dacl_to_current_user(temp.path())?;
    let target_path = target.to_path_buf();
    temp.persist(&target_path).map_err(|e| e.error)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&target_path, fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(windows)]
    restrict_dacl_to_current_user(&target_path)?;
    Ok(())
}

/// Replace `path`'s DACL with a single ACCESS_ALLOWED ACE for the current
/// user and mark it protected so parent-directory ACEs are not inherited.
#[cfg(windows)]
fn restrict_dacl_to_current_user(path: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{CloseHandle, ERROR_SUCCESS, GENERIC_ALL, LocalFree};
    use windows_sys::Win32::Security::Authorization::{
        EXPLICIT_ACCESS_W, NO_MULTIPLE_TRUSTEE, SE_FILE_OBJECT, SET_ACCESS, SetEntriesInAclW,
        SetNamedSecurityInfoW, TRUSTEE_IS_SID, TRUSTEE_IS_USER, TRUSTEE_W,
    };
    use windows_sys::Win32::Security::{
        ACL, DACL_SECURITY_INFORMATION, GetTokenInformation, NO_INHERITANCE,
        PROTECTED_DACL_SECURITY_INFORMATION, TOKEN_QUERY, TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    let mut token = std::ptr::null_mut();
    // SAFETY: process handle is the calling process; TOKEN_QUERY is valid.
    let ok = unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) };
    if ok == 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut needed = 0u32;
    unsafe {
        GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut needed);
    }
    let mut buf = vec![0u8; needed as usize];
    let ok = unsafe {
        GetTokenInformation(
            token,
            TokenUser,
            buf.as_mut_ptr() as *mut core::ffi::c_void,
            needed,
            &mut needed,
        )
    };
    unsafe { CloseHandle(token) };
    if ok == 0 {
        return Err(std::io::Error::last_os_error());
    }
    let token_user = unsafe { &*(buf.as_ptr() as *const TOKEN_USER) };
    let sid = token_user.User.Sid;

    let mut access = unsafe { std::mem::zeroed::<EXPLICIT_ACCESS_W>() };
    access.grfAccessPermissions = GENERIC_ALL;
    access.grfAccessMode = SET_ACCESS;
    access.grfInheritance = NO_INHERITANCE;
    access.Trustee = TRUSTEE_W {
        pMultipleTrustee: std::ptr::null_mut(),
        MultipleTrusteeOperation: NO_MULTIPLE_TRUSTEE,
        TrusteeForm: TRUSTEE_IS_SID,
        TrusteeType: TRUSTEE_IS_USER,
        ptstrName: sid as windows_sys::core::PWSTR,
    };

    let mut new_acl: *mut ACL = std::ptr::null_mut();
    // SAFETY: one EXPLICIT_ACCESS, no old ACL (null), out-param for LocalAlloc'd ACL.
    let err = unsafe { SetEntriesInAclW(1, &access, std::ptr::null(), &mut new_acl) };
    if err != ERROR_SUCCESS {
        return Err(std::io::Error::from_raw_os_error(err as i32));
    }

    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let err = unsafe {
        SetNamedSecurityInfoW(
            wide.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            new_acl,
            std::ptr::null_mut(),
        )
    };
    unsafe { LocalFree(new_acl as _) };
    if err != ERROR_SUCCESS {
        return Err(std::io::Error::from_raw_os_error(err as i32));
    }
    Ok(())
}

#[async_trait::async_trait]
impl SessionStore for BoundedSessionStore {
    async fn load(&self, session_id: &str) -> Result<Option<SessionState>, SessionStoreError> {
        let mut inner = self.inner.write().await;
        let stamp = inner.tick();
        if let Some(entry) = inner.sessions.get_mut(session_id) {
            entry.last_used = stamp;
            return Ok(Some(entry.state.clone()));
        }
        let Some(dir) = self.persist_dir.as_ref() else {
            return Ok(None);
        };
        let Some(state) = load_from_disk(dir, session_id) else {
            return Ok(None);
        };
        let victims = if inner.sessions.len() >= MAX_SESSIONS {
            inner.evict_to_cap()
        } else {
            Vec::new()
        };
        inner.sessions.insert(
            session_id.to_owned(),
            Entry {
                state: state.clone(),
                last_used: stamp,
            },
        );
        if let Some(dir) = self.persist_dir.as_ref() {
            for id in victims {
                delete_from_disk(dir, &id);
            }
        }
        Ok(Some(state))
    }

    async fn store(&self, session_id: &str, state: &SessionState) -> Result<(), SessionStoreError> {
        let mut inner = self.inner.write().await;
        // Only evict when inserting a genuinely new id; re-storing an existing
        // session must not evict a peer.
        let victims = if !inner.sessions.contains_key(session_id) {
            inner.evict_to_cap()
        } else {
            Vec::new()
        };
        let stamp = inner.tick();
        inner.sessions.insert(
            session_id.to_owned(),
            Entry {
                state: state.clone(),
                last_used: stamp,
            },
        );
        if let Some(dir) = self.persist_dir.as_ref() {
            persist_to_disk(dir, session_id, state);
            for id in victims {
                delete_from_disk(dir, &id);
            }
        }
        Ok(())
    }

    async fn delete(&self, session_id: &str) -> Result<(), SessionStoreError> {
        self.inner.write().await.sessions.remove(session_id);
        if let Some(dir) = self.persist_dir.as_ref() {
            delete_from_disk(dir, session_id);
        }
        Ok(())
    }
}

/// Shared handle type used by the server.
pub type SharedSessionStore = Arc<BoundedSessionStore>;

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::InitializeRequestParams;

    fn state() -> SessionState {
        SessionState::new(InitializeRequestParams::default())
    }

    #[tokio::test]
    async fn store_then_load_roundtrips() {
        let s = BoundedSessionStore::new();
        assert!(s.load("missing").await.unwrap().is_none());
        s.store("a", &state()).await.unwrap();
        assert!(s.load("a").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn delete_removes_entry() {
        let s = BoundedSessionStore::new();
        s.store("a", &state()).await.unwrap();
        s.delete("a").await.unwrap();
        assert!(s.load("a").await.unwrap().is_none());
        // Deleting a missing id is a no-op, not an error.
        s.delete("a").await.unwrap();
    }

    #[tokio::test]
    async fn restore_returns_none_after_idle_close_does_not_delete() {
        // Models the real flow: idle timeout calls close_session (in-memory map
        // only) but never touches the store, so the entry survives for restore.
        let s = BoundedSessionStore::new();
        s.store("sess", &state()).await.unwrap();
        // No delete happened (idle close path) -> still restorable.
        assert!(s.load("sess").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn evicts_lru_when_over_cap_and_keeps_bounded() {
        let s = BoundedSessionStore::new();
        // Fill exactly to cap.
        for i in 0..MAX_SESSIONS {
            s.store(&format!("k{i}"), &state()).await.unwrap();
        }
        assert_eq!(s.session_count().await, MAX_SESSIONS);

        // Touch "k0" so it is most-recently-used; "k1" is now the LRU victim.
        assert!(s.load("k0").await.unwrap().is_some());

        // One more insert must evict the LRU (k1), not k0, and stay bounded.
        s.store("overflow", &state()).await.unwrap();
        assert_eq!(s.session_count().await, MAX_SESSIONS);
        assert!(s.load("k0").await.unwrap().is_some(), "recently-used kept");
        assert!(s.load("overflow").await.unwrap().is_some(), "new kept");
        assert!(s.load("k1").await.unwrap().is_none(), "LRU evicted");
    }

    #[tokio::test]
    async fn re_store_existing_does_not_evict_peer() {
        let s = BoundedSessionStore::new();
        for i in 0..MAX_SESSIONS {
            s.store(&format!("k{i}"), &state()).await.unwrap();
        }
        // Re-storing an existing id must not push us over cap / evict anyone.
        s.store("k0", &state()).await.unwrap();
        assert_eq!(s.session_count().await, MAX_SESSIONS);
        assert!(s.load("k1").await.unwrap().is_some());
    }

    /// Process death (worker scale-to-zero) drops the in-memory map. The next
    /// worker must still restore from what the previous instance wrote to disk.
    #[tokio::test]
    async fn persisted_session_loads_from_new_instance() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        {
            let s = BoundedSessionStore::with_persist(dir.path());
            s.store("sess", &state()).await.unwrap();
        }
        let s2 = BoundedSessionStore::with_persist(dir.path());
        assert!(
            s2.load("sess").await.unwrap().is_some(),
            "a new store instance must load the session written by a previous instance"
        );
    }

    /// A corrupt/unreadable persist file must not panic the worker: treat as
    /// missing so rmcp 404s instead of crashing restore.
    #[tokio::test]
    async fn corrupt_persist_file_is_missing_not_panic() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(dir.path().join("sess"), b"not-json{{{{").expect("write garbage");
        let s = BoundedSessionStore::with_persist(dir.path());
        assert!(
            s.load("sess").await.unwrap().is_none(),
            "corrupt persist file must load as None, not error/panic"
        );
    }

    #[tokio::test]
    async fn delete_removes_persist_file_so_respawn_cannot_restore() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        {
            let s = BoundedSessionStore::with_persist(dir.path());
            s.store("sess", &state()).await.unwrap();
            s.delete("sess").await.unwrap();
        }
        let s2 = BoundedSessionStore::with_persist(dir.path());
        assert!(
            s2.load("sess").await.unwrap().is_none(),
            "explicit delete must drop the persist file, not only the in-memory entry"
        );
    }

    #[tokio::test]
    async fn lru_evict_removes_persist_file() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        {
            let s = BoundedSessionStore::with_persist(dir.path());
            for i in 0..MAX_SESSIONS {
                s.store(&format!("k{i}"), &state()).await.unwrap();
            }
            assert!(s.load("k0").await.unwrap().is_some());
            s.store("overflow", &state()).await.unwrap();
        }
        let s2 = BoundedSessionStore::with_persist(dir.path());
        assert!(
            s2.load("k1").await.unwrap().is_none(),
            "LRU-evicted session must not be restorable from disk"
        );
        assert!(s2.load("k0").await.unwrap().is_some(), "recently-used kept");
        assert!(s2.load("overflow").await.unwrap().is_some(), "new kept");
    }

    #[test]
    fn mcp_sessions_dir_is_under_data_dir() {
        let dir = PathBuf::from("/tmp/ce-data");
        assert_eq!(mcp_sessions_dir(&dir), dir.join("mcp-sessions"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn persist_file_mode_is_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::TempDir::new().expect("tempdir");
        let s = BoundedSessionStore::with_persist(dir.path());
        s.store("sess", &state()).await.unwrap();
        let mode = std::fs::metadata(dir.path().join("sess"))
            .expect("stat")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "session file must be 0o600, got 0o{mode:o}");
    }

    /// Disk persist failing (dir is a file → create_dir_all fails) must not
    /// panic: in-memory entry lives, a new process cannot restore (404).
    #[tokio::test]
    async fn persist_write_failure_keeps_memory_next_process_misses() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let not_a_dir = dir.path().join("blocked");
        std::fs::write(&not_a_dir, b"x").expect("blocker file");
        let s = BoundedSessionStore::with_persist(&not_a_dir);
        s.store("sess", &state())
            .await
            .expect("store must succeed in memory when persist fails");
        assert!(
            s.load("sess").await.unwrap().is_some(),
            "in-memory entry must survive a persist write failure"
        );
        let s2 = BoundedSessionStore::with_persist(&not_a_dir);
        assert!(
            s2.load("sess").await.unwrap().is_none(),
            "next process must miss (404) when persist never landed"
        );
    }

    #[tokio::test]
    async fn restore_same_id_twice_is_idempotent() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        {
            let s = BoundedSessionStore::with_persist(dir.path());
            s.store("sess", &state()).await.unwrap();
        }
        let s2 = BoundedSessionStore::with_persist(dir.path());
        assert!(s2.load("sess").await.unwrap().is_some());
        assert!(
            s2.load("sess").await.unwrap().is_some(),
            "second load of the same id must succeed (memory hit after disk restore)"
        );
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn persist_file_acl_is_current_user_only() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let s = BoundedSessionStore::with_persist(dir.path());
        s.store("sess", &state()).await.unwrap();
        let path = dir.path().join("sess");
        let (ace_count, only_current_user) =
            windows_dacl_ace_count_and_current_user(&path).expect("read DACL");
        assert_eq!(
            ace_count, 1,
            "DACL must not inherit parent ACEs (Users/Admin/SYSTEM); got {ace_count} ACEs"
        );
        assert!(
            only_current_user,
            "the single ACE must be the current user, not an inherited group"
        );
    }

    #[cfg(windows)]
    fn windows_dacl_ace_count_and_current_user(path: &Path) -> Result<(u32, bool), String> {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Foundation::{ERROR_SUCCESS, LocalFree};
        use windows_sys::Win32::Security::Authorization::{GetNamedSecurityInfoW, SE_FILE_OBJECT};
        use windows_sys::Win32::Security::{
            ACCESS_ALLOWED_ACE, ACE_HEADER, ACL, ACL_SIZE_INFORMATION, AclSizeInformation,
            DACL_SECURITY_INFORMATION, EqualSid, GetAce, GetAclInformation, TOKEN_USER,
        };

        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let mut sd = std::ptr::null_mut();
        let mut dacl: *mut ACL = std::ptr::null_mut();
        let err = unsafe {
            GetNamedSecurityInfoW(
                wide.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut dacl,
                std::ptr::null_mut(),
                &mut sd,
            )
        };
        if err != ERROR_SUCCESS {
            return Err(format!("GetNamedSecurityInfoW {err}"));
        }
        if dacl.is_null() {
            unsafe {
                LocalFree(sd as _);
            }
            return Err("null DACL".into());
        }

        let mut info = ACL_SIZE_INFORMATION {
            AceCount: 0,
            AclBytesInUse: 0,
            AclBytesFree: 0,
        };
        let ok = unsafe {
            GetAclInformation(
                dacl,
                &mut info as *mut _ as *mut core::ffi::c_void,
                std::mem::size_of::<ACL_SIZE_INFORMATION>() as u32,
                AclSizeInformation,
            )
        };
        if ok == 0 {
            unsafe {
                LocalFree(sd as _);
            }
            return Err("GetAclInformation failed".into());
        }
        let ace_count = info.AceCount;
        if ace_count != 1 {
            unsafe {
                LocalFree(sd as _);
            }
            return Ok((ace_count, false));
        }

        let mut ace: *mut core::ffi::c_void = std::ptr::null_mut();
        let ok = unsafe { GetAce(dacl, 0, &mut ace) };
        if ok == 0 || ace.is_null() {
            unsafe {
                LocalFree(sd as _);
            }
            return Err("GetAce failed".into());
        }
        let header = unsafe { &*(ace as *const ACE_HEADER) };
        if header.AceType != 0 || (header.AceFlags & 0x10) != 0 {
            unsafe {
                LocalFree(sd as _);
            }
            return Ok((ace_count, false));
        }
        let allowed = unsafe { &*(ace as *const ACCESS_ALLOWED_ACE) };
        let file_sid = std::ptr::addr_of!(allowed.SidStart) as windows_sys::Win32::Security::PSID;

        let current = current_user_sid_buf().inspect_err(|_| unsafe {
            LocalFree(sd as _);
        })?;
        let token_user = unsafe { &*(current.as_ptr() as *const TOKEN_USER) };
        let same = unsafe { EqualSid(file_sid, token_user.User.Sid) } != 0;
        unsafe {
            LocalFree(sd as _);
        }
        Ok((ace_count, same))
    }

    #[cfg(windows)]
    fn current_user_sid_buf() -> Result<Vec<u8>, String> {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::Security::{GetTokenInformation, TOKEN_QUERY, TokenUser};
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

        let mut token = std::ptr::null_mut();
        let ok = unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) };
        if ok == 0 {
            return Err("OpenProcessToken failed".into());
        }
        let mut needed = 0u32;
        unsafe {
            GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut needed);
        }
        let mut buf = vec![0u8; needed as usize];
        let ok = unsafe {
            GetTokenInformation(
                token,
                TokenUser,
                buf.as_mut_ptr() as *mut core::ffi::c_void,
                needed,
                &mut needed,
            )
        };
        unsafe { CloseHandle(token) };
        if ok == 0 {
            return Err("GetTokenInformation failed".into());
        }
        Ok(buf)
    }
}
