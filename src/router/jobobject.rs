//! Windows Job Object: kill-on-close worker containment.
//!
//! Each worker self-exits after its idle window — that is the PRIMARY reclaim
//! mechanism and it is OS-agnostic. The Job Object is the SECONDARY guarantee
//! for one specific failure: the ROUTER process dying (crash, kill -9, power
//! pull) while workers are alive. Without it, those workers would be orphaned —
//! still holding RocksDB LOCKs — until their own idle timers fire (or forever if
//! they were mid-request). With a Job Object created with
//! `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, the OS terminates every assigned worker
//! the instant the router's last handle to the job closes, which happens
//! automatically when the router process exits. No orphaned LOCK holders.
//!
//! On non-Windows this is a no-op shim: Unix has `PR_SET_PDEATHSIG` / process
//! groups for the same job, but the current deployment target is Windows (per
//! the project's Defender/LOCK notes) and worker idle self-exit already covers
//! the normal path everywhere. Keeping a no-op here lets the router code call
//! `assign` unconditionally.

#[cfg(windows)]
mod imp {
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;

    use tracing::{info, warn};
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };

    /// Owns a Windows Job Object handle configured for kill-on-close. Held by the
    /// router for its whole lifetime; dropping it (router exit) closes the handle
    /// and the OS kills every assigned worker.
    pub struct JobObject {
        handle: HANDLE,
    }

    // The HANDLE is an opaque OS handle the router owns for its lifetime; sharing
    // the JobObject across the router's async tasks is safe (assignment is a
    // one-shot syscall per child, and the handle is never mutated after creation).
    unsafe impl Send for JobObject {}
    unsafe impl Sync for JobObject {}

    impl JobObject {
        /// Create a kill-on-close job object. Returns `None` (logged) if the OS
        /// calls fail — the router still runs; it just loses the crash-orphan
        /// guarantee and relies on worker idle self-exit alone.
        pub fn new() -> Option<Self> {
            // SAFETY: standard Win32 job-object creation. Null name + null
            // security attrs → an unnamed job owned by this process.
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if handle.is_null() {
                warn!("CreateJobObjectW failed; worker crash-orphan guarantee disabled");
                return None;
            }

            // Configure kill-on-close: when the router's last handle to the job
            // closes (i.e. the router exits), terminate all assigned processes.
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

            // SAFETY: `info` is a correctly-sized, zero-initialized struct of the
            // class we name; length matches its size.
            let ok = unsafe {
                SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const core::ffi::c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if ok == 0 {
                warn!("SetInformationJobObject(kill-on-close) failed; guarantee disabled");
                // Close the handle we created; we won't use it.
                unsafe {
                    windows_sys::Win32::Foundation::CloseHandle(handle);
                }
                return None;
            }

            info!("worker Job Object created (kill-on-close)");
            Some(Self { handle })
        }

        /// Assign a spawned worker to the job so it dies with the router.
        /// Best-effort: a failure is logged but does not abort the spawn (the
        /// worker's own idle self-exit still bounds its lifetime).
        pub fn assign(&self, child: &Child) {
            let child_handle = child.as_raw_handle() as HANDLE;
            // SAFETY: `child_handle` is a live process handle owned by `child`
            // (valid for the borrow); `self.handle` is our job object.
            let ok = unsafe { AssignProcessToJobObject(self.handle, child_handle) };
            if ok == 0 {
                warn!(
                    pid = child.id(),
                    "AssignProcessToJobObject failed; worker not crash-contained"
                );
            }
        }
    }

    impl Drop for JobObject {
        fn drop(&mut self) {
            // Closing the handle triggers kill-on-close for all assigned workers.
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(self.handle);
            }
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use std::process::Child;

    /// No-op job object on non-Windows. Worker idle self-exit is the reclaim
    /// path; crash-orphan containment via process groups is left for a future
    /// Unix deployment.
    pub struct JobObject;

    impl JobObject {
        pub fn new() -> Option<Self> {
            Some(Self)
        }
        pub fn assign(&self, _child: &Child) {}
    }
}

pub use imp::JobObject;
