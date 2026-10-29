//! Agent entries for session actors. An actor holds a [`Lease`] for as long as it runs; the
//! [`Registrar`] refcounts leases per session id and one background thread rewrites this
//! process's `Holder::Agent` entries to match, so callers never wait on the file lock.
//! The process registrar removes those entries synchronously at exit (see [`release_at_exit`]),
//! since leases still queued for the worker would otherwise outlive the pid.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Once, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use agent_client_protocol as acp;
use chrono::{DateTime, Utc};

use crate::{ActiveSession, Holder, LOCK_ACQUIRE_TIMEOUT, is_pid_alive, with_locked_state};

const RETRY_INTERVAL: Duration = Duration::from_secs(1);

pub struct Registrar {
    shared: Arc<Shared>,
}

/// Keeps the session's agent entry while alive; dropping the last lease for an id removes it.
pub struct Lease {
    shared: Arc<Shared>,
    session_id: acp::SessionId,
}

struct Shared {
    root: PathBuf,
    lock_timeout: Duration,
    retry_interval: Duration,
    state: Mutex<State>,
    cv: Condvar,
}

#[derive(Default)]
struct State {
    held: HashMap<acp::SessionId, Held>,
    generation: u64,
    applied_generation: u64,
    worker_started: bool,
    closed: bool,
    /// Set by [`Registrar::release`]; no agent entry is written afterwards.
    released: bool,
    last_failed: bool,
}

struct Held {
    count: usize,
    cwd: String,
    opened_at: DateTime<Utc>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Registrar {
    /// No thread and no I/O until the first [`Registrar::hold`].
    pub fn new(root: PathBuf) -> Self {
        Self::with_timing(root, LOCK_ACQUIRE_TIMEOUT, RETRY_INTERVAL)
    }

    fn with_timing(root: PathBuf, lock_timeout: Duration, retry_interval: Duration) -> Self {
        Self {
            shared: Arc::new(Shared {
                root,
                lock_timeout,
                retry_interval,
                state: Mutex::new(State::default()),
                cv: Condvar::new(),
            }),
        }
    }

    pub fn hold(&self, session_id: acp::SessionId, cwd: String) -> Lease {
        let mut state = self.shared.lock();
        let held = state
            .held
            .entry(session_id.clone())
            .or_insert_with(|| Held {
                count: 0,
                cwd,
                opened_at: Utc::now(),
            });
        held.count += 1;
        if held.count == 1 {
            state.generation += 1;
            self.shared.cv.notify_all();
        }
        if !state.worker_started && !state.released {
            state.worker_started = true;
            let shared = Arc::clone(&self.shared);
            if let Err(e) = std::thread::Builder::new()
                .name("active-sessions".into())
                .spawn(move || run_worker(&shared))
            {
                tracing::warn!(error = %e, "active-session registrar thread failed to start");
            }
        }
        Lease {
            shared: Arc::clone(&self.shared),
            session_id,
        }
    }

    pub fn is_held(&self, session_id: &acp::SessionId) -> bool {
        self.shared.lock().held.contains_key(session_id)
    }

    /// Waits until the file reflects every lease change made so far.
    #[doc(hidden)]
    pub fn wait_synced(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut state = self.shared.lock();
        loop {
            if state.applied_generation == state.generation && !state.last_failed {
                return true;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return false;
            }
            state = self
                .shared
                .cv
                .wait_timeout(state, remaining)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    /// Removes this process's agent entries now and stops writing them, whatever leases are
    /// still held. For process exit and exec, where queued lease drops would never be applied.
    /// Returns false if the lock stayed busy past the timeout.
    pub fn release(&self) -> bool {
        {
            let mut state = self.shared.lock();
            if state.released {
                return true;
            }
            state.released = true;
            state.closed = true;
            self.shared.cv.notify_all();
            if !state.worker_started {
                return true;
            }
        }
        let me = std::process::id();
        match with_locked_state(&self.shared.root, self.shared.lock_timeout, |sessions| {
            sessions.retain(|s| is_pid_alive(s.pid) && !(s.pid == me && s.holder == Holder::Agent));
        }) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(error = %e, "active-session registry release failed");
                false
            }
        }
    }
}

impl Drop for Registrar {
    fn drop(&mut self) {
        self.shared.lock().closed = true;
        self.shared.cv.notify_all();
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let mut state = self.shared.lock();
        let Some(held) = state.held.get_mut(&self.session_id) else {
            return;
        };
        held.count = held.count.saturating_sub(1);
        if held.count == 0 {
            state.held.remove(&self.session_id);
            state.generation += 1;
            self.shared.cv.notify_all();
        }
    }
}

/// Level-triggered: each pass writes the current lease set, so flaps collapse and entries an
/// older binary stripped are restored on the next change.
fn run_worker(shared: &Shared) {
    let me = std::process::id();
    let mut file_has_ours = false;
    loop {
        let mut state = shared.lock();
        while state.generation == state.applied_generation && !state.closed {
            state = shared
                .cv
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        if state.generation == state.applied_generation {
            return;
        }
        let generation = state.generation;
        let snapshot: Vec<ActiveSession> = state
            .held
            .iter()
            .map(|(session_id, held)| ActiveSession {
                session_id: session_id.clone(),
                pid: me,
                cwd: held.cwd.clone(),
                opened_at: held.opened_at,
                holder: Holder::Agent,
            })
            .collect();
        drop(state);

        let outcome = if snapshot.is_empty() && !file_has_ours {
            Ok(())
        } else {
            let has_ours = !snapshot.is_empty();
            with_locked_state(&shared.root, shared.lock_timeout, |sessions| {
                sessions
                    .retain(|s| is_pid_alive(s.pid) && !(s.pid == me && s.holder == Holder::Agent));
                // Checked under the file lock: a release that set the flag first then writes
                // after us, and one that sets it later is ordered after this write anyway.
                if !shared.lock().released {
                    sessions.extend(snapshot);
                }
            })
            .map(|()| file_has_ours = has_ours)
        };

        let mut state = shared.lock();
        match outcome {
            Ok(()) => {
                state.applied_generation = generation;
                state.last_failed = false;
                shared.cv.notify_all();
            }
            Err(e) => {
                tracing::warn!(error = %e, "active-session registry update failed; retrying");
                state.last_failed = true;
                shared.cv.notify_all();
                if state.closed {
                    return;
                }
                drop(
                    shared
                        .cv
                        .wait_timeout(state, shared.retry_interval)
                        .unwrap_or_else(PoisonError::into_inner),
                );
            }
        }
    }
}

static REGISTRAR: OnceLock<Registrar> = OnceLock::new();

fn global() -> &'static Registrar {
    REGISTRAR.get_or_init(|| Registrar::new(xai_grok_config::grok_home()))
}

/// [`Registrar::hold`] on the process registrar rooted at the grok home. The first call
/// installs an exit hook so `exit` and a return from `main` release this process's entries.
pub fn hold(session_id: acp::SessionId, cwd: String) -> Lease {
    install_exit_hook();
    global().hold(session_id, cwd)
}

/// [`Registrar::release`] on the process registrar, if any lease was ever taken. Call before
/// `exec`, which keeps the pid and skips exit hooks.
pub fn release_at_exit() -> bool {
    REGISTRAR.get().is_none_or(Registrar::release)
}

fn install_exit_hook() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        #[cfg(unix)]
        {
            extern "C" fn release_on_exit() {
                let _ = std::panic::catch_unwind(release_at_exit);
            }
            // SAFETY: registers a plain `extern "C" fn()` that does not unwind.
            if unsafe { libc::atexit(release_on_exit) } != 0 {
                tracing::warn!("active-session exit hook failed to install");
            }
        }
    });
}

/// Roots the process registrar at a fresh temp directory so test binaries that spawn session
/// actors never touch the real grok home. Must run before the first [`hold`] (e.g. in a ctor).
#[doc(hidden)]
pub fn redirect_to_temp_for_tests() {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let root = std::env::temp_dir().join(format!(
        "grok-active-sessions-test-{}-{nanos}",
        std::process::id()
    ));
    let _ = REGISTRAR.set(Registrar::new(root));
}

#[doc(hidden)]
pub fn global_root() -> PathBuf {
    global().shared.root.clone()
}

/// [`Registrar::wait_synced`] on the process registrar.
#[doc(hidden)]
pub fn wait_synced(timeout: Duration) -> bool {
    global().wait_synced(timeout)
}

pub fn is_held(session_id: &acp::SessionId) -> bool {
    global().is_held(session_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DATA_FILENAME, LOCK_FILENAME, list_in, open_lock_file, register_in, try_unregister_in,
    };
    use tempfile::TempDir;

    const SYNC: Duration = Duration::from_secs(5);

    fn sid(id: &str) -> acp::SessionId {
        acp::SessionId::new(id)
    }

    fn entries(root: &std::path::Path) -> Vec<(String, u32, Holder)> {
        list_in(root)
            .unwrap()
            .into_iter()
            .map(|s| (s.session_id.0.to_string(), s.pid, s.holder))
            .collect()
    }

    fn agent(id: &str) -> (String, u32, Holder) {
        (id.to_string(), std::process::id(), Holder::Agent)
    }

    #[test]
    fn lease_registers_and_unregisters() {
        let dir = TempDir::new().unwrap();
        let registrar = Registrar::new(dir.path().to_path_buf());
        let lease = registrar.hold(sid("s"), "/work".into());
        assert!(registrar.wait_synced(SYNC));
        let listed = list_in(dir.path()).unwrap();
        assert_eq!(vec![agent("s")], entries(dir.path()));
        assert!(listed.iter().all(|s| s.cwd == "/work"));

        drop(lease);
        assert!(registrar.wait_synced(SYNC));
        assert!(entries(dir.path()).is_empty());
    }

    #[test]
    fn overlapping_leases_refcount() {
        let dir = TempDir::new().unwrap();
        let registrar = Registrar::new(dir.path().to_path_buf());
        let a = registrar.hold(sid("s"), "/work".into());
        let b = registrar.hold(sid("s"), "/work".into());
        assert!(registrar.wait_synced(SYNC));
        assert_eq!(vec![agent("s")], entries(dir.path()));

        drop(a);
        assert!(registrar.wait_synced(SYNC));
        assert!(registrar.is_held(&sid("s")));
        assert_eq!(vec![agent("s")], entries(dir.path()));

        drop(b);
        assert!(registrar.wait_synced(SYNC));
        assert!(!registrar.is_held(&sid("s")));
        assert!(entries(dir.path()).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn leader_keying_different_pids() {
        let dir = TempDir::new().unwrap();
        let me = std::process::id();
        let parent = std::os::unix::process::parent_id();
        let client = |pid| ActiveSession {
            session_id: sid("s"),
            pid,
            cwd: "/work".into(),
            opened_at: Utc::now(),
            holder: Holder::Client,
        };
        register_in(dir.path(), client(parent)).unwrap();
        register_in(dir.path(), client(me)).unwrap();
        let registrar = Registrar::new(dir.path().to_path_buf());
        let lease = registrar.hold(sid("s"), "/work".into());
        assert!(registrar.wait_synced(SYNC));
        assert_eq!(3, entries(dir.path()).len());

        assert!(try_unregister_in(dir.path(), &sid("s")).unwrap());
        let mut left = entries(dir.path());
        left.sort_by_key(|(_, pid, holder)| (*pid, *holder as u8));
        let mut want = vec![("s".to_string(), parent, Holder::Client), agent("s")];
        want.sort_by_key(|(_, pid, holder)| (*pid, *holder as u8));
        assert_eq!(want, left);

        drop(lease);
        assert!(registrar.wait_synced(SYNC));
        assert_eq!(
            vec![("s".to_string(), parent, Holder::Client)],
            entries(dir.path())
        );
    }

    #[test]
    fn lease_retries_after_lock_contention() {
        let dir = TempDir::new().unwrap();
        let registrar = Registrar::with_timing(
            dir.path().to_path_buf(),
            Duration::from_millis(20),
            Duration::from_millis(50),
        );
        let holder = open_lock_file(&dir.path().join(LOCK_FILENAME)).unwrap();
        holder.lock().unwrap();
        let _lease = registrar.hold(sid("s"), "/work".into());
        assert!(!registrar.wait_synced(Duration::from_millis(300)));

        holder.unlock().unwrap();
        assert!(registrar.wait_synced(SYNC));
        assert_eq!(vec![agent("s")], entries(dir.path()));
    }

    #[test]
    fn release_removes_entries_of_live_leases_and_stops_writes() {
        let dir = TempDir::new().unwrap();
        let registrar = Registrar::new(dir.path().to_path_buf());
        let held = registrar.hold(sid("s"), "/work".into());
        assert!(registrar.wait_synced(SYNC));
        assert_eq!(vec![agent("s")], entries(dir.path()));

        assert!(registrar.release());
        assert!(entries(dir.path()).is_empty());
        let late = registrar.hold(sid("late"), "/work".into());
        drop(held);
        std::thread::sleep(Duration::from_millis(100));
        assert!(entries(dir.path()).is_empty());
        drop(late);
    }

    #[test]
    fn no_file_without_leases() {
        let dir = TempDir::new().unwrap();
        let registrar = Registrar::new(dir.path().to_path_buf());
        assert!(registrar.wait_synced(SYNC));
        drop(registrar);
        assert!(!dir.path().join(DATA_FILENAME).exists());
        assert!(!dir.path().join(LOCK_FILENAME).exists());
    }

    #[test]
    fn reconcile_prunes_dead_pids() {
        let dir = TempDir::new().unwrap();
        register_in(
            dir.path(),
            ActiveSession {
                session_id: sid("crashed"),
                pid: 2_000_000_000,
                cwd: "/work".into(),
                opened_at: Utc::now(),
                holder: Holder::Client,
            },
        )
        .unwrap();
        let registrar = Registrar::new(dir.path().to_path_buf());
        let _lease = registrar.hold(sid("s"), "/work".into());
        assert!(registrar.wait_synced(SYNC));
        assert_eq!(vec![agent("s")], entries(dir.path()));
    }
}
