//! Tracks open sessions (UI clients and session actors) in `<grok home>/active_sessions.json`.
//! A clean exit removes the entry, a crash leaves it behind, and the next [`register`] or
//! [`Lease`] reconcile prunes entries whose PID is dead.

#![deny(clippy::indexing_slicing)]

use std::fs::{self, File, OpenOptions, TryLockError};
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use agent_client_protocol as acp;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

mod lease;
pub use lease::{
    Lease, Registrar, global_root, hold, is_held, redirect_to_temp_for_tests, release_at_exit,
    wait_synced,
};

/// Who in process `pid` has the session open. Entries are keyed by (session_id, pid, holder),
/// so a UI client and the actor in the same or another process never remove each other.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Holder {
    /// A UI or `-p` front end; entries written before the field existed read as this.
    #[default]
    Client,
    /// A running session actor (the writer), written only through [`Lease`].
    Agent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActiveSession {
    pub session_id: acp::SessionId,
    pub pid: u32,
    pub cwd: String,
    pub opened_at: DateTime<Utc>,
    #[serde(default)]
    pub holder: Holder,
}

const DATA_FILENAME: &str = "active_sessions.json";
const LOCK_FILENAME: &str = "active_sessions.lock";
const TMP_FILENAME: &str = "active_sessions.json.tmp";

/// On an NFS home `flock` is a network-lock-manager lock with no lease: a holder killed at the
/// wrong moment strands it forever, so never wait unbounded. A live holder only does one small
/// read-modify-write, so anything held this long is stranded.
const LOCK_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(2);
const LOCK_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// How long a non-blocking unregister waits when the holder is this process's own registrar
/// worker, which keeps the lock only for one small read and rewrite.
const SAME_PROCESS_WAIT: Duration = Duration::from_millis(250);

/// Callers inside [`with_locked_state`] in this process, waiting for or holding the lock.
static IN_PROCESS_LOCKERS: AtomicUsize = AtomicUsize::new(0);

struct InProcessLocker;

impl InProcessLocker {
    fn enter() -> Self {
        IN_PROCESS_LOCKERS.fetch_add(1, Ordering::AcqRel);
        Self
    }
}

impl Drop for InProcessLocker {
    fn drop(&mut self) {
        IN_PROCESS_LOCKERS.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Register a session as active (idempotent by session_id, pid and holder) and prune entries
/// whose PID is dead.
/// Fails with [`io::ErrorKind::TimedOut`] if the lock is still held after `LOCK_ACQUIRE_TIMEOUT`.
pub fn register(session: ActiveSession) -> io::Result<()> {
    register_in(&xai_grok_config::grok_home(), session)
}

/// Non-blocking unregister of this process's client entry, for signal handlers and the UI
/// thread. Waits up to `SAME_PROCESS_WAIT` only when this process's own registrar worker holds
/// the lock. Returns `Ok(false)` on other contention; the orphan stays until this pid dies.
pub fn try_unregister(session_id: &acp::SessionId) -> io::Result<bool> {
    try_unregister_in(&xai_grok_config::grok_home(), session_id)
}

pub fn register_in(root: &Path, session: ActiveSession) -> io::Result<()> {
    with_locked_state(root, LOCK_ACQUIRE_TIMEOUT, |sessions| {
        sessions.retain(|s| {
            is_pid_alive(s.pid)
                && !(s.session_id == session.session_id
                    && s.pid == session.pid
                    && s.holder == session.holder)
        });
        sessions.push(session);
    })
}

pub fn try_unregister_in(root: &Path, session_id: &acp::SessionId) -> io::Result<bool> {
    let me = std::process::id();
    let remove = |sessions: &mut Vec<ActiveSession>| {
        sessions.retain(|s| {
            !(s.session_id == *session_id && s.pid == me && s.holder == Holder::Client)
        });
    };
    let mut outcome = with_locked_state(root, Duration::ZERO, remove);
    if outcome
        .as_ref()
        .is_err_and(|e| e.kind() == io::ErrorKind::TimedOut)
    {
        let wait = if IN_PROCESS_LOCKERS.load(Ordering::Acquire) > 0 {
            SAME_PROCESS_WAIT
        } else {
            Duration::ZERO
        };
        outcome = with_locked_state(root, wait, remove);
    }
    match outcome {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::TimedOut => Ok(false),
        Err(e) => Err(e),
    }
}

pub fn list_in(root: &Path) -> io::Result<Vec<ActiveSession>> {
    let data_path = root.join(DATA_FILENAME);
    read_data_file(&data_path)
}

fn with_locked_state<F, R>(root: &Path, timeout: Duration, mutate: F) -> io::Result<R>
where
    F: FnOnce(&mut Vec<ActiveSession>) -> R,
{
    let lock_path = root.join(LOCK_FILENAME);
    let data_path = root.join(DATA_FILENAME);
    let tmp_path = root.join(TMP_FILENAME);

    let _in_process = InProcessLocker::enter();
    fs::create_dir_all(root)?;
    // Closing the file releases the lock, so every return path unlocks by dropping it.
    let lock_file = open_lock_file(&lock_path)?;
    let deadline = Instant::now() + timeout;
    loop {
        match lock_file.try_lock() {
            Ok(()) => break,
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Error(e)) => return Err(e),
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("active-session registry lock still held after {timeout:?}"),
            ));
        }
        std::thread::sleep(LOCK_POLL_INTERVAL.min(remaining));
    }

    let mut sessions = read_data_file(&data_path)?;
    let result = mutate(&mut sessions);
    write_data_file_atomic(&tmp_path, &data_path, &sessions)?;
    Ok(result)
}

fn open_lock_file(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

fn read_data_file(path: &Path) -> io::Result<Vec<ActiveSession>> {
    match fs::read(path) {
        Ok(bytes) if bytes.is_empty() => Ok(Vec::new()),
        Ok(bytes) => match serde_json::from_slice::<Vec<ActiveSession>>(&bytes) {
            Ok(sessions) => Ok(sessions),
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "active_sessions.json is corrupted, starting with empty list"
                );
                Ok(Vec::new())
            }
        },
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

fn write_data_file_atomic(
    tmp_path: &Path,
    data_path: &Path,
    sessions: &[ActiveSession],
) -> io::Result<()> {
    let json = serde_json::to_string_pretty(sessions)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(tmp_path, json.as_bytes())?;
    fs::rename(tmp_path, data_path).inspect_err(|_| {
        let _ = fs::remove_file(tmp_path);
    })
}

fn is_pid_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        let pid_i = match i32::try_from(pid) {
            Ok(p) if p > 0 => p,
            _ => return false,
        };
        let ret = unsafe { libc::kill(pid_i as libc::pid_t, 0) };
        if ret == 0 {
            return true;
        }
        io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(windows)]
    {
        use windows::Win32::Foundation::CloseHandle;
        use windows::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
        let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) };
        match handle {
            Ok(h) => {
                let _ = unsafe { CloseHandle(h) };
                true
            }
            Err(_) => false,
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        // Conservative: assume alive if we can't check.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_session(id: &str, pid: u32) -> ActiveSession {
        ActiveSession {
            session_id: acp::SessionId::new(id),
            pid,
            cwd: "/tmp/test".into(),
            opened_at: Utc::now(),
            holder: Holder::Client,
        }
    }

    fn with_holder(mut s: ActiveSession, holder: Holder) -> ActiveSession {
        s.holder = holder;
        s
    }

    fn keys(root: &Path) -> Vec<(String, u32, Holder)> {
        let mut keys: Vec<_> = list_in(root)
            .unwrap()
            .into_iter()
            .map(|s| (s.session_id.0.to_string(), s.pid, s.holder))
            .collect();
        keys.sort_by_key(|(id, pid, holder)| (id.clone(), *pid, *holder as u8));
        keys
    }

    #[test]
    fn register_is_idempotent() {
        let dir = TempDir::new().unwrap();
        let s = make_session("s1", std::process::id());
        register_in(dir.path(), s.clone()).unwrap();
        register_in(dir.path(), s).unwrap();
        assert_eq!(1, list_in(dir.path()).unwrap().len());
    }

    #[test]
    fn concurrent_registers_no_corruption() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_path_buf();
        std::thread::scope(|s| {
            for i in 0..10 {
                let p = path.clone();
                s.spawn(move || {
                    register_in(&p, make_session(&format!("s{i}"), std::process::id())).unwrap()
                });
            }
        });
        assert_eq!(10, list_in(dir.path()).unwrap().len());
    }

    #[test]
    fn try_unregister_skips_if_locked() {
        let dir = TempDir::new().unwrap();
        let s = make_session("s1", std::process::id());
        register_in(dir.path(), s.clone()).unwrap();

        let lock_file = open_lock_file(&dir.path().join(LOCK_FILENAME)).unwrap();
        lock_file.lock().unwrap();
        assert!(!try_unregister_in(dir.path(), &s.session_id).unwrap());
        lock_file.unlock().unwrap();
        assert_eq!(1, list_in(dir.path()).unwrap().len());
    }

    #[test]
    fn locked_update_times_out_when_lock_stays_held() {
        let dir = TempDir::new().unwrap();
        register_in(dir.path(), make_session("s1", std::process::id())).unwrap();
        let holder = open_lock_file(&dir.path().join(LOCK_FILENAME)).unwrap();
        holder.lock().unwrap();

        let outcome = with_locked_state(dir.path(), Duration::from_millis(50), Vec::clear);

        assert_eq!(io::ErrorKind::TimedOut, outcome.unwrap_err().kind());
        assert_eq!(1, list_in(dir.path()).unwrap().len());
    }

    #[test]
    fn locked_update_retries_until_holder_releases() {
        let dir = TempDir::new().unwrap();
        let holder = open_lock_file(&dir.path().join(LOCK_FILENAME)).unwrap();
        holder.lock().unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            holder.unlock().unwrap();
        });

        with_locked_state(dir.path(), LOCK_ACQUIRE_TIMEOUT, |sessions| {
            sessions.push(make_session("s1", std::process::id()));
        })
        .unwrap();
        release.join().unwrap();
        assert_eq!(1, list_in(dir.path()).unwrap().len());
    }

    #[test]
    fn corrupt_file_recovers() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join(DATA_FILENAME), "garbage{{{").unwrap();
        assert!(list_in(dir.path()).unwrap().is_empty());
        register_in(dir.path(), make_session("s1", std::process::id())).unwrap();
        assert_eq!(1, list_in(dir.path()).unwrap().len());
    }

    #[test]
    fn try_unregister_waits_out_same_process_writer() {
        let dir = TempDir::new().unwrap();
        register_in(dir.path(), make_session("s", std::process::id())).unwrap();
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        std::thread::scope(|scope| {
            let root = dir.path();
            scope.spawn(move || {
                with_locked_state(root, LOCK_ACQUIRE_TIMEOUT, |_| {
                    held_tx.send(()).unwrap();
                    let _ = release_rx.recv_timeout(Duration::from_millis(50));
                })
                .unwrap();
            });
            held_rx.recv().unwrap();
            assert!(try_unregister_in(root, &acp::SessionId::new("s")).unwrap());
            drop(release_tx);
        });
        assert!(list_in(dir.path()).unwrap().is_empty());
    }

    #[test]
    fn legacy_entry_without_holder_reads_as_client() {
        let dir = TempDir::new().unwrap();
        let me = std::process::id();
        fs::write(
            dir.path().join(DATA_FILENAME),
            format!(
                r#"[{{"session_id":"s","pid":{me},"cwd":"/","opened_at":"2026-01-01T00:00:00Z"}}]"#
            ),
        )
        .unwrap();
        assert_eq!(
            vec![("s".to_string(), me, Holder::Client)],
            keys(dir.path())
        );
    }

    #[test]
    fn register_keys_by_id_pid_holder() {
        let dir = TempDir::new().unwrap();
        let me = std::process::id();
        let client = make_session("s", me);
        let agent = with_holder(make_session("s", me), Holder::Agent);
        for _ in 0..2 {
            register_in(dir.path(), client.clone()).unwrap();
            register_in(dir.path(), agent.clone()).unwrap();
        }
        assert_eq!(
            vec![
                ("s".to_string(), me, Holder::Client),
                ("s".to_string(), me, Holder::Agent),
            ],
            keys(dir.path())
        );
    }

    #[cfg(unix)]
    #[test]
    fn register_keeps_other_pid_same_id() {
        let dir = TempDir::new().unwrap();
        let me = std::process::id();
        let parent = std::os::unix::process::parent_id();
        register_in(dir.path(), make_session("s", parent)).unwrap();
        register_in(dir.path(), make_session("s", me)).unwrap();
        let mut pids: Vec<u32> = list_in(dir.path())
            .unwrap()
            .into_iter()
            .map(|s| s.pid)
            .collect();
        pids.sort_unstable();
        let mut want = vec![parent, me];
        want.sort_unstable();
        assert_eq!(want, pids);
    }

    #[cfg(unix)]
    #[test]
    fn client_unregister_leaves_agent_and_foreign_entries() {
        let dir = TempDir::new().unwrap();
        let me = std::process::id();
        let parent = std::os::unix::process::parent_id();
        register_in(
            dir.path(),
            with_holder(make_session("s", me), Holder::Agent),
        )
        .unwrap();
        register_in(dir.path(), make_session("s", parent)).unwrap();
        register_in(dir.path(), make_session("s", me)).unwrap();

        assert!(try_unregister_in(dir.path(), &acp::SessionId::new("s")).unwrap());

        let left = list_in(dir.path()).unwrap();
        assert_eq!(2, left.len());
        assert!(
            left.iter()
                .any(|s| s.pid == me && s.holder == Holder::Agent)
        );
        assert!(
            left.iter()
                .any(|s| s.pid == parent && s.holder == Holder::Client)
        );
    }
}
