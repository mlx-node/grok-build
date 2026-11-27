//! Regression: settings saves must serialize against config-init-flock writers, or the two
//! domains interleave read-modify-writes and the last atomic rename drops the other side's edit.

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn wait_for(deadline: Instant, mut check: impl FnMut() -> bool) -> bool {
    while Instant::now() < deadline {
        if check() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    false
}

#[cfg(target_os = "linux")]
fn open_fd_paths() -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir("/proc/self/fd") else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| std::fs::read_link(entry.path()).ok())
        .collect()
}

/// macOS has no `/proc`, and `/dev/fd` entries are not links: ask each fd for its path.
#[cfg(target_os = "macos")]
fn open_fd_paths() -> Vec<PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    let Ok(entries) = std::fs::read_dir("/dev/fd") else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().to_str()?.parse::<libc::c_int>().ok())
        .filter_map(|fd| {
            let mut buf = [0u8; libc::PATH_MAX as usize];
            // SAFETY: F_GETPATH writes at most PATH_MAX bytes, NUL included, into `buf`.
            if unsafe { libc::fcntl(fd, libc::F_GETPATH, buf.as_mut_ptr()) } != 0 {
                return None;
            }
            let len = buf.iter().position(|&b| b == 0)?;
            Some(PathBuf::from(std::ffi::OsStr::from_bytes(&buf[..len])))
        })
        .collect()
}

/// `acquire_init_lock` opens `.config-init.lock` before its 1s `try_lock` loop.
/// A second fd means the saver is already in that loop.
fn saver_has_opened_init_lock(lock_path: &Path) -> bool {
    let opens = open_fd_paths()
        .into_iter()
        .filter(|linked| {
            if linked == lock_path {
                return true;
            }
            match (dunce::canonicalize(linked), dunce::canonicalize(lock_path)) {
                (Ok(linked), Ok(lock_path)) => linked == lock_path,
                _ => false,
            }
        })
        .count();
    opens >= 2
}

#[test]
fn settings_save_serializes_against_init_flock_writer() {
    // One #[test] per binary: the env is process-global (same rule as
    // acp_harness::run_agent_test).
    let grok_home = tempfile::tempdir().expect("grok home");
    // SAFETY: no other threads are running yet.
    unsafe { std::env::set_var("GROK_HOME", grok_home.path()) };

    let config_path = grok_home.path().join("config.toml");
    std::fs::write(&config_path, "[cli]\n").unwrap();

    // Flock-holding writer mid read-modify-write (the shape of
    // session-start auto-enable's `add_enabled_plugin`).
    let flock =
        xai_grok_shell::util::config::acquire_init_lock(grok_home.path()).expect("init flock");
    let stale_read = std::fs::read_to_string(&config_path).unwrap();

    // Concurrent settings toggle on its own runtime thread.
    let saver = std::thread::spawn(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(xai_grok_shell::util::config::update_config(|cfg| {
                cfg.ui.simple_mode = Some(true);
            }))
    });

    // The saver fails closed after 1s of `try_lock` retries. A fixed sleep spends
    // that budget before this thread is scheduled to drop the flock.
    let lock_path = grok_home.path().join(".config-init.lock");
    assert!(
        wait_for(Instant::now() + Duration::from_secs(10), || {
            saver_has_opened_init_lock(&lock_path)
        }),
        "settings save never opened {}",
        lock_path.display()
    );
    let mut modified: toml::Value = toml::from_str(&stale_read).unwrap();
    modified.as_table_mut().unwrap().insert(
        "plugins".into(),
        toml::from_str("enabled = [\"demo\"]").unwrap(),
    );
    xai_grok_shell::util::config::atomic_write_string(
        &config_path,
        &toml::to_string_pretty(&modified).unwrap(),
    )
    .unwrap();
    drop(flock);

    saver
        .join()
        .expect("saver thread")
        .expect("settings save succeeds");

    let merged: toml::Value =
        toml::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
    assert_eq!(
        merged
            .get("plugins")
            .and_then(|p| p.get("enabled"))
            .and_then(|e| e.as_array())
            .map(|a| a.len()),
        Some(1),
        "flock writer's plugin enable must survive: {merged}"
    );
    assert_eq!(
        merged
            .get("ui")
            .and_then(|u| u.get("simple_mode"))
            .and_then(|v| v.as_bool()),
        Some(true),
        "settings save must survive the flock writer's rename: {merged}"
    );
}
