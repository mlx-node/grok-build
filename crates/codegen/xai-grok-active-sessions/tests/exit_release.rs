//! A process that exits while its session actors still hold leases leaves no agent entry: the
//! exit hook removes them even though the worker never saw the leases drop.

use std::process::Command;
use std::time::{Duration, Instant};

use tempfile::TempDir;
use xai_grok_active_sessions::{Holder, hold, list_in, wait_synced};

const CHILD_ENV: &str = "ACTIVE_SESSIONS_EXIT_CHILD";

#[test]
fn exit_with_held_lease_removes_agent_entry() {
    if std::env::var_os(CHILD_ENV).is_some() {
        let lease = hold(
            agent_client_protocol::SessionId::new("held-at-exit"),
            "/work".into(),
        );
        assert!(wait_synced(Duration::from_secs(5)));
        std::mem::forget(lease);
        std::process::exit(0);
    }

    let home = TempDir::new().unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "exit_with_held_lease_removes_agent_entry",
            "--nocapture",
        ])
        .env(CHILD_ENV, "1")
        .env("GROK_HOME", home.path())
        .status()
        .unwrap();
    assert!(status.success());

    let deadline = Instant::now() + Duration::from_secs(5);
    while !home.path().join("active_sessions.json").exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        home.path().join("active_sessions.json").exists(),
        "the child registered its lease"
    );
    let left: Vec<_> = list_in(home.path())
        .unwrap()
        .into_iter()
        .filter(|s| s.holder == Holder::Agent)
        .collect();
    assert!(left.is_empty(), "agent entries left after exit: {left:?}");
}
