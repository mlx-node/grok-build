//! The tool-round check that gives a project agent folder a tool made its `.gitignore`.

use super::support::*;
use super::*;
use xai_grok_tools::implementations::opencode::OpenCodeWriteTool;
use xai_grok_tools::registry::types::ToolConfig;

fn write_call(path: &std::path::Path) -> crate::sampling::types::ToolCallResponse {
    crate::sampling::types::ToolCallResponse {
        id: "write-workflow".to_owned(),
        kind: "function".to_owned(),
        function: crate::sampling::types::ToolCallFunction::new(
            "write",
            serde_json::json!({"file_path": path, "content": "complete(\"ok\");\n"}).to_string(),
        ),
    }
}

/// One tool round in `project` whose only call writes `.mlx-agent/workflows/a.rhai`.
async fn write_a_workflow_in(project: &std::path::Path) {
    let (gateway_tx, _gateway_rx) =
        tokio::sync::mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
    let (persistence_tx, _persistence_rx) =
        tokio::sync::mpsc::unbounded_channel::<PersistenceMsg>();
    let mut actor = create_test_actor(0, 256_000, 85, gateway_tx, persistence_tx).await;
    *actor.agent.borrow_mut() =
        test_agent_with_tools(vec![ToolConfig::for_tool::<OpenCodeWriteTool>()]).await;
    actor.tool_context.cwd = xai_grok_paths::AbsPathBuf::new(project.to_path_buf()).unwrap();
    let toolset = actor.agent.borrow().tool_bridge().toolset();
    actor
        .workspace_ops
        .bind_local_session(
            &actor.session_id_string(),
            project.to_path_buf(),
            actor.tool_context.hunk_tracker_handle.clone(),
            toolset,
            None,
        )
        .expect("bind_local_session");
    actor.session_info.cwd = project.to_string_lossy().into_owned();
    actor
        .execute_tool_calls(
            vec![write_call(&project.join(".mlx-agent/workflows/a.rhai"))],
            None,
        )
        .await
        .expect("execute");
    assert!(
        project.join(".mlx-agent/workflows/a.rhai").is_file(),
        "the write tool ran"
    );
}

fn git_is_clean(repo: &git2::Repository) -> bool {
    let mut options = git2::StatusOptions::new();
    options.include_untracked(true).recurse_untracked_dirs(true);
    repo.statuses(Some(&mut options)).unwrap().is_empty()
}

#[tokio::test(flavor = "current_thread")]
async fn a_tool_made_agent_folder_gets_its_gitignore() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let tmp = tempfile::tempdir().unwrap();
            let project = dunce::canonicalize(tmp.path()).unwrap();
            let repo = git2::Repository::init(&project).unwrap();
            write_a_workflow_in(&project).await;
            assert_eq!(
                std::fs::read_to_string(project.join(".mlx-agent/.gitignore")).unwrap(),
                "*\n"
            );
            assert!(git_is_clean(&repo), "the model's folder stays out of git");
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn a_folder_that_existed_before_the_round_gets_no_gitignore() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let tmp = tempfile::tempdir().unwrap();
            let project = dunce::canonicalize(tmp.path()).unwrap();
            std::fs::create_dir(project.join(".mlx-agent")).unwrap();
            write_a_workflow_in(&project).await;
            assert!(
                !project.join(".mlx-agent/.gitignore").exists(),
                "a folder the user made is left as it is"
            );
        })
        .await;
}
