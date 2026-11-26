use super::*;

#[derive(Clone)]
struct RunShellChildTestRunner {
    contexts: std::sync::Arc<parking_lot::Mutex<std::collections::VecDeque<SubagentSpawnContext>>>,
    complete_first: std::sync::Arc<std::sync::atomic::AtomicBool>,
    gateway: GatewaySender,
}

impl RunShellChildTestRunner {
    fn new(
        contexts: impl IntoIterator<Item = SubagentSpawnContext>,
        complete_first: bool,
        gateway: GatewaySender,
    ) -> Self {
        Self {
            contexts: std::sync::Arc::new(parking_lot::Mutex::new(contexts.into_iter().collect())),
            complete_first: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(complete_first)),
            gateway,
        }
    }
}

impl xai_grok_tools::implementations::grok_build::task::coordinator::ChildRunner
    for RunShellChildTestRunner
{
    type Control = ShellChildRuntime;
    type RootControl =
        xai_grok_tools::implementations::grok_build::task::root_control::NoRootControl;
    type CompletionData = ShellCompletionData;
    type RunFuture = xai_grok_tools::implementations::grok_build::task::coordinator::LocalBoxFuture<
        ChildRunOutput<ShellCompletionData>,
    >;
    type ValidateFuture =
        xai_grok_tools::implementations::grok_build::task::coordinator::LocalBoxFuture<
            SubagentValidateTypeOutcome,
        >;
    type DescribeFuture =
        xai_grok_tools::implementations::grok_build::task::coordinator::LocalBoxFuture<
            SubagentDescribeOutcome,
        >;

    fn run(
        &self,
        run: xai_grok_tools::implementations::grok_build::task::coordinator::ChildRunRequest<
            Self::Control,
        >,
    ) -> Self::RunFuture {
        if self
            .complete_first
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            return Box::pin(std::future::ready(ChildRunOutput {
                result: SubagentResult {
                    success: true,
                    output: std::sync::Arc::from("prior output"),
                    subagent_id: run.request.id.clone(),
                    child_session_id: run.request.id,
                    tool_calls: 2,
                    turns: 1,
                    duration_ms: 7,
                    ..Default::default()
                },
                completion_data: ShellCompletionData::default(),
                snapshot_ref: Some("refs/grok/subagents/prior".to_owned()),
            }));
        }
        let ctx = self.contexts.lock().pop_front().expect("run context");
        let gateway = self.gateway.clone();
        let completion_data = ShellCompletionData::from_context(&ctx, run.attempt_id.clone(), None);
        Box::pin(async move { run_shell_child(run, ctx, completion_data, gateway, None).await })
    }

    fn validate_type(
        &self,
        _subagent_type: String,
        _parent_session_id: String,
    ) -> Self::ValidateFuture {
        Box::pin(std::future::ready(SubagentValidateTypeOutcome::Ok))
    }

    fn describe_type(
        &self,
        _subagent_type: String,
        _harness_agent_type: Option<String>,
        _parent_session_id: String,
    ) -> Self::DescribeFuture {
        Box::pin(std::future::ready(SubagentDescribeOutcome::Unavailable))
    }

    fn supports_wake(&self) -> bool {
        true
    }

    fn on_completed(
        &self,
        completion: ChildCompletion<Self::CompletionData>,
        terminal_published: Box<dyn FnOnce() + Send>,
    ) {
        present_child_completion(completion, &self.gateway, false);
        terminal_published();
    }
}

fn prior_wake_meta(id: &str, model_id: &str) -> SubagentMeta {
    SubagentMeta {
        subagent_id: id.to_owned(),
        attempt_id: Some("at1.prior".to_owned()),
        parent_session_id: "setup-parent".to_owned(),
        child_session_id: id.to_owned(),
        subagent_type: "general-purpose".to_owned(),
        description: "prior description".to_owned(),
        prompt: "prior prompt".to_owned(),
        status: "completed".to_owned(),
        started_at: chrono::Utc::now(),
        completed_at: Some(chrono::Utc::now()),
        duration_ms: Some(7),
        tool_calls: Some(2),
        turns: Some(1),
        error: None,
        effective_context_source: Some("new".to_owned()),
        context_normalized: false,
        fork_copy_error: None,
        persona: None,
        resumed_from: None,
        child_cwd: Some("/tmp".to_owned()),
        worktree_path: None,
        snapshot_ref: Some("refs/grok/subagents/prior".to_owned()),
        effective_model_id: Some(model_id.to_owned()),
    }
}

async fn assert_wake_setup_failure_preserves_prior_durable_state(
    build_failure: impl FnOnce(std::path::PathBuf) -> SubagentSetupFailure,
) {
    use crate::session::storage::StorageAdapter;
    use xai_grok_sampling_types::conversation::ConversationItem;
    use xai_grok_tools::implementations::grok_build::task::backend::{
        ChannelBackend, SubagentBackend,
    };
    use xai_grok_tools::implementations::grok_build::task::coordinator::{
        CoordinatorConfig, SubagentCoordinator,
    };

    let mut ctx = ctx_with_toggle(HashMap::new());
    ctx.sampling_config.model = "test".to_owned();
    ctx.model_id = acp::ModelId::new("test");
    ctx.parent_agent_name = Some("general-purpose".to_owned());
    let meta_dir = tempfile::tempdir().expect("meta dir");
    let id = uuid::Uuid::now_v7().to_string();
    let prior_meta = prior_wake_meta(&id, "test");
    assert!(write_subagent_meta(meta_dir.path(), &prior_meta));
    assert!(write_subagent_output(meta_dir.path(), "prior output"));
    ctx.setup_failure = Some(build_failure(meta_dir.path().to_path_buf()));
    ctx.parent_session_id = "setup-parent".into();
    ctx.parent_cwd = std::path::PathBuf::from("/tmp");
    let child_info = SessionInfo {
        id: acp::SessionId::new(id.clone()),
        cwd: "/tmp".to_owned(),
    };
    let storage = crate::session::storage::jsonl::JsonlStorageAdapter::with_root(
        crate::util::grok_home::grok_home(),
    );
    storage
        .init_session(&child_info, acp::ModelId::new("test"))
        .await
        .expect("session");
    storage
        .append_chat_message(&child_info, &ConversationItem::system("prior system"))
        .await
        .expect("system message");
    storage
        .append_chat_message(&child_info, &ConversationItem::assistant("prior work"))
        .await
        .expect("assistant message");
    storage
        .update_current_model(&child_info, &acp::ModelId::new("prior-model"))
        .await
        .expect("prior model");
    ctx.model_id = acp::ModelId::new("rejected-wake-model");
    ctx.sampling_config.model = "rejected-wake-model".to_owned();
    let child_session_dir = crate::session::persistence::session_dir(&child_info);
    let prior_transcript =
        std::fs::read(child_session_dir.join("chat_history.jsonl")).expect("prior transcript");
    let prior_summary_bytes =
        std::fs::read(child_session_dir.join("summary.json")).expect("prior summary");
    let prior_summary: crate::session::persistence::Summary =
        serde_json::from_slice(&prior_summary_bytes).expect("parse prior summary");
    let (parent_cmd_tx, mut parent_cmd_rx) = mpsc::unbounded_channel();
    ctx.parent_cmd_tx = Some(parent_cmd_tx);
    let (gateway, mut gateway_rx) = test_gateway_with_receiver();
    let (command_tx, command_rx) = SubagentCoordinator::<RunShellChildTestRunner>::channel();
    let coordinator = tokio::task::spawn_local(
        SubagentCoordinator::from_channel(
            command_rx,
            RunShellChildTestRunner::new([ctx], true, gateway),
            CoordinatorConfig::default(),
        )
        .run(),
    );
    let backend = ChannelBackend::for_coordinator_session(command_tx, "setup-parent");
    assert!(
        backend
            .spawn(auto_wake_test_request(&id), None)
            .await
            .expect("prior spawn")
            .success
    );
    while parent_cmd_rx.try_recv().is_ok() {}
    while gateway_rx.try_recv().is_ok() {}
    assert_eq!(
        backend
            .send_active_message(
                ActiveAgentMessageRequest::try_new(&id, "continue").expect("wake request")
            )
            .await,
        ActiveAgentMessageOutcome::NotActiveOrFinalizing
    );
    let restored = backend
        .query(&id, false, None)
        .await
        .expect("restored prior snapshot");
    assert!(matches!(
        restored.status,
        SubagentSnapshotStatus::Completed { ref output, .. } if output == "prior output"
    ));
    drop(backend);
    coordinator.await.expect("coordinator");

    let restored_meta: SubagentMeta = serde_json::from_str(
        &std::fs::read_to_string(meta_dir.path().join("meta.json")).expect("meta"),
    )
    .expect("metadata");
    assert_eq!(prior_meta.attempt_id, restored_meta.attempt_id);
    assert_eq!(prior_meta.status, restored_meta.status);
    assert_eq!(prior_meta.completed_at, restored_meta.completed_at);
    assert_eq!(prior_meta.snapshot_ref, restored_meta.snapshot_ref);
    assert_eq!(prior_meta.error, restored_meta.error);
    assert_eq!(
        read_subagent_output(meta_dir.path()).as_deref(),
        Some("prior output")
    );
    assert_eq!(
        std::fs::read(child_session_dir.join("chat_history.jsonl"))
            .expect("transcript after rejected wake"),
        prior_transcript,
    );
    let restored_summary_bytes =
        std::fs::read(child_session_dir.join("summary.json")).expect("restored summary");
    let restored_summary: crate::session::persistence::Summary =
        serde_json::from_slice(&restored_summary_bytes).expect("parse restored summary");
    assert_eq!(restored_summary_bytes, prior_summary_bytes);
    assert_eq!(
        restored_summary.current_model_id,
        prior_summary.current_model_id
    );
    assert_eq!(
        restored_summary.next_trace_turn,
        prior_summary.next_trace_turn
    );
    assert_eq!(restored_summary.attempt_id, prior_summary.attempt_id);
    assert!(parent_cmd_rx.try_recv().is_err());
    assert!(gateway_rx.try_recv().is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn wake_setup_failures_preserve_prior_durable_state_and_lifecycle() {
    tokio::task::LocalSet::new()
        .run_until(async {
            assert_wake_setup_failure_preserves_prior_durable_state(|meta_dir| {
                SubagentSetupFailure::SamplingClient { meta_dir }
            })
            .await;
            let blocker = tempfile::NamedTempFile::new().expect("blocker");
            assert_wake_setup_failure_preserves_prior_durable_state(|meta_dir| {
                SubagentSetupFailure::Persistence {
                    meta_dir,
                    persistence_dir: blocker.path().to_path_buf(),
                }
            })
            .await;
        })
        .await;
}

fn configure_completion_harness(
    ctx: &mut SubagentSpawnContext,
    server: &xai_grok_test_support::MockInferenceServer,
    harness: RunShellChildHarnessConfig,
) {
    ctx.run_shell_child_harness = Some(harness);
    ctx.parent_session_id = "setup-parent".into();
    ctx.sampling_config.base_url = server.url();
    ctx.sampling_config.model = "test-model".into();
    ctx.sampling_config.api_backend = crate::sampling::ApiBackend::Responses;
    ctx.model_id = acp::ModelId::new("test-model");
}

async fn acknowledge_parent_usage(mut parent_cmd_rx: mpsc::UnboundedReceiver<SessionCommand>) {
    while let Some(command) = parent_cmd_rx.recv().await {
        if let SessionCommand::RecordSubagentUsage { respond_to, .. } = command {
            let _ = respond_to.send(());
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn unpublished_wake_completion_preserves_prior_durable_state_and_worktree() {
    xai_test_utils::require_git!();
    use crate::session::storage::StorageAdapter;
    use xai_grok_tools::implementations::grok_build::task::backend::{
        ChannelBackend, SubagentBackend,
    };
    use xai_grok_tools::implementations::grok_build::task::coordinator::{
        CoordinatorConfig, SubagentCoordinator,
    };
    use xai_test_utils::git::{run_git, seed_repo_with_remote};

    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let temp = tempfile::TempDir::new().expect("tempdir");
            let (repo, _remote) = seed_repo_with_remote(temp.path());
            let id = uuid::Uuid::now_v7().to_string();
            let worktree = temp.path().join("wake-worktree");
            xai_fast_worktree::WorktreeBuilder::new(&repo, &worktree)
                .create()
                .expect("worktree");
            let meta_dir = temp.path().join("meta");
            let mut prior_meta = prior_wake_meta(&id, "test-model");
            prior_meta.child_cwd = Some(worktree.to_string_lossy().into_owned());
            prior_meta.worktree_path = Some(worktree.to_string_lossy().into_owned());
            prior_meta.snapshot_ref = Some("refs/grok/subagents/prior".to_owned());
            let prior_head = run_git(&repo, &["rev-parse", "HEAD"]);
            let prior_ref = prior_meta.snapshot_ref.as_deref().expect("snapshot ref");
            run_git(&repo, &["update-ref", prior_ref, &prior_head]);
            assert!(write_subagent_meta(&meta_dir, &prior_meta));
            assert!(write_subagent_output(&meta_dir, "prior output"));

            let child_info = SessionInfo {
                id: acp::SessionId::new(id.clone()),
                cwd: worktree.to_string_lossy().into_owned(),
            };
            let storage = crate::session::storage::jsonl::JsonlStorageAdapter::with_root(
                crate::util::grok_home::grok_home(),
            );
            storage
                .init_session(&child_info, acp::ModelId::new("test-model"))
                .await
                .expect("session");
            storage
                .append_chat_message(
                    &child_info,
                    &xai_grok_sampling_types::conversation::ConversationItem::system(
                        "prior system",
                    ),
                )
                .await
                .expect("system message");
            storage
                .append_chat_message(
                    &child_info,
                    &xai_grok_sampling_types::conversation::ConversationItem::assistant(
                        "prior work",
                    ),
                )
                .await
                .expect("assistant message");

            let server = xai_grok_test_support::MockInferenceServer::start()
                .await
                .expect("mock server");
            let mut ctx = ctx_with_toggle(HashMap::new());
            let completion_harness = RunShellChildHarnessConfig::new(
                meta_dir.clone(),
                InitialAttemptBehavior::CompleteBeforeAdmission,
            );
            configure_completion_harness(&mut ctx, &server, completion_harness.clone());
            ctx.parent_cwd = repo.clone();
            let mut config = crate::agent::config::Config::default();
            config.feature_values.insert(
                crate::agent::config::Feature::SubagentWorktreeSnapshot,
                true,
            );
            ctx.agent_config = Some(config);
            let (parent_cmd_tx, parent_cmd_rx) = mpsc::unbounded_channel();
            ctx.parent_cmd_tx = Some(parent_cmd_tx);
            let usage_ack = tokio::task::spawn_local(acknowledge_parent_usage(parent_cmd_rx));
            let (gateway, mut gateway_rx) = test_gateway_with_receiver();
            let (command_tx, command_rx) =
                SubagentCoordinator::<RunShellChildTestRunner>::channel();
            let coordinator = tokio::task::spawn_local(
                SubagentCoordinator::from_channel(
                    command_rx,
                    RunShellChildTestRunner::new([ctx], true, gateway),
                    CoordinatorConfig::default(),
                )
                .run(),
            );
            let backend = ChannelBackend::for_coordinator_session(command_tx, "setup-parent");
            assert!(
                backend
                    .spawn(auto_wake_test_request(&id), None)
                    .await
                    .expect("prior spawn")
                    .success
            );
            while gateway_rx.try_recv().is_ok() {}

            assert_eq!(
                backend
                    .send_active_message(
                        ActiveAgentMessageRequest::try_new(&id, "continue").expect("wake request")
                    )
                    .await,
                ActiveAgentMessageOutcome::NotActiveOrFinalizing
            );
            let restored = backend
                .query(&id, false, None)
                .await
                .expect("restored prior snapshot");
            assert!(matches!(
                restored.status,
                SubagentSnapshotStatus::Completed { ref output, .. }
                    if output == "prior output"
            ));
            drop(backend);
            coordinator.await.expect("coordinator");
            usage_ack.abort();

            let restored_meta: SubagentMeta = serde_json::from_str(
                &std::fs::read_to_string(meta_dir.join("meta.json")).expect("meta"),
            )
            .expect("metadata");
            assert_eq!(restored_meta.attempt_id, prior_meta.attempt_id);
            assert_eq!(restored_meta.status, prior_meta.status);
            assert_eq!(restored_meta.completed_at, prior_meta.completed_at);
            assert_eq!(restored_meta.snapshot_ref, prior_meta.snapshot_ref);
            assert_eq!(run_git(&repo, &["rev-parse", prior_ref]), prior_head);
            assert_eq!(
                read_subagent_output(&meta_dir).as_deref(),
                Some("prior output")
            );
            assert!(worktree.is_dir());
            assert!(
                std::iter::from_fn(|| gateway_rx.try_recv().ok()).all(|message| {
                    !matches!(
                        message,
                        xai_acp_lib::AcpClientMessage::ExtNotification(args)
                            if args.request.params.get().contains("subagent_spawned")
                    )
                }),
                "unpublished wake must not emit SubagentSpawned"
            );
        })
        .await;
}

/// Serial: the worktree base reads `$GROK_HOME` on every call, and `#[serial]` tests set it.
#[tokio::test(flavor = "current_thread")]
#[serial_test::serial]
async fn ordinary_spawn_with_failed_metadata_write_persists_output_and_disposes_worktree() {
    xai_test_utils::require_git!();
    use xai_grok_tools::implementations::grok_build::task::backend::{
        ChannelBackend, SubagentBackend,
    };
    use xai_grok_tools::implementations::grok_build::task::coordinator::{
        CoordinatorConfig, SubagentCoordinator,
    };
    use xai_test_utils::git::seed_repo_with_remote;

    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let temp = tempfile::TempDir::new().expect("tempdir");
            let (repo, _remote) = seed_repo_with_remote(temp.path());
            let meta_dir = temp.path().join("meta");
            let server = xai_grok_test_support::MockInferenceServer::start()
                .await
                .expect("mock server");
            server.set_response("ordinary output");
            let id = uuid::Uuid::now_v7().to_string();
            let mut ctx = ctx_with_toggle(HashMap::new());
            configure_completion_harness(
                &mut ctx,
                &server,
                RunShellChildHarnessConfig::new(meta_dir.clone(), InitialAttemptBehavior::Normal),
            );
            ctx.parent_cwd = repo;
            ctx.fail_start_metadata_write = true;
            let mut config = crate::agent::config::Config::default();
            config.feature_values.insert(
                crate::agent::config::Feature::SubagentWorktreeSnapshot,
                true,
            );
            ctx.agent_config = Some(config);
            let (parent_cmd_tx, parent_cmd_rx) = mpsc::unbounded_channel();
            ctx.parent_cmd_tx = Some(parent_cmd_tx);
            let usage_ack = tokio::task::spawn_local(acknowledge_parent_usage(parent_cmd_rx));
            let (gateway, _gateway_rx) = test_gateway_with_receiver();
            let (command_tx, command_rx) =
                SubagentCoordinator::<RunShellChildTestRunner>::channel();
            let coordinator = tokio::task::spawn_local(
                SubagentCoordinator::from_channel(
                    command_rx,
                    RunShellChildTestRunner::new([ctx], false, gateway),
                    CoordinatorConfig::default(),
                )
                .run(),
            );
            let backend = ChannelBackend::for_coordinator_session(command_tx, "setup-parent");
            let mut request = auto_wake_test_request(&id);
            request.runtime_overrides.isolation =
                Some(xai_tool_types::SubagentIsolationMode::Worktree);
            let result = backend.spawn(request, None).await.expect("ordinary spawn");
            assert!(result.success);
            let output = result.output.to_string();
            assert_eq!(
                read_subagent_output(&meta_dir).as_deref(),
                Some(output.as_str())
            );
            let persisted: SubagentMeta = serde_json::from_str(
                &std::fs::read_to_string(meta_dir.join("meta.json")).expect("completion meta"),
            )
            .expect("metadata");
            assert_eq!(persisted.status, "completed");
            let worktree = persisted.worktree_path.as_deref().expect("worktree path");
            assert!(!std::path::Path::new(worktree).exists());
            assert!(persisted.snapshot_ref.is_some());

            drop(backend);
            coordinator.await.expect("coordinator");
            usage_ack.abort();
        })
        .await;
}

/// Serial: the worktree base reads `$GROK_HOME` on every call, and `#[serial]` tests set it.
#[tokio::test(flavor = "current_thread")]
#[serial_test::serial]
async fn ordinary_spawn_disposes_worktree_when_only_remote_settings_enable_snapshot() {
    xai_test_utils::require_git!();
    use xai_grok_tools::implementations::grok_build::task::backend::{
        ChannelBackend, SubagentBackend,
    };
    use xai_grok_tools::implementations::grok_build::task::coordinator::{
        CoordinatorConfig, SubagentCoordinator,
    };
    use xai_test_utils::git::seed_repo_with_remote;

    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let temp = tempfile::TempDir::new().expect("tempdir");
            let (repo, _remote) = seed_repo_with_remote(temp.path());
            let meta_dir = temp.path().join("meta");
            let server = xai_grok_test_support::MockInferenceServer::start()
                .await
                .expect("mock server");
            server.set_response("ordinary output");
            let id = uuid::Uuid::now_v7().to_string();
            let mut ctx = ctx_with_toggle(HashMap::new());
            configure_completion_harness(
                &mut ctx,
                &server,
                RunShellChildHarnessConfig::new(meta_dir.clone(), InitialAttemptBehavior::Normal),
            );
            ctx.parent_cwd = repo;
            ctx.remote_settings = Some(crate::util::config::RemoteSettings {
                subagent_worktree_snapshot_enabled: Some(true),
                ..Default::default()
            });
            let (parent_cmd_tx, parent_cmd_rx) = mpsc::unbounded_channel();
            ctx.parent_cmd_tx = Some(parent_cmd_tx);
            let usage_ack = tokio::task::spawn_local(acknowledge_parent_usage(parent_cmd_rx));
            let (gateway, _gateway_rx) = test_gateway_with_receiver();
            let (command_tx, command_rx) =
                SubagentCoordinator::<RunShellChildTestRunner>::channel();
            let coordinator = tokio::task::spawn_local(
                SubagentCoordinator::from_channel(
                    command_rx,
                    RunShellChildTestRunner::new([ctx], false, gateway),
                    CoordinatorConfig::default(),
                )
                .run(),
            );
            let backend = ChannelBackend::for_coordinator_session(command_tx, "setup-parent");
            let mut request = auto_wake_test_request(&id);
            request.runtime_overrides.isolation =
                Some(xai_tool_types::SubagentIsolationMode::Worktree);
            let result = backend.spawn(request, None).await.expect("ordinary spawn");
            assert!(result.success);
            let persisted: SubagentMeta = serde_json::from_str(
                &std::fs::read_to_string(meta_dir.join("meta.json")).expect("completion meta"),
            )
            .expect("metadata");
            assert_eq!(persisted.status, "completed");
            let worktree = persisted.worktree_path.as_deref().expect("worktree path");
            assert!(
                !std::path::Path::new(worktree).exists(),
                "remote subagent_worktree_snapshot_enabled must reach the post-spawn dispose gate"
            );
            assert!(persisted.snapshot_ref.is_some());

            drop(backend);
            coordinator.await.expect("coordinator");
            usage_ack.abort();
        })
        .await;
}

/// The child's actor no longer binds itself; `run_shell_child` binds it once the actor is up,
/// ahead of the first turn, and teardown releases it.
#[tokio::test(flavor = "current_thread")]
async fn ordinary_spawn_binds_the_child_workspace_session_before_its_first_turn() {
    use xai_grok_tools::implementations::grok_build::task::backend::{
        ChannelBackend, SubagentBackend,
    };
    use xai_grok_tools::implementations::grok_build::task::coordinator::{
        CoordinatorConfig, SubagentCoordinator,
    };

    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let temp = tempfile::TempDir::new().expect("tempdir");
            let meta_dir = temp.path().join("meta");
            let server = xai_grok_test_support::MockInferenceServer::start()
                .await
                .expect("mock server");
            server.set_response("bound output");
            // Park the child's first turn at the model so its binding is observable mid-turn.
            server.hold_agent_completions();
            let id = uuid::Uuid::now_v7().to_string();
            let mut ctx = ctx_with_toggle(HashMap::new());
            configure_completion_harness(
                &mut ctx,
                &server,
                RunShellChildHarnessConfig::new(meta_dir.clone(), InitialAttemptBehavior::Normal),
            );
            ctx.parent_cwd = temp.path().to_path_buf();
            let workspace_ops = ctx.workspace_ops.clone();
            let (parent_cmd_tx, parent_cmd_rx) = mpsc::unbounded_channel();
            ctx.parent_cmd_tx = Some(parent_cmd_tx);
            let usage_ack = tokio::task::spawn_local(acknowledge_parent_usage(parent_cmd_rx));
            let (gateway, _gateway_rx) = test_gateway_with_receiver();
            let (command_tx, command_rx) =
                SubagentCoordinator::<RunShellChildTestRunner>::channel();
            let coordinator = tokio::task::spawn_local(
                SubagentCoordinator::from_channel(
                    command_rx,
                    RunShellChildTestRunner::new([ctx], false, gateway),
                    CoordinatorConfig::default(),
                )
                .run(),
            );
            let backend = ChannelBackend::for_coordinator_session(command_tx, "setup-parent");
            let request = auto_wake_test_request(&id);
            let spawned =
                tokio::task::spawn_local(async move { backend.spawn(request, None).await });
            let first_turn_at_model = async {
                while server.request_count() == 0 {
                    tokio::task::yield_now().await;
                }
            };
            tokio::time::timeout(std::time::Duration::from_secs(30), first_turn_at_model)
                .await
                .expect("the child's first turn reaches the model");
            let workspace = workspace_ops
                .workspace_handle()
                .expect("local workspace ops");
            assert!(
                workspace.session(&id).is_some(),
                "the child's toolset is bound before its first turn dispatches"
            );
            server.release_agent_completions();
            let result = spawned.await.expect("spawn task").expect("ordinary spawn");
            assert!(result.success);
            assert!(
                workspace.session(&id).is_none(),
                "teardown releases the child's binding"
            );

            coordinator.await.expect("coordinator");
            usage_ack.abort();
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn unacked_wake_start_and_abort_fail_closed_without_parking_runner() {
    use xai_grok_tools::implementations::grok_build::task::backend::{
        ChannelBackend, SubagentBackend,
    };
    use xai_grok_tools::implementations::grok_build::task::coordinator::{
        CoordinatorConfig, SubagentCoordinator,
    };

    let local = tokio::task::LocalSet::new();
    local
        .run_until(Box::pin(async {
            let meta_dir = tempfile::tempdir().expect("meta dir");
            let server = xai_grok_test_support::MockInferenceServer::start()
                .await
                .expect("mock server");
            server.set_response("completed output");
            let id = uuid::Uuid::now_v7().to_string();
            let mut ordinary_ctx = ctx_with_toggle(HashMap::new());
            configure_completion_harness(
                &mut ordinary_ctx,
                &server,
                RunShellChildHarnessConfig::new(
                    meta_dir.path().to_path_buf(),
                    InitialAttemptBehavior::Normal,
                ),
            );
            let (parent_cmd_tx, parent_cmd_rx) = mpsc::unbounded_channel();
            ordinary_ctx.parent_cmd_tx = Some(parent_cmd_tx.clone());
            let usage_ack = tokio::task::spawn_local(acknowledge_parent_usage(parent_cmd_rx));
            let mut wake_ctx = ctx_with_toggle(HashMap::new());
            configure_completion_harness(
                &mut wake_ctx,
                &server,
                RunShellChildHarnessConfig::new(
                    meta_dir.path().to_path_buf(),
                    InitialAttemptBehavior::Normal,
                )
                .hold_wake_flush_acks(),
            );
            wake_ctx.parent_cmd_tx = Some(parent_cmd_tx);
            let child_cwd = ordinary_ctx.parent_cwd.to_string_lossy().into_owned();
            let (gateway, mut gateway_rx) = test_gateway_with_receiver();
            let (command_tx, command_rx) =
                SubagentCoordinator::<RunShellChildTestRunner>::channel();
            let coordinator = tokio::task::spawn_local(
                SubagentCoordinator::from_channel(
                    command_rx,
                    RunShellChildTestRunner::new([ordinary_ctx, wake_ctx], false, gateway),
                    CoordinatorConfig::default(),
                )
                .run(),
            );
            let backend = ChannelBackend::for_coordinator_session(command_tx, "setup-parent");
            let ordinary = backend
                .spawn(auto_wake_test_request(&id), None)
                .await
                .expect("ordinary spawn");
            assert!(
                ordinary.success,
                "ordinary spawn failed: {:?}",
                ordinary.error
            );
            let child_info = SessionInfo {
                id: acp::SessionId::new(id.clone()),
                cwd: child_cwd,
            };
            let child_session_dir = crate::session::persistence::session_dir(&child_info);
            let prior_transcript = std::fs::read(child_session_dir.join("chat_history.jsonl"))
                .expect("prior transcript");
            let prior_summary: crate::session::persistence::Summary = serde_json::from_slice(
                &std::fs::read(child_session_dir.join("summary.json")).expect("prior summary"),
            )
            .expect("parse prior summary");
            let prior_meta =
                std::fs::read(meta_dir.path().join("meta.json")).expect("prior metadata");
            while gateway_rx.try_recv().is_ok() {}

            let wake = backend
                .send_active_message(
                    ActiveAgentMessageRequest::try_new(&id, "continue").expect("wake request"),
                )
                .await;
            assert_eq!(wake, ActiveAgentMessageOutcome::NotActiveOrFinalizing);
            let restored = backend
                .query(&id, true, Some(60_000))
                .await
                .expect("restored prior snapshot");
            assert!(matches!(
                restored.status,
                SubagentSnapshotStatus::Completed { ref output, .. }
                    if output == "completed output"
            ));
            assert_eq!(
                std::fs::read(child_session_dir.join("chat_history.jsonl"))
                    .expect("transcript after timeout"),
                prior_transcript
            );
            assert_eq!(
                std::fs::read(meta_dir.path().join("meta.json")).expect("metadata after timeout"),
                prior_meta
            );
            let restored_summary: crate::session::persistence::Summary = serde_json::from_slice(
                &std::fs::read(child_session_dir.join("summary.json"))
                    .expect("summary after timeout"),
            )
            .expect("parse summary after timeout");
            assert_eq!(restored_summary.attempt_id, prior_summary.attempt_id);
            assert_eq!(
                restored_summary.next_trace_turn,
                prior_summary.next_trace_turn
            );
            assert!(
                std::iter::from_fn(|| gateway_rx.try_recv().ok()).all(|message| {
                    !matches!(
                        message,
                        xai_acp_lib::AcpClientMessage::ExtNotification(args)
                            if args.request.params.get().contains("subagent_spawned")
                    )
                }),
                "timed-out wake must not publish a new lifecycle"
            );

            drop(backend);
            coordinator.await.expect("coordinator");
            usage_ack.abort();
        }))
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn rejected_deferred_start_restores_prior_without_publication() {
    use xai_grok_tools::implementations::grok_build::task::backend::{
        ChannelBackend, SubagentBackend,
    };
    use xai_grok_tools::implementations::grok_build::task::coordinator::{
        CoordinatorConfig, SubagentCoordinator,
    };

    tokio::task::LocalSet::new()
        .run_until(Box::pin(async {
            let meta_dir = tempfile::tempdir().expect("meta dir");
            let server = xai_grok_test_support::MockInferenceServer::start()
                .await
                .expect("mock server");
            server.set_response("completed output");
            let id = uuid::Uuid::now_v7().to_string();
            let mut ordinary_ctx = ctx_with_toggle(HashMap::new());
            configure_completion_harness(
                &mut ordinary_ctx,
                &server,
                RunShellChildHarnessConfig::new(
                    meta_dir.path().to_path_buf(),
                    InitialAttemptBehavior::Normal,
                ),
            );
            let (parent_cmd_tx, parent_cmd_rx) = mpsc::unbounded_channel();
            ordinary_ctx.parent_cmd_tx = Some(parent_cmd_tx.clone());
            let usage_ack = tokio::task::spawn_local(acknowledge_parent_usage(parent_cmd_rx));
            let child_cwd = ordinary_ctx.parent_cwd.to_string_lossy().into_owned();
            let mut wake_ctx = ctx_with_toggle(HashMap::new());
            configure_completion_harness(
                &mut wake_ctx,
                &server,
                RunShellChildHarnessConfig::new(
                    meta_dir.path().to_path_buf(),
                    InitialAttemptBehavior::Normal,
                )
                .reject_deferred_start_commit(),
            );
            wake_ctx.parent_cmd_tx = Some(parent_cmd_tx);
            let (gateway, mut gateway_rx) = test_gateway_with_receiver();
            let (command_tx, command_rx) =
                SubagentCoordinator::<RunShellChildTestRunner>::channel();
            let coordinator = tokio::task::spawn_local(
                SubagentCoordinator::from_channel(
                    command_rx,
                    RunShellChildTestRunner::new([ordinary_ctx, wake_ctx], false, gateway),
                    CoordinatorConfig::default(),
                )
                .run(),
            );
            let backend = ChannelBackend::for_coordinator_session(command_tx, "setup-parent");
            let ordinary = backend
                .spawn(auto_wake_test_request(&id), None)
                .await
                .expect("ordinary spawn");
            assert!(
                ordinary.success,
                "ordinary spawn failed: {:?}",
                ordinary.error
            );
            let child_info = SessionInfo {
                id: acp::SessionId::new(id.clone()),
                cwd: child_cwd,
            };
            let child_session_dir = crate::session::persistence::session_dir(&child_info);
            let prior_summary =
                std::fs::read(child_session_dir.join("summary.json")).expect("prior summary");
            let prior_meta =
                std::fs::read(meta_dir.path().join("meta.json")).expect("prior metadata");
            assert!(write_subagent_output(
                meta_dir.path(),
                "prior output sentinel"
            ));
            while gateway_rx.try_recv().is_ok() {}

            assert_eq!(
                backend
                    .send_active_message(
                        ActiveAgentMessageRequest::try_new(&id, "continue").expect("wake request")
                    )
                    .await,
                ActiveAgentMessageOutcome::NotActiveOrFinalizing
            );
            let restored = backend
                .query(&id, true, Some(60_000))
                .await
                .expect("restored prior completion");
            assert!(
                matches!(restored.status, SubagentSnapshotStatus::Completed { .. }),
                "expected restored completion, got {:?}",
                restored.status
            );
            assert_eq!(
                std::fs::read(meta_dir.path().join("meta.json")).expect("restored metadata"),
                prior_meta
            );
            assert_eq!(
                std::fs::read(child_session_dir.join("summary.json")).expect("restored summary"),
                prior_summary
            );
            assert_eq!(
                read_subagent_output(meta_dir.path()).as_deref(),
                Some("prior output sentinel")
            );
            assert!(
                std::iter::from_fn(|| gateway_rx.try_recv().ok()).all(|message| {
                    !matches!(
                        message,
                        xai_acp_lib::AcpClientMessage::ExtNotification(args)
                            if args.request.params.get().contains("subagent_spawned")
                    )
                }),
                "rejected settle must not publish SubagentSpawned"
            );

            drop(backend);
            coordinator.await.expect("coordinator");
            usage_ack.abort();
        }))
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn started_wake_with_failed_metadata_write_preserves_prior_durable_artifacts() {
    use xai_grok_tools::implementations::grok_build::task::backend::{
        ChannelBackend, SubagentBackend,
    };
    use xai_grok_tools::implementations::grok_build::task::coordinator::{
        CoordinatorConfig, SubagentCoordinator,
    };

    let local = tokio::task::LocalSet::new();
    local
        .run_until(Box::pin(async {
            let meta_dir = tempfile::tempdir().expect("meta dir");
            let server = xai_grok_test_support::MockInferenceServer::start()
                .await
                .expect("mock server");
            server.set_response("completed output");
            let id = uuid::Uuid::now_v7().to_string();
            let mut ordinary_ctx = ctx_with_toggle(HashMap::new());
            configure_completion_harness(
                &mut ordinary_ctx,
                &server,
                RunShellChildHarnessConfig::new(
                    meta_dir.path().to_path_buf(),
                    InitialAttemptBehavior::Normal,
                ),
            );
            let (parent_cmd_tx, parent_cmd_rx) = mpsc::unbounded_channel();
            ordinary_ctx.parent_cmd_tx = Some(parent_cmd_tx.clone());
            let usage_ack = tokio::task::spawn_local(acknowledge_parent_usage(parent_cmd_rx));
            let mut wake_ctx = ctx_with_toggle(HashMap::new());
            configure_completion_harness(
                &mut wake_ctx,
                &server,
                RunShellChildHarnessConfig::new(
                    meta_dir.path().to_path_buf(),
                    InitialAttemptBehavior::Normal,
                ),
            );
            wake_ctx.parent_cmd_tx = Some(parent_cmd_tx);
            wake_ctx.fail_start_metadata_write = true;
            let child_cwd = ordinary_ctx.parent_cwd.to_string_lossy().into_owned();
            let (gateway, mut gateway_rx) = test_gateway_with_receiver();
            let (command_tx, command_rx) =
                SubagentCoordinator::<RunShellChildTestRunner>::channel();
            let coordinator = tokio::task::spawn_local(
                SubagentCoordinator::from_channel(
                    command_rx,
                    RunShellChildTestRunner::new([ordinary_ctx, wake_ctx], false, gateway),
                    CoordinatorConfig::default(),
                )
                .run(),
            );
            let backend = ChannelBackend::for_coordinator_session(command_tx, "setup-parent");
            let ordinary = backend
                .spawn(auto_wake_test_request(&id), None)
                .await
                .expect("ordinary spawn");
            assert!(
                ordinary.success,
                "ordinary spawn failed: {:?}",
                ordinary.error
            );
            let child_info = SessionInfo {
                id: acp::SessionId::new(id.clone()),
                cwd: child_cwd,
            };
            let child_session_dir = crate::session::persistence::session_dir(&child_info);
            let prior_summary: crate::session::persistence::Summary = serde_json::from_slice(
                &std::fs::read(child_session_dir.join("summary.json")).expect("prior summary"),
            )
            .expect("parse prior summary");
            let prior_meta =
                std::fs::read(meta_dir.path().join("meta.json")).expect("prior metadata");
            assert!(write_subagent_output(
                meta_dir.path(),
                "prior output sentinel"
            ));
            while gateway_rx.try_recv().is_ok() {}

            assert!(matches!(
                backend
                    .send_active_message(
                        ActiveAgentMessageRequest::try_new(&id, "continue").expect("wake request")
                    )
                    .await,
                ActiveAgentMessageOutcome::Accepted { .. }
            ));
            let wake = backend
                .query(&id, true, Some(60_000))
                .await
                .expect("wake completion");
            assert!(matches!(
                wake.status,
                SubagentSnapshotStatus::Completed { .. }
            ));

            assert_eq!(
                std::fs::read(meta_dir.path().join("meta.json")).expect("restored metadata"),
                prior_meta,
            );
            assert_eq!(
                read_subagent_output(meta_dir.path()).as_deref(),
                Some("prior output sentinel"),
            );
            let restored_summary: crate::session::persistence::Summary = serde_json::from_slice(
                &std::fs::read(child_session_dir.join("summary.json")).expect("restored summary"),
            )
            .expect("parse restored summary");
            assert_eq!(
                restored_summary.next_trace_turn,
                prior_summary.next_trace_turn.saturating_add(1)
            );
            assert_ne!(restored_summary.attempt_id, prior_summary.attempt_id);
            assert!(
                restored_summary
                    .attempt_id
                    .as_deref()
                    .and_then(xai_message_delivery_core::AttemptId::parse)
                    .is_some()
            );

            drop(backend);
            coordinator.await.expect("coordinator");
            usage_ack.abort();
        }))
        .await;
}

/// Runs one real child spawn to completion against the mock model; returns its first agent request body.
/// The registry is both the process snapshot (a task child's) and the parent session's (a workflow child's).
async fn first_child_request_body(
    owner: SubagentOwner,
    parent_cwd: std::path::PathBuf,
    registry: Option<std::sync::Arc<xai_grok_agent::plugins::PluginRegistry>>,
) -> serde_json::Value {
    use xai_grok_tools::implementations::grok_build::task::backend::{
        ChannelBackend, SubagentBackend,
    };
    use xai_grok_tools::implementations::grok_build::task::coordinator::{
        CoordinatorConfig, SubagentCoordinator,
    };
    const TASK: &str = "K1 probe task: list the folder.";
    let meta = tempfile::TempDir::new().expect("meta dir");
    let server = xai_grok_test_support::MockInferenceServer::start()
        .await
        .expect("mock server");
    server.set_response("probe output");
    let mut ctx = ctx_with_toggle(HashMap::new());
    configure_completion_harness(
        &mut ctx,
        &server,
        RunShellChildHarnessConfig::new(meta.path().to_path_buf(), InitialAttemptBehavior::Normal),
    );
    ctx.parent_cwd = parent_cwd;
    ctx.plugin_registry = registry.clone();
    ctx.parent_plugin_registry = registry;
    let (parent_cmd_tx, parent_cmd_rx) = mpsc::unbounded_channel();
    ctx.parent_cmd_tx = Some(parent_cmd_tx);
    let usage_ack = tokio::task::spawn_local(acknowledge_parent_usage(parent_cmd_rx));
    let (gateway, _gateway_rx) = test_gateway_with_receiver();
    let (command_tx, command_rx) = SubagentCoordinator::<RunShellChildTestRunner>::channel();
    let coordinator = tokio::task::spawn_local(
        SubagentCoordinator::from_channel(
            command_rx,
            RunShellChildTestRunner::new([ctx], false, gateway),
            CoordinatorConfig::default(),
        )
        .run(),
    );
    let backend = ChannelBackend::for_coordinator_session(command_tx, "setup-parent");
    let mut request = auto_wake_test_request(&uuid::Uuid::now_v7().to_string());
    request.prompt = TASK.to_owned();
    request.owner = owner;
    let result = backend.spawn(request, None).await.expect("child spawn");
    assert!(result.success, "child run: {:?}", result.output);
    drop(backend);
    coordinator.await.expect("coordinator");
    usage_ack.abort();
    server
        .request_bodies()
        .into_iter()
        .filter(|body| {
            body.pointer("/tool_choice/name")
                .and_then(|name| name.as_str())
                != Some("session_title")
        })
        .find(|body| body.to_string().contains(TASK))
        .expect("the child's first agent request reached the model")
}

/// Mutation that turns it red: drop the `start_workflow_child` call in `spawn_session_actor`
/// (the workflow child's first request then has no skill path).
#[tokio::test(flavor = "current_thread")]
async fn workflow_child_first_request_lists_parent_skills() {
    use xai_grok_agent::plugins::SharedPluginRegistryHandle;
    use xai_grok_agent::plugins::discovery::DiscoveryConfig;
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let workdir = tempfile::TempDir::new().expect("workdir");
            let plugin_dir = workdir.path().join("probe-plugin");
            let skill_dir = plugin_dir.join("skills").join("probe-web");
            std::fs::create_dir_all(&skill_dir).expect("plugin skill dir");
            std::fs::write(plugin_dir.join("plugin.json"), r#"{"name":"probe-plugin"}"#)
                .expect("plugin manifest");
            std::fs::write(
                skill_dir.join("SKILL.md"),
                "---\nname: probe-web\ndescription: Searches the web.\n---\n\nRun the search.\n",
            )
            .expect("plugin skill");
            let registry = SharedPluginRegistryHandle::new(None, Vec::new()).build_for_cwd(
                workdir.path(),
                &DiscoveryConfig::default(),
                &[plugin_dir],
                false,
            );
            assert!(registry.is_some(), "session plugin dir must build a registry");
            let skill_path = "probe-plugin/skills/probe-web/SKILL.md";
            let input_with_skill = |body: &serde_json::Value| -> Vec<(String, usize)> {
                body["input"]
                    .as_array()
                    .expect("responses input")
                    .iter()
                    .filter(|item| item.to_string().contains(skill_path))
                    .map(|item| {
                        (
                            item["role"].as_str().unwrap_or_default().to_owned(),
                            item.to_string().matches(skill_path).count(),
                        )
                    })
                    .collect()
            };

            let workflow_body = first_child_request_body(
                SubagentOwner::workflow("wf_k1"),
                workdir.path().to_path_buf(),
                registry.clone(),
            )
            .await;
            assert_eq!(
                input_with_skill(&workflow_body),
                vec![("user".to_owned(), 1)],
                "a workflow child's first request lists the parent's skill once, in a user reminder: {workflow_body}"
            );

            let task_body = first_child_request_body(
                SubagentOwner::Task,
                workdir.path().to_path_buf(),
                registry,
            )
            .await;
            assert!(
                input_with_skill(&task_body).is_empty(),
                "a task child keeps its old start: {task_body}"
            );
        })
        .await;
}

/// Mutations that turn it red: pass no label from `run_shell_child` (the title becomes the prompt's words),
/// and keep the model title call for children.
#[tokio::test(flavor = "current_thread")]
async fn child_title_is_its_description_with_no_title_call() {
    use xai_grok_tools::implementations::grok_build::task::backend::{
        ChannelBackend, SubagentBackend,
    };
    use xai_grok_tools::implementations::grok_build::task::coordinator::{
        CoordinatorConfig, SubagentCoordinator,
    };
    tokio::task::LocalSet::new()
        .run_until(async {
            let workdir = tempfile::TempDir::new().expect("workdir");
            let meta = tempfile::TempDir::new().expect("meta dir");
            let server = xai_grok_test_support::MockInferenceServer::start()
                .await
                .expect("mock server");
            server.set_response("label probe output");
            let mut ctx = ctx_with_toggle(HashMap::new());
            configure_completion_harness(
                &mut ctx,
                &server,
                RunShellChildHarnessConfig::new(
                    meta.path().to_path_buf(),
                    InitialAttemptBehavior::Normal,
                ),
            );
            ctx.parent_cwd = workdir.path().to_path_buf();
            let (parent_cmd_tx, parent_cmd_rx) = mpsc::unbounded_channel();
            ctx.parent_cmd_tx = Some(parent_cmd_tx);
            let usage_ack = tokio::task::spawn_local(acknowledge_parent_usage(parent_cmd_rx));
            let (gateway, _gateway_rx) = test_gateway_with_receiver();
            let (command_tx, command_rx) =
                SubagentCoordinator::<RunShellChildTestRunner>::channel();
            let coordinator = tokio::task::spawn_local(
                SubagentCoordinator::from_channel(
                    command_rx,
                    RunShellChildTestRunner::new([ctx], false, gateway),
                    CoordinatorConfig::default(),
                )
                .run(),
            );
            let backend = ChannelBackend::for_coordinator_session(command_tx, "setup-parent");
            let id = uuid::Uuid::now_v7().to_string();
            let mut request = auto_wake_test_request(&id);
            request.prompt = "Walk the folder and name each file you find in it".to_owned();
            request.description = "track:alpha".to_owned();
            let result = backend.spawn(request, None).await.expect("child spawn");
            assert!(result.success, "child run: {:?}", result.error);
            drop(backend);
            coordinator.await.expect("coordinator");
            usage_ack.abort();

            let child_dir = crate::session::persistence::session_dir(&SessionInfo {
                id: acp::SessionId::new(id),
                cwd: workdir.path().to_string_lossy().into_owned(),
            });
            let summary: crate::session::persistence::Summary = serde_json::from_slice(
                &std::fs::read(child_dir.join("summary.json")).expect("child summary"),
            )
            .expect("parse child summary");
            assert_eq!(summary.display_title(), "track:alpha");
            let title_requests = server
                .request_bodies()
                .into_iter()
                .filter(|body| body.to_string().contains("session_title"))
                .count();
            assert_eq!(title_requests, 0);
        })
        .await;
}

/// A reads a file; B resumes A and reads it again; C resumes A and its first model call is refused.
/// Mutations that turn it red: report the lifetime counts (B: 2 tool calls, 2 turns), snapshot after the first prompt
/// (B: 0 turns), snapshot at the end (B: 0 and 0), and gate the token report on the attempt's own tool calls (C: 0 tokens).
#[tokio::test(flavor = "current_thread")]
async fn resumed_child_reports_only_its_own_counts() {
    use crate::session::storage::StorageAdapter;
    use xai_grok_test_support::{Conversation, MockToolCall, StatusFailure, Tool};
    use xai_grok_tools::implementations::grok_build::task::backend::{
        ChannelBackend, SubagentBackend,
    };
    use xai_grok_tools::implementations::grok_build::task::coordinator::{
        CoordinatorConfig, SubagentCoordinator,
    };
    tokio::task::LocalSet::new()
        .run_until(async {
            let workdir = tempfile::TempDir::new().expect("workdir");
            std::fs::write(workdir.path().join("AGENTS.md"), "Keep answers short.\n")
                .expect("project instructions");
            let notes = workdir.path().join("notes.txt");
            std::fs::write(&notes, "probe notes\n").expect("notes");
            let read_notes = || {
                MockToolCall::new(
                    Tool::Read,
                    serde_json::json!({ "target_file": notes.to_string_lossy() }),
                )
            };
            let server = xai_grok_test_support::MockInferenceServer::start()
                .await
                .expect("mock server");
            server.set_conversations([
                Conversation::nth(1).calls([read_notes()]).reply("A done"),
                Conversation::nth(2).calls([read_notes()]).reply("B done"),
                Conversation::nth(3)
                    .refuse(StatusFailure::new(400))
                    .reply("C never answers"),
            ]);
            let (parent_cmd_tx, parent_cmd_rx) = mpsc::unbounded_channel();
            let usage_ack = tokio::task::spawn_local(acknowledge_parent_usage(parent_cmd_rx));
            let metas: Vec<tempfile::TempDir> = (0..3)
                .map(|_| tempfile::TempDir::new().expect("meta dir"))
                .collect();
            let contexts: Vec<SubagentSpawnContext> = metas
                .iter()
                .map(|meta| {
                    let mut ctx = ctx_with_toggle(HashMap::new());
                    configure_completion_harness(
                        &mut ctx,
                        &server,
                        RunShellChildHarnessConfig::new(
                            meta.path().to_path_buf(),
                            InitialAttemptBehavior::Normal,
                        ),
                    );
                    ctx.parent_cwd = workdir.path().to_path_buf();
                    ctx.parent_cmd_tx = Some(parent_cmd_tx.clone());
                    ctx
                })
                .collect();
            drop(parent_cmd_tx);
            let (gateway, mut gateway_rx) = test_gateway_with_receiver();
            let (command_tx, command_rx) =
                SubagentCoordinator::<RunShellChildTestRunner>::channel();
            let coordinator = tokio::task::spawn_local(
                SubagentCoordinator::from_channel(
                    command_rx,
                    RunShellChildTestRunner::new(contexts, false, gateway),
                    CoordinatorConfig::default(),
                )
                .run(),
            );
            let backend = ChannelBackend::for_coordinator_session(command_tx, "setup-parent");
            let request = |id: &str, prompt: &str, resume_from: Option<&str>| {
                let mut request = auto_wake_test_request(id);
                request.prompt = prompt.to_owned();
                request.resume_from = resume_from.map(str::to_owned);
                request.run_in_background = false;
                request
            };
            let [a_id, b_id, c_id] = [(); 3].map(|()| uuid::Uuid::now_v7().to_string());

            let a = backend
                .spawn(request(&a_id, "Read the notes file, then say done.", None), None)
                .await
                .expect("A spawn");
            assert!(a.success, "A: {:?}", a.error);
            assert_eq!((a.turns, a.tool_calls), (1, 1));
            let a_history = crate::session::storage::jsonl::JsonlStorageAdapter::with_root(
                crate::util::grok_home::grok_home(),
            )
            .load_session_without_updates(&SessionInfo {
                id: acp::SessionId::new(a_id.clone()),
                cwd: workdir.path().to_string_lossy().into_owned(),
            })
            .await
            .expect("A history")
            .chat_history;
            let a_users = a_history
                .iter()
                .filter(|item| {
                    matches!(
                        item,
                        xai_grok_sampling_types::conversation::ConversationItem::User(_)
                    )
                })
                .count();
            assert!(
                a_users >= 2,
                "the resume seed counts A's user items minus one, so B must start at turn 1 or later: {a_users}"
            );

            let b = backend
                .spawn(
                    request(&b_id, "Read the notes again, then say done.", Some(&a_id)),
                    None,
                )
                .await
                .expect("B spawn");
            assert!(b.success, "B: {:?}", b.error);
            assert_eq!((b.turns, b.tool_calls), (1, 1));

            let c = backend
                .spawn(request(&c_id, "Say done.", Some(&a_id)), None)
                .await
                .expect("C spawn");
            assert!(!c.success, "C must fail on the refused call");
            assert_eq!((c.turns, c.tool_calls), (1, 0));
            drop(backend);
            coordinator.await.expect("coordinator");
            usage_ack.abort();
            let unserved: Vec<String> = server
                .finish_scripts()
                .iter()
                .map(ToString::to_string)
                .collect();
            assert_eq!(unserved, ["conversation 3: 0 of 1 entries served"]);

            let mut finished = HashMap::new();
            while let Ok(message) = gateway_rx.try_recv() {
                let xai_acp_lib::AcpClientMessage::ExtNotification(args) = message else {
                    continue;
                };
                let Ok(notification) =
                    serde_json::from_str::<SessionNotification>(args.request.params.get())
                else {
                    continue;
                };
                if let SessionUpdate::SubagentFinished {
                    subagent_id,
                    turns,
                    tool_calls,
                    tokens_used,
                    ..
                } = notification.update
                {
                    finished.insert(subagent_id, (turns, tool_calls, tokens_used));
                }
            }
            assert_eq!((finished[&a_id].0, finished[&a_id].1), (1, 1));
            assert_eq!((finished[&b_id].0, finished[&b_id].1), (1, 1));
            assert_eq!((finished[&c_id].0, finished[&c_id].1), (1, 0));
            assert!(
                finished[&c_id].2 > 0,
                "a failed resume still reports its inherited context: {:?}",
                finished[&c_id]
            );
        })
        .await;
}
