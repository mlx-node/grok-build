use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use xai_acp_lib::AcpAgentGatewaySender;

use super::support::*;
use super::*;

fn install_notification_client_hook(actor: &SessionActor) {
    let mut client_hooks = crate::extensions::hooks::ClientHooks::new();
    client_hooks.insert(
        xai_grok_hooks::event::HookEventName::Notification,
        vec![crate::extensions::hooks::ClientHookGroup {
            matcher: None,
            callback_ids: vec!["cb_permission".to_string()],
            timeout: None,
        }],
    );
    *actor.client_hooks.borrow_mut() = client_hooks;
}

async fn setup_actor(
    yolo: bool,
    park_until_hook: bool,
) -> (Arc<SessionActor>, Arc<AtomicUsize>, tokio::task::LocalSet) {
    setup_actor_with_pre_tool_use_hook(yolo, park_until_hook, None).await
}

async fn setup_actor_with_pre_tool_use_hook(
    yolo: bool,
    park_until_hook: bool,
    pre_tool_use_script: Option<&str>,
) -> (Arc<SessionActor>, Arc<AtomicUsize>, tokio::task::LocalSet) {
    let local = tokio::task::LocalSet::new();
    let (actor, hooks) = local
        .run_until(async {
            let (gateway_tx, gateway_rx) =
                tokio::sync::mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
            let hook_gateway = AcpAgentGatewaySender::new(gateway_tx.clone());
            let (persistence_tx, _persistence_rx) =
                tokio::sync::mpsc::unbounded_channel::<PersistenceMsg>();
            let mut actor = create_test_actor(0, 256_000, 85, gateway_tx, persistence_tx).await;
            *actor.agent.borrow_mut() = test_agent_with_tools(read_and_edit_toolset()).await;
            install_permission_manager(&mut actor, yolo, hook_gateway);
            install_notification_client_hook(&actor);
            if let Some(script) = pre_tool_use_script {
                install_pre_tool_use_hooks(
                    &mut actor,
                    vec![pre_tool_use_spec("test/ask", None, script)],
                );
            }
            let actor = Arc::new(actor);
            actor.wire_permission_prompt_notification();
            let hooks = Arc::new(AtomicUsize::new(0));
            spawn_gateway_loop_counting_prompt_hooks(gateway_rx, hooks.clone(), park_until_hook);
            (actor, hooks)
        })
        .await;
    (actor, hooks, local)
}

#[tokio::test(flavor = "current_thread")]
async fn auto_allowed_tool_does_not_fire_permission_prompt_notification() {
    let (actor, hooks, local) = setup_actor(/*yolo=*/ false, /*park_until_hook=*/ false).await;
    local
        .run_until(async {
            let result = prepare_call(&actor, read_file_call("call_safe")).await;
            assert!(
                result.is_ok(),
                "read must auto-allow; got {:?}",
                result.err()
            );
            drain_gateway_turns().await;
            assert_eq!(
                hooks.load(Ordering::SeqCst),
                0,
                "auto-allowed tool permission must not fire permission_prompt"
            );
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn real_user_prompt_fires_permission_prompt_notification() {
    let (actor, hooks, local) = setup_actor(/*yolo=*/ false, /*park_until_hook=*/ true).await;
    local
        .run_until(async {
            let result = prepare_call(&actor, search_replace_call("call_unsafe")).await;
            assert!(
                result.is_ok(),
                "prompted allow-once must prepare; got {:?}",
                result.err()
            );
            assert_eq!(
                hooks.load(Ordering::SeqCst),
                1,
                "a real user permission prompt must fire permission_prompt once"
            );
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn inherited_handle_second_wire_does_not_steal_parent_hook() {
    let (parent, hooks, local) = setup_actor(/*yolo=*/ false, /*park_until_hook=*/ true).await;
    local
        .run_until(async {
            let (child_gateway_tx, child_gateway_rx) =
                tokio::sync::mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
            let (persistence_tx, _persistence_rx) =
                tokio::sync::mpsc::unbounded_channel::<PersistenceMsg>();
            let mut child =
                create_test_actor(0, 256_000, 85, child_gateway_tx, persistence_tx).await;
            child.startup_hints.is_subagent = true;
            child.permissions = parent.permissions.clone();
            install_notification_client_hook(&child);
            let child = Arc::new(child);
            child.wire_permission_prompt_notification();
            let (steal_tx, _steal_rx) = tokio::sync::mpsc::unbounded_channel();
            child.permissions.set_user_prompt_notify(steal_tx);
            drop(child_gateway_rx);

            let result = prepare_call(&parent, search_replace_call("call_shared")).await;
            assert!(
                result.is_ok(),
                "prompted allow-once must prepare; got {:?}",
                result.err()
            );
            assert_eq!(
                hooks.load(Ordering::SeqCst),
                1,
                "parent hook must still fire once after a cloned handle tries to re-wire"
            );
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn yolo_tool_does_not_fire_permission_prompt_notification() {
    let (actor, hooks, local) = setup_actor(/*yolo=*/ true, /*park_until_hook=*/ false).await;
    local
        .run_until(async {
            let result = prepare_call(&actor, search_replace_call("call_yolo")).await;
            assert!(
                result.is_ok(),
                "yolo must auto-allow; got {:?}",
                result.err()
            );
            drain_gateway_turns().await;
            assert_eq!(
                hooks.load(Ordering::SeqCst),
                0,
                "yolo auto-approve must not fire permission_prompt"
            );
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn hook_ask_under_yolo_fires_permission_prompt_notification() {
    let (actor, hooks, local) = setup_actor_with_pre_tool_use_hook(
        /*yolo=*/ true,
        /*park_until_hook=*/ true,
        Some(r#"echo '{"hookSpecificOutput":{"permissionDecision":"ask"}}'"#),
    )
    .await;
    local
        .run_until(async {
            let result = prepare_call(&actor, search_replace_call("call_hook_ask")).await;
            assert!(
                result.is_ok(),
                "the forced prompt was approved, so the call must run; got {:?}",
                result.err()
            );
            assert_eq!(
                hooks.load(Ordering::SeqCst),
                1,
                "a hook ask forces a real prompt, so permission_prompt must fire once"
            );
        })
        .await;
}

fn run_terminal_command_tool() -> xai_grok_tools::registry::types::ToolConfig {
    xai_grok_tools::registry::types::ToolConfig {
        params: Some(
            serde_json::from_value(serde_json::json!({ "enabled_background": false })).unwrap(),
        ),
        ..xai_grok_tools::registry::types::ToolConfig::from(
            &xai_grok_tools::implementations::grok_build::BashTool,
        )
        .with_name("run_terminal_command")
    }
}

fn run_terminal_command_call(id: &str, command: &str) -> ToolCallResponse {
    ToolCallResponse {
        id: id.to_string(),
        kind: "function".to_string(),
        function: crate::sampling::types::ToolCallFunction::new(
            "run_terminal_command",
            serde_json::json!({ "command": command, "description": "list files" }).to_string(),
        ),
    }
}

#[derive(Default)]
struct CapturedAsk {
    session_id: Option<String>,
    meta: Option<acp::Meta>,
}

fn spawn_gateway_capturing_asks(
    gateway_rx: tokio::sync::mpsc::UnboundedReceiver<xai_acp_lib::AcpClientMessage>,
) -> Arc<std::sync::Mutex<Vec<CapturedAsk>>> {
    let asks: Arc<std::sync::Mutex<Vec<CapturedAsk>>> = Arc::default();
    let captured = asks.clone();
    let mut gateway_rx = gateway_rx;
    tokio::task::spawn_local(async move {
        while let Some(msg) = gateway_rx.recv().await {
            match msg {
                xai_acp_lib::AcpClientMessage::RequestPermission(args) => {
                    captured.lock().unwrap().push(CapturedAsk {
                        session_id: Some(args.request.session_id.0.to_string()),
                        meta: args.request.meta.clone(),
                    });
                    let _ = args
                        .response_tx
                        .send(Ok(acp::RequestPermissionResponse::new(
                            acp::RequestPermissionOutcome::Selected(
                                acp::SelectedPermissionOutcome::new(acp::PermissionOptionId::new(
                                    "allow-once",
                                )),
                            ),
                        )));
                }
                xai_acp_lib::AcpClientMessage::SessionNotification(args) => {
                    let _ = args.response_tx.send(Ok(()));
                }
                _ => {}
            }
        }
    });
    asks
}

#[tokio::test(flavor = "current_thread")]
async fn child_shell_call_asks_the_parent_through_the_parent_hook_with_its_requester() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, gateway_rx) =
                tokio::sync::mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
            let (persistence_tx, _persistence_rx) =
                tokio::sync::mpsc::unbounded_channel::<PersistenceMsg>();
            let mut parent =
                create_test_actor(0, 256_000, 85, gateway_tx.clone(), persistence_tx).await;
            parent.session_info.id = acp::SessionId::new("parent-session");
            install_permission_manager(
                &mut parent,
                /*yolo=*/ false,
                AcpAgentGatewaySender::new(gateway_tx),
            );
            install_pre_tool_use_hooks(
                &mut parent,
                vec![pre_tool_use_spec(
                    "test/ask",
                    Some("Bash"),
                    r#"echo '{"hookSpecificOutput":{"permissionDecision":"ask"}}'"#,
                )],
            );
            let asks = spawn_gateway_capturing_asks(gateway_rx);

            let (child_gateway_tx, child_gateway_rx) =
                tokio::sync::mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
            let (child_persistence_tx, _child_persistence_rx) =
                tokio::sync::mpsc::unbounded_channel::<PersistenceMsg>();
            let mut child =
                create_test_actor(0, 256_000, 85, child_gateway_tx, child_persistence_tx).await;
            let child_cwd = tempfile::tempdir().unwrap();
            child.session_info.id = acp::SessionId::new("child-session");
            child.session_info.cwd = child_cwd.path().to_string_lossy().into_owned();
            child.startup_hints.is_subagent = true;
            child.startup_hints.subagent_label = Some("probe-0".to_owned());
            child.startup_hints.workflow_run_id = Some("wf_t".to_owned());
            *child.agent.borrow_mut() =
                test_agent_with_tools(vec![run_terminal_command_tool()]).await;
            child.permissions = parent.permissions.clone();
            *child.hook_registry.borrow_mut() = parent.hook_registry.borrow().clone();
            child.hook_resolved_workspace_root = parent.hook_resolved_workspace_root.clone();
            drain_gateway(child_gateway_rx);

            let result =
                prepare_call(&child, run_terminal_command_call("call_child_ls", "ls")).await;
            assert!(
                result.is_ok(),
                "the hook ask was approved, so the child's call must prepare; got {:?}",
                result.err()
            );

            let asks = asks.lock().unwrap();
            assert_eq!(
                asks.len(),
                1,
                "the parent's ask hook must force one prompt for the child's safe-listed `ls`"
            );
            let ask = &asks[0];
            assert_eq!(
                ask.session_id.as_deref(),
                Some("parent-session"),
                "a child's ask goes out under the parent's sessionId"
            );
            let meta = ask.meta.as_ref().expect("a hook ask always carries meta");
            assert_eq!(
                meta.get(xai_grok_workspace::permission::HOOK_ASK_META_KEY),
                Some(&serde_json::json!({ "hookName": "test/ask" })),
                "the parent's guard hook must be the one that asked"
            );
            assert_eq!(
                meta.get(xai_grok_workspace::permission::REQUESTER_META_KEY),
                Some(&serde_json::json!({
                    "sessionId": "child-session",
                    "label": "probe-0",
                    "runId": "wf_t",
                    "cwd": child_cwd.path().to_string_lossy(),
                })),
                "the ask must name the child that made it"
            );
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn parent_own_ask_carries_no_requester() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, gateway_rx) =
                tokio::sync::mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
            let (persistence_tx, _persistence_rx) =
                tokio::sync::mpsc::unbounded_channel::<PersistenceMsg>();
            let mut parent =
                create_test_actor(0, 256_000, 85, gateway_tx.clone(), persistence_tx).await;
            *parent.agent.borrow_mut() =
                test_agent_with_tools(vec![run_terminal_command_tool()]).await;
            install_permission_manager(
                &mut parent,
                /*yolo=*/ false,
                AcpAgentGatewaySender::new(gateway_tx),
            );
            install_pre_tool_use_hooks(
                &mut parent,
                vec![pre_tool_use_spec(
                    "test/ask",
                    Some("Bash"),
                    r#"echo '{"hookSpecificOutput":{"permissionDecision":"ask"}}'"#,
                )],
            );
            let asks = spawn_gateway_capturing_asks(gateway_rx);

            let result =
                prepare_call(&parent, run_terminal_command_call("call_parent_ls", "ls")).await;
            assert!(
                result.is_ok(),
                "approved ask must prepare; got {:?}",
                result.err()
            );

            let asks = asks.lock().unwrap();
            assert_eq!(asks.len(), 1, "the ask hook must force one prompt");
            let meta = asks[0]
                .meta
                .as_ref()
                .expect("a hook ask always carries meta");
            assert!(
                meta.get(xai_grok_workspace::permission::HOOK_ASK_META_KEY)
                    .is_some(),
                "the parent's own ask keeps hookAsk"
            );
            assert!(
                meta.get(xai_grok_workspace::permission::REQUESTER_META_KEY)
                    .is_none(),
                "the parent's own ask carries no requester"
            );
        })
        .await;
}
