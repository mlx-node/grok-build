//! `x.ai/workflows/stop`: a client stops one workflow run of a session.

use agent_client_protocol as acp;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use xai_grok_tools::implementations::grok_build::workflow::{
    WorkflowControl, WorkflowLaunchAck, WorkflowLaunchEnvelope, WorkflowLaunchRequest,
    WorkflowSource, WorkflowToolInput,
};

use crate::extensions::agent_runtime::AgentRuntime;

use super::ExtResult;

pub(crate) const WORKFLOWS_STOP_METHOD: &str = "x.ai/workflows/stop";

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkflowsStopRequest {
    session_id: String,
    run_id: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "outcome", rename_all = "lowercase")]
enum WorkflowsStopResponse {
    #[serde(rename_all = "camelCase")]
    Stopped {
        run_id: String,
        name: String,
    },
    Rejected {
        code: &'static str,
        detail: String,
    },
}

pub(crate) async fn handle(agent: &dyn AgentRuntime, args: &acp::ExtRequest) -> ExtResult {
    let req = parse_stop_request(args)?;
    let session_id = acp::SessionId::new(req.session_id.as_str());
    let Some(handle) = agent.session_handle_waiting_for_load(&session_id).await else {
        return Err(
            acp::Error::invalid_params().data(format!("unknown session id: {}", req.session_id))
        );
    };
    stop_run(&handle.workflow_launch_tx, req.run_id).await
}

fn parse_stop_request(args: &acp::ExtRequest) -> Result<WorkflowsStopRequest, acp::Error> {
    let req: WorkflowsStopRequest = super::parse_params(args)?;
    if req.session_id.trim().is_empty() {
        return Err(acp::Error::invalid_params().data("sessionId must not be blank"));
    }
    if req.run_id.trim().is_empty() {
        return Err(acp::Error::invalid_params().data("runId must not be blank"));
    }
    Ok(req)
}

async fn stop_run(tx: &mpsc::UnboundedSender<WorkflowLaunchEnvelope>, run_id: String) -> ExtResult {
    let (ack_tx, ack_rx) = oneshot::channel();
    let input = WorkflowToolInput {
        source: WorkflowSource::Stop { run_id },
        agent_budget: None,
        args: None,
        validate_only: false,
    };
    if tx.send((WorkflowLaunchRequest { input }, ack_tx)).is_err() {
        return Err(acp::Error::internal_error().data("workflow request service is gone"));
    }
    let ack = ack_rx.await.map_err(|_| {
        acp::Error::internal_error().data("workflow request service dropped the stop request")
    })?;
    let response = match ack {
        WorkflowLaunchAck::Controlled {
            run_id,
            name,
            control: WorkflowControl::Stop,
        } => WorkflowsStopResponse::Stopped { run_id, name },
        WorkflowLaunchAck::Rejected { code, detail } => {
            WorkflowsStopResponse::Rejected { code, detail }
        }
        other => {
            return Err(acp::Error::internal_error()
                .data(format!("unexpected workflow stop answer: {other:?}")));
        }
    };
    super::to_raw_response(&response)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::session::workflow::manager::WorkflowManager;
    use crate::session::workflow::request_service::{RequestServiceOptions, spawn_request_service};
    use crate::session::workflow::tracker::{WorkflowRunStatus, WorkflowTracker};

    fn service(
        manager: Arc<tokio::sync::Mutex<WorkflowManager>>,
        enabled: bool,
    ) -> mpsc::UnboundedSender<WorkflowLaunchEnvelope> {
        let (tx, rx) = mpsc::unbounded_channel();
        spawn_request_service(
            rx,
            RequestServiceOptions {
                manager,
                cwd: std::env::temp_dir(),
                session_dir: std::env::temp_dir(),
                enabled,
            },
        );
        tx
    }

    fn start_run(tracker: &parking_lot::Mutex<WorkflowTracker>, run_id: &str) {
        tracker.lock().start_run(
            run_id.to_owned(),
            "deep-research".into(),
            "obj".into(),
            Vec::new(),
            None,
            None,
        );
    }

    async fn answer(
        tx: &mpsc::UnboundedSender<WorkflowLaunchEnvelope>,
        run_id: &str,
    ) -> serde_json::Value {
        let response = stop_run(tx, run_id.to_owned())
            .await
            .expect("stop answers with a JSON-RPC success");
        serde_json::from_str(response.0.get()).expect("stop answer is JSON")
    }

    fn ext_request(params: serde_json::Value) -> acp::ExtRequest {
        acp::ExtRequest::new(
            WORKFLOWS_STOP_METHOD,
            serde_json::value::to_raw_value(&params).unwrap().into(),
        )
    }

    #[tokio::test]
    async fn stop_of_an_active_run_cancels_it_and_queues_no_wake() {
        let (manager, tracker) = WorkflowManager::test_bundle();
        start_run(&tracker, "wf_live");
        let (_done_tx, done_rx) = oneshot::channel();
        manager
            .lock()
            .await
            .test_insert_active_run("wf_live".to_owned(), done_rx);
        let tx = service(manager, true);

        assert_eq!(
            answer(&tx, "wf_live").await,
            serde_json::json!({ "outcome": "stopped", "runId": "wf_live", "name": "deep-research" })
        );
        let state = tracker.lock().get("wf_live").unwrap();
        assert_eq!(state.status, WorkflowRunStatus::Cancelled);
        assert!(
            !tracker
                .lock()
                .is_unreported_completion("wf_live", state.revision),
            "a client stop must not queue a completion wake turn"
        );
    }

    #[tokio::test]
    async fn stop_of_an_unknown_run_is_rejected() {
        let (manager, _tracker) = WorkflowManager::test_bundle();
        let tx = service(manager, true);

        let answer = answer(&tx, "wf_nope").await;
        assert_eq!(answer["outcome"], "rejected");
        assert_eq!(answer["code"], "workflow_control_unknown_run");
        assert!(answer["detail"].as_str().is_some_and(|d| !d.is_empty()));
        assert_eq!(answer.as_object().unwrap().len(), 3, "{answer}");
    }

    #[tokio::test]
    async fn stop_of_a_finished_run_is_not_applicable() {
        let (manager, tracker) = WorkflowManager::test_bundle();
        start_run(&tracker, "wf_done");
        tracker.lock().apply_outcome(
            "wf_done",
            &xai_workflow::WorkflowOutcome::Completed {
                result: serde_json::json!("report"),
            },
        );
        let tx = service(manager, true);

        let answer = answer(&tx, "wf_done").await;
        assert_eq!(answer["outcome"], "rejected");
        assert_eq!(answer["code"], "workflow_control_not_applicable");
        assert_eq!(
            tracker.lock().get("wf_done").unwrap().status,
            WorkflowRunStatus::Complete
        );
    }

    #[tokio::test]
    async fn stop_while_workflows_are_disabled_is_rejected() {
        let (manager, _tracker) = WorkflowManager::test_bundle();
        let tx = service(manager, false);

        let answer = answer(&tx, "wf_any").await;
        assert_eq!(answer["outcome"], "rejected");
        assert_eq!(answer["code"], "workflows_disabled");
    }

    #[test]
    fn missing_or_blank_fields_are_invalid_params() {
        let invalid = acp::Error::invalid_params().code;
        for params in [
            serde_json::json!({ "sessionId": "s" }),
            serde_json::json!({ "runId": "wf_x" }),
            serde_json::json!({ "sessionId": " ", "runId": "wf_x" }),
            serde_json::json!({ "sessionId": "s", "runId": "" }),
        ] {
            let err =
                parse_stop_request(&ext_request(params.clone())).expect_err("rejected params");
            assert_eq!(err.code, invalid, "{params}");
        }
        let req = parse_stop_request(&ext_request(
            serde_json::json!({ "sessionId": "s", "runId": "wf_x" }),
        ))
        .unwrap();
        assert_eq!(
            (req.session_id.as_str(), req.run_id.as_str()),
            ("s", "wf_x")
        );
    }
}
