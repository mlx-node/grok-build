//! mlx-node fork: a session's Messages requests carry `extra_body.reasoning_effort` only when the model's
//! `/v1/models` row is an mlx-node engine's and lists its levels, and only on the base URL that listed it.
//! Any other Messages endpoint gets the request without the field.

mod acp_harness;

use std::rc::Rc;

use acp_harness::{
    AutoApproveClient, RPC_TIMEOUT, allow_once, connect_and_auth, connect_and_auth_with_config,
    new_session, prompt_turn, run_agent_test_with_models, run_agent_test_with_two_servers,
};
use agent_client_protocol::{self as acp, Agent as _};
use serde_json::{Value, json};
use xai_grok_test_support::{EnvGuard, MockInferenceServer, MockModelEntry, ScriptedResponse};

fn test_row(owned_by: &str) -> MockModelEntry {
    MockModelEntry::new("test-model")
        .with_api_backend("messages")
        .with_owned_by(owned_by)
        .with_reasoning_efforts(vec![
            json!({ "value": "low", "label": "Low" }),
            json!({ "value": "high", "label": "High", "default": true }),
        ])
}

fn messages(server: &MockInferenceServer) -> Vec<Value> {
    server
        .requests()
        .into_iter()
        .filter(|r| r.path == "/v1/messages")
        .filter_map(|r| r.body)
        .collect()
}

/// The Messages bodies `server` received for `test-model`.
fn turn_messages(server: &MockInferenceServer) -> Vec<Value> {
    messages(server)
        .into_iter()
        .filter(|body| body.get("model") == Some(&json!("test-model")))
        .collect()
}

/// The Messages bodies of a session that names the model in `session/new` (a model switch after spawn), then of
/// one that takes the default model (the spawn config).
fn messages_bodies_for_row(owned_by: &str) -> [Vec<Value>; 2] {
    let bodies = Rc::new(std::cell::RefCell::new([Vec::new(), Vec::new()]));
    let out = Rc::clone(&bodies);
    run_agent_test_with_models(vec![test_row(owned_by)], |cwd, server| async move {
        let (conn, _init) = connect_and_auth(AutoApproveClient, "test-client").await;
        let named = new_session(&conn, &cwd).await;
        prompt_turn(&conn, &named, "hi").await;
        let after_named = messages(&server);
        let default = tokio::time::timeout(
            RPC_TIMEOUT,
            conn.new_session(acp::NewSessionRequest::new(cwd.clone())),
        )
        .await
        .expect("session/new timed out")
        .expect("session/new failed")
        .session_id;
        prompt_turn(&conn, &default, "hi").await;
        let all = messages(&server);
        let mut out = out.borrow_mut();
        out[1] = all[after_named.len()..].to_vec();
        out[0] = after_named;
    });
    bodies.take()
}

/// `output_config` and `extra_body` of a body: the request carries the whole system prompt, too long for a failure message.
fn wire_effort(body: &Value) -> (Option<&Value>, Option<&Value>) {
    (body.get("output_config"), body.get("extra_body"))
}

fn with_effort(bodies: &[Value]) -> Vec<&Value> {
    let with: Vec<&Value> = bodies
        .iter()
        .filter(|body| body.pointer("/output_config/effort").is_some())
        .collect();
    assert!(
        with.iter()
            .any(|body| body.pointer("/output_config/effort") == Some(&json!("high"))),
        "the turn sends the row's default level: {:?}",
        bodies.iter().map(wire_effort).collect::<Vec<_>>()
    );
    with
}

fn assert_no_extra_body(bodies: &[Value]) {
    with_effort(bodies);
    for body in bodies {
        assert!(body.get("extra_body").is_none(), "{:?}", wire_effort(body));
    }
}

fn assert_extra_body(bodies: &[Value]) {
    for body in with_effort(bodies) {
        assert_eq!(
            body.get("extra_body"),
            Some(&json!({ "reasoning_effort": body["output_config"]["effort"] })),
            "{:?}",
            wire_effort(body)
        );
    }
}

#[test]
fn an_mlx_node_engine_row_with_levels_gets_the_effort_in_extra_body() {
    for bodies in messages_bodies_for_row("mlx-node") {
        assert_extra_body(&bodies);
    }
}

#[test]
fn a_plain_messages_endpoint_row_with_levels_gets_no_extra_body() {
    for bodies in messages_bodies_for_row("anthropic") {
        assert_no_extra_body(&bodies);
    }
}

/// One turn on `test-model` over `conn`; returns the bodies `inference` received for it. `other` must receive none.
async fn one_turn(
    conn: acp::ClientSideConnection,
    cwd: std::path::PathBuf,
    inference: &MockInferenceServer,
    other: &MockInferenceServer,
) -> Vec<Value> {
    let session = new_session(&conn, &cwd).await;
    prompt_turn(&conn, &session, "hi").await;
    assert!(
        turn_messages(other).is_empty(),
        "the turn goes to the inference server only"
    );
    turn_messages(inference)
}

/// The mlx-node engine lists the model, but a configured `api_base_url` sends its API-key turns to a server that never listed it.
#[test]
fn an_api_base_url_override_sends_no_extra_body_to_the_other_server() {
    run_agent_test_with_two_servers(
        [vec![test_row("mlx-node")], Vec::new()],
        |cwd, mlx, strict| async move {
            let text = format!("[model.test-model]\napi_base_url = \"{}\"\n", strict.url());
            let home = std::env::var_os("GROK_HOME").expect("the harness sets GROK_HOME");
            std::fs::write(std::path::Path::new(&home).join("config.toml"), &text)
                .expect("write config.toml");
            let config = xai_grok_shell::agent::config::Config::new_from_toml_cfg(
                &toml::from_str(&text).expect("config parses"),
            )
            .expect("valid config");
            let (conn, _init) =
                connect_and_auth_with_config(AutoApproveClient, "test-client", config).await;
            assert_no_extra_body(&one_turn(conn, cwd, &strict, &mlx).await);
        },
    );
}

/// The list comes from the mlx-node engine's `/v1/models`, but inference goes to another base URL.
#[test]
fn a_models_list_on_another_server_sends_no_extra_body_to_the_inference_server() {
    run_agent_test_with_two_servers(
        [vec![test_row("mlx-node")], Vec::new()],
        |cwd, mlx, strict| async move {
            let _list = EnvGuard::set("GROK_MODELS_LIST_URL", format!("{}/models", mlx.url()));
            let _inference = EnvGuard::set("GROK_CLI_CHAT_PROXY_BASE_URL", strict.url());
            let (conn, _init) = connect_and_auth(AutoApproveClient, "test-client").await;
            assert_no_extra_body(&one_turn(conn, cwd, &strict, &mlx).await);
            assert!(
                mlx.request_count_for("/v1/models") > 0,
                "the catalog comes from the mlx-node engine"
            );
        },
    );
}

/// Records, for each `x.ai/models/update` the agent sends after it applies a catalog, how many `/v1/models` fetches
/// the server had answered by then.
#[derive(Clone)]
struct CatalogUpdates {
    server: Rc<MockInferenceServer>,
    fetches_at_update: Rc<std::cell::RefCell<Vec<usize>>>,
    changed: Rc<tokio::sync::Notify>,
}

impl CatalogUpdates {
    fn new(server: Rc<MockInferenceServer>) -> Self {
        Self {
            server,
            fetches_at_update: Rc::default(),
            changed: Rc::default(),
        }
    }

    /// Waits for a catalog applied from a `/v1/models` fetch after the first `fetched`.
    async fn wait_for_a_catalog_fetched_after(&self, fetched: usize) {
        tokio::time::timeout(RPC_TIMEOUT, async {
            while !self
                .fetches_at_update
                .borrow()
                .iter()
                .any(|&fetches| fetches > fetched)
            {
                self.changed.notified().await;
            }
        })
        .await
        .expect("the refreshed catalog never applied");
    }
}

#[async_trait::async_trait(?Send)]
impl acp::Client for CatalogUpdates {
    async fn request_permission(
        &self,
        args: acp::RequestPermissionRequest,
    ) -> acp::Result<acp::RequestPermissionResponse> {
        Ok(acp::RequestPermissionResponse::new(allow_once(&args)))
    }

    async fn session_notification(&self, _args: acp::SessionNotification) -> acp::Result<()> {
        Ok(())
    }

    async fn ext_notification(&self, args: acp::ExtNotification) -> acp::Result<()> {
        if args.method.as_ref().ends_with("x.ai/models/update") {
            self.fetches_at_update
                .borrow_mut()
                .push(self.server.request_count_for("/v1/models"));
            self.changed.notify_one();
        }
        Ok(())
    }
}

/// The Messages bodies of one session's turn before and after the endpoint's row changes from `before` to `after`.
/// The first turn's reply carries a new `x-models-etag`, the production trigger for a catalog refresh.
fn bodies_across_a_refresh(before: &str, after: &str) -> [Vec<Value>; 2] {
    let bodies = Rc::new(std::cell::RefCell::new([Vec::new(), Vec::new()]));
    let out = Rc::clone(&bodies);
    let after = after.to_owned();
    run_agent_test_with_models(vec![test_row(before)], |cwd, server| async move {
        let updates = CatalogUpdates::new(Rc::clone(&server));
        let (conn, _init) = connect_and_auth(updates.clone(), "test-client").await;
        let session = new_session(&conn, &cwd).await;
        server.set_models(vec![test_row(&after)]);
        for _ in 0..3 {
            let mut reply = ScriptedResponse::sse(xai_grok_test_support::sse::messages_api_script(
                "ok",
                "test-model",
                "end_turn",
            ));
            reply
                .headers
                .push(("x-models-etag".to_owned(), "catalog-2".to_owned()));
            server.enqueue_response("/v1/messages", reply);
        }
        let fetched = server.request_count_for("/v1/models");
        // The disk cache is past its TTL, or the refresh would reread it instead of the endpoint.
        let cache = xai_dirs::grok_home().join("models_cache.json");
        if cache.exists() {
            std::fs::remove_file(cache).expect("expire the models cache");
        }
        prompt_turn(&conn, &session, "first").await;
        updates.wait_for_a_catalog_fetched_after(fetched).await;
        let first = messages(&server);
        prompt_turn(&conn, &session, "second").await;
        let all = messages(&server);
        let mut out = out.borrow_mut();
        out[1] = all[first.len()..].to_vec();
        out[0] = first;
    });
    bodies.take()
}

#[test]
fn a_refresh_that_drops_the_mlx_node_row_stops_extra_body_on_a_running_session() {
    let [before, after] = bodies_across_a_refresh("mlx-node", "anthropic");
    assert_extra_body(&before);
    assert_no_extra_body(&after);
}

#[test]
fn a_refresh_that_lists_an_mlx_node_row_starts_extra_body_on_a_running_session() {
    let [before, after] = bodies_across_a_refresh("anthropic", "mlx-node");
    assert_no_extra_body(&before);
    assert_extra_body(&after);
}
