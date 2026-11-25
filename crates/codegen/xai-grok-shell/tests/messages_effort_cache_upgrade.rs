//! mlx-node fork: a models cache written by `mlx-agent-v0.3.4` has no mlx-node `extra_body` capability on its rows.
//! Its own binary: the models cache path resolves once per process.

mod acp_harness;

use acp_harness::{
    AutoApproveClient, connect_and_auth, new_session, prompt_turn, run_agent_test_with_models,
};
use serde_json::json;
use xai_grok_shell::agent::config::{EndpointsConfig, ModelEntry};
use xai_grok_test_support::MockModelEntry;

/// The `models_cache.json` v0.3.4 writes for the harness's API-key login after it fetched the mlx-node row: same version
/// string (tag releases do not set `GROK_VERSION`), same scope, a row without the capability, fetched just now.
fn v0_3_4_cache(server_url: &str) -> serde_json::Value {
    let mut row = ModelEntry::fallback("test-model", &EndpointsConfig::default());
    row.info.id = Some("test-model".to_owned());
    row.info.base_url = server_url.to_owned();
    row.info.api_backend = xai_grok_sampling_types::ApiBackend::Messages;
    row.info.reasoning_efforts = xai_grok_sampling_types::parse_reasoning_effort_options(
        &[
            json!({ "value": "low", "label": "Low" }),
            json!({ "value": "high", "label": "High", "default": true }),
        ],
        "reasoningEfforts",
    );
    row.info.supports_reasoning_effort = true;
    row.info.reasoning_effort = Some(xai_grok_sampling_types::ReasoningEffort::High);
    row.api_base_url = Some(server_url.to_owned());
    let row = serde_json::to_value(row).expect("row serializes");
    for field in ["extra_body_reasoning_effort", "discovery_base_url"] {
        assert!(
            row["info"].get(field).is_none(),
            "v0.3.4 never wrote `{field}`"
        );
    }
    json!({
        "fetched_at": chrono::Utc::now(),
        "grok_version": xai_grok_version::VERSION,
        "auth_method": "api_key",
        "origin": format!("{server_url}/models"),
        "identity": xai_grok_cloud_config::remote_settings::scope_hash(&[
            "models-api-key",
            "test-key-for-ci",
            "",
        ]),
        "models": { "test-model": row },
    })
}

#[test]
fn a_catalog_cached_by_v0_3_4_is_refetched_before_the_first_mlx_node_request() {
    let row = MockModelEntry::new("test-model")
        .with_api_backend("messages")
        .with_owned_by("mlx-node")
        .with_reasoning_efforts(vec![
            json!({ "value": "low", "label": "Low" }),
            json!({ "value": "high", "label": "High", "default": true }),
        ]);
    run_agent_test_with_models(vec![row], |cwd, server| async move {
        let home = std::path::PathBuf::from(std::env::var_os("GROK_HOME").expect("GROK_HOME"));
        assert_eq!(
            xai_dirs::grok_home(),
            home,
            "the models cache must resolve under this test's home"
        );
        std::fs::write(
            home.join("models_cache.json"),
            serde_json::to_vec_pretty(&v0_3_4_cache(&server.url())).expect("cache serializes"),
        )
        .expect("write the v0.3.4 cache");

        let (conn, _init) = connect_and_auth(AutoApproveClient, "test-client").await;
        let session = new_session(&conn, &cwd).await;
        prompt_turn(&conn, &session, "hi").await;

        let requests = server.requests();
        let first_turn = requests
            .iter()
            .position(|r| r.path == "/v1/messages")
            .expect("the turn reaches the server");
        assert!(
            requests[..first_turn]
                .iter()
                .any(|r| r.path == "/v1/models"),
            "the v0.3.4 catalog must be refetched before the first Messages request: {:?}",
            requests.iter().map(|r| &r.path).collect::<Vec<_>>()
        );
        let bodies: Vec<_> = requests
            .into_iter()
            .filter(|r| r.path == "/v1/messages")
            .filter_map(|r| r.body)
            .filter(|body| body.pointer("/output_config/effort").is_some())
            .collect();
        assert!(!bodies.is_empty(), "the turn sends the row's level");
        for body in &bodies {
            assert_eq!(
                body.get("extra_body"),
                Some(&json!({ "reasoning_effort": body["output_config"]["effort"] })),
                "{:?}",
                (body.get("output_config"), body.get("extra_body"))
            );
        }
    });
}
