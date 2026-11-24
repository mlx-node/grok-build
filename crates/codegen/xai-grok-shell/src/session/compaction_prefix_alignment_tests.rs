use super::{apply_turn_image_budget_and_prune, summary_must_strip_reasoning};
use crate::sampling::{Client, ConversationItem, SamplerConfig, ToolCall};
use crate::session::helpers::full_replace_compaction::ShellCompactionSampler;
use agent_client_protocol as acp;
use axum::Router;
use axum::body::Bytes;
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::post;
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;
use xai_chat_state::compaction_utils::prepare_conversation_for_verbatim_summarization;
use xai_grok_compaction::{CompactionPrompt, CompactionSampler};
use xai_grok_sampling_types::{ApiBackend, ReasoningEffort, ToolSpec, synthesized_reasoning_item};

const SUMMARY: &str = "<summary>ok</summary>";

fn summary_stream() -> Vec<Event> {
    [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_test", "type": "message", "role": "assistant",
                "content": [], "model": "test-model", "stop_reason": null,
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }
        }),
        json!({ "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""} }),
        json!({ "type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": SUMMARY} }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5} }),
        json!({ "type": "message_stop" }),
    ]
    .into_iter()
    .map(|event| Event::default().data(event.to_string()))
    .collect()
}

/// Captures every `/v1/messages` body in arrival order.
async fn capture_server() -> (String, Arc<Mutex<Vec<Value>>>) {
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let sink = bodies.clone();
    let app = Router::new().route(
        "/v1/messages",
        post(move |body: Bytes| {
            let sink = sink.clone();
            async move {
                sink.lock()
                    .unwrap()
                    .push(serde_json::from_slice(&body).unwrap());
                let events = summary_stream()
                    .into_iter()
                    .map(Ok::<_, std::convert::Infallible>);
                Sse::new(futures_util::stream::iter(events))
                    .keep_alive(KeepAlive::default())
                    .into_response()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}/v1"), bodies)
}

fn reasoning(text: &str, signature: Option<&str>) -> ConversationItem {
    let mut item = synthesized_reasoning_item(text);
    item.encrypted_content = signature.map(str::to_owned);
    ConversationItem::Reasoning(item)
}

fn tool_call(id: &str) -> ConversationItem {
    ConversationItem::assistant_tool_calls(vec![ToolCall {
        id: id.into(),
        name: "grep".to_string(),
        arguments: r#"{"pattern":"fn main"}"#.into(),
    }])
}

/// A local-server history: the stream stores no signature, so every reasoning item is unsigned.
fn history(signature: Option<&str>) -> Vec<ConversationItem> {
    vec![
        ConversationItem::system("You are a coding agent."),
        ConversationItem::user("<user_query>\nfind main\n</user_query>"),
        reasoning("look for main first", signature),
        tool_call("c1"),
        ConversationItem::tool_result("c1", "src/main.rs:1:fn main() {}"),
        reasoning("found it, report", signature),
        ConversationItem::assistant("main is in src/main.rs"),
        ConversationItem::user("<user_query>\nnow find tests\n</user_query>"),
        reasoning("grep for #[test]", signature),
        tool_call("c2"),
        ConversationItem::tool_result("c2", "src/lib.rs:9:#[test]"),
    ]
}

fn tools() -> Vec<ToolSpec> {
    vec![ToolSpec {
        name: "grep".to_string(),
        description: Some("Search files".to_string()),
        parameters: json!({"type": "object", "properties": {"pattern": {"type": "string"}}}),
    }]
}

/// Breakpoints follow each request's own tip; they are request metadata, not prompt text.
fn without_cache_control(mut value: Value) -> Value {
    fn strip(value: &mut Value) {
        match value {
            Value::Object(map) => {
                map.remove("cache_control");
                map.values_mut().for_each(strip);
            }
            Value::Array(items) => items.iter_mut().for_each(strip),
            _ => {}
        }
    }
    strip(&mut value);
    value
}

fn thinking_blocks(messages: &[Value]) -> usize {
    messages
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|b| b.get("type") == Some(&json!("thinking")))
        .count()
}

/// The main-loop request and the compaction request for the same history, as sent on the wire.
async fn main_and_compaction_bodies(signature: Option<&str>) -> (Value, Value) {
    let (base_url, bodies) = capture_server().await;
    let sampler_config = SamplerConfig {
        api_key: Some("test-api-key".into()),
        base_url: base_url.clone(),
        model: "test-model".into(),
        max_completion_tokens: Some(1000),
        api_backend: ApiBackend::Messages,
        context_window: 256_000,
        reasoning_effort: Some(ReasoningEffort::High),
        ..Default::default()
    };
    let client = Client::new(sampler_config.clone()).unwrap();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    let chat_state = xai_chat_state::ChatStateActor::spawn(
        history(signature),
        xai_grok_sampling_types::SamplingConfig {
            base_url,
            model: "test-model".to_string(),
            api_backend: ApiBackend::Messages,
            context_window: std::num::NonZeroU64::new(256_000).unwrap(),
            reasoning_effort: Some(ReasoningEffort::High),
            ..Default::default()
        },
        Box::new(xai_chat_state::NullChatPersistence),
        event_tx,
        tokio_util::sync::CancellationToken::new(),
    );

    let main_request = chat_state
        .build_request(
            tools(),
            None,
            false,
            None,
            "conv".to_string(),
            "req".to_string(),
        )
        .await
        .unwrap();
    let (mut stream, _) = client
        .conversation_stream_messages(main_request)
        .await
        .unwrap();
    while stream.next().await.is_some() {}

    let conversation = chat_state.get_conversation().await;
    let strips = summary_must_strip_reasoning(Some(&ApiBackend::Messages), &conversation);
    let turns = apply_turn_image_budget_and_prune(
        &chat_state,
        prepare_conversation_for_verbatim_summarization(conversation, strips),
    )
    .await;
    let compaction_tools = tools();
    let sampler = ShellCompactionSampler::new(
        false,
        None,
        compaction_tools.clone(),
        vec![],
        xai_chat_state::estimate_tool_specs_tokens(&compaction_tools),
        client,
        acp::SessionId::new("prefix-alignment"),
        sampler_config,
        std::time::Duration::from_secs(30),
        0,
        crate::util::config::CompactionToolChoice::Auto,
        crate::util::config::DEFAULT_COMPACTION_THINKING_BUDGET_TOKENS,
        tokio_util::sync::CancellationToken::new(),
    );
    let prompt = CompactionPrompt {
        system: String::new(),
        user: String::new(),
    };
    let output = sampler
        .sample_compaction(&turns, &prompt, std::time::Duration::from_secs(30))
        .await
        .unwrap_or_else(|e| panic!("compaction must succeed: {e}"));
    assert_eq!(output.response, SUMMARY);

    let mut bodies = std::mem::take(&mut *bodies.lock().unwrap());
    assert_eq!(bodies.len(), 2, "one main request, one compaction request");
    let compaction = bodies.pop().unwrap();
    let main = bodies.pop().unwrap();
    (main, compaction)
}

#[tokio::test]
async fn compaction_request_extends_the_main_loop_prefix_with_unsigned_reasoning() {
    let (main, compaction) = main_and_compaction_bodies(None).await;
    let main = without_cache_control(main);
    let compaction = without_cache_control(compaction);

    assert_eq!(compaction["system"], main["system"], "system block");
    assert_eq!(compaction["tools"], main["tools"], "tools block");
    let main_messages = main["messages"].as_array().unwrap();
    let compaction_messages = compaction["messages"].as_array().unwrap();
    assert_eq!(
        thinking_blocks(main_messages),
        3,
        "the main loop replays unsigned reasoning"
    );
    assert_eq!(
        compaction_messages.len(),
        main_messages.len() + 1,
        "the compaction request is the main history plus its own prompt"
    );
    for (index, (sent, expected)) in compaction_messages.iter().zip(main_messages).enumerate() {
        assert_eq!(sent, expected, "message {index} differs from the main loop");
    }
    let prompt = compaction_messages.last().unwrap();
    assert_eq!(prompt["role"], json!("user"));
    assert_eq!(
        main["thinking"],
        json!({"type": "adaptive", "display": "summarized"}),
        "the main loop keeps the session's thinking"
    );
    assert_eq!(
        compaction["thinking"],
        json!({"type": "enabled", "budget_tokens": 4096}),
        "the summary call has its own budget"
    );
}

#[tokio::test]
async fn compaction_request_drops_signed_reasoning_on_messages() {
    let (main, compaction) = main_and_compaction_bodies(Some("sig")).await;
    assert_eq!(thinking_blocks(main["messages"].as_array().unwrap()), 3);
    assert_eq!(
        thinking_blocks(compaction["messages"].as_array().unwrap()),
        0,
        "a signing provider rejects a mutated signed history"
    );
}

#[test]
fn only_signed_reasoning_on_messages_is_stripped() {
    let unsigned = history(None);
    let signed = history(Some("sig"));
    assert!(!summary_must_strip_reasoning(
        Some(&ApiBackend::Messages),
        &unsigned
    ));
    assert!(summary_must_strip_reasoning(
        Some(&ApiBackend::Messages),
        &signed
    ));
    assert!(!summary_must_strip_reasoning(
        Some(&ApiBackend::Responses),
        &signed
    ));
    assert!(!summary_must_strip_reasoning(None, &signed));
}
