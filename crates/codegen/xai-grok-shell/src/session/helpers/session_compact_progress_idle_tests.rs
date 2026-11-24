use super::*;
use crate::sampling::{Client, ConversationItem, SamplerConfig};
use axum::Router;
use axum::response::IntoResponse;
use axum::response::sse::{Event, Sse};
use axum::routing::post;
use futures_util::stream;
use serde_json::json;
use std::time::Duration;
use tokio::net::TcpListener;

const SUMMARY: &str = "<summary>ok</summary>";
const IDLE: Duration = Duration::from_millis(300);
const GAP: Duration = Duration::from_millis(60);
/// Longer than `IDLE`, so only the pings can keep the call alive.
const PROMPT_READ: Duration = Duration::from_millis(900);

#[derive(Clone, Copy)]
enum WhilePromptIsRead {
    AdvancingPings,
    StalledPings,
    Silence,
}

fn data(value: serde_json::Value) -> Event {
    Event::default().data(value.to_string())
}

fn ping(prefill_done: u32) -> Event {
    data(json!({
        "type": "ping",
        "mlx_progress": {"phase": "prefill", "prefill_total": 100_000, "prefill_done": prefill_done}
    }))
}

fn summary_events() -> Vec<Event> {
    vec![
        data(
            json!({ "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""} }),
        ),
        data(
            json!({ "type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": SUMMARY} }),
        ),
        data(json!({ "type": "content_block_stop", "index": 0 })),
        data(
            json!({ "type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5} }),
        ),
        data(json!({ "type": "message_stop" })),
    ]
}

/// `message_start`, then a prompt read longer than the idle limit, then the summary.
async fn compact_against(read: WhilePromptIsRead) -> Result<CompactOutput, CompactFailure> {
    let app = Router::new().route(
        "/v1/messages",
        post(move || async move {
            let start = data(json!({
                "type": "message_start",
                "message": {
                    "id": "msg_test", "type": "message", "role": "assistant",
                    "content": [], "model": "test-model", "stop_reason": null,
                    "usage": {"input_tokens": 10, "output_tokens": 0}
                }
            }));
            let steps = (PROMPT_READ.as_millis() / GAP.as_millis()) as u32;
            let reading: Vec<(Duration, Option<Event>)> = match read {
                WhilePromptIsRead::AdvancingPings => {
                    (1..=steps).map(|i| (GAP, Some(ping(i * 1000)))).collect()
                }
                WhilePromptIsRead::StalledPings => {
                    (1..=steps).map(|_| (GAP, Some(ping(1000)))).collect()
                }
                WhilePromptIsRead::Silence => vec![(PROMPT_READ, None)],
            };
            let events =
                stream::iter(std::iter::once((Duration::ZERO, Some(start))).chain(reading))
                    .then(|(wait, event)| async move {
                        tokio::time::sleep(wait).await;
                        event
                    })
                    .filter_map(|event| async move { event })
                    .chain(stream::iter(summary_events()))
                    .map(Ok::<_, std::convert::Infallible>);
            Sse::new(events).into_response()
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let config = SamplerConfig {
        api_key: Some("test-api-key".into()),
        base_url: format!("http://{addr}/v1"),
        model: "test-model".into(),
        max_completion_tokens: Some(1000),
        api_backend: ApiBackend::Messages,
        context_window: 256_000,
        ..Default::default()
    };
    let client = Client::new(config.clone()).unwrap();
    generate_session_compact(
        vec![
            ConversationItem::system("sys"),
            ConversationItem::user("Summarize the conversation so far."),
        ],
        0,
        vec![],
        vec![],
        client,
        acp::SessionId::new("progress-idle"),
        &config,
        IDLE,
        0,
        crate::util::config::CompactionToolChoice::Auto,
        crate::util::config::DEFAULT_COMPACTION_THINKING_BUDGET_TOKENS,
        &tokio_util::sync::CancellationToken::new(),
    )
    .await
}

fn idle_timeout_or_outcome(result: Result<CompactOutput, CompactFailure>) -> Result<(), String> {
    match result {
        Err(CompactFailure::Transient(e)) if format!("{e:?}").contains("stream idle timeout") => {
            Ok(())
        }
        Err(other) => Err(format!("other failure: {other:?}")),
        Ok(output) => Err(format!("finished: {}", output.content)),
    }
}

#[tokio::test]
async fn advancing_progress_pings_keep_a_long_prompt_read_alive() {
    let result = compact_against(WhilePromptIsRead::AdvancingPings).await;
    match result {
        Ok(output) => assert_eq!(output.content, SUMMARY),
        Err(e) => panic!("pings that advance must count as activity: {e:?}"),
    }
}

#[tokio::test]
async fn stalled_progress_pings_end_as_idle_timeout() {
    let result = compact_against(WhilePromptIsRead::StalledPings).await;
    idle_timeout_or_outcome(result).unwrap();
}

#[tokio::test]
async fn silence_ends_as_idle_timeout() {
    let result = compact_against(WhilePromptIsRead::Silence).await;
    idle_timeout_or_outcome(result).unwrap();
}
