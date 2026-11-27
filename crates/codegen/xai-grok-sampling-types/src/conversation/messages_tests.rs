use super::test_support::*;
use super::*;

fn messages_test_request(reasoning_effort: Option<crate::ReasoningEffort>) -> ConversationRequest {
    ConversationRequest {
        items: vec![ConversationItem::user("Hello")],
        model: Some("test-model".to_string()),
        reasoning_effort,
        ..Default::default()
    }
}

#[test]
fn json_schema_and_reasoning_effort_are_orthogonal_in_output_config() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "x": { "type": "string" } },
        "required": ["x"]
    });
    let mut req = ConversationRequest::from_items(vec![ConversationItem::user("go")])
        .with_json_schema(schema);
    req.reasoning_effort = Some(crate::ReasoningEffort::High);

    let msgs = build_messages_request(&req, false);
    let oc = msgs.output_config.expect("output_config present");
    assert_eq!(oc.effort.as_deref(), Some("high"));
    assert!(oc.format.is_some());
    assert!(
        msgs.thinking.is_some(),
        "thinking set when effort is present"
    );
}

#[test]
fn test_messages_request_wire_format_for_supported_variants() {
    for (variant, expected) in [
        (crate::ReasoningEffort::Low, "low"),
        (crate::ReasoningEffort::Medium, "medium"),
        (crate::ReasoningEffort::High, "high"),
        (crate::ReasoningEffort::Xhigh, "xhigh"),
        (crate::ReasoningEffort::Max, "max"),
    ] {
        let req = messages_test_request(Some(variant));
        let msgs = build_messages_request(&req, false);
        let json = serde_json::to_value(&msgs).unwrap();
        assert_eq!(
            json.pointer("/output_config/effort")
                .and_then(|v| v.as_str()),
            Some(expected),
            "{variant:?} should map to output_config.effort={expected:?}; got: {json:#}",
        );
        assert_eq!(
            json.pointer("/thinking/type").and_then(|v| v.as_str()),
            Some("adaptive"),
            "{variant:?} should auto-pair thinking.type=adaptive; got: {json:#}",
        );
    }
}

#[test]
fn test_messages_request_omits_output_config_when_no_supported_effort() {
    let none_or_unsupported = [
        None,
        Some(crate::ReasoningEffort::None),
        Some(crate::ReasoningEffort::Minimal),
    ];
    for input in none_or_unsupported {
        let req = messages_test_request(input);
        let msgs = build_messages_request(&req, false);
        assert!(
            msgs.output_config.is_none(),
            "input {input:?} must not produce output_config",
        );
        assert!(
            msgs.thinking.is_none(),
            "input {input:?} must not auto-pair thinking",
        );
    }
}

const MLX_SERVER_EFFORTS: [&str; 6] = ["none", "low", "medium", "high", "xhigh", "max"];

const ALL_EFFORTS: [crate::ReasoningEffort; 7] = [
    crate::ReasoningEffort::None,
    crate::ReasoningEffort::Minimal,
    crate::ReasoningEffort::Low,
    crate::ReasoningEffort::Medium,
    crate::ReasoningEffort::High,
    crate::ReasoningEffort::Xhigh,
    crate::ReasoningEffort::Max,
];

/// The body for an mlx-node engine that lists `reasoningEfforts`.
fn mlx_body(reasoning_effort: Option<crate::ReasoningEffort>) -> serde_json::Value {
    serde_json::to_value(build_messages_request(
        &messages_test_request(reasoning_effort),
        true,
    ))
    .unwrap()
}

fn top_level_keys(json: &serde_json::Value) -> Vec<&str> {
    let mut keys: Vec<&str> = json
        .as_object()
        .expect("request body is an object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    keys
}

#[test]
fn test_messages_request_extra_body_carries_the_level_as_is() {
    for (variant, expected) in [
        (crate::ReasoningEffort::None, "none"),
        (crate::ReasoningEffort::Low, "low"),
        (crate::ReasoningEffort::Medium, "medium"),
        (crate::ReasoningEffort::High, "high"),
        (crate::ReasoningEffort::Xhigh, "xhigh"),
        (crate::ReasoningEffort::Max, "max"),
    ] {
        let json = mlx_body(Some(variant));
        assert_eq!(
            json.get("extra_body"),
            Some(&serde_json::json!({ "reasoning_effort": expected })),
            "{variant:?}: {json:#}",
        );
    }
}

#[test]
fn test_messages_request_extra_body_never_sends_minimal() {
    for input in [None, Some(crate::ReasoningEffort::Minimal)] {
        let json = mlx_body(input);
        assert!(json.get("extra_body").is_none(), "{input:?}: {json:#}");
        assert_eq!(
            json,
            mlx_body(None),
            "{input:?}: the body equals the one with no level"
        );
    }
}

#[test]
fn test_messages_request_every_sent_level_is_one_the_mlx_server_accepts() {
    for variant in ALL_EFFORTS {
        let json = mlx_body(Some(variant));
        if let Some(level) = json.pointer("/extra_body/reasoning_effort") {
            let level = level.as_str().expect("level is a string");
            assert!(
                MLX_SERVER_EFFORTS.contains(&level),
                "{variant:?} sends {level:?}, which the MLX server rejects",
            );
        }
    }
}

#[test]
fn test_messages_request_off_adds_only_extra_body() {
    let unset = mlx_body(None);
    let mut off = mlx_body(Some(crate::ReasoningEffort::None));
    let extra_body = off
        .as_object_mut()
        .and_then(|body| body.remove("extra_body"))
        .expect("Off sends extra_body");
    assert_eq!(
        extra_body,
        serde_json::json!({ "reasoning_effort": "none" })
    );
    assert_eq!(off, unset, "Off sends no thinking and no output_config");
}

#[test]
fn test_messages_request_extra_body_leaves_output_config_and_thinking_as_they_were() {
    let json = mlx_body(Some(crate::ReasoningEffort::Low));
    assert_eq!(
        top_level_keys(&json),
        [
            "extra_body",
            "max_tokens",
            "messages",
            "model",
            "output_config",
            "thinking"
        ],
        "{json:#}",
    );
    assert_eq!(
        json.get("output_config"),
        Some(&serde_json::json!({ "effort": "low" })),
        "{json:#}",
    );
    assert_eq!(
        json.get("thinking"),
        Some(&serde_json::json!({ "type": "adaptive", "display": "summarized" })),
        "{json:#}",
    );
    assert_eq!(
        json.get("extra_body"),
        Some(&serde_json::json!({ "reasoning_effort": "low" })),
        "{json:#}",
    );
}

#[test]
fn test_messages_request_plain_endpoint_never_gets_extra_body() {
    for variant in ALL_EFFORTS {
        let json = serde_json::to_value(build_messages_request(
            &messages_test_request(Some(variant)),
            false,
        ))
        .unwrap();
        assert!(json.get("extra_body").is_none(), "{variant:?}: {json:#}");
    }
}

fn golden_request(
    effort: Option<crate::ReasoningEffort>,
    thinking_budget_tokens: Option<u32>,
) -> ConversationRequest {
    ConversationRequest {
        items: vec![ConversationItem::user("hi")],
        model: Some("q35".to_string()),
        max_output_tokens: Some(64),
        reasoning_effort: effort,
        thinking_budget_tokens,
        ..Default::default()
    }
}

/// mlx-node's `__test__/server/messages-effort.test.ts` posts these exact bytes to its server; change both together.
const MLX_NODE_GOLDEN_BODIES: [(crate::ReasoningEffort, &str); 2] = [
    (
        crate::ReasoningEffort::None,
        r#"{"model":"q35","messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64,"extra_body":{"reasoning_effort":"none"}}"#,
    ),
    (
        crate::ReasoningEffort::Low,
        r#"{"model":"q35","messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64,"thinking":{"type":"adaptive","display":"summarized"},"output_config":{"effort":"low"},"extra_body":{"reasoning_effort":"low"}}"#,
    ),
];

#[test]
fn test_messages_request_mlx_node_golden_bodies() {
    for (effort, golden) in MLX_NODE_GOLDEN_BODIES {
        assert_eq!(
            serde_json::to_string(&build_messages_request(
                &golden_request(Some(effort), None),
                true
            ))
            .unwrap(),
            golden,
            "{effort:?}",
        );
    }
}

/// The bytes tag `mlx-agent-v0.3.4` sends for these requests; a plain Messages endpoint must keep getting them.
const PLAIN_ENDPOINT_GOLDEN_BODIES: [(Option<crate::ReasoningEffort>, Option<u32>, &str); 9] = [
    (
        None,
        None,
        r#"{"model":"q35","messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64}"#,
    ),
    (
        Some(crate::ReasoningEffort::None),
        None,
        r#"{"model":"q35","messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64}"#,
    ),
    (
        Some(crate::ReasoningEffort::Minimal),
        None,
        r#"{"model":"q35","messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64}"#,
    ),
    (
        Some(crate::ReasoningEffort::Low),
        None,
        r#"{"model":"q35","messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64,"thinking":{"type":"adaptive","display":"summarized"},"output_config":{"effort":"low"}}"#,
    ),
    (
        Some(crate::ReasoningEffort::Medium),
        None,
        r#"{"model":"q35","messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64,"thinking":{"type":"adaptive","display":"summarized"},"output_config":{"effort":"medium"}}"#,
    ),
    (
        Some(crate::ReasoningEffort::High),
        None,
        r#"{"model":"q35","messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64,"thinking":{"type":"adaptive","display":"summarized"},"output_config":{"effort":"high"}}"#,
    ),
    (
        Some(crate::ReasoningEffort::Xhigh),
        None,
        r#"{"model":"q35","messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64,"thinking":{"type":"adaptive","display":"summarized"},"output_config":{"effort":"xhigh"}}"#,
    ),
    (
        Some(crate::ReasoningEffort::Max),
        None,
        r#"{"model":"q35","messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64,"thinking":{"type":"adaptive","display":"summarized"},"output_config":{"effort":"max"}}"#,
    ),
    (
        Some(crate::ReasoningEffort::Low),
        Some(4096),
        r#"{"model":"q35","messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64,"thinking":{"type":"enabled","budget_tokens":4096},"output_config":{"effort":"low"}}"#,
    ),
];

#[test]
fn test_messages_request_plain_endpoint_golden_bodies_match_v0_3_4() {
    for (effort, budget, golden) in PLAIN_ENDPOINT_GOLDEN_BODIES {
        assert_eq!(
            serde_json::to_string(&build_messages_request(
                &golden_request(effort, budget),
                false
            ))
            .unwrap(),
            golden,
            "{effort:?} budget={budget:?}",
        );
    }
}

#[test]
fn test_messages_request_thinking_carries_summarized_display() {
    let req = ConversationRequest {
        reasoning_effort: Some(crate::ReasoningEffort::High),
        ..ConversationRequest::from_items(vec![ConversationItem::user("hi")])
            .with_model("messages-compatible-model")
    };
    let msg = build_messages_request(&req, false);
    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(
        json.pointer("/thinking/type").and_then(|v| v.as_str()),
        Some("adaptive"),
        "thinking.type should be 'adaptive'; got: {json:#}",
    );
    assert_eq!(
        json.pointer("/thinking/display").and_then(|v| v.as_str()),
        Some("summarized"),
        "thinking.display must be 'summarized' so 4.7+ surfaces thinking content; got: {json:#}",
    );
}

#[test]
fn test_messages_request_omits_thinking_when_effort_unset() {
    let req = ConversationRequest::from_items(vec![ConversationItem::user("hi")])
        .with_model("messages-compatible-model");
    let msg = build_messages_request(&req, false);
    let json = serde_json::to_value(&msg).unwrap();
    assert!(
        json.get("thinking").is_none()
            || json
                .pointer("/thinking")
                .map(|v| v.is_null())
                .unwrap_or(false),
        "thinking must be absent when reasoning_effort is unset; got: {json:#}",
    );
    assert!(
        json.get("output_config").is_none()
            || json
                .pointer("/output_config")
                .map(|v| v.is_null())
                .unwrap_or(false),
        "output_config must be absent when reasoning_effort is unset; got: {json:#}",
    );
}

#[test]
fn test_messages_request_previous_tip_skips_a_trailing_user_run() {
    let mut items = vec![
        ConversationItem::system("You are a helpful assistant."),
        ConversationItem::user("Fix the bug"),
    ];
    items.extend(agent_turn(0));
    items.extend(agent_turn(1));
    // The shape after a parallel batch: tool results, then followups.
    items.push(ConversationItem::user("[Image content]"));
    items.push(ConversationItem::user("<system-reminder>"));

    let json = serde_json::to_value(build_messages_request(
        &ConversationRequest::from_items(items).with_model("messages-compatible-model"),
        false,
    ))
    .unwrap();
    let Some(messages) = json.get("messages").and_then(|v| v.as_array()) else {
        panic!("expected messages array: {json:#}");
    };

    let marked: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| marker_on_last_block(m).is_some())
        .map(|(i, _)| i)
        .collect();
    let last_assistant = messages
        .iter()
        .rposition(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        .unwrap();
    let [prev_tip, tip] = marked.as_slice() else {
        panic!("tip and previous tip only: {json:#}");
    };
    assert_eq!(*tip, messages.len() - 1, "tip: {json:#}");
    assert!(
        *prev_tip < last_assistant,
        "the previous tip must sit before the last assistant turn, not inside \
             the trailing user run; got {marked:?} in {json:#}",
    );
}

#[test]
fn test_messages_request_cache_breakpoint_marks_an_image_tip() {
    let req = ConversationRequest::from_items(vec![
        ConversationItem::system("You are a helpful assistant."),
        ConversationItem::User(UserItem {
            content: vec![
                ContentPart::Text {
                    text: "what is in this screenshot".into(),
                },
                ContentPart::Image {
                    url: "data:image/png;base64,iVBOR".into(),
                },
            ],
            ..Default::default()
        }),
    ])
    .with_model("messages-compatible-model");

    let json = serde_json::to_value(build_messages_request(&req, false)).unwrap();
    let Some(first_msg) = json
        .get("messages")
        .and_then(|v| v.as_array())
        .and_then(|m| m.first())
    else {
        panic!("expected first message: {json:#}");
    };
    let Some(blocks) = first_msg.get("content").and_then(|c| c.as_array()) else {
        panic!("expected content array: {json:#}");
    };

    assert_eq!(
        blocks
            .last()
            .and_then(|b| b.get("type"))
            .and_then(|t| t.as_str()),
        Some("image")
    );
    assert_eq!(
        marker_on_last_block(first_msg),
        Some("ephemeral"),
        "{json:#}",
    );
    assert!(
        blocks
            .first()
            .is_some_and(|b| b.get("cache_control").is_none()),
        "{json:#}"
    );
}

#[test]
fn test_messages_request_cache_breakpoint_skips_thinking() {
    let req = ConversationRequest::from_items(vec![
        ConversationItem::user("Fix the bug"),
        ConversationItem::Reasoning(synthesized_reasoning_item("weighing options")),
        ConversationItem::assistant("Fixed it."),
    ])
    .with_model("messages-compatible-model");

    let json = serde_json::to_value(build_messages_request(&req, false)).unwrap();
    let Some(asst) = json
        .get("messages")
        .and_then(|v| v.as_array())
        .and_then(|m| m.get(1))
    else {
        panic!("expected assistant message: {json:#}");
    };
    let Some(blocks) = asst.get("content").and_then(|c| c.as_array()) else {
        panic!("expected content array: {json:#}");
    };

    let thinking = blocks
        .iter()
        .find(|b| b.get("type").and_then(|t| t.as_str()) == Some("thinking"))
        .expect("reasoning should emit a thinking block");
    assert!(thinking.get("cache_control").is_none(), "{json:#}");
    assert_eq!(marker_on_last_block(asst), Some("ephemeral"), "{json:#}",);
}

#[test]
fn test_btw_cross_api_messages_no_regressions() {
    let items = btw_prepare_items(btw_mid_turn_conversation());
    let req = ConversationRequest::from_items(items);
    let msg = build_messages_request(&req, false);
    let json = serde_json::to_value(&msg).unwrap();

    let messages = json.get("messages").unwrap().as_array().unwrap();

    // No thinking blocks anywhere.
    for (i, m) in messages.iter().enumerate() {
        if let Some(content) = m.get("content").and_then(|c| c.as_array()) {
            for block in content {
                assert_ne!(
                    block.get("type").and_then(|t| t.as_str()),
                    Some("thinking"),
                    "messages[{i}] must not contain thinking blocks",
                );
            }
        }
    }

    // Last assistant message must not have unanswered tool_use.
    let last_assistant = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        .expect("should have an assistant message");
    if let Some(content) = last_assistant.get("content").and_then(|c| c.as_array()) {
        for block in content {
            assert_ne!(
                block.get("type").and_then(|t| t.as_str()),
                Some("tool_use"),
                "last assistant in btw request must not have unanswered tool_use",
            );
        }
    }

    // Top-level thinking must be absent (no reasoning_effort set).
    assert!(
        json.get("thinking").is_none() || json.pointer("/thinking").is_some_and(|v| v.is_null()),
        "top-level thinking must be absent; got: {json:#}",
    );

    assert!(
        json.get("temperature").is_none()
            || json.pointer("/temperature").is_some_and(|v| v.is_null()),
        "temperature must be absent so proxy defaults can apply; got: {json:#}",
    );

    // The completed tool pair (call_1) must survive.
    let has_tool_use_call_1 = messages.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_array())
            .is_some_and(|blocks| {
                blocks.iter().any(|b| {
                    b.get("type").and_then(|t| t.as_str()) == Some("tool_use")
                        && b.get("id").and_then(|id| id.as_str()) == Some("call_1")
                })
            })
    });
    assert!(
        has_tool_use_call_1,
        "completed tool_use call_1 must survive"
    );

    let has_tool_result_call_1 = messages.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_array())
            .is_some_and(|blocks| {
                blocks.iter().any(|b| {
                    b.get("type").and_then(|t| t.as_str()) == Some("tool_result")
                        && b.get("tool_use_id").and_then(|id| id.as_str()) == Some("call_1")
                })
            })
    });
    assert!(
        has_tool_result_call_1,
        "completed tool_result for call_1 must survive"
    );
}

#[test]
fn test_tool_result_with_images_to_anthropic() {
    let req = ConversationRequest::from_items(vec![
        ConversationItem::user("Read this"),
        ConversationItem::Assistant(AssistantItem {
            content: String::new().into(),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "read_file".to_string(),
                arguments: "{}".into(),
            }],
            model_id: None,
            model_fingerprint: None,
            reasoning_effort: None,
        }),
        ConversationItem::tool_result_with_images(
            "call_1",
            "Read image file: photo.png",
            vec![ContentPart::Image {
                url: "data:image/png;base64,iVBOR".into(),
            }],
        ),
    ]);

    let messages_req = build_messages_request(&req, false);

    // Find the user message that contains the tool result (the Messages API wraps tool results in user messages)
    let tool_result_msg = messages_req
        .messages
        .iter()
        .find(|m| {
            if let crate::messages::MessageContent::Blocks(blocks) = &m.content {
                blocks
                    .iter()
                    .any(|b| matches!(b, crate::messages::ContentBlock::ToolResult { .. }))
            } else {
                false
            }
        })
        .expect("Expected a message with ToolResult block");

    let crate::messages::MessageContent::Blocks(blocks) = &tool_result_msg.content else {
        panic!("Expected Blocks");
    };
    let tool_result_block = blocks
        .iter()
        .find_map(|b| {
            if let crate::messages::ContentBlock::ToolResult { content, .. } = b {
                Some(content)
            } else {
                None
            }
        })
        .unwrap();

    let crate::messages::ToolResultContent::Blocks(inner) = tool_result_block else {
        panic!("Expected ToolResultContent::Blocks, got Text");
    };
    let [t0, t1] = inner.as_slice() else {
        panic!("expected two inner blocks: {inner:?}");
    };
    assert!(
        matches!(t0, crate::messages::ContentBlock::Text { text, .. } if text == "Read image file: photo.png")
    );
    assert!(
        matches!(t1, crate::messages::ContentBlock::Image { source: crate::messages::ImageSource::Base64 { media_type, data }, .. } if media_type == "image/png" && data == "iVBOR")
    );
}

#[test]
fn upgrade_legacy_reasoning_singular_anthropic_no_id() {
    // Messages streaming sets id = "" (see stream/messages.rs:340).
    // The upgrader must still emit a sibling carrying text and signature
    let raw = serde_json::json!({
        "type": "assistant",
        "content": "answer",
        "reasoning": {
            "text": "Let me think about this...",
            "encrypted": "signature-bytes-here",
            "id": ""
        },
        "model_id": "messages-compatible-model"
    });
    let mut seen = std::collections::HashSet::new();
    let siblings = upgrade_legacy_reasoning(&raw, &mut seen);
    assert_eq!(siblings.len(), 1);
    let Some(ConversationItem::Reasoning(r)) = siblings.first() else {
        panic!("expected Reasoning sibling: {siblings:?}");
    };
    assert_eq!(r.id, "");
    assert_eq!(r.encrypted_content.as_deref(), Some("signature-bytes-here"));
}
