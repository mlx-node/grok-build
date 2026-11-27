use std::path::Path;
use std::time::Duration;

use agent_client_protocol as acp;
use xai_grok_test_support::MockInferenceServer;

use super::{
    ExplicitSessionOpen, OaiCompatClient, PersistenceContentChunk, PersistenceHandle,
    PersistenceMsg, SessionDeps, StorageMode, Summary, default_model_id, new,
    new_with_explicit_dir, session_dir,
};
use crate::session::helpers::session_summary::title_fallback_from_user_text;
use crate::session::info::Info;
use crate::session::storage::{JsonlStorageAdapter, StorageAdapter};

const PROMPT: &str = "Survey the repo layout and list every crate under the codegen folder for me";

fn title_sampler(server: &MockInferenceServer) -> OaiCompatClient {
    OaiCompatClient::new(xai_grok_sampler::SamplerConfig {
        api_key: Some("test-key".to_owned()),
        base_url: server.url(),
        model: "title-model".to_owned(),
        api_backend: xai_grok_sampler::ApiBackend::Responses,
        max_retries: Some(0),
        ..Default::default()
    })
    .unwrap()
}

fn title_requests(server: &MockInferenceServer) -> usize {
    server
        .request_bodies()
        .iter()
        .filter(|body| body.to_string().contains("session_title"))
        .count()
}

fn send_prompt(handle: &PersistenceHandle) {
    handle
        .tx
        .send(PersistenceMsg::ContentChunk(PersistenceContentChunk::new(
            vec![acp::ContentBlock::Text(acp::TextContent::new(PROMPT))],
        )))
        .unwrap();
}

/// Two round trips: the first lets the chunk queue its `GeneratedTitle`, the second lets the actor apply it.
async fn settle(handle: &PersistenceHandle) {
    for _ in 0..2 {
        let (respond_to, ack) = tokio::sync::oneshot::channel();
        handle
            .tx
            .send(PersistenceMsg::FlushAndAck { respond_to })
            .unwrap();
        ack.await.unwrap().unwrap();
    }
}

fn read_title(dir: &Path) -> String {
    let summary: Summary =
        serde_json::from_slice(&std::fs::read(dir.join("summary.json")).unwrap()).unwrap();
    summary.display_title().to_owned()
}

/// A model title is adopted only after its reply, so its request is logged by the time this returns.
async fn wait_for_title(dir: &Path) -> String {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let title = read_title(dir);
        if !title.is_empty() {
            return title;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "no title was adopted"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn child_info(home: &Path, id: &str) -> Info {
    Info {
        id: acp::SessionId::new(id),
        cwd: home.to_string_lossy().into_owned(),
    }
}

async fn open_child(
    info: &Info,
    dir: &Path,
    server: &MockInferenceServer,
    open: ExplicitSessionOpen,
    label: Option<&str>,
) -> PersistenceHandle {
    new_with_explicit_dir(
        info,
        dir.to_path_buf(),
        default_model_id(),
        title_sampler(server),
        "title-model".to_owned(),
        open,
        label,
    )
    .await
    .unwrap()
}

fn new_open() -> ExplicitSessionOpen {
    ExplicitSessionOpen::New {
        identity: None,
        next_trace_turn: None,
    }
}

async fn seed_child_dir(info: &Info, dir: &Path, title: Option<&str>) {
    let storage = JsonlStorageAdapter::with_explicit_session_dir(dir.to_path_buf());
    storage
        .init_session(info, default_model_id())
        .await
        .unwrap();
    if let Some(title) = title {
        assert!(
            storage
                .set_generated_title_if_absent(info, title.to_owned())
                .await
                .unwrap()
        );
    }
}

/// Mutation that turns it red: drop the label write in `new_with_explicit_dir` (the title becomes the prompt's words).
#[tokio::test]
async fn new_child_title_is_its_label_with_no_title_call() {
    let home = tempfile::TempDir::new().unwrap();
    let server = MockInferenceServer::start().await.unwrap();
    let dir = home.path().join("child");
    let info = child_info(home.path(), "child-label");

    let handle = open_child(&info, &dir, &server, new_open(), Some("track:alpha")).await;
    assert_eq!(read_title(&dir), "track:alpha");
    send_prompt(&handle);
    settle(&handle).await;

    assert_eq!(read_title(&dir), "track:alpha");
    assert_eq!(title_requests(&server), 0);
}

/// Mutations that turn it red: keep the model call for children (`llm: true` in `new_with_explicit_dir`),
/// and send no title when `llm` is false.
#[tokio::test]
async fn new_child_without_a_label_takes_the_prompt_words_with_no_title_call() {
    let home = tempfile::TempDir::new().unwrap();
    let server = MockInferenceServer::start().await.unwrap();
    let dir = home.path().join("child");
    let info = child_info(home.path(), "child-blank-label");

    let handle = open_child(&info, &dir, &server, new_open(), Some("  ")).await;
    assert_eq!(read_title(&dir), "");
    send_prompt(&handle);

    assert_eq!(
        wait_for_title(&dir).await,
        title_fallback_from_user_text(PROMPT)
    );
    assert_eq!(title_requests(&server), 0);
}

/// Mutation that turns it red: write the label over a title the child already has (a resume copies its source's title).
#[tokio::test]
async fn new_child_keeps_a_copied_title_over_its_label() {
    let home = tempfile::TempDir::new().unwrap();
    let server = MockInferenceServer::start().await.unwrap();
    let dir = home.path().join("child");
    let info = child_info(home.path(), "child-copied-title");
    seed_child_dir(&info, &dir, Some("Copied title")).await;

    let handle = open_child(&info, &dir, &server, new_open(), Some("track:alpha")).await;
    assert_eq!(read_title(&dir), "Copied title");
    send_prompt(&handle);
    settle(&handle).await;

    assert_eq!(read_title(&dir), "Copied title");
    assert_eq!(title_requests(&server), 0);
}

#[tokio::test]
async fn woken_child_keeps_its_label_off_and_makes_no_title_call() {
    let home = tempfile::TempDir::new().unwrap();
    let server = MockInferenceServer::start().await.unwrap();
    let dir = home.path().join("child");
    let info = child_info(home.path(), "child-wake");
    seed_child_dir(&info, &dir, None).await;

    let handle = open_child(
        &info,
        &dir,
        &server,
        ExplicitSessionOpen::Wake,
        Some("track:alpha"),
    )
    .await;
    assert_eq!(read_title(&dir), "");
    send_prompt(&handle);

    assert_eq!(
        wait_for_title(&dir).await,
        title_fallback_from_user_text(PROMPT)
    );
    assert_eq!(title_requests(&server), 0);
}

/// Mutation that turns it red: `llm: false` in `new` too (the main session's title then needs no model call).
#[tokio::test]
async fn main_session_title_still_comes_from_one_model_call() {
    let home = tempfile::TempDir::new().unwrap();
    let server = MockInferenceServer::start().await.unwrap();
    let cwd = home.path().join("repo");
    std::fs::create_dir_all(&cwd).unwrap();
    let info = child_info(&cwd, &uuid::Uuid::now_v7().to_string());

    let (handle, _identity) = new(
        &info,
        default_model_id(),
        SessionDeps {
            sampling_client: title_sampler(&server),
            storage_mode: StorageMode::Local,
            auth_manager: None,
            relay_sync: None,
            gateway: None,
            session_summary_model: "title-model".to_owned(),
            registry_title_sync: None,
            search_index: crate::session::storage::search::SharedSearchIndex::never_indexed(),
            session_kind: None,
        },
    )
    .await
    .unwrap();
    send_prompt(&handle);

    assert!(!wait_for_title(&session_dir(&info)).await.is_empty());
    assert_eq!(title_requests(&server), 1);
}
