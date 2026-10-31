//! Every API failure is an OpenAI-shaped JSON body with a stable `code` and a `fix` hint, built from
//! the one `ApiError` enum; none of them leaks prompt text or a path.

use std::sync::Arc;

use helios_core::api::{router, AppState};
use helios_core::error::{ApiError, SchedulerFault};
use helios_core::tokenizer::PromptTokenizer;
use helios_core::StateMachine;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const FIXTURE: &str = "tests/fixtures/dummy_tokenizer.json";
const SECRET: &str = "sentinel-prompt-text-do-not-leak";

async fn post(app: AppState, body: &str) -> (u16, serde_json::Value) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let service = router().with_state(app);
    tokio::spawn(async move { axum::serve(listener, service).await.unwrap() });
    let mut s = TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut raw = String::new();
    s.read_to_string(&mut raw).await.unwrap();
    let status = raw.split_whitespace().nth(1).unwrap().parse().unwrap();
    let json = raw.split("\r\n\r\n").nth(1).map(|b| serde_json::from_str(b.trim()).unwrap_or_default());
    (status, json.unwrap_or_default())
}

fn ready_app() -> AppState {
    let sm = StateMachine::new();
    sm.mark_ready().unwrap();
    AppState::from(Arc::new(sm))
}

fn chat(content: &str) -> String {
    serde_json::json!({ "model": "m", "messages": [{ "role": "user", "content": content }] }).to_string()
}

fn assert_shape(json: &serde_json::Value, code: &str) {
    let e = &json["error"];
    assert_eq!(e["code"], code, "{json}");
    for key in ["message", "type", "fix"] {
        assert!(e[key].as_str().is_some_and(|s| !s.is_empty()), "missing {key}: {json}");
    }
    assert!(e.get("param").is_some(), "param key must be present (null when unused): {json}");
    assert!(!json.to_string().contains(SECRET), "prompt text leaked: {json}");
}

#[tokio::test]
async fn no_tokenizer_is_501_with_the_error_shape() {
    let (status, json) = post(ready_app(), &chat(SECRET)).await;
    assert_eq!(status, 501);
    assert_shape(&json, "no_tokenizer");
    assert_eq!(json["error"]["type"], "server_error");
}

#[tokio::test]
async fn overlong_prompt_is_400_naming_the_messages_field() {
    let mut app = ready_app();
    app.tokenizer = Some(PromptTokenizer::new(FIXTURE, Some(3)));
    let (status, json) = post(app, &chat(&format!("{SECRET} hello world hello world"))).await;
    assert_eq!(status, 400);
    assert_shape(&json, "context_length_exceeded");
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["param"], "messages");
    assert!(json["error"]["message"].as_str().unwrap().contains("limit is 3"), "{json}");
}

#[tokio::test]
async fn unreadable_tokenizer_file_is_422_without_its_path() {
    let mut app = ready_app();
    app.tokenizer = Some(PromptTokenizer::new("tests/fixtures/no-such-dir/tokenizer.json", None));
    let (status, json) = post(app, &chat("hello")).await;
    assert_eq!(status, 422);
    assert_shape(&json, "tokenize_failed");
    assert!(!json.to_string().contains("no-such-dir"), "path leaked: {json}");
}

#[tokio::test]
async fn missing_scheduler_is_503() {
    let mut app = ready_app();
    app.tokenizer = Some(PromptTokenizer::new(FIXTURE, None));
    let (status, json) = post(app, &chat("hello world")).await;
    assert_eq!(status, 503);
    assert_shape(&json, "scheduler_not_wired");
}

#[tokio::test]
async fn draining_engine_is_503_engine_not_serving() {
    let sm = StateMachine::new();
    sm.mark_ready().unwrap();
    sm.drain_for_shutdown().unwrap();
    let (status, json) = post(AppState::from(Arc::new(sm)), &chat("hello")).await;
    assert_eq!(status, 503);
    assert_shape(&json, "engine_not_serving");
    assert!(json["error"]["message"].as_str().unwrap().contains("Draining"), "{json}");
}

#[test]
fn every_variant_has_a_distinct_code_a_status_and_a_fix() {
    let all = [
        ApiError::EngineNotServing { status: "Stopped".into() },
        ApiError::NoTokenizer,
        ApiError::PromptTooLong { tokens: 9, max: 3 },
        ApiError::TokenizeFailed,
        ApiError::EmptyPrompt,
        ApiError::Scheduler(SchedulerFault::NotWired),
        ApiError::Scheduler(SchedulerFault::QueueClosed),
        ApiError::Scheduler(SchedulerFault::Overloaded),
        ApiError::Scheduler(SchedulerFault::ClosedEarly),
        ApiError::FirstEventTimeout,
        ApiError::GenerationFailed,
        ApiError::DecodeFailed,
    ];
    let mut codes: Vec<_> = all.iter().map(|e| e.code()).collect();
    codes.sort_unstable();
    codes.dedup();
    assert_eq!(codes.len(), all.len(), "codes must be unique");
    for e in &all {
        assert!(!e.fix().is_empty() && !e.to_string().is_empty());
        assert!(e.status().as_u16() >= 400, "{e}");
        assert_eq!(e.body()["error"]["code"], e.code());
    }
    assert_eq!(ApiError::FirstEventTimeout.status().as_u16(), 504);
    // A full queue is the client cue to back off: 429 with the OpenAI rate-limit shape.
    let full = ApiError::Scheduler(SchedulerFault::Overloaded);
    assert_eq!(full.status().as_u16(), 429);
    assert_eq!(full.body()["error"]["type"], "rate_limit_error");
    assert_eq!(full.body()["error"]["code"], "rate_limit_exceeded");
}
