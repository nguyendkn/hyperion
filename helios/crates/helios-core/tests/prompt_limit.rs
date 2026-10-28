//! A prompt longer than the context limit is an error the client can act on
//! (HTTP 400), never silently truncated. Found on an H100 pod: a ~60k-token
//! prompt against a 40960-token limit was cut and accepted.

use std::sync::Arc;

use helios_core::api::{router, AppState};
use helios_core::tokenizer::{ChatMessageForEncode, PromptTooLong, PromptTokenizer};
use helios_core::StateMachine;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const FIXTURE: &str = "tests/fixtures/dummy_tokenizer.json";

fn msg(content: &str) -> [ChatMessageForEncode<'_>; 1] {
    [ChatMessageForEncode { role: "user", content }]
}

#[tokio::test]
async fn a_prompt_over_the_limit_is_an_error_not_a_truncation() {
    let unlimited = PromptTokenizer::new(FIXTURE, Some(1000));
    let full = unlimited.encode_messages(&msg("hello world hello world hello world")).await.unwrap();
    assert!(full.len() > 3, "fixture must yield more than 3 tokens: {full:?}");

    let limited = PromptTokenizer::new(FIXTURE, Some(3));
    let err = limited.encode_messages(&msg("hello world hello world hello world")).await.unwrap_err();
    let too_long = err.downcast_ref::<PromptTooLong>().expect("typed PromptTooLong error");
    assert_eq!((too_long.tokens, too_long.max), (full.len(), 3));
    assert!(err.to_string().contains("limit is 3"), "{err}");
}

#[tokio::test]
async fn a_prompt_at_the_limit_is_accepted_whole() {
    let full = PromptTokenizer::new(FIXTURE, Some(1000)).encode_messages(&msg("hello world")).await.unwrap();
    let exact = PromptTokenizer::new(FIXTURE, Some(full.len())).encode_messages(&msg("hello world")).await.unwrap();
    assert_eq!(exact, full);
}

#[tokio::test]
async fn http_returns_400_with_the_limit_for_an_overlong_prompt() {
    let sm = StateMachine::new();
    sm.mark_ready().unwrap();
    let mut app = AppState::from(Arc::new(sm));
    app.tokenizer = Some(PromptTokenizer::new(FIXTURE, Some(3)));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let service = router().with_state(app);
    tokio::spawn(async move { axum::serve(listener, service).await.unwrap() });

    let body = r#"{"model":"m","messages":[{"role":"user","content":"hello world hello world hello world"}]}"#;
    let mut s = TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut raw = String::new();
    s.read_to_string(&mut raw).await.unwrap();
    assert_eq!(raw.split_whitespace().nth(1), Some("400"), "{raw}");
    assert!(raw.contains("limit is 3"), "{raw}");
}
