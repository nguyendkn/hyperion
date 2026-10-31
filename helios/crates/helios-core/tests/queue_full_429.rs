//! A full scheduler queue is a 429 in the OpenAI shape (`rate_limit_exceeded`), not a 503: the client is told to back off and retry.

use std::sync::Arc;

use helios_core::api::{router, AppState};
use helios_core::scheduler::{ingress_channel, InferenceRequest, SamplingParams, INGRESS_CAPACITY};
use helios_core::tokenizer::PromptTokenizer;
use helios_core::StateMachine;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[tokio::test]
async fn a_full_scheduler_queue_is_429_in_the_openai_shape() {
    // Nobody reads the ingress, so after INGRESS_CAPACITY requests the next send waits and times out.
    let (tx, _rx) = ingress_channel();
    for id in 0..INGRESS_CAPACITY as u64 {
        let (response_tx, _response_rx) = tokio::sync::mpsc::unbounded_channel();
        let request = InferenceRequest { request_id: id, token_ids: vec![1], sampling_params: SamplingParams { temperature: None, top_p: None, max_tokens: 1 }, response_tx };
        tx.send(request).await.unwrap();
    }
    let sm = StateMachine::new();
    sm.mark_ready().unwrap();
    let mut app = AppState::from(Arc::new(sm));
    app.tokenizer = Some(PromptTokenizer::new("tests/fixtures/dummy_tokenizer.json", None));
    app.ingress_tx = Some(tx);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let service = router().with_state(app);
    tokio::spawn(async move { axum::serve(listener, service).await.unwrap() });
    let body = serde_json::json!({ "model": "m", "messages": [{ "role": "user", "content": "hello world" }] }).to_string();
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let head = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len());
    stream.write_all(head.as_bytes()).await.unwrap();
    let mut raw = String::new();
    stream.read_to_string(&mut raw).await.unwrap();
    let status: u16 = raw.split_whitespace().nth(1).unwrap().parse().unwrap();
    let json: serde_json::Value = serde_json::from_str(raw.split("\r\n\r\n").nth(1).unwrap().trim()).unwrap();
    assert_eq!(status, 429, "{json}");
    assert_eq!(json["error"]["code"], "rate_limit_exceeded");
    assert_eq!(json["error"]["type"], "rate_limit_error");
    assert!(json["error"]["fix"].as_str().is_some_and(|f| !f.is_empty()));
}
