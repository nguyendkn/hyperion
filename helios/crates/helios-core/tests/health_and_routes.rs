//! Public routes over a real socket: `/health` reflects the engine state, and
//! the chat endpoint lives at `/v1/chat/completions` (not at the old root path).

use std::sync::Arc;

use helios_core::api::{router, AppState};
use helios_core::StateMachine;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn serve(sm: StateMachine) -> std::net::SocketAddr {
    let app = router().with_state(AppState::from(Arc::new(sm)));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

/// Minimal HTTP/1.1 client: returns (status code, body).
async fn request(addr: std::net::SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut raw = String::new();
    s.read_to_string(&mut raw).await.unwrap();
    let code = raw.split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
    (code, body)
}

#[tokio::test]
async fn health_is_503_before_the_engine_is_ready() {
    let addr = serve(StateMachine::new()).await;
    let (code, body) = request(addr, "GET", "/health", "").await;
    assert_eq!(code, 503);
    assert!(body.contains("Uninitialized"), "{body}");
}

#[tokio::test]
async fn health_is_200_when_ready() {
    let sm = StateMachine::new();
    sm.mark_ready().unwrap();
    let (code, body) = request(serve(sm).await, "GET", "/health", "").await;
    assert_eq!(code, 200);
    assert!(body.contains("Ready"), "{body}");
}

#[tokio::test]
async fn health_is_503_while_draining_and_after_stop() {
    let sm = StateMachine::new();
    sm.mark_ready().unwrap();
    sm.drain().unwrap();
    let addr = serve(sm.clone()).await;
    assert_eq!(request(addr, "GET", "/health", "").await.0, 503);
    sm.shutdown().unwrap();
    assert_eq!(request(addr, "GET", "/health", "").await.0, 503);
}

#[tokio::test]
async fn chat_lives_under_v1_and_not_at_the_root() {
    let sm = StateMachine::new();
    sm.mark_ready().unwrap();
    let addr = serve(sm).await;
    let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
    // No tokenizer is wired: the handler answers 501, which proves the route exists.
    assert_eq!(request(addr, "POST", "/v1/chat/completions", body).await.0, 501);
    assert_eq!(request(addr, "POST", "/chat/completions", body).await.0, 404);
}
