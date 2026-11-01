//! A request whose estimated time to its first token exceeds `--first-token-timeout` is answered 429 at once (nothing reaches the scheduler);
//! one whose estimate is below it is queued and counted in the gauge.

use std::sync::Arc;
use std::time::Duration;

use helios_core::api::{router, AppState};
use helios_core::load_gauge::LoadGauge;
use helios_core::scheduler::ingress_channel;
use helios_core::tokenizer::PromptTokenizer;
use helios_core::StateMachine;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn post(addr: std::net::SocketAddr) -> (u16, serde_json::Value) {
    let body = serde_json::json!({ "model": "m", "messages": [{ "role": "user", "content": "hello world" }] }).to_string();
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let head = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len());
    stream.write_all(head.as_bytes()).await.unwrap();
    let mut raw = String::new();
    stream.read_to_string(&mut raw).await.unwrap();
    let status: u16 = raw.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, serde_json::from_str(raw.split("\r\n\r\n").nth(1).unwrap().trim()).unwrap())
}

fn app(gauge: Arc<LoadGauge>, timeout: Duration) -> (AppState, tokio::sync::mpsc::Receiver<helios_core::scheduler::InferenceRequest>) {
    let (tx, rx) = ingress_channel();
    let sm = StateMachine::new();
    sm.mark_ready().unwrap();
    let mut app = AppState::from(Arc::new(sm));
    app.tokenizer = Some(PromptTokenizer::new("tests/fixtures/dummy_tokenizer.json", None));
    app.ingress_tx = Some(tx);
    app.gauge = gauge;
    app.first_token_timeout = timeout;
    (app, rx)
}

async fn serve(app: AppState) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let service = router().with_state(app);
    tokio::spawn(async move { axum::serve(listener, service).await.unwrap() });
    addr
}

#[tokio::test]
async fn a_request_that_cannot_reach_its_first_token_in_time_is_refused_at_once() {
    let gauge = Arc::new(LoadGauge::new());
    gauge.record_prefill(1, Duration::from_secs(10), Duration::ZERO); // 10 s per prompt token: any prompt waits longer than the 1 s timeout
    let (app, mut rx) = app(gauge.clone(), Duration::from_secs(1));
    let addr = serve(app).await;
    let started = std::time::Instant::now();
    let (status, json) = post(addr).await;
    assert_eq!(status, 429, "{json}");
    assert_eq!(json["error"]["code"], "rate_limit_exceeded");
    assert!(started.elapsed() < Duration::from_millis(900), "refused before the timeout, not after it: {:?}", started.elapsed());
    assert!(rx.try_recv().is_err(), "a refused request never reaches the scheduler");
    assert_eq!(gauge.queued_tokens(), 0, "and is not counted as waiting");
}

#[tokio::test]
async fn a_request_that_fits_the_timeout_is_queued_and_counted() {
    let gauge = Arc::new(LoadGauge::new());
    gauge.record_prefill(1, Duration::from_micros(1), Duration::ZERO);
    let (app, mut rx) = app(gauge.clone(), Duration::from_secs(1));
    let addr = serve(app).await;
    let client = tokio::spawn(post(addr));
    let request = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.expect("the request reached the ingress").expect("an open channel");
    assert_eq!(gauge.queued_tokens(), request.token_ids.len() as u64, "its prompt tokens wait for prefill until the scheduler settles them");
    // Nobody answers, and the request (with its response channel) stays alive: the handler ends with the first-token timeout (504).
    let (status, _) = client.await.unwrap();
    assert_eq!(status, 504);
    drop(request);
}
