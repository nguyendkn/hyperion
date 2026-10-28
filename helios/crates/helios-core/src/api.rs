//! HTTP API surface — Axum server exposing an OpenAI-compatible
//! `/v1/chat/completions`. The handler tokenizes the inbound prompt before
//! handing it to the scheduler; the tokenizer itself lives in
//! [`crate::tokenizer`] so the encode step is testable without a server.

use std::sync::Arc;

use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};

use crate::scheduler::{InferenceRequest, IngressSender, SamplingParams, StreamEvent};
use crate::state::StateMachine;
use crate::tokenizer::{ChatMessageForEncode, PromptTokenizer};

/// How long the handler waits for the scheduler to *accept* a request (the
/// ingress send) before declaring 503. The ingress channel is bounded
/// ([`crate::scheduler::INGRESS_CAPACITY`]); a full queue under load makes
/// `send().await` block, and we surface that as a fast 503 rather than a
/// hung connection.
const INGRESS_SEND_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);

/// How long the handler waits for the *first* stream event from the
/// scheduler. Once the first event arrives we trust the scheduler is alive
/// for this request and drop the timeout on subsequent reads (a decode loop
/// can legitimately run long). The first-event latency is a liveness probe.
const FIRST_EVENT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Shared state threaded through every handler. Combines the engine lifecycle
/// (`StateMachine`), the tokenizer loaded from the package, and the ingress
/// sender into the scheduler queue.
///
/// `tokenizer` and `ingress_tx` are `Option` because a server can boot before
/// either is wired — a package without `tokenizer.json`, or a scheduler task
/// that hasn't been spawned yet. Each handler returns a clear capability error
/// (501 / 503) rather than panicking.
#[derive(Clone)]
pub struct AppState {
    pub state: Arc<StateMachine>,
    pub tokenizer: Option<PromptTokenizer>,
    pub ingress_tx: Option<IngressSender>,
}

impl From<Arc<StateMachine>> for AppState {
    /// Convenience for the CLI path that has a state machine but no tokenizer
    /// or scheduler yet — keeps early-bootstrap call sites working until the
    /// CLI wires both halves.
    fn from(sm: Arc<StateMachine>) -> Self {
        Self {
            state: sm,
            tokenizer: None,
            ingress_tx: None,
        }
    }
}

/// Top-level router, generic over the app's state handle.
///
/// - `GET /health` (root): liveness/readiness of the engine state machine.
/// - `POST /v1/chat/completions`: the OpenAI-compatible endpoint. The `/v1`
///   prefix is applied here so every caller (CLI, tests) gets the same public
///   path; `/v1/models` and friends belong to the serving phase.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/health", get(health))
        .nest("/v1", Router::new().route("/chat/completions", post(chat_completions)))
}

/// `200` while the engine can take work (`Ready`, `Loaded`, `Serving`), `503`
/// otherwise. The body names the state and nothing else: it must not leak
/// paths or configuration.
async fn health(axum::extract::State(app): axum::extract::State<AppState>) -> axum::response::Response {
    use crate::state::EngineStatus::{Loaded, Ready, Serving};
    let status = app.state.status();
    let code = if matches!(status, Ready | Loaded | Serving) {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    };
    (code, axum::Json(serde_json::json!({ "status": format!("{status:?}") }))).into_response()
}

/// Minimal OpenAI-compatible request body. Only the fields the runtime needs
/// to schedule are typed; everything else passes through.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
}

fn default_max_tokens() -> u32 {
    1024
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Handler. Tokenize → enqueue to the scheduler → drain the response stream
/// into a single (non-streaming) `ChatCompletionResponse`. Streaming SSE
/// output lands later; this rounds the whole generation into one payload so
/// the vertical slice is observable end-to-end.
async fn chat_completions(
    axum::extract::State(app): axum::extract::State<AppState>,
    axum::Json(req): axum::Json<ChatCompletionRequest>,
) -> axum::response::Response {
    // Engine must be alive to accept work. `Draining` (in-flight shutdown)
    // and `Stopped` reject new requests.
    let status = app.state.status();
    if matches!(
        status,
        crate::state::EngineStatus::Uninitialized
            | crate::state::EngineStatus::Draining
            | crate::state::EngineStatus::Stopped
    ) {
        return (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            format!("engine not serving (status={status:?})"),
        )
            .into_response();
    }

    // 1. Tokenize. No tokenizer configured → the package codec hasn't been
    //    wired yet; return 501 with a clear message rather than a panic.
    let Some(tok) = app.tokenizer.as_ref() else {
        return (
            axum::http::StatusCode::NOT_IMPLEMENTED,
            "no tokenizer loaded: helios-pkg codec not wired",
        )
            .into_response();
    };

    let for_encode: Vec<ChatMessageForEncode<'_>> = req
        .messages
        .iter()
        .map(|m| ChatMessageForEncode {
            role: m.role.as_str(),
            content: m.content.as_str(),
        })
        .collect();

    let ids = match tok.encode_messages(&for_encode).await {
        Ok(ids) => ids,
        Err(e) => {
            // Too long is the client's to fix (400); anything else is a
            // tokenizer failure (422).
            if let Some(too_long) = e.downcast_ref::<crate::tokenizer::PromptTooLong>() {
                return (axum::http::StatusCode::BAD_REQUEST, too_long.to_string()).into_response();
            }
            return (
                axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                format!("tokenize failed: {e}"),
            )
                .into_response();
        }
    };

    // Context-length guard. The tokenizer rejects over-long prompts (400
    // above); here we also refuse a degenerate empty encode (token file/empty prompt mismatch).
    if ids.is_empty() {
        return (
            axum::http::StatusCode::UNPROCESSABLE_ENTITY,
            "prompt encoded to zero tokens",
        )
            .into_response();
    }

    // 2. Build the response channel + the request envelope, then enqueue.
    //    No scheduler wired (ingress_tx is None) → 503 capability error.
    //    A failed/timeout send → 503 (queue full or scheduler gone).
    let Some(ingress_tx) = app.ingress_tx.as_ref() else {
        return (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "scheduler not wired: ingress channel absent",
        )
            .into_response();
    };

    let (response_tx, mut response_rx) =
        tokio::sync::mpsc::unbounded_channel::<StreamEvent>();

    let request = InferenceRequest {
        request_id: chrono_like_id(),
        token_ids: ids.clone(),
        sampling_params: SamplingParams {
            temperature: req.temperature,
            top_p: req.top_p,
            max_tokens: req.max_tokens,
        },
        response_tx,
    };

    match tokio::time::timeout(INGRESS_SEND_TIMEOUT, ingress_tx.send(request)).await {
        Ok(Ok(())) => { /* enqueued */ }
        Ok(Err(_send_err)) => {
            // Scheduler dropped its receiver (task ended) → fail fast.
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "scheduler unavailable: queue closed",
            )
                .into_response();
        }
        Err(_elapsed) => {
            // Bounded ingress queue is full — backpressure surfaced as 503.
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "scheduler overloaded: ingress send timed out",
            )
                .into_response();
        }
    }

    // 3. Drain the response stream. First event gated on a liveness timeout;
    //    subsequent events read without a timeout (decode can run long). We
    //    accumulate token ids until a terminal `Done`/`Error` arrives or the
    //    sender is dropped (recv → None).
    let first = match tokio::time::timeout(FIRST_EVENT_TIMEOUT, response_rx.recv()).await {
        Ok(Some(ev)) => ev,
        Ok(None) => {
            // Scheduler closed the channel before any event — treated as
            // 503; the scheduler didn't even start this request.
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "scheduler closed stream before first event",
            )
                .into_response();
        }
        Err(_elapsed) => {
            return (
                axum::http::StatusCode::GATEWAY_TIMEOUT,
                "scheduler first-event timeout",
            )
                .into_response();
        }
    };

    let mut generated: Vec<u32> = Vec::new();
    let mut finish_reason = String::from("stop");

    // Fold the first event, then keep reading until terminal/drop.
    let mut current = Some(first);
    loop {
        match current.take() {
            Some(StreamEvent::Tokens(ids)) => generated.extend_from_slice(&ids),
            Some(StreamEvent::Done { finish_reason: fr, .. }) => {
                finish_reason = match fr {
                    crate::scheduler::FinishReason::Length => "length".into(),
                    crate::scheduler::FinishReason::Stop => "stop".into(),
                };
                break;
            }
            Some(StreamEvent::Error(msg)) => {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("scheduler error: {msg}"),
                )
                    .into_response();
            }
            None => {
                // Channel closed without a terminal event — treat as
                // truncated, finish with whatever we have.
                finish_reason = "stop".into();
                break;
            }
        }
        current = response_rx.recv().await;
    }

    // 4. Shape the OpenAI response. Decode the generated ids back to text
    //    through the same tokenizer; if decode fails we surface the error.
    let content = match tok.decode(&generated).await {
        Ok(text) => text,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("decode generated tokens failed: {e}"),
            )
                .into_response();
        }
    };

    let resp = ChatCompletionResponse {
        id: format!("chatcmpl-{}", chrono_like_id()),
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content,
            },
            finish_reason,
        }],
    };

    (axum::http::StatusCode::OK, axum::Json(resp)).into_response()
}

/// Cheap monotonic-ish id without pulling `chrono`. Uses a process-static
/// counter so ids are unique within one server run; collision-irrelevant for
/// the dev probe — real ids come from the scheduler once it's wired.
fn chrono_like_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    SEQ.fetch_add(1, Ordering::Relaxed) + 1
}
