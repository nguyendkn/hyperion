//! HTTP API surface — Axum server exposing an OpenAI-compatible
//! `/v1/chat/completions`. The handler tokenizes the inbound prompt before
//! handing it to the scheduler; the tokenizer itself lives in
//! [`crate::tokenizer`] so the encode step is testable without a server.

use std::sync::Arc;

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, SchedulerFault};
use crate::inflight::InflightTracker;
use crate::scheduler::{InferenceRequest, IngressSender, SamplingParams, StreamEvent};
use crate::state::StateMachine;
use crate::tokenizer::{ChatMessageForEncode, PromptTokenizer, StreamDecoder};

/// How long the handler waits for the scheduler to *accept* a request (the
/// ingress send) before declaring 429. The ingress channel is bounded
/// ([`crate::scheduler::INGRESS_CAPACITY`]); a full queue under load makes
/// `send().await` block, and we surface that as a fast 429 rather than a
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
    /// Requests being served; the CLI waits on it before it stops the scheduler.
    pub inflight: InflightTracker,
    /// `max_tokens` when the client omits it, capped by the context left after the prompt (`--default-max-tokens`).
    pub default_max_tokens: u32,
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
            inflight: InflightTracker::new(),
            default_max_tokens: DEFAULT_MAX_TOKENS,
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
    /// Omitted: `min(context left after the prompt, AppState::default_max_tokens)`.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// `true`: answer as server-sent events, one chunk per piece of text.
    #[serde(default)]
    pub stream: bool,
    /// `{"include_usage": true}` adds a final chunk with the token counts to a streamed answer.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// The default of `--default-max-tokens`.
pub const DEFAULT_MAX_TOKENS: u32 = 1024;

/// OpenAI `usage`: prompt and completion tokens as the server counted them.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl Usage {
    pub fn new(prompt: usize, completion: usize) -> Self {
        Self { prompt_tokens: prompt as u32, completion_tokens: completion as u32, total_tokens: (prompt + completion) as u32 }
    }
}

/// The `max_tokens` a request runs with: its own value, else the configured default capped by the context that is left after the prompt
/// (at least one token: a prompt that fills the context was already refused by the tokenizer).
pub fn effective_max_tokens(requested: Option<u32>, default: u32, context: usize, prompt: usize) -> u32 {
    requested.unwrap_or_else(|| default.min(context.saturating_sub(prompt) as u32).max(1))
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
    pub usage: Usage,
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
    // Counted from here until the response (or the SSE stream task) is done, so a shutdown drain
    // waits for it. Dropped with the future if the client disconnects.
    let guard = app.inflight.enter();
    match serve_chat(&app, req, guard).await {
        Ok(response) => response,
        Err(e) => {
            tracing::warn!(code = e.code(), status = e.status().as_u16(), "chat request failed");
            e.into_response()
        }
    }
}

async fn serve_chat(
    app: &AppState,
    req: ChatCompletionRequest,
    guard: crate::inflight::InflightGuard,
) -> Result<axum::response::Response, ApiError> {
    // Engine must be alive to accept work. `Draining` (in-flight shutdown)
    // and `Stopped` reject new requests.
    let status = app.state.status();
    if matches!(
        status,
        crate::state::EngineStatus::Uninitialized
            | crate::state::EngineStatus::Draining
            | crate::state::EngineStatus::Stopped
    ) {
        return Err(ApiError::EngineNotServing { status: format!("{status:?}") });
    }

    // 1. Tokenize. No tokenizer configured -> the capability is not wired: 501.
    let tok = app.tokenizer.as_ref().ok_or(ApiError::NoTokenizer)?;

    let for_encode: Vec<ChatMessageForEncode<'_>> = req
        .messages
        .iter()
        .map(|m| ChatMessageForEncode {
            role: m.role.as_str(),
            content: m.content.as_str(),
        })
        .collect();

    let ids = tok.encode_messages(&for_encode).await.map_err(|e| {
        // Too long is the client's to fix (400); anything else is a tokenizer failure (422).
        // The cause goes to the log only: it may quote the prompt or a file path.
        match e.downcast_ref::<crate::tokenizer::PromptTooLong>() {
            Some(t) => ApiError::PromptTooLong { tokens: t.tokens, max: t.max },
            None => {
                tracing::warn!(error = %e, "tokenize failed");
                ApiError::TokenizeFailed
            }
        }
    })?;

    // The tokenizer rejects over-long prompts (400 above); a degenerate empty encode is refused
    // here too.
    if ids.is_empty() {
        return Err(ApiError::EmptyPrompt);
    }

    // 2. Build the response channel + the request envelope, then enqueue.
    let ingress_tx = app.ingress_tx.as_ref().ok_or(ApiError::Scheduler(SchedulerFault::NotWired))?;

    let (response_tx, mut response_rx) =
        tokio::sync::mpsc::unbounded_channel::<StreamEvent>();

    let request = InferenceRequest {
        request_id: chrono_like_id(),
        token_ids: ids.clone(),
        sampling_params: SamplingParams {
            temperature: req.temperature,
            top_p: req.top_p,
            max_tokens: effective_max_tokens(req.max_tokens, app.default_max_tokens, tok.max_length(), ids.len()),
        },
        response_tx,
    };

    match tokio::time::timeout(INGRESS_SEND_TIMEOUT, ingress_tx.send(request)).await {
        Ok(Ok(())) => { /* enqueued */ }
        // Scheduler dropped its receiver (task ended).
        Ok(Err(_send_err)) => return Err(ApiError::Scheduler(SchedulerFault::QueueClosed)),
        // Bounded ingress queue is full: backpressure surfaced as 429 (OpenAI-shaped rate_limit_exceeded).
        Err(_elapsed) => return Err(ApiError::Scheduler(SchedulerFault::Overloaded)),
    }

    // 3. Drain the response stream. First event gated on a liveness timeout;
    //    subsequent events read without a timeout (decode can run long). We
    //    accumulate token ids until a terminal `Done`/`Error` arrives or the
    //    sender is dropped (recv -> None).
    let first = match tokio::time::timeout(FIRST_EVENT_TIMEOUT, response_rx.recv()).await {
        Ok(Some(ev)) => ev,
        // Scheduler closed the channel before any event: it never started this request.
        Ok(None) => return Err(ApiError::Scheduler(SchedulerFault::ClosedEarly)),
        Err(_elapsed) => return Err(ApiError::FirstEventTimeout),
    };

    // Streaming: a failure before the first token is still a normal HTTP error; once text starts
    // flowing the status is already 200 and later errors travel inside the stream.
    if req.stream {
        if let StreamEvent::Error(msg) = &first {
            tracing::error!(error = %msg, "scheduler error before the first token");
            return Err(ApiError::GenerationFailed);
        }
        let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
        return Ok(stream_response(tok.clone(), req.model.clone(), first, response_rx, guard, ids.len(), include_usage).into_response());
    }

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
                tracing::error!(error = %msg, "scheduler error during generation");
                return Err(ApiError::GenerationFailed);
            }
            None => {
                // Channel closed without a terminal event: finish with whatever we have.
                finish_reason = "stop".into();
                break;
            }
        }
        current = response_rx.recv().await;
    }

    // 4. Shape the OpenAI response. Decode the generated ids back to text
    //    through the same tokenizer; if decode fails we surface the error.
    let content = tok.decode(&generated).await.map_err(|e| {
        tracing::error!(error = %e, "decode of generated tokens failed");
        ApiError::DecodeFailed
    })?;

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
        usage: Usage::new(ids.len(), generated.len()),
    };

    Ok((axum::http::StatusCode::OK, axum::Json(resp)).into_response())
}

/// One OpenAI `chat.completion.chunk` as the JSON text of an SSE `data:` line.
pub fn chunk_json(id: &str, model: &str, delta: serde_json::Value, finish_reason: Option<&str>) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
    })
    .to_string()
}

/// The closing chunk of `stream_options.include_usage`: no choices, the counts.
pub fn usage_chunk_json(id: &str, model: &str, usage: Usage) -> String {
    serde_json::json!({ "id": id, "object": "chat.completion.chunk", "model": model, "choices": [], "usage": usage }).to_string()
}

/// Answer with server-sent events: a role chunk, one content chunk per piece of decoded text, a
/// final chunk carrying `finish_reason`, then `data: [DONE]`. A task owns the scheduler's event
/// stream and the incremental detokeniser; if the client disconnects, its sends fail and the task
/// ends (the generation itself runs to its end, which is harmless but not yet cancelled).
fn stream_response(
    tok: PromptTokenizer,
    model: String,
    first: StreamEvent,
    mut events: tokio::sync::mpsc::UnboundedReceiver<StreamEvent>,
    guard: crate::inflight::InflightGuard,
    prompt_tokens: usize,
    include_usage: bool,
) -> impl IntoResponse {
    let (tx, out) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let id = format!("chatcmpl-{}", chrono_like_id());
    tokio::spawn(async move {
        // Held until the stream is finished, so a shutdown drain waits for it.
        let _guard = guard;
        let send = |data: String| tx.send(Ok(Event::default().data(data))).is_ok();
        if !send(chunk_json(&id, &model, serde_json::json!({ "role": "assistant", "content": "" }), None)) {
            return;
        }
        let mut decoder = StreamDecoder::new();
        let mut completion = 0usize;
        let mut finish = "stop".to_string();
        let mut current = Some(first);
        loop {
            match current.take() {
                Some(StreamEvent::Tokens(ids)) => {
                    completion += ids.len();
                    for token in ids {
                        match decoder.push(&tok, token).await {
                            Ok(text) if !text.is_empty() => {
                                if !send(chunk_json(&id, &model, serde_json::json!({ "content": text }), None)) {
                                    return;
                                }
                            }
                            Ok(_) => {}
                            Err(e) => {
                                tracing::error!(error = %e, "incremental decode failed");
                                send(ApiError::DecodeFailed.body().to_string());
                                return;
                            }
                        }
                    }
                }
                Some(StreamEvent::Done { finish_reason, .. }) => {
                    finish = match finish_reason {
                        crate::scheduler::FinishReason::Length => "length".into(),
                        crate::scheduler::FinishReason::Stop => "stop".into(),
                    };
                    break;
                }
                Some(StreamEvent::Error(message)) => {
                    tracing::error!(error = %message, "scheduler error mid-stream");
                    send(ApiError::GenerationFailed.body().to_string());
                    send("[DONE]".into());
                    return;
                }
                None => break,
            }
            current = events.recv().await;
        }
        if let Ok(rest) = decoder.finish(&tok).await {
            if !rest.is_empty() && !send(chunk_json(&id, &model, serde_json::json!({ "content": rest }), None)) {
                return;
            }
        }
        send(chunk_json(&id, &model, serde_json::json!({}), Some(&finish)));
        if include_usage {
            send(usage_chunk_json(&id, &model, Usage::new(prompt_tokens, completion)));
        }
        send("[DONE]".into());
    });
    Sse::new(tokio_stream::wrappers::UnboundedReceiverStream::new(out))
}

/// Cheap monotonic-ish id without pulling `chrono`. Uses a process-static
/// counter so ids are unique within one server run; collision-irrelevant for
/// the dev probe — real ids come from the scheduler once it's wired.
fn chrono_like_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    SEQ.fetch_add(1, Ordering::Relaxed) + 1
}
