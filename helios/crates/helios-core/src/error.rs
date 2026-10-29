//! The API's one error type. Every failure a handler can return is a variant here, and the
//! status code, the OpenAI-shaped body and the machine-readable `fix` hint all come from this
//! file, so a new failure cannot forget one of them.
//!
//! Body: `{"error": {"message", "type", "param", "code", "fix"}}`. The first four are what OpenAI
//! clients parse; `fix` is additive (a client that does not know it ignores it). Messages never
//! carry prompt text, file paths or configuration: those go to the log, not to the client.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

/// Why the scheduler could not take or finish a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerFault {
    /// No scheduler is wired into the server.
    NotWired,
    /// The scheduler's receiver is gone (its task ended).
    QueueClosed,
    /// The bounded ingress queue stayed full past the send timeout.
    Overloaded,
    /// The scheduler closed the stream before the first event.
    ClosedEarly,
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("the engine is not serving (state: {status})")]
    EngineNotServing { status: String },
    #[error("no tokenizer is loaded: this package cannot serve chat requests")]
    NoTokenizer,
    #[error("the prompt has {tokens} tokens but the limit is {max}")]
    PromptTooLong { tokens: usize, max: usize },
    #[error("the prompt could not be tokenized")]
    TokenizeFailed,
    #[error("the prompt encoded to zero tokens")]
    EmptyPrompt,
    #[error("the scheduler cannot take the request ({0:?})")]
    Scheduler(SchedulerFault),
    #[error("the scheduler did not answer in time")]
    FirstEventTimeout,
    #[error("generation failed")]
    GenerationFailed,
    #[error("the generated tokens could not be decoded")]
    DecodeFailed,
}

impl ApiError {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::EngineNotServing { .. } | Self::Scheduler(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::NoTokenizer => StatusCode::NOT_IMPLEMENTED,
            Self::PromptTooLong { .. } => StatusCode::BAD_REQUEST,
            Self::TokenizeFailed | Self::EmptyPrompt => StatusCode::UNPROCESSABLE_ENTITY,
            Self::FirstEventTimeout => StatusCode::GATEWAY_TIMEOUT,
            Self::GenerationFailed | Self::DecodeFailed => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// OpenAI's `type`: `invalid_request_error` when the client can change the request,
    /// `server_error` otherwise.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::PromptTooLong { .. } | Self::TokenizeFailed | Self::EmptyPrompt => "invalid_request_error",
            _ => "server_error",
        }
    }

    /// Stable snake_case identifier clients can switch on.
    pub fn code(&self) -> &'static str {
        match self {
            Self::EngineNotServing { .. } => "engine_not_serving",
            Self::NoTokenizer => "no_tokenizer",
            Self::PromptTooLong { .. } => "context_length_exceeded",
            Self::TokenizeFailed => "tokenize_failed",
            Self::EmptyPrompt => "empty_prompt",
            Self::Scheduler(SchedulerFault::NotWired) => "scheduler_not_wired",
            Self::Scheduler(SchedulerFault::QueueClosed) => "scheduler_unavailable",
            Self::Scheduler(SchedulerFault::Overloaded) => "scheduler_overloaded",
            Self::Scheduler(SchedulerFault::ClosedEarly) => "scheduler_closed_stream",
            Self::FirstEventTimeout => "scheduler_timeout",
            Self::GenerationFailed => "generation_failed",
            Self::DecodeFailed => "decode_failed",
        }
    }

    /// The request field at fault, when there is one.
    pub fn param(&self) -> Option<&'static str> {
        match self {
            Self::PromptTooLong { .. } | Self::TokenizeFailed | Self::EmptyPrompt => Some("messages"),
            _ => None,
        }
    }

    /// What to do next, in one sentence.
    pub fn fix(&self) -> &'static str {
        match self {
            Self::EngineNotServing { .. } => "retry after the engine reports ready on GET /health",
            Self::NoTokenizer => "start the server with a package that ships tokenizer.json",
            Self::PromptTooLong { .. } => "shorten the messages, or start the server with a larger --max-seq-len",
            Self::TokenizeFailed => "check that the message contents are valid text",
            Self::EmptyPrompt => "send at least one message with content",
            Self::Scheduler(SchedulerFault::Overloaded) => "retry with backoff; the server is at capacity",
            Self::Scheduler(_) | Self::FirstEventTimeout => "retry later; if it persists, check the server log",
            Self::GenerationFailed | Self::DecodeFailed => "retry; if it persists, check the server log",
        }
    }

    /// The JSON body, separate from the response so tests can inspect it without a server.
    pub fn body(&self) -> serde_json::Value {
        serde_json::json!({ "error": {
            "message": self.to_string(),
            "type": self.kind(),
            "param": self.param(),
            "code": self.code(),
            "fix": self.fix(),
        }})
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status(), axum::Json(self.body())).into_response()
    }
}
