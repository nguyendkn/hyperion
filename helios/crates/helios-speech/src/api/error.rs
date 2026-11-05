use serde_json::{json, Value};

/// An API refusal in the OpenAI-shaped body of the LLM endpoints: `{"error": {"message", "type", "param", "code"}}`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub param: Option<String>,
    pub code: &'static str,
}

impl ApiError {
    pub fn bad_request(code: &'static str, param: Option<&str>, message: impl Into<String>) -> Self {
        Self { status: 400, message: message.into(), param: param.map(str::to_string), code }
    }
    pub fn unsupported_field(name: &str) -> Self {
        Self::bad_request("unsupported_parameter", Some(name), format!("parameter `{name}` is not supported"))
    }
    /// 429 naming the model whose queue is full.
    pub fn queue_full(model: &str) -> Self {
        Self { status: 429, message: format!("the queue of model `{model}` is full, retry later"), param: Some("model".into()), code: "model_queue_full" }
    }
    pub fn body(&self) -> Value {
        let kind = if self.status >= 500 { "server_error" } else if self.status == 429 { "rate_limit_error" } else { "invalid_request_error" };
        json!({ "error": { "message": self.message, "type": kind, "param": self.param, "code": self.code } })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn body_shape() {
        let b = ApiError::unsupported_field("prompt").body();
        assert_eq!(b["error"]["type"], "invalid_request_error");
        assert_eq!(b["error"]["param"], "prompt");
        assert_eq!(ApiError::queue_full("tts").body()["error"]["type"], "rate_limit_error");
    }
}
