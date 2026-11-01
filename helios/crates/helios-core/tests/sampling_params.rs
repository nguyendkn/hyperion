//! How a request's sampling fields become the sampling a sequence runs with: the request wins, then the manifest defaults, then the OpenAI defaults;
//! out-of-range and unimplemented fields are a 400 that names the parameter.

use helios_core::api::{resolve_sampling, ChatCompletionRequest, RequestDefaults};
use helios_core::error::ApiError;

fn request(extra: serde_json::Value) -> ChatCompletionRequest {
    let mut body = serde_json::json!({ "model": "m", "messages": [{ "role": "user", "content": "hi" }] });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    serde_json::from_value(body).unwrap()
}

fn invalid(extra: serde_json::Value, defaults: &RequestDefaults) -> (&'static str, String) {
    match resolve_sampling(&request(extra), defaults, 8) {
        Err(ApiError::InvalidParameter { name, reason }) => (name, reason.to_string()),
        other => panic!("expected a 400, got {other:?}"),
    }
}

#[test]
fn an_omitted_field_takes_the_manifest_default_and_without_one_the_openai_default() {
    let openai = resolve_sampling(&request(serde_json::json!({})), &RequestDefaults::default(), 8).unwrap();
    assert_eq!((openai.temperature, openai.top_p, openai.top_k, openai.repetition_penalty, openai.seed), (Some(1.0), Some(1.0), Some(0), Some(1.0), None));
    let manifest = RequestDefaults { temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.05 };
    let got = resolve_sampling(&request(serde_json::json!({})), &manifest, 8).unwrap();
    assert_eq!((got.temperature, got.top_p, got.top_k, got.repetition_penalty), (Some(0.7), Some(0.8), Some(20), Some(1.05)));
    let own = resolve_sampling(&request(serde_json::json!({ "temperature": 0.0, "top_p": 0.5, "top_k": 3, "seed": 99 })), &manifest, 8).unwrap();
    assert_eq!((own.temperature, own.top_p, own.top_k, own.seed, own.max_tokens), (Some(0.0), Some(0.5), Some(3), Some(99), 8));
    assert_eq!(own.repetition_penalty, Some(1.0), "greedy drops the penalty the manifest suggests (only a penalty the client sends is refused)");
    let warm = resolve_sampling(&request(serde_json::json!({ "temperature": 0.5 })), &manifest, 8).unwrap();
    assert_eq!(warm.repetition_penalty, Some(1.05), "a field the request omits still comes from the manifest when it samples");
}

#[test]
fn out_of_range_and_unimplemented_fields_are_refused_naming_the_parameter() {
    let d = RequestDefaults::default();
    assert_eq!(invalid(serde_json::json!({ "temperature": 2.5 }), &d).0, "temperature");
    assert_eq!(invalid(serde_json::json!({ "temperature": -0.1 }), &d).0, "temperature");
    assert_eq!(invalid(serde_json::json!({ "top_p": 1.5 }), &d).0, "top_p");
    assert_eq!(invalid(serde_json::json!({ "repetition_penalty": 0.0 }), &d).0, "repetition_penalty");
    assert_eq!(invalid(serde_json::json!({ "temperature": 0, "repetition_penalty": 1.2 }), &d).0, "repetition_penalty");
    assert_eq!(invalid(serde_json::json!({ "presence_penalty": 0.5 }), &d).0, "presence_penalty");
    assert_eq!(invalid(serde_json::json!({ "frequency_penalty": -1.0 }), &d).0, "frequency_penalty");
    // Zero penalties are what a client that sends the field with its neutral value means.
    assert!(resolve_sampling(&request(serde_json::json!({ "presence_penalty": 0, "frequency_penalty": 0 })), &d, 8).is_ok());
    let error = ApiError::InvalidParameter { name: "top_p", reason: "must be between 0 and 1" };
    assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(error.body()["error"]["param"], "top_p");
    assert_eq!(error.body()["error"]["code"], "invalid_parameter");
}
