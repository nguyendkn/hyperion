//! The shape of the OpenAI streaming chunks. The success path of the stream itself needs a
//! backend that can generate and is exercised on the GPU (docs/unverified-ledger.md).

use helios_core::api::{chunk_json, ChatCompletionRequest};
use serde_json::{json, Value};

fn parse(s: &str) -> Value {
    serde_json::from_str(s).expect("a chunk is JSON")
}

#[test]
fn a_content_chunk_has_the_openai_shape() {
    let chunk = parse(&chunk_json("chatcmpl-1", "m", json!({ "content": "Hi" }), None));
    assert_eq!(chunk["object"], "chat.completion.chunk");
    assert_eq!(chunk["id"], "chatcmpl-1");
    assert_eq!(chunk["model"], "m");
    assert_eq!(chunk["choices"][0]["index"], 0);
    assert_eq!(chunk["choices"][0]["delta"]["content"], "Hi");
    assert!(chunk["choices"][0]["finish_reason"].is_null(), "no finish reason until the last chunk");
}

#[test]
fn the_last_chunk_carries_the_finish_reason_and_an_empty_delta() {
    let chunk = parse(&chunk_json("c", "m", json!({}), Some("length")));
    assert_eq!(chunk["choices"][0]["finish_reason"], "length");
    assert_eq!(chunk["choices"][0]["delta"], json!({}));
}

#[test]
fn the_first_chunk_announces_the_assistant_role() {
    let chunk = parse(&chunk_json("c", "m", json!({ "role": "assistant", "content": "" }), None));
    assert_eq!(chunk["choices"][0]["delta"]["role"], "assistant");
}

#[test]
fn text_with_quotes_newlines_and_unicode_survives_the_json_encoding() {
    let text = "say \"hi\"\n你好 🙂";
    let chunk = parse(&chunk_json("c", "m", json!({ "content": text }), None));
    assert_eq!(chunk["choices"][0]["delta"]["content"], text);
}

#[test]
fn stream_defaults_to_false_and_can_be_enabled() {
    let off: ChatCompletionRequest = serde_json::from_str(r#"{"model":"m","messages":[]}"#).unwrap();
    assert!(!off.stream);
    let on: ChatCompletionRequest = serde_json::from_str(r#"{"model":"m","messages":[],"stream":true}"#).unwrap();
    assert!(on.stream);
}
