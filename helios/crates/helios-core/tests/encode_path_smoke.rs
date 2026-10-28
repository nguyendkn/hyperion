//! Smoke test for the encode path: `ChatMessageForEncode[]` → flatten →
//! HuggingFace `tokenizers` encode → `Vec<u32>`. This is the vertical-slice
//! verification that the API→tokenizer seam produces a well-formed token-id
//! stream *before* the scheduler seam (`InferenceRequest`) consumes it.
//!
//! Uses a minimal WordLevel tokenizer fixture (`fixtures/dummy_tokenizer.json`)
//! rather than a multi-GB Llama/Qwen model — YAGNI: we only need to prove the
//! encode machinery is sound (OnceCell lazy load, flatten join, special-token
//! recognition, idempotency), not that we can tokenize real prose. A real
//! `tokenizer.json` lands once `helios-pkg` carries it from a `.hyperion`
//! package; until then this fixture stands in.

use helios_core::tokenizer::{ChatMessageForEncode, PromptTokenizer};

/// Path to the dummy tokenizer fixture, resolved relative to the crate root
/// so the test works regardless of the cwd `cargo test` is invoked from.
const FIXTURE: &str = "tests/fixtures/dummy_tokenizer.json";

/// Build a `PromptTokenizer` over the fixture with the default context cap.
/// `unwrap` is fine here — if the fixture won't load, every test should fail
/// loudly with the loader error, not a vague downstream assertion.
fn fixture_tokenizer() -> PromptTokenizer {
    PromptTokenizer::new(FIXTURE, None)
}

/// Helper: encode a slice of (role, content) pairs through the fixture.
async fn encode(pairs: &[(&str, &str)]) -> Vec<u32> {
    let tok = fixture_tokenizer();
    let msgs: Vec<ChatMessageForEncode<'_>> = pairs
        .iter()
        .map(|(r, c)| ChatMessageForEncode { role: r, content: c })
        .collect();
    tok.encode_messages(&msgs)
        .await
        .expect("encode should succeed against a valid fixture")
}

#[tokio::test]
async fn single_message_encodes_to_nonempty_ids() {
    let ids = encode(&[("user", "hello world")]).await;
    assert!(!ids.is_empty(), "encode must produce at least one token id");
}

#[tokio::test]
async fn flatten_join_is_observed_by_encoder() {
    // flatten("user","a") = "user:a"; flatten of two msgs joins with "\n".
    // The encoder must see a *different* id stream for one vs two messages —
    // this proves the flatten step (in tokenizer.rs) actually shapes the
    // input and is not bypassed.
    let one = encode(&[("user", "hello world")]).await;
    let two = encode(&[
        ("system", "you are helpful"),
        ("user", "hello world"),
    ])
    .await;
    assert_ne!(
        one, two,
        "multi-message flatten must produce a distinct token stream"
    );
}

#[tokio::test]
async fn role_prefix_becomes_part_of_input() {
    // `user:hello` does not split on ':' under the Whitespace pre-tokenizer,
    // so it falls through to <unk> (id 16). A bare `hello world` (no role
    // prefix) tokenizes to [hello=6, world=7]. The two must differ, proving
    // the role prefix is part of the string the tokenizer sees — not stripped
    // or swallowed by the wrapper.
    let with_role = encode(&[("user", "hello world")]).await;
    // Encode the same content with an empty-ish role so the prefix cannot
    // accidentally match a vocab entry.
    let bare = {
        let tok = fixture_tokenizer();
        // Flatten with empty role yields ":hello world" → still differs.
        let msgs = [ChatMessageForEncode { role: "", content: "hello world" }];
        tok.encode_messages(&msgs).await.unwrap()
    };
    assert_ne!(
        with_role, bare,
        "role prefix must influence the token stream, not be a no-op"
    );
}

#[tokio::test]
async fn encode_is_idempotent_across_calls() {
    // The OnceCell should make the second encode reuse the same loaded
    // tokenizer instance — same input, same output. This also exercises the
    // lifetime fix in `get()` (borrow on `self`, not a dropped local Arc).
    let tok = fixture_tokenizer();
    let msgs = [ChatMessageForEncode { role: "user", content: "hi" }];
    let first = tok.encode_messages(&msgs).await.unwrap();
    let second = tok.encode_messages(&msgs).await.unwrap();
    assert_eq!(first, second, "repeated encode must be deterministic");
}

#[tokio::test]
async fn known_word_round_trips_to_expected_id() {
    // The fixture vocab maps `hello` → 6 and `world` → 7, and `<unk>` → 16.
    // flatten(role="", content="hello world") yields ":hello world"; the
    // Whitespace pre-tokenizer splits non-word runes off, so the stream is
    // `:`(<unk>=16), `hello`=6, `world`=7 → [16, 6, 7]. This pins the
    // fixture + encoder contract: there is no "raw" encode path (flatten
    // always prepends `role:`), so the colon is unavoidable and falls to
    // <unk>. A regression in the fixture vocab or the flatten step shows up
    // here.
    let tok = fixture_tokenizer();
    let msgs = [ChatMessageForEncode { role: "", content: "hello world" }];
    let ids = tok.encode_messages(&msgs).await.unwrap();
    assert_eq!(ids, vec![16, 6, 7], "':hello world' must map to [16, 6, 7]");
}

#[tokio::test]
async fn decode_recovers_text_for_known_ids() {
    // Round-trip: encode → ids → decode. Only asserts the decode path returns
    // *something* non-empty for ids the tokenizer itself produced — exact
    // text recovery is vocab/decoder dependent and not the point of this test.
    let tok = fixture_tokenizer();
    let msgs = [ChatMessageForEncode { role: "", content: "hello world" }];
    let ids = tok.encode_messages(&msgs).await.unwrap();
    let text = tok.decode(&ids).await.expect("decode must succeed for known ids");
    assert!(!text.is_empty(), "decode of encoded ids must be non-empty");
    assert!(
        text.contains("hello") && text.contains("world"),
        "decoded text must contain the source words; got {text:?}"
    );
}

#[tokio::test]
async fn empty_message_list_is_handled_gracefully() {
    // No messages → flatten returns "" → the encoder must not panic. This is
    // the boundary the API handler guards against (ids.is_empty() → 422), so
    // the tokenizer itself should return an empty (or error) rather than
    // crash. We assert no panic; the exact return is implementation-defined.
    let tok = fixture_tokenizer();
    let msgs: [ChatMessageForEncode<'_>; 0] = [];
    let result = tok.encode_messages(&msgs).await;
    // Either an empty Vec or an Err is acceptable; a panic is not.
    match result {
        Ok(ids) => assert!(ids.is_empty(), "empty input must yield empty ids"),
        Err(_) => { /* acceptable: encoder may reject empty input */ }
    }
}
