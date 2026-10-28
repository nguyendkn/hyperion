//! Real tokenizers of every model in `testdata/models.json`, generic over the
//! registry (no model is named here). The files are not committed: fetch them
//! with `python scripts/fetch-model-fixtures.py`. A model whose files are
//! missing is skipped with a visible message, unless
//! `HYPERION_REQUIRE_MODEL_FIXTURES=1`, which turns that into a failure.
//!
//! Origin: `tokenizers` 0.19.1 could not parse a current model's
//! `tokenizer.json` ("data did not match any variant of untagged enum
//! ModelWrapper"), which blocked every end-to-end path for that family.

use std::path::PathBuf;

use helios_core::tokenizer::{ChatMessageForEncode, PromptTokenizer};
use serde_json::Value;

fn testdata() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../testdata")
}

/// Registry ids, paired with the model directory when its files are present.
fn models() -> Vec<(String, Option<PathBuf>)> {
    let registry: Value = serde_json::from_slice(&std::fs::read(testdata().join("models.json")).unwrap()).unwrap();
    registry["models"]
        .as_object()
        .unwrap()
        .keys()
        .map(|id| {
            let dir = testdata().join("models").join(id);
            let present = dir.join("tokenizer.json").is_file() && dir.join("config.json").is_file();
            (id.clone(), present.then_some(dir))
        })
        .collect()
}

fn require() -> bool {
    std::env::var_os("HYPERION_REQUIRE_MODEL_FIXTURES").is_some()
}

async fn encode(dir: &PathBuf, text: &str) -> Vec<u32> {
    let tok = PromptTokenizer::new(dir.join("tokenizer.json").to_string_lossy().into_owned(), Some(40_960));
    tok.encode_messages(&[ChatMessageForEncode { role: "user", content: text }])
        .await
        .expect("the real tokenizer.json must load and encode")
}

/// Run `check` for every model that has files; skip (or fail) the rest.
async fn for_each_model<F, Fut>(check: F)
where
    F: Fn(String, PathBuf) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    for (id, dir) in models() {
        match dir {
            Some(dir) => check(id, dir).await,
            None if require() => panic!("{id}: files missing; run `python scripts/fetch-model-fixtures.py {id}`"),
            None => eprintln!("SKIPPED {id}: files missing; run `python scripts/fetch-model-fixtures.py {id}`"),
        }
    }
}

fn vocab_size(dir: &PathBuf) -> u32 {
    let cfg: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    cfg["vocab_size"].as_u64().expect("config.json has vocab_size") as u32
}

#[tokio::test]
async fn multilingual_prompt_encodes_within_the_vocabulary() {
    for_each_model(|id, dir| async move {
        let ids = encode(&dir, "Xin chào, 你好, hello!").await;
        assert!(ids.len() >= 5, "{id}: expected several tokens, got {ids:?}");
        let vocab = vocab_size(&dir);
        assert!(ids.iter().all(|&t| t < vocab), "{id}: id outside vocab {vocab}: {ids:?}");
    })
    .await;
}

#[tokio::test]
async fn encoding_is_deterministic() {
    for_each_model(|id, dir| async move {
        assert_eq!(encode(&dir, "hello world").await, encode(&dir, "hello world").await, "{id}");
    })
    .await;
}

#[tokio::test]
async fn an_overlong_prompt_is_rejected_not_truncated() {
    for_each_model(|id, dir| async move {
        let text = "hello world ".repeat(2_000);
        let tok = PromptTokenizer::new(dir.join("tokenizer.json").to_string_lossy().into_owned(), Some(100));
        let err = tok
            .encode_messages(&[ChatMessageForEncode { role: "user", content: &text }])
            .await
            .expect_err(&format!("{id}: a prompt far over the limit must be refused"));
        let too_long = err.downcast_ref::<helios_core::tokenizer::PromptTooLong>().expect("typed error");
        assert!(too_long.tokens > 100 && too_long.max == 100, "{id}: {too_long}");
    })
    .await;
}
