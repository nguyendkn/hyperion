//! Incremental detokenisation for streaming. Feeding the generated ids one at a time must give
//! text deltas that (a) never contain a half-decoded character and (b) concatenate to exactly the
//! text of decoding all the ids at once. Byte-level tokenizers split multi-byte characters (emoji,
//! CJK) across tokens, which is what makes this non-trivial. Generic over the model registry
//! (no model is named); missing files skip unless `HYPERION_REQUIRE_MODEL_FIXTURES` is set.

use std::path::PathBuf;

use helios_core::tokenizer::{ChatMessageForEncode, PromptTokenizer, StreamDecoder};
use serde_json::Value;

fn testdata() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../testdata")
}

/// Model tokenizers present on this machine.
fn tokenizers() -> Vec<(String, PathBuf)> {
    let registry: Value = serde_json::from_slice(&std::fs::read(testdata().join("models.json")).unwrap()).unwrap();
    let mut out = Vec::new();
    for id in registry["models"].as_object().unwrap().keys() {
        let path = testdata().join("models").join(id).join("tokenizer.json");
        if path.is_file() {
            out.push((id.clone(), path));
        } else if std::env::var_os("HYPERION_REQUIRE_MODEL_FIXTURES").is_some() {
            panic!("{id}: run `python scripts/fetch-model-fixtures.py {id}`");
        } else {
            eprintln!("SKIPPED {id}: tokenizer.json missing");
        }
    }
    out
}

/// Token ids of `text` as the tokenizer produces them (no chat template, no specials).
async fn ids_of(tok: &PromptTokenizer, text: &str) -> Vec<u32> {
    // `encode_messages` prepends `role:`; strip that prefix's tokens by encoding it alone.
    let with = tok.encode_messages(&[ChatMessageForEncode { role: "x", content: text }]).await.unwrap();
    let prefix = tok.encode_messages(&[ChatMessageForEncode { role: "x", content: "" }]).await.unwrap();
    with[prefix.len()..].to_vec()
}

#[tokio::test]
async fn deltas_never_split_a_character_and_add_up_to_the_full_text() {
    let samples = ["Xin chào, 你好, hello!", "A smiling face 🙂 and a rocket 🚀 end.", "日本語のテキスト、そして émojis 😀😀 ok", "plain ascii only"];
    for (id, path) in tokenizers() {
        let tok = PromptTokenizer::new(path.to_string_lossy().into_owned(), Some(100_000));
        for text in samples {
            let ids = ids_of(&tok, text).await;
            let full = tok.decode(&ids).await.unwrap();
            let mut decoder = StreamDecoder::new();
            let mut joined = String::new();
            let mut held_back = 0;
            for &t in &ids {
                let delta = decoder.push(&tok, t).await.unwrap();
                assert!(!delta.contains('\u{FFFD}'), "{id}: a delta contains a half-decoded character for {text:?}: {delta:?}");
                if delta.is_empty() {
                    held_back += 1;
                }
                joined.push_str(&delta);
            }
            joined.push_str(&decoder.finish(&tok).await.unwrap());
            assert_eq!(joined, full, "{id}: streamed text differs from the full decode for {text:?}");
            println!("{id}: {text:?} -> {} tokens, {held_back} held back, text identical", ids.len());
        }
    }
}

#[tokio::test]
async fn an_empty_stream_gives_empty_text() {
    for (_, path) in tokenizers() {
        let tok = PromptTokenizer::new(path.to_string_lossy().into_owned(), Some(1000));
        assert_eq!(StreamDecoder::new().finish(&tok).await.unwrap(), "");
    }
}
