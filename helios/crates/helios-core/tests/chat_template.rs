//! The model's own chat template (`chat_template` in `tokenizer_config.json`) must render the
//! same text, and tokenise to the same ids, as Hugging Face's `apply_chat_template` for every
//! model in `testdata/models.json` that has a reference file (`scripts/gen-chat-reference.py`).
//! Generic over the registry: no model is named here. Missing files skip with a message unless
//! `HYPERION_REQUIRE_MODEL_FIXTURES` is set.

use std::path::PathBuf;

use helios_core::chat_template::ChatTemplate;
use helios_core::tokenizer::{ChatMessageForEncode, PromptTokenizer};
use serde_json::Value;

fn testdata() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../testdata")
}

fn require() -> bool {
    std::env::var_os("HYPERION_REQUIRE_MODEL_FIXTURES").is_some()
}

/// (model id, model dir, reference JSON) for every model that has all the files.
fn cases() -> Vec<(String, PathBuf, Value)> {
    let registry: Value = serde_json::from_slice(&std::fs::read(testdata().join("models.json")).unwrap()).unwrap();
    let mut out = Vec::new();
    for id in registry["models"].as_object().unwrap().keys() {
        let dir = testdata().join("models").join(id);
        let reference = testdata().join("oracle").join(format!("chat-{id}.json"));
        let has_config = dir.join("tokenizer_config.json").is_file() && dir.join("tokenizer.json").is_file();
        if !has_config || !reference.is_file() {
            // Only models that declare a tokenizer_config in the registry are expected to have it.
            let expected = registry["models"][id]["files"].get("tokenizer_config.json").is_some();
            if expected && require() {
                panic!("{id}: run `python scripts/fetch-model-fixtures.py {id}` and `python scripts/gen-chat-reference.py {id}`");
            }
            if expected {
                eprintln!("SKIPPED {id}: chat template files or reference missing (scripts/gen-chat-reference.py)");
            }
            continue;
        }
        out.push((id.clone(), dir, serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap()));
    }
    out
}

fn messages(case: &Value) -> Vec<(String, String)> {
    case["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| (m["role"].as_str().unwrap().to_string(), m["content"].as_str().unwrap().to_string()))
        .collect()
}

#[test]
fn rendered_text_equals_huggingface() {
    for (id, dir, reference) in cases() {
        let template = ChatTemplate::from_tokenizer_config(&std::fs::read_to_string(dir.join("tokenizer_config.json")).unwrap())
            .unwrap_or_else(|e| panic!("{id}: {e:#}"))
            .unwrap_or_else(|| panic!("{id}: tokenizer_config.json has no chat_template"));
        for case in reference["cases"].as_array().unwrap() {
            let msgs = messages(case);
            let view: Vec<ChatMessageForEncode> = msgs.iter().map(|(r, c)| ChatMessageForEncode { role: r, content: c }).collect();
            let text = template.render(&view, true).unwrap_or_else(|e| panic!("{id}: {e:#}"));
            assert_eq!(text, case["text"].as_str().unwrap(), "{id}: rendering differs for {msgs:?}");
        }
        println!("{id}: {} conversations render exactly like Hugging Face", reference["cases"].as_array().unwrap().len());
    }
}

#[tokio::test]
async fn token_ids_equal_huggingface() {
    for (id, dir, reference) in cases() {
        let template = ChatTemplate::from_tokenizer_config(&std::fs::read_to_string(dir.join("tokenizer_config.json")).unwrap())
            .unwrap()
            .unwrap();
        let tokenizer = PromptTokenizer::new(dir.join("tokenizer.json").to_string_lossy().into_owned(), Some(32_768))
            .with_chat_template(template);
        for case in reference["cases"].as_array().unwrap() {
            let msgs = messages(case);
            let view: Vec<ChatMessageForEncode> = msgs.iter().map(|(r, c)| ChatMessageForEncode { role: r, content: c }).collect();
            let ids = tokenizer.encode_messages(&view).await.unwrap_or_else(|e| panic!("{id}: {e:#}"));
            let want: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            assert_eq!(ids, want, "{id}: token ids differ for {msgs:?}");
        }
        println!("{id}: {} conversations tokenise to Hugging Face's ids", reference["cases"].as_array().unwrap().len());
    }
}

// ---- behaviour that needs no model files ----

fn config(template: &str) -> String {
    serde_json::json!({ "chat_template": template }).to_string()
}

#[test]
fn a_config_without_a_template_gives_none() {
    assert!(ChatTemplate::from_tokenizer_config("{}").unwrap().is_none());
}

#[test]
fn a_broken_template_is_an_error_not_a_panic() {
    assert!(ChatTemplate::from_tokenizer_config(&config("{% for x in %}")).is_err());
}

#[test]
fn the_template_sees_messages_and_the_generation_prompt_flag() {
    let t = ChatTemplate::from_tokenizer_config(&config(
        "{% for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}",
    ))
    .unwrap()
    .unwrap();
    let msgs = [ChatMessageForEncode { role: "user", content: "hi" }, ChatMessageForEncode { role: "assistant", content: "yo" }];
    assert_eq!(t.render(&msgs, true).unwrap(), "[user:hi][assistant:yo]<assistant>");
    assert_eq!(t.render(&msgs, false).unwrap(), "[user:hi][assistant:yo]");
}

#[test]
fn raise_exception_in_a_template_surfaces_as_an_error() {
    let t = ChatTemplate::from_tokenizer_config(&config("{{ raise_exception('roles must alternate') }}")).unwrap().unwrap();
    let err = t.render(&[], true).unwrap_err().to_string();
    assert!(err.contains("roles must alternate"), "{err}");
}
