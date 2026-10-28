//! Writes the token ids of a varied corpus to the file named by `HYPERION_DUMP_IDS`, for every
//! model tokenizer on this machine. Run once per regex backend of the `tokenizers` crate
//! (`onig` and `fancy-regex`) and diff the two files: any difference means the backends tokenise
//! differently and the cheaper one cannot replace the reference. Does nothing when the variable is
//! unset, so it is inert in normal test runs.

use std::path::PathBuf;

use helios_core::tokenizer::{ChatMessageForEncode, PromptTokenizer};
use serde_json::{json, Value};

const CORPUS: &[&str] = &[
    "The capital of France is",
    "Xin chào, 你好, hello! 🙂🚀",
    "日本語のテキスト、そして émojis 😀😀 ok",
    "  leading and trailing   spaces   ",
    "tabs\tand\nnewlines\r\nmixed \u{a0}nbsp",
    "fn main() { println!(\"{}\", 1 + 2 * 3); } // comment",
    "def f(x):\n    return [i**2 for i in range(x) if i % 2 == 0]\n",
    "https://example.com/path?query=1&b=2#frag user@example.org",
    "1234567890 3.14159 -42 1e-9 0xFF 1_000_000",
    "I'm you're they've we'll it's don't \"quoted\" 'single'",
    "ÀÉÎÕÜ ñ ß ø å — – … « » “ ”",
    "Привет, мир! مرحبا بالعالم שלום עולם नमस्ते दुनिया",
    "a\u{0301} e\u{0301} combining marks and zero\u{200b}width",
    "ALL CAPS and camelCase and snake_case and kebab-case",
    "",
    "\n\n\n",
    "<|im_start|>user\nhi<|im_end|>",
    "end.",
];

#[tokio::test]
async fn dump_ids_for_the_regex_backend_comparison() {
    let Some(out) = std::env::var_os("HYPERION_DUMP_IDS") else {
        return;
    };
    let testdata = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../testdata");
    let registry: Value = serde_json::from_slice(&std::fs::read(testdata.join("models.json")).unwrap()).unwrap();
    let mut all = serde_json::Map::new();
    for id in registry["models"].as_object().unwrap().keys() {
        let path = testdata.join("models").join(id).join("tokenizer.json");
        if !path.is_file() {
            continue;
        }
        let tok = PromptTokenizer::new(path.to_string_lossy().into_owned(), Some(1_000_000));
        // The prefix "x:" that `encode_messages` adds is part of every sample, identically for both backends.
        let mut rows = Vec::new();
        for text in CORPUS {
            let ids = tok.encode_messages(&[ChatMessageForEncode { role: "x", content: text }]).await.unwrap();
            rows.push(json!({ "text": text, "ids": ids }));
        }
        all.insert(id.clone(), Value::Array(rows));
    }
    std::fs::write(out, serde_json::to_string_pretty(&Value::Object(all)).unwrap()).unwrap();
}
