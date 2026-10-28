//! Prompt tokenizer — wraps HuggingFace `tokenizers` (Rust core, Apache-2.0)
//! to turn an OpenAI-style `messages` array into the `Vec<u32>` token-id
//! stream the HAL consumes.
//!
//! Why a wrapper at all (vs. calling `tokenizers::Tokenizer` inline from
//! `api.rs`): the encode step has its own concerns — loading the tokenizer
//! file named by the manifest, setting a context-length truncation that
//! matches the model's `max_position_embeddings`, and flattening the
//! chat-template into a single prompt string. Keeping those here lets
//! `api.rs` stay focused on HTTP and lets the scheduler receive a
//! ready-to-prefill token slice.
//!
//! No BPE/WPM reimplementation: we delegate to the library, which handles
//! DeepSeek/TikToken/HF edge cases we have no business hand-rolling.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use tokenizers::Tokenizer;
use tokio::sync::OnceCell;

/// Context length cap applied at encode time. The real value comes from the
/// manifest's tokenizer config; this is the conservative default used when no
/// explicit cap is wired through yet. Tokens beyond this are truncated
/// rightward (preserves the prompt's opening, matches HF default).
const DEFAULT_MAX_LENGTH: usize = 4096;

/// Thread-safe, load-once tokenizer handle. `Tokenizer` is itself `Send +
/// Sync` once built, so the `OnceCell` just defers the file read off the hot
/// path. `Arc` lets future per-request code clone a cheap handle.
#[derive(Clone)]
pub struct PromptTokenizer {
    inner: Arc<OnceCell<Tokenizer>>,
    max_length: usize,
    /// Path to `tokenizer.json`, stashed here so the lazy `OnceCell`
    /// initializer (which is a `FnOnce` borrowing nothing) can read it.
    path: String,
}

impl PromptTokenizer {
    /// Defer tokenizer load until first encode. `tokenizer_path` is the
    /// `tokenizer.json` the manifest points at (HuggingFace fast-tokenizer
    /// format). `max_length` overrides [`DEFAULT_MAX_LENGTH`] when the
    /// manifest carries a real context window.
    pub fn new(tokenizer_path: impl Into<String>, max_length: Option<usize>) -> Self {
        Self {
            inner: Arc::new(OnceCell::new()),
            max_length: max_length.unwrap_or(DEFAULT_MAX_LENGTH),
            // The path is stashed inside the cell's getter closure below.
            path: tokenizer_path.into(),
        }
    }

    /// Load the tokenizer file lazily. Idempotent after the first call.
    ///
    /// Calls `get_or_try_init` directly on `self.inner` (via `Arc::as_ref`)
    /// rather than cloning the cell into a local — the returned `&Tokenizer`
    /// then borrows `self` (long-lived) instead of a dropped local `Arc`.
    async fn get(&self) -> Result<&Tokenizer> {
        let path = self.path.clone();
        let max_length = self.max_length;
        self.inner
            .as_ref()
            .get_or_try_init(move || async move {
                let mut tok = Tokenizer::from_file(&path)
                    .map_err(|e| anyhow!("load tokenizer {p}: {e}", p = path))?;
                // Apply a right-truncation so encode() can never overrun the
                // model's context window. `with_truncation` mutates in place
                // and returns `&mut Self` (does not consume); we ignore the
                // `&mut` return and keep `tok` as the owned, configured cell.
                let trunc = tokenizers::TruncationParams {
                    max_length,
                    ..Default::default()
                };
                tok.with_truncation(Some(trunc))
                    .map_err(|e| anyhow!("set truncation: {e}"))?;
                Ok::<_, anyhow::Error>(tok)
            })
            .await
    }

    /// Encode an OpenAI-style message list into token ids. Flattens the
    /// messages with a minimal `role:content` join; a real chat template
    /// (Jinja) lands once `helios-pkg` carries it — for now this is enough
    /// for the runtime to hand a deterministic token slice to the HAL.
    ///
    /// Returns the token ids and the (post-truncation) length actually sent.
    /// `add_special_tokens=true` lets the tokenizer emit BOS/EOS per its
    /// config, which the HAL assumes.
    pub async fn encode_messages(&self, messages: &[ChatMessageForEncode<'_>]) -> Result<Vec<u32>> {
        let prompt = flatten_messages(messages);
        let tok = self.get().await?;
        let enc = tok
            .encode(prompt.as_str(), true)
            .map_err(|e| anyhow!("tokenizer encode failed: {e}"))
            .context("prompt encode")?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode a token id stream back to text (for streaming responses). Not
    /// on the prefill hot path; included so the API layer can surface
    /// generated tokens without pulling another dep.
    pub async fn decode(&self, ids: &[u32]) -> Result<String> {
        let tok = self.get().await?;
        tok.decode(ids, true)
            .map_err(|e| anyhow!("tokenizer decode failed: {e}"))
            .context("token decode")
    }
}

/// Flatten `[{role, content}, ...]` into the single string the tokenizer
/// sees. A real chat template lives in the package; this minimal join keeps
/// the runtime deterministic and toolchain-free for now.
fn flatten_messages(messages: &[ChatMessageForEncode]) -> String {
    let mut out = String::new();
    for m in messages {
        if !out.is_empty() {
            out.push('\n');
        }
        // role:content keeps the tokenizer's pre-tokenizer honest about
        // boundaries without inventing a template grammar.
        out.push_str(m.role);
        out.push(':');
        out.push_str(&m.content);
    }
    out
}

/// Borrowed view of a chat message for encoding — avoids taking ownership
/// of the API-layer struct so this module doesn't depend on `api.rs`'s
/// serde types. Mirrors `api::ChatMessage` field-for-field.
pub struct ChatMessageForEncode<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flatten helper is the only logic we can test without a real
    /// `tokenizer.json` on disk (that file comes from a `.hyperion` package).
    fn flattened(role: &str, content: &str) -> String {
        flatten_messages(&[ChatMessageForEncode { role, content }])
    }

    #[test]
    fn flatten_single_message() {
        assert_eq!(flattened("user", "hello"), "user:hello");
    }

    #[test]
    fn flatten_multiple_messages_join_with_newline() {
        let msgs = [
            ChatMessageForEncode { role: "system", content: "you are helpful" },
            ChatMessageForEncode { role: "user", content: "hi" },
        ];
        assert_eq!(flatten_messages(&msgs), "system:you are helpful\nuser:hi");
    }
}
