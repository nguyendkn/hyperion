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

use crate::chat_template::ChatTemplate;
use tokenizers::Tokenizer;
use tokio::sync::OnceCell;

/// Longest prompt (in tokens) accepted when no explicit limit is wired through.
/// The CLI passes the serving limit (`--max-seq-len`, at most the model's
/// `max_seq_len`). A longer prompt is rejected, never truncated: silently
/// dropping the end of a prompt changes the answer without telling the client.
const DEFAULT_MAX_LENGTH: usize = 4096;

/// The prompt has more tokens than the serving limit. The API maps this to
/// HTTP 400 so the client can shorten the prompt.
#[derive(Debug, thiserror::Error)]
#[error("prompt has {tokens} tokens but the limit is {max}: shorten the prompt or raise --max-seq-len")]
pub struct PromptTooLong {
    pub tokens: usize,
    pub max: usize,
}

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
    /// The model's own chat template. Without one, messages fall back to a plain
    /// `role:content` join.
    chat_template: Option<Arc<ChatTemplate>>,
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
            chat_template: None,
        }
    }

    /// The longest prompt (and context) this tokenizer accepts.
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Format prompts with the model's chat template instead of the plain join.
    pub fn with_chat_template(mut self, template: ChatTemplate) -> Self {
        self.chat_template = Some(Arc::new(template));
        self
    }

    /// Load the tokenizer file lazily. Idempotent after the first call.
    ///
    /// Calls `get_or_try_init` directly on `self.inner` (via `Arc::as_ref`)
    /// rather than cloning the cell into a local — the returned `&Tokenizer`
    /// then borrows `self` (long-lived) instead of a dropped local `Arc`.
    async fn get(&self) -> Result<&Tokenizer> {
        let path = self.path.clone();
        self.inner
            .as_ref()
            .get_or_try_init(move || async move {
                let tok = Tokenizer::from_file(&path)
                    .map_err(|e| anyhow!("load tokenizer {p}: {e}", p = path))?;
                Ok::<_, anyhow::Error>(tok)
            })
            .await
    }

    /// Encode an OpenAI-style message list into token ids. Flattens the
    /// messages with a minimal `role:content` join; a real chat template
    /// (Jinja) lands once `helios-pkg` carries it — for now this is enough
    /// for the runtime to hand a deterministic token slice to the HAL.
    ///
    /// Fails with [`PromptTooLong`] when the prompt exceeds the limit; nothing is
    /// truncated. `add_special_tokens=true` lets the tokenizer emit BOS/EOS per its
    /// config, which the HAL assumes.
    pub async fn encode_messages(&self, messages: &[ChatMessageForEncode<'_>]) -> Result<Vec<u32>> {
        // Loading the file is the only async part and happens once; rendering and BPE encoding are
        // CPU-bound (milliseconds for a long prompt), so they run on the blocking pool and never
        // stall the runtime threads that serve other requests and the health endpoint.
        self.get().await?;
        let cell = self.inner.clone();
        let template = self.chat_template.clone();
        let max_length = self.max_length;
        let owned: Vec<(String, String)> =
            messages.iter().map(|m| (m.role.to_string(), m.content.to_string())).collect();
        tokio::task::spawn_blocking(move || {
            let borrowed: Vec<ChatMessageForEncode<'_>> =
                owned.iter().map(|(role, content)| ChatMessageForEncode { role, content }).collect();
            // With a chat template the rendered text already contains every special token, so it
            // is tokenised without adding more (what `apply_chat_template` + `tokenizer(...,
            // add_special_tokens=False)` does in Hugging Face).
            let (prompt, add_special_tokens) = match &template {
                Some(template) => (template.render(&borrowed, true)?, false),
                None => (flatten_messages(&borrowed), true),
            };
            let tok = cell.get().ok_or_else(|| anyhow!("tokenizer not loaded"))?;
            let enc = tok
                .encode(prompt.as_str(), add_special_tokens)
                .map_err(|e| anyhow!("tokenizer encode failed: {e}"))
                .context("prompt encode")?;
            let ids = enc.get_ids().to_vec();
            if ids.len() > max_length {
                return Err(PromptTooLong { tokens: ids.len(), max: max_length }.into());
            }
            Ok(ids)
        })
        .await
        .map_err(|e| anyhow!("tokenization task failed: {e}"))?
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

/// Turns a stream of generated ids into text deltas without ever emitting a half-decoded
/// character: a byte-level tokenizer may split one character (an emoji, a CJK glyph) over several
/// tokens, and decoding the prefix then ends in U+FFFD until the rest arrives, so that tail is held
/// back. The whole id list is decoded on every push (quadratic in the length, fine for the lengths
/// served today; an offset-based decoder is the later optimisation).
#[derive(Default)]
pub struct StreamDecoder {
    ids: Vec<u32>,
    /// Bytes of the decoded text already handed out.
    emitted: usize,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one token; returns the new text that is safe to show (possibly empty).
    pub async fn push(&mut self, tok: &PromptTokenizer, id: u32) -> Result<String> {
        self.ids.push(id);
        let text = tok.decode(&self.ids).await?;
        if text.ends_with('\u{FFFD}') {
            return Ok(String::new());
        }
        Ok(self.take(&text))
    }

    /// Everything not yet emitted, including a trailing incomplete character (which then shows
    /// as U+FFFD, exactly as a full decode would).
    pub async fn finish(&mut self, tok: &PromptTokenizer) -> Result<String> {
        if self.ids.is_empty() {
            return Ok(String::new());
        }
        let text = tok.decode(&self.ids).await?;
        Ok(self.take(&text))
    }

    fn take(&mut self, text: &str) -> String {
        let delta = text.get(self.emitted..).unwrap_or("").to_string();
        self.emitted = text.len().max(self.emitted);
        delta
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
