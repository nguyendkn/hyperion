//! Render a model's own chat template. Hugging Face ships the prompt format of every model
//! as a Jinja template in `tokenizer_config.json` (`chat_template`); rendering that template,
//! with the same settings Hugging Face uses, reproduces the exact prompt the model was trained
//! on without any per-model code here. The template is loaded once and rendered per request.

use anyhow::{anyhow, Context, Result};
use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Serde, Value};
use minijinja::{context, Environment, Error, ErrorKind};
use serde::Serialize;

use crate::tokenizer::ChatMessageForEncode;

const TEMPLATE_NAME: &str = "chat";

/// A compiled chat template. Cheap to render, safe to share behind an `Arc`.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

/// A special token in `tokenizer_config.json` is a string or `{"content": "..."}`.
fn token_text(config: &serde_json::Value, key: &str) -> String {
    match &config[key] {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(o) => o.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        _ => String::new(),
    }
}

impl ChatTemplate {
    /// Build from the text of a `tokenizer_config.json`. `Ok(None)` when it defines no
    /// (single-string) `chat_template`; an error when the template does not compile.
    pub fn from_tokenizer_config(config_json: &str) -> Result<Option<Self>> {
        let config: serde_json::Value = serde_json::from_str(config_json).context("tokenizer_config.json is not valid JSON")?;
        let Some(source) = config["chat_template"].as_str() else {
            return Ok(None);
        };

        let mut env = Environment::new();
        // The settings Hugging Face's `apply_chat_template` uses.
        env.set_syntax(SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build().map_err(|e| anyhow!("{e:#}"))?);
        // Templates call Python string/dict methods (`.startswith`, `.items()`, ...).
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |message: String| -> std::result::Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        });
        env.add_template_owned(TEMPLATE_NAME, source.to_string()).map_err(|e| anyhow!("chat template does not compile: {e:#}"))?;
        Ok(Some(Self { env, bos_token: token_text(&config, "bos_token"), eos_token: token_text(&config, "eos_token") }))
    }

    /// The prompt text for `messages`; `add_generation_prompt` appends the opening of the
    /// assistant turn so the model continues as the assistant.
    pub fn render(&self, messages: &[ChatMessageForEncode<'_>], add_generation_prompt: bool) -> Result<String> {
        let view: Vec<Message> = messages.iter().map(|m| Message { role: m.role, content: m.content }).collect();
        let template = self.env.get_template(TEMPLATE_NAME).map_err(|e| anyhow!("{e:#}"))?;
        template
            .render(context! {
                messages => Value::from(Serde(&view)),
                add_generation_prompt => add_generation_prompt,
                bos_token => &self.bos_token,
                eos_token => &self.eos_token,
            })
            .map_err(|e| anyhow!("rendering the chat template failed: {e:#}"))
    }
}
