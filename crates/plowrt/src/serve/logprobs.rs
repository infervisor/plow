//! OpenAI `logprobs` on the wire: request parsing and the chat / completions response shapes.
//!
//! Chat: `logprobs: true` + `top_logprobs: 0..=20` -> `choices[].logprobs.content[]`.
//! Completions: `logprobs: 0..=20` -> `choices[].logprobs.{tokens, token_logprobs, top_logprobs,
//! text_offset}`. Extensions (vLLM names): `logprobs_mode: "raw_logprobs" | "raw_logits"` (the
//! latter reports the model's raw last-position logits in the same fields) and
//! `return_tokens_as_token_ids` (`token` = `"token_id:<id>"`). Values are over the raw model
//! distribution (after the checkpoint's final-logit softcap, before temperature and penalties).

use std::sync::Arc;

use serde::Serialize;
use serde_json::Value;

use crate::text::logprobs::{LogprobRequest, TokenLogprobs, MAX_TOP_LOGPROBS};
use crate::text::tokenizer::Tokenize;

/// A request field that failed validation: `(message, param)`.
pub type Invalid = (String, &'static str);

fn mode_raw_logits(mode: Option<&str>) -> Result<bool, Invalid> {
    match mode {
        None | Some("raw_logprobs") => Ok(false),
        Some("raw_logits") => Ok(true),
        Some(m) => Err((
            format!("logprobs_mode {m:?} is not supported (raw_logprobs, raw_logits)"),
            "logprobs_mode",
        )),
    }
}

fn top_count(v: &Value, param: &'static str) -> Result<u8, Invalid> {
    v.as_u64()
        .filter(|&n| n <= u64::from(MAX_TOP_LOGPROBS))
        .map(|n| n as u8)
        .ok_or_else(|| (format!("`{param}` must be an integer in 0..={MAX_TOP_LOGPROBS}"), param))
}

/// Chat: `logprobs` (bool) and `top_logprobs` (int, requires `logprobs: true`).
pub fn parse_chat(
    logprobs: Option<&Value>,
    top: Option<&Value>,
    mode: Option<&str>,
) -> Result<Option<LogprobRequest>, Invalid> {
    let on = match logprobs.filter(|v| !v.is_null()) {
        None => false,
        Some(v) => v.as_bool().ok_or_else(|| ("`logprobs` must be a boolean".to_string(), "logprobs"))?,
    };
    let top = top.filter(|v| !v.is_null()).map(|v| top_count(v, "top_logprobs")).transpose()?;
    if !on {
        if top.is_some_and(|n| n > 0) {
            return Err(("`top_logprobs` requires `logprobs: true`".into(), "top_logprobs"));
        }
        return Ok(None);
    }
    Ok(Some(LogprobRequest { top: top.unwrap_or(0), raw_logits: mode_raw_logits(mode)? }))
}

/// Completions: `logprobs` is the number of alternatives (0 = the sampled token only).
pub fn parse_completion(logprobs: Option<&Value>, mode: Option<&str>) -> Result<Option<LogprobRequest>, Invalid> {
    match logprobs.filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => Ok(Some(LogprobRequest { top: top_count(v, "logprobs")?, raw_logits: mode_raw_logits(mode)? })),
    }
}

/// Renders token ids for the logprobs fields.
#[derive(Clone)]
pub struct TokenText {
    pub tok: Arc<dyn Tokenize>,
    pub as_ids: bool,
}

impl TokenText {
    fn text(&self, id: u32) -> String {
        if self.as_ids {
            format!("token_id:{id}")
        } else {
            self.tok.decode(&[id])
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatLogprobs {
    pub content: Vec<ChatTokenLogprob>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatTokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<ChatTopLogprob>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatTopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

pub fn chat_entry(fmt: &TokenText, id: u32, lp: &TokenLogprobs) -> ChatTokenLogprob {
    let token = fmt.text(id);
    ChatTokenLogprob {
        bytes: token.as_bytes().to_vec(),
        token,
        logprob: lp.logprob,
        top_logprobs: lp
            .top
            .iter()
            .map(|&(t, v)| {
                let token = fmt.text(t);
                ChatTopLogprob { bytes: token.as_bytes().to_vec(), token, logprob: v }
            })
            .collect(),
    }
}

/// Completions `logprobs` object, accumulated token by token.
#[derive(Clone, Debug, Default, Serialize)]
pub struct CompletionLogprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    pub top_logprobs: Vec<serde_json::Map<String, Value>>,
    pub text_offset: Vec<usize>,
}

impl CompletionLogprobs {
    /// Append one token; `offset` is its start in the returned text.
    pub fn push(&mut self, fmt: &TokenText, id: u32, lp: &TokenLogprobs, offset: usize) {
        self.tokens.push(fmt.text(id));
        self.token_logprobs.push(lp.logprob);
        self.top_logprobs.push(lp.top.iter().map(|&(t, v)| (fmt.text(t), Value::from(v))).collect());
        self.text_offset.push(offset);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chat_fields_validate() {
        assert_eq!(parse_chat(None, None, None), Ok(None));
        assert_eq!(
            parse_chat(Some(&json!(true)), Some(&json!(5)), None),
            Ok(Some(LogprobRequest { top: 5, raw_logits: false }))
        );
        assert_eq!(
            parse_chat(Some(&json!(true)), None, Some("raw_logits")),
            Ok(Some(LogprobRequest { top: 0, raw_logits: true }))
        );
        assert!(parse_chat(Some(&json!(false)), Some(&json!(3)), None).is_err());
        assert!(parse_chat(Some(&json!(true)), Some(&json!(21)), None).is_err());
        assert!(parse_chat(Some(&json!(true)), None, Some("processed")).is_err());
        assert_eq!(
            parse_completion(Some(&json!(2)), None),
            Ok(Some(LogprobRequest { top: 2, raw_logits: false }))
        );
        assert!(parse_completion(Some(&json!(true)), None).is_err());
    }
}
