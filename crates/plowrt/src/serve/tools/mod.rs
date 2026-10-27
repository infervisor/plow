//! OpenAI tool calling (`tools`, `tool_choice`, `tool_calls`, `role: "tool"`) for any model whose
//! chat template renders tools.
//!
//! Two halves, both model-agnostic:
//!
//! * REQUEST. The checkpoint's own template renders the tool schemas and the tool-call history
//!   ([`request`]), with the conversation mapped the way `transformers` + vLLM hand it to
//!   `apply_chat_template`. Whether a template renders `tools` at all is PROBED at load
//!   ([`ToolSupport::probe`]): a template that ignores them (Mixtral, DeepSeek-V3.x) refuses the
//!   request with a 400 instead of silently answering without tools.
//! * RESPONSE. Each model family answers in its own call syntax. [`ToolFormat`] is selected from
//!   the template's own markers, never from the model name, and [`parse`] holds one streaming
//!   parser per syntax. The text is decoded with special tokens KEPT for these requests (most
//!   families make their call markers special tokens), parsed, and only then stripped.

pub mod parse;
pub mod request;
pub mod schema;
pub mod stream;

#[cfg(test)]
mod http_tests;
#[cfg(test)]
mod parity_tests;

use std::sync::atomic::{AtomicU64, Ordering};

pub use plow_asset::chat::pyjson;
pub use plow_asset::chat::{reasoning_markers, ToolFormat, ToolSupport};

/// The `id` a parsed call gets on the wire.
pub fn call_id(format: ToolFormat) -> String {
    match format {
        // Its template refuses any id that is not 9 characters long.
        ToolFormat::Mistral => random_alnum(9),
        _ => format!("call_{}", random_alnum(24)),
    }
}

/// Special-token text, removed from what a client sees after the tool parser has read it.
#[derive(Debug)]
pub struct SpecialText {
    /// Longest first, so a token that prefixes another does not cut it short.
    tokens: Vec<String>,
    first: [bool; 256],
    longest: usize,
}

impl SpecialText {
    pub fn new(mut tokens: Vec<String>) -> Self {
        tokens.retain(|t| !t.is_empty());
        tokens.sort_by_key(|t| std::cmp::Reverse(t.len()));
        tokens.dedup();
        let mut first = [false; 256];
        for t in &tokens {
            first[t.as_bytes()[0] as usize] = true;
        }
        let longest = tokens.first().map_or(0, String::len);
        SpecialText {
            tokens,
            first,
            longest,
        }
    }

    pub fn strip(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        self.strip_into(s, &mut out);
        out
    }

    /// [`Self::strip`], appended to `out`.
    pub fn strip_into(&self, s: &str, out: &mut String) {
        let Some(first) = s.bytes().position(|b| self.first[b as usize]) else {
            out.push_str(s);
            return;
        };
        out.push_str(&s[..first]);
        let mut i = first;
        'outer: while i < s.len() {
            if self.first[s.as_bytes()[i] as usize] {
                for t in &self.tokens {
                    if s[i..].starts_with(t.as_str()) {
                        i += t.len();
                        continue 'outer;
                    }
                }
            }
            // A token's first byte is never a UTF-8 continuation byte, so `next` is a char boundary.
            let next = s.as_bytes()[i + 1..]
                .iter()
                .position(|&b| self.first[b as usize])
                .map_or(s.len(), |k| i + 1 + k);
            out.push_str(&s[i..next]);
            i = next;
        }
    }

    /// [`Self::strip_into`] over a stream of pieces: `hold` carries a tail that may begin a token
    /// into the next piece (`last` flushes it), so a token split across pieces is still removed.
    pub fn strip_stream(&self, hold: &mut String, s: &str, out: &mut String, last: bool) {
        if hold.is_empty() {
            let keep = if last { 0 } else { self.partial_tail(s) };
            self.strip_into(&s[..s.len() - keep], out);
            hold.push_str(&s[s.len() - keep..]);
            return;
        }
        hold.push_str(s);
        let w = std::mem::take(hold);
        let keep = if last { 0 } else { self.partial_tail(&w) };
        self.strip_into(&w[..w.len() - keep], out);
        hold.push_str(&w[w.len() - keep..]);
    }

    /// Bytes at the end of `s` that are a proper prefix of some token.
    fn partial_tail(&self, s: &str) -> usize {
        let b = s.as_bytes();
        let from = b.len().saturating_sub(self.longest.saturating_sub(1));
        (from..b.len())
            .find(|&p| {
                self.first[b[p] as usize] && {
                    let rest = &s[p..];
                    self.tokens
                        .iter()
                        .any(|t| t.len() > rest.len() && t.starts_with(rest))
                }
            })
            .map_or(0, |p| b.len() - p)
    }

    /// Whether `s` contains any special token's text.
    pub fn any_in(&self, s: &str) -> bool {
        self.tokens.iter().any(|t| s.contains(t.as_str()))
    }
}

static ID_SEQ: std::sync::LazyLock<AtomicU64> =
    std::sync::LazyLock::new(|| AtomicU64::new(rand::random::<u64>()));

fn random_alnum(n: usize) -> String {
    const A: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut x = ID_SEQ.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed) ^ rand::random::<u64>();
    (0..n)
        .map(|_| {
            // splitmix64 step per character
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            A[((z ^ (z >> 31)) % A.len() as u64) as usize] as char
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_text_strips_only_whole_tokens() {
        let s = SpecialText::new(vec![
            "<|tool_call>".into(),
            "<|\"|>".into(),
            "<|turn>".into(),
        ]);
        assert_eq!(s.strip("a<|\"|>b<|tool_call>c<|tool"), "abc<|tool");
        assert_eq!(s.strip("plain"), "plain");
        assert_eq!(s.strip("é<|turn>ü"), "éü");
    }

    #[test]
    fn ids_have_the_shape_their_format_requires() {
        let m = call_id(ToolFormat::Mistral);
        assert_eq!(m.len(), 9);
        assert!(m.chars().all(|c| c.is_ascii_alphanumeric()));
        let h = call_id(ToolFormat::Hermes);
        assert!(h.starts_with("call_") && h.len() == 29);
        assert_ne!(call_id(ToolFormat::Hermes), h);
    }
}
