//! §H Tokenizer — the HuggingFace `tokenizers` library (BPE / WordPiece /
//! Unigram, loads a model's `tokenizer.json`) under the `hf-tokenizer` feature;
//! a byte-level fallback stands in by default so the runtime works offline.
//!
//! [`load_tokenizer`] picks the best available tokenizer for a model directory:
//! a real `tokenizer.json` when present (feature on), else [`ByteTokenizer`].

use std::path::Path;
use std::sync::Arc;

/// A tokenizer the runtime can encode/decode with. `Send + Sync` so a loaded
/// tokenizer can be shared across the async request handlers.
pub trait Tokenize: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn encode_with_special_tokens(&self, text: &str, _add_special_tokens: bool) -> Vec<u32> {
        self.encode(text)
    }
    fn decode(&self, ids: &[u32]) -> String;
    /// Number of token ids accepted by the model embedding table.
    fn vocab_size(&self) -> usize;
    /// True for the byte-fallback tokenizer. A real model served through the
    /// byte fallback produces silent garbage (the ids bear no relation to the
    /// checkpoint's vocab), so the GPU-engine install path refuses it loudly.
    fn is_byte_fallback(&self) -> bool {
        false
    }
}

/// UTF-8 byte tokenizer: id = byte value. Deterministic, dependency-free — the
/// fallback when no `tokenizer.json` is present or the feature is off.
pub struct ByteTokenizer;

impl Tokenize for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().map(|&id| id as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn vocab_size(&self) -> usize {
        256
    }

    fn is_byte_fallback(&self) -> bool {
        true
    }
}

/// Load the best tokenizer for a compiled-model directory: a real HF
/// `tokenizer.json` (feature `hf-tokenizer`) when present, else the byte
/// fallback. Never fails — a missing/broken tokenizer degrades to bytes with a
/// warning, so serving still works.
pub fn load_tokenizer(dir: &Path) -> Arc<dyn Tokenize> {
    #[cfg(feature = "hf-tokenizer")]
    {
        let path = dir.join("tokenizer.json");
        if path.exists() {
            match HfTokenizer::from_file(&path) {
                Ok(t) => {
                    tracing::info!(path = %path.display(), "loaded HF tokenizer");
                    return Arc::new(t);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "tokenizer.json failed to load; byte fallback")
                }
            }
        }
    }
    let _ = dir;
    Arc::new(ByteTokenizer)
}

/// A HuggingFace `tokenizers`-backed tokenizer loaded from a `tokenizer.json`.
#[cfg(feature = "hf-tokenizer")]
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

#[cfg(feature = "hf-tokenizer")]
impl HfTokenizer {
    /// Load a `tokenizer.json` from disk.
    pub fn from_file(path: &std::path::Path) -> crate::Result<Self> {
        let mut inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| crate::RuntimeError::Msg(format!("tokenizer load: {e}")))?;
        if let Some(dir) = path.parent() {
            let config = [
                dir.join("tokenizer_config.json"),
                dir.join("checkpoint/tokenizer_config.json"),
            ]
            .into_iter()
            .find_map(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
            if let Some(config) = config.filter(|c| {
                matches!(
                    c["tokenizer_class"].as_str(),
                    Some("Qwen2Tokenizer" | "Qwen2TokenizerFast")
                )
            }) {
                use tokenizers::pre_tokenizers::{
                    byte_level::ByteLevel,
                    sequence::Sequence,
                    split::{Split, SplitPattern},
                };
                // Transformers reconstructs Qwen2's processors from its class, overriding
                // tokenizer.json's combining-mark regex. Match that reference API behavior.
                let split = Split::new(
                    SplitPattern::Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+".into()),
                    tokenizers::SplitDelimiterBehavior::Isolated, false,
                ).map_err(|e| crate::RuntimeError::Msg(format!("Qwen2 tokenizer: {e}")))?;
                let prefix = config["add_prefix_space"].as_bool().unwrap_or(false);
                inner.with_pre_tokenizer(Some(Sequence::new(vec![
                    split.into(),
                    ByteLevel::new(prefix, true, false).into(),
                ])));
                inner.with_decoder(Some(ByteLevel::default()));
            }
        }
        Ok(HfTokenizer { inner })
    }
}

#[cfg(feature = "hf-tokenizer")]
impl Tokenize for HfTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_with_special_tokens(text, false)
    }

    fn encode_with_special_tokens(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).unwrap_or_default()
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
