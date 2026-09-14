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
        for base in [dir.to_path_buf(), dir.join("checkpoint")] {
            if base.join("vocab.json").is_file() && base.join("merges.txt").is_file() {
                match HfTokenizer::from_qwen2_files(&base) {
                    Ok(t) => return Arc::new(t),
                    Err(e) => tracing::warn!(error = %e, "Qwen2 tokenizer failed to load"),
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
    /// `get_vocab_size(true)` rebuilds the whole vocab map per call (~10 ms on a 154k vocab);
    /// the vocabulary never changes after load.
    vocab_size: usize,
    fast: bool,
    split: bool,
}

/// Pre-tokenizer patterns under which a single ASCII space between two ASCII letters always ends
/// one pre-token and starts the next, with no lookaround reaching across it. Encoding pieces cut
/// there gives the whole-text ids. GLM-5.x `tokenizer.json`, then the Qwen2 override below.
#[cfg(feature = "hf-tokenizer")]
const SPLIT_SAFE_PATTERNS: &[&str] = &[
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    QWEN2_PATTERN,
];

#[cfg(feature = "hf-tokenizer")]
const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Smallest piece a split encode hands a thread.
#[cfg(feature = "hf-tokenizer")]
const SPLIT_MIN_BYTES: usize = 4096;

#[cfg(feature = "hf-tokenizer")]
static ENCODE_POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();

#[cfg(feature = "hf-tokenizer")]
fn split_safe(t: &tokenizers::Tokenizer) -> bool {
    use tokenizers::PostProcessor;
    let pre = t
        .get_pre_tokenizer()
        .and_then(|p| serde_json::to_value(p).ok())
        .unwrap_or_default();
    let steps = pre["pretokenizers"].as_array().map_or(&[][..], Vec::as_slice);
    let pre_ok = pre["type"] == "Sequence"
        && matches!(steps, [split, byte_level]
            if split["type"] == "Split"
                && split["pattern"]["Regex"].as_str().is_some_and(|r| SPLIT_SAFE_PATTERNS.contains(&r))
                && split["behavior"] == "Isolated"
                && split["invert"] == false
                && byte_level["type"] == "ByteLevel"
                && byte_level["add_prefix_space"] == false
                && byte_level["use_regex"] == false);
    pre_ok
        && t.get_normalizer().is_none()
        && t.get_truncation().is_none()
        && t.get_padding().is_none()
        && t.get_post_processor().map_or(true, |p| p.added_tokens(false) == 0)
        && matches!(t.get_model(), tokenizers::models::ModelWrapper::BPE(b) if b.dropout.is_none())
        && t.get_added_tokens_decoder()
            .values()
            .all(|a| !a.lstrip && !a.rstrip && !a.single_word && !a.content.contains(' '))
}

/// Cut `text` into at most `parts` pieces, each cut just before a space that sits between two
/// ASCII letters.
#[cfg(feature = "hf-tokenizer")]
fn split_pieces(text: &str, parts: usize) -> Vec<&str> {
    let b = text.as_bytes();
    let mut out = Vec::with_capacity(parts);
    let mut start = 0;
    for i in 1..parts {
        let mut p = (b.len() * i / parts).max(start + 1);
        while p + 1 < b.len()
            && !(b[p] == b' ' && b[p - 1].is_ascii_alphabetic() && b[p + 1].is_ascii_alphabetic())
        {
            p += 1;
        }
        if p + 1 >= b.len() {
            break;
        }
        out.push(&text[start..p]);
        start = p;
    }
    out.push(&text[start..]);
    out
}

#[cfg(feature = "hf-tokenizer")]
fn encode_split(
    inner: &tokenizers::Tokenizer,
    pool: &rayon::ThreadPool,
    text: &str,
    add_special_tokens: bool,
) -> Option<Vec<u32>> {
    use rayon::prelude::*;
    let parts = (text.len() / SPLIT_MIN_BYTES).min(pool.current_num_threads());
    if parts < 2 {
        return None;
    }
    let pieces = split_pieces(text, parts);
    let ids: tokenizers::Result<Vec<Vec<u32>>> = pool.install(|| {
        pieces
            .par_iter()
            .map(|p| inner.encode_fast(*p, add_special_tokens).map(|e| e.get_ids().to_vec()))
            .collect()
    });
    Some(ids.map(|v| v.concat()).unwrap_or_default())
}

#[cfg(feature = "hf-tokenizer")]
impl HfTokenizer {
    pub fn from_qwen2_files(dir: &Path) -> crate::Result<Self> {
        let fail = |e: String| crate::RuntimeError::Msg(format!("Qwen2 tokenizer: {e}"));
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("tokenizer_config.json")).map_err(|e| fail(e.to_string()))?,
        )
        .map_err(|e| fail(e.to_string()))?;
        if !matches!(
            config["tokenizer_class"].as_str(),
            Some("Qwen2Tokenizer" | "Qwen2TokenizerFast")
        ) {
            return Err(fail(
                "vocab/merges loading requires a Qwen2 tokenizer class".into(),
            ));
        }
        let model = tokenizers::models::bpe::BPE::from_file(
            &dir.join("vocab.json").to_string_lossy(),
            &dir.join("merges.txt").to_string_lossy(),
        )
        .build()
        .map_err(|e| fail(e.to_string()))?;
        let mut inner = tokenizers::Tokenizer::new(model);
        let added = config["added_tokens_decoder"]
            .as_object()
            .ok_or_else(|| fail("missing added_tokens_decoder".into()))?;
        let mut tokens = added
            .iter()
            .map(|(id, value)| {
                let id = id.parse::<u32>().map_err(|e| fail(e.to_string()))?;
                let token: tokenizers::AddedToken =
                    serde_json::from_value(value.clone()).map_err(|e| fail(e.to_string()))?;
                Ok((id, token))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        tokens.sort_by_key(|(id, _)| *id);
        for (id, token) in tokens {
            let content = token.content.clone();
            inner.add_tokens(&[token]);
            if inner.token_to_id(&content) != Some(id) {
                return Err(fail(format!("token {content:?} must have id {id}")));
            }
        }
        Self::qwen2_processors(&mut inner, &config)?;
        Ok(Self::loaded(inner))
    }

    fn loaded(inner: tokenizers::Tokenizer) -> Self {
        let rt = crate::config::RuntimeConfig::get();
        let split = match rt.encode_threads.filter(|&n| n >= 2) {
            Some(_) if !split_safe(&inner) => {
                tracing::warn!("PLOW_ENCODE_THREADS: tokenizer is not split-safe; encoding serially");
                false
            }
            Some(n) => ENCODE_POOL
                .get_or_init(|| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(n as usize)
                        .thread_name(|i| format!("plow-encode-{i}"))
                        .build()
                        .map_err(|e| tracing::warn!(error = %e, "PLOW_ENCODE_THREADS: pool build failed; encoding serially"))
                        .ok()
                })
                .is_some(),
            None => false,
        };
        HfTokenizer {
            vocab_size: inner.get_vocab_size(true),
            inner,
            fast: rt.encode_fast,
            split,
        }
    }

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
                Self::qwen2_processors(&mut inner, &config)?;
            }
        }
        Ok(Self::loaded(inner))
    }

    fn qwen2_processors(
        inner: &mut tokenizers::Tokenizer,
        config: &serde_json::Value,
    ) -> crate::Result<()> {
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
        Ok(())
    }
}

#[cfg(feature = "hf-tokenizer")]
impl Tokenize for HfTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_with_special_tokens(text, false)
    }

    fn encode_with_special_tokens(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        if let Some(Some(pool)) = self.split.then(|| ENCODE_POOL.get()).flatten() {
            if let Some(ids) = encode_split(&self.inner, pool, text, add_special_tokens) {
                return ids;
            }
        }
        if self.fast {
            self.inner.encode_fast(text, add_special_tokens)
        } else {
            self.inner.encode(text, add_special_tokens)
        }
        .map(|e| e.get_ids().to_vec())
        .unwrap_or_default()
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).unwrap_or_default()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(all(test, feature = "hf-tokenizer"))]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn glm_like(add_prefix_space: bool) -> tokenizers::Tokenizer {
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort_unstable();
        let mut vocab: serde_json::Map<String, serde_json::Value> = alphabet
            .iter()
            .enumerate()
            .map(|(i, c)| (c.to_string(), i.into()))
            .collect();
        let merges = ["h e", "l l", "he ll", "Ġ w", "Ġw o", "o r", "Ġ a", "i t", "' s", "1 2"];
        for m in merges {
            let n = vocab.len();
            vocab.insert(m.replace(' ', ""), n.into());
        }
        let n = vocab.len();
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "normalizer": null,
            "added_tokens": [{"id": n, "content": "<|user|>", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true}],
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": SPLIT_SAFE_PATTERNS[0]}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": add_prefix_space, "trim_offsets": true, "use_regex": false}]},
            "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true},
            "decoder": null,
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": true,
                "vocab": vocab, "merges": merges}
        });
        tokenizers::Tokenizer::from_str(&json.to_string()).unwrap()
    }

    fn texts() -> Vec<String> {
        let frags = [
            "hello", " world", "it's", " a", "  ", "\n", "\r\n", "\t", "123", "4567", "é", "日本",
            "<|user|>", "x", " 's", "!?", "a b", "a  b", "a \nb", "ab12 cd", " ", "Z",
        ];
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        (0..64)
            .map(|_| {
                let mut s = String::new();
                while s.len() < 40_000 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    s.push_str(frags[(seed % frags.len() as u64) as usize]);
                }
                s
            })
            .collect()
    }

    #[test]
    fn pieces_cut_only_between_ascii_letters_and_rejoin() {
        for text in texts() {
            for parts in [2, 3, 8] {
                let pieces = split_pieces(&text, parts);
                assert!(pieces.len() <= parts);
                assert_eq!(pieces.concat(), text);
                for w in pieces.windows(2) {
                    let (a, b) = (w[0].as_bytes(), w[1].as_bytes());
                    assert!(a.last().unwrap().is_ascii_alphabetic());
                    assert!(b[0] == b' ' && b[1].is_ascii_alphabetic());
                }
            }
        }
        assert_eq!(split_pieces("no-cut-here", 4), vec!["no-cut-here"]);
    }

    #[test]
    fn split_encode_matches_whole_text_encode() {
        let t = glm_like(false);
        assert!(split_safe(&t));
        assert!(!split_safe(&glm_like(true)));
        let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
        for text in texts() {
            for special in [false, true] {
                let whole = t.encode(text.as_str(), special).unwrap().get_ids().to_vec();
                let split = encode_split(&t, &pool, &text, special).expect("long text splits");
                assert_eq!(split, whole);
            }
        }
        assert!(encode_split(&t, &pool, "hello world", false).is_none());
    }

    #[test]
    fn vocab_size_is_the_loaded_vocab_with_added_tokens() {
        let dir = std::env::temp_dir().join(format!("plowrt_tok_vocab_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        let t = glm_like(false);
        t.save(&path, false).unwrap();
        let hf = HfTokenizer::from_file(&path).unwrap();
        assert_eq!(hf.vocab_size(), t.get_vocab_size(true));
        assert_eq!(hf.vocab_size(), 256 + 10 + 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
