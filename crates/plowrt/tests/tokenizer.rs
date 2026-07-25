//! §H Tokenization: byte fallback always works; a real HF `tokenizer.json` is
//! loaded and used when the `hf-tokenizer` feature is on.

use plowrt::text::tokenizer::{load_tokenizer, ByteTokenizer, Tokenize};

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("plowrt_tok_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn byte_tokenizer_roundtrips() {
    let t = ByteTokenizer;
    let ids = t.encode("hi");
    assert_eq!(ids, vec![104, 105]);
    assert_eq!(t.decode(&ids), "hi");
}

#[test]
fn falls_back_to_bytes_without_tokenizer_json() {
    let dir = tmpdir("nojson");
    let t = load_tokenizer(&dir);
    // Byte behavior: 'A' == 65.
    assert_eq!(t.encode("A"), vec![65]);
    std::fs::remove_dir_all(&dir).ok();
}

/// A minimal WordLevel `tokenizer.json` the HF `tokenizers` library can load.
#[cfg(feature = "hf-tokenizer")]
const WORDLEVEL_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": { "hello": 0, "world": 1, "[UNK]": 2 },
    "unk_token": "[UNK]"
  }
}"#;

#[cfg(feature = "hf-tokenizer")]
#[test]
fn loads_and_uses_real_hf_tokenizer() {
    let dir = tmpdir("hf");
    std::fs::write(dir.join("tokenizer.json"), WORDLEVEL_JSON).unwrap();

    let t = load_tokenizer(&dir);
    // The real tokenizer maps words to their WordLevel ids (not bytes).
    assert_eq!(t.encode("hello world"), vec![0, 1]);
    assert_eq!(t.encode("hello nope"), vec![0, 2]); // OOV → [UNK]
    std::fs::remove_dir_all(&dir).ok();
}
