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
    assert_eq!(t.encode_with_special_tokens("hi", true), ids);
    assert_eq!(t.encode_with_special_tokens("hi", false), ids);
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

#[cfg(feature = "hf-tokenizer")]
#[test]
fn qwen2_class_processors_match_transformers_combining_marks() {
    use plowrt::text::tokenizer::HfTokenizer;
    let dir = tmpdir("qwen2");
    let path = dir.join("tokenizer.json");
    std::fs::write(
        &path,
        include_str!("fixtures/qwen2-combining-tokenizer.json"),
    )
    .unwrap();
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen2-combining-cases.json")).unwrap();
    let texts: Vec<_> = cases
        .as_array()
        .unwrap()
        .iter()
        .map(|case| case["text"].as_str().unwrap())
        .collect();
    let raw = HfTokenizer::from_file(&path).unwrap();
    assert_eq!(raw.encode(texts[0]).len(), 111);
    assert_eq!(raw.encode(texts[1]).len(), 7);

    // Other tokenizer classes retain their serialized processors.
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"tokenizer_class":"GemmaTokenizerFast"}"#,
    )
    .unwrap();
    let unchanged = HfTokenizer::from_file(&path).unwrap();
    for text in &texts {
        assert_eq!(unchanged.encode(text), raw.encode(text));
    }
    std::fs::remove_file(dir.join("tokenizer_config.json")).unwrap();

    std::fs::create_dir(dir.join("checkpoint")).unwrap();
    for class in ["Qwen2Tokenizer", "Qwen2TokenizerFast"] {
        std::fs::write(
            dir.join("checkpoint/tokenizer_config.json"),
            serde_json::json!({"tokenizer_class":class}).to_string(),
        )
        .unwrap();
        let fixed = HfTokenizer::from_file(&path).unwrap();
        for case in cases.as_array().unwrap() {
            let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            let text = case["text"].as_str().unwrap();
            assert_eq!(fixed.encode(text), expected);
            assert_eq!(fixed.encode_with_special_tokens(text, true), expected);
            assert_eq!(fixed.decode(&expected), raw.decode(&raw.encode(text)));
        }
        assert_eq!(fixed.encode(texts[0]).len(), 128);
        assert_eq!(fixed.encode(texts[1]).len(), 12);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
