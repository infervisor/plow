//! Chat-template parity with `transformers` for tool conversations, per model family.
//!
//! `tests/fixtures/toolcall/<family>/` holds the checkpoint's own template and the reference
//! renders from `scripts/llm/toolcall_fixtures.py` (`render_jinja_template`, the code path of
//! `apply_chat_template(messages, tools=...)`). Each OpenAI-shaped request goes through the
//! handler's mapping and renderer and must produce the same text, and the same token ids when
//! the family's `tokenizer.json` recorded in the fixture is on this host.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::request::{render, RenderError};
use super::{ToolFormat, ToolSupport};
use crate::serve::openai::ChatRequest;
use crate::serve::template::{ChatTemplate, RenderOpts};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toolcall")
}

fn expected_support(family: &str) -> ToolSupport {
    use ToolFormat::*;
    ToolSupport::Format(match family {
        f if f.starts_with("gemma4") => Gemma4,
        "qwen3" | "qwen2.5" => Hermes,
        "qwen3.5" | "qwen3-coder" => Qwen3Xml,
        f if f.starts_with("llama3") => Llama3Json,
        "mistral-v0.3" => Mistral,
        f if f.starts_with("glm") => Glm45,
        "kimi-k2" => KimiK2,
        "gpt-oss" => Harmony,
        // Their templates never render `tools`.
        "deepseek-v3.1" | "mixtral" => return ToolSupport::None,
        f => panic!("no expectation for fixture family {f}"),
    })
}

#[test]
fn every_family_renders_tool_conversations_like_transformers() {
    let mut families: Vec<_> = std::fs::read_dir(fixtures()).expect("fixtures").map(|e| e.unwrap().path()).collect();
    families.sort();
    assert!(families.len() >= 14, "fixture families missing: {families:?}");
    let mut id_checked = 0;
    for dir in families {
        let family = dir.file_name().unwrap().to_str().unwrap().to_string();
        let t = ChatTemplate::load(&dir).unwrap_or_else(|| panic!("{family}: template compiles"));
        assert_eq!(t.tools, expected_support(&family), "{family}: tool support");
        let fx: Value = serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
        #[cfg(feature = "hf-tokenizer")]
        let tok = fx["tokenizer"]
            .as_str()
            .map(Path::new)
            .filter(|p| p.exists())
            .map(|p| crate::text::tokenizer::HfTokenizer::from_file(p).expect("tokenizer loads"));
        for case in fx["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let mut body = case["request"].clone();
            body["model"] = "m".into();
            let req: ChatRequest = serde_json::from_value(body).unwrap();
            let opts = RenderOpts { tools: req.tools.clone(), ..Default::default() };
            let got = render(&t, &req.messages, &opts);
            match (case.get("expected").and_then(Value::as_str), got) {
                (Some(want), Ok(mut got)) => {
                    // `strftime_now` renders today; the fixture holds the day it was captured.
                    let dated = |s: &str| {
                        s.split('\n').find(|l| l.starts_with("Current date: ") || l.starts_with("Today Date: ")).map(str::to_owned)
                    };
                    if let (Some(today), Some(captured)) = (dated(&got), dated(want)) {
                        got = got.replacen(&today, &captured, 1);
                    }
                    assert!(got == want, "{family}/{name}: render differs\n--- plowrt\n{got}\n--- transformers\n{want}");
                    #[cfg(feature = "hf-tokenizer")]
                    if let (Some(tok), Some(ids)) = (&tok, case.get("ids")) {
                        use crate::text::tokenizer::Tokenize;
                        let want: Vec<u32> = serde_json::from_value(ids.clone()).unwrap();
                        assert_eq!(tok.encode(&got), want, "{family}/{name}: token ids");
                        id_checked += 1;
                    }
                }
                (None, Err(RenderError::Template(_))) => {}
                (want, got) => panic!("{family}/{name}: transformers {want:?} vs plowrt {got:?} ({:?})", case.get("error")),
            }
        }
    }
    eprintln!("token ids compared for {id_checked} renders");
}
