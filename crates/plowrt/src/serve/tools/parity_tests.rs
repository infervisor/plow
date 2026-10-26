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
        "hermes3" => Hermes,
        // vLLM's tool templates for the DeepSeek checkpoints.
        "deepseek-v3-tools" | "deepseek-r1-tools" => DeepSeekV3,
        "deepseek-v3.1-tools" => DeepSeekV31,
        // Their own templates never render `tools`.
        f if f.starts_with("deepseek") || f == "mixtral" => return ToolSupport::None,
        f => panic!("no expectation for fixture family {f}"),
    })
}

#[test]
fn every_family_renders_tool_conversations_like_transformers() {
    let mut families: Vec<_> = std::fs::read_dir(fixtures())
        .expect("fixtures")
        .map(|e| e.unwrap().path())
        .collect();
    families.sort();
    assert!(
        families.len() >= 14,
        "fixture families missing: {families:?}"
    );
    let mut id_checked = 0;
    for dir in families {
        let family = dir.file_name().unwrap().to_str().unwrap().to_string();
        let t = crate::serve::template::load(&dir)
            .unwrap_or_else(|| panic!("{family}: template compiles"));
        assert_eq!(t.tools, expected_support(&family), "{family}: tool support");
        let fx: Value =
            serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
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
            let opts = RenderOpts {
                tools: req.tools.clone(),
                ..Default::default()
            };
            let got = render(&t, &req.messages, &opts);
            match (case.get("expected").and_then(Value::as_str), got) {
                (Some(want), Ok(mut got)) => {
                    // `strftime_now` renders today; the fixture holds the day it was captured.
                    let dated = |s: &str| {
                        s.split('\n')
                            .find(|l| {
                                l.starts_with("Current date: ") || l.starts_with("Today Date: ")
                            })
                            .map(str::to_owned)
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
                (want, got) => panic!(
                    "{family}/{name}: transformers {want:?} vs plowrt {got:?} ({:?})",
                    case.get("error")
                ),
            }
        }
    }
    eprintln!("token ids compared for {id_checked} renders");
}

/// What the compiler declares for a family (version-2 `serve.json`), serialized, parsed back and
/// compiled from the declaration alone, decides tools, openers and reasoning exactly as the
/// version-1 server's probes did for the same template.
#[test]
fn emitted_declarations_reload_to_the_probed_decisions() {
    use crate::serve::reasoning::ReasoningMode;
    use plow_asset::serve_manifest::ServeManifest;
    let mut families: Vec<_> = std::fs::read_dir(fixtures())
        .expect("fixtures")
        .map(|e| e.unwrap().path())
        .collect();
    families.sort();
    for dir in families {
        let family = dir.file_name().unwrap().to_str().unwrap().to_string();
        let probed = crate::serve::template::load(&dir)
            .unwrap_or_else(|| panic!("{family}: template compiles"));
        let emitted = ServeManifest::emit(&dir, &dir).unwrap_or_else(|e| panic!("{family}: {e}"));
        let reloaded = ServeManifest::parse(&emitted.to_json()).unwrap();
        assert_eq!(reloaded, emitted, "{family}: serialize/parse");
        let chat = reloaded.chat.as_ref().unwrap();
        let response = chat.response.as_ref().unwrap();
        let declared = ChatTemplate::compile_declared(
            chat.template.clone().unwrap(),
            "declared".into(),
            chat.bos_token.clone(),
            chat.eos_token.clone(),
            response.tool_support().unwrap(),
            ReasoningMode::declared(&response.reasoning).markers(),
        )
        .unwrap();
        assert_eq!(declared.tools, probed.tools, "{family}: tool support");
        assert_eq!(
            declared.tools,
            expected_support(&family),
            "{family}: expectation"
        );
        // Request planning from the reloaded declaration: tool requests are accepted or refused
        // as before, with the same forced-call opener bytes.
        for choice in [
            serde_json::json!("auto"),
            serde_json::json!("required"),
            serde_json::json!({"type": "function", "function": {"name": "get_weather"}}),
        ] {
            let body = serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "Weather in Oslo?"}],
                "tools": [WEATHER_TOOL.clone()], "tool_choice": choice});
            let req: ChatRequest = serde_json::from_value(body).unwrap();
            let (want, got) = (
                super::request::plan(&req, Some(probed.tools)),
                super::request::plan(&req, Some(declared.tools)),
            );
            match (want, got) {
                (Ok(want), Ok(got)) => {
                    assert_eq!(
                        want.template_tools, got.template_tools,
                        "{family}: template tools"
                    );
                    let opener =
                        |p: &super::request::Plan| p.parse.as_ref().and_then(|s| s.opener());
                    assert_eq!(opener(&want), opener(&got), "{family}/{choice}: opener");
                    if choice != "auto" {
                        assert!(
                            opener(&got).is_some_and(|o| !o.is_empty()),
                            "{family}/{choice}"
                        );
                    }
                }
                (Err(want), Err(got)) => assert_eq!(want.code, got.code, "{family}/{choice}"),
                (want, got) => panic!("{family}/{choice}: probed {want:?} vs declared {got:?}"),
            }
        }
        // Every fixture conversation renders the same bytes through the declared template.
        let fx: Value =
            serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
        for case in fx["cases"].as_array().unwrap() {
            let mut body = case["request"].clone();
            body["model"] = "m".into();
            let req: ChatRequest = serde_json::from_value(body).unwrap();
            let opts = RenderOpts {
                tools: req.tools.clone(),
                ..Default::default()
            };
            let (a, b) = (
                render(&probed, &req.messages, &opts),
                render(&declared, &req.messages, &opts),
            );
            assert_eq!(a.is_ok(), b.is_ok(), "{family}/{}", case["name"]);
            if let (Ok(a), Ok(b)) = (a, b) {
                assert_eq!(a, b, "{family}/{}: rendered bytes", case["name"]);
            }
        }
        // The version-1 server's precedence for the same template and files.
        let v1 = crate::asset::serve::ServeInfo {
            manifest: ServeManifest::from_checkpoint(&dir, &dir),
            multimodal: None,
            from_packet: true,
        };
        assert_eq!(
            ReasoningMode::declared(&response.reasoning),
            ReasoningMode::resolve(&v1, probed.reasoning),
            "{family}: reasoning"
        );
        assert_eq!(
            declared.thinking_knob, probed.thinking_knob,
            "{family}: thinking knob"
        );
        // The decision for families a legacy (manifest-less) packet forced `<think>` splitting on:
        // harmony frames its trace in channels its parser reads, and Kimi-K2 writes none the
        // template shows; neither declares markers (their fixtures carry no tokenizer file, so
        // the `<think>` tokens a real K3 vocabulary has are not in play here).
        if family == "gpt-oss" {
            assert_eq!(response.tools, "harmony");
            assert_eq!(
                response.reasoning,
                plow_asset::serve_manifest::ReasoningDecl::None
            );
        }
        if family == "kimi-k2" {
            assert_eq!(
                response.reasoning,
                plow_asset::serve_manifest::ReasoningDecl::None
            );
        }
    }
}

static WEATHER_TOOL: std::sync::LazyLock<Value> = std::sync::LazyLock::new(|| {
    serde_json::json!({"type": "function", "function": {"name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})
});
