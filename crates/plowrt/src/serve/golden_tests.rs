//! Serving goldens for the production packets: the chat handler's prompt selection, rendered
//! prompt bytes, token ids, tool decisions, reasoning split and stop set for a fixed set of
//! conversations, frozen per asset directory in `tests/fixtures/serve_golden/<name>.json`.
//!
//! A packet re-emitted with new serving metadata must produce byte-identical prompts and
//! identical ids, or it needs GPU re-qualification. The fixture pins the inputs the choice is
//! made from (tokenizer digest, `serve.json` digest, template source and digest), so a golden
//! that no longer matches says which input moved.
//!
//! Needs the assets, so it runs only when `PLOW_SERVE_GOLDEN` lists them:
//! `PLOW_SERVE_GOLDEN="qwen38-bf16=/path/to/assets,gemma4-12b=/path/to/assets"`.
//! `PLOW_SERVE_GOLDEN_RECORD=1` writes the fixtures instead of checking them.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::asset::ModelBundle;
use crate::serve::chat::{chat_prompt, response_stream, PromptError};
use crate::serve::openai::ChatRequest;
use crate::serve::reasoning::ReasoningMode;
use crate::serve::template::RenderOpts;
use crate::serve::tools::request::plan;

const ENV: &str = "PLOW_SERVE_GOLDEN";
const RECORD: &str = "PLOW_SERVE_GOLDEN_RECORD";

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/serve_golden")
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_sha256(path: &Path) -> Option<String> {
    std::fs::read(path).ok().map(|b| sha256_hex(&b))
}

const WEATHER_TOOL: &str = r#"{"type": "function", "function": {"name": "get_weather",
  "description": "Current weather for a city.",
  "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
  "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}"#;

/// The conversations every production packet is frozen on. Each exercises one handler decision:
/// roles, history, tools and `tool_choice`, the template's thinking switch, reasoning effort,
/// prefix continuation. Add a case only with a new fixture recording for every packet.
fn conversations() -> Vec<(&'static str, Value)> {
    let tool: Value = serde_json::from_str(WEATHER_TOOL).unwrap();
    let user = json!({"role": "user", "content": "What is the capital of France?"});
    vec![
        ("user", json!({"messages": [user]})),
        (
            "system_user",
            json!({"messages": [
                {"role": "system", "content": "You are a terse assistant."},
                user,
            ]}),
        ),
        (
            "developer_user",
            json!({"messages": [
                {"role": "developer", "content": "Answer in one word."},
                user,
            ]}),
        ),
        (
            "multi_turn",
            json!({"messages": [
                user,
                {"role": "assistant", "content": "Paris."},
                {"role": "user", "content": "And of Spain?"},
            ]}),
        ),
        (
            "content_parts",
            json!({"messages": [
                {"role": "user", "content": [{"type": "text", "text": "Name three primes."}]},
            ]}),
        ),
        (
            "tools_auto",
            json!({"messages": [{"role": "user", "content": "Weather in Oslo?"}], "tools": [tool]}),
        ),
        (
            "tools_required",
            json!({"messages": [{"role": "user", "content": "Weather in Oslo?"}], "tools": [tool],
                   "tool_choice": "required"}),
        ),
        (
            "tools_named",
            json!({"messages": [{"role": "user", "content": "Weather in Oslo?"}], "tools": [tool],
                   "tool_choice": {"type": "function", "function": {"name": "get_weather"}}}),
        ),
        (
            "tools_none",
            json!({"messages": [{"role": "user", "content": "Weather in Oslo?"}], "tools": [tool],
                   "tool_choice": "none"}),
        ),
        (
            "tool_result",
            json!({"messages": [
                {"role": "user", "content": "Weather in Oslo?"},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "call_golden0001",
                  "type": "function", "function": {"name": "get_weather",
                  "arguments": "{\"city\": \"Oslo\"}"}}]},
                {"role": "tool", "tool_call_id": "call_golden0001", "content": "{\"temp_c\": 4}"},
            ], "tools": [tool]}),
        ),
        (
            "thinking_off",
            json!({"messages": [user], "chat_template_kwargs": {"enable_thinking": false}}),
        ),
        (
            "thinking_on",
            json!({"messages": [user], "chat_template_kwargs": {"enable_thinking": true}}),
        ),
        (
            "reasoning_effort_high",
            json!({"messages": [user], "reasoning_effort": "high"}),
        ),
        (
            "continue_final",
            json!({"messages": [user, {"role": "assistant", "content": "The capital"}],
                   "continue_final_message": true}),
        ),
        (
            "no_generation_prompt",
            json!({"messages": [user], "add_generation_prompt": false}),
        ),
    ]
}

/// What the handler decides for one conversation, as `chat_completions_with` does it.
fn replay(bundle: &ModelBundle, body: &Value) -> Value {
    let mut body = body.clone();
    body["model"] = json!("golden");
    let req: ChatRequest = serde_json::from_value(body).expect("request body");
    let tool_support = bundle.chat_template().map(|t| t.tools);
    let tool_plan = match plan(&req, tool_support) {
        Ok(p) => p,
        Err(r) => {
            return json!({"refused": {"param": r.param, "code": r.code, "message": r.message}})
        }
    };
    let render_opts = RenderOpts {
        kwargs: req.chat_template_kwargs.clone().unwrap_or_default(),
        reasoning_effort: req.reasoning_effort.clone(),
        continue_final_message: req.continue_final_message.unwrap_or(false),
        no_generation_prompt: req.add_generation_prompt == Some(false),
        tools: tool_plan.template_tools.clone(),
    };
    let mut prompt = match chat_prompt(Some(bundle), &req.messages, &render_opts) {
        Ok(p) => p,
        Err(PromptError::Template(e)) => return json!({"template_error": e}),
        Err(PromptError::Request(r)) => {
            return json!({"refused": {"param": r.param, "code": r.code, "message": r.message}})
        }
        Err(PromptError::NoTemplate(e)) => return json!({"no_template": e}),
    };
    let reasoning_mode = ReasoningMode::for_bundle(bundle);
    let mut reasoning_open = reasoning_mode.prompt_opens(&prompt);
    let thinking = reasoning_mode.markers().is_some()
        && (reasoning_open
            || bundle
                .chat_template()
                .is_some_and(|t| t.thinking_requested(&render_opts.kwargs)));
    let opener = tool_plan.parse.as_ref().and_then(|s| s.opener());
    if let Some(o) = &opener {
        if reasoning_open {
            prompt.push_str(reasoning_mode.close_marker().unwrap_or_default());
            reasoning_open = false;
        }
        prompt.push_str(o);
    }
    let tool_stream = response_stream(
        &tool_plan,
        tool_support,
        reasoning_mode,
        reasoning_open,
        bundle,
        false,
        opener.as_deref(),
    )
    .is_some();
    let ids = bundle.tokenizer().encode(&prompt);
    json!({
        "prompt": prompt,
        "ids": ids,
        "parse_format": tool_plan.parse.as_ref().map(|s| s.format.name()),
        "opener": opener,
        "reasoning_open": reasoning_open,
        "thinking": thinking,
        "tool_stream": tool_stream,
    })
}

/// The inputs the decisions are made from, so a mismatch names what moved.
fn inputs(dir: &Path, bundle: &ModelBundle) -> Value {
    let pkt = crate::asset::devblob::DevBlob::find_in_dir(dir).expect("packet lookup");
    let serve_section = pkt
        .as_ref()
        .and_then(|p| crate::asset::serve::read_section(p).expect("serve.json read"))
        .map(|b| sha256_hex(&b));
    let checkpoint = crate::asset::serve::checkpoint_dir(dir);
    let template = bundle.chat_template();
    let template_text = if bundle.serve().from_packet {
        bundle
            .serve()
            .manifest
            .chat
            .as_ref()
            .and_then(|c| c.template.clone())
    } else {
        plow_asset::serve_manifest::find_chat_template(dir).map(|t| t.text)
    };
    // Host paths out of the fixture: the source names the file relative to the asset or
    // checkpoint directory.
    let portable = |source: &str| {
        let strip = |base: &Path| {
            let b = base.to_string_lossy();
            source
                .strip_prefix(b.as_ref())
                .map(|rest| rest.trim_start_matches('/').to_string())
        };
        match (strip(dir), strip(&checkpoint)) {
            (Some(rest), _) => format!("<assets>/{rest}"),
            (None, Some(rest)) => format!("<checkpoint>/{rest}"),
            (None, None) => source.to_string(),
        }
    };
    json!({
        "tokenizer_sha256": file_sha256(&dir.join("tokenizer.json"))
            .or_else(|| file_sha256(&checkpoint.join("tokenizer.json"))),
        "serve_json_sha256": serve_section,
        "template_source": template.map(|t| portable(&t.source)),
        "template_sha256": template_text.map(|t| sha256_hex(t.as_bytes())),
    })
}

fn model_facts(bundle: &ModelBundle) -> Value {
    let serve = bundle.serve();
    let template = bundle.chat_template();
    let chat = serve.manifest.chat.as_ref();
    let mut stop_ids = serve.manifest.stop_token_ids.clone();
    stop_ids.sort_unstable();
    stop_ids.dedup();
    let empty = Default::default();
    let off = [("enable_thinking", false), ("thinking", false)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), Value::Bool(v)))
        .collect();
    json!({
        "from_packet": serve.from_packet,
        "serve_version": serve.from_packet.then_some(serve.manifest.version),
        "tool_support": template.map(|t| format!("{:?}", t.tools)),
        "reasoning": format!("{:?}", bundle.reasoning()),
        "reasoning_in_specials": bundle.reasoning_in_specials(),
        "template_reasoning": template.and_then(|t| t.reasoning),
        "thinking_knob": template.and_then(|t| t.thinking_knob),
        "thinking_default": template.map(|t| t.thinking_requested(&empty)),
        "thinking_off": template.map(|t| t.thinking_requested(&off)),
        "builtin": chat.and_then(|c| c.builtin.clone()),
        "bos_token": chat.and_then(|c| c.bos_token.clone()),
        "eos_token": chat.and_then(|c| c.eos_token.clone()),
        "stop_token_ids": stop_ids,
        "sampling": serve.manifest.sampling,
    })
}

fn snapshot(dir: &Path) -> Value {
    let bundle = ModelBundle::load(dir).expect("bundle loads");
    let cases: serde_json::Map<String, Value> = conversations()
        .iter()
        .map(|(name, body)| (name.to_string(), replay(&bundle, body)))
        .collect();
    json!({
        "inputs": inputs(dir, &bundle),
        "model": model_facts(&bundle),
        "cases": cases,
    })
}

fn assets() -> Vec<(String, PathBuf)> {
    let Ok(spec) = std::env::var(ENV) else {
        return Vec::new();
    };
    spec.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let (name, dir) = s
                .split_once('=')
                .unwrap_or_else(|| panic!("{ENV}: `{s}` is not name=dir"));
            (name.trim().to_string(), PathBuf::from(dir.trim()))
        })
        .collect()
}

#[test]
fn production_packets_match_their_serve_goldens() {
    let assets = assets();
    if assets.is_empty() {
        eprintln!("skipped: {ENV} names no asset directories");
        return;
    }
    let record = std::env::var_os(RECORD).is_some();
    let mut failures = Vec::new();
    for (name, dir) in assets {
        let got = snapshot(&dir);
        let path = fixtures().join(format!("{name}.json"));
        if record {
            std::fs::create_dir_all(fixtures()).unwrap();
            let mut text = serde_json::to_string_pretty(&got).unwrap();
            text.push('\n');
            std::fs::write(&path, text).unwrap();
            eprintln!("recorded {}", path.display());
            continue;
        }
        let want: Value = serde_json::from_slice(
            &std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
        )
        .unwrap();
        if got != want {
            let actual = path.with_extension("actual.json");
            std::fs::write(&actual, serde_json::to_string_pretty(&got).unwrap()).unwrap();
            let mut diffs = Vec::new();
            for key in ["inputs", "model"] {
                if got[key] != want[key] {
                    diffs.push(format!("{key}: {} != {}", got[key], want[key]));
                }
            }
            for (case, want_case) in want["cases"].as_object().into_iter().flatten() {
                if got["cases"][case] != *want_case {
                    diffs.push(format!("case {case} differs"));
                }
            }
            failures.push(format!(
                "{name}: golden {} differs (actual written to {}):\n  {}",
                path.display(),
                actual.display(),
                diffs.join("\n  ")
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The fixture set itself is well-formed: every checked-in golden covers every conversation.
#[test]
fn checked_in_goldens_cover_every_conversation() {
    let Ok(entries) = std::fs::read_dir(fixtures()) else {
        return;
    };
    let names: Vec<_> = conversations().into_iter().map(|(n, _)| n).collect();
    for e in entries {
        let path = e.unwrap().path();
        if path.extension().is_some_and(|x| x == "json")
            && !path.to_string_lossy().ends_with(".actual.json")
        {
            let v: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            for n in &names {
                assert!(
                    v["cases"].get(n).is_some(),
                    "{}: no case {n}; re-record it",
                    path.display()
                );
            }
        }
    }
}
