//! `/v1/chat/completions` end to end over the production router, CPU only: a bundle carrying
//! Gemma 4's own chat template and a byte-level tokenizer with its special tokens, answered by a
//! scripted dispatcher (`mux::scripted_mux`) that replays model text. Covers tools with and
//! without streaming, parallel calls, `tool_choice` modes, `strict`, logprobs on tool deltas,
//! reasoning separation and a multi-turn tool loop.

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::serve::{app, AppState};

const SPECIALS: &[&str] = &[
    "<bos>", "<eos>", "<pad>", "<|turn>", "<turn|>", "<|tool_call>", "<tool_call|>", "<|\"|>", "<|channel>", "<channel|>",
    "<|tool_response>", "<tool_response|>", "<|tool>", "<tool|>", "<|think|>",
];

/// GPT-2's byte-to-unicode table, the vocabulary of a byte-level BPE with no merges.
fn byte_char(b: u8) -> char {
    let printable = |b: u8| (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    if printable(b) {
        return char::from(b);
    }
    let n = (0..b).filter(|&x| !printable(x)).count() as u32;
    char::from_u32(256 + n).unwrap()
}

fn tokenizer_json() -> Value {
    let vocab: serde_json::Map<String, Value> = (0..=255u8).map(|b| (byte_char(b).to_string(), json!(b))).collect();
    let added: Vec<Value> = SPECIALS
        .iter()
        .enumerate()
        .map(|(i, t)| json!({"id": 256 + i, "content": t, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}))
        .collect();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added, "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null, "end_of_word_suffix": null,
                  "fuse_unk": false, "byte_fallback": false, "vocab": vocab, "merges": []}
    })
}

/// What the scripted model saw and should say next.
#[derive(Default)]
struct Script {
    prompts: Vec<String>,
    reply: Vec<String>,
}

struct Harness {
    app: axum::Router,
    script: Arc<Mutex<Script>>,
    _dir: TempDir,
}

struct TempDir(std::path::PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn harness() -> Harness {
    static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt-tool-http-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let fx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toolcall/gemma4-e4b");
    std::fs::copy(fx.join("chat_template.jinja"), dir.join("chat_template.jinja")).unwrap();
    std::fs::copy(fx.join("tokenizer_config.json"), dir.join("tokenizer_config.json")).unwrap();
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json().to_string()).unwrap();
    std::fs::write(
        dir.join("weights.json"),
        r#"{"network":"toolm","gpu":"cpu","num_gpus":1,"parallel":"none","weight_shared":false,"buckets":[]}"#,
    )
    .unwrap();
    let backend: Arc<dyn crate::device::Backend> = Arc::new(crate::device::cpu::CpuBackend::new(1));
    let registry = crate::orch::Registry::new();
    let slug = registry.load(&dir, None).unwrap();
    let bundle = registry.get(&slug).unwrap();
    let state = Arc::new(AppState::new(registry, Arc::new(crate::exec::ExecutorSet::bringup(backend).unwrap())));
    state.scripted.write().push(slug.clone());
    let script = Arc::new(Mutex::new(Script::default()));
    let s = Arc::clone(&script);
    state.install_mux(
        slug,
        crate::serve::mux::scripted_mux(move |job| {
            let mut text = String::new();
            bundle.tokenizer().decode_append(&job.prompt_ids, true, &mut text);
            let mut s = s.lock().unwrap();
            s.prompts.push(text);
            s.reply.clone()
        }),
    );
    Harness { app: app(state), script, _dir: TempDir(dir) }
}

impl Harness {
    fn reply(&self, pieces: &[&str]) {
        self.script.lock().unwrap().reply = pieces.iter().map(|p| p.to_string()).collect();
    }

    fn last_prompt(&self) -> String {
        self.script.lock().unwrap().prompts.last().cloned().unwrap_or_default()
    }

    async fn post(&self, mut body: Value) -> (StatusCode, String) {
        body["model"] = json!("toolm");
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = self.app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    async fn json(&self, body: Value) -> (StatusCode, Value) {
        let (code, text) = self.post(body).await;
        (code, serde_json::from_str(&text).unwrap_or_else(|_| panic!("not JSON: {text}")))
    }

    /// The `data:` payloads of an SSE body, and whether it ended with `[DONE]`.
    async fn sse(&self, mut body: Value) -> (Vec<Value>, bool) {
        body["stream"] = json!(true);
        let (code, text) = self.post(body).await;
        assert_eq!(code, StatusCode::OK, "{text}");
        let mut frames = Vec::new();
        let mut done = false;
        for line in text.lines() {
            let Some(d) = line.strip_prefix("data: ") else { continue };
            if d == "[DONE]" {
                done = true;
            } else {
                frames.push(serde_json::from_str(d).unwrap());
            }
        }
        (frames, done)
    }
}

fn tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
         "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
        {"type": "function", "function": {"name": "search", "description": "Web search.",
         "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}
    ])
}

fn user(q: &str) -> Value {
    json!([{"role": "user", "content": q}])
}

/// Split model text into token-like pieces (every marker whole, 3-byte runs between).
fn toks(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < text.len() {
        if let Some(m) = SPECIALS.iter().find(|m| text[i..].starts_with(**m)) {
            out.push(m.to_string());
            i += m.len();
            continue;
        }
        let mut j = (i + 3).min(text.len());
        while !text.is_char_boundary(j) {
            j += 1;
        }
        if let Some(k) = SPECIALS.iter().filter_map(|m| text[i..j].find(m)).min() {
            j = i + k.max(1);
        }
        out.push(text[i..j].to_string());
        i = j;
    }
    out
}

/// Accumulate streamed tool-call deltas by index, as the OpenAI SDK does.
fn accumulate(frames: &[Value]) -> (String, String, Vec<(String, String, String)>, Option<String>) {
    let (mut content, mut reasoning, mut calls, mut finish) = (String::new(), String::new(), Vec::<(String, String, String)>::new(), None);
    for f in frames {
        let Some(c) = f["choices"].get(0) else { continue };
        let d = &c["delta"];
        content.push_str(d["content"].as_str().unwrap_or_default());
        reasoning.push_str(d["reasoning_content"].as_str().unwrap_or_default());
        for t in d["tool_calls"].as_array().into_iter().flatten() {
            let i = t["index"].as_u64().unwrap() as usize;
            if let Some(id) = t["id"].as_str() {
                assert_eq!(i, calls.len(), "a call's first delta carries its id: {t}");
                assert_eq!(t["type"], "function");
                calls.push((id.to_string(), t["function"]["name"].as_str().unwrap().to_string(), String::new()));
            }
            calls[i].2.push_str(t["function"]["arguments"].as_str().unwrap_or_default());
        }
        if let Some(r) = c["finish_reason"].as_str() {
            finish = Some(r.to_string());
        }
    }
    (content, reasoning, calls, finish)
}

const CALL: &str = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:2}<tool_call|>";
const CALL2: &str = "<|tool_call>call:search{q:<|\"|>flights to Zürich<|\"|>}<tool_call|>";

#[tokio::test]
async fn streamed_tool_calls_arrive_as_incremental_argument_deltas() {
    let h = harness();
    let text = format!("Let me check.{CALL}{CALL2}");
    let pieces = toks(&text);
    h.reply(&pieces.iter().map(String::as_str).collect::<Vec<_>>());
    let (frames, done) = h.sse(json!({"messages": user("Weather in Paris and flights?"), "tools": tools(),
        "stream_options": {"include_usage": true}})).await;
    assert!(done);
    assert!(h.last_prompt().contains("<|tool>declaration:get_weather"), "the template rendered the tools");
    let (content, _, calls, finish) = accumulate(&frames);
    assert_eq!(content, "Let me check.");
    assert_eq!(finish.as_deref(), Some("tool_calls"));
    assert_eq!(calls.len(), 2);
    assert_eq!((calls[0].1.as_str(), serde_json::from_str::<Value>(&calls[0].2).unwrap()), ("get_weather", json!({"city": "Paris", "days": 2})));
    assert_eq!(serde_json::from_str::<Value>(&calls[1].2).unwrap(), json!({"q": "flights to Zürich"}));
    assert!(calls.iter().all(|c| c.0.starts_with("call_") && c.0.len() == 29));
    let arg_frames = frames.iter().filter(|f| f["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str().is_some_and(|a| !a.is_empty())).count();
    assert!(arg_frames >= 6, "arguments stream in several deltas, not one per call ({arg_frames})");
    // every frame is one chat.completion.chunk of the same stream, usage last
    let id = frames[0]["id"].clone();
    assert!(frames.iter().all(|f| f["id"] == id && f["object"] == "chat.completion.chunk"));
    assert_eq!(frames[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(frames.last().unwrap()["usage"]["completion_tokens"], json!(pieces.len()));
    assert!(!frames.iter().any(|f| f.to_string().contains("tool_call>")), "markers never reach the client");
}

#[tokio::test]
async fn buffered_parallel_calls_and_parallel_tool_calls_false() {
    let h = harness();
    let text = format!("{CALL}\n{CALL2}");
    h.reply(&toks(&text).iter().map(String::as_str).collect::<Vec<_>>());
    let (code, v) = h.json(json!({"messages": user("both"), "tools": tools()})).await;
    assert_eq!(code, StatusCode::OK, "{v}");
    let m = &v["choices"][0]["message"];
    assert_eq!(m["content"], Value::Null, "a turn that is only calls answers null content");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let calls = m["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(serde_json::from_str::<Value>(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap(), json!({"city": "Paris", "days": 2}));
    let (_, v) = h.json(json!({"messages": user("both"), "tools": tools(), "parallel_tool_calls": false})).await;
    assert_eq!(v["choices"][0]["message"]["tool_calls"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn tool_choice_none_required_and_named() {
    let h = harness();
    h.reply(&["It", " is", " sunny."]);
    let (code, v) = h.json(json!({"messages": user("weather?"), "tools": tools(), "tool_choice": "none"})).await;
    assert_eq!(code, StatusCode::OK);
    assert!(!h.last_prompt().contains("declaration:get_weather"), "none renders no tools");
    assert_eq!(v["choices"][0]["message"]["content"], "It is sunny.");
    assert!(v["choices"][0]["message"].get("tool_calls").is_none());

    // required: the prompt opens a call and the model writes the rest
    h.reply(&["search{q:", "<|\"|>", "x", "<|\"|>", "}", "<tool_call|>"]);
    let (code, v) = h.json(json!({"messages": user("anything"), "tools": tools(), "tool_choice": "required"})).await;
    assert_eq!(code, StatusCode::OK, "{v}");
    assert!(h.last_prompt().ends_with("<|turn>model\n<|tool_call>call:"), "{}", h.last_prompt());
    assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "search");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");

    // named, streamed: the head rides the first frame, before the model wrote anything of it
    h.reply(&["city:", "<|\"|>", "Oslo", "<|\"|>", "}", "<tool_call|>"]);
    let named = json!({"type": "function", "function": {"name": "get_weather"}});
    let (frames, done) = h.sse(json!({"messages": user("weather"), "tools": tools(), "tool_choice": named})).await;
    assert!(done && h.last_prompt().ends_with("<|tool_call>call:get_weather{"));
    assert_eq!(frames[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "get_weather");
    let (_, _, calls, finish) = accumulate(&frames);
    assert_eq!((calls.len(), finish.as_deref()), (1, Some("tool_calls")));
    assert_eq!(serde_json::from_str::<Value>(&calls[0].2).unwrap(), json!({"city": "Oslo"}));

    // a forced call the model's output does not complete as a declared one fails the turn
    h.reply(&["launch_rockets{}", "<tool_call|>"]);
    let (code, v) = h.json(json!({"messages": user("x"), "tools": tools(), "tool_choice": "required"})).await;
    assert_eq!(code, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(v["error"]["code"], "invalid_tool_call");
    // refusals stay 400
    let (code, v) = h.json(json!({"messages": user("x"), "tools": tools(), "tool_choice": {"type": "function", "function": {"name": "nope"}}})).await;
    assert_eq!((code, v["error"]["param"].as_str()), (StatusCode::BAD_REQUEST, Some("tool_choice")));
}

#[tokio::test]
async fn strict_arguments_are_validated() {
    let h = harness();
    let mut t = tools();
    t[0]["function"]["strict"] = json!(true);
    t[0]["function"]["parameters"]["additionalProperties"] = json!(false);
    h.reply(&["<|tool_call>", "call:get_weather{", "town:", "<|\"|>Paris<|\"|>", "}", "<tool_call|>"]);
    let (code, v) = h.json(json!({"messages": user("w"), "tools": t})).await;
    assert_eq!(code, StatusCode::INTERNAL_SERVER_ERROR, "{v}");
    assert_eq!(v["error"]["code"], "invalid_tool_call");
    assert!(v["error"]["message"].as_str().unwrap().contains("missing required property `city`"), "{v}");
    // streamed: an error object, and no [DONE]
    let (frames, done) = h.sse(json!({"messages": user("w"), "tools": t})).await;
    assert!(!done);
    assert_eq!(frames.last().unwrap()["error"]["code"], "invalid_tool_call");
    assert!(frames.iter().all(|f| !f.to_string().contains("town")), "strict arguments are held until they validate");
    // valid strict arguments go out in one delta
    h.reply(&toks(CALL).iter().map(String::as_str).collect::<Vec<_>>());
    t[0]["function"]["parameters"]["properties"]["days"] = json!({"type": "integer", "maximum": 7});
    let (frames, done) = h.sse(json!({"messages": user("w"), "tools": t})).await;
    let (_, _, calls, _) = accumulate(&frames);
    assert!(done && calls.len() == 1);
    let arg_frames = frames.iter().filter(|f| f["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str().is_some_and(|a| !a.is_empty())).count();
    assert_eq!(arg_frames, 1);
}

#[tokio::test]
async fn logprobs_ride_tool_call_deltas() {
    let h = harness();
    h.reply(&toks(CALL).iter().map(String::as_str).collect::<Vec<_>>());
    let (frames, _) = h.sse(json!({"messages": user("w"), "tools": tools(), "logprobs": true})).await;
    let with_tools: Vec<&Value> = frames.iter().filter(|f| f["choices"][0]["delta"].get("tool_calls").is_some()).collect();
    assert!(!with_tools.is_empty());
    for f in with_tools {
        assert_eq!(f["choices"][0]["logprobs"]["content"][0]["logprob"], json!(-0.25), "{f}");
    }
    let (_, v) = h.json(json!({"messages": user("w"), "tools": tools(), "logprobs": true})).await;
    assert_eq!(v["choices"][0]["logprobs"]["content"].as_array().unwrap().len(), toks(CALL).len());
}

#[tokio::test]
async fn reasoning_is_separated_from_content() {
    let h = harness();
    let thought = ["<|channel>", "thought", "\n", "The user", " wants 2+2.", "<channel|>", "It is", " 4."];
    h.reply(&thought);
    let (_, v) = h.json(json!({"messages": user("2+2?")})).await;
    let m = &v["choices"][0]["message"];
    assert_eq!((m["content"].as_str(), m["reasoning_content"].as_str()), (Some("It is 4."), Some("The user wants 2+2.")));
    assert_eq!(v["usage"]["completion_tokens_details"]["reasoning_tokens"], json!(6));
    let (frames, _) = h.sse(json!({"messages": user("2+2?")})).await;
    let (content, reasoning, _, finish) = accumulate(&frames);
    assert_eq!((content.as_str(), reasoning.as_str(), finish.as_deref()), ("It is 4.", "The user wants 2+2.", Some("stop")));
    // include_reasoning: false drops the trace, on both paths
    let (_, v) = h.json(json!({"messages": user("2+2?"), "include_reasoning": false})).await;
    assert!(v["choices"][0]["message"].get("reasoning_content").is_none());
    assert_eq!(v["choices"][0]["message"]["content"], "It is 4.");
    let (frames, _) = h.sse(json!({"messages": user("2+2?"), "include_reasoning": false})).await;
    assert!(!frames.iter().any(|f| f.to_string().contains("reasoning_content")));
}

#[tokio::test]
async fn a_tool_loop_answers_without_leaking_the_thought_channel() {
    let h = harness();
    let convo = json!([
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18}"}
    ]);
    // after a tool result Gemma 4 opens an empty thought channel before answering
    h.reply(&["<|channel>", "thought", "\n", "<channel|>", "It is", " 18°C", " in Paris."]);
    let (code, v) = h.json(json!({"messages": convo, "tools": tools()})).await;
    assert_eq!(code, StatusCode::OK, "{v}");
    let p = h.last_prompt();
    assert!(p.contains("<|tool_response>") && p.ends_with("<tool_response|>"), "{p}");
    let m = &v["choices"][0]["message"];
    assert_eq!(m["content"], "It is 18°C in Paris.");
    assert!(m.get("reasoning_content").is_none() && m.get("tool_calls").is_none());
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    let (frames, _) = h.sse(json!({"messages": convo, "tools": tools()})).await;
    let (content, _, calls, _) = accumulate(&frames);
    assert_eq!((content.as_str(), calls.len()), ("It is 18°C in Paris.", 0));
}

#[tokio::test]
async fn a_call_cut_by_max_tokens_finishes_length() {
    let h = harness();
    h.reply(&["<|tool_call>", "call:get_weather{", "city:", "<|\"|>Par"]);
    let (_, v) = h.json(json!({"messages": user("w"), "tools": tools(), "max_tokens": 4})).await;
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{\"city\":\"Par");
}
