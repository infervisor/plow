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
    "<bos>",
    "<eos>",
    "<pad>",
    "<|turn>",
    "<turn|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|\"|>",
    "<|channel>",
    "<channel|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|tool>",
    "<tool|>",
    "<|think|>",
];

/// GPT-2's byte-to-unicode table, the vocabulary of a byte-level BPE with no merges.
fn byte_char(b: u8) -> char {
    let printable = |b: u8| {
        (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b)
    };
    if printable(b) {
        return char::from(b);
    }
    let n = (0..b).filter(|&x| !printable(x)).count() as u32;
    char::from_u32(256 + n).unwrap()
}

fn tokenizer_json() -> Value {
    let vocab: serde_json::Map<String, Value> = (0..=255u8)
        .map(|b| (byte_char(b).to_string(), json!(b)))
        .collect();
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
    /// Replies for the next jobs, before `reply`.
    queue: std::collections::VecDeque<Vec<String>>,
}

struct Harness {
    app: axum::Router,
    bundle: Arc<crate::asset::ModelBundle>,
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
    harness_for("gemma4-e4b")
}

/// `fixture`: a `tests/fixtures/toolcall` family (`gemma4-12b` carries the 26B's template).
fn harness_for(fixture: &str) -> Harness {
    static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt-tool-http-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let fx = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/toolcall")
        .join(fixture);
    std::fs::copy(
        fx.join("chat_template.jinja"),
        dir.join("chat_template.jinja"),
    )
    .unwrap();
    std::fs::copy(
        fx.join("tokenizer_config.json"),
        dir.join("tokenizer_config.json"),
    )
    .unwrap();
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
    let state = Arc::new(AppState::new(
        registry,
        Arc::new(crate::exec::ExecutorSet::bringup(backend).unwrap()),
    ));
    state.scripted.write().push(slug.clone());
    let model = Arc::clone(&bundle);
    let script = Arc::new(Mutex::new(Script::default()));
    let s = Arc::clone(&script);
    state.install_mux(
        slug,
        crate::serve::mux::scripted_mux(move |job| {
            let mut text = String::new();
            bundle
                .tokenizer()
                .decode_append(&job.prompt_ids, true, &mut text);
            let mut s = s.lock().unwrap();
            s.prompts.push(text);
            s.queue.pop_front().unwrap_or_else(|| s.reply.clone())
        }),
    );
    Harness {
        app: app(state),
        bundle: model,
        script,
        _dir: TempDir(dir),
    }
}

impl Harness {
    fn reply(&self, pieces: &[&str]) {
        self.script.lock().unwrap().reply = pieces.iter().map(|p| p.to_string()).collect();
    }

    fn last_prompt(&self) -> String {
        self.script
            .lock()
            .unwrap()
            .prompts
            .last()
            .cloned()
            .unwrap_or_default()
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
        (
            code,
            serde_json::from_str(&text).unwrap_or_else(|_| panic!("not JSON: {text}")),
        )
    }

    /// The `data:` payloads of an SSE body, and whether it ended with `[DONE]`.
    async fn sse(&self, mut body: Value) -> (Vec<Value>, bool) {
        body["stream"] = json!(true);
        let (code, text) = self.post(body).await;
        assert_eq!(code, StatusCode::OK, "{text}");
        let mut frames = Vec::new();
        let mut done = false;
        for line in text.lines() {
            let Some(d) = line.strip_prefix("data: ") else {
                continue;
            };
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
fn accumulate(
    frames: &[Value],
) -> (
    String,
    String,
    Vec<(String, String, String)>,
    Option<String>,
) {
    let (mut content, mut reasoning, mut calls, mut finish) = (
        String::new(),
        String::new(),
        Vec::<(String, String, String)>::new(),
        None,
    );
    for f in frames {
        let Some(c) = f["choices"].get(0) else {
            continue;
        };
        let d = &c["delta"];
        content.push_str(d["content"].as_str().unwrap_or_default());
        reasoning.push_str(d["reasoning"].as_str().unwrap_or_default());
        for t in d["tool_calls"].as_array().into_iter().flatten() {
            let i = t["index"].as_u64().unwrap() as usize;
            if let Some(id) = t["id"].as_str() {
                assert_eq!(i, calls.len(), "a call's first delta carries its id: {t}");
                assert_eq!(t["type"], "function");
                calls.push((
                    id.to_string(),
                    t["function"]["name"].as_str().unwrap().to_string(),
                    String::new(),
                ));
            }
            calls[i]
                .2
                .push_str(t["function"]["arguments"].as_str().unwrap_or_default());
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
    let (frames, done) = h
        .sse(
            json!({"messages": user("Weather in Paris and flights?"), "tools": tools(),
        "stream_options": {"include_usage": true}}),
        )
        .await;
    assert!(done);
    assert!(
        h.last_prompt().contains("<|tool>declaration:get_weather"),
        "the template rendered the tools"
    );
    let (content, _, calls, finish) = accumulate(&frames);
    assert_eq!(content, "Let me check.");
    assert_eq!(finish.as_deref(), Some("tool_calls"));
    assert_eq!(calls.len(), 2);
    assert_eq!(
        (
            calls[0].1.as_str(),
            serde_json::from_str::<Value>(&calls[0].2).unwrap()
        ),
        ("get_weather", json!({"city": "Paris", "days": 2}))
    );
    assert_eq!(
        serde_json::from_str::<Value>(&calls[1].2).unwrap(),
        json!({"q": "flights to Zürich"})
    );
    assert!(calls
        .iter()
        .all(|c| c.0.starts_with("call_") && c.0.len() == 29));
    let arg_frames = frames
        .iter()
        .filter(|f| {
            f["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .is_some_and(|a| !a.is_empty())
        })
        .count();
    assert!(
        arg_frames >= 6,
        "arguments stream in several deltas, not one per call ({arg_frames})"
    );
    // every frame is one chat.completion.chunk of the same stream, usage last
    let id = frames[0]["id"].clone();
    assert!(frames
        .iter()
        .all(|f| f["id"] == id && f["object"] == "chat.completion.chunk"));
    assert_eq!(frames[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(
        frames.last().unwrap()["usage"]["completion_tokens"],
        json!(pieces.len())
    );
    assert!(
        !frames.iter().any(|f| f.to_string().contains("tool_call>")),
        "markers never reach the client"
    );
}

#[tokio::test]
async fn buffered_parallel_calls_and_parallel_tool_calls_false() {
    let h = harness();
    let text = format!("{CALL}\n{CALL2}");
    h.reply(&toks(&text).iter().map(String::as_str).collect::<Vec<_>>());
    let (code, v) = h
        .json(json!({"messages": user("both"), "tools": tools()}))
        .await;
    assert_eq!(code, StatusCode::OK, "{v}");
    let m = &v["choices"][0]["message"];
    assert_eq!(
        m["content"],
        Value::Null,
        "a turn that is only calls answers null content"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let calls = m["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(
        serde_json::from_str::<Value>(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap(),
        json!({"city": "Paris", "days": 2})
    );
    let (_, v) = h
        .json(json!({"messages": user("both"), "tools": tools(), "parallel_tool_calls": false}))
        .await;
    assert_eq!(
        v["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn tool_choice_none_required_and_named() {
    let h = harness();
    h.reply(&["It", " is", " sunny."]);
    let (code, v) = h
        .json(json!({"messages": user("weather?"), "tools": tools(), "tool_choice": "none"}))
        .await;
    assert_eq!(code, StatusCode::OK);
    assert!(
        !h.last_prompt().contains("declaration:get_weather"),
        "none renders no tools"
    );
    assert_eq!(v["choices"][0]["message"]["content"], "It is sunny.");
    assert!(v["choices"][0]["message"].get("tool_calls").is_none());

    // required: the prompt opens a call and the model writes the rest
    h.reply(&["search{q:", "<|\"|>", "x", "<|\"|>", "}", "<tool_call|>"]);
    let (code, v) = h
        .json(json!({"messages": user("anything"), "tools": tools(), "tool_choice": "required"}))
        .await;
    assert_eq!(code, StatusCode::OK, "{v}");
    assert!(
        h.last_prompt().ends_with("<|turn>model\n<|tool_call>call:"),
        "{}",
        h.last_prompt()
    );
    assert_eq!(
        v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "search"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");

    // named, streamed: the head rides the first frame, before the model wrote anything of it
    h.reply(&["city:", "<|\"|>", "Oslo", "<|\"|>", "}", "<tool_call|>"]);
    let named = json!({"type": "function", "function": {"name": "get_weather"}});
    let (frames, done) = h
        .sse(json!({"messages": user("weather"), "tools": tools(), "tool_choice": named}))
        .await;
    assert!(done && h.last_prompt().ends_with("<|tool_call>call:get_weather{"));
    assert_eq!(
        frames[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    let (_, _, calls, finish) = accumulate(&frames);
    assert_eq!((calls.len(), finish.as_deref()), (1, Some("tool_calls")));
    assert_eq!(
        serde_json::from_str::<Value>(&calls[0].2).unwrap(),
        json!({"city": "Oslo"})
    );

    // a forced call the model's output does not complete as a declared one fails the turn
    h.reply(&["launch_rockets{}", "<tool_call|>"]);
    let (code, v) = h
        .json(json!({"messages": user("x"), "tools": tools(), "tool_choice": "required"}))
        .await;
    assert_eq!(code, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(v["error"]["code"], "invalid_tool_call");
    // refusals stay 400
    let (code, v) = h.json(json!({"messages": user("x"), "tools": tools(), "tool_choice": {"type": "function", "function": {"name": "nope"}}})).await;
    assert_eq!(
        (code, v["error"]["param"].as_str()),
        (StatusCode::BAD_REQUEST, Some("tool_choice"))
    );
}

#[tokio::test]
async fn strict_arguments_are_validated() {
    let h = harness();
    let mut t = tools();
    t[0]["function"]["strict"] = json!(true);
    t[0]["function"]["parameters"]["additionalProperties"] = json!(false);
    h.reply(&[
        "<|tool_call>",
        "call:get_weather{",
        "town:",
        "<|\"|>Paris<|\"|>",
        "}",
        "<tool_call|>",
    ]);
    let (code, v) = h.json(json!({"messages": user("w"), "tools": t})).await;
    assert_eq!(code, StatusCode::INTERNAL_SERVER_ERROR, "{v}");
    assert_eq!(v["error"]["code"], "invalid_tool_call");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("missing required property `city`"),
        "{v}"
    );
    // streamed: an error object, and no [DONE]
    let (frames, done) = h.sse(json!({"messages": user("w"), "tools": t})).await;
    assert!(!done);
    assert_eq!(frames.last().unwrap()["error"]["code"], "invalid_tool_call");
    assert!(
        frames.iter().all(|f| !f.to_string().contains("town")),
        "strict arguments are held until they validate"
    );
    // valid strict arguments go out in one delta
    h.reply(&toks(CALL).iter().map(String::as_str).collect::<Vec<_>>());
    t[0]["function"]["parameters"]["properties"]["days"] = json!({"type": "integer", "maximum": 7});
    let (frames, done) = h.sse(json!({"messages": user("w"), "tools": t})).await;
    let (_, _, calls, _) = accumulate(&frames);
    assert!(done && calls.len() == 1);
    let arg_frames = frames
        .iter()
        .filter(|f| {
            f["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .is_some_and(|a| !a.is_empty())
        })
        .count();
    assert_eq!(arg_frames, 1);
}

#[cfg(any(feature = "cuda", feature = "cpu"))]
#[tokio::test]
async fn logprobs_ride_tool_call_deltas() {
    let h = harness();
    h.reply(&toks(CALL).iter().map(String::as_str).collect::<Vec<_>>());
    let (frames, _) = h
        .sse(json!({"messages": user("w"), "tools": tools(), "logprobs": true}))
        .await;
    let with_tools: Vec<&Value> = frames
        .iter()
        .filter(|f| f["choices"][0]["delta"].get("tool_calls").is_some())
        .collect();
    assert!(!with_tools.is_empty());
    for f in with_tools {
        assert_eq!(
            f["choices"][0]["logprobs"]["content"][0]["logprob"],
            json!(-0.25),
            "{f}"
        );
    }
    let (_, v) = h
        .json(json!({"messages": user("w"), "tools": tools(), "logprobs": true}))
        .await;
    assert_eq!(
        v["choices"][0]["logprobs"]["content"]
            .as_array()
            .unwrap()
            .len(),
        toks(CALL).len()
    );
}

#[tokio::test]
async fn reasoning_is_separated_from_content() {
    let h = harness();
    let thought = [
        "<|channel>",
        "thought",
        "\n",
        "The user",
        " wants 2+2.",
        "<channel|>",
        "It is",
        " 4.",
    ];
    h.reply(&thought);
    let (_, v) = h.json(json!({"messages": user("2+2?")})).await;
    let m = &v["choices"][0]["message"];
    assert_eq!(
        (m["content"].as_str(), m["reasoning"].as_str()),
        (Some("It is 4."), Some("The user wants 2+2."))
    );
    assert_eq!(
        v["usage"]["completion_tokens_details"]["reasoning_tokens"],
        json!(6)
    );
    let (frames, _) = h.sse(json!({"messages": user("2+2?")})).await;
    let (content, reasoning, _, finish) = accumulate(&frames);
    assert_eq!(
        (content.as_str(), reasoning.as_str(), finish.as_deref()),
        ("It is 4.", "The user wants 2+2.", Some("stop"))
    );
    // include_reasoning: false drops the trace, on both paths
    let (_, v) = h
        .json(json!({"messages": user("2+2?"), "include_reasoning": false}))
        .await;
    assert!(v["choices"][0]["message"].get("reasoning").is_none());
    assert_eq!(v["choices"][0]["message"]["content"], "It is 4.");
    let (frames, _) = h
        .sse(json!({"messages": user("2+2?"), "include_reasoning": false}))
        .await;
    assert!(!frames
        .iter()
        .any(|f| f.to_string().contains("\"reasoning\"")));
    // a turn cut inside its trace answers `content: null`, as vLLM does
    h.reply(&["<|channel>", "thought", "\n", "Still", " thinking"]);
    let (_, v) = h.json(json!({"messages": user("2+2?")})).await;
    let m = &v["choices"][0]["message"];
    assert_eq!(
        (&m["content"], m["reasoning"].as_str()),
        (&Value::Null, Some("Still thinking"))
    );
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
    h.reply(&[
        "<|channel>",
        "thought",
        "\n",
        "<channel|>",
        "It is",
        " 18°C",
        " in Paris.",
    ]);
    let (code, v) = h.json(json!({"messages": convo, "tools": tools()})).await;
    assert_eq!(code, StatusCode::OK, "{v}");
    let p = h.last_prompt();
    assert!(
        p.contains("<|tool_response>") && p.ends_with("<tool_response|>"),
        "{p}"
    );
    let m = &v["choices"][0]["message"];
    assert_eq!(m["content"], "It is 18°C in Paris.");
    assert!(m.get("reasoning").is_none() && m.get("tool_calls").is_none());
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    let (frames, _) = h.sse(json!({"messages": convo, "tools": tools()})).await;
    let (content, _, calls, _) = accumulate(&frames);
    assert_eq!((content.as_str(), calls.len()), ("It is 18°C in Paris.", 0));
}

#[tokio::test]
async fn a_call_cut_by_max_tokens_finishes_length() {
    let h = harness();
    h.reply(&["<|tool_call>", "call:get_weather{", "city:", "<|\"|>Par"]);
    let (_, v) = h
        .json(json!({"messages": user("w"), "tools": tools(), "max_tokens": 4}))
        .await;
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(
        v["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Par"
    );
}

/// `continue_final_message` continues the last assistant turn: the prompt ends right after its
/// text, as `transformers` cuts it. Gemma 4's template closes every turn (`<turn|>`), so binding
/// `add_generation_prompt: false` alone ended the prompt on a closed turn.
#[tokio::test]
async fn continue_final_message_ends_the_prompt_inside_the_final_turn() {
    let h = harness_for("gemma4-12b");
    h.reply(&[" Paris."]);
    let msgs = json!([{"role": "user", "content": "Capital of France?"},
                      {"role": "assistant", "content": "The capital of France is"}]);
    let (code, body) = h
        .json(json!({"messages": msgs, "continue_final_message": true, "add_generation_prompt": false}))
        .await;
    assert_eq!(code, StatusCode::OK, "{body}");
    assert!(
        h.last_prompt()
            .ends_with("<|turn>model\nThe capital of France is"),
        "{:?}",
        h.last_prompt()
    );
    // `add_generation_prompt: false` alone closes the turn and opens no model turn.
    let (code, _) = h
        .json(json!({"messages": user("hi"), "add_generation_prompt": false}))
        .await;
    assert_eq!(code, StatusCode::OK);
    assert!(
        h.last_prompt().ends_with("<|turn>user\nhi<turn|>\n"),
        "{:?}",
        h.last_prompt()
    );
}

/// vLLM 0.28 names a replayed trace `reasoning` (and maps `reasoning_content` onto it); the
/// template reads either, so a tool loop replayed with `reasoning` must render it.
#[tokio::test]
async fn replayed_reasoning_field_is_rendered_like_reasoning_content() {
    let h = harness_for("gemma4-12b");
    h.reply(&["It is 18C."]);
    let hist = |field: &str| {
        let mut call = json!({"role": "assistant", "content": null,
            "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]});
        call[field] = json!("Need the weather tool.");
        json!([{"role": "user", "content": "Weather in Paris?"}, call,
               {"role": "tool", "tool_call_id": "call_1", "content": "18C"}])
    };
    let mut prompts = Vec::new();
    for field in ["reasoning", "reasoning_content"] {
        let (code, body) = h
            .json(json!({"messages": hist(field), "tools": tools(),
                         "chat_template_kwargs": {"enable_thinking": true}}))
            .await;
        assert_eq!(code, StatusCode::OK, "{body}");
        prompts.push(h.last_prompt());
    }
    assert!(
        prompts[0].contains("<|channel>thought\nNeed the weather tool.\n<channel|>"),
        "{:?}",
        prompts[0]
    );
    assert_eq!(prompts[0], prompts[1]);
}

impl Harness {
    /// `(plowrt_requests_thinking_total, plowrt_reasoning_tokens_total, histogram count)`.
    async fn reasoning_metrics(&self) -> (u64, u64, u64) {
        let req = Request::get("/metrics").body(Body::empty()).unwrap();
        let resp = self.app.clone().oneshot(req).await.unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let read = |name: &str| {
            let line = format!("{name}{{model_name=\"toolm\",engine=\"0\"}} ");
            text.lines()
                .find_map(|l| l.strip_prefix(&line))
                .unwrap_or_else(|| panic!("{name} missing:\n{text}"))
                .parse()
                .unwrap()
        };
        (
            read("plowrt_requests_thinking_total"),
            read("plowrt_reasoning_tokens_total"),
            read("plowrt_request_reasoning_tokens_count"),
        )
    }
}

/// Thinking off counts nothing; thinking on counts the request and its trace tokens, the same
/// for a buffered and a streamed response, and equal to `usage.reasoning_tokens`; a generation
/// cut inside its trace still counts what it generated.
#[tokio::test]
async fn reasoning_metrics_count_thinking_requests_and_trace_tokens() {
    let h = harness_for("gemma4-12b");
    let on = json!({"enable_thinking": true});
    h.reply(&["Four", "."]);
    let (code, _) = h.json(json!({"messages": user("2+2?")})).await;
    assert_eq!(code, StatusCode::OK);
    let _ = h.sse(json!({"messages": user("2+2?")})).await;
    assert_eq!(h.reasoning_metrics().await, (0, 0, 0), "thinking off");

    let trace = toks("<|channel>thought\nTwo plus two is four.<channel|>Four.");
    let pieces: Vec<&str> = trace.iter().map(String::as_str).collect();
    h.reply(&pieces);
    let (code, body) = h
        .json(json!({"messages": user("2+2?"), "chat_template_kwargs": on}))
        .await;
    assert_eq!(code, StatusCode::OK, "{body}");
    let usage = body["usage"]["completion_tokens_details"]["reasoning_tokens"]
        .as_u64()
        .unwrap();
    assert!(usage > 0);
    assert_eq!(h.reasoning_metrics().await, (1, usage, 1), "buffered");
    let (frames, _) = h
        .sse(json!({"messages": user("2+2?"), "chat_template_kwargs": on,
                    "stream_options": {"include_usage": true}}))
        .await;
    let streamed = frames.last().unwrap()["usage"]["completion_tokens_details"]["reasoning_tokens"]
        .as_u64()
        .unwrap();
    assert_eq!(streamed, usage);
    assert_eq!(h.reasoning_metrics().await, (2, 2 * usage, 2), "streamed");

    // `max_tokens` lands inside the trace: finish `length`, every generated token is trace.
    let cut = &pieces[..4];
    h.reply(cut);
    let (code, body) = h
        .json(json!({"messages": user("2+2?"), "chat_template_kwargs": on, "max_tokens": 4}))
        .await;
    assert_eq!(code, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(h.reasoning_metrics().await, (3, 2 * usage + 4, 3), "cut");
}

impl Harness {
    async fn post_to(&self, path: &str, body: &str) -> (StatusCode, Value) {
        let req = Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = self.app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }
}

/// Request-shape parity with vLLM: `max_tokens` and `max_completion_tokens` together (the
/// latter wins), `response_format: text`, `reasoning_effort` turning Gemma 4's thinking on,
/// schema errors as 400 with the error envelope, and `/v1/completions`' default of 16 tokens.
#[tokio::test]
async fn request_fields_are_read_as_vllm_reads_them() {
    let h = harness_for("gemma4-12b");
    let many: Vec<String> = (0..20).map(|i| format!("w{i} ")).collect();
    let many: Vec<&str> = many.iter().map(String::as_str).collect();
    h.reply(&many);
    let (code, body) = h
        .json(json!({"messages": user("hi"), "max_tokens": 100, "max_completion_tokens": 5}))
        .await;
    assert_eq!(code, StatusCode::OK, "{body}");
    assert_eq!(
        body["choices"][0]["finish_reason"], "length",
        "5 wins over 100"
    );

    let (code, body) = h
        .json(json!({"messages": user("hi"), "response_format": {"type": "text"}}))
        .await;
    assert_eq!(code, StatusCode::OK, "{body}");
    let (code, _) = h
        .json(json!({"messages": user("hi"), "response_format": {"type": "json_object"}}))
        .await;
    assert_eq!(code, StatusCode::BAD_REQUEST);

    h.reply(&["ok"]);
    for (extra, thinks) in [
        (json!({"reasoning_effort": "high"}), true),
        (json!({"reasoning_effort": "none"}), false),
        (
            json!({"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": false}}),
            false,
        ),
        (json!({}), false),
    ] {
        let mut b = json!({"messages": user("hi")});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        let (code, body) = h.json(b).await;
        assert_eq!(code, StatusCode::OK, "{body}");
        assert_eq!(h.last_prompt().contains("<|think|>"), thinks, "{extra}");
    }

    for bad in [
        r#"{"model": "toolm", "messages": "hi"}"#,
        r#"{"model": "toolm"}"#,
        r#"{"model": "toolm", "messages": [{"role": "user", "content": "hi"}], "max_tokens": -5}"#,
    ] {
        let (code, body) = h.post_to("/v1/chat/completions", bad).await;
        assert_eq!(code, StatusCode::BAD_REQUEST, "{bad}");
        assert_eq!(body["error"]["code"], "invalid_json", "{body}");
    }

    h.reply(&many);
    let (code, body) = h
        .post_to(
            "/v1/completions",
            r#"{"model": "toolm", "prompt": "Hello"}"#,
        )
        .await;
    assert_eq!(code, StatusCode::OK, "{body}");
    assert_eq!(
        body["choices"][0]["finish_reason"], "length",
        "default 16 < 20"
    );
}

/// `max_tokens` landing inside a call's head (before its name is complete) used to return the
/// head's syntax as the answer (`content: "call:get_weather"`); it is call framing, not text.
#[tokio::test]
async fn a_call_head_cut_by_max_tokens_is_not_content() {
    let h = harness();
    let pieces = ["Checking.", "<|tool_call>", "call", ":get", "_weather"];
    h.reply(&pieces);
    let (_, v) = h
        .json(json!({"messages": user("w"), "tools": tools(), "max_tokens": 5}))
        .await;
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["choices"][0]["message"]["content"], "Checking.", "{v}");
    let (frames, _) = h
        .sse(json!({"messages": user("w"), "tools": tools(), "max_tokens": 5}))
        .await;
    let (content, _, calls, finish) = accumulate(&frames);
    assert_eq!((content.as_str(), calls.len()), ("Checking.", 0));
    assert_eq!(finish.as_deref(), Some("length"));
    // A turn that ENDS (not cut) inside a head still returns it as text.
    let (_, v) = h
        .json(json!({"messages": user("w"), "tools": tools(), "max_tokens": 50}))
        .await;
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        v["choices"][0]["message"]["content"],
        "Checking.call:get_weather"
    );
}

/// Thinking-control request fields are validated, each refusal naming its field.
#[tokio::test]
async fn thinking_control_fields_are_validated() {
    let h = harness_for("gemma4-12b");
    h.reply(&["Four."]);
    let on = json!({"enable_thinking": true});
    for (body, field) in [
        (
            json!({"thinking_token_budget": -2}),
            "thinking_token_budget",
        ),
        (json!({"reasoning_end": "Replying now."}), "reasoning_end"),
        (json!({"max_reply_tokens": 0}), "max_reply_tokens"),
        (
            json!({"max_reply_tokens": 9, "max_tokens": 8}),
            "max_reply_tokens",
        ),
    ] {
        let mut body = body;
        body["messages"] = user("2+2?");
        body["chat_template_kwargs"] = on.clone();
        let (code, v) = h.json(body).await;
        assert_eq!(code, StatusCode::BAD_REQUEST, "{v}");
        assert_eq!(v["error"]["param"], field, "{v}");
    }
    let (code, v) = h
        .json(json!({"messages": user("2+2?"), "chat_template_kwargs": on,
                     "thinking_token_budget": 16, "max_reply_tokens": 8,
                     "reasoning_end": "Replying now.<channel|>"}))
        .await;
    assert_eq!(code, StatusCode::OK, "{v}");
    // markers that are text, not special tokens, cannot be controlled by id
    let h = harness_for("qwen3");
    h.reply(&["Four."]);
    let (code, v) = h
        .json(json!({"messages": user("2+2?"), "thinking_token_budget": 16}))
        .await;
    assert_eq!(code, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["error"]["param"], "thinking_token_budget", "{v}");
}

/// `thinking_token_budget: 0` renders the template's thinking-off prompt.
#[tokio::test]
async fn a_zero_budget_turns_thinking_off() {
    let h = harness_for("gemma4-12b");
    h.reply(&["Four."]);
    let on = json!({"enable_thinking": true});
    let (code, _) = h
        .json(json!({"messages": user("2+2?"), "chat_template_kwargs": on}))
        .await;
    assert_eq!(code, StatusCode::OK);
    assert!(
        h.last_prompt().ends_with("<|turn>model\n"),
        "{}",
        h.last_prompt()
    );
    let (code, _) = h
        .json(json!({"messages": user("2+2?"), "chat_template_kwargs": on, "thinking_token_budget": 0}))
        .await;
    assert_eq!(code, StatusCode::OK);
    assert!(
        h.last_prompt()
            .ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
        "{}",
        h.last_prompt()
    );
}

/// Model defaults apply under the request's fields; `thought_prefix` seeds the trace when
/// thinking is on (only its seed text after a tool result) and the split reads the seeded trace
/// as reasoning, on both response paths.
#[tokio::test]
async fn chat_defaults_seed_the_trace_under_request_fields() {
    let h = harness_for("gemma4-12b");
    h.bundle
        .serving()
        .chat
        .set(crate::serve::config::ChatDefaults {
            max_tokens: Some(6),
            thought_prefix: Some("<|channel>thought\nThe caller just said".into()),
            chat_template_kwargs: Some(
                [("enable_thinking".to_string(), json!(true))]
                    .into_iter()
                    .collect(),
            ),
            ..Default::default()
        })
        .unwrap();
    let pieces = toks(" \"hi\".<channel|>Aap kaun?");
    let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
    h.reply(&pieces);
    let (code, v) = h
        .json(json!({"messages": user("hi"), "max_tokens": 64}))
        .await;
    assert_eq!(code, StatusCode::OK, "{v}");
    assert!(
        h.last_prompt()
            .ends_with("<|turn>model\n<|channel>thought\nThe caller just said"),
        "{}",
        h.last_prompt()
    );
    let m = &v["choices"][0]["message"];
    assert_eq!(
        (m["content"].as_str(), m["reasoning"].as_str()),
        (Some("Aap kaun?"), Some("\"hi\"."))
    );
    let (frames, _) = h
        .sse(json!({"messages": user("hi"), "max_tokens": 64}))
        .await;
    let (content, reasoning, _, _) = accumulate(&frames);
    assert_eq!(
        (content.as_str(), reasoning.as_str()),
        ("Aap kaun?", "\"hi\".")
    );
    // the default `max_tokens` applies when the request names none
    let (_, v) = h.json(json!({"messages": user("hi")})).await;
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    // thinking off: no seed
    h.reply(&["Hi."]);
    let (_, _) = h
        .json(json!({"messages": user("hi"), "chat_template_kwargs": {"enable_thinking": false}}))
        .await;
    assert!(
        h.last_prompt().ends_with("<|channel>thought\n<channel|>"),
        "{}",
        h.last_prompt()
    );
    // after a tool result the template opened the channel: only the seed text follows
    let convo = json!([
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18}"}
    ]);
    h.reply(&["18.", "<channel|>", "It is 18."]);
    let (code, v) = h.json(json!({"messages": convo, "tools": tools()})).await;
    assert_eq!(code, StatusCode::OK, "{v}");
    let p = h.last_prompt();
    assert!(
        p.ends_with("<|channel>thought\nThe caller just said") && !p.contains("thought\n\nThe"),
        "{p}"
    );
    assert_eq!(v["choices"][0]["message"]["content"], "It is 18.");
}

/// The voice guard moves planning lines out of `content` identically on both paths, reports it
/// in `x_plow_thinking`, answers an empty turn with the caller's fallback line, and counts both.
#[tokio::test]
async fn the_voice_guard_moves_planning_lines_on_both_paths() {
    let h = harness_for("gemma4-12b");
    h.bundle
        .serving()
        .chat
        .set(crate::serve::config::ChatDefaults {
            reply_guard: Some(crate::serve::reply_guard::GuardMode::Voice),
            reply_guard_lines: Some(crate::serve::reply_guard::GuardLines {
                wait: [("hi".to_string(), "एक सेकंड, कृपया।".to_string())].into(),
                goodbye: Default::default(),
            }),
            ..Default::default()
        })
        .unwrap();
    let on = json!({"enable_thinking": true});
    let pieces =
        toks("<|channel>thought\nplan<channel|>Wait, the prompt says ask once.\nAap kaun hain?");
    let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
    h.reply(&pieces);
    let body = json!({"messages": user("हेलो"), "chat_template_kwargs": on});
    let (code, v) = h.json(body.clone()).await;
    assert_eq!(code, StatusCode::OK, "{v}");
    let c = &v["choices"][0];
    assert_eq!(c["message"]["content"], "Aap kaun hain?");
    assert_eq!(
        c["message"]["reasoning"],
        "plan\nWait, the prompt says ask once."
    );
    assert_eq!(c["x_plow_thinking"]["guard_moved"], true);
    let (frames, _) = h.sse(body.clone()).await;
    let (content, reasoning, _, _) = accumulate(&frames);
    assert_eq!(
        (content.as_str(), reasoning.as_str()),
        ("Aap kaun hain?", "plan\nWait, the prompt says ask once.")
    );
    // nothing left to say: the caller's fallback line
    let pieces = toks("<|channel>thought\nplan<channel|>None.\n");
    let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
    h.reply(&pieces);
    let (_, v) = h.json(body.clone()).await;
    assert_eq!(v["choices"][0]["message"]["content"], "एक सेकंड, कृपया।");
    let (frames, _) = h.sse(body).await;
    assert_eq!(accumulate(&frames).0, "एक सेकंड, कृपया।");
    let req = Request::get("/metrics").body(Body::empty()).unwrap();
    let resp = h.app.clone().oneshot(req).await.unwrap();
    let text = String::from_utf8(
        resp.into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    let l = "{model_name=\"toolm\",engine=\"0\"}";
    assert!(
        text.contains(&format!("plowrt_reply_guard_moved_total{l} 4\n")),
        "{text}"
    );
    assert!(text.contains(&format!("plowrt_reply_guard_answer_reasoning_total{l} 0\n")));
    assert!(text.contains(
        "plowrt_reply_guard_fallback_total{model_name=\"toolm\",engine=\"0\",kind=\"wait\"} 2\n"
    ));
}

/// `POST /v1/requests/{id}/reasoning_end` reaches a registered live request and 404s otherwise.
#[tokio::test]
async fn live_reasoning_end_finds_only_requests_in_flight() {
    let h = harness();
    let flag = crate::serve::think::live_register(&std::sync::Arc::from("req-http-live"));
    let (code, v) = h
        .post_to("/v1/requests/req-http-live/reasoning_end", "")
        .await;
    assert_eq!(code, StatusCode::OK, "{v}");
    assert!(flag.load(std::sync::atomic::Ordering::Relaxed));
    drop(flag);
    let (code, v) = h
        .post_to("/v1/requests/req-http-live/reasoning_end", "")
        .await;
    assert_eq!(code, StatusCode::NOT_FOUND, "{v}");
    assert_eq!(v["error"]["code"], "request_not_found");
}

/// `reply_regenerate`: a guarded turn left silent is generated once more from its own trace,
/// closed, and that reply is the answer; the fallback line only when it is silent too.
#[tokio::test]
async fn a_silent_guarded_turn_is_regenerated_from_its_trace() {
    let h = harness_for("gemma4-12b");
    h.bundle
        .serving()
        .chat
        .set(crate::serve::config::ChatDefaults {
            reply_guard: Some(crate::serve::reply_guard::GuardMode::Voice),
            reply_regenerate: true,
            reply_guard_lines: Some(crate::serve::reply_guard::GuardLines {
                wait: [("hi".to_string(), "एक सेकंड।".to_string())].into(),
                goodbye: Default::default(),
            }),
            chat_template_kwargs: Some(
                [("enable_thinking".to_string(), json!(true))]
                    .into_iter()
                    .collect(),
            ),
            ..Default::default()
        })
        .unwrap();
    let silent: Vec<String> = toks("<|channel>thought\nplan<channel|>None.\n");
    let retry = toks("Aap kaun hain?");
    for stream in [false, true] {
        {
            let mut s = h.script.lock().unwrap();
            s.queue = [silent.clone(), retry.clone()].into();
        }
        let body = json!({"messages": user("hi")});
        let content = if stream {
            accumulate(&h.sse(body).await.0).0
        } else {
            h.json(body).await.1["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .to_string()
        };
        assert_eq!(content, "Aap kaun hain?", "stream {stream}");
        assert!(
            h.last_prompt()
                .ends_with("<|turn>model\n<|channel>thought\nplan<channel|>"),
            "{}",
            h.last_prompt()
        );
    }
    // silent again: the fallback line
    {
        let mut s = h.script.lock().unwrap();
        s.queue = [silent.clone(), toks("None.")].into();
    }
    let (_, v) = h.json(json!({"messages": user("hi")})).await;
    assert_eq!(v["choices"][0]["message"]["content"], "एक सेकंड।");
}

/// The response carries what OpenAI, vLLM and llama.cpp clients each read: the trace under
/// `reasoning` and `reasoning_content`, `system_fingerprint` on the body and every chunk, `logprobs`
/// and `refusal` as `null`; llama.cpp's request names are accepted.
#[tokio::test]
async fn responses_carry_the_fields_openai_vllm_and_llamacpp_clients_read() {
    let h = harness_for("qwen3");
    let pieces = toks("<think>plan</think>Four.");
    let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
    h.reply(&pieces);
    let (code, v) = h.json(json!({"messages": user("2+2?")})).await;
    assert_eq!(code, StatusCode::OK, "{v}");
    let c = &v["choices"][0];
    assert_eq!(
        (
            c["message"]["reasoning"].as_str(),
            c["message"]["reasoning_content"].as_str()
        ),
        (Some("plan"), Some("plan"))
    );
    assert_eq!(c["message"]["content"], "Four.");
    assert!(c["message"]["refusal"].is_null() && c["message"].get("refusal").is_some());
    assert!(c["logprobs"].is_null() && c.get("logprobs").is_some());
    assert_eq!(
        v["system_fingerprint"],
        crate::serve::openai::SYSTEM_FINGERPRINT
    );
    let (frames, _) = h.sse(json!({"messages": user("2+2?")})).await;
    assert!(frames
        .iter()
        .all(|f| f["system_fingerprint"] == crate::serve::openai::SYSTEM_FINGERPRINT));
    let both: String = frames
        .iter()
        .filter_map(|f| f["choices"][0]["delta"]["reasoning_content"].as_str())
        .collect();
    let (_, reasoning, _, _) = accumulate(&frames);
    assert_eq!((reasoning.as_str(), both.as_str()), ("plan", "plan"));
    // include_reasoning: false drops both names
    let (_, v) = h
        .json(json!({"messages": user("2+2?"), "include_reasoning": false}))
        .await;
    let m = &v["choices"][0]["message"];
    assert!(m.get("reasoning").is_none() && m.get("reasoning_content").is_none());
    // llama.cpp: the trace left in `content`
    let (_, v) = h
        .json(json!({"messages": user("2+2?"), "reasoning_format": "none"}))
        .await;
    assert_eq!(
        v["choices"][0]["message"]["content"],
        "<think>plan</think>Four."
    );
    let (code, v) = h
        .json(json!({"messages": user("2+2?"), "reasoning_format": "deepseek-legacy"}))
        .await;
    assert_eq!(
        (code, v["error"]["param"].as_str()),
        (StatusCode::BAD_REQUEST, Some("reasoning_format"))
    );
    // llama.cpp's budget names are `thinking_token_budget`
    for name in ["reasoning_budget_tokens", "thinking_budget_tokens"] {
        let (code, v) = h.json(json!({"messages": user("2+2?"), name: -2})).await;
        assert_eq!(
            (code, v["error"]["param"].as_str()),
            (StatusCode::BAD_REQUEST, Some("thinking_token_budget")),
            "{name}"
        );
    }
}

/// The thinking budget, the reply cap and `max_tokens` are separate budgets; a thinking budget
/// that leaves the reply no room inside `max_tokens` is refused rather than ending every turn
/// inside its trace.
#[tokio::test]
async fn a_thinking_budget_must_leave_room_for_the_reply() {
    let h = harness_for("gemma4-12b");
    h.reply(&["Four."]);
    let on = json!({"enable_thinking": true});
    let ask = |extra: Value| {
        let mut b = json!({"messages": user("2+2?"), "chat_template_kwargs": on});
        b.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        b
    };
    for extra in [
        json!({"thinking_token_budget": 300, "max_tokens": 128}),
        json!({"thinking_token_budget": 100, "max_tokens": 128, "max_reply_tokens": 60}),
    ] {
        let (code, v) = h.json(ask(extra.clone())).await;
        assert_eq!(code, StatusCode::BAD_REQUEST, "{extra}: {v}");
        assert_eq!(v["error"]["param"], "thinking_token_budget");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("max_tokens"));
    }
    // `reasoning_end: ""`: the bare marker (this fixture's tokenizer spells the default bridge byte by byte)
    for extra in [
        json!({"thinking_token_budget": 64, "max_tokens": 128, "reasoning_end": ""}),
        json!({"thinking_token_budget": 32, "max_tokens": 128, "max_reply_tokens": 60, "reasoning_end": ""}),
    ] {
        let (code, v) = h.json(ask(extra.clone())).await;
        assert_eq!(code, StatusCode::OK, "{extra}: {v}");
    }
}

/// Server-wide defaults from vLLM / llama.cpp flag names reach every model, a model's file entry
/// wins over them, `--reasoning-parser` is checked against the template, and `--chat-template`
/// replaces the packet's template.
#[tokio::test]
async fn engine_flags_install_as_defaults_under_the_file_and_the_request() {
    use crate::serve::config::{
        install_chat_defaults, ChatDefaults, GenerationDefaults, ServerReasoning,
    };
    let h = harness_for("qwen3");
    let models = vec![("toolm".to_string(), Arc::clone(&h.bundle))];
    let resolve = |n: &str| (n == "toolm").then(|| ("toolm".to_string(), Arc::clone(&h.bundle)));
    // a parser that splits on other markers is refused; qwen3's and an unlisted one are not
    let gemma = ServerReasoning {
        parser: Some("gemma4".into()),
        ..Default::default()
    };
    let err = install_chat_defaults(&gemma, None, &models, resolve).unwrap_err();
    assert!(
        err.contains("<|channel>") && err.contains("<think>"),
        "{err}"
    );
    let start = ServerReasoning {
        start_str: Some("[THINK]".into()),
        ..Default::default()
    };
    assert!(install_chat_defaults(&start, None, &models, resolve).is_err());
    let server = ServerReasoning {
        parser: Some("qwen3".into()),
        defaults: ChatDefaults {
            reasoning_format: Some("none".into()),
            include_reasoning: Some(true),
            generation: Some(GenerationDefaults {
                max_new_tokens: Some(3),
                ..Default::default()
            }),
            ..Default::default()
        },
        chat_template: Some("{% for m in messages %}<<{{ m.content }}>>{% endfor %}".into()),
        ..Default::default()
    };
    let dir = std::env::temp_dir().join(format!("plowrt-engine-flags-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("defaults.json");
    std::fs::write(&file, r#"{"toolm": {"reasoning_format": "auto"}}"#).unwrap();
    install_chat_defaults(&server, Some(&file), &models, resolve).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let pieces = toks("<think>plan</think>Four.");
    let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
    h.reply(&pieces);
    let (code, v) = h.json(json!({"messages": user("2+2?")})).await;
    assert_eq!(code, StatusCode::OK, "{v}");
    assert_eq!(h.last_prompt(), "<<2+2?>>", "--chat-template");
    let c = &v["choices"][0];
    // the file's `auto` beat the server's `none`: the trace is split
    assert_eq!(
        (
            c["message"]["reasoning"].as_str(),
            c["message"]["content"].as_str()
        ),
        (Some("plan"), Some("Four."))
    );
    // max_new_tokens 3 is the default max_tokens (the scripted reply is longer)
    assert_eq!(c["finish_reason"], "length");
    // the request still overrides both
    let (_, v) = h
        .json(json!({"messages": user("2+2?"), "max_tokens": 64, "reasoning_format": "none"}))
        .await;
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        v["choices"][0]["message"]["content"],
        "<think>plan</think>Four."
    );
}
