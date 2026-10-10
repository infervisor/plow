//! Per-token host cost of the chat response path, CPU only (ignored; run in release with
//! `--nocapture`): the reasoning split, the tool-call parser, special-token stripping and the
//! SSE frame build, for a plain answer and for tool-call streams. Reports ns and heap
//! allocations per token. `TOOLBENCH_TOKENIZER` (a `tokenizer.json`) supplies the real
//! special-token list; otherwise a Gemma-like list is used.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

use plowrt::serve::openai::{ChunkChoice, Delta};
use plowrt::serve::reasoning::{ReasoningMode, ReasoningSplit};
use plowrt::serve::stream::FrameHead;
use plowrt::serve::tools::request::{Force, ParseSpec, ToolDef};
use plowrt::serve::tools::stream::ToolStream;
use plowrt::serve::tools::{SpecialText, ToolFormat};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        System.alloc_zeroed(l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        System.realloc(p, l, n)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const MARKERS: &[&str] = &[
    "<|tool_call>", "<tool_call|>", "<|\"|>", "<|channel>", "<channel|>", "<tool_call>", "</tool_call>",
];

/// Token-like pieces: every marker its own piece, other text in 4-byte runs (a BPE average).
fn pieces(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut run = String::new();
    while i < text.len() {
        if let Some(m) = MARKERS.iter().find(|m| text[i..].starts_with(**m)) {
            if !run.is_empty() {
                out.push(std::mem::take(&mut run));
            }
            out.push(m.to_string());
            i += m.len();
            continue;
        }
        let c = text[i..].chars().next().unwrap();
        run.push(c);
        i += c.len_utf8();
        if run.len() >= 4 {
            out.push(std::mem::take(&mut run));
        }
    }
    if !run.is_empty() {
        out.push(run);
    }
    out
}

fn specials() -> Arc<SpecialText> {
    let list = std::env::var("TOOLBENCH_TOKENIZER")
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| {
            Some(
                v.get("added_tokens")?
                    .as_array()?
                    .iter()
                    .filter(|e| e["special"].as_bool() == Some(true))
                    .filter_map(|e| e["content"].as_str().map(String::from))
                    .collect::<Vec<_>>(),
            )
        })
        .unwrap_or_else(|| {
            ["<|tool_call>", "<tool_call|>", "<|\"|>", "<|tool_response>", "<tool_response|>", "<|channel>", "<channel|>",
             "<|turn>", "<turn|>", "<bos>", "<eos>", "<pad>", "<|tool>", "<tool|>", "<|image|>", "<|audio|>"]
                .map(String::from)
                .to_vec()
        });
    Arc::new(SpecialText::new(list))
}

fn spec(format: ToolFormat) -> ParseSpec {
    let tool = |name: &str, params| ToolDef { name: name.into(), params, strict: false };
    let tools = vec![
        tool("search_flights", serde_json::json!({"type": "object", "properties": {
            "origin": {"type": "string"}, "destination": {"type": "string"}, "passengers": {"type": "integer"},
            "filters": {"type": "object"}}})),
        tool("write_file", serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}})),
    ];
    ParseSpec { format, tools: Arc::new(tools), parallel: true, force: Force::Auto, history_calls: 0 }
}

fn prose(n: usize) -> String {
    let words = ["The ", "weather ", "in ", "Paris ", "is ", "mild, ", "with ", "a ", "light ", "breeze ", "and ", "clear ", "skies. "];
    (0..n).map(|i| words[i % words.len()]).collect()
}

fn code(n: usize) -> String {
    let lines = ["fn main() {\n", "    let x = vec![1, 2, 3];\n", "    println!(\"{:?}\", x);\n", "}\n"];
    (0..n).map(|i| lines[i % lines.len()]).collect()
}

#[derive(Clone, Copy)]
enum Path {
    /// The plain reasoning split (no tools, markers that are not special tokens).
    Plain,
    /// The response stream without tools (reasoning markers that are special tokens: Gemma 4).
    NoTools,
    Tools(ToolFormat),
}

struct Case {
    name: &'static str,
    path: Path,
    text: String,
}

fn cases() -> Vec<Case> {
    let g_short = "<|tool_call>call:search_flights{origin:<|\"|>O'Hare<|\"|>,destination:<|\"|>Zürich<|\"|>,passengers:2,filters:{max_price:450.5,airlines:[<|\"|>LX<|\"|>,<|\"|>UA<|\"|>]}}<tool_call|>".to_string();
    let g_long = format!("<|tool_call>call:write_file{{path:<|\"|>src/main.rs<|\"|>,content:<|\"|>{}<|\"|>}}<tool_call|>", code(400));
    let g_huge = format!("<|tool_call>call:write_file{{path:<|\"|>src/main.rs<|\"|>,content:<|\"|>{}<|\"|>}}<tool_call|>", code(1600));
    let h_long = format!(
        "<tool_call>\n{{\"name\": \"write_file\", \"arguments\": {{\"path\": \"src/main.rs\", \"content\": {}}}}}\n</tool_call>",
        serde_json::to_string(&code(400)).unwrap()
    );
    vec![
        Case { name: "plain_text (no tools)", path: Path::Plain, text: prose(1200) },
        Case { name: "gemma4 no tools (channel split)", path: Path::NoTools, text: prose(1200) },
        Case { name: "tools_text_answer gemma4", path: Path::Tools(ToolFormat::Gemma4), text: prose(1200) },
        Case { name: "gemma4 short call", path: Path::Tools(ToolFormat::Gemma4), text: g_short },
        Case { name: "gemma4 long call (16 KB args)", path: Path::Tools(ToolFormat::Gemma4), text: g_long },
        Case { name: "gemma4 huge call (64 KB args)", path: Path::Tools(ToolFormat::Gemma4), text: g_huge },
        Case { name: "hermes long call (16 KB args)", path: Path::Tools(ToolFormat::Hermes), text: h_long },
    ]
}

/// The streamed loop body of `chat.rs`: one frame per token.
fn run(case: &Case, toks: &[String], strip: &Arc<SpecialText>, head: &FrameHead) -> usize {
    let mut bytes = 0;
    match case.path {
        Path::Plain => {
            let mut split = ReasoningSplit::new(ReasoningMode::ThinkTag, false);
            for t in toks {
                let (r, c) = split.push(t);
                let f = head.frame(&ChunkChoice {
                    index: 0,
                    delta: Delta { role: None, content: c, reasoning_content: r, tool_calls: None },
                    logprobs: None,
                    finish_reason: None,
                    x_plow_finish_reason: None,
                });
                bytes += f.len();
            }
        }
        Path::NoTools | Path::Tools(_) => {
            let sp = match case.path {
                Path::Tools(f) => Some(spec(f)),
                _ => None,
            };
            let mode = ReasoningMode::Tags { open: "<|channel>thought", close: "<channel|>" };
            let mut ts = ToolStream::new(sp.as_ref(), mode, false, strip.clone(), true);
            for t in toks {
                bytes += ts.push(t).frame(head, None, None, true, None).len();
            }
            let _ = ts.finish(false);
        }
    }
    bytes
}

#[test]
#[ignore]
fn tool_stream_per_token_cost() {
    let strip = specials();
    let head = FrameHead::new("chatcmpl-1", "chat.completion.chunk", 1, "m");
    let reps: usize = std::env::var("TOOLBENCH_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    // The box is shared: each case reports the best and the median of several timed trials.
    let trials = 9;
    eprintln!("{:<34} {:>7} {:>9} {:>9} {:>12}", "case", "tokens", "min ns/t", "med ns/t", "allocs/token");
    for case in cases() {
        let toks = pieces(&case.text);
        let mut sink = 0;
        for _ in 0..reps / 10 + 1 {
            sink += run(&case, &toks, &strip, &head);
        }
        let a0 = ALLOCS.load(Relaxed);
        let mut ns = Vec::new();
        for _ in 0..trials {
            let t0 = Instant::now();
            for _ in 0..reps {
                sink += run(&case, &toks, &strip, &head);
            }
            ns.push(t0.elapsed().as_nanos() as f64 / (reps * toks.len()) as f64);
        }
        let allocs = (ALLOCS.load(Relaxed) - a0) as f64 / (trials * reps * toks.len()) as f64;
        ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("{:<34} {:>7} {:>9.1} {:>9.1} {:>12.2}", case.name, toks.len(), ns[0], ns[trials / 2], allocs);
        assert!(sink > 0);
    }
}
