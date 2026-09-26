//! `plowrt dsv41-serve`: an OpenAI-compatible `/v1/completions` server over the V4.1 engine.
//!
//! One scheduler thread owns the engine. Each loop admits at most one waiting request (a single-
//! sequence prefill, which also yields its first token), then advances every active slot by one
//! decode step. Tokens stream back per request as SSE chunks in the OpenAI completions format,
//! with a final usage frame when `stream_options.include_usage` is set.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::text::tokenizer::{load_tokenizer, Tokenize};

use super::engine::{Engine, EngineOpts};

pub struct ServeOpts {
    pub ckpt: PathBuf,
    pub cubin: PathBuf,
    pub port: u16,
    pub model_name: String,
    pub engine: EngineOpts,
}

enum Ev {
    Token(u32),
    Done { prompt: usize, completion: usize, reason: &'static str },
    Err(String),
}

struct Job {
    prompt: Vec<u32>,
    max_tokens: usize,
    ignore_eos: bool,
    tx: mpsc::UnboundedSender<Ev>,
}

struct AppState {
    jobs: mpsc::UnboundedSender<Job>,
    tok: Arc<dyn Tokenize>,
    model: String,
}

struct Active {
    slot: usize,
    pos: usize,
    last: u32,
    generated: usize,
    n_prompt: usize,
    max_tokens: usize,
    ignore_eos: bool,
    tx: mpsc::UnboundedSender<Ev>,
}

/// The scheduler loop (runs on its own OS thread; owns the engine).
fn scheduler(mut eng: Engine, mut rx: mpsc::UnboundedReceiver<Job>, max_len: usize) {
    let eos = eng.cfg.eos_id;
    let mut free: Vec<usize> = (0..eng.n_slots()).rev().collect();
    let mut waiting: VecDeque<Job> = VecDeque::new();
    let mut active: Vec<Active> = Vec::new();
    let finish = |a: &Active, reason: &'static str| {
        let _ = a.tx.send(Ev::Done { prompt: a.n_prompt, completion: a.generated, reason });
    };
    loop {
        while let Ok(j) = rx.try_recv() {
            waiting.push_back(j);
        }
        if active.is_empty() && waiting.is_empty() {
            match rx.blocking_recv() {
                Some(j) => waiting.push_back(j),
                None => return,
            }
        }
        // admit one prefill
        if !free.is_empty() {
            if let Some(j) = waiting.pop_front() {
                if j.prompt.is_empty() || j.prompt.len() + j.max_tokens > max_len {
                    let _ = j.tx.send(Ev::Err(format!(
                        "prompt ({}) + max_tokens ({}) exceeds the engine's max length {max_len}",
                        j.prompt.len(),
                        j.max_tokens
                    )));
                } else {
                    let slot = free.pop().unwrap();
                    let r = eng.release(slot).and_then(|_| eng.step(&[(slot, j.prompt.clone(), 0)], false));
                    match r {
                        Ok(t) => {
                            let a = Active {
                                slot,
                                pos: j.prompt.len(),
                                last: t[0],
                                generated: 1,
                                n_prompt: j.prompt.len(),
                                max_tokens: j.max_tokens,
                                ignore_eos: j.ignore_eos,
                                tx: j.tx,
                            };
                            let sent = a.tx.send(Ev::Token(t[0])).is_ok();
                            let stop = (!a.ignore_eos && t[0] == eos) || a.generated >= a.max_tokens || !sent;
                            if stop {
                                finish(&a, if a.generated >= a.max_tokens { "length" } else { "stop" });
                                free.push(slot);
                            } else {
                                active.push(a);
                            }
                        }
                        Err(e) => {
                            let _ = j.tx.send(Ev::Err(format!("prefill: {e}")));
                            free.push(slot);
                        }
                    }
                }
            }
        }
        if active.is_empty() {
            continue;
        }
        let seqs: Vec<(usize, Vec<u32>, usize)> = active.iter().map(|a| (a.slot, vec![a.last], a.pos)).collect();
        match eng.step(&seqs, true) {
            Ok(toks) => {
                let mut keep = Vec::with_capacity(active.len());
                for (mut a, t) in active.drain(..).zip(toks) {
                    a.pos += 1;
                    a.last = t;
                    a.generated += 1;
                    let sent = a.tx.send(Ev::Token(t)).is_ok();
                    let length = a.generated >= a.max_tokens || a.pos + 1 >= max_len;
                    if !sent || length || (!a.ignore_eos && t == eos) {
                        finish(&a, if length { "length" } else { "stop" });
                        free.push(a.slot);
                    } else {
                        keep.push(a);
                    }
                }
                active = keep;
            }
            Err(e) => {
                for a in active.drain(..) {
                    let _ = a.tx.send(Ev::Err(format!("decode: {e}")));
                    free.push(a.slot);
                }
            }
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

async fn completions(State(st): State<Arc<AppState>>, Json(req): Json<Value>) -> Response {
    let prompt: Vec<u32> = match &req["prompt"] {
        Value::String(s) => st.tok.encode_with_special_tokens(s, true),
        Value::Array(a) if a.iter().all(|v| v.is_u64()) => a.iter().map(|v| v.as_u64().unwrap() as u32).collect(),
        Value::Array(a) if a.len() == 1 && a[0].is_string() => st.tok.encode_with_special_tokens(a[0].as_str().unwrap(), true),
        _ => return (StatusCode::BAD_REQUEST, "prompt must be a string or a list of token ids").into_response(),
    };
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(16).max(1) as usize;
    let ignore_eos = req["ignore_eos"].as_bool().unwrap_or(false);
    let streaming = req["stream"].as_bool().unwrap_or(false);
    let usage = req["stream_options"]["include_usage"].as_bool().unwrap_or(false);
    let (tx, mut rx) = mpsc::unbounded_channel();
    if st.jobs.send(Job { prompt, max_tokens, ignore_eos, tx }).is_err() {
        return (StatusCode::SERVICE_UNAVAILABLE, "engine stopped").into_response();
    }
    let id = format!("cmpl-{}", now_secs());
    let model = st.model.clone();
    let tok = st.tok.clone();
    if !streaming {
        let mut ids = Vec::new();
        loop {
            match rx.recv().await {
                Some(Ev::Token(t)) => ids.push(t),
                Some(Ev::Done { prompt, completion, reason }) => {
                    return Json(json!({
                        "id": id, "object": "text_completion", "created": now_secs(), "model": model,
                        "choices": [{"index": 0, "text": tok.decode(&ids), "logprobs": null, "finish_reason": reason}],
                        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
                    }))
                    .into_response();
                }
                Some(Ev::Err(e)) => return (StatusCode::BAD_REQUEST, e).into_response(),
                None => return (StatusCode::INTERNAL_SERVER_ERROR, "engine dropped the request").into_response(),
            }
        }
    }
    struct S {
        rx: mpsc::UnboundedReceiver<Ev>,
        ids: Vec<u32>,
        emitted: usize,
        tail: VecDeque<String>,
        done: bool,
    }
    let init = S { rx, ids: Vec::new(), emitted: 0, tail: VecDeque::new(), done: false };
    let body = stream::unfold(init, move |mut s| {
        let (id, model, tok) = (id.clone(), model.clone(), tok.clone());
        async move {
            if let Some(d) = s.tail.pop_front() {
                return Some((Ok::<Event, std::convert::Infallible>(Event::default().data(d)), s));
            }
            if s.done {
                return None;
            }
            let chunk = |text: &str, reason: Value| {
                json!({"id": id, "object": "text_completion", "created": now_secs(), "model": model,
                       "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": reason}]})
                .to_string()
            };
            match s.rx.recv().await {
                Some(Ev::Token(t)) => {
                    s.ids.push(t);
                    let text = tok.decode(&s.ids);
                    // hold back an incomplete UTF-8 sequence until its next byte arrives
                    let delta = if text.ends_with('\u{fffd}') || text.len() < s.emitted {
                        String::new()
                    } else {
                        let d = text[s.emitted..].to_string();
                        s.emitted = text.len();
                        d
                    };
                    Some((Ok(Event::default().data(chunk(&delta, Value::Null))), s))
                }
                Some(Ev::Done { prompt, completion, reason }) => {
                    s.done = true;
                    if usage {
                        s.tail.push_back(
                            json!({"id": id, "object": "text_completion", "created": now_secs(), "model": model, "choices": [],
                                   "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}})
                            .to_string(),
                        );
                    }
                    s.tail.push_back("[DONE]".into());
                    Some((Ok(Event::default().data(chunk("", json!(reason)))), s))
                }
                Some(Ev::Err(e)) => {
                    s.done = true;
                    s.tail.push_back("[DONE]".into());
                    Some((Ok(Event::default().data(json!({"error": {"message": e}}).to_string())), s))
                }
                None => None,
            }
        }
    });
    Sse::new(body).into_response()
}

async fn health() -> &'static str {
    "ok"
}

async fn models(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": st.model, "object": "model", "owned_by": "plowrt"}]}))
}

pub async fn run(opts: ServeOpts) -> Result<(), Box<dyn std::error::Error>> {
    let cubin = std::fs::read(&opts.cubin)?;
    let tok = load_tokenizer(&opts.ckpt);
    if tok.is_byte_fallback() {
        return Err(format!("no tokenizer.json in {}", opts.ckpt.display()).into());
    }
    let max_len = opts.engine.max_len;
    let t0 = Instant::now();
    let eng = Engine::load(&opts.ckpt, &cubin, opts.engine)?;
    eprintln!("dsv41: engine loaded in {:.0}s", t0.elapsed().as_secs_f32());
    let (jtx, jrx) = mpsc::unbounded_channel();
    std::thread::Builder::new().name("dsv41-sched".into()).spawn(move || scheduler(eng, jrx, max_len))?;
    let state = Arc::new(AppState { jobs: jtx, tok, model: opts.model_name });
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/healthz", get(health))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", opts.port)).await?;
    eprintln!("dsv41: serving on :{}", opts.port);
    axum::serve(listener, app).tcp_nodelay(true).await?;
    Ok(())
}
