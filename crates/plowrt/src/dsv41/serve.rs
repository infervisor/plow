//! `plowrt dsv41-serve`: an OpenAI-compatible `/v1/completions` server over the V4.1 engine.
//!
//! One scheduler thread owns the engine. Each loop admits at most one waiting request (a single-
//! sequence prefill, which also yields its first token), then advances every active slot by one
//! decode step. Detokenization and stop-string matching run on the scheduler thread, so a stop
//! frees its slot at once; the HTTP side only frames what it is sent.
//!
//! Failure policy: a request error (bad input, unsupported parameter, over-long prompt) is a 4xx
//! with an OpenAI error body and never reaches the engine. An engine error fails the requests in
//! that step. A device fault (the CUDA context is poisoned and every later launch would fail) marks
//! `/health` unhealthy and exits the process, so a supervisor restarts it instead of it serving
//! errors behind a green health check.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::error::RuntimeError;
use crate::text::tokenizer::{load_tokenizer, Tokenize};

use super::engine::{Engine, EngineOpts, Sampling, Ticket};

pub struct ServeOpts {
    pub ckpt: PathBuf,
    pub cubin: PathBuf,
    pub port: u16,
    pub model_name: String,
    /// Requests admitted but not finished (queued + running); beyond it the server answers 429.
    pub max_queued: usize,
    pub engine: EngineOpts,
    /// Measure instead of serving: (rung spec, reps per rung, report path). See
    /// [`Engine::rung_bench`].
    pub rung_bench: Option<(String, usize, Option<PathBuf>)>,
}

/// What the scheduler sends a request's HTTP side.
enum Ev {
    /// One generated token's text (possibly empty: an incomplete UTF-8 sequence, or text held
    /// back because it may begin a stop string). One per token, so clients can time each token.
    Text(String),
    Done { prompt: usize, completion: usize, reason: &'static str },
    Err { status: StatusCode, message: String },
}

struct Job {
    prompt: Vec<u32>,
    max_tokens: usize,
    min_tokens: usize,
    ignore_eos: bool,
    stop: Vec<String>,
    sampling: Sampling,
    arrived: Instant,
    tx: mpsc::UnboundedSender<Ev>,
}

/// Scheduler liveness, read by `/health`.
struct Health {
    /// Set once the scheduler thread exits or the device faults.
    dead: AtomicBool,
    /// Milliseconds since `epoch` when the current engine step started, 0 when idle.
    busy_since_ms: AtomicU64,
    epoch: Instant,
}

impl Health {
    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64 + 1
    }
}

/// A step that runs longer than this is a hang (a 20k-token prefill takes seconds, not minutes).
const STEP_HANG_MS: u64 = 300_000;

struct AppState {
    jobs: mpsc::UnboundedSender<Job>,
    tok: Arc<dyn Tokenize>,
    model: String,
    vocab: usize,
    max_len: usize,
    max_queued: usize,
    inflight: Arc<AtomicUsize>,
    health: Arc<Health>,
    next_id: AtomicU64,
}

/// Incremental detokenization (the prefix/read-offset scheme `serve::mux` uses): decode only the
/// tokens since the last emitted boundary, and never emit a trailing incomplete UTF-8 sequence.
struct Detok {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl Detok {
    fn new() -> Self {
        Detok { ids: Vec::new(), prefix: 0, read: 0 }
    }
    fn push(&mut self, tok: &dyn Tokenize, id: u32) -> String {
        self.ids.push(id);
        let prefix_text = tok.decode(&self.ids[self.prefix..self.read]);
        let new_text = tok.decode(&self.ids[self.prefix..]);
        if new_text.len() > prefix_text.len() && !new_text.ends_with('\u{fffd}') && new_text.is_char_boundary(prefix_text.len()) {
            let delta = new_text[prefix_text.len()..].to_string();
            self.prefix = self.read;
            self.read = self.ids.len();
            delta
        } else {
            String::new()
        }
    }
    /// Whatever is still held back (an incomplete sequence at the very end is emitted as-is).
    fn flush(&mut self, tok: &dyn Tokenize) -> String {
        let prefix_text = tok.decode(&self.ids[self.prefix..self.read]);
        let all = tok.decode(&self.ids[self.prefix..]);
        self.prefix = self.ids.len();
        self.read = self.ids.len();
        if all.len() > prefix_text.len() && all.is_char_boundary(prefix_text.len()) {
            all[prefix_text.len()..].to_string()
        } else {
            String::new()
        }
    }
}

/// Stop-string matching over the streamed text. Text that could still turn out to begin a stop
/// string is held back; on a match the text before it is released and the rest discarded.
struct StopScan {
    stops: Vec<String>,
    pending: String,
    hold: usize,
}

impl StopScan {
    fn new(stops: Vec<String>) -> Self {
        let hold = stops.iter().map(|s| s.len()).max().unwrap_or(0).saturating_sub(1);
        StopScan { stops, pending: String::new(), hold }
    }
    /// Feed text; returns (text safe to emit, matched a stop).
    fn feed(&mut self, text: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (text.to_string(), false);
        }
        self.pending.push_str(text);
        if let Some(at) = self.stops.iter().filter_map(|s| self.pending.find(s.as_str())).min() {
            let out = self.pending[..at].to_string();
            self.pending.clear();
            return (out, true);
        }
        let mut cut = self.pending.len().saturating_sub(self.hold);
        while !self.pending.is_char_boundary(cut) {
            cut -= 1;
        }
        let out = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        (out, false)
    }
    fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

struct Active {
    slot: usize,
    pos: usize,
    last: u32,
    generated: usize,
    n_prompt: usize,
    job: Job,
    detok: Detok,
    stop: StopScan,
    first_token: Instant,
}

enum Outcome {
    Continue,
    Finished(&'static str),
    Gone,
}

impl Active {
    /// Account one generated token: stream its text and decide whether the sequence is done.
    fn accept(&mut self, t: u32, tok: &dyn Tokenize, eos: u32, max_len: usize) -> Outcome {
        self.generated += 1;
        let is_eos = t == eos && !self.job.ignore_eos && self.generated > self.job.min_tokens;
        let delta = if is_eos { String::new() } else { self.detok.push(tok, t) };
        let (text, stopped) = self.stop.feed(&delta);
        let stopped = stopped && self.generated > self.job.min_tokens;
        let length = self.generated >= self.job.max_tokens || self.pos + 1 >= max_len;
        let mut text = text;
        if is_eos || stopped || length {
            if !stopped {
                let tail = self.detok.flush(tok);
                let (t2, _) = self.stop.feed(&tail);
                text.push_str(&t2);
                text.push_str(&self.stop.finish());
            }
        }
        if self.job.tx.send(Ev::Text(text)).is_err() {
            return Outcome::Gone;
        }
        if is_eos || stopped {
            Outcome::Finished("stop")
        } else if length {
            Outcome::Finished("length")
        } else {
            Outcome::Continue
        }
    }
}

fn is_fatal(e: &RuntimeError) -> bool {
    matches!(e, RuntimeError::DeviceFault { .. })
}

/// The scheduler loop (runs on its own OS thread; owns the engine).
fn scheduler(mut eng: Engine, mut rx: mpsc::UnboundedReceiver<Job>, max_len: usize, tok: Arc<dyn Tokenize>, inflight: Arc<AtomicUsize>, health: Arc<Health>) {
    struct Exit(Arc<Health>);
    impl Drop for Exit {
        fn drop(&mut self) {
            self.0.dead.store(true, Ordering::SeqCst);
        }
    }
    let _exit = Exit(health.clone());
    let eos = eng.cfg.eos_id;
    let mut free: Vec<usize> = (0..eng.n_slots()).rev().collect();
    let mut waiting: VecDeque<Job> = VecDeque::new();
    // Decode groups, one per decode lane (lane 0 alone when there are none): each group's step is
    // issued as soon as its previous one is collected, so up to n groups flow through the pipeline
    // stages together. A prefill runs on lane 0 alongside them. Tickets are collected oldest first.
    let n_groups = (eng.n_lanes() - 1).max(1);
    let group_lane = |g: usize| if eng.n_lanes() > 1 { g + 1 } else { 0 };
    let mut groups: Vec<Group> = (0..n_groups).map(|g| Group { lane: group_lane(g), members: Vec::new(), inflight: None }).collect();
    let mut prefill: Option<(Ticket, Job, usize)> = None;
    let mut order: VecDeque<Pending> = VecDeque::new();
    let done = |job: &Job, n_prompt: usize, generated: usize, reason: &'static str, first: Option<Instant>| {
        let _ = job.tx.send(Ev::Done { prompt: n_prompt, completion: generated, reason });
        inflight.fetch_sub(1, Ordering::SeqCst);
        let now = Instant::now();
        let ttft = first.map(|f| (f - job.arrived).as_secs_f64() * 1e3).unwrap_or(f64::NAN);
        let tpot = match first {
            Some(f) if generated > 1 => (now - f).as_secs_f64() * 1e3 / (generated - 1) as f64,
            _ => f64::NAN,
        };
        tracing::info!(target: "dsv41", prompt = n_prompt, completion = generated, reason, ttft_ms = format!("{ttft:.1}"), tpot_ms = format!("{tpot:.2}"), "request done");
    };
    let fail = |job: &Job, status: StatusCode, msg: String| {
        let _ = job.tx.send(Ev::Err { status, message: msg });
        inflight.fetch_sub(1, Ordering::SeqCst);
    };
    let fatal = |e: &RuntimeError| {
        tracing::error!(target: "dsv41", "device fault, exiting: {e}");
        health.dead.store(true, Ordering::SeqCst);
        // Give in-flight error frames a moment to flush, then exit for the supervisor.
        std::thread::sleep(Duration::from_millis(200));
        std::process::exit(1);
    };
    // A failed issue or collect drains the engine: every step in flight is done or abandoned, so
    // every in-flight sequence fails (their caches may hold a partial step).
    let fail_all = |groups: &mut Vec<Group>, prefill: &mut Option<(Ticket, Job, usize)>, order: &mut VecDeque<Pending>, free: &mut Vec<usize>, e: &RuntimeError| {
        order.clear();
        for g in groups.iter_mut() {
            g.inflight = None;
            for a in g.members.drain(..) {
                fail(&a.job, StatusCode::INTERNAL_SERVER_ERROR, format!("decode failed: {e}"));
                free.push(a.slot);
            }
        }
        if let Some((_, j, slot)) = prefill.take() {
            fail(&j, StatusCode::INTERNAL_SERVER_ERROR, format!("prefill failed: {e}"));
            free.push(slot);
        }
        if is_fatal(e) {
            fatal(e);
        }
    };
    loop {
        while let Ok(j) = rx.try_recv() {
            waiting.push_back(j);
        }
        let idle = order.is_empty() && groups.iter().all(|g| g.members.is_empty());
        if idle && waiting.is_empty() {
            match rx.blocking_recv() {
                Some(j) => waiting.push_back(j),
                None => return,
            }
        }
        // drop requests whose client already left
        while let Some(j) = waiting.front() {
            if j.tx.is_closed() {
                let j = waiting.pop_front().unwrap();
                inflight.fetch_sub(1, Ordering::SeqCst);
                drop(j);
            } else {
                break;
            }
        }
        // issue one prefill on lane 0
        if prefill.is_none() {
            if let (Some(slot), true) = (free.last().copied(), !waiting.is_empty()) {
                let j = waiting.pop_front().unwrap();
                free.pop();
                match eng.release(slot).and_then(|_| eng.issue(0, &[(slot, j.prompt.clone(), 0)], false, &[j.sampling])) {
                    Ok(t) => {
                        prefill = Some((t, j, slot));
                        order.push_back(Pending::Prefill);
                    }
                    Err(e) => {
                        fail(&j, StatusCode::INTERNAL_SERVER_ERROR, format!("prefill failed: {e}"));
                        free.push(slot);
                        fail_all(&mut groups, &mut prefill, &mut order, &mut free, &e);
                        continue;
                    }
                }
            }
        }
        // issue every idle group's next decode step
        let mut issue_err = None;
        for (gi, g) in groups.iter_mut().enumerate() {
            if g.inflight.is_some() || g.members.is_empty() {
                continue;
            }
            let seqs: Vec<(usize, Vec<u32>, usize)> = g.members.iter().map(|a| (a.slot, vec![a.last], a.pos)).collect();
            let samp: Vec<Sampling> = g.members.iter().map(|a| a.job.sampling.at(a.generated as u64)).collect();
            match eng.issue(g.lane, &seqs, true, &samp) {
                Ok(t) => {
                    g.inflight = Some((t, seqs.len()));
                    order.push_back(Pending::Group(gi));
                }
                Err(e) => {
                    issue_err = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = issue_err {
            fail_all(&mut groups, &mut prefill, &mut order, &mut free, &e);
            continue;
        }
        // collect the oldest step in flight
        let Some(p) = order.pop_front() else { continue };
        health.busy_since_ms.store(health.now_ms(), Ordering::SeqCst);
        match p {
            Pending::Prefill => {
                let (t, j, slot) = prefill.take().expect("prefill in flight");
                let r = eng.collect(t);
                health.busy_since_ms.store(0, Ordering::SeqCst);
                match r {
                    Ok(t) => {
                        let n_prompt = j.prompt.len();
                        let mut a = Active {
                            slot,
                            pos: n_prompt,
                            last: t[0],
                            generated: 0,
                            n_prompt,
                            detok: Detok::new(),
                            stop: StopScan::new(j.stop.clone()),
                            first_token: Instant::now(),
                            job: j,
                        };
                        match a.accept(t[0], tok.as_ref(), eos, max_len) {
                            // join the smallest group (its next issue picks the sequence up)
                            Outcome::Continue => groups.iter_mut().min_by_key(|g| g.members.len()).expect("a group").members.push(a),
                            Outcome::Finished(reason) => {
                                done(&a.job, a.n_prompt, a.generated, reason, Some(a.first_token));
                                free.push(slot);
                            }
                            Outcome::Gone => {
                                inflight.fetch_sub(1, Ordering::SeqCst);
                                free.push(slot);
                            }
                        }
                    }
                    Err(e) => {
                        fail(&j, StatusCode::INTERNAL_SERVER_ERROR, format!("prefill failed: {e}"));
                        free.push(slot);
                        fail_all(&mut groups, &mut prefill, &mut order, &mut free, &e);
                    }
                }
            }
            Pending::Group(gi) => {
                let (t, n_issued) = groups[gi].inflight.take().expect("group in flight");
                let r = eng.collect(t);
                health.busy_since_ms.store(0, Ordering::SeqCst);
                match r {
                    Ok(toks) => {
                        // members past n_issued joined while the step was in flight: untouched
                        let g = &mut groups[gi];
                        let mut keep = Vec::with_capacity(g.members.len());
                        for (i, mut a) in g.members.drain(..).enumerate() {
                            if i >= n_issued {
                                keep.push(a);
                                continue;
                            }
                            a.pos += 1;
                            a.last = toks[i];
                            match a.accept(toks[i], tok.as_ref(), eos, max_len) {
                                Outcome::Continue => keep.push(a),
                                Outcome::Finished(reason) => {
                                    done(&a.job, a.n_prompt, a.generated, reason, Some(a.first_token));
                                    free.push(a.slot);
                                }
                                Outcome::Gone => {
                                    inflight.fetch_sub(1, Ordering::SeqCst);
                                    free.push(a.slot);
                                }
                            }
                        }
                        g.members = keep;
                    }
                    Err(e) => fail_all(&mut groups, &mut prefill, &mut order, &mut free, &e),
                }
            }
        }
    }
}

/// A decode group: the sequences stepped together on one lane.
struct Group {
    lane: usize,
    members: Vec<Active>,
    /// The step in flight and how many members (a prefix) it covers.
    inflight: Option<(Ticket, usize)>,
}

enum Pending {
    Prefill,
    Group(usize),
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// An OpenAI-shaped error response.
fn api_error(status: StatusCode, message: impl Into<String>) -> Response {
    let kind = if status.is_client_error() { "invalid_request_error" } else { "server_error" };
    (status, Json(json!({"error": {"message": message.into(), "type": kind, "param": null, "code": null}}))).into_response()
}

/// The parameters this server does not implement: refused by name rather than silently ignored.
fn unsupported(req: &Value) -> Option<&'static str> {
    let num = |k: &str| req.get(k).and_then(|v| v.as_f64());
    if req.get("n").and_then(|v| v.as_u64()).is_some_and(|n| n != 1) {
        return Some("n");
    }
    if req.get("best_of").and_then(|v| v.as_u64()).is_some_and(|n| n != 1) {
        return Some("best_of");
    }
    if req.get("logprobs").is_some_and(|v| !v.is_null() && v != &json!(false) && v != &json!(0)) {
        return Some("logprobs");
    }
    if req.get("echo").and_then(|v| v.as_bool()) == Some(true) {
        return Some("echo");
    }
    if req.get("suffix").is_some_and(|v| !v.is_null()) {
        return Some("suffix");
    }
    if num("top_p").is_some_and(|p| p < 1.0) {
        return Some("top_p");
    }
    if req.get("top_k").and_then(|v| v.as_i64()).is_some_and(|k| k > 0) {
        return Some("top_k");
    }
    for k in ["presence_penalty", "frequency_penalty"] {
        if num(k).is_some_and(|p| p != 0.0) {
            return Some(if k == "presence_penalty" { "presence_penalty" } else { "frequency_penalty" });
        }
    }
    if num("repetition_penalty").is_some_and(|p| p != 1.0) {
        return Some("repetition_penalty");
    }
    if req.get("logit_bias").and_then(|v| v.as_object()).is_some_and(|o| !o.is_empty()) {
        return Some("logit_bias");
    }
    None
}

async fn completions(State(st): State<Arc<AppState>>, body: Result<Json<Value>, JsonRejection>) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}")),
    };
    if let Some(p) = unsupported(&req) {
        return api_error(StatusCode::BAD_REQUEST, format!("unsupported parameter: {p}"));
    }
    let prompt: Vec<u32> = match &req["prompt"] {
        Value::String(s) => {
            let (tok, s) = (st.tok.clone(), s.clone());
            match tokio::task::spawn_blocking(move || tok.encode_with_special_tokens(&s, true)).await {
                Ok(ids) => ids,
                Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("tokenizer: {e}")),
            }
        }
        Value::Array(a) if !a.is_empty() && a.iter().all(|v| v.is_u64()) => a.iter().map(|v| v.as_u64().unwrap().min(u32::MAX as u64) as u32).collect(),
        Value::Array(a) if a.len() == 1 && a[0].is_string() => {
            let (tok, s) = (st.tok.clone(), a[0].as_str().unwrap().to_string());
            match tokio::task::spawn_blocking(move || tok.encode_with_special_tokens(&s, true)).await {
                Ok(ids) => ids,
                Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("tokenizer: {e}")),
            }
        }
        Value::Array(a) if a.len() > 1 => return api_error(StatusCode::BAD_REQUEST, "batched prompts are not supported; send one prompt per request"),
        _ => return api_error(StatusCode::BAD_REQUEST, "prompt must be a string or a list of token ids"),
    };
    let max_tokens = match req.get("max_tokens") {
        None | Some(Value::Null) => 16,
        Some(v) => match v.as_u64() {
            Some(n) if n >= 1 => n as usize,
            _ => return api_error(StatusCode::BAD_REQUEST, "max_tokens must be a positive integer"),
        },
    };
    let min_tokens = req["min_tokens"].as_u64().unwrap_or(0) as usize;
    let temperature = req["temperature"].as_f64().unwrap_or(1.0);
    if !(0.0..=100.0).contains(&temperature) {
        return api_error(StatusCode::BAD_REQUEST, "temperature must be in [0, 100]");
    }
    let stop: Vec<String> = match &req["stop"] {
        Value::Null => Vec::new(),
        Value::String(s) if !s.is_empty() => vec![s.clone()],
        Value::String(_) => Vec::new(),
        Value::Array(a) if a.len() <= 4 && a.iter().all(|v| v.is_string()) => a.iter().filter_map(|v| v.as_str()).filter(|s| !s.is_empty()).map(String::from).collect(),
        _ => return api_error(StatusCode::BAD_REQUEST, "stop must be a string or up to 4 strings"),
    };
    // Validate here, not in the engine: an id past the vocab is an out-of-bounds embedding read on
    // the device, which poisons the CUDA context for every later request.
    if prompt.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "prompt is empty");
    }
    if let Some(bad) = prompt.iter().find(|&&t| t as usize >= st.vocab) {
        return api_error(StatusCode::BAD_REQUEST, format!("token id {bad} is outside the vocabulary ({})", st.vocab));
    }
    if prompt.len().checked_add(max_tokens).is_none_or(|n| n > st.max_len) {
        return api_error(
            StatusCode::BAD_REQUEST,
            format!("prompt ({} tokens) + max_tokens ({max_tokens}) exceeds the maximum length {}", prompt.len(), st.max_len),
        );
    }
    if st.health.dead.load(Ordering::SeqCst) {
        return api_error(StatusCode::SERVICE_UNAVAILABLE, "engine is not running");
    }
    if st.inflight.fetch_add(1, Ordering::SeqCst) >= st.max_queued {
        st.inflight.fetch_sub(1, Ordering::SeqCst);
        return api_error(StatusCode::TOO_MANY_REQUESTS, format!("server is at capacity ({} requests in flight)", st.max_queued));
    }
    let seed = req["seed"].as_u64().unwrap_or_else(|| st.next_id.load(Ordering::Relaxed).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let streaming = req["stream"].as_bool().unwrap_or(false);
    let usage = req["stream_options"]["include_usage"].as_bool().unwrap_or(false);
    let (tx, mut rx) = mpsc::unbounded_channel();
    let job = Job {
        prompt,
        max_tokens,
        min_tokens,
        ignore_eos: req["ignore_eos"].as_bool().unwrap_or(false),
        stop,
        sampling: Sampling { temperature: temperature as f32, seed, step: 0 },
        arrived: Instant::now(),
        tx,
    };
    if st.jobs.send(job).is_err() {
        st.inflight.fetch_sub(1, Ordering::SeqCst);
        return api_error(StatusCode::SERVICE_UNAVAILABLE, "engine is not running");
    }
    let id = format!("cmpl-{:x}", st.next_id.fetch_add(1, Ordering::Relaxed));
    let created = now_secs();
    let model = st.model.clone();
    if !streaming {
        let mut text = String::new();
        loop {
            match rx.recv().await {
                Some(Ev::Text(t)) => text.push_str(&t),
                Some(Ev::Done { prompt, completion, reason }) => {
                    return Json(json!({
                        "id": id, "object": "text_completion", "created": created, "model": model,
                        "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": reason}],
                        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
                    }))
                    .into_response();
                }
                Some(Ev::Err { status, message }) => return api_error(status, message),
                None => return api_error(StatusCode::INTERNAL_SERVER_ERROR, "engine dropped the request"),
            }
        }
    }
    struct S {
        rx: mpsc::UnboundedReceiver<Ev>,
        tail: VecDeque<String>,
        done: bool,
    }
    let init = S { rx, tail: VecDeque::new(), done: false };
    let body = stream::unfold(init, move |mut s| {
        let (id, model) = (id.clone(), model.clone());
        async move {
            if let Some(d) = s.tail.pop_front() {
                return Some((Ok::<Event, std::convert::Infallible>(Event::default().data(d)), s));
            }
            if s.done {
                return None;
            }
            let chunk = |text: &str, reason: Value| {
                json!({"id": id, "object": "text_completion", "created": created, "model": model,
                       "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": reason}]})
                .to_string()
            };
            match s.rx.recv().await {
                Some(Ev::Text(t)) => Some((Ok(Event::default().data(chunk(&t, Value::Null))), s)),
                Some(Ev::Done { prompt, completion, reason }) => {
                    s.done = true;
                    if usage {
                        s.tail.push_back(
                            json!({"id": id, "object": "text_completion", "created": created, "model": model, "choices": [],
                                   "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}})
                            .to_string(),
                        );
                    }
                    s.tail.push_back("[DONE]".into());
                    Some((Ok(Event::default().data(chunk("", json!(reason)))), s))
                }
                Some(Ev::Err { message, .. }) => {
                    // No [DONE] after an error: a client must not score a failed stream as complete.
                    s.done = true;
                    Some((Ok(Event::default().data(json!({"error": {"message": message, "type": "server_error"}}).to_string())), s))
                }
                None => None,
            }
        }
    });
    Sse::new(body).keep_alive(KeepAlive::default()).into_response()
}

async fn health_check(State(st): State<Arc<AppState>>) -> Response {
    let h = &st.health;
    if h.dead.load(Ordering::SeqCst) {
        return (StatusCode::SERVICE_UNAVAILABLE, "engine stopped").into_response();
    }
    let busy = h.busy_since_ms.load(Ordering::SeqCst);
    if busy != 0 && h.now_ms().saturating_sub(busy) > STEP_HANG_MS {
        return (StatusCode::SERVICE_UNAVAILABLE, "engine step stalled").into_response();
    }
    "ok".into_response()
}

async fn models(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": st.model, "object": "model", "owned_by": "plowrt", "max_model_len": st.max_len}]}))
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
    tracing::info!(target: "dsv41", "engine loaded in {:.0}s", t0.elapsed().as_secs_f32());
    let vocab = eng.cfg.vocab;
    if let Some((spec, reps, path)) = opts.rung_bench {
        let mut eng = eng;
        let report = eng.rung_bench(&spec, reps.max(1))?;
        if let Some(p) = path {
            std::fs::write(&p, &report)?;
            tracing::info!(target: "dsv41", "rung report written to {}", p.display());
        }
        return Ok(());
    }
    let (jtx, jrx) = mpsc::unbounded_channel();
    let inflight = Arc::new(AtomicUsize::new(0));
    let health = Arc::new(Health { dead: AtomicBool::new(false), busy_since_ms: AtomicU64::new(0), epoch: Instant::now() });
    {
        let (tok, inflight, health) = (tok.clone(), inflight.clone(), health.clone());
        std::thread::Builder::new().name("dsv41-sched".into()).spawn(move || scheduler(eng, jrx, max_len, tok, inflight, health))?;
    }
    let state = Arc::new(AppState {
        jobs: jtx,
        tok,
        model: opts.model_name,
        vocab,
        max_len,
        max_queued: opts.max_queued,
        inflight,
        health,
        next_id: AtomicU64::new(1),
    });
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/health", get(health_check))
        .route("/healthz", get(health_check))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", opts.port)).await?;
    tracing::info!(target: "dsv41", "serving on :{}", opts.port);
    axum::serve(listener, app).tcp_nodelay(true).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Detok, StopScan};
    use crate::text::tokenizer::{ByteTokenizer, Tokenize};

    /// Stream `text` byte by byte through the detokenizer, as the scheduler does per token.
    fn stream(text: &str) -> (Vec<String>, String) {
        let tok = ByteTokenizer;
        let mut d = Detok::new();
        let deltas: Vec<String> = tok.encode(text).into_iter().map(|id| d.push(&tok, id)).collect();
        let tail = d.flush(&tok);
        (deltas, tail)
    }

    #[test]
    fn detok_reassembles_multibyte_text_without_replacement_chars() {
        let text = "h\u{e9}llo \u{2192} \u{4e16}\u{754c}!";
        let (deltas, tail) = stream(text);
        let joined: String = deltas.concat() + &tail;
        assert_eq!(joined, text);
        assert!(deltas.iter().all(|d| !d.contains('\u{fffd}')), "a partial UTF-8 sequence leaked: {deltas:?}");
    }

    #[test]
    fn detok_flushes_a_trailing_incomplete_sequence() {
        let tok = ByteTokenizer;
        let mut d = Detok::new();
        let mut out = String::new();
        for id in "ab".bytes().map(u32::from).chain([0xE4]) {
            out.push_str(&d.push(&tok, id));
        }
        assert_eq!(out, "ab");
        assert_eq!(d.flush(&tok), "\u{fffd}");
    }

    #[test]
    fn stop_scan_truncates_at_the_earliest_stop_across_chunks() {
        let mut s = StopScan::new(vec!["STOP".into(), "\n\n".into()]);
        let mut out = String::new();
        let mut hit = false;
        for chunk in ["hello ", "wor", "ld S", "TO", "P and more"] {
            let (t, stopped) = s.feed(chunk);
            out.push_str(&t);
            if stopped {
                hit = true;
                break;
            }
        }
        assert!(hit);
        assert_eq!(out, "hello world ");
    }

    #[test]
    fn stop_scan_releases_held_text_that_never_matches() {
        let mut s = StopScan::new(vec!["END".into()]);
        let (a, _) = s.feed("abcE");
        let (b, _) = s.feed("N");
        let (c, stopped) = s.feed("x");
        assert!(!stopped);
        assert_eq!(a + &b + &c + &s.finish(), "abcENx");
    }

    #[test]
    fn stop_scan_without_stops_passes_everything() {
        let mut s = StopScan::new(Vec::new());
        assert_eq!(s.feed("anything"), ("anything".to_string(), false));
    }
}
