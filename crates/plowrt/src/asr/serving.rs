use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        DefaultBodyLimit, Multipart, State,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit, Semaphore};

use super::{
    frontend::{decode_wav, decode_wav_chunk, AudioError, MAX_SAMPLES, SAMPLE_RATE},
    FinalizationPolicy, Transcriber, Transcript, TranscriptionInput,
};
use crate::serve::session::RequestIds;

#[cfg(feature = "cuda")]
mod shared;
#[cfg(feature = "cuda")]
pub use shared::preload;

const BATCH_FORMATION_WINDOW: Duration = Duration::from_millis(5);

pub struct AsrServer {
    backend: Backend,
    uploads: Arc<Semaphore>,
    sessions: Arc<Semaphore>,
    /// Recordings sent as `append` uploads, by (model, `X-Session-Id`).
    recordings: parking_lot::Mutex<HashMap<(String, Arc<str>), Arc<tokio::sync::Mutex<Recording>>>>,
}

/// One session's recording so far (HTTP `append` uploads), until its `final` upload.
#[derive(Default)]
struct Recording {
    samples: Vec<f32>,
    windows: Arc<parking_lot::Mutex<WindowCache>>,
    used: Option<Instant>,
    /// The last partial transcript and when the next may run (`--asr-partial-duty`).
    partial: Option<(String, Instant)>,
}

/// Recordings a process keeps at once; past it the least recently used idle one goes.
const MAX_RECORDINGS: usize = 1024;

enum Backend {
    /// `plowrt asr`: one model on a private cohort engine.
    Cohort { model: String, mux: AsrMux, finalization: FinalizationPolicy },
    /// `plowrt serve`: any registry model with a causal audio pipeline, through its text mux.
    #[cfg(feature = "cuda")]
    Serve(Arc<crate::serve::AppState>),
}

/// Where one request's transcription runs.
#[derive(Clone)]
enum Route {
    Cohort(AsrMux),
    #[cfg(feature = "cuda")]
    Shared(Arc<shared::SharedAsr>, crate::serve::mux::ModelMux),
}

/// Encoder rows of a growing recording's completed attention windows, one entry per window, kept
/// for its later partial transcripts.
#[derive(Default)]
pub(crate) struct WindowCache {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    rows: Vec<Arc<[f32]>>,
    /// The last partial's output tokens: the next partial forces all but their tail.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    draft: Vec<u32>,
}

/// How one transcription runs beyond its audio.
#[derive(Default)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct AsrOpts {
    /// The answer a client waits on (a final); otherwise a revisable partial.
    pub final_pass: bool,
    /// With a session: the decoder rows are retained for, and resumed from, its requests.
    pub ids: Option<RequestIds>,
    /// A partial of a growing recording: reuse (and extend) its completed windows' encoder rows.
    pub windows: Option<Arc<parking_lot::Mutex<WindowCache>>>,
    /// Transcript text deltas as the decoder produces them.
    pub deltas: Option<mpsc::UnboundedSender<String>>,
    /// What the session's admission reused.
    pub report: Option<crate::serve::session::Report>,
}

impl Route {
    fn submit(
        &self,
        samples: Vec<f32>,
        language: Option<String>,
        context: String,
        cancel: Arc<AtomicBool>,
        opts: AsrOpts,
    ) -> Result<oneshot::Receiver<crate::Result<Transcript>>, SubmitError> {
        match self {
            Route::Cohort(mux) => {
                let _ = opts;
                mux.submit(samples, language, context, cancel)
            }
            #[cfg(feature = "cuda")]
            Route::Shared(asr, mux) => asr.submit(mux.clone(), samples, language, context, cancel, opts),
        }
    }

    /// Whether this route keeps session state (retained decoder rows, appended audio).
    fn sessions(&self) -> bool {
        #[cfg(feature = "cuda")]
        if matches!(self, Route::Shared(..)) {
            return true;
        }
        false
    }
}

#[derive(Clone)]
struct AsrMux {
    tx: mpsc::Sender<AsrJob>,
}

struct AsrJob {
    samples: Vec<f32>,
    language: Option<String>,
    context: String,
    cancel: Arc<AtomicBool>,
    respond: oneshot::Sender<crate::Result<Transcript>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubmitError {
    Full,
    Closed,
}

impl AsrMux {
    fn spawn(mut engine: Box<dyn Transcriber>) -> (Self, usize, FinalizationPolicy) {
        let batch_capacity = engine.batch_capacity().max(1);
        let finalization = engine.finalization_policy();
        let ingress_capacity = batch_capacity.saturating_mul(4).max(4);
        let (tx, mut rx) = mpsc::channel::<AsrJob>(ingress_capacity);
        std::thread::Builder::new()
            .name("plow-asr-engine".into())
            .spawn(move || {
                let mut cohort = VecDeque::with_capacity(batch_capacity);
                while let Some(first) = rx.blocking_recv() {
                    cohort.push_back(first);
                    if batch_capacity > 1 {
                        let deadline = Instant::now() + BATCH_FORMATION_WINDOW;
                        while cohort.len() < batch_capacity {
                            match rx.try_recv() {
                                Ok(job) => cohort.push_back(job),
                                Err(mpsc::error::TryRecvError::Disconnected) => break,
                                Err(mpsc::error::TryRecvError::Empty) => {
                                    let remaining =
                                        deadline.saturating_duration_since(Instant::now());
                                    if remaining.is_zero() {
                                        break;
                                    }
                                    std::thread::sleep(remaining.min(Duration::from_micros(100)));
                                }
                            }
                        }
                        cohort
                            .make_contiguous()
                            .sort_by_key(|job| std::cmp::Reverse(job.samples.len()));
                    }
                    let requests: Vec<_> = cohort
                        .iter()
                        .map(|job| TranscriptionInput {
                            samples: &job.samples,
                            language: job.language.as_deref(),
                            context: &job.context,
                            cancel: &job.cancel,
                        })
                        .collect();
                    match engine.transcribe_batch(&requests) {
                        Ok(results) if results.len() == cohort.len() => {
                            for (job, result) in cohort.drain(..).zip(results) {
                                let _ = job.respond.send(result);
                            }
                        }
                        Ok(results) => {
                            let error = crate::RuntimeError::Msg(format!(
                                "ASR engine returned {} results for {} requests",
                                results.len(),
                                cohort.len()
                            ));
                            fanout_batch_error(&mut cohort, error);
                        }
                        Err(error) => fanout_batch_error(&mut cohort, error),
                    }
                }
            })
            .expect("failed to spawn ASR engine thread");
        (Self { tx }, ingress_capacity, finalization)
    }

    fn submit(
        &self,
        samples: Vec<f32>,
        language: Option<String>,
        context: String,
        cancel: Arc<AtomicBool>,
    ) -> Result<oneshot::Receiver<crate::Result<Transcript>>, SubmitError> {
        let (respond, receive) = oneshot::channel();
        let job = AsrJob {
            samples,
            language,
            context,
            cancel,
            respond,
        };
        self.tx.try_send(job).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => SubmitError::Full,
            mpsc::error::TrySendError::Closed(_) => SubmitError::Closed,
        })?;
        Ok(receive)
    }
}

fn fanout_batch_error(cohort: &mut VecDeque<AsrJob>, error: crate::RuntimeError) {
    let message = error.to_string();
    for job in cohort.drain(..) {
        let error = match &error {
            crate::RuntimeError::Rejected(reason) => crate::RuntimeError::Rejected(reason.clone()),
            crate::RuntimeError::ContextLength(reason) => {
                crate::RuntimeError::ContextLength(reason.clone())
            }
            crate::RuntimeError::DeviceFault { info } => {
                crate::RuntimeError::DeviceFault { info: info.clone() }
            }
            _ => crate::RuntimeError::Msg(message.clone()),
        };
        let _ = job.respond.send(Err(error));
    }
}

struct Cancellation(Arc<AtomicBool>);
impl Drop for Cancellation {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl AsrServer {
    pub fn new(model: String, engine: impl Transcriber + 'static) -> Arc<Self> {
        let (mux, ingress_capacity, finalization) = AsrMux::spawn(Box::new(engine));
        Arc::new(Self {
            backend: Backend::Cohort { model, mux, finalization },
            uploads: Arc::new(Semaphore::new(ingress_capacity)),
            sessions: Arc::new(Semaphore::new(ingress_capacity)),
            recordings: Default::default(),
        })
    }

    /// Transcription for every model `state` serves whose packet declares a causal audio pipeline:
    /// the prompt and encoder run here, the decoder on the model's continuous-batching mux.
    #[cfg(feature = "cuda")]
    pub fn for_serve(state: Arc<crate::serve::AppState>) -> Arc<Self> {
        Arc::new(Self {
            backend: Backend::Serve(state),
            uploads: Arc::new(Semaphore::new(shared::UPLOADS)),
            sessions: Arc::new(Semaphore::new(shared::UPLOADS)),
            recordings: Default::default(),
        })
    }

    /// A session's recording, created on first use. Idle recordings go after the session TTL
    /// (60 s when retention is off), and past [`MAX_RECORDINGS`] the least recently used idle one.
    fn recording(&self, model: &str, session: &Arc<str>) -> Arc<tokio::sync::Mutex<Recording>> {
        let ttl = Some(crate::serve::session::retention_ttl())
            .filter(|t| !t.is_zero())
            .unwrap_or(Duration::from_secs(60));
        let now = Instant::now();
        let idle = |r: &Arc<tokio::sync::Mutex<Recording>>| r.try_lock().ok().map(|r| r.used);
        let mut map = self.recordings.lock();
        map.retain(|_, r| !matches!(idle(r), Some(Some(used)) if now.saturating_duration_since(used) >= ttl));
        let key = (model.to_owned(), session.clone());
        if map.len() >= MAX_RECORDINGS && !map.contains_key(&key) {
            let lru = map
                .iter()
                .filter_map(|(k, r)| Some((k.clone(), idle(r)?.unwrap_or(now))))
                .min_by_key(|&(_, used)| used)
                .map(|(k, _)| k);
            if let Some(k) = lru {
                map.remove(&k);
            }
        }
        map.entry(key).or_default().clone()
    }

    /// The served model's metrics (`plowrt serve` only).
    fn metrics(&self, _model: &str) -> Option<Arc<crate::obs::Metrics>> {
        match &self.backend {
            Backend::Cohort { .. } => None,
            #[cfg(feature = "cuda")]
            Backend::Serve(state) => Some(state.model_metrics(_model)),
        }
    }

    async fn route(&self, model: &str) -> Result<(Route, FinalizationPolicy), Response> {
        match &self.backend {
            Backend::Cohort { model: served, mux, finalization } => {
                if model != served {
                    return Err(failure(StatusCode::NOT_FOUND, "unknown ASR model"));
                }
                Ok((Route::Cohort(mux.clone()), *finalization))
            }
            #[cfg(feature = "cuda")]
            Backend::Serve(state) => shared::route(state, model).await,
        }
    }
    pub fn router(self: Arc<Self>, websocket: bool) -> Router {
        self.transcription_router(websocket)
            .route("/health", get(|| async { StatusCode::OK }))
    }

    /// The transcription routes alone, to merge into another server's router.
    pub fn transcription_router(self: Arc<Self>, websocket: bool) -> Router {
        let mut router = Router::new().route("/v1/audio/transcriptions", post(transcription));
        if websocket {
            router = router.route("/v1/audio/transcriptions/stream", get(upgrade));
        }
        router
            .layer(DefaultBodyLimit::max(4 * 1024 * 1024))
            .with_state(self)
    }
}

fn failure(status: StatusCode, message: impl ToString) -> Response {
    (
        status,
        Json(json!({"error":{"message":message.to_string(),"type":"transcription_error"}})),
    )
        .into_response()
}

fn runtime_failure(error: crate::RuntimeError) -> Response {
    let status = match &error {
        crate::RuntimeError::Rejected(_) | crate::RuntimeError::ContextLength(_) => {
            StatusCode::BAD_REQUEST
        }
        crate::RuntimeError::DeviceFault { ref info } if info.fatal => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    failure(status, error)
}

async fn transcription(
    State(state): State<Arc<AsrServer>>,
    headers: axum::http::HeaderMap,
    multipart: Multipart,
) -> Response {
    let mut ids = match RequestIds::from_headers(&headers) {
        Ok(ids) => ids,
        Err(e) => return failure(StatusCode::BAD_REQUEST, e),
    };
    let mut response = transcribe_upload(state, multipart, &mut ids).await;
    ids.stamp(&mut response);
    response
}

fn form_bool(fields: &std::collections::HashMap<String, String>, name: &str) -> Result<bool, Response> {
    match fields.get(name).map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        None | Some("false" | "0") => Ok(false),
        Some("true" | "1") => Ok(true),
        Some(_) => Err(failure(StatusCode::BAD_REQUEST, format!("{name} must be true or false"))),
    }
}

async fn transcribe_upload(state: Arc<AsrServer>, mut multipart: Multipart, ids: &mut RequestIds) -> Response {
    let upload = match state.uploads.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return failure(StatusCode::TOO_MANY_REQUESTS, "too many ASR uploads"),
    };
    let mut file = None;
    let mut fields = std::collections::HashMap::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let field = match tokio::time::timeout_at(deadline, multipart.next_field()).await {
            Ok(Ok(Some(f))) => f,
            Ok(Ok(None)) => break,
            Ok(Err(e)) => return failure(e.status(), e),
            Err(_) => return failure(StatusCode::REQUEST_TIMEOUT, "upload timed out"),
        };
        let name = field.name().unwrap_or("").to_owned();
        if name == "file" {
            if file.is_some() {
                return failure(StatusCode::BAD_REQUEST, "duplicate file");
            }
            file = match tokio::time::timeout_at(deadline, field.bytes()).await {
                Ok(Ok(b)) => Some(b),
                Ok(Err(e)) => return failure(e.status(), e),
                Err(_) => return failure(StatusCode::REQUEST_TIMEOUT, "upload timed out"),
            };
        } else {
            if !matches!(
                name.as_str(),
                "model" | "language" | "prompt" | "response_format" | "temperature" | "stream" | "append" | "final"
                    | "offset" | "session_id" | "prompt_cache_key" | "turn_id" | "turn_budget_ms"
            ) || fields.contains_key(&name)
            {
                return failure(StatusCode::BAD_REQUEST, "unknown or duplicate field");
            }
            let value = match tokio::time::timeout_at(deadline, field.text()).await {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => return failure(e.status(), e),
                Err(_) => return failure(StatusCode::REQUEST_TIMEOUT, "upload timed out"),
            };
            fields.insert(name, value);
        }
    }
    // Form fields as the JSON endpoints' body fallbacks: `session_id` > `X-Session-Id`; `turn_id` / `turn_budget_ms` below their headers.
    let route_fields = crate::serve::session::RouteFields {
        session_id: fields.remove("session_id"),
        prompt_cache_key: fields.remove("prompt_cache_key"),
        metadata: Some(
            ["turn_id", "turn_budget_ms"]
                .into_iter()
                .filter_map(|k| Some((k.to_owned(), serde_json::Value::String(fields.remove(k)?))))
                .collect(),
        ),
        trace: None,
    };
    if let Err(e) = ids.apply_body(&route_fields) {
        return failure(StatusCode::BAD_REQUEST, e);
    }
    if let Some(r) = crate::serve::overload::gate(ids) {
        return r;
    }
    let ids = &*ids;
    let Some(model) = fields.get("model").cloned() else {
        return failure(StatusCode::BAD_REQUEST, "model is required");
    };
    let route = match state.route(&model).await {
        Ok((route, _)) => route,
        Err(response) => return response,
    };
    let format = fields
        .remove("response_format")
        .unwrap_or_else(|| "json".into());
    if !matches!(format.as_str(), "json" | "text") {
        return failure(
            StatusCode::BAD_REQUEST,
            "response_format must be json or text",
        );
    }
    if fields
        .get("temperature")
        .is_some_and(|v| v.parse::<f32>().ok() != Some(0.0))
    {
        return failure(StatusCode::BAD_REQUEST, "ASR supports greedy decoding only");
    }
    let (stream, append, finish) = match (form_bool(&fields, "stream"), form_bool(&fields, "append"), form_bool(&fields, "final")) {
        (Ok(s), Ok(a), Ok(f)) => (s, a, f),
        (Err(e), ..) | (_, Err(e), _) | (.., Err(e)) => return e,
    };
    let recorded = append || finish;
    if recorded && (ids.session.is_none() || !route.sessions()) {
        return failure(StatusCode::BAD_REQUEST, "append/final need an X-Session-Id on plowrt serve");
    }
    let Some(in_flight) = ids.begin(&model) else {
        return failure(StatusCode::CONFLICT, format!("request {} is already in flight in this session", ids.request));
    };
    if file.is_none() && !finish {
        return failure(StatusCode::BAD_REQUEST, "file is required");
    }
    let offset = match fields.get("offset").map(|v| v.trim().parse::<usize>()) {
        None => None,
        Some(Ok(offset)) if recorded => Some(offset),
        Some(_) => return failure(StatusCode::BAD_REQUEST, "offset must be a sample count, with append or final"),
    };
    let language = fields.remove("language");
    let context = fields.remove("prompt").unwrap_or_default();
    let samples = match file {
        None => Vec::new(),
        Some(file) => match tokio::task::spawn_blocking(move || if recorded { decode_wav_chunk(&file) } else { decode_wav(&file) }).await {
            Ok(Ok(samples)) => samples,
            Ok(Err(error)) => {
                let status = match &error {
                    AudioError::Invalid(_) => StatusCode::BAD_REQUEST,
                    AudioError::Unsupported(_) => StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    AudioError::TooLong => StatusCode::PAYLOAD_TOO_LARGE,
                };
                return failure(status, error);
            }
            Err(error) => return failure(StatusCode::INTERNAL_SERVER_ERROR, error),
        },
    };
    drop(upload);
    let finals = !recorded || finish;
    let mut run = crate::serve::turns::StageRun::start(
        ids,
        crate::serve::turns::Kind::Asr,
        &model,
        state.metrics(&model),
        Instant::now(),
        finals,
    );
    let ids = &RequestIds { turn_key: run.key(), ..ids.clone() };
    let (report, report_rx) = ids.report();
    let mut opts = AsrOpts { final_pass: finals, ids: Some(ids.clone()), report, ..Default::default() };
    // A session recording: append, then transcribe all of it (partial) or finish it (final).
    let mut recorded_samples = 0;
    let (samples, mut recording) = if recorded {
        let session = ids.session.clone().expect("checked");
        let recording = state.recording(&model, &session);
        let mut rec = recording.clone().lock_owned().await;
        // The client's count of samples it sent before this upload: a recording this process lost
        // (restart, failover, TTL) or one out of step is refused with what it holds, and the client
        // resends from there.
        if offset.is_some_and(|o| o != rec.samples.len()) {
            let expected = rec.samples.len();
            return (
                StatusCode::CONFLICT,
                Json(json!({"error":{"message":format!("session recording holds {expected} samples; resend from offset {expected}"),
                    "type":"transcription_error","code":"session_audio_offset"},"expected_offset":expected})),
            )
                .into_response();
        }
        if rec.samples.len() + samples.len() > MAX_SAMPLES {
            return failure(StatusCode::PAYLOAD_TOO_LARGE, "session audio exceeds 30 seconds");
        }
        rec.samples.extend_from_slice(&samples);
        rec.used = Some(Instant::now());
        recorded_samples = rec.samples.len();
        let all = if finish {
            rec.windows = Default::default();
            rec.partial = None;
            std::mem::take(&mut rec.samples)
        } else {
            // Inside the session's partial duty cycle: answer the last partial again.
            if let Some((text, _)) = rec.partial.as_ref().filter(|(_, next)| !stream && Instant::now() < *next) {
                let reply = json!({"text": text, "final": false, "offset": recorded_samples});
                return if format == "text" { text.clone().into_response() } else { Json(reply).into_response() };
            }
            opts.windows = Some(rec.windows.clone());
            rec.samples.clone()
        };
        if !finish && all.len() < SAMPLE_RATE as usize / 2 {
            let reply = json!({"text": "", "final": false, "offset": recorded_samples});
            return if stream {
                sse_events(vec![transcript_event("transcript.text.done", ids, reply)])
            } else if format == "text" {
                String::new().into_response()
            } else {
                Json(reply).into_response()
            };
        }
        (all, Some(rec))
    } else {
        (samples, None)
    };
    let cancel = Cancellation(Arc::new(AtomicBool::new(false)));
    let (delta_tx, delta_rx) = mpsc::unbounded_channel();
    if stream {
        opts.deltas = Some(delta_tx);
    }
    let work = match route.submit(samples, language, context, cancel.0.clone(), opts) {
        Ok(work) => work,
        Err(SubmitError::Full) => return failure(StatusCode::TOO_MANY_REQUESTS, "ASR queue full"),
        Err(SubmitError::Closed) => {
            return failure(StatusCode::SERVICE_UNAVAILABLE, "ASR engine unavailable")
        }
    };
    let offset = recorded.then_some(recorded_samples);
    if stream {
        let cache = crate::serve::session::CacheOutcome::received(report_rx).await;
        run.admitted(cache.and_then(|c| c.at));
        let stamped = run.headers();
        let mut response = sse_transcript(work, delta_rx, ids.clone(), finals, offset, cache, (in_flight, cancel, recording), run);
        response.headers_mut().extend(stamped);
        if let Some(cache) = cache {
            cache.stamp(&mut response);
        }
        return response;
    }
    let started = Instant::now();
    let result = work.await;
    if let (Some(rec), false, Ok(Ok(result))) = (recording.as_mut(), finals, &result) {
        let duty = crate::config::RuntimeConfig::get().asr_partial_duty.clamp(0.01, 1.0) * if crate::serve::overload::level() >= 1 { 0.5 } else { 1.0 };
        let rest = started.elapsed().mul_f64(1.0 / duty - 1.0);
        rec.partial = Some((result.text.clone(), Instant::now() + rest));
    }
    drop((in_flight, recording));
    let cache = crate::serve::session::CacheOutcome::received(report_rx).await;
    run.admitted(cache.and_then(|c| c.at));
    if matches!(result, Ok(Ok(_))) {
        run.first();
        run.done();
    }
    let mut response = match result {
        Ok(Ok(result)) if format == "text" => result.text.into_response(),
        Ok(Ok(result)) if recorded => Json(json!({"text":result.text,"final":finals,"offset":recorded_samples})).into_response(),
        Ok(Ok(result)) => Json(json!({"text":result.text})).into_response(),
        Ok(Err(error)) => return runtime_failure(error),
        Err(error) => return failure(StatusCode::SERVICE_UNAVAILABLE, error),
    };
    run.stamp(&mut response);
    if let Some(cache) = cache {
        cache.stamp(&mut response);
    }
    response
}

fn transcript_event(kind: &str, ids: &RequestIds, mut body: serde_json::Value) -> axum::response::sse::Event {
    body["type"] = kind.into();
    body["request_id"] = ids.request.as_ref().into();
    if let Some(session) = &ids.session {
        body["session_id"] = session.as_ref().into();
    }
    if let Some(key) = &ids.turn_key {
        body["turn_id"] = key.turn.as_ref().into();
    }
    axum::response::sse::Event::default().data(body.to_string())
}

fn sse_events(events: Vec<axum::response::sse::Event>) -> Response {
    let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
    axum::response::sse::Sse::new(stream).into_response()
}

/// OpenAI streamed transcription: `transcript.text.delta` events as the decoder produces text, then
/// `transcript.text.done` with the whole transcript (`final: false` for a session's partial).
fn sse_transcript<H: Send + 'static>(
    mut work: oneshot::Receiver<crate::Result<Transcript>>,
    mut deltas: mpsc::UnboundedReceiver<String>,
    ids: RequestIds,
    finals: bool,
    offset: Option<usize>,
    cache: Option<crate::serve::session::CacheOutcome>,
    held: H,
    mut run: crate::serve::turns::StageRun,
) -> Response {
    let (tx, mut rx) = mpsc::channel::<axum::response::sse::Event>(64);
    tokio::spawn(async move {
        let _held = held;
        let mut shown = String::new();
        let result = loop {
            tokio::select! {
                Some(delta) = deltas.recv() => {
                    shown.push_str(&delta);
                    if tx.send(transcript_event("transcript.text.delta", &ids, json!({"delta": delta}))).await.is_err() {
                        return;
                    }
                }
                result = &mut work => break result,
            }
        };
        let event = match result {
            Ok(Ok(result)) => {
                while let Ok(delta) = deltas.try_recv() {
                    shown.push_str(&delta);
                    let _ = tx.send(transcript_event("transcript.text.delta", &ids, json!({"delta": delta}))).await;
                }
                if let Some(rest) = result.text.strip_prefix(shown.as_str()).filter(|r| !r.is_empty()) {
                    let _ = tx.send(transcript_event("transcript.text.delta", &ids, json!({"delta": rest}))).await;
                }
                let mut done = json!({"text": result.text, "language": result.language, "final": finals});
                if let Some(offset) = offset {
                    done["offset"] = offset.into();
                }
                if let Some(cache) = cache {
                    done["session_cache"] = cache.status.as_str().into();
                    done["cached_tokens"] = cache.rows.into();
                }
                run.first();
                run.done();
                let _ = tx.send(transcript_event("transcript.text.done", &ids, done)).await;
                run.sse_comment()
            }
            Ok(Err(error)) => transcript_event("error", &ids, json!({"message": error.to_string()})),
            Err(error) => transcript_event("error", &ids, json!({"message": error.to_string()})),
        };
        let _ = tx.send(event).await;
    });
    let stream = futures::stream::poll_fn(move |cx| rx.poll_recv(cx).map(|e| e.map(Ok::<_, std::convert::Infallible>)));
    axum::response::sse::Sse::new(stream).into_response()
}

async fn upgrade(
    State(state): State<Arc<AsrServer>>,
    headers: axum::http::HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    let mut ids = match RequestIds::from_headers(&headers) {
        Ok(ids) => ids,
        Err(e) => return failure(StatusCode::BAD_REQUEST, e),
    };
    if let Some(r) = crate::serve::overload::gate(&ids) {
        return r;
    }
    let permit = match state.sessions.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return failure(StatusCode::TOO_MANY_REQUESTS, "too many ASR sessions"),
    };
    // Each connection is a session: its partials resume the decoder rows the last one retained.
    ids.session.get_or_insert_with(|| crate::serve::session::minted::session().into());
    let echo = ids.clone();
    let mut response = ws
        .max_message_size(65536)
        .max_frame_size(65536)
        .on_upgrade(move |socket| async move {
            stream(state, socket, permit, ids).await;
        })
        .into_response();
    echo.stamp(&mut response);
    response
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Start {
    #[serde(rename = "type")]
    kind: String,
    version: u32,
    model: String,
    sample_rate: u32,
    format: String,
    language: Option<String>,
    #[serde(default)]
    prompt: String,
    /// Revisable partial transcripts while audio arrives (`"type":"partial"` events).
    #[serde(default)]
    partials: bool,
}

/// Audio between partial transcriptions. On `plowrt serve` a partial encodes only the open encoder
/// window and resumes the connection's retained decoder rows; the cohort engine re-transcribes the
/// whole buffer.
const PARTIAL_STRIDE: usize = SAMPLE_RATE as usize;

fn common_prefix_bytes(a: &str, b: &str) -> usize {
    a.char_indices().zip(b.chars()).take_while(|((_, x), y)| x == y).map(|((i, x), _)| i + x.len_utf8()).last().unwrap_or(0)
}

async fn send(socket: &mut WebSocket, value: serde_json::Value) -> bool {
    matches!(
        tokio::time::timeout(
            Duration::from_secs(30),
            socket.send(Message::Text(value.to_string()))
        )
        .await,
        Ok(Ok(()))
    )
}

fn append_final_padding(samples: &mut Vec<f32>, count: usize, amplitude: f32) {
    if amplitude == 0.0 {
        samples.resize(samples.len() + count, 0.0);
        return;
    }
    let mut state = 0x9e3779b97f4a7c15u64;
    samples.reserve(count);
    for _ in 0..count {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let centered = ((state >> 48) as i32 - 32768) as f32 / 32768.0;
        samples.push(centered * amplitude);
    }
}

async fn stream(state: Arc<AsrServer>, mut socket: WebSocket, _permit: OwnedSemaphorePermit, ids: RequestIds) {
    let cancel = Cancellation(Arc::new(AtomicBool::new(false)));
    let Some(Ok(Message::Text(text))) =
        tokio::time::timeout(Duration::from_secs(30), socket.recv())
            .await
            .ok()
            .flatten()
    else {
        return;
    };
    let routed = match serde_json::from_str::<Start>(&text) {
        Ok(s)
            if s.kind == "start"
                && s.version == 1
                && s.sample_rate == SAMPLE_RATE
                && s.format == "pcm_s16le" =>
        {
            state.route(&s.model).await.ok().map(|r| (s, r))
        }
        _ => None,
    };
    let (start, (route, finalization)) = match routed {
        Some(routed) => routed,
        None => {
            send(
                &mut socket,
                json!({"type":"error","message":"invalid start","terminal":true}),
            )
            .await;
            return;
        }
    };
    let language = start.language.clone();
    let partials = start.partials;
    let session = ids.session.clone().unwrap_or_default();
    let windows: Arc<parking_lot::Mutex<WindowCache>> = Default::default();
    let request = |final_pass, turn_key: Option<crate::serve::turns::TurnKey>| AsrOpts {
        final_pass,
        ids: Some(RequestIds { turn_key, ..ids.with_new_request() }),
        windows: (!final_pass).then(|| windows.clone()),
        deltas: None,
        report: None,
    };
    let max_audio_samples = MAX_SAMPLES.saturating_sub(finalization.final_padding_samples);
    let initial_credit = 16000usize.min(max_audio_samples);
    if !send(&mut socket,json!({"type":"ready","version":1,"session_id":&*session,"request_id":&*ids.request,
        "sample_rate":SAMPLE_RATE,"format":"pcm_s16le","max_chunk_bytes":32000,"credit_samples":initial_credit,
        "max_audio_samples":max_audio_samples,"partial_mode":if partials {"revision"} else {"final_only"}})).await{return;}
    let mut samples = Vec::new();
    let mut sequence = 0u64;
    let mut credit = initial_credit;
    let (mut revision, mut last_partial, mut partial_at) = (0u64, String::new(), 0usize);
    let mut pending: Option<oneshot::Receiver<crate::Result<Transcript>>> = None;
    loop {
        let message = tokio::select! {
            m = tokio::time::timeout(Duration::from_secs(30), socket.recv()) => match m {
                Ok(Some(Ok(m))) => m,
                _ => return,
            },
            Ok(result) = async { pending.as_mut().expect("guarded").await }, if pending.is_some() => {
                pending = None;
                if let Err(error) = &result {
                    tracing::warn!(%error, samples = samples.len(), "ASR partial failed");
                }
                if let Ok(result) = result {
                    revision += 1;
                    let stable = common_prefix_bytes(&last_partial, &result.text);
                    if !send(&mut socket, json!({"type":"partial","revision":revision,"text":result.text,
                        "language":result.language,"stable_prefix_bytes":stable})).await {
                        return;
                    }
                    last_partial = result.text;
                }
                continue;
            }
        };
        let finish = match message {
            Message::Binary(bytes) => {
                if bytes.len() < 10
                    || bytes.len() > 32008
                    || (bytes.len() - 8) % 2 != 0
                    || u64::from_le_bytes(bytes[..8].try_into().unwrap()) != sequence
                    || (bytes.len() - 8) / 2 > credit
                    || samples.len() + (bytes.len() - 8) / 2 > max_audio_samples
                {
                    send(&mut socket,json!({"type":"error","message":"invalid PCM sequence or credit/length exceeded","terminal":true})).await;
                    return;
                }
                sequence += 1;
                credit -= (bytes.len() - 8) / 2;
                samples.extend(
                    bytes[8..]
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0),
                );
                false
            }
            Message::Text(text) => match serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v["type"].as_str().map(str::to_owned))
                .as_deref()
            {
                Some("finish") => true,
                Some("cancel") => return,
                _ => {
                    send(&mut socket,json!({"type":"error","message":"expected finish or cancel","terminal":true})).await;
                    return;
                }
            },
            Message::Close(_) => return,
            Message::Ping(_) | Message::Pong(_) => continue,
        };
        if finish {
            if samples.len() < SAMPLE_RATE as usize / 2 {
                send(&mut socket,json!({"type":"error","message":"audio must contain at least 0.5 seconds","terminal":true})).await;
                return;
            }
            append_final_padding(
                &mut samples,
                finalization.final_padding_samples,
                finalization.final_padding_amplitude,
            );
            // The final supersedes an in-flight partial; dropping its receiver cancels it.
            drop(pending.take());
            let mut run = crate::serve::turns::StageRun::start(
                &ids,
                crate::serve::turns::Kind::Asr,
                &start.model,
                state.metrics(&start.model),
                Instant::now(),
                true,
            );
            let mut work = match route.submit(samples, language, start.prompt, cancel.0.clone(), request(true, run.key())) {
                Ok(work) => work,
                Err(SubmitError::Full) => {
                    send(
                        &mut socket,
                        json!({"type":"error","message":"ASR queue full","terminal":true}),
                    )
                    .await;
                    return;
                }
                Err(SubmitError::Closed) => {
                    send(
                        &mut socket,
                        json!({"type":"error","message":"ASR engine unavailable","terminal":true}),
                    )
                    .await;
                    return;
                }
            };
            let result = loop {
                tokio::select! {
                    result=&mut work=>break result,
                    incoming=socket.recv()=>match incoming {
                        Some(Ok(Message::Ping(_)|Message::Pong(_)))=>{},
                        other=>{
                            let cancelled=matches!(&other,Some(Ok(Message::Text(text))) if serde_json::from_str::<serde_json::Value>(text)
                                .ok().is_some_and(|v|v["type"]=="cancel"));
                            cancel.0.store(true,Ordering::Relaxed);
                            let _=work.await;
                            if !cancelled {send(&mut socket,json!({"type":"error","message":"input sent after finish or invalid event","terminal":true})).await;}
                            return;
                        }
                    }
                }
            };
            match result {
                Ok(Ok(result)) => {
                    run.first();
                    run.done();
                    send(
                        &mut socket,
                        json!({"type":"final","revision":revision + 1,
                        "text":result.text,"language":result.language,
                        "stable_prefix_bytes":result.text.len(),
                        "turn_id":run.turn_id.as_deref(),"traceparent":run.traceparent(),
                        "server_timing":run.timing().header()}),
                    )
                    .await;
                }
                Ok(Err(error)) => {
                    send(
                        &mut socket,
                        json!({"type":"error","message":error.to_string(),"terminal":true}),
                    )
                    .await;
                }
                Err(_) => {
                    send(&mut socket,json!({"type":"error","message":"ASR engine response channel closed","terminal":true})).await;
                }
            }
            return;
        }
        if partials && pending.is_none() && samples.len() >= partial_at + PARTIAL_STRIDE {
            partial_at = samples.len();
            // A full queue skips this partial; the next stride retries.
            pending = route.submit(samples.clone(), language.clone(), start.prompt.clone(), cancel.0.clone(), request(false, None)).ok();
        }
        let grant = (16000 - samples.len() % 16000)
            .min(max_audio_samples - samples.len())
            .saturating_sub(credit);
        credit += grant;
        if grant > 0 && !send(&mut socket, json!({"type":"credit","credit_samples":grant})).await {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use futures::{SinkExt, StreamExt};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[test]
    fn final_padding_noise_is_bounded_and_deterministic() {
        let mut left = vec![0.5];
        let mut right = left.clone();
        append_final_padding(&mut left, 1024, 100.0 / 32768.0);
        append_final_padding(&mut right, 1024, 100.0 / 32768.0);
        assert_eq!(left, right);
        assert!(left[1..].iter().any(|sample| *sample != 0.0));
        assert!(left[1..]
            .iter()
            .all(|sample| sample.abs() <= 100.0 / 32768.0));
    }

    #[tokio::test]
    async fn mux_forms_one_batch_and_preserves_member_results() {
        struct Batched {
            sizes: Arc<std::sync::Mutex<Vec<usize>>>,
            sample_lengths: Arc<std::sync::Mutex<Vec<usize>>>,
        }
        impl Transcriber for Batched {
            fn language(&self, language: Option<&str>) -> crate::Result<Option<String>> {
                Ok(language.map(str::to_owned))
            }
            fn batch_capacity(&self) -> usize {
                4
            }
            fn transcribe(
                &mut self,
                _: &[f32],
                _: Option<&str>,
                _: &str,
                _: &AtomicBool,
            ) -> crate::Result<Transcript> {
                unreachable!("batch override must be used")
            }
            fn transcribe_batch(
                &mut self,
                requests: &[TranscriptionInput<'_>],
            ) -> crate::Result<Vec<crate::Result<Transcript>>> {
                self.sizes.lock().unwrap().push(requests.len());
                self.sample_lengths
                    .lock()
                    .unwrap()
                    .extend(requests.iter().map(|request| request.samples.len()));
                Ok(requests
                    .iter()
                    .map(|request| {
                        if request.cancel.load(Ordering::Relaxed) {
                            Err(crate::RuntimeError::Rejected("cancelled".into()))
                        } else {
                            Ok(Transcript {
                                text: request.context.to_owned(),
                                language: request.language.map(str::to_owned),
                            })
                        }
                    })
                    .collect())
            }
        }
        let sizes = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sample_lengths = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (mux, _, _) = AsrMux::spawn(Box::new(Batched {
            sizes: sizes.clone(),
            sample_lengths: sample_lengths.clone(),
        }));
        let cancels: Vec<_> = (0..4).map(|i| Arc::new(AtomicBool::new(i == 2))).collect();
        let mut replies = Vec::new();
        for (i, cancel) in cancels.into_iter().enumerate() {
            replies.push(
                mux.submit(
                    vec![i as f32; i + 1],
                    Some(format!("language-{i}")),
                    format!("request-{i}"),
                    cancel,
                )
                .unwrap(),
            );
        }
        for (i, reply) in replies.into_iter().enumerate() {
            let result = reply.await.unwrap();
            if i == 2 {
                assert!(result.is_err());
            } else {
                let result = result.unwrap();
                assert_eq!(result.text, format!("request-{i}"));
                assert_eq!(
                    result.language.as_deref(),
                    Some(format!("language-{i}").as_str())
                );
            }
        }
        assert_eq!(*sizes.lock().unwrap(), [4]);
        assert_eq!(*sample_lengths.lock().unwrap(), [4, 3, 2, 1]);
    }

    #[tokio::test]
    async fn mux_rejects_only_after_its_bounded_queue_is_full() {
        struct Held {
            started: Arc<tokio::sync::Notify>,
            release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
        }
        impl Transcriber for Held {
            fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> {
                Ok(None)
            }
            fn transcribe(
                &mut self,
                _: &[f32],
                _: Option<&str>,
                _: &str,
                _: &AtomicBool,
            ) -> crate::Result<Transcript> {
                self.started.notify_one();
                let (lock, condition) = &*self.release;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = condition.wait(released).unwrap();
                }
                Ok(Transcript {
                    text: "done".into(),
                    language: None,
                })
            }
        }
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let (mux, ingress_capacity, _) = AsrMux::spawn(Box::new(Held {
            started: started.clone(),
            release: release.clone(),
        }));
        let submit = || {
            mux.submit(
                vec![0.0],
                None,
                String::new(),
                Arc::new(AtomicBool::new(false)),
            )
        };
        let active = submit().unwrap();
        tokio::time::timeout(Duration::from_secs(5), started.notified())
            .await
            .unwrap();
        let queued: Vec<_> = (0..ingress_capacity).map(|_| submit().unwrap()).collect();
        assert_eq!(submit().unwrap_err(), SubmitError::Full);
        *release.0.lock().unwrap() = true;
        release.1.notify_one();
        assert!(active.await.unwrap().is_ok());
        for reply in queued {
            assert!(reply.await.unwrap().is_ok());
        }
    }

    struct Fake;
    impl Transcriber for Fake {
        fn language(&self, language: Option<&str>) -> crate::Result<Option<String>> {
            if language.is_some_and(|l| l != "English") {
                return Err(crate::RuntimeError::Rejected("language".into()));
            }
            Ok(language.map(str::to_owned))
        }
        fn transcribe(
            &mut self,
            samples: &[f32],
            language: Option<&str>,
            _: &str,
            cancel: &AtomicBool,
        ) -> crate::Result<Transcript> {
            assert!(samples.len() >= 8000);
            std::thread::sleep(Duration::from_millis(30));
            if cancel.load(Ordering::Relaxed) {
                return Err(crate::RuntimeError::Rejected("cancelled".into()));
            }
            Ok(Transcript {
                text: "hello".into(),
                language: self.language(language)?,
            })
        }
    }

    fn request(model: &str, format: &str) -> Request<Body> {
        request_wav(model, format, 16000, 8000)
    }

    fn request_wav(model: &str, format: &str, sample_rate: u32, samples: usize) -> Request<Body> {
        let mut wav = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(
            &mut wav,
            hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..samples {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();
        let mut body=format!("--audio\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n--audio\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\n{format}\r\n--audio\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\nContent-Type: audio/wav\r\n\r\n").into_bytes();
        body.extend(wav.into_inner());
        body.extend(b"\r\n--audio--\r\n");
        Request::post("/v1/audio/transcriptions")
            .header("content-type", "multipart/form-data; boundary=audio")
            .body(Body::from(body))
            .unwrap()
    }

    #[tokio::test]
    async fn offline_api_formats_and_admission() {
        let server = AsrServer::new("test".into(), Fake);
        let app = server.clone().router(false);
        for (format, expected) in [("json", r#"{"text":"hello"}"#), ("text", "hello")] {
            let response = app.clone().oneshot(request("test", format)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                response.into_body().collect().await.unwrap().to_bytes(),
                expected
            );
        }
        assert_eq!(
            app.clone()
                .oneshot(request("missing", "json"))
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            app.clone()
                .oneshot(request("test", "srt"))
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
        for (rate, samples, status) in [
            (8000, 8000, StatusCode::UNSUPPORTED_MEDIA_TYPE),
            (16000, 7999, StatusCode::BAD_REQUEST),
            (16000, 480001, StatusCode::PAYLOAD_TOO_LARGE),
        ] {
            assert_eq!(
                app.clone()
                    .oneshot(request_wav("test", "json", rate, samples))
                    .await
                    .unwrap()
                    .status(),
                status
            );
        }
        let upload_capacity = server.uploads.available_permits();
        let permit = server
            .uploads
            .clone()
            .acquire_many_owned(upload_capacity as u32)
            .await
            .unwrap();
        assert_eq!(
            app.clone()
                .oneshot(request("test", "json"))
                .await
                .unwrap()
                .status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        drop(permit);
        assert_eq!(
            app.oneshot(
                Request::get("/v1/audio/transcriptions/stream")
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap()
            .status(),
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn websocket_finishes_and_rejects_bad_sequence() {
        use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = AsrServer::new("test".into(), Fake).router(true);
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let url = format!("ws://{address}/v1/audio/transcriptions/stream");
        for invalid in [false, true] {
            let (mut socket, _) = connect_async(&url).await.unwrap();
            socket.send(ClientMessage::Text(json!({"type":"start","version":1,"model":"test","sample_rate":16000,"format":"pcm_s16le"}).to_string())).await.unwrap();
            let ready = socket.next().await.unwrap().unwrap().into_text().unwrap();
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&ready).unwrap()["type"],
                "ready"
            );
            let mut audio = vec![0u8; 32008];
            if invalid {
                audio[0] = 1;
            }
            socket.send(ClientMessage::Binary(audio)).await.unwrap();
            if !invalid {
                socket
                    .send(ClientMessage::Text(r#"{"type":"finish"}"#.into()))
                    .await
                    .unwrap();
            }
            loop {
                let message = tokio::time::timeout(Duration::from_secs(5), socket.next())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                let event: serde_json::Value =
                    serde_json::from_str(&message.into_text().unwrap()).unwrap();
                if invalid {
                    assert_eq!(event["type"], "error");
                    break;
                }
                if event["type"] == "final" {
                    assert_eq!(event["text"], "hello");
                    break;
                }
            }
            let _ = socket.close(None).await;
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        task.abort();
    }

    #[tokio::test]
    async fn websocket_streams_audio_into_one_padded_final() {
        use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};
        struct Counting {
            calls: Arc<std::sync::atomic::AtomicUsize>,
            sample_lengths: Arc<std::sync::Mutex<Vec<usize>>>,
        }
        impl Transcriber for Counting {
            fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> {
                Ok(None)
            }
            fn transcribe(
                &mut self,
                samples: &[f32],
                _: Option<&str>,
                _: &str,
                _: &AtomicBool,
            ) -> crate::Result<Transcript> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                self.sample_lengths.lock().unwrap().push(samples.len());
                Ok(Transcript {
                    text: "hello".into(),
                    language: None,
                })
            }
            fn finalization_policy(&self) -> FinalizationPolicy {
                FinalizationPolicy {
                    final_padding_samples: 8000,
                    ..FinalizationPolicy::default()
                }
            }
        }
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sample_lengths = Arc::new(std::sync::Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = AsrServer::new(
            "test".into(),
            Counting {
                calls: calls.clone(),
                sample_lengths: sample_lengths.clone(),
            },
        )
        .router(true);
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (mut socket, _) =
            connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
                .await
                .unwrap();
        socket.send(ClientMessage::Text(json!({"type":"start","version":1,"model":"test","sample_rate":16000,"format":"pcm_s16le"}).to_string())).await.unwrap();
        socket.next().await.unwrap().unwrap();
        for sequence in 0u64..2 {
            let mut audio = Vec::with_capacity(32008);
            audio.extend(sequence.to_le_bytes());
            audio.resize(32008, 0);
            socket.send(ClientMessage::Binary(audio)).await.unwrap();
            if sequence == 0 {
                loop {
                    let event: serde_json::Value = serde_json::from_str(
                        &socket.next().await.unwrap().unwrap().into_text().unwrap(),
                    )
                    .unwrap();
                    if event["type"] == "credit" {
                        break;
                    }
                }
            }
        }
        socket
            .send(ClientMessage::Text(r#"{"type":"finish"}"#.into()))
            .await
            .unwrap();
        loop {
            let event: serde_json::Value =
                serde_json::from_str(&socket.next().await.unwrap().unwrap().into_text().unwrap())
                    .unwrap();
            if event["type"] == "final" {
                break;
            }
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(*sample_lengths.lock().unwrap(), [40000]);
        task.abort();
    }

    #[tokio::test]
    async fn websocket_cancel_retains_session_until_worker_returns() {
        use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};
        struct Held {
            started: Arc<tokio::sync::Notify>,
            cancelled: Arc<tokio::sync::Notify>,
            release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
        }
        impl Transcriber for Held {
            fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> {
                Ok(None)
            }
            fn transcribe(
                &mut self,
                _: &[f32],
                _: Option<&str>,
                _: &str,
                cancel: &AtomicBool,
            ) -> crate::Result<Transcript> {
                self.started.notify_one();
                while !cancel.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                self.cancelled.notify_one();
                let (lock, condition) = &*self.release;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = condition.wait(released).unwrap();
                }
                Err(crate::RuntimeError::Rejected("cancelled".into()))
            }
        }
        let started = Arc::new(tokio::sync::Notify::new());
        let cancelled = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let server = AsrServer::new(
            "test".into(),
            Held {
                started: started.clone(),
                cancelled: cancelled.clone(),
                release: release.clone(),
            },
        );
        let session_capacity = server.sessions.available_permits();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = server.clone().router(true);
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (mut socket, _) =
            connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
                .await
                .unwrap();
        socket.send(ClientMessage::Text(json!({"type":"start","version":1,"model":"test","sample_rate":16000,"format":"pcm_s16le"}).to_string())).await.unwrap();
        socket.next().await.unwrap().unwrap();
        socket
            .send(ClientMessage::Binary(vec![0; 32008]))
            .await
            .unwrap();
        socket
            .send(ClientMessage::Text(r#"{"type":"finish"}"#.into()))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), started.notified())
            .await
            .unwrap();
        socket
            .send(ClientMessage::Text(r#"{"type":"cancel"}"#.into()))
            .await
            .unwrap();
        let observed = tokio::time::timeout(Duration::from_secs(5), cancelled.notified()).await;
        let held = server.sessions.available_permits() + 1 == session_capacity;
        *release.0.lock().unwrap() = true;
        release.1.notify_one();
        observed.unwrap();
        assert!(held, "cancellation cannot retire an active engine job");
        let permits = tokio::time::timeout(
            Duration::from_secs(5),
            server
                .sessions
                .clone()
                .acquire_many_owned(session_capacity as u32),
        )
        .await
        .unwrap()
        .unwrap();
        drop(permits);
        task.abort();
    }
}
