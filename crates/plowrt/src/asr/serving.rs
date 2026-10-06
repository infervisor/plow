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
use tokio::sync::{mpsc, oneshot, watch, OwnedSemaphorePermit, Semaphore};

use super::{
    endpoint::{EndpointConfig, Endpointer, Segment},
    frontend::{decode_wav, decode_wav_chunk, AudioError, Resampler, MAX_SAMPLES, SAMPLE_RATE},
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
    /// Set on shutdown: new work is refused (503) and sessions still receiving audio end.
    shutdown: watch::Sender<bool>,
    /// Admission-to-answer deadline of one transcription (`--asr-request-timeout-ms`).
    request_timeout: Option<Duration>,
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
    metrics: Arc<crate::obs::Metrics>,
}

struct AsrJob {
    metrics: Arc<crate::obs::Metrics>,
    queued_at: Instant,
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
    fn spawn(engine: Box<dyn Transcriber>) -> (Self, usize, FinalizationPolicy) {
        Self::spawn_with(engine, Arc::new(crate::obs::Metrics::default()))
    }

    fn spawn_with(mut engine: Box<dyn Transcriber>, metrics: Arc<crate::obs::Metrics>) -> (Self, usize, FinalizationPolicy) {
        metrics.serving.asr.cohort.store(true, Ordering::Relaxed);
        let _ = crate::obs::serving::started_at_unix_ms();
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
                    for job in &cohort {
                        let metrics = &job.metrics.serving.asr;
                        metrics.queued.fetch_sub(1, Ordering::Relaxed);
                        metrics.running.fetch_add(1, Ordering::Relaxed);
                        metrics.queue.duration(job.queued_at.elapsed());
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
                                finish_job(job, result);
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
        (Self { tx, metrics }, ingress_capacity, finalization)
    }

    fn submit(
        &self,
        samples: Vec<f32>,
        language: Option<String>,
        context: String,
        cancel: Arc<AtomicBool>,
    ) -> Result<oneshot::Receiver<crate::Result<Transcript>>, SubmitError> {
        let permit = self.tx.try_reserve().map_err(|error| {
            self.metrics.serving.asr.rejected.fetch_add(1, Ordering::Relaxed);
            match error {
                mpsc::error::TrySendError::Full(_) => SubmitError::Full,
                mpsc::error::TrySendError::Closed(_) => SubmitError::Closed,
            }
        })?;
        let (respond, receive) = oneshot::channel();
        let job = AsrJob {
            metrics: self.metrics.clone(),
            queued_at: Instant::now(),
            samples,
            language,
            context,
            cancel,
            respond,
        };
        self.metrics.serving.asr.jobs.fetch_add(1, Ordering::Relaxed);
        self.metrics.serving.asr.queued.fetch_add(1, Ordering::Relaxed);
        permit.send(job);
        Ok(receive)
    }
}

fn finish_job(job: AsrJob, result: crate::Result<Transcript>) {
    let metrics = &job.metrics.serving.asr;
    let elapsed = job.queued_at.elapsed();
    metrics.running.fetch_sub(1, Ordering::Relaxed);
    metrics.e2e.duration(elapsed);
    if job.cancel.load(Ordering::Relaxed) || job.respond.is_closed() {
        metrics.cancelled.fetch_add(1, Ordering::Relaxed);
    } else if result.is_ok() {
        metrics.completed.fetch_add(1, Ordering::Relaxed);
        metrics.first_transcript.duration(elapsed);
    } else {
        metrics.errors.fetch_add(1, Ordering::Relaxed);
    }
    let _ = job.respond.send(result);
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
        finish_job(job, Err(error));
    }
}

struct AsrSessionMetrics(Option<Arc<crate::obs::Metrics>>);
impl AsrSessionMetrics {
    fn new(metrics: Option<Arc<crate::obs::Metrics>>) -> Self {
        if let Some(m) = &metrics {
            m.serving.asr.websocket_sessions.fetch_add(1, Ordering::Relaxed);
            m.serving.asr.active_sessions.fetch_add(1, Ordering::Relaxed);
        }
        Self(metrics)
    }
}
impl Drop for AsrSessionMetrics {
    fn drop(&mut self) {
        if let Some(m) = &self.0 { m.serving.asr.active_sessions.fetch_sub(1, Ordering::Relaxed); }
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
            shutdown: watch::channel(false).0,
            request_timeout: configured_timeout(),
            recordings: Default::default(),
        })
    }

    /// Replace the configured transcription deadline; only before the server is shared.
    pub fn with_request_timeout(mut self: Arc<Self>, timeout: Option<Duration>) -> Arc<Self> {
        Arc::get_mut(&mut self).expect("AsrServer already shared").request_timeout = timeout;
        self
    }

    /// Stop admitting work: requests and new sessions get 503, sessions still receiving audio get
    /// a terminal error and close 1001, transcriptions already submitted run to their answer.
    pub fn begin_shutdown(&self) {
        self.shutdown.send_replace(true);
    }

    fn draining(&self) -> bool {
        *self.shutdown.borrow()
    }

    /// No queued or running job and no open session (the `plowrt asr` drain condition).
    pub fn idle(&self) -> bool {
        let Some(metrics) = self.metrics("") else { return true };
        let asr = &metrics.serving.asr;
        asr.queued.load(Ordering::Relaxed) == 0
            && asr.running.load(Ordering::Relaxed) == 0
            && asr.active_sessions.load(Ordering::Relaxed) == 0
    }

    /// Ready to take work: not shutting down and (cohort) the engine worker still running.
    fn ready(&self) -> bool {
        match &self.backend {
            Backend::Cohort { mux, .. } => !self.draining() && !mux.tx.is_closed(),
            #[cfg(feature = "cuda")]
            Backend::Serve(_) => !self.draining(),
        }
    }

    /// Transcription for every model `state` serves whose packet declares a causal audio pipeline:
    /// the prompt and encoder run here, the decoder on the model's continuous-batching mux.
    #[cfg(feature = "cuda")]
    pub fn for_serve(state: Arc<crate::serve::AppState>) -> Arc<Self> {
        Arc::new(Self {
            shutdown: state.shutdown.clone(),
            backend: Backend::Serve(state),
            uploads: Arc::new(Semaphore::new(shared::UPLOADS)),
            sessions: Arc::new(Semaphore::new(shared::UPLOADS)),
            request_timeout: configured_timeout(),
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

    /// The served model's bounded atomic counters.
    fn metrics(&self, _model: &str) -> Option<Arc<crate::obs::Metrics>> {
        match &self.backend {
            Backend::Cohort { mux, .. } => Some(mux.metrics.clone()),
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
            Backend::Serve(state) => {
                if let Some((mux, finalization)) = packet_models().read().get(model) {
                    return Ok((Route::Cohort(mux.clone()), *finalization));
                }
                shared::route(state, model).await
            }
        }
    }
    pub fn router(self: Arc<Self>, websocket: bool) -> Router {
        let model = match &self.backend {
            Backend::Cohort { model, .. } => Some(model.clone()),
            #[cfg(feature = "cuda")]
            Backend::Serve(_) => None,
        };
        let metrics = match &self.backend {
            Backend::Cohort { model, mux, .. } => Some((model.clone(), mux.metrics.clone())),
            #[cfg(feature = "cuda")]
            Backend::Serve(_) => None,
        };
        let health = Arc::clone(&self);
        let mut router = self.transcription_router(websocket).route(
            "/health",
            get(move || {
                let ready = health.ready();
                async move {
                    if ready { (StatusCode::OK, "ok") } else { (StatusCode::SERVICE_UNAVAILABLE, "unavailable") }
                }
            }),
        );
        if let Some((model, metrics)) = metrics {
            let models = vec![(model, metrics, true)];
            let json_models = models.clone();
            router = router.route("/metrics", get(move || {
                let mut out = String::new();
                crate::obs::serving::ServingMetrics::write(&mut out, &models);
                async move { ([("content-type", "text/plain; version=0.0.4; charset=utf-8")], out) }
            })).route("/v1/metrics", get(move || {
                let snapshot = crate::obs::serving::snapshot(&json_models);
                async move { Json(snapshot) }
            }));
        }
        if let Some(model) = model {
            let mut endpoints = vec!["audio/transcriptions"];
            if websocket { endpoints.push("audio/transcriptions/stream"); }
            let card = json!({"id": model, "root": model, "object": "model", "created": 0,
                "owned_by": "plow", "x_plow_endpoints": endpoints,
                "input_modalities": ["audio"], "output_modalities": ["text"]});
            let list_card = card.clone();
            router = router.route("/v1/models", get(move || {
                let card = list_card.clone();
                async move { Json(json!({"object": "list", "data": [card]})) }
            })).route("/v1/models/:model", get(move |axum::extract::Path(id): axum::extract::Path<String>| {
                let card = card.clone();
                async move {
                    if card["id"] == id { Json(card).into_response() }
                    else { failure(StatusCode::NOT_FOUND, "unknown ASR model") }
                }
            }));
        }
        router
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

/// Served models (on `plowrt serve`) whose ASR encoder thread or packet engine has exited.
#[cfg(feature = "cuda")]
pub fn dead_encoders() -> Vec<String> {
    let mut dead = shared::dead_encoders();
    dead.extend(packet_models().read().iter().filter(|(_, (mux, _))| mux.tx.is_closed()).map(|(n, _)| n.clone()));
    dead
}

/// Packet ASR models `plowrt serve` hosts on their own cohort engines (`--asr-packet`), by name.
#[cfg(feature = "cuda")]
fn packet_models() -> &'static parking_lot::RwLock<HashMap<String, (AsrMux, FinalizationPolicy)>> {
    static MODELS: std::sync::OnceLock<parking_lot::RwLock<HashMap<String, (AsrMux, FinalizationPolicy)>>> =
        std::sync::OnceLock::new();
    MODELS.get_or_init(Default::default)
}

/// Serve `engine` as `name` on `plowrt serve`, counting into that model's serve metrics.
#[cfg(feature = "cuda")]
pub fn host_packet_model(state: &crate::serve::AppState, name: String, engine: Box<dyn Transcriber>) {
    let (mux, _, finalization) = AsrMux::spawn_with(engine, state.model_metrics(&name));
    packet_models().write().insert(name, (mux, finalization));
}

/// Names of the packet ASR models `plowrt serve` hosts, sorted.
#[cfg(feature = "cuda")]
pub fn packet_model_names() -> Vec<String> {
    let mut names: Vec<_> = packet_models().read().keys().cloned().collect();
    names.sort();
    names
}

/// No packet-model transcription queued or running (the serve drain condition).
#[cfg(feature = "cuda")]
pub fn packet_models_idle() -> bool {
    packet_models().read().values().all(|(mux, _)| {
        let asr = &mux.metrics.serving.asr;
        asr.queued.load(Ordering::Relaxed) == 0 && asr.running.load(Ordering::Relaxed) == 0
    })
}

/// Whether `slug` serves `/v1/audio/transcriptions` on `plowrt serve`.
#[cfg(feature = "cuda")]
pub fn serves_audio(state: &crate::serve::AppState, slug: &str) -> bool {
    shared::serves_audio(state, slug)
}

fn configured_timeout() -> Option<Duration> {
    let ms = crate::config::RuntimeConfig::get().asr_request_timeout_ms;
    (ms > 0).then(|| Duration::from_millis(ms))
}

fn failure(status: StatusCode, message: impl ToString) -> Response {
    (
        status,
        Json(json!({"error":{"message":message.to_string(),"type":"transcription_error"}})),
    )
        .into_response()
}

/// 429 with `Retry-After`: a full queue, retryable.
fn busy(message: impl ToString) -> Response {
    let mut response = failure(StatusCode::TOO_MANY_REQUESTS, message);
    response.headers_mut().insert(axum::http::header::RETRY_AFTER, axum::http::HeaderValue::from_static("1"));
    response
}

const SHUTTING_DOWN: &str = "server shutting down";
const DEADLINE: &str = "transcription deadline exceeded";

fn runtime_failure(error: crate::RuntimeError) -> Response {
    let status = match &error {
        crate::RuntimeError::Overloaded(_) => return busy(error),
        crate::RuntimeError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
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
    if state.draining() {
        return failure(StatusCode::SERVICE_UNAVAILABLE, SHUTTING_DOWN);
    }
    let upload = match state.uploads.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return busy("too many ASR uploads"),
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
    let metrics = state.metrics(&model);
    if let Some(metrics) = &metrics {
        metrics.serving.asr.http_requests.fetch_add(1, Ordering::Relaxed);
    }
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
        metrics,
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
        Err(SubmitError::Full) => return busy("ASR queue full"),
        Err(SubmitError::Closed) => {
            return failure(StatusCode::SERVICE_UNAVAILABLE, "ASR engine unavailable")
        }
    };
    let deadline = state.request_timeout.map(|t| tokio::time::Instant::now() + t);
    let offset = recorded.then_some(recorded_samples);
    if stream {
        let cache = crate::serve::session::CacheOutcome::received(report_rx).await;
        run.admitted(cache.and_then(|c| c.at));
        let stamped = run.headers();
        let mut response = sse_transcript(work, delta_rx, ids.clone(), finals, offset, cache, cancel.0.clone(), deadline, (in_flight, cancel, recording), run);
        response.headers_mut().extend(stamped);
        if let Some(cache) = cache {
            cache.stamp(&mut response);
        }
        return response;
    }
    let started = Instant::now();
    let result = match deadline {
        Some(deadline) => match tokio::time::timeout_at(deadline, work).await {
            Ok(result) => result,
            Err(_) => {
                cancel.0.store(true, Ordering::Relaxed);
                return failure(StatusCode::GATEWAY_TIMEOUT, DEADLINE);
            }
        },
        None => work.await,
    };
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

/// `deadline`, or never.
async fn sleep_until(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
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
    cancel: Arc<AtomicBool>,
    deadline: Option<tokio::time::Instant>,
    held: H,
    mut run: crate::serve::turns::StageRun,
) -> Response {
    let (tx, mut rx) = mpsc::channel::<axum::response::sse::Event>(64);
    tokio::spawn(async move {
        let _held = held;
        let mut shown = String::new();
        let result = loop {
            tokio::select! {
                _ = tx.closed() => {
                    cancel.store(true, Ordering::Relaxed);
                    let _ = work.await;
                    return;
                },
                Some(delta) = deltas.recv() => {
                    shown.push_str(&delta);
                    if tx.send(transcript_event("transcript.text.delta", &ids, json!({"delta": delta}))).await.is_err() {
                        cancel.store(true, Ordering::Relaxed);
                        let _ = work.await;
                        return;
                    }
                }
                result = &mut work => break result,
                _ = sleep_until(deadline) => {
                    cancel.store(true, Ordering::Relaxed);
                    let _ = tx.send(transcript_event("error", &ids, json!({"message": DEADLINE, "code": "timeout"}))).await;
                    let _ = work.await;
                    return;
                }
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
    if state.draining() {
        return failure(StatusCode::SERVICE_UNAVAILABLE, SHUTTING_DOWN);
    }
    if let Some(r) = crate::serve::overload::gate(&ids) {
        return r;
    }
    let permit = match state.sessions.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return busy("too many ASR sessions"),
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
    /// The final transcript's text as it decodes (`"type":"delta"` events before `final`).
    #[serde(default)]
    deltas: bool,
    /// `utterance` (default): one transcript per connection. `continuous`: unbounded audio,
    /// endpointed into segments with one `final` each.
    #[serde(default)]
    mode: Option<String>,
    /// Continuous mode: silence that ends a segment (200..=2000 ms, default 600).
    min_silence_ms: Option<u32>,
    /// Continuous mode: a longer segment is cut at its quietest recent frame (default 25000 ms).
    max_segment_ms: Option<u32>,
}

/// Client PCM rates the stream accepts; audio is resampled to 16 kHz as it arrives.
const STREAM_RATES: [u32; 7] = [8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000];
/// Server pings a stream this often, and ends one whose peer has not answered for `PONG_TIMEOUT`.
const PING_INTERVAL: Duration = Duration::from_secs(15);
const PONG_TIMEOUT: Duration = Duration::from_secs(45);
/// A stream with no audio or control message for this long ends.
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);

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

/// A terminal error, then close 1001 (going away).
async fn going_away(socket: &mut WebSocket, message: &str) {
    send(socket, json!({"type":"error","message":message,"terminal":true})).await;
    let close = Message::Close(Some(axum::extract::ws::CloseFrame { code: 1001, reason: message.to_owned().into() }));
    let _ = tokio::time::timeout(Duration::from_secs(5), socket.send(close)).await;
}

async fn stream(state: Arc<AsrServer>, mut socket: WebSocket, _permit: OwnedSemaphorePermit, ids: RequestIds) {
    let cancel = Cancellation(Arc::new(AtomicBool::new(false)));
    let mut shutdown = state.shutdown.subscribe();
    let Some(Ok(Message::Text(text))) =
        tokio::time::timeout(IDLE_TIMEOUT, socket.recv())
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
                && STREAM_RATES.contains(&s.sample_rate)
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
    if state.draining() {
        going_away(&mut socket, SHUTTING_DOWN).await;
        return;
    }
    match start.mode.as_deref() {
        None | Some("utterance") => {}
        Some("continuous") => return continuous(state, socket, ids, start, route, finalization, cancel).await,
        Some(_) => {
            send(&mut socket, json!({"type":"error","message":"invalid start","terminal":true})).await;
            return;
        }
    }
    let _metrics = AsrSessionMetrics::new(state.metrics(&start.model));
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
    // Credit and limits count the client's samples; `samples` holds them resampled to 16 kHz.
    let rate = start.sample_rate as usize;
    let mut resampler = Resampler::new(start.sample_rate).expect("a listed stream rate");
    let max_audio_samples = MAX_SAMPLES.saturating_sub(finalization.final_padding_samples) * rate / SAMPLE_RATE as usize;
    let initial_credit = rate.min(max_audio_samples);
    if !send(&mut socket,json!({"type":"ready","version":1,"session_id":&*session,"request_id":&*ids.request,
        "sample_rate":start.sample_rate,"format":"pcm_s16le","max_chunk_bytes":32000,"credit_samples":initial_credit,
        "max_audio_samples":max_audio_samples,"partial_mode":if partials {"revision"} else {"final_only"},
        "deltas":start.deltas,"mode":"utterance"})).await{return;}
    let mut samples = Vec::new();
    let mut received = 0usize;
    let mut sequence = 0u64;
    let mut credit = initial_credit;
    let (mut revision, mut last_partial, mut partial_at) = (0u64, String::new(), 0usize);
    let mut pending: Option<oneshot::Receiver<crate::Result<Transcript>>> = None;
    let mut ping = tokio::time::interval_at(tokio::time::Instant::now() + PING_INTERVAL, PING_INTERVAL);
    let mut last_pong = tokio::time::Instant::now();
    let mut idle_at = tokio::time::Instant::now() + IDLE_TIMEOUT;
    loop {
        let message = tokio::select! {
            m = tokio::time::timeout_at(idle_at, socket.recv()) => match m {
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
            _ = ping.tick() => {
                if last_pong.elapsed() >= PONG_TIMEOUT {
                    going_away(&mut socket, "ping timeout").await;
                    return;
                }
                if socket.send(Message::Ping(Vec::new())).await.is_err() {
                    return;
                }
                continue;
            }
            Ok(()) = shutdown.changed() => {
                if *shutdown.borrow() {
                    going_away(&mut socket, SHUTTING_DOWN).await;
                    return;
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
                    || received + (bytes.len() - 8) / 2 > max_audio_samples
                {
                    send(&mut socket,json!({"type":"error","message":"invalid PCM sequence or credit/length exceeded","terminal":true})).await;
                    return;
                }
                idle_at = tokio::time::Instant::now() + IDLE_TIMEOUT;
                sequence += 1;
                credit -= (bytes.len() - 8) / 2;
                received += (bytes.len() - 8) / 2;
                let pcm: Vec<f32> = bytes[8..]
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();
                samples.extend(resampler.push(&pcm));
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
            Message::Pong(_) => {
                last_pong = tokio::time::Instant::now();
                continue;
            }
            Message::Ping(_) => continue,
        };
        if finish {
            samples.extend(resampler.finish());
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
            let (delta_tx, mut delta_rx) = mpsc::unbounded_channel();
            let mut opts = request(true, run.key());
            if start.deltas {
                opts.deltas = Some(delta_tx);
            }
            let mut work = match route.submit(samples, language, start.prompt, cancel.0.clone(), opts) {
                Ok(work) => work,
                Err(SubmitError::Full) => {
                    send(
                        &mut socket,
                        json!({"type":"error","message":"ASR queue full","code":"overloaded","terminal":true}),
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
            // A final already submitted runs to its answer through shutdown (the drain bounds it).
            let deadline = state.request_timeout.map(|t| tokio::time::Instant::now() + t);
            let mut shown = String::new();
            let result = loop {
                tokio::select! {
                    result=&mut work=>break result,
                    Some(delta) = delta_rx.recv(), if start.deltas => {
                        shown.push_str(&delta);
                        if !send(&mut socket, json!({"type":"delta","text":delta})).await {
                            cancel.0.store(true,Ordering::Relaxed);
                            let _=work.await;
                            return;
                        }
                    }
                    _ = sleep_until(deadline) => {
                        cancel.0.store(true,Ordering::Relaxed);
                        let _=work.await;
                        send(&mut socket,json!({"type":"error","message":DEADLINE,"code":"timeout","terminal":true})).await;
                        return;
                    }
                    _ = ping.tick() => {
                        if last_pong.elapsed() >= PONG_TIMEOUT || socket.send(Message::Ping(Vec::new())).await.is_err() {
                            cancel.0.store(true,Ordering::Relaxed);
                            let _=work.await;
                            return;
                        }
                    }
                    incoming=socket.recv()=>match incoming {
                        Some(Ok(Message::Pong(_)))=>last_pong=tokio::time::Instant::now(),
                        Some(Ok(Message::Ping(_)))=>{},
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
                    if start.deltas {
                        // A route that decodes without deltas (the cohort engine) sends the whole
                        // text as one delta here.
                        while let Ok(delta) = delta_rx.try_recv() {
                            shown.push_str(&delta);
                            send(&mut socket, json!({"type":"delta","text":delta})).await;
                        }
                        if let Some(rest) = result.text.strip_prefix(shown.as_str()).filter(|r| !r.is_empty()) {
                            send(&mut socket, json!({"type":"delta","text":rest})).await;
                        }
                    }
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
                    let code = match &error {
                        crate::RuntimeError::Overloaded(_) => Some("overloaded"),
                        crate::RuntimeError::Unavailable(_) => Some("unavailable"),
                        _ => None,
                    };
                    send(
                        &mut socket,
                        json!({"type":"error","message":error.to_string(),"code":code,"terminal":true}),
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
        let grant = (rate - received % rate)
            .min(max_audio_samples - received)
            .saturating_sub(credit);
        credit += grant;
        if grant > 0 && !send(&mut socket, json!({"type":"credit","credit_samples":grant})).await {
            return;
        }
    }
}

/// Segments a continuous session may have awaiting results; past it credit is withheld.
const MAX_SEGMENTS_IN_FLIGHT: usize = 2;

/// One continuous-session segment submitted for its final.
struct Flight {
    segment: u64,
    start_ms: u64,
    end_ms: u64,
    work: oneshot::Receiver<crate::Result<Transcript>>,
    deltas: mpsc::UnboundedReceiver<String>,
    shown: String,
    deadline: Option<tokio::time::Instant>,
    run: crate::serve::turns::StageRun,
    cancel: Cancellation,
}

enum FrontEvent {
    Delta(String),
    Done(Result<crate::Result<Transcript>, oneshot::error::RecvError>),
    Timeout,
}

/// The oldest segment's next delta, its answer, or its deadline: finals leave in segment order.
async fn front_event(flights: &mut VecDeque<Flight>) -> FrontEvent {
    let f = flights.front_mut().expect("guarded");
    tokio::select! {
        biased;
        Some(delta) = f.deltas.recv() => FrontEvent::Delta(delta),
        result = &mut f.work => FrontEvent::Done(result),
        _ = sleep_until(f.deadline) => FrontEvent::Timeout,
    }
}

/// `mode: continuous`: unbounded audio, endpointed into segments that are transcribed while
/// audio keeps arriving; one `final` per segment, in order, then `done` after `finish`.
async fn continuous(
    state: Arc<AsrServer>,
    mut socket: WebSocket,
    ids: RequestIds,
    start: Start,
    route: Route,
    finalization: FinalizationPolicy,
    _cancel: Cancellation,
) {
    let cap_ms = (MAX_SAMPLES.saturating_sub(finalization.final_padding_samples) / 16) as u32;
    let min_silence_ms = start.min_silence_ms.unwrap_or(600);
    let max_segment_ms = start.max_segment_ms.unwrap_or(cap_ms.min(25_000));
    if !(200..=2000).contains(&min_silence_ms) || !(4_000..=cap_ms).contains(&max_segment_ms) {
        send(&mut socket, json!({"type":"error","message":"invalid start","terminal":true})).await;
        return;
    }
    let _metrics = AsrSessionMetrics::new(state.metrics(&start.model));
    let mut shutdown = state.shutdown.subscribe();
    let session = ids.session.clone().unwrap_or_default();
    let rate = start.sample_rate as usize;
    let mut resampler = Resampler::new(start.sample_rate).expect("a listed stream rate");
    let mut endpointer = Endpointer::new(EndpointConfig { min_silence_ms, max_segment_ms });
    if !send(&mut socket, json!({"type":"ready","version":1,"session_id":&*session,"request_id":&*ids.request,
        "sample_rate":start.sample_rate,"format":"pcm_s16le","max_chunk_bytes":32000,"credit_samples":rate,
        "max_audio_samples":null,"partial_mode":if start.partials {"revision"} else {"final_only"},
        "deltas":start.deltas,"mode":"continuous","min_silence_ms":min_silence_ms,
        "max_segment_ms":max_segment_ms})).await {
        return;
    }
    let opts = |final_pass, turn_key, windows, deltas| AsrOpts {
        final_pass,
        ids: Some(RequestIds { turn_key, ..ids.with_new_request() }),
        windows,
        deltas,
        report: None,
    };
    let launch = |segment: &Segment| -> Result<Flight, SubmitError> {
        let mut samples = segment.samples.clone();
        if samples.len() < SAMPLE_RATE as usize / 2 {
            samples.resize(SAMPLE_RATE as usize / 2, 0.0);
        }
        append_final_padding(&mut samples, finalization.final_padding_samples, finalization.final_padding_amplitude);
        let run = crate::serve::turns::StageRun::start(
            &ids,
            crate::serve::turns::Kind::Asr,
            &start.model,
            state.metrics(&start.model),
            Instant::now(),
            true,
        );
        let (tx, deltas) = mpsc::unbounded_channel();
        let cancel = Cancellation(Arc::new(AtomicBool::new(false)));
        let work = route.submit(
            samples,
            start.language.clone(),
            start.prompt.clone(),
            cancel.0.clone(),
            opts(true, run.key(), None, start.deltas.then_some(tx)),
        )?;
        Ok(Flight {
            segment: segment.index,
            start_ms: segment.start / 16,
            end_ms: segment.end / 16,
            work,
            deltas,
            shown: String::new(),
            deadline: state.request_timeout.map(|t| tokio::time::Instant::now() + t),
            run,
            cancel,
        })
    };
    let mut credit = rate;
    let mut sequence = 0u64;
    let mut waiting: VecDeque<Segment> = VecDeque::new();
    let mut flights: VecDeque<Flight> = VecDeque::new();
    let mut emitted = 0u64;
    let mut finishing = false;
    let mut partial: Option<(u64, Cancellation, oneshot::Receiver<crate::Result<Transcript>>)> = None;
    let (mut partial_segment, mut partial_at, mut revision) = (u64::MAX, 0usize, 0u64);
    let mut last_partial = String::new();
    let mut windows: Arc<parking_lot::Mutex<WindowCache>> = Default::default();
    let mut ping = tokio::time::interval_at(tokio::time::Instant::now() + PING_INTERVAL, PING_INTERVAL);
    let mut last_pong = tokio::time::Instant::now();
    let mut idle_at = tokio::time::Instant::now() + IDLE_TIMEOUT;
    loop {
        while flights.len() < MAX_SEGMENTS_IN_FLIGHT {
            let Some(segment) = waiting.front() else { break };
            match launch(segment) {
                Ok(flight) => {
                    waiting.pop_front();
                    flights.push_back(flight);
                }
                Err(SubmitError::Full) if !flights.is_empty() => break,
                Err(SubmitError::Full) => {
                    send(&mut socket, json!({"type":"error","message":"ASR queue full","code":"overloaded","terminal":true})).await;
                    return;
                }
                Err(SubmitError::Closed) => {
                    send(&mut socket, json!({"type":"error","message":"ASR engine unavailable","code":"unavailable","terminal":true})).await;
                    return;
                }
            }
        }
        if finishing && flights.is_empty() && waiting.is_empty() {
            send(&mut socket, json!({"type":"done","segments":emitted})).await;
            let close = Message::Close(Some(axum::extract::ws::CloseFrame { code: 1000, reason: "done".into() }));
            let _ = tokio::time::timeout(Duration::from_secs(5), socket.send(close)).await;
            return;
        }
        let backpressured = flights.len() + waiting.len() >= MAX_SEGMENTS_IN_FLIGHT;
        if !finishing && !backpressured && credit < rate {
            let grant = rate - credit;
            credit = rate;
            if !send(&mut socket, json!({"type":"credit","credit_samples":grant})).await {
                return;
            }
        }
        if finishing || backpressured {
            // The client may not send: no idle deadline while the server holds it up.
            idle_at = tokio::time::Instant::now() + IDLE_TIMEOUT;
        }
        let message = tokio::select! {
            m = tokio::time::timeout_at(idle_at, socket.recv()) => match m {
                Ok(Some(Ok(m))) => m,
                _ => return,
            },
            event = front_event(&mut flights), if !flights.is_empty() => {
                match event {
                    FrontEvent::Delta(delta) => {
                        let f = flights.front_mut().expect("guarded");
                        f.shown.push_str(&delta);
                        if !send(&mut socket, json!({"type":"delta","segment":f.segment,"text":delta})).await {
                            return;
                        }
                    }
                    FrontEvent::Timeout => {
                        let mut f = flights.pop_front().expect("guarded");
                        f.cancel.0.store(true, Ordering::Relaxed);
                        let _ = (&mut f.work).await;
                        emitted += 1;
                        if !send(&mut socket, json!({"type":"error","segment":f.segment,"message":DEADLINE,
                            "code":"timeout","terminal":false})).await {
                            return;
                        }
                    }
                    FrontEvent::Done(result) => {
                        let mut f = flights.pop_front().expect("guarded");
                        emitted += 1;
                        if partial.as_ref().is_some_and(|p| p.0 == f.segment) {
                            partial = None;
                        }
                        let sent = match result {
                            Ok(Ok(result)) => {
                                if start.deltas {
                                    // A route that decodes without deltas (the cohort engine)
                                    // sends the whole text as one delta.
                                    while let Ok(delta) = f.deltas.try_recv() {
                                        f.shown.push_str(&delta);
                                        send(&mut socket, json!({"type":"delta","segment":f.segment,"text":delta})).await;
                                    }
                                    if let Some(rest) = result.text.strip_prefix(f.shown.as_str()).filter(|r| !r.is_empty()) {
                                        send(&mut socket, json!({"type":"delta","segment":f.segment,"text":rest})).await;
                                    }
                                }
                                f.run.first();
                                f.run.done();
                                send(&mut socket, json!({"type":"final","segment":f.segment,"start_ms":f.start_ms,
                                    "end_ms":f.end_ms,"text":result.text,"language":result.language,
                                    "stable_prefix_bytes":result.text.len(),"turn_id":f.run.turn_id.as_deref(),
                                    "traceparent":f.run.traceparent(),"server_timing":f.run.timing().header()})).await
                            }
                            Ok(Err(error)) => {
                                let code = match &error {
                                    crate::RuntimeError::Overloaded(_) => Some("overloaded"),
                                    crate::RuntimeError::Unavailable(_) => Some("unavailable"),
                                    _ => None,
                                };
                                send(&mut socket, json!({"type":"error","segment":f.segment,"message":error.to_string(),
                                    "code":code,"terminal":false})).await
                            }
                            Err(_) => {
                                send(&mut socket, json!({"type":"error","segment":f.segment,
                                    "message":"ASR engine response channel closed","terminal":false})).await
                            }
                        };
                        if !sent {
                            return;
                        }
                    }
                }
                continue;
            }
            Ok(result) = async { (&mut partial.as_mut().expect("guarded").2).await }, if partial.is_some() => {
                let (segment, _, _) = partial.take().expect("guarded");
                if let Ok(result) = result {
                    if segment == partial_segment && segment >= emitted {
                        revision += 1;
                        let stable = common_prefix_bytes(&last_partial, &result.text);
                        if !send(&mut socket, json!({"type":"partial","segment":segment,"revision":revision,
                            "text":result.text,"language":result.language,"stable_prefix_bytes":stable})).await {
                            return;
                        }
                        last_partial = result.text;
                    }
                }
                continue;
            }
            _ = ping.tick() => {
                if last_pong.elapsed() >= PONG_TIMEOUT {
                    going_away(&mut socket, "ping timeout").await;
                    return;
                }
                if socket.send(Message::Ping(Vec::new())).await.is_err() {
                    return;
                }
                continue;
            }
            Ok(()) = shutdown.changed() => {
                if *shutdown.borrow() && !finishing {
                    going_away(&mut socket, SHUTTING_DOWN).await;
                    return;
                }
                continue;
            }
        };
        match message {
            Message::Binary(bytes) if !finishing => {
                if bytes.len() < 10
                    || bytes.len() > 32008
                    || (bytes.len() - 8) % 2 != 0
                    || u64::from_le_bytes(bytes[..8].try_into().unwrap()) != sequence
                    || (bytes.len() - 8) / 2 > credit
                {
                    send(&mut socket, json!({"type":"error","message":"invalid PCM sequence or credit exceeded","terminal":true})).await;
                    return;
                }
                idle_at = tokio::time::Instant::now() + IDLE_TIMEOUT;
                sequence += 1;
                credit -= (bytes.len() - 8) / 2;
                let pcm: Vec<f32> = bytes[8..]
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();
                waiting.extend(endpointer.push(&resampler.push(&pcm)));
                if start.partials && partial.is_none() {
                    if let Some((segment, audio)) = endpointer.open_audio() {
                        if segment != partial_segment {
                            (partial_segment, partial_at, revision) = (segment, 0, 0);
                            last_partial.clear();
                            windows = Default::default();
                        }
                        if audio.len() >= (partial_at + PARTIAL_STRIDE).max(SAMPLE_RATE as usize / 2) {
                            partial_at = audio.len();
                            let cancel = Cancellation(Arc::new(AtomicBool::new(false)));
                            // A full queue skips this partial; the next stride retries.
                            if let Ok(work) = route.submit(audio.to_vec(), start.language.clone(), start.prompt.clone(),
                                cancel.0.clone(), opts(false, None, Some(windows.clone()), None)) {
                                partial = Some((segment, cancel, work));
                            }
                        }
                    }
                }
            }
            Message::Text(text) if !finishing => match serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v["type"].as_str().map(str::to_owned))
                .as_deref()
            {
                Some("finish") => {
                    finishing = true;
                    partial = None;
                    let tail = resampler.finish();
                    waiting.extend(endpointer.push(&tail));
                    waiting.extend(endpointer.finish());
                }
                Some("cancel") => return,
                _ => {
                    send(&mut socket, json!({"type":"error","message":"expected finish or cancel","terminal":true})).await;
                    return;
                }
            },
            Message::Text(text) if serde_json::from_str::<serde_json::Value>(&text).ok().is_some_and(|v| v["type"] == "cancel") => return,
            Message::Pong(_) => last_pong = tokio::time::Instant::now(),
            Message::Ping(_) => {}
            Message::Close(_) => return,
            _ => {
                send(&mut socket, json!({"type":"error","message":"input sent after finish or invalid event","terminal":true})).await;
                return;
            }
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
        let stats = mux.metrics.serving.asr.snapshot().unwrap();
        assert_eq!((stats.jobs, stats.completed, stats.cancelled, stats.errors), (4, 3, 1, 0));
        assert_eq!((stats.queued, stats.running), (0, 0));
        assert_eq!((stats.e2e.count, stats.queue.count, stats.first_transcript.count), (4, 4, 3));
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
        let metrics = mux.metrics.serving.asr.snapshot().unwrap();
        assert_eq!((metrics.running, metrics.queued, metrics.rejected), (1, ingress_capacity as u64, 1));
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
    async fn standalone_discovery_preserves_model_identity_and_audio_capabilities() {
        let app = AsrServer::new("nemotron-asr-0.6b".into(), Fake).router(true);
        let response = app.clone().oneshot(Request::get("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let list: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(list["data"][0]["id"], "nemotron-asr-0.6b");
        assert_eq!(list["data"][0]["x_plow_endpoints"], json!(["audio/transcriptions", "audio/transcriptions/stream"]));
        for (name, status) in [("nemotron-asr-0.6b", 200), ("transcribe", 404)] {
            assert_eq!(app.clone().oneshot(Request::get(format!("/v1/models/{name}")).body(Body::empty()).unwrap()).await.unwrap().status(), status);
            assert_eq!(app.clone().oneshot(request(name, "json")).await.unwrap().status(), status);
        }
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
            (7000, 7000, StatusCode::UNSUPPORTED_MEDIA_TYPE),
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
    async fn dropped_http_audio_stream_cancels_engine_work_and_balances_metrics() {
        struct UntilCancelled(Arc<tokio::sync::Notify>, Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>);
        impl Transcriber for UntilCancelled {
            fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> { Ok(None) }
            fn transcribe(&mut self, _: &[f32], _: Option<&str>, _: &str, cancel: &AtomicBool) -> crate::Result<Transcript> {
                while !cancel.load(Ordering::Relaxed) { std::thread::sleep(Duration::from_millis(1)); }
                self.0.notify_one();
                let mut released = self.1.0.lock().unwrap();
                while !*released { released = self.1.1.wait(released).unwrap(); }
                Err(crate::RuntimeError::Rejected("cancelled".into()))
            }
        }
        let cancelled = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let server = AsrServer::new("test".into(), UntilCancelled(cancelled.clone(), release.clone()));
        let (mut parts, body) = request("test", "json").into_parts();
        parts.headers.insert("x-session-id", "http-cancellation-session".parse().unwrap());
        parts.headers.insert("x-request-id", "http-cancellation-request".parse().unwrap());
        let ids = RequestIds::from_headers(&parts.headers).unwrap();
        let mut body = body.collect().await.unwrap().to_bytes().to_vec();
        body.truncate(body.len() - b"--audio--\r\n".len());
        body.extend_from_slice(b"--audio\r\nContent-Disposition: form-data; name=\"stream\"\r\n\r\ntrue\r\n--audio--\r\n");
        let response = server.clone().router(true).oneshot(Request::from_parts(parts, Body::from(body))).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        drop(response);
        tokio::time::timeout(Duration::from_secs(1), cancelled.notified()).await.unwrap();
        let metrics = server.metrics("test").unwrap();
        let retained = ids.begin("test").is_none();
        let running = metrics.serving.asr.running.load(Ordering::Relaxed);
        *release.0.lock().unwrap() = true;
        release.1.notify_one();
        assert!(retained, "request identity remains in flight until the cancelled worker returns");
        assert_eq!(running, 1);
        tokio::time::timeout(Duration::from_secs(1), async {
            while metrics.serving.asr.cancelled.load(Ordering::Relaxed) == 0 { tokio::task::yield_now().await; }
        }).await.unwrap();
        let stats = metrics.serving.asr.snapshot().unwrap();
        assert_eq!((stats.jobs, stats.cancelled, stats.running, stats.queued, stats.errors), (1, 1, 0, 0, 0));
    }

    #[tokio::test]
    async fn standalone_metrics_count_work_and_publish_the_served_identity() {
        let server = AsrServer::new("named-asr".into(), Fake);
        let app = server.clone().router(true);
        let ok = app.clone().oneshot(request("named-asr", "json")).await.unwrap();
        assert_eq!(ok.status(), StatusCode::OK);
        let bad = app.clone().oneshot(request("named-asr", "xml")).await.unwrap();
        assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
        let Backend::Cohort { mux, .. } = &server.backend else { panic!("cohort") };
        let error = mux.submit(vec![0.0; 8000], Some("Invalid".into()), String::new(), Arc::new(AtomicBool::new(false))).unwrap().await.unwrap();
        assert!(error.is_err());
        let response = app.clone().oneshot(Request::get("/v1/metrics").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let snapshot: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(snapshot["object"], "runtime.metrics");
        assert!(snapshot["started_at_unix_ms"].as_u64().unwrap() <= snapshot["observed_at_unix_ms"].as_u64().unwrap());
        let model = &snapshot["models"][0];
        assert_eq!(model["id"], "named-asr");
        assert_eq!(model["requests"], 2);
        assert_eq!(model["completed"], 1);
        assert_eq!(model["asr"]["errors"], 1);
        assert_eq!(model["asr"]["http_requests"], 2);
        assert_eq!(model["asr"]["first_transcript"]["count"], 1);
        assert_eq!(model["asr"]["e2e"]["count"], 2);
        assert!(model["ttft"]["p95_ms"].is_null(), "audio transcripts must not invent token latency");
        let response = app.oneshot(Request::get("/metrics").body(Body::empty()).unwrap()).await.unwrap();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("plowrt_asr_jobs_errors_total{model_name=\"named-asr\",engine=\"0\"} 1"));
        assert!(text.contains("plowrt_asr_jobs_running{model_name=\"named-asr\",engine=\"0\"} 0"));
    }

    #[tokio::test]
    async fn http_discovery_and_transcription_reuse_one_tcp_socket() {
        use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = AsrServer::new("test".into(), Fake).router(true);
        let task = tokio::spawn(async move { axum::serve(listener, app).tcp_nodelay(true).await.unwrap() });
        let mut socket = BufReader::new(tokio::net::TcpStream::connect(address).await.unwrap());
        let request = request("test", "json");
        let (parts, body) = request.into_parts();
        let body = body.collect().await.unwrap().to_bytes();
        for transcription in [false, true, false] {
            let head = if transcription {
                format!("POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {address}\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n", parts.headers["content-type"].to_str().unwrap(), body.len())
            } else {
                format!("GET /v1/models HTTP/1.1\r\nHost: {address}\r\n\r\n")
            };
            socket.get_mut().write_all(head.as_bytes()).await.unwrap();
            if transcription { socket.get_mut().write_all(&body).await.unwrap(); }
            let mut line = String::new();
            socket.read_line(&mut line).await.unwrap();
            assert!(line.starts_with("HTTP/1.1 200"), "{line}");
            let mut length = None;
            loop {
                line.clear();
                assert!(socket.read_line(&mut line).await.unwrap() > 0);
                if line == "\r\n" { break; }
                if let Some((name, value)) = line.split_once(':') {
                    if name.eq_ignore_ascii_case("content-length") { length = Some(value.trim().parse::<usize>().unwrap()); }
                }
            }
            let mut response = vec![0; length.expect("bounded JSON response")];
            socket.read_exact(&mut response).await.unwrap();
            let response: serde_json::Value = serde_json::from_slice(&response).unwrap();
            if transcription { assert_eq!(response["text"], "hello"); }
            else { assert_eq!(response["data"][0]["id"], "test"); }
        }
        task.abort();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_socket_supports_audio_websocket_upgrade() {
        let (client, server) = tokio::net::UnixStream::pair().unwrap();
        let app = AsrServer::new("test".into(), Fake).router(true);
        let task = tokio::spawn(async move {
            hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                .serve_connection_with_upgrades(hyper_util::rt::TokioIo::new(server), hyper_util::service::TowerToHyperService::new(app))
                .await.unwrap();
        });
        let (mut socket, _) = tokio_tungstenite::client_async("ws://localhost/v1/audio/transcriptions/stream", client).await.unwrap();
        socket.send(tokio_tungstenite::tungstenite::Message::Text(json!({"type":"start","version":1,"model":"test","sample_rate":16000,"format":"pcm_s16le"}).to_string())).await.unwrap();
        let ready = socket.next().await.unwrap().unwrap().into_text().unwrap();
        assert_eq!(serde_json::from_str::<serde_json::Value>(&ready).unwrap()["type"], "ready");
        socket.close(None).await.unwrap();
        task.await.unwrap();
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
        held_websocket_cancellation(false).await;
    }

    #[tokio::test]
    async fn websocket_disconnect_retains_session_until_worker_returns() {
        held_websocket_cancellation(true).await;
    }

    async fn held_websocket_cancellation(disconnect: bool) {
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
        socket.send(ClientMessage::Ping(b"alive".to_vec())).await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                match socket.next().await.unwrap().unwrap() {
                    ClientMessage::Pong(payload) => { assert_eq!(payload, b"alive"); break; }
                    ClientMessage::Text(_) => {}, // Buffered credit event.
                    message => panic!("unexpected heartbeat reply: {message:?}"),
                }
            }
        }).await.unwrap();
        if disconnect {
            drop(socket);
        } else {
            socket.send(ClientMessage::Text(r#"{"type":"cancel"}"#.into())).await.unwrap();
        }
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
        let metrics = server.metrics("test").unwrap();
        let stats = metrics.serving.asr.snapshot().unwrap();
        assert_eq!((stats.websocket_sessions, stats.active_sessions, stats.cancelled, stats.running), (1, 0, 1, 0));
        task.abort();
    }

    #[tokio::test]
    async fn errors_map_to_retryable_statuses() {
        let busy = runtime_failure(crate::RuntimeError::Overloaded("ASR queue full".into()));
        assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(busy.headers()[axum::http::header::RETRY_AFTER], "1");
        let gone = runtime_failure(crate::RuntimeError::Unavailable("draining".into()));
        assert_eq!(gone.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(runtime_failure(crate::RuntimeError::Rejected("bad".into())).status(), StatusCode::BAD_REQUEST);
    }

    struct UntilCancelled(Arc<tokio::sync::Notify>);
    impl Transcriber for UntilCancelled {
        fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> {
            Ok(None)
        }
        fn transcribe(&mut self, _: &[f32], _: Option<&str>, _: &str, cancel: &AtomicBool) -> crate::Result<Transcript> {
            while !cancel.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(1));
            }
            self.0.notify_one();
            Err(crate::RuntimeError::Rejected("cancelled".into()))
        }
    }

    #[tokio::test]
    async fn deadline_answers_504_and_cancels_the_work() {
        let cancelled = Arc::new(tokio::sync::Notify::new());
        let server = AsrServer::new("test".into(), UntilCancelled(cancelled.clone()))
            .with_request_timeout(Some(Duration::from_millis(50)));
        let response = server.clone().router(false).oneshot(request("test", "json")).await.unwrap();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        tokio::time::timeout(Duration::from_secs(5), cancelled.notified()).await.unwrap();
    }

    #[tokio::test]
    async fn full_queue_is_429_with_retry_after() {
        struct Blocks(Arc<tokio::sync::Notify>);
        impl Transcriber for Blocks {
            fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> {
                Ok(None)
            }
            fn transcribe(&mut self, _: &[f32], _: Option<&str>, _: &str, cancel: &AtomicBool) -> crate::Result<Transcript> {
                self.0.notify_one();
                while !cancel.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(crate::RuntimeError::Rejected("cancelled".into()))
            }
        }
        let started = Arc::new(tokio::sync::Notify::new());
        let server = AsrServer::new("test".into(), Blocks(started.clone()));
        let Backend::Cohort { mux, .. } = &server.backend else { panic!("cohort") };
        let cancel = Arc::new(AtomicBool::new(false));
        let mut held = vec![mux.submit(vec![0.0; 8000], None, String::new(), cancel.clone()).unwrap()];
        tokio::time::timeout(Duration::from_secs(5), started.notified()).await.unwrap();
        while let Ok(reply) = mux.submit(vec![0.0; 8000], None, String::new(), cancel.clone()) {
            held.push(reply);
        }
        let response = server.clone().router(false).oneshot(request("test", "json")).await.unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[axum::http::header::RETRY_AFTER], "1");
        cancel.store(true, Ordering::Relaxed);
    }

    #[tokio::test]
    async fn health_and_admission_follow_shutdown_and_engine_exit() {
        let health = |app: Router| async move {
            app.oneshot(Request::get("/health").body(Body::empty()).unwrap()).await.unwrap().status()
        };
        let server = AsrServer::new("test".into(), Fake);
        let app = server.clone().router(true);
        assert_eq!(health(app.clone()).await, StatusCode::OK);
        server.begin_shutdown();
        assert_eq!(health(app.clone()).await, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(app.oneshot(request("test", "json")).await.unwrap().status(), StatusCode::SERVICE_UNAVAILABLE);

        struct Crashes;
        impl Transcriber for Crashes {
            fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> {
                Ok(None)
            }
            fn transcribe(&mut self, _: &[f32], _: Option<&str>, _: &str, _: &AtomicBool) -> crate::Result<Transcript> {
                panic!("engine worker exits");
            }
        }
        let server = AsrServer::new("test".into(), Crashes);
        let app = server.clone().router(false);
        assert_eq!(health(app.clone()).await, StatusCode::OK);
        let _ = app.clone().oneshot(request("test", "json")).await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while health(app.clone()).await != StatusCode::SERVICE_UNAVAILABLE {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn wav_at_48k_is_resampled() {
        let app = AsrServer::new("test".into(), Fake).router(false);
        let response = app.oneshot(request_wav("test", "json", 48_000, 24_000)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }


    type ClientSocket = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

    /// The next text event, or `{"type":"close","code":..}`; pings and pongs skipped.
    async fn next_event(socket: &mut ClientSocket) -> serde_json::Value {
        use tokio_tungstenite::tungstenite::Message as ClientMessage;
        loop {
            match tokio::time::timeout(Duration::from_secs(5), socket.next()).await.unwrap().unwrap().unwrap() {
                ClientMessage::Text(text) => return serde_json::from_str(&text).unwrap(),
                ClientMessage::Close(frame) => return json!({"type":"close","code":u16::from(frame.unwrap().code)}),
                _ => {}
            }
        }
    }

    #[tokio::test]
    async fn websocket_resamples_48k_streams_deltas_and_ends_on_shutdown() {
        use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};
        struct Lengths(Arc<std::sync::Mutex<Vec<usize>>>);
        impl Transcriber for Lengths {
            fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> {
                Ok(None)
            }
            fn transcribe(&mut self, samples: &[f32], _: Option<&str>, _: &str, _: &AtomicBool) -> crate::Result<Transcript> {
                self.0.lock().unwrap().push(samples.len());
                Ok(Transcript { text: "hello".into(), language: None })
            }
        }
        let lengths = Arc::new(std::sync::Mutex::new(Vec::new()));
        let server = AsrServer::new("test".into(), Lengths(lengths.clone()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = server.clone().router(true);
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let url = format!("ws://{address}/v1/audio/transcriptions/stream");

        let (mut socket, _) = connect_async(&url).await.unwrap();
        socket.send(ClientMessage::Text(json!({"type":"start","version":1,"model":"test","sample_rate":48000,"format":"pcm_s16le","deltas":true}).to_string())).await.unwrap();
        let ready = next_event(&mut socket).await;
        assert_eq!((ready["type"].as_str(), ready["sample_rate"].as_u64(), ready["credit_samples"].as_u64()), (Some("ready"), Some(48_000), Some(48_000)));
        for sequence in 0u64..3 {
            let mut audio = sequence.to_le_bytes().to_vec();
            audio.resize(32_008, 0);
            socket.send(ClientMessage::Binary(audio)).await.unwrap();
        }
        socket.send(ClientMessage::Text(r#"{"type":"finish"}"#.into())).await.unwrap();
        let mut events = Vec::new();
        loop {
            let event = next_event(&mut socket).await;
            if event["type"] == "credit" {
                continue;
            }
            events.push((event["type"].as_str().unwrap().to_owned(), event["text"].as_str().unwrap_or("").to_owned()));
            if event["type"] == "final" {
                break;
            }
        }
        assert_eq!(events, [("delta".to_owned(), "hello".to_owned()), ("final".to_owned(), "hello".to_owned())]);
        assert_eq!(*lengths.lock().unwrap(), [16_000]);

        let (mut socket, _) = connect_async(&url).await.unwrap();
        socket.send(ClientMessage::Text(json!({"type":"start","version":1,"model":"test","sample_rate":8000,"format":"pcm_s16le"}).to_string())).await.unwrap();
        assert_eq!(next_event(&mut socket).await["type"], "ready");
        server.begin_shutdown();
        let error = next_event(&mut socket).await;
        assert_eq!((error["type"].as_str(), error["message"].as_str(), error["terminal"].as_bool()), (Some("error"), Some(SHUTTING_DOWN), Some(true)));
        assert_eq!(next_event(&mut socket).await, json!({"type":"close","code":1001}));
        match connect_async(&url).await {
            Err(tokio_tungstenite::tungstenite::Error::Http(response)) => assert_eq!(response.status(), 503),
            other => panic!("a new session after shutdown must be refused: {:?}", other.map(|_| ())),
        }
        task.abort();
    }


    /// Numbers each transcription (`seg0`, `seg1`, ...); `gate` holds every call until opened.
    struct Numbered {
        calls: Arc<std::sync::atomic::AtomicUsize>,
        gate: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    }
    impl Transcriber for Numbered {
        fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> {
            Ok(None)
        }
        fn transcribe(&mut self, samples: &[f32], _: Option<&str>, _: &str, _: &AtomicBool) -> crate::Result<Transcript> {
            assert!(samples.len() >= 8_000 && samples.len() <= MAX_SAMPLES);
            let (open, wake) = &*self.gate;
            let mut open = open.lock().unwrap();
            while !*open {
                open = wake.wait(open).unwrap();
            }
            let n = self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Transcript { text: format!("seg{n}"), language: None })
        }
    }

    fn numbered(open: bool) -> (Numbered, Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>) {
        let gate = Arc::new((std::sync::Mutex::new(open), std::sync::Condvar::new()));
        (Numbered { calls: Default::default(), gate: gate.clone() }, gate)
    }

    fn tone_or_silence(spans: &[(f32, bool)]) -> Vec<f32> {
        spans
            .iter()
            .flat_map(|&(seconds, tone)| {
                (0..(seconds * 16_000.0) as usize).map(move |i| if tone { crate::asr::endpoint::speechlike(i) } else { 0.0 })
            })
            .collect()
    }

    async fn continuous_server(engine: Numbered) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = AsrServer::new("test".into(), engine).router(true);
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("ws://{address}/v1/audio/transcriptions/stream"), task)
    }

    /// Streams `audio` under credit, then `finish`; every event after `ready` until the close.
    /// `stall` (ms) reports how long the client last waited at zero credit with audio left.
    async fn run_continuous(url: &str, start: serde_json::Value, audio: &[f32]) -> (serde_json::Value, Vec<serde_json::Value>) {
        use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};
        let (mut socket, _) = connect_async(url).await.unwrap();
        socket.send(ClientMessage::Text(start.to_string())).await.unwrap();
        let ready = next_event(&mut socket).await;
        let mut credit = ready["credit_samples"].as_u64().unwrap() as usize;
        let pcm: Vec<i16> = audio.iter().map(|x| (x * 32767.0) as i16).collect();
        let (mut sent, mut sequence, mut events) = (0usize, 0u64, Vec::new());
        while sent < pcm.len() {
            if credit == 0 {
                let event = next_event(&mut socket).await;
                if event["type"] == "credit" {
                    credit += event["credit_samples"].as_u64().unwrap() as usize;
                } else {
                    events.push(event);
                }
                continue;
            }
            let n = credit.min(16_000).min(pcm.len() - sent);
            let mut frame = sequence.to_le_bytes().to_vec();
            frame.extend(pcm[sent..sent + n].iter().flat_map(|s| s.to_le_bytes()));
            socket.send(ClientMessage::Binary(frame)).await.unwrap();
            (sent, sequence, credit) = (sent + n, sequence + 1, credit - n);
        }
        socket.send(ClientMessage::Text(r#"{"type":"finish"}"#.into())).await.unwrap();
        loop {
            let event = next_event(&mut socket).await;
            if event["type"] == "close" {
                events.push(event);
                return (ready, events);
            }
            if event["type"] != "credit" {
                events.push(event);
            }
        }
    }

    fn finals(events: &[serde_json::Value]) -> Vec<&serde_json::Value> {
        events.iter().filter(|e| e["type"] == "final").collect()
    }

    #[tokio::test]
    async fn continuous_session_finals_each_segment_in_order() {
        let (engine, _) = numbered(true);
        let (url, task) = continuous_server(engine).await;
        let spans: Vec<_> = (0..7).flat_map(|_| [(8.0, true), (2.0, false)]).collect();
        let start = json!({"type":"start","version":1,"model":"test","sample_rate":16000,"format":"pcm_s16le","mode":"continuous","deltas":true});
        let (ready, events) = run_continuous(&url, start, &tone_or_silence(&spans)).await;
        assert_eq!((ready["mode"].as_str(), ready["min_silence_ms"].as_u64(), ready["max_audio_samples"].is_null()), (Some("continuous"), Some(600), true));
        let finals = finals(&events);
        assert_eq!(finals.len(), 7);
        for (i, f) in finals.iter().enumerate() {
            assert_eq!((f["segment"].as_u64(), f["text"].as_str()), (Some(i as u64), Some(format!("seg{i}").as_str())));
            let (start_ms, end_ms) = (f["start_ms"].as_u64().unwrap(), f["end_ms"].as_u64().unwrap());
            assert!(start_ms + 200 >= i as u64 * 10_000 && end_ms <= i as u64 * 10_000 + 8_300, "{f}");
        }
        assert!(finals.windows(2).all(|w| w[0]["end_ms"].as_u64() <= w[1]["start_ms"].as_u64()));
        // Every final is preceded by its whole text as a delta (the cohort route has no token deltas).
        let deltas = events.iter().filter(|e| e["type"] == "delta").count();
        assert_eq!(deltas, 7);
        let n = events.len();
        assert_eq!((&events[n - 2]["type"], &events[n - 2]["segments"]), (&json!("done"), &json!(7)));
        assert_eq!(events[n - 1], json!({"type":"close","code":1000}));
        task.abort();
    }

    #[tokio::test]
    async fn continuous_session_cuts_long_speech_and_ignores_silence() {
        let (engine, _) = numbered(true);
        let (url, task) = continuous_server(engine).await;
        let start = json!({"type":"start","version":1,"model":"test","sample_rate":16000,"format":"pcm_s16le","mode":"continuous","max_segment_ms":20000});
        let (_, events) = run_continuous(&url, start.clone(), &tone_or_silence(&[(60.0, true)])).await;
        let finals = finals(&events);
        assert!(finals.len() >= 3);
        assert!(finals.iter().all(|f| f["end_ms"].as_u64().unwrap() - f["start_ms"].as_u64().unwrap() <= 20_000));
        assert_eq!(finals.last().unwrap()["end_ms"].as_u64(), Some(60_000));

        let (_, events) = run_continuous(&url, start, &tone_or_silence(&[(10.0, false)])).await;
        assert_eq!(events, [json!({"type":"done","segments":0}), json!({"type":"close","code":1000})]);
        task.abort();
    }

    #[tokio::test]
    async fn continuous_session_withholds_credit_while_segments_wait() {
        use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};
        let (engine, gate) = numbered(false);
        let (url, task) = continuous_server(engine).await;
        let (mut socket, _) = connect_async(&url).await.unwrap();
        socket.send(ClientMessage::Text(json!({"type":"start","version":1,"model":"test","sample_rate":16000,
            "format":"pcm_s16le","mode":"continuous"}).to_string())).await.unwrap();
        let mut credit = next_event(&mut socket).await["credit_samples"].as_u64().unwrap() as usize;
        let spans: Vec<_> = (0..6).flat_map(|_| [(1.5, true), (1.0, false)]).collect();
        let pcm: Vec<i16> = tone_or_silence(&spans).iter().map(|x| (x * 32767.0) as i16).collect();
        let (mut sent, mut sequence) = (0usize, 0u64);
        // With the engine held, credit stops once two segments wait; the stream cannot finish.
        let starved = loop {
            if credit == 0 {
                match tokio::time::timeout(Duration::from_millis(500), socket.next()).await {
                    Err(_) => break sent,
                    Ok(Some(Ok(ClientMessage::Text(text)))) => {
                        let event: serde_json::Value = serde_json::from_str(&text).unwrap();
                        assert_eq!(event["type"], "credit");
                        credit += event["credit_samples"].as_u64().unwrap() as usize;
                    }
                    Ok(_) => {}
                }
                continue;
            }
            assert!(sent < pcm.len(), "credit never ran out");
            let n = credit.min(16_000).min(pcm.len() - sent);
            let mut frame = sequence.to_le_bytes().to_vec();
            frame.extend(pcm[sent..sent + n].iter().flat_map(|s| s.to_le_bytes()));
            socket.send(ClientMessage::Binary(frame)).await.unwrap();
            (sent, sequence, credit) = (sent + n, sequence + 1, credit - n);
        };
        assert!(starved < pcm.len());
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        let mut finals = 0;
        loop {
            if credit > 0 && sent < pcm.len() {
                let n = credit.min(16_000).min(pcm.len() - sent);
                let mut frame = sequence.to_le_bytes().to_vec();
                frame.extend(pcm[sent..sent + n].iter().flat_map(|s| s.to_le_bytes()));
                socket.send(ClientMessage::Binary(frame)).await.unwrap();
                (sent, sequence, credit) = (sent + n, sequence + 1, credit - n);
                if sent == pcm.len() {
                    socket.send(ClientMessage::Text(r#"{"type":"finish"}"#.into())).await.unwrap();
                }
                continue;
            }
            let event = next_event(&mut socket).await;
            match event["type"].as_str() {
                Some("credit") => credit += event["credit_samples"].as_u64().unwrap() as usize,
                Some("final") => finals += 1,
                Some("done") => break,
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(finals, 6);
        task.abort();
    }

    #[tokio::test]
    async fn utterance_mode_is_the_default_and_bad_continuous_settings_fail() {
        let (engine, _) = numbered(true);
        let (url, task) = continuous_server(engine).await;
        use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};
        for (start, expect) in [
            (json!({"type":"start","version":1,"model":"test","sample_rate":16000,"format":"pcm_s16le"}), "ready"),
            (json!({"type":"start","version":1,"model":"test","sample_rate":16000,"format":"pcm_s16le","mode":"continuous","min_silence_ms":100}), "error"),
            (json!({"type":"start","version":1,"model":"test","sample_rate":16000,"format":"pcm_s16le","mode":"continuous","max_segment_ms":40000}), "error"),
            (json!({"type":"start","version":1,"model":"test","sample_rate":16000,"format":"pcm_s16le","mode":"batch"}), "error"),
        ] {
            let (mut socket, _) = connect_async(&url).await.unwrap();
            socket.send(ClientMessage::Text(start.to_string())).await.unwrap();
            let event = next_event(&mut socket).await;
            assert_eq!(event["type"], expect, "{start}");
            if expect == "ready" {
                assert_eq!(event["mode"], "utterance");
            }
        }
        task.abort();
    }

    #[cfg(feature = "cuda")]
    #[tokio::test]
    async fn serve_hosts_packet_models_beside_the_registry() {
        let backend: Arc<dyn crate::device::Backend> = Arc::new(crate::device::cpu::CpuBackend::new(1));
        let execset = Arc::new(crate::exec::ExecutorSet::bringup(backend).unwrap());
        let state = Arc::new(crate::serve::AppState::new(crate::orch::Registry::new(), execset));
        host_packet_model(&state, "packet-host-test".into(), Box::new(Fake));
        assert!(packet_model_names().contains(&"packet-host-test".to_owned()));
        let app = AsrServer::for_serve(Arc::clone(&state)).transcription_router(true);
        let response = app.clone().oneshot(request("packet-host-test", "text")).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(&response.into_body().collect().await.unwrap().to_bytes()[..], b"hello");
        assert_eq!(state.model_metrics("packet-host-test").serving.asr.completed.load(Ordering::Relaxed), 1);
        assert_eq!(app.oneshot(request("packet-host-missing", "text")).await.unwrap().status(), StatusCode::NOT_FOUND);
        let list = crate::serve::models::list_models(State(Arc::clone(&state))).await.0;
        let card = list.data.iter().find(|c| c.id == "packet-host-test").unwrap();
        assert_eq!(card.x_plow_endpoints, ["audio/transcriptions", "audio/transcriptions/stream"]);
        assert!(!dead_encoders().contains(&"packet-host-test".to_owned()));
        assert!(packet_models_idle());
    }
}
