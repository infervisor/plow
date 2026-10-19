//! OpenAI `POST /v1/audio/speech` on `plowrt serve`.
//!
//! One API for every speech model: each request is checked against the model's
//! [`super::api::SpeechSpec`] the same way, then runs on the model's engine. Long inputs are
//! spoken as sentence segments, streams answer with their first audio, and a whole reply carries
//! the same headers on every model.
//!
//! A model serves speech when its packet declares a `tts.codec_lm.v1` pipeline. The LM stage is
//! submitted to that model's continuous-batching mux like a completion; the codec stage runs on
//! the model's [`Codec`] worker. `stream: true` returns chunked audio: each new frame decodes a
//! window of the codec's `stream.window_frames` and emits the frames that have
//! `stream.lookahead_frames` of right context (both `codec.pkt` parameters).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderName, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use parking_lot::Mutex;
use serde::Deserialize;

use super::api::{Checked, SpeechSpec};
use super::codec::{Codec, Urgency};
use super::realtime::Ticket;
use super::{pcm16, wav_header, SpeechContract};
use crate::serve::session::{InFlight, RequestIds};
use crate::serve::stream::{self as stream_mod, StreamChunk};
use crate::serve::AppState;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechRequest {
    pub model: String,
    /// At most [`super::MAX_INPUT_CHARS`] characters; a long input is spoken as sentence segments.
    pub input: String,
    /// One of the model's voices (its `/v1/models` card: `x_plow_speech.voices`).
    pub voice: String,
    /// `wav` (default) or `pcm` (raw s16le mono at the pipeline sample rate).
    #[serde(default)]
    pub response_format: Option<String>,
    /// OpenAI `speed`; only 1.0 is supported.
    #[serde(default)]
    pub speed: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// -1 or 0: off. Models whose sampler has no top-k reject it.
    #[serde(default)]
    pub top_k: Option<i32>,
    /// Classifier-free guidance weight on guided models (cond + w (cond - uncond)); others
    /// reject it.
    #[serde(default)]
    pub cfg_weight: Option<f32>,
    /// OpenAI's free-text description of the voice or delivery, on models that take one.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Base64 audio of the voice to clone, with its exact transcript `ref_text`, on models that
    /// clone.
    #[serde(default)]
    pub ref_audio: Option<String>,
    #[serde(default)]
    pub ref_text: Option<String>,
    /// Same seed, text and settings: the same audio. Unseeded requests sample anew.
    #[serde(default)]
    pub seed: Option<u64>,
    /// The model's audio tokens per segment (at most the card's `x_plow_speech.max_tokens`);
    /// default: the model's budget for the segment's text.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// ISO 639-1 code of the input's language, one of the card's `x_plow_speech.languages`
    /// (default `default_language`). Models without languages reject it.
    #[serde(default)]
    pub language: Option<String>,
    /// Routing (OpenRouter-compatible): see [`crate::serve::session::RequestIds::apply_body`].
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    pub trace: Option<serde_json::Value>,
}

impl SpeechRequest {
    fn route(&self) -> crate::serve::session::RouteFields {
        crate::serve::session::RouteFields {
            session_id: self.session_id.clone(),
            prompt_cache_key: self.prompt_cache_key.clone(),
            metadata: self.metadata.clone(),
            trace: self.trace.clone(),
        }
    }
}

type Bound<T> = Mutex<HashMap<(PathBuf, u8), Option<Arc<T>>>>;

/// Guided speech front-ends (`tts.guided_lm.v1`) by asset directory and device, bound on first use.
fn guided_models() -> &'static Bound<super::guided_speech::GuidedSpeech> {
    static M: OnceLock<Bound<super::guided_speech::GuidedSpeech>> = OnceLock::new();
    M.get_or_init(Default::default)
}

fn guided_model(
    assets: &Path,
    credit: Arc<crate::sched::admission::DownstreamCredit>,
    device: u8,
) -> Result<Option<Arc<super::guided_speech::GuidedSpeech>>, String> {
    let at = (assets.to_path_buf(), device);
    if let Some(m) = guided_models().lock().get(&at) {
        return Ok(m.clone());
    }
    let model = if super::guided_lm::GuidedLmContract::load(assets)
        .map_err(|e| e.to_string())?
        .is_some()
    {
        let g = super::guided_speech::GuidedSpeech::start(assets, credit, device)
            .map_err(|e| e.to_string())?;
        tracing::info!(dir = %assets.display(), device, "tts: guided speech pipeline bound to the serve mux");
        Some(Arc::new(g))
    } else {
        None
    };
    guided_models().lock().insert(at, model.clone());
    Ok(model)
}

/// Bind every served speech model's host stages now (vocoder / codec graphs, prompt tables), so
/// the first request does not pay them.
pub fn preload(state: &AppState) -> Result<(), String> {
    for slug in state.registry.slugs() {
        if state.mux(&slug).is_some() {
            bind(state, &slug)?;
        }
    }
    Ok(())
}

/// Bind `slug`'s speech pipeline now (a no-op for a model without one, or already bound).
pub fn bind(state: &AppState, slug: &str) -> Result<(), String> {
    let bundle = state
        .registry
        .get(state.model_of(slug))
        .map_err(|e| e.to_string())?;
    let device = state.ordinal_of(slug);
    let bound = guided_model(&bundle.dir, state.downstream(slug), device).and_then(|g| match g {
        Some(_) => Ok(()),
        None => speech_model(&bundle.dir, state.downstream(slug), device).map(drop),
    });
    bound.map_err(|e| format!("{slug}: speech pipeline failed to bind: {e}"))
}

/// `slug` has a bound speech pipeline (a TTS model): its catalogue card lists `audio/speech`.
pub fn serves_speech(state: &AppState, slug: &str) -> bool {
    let Ok(bundle) = state.registry.get(slug) else {
        return false;
    };
    guided_models()
        .lock()
        .iter()
        .any(|((dir, _), g)| *dir == bundle.dir && g.is_some())
        || speech_models()
            .lock()
            .iter()
            .any(|((dir, _), m)| *dir == bundle.dir && m.is_some())
}

enum Pipeline {
    Guided(Arc<super::guided_speech::GuidedSpeech>),
    Speech(Arc<SpeechModel>),
    /// Bound, and the model has no speech pipeline.
    None,
}

/// `dir`'s bound pipeline; `None` when it is not bound (not resident, or mid-switch). Never
/// binds: a managed model's pipeline binds with its engine ([`bind`]).
fn bound_pipeline(dir: &Path, device: u8) -> Option<Pipeline> {
    let at = (dir.to_path_buf(), device);
    if let Some(Some(g)) = guided_models().lock().get(&at) {
        return Some(Pipeline::Guided(Arc::clone(g)));
    }
    match speech_models().lock().get(&at) {
        Some(Some(m)) => Some(Pipeline::Speech(Arc::clone(m))),
        Some(None) => Some(Pipeline::None),
        None => None,
    }
}

/// Bind `dir`'s pipeline on `device` now (unmanaged serve: on first use).
fn bind_pipeline(
    dir: &Path,
    credit: Arc<crate::sched::admission::DownstreamCredit>,
    device: u8,
) -> Result<Pipeline, String> {
    if let Some(g) = guided_model(dir, Arc::clone(&credit), device)? {
        return Ok(Pipeline::Guided(g));
    }
    Ok(match speech_model(dir, credit, device)? {
        Some(m) => Pipeline::Speech(m),
        None => Pipeline::None,
    })
}

/// `slug`'s [`SpeechSpec`] when it serves speech (its `/v1/models` card).
pub fn speech_spec(state: &AppState, slug: &str) -> Option<SpeechSpec> {
    let bundle = state.registry.get(slug).ok()?;
    match bound_pipeline(&bundle.dir, state.ordinal_of(slug))? {
        Pipeline::Guided(g) => Some(g.spec.clone()),
        Pipeline::Speech(m) => Some(m.spec(&**bundle.tokenizer()).clone()),
        Pipeline::None => None,
    }
}

/// Forget `dir`'s speech pipeline on `device`: its vocoder/codec threads exit, freeing their
/// packet runtimes, once the last request holding it ends.
pub fn release(dir: &Path, device: u8) {
    let at = (dir.to_path_buf(), device);
    guided_models().lock().remove(&at);
    speech_models().lock().remove(&at);
}

/// Guided speech (Chatterbox): a long input is spoken as segments, one generation each, in order.
#[allow(clippy::too_many_arguments)]
async fn speech_on_guided(
    w: Arc<super::guided_speech::GuidedSpeech>,
    mux: crate::serve::mux::ModelMux,
    req: SpeechRequest,
    checked: Checked,
    t_arrive: Instant,
    ids: &RequestIds,
    in_flight: InFlight,
    report: Option<crate::serve::session::Report>,
    report_rx: Option<tokio::sync::oneshot::Receiver<crate::serve::session::CacheOutcome>>,
    mut run: crate::serve::turns::StageRun,
) -> Response {
    let lang = checked.language.clone();
    let segs: Vec<String> =
        super::segments(&req.input, w.segment_chars(&req.input, lang.as_deref()))
            .into_iter()
            .map(str::to_owned)
            .collect();
    // Later segments resume no session: the first one's ticket carries the voice prefix.
    let later_ids = RequestIds {
        session: None,
        ..ids.clone()
    };
    if req.stream {
        let first = match w.synthesize_stream(
            &mux,
            req.voice.clone(),
            segs[0].clone(),
            lang.as_deref(),
            &checked.sampling,
            checked.seed,
            ids,
            report,
        ) {
            Ok(rx) => rx,
            Err(e) => return speech_failure(e),
        };
        let cache = crate::serve::session::CacheOutcome::received(report_rx).await;
        run.admitted(cache.and_then(|o| o.at));
        let (out_tx, out_rx) = tokio::sync::mpsc::channel(64);
        let (head_tx, head_rx) = tokio::sync::oneshot::channel();
        let wav = checked.wav.then(|| wav_header(w.sample_rate, u32::MAX));
        tokio::spawn(guided_stream_task(
            w, mux, req.voice, segs, checked, later_ids, first, out_tx, head_tx, t_arrive, run,
        ));
        return stream_response(out_rx, head_rx, wav, in_flight, cache).await;
    }
    let (mut pcm, mut joiner) = (Vec::new(), super::Joins::new(w.sample_rate));
    let (mut tokens, mut t3_ms, mut s3gen_ms) = (0, 0.0, 0.0);
    let mut report = report;
    for (k, text) in segs.iter().enumerate() {
        let a = match w
            .synthesize(
                &mux,
                req.voice.clone(),
                text.clone(),
                lang.as_deref(),
                &checked.sampling,
                checked.seed.wrapping_add(k as u64),
                if k == 0 { ids } else { &later_ids },
                report.take(),
            )
            .await
        {
            Ok(a) => a,
            Err(e) => return speech_failure(e),
        };
        (tokens, t3_ms, s3gen_ms) = (tokens + a.tokens, t3_ms + a.t3_ms, s3gen_ms + a.s3gen_ms);
        if segs.len() == 1 {
            pcm = a.pcm;
        } else {
            if k > 0 {
                joiner.next_segment();
            }
            joiner.push(&a.pcm, f64::INFINITY, &mut pcm);
        }
    }
    let cache = crate::serve::session::CacheOutcome::received(report_rx).await;
    run.admitted(cache.and_then(|o| o.at));
    run.first();
    run.done();
    let total = t_arrive.elapsed().as_secs_f64();
    let audio_s = pcm.len() as f64 / f64::from(w.sample_rate);
    tracing::info!(
        tokens,
        segments = segs.len(),
        audio_s,
        t3_ms,
        s3gen_ms,
        total_ms = total * 1e3,
        rtf = total / audio_s,
        "tts: chatterbox speech"
    );
    let mut response = whole_response(checked.wav, w.sample_rate, &pcm, t3_ms);
    run.stamp(&mut response);
    if let Some(cache) = cache {
        cache.stamp(&mut response);
    }
    response
}

/// Forwards a guided stream's segments in order; the stamped headers go to `head` with the first
/// audio.
#[allow(clippy::too_many_arguments)]
async fn guided_stream_task(
    w: Arc<super::guided_speech::GuidedSpeech>,
    mux: crate::serve::mux::ModelMux,
    voice: String,
    segs: Vec<String>,
    checked: Checked,
    later_ids: RequestIds,
    mut ev: tokio::sync::mpsc::UnboundedReceiver<super::guided_speech::StreamEvent>,
    out: tokio::sync::mpsc::Sender<Result<Vec<u8>, std::io::Error>>,
    head: tokio::sync::oneshot::Sender<axum::http::HeaderMap>,
    t_arrive: Instant,
    mut run: crate::serve::turns::StageRun,
) {
    use super::guided_speech::StreamEvent;
    let sr = f64::from(w.sample_rate);
    let mut head = Some(head);
    let mut joiner = super::Joins::new(w.sample_rate);
    // Samples sent and when the first went out (the client's playback clock).
    let (mut sent, mut playing, mut first) = (0usize, None::<Instant>, None);
    let (mut tokens, mut t3, mut s3) = (0, 0.0, 0.0);
    let fail = |e: String| Err(std::io::Error::other(e));
    for (k, text) in segs.iter().enumerate() {
        if k > 0 {
            ev = match w.synthesize_stream(
                &mux,
                voice.clone(),
                text.clone(),
                checked.language.as_deref(),
                &checked.sampling,
                checked.seed.wrapping_add(k as u64),
                &later_ids,
                None,
            ) {
                Ok(rx) => rx,
                Err(e) => return drop(out.send(fail(speech_error_text(e))).await),
            };
            joiner.next_segment();
        }
        loop {
            match ev.recv().await {
                Some(StreamEvent::Pcm(p)) => {
                    let fresh = if segs.len() == 1 {
                        p
                    } else {
                        let lead = playing.map_or(f64::INFINITY, |t| {
                            sent as f64 / sr - t.elapsed().as_secs_f64()
                        });
                        let mut v = Vec::with_capacity(p.len());
                        joiner.push(&p, lead, &mut v);
                        v
                    };
                    if fresh.is_empty() {
                        continue;
                    }
                    run.audio(fresh.len(), sr);
                    if let Some(h) = head.take() {
                        let _ = h.send(run.headers());
                    }
                    first.get_or_insert_with(|| t_arrive.elapsed());
                    playing.get_or_insert_with(Instant::now);
                    sent += fresh.len();
                    let mut bytes = Vec::with_capacity(fresh.len() * 2);
                    pcm16(&fresh, &mut bytes);
                    if out.send(Ok(bytes)).await.is_err() {
                        return; // client gone: dropping `ev` frees the request's pair
                    }
                }
                Some(StreamEvent::Done {
                    tokens: n,
                    t3_ms,
                    s3gen_ms,
                }) => {
                    (tokens, t3, s3) = (tokens + n, t3 + t3_ms, s3 + s3gen_ms);
                    break;
                }
                Some(StreamEvent::Err(e)) => return drop(out.send(fail(e)).await),
                None => {
                    return drop(out.send(fail("speech stream ended early".into())).await);
                }
            }
        }
    }
    run.done();
    let total = t_arrive.elapsed().as_secs_f64();
    let audio_s = sent as f64 / sr;
    tracing::info!(
        tokens,
        segments = segs.len(),
        audio_s,
        t3_ms = t3,
        s3gen_ms = s3,
        ttfa_ms = first.map(|d| d.as_secs_f64() * 1e3),
        total_ms = total * 1e3,
        rtf = total / audio_s,
        "tts: chatterbox stream"
    );
}

/// A stream's response, sent with its first audio: an error before any audio is a status, and the
/// headers carry the first audio's timing.
async fn stream_response(
    mut out_rx: tokio::sync::mpsc::Receiver<Result<Vec<u8>, std::io::Error>>,
    head: tokio::sync::oneshot::Receiver<axum::http::HeaderMap>,
    wav: Option<Vec<u8>>,
    in_flight: InFlight,
    cache: Option<crate::serve::session::CacheOutcome>,
) -> Response {
    use futures::StreamExt;
    let first = match out_rx.recv().await {
        Some(Ok(bytes)) => bytes,
        Some(Err(e)) => return server_error(e.to_string()),
        None => return server_error("speech stream ended before any audio"),
    };
    let stamped = head.await.unwrap_or_default();
    let content_type = if wav.is_some() {
        "audio/wav"
    } else {
        "audio/pcm"
    };
    let lead = futures::stream::iter(wav.into_iter().chain([first]).map(Ok));
    let rest = futures::stream::poll_fn(move |cx| {
        let _held = &in_flight;
        out_rx.poll_recv(cx)
    });
    let mut response = (
        [(header::CONTENT_TYPE, content_type)],
        Body::from_stream(lead.chain(rest)),
    )
        .into_response();
    response.headers_mut().extend(stamped);
    if let Some(cache) = cache {
        cache.stamp(&mut response);
    }
    response
}

/// A whole reply: the audio and the same headers on every model.
fn whole_response(wav: bool, sample_rate: u32, pcm: &[f32], lm_ms: f64) -> Response {
    let mut out = if wav {
        wav_header(sample_rate, (pcm.len() * 2) as u32)
    } else {
        Vec::new()
    };
    pcm16(pcm, &mut out);
    let content_type = if wav { "audio/wav" } else { "audio/pcm" };
    (
        [
            (header::CONTENT_TYPE, content_type.to_string()),
            (
                HeaderName::from_static("x-plow-audio-seconds"),
                format!("{:.3}", pcm.len() as f64 / f64::from(sample_rate)),
            ),
            (
                HeaderName::from_static("x-plow-lm-ms"),
                format!("{lm_ms:.1}"),
            ),
        ],
        out,
    )
        .into_response()
}

pub struct SpeechModel {
    pub contract: SpeechContract,
    codec: Codec,
    realtime: Arc<super::realtime::RealTime>,
    spec: OnceLock<SpeechSpec>,
}

impl SpeechModel {
    /// The model's [`SpeechSpec`] (its voices may come from the tokenizer: built once).
    pub fn spec(&self, tok: &dyn crate::text::tokenizer::Tokenize) -> &SpeechSpec {
        self.spec.get_or_init(|| self.contract.spec(tok))
    }
}

/// Speech models by asset directory and device, bound on first use.
fn speech_models() -> &'static Bound<SpeechModel> {
    static M: OnceLock<Bound<SpeechModel>> = OnceLock::new();
    M.get_or_init(Default::default)
}

/// `credit` is the serving model's downstream credit; the codec's backlog gates its admission.
pub fn speech_model(
    assets: &Path,
    credit: Arc<crate::sched::admission::DownstreamCredit>,
    device: u8,
) -> Result<Option<Arc<SpeechModel>>, String> {
    let at = (assets.to_path_buf(), device);
    if let Some(m) = speech_models().lock().get(&at) {
        return Ok(m.clone());
    }
    let model = match SpeechContract::load(assets).map_err(|e| e.to_string())? {
        None => None,
        Some(contract) => {
            let mut codec = Codec::load_on(assets, device)?;
            codec.couple(credit);
            if codec.frame_codes != contract.frame_codes
                || codec.frame_samples != contract.frame_samples
            {
                return Err(format!(
                    "codec packet frames ({} codes, {} samples) disagree with the speech contract ({}, {})",
                    codec.frame_codes, codec.frame_samples, contract.frame_codes, contract.frame_samples
                ));
            }
            tracing::info!(pipeline = %contract.pipeline, sample_rate = contract.sample_rate, "tts: speech pipeline bound");
            let realtime = super::realtime::RealTime::new(
                contract.frame_samples as f64 / f64::from(contract.sample_rate),
                contract.frame_codes,
            );
            Some(Arc::new(SpeechModel {
                contract,
                codec,
                realtime,
                spec: OnceLock::new(),
            }))
        }
    };
    speech_models().lock().insert(at, model.clone());
    Ok(model)
}

fn bad(msg: impl Into<String>, param: &str) -> Response {
    crate::serve::api_error(
        StatusCode::BAD_REQUEST,
        msg,
        "invalid_request_error",
        Some("invalid_value"),
        Some(param.into()),
    )
}

/// A full model queue; the guided path's is [`super::guided_speech::QUEUE_FULL`].
const MUX_FULL: &str = "model request queue full";

/// Every 429 says when to retry.
fn overloaded(msg: &str, retry: std::time::Duration) -> Response {
    let mut r = crate::serve::api_error(
        StatusCode::TOO_MANY_REQUESTS,
        msg,
        "rate_limit_error",
        Some("server_overloaded"),
        None,
    );
    r.headers_mut().insert(
        header::RETRY_AFTER,
        axum::http::HeaderValue::from(retry.as_secs().max(1)),
    );
    r
}

/// `req` against the model's spec; a 400 naming the first failing field.
fn check(spec: &SpeechSpec, req: &SpeechRequest) -> Result<Checked, Response> {
    spec.check(req).map_err(|e| bad(e.message, e.param))
}

fn busy(retry: std::time::Duration) -> Response {
    overloaded("speech is at real-time capacity; retry", retry)
}

fn server_error(msg: impl Into<String>) -> Response {
    let msg = msg.into();
    if msg == super::guided_speech::QUEUE_FULL || msg == MUX_FULL {
        return overloaded(&msg, std::time::Duration::from_secs(1));
    }
    // The text alone overflows the (narrowed) context: the request, not the server, is at fault.
    if msg.starts_with("context length exceeded") {
        return crate::serve::api_error(
            StatusCode::BAD_REQUEST,
            msg,
            "invalid_request_error",
            Some("context_length_exceeded"),
            Some("input".into()),
        );
    }
    crate::serve::api_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        msg,
        "server_error",
        None,
        None,
    )
}

fn speech_error_text(e: super::guided_speech::SpeechError) -> String {
    match e {
        super::guided_speech::SpeechError::Invalid(msg, _)
        | super::guided_speech::SpeechError::Failed(msg) => msg,
    }
}

fn speech_failure(e: super::guided_speech::SpeechError) -> Response {
    match e {
        super::guided_speech::SpeechError::Invalid(msg, param) => bad(msg, param),
        super::guided_speech::SpeechError::Failed(msg) => server_error(msg),
    }
}

pub async fn speech(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    req: Result<Json<SpeechRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let mut ids = match RequestIds::from_headers(&headers) {
        Ok(ids) => ids,
        Err(e) => return bad(e, "x-session-id"),
    };
    if let Err(e) = req
        .as_ref()
        .map_or(Ok(()), |Json(r)| ids.apply_body(&r.route()))
    {
        return bad(e, "session_id");
    }
    if let Some(r) = crate::serve::overload::gate(&ids) {
        return r;
    }
    let (report, mut report_rx) = ids.report();
    let mut response = speech_with(state, req, &ids, report, &mut report_rx).await;
    ids.stamp(&mut response);
    if response.status().is_success() {
        if let Some(cache) = crate::serve::session::CacheOutcome::received(report_rx).await {
            cache.stamp(&mut response);
        }
    }
    response
}

async fn speech_with(
    state: Arc<AppState>,
    req: Result<Json<SpeechRequest>, axum::extract::rejection::JsonRejection>,
    ids: &RequestIds,
    report: Option<crate::serve::session::Report>,
    report_rx: &mut Option<tokio::sync::oneshot::Receiver<crate::serve::session::CacheOutcome>>,
) -> Response {
    let Json(mut req) = match req {
        Ok(r) => r,
        Err(e) => return crate::serve::json_rejection(e),
    };
    let t_arrive = Instant::now();
    if let Some(canonical) = state.registry.resolve(&req.model) {
        req.model = canonical;
    }
    // The pipeline binds with the engine and leaves with it: an eviction between the residency
    // check and the lookup sends the request around again.
    let mut attempts = 0;
    let dp = state.dp_set(&req.model).cloned();
    let mut exclude = 0u32;
    let mut key = req.model.clone();
    let (mux, bundle, pipeline) = loop {
        let routed = match dp.as_deref() {
            Some(set) => {
                if let Err(e) = state.dp_admit(set).await {
                    return crate::serve::api_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        e.to_string(),
                        "server_error",
                        None,
                        None,
                    );
                }
                state
                    .dp_route(set, ids.session.as_deref(), None, exclude)
                    .map(|(rank, mux, _, _)| {
                        exclude |= 1 << rank;
                        key.clone_from(&set.ranks[rank].key);
                        mux
                    })
            }
            None => {
                if let Some(mgr) = state.manager_for(&req.model) {
                    if mgr.manages(&req.model) {
                        if let Err(e) = mgr.ensure_resident(&req.model).await {
                            return crate::serve::api_error(
                                StatusCode::SERVICE_UNAVAILABLE,
                                e.to_string(),
                                "server_error",
                                None,
                                None,
                            );
                        }
                    }
                }
                state.mux(&req.model)
            }
        };
        let (Some(mux), Ok(bundle)) = (routed, state.registry.get(&req.model)) else {
            attempts += 1;
            if attempts < 3
                && (dp.is_some()
                    || state
                        .manager_for(&req.model)
                        .is_some_and(|m| m.manages(&req.model)))
            {
                continue;
            }
            return crate::serve::api_error(
                StatusCode::NOT_FOUND,
                format!("no model registered for '{}'.", req.model),
                "invalid_request_error",
                Some("model_not_found"),
                Some("model".into()),
            );
        };
        let managed = dp.is_some()
            || state
                .manager_for(&req.model)
                .is_some_and(|m| m.manages(&req.model));
        match bound_pipeline(&bundle.dir, state.ordinal_of(&key)) {
            Some(pipeline) => break (mux, bundle, Some(pipeline)),
            None if !managed => break (mux, bundle, None),
            None => {
                attempts += 1;
                if attempts >= 3 {
                    return crate::serve::api_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "speech pipeline is switching; retry",
                        "server_error",
                        None,
                        None,
                    );
                }
            }
        }
    };
    let Some(in_flight) = ids.begin(&req.model) else {
        return crate::serve::api_error(
            StatusCode::CONFLICT,
            format!(
                "request {} is already in flight in this session",
                ids.request
            ),
            "invalid_request_error",
            Some("duplicate_request_id"),
            None,
        );
    };
    let mut run = crate::serve::turns::StageRun::start(
        ids,
        crate::serve::turns::Kind::Tts,
        &req.model,
        Some(state.model_metrics(&key)),
        t_arrive,
        true,
    );
    let ids = &RequestIds {
        turn_key: run.key(),
        ..ids.clone()
    };
    let pipeline = match pipeline {
        Some(p) => p,
        // Unmanaged (single-model) serve: bind on first use.
        None => match tokio::task::block_in_place(|| {
            bind_pipeline(
                &bundle.dir,
                state.downstream(&req.model),
                state.ordinal_of(&req.model),
            )
        }) {
            Ok(p) => p,
            Err(e) => return server_error(format!("speech pipeline: {e}")),
        },
    };
    let model = match pipeline {
        Pipeline::Guided(g) => {
            let checked = match check(&g.spec, &req) {
                Ok(c) => c,
                Err(r) => return r,
            };
            return speech_on_guided(
                g,
                mux,
                req,
                checked,
                t_arrive,
                ids,
                in_flight,
                report,
                report_rx.take(),
                run,
            )
            .await;
        }
        Pipeline::Speech(m) => m,
        Pipeline::None => {
            return crate::serve::models::unserved(
                &req.model,
                "audio/speech",
                &crate::serve::models::endpoints(&state, &req.model),
            )
        }
    };
    let tok = bundle.tokenizer();
    let checked = match check(model.spec(&**tok), &req) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let c = &model.contract;
    let voice = req.voice.clone();
    let wav = checked.wav;
    // Past the cap the reference budget clips the audio: a longer input is spoken in order as
    // sentence-sized segments, each with its full budget.
    let maker = SegmentMaker {
        tok: Arc::clone(tok),
        contract: c.clone(),
        voice: voice.clone(),
        max_tokens: checked.sampling.max_tokens,
        temperature: checked.sampling.temperature,
        top_p: checked.sampling.top_p,
        repetition_penalty: checked.sampling.repetition_penalty,
        top_k: checked.sampling.top_k.unwrap_or(0),
        seed: checked.seed,
    };
    let segs: Vec<Segment> = super::segments(&req.input, c.segment_chars())
        .into_iter()
        .enumerate()
        .map(|(k, t)| maker.make(k, t))
        .collect();

    let need = super::realtime::Need {
        stream: req.stream,
        chars: segs.iter().map(|s| s.chars).collect(),
        budget: segs
            .iter()
            .map(|s| s.gen.max_tokens / c.frame_codes)
            .collect(),
    };
    let ticket = match model.realtime.admit(need).await {
        Ok(t) => Arc::new(t),
        Err(retry) => return busy(retry),
    };
    let class = if req.stream {
        crate::serve::mux::JobClass::Critical
    } else {
        crate::serve::mux::JobClass::Normal
    };
    let first = segs[0].clone();
    let (tx, rx) = stream_mod::channel();
    let opts = crate::serve::mux::JobOpts {
        class,
        raw_tokens: true,
        speech: None,
        session: ids.session.as_ref().and_then(|_| {
            ids.ticket(
                crate::serve::session::row_keys(&first.prompt_ids, &[], &[]),
                report,
            )
        }),
        turn: ids.turn_key.clone(),
        continuing: run.continuing(),
        prefix: None,
        tenant: None,
        round: 0,
        cached: None,
        mm: None,
    };
    let job = crate::serve::mux::Job {
        prompt_ids: first.prompt_ids,
        gen: first.gen,
        arrived: Instant::now(),
        respond: tx,
        opts,
    };
    if let Err(err) = mux.submit_arrived(job, t_arrive, Some(mux.ingress())) {
        return match err {
            crate::serve::mux::SubmitError::Full(_) => {
                overloaded(MUX_FULL, std::time::Duration::from_secs(1))
            }
            crate::serve::mux::SubmitError::Closed(_) => crate::serve::api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "model dispatcher unavailable",
                "server_error",
                None,
                None,
            ),
        };
    }
    // The codec's noise seed: odd, and unrelated to the LM's.
    let seed = super::api::mix(checked.seed) | 1;
    let mut later = Later {
        mux: mux.clone(),
        segs,
        class,
        seed,
        maker,
    };
    if req.stream {
        let (out_tx, out_rx) = tokio::sync::mpsc::channel(64);
        let (head_tx, head_rx) = tokio::sync::oneshot::channel();
        let cache = crate::serve::session::CacheOutcome::received(report_rx.take()).await;
        run.admitted(cache.and_then(|o| o.at));
        let header = wav.then(|| wav_header(c.sample_rate, u32::MAX));
        tokio::spawn(stream_task(
            Arc::clone(&model),
            rx,
            later,
            ticket,
            out_tx,
            head_tx,
            seed,
            t_arrive,
            run,
        ));
        return stream_response(out_rx, head_rx, header, in_flight, cache).await;
    }
    let (mut pcm, mut n_tokens, mut frames, mut t_lm) =
        (Vec::new(), 0, 0, std::time::Duration::ZERO);
    // Sample offsets where one segment's audio follows another's.
    let (mut joins, mut joiner) = (Vec::new(), super::Joins::new(c.sample_rate));
    let mut rx = Some(rx);
    let mut k = 0;
    while k < later.segs.len() {
        let (mut attempt, mut tries) = (0, 0);
        loop {
            let rx = match rx.take() {
                Some(rx) => rx,
                None => match later.submit(k, attempt) {
                    Ok(rx) => rx,
                    Err(e) => return server_error(e),
                },
            };
            let t0 = Instant::now();
            ticket.segment(k);
            let (codes, n) = match collect_codes(c, rx, &ticket).await {
                Ok(v) => v,
                Err(e) => return crate::serve::api_error_for(&e),
            };
            ticket.segment_done();
            t_lm += if k == 0 && attempt == 0 {
                t_arrive.elapsed()
            } else {
                t0.elapsed()
            };
            n_tokens += n;
            let f = codes.len() / c.frame_codes;
            let p = if f == 0 {
                Vec::new()
            } else {
                match decode_all(
                    &model,
                    &codes[..f * c.frame_codes],
                    f,
                    seed ^ segment_salt(k),
                )
                .await
                {
                    Ok(p) => p,
                    Err(e) => return server_error(e),
                }
            };
            frames += f;
            if later.segs.len() == 1 {
                pcm = p;
                break;
            }
            match later.judge(k, f, f > 0 && joiner.failed(&p), &mut tries) {
                Verdict::Keep => {}
                Verdict::Retry => {
                    attempt += 1;
                    continue;
                }
                Verdict::Fail => return server_error(format!("segment {k} produced no speech")),
            }
            if !pcm.is_empty() {
                joiner.next_segment();
                joins.push(pcm.len());
            }
            joiner.push(&p, f64::INFINITY, &mut pcm);
            break;
        }
        k += 1;
    }
    drop(ticket);
    if frames == 0 {
        return server_error("model produced no audio frames");
    }
    let audio_s = pcm.len() as f64 / f64::from(c.sample_rate);
    let total = t_arrive.elapsed().as_secs_f64();
    tracing::info!(
        tokens = n_tokens,
        frames,
        segments = later.segs.len(),
        ?joins,
        audio_s,
        lm_ms = t_lm.as_secs_f64() * 1e3,
        total_ms = total * 1e3,
        rtf = total / audio_s,
        "tts: speech"
    );
    let cache = crate::serve::session::CacheOutcome::received(report_rx.take()).await;
    run.admitted(cache.and_then(|o| o.at));
    run.first();
    run.done();
    let mut response = whole_response(wav, c.sample_rate, &pcm, t_lm.as_secs_f64() * 1e3);
    run.stamp(&mut response);
    if let Some(cache) = cache {
        cache.stamp(&mut response);
    }
    response
}

/// A mute or droning segment of a multi-segment request is generated again with another seed;
/// still failing, it is split in two (Orpheus goes mute on some long run-on segments and not on
/// their halves), down to `SPLIT_MIN_CHARS`; only then does the request fail.
const MUTE_RETRIES: usize = 1;
const SPLIT_MIN_CHARS: usize = 40;

/// One segment's LM job.
#[derive(Clone)]
struct Segment {
    text: String,
    prompt_ids: Vec<u32>,
    gen: crate::serve::GenParams,
    chars: usize,
}

/// Builds a segment's prompt and generation parameters from its text.
struct SegmentMaker {
    tok: Arc<dyn crate::text::tokenizer::Tokenize>,
    contract: SpeechContract,
    voice: String,
    max_tokens: Option<usize>,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    top_k: usize,
    /// The request's seed; segment `k` samples with `seed + k`.
    seed: u64,
}

impl SegmentMaker {
    fn make(&self, k: usize, text: &str) -> Segment {
        let c = &self.contract;
        let mut prompt_ids = c.prefix.clone();
        prompt_ids.extend(
            self.tok
                .encode_with_special_tokens(&c.prompt_text(&self.voice, text), false),
        );
        prompt_ids.extend_from_slice(&c.suffix);
        let mut gen = crate::serve::GenParams::default();
        gen.max_tokens = self
            .max_tokens
            .unwrap_or_else(|| c.max_new_tokens(text))
            .min(c.max_new_tokens_cap);
        gen.params.temperature = self.temperature;
        gen.params.top_p = self.top_p;
        gen.params.repetition_penalty = self.repetition_penalty;
        gen.params.top_k = self.top_k;
        gen.seed = Some(self.seed.wrapping_add(k as u64));
        gen.stop_token_ids = c.stops.clone();
        Segment {
            text: text.to_owned(),
            prompt_ids,
            gen,
            chars: text.chars().count(),
        }
    }
}

/// The request's segments; each after the first is submitted as the previous one finishes.
struct Later {
    mux: crate::serve::mux::ModelMux,
    segs: Vec<Segment>,
    class: crate::serve::mux::JobClass,
    /// Base of a retry's sampling seed when the request gave none (unseeded sampling repeats).
    seed: u64,
    maker: SegmentMaker,
}

impl Later {
    /// The fate of segment `k`'s attempt with `frames` frames, `failed` (mute or droning) or not;
    /// `tries` counts the attempts at its current text. A split replaces segment `k` by its
    /// halves and generates the first.
    fn judge(&mut self, k: usize, frames: usize, failed: bool, tries: &mut usize) -> Verdict {
        let v = verdict(frames, failed, *tries);
        if v == Verdict::Retry {
            *tries += 1;
            tracing::warn!(
                segment = k,
                tries = *tries,
                "tts: mute or droning segment; retrying"
            );
        }
        if v != Verdict::Fail {
            return v;
        }
        let Some(halves) = split_halves(&self.segs[k].text) else {
            tracing::warn!(
                segment = k,
                chars = self.segs[k].chars,
                "tts: segment produced no speech; failing the request"
            );
            return v;
        };
        tracing::warn!(
            segment = k,
            chars = self.segs[k].chars,
            "tts: segment still mute; splitting it"
        );
        let made: Vec<Segment> = halves.iter().map(|t| self.maker.make(k, t)).collect();
        self.segs.splice(k..=k, made);
        *tries = 0;
        Verdict::Retry
    }

    fn submit(&self, k: usize, attempt: usize) -> Result<stream_mod::ChunkReceiver, String> {
        let mut s = self.segs[k].clone();
        if attempt > 0 {
            s.gen.seed =
                Some(s.gen.seed.unwrap_or(self.seed ^ k as u64) ^ ((attempt as u64) << 48));
        }
        let (tx, rx) = stream_mod::channel();
        let opts = crate::serve::mux::JobOpts {
            class: self.class,
            raw_tokens: true,
            speech: None,
            session: None,
            turn: None,
            continuing: false,
            prefix: None,
            tenant: None,
            round: 0,
            cached: None,
            mm: None,
        };
        let job = crate::serve::mux::Job {
            prompt_ids: s.prompt_ids,
            gen: s.gen,
            arrived: Instant::now(),
            respond: tx,
            opts,
        };
        self.mux.submit(job).map(|()| rx).map_err(|e| match e {
            crate::serve::mux::SubmitError::Full(_) => MUX_FULL.to_string(),
            crate::serve::mux::SubmitError::Closed(_) => "model dispatcher unavailable".to_string(),
        })
    }
}

/// `text` as two segments of about half its length, or `None` under `SPLIT_MIN_CHARS`.
fn split_halves(text: &str) -> Option<Vec<String>> {
    let n = text.chars().count();
    if n < SPLIT_MIN_CHARS {
        return None;
    }
    let parts: Vec<String> = super::segments(text, n.div_ceil(2) + n / 8)
        .into_iter()
        .map(str::to_owned)
        .collect();
    (parts.len() >= 2).then_some(parts)
}

/// Codec seed salt of segment `k` (0 for the first: a one-segment request decodes as before).
fn segment_salt(k: usize) -> u64 {
    (k as u64) << 40
}

/// Whole utterance; beyond the codec's frame capacity, windows with the codec's context frames
/// on each side keep their centres.
async fn decode_all(
    model: &SpeechModel,
    codes: &[i32],
    frames: usize,
    seed: u64,
) -> Result<Vec<f32>, String> {
    let (fc, fs, max) = (
        model.contract.frame_codes,
        model.contract.frame_samples,
        model.codec.max_frames,
    );
    if frames <= max {
        return model
            .codec
            .decode(codes.to_vec(), frames, seed, Urgency::Whole)
            .await;
    }
    let window = model.codec.window;
    let step = max - 2 * window;
    let mut pcm = Vec::with_capacity(frames * fs);
    let mut s = 0;
    while s < frames {
        let e = (s + step).min(frames);
        let (ws, we) = (s.saturating_sub(window), (e + window).min(frames));
        let w = model
            .codec
            .decode(
                codes[ws * fc..we * fc].to_vec(),
                we - ws,
                seed ^ s as u64,
                Urgency::Whole,
            )
            .await?;
        pcm.extend_from_slice(
            w.get((s - ws) * fs..(e - ws) * fs)
                .ok_or_else(|| short_window(w.len(), we - ws))?,
        );
        s = e;
    }
    Ok(pcm)
}

fn short_window(samples: usize, frames: usize) -> String {
    format!("codec returned {samples} samples for a {frames}-frame window")
}

/// Drain the LM stream promptly (the mux cuts a consumer that falls behind) keeping the codes.
async fn collect_codes(
    c: &SpeechContract,
    mut rx: stream_mod::ChunkReceiver,
    ticket: &Ticket,
) -> crate::Result<(Vec<i32>, usize)> {
    let (mut codes, mut n) = (Vec::new(), 0);
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { id, .. } => {
                n += 1;
                if let Some(code) = c.code_of(codes.len(), id) {
                    codes.push(code);
                    if codes.len() % c.frame_codes == 0 {
                        ticket.frame();
                    }
                }
            }
            StreamChunk::Done { .. } => break,
            StreamChunk::Err(e) => return Err(e),
        }
    }
    Ok((codes, n))
}

/// The emission plan for a stream holding `n` complete frames of which `emitted` are sent:
/// `Some((window_start, window_end, emit_to))`, or `None` when nothing new is ready.
///
/// Every window carries `window - lookahead - 1` frames of left context and `lookahead` of right
/// context (the one-frame window's shape). After the first audio, frames go out `chunk` at a time:
/// a window costs its codec capacity whatever it emits, and one frame per window made streaming
/// Veena c64 36% slower than whole-utterance decoding.
fn stream_step(
    n: usize,
    emitted: usize,
    done: bool,
    window: usize,
    lookahead: usize,
    chunk: usize,
) -> Option<(usize, usize, usize)> {
    let upto = if done { n } else { n.saturating_sub(lookahead) };
    let ready = upto > emitted && (done || emitted == 0 || upto - emitted >= chunk);
    ready.then(|| {
        (
            emitted.saturating_sub(window.saturating_sub(lookahead + 1)),
            n.min(upto + lookahead),
            upto,
        )
    })
}

enum SegMsg {
    Frame(Vec<i32>),
    /// The current segment attempt generated its last frame. A multi-segment drain then waits for
    /// the emitter's verdict on it before it starts anything else.
    End,
    /// The current segment was mute or droned and generates again: drop its frames.
    Retry,
    Err(String),
}

/// What becomes of a multi-segment request's segment attempt.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Verdict {
    Keep,
    Retry,
    /// Still without speech after `MUTE_RETRIES`: the request fails rather than skip the text.
    Fail,
}

fn verdict(frames: usize, failed: bool, attempt: usize) -> Verdict {
    match (frames > 0 && !failed, attempt < MUTE_RETRIES) {
        (true, _) => Verdict::Keep,
        (false, true) => Verdict::Retry,
        (false, false) => Verdict::Fail,
    }
}

/// Identity of a segment attempt: a cut or verdict raised for one never acts on another.
fn attempt_tag(segment: usize, attempt: usize) -> u64 {
    ((segment as u64 + 1) << 32) | attempt as u64
}

/// The emitter's verdict for `tag`, ignoring any other (stale) one; `None` when it is gone.
async fn verdict_for(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<(u64, bool)>,
    tag: u64,
) -> Option<bool> {
    loop {
        let (t, failed) = rx.recv().await?;
        if t == tag {
            return Some(failed);
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn stream_task(
    model: Arc<SpeechModel>,
    rx: stream_mod::ChunkReceiver,
    mut later: Later,
    ticket: Arc<Ticket>,
    out: tokio::sync::mpsc::Sender<Result<Vec<u8>, std::io::Error>>,
    head: tokio::sync::oneshot::Sender<axum::http::HeaderMap>,
    seed: u64,
    t_arrive: Instant,
    mut run: crate::serve::turns::StageRun,
) {
    use std::sync::atomic::Ordering::SeqCst;
    let mut head = Some(head);
    let c = model.contract.clone();
    let sr = f64::from(c.sample_rate);
    let (fc, fs) = (c.frame_codes, c.frame_samples);
    let n_segs = later.segs.len();
    let (ftx, mut frx) = tokio::sync::mpsc::unbounded_channel::<SegMsg>();
    // The emitter's verdict per segment attempt (multi-segment only), tagged by `attempt_tag`.
    let (vtx, mut vrx) = tokio::sync::mpsc::unbounded_channel::<(u64, bool)>();
    // The segment attempt (`attempt_tag`) the emitter cut short; 0: none.
    let cut = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let lm = Arc::clone(&ticket);
    let lm_cut = Arc::clone(&cut);
    // The LM drain never waits on the codec within a segment. Between segments of a multi-segment
    // stream it waits for the emitter's verdict, so a retry targets the attempt found mute.
    tokio::spawn(async move {
        let drain = async {
            let mut rx = Some(rx);
            let mut k = 0;
            while k < later.segs.len() {
                // `attempt` names each generation of segment `k` (a retry or a split's first
                // half), `tries` the generations of its current text.
                let (mut attempt, mut tries) = (0, 0);
                loop {
                    let mut rx = match rx.take() {
                        Some(rx) => rx,
                        None => match later.submit(k, attempt) {
                            Ok(rx) => rx,
                            Err(e) => return drop(ftx.send(SegMsg::Err(e))),
                        },
                    };
                    let tag = attempt_tag(k, attempt);
                    lm.segment(k);
                    let (mut codes, mut frames) = (Vec::new(), 0);
                    loop {
                        match rx.recv().await {
                            Some(StreamChunk::Token { id, .. }) => {
                                if let Some(code) = c.code_of(codes.len(), id) {
                                    codes.push(code);
                                    if codes.len() % fc == 0 {
                                        frames += 1;
                                        lm.frame();
                                        if ftx
                                            .send(SegMsg::Frame(codes[codes.len() - fc..].to_vec()))
                                            .is_err()
                                        {
                                            return;
                                        }
                                    }
                                }
                                if lm_cut.load(SeqCst) == tag {
                                    break;
                                }
                            }
                            Some(StreamChunk::Done { .. }) => break,
                            Some(StreamChunk::Err(e)) => {
                                tracing::warn!(error = %e, "tts: stream LM error");
                                return drop(ftx.send(SegMsg::Err(e.to_string())));
                            }
                            None => {
                                return drop(ftx.send(SegMsg::Err(
                                    "stream LM ended before its segment".into(),
                                )))
                            }
                        }
                    }
                    drop(rx);
                    lm.segment_done();
                    if n_segs == 1 {
                        if frames == 0 {
                            return drop(
                                ftx.send(SegMsg::Err("model produced no audio frames".into())),
                            );
                        }
                        return drop(ftx.send(SegMsg::End));
                    }
                    if ftx.send(SegMsg::End).is_err() {
                        return;
                    }
                    let Some(failed) = verdict_for(&mut vrx, tag).await else {
                        return;
                    };
                    match later.judge(k, frames, failed, &mut tries) {
                        Verdict::Keep => break,
                        Verdict::Retry => {
                            attempt += 1;
                            if ftx.send(SegMsg::Retry).is_err() {
                                return;
                            }
                        }
                        Verdict::Fail => {
                            return drop(
                                ftx.send(SegMsg::Err(format!("segment {k} produced no speech"))),
                            )
                        }
                    }
                }
                k += 1;
            }
        };
        drain.await;
        lm.lm_done();
    });
    let (mut frames, mut emitted, mut first) = (Vec::<i32>::new(), 0usize, None);
    // Segment and attempt, frames the earlier segments emitted, and whether this attempt's verdict
    // is out (the next message is a retry, the next segment, or the end).
    let (mut seg, mut attempt, mut total, mut joins, mut judged) =
        (0usize, 0usize, 0usize, Vec::new(), false);
    // Samples sent and when the first went out (the client's playback clock).
    let (mut sent, mut playing): (usize, Option<Instant>) = (0, None);
    let mut joiner = super::Joins::new(model.contract.sample_rate);
    // Dropping the rest of the current attempt (cut short).
    let mut skipping = false;
    let mut first_frame_at: Option<Instant> = None;
    loop {
        let mut done = false;
        let msg = frx.recv().await;
        if judged {
            judged = false;
            match msg {
                None => break,
                Some(SegMsg::Retry) => {
                    (frames, emitted, skipping) = (Vec::new(), 0, false);
                    attempt += 1;
                    joiner.next_segment();
                    continue;
                }
                Some(SegMsg::Err(_)) => {}
                Some(_) => {
                    total += emitted;
                    joins.push(sent);
                    joiner.next_segment();
                    (frames, emitted, skipping) = (Vec::new(), 0, false);
                    (seg, attempt) = (seg + 1, 0);
                }
            }
        }
        match msg {
            None => done = true,
            Some(SegMsg::End) => done = true,
            Some(SegMsg::Err(e)) => return drop(out.send(Err(std::io::Error::other(e))).await),
            Some(SegMsg::Retry) => {}
            Some(SegMsg::Frame(f)) => {
                frames.extend(f);
                while let Ok(m) = frx.try_recv() {
                    match m {
                        SegMsg::Frame(f) => frames.extend(f),
                        SegMsg::End => {
                            done = true;
                            break;
                        }
                        SegMsg::Retry => {}
                        SegMsg::Err(e) => {
                            return drop(out.send(Err(std::io::Error::other(e))).await)
                        }
                    }
                }
                first_frame_at.get_or_insert_with(Instant::now);
            }
        }
        let first_audio = first.is_none() && emitted == 0;
        let chunk = (model.codec.min_frames + 1)
            .saturating_sub(model.codec.window)
            .max(1);
        let lookahead = if emitted == 0 {
            model
                .codec
                .lookahead
                .min(crate::config::RuntimeConfig::get().tts_first_lookahead)
        } else {
            model.codec.lookahead
        };
        let n = frames.len() / fc;
        // A short first lookahead buys time to first audio, but the next window needs the full
        // lookahead: hold the first audio until it covers producing those frames at the measured
        // frame rate (a stream near real time otherwise underruns right after its first chunk).
        if first_audio && !done && lookahead < model.codec.lookahead {
            let Some(t0) = first_frame_at.filter(|_| n >= 2) else {
                continue;
            };
            let per_frame = t0.elapsed().as_secs_f64() / (n - 1) as f64;
            let audio_per_frame = fs as f64 / sr;
            let gap = (chunk + model.codec.lookahead - lookahead) as f64;
            let need = ((gap * per_frame * 1.1) / audio_per_frame).ceil() as usize;
            let mut need = need.clamp(1, chunk + model.codec.lookahead);
            // A later segment's first window waits for its prefill and frames: bank that too.
            if n_segs > 1 {
                need += model.codec.lookahead + 2;
            }
            // Slower than real time (a wide batch): also bank the audio the rest of the utterance
            // falls behind by at this rate, holding at most MAX_HOLD_S.
            let deficit = ticket.remaining_frames() * (per_frame - audio_per_frame);
            if deficit > 0.0 && t0.elapsed().as_secs_f64() < super::realtime::MAX_HOLD_S {
                need = need.max((deficit / audio_per_frame).ceil() as usize);
            }
            if n.saturating_sub(lookahead) < need {
                continue;
            }
        }
        if let Some((s, e, upto)) =
            stream_step(n, emitted, done, model.codec.window, lookahead, chunk)
                .filter(|_| !skipping)
        {
            let window = frames[s * fc..e * fc].to_vec();
            let urgency = if first_audio {
                Urgency::First
            } else {
                Urgency::Stream
            };
            let wseed = seed ^ segment_salt(seg) ^ (emitted as u64).wrapping_mul(0x9E37_79B9);
            match model.codec.decode(window, e - s, wseed, urgency).await {
                Ok(pcm) => {
                    let Some(fresh) = pcm.get((emitted - s) * fs..(upto - s) * fs) else {
                        return drop(
                            out.send(Err(std::io::Error::other(short_window(pcm.len(), e - s))))
                                .await,
                        );
                    };
                    let joined;
                    let fresh = if n_segs == 1 {
                        fresh
                    } else {
                        let lead = playing.map_or(f64::INFINITY, |t| {
                            sent as f64 / sr - t.elapsed().as_secs_f64()
                        });
                        let mut v = Vec::with_capacity(fresh.len());
                        joiner.push(fresh, lead, &mut v);
                        if joiner.runaway() || joiner.mute() || joiner.drone() {
                            cut.store(attempt_tag(seg, attempt), SeqCst);
                            skipping = true;
                        }
                        joined = v;
                        &joined[..]
                    };
                    emitted = upto;
                    if !fresh.is_empty() {
                        let mut bytes = Vec::new();
                        pcm16(fresh, &mut bytes);
                        if first.is_none() {
                            ticket.first_audio();
                        }
                        first.get_or_insert_with(|| t_arrive.elapsed());
                        playing.get_or_insert_with(Instant::now);
                        sent += fresh.len();
                        ticket.sent(sent as f64 / sr);
                        run.audio(fresh.len(), sr);
                        if let Some(h) = head.take() {
                            let _ = h.send(run.headers());
                        }
                        if out.send(Ok(bytes)).await.is_err() {
                            return; // client gone: dropping frx ends the drain, which cancels the slot
                        }
                    }
                }
                Err(e) => return drop(out.send(Err(std::io::Error::other(e))).await),
            }
        }
        if done {
            if n_segs == 1 {
                break;
            }
            // The drain waits on this: keep the attempt, generate it again, or fail the request.
            if vtx
                .send((attempt_tag(seg, attempt), joiner.segment_failed()))
                .is_err()
            {
                break;
            }
            judged = true;
        }
    }
    run.done();
    tracing::info!(
        frames = total + emitted,
        segments = n_segs,
        ?joins,
        ttfa_ms = first.map(|d| d.as_secs_f64() * 1e3),
        total_ms = t_arrive.elapsed().as_secs_f64() * 1e3,
        "tts: stream"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: usize = 6;
    const LOOKAHEAD: usize = 2;

    /// A segment attempt without frames or speech is generated again, then fails the request:
    /// never kept (a silent 200 missing part of the text).
    #[test]
    fn a_segment_without_speech_retries_then_fails() {
        assert_eq!(verdict(40, false, 0), Verdict::Keep);
        assert_eq!(verdict(0, false, 0), Verdict::Retry);
        assert_eq!(verdict(40, true, MUTE_RETRIES - 1), Verdict::Retry);
        assert_eq!(verdict(0, false, MUTE_RETRIES), Verdict::Fail);
        assert_eq!(verdict(40, true, MUTE_RETRIES), Verdict::Fail);
        assert_eq!(verdict(40, false, MUTE_RETRIES), Verdict::Keep);
        let t = "As for etchings they are of two kinds british and foreign, he laments most bitterly the divorce";
        let h = split_halves(t).unwrap();
        assert_eq!(h.len(), 2);
        assert_eq!(h.join(" "), t);
        assert!(split_halves("Too short to split again.").is_none());
    }

    /// Cuts and verdicts carry their segment attempt: a stale one (another segment, or an
    /// earlier attempt of this one) never acts on the current attempt.
    #[tokio::test(flavor = "current_thread")]
    async fn segment_signals_act_only_on_their_attempt() {
        let tags = [
            attempt_tag(0, 0),
            attempt_tag(0, 1),
            attempt_tag(1, 0),
            attempt_tag(1, 1),
        ];
        for (i, a) in tags.iter().enumerate() {
            assert!(*a != 0 && tags.iter().skip(i + 1).all(|b| b != a));
        }
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        // A verdict for segment 0 arrives while the drain waits on segment 1: ignored.
        tx.send((attempt_tag(0, 0), true)).unwrap();
        tx.send((attempt_tag(1, 0), false)).unwrap();
        assert_eq!(verdict_for(&mut rx, attempt_tag(1, 0)).await, Some(false));
        tx.send((attempt_tag(1, 0), true)).unwrap();
        tx.send((attempt_tag(1, 1), true)).unwrap();
        assert_eq!(verdict_for(&mut rx, attempt_tag(1, 1)).await, Some(true));
        drop(tx);
        assert_eq!(verdict_for(&mut rx, attempt_tag(2, 0)).await, None);
    }

    /// A request the guided pipeline cannot take is the client's error, not a server fault.
    #[test]
    fn guided_speech_failures_map_to_their_status() {
        use super::super::guided_speech::{SpeechError, QUEUE_FULL};
        let invalid = speech_failure(SpeechError::Invalid("unknown voice \"x\"".into(), "voice"));
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            speech_failure(SpeechError::Failed(QUEUE_FULL.into())).status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            speech_failure(SpeechError::Failed("render failed".into())).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        let long = crate::RuntimeError::ContextLength(
            "prompt + max_tokens = 1031 exceeds the compiled context 1024".into(),
        );
        assert_eq!(
            speech_failure(SpeechError::Failed(long.to_string())).status(),
            StatusCode::BAD_REQUEST
        );
    }

    /// Every frame is emitted exactly once, in order, each with LOOKAHEAD right context until
    /// the final flush and the one-frame window's left context; later windows emit CHUNK frames
    /// and fit the smallest capacity (WINDOW + CHUNK - 1 frames).
    #[test]
    fn stream_plan_covers_each_frame_once() {
        for chunk in 1..=3 {
            for total in 1..20 {
                let (mut emitted, mut seen) = (0, Vec::new());
                for n in 1..=total {
                    if let Some((s, e, upto)) =
                        stream_step(n, emitted, false, WINDOW, LOOKAHEAD, chunk)
                    {
                        assert!(s <= emitted && upto <= e && e <= n && e - s <= WINDOW + chunk - 1);
                        assert!(emitted - s == emitted.min(WINDOW - LOOKAHEAD - 1));
                        assert!(e - upto >= LOOKAHEAD.min(n - upto));
                        assert!(emitted == 0 || upto - emitted == chunk);
                        seen.extend(emitted..upto);
                        emitted = upto;
                    }
                }
                if let Some((_, e, upto)) =
                    stream_step(total, emitted, true, WINDOW, LOOKAHEAD, chunk)
                {
                    assert_eq!((e, upto), (total, total));
                    seen.extend(emitted..upto);
                }
                assert_eq!(seen, (0..total).collect::<Vec<_>>());
            }
        }
    }
}
