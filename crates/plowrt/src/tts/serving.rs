//! OpenAI `POST /v1/audio/speech` on `plowrt serve`.
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

use super::codec::{Codec, Urgency};
use super::{pcm16, wav_header, SpeechContract};
use crate::serve::session::{InFlight, RequestIds};
use crate::serve::stream::{self as stream_mod, StreamChunk};
use crate::serve::AppState;


#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    /// Speaker; must be a `<spk_{voice}>` vocabulary token of the model.
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
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// ISO 639-1 code of the input's language for multilingual models (e.g. `zh`); default is the
    /// packet's `text.default_language`. Models without language selection reject it.
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

/// Guided speech front-ends (`tts.guided_lm.v1`) by asset directory, bound on first use.
fn guided_models() -> &'static Mutex<HashMap<PathBuf, Option<Arc<super::guided_speech::GuidedSpeech>>>> {
    static M: OnceLock<Mutex<HashMap<PathBuf, Option<Arc<super::guided_speech::GuidedSpeech>>>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

fn guided_model(
    assets: &Path,
    credit: Arc<crate::sched::admission::DownstreamCredit>,
) -> Result<Option<Arc<super::guided_speech::GuidedSpeech>>, String> {
    if let Some(m) = guided_models().lock().get(assets) {
        return Ok(m.clone());
    }
    let model = if super::guided_lm::GuidedLmContract::load(assets).map_err(|e| e.to_string())?.is_some() {
        let g = super::guided_speech::GuidedSpeech::start(assets, credit).map_err(|e| e.to_string())?;
        tracing::info!(dir = %assets.display(), "tts: guided speech pipeline bound to the serve mux");
        Some(Arc::new(g))
    } else {
        None
    };
    guided_models().lock().insert(assets.to_path_buf(), model.clone());
    Ok(model)
}

/// Bind every served speech model's host stages now (vocoder / codec graphs, prompt tables), so
/// the first request does not pay them.
pub fn preload(state: &AppState) {
    for slug in state.registry.slugs() {
        let (Some(_), Ok(bundle)) = (state.mux(&slug), state.registry.get(&slug)) else {
            continue;
        };
        let bound = guided_model(&bundle.dir, state.downstream(&slug)).and_then(|g| match g {
            Some(_) => Ok(()),
            None => speech_model(&bundle.dir, state.downstream(&slug)).map(drop),
        });
        if let Err(e) = bound {
            tracing::warn!(%slug, error = %e, "tts: speech pipeline failed to bind");
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn speech_on_guided(
    w: Arc<super::guided_speech::GuidedSpeech>,
    mux: crate::serve::mux::ModelMux,
    req: SpeechRequest,
    t_arrive: Instant,
    ids: &RequestIds,
    in_flight: InFlight,
    report: Option<crate::serve::session::Report>,
    report_rx: Option<tokio::sync::oneshot::Receiver<crate::serve::session::CacheOutcome>>,
    mut run: crate::serve::turns::StageRun,
) -> Response {
    let wav = match req.response_format.as_deref().unwrap_or("wav") {
        "wav" => true,
        "pcm" => false,
        f => return bad(format!("response_format {f:?} unsupported; use wav or pcm"), "response_format"),
    };
    if req.input.trim().is_empty() {
        return bad("`input` is empty", "input");
    }
    let seed = req.seed.unwrap_or_else(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
    });
    let lang = match w.language(req.language.as_deref()) {
        Ok(l) => l,
        Err(e) => return bad(e, "language"),
    };
    if req.stream {
        let mut ev = match w.synthesize_stream(&mux, req.voice.clone(), req.input.clone(), lang.as_deref(), seed, ids, report) {
            Ok(rx) => rx,
            Err(e) => return server_error(e),
        };
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, std::io::Error>>(64);
        if wav {
            let _ = out_tx.try_send(Ok(wav_header(w.sample_rate, u32::MAX)));
        }
        let sr = f64::from(w.sample_rate);
        // The headers go out with the first audio, so its Server-Timing is in them; the first body
        // byte is no later for it.
        let first_event = ev.recv().await;
        let cache = crate::serve::session::CacheOutcome::received(report_rx).await;
        run.admitted(cache.and_then(|c| c.at));
        if let Some(super::guided_speech::StreamEvent::Pcm(p)) = &first_event {
            run.audio(p.len(), sr);
        }
        let stamped = run.headers();
        tokio::spawn(async move {
            let (mut samples, mut first) = (0usize, None);
            let mut next = first_event;
            while let Some(e) = next.take() {
                match e {
                    super::guided_speech::StreamEvent::Pcm(p) => {
                        if first.is_some() {
                            run.audio(p.len(), sr);
                        }
                        first.get_or_insert_with(|| t_arrive.elapsed());
                        samples += p.len();
                        let mut bytes = Vec::with_capacity(p.len() * 2);
                        pcm16(&p, &mut bytes);
                        if out_tx.send(Ok(bytes)).await.is_err() {
                            return;
                        }
                    }
                    super::guided_speech::StreamEvent::Done { tokens, t3_ms, s3gen_ms } => {
                        run.done();
                        let total = t_arrive.elapsed().as_secs_f64();
                        let audio_s = samples as f64 / sr;
                        tracing::info!(tokens, audio_s, t3_ms, s3gen_ms, ttfa_ms = first.map(|d| d.as_secs_f64() * 1e3), total_ms = total * 1e3, rtf = total / audio_s, "tts: chatterbox stream");
                        return;
                    }
                    super::guided_speech::StreamEvent::Err(e) => return drop(out_tx.send(Err(std::io::Error::other(e))).await),
                }
                next = ev.recv().await;
            }
        });
        let body = Body::from_stream(futures::stream::poll_fn(move |cx| {
            let _held = &in_flight;
            out_rx.poll_recv(cx)
        }));
        let ct = if wav { "audio/wav" } else { "audio/pcm" };
        let mut response = ([(header::CONTENT_TYPE, ct)], body).into_response();
        response.headers_mut().extend(stamped);
        if let Some(cache) = cache {
            cache.stamp(&mut response);
        }
        return response;
    }
    let result = w.synthesize(&mux, req.voice.clone(), req.input.clone(), lang.as_deref(), seed, ids, report).await;
    let cache = crate::serve::session::CacheOutcome::received(report_rx).await;
    run.admitted(cache.and_then(|c| c.at));
    if result.is_ok() {
        run.first();
        run.done();
    }
    let mut response = match result {
        Err(e) => server_error(e),
        Ok(a) => {
            let audio_s = a.pcm.len() as f64 / f64::from(w.sample_rate);
            let mut out = if wav { wav_header(w.sample_rate, (a.pcm.len() * 2) as u32) } else { Vec::new() };
            pcm16(&a.pcm, &mut out);
            let total = t_arrive.elapsed().as_secs_f64();
            tracing::info!(tokens = a.tokens, audio_s, t3_ms = a.t3_ms, s3gen_ms = a.s3gen_ms, total_ms = total * 1e3, rtf = total / audio_s, "tts: chatterbox speech");
            let ct = if wav { "audio/wav" } else { "audio/pcm" };
            ([(header::CONTENT_TYPE, ct.to_string()), (HeaderName::from_static("x-plow-audio-seconds"), format!("{audio_s:.3}"))], out)
                .into_response()
        }
    };
    run.stamp(&mut response);
    if let (Some(cache), true) = (cache, response.status().is_success()) {
        cache.stamp(&mut response);
    }
    response
}

pub struct SpeechModel {
    pub contract: SpeechContract,
    codec: Codec,
}

/// Speech models by asset directory, bound on first use.
fn speech_models() -> &'static Mutex<HashMap<PathBuf, Option<Arc<SpeechModel>>>> {
    static M: OnceLock<Mutex<HashMap<PathBuf, Option<Arc<SpeechModel>>>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

/// `credit` is the serving model's downstream credit; the codec's backlog gates its admission.
pub fn speech_model(
    assets: &Path,
    credit: Arc<crate::sched::admission::DownstreamCredit>,
) -> Result<Option<Arc<SpeechModel>>, String> {
    if let Some(m) = speech_models().lock().get(assets) {
        return Ok(m.clone());
    }
    let model = match SpeechContract::load(assets).map_err(|e| e.to_string())? {
        None => None,
        Some(contract) => {
            let mut codec = Codec::load(assets)?;
            codec.couple(credit);
            if codec.frame_codes != contract.frame_codes || codec.frame_samples != contract.frame_samples {
                return Err(format!(
                    "codec packet frames ({} codes, {} samples) disagree with the speech contract ({}, {})",
                    codec.frame_codes, codec.frame_samples, contract.frame_codes, contract.frame_samples
                ));
            }
            tracing::info!(pipeline = %contract.pipeline, sample_rate = contract.sample_rate, "tts: speech pipeline bound");
            Some(Arc::new(SpeechModel { contract, codec }))
        }
    };
    speech_models().lock().insert(assets.to_path_buf(), model.clone());
    Ok(model)
}

fn bad(msg: impl Into<String>, param: &str) -> Response {
    crate::serve::api_error(StatusCode::BAD_REQUEST, msg, "invalid_request_error", Some("invalid_value"), Some(param.into()))
}

fn server_error(msg: impl Into<String>) -> Response {
    let msg = msg.into();
    if msg == super::guided_speech::QUEUE_FULL {
        return crate::serve::api_error(StatusCode::TOO_MANY_REQUESTS, msg, "rate_limit_error", Some("server_overloaded"), None);
    }
    crate::serve::api_error(StatusCode::INTERNAL_SERVER_ERROR, msg, "server_error", None, None)
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
    if let Err(e) = req.as_ref().map_or(Ok(()), |Json(r)| ids.apply_body(&r.route())) {
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
        Err(e) => return crate::serve::api_error(e.status(), e.body_text(), "invalid_request_error", Some("invalid_json"), None),
    };
    // The same ranges chat and completions enforce: `repetition_penalty: 0` divided the logits
    // by zero and sampled from NaN probabilities.
    let sampling = crate::serve::openai::SamplingFields {
        temperature: req.temperature,
        top_p: req.top_p,
        repetition_penalty: req.repetition_penalty,
        ..Default::default()
    };
    if let Err(e) = sampling.validate() {
        return bad(e.message, e.field);
    }
    let t_arrive = Instant::now();
    if let Some(canonical) = state.registry.resolve(&req.model) {
        req.model = canonical;
    }
    if let Some(mgr) = state.manager_for(&req.model) {
        if mgr.manages(&req.model) {
            if let Err(e) = mgr.ensure_resident(&req.model).await {
                return crate::serve::api_error(StatusCode::SERVICE_UNAVAILABLE, e.to_string(), "server_error", None, None);
            }
        }
    }
    let (Some(mux), Ok(bundle)) = (state.mux(&req.model), state.registry.get(&req.model)) else {
        return crate::serve::api_error(
            StatusCode::NOT_FOUND,
            format!("no model registered for '{}'.", req.model),
            "invalid_request_error",
            Some("model_not_found"),
            Some("model".into()),
        );
    };
    let Some(in_flight) = ids.begin(&req.model) else {
        return crate::serve::api_error(StatusCode::CONFLICT, format!("request {} is already in flight in this session", ids.request), "invalid_request_error", Some("duplicate_request_id"), None);
    };
    let mut run = crate::serve::turns::StageRun::start(
        ids,
        crate::serve::turns::Kind::Tts,
        &req.model,
        Some(state.model_metrics(&req.model)),
        t_arrive,
        true,
    );
    let ids = &RequestIds { turn_key: run.key(), ..ids.clone() };
    match tokio::task::block_in_place(|| guided_model(&bundle.dir, state.downstream(&req.model))) {
        Ok(Some(g)) => return speech_on_guided(g, mux, req, t_arrive, ids, in_flight, report, report_rx.take(), run).await,
        Ok(None) => {}
        Err(e) => return server_error(format!("speech pipeline: {e}")),
    }
    let model = match tokio::task::block_in_place(|| speech_model(&bundle.dir, state.downstream(&req.model))) {
        Ok(Some(m)) => m,
        Ok(None) => return bad(format!("model '{}' declares no speech pipeline", req.model), "model"),
        Err(e) => return server_error(format!("speech pipeline: {e}")),
    };
    let c = &model.contract;
    if req.input.trim().is_empty() {
        return bad("`input` is empty", "input");
    }
    if req.speed.is_some_and(|s| s != 1.0) {
        return bad("only speed 1.0 is supported", "speed");
    }
    let tok = bundle.tokenizer();
    let voice = req.voice.clone();
    if tok.encode_with_special_tokens(&c.voice_token(&voice), false).len() != 1 {
        return bad(format!("unknown voice {voice:?}: {} is not a vocabulary token", c.voice_token(&voice)), "voice");
    }
    let wav = match req.response_format.as_deref().unwrap_or("wav") {
        "wav" => true,
        "pcm" => false,
        f => return bad(format!("response_format {f:?} unsupported; use wav or pcm"), "response_format"),
    };
    let mut prompt_ids = c.prefix.clone();
    prompt_ids.extend(tok.encode_with_special_tokens(&c.prompt_text(&voice, &req.input), false));
    prompt_ids.extend_from_slice(&c.suffix);

    let mut gen = crate::serve::GenParams::default();
    gen.max_tokens = req.max_tokens.unwrap_or_else(|| c.max_new_tokens(&req.input)).min(c.max_new_tokens_cap);
    gen.params.temperature = req.temperature.unwrap_or(c.temperature);
    gen.params.top_p = req.top_p.unwrap_or(c.top_p);
    gen.params.repetition_penalty = req.repetition_penalty.unwrap_or(1.0);
    gen.seed = req.seed;
    gen.stop_token_ids = c.stops.clone();

    let (tx, rx) = stream_mod::channel();
    let opts = crate::serve::mux::JobOpts {
        class: if req.stream { crate::serve::mux::JobClass::Critical } else { crate::serve::mux::JobClass::Normal },
        raw_tokens: true,
        speech: None,
        session: ids.session.as_ref().and_then(|_| ids.ticket(crate::serve::session::row_keys(&prompt_ids, &[], &[]), report)),
        turn: ids.turn_key.clone(),
        continuing: run.continuing(),
    };
    let job = crate::serve::mux::Job { prompt_ids, gen, arrived: Instant::now(), respond: tx, opts };
    if let Err(err) = mux.submit_arrived(job, t_arrive, Some(mux.ingress())) {
        return match err {
            crate::serve::mux::SubmitError::Full(_) => {
                crate::serve::api_error(StatusCode::TOO_MANY_REQUESTS, "model request queue full", "rate_limit_error", Some("server_overloaded"), None)
            }
            crate::serve::mux::SubmitError::Closed(_) => {
                crate::serve::api_error(StatusCode::SERVICE_UNAVAILABLE, "model dispatcher unavailable", "server_error", None, None)
            }
        };
    }
    let seed = req.seed.unwrap_or_else(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
    }) | 1;
    let content_type = if wav { "audio/wav" } else { "audio/pcm" };
    if req.stream {
        let (out_tx, out_rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, std::io::Error>>(64);
        if wav {
            let _ = out_tx.try_send(Ok(wav_header(c.sample_rate, u32::MAX)));
        }
        let cache = crate::serve::session::CacheOutcome::received(report_rx.take()).await;
        run.admitted(cache.and_then(|c| c.at));
        let stamped = run.headers();
        tokio::spawn(stream_task(Arc::clone(&model), rx, out_tx, seed, t_arrive, run));
        let mut out_rx = out_rx;
        let body = Body::from_stream(futures::stream::poll_fn(move |cx| {
            let _held = &in_flight;
            out_rx.poll_recv(cx)
        }));
        let mut response = ([(header::CONTENT_TYPE, content_type)], body).into_response();
        response.headers_mut().extend(stamped);
        if let Some(cache) = cache {
            cache.stamp(&mut response);
        }
        return response;
    }
    let (codes, n_tokens) = match collect_codes(c, rx).await {
        Ok(v) => v,
        Err(e) => return crate::serve::api_error_for(&e),
    };
    let frames = codes.len() / c.frame_codes;
    if frames == 0 {
        return server_error("model produced no audio frames");
    }
    let t_lm = t_arrive.elapsed();
    let pcm = match decode_all(&model, &codes[..frames * c.frame_codes], frames, seed).await {
        Ok(p) => p,
        Err(e) => return server_error(e),
    };
    let mut out = if wav { wav_header(c.sample_rate, (pcm.len() * 2) as u32) } else { Vec::new() };
    pcm16(&pcm, &mut out);
    let audio_s = pcm.len() as f64 / f64::from(c.sample_rate);
    let total = t_arrive.elapsed().as_secs_f64();
    tracing::info!(tokens = n_tokens, frames, audio_s, lm_ms = t_lm.as_secs_f64() * 1e3, total_ms = total * 1e3, rtf = total / audio_s, "tts: speech");
    run.first();
    run.done();
    let mut response = (
        [
            (header::CONTENT_TYPE, content_type.to_string()),
            (HeaderName::from_static("x-plow-audio-seconds"), format!("{audio_s:.3}")),
            (HeaderName::from_static("x-plow-lm-ms"), format!("{:.1}", t_lm.as_secs_f64() * 1e3)),
        ],
        out,
    )
        .into_response();
    run.stamp(&mut response);
    response
}

/// Whole utterance; beyond the codec's frame capacity, windows with the codec's context frames
/// on each side keep their centres.
async fn decode_all(model: &SpeechModel, codes: &[i32], frames: usize, seed: u64) -> Result<Vec<f32>, String> {
    let (fc, fs, max) = (model.contract.frame_codes, model.contract.frame_samples, model.codec.max_frames);
    if frames <= max {
        return model.codec.decode(codes.to_vec(), frames, seed, Urgency::Whole).await;
    }
    let window = model.codec.window;
    let step = max - 2 * window;
    let mut pcm = Vec::with_capacity(frames * fs);
    let mut s = 0;
    while s < frames {
        let e = (s + step).min(frames);
        let (ws, we) = (s.saturating_sub(window), (e + window).min(frames));
        let w = model.codec.decode(codes[ws * fc..we * fc].to_vec(), we - ws, seed ^ s as u64, Urgency::Whole).await?;
        pcm.extend_from_slice(&w[(s - ws) * fs..(e - ws) * fs]);
        s = e;
    }
    Ok(pcm)
}

/// Drain the LM stream promptly (the mux cuts a consumer that falls behind) keeping the codes.
async fn collect_codes(c: &SpeechContract, mut rx: stream_mod::ChunkReceiver) -> crate::Result<(Vec<i32>, usize)> {
    let (mut codes, mut n) = (Vec::new(), 0);
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { id, .. } => {
                n += 1;
                if let Some(code) = c.code_of(codes.len(), id) {
                    codes.push(code);
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
fn stream_step(n: usize, emitted: usize, done: bool, window: usize, lookahead: usize, chunk: usize) -> Option<(usize, usize, usize)> {
    let upto = if done { n } else { n.saturating_sub(lookahead) };
    let ready = upto > emitted && (done || emitted == 0 || upto - emitted >= chunk);
    ready.then(|| (emitted.saturating_sub(window.saturating_sub(lookahead + 1)), n.min(upto + lookahead), upto))
}

async fn stream_task(
    model: Arc<SpeechModel>,
    mut rx: stream_mod::ChunkReceiver,
    out: tokio::sync::mpsc::Sender<Result<Vec<u8>, std::io::Error>>,
    seed: u64,
    t_arrive: Instant,
    mut run: crate::serve::turns::StageRun,
) {
    let c = model.contract.clone();
    let sr = f64::from(c.sample_rate);
    let (fc, fs) = (c.frame_codes, c.frame_samples);
    let (ftx, mut frx) = tokio::sync::mpsc::unbounded_channel::<Vec<i32>>();
    // The LM drain never waits on the codec.
    tokio::spawn(async move {
        let mut codes = Vec::new();
        while let Some(chunk) = rx.recv().await {
            match chunk {
                StreamChunk::Token { id, .. } => {
                    if let Some(code) = c.code_of(codes.len(), id) {
                        codes.push(code);
                        if codes.len() % fc == 0 && ftx.send(codes[codes.len() - fc..].to_vec()).is_err() {
                            return;
                        }
                    }
                }
                StreamChunk::Done { .. } => return,
                StreamChunk::Err(e) => return tracing::warn!(error = %e, "tts: stream LM error"),
            }
        }
    });
    let (mut frames, mut emitted, mut first) = (Vec::<i32>::new(), 0usize, None);
    loop {
        let next = frx.recv().await;
        let done = next.is_none();
        if let Some(f) = next {
            frames.extend(f);
            while let Ok(f) = frx.try_recv() {
                frames.extend(f);
            }
        }
        let chunk = (model.codec.min_frames + 1).saturating_sub(model.codec.window).max(1);
        let lookahead = if emitted == 0 {
            model.codec.lookahead.min(crate::config::RuntimeConfig::get().tts_first_lookahead)
        } else {
            model.codec.lookahead
        };
        if let Some((s, e, upto)) = stream_step(frames.len() / fc, emitted, done, model.codec.window, lookahead, chunk) {
            let window = frames[s * fc..e * fc].to_vec();
            let urgency = if emitted == 0 { Urgency::First } else { Urgency::Stream };
            match model.codec.decode(window, e - s, seed ^ (emitted as u64).wrapping_mul(0x9E37_79B9), urgency).await {
                Ok(pcm) => {
                    let mut bytes = Vec::new();
                    pcm16(&pcm[(emitted - s) * fs..(upto - s) * fs], &mut bytes);
                    first.get_or_insert_with(|| t_arrive.elapsed());
                    run.audio((upto - emitted) * fs, sr);
                    if out.send(Ok(bytes)).await.is_err() {
                        return; // client gone: dropping frx ends the drain, which cancels the slot
                    }
                    emitted = upto;
                }
                Err(e) => return drop(out.send(Err(std::io::Error::other(e))).await),
            }
        }
        if done {
            break;
        }
    }
    run.done();
    tracing::info!(
        frames = emitted,
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

    /// Every frame is emitted exactly once, in order, each with LOOKAHEAD right context until
    /// the final flush and the one-frame window's left context; later windows emit CHUNK frames
    /// and fit the smallest capacity (WINDOW + CHUNK - 1 frames).
    #[test]
    fn stream_plan_covers_each_frame_once() {
        for chunk in 1..=3 {
            for total in 1..20 {
                let (mut emitted, mut seen) = (0, Vec::new());
                for n in 1..=total {
                    if let Some((s, e, upto)) = stream_step(n, emitted, false, WINDOW, LOOKAHEAD, chunk) {
                        assert!(s <= emitted && upto <= e && e <= n && e - s <= WINDOW + chunk - 1);
                        assert!(emitted - s == emitted.min(WINDOW - LOOKAHEAD - 1));
                        assert!(e - upto >= LOOKAHEAD.min(n - upto));
                        assert!(emitted == 0 || upto - emitted == chunk);
                        seen.extend(emitted..upto);
                        emitted = upto;
                    }
                }
                if let Some((_, e, upto)) = stream_step(total, emitted, true, WINDOW, LOOKAHEAD, chunk) {
                    assert_eq!((e, upto), (total, total));
                    seen.extend(emitted..upto);
                }
                assert_eq!(seen, (0..total).collect::<Vec<_>>());
            }
        }
    }
}
