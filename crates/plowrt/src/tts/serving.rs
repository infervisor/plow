//! OpenAI `POST /v1/audio/speech` on `plowrt serve`.
//!
//! A model serves speech when its packet declares a `tts.codec_lm.v1` pipeline. The LM stage is
//! submitted to that model's continuous-batching mux like a completion; the codec stage runs on
//! the model's [`Codec`] worker. `stream: true` returns chunked audio: each new frame decodes a
//! window of `WINDOW` frames and emits the frames that have `LOOKAHEAD` frames of right context.

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

use super::codec::Codec;
use super::{pcm16, wav_header, SpeechContract};
use crate::serve::stream::{self as stream_mod, StreamChunk};
use crate::serve::AppState;

const WINDOW: usize = 6;
const LOOKAHEAD: usize = 2;
const CODEC_MAX_BATCH: usize = 64;
const CODEC_MAX_FRAMES: usize = 128;

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
}

/// Speech pipelines that own their engine (`tts.t3_cfg.v1`), by served model name.
fn workers() -> &'static Mutex<HashMap<String, Arc<super::chatterbox::ChatterboxWorker>>> {
    static W: OnceLock<Mutex<HashMap<String, Arc<super::chatterbox::ChatterboxWorker>>>> = OnceLock::new();
    W.get_or_init(Default::default)
}

/// Split `plowrt serve --assets` into text-engine assets and self-hosted speech pipelines. Each
/// `tts.t3_cfg.v1` asset starts a Chatterbox worker (its own engine on `device`) served under
/// the directory name; the rest go to the text registry unchanged.
pub fn start_speech_workers(assets: Vec<PathBuf>, device: u8) -> crate::Result<Vec<PathBuf>> {
    let mut text = Vec::new();
    for dir in assets {
        if super::t3::T3Contract::load(&dir)?.is_none() {
            text.push(dir);
            continue;
        }
        let name = dir.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let w = super::chatterbox::ChatterboxWorker::start(&dir, device)?;
        tracing::info!(model = %name, dir = %dir.display(), "tts: chatterbox speech pipeline ready");
        workers().lock().insert(name, Arc::new(w));
    }
    Ok(text)
}

async fn speech_on_worker(w: Arc<super::chatterbox::ChatterboxWorker>, req: SpeechRequest, t_arrive: Instant) -> Response {
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
    if req.stream {
        let mut ev = match w.synthesize_stream(req.voice.clone(), req.input.clone(), seed) {
            Ok(rx) => rx,
            Err(e) => return server_error(e),
        };
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, std::io::Error>>(64);
        if wav {
            let _ = out_tx.try_send(Ok(wav_header(w.sample_rate, u32::MAX)));
        }
        let sr = f64::from(w.sample_rate);
        tokio::spawn(async move {
            let (mut samples, mut first) = (0usize, None);
            while let Some(e) = ev.recv().await {
                match e {
                    super::chatterbox::StreamEvent::Pcm(p) => {
                        first.get_or_insert_with(|| t_arrive.elapsed());
                        samples += p.len();
                        let mut bytes = Vec::with_capacity(p.len() * 2);
                        pcm16(&p, &mut bytes);
                        if out_tx.send(Ok(bytes)).await.is_err() {
                            return;
                        }
                    }
                    super::chatterbox::StreamEvent::Done { tokens, t3_ms, s3gen_ms } => {
                        let total = t_arrive.elapsed().as_secs_f64();
                        let audio_s = samples as f64 / sr;
                        tracing::info!(tokens, audio_s, t3_ms, s3gen_ms, ttfa_ms = first.map(|d| d.as_secs_f64() * 1e3), total_ms = total * 1e3, rtf = total / audio_s, "tts: chatterbox stream");
                        return;
                    }
                    super::chatterbox::StreamEvent::Err(e) => return drop(out_tx.send(Err(std::io::Error::other(e))).await),
                }
            }
        });
        let body = Body::from_stream(futures::stream::poll_fn(move |cx| out_rx.poll_recv(cx)));
        let ct = if wav { "audio/wav" } else { "audio/pcm" };
        return ([(header::CONTENT_TYPE, ct)], body).into_response();
    }
    match w.synthesize(req.voice.clone(), req.input.clone(), seed).await {
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
    }
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

pub fn speech_model(assets: &Path) -> Result<Option<Arc<SpeechModel>>, String> {
    if let Some(m) = speech_models().lock().get(assets) {
        return Ok(m.clone());
    }
    let model = match SpeechContract::load(assets).map_err(|e| e.to_string())? {
        None => None,
        Some(contract) => {
            let codec = Codec::load(assets, CODEC_MAX_BATCH, CODEC_MAX_FRAMES)?;
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
    crate::serve::api_error(StatusCode::INTERNAL_SERVER_ERROR, msg, "server_error", None, None)
}

pub async fn speech(
    State(state): State<Arc<AppState>>,
    req: Result<Json<SpeechRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(mut req) = match req {
        Ok(r) => r,
        Err(e) => return crate::serve::api_error(e.status(), e.body_text(), "invalid_request_error", Some("invalid_json"), None),
    };
    let t_arrive = Instant::now();
    // Bound first: a guard in the `if let` scrutinee would live across the await.
    let worker = workers().lock().get(&req.model).cloned();
    if let Some(w) = worker {
        return speech_on_worker(w, req, t_arrive).await;
    }
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
    let model = match tokio::task::block_in_place(|| speech_model(&bundle.dir)) {
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
    let job = crate::serve::mux::Job { prompt_ids, gen, arrived: Instant::now(), respond: tx };
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
        tokio::spawn(stream_task(Arc::clone(&model), rx, out_tx, seed, t_arrive));
        let mut out_rx = out_rx;
        let body = Body::from_stream(futures::stream::poll_fn(move |cx| out_rx.poll_recv(cx)));
        return ([(header::CONTENT_TYPE, content_type)], body).into_response();
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
    (
        [
            (header::CONTENT_TYPE, content_type.to_string()),
            (HeaderName::from_static("x-plow-audio-seconds"), format!("{audio_s:.3}")),
            (HeaderName::from_static("x-plow-lm-ms"), format!("{:.1}", t_lm.as_secs_f64() * 1e3)),
        ],
        out,
    )
        .into_response()
}

/// Whole utterance; beyond the codec's frame capacity, windows with `WINDOW` frames of context
/// on each side keep their centres.
async fn decode_all(model: &SpeechModel, codes: &[i32], frames: usize, seed: u64) -> Result<Vec<f32>, String> {
    let (fc, fs, max) = (model.contract.frame_codes, model.contract.frame_samples, model.codec.max_frames);
    if frames <= max {
        return model.codec.decode(codes.to_vec(), frames, seed).await;
    }
    let step = max - 2 * WINDOW;
    let mut pcm = Vec::with_capacity(frames * fs);
    let mut s = 0;
    while s < frames {
        let e = (s + step).min(frames);
        let (ws, we) = (s.saturating_sub(WINDOW), (e + WINDOW).min(frames));
        let w = model.codec.decode(codes[ws * fc..we * fc].to_vec(), we - ws, seed ^ s as u64).await?;
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
fn stream_step(n: usize, emitted: usize, done: bool) -> Option<(usize, usize, usize)> {
    let upto = if done { n } else { n.saturating_sub(LOOKAHEAD) };
    (upto > emitted).then(|| {
        let e = n.min(upto + LOOKAHEAD);
        (e.saturating_sub(WINDOW).min(emitted), e, upto)
    })
}

async fn stream_task(
    model: Arc<SpeechModel>,
    mut rx: stream_mod::ChunkReceiver,
    out: tokio::sync::mpsc::Sender<Result<Vec<u8>, std::io::Error>>,
    seed: u64,
    t_arrive: Instant,
) {
    let c = model.contract.clone();
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
        if let Some((s, e, upto)) = stream_step(frames.len() / fc, emitted, done) {
            let window = frames[s * fc..e * fc].to_vec();
            match model.codec.decode(window, e - s, seed ^ (emitted as u64).wrapping_mul(0x9E37_79B9)).await {
                Ok(pcm) => {
                    let mut bytes = Vec::new();
                    pcm16(&pcm[(emitted - s) * fs..(upto - s) * fs], &mut bytes);
                    first.get_or_insert_with(|| t_arrive.elapsed());
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

    /// Every frame is emitted exactly once, in order, each with LOOKAHEAD right context until
    /// the final flush.
    #[test]
    fn stream_plan_covers_each_frame_once() {
        for total in 1..20 {
            let (mut emitted, mut seen) = (0, Vec::new());
            for n in 1..=total {
                if let Some((s, e, upto)) = stream_step(n, emitted, false) {
                    assert!(s <= emitted && upto <= e && e <= n && e - s <= WINDOW + LOOKAHEAD);
                    assert!(e - upto >= LOOKAHEAD.min(n - upto));
                    seen.extend(emitted..upto);
                    emitted = upto;
                }
            }
            if let Some((_, e, upto)) = stream_step(total, emitted, true) {
                assert_eq!((e, upto), (total, total));
                seen.extend(emitted..upto);
            }
            assert_eq!(seen, (0..total).collect::<Vec<_>>());
        }
    }
}
