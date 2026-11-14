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
//! `stream.lookahead_frames` of right context (both `codec.pkt` parameters). A stateful codec
//! (`stream.stateful`) instead decodes each frame once, in steps on the stream's codec slot.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use axum::body::Body;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{header, HeaderName, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use parking_lot::Mutex;
use serde::Deserialize;

use super::api::{Checked, SpeechSpec};
use super::codec::{Codec, Urgency};
use super::realtime::Ticket;
use super::SpeechContract;
use crate::serve::session::{InFlight, RequestIds};
use crate::serve::stream::{self as stream_mod, StreamChunk};
use crate::serve::AppState;

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    #[serde(default)]
    pub tags: super::tags::Mode,
    pub model: String,
    /// At most [`super::MAX_INPUT_CHARS`] characters; a long input is spoken as sentence segments.
    pub input: String,
    /// One of the model's voices (its `/v1/models` card: `x_plow_speech.voices`).
    pub voice: String,
    /// `wav` (default) or `pcm` (raw s16le mono at the reply's sample rate).
    #[serde(default)]
    pub response_format: Option<String>,
    /// Hz the reply is resampled to, within `MIN_OUT_RATE..=MAX_OUT_RATE`; default: the model's
    /// own rate (its card's `x_plow_speech.sample_rate`).
    #[serde(default)]
    pub sample_rate: Option<u32>,
    /// OpenAI `speed`, within `MIN_SPEED..=MAX_SPEED`: the reply is time-stretched without
    /// moving its pitch. 1.0 (default) leaves the audio untouched.
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
    /// Models that normalise text (card `text_normalization`): `false` speaks the input as given.
    #[serde(default)]
    pub normalize: Option<bool>,
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
    /// Dual-stream models: the whole text in the prompt instead of one token per frame.
    #[serde(default)]
    pub non_streaming_mode: bool,
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

fn stamp_tags(response: &mut Response, report: &super::tags::Report) {
    if !report.active {
        return;
    }
    for (key, names) in [
        ("x-plow-tags-applied", &report.applied),
        ("x-plow-tags-dropped", &report.dropped),
    ] {
        if let Ok(value) = names.join(",").parse() {
            response
                .headers_mut()
                .insert(HeaderName::from_static(key), value);
        }
    }
}

fn silence_response(
    out: &mut super::OutPcm,
    ms: Option<u32>,
    rate: u32,
    report: &super::tags::Report,
) -> Response {
    let n = (f64::from(ms.unwrap_or(0)) * f64::from(rate) * f64::from(out.speed()) / 1000.0).round()
        as usize;
    out.keep_onset();
    let mut response = whole_response(out, &vec![0.0; n], 0.0);
    stamp_tags(&mut response, report);
    response
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
    buckets: &Arc<crate::obs::bucket::BucketDemand>,
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
        let g = super::guided_speech::GuidedSpeech::start(assets, credit, buckets, device)
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
    let buckets = &state.model_metrics(slug).buckets;
    let credit = state.downstream(slug);
    let bound = guided_model(&bundle.dir, credit, buckets, device).and_then(|g| match g {
        Some(_) => Ok(()),
        None => speech_model(&bundle.dir, state.downstream(slug), buckets, device).map(drop),
    });
    bound.map_err(|e| format!("{slug}: speech pipeline failed to bind: {e}"))
}

/// `slug` has a bound speech pipeline (a TTS model): its catalogue card lists `audio/speech`.
pub fn serves_speech(state: &AppState, slug: &str) -> bool {
    let Ok(bundle) = state.registry.get(slug) else {
        return false;
    };
    bundle.is_speech()
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
    buckets: &Arc<crate::obs::bucket::BucketDemand>,
    device: u8,
) -> Result<Pipeline, String> {
    if let Some(g) = guided_model(dir, Arc::clone(&credit), buckets, device)? {
        return Ok(Pipeline::Guided(g));
    }
    Ok(match speech_model(dir, credit, buckets, device)? {
        Some(m) => Pipeline::Speech(m),
        None => Pipeline::None,
    })
}

/// `slug`'s [`SpeechSpec`] when it serves speech (its `/v1/models` card).
pub fn speech_spec(state: &AppState, slug: &str) -> Option<SpeechSpec> {
    let bundle = state.registry.get(slug).ok()?;
    if let Some(pipe) = bound_pipeline(&bundle.dir, state.ordinal_of(slug)) {
        return match pipe {
            Pipeline::Guided(g) => Some(g.spec.clone()),
            Pipeline::Speech(m) => Some(m.spec(&**bundle.tokenizer()).clone()),
            Pipeline::None => None,
        };
    }
    if let Ok(Some(s)) = SpeechContract::load(&bundle.dir) {
        return Some(s.spec(&**bundle.tokenizer()));
    }
    None
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
    let mut req = req;
    let mut checked = checked;
    let rendered = match super::tags::render(
        w.spec.tags.as_deref(),
        &req.input,
        req.tags,
        checked.language.as_deref(),
        req.instructions.as_deref(),
        checked.speed,
    ) {
        Ok(r) => r,
        Err(e) => return bad(e, "tags"),
    };
    let tag_report = rendered.report.clone();
    checked.speed = rendered.speed;
    let (segs, trailing_pause) = rendered.segments(
        w.segment_chars(&req.input, checked.language.as_deref()),
        str::to_owned,
    );
    req.instructions = rendered.instructions;
    let lang = checked.language.clone();
    let voice = match (&req.ref_audio, w.clone_input()) {
        (None, _) => super::guided_speech::Voice::Named(checked.voice.clone()),
        (Some(audio), Some((rate, max))) => {
            let pcm = match reference_pcm(audio, rate, max) {
                Ok(p) => p,
                Err((msg, param)) => return bad(msg, param),
            };
            let enc = Arc::clone(&w);
            match tokio::task::spawn_blocking(move || enc.clone_voice(&pcm)).await {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => return speech_failure(e),
                Err(e) => {
                    return speech_failure(super::guided_speech::SpeechError::Failed(e.to_string()))
                }
            }
        }
        (Some(_), None) => return bad("this model has no voice cloning", "ref_audio"),
    };
    // Later segments resume no session: the first one's ticket carries the voice prefix.
    let later_ids = RequestIds {
        session_end: false,
        session: None,
        ..ids.clone()
    };
    let mut out_pcm = match super::OutPcm::new(
        w.sample_rate,
        checked.sample_rate,
        checked.speed,
        checked.wav,
    ) {
        Ok(o) => o,
        Err(e) => return server_error(e),
    };
    if segs.is_empty() {
        let mut response =
            silence_response(&mut out_pcm, trailing_pause, w.sample_rate, &tag_report);
        run.first();
        run.done();
        run.stamp(&mut response);
        return response;
    }
    if req.stream {
        let first = match w.synthesize_stream(
            &mux,
            voice.clone(),
            segs[0].text.clone(),
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
        let header = out_pcm.header(u32::MAX);
        tokio::spawn(guided_stream_task(
            w,
            mux,
            voice,
            segs,
            trailing_pause,
            checked,
            out_pcm,
            later_ids,
            first,
            out_tx,
            head_tx,
            t_arrive,
            run,
        ));
        let mut response = stream_response(out_rx, head_rx, header, in_flight, cache, false).await;
        stamp_tags(&mut response, &tag_report);
        return response;
    }
    let (mut pcm, mut joiner) = (Vec::new(), super::Joins::new(w.sample_rate));
    let (mut tokens, mut t3_ms, mut s3gen_ms) = (0, 0.0, 0.0);
    let mut report = report;
    for (k, text) in segs.iter().enumerate() {
        let a = match w
            .synthesize(
                &mux,
                voice.clone(),
                text.text.clone(),
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
        if segs.len() == 1 && segs[0].pause_before.is_none() && trailing_pause.is_none() {
            pcm = a.pcm;
        } else {
            if let Some(ms) = text.pause_before {
                joiner.pause(ms, checked.speed, &mut pcm);
            } else if k > 0 {
                joiner.next_segment();
            }
            joiner.tagged_tail(
                segs.get(k + 1)
                    .map_or(trailing_pause.is_some(), |s| s.pause_before.is_some()),
            );
            joiner.push(&a.pcm, f64::INFINITY, &mut pcm);
        }
    }
    if let Some(ms) = trailing_pause {
        joiner.pause(ms, checked.speed, &mut pcm);
    }
    let cache = crate::serve::session::CacheOutcome::received(report_rx).await;
    run.admitted(cache.and_then(|o| o.at));
    run.first();
    run.done();
    let total = t_arrive.elapsed().as_secs_f64();
    let audio_s = pcm.len() as f64 / f64::from(w.sample_rate) / f64::from(checked.speed);
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
    let mut response = whole_response(&mut out_pcm, &pcm, t3_ms);
    stamp_tags(&mut response, &tag_report);
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
    voice: super::guided_speech::Voice,
    segs: Vec<super::tags::TextSegment>,
    trailing_pause: Option<u32>,
    checked: Checked,
    mut out_pcm: super::OutPcm,
    later_ids: RequestIds,
    mut ev: tokio::sync::mpsc::UnboundedReceiver<super::guided_speech::StreamEvent>,
    out: tokio::sync::mpsc::Sender<Result<Vec<u8>, std::io::Error>>,
    head: tokio::sync::oneshot::Sender<axum::http::HeaderMap>,
    t_arrive: Instant,
    mut run: crate::serve::turns::StageRun,
) {
    use super::guided_speech::StreamEvent;
    // The playback rate, not the model's: `speed` changes how much playback time each sample
    // buys, which is what the lead, the realtime projection and the delivered-audio metrics all
    // measure. At speed 1.0 this is the model's rate.
    let sr = f64::from(w.sample_rate) * f64::from(out_pcm.speed());
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
                text.text.clone(),
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
        if let Some(ms) = text.pause_before {
            let mut pcm = Vec::new();
            joiner.pause(ms, checked.speed, &mut pcm);
            let mut bytes = Vec::new();
            out_pcm.push(&pcm, &mut bytes);
            sent += pcm.len();
            if !bytes.is_empty() {
                run.audio(pcm.len(), sr);
                first.get_or_insert_with(|| t_arrive.elapsed());
                playing.get_or_insert_with(Instant::now);
                if let Some(h) = head.take() {
                    let _ = h.send(run.headers());
                }
                if out.send(Ok(bytes)).await.is_err() {
                    return;
                }
            }
        }
        joiner.tagged_tail(
            segs.get(k + 1)
                .map_or(trailing_pause.is_some(), |s| s.pause_before.is_some()),
        );
        loop {
            match ev.recv().await {
                Some(StreamEvent::Pcm(p)) => {
                    let fresh = if segs.len() == 1
                        && segs[0].pause_before.is_none()
                        && trailing_pause.is_none()
                    {
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
                    out_pcm.push(&fresh, &mut bytes);
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
    let mut tail = Vec::new();
    if let Some(ms) = trailing_pause {
        let mut pcm = Vec::new();
        joiner.pause(ms, checked.speed, &mut pcm);
        out_pcm.push(&pcm, &mut tail);
    }
    out_pcm.finish(&mut tail);
    if !tail.is_empty() && out.send(Ok(tail)).await.is_err() {
        return;
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
    coalesce: bool,
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
    let mut pending_error = None;
    let rest = futures::stream::poll_fn(move |cx| {
        let _held = &in_flight;
        if coalesce {
            poll_audio(&mut out_rx, &mut pending_error, cx)
        } else {
            out_rx.poll_recv(cx)
        }
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

fn poll_audio(
    rx: &mut tokio::sync::mpsc::Receiver<Result<Vec<u8>, std::io::Error>>,
    pending_error: &mut Option<std::io::Error>,
    cx: &mut std::task::Context<'_>,
) -> std::task::Poll<Option<Result<Vec<u8>, std::io::Error>>> {
    use std::task::Poll;
    if let Some(error) = pending_error.take() {
        return Poll::Ready(Some(Err(error)));
    }
    let mut bytes = match rx.poll_recv(cx) {
        Poll::Ready(Some(Ok(bytes))) => bytes,
        other => return other,
    };
    for _ in 0..rx.len() {
        match rx.try_recv() {
            Ok(Ok(next)) => bytes.extend_from_slice(&next),
            Ok(Err(error)) => {
                *pending_error = Some(error);
                break;
            }
            Err(_) => break,
        }
    }
    Poll::Ready(Some(Ok(bytes)))
}

/// A whole reply: the audio and the same headers on every model.
fn whole_response(out: &mut super::OutPcm, pcm: &[f32], lm_ms: f64) -> Response {
    let mut body = Vec::new();
    out.push(pcm, &mut body);
    out.finish(&mut body);
    let audio_s = (body.len() / 2) as f64 / f64::from(out.rate());
    let content_type = out.content_type();
    let mut bytes = out.header(body.len() as u32).unwrap_or_default();
    bytes.extend_from_slice(&body);
    (
        [
            (header::CONTENT_TYPE, content_type.to_string()),
            (
                HeaderName::from_static("x-plow-audio-seconds"),
                format!("{audio_s:.3}"),
            ),
            (
                HeaderName::from_static("x-plow-lm-ms"),
                format!("{lm_ms:.1}"),
            ),
        ],
        bytes,
    )
        .into_response()
}

pub struct SpeechModel {
    pub contract: SpeechContract,
    codec: Codec,
    /// The contract's text encoder: its rows are every prompt row.
    encoder: Option<Arc<super::text_encoder::TextEncoder>>,
    /// The contract's reference encoder (voice cloning).
    ref_encoder: Option<Arc<super::ref_encoder::RefEncoder>>,
    realtime: Arc<super::realtime::RealTime>,
    spec: OnceLock<SpeechSpec>,
    /// Maya 2's cloned-speaker conditioning (`spk` / `icl_spk`), when the bundle carries it.
    speaker: Option<Arc<Maya2Speaker>>,
    /// Running frame period of this model's stateful streams, microseconds (0: none measured).
    frame_period: FramePeriod,
}

/// ECAPA and the speaker tables of a Maya 2 bundle (`checkpoint/ecapa`, `checkpoint/speaker`).
struct Maya2Speaker {
    ecapa: super::ecapa::Ecapa,
    tables: super::maya2::Speaker,
    /// Conditionings by reference audio (base64 hash): a voice reused across requests embeds once.
    cache: parking_lot::Mutex<std::collections::HashMap<u64, Arc<SpeakerCond>>>,
}

/// One cloned speaker's anchor row and depth memory.
struct SpeakerCond {
    anchor: Vec<f32>,
    memory: Arc<[f32]>,
}

impl Maya2Speaker {
    const CACHE: usize = 64;

    fn load(assets: &Path) -> Option<Self> {
        let ck = assets.join("checkpoint");
        let (e, t) = (ck.join("ecapa"), ck.join("speaker"));
        if !e.is_dir() || !t.is_dir() {
            return None;
        }
        let load = || -> Result<Self, String> {
            Ok(Maya2Speaker {
                ecapa: super::ecapa::Ecapa::load(&e).map_err(|x| x.to_string())?,
                tables: super::maya2::Speaker::load(&t)?,
                cache: Default::default(),
            })
        };
        load()
            .inspect_err(
                |e| tracing::warn!(error = %e, "tts: maya2 speaker conditioning unavailable"),
            )
            .ok()
    }

    /// The conditioning of `pcm` (24 kHz), keyed by its request encoding `key`.
    fn cond(&self, key: &str, pcm: &[f32]) -> Result<Arc<SpeakerCond>, String> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut h);
        let k = h.finish();
        if let Some(c) = self.cache.lock().get(&k) {
            return Ok(Arc::clone(c));
        }
        let e = self.ecapa.embed(pcm)?;
        let c = Arc::new(SpeakerCond {
            anchor: self.tables.anchor_row(&e),
            memory: self.tables.memory(&e).into(),
        });
        let mut cache = self.cache.lock();
        if cache.len() >= Self::CACHE {
            cache.clear();
        }
        cache.insert(k, Arc::clone(&c));
        Ok(c)
    }
}

/// A model's running frame period over its stateful streams, with when it was last observed:
/// (ms since process start) << 32 | microseconds. Streams observe it every few frames, so under
/// load it is always fresh; once the streams that measured it are gone it says nothing about
/// the next request, and after [`FramePeriod::FRESH_MS`] it is no longer used.
#[derive(Default)]
struct FramePeriod(std::sync::atomic::AtomicU64);

impl FramePeriod {
    const FRESH_MS: u32 = 1_000;

    fn now_ms() -> u32 {
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_millis() as u32
    }

    /// The period in seconds, while an observation is fresh.
    fn recent(&self, now_ms: u32) -> Option<f64> {
        let v = self.0.load(std::sync::atomic::Ordering::Relaxed);
        let (at, us) = ((v >> 32) as u32, v as u32);
        (us > 0 && now_ms.wrapping_sub(at) <= Self::FRESH_MS).then(|| f64::from(us) * 1e-6)
    }

    fn observe(&self, seconds: f64, now_ms: u32) {
        let us = seconds * 1e6;
        let us = match self.recent(now_ms) {
            Some(old) => old * 1e6 + 0.1 * (us - old * 1e6),
            None => us,
        };
        let us = us.clamp(1.0, f64::from(u32::MAX)) as u64;
        self.0.store(
            u64::from(now_ms) << 32 | us,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Whether a request starting now banks a frame: live streams are producing frames slower
    /// than half real time (`frame`: one frame's audio, seconds).
    fn banks(&self, frame: f64, now_ms: u32) -> bool {
        self.recent(now_ms)
            .is_some_and(|period| period > 0.5 * frame)
    }
}

impl SpeechModel {
    /// Frames a stream's first stateful codec step waits for: the contract's, plus one banked frame
    /// while frames come slower than half real time (a stream then starts with the lead that lets
    /// its later steps wait for more than one frame). An explicit `PLOW_TTS_STREAM_FIRST_FRAMES`
    /// or `PLOW_TTS_STREAM_DEADLINE=0` keeps the contract's.
    fn first_frames(&self) -> usize {
        let cfg = crate::config::RuntimeConfig::get();
        let frame = self.contract.frame_samples as f64 / f64::from(self.contract.sample_rate);
        let bank = cfg.tts_stream_deadline
            && cfg.tts_stream_first_frames.is_none()
            && self.frame_period.banks(frame, FramePeriod::now_ms());
        self.contract.first_frames() + usize::from(bank)
    }

    fn observe_frame_period(&self, seconds: f64) {
        self.frame_period.observe(seconds, FramePeriod::now_ms());
    }

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
    buckets: &Arc<crate::obs::bucket::BucketDemand>,
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
            codec.observe_buckets(buckets);
            if codec.frame_codes != contract.frame_codes
                || codec.frame_samples != contract.frame_samples
            {
                return Err(format!(
                    "codec packet frames ({} codes, {} samples) disagree with the speech contract ({}, {})",
                    codec.frame_codes, codec.frame_samples, contract.frame_codes, contract.frame_samples
                ));
            }
            if contract.conditioning.is_some() && codec.step_frames == 0 {
                return Err("external speech conditioning requires a stateful codec".into());
            }
            let encoder = match &contract.text_encoder {
                Some(sub) => {
                    let e = super::text_encoder::TextEncoder::load(&assets.join(sub), device)?;
                    e.observe_buckets(buckets);
                    Some(Arc::new(e))
                }
                None => None,
            };
            let reference_file = contract.clone.as_ref().map(|c| &c.encoder).or_else(|| {
                contract
                    .conditioning
                    .as_ref()
                    .map(|c| &c.packets.reference_packet)
            });
            let ref_encoder = match reference_file {
                Some(file) => {
                    let mut e = super::ref_encoder::RefEncoder::load(&assets.join(file), device)?;
                    e.observe_buckets(buckets);
                    if e.frame_codes != contract.frame_codes
                        || e.frame_samples != contract.frame_samples
                        || e.sample_rate != contract.sample_rate
                        || e.codebook != contract.codebook
                    {
                        return Err(format!(
                            "{file} frames ({} codes, {} samples at {} Hz) disagree with the speech contract",
                            e.frame_codes, e.frame_samples, e.sample_rate
                        ));
                    }
                    if let Some(c) = &contract.conditioning {
                        let main = crate::asset::devblob::DevBlob::find_in_dir(assets)
                            .map_err(|e| e.to_string())?
                            .ok_or("speech packet is missing")?;
                        let pad = super::ref_encoder::SpeakerEncoder::text_pad(&main, c)?;
                        let speaker = super::ref_encoder::SpeakerEncoder::load(
                            &assets.join(&c.packets.speaker_packet),
                            device,
                        )?;
                        e = e.with_speaker(speaker, pad)?;
                    }
                    Some(Arc::new(e))
                }
                None => None,
            };
            tracing::info!(pipeline = %contract.pipeline, sample_rate = contract.sample_rate, "tts: speech pipeline bound");
            let realtime = super::realtime::RealTime::new(
                contract.frame_samples as f64 / f64::from(contract.sample_rate),
                contract.tokens_per_frame(),
            );
            let speaker = contract
                .maya2
                .as_ref()
                .and_then(|_| Maya2Speaker::load(assets))
                .map(Arc::new);
            Some(Arc::new(SpeechModel {
                contract,
                codec,
                encoder,
                ref_encoder,
                realtime,
                spec: OnceLock::new(),
                speaker,
                frame_period: Default::default(),
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

/// WebSocket streaming speech endpoint: accepts requests on `/v1/audio/speech` (GET upgrade)
/// or `/v1/tts/stream`.
pub async fn ws_speech(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    let ids = match RequestIds::from_headers(&headers) {
        Ok(ids) => ids,
        Err(e) => return bad(e, "x-session-id"),
    };
    if let Some(r) = crate::serve::overload::gate(&ids) {
        return r;
    }
    ws.max_message_size(1024 * 1024)
        .max_frame_size(1024 * 1024)
        .on_upgrade(move |socket| async move {
            ws_speech_session(state, socket, ids).await;
        })
}

async fn ws_speech_session(state: Arc<AppState>, mut socket: WebSocket, handshake_ids: RequestIds) {
    while let Some(msg) = socket.recv().await {
        let text = match msg {
            Ok(Message::Text(t)) => t,
            Ok(Message::Ping(p)) => {
                let _ = socket.send(Message::Pong(p)).await;
                continue;
            }
            Ok(Message::Close(_)) | Err(_) => break,
            _ => continue,
        };
        let mut req: SpeechRequest = match serde_json::from_str(&text) {
            Ok(r) => r,
            Err(e) => {
                let _ = socket
                    .send(Message::Text(
                        serde_json::json!({
                            "error": {
                                "message": e.to_string(),
                                "type": "invalid_request_error"
                            }
                        })
                        .to_string(),
                    ))
                    .await;
                continue;
            }
        };
        req.stream = true;
        let model_metrics = state.model_metrics(&req.model);
        model_metrics
            .serving
            .tts
            .active
            .store(true, std::sync::atomic::Ordering::Relaxed);
        model_metrics
            .serving
            .tts
            .websocket_sessions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut ids = handshake_ids.with_new_request();
        if let Err(e) = ids.apply_body(&req.route()) {
            let _ = socket
                .send(Message::Text(
                    serde_json::json!({
                        "error": {
                            "message": e,
                            "type": "invalid_request_error"
                        }
                    })
                    .to_string(),
                ))
                .await;
            continue;
        }
        let (report, mut report_rx) = ids.report();
        let mut response = speech_with(
            Arc::clone(&state),
            Ok(Json(req)),
            &ids,
            report,
            &mut report_rx,
        )
        .await;
        ids.stamp(&mut response);

        if !response.status().is_success() {
            let bytes = match axum::body::to_bytes(response.into_body(), 64 * 1024).await {
                Ok(b) => String::from_utf8_lossy(&b).into_owned(),
                Err(e) => format!(r#"{{"error":{{"message":"{e}"}}}}"#),
            };
            let _ = socket.send(Message::Text(bytes)).await;
            continue;
        }

        if response.headers().contains_key("x-plow-tags-applied") {
            let report = serde_json::json!({"type":"tags", "applied":response.headers().get("x-plow-tags-applied").and_then(|v|v.to_str().ok()).unwrap_or(""), "dropped":response.headers().get("x-plow-tags-dropped").and_then(|v|v.to_str().ok()).unwrap_or("")});
            if socket
                .send(Message::Text(report.to_string()))
                .await
                .is_err()
            {
                break;
            }
        }
        let mut stream = response.into_body().into_data_stream();
        let mut sent_any = false;
        let mut stream_err: Option<String> = None;
        let mut clean_eof = false;
        loop {
            tokio::select! {
                chunk = stream.next() => {
                    match chunk {
                        Some(Ok(c)) if !c.is_empty() => {
                            sent_any = true;
                            if socket.send(Message::Binary(c.to_vec())).await.is_err() {
                                break;
                            }
                        }
                        Some(Ok(_)) => {}
                        Some(Err(e)) => {
                            stream_err = Some(e.to_string());
                            break;
                        }
                        None => {
                            clean_eof = true;
                            break;
                        }
                    }
                }
                msg = socket.recv() => {
                    match msg {
                        Some(Ok(Message::Ping(p))) => {
                            let _ = socket.send(Message::Pong(p)).await;
                        }
                        Some(Ok(Message::Close(_))) | None => break,
                        _ => {}
                    }
                }
            }
        }
        if let Some(err_msg) = stream_err {
            let _ = socket
                .send(Message::Text(
                    serde_json::json!({
                        "error": {
                            "message": err_msg,
                            "type": "server_error"
                        }
                    })
                    .to_string(),
                ))
                .await;
        } else if clean_eof && sent_any {
            let _ = socket
                .send(Message::Text(serde_json::json!({"done": true}).to_string()))
                .await;
        }
    }
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
    crate::obs::ttft::reset();
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
                if let Err(e) = state.dp_admit(set, false).await {
                    return e.with_retry_after(crate::serve::api_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        e.to_string(),
                        "server_error",
                        None,
                        None,
                    ));
                }
                state
                    .dp_route(set, ids.session.as_deref(), None, exclude, false)
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
                            return e.with_retry_after(crate::serve::api_error(
                                StatusCode::SERVICE_UNAVAILABLE,
                                e.to_string(),
                                "server_error",
                                None,
                                None,
                            ));
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
    let model_metrics = state.model_metrics(&key);
    model_metrics
        .serving
        .tts
        .active
        .store(true, std::sync::atomic::Ordering::Relaxed);
    model_metrics
        .serving
        .tts
        .http_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut run = crate::serve::turns::StageRun::start(
        ids,
        crate::serve::turns::Kind::Tts,
        &req.model,
        Some(model_metrics),
        t_arrive,
        true,
    );
    let ids = &RequestIds {
        session_end: false,
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
                &state.model_metrics(&req.model).buckets,
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
    let mut checked = match check(model.spec(&**tok), &req) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let c = &model.contract;
    let rendered = match super::tags::render(
        c.tags.as_deref(),
        &req.input,
        req.tags,
        checked.language.as_deref(),
        req.instructions.as_deref(),
        checked.speed,
    ) {
        Ok(r) => r,
        Err(e) => return bad(e, "tags"),
    };
    let tag_report = rendered.report.clone();
    checked.speed = rendered.speed;
    req.instructions = rendered.instructions.clone();
    // The guidance contrasts the instructed voice with the uninstructed one.
    if c.depth_cfg()
        && checked.sampling.cfg_weight.is_some_and(|w| w > 0.0)
        && req
            .instructions
            .as_deref()
            .is_none_or(|i| i.trim().is_empty())
    {
        return bad(
            "`cfg_weight` guides the voice towards `instructions`; give instructions or omit it",
            "cfg_weight",
        );
    }
    let voice = checked.voice.clone();
    let mut out_pcm = match super::OutPcm::new(
        c.sample_rate,
        checked.sample_rate,
        checked.speed,
        checked.wav,
    ) {
        Ok(o) => o,
        Err(e) => return server_error(e),
    };
    if req.stream && crate::config::RuntimeConfig::get().tts_onset_trim {
        out_pcm.trim_onset(c.onset_trim, c.sample_rate);
    }
    let reference = match (&req.ref_audio, &req.ref_text, &model.ref_encoder) {
        (Some(audio), text, Some(enc)) if text.is_some() || c.conditioning.is_some() => {
            match ref_clip(&model, tok, &voice, enc, audio, text.as_deref()) {
                Ok(r) => Some(Arc::new(r)),
                Err((msg, param)) => return bad(msg, param),
            }
        }
        _ => None,
    };
    // Past the cap the reference budget clips the audio: a longer input is spoken in order as
    // sentence-sized segments, each with its full budget.
    // Maya 2 guesses an unnamed language from the text's script.
    let language = checked.language.clone().or_else(|| {
        c.maya2
            .as_ref()
            .map(|_| super::maya2::detect_lang(&req.input).to_owned())
    });
    // Maya 2 clones with the reference's ECAPA speaker (`icl_spk` with a transcript, `spk` without);
    // a bundle without the speaker tables clones by `icl` alone and needs the transcript.
    let speaker = match (&c.maya2, &req.ref_audio, &model.speaker, &model.ref_encoder) {
        (Some(_), Some(audio), Some(sp), Some(enc)) => {
            let pcm = match reference_pcm(audio, c.sample_rate, enc.max_samples()) {
                Ok(p) => p,
                Err((msg, param)) => return bad(msg, param),
            };
            match tokio::task::block_in_place(|| sp.cond(audio, &pcm)) {
                Ok(cond) => Some(cond),
                Err(e) => return bad(format!("reference audio: {e}"), "ref_audio"),
            }
        }
        (Some(_), Some(_), None, _) if req.ref_text.is_none() => {
            return bad(
                "this bundle clones only with the reference's transcript; give `ref_text`",
                "ref_text",
            )
        }
        _ => None,
    };
    let (texts, trailing_pause) =
        rendered.segments(c.segment_chars(), |text| match (&c.maya2, &language) {
            (Some(_), Some(lang)) if req.normalize != Some(false) => {
                super::maya2_text::normalise_text(text, lang)
            }
            _ => text.to_owned(),
        });
    if texts.first().is_some_and(|s| s.pause_before.is_some()) {
        out_pcm.keep_onset();
    }
    if texts.is_empty() {
        let mut response =
            silence_response(&mut out_pcm, trailing_pause, c.sample_rate, &tag_report);
        run.first();
        run.done();
        run.stamp(&mut response);
        return response;
    }
    let maker = SegmentMaker {
        tok: Arc::clone(tok),
        contract: c.clone(),
        voice: voice.clone(),
        language,
        speaker,
        instructions: req
            .instructions
            .as_deref()
            .map(str::trim)
            .filter(|i| !i.is_empty())
            .map(str::to_owned),
        encoder: model.encoder.clone(),
        reference,
        cfg_weight: checked.sampling.cfg_weight.unwrap_or(0.0),
        max_tokens: checked.sampling.max_tokens,
        temperature: checked.sampling.temperature,
        top_p: checked.sampling.top_p,
        repetition_penalty: checked.sampling.repetition_penalty,
        top_k: checked.sampling.top_k.unwrap_or(0),
        seed: checked.seed,
        streaming: !req.non_streaming_mode,
        device_frames: req.stream,
    };
    let mut segs: Vec<Segment> = Vec::new();
    for t in &texts {
        let first = segs.len();
        if !maker.make_fitting(t, &mut segs) {
            return bad(
                "`instructions` leave no room for text in the text encoder's prompt",
                "instructions",
            );
        }
        segs[first].pause_before = t.pause_before;
    }
    for k in 0..segs.len() {
        segs[k].explicit_tail = segs
            .get(k + 1)
            .map_or(trailing_pause.is_some(), |s| s.pause_before.is_some());
    }

    let need = super::realtime::Need {
        stream: req.stream,
        chars: segs.iter().map(|s| s.chars).collect(),
        budget: segs
            .iter()
            .map(|s| s.gen.max_tokens / c.tokens_per_frame())
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
    let mut conditioned_slot = if c.conditioning.is_some() {
        match model.codec.acquire_slot() {
            Some(slot) => Some(slot),
            None => return busy(std::time::Duration::from_millis(100)),
        }
    } else {
        None
    };
    let first = segs[0].clone();
    let mut speech = match maker.speech(&first) {
        Ok(s) => s,
        Err(e) => return server_error(e),
    };
    if let Some(sp) = speech
        .as_mut()
        .filter(|_| req.stream && model.codec.step_frames > 0)
    {
        sp.first_tokens = model.first_frames();
        sp.first_audio = Some(Arc::new(
            crate::serve::mux::FirstAudio::new(sp.first_tokens).track_playback(
                mux.mapped_rows() || sp.batch_delivery,
                c.frame_samples,
                c.sample_rate,
            ),
        ));
    }
    let first_audio = speech.as_ref().and_then(|sp| sp.first_audio.clone());
    let (tx, rx) = stream_mod::channel();
    let opts = crate::serve::mux::JobOpts {
        session_end: false,
        class,
        raw_tokens: true,
        speech,
        // Encoder and cloning prompts' ids are placeholders: no session can key on them.
        session: ids
            .session
            .as_ref()
            .filter(|_| first.enc_ids.is_none() && first.reference.is_none())
            .and_then(|_| {
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
        export: None,
        import: None,
        prompt_logprobs: false,
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
        let header = out_pcm.header(u32::MAX);
        tokio::spawn(stream_task(
            Arc::clone(&model),
            rx,
            later,
            ticket,
            out_pcm,
            out_tx,
            head_tx,
            seed,
            conditioned_slot,
            t_arrive,
            run,
            first_audio,
            trailing_pause,
        ));
        let mut response = stream_response(
            out_rx,
            head_rx,
            header,
            in_flight,
            cache,
            c.batch_delivery && crate::config::RuntimeConfig::process_get().tick_delivery,
        )
        .await;
        stamp_tags(&mut response, &tag_report);
        return response;
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
            let (codes, completed_pcm, n) = match collect_audio(c, rx, &ticket).await {
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
            let f = completed_pcm
                .as_ref()
                .map_or(codes.len() / c.frame_codes, |p| p.len() / c.frame_samples);
            let p = if let Some(pcm) = completed_pcm {
                pcm
            } else if f == 0 {
                Vec::new()
            } else {
                match decode_all(
                    &model,
                    &codes[..f * c.frame_codes],
                    f,
                    seed ^ segment_salt(k),
                    later.maker.reference.as_deref(),
                    &mut conditioned_slot,
                )
                .await
                {
                    Ok(p) => p,
                    Err(e) => return server_error(e),
                }
            };
            frames += f;
            if later.segs.len() == 1
                && later.segs[0].pause_before.is_none()
                && trailing_pause.is_none()
            {
                pcm = p;
                break;
            }
            match later.judge(k, f, f > 0 && joiner.failed(&p), &mut tries) {
                Verdict::Keep => {}
                Verdict::Retry => {
                    if let Some(m) = run.metrics() {
                        m.serving
                            .tts
                            .watchdog_recovers
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    attempt += 1;
                    continue;
                }
                Verdict::Fail => return server_error(format!("segment {k} produced no speech")),
            }
            if let Some(ms) = later.segs[k].pause_before {
                joiner.pause(ms, checked.speed, &mut pcm);
            } else if !pcm.is_empty() {
                joiner.next_segment();
                joins.push(pcm.len());
            }
            joiner.tagged_tail(later.segs[k].explicit_tail);
            joiner.push(&p, f64::INFINITY, &mut pcm);
            break;
        }
        k += 1;
    }
    if let Some(ms) = trailing_pause {
        joiner.pause(ms, checked.speed, &mut pcm);
    }
    drop(ticket);
    if frames == 0 {
        return server_error("model produced no audio frames");
    }
    let audio_s = pcm.len() as f64 / f64::from(c.sample_rate) / f64::from(checked.speed);
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
    let mut response = whole_response(&mut out_pcm, &pcm, t_lm.as_secs_f64() * 1e3);
    stamp_tags(&mut response, &tag_report);
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

/// A cloning reference (base64 or a data URL) as mono PCM at `rate`, at most `max` samples.
fn reference_pcm(audio: &str, rate: u32, max: usize) -> Result<Vec<f32>, (String, &'static str)> {
    use crate::serve::mm::media;
    use base64::Engine as _;
    let bad = |m: String| (m, "ref_audio");
    let bytes = if audio.starts_with("data:") {
        media::data_url(audio).map_err(bad)?.1
    } else {
        base64::engine::general_purpose::STANDARD
            .decode(audio.trim())
            .map_err(|e| bad(format!("`ref_audio` is not base64 audio: {e}")))?
    };
    let seconds = max as f64 / f64::from(rate);
    let (pcm, from) = media::decode_audio(&bytes, "", seconds + 1.0).map_err(bad)?;
    let pcm = media::resample(&pcm, from, rate).map_err(bad)?;
    if pcm.is_empty() || pcm.len() > max {
        return Err(bad(format!(
            "`ref_audio` must hold audio of at most {seconds:.2} s"
        )));
    }
    Ok(pcm)
}

/// Decode, resample and frame a cloning reference; its codes come later, on the engine thread.
fn ref_clip(
    model: &SpeechModel,
    tok: &Arc<dyn crate::text::tokenizer::Tokenize>,
    voice: &str,
    enc: &Arc<super::ref_encoder::RefEncoder>,
    audio: &str,
    text: Option<&str>,
) -> Result<RefClip, (String, &'static str)> {
    let bad = |m: String| (m, "ref_audio");
    let pcm = reference_pcm(audio, enc.sample_rate, enc.max_samples())?;
    if pcm.len() < enc.min_samples() {
        return Err(bad(
            "`ref_audio` is too short for the speaker encoder".into()
        ));
    }
    let c = &model.contract;
    if c.conditioning.is_some() {
        let dual = c
            .dual_stream
            .as_ref()
            .ok_or_else(|| bad("conditioning needs a dual-stream prompt".into()))?;
        let ids = text
            .map(|text| {
                tok.encode_with_special_tokens(
                    &dual.text_template.replace("{input}", text.trim()),
                    false,
                )
            })
            .unwrap_or_default();
        return Ok(RefClip {
            ids,
            lang: None,
            frames: if text.is_some() {
                pcm.len().div_ceil(enc.frame_samples)
            } else {
                0
            },
            eos: None,
            pcm,
            encoder: Arc::clone(enc),
            conditioned: true,
            encoding: OnceLock::new(),
        });
    }
    let text = text.ok_or_else(|| ("voice cloning needs `ref_text`".into(), "ref_text"))?;
    let Some(clone) = &c.clone else {
        return Err(bad("this model has no voice cloning".into()));
    };
    let tok = model.encoder.as_ref().map_or(tok, |e| &e.tok);
    let mut ids = clone.head.clone();
    // Maya 2 (`icl`): the transcript in its own language, no speaker name.
    let lang = c.maya2.as_ref().map(|_| super::maya2::detect_lang(text));
    let voice = if c.maya2.is_some() {
        MAYA2_CLONE_VOICE
    } else {
        voice
    };
    ids.extend(
        tok.encode_with_special_tokens(&c.prompt_text(voice, lang, None, text.trim()), false),
    );
    ids.extend_from_slice(&clone.mid);
    if model.encoder.as_ref().is_some_and(|e| ids.len() > e.rows()) {
        return Err((
            "`ref_text` does not fit the text encoder's prompt".into(),
            "ref_text",
        ));
    }
    Ok(RefClip {
        ids,
        lang: lang.map(str::to_owned),
        frames: pcm.len().div_ceil(enc.frame_samples),
        eos: clone.eos_code,
        pcm,
        encoder: Arc::clone(enc),
        conditioned: false,
        encoding: OnceLock::new(),
    })
}

/// One segment's LM job.
#[derive(Clone)]
struct Segment {
    pause_before: Option<u32>,
    explicit_tail: bool,
    native: Vec<super::tags::Native>,
    text: String,
    /// With a text encoder: zeros sizing the prefill; every row is the encoder's row of
    /// `enc_ids`, encoded when the segment is submitted.
    prompt_ids: Vec<u32>,
    enc_ids: Option<Vec<u32>>,
    /// A guided segment's unconditional encoder ids (the prompt without its instructions).
    uncond_ids: Option<Vec<u32>>,
    /// Voice cloning: the prompt opens with the reference (its transcript's rows, its frames and
    /// any end frame); `enc_ids` / `uncond_ids` are the text's segment after it.
    reference: Option<Arc<RefClip>>,
    gen: crate::serve::GenParams,
    chars: usize,
    /// Index of the segment in its request.
    k: usize,
    /// A dual-stream prompt's frames and text feed (`Err`: the text has no rows).
    dual: Option<Result<plow_asset::dual_stream::Layout, String>>,
    speaker_row: Option<usize>,
}

/// The voice a Maya 2 clone renders as: the plain prompt (`icl` names no speaker).
const MAYA2_CLONE_VOICE: &str = "plain";

/// A request's voice-cloning reference.
struct RefClip {
    /// Maya 2: the transcript's language; a text in the same language repeats no language tag.
    lang: Option<String>,
    /// The transcript's prompt segment (encoder ids with a text encoder, else token ids): the
    /// contract's clone head, the transcript, its mid.
    ids: Vec<u32>,
    /// The reference audio at the encoder's rate; `frames` = its frame count.
    pcm: Vec<f32>,
    frames: usize,
    /// The end frame's code on every codebook, when the reference has one.
    eos: Option<u32>,
    encoder: Arc<super::ref_encoder::RefEncoder>,
    /// `[frames][frame_codes]`, encoded on the engine thread by the first segment to prefill.
    conditioned: bool,
    encoding: OnceLock<Result<Arc<super::ref_encoder::ReferenceEncoding>, String>>,
}

impl RefClip {
    /// Prompt rows ahead of the text's segment: transcript, frames, end frame.
    fn rows(&self) -> usize {
        self.ids.len() + self.frames + usize::from(self.eos.is_some())
    }

    fn codes(&self) -> Result<&[u32], String> {
        self.encoded()?
            .codes
            .as_deref()
            .ok_or_else(|| "reference has no codec frames".into())
    }

    fn encoded(&self) -> Result<&super::ref_encoder::ReferenceEncoding, String> {
        self.encoding
            .get_or_init(|| {
                if self.conditioned {
                    self.encoder.conditioned(&self.pcm, self.frames > 0)
                } else {
                    Ok(Arc::new(super::ref_encoder::ReferenceEncoding {
                        codes: Some(self.encoder.encode(&self.pcm)?.into()),
                        speaker: None,
                    }))
                }
            })
            .as_deref()
            .map_err(Clone::clone)
    }

    /// `[rows][frame_codes]` codes of those rows (zeros on the transcript's), and each frame's
    /// first code.
    fn frame_codes(&self, frame_codes: usize) -> Result<(Vec<u32>, Vec<u32>), String> {
        let codes = self.codes()?;
        let mut v = vec![0; self.ids.len() * frame_codes];
        v.extend_from_slice(codes);
        if let Some(eos) = self.eos {
            v.extend(std::iter::repeat_n(eos, frame_codes));
        }
        Ok((v, codes.iter().step_by(frame_codes).copied().collect()))
    }
}

/// Builds a segment's prompt and generation parameters from its text.
struct SegmentMaker {
    tok: Arc<dyn crate::text::tokenizer::Tokenize>,
    contract: SpeechContract,
    voice: String,
    /// The template's `{language}`.
    language: Option<String>,
    /// Maya 2: the cloned speaker's conditioning.
    speaker: Option<Arc<SpeakerCond>>,
    instructions: Option<String>,
    encoder: Option<Arc<super::text_encoder::TextEncoder>>,
    reference: Option<Arc<RefClip>>,
    /// Depth-stage CFG weight w (scale s = 1 + w); 0 = off.
    cfg_weight: f32,
    max_tokens: Option<usize>,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    top_k: usize,
    /// The request's seed; segment `k` samples with `seed + k`.
    seed: u64,
    /// Dual-stream: the text rides the decode steps.
    streaming: bool,
    device_frames: bool,
}

impl SegmentMaker {
    fn make(&self, k: usize, text: &str) -> Segment {
        let c = &self.contract;
        let tok = self.encoder.as_ref().map_or(&self.tok, |e| &e.tok);
        // A cloning prompt's text segment follows the reference with the clone tail.
        let lead = match (&self.reference, &c.clone) {
            (Some(_), Some(clone)) => &clone.tail,
            _ => &c.prefix,
        };
        let ids = |instructions: Option<&str>| {
            let mut ids = lead.clone();
            let cloned = self.reference.is_some() || self.speaker.is_some();
            let mut prompt = match (cloned, &c.maya2) {
                (true, Some(_)) => c.prompt_text(
                    MAYA2_CLONE_VOICE,
                    self.language.as_deref(),
                    instructions,
                    text,
                ),
                _ => c.prompt_text(&self.voice, self.language.as_deref(), instructions, text),
            };
            if let (Some(r), Some(l)) = (&self.reference, &self.language) {
                if r.lang.as_ref() == Some(l) {
                    prompt = prompt.replacen(&format!("<|lang:{l}|>"), "", 1);
                }
            }
            ids.extend(tok.encode_with_special_tokens(&prompt, false));
            ids.extend_from_slice(&c.suffix);
            ids
        };
        let dual = c.dual_stream.as_ref().map(|d| {
            let text_ids =
                tok.encode_with_special_tokens(&d.text_template.replace("{input}", text), false);
            let ins_ids = match (&d.instructions_template, self.instructions.as_deref()) {
                (Some(t), Some(i)) => {
                    tok.encode_with_special_tokens(&t.replace("{instructions}", i), false)
                }
                _ => Vec::new(),
            };
            let language = self.language.as_deref().unwrap_or("auto");
            if let Some(r) = self.reference.as_ref().filter(|r| r.conditioned) {
                let placeholders = vec![0; r.frames * c.frame_codes];
                let reference = (r.frames > 0).then_some(plow_asset::dual_stream::Reference {
                    text: &r.ids,
                    codes: &placeholders,
                    codebook_size: c.codebook,
                });
                d.layout_conditioned(&text_ids, &ins_ids, language, self.streaming, reference)
                    .map(|l| (l.layout, Some(l.speaker_row)))
            } else {
                d.layout(&text_ids, &ins_ids, &self.voice, language, self.streaming)
                    .map(|l| (l, None))
            }
        });
        let speaker_row = dual
            .as_ref()
            .and_then(|r| r.as_ref().ok())
            .and_then(|(_, row)| *row);
        let dual = dual.map(|r| r.map(|(layout, _)| layout));
        let mut prompt_ids = match &dual {
            Some(Ok(l)) => l.ids.clone(),
            Some(Err(_)) => Vec::new(),
            None => ids(self.instructions.as_deref()),
        };
        let uncond_ids = (self.cfg_weight > 0.0 && self.encoder.is_some()).then(|| ids(None));
        let enc_ids = self.encoder.is_some().then(|| {
            let ids = prompt_ids.clone();
            prompt_ids.fill(0);
            ids
        });
        if let Some(r) = self.reference.as_ref().filter(|r| !r.conditioned) {
            let mut head = if self.encoder.is_some() {
                vec![0; r.ids.len()]
            } else {
                r.ids.clone()
            };
            // Each frame row's token is its first code's class (filled in once encoded).
            head.extend(std::iter::repeat_n(c.audio_token_base, r.frames));
            head.extend(r.eos.map(|eos| c.audio_token_base + eos));
            head.append(&mut prompt_ids);
            prompt_ids = head;
        }
        let mut gen = crate::serve::GenParams::default();
        gen.max_tokens = match &c.maya2 {
            Some(_) => self.maya2_plan(text, k).cap as usize,
            None => self
                .max_tokens
                .unwrap_or_else(|| c.max_new_tokens(text))
                .min(c.max_new_tokens_cap),
        };
        gen.params.temperature = self.temperature;
        gen.params.top_p = self.top_p;
        gen.params.repetition_penalty = self.repetition_penalty;
        gen.params.top_k = self.top_k;
        gen.seed = Some(self.seed.wrapping_add(k as u64));
        gen.stop_token_ids = c.stops.clone();
        if c.depth {
            // The depth stage samples the frame; the backbone's own draw is discarded.
            gen.params = Default::default();
            gen.params.temperature = 0.0;
        }
        Segment {
            pause_before: None,
            explicit_tail: false,
            native: Vec::new(),
            text: text.to_owned(),
            prompt_ids,
            enc_ids,
            uncond_ids,
            reference: self.reference.clone(),
            gen,
            chars: text.chars().count(),
            k,
            dual,
            speaker_row,
        }
    }
}

impl SegmentMaker {
    /// Maya 2's controller plan for segment `k`: the request's cb0 sampling over the packet's
    /// defaults, and the piece's frame budget and loop-guard windows.
    fn maya2_plan(&self, text: &str, k: usize) -> super::maya2::Plan {
        let c = &self.contract;
        let ctl = c.maya2.as_ref().expect("maya2 contract");
        let mut cb0 = ctl.cb0;
        // `engine.py` clamps the temperature to [0.05, 2].
        cb0.temperature = self.temperature.clamp(0.05, 2.0);
        cb0.top_p = self.top_p;
        cb0.presence_penalty = self.repetition_penalty;
        if self.top_k > 0 {
            cb0.top_k = self.top_k as u32;
        }
        ctl.plan(
            text,
            self.language.as_deref().unwrap_or("en"),
            self.max_tokens,
            cb0,
            self.seed
                .wrapping_mul(1_000_003)
                .wrapping_add(k as u64 * 7919),
        )
    }

    /// Append `text`'s segments: a text-encoder prompt is one launch, so a segment whose prompt
    /// exceeds the encoder is halved (at whitespace when it can) until each fits. `false` when a
    /// single character does not fit.
    fn make_fitting(&self, text: &super::tags::TextSegment, out: &mut Vec<Segment>) -> bool {
        let mut s = self.make(out.len(), &text.text);
        s.native = text.native.clone();
        let rows = self.encoder.as_ref().map_or(usize::MAX, |e| e.rows());
        if s.enc_ids.as_ref().is_none_or(|ids| ids.len() <= rows) {
            out.push(s);
            return true;
        }
        let chars: Vec<_> = text.text.char_indices().collect();
        if chars.len() < 2 {
            return false;
        }
        let mid = chars.len() / 2;
        let cut = (1..mid)
            .rev()
            .find(|&i| chars[i].1.is_whitespace())
            .unwrap_or(mid);
        let Some(cut) = super::tags::safe_cut(&text.text, chars[cut].0, &text.native) else {
            return false;
        };
        let Some(halves) = super::tags::split_at(text, cut) else {
            return false;
        };
        halves
            .iter()
            .all(|half| half.text.is_empty() || self.make_fitting(half, out))
    }

    /// A depth-stage packet's per-segment sampling and, with a text encoder, its prompt rows
    /// (`None` for codec-token packets).
    fn speech(&self, s: &Segment) -> Result<Option<Box<crate::serve::mux::SpeechJob>>, String> {
        let c = &self.contract;
        if !c.depth {
            return Ok(None);
        }
        let dual = s.dual.clone().transpose()?;
        // With a reference, the text's segment follows its rows; with a text encoder, the
        // transcript's rows are overlay rows too.
        let (head, ref_ids) = match (
            s.reference.as_ref().filter(|r| !r.conditioned),
            &self.encoder,
        ) {
            (Some(r), Some(_)) => (r.rows(), Some(r.ids.clone())),
            (Some(r), None) => (r.rows(), None),
            (None, _) => (0, None),
        };
        let rows = |ids: &[u32]| -> Vec<u32> {
            let lead = ref_ids.as_ref().map_or(0, |r| r.len() as u32);
            (0..lead)
                .chain(head as u32..(head + ids.len()) as u32)
                .collect()
        };
        let mut overlay_pos = s.enc_ids.as_deref().map_or(Vec::new(), rows);
        // Maya 2's cloned speaker: its anchor row at the reference's and the text's `<|audio|>`
        // sentinels (each the last row of its prompt span).
        let mut overlay = Vec::new();
        if let Some(row) = s.speaker_row {
            overlay_pos.push(u32::try_from(row).map_err(|_| "speaker row overflows")?);
        }
        if let (Some(sp), Some(_)) = (&self.speaker, &c.maya2) {
            if let Some(r) = &s.reference {
                overlay_pos.push(r.ids.len() as u32 - 1);
            }
            overlay_pos.push(s.prompt_ids.len() as u32 - 1);
            for _ in &overlay_pos {
                overlay.extend_from_slice(&sp.anchor);
            }
        }
        let cfg = s.uncond_ids.as_ref().map(|ids| crate::serve::mux::CfgJob {
            uncond_overlay: Vec::new(),
            params: crate::text::sample::CfgParams {
                cfg_weight: self.cfg_weight,
                temperature: 0.0,
                min_p: 0.0,
                top_p: 1.0,
                repetition_penalty: 1.0,
            },
            history: Vec::new(),
            seed: None,
            uncond_ids: s.prompt_ids[..head]
                .iter()
                .copied()
                .chain(std::iter::repeat_n(0, ids.len()))
                .collect(),
            uncond_overlay_pos: rows(ids),
        });
        if s.enc_ids.is_some() && self.encoder.is_none() {
            return Err("no text encoder".into());
        }
        let encode: Option<crate::serve::mux::SpeechEncode> = (s.enc_ids.is_some()
            || s.reference.is_some())
        .then(|| {
            let (e, ids, uncond) = (
                self.encoder.clone(),
                s.enc_ids.clone(),
                s.uncond_ids.clone(),
            );
            let reference = s.reference.clone();
            let (k, base, codebook) = (c.frame_codes, c.audio_token_base, c.codebook);
            Box::new(
                move |sp: &mut crate::serve::mux::SpeechJob, prompt: &mut [u32]| {
                    if let Some(r) = &reference {
                        if r.conditioned {
                            let encoded = r.encoded()?;
                            sp.overlay = encoded
                                .speaker
                                .as_deref()
                                .ok_or("reference has no speaker overlay")?
                                .to_vec();
                            if r.frames > 0 {
                                let codes = encoded
                                    .codes
                                    .as_deref()
                                    .ok_or("reference has no codec frames")?;
                                let start = prompt
                                    .len()
                                    .checked_sub(r.frames)
                                    .ok_or("reference prompt is too short")?;
                                if codes.len() != r.frames * k
                                    || sp.frame_codes.len() != prompt.len() * k
                                    || codes.iter().any(|&v| v >= codebook)
                                {
                                    return Err(
                                        "reference code layout disagrees with the prompt".into()
                                    );
                                }
                                sp.frame_codes[start * k..].copy_from_slice(codes);
                            }
                            return Ok(());
                        }
                        let (codes, first) = r.frame_codes(k)?;
                        let rows = r.ids.len()..r.ids.len() + r.frames;
                        for (t, &c0) in prompt[rows.clone()].iter_mut().zip(&first) {
                            *t = base + c0;
                        }
                        if let Some(cfg) = sp.cfg.as_mut() {
                            for (t, &c0) in cfg.uncond_ids[rows].iter_mut().zip(&first) {
                                *t = base + c0;
                            }
                        }
                        sp.frame_codes = codes;
                    }
                    let (Some(e), Some(ids)) = (e, ids) else {
                        return Ok(());
                    };
                    // Each text segment is its own encoder launch (positions from 0).
                    let lead = match &ref_ids {
                        Some(r) => e.encode(r)?,
                        None => Vec::new(),
                    };
                    sp.overlay = [lead.as_slice(), &e.encode(&ids)?].concat();
                    if let (Some(ids), Some(cfg)) = (uncond, sp.cfg.as_mut()) {
                        cfg.uncond_overlay = [lead.as_slice(), &e.encode(&ids)?].concat();
                    }
                    Ok(())
                },
            ) as crate::serve::mux::SpeechEncode
        });
        let cfg_scale = cfg.as_ref().map(|_| 1.0 + self.cfg_weight);
        {
            // The reference keeps temperature and top-p as bf16 tensors (0.7 -> 0.69921875).
            let bf16 = |x: f32| {
                let b = x.to_bits();
                f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
            };
            // Breeze's reference samples with f32 parameters.
            let param = |x: f32| if c.breeze_sampler { x } else { bf16(x) };
            // The depth samplers divide by temperature: greedy is top-k 1 at temperature 1.
            let greedy = self.temperature <= 0.0;
            Ok(Some(Box::new(crate::serve::mux::SpeechJob {
                device_frames: self.device_frames,
                batch_delivery: c.batch_delivery
                    && self.streaming
                    && crate::config::RuntimeConfig::process_get().tick_delivery,
                cfg,
                slot_prefix_reuse: c.slot_prefix_reuse,
                overlay,
                overlay_pos,
                encode,
                frame_codes: dual.as_ref().map_or(Vec::new(), |l| l.codes.clone()),
                frame_codes_per_row: c.frame_codes,
                feed: dual.as_ref().map(|l| (l.feed.clone(), l.tail)),
                depth: Some(crate::serve::mux::DepthJob {
                    seed: s.gen.seed.unwrap_or(self.seed),
                    temperature: if greedy { 1.0 } else { param(self.temperature) },
                    top_p: param(self.top_p),
                    top_k: if greedy {
                        1
                    } else if self.top_k == 0 {
                        c.top_k as u32
                    } else {
                        self.top_k as u32
                    },
                    ras_temperature: bf16(1.0),
                    ras_top_p: bf16(0.9),
                    repetition_penalty: if c.breeze_sampler {
                        self.repetition_penalty
                    } else {
                        1.0
                    },
                    cfg_scale,
                    semantic_base: c.audio_token_base,
                    stop_class: c.codebook,
                    stop_token: c.stops[0],
                    maya2: c.maya2.as_ref().map(|_| self.maya2_plan(&s.text, s.k)),
                }),
                maya2_memory: self.speaker.as_ref().map(|sp| Arc::clone(&sp.memory)),
                ..Default::default()
            })))
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
        let Some(halves) = split_halves(&self.segs[k].text, &self.segs[k].native) else {
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
        let mut made: Vec<Segment> = halves
            .iter()
            .map(|t| {
                let mut s = self.maker.make(k, &t.text);
                s.native = t.native.clone();
                s
            })
            .collect();
        made[0].pause_before = self.segs[k].pause_before;
        made.last_mut().unwrap().explicit_tail = self.segs[k].explicit_tail;
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
        let speech = self.maker.speech(&s)?;
        let (tx, rx) = stream_mod::channel();
        let opts = crate::serve::mux::JobOpts {
            session_end: false,
            class: self.class,
            raw_tokens: true,
            speech,
            session: None,
            turn: None,
            continuing: false,
            prefix: None,
            tenant: None,
            round: 0,
            cached: None,
            mm: None,
            export: None,
            import: None,
            prompt_logprobs: false,
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
fn split_halves(
    text: &str,
    native: &[super::tags::Native],
) -> Option<Vec<super::tags::TextSegment>> {
    let n = text.chars().count();
    if n < SPLIT_MIN_CHARS {
        return None;
    }
    let parts = super::tags::segment(text, n.div_ceil(2) + n / 8, native);
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
    reference: Option<&RefClip>,
    reserved: &mut Option<super::codec::StreamSlot>,
) -> Result<Vec<f32>, String> {
    if model.contract.conditioning.is_none() {
        return decode_plain(model, codes, frames, seed).await;
    }
    let reference = reference
        .filter(|r| r.conditioned)
        .ok_or("conditioned request has no reference")?;
    let input = if reference.frames > 0 {
        let prefix = reference.codes()?;
        let mut joined = Vec::with_capacity(prefix.len() + codes.len());
        for &v in prefix {
            joined.push(i32::try_from(v).map_err(|_| "reference code overflows")?);
        }
        joined.extend_from_slice(codes);
        joined
    } else {
        codes.to_vec()
    };
    let total = frames
        .checked_add(reference.frames)
        .ok_or("codec frame count overflows")?;
    let mut slot = reserved
        .take()
        .ok_or("conditioned request lost its codec slot")?;
    slot.restart();
    let (slot, mut pcm) = model
        .codec
        .decode_stateful(slot, input.into(), total, Urgency::Whole)
        .await?;
    *reserved = Some(slot);
    if reference.frames > 0 {
        let crop = reference_crop(reference.frames, total, pcm.len())?;
        pcm.copy_within(crop.., 0);
        pcm.truncate(pcm.len() - crop);
    }
    Ok(pcm)
}

fn reference_crop(
    reference_frames: usize,
    total_frames: usize,
    samples: usize,
) -> Result<usize, String> {
    if total_frames == 0 || reference_frames > total_frames {
        return Err("invalid codec reference crop".into());
    }
    // Preserve the source's floating division, multiplication and truncation (including one-sample rounding).
    Ok((reference_frames as f64 / total_frames as f64 * samples as f64) as usize)
}

async fn decode_plain(
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

fn append_completed_pcm(
    codes: &[i32],
    output: &mut Option<Vec<f32>>,
    frame: &stream_mod::SpeechPcm,
    samples: usize,
) -> crate::Result<()> {
    if !codes.is_empty() || frame.len() != samples {
        return Err(crate::RuntimeError::Rejected(
            "mixed or invalid codec completion".into(),
        ));
    }
    output
        .get_or_insert_with(Vec::new)
        .extend(frame.iter().map(|&s| f32::from(s) / 32767.0));
    Ok(())
}

/// Drain promptly: holding a pinned tick slab until the utterance ends exhausts its bounded pool.
async fn collect_audio(
    c: &SpeechContract,
    mut rx: stream_mod::ChunkReceiver,
    ticket: &Ticket,
) -> crate::Result<(Vec<i32>, Option<Vec<f32>>, usize)> {
    let (mut codes, mut pcm, mut n) = (Vec::new(), None, 0);
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { id, mut frame, .. } => {
                n += 1;
                if let Some(pcm_frame) = stream_mod::SpeechFrame::take_pcm(&mut frame) {
                    append_completed_pcm(&codes, &mut pcm, &pcm_frame, c.frame_samples)?;
                    ticket.frame();
                } else {
                    let before = codes.len();
                    let complete = c.push_codes(&mut codes, id, frame.as_deref());
                    if pcm.is_some() && codes.len() != before {
                        return Err(crate::RuntimeError::Rejected(
                            "mixed codec completion paths".into(),
                        ));
                    }
                    if complete {
                        ticket.frame();
                    }
                }
            }
            StreamChunk::Done { .. } => break,
            StreamChunk::Err(e) => return Err(e),
        }
    }
    Ok((codes, pcm, n))
}

/// The emission plan for a stream holding `n` complete frames of which `emitted` are sent:
/// `Some((window_start, window_end, emit_to))`, or `None` when nothing new is ready.
///
/// Every window carries `window - lookahead - 1` frames of left context and `lookahead` of right
/// context (the one-frame window's shape). After the first audio, frames go out `chunk` at a time
/// (`ramp`: `min(emitted, chunk)`, so the windows right after the first audio wait for fewer frames):
/// a window costs its codec capacity whatever it emits, and one frame per window made streaming
/// Veena c64 36% slower than whole-utterance decoding.
fn stream_step(
    n: usize,
    emitted: usize,
    done: bool,
    window: usize,
    lookahead: usize,
    chunk: usize,
    ramp: bool,
) -> Option<(usize, usize, usize)> {
    let upto = if done { n } else { n.saturating_sub(lookahead) };
    let chunk = if ramp {
        chunk.min(emitted.max(1))
    } else {
        chunk
    };
    let ready = upto > emitted && (done || emitted == 0 || upto - emitted >= chunk);
    ready.then(|| {
        (
            emitted.saturating_sub(window.saturating_sub(lookahead + 1)),
            n.min(upto + lookahead),
            upto,
        )
    })
}

/// Frames the window after a first audio over `n` frames waits for: `chunk`, or ramped what the
/// first window emits (`n - lookahead`, however many frames arrived together), at most `chunk`:
/// [`stream_step`]'s chunk at that `emitted`.
fn next_window_chunk(n: usize, lookahead: usize, chunk: usize, ramp: bool) -> usize {
    if ramp {
        n.saturating_sub(lookahead).clamp(1, chunk)
    } else {
        chunk
    }
}

/// New frames the next stateful codec step of a stream holding `n` complete frames, `emitted`
/// of them sent, decodes: every committed frame up to `cap`; a segment's first step waits for
/// `first` frames until the segment is done. `urgent`: playback runs dry before another frame
/// and its step can arrive, so a started stream steps with what it has.
fn step_plan(
    n: usize,
    emitted: usize,
    done: bool,
    first: usize,
    cap: usize,
    urgent: bool,
) -> Option<usize> {
    let new = n.saturating_sub(emitted);
    let ready = done || (urgent && emitted > 0) || new >= step_need(emitted, first, cap);
    (new > 0 && ready).then(|| new.min(cap))
}

/// Whether a playing stream cannot wait for one more frame: the audio it holds ahead of playback
/// (`due` = when the sent audio has played out) is under one measured frame period plus the
/// codec step that follows it (`step_s`, with margin). Waiting for as many frames as were emitted
/// only holds while frames come at least twice as fast as they play.
/// When the frames [`step_urgent`] counts began to arrive. The count restarts with every segment
/// and retry, so the clock restarts with it: a request-old clock over a fresh count reads as a
/// frame period of seconds and turns every step urgent.
#[derive(Default)]
struct FramePace(Option<Instant>);

impl FramePace {
    fn frame(&mut self) {
        self.0.get_or_insert_with(Instant::now);
    }

    fn restart(&mut self) {
        self.0 = None;
    }

    fn since(&self) -> Option<Instant> {
        self.0
    }
}

fn step_urgent(
    due: Option<Instant>,
    first_frame_at: Option<Instant>,
    n: usize,
    step_s: f64,
) -> bool {
    let (Some(due), Some(t0)) = (due, first_frame_at) else {
        return false;
    };
    if n < 2 {
        return false;
    }
    let per_frame = t0.elapsed().as_secs_f64() / (n - 1) as f64;
    let lead = due.saturating_duration_since(Instant::now()).as_secs_f64();
    lead < urgent_lead(per_frame, step_s)
}

/// The playback lead under which a step is urgent: one frame period plus the codec step that
/// follows it (with margin), and never more than two frame periods, so a stream whose frames come
/// fast steps no earlier than that.
fn urgent_lead(per_frame: f64, step_s: f64) -> f64 {
    (per_frame + 1.5 * step_s + 0.010).min(2.0 * per_frame)
}

/// New frames [`step_plan`]'s next step waits for (short of the segment's end).
fn step_need(emitted: usize, first: usize, cap: usize) -> usize {
    // A step costs about the same at 1 frame as at `cap`: after the first, each step waits for as
    // many new frames as were already emitted (their audio covers producing them), up to `cap`.
    if emitted == 0 {
        first
    } else {
        emitted.min(cap)
    }
}

enum SegMsg {
    Start {
        pause: Option<u32>,
        explicit_tail: bool,
    },
    Frame(
        super::codec::Codes,
        Option<stream_mod::SpeechPcm>,
        Option<tokio::sync::OwnedSemaphorePermit>,
    ),
    /// The current segment attempt generated its last frame. A multi-segment drain then waits for
    /// the emitter's verdict on it before it starts anything else.
    End,
    /// The current segment was mute or droned and generates again: drop its frames.
    Retry,
    Err(String),
}

#[derive(Default)]
struct PcmFrames {
    pending: std::collections::VecDeque<(
        stream_mod::SpeechPcm,
        Option<tokio::sync::OwnedSemaphorePermit>,
    )>,
    received: usize,
}

impl PcmFrames {
    fn push(
        &mut self,
        pcm: Option<stream_mod::SpeechPcm>,
        permit: Option<tokio::sync::OwnedSemaphorePermit>,
        samples: usize,
    ) -> Result<(), String> {
        if let Some(pcm) = pcm {
            if pcm.len() != samples {
                return Err("codec tick PCM length disagrees with frame size".into());
            }
            self.received = self
                .received
                .checked_add(1)
                .ok_or("PCM frame count overflow")?;
            self.pending.push_back((pcm, permit));
        }
        Ok(())
    }

    fn clear(&mut self) {
        self.pending.clear();
        self.received = 0;
    }
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
    mut out_pcm: super::OutPcm,
    out: tokio::sync::mpsc::Sender<Result<Vec<u8>, std::io::Error>>,
    head: tokio::sync::oneshot::Sender<axum::http::HeaderMap>,
    seed: u64,
    initial_slot: Option<super::codec::StreamSlot>,
    t_arrive: Instant,
    mut run: crate::serve::turns::StageRun,
    first_audio: Option<Arc<crate::serve::mux::FirstAudio>>,
    trailing_pause: Option<u32>,
) {
    use std::sync::atomic::Ordering::SeqCst;
    // However the stream ends, the decode loop stops handing it the device.
    struct Heard(Option<Arc<crate::serve::mux::FirstAudio>>);
    impl Drop for Heard {
        fn drop(&mut self) {
            if let Some(fa) = &self.0 {
                fa.hear();
            }
        }
    }
    let audible = Heard(first_audio);
    struct ActiveGuard(Option<Arc<crate::obs::Metrics>>);
    impl Drop for ActiveGuard {
        fn drop(&mut self) {
            if let Some(m) = &self.0 {
                m.serving
                    .tts
                    .active_streams
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    if let Some(m) = run.metrics() {
        m.serving
            .tts
            .active
            .store(true, std::sync::atomic::Ordering::Relaxed);
        m.serving
            .tts
            .active_streams
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let _stream_guard = ActiveGuard(run.metrics().cloned());
    let mut head = Some(head);
    let c = model.contract.clone();
    // Playback, not model, rate: see `guided_stream_task`.
    let sr = f64::from(c.sample_rate) * f64::from(out_pcm.speed());
    let (fc, fs) = (c.frame_codes, c.frame_samples);
    let n_segs = later.segs.len();
    let reference = later
        .maker
        .reference
        .as_ref()
        .filter(|r| r.conditioned && r.frames > 0)
        .cloned();
    // The request's first step size, as its first-audio handoff was told (`SpeechModel::first_frames`).
    let first_frames = audible
        .0
        .as_ref()
        .map_or_else(|| model.contract.first_frames(), |fa| fa.next().max(1));
    let pcm_queue_capacity = rx.max_capacity().max(first_frames);
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
            let mut pcm_budget: Option<Arc<tokio::sync::Semaphore>> = None;
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
                    if (later.segs[k].pause_before.is_some() || later.segs[k].explicit_tail)
                        && ftx
                            .send(SegMsg::Start {
                                pause: later.segs[k].pause_before,
                                explicit_tail: later.segs[k].explicit_tail,
                            })
                            .is_err()
                    {
                        return;
                    }
                    let tag = attempt_tag(k, attempt);
                    lm.segment(k);
                    let (mut codes, mut frames) = (Vec::new(), 0);
                    loop {
                        match rx.recv().await {
                            Some(StreamChunk::Token { id, mut frame, .. }) => {
                                let pcm = stream_mod::SpeechFrame::take_pcm(&mut frame);
                                let frame = if pcm.is_some() {
                                    Some(super::codec::Codes::default())
                                } else {
                                    c.take_frame(&mut codes, id, frame)
                                };
                                if let Some(frame) = frame {
                                    let permit = if pcm.is_some() {
                                        let budget = pcm_budget.get_or_insert_with(|| {
                                            Arc::new(tokio::sync::Semaphore::new(
                                                pcm_queue_capacity,
                                            ))
                                        });
                                        match budget.clone().acquire_owned().await {
                                            Ok(permit) => Some(permit),
                                            Err(_) => return,
                                        }
                                    } else {
                                        None
                                    };
                                    frames += 1;
                                    lm.frame();
                                    if ftx.send(SegMsg::Frame(frame, pcm, permit)).is_err() {
                                        return;
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
    let (mut frames, mut emitted, mut first) = (super::codec::Codes::default(), 0usize, None);
    let mut pcm_frames = PcmFrames::default();
    let mut tick_audio = false;
    let mut pcm_completion = None;
    // Segment and attempt, frames the earlier segments emitted, and whether this attempt's verdict
    // is out (the next message is a retry, the next segment, or the end).
    let (mut seg, mut attempt, mut total, mut joins, mut judged) =
        (0usize, 0usize, 0usize, Vec::new(), false);
    // Samples sent and when the first went out (the client's playback clock).
    let (mut sent, mut playing): (usize, Option<Instant>) = (0, None);
    let mut reported = 0usize;
    let mut joiner = super::Joins::new(model.contract.sample_rate);
    // Dropping the rest of the current attempt (cut short).
    let mut skipping = false;
    let mut first_frame_at: Option<Instant> = None;
    // The deadline rule's clock: restarts with the frame count (segment, retry).
    let mut pace = FramePace::default();
    // (when, frames seen) of the last frame-period sample fed to the model's estimate.
    let mut period_mark: Option<(Instant, usize)> = None;
    // First-audio trace (`plow::tts::first_audio` at debug): codec steps until audible, the last one's time.
    let (mut first_steps, mut first_step_ms) = (0u32, 0.0f64);
    // A stateful codec decodes each new frame once on this stream's slot; without a free slot
    // the stream decodes windows.
    let mut slot = initial_slot;
    let mut primed = false;
    let mut windowed = model.codec.step_frames == 0;
    let stream_deadline = crate::config::RuntimeConfig::get().tts_stream_deadline;
    let step_cap = crate::config::RuntimeConfig::get()
        .tts_stream_step_frames
        .map_or(model.codec.step_frames, |n| {
            n.clamp(1, model.codec.step_frames.max(1))
        });
    // This stream's codec step time (seconds, running average): what an urgent step still costs.
    let mut step_s = 0.02f64;
    loop {
        let mut done = false;
        let msg = frx.recv().await;
        if judged {
            judged = false;
            match msg {
                None => break,
                Some(SegMsg::Retry) => {
                    if let Some(m) = run.metrics() {
                        m.serving
                            .tts
                            .watchdog_recovers
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    (frames, emitted, skipping) = (super::codec::Codes::default(), 0, false);
                    pcm_frames.clear();
                    tick_audio = false;
                    pcm_completion = None;
                    pace.restart();
                    period_mark = None;
                    attempt += 1;
                    joiner.next_segment();
                    if let Some(s) = slot.as_mut() {
                        s.restart();
                    }
                    primed = false;
                    continue;
                }
                Some(SegMsg::Err(_)) => {}
                Some(_) => {
                    total += emitted;
                    joins.push(sent);
                    joiner.next_segment();
                    (frames, emitted, skipping) = (super::codec::Codes::default(), 0, false);
                    pcm_frames.clear();
                    tick_audio = false;
                    pcm_completion = None;
                    pace.restart();
                    period_mark = None;
                    (seg, attempt) = (seg + 1, 0);
                    if let Some(s) = slot.as_mut() {
                        s.restart();
                    }
                    primed = false;
                }
            }
        }
        match msg {
            None => done = true,
            Some(SegMsg::End) => done = true,
            Some(SegMsg::Err(e)) => {
                if let Some(m) = run.metrics() {
                    m.serving
                        .tts
                        .errors
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return drop(out.send(Err(std::io::Error::other(e))).await);
            }
            Some(SegMsg::Retry) => {}
            Some(SegMsg::Start {
                pause,
                explicit_tail,
            }) => {
                if attempt == 0 {
                    if let Some(ms) = pause {
                        let mut pcm = Vec::new();
                        joiner.pause(ms, out_pcm.speed(), &mut pcm);
                        let mut bytes = Vec::new();
                        out_pcm.push(&pcm, &mut bytes);
                        sent += pcm.len();
                        if !bytes.is_empty() {
                            ticket.first_audio();
                            ticket.sent(sent as f64 / sr);
                            run.audio(sent - reported, sr);
                            reported = sent;
                            first.get_or_insert_with(|| t_arrive.elapsed());
                            playing.get_or_insert_with(Instant::now);
                            if let Some(h) = head.take() {
                                let _ = h.send(run.headers());
                            }
                            if out.send(Ok(bytes)).await.is_err() {
                                return;
                            }
                        }
                    }
                }
                joiner.tagged_tail(explicit_tail);
                continue;
            }
            Some(SegMsg::Frame(f, pcm, permit)) => {
                if pcm_completion.is_some_and(|previous| previous != pcm.is_some()) {
                    return drop(
                        out.send(Err(std::io::Error::other("mixed codec completion paths")))
                            .await,
                    );
                }
                tick_audio = pcm.is_some();
                pcm_completion = Some(tick_audio);
                if let Err(e) = pcm_frames.push(pcm, permit, fs) {
                    return drop(out.send(Err(std::io::Error::other(e))).await);
                }
                if tick_audio {
                    slot = None;
                    windowed = false;
                }
                frames.extend(f, fc);
                while let Ok(m) = frx.try_recv() {
                    match m {
                        SegMsg::Frame(f, pcm, permit) => {
                            if pcm.is_some() != tick_audio {
                                return drop(
                                    out.send(Err(std::io::Error::other(
                                        "mixed codec completion paths",
                                    )))
                                    .await,
                                );
                            }
                            if let Err(e) = pcm_frames.push(pcm, permit, fs) {
                                return drop(out.send(Err(std::io::Error::other(e))).await);
                            }
                            frames.extend(f, fc);
                        }
                        SegMsg::End => {
                            done = true;
                            break;
                        }
                        SegMsg::Retry => {}
                        SegMsg::Start { .. } => {
                            return drop(
                                out.send(Err(std::io::Error::other("unexpected segment boundary")))
                                    .await,
                            );
                        }
                        SegMsg::Err(e) => {
                            return drop(out.send(Err(std::io::Error::other(e))).await)
                        }
                    }
                }
                first_frame_at.get_or_insert_with(Instant::now);
                pace.frame();
                if slot.is_some() || tick_audio {
                    let seen = if tick_audio {
                        pcm_frames.received
                    } else {
                        frames.len() / fc.max(1)
                    };
                    match period_mark {
                        Some((t, at)) if seen >= at + 8 => {
                            model.observe_frame_period(
                                t.elapsed().as_secs_f64() / (seen - at) as f64,
                            );
                            period_mark = Some((Instant::now(), seen));
                        }
                        // A new segment restarts the frame count.
                        Some((_, at)) if seen >= at => {}
                        _ => period_mark = Some((Instant::now(), seen)),
                    }
                }
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
        // Windowed streams ramp their chunk after the first audio (1, 2, .. `chunk` frames: the
        // packet's `stream.chunk_ramp`, `PLOW_TTS_CHUNK_RAMP` overrides), so the second window
        // needs fewer frames and the first audio is held for fewer.
        let ramp = crate::config::RuntimeConfig::get()
            .tts_chunk_ramp
            .unwrap_or_else(|| model.codec.parameters.get("stream.chunk_ramp") == Some(&1));
        let n = if tick_audio {
            pcm_frames.received
        } else {
            frames.len() / fc
        };
        if !tick_audio && !windowed && slot.is_none() && n > emitted && !skipping {
            slot = model.codec.acquire_slot();
            windowed = slot.is_none();
            if windowed {
                // Windows follow their own plan: no per-step handoff.
                if let Some(fa) = &audible.0 {
                    fa.hear();
                }
            }
        }
        if !primed && slot.is_some() && n > emitted && !skipping {
            if let Some(reference) = &reference {
                let prefix = match reference.codes().and_then(|codes| {
                    codes
                        .iter()
                        .map(|&v| i32::try_from(v).map_err(|_| "reference code overflows".into()))
                        .collect::<Result<Vec<_>, String>>()
                }) {
                    Ok(prefix) => prefix,
                    Err(e) => return drop(out.send(Err(std::io::Error::other(e))).await),
                };
                match model
                    .codec
                    .prime_codes(
                        slot.take().unwrap(),
                        prefix.into(),
                        reference.frames,
                        Urgency::First,
                        None,
                    )
                    .await
                {
                    Ok(ready) => slot = Some(ready),
                    Err(e) => return drop(out.send(Err(std::io::Error::other(e))).await),
                }
            }
            primed = true;
        }
        // A short first lookahead buys time to first audio, but the next window needs the full
        // lookahead: hold the first audio until it covers producing those frames at the measured
        // frame rate (a stream near real time otherwise underruns right after its first chunk).
        if windowed && first_audio && !done && lookahead < model.codec.lookahead {
            let Some(t0) = first_frame_at.filter(|_| n >= 2) else {
                continue;
            };
            let per_frame = t0.elapsed().as_secs_f64() / (n - 1) as f64;
            let audio_per_frame = fs as f64 / sr;
            // Frames the next window waits for: its chunk (ramped: what this one emits, at most
            // `chunk`) plus the lookahead this window lacks.
            let next_chunk = next_window_chunk(n, lookahead, chunk, ramp);
            let gap = (next_chunk + model.codec.lookahead - lookahead) as f64;
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
        // Windowed: one window per message. Stateful: steps until every committed frame is out.
        while !skipping {
            // Until audio is heard: a step whose output the onset trim drops leaves the next one
            // the stream's first audio.
            let urgency = if first.is_none() {
                Urgency::First
            } else {
                Urgency::Stream
            };
            let due = playing.map(|t| t + std::time::Duration::from_secs_f64(sent as f64 / sr));
            // (PCM, its first fresh frame, frames emitted after it, frames decoded)
            let mut direct_pcm = Vec::new();
            let decoded = if tick_audio {
                if emitted == 0 && !done && n < first_frames {
                    break;
                }
                let mut pcm = Vec::with_capacity(pcm_frames.pending.len() * fs);
                while let Some((frame, permit)) = pcm_frames.pending.pop_front() {
                    pcm.extend(frame.iter().map(|&sample| sample as f32 / 32767.0));
                    direct_pcm.push((frame, permit));
                }
                if pcm.is_empty() {
                    break;
                }
                if pcm.len() != (n - emitted) * fs {
                    Err("codec tick PCM length disagrees with frame count".into())
                } else {
                    Ok((pcm, 0, n, n - emitted))
                }
            } else if let Some(slot) = slot.as_mut() {
                let urgent = stream_deadline && step_urgent(due, pace.since(), n, step_s);
                let Some(k) = step_plan(n, emitted, done, first_frames, step_cap, urgent) else {
                    break;
                };
                let codes = frames.slice(emitted * fc..(emitted + k) * fc, fc);
                let t_codec = (crate::obs::ttft::on() && emitted == 0).then(|| {
                    if let Some(t) = first_frame_at {
                        crate::obs::ttft::SPEECH_FRAMES.add(t.elapsed().as_nanos() as u64);
                    }
                    Instant::now()
                });
                let t_step = Instant::now();
                let r = model.codec.step_codes(slot, codes, k, urgency, due).await;
                let took = t_step.elapsed().as_secs_f64();
                step_s += 0.25 * (took - step_s);
                if first.is_none() {
                    first_step_ms = took * 1e3;
                    first_steps += 1;
                }
                if let Some(t) = t_codec {
                    crate::obs::ttft::SPEECH_CODEC.add(t.elapsed().as_nanos() as u64);
                }
                r.map(|pcm| (pcm, 0, emitted + k, k))
            } else {
                let Some((s, e, upto)) =
                    stream_step(n, emitted, done, model.codec.window, lookahead, chunk, ramp)
                else {
                    break;
                };
                let window = frames.slice(s * fc..e * fc, fc);
                let wseed = seed ^ segment_salt(seg) ^ (emitted as u64).wrapping_mul(0x9E37_79B9);
                model
                    .codec
                    .decode_codes(window, e - s, wseed, urgency, due)
                    .await
                    .map(|pcm| (pcm, emitted - s, upto, e - s))
            };
            match decoded {
                Ok((pcm, skip, upto, decoded)) => {
                    let Some(fresh) = pcm.get(skip * fs..(skip + upto - emitted) * fs) else {
                        return drop(
                            out.send(Err(std::io::Error::other(short_window(pcm.len(), decoded))))
                                .await,
                        );
                    };
                    let joined;
                    let fresh = if n_segs == 1 && !joiner.has_tagged_boundary() {
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
                    if slot.is_some() || tick_audio {
                        frames.release(emitted..upto);
                        if let Some(fa) = &audible.0 {
                            fa.set_next(
                                upto + if tick_audio {
                                    1
                                } else {
                                    step_need(upto, first_frames, step_cap)
                                },
                            );
                        }
                    }
                    emitted = upto;
                    let mut bytes = Vec::new();
                    if !fresh.is_empty() {
                        let trimmed = out_pcm.trimmed();
                        if tick_audio && n_segs == 1 {
                            for (frame, _) in &direct_pcm {
                                out_pcm.push_i16(frame, &mut bytes);
                            }
                        } else {
                            out_pcm.push(fresh, &mut bytes);
                        }
                        // Trimmed leading silence is never played: it neither starts playback
                        // nor counts as audio ahead of it (held samples count once sent).
                        sent += fresh.len();
                        sent -= out_pcm.trimmed() - trimmed;
                    }
                    if !bytes.is_empty() {
                        if first.is_none() {
                            if let Some(fa) = &audible.0 {
                                fa.hear();
                            }
                            ticket.first_audio();
                            crate::obs::ttft::dump(t_arrive.elapsed().as_nanos() as u64, 0);
                            tracing::debug!(
                                target: "plow::tts::first_audio",
                                audio_ms = t_arrive.elapsed().as_secs_f64() * 1e3,
                                frame_ms = first_frame_at
                                    .map(|t| t.duration_since(t_arrive).as_secs_f64() * 1e3),
                                frames = n,
                                steps = first_steps,
                                last_step_ms = first_step_ms,
                                "first audio"
                            );
                        }
                        first.get_or_insert_with(|| t_arrive.elapsed());
                        playing.get_or_insert_with(Instant::now);
                        ticket.sent(sent as f64 / sr);
                        run.audio(sent - reported, sr);
                        reported = sent;
                        if let Some(h) = head.take() {
                            let _ = h.send(run.headers());
                        }
                        if out.send(Ok(bytes)).await.is_err() {
                            if let Some(m) = run.metrics() {
                                m.serving
                                    .tts
                                    .cancelled
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            return; // client gone: dropping frx ends the drain, which cancels the slot
                        }
                        if let (Some(fa), Some(started)) =
                            (audible.0.as_ref().and_then(|fa| fa.frame_budget()), playing)
                        {
                            fa.set_playback_deadline(
                                started + std::time::Duration::from_secs_f64(sent as f64 / sr),
                                std::time::Duration::from_secs_f64(step_s),
                            );
                        }
                    }
                }
                Err(e) => {
                    if let Some(m) = run.metrics() {
                        m.serving
                            .tts
                            .errors
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    return drop(out.send(Err(std::io::Error::other(e))).await);
                }
            }
            if windowed {
                break;
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
    let mut tail = Vec::new();
    if let Some(ms) = trailing_pause {
        let mut pcm = Vec::new();
        joiner.pause(ms, out_pcm.speed(), &mut pcm);
        out_pcm.push(&pcm, &mut tail);
    }
    out_pcm.finish(&mut tail);
    if !tail.is_empty() {
        // A reply shorter than the onset trim's window: its held samples are its first audio.
        if first.is_none() {
            ticket.first_audio();
            first = Some(t_arrive.elapsed());
        }
        run.audio(sent - reported, sr);
        if let Some(h) = head.take() {
            let _ = h.send(run.headers());
        }
        if out.send(Ok(tail)).await.is_err() {
            return;
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
    #[cfg(feature = "cuda")]
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "GPU: existing-packet tag-map gate server, run through gpuq"]
    async fn packet_tag_map_server() {
        use crate::{device::cuda::CudaBackend, exec::ExecutorSet, orch::Registry};
        let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
        let assets = std::path::PathBuf::from(std::env::var_os("TTS_TAG_GATE_ASSETS").unwrap());
        let path = std::env::var_os("TTS_TAG_GATE_MAPS").unwrap();
        let profile = std::env::var("TTS_TAG_GATE_PROFILE").unwrap();
        let maps: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let fixture = &maps[&profile];
        let pipe: plow_asset::packet_pipeline::PacketPipeline =
            serde_json::from_value(fixture["pipeline"].clone()).unwrap();
        let map = super::super::tags::load(&pipe).unwrap().unwrap();
        let port: u16 = std::env::var("TTS_TAG_GATE_PORT").unwrap().parse().unwrap();
        let registry = Registry::new();
        let slug = registry.load(&assets, None).unwrap();
        let backend = Arc::new(CudaBackend::new(0).unwrap());
        let executors = Arc::new(ExecutorSet::bringup(backend.clone()).unwrap());
        let state = Arc::new(AppState::new(registry, executors));
        let manager = Arc::new(
            crate::serve::manager::ModelManager::new(
                backend,
                &state,
                crate::serve::mux::MuxConfig {
                    max_hold_ms: 0.0,
                    ..Default::default()
                },
                vec![(
                    slug.clone(),
                    assets.clone(),
                    crate::asset::serve::checkpoint_dir(&assets),
                )],
                None,
            )
            .unwrap(),
        );
        state.set_slug_group(&slug, 0);
        state.install_turns_per_group(&[(None, true)]);
        state.install_managers(vec![manager.clone()]);
        manager.load_initial().await.unwrap();
        super::bind(&state, &slug).unwrap();
        let tok = state.registry.get(&slug).unwrap().tokenizer().clone();
        let guided = {
            let mut models = super::guided_models().lock();
            if let Some(Some(model)) = models.get_mut(&(assets.clone(), 0)) {
                map.validate_binding(false, |text| {
                    tok.encode_with_special_tokens(text, false).len() == 1
                })
                .unwrap();
                Arc::get_mut(model).unwrap().set_test_tags(map.clone());
                true
            } else {
                false
            }
        };
        if !guided {
            let mut models = super::speech_models().lock();
            let model =
                Arc::get_mut(models.get_mut(&(assets, 0)).unwrap().as_mut().unwrap()).unwrap();
            map.validate_binding(
                model.contract.instructions.is_some()
                    || model
                        .contract
                        .dual_stream
                        .as_ref()
                        .is_some_and(|d| d.instructions_template.is_some()),
                |text| tok.encode_with_special_tokens(text, false).len() == 1,
            )
            .unwrap();
            model.contract.tags = Some(map);
            let languages: Vec<String> =
                serde_json::from_value(fixture["languages"].clone()).unwrap();
            if !languages.is_empty() {
                model.contract.languages = languages;
            }
            model.spec = OnceLock::new();
        }
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
            .await
            .unwrap();
        axum::serve(listener, crate::serve::app(state))
            .await
            .unwrap();
    }

    #[test]
    fn speech_tag_modes_and_headers_are_explicit() {
        use super::super::tags::{Mode, Report};
        for (value, expected) in [
            ("auto", Mode::Auto),
            ("strict", Mode::Strict),
            ("off", Mode::Off),
        ] {
            let req: SpeechRequest = serde_json::from_value(
                serde_json::json!({"model":"m","voice":"v","input":"Hello","tags":value}),
            )
            .unwrap();
            assert_eq!(req.tags, expected);
        }
        for value in [
            serde_json::json!(null),
            serde_json::json!("bogus"),
            serde_json::json!(7),
            serde_json::json!([]),
            serde_json::json!({}),
            serde_json::json!(true),
            serde_json::json!(-7),
            serde_json::json!(1.5),
            serde_json::json!([{"nested": [null, "strict"]}]),
            serde_json::json!({"nested": [null, "off"]}),
        ] {
            let req: SpeechRequest = serde_json::from_value(
                serde_json::json!({"model":"m","voice":"v","input":"Hello","tags":value}),
            )
            .unwrap();
            assert_eq!(req.tags, Mode::Auto);
        }
        let req: SpeechRequest =
            serde_json::from_value(serde_json::json!({"model":"m","voice":"v","input":"Hello"}))
                .unwrap();
        assert_eq!(req.tags, Mode::Auto);
        let mut response = Response::new(Body::empty());
        stamp_tags(&mut response, &Report::default());
        assert!(!response.headers().contains_key("x-plow-tags-applied"));
        stamp_tags(
            &mut response,
            &Report {
                active: true,
                applied: vec!["pause".into()],
                dropped: vec!["laugh".into()],
            },
        );
        assert_eq!(response.headers()["x-plow-tags-applied"], "pause");
        assert_eq!(response.headers()["x-plow-tags-dropped"], "laugh");
    }

    #[tokio::test]
    async fn silence_only_reply_keeps_requested_duration_after_conversion() {
        let report = super::super::tags::Report {
            active: true,
            applied: vec!["pause".into()],
            dropped: Vec::new(),
        };
        for rate in [8000, 24000, 48000] {
            for speed in [0.5, 1.0, 1.14, 2.0] {
                let mut out = super::super::OutPcm::new(24000, rate, speed, false).unwrap();
                let reply = silence_response(&mut out, Some(600), 24000, &report);
                let bytes = axum::body::to_bytes(reply.into_body(), usize::MAX)
                    .await
                    .unwrap();
                assert!(bytes.iter().all(|&b| b == 0));
                let ms = bytes.len() as f64 / 2.0 / f64::from(rate) * 1000.0;
                assert!(
                    (ms - 600.0).abs() <= 80.0,
                    "rate={rate} speed={speed} duration={ms}"
                );
            }
        }
    }

    #[test]
    fn whole_completed_pcm_uses_audio_and_rejects_mixed_or_short_frames() {
        let bytes: Vec<_> = [-32767i16, -1, 0, 32767]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        let frame = stream_mod::SpeechPcm::from_buffer(bytes.into(), 0..8).unwrap();
        let mut output = None;
        append_completed_pcm(&[], &mut output, &frame, 4).unwrap();
        assert_eq!(output.as_ref().unwrap(), &[-1.0, -1.0 / 32767.0, 0.0, 1.0]);
        let original = output.clone();
        assert!(append_completed_pcm(&[1], &mut output, &frame, 4).is_err());
        assert_eq!(output, original);
        assert!(append_completed_pcm(&[], &mut output, &frame, 3).is_err());
        assert_eq!(output, original);
        append_completed_pcm(&[], &mut output, &frame, 4).unwrap();
        assert_eq!(output.as_ref().unwrap().len(), 8);
    }

    #[test]
    fn pcm_frames_count_without_host_codes_and_reset_between_attempts() {
        fn frame(samples: usize) -> stream_mod::SpeechPcm {
            let bytes: Vec<_> = (0..samples)
                .flat_map(|i| (i as i16).to_le_bytes())
                .collect();
            stream_mod::SpeechPcm::from_buffer(bytes.into(), 0..samples * 2).unwrap()
        }
        let mut frames = super::PcmFrames::default();
        frames.push(None, None, 4).unwrap();
        assert_eq!(frames.received, 0);
        frames.push(Some(frame(4)), None, 4).unwrap();
        let retained = frames.pending.pop_front().unwrap();
        frames.push(Some(frame(4)), None, 4).unwrap();
        assert_eq!(frames.received, 2);
        assert_eq!(frames.pending.len(), 1);
        assert!(frames.push(Some(frame(3)), None, 4).is_err());
        assert_eq!(frames.received, 2);
        assert_eq!(frames.pending.len(), 1);
        frames.clear();
        assert_eq!(frames.received, 0);
        frames.push(Some(frame(4)), None, 4).unwrap();
        assert_eq!(frames.received, 1);
        assert_eq!(&*retained.0, &[0, 1, 2, 3]);
        frames.received = usize::MAX;
        assert!(frames.push(Some(frame(4)), None, 4).is_err());
        assert_eq!(frames.pending.len(), 1);
    }

    #[test]
    fn pcm_budget_bounds_pending_and_consuming_frames_through_first_audio() {
        let channel_capacity = 1usize;
        let first_frames = 3usize;
        let budget = Arc::new(tokio::sync::Semaphore::new(
            channel_capacity.max(first_frames),
        ));
        let mut frames = super::PcmFrames::default();
        for _ in 0..first_frames {
            let pcm = stream_mod::SpeechPcm::from_buffer(vec![0; 8].into(), 0..8).unwrap();
            let permit = budget.clone().try_acquire_owned().unwrap();
            frames.push(Some(pcm), Some(permit), 4).unwrap();
        }
        assert_eq!(frames.received, first_frames);
        assert!(budget.clone().try_acquire_owned().is_err());
        let consuming = frames.pending.pop_front().unwrap();
        assert!(budget.clone().try_acquire_owned().is_err());
        drop(consuming);
        assert_eq!(budget.available_permits(), 1);
        frames.clear();
        assert_eq!(budget.available_permits(), first_frames);
        let pcm = stream_mod::SpeechPcm::from_buffer(vec![0; 6].into(), 0..6).unwrap();
        let permit = budget.clone().try_acquire_owned().unwrap();
        assert!(frames.push(Some(pcm), Some(permit), 4).is_err());
        assert_eq!(budget.available_permits(), first_frames);
        let pcm = stream_mod::SpeechPcm::from_buffer(vec![0; 8].into(), 0..8).unwrap();
        let permit = budget.clone().try_acquire_owned().unwrap();
        frames.push(Some(pcm), Some(permit), 4).unwrap();
        drop(frames);
        assert_eq!(budget.available_permits(), first_frames);
    }

    #[test]
    fn delivery_audio_coalesces_ready_bytes_and_preserves_a_following_error() {
        use std::task::{Context, Poll};
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let mut error = None;
        let mut cx = Context::from_waker(futures::task::noop_waker_ref());
        assert!(super::poll_audio(&mut rx, &mut error, &mut cx).is_pending());
        tx.try_send(Ok(vec![1, 2])).unwrap();
        tx.try_send(Ok(vec![3, 4])).unwrap();
        tx.try_send(Err(std::io::Error::other("broken"))).unwrap();
        tx.try_send(Ok(vec![5])).unwrap();
        assert!(matches!(super::poll_audio(&mut rx, &mut error, &mut cx),
            Poll::Ready(Some(Ok(bytes))) if bytes == [1, 2, 3, 4]));
        assert!(matches!(super::poll_audio(&mut rx, &mut error, &mut cx),
            Poll::Ready(Some(Err(e))) if e.to_string() == "broken"));
        assert!(matches!(super::poll_audio(&mut rx, &mut error, &mut cx),
            Poll::Ready(Some(Ok(bytes))) if bytes == [5]));
        drop(tx);
        assert!(matches!(
            super::poll_audio(&mut rx, &mut error, &mut cx),
            Poll::Ready(None)
        ));
    }
    use super::*;

    #[test]
    fn whole_reference_crop_keeps_source_floating_rounding() {
        assert!(reference_crop(1, 0, 0).is_err());
        assert!(reference_crop(2, 1, 1920).is_err());
        let mut rounded = 0;
        for reference in 1..=256 {
            for fresh in 1..=512 {
                let total = reference + fresh;
                let got = reference_crop(reference, total, total * 1920).unwrap();
                assert!(got == reference * 1920 || got + 1 == reference * 1920);
                rounded += usize::from(got != reference * 1920);
            }
        }
        assert!(rounded > 0);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn conditioned_prompts_finalize_overlay_and_reference_codes_on_the_engine_path() {
        use crate::text::tokenizer::{ByteTokenizer, Tokenize};
        let c =
            SpeechContract::from_pipeline(&super::super::tests::conditioned_pipeline()).unwrap();
        let tok: Arc<dyn Tokenize> = Arc::new(ByteTokenizer);
        let d = c.dual_stream.as_ref().unwrap();
        for transcript in [false, true] {
            for streaming in [false, true] {
                let (encoder, calls) =
                    super::super::ref_encoder::tests::counted_encoder(true, false);
                let codes: Vec<_> = (0..32).map(|v| v + 10).collect();
                let row: Arc<[f32]> = vec![0.25, 0.5, -1., 2.].into();
                let reference = Arc::new(RefClip {
                    lang: None,
                    ids: if transcript {
                        tok.encode("abcrefdefgh")
                    } else {
                        Vec::new()
                    },
                    pcm: vec![0.25, 0.5, 0.75],
                    frames: if transcript { 2 } else { 0 },
                    eos: None,
                    encoder: Arc::new(encoder),
                    conditioned: true,
                    encoding: OnceLock::from(Ok(Arc::new(
                        super::super::ref_encoder::ReferenceEncoding {
                            codes: transcript.then(|| codes.clone().into()),
                            speaker: Some(row.clone()),
                        },
                    ))),
                });
                let maker = SegmentMaker {
                    tok: tok.clone(),
                    contract: c.clone(),
                    voice: "default".into(),
                    language: Some("english".into()),
                    speaker: None,
                    instructions: None,
                    encoder: None,
                    reference: Some(reference.clone()),
                    cfg_weight: 0.,
                    max_tokens: None,
                    temperature: 0.9,
                    top_p: 1.,
                    repetition_penalty: 1.05,
                    top_k: 50,
                    seed: 17,
                    streaming,
                    device_frames: true,
                };
                for width in [1, 8, 32] {
                    for k in 0..width {
                        let text = format!("words{k}");
                        let segment = maker.make(k, &text);
                        let original_ids = segment.prompt_ids.clone();
                        let mut speech = maker.speech(&segment).unwrap().unwrap();
                        assert!(speech.slot_prefix_reuse && speech.overlay.is_empty());
                        let mut prompt = segment.prompt_ids.clone();
                        speech.encode.take().unwrap()(&mut speech, &mut prompt).unwrap();
                        assert_eq!(prompt, original_ids, "conditioning overwrote text ids");
                        assert_eq!(speech.overlay, &*row);
                        let expected = d
                            .layout_conditioned(
                                &tok.encode(&format!("abc{text}defgh")),
                                &[],
                                "english",
                                streaming,
                                transcript.then_some(plow_asset::dual_stream::Reference {
                                    text: &reference.ids,
                                    codes: &codes,
                                    codebook_size: c.codebook,
                                }),
                            )
                            .unwrap();
                        assert_eq!(prompt, expected.layout.ids);
                        assert_eq!(speech.overlay_pos, [expected.speaker_row as u32]);
                        assert_eq!(speech.frame_codes, expected.layout.codes);
                        assert_eq!(
                            speech.feed,
                            Some((expected.layout.feed, expected.layout.tail))
                        );
                    }
                }
                assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
            }
        }
    }

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
        let h = split_halves(t, &[]).unwrap();
        assert_eq!(h.len(), 2);
        assert_eq!(
            h.iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            t
        );
        assert!(split_halves("Too short to split again.", &[]).is_none());
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

    /// Stateful steps decode every committed frame once, in order, at most CAP per step; a
    /// segment's first step waits for FIRST frames unless the segment is done.
    #[test]
    fn first_audio_guard_prices_the_window_stream_step_takes_next() {
        const WINDOW: usize = 6;
        for (first_la, full_la) in [(1, 2), (0, 2), (2, 2)] {
            for chunk in 1..=4 {
                // n frames arrive together (coalesced) before the first audio.
                for n in first_la + 1..=first_la + 6 {
                    let (_, _, emitted) =
                        stream_step(n, 0, false, WINDOW, first_la, chunk, true).unwrap();
                    assert_eq!(emitted, n - first_la);
                    let want = next_window_chunk(n, first_la, chunk, true);
                    // The second window (full lookahead) fires at exactly emitted + want + la.
                    let fire = (n..n + 16)
                        .find(|&m| {
                            stream_step(m, emitted, false, WINDOW, full_la, chunk, true).is_some()
                        })
                        .unwrap();
                    assert_eq!(
                        fire,
                        (emitted + want + full_la).max(n),
                        "la {first_la} chunk {chunk} n {n}"
                    );
                    assert_eq!(next_window_chunk(n, first_la, chunk, false), chunk);
                }
            }
        }
    }

    #[test]
    fn step_plan_covers_each_frame_once() {
        const CAP: usize = 17;
        assert_eq!(step_plan(3, 0, false, 4, CAP, false), None);
        assert_eq!(step_plan(3, 0, true, 4, CAP, false), Some(3));
        assert_eq!(step_plan(4, 0, false, 4, CAP, false), Some(4));
        assert_eq!(step_plan(5, 4, false, 4, CAP, false), None);
        assert_eq!(step_plan(8, 4, false, 4, CAP, false), Some(4));
        assert_eq!(step_plan(30, 21, false, 4, CAP, false), None);
        assert_eq!(step_plan(38, 21, false, 4, CAP, false), Some(CAP));
        assert_eq!(step_plan(40, 4, false, 4, CAP, false), Some(CAP));
        assert_eq!(step_plan(4, 4, true, 4, CAP, false), None);
        // Urgent: a started stream steps with what it has; the first step still waits.
        assert_eq!(step_plan(30, 21, false, 4, CAP, true), Some(9));
        assert_eq!(step_plan(21, 21, false, 4, CAP, true), None);
        assert_eq!(step_plan(3, 0, false, 4, CAP, true), None);
        // Fast frames: never urgent with more lead than two frame periods (10 ms frames, a 4 ms
        // codec step and 23 ms of lead wait for the next frame).
        assert_eq!(urgent_lead(0.010, 0.004), 0.020);
        for per_frame in [0.001, 0.010, 0.040, 0.080, 0.200] {
            for step_s in [0.0, 0.004, 0.020, 0.500] {
                assert!(urgent_lead(per_frame, step_s) <= 2.0 * per_frame);
            }
        }
        assert!((urgent_lead(0.070, 0.010) - 0.095).abs() < 1e-9);
        // Banking follows live load only: 32 slow streams, then idle, then one request.
        let period = FramePeriod::default();
        assert!(!period.banks(0.080, 0), "nothing measured");
        for ms in (0..5_000).step_by(50) {
            period.observe(0.060, ms);
        }
        assert!(period.banks(0.080, 5_000), "live streams at RTF 0.75 bank");
        assert!(
            !period.banks(0.080, 5_000 + FramePeriod::FRESH_MS + 1),
            "idle: no extra frame"
        );
        // The next load starts from its own measurement, not the stale one.
        period.observe(0.020, 60_000);
        assert!((period.recent(60_000).unwrap() - 0.020).abs() < 1e-9);
        assert!(!period.banks(0.080, 60_000));
        // The deadline clock restarts with the frame count. Ten seconds into a request, a later
        // segment (or a retry) has five frames and a second of audio buffered: over the request's
        // clock that reads as a 2.5 s frame period and is urgent; over the segment's it is not.
        let now = Instant::now();
        let (began, due) = (
            now - std::time::Duration::from_secs(10),
            now + std::time::Duration::from_secs(1),
        );
        assert!(step_urgent(Some(due), Some(began), 5, 0.02));
        let mut pace = FramePace(Some(began));
        for _segment_then_retry in 0..2 {
            pace.restart();
            assert!(!step_urgent(Some(due), pace.since(), 5, 0.02));
            pace.frame();
            let since = pace.since().unwrap();
            assert!(
                since >= now,
                "the clock starts at the segment's first frame"
            );
            pace.frame();
            assert_eq!(pace.since(), Some(since));
            assert!(!step_urgent(Some(due), pace.since(), 5, 0.02));
        }
        for total in 1..60 {
            for arrive in 1..20 {
                let (mut emitted, mut seen, mut n) = (0, Vec::new(), 0);
                while n < total || emitted < total {
                    n = (n + arrive).min(total);
                    while let Some(k) = step_plan(n, emitted, n == total, 4, CAP, false) {
                        assert!(k <= CAP && (emitted > 0 || k >= 4.min(total)));
                        seen.extend(emitted..emitted + k);
                        emitted += k;
                    }
                }
                assert_eq!(seen, (0..total).collect::<Vec<_>>());
            }
        }
    }

    /// Every frame is emitted exactly once, in order, each with LOOKAHEAD right context until
    /// the final flush and the one-frame window's left context; later windows emit CHUNK frames
    /// and fit the smallest capacity (WINDOW + CHUNK - 1 frames).
    #[test]
    fn stream_plan_covers_each_frame_once() {
        for ramp in [false, true] {
            for chunk in 1..=3 {
                for total in 1..20 {
                    let (mut emitted, mut seen) = (0, Vec::new());
                    for n in 1..=total {
                        if let Some((s, e, upto)) =
                            stream_step(n, emitted, false, WINDOW, LOOKAHEAD, chunk, ramp)
                        {
                            assert!(
                                s <= emitted && upto <= e && e <= n && e - s <= WINDOW + chunk - 1
                            );
                            assert!(emitted - s == emitted.min(WINDOW - LOOKAHEAD - 1));
                            assert!(e - upto >= LOOKAHEAD.min(n - upto));
                            let want = if ramp { chunk.min(emitted) } else { chunk };
                            assert!(emitted == 0 || upto - emitted == want);
                            seen.extend(emitted..upto);
                            emitted = upto;
                        }
                    }
                    if let Some((_, e, upto)) =
                        stream_step(total, emitted, true, WINDOW, LOOKAHEAD, chunk, ramp)
                    {
                        assert_eq!((e, upto), (total, total));
                        seen.extend(emitted..upto);
                    }
                    assert_eq!(seen, (0..total).collect::<Vec<_>>());
                }
            }
        }
        // Ramped: after a one-frame first chunk the next windows emit 1, 2, then `chunk` frames.
        let mut emitted = 0;
        let mut sizes = Vec::new();
        for n in 1..=14 {
            if let Some((_, _, upto)) = stream_step(n, emitted, false, WINDOW, 1, 3, true) {
                sizes.push(upto - emitted);
                emitted = upto;
            }
        }
        assert_eq!(sizes, [1, 1, 2, 3, 3, 3]);
    }

    /// Request identity is reset per WebSocket message: turns and budgets do not leak across
    /// requests, and each message obtains a fresh request id.
    #[test]
    fn websocket_request_ids_reset_per_message() {
        use crate::serve::session::{RequestIds, RouteFields};
        use axum::http::HeaderMap;

        let handshake = RequestIds::from_headers(&HeaderMap::new()).unwrap();
        let r1 = RouteFields {
            metadata: Some(
                serde_json::json!({
                    "turn_id": "turn-1",
                    "turn_budget_ms": 500,
                })
                .as_object()
                .unwrap()
                .clone(),
            ),
            ..Default::default()
        };
        let mut msg1_ids = handshake.with_new_request();
        msg1_ids.apply_body(&r1).unwrap();
        assert_eq!(msg1_ids.turn.as_deref(), Some("turn-1"));
        assert_eq!(msg1_ids.budget_ms, Some(500));

        let r2 = RouteFields {
            metadata: Some(
                serde_json::json!({
                    "turn_id": "turn-2",
                    "turn_budget_ms": 1000,
                })
                .as_object()
                .unwrap()
                .clone(),
            ),
            ..Default::default()
        };
        let mut msg2_ids = handshake.with_new_request();
        msg2_ids.apply_body(&r2).unwrap();
        assert_eq!(msg2_ids.turn.as_deref(), Some("turn-2"));
        assert_eq!(msg2_ids.budget_ms, Some(1000));
        assert_ne!(msg1_ids.request, msg2_ids.request);
    }
}
