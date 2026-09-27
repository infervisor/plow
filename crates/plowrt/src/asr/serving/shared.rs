//! Causal audio LM transcription on `plowrt serve`: the log-mel frontend and prompt run on the
//! blocking pool, `encoder.pkt` on one encoder thread, and the decoder as an ordinary [`Job`] on
//! the model's continuous-batching mux, its encoder rows spliced over the audio placeholders.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use axum::http::StatusCode;
use axum::response::Response;
use parking_lot::Mutex;
use tokio::sync::{oneshot, Semaphore};

use super::{failure, AsrOpts, Route, SubmitError, WindowCache};
use crate::asr::audio_lm::{AudioLmPrompt, AudioLmRequest, PacketAudioEncoder};
use crate::asr::frontend::MelFeatures;
use crate::asr::{FinalizationPolicy, Transcript};
use crate::serve::mux::{Job, JobClass, JobOpts, ModelMux, SpeechJob};
use crate::serve::stream::{self as stream_mod, FinishReason, StreamChunk};
use crate::serve::AppState;
use crate::{Result, RuntimeError};

/// Concurrent uploads / sessions the serve front accepts per process.
pub(super) const UPLOADS: usize = 256;

struct Encode {
    features: MelFeatures,
    respond: oneshot::Sender<Result<Vec<f32>>>,
}

pub(super) struct SharedAsr {
    prompt: Arc<AudioLmPrompt>,
    encode: std::sync::mpsc::Sender<Encode>,
    max_context: usize,
    /// Requests between submit and their answer: the front's bound. Past it a request is refused
    /// at once; under it a request waits in the mux queue (a full ingress makes submit wait).
    inflight: Arc<Semaphore>,
    /// Log-mel frames of one encoder attention window (0: the encoder is not windowed).
    window_frames: usize,
}

impl SharedAsr {
    fn load(dir: &Path, max_context: usize, batch: usize) -> Result<Self> {
        let checkpoint = dir.join("checkpoint");
        let checkpoint = if checkpoint.is_dir() { checkpoint } else { dir.to_path_buf() };
        let prompt = AudioLmPrompt::load(&dir.join("model.pkt"), &checkpoint)?;
        let mut encoder = PacketAudioEncoder::load(&dir.join("encoder.pkt"), "cuda")?;
        if encoder.output_width() != prompt.hidden() {
            return Err(RuntimeError::Rejected("audio packet output width does not match the decoder".into()));
        }
        let chunking = prompt.chunking();
        let window_frames = encoder.window_rows() / chunking.chunk_frames.div_ceil(chunking.frame_stride) * chunking.chunk_frames;
        let (encode, rx) = std::sync::mpsc::channel::<Encode>();
        std::thread::Builder::new()
            .name("plow-asr-encoder".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    let _ = job.respond.send(encoder.encode(&job.features));
                }
            })
            .map_err(|e| RuntimeError::Msg(format!("spawn ASR encoder thread: {e}")))?;
        Ok(Self {
            prompt: Arc::new(prompt),
            encode,
            max_context,
            inflight: Arc::new(Semaphore::new(batch.saturating_mul(4).max(UPLOADS))),
            window_frames,
        })
    }

    pub(super) fn submit(
        self: &Arc<Self>,
        mux: ModelMux,
        samples: Vec<f32>,
        language: Option<String>,
        context: String,
        cancel: Arc<AtomicBool>,
        opts: AsrOpts,
    ) -> std::result::Result<oneshot::Receiver<Result<Transcript>>, SubmitError> {
        let permit = self.inflight.clone().try_acquire_owned().map_err(|_| SubmitError::Full)?;
        let (mut tx, rx) = oneshot::channel();
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let result = match opts.windows.clone().filter(|_| !opts.final_pass) {
                Some(windows) => this.run_partial(mux, samples, windows, language, context, &cancel, opts, &mut tx).await,
                None => this.run(mux, samples, language, context, &cancel, opts, &mut tx).await,
            };
            drop(permit);
            let _ = tx.send(result);
        });
        Ok(rx)
    }

    async fn encode_rows(&self, features: MelFeatures) -> Result<Vec<f32>> {
        let (tx, rx) = oneshot::channel();
        self.encode
            .send(Encode { features, respond: tx })
            .map_err(|_| RuntimeError::Msg("ASR encoder thread is gone".into()))?;
        rx.await.map_err(|_| RuntimeError::Msg("ASR encoder thread is gone".into()))?
    }

    #[allow(clippy::too_many_arguments)]
    async fn run(
        &self,
        mux: ModelMux,
        samples: Vec<f32>,
        language: Option<String>,
        context: String,
        cancel: &AtomicBool,
        opts: AsrOpts,
        answer: &mut oneshot::Sender<Result<Transcript>>,
    ) -> Result<Transcript> {
        let arrived = Instant::now();
        let prompt = Arc::clone(&self.prompt);
        let max_context = self.max_context;
        let AudioLmRequest { features, ids, audio_positions, language } = tokio::task::spawn_blocking(move || {
            prompt.request(&samples, language.as_deref(), &context, max_context)
        })
        .await
        .map_err(|e| RuntimeError::Msg(format!("ASR prompt task: {e}")))??;
        cancelled(cancel)?;
        let overlay = self.encode_rows(features).await?;
        cancelled(cancel)?;
        self.decode(mux, ids, audio_positions, overlay, language, cancel, opts, answer, arrived).await
    }

    /// A revisable partial transcript of a growing recording: the encoder rows of its completed
    /// attention windows come from `windows` (each window is encoded once, when it completes), only
    /// the open window is encoded again, and the session's retained decoder rows cover the prompt
    /// through the completed windows.
    #[allow(clippy::too_many_arguments)]
    async fn run_partial(
        &self,
        mux: ModelMux,
        samples: Vec<f32>,
        windows: Arc<Mutex<WindowCache>>,
        language: Option<String>,
        context: String,
        cancel: &AtomicBool,
        opts: AsrOpts,
        answer: &mut oneshot::Sender<Result<Transcript>>,
    ) -> Result<Transcript> {
        let arrived = Instant::now();
        let prompt = Arc::clone(&self.prompt);
        let (features, language, context) = tokio::task::spawn_blocking(move || {
            let language = prompt.stream_language(language.as_deref(), &context)?;
            Ok::<_, RuntimeError>((prompt.features(&samples)?, language, context))
        })
        .await
        .map_err(|e| RuntimeError::Msg(format!("ASR prompt task: {e}")))??;
        cancelled(cancel)?;
        let wf = self.window_frames;
        // A window is final once the frames after it cover the STFT's right context.
        let stable = if wf == 0 { 0 } else { features.frames.saturating_sub(STABLE_MARGIN_FRAMES) / wf };
        let cached: Vec<Arc<[f32]>> = {
            let mut w = windows.lock();
            w.rows.truncate(stable);
            w.rows.clone()
        };
        let mut pieces: Vec<(usize, usize)> = (cached.len()..stable).map(|w| (w * wf, (w + 1) * wf)).collect();
        if stable * wf < features.frames {
            pieces.push((stable * wf, features.frames));
        }
        let encoded = futures::future::try_join_all(pieces.iter().map(|&(a, b)| self.encode_rows(slice_frames(&features, a, b)))).await?;
        cancelled(cancel)?;
        let fresh = stable - cached.len();
        let hidden = self.prompt.hidden();
        let mut overlay = Vec::with_capacity(self.prompt.chunking().rows(features.frames) * hidden);
        for rows in &cached {
            overlay.extend_from_slice(rows);
        }
        for rows in &encoded {
            overlay.extend_from_slice(rows);
        }
        {
            let mut w = windows.lock();
            if w.rows.len() == cached.len() {
                w.rows.extend(encoded[..fresh].iter().map(|rows| Arc::from(rows.as_slice())));
            }
        }
        let rows = overlay.len() / hidden;
        let (ids, audio_positions) = self.prompt.prompt(rows, language.as_deref(), &context, self.max_context)?;
        tracing::debug!(frames = features.frames, windows_cached = cached.len(), windows_encoded = fresh, open_frames = features.frames - stable * wf, "asr: partial encode");
        self.decode(mux, ids, audio_positions, overlay, language, cancel, opts, answer, arrived).await
    }

    #[allow(clippy::too_many_arguments)]
    async fn decode(
        &self,
        mux: ModelMux,
        ids: Vec<u32>,
        audio_positions: Vec<usize>,
        overlay: Vec<f32>,
        language: Option<String>,
        cancel: &AtomicBool,
        mut opts: AsrOpts,
        answer: &mut oneshot::Sender<Result<Transcript>>,
        arrived: Instant,
    ) -> Result<Transcript> {
        if overlay.len() != audio_positions.len() * self.prompt.hidden() {
            return Err(RuntimeError::Rejected(format!(
                "{} audio rows for {} placeholders",
                overlay.len() / self.prompt.hidden(),
                audio_positions.len()
            )));
        }
        let encoded = arrived.elapsed();

        let mut gen = crate::serve::GenParams::default();
        gen.max_tokens = self.prompt.max_tokens();
        gen.params.temperature = 0.0;
        gen.stop_token_ids = self.prompt.stop().to_vec();
        let prompt_tokens = ids.len();
        let overlay_pos: Vec<u32> = audio_positions.into_iter().map(|p| p as u32).collect();
        let report = opts.report.take();
        let session = opts.ids.as_ref().filter(|i| i.session.is_some()).and_then(|i| {
            i.ticket(crate::serve::session::row_keys(&ids, &overlay_pos, &[&overlay]), report)
        });
        let (respond, mut stream) = stream_mod::channel();
        let job = Job {
            prompt_ids: ids,
            gen,
            arrived,
            respond,
            opts: JobOpts {
                class: if opts.final_pass { JobClass::Critical } else { JobClass::Bulk },
                raw_tokens: true,
                session,
                speech: Some(Box::new(SpeechJob { overlay, overlay_pos, pos_base: None, cfg: None })),
            },
        };
        mux.submit_wait(job).await.map_err(|e| match e {
            crate::serve::mux::SubmitError::Full(_) => RuntimeError::Rejected("ASR queue full".into()),
            crate::serve::mux::SubmitError::Closed(_) => RuntimeError::Msg("model dispatcher unavailable".into()),
        })?;
        let mut output = Vec::new();
        let mut shown = 0usize;
        let cached_tokens = loop {
            let chunk = tokio::select! {
                chunk = stream.recv() => chunk,
                // Client gone: dropping the stream frees the slot on the next tick.
                _ = answer.closed() => return Err(RuntimeError::Rejected("ASR cancelled".into())),
            };
            match chunk {
                Some(StreamChunk::Token { id, .. }) => {
                    cancelled(cancel)?;
                    output.push(id);
                    if let Some(deltas) = &opts.deltas {
                        if let Some(text) = self.prompt.text_so_far(&output, language.as_deref()) {
                            if text.len() > shown && text.is_char_boundary(shown) {
                                let _ = deltas.send(text[shown..].to_owned());
                                shown = text.len();
                            }
                        }
                    }
                }
                Some(StreamChunk::Done { reason: FinishReason::Length, .. }) => {
                    return Err(RuntimeError::Rejected("ASR exceeded output token limit".into()))
                }
                Some(StreamChunk::Done { usage, .. }) => break usage.cached_tokens,
                Some(StreamChunk::Err(e)) => return Err(e),
                None => return Err(RuntimeError::Msg("ASR stream ended without a result".into())),
            }
        };
        let result = self.prompt.transcript(&output, language.as_deref());
        tracing::debug!(
            prompt_tokens,
            cached_tokens,
            output_tokens = output.len(),
            final_pass = opts.final_pass,
            encoded_ms = encoded.as_secs_f64() * 1e3,
            total_ms = arrived.elapsed().as_secs_f64() * 1e3,
            "ASR completed (serve mux)"
        );
        result
    }
}

/// Frames the log-mel STFT window reaches past a frame (centered, ±`fft/2` samples), with slack.
const STABLE_MARGIN_FRAMES: usize = 4;

fn cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(RuntimeError::Rejected("ASR cancelled".into()))
    } else {
        Ok(())
    }
}

/// Frames `a..b` of `[bin][frame]` features.
fn slice_frames(features: &MelFeatures, a: usize, b: usize) -> MelFeatures {
    let bins = features.values.len() / features.frames.max(1);
    let mut values = Vec::with_capacity(bins * (b - a));
    for bin in 0..bins {
        values.extend_from_slice(&features.values[bin * features.frames + a..bin * features.frames + b]);
    }
    MelFeatures { values, frames: b - a }
}

/// Audio LM front-ends by asset directory, bound on first use (`None`: not an audio LM).
fn models() -> &'static Mutex<HashMap<PathBuf, Option<Arc<SharedAsr>>>> {
    static M: OnceLock<Mutex<HashMap<PathBuf, Option<Arc<SharedAsr>>>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

fn is_audio_lm(dir: &Path) -> Result<bool> {
    let packet = dir.join("model.pkt");
    if !packet.is_file() {
        return Ok(false);
    }
    let asset = crate::exec::packet_runtime::PacketAsset::load(&packet)?;
    Ok(asset
        .pipelines()
        .iter()
        .any(|p| p.driver == "causal.v1" && p.parameters.get("overlay_rows").is_some_and(|&r| r > 0))
        && dir.join("encoder.pkt").is_file())
}

fn shared_asr(state: &AppState, slug: &str, dir: &Path) -> Result<Option<Arc<SharedAsr>>> {
    if let Some(m) = models().lock().get(dir) {
        return Ok(m.clone());
    }
    let model = if is_audio_lm(dir)? {
        let (max_context, batch) = state
            .gpu_engine(slug)
            .map(|e| {
                let e = e.lock();
                (e.max_ctx(), e.batch())
            })
            .ok_or_else(|| RuntimeError::Rejected(format!("{slug} has no GPU engine")))?;
        let m = SharedAsr::load(dir, max_context, batch)?;
        tracing::info!(%slug, dir = %dir.display(), "asr: audio LM front bound to the serve mux");
        Some(Arc::new(m))
    } else {
        None
    };
    models().lock().insert(dir.to_path_buf(), model.clone());
    Ok(model)
}

/// Bind every resident audio LM's front-end now, so the first request does not pay the encoder
/// load.
pub fn preload(state: &AppState) {
    for slug in state.registry.slugs() {
        let (Some(_), Ok(bundle)) = (state.mux(&slug), state.registry.get(&slug)) else {
            continue;
        };
        if let Err(e) = shared_asr(state, &slug, &bundle.dir) {
            tracing::warn!(%slug, error = %e, "asr: audio LM front failed to load");
        }
    }
}

pub(super) async fn route(
    state: &Arc<AppState>,
    model: &str,
) -> std::result::Result<(Route, FinalizationPolicy), Response> {
    let slug = state.registry.resolve(model).unwrap_or_else(|| model.to_owned());
    if let Some(mgr) = state.manager_for(&slug) {
        if mgr.manages(&slug) {
            if let Err(e) = mgr.ensure_resident(&slug).await {
                return Err(failure(StatusCode::SERVICE_UNAVAILABLE, e));
            }
        }
    }
    let (Some(mux), Ok(bundle)) = (state.mux(&slug), state.registry.get(&slug)) else {
        return Err(failure(StatusCode::NOT_FOUND, "unknown ASR model"));
    };
    match tokio::task::block_in_place(|| shared_asr(state, &slug, &bundle.dir)) {
        Ok(Some(asr)) => {
            let finalization = asr.prompt.finalization_policy();
            Ok((Route::Shared(asr, mux), finalization))
        }
        Ok(None) => Err(failure(StatusCode::NOT_FOUND, "model declares no ASR pipeline")),
        Err(e) => Err(failure(StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}
