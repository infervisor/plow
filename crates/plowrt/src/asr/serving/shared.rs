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

use super::{failure, Route, SubmitError};
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
    /// Requests between submit and their answer. Sized to the mux ingress, so a full mux is
    /// reported here, synchronously, as a full queue.
    inflight: Arc<Semaphore>,
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
            inflight: Arc::new(Semaphore::new(batch.saturating_mul(4).max(1))),
        })
    }

    pub(super) fn submit(
        self: &Arc<Self>,
        mux: ModelMux,
        samples: Vec<f32>,
        language: Option<String>,
        context: String,
        cancel: Arc<AtomicBool>,
        final_pass: bool,
    ) -> std::result::Result<oneshot::Receiver<Result<Transcript>>, SubmitError> {
        let permit = self.inflight.clone().try_acquire_owned().map_err(|_| SubmitError::Full)?;
        let (mut tx, rx) = oneshot::channel();
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let result = this.run(mux, samples, language, context, &cancel, final_pass, &mut tx).await;
            drop(permit);
            let _ = tx.send(result);
        });
        Ok(rx)
    }

    async fn run(
        &self,
        mux: ModelMux,
        samples: Vec<f32>,
        language: Option<String>,
        context: String,
        cancel: &AtomicBool,
        final_pass: bool,
        answer: &mut oneshot::Sender<Result<Transcript>>,
    ) -> Result<Transcript> {
        let arrived = Instant::now();
        let cancelled = || {
            if cancel.load(Ordering::Relaxed) {
                Err(RuntimeError::Rejected("ASR cancelled".into()))
            } else {
                Ok(())
            }
        };
        let prompt = Arc::clone(&self.prompt);
        let max_context = self.max_context;
        let AudioLmRequest { features, ids, audio_positions, language } = tokio::task::spawn_blocking(move || {
            prompt.request(&samples, language.as_deref(), &context, max_context)
        })
        .await
        .map_err(|e| RuntimeError::Msg(format!("ASR prompt task: {e}")))??;
        cancelled()?;
        let (tx, rx) = oneshot::channel();
        self.encode
            .send(Encode { features, respond: tx })
            .map_err(|_| RuntimeError::Msg("ASR encoder thread is gone".into()))?;
        let overlay = rx.await.map_err(|_| RuntimeError::Msg("ASR encoder thread is gone".into()))??;
        if overlay.len() != audio_positions.len() * self.prompt.hidden() {
            return Err(RuntimeError::Rejected(format!(
                "{} audio rows for {} placeholders",
                overlay.len() / self.prompt.hidden(),
                audio_positions.len()
            )));
        }
        cancelled()?;
        let encoded = arrived.elapsed();

        let mut gen = crate::serve::GenParams::default();
        gen.max_tokens = self.prompt.max_tokens();
        gen.params.temperature = 0.0;
        gen.stop_token_ids = self.prompt.stop().to_vec();
        let prompt_tokens = ids.len();
        let (respond, mut stream) = stream_mod::channel();
        let job = Job {
            prompt_ids: ids,
            gen,
            arrived,
            respond,
            opts: JobOpts {
                class: if final_pass { JobClass::Critical } else { JobClass::Bulk },
                raw_tokens: true,
                speech: Some(Box::new(SpeechJob {
                    overlay,
                    overlay_pos: audio_positions.into_iter().map(|p| p as u32).collect(),
                    pos_base: None,
                    cfg: None,
                })),
            },
        };
        mux.submit(job).map_err(|e| match e {
            crate::serve::mux::SubmitError::Full(_) => RuntimeError::Rejected("ASR queue full".into()),
            crate::serve::mux::SubmitError::Closed(_) => RuntimeError::Msg("model dispatcher unavailable".into()),
        })?;
        let mut output = Vec::new();
        loop {
            let chunk = tokio::select! {
                chunk = stream.recv() => chunk,
                // Client gone: dropping the stream frees the slot on the next tick.
                _ = answer.closed() => return Err(RuntimeError::Rejected("ASR cancelled".into())),
            };
            match chunk {
                Some(StreamChunk::Token { id, .. }) => {
                    cancelled()?;
                    output.push(id);
                }
                Some(StreamChunk::Done { reason: FinishReason::Length, .. }) => {
                    return Err(RuntimeError::Rejected("ASR exceeded output token limit".into()))
                }
                Some(StreamChunk::Done { .. }) => break,
                Some(StreamChunk::Err(e)) => return Err(e),
                None => return Err(RuntimeError::Msg("ASR stream ended without a result".into())),
            }
        }
        let result = self.prompt.transcript(&output, language.as_deref());
        tracing::debug!(
            prompt_tokens,
            output_tokens = output.len(),
            encoded_ms = encoded.as_secs_f64() * 1e3,
            total_ms = arrived.elapsed().as_secs_f64() * 1e3,
            "ASR completed (serve mux)"
        );
        result
    }
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
