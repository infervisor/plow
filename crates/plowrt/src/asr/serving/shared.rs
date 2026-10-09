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
    /// A final's audio: encoded ahead of partials.
    urgent: bool,
    respond: oneshot::Sender<Result<Vec<f32>>>,
}

/// Every queued utterance shares the next packed launch, in arrival order, as many as its largest
/// capacity holds; a lone utterance (or a packet without packed programs) runs alone.
///
/// Finals go first. (The encoder stays off the co-tenant device turn: taking it there made a
/// final's encode wait out other models' ticks, 52 -> 212 ms p50 at 50 calls, where running
/// alongside them costs less.)
/// `cost_id`: the encoder's id in [`crate::sched::cost`]; launches are sized at a 10 ms mel hop.
fn encode_loop(rx: std::sync::mpsc::Receiver<Encode>, mut encoder: PacketAudioEncoder, cost_id: usize) {
    let max_chunks = encoder.max_packed_chunks();
    let mut pending: std::collections::VecDeque<Encode> = Default::default();
    while let Ok(first) = rx.recv() {
        pending.push_back(first);
        while !pending.is_empty() {
            pending.extend(rx.try_iter());
            pending.make_contiguous().sort_by_key(|job| !job.urgent);
            let mut chunks = 0;
            let n = pending
                .iter()
                .take_while(|job| {
                    chunks += encoder.chunks(job.features.frames);
                    chunks <= max_chunks
                })
                .count();
            // One utterance runs its single-utterance capacity: the same bits, a shorter launch.
            if n <= 1 {
                let job = pending.pop_front().unwrap();
                let started = Instant::now();
                let rows = encoder.encode(&job.features);
                tracing::debug!(
                    frames = job.features.frames,
                    urgent = job.urgent,
                    queued = pending.len(),
                    wall_ms = started.elapsed().as_secs_f64() * 1e3,
                    "asr: single encoder launch"
                );
                crate::sched::cost::record_id(cost_id, crate::sched::cost::Op::Encode { audio_ms: job.features.frames as u32 * 10 }, started.elapsed());
                let _ = job.respond.send(rows);
                continue;
            }
            let batch: Vec<Encode> = pending.drain(..n).collect();
            let features: Vec<&MelFeatures> = batch.iter().map(|job| &job.features).collect();
            let started = Instant::now();
            let encoded = encoder.encode_packed(&features);
            tracing::debug!(
                items = n,
                finals = batch.iter().filter(|job| job.urgent).count(),
                chunks = features.iter().map(|f| encoder.chunks(f.frames)).sum::<usize>(),
                gpu_ms = encoder.last_gpu_us() / 1e3,
                wall_ms = started.elapsed().as_secs_f64() * 1e3,
                "asr: packed encoder launch"
            );
            crate::sched::cost::record_id(cost_id, crate::sched::cost::Op::Encode { audio_ms: features.iter().map(|f| f.frames as u32 * 10).sum() }, started.elapsed());
            match encoded {
                Ok(rows) => {
                    for (job, rows) in batch.into_iter().zip(rows) {
                        let _ = job.respond.send(Ok(rows));
                    }
                }
                Err(e) => {
                    for job in batch {
                        let _ = job.respond.send(Err(RuntimeError::Msg(format!("ASR encoder: {e}"))));
                    }
                }
            }
        }
    }
}

pub(super) struct SharedAsr {
    prompt: Arc<AudioLmPrompt>,
    /// Taken at [`release`]: the encoder thread then exits, freeing its packet, even while an idle
    /// session still holds this front.
    encode: Mutex<Option<std::sync::mpsc::Sender<Encode>>>,
    max_context: usize,
    /// Requests between submit and their answer: the front's bound. Past it a request is refused
    /// at once; under it a request waits in the mux queue (a full ingress makes submit wait).
    inflight: Arc<Semaphore>,
    /// Log-mel frames of one encoder attention window (0: the encoder is not windowed).
    window_frames: usize,
    slug: String,
    /// Cleared when the encoder thread exits (a panic included): `/health` reports the model.
    alive: Arc<AtomicBool>,
}

/// Clears `alive` however the encoder thread ends.
struct Alive(Arc<AtomicBool>);
impl Drop for Alive {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

impl SharedAsr {
    /// The instance key (model, or `model#rank`) whose dispatcher this front feeds.
    pub(super) fn key(&self) -> &str {
        &self.slug
    }

    fn load(slug: &str, dir: &Path, max_context: usize, batch: usize, device: u8) -> Result<Self> {
        let checkpoint = crate::asset::serve::checkpoint_dir(dir);
        let checkpoint = if checkpoint.is_dir() { checkpoint } else { dir.to_path_buf() };
        let prompt = AudioLmPrompt::load(&dir.join("model.pkt"), &checkpoint)?;
        let encoder_path =
            crate::exec::packet_runtime::stage_packet(&dir.join("model.pkt"), "encoder.packet", "encoder.pkt")?;
        let mut encoder = PacketAudioEncoder::load_on(&encoder_path, "cuda", device)?;
        let warm = Instant::now();
        encoder.warm()?;
        tracing::info!(ms = warm.elapsed().as_millis() as u64, packed_chunks = encoder.max_packed_chunks(), "asr: encoder graphs warmed");
        if encoder.output_width() != prompt.hidden() {
            return Err(RuntimeError::Rejected("audio packet output width does not match the decoder".into()));
        }
        let chunking = prompt.chunking();
        let window_frames = encoder.window_rows() / chunking.chunk_frames.div_ceil(chunking.frame_stride) * chunking.chunk_frames;
        let (encode, rx) = std::sync::mpsc::channel::<Encode>();
        let cost_id = crate::sched::cost::id(&encoder_path.to_string_lossy());
        let alive = Arc::new(AtomicBool::new(true));
        let guard = Alive(Arc::clone(&alive));
        std::thread::Builder::new()
            .name("plow-asr-encoder".into())
            .spawn(move || {
                let _alive = guard;
                encode_loop(rx, encoder, cost_id)
            })
            .map_err(|e| RuntimeError::Msg(format!("spawn ASR encoder thread: {e}")))?;
        Ok(Self {
            prompt: Arc::new(prompt),
            encode: Mutex::new(Some(encode)),
            max_context,
            inflight: Arc::new(Semaphore::new(batch.saturating_mul(4).max(UPLOADS))),
            window_frames,
            slug: slug.to_owned(),
            alive,
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
                // Work held across a preempt (an idle session's next turn) is told so up front.
                _ if mux.preempted() => Err(RuntimeError::Unavailable(crate::serve::mux::PREEMPTED.into())),
                Some(windows) => this.run_partial(mux, samples, windows, language, context, &cancel, opts, &mut tx).await,
                None => this.run(mux, samples, language, context, &cancel, opts, &mut tx).await,
            };
            drop(permit);
            let _ = tx.send(result);
        });
        Ok(rx)
    }

    async fn encode_rows(&self, features: MelFeatures, urgent: bool) -> Result<Vec<f32>> {
        let (tx, rx) = oneshot::channel();
        let encode = self.encode.lock().clone().ok_or_else(|| RuntimeError::Unavailable("ASR front released; retry".into()))?;
        encode
            .send(Encode { features, urgent, respond: tx })
            .map_err(|_| RuntimeError::Unavailable("ASR encoder thread is gone".into()))?;
        rx.await.map_err(|_| RuntimeError::Unavailable("ASR encoder thread is gone".into()))?
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
        let front = arrived.elapsed();
        cancelled(cancel)?;
        let frames = features.frames;
        let overlay = self.encode_rows(features, opts.final_pass).await?;
        tracing::debug!(
            final_pass = opts.final_pass,
            frames,
            frontend_ms = front.as_secs_f64() * 1e3,
            encode_ms = (arrived.elapsed() - front).as_secs_f64() * 1e3,
            "asr: front + encode"
        );
        cancelled(cancel)?;
        let (result, _) =
            self.decode(mux, ids, audio_positions, overlay, Vec::new(), language, cancel, opts, answer, arrived).await?;
        Ok(result)
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
        let encoded = futures::future::try_join_all(pieces.iter().map(|&(a, b)| self.encode_rows(slice_frames(&features, a, b), false))).await?;
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
        // Local agreement: the previous partial's transcript, less its last DRAFT_TAIL tokens, is
        // forced as prompt rows (one prefill, and resumed from the session's retained rows)
        // instead of decoded token by token again. Partials are revisable; the final decodes the
        // whole recording from scratch.
        let forced = {
            let w = windows.lock();
            w.draft[..w.draft.len().saturating_sub(DRAFT_TAIL)].to_vec()
        };
        let (result, output) =
            self.decode(mux, ids, audio_positions, overlay, forced, language, cancel, opts, answer, arrived).await?;
        windows.lock().draft = output;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    async fn decode(
        &self,
        mux: ModelMux,
        ids: Vec<u32>,
        audio_positions: Vec<usize>,
        overlay: Vec<f32>,
        forced: Vec<u32>,
        language: Option<String>,
        cancel: &AtomicBool,
        mut opts: AsrOpts,
        answer: &mut oneshot::Sender<Result<Transcript>>,
        arrived: Instant,
    ) -> Result<(Transcript, Vec<u32>)> {
        let mut ids = ids;
        ids.extend_from_slice(&forced);
        if overlay.len() != audio_positions.len() * self.prompt.hidden() {
            return Err(RuntimeError::Rejected(format!(
                "{} audio rows for {} placeholders",
                overlay.len() / self.prompt.hidden(),
                audio_positions.len()
            )));
        }
        let encoded = arrived.elapsed();

        let mut gen = crate::serve::GenParams::default();
        gen.max_tokens = self.prompt.max_tokens().saturating_sub(forced.len()).max(1);
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
                class: if opts.final_pass { JobClass::Final } else { JobClass::Bulk },
                raw_tokens: true,
                session,
                turn: opts.ids.as_ref().and_then(|i| i.turn_key.clone()),
                continuing: false,
                speech: Some(Box::new(SpeechJob { overlay, overlay_pos, pos_base: None, cfg: None, first_tokens: 0 })),
                prefix: None,
                mm: None,
            },
        };
        // Released once the job is on the channel: the dispatcher drains it, not closes on it.
        let ingress = opts.ingress.take();
        // A preempt does not wait for this request's ingress: tell it so instead of submitting.
        let preempted = || RuntimeError::Unavailable(crate::serve::mux::PREEMPTED.into());
        if mux.preempted() {
            return Err(preempted());
        }
        mux.submit_wait(job).await.map_err(|e| match e {
            crate::serve::mux::SubmitError::Full(_) => RuntimeError::Overloaded("ASR queue full".into()),
            crate::serve::mux::SubmitError::Closed(_) if mux.preempted() => preempted(),
            crate::serve::mux::SubmitError::Closed(_) => RuntimeError::Unavailable("model dispatcher unavailable".into()),
        })?;
        drop(ingress);
        let mut output = forced;
        let mut shown = 0usize;
        let submitted = arrived.elapsed();
        let mut first_token = None;
        let cached_tokens = loop {
            let chunk = tokio::select! {
                chunk = stream.recv() => chunk,
                // Client gone: dropping the stream frees the slot on the next tick.
                _ = answer.closed() => return Err(RuntimeError::Rejected("ASR cancelled".into())),
            };
            match chunk {
                Some(StreamChunk::Token { id, .. }) => {
                    cancelled(cancel)?;
                    first_token.get_or_insert_with(|| arrived.elapsed());
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
                // A preempted slot's tokens so far are not the transcript.
                Some(StreamChunk::Done { reason: FinishReason::Preempted, .. }) => return Err(preempted()),
                Some(StreamChunk::Done { usage, .. }) => break usage.cached_tokens,
                Some(StreamChunk::Err(_)) if mux.preempted() => return Err(preempted()),
                Some(StreamChunk::Err(e)) => return Err(e),
                None => return Err(RuntimeError::Msg("ASR stream ended without a result".into())),
            }
        };
        let result = self.prompt.transcript(&output, language.as_deref()).map(|t| (t, output.clone()));
        tracing::debug!(
            prompt_tokens,
            cached_tokens,
            output_tokens = output.len(),
            final_pass = opts.final_pass,
            encoded_ms = encoded.as_secs_f64() * 1e3,
            submitted_ms = submitted.as_secs_f64() * 1e3,
            first_token_ms = first_token.unwrap_or_default().as_secs_f64() * 1e3,
            total_ms = arrived.elapsed().as_secs_f64() * 1e3,
            "ASR completed (serve mux)"
        );
        result
    }
}

/// Tokens at the end of a partial's transcript that the next partial decodes again: the words the
/// open audio window may still revise.
const DRAFT_TAIL: usize = 4;

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

type Fronts = Mutex<HashMap<(PathBuf, u8), Option<Arc<SharedAsr>>>>;

/// Audio LM front-ends by asset directory and device, bound on first use (`None`: not an audio LM).
fn models() -> &'static Fronts {
    static M: OnceLock<Fronts> = OnceLock::new();
    M.get_or_init(Default::default)
}

fn is_audio_lm(dir: &Path) -> Result<bool> {
    let packet = dir.join("model.pkt");
    if !packet.is_file() {
        return Ok(false);
    }
    let Some(asset) = crate::exec::packet_runtime::PacketAsset::load_if_present(&packet)? else {
        return Ok(false);
    };
    Ok(asset
        .pipelines()
        .iter()
        .any(|p| p.driver == "causal.v1" && p.parameters.get("overlay_rows").is_some_and(|&r| r > 0))
        && dir.join(asset.stage_file("encoder.packet", "encoder.pkt")?).is_file())
}

/// `slug` is an instance key: the front binds on its engine's device.
fn shared_asr(state: &AppState, slug: &str, dir: &Path) -> Result<Option<Arc<SharedAsr>>> {
    let device = state.ordinal_of(slug);
    let at = (dir.to_path_buf(), device);
    if let Some(m) = models().lock().get(&at) {
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
        let m = SharedAsr::load(slug, dir, max_context, batch, device)?;
        tracing::info!(%slug, device, dir = %dir.display(), "asr: audio LM front bound to the serve mux");
        Some(Arc::new(m))
    } else {
        None
    };
    models().lock().insert(at, model.clone());
    Ok(model)
}

/// Served models whose encoder thread has exited.
pub(super) fn dead_encoders() -> Vec<String> {
    models()
        .lock()
        .values()
        .flatten()
        .filter(|asr| !asr.alive.load(Ordering::Relaxed))
        .map(|asr| asr.slug.clone())
        .collect()
}

/// Whether `slug` is a bound audio LM on any device (preload binds every resident one); never
/// loads.
pub(super) fn serves_audio(state: &AppState, slug: &str) -> bool {
    let Ok(bundle) = state.registry.get(slug) else { return false };
    models().lock().iter().any(|((dir, _), m)| *dir == bundle.dir && m.is_some())
}

/// Bind instance `slug`'s front-end now (a no-op for a model that is not an audio LM or is bound).
pub fn bind(state: &AppState, slug: &str) -> Result<()> {
    let bundle = state.registry.get(state.model_of(slug))?;
    shared_asr(state, slug, &bundle.dir).map(drop)
}

/// Forget `dir`'s front-end on `device`: its encoder thread exits, freeing the encoder runtime,
/// once the encodes already queued finish — a session still holding the front does not keep it.
pub fn release(dir: &Path, device: u8) {
    if let Some(Some(asr)) = models().lock().remove(&(dir.to_path_buf(), device)) {
        asr.encode.lock().take();
    }
}

/// Bind every resident audio LM's front-end now, so the first request does not pay the encoder
/// load. A front-end that cannot load fails the serve at startup.
pub fn preload(state: &AppState) -> Result<()> {
    for slug in state.registry.slugs() {
        let (Some(_), Ok(bundle)) = (state.mux(&slug), state.registry.get(&slug)) else {
            continue;
        };
        shared_asr(state, &slug, &bundle.dir)
            .map_err(|e| RuntimeError::Rejected(format!("{slug}: asr front-end failed to load: {e}")))?;
    }
    Ok(())
}

/// The front and dispatcher a request (or a realtime session) for `model` runs on. A DP model
/// picks a rank here, sticky by `session`.
pub(super) async fn route(
    state: &Arc<AppState>,
    model: &str,
    session: Option<&str>,
) -> std::result::Result<(Route, FinalizationPolicy), Response> {
    let slug = state.registry.resolve(model).unwrap_or_else(|| model.to_owned());
    let dp = state.dp_set(&slug).cloned();
    let managed = dp.is_some() || state.manager_for(&slug).is_some_and(|m| m.manages(&slug));
    // A managed model's front binds with its engine and leaves with it: an eviction between the
    // residency check and the lookup sends the request around again.
    let mut attempts = 0;
    let mut exclude = 0u32;
    let (key, mux, bundle, bound, ingress) = loop {
        attempts += 1;
        let (key, mux) = match dp.as_deref() {
            Some(set) => {
                if let Err(e) = state.dp_admit(set).await {
                    return Err(failure(StatusCode::SERVICE_UNAVAILABLE, e));
                }
                match state.dp_route(set, session, None, exclude) {
                    Some((rank, mux, _, _)) => {
                        exclude |= 1 << rank;
                        (set.ranks[rank].key.clone(), Some(mux))
                    }
                    None => (slug.clone(), None),
                }
            }
            None => {
                if managed {
                    if let Err(e) = state.manager_for(&slug).expect("managed").ensure_resident(&slug).await {
                        return Err(failure(StatusCode::SERVICE_UNAVAILABLE, e));
                    }
                }
                (slug.clone(), state.mux(&slug))
            }
        };
        let (Some(mux), Ok(bundle)) = (mux, state.registry.get(&slug)) else {
            if managed && attempts < 3 {
                continue;
            }
            return Err(failure(StatusCode::NOT_FOUND, "unknown ASR model"));
        };
        // Counted before the residency re-check: an eviction that removes the mux after this
        // point drains with the request counted, so the request's submission is served.
        let ingress = mux.ingress_owned();
        let at = (bundle.dir.clone(), state.ordinal_of(&key));
        let bound = models().lock().get(&at).cloned().filter(|_| state.mux(&key).is_some());
        match bound {
            None if managed && attempts < 3 => continue,
            None if managed => return Err(failure(StatusCode::SERVICE_UNAVAILABLE, "ASR front is switching; retry")),
            bound => break (key, mux, bundle, bound, ingress),
        }
    };
    let bound = match bound {
        Some(bound) => Ok(bound),
        None => tokio::task::block_in_place(|| shared_asr(state, &key, &bundle.dir)),
    };
    match bound {
        Ok(Some(asr)) => {
            let finalization = asr.prompt.finalization_policy();
            let ingress = Arc::new(parking_lot::Mutex::new(Some(ingress)));
            Ok((Route::Shared(asr, mux, ingress), finalization))
        }
        Ok(None) => Err(crate::serve::models::unserved(&slug, "audio/transcriptions", &crate::serve::models::endpoints(state, &slug))),
        Err(e) => Err(failure(StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}
