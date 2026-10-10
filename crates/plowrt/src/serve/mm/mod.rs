//! Multimodal chat input: image and audio content parts become encoder rows that replace the
//! placeholder tokens of the model's own chat template.
//!
//! Driven entirely by the text packet's `plow.multimodal.v1` contract and its encoder sidecars:
//! per modality the placeholder id the template renders, the soft-token wrapping, the host
//! preprocessing parameters and the encoder packet. A model whose packet has no contract (or no
//! modality for a part) refuses the part with a 400; nothing is dropped.
//!
//! Soft-token rows travel as prompt ids with bit 31 set whose low bits hash the media content and
//! the row index, so every prefix-cache and session key over token ids tells two images (or clips)
//! apart even when the text around them is identical. The rows themselves sit in the LM's
//! `in.mm_slab` from admission until the request ends; `in.mm_table` maps id -> slab row.
//!
//! Each engine instance (a model, or one DP rank) owns its [`MmModel`]: its encoders run on that
//! engine's device and its [`Slab`] mirrors that engine's table. It is created with the engine and
//! dropped with it ([`crate::serve::AppState::install_mm`]).

pub mod encoder;
pub mod media;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use plow_asset::multimodal::{MmContract, MmModality, Processor, ROW_ID_BIT, SLAB_TENSOR, TABLE_TENSOR};
use sha2::{Digest, Sha256};

use crate::serve::openai::{ContentPart, Message};

/// A request the media step refuses: the 400 (or 503) message and the offending field.
#[derive(Debug)]
pub struct MmError {
    pub status: u16,
    pub message: String,
}

impl MmError {
    fn bad(message: impl Into<String>) -> Self {
        Self { status: 400, message: message.into() }
    }
    fn busy(message: impl Into<String>) -> Self {
        Self { status: 503, message: message.into() }
    }
}

/// Request limits (`PLOW_MM_*`).
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_images: usize,
    pub max_audio: usize,
    pub max_image_pixels: u64,
    pub max_audio_seconds: f64,
}

impl Limits {
    pub fn from_env() -> Self {
        let c = crate::config::RuntimeConfig::get();
        Self {
            max_images: c.mm_max_images,
            max_audio: c.mm_max_audio,
            max_image_pixels: c.mm_max_image_pixels,
            max_audio_seconds: c.mm_max_audio_seconds as f64,
        }
    }
}

/// Encoder rows of in-flight prompts in the LM: a host mirror of `in.mm_table` and the free slab
/// rows. Reservations are taken at submit (so staging never runs out), rows at staging.
pub struct Slab {
    hidden: usize,
    rows: u32,
    cap: u32,
    state: Mutex<SlabState>,
}

#[derive(Default)]
struct SlabState {
    reserved: u32,
    /// id -> (table slot, slab row, holders).
    live: HashMap<u32, (u32, u32, u32)>,
    free: Vec<u32>,
    /// Table slots holding a live id.
    used: Vec<bool>,
    /// Slots whose id was released; written as tombstones before the next insert.
    tombs: Vec<u32>,
}

const TOMBSTONE: u32 = 1;

impl Slab {
    fn new(c: &MmContract) -> Self {
        Self {
            hidden: c.hidden as usize,
            rows: c.slab_rows,
            cap: c.table_capacity,
            state: Mutex::new(SlabState {
                free: (0..c.slab_rows).rev().collect(),
                used: vec![false; c.table_capacity as usize],
                ..Default::default()
            }),
        }
    }

    /// (rows reserved by admitted requests, rows staged on the device, slab rows).
    pub fn occupancy(&self) -> (u32, u32, u32) {
        let s = self.state.lock();
        (s.reserved, s.live.len() as u32, self.rows)
    }

    fn reserve(&self, n: u32) -> bool {
        let mut s = self.state.lock();
        if s.reserved + n > self.rows {
            return false;
        }
        s.reserved += n;
        true
    }

    /// Put `job`'s rows on this engine's device through `write(tensor, offset, bytes)`, in stream
    /// order: slab rows before the table entries that name them. Then every bit-31 id of `prompt`
    /// must be in this table: a missing one would be served as the pad row, silently.
    pub fn stage(
        self: &Arc<Self>,
        job: &mut MmJob,
        prompt: &[u32],
        mut write: impl FnMut(&str, u64, &[u8]) -> crate::Result<()>,
    ) -> crate::Result<()> {
        if job.staged {
            return Ok(());
        }
        if !Arc::ptr_eq(&job.slab, self) {
            return Err(crate::RuntimeError::Rejected("multimodal rows were reserved on another engine".into()));
        }
        let mut s = self.state.lock();
        let tombs = std::mem::take(&mut s.tombs);
        for slot in tombs {
            if !s.used[slot as usize] {
                write(TABLE_TENSOR, u64::from(slot) * 8, &TOMBSTONE.to_le_bytes())?;
            }
        }
        let row_bytes = self.hidden * 2;
        for (i, &id) in job.ids.iter().enumerate() {
            if let Some(e) = s.live.get_mut(&id) {
                e.2 += 1;
                continue;
            }
            let row = s.free.pop().ok_or_else(|| crate::RuntimeError::Rejected("multimodal slab exhausted".into()))?;
            let mut slot = id & (self.cap - 1);
            while s.used[slot as usize] {
                slot = (slot + 1) & (self.cap - 1);
            }
            write(SLAB_TENSOR, u64::from(row) * row_bytes as u64, bytemuck::cast_slice(&job.rows[i * self.hidden..(i + 1) * self.hidden]))?;
            let entry: [u32; 2] = [id, row];
            write(TABLE_TENSOR, u64::from(slot) * 8, bytemuck::cast_slice(&entry))?;
            s.used[slot as usize] = true;
            s.live.insert(id, (slot, row, 1));
        }
        job.staged = true;
        match prompt.iter().find(|&&id| id & ROW_ID_BIT != 0 && !s.live.contains_key(&id)) {
            Some(id) => Err(crate::RuntimeError::Rejected(format!("multimodal row {id:#010x} is not resident on this engine"))),
            None => Ok(()),
        }
    }

    fn release(&self, ids: &[u32], staged: bool, reserved: u32) {
        let mut s = self.state.lock();
        s.reserved -= reserved;
        if !staged {
            return;
        }
        for id in ids {
            let Some(e) = s.live.get_mut(id) else { continue };
            e.2 -= 1;
            if e.2 == 0 {
                let (slot, row, _) = s.live.remove(id).unwrap();
                s.used[slot as usize] = false;
                s.free.push(row);
                s.tombs.push(slot);
            }
        }
    }
}

/// One request's soft-token rows: ids (bit 31, content hashed) and bf16 rows in prompt order.
pub struct MmJob {
    ids: Vec<u32>,
    rows: Vec<u16>,
    slab: Arc<Slab>,
    reserved: u32,
    staged: bool,
}

impl MmJob {
    pub fn staged(&self) -> bool {
        self.staged
    }

    /// Move the reservation to `slab` (the job was routed to another engine). `false` when it is full.
    pub fn rebind(&mut self, slab: &Arc<Slab>) -> bool {
        if Arc::ptr_eq(&self.slab, slab) {
            return true;
        }
        debug_assert!(!self.staged, "a staged job is not rerouted");
        if !slab.reserve(self.reserved) {
            return false;
        }
        self.slab.release(&[], false, self.reserved);
        self.slab = Arc::clone(slab);
        true
    }
}

impl Drop for MmJob {
    fn drop(&mut self) {
        self.slab.release(&self.ids, self.staged, self.reserved);
    }
}

enum Work {
    Images(Vec<media::Patches>, tokio::sync::oneshot::Sender<crate::Result<Vec<Vec<f32>>>>),
    Audio(media::Mel, usize, tokio::sync::oneshot::Sender<crate::Result<Vec<f32>>>),
}

/// An encoder on its own thread (its own launches; the LM dispatcher never waits on it).
struct Worker {
    tx: std::sync::mpsc::Sender<Work>,
}

impl Worker {
    fn spawn(path: PathBuf, device: u8, processor: Processor) -> crate::Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<Work>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("mm-encoder".into())
            .spawn(move || {
                let loaded = encoder::Encoder::load(&path, device).and_then(|e| {
                    if e.processor() == processor {
                        Ok(e)
                    } else {
                        Err(crate::RuntimeError::Rejected(format!(
                            "{}: encoder takes {} input but the contract preprocesses with {}; rebuild the packet",
                            path.display(),
                            e.processor().name(),
                            processor.name()
                        )))
                    }
                });
                let mut enc = match loaded {
                    Ok(e) => {
                        let _ = ready_tx.send(Ok(()));
                        e
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                tracing::info!(packet = %path.display(), "multimodal encoder loaded");
                while let Ok(work) = rx.recv() {
                    match work {
                        Work::Images(items, reply) => {
                            let refs: Vec<&media::Patches> = items.iter().collect();
                            let _ = reply.send(enc.encode_images(&refs));
                        }
                        Work::Audio(mel, tokens, reply) => {
                            let _ = reply.send(enc.encode_audio(&mel, tokens));
                        }
                    }
                }
            })
            .map_err(|e| crate::RuntimeError::Rejected(format!("encoder thread: {e}")))?;
        ready_rx.recv().map_err(|_| crate::RuntimeError::Rejected("encoder thread died".into()))??;
        Ok(Self { tx })
    }
}

/// One engine's multimodal runtime: its contract, slab and encoders (loaded on first use, on the
/// engine's device). Dropping it stops the encoder threads, which free their packets.
pub struct MmModel {
    pub contract: MmContract,
    dir: PathBuf,
    device: u8,
    slab: Arc<Slab>,
    workers: Mutex<HashMap<String, Arc<Mutex<Option<Worker>>>>>,
    pub stats: MmStats,
}

/// Per-engine media counters for `/metrics` (per request, never per token).
#[derive(Default)]
pub struct MmStats {
    /// Requests refused with 503 because the slab had no room.
    pub slab_full: AtomicU64,
    pub image: EncodeStats,
    pub audio: EncodeStats,
}

/// Encoder calls and their wall time from submit to rows ready (queueing included), µs.
#[derive(Default)]
pub struct EncodeStats {
    pub calls: AtomicU64,
    pub micros: AtomicU64,
}

impl MmStats {
    fn record(&self, s: &EncodeStats, since: std::time::Instant) {
        s.calls.fetch_add(1, Ordering::Relaxed);
        s.micros.fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

impl MmModel {
    pub fn new(dir: &Path, contract: &MmContract, device: u8) -> Self {
        Self {
            contract: contract.clone(),
            dir: dir.to_path_buf(),
            device,
            slab: Arc::new(Slab::new(contract)),
            workers: Mutex::new(HashMap::new()),
            stats: MmStats::default(),
        }
    }

    pub fn device(&self) -> u8 {
        self.device
    }

    pub fn slab(&self) -> &Arc<Slab> {
        &self.slab
    }

    fn worker(&self, m: &MmModality) -> crate::Result<std::sync::mpsc::Sender<Work>> {
        let cell = Arc::clone(self.workers.lock().entry(m.kind.clone()).or_default());
        let mut w = cell.lock();
        if w.is_none() {
            let processor = m.processor().map_err(crate::RuntimeError::Rejected)?;
            *w = Some(Worker::spawn(self.dir.join(&m.packet), self.device, processor)?);
        }
        Ok(w.as_ref().unwrap().tx.clone())
    }
}

/// Content parts with media, in conversation order (the order the template renders them).
pub fn media_parts(messages: &[Message]) -> Vec<&ContentPart> {
    messages
        .iter()
        .filter_map(|m| match &m.content {
            Some(crate::serve::openai::Content::Parts(parts)) => Some(parts.iter().filter(|p| p.media_kind().is_some())),
            _ => None,
        })
        .flatten()
        .collect()
}

fn patch_params(m: &MmModality) -> Result<media::PatchParams, MmError> {
    let u = |k: &str| m.param(k).map(|v| v as u32).ok_or_else(|| MmError::bad(format!("image contract lacks {k}")));
    let f = |k: &str, d: f32| m.param_f32(k).unwrap_or(d);
    Ok(media::PatchParams {
        patch: u("patch_size")?,
        pool: u("pool")?,
        max_soft_tokens: u("max_soft_tokens")?,
        rescale: f("rescale_f32", 1.0 / 255.0),
        normalize: m.param("normalize") == Some(1),
        mean: [f("mean0_f32", 0.0), f("mean1_f32", 0.0), f("mean2_f32", 0.0)],
        std: [f("std0_f32", 1.0), f("std1_f32", 1.0), f("std2_f32", 1.0)],
    })
}

fn mel_params(m: &MmModality) -> Result<media::MelParams, MmError> {
    let u = |k: &str| m.param(k).map(|v| v as usize).ok_or_else(|| MmError::bad(format!("audio contract lacks {k}")));
    let f = |k: &str| m.param_f32(k).map(f64::from).ok_or_else(|| MmError::bad(format!("audio contract lacks {k}")));
    Ok(media::MelParams {
        sample_rate: u("sample_rate")? as u32,
        frame_length: u("frame_length")?,
        hop_length: u("hop_length")?,
        fft_length: u("fft_length")?,
        mel_bins: u("mel_bins")?,
        min_frequency: f("min_frequency_f32")?,
        max_frequency: f("max_frequency_f32")?,
        mel_floor: f("mel_floor_f32")?,
        pad_multiple: u("pad_multiple").unwrap_or(1),
    })
}

/// A decoded, preprocessed media item and the digest its row ids derive from.
enum Prepared {
    Image(media::Patches, [u8; 32]),
    Audio(media::Mel, usize, [u8; 32]),
}

fn digest(kind: &str, bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(kind.as_bytes());
    h.update([0u8]);
    h.update(bytes);
    h.finalize().into()
}

/// Bit-31 ids of `rows` soft-token rows of the item with `digest`.
pub fn row_ids(digest: &[u8; 32], rows: usize) -> Vec<u32> {
    (0..rows as u32)
        .map(|k| {
            let mut h = Sha256::new();
            h.update(digest);
            h.update(k.to_le_bytes());
            let d = h.finalize();
            ROW_ID_BIT | (u32::from_le_bytes([d[0], d[1], d[2], d[3]]) & !ROW_ID_BIT)
        })
        .collect()
}

fn prepare_one(part: &ContentPart, contract: &MmContract, limits: &Limits) -> Result<Prepared, MmError> {
    let kind = part.media_kind().unwrap_or_default();
    let modality = contract
        .modality(kind)
        .ok_or_else(|| MmError::bad(format!("unsupported content type for this model: {kind}")))?;
    // Validated when the packet loaded; the kind of each processor matches its modality's.
    let processor = modality.processor().map_err(|e| MmError { status: 500, message: e })?;
    match (part, processor) {
        (
            ContentPart::ImageUrl { image_url: crate::serve::openai::ImageUrl { url, .. } } | ContentPart::InputImage { image_url: url },
            Processor::AspectPatches,
        ) => {
            if url.starts_with("http://") || url.starts_with("https://") {
                return Err(MmError::bad("image URLs must be data: URLs; this server does not fetch http(s) media"));
            }
            let (_, bytes) = media::data_url(url).map_err(MmError::bad)?;
            let img = media::decode_image(&bytes, limits.max_image_pixels).map_err(MmError::bad)?;
            let patches = media::image_patches(&img, &patch_params(modality)?).map_err(MmError::bad)?;
            Ok(Prepared::Image(patches, digest(kind, &img.data)))
        }
        (ContentPart::InputAudio { input_audio }, Processor::SemicausalLogMel | Processor::WaveformFrames) => {
            let format = input_audio.format.as_deref().unwrap_or("wav").to_ascii_lowercase();
            let bytes = media::base64_decode(&input_audio.data).map_err(MmError::bad)?;
            let (samples, rate) = media::decode_audio(&bytes, &format, limits.max_audio_seconds).map_err(MmError::bad)?;
            audio(samples, rate, modality, processor, limits, kind)
        }
        (ContentPart::AudioUrl { audio_url }, Processor::SemicausalLogMel | Processor::WaveformFrames) => {
            if audio_url.url.starts_with("http://") || audio_url.url.starts_with("https://") {
                return Err(MmError::bad("audio URLs must be data: URLs; this server does not fetch http(s) media"));
            }
            let (mime, bytes) = media::data_url(&audio_url.url).map_err(MmError::bad)?;
            let (samples, rate) = media::decode_audio(&bytes, media::mime_audio_format(&mime), limits.max_audio_seconds).map_err(MmError::bad)?;
            audio(samples, rate, modality, processor, limits, kind)
        }
        (_, p) => Err(MmError::bad(format!("a {kind} part cannot use the {} processor", p.name()))),
    }
}

fn audio(samples: Vec<f32>, rate: u32, m: &MmModality, processor: Processor, limits: &Limits, kind: &str) -> Result<Prepared, MmError> {
    let seconds = samples.len() as f64 / f64::from(rate.max(1));
    if processor == Processor::WaveformFrames {
        let u = |k: &str| m.param(k).map(|v| v as usize).ok_or_else(|| MmError::bad(format!("audio contract lacks {k}")));
        let (sample_rate, frame) = (u("sample_rate")? as u32, u("frame_samples")?);
        let cap = (u("max_soft_tokens")? * frame) as f64 / f64::from(sample_rate);
        if seconds > limits.max_audio_seconds.min(cap) {
            return Err(MmError::bad(format!(
                "audio clip is {seconds:.1} s; at most {:.1} s per clip",
                limits.max_audio_seconds.min(cap)
            )));
        }
        if samples.is_empty() {
            return Err(MmError::bad("audio clip is empty"));
        }
        let samples = media::resample(&samples, rate, sample_rate);
        let raw: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        let frames = media::waveform_frames(&samples, frame);
        let tokens = frames.valid_frames;
        return Ok(Prepared::Audio(frames, tokens, digest(kind, &raw)));
    }
    let p = mel_params(m)?;
    let cap = m.param("max_samples").map_or(f64::INFINITY, |s| s as f64 / f64::from(p.sample_rate));
    if seconds > limits.max_audio_seconds.min(cap) {
        return Err(MmError::bad(format!(
            "audio clip is {seconds:.1} s; at most {:.1} s per clip",
            limits.max_audio_seconds.min(cap)
        )));
    }
    if samples.is_empty() {
        return Err(MmError::bad("audio clip is empty"));
    }
    let samples = media::resample(&samples, rate, p.sample_rate);
    let mel = media::log_mel(&samples, &p);
    let tokens = media::audio_tokens(mel.valid_frames, m.param("subsample").unwrap_or(4) as usize);
    if tokens == 0 {
        return Err(MmError::bad("audio clip is too short"));
    }
    let raw: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(Prepared::Audio(mel, tokens, digest(kind, &raw)))
}

/// Round-to-nearest-even bf16 (the checkpoint's projection output dtype).
fn bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    (bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// A request's preprocessed media and the soft-token ids its prompt was expanded with, waiting
/// for the engine it is routed to ([`encode`]).
pub struct Pending {
    prepared: Vec<Prepared>,
    ids: Vec<Vec<u32>>,
}

fn soft_tokens(p: &Prepared) -> usize {
    match p {
        Prepared::Image(patches, _) => patches.soft_tokens as usize,
        Prepared::Audio(_, tokens, _) => *tokens,
    }
}

/// Expand `prompt_ids` for the media parts of `messages`: each placeholder the template rendered
/// becomes `begin, rows.., end`. Host work only, so the prompt can route before any encoder runs.
/// `None` when the conversation has no media.
pub async fn expand(
    contract: &MmContract,
    messages: &[Message],
    prompt_ids: &mut Vec<u32>,
    limits: Limits,
) -> Result<Option<Pending>, MmError> {
    let parts: Vec<ContentPart> = media_parts(messages).into_iter().cloned().collect();
    if parts.is_empty() {
        return Ok(None);
    }
    let images = parts.iter().filter(|p| p.media_kind() == Some("image")).count();
    let clips = parts.len() - images;
    if images > limits.max_images {
        return Err(MmError::bad(format!("{images} images; at most {} per request", limits.max_images)));
    }
    if clips > limits.max_audio {
        return Err(MmError::bad(format!("{clips} audio clips; at most {} per request", limits.max_audio)));
    }
    for p in &parts {
        let kind = p.media_kind().unwrap_or_default();
        if contract.modality(kind).is_none() {
            return Err(MmError::bad(format!("unsupported content type for this model: {kind}")));
        }
    }
    let prepared = {
        let contract = contract.clone();
        tokio::task::spawn_blocking(move || parts.iter().map(|p| prepare_one(p, &contract, &limits)).collect::<Result<Vec<_>, _>>())
            .await
            .map_err(|e| MmError::bad(format!("media preprocessing failed: {e}")))??
    };
    // The template's placeholders, in order, must be exactly the media parts.
    let mut slots: Vec<(usize, &MmModality)> = Vec::new();
    for (pos, id) in prompt_ids.iter().enumerate() {
        if let Some(m) = contract.modalities.iter().find(|m| m.placeholder == *id) {
            slots.push((pos, m));
        }
    }
    let kinds: Vec<&str> = prepared.iter().map(|p| if matches!(p, Prepared::Image(..)) { "image" } else { "audio" }).collect();
    if slots.len() != kinds.len() || slots.iter().zip(&kinds).any(|((_, m), k)| m.kind != *k) {
        return Err(MmError::bad(format!(
            "the chat template rendered {} media placeholders for {} media parts; the prompt text must not contain \
             placeholder tokens",
            slots.len(),
            kinds.len()
        )));
    }
    let ids: Vec<Vec<u32>> = prepared
        .iter()
        .map(|p| {
            let d = match p {
                Prepared::Image(_, d) | Prepared::Audio(_, _, d) => d,
            };
            row_ids(d, soft_tokens(p))
        })
        .collect();
    // Expand placeholders back to front so earlier positions stay valid.
    for ((pos, m), ids) in slots.iter().zip(&ids).rev() {
        let mut with: Vec<u32> = Vec::with_capacity(ids.len() + 2);
        with.extend(m.begin);
        with.extend(ids);
        with.extend(m.end);
        prompt_ids.splice(*pos..pos + 1, with);
    }
    if prompt_ids.last().is_some_and(|&t| t & ROW_ID_BIT != 0) {
        return Err(MmError::bad("a conversation may not end on a media item"));
    }
    Ok(Some(Pending { prepared, ids }))
}

/// Reserve `pending`'s rows in `mm`'s slab, then encode it on `mm`'s encoders (its engine's
/// device). The reservation is taken first so a full slab refuses before any encoder work, and
/// it is held by the job from then on: a cancelled or failed request returns it on drop.
pub async fn encode(mm: &MmModel, pending: Pending) -> Result<Box<MmJob>, MmError> {
    let Pending { prepared, ids } = pending;
    let contract = &mm.contract;
    let total: usize = ids.iter().map(Vec::len).sum();
    let n = u32::try_from(total).unwrap_or(u32::MAX);
    if n > mm.slab.rows {
        return Err(MmError::bad(format!(
            "the media need {total} soft-token rows; this model holds at most {} per request; send fewer or smaller items",
            mm.slab.rows
        )));
    }
    if !mm.slab.reserve(n) {
        mm.stats.slab_full.fetch_add(1, Ordering::Relaxed);
        return Err(MmError::busy(format!("multimodal rows are full ({} rows in flight); retry", mm.slab.rows)));
    }
    let mut job = Box::new(MmJob { ids: Vec::new(), rows: Vec::new(), slab: Arc::clone(&mm.slab), reserved: n, staged: false });
    let encode_err = |e: crate::RuntimeError| MmError { status: 500, message: format!("encoder failed: {e}") };
    // All images in one batch per launch rung, each clip on its own; both encoders start before
    // either is awaited.
    let mut image_index = Vec::new();
    let mut images = Vec::new();
    let mut clips = Vec::new();
    for (i, p) in prepared.into_iter().enumerate() {
        match p {
            Prepared::Image(patches, _) => {
                image_index.push(i);
                images.push(patches);
            }
            Prepared::Audio(mel, tokens, _) => clips.push((i, mel, tokens)),
        }
    }
    let started = std::time::Instant::now();
    let image_rx = match images.is_empty() {
        true => None,
        false => {
            let tx = mm.worker(contract.modality("image").unwrap()).map_err(encode_err)?;
            let (reply, rx) = tokio::sync::oneshot::channel();
            tx.send(Work::Images(images, reply)).map_err(|_| MmError::busy("image encoder is unavailable"))?;
            Some(rx)
        }
    };
    let mut clip_rx = Vec::with_capacity(clips.len());
    for (i, mel, tokens) in clips {
        let tx = mm.worker(contract.modality("audio").unwrap()).map_err(encode_err)?;
        let (reply, rx) = tokio::sync::oneshot::channel();
        tx.send(Work::Audio(mel, tokens, reply)).map_err(|_| MmError::busy("audio encoder is unavailable"))?;
        clip_rx.push((i, rx));
    }
    let mut rows: Vec<Option<Vec<f32>>> = vec![None; ids.len()];
    if let Some(rx) = image_rx {
        let out = rx.await.map_err(|_| MmError::busy("image encoder is unavailable"))?.map_err(encode_err)?;
        mm.stats.record(&mm.stats.image, started);
        for (i, r) in image_index.into_iter().zip(out) {
            rows[i] = Some(r);
        }
    }
    for (i, rx) in clip_rx {
        rows[i] = Some(rx.await.map_err(|_| MmError::busy("audio encoder is unavailable"))?.map_err(encode_err)?);
        mm.stats.record(&mm.stats.audio, started);
    }
    let hidden = contract.hidden as usize;
    job.ids.reserve_exact(total);
    job.rows.reserve_exact(total * hidden);
    for (ids, r) in ids.iter().zip(rows) {
        let r = r.unwrap_or_default();
        if r.len() != ids.len() * hidden {
            return Err(MmError {
                status: 500,
                message: format!("encoder returned {} values for {} soft tokens of width {hidden}", r.len(), ids.len()),
            });
        }
        job.ids.extend(ids);
        job.rows.extend(r.iter().map(|v| bf16_bits(*v)));
    }
    Ok(job)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> MmContract {
        MmContract { version: 1, hidden: 2, pad_token: 0, slab_rows: 4, table_capacity: 8, modalities: Vec::new() }
    }

    #[test]
    fn row_ids_are_content_keyed_and_tagged() {
        let a = row_ids(&digest("image", b"one"), 3);
        let b = row_ids(&digest("image", b"two"), 3);
        assert!(a.iter().chain(&b).all(|&id| id & ROW_ID_BIT != 0 && id > TOMBSTONE));
        assert_ne!(a, b);
        assert_eq!(a, row_ids(&digest("image", b"one"), 3));
        assert_ne!(a[0], a[1]);
    }

    #[test]
    fn slab_stages_shares_and_releases() {
        let slab = Arc::new(Slab::new(&contract()));
        let table = std::cell::RefCell::new(vec![0u32; 16]);
        let slab_rows = std::cell::RefCell::new(vec![0u16; 8]);
        let write = |t: &str, off: u64, b: &[u8]| -> crate::Result<()> {
            if t == TABLE_TENSOR {
                let words: &[u32] = bytemuck::cast_slice(b);
                table.borrow_mut()[off as usize / 4..off as usize / 4 + words.len()].copy_from_slice(words);
            } else {
                let h: &[u16] = bytemuck::cast_slice(b);
                slab_rows.borrow_mut()[off as usize / 2..off as usize / 2 + h.len()].copy_from_slice(h);
            }
            Ok(())
        };
        assert!(slab.reserve(2));
        let mut a = MmJob { ids: vec![0x8000_0005, 0x8000_000d], rows: vec![1, 2, 3, 4], slab: slab.clone(), reserved: 2, staged: false };
        slab.stage(&mut a, &[], write).unwrap();
        // Both ids home at slot 5; the second probes to 6. Rows come off the free list from 0.
        assert_eq!(&table.borrow()[10..14], &[0x8000_0005, 0, 0x8000_000d, 1]);
        assert_eq!(&slab_rows.borrow()[0..4], &[1, 2, 3, 4]);
        assert!(slab.reserve(2));
        assert!(!slab.reserve(1));
        let mut b = MmJob { ids: vec![0x8000_000d], rows: vec![3, 4], slab: slab.clone(), reserved: 2, staged: false };
        slab.stage(&mut b, &[], write).unwrap();
        drop(a);
        // 0x...d is still held by `b`; 0x...5 is a tombstone at the next stage.
        assert!(slab.reserve(1));
        let mut c = MmJob { ids: vec![0x8000_0105], rows: vec![9, 9], slab: slab.clone(), reserved: 1, staged: false };
        slab.stage(&mut c, &[], write).unwrap();
        assert_eq!(table.borrow()[10], 0x8000_0105, "a freed slot is reused");
        drop(b);
        drop(c);
        assert!(slab.reserve(4));
    }

    /// A device table the writes land in, as `(table words, slab halves)`.
    type Device = std::cell::RefCell<(Vec<u32>, Vec<u16>)>;

    fn device() -> Device {
        std::cell::RefCell::new((vec![0; 16], vec![0; 8]))
    }

    fn write_to(d: &Device) -> impl FnMut(&str, u64, &[u8]) -> crate::Result<()> + '_ {
        move |t, off, b| {
            let mut d = d.borrow_mut();
            if t == TABLE_TENSOR {
                let words: &[u32] = bytemuck::cast_slice(b);
                d.0[off as usize / 4..off as usize / 4 + words.len()].copy_from_slice(words);
            } else {
                let h: &[u16] = bytemuck::cast_slice(b);
                d.1[off as usize / 2..off as usize / 2 + h.len()].copy_from_slice(h);
            }
            Ok(())
        }
    }

    fn job(slab: &Arc<Slab>, ids: &[u32]) -> MmJob {
        assert!(slab.reserve(ids.len() as u32));
        MmJob { ids: ids.to_vec(), rows: vec![7; ids.len() * 2], slab: slab.clone(), reserved: ids.len() as u32, staged: false }
    }

    #[test]
    fn each_rank_stages_the_same_media_into_its_own_table() {
        let c = contract();
        let (rank0, rank1) = (Arc::new(Slab::new(&c)), Arc::new(Slab::new(&c)));
        let (dev0, dev1) = (device(), device());
        let id = 0x8000_0005;
        let prompt = [2, id, 3];
        let mut a = job(&rank0, &[id]);
        rank0.stage(&mut a, &prompt, write_to(&dev0)).unwrap();
        // Same image, concurrently on the other rank: its own table must get the row too.
        let mut b = job(&rank1, &[id]);
        rank1.stage(&mut b, &prompt, write_to(&dev1)).unwrap();
        for d in [&dev0, &dev1] {
            assert_eq!(&d.borrow().0[10..12], &[id, 0]);
            assert_eq!(&d.borrow().1[0..2], &[7, 7]);
        }
    }

    #[test]
    fn rows_reserved_on_another_engine_are_refused_until_rebound() {
        let c = contract();
        let (rank0, rank1) = (Arc::new(Slab::new(&c)), Arc::new(Slab::new(&c)));
        let dev1 = device();
        let id = 0x8000_0005;
        let mut j = job(&rank0, &[id]);
        assert!(rank1.stage(&mut j, &[id, 1], write_to(&dev1)).is_err());
        assert_eq!(dev1.borrow().0, vec![0; 16], "nothing reaches the other engine's table");
        assert!(j.rebind(&rank1));
        assert!(rank0.reserve(4), "the old engine's reservation is returned");
        assert!(!rank1.reserve(4), "the new engine holds it");
        rank1.stage(&mut j, &[id, 1], write_to(&dev1)).unwrap();
        assert_eq!(&dev1.borrow().0[10..12], &[id, 0]);
    }

    fn run<T>(f: impl std::future::Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(f)
    }

    #[test]
    fn slab_refusals_come_before_encoder_work_and_return_their_rows() {
        let mm = MmModel::new(Path::new("/nonexistent"), &contract(), 0);
        let pending = |n: usize| Pending { prepared: Vec::new(), ids: vec![(0..n as u32).map(|k| ROW_ID_BIT | (k + 2)).collect()] };
        let err = run(encode(&mm, pending(5))).err().unwrap();
        assert_eq!(err.status, 400, "more rows than the slab holds never fit: {}", err.message);
        assert!(mm.slab.reserve(3));
        let err = run(encode(&mm, pending(2))).err().unwrap();
        assert_eq!((err.status, mm.stats.slab_full.load(Ordering::Relaxed)), (503, 1), "{}", err.message);
        mm.slab.release(&[], false, 3);
        // Admitted, then the encoder step fails (no rows here): the reservation is returned.
        let err = run(encode(&mm, pending(4))).err().unwrap();
        assert_eq!(err.status, 500, "{}", err.message);
        assert_eq!(mm.slab.occupancy(), (0, 0, 4));
    }

    #[test]
    fn occupancy_tracks_reservations_and_staged_rows() {
        let slab = Arc::new(Slab::new(&contract()));
        let dev = device();
        let mut a = job(&slab, &[0x8000_0005, 0x8000_0006]);
        assert_eq!(slab.occupancy(), (2, 0, 4));
        slab.stage(&mut a, &[], write_to(&dev)).unwrap();
        let mut b = job(&slab, &[0x8000_0006]);
        slab.stage(&mut b, &[], write_to(&dev)).unwrap();
        assert_eq!(slab.occupancy(), (3, 2, 4), "a shared id stages once");
        drop(a);
        assert_eq!(slab.occupancy(), (1, 1, 4));
        drop(b);
        assert_eq!(slab.occupancy(), (0, 0, 4), "no rows leak");
    }

    #[test]
    fn a_prompt_row_missing_from_the_table_fails_the_request() {
        let slab = Arc::new(Slab::new(&contract()));
        let dev = device();
        let mut j = job(&slab, &[0x8000_0005]);
        let err = slab.stage(&mut j, &[1, 0x8000_0005, 0x8000_0009, 2], write_to(&dev)).unwrap_err();
        assert!(err.to_string().contains("0x80000009"), "{err}");
        drop(j);
        assert!(slab.reserve(4), "the failed job releases what it staged");
    }
}
