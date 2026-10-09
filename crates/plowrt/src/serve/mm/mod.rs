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

pub mod encoder;
pub mod media;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use plow_asset::multimodal::{MmContract, MmModality, ROW_ID_BIT, SLAB_TENSOR, TABLE_TENSOR};
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

    fn reserve(&self, n: u32) -> bool {
        let mut s = self.state.lock();
        if s.reserved + n > self.rows {
            return false;
        }
        s.reserved += n;
        true
    }

    /// Put `job`'s rows on the device through `write(tensor, offset, bytes)`, in stream order:
    /// slab rows before the table entries that name them.
    pub fn stage(&self, job: &mut MmJob, mut write: impl FnMut(&str, u64, &[u8]) -> crate::Result<()>) -> crate::Result<()> {
        if job.staged {
            return Ok(());
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
        Ok(())
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
    pub fn slab(&self) -> Arc<Slab> {
        Arc::clone(&self.slab)
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
    fn spawn(path: PathBuf, device: u8) -> crate::Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<Work>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("mm-encoder".into())
            .spawn(move || {
                let mut enc = match encoder::Encoder::load(&path, device) {
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

/// A model's multimodal runtime: its contract, slab and encoders (loaded on first use).
pub struct MmModel {
    pub contract: MmContract,
    dir: PathBuf,
    device: u8,
    slab: Arc<Slab>,
    workers: Mutex<HashMap<String, Arc<Mutex<Option<Worker>>>>>,
}

impl MmModel {
    fn worker(&self, m: &MmModality) -> crate::Result<std::sync::mpsc::Sender<Work>> {
        let cell = Arc::clone(self.workers.lock().entry(m.kind.clone()).or_default());
        let mut w = cell.lock();
        if w.is_none() {
            *w = Some(Worker::spawn(self.dir.join(&m.packet), self.device)?);
        }
        Ok(w.as_ref().unwrap().tx.clone())
    }
}

/// The multimodal runtime of the bundle in `dir` with `contract`, created once per bundle.
pub fn model(dir: &Path, contract: &MmContract, device: u8) -> Arc<MmModel> {
    static MODELS: std::sync::OnceLock<Mutex<HashMap<PathBuf, Arc<MmModel>>>> = std::sync::OnceLock::new();
    let mut map = MODELS.get_or_init(Default::default).lock();
    Arc::clone(map.entry(dir.to_path_buf()).or_insert_with(|| {
        Arc::new(MmModel {
            contract: contract.clone(),
            dir: dir.to_path_buf(),
            device,
            slab: Arc::new(Slab::new(contract)),
            workers: Mutex::new(HashMap::new()),
        })
    }))
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
    match part {
        ContentPart::ImageUrl { image_url: crate::serve::openai::ImageUrl { url, .. } } | ContentPart::InputImage { image_url: url } => {
            if url.starts_with("http://") || url.starts_with("https://") {
                return Err(MmError::bad("image URLs must be data: URLs; this server does not fetch http(s) media"));
            }
            let (_, bytes) = media::data_url(url).map_err(MmError::bad)?;
            let img = media::decode_image(&bytes, limits.max_image_pixels).map_err(MmError::bad)?;
            let patches = media::image_patches(&img, &patch_params(modality)?).map_err(MmError::bad)?;
            Ok(Prepared::Image(patches, digest(kind, &img.data)))
        }
        ContentPart::InputAudio { input_audio } => {
            let format = input_audio.format.as_deref().unwrap_or("wav").to_ascii_lowercase();
            if format != "wav" {
                return Err(MmError::bad(format!("input_audio format {format:?} is not supported; send wav")));
            }
            let bytes = media::base64_decode(&input_audio.data).map_err(MmError::bad)?;
            audio(&bytes, modality, limits, kind)
        }
        ContentPart::AudioUrl { audio_url } => {
            if audio_url.url.starts_with("http://") || audio_url.url.starts_with("https://") {
                return Err(MmError::bad("audio URLs must be data: URLs; this server does not fetch http(s) media"));
            }
            let (_, bytes) = media::data_url(&audio_url.url).map_err(MmError::bad)?;
            audio(&bytes, modality, limits, kind)
        }
        _ => Err(MmError::bad("not a media part")),
    }
}

fn audio(bytes: &[u8], m: &MmModality, limits: &Limits, kind: &str) -> Result<Prepared, MmError> {
    let (samples, rate) = media::decode_wav(bytes).map_err(MmError::bad)?;
    let seconds = samples.len() as f64 / f64::from(rate.max(1));
    if m.processor == "waveform_frames" {
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

/// Expand `prompt_ids` for the media parts of `messages` and encode them: each placeholder the
/// template rendered becomes `begin, rows.., end`. `None` when the conversation has no media.
pub async fn prepare(
    mm: &Arc<MmModel>,
    messages: &[Message],
    prompt_ids: &mut Vec<u32>,
    limits: Limits,
) -> Result<Option<Box<MmJob>>, MmError> {
    let parts: Vec<ContentPart> = media_parts(messages).into_iter().cloned().collect();
    if parts.is_empty() {
        return Ok(None);
    }
    let contract = mm.contract.clone();
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
    // Encode: all images in one batch per launch rung, each clip on its own.
    let mut rows: Vec<Option<Vec<f32>>> = vec![None; prepared.len()];
    let image_items: Vec<(usize, media::Patches)> = prepared
        .iter()
        .enumerate()
        .filter_map(|(i, p)| match p {
            Prepared::Image(patches, _) => Some((
                i,
                media::Patches {
                    values: patches.values.clone(),
                    positions: patches.positions.clone(),
                    grid: patches.grid,
                    soft_tokens: patches.soft_tokens,
                },
            )),
            _ => None,
        })
        .collect();
    let encode_err = |e: crate::RuntimeError| MmError { status: 500, message: format!("encoder failed: {e}") };
    if !image_items.is_empty() {
        let m = contract.modality("image").unwrap();
        let tx = mm.worker(m).map_err(encode_err)?;
        let (reply, rx) = tokio::sync::oneshot::channel();
        let (index, items): (Vec<usize>, Vec<media::Patches>) = image_items.into_iter().unzip();
        tx.send(Work::Images(items, reply)).map_err(|_| MmError::busy("image encoder is unavailable"))?;
        let out = rx.await.map_err(|_| MmError::busy("image encoder is unavailable"))?.map_err(encode_err)?;
        for (i, r) in index.into_iter().zip(out) {
            rows[i] = Some(r);
        }
    }
    for (i, p) in prepared.iter().enumerate() {
        if let Prepared::Audio(mel, tokens, _) = p {
            let m = contract.modality("audio").unwrap();
            let tx = mm.worker(m).map_err(encode_err)?;
            let (reply, rx) = tokio::sync::oneshot::channel();
            let mel = media::Mel { values: mel.values.clone(), frames: mel.frames, valid_frames: mel.valid_frames };
            tx.send(Work::Audio(mel, *tokens, reply)).map_err(|_| MmError::busy("audio encoder is unavailable"))?;
            rows[i] = Some(rx.await.map_err(|_| MmError::busy("audio encoder is unavailable"))?.map_err(encode_err)?);
        }
    }
    // Expand placeholders back to front so earlier positions stay valid.
    let hidden = contract.hidden as usize;
    let mut ids_all: Vec<Vec<u32>> = Vec::with_capacity(prepared.len());
    for (p, r) in prepared.iter().zip(&rows) {
        let n = r.as_ref().map_or(0, |r| r.len() / hidden);
        let d = match p {
            Prepared::Image(_, d) | Prepared::Audio(_, _, d) => d,
        };
        ids_all.push(row_ids(d, n));
    }
    for ((pos, m), ids) in slots.iter().zip(&ids_all).rev() {
        let mut with: Vec<u32> = Vec::with_capacity(ids.len() + 2);
        with.extend(m.begin);
        with.extend(ids);
        with.extend(m.end);
        prompt_ids.splice(*pos..pos + 1, with);
    }
    if prompt_ids.last().is_some_and(|&t| t & ROW_ID_BIT != 0) {
        return Err(MmError::bad("a conversation may not end on a media item"));
    }
    let total: usize = ids_all.iter().map(Vec::len).sum();
    let mut job_ids = Vec::with_capacity(total);
    let mut job_rows = Vec::with_capacity(total * hidden);
    for (ids, r) in ids_all.iter().zip(rows) {
        job_ids.extend(ids);
        job_rows.extend(r.unwrap_or_default().iter().map(|v| (v.to_bits() >> 16) as u16));
    }
    let n = total as u32;
    if !mm.slab.reserve(n) {
        return Err(MmError::busy(format!("multimodal rows are full ({} in flight); retry", mm.slab.rows)));
    }
    Ok(Some(Box::new(MmJob { ids: job_ids, rows: job_rows, slab: Arc::clone(&mm.slab), reserved: n, staged: false })))
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
        slab.stage(&mut a, write).unwrap();
        // Both ids home at slot 5; the second probes to 6. Rows come off the free list from 0.
        assert_eq!(&table.borrow()[10..14], &[0x8000_0005, 0, 0x8000_000d, 1]);
        assert_eq!(&slab_rows.borrow()[0..4], &[1, 2, 3, 4]);
        assert!(slab.reserve(2));
        assert!(!slab.reserve(1));
        let mut b = MmJob { ids: vec![0x8000_000d], rows: vec![3, 4], slab: slab.clone(), reserved: 2, staged: false };
        slab.stage(&mut b, write).unwrap();
        drop(a);
        // 0x...d is still held by `b`; 0x...5 is a tombstone at the next stage.
        assert!(slab.reserve(1));
        let mut c = MmJob { ids: vec![0x8000_0105], rows: vec![9, 9], slab: slab.clone(), reserved: 1, staged: false };
        slab.stage(&mut c, write).unwrap();
        assert_eq!(table.borrow()[10], 0x8000_0105, "a freed slot is reused");
        drop(b);
        drop(c);
        assert!(slab.reserve(4));
    }
}
