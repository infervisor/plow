//! The codec stage of a speech pipeline: a `codec.v1` packet (codes or speech tokens in, PCM out,
//! one program or program sequence per (batch, frames) capacity: roles `decode.b{B}.f{F}` or
//! `synth.b{B}.t{T}.{stage}`) on the packet runtime, driven by one worker thread. Pending jobs share
//! each launch the way concurrent streams share LM decode steps; per-item valid lengths, seeds and
//! voice indices make every item decode exactly as it would alone.

use std::path::Path;
use std::sync::mpsc;

use parking_lot::Mutex;

use crate::exec::packet_runtime::{load_packet_runtime, PacketAsset, PacketTensor};

pub const PACKET: &str = "codec.pkt";
const DRIVER: &str = "codec.v1";

/// Launch order of pending work, most urgent first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Urgency {
    /// A stream's first audio.
    First,
    /// A stream's later windows.
    Stream,
    /// A whole utterance nobody hears until it is done.
    Whole,
}

struct Job {
    codes: Vec<i32>,
    frames: usize,
    seed: u64,
    voice: u32,
    urgency: Urgency,
    reply: tokio::sync::oneshot::Sender<Result<Vec<f32>, String>>,
    /// Backlog charged to the feeding model's mux until this job is answered.
    _work: Option<crate::sched::admission::DownstreamWork>,
}

pub struct Codec {
    tx: Mutex<mpsc::Sender<Job>>,
    credit: Option<std::sync::Arc<crate::sched::admission::DownstreamCredit>>,
    /// Largest batch one launch holds.
    pub max_batch: usize,
    pub max_frames: usize,
    /// Frames the smallest capacity holds: a shorter window costs as much.
    pub min_frames: usize,
    pub frame_codes: usize,
    pub frame_samples: usize,
    /// Streaming: frames of context a decode window carries, and right context before a frame
    /// is final.
    pub window: usize,
    pub lookahead: usize,
    /// Voice names in voice-index order (empty when the packet takes no voice).
    pub voices: Vec<String>,
    /// Optional packet parameters (e.g. a streaming schedule).
    pub parameters: std::collections::BTreeMap<String, u64>,
}

struct Bound {
    runtime: Box<dyn crate::exec::packet_runtime::PacketRuntime>,
    /// (batch, frames, programs in order), ascending by batch * frames.
    capacities: Vec<(usize, usize, Vec<usize>)>,
    codes: PacketTensor,
    seed: PacketTensor,
    voice: Option<PacketTensor>,
    pcm: PacketTensor,
    /// Valid-length tensor and rows per frame at each time resolution.
    lengths: Vec<(PacketTensor, usize)>,
    frame_codes: usize,
    frame_samples: usize,
    window: usize,
    lookahead: usize,
    voices: Vec<String>,
    parameters: std::collections::BTreeMap<String, u64>,
}

impl Codec {
    pub fn load(assets: &Path) -> Result<Self, String> {
        let path = assets.join(PACKET);
        if !path.is_file() {
            return Err(format!("{} missing: emit it with PLOW_TTS_CODEC_DIR (docs/runtime/tts.md)", path.display()));
        }
        Self::load_packet(&path)
    }

    pub fn load_packet(path: &Path) -> Result<Self, String> {
        let path = path.to_path_buf();
        let (tx, rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("plow-tts-codec".into())
            .spawn(move || {
                let bound = match bind(&path) {
                    Ok(b) => b,
                    Err(e) => return drop(ready_tx.send(Err(e))),
                };
                let info = (
                    bound.capacities.iter().map(|c| c.0).max().unwrap_or(1),
                    bound.capacities.iter().map(|c| c.1).max().unwrap_or(0),
                    bound.capacities.iter().map(|c| c.1).min().unwrap_or(0),
                    bound.frame_codes,
                    bound.frame_samples,
                    bound.window,
                    bound.lookahead,
                    bound.voices.clone(),
                    bound.parameters.clone(),
                );
                let _ = ready_tx.send(Ok(info));
                run(rx, bound);
            })
            .map_err(|e| e.to_string())?;
        let (max_batch, max_frames, min_frames, frame_codes, frame_samples, window, lookahead, voices, parameters) =
            ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Codec {
            tx: Mutex::new(tx),
            credit: None,
            max_batch,
            max_frames,
            min_frames,
            frame_codes,
            frame_samples,
            window,
            lookahead,
            voices,
            parameters,
        })
    }

    /// Charge this stage's backlog to the feeding model's admission: past two full launches of
    /// pending work, the model seats no new request.
    pub fn couple(&mut self, credit: std::sync::Arc<crate::sched::admission::DownstreamCredit>) {
        credit.set_limit(2 * self.max_batch);
        self.credit = Some(credit);
    }

    /// `frames * frame_codes` codebook ids -> `frames * frame_samples` samples.
    pub async fn decode(&self, codes: Vec<i32>, frames: usize, seed: u64, urgency: Urgency) -> Result<Vec<f32>, String> {
        self.decode_as(codes, frames, seed, 0, urgency).await
    }

    /// [`Self::decode`] for the packet's voice `voice` (an index into [`Self::voices`]).
    pub async fn decode_voice(&self, codes: Vec<i32>, frames: usize, seed: u64, voice: u32) -> Result<Vec<f32>, String> {
        self.decode_as(codes, frames, seed, voice, Urgency::Stream).await
    }

    async fn decode_as(&self, codes: Vec<i32>, frames: usize, seed: u64, voice: u32, urgency: Urgency) -> Result<Vec<f32>, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let _work = self.credit.as_ref().map(|c| c.work());
        self.tx
            .lock()
            .send(Job { codes, frames, seed, voice, urgency, reply, _work })
            .map_err(|_| "codec worker stopped".to_string())?;
        rx.await.map_err(|_| "codec worker dropped the job".to_string())?
    }
}

fn bind(path: &Path) -> Result<Bound, String> {
    let e = |x: crate::RuntimeError| x.to_string();
    let loaded = load_packet_runtime(path, "cuda").map_err(e)?;
    let runtime = loaded.runtime;
    let asset = PacketAsset::load(path).map_err(e)?;
    let pipeline = asset.bind_driver(DRIVER, runtime.as_ref()).map_err(e)?;
    let param = |k: &str| pipeline.parameter(k).map(|v| v as usize).map_err(e);
    let mut sequences: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize)>> = Default::default();
    for (role, program) in pipeline.programs() {
        let parts: Vec<&str> = role.split('.').collect();
        let dims = |b: &str, u: &str, bp: &str, up: &str| {
            Some((b.strip_prefix(bp)?.parse::<usize>().ok()?, u.strip_prefix(up)?.parse::<usize>().ok()?))
        };
        match parts.as_slice() {
            ["decode", b, f] => {
                if let Some(key) = dims(b, f, "b", "f") {
                    sequences.entry(key).or_default().push((0, program));
                }
            }
            ["synth", b, t, stage] => {
                if let (Some(key), Ok(stage)) = (dims(b, t, "b", "t"), stage.parse::<usize>()) {
                    sequences.entry(key).or_default().push((stage, program));
                }
            }
            _ => {}
        }
    }
    let mut capacities: Vec<(usize, usize, Vec<usize>)> = sequences
        .into_iter()
        .map(|((b, f), mut stages)| {
            stages.sort();
            (b, f, stages.into_iter().map(|(_, p)| p).collect())
        })
        .collect();
    if capacities.is_empty() {
        return Err("codec packet declares no decode capacity".into());
    }
    capacities.sort_by_key(|&(b, f, _)| (b * f, f));
    // Run every capacity once now: each program sequence is captured as a CUDA graph on first
    // use, and a capture that overlaps another thread's context synchronize (the LM engine) fails.
    let t = std::time::Instant::now();
    let mut runtime = runtime;
    for (_, _, programs) in &capacities {
        runtime.run_sequence(programs).map_err(e)?;
    }
    tracing::info!(capacities = capacities.len(), ms = t.elapsed().as_millis() as u64, "codec graphs warmed");
    let lengths = (0..param("lengths.count")?)
        .map(|k| Ok((pipeline.tensor(&format!("lengths.{k}")).map_err(e)?, param(&format!("lengths.{k}.rows_per_frame"))?)))
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Bound {
        codes: pipeline.tensor("codes").map_err(e)?,
        seed: pipeline.tensor("seed").map_err(e)?,
        pcm: pipeline.tensor("pcm").map_err(e)?,
        frame_codes: param("codec.frame_codes")?,
        frame_samples: param("codec.frame_samples")?,
        window: param("stream.window_frames")?,
        lookahead: param("stream.lookahead_frames")?,
        voice: pipeline.tensor("voice").ok(),
        voices: pipeline.optional_string("voices").map(|v| v.lines().map(str::to_owned).collect()).unwrap_or_default(),
        parameters: asset
            .pipelines()
            .iter()
            .find(|p| p.driver == DRIVER)
            .map(|p| p.parameters.clone())
            .unwrap_or_default(),
        lengths,
        capacities,
        runtime,
    })
}

impl Bound {
    /// The smallest capacity holding `batch` items of `frames` frames.
    fn capacity(&self, batch: usize, frames: usize) -> Option<&(usize, usize, Vec<usize>)> {
        self.capacities.iter().find(|&&(b, f, _)| b >= batch && f >= frames)
    }

    fn decode(&mut self, jobs: &[Job]) -> Result<Vec<Vec<f32>>, String> {
        let frames = jobs.iter().map(|j| j.frames).max().unwrap_or(0);
        let (cb, cf, programs) = self
            .capacity(jobs.len(), frames)
            .cloned()
            .ok_or_else(|| format!("{} x {frames} frames exceeds every codec capacity", jobs.len()))?;
        let fc = self.frame_codes;
        // Only the capacity's region of the (widest-capacity) codes and PCM tensors moves: the
        // whole PCM tensor is hundreds of MB.
        let mut codes = vec![0u32; cb * cf * fc];
        for (i, j) in jobs.iter().enumerate() {
            for (k, &c) in j.codes.iter().enumerate() {
                codes[i * cf * fc + k] = c.clamp(0, i32::MAX) as u32;
            }
        }
        let e = |x: crate::RuntimeError| x.to_string();
        self.runtime.write_tensor_at(self.codes, 0, bytemuck::cast_slice(&codes)).map_err(e)?;
        for &(tensor, per_frame) in &self.lengths {
            let mut len = vec![0u32; tensor.bytes / 4];
            for (i, j) in jobs.iter().enumerate() {
                len[i] = (j.frames * per_frame) as u32;
            }
            self.runtime.write_tensor(tensor, bytemuck::cast_slice(&len)).map_err(e)?;
        }
        let mut seeds = vec![0u64; self.seed.bytes / 8];
        for (i, j) in jobs.iter().enumerate() {
            seeds[i] = j.seed;
        }
        self.runtime.write_tensor(self.seed, bytemuck::cast_slice(&seeds)).map_err(e)?;
        if let Some(voice) = self.voice {
            let mut v = vec![0u32; voice.bytes / 4];
            for (i, j) in jobs.iter().enumerate() {
                v[i] = j.voice;
            }
            self.runtime.write_tensor(voice, bytemuck::cast_slice(&v)).map_err(e)?;
        }
        self.runtime.run_sequence(&programs).map_err(e)?;
        let mut pcm = vec![0f32; cb * cf * self.frame_samples];
        self.runtime.read_tensor_at(self.pcm, 0, bytemuck::cast_slice_mut(&mut pcm)).map_err(e)?;
        let per = cf * self.frame_samples;
        debug_assert!(cb * per <= pcm.len());
        Ok(jobs.iter().enumerate().map(|(i, j)| pcm[i * per..i * per + j.frames * self.frame_samples].to_vec()).collect())
    }
}

fn run(rx: mpsc::Receiver<Job>, mut codec: Bound) {
    let max_batch = codec.capacities.iter().map(|c| c.0).max().unwrap_or(1);
    let mut pending: Vec<Job> = Vec::new();
    while let Ok(first) = rx.recv() {
        pending.push(first);
        pending.extend(rx.try_iter());
        // Most urgent first; within an urgency longest first, so a launch's capacity is mostly
        // set by its first job.
        pending.sort_by_key(|j| (j.urgency, std::cmp::Reverse(j.frames)));
        while !pending.is_empty() {
            let mut frames = pending[0].frames;
            let mut n = 1;
            while n < pending.len()
                && n < max_batch
                && codec.capacity(n + 1, frames.max(pending[n].frames)).is_some()
            {
                frames = frames.max(pending[n].frames);
                n += 1;
            }
            let batch: Vec<Job> = pending.drain(..n).collect();
            if frames == 0 || batch.iter().any(|j| j.codes.len() != j.frames * codec.frame_codes) {
                for j in batch {
                    let _ = j.reply.send(Err("codec job has no frames or a partial frame".into()));
                }
                continue;
            }
            match codec.decode(&batch) {
                Ok(pcm) => {
                    for (j, p) in batch.into_iter().zip(pcm) {
                        let _ = j.reply.send(Ok(p));
                    }
                }
                Err(e) => {
                    for j in batch {
                        let _ = j.reply.send(Err(e.clone()));
                    }
                }
            }
        }
    }
}
