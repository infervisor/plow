//! The codec stage of a speech pipeline: the asset's `codec.pkt` (`codec.v1` driver: codes in,
//! PCM out, one program per (batch, frames) capacity) on the packet runtime, driven by one worker
//! thread. Pending jobs share each launch the way concurrent streams share LM decode steps; per-item
//! valid lengths make a partly filled capacity decode exactly like its own size.

use std::path::Path;
use std::sync::mpsc;

use parking_lot::Mutex;

use crate::exec::packet_runtime::{load_packet_runtime, PacketAsset, PacketTensor};

pub const PACKET: &str = "codec.pkt";
const DRIVER: &str = "codec.v1";

struct Job {
    codes: Vec<i32>,
    frames: usize,
    seed: u64,
    reply: tokio::sync::oneshot::Sender<Result<Vec<f32>, String>>,
}

pub struct Codec {
    tx: Mutex<mpsc::Sender<Job>>,
    pub max_frames: usize,
    pub frame_codes: usize,
    pub frame_samples: usize,
    /// Streaming: frames of context a decode window carries, and right context before a frame
    /// is final.
    pub window: usize,
    pub lookahead: usize,
}

struct Bound {
    runtime: Box<dyn crate::exec::packet_runtime::PacketRuntime>,
    /// (batch, frames, program), ascending by batch * frames.
    capacities: Vec<(usize, usize, usize)>,
    codes: PacketTensor,
    seed: PacketTensor,
    pcm: PacketTensor,
    /// Valid-length tensor and rows per frame at each time resolution.
    lengths: Vec<(PacketTensor, usize)>,
    frame_codes: usize,
    frame_samples: usize,
    window: usize,
    lookahead: usize,
}

impl Codec {
    pub fn load(assets: &Path) -> Result<Self, String> {
        let path = assets.join(PACKET);
        if !path.is_file() {
            return Err(format!("{} missing: emit it with PLOW_TTS_CODEC_DIR (docs/runtime/tts.md)", path.display()));
        }
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
                    bound.capacities.iter().map(|c| c.1).max().unwrap_or(0),
                    bound.frame_codes,
                    bound.frame_samples,
                    bound.window,
                    bound.lookahead,
                );
                let _ = ready_tx.send(Ok(info));
                run(rx, bound);
            })
            .map_err(|e| e.to_string())?;
        let (max_frames, frame_codes, frame_samples, window, lookahead) = ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Codec { tx: Mutex::new(tx), max_frames, frame_codes, frame_samples, window, lookahead })
    }

    /// `frames * frame_codes` codebook ids -> `frames * frame_samples` samples.
    pub async fn decode(&self, codes: Vec<i32>, frames: usize, seed: u64) -> Result<Vec<f32>, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.tx
            .lock()
            .send(Job { codes, frames, seed, reply })
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
    let mut capacities = Vec::new();
    for (role, program) in pipeline.programs() {
        let dims = role
            .strip_prefix("decode.b")
            .and_then(|r| r.split_once(".f"))
            .and_then(|(b, f)| Some((b.parse::<usize>().ok()?, f.parse::<usize>().ok()?)));
        if let Some((b, f)) = dims {
            capacities.push((b, f, program));
        }
    }
    if capacities.is_empty() {
        return Err("codec packet declares no decode capacity".into());
    }
    capacities.sort_by_key(|&(b, f, _)| (b * f, f));
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
        lengths,
        capacities,
        runtime,
    })
}

impl Bound {
    /// The smallest capacity holding `batch` items of `frames` frames.
    fn capacity(&self, batch: usize, frames: usize) -> Option<(usize, usize, usize)> {
        self.capacities.iter().copied().find(|&(b, f, _)| b >= batch && f >= frames)
    }

    fn decode(&mut self, jobs: &[Job]) -> Result<Vec<Vec<f32>>, String> {
        let frames = jobs.iter().map(|j| j.frames).max().unwrap_or(0);
        let (cb, cf, program) = self
            .capacity(jobs.len(), frames)
            .ok_or_else(|| format!("{} x {frames} frames exceeds every codec capacity", jobs.len()))?;
        let fc = self.frame_codes;
        let mut codes = vec![0u32; self.codes.bytes / 4];
        for (i, j) in jobs.iter().enumerate() {
            for (k, &c) in j.codes.iter().enumerate() {
                codes[i * cf * fc + k] = c.clamp(0, i32::MAX) as u32;
            }
        }
        let e = |x: crate::RuntimeError| x.to_string();
        self.runtime.write_tensor(self.codes, bytemuck::cast_slice(&codes)).map_err(e)?;
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
        self.runtime.run(program).map_err(e)?;
        let mut pcm = vec![0f32; self.pcm.bytes / 4];
        self.runtime.read_tensor(self.pcm, bytemuck::cast_slice_mut(&mut pcm)).map_err(e)?;
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
        // Longest first so each launch's capacity is set by its first job.
        pending.sort_by_key(|j| std::cmp::Reverse(j.frames));
        while !pending.is_empty() {
            let frames = pending[0].frames;
            let mut n = 1;
            while n < pending.len() && n < max_batch && codec.capacity(n + 1, frames).is_some() {
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
