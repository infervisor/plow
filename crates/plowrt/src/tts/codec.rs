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

/// A window of a stream on a cached capacity (`csynth`): the source carries the stream's phase,
/// which is `phase` (per harmonic) at sample `seam` of the window; the reply's phase is the
/// source's at sample `next_seam`.
#[derive(Clone, Debug, Default)]
pub struct Window {
    pub seam: u32,
    pub next_seam: u32,
    pub phase: Vec<f32>,
}

/// Called by the codec worker between the program segments of one launch (see
/// [`Codec::set_yield`]).
pub type YieldHook = std::sync::Arc<dyn Fn() + Send + Sync>;

use crate::serve::cosched::Band;

/// A render due `mine` (its launch's own `Due`), holding the device, yields between CFM steps
/// when the best waiter outranks it ([`crate::serve::cosched::Due::outranks`] with a `margin_ns`
/// slack margin) and is a first output or about to miss: never to decode, as the render thread is
/// serial and every stream behind this launch waits out the yield.
pub fn should_yield(
    waiter: Option<crate::serve::cosched::Due>,
    mine: &crate::serve::cosched::Due,
    now: std::time::Instant,
    margin_ns: i64,
) -> bool {
    waiter.is_some_and(|w| w.rank(now).0 <= Band::First && w.outranks(mine, now, margin_ns))
}

/// The `Due` a render takes (and takes back) the device by: at least a first output's band. The
/// render thread is serial, so a stream that turns due while it waits (a first audio) waits
/// behind this launch.
pub fn serial_due(mine: crate::serve::cosched::Due) -> crate::serve::cosched::Due {
    crate::serve::cosched::Due { band: mine.band.min(Band::First), ..mine }
}

/// [`should_yield`]'s margin: `PLOW_RENDER_YIELD_MARGIN_MS` (default 5 ms).
pub fn yield_margin_ns() -> i64 {
    (crate::config::RuntimeConfig::get().render_yield_margin_ms as i64).saturating_mul(1_000_000)
}

/// The render's turn and its launch's own `Due` (before [`serial_due`]) while a launch runs;
/// empty between launches.
pub type HeldTurn = std::sync::Arc<Mutex<Option<(crate::serve::cosched::Turn, crate::serve::cosched::Due)>>>;

/// A yield hook handing the render's held turn to a tighter waiter ([`should_yield`]) and taking
/// it back by [`serial_due`] before the next segment.
pub fn turn_yield(dt: std::sync::Arc<crate::serve::cosched::DeviceTurn>, held: HeldTurn) -> YieldHook {
    let margin = yield_margin_ns();
    std::sync::Arc::new(move || {
        let mut held = held.lock();
        let Some((turn, due)) = held.as_mut() else { return };
        let now = std::time::Instant::now();
        if !should_yield(dt.tightest_waiter(), due, now, margin) {
            return;
        }
        turn.release();
        futures::executor::block_on(turn.take_due(&dt, serial_due(*due)));
    })
}

/// A decode's PCM and, for a [`Window`], the source phase at its `next_seam`.
pub struct Decoded {
    pub pcm: Vec<f32>,
    pub phase: Vec<f32>,
}

struct Job {
    codes: Vec<i32>,
    frames: usize,
    seed: u64,
    voice: u32,
    urgency: Urgency,
    window: Option<Window>,
    reply: tokio::sync::oneshot::Sender<Result<Decoded, String>>,
    /// Backlog charged to the feeding model's mux until this job is answered.
    _work: Option<crate::sched::admission::DownstreamWork>,
}

pub struct Codec {
    tx: Mutex<mpsc::Sender<Job>>,
    yield_hook: std::sync::Arc<Mutex<Option<YieldHook>>>,
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
    /// Largest window a cached capacity holds (0: the packet has none).
    pub max_window: usize,
    /// The cached capacities as (batch, frames).
    pub window_capacities: Vec<(usize, usize)>,
    /// Optional packet parameters (e.g. a streaming schedule).
    pub parameters: std::collections::BTreeMap<String, u64>,
}

type Capacity = (usize, usize, Vec<usize>);

struct Bound {
    runtime: Box<dyn crate::exec::packet_runtime::PacketRuntime>,
    /// (batch, frames, programs in order), ascending by batch * frames.
    capacities: Vec<Capacity>,
    /// The same for windows ([`Window`]) on the cached capacities.
    cached: Vec<Capacity>,
    /// Window source phase in/out and seams (packets with cached capacities).
    phase: Option<[PacketTensor; 4]>,
    harmonics: usize,
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
    yield_hook: std::sync::Arc<Mutex<Option<YieldHook>>>,
    /// Programs per segment between yield points (0: one segment).
    yield_programs: usize,
}

impl Codec {
    pub fn load(assets: &Path) -> Result<Self, String> {
        let path = crate::exec::packet_runtime::stage_packet(&assets.join("model.pkt"), "codec.packet", PACKET)
            .map_err(|e| e.to_string())?;
        if !path.is_file() {
            return Err(format!("{} missing: emit it with PLOW_TTS_CODEC_DIR (docs/runtime/tts.md)", path.display()));
        }
        Self::load_packet(&path)
    }

    pub fn load_packet(path: &Path) -> Result<Self, String> {
        let path = path.to_path_buf();
        let (tx, rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel();
        let yield_hook = std::sync::Arc::new(Mutex::new(None));
        let hook = yield_hook.clone();
        std::thread::Builder::new()
            .name("plow-tts-codec".into())
            .spawn(move || {
                let mut bound = match bind(&path) {
                    Ok(b) => b,
                    Err(e) => return drop(ready_tx.send(Err(e))),
                };
                bound.yield_hook = hook;
                let max_window = bound.cached.iter().map(|c| c.1).max().unwrap_or(0);
                let info = (
                    bound.capacities.iter().chain(&bound.cached).map(|c| c.0).max().unwrap_or(1),
                    bound.capacities.iter().map(|c| c.1).max().unwrap_or(0),
                    bound.capacities.iter().map(|c| c.1).min().unwrap_or(0),
                    bound.frame_codes,
                    bound.frame_samples,
                    bound.window,
                    bound.lookahead,
                    bound.voices.clone(),
                    bound.parameters.clone(),
                    max_window,
                    bound.cached.iter().map(|c| (c.0, c.1)).collect::<Vec<_>>(),
                );
                let _ = ready_tx.send(Ok(info));
                run(rx, bound);
            })
            .map_err(|e| e.to_string())?;
        let (max_batch, max_frames, min_frames, frame_codes, frame_samples, window, lookahead, voices, parameters, max_window, window_capacities) =
            ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Codec {
            tx: Mutex::new(tx),
            yield_hook,
            credit: None,
            max_batch,
            max_frames,
            min_frames,
            frame_codes,
            frame_samples,
            window,
            lookahead,
            voices,
            max_window,
            window_capacities,
            parameters,
        })
    }

    /// Split each launch into segments of `render.yield_programs` programs (a CFM step) and call
    /// `hook` between them, so a co-scheduler can hand the device to another model mid-render.
    /// `None` restores one launch per decode.
    pub fn set_yield(&self, hook: Option<YieldHook>) {
        *self.yield_hook.lock() = hook;
    }

    /// Charge this stage's backlog to the feeding model's admission: past two full launches of
    /// pending work, the model seats no new request.
    pub fn couple(&mut self, credit: std::sync::Arc<crate::sched::admission::DownstreamCredit>) {
        credit.set_limit(2 * self.max_batch);
        self.credit = Some(credit);
    }

    /// `frames * frame_codes` codebook ids -> `frames * frame_samples` samples.
    pub async fn decode(&self, codes: Vec<i32>, frames: usize, seed: u64, urgency: Urgency) -> Result<Vec<f32>, String> {
        self.decode_as(codes, frames, seed, 0, urgency, None).await.map(|d| d.pcm)
    }

    /// [`Self::decode`] for the packet's voice `voice` (an index into [`Self::voices`]).
    pub async fn decode_voice(&self, codes: Vec<i32>, frames: usize, seed: u64, voice: u32) -> Result<Vec<f32>, String> {
        self.decode_as(codes, frames, seed, voice, Urgency::Stream, None).await.map(|d| d.pcm)
    }

    /// A stream window on the cached capacities (`max_window > 0`).
    pub async fn decode_window(&self, codes: Vec<i32>, frames: usize, seed: u64, voice: u32, window: Window) -> Result<Decoded, String> {
        self.decode_as(codes, frames, seed, voice, Urgency::Stream, Some(window)).await
    }

    async fn decode_as(
        &self,
        codes: Vec<i32>,
        frames: usize,
        seed: u64,
        voice: u32,
        urgency: Urgency,
        window: Option<Window>,
    ) -> Result<Decoded, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let _work = self.credit.as_ref().map(|c| c.work());
        self.tx
            .lock()
            .send(Job { codes, frames, seed, voice, urgency, window, reply, _work })
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
    let mut cached: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize)>> = Default::default();
    let mut prefill: std::collections::BTreeMap<u32, Vec<(usize, usize)>> = Default::default();
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
            ["synth", b, t, stage] | ["csynth", b, t, stage] => {
                if let (Some(key), Ok(stage)) = (dims(b, t, "b", "t"), stage.parse::<usize>()) {
                    let family = if parts[0] == "synth" { &mut sequences } else { &mut cached };
                    family.entry(key).or_default().push((stage, program));
                }
            }
            ["prefill", v, stage] => {
                if let (Some(v), Ok(stage)) = (v.strip_prefix('v').and_then(|v| v.parse::<u32>().ok()), stage.parse::<usize>()) {
                    prefill.entry(v).or_default().push((stage, program));
                }
            }
            _ => {}
        }
    }
    let ordered = |mut stages: Vec<(usize, usize)>| -> Vec<usize> {
        stages.sort();
        stages.into_iter().map(|(_, p)| p).collect()
    };
    let family = |seqs: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize)>>| -> Vec<Capacity> {
        let mut caps: Vec<Capacity> = seqs.into_iter().map(|((b, f), stages)| (b, f, ordered(stages))).collect();
        caps.sort_by_key(|&(b, f, _)| (b * f, f));
        caps
    };
    let capacities = family(sequences);
    let cached = family(cached);
    if capacities.is_empty() {
        return Err("codec packet declares no decode capacity".into());
    }
    // Run every capacity once now: each program sequence is captured as a CUDA graph on first
    // use, and a capture that overlaps another thread's context synchronize (the LM engine) fails.
    let t = std::time::Instant::now();
    let mut runtime = runtime;
    for (_, _, programs) in capacities.iter().chain(&cached) {
        runtime.run_sequence(programs).map_err(e)?;
    }
    tracing::info!(capacities = capacities.len() + cached.len(), ms = t.elapsed().as_millis() as u64, "codec graphs warmed");
    let lengths = (0..param("lengths.count")?)
        .map(|k| Ok((pipeline.tensor(&format!("lengths.{k}")).map_err(e)?, param(&format!("lengths.{k}.rows_per_frame"))?)))
        .collect::<Result<Vec<_>, String>>()?;
    let seed = pipeline.tensor("seed").map_err(e)?;
    let voice = pipeline.tensor("voice").ok();
    // The cached capacities read each voice's prompt K/V: fill them once, from the prompt alone.
    if !cached.is_empty() {
        let (Some(voice), Some(&(count, _))) = (voice, lengths.first()) else {
            return Err("cached codec capacities need voice and length inputs".into());
        };
        for (v, stages) in prefill {
            let zero = |t: PacketTensor| vec![0u8; t.bytes];
            let mut vb = zero(voice);
            vb[..4].copy_from_slice(&v.to_le_bytes());
            runtime.write_tensor(voice, &vb).map_err(e)?;
            runtime.write_tensor(count, &zero(count)).map_err(e)?;
            let mut sb = zero(seed);
            sb[..8].copy_from_slice(&0x5eed_u64.to_le_bytes());
            runtime.write_tensor(seed, &sb).map_err(e)?;
            runtime.run_sequence(&ordered(stages)).map_err(e)?;
        }
    }
    let phase = ["phase", "phase_out", "seam", "next_seam"].map(|k| pipeline.tensor(k).ok());
    let phase = phase.iter().all(Option::is_some).then(|| phase.map(Option::unwrap));
    if !cached.is_empty() && phase.is_none() {
        return Err("cached codec capacities without the window phase tensors".into());
    }
    Ok(Bound {
        cached,
        phase,
        harmonics: pipeline.parameter("vocoder.harmonics").map(|v| v as usize).unwrap_or(0),
        codes: pipeline.tensor("codes").map_err(e)?,
        seed,
        pcm: pipeline.tensor("pcm").map_err(e)?,
        frame_codes: param("codec.frame_codes")?,
        frame_samples: param("codec.frame_samples")?,
        window: param("stream.window_frames")?,
        lookahead: param("stream.lookahead_frames")?,
        voice,
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
        yield_hook: Default::default(),
        yield_programs: pipeline.parameter("render.yield_programs").map(|v| v as usize).unwrap_or(0),
    })
}

impl Bound {
    /// The smallest capacity (of the window family when `window`) holding `batch` items of
    /// `frames` frames.
    fn capacity(&self, window: bool, batch: usize, frames: usize) -> Option<&Capacity> {
        let family = if window { &self.cached } else { &self.capacities };
        family.iter().find(|&&(b, f, _)| b >= batch && f >= frames)
    }

    fn decode(&mut self, jobs: &[Job]) -> Result<Vec<Decoded>, String> {
        let frames = jobs.iter().map(|j| j.frames).max().unwrap_or(0);
        let window = jobs[0].window.is_some();
        let (cb, cf, programs) = self
            .capacity(window, jobs.len(), frames)
            .cloned()
            .ok_or_else(|| format!("{} x {frames} frames exceeds every codec capacity", jobs.len()))?;
        let fc = self.frame_codes;
        let t0 = std::time::Instant::now();
        // Only the capacity's region of the (widest-capacity) input and output tensors moves: the
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
        let nh = self.harmonics;
        let windows = if window { self.phase } else { None };
        if let Some([phase, _, seam, next_seam]) = windows {
            let mut p = vec![0f32; phase.bytes / 4];
            let (mut s, mut ns) = (vec![0u32; seam.bytes / 4], vec![0u32; next_seam.bytes / 4]);
            for (i, w) in jobs.iter().filter_map(|j| j.window.as_ref()).enumerate() {
                p[i * nh..][..w.phase.len().min(nh)].copy_from_slice(&w.phase[..w.phase.len().min(nh)]);
                (s[i], ns[i]) = (w.seam, w.next_seam);
            }
            self.runtime.write_tensor(phase, bytemuck::cast_slice(&p)).map_err(e)?;
            self.runtime.write_tensor(seam, bytemuck::cast_slice(&s)).map_err(e)?;
            self.runtime.write_tensor(next_seam, bytemuck::cast_slice(&ns)).map_err(e)?;
        }
        let t1 = std::time::Instant::now();
        let hook = self.yield_hook.lock().clone();
        let mut gpu_us = 0.0;
        match hook.filter(|_| self.yield_programs > 0) {
            Some(hook) => {
                for (k, segment) in programs.chunks(self.yield_programs).enumerate() {
                    if k > 0 {
                        hook();
                    }
                    self.runtime.run_sequence(segment).map_err(e)?;
                    gpu_us += self.runtime.last_run_us();
                }
            }
            None => {
                self.runtime.run_sequence(&programs).map_err(e)?;
                gpu_us = self.runtime.last_run_us();
            }
        }
        let t2 = std::time::Instant::now();
        let mut pcm = vec![0f32; cb * cf * self.frame_samples];
        self.runtime.read_tensor_at(self.pcm, 0, bytemuck::cast_slice_mut(&mut pcm)).map_err(e)?;
        let mut phase_out = Vec::new();
        if let Some([_, out, _, _]) = windows {
            phase_out = vec![0f32; out.bytes / 4];
            self.runtime.read_tensor(out, bytemuck::cast_slice_mut(&mut phase_out)).map_err(e)?;
        }
        tracing::debug!(
            target: "plowrt::tts::codec_launch",
            jobs = jobs.len(),
            window,
            frames,
            cb,
            cf,
            h2d_us = (t1 - t0).as_micros() as u64,
            run_us = (t2 - t1).as_micros() as u64,
            gpu_us = gpu_us as u64,
            d2h_us = t2.elapsed().as_micros() as u64,
            "codec launch"
        );
        let per = cf * self.frame_samples;
        debug_assert!(cb * per <= pcm.len());
        Ok(jobs
            .iter()
            .enumerate()
            .map(|(i, j)| Decoded {
                pcm: pcm[i * per..i * per + j.frames * self.frame_samples].to_vec(),
                phase: if window { phase_out[i * nh..(i + 1) * nh].to_vec() } else { Vec::new() },
            })
            .collect())
    }
}

fn run(rx: mpsc::Receiver<Job>, mut codec: Bound) {
    let max_batch = codec.capacities.iter().chain(&codec.cached).map(|c| c.0).max().unwrap_or(1);
    let mut pending: Vec<Job> = Vec::new();
    while let Ok(first) = rx.recv() {
        pending.push(first);
        pending.extend(rx.try_iter());
        // Most urgent first; within an urgency longest first, so a launch's capacity is mostly
        // set by its first job.
        pending.sort_by_key(|j| (j.urgency, std::cmp::Reverse(j.frames)));
        while !pending.is_empty() {
            // One family per launch: the first job's, then the next jobs of that family that fit.
            let window = pending[0].window.is_some();
            let mut frames = pending[0].frames;
            let mut take = vec![0];
            for (k, j) in pending.iter().enumerate().skip(1) {
                if take.len() >= max_batch {
                    break;
                }
                if j.window.is_some() != window {
                    continue;
                }
                if codec.capacity(window, take.len() + 1, frames.max(j.frames)).is_none() {
                    break;
                }
                frames = frames.max(j.frames);
                take.push(k);
            }
            let mut batch: Vec<Job> = take.iter().rev().map(|&k| pending.remove(k)).collect();
            batch.reverse();
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

#[cfg(test)]
mod tests {
    use super::should_yield;
    use crate::serve::cosched::{Band, Due};
    use std::time::{Duration, Instant};

    #[test]
    fn render_yields_only_to_an_outranking_first_output_or_near_miss() {
        let now = Instant::now();
        let at = |ms: i64, band| Due {
            deadline: if ms >= 0 { now + Duration::from_millis(ms as u64) } else { now - Duration::from_millis(-ms as u64) },
            cost: Duration::ZERO,
            band,
        };
        let m = 5_000_000;
        let stream = at(100, Band::Stream);
        assert!(!should_yield(None, &stream, now, m));
        assert!(should_yield(Some(at(20, Band::Stream)), &stream, now, m), "about to miss");
        assert!(!should_yield(Some(at(40, Band::Stream)), &stream, now, m), "decode, not urgent");
        assert!(!should_yield(Some(at(200, Band::Stream)), &stream, now, m));
        assert!(should_yield(Some(at(-50, Band::Stream)), &at(-10, Band::Stream), now, m), "both late: the later one first");
        assert!(!should_yield(Some(at(-5, Band::Stream)), &at(-10, Band::Stream), now, m), "within the margin");
        // A first output takes the device from window renders, not from an urgent one or a
        // tighter first audio.
        assert!(should_yield(Some(at(700, Band::First)), &stream, now, m));
        assert!(!should_yield(Some(at(700, Band::First)), &at(20, Band::Stream), now, m));
        assert!(!should_yield(Some(at(700, Band::First)), &at(400, Band::First), now, m));
        assert!(!should_yield(Some(at(10, Band::Stream)), &at(5, Band::Stream), now, m));
        assert!(!should_yield(Some(at(10, Band::Bulk)), &stream, now, m));
        // A window render takes the device as a first output; a first output stays one.
        assert_eq!(super::serial_due(stream).band, Band::First);
        assert_eq!(super::serial_due(at(400, Band::First)).band, Band::First);
        // An ASR final takes the device from any render that is not urgent; late, it competes with
        // an urgent render by slack.
        assert!(should_yield(Some(at(480, Band::Final)), &at(300, Band::First), now, m));
        assert!(!should_yield(Some(at(480, Band::Final)), &at(20, Band::Stream), now, m));
        assert!(!should_yield(Some(at(-10, Band::Final)), &at(-300, Band::Stream), now, m));
        assert!(should_yield(Some(at(-300, Band::Final)), &at(-10, Band::Stream), now, m));
    }
}
