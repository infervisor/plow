//! Voice activity detection on a `vad.v1` packet: per launch, one 16 kHz frame of each stream
//! (with its left context) and its recurrent state in, the frame's speech probability and the next
//! state out; one program per batch capacity (`step.b{B}`). One worker thread owns the packet
//! runtime and runs the pending frames of every stream together, a frame per stream per launch, so
//! streams advance in lockstep and a stream's probabilities never depend on its batch.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, OnceLock};

use parking_lot::Mutex;

use crate::exec::packet_runtime::{load_packet_runtime_on, PacketAsset, PacketRuntime, PacketTensor};

pub const DRIVER: &str = "vad.v1";

struct Job {
    /// `frames` windows of `context + frame` samples.
    windows: Vec<f32>,
    frames: usize,
    state: Vec<Vec<f32>>,
    reply: tokio::sync::oneshot::Sender<Result<(Vec<f32>, Vec<Vec<f32>>), String>>,
}

pub struct Vad {
    tx: Mutex<mpsc::Sender<Job>>,
    pub frame_samples: usize,
    pub context_samples: usize,
    pub sample_rate: u32,
    /// Floats per stream of each recurrent state tensor.
    state_widths: Vec<usize>,
    pub max_batch: usize,
    pub backend: &'static str,
    pub stats: Arc<VadStats>,
}

/// Worker counters: launches, frames scored, device (or engine) time of the launches.
#[derive(Default)]
pub struct VadStats {
    pub launches: AtomicU64,
    pub frames: AtomicU64,
    pub device_ns: AtomicU64,
}

struct Bound {
    stats: Arc<VadStats>,
    runtime: Box<dyn PacketRuntime>,
    /// (batch, program), ascending.
    capacities: Vec<(usize, usize)>,
    audio: PacketTensor,
    prob: PacketTensor,
    state: Vec<(PacketTensor, PacketTensor)>,
    window: usize,
    widths: Vec<usize>,
}

impl Vad {
    /// Bind the `vad.v1` pipeline of `path` on `backend` (`cuda`, `cpu`, `auto`).
    pub fn load(path: &Path, backend: &str, device: u8) -> Result<Arc<Self>, String> {
        let (path, backend_name) = (path.to_path_buf(), backend.to_owned());
        let (tx, rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel();
        let stats = Arc::new(VadStats::default());
        let worker_stats = stats.clone();
        std::thread::Builder::new()
            .name("plow-vad".into())
            .spawn(move || {
                let (mut bound, geometry, backend) = match bind(&path, &backend_name, device) {
                    Ok(b) => b,
                    Err(e) => return drop(ready_tx.send(Err(e))),
                };
                bound.stats = worker_stats;
                let _ = ready_tx.send(Ok((geometry, bound.widths.clone(), bound.capacities.last().map_or(0, |c| c.0), backend)));
                run(rx, bound);
            })
            .map_err(|e| e.to_string())?;
        let ((frame_samples, context_samples, sample_rate), state_widths, max_batch, backend) =
            ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Arc::new(Self { tx: Mutex::new(tx), frame_samples, context_samples, sample_rate, state_widths, max_batch, backend, stats }))
    }

    /// A stream with zero state and context.
    pub fn stream(self: &Arc<Self>) -> VadStream {
        VadStream {
            vad: Arc::clone(self),
            context: vec![0.0; self.context_samples],
            pending: Vec::new(),
            state: self.state_widths.iter().map(|&w| vec![0.0; w]).collect(),
        }
    }
}

fn bind(path: &Path, backend: &str, device: u8) -> Result<(Bound, (usize, usize, u32), &'static str), String> {
    let e = |x: crate::RuntimeError| x.to_string();
    let loaded = load_packet_runtime_on(path, backend, device).map_err(e)?;
    let mut runtime = loaded.runtime;
    let asset = PacketAsset::load(path).map_err(e)?;
    let pipeline = asset.bind_driver(DRIVER, runtime.as_ref()).map_err(e)?;
    let param = |k: &str| pipeline.parameter(k).map(|v| v as usize).map_err(e);
    let (frame, context) = (param("vad.frame_samples")?, param("vad.context_samples")?);
    let rate = param("audio.sample_rate")? as u32;
    let mut capacities: Vec<(usize, usize)> = pipeline
        .programs()
        .filter_map(|(role, program)| Some((role.strip_prefix("step.b")?.parse().ok()?, program)))
        .collect();
    capacities.sort();
    let bmax = capacities.last().map(|c| c.0).ok_or("vad packet declares no step capacity")?;
    let audio = pipeline.tensor("audio").map_err(e)?;
    let prob = pipeline.tensor("prob").map_err(e)?;
    let state = (0..param("state.count")?)
        .map(|k| Ok((pipeline.tensor(&format!("state.{k}")).map_err(e)?, pipeline.tensor(&format!("state_out.{k}")).map_err(e)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let window = frame + context;
    if audio.bytes != bmax * window * 4 || prob.bytes != bmax * 4 || state.iter().any(|(s, o)| s.bytes != o.bytes || s.bytes % (bmax * 4) != 0) {
        return Err("vad packet tensors do not match its capacities".into());
    }
    let widths = state.iter().map(|(s, _)| s.bytes / bmax / 4).collect();
    // Each capacity once now: a CUDA program sequence is captured as a graph on first use, which
    // must not overlap another thread's context synchronize.
    for &(_, program) in &capacities {
        runtime.run_sequence(&[program]).map_err(e)?;
    }
    let stats = Default::default();
    Ok((Bound { stats, runtime, capacities, audio, prob, state, window, widths }, (frame, context, rate), loaded.backend))
}

struct Active {
    job: Job,
    next: usize,
    probs: Vec<f32>,
}

fn run(rx: mpsc::Receiver<Job>, mut bound: Bound) {
    let mut active: std::collections::VecDeque<Active> = Default::default();
    loop {
        if active.is_empty() {
            match rx.recv() {
                Ok(job) => active.push_back(Active { probs: Vec::with_capacity(job.frames), job, next: 0 }),
                Err(_) => return,
            }
        }
        while let Ok(job) = rx.try_recv() {
            active.push_back(Active { probs: Vec::with_capacity(job.frames), job, next: 0 });
        }
        let n = active.len().min(bound.capacities.last().map_or(1, |c| c.0));
        if let Err(error) = bound.step(&mut active.make_contiguous()[..n]) {
            for a in active.drain(..n) {
                let _ = a.job.reply.send(Err(error.clone()));
            }
            continue;
        }
        // Finished streams answer; the rest rotate behind the waiting ones.
        for _ in 0..n {
            let a = active.pop_front().expect("stepped");
            if a.next == a.job.frames {
                let Active { job, probs, .. } = a;
                let _ = job.reply.send(Ok((probs, job.state)));
            } else {
                active.push_back(a);
            }
        }
    }
}

impl Bound {
    /// One launch: the next frame of each of `step`.
    fn step(&mut self, step: &mut [Active]) -> Result<(), String> {
        let n = step.len();
        let &(_, program) = self.capacities.iter().find(|c| c.0 >= n).ok_or("no vad capacity")?;
        let e = |x: crate::RuntimeError| x.to_string();
        let w = self.window;
        let mut audio = Vec::with_capacity(n * w * 4);
        for a in step.iter() {
            audio.extend(a.job.windows[a.next * w..(a.next + 1) * w].iter().flat_map(|x| x.to_le_bytes()));
        }
        self.runtime.write_tensor_at(self.audio, 0, &audio).map_err(e)?;
        for (k, &(tensor, _)) in self.state.iter().enumerate() {
            let bytes: Vec<u8> = step.iter().flat_map(|a| a.job.state[k].iter().flat_map(|x| x.to_le_bytes())).collect();
            self.runtime.write_tensor_at(tensor, 0, &bytes).map_err(e)?;
        }
        self.runtime.run_sequence(&[program]).map_err(e)?;
        self.stats.launches.fetch_add(1, Ordering::Relaxed);
        self.stats.frames.fetch_add(n as u64, Ordering::Relaxed);
        self.stats.device_ns.fetch_add((self.runtime.last_run_us() * 1e3) as u64, Ordering::Relaxed);
        let mut prob = vec![0u8; n * 4];
        self.runtime.read_tensor_at(self.prob, 0, &mut prob).map_err(e)?;
        let mut states = Vec::with_capacity(self.state.len());
        for (k, &(_, out)) in self.state.iter().enumerate() {
            let mut bytes = vec![0u8; n * self.widths[k] * 4];
            self.runtime.read_tensor_at(out, 0, &mut bytes).map_err(e)?;
            states.push(bytes);
        }
        let f32s = |b: &[u8]| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();
        let probs = f32s(&prob);
        let states: Vec<Vec<f32>> = states.iter().map(|b| f32s(b)).collect();
        for (i, a) in step.iter_mut().enumerate() {
            a.probs.push(probs[i]);
            for (k, s) in states.iter().enumerate() {
                let width = self.widths[k];
                a.job.state[k].copy_from_slice(&s[i * width..(i + 1) * width]);
            }
            a.next += 1;
        }
        Ok(())
    }
}

/// One stream's VAD state: its recurrent state, the last `context` samples and a partial frame.
pub struct VadStream {
    vad: Arc<Vad>,
    context: Vec<f32>,
    pending: Vec<f32>,
    state: Vec<Vec<f32>>,
}

impl VadStream {
    pub fn frame_samples(&self) -> usize {
        self.vad.frame_samples
    }

    /// Append 16 kHz samples; returns the whole frames they complete (their samples, in order) and
    /// one speech probability per frame. A partial frame waits for the next call.
    pub async fn push(&mut self, samples: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        self.pending.extend_from_slice(samples);
        let (frame, context) = (self.vad.frame_samples, self.vad.context_samples);
        let frames = self.pending.len() / frame;
        if frames == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut windows = Vec::with_capacity(frames * (frame + context));
        for k in 0..frames {
            windows.extend_from_slice(&self.context);
            let f = &self.pending[k * frame..(k + 1) * frame];
            windows.extend_from_slice(f);
            self.context.copy_from_slice(&f[frame - context..]);
        }
        let (reply, rx) = tokio::sync::oneshot::channel();
        let state = std::mem::take(&mut self.state);
        self.vad
            .tx
            .lock()
            .send(Job { windows, frames, state, reply })
            .map_err(|_| "vad worker stopped".to_string())?;
        let (probs, state) = rx.await.map_err(|_| "vad worker dropped the job".to_string())??;
        self.state = state;
        Ok((self.pending.drain(..frames * frame).collect(), probs))
    }

    /// The partial frame not yet scored (end of a turn or stream).
    pub fn take_rest(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.pending)
    }
}

/// The process's VAD (`--vad-packet`), loaded once at startup by [`init`].
static SHARED: OnceLock<Option<Arc<Vad>>> = OnceLock::new();

/// Load `--vad-packet` (if set) on CUDA device 0, or the named backend. Idempotent.
pub fn init() -> Result<(), String> {
    if SHARED.get().is_some() {
        return Ok(());
    }
    let cfg = crate::config::RuntimeConfig::get();
    let vad = match cfg.vad_packet.as_deref() {
        None => None,
        Some(spec) => {
            let (path, backend) = match spec.split_once(",backend=") {
                Some((p, b)) => (p, b),
                None => (spec, "cuda"),
            };
            let vad = Vad::load(Path::new(path), backend, 0).map_err(|e| format!("--vad-packet {path}: {e}"))?;
            tracing::info!(packet = path, backend = vad.backend, frame = vad.frame_samples, max_batch = vad.max_batch, "vad loaded");
            Some(vad)
        }
    };
    let _ = SHARED.set(vad);
    Ok(())
}

/// The process's VAD, if one was loaded.
pub fn shared() -> Option<Arc<Vad>> {
    SHARED.get().cloned().flatten()
}
