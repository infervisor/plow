//! The codec stage of a speech pipeline: a `codec.v1` packet (codes or speech tokens in, PCM out,
//! one program or program sequence per (batch, frames) capacity: roles `decode.b{B}.f{F}` or
//! `synth.b{B}.t{T}.{stage}`) on the packet runtime, driven by one worker thread. Pending jobs share
//! each launch the way concurrent streams share LM decode steps; per-item valid lengths, seeds and
//! voice indices make every item decode exactly as it would alone.

use std::path::Path;
use std::sync::mpsc;

use parking_lot::Mutex;

use crate::exec::packet_runtime::{load_packet_runtime_on, PacketAsset, PacketTensor};

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

/// A cloned voice's conditionals for the voice-slot capacities (`vsynth`): its prompt speech
/// tokens, their `[2 * tokens][80]` prompt mels and the `[80]` speaker vector.
#[derive(Debug)]
pub struct SlotVoice {
    pub prompt_token: Vec<u32>,
    pub prompt_feat: Vec<f32>,
    pub spks: Vec<f32>,
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
    crate::serve::cosched::Due {
        band: mine.band.min(Band::First),
        ..mine
    }
}

/// [`should_yield`]'s margin: `PLOW_RENDER_YIELD_MARGIN_MS` (default 5 ms).
pub fn yield_margin_ns() -> i64 {
    (crate::config::RuntimeConfig::get().render_yield_margin_ms as i64).saturating_mul(1_000_000)
}

/// The render's turn and its launch's own `Due` (before [`serial_due`]) while a launch runs;
/// empty between launches.
pub type HeldTurn =
    std::sync::Arc<Mutex<Option<(crate::serve::cosched::Turn, crate::serve::cosched::Due)>>>;

/// A yield hook handing the render's held turn to a tighter waiter ([`should_yield`]) and taking
/// it back by [`serial_due`] before the next segment.
pub fn turn_yield(
    dt: std::sync::Arc<crate::serve::cosched::DeviceTurn>,
    held: HeldTurn,
) -> YieldHook {
    let margin = yield_margin_ns();
    std::sync::Arc::new(move || {
        let mut held = held.lock();
        let Some((turn, due)) = held.as_mut() else {
            return;
        };
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

/// A stream's codec state slot ([`Codec::acquire_slot`]); `gen` tells its reuses apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotId {
    index: u32,
    gen: u32,
}

#[derive(Clone, Copy, Default)]
struct Slot {
    gen: u32,
    busy: bool,
    /// A launch holding it failed: its state is lost until released.
    failed: bool,
    /// Frames committed (decoded) on it.
    pos: usize,
}

/// The stateful step programs' stream slots, shared by the streams (acquire) and the codec worker
/// (check, commit, release).
pub struct SlotPool {
    slots: Vec<Slot>,
    free: Vec<u32>,
    warned: bool,
}

impl SlotPool {
    fn new(n: usize) -> Self {
        SlotPool {
            slots: vec![Slot::default(); n],
            free: (0..n as u32).rev().collect(),
            warned: false,
        }
    }

    fn acquire(&mut self) -> Option<SlotId> {
        let index = self.free.pop()?;
        let s = &mut self.slots[index as usize];
        (s.busy, s.failed, s.pos) = (true, false, 0);
        Some(SlotId { index, gen: s.gen })
    }

    /// Frees `id` (a stale or repeated release is ignored); the worker calls it only once no
    /// launch holds the slot.
    fn release(&mut self, id: SlotId) {
        let s = &mut self.slots[id.index as usize];
        if s.busy && s.gen == id.gen {
            (s.busy, s.gen) = (false, s.gen.wrapping_add(1));
            self.free.push(id.index);
        }
    }

    /// A step of `id` from frame `pos`: its committed position, or 0 to restart.
    fn check(&self, id: SlotId, pos: usize) -> Result<(), &'static str> {
        match self.slots.get(id.index as usize) {
            Some(s) if s.busy && s.gen == id.gen => {
                if s.failed {
                    Err("codec stream slot failed")
                } else if pos != 0 && pos != s.pos {
                    Err("codec stream step out of order")
                } else {
                    Ok(())
                }
            }
            _ => Err("stale codec stream slot"),
        }
    }

    fn commit(&mut self, id: SlotId, pos: usize) {
        self.slots[id.index as usize].pos = pos;
    }

    fn fail(&mut self, id: SlotId) {
        self.slots[id.index as usize].failed = true;
    }
}

/// A stream's hold on a codec slot; dropping it frees the slot once the worker reaches it, after
/// every launch submitted before.
pub struct StreamSlot {
    id: SlotId,
    pos: usize,
    tx: mpsc::Sender<Msg>,
}

impl StreamSlot {
    /// Decode the next steps as a new stream (state read as zeros).
    pub fn restart(&mut self) {
        self.pos = 0;
    }
}

impl Drop for StreamSlot {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Release(self.id));
    }
}

#[derive(Clone, Copy)]
struct Step {
    slot: SlotId,
    pos: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Decode,
    Window,
    Step,
    Slot,
}

enum Msg {
    Job(Job),
    Release(SlotId),
}

struct Job {
    codes: Vec<i32>,
    frames: usize,
    seed: u64,
    voice: u32,
    urgency: Urgency,
    window: Option<Window>,
    step: Option<Step>,
    clone: Option<std::sync::Arc<SlotVoice>>,
    reply: tokio::sync::oneshot::Sender<Result<Decoded, String>>,
    /// Backlog charged to the feeding model's mux until this job is answered.
    _work: Option<crate::sched::admission::DownstreamWork>,
}

impl Job {
    fn family(&self) -> Family {
        match (&self.window, &self.step, &self.clone) {
            (_, Some(_), _) => Family::Step,
            (Some(_), None, _) => Family::Window,
            (None, None, Some(_)) => Family::Slot,
            (None, None, None) => Family::Decode,
        }
    }
}

pub struct Codec {
    tx: Mutex<mpsc::Sender<Msg>>,
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
    /// Most new frames one stateful step decodes (0: the packet has no step programs).
    pub step_frames: usize,
    /// Most prompt tokens a [`SlotVoice`] may have (0: the packet has no voice slots).
    pub prompt_max: usize,
    slots: Option<std::sync::Arc<Mutex<SlotPool>>>,
}

type Capacity = (usize, usize, Vec<usize>);

/// The step programs' per-launch inputs and host staging buffers.
struct StepIo {
    slot: PacketTensor,
    pos: PacketTensor,
    rope: PacketTensor,
    /// `stream.n.{k}` per time resolution (rows per frame from [`Bound::lengths`]).
    n: Vec<PacketTensor>,
    slots: usize,
    codes_buf: Vec<u32>,
    slot_buf: Vec<u32>,
    pos_buf: Vec<u32>,
    rope_buf: Vec<u32>,
    n_buf: Vec<u32>,
    len_buf: Vec<u32>,
    pcm_buf: Vec<f32>,
}

/// The packet's voice slot tables (`slot.prompt_token`, `slot.prompt_feat`, `slot.spks`,
/// `slot.prompt_len`) and the voice each slot holds; the worker assigns slots per launch.
struct VoiceSlots {
    tables: [PacketTensor; 4],
    prompt_max: usize,
    resident: Vec<Option<std::sync::Arc<SlotVoice>>>,
    used: Vec<u64>,
    clock: u64,
}

impl VoiceSlots {
    /// Slots for `voices` (at most one launch's worth), writing the tables of those not resident.
    fn assign(
        &mut self,
        rt: &mut dyn crate::exec::packet_runtime::PacketRuntime,
        voices: &[&std::sync::Arc<SlotVoice>],
    ) -> Result<Vec<u32>, String> {
        let e = |x: crate::RuntimeError| x.to_string();
        let mut out: Vec<u32> = Vec::with_capacity(voices.len());
        for v in voices {
            self.clock += 1;
            let hit = self
                .resident
                .iter()
                .position(|r| r.as_ref().is_some_and(|r| std::sync::Arc::ptr_eq(r, v)));
            let s = match hit {
                Some(s) => s,
                None => {
                    let s = (0..self.resident.len())
                        .filter(|s| !out.contains(&(*s as u32)))
                        .min_by_key(|&s| self.used[s])
                        .ok_or("more cloned voices in a launch than voice slots")?;
                    let (pm, p) = (self.prompt_max, v.prompt_token.len());
                    let mut tok = vec![0u32; pm];
                    tok[..p].copy_from_slice(&v.prompt_token);
                    let mut feat = vec![0f32; 2 * pm * 80];
                    feat[..v.prompt_feat.len()].copy_from_slice(&v.prompt_feat);
                    // Unmapped until every table is written: a failed write leaves no voice on it.
                    self.resident[s] = None;
                    let [t, f, k, l] = self.tables;
                    rt.write_tensor_at(t, s * pm * 4, bytemuck::cast_slice(&tok))
                        .map_err(e)?;
                    rt.write_tensor_at(f, s * feat.len() * 4, bytemuck::cast_slice(&feat))
                        .map_err(e)?;
                    rt.write_tensor_at(k, s * 80 * 4, bytemuck::cast_slice(&v.spks))
                        .map_err(e)?;
                    rt.write_tensor_at(l, s * 4, &(p as u32).to_le_bytes())
                        .map_err(e)?;
                    self.resident[s] = Some((*v).clone());
                    s
                }
            };
            self.used[s] = self.clock;
            out.push(s as u32);
        }
        Ok(out)
    }
}

struct Bound {
    runtime: Box<dyn crate::exec::packet_runtime::PacketRuntime>,
    /// (batch, frames, programs in order), ascending by batch * frames.
    capacities: Vec<Capacity>,
    /// The same for windows ([`Window`]) on the cached capacities.
    cached: Vec<Capacity>,
    /// Stateful stream steps as (batch, new frames, [program]).
    steps: Vec<Capacity>,
    step_io: Option<StepIo>,
    slots: Option<std::sync::Arc<Mutex<SlotPool>>>,
    /// Voice-slot capacities (`vsynth`) and the slots their items read.
    slot_caps: Vec<Capacity>,
    voice_slots: Option<VoiceSlots>,
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
        Self::load_on(assets, 0)
    }

    /// As [`Self::load`], on CUDA device `device`.
    pub fn load_on(assets: &Path, device: u8) -> Result<Self, String> {
        let path = crate::exec::packet_runtime::stage_packet(
            &assets.join("model.pkt"),
            "codec.packet",
            PACKET,
        )
        .map_err(|e| e.to_string())?;
        if !path.is_file() {
            return Err(format!(
                "{} missing: emit it with PLOW_TTS_CODEC_DIR (docs/runtime/tts.md)",
                path.display()
            ));
        }
        Self::load_packet_on(&path, device)
    }

    pub fn load_packet(path: &Path) -> Result<Self, String> {
        Self::load_packet_on(path, 0)
    }

    pub fn load_packet_on(path: &Path, device: u8) -> Result<Self, String> {
        let path = path.to_path_buf();
        let (tx, rx) = mpsc::channel::<Msg>();
        let (ready_tx, ready_rx) = mpsc::channel();
        let yield_hook = std::sync::Arc::new(Mutex::new(None));
        let hook = yield_hook.clone();
        std::thread::Builder::new()
            .name("plow-tts-codec".into())
            .spawn(move || {
                let mut bound = match bind(&path, device) {
                    Ok(b) => b,
                    Err(e) => return drop(ready_tx.send(Err(e))),
                };
                bound.yield_hook = hook;
                let max_window = bound.cached.iter().map(|c| c.1).max().unwrap_or(0);
                let info = (
                    bound
                        .capacities
                        .iter()
                        .chain(&bound.cached)
                        .chain(&bound.steps)
                        .map(|c| c.0)
                        .max()
                        .unwrap_or(1),
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
                    bound.steps.iter().map(|c| c.1).max().unwrap_or(0),
                    bound.slots.clone(),
                    bound.voice_slots.as_ref().map_or(0, |v| v.prompt_max),
                );
                let _ = ready_tx.send(Ok(info));
                run(rx, bound);
            })
            .map_err(|e| e.to_string())?;
        let (
            max_batch,
            max_frames,
            min_frames,
            frame_codes,
            frame_samples,
            window,
            lookahead,
            voices,
            parameters,
            max_window,
            window_capacities,
            step_frames,
            slots,
            prompt_max,
        ) = ready_rx.recv().map_err(|e| e.to_string())??;
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
            step_frames,
            prompt_max,
            slots,
        })
    }

    /// A slot for a stream's stateful steps; `None` without step programs or with every slot
    /// held (the stream then decodes windows).
    pub fn acquire_slot(&self) -> Option<StreamSlot> {
        let mut pool = self.slots.as_ref()?.lock();
        let Some(id) = pool.acquire() else {
            if !std::mem::replace(&mut pool.warned, true) {
                tracing::warn!(
                    slots = pool.slots.len(),
                    "tts: codec stream slots exhausted, streams fall back to windowed decode"
                );
            }
            return None;
        };
        Some(StreamSlot {
            id,
            pos: 0,
            tx: self.tx.lock().clone(),
        })
    }

    /// The next `frames` frames (at most [`Self::step_frames`]) of `slot`'s stream ->
    /// `frames * frame_samples` samples continuing its audio.
    pub async fn step(
        &self,
        slot: &mut StreamSlot,
        codes: Vec<i32>,
        frames: usize,
        urgency: Urgency,
    ) -> Result<Vec<f32>, String> {
        let step = Step {
            slot: slot.id,
            pos: slot.pos,
        };
        let pcm = self
            .submit(codes, frames, 0, 0, urgency, None, Some(step), None)
            .await?
            .pcm;
        slot.pos += frames;
        Ok(pcm)
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
    pub async fn decode(
        &self,
        codes: Vec<i32>,
        frames: usize,
        seed: u64,
        urgency: Urgency,
    ) -> Result<Vec<f32>, String> {
        self.decode_as(codes, frames, seed, 0, urgency, None)
            .await
            .map(|d| d.pcm)
    }

    /// [`Self::decode`] for the packet's voice `voice` (an index into [`Self::voices`]).
    pub async fn decode_voice(
        &self,
        codes: Vec<i32>,
        frames: usize,
        seed: u64,
        voice: u32,
    ) -> Result<Vec<f32>, String> {
        self.decode_as(codes, frames, seed, voice, Urgency::Stream, None)
            .await
            .map(|d| d.pcm)
    }

    /// [`Self::decode`] in a cloned voice, on the voice-slot capacities (`prompt_max > 0`).
    pub async fn decode_slot(
        &self,
        codes: Vec<i32>,
        frames: usize,
        seed: u64,
        voice: std::sync::Arc<SlotVoice>,
    ) -> Result<Vec<f32>, String> {
        self.submit(
            codes,
            frames,
            seed,
            0,
            Urgency::Stream,
            None,
            None,
            Some(voice),
        )
        .await
        .map(|d| d.pcm)
    }

    /// A stream window on the cached capacities (`max_window > 0`).
    pub async fn decode_window(
        &self,
        codes: Vec<i32>,
        frames: usize,
        seed: u64,
        voice: u32,
        window: Window,
    ) -> Result<Decoded, String> {
        self.decode_as(codes, frames, seed, voice, Urgency::Stream, Some(window))
            .await
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
        self.submit(codes, frames, seed, voice, urgency, window, None, None)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit(
        &self,
        codes: Vec<i32>,
        frames: usize,
        seed: u64,
        voice: u32,
        urgency: Urgency,
        window: Option<Window>,
        step: Option<Step>,
        clone: Option<std::sync::Arc<SlotVoice>>,
    ) -> Result<Decoded, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let _work = self.credit.as_ref().map(|c| c.work());
        self.tx
            .lock()
            .send(Msg::Job(Job {
                codes,
                frames,
                seed,
                voice,
                urgency,
                window,
                step,
                clone,
                reply,
                _work,
            }))
            .map_err(|_| "codec worker stopped".to_string())?;
        rx.await
            .map_err(|_| "codec worker dropped the job".to_string())?
    }
}

fn bind(path: &Path, device: u8) -> Result<Bound, String> {
    let e = |x: crate::RuntimeError| x.to_string();
    let loaded = load_packet_runtime_on(path, "cuda", device).map_err(e)?;
    let runtime = loaded.runtime;
    let asset = PacketAsset::load(path).map_err(e)?;
    let pipeline = asset.bind_driver(DRIVER, runtime.as_ref()).map_err(e)?;
    let param = |k: &str| pipeline.parameter(k).map(|v| v as usize).map_err(e);
    let mut sequences: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize)>> =
        Default::default();
    let mut cached: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize)>> =
        Default::default();
    let mut steps: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize)>> =
        Default::default();
    let mut vsynth: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize)>> =
        Default::default();
    let mut prefill: std::collections::BTreeMap<u32, Vec<(usize, usize)>> = Default::default();
    for (role, program) in pipeline.programs() {
        let parts: Vec<&str> = role.split('.').collect();
        let dims = |b: &str, u: &str, bp: &str, up: &str| {
            Some((
                b.strip_prefix(bp)?.parse::<usize>().ok()?,
                u.strip_prefix(up)?.parse::<usize>().ok()?,
            ))
        };
        match parts.as_slice() {
            ["decode", b, f] => {
                if let Some(key) = dims(b, f, "b", "f") {
                    sequences.entry(key).or_default().push((0, program));
                }
            }
            ["step", b, f] => {
                if let Some(key) = dims(b, f, "b", "f") {
                    steps.entry(key).or_default().push((0, program));
                }
            }
            [kind @ ("synth" | "csynth" | "vsynth"), b, t, stage] => {
                if let (Some(key), Ok(stage)) = (dims(b, t, "b", "t"), stage.parse::<usize>()) {
                    let family = match *kind {
                        "synth" => &mut sequences,
                        "csynth" => &mut cached,
                        _ => &mut vsynth,
                    };
                    family.entry(key).or_default().push((stage, program));
                }
            }
            ["prefill", v, stage] => {
                if let (Some(v), Ok(stage)) = (
                    v.strip_prefix('v').and_then(|v| v.parse::<u32>().ok()),
                    stage.parse::<usize>(),
                ) {
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
    let family =
        |seqs: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize)>>| -> Vec<Capacity> {
            let mut caps: Vec<Capacity> = seqs
                .into_iter()
                .map(|((b, f), stages)| (b, f, ordered(stages)))
                .collect();
            caps.sort_by_key(|&(b, f, _)| (b * f, f));
            caps
        };
    let capacities = family(sequences);
    let cached = family(cached);
    let slot_caps = family(vsynth);
    let stateful = pipeline.parameter("stream.stateful").unwrap_or(0) == 1;
    let steps = if stateful { family(steps) } else { Vec::new() };
    if capacities.is_empty() {
        return Err("codec packet declares no decode capacity".into());
    }
    let mut runtime = runtime;
    let step_io = if steps.is_empty() {
        None
    } else {
        let slots = param("stream.slots")?;
        let slot = pipeline.tensor("stream.slot").map_err(e)?;
        let elems = |t: PacketTensor| vec![0u32; t.bytes / 4];
        let pos = pipeline.tensor("stream.pos").map_err(e)?;
        let rope = pipeline.tensor("stream.rope_pos").map_err(e)?;
        let mut slot_buf = elems(slot);
        if steps.iter().any(|&(b, n, _)| {
            b > slots.min(slot_buf.len()).min(pos.bytes / 4) || b * n > rope.bytes / 4
        }) {
            return Err("codec step capacity exceeds its stream slots or inputs".into());
        }
        // Warm-up launches below take distinct slots (pos 0: state written as zeros).
        for (b, v) in slot_buf.iter_mut().enumerate() {
            *v = b as u32;
        }
        runtime
            .write_tensor(slot, bytemuck::cast_slice(&slot_buf))
            .map_err(e)?;
        let n = (0..param("lengths.count")?)
            .map(|k| pipeline.tensor(&format!("stream.n.{k}")).map_err(e))
            .collect::<Result<Vec<_>, String>>()?;
        Some(StepIo {
            pos_buf: elems(pos),
            rope_buf: elems(rope),
            n_buf: elems(slot),
            slot,
            pos,
            rope,
            n,
            slots,
            codes_buf: Vec::new(),
            len_buf: Vec::new(),
            slot_buf,
            pcm_buf: Vec::new(),
        })
    };
    // Run every capacity once now: each program sequence is captured as a CUDA graph on first
    // use, and a capture that overlaps another thread's context synchronize (the LM engine) fails.
    let t = std::time::Instant::now();
    let voice_slots = if slot_caps.is_empty() {
        None
    } else {
        let t = |k: &str| pipeline.tensor(k).map_err(e);
        let tables = [
            t("slot.prompt_token")?,
            t("slot.prompt_feat")?,
            t("slot.spks")?,
            t("slot.prompt_len")?,
        ];
        let n = param("voice.slots")?;
        if slot_caps.iter().any(|c| c.0 > n) {
            return Err("codec voice-slot capacity exceeds its voice slots".into());
        }
        // Empty slots (prompt length 0) until a launch assigns them.
        for t in tables {
            runtime.write_tensor(t, &vec![0u8; t.bytes]).map_err(e)?;
        }
        Some(VoiceSlots {
            tables,
            prompt_max: param("prompt.max")?,
            resident: vec![None; n],
            used: vec![0; n],
            clock: 0,
        })
    };
    for (_, _, programs) in capacities
        .iter()
        .chain(&cached)
        .chain(&steps)
        .chain(&slot_caps)
    {
        runtime.run_sequence(programs).map_err(e)?;
    }
    tracing::info!(
        capacities = capacities.len() + cached.len() + steps.len() + slot_caps.len(),
        ms = t.elapsed().as_millis() as u64,
        "codec graphs warmed"
    );
    let lengths = (0..param("lengths.count")?)
        .map(|k| {
            Ok((
                pipeline.tensor(&format!("lengths.{k}")).map_err(e)?,
                param(&format!("lengths.{k}.rows_per_frame"))?,
            ))
        })
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
    let phase = phase
        .iter()
        .all(Option::is_some)
        .then(|| phase.map(Option::unwrap));
    if !cached.is_empty() && phase.is_none() {
        return Err("cached codec capacities without the window phase tensors".into());
    }
    Ok(Bound {
        slots: step_io
            .as_ref()
            .map(|io| std::sync::Arc::new(Mutex::new(SlotPool::new(io.slots)))),
        steps,
        step_io,
        cached,
        slot_caps,
        voice_slots,
        phase,
        harmonics: pipeline
            .parameter("vocoder.harmonics")
            .map(|v| v as usize)
            .unwrap_or(0),
        codes: pipeline.tensor("codes").map_err(e)?,
        seed,
        pcm: pipeline.tensor("pcm").map_err(e)?,
        frame_codes: param("codec.frame_codes")?,
        frame_samples: param("codec.frame_samples")?,
        window: param("stream.window_frames")?,
        lookahead: param("stream.lookahead_frames")?,
        voice,
        voices: pipeline
            .optional_string("voices")
            .map(|v| v.lines().map(str::to_owned).collect())
            .unwrap_or_default(),
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
        yield_programs: pipeline
            .parameter("render.yield_programs")
            .map(|v| v as usize)
            .unwrap_or(0),
    })
}

/// The smallest of `family` (ascending by batch * frames) holding `batch` items of `frames`.
fn smallest(family: &[Capacity], batch: usize, frames: usize) -> Option<&Capacity> {
    family.iter().find(|&&(b, f, _)| b >= batch && f >= frames)
}

/// Indices into `pending` (most urgent first) of one launch: the first job's family, then each
/// later job of that family while `fits(family, batch, frames)`, never a stream slot twice.
fn pack(
    pending: &[Job],
    max_batch: usize,
    fits: impl Fn(Family, usize, usize) -> bool,
) -> Vec<usize> {
    let family = pending[0].family();
    let mut frames = pending[0].frames;
    let mut take = vec![0];
    for (k, j) in pending.iter().enumerate().skip(1) {
        if take.len() >= max_batch {
            break;
        }
        if j.family() != family
            || j.step.is_some_and(|s| {
                take.iter().any(|&t| {
                    pending[t]
                        .step
                        .is_some_and(|o| o.slot.index == s.slot.index)
                })
            })
        {
            continue;
        }
        if !fits(family, take.len() + 1, frames.max(j.frames)) {
            break;
        }
        frames = frames.max(j.frames);
        take.push(k);
    }
    take
}

impl Bound {
    fn capacity(&self, family: Family, batch: usize, frames: usize) -> Option<&Capacity> {
        let caps = match family {
            Family::Decode => &self.capacities,
            Family::Window => &self.cached,
            Family::Step => &self.steps,
            Family::Slot => &self.slot_caps,
        };
        smallest(caps, batch, frames)
    }

    /// Why `j` cannot launch, if it cannot.
    fn check(&self, j: &Job) -> Result<(), &'static str> {
        if j.frames == 0 || j.codes.len() != j.frames * self.frame_codes {
            return Err("codec job has no frames or a partial frame");
        }
        if let Some(v) = &j.clone {
            let pm = self.voice_slots.as_ref().map_or(0, |s| s.prompt_max);
            let p = v.prompt_token.len();
            if p == 0
                || p > pm
                || v.prompt_feat.len() != 2 * p * 80
                || v.spks.len() != 80
                || !v.prompt_feat.iter().chain(&v.spks).all(|x| x.is_finite())
            {
                return Err("cloned voice does not fit the codec's voice slots");
            }
            if self.capacity(Family::Slot, 1, j.frames).is_none() {
                return Err("cloned-voice render exceeds every voice-slot capacity");
            }
        }
        let Some(step) = j.step else { return Ok(()) };
        if self.capacity(Family::Step, 1, j.frames).is_none() {
            return Err("codec stream step exceeds every step capacity");
        }
        match &self.slots {
            Some(pool) => pool.lock().check(step.slot, step.pos),
            None => Err("codec packet has no stream steps"),
        }
    }

    fn launch(&mut self, jobs: &[Job]) -> Result<Vec<Decoded>, String> {
        if jobs[0].step.is_none() {
            return self.decode(jobs);
        }
        let out = self.step(jobs);
        if let Some(pool) = &self.slots {
            let mut pool = pool.lock();
            for j in jobs {
                let s = j.step.expect("one family per launch");
                match out {
                    Ok(_) => pool.commit(s.slot, s.pos + j.frames),
                    Err(_) => pool.fail(s.slot),
                }
            }
        }
        out
    }

    /// One launch of stream steps on distinct slots; padding items take other slots with n = 0
    /// and pos = 1, which rewrites their state unchanged.
    fn step(&mut self, jobs: &[Job]) -> Result<Vec<Decoded>, String> {
        let frames = jobs.iter().map(|j| j.frames).max().unwrap_or(0);
        let (cb, cn, programs) = self
            .capacity(Family::Step, jobs.len(), frames)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "{} x {frames} step frames exceeds every capacity",
                    jobs.len()
                )
            })?;
        let (fc, fs) = (self.frame_codes, self.frame_samples);
        let io = self
            .step_io
            .as_mut()
            .ok_or("codec packet has no stream steps")?;
        let t0 = std::time::Instant::now();
        io.codes_buf.clear();
        io.codes_buf.resize(cb * cn * fc, 0);
        let mut spare = (0..io.slots as u32).filter(|&s| {
            !jobs
                .iter()
                .any(|j| j.step.is_some_and(|st| st.slot.index == s))
        });
        for b in 0..cb {
            let (slot, pos, n) = match jobs.get(b) {
                Some(j) => {
                    let st = j.step.expect("one family per launch");
                    for (d, &c) in io.codes_buf[b * cn * fc..].iter_mut().zip(&j.codes) {
                        *d = c.clamp(0, i32::MAX) as u32;
                    }
                    (st.slot.index, st.pos as u32, j.frames as u32)
                }
                None => (spare.next().ok_or("no spare codec stream slot")?, 1, 0),
            };
            (io.slot_buf[b], io.pos_buf[b], io.n_buf[b]) = (slot, pos, n);
            for r in 0..cn {
                io.rope_buf[b * cn + r] = pos + r as u32;
            }
        }
        let e = |x: crate::RuntimeError| x.to_string();
        let rt = &mut self.runtime;
        rt.write_tensor_at(self.codes, 0, bytemuck::cast_slice(&io.codes_buf))
            .map_err(e)?;
        rt.write_tensor(io.slot, bytemuck::cast_slice(&io.slot_buf))
            .map_err(e)?;
        rt.write_tensor(io.pos, bytemuck::cast_slice(&io.pos_buf))
            .map_err(e)?;
        rt.write_tensor(io.rope, bytemuck::cast_slice(&io.rope_buf))
            .map_err(e)?;
        for (&t, &(_, per_frame)) in io.n.iter().zip(&self.lengths) {
            io.len_buf.clear();
            io.len_buf.resize(t.bytes / 4, 0);
            for (l, &n) in io.len_buf.iter_mut().zip(&io.n_buf[..cb]) {
                *l = n * per_frame as u32;
            }
            rt.write_tensor(t, bytemuck::cast_slice(&io.len_buf))
                .map_err(e)?;
        }
        let t1 = std::time::Instant::now();
        rt.run_sequence(&programs).map_err(e)?;
        let gpu_us = rt.last_run_us();
        let t2 = std::time::Instant::now();
        io.pcm_buf.resize(cb * cn * fs, 0.0);
        rt.read_tensor_at(
            self.pcm,
            0,
            bytemuck::cast_slice_mut(&mut io.pcm_buf[..cb * cn * fs]),
        )
        .map_err(e)?;
        tracing::debug!(
            target: "plowrt::tts::codec_launch",
            jobs = jobs.len(),
            step = true,
            frames,
            cb,
            cf = cn,
            h2d_us = (t1 - t0).as_micros() as u64,
            run_us = (t2 - t1).as_micros() as u64,
            gpu_us = gpu_us as u64,
            d2h_us = t2.elapsed().as_micros() as u64,
            "codec launch"
        );
        Ok(jobs
            .iter()
            .enumerate()
            .map(|(b, j)| Decoded {
                pcm: io.pcm_buf[b * cn * fs..][..j.frames * fs].to_vec(),
                phase: Vec::new(),
            })
            .collect())
    }

    fn decode(&mut self, jobs: &[Job]) -> Result<Vec<Decoded>, String> {
        let frames = jobs.iter().map(|j| j.frames).max().unwrap_or(0);
        let window = jobs[0].window.is_some();
        let family = jobs[0].family();
        let (cb, cf, programs) = self
            .capacity(family, jobs.len(), frames)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "{} x {frames} frames exceeds every codec capacity",
                    jobs.len()
                )
            })?;
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
        self.runtime
            .write_tensor_at(self.codes, 0, bytemuck::cast_slice(&codes))
            .map_err(e)?;
        for &(tensor, per_frame) in &self.lengths {
            let mut len = vec![0u32; tensor.bytes / 4];
            for (i, j) in jobs.iter().enumerate() {
                len[i] = (j.frames * per_frame) as u32;
            }
            self.runtime
                .write_tensor(tensor, bytemuck::cast_slice(&len))
                .map_err(e)?;
        }
        let mut seeds = vec![0u64; self.seed.bytes / 8];
        for (i, j) in jobs.iter().enumerate() {
            seeds[i] = j.seed;
        }
        self.runtime
            .write_tensor(self.seed, bytemuck::cast_slice(&seeds))
            .map_err(e)?;
        let slots = match (family, self.voice_slots.as_mut()) {
            (Family::Slot, Some(vs)) => {
                let voices: Vec<_> = jobs.iter().filter_map(|j| j.clone.as_ref()).collect();
                Some(vs.assign(self.runtime.as_mut(), &voices)?)
            }
            (Family::Slot, None) => return Err("codec packet has no voice slots".into()),
            _ => None,
        };
        if let Some(voice) = self.voice {
            let mut v = vec![0u32; voice.bytes / 4];
            for (i, j) in jobs.iter().enumerate() {
                v[i] = slots.as_ref().map_or(j.voice, |s| s[i]);
            }
            self.runtime
                .write_tensor(voice, bytemuck::cast_slice(&v))
                .map_err(e)?;
        }
        let nh = self.harmonics;
        let windows = if window { self.phase } else { None };
        if let Some([phase, _, seam, next_seam]) = windows {
            let mut p = vec![0f32; phase.bytes / 4];
            let (mut s, mut ns) = (vec![0u32; seam.bytes / 4], vec![0u32; next_seam.bytes / 4]);
            for (i, w) in jobs.iter().filter_map(|j| j.window.as_ref()).enumerate() {
                p[i * nh..][..w.phase.len().min(nh)]
                    .copy_from_slice(&w.phase[..w.phase.len().min(nh)]);
                (s[i], ns[i]) = (w.seam, w.next_seam);
            }
            self.runtime
                .write_tensor(phase, bytemuck::cast_slice(&p))
                .map_err(e)?;
            self.runtime
                .write_tensor(seam, bytemuck::cast_slice(&s))
                .map_err(e)?;
            self.runtime
                .write_tensor(next_seam, bytemuck::cast_slice(&ns))
                .map_err(e)?;
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
        self.runtime
            .read_tensor_at(self.pcm, 0, bytemuck::cast_slice_mut(&mut pcm))
            .map_err(e)?;
        let mut phase_out = Vec::new();
        if let Some([_, out, _, _]) = windows {
            phase_out = vec![0f32; out.bytes / 4];
            self.runtime
                .read_tensor(out, bytemuck::cast_slice_mut(&mut phase_out))
                .map_err(e)?;
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
                phase: if window {
                    phase_out[i * nh..(i + 1) * nh].to_vec()
                } else {
                    Vec::new()
                },
            })
            .collect())
    }
}

fn run(rx: mpsc::Receiver<Msg>, mut codec: Bound) {
    let max_batch = codec
        .capacities
        .iter()
        .chain(&codec.cached)
        .chain(&codec.steps)
        .chain(&codec.slot_caps)
        .map(|c| c.0)
        .max()
        .unwrap_or(1);
    let mut pending: Vec<Job> = Vec::new();
    while let Ok(first) = rx.recv() {
        for m in std::iter::once(first).chain(rx.try_iter()) {
            match m {
                Msg::Job(j) => pending.push(j),
                // The stream is gone and no launch holds its slot: drop its queued step.
                Msg::Release(id) => {
                    pending.retain(|j| j.step.is_none_or(|s| s.slot != id));
                    if let Some(pool) = &codec.slots {
                        pool.lock().release(id);
                    }
                }
            }
        }
        // Most urgent first; within an urgency longest first, so a launch's capacity is mostly
        // set by its first job.
        pending.sort_by_key(|j| (j.urgency, std::cmp::Reverse(j.frames)));
        while !pending.is_empty() {
            // A failed launch or a committed step can turn a queued job unlaunchable.
            let mut k = 0;
            while k < pending.len() {
                match codec.check(&pending[k]) {
                    Ok(()) => k += 1,
                    Err(e) => drop(pending.remove(k).reply.send(Err(e.into()))),
                }
            }
            if pending.is_empty() {
                break;
            }
            let take = pack(&pending, max_batch, |f, b, n| {
                codec.capacity(f, b, n).is_some()
            });
            let mut batch: Vec<Job> = take.iter().rev().map(|&k| pending.remove(k)).collect();
            batch.reverse();
            match codec.launch(&batch) {
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
    use super::*;

    fn job(window: bool, frames: usize, slot: Option<u32>) -> Job {
        Job {
            codes: Vec::new(),
            frames,
            seed: 0,
            voice: 0,
            urgency: Urgency::Stream,
            window: window.then(Window::default),
            step: slot.map(|index| Step {
                slot: SlotId { index, gen: 0 },
                pos: 0,
            }),
            clone: None,
            reply: tokio::sync::oneshot::channel().0,
            _work: None,
        }
    }

    #[test]
    fn slot_pool_lifecycle() {
        let mut pool = SlotPool::new(2);
        let a = pool.acquire().unwrap();
        let b = pool.acquire().unwrap();
        assert_ne!(a.index, b.index);
        assert!(
            pool.acquire().is_none(),
            "exhausted: the stream decodes windows"
        );
        assert_eq!(pool.check(a, 0), Ok(()));
        assert!(pool.check(a, 3).is_err(), "nothing committed yet");
        pool.commit(a, 3);
        assert_eq!(pool.check(a, 3), Ok(()));
        assert!(pool.check(a, 2).is_err());
        assert_eq!(pool.check(a, 0), Ok(()), "a restart");
        pool.release(a);
        assert!(pool.check(a, 0).is_err(), "released");
        let c = pool.acquire().unwrap();
        assert_eq!(c.index, a.index);
        assert_ne!(c.gen, a.gen);
        assert!(
            pool.check(a, 0).is_err(),
            "a stale job never runs on the reused slot"
        );
        assert_eq!(pool.check(c, 0), Ok(()), "a reused slot starts fresh");
        pool.release(a);
        assert_eq!(pool.check(c, 0), Ok(()), "a stale release is ignored");
        pool.fail(b);
        assert!(pool.check(b, 0).is_err(), "failed until released");
        pool.release(b);
        let d = pool.acquire().unwrap();
        assert_eq!((d.index, pool.check(d, 0)), (b.index, Ok(())));
    }

    #[test]
    fn steps_pack_into_the_smallest_capacity() {
        let caps: Vec<Capacity> = [
            (1, 1),
            (1, 4),
            (1, 17),
            (4, 17),
            (8, 17),
            (16, 17),
            (32, 17),
        ]
        .map(|(b, n)| (b, n, Vec::new()))
        .to_vec();
        let at = |b, n| smallest(&caps, b, n).map(|c| (c.0, c.1));
        assert_eq!(at(1, 1), Some((1, 1)));
        assert_eq!(at(1, 3), Some((1, 4)));
        assert_eq!(at(1, 5), Some((1, 17)));
        assert_eq!(at(2, 1), Some((4, 17)));
        assert_eq!(at(5, 17), Some((8, 17)));
        assert_eq!(at(33, 1), None);
        assert_eq!(at(1, 18), None);

        // One family per launch, never a slot twice, at most `max_batch`.
        let pending = [
            job(false, 4, Some(3)),
            job(false, 9, None),
            job(false, 2, Some(3)),
            job(true, 2, None),
            job(false, 17, Some(5)),
            job(false, 1, Some(7)),
        ];
        let fits = |f: Family, b, n| {
            assert_eq!(f, Family::Step);
            at(b, n).is_some()
        };
        assert_eq!(pack(&pending, 32, fits), vec![0, 4, 5]);
        assert_eq!(pack(&pending, 2, fits), vec![0, 4]);
        // A job past every capacity ends the launch.
        assert_eq!(
            pack(&pending, 32, |_, b, n| at(b, n.min(16)).is_some() && n <= 4),
            vec![0]
        );
        assert_eq!(pack(&pending[1..], 32, |_, _, _| true), vec![0]);
    }
    use crate::serve::cosched::{Band, Due};
    use std::time::{Duration, Instant};

    #[test]
    fn render_yields_only_to_an_outranking_first_output_or_near_miss() {
        let now = Instant::now();
        let at = |ms: i64, band| Due {
            deadline: if ms >= 0 {
                now + Duration::from_millis(ms as u64)
            } else {
                now - Duration::from_millis(-ms as u64)
            },
            cost: Duration::ZERO,
            band,
        };
        let m = 5_000_000;
        let stream = at(100, Band::Stream);
        assert!(!should_yield(None, &stream, now, m));
        assert!(
            should_yield(Some(at(20, Band::Stream)), &stream, now, m),
            "about to miss"
        );
        assert!(
            !should_yield(Some(at(40, Band::Stream)), &stream, now, m),
            "decode, not urgent"
        );
        assert!(!should_yield(Some(at(200, Band::Stream)), &stream, now, m));
        assert!(
            should_yield(Some(at(-50, Band::Stream)), &at(-10, Band::Stream), now, m),
            "both late: the later one first"
        );
        assert!(
            !should_yield(Some(at(-5, Band::Stream)), &at(-10, Band::Stream), now, m),
            "within the margin"
        );
        // A first output takes the device from window renders, not from an urgent one or a
        // tighter first audio.
        assert!(should_yield(Some(at(700, Band::First)), &stream, now, m));
        assert!(!should_yield(
            Some(at(700, Band::First)),
            &at(20, Band::Stream),
            now,
            m
        ));
        assert!(!should_yield(
            Some(at(700, Band::First)),
            &at(400, Band::First),
            now,
            m
        ));
        assert!(!should_yield(
            Some(at(10, Band::Stream)),
            &at(5, Band::Stream),
            now,
            m
        ));
        assert!(!should_yield(Some(at(10, Band::Bulk)), &stream, now, m));
        // A window render takes the device as a first output; a first output stays one.
        assert_eq!(super::serial_due(stream).band, Band::First);
        assert_eq!(super::serial_due(at(400, Band::First)).band, Band::First);
        // An ASR final takes the device from any render that is not urgent; late, it competes with
        // an urgent render by slack.
        assert!(should_yield(
            Some(at(480, Band::Final)),
            &at(300, Band::First),
            now,
            m
        ));
        assert!(!should_yield(
            Some(at(480, Band::Final)),
            &at(20, Band::Stream),
            now,
            m
        ));
        assert!(!should_yield(
            Some(at(-10, Band::Final)),
            &at(-300, Band::Stream),
            now,
            m
        ));
        assert!(should_yield(
            Some(at(-300, Band::Final)),
            &at(-10, Band::Stream),
            now,
            m
        ));
    }
}
