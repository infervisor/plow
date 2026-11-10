//! The codec stage of a speech pipeline: a `codec.v1` packet (codes or speech tokens in, PCM out,
//! one program or program sequence per (batch, frames) capacity: roles `decode.b{B}.f{F}` or
//! `synth.b{B}.t{T}.{stage}`) on the packet runtime, driven by one worker thread. Pending jobs share
//! each launch the way concurrent streams share LM decode steps; per-item valid lengths, seeds and
//! voice indices make every item decode exactly as it would alone.

use std::path::Path;
use std::sync::mpsc;

use parking_lot::Mutex;

use crate::exec::packet_runtime::{
    load_packet_runtime_on, PacketAsset, PacketData, PacketTensor, PacketWrite,
};

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

/// A voice-slot render's NSF initial phases (per harmonic): the `synth` programs' own draw
/// (`RandF32` stream 2 at (harmonic, 0), `(u - 0.5) * tau`, harmonic 0 at 0), fed through the
/// window phase input with an out-of-range seam so a whole render matches `synth` bit for bit.
pub fn source_phase(seed: u64, harmonics: usize) -> Vec<f32> {
    let mix = |z: u64| {
        let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    (0..harmonics as u64)
        .map(|h| {
            if h == 0 {
                return 0.0;
            }
            let x = mix(seed ^ mix((2u64 << 56) ^ (h << 32)));
            ((x >> 40) as f32 * (1.0 / 16_777_216.0) - 0.5) * std::f32::consts::TAU
        })
        .collect()
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

    fn live(&self) -> usize {
        self.slots.len() - self.free.len()
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
    /// A cloned-voice window on the slot cache capacities (`cvsynth`).
    SlotCached,
}

impl Family {
    /// The bucket-demand family of this capacity list (its batch dimension).
    fn bucket(self) -> crate::obs::bucket::Family {
        use crate::obs::bucket::Family as B;
        match self {
            Family::Decode => B::CodecDecode,
            Family::Window => B::CodecWindow,
            Family::Step => B::Codec,
            Family::Slot => B::CodecSlot,
            Family::SlotCached => B::CodecSlotWindow,
        }
    }
}

enum Msg {
    Job(Job),
    Release(SlotId),
    Observe(std::sync::Arc<crate::obs::bucket::BucketDemand>),
}

pub(crate) use super::Codes;

fn code_writes<'a>(
    tensor: PacketTensor,
    host: &'a [u32],
    jobs: &'a [Job],
    capacity_frames: usize,
    fc: usize,
) -> smallvec::SmallVec<[PacketWrite<'a>; 64]> {
    if jobs
        .iter()
        .all(|j| j.codes.sources.iter().all(Option::is_none))
    {
        return smallvec::smallvec![PacketWrite {
            tensor,
            offset: 0,
            data: PacketData::Host(bytemuck::cast_slice(host)),
        }];
    }
    let mut writes = smallvec::smallvec![PacketWrite {
        tensor,
        offset: 0,
        data: PacketData::Zero(std::mem::size_of_val(host)),
    }];
    for frame in 0..capacity_frames {
        for (b, job) in jobs.iter().enumerate().filter(|(_, j)| frame < j.frames) {
            let words = (b * capacity_frames + frame) * fc;
            let source = job
                .codes
                .sources
                .get(frame)
                .and_then(Option::as_ref)
                .filter(|_| {
                    job.codes[frame * fc..(frame + 1) * fc]
                        .iter()
                        .all(|&c| c >= 0)
                });
            writes.push(PacketWrite {
                tensor,
                offset: words * 4,
                data: source.map_or_else(
                    || PacketData::Host(bytemuck::cast_slice(&host[words..words + fc])),
                    PacketData::Buffer,
                ),
            });
        }
    }
    writes
}

struct Job {
    codes: Codes,
    frames: usize,
    seed: u64,
    voice: u32,
    urgency: Urgency,
    /// When the stream's playback reaches the end of the audio already sent: a step may wait
    /// for a batch until shortly before it.
    due: Option<std::time::Instant>,
    window: Option<Window>,
    step: Option<Step>,
    clone: Option<std::sync::Arc<SlotVoice>>,
    /// The clone's prompt K/V is resident in a cache slot (set by the worker).
    cached: bool,
    reply: tokio::sync::oneshot::Sender<Result<Decoded, String>>,
    /// Backlog charged to the feeding model's mux until this job is answered.
    _work: Option<crate::sched::admission::DownstreamWork>,
    /// A first audio holds the feeding model's next tick until it is answered: both stages'
    /// kernels take the whole device, so a tick launched ahead of it delays the first audio.
    _urgent: Option<crate::sched::admission::DownstreamUrgent>,
    /// PCM of this step's frames already decoded by earlier pieces of a bounded launch.
    carry: Vec<f32>,
    /// Enqueued at (kept across the pieces of a bounded launch): the aging base.
    since: std::time::Instant,
    /// A first audio's target (`since` + the packet's `stream.first_audio_target_ms`); a stream
    /// step's is its `due`.
    deadline: Option<std::time::Instant>,
}

impl Job {
    /// When this job must be done: its first-audio target or playback `due`.
    fn deadline(&self) -> Option<std::time::Instant> {
        self.deadline.or(self.due)
    }

    fn family(&self) -> Family {
        match (&self.window, &self.step, &self.clone) {
            (_, Some(_), _) => Family::Step,
            (Some(_), None, Some(_)) if self.cached => Family::SlotCached,
            (_, None, Some(_)) => Family::Slot,
            (Some(_), None, None) => Family::Window,
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
    /// A first audio's target from enqueue (`stream.first_audio_target_ms`, else the turn TTFA
    /// target): the deadline the scheduler ranks and bounds launches by.
    first_target: std::time::Duration,
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
    /// Transformer rows per frame (`stream.transformer_rows_per_frame`, Mimi 2): `rope` covers the
    /// rows and `row_pos` is each item's position in rows.
    xf_rpf: usize,
    row_pos: Option<PacketTensor>,
    codes_buf: Vec<u32>,
    slot_buf: Vec<u32>,
    pos_buf: Vec<u32>,
    rope_buf: Vec<u32>,
    row_buf: Vec<u32>,
    n_buf: Vec<u32>,
    len_buf: Vec<Vec<u32>>,
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

/// Cloned voices' prompt K/V caches (`cprefill` writes cache slot `cslot`); `cvsynth` windows
/// attend to them. A prefill depends only on the voice, so an evicted voice refills identically.
struct CacheSlots {
    prefill: Vec<usize>,
    cslot: PacketTensor,
    resident: Vec<Option<std::sync::Arc<SlotVoice>>>,
    used: Vec<std::time::Instant>,
}

/// A cache slot is taken from another voice only after this long unused: busier voices render
/// on the voice-slot capacities instead of refilling caches every window.
const CACHE_IDLE: std::time::Duration = std::time::Duration::from_secs(2);
/// Cached windows attend to `2 P - 8` prompt rows plus 8 tail rows.
const CACHE_MIN_PROMPT: usize = 4;

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
    /// Cloned-voice window capacities over the slot caches (`cvsynth`).
    cslot_caps: Vec<Capacity>,
    cache_slots: Option<CacheSlots>,
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
    buckets: std::sync::Arc<crate::obs::bucket::BucketDemand>,
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
            first_target: first_audio_target(&parameters),
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
    /// `frames * frame_samples` samples continuing its audio. `due`: see [`Job::due`].
    pub async fn step(
        &self,
        slot: &mut StreamSlot,
        codes: Vec<i32>,
        frames: usize,
        urgency: Urgency,
        due: Option<std::time::Instant>,
    ) -> Result<Vec<f32>, String> {
        self.step_codes(slot, codes.into(), frames, urgency, due)
            .await
    }

    pub(crate) async fn step_codes(
        &self,
        slot: &mut StreamSlot,
        codes: Codes,
        frames: usize,
        urgency: Urgency,
        due: Option<std::time::Instant>,
    ) -> Result<Vec<f32>, String> {
        let step = Step {
            slot: slot.id,
            pos: slot.pos,
        };
        let pcm = self
            .submit(codes, frames, 0, 0, (urgency, due), None, Some(step), None)
            .await?
            .pcm;
        slot.pos += frames;
        Ok(pcm)
    }

    /// Consume a reference prefix without emitting its audio; failure or cancellation releases the slot.
    pub(crate) async fn prime_codes(
        &self,
        mut slot: StreamSlot,
        codes: Codes,
        frames: usize,
        urgency: Urgency,
        due: Option<std::time::Instant>,
    ) -> Result<StreamSlot, String> {
        if self.step_frames == 0 || slot.pos != 0 {
            return Err("codec priming requires a fresh stateful slot".into());
        }
        if frames.checked_mul(self.frame_codes) != Some(codes.len()) {
            return Err("codec reference prefix has incomplete frames".into());
        }
        let pool = self
            .slots
            .as_ref()
            .ok_or("codec has no stateful slot pool")?;
        pool.lock().check(slot.id, 0).map_err(str::to_owned)?;
        let mut start = 0;
        while start < frames {
            let n = (frames - start).min(self.step_frames);
            let chunk = codes.slice(
                start * self.frame_codes..(start + n) * self.frame_codes,
                self.frame_codes,
            );
            let pcm = match self.step_codes(&mut slot, chunk, n, urgency, due).await {
                Ok(pcm) => pcm,
                Err(e) => {
                    pool.lock().fail(slot.id);
                    return Err(e);
                }
            };
            if n.checked_mul(self.frame_samples) != Some(pcm.len())
                || pcm.iter().any(|v| !v.is_finite())
            {
                pool.lock().fail(slot.id);
                return Err("codec returned invalid reference-prefix PCM".into());
            }
            start += n;
        }
        Ok(slot)
    }

    pub(crate) async fn decode_stateful(
        &self,
        mut slot: StreamSlot,
        codes: Codes,
        frames: usize,
        urgency: Urgency,
    ) -> Result<(StreamSlot, Vec<f32>), String> {
        if self.step_frames == 0
            || slot.pos != 0
            || frames.checked_mul(self.frame_codes) != Some(codes.len())
        {
            return Err("stateful decoding requires a fresh slot and complete frames".into());
        }
        let pool = self
            .slots
            .as_ref()
            .ok_or("codec has no stateful slot pool")?;
        pool.lock().check(slot.id, 0).map_err(str::to_owned)?;
        let samples = frames
            .checked_mul(self.frame_samples)
            .ok_or("codec sample count overflows")?;
        let mut output = Vec::with_capacity(samples);
        let mut start = 0;
        while start < frames {
            let n = (frames - start).min(self.step_frames);
            let chunk = codes.slice(
                start * self.frame_codes..(start + n) * self.frame_codes,
                self.frame_codes,
            );
            let pcm = match self.step_codes(&mut slot, chunk, n, urgency, None).await {
                Ok(pcm) => pcm,
                Err(e) => {
                    pool.lock().fail(slot.id);
                    return Err(e);
                }
            };
            if n.checked_mul(self.frame_samples) != Some(pcm.len())
                || pcm.iter().any(|v| !v.is_finite())
            {
                pool.lock().fail(slot.id);
                return Err("codec returned invalid stateful PCM".into());
            }
            output.extend_from_slice(&pcm);
            start += n;
        }
        Ok((slot, output))
    }

    /// Split each launch into segments of `render.yield_programs` programs (a CFM step) and call
    /// `hook` between them, so a co-scheduler can hand the device to another model mid-render.
    /// `None` restores one launch per decode.
    pub fn set_yield(&self, hook: Option<YieldHook>) {
        *self.yield_hook.lock() = hook;
    }

    /// Count stream-step launches into the serving model's bucket-demand counters.
    pub fn observe_buckets(&self, buckets: &std::sync::Arc<crate::obs::bucket::BucketDemand>) {
        let _ = self.tx.lock().send(Msg::Observe(buckets.clone()));
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

    pub(crate) async fn decode_codes(
        &self,
        codes: Codes,
        frames: usize,
        seed: u64,
        urgency: Urgency,
        due: Option<std::time::Instant>,
    ) -> Result<Vec<f32>, String> {
        self.submit(codes, frames, seed, 0, (urgency, due), None, None, None)
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
            (Urgency::Stream, None),
            None,
            None,
            Some(voice),
        )
        .await
        .map(|d| d.pcm)
    }

    /// [`Self::decode_window`] in a cloned voice, on the voice-slot capacities.
    pub async fn decode_slot_window(
        &self,
        codes: Vec<i32>,
        frames: usize,
        seed: u64,
        voice: std::sync::Arc<SlotVoice>,
        window: Window,
    ) -> Result<Decoded, String> {
        self.submit(
            codes,
            frames,
            seed,
            0,
            (Urgency::Stream, None),
            Some(window),
            None,
            Some(voice),
        )
        .await
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
        self.submit(
            codes,
            frames,
            seed,
            voice,
            (urgency, None),
            window,
            None,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit(
        &self,
        codes: impl Into<Codes>,
        frames: usize,
        seed: u64,
        voice: u32,
        (urgency, due): (Urgency, Option<std::time::Instant>),
        window: Option<Window>,
        step: Option<Step>,
        clone: Option<std::sync::Arc<SlotVoice>>,
    ) -> Result<Decoded, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let now = std::time::Instant::now();
        let _work = self.credit.as_ref().map(|c| c.work());
        let _urgent = self
            .credit
            .as_ref()
            .filter(|_| urgency == Urgency::First)
            .map(|c| c.urgent_work());
        self.tx
            .lock()
            .send(Msg::Job(Job {
                codes: codes.into(),
                frames,
                seed,
                voice,
                urgency,
                due,
                window,
                step,
                clone,
                cached: false,
                reply,
                _work,
                _urgent,
                carry: Vec::new(),
                since: now,
                deadline: (urgency == Urgency::First).then(|| now + self.first_target),
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
    let mut cvsynth: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize)>> =
        Default::default();
    let mut cprefill: Vec<(usize, usize)> = Vec::new();
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
            [kind @ ("synth" | "csynth" | "vsynth" | "cvsynth"), b, t, stage] => {
                if let (Some(key), Ok(stage)) = (dims(b, t, "b", "t"), stage.parse::<usize>()) {
                    let family = match *kind {
                        "synth" => &mut sequences,
                        "csynth" => &mut cached,
                        "vsynth" => &mut vsynth,
                        _ => &mut cvsynth,
                    };
                    family.entry(key).or_default().push((stage, program));
                }
            }
            ["cprefill", stage] => {
                if let Ok(stage) = stage.parse::<usize>() {
                    cprefill.push((stage, program));
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
    let cslot_caps = family(cvsynth);
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
        let xf_rpf = pipeline
            .parameter("stream.transformer_rows_per_frame")
            .unwrap_or(1)
            .max(1) as usize;
        let row_pos = if xf_rpf > 1 {
            Some(pipeline.tensor("stream.row_pos").map_err(e)?)
        } else {
            None
        };
        let mut slot_buf = elems(slot);
        if steps.iter().any(|&(b, n, _)| {
            b > slots.min(slot_buf.len()).min(pos.bytes / 4)
                || b * n * xf_rpf > rope.bytes / 4
                || row_pos.is_some_and(|t| b > t.bytes / 4)
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
        let len_buf = n.iter().map(|t| vec![0u32; t.bytes / 4]).collect();
        Some(StepIo {
            pos_buf: elems(pos),
            rope_buf: elems(rope),
            row_buf: row_pos.map(elems).unwrap_or_default(),
            n_buf: elems(slot),
            slot,
            pos,
            rope,
            n,
            slots,
            xf_rpf,
            row_pos,
            codes_buf: Vec::new(),
            len_buf,
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
    let cache_slots = if cslot_caps.is_empty() {
        None
    } else {
        let n = param("voice.cache_slots")?;
        if voice_slots.is_none() || cprefill.is_empty() || n == 0 {
            return Err(
                "codec slot-cache capacities need voice slots, cache slots and cprefill".into(),
            );
        }
        let cslot = pipeline.tensor("cslot").map_err(e)?;
        if cslot_caps.iter().any(|c| c.0 > cslot.bytes / 4) {
            return Err("codec slot-cache capacity exceeds its cslot input".into());
        }
        runtime
            .write_tensor(cslot, &vec![0u8; cslot.bytes])
            .map_err(e)?;
        Some(CacheSlots {
            prefill: ordered(cprefill),
            cslot,
            resident: vec![None; n],
            used: vec![std::time::Instant::now(); n],
        })
    };
    for (_, _, programs) in capacities
        .iter()
        .chain(&cached)
        .chain(&steps)
        .chain(&slot_caps)
        .chain(&cslot_caps)
    {
        runtime.run_sequence(programs).map_err(e)?;
    }
    if let Some(c) = &cache_slots {
        runtime.run_sequence(&c.prefill).map_err(e)?;
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
    if !(cached.is_empty() && slot_caps.is_empty()) && phase.is_none() {
        return Err("cached codec capacities without the window phase tensors".into());
    }
    let mut bound = Bound {
        slots: step_io
            .as_ref()
            .map(|io| std::sync::Arc::new(Mutex::new(SlotPool::new(io.slots)))),
        steps,
        step_io,
        cached,
        slot_caps,
        voice_slots,
        cslot_caps,
        cache_slots,
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
        buckets: Default::default(),
        yield_programs: pipeline
            .parameter("render.yield_programs")
            .map(|v| v as usize)
            .unwrap_or(0),
    };
    let mut inputs: smallvec::SmallVec<[PacketTensor; 16]> =
        smallvec::smallvec![bound.codes, bound.seed];
    inputs.extend(bound.lengths.iter().map(|&(t, _)| t));
    if let Some(io) = &bound.step_io {
        inputs.extend([io.slot, io.pos, io.rope]);
        inputs.extend(io.row_pos);
        inputs.extend(io.n.iter().copied());
    }
    bound.runtime.prepare_writes(&inputs).map_err(e)?;
    Ok(bound)
}

/// The smallest of `family` (ascending by batch * frames) holding `batch` items of `frames`.
fn smallest(family: &[Capacity], batch: usize, frames: usize) -> Option<&Capacity> {
    family.iter().find(|&&(b, f, _)| b >= batch && f >= frames)
}

/// Indices into `pending` (most urgent first) of one launch: the first job's family, then each
/// later job of that family while `fits(family, batch, frames)`, never a stream slot twice. A
/// first-audio head takes no job longer than itself: a longer one would delay its first audio.
fn pack(
    pending: &[Job],
    max_batch: usize,
    fits: impl Fn(Family, usize, usize) -> bool,
) -> Vec<usize> {
    let family = pending[0].family();
    let first = pending[0].urgency == Urgency::First;
    let mut frames = pending[0].frames;
    let mut take = vec![0];
    for (k, j) in pending.iter().enumerate().skip(1) {
        if take.len() >= max_batch {
            break;
        }
        if j.family() != family
            || (first && j.frames > frames)
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
    fn capacities_of(&self, family: Family) -> &[Capacity] {
        match family {
            Family::Decode => &self.capacities,
            Family::Window => &self.cached,
            Family::Step => &self.steps,
            Family::Slot => &self.slot_caps,
            Family::SlotCached => &self.cslot_caps,
        }
    }

    fn capacity(&self, family: Family, batch: usize, frames: usize) -> Option<&Capacity> {
        smallest(self.capacities_of(family), batch, frames)
    }

    /// Why `j` cannot launch, if it cannot.
    fn check(&self, j: &Job) -> Result<(), &'static str> {
        if j.frames == 0 || j.codes.len() != j.frames * self.frame_codes {
            return Err("codec job has no frames or a partial frame");
        }
        if !j.codes.sources.is_empty()
            && (j.codes.sources.len() != j.frames
                || j.codes
                    .sources
                    .iter()
                    .flatten()
                    .any(|s| s.len() != self.frame_codes * 4))
        {
            return Err("codec device frames disagree with the code shape");
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

    /// Mark the cloned-voice windows whose voice holds (or now takes) a cache slot; a voice
    /// without one renders on the voice-slot capacities.
    fn place_caches(&mut self, pending: &mut [Job]) {
        let Some(cs) = self.cache_slots.as_ref() else {
            return;
        };
        let n = cs.resident.len();
        for k in 0..pending.len() {
            let j = &pending[k];
            let Some(v) = j.clone.as_ref().filter(|_| j.window.is_some() && !j.cached) else {
                continue;
            };
            if self.check(j).is_err() {
                continue;
            }
            if v.prompt_token.len() < CACHE_MIN_PROMPT
                || smallest(&self.cslot_caps, 1, j.frames).is_none()
            {
                continue;
            }
            let v = std::sync::Arc::clone(v);
            let cs = self.cache_slots.as_ref().expect("checked above");
            let hit = cs
                .resident
                .iter()
                .position(|r| r.as_ref().is_some_and(|r| std::sync::Arc::ptr_eq(r, &v)));
            let victim = match hit {
                Some(_) => None,
                None => {
                    let held = |s: usize| {
                        cs.resident[s].as_ref().is_some_and(|r| {
                            pending.iter().any(|p| {
                                p.cached
                                    && p.clone
                                        .as_ref()
                                        .is_some_and(|c| std::sync::Arc::ptr_eq(c, r))
                            })
                        })
                    };
                    (0..n)
                        .filter(|&s| {
                            cs.resident[s].is_none()
                                || (!held(s) && cs.used[s].elapsed() >= CACHE_IDLE)
                        })
                        .min_by_key(|&s| (cs.resident[s].is_some(), cs.used[s]))
                }
            };
            let slot = hit.or(victim.filter(|&s| match self.prefill_cache(s, &v) {
                Ok(()) => true,
                Err(err) => {
                    tracing::warn!(error = %err, "tts: voice cache prefill failed");
                    false
                }
            }));
            if slot.is_some() {
                for p in pending.iter_mut() {
                    if p.window.is_some()
                        && p.clone
                            .as_ref()
                            .is_some_and(|c| std::sync::Arc::ptr_eq(c, &v))
                        && smallest(&self.cslot_caps, 1, p.frames).is_some()
                    {
                        p.cached = true;
                    }
                }
            }
        }
    }

    /// Cache slot `s` <- `v`'s prompt K/V: its tables into a voice slot, then `cprefill`
    /// (prompt alone, seed 0x5eed, as the builtin voices' prefill).
    fn prefill_cache(&mut self, s: usize, v: &std::sync::Arc<SlotVoice>) -> Result<(), String> {
        let e = |x: crate::RuntimeError| x.to_string();
        let cs = self.cache_slots.as_mut().ok_or("no slot caches")?;
        cs.resident[s] = None;
        let vs = self.voice_slots.as_mut().ok_or("no voice slots")?;
        let table = vs.assign(self.runtime.as_mut(), &[v])?[0];
        let (voice, Some(&(count, _))) =
            (self.voice.ok_or("no voice input")?, self.lengths.first())
        else {
            return Err("slot cache prefill needs a length input".into());
        };
        let zero = |t: PacketTensor| vec![0u8; t.bytes];
        let mut vb = zero(voice);
        vb[..4].copy_from_slice(&table.to_le_bytes());
        self.runtime.write_tensor(voice, &vb).map_err(e)?;
        self.runtime.write_tensor(count, &zero(count)).map_err(e)?;
        let mut sb = zero(self.seed);
        sb[..8].copy_from_slice(&0x5eed_u64.to_le_bytes());
        self.runtime.write_tensor(self.seed, &sb).map_err(e)?;
        let cs = self.cache_slots.as_mut().expect("checked above");
        let mut cb = zero(cs.cslot);
        cb[..4].copy_from_slice(&(s as u32).to_le_bytes());
        self.runtime.write_tensor(cs.cslot, &cb).map_err(e)?;
        self.runtime.run_sequence(&cs.prefill).map_err(e)?;
        cs.resident[s] = Some(std::sync::Arc::clone(v));
        cs.used[s] = std::time::Instant::now();
        Ok(())
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
                    for (d, &c) in io.codes_buf[b * cn * fc..].iter_mut().zip(j.codes.iter()) {
                        *d = c.clamp(0, i32::MAX) as u32;
                    }
                    (st.slot.index, st.pos as u32, j.frames as u32)
                }
                None => (spare.next().ok_or("no spare codec stream slot")?, 1, 0),
            };
            (io.slot_buf[b], io.pos_buf[b], io.n_buf[b]) = (slot, pos, n);
            let rows = cn * io.xf_rpf;
            let row0 = pos * io.xf_rpf as u32;
            for r in 0..rows {
                io.rope_buf[b * rows + r] = row0 + r as u32;
            }
            if let Some(v) = io.row_buf.get_mut(b) {
                *v = row0;
            }
        }
        let e = |x: crate::RuntimeError| x.to_string();
        let rt = &mut self.runtime;
        let mut writes = code_writes(self.codes, &io.codes_buf, jobs, cn, fc);
        writes.extend([
            PacketWrite {
                tensor: io.slot,
                offset: 0,
                data: PacketData::Host(bytemuck::cast_slice(&io.slot_buf)),
            },
            PacketWrite {
                tensor: io.pos,
                offset: 0,
                data: PacketData::Host(bytemuck::cast_slice(&io.pos_buf)),
            },
            PacketWrite {
                tensor: io.rope,
                offset: 0,
                data: PacketData::Host(bytemuck::cast_slice(&io.rope_buf)),
            },
        ]);
        if let Some(t) = io.row_pos {
            writes.push(PacketWrite {
                tensor: t,
                offset: 0,
                data: PacketData::Host(bytemuck::cast_slice(&io.row_buf)),
            });
        }
        for ((&t, &(_, per_frame)), values) in io.n.iter().zip(&self.lengths).zip(&mut io.len_buf) {
            values.fill(0);
            for (l, &n) in values.iter_mut().zip(&io.n_buf[..cb]) {
                *l = n * per_frame as u32;
            }
            writes.push(PacketWrite {
                tensor: t,
                offset: 0,
                data: PacketData::Host(bytemuck::cast_slice(values)),
            });
        }
        rt.write_tensors(&writes).map_err(e)?;
        let t1 = std::time::Instant::now();
        rt.run_sequence(&programs).map_err(e)?;
        let gpu_us = rt.last_run_us();
        let t2 = std::time::Instant::now();
        self.buckets.record(
            crate::obs::bucket::Family::Codec,
            jobs.len(),
            cb,
            Some(gpu_us as u64),
        );
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
        let mut lengths: smallvec::SmallVec<[(PacketTensor, Vec<u32>); 16]> =
            smallvec::SmallVec::new();
        for &(tensor, per_frame) in &self.lengths {
            let mut len = vec![0u32; tensor.bytes / 4];
            for (i, j) in jobs.iter().enumerate() {
                len[i] = (j.frames * per_frame) as u32;
            }
            lengths.push((tensor, len));
        }
        let mut seeds = vec![0u64; self.seed.bytes / 8];
        for (i, j) in jobs.iter().enumerate() {
            seeds[i] = j.seed;
        }
        let mut writes = code_writes(self.codes, &codes, jobs, cf, fc);
        writes.extend(lengths.iter().map(|(tensor, len)| PacketWrite {
            tensor: *tensor,
            offset: 0,
            data: PacketData::Host(bytemuck::cast_slice(len)),
        }));
        writes.push(PacketWrite {
            tensor: self.seed,
            offset: 0,
            data: PacketData::Host(bytemuck::cast_slice(&seeds)),
        });
        self.runtime.write_tensors(&writes).map_err(e)?;
        let slotted = matches!(family, Family::Slot | Family::SlotCached);
        let slots = match (slotted, self.voice_slots.as_mut()) {
            (true, Some(vs)) => {
                let voices: Vec<_> = jobs.iter().filter_map(|j| j.clone.as_ref()).collect();
                Some(vs.assign(self.runtime.as_mut(), &voices)?)
            }
            (true, None) => return Err("codec packet has no voice slots".into()),
            _ => None,
        };
        if family == Family::SlotCached {
            let cs = self
                .cache_slots
                .as_mut()
                .ok_or("codec packet has no slot caches")?;
            let mut c = vec![0u32; cs.cslot.bytes / 4];
            for (i, j) in jobs.iter().enumerate() {
                let v = j.clone.as_ref().ok_or("cached window without a voice")?;
                let k = cs
                    .resident
                    .iter()
                    .position(|r| r.as_ref().is_some_and(|r| std::sync::Arc::ptr_eq(r, v)))
                    .ok_or("cloned voice left its cache slot")?;
                cs.used[k] = std::time::Instant::now();
                c[i] = k as u32;
            }
            self.runtime
                .write_tensor(cs.cslot, bytemuck::cast_slice(&c))
                .map_err(e)?;
        }
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
        let windows = if window || slotted { self.phase } else { None };
        if let Some([phase, _, seam, next_seam]) = windows {
            let mut p = vec![0f32; phase.bytes / 4];
            let (mut s, mut ns) = (vec![0u32; seam.bytes / 4], vec![0u32; next_seam.bytes / 4]);
            for (i, j) in jobs.iter().enumerate() {
                let whole;
                let w = match &j.window {
                    Some(w) => w,
                    None => {
                        whole = Window {
                            seam: u32::MAX,
                            next_seam: 0,
                            phase: source_phase(j.seed, nh),
                        };
                        &whole
                    }
                };
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
        self.buckets
            .record(family.bucket(), jobs.len(), cb, Some(gpu_us as u64));
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
                phase: if j.window.is_some() && windows.is_some() {
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
        .chain(&codec.cslot_caps)
        .map(|c| c.0)
        .max()
        .unwrap_or(1);
    let hold_guard = std::time::Duration::from_millis(
        crate::config::RuntimeConfig::get().tts_codec_hold_ms as u64,
    );
    let cfg = crate::config::RuntimeConfig::get();
    let cap_us = cfg.tts_codec_launch_us as f64;
    let sched = cfg.tts_codec_sched;
    let urgent_slack = std::time::Duration::from_millis(
        codec
            .parameters
            .get("stream.urgent_slack_ms")
            .copied()
            .unwrap_or(URGENT_SLACK_MS),
    );
    let max_wait = crate::serve::cosched::max_wait();
    // Every stream-step launch is timed: `split` prices launches whatever the bound is.
    let learn = true;
    let mut costs = StepCosts::default();
    let mut pending: Vec<Job> = Vec::new();
    // The rest of a split launch: it launches next, whatever arrives meanwhile.
    let mut owed: Vec<Job> = Vec::new();
    let mut hold = None;
    let mut open = true;
    while open || !pending.is_empty() || !owed.is_empty() {
        let got = if !open || !owed.is_empty() || (!pending.is_empty() && hold.is_none()) {
            None
        } else if let Some(wait) = hold {
            match rx.recv_timeout(wait) {
                Ok(m) => Some(m),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    open = false;
                    None
                }
            }
        } else {
            match rx.recv() {
                Ok(m) => Some(m),
                Err(_) => break,
            }
        };
        // A decode tick hands its frames to every stream at once, so their steps arrive as one
        // burst: take the burst as one launch rather than launching on its first job. The wait
        // ends at the first gap, or once every live stream has a step queued.
        let gather_end = std::time::Instant::now() + 4 * STEP_GATHER;
        let mut burst = got;
        loop {
            for m in burst.take().into_iter().chain(rx.try_iter()) {
                match m {
                    Msg::Job(j) => pending.push(j),
                    // The stream is gone and no launch holds its slot: drop its queued step.
                    Msg::Release(id) => {
                        pending.retain(|j| j.step.is_none_or(|s| s.slot != id));
                        owed.retain(|j| j.step.is_none_or(|s| s.slot != id));
                        if let Some(pool) = &codec.slots {
                            pool.lock().release(id);
                        }
                    }
                    Msg::Observe(buckets) => {
                        for family in [
                            Family::Decode,
                            Family::Window,
                            Family::Step,
                            Family::Slot,
                            Family::SlotCached,
                        ] {
                            let caps = codec.capacities_of(family);
                            buckets.declare(family.bucket(), caps.iter().map(|c| c.0));
                        }
                        codec.buckets = buckets;
                    }
                }
            }
            let live = codec.slots.as_ref().map_or(0, |pool| pool.lock().live());
            let left = gather_end.saturating_duration_since(std::time::Instant::now());
            if !open || !owed.is_empty() || left.is_zero() || !gathers(&pending, live) {
                break;
            }
            match rx.recv_timeout(STEP_GATHER.min(left)) {
                Ok(m) => burst = Some(m),
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    open = false;
                    break;
                }
            }
        }
        // Most urgent first; within an urgency longest first, so a launch's capacity is mostly
        // set by its first job. Deadline scheduling ranks by (band, slack) instead.
        if sched {
            let now = std::time::Instant::now();
            let caps = &codec.steps;
            pending.sort_by_cached_key(|j| {
                rank(j, now, urgent_slack, max_wait, job_cost(caps, &costs, j))
            });
        } else {
            pending.sort_by_key(|j| (j.urgency, std::cmp::Reverse(j.frames)));
        }
        codec.place_caches(&mut pending);
        // A failed launch or a committed step can turn a queued job unlaunchable.
        for queue in [&mut pending, &mut owed] {
            let mut k = 0;
            while k < queue.len() {
                match codec.check(&queue[k]) {
                    Ok(()) => k += 1,
                    Err(e) => drop(queue.remove(k).reply.send(Err(e.into()))),
                }
            }
        }
        hold = if open && owed.is_empty() {
            let now = std::time::Instant::now();
            hold_for(&pending, hold_guard, now).and_then(|w| {
                if sched {
                    sched_hold(
                        w,
                        &pending,
                        now,
                        urgent_slack,
                        max_wait,
                        &codec.steps,
                        &costs,
                    )
                } else {
                    Some(w)
                }
            })
        } else {
            None
        };
        if owed.is_empty() && (pending.is_empty() || hold.is_some()) {
            continue;
        }
        let mut batch = std::mem::take(&mut owed);
        if batch.is_empty() {
            let take = pack(&pending, max_batch, |f, b, n| {
                codec.capacity(f, b, n).is_some()
            });
            let mut keep = take.len();
            if pending[take[0]].step.is_some() {
                let family = pending[take[0]].family();
                let frames: Vec<usize> = take.iter().map(|&k| pending[k].frames).collect();
                keep = split(&frames, &costs, |b, n| {
                    codec.capacity(family, b, n).map(|c| (c.0, c.1))
                });
            }
            batch = take.iter().rev().map(|&k| pending.remove(k)).collect();
            batch.reverse();
            owed = batch.split_off(keep);
        }
        // Bounded stream steps: a launch whose measured time exceeds the bound runs the measured
        // capacity under it that decodes the most frames per second; the jobs it leaves out stay
        // queued and the frames it cuts decode as the same steps' next pieces, after anything
        // more urgent that arrived meanwhile.
        let mut rests: Vec<Option<(Codes, usize)>> = Vec::new();
        let now = std::time::Instant::now();
        let waiting = |jobs: &[Job]| {
            launch_bound(cap_us, sched, jobs, &codec.steps, &costs, urgent_slack, now)
        };
        if let Some(mut bound_us) = waiting(&pending).filter(|_| batch[0].step.is_some()) {
            let frames: Vec<usize> = batch.iter().map(|j| j.frames).collect();
            let mut cut = bound_step(&codec.steps, &costs, &frames, bound_us, sched);
            // A job the cut defers waits behind this launch too: its slack bounds it. The bound
            // only tightens, over the batch's finitely many slacks.
            while let Some((keep, _)) = cut {
                match waiting(&batch[keep..]) {
                    Some(t) if t < bound_us => {
                        bound_us = t;
                        cut = bound_step(&codec.steps, &costs, &frames, t, sched).or(cut);
                    }
                    _ => break,
                }
            }
            if let Some((keep, cap)) = cut {
                for j in batch.drain(keep..).rev() {
                    pending.insert(0, j);
                }
                let fc = codec.frame_codes;
                rests = batch
                    .iter_mut()
                    .map(|j| {
                        (j.frames > cap).then(|| {
                            let rest = j.codes.split_off(cap * fc, fc);
                            let n = j.frames - cap;
                            j.frames = cap;
                            (rest, n)
                        })
                    })
                    .collect();
            }
        }
        let t = (learn && batch[0].step.is_some()).then(std::time::Instant::now);
        let out = codec.launch(&batch);
        costs.observe_launch(learn, &codec.steps, &batch, || {
            t.unwrap().elapsed().as_secs_f64() * 1e6
        });
        let rests = drain_rests(&mut rests, batch.len());
        match out {
            Ok(pcm) => {
                for ((mut j, mut p), rest) in batch.into_iter().zip(pcm).zip(rests) {
                    match rest {
                        Some((codes, n)) => {
                            j.carry.extend_from_slice(&p.pcm);
                            if let Some(st) = j.step.as_mut() {
                                st.pos += j.frames;
                            }
                            (j.codes, j.frames) = (codes, n);
                            pending.push(j);
                        }
                        None => {
                            if !j.carry.is_empty() {
                                j.carry.extend_from_slice(&p.pcm);
                                p.pcm = std::mem::take(&mut j.carry);
                            }
                            let _ = j.reply.send(Ok(p));
                        }
                    }
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

/// A launch's fixed cost in padded row-frames, for capacities not yet measured (Mimi on H100:
/// 4.3 ms at one row, 16.2 ms at 128 rows x 1 frame).
const LAUNCH_FLOOR_ROW_FRAMES: f64 = 48.0;

/// How many of a packed launch's steps (`frames`, in launch order) launch now. A launch pads every
/// step to its longest, so it costs about rows x longest: a few long steps would set the frames of
/// many short ones. The prefix that launches is the one whose launch plus one more launch for the
/// rest costs least, by measured capacity cost (unmeasured capacities are priced from the measured
/// ones in proportion to their padded row-frames). `capacity(batch, frames)`: the capacity that
/// launch would run.
fn split(
    frames: &[usize],
    costs: &StepCosts,
    capacity: impl Fn(usize, usize) -> Option<(usize, usize)>,
) -> usize {
    let n = frames.len();
    if frames.iter().all(|&f| f == frames[0]) {
        return n;
    }
    let model = |(b, f): (usize, usize)| LAUNCH_FLOOR_ROW_FRAMES + (b * f) as f64;
    // Microseconds per modelled row-frame, over what has been measured.
    let (us, units) = costs.0.iter().fold((0.0, 0.0), |(us, units), (&cap, &c)| {
        (us + c, units + model(cap))
    });
    let scale = if units > 0.0 { us / units } else { 1.0 };
    let price = |batch: usize, longest: usize| {
        let cap = capacity(batch, longest).unwrap_or((batch, longest));
        costs.get(cap.0, cap.1).unwrap_or(scale * model(cap))
    };
    let mut rest_longest = vec![0; n + 1];
    for k in (0..n).rev() {
        rest_longest[k] = rest_longest[k + 1].max(frames[k]);
    }
    let (mut best, mut best_cost, mut longest) = (n, price(n, rest_longest[0]), 0);
    for k in 1..n {
        longest = longest.max(frames[k - 1]);
        let cost = price(k, longest) + price(n - k, rest_longest[k]);
        if cost < best_cost {
            (best, best_cost) = (k, cost);
        }
    }
    best
}

/// The gap that ends a burst of stream steps ([`gathers`]).
const STEP_GATHER: std::time::Duration = std::time::Duration::from_micros(150);

/// Whether the worker waits a moment for more of a burst: only stream steps are queued and
/// fewer of them than there are live stream slots.
fn gathers(pending: &[Job], live: usize) -> bool {
    !pending.is_empty() && pending.len() < live && pending.iter().all(|j| j.step.is_some())
}

fn drain_rests(
    rests: &mut Vec<Option<(Codes, usize)>>,
    count: usize,
) -> impl Iterator<Item = Option<(Codes, usize)>> + '_ {
    rests
        .drain(..)
        .chain(std::iter::repeat_with(|| None))
        .take(count)
}

/// Measured wall time of each stream-step capacity `(batch, frames)`, µs (moving average).
#[derive(Default)]
struct StepCosts(std::collections::HashMap<(usize, usize), f64>);

impl StepCosts {
    /// `learn`: a fixed cap or the deadline scheduler prices launches; otherwise nothing is timed.
    fn observe_launch(
        &mut self,
        learn: bool,
        caps: &[Capacity],
        jobs: &[Job],
        elapsed: impl FnOnce() -> f64,
    ) {
        if learn && jobs[0].step.is_some() {
            let frames = jobs.iter().map(|j| j.frames).max().unwrap_or(0);
            if let Some(&(cb, cn, _)) = smallest(caps, jobs.len(), frames) {
                self.observe(cb, cn, elapsed());
            }
        }
    }

    fn observe(&mut self, cb: usize, cn: usize, us: f64) {
        let c = self.0.entry((cb, cn)).or_insert(us);
        *c += 0.25 * (us - *c);
    }

    fn get(&self, cb: usize, cn: usize) -> Option<f64> {
        self.0.get(&(cb, cn)).copied()
    }
}

/// `stream.first_audio_target_ms` of the codec packet, else the turn's TTFA target
/// (`PLOW_TURN_TTS_TTFA_MS`).
fn first_audio_target(parameters: &std::collections::BTreeMap<String, u64>) -> std::time::Duration {
    std::time::Duration::from_millis(
        parameters
            .get("stream.first_audio_target_ms")
            .copied()
            .unwrap_or_else(|| crate::config::RuntimeConfig::get().turn_tts_ttfa_ms),
    )
}

/// Default `stream.urgent_slack_ms`: a stream step this close to its playback deadline (after its
/// own launch) outranks every first audio (a stall fails the request; a late first chunk is a
/// number).
const URGENT_SLACK_MS: u64 = 30;

/// Priority bands of the deadline scheduler, most urgent first.
const BAND_URGENT: u8 = 0;
const BAND_FIRST: u8 = 1;
const BAND_STREAM: u8 = 2;
const BAND_BULK: u8 = 3;

/// The measured cost of `j`'s own launch alone (its smallest step capacity), µs; 0 unmeasured.
fn job_cost(caps: &[Capacity], costs: &StepCosts, j: &Job) -> f64 {
    j.step
        .and_then(|_| smallest(caps, 1, j.frames))
        .and_then(|&(cb, cn, _)| costs.get(cb, cn))
        .unwrap_or(0.0)
}

/// Launch order key `(band, slack ns, enqueued)`: first audios in [`BAND_FIRST`] by slack to their
/// target; stream steps in [`BAND_STREAM`] by slack to playback, lifted to [`BAND_URGENT`] within
/// `urgent_slack` of it (after `cost_us`); whole decodes last. A job queued `max_wait` is urgent
/// whatever its deadline, ranked by how long past it (the starvation bound); ties by enqueue time, so pieces of a cut launch
/// keep their place.
fn rank(
    j: &Job,
    now: std::time::Instant,
    urgent_slack: std::time::Duration,
    max_wait: std::time::Duration,
    cost_us: f64,
) -> (u8, i64, std::time::Instant) {
    let ns = |d: Option<std::time::Instant>| match d {
        Some(d) if d >= now => (d - now).as_nanos() as i64,
        Some(d) => -((now - d).as_nanos() as i64),
        None => i64::MAX / 2,
    };
    let slack = ns(j.deadline()).saturating_sub((cost_us * 1e3) as i64);
    let age = now.saturating_duration_since(j.since);
    let aged = age >= max_wait;
    // An aged job's deadline is its starvation bound: the longest overdue first, ahead of any
    // step that only now turns urgent.
    let slack = if aged {
        slack.min(-((age - max_wait).as_nanos() as i64))
    } else {
        slack
    };
    let band = match j.urgency {
        _ if aged => BAND_URGENT,
        Urgency::First => BAND_FIRST,
        // Only a step that can still make its deadline: a late one is a stall already, and
        // late steps ahead of first audios starve every arrival once a server falls behind.
        Urgency::Stream
            if j.due.is_some() && (0..=urgent_slack.as_nanos() as i64).contains(&slack) =>
        {
            BAND_URGENT
        }
        Urgency::Stream => BAND_STREAM,
        Urgency::Whole => BAND_BULK,
    };
    (band, slack, j.since)
}

/// The bound on the next stream-step launch, µs: the fixed cap (`PLOW_TTS_CODEC_LAUNCH_US`, 0 =
/// none) and, under deadline scheduling, the slack of the tightest first audio or urgent step
/// still waiting (`pending`, after the batch was taken) less its own launch, so the launch in
/// front of it ends in time for it. `None`: unbounded.
fn launch_bound(
    cap_us: f64,
    sched: bool,
    pending: &[Job],
    caps: &[Capacity],
    costs: &StepCosts,
    urgent_slack: std::time::Duration,
    now: std::time::Instant,
) -> Option<f64> {
    let mut bound = (cap_us > 0.0).then_some(cap_us);
    if sched {
        for j in pending {
            if j.urgency == Urgency::Whole {
                continue;
            }
            // Past its deadline whatever its (possibly unmeasured) cost: see below.
            let Some(d) = j.deadline().filter(|&d| d > now) else {
                continue;
            };
            let slack = (d - now).as_secs_f64() * 1e6 - job_cost(caps, costs, j);
            // A waiter already past its deadline does not tighten the launch ahead of it: the
            // miss is sunk, and bounding every launch for it collapses the frame rate.
            if slack >= 0.0
                && (j.urgency == Urgency::First || slack <= urgent_slack.as_secs_f64() * 1e6)
            {
                bound = Some(bound.map_or(slack, |b: f64| b.min(slack)));
            }
        }
    }
    bound
}

/// A bounded launch of stream steps of `frames` (launch order): `(jobs kept, frames per job)`
/// on the measured capacity at most `bound_us` that decodes the most frames per µs, when the
/// capacity `frames` would take is measured above the bound. `None` launches as packed: under
/// the bound, not yet measured, or (without `progress`) no measured capacity fits the bound.
/// With `progress`, nothing fitting runs the fastest launch within [`PROGRESS_SPAN`] of the cheapest
/// measured one of these jobs instead: a bound under
/// the smallest legal piece is a miss to record.
fn bound_step(
    caps: &[Capacity],
    costs: &StepCosts,
    frames: &[usize],
    bound_us: f64,
    progress: bool,
) -> Option<(usize, usize)> {
    let longest = frames.iter().copied().max()?;
    let &(cb, cn, _) = smallest(caps, frames.len(), longest)?;
    if costs.get(cb, cn)? <= bound_us {
        return None;
    }
    // (µs, jobs kept, frames per job, frames decoded), priced by the capacity the launch of
    // these pieces selects, not the candidate's.
    let pieces = caps.iter().filter_map(|&(cb, cn, _)| {
        let keep = cb.min(frames.len());
        let longest = frames[..keep].iter().map(|&f| f.min(cn)).max().unwrap_or(0);
        let &(lb, ln, _) = smallest(caps, keep, longest)?;
        let decoded: usize = frames[..keep].iter().map(|&f| f.min(cn)).sum();
        Some((costs.get(lb, ln)?, keep, cn, decoded))
    });
    let fastest = |limit: f64| {
        let mut best: Option<(f64, usize, usize, usize)> = None;
        for (us, keep, cn, decoded) in pieces.clone().filter(|p| p.0 <= limit) {
            let rate = decoded as f64 / us;
            if best.is_none_or(|(r, d, ..)| rate > r || (rate == r && decoded > d)) {
                best = Some((rate, decoded, keep, cn));
            }
        }
        best.map(|(_, _, keep, cap)| (keep, cap))
    };
    fastest(bound_us).or_else(|| {
        let cheapest = pieces.clone().map(|p| p.0).min_by(f64::total_cmp)?;
        progress
            .then(|| fastest(PROGRESS_SPAN * cheapest))
            .flatten()
    })
}

/// The progress piece of a bound nothing fits: the fastest launch within this factor of the
/// cheapest one's time (b1.f1 alone decodes a tenth of what b8.f1 does in 1.3x the time).
const PROGRESS_SPAN: f64 = 2.0;

/// Stream steps that wait together for one launch: fewer than this many pending hold.
const HOLD_BATCH: usize = 8;

/// How long the worker waits for more stream steps before launching `pending`: while every
/// pending job is a stream step with a `due`, there are fewer than [`HOLD_BATCH`], and the
/// soonest is due later than `guard` from `now` (a zero `guard` never holds).
fn hold_for(
    pending: &[Job],
    guard: std::time::Duration,
    now: std::time::Instant,
) -> Option<std::time::Duration> {
    if guard.is_zero() || pending.is_empty() || pending.len() >= HOLD_BATCH {
        return None;
    }
    let mut soonest = None::<std::time::Instant>;
    for j in pending {
        if j.urgency != Urgency::Stream || j.step.is_none() {
            return None;
        }
        let due = j.due?;
        soonest = Some(soonest.map_or(due, |s| s.min(due)));
    }
    soonest?
        .checked_duration_since(now + guard)
        .filter(|w| !w.is_zero())
}

/// Under deadline scheduling a hold of `wait` ends when a held step turns urgent (its own launch
/// counted) or ages past `max_wait`. `None`: launch now.
fn sched_hold(
    wait: std::time::Duration,
    pending: &[Job],
    now: std::time::Instant,
    urgent_slack: std::time::Duration,
    max_wait: std::time::Duration,
    caps: &[Capacity],
    costs: &StepCosts,
) -> Option<std::time::Duration> {
    let mut wait = wait;
    for j in pending {
        let cost = std::time::Duration::from_secs_f64(job_cost(caps, costs, j) * 1e-6);
        let urgent = j.due.map_or(std::time::Duration::MAX, |d| {
            d.saturating_duration_since(now + cost + urgent_slack)
        });
        let ages = (j.since + max_wait).saturating_duration_since(now);
        wait = wait.min(urgent).min(ages);
    }
    (!wait.is_zero()).then_some(wait)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queued_codec() -> (Codec, mpsc::Receiver<Msg>) {
        let (tx, rx) = mpsc::channel();
        (
            Codec {
                tx: Mutex::new(tx),
                yield_hook: Default::default(),
                credit: None,
                first_target: std::time::Duration::from_millis(800),
                max_batch: 2,
                max_frames: 8,
                min_frames: 1,
                frame_codes: 2,
                frame_samples: 3,
                window: 0,
                lookahead: 0,
                voices: Vec::new(),
                max_window: 0,
                window_capacities: Vec::new(),
                parameters: Default::default(),
                step_frames: 2,
                prompt_max: 0,
                slots: Some(std::sync::Arc::new(Mutex::new(SlotPool::new(2)))),
            },
            rx,
        )
    }

    #[test]
    fn stateful_whole_decoding_preserves_history_and_owned_slot_lifetime() {
        let (codec, rx) = queued_codec();
        let slot = codec.acquire_slot().unwrap();
        let worker = std::thread::spawn(move || {
            for (start, n) in [(0, 2), (2, 2), (4, 1)] {
                let Msg::Job(j) = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap()
                else {
                    panic!("expected stateful decode job")
                };
                assert_eq!(j.step.unwrap().pos, start);
                assert_eq!(j.frames, n);
                assert_eq!(j.urgency, Urgency::Whole);
                assert_eq!(
                    j.codes.host,
                    (start as i32 * 2..(start + n) as i32 * 2).collect::<Vec<_>>()
                );
                assert!(j
                    .reply
                    .send(Ok(Decoded {
                        pcm: (start * 3..(start + n) * 3).map(|v| v as f32).collect(),
                        phase: Vec::new()
                    }))
                    .is_ok());
            }
            rx
        });
        let (slot, pcm) = futures::executor::block_on(codec.decode_stateful(
            slot,
            (0..10).collect::<Vec<_>>().into(),
            5,
            Urgency::Whole,
        ))
        .unwrap();
        assert_eq!(slot.pos, 5);
        assert_eq!(pcm, (0..15).map(|v| v as f32).collect::<Vec<_>>());
        let rx = worker.join().unwrap();
        drop(slot);
        assert!(matches!(rx.try_recv().unwrap(), Msg::Release(_)));
    }

    #[test]
    fn stateful_whole_decoding_rejects_bad_pcm_and_releases_cancelled_work() {
        for bad in [Vec::new(), vec![f32::NAN; 3]] {
            let (codec, rx) = queued_codec();
            let slot = codec.acquire_slot().unwrap();
            let id = slot.id;
            let worker = std::thread::spawn(move || {
                let Msg::Job(j) = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap()
                else {
                    panic!("expected decode job")
                };
                assert!(j
                    .reply
                    .send(Ok(Decoded {
                        pcm: bad,
                        phase: Vec::new()
                    }))
                    .is_ok());
                rx
            });
            assert!(futures::executor::block_on(codec.decode_stateful(
                slot,
                vec![1, 2].into(),
                1,
                Urgency::Whole
            ))
            .is_err());
            assert!(codec.slots.as_ref().unwrap().lock().slots[id.index as usize].failed);
            assert!(matches!(
                worker.join().unwrap().try_recv().unwrap(),
                Msg::Release(_)
            ));
        }
        let (codec, rx) = queued_codec();
        let slot = codec.acquire_slot().unwrap();
        let mut future =
            Box::pin(codec.decode_stateful(slot, vec![1, 2].into(), 1, Urgency::Whole));
        let waker = futures::task::noop_waker();
        let mut context = std::task::Context::from_waker(&waker);
        assert!(std::future::Future::poll(future.as_mut(), &mut context).is_pending());
        let Msg::Job(job) = rx.try_recv().unwrap() else {
            panic!("expected decode job")
        };
        drop(future);
        assert!(matches!(rx.try_recv().unwrap(), Msg::Release(_)));
        assert!(job
            .reply
            .send(Ok(Decoded {
                pcm: vec![0.; 3],
                phase: Vec::new()
            }))
            .is_err());
    }

    #[test]
    fn reference_priming_preserves_chunks_positions_sources_and_deadlines() {
        let (codec, rx) = queued_codec();
        let slot = codec.acquire_slot().unwrap();
        let due = Some(std::time::Instant::now());
        let values: Vec<i32> = (0..10).collect();
        let source =
            crate::exec::packet_runtime::PacketBuffer::from(bytemuck::cast_slice(&values).to_vec());
        let mut prefix = Codes::from(values);
        prefix.sources = (0..5)
            .map(|n| Some(source.slice(n * 8..(n + 1) * 8).unwrap()))
            .collect();
        drop(source);
        let worker = std::thread::spawn(move || {
            for (start, n) in [(0, 2), (2, 2), (4, 1)] {
                let Msg::Job(j) = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap()
                else {
                    panic!("expected priming job")
                };
                assert_eq!(j.step.unwrap().pos, start);
                assert_eq!(j.frames, n);
                assert_eq!((j.urgency, j.due), (Urgency::First, due));
                assert_eq!(
                    j.codes.host,
                    (start as i32 * 2..(start + n) as i32 * 2).collect::<Vec<_>>()
                );
                assert_eq!(j.codes.sources.len(), n);
                for (frame, source) in j.codes.sources.iter().enumerate() {
                    let mut values = [0i32; 2];
                    source
                        .as_ref()
                        .unwrap()
                        .read(bytemuck::cast_slice_mut(&mut values))
                        .unwrap();
                    assert_eq!(values, j.codes[frame * 2..frame * 2 + 2]);
                }
                assert!(j
                    .reply
                    .send(Ok(Decoded {
                        pcm: vec![0.; n * 3],
                        phase: Vec::new()
                    }))
                    .is_ok());
            }
            rx
        });
        let slot =
            futures::executor::block_on(codec.prime_codes(slot, prefix, 5, Urgency::First, due))
                .unwrap();
        assert_eq!(slot.pos, 5);
        let rx = worker.join().unwrap();
        assert!(rx.try_recv().is_err());
        drop(slot);
        assert!(matches!(rx.try_recv().unwrap(), Msg::Release(_)));
    }

    #[test]
    fn reference_priming_rejects_invalid_inputs_before_submission() {
        for case in 0..6 {
            let (mut codec, rx) = queued_codec();
            let mut slot = codec.acquire_slot().unwrap();
            let id = slot.id;
            let (codes, frames) = match case {
                0 => (vec![1, 2, 3], 2),
                1 => (vec![], usize::MAX),
                2 => {
                    slot.pos = 1;
                    (vec![1, 2], 1)
                }
                3 => {
                    codec.step_frames = 0;
                    (vec![1, 2], 1)
                }
                4 => {
                    codec.slots.as_ref().unwrap().lock().fail(id);
                    (vec![], 0)
                }
                _ => {
                    codec.slots.as_ref().unwrap().lock().release(id);
                    (vec![], 0)
                }
            };
            assert!(futures::executor::block_on(codec.prime_codes(
                slot,
                codes.into(),
                frames,
                Urgency::First,
                None
            ))
            .is_err());
            assert!(matches!(rx.try_recv().unwrap(), Msg::Release(released) if released == id));
            assert!(rx.try_recv().is_err());
        }
        let (codec, rx) = queued_codec();
        let slot = codec.acquire_slot().unwrap();
        let slot = futures::executor::block_on(codec.prime_codes(
            slot,
            Codes::default(),
            0,
            Urgency::First,
            None,
        ))
        .unwrap();
        assert!(rx.try_recv().is_err());
        assert_eq!(slot.pos, 0);
        drop(slot);
        assert!(matches!(rx.try_recv().unwrap(), Msg::Release(_)));
    }

    #[test]
    fn reference_priming_stops_after_failed_or_invalid_step_output() {
        for failure in 0..3 {
            let (codec, rx) = queued_codec();
            let slot = codec.acquire_slot().unwrap();
            let id = slot.id;
            let worker = std::thread::spawn(move || {
                let Msg::Job(j) = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap()
                else {
                    panic!("expected priming job")
                };
                let result = match failure {
                    0 => Err("injected codec failure".into()),
                    1 => Ok(Decoded {
                        pcm: vec![0.; 5],
                        phase: Vec::new(),
                    }),
                    _ => Ok(Decoded {
                        pcm: vec![f32::NAN; 6],
                        phase: Vec::new(),
                    }),
                };
                assert!(j.reply.send(result).is_ok());
                rx
            });
            assert!(futures::executor::block_on(codec.prime_codes(
                slot,
                vec![1; 10].into(),
                5,
                Urgency::First,
                None
            ))
            .is_err());
            assert!(codec.slots.as_ref().unwrap().lock().check(id, 0).is_err());
            let rx = worker.join().unwrap();
            assert!(matches!(rx.try_recv().unwrap(), Msg::Release(released) if released == id));
            assert!(rx.try_recv().is_err(), "failure submitted another step");
        }
    }

    #[test]
    fn cancelled_reference_priming_releases_slot_and_retains_queued_input() {
        use std::future::Future;
        let (codec, rx) = queued_codec();
        let slot = codec.acquire_slot().unwrap();
        let id = slot.id;
        let source = crate::exec::packet_runtime::PacketBuffer::from(
            bytemuck::cast_slice(&[3i32, 4]).to_vec(),
        );
        let prefix = Codes::frame(vec![3, 4], Some(source));
        let mut pending = Box::pin(codec.prime_codes(slot, prefix, 1, Urgency::First, None));
        let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
        assert!(pending.as_mut().poll(&mut cx).is_pending());
        let Msg::Job(job) = rx.try_recv().unwrap() else {
            panic!("expected queued step")
        };
        drop(pending);
        assert!(matches!(rx.try_recv().unwrap(), Msg::Release(released) if released == id));
        let mut values = [0i32; 2];
        job.codes.sources[0]
            .as_ref()
            .unwrap()
            .read(bytemuck::cast_slice_mut(&mut values))
            .unwrap();
        assert_eq!(values, [3, 4]);
        assert!(job
            .reply
            .send(Ok(Decoded {
                pcm: vec![0.; 3],
                phase: Vec::new()
            }))
            .is_err());
        let mut pool = codec.slots.as_ref().unwrap().lock();
        pool.release(id);
        let reused = pool.acquire().unwrap();
        assert_eq!(reused.index, id.index);
        assert_ne!(reused.gen, id.gen);
        assert!(pool.check(id, 0).is_err());
    }

    #[test]
    #[ignore = "GPU correctness; SPEECH_BATCH_TEST_ASSETS, REF_CODEC_REFERENCE, CODEC_PRIMING_RESULT_JSON, optional CODEC_PCM_GATE=int16_rms; run through gpuq"]
    fn reference_priming_preserves_fresh_audio_at_serving_concurrencies() {
        async fn stream(
            codec: &Codec,
            slots: &mut [Option<StreamSlot>],
            items: &[(String, Vec<i32>, usize)],
            fresh: usize,
            prime: bool,
        ) -> Vec<Vec<f32>> {
            futures::future::join_all(slots.iter_mut().zip(items).map(
                |(owner, (_, codes, frames))| async move {
                    let mut slot = owner.take().unwrap();
                    slot.restart();
                    let fc = codec.frame_codes;
                    if prime {
                        slot = codec
                            .prime_codes(
                                slot,
                                codes[..frames * fc].to_vec().into(),
                                *frames,
                                Urgency::First,
                                None,
                            )
                            .await
                            .unwrap();
                    } else {
                        let mut start = 0;
                        while start < *frames {
                            let n = (*frames - start).min(codec.step_frames);
                            codec
                                .step(
                                    &mut slot,
                                    codes[start * fc..(start + n) * fc].to_vec(),
                                    n,
                                    Urgency::First,
                                    None,
                                )
                                .await
                                .unwrap();
                            start += n;
                        }
                    }
                    assert_eq!(slot.pos, *frames);
                    let mut output = Vec::new();
                    let mut start = 0;
                    for n in [1, 2, fresh.saturating_sub(3)] {
                        if n == 0 {
                            continue;
                        }
                        let s = frames + start;
                        output.extend(
                            codec
                                .step(
                                    &mut slot,
                                    codes[s * fc..(s + n) * fc].to_vec(),
                                    n,
                                    Urgency::First,
                                    None,
                                )
                                .await
                                .unwrap(),
                        );
                        start += n;
                    }
                    assert_eq!(start, fresh);
                    assert_eq!(slot.pos, frames + fresh);
                    *owner = Some(slot);
                    output
                },
            ))
            .await
        }
        let assets = std::path::PathBuf::from(std::env::var("SPEECH_BATCH_TEST_ASSETS").unwrap());
        let reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(std::env::var("REF_CODEC_REFERENCE").unwrap()).unwrap(),
        )
        .unwrap();
        let path = crate::exec::packet_runtime::stage_packet(
            &assets.join("model.pkt"),
            "codec.packet",
            PACKET,
        )
        .unwrap();
        let image = crate::asset::packet_image::PacketImage::open(&path).unwrap();
        let windows: std::collections::BTreeSet<usize> = image
            .blob
            .progs
            .iter()
            .flat_map(|p| &p.insts)
            .filter(|d| d.op == packet::dev::DevOp::AttentionF32 as u16 && d.i[6] & 8 != 0)
            .map(|d| (d.i[6] >> 16) as usize)
            .collect();
        assert!(!windows.is_empty() && !windows.contains(&0));
        let codec = Codec::load_on(&assets, 0).unwrap();
        assert_eq!(
            codec.frame_codes as u64,
            reference["frame_codes"].as_u64().unwrap()
        );
        assert!(codec.step_frames >= 5);
        let fresh = 5;
        let longest = 2 * codec.step_frames + 3;
        assert!(codec.max_frames >= longest + fresh);
        let sources: Vec<_> = reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["frames"].as_u64().unwrap() as usize >= longest + fresh)
            .map(|c| {
                let bytes = std::fs::read(c["codes"].as_str().unwrap()).unwrap();
                assert_eq!(
                    bytes.len(),
                    c["frames"].as_u64().unwrap() as usize * codec.frame_codes * 4
                );
                let codes: Vec<i32> = bytes
                    .chunks_exact(4)
                    .map(|b| {
                        let code = u32::from_le_bytes(b.try_into().unwrap());
                        assert!(code < codec.parameters["codec.codebook"] as u32);
                        code as i32
                    })
                    .collect();
                (c["case"].as_str().unwrap().to_owned(), codes)
            })
            .collect();
        assert!(!sources.is_empty());
        let rows_per_frame = codec
            .parameters
            .get("stream.transformer_rows_per_frame")
            .copied()
            .unwrap_or(1) as usize;
        assert!(rows_per_frame > 0);
        let available = sources
            .iter()
            .map(|(_, codes)| codes.len() / codec.frame_codes)
            .max()
            .unwrap()
            .min(codec.max_frames);
        let mut boundaries = Vec::new();
        for &window in &windows {
            let frames = window.div_ceil(rows_per_frame);
            assert!(frames > fresh && 2 * frames + fresh <= available);
            for center in [frames, 2 * frames] {
                boundaries.extend([center - 1, center, center + 1]);
            }
        }
        boundaries.push(available - fresh);
        boundaries.retain(|&frames| frames + fresh <= available);
        boundaries.sort_unstable();
        boundaries.dedup();
        let mut slots: Vec<_> = (0..32)
            .map(|_| Some(codec.acquire_slot().expect("32 stream slots required")))
            .collect();
        let mut checks = Vec::new();
        let mut outputs = Vec::new();
        let mut owned_outputs = Vec::new();
        let mut all_pass = true;
        // int16_rms: absolute RMS error at or below int16 output rounding noise (Qwen3-TTS, user
        // decision 2026-11-09); else relative L2 below 1e-5.
        let int16_rms = match std::env::var("CODEC_PCM_GATE").as_deref() {
            Ok("int16_rms") => true,
            Ok("relative_l2") | Err(_) => false,
            Ok(other) => panic!("unknown CODEC_PCM_GATE {other}"),
        };
        let rms_limit = 1.0 / 32768.0 / 12f64.sqrt();
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
            for history in [false, true] {
              for width in [1, 8, 32] {
               let cohorts = if history { boundaries.len().div_ceil(width) } else { 1 };
               for cohort in 0..cohorts {
                let items: Vec<_> = (0..width).map(|b| {
                    let frames = if history { boundaries[(cohort * width + b) % boundaries.len()] }
                    else if width == 1 { codec.step_frames + 1 } else {
                        [0, 1, codec.step_frames, codec.step_frames + 1, longest][b % 5]
                    };
                    let eligible: Vec<_> = sources.iter()
                        .filter(|(_, codes)| codes.len() >= (frames + fresh) * codec.frame_codes).collect();
                    let (name, codes) = eligible[(cohort * width + b) % eligible.len()];
                    (name.clone(), codes[..(frames + fresh) * codec.frame_codes].to_vec(), frames)
                }).collect();
                let whole = futures::future::join_all(items.iter().map(|(_, codes, frames)|
                    codec.decode(codes.clone(), frames + fresh, 0, Urgency::Whole))).await;
                let control = stream(&codec, &mut slots[..width], &items, fresh, false).await;
                let actual = stream(&codec, &mut slots[..width], &items, fresh, true).await;
                let owned = futures::future::join_all(slots[..width].iter_mut().zip(&items).map(|(owner, (_, codes, frames))| {
                    let codec = &codec;
                    async move {
                        let mut slot = owner.take().unwrap();
                        slot.restart();
                        let (slot, pcm) = codec.decode_stateful(slot, codes.clone().into(), frames + fresh, Urgency::Whole).await.unwrap();
                        *owner = Some(slot);
                        pcm
                    }
                })).await;
                for (b, ((name, _, frames), whole)) in items.iter().zip(whole).enumerate() {
                    let whole = whole.unwrap();
                    assert_eq!(whole.len(), (frames + fresh) * codec.frame_samples);
                    let expected = &whole[frames * codec.frame_samples..];
                    assert_eq!(actual[b].len(), expected.len());
                    assert_eq!(control[b].len(), expected.len());
                    let exact = actual[b].iter().zip(&control[b]).all(|(a, b)| a.to_bits() == b.to_bits());
                    let finite = actual[b].iter().chain(&control[b]).chain(expected).all(|v| v.is_finite());
                    let error = actual[b].iter().zip(expected).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum::<f64>();
                    let norm = expected.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
                    let relative_l2 = (error / norm.max(1e-30)).sqrt();
                    assert_eq!(owned[b].len(), whole.len());
                    let owned_finite = owned[b].iter().all(|v| v.is_finite());
                    let owned_error = owned[b].iter().zip(&whole).map(|(&a, &b)| (f64::from(a)-f64::from(b)).powi(2)).sum::<f64>();
                    let owned_norm = whole.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
                    let owned_relative = (owned_error / owned_norm.max(1e-30)).sqrt();
                    let rms = (error / expected.len() as f64).sqrt();
                    let owned_rms = (owned_error / whole.len() as f64).sqrt();
                    let close = if int16_rms {
                        rms <= rms_limit && owned_rms <= rms_limit
                    } else {
                        relative_l2 < 1e-5 && owned_relative < 1e-5
                    };
                    let pass = exact && finite && owned_finite && close;
                    all_pass &= pass;
                    let offset = outputs.len();
                    outputs.extend_from_slice(expected);
                    outputs.extend_from_slice(&control[b]);
                    outputs.extend_from_slice(&actual[b]);
                    let owned_offset = owned_outputs.len();
                    owned_outputs.extend_from_slice(&owned[b]);
                    checks.push(serde_json::json!({"suite":if history {"history"} else {"chunking"},
                        "cohort":cohort,"width":width,"item":b,"source":name,"reference_frames":frames,
                        "fresh_frames":fresh,"samples":expected.len(),"stateful_exact":exact,"finite":finite,
                        "whole_relative_l2":relative_l2,"pass":pass,"output_offset_f32":offset,
                        "owned_whole_offset_f32":owned_offset,"owned_whole_samples":owned[b].len(),
                        "owned_whole_relative_l2":owned_relative,"whole_rms":rms,"owned_whole_rms":owned_rms}));
                }
               }
              }
            }
        });
        assert_eq!(
            checks.iter().filter(|c| c["suite"] == "chunking").count(),
            41
        );
        for width in [1, 8, 32] {
            for &frames in &boundaries {
                assert!(checks.iter().any(|c| c["suite"] == "history"
                    && c["width"] == width
                    && c["reference_frames"] == frames));
            }
        }
        // Every reference case decoded whole at its full length, for the replay checker's per-case gate.
        let mut full_outputs = Vec::new();
        let mut full_cases = Vec::new();
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                for c in reference["cases"].as_array().unwrap() {
                    let frames = c["frames"].as_u64().unwrap() as usize;
                    assert!(frames <= codec.max_frames);
                    let codes: Vec<i32> = std::fs::read(c["codes"].as_str().unwrap())
                        .unwrap()
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as i32)
                        .collect();
                    assert_eq!(codes.len(), frames * codec.frame_codes);
                    let pcm = codec
                        .decode(codes, frames, 0, Urgency::Whole)
                        .await
                        .unwrap();
                    assert_eq!(pcm.len(), frames * codec.frame_samples);
                    full_cases.push(serde_json::json!({"case":c["case"],"frames":frames,
                    "samples":pcm.len(),"offset_f32":full_outputs.len()}));
                    full_outputs.extend_from_slice(&pcm);
                }
            });
        let path = std::path::PathBuf::from(std::env::var("CODEC_PRIMING_RESULT_JSON").unwrap());
        let full_path = path.with_extension("full.f32");
        std::fs::write(&full_path, bytemuck::cast_slice(&full_outputs)).unwrap();
        std::fs::write(path.with_extension("f32"), bytemuck::cast_slice(&outputs)).unwrap();
        let owned_path = path.with_extension("owned.f32");
        std::fs::write(&owned_path, bytemuck::cast_slice(&owned_outputs)).unwrap();
        std::fs::write(path, serde_json::to_vec_pretty(&serde_json::json!({
            "scope":"saved reference codes, chunking and attention-history boundaries; stateful priming vs ordinary steps and concatenated decode; no timings",
            "attention_window_rows":windows,"transformer_rows_per_frame":rows_per_frame,
            "boundary_reference_frames":boundaries,"max_available_frames":available,
            "pass":all_pass,"whole_relative_l2_limit":1e-5,
            "pcm_gate":if int16_rms {"int16_rms"} else {"relative_l2"},"int16_rms_limit":rms_limit,"checks":checks,"owned_whole_pcm":owned_path,
            "full_cases":full_cases,"full_pcm":full_path})).unwrap()).unwrap();
        assert!(all_pass, "reference priming changed fresh audio");
    }

    #[test]
    fn retained_history_keeps_frame_alignment_across_windows_and_split_steps() {
        let source = |v: &[u32]| {
            crate::exec::packet_runtime::PacketBuffer::from(bytemuck::cast_slice(v).to_vec())
        };
        let mut history = Codes::from(vec![9, 10]);
        history.extend(Codes::frame(vec![1, 2], Some(source(&[1, 2]))), 2);
        history.extend(vec![3, 4].into(), 2);
        history.extend(Codes::frame(vec![5, 6], Some(source(&[5, 6]))), 2);
        assert_eq!(
            history
                .sources
                .iter()
                .map(Option::is_some)
                .collect::<Vec<_>>(),
            [false, true, false, true]
        );
        let mut window = history.slice(2..8, 2);
        drop(history);
        let mut rest = window.split_off(2, 2);
        assert_eq!(window.host, [1, 2]);
        assert_eq!(rest.host, [3, 4, 5, 6]);
        assert!(rest.sources[0].is_none());
        for (codes, frame) in [(&window, 0), (&rest, 1)] {
            let mut values = [0u32; 2];
            codes.sources[frame]
                .as_ref()
                .unwrap()
                .read(bytemuck::cast_slice_mut(&mut values))
                .unwrap();
            assert_eq!(
                bytemuck::cast_slice::<u32, i32>(&values),
                &codes[frame * 2..(frame + 1) * 2]
            );
        }
        rest.release(0..2);
        assert!(rest.sources.iter().all(Option::is_none));
        assert_eq!(rest.host, [3, 4, 5, 6]);
        drop(window);
        drop(rest);
    }

    #[test]
    fn retained_code_rows_preserve_padding_and_signed_code_clamping() {
        let source = |v: &[u32]| {
            crate::exec::packet_runtime::PacketBuffer::from(bytemuck::cast_slice(v).to_vec())
        };
        let mut a = job(false, 2, None);
        a.codes = Codes::frame(vec![1, 2], Some(source(&[1, 2])));
        a.codes
            .extend(Codes::frame(vec![-1, 4], Some(source(&[u32::MAX, 4]))), 2);
        let mut b = job(false, 1, None);
        b.codes = vec![5, 6].into();
        let mut jobs = [a, b];
        let dense = [1u32, 2, 0, 4, 0, 0, 5, 6, 0, 0, 0, 0];
        let tensor = PacketTensor {
            handle: 0,
            bytes: std::mem::size_of_val(&dense),
        };
        let writes = code_writes(tensor, &dense, &jobs, 3, 2);
        assert!(matches!(writes[0].data, PacketData::Zero(48)));
        assert!(matches!(writes[1].data, PacketData::Buffer(_)));
        assert!(matches!(writes.last().unwrap().data, PacketData::Host(_)));
        let mut actual = vec![0xa5; tensor.bytes];
        for w in writes {
            w.check().unwrap();
            w.data
                .read(&mut actual[w.offset..w.offset + w.data.len()])
                .unwrap();
        }
        assert_eq!(actual, bytemuck::cast_slice::<u32, u8>(&dense));
        jobs[0].codes.release(0..2);
        let writes = code_writes(tensor, &dense, &jobs, 3, 2);
        assert_eq!(writes.len(), 1);
        assert!(matches!(writes[0].data, PacketData::Host(_)));
    }

    impl Job {
        fn clone_for_test(&self) -> Job {
            let mut j = job(
                self.window.is_some(),
                self.frames,
                self.step.map(|s| s.slot.index),
            );
            j.urgency = self.urgency;
            j.due = self.due;
            j.deadline = self.deadline;
            j.since = self.since;
            j
        }
    }

    fn job(window: bool, frames: usize, slot: Option<u32>) -> Job {
        Job {
            codes: Codes::default(),
            frames,
            seed: 0,
            voice: 0,
            urgency: Urgency::Stream,
            due: None,
            window: window.then(Window::default),
            step: slot.map(|index| Step {
                slot: SlotId { index, gen: 0 },
                pos: 0,
            }),
            clone: None,
            cached: false,
            reply: tokio::sync::oneshot::channel().0,
            _work: None,
            _urgent: None,
            carry: Vec::new(),
            since: std::time::Instant::now(),
            deadline: None,
        }
    }

    #[cfg(feature = "cuda")]
    struct SerialWrites(Box<dyn crate::exec::packet_runtime::PacketRuntime>);

    #[cfg(feature = "cuda")]
    struct SharedRuntime(
        std::sync::Arc<Mutex<Box<dyn crate::exec::packet_runtime::PacketRuntime>>>,
    );

    #[cfg(feature = "cuda")]
    impl crate::exec::packet_runtime::PacketRuntime for SharedRuntime {
        fn tensor(&self, name: &str) -> Option<PacketTensor> {
            self.0.lock().tensor(name)
        }
        fn write_tensor(&mut self, t: PacketTensor, bytes: &[u8]) -> crate::Result<()> {
            self.0.lock().write_tensor(t, bytes)
        }
        fn write_tensors(&mut self, writes: &[PacketWrite<'_>]) -> crate::Result<()> {
            self.0.lock().write_tensors(writes)
        }
        fn write_tensor_at(
            &mut self,
            t: PacketTensor,
            at: usize,
            bytes: &[u8],
        ) -> crate::Result<()> {
            self.0.lock().write_tensor_at(t, at, bytes)
        }
        fn read_tensor(&self, t: PacketTensor, bytes: &mut [u8]) -> crate::Result<()> {
            self.0.lock().read_tensor(t, bytes)
        }
        fn read_tensor_at(
            &self,
            t: PacketTensor,
            at: usize,
            bytes: &mut [u8],
        ) -> crate::Result<()> {
            self.0.lock().read_tensor_at(t, at, bytes)
        }
        fn snapshot_tensor(
            &mut self,
            t: PacketTensor,
            at: usize,
            bytes: usize,
        ) -> crate::Result<crate::exec::packet_runtime::PacketBuffer> {
            self.0.lock().snapshot_tensor(t, at, bytes)
        }
        fn write_tensor_from(
            &mut self,
            t: PacketTensor,
            at: usize,
            source: &crate::exec::packet_runtime::PacketBuffer,
        ) -> crate::Result<()> {
            self.0.lock().write_tensor_from(t, at, source)
        }
        fn copy_tensor(
            &mut self,
            s: PacketTensor,
            so: usize,
            t: PacketTensor,
            to: usize,
            bytes: usize,
        ) -> crate::Result<()> {
            self.0.lock().copy_tensor(s, so, t, to, bytes)
        }
        fn create_tensor(&mut self, bytes: usize) -> crate::Result<PacketTensor> {
            self.0.lock().create_tensor(bytes)
        }
        fn run(&mut self, p: usize) -> crate::Result<()> {
            self.0.lock().run(p)
        }
        fn run_sequence(&mut self, p: &[usize]) -> crate::Result<()> {
            self.0.lock().run_sequence(p)
        }
        fn last_run_us(&self) -> f64 {
            self.0.lock().last_run_us()
        }
    }

    #[cfg(feature = "cuda")]
    impl crate::exec::packet_runtime::PacketRuntime for SerialWrites {
        fn tensor(&self, name: &str) -> Option<PacketTensor> {
            self.0.tensor(name)
        }
        fn write_tensor(&mut self, t: PacketTensor, bytes: &[u8]) -> crate::Result<()> {
            self.0.write_tensor(t, bytes)
        }
        fn write_tensors(&mut self, writes: &[PacketWrite<'_>]) -> crate::Result<()> {
            for w in writes {
                w.check()?;
            }
            for w in writes {
                let PacketData::Host(bytes) = w.data else {
                    panic!("serial control expects host writes");
                };
                if w.offset == 0 && bytes.len() == w.tensor.bytes {
                    self.0.write_tensor(w.tensor, bytes)?;
                } else {
                    self.0.write_tensor_at(w.tensor, w.offset, bytes)?;
                }
            }
            Ok(())
        }
        fn write_tensor_at(
            &mut self,
            t: PacketTensor,
            at: usize,
            bytes: &[u8],
        ) -> crate::Result<()> {
            self.0.write_tensor_at(t, at, bytes)
        }
        fn read_tensor(&self, t: PacketTensor, bytes: &mut [u8]) -> crate::Result<()> {
            self.0.read_tensor(t, bytes)
        }
        fn read_tensor_at(
            &self,
            t: PacketTensor,
            at: usize,
            bytes: &mut [u8],
        ) -> crate::Result<()> {
            self.0.read_tensor_at(t, at, bytes)
        }
        fn copy_tensor(
            &mut self,
            s: PacketTensor,
            so: usize,
            t: PacketTensor,
            to: usize,
            n: usize,
        ) -> crate::Result<()> {
            self.0.copy_tensor(s, so, t, to, n)
        }
        fn run(&mut self, p: usize) -> crate::Result<()> {
            self.0.run(p)
        }
        fn run_sequence(&mut self, p: &[usize]) -> crate::Result<()> {
            self.0.run_sequence(p)
        }
        fn last_run_us(&self) -> f64 {
            self.0.last_run_us()
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "GPU correctness; SPEECH_BATCH_TEST_ASSETS, RUST_MIN_STACK=67108864; run through gpuq"]
    fn codec_bulk_inputs_preserve_pcm_for_decode_and_stream_steps() {
        check_codec_input_paths(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "GPU correctness; SPEECH_BATCH_TEST_ASSETS, RUST_MIN_STACK=67108864; run through gpuq"]
    fn codec_owned_inputs_preserve_pcm_after_producer_overwrite() {
        check_codec_input_paths(true);
    }

    #[cfg(feature = "cuda")]
    fn check_codec_input_paths(device: bool) {
        let assets = std::path::PathBuf::from(std::env::var("SPEECH_BATCH_TEST_ASSETS").unwrap());
        let path = assets.join("codec.pkt");
        let mut control = bind(&path, 0).unwrap();
        let mut treatment = bind(&path, 0).unwrap();
        let copies = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut producer = None;
        if device {
            let runtime = std::sync::Arc::new(Mutex::new(control.runtime));
            let staging = runtime.lock().create_tensor(treatment.codes.bytes).unwrap();
            control.runtime = Box::new(SharedRuntime(runtime.clone()));
            producer = Some((runtime, staging));
        }
        control.runtime = Box::new(SerialWrites(control.runtime));
        let fc = control.frame_codes;
        let own = |jobs: &mut [Job]| {
            let Some((runtime, staging)) = &producer else {
                return;
            };
            let mut runtime = runtime.lock();
            for (b, j) in jobs.iter_mut().enumerate() {
                if b % 3 == 1 {
                    continue;
                }
                let mut bytes = bytemuck::cast_slice::<i32, u8>(&j.codes.host).to_vec();
                runtime.write_tensor_at(*staging, 0, &bytes).unwrap();
                let snapshot = runtime.snapshot_tensor(*staging, 0, bytes.len()).unwrap();
                assert!(snapshot.cuda_source().is_some());
                bytes.fill(0xdd);
                runtime.write_tensor_at(*staging, 0, &bytes).unwrap();
                j.codes.sources = (0..j.frames)
                    .map(|f| {
                        (f % 2 == 0).then(|| snapshot.slice(f * fc * 4..(f + 1) * fc * 4).unwrap())
                    })
                    .collect();
                copies.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        };
        let method = if device { "owned" } else { "bulk" };
        let maximum = control.capacities.iter().map(|c| c.0).max().unwrap();
        let equal = |a: Vec<Decoded>, b: Vec<Decoded>| {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(&b) {
                assert!(a.pcm.iter().chain(&b.pcm).all(|v| v.is_finite()));
                assert_eq!(a.pcm.len(), b.pcm.len());
                assert!(
                    a.pcm
                        .iter()
                        .zip(&b.pcm)
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "bulk writes changed decoded PCM"
                );
            }
        };
        for width in [1, 8, 32].into_iter().filter(|&n| n <= maximum) {
            for frames in [1, 2, 5] {
                let mut jobs: Vec<_> = (0..width)
                    .map(|b| {
                        let mut j = job(false, frames, None);
                        j.seed = 77 + b as u64;
                        j.codes = (0..frames * control.frame_codes)
                            .map(|i| ((i * 37 + b * 17) % 512) as i32)
                            .collect();
                        j
                    })
                    .collect();
                let expected = control.decode(&jobs).unwrap();
                own(&mut jobs);
                equal(expected, treatment.decode(&jobs).unwrap());
                eprintln!("codec {method} inputs: decode width={width} frames={frames}, PCM exact");
            }
        }
        let maximum = control.steps.iter().map(|c| c.0).max().unwrap_or(0);
        for width in [1, 8, 32].into_iter().filter(|&n| n <= maximum) {
            let mut position = 0;
            for frames in [1, 2, 5] {
                let mut jobs: Vec<_> = (0..width)
                    .map(|b| {
                        let mut j = job(false, frames, Some(b as u32));
                        j.step.as_mut().unwrap().pos = position;
                        j.codes = (0..frames * control.frame_codes)
                            .map(|i| ((i * 37 + b * 17 + position * 13) % 512) as i32)
                            .collect();
                        j
                    })
                    .collect();
                let expected = control.step(&jobs).unwrap();
                own(&mut jobs);
                equal(expected, treatment.step(&jobs).unwrap());
                eprintln!("codec {method} inputs: stream width={width} position={position} frames={frames}, PCM exact");
                position += frames;
            }
        }
        assert!(treatment.runtime.last_run_us().is_finite());
        assert_eq!(
            copies.load(std::sync::atomic::Ordering::Relaxed) > 0,
            device
        );
    }

    #[test]
    fn disabled_launch_bound_keeps_rests_unallocated_and_learns_costs() {
        let caps = vec![(1, 4, Vec::new()), (8, 4, Vec::new())];
        let mut costs = StepCosts::default();
        let mut rests = Vec::new();
        for width in [1, 8, 1] {
            let jobs: Vec<_> = (0..width).map(|n| job(false, 4, Some(n))).collect();
            assert_eq!(
                drain_rests(&mut rests, jobs.len())
                    .filter(Option::is_none)
                    .count(),
                jobs.len()
            );
            assert_eq!(
                rests.capacity(),
                0,
                "disabled launch allocated remainder storage"
            );
            // Nothing is timed without a cap or the scheduler; either prices every launch.
            costs.observe_launch(false, &caps, &jobs, || {
                panic!("disabled launch read timing")
            });
            costs.observe_launch(true, &caps, &jobs, || 1_000.0 * width as f64);
        }
        assert_eq!(costs.get(1, 4), Some(1_000.0));
        assert_eq!(costs.get(8, 4), Some(8_000.0));
    }

    #[test]
    fn bounded_launch_keeps_remainders_and_updates_costs() {
        let caps = vec![(1, 4, Vec::new()), (2, 8, Vec::new())];
        let jobs = [job(false, 3, Some(0)), job(false, 7, Some(1))];
        let mut costs = StepCosts::default();
        costs.observe_launch(true, &caps, &jobs, || 20.0);
        costs.observe_launch(true, &caps, &jobs, || 40.0);
        assert_eq!(costs.get(2, 8), Some(25.0));
        let mut rests = vec![Some((vec![11, 12].into(), 1)), None];
        let mut drained = drain_rests(&mut rests, jobs.len());
        let (codes, frames) = drained.next().unwrap().unwrap();
        assert_eq!(codes.host, [11, 12]);
        assert_eq!(frames, 1);
        assert!(drained.next().unwrap().is_none());
        assert!(drained.next().is_none());
        drop(drained);
        assert!(rests.is_empty());
        costs.observe_launch(true, &caps, &[job(false, 3, None)], || {
            panic!("non-step timing")
        });
        assert_eq!(costs.get(1, 4), None);
    }

    #[test]
    fn bounded_steps_take_the_fastest_measured_capacity_under_the_bound() {
        let caps: Vec<Capacity> = [(1, 1), (1, 17), (4, 4), (4, 17), (8, 1), (8, 17)]
            .into_iter()
            .map(|(b, f)| (b, f, Vec::new()))
            .collect();
        let mut costs = StepCosts::default();
        let frames = [17; 8];
        // Unmeasured: launch as packed.
        assert_eq!(bound_step(&caps, &costs, &frames, 17_500.0, false), None);
        for (b, f, us) in [
            (1, 1, 4000.0),
            (1, 17, 7400.0),
            (4, 4, 7500.0),
            (4, 17, 16900.0),
        ] {
            costs.observe(b, f, us);
        }
        costs.observe(8, 17, 29900.0);
        // b4.f17: 68 frames in 16.9 ms beats b1.f17 (17 in 7.4) and b4.f4 (16 in 7.5).
        assert_eq!(
            bound_step(&caps, &costs, &frames, 17_500.0, false),
            Some((4, 17))
        );
        assert_eq!(
            bound_step(&caps, &costs, &frames, 8_000.0, false),
            Some((1, 17))
        );
        // Under the bound, or nothing measured fits it: as packed.
        assert_eq!(bound_step(&caps, &costs, &frames, 30_000.0, false), None);
        assert_eq!(bound_step(&caps, &costs, &frames, 3_000.0, false), None);
        assert_eq!(bound_step(&caps, &costs, &[2], 3_000.0, false), None);
        // A cheap wider capacity the cut launch would not select does not qualify it.
        let caps: Vec<Capacity> = vec![(4, 2, Vec::new()), (16, 2, Vec::new())];
        let mut costs = StepCosts::default();
        costs.observe(4, 2, 20_000.0);
        costs.observe(16, 2, 5_000.0);
        assert_eq!(bound_step(&caps, &costs, &[2, 2], 10_000.0, false), None);
        // Progress: nothing measured fits the bound -> the cheapest measured piece, never the
        // whole cohort; unmeasured still launches as packed (learning).
        assert_eq!(
            bound_step(&caps, &costs, &[2, 2], 10_000.0, true),
            Some((2, 2))
        );
        assert_eq!(
            bound_step(&caps, &costs, &[2; 16], 1_000.0, true),
            Some((16, 2))
        );
        assert_eq!(
            bound_step(&caps, &StepCosts::default(), &[2, 2], 10.0, true),
            None
        );
        // The fallback is priced by the capacity its launch selects: two 2-frame jobs under a
        // 1 ms bound run b1.f1 (6 ms), not b2.f2 on the 20 ms b4 capacity; and it takes the
        // fastest piece near the cheapest, not the cheapest (b8.f1 over b1.f1).
        let caps: Vec<Capacity> = [(1, 1), (4, 2), (16, 2)]
            .into_iter()
            .map(|(b, f)| (b, f, Vec::new()))
            .collect();
        let mut costs = StepCosts::default();
        costs.observe(1, 1, 6_000.0);
        costs.observe(4, 2, 20_000.0);
        costs.observe(16, 2, 5_000.0);
        assert_eq!(
            bound_step(&caps, &costs, &[2, 2], 1_000.0, true),
            Some((1, 1))
        );
        let caps: Vec<Capacity> = [(1, 1), (8, 1), (8, 17)]
            .into_iter()
            .map(|(b, f)| (b, f, Vec::new()))
            .collect();
        let mut costs = StepCosts::default();
        costs.observe(1, 1, 4_700.0);
        costs.observe(8, 1, 6_100.0);
        costs.observe(8, 17, 23_700.0);
        assert_eq!(bound_step(&caps, &costs, &[17; 8], 0.0, true), Some((8, 1)));
    }

    #[test]
    fn deadline_rank_orders_first_audio_urgent_steps_and_ages() {
        use std::time::{Duration, Instant};
        let now = Instant::now();
        let ms = |v: u64| Duration::from_millis(v);
        let slack = ms(30);
        let wait = ms(2000);
        let mk = |u: Urgency, deadline: Option<u64>, due: Option<u64>, since_ms: u64| {
            let mut j = job(false, 4, Some(1));
            j.urgency = u;
            j.deadline = deadline.map(|d| now + ms(d));
            j.due = due.map(|d| now + ms(d));
            j.since = now - ms(since_ms);
            j
        };
        let first = mk(Urgency::First, Some(25), None, 3);
        let later_first = mk(Urgency::First, Some(40), None, 1);
        let stream = mk(Urgency::Stream, None, Some(400), 10);
        let near = mk(Urgency::Stream, None, Some(20), 10);
        let whole = mk(Urgency::Whole, None, None, 500);
        let old = mk(Urgency::Stream, None, Some(900), 2500);
        let r = |j: &Job, cost: f64| rank(j, now, slack, wait, cost);
        // Bands: an about-to-underrun step and a starved job before any first audio, first audios
        // before streams, whole decodes last.
        assert!(r(&near, 0.0) < r(&first, 0.0));
        assert!(r(&old, 0.0).0 == BAND_URGENT && r(&old, 0.0) < r(&first, 0.0));
        assert!(r(&first, 0.0) < r(&later_first, 0.0));
        assert!(r(&later_first, 0.0) < r(&stream, 0.0));
        assert!(r(&stream, 0.0) < r(&whole, 0.0));
        // A step's own launch cost counts against its slack: 400 ms due less a 380 ms launch is urgent.
        assert_eq!(r(&stream, 380_000.0).0, BAND_URGENT);
        assert_eq!(r(&stream, 100_000.0).0, BAND_STREAM);
        // Pieces keep their place: same band and slack, the earlier enqueue first.
        let (a, b) = (
            mk(Urgency::Stream, None, Some(400), 50),
            mk(Urgency::Stream, None, Some(400), 5),
        );
        assert!(r(&a, 0.0) < r(&b, 0.0));

        // The launch bound: the fixed cap, tightened by a waiting first audio's slack less its
        // own launch; a far-off stream step does not tighten it; nothing waiting keeps the cap.
        let caps: Vec<Capacity> = vec![(1, 4, Vec::new()), (8, 17, Vec::new())];
        let mut costs = StepCosts::default();
        costs.observe(1, 4, 4_000.0);
        let b = |cap: f64, sched: bool, pending: &[Job]| {
            launch_bound(cap, sched, pending, &caps, &costs, slack, now)
        };
        assert_eq!(b(0.0, false, &[first.clone_for_test()]), None);
        assert_eq!(
            b(10_000.0, false, &[first.clone_for_test()]),
            Some(10_000.0)
        );
        assert_eq!(b(0.0, true, &[stream.clone_for_test()]), None);
        let fb = b(0.0, true, &[first.clone_for_test()]).unwrap();
        assert!((fb - 21_000.0).abs() < 1_500.0, "{fb}");
        assert!(b(10_000.0, true, &[first.clone_for_test()]).unwrap() <= 10_000.0);
        // An urgent step tightens it by its due less its own measured launch (b1.f4, 4 ms).
        assert_eq!(
            b(0.0, true, &[near.clone_for_test()]).map(|v| v.round()),
            Some(16_000.0)
        );
        // The packet's urgent threshold is the bound's too.
        assert_eq!(
            launch_bound(
                0.0,
                true,
                &[stream.clone_for_test()],
                &caps,
                &costs,
                ms(500),
                now
            )
            .map(|v| v.round()),
            Some(396_000.0)
        );
        // Aged jobs rank by how long past the starvation bound, ahead of a step turning urgent:
        // sustained urgent streams cannot keep an aged whole decode or window waiting.
        let old_whole = mk(Urgency::Whole, None, None, 2600);
        let old_window = mk(Urgency::Stream, None, None, 2300);
        assert!(r(&old_whole, 0.0) < r(&old, 0.0));
        assert!(r(&old, 0.0) < r(&old_window, 0.0));
        let overdue = mk(Urgency::Stream, None, Some(0), 10);
        assert!(r(&old_window, 0.0) < r(&overdue, 0.0));
        // A hold ends when a held step turns urgent after its own launch, or ages.
        let h = |j: &Job| {
            sched_hold(
                ms(70),
                std::slice::from_ref(j),
                now,
                slack,
                wait,
                &caps,
                &costs,
            )
        };
        assert_eq!(h(&mk(Urgency::Stream, None, Some(400), 10)), Some(ms(70)));
        assert_eq!(h(&mk(Urgency::Stream, None, Some(84), 10)), Some(ms(50)));
        assert_eq!(h(&mk(Urgency::Stream, None, Some(30), 10)), None);
        assert_eq!(h(&mk(Urgency::Stream, None, Some(900), 1990)), Some(ms(10)));
        // A deadline already passed does not tighten the bound (the miss is sunk); the cap stays.
        let late = mk(Urgency::First, Some(0), None, 100);
        assert_eq!(b(0.0, true, &[late.clone_for_test()]), None);
        // Also with nothing measured (cost 0): expiry is checked apart from the estimate.
        assert_eq!(
            launch_bound(
                0.0,
                true,
                &[late.clone_for_test()],
                &caps,
                &StepCosts::default(),
                slack,
                now
            ),
            None
        );
        assert_eq!(b(10_000.0, true, &[late]), Some(10_000.0));
        // A late step is no longer urgent: it leads the stream band, behind first audios.
        let stalled = mk(Urgency::Stream, None, Some(2), 10);
        assert_eq!(r(&stalled, 4_000.0).0, BAND_STREAM);
        assert!(r(&first, 0.0) < r(&stalled, 4_000.0));
        assert!(r(&stalled, 4_000.0) < r(&stream, 0.0));
        assert_eq!(b(0.0, true, &[stalled]), None);
    }

    #[test]
    fn source_phase_matches_the_synth_draw() {
        // scripts/tts/s3gen_packet_check.py runiform(111, 2, h, 0), (u - 0.5) * tau in f32.
        let bits: Vec<u32> = source_phase(111, 9).iter().map(|x| x.to_bits()).collect();
        assert_eq!(
            bits,
            [
                0x0, 0xbffbbefa, 0x3e1a7124, 0xbf9c05b2, 0xc006eb65, 0x3fbae945, 0xbffb37cd,
                0xc0259018, 0x3ec18d7f
            ]
        );
    }

    #[test]
    fn launches_split_where_padding_costs_more_than_a_second_launch() {
        let caps = [
            (1, 1),
            (1, 2),
            (1, 17),
            (4, 2),
            (128, 1),
            (128, 2),
            (128, 17),
        ];
        let cap = |b: usize, n: usize| caps.iter().copied().find(|&(cb, cn)| cb >= b && cn >= n);
        let mut costs = StepCosts::default();
        let mixed = |long: usize, long_frames: usize, short: usize| {
            let mut f = vec![long_frames; long];
            f.extend(std::iter::repeat_n(1, short));
            f
        };
        // Unmeasured: priced by padded row-frames.
        assert_eq!(split(&mixed(1, 2, 127), &costs, cap), 1);
        assert_eq!(split(&mixed(1, 17, 100), &costs, cap), 1);
        assert_eq!(split(&mixed(126, 2, 2), &costs, cap), 128);
        assert_eq!(split(&[1; 128], &costs, cap), 128);
        assert_eq!(split(&[2], &costs, cap), 1);
        // Measured costs decide: here two frames cost no more than one, so nothing splits.
        for (b, n, us) in [(1, 2, 4_500.0), (128, 1, 16_000.0), (128, 2, 17_000.0)] {
            costs.observe(b, n, us);
        }
        assert_eq!(split(&mixed(1, 2, 127), &costs, cap), 128);
        costs = StepCosts::default();
        for (b, n, us) in [(1, 2, 4_500.0), (128, 1, 16_000.0), (128, 2, 33_000.0)] {
            costs.observe(b, n, us);
        }
        assert_eq!(split(&mixed(1, 2, 127), &costs, cap), 1);
    }

    #[test]
    fn step_bursts_gather_until_every_live_stream_is_queued() {
        let step = || job(false, 1, Some(0));
        assert!(!gathers(&[], 4));
        assert!(!gathers(&[step()], 1), "a lone stream launches at once");
        assert!(gathers(&[step()], 2));
        assert!(!gathers(&[step(), step()], 2));
        assert!(
            !gathers(&[step(), job(false, 1, None)], 4),
            "decodes do not wait"
        );
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

    #[test]
    fn first_audio_launches_without_longer_steps_and_later_steps_hold() {
        let mut first = job(false, 2, Some(1));
        first.urgency = Urgency::First;
        let pending = [first, job(false, 17, Some(2)), job(false, 2, Some(3))];
        assert_eq!(pack(&pending, 32, |_, _, _| true), vec![0, 2]);

        let now = std::time::Instant::now();
        let ms = std::time::Duration::from_millis;
        let step = |due_ms: Option<u64>| {
            let mut j = job(false, 17, Some(4));
            j.due = due_ms.map(|d| now + ms(d));
            j
        };
        let g = ms(300);
        assert_eq!(
            hold_for(&[step(Some(1000)), step(Some(700))], g, now),
            Some(ms(400))
        );
        assert_eq!(hold_for(&[step(Some(1000)), step(Some(200))], g, now), None);
        assert_eq!(hold_for(&[step(Some(1000)), step(None)], g, now), None);
        assert_eq!(hold_for(&[step(Some(1000))], ms(0), now), None);
        let full: Vec<Job> = (0..HOLD_BATCH).map(|_| step(Some(1000))).collect();
        assert_eq!(hold_for(&full, g, now), None);
        let mut w = step(Some(1000));
        w.urgency = Urgency::First;
        assert_eq!(hold_for(&[w], g, now), None);
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
