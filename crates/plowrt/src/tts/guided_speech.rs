//! Guided speech on `plowrt serve`: a guided token LM (`tts.guided_lm.v1`, served by the model's
//! continuous-batching mux as CFG slot pairs) feeding a token vocoder (`s3gen.pkt`, a `codec.v1`
//! packet on the packet runtime). Each request is a mux job whose prefill rows are host
//! embeddings; its speech tokens are forwarded, as they are committed, to one render thread that
//! renders batches of utterances, so decoding never waits on audio rendering.
//!
//! Streaming (schedule from the vocoder packet's `stream.*` parameters): the vocoder is not causal
//! over tokens, so every `chunk` tokens a stream renders a window and emits the audio of all but
//! its last `hold` tokens, crossfading `fade` samples into the previous render's tail. With the
//! packet's cached capacities a window is the new tokens plus `stream.context_tokens` of left
//! context (the voice prompt's attention K/V are cached), and its source continues the stream's
//! NSF phase at the seam; without them a stream re-renders its whole token prefix (noise keyed by
//! frame, so re-renders of a prefix agree up to that lookahead). A render sharing the GPU with the LM's back-to-back cooperative decode launches runs
//! several times slower, so at low load the LM pauses while a batch holding a first chunk renders:
//! first audio is then prefill + `first` tokens + one uncontended render. The pause is the mux's
//! downstream urgency ([`DownstreamCredit::set_urgent`]). Whole utterances are rendered in batches
//! (`render.min_batch` / `render.hold_ms`): a lone small render costs several times more GPU per
//! utterance, taken from the LM.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

use super::codec::{Codec, Decoded, Window};
use super::guided_lm::{GuidedLmContract, PromptTables};
use crate::sched::admission::DownstreamCredit;
use crate::serve::mux::{CfgJob, Job, JobClass, JobOpts, ModelMux, SpeechJob, SubmitError};
use crate::serve::stream::StreamChunk;
use crate::{Result, RuntimeError};

pub const QUEUE_FULL: &str = "speech request queue full";

/// The vocoder packet beside the LM packet.
pub const VOCODER: &str = "s3gen.pkt";

#[derive(Clone, Copy, Debug)]
struct Schedule {
    first: usize,
    chunk: usize,
    hold: usize,
    fade: usize,
    samples_per_token: usize,
    max_tokens: usize,
    /// Batch forming: a render of fewer than `min_batch` utterances (no first chunk among them)
    /// waits up to `batch_hold` for more while others are still being generated. Small vocoder
    /// batches cost several times more GPU per utterance, and that time comes out of the LM's.
    min_batch: usize,
    batch_hold: std::time::Duration,
    /// Final tokens whose audio is cut from the utterance (`lm.trim_tail_tokens`).
    trim_tail: usize,
    /// Streams render windows on the cached capacities: left-context tokens and the largest
    /// window (0 = prefix re-renders).
    context: usize,
    max_window: usize,
    harmonics: usize,
    /// Window batching: due windows wait (to share a wider launch) while every one of them has
    /// more than `slack` of audio buffered, or, with more than twice as many streams live as due,
    /// up to `window_hold` — until `window_batch` windows are due.
    slack: std::time::Duration,
    window_hold: std::time::Duration,
    window_batch: usize,
    /// Render cost model for fitting windows to a capacity, in frames: a launch's floor and each
    /// item's frames beyond its window (the prompt tail).
    launch_frames: usize,
    item_frames: usize,
    sample_rate: f64,
}

impl Schedule {
    fn from_codec(c: &Codec, trim_tail: usize) -> std::result::Result<Self, String> {
        let p = |k: &str| c.parameters.get(k).map(|&v| v as usize).ok_or(format!("vocoder packet lacks {k}"));
        Ok(Self {
            first: p("stream.first_tokens")?,
            chunk: p("stream.chunk_tokens")?,
            hold: p("stream.hold_tokens")?,
            fade: p("stream.fade_samples")?,
            samples_per_token: c.frame_samples,
            max_tokens: c.max_frames,
            min_batch: c.parameters.get("render.min_batch").map_or(1, |&v| v as usize),
            batch_hold: std::time::Duration::from_millis(c.parameters.get("render.hold_ms").copied().unwrap_or(0)),
            trim_tail,
            context: c.parameters.get("stream.context_tokens").map_or(0, |&v| v as usize),
            max_window: if crate::config::RuntimeConfig::get().tts_stream_windows { c.max_window } else { 0 },
            harmonics: c.parameters.get("vocoder.harmonics").map_or(0, |&v| v as usize),
            slack: std::time::Duration::from_millis(c.parameters.get("render.slack_ms").copied().unwrap_or(600)),
            window_hold: std::time::Duration::from_millis(c.parameters.get("render.window_hold_ms").copied().unwrap_or(500)),
            window_batch: c.parameters.get("render.window_batch").map_or(c.max_batch, |&v| v as usize),
            launch_frames: c.parameters.get("render.launch_frames").map_or(370, |&v| v as usize),
            item_frames: c.parameters.get("render.item_frames").map_or(4, |&v| v as usize),
            sample_rate: c.parameters.get("audio.sample_rate").map_or(24000.0, |&v| v as f64),
        })
    }

    fn windowed(&self) -> bool {
        self.max_window > 0
    }
}

/// Window length for one launch of stream windows: of the cached capacities' frame counts, the one
/// delivering the most new tokens per unit of render cost. `jobs` are (tokens available from the
/// window start, left context, closed). Windows are clipped to it, so a launch pays for no
/// padding beyond the windows too short to fill it.
fn fit_window(caps: &[(usize, usize)], jobs: &[(usize, usize, bool)], sc: &Schedule) -> usize {
    let mut best = (0.0, sc.max_window);
    for f in caps.iter().map(|c| c.1).collect::<std::collections::BTreeSet<_>>() {
        let Some(batch) = caps.iter().filter(|c| c.1 == f && c.0 >= jobs.len()).map(|c| c.0).min() else { continue };
        let new: usize = jobs
            .iter()
            .map(|&(avail, ctx, closed)| {
                let hold = if closed && avail <= f { 0 } else { sc.hold };
                avail.min(f).saturating_sub(ctx + hold)
            })
            .sum();
        let yield_ = new as f64 / (sc.launch_frames + batch * (f + sc.item_frames)) as f64;
        if yield_ > best.0 {
            best = (yield_, f);
        }
    }
    best.1
}

#[derive(Debug, Clone)]
pub struct SpeechAudio {
    pub pcm: Vec<f32>,
    pub tokens: usize,
    pub t3_ms: f64,
    pub s3gen_ms: f64,
}

pub enum StreamEvent {
    Pcm(Vec<f32>),
    Done { tokens: usize, t3_ms: f64, s3gen_ms: f64 },
    Err(String),
}

enum Reply {
    Whole(tokio::sync::oneshot::Sender<std::result::Result<SpeechAudio, String>>),
    Stream(tokio::sync::mpsc::UnboundedSender<StreamEvent>),
}

enum S3Msg {
    Open { id: usize, voice: String, seed: u64, reply: Reply },
    Token { id: usize, token: u32 },
    Close { id: usize, t3_ms: f64 },
    /// The LM failed the request.
    Fail { id: usize, error: String },
    /// The client went away: forget the utterance.
    Drop { id: usize },
}

/// The host side of a guided speech model: prompt tables and contract (from the packet) and the
/// render thread. The LM itself is the registry model's mux.
pub struct GuidedSpeech {
    c: GuidedLmContract,
    tables: PromptTables,
    render: parking_lot::Mutex<mpsc::Sender<S3Msg>>,
    next_id: AtomicUsize,
    pub sample_rate: u32,
}

struct Utterance {
    voice: String,
    seed: u64,
    reply: Reply,
    tokens: Vec<u32>,
    t3_ms: Option<f64>,
    s3gen_ms: f64,
    rendered: usize,
    emitted: usize,
    tail: Vec<f32>,
    /// Source phase at the next seam (windowed streams).
    phase: Vec<f32>,
    /// When the stream's first audio went out (its playback clock).
    started: Option<std::time::Instant>,
}

/// One render of an utterance: tokens `[start, end)`; `last` = the utterance's final audio.
#[derive(Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
    last: bool,
}

impl Utterance {
    /// An open stream whose first audio has not been rendered yet.
    fn first_chunk(&self) -> bool {
        self.rendered == 0 && self.t3_ms.is_none() && matches!(self.reply, Reply::Stream(_))
    }

    /// Due for a render: a closed utterance always; an open stream once a chunk has arrived.
    fn due(&self, sc: &Schedule) -> bool {
        match (&self.reply, self.t3_ms) {
            (_, Some(_)) => true,
            (Reply::Whole(_), None) => false,
            (Reply::Stream(_), None) => {
                let n = self.tokens.len();
                if self.rendered == 0 { n >= sc.first } else { n >= self.rendered + sc.chunk }
            }
        }
    }

    /// What the next render covers: a stream window `[emitted - context, now)` (at most
    /// `max_window` tokens) on the cached capacities, else the whole prefix.
    fn span(&self, sc: &Schedule, limit: usize) -> Span {
        let n = self.tokens.len();
        if !(sc.windowed() && matches!(self.reply, Reply::Stream(_))) {
            return Span { start: 0, end: n.min(sc.max_tokens), last: self.t3_ms.is_some() };
        }
        let start = (self.emitted / sc.samples_per_token).saturating_sub(sc.context);
        let end = n.min(start + limit.min(sc.max_window));
        Span { start, end, last: self.t3_ms.is_some() && end == n }
    }

    /// The window of `span`: phase at the seam (where this render's audio starts to be heard) and
    /// the sample of the next seam, both relative to the window start.
    fn window(&self, span: Span, sc: &Schedule) -> Window {
        let spt = sc.samples_per_token;
        let next = if span.last { 0 } else { span.end.saturating_sub(sc.hold).saturating_sub(span.start) * spt };
        let phase = if self.rendered == 0 { initial_phase(self.seed, sc.harmonics) } else { self.phase.clone() };
        Window { seam: (self.emitted - span.start * spt) as u32, next_seam: next as u32, phase }
    }

    /// Audio sent but not yet played on a real-time client (zero before the first audio).
    fn buffered(&self, now: std::time::Instant, sc: &Schedule) -> std::time::Duration {
        self.started.map_or(std::time::Duration::ZERO, |t| {
            std::time::Duration::from_secs_f64(self.emitted as f64 / sc.sample_rate).saturating_sub(now - t)
        })
    }

    /// Consume a render of `span`; returns false when the utterance is finished.
    fn take(&mut self, d: Decoded, span: Span, ms: f64, sc: &Schedule) -> bool {
        self.s3gen_ms += ms;
        self.rendered = span.end;
        self.phase = d.phase.iter().map(|p| p.rem_euclid(std::f32::consts::TAU)).collect();
        let last = span.last;
        let base = span.start * sc.samples_per_token;
        let pcm = &d.pcm[..];
        let pcm = if last {
            &pcm[..pcm.len().min((self.tokens.len().saturating_sub(sc.trim_tail).max(1) * sc.samples_per_token).saturating_sub(base))]
        } else {
            pcm
        };
        match &self.reply {
            Reply::Whole(_) => {
                let Reply::Whole(tx) = std::mem::replace(&mut self.reply, Reply::Stream(tokio::sync::mpsc::unbounded_channel().0))
                else {
                    unreachable!()
                };
                let _ = tx.send(Ok(SpeechAudio {
                    pcm: pcm.to_vec(),
                    tokens: self.tokens.len(),
                    t3_ms: self.t3_ms.unwrap_or(0.0),
                    s3gen_ms: self.s3gen_ms,
                }));
                false
            }
            Reply::Stream(tx) => {
                let end = if last { pcm.len() } else { pcm.len().saturating_sub(sc.hold * sc.samples_per_token) };
                if base + end > self.emitted {
                    let mut chunk = pcm[self.emitted - base..end].to_vec();
                    for (i, (o, t)) in chunk.iter_mut().zip(&self.tail).enumerate() {
                        let w = (i as f32 + 0.5) / self.tail.len() as f32;
                        *o = *t * (1.0 - w) + *o * w;
                    }
                    self.tail = pcm[end..pcm.len().min(end + sc.fade)].to_vec();
                    self.emitted = base + end;
                    self.started.get_or_insert_with(std::time::Instant::now);
                    if tx.send(StreamEvent::Pcm(chunk)).is_err() {
                        return false;
                    }
                }
                if last {
                    let _ = tx.send(StreamEvent::Done {
                        tokens: self.tokens.len(),
                        t3_ms: self.t3_ms.unwrap_or(0.0),
                        s3gen_ms: self.s3gen_ms,
                    });
                }
                !last
            }
        }
    }

    fn fail(self, e: String) {
        match self.reply {
            Reply::Whole(tx) => drop(tx.send(Err(e))),
            Reply::Stream(tx) => drop(tx.send(StreamEvent::Err(e))),
        }
    }
}

/// A stream's NSF source starts at random harmonic phases in [-pi, pi) (the fundamental at 0),
/// drawn from the request seed.
fn initial_phase(seed: u64, harmonics: usize) -> Vec<f32> {
    let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..harmonics)
        .map(|h| {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            if h == 0 { 0.0 } else { ((z >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * std::f32::consts::TAU }
        })
        .collect()
}

fn render_loop(vocoder: &Codec, sc: Schedule, rx: mpsc::Receiver<S3Msg>, credit: &DownstreamCredit) {
    let max_batch = vocoder.max_batch;
    let mut live: HashMap<usize, Utterance> = HashMap::new();
    let mut held_since: Option<std::time::Instant> = None;
    let apply = |live: &mut HashMap<usize, Utterance>, m: S3Msg| match m {
        S3Msg::Open { id, voice, seed, reply } => {
            live.insert(
                id,
                Utterance {
                    voice,
                    seed,
                    reply,
                    tokens: Vec::new(),
                    t3_ms: None,
                    s3gen_ms: 0.0,
                    rendered: 0,
                    emitted: 0,
                    tail: Vec::new(),
                    phase: Vec::new(),
                    started: None,
                },
            );
        }
        S3Msg::Token { id, token } => {
            if let Some(u) = live.get_mut(&id) {
                u.tokens.push(token);
            }
        }
        S3Msg::Close { id, t3_ms } => {
            if let Some(u) = live.get_mut(&id) {
                u.t3_ms = Some(t3_ms);
            }
        }
        S3Msg::Fail { id, error } => {
            if let Some(u) = live.remove(&id) {
                u.fail(error);
            }
        }
        S3Msg::Drop { id } => {
            live.remove(&id);
        }
    };
    loop {
        if !live.values().any(|u| u.due(&sc)) {
            match rx.recv() {
                Ok(m) => apply(&mut live, m),
                Err(_) => return,
            }
        }
        while let Ok(m) = rx.try_recv() {
            apply(&mut live, m);
        }
        let empty: Vec<usize> = live.iter().filter(|(_, u)| u.t3_ms.is_some() && u.tokens.is_empty()).map(|(&k, _)| k).collect();
        for id in empty {
            if let Some(u) = live.remove(&id) {
                u.fail("T3 produced no speech tokens".into());
            }
        }
        // First chunks of streams first (time to first audio), then closed utterances, then the
        // streams furthest behind.
        let mut due: Vec<usize> = live.iter().filter(|(_, u)| u.due(&sc)).map(|(&k, _)| k).collect();
        due.sort_by_key(|k| {
            let u = &live[k];
            let class = if u.first_chunk() { 0 } else if u.t3_ms.is_some() { 1 } else { 2 };
            (class, u.rendered as isize - u.tokens.len() as isize, *k)
        });
        let stream = |u: &Utterance| matches!(u.reply, Reply::Stream(_));
        if sc.windowed() && due.iter().any(|k| stream(&live[k])) {
            // Stream windows are short and a launch costs a large floor, so due windows (first
            // chunks too) share one launch, whole utterances wait for none, and a started stream
            // with audio to spare waits for more windows to join. Once a launch goes, streams
            // halfway to their next window ride along.
            due.retain(|k| stream(&live[k]));
            let now = std::time::Instant::now();
            let spare = |u: &Utterance| u.started.is_some() && u.t3_ms.is_none() && u.buffered(now, &sc) > sc.slack;
            // Loaded (most live streams not yet due): hold for a wider launch.
            let loaded = live.len() > 2 * due.len();
            let hold_left = if loaded { sc.window_hold.saturating_sub(held_since.get_or_insert(now).elapsed()) } else { std::time::Duration::ZERO };
            if due.len() < sc.window_batch && (due.iter().all(|k| spare(&live[k])) || !hold_left.is_zero()) {
                let wait = due.iter().map(|k| live[k].buffered(now, &sc).saturating_sub(sc.slack)).min().unwrap_or_default().max(hold_left);
                match rx.recv_timeout(wait) {
                    Ok(m) => apply(&mut live, m),
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                }
                continue;
            }
            let half = sc.chunk / 2;
            let mut riders: Vec<usize> = live
                .iter()
                .filter(|(k, u)| stream(u) && u.rendered > 0 && !due.contains(k) && u.tokens.len() >= u.rendered + half)
                .map(|(&k, _)| k)
                .collect();
            riders.sort_by_key(|k| (live[k].buffered(now, &sc), *k));
            due.extend(riders);
        } else if due.first().is_some_and(|k| live[k].first_chunk()) {
            // First chunks render by themselves: joined with longer renders they would pay the
            // batch's token capacity and wait for its whole launch.
            due.retain(|k| live[k].first_chunk());
        }
        due.truncate(max_batch);
        if due.is_empty() {
            continue;
        }
        let batchable = if sc.windowed() { !stream(&live[&due[0]]) } else { !live[&due[0]].first_chunk() };
        if batchable && due.len() < sc.min_batch.min(max_batch) && live.len() > due.len() {
            let left = sc.batch_hold.saturating_sub(held_since.get_or_insert_with(std::time::Instant::now).elapsed());
            if !left.is_zero() {
                match rx.recv_timeout(left) {
                    Ok(m) => apply(&mut live, m),
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                }
                continue;
            }
        }
        held_since = None;
        let mut limit = sc.max_window;
        if sc.windowed() && stream(&live[&due[0]]) {
            let jobs: Vec<(usize, usize, bool)> = due
                .iter()
                .map(|k| {
                    let (u, s) = (&live[k], live[k].span(&sc, sc.max_window));
                    (u.tokens.len() - s.start, u.emitted / sc.samples_per_token - s.start, u.t3_ms.is_some())
                })
                .collect();
            limit = fit_window(&vocoder.window_capacities, &jobs, &sc);
            // A window adding no audio at that length waits for the next launch.
            let hold = sc.hold;
            due.retain(|k| {
                let u = &live[k];
                let s = u.span(&sc, limit);
                let ctx = u.emitted / sc.samples_per_token - s.start;
                u.rendered == 0 || s.last || s.end - s.start > ctx + hold
            });
            if due.is_empty() {
                continue;
            }
        }
        let first = due.iter().any(|k| live[k].first_chunk());
        // Cleared on every exit, a panicking render included, so the LM never waits forever.
        struct Urgent<'a>(&'a DownstreamCredit);
        impl Drop for Urgent<'_> {
            fn drop(&mut self) {
                self.0.set_urgent(false);
            }
        }
        // Pausing the LM buys a lone stream its first audio sooner; with many utterances in flight
        // every pause delays all of them (and the requests queued behind them) instead.
        // Under `--co-sched deadline` the render takes the device turn like a mux tick instead.
        let mut turn = crate::serve::cosched::Turn::default();
        if let Some(dt) = credit.device_turn() {
            let now = std::time::Instant::now();
            let due_now = first || due.iter().any(|k| live[k].started.is_some() && live[k].buffered(now, &sc) <= sc.slack);
            let urgency = if due_now { crate::serve::cosched::Urgency::Deadline } else { crate::serve::cosched::Urgency::Normal };
            futures::executor::block_on(turn.take_at(dt, urgency));
        } else if first && live.len() <= sc.min_batch.max(1) {
            credit.set_urgent(true);
        }
        let guard = Urgent(credit);
        let t = std::time::Instant::now();
        // Submitted together so the vocoder worker batches them into one launch.
        let spans: Vec<Span> = due.iter().map(|k| live[k].span(&sc, limit)).collect();
        let renders: Vec<_> = due
            .iter()
            .zip(&spans)
            .map(|(k, &span)| {
                let u = &live[k];
                let voice = vocoder.voices.iter().position(|v| *v == u.voice);
                let codes: Vec<i32> = u.tokens[span.start..span.end].iter().map(|&t| t as i32).collect();
                let window = (sc.windowed() && matches!(u.reply, Reply::Stream(_))).then(|| u.window(span, &sc));
                // Windows draw fresh noise: their frames are keyed from the window start.
                let seed = u.seed ^ (span.start as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let frames = span.end - span.start;
                async move {
                    match (voice, window) {
                        (Some(v), Some(w)) => vocoder.decode_window(codes, frames, seed, v as u32, w).await,
                        (Some(v), None) => vocoder.decode_voice(codes, frames, seed, v as u32).await.map(|pcm| Decoded { pcm, phase: Vec::new() }),
                        (None, _) => Err(format!("unknown voice {:?}", u.voice)),
                    }
                }
            })
            .collect();
        let results = futures::executor::block_on(futures::future::join_all(renders));
        drop((guard, turn));
        let ms = t.elapsed().as_secs_f64() * 1e3;
        tracing::debug!(renders = due.len(), tokens = ?spans.iter().map(|s| s.end - s.start).collect::<Vec<_>>(), ms, "vocoder render");
        for ((k, pcm), span) in due.into_iter().zip(results).zip(spans) {
            match pcm {
                Err(e) => {
                    if let Some(u) = live.remove(&k) {
                        u.fail(e);
                    }
                }
                Ok(pcm) => {
                    let keep = live.get_mut(&k).is_some_and(|u| u.take(pcm, span, ms, &sc));
                    if !keep {
                        live.remove(&k);
                    }
                }
            }
        }
    }
}

impl GuidedSpeech {
    /// Bind the packet's prompt tables and start the render thread. `credit` is the serving
    /// model's downstream credit: the vocoder's backlog gates the model's admission, and a render
    /// holding a first chunk holds its ticks.
    pub fn start(assets: &Path, credit: Arc<DownstreamCredit>) -> Result<Self> {
        let c = GuidedLmContract::load(assets)?
            .ok_or_else(|| RuntimeError::Rejected(format!("{} declares no guided LM pipeline", assets.display())))?;
        let tables = PromptTables::load(assets, c.hidden)?;
        let trim_tail = c.trim_tail_tokens;
        let (s_tx, s_rx) = mpsc::channel::<S3Msg>();
        let (s_ready_tx, s_ready_rx) = mpsc::channel::<Result<()>>();
        let dir = assets.to_path_buf();
        std::thread::Builder::new()
            .name("plow-tts-render".into())
            .spawn(move || {
                let mut vocoder = match Codec::load_packet(&dir.join(VOCODER)) {
                    Ok(v) => v,
                    Err(e) => return drop(s_ready_tx.send(Err(RuntimeError::Device(e)))),
                };
                vocoder.couple(Arc::clone(&credit));
                let sc = match Schedule::from_codec(&vocoder, trim_tail) {
                    Ok(sc) => sc,
                    Err(e) => return drop(s_ready_tx.send(Err(RuntimeError::Rejected(e)))),
                };
                let _ = s_ready_tx.send(Ok(()));
                render_loop(&vocoder, sc, s_rx, &credit);
            })
            .map_err(|e| RuntimeError::Device(e.to_string()))?;
        s_ready_rx.recv().map_err(|e| RuntimeError::Device(e.to_string()))??;
        Ok(GuidedSpeech { c, tables, render: parking_lot::Mutex::new(s_tx), next_id: AtomicUsize::new(0), sample_rate: 24000 })
    }

    /// The mux job for one request: both CFG members' prefill rows as overlays, the decode
    /// position base, the packet's sampling chain.
    fn job(
        &self,
        voice: &str,
        text: &str,
        lang: Option<&str>,
        seed: u64,
        class: JobClass,
        respond: crate::serve::stream::ChunkSender,
        request: &crate::serve::session::RequestIds,
        report: Option<crate::serve::session::Report>,
    ) -> Result<Job> {
        let c = &self.c;
        let ids = self.tables.text_ids(text, lang)?;
        let cond = self.tables.prefill_rows(c, voice, &ids, false)?;
        let uncond = self.tables.prefill_rows(c, voice, &ids, true)?;
        let n = cond.len() / c.hidden;
        let mut gen = crate::serve::GenParams::default();
        gen.max_tokens = c.max_speech_tokens;
        gen.params.temperature = 0.0;
        gen.stop_token_ids = vec![c.stop_speech];
        // Every row is an overlay; the last (BOS) is also a decode embedding at `pos_base`, which
        // packed prefill uses for it.
        let mut prompt_ids = vec![0; n];
        prompt_ids[n - 1] = c.start_speech;
        let overlay_pos: Vec<u32> = (0..n as u32).collect();
        // Both CFG members' rows key a position: a session reuses the voice conditioning prefix.
        let session = request.session.as_ref().and_then(|_| {
            request.ticket(crate::serve::session::row_keys(&prompt_ids, &overlay_pos, &[&cond, &uncond]), report)
        });
        Ok(Job {
            prompt_ids,
            gen,
            arrived: std::time::Instant::now(),
            respond,
            opts: JobOpts {
                class,
                raw_tokens: true,
                session,
                speech: Some(Box::new(SpeechJob {
                    overlay: cond,
                    overlay_pos,
                    // Decode token k takes speech_pos[k + 1]: base = prefill rows - 1.
                    pos_base: Some(n as u32 - 1),
                    cfg: Some(CfgJob { uncond_overlay: uncond, params: c.cfg(), history: vec![c.start_speech], seed: Some(seed) }),
                })),
            },
        })
    }

    fn submit(
        &self,
        mux: &ModelMux,
        voice: String,
        text: String,
        lang: Option<&str>,
        seed: u64,
        reply: Reply,
        ids: &crate::serve::session::RequestIds,
        report: Option<crate::serve::session::Report>,
    ) -> std::result::Result<(), String> {
        let (respond, mut tokens) = crate::serve::stream::channel();
        let probe = match &reply {
            Reply::Stream(tx) => Some(tx.clone()),
            Reply::Whole(_) => None,
        };
        let class = if probe.is_some() { JobClass::Critical } else { JobClass::Normal };
        let job = self.job(&voice, &text, lang, seed, class, respond, ids, report).map_err(|e| e.to_string())?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let render = self.render.lock().clone();
        let _ = render.send(S3Msg::Open { id, voice, seed, reply });
        if let Err(e) = mux.submit(job) {
            let _ = render.send(S3Msg::Drop { id });
            return Err(match e {
                SubmitError::Full(_) => QUEUE_FULL.to_string(),
                SubmitError::Closed(_) => "speech model dispatcher unavailable".to_string(),
            });
        }
        let valid_below = self.c.valid_below;
        let t0 = std::time::Instant::now();
        tokio::spawn(async move {
            let msg = loop {
                let msg = match tokens.recv().await {
                    Some(StreamChunk::Token { id: token, .. }) => {
                        // A vanished stream client: dropping `tokens` frees the pair next tick.
                        if probe.as_ref().is_some_and(|p| p.is_closed()) {
                            break S3Msg::Drop { id };
                        }
                        if token < valid_below {
                            let _ = render.send(S3Msg::Token { id, token });
                        }
                        continue;
                    }
                    Some(StreamChunk::Done { .. }) => S3Msg::Close { id, t3_ms: t0.elapsed().as_secs_f64() * 1e3 },
                    Some(StreamChunk::Err(e)) => S3Msg::Fail { id, error: e.to_string() },
                    None => S3Msg::Fail { id, error: "speech LM stream ended without a result".into() },
                };
                break msg;
            };
            let _ = render.send(msg);
        });
        Ok(())
    }

    /// The request's language as the packet's text rules resolve it (400 on an unsupported one).
    pub fn language(&self, lang: Option<&str>) -> std::result::Result<Option<String>, String> {
        self.tables.language(lang).map_err(|e| e.to_string())
    }

    pub async fn synthesize(
        &self,
        mux: &ModelMux,
        voice: String,
        text: String,
        lang: Option<&str>,
        seed: u64,
        ids: &crate::serve::session::RequestIds,
        report: Option<crate::serve::session::Report>,
    ) -> std::result::Result<SpeechAudio, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.submit(mux, voice, text, lang, seed, Reply::Whole(reply), ids, report)?;
        rx.await.map_err(|_| "chatterbox render dropped the request".to_string())?
    }

    pub fn synthesize_stream(
        &self,
        mux: &ModelMux,
        voice: String,
        text: String,
        lang: Option<&str>,
        seed: u64,
        ids: &crate::serve::session::RequestIds,
        report: Option<crate::serve::session::Report>,
    ) -> std::result::Result<tokio::sync::mpsc::UnboundedReceiver<StreamEvent>, String> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.submit(mux, voice, text, lang, seed, Reply::Stream(tx), ids, report)?;
        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(tokens: usize) -> (Utterance, tokio::sync::mpsc::UnboundedReceiver<StreamEvent>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let u = Utterance {
            voice: String::new(),
            seed: 1,
            reply: Reply::Stream(tx),
            tokens: vec![0; tokens],
            t3_ms: None,
            s3gen_ms: 0.0,
            rendered: 0,
            emitted: 0,
            tail: Vec::new(),
            phase: Vec::new(),
            started: None,
        };
        (u, rx)
    }

    const SC: Schedule = Schedule {
        first: 20,
        chunk: 25,
        hold: 3,
        fade: 480,
        samples_per_token: 960,
        max_tokens: 1000,
        min_batch: 1,
        batch_hold: std::time::Duration::ZERO,
        trim_tail: 0,
        context: 0,
        max_window: 0,
        harmonics: 0,
        slack: std::time::Duration::ZERO,
        window_hold: std::time::Duration::ZERO,
        window_batch: 1,
        launch_frames: 370,
        item_frames: 4,
        sample_rate: 24000.0,
    };

    /// Drives `u` to completion; renders produce sample `i` (absolute) = `i % 1024`.
    fn drive(u: &mut Utterance, sc: &Schedule, total: usize) {
        for n in 1..=total {
            u.tokens.push(0);
            if n == total {
                u.t3_ms = Some(1.0);
            }
            while u.due(sc) {
                let span = u.span(sc, sc.max_window);
                let base = span.start * sc.samples_per_token;
                let pcm: Vec<f32> = (base..span.end * sc.samples_per_token).map(|i| (i % 1024) as f32).collect();
                let more = u.take(Decoded { pcm, phase: vec![0.0; sc.harmonics] }, span, 1.0, sc);
                assert_eq!(more, !span.last);
                if !more {
                    return;
                }
            }
        }
    }

    fn collect(rx: &mut tokio::sync::mpsc::UnboundedReceiver<StreamEvent>, total: usize, sc: &Schedule) {
        let mut next = 0usize;
        let mut done = false;
        while let Ok(ev) = rx.try_recv() {
            match ev {
                StreamEvent::Pcm(p) => {
                    // Identical renders: the crossfade is the identity.
                    for v in p {
                        assert!((v - (next % 1024) as f32).abs() < 1e-3, "sample {next} = {v}");
                        next += 1;
                    }
                }
                StreamEvent::Done { tokens, .. } => {
                    assert_eq!(tokens, total);
                    done = true;
                }
                StreamEvent::Err(e) => panic!("{e}"),
            }
        }
        assert!(done);
        assert_eq!(next, total * sc.samples_per_token);
    }

    #[test]
    fn fitted_window_trades_padding_for_length() {
        let caps = [(1, 32), (8, 32), (8, 64), (64, 32), (64, 64)];
        let sc = Schedule { max_window: 64, ..SC };
        // Long backlogs fill the widest window; short ones would pad it.
        assert_eq!(fit_window(&caps, &[(70, 8, false); 40], &sc), 64);
        assert_eq!(fit_window(&caps, &[(34, 8, false); 40], &sc), 32);
        // Batch of 1: a lone first chunk.
        assert_eq!(fit_window(&caps, &[(20, 0, false)], &sc), 32);
    }

    /// Windows of `context` left tokens (capped at `max_window`) emit every sample exactly once.
    #[test]
    fn stream_windows_cover_each_sample_once() {
        let sc = Schedule { context: 8, max_window: 32, harmonics: 9, ..SC };
        for total in [5, 20, 21, 44, 90, 200] {
            let (mut u, mut rx) = stream(0);
            drive(&mut u, &sc, total);
            collect(&mut rx, total, &sc);
        }
    }

    /// Renders of a growing prefix emit every sample exactly once, in order, then Done.
    #[test]
    fn stream_renders_cover_each_sample_once() {
        let (mut u, mut rx) = stream(0);
        let mut total = 0;
        for n in 1..=90 {
            u.tokens.push(0);
            if n == 90 {
                u.t3_ms = Some(1.0);
            }
            if u.due(&SC) {
                let pcm: Vec<f32> = (0..n * SC.samples_per_token).map(|i| (i % 1024) as f32).collect();
                let more = u.take(Decoded { pcm, phase: Vec::new() }, u.span(&SC, SC.max_window), 1.0, &SC);
                assert_eq!(more, n != 90);
            }
        }
        let mut next = 0usize;
        while let Ok(ev) = rx.try_recv() {
            match ev {
                StreamEvent::Pcm(p) => {
                    // Identical renders: the crossfade is the identity.
                    for v in p {
                        assert!((v - (next % 1024) as f32).abs() < 1e-3, "sample {next} = {v}");
                        next += 1;
                    }
                }
                StreamEvent::Done { tokens, .. } => {
                    assert_eq!(tokens, 90);
                    total = next;
                }
                StreamEvent::Err(e) => panic!("{e}"),
            }
        }
        assert_eq!(total, 90 * SC.samples_per_token);
    }
}
