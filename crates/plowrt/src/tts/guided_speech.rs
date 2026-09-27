//! The guided speech worker: a guided token LM (`tts.t3_cfg.v1` packet on its own `GpuEngine`)
//! feeding a token vocoder (`s3gen.pkt`, a `codec.v1` packet on the packet runtime). Two threads:
//! the LM thread batches requests continuously over CFG slot pairs and forwards each speech token
//! as it is committed; the render thread renders batches of utterances, so decoding never waits on
//! audio rendering.
//!
//! Streaming (schedule from the vocoder packet's `stream.*` parameters): the vocoder is not causal
//! over tokens, so a stream re-renders its whole token prefix every `chunk` tokens and emits the
//! audio of all but the last `hold` tokens, crossfading `fade` samples into the previous render's
//! tail. The noise streams are keyed by frame, so re-renders of a prefix agree up to that
//! lookahead. A render sharing the GPU with the LM's back-to-back cooperative decode launches runs
//! several times slower, so the LM pauses while a batch holding a first chunk renders: first audio
//! is then prefill + `first` tokens + one uncontended render.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use super::codec::Codec;
use super::guided_lm::{GuidedLm, GuidedJob};
use crate::{Result, RuntimeError};

/// Requests waiting for a slot pair; beyond this the route answers 429.
const QUEUE: usize = 64;
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
}

impl Schedule {
    fn from_codec(c: &Codec) -> std::result::Result<Self, String> {
        let p = |k: &str| c.parameters.get(k).map(|&v| v as usize).ok_or(format!("vocoder packet lacks {k}"));
        Ok(Self {
            first: p("stream.first_tokens")?,
            chunk: p("stream.chunk_tokens")?,
            hold: p("stream.hold_tokens")?,
            fade: p("stream.fade_samples")?,
            samples_per_token: c.frame_samples,
            max_tokens: c.max_frames,
        })
    }
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

struct SpeechRequest {
    voice: String,
    text: String,
    seed: u64,
    reply: Reply,
}

enum S3Msg {
    Open { id: usize, voice: String, seed: u64, reply: Reply },
    Token { id: usize, token: u32 },
    Close { id: usize, t3_ms: f64 },
    /// The client went away: forget the utterance.
    Drop { id: usize },
}

pub struct GuidedSpeechWorker {
    tx: parking_lot::Mutex<mpsc::SyncSender<SpeechRequest>>,
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

    /// Consume a render of `self.tokens`; returns false when the utterance is finished.
    fn take(&mut self, pcm: &[f32], ms: f64, sc: &Schedule) -> bool {
        self.s3gen_ms += ms;
        self.rendered = self.tokens.len();
        let last = self.t3_ms.is_some();
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
                if end > self.emitted {
                    let mut chunk = pcm[self.emitted..end].to_vec();
                    for (i, (o, t)) in chunk.iter_mut().zip(&self.tail).enumerate() {
                        let w = (i as f32 + 0.5) / self.tail.len() as f32;
                        *o = *t * (1.0 - w) + *o * w;
                    }
                    self.tail = pcm[end..pcm.len().min(end + sc.fade)].to_vec();
                    self.emitted = end;
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

fn render_loop(vocoder: &Codec, sc: Schedule, rx: mpsc::Receiver<S3Msg>, urgent: &AtomicBool) {
    let max_batch = 8;
    let mut live: HashMap<usize, Utterance> = HashMap::new();
    let apply = |live: &mut HashMap<usize, Utterance>, m: S3Msg| match m {
        S3Msg::Open { id, voice, seed, reply } => {
            live.insert(
                id,
                Utterance { voice, seed, reply, tokens: Vec::new(), t3_ms: None, s3gen_ms: 0.0, rendered: 0, emitted: 0, tail: Vec::new() },
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
        due.truncate(max_batch);
        if due.is_empty() {
            continue;
        }
        let first = due.iter().any(|k| live[k].first_chunk());
        // Cleared on every exit, a panicking render included, so the LM never waits forever.
        struct Urgent<'a>(&'a AtomicBool);
        impl Drop for Urgent<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        urgent.store(first, Ordering::Release);
        let guard = Urgent(urgent);
        let t = std::time::Instant::now();
        // Submitted together so the vocoder worker batches them into one launch.
        let renders: Vec<_> = due
            .iter()
            .map(|k| {
                let u = &live[k];
                let n = u.tokens.len().min(sc.max_tokens);
                let voice = vocoder.voices.iter().position(|v| *v == u.voice);
                let codes: Vec<i32> = u.tokens[..n].iter().map(|&t| t as i32).collect();
                async move {
                    match voice {
                        Some(v) => vocoder.decode_voice(codes, n, u.seed, v as u32).await,
                        None => Err(format!("unknown voice {:?}", u.voice)),
                    }
                }
            })
            .collect();
        let results = futures::executor::block_on(futures::future::join_all(renders));
        drop(guard);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        tracing::debug!(renders = due.len(), tokens = ?due.iter().map(|k| live[k].tokens.len()).collect::<Vec<_>>(), ms, "vocoder render");
        for (k, pcm) in due.into_iter().zip(results) {
            match pcm {
                Err(e) => {
                    if let Some(u) = live.remove(&k) {
                        u.fail(e);
                    }
                }
                Ok(pcm) => {
                    let keep = live.get_mut(&k).is_some_and(|u| u.take(&pcm, ms, &sc));
                    if !keep {
                        live.remove(&k);
                    }
                }
            }
        }
    }
}

impl GuidedSpeechWorker {
    pub fn start(assets: &Path, device: u8) -> Result<Self> {
        let (tx, rx) = mpsc::sync_channel::<SpeechRequest>(QUEUE);
        let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
        let (s_tx, s_rx) = mpsc::channel::<S3Msg>();
        let dir = assets.to_path_buf();
        let dir2 = dir.clone();
        let (s_ready_tx, s_ready_rx) = mpsc::channel::<Result<()>>();
        let urgent = Arc::new(AtomicBool::new(false));
        let urgent2 = Arc::clone(&urgent);
        std::thread::Builder::new()
            .name("plow-tts-t3".into())
            .spawn(move || {
                let mut t3 = match GuidedLm::load(&dir, device) {
                    Ok(t) => t,
                    Err(e) => return drop(ready_tx.send(Err(e))),
                };
                let valid_below = t3.c.valid_below;
                let _ = ready_tx.send(Ok(()));
                // Per request: arrival time and, for a stream, a handle to tell a vanished client.
                let started: std::cell::RefCell<HashMap<usize, (std::time::Instant, Option<tokio::sync::mpsc::UnboundedSender<StreamEvent>>)>> =
                    Default::default();
                let arrivals = std::cell::Cell::new(0usize);
                let res = t3.serve(
                    |block| {
                        let req = if block { rx.recv().ok() } else { rx.try_recv().ok() }?;
                        let id = arrivals.get();
                        arrivals.set(id + 1);
                        let probe = match &req.reply {
                            Reply::Stream(tx) => Some(tx.clone()),
                            Reply::Whole(_) => None,
                        };
                        started.borrow_mut().insert(id, (std::time::Instant::now(), probe));
                        let _ = s_tx.send(S3Msg::Open { id, voice: req.voice.clone(), seed: req.seed, reply: req.reply });
                        Some(GuidedJob { voice: req.voice, text: req.text, seed: Some(req.seed), max_tokens: None })
                    },
                    |id, token| {
                        if started.borrow().get(&id).and_then(|(_, p)| p.as_ref()).is_some_and(|p| p.is_closed()) {
                            started.borrow_mut().remove(&id);
                            let _ = s_tx.send(S3Msg::Drop { id });
                            return false;
                        }
                        while urgent.load(Ordering::Acquire) {
                            std::thread::sleep(std::time::Duration::from_micros(50));
                        }
                        if token < valid_below {
                            let _ = s_tx.send(S3Msg::Token { id, token });
                        }
                        true
                    },
                    |id, _| {
                        if let Some((t0, _)) = started.borrow_mut().remove(&id) {
                            let _ = s_tx.send(S3Msg::Close { id, t3_ms: t0.elapsed().as_secs_f64() * 1e3 });
                        }
                    },
                );
                if let Err(e) = res {
                    tracing::error!(error = %e, "chatterbox T3 worker stopped");
                }
            })
            .map_err(|e| RuntimeError::Device(e.to_string()))?;
        ready_rx.recv().map_err(|e| RuntimeError::Device(e.to_string()))??;
        // After the engine: its backend loads the real driver by path. The stage's static CUDA
        // runtime then resolves `libcuda.so.1` to that library instead of searching (which can
        // land on a toolkit stub: "driver version is insufficient").
        std::thread::Builder::new()
            .name("plow-tts-render".into())
            .spawn(move || {
                let vocoder = match Codec::load_packet(&dir2.join(VOCODER)) {
                    Ok(v) => v,
                    Err(e) => return drop(s_ready_tx.send(Err(RuntimeError::Device(e)))),
                };
                let sc = match Schedule::from_codec(&vocoder) {
                    Ok(sc) => sc,
                    Err(e) => return drop(s_ready_tx.send(Err(RuntimeError::Rejected(e)))),
                };
                let _ = s_ready_tx.send(Ok(()));
                render_loop(&vocoder, sc, s_rx, &urgent2);
            })
            .map_err(|e| RuntimeError::Device(e.to_string()))?;
        s_ready_rx.recv().map_err(|e| RuntimeError::Device(e.to_string()))??;
        Ok(GuidedSpeechWorker { tx: parking_lot::Mutex::new(tx), sample_rate: 24000 })
    }

    fn submit(&self, voice: String, text: String, seed: u64, reply: Reply) -> std::result::Result<(), String> {
        self.tx.lock().try_send(SpeechRequest { voice, text, seed, reply }).map_err(|e| match e {
            mpsc::TrySendError::Full(_) => QUEUE_FULL.to_string(),
            mpsc::TrySendError::Disconnected(_) => "speech worker stopped".to_string(),
        })
    }

    pub async fn synthesize(&self, voice: String, text: String, seed: u64) -> std::result::Result<SpeechAudio, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.submit(voice, text, seed, Reply::Whole(reply))?;
        rx.await.map_err(|_| "chatterbox worker dropped the request".to_string())?
    }

    pub fn synthesize_stream(
        &self,
        voice: String,
        text: String,
        seed: u64,
    ) -> std::result::Result<tokio::sync::mpsc::UnboundedReceiver<StreamEvent>, String> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.submit(voice, text, seed, Reply::Stream(tx))?;
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
        };
        (u, rx)
    }

    const SC: Schedule = Schedule { first: 20, chunk: 25, hold: 3, fade: 480, samples_per_token: 960, max_tokens: 1000 };

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
                let more = u.take(&pcm, 1.0, &SC);
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
