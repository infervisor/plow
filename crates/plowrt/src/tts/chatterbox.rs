//! The Chatterbox speech worker: T3 (`tts.t3_cfg.v1` packet on its own `GpuEngine`) feeding the
//! S3Gen stage. Two threads: the T3 thread batches requests continuously over CFG slot pairs and
//! forwards each speech token as it is committed; the S3Gen thread renders batches of utterances,
//! so decoding never waits on audio rendering.
//!
//! Streaming: S3Gen is not causal over tokens (the conformer encoder attends both ways), so a
//! stream re-renders its whole token prefix every `STREAM_CHUNK` tokens and emits the audio of all
//! but the last `STREAM_HOLD` tokens, crossfading `FADE` samples into the previous render's tail.
//! The noise streams are keyed by frame, so re-renders of a prefix agree up to that lookahead.
//! A render sharing the GPU with T3's back-to-back cooperative decode launches runs ~6x slower
//! (220 vs 35 ms for a first chunk), so T3 pauses while a batch holding a first chunk renders:
//! first audio is then prefill + `STREAM_FIRST` tokens + one uncontended render.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use super::s3gen::{Render, S3Gen, SAMPLES_PER_TOKEN};
use super::t3::{T3Engine, T3Job};
use crate::{Result, RuntimeError};

const S3_BATCH: usize = 8;
const S3_MAX_TOKENS: usize = 1000;
const STREAM_FIRST: usize = 20;
const STREAM_CHUNK: usize = 25;
const STREAM_HOLD: usize = 3;
const FADE: usize = 480;

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
}

pub struct ChatterboxWorker {
    tx: parking_lot::Mutex<mpsc::Sender<SpeechRequest>>,
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
    /// Due for a render: a closed utterance always; an open stream once a chunk has arrived.
    fn due(&self) -> bool {
        match (&self.reply, self.t3_ms) {
            (_, Some(_)) => true,
            (Reply::Whole(_), None) => false,
            (Reply::Stream(_), None) => {
                let n = self.tokens.len();
                if self.rendered == 0 { n >= STREAM_FIRST } else { n >= self.rendered + STREAM_CHUNK }
            }
        }
    }

    /// Consume a render of `self.tokens`; returns false when the utterance is finished.
    fn take(&mut self, pcm: &[f32], ms: f64) -> bool {
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
                let end = if last { pcm.len() } else { pcm.len().saturating_sub(STREAM_HOLD * SAMPLES_PER_TOKEN) };
                if end > self.emitted {
                    let mut chunk = pcm[self.emitted..end].to_vec();
                    for (i, (o, t)) in chunk.iter_mut().zip(&self.tail).enumerate() {
                        let w = (i as f32 + 0.5) / self.tail.len() as f32;
                        *o = *t * (1.0 - w) + *o * w;
                    }
                    self.tail = pcm[end..pcm.len().min(end + FADE)].to_vec();
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

fn s3gen_loop(s3: &mut S3Gen, rx: mpsc::Receiver<S3Msg>, urgent: &AtomicBool) {
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
    };
    loop {
        if !live.values().any(Utterance::due) {
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
        // Closed utterances first, then the streams furthest behind.
        let mut due: Vec<usize> = live.iter().filter(|(_, u)| u.due()).map(|(&k, _)| k).collect();
        due.sort_by_key(|k| {
            let u = &live[k];
            (u.t3_ms.is_none(), u.rendered as isize - u.tokens.len() as isize, *k)
        });
        due.truncate(s3.max_batch);
        if due.is_empty() {
            continue;
        }
        let first = due.iter().any(|k| {
            let u = &live[k];
            u.rendered == 0 && u.t3_ms.is_none() && matches!(u.reply, Reply::Stream(_))
        });
        urgent.store(first, Ordering::Release);
        let t = std::time::Instant::now();
        let mut pcms: Vec<Vec<f32>> = vec![Vec::new(); due.len()];
        let res = {
            let items: Vec<Render> = due
                .iter()
                .map(|k| {
                    let u = &live[k];
                    let n = u.tokens.len().min(S3_MAX_TOKENS);
                    Render { voice: &u.voice, tokens: &u.tokens[..n], seed: u.seed }
                })
                .collect();
            s3.synthesize_batch(&items, |i, pcm| pcms[i] = pcm.to_vec())
        };
        urgent.store(false, Ordering::Release);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        tracing::debug!(renders = due.len(), tokens = ?due.iter().map(|k| live[k].tokens.len()).collect::<Vec<_>>(), ms, "s3gen render");
        for (k, pcm) in due.into_iter().zip(pcms) {
            match &res {
                Err(e) => {
                    if let Some(u) = live.remove(&k) {
                        u.fail(e.to_string());
                    }
                }
                Ok(()) => {
                    let keep = live.get_mut(&k).is_some_and(|u| u.take(&pcm, ms));
                    if !keep {
                        live.remove(&k);
                    }
                }
            }
        }
    }
}

impl ChatterboxWorker {
    pub fn start(assets: &Path, device: u8) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<SpeechRequest>();
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
                let mut t3 = match T3Engine::load(&dir, device) {
                    Ok(t) => t,
                    Err(e) => return drop(ready_tx.send(Err(e))),
                };
                let valid_below = t3.c.s3_valid_below;
                let _ = ready_tx.send(Ok(()));
                let started: std::cell::RefCell<Vec<std::time::Instant>> = Default::default();
                let res = t3.serve(
                    |block| {
                        let req = if block { rx.recv().ok() } else { rx.try_recv().ok() }?;
                        let id = {
                            let mut s = started.borrow_mut();
                            s.push(std::time::Instant::now());
                            s.len() - 1
                        };
                        let _ = s_tx.send(S3Msg::Open { id, voice: req.voice.clone(), seed: req.seed, reply: req.reply });
                        Some(T3Job { voice: req.voice, text: req.text, seed: Some(req.seed), max_tokens: None })
                    },
                    |id, token| {
                        while urgent.load(Ordering::Acquire) {
                            std::thread::sleep(std::time::Duration::from_micros(50));
                        }
                        if token < valid_below {
                            let _ = s_tx.send(S3Msg::Token { id, token });
                        }
                    },
                    |id, _| {
                        let t3_ms = started.borrow()[id].elapsed().as_secs_f64() * 1e3;
                        let _ = s_tx.send(S3Msg::Close { id, t3_ms });
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
            .name("plow-tts-s3gen".into())
            .spawn(move || {
                let mut s3 = match S3Gen::load(&dir2, S3_BATCH, S3_MAX_TOKENS).and_then(|mut s| s.warm().map(|()| s)) {
                    Ok(s) => s,
                    Err(e) => return drop(s_ready_tx.send(Err(e))),
                };
                let _ = s_ready_tx.send(Ok(()));
                s3gen_loop(&mut s3, s_rx, &urgent2);
            })
            .map_err(|e| RuntimeError::Device(e.to_string()))?;
        s_ready_rx.recv().map_err(|e| RuntimeError::Device(e.to_string()))??;
        Ok(ChatterboxWorker { tx: parking_lot::Mutex::new(tx), sample_rate: 24000 })
    }

    fn submit(&self, voice: String, text: String, seed: u64, reply: Reply) -> std::result::Result<(), String> {
        self.tx.lock().send(SpeechRequest { voice, text, seed, reply }).map_err(|_| "chatterbox worker stopped".to_string())
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
            if u.due() {
                let pcm: Vec<f32> = (0..n * SAMPLES_PER_TOKEN).map(|i| (i % 1024) as f32).collect();
                let more = u.take(&pcm, 1.0);
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
        assert_eq!(total, 90 * SAMPLES_PER_TOKEN);
    }
}
