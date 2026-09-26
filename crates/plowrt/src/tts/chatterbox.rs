//! The Chatterbox speech worker: T3 (`tts.t3_cfg.v1` packet on its own `GpuEngine`) feeding the
//! S3Gen stage. Two threads: the T3 thread batches requests continuously over CFG slot pairs;
//! finished token sequences go to the S3Gen thread, so decoding never waits on audio rendering.

use std::path::Path;
use std::sync::mpsc;

use super::s3gen::S3Gen;
use super::t3::{T3Engine, T3Job};
use crate::{Result, RuntimeError};

pub struct SpeechRequest {
    pub voice: String,
    pub text: String,
    pub seed: u64,
    pub reply: tokio::sync::oneshot::Sender<std::result::Result<SpeechAudio, String>>,
}

#[derive(Debug, Clone)]
pub struct SpeechAudio {
    pub pcm: Vec<f32>,
    pub tokens: usize,
    pub t3_ms: f64,
    pub s3gen_ms: f64,
}

pub struct ChatterboxWorker {
    tx: parking_lot::Mutex<mpsc::Sender<SpeechRequest>>,
    pub sample_rate: u32,
}

struct Pending {
    voice: String,
    seed: u64,
    t0: std::time::Instant,
    reply: tokio::sync::oneshot::Sender<std::result::Result<SpeechAudio, String>>,
}

impl ChatterboxWorker {
    pub fn start(assets: &Path, device: u8) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<SpeechRequest>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
        let (s_tx, s_rx) = mpsc::channel::<(Pending, Vec<u32>, f64)>();
        let dir = assets.to_path_buf();
        let dir2 = dir.clone();
        let (s_ready_tx, s_ready_rx) = mpsc::channel::<Result<()>>();
        std::thread::Builder::new()
            .name("plow-tts-s3gen".into())
            .spawn(move || {
                let mut s3 = match S3Gen::load(&dir2, 1000) {
                    Ok(s) => s,
                    Err(e) => return drop(s_ready_tx.send(Err(e))),
                };
                let _ = s_ready_tx.send(Ok(()));
                while let Ok((p, tokens, t3_ms)) = s_rx.recv() {
                    let t = std::time::Instant::now();
                    let r = s3.synthesize(&p.voice, &tokens, p.seed).map_err(|e| e.to_string()).map(|pcm| SpeechAudio {
                        pcm,
                        tokens: tokens.len(),
                        t3_ms,
                        s3gen_ms: t.elapsed().as_secs_f64() * 1e3,
                    });
                    let _ = p.reply.send(r);
                }
            })
            .map_err(|e| RuntimeError::Device(e.to_string()))?;
        s_ready_rx.recv().map_err(|e| RuntimeError::Device(e.to_string()))??;
        std::thread::Builder::new()
            .name("plow-tts-t3".into())
            .spawn(move || {
                let mut t3 = match T3Engine::load(&dir, device) {
                    Ok(t) => t,
                    Err(e) => return drop(ready_tx.send(Err(e))),
                };
                let valid_below = t3.c.s3_valid_below;
                let _ = ready_tx.send(Ok(()));
                // Both callbacks touch the table; they never run at the same time.
                let pending: std::cell::RefCell<Vec<Option<Pending>>> = Default::default();
                let res = t3.serve(
                    |block| {
                        let req = if block { rx.recv().ok() } else { rx.try_recv().ok() }?;
                        pending.borrow_mut().push(Some(Pending { voice: req.voice.clone(), seed: req.seed, t0: std::time::Instant::now(), reply: req.reply }));
                        Some(T3Job { voice: req.voice, text: req.text, seed: Some(req.seed), max_tokens: None })
                    },
                    |i, out| {
                        let Some(p) = pending.borrow_mut().get_mut(i).and_then(Option::take) else { return };
                        let tokens: Vec<u32> = out.tokens.into_iter().filter(|&t| t < valid_below).collect();
                        if tokens.is_empty() {
                            let _ = p.reply.send(Err("T3 produced no speech tokens".into()));
                            return;
                        }
                        let t3_ms = p.t0.elapsed().as_secs_f64() * 1e3;
                        let _ = s_tx.send((p, tokens, t3_ms));
                    },
                );
                if let Err(e) = res {
                    tracing::error!(error = %e, "chatterbox T3 worker stopped");
                }
            })
            .map_err(|e| RuntimeError::Device(e.to_string()))?;
        ready_rx.recv().map_err(|e| RuntimeError::Device(e.to_string()))??;
        Ok(ChatterboxWorker { tx: parking_lot::Mutex::new(tx), sample_rate: 24000 })
    }

    pub async fn synthesize(&self, voice: String, text: String, seed: u64) -> std::result::Result<SpeechAudio, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.tx
            .lock()
            .send(SpeechRequest { voice, text, seed, reply })
            .map_err(|_| "chatterbox worker stopped".to_string())?;
        rx.await.map_err(|_| "chatterbox worker dropped the request".to_string())?
    }
}
