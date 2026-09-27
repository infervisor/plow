//! Guided speech on `plowrt serve`: a guided token LM (`tts.guided_lm.v1`, served by the model's
//! continuous-batching mux as CFG slot pairs) feeding a token vocoder (`s3gen.pkt`, a `codec.v1`
//! packet on the packet runtime). Each request is a mux job whose prefill rows are host
//! embeddings; its speech tokens are forwarded, as they are committed, to one render thread that
//! renders batches of utterances, so decoding never waits on audio rendering.
//!
//! Streaming (schedule from the vocoder packet's `stream.*` parameters): the vocoder is not causal
//! over tokens, so a stream re-renders its whole token prefix every `chunk` tokens and emits the
//! audio of all but the last `hold` tokens, crossfading `fade` samples into the previous render's
//! tail. The noise streams are keyed by frame, so re-renders of a prefix agree up to that
//! lookahead. A render sharing the GPU with the LM's back-to-back cooperative decode launches runs
//! several times slower, so the LM pauses while a batch holding a first chunk renders: first audio
//! is then prefill + `first` tokens + one uncontended render. The pause is the mux's downstream
//! urgency ([`DownstreamCredit::set_urgent`]).

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

use super::codec::Codec;
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

fn render_loop(vocoder: &Codec, sc: Schedule, rx: mpsc::Receiver<S3Msg>, credit: &DownstreamCredit) {
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
        due.truncate(max_batch);
        if due.is_empty() {
            continue;
        }
        let first = due.iter().any(|k| live[k].first_chunk());
        // Cleared on every exit, a panicking render included, so the LM never waits forever.
        struct Urgent<'a>(&'a DownstreamCredit);
        impl Drop for Urgent<'_> {
            fn drop(&mut self) {
                self.0.set_urgent(false);
            }
        }
        if first {
            credit.set_urgent(true);
        }
        let guard = Urgent(credit);
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

impl GuidedSpeech {
    /// Bind the packet's prompt tables and start the render thread. `credit` is the serving
    /// model's downstream credit: the vocoder's backlog gates the model's admission, and a render
    /// holding a first chunk holds its ticks.
    pub fn start(assets: &Path, credit: Arc<DownstreamCredit>) -> Result<Self> {
        let c = GuidedLmContract::load(assets)?
            .ok_or_else(|| RuntimeError::Rejected(format!("{} declares no guided LM pipeline", assets.display())))?;
        let tables = PromptTables::load(assets, c.hidden)?;
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
                let sc = match Schedule::from_codec(&vocoder) {
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
        seed: u64,
        class: JobClass,
        respond: crate::serve::stream::ChunkSender,
        request: &crate::serve::session::RequestIds,
        report: Option<crate::serve::session::Report>,
    ) -> Result<Job> {
        let c = &self.c;
        let ids = self.tables.text_ids(text)?;
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
        let job = self.job(&voice, &text, seed, class, respond, ids, report).map_err(|e| e.to_string())?;
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

    pub async fn synthesize(
        &self,
        mux: &ModelMux,
        voice: String,
        text: String,
        seed: u64,
        ids: &crate::serve::session::RequestIds,
        report: Option<crate::serve::session::Report>,
    ) -> std::result::Result<SpeechAudio, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.submit(mux, voice, text, seed, Reply::Whole(reply), ids, report)?;
        rx.await.map_err(|_| "chatterbox render dropped the request".to_string())?
    }

    pub fn synthesize_stream(
        &self,
        mux: &ModelMux,
        voice: String,
        text: String,
        seed: u64,
        ids: &crate::serve::session::RequestIds,
        report: Option<crate::serve::session::Report>,
    ) -> std::result::Result<tokio::sync::mpsc::UnboundedReceiver<StreamEvent>, String> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.submit(mux, voice, text, seed, Reply::Stream(tx), ids, report)?;
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
