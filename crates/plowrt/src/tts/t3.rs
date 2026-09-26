//! `tts.t3_cfg.v1` — Chatterbox T3, the speech-token stage of Chatterbox.
//!
//! The packet is a causal LM whose prefill rows are host embeddings (`overlay`) and whose decode
//! embedding adds a learned speech position from the per-slot `pos_base` (`EmbedPosBf16`).
//! Classifier-free guidance runs each request on TWO slots (conditional text / text embedding
//! zeroed); the host combines their logits, applies the repetition penalty, temperature and min_p
//! (an 8194-wide head, cheap on the host) and feeds the drawn token to both slots.
//!
//! Prefill rows, as the reference `T3.inference` builds them:
//! `voice rows (T3CondEnc) ++ [text_emb[id] + text_pos[i] for i, id in [SOT] + text + [EOT]] ++
//! BOS ++ BOS`, where BOS = speech_emb[start_speech] + speech_pos[0] (the reference appends BOS
//! twice) and the unconditional row zeroes text_emb but keeps text_pos. Decode token k gets
//! speech_pos[k + 1], so `pos_base = prefill_rows - 1`.

use std::collections::HashMap;
use std::path::Path;

use plow_asset::packet_pipeline::PacketPipeline;

use crate::exec::gpu::{GpuEngine, PrefillStep};
use crate::{Result, RuntimeError};

pub const DRIVER: &str = "tts.t3_cfg.v1";

#[derive(Clone, Debug, PartialEq)]
pub struct T3Contract {
    pub hidden: usize,
    pub overlay_rows: usize,
    pub decode_capacity: usize,
    pub start_text: u32,
    pub stop_text: u32,
    pub start_speech: u32,
    pub stop_speech: u32,
    pub text_vocab: usize,
    pub speech_vocab: usize,
    pub max_speech_tokens: usize,
    pub s3_valid_below: u32,
    pub cfg_weight: f32,
    pub temperature: f32,
    pub min_p: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl T3Contract {
    pub fn from_pipeline(p: &PacketPipeline) -> Result<Self> {
        if p.driver != DRIVER {
            return Err(RuntimeError::Rejected(format!("pipeline {} is {}, not {DRIVER}", p.name, p.driver)));
        }
        let get = |k: &str| {
            p.parameters
                .get(k)
                .copied()
                .ok_or_else(|| RuntimeError::Rejected(format!("T3 pipeline lacks parameter {k}")))
        };
        let f = |k: &str| get(k).map(|v| f32::from_bits(v as u32));
        Ok(T3Contract {
            hidden: get("hidden")? as usize,
            overlay_rows: get("overlay_rows")? as usize,
            decode_capacity: get("decode_capacity")? as usize,
            start_text: get("t3.start_text")? as u32,
            stop_text: get("t3.stop_text")? as u32,
            start_speech: get("t3.start_speech")? as u32,
            stop_speech: get("t3.stop_speech")? as u32,
            text_vocab: get("t3.text_vocab")? as usize,
            speech_vocab: get("t3.speech_vocab")? as usize,
            max_speech_tokens: get("t3.max_speech_tokens")? as usize,
            s3_valid_below: get("t3.s3_valid_below")? as u32,
            cfg_weight: f("t3.cfg_weight_f32")?,
            temperature: f("t3.temperature_f32")?,
            min_p: f("t3.min_p_f32")?,
            top_p: f("t3.top_p_f32")?,
            repetition_penalty: f("t3.repetition_penalty_f32")?,
        })
    }

    pub fn load(assets: &Path) -> Result<Option<Self>> {
        let asset = crate::exec::packet_runtime::PacketAsset::load(&assets.join("model.pkt"))?;
        let mut found = asset.pipelines().iter().filter(|p| p.driver == DRIVER);
        match (found.next(), found.next()) {
            (None, _) => Ok(None),
            (Some(p), None) => Self::from_pipeline(p).map(Some),
            (Some(_), Some(_)) => Err(RuntimeError::Rejected("ambiguous: several T3 pipelines".into())),
        }
    }
}

/// Little-endian f32s from bytes of any alignment (mmap'd safetensors data need not be aligned).
fn le_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Host tables for prefill rows (`in.prompt.*` packet tensors) and the packet's text rules.
pub struct T3Tables {
    hidden: usize,
    text_emb: Vec<f32>,
    text_pos: Vec<f32>,
    bos: Vec<f32>,
    bos_repeat: usize,
    uncond_drops_text: bool,
    voices: HashMap<String, Vec<f32>>,
    rules: String,
    tokenizer: tokenizers::Tokenizer,
}

impl T3Tables {
    pub fn load(assets: &Path, hidden: usize) -> Result<Self> {
        let path = assets.join("model.pkt");
        let raw = std::fs::read(&path).map_err(|source| RuntimeError::Io { path: path.clone(), source })?;
        let blob = crate::asset::devblob::DevBlob::parse(&raw)?;
        let host = |name: &str| -> Option<Vec<f32>> {
            let t = blob.tensors.iter().find(|t| t.name == name)?;
            Some(le_f32s(&blob.init[t.init.clone()?]))
        };
        let need = |name: &str| host(name).ok_or_else(|| RuntimeError::Rejected(format!("packet lacks {name}")));
        let (text_emb, text_pos, bos) = (need("in.prompt.text_table")?, need("in.prompt.text_pos")?, need("in.prompt.bos_row")?);
        if bos.len() != hidden || text_emb.len() % hidden != 0 || text_pos.len() % hidden != 0 {
            return Err(RuntimeError::Rejected("T3 host tables disagree with hidden".into()));
        }
        let mut voices = HashMap::new();
        for t in blob.tensors.iter().filter(|t| t.name.starts_with("in.prompt.voice.")) {
            let rows = host(&t.name).unwrap_or_default();
            if rows.is_empty() || rows.len() % hidden != 0 {
                return Err(RuntimeError::Rejected(format!("{}: not [rows][{hidden}] f32", t.name)));
            }
            voices.insert(t.name["in.prompt.voice.".len()..].to_string(), rows);
        }
        let asset = crate::exec::packet_runtime::PacketAsset::load(&path)?;
        let pipe = asset
            .pipelines()
            .iter()
            .find(|p| p.driver == DRIVER)
            .ok_or_else(|| RuntimeError::Rejected(format!("packet has no {DRIVER} pipeline")))?;
        let rules = pipe.strings.get("text.rules").cloned().unwrap_or_default();
        crate::text::rules::validate(&rules)?;
        let tk = assets.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tk)
            .map_err(|e| RuntimeError::Rejected(format!("{}: {e}", tk.display())))?;
        Ok(T3Tables {
            hidden,
            text_emb,
            text_pos,
            bos,
            bos_repeat: pipe.parameters.get("prompt.bos_repeat").copied().unwrap_or(1) as usize,
            uncond_drops_text: pipe.parameters.get("cfg.uncond_drops_text").copied().unwrap_or(0) == 1,
            voices,
            rules,
            tokenizer,
        })
    }

    pub fn voices(&self) -> impl Iterator<Item = &str> {
        self.voices.keys().map(String::as_str)
    }

    /// Tokenize after the packet's text rules.
    pub fn text_ids(&self, text: &str) -> Result<Vec<u32>> {
        let t = crate::text::rules::apply(&self.rules, text)?;
        let enc = self.tokenizer.encode(t, true).map_err(|e| RuntimeError::Rejected(format!("tokenize: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// The prefill rows ([rows][hidden] f32) for one CFG member.
    pub fn prefill_rows(&self, c: &T3Contract, voice: &str, text_ids: &[u32], uncond: bool) -> Result<Vec<f32>> {
        let h = self.hidden;
        let cond = self.voices.get(voice).ok_or_else(|| RuntimeError::Rejected(format!("unknown voice {voice:?}")))?;
        let ids: Vec<u32> = std::iter::once(c.start_text).chain(text_ids.iter().copied()).chain(std::iter::once(c.stop_text)).collect();
        if ids.len() * h > self.text_pos.len() || ids.iter().any(|&i| i as usize >= c.text_vocab) {
            return Err(RuntimeError::Rejected("text exceeds the T3 text tables".into()));
        }
        let mut rows = Vec::with_capacity(cond.len() + (ids.len() + self.bos_repeat) * h);
        rows.extend_from_slice(cond);
        for (i, &id) in ids.iter().enumerate() {
            let pos = &self.text_pos[i * h..(i + 1) * h];
            if uncond && self.uncond_drops_text {
                rows.extend_from_slice(pos);
            } else {
                let e = &self.text_emb[id as usize * h..(id as usize + 1) * h];
                rows.extend(e.iter().zip(pos).map(|(a, b)| a + b));
            }
        }
        for _ in 0..self.bos_repeat {
            rows.extend_from_slice(&self.bos);
        }
        Ok(rows)
    }
}

/// Deterministic per-request draws (splitmix64).
#[derive(Clone)]
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// The reference sampling chain over one CFG pair: cond + w(cond - uncond), repetition penalty
/// over the sequence so far (BOS included, as the reference's `generated_ids`), temperature,
/// min_p, top_p, then a draw. `u` in [0,1); `None` = greedy (argmax of the guided logits).
pub fn sample_cfg(c: &T3Contract, cond: &[f32], uncond: &[f32], history: &[u32], u: Option<f32>, scratch: &mut Vec<f32>) -> u32 {
    scratch.clear();
    scratch.extend(cond.iter().zip(uncond).map(|(a, b)| a + c.cfg_weight * (a - b)));
    let Some(u) = u else {
        return argmax(scratch);
    };
    let p = c.repetition_penalty;
    if p != 1.0 {
        for &t in history {
            if let Some(x) = scratch.get_mut(t as usize) {
                *x = if *x < 0.0 { *x * p } else { *x / p };
            }
        }
    }
    let inv_t = 1.0 / c.temperature;
    let m = scratch.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f32;
    for x in scratch.iter_mut() {
        *x = ((*x - m) * inv_t).exp();
        total += *x;
    }
    // min_p over probabilities == min_p over unnormalised weights (the max weight is 1).
    let floor = c.min_p;
    let mut kept = 0.0f32;
    for x in scratch.iter_mut() {
        if *x < floor {
            *x = 0.0;
        } else {
            kept += *x;
        }
    }
    if c.top_p < 1.0 {
        let mut order: Vec<usize> = (0..scratch.len()).filter(|&i| scratch[i] > 0.0).collect();
        order.sort_by(|&a, &b| scratch[b].total_cmp(&scratch[a]));
        let (mut acc, mut cut) = (0.0f32, order.len());
        for (k, &i) in order.iter().enumerate() {
            acc += scratch[i];
            if acc >= c.top_p * kept {
                cut = k + 1;
                break;
            }
        }
        for &i in &order[cut..] {
            kept -= scratch[i];
            scratch[i] = 0.0;
        }
    }
    let _ = total;
    let target = u * kept;
    let mut acc = 0.0f32;
    for (i, &x) in scratch.iter().enumerate() {
        acc += x;
        if x > 0.0 && acc > target {
            return i as u32;
        }
    }
    argmax(scratch)
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// One request in flight on a slot pair.
pub struct T3Job {
    pub voice: String,
    pub text: String,
    /// `None` = greedy guided decoding (the numerics gate); `Some(seed)` = the sampling chain.
    pub seed: Option<u64>,
    pub max_tokens: Option<usize>,
}

struct Active {
    job_index: usize,
    cond: usize,
    uncond: usize,
    history: Vec<u32>,
    out: Vec<u32>,
    rng: Rng,
    greedy: bool,
    max: usize,
    last: u32,
}

/// T3 over a `GpuEngine` it owns: prefill and CFG decode for a batch of jobs, continuous over
/// slot pairs (a finished pair is refilled from the pending queue before the next step).
pub struct T3Engine {
    pub e: GpuEngine,
    pub c: T3Contract,
    pub tables: T3Tables,
    logits: Vec<f32>,
    uncond: Vec<f32>,
    scratch: Vec<f32>,
    raw: Vec<u8>,
}

/// Per-job result: speech tokens (stop excluded) and timing.
#[derive(Debug, Clone, Default)]
pub struct T3Output {
    pub tokens: Vec<u32>,
    pub prefill_us: u64,
    pub decode_us: u64,
    pub steps: usize,
}

impl T3Engine {
    pub fn load(assets: &Path, device: u8) -> Result<Self> {
        let c = T3Contract::load(assets)?
            .ok_or_else(|| RuntimeError::Rejected(format!("{} declares no {DRIVER} pipeline", assets.display())))?;
        let be = std::sync::Arc::new(crate::device::cuda::CudaBackend::new(device)?);
        let e = GpuEngine::load(be, assets, &assets.join("checkpoint"))?;
        if e.vocab() != c.speech_vocab {
            return Err(RuntimeError::Rejected("T3 engine vocab disagrees with the contract".into()));
        }
        let tables = T3Tables::load(assets, c.hidden)?;
        Ok(T3Engine { e, c, tables, logits: Vec::new(), uncond: Vec::new(), scratch: Vec::new(), raw: Vec::new() })
    }

    pub fn pairs(&self) -> usize {
        self.e.batch() / 2
    }

    /// Prefill one CFG member on `slot`; returns its last-row logits in `self.logits`.
    fn prefill_member(&mut self, slot: usize, rows: &[f32]) -> Result<()> {
        let h = self.c.hidden;
        let n = rows.len() / h;
        if n > self.c.overlay_rows {
            return Err(RuntimeError::ContextLength(format!("T3 prompt {n} rows > overlay capacity {}", self.c.overlay_rows)));
        }
        self.e.begin_slot(slot, n + self.c.max_speech_tokens + 2)?;
        self.e.write_tensor("in.encoder_overlay", 0, bytemuck::cast_slice(rows))?;
        let index: Vec<u32> = (0..self.c.overlay_rows as u32).map(|i| if (i as usize) < n { i } else { u32::MAX }).collect();
        self.e.write_tensor("in.encoder_overlay_index", 0, bytemuck::cast_slice(&index))?;
        self.e.write_tensor("in.pos_base", (slot * 4) as u64, &((n - 1) as u32).to_le_bytes())?;
        let ids = vec![0u32; n];
        match self.e.prefill_chunk(slot, &ids, n)? {
            PrefillStep::Done(_) => {}
            PrefillStep::Progress(_) => return Err(RuntimeError::Rejected("T3 prefill must fit one chunk".into())),
        }
        self.read_logits_row(0)
    }

    fn read_logits_row(&mut self, row: usize) -> Result<()> {
        let v = self.c.speech_vocab;
        self.raw.resize(v * 2, 0);
        self.e.read_tensor_range("act.logits", (row * v * 2) as u64, &mut self.raw)?;
        self.logits.resize(v, 0.0);
        for (o, b) in self.logits.iter_mut().zip(self.raw.chunks_exact(2)) {
            *o = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
        }
        Ok(())
    }

    fn admit(&mut self, jobs: &[T3Job], index: usize, pair: usize, out: &mut [T3Output]) -> Result<Active> {
        let t0 = std::time::Instant::now();
        let job = &jobs[index];
        let ids = self.tables.text_ids(&job.text)?;
        let (cond, uncond) = (2 * pair, 2 * pair + 1);
        let rows = self.tables.prefill_rows(&self.c, &job.voice, &ids, true)?;
        self.prefill_member(uncond, &rows)?;
        std::mem::swap(&mut self.logits, &mut self.uncond);
        let rows = self.tables.prefill_rows(&self.c, &job.voice, &ids, false)?;
        self.prefill_member(cond, &rows)?;
        let greedy = job.seed.is_none();
        let mut a = Active {
            job_index: index,
            cond,
            uncond,
            history: vec![self.c.start_speech],
            out: Vec::new(),
            rng: Rng::new(job.seed.unwrap_or(0)),
            greedy,
            max: job.max_tokens.unwrap_or(self.c.max_speech_tokens).min(self.c.max_speech_tokens),
            last: 0,
        };
        let u = (!greedy).then(|| a.rng.unit());
        a.last = sample_cfg(&self.c, &self.logits, &self.uncond, &a.history, u, &mut self.scratch);
        out[index].prefill_us = t0.elapsed().as_micros() as u64;
        Ok(a)
    }

    /// Numerics probe: text ids and the (cond, uncond) last-prefill logits on slot pair 0.
    pub fn probe_prefill(&mut self, voice: &str, text: &str) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>)> {
        let ids = self.tables.text_ids(text)?;
        let rows = self.tables.prefill_rows(&self.c, voice, &ids, false)?;
        self.prefill_member(0, &rows)?;
        let cond = self.logits.clone();
        let rows = self.tables.prefill_rows(&self.c, voice, &ids, true)?;
        self.prefill_member(1, &rows)?;
        Ok((ids, cond, self.logits.clone()))
    }

    /// Run every job to completion; `on_done(index, output)` fires as each finishes.
    pub fn run(&mut self, jobs: &[T3Job], on_done: impl FnMut(usize, T3Output)) -> Result<()> {
        let mut it = jobs.iter();
        self.serve(
            |_| {
                it.next().map(|j| T3Job {
                    voice: j.voice.clone(),
                    text: j.text.clone(),
                    seed: j.seed,
                    max_tokens: j.max_tokens,
                })
            },
            |_, _| {},
            on_done,
        )
    }

    /// Continuous batching over slot pairs. `next(block)` supplies the next job (arrival index =
    /// call order); it is polled while a pair is free and asked to BLOCK only when nothing is in
    /// flight, and `None` from a blocking call ends the loop. `on_done(index, output)` fires as
    /// each job finishes; `on_token(index, token)` fires for each speech token as it is committed.
    /// A job whose admission fails (unknown voice, text too long) is reported
    /// through `on_done` with no tokens and does not stop the loop.
    pub fn serve(
        &mut self,
        mut next: impl FnMut(bool) -> Option<T3Job>,
        mut on_token: impl FnMut(usize, u32),
        mut on_done: impl FnMut(usize, T3Output),
    ) -> Result<()> {
        let mut jobs: Vec<T3Job> = Vec::new();
        let mut outputs: Vec<T3Output> = Vec::new();
        let mut started: Vec<std::time::Instant> = Vec::new();
        let mut active: Vec<Option<Active>> = (0..self.pairs()).map(|_| None).collect();
        let mut toks = Vec::new();
        let mut feeds = Vec::new();
        let mut closed = false;
        loop {
            for pair in 0..active.len() {
                if active[pair].is_some() || closed {
                    continue;
                }
                let idle = active.iter().all(Option::is_none);
                let Some(job) = next(idle) else {
                    closed = idle;
                    break;
                };
                let index = jobs.len();
                jobs.push(job);
                outputs.push(T3Output::default());
                started.push(std::time::Instant::now());
                match self.admit(&jobs, index, pair, &mut outputs) {
                    Ok(a) => active[pair] = Some(a),
                    Err(e) => {
                        tracing::warn!(error = %e, "t3: request rejected at admission");
                        on_done(index, std::mem::take(&mut outputs[index]));
                    }
                }
            }
            for pair in 0..active.len() {
                let finished = active[pair].as_ref().is_some_and(|a| a.last == self.c.stop_speech || a.out.len() >= a.max);
                if finished {
                    let a = active[pair].take().expect("checked");
                    let mut o = std::mem::take(&mut outputs[a.job_index]);
                    o.tokens = a.out;
                    o.steps = o.tokens.len();
                    o.decode_us = started[a.job_index].elapsed().as_micros() as u64;
                    on_done(a.job_index, o);
                }
            }
            if active.iter().all(Option::is_none) {
                if closed {
                    return Ok(());
                }
                continue;
            }
            feeds.clear();
            for a in active.iter_mut().flatten() {
                a.out.push(a.last);
                a.history.push(a.last);
                on_token(a.job_index, a.last);
                feeds.push((a.cond, a.last));
                feeds.push((a.uncond, a.last));
            }
            self.e.step_slots(&feeds, &mut toks)?;
            for pair in 0..active.len() {
                let Some(a) = active[pair].as_ref() else { continue };
                if a.out.len() >= a.max {
                    continue;
                }
                let (cond, uncond) = (a.cond, a.uncond);
                self.read_logits_row(uncond)?;
                std::mem::swap(&mut self.logits, &mut self.uncond);
                self.read_logits_row(cond)?;
                let a = active[pair].as_mut().expect("checked");
                let u = (!a.greedy).then(|| a.rng.unit());
                a.last = sample_cfg(&self.c, &self.logits, &self.uncond, &a.history, u, &mut self.scratch);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> T3Contract {
        T3Contract {
            hidden: 4,
            overlay_rows: 8,
            decode_capacity: 2,
            start_text: 255,
            stop_text: 0,
            start_speech: 6561,
            stop_speech: 6562,
            text_vocab: 704,
            speech_vocab: 6,
            max_speech_tokens: 10,
            s3_valid_below: 6561,
            cfg_weight: 0.5,
            temperature: 0.8,
            min_p: 0.05,
            top_p: 1.0,
            repetition_penalty: 1.2,
        }
    }

    #[test]
    fn guided_greedy_and_sampling_respect_the_chain() {
        let c = contract();
        let cond = [1.0, 3.0, 2.0, -1.0, 0.0, 2.9];
        let uncond = [1.0, 3.0, 0.0, -1.0, 0.0, 3.0];
        let mut s = Vec::new();
        // guided = cond + 0.5*(cond - uncond): [1, 3, 3, -1, 0, 2.85] -> argmax first max = 1
        assert_eq!(sample_cfg(&c, &cond, &uncond, &[], None, &mut s), 1);
        // penalty(tok 1) then /0.8: weights exp((x - 3)/0.8) = [.082, .535, 1, .0067, .0235, .829];
        // min_p 0.05 keeps {0, 1, 2, 5}, and every kept token is reachable.
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..1000 {
            let t = sample_cfg(&c, &cond, &uncond, &[1], Some(i as f32 / 1000.0), &mut s);
            assert!([0, 1, 2, 5].contains(&t), "drew {t}");
            seen.insert(t);
        }
        assert_eq!(seen.into_iter().collect::<Vec<_>>(), [0, 1, 2, 5]);
    }
}
