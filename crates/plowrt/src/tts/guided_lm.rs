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
pub use crate::text::sample::{sample_cfg, CfgParams, SplitMix};
use crate::{Result, RuntimeError};

pub const DRIVER: &str = "tts.guided_lm.v1";

#[derive(Clone, Debug, PartialEq)]
pub struct GuidedLmContract {
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
    pub valid_below: u32,
    pub cfg_weight: f32,
    pub temperature: f32,
    pub min_p: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// Speech tokens at the end of an utterance whose audio is cut (the reference drops them).
    pub trim_tail_tokens: usize,
}

impl GuidedLmContract {
    pub fn from_pipeline(p: &PacketPipeline) -> Result<Self> {
        if p.driver != DRIVER {
            return Err(RuntimeError::Rejected(format!("pipeline {} is {}, not {DRIVER}", p.name, p.driver)));
        }
        let get = |k: &str| {
            p.parameters
                .get(k)
                .copied()
                .ok_or_else(|| RuntimeError::Rejected(format!("guided LM pipeline lacks parameter {k}")))
        };
        let f = |k: &str| get(k).map(|v| f32::from_bits(v as u32));
        Ok(GuidedLmContract {
            hidden: get("hidden")? as usize,
            overlay_rows: get("overlay_rows")? as usize,
            decode_capacity: get("decode_capacity")? as usize,
            start_text: get("lm.start_text")? as u32,
            stop_text: get("lm.stop_text")? as u32,
            start_speech: get("lm.start_speech")? as u32,
            stop_speech: get("lm.stop_speech")? as u32,
            text_vocab: get("lm.text_vocab")? as usize,
            speech_vocab: get("lm.speech_vocab")? as usize,
            max_speech_tokens: get("lm.max_speech_tokens")? as usize,
            valid_below: get("lm.valid_below")? as u32,
            cfg_weight: f("lm.cfg_weight_f32")?,
            temperature: f("lm.temperature_f32")?,
            min_p: f("lm.min_p_f32")?,
            top_p: f("lm.top_p_f32")?,
            repetition_penalty: f("lm.repetition_penalty_f32")?,
            trim_tail_tokens: p.parameters.get("lm.trim_tail_tokens").copied().unwrap_or(0) as usize,
        })
    }

    pub fn cfg(&self) -> CfgParams {
        CfgParams {
            cfg_weight: self.cfg_weight,
            temperature: self.temperature,
            min_p: self.min_p,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
        }
    }

    pub fn load(assets: &Path) -> Result<Option<Self>> {
        let Some(asset) = crate::exec::packet_runtime::PacketAsset::load_if_present(&assets.join("model.pkt"))? else {
            return Ok(None);
        };
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
pub struct PromptTables {
    hidden: usize,
    text_emb: Vec<f32>,
    text_pos: Vec<f32>,
    bos: Vec<f32>,
    bos_repeat: usize,
    uncond_drops_text: bool,
    voices: HashMap<String, Vec<f32>>,
    rules: crate::text::rules::TextRules,
    tokenizer: tokenizers::Tokenizer,
}

impl PromptTables {
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
        let tables = match blob.reserved_metadata(&raw, crate::text::rules::Tables::SECTION)? {
            Some(b) => crate::text::rules::Tables::parse(b)?,
            None => Default::default(),
        };
        const LANG: &str = "text.rules.lang.";
        let rules = crate::text::rules::TextRules::compile(
            pipe.strings.get("text.rules").map_or("", String::as_str),
            pipe.strings.iter().filter_map(|(k, v)| Some((k.strip_prefix(LANG)?, v.as_str()))),
            pipe.strings.get("text.default_language").map(String::as_str),
            &tables,
        )?;
        let tk = assets.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tk)
            .map_err(|e| RuntimeError::Rejected(format!("{}: {e}", tk.display())))?;
        Ok(PromptTables {
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

    /// The language a request selects (`None` for a packet without language selection).
    pub fn language(&self, lang: Option<&str>) -> Result<Option<String>> {
        self.rules.resolve(lang)
    }

    /// Tokenize after the packet's text rules; `lang` as resolved by [`Self::language`].
    pub fn text_ids(&self, text: &str, lang: Option<&str>) -> Result<Vec<u32>> {
        let t = self.rules.apply(text, lang)?;
        let enc = self.tokenizer.encode(t, true).map_err(|e| RuntimeError::Rejected(format!("tokenize: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// The prefill rows ([rows][hidden] f32) for one CFG member.
    pub fn prefill_rows(&self, c: &GuidedLmContract, voice: &str, text_ids: &[u32], uncond: bool) -> Result<Vec<f32>> {
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

/// T3 on a `GpuEngine` it owns, one CFG pair at a time: the numerics gate (`examples/t3_check`).
/// Serving runs T3 on the model's mux (`tts::guided_speech`).
pub struct GuidedLm {
    pub e: GpuEngine,
    pub c: GuidedLmContract,
    pub tables: PromptTables,
    logits: Vec<f32>,
    uncond: Vec<f32>,
    scratch: Vec<f32>,
    raw: Vec<u8>,
}

impl GuidedLm {
    pub fn load(assets: &Path, device: u8) -> Result<Self> {
        let c = GuidedLmContract::load(assets)?
            .ok_or_else(|| RuntimeError::Rejected(format!("{} declares no {DRIVER} pipeline", assets.display())))?;
        let be = std::sync::Arc::new(crate::device::cuda::CudaBackend::new(device)?);
        let e = GpuEngine::load(be, assets, &assets.join("checkpoint"))?;
        if e.vocab() != c.speech_vocab {
            return Err(RuntimeError::Rejected("T3 engine vocab disagrees with the contract".into()));
        }
        let tables = PromptTables::load(assets, c.hidden)?;
        Ok(GuidedLm { e, c, tables, logits: Vec::new(), uncond: Vec::new(), scratch: Vec::new(), raw: Vec::new() })
    }

    /// Prefill one CFG member on `slot`; returns its last-row logits in `self.logits`.
    fn prefill_member(&mut self, slot: usize, rows: &[f32]) -> Result<()> {
        let h = self.c.hidden;
        let n = rows.len() / h;
        if n > self.c.overlay_rows {
            return Err(RuntimeError::ContextLength(format!("T3 prompt {n} rows > overlay capacity {}", self.c.overlay_rows)));
        }
        self.e.begin_slot(slot, n + self.c.max_speech_tokens + 2)?;
        self.e.write_tensor("in.pos_base", (slot * 4) as u64, &((n - 1) as u32).to_le_bytes())?;
        let ids = vec![0u32; n];
        // Chunk-relative overlay rows, as the mux stages them: launch row r reads overlay row r.
        let window = self.c.overlay_rows.min(self.e.pf_max_rows().max(1));
        let mut c0 = 0;
        loop {
            let hi = n.min(c0 + window);
            self.e.write_tensor("in.encoder_overlay", 0, bytemuck::cast_slice(&rows[c0 * h..hi * h]))?;
            let index: Vec<u32> = (0..self.c.overlay_rows).map(|r| if c0 + r < hi { r as u32 } else { u32::MAX }).collect();
            self.e.write_tensor("in.encoder_overlay_index", 0, bytemuck::cast_slice(&index))?;
            match self.e.prefill_chunk(slot, &ids, n)? {
                PrefillStep::Done(_) => break,
                PrefillStep::Progress(p) => c0 = p,
            }
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

    /// Numerics probe: text ids and the (cond, uncond) last-prefill logits on slot pair 0.
    pub fn probe_prefill(&mut self, voice: &str, text: &str, lang: Option<&str>) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>)> {
        let lang = self.tables.language(lang)?;
        let ids = self.tables.text_ids(text, lang.as_deref())?;
        let rows = self.tables.prefill_rows(&self.c, voice, &ids, false)?;
        self.prefill_member(0, &rows)?;
        let cond = self.logits.clone();
        let rows = self.tables.prefill_rows(&self.c, voice, &ids, true)?;
        self.prefill_member(1, &rows)?;
        Ok((ids, cond, self.logits.clone()))
    }

    /// Greedy guided decoding of one request on slot pair 0: speech tokens, stop excluded.
    pub fn greedy(&mut self, voice: &str, text: &str, lang: Option<&str>, max_tokens: usize) -> Result<Vec<u32>> {
        let lang = self.tables.language(lang)?;
        let ids = self.tables.text_ids(text, lang.as_deref())?;
        let rows = self.tables.prefill_rows(&self.c, voice, &ids, true)?;
        self.prefill_member(1, &rows)?;
        std::mem::swap(&mut self.logits, &mut self.uncond);
        let rows = self.tables.prefill_rows(&self.c, voice, &ids, false)?;
        self.prefill_member(0, &rows)?;
        let p = self.c.cfg();
        let mut out = Vec::new();
        let mut toks = Vec::new();
        let mut last = sample_cfg(&p, &self.logits, &self.uncond, [], None, &mut self.scratch);
        while last != self.c.stop_speech && out.len() < max_tokens {
            out.push(last);
            self.e.step_slots(&[(0, last), (1, last)], &mut toks)?;
            self.read_logits_row(1)?;
            std::mem::swap(&mut self.logits, &mut self.uncond);
            self.read_logits_row(0)?;
            last = sample_cfg(&p, &self.logits, &self.uncond, [], None, &mut self.scratch);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> GuidedLmContract {
        GuidedLmContract {
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
            valid_below: 6561,
            cfg_weight: 0.5,
            temperature: 0.8,
            min_p: 0.05,
            top_p: 1.0,
            repetition_penalty: 1.2,
            trim_tail_tokens: 0,
        }
    }

    #[test]
    fn guided_greedy_and_sampling_respect_the_chain() {
        let c = contract().cfg();
        let cond = [1.0, 3.0, 2.0, -1.0, 0.0, 2.9];
        let uncond = [1.0, 3.0, 0.0, -1.0, 0.0, 3.0];
        let mut s = Vec::new();
        // guided = cond + 0.5*(cond - uncond): [1, 3, 3, -1, 0, 2.85] -> argmax first max = 1
        assert_eq!(sample_cfg(&c, &cond, &uncond, [], None, &mut s), 1);
        // penalty(tok 1) then /0.8: weights exp((x - 3)/0.8) = [.082, .535, 1, .0067, .0235, .829];
        // min_p 0.05 keeps {0, 1, 2, 5}, and every kept token is reachable.
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..1000 {
            let t = sample_cfg(&c, &cond, &uncond, [1], Some(i as f32 / 1000.0), &mut s);
            assert!([0, 1, 2, 5].contains(&t), "drew {t}");
            seen.insert(t);
        }
        assert_eq!(seen.into_iter().collect::<Vec<_>>(), [0, 1, 2, 5]);
    }

    /// The penalty applies once per distinct history token (HF gather / scatter), not per occurrence.
    #[test]
    fn repetition_penalty_once_per_distinct_token() {
        let c = contract().cfg();
        let cond = [1.0, 3.0, 2.0, -1.0, 0.0, 2.9];
        let uncond = [1.0, 3.0, 0.0, -1.0, 0.0, 3.0];
        let mut s = Vec::new();
        for i in 0..1000 {
            let u = Some(i as f32 / 1000.0);
            assert_eq!(sample_cfg(&c, &cond, &uncond, [1, 1, 1, 5, 5], u, &mut s), sample_cfg(&c, &cond, &uncond, [1, 5], u, &mut s));
        }
    }
}
