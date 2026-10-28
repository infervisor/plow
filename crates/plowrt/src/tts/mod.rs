//! Text-to-speech packet pipelines.
//!
//! The speech counterpart of [`crate::asr`]: the compiled asset declares the pipeline
//! (`packet_pipeline.json`, driver `tts.codec_lm.v1`, emitted by `plowc --tts-profile`), and the
//! runtime binds that declaration instead of knowing any checkpoint. A codec-token LM runs on the
//! served engine's continuous-batching mux like any causal model; the codes it emits are decoded
//! by the codec stage the contract names ([`codec`]). See docs/arch/24-tts-pipelines.md.

#[cfg(feature = "cuda")]
pub mod api;
#[cfg(feature = "cuda")]
pub mod codec;
#[cfg(feature = "cuda")]
pub mod guided_lm;
#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
pub mod guided_speech;
#[cfg(feature = "cuda")]
pub mod realtime;
#[cfg(feature = "cuda")]
pub mod ref_encoder;
#[cfg(feature = "cuda")]
pub mod serving;
#[cfg(feature = "cuda")]
pub mod text_encoder;
#[cfg(feature = "cuda")]
pub mod voice_cond;

pub(crate) mod stretch;

use std::path::Path;

use plow_asset::packet_pipeline::PacketPipeline;

use crate::{Result, RuntimeError};

pub const DRIVER: &str = "tts.codec_lm.v1";

/// The numeric contract of one `tts.codec_lm.v1` pipeline, read from packet metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeechContract {
    pub pipeline: String,
    /// Prompt text with `{voice}` and `{input}` placeholders.
    pub prompt_template: String,
    /// The voice's vocabulary token with a `{voice}` placeholder.
    pub voice_token: String,
    /// Named voices (`prompt.voices`, one per line); empty: a voice is known when its
    /// `voice_token` is one vocabulary token.
    pub voices: Vec<String>,
    pub prefix: Vec<u32>,
    pub suffix: Vec<u32>,
    pub stops: Vec<u32>,
    pub sample_rate: u32,
    pub frame_codes: usize,
    pub codebook: u32,
    pub frame_samples: usize,
    pub audio_token_base: u32,
    pub per_char_frames: f32,
    pub max_new_tokens_cap: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// Default repetition penalty; 1.0 (none) on packets without one.
    pub repetition_penalty: f32,
    /// Default top-k; 0 (none) on packets without one.
    pub top_k: usize,
    /// Frames come from the packet's depth stage: one LM token per frame (`audio_token_base +
    /// class`), its `frame_codes` codes riding beside it (the stream chunk's `frame`).
    pub depth: bool,
    /// The depth stage samples like Breeze (f32 parameters, backbone repetition penalty), not
    /// like Fish (bf16 parameters, RAS, no penalty).
    pub breeze_sampler: bool,
    /// Rendered into the template's `{instructions}` (`{instructions}` = the request's text);
    /// `None`: the model takes no instructions.
    pub instructions: Option<String>,
    /// Sub-bundle of the encoder-mode packet whose output rows are the prompt's embeddings.
    pub text_encoder: Option<String>,
    /// Voice cloning through a reference encoder packet beside the LM.
    pub clone: Option<CloneContract>,
}

/// A cloning prompt: `head`, the reference transcript's prompt text, `mid`, the reference frames
/// (one frame token each), an end frame whose every code is `eos_code` (if any), `tail`, then the
/// text's prompt text and the suffix. `head` and `tail` default to the prompt prefix.
#[derive(Clone, Debug, PartialEq)]
pub struct CloneContract {
    pub encoder: String,
    pub head: Vec<u32>,
    pub mid: Vec<u32>,
    pub tail: Vec<u32>,
    pub eos_code: Option<u32>,
}

impl SpeechContract {
    pub fn from_pipeline(p: &PacketPipeline) -> Result<Self> {
        if p.driver != DRIVER {
            return Err(RuntimeError::Rejected(format!(
                "pipeline {} is {}, not {DRIVER}",
                p.name, p.driver
            )));
        }
        let get = |k: &str| {
            p.parameters.get(k).copied().ok_or_else(|| {
                RuntimeError::Rejected(format!("speech pipeline lacks parameter {k}"))
            })
        };
        let list = |k: &str| -> Result<Vec<u32>> {
            (0..get(&format!("{k}.count"))?)
                .map(|i| get(&format!("{k}.{i}")).map(|v| v as u32))
                .collect()
        };
        let f32p = |k: &str| get(k).map(|v| f32::from_bits(v as u32));
        let mut c = SpeechContract {
            pipeline: p.name.clone(),
            prompt_template: p.strings.get("prompt.template").cloned().ok_or_else(|| {
                RuntimeError::Rejected("speech pipeline lacks prompt.template".into())
            })?,
            voice_token: p
                .strings
                .get("prompt.voice_token")
                .cloned()
                .ok_or_else(|| {
                    RuntimeError::Rejected("speech pipeline lacks prompt.voice_token".into())
                })?,
            voices: p
                .strings
                .get("prompt.voices")
                .map(|v| v.lines().map(str::to_owned).collect())
                .unwrap_or_default(),
            prefix: list("prompt.prefix")?,
            suffix: list("prompt.suffix")?,
            stops: list("stop")?,
            sample_rate: get("audio.sample_rate")? as u32,
            frame_codes: get("codec.frame_codes")? as usize,
            codebook: get("codec.codebook")? as u32,
            frame_samples: get("codec.frame_samples")? as usize,
            audio_token_base: get("audio.token_base")? as u32,
            per_char_frames: f32p("tokens.per_char_frames_f32")?,
            max_new_tokens_cap: get("tokens.max_new_cap")? as usize,
            temperature: f32p("sampling.temperature_f32")?,
            top_p: f32p("sampling.top_p_f32")?,
            repetition_penalty: f32p("sampling.repetition_penalty_f32").unwrap_or(1.0),
            top_k: get("sampling.top_k").unwrap_or(0) as usize,
            depth: get("frames.depth").unwrap_or(0) != 0,
            breeze_sampler: get("frames.depth_sampler").unwrap_or(0) == 1,
            instructions: p.strings.get("prompt.instructions").cloned(),
            text_encoder: p.strings.get("text.encoder").cloned(),
            clone: None,
        };
        if let Some(encoder) = p.strings.get("clone.encoder") {
            let listed = |k: &str, or: &[u32]| -> Result<Vec<u32>> {
                match p.strings.get(k) {
                    Some(ids) => ids
                        .split_ascii_whitespace()
                        .map(|id| {
                            id.parse().map_err(|_| {
                                RuntimeError::Rejected(format!(
                                    "speech pipeline {k}: bad id {id:?}"
                                ))
                            })
                        })
                        .collect(),
                    None => Ok(or.to_vec()),
                }
            };
            c.clone = Some(CloneContract {
                encoder: encoder.clone(),
                head: listed("clone.head", &c.prefix)?,
                mid: listed("clone.mid", &[])?,
                tail: listed("clone.tail", &c.prefix)?,
                eos_code: p.parameters.get("clone.eos_code").map(|&v| v as u32),
            });
        }
        if !c.prompt_template.contains("{input}") || !c.voice_token.contains("{voice}") {
            return Err(RuntimeError::Rejected(
                "speech prompt templates need {input} and {voice}".into(),
            ));
        }
        Ok(c)
    }

    /// The single speech pipeline of the bundle's packet, or `None` for a non-speech packet.
    pub fn load(assets: &Path) -> Result<Option<Self>> {
        let Some(pkt) = crate::asset::devblob::DevBlob::find_in_dir(assets)? else {
            return Ok(None);
        };
        let Some(asset) = crate::exec::packet_runtime::PacketAsset::load_if_present(&pkt)? else {
            return Ok(None);
        };
        let mut found = asset.pipelines().iter().filter(|p| p.driver == DRIVER);
        match (found.next(), found.next()) {
            (None, _) => Ok(None),
            (Some(p), None) => Self::from_pipeline(p).map(Some),
            (Some(_), Some(_)) => Err(RuntimeError::Rejected(
                "ambiguous: several speech pipelines".into(),
            )),
        }
    }

    /// The packet's prompt template; the tokenizer encodes it between prefix and suffix.
    pub fn prompt_text(&self, voice: &str, instructions: Option<&str>, input: &str) -> String {
        let ins = match (&self.instructions, instructions) {
            (Some(t), Some(i)) => t.replace("{instructions}", i),
            _ => String::new(),
        };
        self.prompt_template
            .replace("{voice}", voice)
            .replace("{instructions}", &ins)
            .replace("{input}", input)
    }

    /// The voice's token; it must be one vocabulary token, which is how a voice is known to exist.
    pub fn voice_token(&self, voice: &str) -> String {
        self.voice_token.replace("{voice}", voice)
    }

    pub fn max_new_tokens(&self, input: &str) -> usize {
        let frames = (input.chars().count() as f32 * self.per_char_frames) as usize;
        (frames * self.tokens_per_frame() + 21).min(self.max_new_tokens_cap)
    }

    /// Longest input whose [`Self::max_new_tokens`] the cap does not clip; a longer one is spoken
    /// as [`segments`] of at most this many characters.
    pub fn segment_chars(&self) -> usize {
        let frames = self.max_new_tokens_cap.saturating_sub(21) / self.tokens_per_frame();
        let fits = |c: usize| (c as f32 * self.per_char_frames) as usize <= frames;
        let mut c = (frames as f32 / self.per_char_frames) as usize;
        while c > 1 && !fits(c) {
            c -= 1;
        }
        while c < MAX_INPUT_CHARS && fits(c + 1) {
            c += 1;
        }
        c.max(1)
    }

    /// What a client may send this model ([`api::SpeechSpec`]). Without `prompt.voices`, the
    /// voices are the tokenizer's special tokens that fill `voice_token`.
    #[cfg(feature = "cuda")]
    pub fn spec(&self, tok: &dyn crate::text::tokenizer::Tokenize) -> api::SpeechSpec {
        let mut voices = if self.voices.is_empty() {
            let (pre, post) = self
                .voice_token
                .split_once("{voice}")
                .unwrap_or((&self.voice_token, ""));
            tok.special_tokens()
                .iter()
                .filter_map(|t| t.strip_prefix(pre)?.strip_suffix(post))
                .filter(|v| !v.is_empty())
                .map(str::to_owned)
                .collect()
        } else {
            self.voices.clone()
        };
        voices.sort_unstable();
        voices.dedup();
        let mut spec = api::SpeechSpec {
            voices,
            languages: Vec::new(),
            default_language: None,
            sample_rate: self.sample_rate,
            temperature: self.temperature,
            top_p: self.top_p,
            // Both depth samplers take at most 64 candidates; Fish's (SampleFishBf16) has no penalty.
            repetition_penalty: (!self.depth || self.breeze_sampler)
                .then_some(self.repetition_penalty),
            max_tokens: self.max_new_tokens_cap,
            top_k: Some(self.top_k),
            top_k_max: self.depth.then_some(64),
            // Breeze guides against the uninstructed prompt: w = s - 1, off by default.
            cfg_weight: self.depth_cfg().then_some(0.0),
            instructions: self.instructions.is_some(),
            voice_cloning: self.clone.is_some(),
            clone_transcript: true,
        };
        spec.narrow_voices(&crate::config::RuntimeConfig::get().tts_voices);
        spec
    }

    /// Classifier-free guidance in the depth stage, contrasting with the prompt without its
    /// instructions.
    pub fn depth_cfg(&self) -> bool {
        self.breeze_sampler && self.instructions.is_some()
    }

    /// LM tokens per audio frame.
    pub fn tokens_per_frame(&self) -> usize {
        if self.depth {
            1
        } else {
            self.frame_codes.max(1)
        }
    }

    /// Append the codes LM token `id` (with its depth-stage `frame`) carries; true when they
    /// complete a frame.
    pub fn push_codes(&self, codes: &mut Vec<i32>, id: u32, frame: Option<&[u32]>) -> bool {
        if self.depth {
            return match frame {
                Some(f) if f.len() == self.frame_codes => {
                    codes.extend(f.iter().map(|&x| x as i32));
                    true
                }
                _ => false,
            };
        }
        match self.code_of(codes.len(), id) {
            Some(code) => {
                codes.push(code);
                codes.len() % self.frame_codes == 0
            }
            None => false,
        }
    }

    /// Codebook id of the `n`-th kept audio token, or `None` when `tok` is not the code its frame
    /// position expects (dropped, as the reference decoder drops out-of-range ids).
    pub fn code_of(&self, n: usize, tok: u32) -> Option<i32> {
        let lo = self.audio_token_base + (n % self.frame_codes) as u32 * self.codebook;
        (lo..lo + self.codebook)
            .contains(&tok)
            .then(|| (tok - lo) as i32)
    }
}

/// Longest `input` a speech request takes, in characters (400 beyond): about 7 minutes of English
/// audio, spoken as [`segments`].
pub const MAX_INPUT_CHARS: usize = 4096;

/// `text` as consecutive slices of at most `max` characters, outer whitespace trimmed: whole
/// sentences packed greedily; a longer sentence splits after clause marks, then between words,
/// then anywhere. A text of at most `max` characters is the one segment, unchanged.
pub fn segments(text: &str, max: usize) -> Vec<&str> {
    let max = max.max(1);
    if text.chars().count() <= max {
        return vec![text];
    }
    let mut pieces = Vec::new();
    for r in cuts(text, (0, text.len()), Cut::Sentence) {
        split_piece(text, r, max, Cut::Clause, &mut pieces);
    }
    let mut out = Vec::new();
    let mut cur: Option<(usize, usize)> = None;
    for (s, e) in pieces {
        cur = match cur {
            Some((cs, _)) if text[cs..e].chars().count() <= max => Some((cs, e)),
            Some((cs, ce)) => {
                out.push(&text[cs..ce]);
                Some((s, e))
            }
            None => Some((s, e)),
        };
    }
    out.extend(cur.map(|(s, e)| &text[s..e]));
    out
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
enum Cut {
    Sentence,
    Clause,
    Word,
    Char,
}

fn split_piece(text: &str, r: (usize, usize), max: usize, cut: Cut, out: &mut Vec<(usize, usize)>) {
    if text[r.0..r.1].chars().count() <= max {
        return out.push(r);
    }
    if cut == Cut::Char {
        let mut s = r.0;
        for (n, (i, _)) in text[r.0..r.1].char_indices().enumerate() {
            if n > 0 && n % max == 0 {
                out.push((s, r.0 + i));
                s = r.0 + i;
            }
        }
        return out.push((s, r.1));
    }
    let next = if cut == Cut::Clause {
        Cut::Word
    } else {
        Cut::Char
    };
    let mut parts = Vec::new();
    for sub in cuts(text, r, cut) {
        split_piece(text, sub, max, next, &mut parts);
    }
    balance(text, &parts, max, out);
}

/// `parts` (consecutive, each at most `max` characters) packed into about equal groups of at
/// most `max`: greedy packing left a sentence's last word or two as a segment of its own, and
/// Orpheus goes mute on such a fragment ("busy.") whatever the seed.
fn balance(text: &str, parts: &[(usize, usize)], max: usize, out: &mut Vec<(usize, usize)>) {
    let (Some(&(s0, _)), Some(&(_, e0))) = (parts.first(), parts.last()) else {
        return;
    };
    let total = text[s0..e0].chars().count();
    let groups = total.div_ceil(max);
    let target = total.div_ceil(groups);
    let mut cur: Option<(usize, usize)> = None;
    for &(s, e) in parts {
        cur = match cur {
            Some((cs, ce))
                if text[cs..ce].chars().count() >= target || text[cs..e].chars().count() > max =>
            {
                out.push((cs, ce));
                Some((s, e))
            }
            Some((cs, _)) => Some((cs, e)),
            None => Some((s, e)),
        };
    }
    out.extend(cur);
}

/// `r` split after each `cut` boundary, every piece trimmed and non-empty.
fn cuts(text: &str, r: (usize, usize), cut: Cut) -> Vec<(usize, usize)> {
    let s = &text[r.0..r.1];
    let mut out = Vec::new();
    let mut push = |a: usize, b: usize| {
        let p = &text[a..b];
        let lead = p.len() - p.trim_start().len();
        let t = p.trim();
        if !t.is_empty() {
            out.push((a + lead, a + lead + t.len()));
        }
    };
    let mut start = r.0;
    let mut it = s.char_indices().peekable();
    while let Some((_, c)) = it.next() {
        let end = match cut {
            Cut::Sentence if matches!(c, '\n' | '।' | '॥' | '。' | '！' | '？') => true,
            Cut::Sentence if matches!(c, '.' | '!' | '?' | '…') => {
                while let Some(&(_, n)) = it.peek() {
                    if !matches!(
                        n,
                        '.' | '!' | '?' | '…' | '"' | '\'' | '”' | '’' | ')' | ']' | '»'
                    ) {
                        break;
                    }
                    it.next();
                }
                it.peek().is_none_or(|&(_, n)| n.is_whitespace())
            }
            Cut::Clause if matches!(c, ',' | ';' | ':' | '—' | '–' | '،' | '、' | '，') => {
                it.peek().is_none_or(|&(_, n)| n.is_whitespace())
            }
            Cut::Word => c.is_whitespace(),
            _ => false,
        };
        if end {
            let b = it.peek().map_or(r.1, |&(j, _)| r.0 + j);
            push(start, b);
            start = b;
        }
    }
    push(start, r.1);
    out
}

/// Silence across the joins of a multi-segment utterance. A segment can open with seconds of
/// silence (Orpheus) or run on silent to its budget, so a run of silent blocks past `KEEP_S` is
/// held back: speech resuming within the segment sends it (a real pause); at a join only enough to
/// make a `JOIN_PAUSE_S` pause is sent; a run reaching `RUNAWAY_S` ends the segment, and a segment
/// silent for `RUNAWAY_S` from its start is mute (retried). A segment that drones (a sustained hum
/// or "rrrr" with no syllable modulation for `DRONE_S`) is retried too. Silence is held only while
/// the client has more than `HOLD_LEAD_S` of audio beyond it.
pub(crate) struct Joins {
    rate: f64,
    /// Silent samples of the current run sent, and held back.
    sent: usize,
    held: Vec<f32>,
    /// The current segment has spoken; the first segment's leading silence is never held.
    speech: bool,
    first: bool,
    /// Silent samples since the segment started, until it speaks; and its sounding samples.
    quiet: usize,
    loud: usize,
    /// The segment's envelope over the last `DRONE_S` (RMS per `ENV_BLOCKS` blocks), the block
    /// energy accumulating toward the next point, and whether it has droned.
    env: std::collections::VecDeque<f32>,
    acc: (f32, usize),
    droned: bool,
}

impl Joins {
    const BLOCK: usize = 256;
    /// RMS under -45 dBFS.
    const SILENCE: f32 = 0.0056;
    const KEEP_S: f64 = 0.3;
    const JOIN_PAUSE_S: f64 = 0.5;
    const RUNAWAY_S: f64 = 3.0;
    const HOLD_LEAD_S: f64 = 1.5;
    /// Envelope points of ~53 ms; a drone is `DRONE_S` of sound whose envelope varies less than
    /// `DRONE_CV` (std / mean). Orpheus speech windows measure >= 0.36, its "rrrr" drone 0.01-0.02;
    /// genuine leading silence runs to 2.9 s, so mute detection stays at `RUNAWAY_S`.
    const ENV_BLOCKS: usize = 5;
    const DRONE_S: f64 = 2.0;
    const DRONE_CV: f32 = 0.15;

    pub(crate) fn new(sample_rate: u32) -> Self {
        Joins {
            rate: f64::from(sample_rate),
            sent: 0,
            held: Vec::new(),
            speech: false,
            first: true,
            quiet: 0,
            loud: 0,
            env: Default::default(),
            acc: (0.0, 0),
            droned: false,
        }
    }

    /// The next segment starts: the previous one's held silence is dropped.
    pub(crate) fn next_segment(&mut self) {
        self.held.clear();
        self.speech = false;
        self.first = false;
        self.quiet = 0;
        self.loud = 0;
        self.env.clear();
        self.acc = (0.0, 0);
        self.droned = false;
    }

    /// Append to `out` what of `pcm` to send now; `lead_s` is the audio the client has buffered.
    pub(crate) fn push(&mut self, pcm: &[f32], lead_s: f64, out: &mut Vec<f32>) {
        let rate = self.rate;
        let samples = |s: f64| (s * rate) as usize;
        for b in pcm.chunks(Self::BLOCK) {
            let ms = b.iter().map(|x| x * x).sum::<f32>() / b.len() as f32;
            self.envelope(ms);
            let rms = ms.sqrt();
            if rms >= Self::SILENCE {
                self.loud += b.len();
                let room = if self.speech || self.first {
                    self.held.len()
                } else {
                    samples(Self::JOIN_PAUSE_S).saturating_sub(self.sent)
                };
                out.extend_from_slice(&self.held[self.held.len().saturating_sub(room)..]);
                self.held.clear();
                self.sent = 0;
                self.speech = true;
                out.extend_from_slice(b);
                continue;
            }
            if !self.speech {
                self.quiet += b.len();
            }
            let held_s = (self.held.len() + b.len()) as f64 / self.rate;
            let hold = (self.speech || !self.first)
                && self.sent >= samples(Self::KEEP_S)
                && lead_s - held_s > Self::HOLD_LEAD_S;
            if hold {
                self.held.extend_from_slice(b);
            } else {
                self.sent += self.held.len() + b.len();
                out.append(&mut self.held);
                out.extend_from_slice(b);
            }
        }
    }

    fn envelope(&mut self, mean_square: f32) {
        self.acc = (self.acc.0 + mean_square, self.acc.1 + 1);
        if self.acc.1 < Self::ENV_BLOCKS {
            return;
        }
        self.env.push_back((self.acc.0 / self.acc.1 as f32).sqrt());
        self.acc = (0.0, 0);
        let points = (Self::DRONE_S * self.rate) as usize / (Self::BLOCK * Self::ENV_BLOCKS);
        while self.env.len() > points {
            self.env.pop_front();
        }
        if self.env.len() == points && self.env.iter().all(|&e| e >= Self::SILENCE) {
            let mean = self.env.iter().sum::<f32>() / points as f32;
            let var = self
                .env
                .iter()
                .map(|e| (e - mean) * (e - mean))
                .sum::<f32>()
                / points as f32;
            self.droned |= var.sqrt() < Self::DRONE_CV * mean;
        }
    }

    /// The current segment has droned for `DRONE_S`.
    pub(crate) fn drone(&self) -> bool {
        self.droned
    }

    /// The current segment, pushed whole, held under a quarter second of sound or droned.
    pub(crate) fn segment_failed(&self) -> bool {
        (self.loud as f64) < 0.25 * self.rate || self.droned
    }

    /// Whether `pcm` (one segment) is mute or drones: generate it again.
    pub(crate) fn failed(&self, pcm: &[f32]) -> bool {
        let mut probe = Joins::new(self.rate as u32);
        probe.push(pcm, f64::INFINITY, &mut Vec::new());
        !self.speaks(pcm) || probe.drone()
    }

    /// The current segment has spoken and then stayed silent for `RUNAWAY_S`.
    pub(crate) fn runaway(&self) -> bool {
        self.speech && (self.sent + self.held.len()) as f64 >= Self::RUNAWAY_S * self.rate
    }

    /// The current segment has been silent for `RUNAWAY_S` since it started.
    pub(crate) fn mute(&self) -> bool {
        !self.speech && self.quiet as f64 >= Self::RUNAWAY_S * self.rate
    }

    /// Whether `pcm` holds a quarter second of sound.
    pub(crate) fn speaks(&self, pcm: &[f32]) -> bool {
        let loud = pcm.chunks(Self::BLOCK).filter(|b| {
            (b.iter().map(|x| x * x).sum::<f32>() / b.len() as f32).sqrt() >= Self::SILENCE
        });
        loud.map(<[f32]>::len).sum::<usize>() as f64 >= 0.25 * self.rate
    }
}

pub(crate) fn pcm16(samples: &[f32], out: &mut Vec<u8>) {
    out.reserve(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
}

/// RIFF/WAVE header for mono s16; `data_bytes = u32::MAX` for an open-ended stream.
pub(crate) fn wav_header(sample_rate: u32, data_bytes: u32) -> Vec<u8> {
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&data_bytes.saturating_add(36).to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes());
    h.extend_from_slice(&sample_rate.to_le_bytes());
    h.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    h.extend_from_slice(&2u16.to_le_bytes());
    h.extend_from_slice(&16u16.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_bytes.to_le_bytes());
    h
}

/// A speech reply's output stage: the model's PCM at its own rate, written as s16le mono at the
/// reply's. Fed in pieces, so one instance serves a whole stream — the resampler is the ASR
/// frontend's, whose polyphase output does not depend on where the piece boundaries fall.
pub(crate) struct OutPcm {
    wav: bool,
    rate: u32,
    speed: f32,
    /// `None` at the model's own speed; stretching runs before resampling, at the model's rate.
    stretch: Option<stretch::Stretch>,
    /// `None` when the reply is served at the model's own rate.
    resampler: Option<crate::asr::frontend::Resampler>,
}

impl OutPcm {
    pub fn new(from: u32, to: u32, speed: f32, wav: bool) -> std::result::Result<Self, String> {
        let stretch = (speed != 1.0)
            .then(|| stretch::Stretch::new(from, speed))
            .transpose()?;
        let resampler = (from != to)
            .then(|| crate::asr::frontend::Resampler::to(from, to))
            .transpose()?;
        Ok(Self {
            wav,
            rate: to,
            speed,
            stretch,
            resampler,
        })
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// How much faster the reply plays than the model synthesized it: one model-rate sample buys
    /// `1 / (rate * speed)` seconds of playback.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    pub fn content_type(&self) -> &'static str {
        if self.wav {
            "audio/wav"
        } else {
            "audio/pcm"
        }
    }

    /// The WAV header for `data_bytes` of audio; `None` on a raw `pcm` reply.
    pub fn header(&self, data_bytes: u32) -> Option<Vec<u8>> {
        self.wav.then(|| wav_header(self.rate, data_bytes))
    }

    pub fn push(&mut self, pcm: &[f32], out: &mut Vec<u8>) {
        let stretched = self.stretch.as_mut().map(|s| s.push(pcm));
        let pcm = stretched.as_deref().unwrap_or(pcm);
        match &mut self.resampler {
            Some(r) => pcm16(&r.push(pcm), out),
            None => pcm16(pcm, out),
        }
    }

    /// The stretch's last frame and the resampler's tail, neither of which a `push` could
    /// complete: dropping them would clip the end of every stretched or resampled reply.
    pub fn finish(&mut self, out: &mut Vec<u8>) {
        let tail = self
            .stretch
            .as_mut()
            .map(|s| s.finish())
            .unwrap_or_default();
        match &mut self.resampler {
            Some(r) => {
                let mut v = r.push(&tail);
                v.extend(r.finish());
                pcm16(&v, out);
            }
            None => pcm16(&tail, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn pipeline() -> PacketPipeline {
        let mut p: BTreeMap<String, u64> = BTreeMap::new();
        for (k, v) in [
            ("prompt.prefix.count", 1),
            ("prompt.prefix.0", 128259),
            ("prompt.suffix.count", 3),
            ("prompt.suffix.0", 128260),
            ("prompt.suffix.1", 128261),
            ("prompt.suffix.2", 128257),
            ("stop.count", 2),
            ("stop.0", 128258),
            ("stop.1", 128262),
            ("audio.sample_rate", 24000),
            ("codec.frame_codes", 7),
            ("codec.codebook", 4096),
            ("codec.frame_samples", 2048),
            ("audio.token_base", 128266),
            ("tokens.per_char_frames_f32", u64::from(1.3f32.to_bits())),
            ("tokens.max_new_cap", 700),
            ("sampling.temperature_f32", u64::from(0.4f32.to_bits())),
            ("sampling.top_p_f32", u64::from(0.9f32.to_bits())),
        ] {
            p.insert(k.into(), v);
        }
        PacketPipeline {
            strings: BTreeMap::from([
                ("prompt.template".into(), "<spk_{voice}> {input}".into()),
                ("prompt.voice_token".into(), "<spk_{voice}>".into()),
            ]),
            name: "speech".into(),
            driver: DRIVER.into(),
            programs: BTreeMap::new(),
            tensors: BTreeMap::new(),
            parameters: p,
        }
    }

    #[test]
    fn binds_contract_from_pipeline_parameters() {
        let c = SpeechContract::from_pipeline(&pipeline()).unwrap();
        assert_eq!(c.prefix, [128259]);
        assert_eq!(c.suffix, [128260, 128261, 128257]);
        assert_eq!(c.stops, [128258, 128262]);
        assert_eq!(c.temperature, 0.4);
        assert_eq!(c.repetition_penalty, 1.0);
        let mut with_pen = pipeline();
        with_pen.parameters.insert(
            "sampling.repetition_penalty_f32".into(),
            u64::from(1.1f32.to_bits()),
        );
        assert_eq!(
            SpeechContract::from_pipeline(&with_pen)
                .unwrap()
                .repetition_penalty,
            1.1
        );
        let mut bad = pipeline();
        bad.parameters.remove("stop.1");
        assert!(SpeechContract::from_pipeline(&bad).is_err());
        bad = pipeline();
        bad.driver = "causal.v1".into();
        assert!(SpeechContract::from_pipeline(&bad).is_err());
    }

    #[test]
    fn codes_follow_frame_position() {
        let c = SpeechContract::from_pipeline(&pipeline()).unwrap();
        assert_eq!(c.code_of(0, 128266 + 5), Some(5));
        assert_eq!(c.code_of(1, 128266 + 4096 + 7), Some(7));
        assert_eq!(c.code_of(1, 128266 + 7), None);
        assert_eq!(c.code_of(6, 128266 + 6 * 4096 + 4095), Some(4095));
        assert_eq!(c.code_of(0, 128258), None);
    }

    #[test]
    fn max_tokens_matches_reference_formula() {
        let c = SpeechContract::from_pipeline(&pipeline()).unwrap();
        // scripts/tts/veena_ref.py max_new: min(int(len*1.3)*7+21, 700)
        assert_eq!(
            c.max_new_tokens("Hello, how are you doing today?"),
            40 * 7 + 21
        );
        assert_eq!(c.max_new_tokens(&"x".repeat(200)), 700);
    }

    /// Segments are ordered, non-overlapping slices of the text, at most `max` characters, with
    /// only whitespace between them.
    fn check_cover(text: &str, max: usize) -> Vec<&str> {
        let segs = segments(text, max);
        let mut at = 0;
        for s in &segs {
            assert!(!s.is_empty() && s.chars().count() <= max, "{s:?}");
            let off = s.as_ptr() as usize - text.as_ptr() as usize;
            assert!(
                off >= at && text[at..off].trim().is_empty(),
                "gap {:?}",
                &text[at..off]
            );
            at = off + s.len();
        }
        assert!(text[at..].trim().is_empty());
        segs
    }

    #[test]
    fn short_input_is_one_unchanged_segment() {
        assert_eq!(segments("  Hello there.  ", 16), ["  Hello there.  "]);
    }

    #[test]
    fn sentences_pack_up_to_the_limit() {
        let t = "One two three. Four five six! Seven eight nine? Ten eleven.";
        assert_eq!(
            check_cover(t, 30),
            [
                "One two three. Four five six!",
                "Seven eight nine? Ten eleven."
            ]
        );
        assert_eq!(
            check_cover(t, 17),
            [
                "One two three.",
                "Four five six!",
                "Seven eight nine?",
                "Ten eleven."
            ]
        );
        // No whitespace after the point: a number, not a sentence end.
        assert_eq!(
            check_cover("It costs 3.5 dollars. Then more.", 22),
            ["It costs 3.5 dollars.", "Then more."]
        );
        assert_eq!(
            check_cover("\"Stop!\" she said. Go on.", 18),
            ["\"Stop!\" she said.", "Go on."]
        );
    }

    #[test]
    fn danda_ends_a_sentence() {
        let t = "मेरा नाम वीणा है। मैं हिंदी बोलती हूँ। आप कैसे हैं?";
        let segs = check_cover(t, 20);
        assert_eq!(segs, ["मेरा नाम वीणा है।", "मैं हिंदी बोलती हूँ।", "आप कैसे हैं?"]);
    }

    #[test]
    fn long_sentences_split_at_clauses_then_words_then_chars() {
        let t = "alpha beta gamma, delta epsilon zeta, eta theta iota kappa lambda mu";
        assert_eq!(
            check_cover(t, 40),
            [
                "alpha beta gamma, delta epsilon zeta,",
                "eta theta iota kappa lambda mu"
            ]
        );
        let w = check_cover("aaaa bbbb cccc dddd eeee", 10);
        assert_eq!(w, ["aaaa bbbb", "cccc dddd", "eeee"]);
        // A sentence just over the limit splits in about equal halves, not into a lone tail word.
        let t = "In fact there is nothing he can do in these dominions as well as our nomes whose numbers are so great that it worries us to keep them all busy.";
        let h = check_cover(t, 141);
        assert_eq!(h.len(), 2);
        assert!(h.iter().all(|s| s.chars().count() >= 60), "{h:?}");
        let x = "x".repeat(25);
        let c = check_cover(&x, 10);
        assert_eq!(c.iter().map(|s| s.len()).collect::<Vec<_>>(), [10, 10, 5]);
        for max in [1, 7, 33, 151] {
            check_cover(
                &"The quick brown fox, jumps over; the lazy dog. ".repeat(40),
                max,
            );
        }
    }

    #[test]
    fn segment_chars_is_the_longest_unclipped_input() {
        let c = SpeechContract::from_pipeline(&pipeline()).unwrap();
        let n = c.segment_chars();
        assert_eq!(n, 75);
        let budget = |k: usize| (k as f32 * c.per_char_frames) as usize * c.frame_codes + 21;
        assert!(budget(n) <= c.max_new_tokens_cap && budget(n + 1) > c.max_new_tokens_cap);
    }

    const SR: u32 = 24000;

    fn tone(s: f64) -> Vec<f32> {
        (0..(s * f64::from(SR)) as usize)
            .map(|i| 0.3 * (i as f32 * 0.05).sin())
            .collect()
    }

    fn quiet(s: f64) -> Vec<f32> {
        vec![0.0; (s * f64::from(SR)) as usize]
    }

    fn secs(v: &[f32]) -> f64 {
        v.len() as f64 / f64::from(SR)
    }

    #[test]
    fn joins_keep_internal_pauses_and_bound_join_pauses() {
        let mut j = Joins::new(SR);
        let mut out = Vec::new();
        // First segment: leading silence passes untouched, an internal 2 s pause too.
        for p in [quiet(1.5), tone(1.0), quiet(2.0), tone(1.0), quiet(1.2)] {
            j.push(&p, f64::INFINITY, &mut out);
        }
        assert!(
            (secs(&out) - (1.5 + 1.0 + 2.0 + 1.0 + 0.3)).abs() < 0.02,
            "{}",
            secs(&out)
        );
        // Next segment opens with 2 s of silence: the join pause is 0.5 s in all.
        j.next_segment();
        let before = out.len();
        for p in [quiet(2.0), tone(1.0)] {
            j.push(&p, f64::INFINITY, &mut out);
        }
        assert!(
            (secs(&out[before..]) - (0.2 + 1.0)).abs() < 0.02,
            "{}",
            secs(&out[before..])
        );
        assert!(!j.runaway());
        j.push(&quiet(3.1), f64::INFINITY, &mut out);
        assert!(j.runaway() && !j.mute());
        j.next_segment();
        j.push(&quiet(2.9), f64::INFINITY, &mut out);
        assert!(!j.mute());
        j.push(&quiet(0.2), f64::INFINITY, &mut out);
        assert!(
            j.mute() && !j.speaks(&quiet(1.0)) && !j.speaks(&tone(0.1)) && j.speaks(&tone(0.3))
        );
    }

    /// A steady drone is a failed segment; speech-like modulation (4 Hz syllables) is not.
    #[test]
    fn joins_catch_a_drone_but_not_modulated_sound() {
        let j = Joins::new(SR);
        let syllables: Vec<f32> = tone(3.0)
            .iter()
            .enumerate()
            .map(|(i, x)| {
                x * (0.55 + 0.45 * (i as f32 * 4.0 * std::f32::consts::TAU / SR as f32).sin())
            })
            .collect();
        assert!(!j.failed(&syllables));
        assert!(j.failed(&tone(2.5)));
        assert!(!j.failed(&tone(1.5)), "shorter than DRONE_S");
        assert!(j.failed(&quiet(2.0)), "mute");
        let mut s = Joins::new(SR);
        s.push(&syllables, f64::INFINITY, &mut Vec::new());
        assert!(!s.drone());
        assert!(!s.segment_failed());
        s.push(&tone(2.2), f64::INFINITY, &mut Vec::new());
        assert!(s.drone() && s.segment_failed());
        s.next_segment();
        assert!(
            !s.drone() && s.segment_failed(),
            "nothing pushed yet: no speech"
        );
        s.push(&quiet(1.0), f64::INFINITY, &mut Vec::new());
        assert!(s.segment_failed());
    }

    #[test]
    fn joins_never_hold_silence_into_an_underrun() {
        let mut j = Joins::new(SR);
        let mut out = Vec::new();
        j.push(&tone(1.0), 0.0, &mut out);
        j.next_segment();
        // The client has 1.7 s buffered: at most ~0.2 s of silence may be held back.
        j.push(&quiet(2.0), 1.7, &mut out);
        assert!(secs(&out) > 1.0 + 1.75, "{}", secs(&out));
    }

    #[test]
    fn wav_header_is_44_bytes_with_sizes() {
        let h = wav_header(24000, 4800);
        assert_eq!(h.len(), 44);
        assert_eq!(&h[40..44], &4800u32.to_le_bytes());
        assert_eq!(&h[24..28], &24000u32.to_le_bytes());
    }
}
