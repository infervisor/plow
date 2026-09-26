//! Text-to-speech packet pipelines.
//!
//! The speech counterpart of [`crate::asr`]: the compiled asset declares the pipeline
//! (`packet_pipeline.json`, driver `tts.codec_lm.v1`, emitted by `plowc --tts-profile`), and the
//! runtime binds that declaration instead of knowing any checkpoint. A codec-token LM runs on the
//! served engine's continuous-batching mux like any causal model; the codes it emits are decoded
//! by the codec stage the contract names ([`codec`]). See docs/arch/24-tts-pipelines.md.

#[cfg(feature = "cuda")]
pub mod codec;
#[cfg(feature = "cuda")]
pub mod serving;
#[cfg(feature = "cuda")]
pub mod t3;

use std::path::Path;

use plow_asset::packet_pipeline::PacketPipeline;

use crate::{Result, RuntimeError};

pub const DRIVER: &str = "tts.codec_lm.v1";
const PROMPT_SPEAKER_TAG: u64 = 1;
const CODEC_SNAC24K_FRAME7: u64 = 1;

/// The numeric contract of one `tts.codec_lm.v1` pipeline, read from packet metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeechContract {
    pub pipeline: String,
    pub prompt_format: u64,
    pub prefix: Vec<u32>,
    pub suffix: Vec<u32>,
    pub stops: Vec<u32>,
    pub codec_kind: u64,
    pub sample_rate: u32,
    pub frame_codes: usize,
    pub codebook: u32,
    pub frame_samples: usize,
    pub audio_token_base: u32,
    pub per_char_frames: f32,
    pub max_new_tokens_cap: usize,
    pub temperature: f32,
    pub top_p: f32,
}

impl SpeechContract {
    pub fn from_pipeline(p: &PacketPipeline) -> Result<Self> {
        if p.driver != DRIVER {
            return Err(RuntimeError::Rejected(format!("pipeline {} is {}, not {DRIVER}", p.name, p.driver)));
        }
        let get = |k: &str| {
            p.parameters
                .get(k)
                .copied()
                .ok_or_else(|| RuntimeError::Rejected(format!("speech pipeline lacks parameter {k}")))
        };
        let list = |k: &str| -> Result<Vec<u32>> {
            (0..get(&format!("{k}.count"))?).map(|i| get(&format!("{k}.{i}")).map(|v| v as u32)).collect()
        };
        let f32p = |k: &str| get(k).map(|v| f32::from_bits(v as u32));
        let c = SpeechContract {
            pipeline: p.name.clone(),
            prompt_format: get("prompt.format")?,
            prefix: list("prompt.prefix")?,
            suffix: list("prompt.suffix")?,
            stops: list("stop")?,
            codec_kind: get("codec.kind")?,
            sample_rate: get("audio.sample_rate")? as u32,
            frame_codes: get("codec.frame_codes")? as usize,
            codebook: get("codec.codebook")? as u32,
            frame_samples: get("codec.frame_samples")? as usize,
            audio_token_base: get("audio.token_base")? as u32,
            per_char_frames: f32p("tokens.per_char_frames_f32")?,
            max_new_tokens_cap: get("tokens.max_new_cap")? as usize,
            temperature: f32p("sampling.temperature_f32")?,
            top_p: f32p("sampling.top_p_f32")?,
        };
        if c.prompt_format != PROMPT_SPEAKER_TAG {
            return Err(RuntimeError::Rejected(format!("unsupported prompt.format {}", c.prompt_format)));
        }
        if c.codec_kind != CODEC_SNAC24K_FRAME7 || c.frame_codes != 7 || c.frame_samples != 2048 || c.codebook != 4096 {
            return Err(RuntimeError::Rejected("unsupported codec contract (want SNAC-24k, 7 codes/frame)".into()));
        }
        Ok(c)
    }

    /// The single speech pipeline of `<assets>/model.pkt`, or `None` for a non-speech packet.
    pub fn load(assets: &Path) -> Result<Option<Self>> {
        let asset = crate::exec::packet_runtime::PacketAsset::load(&assets.join("model.pkt"))?;
        let mut found = asset.pipelines().iter().filter(|p| p.driver == DRIVER);
        match (found.next(), found.next()) {
            (None, _) => Ok(None),
            (Some(p), None) => Self::from_pipeline(p).map(Some),
            (Some(_), Some(_)) => Err(RuntimeError::Rejected("ambiguous: several speech pipelines".into())),
        }
    }

    /// Prompt text for `prompt.format` 1; the tokenizer encodes it between prefix and suffix.
    pub fn prompt_text(&self, voice: &str, input: &str) -> String {
        format!("<spk_{voice}> {input}")
    }

    /// The speaker tag must be one vocabulary token, which is how a voice is known to exist.
    pub fn voice_token(&self, voice: &str) -> String {
        format!("<spk_{voice}>")
    }

    pub fn max_new_tokens(&self, input: &str) -> usize {
        let frames = (input.chars().count() as f32 * self.per_char_frames) as usize;
        (frames * self.frame_codes + 21).min(self.max_new_tokens_cap)
    }

    /// Codebook id of the `n`-th kept audio token, or `None` when `tok` is not the code its frame
    /// position expects (dropped, as the reference decoder drops out-of-range ids).
    pub fn code_of(&self, n: usize, tok: u32) -> Option<i32> {
        let lo = self.audio_token_base + (n % self.frame_codes) as u32 * self.codebook;
        (lo..lo + self.codebook).contains(&tok).then(|| (tok - lo) as i32)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn pipeline() -> PacketPipeline {
        let mut p: BTreeMap<String, u64> = BTreeMap::new();
        for (k, v) in [
            ("prompt.format", 1),
            ("prompt.prefix.count", 1),
            ("prompt.prefix.0", 128259),
            ("prompt.suffix.count", 3),
            ("prompt.suffix.0", 128260),
            ("prompt.suffix.1", 128261),
            ("prompt.suffix.2", 128257),
            ("stop.count", 2),
            ("stop.0", 128258),
            ("stop.1", 128262),
            ("codec.kind", 1),
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
        assert_eq!(c.max_new_tokens("Hello, how are you doing today?"), 40 * 7 + 21);
        assert_eq!(c.max_new_tokens(&"x".repeat(200)), 700);
    }

    #[test]
    fn wav_header_is_44_bytes_with_sizes() {
        let h = wav_header(24000, 4800);
        assert_eq!(h.len(), 44);
        assert_eq!(&h[40..44], &4800u32.to_le_bytes());
        assert_eq!(&h[24..28], &24000u32.to_le_bytes());
    }
}
