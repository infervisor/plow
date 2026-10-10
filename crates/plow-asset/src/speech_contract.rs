//! The ASR/VAD packet contract (`docs/runtime/asr-packet-contract.md`): the pipeline parameters
//! and strings a speech packet declares so that plowrt runs it without naming a model. Emitters
//! write these; plowrt reads them and refuses a packet whose contract it does not implement.

use std::collections::BTreeMap;

/// Pipeline parameter carrying the contract version of a speech pipeline.
pub const CONTRACT: &str = "contract";

/// Generic frame classifier (VAD): per-stream arena, `step.<bank>` state-bank programs rotated
/// each frame, a `context + frame` sample window in, one probability out.
pub const VAD_DRIVER: &str = "vad.frame.v1";
pub const VAD_CONTRACT: u64 = 1;
/// The driver of contract-0 VAD packets, refused (re-emit with `scripts/asr/silero_vad_build.sh`).
pub const VAD_DRIVER_V0: &str = "vad.silero.v1";

/// Where a VAD packet's programs run (`executor`): plowrt's host op interpreter or a device.
pub const EXECUTOR_HOST: u64 = 1;
pub const EXECUTOR_DEVICE: u64 = 2;

/// Contract of the ASR pipelines (`rnnt.greedy.v1`, the audio-LM `causal.v1` decoder and its
/// `audio.encode` encoder). Version 1 adds the packet-held output vocabulary, detokenizer and
/// language declaration to `rnnt.greedy.v1`; packets without the `contract` parameter are 0.
pub const ASR_CONTRACT: u64 = 1;

/// Metadata section of an `rnnt.greedy.v1` packet holding its output vocabulary.
pub const VOCABULARY_SECTION: &str = "asr_vocabulary.json";

/// `asr_vocabulary.json`: token id -> piece.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vocabulary {
    pub pieces: Vec<String>,
}

/// `output.detokenizer` of a token-emitting decoder: pieces joined, `output.word_boundary` turned
/// into a space, no space before an `output.no_space_before` piece, and pieces wrapped in `<...>`
/// dropped when `output.skip_bracketed` is 1.
pub const DETOKENIZER_SENTENCEPIECE: &str = "sentencepiece";

/// How a token-emitting ASR decoder's ids become text, and the language it transcribes: the
/// `output.*` / `language*` strings of its pipeline plus the [`VOCABULARY_SECTION`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenOutput {
    pub pieces: Vec<String>,
    pub word_boundary: String,
    pub no_space_before: Vec<String>,
    pub skip_bracketed: bool,
    /// The one language a prompted packet transcribes; `None`: the model detects it.
    pub language: Option<String>,
    /// `(requested, language)`: request spellings accepted for `language`.
    pub language_aliases: Vec<(String, String)>,
}

impl TokenOutput {
    /// SentencePiece output: `▁` marks a word start; no space before sentence punctuation.
    pub fn sentencepiece(pieces: Vec<String>, no_space_before: &[&str]) -> Self {
        Self {
            pieces,
            word_boundary: "\u{2581}".into(),
            no_space_before: no_space_before.iter().map(|s| (*s).to_owned()).collect(),
            skip_bracketed: true,
            language: None,
            language_aliases: Vec::new(),
        }
    }

    /// The contract parameter and strings, into a pipeline.
    pub fn apply(&self, parameters: &mut BTreeMap<String, u64>, strings: &mut BTreeMap<String, String>) {
        parameters.insert(CONTRACT.into(), ASR_CONTRACT);
        parameters.insert("output.skip_bracketed".into(), u64::from(self.skip_bracketed));
        strings.insert("output.detokenizer".into(), DETOKENIZER_SENTENCEPIECE.into());
        strings.insert("output.word_boundary".into(), self.word_boundary.clone());
        strings.insert("output.no_space_before".into(), self.no_space_before.join("\n"));
        if let Some(language) = &self.language {
            strings.insert("language".into(), language.clone());
            let aliases: Vec<String> = self.language_aliases.iter().map(|(a, l)| format!("{a}={l}")).collect();
            strings.insert("language.aliases".into(), aliases.join("\n"));
        }
    }

    pub fn vocabulary_section(&self) -> Vec<u8> {
        serde_json::to_vec(&Vocabulary { pieces: self.pieces.clone() }).expect("strings serialize")
    }

    /// Read back from a pipeline's strings and its vocabulary section.
    pub fn from_pipeline(
        parameter: impl Fn(&str) -> Option<u64>,
        string: impl Fn(&str) -> Option<String>,
        vocabulary: &[u8],
    ) -> Result<Self, String> {
        let text = |name: &str| string(name).ok_or_else(|| format!("ASR output string {name} is missing"));
        if text("output.detokenizer")? != DETOKENIZER_SENTENCEPIECE {
            return Err("unsupported ASR output detokenizer".into());
        }
        let lines = |value: String| value.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect::<Vec<_>>();
        let vocabulary: Vocabulary =
            serde_json::from_slice(vocabulary).map_err(|e| format!("{VOCABULARY_SECTION}: {e}"))?;
        let language = string("language");
        let language_aliases = match &language {
            None => Vec::new(),
            Some(_) => lines(text("language.aliases")?)
                .into_iter()
                .map(|l| l.split_once('=').map(|(a, b)| (a.to_owned(), b.to_owned())).ok_or("malformed language alias"))
                .collect::<Result<_, _>>()?,
        };
        Ok(Self {
            pieces: vocabulary.pieces,
            word_boundary: text("output.word_boundary")?,
            no_space_before: lines(text("output.no_space_before")?),
            skip_bracketed: parameter("output.skip_bracketed").ok_or("output.skip_bracketed is missing")? == 1,
            language,
            language_aliases,
        })
    }
}

/// Host policy of an audio-LM decoder (`causal.v1` with an audio overlay), contract 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioLmPolicy {
    /// Low-level noise appended to a stream's final piece so the model hears it end.
    pub final_padding_samples: u32,
    pub final_padding_amplitude: f32,
    /// Context positions kept free for the transcript: `per_row * audio rows + extra`.
    pub output_reserve_per_row: u32,
    pub output_reserve_extra: u32,
}

impl AudioLmPolicy {
    pub fn to_parameters(&self, out: &mut BTreeMap<String, u64>) {
        out.insert(CONTRACT.into(), ASR_CONTRACT);
        out.insert("finalize.padding_samples".into(), u64::from(self.final_padding_samples));
        out.insert("finalize.padding_amplitude_f32".into(), f32_param(self.final_padding_amplitude));
        out.insert("output.reserve_per_row".into(), u64::from(self.output_reserve_per_row));
        out.insert("output.reserve_extra".into(), u64::from(self.output_reserve_extra));
    }

    pub fn from_parameters(parameter: impl Fn(&str) -> Option<u64>) -> Result<Self, String> {
        let get = |name: &str| parameter(name).ok_or_else(|| format!("audio LM parameter {name} is missing"));
        let u32_of = |name: &str| get(name).and_then(|v| u32::try_from(v).map_err(|_| format!("audio LM parameter {name} overflows")));
        Ok(Self {
            final_padding_samples: u32_of("finalize.padding_samples")?,
            final_padding_amplitude: param_f32(get("finalize.padding_amplitude_f32")?)
                .ok_or("finalize.padding_amplitude_f32 is not a finite f32")?,
            output_reserve_per_row: u32_of("output.reserve_per_row")?,
            output_reserve_extra: u32_of("output.reserve_extra")?,
        })
    }
}

/// The contract version a speech pipeline declares (0 when it predates the parameter), refused
/// when newer than `implemented`.
pub fn check_contract(parameter: Option<u64>, implemented: u64, what: &str) -> Result<u64, String> {
    let version = parameter.unwrap_or(0);
    if version > implemented {
        return Err(format!("{what} packet contract {version}; this plowrt implements {implemented}"));
    }
    if version < implemented {
        return Err(format!(
            "{what} packet contract {version} predates {implemented}: re-emit it (bundles with compiler check \
             receipts), or upgrade its metadata with asr_packet_upgrade (programs and weights are unchanged)"
        ));
    }
    Ok(version)
}

/// f32 parameters are stored as their bit pattern.
pub fn f32_param(value: f32) -> u64 {
    u64::from(value.to_bits())
}

pub fn param_f32(value: u64) -> Option<f32> {
    u32::try_from(value).ok().map(f32::from_bits).filter(|v| v.is_finite())
}

/// Speech-segment policy of a VAD packet (`policy.*`): the model's recommended defaults for
/// `/v1/audio/vad` and the turn detector. Requests may override them within [`VadBounds`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VadPolicy {
    /// Speech starts at or above this probability.
    pub threshold: f32,
    /// Speech ends below `max(threshold - release_offset, release_floor)` (hysteresis).
    pub release_offset: f32,
    pub release_floor: f32,
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub speech_pad_ms: u32,
    /// 0 = unbounded.
    pub max_speech_ms: u32,
    /// Silences longer than this are cut candidates when a segment reaches `max_speech_ms`.
    pub min_silence_at_max_speech_ms: u32,
}

/// No-speech upload gate (`gate.*`): segments as [`VadPolicy`] with these durations; an upload
/// with less than `min_total_speech_ms` of speech gets an empty transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VadGate {
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub speech_pad_ms: u32,
    pub min_total_speech_ms: u32,
}

/// Upper bounds a request override may reach (`bounds.*`); thresholds are always within 0..=1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VadBounds {
    pub max_duration_ms: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VadContract {
    pub policy: VadPolicy,
    pub gate: VadGate,
    pub bounds: VadBounds,
}

impl VadContract {
    pub fn to_parameters(&self, out: &mut BTreeMap<String, u64>) {
        let p = &self.policy;
        for (name, value) in [
            ("policy.threshold_f32", f32_param(p.threshold)),
            ("policy.release_offset_f32", f32_param(p.release_offset)),
            ("policy.release_floor_f32", f32_param(p.release_floor)),
            ("policy.min_speech_ms", u64::from(p.min_speech_ms)),
            ("policy.min_silence_ms", u64::from(p.min_silence_ms)),
            ("policy.speech_pad_ms", u64::from(p.speech_pad_ms)),
            ("policy.max_speech_ms", u64::from(p.max_speech_ms)),
            ("policy.min_silence_at_max_speech_ms", u64::from(p.min_silence_at_max_speech_ms)),
            ("gate.min_speech_ms", u64::from(self.gate.min_speech_ms)),
            ("gate.min_silence_ms", u64::from(self.gate.min_silence_ms)),
            ("gate.speech_pad_ms", u64::from(self.gate.speech_pad_ms)),
            ("gate.min_total_speech_ms", u64::from(self.gate.min_total_speech_ms)),
            ("bounds.max_duration_ms", u64::from(self.bounds.max_duration_ms)),
        ] {
            out.insert(name.into(), value);
        }
    }

    pub fn from_parameters(parameter: impl Fn(&str) -> Option<u64>) -> Result<Self, String> {
        let get = |name: &str| parameter(name).ok_or_else(|| format!("VAD parameter {name} is missing"));
        let ms = |name: &str| get(name).and_then(|v| u32::try_from(v).map_err(|_| format!("VAD parameter {name} overflows")));
        let unit = |name: &str| {
            get(name)
                .and_then(|v| param_f32(v).ok_or_else(|| format!("VAD parameter {name} is not a finite f32")))
                .and_then(|v| (0.0..=1.0).contains(&v).then_some(v).ok_or_else(|| format!("VAD parameter {name} is outside 0..=1")))
        };
        let contract = Self {
            policy: VadPolicy {
                threshold: unit("policy.threshold_f32")?,
                release_offset: unit("policy.release_offset_f32")?,
                release_floor: unit("policy.release_floor_f32")?,
                min_speech_ms: ms("policy.min_speech_ms")?,
                min_silence_ms: ms("policy.min_silence_ms")?,
                speech_pad_ms: ms("policy.speech_pad_ms")?,
                max_speech_ms: ms("policy.max_speech_ms")?,
                min_silence_at_max_speech_ms: ms("policy.min_silence_at_max_speech_ms")?,
            },
            gate: VadGate {
                min_speech_ms: ms("gate.min_speech_ms")?,
                min_silence_ms: ms("gate.min_silence_ms")?,
                speech_pad_ms: ms("gate.speech_pad_ms")?,
                min_total_speech_ms: ms("gate.min_total_speech_ms")?,
            },
            bounds: VadBounds { max_duration_ms: ms("bounds.max_duration_ms")? },
        };
        let p = &contract.policy;
        let max = contract.bounds.max_duration_ms;
        if [p.min_speech_ms, p.min_silence_ms, p.speech_pad_ms, p.max_speech_ms, p.min_silence_at_max_speech_ms]
            .iter()
            .any(|&v| v > max)
        {
            return Err("VAD policy default exceeds its declared bound".into());
        }
        Ok(contract)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vad_contract_round_trips_and_checks_bounds() {
        let contract = VadContract {
            policy: VadPolicy {
                threshold: 0.5,
                release_offset: 0.15,
                release_floor: 0.01,
                min_speech_ms: 250,
                min_silence_ms: 100,
                speech_pad_ms: 30,
                max_speech_ms: 0,
                min_silence_at_max_speech_ms: 98,
            },
            gate: VadGate { min_speech_ms: 150, min_silence_ms: 150, speech_pad_ms: 0, min_total_speech_ms: 250 },
            bounds: VadBounds { max_duration_ms: 600_000 },
        };
        let mut params = BTreeMap::new();
        contract.to_parameters(&mut params);
        assert_eq!(VadContract::from_parameters(|n| params.get(n).copied()).unwrap(), contract);
        params.insert("bounds.max_duration_ms".into(), 10);
        assert!(VadContract::from_parameters(|n| params.get(n).copied()).is_err());
        params.remove("policy.threshold_f32");
        assert!(VadContract::from_parameters(|n| params.get(n).copied()).is_err());
    }

    #[test]
    fn contracts_fail_closed_on_other_versions() {
        assert_eq!(check_contract(Some(ASR_CONTRACT), ASR_CONTRACT, "x"), Ok(ASR_CONTRACT));
        assert!(check_contract(None, ASR_CONTRACT, "x").unwrap_err().contains("asr_packet_upgrade"));
        assert!(check_contract(Some(ASR_CONTRACT + 1), ASR_CONTRACT, "x").is_err());
    }

    #[test]
    fn token_output_round_trips() {
        let mut output = TokenOutput::sentencepiece(vec!["\u{2581}a".into(), "<b>".into()], &[".", "?"]);
        output.language = Some("en-US".into());
        output.language_aliases = vec![("en".into(), "en-US".into())];
        let (mut params, mut strings) = (BTreeMap::new(), BTreeMap::new());
        output.apply(&mut params, &mut strings);
        assert_eq!(params[CONTRACT], ASR_CONTRACT);
        let back = TokenOutput::from_pipeline(|n| params.get(n).copied(), |n| strings.get(n).cloned(), &output.vocabulary_section());
        assert_eq!(back.unwrap(), output);
        strings.insert("output.detokenizer".into(), "bpe".into());
        assert!(TokenOutput::from_pipeline(|n| params.get(n).copied(), |n| strings.get(n).cloned(), &output.vocabulary_section()).is_err());
    }
}
