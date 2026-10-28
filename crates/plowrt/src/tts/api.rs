//! The model-independent speech API. Every speech model declares a [`SpeechSpec`] (from its
//! packet), and every `POST /v1/audio/speech` request is checked against it the same way, in one
//! order, with one set of messages, before it reaches the model's engine. A model's specifics
//! (voices, languages, sample rate, sampling defaults, token budget, which optional controls it
//! takes) live in the spec; the engines only see a [`Checked`] request. An optional control a model
//! does not take (`top_k`, `repetition_penalty`, `cfg_weight`, `instructions`, `ref_audio`) is a
//! 400 naming the field,
//! never silently dropped.

use super::serving::SpeechRequest;
use super::stretch::{MAX_SPEED, MIN_SPEED};
use super::MAX_INPUT_CHARS;

/// Response formats every speech model serves.
pub const FORMATS: [&str; 2] = ["wav", "pcm"];

/// Rates a reply may be resampled to, the range the resampler covers.
pub const MIN_OUT_RATE: u32 = crate::asr::frontend::MIN_INPUT_RATE;
pub const MAX_OUT_RATE: u32 = crate::asr::frontend::MAX_INPUT_RATE;

#[derive(Clone, Debug, PartialEq)]
pub struct SpeechSpec {
    /// Sorted.
    pub voices: Vec<String>,
    /// Sorted ISO 639-1 codes; empty: the model has no language selection.
    pub languages: Vec<String>,
    pub default_language: Option<String>,
    pub sample_rate: u32,
    pub temperature: f32,
    pub top_p: f32,
    /// Default; `None`: the model's sampler has no repetition penalty.
    pub repetition_penalty: Option<f32>,
    /// Largest `max_tokens`: the model's audio-token budget of one segment.
    pub max_tokens: usize,
    /// Default `top_k` (0: off); `None`: the model's sampler has no top-k.
    pub top_k: Option<usize>,
    /// The largest `top_k` the sampler takes; set, top-k cannot be turned off.
    pub top_k_max: Option<usize>,
    /// Default classifier-free guidance weight, logits = cond + w (cond - uncond); `None`: the
    /// model is not guided. (A `cfg_scale` s of uncond + s (cond - uncond) is w = s - 1.)
    pub cfg_weight: Option<f32>,
    /// Takes `instructions`: a free-text description of the voice or delivery.
    pub instructions: bool,
    /// Takes `ref_audio`: zero-shot cloning of the reference's voice.
    pub voice_cloning: bool,
    /// Cloning also takes the reference's transcript `ref_text` (required); unset, it is refused.
    pub clone_transcript: bool,
}

/// The request's sampling, the spec's defaults filled in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// Per segment; `None`: the model's own budget for the segment's text.
    pub max_tokens: Option<usize>,
    /// Set when the spec has a top-k (0: off).
    pub top_k: Option<usize>,
    /// Set when the spec is guided.
    pub cfg_weight: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Checked {
    pub wav: bool,
    /// The reply's sample rate: the request's, else the model's own.
    pub sample_rate: u32,
    /// The reply's speed; 1.0 leaves the audio unstretched.
    pub speed: f32,
    pub language: Option<String>,
    pub sampling: Sampling,
    /// The request's seed, or a fresh one: an unseeded request samples anew every time.
    pub seed: u64,
}

#[derive(Debug, PartialEq)]
pub struct Invalid {
    pub message: String,
    pub param: &'static str,
}

fn invalid(message: impl Into<String>, param: &'static str) -> Invalid {
    Invalid {
        message: message.into(),
        param,
    }
}

impl SpeechSpec {
    /// Keeps only the voices `allowed` names (`--tts-voice`), so a deployment serves a subset of
    /// what the model has. A name the model lacks is warned about; no name matching leaves the
    /// list alone, which is how one flat list narrows one model of several without emptying the
    /// others. Empty `allowed` keeps the model's own list.
    pub fn narrow_voices(&mut self, allowed: &[String]) {
        let allowed: Vec<&str> = allowed
            .iter()
            .map(|v| v.trim())
            .filter(|v| !v.is_empty())
            .collect();
        if allowed.is_empty() {
            return;
        }
        let mine = |v: &str| self.voices.iter().any(|m| m == v);
        if !allowed.iter().copied().any(mine) {
            return;
        }
        for v in allowed.iter().copied().filter(|v| !mine(v)) {
            tracing::warn!(
                voice = v,
                "tts: --tts-voice names a voice this model does not have"
            );
        }
        self.voices.retain(|v| allowed.contains(&v.as_str()));
    }

    /// `req` against this model, first failing field first: format, sample rate, input, speed,
    /// language, voice, instructions, reference, sampling, cfg_weight, max_tokens.
    pub fn check(&self, req: &SpeechRequest) -> Result<Checked, Invalid> {
        let wav = match req.response_format.as_deref().unwrap_or("wav") {
            "wav" => true,
            "pcm" => false,
            f => {
                return Err(invalid(
                    format!(
                        "response_format {f:?} is not supported; use one of: {}",
                        FORMATS.join(", ")
                    ),
                    "response_format",
                ))
            }
        };
        let sample_rate = match req.sample_rate {
            None => self.sample_rate,
            Some(r) if (MIN_OUT_RATE..=MAX_OUT_RATE).contains(&r) => r,
            Some(r) => {
                return Err(invalid(
                    format!("sample_rate {r} is outside {MIN_OUT_RATE}..={MAX_OUT_RATE} Hz"),
                    "sample_rate",
                ))
            }
        };
        if req.input.trim().is_empty() {
            return Err(invalid("`input` is empty", "input"));
        }
        let n = req.input.chars().count();
        if n > MAX_INPUT_CHARS {
            return Err(invalid(
                format!("`input` is {n} characters; at most {MAX_INPUT_CHARS} are accepted"),
                "input",
            ));
        }
        let speed = match req.speed {
            None => 1.0,
            Some(s) if (MIN_SPEED..=MAX_SPEED).contains(&s) => s,
            Some(s) => {
                return Err(invalid(
                    format!("speed {s} is outside {MIN_SPEED}..={MAX_SPEED}"),
                    "speed",
                ))
            }
        };
        let language = match req.language.as_deref().map(str::to_lowercase) {
            None => self.default_language.clone(),
            Some(l) if self.languages.is_empty() => {
                return Err(invalid(
                    format!(
                        "language {l:?}: this model has no language selection; omit `language`"
                    ),
                    "language",
                ))
            }
            Some(l) if self.languages.contains(&l) => Some(l),
            Some(l) => {
                return Err(invalid(
                    format!(
                        "unsupported language {l:?}; languages: {}",
                        self.languages.join(", ")
                    ),
                    "language",
                ))
            }
        };
        if !self.voices.contains(&req.voice) {
            return Err(invalid(
                format!(
                    "unknown voice {:?}; voices: {}",
                    req.voice,
                    self.voices.join(", ")
                ),
                "voice",
            ));
        }
        if req.instructions.is_some() && !self.instructions {
            return Err(invalid(
                "this model takes no `instructions`; omit it",
                "instructions",
            ));
        }
        // Prompt text is tokenized on the request path, once per segment split.
        for (text, param) in [
            (&req.instructions, "instructions"),
            (&req.ref_text, "ref_text"),
        ] {
            let n = text.as_deref().map_or(0, |t| t.chars().count());
            if n > MAX_INPUT_CHARS {
                return Err(invalid(
                    format!("`{param}` is {n} characters; at most {MAX_INPUT_CHARS} are accepted"),
                    param,
                ));
            }
        }
        if req.ref_audio.is_some() || req.ref_text.is_some() {
            if !self.voice_cloning {
                return Err(invalid(
                    "this model has no voice cloning; omit `ref_audio` and `ref_text`",
                    "ref_audio",
                ));
            }
            if !self.clone_transcript && req.ref_text.is_some() {
                return Err(invalid(
                    "this model clones from `ref_audio` alone; omit `ref_text`",
                    "ref_text",
                ));
            }
            if req.ref_audio.is_none() || (self.clone_transcript && req.ref_text.is_none()) {
                return Err(invalid(
                    "voice cloning needs both `ref_audio` and its transcript `ref_text`",
                    if req.ref_audio.is_none() {
                        "ref_audio"
                    } else {
                        "ref_text"
                    },
                ));
            }
        }
        // The ranges chat and completions enforce: `repetition_penalty: 0` divided the logits by
        // zero and sampled from NaN probabilities.
        let fields = crate::serve::openai::SamplingFields {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            repetition_penalty: req.repetition_penalty,
            ..Default::default()
        };
        if let Err(e) = fields.validate() {
            return Err(invalid(e.message, e.field));
        }
        if req.top_k.is_some() && self.top_k.is_none() {
            return Err(invalid(
                "this model's sampler has no top-k; omit `top_k`",
                "top_k",
            ));
        }
        if let (Some(k), Some(max)) = (req.top_k, self.top_k_max) {
            if k < 1 || k as usize > max {
                return Err(invalid(
                    format!("`top_k` must be in [1, {max}] for this model; got {k}"),
                    "top_k",
                ));
            }
        }
        if req.repetition_penalty.is_some() && self.repetition_penalty.is_none() {
            return Err(invalid(
                "this model's sampler has no repetition penalty; omit `repetition_penalty`",
                "repetition_penalty",
            ));
        }
        match req.cfg_weight {
            Some(_) if self.cfg_weight.is_none() => {
                return Err(invalid(
                    "this model has no classifier-free guidance; omit `cfg_weight`",
                    "cfg_weight",
                ))
            }
            Some(w) if !(0.0..=10.0).contains(&w) => {
                return Err(invalid(
                    format!("`cfg_weight` must be in [0, 10]; got {w}"),
                    "cfg_weight",
                ))
            }
            _ => {}
        }
        match req.max_tokens {
            Some(0) => return Err(invalid("`max_tokens` must be at least 1", "max_tokens")),
            Some(m) if m > self.max_tokens => {
                return Err(invalid(
                    format!(
                        "`max_tokens` is {m}; this model generates at most {} per segment",
                        self.max_tokens
                    ),
                    "max_tokens",
                ))
            }
            _ => {}
        }
        Ok(Checked {
            wav,
            sample_rate,
            speed,
            language,
            sampling: Sampling {
                temperature: req.temperature.unwrap_or(self.temperature),
                top_p: req.top_p.unwrap_or(self.top_p),
                repetition_penalty: req
                    .repetition_penalty
                    .or(self.repetition_penalty)
                    .unwrap_or(1.0),
                max_tokens: req.max_tokens,
                // vLLM's -1 (off) is this sampler's 0.
                top_k: self
                    .top_k
                    .map(|d| req.top_k.map_or(d, |k| k.max(0) as usize)),
                cfg_weight: self.cfg_weight.map(|d| req.cfg_weight.unwrap_or(d)),
            },
            seed: req.seed.unwrap_or_else(fresh_seed),
        })
    }

    /// The `/v1/models` card's `x_plow_speech`: what a client may send this model.
    pub fn card(&self) -> serde_json::Value {
        serde_json::json!({
            "voices": self.voices,
            "languages": self.languages,
            "default_language": self.default_language,
            "sample_rate": self.sample_rate,
            "response_formats": FORMATS,
            "stream": true,
            "max_input_chars": MAX_INPUT_CHARS,
            "max_tokens": self.max_tokens,
            "instructions": self.instructions,
            "voice_cloning": self.voice_cloning,
            "clone_transcript": self.voice_cloning && self.clone_transcript,
            "defaults": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "cfg_weight": self.cfg_weight,
            },
            "limits": {
                "top_k_max": self.top_k_max,
            },
        })
    }
}

fn fresh_seed() -> u64 {
    mix(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(1, |d| d.as_nanos() as u64))
}

/// splitmix64's finalizer: nearby inputs give unrelated outputs.
pub fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(languages: &[&str]) -> SpeechSpec {
        SpeechSpec {
            voices: vec!["a".into(), "b".into()],
            languages: languages.iter().map(|s| s.to_string()).collect(),
            default_language: languages.first().map(|s| s.to_string()),
            sample_rate: 24000,
            temperature: 0.6,
            top_p: 0.8,
            repetition_penalty: Some(1.1),
            max_tokens: 1000,
            top_k: Some(0),
            top_k_max: None,
            cfg_weight: None,
            instructions: false,
            voice_cloning: false,
            clone_transcript: true,
        }
    }

    fn req(json: serde_json::Value) -> SpeechRequest {
        let mut base = serde_json::json!({"model": "m", "input": "hello", "voice": "a"});
        base.as_object_mut()
            .unwrap()
            .extend(json.as_object().unwrap().clone());
        serde_json::from_value(base).unwrap()
    }

    #[test]
    fn defaults_fill_in_and_the_request_overrides_them() {
        let c = spec(&[]).check(&req(serde_json::json!({}))).unwrap();
        assert!(c.wav);
        assert_eq!(c.language, None);
        assert_eq!(
            c.sampling,
            Sampling {
                temperature: 0.6,
                top_p: 0.8,
                repetition_penalty: 1.1,
                max_tokens: None,
                top_k: Some(0),
                cfg_weight: None,
            }
        );
        let c = spec(&[])
            .check(&req(serde_json::json!({
                "temperature": 0.0, "top_p": 0.5, "repetition_penalty": 1.3, "max_tokens": 10,
                "seed": 7, "response_format": "pcm"
            })))
            .unwrap();
        assert!(!c.wav);
        assert_eq!(c.seed, 7);
        assert_eq!(c.sampling.temperature, 0.0);
        assert_eq!(c.sampling.max_tokens, Some(10));
    }

    #[test]
    fn unseeded_requests_draw_fresh_seeds() {
        let s = spec(&[]);
        let r = req(serde_json::json!({}));
        assert_ne!(s.check(&r).unwrap().seed, s.check(&r).unwrap().seed);
    }

    #[test]
    fn languages_resolve_the_same_way_on_every_model() {
        let multi = spec(&["en", "fr"]);
        assert_eq!(
            multi.check(&req(serde_json::json!({}))).unwrap().language,
            Some("en".into())
        );
        assert_eq!(
            multi
                .check(&req(serde_json::json!({"language": "FR"})))
                .unwrap()
                .language,
            Some("fr".into())
        );
        let e = multi
            .check(&req(serde_json::json!({"language": "de"})))
            .unwrap_err();
        assert_eq!(e.param, "language");
        assert!(e.message.contains("en, fr"), "{}", e.message);
        let e = spec(&[])
            .check(&req(serde_json::json!({"language": "en"})))
            .unwrap_err();
        assert_eq!(e.param, "language");
    }

    /// One order on every model: the first failing field is the one reported.
    #[test]
    fn the_first_failing_field_is_reported() {
        let s = spec(&[]);
        let all_bad = serde_json::json!({
            "response_format": "mp3", "sample_rate": 100, "input": " ", "speed": 3.0,
            "language": "en", "voice": "z", "instructions": "calm", "ref_text": "hi",
            "temperature": 9.0, "cfg_weight": 0.5, "max_tokens": 0
        });
        let order = [
            "response_format",
            "sample_rate",
            "input",
            "speed",
            "language",
            "voice",
            "instructions",
            "ref_audio",
            "temperature",
            "cfg_weight",
            "max_tokens",
        ];
        let mut r = all_bad;
        for (i, field) in order.iter().enumerate() {
            let e = s.check(&req(r.clone())).unwrap_err();
            assert_eq!(e.param, *field);
            let fix = match i {
                0 => ("response_format", serde_json::json!("wav")),
                1 => ("sample_rate", serde_json::Value::Null),
                2 => ("input", serde_json::json!("hello")),
                3 => ("speed", serde_json::json!(1.0)),
                4 => ("language", serde_json::Value::Null),
                5 => ("voice", serde_json::json!("b")),
                6 => ("instructions", serde_json::Value::Null),
                7 => ("ref_text", serde_json::Value::Null),
                8 => ("temperature", serde_json::json!(0.5)),
                9 => ("cfg_weight", serde_json::Value::Null),
                _ => ("max_tokens", serde_json::json!(5)),
            };
            r[fix.0] = fix.1;
        }
        assert!(s.check(&req(r)).is_ok());
    }

    /// The optional controls are taken only where the spec declares them.
    #[test]
    fn optional_controls_follow_the_spec() {
        let mut guided = spec(&[]);
        guided.top_k = None;
        guided.cfg_weight = Some(0.5);
        let c = guided
            .check(&req(serde_json::json!({"cfg_weight": 0.8})))
            .unwrap();
        assert_eq!((c.sampling.cfg_weight, c.sampling.top_k), (Some(0.8), None));
        assert_eq!(
            guided
                .check(&req(serde_json::json!({"top_k": 30})))
                .unwrap_err()
                .param,
            "top_k"
        );
        let plain = spec(&[]);
        let c = plain.check(&req(serde_json::json!({"top_k": -1}))).unwrap();
        assert_eq!(c.sampling.top_k, Some(0));
        // A depth-stage sampler: top-k always on and bounded, no repetition penalty.
        let mut depth = spec(&[]);
        depth.top_k = Some(30);
        depth.top_k_max = Some(64);
        depth.repetition_penalty = None;
        for body in [
            serde_json::json!({"top_k": 0}),
            serde_json::json!({"top_k": -1}),
            serde_json::json!({"top_k": 65}),
        ] {
            assert_eq!(depth.check(&req(body)).unwrap_err().param, "top_k");
        }
        let c = depth.check(&req(serde_json::json!({"top_k": 64}))).unwrap();
        assert_eq!(
            (c.sampling.top_k, c.sampling.repetition_penalty),
            (Some(64), 1.0)
        );
        assert_eq!(
            depth
                .check(&req(serde_json::json!({"repetition_penalty": 1.1})))
                .unwrap_err()
                .param,
            "repetition_penalty"
        );
        let mut cloning = spec(&[]);
        cloning.voice_cloning = true;
        assert_eq!(
            cloning
                .check(&req(serde_json::json!({"ref_audio": "UklGRg=="})))
                .unwrap_err()
                .param,
            "ref_text"
        );
        assert!(cloning
            .check(&req(
                serde_json::json!({"ref_audio": "UklGRg==", "ref_text": "hi"})
            ))
            .is_ok());
        cloning.clone_transcript = false;
        assert!(cloning
            .check(&req(serde_json::json!({"ref_audio": "UklGRg=="})))
            .is_ok());
        assert_eq!(
            cloning
                .check(&req(
                    serde_json::json!({"ref_audio": "UklGRg==", "ref_text": "hi"})
                ))
                .unwrap_err()
                .param,
            "ref_text"
        );
    }

    #[test]
    fn input_and_max_tokens_limits() {
        let mut s = spec(&[]);
        s.instructions = true;
        s.voice_cloning = true;
        let long = "a".repeat(MAX_INPUT_CHARS + 1);
        for param in ["instructions", "ref_text"] {
            let mut body = serde_json::json!({"ref_audio": "UklGRg==", "ref_text": "hi"});
            body[param] = long.clone().into();
            assert_eq!(s.check(&req(body)).unwrap_err().param, param);
        }
        assert_eq!(
            s.check(&req(serde_json::json!({"input": long})))
                .unwrap_err()
                .param,
            "input"
        );
        assert_eq!(
            s.check(&req(serde_json::json!({"max_tokens": 1001})))
                .unwrap_err()
                .param,
            "max_tokens"
        );
        assert!(s
            .check(&req(serde_json::json!({"max_tokens": 1000})))
            .is_ok());
    }

    #[test]
    fn a_reply_takes_the_requested_sample_rate() {
        let s = spec(&[]);
        assert_eq!(
            s.check(&req(serde_json::json!({}))).unwrap().sample_rate,
            24000
        );
        assert_eq!(
            s.check(&req(serde_json::json!({"sample_rate": 8000})))
                .unwrap()
                .sample_rate,
            8000
        );
        for bad in [MIN_OUT_RATE - 1, MAX_OUT_RATE + 1] {
            assert_eq!(
                s.check(&req(serde_json::json!({"sample_rate": bad})))
                    .unwrap_err()
                    .param,
                "sample_rate"
            );
        }
    }

    #[test]
    fn a_reply_takes_the_requested_speed() {
        let s = spec(&[]);
        assert_eq!(s.check(&req(serde_json::json!({}))).unwrap().speed, 1.0);
        for ok in [MIN_SPEED, 0.9, MAX_SPEED] {
            assert_eq!(
                s.check(&req(serde_json::json!({"speed": ok})))
                    .unwrap()
                    .speed,
                ok
            );
        }
        for bad in [MIN_SPEED - 0.01, MAX_SPEED + 0.01] {
            assert_eq!(
                s.check(&req(serde_json::json!({"speed": bad})))
                    .unwrap_err()
                    .param,
                "speed"
            );
        }
    }

    #[test]
    fn a_deployment_narrows_the_voices_a_model_serves() {
        let mut s = spec(&[]);
        s.narrow_voices(&[]);
        assert_eq!(s.voices, ["a", "b"]);

        let mut s = spec(&[]);
        s.narrow_voices(&["b".into(), "  ".into()]);
        assert_eq!(s.voices, ["b"]);
        assert_eq!(
            s.check(&req(serde_json::json!({"voice": "a"})))
                .unwrap_err()
                .param,
            "voice"
        );

        let mut s = spec(&[]);
        s.narrow_voices(&["a".into(), "absent".into()]);
        assert_eq!(s.voices, ["a"]);

        let mut s = spec(&[]);
        s.narrow_voices(&["another-model".into()]);
        assert_eq!(s.voices, ["a", "b"]);
    }
}
