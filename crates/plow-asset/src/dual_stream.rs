//! Dual-stream speech prompts (`prompt.dual_stream` string of a `tts.codec_lm.v1` pipeline): every
//! LM row is a text-side token plus a frame of codec-side codes (`EmbedFrameBf16` over the whole
//! vocabulary). The prompt pairs control codes with text pads, and each decode step's row is the
//! previous frame's codes plus the next text token of the input (then `text_pad`).

use std::collections::BTreeMap;

/// Pipeline string holding the [`DualStream`] JSON.
pub const KEY: &str = "prompt.dual_stream";

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DualStream {
    /// Text of the input's rows, `{input}` = the request's text.
    pub text_template: String,
    /// Text rows ahead of everything else, `{instructions}` = the request's instructions.
    pub instructions_template: Option<String>,
    /// Template tokens before the codec head (plain text rows).
    pub lead: usize,
    /// Template tokens after the input, never spoken.
    pub drop: usize,
    pub text_pad: u32,
    pub text_bos: u32,
    pub text_eos: u32,
    /// Code of a frame position without a codec-side row (a zero table row).
    pub none_code: u32,
    /// Codes of a frame, `none_code` past the first.
    pub frame_codes: usize,
    /// Codec head before the speaker with a language; the language id goes at `language_at`.
    pub codec_language: Vec<u32>,
    pub language_at: usize,
    /// Codec head without a language (`auto`).
    pub codec_auto: Vec<u32>,
    pub codec_pad: u32,
    pub codec_bos: u32,
    /// Lower-case language name -> codec id; `auto` (not listed) takes `codec_auto`.
    pub languages: BTreeMap<String, u32>,
    /// Lower-case voice name -> its speaker code and dialect language.
    pub voices: BTreeMap<String, Voice>,
    /// Requested languages a dialect voice replaces with its dialect.
    pub dialect_languages: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Voice {
    pub code: u32,
    pub dialect: Option<String>,
}

/// One request's rows: `ids[r]` with codes `codes[r * frame_codes..]`, then the text fed at each
/// decode step (`feed[s]` at step `s`, `tail` after).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub ids: Vec<u32>,
    pub codes: Vec<u32>,
    pub feed: Vec<u32>,
    pub tail: u32,
}

pub struct Reference<'a> {
    /// Reference transcript tokenized with `text_template`, including its lead/drop wrappers.
    pub text: &'a [u32],
    /// Complete row-major codec frames, without control or padding codes.
    pub codes: &'a [u32],
    pub codebook_size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConditionedLayout {
    pub layout: Layout,
    /// An overlay must replace this entire row with `bf16(text_pad + speaker_embedding)`.
    pub speaker_row: usize,
}

impl DualStream {
    pub fn parse(json: &str) -> Result<Self, String> {
        let d: Self = serde_json::from_str(json).map_err(|e| format!("{KEY}: {e}"))?;
        if d.frame_codes == 0
            || d.language_at > d.codec_language.len()
            || !d.text_template.contains("{input}")
        {
            return Err(format!("{KEY}: inconsistent layout"));
        }
        Ok(d)
    }

    /// The codec id of `language` (`None` = `auto`) as `voice` speaks it.
    fn language(&self, language: &str, voice: Option<&Voice>) -> Result<Option<u32>, String> {
        let language = language.to_lowercase();
        if let Some(d) = voice
            .and_then(|v| v.dialect.as_ref())
            .filter(|_| self.dialect_languages.contains(&language))
        {
            return self
                .languages
                .get(d)
                .map(|&c| Some(c))
                .ok_or_else(|| format!("unknown dialect {d:?}"));
        }
        if language == "auto" {
            return Ok(None);
        }
        self.languages
            .get(&language)
            .map(|&c| Some(c))
            .ok_or_else(|| format!("unknown language {language:?}"))
    }

    pub fn voice(&self, name: &str) -> Option<&Voice> {
        self.voices.get(&name.to_lowercase())
    }

    /// The rows of `text` (the tokenized `text_template`) after `instructions` (the tokenized
    /// `instructions_template`). `streaming`: the text rides the decode steps one token per
    /// frame; otherwise it is all in the prompt.
    pub fn layout(
        &self,
        text: &[u32],
        instructions: &[u32],
        voice: &str,
        language: &str,
        streaming: bool,
    ) -> Result<Layout, String> {
        let v = self
            .voice(voice)
            .ok_or_else(|| format!("unknown voice {voice:?}"))?;
        let body = self.body(text)?;
        let (mut out, _) = self.start(text, instructions, language, Some(v))?;
        self.append_text(&mut out, body, streaming);
        Ok(out)
    }

    pub fn layout_conditioned(
        &self,
        text: &[u32],
        instructions: &[u32],
        language: &str,
        streaming: bool,
        reference: Option<Reference<'_>>,
    ) -> Result<ConditionedLayout, String> {
        if self.frame_codes == 0 {
            return Err("zero frame width".into());
        }
        let body = self.body(text)?;
        let (mut out, speaker_row) = self.start(text, instructions, language, None)?;
        let Some(reference) = reference else {
            self.append_text(&mut out, body, streaming);
            return Ok(ConditionedLayout {
                layout: out,
                speaker_row,
            });
        };
        let ref_body = self.body(reference.text)?;
        if reference.codebook_size == 0
            || reference.codebook_size > self.none_code
            || reference.codes.is_empty()
            || reference.codes.len() % self.frame_codes != 0
            || reference
                .codes
                .iter()
                .any(|&c| c >= reference.codebook_size)
        {
            return Err("invalid reference codec frames".into());
        }
        let words: Vec<u32> = ref_body
            .iter()
            .chain(body)
            .chain([&self.text_eos])
            .copied()
            .collect();
        if !streaming {
            for &t in &words {
                self.row(&mut out, t, &[self.codec_pad]);
            }
        }
        self.row(
            &mut out,
            if streaming { words[0] } else { self.text_pad },
            &[self.codec_bos],
        );
        let frames = reference.codes.len() / self.frame_codes;
        for (i, codes) in reference.codes.chunks_exact(self.frame_codes).enumerate() {
            let id = if streaming {
                words.get(i + 1).copied().unwrap_or(self.text_pad)
            } else {
                self.text_pad
            };
            self.row(&mut out, id, codes);
        }
        if streaming && words.len() > frames + 1 {
            out.feed.extend_from_slice(&words[frames + 1..]);
        }
        Ok(ConditionedLayout {
            layout: out,
            speaker_row,
        })
    }

    fn body<'a>(&self, text: &'a [u32]) -> Result<&'a [u32], String> {
        if self.lead >= text.len() || self.drop >= text.len() - self.lead {
            return Err("empty input".into());
        }
        Ok(&text[self.lead..text.len() - self.drop])
    }

    fn row(&self, out: &mut Layout, id: u32, codes: &[u32]) {
        out.ids.push(id);
        out.codes.extend_from_slice(codes);
        out.codes.extend(std::iter::repeat_n(
            self.none_code,
            self.frame_codes - codes.len(),
        ));
    }

    fn append_text(&self, out: &mut Layout, body: &[u32], streaming: bool) {
        if streaming {
            self.row(out, body[0], &[self.codec_bos]);
            out.feed.extend_from_slice(&body[1..]);
            out.feed.push(self.text_eos);
        } else {
            for &t in body.iter().chain([&self.text_eos]) {
                self.row(out, t, &[self.codec_pad]);
            }
            self.row(out, self.text_pad, &[self.codec_bos]);
        }
    }

    fn start(
        &self,
        text: &[u32],
        instructions: &[u32],
        language: &str,
        voice: Option<&Voice>,
    ) -> Result<(Layout, usize), String> {
        if self.frame_codes == 0 || self.language_at > self.codec_language.len() {
            return Err("inconsistent layout".into());
        }
        let codec = match self.language(language, voice)? {
            Some(l) => {
                let mut c = self.codec_language.clone();
                c.insert(self.language_at, l);
                c
            }
            None => self.codec_auto.clone(),
        };
        let mut out = Layout {
            ids: Vec::new(),
            codes: Vec::new(),
            feed: Vec::new(),
            tail: self.text_pad,
        };
        for &t in instructions.iter().chain(&text[..self.lead]) {
            self.row(&mut out, t, &[]);
        }
        for code in codec {
            self.row(&mut out, self.text_pad, &[code]);
        }
        let speaker_row = out.ids.len();
        self.row(
            &mut out,
            self.text_pad,
            &[voice.map_or(self.none_code, |v| v.code)],
        );
        self.row(&mut out, self.text_bos, &[self.codec_pad]);
        Ok((out, speaker_row))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen() -> DualStream {
        DualStream {
            text_template: "<|im_start|>assistant\n{input}<|im_end|>\n<|im_start|>assistant\n"
                .into(),
            instructions_template: Some("<|im_start|>user\n{instructions}<|im_end|>\n".into()),
            lead: 3,
            drop: 5,
            text_pad: 151671,
            text_bos: 151672,
            text_eos: 151673,
            none_code: 3072,
            frame_codes: 2,
            codec_language: vec![2154, 2156, 2157],
            language_at: 2,
            codec_auto: vec![2155, 2156, 2157],
            codec_pad: 2148,
            codec_bos: 2149,
            languages: [
                ("english", 2050),
                ("chinese", 2055),
                ("sichuan_dialect", 2062),
            ]
            .map(|(k, v)| (k.to_owned(), v))
            .into(),
            voices: [
                ("ryan", 3061, None),
                ("eric", 2875, Some("sichuan_dialect")),
            ]
            .map(|(k, c, d)| {
                (
                    k.to_owned(),
                    Voice {
                        code: c,
                        dialect: d.map(str::to_owned),
                    },
                )
            })
            .into(),
            dialect_languages: vec!["chinese".into(), "auto".into()],
        }
    }

    const N: u32 = 3072;
    // role (3), "a b c", <|im_end|> \n <|im_start|> assistant \n
    const TEXT: [u32; 11] = [1, 2, 3, 10, 11, 12, 4, 5, 1, 2, 3];

    #[test]
    fn streaming_matches_the_reference_rows() {
        let l = qwen().layout(&TEXT, &[], "Ryan", "English", true).unwrap();
        let (p, b) = (151671, 151672);
        assert_eq!(l.ids, [1, 2, 3, p, p, p, p, p, b, 10]);
        let first: Vec<u32> = l.codes.iter().step_by(2).copied().collect();
        assert_eq!(first, [N, N, N, 2154, 2156, 2050, 2157, 3061, 2148, 2149]);
        assert!(l.codes.iter().skip(1).step_by(2).all(|&c| c == N));
        assert_eq!(l.feed, [11, 12, 151673]);
        assert_eq!(l.tail, p);
    }

    #[test]
    fn auto_dialect_and_non_streaming() {
        let d = qwen();
        let l = d.layout(&TEXT, &[7, 8], "eric", "auto", false).unwrap();
        let first: Vec<u32> = l.codes.iter().step_by(2).copied().collect();
        // Dialect voices speak their dialect for `auto`; the text rides the prompt.
        assert_eq!(
            first,
            [N, N, N, N, N, 2154, 2156, 2062, 2157, 2875, 2148, 2148, 2148, 2148, 2148, 2149]
        );
        assert_eq!(l.ids[..5], [7, 8, 1, 2, 3]);
        assert_eq!(l.ids[11..], [10, 11, 12, 151673, 151671]);
        assert!(l.feed.is_empty());
        let a = d.layout(&TEXT, &[], "ryan", "auto", true).unwrap();
        let first: Vec<u32> = a.codes.iter().step_by(2).copied().collect();
        assert_eq!(first[3..], [2155, 2156, 2157, 3061, 2148, 2149]);
        assert!(d.layout(&TEXT[..8], &[], "ryan", "english", true).is_err());
        assert!(d.layout(&TEXT, &[], "nobody", "english", true).is_err());
        assert!(d.layout(&TEXT, &[], "ryan", "klingon", true).is_err());
    }

    const REF: [u32; 10] = [1, 2, 3, 20, 21, 4, 5, 1, 2, 3];

    fn reference(codes: &[u32]) -> Reference<'_> {
        Reference {
            text: &REF,
            codes,
            codebook_size: 16,
        }
    }

    #[test]
    fn reference_text_longer_than_audio_continues_on_decode() {
        let d = qwen();
        let c = d
            .layout_conditioned(
                &TEXT,
                &[7, 8],
                "english",
                true,
                Some(reference(&[0, 1, 2, 3])),
            )
            .unwrap();
        assert_eq!(c.speaker_row, 9);
        assert_eq!(c.layout.ids[11..], [20, 21, 10]);
        assert_eq!(c.layout.codes[22..], [2149, N, 0, 1, 2, 3]);
        assert_eq!(c.layout.feed, [11, 12, 151673]);
        assert_eq!(c.layout.codes[18..20], [N, N]);
    }

    #[test]
    fn reference_audio_at_least_as_long_as_text_uses_pad_tail() {
        for codes in [
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9][..],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
        ] {
            let c = qwen()
                .layout_conditioned(&TEXT, &[], "english", true, Some(reference(codes)))
                .unwrap();
            assert_eq!(&c.layout.ids[9..15], [20, 21, 10, 11, 12, 151673]);
            assert!(c.layout.ids[15..].iter().all(|&id| id == 151671));
            assert!(c.layout.feed.is_empty());
            assert_eq!(c.layout.tail, 151671);
            assert_eq!(c.layout.codes[20..], *codes);
        }
    }

    #[test]
    fn non_streaming_reference_keeps_text_and_audio_spans_separate() {
        let d = qwen();
        let c = d
            .layout_conditioned(&TEXT, &[], "auto", false, Some(reference(&[0, 1, 2, 3])))
            .unwrap();
        assert_eq!(c.speaker_row, 6);
        assert_eq!(
            c.layout.ids[8..],
            [20, 21, 10, 11, 12, 151673, 151671, 151671, 151671]
        );
        assert!(c.layout.feed.is_empty());
        assert_eq!(
            c.layout.codes[16..],
            [2148, N, 2148, N, 2148, N, 2148, N, 2148, N, 2148, N, 2149, N, 0, 1, 2, 3]
        );
    }

    #[test]
    fn audio_only_conditioning_changes_only_the_speaker_row() {
        let d = qwen();
        for language in ["auto", "english", "chinese"] {
            for streaming in [false, true] {
                let c = d
                    .layout_conditioned(&TEXT, &[7], language, streaming, None)
                    .unwrap();
                let mut expected = d.layout(&TEXT, &[7], "ryan", language, streaming).unwrap();
                expected.codes[c.speaker_row * d.frame_codes] = N;
                assert_eq!(c.layout, expected);
            }
        }
        let mut base = qwen();
        base.voices.clear();
        assert!(base
            .layout_conditioned(&TEXT, &[], "english", true, None)
            .is_ok());
    }

    #[test]
    fn malformed_reference_frames_and_text_are_rejected() {
        let d = qwen();
        for codes in [&[][..], &[1][..], &[16, 0][..], &[u32::MAX, 0][..]] {
            assert!(d
                .layout_conditioned(&TEXT, &[], "english", true, Some(reference(codes)))
                .is_err());
        }
        for size in [0, N + 1] {
            let r = Reference {
                codebook_size: size,
                ..reference(&[0, 1])
            };
            assert!(d
                .layout_conditioned(&TEXT, &[], "english", true, Some(r))
                .is_err());
        }
        let r = Reference {
            text: &REF[..8],
            ..reference(&[0, 1])
        };
        assert!(d
            .layout_conditioned(&TEXT, &[], "english", true, Some(r))
            .is_err());
        assert!(d
            .layout_conditioned(&TEXT[..8], &[], "english", true, None)
            .is_err());
        assert!(d
            .layout_conditioned(&TEXT, &[], "unknown", true, None)
            .is_err());
    }

    #[test]
    fn malformed_layout_geometry_is_rejected_without_panicking() {
        let mut d = qwen();
        d.frame_codes = 0;
        assert!(d
            .layout_conditioned(&TEXT, &[], "auto", true, None)
            .is_err());
        d = qwen();
        d.lead = usize::MAX;
        assert!(d
            .layout_conditioned(&TEXT, &[], "auto", true, None)
            .is_err());
        d = qwen();
        d.drop = usize::MAX;
        assert!(d
            .layout_conditioned(&TEXT, &[], "auto", true, None)
            .is_err());
    }

    #[test]
    #[ignore = "CPU reference tensors; DUAL_STREAM_REFERENCE_CASES JSON, generated from installed reference"]
    fn conditioned_layout_matches_reference_tensors() {
        #[derive(serde::Deserialize)]
        struct Case {
            contract: DualStream,
            text: Vec<u32>,
            instructions: Vec<u32>,
            language: String,
            streaming: bool,
            voice: Option<String>,
            reference_text: Option<Vec<u32>>,
            reference_codes: Vec<u32>,
            codebook_size: u32,
            rows: Vec<Vec<f32>>,
            feed: Vec<Vec<f32>>,
            speaker_row: Option<usize>,
            speaker_overlay: Vec<f32>,
        }
        let path = std::env::var("DUAL_STREAM_REFERENCE_CASES").unwrap();
        let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert!(cases.len() >= 300);
        for (index, case) in cases.iter().enumerate() {
            let d = &case.contract;
            let reference = case.reference_text.as_ref().map(|text| Reference {
                text,
                codes: &case.reference_codes,
                codebook_size: case.codebook_size,
            });
            let (layout, speaker) = match &case.voice {
                Some(voice) => (
                    d.layout(
                        &case.text,
                        &case.instructions,
                        voice,
                        &case.language,
                        case.streaming,
                    )
                    .unwrap(),
                    None,
                ),
                None => {
                    let c = d
                        .layout_conditioned(
                            &case.text,
                            &case.instructions,
                            &case.language,
                            case.streaming,
                            reference,
                        )
                        .unwrap();
                    (c.layout, Some(c.speaker_row))
                }
            };
            assert_eq!(speaker, case.speaker_row, "case {index}");
            assert_eq!(layout.codes.len(), layout.ids.len() * d.frame_codes);
            let render_text = |id| {
                let mut row = vec![0f32; d.frame_codes + 2];
                row[0] = id as f32;
                row
            };
            let rows: Vec<_> = layout
                .ids
                .iter()
                .enumerate()
                .map(|(i, &id)| {
                    if speaker == Some(i) {
                        return case.speaker_overlay.clone();
                    }
                    let mut row = render_text(id);
                    for (k, &code) in layout.codes[i * d.frame_codes..(i + 1) * d.frame_codes]
                        .iter()
                        .enumerate()
                    {
                        if code != d.none_code {
                            row[k + 1] = (code + 1) as f32;
                        }
                    }
                    row
                })
                .collect();
            assert_eq!(rows, case.rows, "prompt rows case {index}");
            let feed: Vec<_> = if layout.feed.is_empty() {
                vec![render_text(layout.tail)]
            } else {
                layout.feed.iter().map(|&id| render_text(id)).collect()
            };
            assert_eq!(feed, case.feed, "decode feed case {index}");
        }
    }
}
