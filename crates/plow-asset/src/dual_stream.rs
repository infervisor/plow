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
    fn language(&self, language: &str, voice: &Voice) -> Result<Option<u32>, String> {
        let language = language.to_lowercase();
        if let Some(d) = voice
            .dialect
            .as_ref()
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
        if text.len() < self.lead + self.drop + 1 {
            return Err("empty input".into());
        }
        let mut codec = match self.language(language, v)? {
            Some(l) => {
                let mut c = self.codec_language.clone();
                c.insert(self.language_at, l);
                c
            }
            None => self.codec_auto.clone(),
        };
        codec.extend([v.code, self.codec_pad, self.codec_bos]);
        let mut out = Layout {
            ids: Vec::new(),
            codes: Vec::new(),
            feed: Vec::new(),
            tail: self.text_pad,
        };
        let mut row = |id: u32, code: u32| {
            out.ids.push(id);
            out.codes.push(code);
            out.codes
                .extend(std::iter::repeat_n(self.none_code, self.frame_codes - 1));
        };
        for &t in instructions.iter().chain(&text[..self.lead]) {
            row(t, self.none_code);
        }
        let n = codec.len();
        for (i, &c) in codec[..n - 1].iter().enumerate() {
            row(
                if i + 2 < n {
                    self.text_pad
                } else {
                    self.text_bos
                },
                c,
            );
        }
        let body = &text[self.lead..text.len() - self.drop];
        if streaming {
            row(body[0], self.codec_bos);
            out.feed = body[1..].to_vec();
            out.feed.push(self.text_eos);
        } else {
            for &t in body.iter().chain([&self.text_eos]) {
                row(t, self.codec_pad);
            }
            row(self.text_pad, self.codec_bos);
        }
        Ok(out)
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
}
