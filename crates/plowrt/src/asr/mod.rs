pub mod conformer;
pub mod frontend;
#[cfg(feature = "gguf")]
pub mod nemotron;
#[cfg(feature = "gguf")]
mod packet;
pub mod qwen;
pub mod rnnt;
pub mod serving;
pub mod subsampling;

use std::path::Path;

use crate::exec::packet_runtime::PacketAsset;

#[derive(Clone, Debug, serde::Serialize)]
pub struct Transcript {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

#[derive(Clone, Copy)]
pub struct TranscriptionInput<'a> {
    pub samples: &'a [f32],
    pub language: Option<&'a str>,
    pub context: &'a str,
    pub cancel: &'a std::sync::atomic::AtomicBool,
}

#[derive(Clone, Copy, Debug)]
pub struct FinalizationPolicy {
    pub final_padding_samples: usize,
    pub final_padding_amplitude: f32,
}

impl Default for FinalizationPolicy {
    fn default() -> Self {
        Self {
            final_padding_samples: 0,
            final_padding_amplitude: 0.0,
        }
    }
}

pub trait Transcriber: Send {
    fn language(&self, requested: Option<&str>) -> crate::Result<Option<String>>;
    fn finalization_policy(&self) -> FinalizationPolicy {
        FinalizationPolicy::default()
    }
    fn batch_capacity(&self) -> usize {
        1
    }
    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: &str,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> crate::Result<Transcript>;

    fn transcribe_batch(
        &mut self,
        requests: &[TranscriptionInput<'_>],
    ) -> crate::Result<Vec<crate::Result<Transcript>>> {
        if requests.is_empty() || requests.len() > self.batch_capacity() {
            return Err(crate::RuntimeError::Rejected(format!(
                "ASR batch needs 1..={} recordings",
                self.batch_capacity()
            )));
        }
        Ok(requests
            .iter()
            .map(|request| {
                self.transcribe(
                    request.samples,
                    request.language,
                    request.context,
                    request.cancel,
                )
            })
            .collect())
    }
}

impl<T: Transcriber + ?Sized> Transcriber for Box<T> {
    fn language(&self, requested: Option<&str>) -> crate::Result<Option<String>> {
        (**self).language(requested)
    }

    fn batch_capacity(&self) -> usize {
        (**self).batch_capacity()
    }

    fn finalization_policy(&self) -> FinalizationPolicy {
        (**self).finalization_policy()
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: &str,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> crate::Result<Transcript> {
        (**self).transcribe(samples, language, context, cancel)
    }

    fn transcribe_batch(
        &mut self,
        requests: &[TranscriptionInput<'_>],
    ) -> crate::Result<Vec<crate::Result<Transcript>>> {
        (**self).transcribe_batch(requests)
    }
}

pub struct LoadedPacketTranscriber {
    pub pipeline: String,
    pub driver: String,
    pub backend: &'static str,
    pub engine: Box<dyn Transcriber>,
}

pub fn load_packet_transcriber(
    packet: &Path,
    tokenizer: &Path,
    backend: &str,
) -> crate::Result<LoadedPacketTranscriber> {
    let asset = PacketAsset::load(packet)?;
    let mut supported = asset
        .pipelines()
        .iter()
        .filter(|pipeline| matches!(pipeline.driver.as_str(), "rnnt.greedy.v1" | "causal.v1"));
    let pipeline = supported.next().ok_or_else(|| {
        crate::RuntimeError::Rejected("packet has no supported ASR pipeline".into())
    })?;
    if supported.next().is_some() {
        return Err(crate::RuntimeError::Rejected(
            "packet has multiple ASR pipelines; selection is ambiguous".into(),
        ));
    }
    let pipeline_name = pipeline.name.clone();
    let driver = pipeline.driver.clone();
    let (engine, loaded_backend) = match driver.as_str() {
        "rnnt.greedy.v1" => load_rnnt_transcriber(packet, tokenizer, backend)?,
        "causal.v1" => load_causal_transcriber(packet, tokenizer, backend)?,
        _ => unreachable!(),
    };
    Ok(LoadedPacketTranscriber {
        pipeline: pipeline_name,
        driver,
        backend: loaded_backend,
        engine,
    })
}

#[cfg(feature = "gguf")]
fn load_rnnt_transcriber(
    packet: &Path,
    tokenizer: &Path,
    backend: &str,
) -> crate::Result<(Box<dyn Transcriber>, &'static str)> {
    let engine = packet::PacketRnntTranscriber::load(packet, tokenizer, backend)?;
    let loaded_backend = engine.backend();
    Ok((Box::new(engine), loaded_backend))
}

#[cfg(not(feature = "gguf"))]
fn load_rnnt_transcriber(
    _packet: &Path,
    _tokenizer: &Path,
    _backend: &str,
) -> crate::Result<(Box<dyn Transcriber>, &'static str)> {
    Err(crate::RuntimeError::Rejected(
        "RNNT ASR packets require the gguf feature".into(),
    ))
}

fn load_causal_transcriber(
    packet: &Path,
    tokenizer: &Path,
    backend: &str,
) -> crate::Result<(Box<dyn Transcriber>, &'static str)> {
    let (engine, loaded_backend) = qwen::QwenAsr::load_with_backend(packet, tokenizer, backend)?;
    Ok((Box::new(engine), loaded_backend))
}

pub fn qwen_audio_rows(frames: usize) -> usize {
    13 * (frames / 100) + (frames % 100).div_ceil(8)
}

pub fn parse_qwen_output(output: &str, forced_language: Option<&str>) -> crate::Result<Transcript> {
    let output = output.trim();
    if output.is_empty() {
        return Ok(Transcript {
            text: String::new(),
            language: None,
        });
    }
    let (language, text) = if let Some(language) = forced_language {
        (Some(language.to_owned()), output)
    } else if let Some((header, text)) = output.split_once("<asr_text>") {
        let language = header
            .strip_prefix("language ")
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| crate::RuntimeError::Rejected("invalid ASR language header".into()))?;
        let language = language.lines().next().unwrap().trim();
        (
            (!language.eq_ignore_ascii_case("none")).then(|| language.to_owned()),
            text,
        )
    } else {
        return Err(crate::RuntimeError::Rejected(
            "missing ASR text marker".into(),
        ));
    };
    Ok(Transcript {
        text: text.trim().to_owned(),
        language,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_lengths_cover_all_tails() {
        for frames in 0usize..=3000 {
            let expected: usize = (0..frames)
                .step_by(100)
                .map(|start| {
                    let mut length = (frames - start).min(100);
                    for _ in 0..3 {
                        length = length.div_ceil(2);
                    }
                    length
                })
                .sum();
            assert_eq!(qwen_audio_rows(frames), expected);
        }
    }

    #[test]
    fn parse_language_and_text() {
        let result = parse_qwen_output("language English<asr_text>Hello.", None).unwrap();
        assert_eq!(result.language.as_deref(), Some("English"));
        assert_eq!(result.text, "Hello.");
        assert_eq!(
            parse_qwen_output("你好", Some("Chinese")).unwrap().text,
            "你好"
        );
        assert!(parse_qwen_output("Hello.", None).is_err());
        assert!(parse_qwen_output("language None<asr_text>", None)
            .unwrap()
            .language
            .is_none());
        assert_eq!(parse_qwen_output("", None).unwrap().text, "");
    }
}
