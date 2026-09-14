use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use gguf_rs_lib::format::metadata::MetadataValue;

use crate::asr::frontend::PacketLogMelFrontend;
use crate::asr::rnnt::{detokenize_sentencepiece, PacketRnnt};
use crate::asr::{Transcriber, Transcript};
use crate::asset::gguf::GgufFile;
use crate::{Result, RuntimeError};

pub struct PacketRnntTranscriber {
    frontend: PacketLogMelFrontend,
    execution: PacketRnnt,
    vocabulary: Vec<String>,
    language: String,
}

impl PacketRnntTranscriber {
    pub fn load(packet: &Path, model: &Path, backend: &str) -> Result<Self> {
        let model = GgufFile::open(model)?;
        let vocabulary = string_array(&model, "asr.tokenizer.vocab")?;
        let execution = PacketRnnt::load(packet, backend)?;
        let frontend = execution.log_mel_frontend()?;
        let prompt_index = usize::try_from(execution.parameter("prompt_index")?)
            .map_err(|_| rejected("prompt index overflows"))?;
        let language = prompt_language(&model, prompt_index)?;
        Ok(Self {
            frontend,
            execution,
            vocabulary,
            language,
        })
    }

    pub fn backend(&self) -> &'static str {
        self.execution.backend()
    }
}

impl Transcriber for PacketRnntTranscriber {
    fn language(&self, requested: Option<&str>) -> Result<Option<String>> {
        let Some(requested) = requested else {
            return Ok(Some(self.language.clone()));
        };
        let compatible = requested.eq_ignore_ascii_case(&self.language)
            || (self.language.eq_ignore_ascii_case("en-US")
                && matches!(requested.to_ascii_lowercase().as_str(), "en" | "english"));
        compatible
            .then(|| Some(self.language.clone()))
            .ok_or_else(|| {
                rejected(format!(
                    "packet was compiled for {}, not {requested}",
                    self.language
                ))
            })
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: &str,
        cancel: &AtomicBool,
    ) -> Result<Transcript> {
        if !context.is_empty() {
            return Err(rejected("speech context is not implemented"));
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(rejected("ASR cancelled"));
        }
        let language = self.language(language)?;
        let features = self.frontend.extract(samples)?;
        let valid_frames = features.frames;
        let expected = self.execution.input_elements();
        if features.values.len() > expected {
            return Err(rejected(format!(
                "audio produces {} feature values; packet capacity is {expected}",
                features.values.len()
            )));
        }
        let mut input = features.values;
        input.resize(expected, 0.0);
        let tokens = self
            .execution
            .transcribe_input_frames(&input, valid_frames)?;
        if cancel.load(Ordering::Relaxed) {
            return Err(rejected("ASR cancelled"));
        }
        Ok(Transcript {
            text: detokenize_sentencepiece(&self.vocabulary, &tokens),
            language,
        })
    }
}

fn prompt_language(model: &GgufFile, prompt_index: usize) -> Result<String> {
    let dictionary = string_array(model, "asr.rnnt.prompt_dictionary")?;
    dictionary
        .iter()
        .filter_map(|entry| entry.rsplit_once(':'))
        .find_map(|(language, index)| {
            (index.parse::<usize>().ok() == Some(prompt_index)).then(|| language.to_owned())
        })
        .ok_or_else(|| rejected(format!("prompt dictionary has no index {prompt_index}")))
}

fn string_array(model: &GgufFile, key: &str) -> Result<Vec<String>> {
    let Some(MetadataValue::Array(values)) = model.metadata().data.get(key) else {
        return Err(rejected(format!("missing {key}")));
    };
    values
        .values
        .iter()
        .map(|value| match value {
            MetadataValue::String(value) => Ok(value.clone()),
            _ => Err(rejected(format!("{key} contains a non-string"))),
        })
        .collect()
}

fn rejected(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!("invalid packet RNNT: {}", message.into()))
}
