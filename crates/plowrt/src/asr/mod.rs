pub mod audio_lm;
pub mod conformer;
pub mod endpoint;
pub mod frontend;
#[cfg(feature = "gguf")]
pub mod nemotron;
pub mod numerals;
mod packet;
pub mod rnnt;
pub mod serving;
pub mod subsampling;
pub mod vad;

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

/// Audio for one stream of [`Transcriber::stream_push_round`].
#[derive(Clone, Copy)]
pub struct StreamPush<'a> {
    pub id: u64,
    pub samples: &'a [f32],
    /// The recording ends here: the text is its final transcript (what [`Transcriber::transcribe`]
    /// answers for the whole recording) and the stream closes.
    pub finish: bool,
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

    /// `transcribe`, reporting the transcript so far each time it grows (streamed deltas). An
    /// engine that decodes in one step reports the final text once.
    fn transcribe_streaming(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: &str,
        cancel: &std::sync::atomic::AtomicBool,
        on_text: &mut dyn FnMut(&str),
    ) -> crate::Result<Transcript> {
        let transcript = self.transcribe(samples, language, context, cancel)?;
        on_text(&transcript.text);
        Ok(transcript)
    }

    /// Open an incremental (cache-aware) stream, or `None` when the engine has none: then partial
    /// transcripts re-run [`Transcriber::transcribe`] on the audio so far.
    fn stream_open(&mut self) -> crate::Result<Option<u64>> {
        Ok(None)
    }

    /// Append audio to stream `id`; returns the transcript of the audio decoded so far (it trails
    /// the audio by the encoder's lookahead), reporting each growth through `on_text`.
    fn stream_push(
        &mut self,
        id: u64,
        samples: &[f32],
        on_text: &mut dyn FnMut(&str),
    ) -> crate::Result<String> {
        let _ = (id, samples, on_text);
        Err(crate::RuntimeError::Rejected(
            "this ASR engine has no incremental stream".into(),
        ))
    }

    fn stream_close(&mut self, id: u64) {
        let _ = id;
    }

    /// Whether [`Transcriber::stream_push_round`] answers finals (`finish`).
    fn stream_finals(&self) -> bool {
        false
    }

    /// Streams that can be open at once; `None`: no fixed limit.
    fn stream_slots(&self) -> Option<usize> {
        None
    }

    /// Appends each push's audio to its stream (one push per id) and steps the next window of
    /// each, their steps sharing launches. A push answers its stream's text once the stream has no
    /// window left (a finished stream then closes); `None`: push it again, with no more audio, to
    /// step the next.
    fn stream_push_round(
        &mut self,
        pushes: &[StreamPush<'_>],
    ) -> Vec<crate::Result<Option<String>>> {
        push_alone(self, pushes)
    }

    /// [`Transcriber::stream_push`] for several streams at once (one per id), in rounds of
    /// [`Transcriber::stream_push_round`]. Returns each stream's text in order.
    fn stream_push_batch(&mut self, pushes: &[StreamPush<'_>]) -> Vec<crate::Result<String>> {
        let mut answers: Vec<Option<crate::Result<String>>> = pushes.iter().map(|_| None).collect();
        let mut round: Vec<(usize, StreamPush<'_>)> = pushes.iter().copied().enumerate().collect();
        while !round.is_empty() {
            let current: Vec<StreamPush<'_>> = round.iter().map(|(_, push)| *push).collect();
            let results = self.stream_push_round(&current);
            let mut next = Vec::with_capacity(round.len());
            for ((index, push), result) in round.into_iter().zip(results) {
                match result {
                    Ok(None) => next.push((
                        index,
                        StreamPush {
                            samples: &[],
                            ..push
                        },
                    )),
                    Ok(Some(text)) => answers[index] = Some(Ok(text)),
                    Err(error) => answers[index] = Some(Err(error)),
                }
            }
            round = next;
        }
        answers
            .into_iter()
            .map(|answer| answer.expect("every push is answered"))
            .collect()
    }

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

/// Each push through [`Transcriber::stream_push`], all its windows at once (no finals).
fn push_alone<T: Transcriber + ?Sized>(
    engine: &mut T,
    pushes: &[StreamPush<'_>],
) -> Vec<crate::Result<Option<String>>> {
    pushes
        .iter()
        .map(|push| {
            if push.finish {
                return Err(crate::RuntimeError::Rejected(
                    "this ASR engine answers no finals from its stream".into(),
                ));
            }
            engine
                .stream_push(push.id, push.samples, &mut |_| {})
                .map(Some)
        })
        .collect()
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

    fn transcribe_streaming(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: &str,
        cancel: &std::sync::atomic::AtomicBool,
        on_text: &mut dyn FnMut(&str),
    ) -> crate::Result<Transcript> {
        (**self).transcribe_streaming(samples, language, context, cancel, on_text)
    }

    fn stream_open(&mut self) -> crate::Result<Option<u64>> {
        (**self).stream_open()
    }

    fn stream_push(
        &mut self,
        id: u64,
        samples: &[f32],
        on_text: &mut dyn FnMut(&str),
    ) -> crate::Result<String> {
        (**self).stream_push(id, samples, on_text)
    }

    fn stream_close(&mut self, id: u64) {
        (**self).stream_close(id)
    }

    fn stream_finals(&self) -> bool {
        (**self).stream_finals()
    }

    fn stream_slots(&self) -> Option<usize> {
        (**self).stream_slots()
    }

    fn stream_push_round(
        &mut self,
        pushes: &[StreamPush<'_>],
    ) -> Vec<crate::Result<Option<String>>> {
        (**self).stream_push_round(pushes)
    }

    fn stream_push_batch(&mut self, pushes: &[StreamPush<'_>]) -> Vec<crate::Result<String>> {
        (**self).stream_push_batch(pushes)
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
        "rnnt.greedy.v1" => {
            let engine = packet::PacketRnntTranscriber::load(packet, backend)?;
            let loaded_backend = engine.backend();
            (Box::new(engine) as Box<dyn Transcriber>, loaded_backend)
        }
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

fn load_causal_transcriber(
    packet: &Path,
    tokenizer: &Path,
    backend: &str,
) -> crate::Result<(Box<dyn Transcriber>, &'static str)> {
    let (engine, loaded_backend) =
        audio_lm::AudioLmAsr::load_with_backend(packet, tokenizer, backend)?;
    Ok((Box::new(engine), loaded_backend))
}
