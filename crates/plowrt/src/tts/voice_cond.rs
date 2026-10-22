//! A voice-conditioning packet (`voice_cond.pkt`, driver `voice.encode.v1`) beside a guided speech
//! LM: one reference clip in, the LM's conditioning rows and the vocoder's voice slot out
//! (Chatterbox zero-shot cloning). One launch sequence per clip, on the smallest capacity
//! `encode.c{seconds}` that holds it.

use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;

use super::codec::SlotVoice;
use crate::exec::packet_runtime::{
    load_packet_runtime_on, PacketAsset, PacketRuntime, PacketTensor,
};

pub const PACKET: &str = "voice_cond.pkt";
const DRIVER: &str = "voice.encode.v1";

/// A cloned voice: the LM's conditioning rows (`[rows][hidden]`) and the vocoder's slot.
#[derive(Debug)]
pub struct ClonedVoice {
    pub rows: Vec<f32>,
    pub slot: Arc<SlotVoice>,
}

struct Io {
    pcm: PacketTensor,
    samples: PacketTensor,
    exaggeration: PacketTensor,
    rows: PacketTensor,
    tokens: PacketTensor,
    prompt_len: PacketTensor,
    prompt_mel: PacketTensor,
    spks: PacketTensor,
}

pub struct VoiceCond {
    runtime: Mutex<Box<dyn PacketRuntime>>,
    /// (samples, program sequence), ascending.
    capacities: Vec<(usize, Vec<usize>)>,
    io: Io,
    pub sample_rate: u32,
}

impl VoiceCond {
    pub fn load(path: &Path, device: u8) -> Result<Self, String> {
        let e = |x: crate::RuntimeError| format!("{}: {x}", path.display());
        let runtime = load_packet_runtime_on(path, "cuda", device)
            .map_err(e)?
            .runtime;
        let asset = PacketAsset::load(path).map_err(e)?;
        let pipeline = asset.bind_driver(DRIVER, runtime.as_ref()).map_err(e)?;
        let sample_rate = pipeline.parameter("audio.sample_rate").map_err(e)? as u32;
        let mut capacities = Vec::new();
        for (role, _) in pipeline.programs() {
            let Some(secs) = role
                .strip_prefix("encode.c")
                .and_then(|r| r.strip_suffix(".0"))
                .and_then(|s| s.parse::<usize>().ok())
            else {
                continue;
            };
            let seq = pipeline
                .program_sequence(&format!("encode.c{secs}"))
                .map_err(e)?;
            capacities.push((secs * sample_rate as usize, seq));
        }
        capacities.sort_unstable();
        if capacities.is_empty() {
            return Err(format!("{}: no encode capacity", path.display()));
        }
        let t = |k: &str| pipeline.tensor(k).map_err(e);
        let io = Io {
            pcm: t("pcm")?,
            samples: t("samples")?,
            exaggeration: t("exaggeration")?,
            rows: t("t3.rows")?,
            tokens: t("gen.tokens")?,
            prompt_len: t("gen.prompt_len")?,
            prompt_mel: t("gen.prompt_mel")?,
            spks: t("gen.spks")?,
        };
        Ok(VoiceCond {
            runtime: Mutex::new(runtime),
            capacities,
            io,
            sample_rate,
        })
    }

    /// Longest clip one launch takes, in samples.
    pub fn max_samples(&self) -> usize {
        self.capacities.last().map_or(0, |c| c.0)
    }

    /// The voice of `pcm` (mono, at `sample_rate`), emotion exaggeration `exaggeration`.
    pub fn encode(&self, pcm: &[f32], exaggeration: f32) -> Result<ClonedVoice, String> {
        let (cap, programs) = self
            .capacities
            .iter()
            .find(|c| c.0 >= pcm.len() && !pcm.is_empty())
            .ok_or_else(|| {
                format!(
                    "reference audio of {} samples; the voice encoder takes 1..={}",
                    pcm.len(),
                    self.max_samples()
                )
            })?;
        let mut padded = vec![0f32; *cap];
        padded[..pcm.len()].copy_from_slice(pcm);
        let io = &self.io;
        let mut rt = self.runtime.lock();
        let e = |x: crate::RuntimeError| format!("voice encoder: {x}");
        rt.write_tensor_at(io.pcm, 0, bytemuck::cast_slice(&padded))
            .map_err(e)?;
        rt.write_tensor(io.samples, &(pcm.len() as u32).to_le_bytes())
            .map_err(e)?;
        rt.write_tensor(io.exaggeration, &exaggeration.to_le_bytes())
            .map_err(e)?;
        rt.run_sequence(programs).map_err(e)?;
        let read_f32 = |rt: &Box<dyn PacketRuntime>, t: PacketTensor, n: usize| {
            let mut v = vec![0f32; n];
            rt.read_tensor_at(t, 0, bytemuck::cast_slice_mut(&mut v))
                .map(|_| v)
        };
        let mut p = [0u32; 1];
        rt.read_tensor(io.prompt_len, bytemuck::cast_slice_mut(&mut p))
            .map_err(e)?;
        let p = p[0] as usize;
        if p == 0 {
            return Err("reference audio is too short to clone".into());
        }
        let mut prompt_token = vec![0u32; p];
        rt.read_tensor_at(io.tokens, 0, bytemuck::cast_slice_mut(&mut prompt_token))
            .map_err(e)?;
        let rows = read_f32(&rt, io.rows, io.rows.bytes / 4).map_err(e)?;
        let prompt_feat = read_f32(&rt, io.prompt_mel, 2 * p * 80).map_err(e)?;
        let spks = read_f32(&rt, io.spks, 80).map_err(e)?;
        Ok(ClonedVoice {
            rows,
            slot: Arc::new(SlotVoice {
                prompt_token,
                prompt_feat,
                spks,
            }),
        })
    }
}
