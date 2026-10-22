//! A codec-encoder packet (`ref_encoder.pkt`, driver `codec.encode.v1`) beside a speech LM: one
//! reference clip in, its codes out, for voice cloning prompts. One launch per clip: the PCM is
//! zero-padded to the smallest capacity `encode.f{F}` that holds it.

use std::path::Path;

use parking_lot::Mutex;

use crate::exec::packet_runtime::{
    load_packet_runtime_on, PacketAsset, PacketRuntime, PacketTensor,
};

pub const PACKET: &str = "ref_encoder.pkt";
const DRIVER: &str = "codec.encode.v1";

pub struct RefEncoder {
    runtime: Mutex<Box<dyn PacketRuntime>>,
    /// (frames, program), ascending.
    capacities: Vec<(usize, usize)>,
    pcm: PacketTensor,
    codes: PacketTensor,
    pub sample_rate: u32,
    pub frame_samples: usize,
    pub frame_codes: usize,
}

impl RefEncoder {
    pub fn load(path: &Path, device: u8) -> Result<Self, String> {
        let e = |x: crate::RuntimeError| format!("{}: {x}", path.display());
        let runtime = load_packet_runtime_on(path, "cuda", device)
            .map_err(e)?
            .runtime;
        let asset = PacketAsset::load(path).map_err(e)?;
        let pipeline = asset.bind_driver(DRIVER, runtime.as_ref()).map_err(e)?;
        let param = |k: &str| pipeline.parameter(k).map_err(e);
        let mut capacities: Vec<(usize, usize)> = pipeline
            .programs()
            .filter_map(|(role, program)| {
                Some((role.strip_prefix("encode.f")?.parse().ok()?, program))
            })
            .collect();
        capacities.sort_unstable();
        if capacities.is_empty() {
            return Err(format!("{}: no encode capacity", path.display()));
        }
        Ok(RefEncoder {
            pcm: pipeline.tensor("pcm").map_err(e)?,
            codes: pipeline.tensor("codes").map_err(e)?,
            sample_rate: param("audio.sample_rate")? as u32,
            frame_samples: param("codec.frame_samples")? as usize,
            frame_codes: param("codec.frame_codes")? as usize,
            capacities,
            runtime: Mutex::new(runtime),
        })
    }

    /// Longest clip one launch takes, in samples.
    pub fn max_samples(&self) -> usize {
        self.capacities
            .last()
            .map_or(0, |&(f, _)| f * self.frame_samples)
    }

    /// `[ceil(n / frame_samples)][frame_codes]` codes of `pcm` (mono, at `sample_rate`).
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<u32>, String> {
        let frames = pcm.len().div_ceil(self.frame_samples);
        let &(cap, program) = self
            .capacities
            .iter()
            .find(|&&(f, _)| f >= frames && frames > 0)
            .ok_or_else(|| {
                format!(
                    "reference audio of {} samples; the encoder takes 1..={}",
                    pcm.len(),
                    self.max_samples()
                )
            })?;
        let mut padded = vec![0f32; cap * self.frame_samples];
        padded[..pcm.len()].copy_from_slice(pcm);
        let mut codes = vec![0u32; frames * self.frame_codes];
        let mut rt = self.runtime.lock();
        let e = |x: crate::RuntimeError| format!("reference encoder: {x}");
        rt.write_tensor_at(self.pcm, 0, bytemuck::cast_slice(&padded))
            .map_err(e)?;
        rt.run(program).map_err(e)?;
        rt.read_tensor_at(self.codes, 0, bytemuck::cast_slice_mut(&mut codes))
            .map_err(e)?;
        Ok(codes)
    }
}
