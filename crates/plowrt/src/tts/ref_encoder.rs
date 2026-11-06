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
/// Recent clips' codes, by exact PCM: a repeated reference skips the launch.
const CACHE: usize = 16;

pub struct RefEncoder {
    runtime: Mutex<Box<dyn PacketRuntime>>,
    /// (pcm, codes), least recently used first.
    cache: Mutex<Vec<(Vec<f32>, Vec<u32>)>>,
    /// (frames, program), ascending.
    capacities: Vec<(usize, usize)>,
    pcm: PacketTensor,
    codes: PacketTensor,
    pub sample_rate: u32,
    pub frame_samples: usize,
    pub frame_codes: usize,
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;

    #[test]
    #[ignore = "SPEAKER_CHECKPOINT and SPEAKER_PACKET_OUT; trained CPU-only sidecar emit with approved Lean verifier"]
    fn emit_trained_speaker_packet() {
        devgen::install_sidecar_verifier(Box::new(|requests| {
            let (certs, hash) =
                lean_verify::call_batch_bound(requests).map_err(|e| e.to_string())?;
            let registry: serde_json::Value = serde_json::from_str(include_str!(
                "../../../../lean-plow/approved-verifiers.json"
            ))
            .unwrap();
            if !registry["verifiers"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v["sha256"] == hash)
            {
                return Err(format!("speaker verifier {hash} is not approved"));
            }
            let certs = certs
                .iter()
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            Ok(Some((certs, hash)))
        }));
        let dir = std::env::var("SPEAKER_CHECKPOINT").unwrap();
        let out = std::env::var("SPEAKER_PACKET_OUT").unwrap();
        let (model, section) =
            devgen::speaker_encoder::lower_speaker_encoder(Path::new(&dir), 132, 90).unwrap();
        devgen::write_sidecar_packet(Path::new(&out), &model, &[section]);
        let receipts: plow_asset::certificates::SidecarCheckReceipts = serde_json::from_slice(
            &std::fs::read(plow_asset::certificates::sidecar_checks_file(Path::new(
                &out,
            )))
            .unwrap(),
        )
        .unwrap();
        receipts
            .validate_packet(&plow_asset::decode_objects::image_sha256(
                &std::fs::read(out).unwrap(),
            ))
            .unwrap();
        assert_eq!(receipts.program_checks.len(), 5);
        assert!(receipts.program_checks.iter().all(Option::is_some));
        println!("all five speaker programs have approved Lean logical-effect receipts");
    }

    #[test]
    #[ignore = "SPEAKER_PACKET, SPEAKER_REFERENCE_JSON, SPEAKER_RESULT_JSON and optional SPEAKER_BACKEND; numerical gate, no benchmark"]
    fn speaker_packet_matches_trained_reference() {
        let path = std::env::var("SPEAKER_PACKET").unwrap();
        let reference = std::env::var("SPEAKER_REFERENCE_JSON").unwrap();
        let result_path = std::env::var("SPEAKER_RESULT_JSON").unwrap();
        let backend = std::env::var("SPEAKER_BACKEND").unwrap_or_else(|_| "cpu".into());
        let options = crate::exec::cpu::engine::CpuEngineOpts {
            threads: 4,
            ..Default::default()
        };
        let mut runtime: Box<dyn PacketRuntime> = if backend == "cpu" {
            Box::new(
                crate::exec::cpu::engine::CpuEngine::load_packet(Path::new(&path), &options)
                    .unwrap(),
            )
        } else {
            load_packet_runtime_on(Path::new(&path), &backend, 0)
                .unwrap()
                .runtime
        };
        let asset = PacketAsset::load(Path::new(&path)).unwrap();
        let pipeline = asset
            .bind_driver("speaker.encode.v1", runtime.as_ref())
            .unwrap();
        let data: serde_json::Value =
            serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
        let mut results = Vec::new();
        let mut all_pass = true;
        for case in data["cases"].as_array().unwrap() {
            let bytes = std::fs::read(case["path"].as_str().unwrap()).unwrap();
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let input = tensors.tensor("input").unwrap();
            let rows = input.shape()[1];
            let raw_mel = tensors.tensor("mel").unwrap();
            let raw_mel: Vec<f32> = raw_mel
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let mel: Vec<f32> = (0..rows * 128)
                .map(|i| raw_mel[(i % 128) * rows + i / 128])
                .collect();
            let mut first_output = None;
            let mut capacities: Vec<_> = pipeline
                .programs()
                .filter_map(|(role, program)| {
                    Some((
                        role.strip_prefix("encode.r")?.parse::<usize>().ok()?,
                        program,
                    ))
                })
                .filter(|&(cap, _)| cap >= rows)
                .collect();
            capacities.sort_unstable();
            for &(cap, program) in capacities.iter().take(2) {
                let poisoned = vec![f32::NAN; cap * 128];
                runtime
                    .write_tensor_at(
                        pipeline.tensor("mel").unwrap(),
                        0,
                        bytemuck::cast_slice(&poisoned),
                    )
                    .unwrap();
                runtime
                    .write_tensor_at(
                        pipeline.tensor("mel").unwrap(),
                        0,
                        bytemuck::cast_slice(&mel),
                    )
                    .unwrap();
                runtime
                    .write_tensor_at(
                        pipeline.tensor("length").unwrap(),
                        0,
                        &(rows as u32).to_le_bytes(),
                    )
                    .unwrap();
                runtime.run(program).unwrap();
                for (reference, actual) in [
                    ("blocks.0", "act.spk.block0"),
                    ("blocks.1", "act.spk.block1"),
                    ("blocks.2", "act.spk.block2"),
                    ("blocks.3", "act.spk.block3"),
                    ("mfa", "act.spk.mfa"),
                    ("asp", "act.spk.pooled"),
                    ("embedding", "act.spk.embedding"),
                ] {
                    let expected = tensors.tensor(reference).unwrap();
                    let expected: Vec<f32> = expected
                        .data()
                        .chunks_exact(2)
                        .map(|b| {
                            f32::from_bits(
                                u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16,
                            )
                        })
                        .collect();
                    let embedding = reference == "embedding";
                    let vector = embedding || reference == "asp";
                    let width = expected.len() / if vector { 1 } else { rows };
                    let count = if vector { width } else { rows * width };
                    let mut actual_bytes = vec![0; count * 4];
                    runtime
                        .read_tensor_at(runtime.tensor(actual).unwrap(), 0, &mut actual_bytes)
                        .unwrap();
                    let actual: Vec<f32> = actual_bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    let expected: Vec<f32> = (0..count)
                        .map(|i| {
                            if vector {
                                expected[i]
                            } else {
                                expected[(i % width) * rows + i / width]
                            }
                        })
                        .collect();
                    let finite = actual.iter().all(|v| v.is_finite());
                    let max_abs = actual
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    let dot: f64 = actual
                        .iter()
                        .zip(&expected)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let norm =
                        |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
                    let cosine = dot / (norm(&actual) * norm(&expected)).max(1e-30);
                    let pass = finite && max_abs <= 2e-3 && cosine >= 0.9999;
                    if embedding {
                        all_pass &= pass;
                        if let Some(first) = &first_output {
                            assert_eq!(&actual, first, "capacity changed output");
                        } else {
                            first_output = Some(actual.clone());
                        }
                    }
                    let row = serde_json::json!({"case": case["case"], "rows": rows, "capacity": cap,
                        "stage": reference, "finite": finite, "max_abs": max_abs, "cosine": cosine,
                        "pass": pass});
                    println!("{row}");
                    results.push(row);
                }
            }
        }
        std::fs::write(
            result_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "backend": backend, "scope": "trained speaker numerical parity, no benchmark",
                "max_abs_limit": 0.002, "cosine_min": 0.9999,
                "pass": all_pass, "results": results,
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(all_pass, "speaker embedding numerical gate failed");
    }
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
            cache: Mutex::new(Vec::new()),
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
        {
            let mut cache = self.cache.lock();
            if let Some(i) = cache.iter().position(|(clip, _)| clip.as_slice() == pcm) {
                let hit = cache.remove(i);
                let codes = hit.1.clone();
                cache.push(hit);
                return Ok(codes);
            }
        }
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
        drop(rt);
        let mut cache = self.cache.lock();
        if cache.len() == CACHE {
            cache.remove(0);
        }
        cache.push((pcm.to_vec(), codes.clone()));
        Ok(codes)
    }
}
