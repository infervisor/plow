//! Packet-bound reference audio encoders for codec codes and speaker embeddings.
//! `codec.encode.v2` carries actual sample lengths; v1 pads PCM to the selected capacity.

use std::path::Path;

use parking_lot::Mutex;

use crate::exec::packet_runtime::{
    load_packet_runtime_on, PacketAsset, PacketData, PacketRuntime, PacketTensor, PacketWrite,
};

pub const PACKET: &str = "ref_encoder.pkt";
/// Recent clips' codes, by exact PCM: a repeated reference skips the launch.
const CACHE: usize = 16;

pub struct RefEncoder {
    runtime: Mutex<Box<dyn PacketRuntime>>,
    /// (pcm, codes), least recently used first.
    cache: Mutex<Vec<(Vec<f32>, Vec<u32>)>>,
    /// (frames, program), ascending.
    capacities: Vec<(usize, usize)>,
    pcm: PacketTensor,
    samples: Option<PacketTensor>,
    codes: PacketTensor,
    pub sample_rate: u32,
    pub frame_samples: usize,
    pub frame_codes: usize,
}

pub struct SpeakerEncoder {
    runtime: Mutex<Box<dyn PacketRuntime>>,
    frontend: crate::asr::frontend::PacketLogMelFrontend,
    plan: plow_asset::packet_pipeline::SpeakerEncoderPlan,
    mel: PacketTensor,
    length: PacketTensor,
    embedding: PacketTensor,
    pub sample_rate: u32,
    pub min_samples: usize,
    pub max_samples: usize,
}

impl SpeakerEncoder {
    pub fn load(path: &Path, device: u8) -> Result<Self, String> {
        let runtime = load_packet_runtime_on(path, "cuda", device)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .runtime;
        Self::bind(path, runtime)
    }

    fn bind(path: &Path, mut runtime: Box<dyn PacketRuntime>) -> Result<Self, String> {
        let e = |x: crate::RuntimeError| format!("{}: {x}", path.display());
        let asset = PacketAsset::load(path).map_err(e)?;
        let descriptor = asset
            .pipelines()
            .iter()
            .find(|p| p.name == "speaker.encode")
            .ok_or("speaker encoder pipeline is missing")?;
        let plan = descriptor.speaker_encoder_plan()?;
        let pipeline = asset.bind(&descriptor.name, runtime.as_ref()).map_err(e)?;
        if pipeline.parameter("audio.frontend.bins").map_err(e)? != plan.input_width as u64 {
            return Err("speaker frontend width disagrees with encoder input".into());
        }
        let frontend =
            crate::asr::frontend::PacketLogMelFrontend::bind(&pipeline, runtime.as_ref())
                .map_err(e)?;
        let min_samples = usize::try_from(pipeline.parameter("audio.min_samples").map_err(e)?)
            .map_err(|_| "speaker minimum sample count overflows")?;
        let max_samples = usize::try_from(pipeline.parameter("audio.max_samples").map_err(e)?)
            .map_err(|_| "speaker maximum sample count overflows")?;
        let sample_rate =
            u32::try_from(frontend.sample_rate()).map_err(|_| "speaker sample rate overflows")?;
        let mel = pipeline.tensor("mel").map_err(e)?;
        let length = pipeline.tensor("length").map_err(e)?;
        let embedding = pipeline.tensor("embedding").map_err(e)?;
        runtime.prepare_writes(&[mel, length]).map_err(e)?;
        Ok(Self {
            runtime: Mutex::new(runtime),
            frontend,
            plan,
            mel,
            length,
            embedding,
            sample_rate,
            min_samples,
            max_samples,
        })
    }

    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<f32>, String> {
        let features = self
            .frontend
            .extract(pcm)
            .map_err(|e| format!("speaker frontend: {e}"))?;
        self.encode_features(&features.values, features.frames, features.bins)
    }

    fn encode_features(
        &self,
        values: &[f32],
        rows: usize,
        width: usize,
    ) -> Result<Vec<f32>, String> {
        let program = self
            .plan
            .program(rows)
            .ok_or("speaker feature row count is outside packet bounds")?;
        if width != self.plan.input_width
            || rows.checked_mul(width) != Some(values.len())
            || values.iter().any(|v| !v.is_finite())
        {
            return Err("invalid speaker features".into());
        }
        let padded;
        let values = if self.plan.reflect_pad == 0 {
            values
        } else {
            padded = reflect_features(values, width, self.plan.reflect_pad);
            &padded
        };
        let count = (rows as u32).to_le_bytes();
        let mut output = vec![0f32; self.plan.output_width];
        let mut runtime = self.runtime.lock();
        let e = |x: crate::RuntimeError| format!("speaker encoder: {x}");
        runtime
            .write_tensors(&[
                PacketWrite {
                    tensor: self.mel,
                    offset: 0,
                    data: PacketData::Host(bytemuck::cast_slice(values)),
                },
                PacketWrite {
                    tensor: self.length,
                    offset: 0,
                    data: PacketData::Host(&count),
                },
            ])
            .map_err(e)?;
        runtime.run(program).map_err(e)?;
        runtime
            .read_tensor(self.embedding, bytemuck::cast_slice_mut(&mut output))
            .map_err(e)?;
        drop(runtime);
        if output.iter().any(|v| !v.is_finite()) {
            return Err("speaker encoder returned nonfinite embedding".into());
        }
        Ok(output)
    }
}

fn reflect_features(values: &[f32], width: usize, pad: usize) -> Vec<f32> {
    let rows = values.len() / width;
    assert_eq!(values.len(), rows * width);
    assert!(rows > pad);
    let mut out = Vec::with_capacity((rows + 2 * pad) * width);
    for i in 0..rows + 2 * pad {
        let mut row = i as isize - pad as isize;
        if row < 0 {
            row = -row;
        }
        if row >= rows as isize {
            row = 2 * rows as isize - 2 - row;
        }
        out.extend_from_slice(&values[row as usize * width..(row as usize + 1) * width]);
    }
    out
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;

    struct CountedRuntime {
        tensors: [Vec<u8>; 3],
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        fail_first: bool,
    }

    impl PacketRuntime for CountedRuntime {
        fn tensor(&self, _: &str) -> Option<PacketTensor> {
            None
        }

        fn write_tensor(&mut self, tensor: PacketTensor, bytes: &[u8]) -> crate::Result<()> {
            self.tensors[tensor.handle].copy_from_slice(bytes);
            Ok(())
        }

        fn read_tensor(&self, tensor: PacketTensor, bytes: &mut [u8]) -> crate::Result<()> {
            bytes.copy_from_slice(&self.tensors[tensor.handle]);
            Ok(())
        }

        fn copy_tensor(
            &mut self,
            _: PacketTensor,
            _: usize,
            _: PacketTensor,
            _: usize,
            _: usize,
        ) -> crate::Result<()> {
            unreachable!()
        }

        fn run(&mut self, program: usize) -> crate::Result<()> {
            assert_eq!(program, 0);
            let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.fail_first && call == 0 {
                return Err(crate::RuntimeError::Device(
                    "injected encoder failure".into(),
                ));
            }
            let code: [u8; 4] = self.tensors[0][..4].try_into().unwrap();
            for dst in self.tensors[2].chunks_exact_mut(4) {
                dst.copy_from_slice(&code);
            }
            Ok(())
        }

        fn last_run_us(&self) -> f64 {
            0.
        }
    }

    fn counted_encoder(
        samples: bool,
        fail_first: bool,
    ) -> (RefEncoder, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let encoder = RefEncoder {
            runtime: Mutex::new(Box::new(CountedRuntime {
                tensors: [vec![0; 16], vec![0; 4], vec![0; 16]],
                calls: calls.clone(),
                fail_first,
            })),
            capacities: vec![(2, 0)],
            pcm: PacketTensor {
                handle: 0,
                bytes: 16,
            },
            samples: samples.then_some(PacketTensor {
                handle: 1,
                bytes: 4,
            }),
            codes: PacketTensor {
                handle: 2,
                bytes: 16,
            },
            sample_rate: 24000,
            frame_samples: 2,
            frame_codes: 2,
            cache: Mutex::new(Vec::new()),
        };
        (encoder, calls)
    }

    #[test]
    fn concurrent_reference_cache_misses_share_one_execution() {
        for samples in [false, true] {
            let (encoder, calls) = counted_encoder(samples, false);
            let start = std::sync::Barrier::new(32);
            let pcm = [0.25, 0.5, 0.75];
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..32)
                    .map(|_| {
                        scope.spawn(|| {
                            assert!(encoder.cached(&pcm).is_none());
                            start.wait();
                            encoder.encode_miss(&pcm).unwrap()
                        })
                    })
                    .collect();
                for handle in handles {
                    assert_eq!(handle.join().unwrap(), [0.25f32.to_bits(); 4]);
                }
            });
            assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
            assert_eq!(
                encoder.encode(&[0.5, 0.5, 0.75]).unwrap(),
                [0.5f32.to_bits(); 4]
            );
            assert_eq!(encoder.encode(&pcm).unwrap(), [0.25f32.to_bits(); 4]);
            assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        }
    }

    #[test]
    fn failed_reference_encoding_is_not_cached() {
        for samples in [false, true] {
            let (encoder, calls) = counted_encoder(samples, true);
            let pcm = [0.25, 0.5, 0.75];
            assert!(encoder.encode(&pcm).is_err());
            assert!(encoder.cache.lock().is_empty());
            assert_eq!(encoder.encode(&pcm).unwrap(), [0.25f32.to_bits(); 4]);
            assert_eq!(encoder.encode(&pcm).unwrap(), [0.25f32.to_bits(); 4]);
            assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
            assert!(encoder.encode(&[]).is_err());
            assert!(encoder.encode(&[0.; 5]).is_err());
            assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        }
    }

    fn speaker_driver(asset: &PacketAsset) -> &str {
        let drivers: Vec<_> = asset
            .pipelines()
            .iter()
            .filter(|p| p.name == "speaker.encode")
            .collect();
        assert_eq!(drivers.len(), 1);
        let driver = drivers[0].driver.as_str();
        assert!(matches!(
            driver,
            "speaker.encode.v1" | "speaker.encode.v2" | "speaker.encode.v3"
        ));
        driver
    }

    #[test]
    fn reflected_feature_input_preserves_rows_and_boundaries() {
        let values = [0., 1., 2., 3., 4., 5.];
        assert_eq!(
            reflect_features(&values, 2, 1),
            [2., 3., 0., 1., 2., 3., 4., 5., 2., 3.]
        );
        assert_eq!(reflect_features(&values, 2, 0), values);
    }

    #[test]
    #[ignore = "SPEAKER_PACKET, SPEAKER_REFERENCE_JSON, SPEAKER_RESULT_JSON and optional SPEAKER_BACKEND; saved PCM through production binding, no benchmark"]
    fn speaker_binding_matches_saved_pcm_reference() {
        let path = std::path::PathBuf::from(std::env::var("SPEAKER_PACKET").unwrap());
        let backend = std::env::var("SPEAKER_BACKEND").unwrap_or_else(|_| "cpu".into());
        let encoder = if backend == "cuda" {
            SpeakerEncoder::load(&path, 0)
        } else {
            SpeakerEncoder::bind(
                &path,
                load_packet_runtime_on(&path, &backend, 0).unwrap().runtime,
            )
        }
        .unwrap();
        let fixture: serde_json::Value = serde_json::from_slice(
            &std::fs::read(std::env::var("SPEAKER_REFERENCE_JSON").unwrap()).unwrap(),
        )
        .unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 4);
        assert_eq!(encoder.sample_rate, 24000);
        assert!(encoder.encode(&[]).is_err());
        assert!(encoder.encode(&vec![0.; encoder.min_samples - 1]).is_err());
        assert!(encoder.encode(&vec![0.; encoder.max_samples + 1]).is_err());
        assert!(encoder
            .encode(&vec![f32::NAN; encoder.min_samples])
            .is_err());
        let mut all_pass = true;
        let mut checks = Vec::new();
        let mut outputs = Vec::new();
        for (iteration, case) in cases.iter().chain(cases.iter().rev()).enumerate() {
            let bytes = std::fs::read(case["path"].as_str().unwrap()).unwrap();
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let pcm = tensors.tensor("pcm").unwrap();
            assert_eq!(pcm.dtype(), safetensors::Dtype::F32);
            let pcm: Vec<f32> = pcm
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let expected = tensors.tensor("embedding").unwrap();
            assert_eq!(expected.dtype(), safetensors::Dtype::BF16);
            let expected: Vec<f32> = expected
                .data()
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16))
                .collect();
            assert_eq!(expected.len(), encoder.plan.output_width);
            encoder
                .runtime
                .lock()
                .write_tensor(
                    encoder.mel,
                    bytemuck::cast_slice(&vec![f32::NAN; encoder.mel.bytes / 4]),
                )
                .unwrap();
            let actual = encoder.encode(&pcm).unwrap();
            assert_eq!(actual.len(), expected.len());
            let max_abs = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            let dot: f64 = actual
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let cosine = dot / (norm(&actual) * norm(&expected)).max(1e-30);
            let finite = actual.iter().all(|v| v.is_finite());
            let pass = finite && max_abs <= 0.002 && cosine >= 0.9999;
            all_pass &= pass;
            let offset = outputs.len();
            outputs.extend_from_slice(&actual);
            checks.push(serde_json::json!({"case":case["case"],"iteration":iteration,"samples":pcm.len(),
                "max_abs":max_abs,"cosine":cosine,"finite":finite,"pass":pass,"output_offset_f32":offset}));
        }
        let result = std::path::PathBuf::from(std::env::var("SPEAKER_RESULT_JSON").unwrap());
        std::fs::write(result.with_extension("f32"), bytemuck::cast_slice(&outputs)).unwrap();
        std::fs::write(result, serde_json::to_vec_pretty(&serde_json::json!({
            "scope":"saved PCM through production speaker binding/frontend; no serving or timing qualification",
            "backend":backend,"max_abs_limit":0.002,"cosine_min":0.9999,"pass":all_pass,"checks":checks})).unwrap()).unwrap();
        assert!(
            all_pass,
            "production speaker binding differs from saved reference"
        );
    }

    #[test]
    #[ignore = "SPEAKER_PACKET, SPEAKER_REFERENCE_JSON, SPEAKER_FRONTEND_RESULT_JSON and RUST_MIN_STACK=67108864; saved PCM frontend parity, CPU only"]
    fn speaker_frontend_matches_trained_reference() {
        let path = std::env::var("SPEAKER_PACKET").unwrap();
        let reference = std::env::var("SPEAKER_REFERENCE_JSON").unwrap();
        let result_path = std::env::var("SPEAKER_FRONTEND_RESULT_JSON").unwrap();
        let runtime = crate::exec::cpu::engine::CpuEngine::load_packet(
            Path::new(&path),
            &crate::exec::cpu::engine::CpuEngineOpts {
                threads: 4,
                ..Default::default()
            },
        )
        .unwrap();
        let asset = PacketAsset::load(Path::new(&path)).unwrap();
        let pipeline = asset.bind_driver(speaker_driver(&asset), &runtime).unwrap();
        let frontend =
            crate::asr::frontend::PacketLogMelFrontend::bind(&pipeline, &runtime).unwrap();
        assert_eq!(frontend.sample_rate(), 24_000);
        let data: serde_json::Value =
            serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
        let mut results = Vec::new();
        let mut all_pass = true;
        for case in data["cases"].as_array().unwrap() {
            let bytes = std::fs::read(case["path"].as_str().unwrap()).unwrap();
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let pcm = tensors.tensor("pcm").unwrap();
            let mel = tensors.tensor("mel").unwrap();
            let input = tensors.tensor("input").unwrap();
            assert_eq!(pcm.dtype(), safetensors::Dtype::F32);
            assert_eq!(mel.dtype(), safetensors::Dtype::F32);
            assert_eq!(input.dtype(), safetensors::Dtype::BF16);
            assert_eq!(pcm.shape()[0], 1);
            assert_eq!(&mel.shape()[..2], &[1, 128]);
            let floats = |t: &safetensors::tensor::TensorView<'_>| -> Vec<f32> {
                t.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            };
            let samples = floats(&pcm);
            let expected = floats(&mel);
            let got = frontend.extract(&samples).unwrap();
            let rows = mel.shape()[2];
            assert_eq!(input.shape(), &[1, rows, 128]);
            assert_eq!(got.frames, rows, "{}", case["case"]);
            assert_eq!(got.bins, 128);
            assert_eq!(got.values.len(), expected.len());
            assert!(got.frames as u64 >= pipeline.parameter("input.min_rows").unwrap());
            assert!(got.frames as u64 <= pipeline.parameter("input.max_rows").unwrap());
            let mut max_abs = 0f32;
            let mut error = 0f64;
            let mut norm = 0f64;
            let mut bf16_changed = 0usize;
            let mut bf16_max_abs = 0f32;
            for (i, &v) in got.values.iter().enumerate() {
                let target = expected[(i % 128) * rows + i / 128];
                max_abs = max_abs.max((v - target).abs());
                error += (f64::from(v) - f64::from(target)).powi(2);
                norm += f64::from(target).powi(2);
                let target = u16::from_le_bytes(input.data()[i * 2..i * 2 + 2].try_into().unwrap());
                let bits = v.to_bits();
                let actual =
                    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000);
                bf16_changed += usize::from(actual.to_bits() >> 16 != u32::from(target));
                bf16_max_abs =
                    bf16_max_abs.max((actual - f32::from_bits(u32::from(target) << 16)).abs());
            }
            let relative_l2 = (error / norm).sqrt();
            let pass =
                got.values.iter().all(|v| v.is_finite()) && max_abs <= 2e-4 && relative_l2 <= 1e-5;
            all_pass &= pass;
            let row = serde_json::json!({"case": case["case"], "samples": samples.len(),
                "frames": rows, "max_abs": max_abs, "relative_l2": relative_l2, "pass": pass,
                "bf16_input_changed": bf16_changed, "bf16_input_elements": got.values.len(),
                "bf16_input_max_abs": bf16_max_abs});
            println!("{row}");
            results.push(row);
        }
        assert_eq!(results.len(), 4);
        std::fs::write(result_path, serde_json::to_vec_pretty(&serde_json::json!({
            "scope": "packet-bound F32 frontend vs saved trained reference, CPU only; no benchmark",
            "max_abs_limit": 2e-4, "relative_l2_limit": 1e-5,
            "pass": all_pass, "results": results,
        })).unwrap()).unwrap();
        assert!(all_pass, "speaker frontend numerical gate failed");
    }

    fn install_sidecar_verifier() {
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
    }

    #[test]
    #[ignore = "REF_CODEC_CHECKPOINT, REF_CODEC_PACKET, optional REF_CODEC_GPU/REF_CODEC_ROUTE; CPU-only codec emit with approved Lean verifier"]
    fn emit_trained_reference_codec() {
        install_sidecar_verifier();
        let dir = std::env::var("REF_CODEC_CHECKPOINT").unwrap();
        let out = std::path::PathBuf::from(std::env::var("REF_CODEC_PACKET").unwrap());
        let gpu = std::env::var("REF_CODEC_GPU").unwrap_or_else(|_| "H100 SXM5".into());
        let (mut model, section) = devgen::mimi_encoder::lower_mimi_encoder(
            Path::new(&dir),
            132,
            packet::devbuild::gpu_fingerprint(&gpu),
        )
        .unwrap();
        let route = std::env::var("REF_CODEC_ROUTE").unwrap_or_else(|_| "emitted".into());
        assert!(matches!(
            route.as_str(),
            "emitted" | "fp32_rvq" | "fp32_projection" | "fp32_classifier"
        ));
        for p in &mut model.progs {
            for d in &mut p.insts {
                if d.op != packet::dev::DevOp::Conv1dF32 as u16 {
                    continue;
                }
                let name = &model.tensors[d.t[2] as usize].name;
                let projection = matches!(name.as_str(), "w.sem.proj" | "w.ac.proj");
                let classifier = name.starts_with("w.sem.E") || name.starts_with("w.ac.E");
                if (route == "fp32_rvq" && (projection || classifier))
                    || (route == "fp32_projection" && projection)
                    || (route == "fp32_classifier" && classifier)
                {
                    assert_eq!((d.i[4], d.i[7]), (1, 1));
                    d.j[1] &= !(1 << 13);
                }
            }
        }
        println!(
            "codec object requires ops mask 0x{:x}",
            devgen::manifest::speech_ops(&model)
        );
        let mut plans = Vec::new();
        let weight = model
            .tensors
            .iter()
            .position(|t| t.name == "w.down.w")
            .unwrap() as u32;
        for (program, p) in model.progs.iter().enumerate() {
            let down = p
                .insts
                .iter()
                .find(|d| d.op == packet::dev::DevOp::Conv1dF32 as u16 && d.t[2] == weight)
                .unwrap();
            plans.push(serde_json::json!({"program": program, "frames": model.prog_t[program],
                "xf": model.tensors[down.t[1] as usize].name, "down": model.tensors[down.t[0] as usize].name}));
        }
        devgen::write_sidecar_packet(&out, &model, &[section]);
        let receipts: plow_asset::certificates::SidecarCheckReceipts = serde_json::from_slice(
            &std::fs::read(plow_asset::certificates::sidecar_checks_file(&out)).unwrap(),
        )
        .unwrap();
        receipts
            .validate_packet(&plow_asset::decode_objects::image_sha256(
                &std::fs::read(&out).unwrap(),
            ))
            .unwrap();
        assert_eq!(receipts.program_checks.len(), model.progs.len());
        assert!(receipts.program_checks.iter().all(Option::is_some));
        std::fs::write(
            out.with_extension("plan.json"),
            serde_json::to_vec_pretty(&plans).unwrap(),
        )
        .unwrap();
        println!(
            "all {} codec programs have approved receipts",
            model.progs.len()
        );
    }

    #[test]
    #[ignore = "REF_CODEC_PACKET, REF_CODEC_REFERENCE, REF_CODEC_RESULT_JSON and optional REF_CODEC_BACKEND; saved-input correctness only"]
    fn reference_codec_matches_saved_reference() {
        let path = std::path::PathBuf::from(std::env::var("REF_CODEC_PACKET").unwrap());
        let backend = std::env::var("REF_CODEC_BACKEND").unwrap_or_else(|_| "cpu".into());
        let runtime = load_packet_runtime_on(&path, &backend, 0).unwrap().runtime;
        let encoder = RefEncoder::bind(&path, runtime).unwrap();
        let fixture: serde_json::Value = serde_json::from_slice(
            &std::fs::read(std::env::var("REF_CODEC_REFERENCE").unwrap()).unwrap(),
        )
        .unwrap();
        let plans: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(path.with_extension("plan.json")).unwrap())
                .unwrap();
        assert!(
            encoder.samples.is_some(),
            "actual-length gate requires codec.encode.v2"
        );
        assert_eq!(
            u64::from(encoder.sample_rate),
            fixture["sample_rate"].as_u64().unwrap()
        );
        assert_eq!(
            encoder.frame_samples as u64,
            fixture["frame_samples"].as_u64().unwrap()
        );
        assert_eq!(
            encoder.frame_codes as u64,
            fixture["frame_codes"].as_u64().unwrap()
        );
        assert!(encoder.encode(&[]).is_err());
        assert!(encoder
            .encode(&vec![0.; encoder.max_samples() + 1])
            .is_err());
        let decode_f32 = |bytes: &[u8]| -> Vec<f32> {
            assert_eq!(bytes.len() % 4, 0);
            bytes
                .chunks_exact(4)
                .map(|p| f32::from_le_bytes(p.try_into().unwrap()))
                .collect()
        };
        let mut checks = Vec::new();
        let mut output_data = Vec::new();
        let mut all_pass = true;
        for case in fixture["cases"].as_array().unwrap() {
            let pcm = decode_f32(&std::fs::read(case["pcm"].as_str().unwrap()).unwrap());
            assert_eq!(pcm.len() as u64, case["samples"].as_u64().unwrap());
            let frames = pcm.len().div_ceil(encoder.frame_samples);
            assert_eq!(frames as u64, case["frames"].as_u64().unwrap());
            let reference = std::fs::read(case["codes"].as_str().unwrap()).unwrap();
            assert_eq!(reference.len(), frames * encoder.frame_codes * 4);
            let first = encoder.encode(&pcm).unwrap();
            let api_exact = bytemuck::cast_slice::<u32, u8>(&first) == reference;
            assert_eq!(first, encoder.encode(&pcm).unwrap(), "cached codes changed");
            all_pass &= api_exact;
            let mut previous = None;
            for &(cap, program) in encoder
                .capacities
                .iter()
                .filter(|&&(cap, _)| cap >= frames)
                .take(2)
            {
                let mut rt = encoder.runtime.lock();
                let poison = vec![f32::NAN; cap * encoder.frame_samples];
                let count = (pcm.len() as u32).to_le_bytes();
                rt.write_tensors(&[
                    PacketWrite {
                        tensor: encoder.pcm,
                        offset: 0,
                        data: PacketData::Host(bytemuck::cast_slice(&poison)),
                    },
                    PacketWrite {
                        tensor: encoder.pcm,
                        offset: 0,
                        data: PacketData::Host(bytemuck::cast_slice(&pcm)),
                    },
                    PacketWrite {
                        tensor: encoder.samples.unwrap(),
                        offset: 0,
                        data: PacketData::Host(&count),
                    },
                ])
                .unwrap();
                rt.run(program).unwrap();
                let mut codes = vec![0u8; reference.len()];
                rt.read_tensor_at(encoder.codes, 0, &mut codes).unwrap();
                let changed = codes
                    .chunks_exact(4)
                    .zip(reference.chunks_exact(4))
                    .filter(|(a, b)| a != b)
                    .count();
                let capacity_equal = previous.as_ref().is_none_or(|old| old == &codes);
                let codes_offset = output_data.len();
                output_data.extend_from_slice(&codes);
                previous = Some(codes);
                let mut stages = Vec::new();
                let plan = plans.iter().find(|p| p["program"] == program).unwrap();
                for name in ["xf", "down"] {
                    let reference = &case[name];
                    let expected =
                        decode_f32(&std::fs::read(reference["path"].as_str().unwrap()).unwrap());
                    assert_eq!(
                        expected.len() as u64,
                        reference["rows"].as_u64().unwrap() * reference["width"].as_u64().unwrap()
                    );
                    let tensor = rt.tensor(plan[name].as_str().unwrap()).unwrap();
                    let mut bytes = vec![0; expected.len() * 4];
                    rt.read_tensor_at(tensor, 0, &mut bytes).unwrap();
                    let output_offset = output_data.len();
                    output_data.extend_from_slice(&bytes);
                    let got = decode_f32(&bytes);
                    let finite = got.iter().all(|x| x.is_finite());
                    let max_abs = got
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let rms = (expected.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>()
                        / expected.len() as f64)
                        .sqrt();
                    all_pass &= finite;
                    stages.push(serde_json::json!({"stage":name,"finite":finite,"max_abs":max_abs,"max_abs_over_rms":f64::from(max_abs)/rms.max(1e-30),
                        "output_offset_bytes":output_offset,"rows":reference["rows"],"width":reference["width"]}));
                }
                let pass = api_exact && changed == 0 && capacity_equal;
                all_pass &= pass;
                let result = serde_json::json!({"case":case["case"],"samples":pcm.len(),"frames":frames,"capacity":cap,
                    "api_exact":api_exact,"changed_codes":changed,"capacity_equal":capacity_equal,"pass":pass,"stages":stages,"codes_offset_bytes":codes_offset});
                println!("{result}");
                checks.push(result);
            }
        }
        let result_path = std::path::PathBuf::from(std::env::var("REF_CODEC_RESULT_JSON").unwrap());
        let output_path = result_path.with_extension("bin");
        std::fs::write(&output_path, output_data).unwrap();
        std::fs::write(result_path, serde_json::to_vec_pretty(&serde_json::json!({
            "scope":"native codec saved-input correctness; discrete codes strict; floating stage errors diagnostic; no timings",
            "backend":backend,"pass":all_pass,"checks":checks,"output_data":output_path})).unwrap()).unwrap();
        assert!(all_pass, "native codec differs from the saved reference");
    }

    #[test]
    #[ignore = "SPEAKER_CHECKPOINT and SPEAKER_PACKET_OUT; trained CPU-only sidecar emit with approved Lean verifier"]
    fn emit_trained_speaker_packet() {
        install_sidecar_verifier();
        let dir = std::env::var("SPEAKER_CHECKPOINT").unwrap();
        let out = std::env::var("SPEAKER_PACKET_OUT").unwrap();
        let (model, section) =
            devgen::speaker_encoder::lower_speaker_encoder(Path::new(&dir), 132, 90).unwrap();
        let programs = model.progs.len();
        println!(
            "speaker object requires ops mask 0x{:x}",
            devgen::manifest::speech_ops(&model)
        );
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
        assert_eq!(receipts.program_checks.len(), programs);
        assert!(receipts.program_checks.iter().all(Option::is_some));
        println!("all {programs} speaker programs have approved Lean logical-effect receipts");
    }

    fn speaker_layer_slice(
        model: &packet::devbuild::Model,
        program: usize,
        key: &str,
        split_bf16: bool,
        continuous_weights: &std::collections::BTreeMap<u32, u32>,
    ) -> (packet::devbuild::Program, u32, u32, usize) {
        use packet::dev::{DevOp, ACT_ROUND_BF16};
        let insts = &model.progs[program].insts;
        let weight = format!("w.{key}.weight");
        let packed = format!("part.{weight}.");
        let bias = model
            .tensors
            .iter()
            .position(|t| t.name == format!("w.{key}.bias"))
            .unwrap() as u32;
        let first = insts
            .iter()
            .position(|d| {
                matches!(d.op, x if x == DevOp::Conv1dF32 as u16 || x == DevOp::DenseGemmF32 as u16)
                    && model
                        .tensors
                        .get(d.t[2] as usize)
                        .is_some_and(|t| t.name == weight || t.name.starts_with(&packed))
            })
            .unwrap();
        let start = if insts[first].op == DevOp::DenseGemmF32 as u16 {
            assert_eq!(insts[first - 1].op, DevOp::CopyColsF32 as u16);
            first - 1
        } else {
            first
        };
        let add = insts
            .iter()
            .enumerate()
            .skip(first)
            .find(|(_, d)| d.op == DevOp::BinaryF32 as u16 && d.t[2] == bias && d.i[3] == 0)
            .unwrap()
            .0;
        assert_eq!(insts[add + 1].op, DevOp::UnaryF32 as u16);
        assert_eq!(insts[add + 1].i[2], ACT_ROUND_BF16);
        let input = insts[start].t[1];
        let output = insts[add].t[0];
        assert_eq!(insts[add + 1].t[..2], [output, output]);
        let mut builder = packet::devbuild::Builder::new(model.n_cu);
        builder.adopt_tensors(
            model
                .tensors
                .iter()
                .map(|t| packet::devbuild::TensorDecl {
                    name: t.name.clone(),
                    bytes: t.bytes,
                    init: None,
                })
                .collect(),
        );
        let mut previous = None;
        let convolution = insts[first].op == DevOp::Conv1dF32 as u16;
        let padded = convolution
            && insts[first].t[6] != packet::dev::TENSOR_NONE
            && model.tensors[insts[first].t[6] as usize].name == "act.spk.padded_length";
        let mut selected = Vec::new();
        if padded && !split_bf16 {
            assert_eq!(insts[0].op, DevOp::LengthAffineU32 as u16);
            selected.push(&insts[0]);
        }
        selected.extend(&insts[start..=add + 1]);
        let pad = if (padded || split_bf16) && convolution {
            (insts[first].i[4] - 1) * insts[first].i[6] / 2
        } else {
            0
        };
        for original in selected {
            let mut source = *original;
            let mut round = None;
            if source.op == DevOp::DenseGemmF32 as u16 {
                if let Some(&weight) = continuous_weights.get(&source.t[2]) {
                    assert_eq!(source.i[7] & !1, 4 | 8);
                    assert_eq!(&source.i[3..7], &[0; 4]);
                    assert!(source.t[3..].iter().all(|&h| h == packet::dev::TENSOR_NONE));
                    let (rows, width, k) = (source.i[0], source.i[1], source.i[2]);
                    assert_eq!(k % 4, 0);
                    if source.i[7] & 1 != 0 {
                        round = Some((rows, width));
                    }
                    source.op = DevOp::Conv1dF32 as u16;
                    source.t[2] = weight;
                    source.i = [1, rows, k, width, 1, 1, 1, 1];
                    source.j = [0, (1 << 13) | (1 << 14) | (1 << 15)];
                }
            }
            if split_bf16 && source.op == DevOp::Conv1dF32 as u16 {
                if !padded {
                    source.i[1] += 2 * pad;
                }
                source.j[0] = 0;
                source.j[1] = (source.j[1] & !(3 | (1 << 16) | (3 << 17))) | (1 << 14);
                if source.t[6] != packet::dev::TENSOR_NONE {
                    source.t[6] = model
                        .tensors
                        .iter()
                        .position(|t| t.name == "in.spk.length")
                        .unwrap() as u32;
                }
                assert!(
                    u64::from(source.i[1]) * u64::from(source.i[2]) * 4
                        <= model.tensors[input as usize].bytes
                );
            }
            let op = *DevOp::ALL
                .iter()
                .find(|&&op| op as u16 == source.op)
                .unwrap();
            let cus = (0..u32::from(source.blocks))
                .map(|cu| cu % model.n_cu)
                .collect();
            previous =
                Some(
                    builder.emit(op, cus, &previous.into_iter().collect::<Vec<_>>(), |d| {
                        d.t = source.t;
                        d.i = source.i;
                        d.f = source.f;
                        d.j = source.j;
                    }),
                );
            if let Some((rows, width)) = round {
                let cus = (0..u32::from(source.blocks))
                    .map(|cu| cu % model.n_cu)
                    .collect();
                previous = Some(
                    builder.emit(DevOp::UnaryF32, cus, &[previous.unwrap()], |d| {
                        d.t[..2].copy_from_slice(&[source.t[0], source.t[0]]);
                        d.i[..3].copy_from_slice(&[rows, width, ACT_ROUND_BF16]);
                    }),
                );
            }
        }
        (builder.finish(), input, output, pad as usize)
    }

    #[test]
    #[ignore = "SPEAKER_CHECKPOINT, SPEAKER_PACKET_OUT, SPEAKER_REFERENCE_JSON, SPEAKER_LAYER_PLAN; CPU-only layer replay emit"]
    fn emit_trained_speaker_layers() {
        install_sidecar_verifier();
        let dir = std::env::var("SPEAKER_CHECKPOINT").unwrap();
        let out = std::env::var("SPEAKER_PACKET_OUT").unwrap();
        let route = std::env::var("SPEAKER_LAYER_ROUTE").unwrap_or_else(|_| "emitted".into());
        assert!(matches!(
            route.as_str(),
            "emitted" | "split_bf16" | "continuous_bf16"
        ));
        let split_bf16 = route == "split_bf16";
        let reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(std::env::var("SPEAKER_REFERENCE_JSON").unwrap()).unwrap(),
        )
        .unwrap();
        let (mut model, section) =
            devgen::speaker_encoder::lower_speaker_encoder(Path::new(&dir), 132, 90).unwrap();
        let mut continuous_weights = std::collections::BTreeMap::new();
        if route == "continuous_bf16" {
            for index in 0..model.tensors.len() {
                let t = &model.tensors[index];
                if !t.name.starts_with("part.") {
                    continue;
                }
                let source = t.init.as_ref().unwrap();
                assert_eq!(source.len() as u64, t.bytes);
                assert_eq!(source.len() % 2, 0);
                let bytes: Vec<_> = source
                    .chunks_exact(2)
                    .flat_map(|p| [0, 0, p[0], p[1]])
                    .collect();
                let tensor = packet::devbuild::TensorDecl {
                    name: format!("wide.{}", t.name),
                    bytes: bytes.len() as u64,
                    init: Some(bytes),
                };
                continuous_weights.insert(index as u32, model.tensors.len() as u32);
                model.tensors.push(tensor);
            }
            assert!(!continuous_weights.is_empty());
        }
        let metadata: plow_asset::packet_pipeline::PacketPipelines =
            serde_json::from_slice(&section.data).unwrap();
        let mut programs = Vec::new();
        let mut plans = Vec::new();
        let mut rows: Vec<_> = reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["mel_rows"].as_u64().unwrap() as usize)
            .collect();
        rows.sort_unstable();
        rows.dedup();
        for rows in rows {
            let cap = [128, 256, 512, 1024, 2048]
                .into_iter()
                .find(|&cap| cap >= rows)
                .unwrap();
            let source = metadata.pipelines[0].programs[&format!("encode.r{cap}.n{rows}")] as usize;
            for name in reference["convolutions"].as_object().unwrap().keys() {
                let key = name
                    .replace(".res2net_block.blocks.", ".res2net.")
                    .replace(".se_block.", ".se.");
                let (program, input, output, pad) =
                    speaker_layer_slice(&model, source, &key, split_bf16, &continuous_weights);
                let index = programs.len();
                programs.push(program);
                plans.push(
                    serde_json::json!({"rows": rows, "capacity": cap, "layer": name,
                    "program": index, "input": model.tensors[input as usize].name,
                    "output": model.tensors[output as usize].name, "reflect_pad": pad,
                    "length_add": if split_bf16 { 2 * pad } else { 0 }, "route": route}),
                );
            }
        }
        model.progs = programs;
        model.prog_t = plans
            .iter()
            .map(|p| p["capacity"].as_u64().unwrap() as u32)
            .collect();
        devgen::write_sidecar_packet(Path::new(&out), &model, &[]);
        let receipts: plow_asset::certificates::SidecarCheckReceipts = serde_json::from_slice(
            &std::fs::read(plow_asset::certificates::sidecar_checks_file(Path::new(
                &out,
            )))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(receipts.program_checks.len(), plans.len());
        assert!(receipts.program_checks.iter().all(Option::is_some));
        std::fs::write(
            std::env::var("SPEAKER_LAYER_PLAN").unwrap(),
            serde_json::to_vec_pretty(&plans).unwrap(),
        )
        .unwrap();
        println!(
            "{} isolated convolution programs emitted with approved receipts",
            plans.len()
        );
    }

    #[test]
    #[ignore = "SPEAKER_PACKET, SPEAKER_REFERENCE_JSON, SPEAKER_LAYER_PLAN, SPEAKER_RESULT_JSON, SPEAKER_BACKEND; layer arithmetic diagnosis, no benchmark"]
    fn speaker_layers_match_trained_reference() {
        let packet = std::env::var("SPEAKER_PACKET").unwrap();
        let backend = std::env::var("SPEAKER_BACKEND").unwrap();
        let result = std::path::PathBuf::from(std::env::var("SPEAKER_RESULT_JSON").unwrap());
        let reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(std::env::var("SPEAKER_REFERENCE_JSON").unwrap()).unwrap(),
        )
        .unwrap();
        let plans: Vec<serde_json::Value> = serde_json::from_slice(
            &std::fs::read(std::env::var("SPEAKER_LAYER_PLAN").unwrap()).unwrap(),
        )
        .unwrap();
        let mut runtime = load_packet_runtime_on(Path::new(&packet), &backend, 0)
            .unwrap()
            .runtime;
        let mut checks = Vec::new();
        let mut outputs = Vec::new();
        for case in reference["cases"].as_array().unwrap() {
            let bytes = std::fs::read(case["path"].as_str().unwrap()).unwrap();
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            for plan in plans.iter().filter(|p| p["rows"] == case["mel_rows"]) {
                let name = plan["layer"].as_str().unwrap();
                let x = tensors.tensor(&format!("conv.{name}.input")).unwrap();
                let y = tensors.tensor(&format!("conv.{name}.output")).unwrap();
                assert_eq!(x.dtype(), safetensors::Dtype::BF16);
                assert_eq!(y.dtype(), safetensors::Dtype::BF16);
                assert_eq!(x.shape()[0], 1);
                assert_eq!(y.shape()[0], 1);
                assert_eq!(x.shape()[2], y.shape()[2]);
                let rows = x.shape()[2];
                let decode = |t: &safetensors::tensor::TensorView<'_>| -> Vec<f32> {
                    let cols = t.shape()[1];
                    (0..rows * cols)
                        .map(|i| {
                            let at = ((i % cols) * rows + i / cols) * 2;
                            f32::from_bits(
                                u32::from(u16::from_le_bytes(
                                    t.data()[at..at + 2].try_into().unwrap(),
                                )) << 16,
                            )
                        })
                        .collect()
                };
                let values = reflect_features(
                    &decode(&x),
                    x.shape()[1],
                    plan["reflect_pad"].as_u64().unwrap() as usize,
                );
                let input = runtime.tensor(plan["input"].as_str().unwrap()).unwrap();
                let poison = vec![f32::NAN; input.bytes / 4];
                runtime
                    .write_tensor_at(input, 0, bytemuck::cast_slice(&poison))
                    .unwrap();
                runtime
                    .write_tensor_at(input, 0, bytemuck::cast_slice(&values))
                    .unwrap();
                let length = runtime.tensor("in.spk.length").unwrap();
                runtime
                    .write_tensor_at(
                        length,
                        0,
                        &((case["mel_rows"].as_u64().unwrap()
                            + plan["length_add"].as_u64().unwrap_or(0))
                            as u32)
                            .to_le_bytes(),
                    )
                    .unwrap();
                runtime
                    .run(plan["program"].as_u64().unwrap() as usize)
                    .unwrap();
                let expected = decode(&y);
                let mut actual = vec![0f32; expected.len()];
                let output = runtime.tensor(plan["output"].as_str().unwrap()).unwrap();
                runtime
                    .read_tensor_at(output, 0, bytemuck::cast_slice_mut(&mut actual))
                    .unwrap();
                let finite = actual.iter().all(|v| v.is_finite());
                let changed = actual
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let max_abs = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let offset = outputs.len();
                outputs.extend_from_slice(bytemuck::cast_slice(&actual));
                let check = serde_json::json!({"case":case["case"], "layer":name,"rows":rows,
                    "width":y.shape()[1],"elements":actual.len(),"changed":changed,"max_abs":max_abs,"finite":finite,
                    "output_offset_bytes":offset,"program":plan["program"],"route":plan["route"]});
                println!("{check}");
                checks.push(check);
            }
        }
        assert_eq!(
            checks.len(),
            reference["cases"].as_array().unwrap().len()
                * reference["convolutions"].as_object().unwrap().len()
        );
        let exact = checks.iter().filter(|c| c["changed"] == 0).count();
        let output_path = result.with_extension("f32");
        std::fs::write(&output_path, outputs).unwrap();
        std::fs::write(&result, serde_json::to_vec_pretty(&serde_json::json!({
            "scope":"native isolated convolution arithmetic diagnosis; no timings", "backend":backend,
            "calls":checks.len(),"exact":exact,"all_finite":checks.iter().all(|c| c["finite"] == true),
            "output_f32_le":output_path,"checks":checks})).unwrap()).unwrap();
        assert!(checks.iter().all(|c| c["finite"] == true));
        assert_eq!(
            exact,
            checks.len(),
            "isolated native convolutions differ from captured reference"
        );
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
            .bind_driver(speaker_driver(&asset), runtime.as_ref())
            .unwrap();
        let input_pad = pipeline
            .optional_parameter("input.reflect_pad")
            .unwrap_or(0) as usize;
        assert_eq!(
            pipeline.optional_parameter("input.reflect_pad").is_some(),
            pipeline.driver() == "speaker.encode.v3"
        );
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
            let padded_mel = reflect_features(&mel, 128, input_pad);
            let mut first_output = None;
            let mut capacities: Vec<_> = pipeline
                .programs()
                .filter_map(|(role, program)| {
                    let suffix = role.strip_prefix("encode.r")?;
                    let cap = match suffix.split_once(".n") {
                        Some((cap, n)) if n.parse::<usize>().ok() == Some(rows) => cap,
                        Some(_) => return None,
                        None => suffix,
                    };
                    Some((cap.parse::<usize>().ok()?, program))
                })
                .filter(|&(cap, _)| cap >= rows)
                .collect();
            capacities.sort_unstable();
            assert_eq!(
                capacities.len(),
                [128, 256, 512, 1024, 2048]
                    .into_iter()
                    .filter(|&cap| cap >= rows)
                    .count(),
                "missing speaker row/capacity binding"
            );
            for &(cap, program) in capacities.iter().take(2) {
                let poisoned = vec![f32::NAN; (cap + 2 * input_pad) * 128];
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
                        bytemuck::cast_slice(&padded_mel),
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
                "backend": backend, "scope": "trained speaker numerical parity, no benchmark", "input_reflect_pad": input_pad,
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
        Self::bind(path, runtime)
    }

    fn bind(path: &Path, mut runtime: Box<dyn PacketRuntime>) -> Result<Self, String> {
        let e = |x: crate::RuntimeError| format!("{}: {x}", path.display());
        let asset = PacketAsset::load(path).map_err(e)?;
        let mut encoders = asset
            .pipelines()
            .iter()
            .filter(|p| matches!(p.driver.as_str(), "codec.encode.v1" | "codec.encode.v2"));
        let descriptor = encoders.next().ok_or("codec encoder pipeline is missing")?;
        if encoders.next().is_some() {
            return Err("codec encoder pipeline is ambiguous".into());
        }
        let has_samples = descriptor.codec_sample_input()?;
        let pipeline = asset.bind(&descriptor.name, runtime.as_ref()).map_err(e)?;
        let samples = has_samples
            .then(|| pipeline.tensor("samples"))
            .transpose()
            .map_err(e)?;
        if let Some(samples) = samples {
            runtime
                .prepare_writes(&[pipeline.tensor("pcm").map_err(e)?, samples])
                .map_err(e)?;
        }
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
            samples,
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
        self.cached(pcm).map_or_else(|| self.encode_miss(pcm), Ok)
    }

    fn cached(&self, pcm: &[f32]) -> Option<Vec<u32>> {
        let mut cache = self.cache.lock();
        let i = cache.iter().position(|(clip, _)| clip.as_slice() == pcm)?;
        let hit = cache.remove(i);
        let codes = hit.1.clone();
        cache.push(hit);
        Some(codes)
    }

    fn encode_miss(&self, pcm: &[f32]) -> Result<Vec<u32>, String> {
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
        let mut rt = self.runtime.lock();
        if let Some(codes) = self.cached(pcm) {
            return Ok(codes);
        }
        let mut codes = vec![0u32; frames * self.frame_codes];
        let e = |x: crate::RuntimeError| format!("reference encoder: {x}");
        if let Some(samples) = self.samples {
            let count = u32::try_from(pcm.len())
                .map_err(|_| "reference sample count overflows")?
                .to_le_bytes();
            rt.write_tensors(&[
                PacketWrite {
                    tensor: self.pcm,
                    offset: 0,
                    data: PacketData::Host(bytemuck::cast_slice(pcm)),
                },
                PacketWrite {
                    tensor: samples,
                    offset: 0,
                    data: PacketData::Host(&count),
                },
            ])
            .map_err(e)?;
        } else {
            let mut padded = vec![0f32; cap * self.frame_samples];
            padded[..pcm.len()].copy_from_slice(pcm);
            rt.write_tensor_at(self.pcm, 0, bytemuck::cast_slice(&padded))
                .map_err(e)?;
        }
        rt.run(program).map_err(e)?;
        rt.read_tensor_at(self.codes, 0, bytemuck::cast_slice_mut(&mut codes))
            .map_err(e)?;
        // Publish before releasing the runtime so waiting misses reuse this execution.
        let mut cache = self.cache.lock();
        if cache.len() == CACHE {
            cache.remove(0);
        }
        cache.push((pcm.to_vec(), codes.clone()));
        Ok(codes)
    }
}
