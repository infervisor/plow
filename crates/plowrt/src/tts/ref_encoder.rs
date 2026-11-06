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

    fn install_speaker_verifier() {
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
    #[ignore = "SPEAKER_CHECKPOINT and SPEAKER_PACKET_OUT; trained CPU-only sidecar emit with approved Lean verifier"]
    fn emit_trained_speaker_packet() {
        install_speaker_verifier();
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
        install_speaker_verifier();
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
