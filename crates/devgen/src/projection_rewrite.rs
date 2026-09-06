use packet::dev::{DevOp, TENSOR_NONE};
use packet::devbuild::{Builder, Model, TensorDecl};
use plow_asset::decode_objects::{DecodeObject, DecodeObjects};
use std::{collections::BTreeMap, path::Path};
use tunedb::projection::{
    select_projection, ProjectionCell, ProjectionMeasurement, PROJECTION_ORACLE,
};

type Result<T> = std::result::Result<T, String>;
fn ordinary(d: &packet::dev::DevInst) -> bool {
    d.op == DevOp::Gemv as u16
        && matches!(d.i[0], 4 | 8 | 16)
        && d.i[3..].iter().all(|&v| v == 0)
        && d.t[3..].iter().all(|&v| v == TENSOR_NONE)
        && d.f[1].to_bits() == 0
        && d.j == [0, 0]
}
pub(crate) fn apply(
    model: &mut Model,
    config: &crate::emit_config::EmitConfig,
    gpu: &str,
    arch: &str,
    output: &Path,
) -> Result<Option<DecodeObjects>> {
    if !config.decode_projection_tuning {
        return Ok(None);
    }
    if arch != "sm_90a" {
        return Err("splitK projection tuning requires a Hopper object".into());
    }
    let path = config
        .decode_objects
        .as_ref()
        .ok_or("projection tuning requires compiled decode object bindings")?;
    let metadata: DecodeObjects =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let lo = packet::devbuild::decode_rung_lo(&model.prog_t);
    let programs: Vec<_> = model
        .prog_t
        .iter()
        .enumerate()
        .skip(lo)
        .map(|(i, &r)| (i, r))
        .collect();
    metadata.validate(
        &programs,
        model.n_cu,
        std::mem::size_of::<packet::dev::DevProgram>(),
    )?;
    crate::decode_objects::append_metadata(model, &mut Vec::new(), &metadata, arch, output)?;
    let Some(root) = config.tunedb_root() else {
        return Ok(None);
    };
    let spec = hwspec::registry::lookup(gpu)
        .ok_or("projection tuning requires a known hardware target")?;
    let hardware = kernelcaps::HardwareFingerprint::from_spec(spec)
        .ok_or("missing hardware fingerprint")?
        .tuning_path();
    let records = tunedb::TuneStore::new(root)
        .load_projection(&hardware)
        .map_err(|e| e.to_string())?;
    if records.is_empty() {
        return Ok(None);
    }
    let compiler = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .map_err(|e| e.to_string())?;
    if !compiler.status.success() {
        return Err("cannot identify CUDA toolchain".into());
    }
    let toolchain = String::from_utf8(compiler.stdout)
        .map_err(|e| e.to_string())?
        .trim()
        .to_owned();
    let implementation = plow_asset::decode_objects::image_sha256(include_bytes!(
        "../../../runtime/nvidia/op_gemm_splitk.cuh"
    ));
    let directory = output.parent().ok_or("missing output directory")?;
    let want = tunedb::Digests {
        implementation,
        interpreter: String::new(),
        toolchain,
        oracle: PROJECTION_ORACLE.into(),
    };
    let (selected, bindings) = plan(model, &metadata, &records, &hardware, &want, |object| {
        let image = std::fs::read(directory.join(&object.file)).map_err(|e| e.to_string())?;
        if !object.matches_image(&image) {
            return Err("projection object SHA256 mismatch".into());
        }
        Ok((
            plow_asset::cubin::global_u32(&image, "plow_gemm_splitk_abi"),
            plow_asset::cubin::global_u32(&image, "plow_arena_bytes")
                .ok_or("object arena export missing")?,
        ))
    })?;
    if rewrite(model, &selected)? == 0 {
        Ok(None)
    } else {
        Ok(Some(bindings))
    }
}

type Selection = BTreeMap<(usize, usize), u32>;
fn plan(
    model: &Model,
    metadata: &DecodeObjects,
    records: &[ProjectionMeasurement],
    hardware: &str,
    want: &tunedb::Digests,
    mut resources: impl FnMut(&DecodeObject) -> Result<(Option<u32>, u32)>,
) -> Result<(Selection, DecodeObjects)> {
    let mut selected = BTreeMap::new();
    let mut bindings = metadata.clone();
    for binding in &mut bindings.programs {
        let baseline = &metadata.objects[&binding.object];
        let mut candidate = None;
        for (pc, d) in model.progs[binding.index]
            .insts
            .iter()
            .enumerate()
            .filter(|(_, d)| ordinary(d))
        {
            let cell = ProjectionCell {
                hardware: hardware.into(),
                n_cu: model.n_cu,
                threads: baseline.threads,
                rows: d.i[0],
                n: d.i[1],
                k: d.i[2],
            };
            let mut best: Option<&ProjectionMeasurement> = None;
            for record in records {
                let mut digest = want.clone();
                digest.interpreter = record.candidate_object.sha256.clone();
                // Stale/failed records must not require or select a candidate object.
                if select_projection(
                    std::slice::from_ref(record),
                    &cell,
                    &digest,
                    baseline,
                    Some(1),
                    record.candidate_object.arena_bytes,
                )
                .is_none()
                {
                    continue;
                }
                let (abi, arena) = resources(&record.candidate_object)?;
                if select_projection(
                    std::slice::from_ref(record),
                    &cell,
                    &digest,
                    baseline,
                    abi,
                    arena,
                )
                .is_none()
                {
                    return Err("qualified projection object capability/resources mismatch".into());
                }
                if best.is_none_or(|old| record.stats.median_ns < old.stats.median_ns) {
                    best = Some(record);
                }
            }
            if let Some(record) = best {
                if candidate
                    .as_ref()
                    .is_some_and(|old| old != &record.candidate_object)
                {
                    return Err("selected projections require conflicting objects within one decode program".into());
                }
                candidate = Some(record.candidate_object.clone());
                selected.insert((binding.index, pc), record.split);
            }
        }
        if let Some(object) = candidate {
            let id =
                if let Some((&id, _)) = bindings.objects.iter().find(|(_, old)| **old == object) {
                    id
                } else {
                    let id = bindings
                        .objects
                        .last_key_value()
                        .map_or(Some(0), |(&id, _)| id.checked_add(1))
                        .ok_or("decode object id overflow")?;
                    bindings.objects.insert(id, object);
                    id
                };
            binding.object = id;
        }
    }
    bindings
        .objects
        .retain(|id, _| bindings.programs.iter().any(|p| p.object == *id));
    Ok((selected, bindings))
}

fn rewrite(model: &mut Model, selected: &BTreeMap<(usize, usize), u32>) -> Result<usize> {
    if selected.is_empty() {
        return Ok(0);
    }
    let lo = packet::devbuild::decode_rung_lo(&model.prog_t);
    let original = plow_asset::program::with_model(model, |p| {
        p.programs
            .iter()
            .enumerate()
            .skip(lo)
            .map(|(index, g)| plow_asset::splitk::dependencies(g).map(|deps| (index, deps)))
            .collect::<Result<BTreeMap<_, _>>>()
    })?;
    let mut sizes = BTreeMap::new();
    for (&(pi, pc), &s) in selected {
        let d = model
            .progs
            .get(pi)
            .and_then(|p| p.insts.get(pc))
            .ok_or("selection out of bounds")?;
        if pi < lo || !ordinary(d) || !matches!(s, 1 | 2 | 4 | 8 | 16) || model.prog_t[pi] != d.i[0]
        {
            return Err("selection is not an ordinary supported decode projection".into());
        }
        let bytes = u64::from(d.i[0]) * u64::from(d.i[1]) * 4;
        sizes
            .entry(d.t[0])
            .and_modify(|v: &mut u64| *v = (*v).max(bytes))
            .or_insert(bytes);
    }
    let mut partials = BTreeMap::new();
    for (output, bytes) in sizes {
        if model.tensors.len() >= packet::dev::TENSOR_NONE16 as usize {
            return Err("scratch handle overflow".into());
        }
        let name = format!("act.splitk.partial.{output}");
        if model.tensors.iter().any(|t| t.name == name) {
            return Err("scratch name collision".into());
        }
        partials.insert(output, model.tensors.len() as u32);
        model.tensors.push(TensorDecl {
            name,
            bytes: bytes + 256,
            init: None,
        });
    }
    for pi in lo..model.progs.len() {
        if !selected.keys().any(|&(p, _)| p == pi) {
            continue;
        }
        let old = &model.progs[pi];
        let mut placement: Vec<Vec<_>> = old
            .insts
            .iter()
            .map(|d| vec![u32::MAX; d.blocks as usize])
            .collect();
        for cu in 0..model.n_cu as usize {
            let offset = old.stream_ofs[cu] as usize;
            for e in &old.stream[offset..offset + old.stream_len[cu] as usize] {
                placement[e.inst as usize][e.slice as usize] = cu as u32;
            }
        }
        let mut b = Builder::new(model.n_cu);
        b.force_uniseg();
        b.adopt_tensors(model.tensors.clone());
        let mut map = Vec::new();
        let mut prior = BTreeMap::new();
        for (pc, d) in old.insts.iter().enumerate() {
            let mut deps: Vec<_> = original[&pi][pc].iter().map(|&p| map[p]).collect();
            let last = if let Some(&s) = selected.get(&(pi, pc)) {
                let partial = partials[&d.t[0]];
                let zero = b.emit(
                    DevOp::ZeroF32,
                    (0..model.n_cu).collect(),
                    &prior.get(&partial).copied().into_iter().collect::<Vec<_>>(),
                    |i| {
                        i.t[0] = partial;
                        i.i[..2].copy_from_slice(&d.i[..2]);
                    },
                );
                deps.push(zero);
                let compute = b.emit(DevOp::GemmSplitK, (0..model.n_cu).collect(), &deps, |i| {
                    i.t[..3].copy_from_slice(&[partial, d.t[1], d.t[2]]);
                    i.i[..3].copy_from_slice(&d.i[..3]);
                    i.i[3] = s;
                });
                let cast = b.emit(
                    DevOp::CastF32Bf16,
                    (0..model.n_cu).collect(),
                    &[compute],
                    |i| {
                        i.t[..2].copy_from_slice(&[d.t[0], partial]);
                        i.i[..2].copy_from_slice(&d.i[..2]);
                    },
                );
                prior.insert(partial, cast);
                cast
            } else {
                b.emit(
                    DevOp::from_u16(d.op).ok_or("unknown opcode")?,
                    placement[pc].clone(),
                    &deps,
                    |i| {
                        i.t = d.t;
                        i.i = d.i;
                        i.f = d.f;
                        i.j = d.j;
                    },
                )
            };
            map.push(last);
        }
        let result = b.finish();
        let packed: Vec<_> = result.insts.iter().map(|d| d.pack()).collect();
        for (pc, d) in old.insts.iter().enumerate() {
            if !selected.contains_key(&(pi, pc)) && d.pack() != packed[map[pc] as usize] {
                return Err("unchanged instruction changed".into());
            }
        }
        if pi + 1 == model.progs.len() {
            model.kv_row_insts = model
                .kv_row_insts
                .iter()
                .map(|&pc| {
                    map.get(pc as usize)
                        .copied()
                        .ok_or("KV patch index out of bounds".into())
                })
                .collect::<Result<_>>()?;
        }
        model.progs[pi] = result;
    }
    let proof = plow_asset::program::with_model(model, plow_asset::splitk::validate)?
        .ok_or("missing rewritten projection")?;
    for (index, canonical) in proof.canonical.iter().enumerate() {
        if canonical.dependencies != original[&(lo + index)] {
            return Err("canonical dependency graph changed".into());
        }
    }
    Ok(selected.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model() -> Model {
        let mut programs = Vec::new();
        let mut tensors = Vec::new();
        for rows in [1, 2, 4, 8, 16] {
            let mut b = Builder::new(2);
            b.force_uniseg();
            let a = b.tensor("act.a", 16 * 64 * 2);
            let w = b.tensor("model.layers.0.mlp.down_proj.weight", 128 * 64 * 2);
            let c = b.tensor("act.c", 16 * 128 * 2);
            b.emit(DevOp::Gemv, vec![0, 1], &[], |d| {
                d.t[..3].copy_from_slice(&[c, a, w]);
                d.i[..3].copy_from_slice(&[rows, 128, 64]);
                d.f[0] = 1e-6;
            });
            tensors = b.tensors();
            programs.push(b.finish());
        }
        Model {
            n_cu: 2,
            target: 0,
            tensors,
            progs: programs,
            prog_t: vec![1, 2, 4, 8, 16],
            gen: vec![],
            kv_row_insts: vec![],
        }
    }
    fn fixture() -> (DecodeObjects, ProjectionMeasurement) {
        let baseline = DecodeObject {
            file: "old.cubin".into(),
            sha256: "a".repeat(64),
            profile: "sm90a".into(),
            entry: "_Z12interp_sm90a11PlowProgram".into(),
            threads: 256,
            arena_bytes: 16448,
            grid: 2,
        };
        let mut candidate = baseline.clone();
        candidate.file = "s8.cubin".into();
        candidate.sha256 = "b".repeat(64);
        candidate.arena_bytes = 82944;
        let metadata = DecodeObjects {
            version: 1,
            kernarg_bytes: std::mem::size_of::<packet::dev::DevProgram>(),
            objects: BTreeMap::from([(0, baseline.clone())]),
            programs: [1, 2, 4, 8, 16]
                .into_iter()
                .enumerate()
                .map(
                    |(index, rows)| plow_asset::decode_objects::DecodeProgramObject {
                        index,
                        rows,
                        object: 0,
                    },
                )
                .collect(),
        };
        let stats = tunedb::projection::ProjectionTiming {
            median_ns: 68.,
            p10_ns: 67.,
            p90_ns: 69.,
            samples: 40,
        };
        let mut scalar = stats.clone();
        scalar.median_ns = 392.;
        scalar.p10_ns = 391.;
        scalar.p90_ns = 393.;
        let record = ProjectionMeasurement {
            cell: ProjectionCell {
                hardware: "test".into(),
                n_cu: 2,
                threads: 256,
                rows: 4,
                n: 128,
                k: 64,
            },
            split: 8,
            baseline_object: baseline,
            candidate_object: candidate,
            baseline_registers: 200,
            candidate_registers: 216,
            native_blocks: vec![tunedb::projection::NativeBlockGuard {
                context_tokens: 1024,
                packet_sha256: "c".repeat(64),
                stats: tunedb::projection::NativeBlockTiming {
                    median_ns: 800.,
                    p95_ns: 810.,
                    samples: 40,
                },
                baseline: tunedb::projection::NativeBlockTiming {
                    median_ns: 1100.,
                    p95_ns: 1110.,
                    samples: 40,
                },
            }],
            digests: tunedb::Digests {
                implementation: "body".into(),
                interpreter: "b".repeat(64),
                toolchain: "cuda".into(),
                oracle: PROJECTION_ORACLE.into(),
            },
            stats,
            baseline: scalar,
            correctness: tunedb::Correctness::Pass,
            state: tunedb::RecordState::Qualified,
            campaign: "block qualification".into(),
        };
        (metadata, record)
    }
    #[test]
    fn missing_stale_failed_records_preserve_binding_resources_and_packet_bytes() {
        let (metadata, r) = fixture();
        for case in 0..6 {
            let mut m = model();
            let before = m.to_blob();
            let mut records = vec![r.clone()];
            match case {
                0 => records.clear(),
                1 => records[0].digests.implementation = "stale".into(),
                2 => records[0].correctness = tunedb::Correctness::Unchecked,
                3 => records[0].state = tunedb::RecordState::Provisional,
                4 => records[0].native_blocks.clear(),
                5 => {
                    records[0].correctness = tunedb::Correctness::Fail {
                        detail: "numerical mismatch".into(),
                    }
                }
                _ => unreachable!(),
            }
            let (selected, bindings) = plan(&m, &metadata, &records, "test", &r.digests, |_| {
                panic!("fallback must not load candidate")
            })
            .unwrap();
            assert_eq!(rewrite(&mut m, &selected).unwrap(), 0);
            assert_eq!(before, m.to_blob());
            assert_eq!(bindings, metadata);
            assert_eq!(
                serde_json::to_vec(&bindings).unwrap(),
                serde_json::to_vec(&metadata).unwrap()
            );
        }
    }
    #[test]
    fn selected_rung_rebinds_only_after_qualification_and_conflicts_reject() {
        let (metadata, r) = fixture();
        let mut m = model();
        let (selected, bindings) = plan(
            &m,
            &metadata,
            std::slice::from_ref(&r),
            "test",
            &r.digests,
            |_| Ok((Some(1), 82944)),
        )
        .unwrap();
        assert_eq!(selected, BTreeMap::from([((2, 0), 8)]));
        assert_eq!(
            bindings.objects[&bindings.programs[0].object],
            r.baseline_object
        );
        assert_eq!(
            bindings.objects[&bindings.programs[2].object],
            r.candidate_object
        );
        assert!(plan(
            &m,
            &metadata,
            std::slice::from_ref(&r),
            "test",
            &r.digests,
            |_| Ok((None, 82944))
        )
        .is_err());
        let mut second = r.clone();
        second.cell.n = 64;
        second.candidate_object.file = "other.cubin".into();
        second.candidate_object.sha256 = "d".repeat(64);
        second.digests.interpreter = "d".repeat(64);
        let mut inst = m.progs[2].insts[0].clone();
        inst.i[1] = 64;
        m.progs[2].insts.push(inst);
        assert!(plan(
            &m,
            &metadata,
            &[r.clone(), second],
            "test",
            &r.digests,
            |_| Ok((Some(1), 82944))
        )
        .unwrap_err()
        .contains("conflicting objects"));
    }
    #[test]
    fn absent_selection_preserves_entire_blob() {
        let mut m = model();
        let before = m.to_blob();
        let config = crate::emit_config::EmitConfig::from_env();
        assert!(!config.decode_projection_tuning);
        assert_eq!(
            apply(
                &mut m,
                &config,
                "unknown",
                "unknown",
                Path::new("/missing/no-output")
            )
            .unwrap(),
            None
        );
        assert_eq!(rewrite(&mut m, &BTreeMap::new()).unwrap(), 0);
        assert_eq!(before, m.to_blob());
    }
    #[test]
    fn measured_rewrite_preserves_b1_b2_and_canonical_graph() {
        let mut m = model();
        let old: Vec<_> = m.progs.iter().map(|p| p.insts[0].pack()).collect();
        assert_eq!(
            rewrite(
                &mut m,
                &BTreeMap::from([((2, 0), 8), ((3, 0), 8), ((4, 0), 8)])
            )
            .unwrap(),
            3
        );
        assert_eq!(m.progs[0].insts[0].pack(), old[0]);
        assert_eq!(m.progs[1].insts[0].pack(), old[1]);
        assert_eq!(m.tensors.len(), 4);
        assert_eq!(m.tensors[3].bytes, 16 * 128 * 4 + 256);
        let proof = plow_asset::program::with_model(&m, plow_asset::splitk::validate)
            .unwrap()
            .unwrap();
        for (canonical, mut expected) in proof.canonical.iter().zip(old) {
            expected.fj[0] = 0;
            assert_eq!(canonical.instructions, [expected]);
        }
    }
    #[test]
    fn unsupported_selection_rejected() {
        for key in [(0, 0), (4, 1), (5, 0)] {
            assert!(rewrite(&mut model(), &BTreeMap::from([(key, 8)])).is_err());
        }
        assert!(rewrite(&mut model(), &BTreeMap::from([((4, 0), 3)])).is_err());
    }
}
