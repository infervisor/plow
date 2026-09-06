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
    let baseline_packet_sha256 = plow_asset::decode_objects::image_sha256(&model.to_blob());
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
                    &baseline_packet_sha256,
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
                    &baseline_packet_sha256,
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
#[path = "projection_rewrite_tests.rs"]
mod tests;
