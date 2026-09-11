use packet::dev::{DevInst64, DevOp, StreamEnt};
use plow_asset::segment_roles::{self, SegmentRoles};
use plowrt::asset::devblob::DevBlob;
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

fn validate_kernel_cases(blob: &DevBlob, manifest: &serde_json::Value) -> Result<(), String> {
    let programs = manifest["kernel_cases"]["programs"]
        .as_array()
        .ok_or("build manifest has no all-op case inventory")?;
    if programs.len() != blob.progs.len() {
        return Err("build manifest program count differs from packet".into());
    }
    for (index, (p, declared)) in blob.progs.iter().zip(programs).enumerate() {
        let phase = if p.role.is_prefill_side() {
            "prefill"
        } else {
            "decode"
        };
        if declared["program"] != index
            || declared["kind"] != phase
            || declared["rows"] != packet::devbuild::program_rows(p.t)
            || declared["packed_only"] != p.role.is_packed_sibling()
            || declared["instruction_count"] != p.insts.len()
        {
            return Err(format!(
                "program {index}: build manifest rung differs from packet"
            ));
        }
        let mut seen = BTreeSet::new();
        for case in declared["cases"].as_array().ok_or("missing kernel cases")? {
            let f: [u32; 2] = serde_json::from_value(case["f_bits"].clone())
                .map_err(|_| "invalid case float bits")?;
            let j: [u32; 2] = serde_json::from_value(case["j"].clone())
                .map_err(|_| "invalid case integer bits")?;
            if f[1] != 0 && j[0] != 0 {
                return Err("case aliases float and integer wire fields".into());
            }
            let fj = [f[0], f[1] | j[0], j[1]];
            let pcs = case["pcs"].as_array().ok_or("missing case PCs")?;
            if pcs.is_empty() {
                return Err(format!("program {index}: empty kernel case"));
            }
            for pc in pcs {
                let pc = pc
                    .as_u64()
                    .and_then(|pc| usize::try_from(pc).ok())
                    .ok_or("invalid case PC")?;
                let inst = p.insts.get(pc).ok_or("case PC outside packet")?;
                let op = DevOp::from_u16(inst.op).ok_or("unknown packet opcode")?;
                if case["arm"] != devgen::manifest::arm_of(op, &inst.i).key() {
                    return Err(format!(
                        "program {index} PC {pc}: kernel dispatch arm differs from packet"
                    ));
                }
                if !seen.insert(pc) {
                    return Err(format!("program {index} PC {pc}: duplicate kernel case"));
                }
                let present = inst.t.map(|id| id != packet::dev::TENSOR_NONE16);
                let bytes = inst
                    .t
                    .map(|id| blob.tensors.get(id as usize).map(|t| t.bytes));
                if case["op"] != inst.op
                    || case["blocks"] != inst.blocks
                    || case["i"] != json!(inst.i)
                    || fj != inst.fj
                    || case["operand_present"] != json!(present)
                    || case["tensor_bytes"] != json!(bytes)
                {
                    return Err(format!(
                        "program {index} PC {pc}: kernel case differs from packet"
                    ));
                }
            }
        }
        if seen.len() != p.insts.len() {
            return Err(format!(
                "program {index}: incomplete all-op kernel case coverage"
            ));
        }
    }
    Ok(())
}

fn validate_queue_work(
    insts: &[DevInst64],
    stream: &[StreamEnt],
    queue: &[StreamEnt],
    l2_domains: u32,
) -> Result<(), String> {
    if l2_domains != 0 {
        use packet::dev::{SE_DOMAIN_MASK, SE_DOMAIN_SHIFT, SE_FINE, SE_NPER_MASK, SE_NPER_SHIFT};
        let mut counts = BTreeMap::<(u32, u16), u32>::new();
        for entry in queue {
            let domain = (entry.flags & SE_DOMAIN_MASK) >> SE_DOMAIN_SHIFT;
            if u32::from(domain) >= l2_domains {
                return Err("queue L2 domain out of range".into());
            }
            if entry.flags & SE_FINE == 0 {
                *counts.entry((entry.inst, domain)).or_default() += 1;
            }
        }
        for entry in queue {
            let domain = (entry.flags & SE_DOMAIN_MASK) >> SE_DOMAIN_SHIFT;
            let count = if entry.flags & SE_FINE == 0 {
                counts[&(entry.inst, domain)]
            } else {
                0
            };
            let expected = if count > 1 { count } else { 0 };
            let actual = u32::from((entry.flags & SE_NPER_MASK) >> SE_NPER_SHIFT);
            if actual != expected {
                return Err("queue L2 rendezvous count differs from work slices".into());
            }
        }
        if stream.iter().any(|entry| entry.flags & SE_NPER_MASK != 0) {
            return Err("static stream contains queue-only L2 rendezvous counts".into());
        }
    }
    let key = |e: &StreamEnt| {
        // The compiler derives the GQ-only count after assigning slices to XCDs.
        let flags = if l2_domains != 0 {
            e.flags & !packet::dev::SE_NPER_MASK
        } else {
            e.flags
        };
        (
            e.inst, e.slice, e.wait_ofs, e.succ_ofs, e.wait_len, e.succ_len, flags, e.seg,
        )
    };
    let mut scheduled: Vec<_> = stream.iter().map(key).collect();
    let mut queued: Vec<_> = queue.iter().map(key).collect();
    scheduled.sort_unstable();
    queued.sort_unstable();
    if scheduled != queued {
        return Err("queue differs from scheduled work or dependencies".into());
    }
    let mut slices = vec![BTreeSet::new(); insts.len()];
    for entry in queue {
        let inst = insts
            .get(entry.inst as usize)
            .ok_or("queue instruction out of range")?;
        if entry.slice >= u32::from(inst.blocks) {
            return Err(format!("PC {}: queue slice out of range", entry.inst));
        }
        if !slices[entry.inst as usize].insert(entry.slice) {
            return Err(format!("PC {}: duplicate queue slice", entry.inst));
        }
    }
    for (pc, inst) in insts.iter().enumerate() {
        if inst.blocks == 0 || slices[pc].len() != usize::from(inst.blocks) {
            return Err(format!("PC {pc}: incomplete queue slice coverage"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiler_cases_cover_every_rung_and_wire_field() {
        use packet::dev::DevInst;
        use packet::devbuild::{Model, Program, TensorDecl};
        let program = || Program {
            n_cu: 1,
            n_counter: 0,
            hier_base: 0,
            insts: vec![DevInst {
                op: DevOp::HeadNormRope as u16,
                blocks: 1,
                wait_len: 0,
                succ_len: 0,
                wait_ofs: 0,
                succ_ofs: 0,
                t: [0; 8],
                i: [128, 8, 256, 0, 0, 0, 0, 0],
                f: [1e-6, 0.0],
                j: [2048, 2047],
            }],
            stream: vec![StreamEnt::default()],
            stream_ofs: vec![0],
            stream_len: vec![1],
            waits: vec![],
            succs: vec![],
            tensors: vec![],
            gq_stream: vec![StreamEnt::default()],
            gq_seg_ofs: vec![0, 1],
            l2_sms: 0,
            l2_domains: 0,
        };
        let model = Model {
            n_cu: 1,
            target: 0,
            tensors: vec![TensorDecl {
                name: "act".into(),
                bytes: 4096,
                init: None,
            }],
            progs: vec![program(), program()],
            kv_row_insts: vec![],
            prog_t: vec![128, 1],
            gen: vec![],
        };
        let blob = DevBlob::parse(&model.to_blob()).unwrap();
        let manifest = devgen::manifest::build(
            &model,
            "sm_90a",
            &devgen::LeanReport::skipped("case audit test"),
        );
        validate_kernel_cases(&blob, &manifest).unwrap();
        for (pointer, value) in [
            ("/kernel_cases/programs/0/rows", json!(512)),
            ("/kernel_cases/programs/1/kind", json!("prefill")),
            ("/kernel_cases/programs/0/cases/0/pcs", json!([])),
            ("/kernel_cases/programs/0/cases/0/pcs", json!([0, 0])),
            ("/kernel_cases/programs/0/cases/0/pcs", json!([1])),
            ("/kernel_cases/programs/0/cases/0/i/2", json!(512)),
            (
                "/kernel_cases/programs/0/cases/0/arm",
                json!("HeadNormRope/hd512"),
            ),
            ("/kernel_cases/programs/0/cases/0/arm", json!(null)),
            ("/kernel_cases/programs/0/cases/0/j/0", json!(4096)),
            ("/kernel_cases/programs/0/cases/0/f_bits/0", json!(0)),
            ("/kernel_cases/programs/0/cases/0/f_bits/1", json!(1)),
            (
                "/kernel_cases/programs/0/cases/0/tensor_bytes/0",
                json!(8192),
            ),
            (
                "/kernel_cases/programs/0/cases/0/operand_present/0",
                json!(false),
            ),
            ("/kernel_cases/programs/1/cases", json!([])),
        ] {
            let mut changed = manifest.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            assert!(validate_kernel_cases(&blob, &changed).is_err(), "{pointer}");
        }
    }

    #[test]
    fn queue_requires_all_slices_and_preserves_dependencies() {
        let insts = [DevInst64 {
            blocks: 2,
            ..Default::default()
        }];
        let stream = [
            StreamEnt::default(),
            StreamEnt {
                slice: 1,
                ..Default::default()
            },
        ];
        assert!(validate_queue_work(&insts, &stream, &[stream[1], stream[0]], 0).is_ok());
        assert!(validate_queue_work(&insts, &stream[..1], &stream[..1], 0)
            .unwrap_err()
            .contains("incomplete"));
        let duplicate = [stream[0], stream[0]];
        assert!(validate_queue_work(&insts, &duplicate, &duplicate, 0)
            .unwrap_err()
            .contains("duplicate"));
        let mut changed = stream;
        changed[1].slice = 2;
        assert!(validate_queue_work(&insts, &changed, &changed, 0)
            .unwrap_err()
            .contains("out of range"));
        changed = stream;
        changed[1].wait_len = 1;
        assert!(validate_queue_work(&insts, &stream, &changed, 0)
            .unwrap_err()
            .contains("dependencies"));
    }

    #[test]
    fn l2_queue_counts_are_derived_without_ignoring_dependencies() {
        use packet::dev::{SE_DOMAIN_SHIFT, SE_FINE, SE_NPER_SHIFT, SE_XCTR};
        let insts = [DevInst64 {
            blocks: 4,
            ..Default::default()
        }];
        let stream = std::array::from_fn::<_, 4, _>(|slice| StreamEnt {
            slice: slice as u32,
            flags: if slice < 2 { 0 } else { 1 << SE_DOMAIN_SHIFT },
            ..Default::default()
        });
        let queue = stream.map(|mut e| {
            e.flags |= 2 << SE_NPER_SHIFT;
            e
        });
        assert!(validate_queue_work(&insts, &stream, &queue, 2).is_ok());
        let mut changed = queue;
        changed[0].flags ^= 1 << SE_NPER_SHIFT;
        assert!(validate_queue_work(&insts, &stream, &changed, 2)
            .unwrap_err()
            .contains("count"));
        changed = queue;
        changed[0].flags ^= SE_XCTR;
        assert!(validate_queue_work(&insts, &stream, &changed, 2)
            .unwrap_err()
            .contains("dependencies"));
        changed = queue;
        changed[0].flags |= SE_FINE;
        assert!(validate_queue_work(&insts, &stream, &changed, 2).is_err());
        assert!(validate_queue_work(&insts, &stream, &queue, 1)
            .unwrap_err()
            .contains("domain"));
        let fine = stream.map(|mut e| {
            e.flags |= SE_FINE;
            e
        });
        assert!(validate_queue_work(&insts, &fine, &fine, 2).is_ok());
        let single = [StreamEnt::default()];
        let insts = [DevInst64 {
            blocks: 1,
            ..Default::default()
        }];
        assert!(validate_queue_work(&insts, &single, &single, 2).is_ok());
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = PathBuf::from(args.next().ok_or("missing model.pkt")?);
    let output = PathBuf::from(args.next().ok_or("missing output.json")?);
    let build_manifest = args.next().map(PathBuf::from);
    if args.next().is_some() {
        return Err("usage: packet_ladder_audit model.pkt audit.json [build.json]".into());
    }
    let raw = std::fs::read(&path)?;
    let blob = DevBlob::parse(&raw)?;
    if let Some(path) = &build_manifest {
        let manifest = serde_json::from_slice(&std::fs::read(path)?)?;
        validate_kernel_cases(&blob, &manifest)?;
    }
    let roles = blob
        .reserved_metadata(&raw, segment_roles::SECTION)?
        .map(SegmentRoles::from_bytes)
        .transpose()?;
    if let Some(roles) = &roles {
        for program in &roles.programs {
            let p = blob
                .progs
                .get(program.index)
                .ok_or("role program out of range")?;
            if program.roles.len() + 1 != p.gq_seg_ofs.len() {
                return Err("role count does not match packet queue windows".into());
            }
        }
    }
    for (index, p) in blob.progs.iter().enumerate() {
        if p.gq_seg_ofs.first() != Some(&0)
            || p.gq_seg_ofs.last().copied().map(|n| n as usize) != Some(p.gq_stream.len())
            || p.gq_seg_ofs.windows(2).any(|w| w[0] > w[1])
        {
            return Err(format!("program {index}: invalid queue window bounds").into());
        }
        validate_queue_work(&p.insts, &p.stream, &p.gq_stream, p.l2_domains)
            .map_err(|e| format!("program {index}: {e}"))?;
        for (pc, inst) in p.insts.iter().enumerate() {
            if DevOp::from_u16(inst.op).is_none() {
                return Err(format!("program {index} PC {pc}: unknown opcode {}", inst.op).into());
            }
        }
    }
    let decode = blob.decode_rung_lo();
    let programs: Vec<_> = blob
        .progs
        .iter()
        .enumerate()
        .map(|(index, p)| {
            let program_roles = roles
                .as_ref()
                .and_then(|r| r.programs.iter().find(|program| program.index == index));
            let mut pc_windows = vec![BTreeSet::new(); p.insts.len()];
            for (window, bounds) in p.gq_seg_ofs.windows(2).enumerate() {
                for entry in &p.gq_stream[bounds[0] as usize..bounds[1] as usize] {
                    pc_windows[entry.inst as usize].insert(window);
                }
            }
            let mut counts = BTreeMap::<String, usize>::new();
            let mut cases = BTreeMap::new();
            let insts: Vec<_> = p
                .insts
                .iter()
                .enumerate()
                .map(|(pc, inst)| {
                    let op = DevOp::from_u16(inst.op)
                        .map(|op| format!("{op:?}"))
                        .unwrap_or_else(|| format!("unknown:{}", inst.op));
                    *counts.entry(op.clone()).or_default() += 1;
                    let tensors: Vec<_> = inst
                        .t
                        .iter()
                        .map(|&id| blob.tensors.get(id as usize).map(|t| t.name.as_str()))
                        .collect();
                    let tensor_bytes: Vec<_> = inst
                        .t
                        .iter()
                        .map(|&id| blob.tensors.get(id as usize).map(|t| t.bytes))
                        .collect();
                    let key = (
                        inst.op,
                        inst.blocks,
                        inst.i,
                        inst.fj,
                        inst.t
                            .map(|id| blob.tensors.get(id as usize).map(|t| t.bytes)),
                        pc_windows[pc]
                            .iter()
                            .map(|&window| program_roles.map(|program| program.roles[window]))
                            .collect::<BTreeSet<_>>(),
                    );
                    cases.entry(key).or_insert_with(Vec::new).push(pc);
                    json!({"pc": pc, "op": op, "blocks": inst.blocks,
                   "i": inst.i, "fj_bits": inst.fj, "t": inst.t,
                   "tensors": tensors, "tensor_bytes": tensor_bytes,
                   "queue_windows": pc_windows[pc]})
                })
                .collect();
            let segments: Vec<_> = p
                .gq_seg_ofs
                .windows(2)
                .enumerate()
                .map(|(window, bounds)| {
                    let entries = &p.gq_stream[bounds[0] as usize..bounds[1] as usize];
                    let pcs: BTreeSet<_> = entries.iter().map(|entry| entry.inst).collect();
                    let role = program_roles.map(|program| program.roles[window]);
                    let object = role.and_then(|id| roles.as_ref()?.objects.get(&id));
                    json!({"window": window, "pcs": pcs, "entries": entries.len(),
                    "segment": entries.first().map(|entry| entry.seg),
                    "declared_role": role, "declared_object": object,
                    "route": match role {
                        None => "runtime_selected",
                        Some(segment_roles::INTERPRETER) => "interpreter",
                        Some(segment_roles::CUBLASLT) => "cublaslt",
                        Some(_) => "native_object",
                    }})
                })
                .collect();
            let cases: Vec<_> = cases
                .into_iter()
                .map(|((op, blocks, i, fj, tensor_bytes, roles), pcs)| {
                    json!({"op": format!("{:?}", DevOp::from_u16(op).unwrap()),
                    "dispatch_arm": devgen::manifest::arm_of(DevOp::from_u16(op).unwrap(), &i).key(),
                    "blocks": blocks, "i": i, "fj_bits": fj,
                    "tensor_bytes": tensor_bytes, "declared_roles": roles, "pcs": pcs,
                    "performance_evidence": null})
                })
                .collect();
            json!({"index": index, "phase": if index < decode {"prefill"} else {"decode"},
               "rows": packet::devbuild::program_rows(p.t),
               "packed_only": p.role.is_packed_sibling(), "op_counts": counts,
               "counter_count": p.n_counter, "wait_count": p.waits.len(),
               "segment_offsets": p.gq_seg_ofs, "segments": segments, "instructions": insts,
               "kernel_cases": cases})
        })
        .collect();
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&json!({
            "packet": path,
            "packet_sha256": plow_asset::decode_objects::image_sha256(&raw),
            "checked_build_manifest": build_manifest,
            "target_fingerprint": blob.target,
            "n_cu": blob.n_cu, "programs": programs,
            "note": "Emitted instructions and declared packet roles only. A missing role means runtime-selected routing; an interpreter role can contain specialized or fused bodies. Object loading, runtime overrides, correctness and performance require separate verification."
        }))?,
    )?;
    Ok(())
}
