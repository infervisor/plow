use packet::dev::DevOp;
use plow_asset::segment_roles::{self, SegmentRoles};
use plowrt::asset::devblob::DevBlob;
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = PathBuf::from(args.next().ok_or("missing model.pkt")?);
    let output = PathBuf::from(args.next().ok_or("missing output.json")?);
    let raw = std::fs::read(&path)?;
    let blob = DevBlob::parse(&raw)?;
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
            json!({"index": index, "phase": if index < decode {"prefill"} else {"decode"},
               "rows": packet::devbuild::program_rows(p.t),
               "packed_only": p.packed_prefill_only, "op_counts": counts,
               "counter_count": p.n_counter, "wait_count": p.waits.len(),
               "segment_offsets": p.gq_seg_ofs, "segments": segments, "instructions": insts})
        })
        .collect();
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&json!({
            "packet": path, "n_cu": blob.n_cu, "programs": programs,
            "note": "Emitted instructions and declared packet roles only. A missing role means runtime-selected routing; an interpreter role can contain specialized or fused bodies. Object loading, runtime overrides, correctness and performance require separate verification."
        }))?,
    )?;
    Ok(())
}
