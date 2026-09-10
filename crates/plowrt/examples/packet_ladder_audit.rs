use packet::dev::DevOp;
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
    let blob = DevBlob::parse(&std::fs::read(&path)?)?;
    let decode = blob.decode_rung_lo();
    let programs: Vec<_> = blob
        .progs
        .iter()
        .enumerate()
        .map(|(index, p)| {
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
                    json!({"pc": pc, "op": op, "blocks": inst.blocks,
                   "i": inst.i, "t": inst.t, "tensors": tensors})
                })
                .collect();
            let segments: Vec<_> = p
                .gq_seg_ofs
                .windows(2)
                .enumerate()
                .map(|(window, bounds)| {
                    let entries = &p.gq_stream[bounds[0] as usize..bounds[1] as usize];
                    let pcs: BTreeSet<_> = entries.iter().map(|entry| entry.inst).collect();
                    json!({"window": window, "pcs": pcs, "entries": entries.len(),
                   "segment": entries.first().map(|entry| entry.seg)})
                })
                .collect();
            json!({"index": index, "phase": if index < decode {"prefill"} else {"decode"},
               "rows": packet::devbuild::program_rows(p.t),
               "packed_only": p.packed_prefill_only, "op_counts": counts,
               "segment_offsets": p.gq_seg_ofs, "segments": segments, "instructions": insts})
        })
        .collect();
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&json!({
            "packet": path, "n_cu": blob.n_cu, "programs": programs,
            "note": "Emitted instructions only; runtime object and library overrides require separate verification."
        }))?,
    )?;
    Ok(())
}
