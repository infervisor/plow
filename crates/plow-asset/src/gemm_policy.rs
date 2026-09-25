//! Final-wire bindings for ordinary BF16 GEMM selection decisions. These do not
//! authenticate measured costs, prove kernel arithmetic or cover runtime rewrites.

use packet::dev::{DevOp, TENSOR_NONE16};
use serde_json::{json, Value};

use crate::program::Packet;

pub const KIND: &str = "dense_gemm_selection_v1";

pub fn binding(packet: &Packet<'_>, program: usize, request: &Value) -> Result<Value, String> {
    let bad = || "unsupported or inconsistent final-wire GEMM policy".to_string();
    if request["policy_kind"] != KIND || request["geometry"]["quant"] != "None" {
        return Err(bad());
    }
    let geometry = request.get("geometry").ok_or_else(bad)?;
    let domain = crate::decode_objects::image_sha256(
        &serde_json::to_vec(geometry).map_err(|e| e.to_string())?,
    );
    let selected = request["selected_opcode"]
        .as_u64()
        .and_then(|n| u16::try_from(n).ok())
        .ok_or_else(bad)?;
    if !matches!(
        DevOp::from_u16(selected),
        Some(DevOp::Gemm | DevOp::GemmMed | DevOp::GemmSmall | DevOp::GemmWide | DevOp::GemmC5)
    ) || request["required"] != json!([domain])
        || request["choices"]
            .as_array()
            .is_none_or(|choices| choices.len() != 1)
        || request["choices"][0]["domain"] != domain
        || !request["choices"][0]["key"]
            .as_str()
            .is_some_and(|key| key.starts_with(&format!("{selected}:")))
        || geometry["n_cu"].as_u64() != Some(packet.n_cu.into())
    {
        return Err(bad());
    }
    let shape: Vec<_> = ["m", "n", "k"]
        .iter()
        .map(|name| {
            geometry[name]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .filter(|n| *n != 0)
                .ok_or_else(bad)
        })
        .collect::<Result<_, _>>()?;
    let p = packet.programs.get(program).ok_or_else(bad)?;
    if !matches!(p.role, packet::devbuild::ProgramRole::PrefillBucket { .. })
        || p.l2_domains != 0
        || packet.n_cu == 0
        || p.stream_ofs.len() != packet.n_cu as usize
        || p.stream_len.len() != packet.n_cu as usize
    {
        return Err(bad());
    }
    let mut sites = Vec::new();
    for (index, d) in p.insts.iter().enumerate() {
        if d.op != selected
            || d.i[..3] != shape
            || d.i[3..] != [0; 5]
            || d.t[..3].contains(&TENSOR_NONE16)
            || d.t[3..].iter().any(|h| *h != TENSOR_NONE16)
            || d.blocks == 0
        {
            continue;
        }
        let mut placements = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        let mut counts = Vec::new();
        for cu in 0..packet.n_cu as usize {
            let start = p.stream_ofs[cu] as usize;
            let entries = start
                .checked_add(p.stream_len[cu] as usize)
                .and_then(|end| p.stream.get(start..end))
                .ok_or_else(bad)?;
            let mut count = 0;
            for e in entries.iter().filter(|e| e.inst as usize == index) {
                if e.flags != 0 || e.slice >= d.blocks.into() || !seen.insert(e.slice) {
                    return Err(bad());
                }
                placements.push(json!({"cu":cu,"slice":e.slice}));
                count += 1;
            }
            counts.push(count);
        }
        // A reduced-CU or uneven static placement cannot inherit the full-CU cost cell.
        let blocks = u32::from(d.blocks);
        if seen.len() != blocks as usize
            || counts.iter().enumerate().any(|(cu, count)| {
                *count != blocks / packet.n_cu + u32::from((cu as u32) < blocks % packet.n_cu)
            })
        {
            continue;
        }
        let mut gq = Vec::new();
        let mut gq_slices = std::collections::BTreeSet::new();
        for e in p.gq_stream.iter().filter(|e| e.inst as usize == index) {
            if e.flags != 0 || e.slice >= blocks || !gq_slices.insert(e.slice) {
                return Err(bad());
            }
            gq.push(json!({"slice":e.slice,"segment":e.seg}));
        }
        if gq_slices != seen {
            return Err("GEMM selected route is absent or incomplete in the final GQ".into());
        }
        let operands = d.t[..3]
            .iter()
            .map(|&id| {
                let tensor = packet.tensors.get(id as usize).ok_or_else(bad)?;
                Ok(json!({"id":id,"name":tensor.name,"capacity_bytes":tensor.bytes}))
            })
            .collect::<Result<Vec<_>, String>>()?;
        sites.push(json!({"instruction":index,"opcode":d.op,"blocks":d.blocks,
            "immediates":d.i,"float_bits":d.fj,"operands":operands,"placement":placements,"gq":gq}));
    }
    if sites.is_empty() {
        return Err("no audited final-wire GEMM matches this selection query".into());
    }
    Ok(
        json!({"schema":1,"scope":"ordinary BF16 GEMM final-wire selection; no runtime rewrite or FP implementation qualification",
        "sites":sites}),
    )
}

pub fn bind(packet: &Packet<'_>, program: usize, request: &Value) -> Result<Value, String> {
    let mut bound = request.clone();
    bound["wire_binding"] = binding(packet, program, request)?;
    Ok(bound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::devbuild::{Builder, Model};

    fn fixture() -> (Model, Value) {
        let mut b = Builder::new(2);
        let out = b.tensor("output", 128 * 256 * 2);
        let input = b.tensor("input", 128 * 512 * 2);
        let weight = b.tensor("weight", 256 * 512 * 2);
        b.emit(DevOp::GemmMed, vec![0, 1], &[], |d| {
            d.t[..3].copy_from_slice(&[out, input, weight]);
            d.i[..3].copy_from_slice(&[128, 256, 512]);
        });
        let p = b.finish();
        let mut decode = Builder::new(2);
        decode.emit(DevOp::Nop, vec![0], &[], |_| {});
        let model = Model {
            n_cu: 2,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p, decode.finish()],
            prog_t: vec![128, 1],
            gen: vec![],
            kv_row_insts: vec![],
        };
        let geometry = json!({"m":128,"n":256,"k":512,"n_cu":2,"quant":"None"});
        let domain = crate::decode_objects::image_sha256(&serde_json::to_vec(&geometry).unwrap());
        let key = format!("{}:implementation", DevOp::GemmMed as u16);
        let request = json!({"policy_kind":KIND,"geometry":geometry,"selected_opcode":DevOp::GemmMed as u16,
            "required":[domain],"choices":[{"domain":domain,"key":key}],
            "candidates":[{"domain":domain,"key":key,"cost":1,"qualified":true}]});
        (model, request)
    }

    #[test]
    fn emitted_gemm_binding_excludes_fusion_tags_and_changed_policy_domains() {
        let (model, request) = fixture();
        let snapshot = crate::program::with_model(&model, |p| binding(p, 0, &request).unwrap());
        assert_eq!(
            snapshot["sites"][0]["placement"].as_array().unwrap().len(),
            2
        );
        for change in 0..13 {
            let (mut model, mut request) = fixture();
            match change {
                0 => model.progs[0].insts[0].op = DevOp::GemmSmall as u16,
                1 => model.progs[0].insts[0].i[1] += 1,
                2 => model.progs[0].insts[0].i[7] = packet::dev::GEMM_WIDE_C8_TAG,
                3 => model.progs[0].insts[0].t[7] = 0,
                4 => model.progs[0].insts[0].t[3] = 0,
                5 => request["selected_opcode"] = json!(DevOp::Gemm as u16),
                6 => request["required"] = json!(["other-domain"]),
                7 => request["geometry"]["n_cu"] = json!(1),
                8 => request["geometry"]["quant"] = json!("W8A8"),
                9 => model.progs[0].l2_domains = 1,
                10 => {
                    model.progs[0].gq_stream.pop();
                }
                11 => model.progs[0].stream[1].slice = 0,
                _ => {
                    model.progs[0].stream_len = vec![2, 0];
                    model.progs[0].stream_ofs = vec![0, 2];
                }
            }
            assert!(
                crate::program::with_model(&model, |p| binding(p, 0, &request)).is_err(),
                "mutation {change}"
            );
        }
        let mut model = model;
        model.progs[0].insts[0].f[0] = 0.125;
        assert_ne!(
            snapshot,
            crate::program::with_model(&model, |p| binding(p, 0, &request).unwrap())
        );
        model.tensors[0].bytes += 2;
        assert_ne!(
            snapshot,
            crate::program::with_model(&model, |p| binding(p, 0, &request).unwrap())
        );
    }
}
