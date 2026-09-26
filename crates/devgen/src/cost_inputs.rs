use packet::dev::{DevInst, DevOp};
use packet::devbuild::{Model, Program};
use serde_json::{json, Value};

fn matrix_work(d: &DevInst) -> Option<Value> {
    let op = DevOp::from_u16(d.op)?;
    let fp8_input = matches!(
        op,
        DevOp::GemmFp8 | DevOp::GemmGluFp8 | DevOp::GemmFp8Block128 | DevOp::GemmFp8Block128Split4
    );
    let (m, n, k, heads, streams, weight_bits, scale_bytes, output_bytes) = if matches!(
        op,
        DevOp::GemmFp8Block128
            | DevOp::GemmFp8Block128Split4
            | DevOp::GemmFp8Blk
            | DevOp::DenseGluFp8Blk
    ) {
        let [m, n, k, _, _, _, _, _] = d.i.map(u64::from);
        let streams = if op == DevOp::DenseGluFp8Blk { 2 } else { 1 };
        (
            m,
            n,
            k,
            1,
            streams,
            8,
            streams * n.div_ceil(128) * k.div_ceil(128) * 4,
            2,
        )
    } else if matches!(op, DevOp::GemmFp8 | DevOp::GemmGluFp8) {
        let [m, n, k, _, _, _, _, _] = d.i.map(u64::from);
        let streams = if op == DevOp::GemmGluFp8 { 2 } else { 1 };
        (m, n, k, 1, streams, 8, streams * n * 4, 2)
    } else if let Some((family, m, n, k, quant)) = op.gemv_case(&d.i) {
        let streams = if family == "gemvglu" { 2u64 } else { 1 };
        let (n, k) = (u64::from(n), u64::from(k));
        let (bits, scales) = match quant {
            "None" => (16u64, 0),
            "W8A8" if op == DevOp::GemvFp8Blk => (8, n.div_ceil(128) * k.div_ceil(128) * 4),
            "W8A8" => (8, n * 4),
            "Mxfp4" => (4, n * k.div_ceil(32)),
            _ => return None,
        };
        // Argmax has a different output contract; it is not a dense matrix write.
        if op == DevOp::GemvArgmax {
            return None;
        }
        (
            u64::from(m),
            n,
            k,
            1u64,
            streams,
            bits,
            scales * streams,
            if op == DevOp::GemvF32 { 4u64 } else { 2 },
        )
    } else if op == DevOp::MlaBmmFp8 {
        let [m, h, n, k, _, _, _, _] = d.i.map(u64::from);
        // The native body applies one scalar weight scale after the
        // activation-group-scaled accumulation, not a weight block grid.
        (m, n, k, h, 1, 8, 4, 2)
    } else if let Some((_, _, half_bytes)) = crate::gemm_tile_of(op, d.i[7]) {
        // Tile-selected BF16 matmuls have no scale or activation-quantization ambiguity.
        if half_bytes != 4 {
            return None;
        }
        (
            u64::from(d.i[0]),
            u64::from(d.i[1]),
            u64::from(d.i[2]),
            1,
            1,
            16,
            0,
            2,
        )
    } else {
        return None;
    };
    let product = |xs: &[u64]| xs.iter().try_fold(1u64, |a, b| a.checked_mul(*b));
    let flops = product(&[2, m, n, k, heads, streams])?;
    let weight_bytes = product(&[n, k, heads, streams, weight_bits])?.div_ceil(8);
    let input_bytes = product(&[m, k, heads, if fp8_input { 1 } else { 2 }])?;
    let output_bytes = product(&[m, n, heads, output_bytes])?;
    let activation_scale_bytes =
        if matches!(op, DevOp::GemmFp8Block128 | DevOp::GemmFp8Block128Split4) {
            product(&[m, k.div_ceil(128), 4])?
        } else if fp8_input {
            product(&[m, 4])?
        } else {
            0
        };
    let partial_output_bytes = if op == DevOp::GemmFp8Block128Split4 {
        output_bytes.checked_mul(4)?
    } else {
        0
    };
    Some(json!({
        "m": m, "n": n, "k": k, "heads": heads, "weight_streams": streams,
        "weight_bits": weight_bits, "dot_flops": flops,
        "logical_weight_bytes": weight_bytes, "logical_scale_bytes": scale_bytes,
        "logical_activation_scale_bytes": activation_scale_bytes,
        "logical_input_bytes": input_bytes, "logical_output_bytes": output_bytes,
        "logical_direct_output_bytes": if partial_output_bytes == 0 { output_bytes } else { 0 },
        "logical_split_partial_output_bytes": partial_output_bytes,
        "assumptions": ["one logical operand visit; not physical HBM traffic",
            "matrix dot only; activation, quantization and fused epilogue arithmetic excluded",
            "output bytes are semantic result capacity; split partial stores and direct stores are separate",
            "compiled rows; inactive-row work and selected object padding require runtime profile"],
    }))
}

pub fn program(m: &Model, p: &Program) -> Value {
    let mut modeled = Vec::new();
    let mut attention = Vec::new();
    let mut collectives = Vec::new();
    let mut unmodeled = Vec::new();
    let mut tensors = std::collections::BTreeSet::new();
    for (index, d) in p.insts.iter().enumerate() {
        tensors.extend(
            d.t.iter()
                .copied()
                .filter(|&t| t != packet::dev::TENSOR_NONE),
        );
        let op = DevOp::from_u16(d.op).map(|op| format!("{op:?}"));
        if let Some(mut work) = attention_work(d) {
            work["instruction"] = json!(index);
            work["op"] = json!(op);
            attention.push(work);
        }
        if let Some(mut work) = collective_work(d) {
            work["instruction"] = json!(index);
            work["op"] = json!(op);
            collectives.push(work);
        }
        if let Some(mut work) = matrix_work(d) {
            work["instruction"] = json!(index);
            work["op"] = json!(op);
            work["workgroups"] = json!(d.blocks);
            modeled.push(work);
        } else {
            unmodeled.push(json!({"instruction": index, "op": op}));
        }
    }
    let allocations: Vec<_> = tensors
        .into_iter()
        .filter_map(|id| {
            let tensor = m.tensors.get(id as usize)?;
            Some(json!({"tensor": id, "name": tensor.name, "capacity_bytes": tensor.bytes}))
        })
        .collect();
    json!({
        "schema": 1,
        "scope": "selected packet instructions; TP-local shapes; no generic model preset",
        "matrix_work": modeled,
        "attention_work": attention,
        "collective_work": collectives,
        "coarse_dependency_witness": p.reduction_witness.as_ref().map(|w| json!({
            "scope": "pre-TR coarse graph; excludes fine dependencies and memory-effect completeness",
            "original": w.original, "retained": w.retained, "paths": w.paths,
        })),
        "unmodeled_instructions": unmodeled,
        "allocation_capacities": allocations,
        "physical_hbm_bytes": null,
        "measured_durations_ns": null,
        "collective_wire_bytes": null,
        "spill_hbm_bytes": null,
        "performance_qualified": false,
        "missing_inputs": ["sparse per-row selected counts and cache dtype",
            "selected object/launch resource envelope and spill traffic",
            "collective algorithm/topology and wire transfers", "measured instruction/segment durations",
            "explicit physical cache/HBM reuse assumptions"],
    })
}

fn attention_work(d: &DevInst) -> Option<Value> {
    let op = DevOp::from_u16(d.op)?;
    if !matches!(
        op,
        DevOp::FlashGatherDecode | DevOp::FlashMlaDecode | DevOp::FlashMlaDecodeFp8
    ) {
        return None;
    }
    let fp8 = op == DevOp::FlashMlaDecodeFp8;
    let sparse = op == DevOp::FlashGatherDecode || fp8 && d.j[0] != 0;
    let rope_fp8 = fp8 && !sparse && d.i[6] != 0;
    let latent_bytes = if fp8 { 512u64 } else { 1024 };
    let rope_bytes = if rope_fp8 { 64u64 } else { 128 };
    let scale_bytes = if fp8 {
        if rope_fp8 {
            8u64
        } else {
            4
        }
    } else {
        0
    };
    Some(json!({
        "rows": d.i[0], "heads": d.i[1], "context_capacity": d.i[2],
        "selected_limit": if sparse { d.i[6].min(d.i[2]) } else { d.i[2] },
        "selected_positions": null,
        "selected_positions_formula": if sparse { "sum(min(row_live, top_k))" } else { "sum(row_live_after_window_mask)" },
        "logical_cache_bytes_per_selected_position": latent_bytes + rope_bytes,
        "logical_scale_bytes_per_selected_position": scale_bytes,
        "logical_index_bytes_per_selected_position": if sparse { 4 } else { 0 },
        "dot_flops_per_selected_position": 2u64 * u64::from(d.i[1]) * (512 + 64 + 512),
        "assumptions": ["semantic heads, not padded physical heads",
            "cache vectors counted once per selected row; physical head/split rereads unspecified",
            "softmax, metadata, pack, merge and workspace traffic excluded"],
    }))
}

fn collective_work(d: &DevInst) -> Option<Value> {
    let op = DevOp::from_u16(d.op)?;
    let (elements, ranks, passes) = match op {
        DevOp::XReduce if d.i[5] == 0 => (u64::from(d.i[0]), d.i[1], "one_shot"),
        DevOp::XReduceTwoShot if d.t[1] == packet::dev::TENSOR_NONE && d.i[7] == 0 => {
            (u64::from(d.i[0]), d.i[1], "two_shot")
        }
        DevOp::XAllGather => (
            u64::from(d.i[0]) + u64::from(d.i[1]) + u64::from(d.i[2]),
            d.i[4],
            "all_gather",
        ),
        _ => return None,
    };
    if ranks == 0 {
        return None;
    }
    Some(json!({
        "elements": elements, "element_bytes": 2, "ranks": ranks,
        "algorithm": passes, "logical_result_bytes": elements * 2,
        "logical_remote_payload_per_rank": match passes {
            "one_shot" => json!(elements * 2 * u64::from(ranks - 1)),
            _ if elements % u64::from(ranks) == 0 => json!(elements / u64::from(ranks) * 2 * u64::from(ranks - 1)
                * if passes == "two_shot" { 2 } else { 1 }),
            _ => Value::Null,
        },
        "assumptions": ["logical peer payload only; excludes protocol, contention, retries and topology routing"],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(op: DevOp, i: [u32; 8]) -> DevInst {
        DevInst {
            op: op as u16,
            i,
            ..Default::default()
        }
    }

    #[test]
    fn selected_gemv_counts_fp8_block_scales_and_all_qkvg_streams() {
        let fp8 = matrix_work(&inst(DevOp::GemvFp8Blk, [16, 256, 512, 0, 0, 0, 0, 0])).unwrap();
        assert_eq!(fp8["dot_flops"], 2 * 16 * 256 * 512);
        assert_eq!(fp8["logical_weight_bytes"], 256 * 512);
        assert_eq!(fp8["logical_scale_bytes"], 2 * 4 * 4);
        let qkvg = matrix_work(&inst(DevOp::GemvQkvg, [1, 128, 512, 64, 32, 16, 0, 0])).unwrap();
        assert_eq!(qkvg["n"], 240);
        assert_eq!(qkvg["logical_weight_bytes"], 240 * 512 * 2);
        let glu = matrix_work(&inst(DevOp::GemvGluFp8, [16, 256, 512, 0, 0, 0, 0, 0])).unwrap();
        assert_eq!(glu["dot_flops"], 4 * 16 * 256 * 512);
        assert_eq!(glu["logical_output_bytes"], 16 * 256 * 2);
    }

    #[test]
    fn stride_wv_preserves_logical_work_but_not_allocation_capacity() {
        let mut d = inst(DevOp::MlaBmmFp8, [16, 8, 256, 512, 0, 0, 0, 0]);
        let contiguous = matrix_work(&d).unwrap();
        d.i[5] = 1024;
        assert_eq!(matrix_work(&d).unwrap(), contiguous);
        assert_eq!(contiguous["logical_scale_bytes"], 4);
        assert!(matrix_work(&inst(
            DevOp::FlashGatherDecode,
            [16, 8, 8192, 0, 8, 0, 2048, 4]
        ))
        .is_none());
    }

    #[test]
    fn native_fp8_gemm_distinguishes_activation_scales_and_split_partials() {
        let normal =
            matrix_work(&inst(DevOp::GemmFp8Block128, [16, 256, 512, 0, 0, 0, 0, 0])).unwrap();
        let split = matrix_work(&inst(
            DevOp::GemmFp8Block128Split4,
            [16, 256, 512, 0, 0, 0, 0, 0],
        ))
        .unwrap();
        assert_eq!(normal["logical_input_bytes"], 16 * 512);
        assert_eq!(normal["logical_activation_scale_bytes"], 16 * 4 * 4);
        assert_eq!(normal["logical_scale_bytes"], 2 * 4 * 4);
        assert_eq!(normal["dot_flops"], split["dot_flops"]);
        assert_eq!(
            split["logical_split_partial_output_bytes"],
            4 * 16 * 256 * 2
        );
        let legacy =
            matrix_work(&inst(DevOp::DenseGluFp8Blk, [16, 256, 512, 0, 0, 0, 0, 0])).unwrap();
        assert_eq!(legacy["logical_input_bytes"], 16 * 512 * 2);
        assert_eq!(legacy["logical_weight_bytes"], 2 * 256 * 512);
    }

    #[test]
    fn sparse_attention_keeps_cache_precision_and_dynamic_selection_separate() {
        let mut d = inst(DevOp::FlashGatherDecode, [16, 8, 71680, 0, 8, 0, 2048, 4]);
        let bf16 = attention_work(&d).unwrap();
        assert_eq!(bf16["selected_limit"], 2048);
        assert_eq!(bf16["logical_cache_bytes_per_selected_position"], 1152);
        assert!(bf16["selected_positions"].is_null());
        d.op = DevOp::FlashMlaDecodeFp8 as u16;
        d.j[0] = 7;
        let fp8 = attention_work(&d).unwrap();
        assert_eq!(fp8["logical_cache_bytes_per_selected_position"], 640);
        assert_eq!(fp8["logical_scale_bytes_per_selected_position"], 4);
        assert_eq!(
            fp8["dot_flops_per_selected_position"],
            bf16["dot_flops_per_selected_position"]
        );
    }

    #[test]
    fn collective_accounting_is_per_rank_not_full_cluster_hbm() {
        let one = collective_work(&inst(DevOp::XReduce, [7168, 8, 0, 0, 0, 0, 0, 0])).unwrap();
        assert_eq!(one["logical_remote_payload_per_rank"], 7168 * 2 * 7);
        let mut d = inst(DevOp::XReduceTwoShot, [7168, 8, 0, 0, 0, 0, 0, 0]);
        d.t[1] = packet::dev::TENSOR_NONE;
        let two = collective_work(&d).unwrap();
        assert_eq!(two["logical_remote_payload_per_rank"], 7168 / 8 * 2 * 7 * 2);
    }
}
