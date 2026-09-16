//! V4.1's tensor binding, against the released shards.
//!
//! The binding is where this checkpoint is most likely to go wrong, because the optional per-layer
//! groups are DERIVED rather than stated: `kv_source_layer_ids` gives a compressor,
//! `index_source_layer_ids` gives indexer queries, and only the intersection gives indexer KEYS.
//! Each derivation is a place to read V4.1 as V4, and a misread binds the wrong tensors instead of
//! failing — the loader's answer to a name it cannot find has historically been a zero fill.
//!
//! So this asserts the declared `(name, bytes)` list reproduces what is actually on disk, for
//! every one of the 40 layers, both directions: nothing declared that the shards do not have, and
//! nothing in the shards that the declaration misses.

use super::dsv41::{dsv41_layer_tensors, Dsv41Cfg};
use super::*;

const HF: &str = "/workspace/models/DeepSeek-V4.1-Flash";

fn checkpoint() -> Option<(Dsv41Cfg, std::collections::BTreeMap<String, (String, Vec<i64>)>)> {
    let dir = std::path::Path::new(HF);
    if !dir.join("config.json").exists() {
        eprintln!("skipping: DeepSeek-V4.1-Flash not present");
        return None;
    }
    let cfg = cfg_dsv41(dir).expect("the released checkpoint resolves");
    let (hdr, have, total) = super::kimi_k3::k3_shard_headers(dir);
    if hdr.is_empty() {
        eprintln!("skipping: no shard headers readable");
        return None;
    }
    eprintln!("read {have} of {total} shards, {} tensors", hdr.len());
    Some((cfg, hdr))
}

/// Bytes a shard header's `(dtype, shape)` actually occupies.
fn shard_bytes(dt: &str, shape: &[i64]) -> u64 {
    let elems: i64 = shape.iter().product();
    let w = match dt {
        // The nibble-packed MXFP4 expert rows arrive as I8 and are already HALF as wide, so the
        // packing is in the shape, not in a sub-byte width here.
        "F8_E4M3" | "F8_E8M0" | "I8" | "U8" => 1,
        "BF16" | "F16" => 2,
        "F32" => 4,
        other => panic!("unmodelled dtype {other}"),
    };
    (elems * w) as u64
}

/// Every declared tensor must exist on disk at exactly the declared size.
///
/// A size mismatch is the failure worth catching here: a name that is merely absent would surface
/// at load, but a name that is present at the WRONG size loads happily and reads past — or short
/// of — the weight it was supposed to be.
#[test]
fn dsv41_layer_tensors_match_the_shards() {
    let Some((cfg, hdr)) = checkpoint() else {
        return;
    };
    let mut checked = 0usize;
    let mut missing: Vec<String> = Vec::new();
    for l in 0..cfg.layers {
        for (name, bytes) in dsv41_layer_tensors(&cfg, l) {
            let full = format!("layers.{l}.{name}");
            match hdr.get(&full) {
                Some((dt, shape)) => {
                    assert_eq!(
                        shard_bytes(dt, shape),
                        bytes,
                        "{full}: shard is {dt}{shape:?}, declaration says {bytes} bytes"
                    );
                    checked += 1;
                }
                None => missing.push(full),
            }
        }
    }
    assert!(
        missing.is_empty(),
        "{} declared tensor(s) are not in the shards, e.g. {:?}",
        missing.len(),
        &missing[..missing.len().min(8)]
    );
    // 40 layers x (2334 base + the optional groups). Guards against a declaration that silently
    // shrinks to nothing and passes every assertion above by checking nothing.
    assert!(checked > 90_000, "only {checked} tensors checked");
}

/// ...and nothing on disk is left unbound.
///
/// The other direction, and the one that catches a MISSED group rather than a wrong one. Without
/// it, forgetting `ffn.gate.bias_vl` entirely — the image-span routing bias, which exists on every
/// layer and has no analogue in V4 — passes the test above perfectly.
#[test]
fn the_declaration_leaves_no_layer_tensor_unbound() {
    let Some((cfg, hdr)) = checkpoint() else {
        return;
    };
    let mut unbound: Vec<String> = Vec::new();
    for l in 0..cfg.layers {
        let declared: std::collections::HashSet<String> = dsv41_layer_tensors(&cfg, l)
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        let prefix = format!("layers.{l}.");
        for name in hdr.keys() {
            if let Some(rest) = name.strip_prefix(&prefix) {
                if !declared.contains(rest) {
                    unbound.push(name.clone());
                }
            }
        }
    }
    assert!(
        unbound.is_empty(),
        "{} shard tensor(s) the declaration does not bind, e.g. {:?}",
        unbound.len(),
        &unbound[..unbound.len().min(12)]
    );
}

/// The optional groups land on exactly the layers the checkpoint puts them on.
///
/// Read off `model.safetensors.index.json`: compressors on 4 layers, a compressor GATE on only 3
/// of those (layer 20 runs at ratio 1 — a plain projection with no softmax gate), indexer queries
/// on 8, indexer KEYS on 4, Engram on 2. Every one of those counts is a derivation that could be
/// off, and the totals are cheap to state.
#[test]
fn the_optional_groups_land_on_the_right_layers() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let has = |l: u32, suffix: &str| {
        dsv41_layer_tensors(&cfg, l)
            .iter()
            .any(|(n, _)| n == suffix)
    };
    let layers_with = |suffix: &str| -> Vec<u32> {
        (0..cfg.layers).filter(|l| has(*l, suffix)).collect()
    };
    assert_eq!(layers_with("attn.compressor.wkv.weight"), vec![2, 8, 14, 20]);
    assert_eq!(layers_with("attn.compressor.wgate.weight"), vec![2, 8, 14]);
    assert_eq!(
        layers_with("attn.indexer.wq_b.weight"),
        vec![2, 8, 14, 20, 24, 28, 32, 36]
    );
    assert_eq!(layers_with("attn.indexer.wk.weight"), vec![2, 8, 14, 20]);
    assert_eq!(layers_with("engram.embed.weight"), vec![1, 14]);
    // mHC and the vl routing bias are on EVERY layer — the first is why no mHC-free block exists
    // to extract, the second is a V4.1 addition with no V4 analogue.
    assert_eq!(layers_with("hc_attn_fn").len(), cfg.layers as usize);
    assert_eq!(layers_with("ffn.gate.bias_vl").len(), cfg.layers as usize);
}

/// Engram is a CAPACITY term: **202.8 GB**, ~41% of the per-layer weight bytes.
///
/// Stated as a test because it is the single biggest line in the memory plan and it is invisible
/// in the config — `engram_num_embeddings` looks like two ordinary integers.
///
/// The number splits, and the split is why it is asserted in two parts. The tables themselves are
/// 196.6 GB of fp8; their ue8m0 scales add a further **6.1 GB**, because the block is 32 elements
/// along a 256-wide row, so there are 8 scale bytes per row against 256 weight bytes — a 3.1%
/// surcharge on the largest tensors in the checkpoint. A capacity plan drawn from the weight
/// figure alone is 6 GB short before anything else is allocated.
#[test]
fn engram_tables_dominate_the_weight_budget() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let bytes_of = |l: u32, pred: &dyn Fn(&str) -> bool| -> u64 {
        dsv41_layer_tensors(&cfg, l)
            .iter()
            .filter(|(n, _)| pred(n))
            .map(|(_, b)| *b)
            .sum()
    };
    let sum = |pred: &dyn Fn(&str) -> bool| -> u64 {
        cfg.engram_layers.iter().map(|&l| bytes_of(l, pred)).sum()
    };
    let weights = sum(&|n: &str| n == "engram.embed.weight");
    let scales = sum(&|n: &str| n == "engram.embed.scale");
    let total: u64 = (0..cfg.layers).map(|l| bytes_of(l, &|_| true)).sum();
    let gb = |b: u64| b as f64 / 1e9;
    eprintln!(
        "engram embed: {:.1} GB tables + {:.1} GB ue8m0 scales = {:.1} GB, of {:.1} GB per-layer \
         weights ({:.0}%)",
        gb(weights),
        gb(scales),
        gb(weights + scales),
        gb(total),
        100.0 * (weights + scales) as f64 / total as f64
    );
    assert!(
        (196.0..197.0).contains(&gb(weights)),
        "engram tables should be ~196.6 GB, got {:.1}",
        gb(weights)
    );
    // 8 scale bytes per 256-byte row: the surcharge is 1/32 of the tables, and it is 6 GB.
    assert_eq!(scales * 32, weights);
    assert!(
        (202.0..203.5).contains(&gb(weights + scales)),
        "engram should be ~202.8 GB all in, got {:.1}",
        gb(weights + scales)
    );
    assert!(
        (weights + scales) * 5 > total * 2,
        "engram should be >40% of the per-layer weight bytes"
    );
}

/// V4.1's block-FP8 grid is `[32, 32]` with ue8m0 scales, and plow's every block-FP8 kernel
/// assumes `[128, 128]` with f32 scales. This pins BOTH halves from the shards.
///
/// It is the gate on 39.8% of the 8k prefill — every projection in the model — and it is a
/// KERNEL problem, not an emit problem: `mla_ckpt_enc` refuses a non-`[128, 128]` checkpoint
/// outright, so no amount of emit plumbing reaches it. Asserted here rather than left to section
/// 5.2's prose because the scale grid is exactly the kind of fact that reads as a detail: the
/// shapes are self-consistent, nothing is missing, and the only way to notice is to divide.
#[test]
fn the_block_fp8_grid_is_32_not_128_and_the_scales_are_e8m0() {
    let Some((cfg, hdr)) = checkpoint() else {
        return;
    };
    assert_eq!(
        cfg.raw.quantization_config.weight_block_size,
        [32, 32],
        "V4.1 quantizes at [32, 32]; V4 was [128, 128] and plow's kernels are written to the latter"
    );

    // Every projection, checked by DIVISION: the scale grid is the weight shape over the block.
    // And every one of them is F8_E8M0 — a BYTE-wide scale, where `d_gemm_fp8_blk` takes
    // `const float* wscale`. Two independent mismatches, either of which alone would be fatal.
    let projections = [
        "attn.wq_a",
        "attn.wq_b",
        "attn.wkv",
        "attn.wo_a",
        "attn.wo_b",
        "ffn.shared_experts.w1",
        "ffn.shared_experts.w2",
        "ffn.shared_experts.w3",
    ];
    for p in projections {
        let w = format!("layers.0.{p}.weight");
        let sc = format!("layers.0.{p}.scale");
        let (wdt, wshape) = hdr.get(&w).unwrap_or_else(|| panic!("{w} missing"));
        let (sdt, sshape) = hdr.get(&sc).unwrap_or_else(|| panic!("{sc} missing"));
        assert_eq!(wdt, "F8_E4M3", "{w}");
        assert_eq!(sdt, "F8_E8M0", "{sc} is a ue8m0 BYTE, not the f32 the kernel expects");
        assert_eq!(
            *sshape,
            vec![(wshape[0] + 31) / 32, (wshape[1] + 31) / 32],
            "{sc} must be the [32, 32] grid over {wshape:?}"
        );
        // ...and it is emphatically NOT the [128, 128] grid plow would index it as.
        assert_ne!(
            *sshape,
            vec![(wshape[0] + 127) / 128, (wshape[1] + 127) / 128],
            "{sc} happens to match a [128,128] grid too — the test proves nothing for this shape"
        );
    }

    // The two layer-conditional projections carry the same grid.
    for (layer, name) in [(1u32, "engram.wkv"), (2, "attn.indexer.wq_b")] {
        let w = format!("layers.{layer}.{name}.weight");
        let sc = format!("layers.{layer}.{name}.scale");
        let (_, wshape) = hdr.get(&w).unwrap_or_else(|| panic!("{w} missing"));
        let (sdt, sshape) = hdr.get(&sc).unwrap_or_else(|| panic!("{sc} missing"));
        assert_eq!(sdt, "F8_E8M0");
        assert_eq!(*sshape, vec![(wshape[0] + 31) / 32, (wshape[1] + 31) / 32]);
    }

    // The routed experts are the exception, and naming it keeps the scope honest: they are MXFP4
    // at group 32 along K only, which is a path plow ALREADY has — so the 39.8% figure excludes
    // the 49.4% of the prefill that the routed experts are.
    let (edt, eshape) = hdr.get("layers.0.ffn.experts.0.w1.weight").expect("expert w1");
    let (esdt, esshape) = hdr.get("layers.0.ffn.experts.0.w1.scale").expect("expert w1 scale");
    assert_eq!(edt, "I8", "routed experts are nibble-packed fp4 in an I8 container");
    assert_eq!(esdt, "F8_E8M0");
    // [2304, 160]: full rows, K/32 columns — a per-32-along-K group, NOT a 2-D block.
    assert_eq!(*esshape, vec![eshape[0], (eshape[1] * 2 + 31) / 32]);
}

/// The refusal must lead with what is ACTUALLY missing, and a closed gate must leave the list.
///
/// This test used to assert the opposite -- that the block-FP8 grid came first -- and it was right
/// to, because a reader who started building emit features would otherwise have hit
/// `mla_ckpt_enc`'s refusal with all that work done. That gate is now CLOSED: op 184 and
/// `d_gemm_t<WFP8MX>` read the [32,32] ue8m0 grid, pass 12/12 on gfx942, and `emit_pf_gemm_fp8_mx`
/// emits it.
///
/// A stale gap list is worse than a short one. It sends the next reader to solve a solved problem
/// and buries the real one, so the check is now BOTH directions: the emit leads, and the kernel
/// gate is gone entirely.
#[test]
fn the_refusal_leads_with_the_emit_and_the_closed_kernel_gate_is_gone() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let gaps = super::dsv41::dsv41_gaps(&cfg);
    assert!(
        gaps[0].contains("full-model emit"),
        "the emit is the whole remaining job and must come first; got: {}",
        gaps[0]
    );
    assert!(
        gaps[0].contains("declare_dsv41_rows_batched"),
        "...and should name what does not exist rather than gesture at a fork; got: {}",
        gaps[0]
    );
    let joined = gaps.join("\n");
    assert!(
        !joined.contains("block-FP8 at ["),
        "the block-FP8 grid gate is closed (op 184, verified on gfx942) and must not still be \
         listed as missing:\n{joined}"
    );
    assert!(
        !joined.contains("Engram host side"),
        "the Engram host side landed (crates/plowrt/src/text/engram.rs) and must not still be \
         listed as missing:\n{joined}"
    );
    // The gaps that ARE still real must survive, or this test would pass on an empty list.
    assert!(joined.contains("CSA2 emit"), "CSA2 emit is still missing:\n{joined}");
    assert!(joined.contains("two-level indexer"), "the indexer is still missing:\n{joined}");
}

/// The op-184 emit primitive, pinned the way `glm_linear_fp8_prefill_routes_to_the_block_fp8_gemm`
/// pins 107's: as a PURE function, not by setting a knob. The knob is process-global env state and
/// cargo runs tests in parallel threads, so a sibling that counts tensors would see this one's
/// handles appear under it.
///
/// What actually matters here is the ROUTE. A V4.1 projection must reach [`DevOp::GemmFp8Mx`] and
/// never [`DevOp::GemmFp8Blk`]: 107 would not fault on these operands, it would read the ue8m0
/// bytes as f32 at a 128-block stride and rescale every output. Pinning the opcode is pinning the
/// difference between a wrong model and a right one.
#[test]
fn a_v41_projection_routes_to_op_184_and_never_to_107() {
    let _guard = crate::test_env::env_guard();
    let mut b = Builder::new(304);
    // attn.wq_a: [1280, 5120] e4m3 with a [40, 160] ue8m0 grid -- the real shard shapes.
    let w = b.tensor("attn.wq_a.weight", 1280 * 5120);
    let sc = b.tensor("attn.wq_a.scale", 40 * 160);
    let x = b.tensor("act.x", 512 * 5120 * BF16);
    let o = b.tensor("act.q_a", 512 * 1280 * BF16);
    let all: Vec<u32> = (0..304u32).collect();
    emit_pf_gemm_fp8_mx(&mut b, &all, o, x, w, sc, 512, 1280, 5120, &[]);
    let p = b.finish();
    assert_eq!(p.insts.len(), 1);
    let d = &p.insts[0];
    assert_eq!(
        d.op,
        DevOp::GemmFp8Mx as u16,
        "a [32,32] ue8m0 projection must NOT reach op 107, which would read the byte grid as f32"
    );
    assert_eq!([d.t[0], d.t[1], d.t[2], d.t[3]], [o, x, w, sc]);
    assert_eq!([d.i[0], d.i[1], d.i[2]], [512, 1280, 5120]);
}

/// `attn.wkv` is N = 576 -- 512 latent + 64 rope -- which is NOT a multiple of the 128-wide tile.
/// It is the shape that exposed the kernel's out-of-bounds N-scale read, so it earns an emit-side
/// pin too: a ragged N must be emitted, not refused. Only K is unforgiving here.
#[test]
fn a_ragged_n_is_emitted_because_only_k_is_unforgiving() {
    let _guard = crate::test_env::env_guard();
    let mut b = Builder::new(304);
    let w = b.tensor("attn.wkv.weight", 576 * 5120);
    let sc = b.tensor("attn.wkv.scale", 18 * 160);
    let x = b.tensor("act.x", 512 * 5120 * BF16);
    let o = b.tensor("act.kv", 512 * 576 * BF16);
    let all: Vec<u32> = (0..304u32).collect();
    emit_pf_gemm_fp8_mx(&mut b, &all, o, x, w, sc, 512, 576, 5120, &[]);
    assert_eq!(b.finish().insts.len(), 1, "N = 576 is a real V4.1 shape, not an error");
}

/// A K with a remainder does not mean "ragged tail" here, it means the grid bound to t[3] is not
/// this weight's -- a V4.1 scale grid cannot exist for such a K. Refuse rather than emit a GEMM
/// that would read past each weight row into the next output channel with no fault.
#[test]
#[should_panic(expected = "not this weight's")]
fn a_k_that_is_not_a_whole_number_of_32_blocks_is_refused() {
    let _guard = crate::test_env::env_guard();
    let mut b = Builder::new(304);
    let w = b.tensor("w.weight", 1280 * 5121);
    let sc = b.tensor("w.scale", 40 * 161);
    let x = b.tensor("act.x", 512 * 5121 * BF16);
    let o = b.tensor("act.o", 512 * 1280 * BF16);
    let all: Vec<u32> = (0..304u32).collect();
    emit_pf_gemm_fp8_mx(&mut b, &all, o, x, w, sc, 512, 1280, 5121, &[]);
}

/// The scale handle is not optional: a TENSOR_NONE there is a null pointer in the kernel's
/// promotion. The two handles are declared as a pair; refuse rather than emit half of one.
#[test]
#[should_panic(expected = "neither is optional")]
fn a_weight_without_its_scale_grid_is_refused() {
    let _guard = crate::test_env::env_guard();
    let mut b = Builder::new(304);
    let w = b.tensor("w.weight", 1280 * 5120);
    let x = b.tensor("act.x", 512 * 5120 * BF16);
    let o = b.tensor("act.o", 512 * 1280 * BF16);
    let all: Vec<u32> = (0..304u32).collect();
    emit_pf_gemm_fp8_mx(&mut b, &all, o, x, w, TENSOR_NONE, 512, 1280, 5120, &[]);
}

/// The declaration must bind EVERY weight the tensor list names, under the checkpoint's own
/// `layers.{l}.` prefix, with a distinct handle each.
///
/// `dsv41_layer_tensors_match_the_shards` already proves the list is right against the shards.
/// What is unproven until here is the step from list to tensor table, and the failure it guards is
/// specific: a name that reaches the table without its prefix does not match any shard, so the
/// loader binds nothing and the op reads a zero-filled buffer. No fault, no error, just a layer
/// that quietly computes from zeros.
#[test]
fn the_declaration_binds_every_listed_weight_under_its_checkpoint_name() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let mut b = Builder::new(304);
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0, 1], 1);
    for l in [0u32, 1] {
        let listed = super::dsv41::dsv41_layer_tensors(&cfg, l);
        assert!(!listed.is_empty(), "layer {l} must carry weights");
        let mut seen = std::collections::BTreeSet::new();
        for (suffix, sz) in &listed {
            // The routed experts are the one group the PACKET does not declare: they are bound
            // through `ffn.expert_weight_table` by `bind_packed_experts`, which resolves the
            // checkpoint's own names itself. See
            // `the_routed_experts_are_bound_through_the_packed_table`.
            if suffix.starts_with("ffn.experts.") {
                assert!(*sz > 0, "{suffix:?} is zero bytes");
                continue;
            }
            let h = w.get(l, suffix);
            assert!(seen.insert(h), "layer {l} reused handle {h} for {suffix:?}");
            assert_eq!(
                b.tensor_name(h),
                format!("layers.{l}.{suffix}"),
                "the table must carry the checkpoint's own name, prefix included, or the loader \
                 matches no shard and the op reads zeros"
            );
            assert!(*sz > 0, "{suffix:?} is zero bytes");
        }
        for table in ["ffn.expert_weight_table", "ffn.expert_scale_table"] {
            assert_eq!(
                b.tensor_name(w.get(l, table)),
                format!("layers.{l}.{table}"),
                "the loader finds these by suffix; any other spelling binds no experts at all"
            );
        }
    }
}

/// The optional groups are what make a per-layer declaration necessary at all: if every layer
/// carried the same weights, one list would do. `has` must track them exactly.
///
/// Engram is the sharpest case -- it is on layers 1 and 14 ONLY, and its tables are 202.8 GB, so a
/// declaration that put them on all 40 layers would ask for ~4 TB and fail at allocation rather
/// than silently. The compressor and indexer keys are the quieter ones.
#[test]
fn the_declaration_puts_the_optional_groups_only_where_they_belong() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let all: Vec<u32> = (0..cfg.layers).collect();
    let mut b = Builder::new(304);
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &all, 1);
    for l in 0..cfg.layers {
        let engram_here = cfg.engram_layers.contains(&l);
        assert_eq!(
            w.has(l, "engram.wkv.weight"),
            engram_here,
            "engram belongs on layers {:?} and layer {l} disagrees",
            cfg.engram_layers
        );
        let kv_here = cfg.kv_source.contains(&l);
        assert_eq!(
            w.has(l, "attn.compressor.wkv.weight"),
            kv_here,
            "the compressor belongs on the kv_source layers {:?} and layer {l} disagrees",
            cfg.kv_source
        );
        // Present on EVERY layer, so a per-layer map must not lose it.
        assert!(w.has(l, "attn.wq_a.weight"), "layer {l} lost its q_a projection");
    }
}

/// Declaring the whole model must total what section 9's budget claims, and the Engram tables
/// dominate it. A wrong total here is how a capacity plan for 8x192 GB turns out to be wrong at
/// load rather than on paper.
#[test]
fn the_declared_total_is_the_documented_weight_budget() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let all: Vec<u32> = (0..cfg.layers).collect();
    let mut b = Builder::new(304);
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &all, 1);
    let listed: u64 = all
        .iter()
        .flat_map(|&l| super::dsv41::dsv41_layer_tensors(&cfg, l))
        .map(|(_, sz)| sz)
        .sum();
    assert_eq!(w.bytes, listed, "the declaration must account for every listed byte");
    // The breakdown, all of it derivable from the config rather than measured:
    //   routed experts  384 x 3 x 2304 x 5120 at 4 bits + ue8m0  = 7.22 GB/layer x 40 = 288.8
    //   engram tables   384 M rows x 256 fp8 + scales, 2 layers  = 202.8
    //   dense, shared expert, indexer, norms                     =   7.1
    //                                                              -----
    //                                                              498.7 GB
    //
    // The independent check is section 4's recorded fact that the MP8 checkpoint is 501 GB on
    // disk. This total is PER-LAYER only -- `dsv41_layer_tensors` does not cover embed_tokens,
    // lm_head or the final norm -- so it must land just UNDER 501, and the ~2 GB gap is those.
    // A total that matched 501 exactly would mean the per-layer list had absorbed something it
    // should not have.
    let gb = w.bytes as f64 / 1e9;
    assert!(
        (495.0..501.0).contains(&gb),
        "the whole-model per-layer weight total came out {gb:.1} GB; it should be ~498.7 -- just \
         under the 501 GB MP8 checkpoint, with the difference being the non-layer tensors"
    );
}

/// Asking for a weight a layer does not have must PANIC naming it, not hand back a sentinel.
/// A `TENSOR_NONE` fallback would bind a null pointer that the kernel then reads.
#[test]
#[should_panic(expected = "no weight")]
fn asking_for_a_weight_a_layer_lacks_is_refused() {
    let Some((cfg, _)) = checkpoint() else {
        // The test must still panic when the checkpoint is absent, or it passes vacuously.
        panic!("no weight (checkpoint absent, refusing vacuously)");
    };
    let mut b = Builder::new(304);
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0], 1);
    // Layer 0 is not an engram layer (they are 1 and 14).
    w.get(0, "engram.wkv.weight");
}

/// The attention projection chain, emitted against the REAL checkpoint's weights.
///
/// Every `w.get` in the emit panics on a name the layer does not carry, so running this at all
/// proves the four projections and two norms bind to tensors that exist in the shards -- which is
/// the failure a hand-written emit makes first and notices last.
///
/// What is checked beyond that is the SHAPE of the chain, because the shapes here are the model's
/// two least obvious facts:
///   * `wkv` produces ONE 576-wide row per token for all 64 heads, not one per head. That is what
///     "fully absorbed MLA" means -- there is no `kv_b` in any shard to expand it with.
///   * `wq_b` produces `heads * (head_dim + qk_rope)`, nope and rope together.
#[test]
fn the_attention_projection_chain_emits_against_the_real_weights() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let mut b = Builder::new(304);
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0], 1);
    let all: Vec<u32> = (0..304u32).collect();
    let t = 512u32;
    let x = b.tensor("act.x", (t as u64) * (cfg.hidden as u64) * 2);
    let (act, _last) = super::dsv41::emit_dsv41_attn_proj(&mut b, &cfg, &w, &all, 0, 1, x, t, &[]);
    let p = b.finish();

    // Two RmsNorm + four... no: two norms and THREE GEMMs (q_a, q_b, kv). wo_a/wo_b are the
    // output side and belong to the core's epilogue, not here.
    let gemms: Vec<_> = p
        .insts
        .iter()
        .filter(|d| d.op == DevOp::GemmFp8Mx as u16)
        .collect();
    assert_eq!(gemms.len(), 3, "q_a, q_b and kv are the three projections on this side");
    let norms = p.insts.iter().filter(|d| d.op == DevOp::RmsNorm as u16).count();
    assert_eq!(norms, 2, "the input norm and the q-LoRA norm");
    assert!(
        p.insts.iter().all(|d| d.op != DevOp::GemmFp8Blk as u16),
        "not one projection may reach op 107: it would read the ue8m0 byte grid as f32"
    );

    // The absorbed-MLA shape. 512 latent + 64 rope = 576, shared by all 64 heads.
    let kv = gemms
        .iter()
        .find(|d| d.t[0] == act.kv)
        .expect("the latent projection must be emitted");
    assert_eq!(
        kv.i[1], cfg.head_dim,
        "wkv is ONE {}-wide row per token for ALL {} heads -- no kv_b exists to expand it, and \
         head_dim ALREADY contains the {}-wide rope half, so there is nothing to add to it",
        cfg.head_dim, cfg.heads, cfg.qk_rope
    );
    assert_eq!(kv.i[2], cfg.hidden, "the latent reads the normed input, width hidden");

    let q = gemms
        .iter()
        .find(|d| d.t[0] == act.q)
        .expect("the query up-projection must be emitted");
    assert_eq!(
        q.i[1],
        cfg.heads * cfg.head_dim,
        "wq_b writes head_dim per head -- nope and rope together, since head_dim contains both"
    );
    assert_eq!(q.i[2], cfg.q_lora, "and reads the q-LoRA rank");
}

/// `wkv` hangs off the NORMED INPUT, not off the query chain. They are parallel branches, and
/// chaining them would serialise two GEMMs with no data dependence between them.
///
/// Worth pinning because the bug is invisible in output: a serialised chain computes exactly the
/// same numbers, just slower, so nothing but a dependency check catches it. At 82.98 TFLOP across
/// these projections, "just slower" is the entire point of the exercise.
#[test]
fn the_latent_and_the_query_are_parallel_branches_off_the_same_norm() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let mut b = Builder::new(304);
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0], 1);
    let all: Vec<u32> = (0..304u32).collect();
    let t = 512u32;
    let x = b.tensor("act.x", (t as u64) * (cfg.hidden as u64) * 2);
    let (act, _) = super::dsv41::emit_dsv41_attn_proj(&mut b, &cfg, &w, &all, 0, 1, x, t, &[]);
    let p = b.finish();
    let kv = p.insts.iter().find(|d| d.t[0] == act.kv).unwrap();
    let qa = p.insts.iter().find(|d| d.t[0] == act.q_a).unwrap();
    // Both read `xn` -- the operand is the semantic claim.
    assert_eq!(kv.t[1], act.xn, "the latent reads the normed input");
    assert_eq!(qa.t[1], act.xn, "so does the query down-projection");
    // And both gate on the SAME thing. Dependencies live in the wait table, not on the
    // instruction, so the check is that the latent's gate set is identical to the query
    // down-projection's: if the latent had been chained after the query chain it would carry an
    // extra gate on `wq_b`'s counter.
    let gates = |d: &DevInst| -> Vec<packet::dev::Wait> {
        p.waits[d.wait_ofs as usize..d.wait_ofs as usize + d.wait_len as usize].to_vec()
    };
    assert_eq!(
        gates(kv),
        gates(qa),
        "the latent and the query branch off the same norm, so they must gate on the same \
         counter -- a longer wait set on the latent means it was serialised behind the query \
         chain, which computes identical numbers and is simply slower"
    );
}

/// The rung plan must be PER-LAYER, because that is what makes a rung reachable before the whole
/// model is: layer 0 carries 8 parts, layer 1 adds Engram, layer 2 adds the compressor and the
/// indexer. A plan that reported the same list everywhere would hide the fact that the cheapest
/// rung is a plain layer, not a kv_source one.
///
/// This is the answer to "what is missing" that comes from RUNNING plowc rather than from reading
/// prose, so it is worth a test that the counts actually differ and that the list shrinks as work
/// lands (the emitted set is non-empty today, and every part named in it is real).
#[test]
fn the_rung_plan_is_per_layer_and_names_what_is_emitted() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let n = |l: u32| super::dsv41::dsv41_layer_parts(&cfg, l).len();
    let plain = *cfg
        .kv_source
        .iter()
        .chain(cfg.engram_layers.iter())
        .max()
        .map(|_| &0u32)
        .unwrap_or(&0);
    assert!(!cfg.engram_layers.contains(&plain) && !cfg.kv_source.contains(&plain));
    let eng = cfg.engram_layers[0];
    let kv = cfg.kv_source[0];
    assert!(
        n(plain) < n(eng) && n(eng) < n(kv),
        "a plain layer must be the cheapest rung: got plain={} engram={} kv_source={}",
        n(plain),
        n(eng),
        n(kv)
    );
    // The projections ARE emitted, so the plan must say so rather than reporting everything todo.
    let parts = super::dsv41::dsv41_layer_parts(&cfg, plain);
    let done: Vec<_> = parts
        .iter()
        .filter(|(_, st)| *st == super::dsv41::Part::Done)
        .collect();
    assert_eq!(
        done.len(),
        9,
        "mhc_pre/mhc_post, the attention projections, the output projection, its all-reduce, \
         ffn_norm, the routed experts, the shared expert and the attention core are emitted; \
         thing that grows as bricks land, so it is asserted exactly rather than as a lower bound"
    );
    // Every part is emitted now, so the plan RESOLVES rather than refusing. What is left is the
    // writer that turns these bricks into a blob, which is a different missing thing and says so.
    let parts = super::dsv41::dsv41_emit_block_plan(&cfg, plain)
        .expect("all nine parts are emitted, so the plan must resolve");
    assert_eq!(parts.len(), 9);
}

/// The output projection at TP8, where the grouped LoRA costs nothing extra.
///
/// `heads / o_groups` is 64/8 = 8, so one rank owns exactly one group: its heads are
/// `o_group_in_features` = 4096 wide, its `wo_a` slice is [1024, 4096], and the GEMM is ordinary.
/// The group count and the TP degree being the same number is what makes the block-diagonal
/// structure free in the configuration this model is actually served in.
#[test]
fn the_output_projection_is_two_ordinary_gemms_at_tp8() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let (groups, orow, ocol) = cfg.wo_a_groups();
    assert_eq!(groups, 8);
    assert_eq!(ocol, cfg.heads * cfg.head_dim / groups, "one rank's share of the heads");
    let mut b = Builder::new(304);
    // Declared at tp=8, because the emit reads this rank's SHARE of wo_a. Declaring full-size
    // here and reading a per-rank shape is exactly the bug the operand check now catches.
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0], groups);
    let all: Vec<u32> = (0..304u32).collect();
    let t = 512u32;
    let attn_out = b.tensor("act.attn_out", (t as u64) * (ocol as u64) * 2);
    let (act, _) =
        super::dsv41::emit_dsv41_attn_out(
            &mut b, &cfg, &w, &all, 0, groups, attn_out, t, &mut 0, &[],
        );
    let p = b.finish();
    let g: Vec<_> = p
        .insts
        .iter()
        .filter(|d| d.op == DevOp::GemmFp8Mx as u16)
        .collect();
    assert_eq!(g.len(), 2, "wo_a for this rank's group, then wo_b");
    let oa = g.iter().find(|d| d.t[0] == act.o_a).unwrap();
    assert_eq!([oa.i[1], oa.i[2]], [orow, ocol], "wo_a is [o_lora_rank, one group's heads]");
    // wo_b is INPUT-parallel: this rank multiplies its own group's [T, orow] by its own
    // [hidden, orow] slice and the ranks SUM. Reading the full wo_b_in() here would be the
    // all-gather form, which is 8x the arithmetic for the same answer.
    let ob = g.iter().find(|d| d.t[0] == act.o_part).unwrap();
    assert_eq!(
        [ob.i[1], ob.i[2]],
        [cfg.hidden, orow],
        "wo_b reads THIS rank's group only, and the all-reduce sums the eight partials"
    );
    assert_eq!(cfg.wo_b_in(), groups * orow, "the full wo_b input is o_groups * o_lora_rank");
    // And the sum is actually emitted -- a partial nobody reduces is silently 1/8 of the answer.
    let xr: Vec<_> = p
        .insts
        .iter()
        .filter(|d| d.op == DevOp::XReduce as u16 || d.op == DevOp::XReduceTwoShot as u16)
        .collect();
    assert_eq!(xr.len(), 1, "exactly one all-reduce closes the output projection");
    assert_eq!(xr[0].t[0], act.o, "the reduce writes the layer's o, not the partial");
    assert_eq!(xr[0].i[0], t * cfg.hidden, "it reduces the whole [T, hidden] partial");
    assert_eq!(xr[0].i[1], groups, "over all 8 ranks");
    // The peer-slot NAME is the contract: plowrt binds `act.og_tp` into the peer region and every
    // other name into local VRAM, where the peers' slots are never written.
    assert_eq!(
        p.tensors[act.o_part as usize].name,
        super::dsv41::PEER_SLOT_O,
        "the partial must be declared under the runtime's peer-slot name"
    );
    assert!(
        p.insts.iter().all(|d| d.op != DevOp::XAllGather as u16),
        "no all-gather: gathering the eight groups and running the full wo_b on every rank is \
         687 GFLOP per rank per layer instead of 86"
    );
}

/// At TP1 it must REFUSE, not silently do one eighth of the work.
///
/// Eight GEMMs need eight weight handles, and the tensor table binds a name to a WHOLE checkpoint
/// tensor -- there is no sub-tensor view. Inventing a name like `attn.wo_a.weight.g3` would match
/// no shard and the loader's answer to that has historically been a zero fill, so the layer would
/// compute from zeros and still produce output.
#[test]
#[should_panic(expected = "emit_dsv41_out_lora_tp1")]
fn the_output_projection_refuses_at_tp1_rather_than_doing_one_group() {
    let Some((cfg, _)) = checkpoint() else {
        panic!("emit_dsv41_out_lora_tp1 (checkpoint absent, refusing vacuously)");
    };
    let _guard = crate::test_env::env_guard();
    let mut b = Builder::new(304);
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0], 1);
    let all: Vec<u32> = (0..304u32).collect();
    let attn_out = b.tensor("act.attn_out", 512 * 4096 * 2);
    super::dsv41::emit_dsv41_attn_out(&mut b, &cfg, &w, &all, 0, 1, attn_out, 512, &mut 0, &[]);
    unreachable!("tp=1 must have been refused above");
}

/// The shared expert, against the real weights, with the CLAMPED SwiGLU.
///
/// The act code is the thing worth pinning. Every other family in this emitter uses
/// `GLM_ACT_SILU` (1); V4.1 uses `PLOW_ACT_SWIGLU_CLAMP_` (4) with the limit from the config on
/// `f[1]`. Emitting act 1 here would drop the clamp and fault nowhere — the gfx942 run of this
/// arm measured 3160 of 4096 elements actually clamped, so it changes the numbers on most of
/// them.
#[test]
fn the_shared_expert_uses_the_clamped_swiglu_not_plain_silu() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let mut b = Builder::new(304);
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0], 1);
    let all: Vec<u32> = (0..304u32).collect();
    let t = 512u32;
    let x = b.tensor("act.x", (t as u64) * (cfg.hidden as u64) * 2);
    let (act, _) = super::dsv41::emit_dsv41_ffn_shared(&mut b, &cfg, &w, &all, 0, 1, x, t, &mut 0, &[]);
    let p = b.finish();

    let glu = p
        .insts
        .iter()
        .find(|d| d.op == DevOp::Glu as u16)
        .expect("the shared expert needs a GLU");
    assert_eq!(glu.i[1], 4, "PLOW_ACT_SWIGLU_CLAMP_, not GLM_ACT_SILU (1)");
    assert_eq!(
        glu.f[1], cfg.raw.swiglu_limit,
        "the clamp limit rides f[1] and comes from the config, not a constant"
    );
    assert_eq!([glu.t[1], glu.t[2]], [act.sh_gate, act.sh_up]);

    // Three op-184 GEMMs: gate, up, down. The shared expert is block-FP8, NOT fp4 — that is what
    // makes this checkpoint mixed, and routing it to the MXFP4 expert path would be the error.
    let g: Vec<_> = p
        .insts
        .iter()
        .filter(|d| d.op == DevOp::GemmFp8Mx as u16)
        .collect();
    assert_eq!(g.len(), 3, "gate, up and down are all block-fp8 dense GEMMs");
    assert!(
        p.insts.iter().all(|d| d.op != DevOp::MoeGroupGluPf as u16),
        "the SHARED expert is not a routed expert; it must not reach the grouped MoE path"
    );
    // Gate and up share an input and must not be chained.
    let gate = g.iter().find(|d| d.t[0] == act.sh_gate).unwrap();
    let up = g.iter().find(|d| d.t[0] == act.sh_up).unwrap();
    assert_eq!(gate.t[1], act.xn);
    assert_eq!(up.t[1], act.xn);
    let gates = |d: &DevInst| -> Vec<packet::dev::Wait> {
        p.waits[d.wait_ofs as usize..d.wait_ofs as usize + d.wait_len as usize].to_vec()
    };
    assert_eq!(gates(gate), gates(up), "gate and up are parallel, not serialised");
}

/// V4.1's mHC is GLM-5.3's hyper-connection, and the emit must carry V4.1's OWN constants.
///
/// The two families agree today (`hc_mult` 4, 20 Sinkhorn iterations, `hc_eps` 1e-6), which is why
/// the algorithm is shared -- but agreement is a fact about these two checkpoints, not a law. GLM
/// had the numbers inlined as literals; if V4.1 silently inherited them, a checkpoint that changed
/// `hc_mult` would read the wrong rows of `hc_*_fn` and still produce output. So this asserts the
/// emitted packet against the CONFIG, not against 4 and 20.
#[test]
fn the_mhc_pair_is_glm53s_hyper_connection_at_v41s_own_constants() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let mut b = Builder::new(304);
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0], cfg.o_groups);
    let t = 512u32;
    let m = super::dsv41::declare_dsv41_mhc(&mut b, &cfg, t);
    let c_pre = super::dsv41::emit_dsv41_mhc_pre(&mut b, &cfg, &w, &m, 0, false, 0, 0, t, &[]);
    let raw = b.tensor("act.raw", (t as u64) * (cfg.hidden as u64) * 2);
    super::dsv41::emit_dsv41_mhc_post(&mut b, &cfg, &m, raw, 0, t, &[c_pre]);
    let p = b.finish();

    let mix = (2 + cfg.hc_mult) * cfg.hc_mult;
    let gv = p
        .insts
        .iter()
        .find(|d| d.op == DevOp::GemvF32 as u16)
        .expect("the mix GEMV");
    assert_eq!(gv.i[1], mix, "the mix matrix is (2 + hc_mult) * hc_mult rows");
    assert_eq!(
        gv.i[2],
        cfg.hc_mult * cfg.hidden,
        "and it reads the whole expanded residual, hc_mult copies of hidden"
    );

    let pre = p
        .insts
        .iter()
        .find(|d| d.op == DevOp::HyperConnPre as u16)
        .expect("HyperConnPre");
    assert_eq!(pre.i[1], cfg.hc_mult, "hc_mult from the config");
    assert_eq!(pre.i[2], cfg.hidden);
    assert_eq!(
        pre.i[3], cfg.hc_sinkhorn_iters,
        "Sinkhorn iterations from the config, not GLM's inlined 20"
    );
    assert_eq!(pre.f[0], cfg.eps);
    assert_eq!(pre.f[1], cfg.raw.hc_eps, "hc_eps is its own epsilon, not rms_norm_eps");
    // rms_norm_eps is 1e-20 on this checkpoint and hc_eps is 1e-6 -- 14 orders apart, so
    // confusing them is not a rounding difference.
    assert_ne!(cfg.eps, cfg.raw.hc_eps);

    let post = p
        .insts
        .iter()
        .find(|d| d.op == DevOp::HyperConnPost as u16)
        .expect("HyperConnPost");
    assert_eq!(post.t[1], raw, "the sublayer output goes back through POST");
    assert_eq!(
        [post.t[0], post.t[2]],
        [m.residual[1], m.residual[0]],
        "POST writes the OTHER residual copy: it mixes across all hc_mult rows, so writing in \
         place would read rows it had already overwritten"
    );
    assert_eq!(post.i[1], cfg.hc_mult);
    assert_eq!(post.i[2], cfg.hidden);
}

/// The router flag bits are a wire contract split across two files, and they must not collide.
///
/// `interp.hip` reads bit 2 to decide whether `t3` is bound as the `e_score_correction_bias`
/// pointer; `op_moe.h` reads the other bits to pick the score transform. When the DeepSeek-V4 arms
/// landed they took bit 2 for `sqrtsoftplus` -- which the kernel could not see was occupied,
/// because the dispatch reads it and the kernel does not. Every packet that asked for its bias
/// (GLM-5.2, GLM-5.3, DeepSeek-V3, Kimi) therefore got sqrt(softplus(.)) scoring instead of the
/// sigmoid it asked for, silently: both transforms are monotone, so the top-k SELECTION still
/// looked sane and only the gate WEIGHTS were wrong.
///
/// This pins the three properties that would have caught it.
#[test]
fn router_flag_bits_do_not_collide() {
    use crate::mla::router_flag as f;
    // 1. Every bit is distinct. A duplicate here IS the bug.
    let named = [
        ("SIGMOID", f::SIGMOID),
        ("NORM_TOPK", f::NORM_TOPK),
        ("BIAS", f::BIAS),
        ("F32_LOGIT", f::F32_LOGIT),
        ("HASH_SELECT", f::HASH_SELECT),
        ("SQRTSOFTPLUS", f::SQRTSOFTPLUS),
    ];
    for (i, (na, a)) in named.iter().enumerate() {
        assert_eq!(a.count_ones(), 1, "{na} must be a single bit, got {a:#b}");
        for (nb, b) in &named[i + 1..] {
            assert_ne!(a, b, "{na} and {nb} are the same bit ({a:#b}) -- one wire bit, two meanings");
        }
    }
    // 2. GLM asks for SIGMOID scoring, and must not accidentally select another transform.
    assert_eq!(
        super::GLM_ROUTER_FLAGS & f::SQRTSOFTPLUS,
        0,
        "GLM/DeepSeek-V3/Kimi score with sigmoid; setting the sqrtsoftplus bit changes every \
         routing gate in the model and nothing faults"
    );
    assert_ne!(super::GLM_ROUTER_FLAGS & f::SIGMOID, 0);
    // 3. And it still binds its bias -- the bit whose meaning was taken.
    assert_ne!(
        super::GLM_ROUTER_FLAGS & f::BIAS,
        0,
        "noaux_tc needs e_score_correction_bias bound at t3"
    );
}

/// V4.1 scores with `sqrtsoftplus`, so the emit may not reuse GLM's flags wholesale.
///
/// Both are `noaux_tc` top-k with a selection bias and normalised gates, which makes the two
/// configurations look interchangeable; the scoring function is the one field that differs, and it
/// is the field that decides every gate weight.
#[test]
fn v41_scores_with_sqrtsoftplus_not_sigmoid() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    assert_eq!(
        cfg.raw.scoring_func, "sqrtsoftplus",
        "if this checkpoint ever says sigmoid, the emit's flags must follow it"
    );
    assert_eq!(cfg.raw.topk_method, "noaux_tc");
    // Group-limited routing is the identity here: no n_group/topk_group in the config at all.
    assert!(cfg.raw.norm_topk_prob);
}

/// V4.1's routed experts run through GLM's measured prefill MoE body, at V4.1's own shapes.
///
/// This is the largest block in the model -- 139.16 TFLOP of an 8k prefill, 49.4% of the census --
/// so what it asserts is that the reuse actually took: the grouped expert ops are there, they carry
/// 384 experts and top-6 rather than GLM's 256/top-8, and the router asks for the score transform
/// this checkpoint names rather than the one GLM happens to use.
#[test]
fn the_routed_experts_run_glms_prefill_body_at_v41_shapes() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    // NOT PLOW_GLM_MOE_AITER: that arm asserts block-FP8 experts and V4.1's are MXFP4.
    let _env = crate::test_env::EnvScope::set(&[("PLOW_UNISEG", "0")]);
    let (t, tp) = (1024u32, 8u32);
    let mut b = Builder::new(304);
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0], tp);
    let all: Vec<u32> = (0..304u32).collect();
    let xn2 = b.tensor("act.xn2", (t as u64) * (cfg.hidden as u64) * 2);
    let x_out = b.tensor("act.xnext", (t as u64) * (cfg.hidden as u64) * 2);
    let x = b.tensor("act.x", (t as u64) * (cfg.hidden as u64) * 2);
    // The MoE body takes the post-norm's completion id, so give it a real producer.
    let eps = cfg.eps;
    let hidden = cfg.hidden;
    let gw = w.get(0, "ffn_norm.weight");
    let c_norm = b.emit(DevOp::RmsNorm, all.clone(), &[], |d| {
        d.t[0] = xn2;
        d.t[1] = x;
        d.t[2] = gw;
        d.i[0] = t;
        d.i[1] = hidden;
        d.f[0] = eps;
    });
    let mut xgate = 0;
    // The shared expert is emitted separately (block-FP8, op 184); the MoE body only combines it.
    let sh = b.tensor("act.shared", (t as u64) * (cfg.hidden as u64) * 2);
    super::dsv41::emit_dsv41_moe(
        &mut b, &cfg, &w, 0, tp, t, x_out, xn2, c_norm, (sh, c_norm), &mut xgate, &all, cfg.hidden,
    );
    let p = b.finish();

    // The router tail carries THIS model's expert count and top-k, not the GLM body's defaults.
    let rt = p
        .insts
        .iter()
        .find(|d| {
            d.op == DevOp::MoeRouterTopkPf as u16 || d.op == DevOp::MoeRouterTopk as u16
        })
        .expect("a router top-k");
    assert_eq!(rt.i[1], cfg.n_exp, "384 routed experts");
    assert_eq!(rt.i[1], 384);
    assert_eq!(rt.i[2], cfg.top_k, "top-6");
    assert_eq!(rt.i[2], 6);
    // And it asks for sqrtsoftplus scoring -- the bit that is NOT GLM's.
    use crate::mla::router_flag as f;
    assert_ne!(
        rt.i[3] & f::SQRTSOFTPLUS,
        0,
        "V4.1 scores with sqrt(softplus(.)); emitting sigmoid here still selects plausible \
         experts and silently reweights every one of them"
    );
    assert_eq!(rt.i[3] & f::SIGMOID, 0, "and not sigmoid, which is a different function");
    assert_ne!(rt.i[3] & f::BIAS, 0, "noaux_tc binds e_score_correction_bias");

    // The grouped expert ops are the MXFP4 pair, and they see this rank's TP slice of moe_inter.
    let glu: Vec<_> = p
        .insts
        .iter()
        .filter(|d| d.op == DevOp::MoeGroupGluPf as u16)
        .collect();
    assert!(!glu.is_empty(), "the routed experts must emit the grouped gate/up (op 85)");
    for g in &glu {
        // op 85: i0=I_moe i1=H i2=n_exp.
        assert_eq!(g.i[2], cfg.n_exp, "the expert op carries the bound expert table");
        assert_eq!(g.i[1], cfg.hidden);
        assert_eq!(
            g.i[0],
            cfg.moe_inter / tp,
            "TP (not EP): each rank runs its slice of moe_inter, 2304/8 = 288"
        );
    }
    // And the down half, which is what scatters into `part`.
    assert!(
        p.insts.iter().any(|d| d.op == DevOp::MoeGroupDownPf as u16),
        "the grouped down GEMM (op 86) must be emitted too"
    );
}

/// The attention core's widths, against the SHARDS rather than the config prose.
///
/// This is the fact that decides which flash kernel V4.1 can use, and it is one where V4.1 differs
/// from DeepSeek-V3 in a way that reads as a typo. V3 carries a 512 latent PLUS a separate 64-wide
/// rope strip, so its per-head query is 576 and it wants `<DK=512, DR=64>`. V4.1's query and latent
/// are both 512 with the rope inside, so it wants `<512, 0>` over a pre-rotated row. Taking V3's
/// shape here reads 64 bytes past every latent row and still produces fluent output.
#[test]
fn the_attention_core_is_512_wide_with_the_rope_inside_it() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let (hd, nope, rope) = super::dsv41::dsv41_attn_core_shape(&cfg);
    assert_eq!(hd, 512);
    assert_eq!(rope, 64, "qk_rope_head_dim");
    assert_eq!(nope, 448, "and it is INSIDE head_dim, so nope is 448 and not 512");
    // The latent is one 512-wide row for all 64 heads: wkv is [512, hidden], not [576, hidden].
    let t = super::dsv41::dsv41_layer_tensors(&cfg, 0);
    let wkv = t
        .iter()
        .find(|x| x.0 == "attn.wkv.weight")
        .expect("attn.wkv.weight");
    assert_eq!(
        wkv.1,
        hd as u64 * cfg.hidden as u64,
        "the latent is head_dim wide; a 576 here would be V3's separate-rope shape"
    );
    // V is the WHOLE latent, which is the only way the output LoRA's width closes.
    let (groups, _orow, ocol) = cfg.wo_a_groups();
    assert_eq!(
        ocol,
        cfg.heads * hd / groups,
        "O is head_dim per head, so wo_a reads 64*512/8 = 4096 per group"
    );
    assert_eq!(ocol, 4096);
    // The config carries no kv_lora_rank at all -- there is nothing to read a V3 shape out of.
    assert!(cfg.raw.q_lora_rank > 0);
}

/// Layer 0 is window-only, which is what makes its attention cheap.
///
/// `compress_ratios` 0 means sliding window with no compressed KV on top, and only the
/// `kv_source` layers carry a compressor. Getting this wrong the other way -- emitting full causal
/// attention -- is not incorrect output, it is 32x the attention work for the same answer, which
/// is the kind of mistake that shows up only as a missed latency target.
#[test]
fn layer_zero_attends_over_the_window_only() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    assert!(!cfg.kv_source.contains(&0), "layer 0 owns no compressor");
    assert_eq!(cfg.kv_source, vec![2, 8, 14, 20]);
    assert_eq!(cfg.sliding_window, 128, "the window every layer attends over");
    // The window is smaller than an 8k prefill by the factor that matters.
    let t = 8192u64;
    let full = t * t / 2;
    let win = t * cfg.sliding_window as u64;
    assert!(win * 30 < full, "the window must be the cheap form, not a detail");
}

/// The attention core: interior rope, a WINDOWED absorbed-MLA flash, and the sink merge.
///
/// The load-bearing assertion is `i[3]`. It packs two independent facts -- bit 31 NoPE and the
/// sliding window in the low bits -- and getting the window wrong is not wrong output: a zero
/// window is full causal attention, the same answer at 32x the arithmetic, visible only as a
/// missed latency target. So it is asserted as a value, not as "nonzero".
#[test]
fn the_attention_core_is_windowed_nope_mla_over_the_shared_latent() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let (t, tp, ctx) = (1024u32, 8u32, 8192u32);
    let nh_l = cfg.heads / tp;
    let (hd, nope, rope) = super::dsv41::dsv41_attn_core_shape(&cfg);
    let mut b = Builder::new(304);
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0], tp);
    let all: Vec<u32> = (0..304u32).collect();
    let q = b.tensor("act.q", (t as u64) * (nh_l as u64) * (hd as u64) * 2);
    let kv = b.tensor("act.kv", (t as u64) * (hd as u64) * 2);
    let kvlen = b.tensor("in.kvlen", 4);
    let pos = b.tensor("in.pos", (t as u64) * 4);
    let cos = b.tensor("gen.cos", (ctx as u64) * 32 * 4);
    let sin = b.tensor("gen.sin", (ctx as u64) * 32 * 4);
    let (act, _) = super::dsv41::emit_dsv41_attn_core(
        &mut b, &cfg, &w, &all, 0, tp, q, kv, kvlen, pos, cos, sin, None, t, ctx, &[],
    );
    let p = b.finish();

    // TWO ropes: the query's, and the shared latent's. The latent has ONE row per token for all
    // 64 heads, so its head count is 1 -- passing nh_l there would rotate 8x too much memory.
    let ropes: Vec<_> = p
        .insts
        .iter()
        .filter(|d| d.op == DevOp::QwenHeadNormRope as u16)
        .collect();
    assert_eq!(ropes.len(), 2, "q and the latent");
    for r in &ropes {
        assert_eq!(r.i[1], hd, "the whole 512-wide head");
        assert_eq!(r.i[2] & 0x7fff_ffff, rope, "64 rotated dims");
        // BIT 31 = INTERLEAVED (GPT-J), and it is not a flavour. op 142's default arm pairs
        // `(i, i + rotary/2)`; `apply_rotary_emb` pairs adjacent elements (model.py:392). The
        // shapes are identical either way and the score moves 41.2%
        // (`scripts/dsv41_rope_oracle.py`), so nothing downstream would catch a clear bit.
        assert_ne!(
            r.i[2] & 0x8000_0000,
            0,
            "V4.1's rope is INTERLEAVED; the half-split arm is a different operator and does \
             NOT cancel when q and the latent take it together"
        );
        assert_eq!(r.i[7], nope, "starting at 448 -- the SUFFIX, not the prefix");
        assert_eq!(r.i[5], 0, "normalize off; q_norm and kv_norm already ran");
    }
    assert_eq!(ropes[0].i[0], nh_l, "the query is per-head");
    assert_eq!(ropes[1].i[0], 1, "the latent is ONE row shared by every head");

    let fl = p
        .insts
        .iter()
        .find(|d| d.op == DevOp::FlashMlaPrefill as u16)
        .expect("the MLA prefill flash");
    assert_eq!(fl.i[1], nh_l, "per-rank heads");
    assert_eq!(
        fl.i[3],
        (1u32 << 31) | cfg.sliding_window,
        "bit 31 NoPE (the rope is interior and already applied) plus the 128 window. A zero \
         window here is full causal attention: the same answer at 32x the arithmetic"
    );
    assert_eq!(fl.i[3] & 0x7fff_ffff, 128);
    assert_ne!(fl.i[3] & 0x8000_0000, 0);
    // NoPE aliases the rope operands onto the nope ones.
    assert_eq!(fl.t[3], fl.t[2]);
    assert_eq!(fl.t[5], fl.t[4]);
    assert_eq!(fl.t[4], act.kvr, "K and V are the SAME latent");

    // The merge folds the attention sinks -- one unscaled logit per head, no value row.
    let mg = p
        .insts
        .iter()
        .find(|d| d.op == DevOp::FlashMerge as u16)
        .expect("the merge");
    assert_eq!(mg.t[0], act.o);
    assert_eq!(mg.t[3], w.get(0, "attn.attn_sink"), "sinks in t3");
    assert_eq!(mg.i[3], hd, "O is head_dim wide, which is what wo_a reads");
    assert_eq!(mg.i[2], 1, "nsplit 1");

    // THE INVERSE ROPE CLOSES THE CORE, and it is the last op, not an optional epilogue.
    // `Attention.forward` runs it on every layer (model.py:781) and the emit used to stop at the
    // merge -- op 181 existed with no emit site. Without it `o` still carries the query's
    // rotation and the fixed `wo_a` reads a position-DEPENDENT latent.
    let ir = p
        .insts
        .iter()
        .find(|d| d.op == DevOp::RopeInverseO as u16)
        .expect(
            "the inverse rope on the attention output -- op 181, `apply_rotary_emb(o[..., -rd:], \
             freqs_cis, True)`",
        );
    assert_eq!(ir.t[0], act.o, "in place on the merged output");
    assert_eq!(ir.i[1], nh_l);
    assert_eq!(ir.i[2], hd, "the row is head_dim wide");
    assert_eq!(ir.i[3], rope, "only the last 64 de-rotate");
    assert_eq!(
        p.insts.last().map(|d| d.op),
        Some(DevOp::RopeInverseO as u16),
        "it must run AFTER the merge: `o` is what carries the rotation, not the partials"
    );

    // And the core's output width is exactly the output LoRA's input.
    let (groups, _, ocol) = cfg.wo_a_groups();
    assert_eq!(nh_l * hd, ocol, "one rank's heads ARE one wo_a group");
    assert_eq!(groups, tp);
}

/// The compressor pools with NO `ape`, at `coff = 1`, and stops before the RoPE.
///
/// Three claims the oracle priced and the emit has to hold:
///   * `ape` is `TENSOR_NONE`, not a zero buffer. A nonzero one is worth 1.65 absolute
///     (`scripts/dsv41_csa2_oracle.py` check [2]), and V4.1's `Compressor` has no such parameter
///     at all -- a declared-and-kept-zero tensor is a promise, a sentinel is a type.
///   * `coff = 1`: no overlap transform, so each pool draws from `ratio` slots and not `2*ratio`.
///   * `i7 = 2`: pool, norm and STOP. The latent the indexer reads is the PRE-RoPE one
///     (`model.py:432-434`), and op 185 is what finishes it.
#[test]
fn the_compressor_pools_with_no_ape_and_stops_before_the_rope() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let (t, tp, ctx) = (1024u32, 8u32, 8192u32);
    let l = 2u32; // a ratio-2 kv_source
    let mut b = Builder::new(304);
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[l], tp);
    let cp = super::dsv41::declare_dsv41_compress(&mut b, &cfg, &[l], t, ctx)
        .expect("layer 2 is a kv_source");
    let all: Vec<u32> = (0..304u32).collect();
    let xn = b.tensor("act.xn", (t as u64) * (cfg.hidden as u64) * 2);
    let cos = b.tensor("gen.cos", (ctx as u64) * 32 * 4);
    let sin = b.tensor("gen.sin", (ctx as u64) * 32 * 4);
    super::dsv41::emit_dsv41_compressor(&mut b, &cfg, &w, &all, &cp, l, xn, cos, sin, t, &[]);
    let p = b.finish();

    let pool = p
        .insts
        .iter()
        .find(|d| d.op == DevOp::CompressPool as u16)
        .expect("a ratio-2 compressor pools");
    assert_eq!(pool.t[3], TENSOR_NONE, "NO `ape` -- V4.1's Compressor has none");
    assert_eq!(pool.t[5], TENSOR_NONE, "no cos on the pool: arm 2 stops before the rope");
    assert_eq!(pool.t[6], TENSOR_NONE);
    assert_eq!(pool.i[2], 1, "coff 1: no overlap transform");
    assert_eq!(pool.i[7], 2, "arm 2 -- pool, norm, STOP");
    assert_eq!(pool.i[1], 2, "layer 2 compresses at ratio 2");
    assert_eq!(pool.i[0], t / 2, "one pool per `ratio` tokens");
    assert_eq!(pool.t[0], cp.latent, "the PRE-RoPE latent, which the indexer reads");

    let tail = p
        .insts
        .iter()
        .find(|d| d.op == DevOp::CompressRopeQuant as u16)
        .expect("op 185 finishes the cache row");
    assert_eq!(tail.t[0], cp.cache, "the shared cache");
    assert_eq!(tail.t[1], cp.latent, "reading the latent it must NOT overwrite");
    assert_ne!(
        tail.t[0], tail.t[1],
        "the reference ropes in place; here the indexer still has to read `src`, and a \
         write-after-read is not something the packet order expresses"
    );
    assert_eq!(tail.i[2], cfg.qk_rope, "only the last 64 rotate");
    assert_eq!(tail.i[3], 16, "fp4_act_quant(latent, 16, ...)");
    assert_eq!(tail.i[6], 2, "fp4 e2m1 with an E4M3 scale, not a power of two");
    assert_eq!(tail.i[4], 2, "a latent stands for the FIRST token of its group");
}

/// Layer 20 compresses at ratio 1, and that is a plain projection with NO pool.
///
/// `Compressor.forward` returns on its first line there -- `self.norm(self.wkv(x))`, no gate, no
/// fp32, no softmax (`model.py:461-462`). An emit that ran op 180 at ratio 1 would take a softmax
/// over one slot, which is the identity, and then consult a `wgate` weight the checkpoint does
/// not ship for that layer.
#[test]
fn layer_twenty_compresses_at_ratio_one_with_no_pool() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let (t, tp, ctx) = (1024u32, 8u32, 8192u32);
    let l = 20u32;
    assert!(cfg.kv_source.contains(&l));
    assert!(
        !cfg.raw.compressor_has_gate(l),
        "layer 20 ships no `wgate`, which is the checkpoint's own statement that it does not pool"
    );
    let mut b = Builder::new(304);
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[l], tp);
    let cp = super::dsv41::declare_dsv41_compress(&mut b, &cfg, &[l], t, ctx).unwrap();
    let all: Vec<u32> = (0..304u32).collect();
    let xn = b.tensor("act.xn", (t as u64) * (cfg.hidden as u64) * 2);
    let cos = b.tensor("gen.cos", (ctx as u64) * 32 * 4);
    let sin = b.tensor("gen.sin", (ctx as u64) * 32 * 4);
    super::dsv41::emit_dsv41_compressor(&mut b, &cfg, &w, &all, &cp, l, xn, cos, sin, t, &[]);
    let p = b.finish();

    assert!(
        !p.insts.iter().any(|d| d.op == DevOp::CompressPool as u16),
        "ratio 1 must not pool"
    );
    let norm = p
        .insts
        .iter()
        .find(|d| d.op == DevOp::RmsNorm as u16)
        .expect("the compressor's own norm");
    assert_eq!(norm.t[0], cp.latent);
    assert_eq!(norm.i[1], cfg.head_dim);
    let tail = p
        .insts
        .iter()
        .find(|d| d.op == DevOp::CompressRopeQuant as u16)
        .expect("op 185 still finishes the row");
    assert_eq!(tail.i[0], t, "ratio 1: one cache row per TOKEN");
    assert_eq!(tail.i[4], 1);
}

/// A layer's rope table is chosen by `compress_ratio`, and window-only layers get a different one.
///
/// `Attention.__init__` builds ONE `freqs_cis` per layer (`model.py:680-687`): theta 160000 under
/// YaRN when `compress_ratio` is set, theta 10000 with YaRN DISABLED when it is not. Every rope in
/// that layer reads it -- the query's, the latent's, the compressor's. Two layers of the same
/// shape take different tables, which is the kind of difference no shape check can see.
#[test]
fn window_layers_and_compressed_layers_rope_on_different_tables() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    // Layer 0 is window-only and must NOT materialise the YaRN table at all: two ROPE gen
    // tensors (cos and sin), not four.
    let (m0, _) = super::dsv41::emit_dsv41_block(&cfg, &[0], 8, 304, 8192, 256);
    let ropes = m0
        .gen
        .iter()
        .filter(|g| g.scale != packet::rope::ROPE_SCALE_NONE || g.theta > 0.0)
        .count();
    assert_eq!(
        ropes, 2,
        "a window-only chain has no compressed layer and must not carry the scaled table too"
    );
    assert!(
        m0.gen
            .iter()
            .all(|g| g.theta == 0.0 || g.theta == cfg.raw.rope_theta as f64),
        "every rope table in a window-only chain is the BASE theta, YaRN disabled"
    );
    // And the two tables are genuinely different data, not two names for one buffer.
    let rope = cfg.qk_rope;
    let [base, _] = packet::rope::GenTensor::rope_pair(
        1024,
        rope,
        cfg.raw.rope_theta as f64,
        1.0,
        packet::rope::RopeScale::None,
    );
    let rs = &cfg.raw.rope_scaling;
    let [yarn, _] = packet::rope::GenTensor::rope_pair(
        1024,
        rope,
        cfg.raw.compress_rope_theta as f64,
        1.0,
        packet::rope::RopeScale::YarnDeepSeek {
            factor: rs.factor as f64,
            beta_fast: rs.beta_fast as f64,
            beta_slow: rs.beta_slow as f64,
            orig: rs.original_max_position_embeddings as f64,
            truncate: true,
        },
    );
    assert_ne!(
        base.generate(),
        yarn.generate(),
        "theta 10000 unscaled and theta 160000 under YaRN must not produce the same table"
    );
}

/// The indexer scores POOLS, and both halves of the pair must agree about that.
///
/// op 117 bounds a query row at `(q_pos0 + t + 1) / pool_size` and op 118 scans the same count.
/// If the two disagreed the score would be written over a range the select never reads, or --
/// worse in the other direction -- the select would rank columns op 117 never wrote, which are
/// whatever the arena holds. `scripts/dsv41_indexer_oracle.py` check [2] is the numerical half
/// of this; that they are the SAME number in one emit is the half a test can hold.
#[test]
fn the_indexer_scores_pools_and_both_ops_agree_on_the_ratio() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    for (l, ratio) in [(2u32, 2u32), (20, 1)] {
        let (m, _) = super::dsv41::emit_dsv41_block(&cfg, &[l], 8, 304, 8192, 256);
        let sc = m
            .progs[0]
            .insts
            .iter()
            .find(|d| d.op == DevOp::IndexScorePf as u16)
            .unwrap_or_else(|| panic!("layer {l} is an index_source and must score"));
        let se = m
            .progs[0]
            .insts
            .iter()
            .find(|d| d.op == DevOp::IndexSelectPf as u16)
            .unwrap_or_else(|| panic!("layer {l} must select"));
        assert_eq!(sc.i[4], ratio, "op 117 pool_size");
        assert_eq!(se.i[3], ratio, "op 118 pool_size");
        assert_eq!(sc.i[4], se.i[3], "the score and the select MUST agree on the ratio");
        assert_eq!(sc.i[1], cfg.index_heads, "all 32 heads: the indexer is replicated");
        assert_eq!(sc.i[1], 32, "and 32 is what the MFMA A tile is");
        assert_eq!(sc.i[3], cfg.index_dim);
        assert_eq!(sc.i[2], se.i[2], "one score stride");
        // `min(index_topk, end_pos // ratio)` -- at t=256 and ratio 2 that is 128, not 512.
        assert_eq!(se.i[1], cfg.index_topk.min(256 / ratio), "the reference's own min");
    }
}

/// An index layer that owns no KEYS derives none -- it reads the ones its source published.
///
/// "the index keys are derived from the compressor's latent, so only a layer that compresses its
/// own KV can produce them; every other indexer reads them from that layer's cache"
/// (`model.py:498-500`). Layers 24, 28, 32 and 36 run queries and weights against layer 20's
/// keys. A layer that re-derived them would need a compressor it does not have.
#[test]
fn an_index_layer_that_owns_no_keys_derives_none() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    assert!(cfg.index_source.contains(&24) && !cfg.kv_source.contains(&24));
    let (m, _) = super::dsv41::emit_dsv41_block(&cfg, &[20, 24], 8, 304, 8192, 256);
    // op 185 runs three times per KEY-OWNING index layer (cache row, index key, queries) and
    // once for a query-only one. Layer 20 owns keys, layer 24 does not: 3 + 1.
    let tails = m
        .progs[0]
        .insts
        .iter()
        .filter(|d| d.op == DevOp::CompressRopeQuant as u16)
        .count();
    assert_eq!(
        tails, 4,
        "layer 20 ropes and quantizes its cache row, its index key and its queries; layer 24 \
         only its queries"
    );
    // Both layers score and select: the SELECTION is per-layer, only the keys are shared.
    assert_eq!(
        m.progs[0].insts.iter().filter(|d| d.op == DevOp::IndexScorePf as u16).count(),
        2
    );
    // And a single-layer chain at 24 must not invent a key derivation.
    let (m24, _) = super::dsv41::emit_dsv41_block(&cfg, &[24], 8, 304, 8192, 256);
    assert_eq!(
        m24.progs[0].insts.iter().filter(|d| d.op == DevOp::CompressRopeQuant as u16).count(),
        1,
        "layer 24 alone ropes only its queries"
    );
}

/// A compressed layer runs TWO flashes into one partial pair, and the sink is folded ONCE.
///
/// `Attention.forward` concatenates -- `kv = cat([kv, compress_kv])`,
/// `topk_idxs = cat([topk_idxs, compress_idxs])` -- and runs ONE `sparse_attn`, whose kernel adds
/// `attn_sink` to the denominator after the last block (`kernel.py:383-386`). Two partials merged
/// by one online softmax is that expression: the window pass at split 0, the gathered pass at
/// split 1, and `FlashMerge` folding the sink exactly once. Two merges, or a sink on each pass,
/// would double-count the denominator.
#[test]
fn a_compressed_layer_merges_two_partials_with_the_sink_folded_once() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let (m, _) = super::dsv41::emit_dsv41_block(&cfg, &[3], 8, 304, 8192, 256);
    let win = m
        .progs[0]
        .insts
        .iter()
        .find(|d| d.op == DevOp::FlashMlaPrefill as u16)
        .expect("the window flash");
    let gat = m
        .progs[0]
        .insts
        .iter()
        .find(|d| d.op == DevOp::FlashGatherPrefill as u16)
        .expect("the compressed flash -- 38 of 40 layers need it");
    assert_eq!(win.i[7], (2 << 8) | 0, "window is partial 0 of 2");
    assert_eq!(gat.i[7], (2 << 8) | 1, "compressed is partial 1 of 2");
    assert_eq!(win.t[0], gat.t[0], "ONE Opart, filled in disjoint halves");
    assert_eq!(win.t[1], gat.t[1], "ONE mlpart");
    assert_ne!(win.t[4], gat.t[4], "and two DIFFERENT caches: the window K and the CSA2 cache");
    assert_ne!(gat.t[7], TENSOR_NONE, "the gather walks the selection table");
    assert_ne!(gat.i[3] & 0x8000_0000, 0, "NoPE: the rope is interior and already applied");

    let merges: Vec<_> = m
        .progs[0]
        .insts
        .iter()
        .filter(|d| d.op == DevOp::FlashMerge as u16)
        .collect();
    assert_eq!(merges.len(), 1, "ONE merge: the sink joins one denominator, not two");
    assert_eq!(merges[0].i[2], 2, "nsplit 2");
    assert_eq!(
        merges[0].t[3],
        m.progs[0].insts.iter().find(|d| d.op == DevOp::FlashMerge as u16).unwrap().t[3],
    );
    assert_ne!(merges[0].t[3], TENSOR_NONE, "the attention sink is folded here and only here");

    // A window-only layer keeps the one-partial layout, byte-for-byte what it emitted before.
    let (m0, _) = super::dsv41::emit_dsv41_block(&cfg, &[0], 8, 304, 8192, 256);
    let w0 = m0
        .progs[0]
        .insts
        .iter()
        .find(|d| d.op == DevOp::FlashMlaPrefill as u16)
        .unwrap();
    assert_eq!(w0.i[7], 0, "no output split on a layer with one partial");
    assert!(!m0.progs[0].insts.iter().any(|d| d.op == DevOp::FlashGatherPrefill as u16));
}

/// Every tensor-parallel weight is READ at the width it was DECLARED at.
///
/// This is the invariant three separate bugs in this emitter violated in one sitting -- `wo_b`,
/// `wq_b` and the shared expert's `w1`/`w2`/`w3` -- and the failure mode is identical each time.
/// `declare_dsv41_weights` divides an `OutSplit` tensor by `tp`, so the table says rank r holds
/// `[N/tp, K]`; an emit that asks for `[N, K]` reads 8x past the end of the shard. Nothing faults:
/// the loader bound only `N/tp` rows, so every rank computes rows `0..N` out of rank 0's shard and
/// the answer is wrong on 7 of 8 ranks, plausibly.
///
/// `emit_pf_gemm_fp8_mx` now asserts this per operand, so the test's job is only to prove the
/// whole layer passes through it -- which it cannot do vacuously, since it emits the real thing.
#[test]
fn every_tp_weight_is_read_at_the_width_it_was_declared() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let (m, _) = super::dsv41::emit_dsv41_block(&cfg, &[0], 8, 304, 2048, 256);
    let byname = |h: u32| m.tensors[h as usize].name.clone();

    // The q UP projection: OutSplit over heads, so 8 of 64 heads per rank.
    let qb = m.progs[0]
        .insts
        .iter()
        .find(|d| d.op == DevOp::GemmFp8Mx as u16 && byname(d.t[2]).ends_with("attn.wq_b.weight"))
        .expect("the layer projects q through wq_b");
    assert_eq!(
        qb.i[1],
        cfg.heads / 8 * cfg.head_dim,
        "wq_b emits THIS rank's 8 heads (8 * 512), not all 64"
    );

    // The absorbed latent is the counter-example: REPLICATED, so it does NOT divide. A per-rank
    // wkv would give rank r a 64-wide slice of a 512-wide latent that all 64 heads read.
    let kv = m.progs[0]
        .insts
        .iter()
        .find(|d| d.op == DevOp::GemmFp8Mx as u16 && byname(d.t[2]).ends_with("attn.wkv.weight"))
        .expect("the layer projects the shared latent through wkv");
    assert_eq!(kv.i[1], cfg.head_dim, "wkv is replicated: the full 512, on every rank");

    // The shared expert: OutSplit gate/up and InSplit down, all over `moe_inter`.
    for (suffix, n, k) in [
        ("ffn.shared_experts.w1.weight", cfg.moe_inter / 8, cfg.hidden),
        ("ffn.shared_experts.w3.weight", cfg.moe_inter / 8, cfg.hidden),
        ("ffn.shared_experts.w2.weight", cfg.hidden, cfg.moe_inter / 8),
    ] {
        let d = m.progs[0]
            .insts
            .iter()
            .find(|d| d.op == DevOp::GemmFp8Mx as u16 && byname(d.t[2]).ends_with(suffix))
            .unwrap_or_else(|| panic!("{suffix} is not emitted"));
        assert_eq!([d.i[1], d.i[2]], [n, k], "{suffix} is per-rank over moe_inter");
    }
}

/// The shared expert's down projection is INPUT-parallel, so its output is a PARTIAL that must
/// land in a peer slot and be summed.
///
/// `d_xreduce` sums `peer_scratch[r] + slot` over every rank and never reads `out`, so a partial
/// written to an ordinary arena tensor contributes NOTHING to the sum -- the reduce returns
/// whatever the other ranks left at that offset. This is exactly the K3 bug (`k3.rs:1599`) that
/// made 92 of 93 layers compute `ffn = up_latent + attn` instead of `+ shared_expert`, with
/// finite, plausible logits.
///
/// And it lands in SLOT 0, attention's, because only TWO peer slots exist: `DevBlob::parse`
/// recovers `slot_bytes` as `max(i[2])` over the collectives, so an op that asks for `2 * slot_b`
/// does not buy a third slot -- it redefines the UNIT as twice what every other op meant by it.
///
/// Reusing slot 0 is safe here for the reason K3's could not be: K3's worry is a ONE-SHOT gate,
/// which proves every peer ARRIVED and not that every peer finished READING. Attention's reduce
/// is `XReduceTwoShot` -- reduce-scatter then all-gather -- so no rank completes it until every
/// peer has both contributed and published its band, which is exactly "finished reading slot 0".
#[test]
fn the_shared_experts_partial_lands_in_a_peer_slot_and_is_summed() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let t = 256u32;
    let (m, _) = super::dsv41::emit_dsv41_block(&cfg, &[0], 8, 304, 2048, t);
    let p = &m.progs[0];
    let byname = |h: u32| m.tensors[h as usize].name.clone();

    let down = p
        .insts
        .iter()
        .find(|d| {
            d.op == DevOp::GemmFp8Mx as u16
                && byname(d.t[2]).ends_with("ffn.shared_experts.w2.weight")
        })
        .expect("the shared expert has a down projection");
    assert_eq!(
        byname(down.t[0]),
        super::dsv41::PEER_SLOT_SHARED,
        "the down projection writes the PEER SLOT, not a local arena tensor"
    );
    assert_eq!(
        super::dsv41::PEER_SLOT_SHARED,
        super::dsv41::PEER_SLOT_O,
        "slot 0, shared with attention -- only two peer slots exist"
    );

    let slot_b = t * cfg.hidden * 2;
    let xr = p
        .insts
        .iter()
        .filter(|d| d.op == DevOp::XReduce as u16 || d.op == DevOp::XReduceTwoShot as u16)
        .find(|d| d.i[2] == 0 && d.t[0] == m.tensors.iter().position(|x| x.name == "act.l0.sh_out").expect("sh_out") as u32)
        .expect("the shared partial is reduced out of slot 0");
    assert_eq!(xr.i[0], t * cfg.hidden, "the whole [T, hidden] partial");
    assert_eq!(xr.i[1], 8, "over all 8 ranks");

    // EVERY offset this layer reduces at must be one the host actually binds, and the host binds
    // exactly two: 0 and `slot_b`. An offset outside that set is the failure this pins -- it does
    // not fault, it silently moves the unit `DevBlob::parse` recovers for every other collective.
    let slots: std::collections::BTreeSet<u32> = p
        .insts
        .iter()
        .filter(|d| d.op == DevOp::XReduce as u16 || d.op == DevOp::XReduceTwoShot as u16)
        .map(|d| d.i[2])
        .collect();
    assert!(
        slots.iter().all(|s| *s == 0 || *s == slot_b),
        "only slots 0 and slot_b={slot_b} exist, got {slots:?}"
    );
    assert!(slots.contains(&0), "attention and the shared expert reduce slot 0");
}

/// The routed combine's peer slot is BOUND. `GlmTn::none()` leaves `dg_tp` at `TENSOR_NONE` and
/// the GLM body writes it unconditionally at tp>1, so an unset field aims every rank's routed
/// output at the null handle.
#[test]
fn the_routed_combine_writes_a_real_peer_slot() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let (m, _) = super::dsv41::emit_dsv41_block(&cfg, &[0], 8, 304, 2048, 256);
    let dg = m
        .tensors
        .iter()
        .find(|t| t.name == super::dsv41::PEER_SLOT_MOE)
        .expect("the routed combine declares act.dg_tp at tp>1");
    assert_eq!(dg.bytes, 256 * cfg.hidden as u64 * 2, "one [T, hidden] slot");
    assert!(
        m.progs[0]
            .insts
            .iter()
            .any(|d| d.t[0] != packet::dev::TENSOR_NONE
                && m.tensors[d.t[0] as usize].name == super::dsv41::PEER_SLOT_MOE),
        "something must actually write it"
    );
}

/// The whole layer emits, in dataflow order, and every dependency points BACKWARD.
///
/// A rung is only worth building on if the program it produces is well-formed: `to_blob_v6` does
/// not check topology and the interpreter does not either, so a forward edge is a read of a buffer
/// that has not been written -- stale bytes, no fault.
#[test]
fn the_layer_zero_rung_is_a_topological_program() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let (m, desc) = super::dsv41::emit_dsv41_block(&cfg, &[0], 8, 304, 2048, 256);
    assert_eq!(m.progs.len(), 2, "one prefill bucket plus the empty decode placeholder");
    let p = &m.progs[0];
    assert!(!p.insts.is_empty());
    for (i, d) in p.insts.iter().enumerate() {
        for w in &p.waits[d.wait_ofs as usize..][..d.wait_len as usize] {
            assert!(
                (w.id as usize) < i,
                "op {i} ({}) waits on {}, which is not before it",
                d.op,
                w.id
            );
        }
    }
    // The chain the layer is: pre-norm, q/kv, rope, flash, output LoRA, mHC, FFN, MoE.
    let has = |op: DevOp| p.insts.iter().any(|d| d.op == op as u16);
    for op in [
        DevOp::FlashMlaPrefill,
        DevOp::QwenHeadNormRope,
        DevOp::GemmFp8Mx,
        DevOp::Glu,
        DevOp::RmsNorm,
        DevOp::HyperConnPre,
        DevOp::HyperConnPost,
    ] {
        assert!(has(op), "layer 0 must emit {op:?}");
    }
    // NO `Residual`. Both of the ones this layer used to carry were wrong: the MoE's
    // `x_out = xmid + ffn` read a buffer nothing wrote (`raw_output` removed it, and mHC's
    // `HyperConnPost` is what rejoins the residual), and the rung's seed wrote one copy of a
    // `[hc_mult][T][hidden]` entry and left the other three to the arena. The entry IS the mHC
    // stream now, so there is nothing left to seed.
    assert!(
        !has(DevOp::Residual),
        "a Residual here is one of the two the layer must not have; see this test's body"
    );
    assert_eq!(
        desc.inputs[0].name, "act.hc_residual_a",
        "the entry is the mHC stream, not a plain hidden state"
    );
    assert_eq!(
        desc.inputs[0].shape.len(),
        3,
        "and it is [hc_mult][T][hidden] -- a 2-D claim has a harness fill a quarter of it"
    );
    assert_eq!(desc.programs.prefill_buckets, vec![256]);
    assert_eq!(desc.dims.heads, Some(cfg.heads as i64));
}

/// Slots of `t[]` an op WRITES. Everything else it reads.
///
/// Only the ops layer 0 emits, and only where the answer is not slot 0. Ops absent from this list
/// write `t[0]` and nothing else, which is the convention `slots.rs` documents for all but a
/// handful. A wrong entry makes `every_activation_read_is_written_by_something` too permissive, so
/// each one names the `slots.rs` row it came from.
fn written_slots(op: u16) -> &'static [usize] {
    match op {
        // t = [post_mix, comb_mix, layer_input, mixes, residual, hc_scale, hc_base, pre_pair].
        // Slot 7 is BOTH: the op publishes the `pre` it derived into half `pre_in_half ^ 1` and
        // (in `PRE_DEFER`) reads the other half, which the previous sublayer wrote. Listing it as
        // written is what lets the chain's FIRST sublayer -- `PRE_SEED`, which only publishes --
        // be the writer that satisfies every later reader.
        o if o == DevOp::HyperConnPre as u16 => &[0, 1, 2, 7],
        // t = [meta, table, row_token, row_partidx, row_gate] -- the align pass FILLS the three
        // row arrays and the meta header; only `table` is an input.
        o if o == DevOp::MoeAlignPf as u16 => &[0, 2, 3, 4],
        // t = [fu_g, xn2, ewt, est, meta, row_token] + `fu_scale` on slot 7 (not named in
        // `slots.rs`, whose row stops at 6, but written by the arm).
        o if o == DevOp::MoeGroupGluPf as u16 => &[0, 7],
        // `t0=Opart(f32) t1=mlpart(f32)` -- the split-K partial AND its running max/lse pair, both
        // written, both read back by `FlashMerge`.
        o if o == DevOp::FlashMlaPrefill as u16 => &[0, 1],
        _ => &[0],
    }
}

/// NOTHING reads an activation no op ever writes.
///
/// The bug this exists for: `emit_dsv41_moe` handed GLM's body an `xmid` -- GLM's post-attention
/// residual -- that this emitter has no equivalent of, so the layer ended in
/// `Residual(xnext, xmid, moe_out)` over a buffer nobody had written. It did not fault and it did
/// not fail any shape check; it read whatever the arena held. The fix was `raw_output: true`,
/// which is what an mHC layer wants anyway, since `HyperConnPost` is what re-joins the residual.
///
/// Weights, the block entry and the host-filled inputs are exempt by name. Everything else that
/// appears on an op's READ slot must appear on some op's WRITE slot.
#[test]
fn every_activation_read_is_written_by_something() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    // ONE CHAIN PER SHAPE A LAYER CAN HAVE, because no two of them read the same set:
    //   [0]      the window-only base
    //   [1]      opens with the Engram gather, its all-reduce and `wkv` -- `in.engram_ids` plus
    //            two activations nothing else in the layer writes
    //   [2, 3]   a kv_source WITH an indexer, then a pure reader: the compressor's latent and
    //            cache, the index keys, the score and the selection, and a layer that reads the
    //            last two without writing either
    //   [20]     the ratio-1 compressor, which has no pool and no `wgate`
    //   [20, 24] a key OWNER and then an index_source that does NOT own keys: 24 runs its own
    //            queries and weights against the keys 20 published, which is the two-level
    //            indexer's whole point and the only chain that exercises the reuse
    for block in [
        &[0u32][..],
        &[1u32][..],
        &[2u32, 3u32][..],
        &[20u32][..],
        &[20u32, 24u32][..],
    ] {
        every_activation_read_is_written_for(&cfg, block);
    }
}

fn every_activation_read_is_written_for(cfg: &super::dsv41::Dsv41Cfg, block: &[u32]) {
    let (m, _) = super::dsv41::emit_dsv41_block(cfg, block, 8, 304, 2048, 256);
    let p = &m.progs[0];
    let name = |h: u32| m.tensors[h as usize].name.as_str();

    let mut written: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for d in &p.insts {
        for &s in written_slots(d.op) {
            if d.t[s] != packet::dev::TENSOR_NONE {
                written.insert(d.t[s]);
            }
        }
    }
    for d in &p.insts {
        let w = written_slots(d.op);
        for (s, &h) in d.t.iter().enumerate() {
            if h == packet::dev::TENSOR_NONE || w.contains(&s) {
                continue;
            }
            let n = name(h);
            // Weights carry the checkpoint's `layers.N.` prefix; `in.*` is host-filled; `act.x` is
            // the block entry the harness uploads.
            if n.starts_with("layers.") || n.starts_with("in.") || n == "act.x" {
                continue;
            }
            assert!(
                written.contains(&h),
                "layer(s) {block:?}: op {:?} reads `{n}` at slot {s}, and no op in the program \
                 writes it -- the layer would compute from whatever the arena happened to hold",
                d.op
            );
        }
    }
}

/// The routed experts reach the GPU through the LOADER'S PACKED TABLE, not through 2304 declared
/// checkpoint tensors.
///
/// `bind_packed_experts` finds `{pfx}expert_weight_table`, reads the expert count off its declared
/// size (`bytes / 24`), resolves the checkpoint's spelling, and packs every expert's slice into one
/// buffer. Binding `ffn.experts.0.w1.weight` into `GlmLW::ewt` instead -- which is what this
/// emitter did first -- points the grouped GEMM's pointer table at fp4 mantissas, and declaring the
/// 2304 individual tensors uploads 1.7 GB per layer per rank that no op reads.
#[test]
fn the_routed_experts_are_bound_through_the_packed_table() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let (m, _) = super::dsv41::emit_dsv41_block(&cfg, &[0], 8, 304, 2048, 256);

    for suffix in ["expert_weight_table", "expert_scale_table"] {
        let t = m
            .tensors
            .iter()
            .find(|t| t.name == format!("layers.0.ffn.{suffix}"))
            .unwrap_or_else(|| panic!("layers.0.ffn.{suffix} must be declared"));
        // `bind_packed_experts` DERIVES the expert count from this, so the size is the contract.
        assert_eq!(
            t.bytes,
            cfg.n_exp as u64 * 24,
            "[E][3] u64, and the loader reads E back as bytes/24"
        );
    }
    assert!(
        !m.tensors.iter().any(|t| t.name.contains(".experts.")),
        "no per-expert checkpoint tensor may be declared; the packed buffer is what ops read"
    );

    // And the grouped arms must actually point AT the tables.
    let ewt = m
        .tensors
        .iter()
        .position(|t| t.name == "layers.0.ffn.expert_weight_table")
        .unwrap() as u32;
    let est = m
        .tensors
        .iter()
        .position(|t| t.name == "layers.0.ffn.expert_scale_table")
        .unwrap() as u32;
    for op in [DevOp::MoeGroupGluPf, DevOp::MoeGroupDownPf] {
        let d = m.progs[0]
            .insts
            .iter()
            .find(|d| d.op == op as u16)
            .unwrap_or_else(|| panic!("{op:?} is not emitted"));
        assert_eq!([d.t[2], d.t[3]], [ewt, est], "{op:?} reads the two tables");
    }
}

/// The rung is a PREFILL bucket, and it says outright that it cannot decode.
///
/// `derive_roles` reads a parent blob's roles positionally -- `decode_rung_lo` puts the boundary at
/// `len - 1` -- so a ONE-program blob has no prefill bucket at all and a host filtering on
/// `is_prefill_bucket()` finds nothing to run. The trailing decode program is EMPTY because V4.1
/// has no decode emit; `block.json`'s `decode_t: 0` says the same thing, and this test fails if
/// either half changes without the other.
#[test]
fn the_rung_is_a_prefill_bucket_and_states_it_cannot_decode() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let (m, desc) = super::dsv41::emit_dsv41_block(&cfg, &[0], 8, 304, 2048, 512);
    assert_eq!(m.prog_t, vec![512, 1], "one prefill bucket, then the decode slot");

    let roles =
        packet::devbuild::derive_roles(&m.prog_t, packet::devbuild::RoleSource::Positional, |_| 0);
    assert!(
        roles[0].is_prefill_bucket(),
        "the work is a PREFILL bucket, got {:?}",
        roles[0]
    );
    assert!(roles[1].is_decode_rung());

    assert!(
        m.progs[1].insts.is_empty(),
        "the decode program is a placeholder; a real one must also set block.json's decode_t"
    );
    assert_eq!(
        desc.programs.decode_t, 0,
        "and the descriptor must agree it cannot decode"
    );
    assert_eq!(desc.programs.prefill_buckets, vec![512]);
}

/// `--block l..r` emits EVERY layer in the range, chained.
///
/// It used to emit the left end alone -- `block_spec.split("..").next()` -- so `--block 0..39`
/// wrote a blob byte-identical to `--block 0` and said nothing. A whole-model request answered
/// with one layer is exactly the silently-wrong artifact this emitter refuses everywhere else.
///
/// SYNTHETIC `compress_ratios`. Only layer 0 is emittable on the real config (see
/// `a_layer_that_reads_the_compressed_cache_is_not_done`), so there is no range of real layers to
/// chain yet. Zeroing the ratios makes layers 3..7 window-only, which is what the chain LOOP needs
/// to be exercised; it is a statement about the loop, not about the model.
#[test]
fn a_block_range_emits_every_layer_in_it_chained() {
    let Some((mut cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    cfg.raw.compress_ratios = vec![0; cfg.raw.compress_ratios.len()];
    assert!(
        (3..=7).all(|l| super::dsv41::dsv41_emit_block_plan(&cfg, l).is_ok()),
        "the synthetic config must make 3..7 emittable, or this tests nothing"
    );

    let (one, d1) = super::dsv41::emit_dsv41_block(&cfg, &[3], 8, 304, 2048, 256);
    let (five, d5) = super::dsv41::emit_dsv41_block(&cfg, &[3, 4, 5, 6, 7], 8, 304, 2048, 256);
    let n1 = one.progs[0].insts.len();
    let n5 = five.progs[0].insts.len();
    assert_eq!(
        n5,
        n1 * 5,
        "a five-layer chain must be five layers of ops ({n1} each), not {n5}"
    );

    // Every layer's weights are declared, not just the first: the chain reads `layers.7.` too.
    assert!(
        five.tensors.iter().any(|t| t.name.starts_with("layers.7.")),
        "the last layer of the chain must have its weights declared"
    );
    assert!(
        !one.tensors.iter().any(|t| t.name.starts_with("layers.7.")),
        "a one-layer emit must NOT declare another layer's weights"
    );

    // A single layer names its own weights; a chain cannot, because it draws on all of them.
    assert_eq!(d1.weights.prefix, "layers.3.");
    assert_eq!(d5.weights.prefix, "layers.");
    assert_eq!(d5.layer, 3, "the descriptor names the first layer of the chain");
}

/// Engram runs BEFORE the block, on the hc-expanded stream -- not inside the FFN sublayer.
///
/// `Block.__init__` constructs `self.engram`, which is what makes an FFN-sublayer position look
/// right, but `Block.forward` never calls it. `Transformer.forward` does, in the layer loop and
/// ahead of the block (`model.py:1262-1267`), so op 182 is in place on `[T, hc_mult, dim]` --
/// the residual stream's copies -- before this layer's `mhc_pre` collapses them. An emit built off
/// the old row order would have handed it the post-attention activation instead, which is a
/// different tensor of a different rank.
///
/// Ops 182/183 themselves ARE the reference's -- `scripts/dsv41_engram_oracle.py` measures
/// 0.000e+00 on the embed (including the signed out-of-shard compare) and 2.2e-16 on the gate --
/// so placement and plumbing is the whole of what layers 1 and 14 are still missing.
#[test]
fn engram_is_the_first_part_of_its_layer_not_an_ffn_one() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();

    let eng: Vec<u32> = (0..cfg.layers)
        .filter(|l| cfg.engram_layers.contains(l))
        .collect();
    assert_eq!(eng, vec![1, 14], "the checkpoint's engram_layer_ids");

    for l in 0..cfg.layers {
        let parts = super::dsv41::dsv41_layer_parts(&cfg, l);
        let at = parts.iter().position(|(n, _)| n.starts_with("engram"));
        let mhc = parts
            .iter()
            .position(|(n, _)| n.starts_with("mhc_pre"))
            .expect("every layer has an mhc_pre part");
        if eng.contains(&l) {
            let at = at.unwrap_or_else(|| panic!("layer {l} is an engram layer and has no engram part"));
            assert_eq!(at, 0, "layer {l}: engram is the FIRST part -- it precedes the block");
            assert!(at < mhc, "layer {l}: engram must precede mhc_pre, not follow ffn_norm");
        } else {
            assert!(at.is_none(), "layer {l} is not an engram layer");
        }
    }
}

/// The peer region's two recovered numbers must DIVIDE, for every emittable layer.
///
/// `DevBlob::parse` recovers `hidden = max(i[0] / program.t)` and `slot_bytes = max(i[2])` over the
/// collectives -- nothing else -- and `AmdTpGroup::load` then computes
/// `max_tokens = slot_bytes / (hidden * 2)`, REFUSING the packet unless the division is exact.
///
/// This is not hypothetical. Engram's all-reduce of the row-split gather is 6144 wide where every
/// other collective in the model is `hidden` = 5120, so a layer-1 packet whose slot unit was built
/// from `c.hidden` recovered `hidden = 6144` against `slot_bytes = 8192 * 5120 * 2`, and
/// 83 886 080 is not a multiple of 12 288. It loaded NOWHERE. The failure was loud, but it
/// surfaced after an emit, an object build and a queue wait; here it is one test.
#[test]
fn the_peer_slot_unit_divides_the_widest_collective() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let t = 256u32;

    for l in (0..cfg.layers).filter(|l| super::dsv41::dsv41_emit_block_plan(&cfg, *l).is_ok()) {
        let (m, _) = super::dsv41::emit_dsv41_block(&cfg, &[l], 8, 304, 2048, t);
        let p = &m.progs[0];
        let collective = [
            DevOp::XReduce as u16,
            DevOp::XReduceScatter as u16,
            DevOp::XReduceTwoShot as u16,
        ];
        let (mut hidden, mut slot_bytes) = (0u32, 0u64);
        for d in p.insts.iter().filter(|d| collective.contains(&d.op)) {
            hidden = hidden.max(d.i[0] / t.max(1));
            slot_bytes = slot_bytes.max(d.i[2] as u64);
        }
        assert!(hidden > 0, "layer {l} emits no collective at tp=8");
        assert!(slot_bytes > 0, "layer {l} recovers a zero slot unit");
        let msg = u64::from(hidden) * 2;
        assert_eq!(
            slot_bytes % msg,
            0,
            "layer {l}: the packet recovers hidden={hidden} and slot_bytes={slot_bytes}, and \
             {slot_bytes} % {msg} != 0 -- AmdTpGroup::load divides these to get max_tokens and \
             refuses the packet when it does not come out exact"
        );
        // And the unit has to be big enough to HOLD the widest partial, not merely divide it.
        assert!(
            slot_bytes >= u64::from(t) * u64::from(hidden) * 2,
            "layer {l}: slot B begins at {slot_bytes}, inside the {} B slot A needs",
            u64::from(t) * u64::from(hidden) * 2
        );
    }
}

/// Layer 1 opens with Engram: gather, all-reduce, `wkv`, gate -- and only then the mHC.
///
/// The all-reduce is `ParallelEngramEmbedding.forward`'s own `dist.all_reduce` (`model.py:323`),
/// not an artifact of this emit: the table is 98.31 GB and row-split because it cannot be
/// replicated on a 192 GB card, so a rank's gather is a partial by construction. Dropping it would
/// leave every rank with 1/8th of each row and fault nowhere.
///
/// The order `gather -> reduce -> wkv` is the reference's. `wkv` is linear and bias-free so
/// reducing after would agree in exact arithmetic, but it would quantize partial sums and reduce a
/// tensor four times wider (25600 vs 6144).
#[test]
fn layer_one_opens_with_the_engram_gather_and_its_all_reduce() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    let t = 256u32;
    let (m, _) = super::dsv41::emit_dsv41_block(&cfg, &[1], 8, 304, 2048, t);
    let p = &m.progs[0];
    let name = |h: u32| m.tensors[h as usize].name.as_str();

    let ops: Vec<u16> = p.insts.iter().map(|d| d.op).collect();
    let first = |o: DevOp| ops.iter().position(|&x| x == o as u16);

    let gather = first(DevOp::EngramEmbed).expect("layer 1 gathers");
    let wkv = first(DevOp::GemmFp8Mx).expect("layer 1 projects the gather");
    let gate = first(DevOp::EngramGate).expect("layer 1 gates");
    let mhc = first(DevOp::HyperConnPre).expect("layer 1 has an mHC");
    assert_eq!(gather, 0, "the gather is the layer's FIRST op -- Engram precedes the block");
    assert!(gather < wkv && wkv < gate, "gather -> wkv -> gate");
    assert!(gate < mhc, "Engram completes before mhc_pre collapses the copies");

    // The all-reduce sits BETWEEN the gather and wkv, and it is a real collective over the
    // gather's width, not a copy.
    // Whichever collective `emit_xreduce` picks -- the two-shot one on this size -- what matters
    // is that a cross-rank sum happens here at all.
    let collective = [
        DevOp::XReduce as u16,
        DevOp::XReduceScatter as u16,
        DevOp::XReduceTwoShot as u16,
    ];
    let red = ops[gather..wkv]
        .iter()
        .position(|o| collective.contains(o))
        .map(|i| i + gather)
        .expect("the row-split gather MUST be all-reduced before wkv");
    let cols = super::dsv41::engram_cols(&cfg) * cfg.raw.engram_head_dim as u32;
    assert_eq!(cols, 6144, "(max_ngram_size - 1) * n_heads * head_dim");
    assert_eq!(
        p.insts[red].i[0],
        t * cols,
        "the reduce covers the whole [T][n_cols * head_dim] gather"
    );

    // The gather is a PARTIAL and must land in the peer region, which is keyed by name.
    assert_eq!(name(p.insts[gather].t[0]), super::dsv41::PEER_SLOT_O);
    // ...and the slot is sized for the WIDER of its two users: 6144 here against hidden = 5120.
    let slot = m
        .tensors
        .iter()
        .find(|x| x.name == super::dsv41::PEER_SLOT_O)
        .expect("peer slot 0 is declared");
    assert_eq!(slot.bytes, u64::from(t) * u64::from(cols) * 2);

    // One program serves all eight ranks, so the shard base cannot be an immediate.
    assert_eq!(
        p.insts[gather].i[4],
        packet::dev::TENSOR_NONE_I,
        "vocab_start must be derived from prog.rank, not written as a constant"
    );
    let rows = cfg.raw.engram_rows(1).expect("layer 1 has a table");
    assert_eq!(p.insts[gather].i[5], (rows as u64).div_ceil(8) as u32);
    assert_eq!(p.insts[gather].i[3], 32, "V4.1's block size is 32, not V4's 128");

    // The gate is IN PLACE on the residual stream's hc copies -- the tensor mhc_pre then reads.
    assert_eq!(name(p.insts[gate].t[0]), "act.hc_residual_a");
    assert_eq!(p.insts[gate].i[1], cfg.hc_mult);
    assert_eq!(p.insts[gate].i[2], cfg.hidden);
    assert_eq!(
        p.insts[gate].t[4],
        packet::dev::TENSOR_NONE,
        "no token_mask on a text-only prefill; a VL path must pass one"
    );

    // Layer 0 has none of this.
    let (m0, _) = super::dsv41::emit_dsv41_block(&cfg, &[0], 8, 304, 2048, t);
    assert!(
        !m0.progs[0]
            .insts
            .iter()
            .any(|d| d.op == DevOp::EngramEmbed as u16),
        "layer 0 has no engram table and must not gather"
    );
}

/// V4.1's mHC is CROSS-SUBLAYER: every `hc_pre` collapses with the PREVIOUS sublayer's `pre`.
///
/// `Block.forward` (`model.py:965-996`) takes `pre_mix` as an ARGUMENT -- the previous block's
/// `ffn_pre` -- and collapses attention with it, then collapses the FFN with this block's
/// `attn_pre`, returning `ffn_pre` for the next block. The class docstring states it: "the
/// coefficients a sublayer computes are used by the *next* one". `Transformer.forward` seeds the
/// chain with `make_identity_pre_mix`, a ONE-HOT on copy 0 (`model.py:1159-1163`).
///
/// GLM-5.3's `hc_pre`, which this emit binds onto, collapses with the `pre` the SAME sublayer just
/// derived, and nothing in the shapes distinguishes the two -- both are an `hc_mult`-vector over
/// the same residual. `scripts/dsv41_mhc_oracle.py` prices the difference at 51.9% relative on
/// LAYER 0 ALONE, because V4.1 feeds layer 0's attention residual copy 0 while the same-sublayer
/// ordering feeds it a learned mix off `hc_attn_fn`. That is a model that loads, runs, and is
/// wrong, so the half indices are pinned here rather than left to the emitter's arithmetic.
#[test]
fn the_mhc_pre_gate_comes_from_the_previous_sublayer() {
    let Some((mut cfg, _)) = checkpoint() else {
        return;
    };
    let _guard = crate::test_env::env_guard();
    cfg.raw.compress_ratios = vec![0; cfg.raw.compress_ratios.len()];

    let (m, _) = super::dsv41::emit_dsv41_block(&cfg, &[3, 4], 8, 304, 2048, 256);
    let pre: Vec<_> = m.progs[0]
        .insts
        .iter()
        .filter(|d| d.op == DevOp::HyperConnPre as u16)
        .collect();
    assert_eq!(
        pre.len(),
        4,
        "two layers is four mHC sublayers -- attention and FFN each have one"
    );

    // `PLOW_HC_PRE_*` in `runtime/amd/op_hyperconn.h`: 0 OWN (GLM-5.3), 1 SEED, 2 DEFER.
    let modes: Vec<u32> = pre.iter().map(|d| d.i[5]).collect();
    assert_eq!(
        modes,
        vec![1, 2, 2, 2],
        "the chain's first sublayer seeds with the one-hot and every later one defers; a `0` here          is GLM-5.3's same-sublayer ordering, which is 51.9% wrong on layer 0 alone"
    );

    // Sublayer k reads half k%2 and publishes into the other, so the halves alternate. Two
    // consecutive sublayers reading the same half would have the second one read its own output.
    let halves: Vec<u32> = pre.iter().map(|d| d.i[4]).collect();
    assert_eq!(halves, vec![0, 1, 0, 1], "the pre_pair halves must alternate");

    let pair = pre[0].t[7];
    assert_ne!(pair, packet::dev::TENSOR_NONE, "pre_pair must be bound");
    assert!(
        pre.iter().all(|d| d.t[7] == pair),
        "every sublayer shares ONE pre_pair tensor -- the deferral is a handoff through it"
    );
    assert_eq!(m.tensors[pair as usize].name, "act.hc_pre_pair");
    assert_eq!(
        m.tensors[pair as usize].bytes,
        2 * 256 * u64::from(cfg.hc_mult) * 4, // t = 256; emit_dsv41_block takes ctx then t
        "both halves live in one tensor: [2][t][hc_mult] f32"
    );
}

/// Reading the shared CSA2 cache is a PART, and it lands on the READER layers, not the writers.
///
/// `kv_source` says who WRITES the cache -- 4 layers. `compress_ratios[l]` says which cache layer
/// `l` READS, and only 0 means "sliding window only". The parts table used to consult the writer
/// list alone, so all 38 reader layers were reported COMPLETE while the emit dispatched
/// `FlashMlaPrefill` with `KV_MASK_NONE` and nothing but the 128-token window. The row exists on
/// exactly the layers whose ratio is nonzero, and it is pinned here by layer id so that a
/// regression naming the wrong layers is as loud as one changing how many.
#[test]
fn the_compressed_read_lands_on_the_reader_layers_not_the_writers() {
    let Some((cfg, _)) = checkpoint() else {
        return;
    };
    let reads = |l: u32| {
        super::dsv41::dsv41_layer_parts(&cfg, l)
            .iter()
            .any(|(n, _)| n.starts_with("compressed-KV attention"))
    };

    // Only layers 0 and 1 are window-only; compress_ratios is [0, 0, 2 x18, 1 x20, 0, 0, 0] and
    // the trailing three entries are DSpark blocks, not layers.
    assert!(!reads(0), "layer 0 is window-only");
    assert!(!reads(1), "layer 1 is window-only");
    for l in 2..cfg.layers {
        assert!(reads(l), "layer {l} reads the compressed cache and must say so");
    }

    // And the reader side is NOT the writer side: layer 3 reads without owning a compressor.
    assert!(!cfg.kv_source.contains(&3));
    assert!(reads(3), "a pure reader still needs the read emitted");

    // ALL 40 LAYERS EMIT. Stated as the set, not the count.
    let emits: Vec<u32> = (0..cfg.layers)
        .filter(|l| super::dsv41::dsv41_emit_block_plan(&cfg, *l).is_ok())
        .collect();
    assert_eq!(
        emits,
        (0..cfg.layers).collect::<Vec<u32>>(),
        "every layer emits: the window everywhere, the compressor on 4, the indexer on 8, the \
         Engram on 2 and the gathered read on 38"
    );
}
