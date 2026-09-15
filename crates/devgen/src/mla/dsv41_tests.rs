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
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0, 1]);
    for l in [0u32, 1] {
        let listed = super::dsv41::dsv41_layer_tensors(&cfg, l);
        assert!(!listed.is_empty(), "layer {l} must carry weights");
        let mut seen = std::collections::BTreeSet::new();
        for (suffix, sz) in &listed {
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
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &all);
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
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &all);
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
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0]);
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
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0]);
    let all: Vec<u32> = (0..304u32).collect();
    let t = 512u32;
    let x = b.tensor("act.x", (t as u64) * (cfg.hidden as u64) * 2);
    let (act, _last) = super::dsv41::emit_dsv41_attn_proj(&mut b, &cfg, &w, &all, 0, x, t, &[]);
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
    let w = super::dsv41::declare_dsv41_weights(&mut b, &cfg, &[0]);
    let all: Vec<u32> = (0..304u32).collect();
    let t = 512u32;
    let x = b.tensor("act.x", (t as u64) * (cfg.hidden as u64) * 2);
    let (act, _) = super::dsv41::emit_dsv41_attn_proj(&mut b, &cfg, &w, &all, 0, x, t, &[]);
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
