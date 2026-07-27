//! `gemma4` — compile the REAL Gemma-4 31B (or 12B) prefill network into a device packet
//! program, straight from the HuggingFace checkpoint.
//!
//! # DEPRECATED
//!
//! Superseded by `plowc --hf-dir <dir>`, which compiles the same checkpoints
//! through the shared driver pipeline. Only built with
//! `--features legacy-gemma-bins`; slated for removal.
//!
//! Reads `config.json` and the safetensors index, and emits packets whose weight tensors
//! are named EXACTLY as the checkpoint names them, so the runtime can bind them by name
//! and hard-fail on anything missing. A silently-absent weight is the worst failure mode
//! in this whole stack: the model still produces fluent text, just wrong text.
//!
//! # The spec, verified against the checkpoint and modeling_gemma4.py — not from memory
//!
//! Every one of these is a silent fluent-but-wrong bug if you get it wrong:
//!
//! * **RMSNorm has NO `+1`.** `x * pow(mean(x^2) + eps, -0.5) * w`, eps INSIDE the power.
//!   Gemma 1/2/3 used `(1 + w)` with zero-init weights; Gemma 4 is ones-init and dropped it.
//! * **Attention scale is 1.0.** There is no `1/sqrt(head_dim)` anywhere — the trained
//!   `q_norm` absorbs it (`self.scaling = 1.0` in the reference).
//! * **`v_norm` is a WEIGHTLESS RMSNorm** over head_dim, applied to V on every layer, and
//!   it has no checkpoint tensor — so it is the easiest thing in the model to omit.
//! * **Full-attention layers have NO `v_proj`** (`attention_k_eq_v: true`). V comes from the
//!   RAW k_proj output: `K = RoPE(k_norm(kv))`, `V = v_norm(kv)`, both from one projection.
//!   Confirmed in the checkpoint: layer 5 ships q/k/o_proj and no v_proj.
//! * **Full layers use `global_head_dim` = 512 and `num_global_key_value_heads` = 4**, not
//!   the sliding layers' 256/16.
//! * **Partial RoPE on full layers**: `rope_angles = int(0.25 * 512 // 2) = 64`, so
//!   `inv_freq[i] = 1e6^(-2i/512)` for i < 64 and **ZERO for i in [64, 256)** — those dims
//!   pass through unrotated (NoPE). Rotated pairs are `(i, i+256)`, not `(i, i+64)`.
//! * **MLP is GeGLU** (gelu_pytorch_tanh), not SwiGLU.
//! * **Sandwich norms**: the residual is added AFTER the post-norm.
//! * **`layer_scalar`** is a learned `[1]` tensor multiplying the whole hidden state at the
//!   end of each layer. We fold it into the second residual's scale — algebraically the
//!   same thing — which means the COMPILER has to read it out of the checkpoint.
//! * **Embedding scale is the BF16-ROUNDED sqrt(hidden)**: 73.5, not 73.3212.
//! * **Tied lm_head**, then `logits = 30 * tanh(logits / 30)`.

use std::path::PathBuf;

use costmodel::hwspec;
use packet::dev::{DevInst, DevOp, TENSOR_NONE};
use packet::devbuild::{Builder, Dep, Model};
use packet::rope::{GenTensor, RopeScale};
use serde_json::Value;

mod checkpoint;
use checkpoint::{layer_scalars, validate_coverage};
mod block;
use block::{parse_block, write_block_descriptor};
mod config;
use config::*;
mod ladder;
mod mla;
use mla::{glm_main, glm_emit_block, kimi_emit_block, nemotron_emit_block, MlaArch};
pub mod manifest;

/// Flash-decode GQA fusion factor on FULL-attention layers. **This must equal the
/// kernel constant** (`PLOW_NV_FA_GF` on sm_120, `PLOW_FA_GF_FULL` on AMD) or the
/// compiler and kernel disagree about how many query heads one work item carries.
/// Previously a bare `let gf = 2` used only in an assertion — it never reached the
/// packet, so it read like a knob and controlled nothing. Qwen3 is gqa=4 and the
/// sm_120 build ships GF=4 (worth a measured 1.71x on flash-decode); GF=2 is a
/// Gemma artifact. The binding invariant is `gqa_local % FA_GF_FULL == 0`.
const FA_GF_FULL: u32 = 2;

/// The GF the packet must be built for — `PLOW_FA_GF_FULL` if set, else the
/// constant above.
///
/// WHY THIS EXISTS. `FA_GF_FULL` is documented as "must equal the kernel
/// constant", and the two derive different things from it: the kernel decides
/// how many query heads one flash work item carries, the compiler decides
/// `nsplit` so that `n_grp * nsplit` fills the resident grid. Nothing enforced
/// the equality, and nothing could: `PLOW_NV_FA_GF_FULL` is an nvcc `-D` on the
/// object while this is a Rust constant in the emitter. They are already out of
/// step on main — `scripts/build_sm120_cubin.sh` ships `-DPLOW_NV_FA_GF_FULL=4`
/// against this `2`.
///
/// For a TUNER the consequence is worse than a wrong constant. Sweeping
/// `-DPLOW_NV_FA_GF_FULL` alone re-splits the work in the kernel while the
/// packet keeps sizing `nsplit` for GF=2, so every arm measures a
/// compiler/kernel DISAGREEMENT rather than the knob — and the sweep would
/// faithfully report the mismatch as the knob's effect. Measured on the
/// Gemma-4-12B full block (layer 5, kv_heads 1) at B=1/ctx 130560 with the
/// packet pinned at 2: GF 2 → 971.8 us, 4 → 689.3, 8 → 986.9, i.e. the widest
/// fusion looked WORST while it was the one furthest from the packet's
/// assumption.
///
/// So GF_FULL is a PAIR — object define plus packet env — in exactly the way
/// `(PLOW_NV_FORCE_MINBLK, --n-cu)` already is, and `DecodeKnobs::emit_env`
/// renders both halves together. Unset leaves every packet byte-identical.
pub(crate) fn fa_gf_full() -> u32 {
    std::env::var("PLOW_FA_GF_FULL")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|v| matches!(v, 1 | 2 | 4 | 8 | 16))
        .unwrap_or(FA_GF_FULL)
}

/// Greatest common divisor (Euclid). Used to grid-align the full-layer flash-decode
/// nsplit to the resident-block count (T9b).
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

const BF16: u64 = 2;
const F32: u64 = 4;
const I32: u64 = 4;
const BM: u32 = 256;
const BN: u32 = 256;

#[cfg(test)]
const GFX950_TILES: [(DevOp, u64, u64, u64); 3] = [
    (DevOp::Gemm, 256, 256, 64), // 144 KiB LDS — best when the shape SATURATES the 256 CUs
    (DevOp::GemmMed, 128, 128, 64), //  72 KiB LDS — fills the chip when a 256-tile leaves CUs idle
    (DevOp::GemmSmall, 64, 128, 64), // 55 KiB LDS — narrow M (short prompts / small chunks)
];

/// LDS the double-buffered A|B staging of a `BMxBNxBK` tile occupies, in bytes. Mirrors
/// `GM_LDS_HALVES_T` in `op_gemm.h`: 2 buffers x (BM+BN) rows x (BK+8 pad) halves x 2 B/half.
pub fn gemm_lds_bytes(bm: u64, bn: u64, bk: u64) -> u64 {
    2 * (bm + bn) * (bk + 8) * 2
}

/// Wall-clock cost of one GEMM tile for one shape — the single ranking used by
/// both `plowc tune` and the device-blob emitters (`pick_tile`).
///
/// Output tiles run in parallel, `n_units` at a time, so wall time is
/// `rounds x (cost of ONE tile)` and one tile costs `max(compute, dma)`: the
/// tile is double-buffered and SRAM-resident, so its operand fill hides behind
/// its matrix compute. Two opposing effects fall out with no hand-tuned
/// constants — a bigger tile has better arithmetic intensity `BM*BN/(BM+BN)`, a
/// smaller one makes more tiles and fills more units.
///
/// A tile whose working set overflows SRAM scores [`u64::MAX`] rather than being
/// dropped, so the ranking stays total. Ties resolve toward the larger tile via
/// a rank term in the low bits; without it, equal-cost shapes would resolve by
/// opcode number, and `GemmSmall` is 14 while `GemmMed` is 15.
pub fn tile_cost(
    spec: &hwspec::GpuSpec,
    kernel: &kernelcaps::KernelSpec,
    m: i64,
    n: i64,
    k: i64,
    n_units: u32,
) -> u64 {
    use costmodel::cost::{dma_cycles, macs_cycles};

    let Some(tile) = kernel.tile else { return u64::MAX };
    let (bm, bn, bk) = (tile.bm as u64, tile.bn as u64, tile.bk as u64);
    let (m, n, k) = (m.max(1) as u64, n.max(1) as u64, k.max(1) as u64);
    let n_units = (n_units as u64).max(1);

    if gemm_lds_bytes(bm, bn, bk) > spec.sm.shared_mem.0 {
        return u64::MAX;
    }

    let tiles = m.div_ceil(bm) * n.div_ceil(bn);
    let rounds = tiles.div_ceil(n_units);
    let k_iters = k.div_ceil(bk);
    let compute = k_iters * macs_cycles(spec, bm * bn * bk, hwspec::MmaDtype::Bf16);
    let dma = dma_cycles(spec, (bm * k + k * bn) * 2, false);
    let cost = rounds.saturating_mul(compute.max(dma));

    // Larger tile first on a tie: rank by descending BM*BN.
    let rank = match bm * bn {
        a if a >= 65536 => 0, // 256x256
        a if a >= 16384 => 1, // 128x128
        _ => 2,               // 64x128 and narrower
    };
    cost.saturating_mul(4).saturating_add(rank)
}

/// Pick the GEMM tile + inner-loop kernel for one `(M,N,K)` shape STATICALLY, from the gfx950
/// hardware spec — every shape in a `plow` schedule is known at compile time, so this is a
/// closed-form choice, not a runtime autotuner.
///
/// The choice is driven entirely by [`hwspec`] quantities, funnelled through the shared
/// [`costmodel`]: the bf16 MFMA rate (`sm.mma.bf16` x `tensor_cores`, via [`macs_cycles`]), the
/// HBM bandwidth (`mem.bandwidth`, via [`dma_cycles`]), the per-CU LDS budget
/// (`sm.shared_mem` = 160 KiB) and the CU count (`sm_count` = 256).
///
/// The model is WALL-CLOCK, not total-work — and that distinction is the whole point. Output
/// tiles run in PARALLEL, `n_cu` at a time, so wall time is `rounds x (cost of ONE tile)` where
/// `rounds = ceil(tiles / n_cu)`. One tile costs `max(compute, dma)`: the tile is double-buffered
/// and LDS-resident, so its HBM operand fill hides behind its MFMA compute. Two opposing effects
/// then fall straight out of the arithmetic, with no hand-tuned constants:
///
///   * a BIGGER tile has better arithmetic intensity (`BM*BN/(BM+BN)`), so lower per-tile DMA —
///     it wins once the shape SATURATES the CUs (q/o/gate/up/down at M=4096: >=256 tiles);
///   * a SMALLER tile makes MORE tiles, so it wins when the big tile leaves CUs idle — the
///     Llama/Qwen k/v projections (N=1024) are only 16x4 = 64 tiles at 256x256 (a quarter of the
///     256 CUs), and drop to a full 16x8 = 256 tiles at 128x128.
///
/// So `pick_tile` now returns `GemmMed` (128x128) for k/v and `Gemm` (256x256) for the wide
/// projections, filling all 256 CUs on both — where the old 3-candidate heuristic, blind to the
/// MFMA rate and to CU fill, pinned k/v to 256x256 and ran them on 64 CUs. It matches the
/// measured T=4096 optima (256x256 best) and generalises: Gemma-31B's kv_proj is N=4096, already
/// saturating, so it stays 256x256 — no regression.
fn pick_tile(m: u32, n: u32, k: u32, n_cu: u32) -> DevOp {
    let spec = hwspec::registry::lookup("MI350X").expect("gfx950 spec in registry");
    let hw = kernelcaps::HardwareFingerprint::from_spec(spec).expect("gfx950 fingerprint");
    let op = kernelcaps::OpSignature::gemm(kernelcaps::Phase::Prefill, m as i64, n as i64, k as i64);

    // The registry decides what is *executable*; the closure decides which of
    // those is fastest. Fusing both halves into one loop over a constant table
    // is what let this function name a tile the target does not implement
    // whenever it ran for a build that was not gfx950.
    let realization = kernelcaps::select_kernel(
        gfx950_gemm_inventory(),
        &op,
        &hw,
        kernelcaps::ProfileId::PrefillDense,
        &kernelcaps::NoMeasurements,
        |kernel| tile_cost(spec, kernel, m as i64, n as i64, k as i64, n_cu),
    )
    .expect("the gfx950 registry serves every prefill GEMM shape");

    realization.kernel.0
}

/// The gfx950 dense-GEMM inventory, derived by probing the interpreter object.
///
/// Probed when possible, analytical fallback otherwise. A hand-written tile
/// table is exactly what drifts from the object being compiled for, and AMD's
/// dispatch default silently no-ops an opcode with no arm
/// (`runtime/amd/interp.hip:785`), so drift would surface as slightly wrong
/// output rather than a crash. However, **requiring hipcc on a machine that
/// only targets NVIDIA** is a worse ergonomic failure than using known-stable
/// tile constants, so when the probe fails (hipcc missing) we fall back to
/// the analytical inventory — the same tiles the test fixture locks in.
#[cfg(not(test))]
fn gfx950_gemm_inventory() -> &'static kernelcaps::Inventory {
    use std::sync::OnceLock;
    static INV: OnceLock<kernelcaps::Inventory> = OnceLock::new();
    INV.get_or_init(|| {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        match kernelcaps::dense_gemm_inventory(&root, hwspec::IsaLevel::Gfx950) {
            Ok(inv) => inv,
            Err(e) => {
                eprintln!(
                    "warning: cannot probe gfx950 kernel inventory ({e}); \
                     using analytical fallback (known tile constants)"
                );
                gfx950_analytical_inventory()
            }
        }
    })
}

/// Analytical fallback inventory for gfx950 — the three tile instantiations
/// from `runtime/amd/op_gemm.h` (GM_BM/BN/BK, GM_MD_*, GM_SM_*). These are
/// compile-time constants in the interpreter object and change only with an
/// intentional ABI-breaking edit to op_gemm.h.
fn gfx950_analytical_inventory() -> kernelcaps::Inventory {
    use packet::dev::DevOp;
    let build = kernelcaps::BuildId::new(
        hwspec::IsaLevel::Gfx950,
        ["PLOW_BUCKET_DECODE=0".to_string()],
        "analytical-fallback",
        "analytical-fallback",
    );
    kernelcaps::Inventory::probed(
        build,
        [
            (DevOp::Gemm, 256, 256, 64, "gfx950:exec_gemm"),
            (DevOp::GemmMed, 128, 128, 64, "gfx950:exec_gemm_med"),
            (DevOp::GemmSmall, 64, 128, 64, "gfx950:exec_gemm_small"),
        ]
        .map(|(op, bm, bn, bk, body)| {
            kernelcaps::KernelSpec::gemm_tile(op, hwspec::IsaLevel::Gfx950, bm, bn, bk, body)
        }),
    )
}

/// Test fixture standing in for a probe.
///
/// This is a test *input*, not shipped data: it never reaches a compiled
/// artifact, and production has no path to it. It exists so the tile-selection
/// regression tests can run on a machine without ROCm, which is the only reason
/// the real probe is unavailable here.
#[cfg(test)]
fn gfx950_gemm_inventory() -> &'static kernelcaps::Inventory {
    use packet::dev::DevOp;
    use std::sync::OnceLock;
    static INV: OnceLock<kernelcaps::Inventory> = OnceLock::new();
    INV.get_or_init(|| {
        let build = kernelcaps::BuildId::new(
            hwspec::IsaLevel::Gfx950,
            ["PLOW_BUCKET_PREFILL=1".to_string()],
            "test-fixture",
            "test-fixture",
        );
        // The three instantiations in runtime/amd/op_gemm.h, with the GM_* tile
        // constants a probe would expand.
        kernelcaps::Inventory::probed(
            build,
            [
                (DevOp::Gemm, 256, 256, 64, "gfx950:exec_gemm"),
                (DevOp::GemmMed, 128, 128, 64, "gfx950:exec_gemm_med"),
                (DevOp::GemmSmall, 64, 128, 64, "gfx950:exec_gemm_small"),
            ]
            .map(|(op, bm, bn, bk, body)| {
                kernelcaps::KernelSpec::gemm_tile(op, hwspec::IsaLevel::Gfx950, bm, bn, bk, body)
            }),
        )
    })
}

fn tiles(m: u32, n: u32) -> u32 {
    m.div_ceil(BM) * n.div_ceil(BN)
}

/// Every tensor the model touches. Prefill and decode SHARE this table: the KV cache is
/// written by prefill and appended to by decode, and the 57 GiB of weights is not getting
/// loaded twice.
#[allow(dead_code)]
struct Tn {
    ids: u32,
    pos: u32,
    kvlen: u32,
    cos_s: u32,
    sin_s: u32,
    cos_f: u32,
    sin_f: u32,
    emb: u32,
    fin: u32,
    head: u32,
    // fp8 weight-only lm_head twin (PLOW_FP8_HEAD=1; tied models only). Separately-
    // labelled variant: vLLM fp8 keeps lm_head bf16, so report as its own row.
    head8: u32,
    head8s: u32,
    x: u32,
    hn: u32,
    qg: u32,
    kg: u32,
    vg: u32,
    q: u32,
    opart: u32,
    mlpart: u32,
    at: u32,
    og: u32,
    gt: u32,
    ut: u32,
    fu: u32,
    dg: u32,
    logits: u32,
    amax: u32,
    // TP (n_gpu>1) peer-mapped partial slots (plans/tp-design.md §7a): the row-parallel
    // o_proj/down write their partial H-vector here (tp-host binds them into peer_scratch),
    // and XReduce sums the N peers' slots into `og`/`dg`. TENSOR_NONE when tp==1.
    og_tp: u32,
    dg_tp: u32,
    kc: Vec<u32>,
    vc: Vec<u32>,
    // fp8-KV: per-(token,kv_head) f32 dequant scales, one per KV row (TENSOR_NONE in bf16 mode).
    kcs: Vec<u32>,
    vcs: Vec<u32>,
    // MoE decode scratch (shared across layers, B=1). dense MLP reuses fu/dg; the MoE branch adds:
    // h1 = post_ffn_norm_1(dense down); routing_table[k]; xn2 = pre_ffn_norm_2(residual);
    // mfu[k,I]; part[k,H] (f32); moe_sum[H]; h2 = post_ffn_norm_2(moe_sum); comb = h1+h2.
    moe_h1: u32,
    moe_tab: u32,
    moe_rscore: u32,
    moe_xn2: u32,
    moe_mfu: u32,
    moe_part: u32,
    moe_sum: u32,
    moe_h2: u32,
    moe_comb: u32,
    // Grouped-MoE PREFILL scratch (plans/p9-26b-prefill-moe.md). Declared only when moe && moe_pf;
    // TENSOR_NONE otherwise so the decode-only blob stays byte-identical. total_pad = rows*top_k +
    // n_exp*128 (PGM_BM). meta = int32[3*n_exp+2] align/sort table; rowtok/rowpart = u32[total_pad]
    // gather maps; rowgate = f32[total_pad]; fug = bf16[total_pad*moe_inter] gathered GLU output.
    moe_meta: u32,
    moe_rowtok: u32,
    moe_rowpart: u32,
    moe_rowgate: u32,
    moe_fug: u32,
    // beat26b w8a8 grouped-MoE prefill: fp8 twin of the gathered GLU output `fug` (uint8
    // [total_pad*moe_inter] e4m3 + f32 fscale[total_pad]), quantized by QuantFp8 between the w8a8
    // GLU and DOWN. TENSOR_NONE unless moe_pf && w8a8. (xn2 reuses xqh/ash — same hidden width.)
    moe_fuq: u32,
    moe_fus: u32,
    // T8 w8a8: reused-per-layer fp8 ACTIVATION quant scratch (uint8 xq + f32 row a_scale), one pair
    // per distinct activation width. Emitted only under PLOW_W8A8; TENSOR_NONE otherwise.
    //   xqh/ash  — hidden-width (q/k/v read n.hn; gate/up read n.hn again).
    //   xqo/aso  — qd-width (o_proj reads n.at).
    //   xqi/asi  — inter-width (down reads n.fu).
    // The three widths never alias in liveness within a layer (the DAG's existing edges serialize
    // qkv→flash→o→norm→gate/up→down), so each pair is reused across all 48 layers.
    xqh: u32,
    ash: u32,
    xqo: u32,
    aso: u32,
    xqi: u32,
    asi: u32,
    lw: Vec<LW>,
}
struct LW {
    wq: u32,
    wk: u32,
    wv: u32,
    wo: u32,
    wg: u32,
    wu: u32,
    wd: u32,
    g_in: u32,
    g_pa: u32,
    g_pf: u32,
    g_po: u32,
    qn: u32,
    kn: u32,
    // Gemma-4 MoE (26B-A4B) per-layer weights + the loader-filled expert pointer table.
    // TENSOR_NONE on the dense 12B/31B path. rproj/rscale/rpes = router; g_pf1/g_pf2/g_pre2 = the
    // three extra sandwich norms. ewt = Persistent u64[E*2] {gate_up base, down base} per expert,
    // filled by the harness/loader from the two FUSED expert tensors' bound bases. The fused expert
    // weights (experts.gate_up_proj [E,2I,H], experts.down_proj [E,H,I]) are declared as pkt tensors
    // (so the loader binds them by name) but are NOT op operands — the SM reaches them via ewt.
    rproj: u32,
    rscale: u32,
    rpes: u32,
    g_pf1: u32,
    g_pf2: u32,
    g_pre2: u32,
    ewt: u32,
    est: u32,
    // FP8 DECODE weights (PLOW_FP8) + their per-output-channel f32 dequant scales. The bf16 wq..wd
    // above stay bound (from the bf16 checkpoint) and feed PREFILL's GEMM; these fp8 twins feed the
    // decode GEMV. TENSOR_NONE in bf16 mode. The fp8 weight/scale tensors are declared under an
    // "fp8/" name prefix that the loader routes to the fp8 checkpoint (see gemma4_chat.c).
    wq8: u32,
    wk8: u32,
    wv8: u32,
    wo8: u32,
    wg8: u32,
    wu8: u32,
    wd8: u32,
    sq: u32,
    sk: u32,
    sv: u32,
    so: u32,
    sg: u32,
    su: u32,
    sd: u32,
}

fn declare(
    b: &mut Builder,
    c: &Cfg,
    ctx: u32,
    ns_pre: u32,
    fp8: bool,
    w8a8: bool,
    fp8_kv: bool,
    fp8_kv_full: bool,
    dbatch: u32,
    moe_pf: bool,
    block: std::ops::Range<usize>,
) -> Tn {
    // ACTIVATIONS ARE SIZED BY THE CHUNK, NOT THE CONTEXT.
    //
    // Every activation used to be `ctx * ...`, which is 131072 rows of scratch for a machine
    // that never has more than MAX_CHUNK=4096 rows in flight (prefill chunk) or 1 (decode).
    // That is a 32x over-allocation and it was 45.7 GiB of the 119 GiB footprint -- more than
    // the KV cache and nearly as much as the weights.
    //
    // Only `ids`/`pos` and the KV cache legitimately span the context: the cache IS the context,
    // and ids/pos are i32 (a rounding error). Everything else holds the CURRENT chunk.
    let rows = ctx.min(max_chunk(c.window));
    // TP head split (plans/tp-design.md §3a): each rank owns heads/N q-heads and kvh/N kv-heads,
    // so every head-dimensioned activation and the KV cache shrink by N. Column/row-parallel
    // weights and the inter/vocab-dimensioned activations shrink by N too. tp==1 => /1, identical.
    let tp = c.tp;
    assert_eq!(c.heads % tp, 0, "--tp {tp} must divide n_head {}", c.heads);
    assert_eq!(
        c.inter % tp,
        0,
        "--tp {tp} must divide intermediate {}",
        c.inter
    );
    // GEMV 8-wide load contract (runtime/nvidia/op_gemm.cuh): the decode GEMV family loads the
    // contraction dim (K) in 8-element vectors guarded only by `k < K`, so a K that is not a
    // multiple of 8 over-reads the final vector past the row. Every dim that becomes a GEMV K —
    // hidden (qkv/gate/up/lm_head), intermediate (down), and each head_dim (attn out) — must be
    // 8-aligned. Holds for all supported checkpoints; enforce it so an unaligned dim fails at
    // emit time instead of silently over-reading on device.
    assert_eq!(c.hidden % 8, 0, "hidden {} must be a multiple of 8 (GEMV 8-wide load)", c.hidden);
    assert_eq!(c.inter % 8, 0, "intermediate {} must be a multiple of 8 (GEMV 8-wide load)", c.inter);
    assert_eq!(c.hd_slide % 8, 0, "head_dim {} must be a multiple of 8 (GEMV 8-wide load)", c.hd_slide);
    assert_eq!(c.hd_full % 8, 0, "global_head_dim {} must be a multiple of 8 (GEMV 8-wide load)", c.hd_full);
    let qd_max = (c.heads / tp) * c.hd_slide.max(c.hd_full);
    // kv activation shards use the per-rank LOCAL kv-head count (shared-kv-head replication clamps
    // it to 1 when tp>kvh, so kvh/tp would under-size to 0 at tp=8 on full layers — §3a/§13.2).
    let kd_max =
        (kvh_local(c.kvh_slide, tp, 0) * c.hd_slide).max(kvh_local(c.kvh_full, tp, 0) * c.hd_full);
    let hd_max = c.hd_slide.max(c.hd_full);
    let inter_sh = c.inter / tp;
    // lm_head is REPLICATED under TP (Phase 2), not vocab-sharded. Two reasons the
    // sharded path is deferred: (1) Gemma ties lm_head to embed_tokens, and the emitted
    // lm_head Gemv reads `emb` from offset 0 with no per-rank vocab offset, so a vocab
    // shard would make every rank argmax the SAME low-vocab slice (silently wrong);
    // (2) XArgmaxFin (the cross-rank id-fold) is a stub. Replicating lm_head keeps the
    // full-vocab argmax correct on every rank (they agree), costs no extra memory (emb
    // is already fully resident for the embed lookup), and is one gemv/token — not the
    // decode bottleneck. Sharded lm_head + XArgmaxFin is a Phase-3 item (§8d, §13).
    let vocab_sh = c.vocab;
    let ac = |b: &mut Builder, n: &str, sz: u64| b.tensor(&format!("act.{n}"), sz);

    // RoPE tables are declared as RECIPES, not expanded bytes: at ctx=131072 the four
    // of them are ~403 MB, which dominated the blob, the load-time H2D, and nothing
    // else. The runtime materialises them from these scalars at bind time; `--no-rope-gen`
    // (Model::bake_gen) puts the bytes back for readers that predate v7.
    let [cs_s, sn_s] = GenTensor::rope_pair(ctx, c.hd_slide, c.theta_slide, 1.0, RopeScale::None);
    let [cs_f, sn_f] =
        GenTensor::rope_pair(ctx, c.hd_full, c.theta_full, c.rope_frac_full, c.rope_scale);

    // MoE row count for buffer sizing: 1 for the decode-only blob (byte-identical to the pre-prefill
    // path), the chunk `rows` when grouped-MoE prefill is enabled. total_pad bounds the token-sorted
    // gathered rows: rows*top_k routed slots + n_exp segments each padded up to the 128-row tile.
    let moe_pf_on = c.moe && moe_pf;
    // BATCH>1 DECODE: the decode MoE scratch is per-ROW ([B][k] table, [B][k][I] mfu,
    // [B][k][H] part, [B][n_exp] scores), so every per-token buffer is sized for B rows too.
    // dbatch==1 leaves every size exactly as it was => byte-identical blob.
    let moe_rows = (if moe_pf_on { rows } else { 1 }).max(dbatch);
    let total_pad = moe_rows * c.top_k + c.n_exp * 128;

    let t = Tn {
        ids: b.tensor("in.ids", ctx as u64 * I32),
        pos: b.tensor("in.pos", ctx as u64 * I32),
        // BATCH>1 (serving pending #4): one KV length per sequence. dbatch==1 => I32, identical.
        kvlen: b.tensor("in.kvlen", dbatch as u64 * I32),
        cos_s: b.tensor_gen("in.cos_slide", cs_s.byte_len(), cs_s),
        sin_s: b.tensor_gen("in.sin_slide", sn_s.byte_len(), sn_s),
        cos_f: b.tensor_gen("in.cos_full", cs_f.byte_len(), cs_f),
        sin_f: b.tensor_gen("in.sin_full", sn_f.byte_len(), sn_f),
        emb: b.tensor(
            &format!("{}embed_tokens.weight", c.prefix),
            (c.vocab * c.hidden) as u64 * BF16,
        ),
        fin: b.tensor(&format!("{}norm.weight", c.prefix), c.hidden as u64 * BF16),
        // Untied lm_head (Llama): a separate top-level "lm_head.weight". Tied models reuse emb.
        head: if c.tied {
            TENSOR_NONE
        } else {
            b.tensor("lm_head.weight", (c.vocab * c.hidden) as u64 * BF16)
        },
        head8: if c.tied && std::env::var("PLOW_FP8_HEAD").ok().as_deref() == Some("1") {
            b.tensor(
                &format!("fp8/{}embed_tokens.weight", c.prefix),
                (c.vocab * c.hidden) as u64,
            )
        } else {
            TENSOR_NONE
        },
        head8s: if c.tied && std::env::var("PLOW_FP8_HEAD").ok().as_deref() == Some("1") {
            b.tensor(
                &format!("fp8/{}embed_tokens.weight_scale", c.prefix),
                c.vocab as u64 * F32,
            )
        } else {
            TENSOR_NONE
        },
        x: ac(b, "x", (rows * c.hidden) as u64 * BF16),
        hn: ac(b, "hn", (rows * c.hidden) as u64 * BF16),
        qg: ac(b, "qg", (rows * qd_max) as u64 * BF16),
        kg: ac(b, "kg", (rows * kd_max) as u64 * BF16),
        vg: ac(b, "vg", (rows * kd_max) as u64 * BF16),
        q: ac(b, "q", (rows * qd_max) as u64 * BF16),
        // Sized for whichever phase needs more. Prefill: ctx * heads * ns_pre * hd.
        // Decode: 1 * heads * ns_dec * hd. Prefill wins for any sane ctx.
        // Sized as the MAX of what each phase needs, not as a product of both.
        //
        // It used to be `ctx * heads * ns_pre.max(8) * hd`. The `.max(8)` is there to cover the
        // DECODE program, whose nsplit is ~16 while prefill's is 1 at large T — but decode needs
        // only ONE row (`1 * heads * ns_dec * hd` = about 1 MB), and multiplying that 8x by CTX
        // is a 64 GiB over-allocation at ctx=128k. It is the difference between 239 GiB (does not
        // fit alongside 57 GiB of weights) and 183 GiB (does).
        // Head-split (heads/tp) attention partials.
        opart: ac(
            b,
            "opart",
            (rows.max(64) * (c.heads / tp) * ns_pre * hd_max).max((c.heads / tp) * 64 * hd_max)
                as u64
                * F32,
        ),
        mlpart: ac(
            b,
            "mlpart",
            (rows.max(64) * (c.heads / tp) * ns_pre * 2).max((c.heads / tp) * 64 * 2) as u64 * F32,
        ),
        at: ac(b, "at", (rows * qd_max) as u64 * BF16),
        og: ac(b, "og", (rows * c.hidden) as u64 * BF16),
        gt: ac(b, "gt", (rows * inter_sh) as u64 * BF16),
        ut: ac(b, "ut", (rows * inter_sh) as u64 * BF16),
        fu: ac(b, "fu", (rows * inter_sh) as u64 * BF16),
        dg: ac(b, "dg", (rows * c.hidden) as u64 * BF16),
        // Only the LAST row's logits are ever read in prefill (i4 = a_row0 on the lm_head), so
        // this is 512 KB, not the 2.1 GB a full-T lm_head would need at ctx=4096. Vocab-column-
        // sharded. BATCH>1 decode reads B rows (one per sequence), so *dbatch; dbatch==1 identical.
        logits: ac(b, "logits", (dbatch * vocab_sh) as u64 * BF16),
        // Per-block argmax partials, one packed u64 each. Needs no zeroing between steps:
        // every block writes its own slot unconditionally. BATCH>1: [dbatch][AMAX_BLOCKS].
        // E5 PLOW_FUSE_ARGMAX: the fused lm_head epilogue (GemvArgmax) runs on all n_cu blocks,
        // so it writes n_cu partials — size for max(AMAX_BLOCKS, n_cu). Gated on the flag, so the
        // default blob is byte-identical.
        amax: ac(
            b,
            "amax.part",
            dbatch as u64 * fuse_argmax_parts(b.n_cu()) as u64 * 8,
        ),
        // TP peer-mapped partials (§7a) — only declared under sharding; tp==1 leaves them absent
        // so the tensor table (and the whole blob) stays byte-identical to the pre-TP path.
        og_tp: if tp > 1 {
            ac(b, "og_tp", (rows * c.hidden) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        dg_tp: if tp > 1 {
            ac(b, "dg_tp", (rows * c.hidden) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        // MoE decode scratch (B=1). Only for the 26B-A4B MoE path; TENSOR_NONE otherwise so the
        // dense 12B/31B blob stays byte-identical. Sized by ONE token (decode); mfu/part are [k,·].
        // moe_rows scales the per-token MoE scratch (1 for decode, chunk `rows` for grouped prefill).
        moe_h1: if c.moe {
            ac(b, "moe.h1", (moe_rows * c.hidden) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        moe_tab: if c.moe {
            ac(b, "moe.table", (moe_rows * c.top_k) as u64 * 8)
        } else {
            TENSOR_NONE
        },
        moe_rscore: if c.moe {
            ac(b, "moe.router_score", (dbatch * c.n_exp) as u64 * F32)
        } else {
            TENSOR_NONE
        },
        moe_xn2: if c.moe {
            ac(b, "moe.xn2", (moe_rows * c.hidden) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        moe_mfu: if c.moe {
            ac(b, "moe.mfu", (dbatch * c.top_k * c.moe_inter) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        moe_part: if c.moe {
            ac(b, "moe.part", (moe_rows * c.top_k * c.hidden) as u64 * F32)
        } else {
            TENSOR_NONE
        },
        moe_sum: if c.moe {
            ac(b, "moe.sum", c.hidden as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        moe_h2: if c.moe {
            ac(b, "moe.h2", c.hidden as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        moe_comb: if c.moe {
            ac(b, "moe.comb", (moe_rows * c.hidden) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        // Grouped-MoE prefill scratch: declared only when moe && moe_pf. Appended AFTER every existing
        // tensor so the decode-only path's handles (and thus its packet bytes) are unchanged.
        moe_meta: if moe_pf_on {
            ac(b, "moe.meta", (3 * c.n_exp + 2) as u64 * I32)
        } else {
            TENSOR_NONE
        },
        moe_rowtok: if moe_pf_on {
            ac(b, "moe.rowtok", total_pad as u64 * I32)
        } else {
            TENSOR_NONE
        },
        moe_rowpart: if moe_pf_on {
            ac(b, "moe.rowpart", total_pad as u64 * I32)
        } else {
            TENSOR_NONE
        },
        moe_rowgate: if moe_pf_on {
            ac(b, "moe.rowgate", total_pad as u64 * F32)
        } else {
            TENSOR_NONE
        },
        moe_fug: if moe_pf_on {
            ac(b, "moe.fug", (total_pad * c.moe_inter) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        moe_fuq: if moe_pf_on && w8a8 {
            ac(b, "moe.fuq", (total_pad * c.moe_inter) as u64) // e4m3, 1 byte/elt
        } else {
            TENSOR_NONE
        },
        moe_fus: if moe_pf_on && w8a8 {
            ac(b, "moe.fus", total_pad as u64 * F32)
        } else {
            TENSOR_NONE
        },
        // T8 w8a8 fp8 activation-quant scratch (uint8 xq [rows*width] + f32 a_scale [rows]). One pair
        // per activation width, reused across all layers. TENSOR_NONE unless w8a8.
        xqh: if w8a8 {
            ac(b, "xqh", (rows * c.hidden) as u64)
        } else {
            TENSOR_NONE
        },
        ash: if w8a8 {
            ac(b, "ash", rows as u64 * F32)
        } else {
            TENSOR_NONE
        },
        xqo: if w8a8 {
            ac(b, "xqo", (rows * qd_max) as u64)
        } else {
            TENSOR_NONE
        },
        aso: if w8a8 {
            ac(b, "aso", rows as u64 * F32)
        } else {
            TENSOR_NONE
        },
        xqi: if w8a8 {
            ac(b, "xqi", (rows * inter_sh) as u64)
        } else {
            TENSOR_NONE
        },
        asi: if w8a8 {
            ac(b, "asi", rows as u64 * F32)
        } else {
            TENSOR_NONE
        },
        kc: Vec::new(),
        vc: Vec::new(),
        kcs: Vec::new(),
        vcs: Vec::new(),
        lw: Vec::new(),
    };
    let mut t = t;
    for l in 0..c.layers {
        // Block extraction: only in-range layers ALLOCATE per-layer tensors; the
        // rest push TENSOR_NONE so the Tn vectors stay FULL length (emit_phase
        // indexes them by absolute `l`). Full model => in_block always true =>
        // byte-identical allocation.
        let in_block = block.contains(&(l as usize));
        let full = c.is_full[l as usize];
        // MIXED fp8-KV (PLOW_FP8_KV_FULL=1, beat-fp8-mma): e4m3 cache on the hd512 FULL layers
        // only. Sliding rings are window-bounded (tiny), so fp8 buys them nothing; keeping them
        // bf16 keeps their shipped prefill/decode arms byte-identical and lets the fp8 PREFILL
        // object build PIPE=1 (the px4 fp8-mma arm is hd512-only).
        let fp8_kv = fp8_kv && (full || !fp8_kv_full);
        let hd = if full { c.hd_full } else { c.hd_slide };
        let kvh = if full { c.kvh_full } else { c.kvh_slide };
        // KV CACHE, HEAD-MAJOR [kv_head][ctx][hd] (see dev_isa.h "THE KV CACHE IS HEAD-MAJOR").
        // Exactly the layout HeadNormRope writes (op_norm.h d_headnorm_rope, out_stride = kvr)
        // and the layout FlashDecode reads with kv_stride — so the cache write is not a separate
        // copy, it IS the norm's store. Head-major makes one head's rows contiguous for the
        // decode read; a byte-repack (token-major, or vLLM-style paging) is a measured null here.
        // Per-layer head split with SHARED-KV-HEAD REPLICATION (plans/tp-design.md §3a/§13.2).
        // Full layers have kvh_full=4 kv-heads: tp<=4 splits cleanly (kvh_local = kvh/tp); at tp=8 a
        // full layer's 4 kv-heads can't split 8 ways, so tp/kvh ranks SHARE (replicate) each kv-head
        // — each such rank owns 1 kv-head (kvh_local=1) plus its heads/tp q-heads. KV storage is then
        // 2x on full layers only (a minority), the design's chosen tradeoff. Sliding layers (16 kv)
        // still split cleanly at tp=8. Requires kvh|tp OR tp|kvh; anything else fails loudly.
        let kvh_local = kvh_local(kvh, tp, l);
        let (kvr, _) = kv_ring(full, ctx, c.window, max_chunk(c.window));
        let qd = (c.heads / tp) * hd; // column-parallel q output shard
        let kd = kvh_local * hd; // column-parallel k/v output shard (KV head-sharded/replicated)
                                 // fp8-KV: the cache is uint8 e4m3 (1 byte/elem, HALF the bf16 footprint) plus a per-row
                                 // f32 scale [kv_head][ctx] (head-major, same RING as the cache). Written by HeadNormRopeFp8,
                                 // read by FlashDecodeFp8 / FlashPrefillFp8. bf16 mode keeps the 2-byte cache and no scales.
        let kv_elt = if fp8_kv { 1 } else { BF16 };
        // BATCH>1 (serving pending #4): the KV cache is BATCH-MAJOR [dbatch][kv_head][ring][hd] —
        // each sequence owns its own ring. d_flash_decode/d_headnorm_rope index it per-batch as
        // ((b*n_kv_head+hkv)*kv_stride+row)*hd, so the per-batch stride is kv_head*ring*hd and the
        // tensor is dbatch* that. dbatch==1 => byte-identical to the single-sequence cache.
        let db = dbatch as u64;
        t.kc.push(if in_block {
            b.tensor(&format!("kv.{l}.k"), db * (kvr * kvh_local * hd) as u64 * kv_elt)
        } else {
            TENSOR_NONE
        });
        t.vc.push(if in_block {
            b.tensor(&format!("kv.{l}.v"), db * (kvr * kvh_local * hd) as u64 * kv_elt)
        } else {
            TENSOR_NONE
        });
        t.kcs.push(if fp8_kv && in_block {
            b.tensor(
                &format!("kv.{l}.k_scale"),
                db * (kvr * kvh_local) as u64 * F32,
            )
        } else {
            TENSOR_NONE
        });
        t.vcs.push(if fp8_kv && in_block {
            b.tensor(
                &format!("kv.{l}.v_scale"),
                db * (kvr * kvh_local) as u64 * F32,
            )
        } else {
            TENSOR_NONE
        });
        let prefix = c.prefix.clone();
        // All per-layer weight declarations funnel through these closures; gating
        // them on `in_block` drops out-of-range layers' weights from the tensor
        // table (the loader binds nothing for them). Full model => always alloc.
        let w = |b: &mut Builder, s: &str, sz: u64| {
            if in_block {
                b.tensor(&format!("{prefix}layers.{l}.{s}"), sz)
            } else {
                TENSOR_NONE
            }
        };
        // T6 L2: in fp8 mode BOTH prefill (GemmFp8) and decode (GemvFp8) consume the fp8 twins, so
        // the bf16 projection weight is DEAD — declaring it still made the loader stream 22 GiB of
        // never-read weight (fp8 pkt was 32.3 GiB = 22.2 bf16 + 10.1 fp8). Elide the bf16 projection
        // in fp8 mode (norms, embedding/lm_head, RoPE stay bf16). Verified: every w.wq..wd reference
        // (fused GemvQkv, bf16 GemmGlu/GemvGlu, bf16 proj arm) is under a `!fp8` guard.
        let wproj = |b: &mut Builder, s: &str, sz: u64| {
            if fp8 || !in_block {
                TENSOR_NONE
            } else {
                b.tensor(&format!("{prefix}layers.{l}.{s}"), sz)
            }
        };
        // A weight that only some architectures ship: declared only when present, else NONE — so
        // the runtime never tries to bind a tensor the checkpoint does not have.
        let wopt = |b: &mut Builder, present: bool, s: &str, sz: u64| {
            if present && in_block {
                b.tensor(&format!("{prefix}layers.{l}.{s}"), sz)
            } else {
                TENSOR_NONE
            }
        };
        let keqv = full && c.k_eq_v;
        let gemma = c.arch == Arch::Gemma4;
        // MoE fused expert weights: declared so the loader binds them by name and the harness
        // derives per-expert ewt bases from their device addresses. Not referenced as op operands
        // (the SM indexes them through the ewt pointer table), so the handles are discarded here.
        if c.moe && in_block {
            let gu_n = (c.n_exp * 2 * c.moe_inter) as u64;
            let dn_n = (c.n_exp * c.hidden) as u64;
            if fp8 {
                b.tensor(
                    &format!("fp8/{prefix}layers.{l}.experts.gate_up_proj"),
                    gu_n * c.hidden as u64,
                );
                b.tensor(
                    &format!("fp8/{prefix}layers.{l}.experts.gate_up_proj_scale"),
                    gu_n * F32,
                );
                b.tensor(
                    &format!("fp8/{prefix}layers.{l}.experts.down_proj"),
                    dn_n * c.moe_inter as u64,
                );
                b.tensor(
                    &format!("fp8/{prefix}layers.{l}.experts.down_proj_scale"),
                    dn_n * F32,
                );
            } else {
                w(b, "experts.gate_up_proj", gu_n * c.hidden as u64 * BF16);
                w(b, "experts.down_proj", dn_n * c.moe_inter as u64 * BF16);
            }
        }
        // FP8 decode twin of a projection: the quantized weight (1 byte/elt) under an "fp8/" name
        // the loader routes to the fp8 checkpoint, plus its per-output-channel f32 dequant scale
        // ("<name>_scale", [out]). `out` is the row count of the [out,in] weight = numel/in.
        let w8 = |b: &mut Builder, s: &str, numel: u64| -> u32 {
            if fp8 && in_block {
                b.tensor(&format!("fp8/{prefix}layers.{l}.{s}"), numel)
            } else {
                TENSOR_NONE
            }
        };
        let sc = |b: &mut Builder, s: &str, out: u64| -> u32 {
            if fp8 && in_block {
                b.tensor(&format!("fp8/{prefix}layers.{l}.{s}_scale"), out * F32)
            } else {
                TENSOR_NONE
            }
        };
        t.lw.push(LW {
            wq: wproj(b, "self_attn.q_proj.weight", (qd * c.hidden) as u64 * BF16),
            wk: wproj(b, "self_attn.k_proj.weight", (kd * c.hidden) as u64 * BF16),
            // Gemma full layers have NO v_proj: V is the raw k_proj output (k_eq_v). Llama/Qwen
            // always have a real v_proj. (fp8 mode elides the bf16 twin like the other projections.)
            wv: wopt(
                b,
                !keqv && !fp8,
                "self_attn.v_proj.weight",
                (kd * c.hidden) as u64 * BF16,
            ),
            wo: wproj(b, "self_attn.o_proj.weight", (c.hidden * qd) as u64 * BF16),
            wg: wproj(
                b,
                "mlp.gate_proj.weight",
                (inter_sh * c.hidden) as u64 * BF16,
            ),
            wu: wproj(b, "mlp.up_proj.weight", (inter_sh * c.hidden) as u64 * BF16),
            wd: wproj(
                b,
                "mlp.down_proj.weight",
                (c.hidden * inter_sh) as u64 * BF16,
            ),
            // fp8 twins (numel bytes) + scales ([out] f32). k_eq_v layers have no v_proj to quantize.
            // Dims use the TP-sharded shard extents (qd/kd/inter_sh); at tp==1 these equal the full
            // extents, so the single-GPU fp8 pkt is unaffected by the TP structure.
            wq8: w8(b, "self_attn.q_proj.weight", (qd * c.hidden) as u64),
            wk8: w8(b, "self_attn.k_proj.weight", (kd * c.hidden) as u64),
            wv8: if keqv {
                TENSOR_NONE
            } else {
                w8(b, "self_attn.v_proj.weight", (kd * c.hidden) as u64)
            },
            wo8: w8(b, "self_attn.o_proj.weight", (c.hidden * qd) as u64),
            wg8: w8(b, "mlp.gate_proj.weight", (inter_sh * c.hidden) as u64),
            wu8: w8(b, "mlp.up_proj.weight", (inter_sh * c.hidden) as u64),
            wd8: w8(b, "mlp.down_proj.weight", (c.hidden * inter_sh) as u64),
            sq: sc(b, "self_attn.q_proj.weight", qd as u64),
            sk: sc(b, "self_attn.k_proj.weight", kd as u64),
            sv: if keqv {
                TENSOR_NONE
            } else {
                sc(b, "self_attn.v_proj.weight", kd as u64)
            },
            so: sc(b, "self_attn.o_proj.weight", c.hidden as u64),
            sg: sc(b, "mlp.gate_proj.weight", inter_sh as u64),
            su: sc(b, "mlp.up_proj.weight", inter_sh as u64),
            sd: sc(b, "mlp.down_proj.weight", c.hidden as u64),
            g_in: w(b, "input_layernorm.weight", c.hidden as u64 * BF16),
            g_pa: w(b, "post_attention_layernorm.weight", c.hidden as u64 * BF16),
            // Gemma's sandwich has two extra norms; Llama/Qwen do not.
            g_pf: wopt(
                b,
                gemma,
                "pre_feedforward_layernorm.weight",
                c.hidden as u64 * BF16,
            ),
            g_po: wopt(
                b,
                gemma,
                "post_feedforward_layernorm.weight",
                c.hidden as u64 * BF16,
            ),
            qn: wopt(
                b,
                c.has_qk_norm,
                "self_attn.q_norm.weight",
                hd as u64 * BF16,
            ),
            kn: wopt(
                b,
                c.has_qk_norm,
                "self_attn.k_norm.weight",
                hd as u64 * BF16,
            ),
            // MoE (26B-A4B): router + FUSED 3D expert weights + the 3 extra sandwich norms. The
            // ewt pointer table is NOT a checkpoint tensor — it is a Persistent buffer the harness/
            // loader fills with per-expert bases derived from the two fused tensors' devp bases.
            rproj: wopt(
                b,
                c.moe,
                "router.proj.weight",
                (c.n_exp * c.hidden) as u64 * BF16,
            ),
            rscale: wopt(b, c.moe, "router.scale", c.hidden as u64 * BF16),
            rpes: wopt(b, c.moe, "router.per_expert_scale", c.n_exp as u64 * BF16),
            g_pf1: wopt(
                b,
                c.moe,
                "post_feedforward_layernorm_1.weight",
                c.hidden as u64 * BF16,
            ),
            g_pf2: wopt(
                b,
                c.moe,
                "post_feedforward_layernorm_2.weight",
                c.hidden as u64 * BF16,
            ),
            g_pre2: wopt(
                b,
                c.moe,
                "pre_feedforward_layernorm_2.weight",
                c.hidden as u64 * BF16,
            ),
            ewt: if c.moe && in_block {
                b.tensor(&format!("moe.ewt.{l}"), (c.n_exp * 2) as u64 * 8)
            } else {
                TENSOR_NONE
            },
            est: if c.moe && fp8 && in_block {
                b.tensor(&format!("moe.est.{l}"), (c.n_exp * 2) as u64 * 8)
            } else {
                TENSOR_NONE
            },
        });
    }
    t
}

const Q_TILE_ROWS: u32 = 8 * 32; // PLOW_WAVES * FA_BQ — keep in step with amd_common.h

/// LDS the GEMM arena holds, in halves. Mirrors `GM_LDS_HALVES` in `op_gemm.h`:
/// `2*(GM_BM+GM_BN)*(GM_BK+8)` = `2*(256+256)*72`. A GEMV can stage its A-operand on-chip only
/// if `M*K` fits here, which [`DevOp::GemvGlu`] requires (it re-reads x per output column).
const GM_LDS_HALVES: u64 = 2 * (256 + 256) * (64 + 8);

/// Largest prefill chunk. Mirrors `PLOW_MAX_CHUNK` in `dev_isa.h`.
///
/// This is the ONLY row count any single program ever processes: chunked prefill never emits a
/// chunk bigger than this, and decode is one row. So it caps BOTH the bucket ladder (a program
/// for T > MAX_CHUNK can never be invoked) and every ACTIVATION tensor (they hold the current
/// chunk, not the context -- only the KV cache spans the context).
const MAX_CHUNK_MAX: u32 = 8192;

/// Smallest chunk the window-derived default will pick (the bucket ladder's floor).
const MAX_CHUNK_MIN: u32 = 128;

/// Default prefill chunk **derived from the model**, not a constant.
///
/// The chunk sizes the sliding-layer KV ring (`ring = next_pow2(window + chunk - 1)`), so on a
/// windowed model an oversized chunk inflates the ring for no benefit: Gemma-4 (window 1024)
/// rang 16384 rows at the old flat 8192 default purely because of the chunk. Picking the chunk
/// from the window puts the ring at `2 * next_pow2(window)` — the smallest it can be without
/// violating the `ring >= window + chunk - 1` invariant.
///
/// MEASURED (Gemma-4-12B, window 1024, RTX 5090, fp8 weights+KV, B=8, ctx 8192):
///
/// | chunk | KV cache  | activations | 4096-tok-prompt prefill run |
/// |-------|-----------|-------------|-----------------------------|
/// | 8192  | 10.66 GiB | 1.84 GiB    | 16.13 s                     |
/// | 1024  |  3.04 GiB | 0.30 GiB    | 16.45 s                     |
///
/// 3.5x less KV and 6x less activation for 2% on a deliberately prefill-dominated shape
/// (4096 in / 32 out), which is the worst case since it is what pays the extra launches.
///
/// **`window == 0` (all-global, e.g. Llama-style) keeps [`MAX_CHUNK_MAX`].** `kv_ring` returns
/// `(ctx, MASK_NONE)` for full-attention layers, so the chunk does not size their cache at all
/// — lowering it there would buy no KV and only cost prefill launches. This is why the default
/// is a function of the model rather than the flat 1024 that Gemma alone would suggest.
fn default_chunk(window: u32) -> u32 {
    if window == 0 {
        MAX_CHUNK_MAX
    } else {
        window.next_power_of_two().clamp(MAX_CHUNK_MIN, MAX_CHUNK_MAX)
    }
}

/// Largest prefill chunk for this compile. `PLOW_MAX_CHUNK` lowers it to buy back
/// sliding-layer KV: the ring is sized `window + chunk - 1` (see [`kv_ring_rows`]), so on a
/// model whose window is far below the chunk it is the CHUNK that sets the ring, not the
/// model. Gemma-4 (window 1024) at the 8192 default rings 16384 rows = 320 KiB/token * 16384
/// = 5.0 GiB/seq; chunk 1024 rings 2048 and costs 0.625 GiB/seq — 8x, for more prefill
/// launches on long prompts.
///
/// Must be a power of two and no larger than [`MAX_CHUNK_MAX`] (the bucket ladder tops out
/// there). Unset = [`default_chunk`] for this model's window; pass `PLOW_MAX_CHUNK=8192` to
/// reproduce the blob this emitter produced before the default became window-derived.
fn max_chunk(window: u32) -> u32 {
    let v = std::env::var("PLOW_MAX_CHUNK")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or_else(|| default_chunk(window));
    assert!(
        v.is_power_of_two() && v <= MAX_CHUNK_MAX,
        "PLOW_MAX_CHUNK {v} must be a power of two <= {MAX_CHUNK_MAX}"
    );
    v
}

/// SLIDING-WINDOW KV RING. Mirrors `PLOW_KV_RING` / `PLOW_KV_MASK_NONE` in `dev_isa.h`.
///
/// Rows a sliding layer's ring needs for a `(window, chunk)` pair, from the dev_isa.h
/// invariant `ring >= window + chunk - 1`, rounded up to a power of two (the kernels index
/// `row & (ring-1)`). The kernel reads the mask per op out of the packet, so the ring size is
/// DATA, not a kernel constant — shrinking it here needs no cubin rebuild.
///
/// `window 1024 + chunk 8192` -> 16384, the historical `KV_RING`, so the default is
/// byte-identical. Every window <= 8193 lands on 16384 at the default chunk.
fn kv_ring_rows(window: u32, chunk: u32) -> u32 {
    (window + chunk - 1).next_power_of_two()
}
const KV_MASK_NONE: u32 = 0xFFFF_FFFF;

/// How many rows a layer's KV cache actually needs, and the mask its row index is ANDed with.
///
/// A sliding layer never looks back further than `window`, so it needs a RING rather than the
/// full context — at ctx=128k that is 100 GiB of never-read cache. A full-attention layer keeps
/// a linear cache and gets `0xFFFFFFFF`, so the AND in the kernels is a no-op there.
///
/// The ring must be at least `window + max_chunk - 1`: a prefill chunk's queries span
/// `[c0, c0+C)` and between them read `[c0-window+1, c0+C-1]`, and the chunk writes all C of its
/// rows before flash reads any of them. See `PLOW_KV_RING` in dev_isa.h. It is a power of two so
/// the kernels can AND rather than divide.
fn kv_ring(full: bool, ctx: u32, window: u32, chunk: u32) -> (u32, u32) {
    if full {
        (ctx, KV_MASK_NONE)
    } else {
        let r = ctx.min(kv_ring_rows(window, chunk)); // no point ringing a cache smaller than the ring
        // `row & (r-1)` is a modulo ONLY when r is a power of two. For a non-pow2 r the AND
        // aliases rows to WRONG (in-bounds) rows — silent corruption. All shipped ctx are
        // pow2; make the invariant loud (leak-audit finding #6).
        assert!(r.is_power_of_two(), "kv_ring size {r} (ctx {ctx}) must be a power of two");
        // The wrap invariant binds ONLY when the ring is shorter than the context. At r == ctx
        // no position is ever reused (`row & (ctx-1) == row` for row < ctx), so the cache is
        // linear-equivalent and `window + chunk - 1` is vacuous — which is why a small-ctx
        // build is safe despite r < window + chunk - 1.
        assert!(
            r == ctx || r >= window + chunk - 1,
            "sliding ring {r} < window {window} + chunk {chunk} - 1 (ctx {ctx}): a chunk's rows \
             would wrap onto their own history — a silent wrong answer, not a crash"
        );
        (r, r - 1)
    }
}

/// This rank's local KV-head count under TP with SHARED-KV-HEAD REPLICATION (plans/tp-design.md
/// §3a/§13.2). Two regimes, both keep every rank's q-heads mapped to a kv-head it owns:
///   - `tp <= kvh_g` (clean split): each rank owns `kvh_g/tp` distinct kv-heads.
///   - `tp  > kvh_g` (replication): `tp/kvh_g` ranks share (replicate) one kv-head; each owns 1.
/// Anything else (neither divides) is unsupported and fails loudly rather than shard silently wrong.
fn kvh_local(kvh_g: u32, tp: u32, l: u32) -> u32 {
    if tp <= kvh_g {
        assert_eq!(
            kvh_g % tp,
            0,
            "--tp {tp} must divide layer {l}'s kv-heads {kvh_g} (§3a/§13.2)"
        );
        kvh_g / tp
    } else {
        assert_eq!(
            tp % kvh_g,
            0,
            "--tp {tp} must be a multiple of layer {l}'s kv-heads {kvh_g} for shared-kv-head \
             replication (§3a/§13.2)"
        );
        1
    }
}

/// Which `d_gemv` workgroups produce output columns `[c0, c1)`?
///
/// Requires `GV_BLOCKED=1` in `op_gemm.h`, where workgroup `s` owns the contiguous run
/// `[s*per, s*per+per)`, `per = ceil(N/nblk)`. Under the DEFAULT interleaved assignment this
/// function would be a lie: a workgroup's columns are `[8s, 8s+8) (mod nblk*8)`, scattered
/// across all of N, so 256 consecutive columns touch EVERY workgroup and the answer is always
/// "all of them" (measured: 128 of 128).
fn gemv_wgs_for_cols(n: u32, nblk: u32, c0: u32, c1: u32) -> Vec<u32> {
    let per = n.div_ceil(nblk);
    (c0 / per..=(c1 - 1) / per).filter(|&w| w < nblk).collect()
}

/// The work items (`token * nhead + head`) that `headnorm_rope` workgroup `j` runs.
///
/// `d_headnorm_rope` walks `for (w = slice*PLOW_WAVES + wave; w < total; w += nblk*PLOW_WAVES)`,
/// so workgroup `j` owns the items whose wave slot lands in `[8j, 8j+8)`.
fn headnorm_items(nblk: u32, total: u32, j: u32) -> Vec<u32> {
    (0..total)
        .filter(|&w| (w % (nblk * WAVES)) / WAVES == j)
        .collect()
}

/// The headnorm workgroup that produces item `w`.
fn headnorm_wg_of(nblk: u32, w: u32) -> u32 {
    (w % (nblk * WAVES)) / WAVES
}

const WAVES: u32 = 8; // PLOW_WAVES

/// MoE GLU → Down dependency map: Down block `b` only needs the GLU blocks that produce
/// the slots it reads. Without this, Down waits for ALL GLU blocks (coarse gate), wasting
/// ~2.6M cycles per layer on the critical path.
///
/// Layout: GLU produces flat `[k * I_moe]` outputs distributed round-robin across `nblk`
/// blocks (per_g = ceil(k*I_moe/nblk) per block). Down produces flat `[k * H]` outputs
/// similarly (per_d = ceil(k*H/nblk)). Down block `b` handles flat indices
/// `[b*per_d, (b+1)*per_d)`. For flat index `f`, `slot = f / H`. Down reads
/// `fu[slot*I_moe..(slot+1)*I_moe]` — so it depends on GLU blocks covering that range.
fn moe_down_fine_map(top_k: u32, i_moe: u32, hidden: u32, nblk: u32) -> Vec<Vec<u32>> {
    let total_glu = top_k * i_moe;
    let total_down = top_k * hidden;
    let per_g = total_glu.div_ceil(nblk);
    let per_d = total_down.div_ceil(nblk);
    (0..nblk)
        .map(|b| {
            let f0 = b * per_d;
            let f1 = ((b + 1) * per_d).min(total_down);
            if f0 >= total_down {
                return vec![];
            }
            let slot_lo = f0 / hidden;
            let slot_hi = (f1 - 1) / hidden;
            let glu_lo = slot_lo * i_moe;
            let glu_hi = (slot_hi + 1) * i_moe;
            let g_first = glu_lo / per_g;
            let g_last = (glu_hi - 1) / per_g;
            (g_first..=g_last.min(nblk - 1)).collect()
        })
        .collect()
}

/// The flash → merge edge is SPARSE, and today it is gated as if it were dense.
///
/// `flash_*` splits its work into `q_tiles * n_head * nsplit` items, item
/// `w = (qt * n_head + h) * nsplit + sp`, run by workgroup `w % nblk_f` (the kernels walk
/// `for (w = slice; w < n_work; w += nblk)`). `flash_merge` splits into `n_bh = n_batch *
/// n_head` items, item `w = b * n_head + h`, run by workgroup `w % nblk_m`. Merge item
/// `(b, h)` folds the `nsplit` partials of that same `(b, h)` and touches nothing else.
///
/// So a merge workgroup needs a handful of flash slices — at Gemma-31B decode, **8 of 256**.
/// Coarse counters make it wait for all 256, and the trace says that wait costs 0.83 ms of a
/// 16.9 ms token: the gate opens on the slowest CU, and 256 CUs doing this work spread over
/// 9.6-16.6 us.
///
/// `rows_per_item` is how many query rows one flash work item covers: `Q_TILE_ROWS` in
/// prefill (flash tiles the q axis) and 1 in decode (there is one query row).
fn flash_merge_map(
    n_bh: u32,
    nsplit: u32,
    rows_per_item: u32,
    n_head: u32,
    nblk_f: u32,
    nblk_m: u32,
) -> Vec<Vec<u32>> {
    (0..nblk_m)
        .map(|j| {
            let mut s: Vec<u32> = (0..n_bh)
                .filter(|w| w % nblk_m == j) // the merge items THIS workgroup runs
                .flat_map(|w| {
                    let (b, h) = (w / n_head, w % n_head);
                    let qt = b / rows_per_item; // which flash q-tile covers this row
                    (0..nsplit).map(move |sp| ((qt * n_head + h) * nsplit + sp) % nblk_f)
                })
                .collect();
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect()
}

/// Emit the layer all-reduce for a row-parallel producer (o_proj/down), all-reduce #1/#2.
/// PREFILL uses the TWO-SHOT (reduce-scatter + all-gather): the [T,hidden] partial is
/// bandwidth-bound, so ~N/2× less fabric than the one-shot (plans/tp-prefill.md §4). DECODE
/// keeps the one-shot — its tiny [1,hidden] message is latency-bound, so a single sync wins.
/// Two-shot consumes TWO xctr gate ids (reduce-scatter + all-gather rendezvous); one-shot
/// consumes one. `slot` is the byte offset of this collective's partial slot (0 or slot_b).
/// Result is BIT-IDENTICAL across the two variants (same f32-acc, r=0..N−1 order).
#[allow(clippy::too_many_arguments)]
fn emit_xreduce(
    b: &mut Builder,
    xgate: &mut u32,
    decode: bool,
    xr_cus: &[u32],
    dep: u32,
    out: u32,
    xr_elems: u32,
    tp: u32,
    slot: u32,
) -> u32 {
    if decode {
        let gate = *xgate;
        *xgate += 1;
        b.emit(DevOp::XReduce, xr_cus.to_vec(), &[dep], |d| {
            d.t[0] = out; // reduced [1,hidden] result (local)
            d.i[0] = xr_elems; // elements to reduce (decode: hidden)
            d.i[1] = tp; // n_gpu
            d.i[2] = slot; // partial slot byte offset (§7a)
            d.i[3] = gate; // xctr gate id (unique per collective)
        })
    } else {
        let gate_rs = *xgate;
        *xgate += 1;
        let gate_ag = *xgate;
        *xgate += 1;
        b.emit(DevOp::XReduceTwoShot, xr_cus.to_vec(), &[dep], |d| {
            d.t[0] = out; // reduced [t,hidden] result (local)
            d.i[0] = xr_elems; // elements to reduce (t*hidden)
            d.i[1] = tp; // n_gpu
            d.i[2] = slot; // partial slot byte offset (§7a)
            d.i[3] = gate_rs; // reduce-scatter rendezvous gate id
            d.i[4] = gate_ag; // all-gather rendezvous gate id
        })
    }
}

/// Which program `emit_phase` is building. This used to be a `decode: bool`, which conflated two
/// INDEPENDENT axes — and the whole point of the enum is to pull them apart:
///
///   * **shape** — one query row, KV *append* + ring mask, decode's nsplit. (`decode_shape`)
///   * **kernel family** — the GEMV opcodes and the fusions that only exist because of them
///     (fold / fuse_norm / gfuse / fuse_qkv / glu_fused), plus flash-DECODE attention. (`gemv`)
///
/// `Decode` is (shape, gemv) = (true, true) and `Prefill` is (false, false) — the only two
/// combinations that existed before — so both stay BYTE-IDENTICAL to the pre-enum emitter.
/// `DecodeTiled` is the new third corner: (true, false), a decode-shaped step built from prefill
/// kernels. See plans/decode-tiled.md.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Prefill,
    Decode,
    /// Decode shape, prefill kernels: tiled GEMM + FlashPrefill at one query row. Targets long
    /// context, where GEMV does not scale with batch (its split is over N, not M) and FlashDecode
    /// caps at n_cu. **Requires prefill opcodes in the interpreter** — the sm_120 build traps on
    /// FlashPrefill(11)/GemmSmall(14)/GemmMed(15)/GemmGlu(20), so this mode is AMD-only today.
    DecodeTiled,
}

impl Mode {
    /// One query row, KV append + ring mask, decode's nsplit and one-shot all-reduce.
    fn decode_shape(self) -> bool {
        self != Mode::Prefill
    }
    /// The GEMV opcode family and every fusion that exists only to serve it, plus flash-decode.
    fn gemv(self) -> bool {
        self == Mode::Decode
    }
}

/// Split router is DEFAULT-ON: the 128-expert score GEMV runs on 16 CTAs (8 experts/CTA)
/// instead of serializing on one CTA. The fused single-CTA path is the escape hatch.
fn gemma_moe_router_split_enabled() -> bool {
    std::env::var("GLM_ROUTER_OLD").ok().as_deref() != Some("1")
        && std::env::var("PLOW_GEMMA_MOE_ROUTER_FUSED").ok().as_deref() != Some("1")
}

/// `nrow` = decode batch B: the score work space is the (row, expert) PAIR space, so B rows
/// scale the useful CTA count (16 CTAs at B=1/E=128, capped at n_cu from B=12 up).
fn gemma_moe_router_split_plan(n_cu: u32, n_exp: u32, nrow: u32) -> Option<(u32, DevOp)> {
    if !gemma_moe_router_split_enabled() {
        return None;
    }
    let max_useful = (nrow * n_exp).div_ceil(8).max(1).min(n_cu.max(1));
    let blocks = std::env::var("PLOW_GEMMA_MOE_ROUTER_BLOCKS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(max_useful)
            .clamp(1, max_useful);
    let op = if std::env::var("PLOW_GEMMA_MOE_ROUTER_EXACT").ok().as_deref() == Some("1") {
        DevOp::MoeRouterGemmaScore
    } else {
        DevOp::MoeRouterGemmaScoreFast
    };
    Some((blocks, op))
}

#[allow(clippy::too_many_arguments)]
fn emit_gemma_moe_router(
    b: &mut Builder,
    dep: u32,
    table: u32,
    resid: u32,
    proj: u32,
    scale: u32,
    pes: u32,
    score: u32,
    hidden: u32,
    n_exp: u32,
    top_k: u32,
    root: f32,
    eps: f32,
    split_plan: Option<(u32, DevOp)>,
    nrow: u32,
) -> u32 {
    // BATCH B>1: the batch row count rides a spare immediate, emitted ONLY when B>1 so the
    // B=1 instruction bytes are untouched (the kernels read 0 as "one row").
    let nb = if nrow > 1 { nrow } else { 0 };
    if let Some((blocks, score_op)) = split_plan {
        assert_ne!(
            score, TENSOR_NONE,
            "split Gemma router requires f32 score scratch"
        );
        let c_score = b.emit(
            score_op,
            (0..blocks).collect(),
            &[dep],
            |d| {
                d.t[0] = score;
                d.t[1] = resid;
                d.t[2] = proj;
                d.t[3] = scale;
                d.i[0] = hidden;
                d.i[1] = n_exp;
                d.i[2] = nb;
                d.f[0] = root;
                d.f[1] = eps;
            },
        );
        // top-k is serial per row; give it one CTA per row so B rows run concurrently.
        let topk_cus: Vec<u32> = (0..nrow.max(1)).collect();
        b.emit(DevOp::MoeRouterGemmaTopk, topk_cus, &[c_score], |d| {
            d.t[0] = table;
            d.t[1] = score;
            d.t[2] = pes;
            d.i[1] = n_exp;
            d.i[2] = top_k;
            d.i[3] = nb;
        })
    } else {
        b.emit(DevOp::MoeRouterGemma, vec![0], &[dep], |d| {
            d.t[0] = table;
            d.t[1] = resid;
            d.t[2] = proj;
            d.t[3] = scale;
            d.t[4] = pes;
            d.i[0] = hidden;
            d.i[1] = n_exp;
            d.i[2] = top_k;
            d.i[3] = nb;
            d.f[0] = root;
            d.f[1] = eps;
        })
    }
}

/// Emit one phase. `t == 1 && decode` is the decode step; otherwise a prefill bucket.
fn emit_phase(
    b: &mut Builder,
    c: &Cfg,
    ls: &[f32],
    n: &Tn,
    t: u32,
    ctx: u32,
    mode: Mode,
    n_cu: u32,
    kv_rows: &mut Vec<u32>,
    fp8: bool,
    w8a8: bool,
    fp8_kv: bool,
    fp8_kv_full: bool,
    block: std::ops::Range<usize>,
    block_mode: bool,
) {
    // The two axes the old `decode` bool used to carry at once. Every former use site below is
    // now one or the other: `decode` for shape, `gemv_family` for kernel family. (Not `gemv` —
    // the `hn_dep` closure below already binds a `gemv: u32` parameter that would shadow it.)
    let decode = mode.decode_shape();
    let gemv_family = mode.gemv();
    let all = b.all();
    // TENSOR-PARALLEL local shards (plans/tp-design.md §3). For tp==1 these equal the full dims,
    // so the whole emit is byte-identical to the pre-TP path; for tp>1 (decode only) every head-,
    // intermediate- and vocab-dimensioned op runs 1/N wide, and o_proj/down get an XReduce.
    let tp = c.tp;
    let heads = c.heads / tp; // this rank's q-heads
    let inter_l = c.inter / tp; // this rank's gate/up/down intermediate lanes
    let vocab_l = c.vocab; // lm_head REPLICATED under TP (Phase 2); see declare() note above
    let mut xgate: u32 = 0; // xctr gate-id allocator for XReduce (unique per collective)
                            // XReduce runs on a REDUCED CU set (F-lever, plans/tp-design.md §8b/§10). The all-reduce is a
                            // tiny memory-bound sum over the H-vector, but EVERY participating workgroup takes a SYSTEM-scope
                            // acquire (a full L2 invalidate) per collective — 2L=120 collectives/token at 31B. Fewer CUs =>
                            // fewer redundant system-acquires and less cross-XCD invalidation, at no bandwidth cost (H=5376
                            // saturates on a handful of workgroups). Default keeps `all` (byte-identical to Phase-2); set
                            // PLOW_XR_CUS=k to cap it (measured lever for the TP=8 NUMA-crossing all-reduce). tp==1 unused.
    let xr_cus: Vec<u32> = {
        let k = std::env::var("PLOW_XR_CUS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(n_cu)
            .clamp(1, n_cu);
        (0..k).collect()
    };
    // TP prefill (plans/tp-prefill.md): the all-reduce partials are [T, hidden], not decode's
    // [1, hidden], so the XReduce reduces `xr_elems = t*hidden` elements. The two peer-scratch
    // partial slots (og_tp/dg_tp, §7a) must not overlap: partial_A occupies [0, rows_max*hidden*2),
    // partial_B starts at `slot_b = rows_max*hidden*2`. rows_max = the largest chunk (= og_tp's
    // declared row count in declare()), so the slot is IDENTICAL across every prefill bucket AND
    // the decode program — the host binds dg_tp at that one fixed offset for all of them. For
    // decode t==1 so xr_elems==hidden and the layout is a superset of the old decode path.
    let rows_max = ctx.min(max_chunk(c.window));
    let xr_elems = t * c.hidden;
    let slot_b = rows_max * c.hidden * BF16 as u32;
    let rows: Vec<u32> = (0..t.min(n_cu).max(1)).collect();
    // Elementwise ops sized to their ACTUAL work, not handed the whole machine.
    //
    // A decode residual is 5376 elements. On 256 CUs that is 21 elements each -- the op is
    // pure gate overhead, and all 256 workgroups still have to be counted into the barrier.
    // One workgroup (512 threads x 8) covers it. Fewer participants, cheaper gate, less
    // counter contention.
    let elem = |n: u32| -> Vec<u32> { (0..n.div_ceil(512 * 8).max(1).min(n_cu)).collect() };
    let ns = if gemv_family {
        n_cu.div_ceil(heads).max(1)
    } else {
        n_cu.div_ceil((t.div_ceil(Q_TILE_ROWS) * heads).max(1))
            .max(1)
    };
    // When nsplit==1 there is nothing for d_flash_merge to combine: flash_prefill normalizes
    // in its own epilogue and writes the final bf16 straight to n.at, and the merge op is not
    // emitted at all. Prefill-only (decode always keeps ns>1).
    let fused = !gemv_family && ns == 1;

    let escale = c.emb_scale;
    // Block mode: no token embedding — `act.x` is uploaded by the harness (the
    // residual-stream input), so Embed would overwrite it. The first in-block
    // layer's RmsNorm reads `act.x` directly (its dep is `&[]`; see below).
    let mut dep = if block_mode {
        0u32
    } else {
        b.emit(DevOp::Embed, rows.clone(), &[], |d| {
            d.t[0] = n.x;
            d.t[1] = n.emb;
            d.t[2] = n.ids;
            d.i[0] = t;
            d.i[1] = c.hidden;
            d.f[0] = escale;
        })
    };

    // In decode, every projection is a GEMV (M=1): a 32x32 matrix core would run with 1 of
    // 32 M-lanes live, and the step is bandwidth-bound on the 57 GiB of weights anyway.
    // In DECODE the RMSNorm is folded into the consuming GEMV (norm mode 2: the GEMV computes
    // the row RMS itself, from the x it already staged in LDS). That deletes the RMSNORM
    // packet, its gate, AND its single-CU serialisation -- a decode norm is a row reduction,
    // so exactly ONE workgroup could do it while the other 255 waited on the counter.
    //
    // In PREFILL the norm stays its own packet: it has T rows, so it already parallelises, and
    // folding it into the GEMM (GEMM_NORM) is a measured LOSS -- the A tile is re-fetched once
    // per N-tile, so the per-element norm work gets multiplied by N/BN.
    let eps = c.eps;
    // `w8`/`scale` are the fp8 twin of the bf16 weight `w` and its per-channel dequant scale; they
    // are used ONLY on the decode fp8 path (DevOp::GemvFp8). Prefill and bf16 decode ignore them.
    // `xq`/`ascale_t` are the T8 w8a8 fp8-quantized activation twin of `a` and its per-row a_scale;
    // they are TENSOR_NONE (and ignored) off the w8a8 path. On the w8a8 path the caller has already
    // emitted the shared QuantFp8 (once per activation site) and threaded its id into `deps`.
    let proj = |b: &mut Builder,
                out: u32,
                a: u32,
                w: u32,
                w8: u32,
                scale: u32,
                xq: u32,
                ascale_t: u32,
                m: u32,
                nn: u32,
                k: u32,
                gamma: u32,
                cus: Vec<u32>,
                deps: &[u32]|
     -> u32 {
        if gemv_family && fp8 {
            return b.emit(DevOp::GemvFp8, cus, deps, |d| {
                d.t[0] = out;
                d.t[1] = a;
                d.t[2] = w8;
                d.t[5] = scale;
                d.i[0] = m;
                d.i[1] = nn;
                d.i[2] = k;
                d.i[4] = 0;
            });
        }
        // PREFILL fp8 tiled GEMM. Two builds share the GEMM_FP8 opcodes; the interp cubin picks the
        // kernel by PLOW_NV_W8A8. T6 w8a16 (default cubin): bf16 activation (t1=a), e4m3 weight (t2)
        // + per-channel dequant scale (t4). T8 w8a8 (PLOW_NV_W8A8 cubin, PLOW_W8A8 emit): BOTH
        // operands e4m3 — t1=xq (per-row-quantized activation), t3=a_scale, t2=w8, t4=w_scale — true
        // mma.sync.m16n8k32. The opcode tracks whatever tile pick_tile would have chosen for bf16.
        if !gemv_family && fp8 {
            let op = match pick_tile(m, nn, k, n_cu) {
                DevOp::GemmMed => DevOp::GemmMedFp8,
                DevOp::GemmSmall => DevOp::GemmSmallFp8,
                _ => DevOp::GemmFp8,
            };
            return b.emit(op, cus, deps, |d| {
                d.t[0] = out;
                d.t[2] = w8;
                d.t[4] = scale;
                if w8a8 {
                    d.t[1] = xq;
                    d.t[3] = ascale_t;
                } else {
                    d.t[1] = a;
                }
                d.i[0] = m;
                d.i[1] = nn;
                d.i[2] = k;
                d.i[4] = 0;
            });
        }
        let fold = gemv_family && gamma != TENSOR_NONE;
        let op = if gemv_family {
            DevOp::Gemv
        } else {
            pick_tile(m, nn, k, n_cu)
        };
        b.emit(op, cus, deps, |d| {
            d.t[0] = out;
            d.t[1] = a;
            d.t[2] = w;
            if fold {
                d.t[4] = gamma;
            }
            d.i[0] = m;
            d.i[1] = nn;
            d.i[2] = k;
            d.i[3] = if fold { 2 } else { 0 };
            d.i[4] = 0;
            d.f[0] = eps;
        })
    };

    // T8 w8a8: emit the ONE shared per-row fp8 activation quant (DevOp::QuantFp8) that a group of
    // GEMMs reading the same activation depends on — the linchpin of correctness. A per-proj quant
    // would race (q's quant would overwrite the xq that k/v read); the single shared quant is
    // required, not merely an optimization. `after` is the producer of `src` (the norm/attn output);
    // the returned id is what the consuming GEMMs must wait on. Off the w8a8 path it is inert and
    // returns `after`, so every caller can thread it uniformly. Row-sliced across `rows` blocks.
    let quant = |b: &mut Builder, xq: u32, ascale_t: u32, src: u32, k: u32, after: u32| -> u32 {
        if !w8a8 {
            return after;
        }
        b.emit(DevOp::QuantFp8, rows.clone(), &[after], |d| {
            d.t[0] = xq;
            d.t[1] = src;
            d.t[2] = ascale_t;
            d.i[0] = t;
            d.i[1] = k;
        })
    };

    // Qwen/Llama PRE-NORM decode fuses each (residual add, RMSNorm) pair into ONE AddNorm packet
    // (see the AddNorm emits in the loop). Deletes 72 packets/token and, more importantly, 72
    // global gates off the critical path — decode here is fixed per-gate tax, not weight streaming.
    let fuse_norm = c.arch != Arch::Gemma4 && gemv_family;
    // Gemma SANDWICH decode fuses each (NormResidual, following RMSNorm) pair into ONE
    // NormResidualNorm packet (Experiment N1) — the narrow→narrow successor to AddNorm. Same two
    // sites as fuse_norm (post-attn→pre-ffn, and end-of-layer→next input norm), but the residual is
    // a post-normed sandwich add with a per-layer scale, not a plain sum. Deletes a gate + an HBM
    // round trip per fused pair. Decode only; prefill keeps the split (T rows parallelise the norm).
    let gfuse = c.arch == Arch::Gemma4 && gemv_family;

    for l in block.clone() {
        let full = c.is_full[l];
        // MIXED fp8-KV (PLOW_FP8_KV_FULL=1): per-layer effective flag — see declare(). Ops keyed
        // on it (HeadNormRope[Fp8], FlashDecode[Fp8], FlashPrefill[Fp8], the fp8-tuned nsplit
        // gates) all follow the LAYER's cache dtype.
        let fp8_kv = fp8_kv && (full || !fp8_kv_full);
        let hd = if full { c.hd_full } else { c.hd_slide };
        // this rank's kv-heads, with shared-kv-head replication for tp > kvh (§3a/§13.2, kvh_local).
        let kvh = kvh_local(if full { c.kvh_full } else { c.kvh_slide }, tp, l as u32);
        let qd = heads * hd; // column-parallel q output shard
        let kd = kvh * hd; // column-parallel k/v output shard
        let (cs, sn) = if full {
            (n.cos_f, n.sin_f)
        } else {
            (n.cos_s, n.sin_s)
        };
        let win = if full { 0 } else { c.window };
        let w = &n.lw[l];
        // k_eq_v is Gemma-full-layer only; Llama/Qwen always have a real v_proj even though every
        // layer is "full". skip_norm bypasses the RMS in HeadNormRope: Llama has no q/k norm and
        // neither model norms V.
        let keqv = full && c.k_eq_v;
        let qk_skip: u32 = if c.has_qk_norm { 0 } else { 1 };
        let v_skip: u32 = if c.has_v_norm { 0 } else { 1 };
        // FUSED Q|K|V (decode, real v_proj): one GEMV packet computes all three projections, on
        // all CUs. Two fewer gates than split3 AND uniform fill instead of the 171/42/43 CU split.
        // Gemma's k_eq_v layers keep the old path (no v_proj to fuse). See DevOp::GemvQkv.
        // FP8 has no QKV-fusion arm (opcode 26 deferred): q/k/v run as three separate GEMV_FP8.
        // T11 packet-reduction probe: PLOW_NO_FUSE_QKV=1 reverts to the historical split3 path
        // (q/k/v as three separate bf16 Gemv packets = +2 packets/layer, uneven CU fill). Tokens
        // are bit-identical (each output column is the same per-column dot). Off by default =>
        // byte-identical stream. Measures the marginal TPOT cost of a 2-gate/layer reduction.
        let fuse_qkv = gemv_family
            && !keqv
            && !fp8
            && std::env::var("PLOW_NO_FUSE_QKV").ok().as_deref() != Some("1");

        // GQA FUSION changes the decode split, and the two have to agree or the machine idles.
        //
        // A fused flash-decode work item is (kv_head, split), not (query_head, split) — it reads
        // each KV row ONCE and dots it against all GF query heads sharing it. That divides
        // n_work by GF, so `nsplit` must be multiplied by it to keep 256 work units on 256 CUs.
        // The kernel picks GF from head_dim (PLOW_FA_GF in dev_isa.h); here we derive nsplit from
        // kv_heads, which is the same statement from the other side.
        //
        // It is PER LAYER because kv_heads is: 16 on a sliding layer (GQA 2), 4 on a full one
        // (GQA 8). A single nsplit for both would leave the full layers on 4 of 256 CUs.
        // The sliding layers' cache is a RING; the full layers' is linear. `kvm` is 0xFFFFFFFF
        // for a full layer, so the AND in the kernels is a no-op there. See kv_rows().
        let (kvr, kvm) = kv_ring(full, ctx, c.window, max_chunk(c.window));
        // GF is the flash-decode GQA fusion factor: query heads carried by ONE work item, and it is
        // the KERNEL constant PLOW_FA_GF(hd) = PLOW_FA_GF_FULL (default 2) — NOT 8. The compiler and
        // kernel must agree (dev_isa.h). GF=2 fuses sliding layers fully (GQA 2) and full layers
        // partially (GQA 8 -> reads each row 4x). Under tp=8 shared-kv-head replication a full layer
        // is GQA 4 locally, still a clean multiple of GF=2. The binding invariant is gqa_local % GF.
        let gf = fa_gf_full(); // MUST track the kernel's PLOW_NV_FA_GF_FULL; see fa_gf_full()
        assert_eq!(
            (heads / kvh) % gf,
            0,
            "layer {l}: GF {gf} must divide GQA {}",
            heads / kvh
        );
        // n_work = n_head/GF * nsplit. Filling all 256 CUs would want nsplit = n_cu*GF/n_head
        // (= 64 on a full layer), and that is WRONG in both directions: it fragments flash into
        // 52-row work items whose per-item overhead swamps them, and it multiplies flash_merge's
        // partials by GF (merge is only 32 workgroups, so it scales with nsplit).
        //
        // We do not need to refill the machine: the fusion cut the traffic by GF, so fewer CUs
        // each doing GF-times-less work can still finish sooner. Swept on the real model:
        //
        //     nsplit   8     16     32     64
        //     token   16.8  16.8   17.8   19.3   ms
        //
        // 16 it is. Above that, flash fragments and merge (only 32 workgroups, and it scales
        // with nsplit) takes back everything the fusion won.
        // CONTEXT-ADAPTIVE default (short-ctx-flash lever). At SHORT ctx the KV is small, so ns16
        // OVER-splits: flash_merge's crit-path busy scales with nsplit (MEASURED Gemma-4-31B TP=1
        // ctx1k: merge busy 1010us@ns16 vs 764us@ns8) and the Opart f32 partials scale with it too,
        // while flash_decode barely benefits from >8 splits when there is little KV to read. At LONG
        // ctx the big full-layer KV read DOES need the fuller split. MEASURED decode ms/tok (ns8/ns16):
        //   ctx    1k          8k           64k
        //   ns8    18.06       18.82        24.67
        //   ns16   18.34       18.82        22.01
        // ns8 wins <=8k (-0.28ms @1k), ties at 8k, and REGRESSES >8k — so gate on the pkt's max_ctx.
        // ns1/ns2 lose EVERYWHERE (flash_decode serialization: busy 3541us@1k, 45189us@64k) — the
        // merge-elision path is a dead end: split-KV PARALLELISM, not the merge, is the ceiling.
        // Default mul: 1 (ns8) for a short-ctx pkt, 2 (ns16) otherwise. PLOW_NS_MUL / PLOW_NS_ABS
        // still override. Crossover measured at ~8k; a pkt compiled for <=8k is a short-ctx pkt.
        let mul_default: u32 = if ctx <= 8192 { 1 } else { 2 };
        let mul: u32 = std::env::var("PLOW_NS_MUL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(mul_default);
        // DECODE nsplit fill target uses the UNSHARDED head count (c.heads), NOT this rank's
        // sharded `heads` (= c.heads/tp). Under Megatron TP the KV cache is HEAD-partitioned (each
        // rank owns c.heads/tp q-heads over the FULL ctx per head), not context-split — so the right
        // per-head split of the KV context is tp-INDEPENDENT. The old `div_ceil(heads)` inflated
        // nsplit by tp (16->64 at tp=4) to refill 256 CUs, which quadrupled flash_merge: merge runs
        // on only heads/tp workgroups, each then reducing tp× the partials — MEASURED 1.00->3.05 ms
        // of a 14.55 ms tp=4 token. Basing the fill on c.heads keeps nsplit=16 at every tp (merge
        // per head unchanged), recovering ~2.3 ms at tp=4 (14.55->12.2). tp==1: c.heads==heads, so
        // byte-identical to the pre-TP path. Decode fragments flash_decode's fill under TP (fewer
        // work-items than CUs) but per-rank flash work is 1/tp anyway; merge, on the crit path, wins.
        let ns = if gemv_family {
            (n_cu * mul).div_ceil(c.heads).max(1)
        } else {
            ns
        };
        // sm_120 (188 SMs) 31B FULL-LAYER OVERSPLIT (campaign T7-31b-decode, RTX PRO 6000). The
        // CU-fill formula gives ns=12 on the 188-SM card at 32 heads (188*2/32), a 1.0x flash fill
        // (n_grp*ns = 16*12 = 192 ≈ 188 SMs). MEASURED: ns16 (1.36x oversubscribe) beats ns12 at
        // EVERY ctx for the dense 31B — the fine full-layer KV splits (10 layers × kv4 × hd512) hide
        // the long-ctx read latency and the 32-workgroup merge still absorbs the extra partials:
        //   ctx      1k      4k      16k     32k     64k     128k    (bf16 ms/tok)
        //   ns12     47.58   47.95   49.58   51.84   55.93   64.11
        //   ns16     47.49   47.79   48.98   50.73   53.95   60.37   (-0.2% .. -5.8%)
        //   ns24     -       -       -       51.16   -       61.59   (over-split: worse than 16)
        // fp8 (weight-only, so identical KV/flash bytes) gains MORE at long ctx: 64k -5.6%, 128k too.
        // Gated to the 31B signature — mixed sliding/full attention with 4-KV full layers — so 12B
        // (kvh_full=1 → ns24 already), Qwen/Llama (kvh_full==kvh_slide, want ns4-8), and short-ctx
        // pkts (<=8192, untested here) are byte-identical. PLOW_NS_MUL/PLOW_NS_ABS still override.
        let ns = if gemv_family && ctx > 8192 && c.kvh_full >= 4 && c.kvh_slide != c.kvh_full {
            ns.max(16)
        } else {
            ns
        };
        // GRID-ALIGNED FULL-LAYER nsplit (T9b-31b-tune, RTX PRO 6000 / 188 SMs). The full
        // layers' flash-decode work is n_grp*nsplit = (heads/GF)*nsplit items spread over n_cu
        // resident blocks. With n_grp=16 and n_cu=188 (gcd 4) that count is RAGGED at every
        // nsplit that is not a multiple of n_cu/gcd = 47: ceil() leaves ~68 blocks doing 2x the
        // work while the rest do 1x, and FLASH_MERGE waits for the slow 2x blocks (MEASURED
        // block-0 gate 658k cyc/op @128k, T9b trace). Rounding the fill target UP to a multiple
        // of `aligned` makes every block do exactly the same number of items.
        //   MEASURED 31B decode @128k (method of record, 120 timed): ns16(base) 58.57 ->
        //   ns47(aligned) 56.60 ms = -3.4%. ns24 (=384 items, ceil 3/block, WORSE imbalance)
        //   was 59.59, SLOWER than base — proving ALIGNMENT, not split count, is the lever
        //   (H2 stopped at ns24 and missed this). Only the 10 hd512/kv4 FULL layers change;
        //   the 50 hd256 sliding layers keep ns16 (their window-1024 KV is tiny, so 47-way
        //   over-splitting them would only add merge partials). Gated to the same 31B long-ctx
        //   signature as the ns.max(16) floor above, plus `full`, plus a <=64 sanity cap so a
        //   shape whose n_grp is coprime to n_cu (aligned would jump to n_cu) falls back.
        //   PLOW_NS_FULL_ABS still overrides for sweeps.
        let ns = if gemv_family && full && ctx > 8192 && c.kvh_full >= 4 && c.kvh_slide != c.kvh_full
        {
            let n_grp = (heads / fa_gf_full()).max(1);
            let aligned = n_cu / gcd(n_grp, n_cu); // smallest grid-aligned nsplit step
            let cand = ns.div_ceil(aligned) * aligned; // round the ns16 target up to it
            if cand <= 64 {
                cand
            } else {
                ns
            }
        } else {
            ns
        };
        // GRID-ALIGNED FULL-LAYER nsplit, 12B SINGLE-GLOBAL-KV-HEAD signature (beat12b-fp8-margin).
        // Gemma-4-12B full layers are kvh_full=1 (ONE kv head serves all 16 q heads): the CU-fill
        // formula gives ns=24 -> n_grp(8)*24 = 192 items on 188 SMs — RAGGED: 4 blocks run 2 items,
        // 184 run 1, and FLASH_MERGE waits for the 2x stragglers, so the full-layer flash runs at
        // ~2x its aligned latency at long ctx. Rounding to a multiple of n_cu/gcd(n_grp,n_cu)=47
        // (376 items = exactly 2/block) fixes it. MEASURED (flashdec_fp8_bw_12b microbench +
        // gemma4_sm120_chat, fp8 weights + fp8 head + fp8-KV, method of record n=112):
        //   decode ms/tok @128k: ns24 16.163 -> ns47 13.988 (-13.5%);  @1k 11.219 -> 11.213 (free)
        // Gated on fp8_kv (an emit-time flag) because the bf16-KV optimum differs (bf16 @128k
        // prefers ns23; microbench 0.436 vs 0.497) — with PLOW_FP8_KV unset the packet stays
        // byte-identical. Same <=128 sanity cap idea as the 31B block; PLOW_NS_ABS/PLOW_NS_FULL_ABS
        // still override below.
        let ns = if gemv_family && full && ctx > 8192 && c.kvh_full == 1 && fp8_kv {
            let n_grp = (heads / fa_gf_full()).max(1);
            let aligned = n_cu / gcd(n_grp, n_cu);
            let cand = ns.div_ceil(aligned) * aligned;
            if cand <= 128 {
                cand
            } else {
                ns
            }
        } else {
            ns
        };
        // WINDOWED-LAYER nsplit cap (beat12b-fp8-margin). A sliding layer's flash span never
        // exceeds `win` rows, so the CU-fill ns=24 over-splits it into 43-row items AND lands on
        // the same ragged 192-items-on-188-SMs grid as the full layers — a FIXED per-token cost
        // (the window doesn't grow with ctx). Cap ns so an item keeps >= 64 rows (a quarter
        // FA_DEC_TILE): win=1024 -> ns 16, n_work = 8*16 = 128 <= 188, no 2x tail. MEASURED
        // (12B fp8kv decode, sliding ns sweep at full ns=47, ms/tok @1k):
        //   ns8 10.937 | ns12 10.956 | ns16 10.921 | ns23 10.990 | ns24 (base) 11.221 | ns47 11.212
        // -0.30 ms at EVERY ctx (@128k 13.978 -> 13.684). fp8_kv-gated like the block above so
        // flag-unset packets stay byte-identical; PLOW_NS_ABS still overrides below.
        let ns = if gemv_family && !full && win > 0 && fp8_kv {
            ns.min((win / 64).max(1))
        } else {
            ns
        };
        // DECODE nsplit ABSOLUTE OVERRIDE (occupancy tuning). PLOW_NS_MUL scales the CU-fill target;
        // PLOW_NS_ABS pins nsplit directly. MEASURED on Qwen3-4B (all-global, GQA 4, MI350X):
        // the default mul=2 (ns=16) OVER-SPLITS flash_decode — each split's fixed overhead (Q
        // re-staging + the flash_merge partial + its barriers) dominates the tiny per-split KV
        // work, so summed flash_decode work grows with nsplit (ns 4/8/16/32/64 -> 59/80/112/174/285
        // ms) and decode ms/tok is best at ns=4-8 (4.3-4.4) vs 4.6 at ns=16. flash_decode is
        // over-fragmented, not under-filled. Inert by default; leaves Gemma's tuned mul path alone.
        let ns = std::env::var("PLOW_NS_ABS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|_| gemv_family)
            .unwrap_or(ns);
        // Full-attention-only decode split override. Unlike PLOW_NS_ABS this does not also
        // over-split Gemma's many hd256 sliding layers. It is the controlled sweep knob for
        // full-layer GQA-fusion experiments on sm_120 (GF4/ns24 => 8 groups * 24 = 192 work
        // items on the 188-SM RTX PRO 6000). Default unset preserves every existing packet.
        let ns = std::env::var("PLOW_NS_FULL_ABS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|_| gemv_family && full)
            .unwrap_or(ns);

        // The norm is ONE packet whose result all of q/k/v share.
        //
        // Folding it into each GEMV instead (norm mode 2, where the GEMV recomputes the row RMS
        // from its LDS-staged x) is CORRECT and deletes the packet and its gate -- and it was
        // MEASURED SLOWER: 22.4 -> 24.4 ms/token. Five consumers (q, k, v, gate, up) then each
        // redo the reduction, so one shared 10 us norm becomes five redundant ones on the
        // critical path, and the two gates saved do not pay for it. The op still supports mode
        // 2 (it is right for a single consumer), but the compiler does not use it here.
        // The end-of-layer AddNorm ALSO produces the NEXT layer's normed input, so for l>0 the
        // input RMSNorm is already done and `dep` carries n.hn directly.
        let c_n = if (fuse_norm || gfuse) && l > block.start {
            dep // previous layer's end-of-layer fused norm already wrote the normed n.hn
        } else {
            // The block's FIRST layer reads the uploaded `act.x` with no producer
            // (Embed was skipped), so its RmsNorm depends on `&[]`. Full model =>
            // block.start==0 and this only affects l==0, whose dep IS the Embed.
            let nd: &[u32] = if block_mode && l == block.start { &[] } else { &[dep] };
            b.emit(DevOp::RmsNorm, rows.clone(), nd, |d| {
                d.t[0] = n.hn;
                d.t[1] = n.x;
                d.t[2] = w.g_in;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            })
        };
        let (qkv_src, qkv_g) = (n.hn, TENSOR_NONE);

        // q, k and v are INDEPENDENT. Running all three on all 256 CUs makes them serialise
        // behind three separate counter gates for no reason: they are bandwidth-bound, so
        // total time is (total weight bytes / aggregate bandwidth) either way -- but disjoint
        // CU sets put them in flight together and cost ONE gate instead of three. Split in
        // proportion to weight bytes so they finish together.
        let (nq, nk, nv);
        let (c_q, c_k, c_v, v_src);
        if fuse_qkv {
            // ONE packet on all CUs: cols [0,qd) -> q, [qd,qd+kd) -> k, [qd+kd,qd+2kd) -> v.
            (nq, nk, nv) = (n_cu, n_cu, n_cu); // unused: fused headnorm deps are coarse
            let fused = b.emit(DevOp::GemvQkv, all.clone(), &[c_n], |d| {
                d.t[0] = n.qg;
                d.t[1] = qkv_src;
                d.t[2] = w.wq;
                d.t[3] = n.kg;
                d.t[4] = w.wk;
                d.t[5] = n.vg;
                d.t[6] = w.wv;
                d.i[0] = t;
                d.i[1] = qd;
                d.i[2] = c.hidden;
                d.i[3] = kd;
                d.i[4] = kd;
            });
            (c_q, c_k, c_v, v_src) = (fused, fused, fused, n.vg);
            let _ = qkv_g; // norm is a shared packet here, never folded into the fused GEMV
        } else {
            let (cq, ck, cv) = if gemv_family {
                split3(n_cu, qd, kd, if keqv { 0 } else { kd })
            } else {
                split3(
                    n_cu,
                    tiles(t, qd),
                    tiles(t, kd),
                    if keqv { 0 } else { tiles(t, kd) },
                )
            };
            (nq, nk, nv) = (cq.len() as u32, ck.len() as u32, cv.len() as u32);
            // w8a8: ONE quant of the (hidden-width) attn input, shared by q/k/v.
            let dq = quant(b, n.xqh, n.ash, qkv_src, c.hidden, c_n);
            let cqc = proj(
                b,
                n.qg,
                qkv_src,
                w.wq,
                w.wq8,
                w.sq,
                n.xqh,
                n.ash,
                t,
                qd,
                c.hidden,
                qkv_g,
                cq,
                &[dq],
            );
            let ckc = proj(
                b,
                n.kg,
                qkv_src,
                w.wk,
                w.wk8,
                w.sk,
                n.xqh,
                n.ash,
                t,
                kd,
                c.hidden,
                qkv_g,
                ck,
                &[dq],
            );
            let (vsrc, cvc) = if keqv {
                (n.kg, ckc) // k_eq_v: V is the RAW k_proj output
            } else {
                (
                    n.vg,
                    proj(
                        b,
                        n.vg,
                        qkv_src,
                        w.wv,
                        w.wv8,
                        w.sv,
                        n.xqh,
                        n.ash,
                        t,
                        kd,
                        c.hidden,
                        qkv_g,
                        cv,
                        &[dq],
                    ),
                )
            };
            (c_q, c_k, c_v, v_src) = (cqc, ckc, cvc, vsrc);
        }

        // headnorm+RoPE for q; and for k/v the store goes STRAIGHT INTO THE KV CACHE at
        // out_row0. In decode that row is the current position, which the runtime patches.
        let hn_cus: Vec<u32> = (0..((t * heads).div_ceil(8)).min(n_cu).max(1)).collect();
        let nhn = hn_cus.len() as u32;

        // gemv -> headnorm. headnorm workgroup j owns whole HEADS, and head h is the 256 (or
        // 512) consecutive output columns [h*hd, h*hd+hd) of the projection — so it needs only
        // the handful of gemv workgroups that produced those columns, not all 128.
        //
        // This is ONLY sparse under GV_BLOCKED (op_gemm.h). With the default wave-interleaved
        // column assignment a gemv workgroup's columns are scattered across all of N, and the
        // fan-in is 128 of 128 — measured, and the reason the first attempt at a fine chain
        // bought nothing.
        let hn_dep = |gemv: u32, nblk_g: u32, nheads: u32| -> Vec<Dep> {
            if !gemv_family || fuse_qkv {
                // the gemv column map assumes d_gemv (GV_BLOCKED); prefill is
                // a GEMM. The fused q|k|v op concatenates all three projections' columns across the
                // SAME 256 workgroups, so a head's per-workgroup producer set is no longer the
                // single-projection map below — fall back to coarse (the fused op is one uniform
                // packet, so all workgroups finish together and coarse costs ~nothing).
                // NOTE: we DECLARE the fine edge; `select_granularity` decides if it survives.
                return vec![Dep::Coarse(gemv)];
            }
            let dim = nheads * hd; // the projection's N
            let map = (0..nhn)
                .map(|j| {
                    let mut s: Vec<u32> = headnorm_items(nhn, t * nheads, j)
                        .into_iter()
                        .flat_map(|w| {
                            let h = w % nheads; // item = token*nheads + head
                            gemv_wgs_for_cols(dim, nblk_g, h * hd, (h + 1) * hd)
                        })
                        .collect();
                    s.sort_unstable();
                    s.dedup();
                    s
                })
                .collect();
            vec![Dep::Fine {
                producer: gemv,
                map,
            }]
        };

        let c_qn = b.emit_dep(
            DevOp::HeadNormRope,
            hn_cus.clone(),
            hn_dep(c_q, nq, heads),
            |d| {
                d.t[0] = n.q;
                d.t[1] = n.qg;
                d.t[2] = w.qn;
                d.t[3] = cs;
                d.t[4] = sn;
                d.t[5] = n.pos;
                d.i[0] = t;
                d.i[1] = heads;
                d.i[2] = hd;
                d.i[3] = 0;
                d.i[4] = qk_skip;
                d.f[0] = c.eps;
            },
        );
        // fp8-KV: the k/v norm STORES the cache as e4m3 with a per-row scale (t6). q is unchanged
        // (it is not cached — flash reads it as bf16), so it stays plain HeadNormRope.
        let hn_op = if fp8_kv {
            DevOp::HeadNormRopeFp8
        } else {
            DevOp::HeadNormRope
        };
        let c_kn = b.emit_dep(hn_op, hn_cus.clone(), hn_dep(c_k, nk, kvh), |d| {
            d.t[0] = n.kc[l];
            d.t[1] = n.kg;
            d.t[2] = w.kn;
            d.t[3] = cs;
            d.t[4] = sn;
            d.t[5] = n.pos;
            d.t[6] = n.kcs[l]; // fp8-KV per-row scale (NONE in bf16 mode)
            d.i[0] = t;
            d.i[1] = kvh;
            d.i[2] = hd;
            d.i[3] = 0;
            d.i[4] = qk_skip;
            d.f[0] = c.eps;
            // j0 = the KV cache's row stride (the RING size on a sliding layer); j1 = the row
            // mask. The write lands in the HEAD-MAJOR cache so flash can stream a head
            // end-to-end. See PLOW_KV_RING in dev_isa.h.
            d.j[0] = kvr;
            d.j[1] = kvm;
            // BATCH>1 decode: i6 = n_batch_kv selects the per-sequence KV ring (each seq writes at
            // its own pos[t]). 0 for prefill/B=1 => legacy single-ring, byte-identical.
            if decode && t > 1 {
                d.i[6] = t;
            }
        });
        if decode {
            kv_rows.push(c_kn);
        }
        // v_norm: WEIGHTLESS (gamma NONE) and NO RoPE (cos NONE).
        // On a full layer V comes from the RAW k_proj output, so its producer is c_k (nk wgs).
        let vn_dep = if keqv {
            hn_dep(c_v, nk, kvh)
        } else {
            hn_dep(c_v, nv, kvh)
        };
        let c_vn = b.emit_dep(hn_op, hn_cus, vn_dep, |d| {
            d.t[0] = n.vc[l];
            d.t[1] = v_src;
            d.t[5] = n.pos;
            d.t[6] = n.vcs[l]; // fp8-KV per-row scale (NONE in bf16 mode)
            d.i[0] = t;
            d.i[1] = kvh;
            d.i[2] = hd;
            d.i[3] = 0;
            d.i[4] = v_skip;
            d.f[0] = c.eps;
            d.j[0] = kvr;
            d.j[1] = kvm;
            if decode && t > 1 {
                d.i[6] = t;
            }
        });
        if decode {
            kv_rows.push(c_vn);
        }

        // headnorm -> flash. A flash work item is (batch, head, split); it reads Q for its own
        // head and the KV cache for head/gqa. Every OTHER row of the cache was written by a
        // PREVIOUS decode step (a previous launch), so within this program flash depends only on
        // the three headnorms' work for its own head — not on all of them.
        let fa_dep = || -> Vec<Dep> {
            if !gemv_family {
                return vec![Dep::Coarse(c_qn), Dep::Coarse(c_kn), Dep::Coarse(c_vn)];
            }
            let nblk_f = all.len() as u32;
            let gqa = heads / kvh;
            let n_work = t * heads * ns; // d_flash_decode: n_batch * n_head * nsplit
            let mk = |kv: bool| -> Vec<Vec<u32>> {
                (0..nblk_f)
                    .map(|f| {
                        let mut s: Vec<u32> = (0..n_work)
                            .filter(|w| w % nblk_f == f) // the items THIS workgroup runs
                            .map(|w| {
                                let h = (w / ns) % heads;
                                let bb = w / (ns * heads);
                                if kv {
                                    headnorm_wg_of(nhn, bb * kvh + h / gqa)
                                } else {
                                    headnorm_wg_of(nhn, bb * heads + h)
                                }
                            })
                            .collect();
                        s.sort_unstable();
                        s.dedup();
                        s
                    })
                    .collect()
            };
            vec![
                Dep::Fine {
                    producer: c_qn,
                    map: mk(false),
                },
                Dep::Fine {
                    producer: c_kn,
                    map: mk(true),
                },
                Dep::Fine {
                    producer: c_vn,
                    map: mk(true),
                },
            ]
        };
        let c_fa = if gemv_family {
            let fa_op = if fp8_kv {
                DevOp::FlashDecodeFp8
            } else {
                DevOp::FlashDecode
            };
            b.emit_dep(fa_op, all.clone(), fa_dep(), |d| {
                d.t[0] = n.opart;
                d.t[1] = n.mlpart;
                d.t[2] = n.q;
                d.t[3] = n.kc[l];
                d.t[4] = n.vc[l];
                d.t[5] = n.kvlen;
                d.t[6] = n.kcs[l];
                d.t[7] = n.vcs[l]; // fp8-KV per-row scales (NONE in bf16 mode)
                                   // BATCH>1: n_batch = t (one query row per sequence, each with its own KV ring).
                d.i[0] = t;
                d.i[1] = heads;
                d.i[2] = kvh;
                d.i[3] = kvr;
                d.i[4] = win;
                d.i[5] = ns;
                d.i[6] = hd;
                d.i[7] = kvm;
                d.f[0] = c.attn_scale;
                // KV row-capacity for the PLOW_NV_KVBOUNDS trap: the b>=1 OOB read past an
                // under-sized KV allocation traps here instead of reading fluent wrong text. Set
                // only for B>1 so the B=1 packet stays byte-identical (j[0] default 0 => no check).
                if t > 1 {
                    d.j[0] = t * kvh * kvr;
                }
            })
        } else {
            let fa_op = if fp8_kv {
                DevOp::FlashPrefillFp8
            } else {
                DevOp::FlashPrefill
            };
            b.emit(fa_op, all.clone(), &[c_qn, c_kn, c_vn], |d| {
                d.t[0] = n.opart;
                d.t[1] = n.mlpart;
                d.t[2] = n.q;
                d.t[3] = n.kc[l];
                d.t[4] = n.vc[l];
                d.t[6] = n.kcs[l];
                d.t[7] = n.vcs[l]; // fp8-KV per-row scales (NONE in bf16 mode)
                                   // Fused epilogue: t[5] is the final bf16 attention output (n.at). When !fused
                                   // it stays NONE and flash_prefill writes the f32 partial for d_flash_merge.
                if fused {
                    d.t[5] = n.at;
                }
                // n_head is THIS RANK's sharded head count under TP (§3a): Q/O buffers hold
                // heads=c.heads/tp heads and flash_merge reads the same `heads`. Passing the
                // unsharded c.heads here read past the sharded Q buffer (the prefill-TP bug).
                d.i[0] = t;
                d.i[1] = t;
                d.i[2] = heads;
                d.i[3] = kvh;
                d.i[4] = 0;
                d.i[5] = win;
                d.i[6] = hd;
                d.i[7] = ns;
                d.f[0] = c.attn_scale;
                d.j[0] = kvr;
                d.j[1] = kvm; // head-major; RING on a sliding layer
            })
        };
        // When fused, flash_prefill already wrote the normalized bf16 to n.at, so there is no
        // FlashMerge op and o_proj depends on the flash op directly. Coarse: n.at row r needs
        // every head of its q-tile, which is spread across the flash workgroups.
        let attn_dep = if fused {
            c_fa
        } else {
            let mg_cus: Vec<u32> = (0..(t * heads).min(n_cu).max(1)).collect();
            let fill = |d: &mut DevInst| {
                d.t[0] = n.at;
                d.t[1] = n.opart;
                d.t[2] = n.mlpart;
                d.i[0] = t;
                d.i[1] = heads;
                d.i[2] = ns;
                d.i[3] = hd;
            };
            // A merge workgroup folds the `ns` partials of its own (row, head) and reads
            // nothing else, so make it wait for exactly those flash slices instead of all 256.
            let map = flash_merge_map(
                t * heads,
                ns,
                if gemv_family { 1 } else { Q_TILE_ROWS },
                heads,
                all.len() as u32,
                mg_cus.len() as u32,
            );
            b.emit_dep(
                DevOp::FlashMerge,
                mg_cus,
                vec![Dep::Fine {
                    producer: c_fa,
                    map,
                }],
                fill,
            )
        };

        // o_proj is ROW-parallel (plans/tp-design.md §3a): input = this rank's qd heads, output =
        // the FULL H-vector but only a PARTIAL sum (this rank's head contribution). Under TP it
        // writes that partial into the peer-mapped og_tp slot and an XReduce sums the N peers'
        // partials into the replicated `og` that NormResidual consumes — all-reduce #1 of the layer.
        // proj() picks the fp8 (GemvFp8) arm on the decode fp8 path via the wo8/so operands.
        // w8a8: quant the (qd-width) attention output feeding o_proj.
        let do_ = quant(b, n.xqo, n.aso, n.at, qd, attn_dep);
        let c_o = if tp > 1 {
            let c_op = proj(
                b,
                n.og_tp,
                n.at,
                w.wo,
                w.wo8,
                w.so,
                n.xqo,
                n.aso,
                t,
                c.hidden,
                qd,
                TENSOR_NONE,
                all.clone(),
                &[do_],
            );
            emit_xreduce(b, &mut xgate, decode, &xr_cus, c_op, n.og, xr_elems, tp, 0)
        } else {
            proj(
                b,
                n.og,
                n.at,
                w.wo,
                w.wo8,
                w.so,
                n.xqo,
                n.aso,
                t,
                c.hidden,
                qd,
                TENSOR_NONE,
                all.clone(),
                &[do_],
            )
        };
        // FIRST RESIDUAL + PRE-MLP NORM — the biggest structural fork.
        //   Gemma SANDWICH: x = x + post_attn_norm(o); then hn = pre_feedforward_norm(x).
        //   Llama/Qwen PRE-NORM: x = x + o (plain); then hn = post_attention_layernorm(x).
        // Gemma applies its post-attn norm to the ATTENTION OUTPUT before the add; Llama/Qwen add
        // the raw output and normalize the residual stream going INTO the MLP.
        let gemma = c.arch == Arch::Gemma4;
        // Pre-MLP norm. Gemma: sandwich (NormResidual) then a separate pre-FF norm. Qwen/Llama
        // decode: x += o, then post_attention_layernorm(x) — fused into ONE AddNorm. Qwen/Llama
        // prefill keeps the split (T rows already parallelise the norm; a parallel agent owns it).
        let c_pf = if fuse_norm {
            b.emit(DevOp::AddNorm, rows.clone(), &[c_o], |d| {
                d.t[0] = n.hn;
                d.t[1] = n.x;
                d.t[2] = n.x;
                d.t[3] = n.og;
                d.t[4] = w.g_pa;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            })
        } else if gfuse {
            // x = x + post_attn_norm(o); hn = pre_feedforward_norm(x) — Gemma sandwich in ONE packet.
            b.emit(DevOp::NormResidualNorm, rows.clone(), &[c_o], |d| {
                d.t[0] = n.hn;
                d.t[1] = n.x;
                d.t[2] = n.x;
                d.t[3] = n.og;
                d.t[4] = w.g_pa;
                d.t[5] = w.g_pf;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
                d.f[1] = 1.0;
            })
        } else {
            let c_r1 = if gemma {
                b.emit(DevOp::NormResidual, rows.clone(), &[c_o], |d| {
                    d.t[0] = n.x;
                    d.t[1] = n.x;
                    d.t[2] = n.og;
                    d.t[3] = w.g_pa;
                    d.i[0] = t;
                    d.i[1] = c.hidden;
                    d.f[0] = c.eps;
                    d.f[1] = 1.0;
                })
            } else {
                b.emit(DevOp::Residual, elem(t * c.hidden), &[c_o], |d| {
                    d.t[0] = n.x;
                    d.t[1] = n.x;
                    d.t[2] = n.og;
                    d.i[0] = t * c.hidden;
                    d.f[0] = 1.0;
                })
            };
            let pre_mlp_norm = if gemma { w.g_pf } else { w.g_pa };
            b.emit(DevOp::RmsNorm, rows.clone(), &[c_r1], |d| {
                d.t[0] = n.hn;
                d.t[1] = n.x;
                d.t[2] = pre_mlp_norm;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            })
        };
        let (mlp_src, mlp_g) = (n.hn, TENSOR_NONE);
        // GATE|UP AS ONE GEMV WITH A FUSED GLU EPILOGUE -- the fusion every BLAS ships.
        //
        // gate and up read the same x and have the same shape, so one GEMV can compute BOTH
        // halves of its own output columns and apply act(gate)*up as it writes them. The GLU is
        // then output-stationary: the workgroup owning column n is the only one that touches it,
        // so the GLU runs exactly once per element and NOTHING is replicated. Three packets
        // (gemv, gemv, glu) collapse to one, and the GLU's global gate -- which 250 of 256 CUs
        // stalled behind, it ran on 6 workgroups -- disappears with it.
        //
        // The DIRECTION is the whole thing. Folding the GLU into the *down* GEMV's LDS staging
        // instead (the consumer's PROLOGUE) was measured at a 39x LOSS: `fu` is down's K
        // dimension, so all 256 of its workgroups stage the whole of it and each recomputes the
        // entire GLU. Fuse into the producer's EPILOGUE, never into the consumer's PROLOGUE.
        //
        // Needs x staged on-chip (its A-operand is read once per output column), so it is a
        // decode-path op; prefill keeps the tiled GEMM triple, where the GLU amortises anyway.
        // Prefill fuses too, via the GEMM epilogue (DevOp::GemmGlu) -- same fusion, same law.
        // The GEMV form needs x staged on-chip; the GEMM form has no such constraint, it just
        // stages a different B tile. Requires the 256x256 tile (its SN axis is what carries
        // gate-vs-up), so only when pick_tile would have chosen Gemm anyway.
        // gate/up are COLUMN-parallel (inter_l lanes on this rank); the GLU is elementwise on the
        // rank's own lanes, so no communication. `c_gl` is the dependency feeding down_proj.
        let glu_fused = gemv_family && (t as u64 * c.hidden as u64) <= GM_LDS_HALVES;
        let gemm_glu = !gemv_family && pick_tile(t, inter_l, c.hidden, n_cu) == DevOp::Gemm;
        // w8a8: quant the (hidden-width) pre-FF norm output feeding gate/up. Reuses xqh/ash (q/k/v
        // already consumed them; the c_pf→o_proj→flash→qkv chain serializes the reuse). Inert
        // (returns c_pf) off the w8a8 path, so glu_fused/bf16 arms below keep their c_pf dep.
        // P9 hoist: emit the MoE router (score + topk) BEFORE the dense-MLP packets. Streams
        // execute in emission order per block, so with the router emitted after the MLP the
        // score/topk blocks only reach it once their dense slices retire — serializing
        // dense + router + expert-GLU. Hoisting turns that into max(dense, router) + GLU.
        // The DAG (deps: c_pf only) is unchanged; only stream position moves.
        let c_rt_hoist = if c.moe && decode {
            let root = (c.hidden as f32).powf(-0.5);
            Some(emit_gemma_moe_router(
                b,
                c_pf,
                n.moe_tab,
                n.x,
                w.rproj,
                w.rscale,
                w.rpes,
                n.moe_rscore,
                c.hidden,
                c.n_exp,
                c.top_k,
                root,
                c.eps,
                gemma_moe_router_split_plan(n_cu, c.n_exp, t),
                t,
            ))
        } else {
            None
        };
        let dmlp = quant(b, n.xqh, n.ash, mlp_src, c.hidden, c_pf);
        let c_gl = if glu_fused {
            // FP8 decode: gate|up fused GEMV+GLU on fp8 weights, each with its own dequant scale.
            if fp8 {
                b.emit(DevOp::GemvGluFp8, all.clone(), &[c_pf], |d| {
                    d.t[0] = n.fu;
                    d.t[1] = mlp_src;
                    d.t[2] = w.wg8;
                    d.t[5] = w.wu8;
                    d.t[3] = w.sg;
                    d.t[4] = w.su;
                    d.i[0] = t;
                    d.i[1] = inter_l;
                    d.i[2] = c.hidden;
                    d.i[5] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU (Llama/Qwen)
                })
            } else {
                b.emit(DevOp::GemvGlu, all.clone(), &[c_pf], |d| {
                    d.t[0] = n.fu;
                    d.t[1] = mlp_src;
                    d.t[2] = w.wg;
                    d.t[5] = w.wu;
                    d.i[0] = t;
                    d.i[1] = inter_l;
                    d.i[2] = c.hidden;
                    d.i[5] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU (Llama/Qwen)
                })
            }
        } else if gemm_glu && fp8 {
            // PREFILL fp8 GLU: w8a16 (default cubin) OR w8a8 (PLOW_NV_W8A8 cubin), same GEMM_GLU_FP8
            // opcode. w8a16: A bf16 (t1=mlp_src), Wg/Wu e4m3 (t2/t5), per-channel g/u scales (t4/t6).
            // w8a8: A e4m3 (t1=xqh) + per-row a_scale (t3=ash); Wg/Wu e4m3 + g/u scales — the
            // epilogue folds a_scale*sg (and a_scale*su) into both streams. Same fusion law.
            b.emit(DevOp::GemmGluFp8, all.clone(), &[dmlp], |d| {
                d.t[0] = n.fu;
                d.t[2] = w.wg8;
                d.t[5] = w.wu8;
                d.t[4] = w.sg;
                d.t[6] = w.su;
                if w8a8 {
                    d.t[1] = n.xqh;
                    d.t[3] = n.ash;
                } else {
                    d.t[1] = mlp_src;
                }
                d.i[0] = t;
                d.i[1] = inter_l;
                d.i[2] = c.hidden;
                d.i[5] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU (Llama/Qwen)
            })
        } else if gemm_glu {
            b.emit(DevOp::GemmGlu, all.clone(), &[c_pf], |d| {
                d.t[0] = n.fu;
                d.t[1] = mlp_src;
                d.t[2] = w.wg;
                d.t[5] = w.wu;
                d.i[0] = t;
                d.i[1] = inter_l;
                d.i[2] = c.hidden;
                d.i[5] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU (Llama/Qwen)
            })
        } else {
            // gate and up: same argument as q/k/v -- independent, so disjoint CU sets.
            let (cg, cu) = if gemv_family {
                split2(n_cu, 1, 1)
            } else {
                split2(n_cu, tiles(t, inter_l), tiles(t, inter_l))
            };
            let c_g = proj(
                b,
                n.gt,
                mlp_src,
                w.wg,
                w.wg8,
                w.sg,
                n.xqh,
                n.ash,
                t,
                inter_l,
                c.hidden,
                mlp_g,
                cg,
                &[dmlp],
            );
            let c_u = proj(
                b,
                n.ut,
                mlp_src,
                w.wu,
                w.wu8,
                w.su,
                n.xqh,
                n.ash,
                t,
                inter_l,
                c.hidden,
                mlp_g,
                cu,
                &[dmlp],
            );
            b.emit(DevOp::Glu, elem(t * inter_l), &[c_g, c_u], |d| {
                d.t[0] = n.fu;
                d.t[1] = n.gt;
                d.t[2] = n.ut;
                d.i[0] = t * inter_l;
                d.i[1] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU
            })
        };
        // down_proj is ROW-parallel (input = inter_l lanes) → a PARTIAL H-vector. Under TP it
        // writes dg_tp and an XReduce sums the N peers into `dg` — all-reduce #2 of the layer,
        // at the second NormResidual boundary (plans/tp-design.md §3a, §8a). proj() picks the fp8
        // (GemvFp8) arm on the decode fp8 path via the wd8/sd operands.
        // w8a8: quant the (inter-width) GLU output feeding down_proj.
        let dfu = quant(b, n.xqi, n.asi, n.fu, inter_l, c_gl);
        let c_d = if tp > 1 {
            let c_dp = proj(
                b,
                n.dg_tp,
                n.fu,
                w.wd,
                w.wd8,
                w.sd,
                n.xqi,
                n.asi,
                t,
                c.hidden,
                inter_l,
                TENSOR_NONE,
                all.clone(),
                &[dfu],
            );
            emit_xreduce(
                b, &mut xgate, decode, &xr_cus, c_dp, n.dg, xr_elems, tp, slot_b,
            )
        } else {
            proj(
                b,
                n.dg,
                n.fu,
                w.wd,
                w.wd8,
                w.sd,
                n.xqi,
                n.asi,
                t,
                c.hidden,
                inter_l,
                TENSOR_NONE,
                all.clone(),
                &[dfu],
            )
        };
        // ===== Gemma-4 26B-A4B MoE branch (decode, B=1; plans/rtx-08-gemma4-moe-26b.md) =====
        // The dense MLP above produced `n.dg`. The MoE block adds a routed-expert branch and sums
        // the two through the sandwich: combined = post_ffn_norm(post_ffn_1(dense) + post_ffn_2(moe)).
        // Router & experts both read the RESIDUAL (n.x, set by c_pf), NOT the pre-MLP norm. The
        // second residual below then consumes `ffn_out` (= moe_comb) instead of n.dg.
        // P9 op72: when the MoE tail is fused (combine+resid+next-norm in one packet), this
        // carries its counter and the SECOND RESIDUAL block below is skipped entirely.
        let mut moe_fused_tail: Option<u32> = None;
        let (ffn_out, c_d) = if c.moe {
            let root = (c.hidden as f32).powf(-0.5);
            // BATCH>1 DECODE: `t` IS the batch B here. Every decode MoE op carries B in a spare
            // immediate, emitted only when B>1 so the B=1 packet stays byte-identical. The routed
            // work space becomes B*k slots; the kernels sweep it CHANNEL-MAJOR so slots that share
            // an expert read that expert's weight rows once from HBM (op_moe.cuh ordering note).
            let nb = if decode && t > 1 { t } else { 0 };
            assert!(
                !decode || t <= 32,
                "MoE decode batch is capped at 32 (per-CTA inv[] scratch, PLOW_MOE_MAXB)"
            );
            if decode {
            // h1 = post_feedforward_layernorm_1(dense MLP output)
            let c_h1 = b.emit(DevOp::RmsNorm, rows.clone(), &[c_d], |d| {
                d.t[0] = n.moe_h1;
                d.t[1] = n.dg;
                d.t[2] = w.g_pf1;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            });
            // router(residual): weightless-rms -> ·scale·root -> softmax -> top-k (lowest-id tie)
            // -> norm_topk -> ·per_expert_scale -> routing_table[k]. The default remains the
            // historical one-block opcode. The opt-in split scores eight experts per CTA, then a
            // one-CTA tail performs the exact serial softmax/top-k/gate ordering.
            let c_rt = c_rt_hoist.expect("router hoisted before dense MLP for moe");
            let _ = root;
            // Expert gate/up (fused) -> mfu[k,I]; expert down + gate scale -> part[k,H].
            // GluNorm fusion (op 71): fuse the pre-feedforward-norm-2 INTO the expert GLU,
            // eliminating a separate RmsNorm op + counter gate. Each CTA redundantly computes
            // the RMS of the residual (5.6 KB @ H=2816, hot in L2 from the router read).
            // Falls back to separate norm + GLU when fp8 (no fused fp8 variant yet).
            let glu_cus: Vec<u32> = (0..n_cu).collect();
            let down_cus: Vec<u32> = (0..n_cu).collect();
            let glu_op = if fp8 {
                DevOp::MoeExpertGluGemmaFp8
            } else {
                DevOp::MoeExpertGluNormGemma
            };
            let c_glu = if fp8 {
                // fp8 path: separate norm + expert GLU (no fused fp8 norm variant)
                let c_xn2_local = b.emit(DevOp::RmsNorm, rows.clone(), &[c_pf], |d| {
                    d.t[0] = n.moe_xn2;
                    d.t[1] = n.x;
                    d.t[2] = w.g_pre2;
                    d.i[0] = t;
                    d.i[1] = c.hidden;
                    d.f[0] = c.eps;
                });
                b.emit(DevOp::MoeExpertGluGemmaFp8, glu_cus, &[c_rt, c_xn2_local], |d| {
                    d.t[0] = n.moe_mfu;
                    d.t[1] = n.moe_xn2;
                    d.t[2] = n.moe_tab;
                    d.t[3] = w.ewt;
                    d.t[4] = w.est;
                    d.i[0] = c.top_k;
                    d.i[1] = c.moe_inter;
                    d.i[2] = c.hidden;
                    d.i[3] = c.n_exp;
                    d.i[5] = nb; // BATCH B (0 at B=1: byte-identical)
                })
            } else {
                // bf16 path: fused norm + expert GLU (one fewer gate)
                b.emit(DevOp::MoeExpertGluNormGemma, glu_cus, &[c_rt, c_pf], |d| {
                    d.t[0] = n.moe_mfu;
                    d.t[1] = n.x;
                    d.t[2] = n.moe_tab;
                    d.t[3] = w.ewt;
                    d.t[4] = w.g_pre2;
                    d.i[0] = c.top_k;
                    d.i[1] = c.moe_inter;
                    d.i[2] = c.hidden;
                    d.i[3] = c.n_exp;
                    d.i[5] = nb; // BATCH B (0 at B=1: byte-identical)
                    d.f[0] = c.eps;
                })
            };
            let down_op = if fp8 {
                DevOp::MoeExpertDownGemmaFp8
            } else {
                DevOp::MoeExpertDownGemma
            };
            let c_dn = vec![b.emit(down_op, down_cus, &[c_glu], |d| {
                d.t[0] = n.moe_part;
                d.t[1] = n.moe_mfu;
                d.t[2] = n.moe_tab;
                d.t[3] = w.ewt;
                d.t[4] = w.est;
                d.i[0] = c.top_k;
                d.i[1] = c.hidden;
                d.i[2] = c.moe_inter;
                d.i[3] = c.n_exp;
                d.i[5] = nb; // BATCH B (0 at B=1: byte-identical)
            })];
            // fused combine + rmsnorm + residual: saves 2 counter gates per layer.
            let mut comb_deps: Vec<u32> = c_dn;
            comb_deps.push(c_h1);
            // op72 MEASURED NEGATIVE in its scalar form (P9, 2026-07-20): +0.18 ms/token on
            // BOTH bf16 (8.04→8.22) and fp8 (6.03→6.21) @40ctx — the 1-block 4-pass scalar
            // body costs more than the packet boundary it removes, and its reduction order
            // differs from the vectorized NormResidualNorm (last-ulp bf16 flips → token
            // drift vs the pair). Oracle is bit-exact vs its own golden. Default OFF; only
            // worth revisiting as a register-cached vectorized body that replicates NRN's
            // summation order. Opt in: PLOW_GEMMA_MOE_TAIL_FUSE=1.
            let tail_fuse = std::env::var("PLOW_GEMMA_MOE_TAIL_FUSE").ok().as_deref()
                == Some("1");
            // op72 is a single-row 1-CTA body and is default-OFF (measured negative); it was not
            // batched. Refuse the combination loudly rather than emit wrong rows 1..B.
            assert!(
                !(tail_fuse && t > 1),
                "PLOW_GEMMA_MOE_TAIL_FUSE is B=1 only (op72 MoeCombineResidNormGemma is not batched)"
            );
            let c_comb = if gfuse && tail_fuse {
                // op72: fused combine + post_ffn norm + sandwich residual + NEXT input norm.
                // One 1-block packet replaces the (op70, NormResidualNorm) pair on the layer
                // tail — the chain next-QKV gates on loses a packet boundary. Bit-exact.
                let next_gin = if l + 1 < block.end {
                    n.lw[l + 1].g_in
                } else {
                    n.fin
                };
                let ct = b.emit(DevOp::MoeCombineResidNormGemma, vec![0], &comb_deps, |d| {
                    d.t[0] = n.hn;
                    d.t[1] = n.x;
                    d.t[2] = n.moe_part;
                    d.t[3] = n.moe_h1;
                    d.t[4] = w.g_pf2;
                    d.t[5] = w.g_po;
                    d.t[6] = next_gin;
                    d.i[0] = c.hidden;
                    d.i[1] = c.top_k;
                    d.f[0] = c.eps;
                    d.f[1] = ls[l];
                });
                moe_fused_tail = Some(ct);
                ct
            } else {
                // BATCH B>1: one CTA per row (the body is a per-row block loop).
                let comb_cus: Vec<u32> = (0..t).collect();
                b.emit(DevOp::MoeCombineNormGemma, comb_cus, &comb_deps, |d| {
                    d.t[0] = n.moe_comb;
                    d.t[1] = n.moe_part;
                    d.t[2] = n.moe_h1;
                    d.t[3] = w.g_pf2;
                    d.i[0] = c.hidden;
                    d.i[1] = c.top_k;
                    d.i[2] = nb; // BATCH B (0 at B=1: byte-identical)
                    d.f[0] = c.eps;
                })
            };
            (n.moe_comb, c_comb)
            } else {
            // ===== GROUPED-MoE PREFILL (T rows; plans/p9-26b-prefill-moe.md) =====
            // h1 = post_ffn_norm_1(dense), T rows. xn2 = pre_ffn_norm_2(residual), T rows.
            let c_h1 = b.emit(DevOp::RmsNorm, rows.clone(), &[c_d], |d| {
                d.t[0] = n.moe_h1;
                d.t[1] = n.dg;
                d.t[2] = w.g_pf1;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            });
            let c_xn2 = b.emit(DevOp::RmsNorm, rows.clone(), &[c_pf], |d| {
                d.t[0] = n.moe_xn2;
                d.t[1] = n.x;
                d.t[2] = w.g_pre2;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            });
            // T-token router -> routing_table[T*k] (block-per-token, bit-identical to decode).
            let c_rt = b.emit(DevOp::MoeRouterGemmaPf, all.clone(), &[c_pf], |d| {
                d.t[0] = n.moe_tab;
                d.t[1] = n.x;
                d.t[2] = w.rproj;
                d.t[3] = w.rscale;
                d.t[4] = w.rpes;
                d.i[0] = c.hidden;
                d.i[1] = c.n_exp;
                d.i[2] = c.top_k;
                d.i[3] = t;
                d.f[0] = root;
                d.f[1] = c.eps;
            });
            // align/sort (SINGLE block): histogram -> padded prefix -> scatter gather maps.
            let c_align = b.emit(DevOp::MoeAlignGemmaPf, vec![0], &[c_rt], |d| {
                d.t[0] = n.moe_meta;
                d.t[1] = n.moe_tab;
                d.t[2] = n.moe_rowtok;
                d.t[3] = n.moe_rowpart;
                d.t[4] = n.moe_rowgate;
                d.i[0] = t;
                d.i[1] = c.n_exp;
                d.i[2] = c.top_k;
            });
            // grouped gate/up GEMM + GeGLU (gathered A, expert-selected B) -> fu_gathered.
            // beat26b: w8a8 arm = native fp8 tensor-core GEMM (both operands e4m3). xn2 is quantized
            // to e4m3 (xqh/ash, hidden width) once; the grouped GLU gathers e4m3 rows and dequants
            // with a_scale[token]*w_scale[chan] in the epilogue. bf16 arm unchanged.
            let c_dn = if w8a8 {
                // total_pad rows the align op touched for THIS bucket (matches align's write extent).
                let moe_total_pad = t * c.top_k + c.n_exp * 128;
                let c_xn2q = quant(b, n.xqh, n.ash, n.moe_xn2, c.hidden, c_xn2);
                let c_glu = b.emit(DevOp::MoeGroupGluGemmaPfW8a8, all.clone(), &[c_align, c_xn2q], |d| {
                    d.t[0] = n.moe_fug;
                    d.t[1] = n.xqh;      // xn2 e4m3
                    d.t[2] = w.ewt;      // fp8 expert weights
                    d.t[3] = n.moe_meta;
                    d.t[4] = n.moe_rowtok;
                    d.t[5] = n.ash;      // per-token a_scale
                    d.t[6] = w.est;      // per-channel weight scales
                    d.i[0] = c.moe_inter;
                    d.i[1] = c.hidden;
                    d.i[2] = c.n_exp;
                    d.i[5] = c.mlp_act;
                });
                // quant the gathered GLU output (total_pad rows, moe_inter width) for the down GEMM.
                let c_fuq = b.emit(DevOp::QuantFp8, all.clone(), &[c_glu], |d| {
                    d.t[0] = n.moe_fuq;
                    d.t[1] = n.moe_fug;
                    d.t[2] = n.moe_fus;
                    d.i[0] = moe_total_pad;
                    d.i[1] = c.moe_inter;
                });
                b.emit(DevOp::MoeGroupDownGemmaPfW8a8, all.clone(), &[c_fuq, c_align], |d| {
                    d.t[0] = n.moe_part;
                    d.t[1] = n.moe_fuq;  // fu e4m3
                    d.t[2] = w.ewt;
                    d.t[3] = n.moe_meta;
                    d.t[4] = n.moe_rowpart;
                    d.t[5] = n.moe_rowgate;
                    d.t[6] = w.est;
                    d.t[7] = n.moe_fus;  // per-row fu scale
                    d.i[0] = c.hidden;
                    d.i[1] = c.moe_inter;
                    d.i[2] = c.n_exp;
                })
            } else {
            let c_glu = b.emit(DevOp::MoeGroupGluGemmaPf, all.clone(), &[c_align, c_xn2], |d| {
                d.t[0] = n.moe_fug;
                d.t[1] = n.moe_xn2;
                d.t[2] = w.ewt;
                d.t[3] = n.moe_meta;
                d.t[4] = n.moe_rowtok;
                d.i[0] = c.moe_inter;
                d.i[1] = c.hidden;
                d.i[2] = c.n_exp;
                d.i[5] = c.mlp_act; // 0 GeGLU (Gemma)
            });
            // grouped down GEMM + gate-scale + scatter -> part[T,k,H].
            b.emit(DevOp::MoeGroupDownGemmaPf, all.clone(), &[c_glu, c_align], |d| {
                d.t[0] = n.moe_part;
                d.t[1] = n.moe_fug;
                d.t[2] = w.ewt;
                d.t[3] = n.moe_meta;
                d.t[4] = n.moe_rowpart;
                d.t[5] = n.moe_rowgate;
                d.i[0] = c.hidden;
                d.i[1] = c.moe_inter;
                d.i[2] = c.n_exp;
            })
            };
            // T-row combine + sandwich: out[t] = RMSNorm(Σ_slot part[t][slot], g_pf2) + h1[t].
            let c_comb = b.emit(DevOp::MoeCombineNormGemmaPf, all.clone(), &[c_dn, c_h1], |d| {
                d.t[0] = n.moe_comb;
                d.t[1] = n.moe_part;
                d.t[2] = n.moe_h1;
                d.t[3] = w.g_pf2;
                d.i[0] = c.hidden;
                d.i[1] = c.top_k;
                d.i[2] = t;
                d.f[0] = c.eps;
            });
            (n.moe_comb, c_comb)
            }
        } else {
            (n.dg, c_d)
        };
        // SECOND RESIDUAL.
        //   Gemma: x = (x + post_ffn_norm(d)) * layer_scalar — the learned scalar folds in.
        //   Llama/Qwen: x = x + d (plain).
        dep = if let Some(ct) = moe_fused_tail {
            // op72 already produced the new residual (n.x) AND the next input norm (n.hn).
            ct
        } else if fuse_norm {
            // x += down; then normalise for the NEXT sublayer's attention (the next layer's
            // input_layernorm, or the model's final norm after the last layer). One packet does
            // the end-of-layer residual AND the next input norm, so the loop top skips c_n.
            let next_gin = if l + 1 < block.end {
                n.lw[l + 1].g_in
            } else {
                n.fin
            };
            b.emit(DevOp::AddNorm, rows.clone(), &[c_d], |d| {
                d.t[0] = n.hn;
                d.t[1] = n.x;
                d.t[2] = n.x;
                d.t[3] = ffn_out;
                d.t[4] = next_gin;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            })
        } else if gfuse {
            // x = (x + post_ffn_norm(down)) * layer_scalar; hn = input_norm(x) for the NEXT layer
            // (or the final norm after the last layer). One packet does the end-of-layer sandwich
            // residual AND the next input norm, so the loop top skips c_n (same as fuse_norm).
            let next_gin = if l + 1 < block.end {
                n.lw[l + 1].g_in
            } else {
                n.fin
            };
            b.emit(DevOp::NormResidualNorm, rows.clone(), &[c_d], |d| {
                d.t[0] = n.hn;
                d.t[1] = n.x;
                d.t[2] = n.x;
                d.t[3] = ffn_out;
                d.t[4] = w.g_po;
                d.t[5] = next_gin;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
                d.f[1] = ls[l];
            })
        } else if gemma {
            b.emit(DevOp::NormResidual, rows.clone(), &[c_d], |d| {
                d.t[0] = n.x;
                d.t[1] = n.x;
                d.t[2] = ffn_out;
                d.t[3] = w.g_po;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
                d.f[1] = ls[l];
            })
        } else {
            b.emit(DevOp::Residual, elem(t * c.hidden), &[c_d], |d| {
                d.t[0] = n.x;
                d.t[1] = n.x;
                d.t[2] = ffn_out;
                d.i[0] = t * c.hidden;
                d.f[0] = 1.0;
            })
        };
    }

    // BLOCK MODE: stop here. `act.x` (n.x) — the post-FFN residual the loop's
    // last layer wrote — IS the block output the harness downloads. The final
    // norm, lm_head, softcap and argmax tail belongs to the whole-model run; a
    // single block emits neither logits nor a sampled token (act.logits stays
    // declared-but-unwritten, satisfying GpuEngine's mandatory-handle check).
    if block_mode {
        return;
    }

    // In the fused path the last layer's end-of-layer fused norm already applied the FINAL norm
    // (its next_gin was n.fin), so n.hn holds the final-normed row and c_f is just that dep.
    let c_f = if fuse_norm || gfuse {
        dep
    } else {
        b.emit(DevOp::RmsNorm, rows.clone(), &[dep], |d| {
            d.t[0] = n.hn;
            d.t[1] = n.x;
            d.t[2] = n.fin;
            d.i[0] = t;
            d.i[1] = c.hidden;
            d.f[0] = c.eps;
        })
    };
    let head_src = n.hn;
    // lm_head over the LAST row only (i4 = a_row0). Weight is the tied embedding table, or the
    // separate lm_head.weight when the checkpoint does not tie (Llama).
    let head_w = if c.tied { n.emb } else { n.head };
    // lm_head is COLUMN(vocab)-parallel (plans/tp-design.md §3a/§8d): each rank produces its
    // vocab_l logit lanes. tp-host binds the rank's vocab slice of the (replicated) weight.
    // PLOW_PF_GEMV_HEAD=1 (PX-6 recommendation A): prefill lm_head is M=1 (see `lm_m` below), but
    // `pick_tile` hands it to the TILED arm, which computes BM=128 rows to keep one — 0.78% row
    // efficiency. Measured 1.991 -> 1.213 ms (-39%) on the FFMA GEMV arm, which reaches 98% of the
    // 5090's 1695.6 GB/s ceiling against the tiled arm's 60%. Decode already takes `DevOp::Gemv`
    // via `gemv_family`; this only redirects the PREFILL emit, hence the `!decode` guard.
    //
    // REQUIRES the prefill cubin built with `-DPLOW_NV_PF_GEMV_HEAD=1` — `case PLOW_DOP_GEMV` is
    // otherwise compiled out of that object (`#if !PLOW_NV_PREFILL`, interp_sm120.cu) and the
    // packet hits `default: __trap()`. That is the loud failure, not a silent wrong answer.
    // Unset = byte-identical. See perf-data/px6-sm-quantization.md.
    let pf_gemv_head =
        !decode && std::env::var("PLOW_PF_GEMV_HEAD").ok().as_deref() == Some("1");
    let lm_op = if gemv_family || pf_gemv_head {
        DevOp::Gemv
    } else {
        pick_tile(1, vocab_l, c.hidden, n_cu)
    };
    // PREFILL takes only the LAST prompt row's logits (M=1, a_row0=t-1). DECODE takes ALL t rows,
    // one per sequence (M=t, a_row0=0) — batch>1 samples a token per sequence. Decode B=1 gives
    // (M=1, a_row0=0), identical to the old (1, t-1) since t==1 there.
    let (lm_m, lm_row0) = if decode { (t, 0) } else { (1, t - 1) };
    // PLOW_FP8_HEAD: weight-only fp8 lm_head (GemvFp8, dequant-on-load, per-row scale).
    // The tied embedding LOOKUP stays bf16 (reads the original table); only the head GEMV
    // reads the fp8 twin. Own reporting row — vLLM's fp8 recipe keeps lm_head bf16.
    let fp8_head = decode && n.head8 != TENSOR_NONE;
    // E5 (rtx-19) PLOW_FUSE_ARGMAX: fuse the greedy-argmax epilogue (+ softcap) into the lm_head
    // GEMV, folding each block's owned vocab slice into an amax partial and dropping the SoftCap +
    // Argmax packets. Greedy B=1 decode on the bf16 head only (fp8 head keeps the classic path).
    let fuse_am = fuse_argmax_on() && decode && gemv_family && !fp8_head && t == 1;
    let lm_op = if fuse_am {
        DevOp::GemvArgmax
    } else if fp8_head {
        DevOp::GemvFp8
    } else {
        lm_op
    };
    let c_lm = b.emit(lm_op, all.clone(), &[c_f], |d| {
        d.t[0] = n.logits;
        d.t[1] = head_src;
        d.t[2] = if fp8_head { n.head8 } else { head_w };
        if fp8_head {
            d.t[5] = n.head8s;
        }
        if fuse_am {
            d.t[3] = n.amax; // packed-u64 partials, one per block
            d.f[0] = c.softcap; // reproduced in the epilogue; 0 = none
        }
        d.i[0] = lm_m;
        d.i[1] = vocab_l;
        d.i[2] = c.hidden;
        d.i[3] = 0;
        d.i[4] = lm_row0;
    });
    // Final-logit softcap: Gemma only (cap 30). Llama/Qwen have none, and d_softcap divides by
    // cap, so it must be SKIPPED (not emitted with cap 0) for them. Fused into GemvArgmax above.
    let c_logits = if fuse_am {
        c_lm
    } else if c.softcap > 0.0 {
        // BATCH>1: softcap the [t][vocab] logit tile (flat t*vocab). B=1 => vocab_l, identical.
        b.emit(DevOp::SoftCap, elem(lm_m * vocab_l), &[c_lm], |d| {
            d.t[0] = n.logits;
            d.t[1] = n.logits;
            d.i[0] = lm_m * vocab_l;
            d.f[0] = c.softcap;
        })
    } else {
        c_lm
    };

    // Greedy sample on device, and write the id into `in.ids` -- the very tensor the NEXT
    // step's EMBED reads. The host never sees the 512 KB logit row: it reads 4 bytes to print
    // the token and to check for EOS, and writes nothing back.
    // BATCH>1: i1 = n_batch. Each sequence argmaxes its OWN [vocab] row into amax[b][*] and
    // ArgmaxFin folds it into ids[b] — one token per sequence, no cross-sequence bleed. i1==0
    // (B=1/prefill) is the single-sequence path, byte-identical.
    // `decode` guard: in a PREFILL program t is the BUCKET SIZE (128..8192), not a batch —
    // without the guard every prefill bucket emitted argmax over t "sequences", reading
    // t*vocab logits from the [dbatch][vocab] tensor (a 64 MiB OOB read at t=8192) and
    // clobbering ids[0..t]. Prefill is always single-sequence (lm_head M=1 → logits row 0).
    let nb_argmax = if decode && t > 1 { t } else { 0 };
    // FUSED (fuse_am): GemvArgmax already wrote the `all.len()` partials — skip the Argmax packet
    // and fold that many. CLASSIC: the 64-block Argmax strides the full vocab, folding AMAX_BLOCKS.
    let (c_am, nparts) = if fuse_am {
        (c_lm, all.len() as u32)
    } else {
        let amax_cus: Vec<u32> = (0..AMAX_BLOCKS).collect();
        let c_am = b.emit(DevOp::Argmax, amax_cus, &[c_logits], |d| {
            d.t[0] = n.amax;
            d.t[1] = n.logits;
            d.i[0] = vocab_l;
            d.i[1] = nb_argmax;
        });
        (c_am, AMAX_BLOCKS)
    };
    let c_fin = b.emit(DevOp::ArgmaxFin, vec![0], &[c_am], |d| {
        d.t[0] = n.ids;
        d.t[1] = n.amax;
        d.i[0] = nparts;
        d.i[1] = nb_argmax;
    });
    // lm_head is REPLICATED under TP (see declare() note): every rank computes the full-vocab
    // argmax and thus the SAME global token id, so no cross-rank XArgmaxFin fold is needed. The
    // sharded lm_head + XArgmaxFin id-fold is a Phase-3 item (§8d, §13); c_fin already wrote the
    // correct global id into in.ids on every rank.
    let _ = c_fin;
}

/// Blocks the argmax partial reduction is spread over. 64 x 512 threads covers a 262144-entry
/// vocab in one strided pass per thread.
const AMAX_BLOCKS: u32 = 64;

/// E5 (rtx-19): PLOW_FUSE_ARGMAX fuses the greedy-argmax epilogue into the lm_head GEMV
/// (`DevOp::GemvArgmax`), replacing the `SoftCap` + `Argmax` packets. Default off → byte-identical.
fn fuse_argmax_on() -> bool {
    std::env::var("PLOW_FUSE_ARGMAX").ok().as_deref() == Some("1")
}

/// Argmax-partial slot count: when fused the lm_head runs on all `n_cu` blocks (one partial each),
/// so the buffer and `ArgmaxFin` fold must cover `max(AMAX_BLOCKS, n_cu)`; classic keeps AMAX_BLOCKS.
fn fuse_argmax_parts(n_cu: u32) -> u32 {
    if fuse_argmax_on() {
        n_cu.max(AMAX_BLOCKS)
    } else {
        AMAX_BLOCKS
    }
}

// ============================================================================
// GLM-5.2-FP8 (GlmMoeDsa) — MLA + DSA + block-fp8 MoE serving path.
//
// A DeepSeek-V3.2-class model: Multi-head Latent Attention (absorbed q_nope/value
// folds + partial INTERLEAVED RoPE on the 64 rope dims), a DSA "lightning indexer"
// (ctx>2048; a no-op below), and a fine-grained sigmoid-router block-fp8 MoE
// (256 routed experts, top-8, +e_score_correction_bias, norm_topk, route_scale 2.5,
// 1 shared expert). This is a WHOLLY SEPARATE emit path from the dense-GQA
// emit_phase above — the op set (FLASH_MLA_DECODE/O_UV_FOLD/MOE_*) and the derived
// weights (absorbed Wqa/Wuv) share nothing with Gemma/Llama/Qwen.
//
// The op sequence emit_glm_block produces is the EXACT 34-op block validated on
// gfx950 by runtime/tests/glm52_real_block_gfx950_test.c against the HF oracle
// (real 256 experts, real [128,128] block-fp8 scales) — see plans/glm52-campaign.md
// "B4-CORE DONE". The offline glm_tests below assert byte-for-op equality with that
// reference, so the emitted layer inherits the harness's passing GPU result.
//
// MILESTONE-1 STAGING (plans/glm52-campaign.md): the query/key RoPE is folded into
// the derived weights at a FIXED position by the host weight-prep (as the B4 harness
// did) — valid for single-token validation. The dynamic INTERLEAVED-RoPE op (coming
// from the kernels branch) replaces the fold for milestone-3 multi-token decode.
// ============================================================================

/// GLM-5.2 (GlmMoeDsa) config — parsed from the real `config.json`. Dims verified in

/// Everything the device-blob emitter needs from the caller. The `PLOW_*`
/// environment knobs (fp8, uniseg, decode-batch, …) are still read inside the
/// emit paths exactly as before; this struct carries only what used to be
/// positional/named CLI arguments so plowc and the legacy `gemma4` bin can both
/// drive the same code.
#[derive(Clone, Debug)]
pub struct EmitArgs {
    /// HuggingFace checkpoint directory (config.json + safetensors).
    pub dir: PathBuf,
    /// Max context tokens the program is compiled for.
    pub ctx: u32,
    /// Output `.pkt` path (a `block.json`/sidecar may be written next to it).
    pub out: String,
    /// Target executor (SM/CU) count.
    pub n_cu: u32,
    /// Tensor-parallel degree (>= 1).
    pub tp: u32,
    /// `--block l` or `l..r` (env `PLOW_BLOCK` fallback): single-block extract.
    pub block_spec: Option<String>,
    /// `--embed-cubin`: interpreter cubin embedded as a blob section.
    pub embed_cubin: Option<String>,
    /// `--embed-hsaco`: interpreter hsaco embedded as a blob section.
    pub embed_hsaco: Option<String>,
    /// Declare the RoPE tables as recipes the runtime materialises (v7 blob)
    /// instead of expanding them into the init section. On by default; the C
    /// harnesses under `runtime/tests/` need it off — see [`Model::bake_gen`].
    pub rope_gen: bool,
    /// `PLOW_NV_PLACE`: `(sms_per_partition, partition_count)` of the target GPU
    /// (from `hwspec::GpuSpec::l2_partitioning`). `Some` groups the device blob's
    /// global-queue stream by L2 domain (via [`packet::devbuild::Builder`]'s
    /// `seg`-as-domain), so a physical-SM-aware interp pulls its domain's
    /// packets. `None` ⇒ byte-identical. Dense-GQA path only. See
    /// `plans/devblob-locality-placement.md`.
    pub l2_layout: Option<(u32, u32)>,
    /// Target GPU spec name (e.g. `"H100 SXM5"`), stamped into the blob header as
    /// [`packet::devbuild::gpu_fingerprint`] so the runtime can warn on a GPU
    /// mismatch. Empty ⇒ unknown (check skipped). Set by plowc from `--gpu`.
    pub gpu: String,
    /// Target ISA for the `build.json` manifest (`"sm_120a"`, `"gfx950"`, …).
    /// METADATA ONLY: it changes nothing about the emitted blob — it is carried
    /// so the manifest says which toolchain a backend should render flags for.
    /// Empty ⇒ the manifest is not written (the legacy `gemma4` CLI).
    pub arch: String,
}

/// Read-only verification hook for [`run_verified`], called with the finished
/// [`packet::devbuild::Model`] immediately before the blob is written
/// (dense-GQA path only for now). An `Err` ABORTS emission.
pub type VerifyHook = Box<dyn Fn(&packet::devbuild::Model) -> Result<(), String>>;

impl EmitArgs {
    /// Parse the legacy `gemma4`/`tinygemma` CLI: named flags anywhere, then
    /// positional `<model-dir> <max_ctx> <out.pkt> [n_cu]`. `PLOW_BLOCK` is the
    /// `--block` fallback. Preserved verbatim so the two entry points agree.
    pub fn from_cli(argv: impl Iterator<Item = String>) -> EmitArgs {
        let mut tp: u32 = 1;
        let mut embed_cubin: Option<String> = None;
        let mut embed_hsaco: Option<String> = None;
        let mut block_spec: Option<String> =
            std::env::var("PLOW_BLOCK").ok().filter(|s| !s.is_empty());
        let mut pos: Vec<String> = Vec::new();
        let mut it = argv;
        while let Some(a) = it.next() {
            match a.as_str() {
                "--tp" => {
                    tp = it
                        .next()
                        .expect("--tp needs a value")
                        .parse()
                        .expect("--tp N")
                }
                s if s.starts_with("--tp=") => tp = s[5..].parse().expect("--tp=N"),
                "--block" => {
                    block_spec = Some(it.next().expect("--block needs a value (l or l..r)"));
                }
                s if s.starts_with("--block=") => {
                    block_spec = Some(s["--block=".len()..].to_string());
                }
                "--embed-cubin" => {
                    embed_cubin = Some(it.next().expect("--embed-cubin needs a path"));
                }
                s if s.starts_with("--embed-cubin=") => {
                    embed_cubin = Some(s["--embed-cubin=".len()..].to_string());
                }
                "--embed-hsaco" => {
                    embed_hsaco = Some(it.next().expect("--embed-hsaco needs a path"));
                }
                s if s.starts_with("--embed-hsaco=") => {
                    embed_hsaco = Some(s["--embed-hsaco=".len()..].to_string());
                }
                _ => pos.push(a),
            }
        }
        let mut pa = pos.into_iter();
        let dir = PathBuf::from(
            pa.next()
                .expect("usage: gemma4 [--tp N] [--embed-cubin <path>] [--embed-hsaco <path>] <model-dir> <max_ctx> <out.pkt> [n_cu]"),
        );
        let ctx: u32 = pa.next().expect("max_ctx").parse().unwrap();
        let out = pa.next().unwrap_or_else(|| "gemma4.pkt".into());
        let n_cu: u32 = pa.next().and_then(|s| s.parse().ok()).unwrap_or(256);
        EmitArgs {
            dir,
            ctx,
            out,
            n_cu,
            tp,
            block_spec,
            embed_cubin,
            embed_hsaco,
            rope_gen: true,
            l2_layout: None,
            gpu: String::new(),
            arch: String::new(),
        }
    }
}

/// Uniform surface for lowering a checkpoint into a PLOWDEV program set, per model
/// family (plans/devgen-trait-refactor.md). Phase 0: only the dense-GQA family
/// implements it, and only the `emit_phase` call sites are routed through it —
/// `declare`/orchestration and the GLM/Nemotron families move behind it in later
/// phases, at which point these signatures generalize. Behavior is byte-identical
/// to the free-function calls it forwards to (guarded by the dense goldens).
trait DevblobEmitter {
    /// Emit one prefill bucket program (`t` rows) into `b`.
    fn emit_prefill(&self, b: &mut Builder, t: u32);
    /// Emit the decode program into `b`; `kv_rows` collects the KV-write inst indices.
    fn emit_decode(&self, b: &mut Builder, dbatch: u32, dmode: Mode, kv_rows: &mut Vec<u32>);
}

/// Dense GQA (Gemma / Llama / Qwen). Arch is DATA (`Cfg.arch` switches), so ONE
/// emitter covers all three; it forwards to the shared `emit_phase`. See the
/// "one DenseGqaEmitter" design decision in plans/devgen-trait-refactor.md.
struct DenseGqaEmitter<'a> {
    c: &'a Cfg,
    ls: &'a [f32],
    /// Declared tensor-handle bundle (owned; produced by `declare` in [`Self::new`]).
    tn: Tn,
    n_cu: u32,
    ctx: u32,
    fp8: bool,
    w8a8: bool,
    fp8_kv: bool,
    fp8_kv_full: bool,
    block: std::ops::Range<usize>,
    block_mode: bool,
}

impl<'a> DenseGqaEmitter<'a> {
    /// Phase 1: the emitter owns the dense tensor declaration. Runs `declare`
    /// into a fresh Builder and returns the emitter plus the declared tensor decls
    /// + gen recipes the `Model` needs. Byte-identical to the old inline `declare`
    /// call (same args, same order).
    #[allow(clippy::too_many_arguments)]
    fn new(
        c: &'a Cfg,
        ls: &'a [f32],
        n_cu: u32,
        ctx: u32,
        fp8: bool,
        w8a8: bool,
        fp8_kv: bool,
        fp8_kv_full: bool,
        block: std::ops::Range<usize>,
        block_mode: bool,
        ns_pre: u32,
        dbatch: u32,
        moe_pf: bool,
    ) -> (Self, Vec<packet::devbuild::TensorDecl>, Vec<GenTensor>) {
        let mut tb = Builder::new(n_cu);
        let tn = declare(
            &mut tb, c, ctx, ns_pre, fp8, w8a8, fp8_kv, fp8_kv_full, dbatch, moe_pf, block.clone(),
        );
        let tensors = tb.tensors();
        let gen = tb.gen_tensors();
        let e = DenseGqaEmitter {
            c,
            ls,
            tn,
            n_cu,
            ctx,
            fp8,
            w8a8,
            fp8_kv,
            fp8_kv_full,
            block,
            block_mode,
        };
        (e, tensors, gen)
    }
}

impl DevblobEmitter for DenseGqaEmitter<'_> {
    fn emit_prefill(&self, b: &mut Builder, t: u32) {
        let mut dummy = Vec::new();
        emit_phase(
            b, self.c, self.ls, &self.tn, t, self.ctx, Mode::Prefill, self.n_cu, &mut dummy,
            self.fp8, self.w8a8, self.fp8_kv, self.fp8_kv_full, self.block.clone(),
            self.block_mode,
        );
    }
    fn emit_decode(&self, b: &mut Builder, dbatch: u32, dmode: Mode, kv_rows: &mut Vec<u32>) {
        // Decode passes w8a8=false, exactly as the historical call site did.
        emit_phase(
            b, self.c, self.ls, &self.tn, dbatch, self.ctx, dmode, self.n_cu, kv_rows, self.fp8,
            false, self.fp8_kv, self.fp8_kv_full, self.block.clone(), self.block_mode,
        );
    }
}

/// Compile a checkpoint into a PLOWDEV device blob at `args.out`. This is the
/// former `gemma4` binary's `main`, verbatim below the argument parsing — the
/// same arch dispatch, the same env knobs, the same byte output.
pub fn run(args: EmitArgs) {
    run_verified(args, None)
}

/// [`run`] plus an optional pre-write verification gate (see [`VerifyHook`]).
/// `run(args)` ≡ `run_verified(args, None)` — byte-identical emission.
pub fn run_verified(args: EmitArgs, verify: Option<VerifyHook>) {
    let EmitArgs {
        dir,
        ctx,
        out,
        n_cu,
        tp,
        block_spec,
        embed_cubin,
        embed_hsaco,
        rope_gen,
        l2_layout,
        gpu,
        arch,
    } = args;

    // GLM-5.2 (GlmMoeDsa) — MLA + DSA + block-fp8 MoE — is a wholly separate emit path (glm_main).
    // Dispatch on model_type before the dense-GQA cfg parse, which would panic on GLM's config.
    let model_type =
        serde_json::from_slice::<Value>(&std::fs::read(dir.join("config.json")).unwrap())
            .ok()
            .and_then(|v| {
                v.get("model_type")
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_default();
    // PLOW_NV_PLACE is wired only on the dense-GQA path below (b/bd builders). The
    // GLM/Kimi/DeepSeek/Nemotron emitters have their own builders and never call
    // set_l2_placement, so the flag would silently no-op there — say so rather
    // than let a user believe placement is active. See devblob-locality-placement.md.
    if l2_layout.is_some()
        && matches!(
            model_type.as_str(),
            "glm_moe_dsa" | "kimi_k2" | "kimi" | "deepseek_v3" | "deepseek_v2" | "nemotron_h"
                | "nemotron3" | "nemotron"
        )
    {
        eprintln!(
            "  PLOW_NV_PLACE ignored: L2-domain placement is dense-GQA only, not wired for \
             model_type {model_type:?} (its emitter is a separate path)"
        );
    }
    if model_type == "glm_moe_dsa" {
        // GLM `--block` (M2, plans/block-asset-harness.md §5.3/§7): single-block
        // extraction on the separate GLM emitter. Absent => the unchanged glm_main
        // path (byte-identical).
        match &block_spec {
            Some(spec) => glm_emit_block(&dir, ctx, &out, n_cu, tp, spec, rope_gen),
            None => glm_main(&dir, ctx, &out, n_cu, tp, rope_gen),
        }
        return;
    }
    // Kimi K2.7 / DeepSeek-V2/V3 (plan §5.0/§5.3, M3): plain MLA + MoE, reusing GLM's MLA + MoE emit
    // (NOT rewrite/kimi.rs). model_type "kimi_k2"/"kimi" => Kimi tag; "deepseek_v3"/"deepseek_v2" =>
    // DeepSeek tag. Only the block-extraction (`--block`) device path is wired in M3; a full-model
    // Kimi device emit (the glm_main analogue) is a later milestone.
    if matches!(
        model_type.as_str(),
        "kimi_k2" | "kimi" | "deepseek_v3" | "deepseek_v2"
    ) {
        let arch = if model_type.starts_with("kimi") {
            MlaArch::Kimi
        } else {
            MlaArch::DeepSeek
        };
        match &block_spec {
            Some(spec) => kimi_emit_block(&dir, ctx, &out, n_cu, tp, spec, arch, rope_gen),
            None => panic!(
                "{model_type}: M3 supports only single-block extraction on the device path — pass \
                 --block <l>[..<r>] (or PLOW_BLOCK). Full-model Kimi/DeepSeek device emit is a later \
                 milestone (plans/block-asset-harness.md §5.3)."
            ),
        }
        return;
    }
    // Nemotron-3 Mamba-2 hybrid (plan §7, M4): mamba mixer (NEW op) + GQA attn + MoE, one block at a
    // time. Only the `--block` device path is wired in M4; a full-model Nemotron emit is a later
    // milestone (the hybrid layer-count + carried-state plumbing).
    if matches!(model_type.as_str(), "nemotron_h" | "nemotron3" | "nemotron") {
        match &block_spec {
            Some(spec) => nemotron_emit_block(&dir, ctx, &out, n_cu, tp, spec, rope_gen),
            None => panic!(
                "{model_type}: M4 supports only single-block extraction — pass --block <l>[..<r>] \
                 (or PLOW_BLOCK). Full-model Nemotron device emit is a later milestone \
                 (plans/block-asset-harness.md §5.3/§7)."
            ),
        }
        return;
    }

    // Phase 2 (plans/devgen-trait-refactor.md): the dense-GQA family is its own
    // function, so run_verified is now pure dispatch (GLM/Kimi/Nemotron early-return
    // above; everything else is dense). Byte-identical — the body moved verbatim.
    emit_dense_gqa(
        dir, ctx, out, n_cu, tp, block_spec, embed_cubin, embed_hsaco, rope_gen, l2_layout, gpu,
        arch, verify,
    );
}

/// Dense-GQA (Gemma / Llama / Qwen) full device-blob emit: parse config, size the
/// bucket ladder, declare tensors + emit every program via [`DenseGqaEmitter`],
/// then serialize + coverage-gate + write the blob. Split out of `run_verified` so
/// dispatch is a clean match; GLM/Kimi/Nemotron are their own emitters.
#[allow(clippy::too_many_arguments)]
fn emit_dense_gqa(
    dir: PathBuf,
    ctx: u32,
    out: String,
    n_cu: u32,
    tp: u32,
    block_spec: Option<String>,
    embed_cubin: Option<String>,
    embed_hsaco: Option<String>,
    rope_gen: bool,
    l2_layout: Option<(u32, u32)>,
    gpu: String,
    arch: String,
    verify: Option<VerifyHook>,
) {
    // Empty --gpu ⇒ unknown target (0), not fnv("") — so the header stamp is 0
    // and unspecified-GPU blobs stay byte-stable (e.g. the golden test).
    let target_fp = if gpu.is_empty() { 0 } else { packet::devbuild::gpu_fingerprint(&gpu) };
    let mut c = cfg_from(&dir);
    assert!(tp >= 1, "--tp must be >= 1");
    c.tp = tp;
    // Resolve the block range now that layer count is known. `l` -> l..l+1;
    // `l..r` -> that half-open range. Absent => the full model (0..layers),
    // which makes every gated site below byte-identical to the pre-block path.
    let block: std::ops::Range<usize> = match &block_spec {
        None => 0..c.layers as usize,
        Some(s) => parse_block(s, c.layers as usize),
    };
    let block_mode = block_spec.is_some();
    // FP8 (PLOW_FP8=1). The 7 projections gain an fp8 (w8a16) twin + per-channel scale. DECODE emits
    // GEMV_FP8 / GEMV_GLU_FP8; PREFILL emits GEMM_FP8 / GEMM_GLU_FP8 (T6 L2 — dequant-to-bf16-in-smem
    // + existing bf16 mma, per-channel scale in the epilogue). Both phases consume the fp8 twins, so
    // the bf16 projection weights are elided in fp8 mode (see `wproj`). The bf16 pkt is byte-identical
    // when unset. See runtime/nvidia/op_gemm.cuh, runtime/amd/op_gemm.h and gemma4_chat.c.
    let fp8 = std::env::var("PLOW_FP8").ok().as_deref() == Some("1");
    // T8 w8a8 (PLOW_W8A8=1, requires PLOW_FP8=1). PREFILL emits the true fp8 tensor-core path:
    // ONE per-row DevOp::QuantFp8 per activation site + GEMM_FP8/GEMM_GLU_FP8 re-pointed at the
    // fp8 activation (t1=xq) + a_scale (t3). The SAME opcodes serve T6 w8a16 (bf16 activation) —
    // the interp cubin selects the kernel by PLOW_NV_W8A8, so the w8a8 pkt MUST run against a
    // PLOW_NV_W8A8=1 prefill cubin (the T6 cubin would misread xq bytes as bf16). Weight side =
    // the same e4m3 twins + per-channel scales T6 declared. Unset => byte-identical emission.
    let w8a8 = std::env::var("PLOW_W8A8").ok().as_deref() == Some("1");
    assert!(
        !w8a8 || fp8,
        "PLOW_W8A8=1 requires PLOW_FP8=1 (the fp8 weight twins + scales)"
    );
    // FP8 KV-CACHE (PLOW_FP8_KV=1). Stores/reads K/V as e4m3 with a per-row f32 scale, halving the
    // decode KV stream (the HBM-bound part of flash-decode) and the KV footprint. Independent of the
    // fp8 WEIGHT path above so both can be A/B'd; the harness routes an fp8-KV pkt to the _fp8kv
    // interpreter objects (which carry the fp8 flash + HeadNormRopeFp8 arms).
    let fp8_kv = std::env::var("PLOW_FP8_KV").ok().as_deref() == Some("1");
    // PLOW_FP8_KV_FULL=1: restrict the e4m3 cache to FULL-attention (hd512) layers — the shape
    // the beat-fp8-mma PIPE=1 fp8-mma prefill flash serves. Requires PLOW_FP8_KV=1.
    let fp8_kv_full = fp8_kv && std::env::var("PLOW_FP8_KV_FULL").ok().as_deref() == Some("1");
    // layer_scalar is a Gemma-only learned per-layer residual scale; Llama/Qwen fold nothing here.
    let ls = if c.arch == Arch::Gemma4 {
        layer_scalars(&dir, c.layers, &c.prefix)
    } else {
        vec![1.0f32; c.layers as usize]
    };

    // Prefill BUCKETS. A 20-token prompt must not pay for a 4096-token program, and T is a
    // compile-time constant of the packets — so the compiler emits several and the runtime
    // picks the smallest that fits. This is what a shape bucket IS.
    // CAPPED AT MAX_CHUNK. Chunked prefill never emits a chunk larger than PLOW_MAX_CHUNK, so a
    // program for T > MAX_CHUNK can never be invoked -- the ladder used to run to 131072 and
    // every rung above 4096 was dead code that still cost compile time and packet size.
    // Tensor-parallel now emits SHARDED PREFILL buckets too (plans/tp-prefill.md): every prefill
    // op is Megatron-sharded in emit_phase (q/k/v/gate/up column-parallel, o/down row-parallel with
    // an XReduce all-reduce, flash head-split) exactly as decode is — the [T,hidden] all-reduce is
    // the only new regime. The full ladder is emitted at every tp; tp==1 stays byte-identical.
    // PLOW_PF_LADDER=wave (PX-6): derive the rungs from this GPU's SM count instead of the
    // power-of-two ladder below. Prefill GEMM cost is a STAIRCASE in tm = ceil(t/BM) -- flat
    // between wave boundaries, so rows inside a tread are free and one row past a tread top
    // costs a whole extra wave. The power-of-two rungs are unrelated to where the treads are:
    // measured on a 170-SM 5090 with the real Gemma-4-12B op mix they give up 9.6% of prefill
    // GEMM time on average over L=128..4096 (worst cells +41.9% at 640 rows, which must be
    // served as 128+512, and +31.6% at 1280 -> 512+1024). The four tread-top rungs the model
    // picks -- 1408, 2176, 640, 1792 -- take the mean loss to 1.4%, and not one is a power of
    // two. See perf-data/px6-sm-quantization.md.
    //
    // Same rung COUNT as the shipped ladder, so blob size and compile time are unchanged; only
    // the rung POSITIONS move. Unset = byte-identical.
    //
    // BM/BN = 128 is the sm_120 tile (PGM_BM/PGM_BN, runtime/nvidia/op_gemm.cuh:712-720). The
    // AMD bodies tile differently, so the ladder is only derived on the NVIDIA path; gate it on
    // the flag rather than guessing the target's tile from here.
    const LADDER_BM: u32 = 128;
    const LADDER_BN: u32 = 128;
    let cap = ctx.min(max_chunk(c.window));
    let shipped: Vec<u32> =
        [128u32, 512, 1024, 2048, 4096, 8192].into_iter().filter(|&x| x <= cap).collect();
    let buckets: Vec<u32> = if std::env::var("PLOW_PF_LADDER").ok().as_deref() == Some("wave") {
        let ops = ladder::ladder_ops(&c, LADDER_BN);
        let max_tm = cap.div_ceil(LADDER_BM).max(1);
        let l: Vec<u32> = ladder::wave_ladder(n_cu, &ops, max_tm, shipped.len())
            .into_iter()
            .map(|tm| tm * LADDER_BM)
            .filter(|&x| x <= cap)
            .collect();
        println!("prefill ladder (PLOW_PF_LADDER=wave, n_cu={n_cu}): {l:?}  [was {shipped:?}]");
        l
    } else {
        shipped
    };
    // The invariant that ties MAX_CHUNK to KV_RING (see dev_isa.h). Break it and a chunk's own
    // rows wrap onto their history: a silent wrong answer, not a crash.
    let chunk = max_chunk(c.window);
    let ring = kv_ring_rows(c.window, chunk);
    assert!(
        ring >= c.window + chunk - 1,
        "KV ring {ring} too small for window {} + chunk {chunk}",
        c.window
    );
    let arows = ctx.min(max_chunk(c.window));
    // opart/mlpart (the flash_prefill partials) are sized in declare() as arows*heads_sharded*ns_pre.
    // The flash writes t*heads_sharded*ns(t) row-splits for a bucket t, where emit_phase derives
    // ns(t) from the SHARDED head count (heads/tp) — so ns_pre must be the worst-case over buckets
    // using that same sharded count, or a high-tp small-bucket program overflows opart (at tp=8 the
    // real ns is 32x the unsharded estimate → a GPU write fault). tp==1: hs==heads, ns_pre==1 (the
    // old value, byte-identical). plans/tp-prefill.md.
    let hs = (c.heads / c.tp).max(1);
    let max_splits = buckets
        .iter()
        .map(|&t| {
            let ns = n_cu.div_ceil((t.div_ceil(Q_TILE_ROWS) * hs).max(1)).max(1);
            t * hs * ns
        })
        .max()
        .unwrap_or(n_cu * Q_TILE_ROWS);
    let ns_pre = max_splits.div_ceil((arows * hs).max(1)).max(1);

    // BATCH>1 DECODE (serving pending #4, "max users supported"): PLOW_DECODE_BATCH=B emits a
    // WORKING batch-B decode program — KV cache, activations, GEMV M, flash n_batch, per-sequence
    // argmax all sized/set for B. B=1 (default) is byte-identical to the pre-batch blob (the serving
    // engine depends on it). B is capped at 32 — serving up to 32 concurrent users. The GEMV
    // ladder instantiates MM in {1,2,4,8} and every dispatcher (d_gemv, d_gemv_qkv, d_gemv_glu
    // and the fp8 twins) walks M in blocks of 8 above that, so the cap is a policy/KV-footprint
    // choice, not a kernel limit: the KV cache is sized dbatch* (7 GiB/seq at ctx=132k on 12B),
    // so B=32 only fits at a reduced ctx. Raising it further needs no kernel work.
    let dbatch: u32 = std::env::var("PLOW_DECODE_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .clamp(1, 32);
    // 26B-A4B MoE decode is BATCHED (B in 1..=32): the router family, the flat expert GLU/down
    // and the combine all carry a batch row count and index [B][k] routing slots. See the
    // work-item ordering note in runtime/nvidia/op_moe.cuh for the weight-reuse design.
    // (The fp8 batch refusal is gone: the fp8 GEMV arms are batched as of the B=32 work.)
    assert!(
        !(c.moe && dbatch > 32),
        "MoE decode batch is capped at 32 (per-CTA inv[] scratch, PLOW_MOE_MAXB)"
    );

    // Grouped-MoE PREFILL (plans/p9-26b-prefill-moe.md): token-sorted grouped expert GEMM buckets.
    // Enabled by default for the 26B-A4B MoE bf16 path; PLOW_MOE_PREFILL=0 restores the decode-only
    // blob (byte-identical to the pre-prefill build — the buffer sizing and new tensors are gated on
    // this flag). beat26b: fp8 grouped MoE prefill is now implemented for the w8a8 path (ops 81/82),
    // so it is enabled under PLOW_W8A8; plain fp8 (w8a16 dequant) grouped prefill is still not
    // implemented and stays decode-only.
    let moe_pf = c.moe
        && (!fp8 || w8a8)
        && std::env::var("PLOW_MOE_PREFILL").ok().as_deref() != Some("0");

    // Phase 1 (plans/devgen-trait-refactor.md): the DenseGqaEmitter owns the dense
    // tensor declaration (declare) and the emit_phase call sites. Byte-identical —
    // `new` forwards to the same `declare`, `emit_*` to the same `emit_phase`.
    let (emitter, tensors, gen) = DenseGqaEmitter::new(
        &c, &ls, n_cu, ctx, fp8, w8a8, fp8_kv, fp8_kv_full, block.clone(), block_mode, ns_pre,
        dbatch, moe_pf,
    );

    let mut progs = Vec::new();
    let mut tlist = Vec::new();
    for &t in &buckets {
        if c.moe && !moe_pf {
            break;
        } // MoE without prefill: decode-only blob
        let mut b = Builder::new(n_cu);
        b.adopt_tensors(tensors.clone());
        b.set_l2_placement(l2_layout); // PLOW_NV_PLACE: None ⇒ byte-identical
        emitter.emit_prefill(&mut b, t);
        progs.push(b.finish());
        tlist.push(t);
    }
    let mut bd = Builder::new(n_cu);
    bd.adopt_tensors(tensors.clone());
    bd.set_l2_placement(l2_layout); // PLOW_NV_PLACE: None ⇒ byte-identical
    let mut kv_rows = Vec::new();
    // `dbatch` is the SAME clamped(1,32) value used by declare() above — emission and
    // allocation must agree, so we reuse it here rather than re-reading the env (an unclamped
    // re-read would emit B>32 ops against buffers declare() sized for 32 → OOB writes).
    // DECODE-TILED (PLOW_DECODE_TILED=1, plans/decode-tiled.md): emit the decode bucket from
    // PREFILL kernels — tiled GEMM + FlashPrefill at one query row — instead of the GEMV family.
    // Targets long context, where GEMV does not scale with batch and FlashDecode caps at n_cu.
    // Unset emits a byte-identical program. **The sm_120 interpreter traps on every prefill
    // opcode** (interp_sm120.cu default arm), so this is AMD-only until those kernels exist; it
    // is a loud trap, not silent garbage. Correctness bar is a token stream IDENTICAL to the
    // Mode::Decode bucket at the same prompt — not "it ran".
    let dmode = if std::env::var("PLOW_DECODE_TILED").ok().as_deref() == Some("1") {
        Mode::DecodeTiled
    } else {
        Mode::Decode
    };
    emitter.emit_decode(&mut bd, dbatch, dmode, &mut kv_rows);
    progs.push(bd.finish());
    tlist.push(dbatch);

    let mut m = Model {
        n_cu,
        target: target_fp, // GPU fingerprint -> BlobHeader.target (runtime GPU-mismatch warn)
        tensors,
        progs,
        kv_row_insts: kv_rows,
        prog_t: tlist,
        gen,
    };

    // Emit v6 with sections when --embed-cubin/--embed-hsaco given, else v5.
    let mut sections = Vec::new();
    // BLOCK MODE: embed the block.json descriptor (plans/block-asset-harness.md
    // §4) as SECT_METADATA — this also forces the to_blob_v6 path — and drop a
    // sibling block.json next to the blob for the record / the harness loader.
    if block_mode {
        use plow_asset::*;
        let l0 = block.start;
        let full = c.is_full[l0];
        let head_dim = if full { c.hd_full } else { c.hd_slide };
        let kv_heads = if full { c.kvh_full } else { c.kvh_slide };
        let kv_tensors: Vec<String> = block
            .clone()
            .flat_map(|l| [format!("kv.{l}.k"), format!("kv.{l}.v")])
            .collect();
        let hidden = c.hidden as i64;
        let ckpt = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| dir.to_string_lossy().into_owned());
        // Top-level dtype reflects the actual compile (fp8 weight twins vs
        // bf16); the act.x tensors stay bf16 (fp8 is weight-only w8a16).
        let desc = BlockDescriptor {
            model: ckpt.clone(),
            arch: "gemma_dense".into(),
            layer: l0 as u32,
            kind: vec!["dense_attn".into(), "dense_ffn".into()],
            hidden,
            dtype: if fp8 { "fp8".into() } else { "bf16".into() },
            dims: BlockDims {
                heads: Some(c.heads as i64),
                head_dim: Some(head_dim as i64),
                kv_heads: Some(kv_heads as i64),
                ..Default::default()
            },
            dsa_role: None,
            inputs: vec![BlockTensor {
                name: "act.x".into(),
                shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(hidden)],
                dtype: "bf16".into(),
            }],
            outputs: vec![BlockTensor {
                name: "act.x".into(),
                shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(hidden)],
                dtype: "bf16".into(),
            }],
            carried_state: vec![CarriedState {
                role: "kv".into(),
                tensors: kv_tensors,
                layout: "head_major".into(),
            }],
            weights: BlockWeights {
                mode: "symlink".into(),
                ckpt,
                prefix: format!("{}layers.{}.", c.prefix, l0),
            },
            programs: BlockPrograms {
                prefill_buckets: buckets.iter().map(|&t| t as i64).collect(),
                decode_t: dbatch as i64,
            },
        };
        sections.push(write_block_descriptor(&out, &desc));
        eprintln!("  block mode: layers {block:?}");
    }
    if let Some(ref path) = embed_cubin {
        sections.push(packet::devbuild::SectionData {
            kind: packet::devbuild::SECT_CUBIN,
            name: "interp_sm120".into(),
            data: std::fs::read(path).expect("--embed-cubin: cannot read file"),
        });
    }
    if let Some(ref path) = embed_hsaco {
        sections.push(packet::devbuild::SectionData {
            kind: packet::devbuild::SECT_HSACO,
            name: "interp_gfx950".into(),
            data: std::fs::read(path).expect("--embed-hsaco: cannot read file"),
        });
    }
    if !rope_gen {
        m.bake_gen();
    }
    // Read-only verification gate (EmitArgs::verify): runs against the exact
    // programs about to be serialized; a rejection aborts before any bytes
    // are written. `None` (the default) is a no-op — emitted bytes identical.
    if let Some(v) = &verify {
        if let Err(e) = v(&m) {
            panic!("devblob verification rejected the emitted program: {e}");
        }
    }
    let blob = if sections.is_empty() {
        m.to_blob()
    } else {
        m.to_blob_v6(&sections)
    };
    // Coverage gate BEFORE the blob lands on disk: a wrong .pkt that exists will be
    // benchmarked by someone. PLOW_SKIP_COVERAGE=1 is the deliberate escape hatch for
    // partial/renamed checkpoints; it is loud because it re-arms the silent-wrong-model
    // failure mode this gate exists to prevent.
    match validate_coverage(
        &dir,
        &c.prefix,
        &m.tensors.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
        block_mode.then(|| block.clone()),
    ) {
        Ok(()) => {}
        Err(e) if std::env::var("PLOW_SKIP_COVERAGE").ok().as_deref() == Some("1") => {
            eprintln!("*** PLOW_SKIP_COVERAGE=1 — EMITTING A MODEL KNOWN TO BE WRONG ***\n{e}");
        }
        Err(e) => {
            eprintln!("gemma4: {e}");
            std::process::exit(1);
        }
    }
    std::fs::write(&out, blob).unwrap();

    // BUILD MANIFEST (`build.json`, beside the .pkt). Derived from `m` — the exact
    // programs just serialized — so it cannot describe a packet other than the one
    // on disk. That is the whole point: the packet and the interpreter object were
    // two independent sources of truth, and every failure in the rtx-2x campaign
    // came out of the gap between them. See crates/devgen/src/manifest.rs.
    // Skipped when `arch` is empty (the legacy `gemma4` CLI), so that path's output
    // is unchanged.
    if !arch.is_empty() {
        let man = manifest::build(&m, &arch);
        let mpath = std::path::Path::new(&out).with_file_name("build.json");
        match serde_json::to_vec_pretty(&man).map(|b| std::fs::write(&mpath, b)) {
            Ok(Ok(())) => eprintln!("  build manifest -> {}", mpath.display()),
            Ok(Err(e)) => eprintln!("  WARN: build.json not written: {e}"),
            Err(e) => eprintln!("  WARN: build.json not serialized: {e}"),
        }
    }

    let wb: u64 = m
        .tensors
        .iter()
        .filter(|x| x.name.starts_with("model.") || x.name.starts_with("fp8/"))
        .map(|x| x.bytes)
        .sum();
    let kb: u64 = m
        .tensors
        .iter()
        .filter(|x| x.name.starts_with("kv."))
        .map(|x| x.bytes)
        .sum();
    let ab: u64 = m
        .tensors
        .iter()
        .filter(|x| x.name.starts_with("act."))
        .map(|x| x.bytes)
        .sum();
    eprintln!(
        "gemma4: {} layers ({} full)  hidden={} inter={}  heads={}  hd={}/{}  kvh={}/{}  vocab={}",
        c.layers,
        c.is_full.iter().filter(|x| **x).count(),
        c.hidden,
        c.inter,
        c.heads,
        c.hd_slide,
        c.hd_full,
        c.kvh_slide,
        c.kvh_full,
        c.vocab
    );
    eprintln!("  max_ctx={}  prefill buckets {:?} + decode", ctx, buckets);
    eprintln!("  layer_scalar[0..4] = {:?}", &ls[..4.min(ls.len())]);
    for (i, p) in m.progs.iter().enumerate() {
        eprintln!(
            "    prog {} (T={:>4}): {:>5} packets, {:>7} workgroup-packets",
            i,
            m.prog_t[i],
            p.insts.len(),
            p.stream.len()
        );
    }
    eprintln!(
        "  weights {:.1} GiB   KV cache {:.2} GiB   activations {:.2} GiB   -> {}",
        wb as f64 / (1u64 << 30) as f64,
        kb as f64 / (1u64 << 30) as f64,
        ab as f64 / (1u64 << 30) as f64,
        out
    );
}


fn split2(n: u32, a: u32, b: u32) -> (Vec<u32>, Vec<u32>) {
    let s = (((n as u64 * a as u64) / (a + b).max(1) as u64).max(1) as u32).min(n - 1);
    ((0..s).collect(), (s..n).collect())
}
fn split3(n: u32, a: u32, b: u32, c: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    if c == 0 {
        let (x, y) = split2(n, a, b);
        return (x, y, Vec::new());
    }
    let tot = (a + b + c).max(1) as u64;
    let sa = (((n as u64 * a as u64) / tot).max(1) as u32).min(n - 2);
    let sb = (((n as u64 * b as u64) / tot).max(1) as u32).min(n - sa - 1);
    (
        (0..sa).collect(),
        (sa..sa + sb).collect(),
        (sa + sb..n).collect(),
    )
}

#[cfg(test)]
mod gemma_router_emit_tests {
    use super::*;

    fn router_program(split_plan: Option<(u32, DevOp)>) -> packet::devbuild::Program {
        router_program_b(split_plan, 1)
    }

    fn router_program_b(
        split_plan: Option<(u32, DevOp)>,
        nrow: u32,
    ) -> packet::devbuild::Program {
        let mut b = Builder::new(188);
        let dep = b.emit(DevOp::Nop, vec![0], &[], |_| {});
        let _ = emit_gemma_moe_router(
            &mut b,
            dep,
            10, // table
            11, // residual
            12, // projection
            13, // channel scale
            14, // per-expert scale
            if split_plan.is_some() {
                15
            } else {
                TENSOR_NONE
            },
            2816,
            128,
            8,
            (2816.0f32).powf(-0.5),
            1e-6,
            split_plan,
            nrow,
        );
        b.finish()
    }

    /// BATCH B>1: the batch row count reaches every router op, top-k gets one CTA per row, and
    /// the score op's CTA count scales with the (row, expert) pair space.
    #[test]
    fn batched_router_emit_carries_b() {
        let plan = gemma_moe_router_split_plan(188, 128, 8);
        let p = router_program_b(plan, 8);
        let score = &p.insts[1];
        let topk = &p.insts[2];
        assert_eq!(score.i[2], 8, "score op carries B");
        assert_eq!(score.blocks, 128, "8 rows x 128 experts / 8 per CTA");
        assert_eq!(topk.i[3], 8, "top-k carries B");
        assert_eq!(topk.blocks, 8, "one top-k CTA per row");
        // B=1 must leave the immediate at 0 so the packet bytes never move.
        let p1 = router_program(gemma_moe_router_split_plan(188, 128, 1));
        assert_eq!(p1.insts[1].i[2], 0);
        assert_eq!(p1.insts[2].i[3], 0);
        assert_eq!(p1.insts[2].blocks, 1);
    }

    #[test]
    fn legacy_router_emit_stays_one_original_opcode() {
        let p = router_program(None);
        assert_eq!(p.insts.len(), 2);
        let r = &p.insts[1];
        assert_eq!(r.op, DevOp::MoeRouterGemma as u16);
        assert_eq!(r.blocks, 1);
        assert_eq!(r.t[..5], [10, 11, 12, 13, 14]);
        assert_eq!(r.i[..3], [2816, 128, 8]);
    }

    #[test]
    fn split_router_emits_parallel_score_then_one_cta_tail() {
        let p = router_program(Some((16, DevOp::MoeRouterGemmaScore)));
        assert_eq!(p.insts.len(), 3);
        let score = &p.insts[1];
        let tail = &p.insts[2];
        assert_eq!(score.op, DevOp::MoeRouterGemmaScore as u16);
        assert_eq!(score.blocks, 16);
        assert_eq!(score.t[..4], [15, 11, 12, 13]);
        assert_eq!(tail.op, DevOp::MoeRouterGemmaTopk as u16);
        assert_eq!(tail.blocks, 1);
        assert_eq!(tail.t[..3], [10, 15, 14]);
        assert_eq!(tail.wait_len, 1);
        assert_eq!(p.waits[tail.wait_ofs as usize].threshold, 16);
    }

    #[test]
    fn fast_router_is_a_distinct_default_off_score_opcode() {
        let p = router_program(Some((16, DevOp::MoeRouterGemmaScoreFast)));
        assert_eq!(p.insts[1].op, DevOp::MoeRouterGemmaScoreFast as u16);
        assert_eq!(p.insts[1].blocks, 16);
        assert_eq!(p.insts[2].op, DevOp::MoeRouterGemmaTopk as u16);
    }
}

#[cfg(test)]
mod mode_tests {
    //! `Mode` exists to split the old `decode: bool` into two independent axes. The ONLY thing
    //! that keeps that refactor honest is that the two pre-existing corners still decode to the
    //! same pair of booleans they were hardcoded to before — `Prefill` was `decode=false`
    //! everywhere, `Decode` was `decode=true` everywhere. If either row below changes, every
    //! emitted program changes with it, silently. (Verified once against real packets: the
    //! Qwen3-4B blob is byte-identical pre/post refactor at ctx 4256/16544 and n_cu 170/256.)
    use super::Mode;

    #[test]
    fn legacy_corners_are_unchanged() {
        assert!(!Mode::Prefill.decode_shape() && !Mode::Prefill.gemv());
        assert!(Mode::Decode.decode_shape() && Mode::Decode.gemv());
    }

    #[test]
    fn decode_tiled_is_decode_shape_on_prefill_kernels() {
        // The whole point: decode's shape (one row, KV append, ring mask) with prefill's
        // kernels (tiled GEMM, FlashPrefill). Neither legacy corner can express this.
        assert!(Mode::DecodeTiled.decode_shape());
        assert!(!Mode::DecodeTiled.gemv());
    }
}

#[cfg(test)]
mod pick_tile_tests {
    //! The hwspec-driven picker is a STATIC, shape-agnostic choice — so it is testable
    //! offline, with no GPU. These lock in the tile chosen for every projection of the three
    //! supported architectures at the prefill chunk sizes that matter, proving the picker both
    //! fills the CUs on the underutilized shapes AND does not regress the ones that already
    //! saturate. `n_cu = 256` (MI350X).
    use super::{gemm_lds_bytes, hwspec, pick_tile, DevOp, GFX950_TILES};
    use costmodel::cost::{dma_cycles, macs_cycles};
    use costmodel::MmaDtype;

    const N_CU: u32 = 256;
    fn pt(m: u32, n: u32, k: u32) -> DevOp {
        pick_tile(m, n, k, N_CU)
    }

    /// The picker exactly as it was before selection moved behind the capability
    /// registry: one loop over a constant table, first-match-wins on ties.
    ///
    /// Kept as the differential reference. The assertions below pin the shapes
    /// that were reasoned about by hand; this pins everything else, which is
    /// what actually rules out a silent regression on some shape nobody listed.
    fn pick_tile_legacy(m: u32, n: u32, k: u32, n_cu: u32) -> DevOp {
        let spec = hwspec::registry::lookup("MI350X").expect("gfx950 spec in registry");
        let lds_budget = spec.sm.shared_mem.0;
        let (m, n, k) = (m as u64, n as u64, k as u64);
        let n_cu = (n_cu as u64).max(1);

        let mut best = (DevOp::Gemm, u64::MAX);
        for (op, bm, bn, bk) in GFX950_TILES {
            if gemm_lds_bytes(bm, bn, bk) > lds_budget {
                continue;
            }
            let tiles = m.div_ceil(bm) * n.div_ceil(bn);
            let rounds = tiles.div_ceil(n_cu);
            let k_iters = k.div_ceil(bk);
            let compute = k_iters * macs_cycles(spec, bm * bn * bk, MmaDtype::Bf16);
            let dma = dma_cycles(spec, (bm * k + k * bn) * 2, false);
            let cost = rounds.saturating_mul(compute.max(dma));
            if cost < best.1 {
                best = (op, cost);
            }
        }
        best.0
    }

    /// Routing selection through the registry must not change a single answer on
    /// the hardware the old picker was written for. Swept rather than sampled:
    /// tie-breaking was the real risk, since the old loop preferred the larger
    /// tile by table order while opcode order would put `GemmSmall` (14) ahead of
    /// `GemmMed` (15).
    #[test]
    fn registry_selection_matches_the_legacy_picker_everywhere() {
        let ms = [1u32, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
        let ns = [128u32, 512, 1024, 2048, 2560, 4096, 5376, 8192, 9728, 14336, 16384, 21504];
        let ks = [128u32, 512, 2560, 4096, 5376, 8192, 14336, 21504];
        let cus = [1u32, 64, 128, 256, 304];

        let mut checked = 0usize;
        for &m in &ms {
            for &n in &ns {
                for &k in &ks {
                    for &n_cu in &cus {
                        let want = pick_tile_legacy(m, n, k, n_cu);
                        let got = pick_tile(m, n, k, n_cu);
                        assert_eq!(
                            got, want,
                            "diverged at m={m} n={n} k={k} n_cu={n_cu}: \
                             registry chose {got:?}, legacy chose {want:?}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, ms.len() * ns.len() * ks.len() * cus.len());
    }

    #[test]
    fn llama31_8b_prefill_4k() {
        // hidden 4096, inter 14336, heads 32, kv_heads 8, hd 128.
        // q/o saturate 256 CUs at 256x256 (16x16 = 256 tiles) — keep the big tile.
        assert_eq!(pt(4096, 4096, 4096), DevOp::Gemm, "q_proj");
        assert_eq!(pt(4096, 4096, 4096), DevOp::Gemm, "o_proj");
        // k/v (N=1024) are only 16x4 = 64 tiles at 256x256 — a QUARTER of the machine. The
        // picker drops to 128x128 (16x8 = 256 tiles) to fill all 256 CUs. This is the fix the
        // old heuristic missed (it pinned k/v to 256x256, blind to CU fill).
        assert_eq!(pt(4096, 1024, 4096), DevOp::GemmMed, "k_proj / v_proj");
        // gate/up (fused GemmGlu path keys off Gemm) and down saturate — keep 256x256.
        assert_eq!(pt(4096, 14336, 4096), DevOp::Gemm, "gate/up (fused)");
        assert_eq!(pt(4096, 4096, 14336), DevOp::Gemm, "down_proj");
    }

    #[test]
    fn llama31_8b_prefill_8k_kv_already_half_full() {
        // At M=8192 k/v already make 32x4 = 128 tiles (half fill) at 256x256; splitting to
        // 128x128 would need 2 rounds for equal cost, so the higher-intensity 256x256 stays.
        assert_eq!(pt(8192, 1024, 4096), DevOp::Gemm, "k/v at 8k");
    }

    #[test]
    fn qwen3_4b_prefill_4k() {
        // hidden 2560, inter 9728, heads 32, kv_heads 8, hd 128.
        assert_eq!(pt(4096, 4096, 2560), DevOp::Gemm, "q_proj");
        assert_eq!(
            pt(4096, 1024, 2560),
            DevOp::GemmMed,
            "k_proj / v_proj (fill)"
        );
        assert_eq!(pt(4096, 9728, 2560), DevOp::Gemm, "gate/up");
        assert_eq!(pt(4096, 2560, 9728), DevOp::Gemm, "down_proj");
    }

    #[test]
    fn gemma31b_no_regression() {
        // hidden 5376, inter 21504. Gemma's kv projections are WIDE (sliding N=4096, global
        // N=2048), so they already saturate — the picker must keep 256x256 everywhere it did
        // before. No Gemma projection is small enough to reselect.
        assert_eq!(pt(4096, 8192, 5376), DevOp::Gemm, "q sliding");
        assert_eq!(pt(4096, 16384, 5376), DevOp::Gemm, "q global");
        assert_eq!(pt(4096, 4096, 5376), DevOp::Gemm, "kv sliding (N=4096)");
        assert_eq!(pt(4096, 2048, 5376), DevOp::Gemm, "kv global (N=2048)");
        assert_eq!(pt(4096, 5376, 8192), DevOp::Gemm, "o sliding");
        assert_eq!(pt(4096, 21504, 5376), DevOp::Gemm, "gate/up");
        assert_eq!(pt(4096, 5376, 21504), DevOp::Gemm, "down");
    }

    #[test]
    fn short_prompt_buckets_use_narrow_tiles() {
        // A 128-row chunk cannot fill 256 CUs with a 256x256 tile (q_proj = 1x16 = 16 tiles),
        // so the picker drops to the narrow-M kernels — matching the measured T=128 optima in
        // op_gemm.h (64x128 fastest for the tall projections at small M).
        assert_eq!(pt(128, 8192, 5376), DevOp::GemmSmall, "T=128 q sliding");
        assert_ne!(
            pt(128, 4096, 4096),
            DevOp::Gemm,
            "T=128 must not pick the big tile"
        );
    }
}

#[cfg(test)]
mod chunk_default_tests {
    use super::{default_chunk, kv_ring, kv_ring_rows, MAX_CHUNK_MAX};

    /// An all-global model (`window == 0`) must keep the full chunk. `kv_ring`
    /// returns `(ctx, MASK_NONE)` for full layers, so a smaller chunk buys no
    /// KV there and only costs prefill launches — the Gemma-shaped default
    /// must not leak onto Llama-shaped networks.
    #[test]
    fn all_global_models_keep_the_full_chunk() {
        assert_eq!(default_chunk(0), MAX_CHUNK_MAX);
        // and the chunk genuinely does not size a full layer's cache
        let (rows_big, mask) = kv_ring(true, 8192, 0, MAX_CHUNK_MAX);
        let (rows_small, _) = kv_ring(true, 8192, 0, 1024);
        assert_eq!(rows_big, rows_small, "chunk must not change a full layer");
        assert_eq!(mask, super::KV_MASK_NONE);
    }

    /// A windowed model derives the chunk from its own window, so the ring
    /// lands at 2 x next_pow2(window) — the floor the invariant allows.
    #[test]
    fn windowed_models_derive_chunk_from_window() {
        assert_eq!(default_chunk(1024), 1024); // Gemma-4
        assert_eq!(default_chunk(4096), 4096);
        assert_eq!(default_chunk(768), 1024); // rounded up to a power of two
        // never below the bucket floor, never above the ladder top
        assert_eq!(default_chunk(1), super::MAX_CHUNK_MIN);
        assert_eq!(default_chunk(1 << 20), MAX_CHUNK_MAX);
    }

    /// The wrap invariant must hold for every window the default picks —
    /// violating it aliases a chunk's rows onto its own history, which is a
    /// silent wrong answer rather than a crash.
    #[test]
    fn derived_chunk_satisfies_the_wrap_invariant() {
        for w in [128u32, 512, 768, 1024, 2048, 4096, 8192, 16384] {
            let c = default_chunk(w);
            let ring = kv_ring_rows(w, c);
            assert!(
                ring >= w + c - 1,
                "window {w} chunk {c} ring {ring} violates ring >= window + chunk - 1"
            );
        }
    }
}
