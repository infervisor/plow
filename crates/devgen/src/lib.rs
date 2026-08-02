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
pub mod k3;
pub mod kda;
use config::*;
mod ladder;
mod mla;
use mla::{glm_emit_block, glm_main, kimi_emit_block, nemotron_emit_block, MlaArch};
pub mod manifest;
pub mod tune_demand;

// # THE `PLOW_*` FLAG AUDIT (target dependence)
//
// Four flags were found that are correct on sm_120 and silently WRONG on gfx950. Each was defined
// when there was one backend; each fails silently rather than loudly on AMD, because AMD's
// dispatch `default:` writes NOTHING while sm_120's is `__trap()`. This is the full sweep, so the
// next person does not repeat it. Flags recorded as neutral are recorded ON PURPOSE.
//
// ## Target-dependent, now gated
//
// | flag | on sm_120 | on gfx950 before the gate | gate |
// |------|-----------|---------------------------|------|
// | `PLOW_FP8` (alone) | emits w8a16, which has a cubin | w8a16 into a w8a8-only arm → null `a_scale` fault | REFUSED (`check_fp8_a_scale_bound`) — ignoring would emit a WRONG packet |
// | `PLOW_L2_PLACE` | L2 domains in `seg` | overwrites the wave-class tag on a MULTI-SEGMENT program → whole prefill on the flash object, zero logits | skipped per PROGRAM by `Builder::finish` when it has >1 wave class; single-class programs (decode) are placed |
// | `PLOW_UNISEG` | collapses segments, spurious there | destroyed the wave-class split → zero logits, 8.7 ms "prefill" | ignored + warned (`Builder::deny_uniseg`) |
// | `PLOW_PF_LADDER=wave` | rungs from the 128x128 sm_120 tile | AMD tiles differently → mis-tuned rungs (degrades, does not corrupt) | ignored quietly |
//
// REFUSE vs IGNORE is decided by one question: what does the caller get if the flag is dropped? A
// correct packet ⇒ ignore, because that is what they wanted. A wrong packet ⇒ refuse, because
// silently substituting a different computation is the failure mode being eliminated.
//
// ## Not a vendor gate — a PAIRED value, fixed by rendering both spellings
//
// `PLOW_FA_GF_FULL` must equal the kernel's constant of the same name. Setting it moved only the
// packet half because `backend_nvcc` rendered the sm_120 spelling and nothing rendered the AMD
// one; `backend_gfx950` now does. Gating would have been the wrong fix — the flag is meaningful on
// both targets, it was the RENDERING that was one-sided.
//
// ## Covered by construction, not by a per-flag gate
//
// `PLOW_FUSE_ARGMAX` emits `GEMV_ARGMAX`, which has no AMD arm — a decode would have argmaxed an
// untouched buffer and returned token 0 forever. So would the `PLOW_GEMMA_MOE_ROUTER_*` /
// `PLOW_GEMMA_MOE_TAIL_FUSE` family, whose opcodes (61-77) are absent wholesale. Gating each is
// whack-a-mole; `check_gfx950_opcode_coverage` checks the emitted STREAM against what the target
// dispatches, so any flag that reaches for a missing arm is caught whatever its name.
//
// ## Verified vendor-NEUTRAL (do not re-audit)
//
// * opcode-changing, but every op has an AMD arm: `PLOW_NO_FUSE_QKV`, `PLOW_FP8_HEAD`,
//   `PLOW_FP8_KV`, `PLOW_FP8_KV_FULL`, `PLOW_MXFP4`, `PLOW_MLA_PREFILL`, `PLOW_MOE_PREFILL`,
//   `PLOW_GLM_DSA`, `PLOW_GLM_FUSE_A/_G/_B1`, `GLM_ROUTER_OLD` (GLM arm), `GLM_GROUP`;
// * tuning constants carried in an existing instruction field, read identically by both
//   interpreters: `PLOW_NS_MUL`, `PLOW_NS_ABS`, `PLOW_NS_FULL_ABS`, `PLOW_GLM_GF`, `PLOW_GLM_NS`,
//   `PLOW_MAX_CHUNK`, `PLOW_DECODE_BATCH`. (`PLOW_NS_FULL_ABS`'s comment cites an sm_120 part —
//   that records where it was MEASURED, not a target dependence.)
// * structure, but geometry-driven rather than ISA-driven: `GLM_EP`, `GLM_MOE_CORESIDENT`,
//   `PLOW_XR_CUS`, `PLOW_SEG_CLASS_SLICE`, `PLOW_FINE_FORCE`, `PLOW_NO_XREDUCE` (which is
//   numerically wrong on BOTH targets and says so);
// * compiler-side only, never in the packet: `PLOW_ROOT`, `PLOW_BLOCK`, `PLOW_SKIP_COVERAGE`,
//   `GLM_FULL`, `GLM_LAYER`, `GLM_NLAYERS`.
//
// ## The inverse case, worth knowing
//
// `PLOW_DECODE_TILED` is AMD-only: it emits prefill opcodes into the decode bucket, and the sm_120
// interpreter traps on every one of them. It is documented as such and needs no gate — a trap is
// the loud failure. `PLOW_PF_GEMV_HEAD` is the other correctly-handled one: default-on where the
// arm exists, opt-in where it does not.

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

    let Some(tile) = kernel.tile else {
        return u64::MAX;
    };
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

    // Larger tile first on a tie: rank by descending BM*BN. Continuous rather
    // than the three hand-written brackets it replaces — those bracketed the
    // three tiles that existed, so the two rungs added by the tile-inventory
    // campaign (192x256 = 49152 and 128x256 = 32768) both fell in the SAME
    // bracket as 256x256 and would have tied with it, then resolved by opcode
    // NUMBER, which is exactly what the rank term exists to prevent. 65536 is
    // the largest tile's BM*BN, so rank stays in 0..=7 and inside the x8
    // headroom the cost is shifted by.
    let rank = 7 - (7 * (bm * bn).min(65536) / 65536);
    cost.saturating_mul(8).saturating_add(rank)
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
/// # Precision is an input, not a label
///
/// `quant` selects which rungs are even legal: `KernelSpec::accepts` compares it, so a
/// `W8A8` signature matches the fp8 instantiations and a `Mxfp4` one the fp4 instantiations.
/// Before this it was hard-wired to `QuantScheme::None`, the inventory registered bf16 tiles
/// only, and the fp8 emit site recovered the encoding *after the fact* by mapping the bf16
/// opcode `pick_tile` returned onto its fp8 twin — a three-arm `match` that had to be kept in
/// sync by hand and that silently mapped anything it did not recognise to the 256x256 tile.
/// That mapping is gone; the selector returns the opcode to emit.
///
/// It matters numerically and not only structurally: fp8 halves the operand bytes without
/// halving the MFMA work, so `tile_cost`'s `max(compute, dma)` tips toward compute and the
/// balance point between CU fill and arithmetic intensity moves. mxfp4 is w4a16 — quarter-byte
/// weights, bf16 activations and a bf16 MFMA — so it moves further again, in the same
/// direction, and it is the encoding whose single hard-coded tile produced the worst measured
/// number in the campaign (Kimi `kv_a_proj`, ≈0.4% of peak).
///
/// # Measured, when a measurement exists
///
/// `NoMeasurements` used to be passed here verbatim, so the analytical model decided every
/// tile on every shape and the `tunedb` the tuning architecture specifies was consulted by the
/// generic `plowc` path only (`plowc/src/tuned.rs`) and never by the AMD emitters. It now
/// reads the same store, for the gfx950 cell, keyed by the same `(m,n,k,quant)` op case a
/// campaign files under. No records, wrong hardware, or digests that have moved since the
/// interpreter was recompiled all degrade to the analytical model — that is what the store's
/// staleness rules are for, and a stale record is more dangerous than none.
fn pick_tile(m: u32, n: u32, k: u32, n_cu: u32, quant: kernelcaps::QuantScheme) -> DevOp {
    select_gemm_over(gfx950_gemm_inventory(), m, n, k, n_cu, quant)
}

/// [`pick_tile`], public, so a caller outside the emitters can ask what this build would emit.
///
/// `plowc tune` reports a selection it re-derives with `NoMeasurements`, which is how it came
/// to print a different tile than the build emits. Anything that wants to *report* the choice
/// should ask the same function that *makes* it.
pub fn gfx950_prefill_tile(
    m: u32,
    n: u32,
    k: u32,
    n_cu: u32,
    quant: kernelcaps::QuantScheme,
) -> DevOp {
    pick_tile(m, n, k, n_cu, quant)
}

/// Whether a qualified measurement was found for this shape, and how many rungs it covers.
///
/// Reported rather than inferred: "the analytical model chose this" and "a measurement chose
/// this" are different claims with different calibration tiers, and a build that silently
/// degrades from the second to the first — because the interpreter was recompiled and every
/// record went stale — looks identical from the outside otherwise.
pub fn gfx950_measured_rungs(m: i64, n: i64, k: i64, quant: kernelcaps::QuantScheme) -> usize {
    gfx950_gemm_measurements()
        .by_case
        .get(&tunedb::gemm_op_case(m, n, k, quant))
        .map(|t| t.len())
        .unwrap_or(0)
}

/// [`pick_tile`] against a caller-supplied inventory.
///
/// Factored out for the two callers that need a RESTRICTED candidate set: [`glu_fusion_wins`],
/// which may only consider tiles the GLU epilogue is instantiated at, and the differential test
/// that pins the three original rungs still ranking among themselves exactly as the
/// pre-registry picker did. Both need the real ranking, not a copy of it — a second copy is how
/// the two `pick_tile` implementations in `plowc/src/bin/` drifted apart.
fn select_gemm_over(
    inv: &kernelcaps::Inventory,
    m: u32,
    n: u32,
    k: u32,
    n_cu: u32,
    quant: kernelcaps::QuantScheme,
) -> DevOp {
    let spec = hwspec::registry::lookup("MI350X").expect("gfx950 spec in registry");
    let hw = kernelcaps::HardwareFingerprint::from_spec(spec).expect("gfx950 fingerprint");
    let mut op =
        kernelcaps::OpSignature::gemm(kernelcaps::Phase::Prefill, m as i64, n as i64, k as i64);
    op.quant = quant;

    // The registry decides what is *executable*; the closure decides which of
    // those is fastest. Fusing both halves into one loop over a constant table
    // is what let this function name a tile the target does not implement
    // whenever it ran for a build that was not gfx950.
    let realization = kernelcaps::select_kernel(
        inv,
        &op,
        &hw,
        kernelcaps::ProfileId::PrefillDense,
        gfx950_gemm_measurements().for_shape(m as i64, n as i64, k as i64, quant),
        |kernel| tile_cost(spec, kernel, m as i64, n as i64, k as i64, n_cu),
    )
    .unwrap_or_else(|e| {
        panic!(
            "{e}\nThe gfx950 prefill inventory carries no {quant:?} GEMM rung. Either the \
             probe could not preprocess the object that holds them (interp_prefill_fp8 / \
             interp_prefill_mxfp4 — `kernelcaps::targets::GFX950_QUANT_OBJECTS`), or the \
             emitter is asking for an encoding this build does not dispatch. Emitting the \
             bf16 tile instead would be silently wrong: it would read fp4 bytes as bf16."
        )
    });

    realization.kernel.0
}

/// Whether the prefill gate|up GEMM should take the FUSED [`DevOp::GemmGlu`] path.
///
/// # Why this is not just `pick_tile(..) == DevOp::Gemm`
///
/// It used to be, and that spelling stopped being correct the moment the inventory grew rungs
/// the GLU epilogue is not instantiated at. `d_gemm_glu` exists at 256x256 only, so once
/// `pick_tile` can answer `GemmWide` or `GemmC5` the old test goes false and gate|up silently
/// falls back to the UNFUSED triple — three packets instead of one, and `gt`/`ut`
/// (M x inter x 2 B each) materialised to HBM and read back. On Gemma-31B at M=2048 that is
/// ~176 MB of traffic bought for a **+6%** tile (measured standalone: c5 1033 TF/s vs c0 974 on
/// `2048x21504x5376`). A clear net loss, and it would have looked like an improvement in every
/// per-tile number.
///
/// So the question this asks is the one the old spelling MEANT: among the tiles that can carry
/// the epilogue, is the 256x256 one the winner — i.e. does the shape fill the machine at
/// 256x256. Restricting the candidate set is what makes it mean that, and it keeps this
/// decision byte-identical to before the new rungs existed.
///
/// **Named residual, not an oversight:** instantiating `d_gemm_glu` at 128x256 and 192x256
/// would let gate|up have both. Both have BN=256, so `SN == 2` holds and the wave->column remap
/// the epilogue needs is legal at either. It is left out here to keep this change surgical; the
/// prize is the +6% above, on the largest GEMM in the model.
fn glu_fusion_wins(m: u32, n: u32, k: u32, n_cu: u32) -> bool {
    select_gemm_over(
        glu_era_inventory(),
        m,
        n,
        k,
        n_cu,
        kernelcaps::QuantScheme::None,
    ) == DevOp::Gemm
}

/// [`glu_fusion_wins`] for the MXFP4 (w4a16) family — asked at the shape the fused arm ACTUALLY
/// runs, which is not the shape it is emitted for.
///
/// [`DevOp::GemmGluMxfp4`] exists at 256x256 only, for the reason its bf16 twin does: the
/// epilogue's wave->column remap needs `SN == 2`, i.e. `BN = 256`. So the candidate set is the
/// three fp4 rungs over which "the winner is 256x256" means "this shape fills the machine at
/// 256x256" — the restriction [`glu_era_inventory`] documents, one encoding over.
///
/// # The width is `2n`, and that is the whole correction
///
/// Under `GLU` a tile emits `BN/2` output columns, not `BN` (`op_gemm.h`: `NB = GLU ? BN/2 : BN`,
/// and `tn = ceil(N / NB)`). So the fused arm's tile grid at output width `n` is
/// `ceil(m/256) x ceil(n/128)` — EXACTLY a plain 256x256 GEMM's grid at width `2n`, which is also
/// exactly the two unfused GEMMs' grids added together. Asking the selector about `(m, n, k)`
/// therefore asks about a machine fill the fused arm never has: it under-counts the tiles by 2x and
/// refuses the fusion on shapes that do fill the machine.
///
/// That is not a theoretical correction. At Kimi-K3's TP8 shared-expert prefill (`n = 768`,
/// `k = 7168`, `t = 8192`) the `(m, n, k)` spelling refuses, and the fused arm measures **-35.2%**
/// against the best unfused rung; at `n = 1536, t = 4096` it refuses and the arm measures -34.7%.
/// See `crates/devgen/tests/mxfp4_glu_fusion.rs` for the table.
///
/// **Named residual:** [`glu_fusion_wins`] has the same `NB = BN/2` property and asks at `n`. Left
/// alone deliberately — correcting it would move bf16 emission for every Gemma/Llama/Qwen shape,
/// which is a separate change with its own measurements to take.
pub fn glu_fusion_wins_mxfp4(m: u32, n: u32, k: u32, n_cu: u32) -> bool {
    select_gemm_over(
        glu_era_inventory_mxfp4(),
        m,
        2 * n,
        k,
        n_cu,
        kernelcaps::QuantScheme::Mxfp4,
    ) == DevOp::GemmMxfp4
}

/// The fp4 rungs [`glu_fusion_wins_mxfp4`] ranks — the three that mirror [`glu_era_inventory`]'s.
fn glu_era_inventory_mxfp4() -> &'static kernelcaps::Inventory {
    use std::sync::OnceLock;
    const RUNGS: [DevOp; 3] = [DevOp::GemmMxfp4, DevOp::GemmMedMxfp4, DevOp::GemmSmallMxfp4];
    static INV: OnceLock<kernelcaps::Inventory> = OnceLock::new();
    INV.get_or_init(|| {
        let src = gfx950_gemm_inventory();
        kernelcaps::Inventory::probed(
            src.build().clone(),
            src.iter().filter(|s| RUNGS.contains(&s.id.0)).cloned(),
        )
    })
}

/// The bf16 rungs that existed when the GLU epilogue was written.
///
/// The only set over which "the winner is 256x256" means "this shape fills the machine at
/// 256x256" — which is the question [`glu_fusion_wins`] is really asking, and the question the
/// pre-campaign `pick_tile(..) == DevOp::Gemm` happened to answer because those were the only
/// tiles there were.
fn glu_era_inventory() -> &'static kernelcaps::Inventory {
    use std::sync::OnceLock;
    const RUNGS: [DevOp; 3] = [DevOp::Gemm, DevOp::GemmMed, DevOp::GemmSmall];
    static INV: OnceLock<kernelcaps::Inventory> = OnceLock::new();
    INV.get_or_init(|| {
        let src = gfx950_gemm_inventory();
        kernelcaps::Inventory::probed(
            src.build().clone(),
            src.iter().filter(|s| RUNGS.contains(&s.id.0)).cloned(),
        )
    })
}

/// Qualified GEMM measurements for the gfx950 cell, loaded once.
///
/// Location comes from `PLOW_TUNEDB` (default `tuning/`, the tree
/// `tuning/README.md` documents and where the sm_120a and sm_90a cells already
/// live). A missing store is not an error: it is the cold-start case, and the
/// analytical model is the declared fallback tier.
struct GemmMeasurements {
    /// `op_case -> (opcode -> median ns)`.
    by_case: std::collections::HashMap<String, std::collections::HashMap<u16, f64>>,
}

/// The measured costs that apply to ONE shape. `select_kernel` asks per kernel,
/// so the shape has to be bound before the trait object is handed over.
struct ShapeCosts<'a>(Option<&'a std::collections::HashMap<u16, f64>>);

impl kernelcaps::MeasuredCosts for ShapeCosts<'_> {
    fn median_ns(&self, kernel: kernelcaps::KernelId) -> Option<f64> {
        self.0?.get(&kernel.raw()).copied()
    }
}

impl GemmMeasurements {
    fn for_shape(
        &self,
        m: i64,
        n: i64,
        k: i64,
        quant: kernelcaps::QuantScheme,
    ) -> &dyn kernelcaps::MeasuredCosts {
        // Leaked rather than returned by value so the borrow outlives the call
        // without threading a lifetime through `pick_tile`; there is one per
        // distinct shape in a compile, and a compile is a process.
        let case = tunedb::gemm_op_case(m, n, k, quant);
        let hit = self.by_case.get(&case);
        // THE ONE PLACE the compiler asks the store about a dense GEMM, and therefore the one
        // place its demand can be observed. `tune_demand` both prints the `PLOW_TUNE_DUMP=1`
        // line (unchanged) and records the lookup as typed data for `plowc tune gemm
        // --shapes auto`.
        //
        // The campaign's shape list in scripts/rebench_tune_gemm.sh was authored BY HAND, and
        // that is how GLM-5.2 came to have exactly two measured shapes (M=128 N=256 K=6144,
        // M=128 N=576 K=6144) while every M>=256 record in the store was a Gemma-31B or Qwen
        // shape with K in {2560,4096,5376,8192,21504} — never GLM's K=6144. So GLM prefill above
        // the smallest bucket selected tiles from the ANALYTICAL MODEL, and `tuned_tile_selection`
        // still passed because SOME qualified record existed. Deriving the list from this call
        // site instead of by hand is what stops that recurring.
        tune_demand::record(m, n, k, quant, hit.is_some());
        Box::leak(Box::new(ShapeCosts(hit)))
    }
}

fn gfx950_gemm_measurements() -> &'static GemmMeasurements {
    use std::sync::OnceLock;
    static M: OnceLock<GemmMeasurements> = OnceLock::new();
    M.get_or_init(|| {
        let mut by_case: std::collections::HashMap<String, std::collections::HashMap<u16, f64>> =
            Default::default();
        // EMPTY means "no store", which is how `plowc --no-tuning` reaches here. Unset means
        // "the default tree" — the two are deliberately different: a compile that never asked
        // about tuning should still get the calibrated answer, and one that explicitly asked
        // for the analytical model must get it.
        let root = match std::env::var("PLOW_TUNEDB") {
            Ok(s) if s.is_empty() => return GemmMeasurements { by_case },
            Ok(s) => s,
            Err(_) => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tuning")
                .to_string_lossy()
                .into_owned(),
        };
        let store = tunedb::TuneStore::new(std::path::PathBuf::from(root));
        // Digests come from the PROBED build, not from a constant: a tile edit
        // in op_gemm.h changes the preprocessed digest, which is what makes the
        // previous campaign's records stale instead of silently authoritative.
        let want = tunedb::Digests {
            implementation: gfx950_gemm_inventory().build().label(),
            interpreter: gfx950_gemm_inventory().build().label(),
            toolchain: gfx950_gemm_inventory().build().toolchain.clone(),
            oracle: tunedb::GEMM_ORACLE.to_string(),
        };
        let Ok(records) = store.load_kernels(tunedb::GFX950_CELL) else {
            return GemmMeasurements { by_case };
        };
        let mut stale = 0usize;
        for r in records {
            if !r.state.is_selectable() {
                continue;
            }
            if !r.digests.stale_against(&want).is_empty() {
                stale += 1;
                continue;
            }
            let e = by_case.entry(r.op_case.clone()).or_default();
            // Best-of, so a re-measured campaign does not depend on file order.
            let cur = e.entry(r.kernel_id).or_insert(f64::INFINITY);
            *cur = cur.min(r.stats.median_ns);
        }
        // TOTAL staleness must be LOUDER than partial staleness, not silent.
        //
        // This was gated on `!by_case.is_empty()`, so the one case that actually matters --
        // EVERY record stale, nothing usable, silent fallback to the analytical model --
        // printed nothing at all. "Fell back to analytical" and "was never measured" produce
        // identical bytes, so silence reads as success. A wholly stale campaign is a
        // re-measure request and has to say so.
        if stale > 0 {
            eprintln!(
                "  tunedb {}: {stale} record(s) skipped as STALE against the probed build {}{}",
                tunedb::GFX950_CELL,
                want.interpreter,
                if by_case.is_empty() {
                    " -- NO usable records remain, so tile selection fell back to the \
                     analytical model. Re-run the campaign or this compile is unmeasured."
                } else {
                    ""
                }
            );
        }
        GemmMeasurements { by_case }
    })
}

/// Hand the GEMV op cases this build has qualified, non-stale measurements for down to
/// [`packet::devbuild`], so its `PLOW_TUNE_DUMP` census can report HIT/MISS.
///
/// # Why the answer is pushed down instead of pulled up
///
/// The GEMV census hook lives in `packet::devbuild::Builder::emit_dep` — the one function
/// every GEMV emit site in this crate, `mla.rs` and `kda.rs` funnels through — because the
/// alternative is instrumenting thirty-odd call sites by hand, which is the same mistake as
/// the hand-authored GEMM shape list one level down. `packet` has no dependencies by design
/// (it is shared with the C/CUDA runtime), so it cannot read the store; this function is the
/// seam.
///
/// The staleness rule is the GEMM one, for the GEMM reason: records are keyed by the
/// PREPROCESSED interpreter digest, and a stale record is more dangerous than none because
/// "fell back to the analytical model" and "was never measured" otherwise produce identical
/// output. Here it is only a census label — a MISS costs nothing but a MISS mislabelled HIT
/// would make an unmeasured campaign look complete, which is the exact failure being fixed.
///
/// Call once per emit. Idempotent: `packet` stores it in a `OnceLock`.
pub fn install_gfx950_gemv_cases() {
    let mut cases: std::collections::HashSet<String> = Default::default();
    let root = match std::env::var("PLOW_TUNEDB") {
        Ok(s) if s.is_empty() => {
            packet::devbuild::set_tuned_gemv_cases(cases);
            return;
        }
        Ok(s) => s,
        Err(_) => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tuning")
            .to_string_lossy()
            .into_owned(),
    };
    let want = tunedb::Digests {
        implementation: gfx950_gemm_inventory().build().label(),
        interpreter: gfx950_gemm_inventory().build().label(),
        toolchain: gfx950_gemm_inventory().build().toolchain.clone(),
        oracle: tunedb::GEMV_ORACLE.to_string(),
    };
    if let Ok(records) =
        tunedb::TuneStore::new(std::path::PathBuf::from(root)).load_kernels(tunedb::GFX950_CELL)
    {
        for r in records {
            // Every GEMV family, not just the plain one: `DevOp::gemv_case` puts the arm in
            // the key (`gemv`/`gemvglu`/`gemvqkv`/`gemvblk`/`gemvargmax`) so the store cannot
            // rank a fused op against an unfused one. `profile` is the field that separates
            // this cell from the prefill-GEMM one inside the same file.
            if r.profile == "decode_gemv"
                && r.state.is_selectable()
                && r.digests.stale_against(&want).is_empty()
            {
                cases.insert(r.op_case);
            }
        }
    }
    packet::devbuild::set_tuned_gemv_cases(cases);
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

/// Every gfx950 prefill GEMM rung, in each of the three weight encodings, exactly as a probe
/// of `runtime/amd/op_gemm.h` would report it: `(bf16, fp8, mxfp4, BM, BN, BK)`.
///
/// ONE table, because there were two identical copies of the three-tile version — the
/// analytical fallback and the test fixture — and a rung added to one and not the other gives
/// a compiler that selects differently under test than in production. That is the same class
/// of drift `kernelcaps` exists to prevent; it just happened to be inside this file.
///
/// These are compile-time constants in the interpreter object and change only with an
/// intentional edit to `op_gemm.h`.
const GFX950_RUNGS: [(DevOp, DevOp, DevOp, i64, i64, i64); 5] = [
    (DevOp::Gemm, DevOp::GemmFp8, DevOp::GemmMxfp4, 256, 256, 64),
    (
        DevOp::GemmMed,
        DevOp::GemmMedFp8,
        DevOp::GemmMedMxfp4,
        128,
        128,
        64,
    ),
    (
        DevOp::GemmSmall,
        DevOp::GemmSmallFp8,
        DevOp::GemmSmallMxfp4,
        64,
        128,
        64,
    ),
    (
        DevOp::GemmWide,
        DevOp::GemmWideFp8,
        DevOp::GemmWideMxfp4,
        128,
        256,
        64,
    ),
    (
        DevOp::GemmC5,
        DevOp::GemmC5Fp8,
        DevOp::GemmC5Mxfp4,
        192,
        256,
        64,
    ),
];

/// Every opcode [`pick_tile`] can answer with, in any encoding — the set a test asks "is this a
/// tiled prefill GEMM?" about.
///
/// Derived from [`GFX950_RUNGS`] rather than hand-listed, because a hand-listed copy is exactly
/// what went stale when the two 128x256/192x256 rungs landed: a gate naming only
/// `Gemm`/`GemmMed`/`GemmSmall` silently stopped matching the shapes the selector had started
/// choosing for. Add a rung to the table and every caller here follows.
#[allow(dead_code)] // read by the emitter TESTS; production code names its rung directly
pub(crate) fn gemm_family_ops() -> Vec<u16> {
    GFX950_RUNGS
        .iter()
        .flat_map(|&(b, f, m, ..)| [b as u16, f as u16, m as u16])
        .collect()
}

/// [`GFX950_RUNGS`] as `KernelSpec`s, tagged with the encoding each serves.
///
/// The `mma_dtype` for mxfp4 is bf16 and not fp4: the fp4 prefill GEMM is w4a16 and dequantizes
/// in the B-fetch, so the matrix instruction it issues is the ordinary bf16 MFMA. Mirrors
/// `kernelcaps::targets::GFX950_QUANT_OBJECTS`, which is what the real probe uses.
fn gfx950_rung_specs(build_label: &str) -> Vec<kernelcaps::KernelSpec> {
    use hwspec::{IsaLevel, MmaDtype};
    use kernelcaps::{KernelSpec, QuantScheme};
    let mut out = Vec::with_capacity(GFX950_RUNGS.len() * 3);
    for (bf16, fp8, mx, bm, bn, bk) in GFX950_RUNGS {
        for (op, quant, mma) in [
            (bf16, QuantScheme::None, MmaDtype::Bf16),
            (fp8, QuantScheme::W8A8, MmaDtype::Fp8),
            (mx, QuantScheme::Mxfp4, MmaDtype::Bf16),
        ] {
            let body = format!("gfx950:{}@{build_label}", op.c_name());
            out.push(
                KernelSpec::gemm_tile(op, IsaLevel::Gfx950, bm, bn, bk, &body)
                    .with_quant(quant, mma),
            );
        }
    }
    out
}

/// Analytical fallback inventory for gfx950, used when the probe cannot run (no hipcc).
fn gfx950_analytical_inventory() -> kernelcaps::Inventory {
    let build = kernelcaps::BuildId::new(
        hwspec::IsaLevel::Gfx950,
        ["PLOW_BUCKET_DECODE=0".to_string()],
        "analytical-fallback",
        "analytical-fallback",
    );
    let specs = gfx950_rung_specs(&build.label());
    kernelcaps::Inventory::probed(build, specs)
}

/// Test fixture standing in for a probe.
///
/// This is a test *input*, not shipped data: it never reaches a compiled
/// artifact, and production has no path to it. It exists so the tile-selection
/// regression tests can run on a machine without ROCm, which is the only reason
/// the real probe is unavailable here.
#[cfg(test)]
fn gfx950_gemm_inventory() -> &'static kernelcaps::Inventory {
    use std::sync::OnceLock;
    static INV: OnceLock<kernelcaps::Inventory> = OnceLock::new();
    INV.get_or_init(|| {
        let build = kernelcaps::BuildId::new(
            hwspec::IsaLevel::Gfx950,
            ["PLOW_BUCKET_PREFILL=1".to_string()],
            "test-fixture",
            "test-fixture",
        );
        let specs = gfx950_rung_specs(&build.label());
        kernelcaps::Inventory::probed(build, specs)
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
    // ===== THE `fp8/` KEY CONTRACT (one spelling, no transformation) =====================
    //
    // A packet tensor named `fp8/<name>` is looked up in the fp8 twin checkpoint under the key
    // `fp8/<name>` — VERBATIM, prefix included. The prefix is part of the key, not a routing
    // marker to be stripped on the way in.
    //
    // It was ambiguous, and ambiguity here is not cosmetic: the emitter declared `fp8/<name>`,
    // `quantize_fp8.py` wrote `fp8/<name>`, and a loader stripped the prefix and looked up
    // `<name>` — so a freshly generated fp8 checkpoint could not load at all. A contract with two
    // accepted spellings is a bug waiting to resurface, so it is stated here once and pinned by
    // `fp8_key_tests` below.
    //
    // Verbatim rather than stripped, deliberately. Stripping is a TRANSFORMATION, and a
    // transformation applied in one place and not another is exactly the failure above; with the
    // key equal to the declared name there is nothing to apply. It also makes a twin file
    // self-describing — every key says `fp8/`, so it cannot be mistaken for a bf16 checkpoint, and
    // a bf16 name can never accidentally resolve against fp8 bytes.
    //
    // The ONE legitimate strip is in `checkpoint.rs`'s coverage gate, which maps a declared
    // `fp8/<name>` back to `<name>` to answer a different question — "is the bf16 weight <name>
    // covered by something?" — against the ORIGINAL checkpoint. That is a coverage mapping, not a
    // key lookup, and it is commented as such there.
    //
    // The scale twin is `fp8/<name>_scale`: f32, one per OUTPUT CHANNEL (row of the [out,in]
    // weight), and the dequant is `w8 * scale` — matching `quantize_fp8.py` (`scale = amax/448`,
    // `w8 = round_e4m3(w/scale)`) and the device epilogue (`acc * a_scale[m] * w_scale[n]`).
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
    assert_eq!(
        c.hidden % 8,
        0,
        "hidden {} must be a multiple of 8 (GEMV 8-wide load)",
        c.hidden
    );
    assert_eq!(
        c.inter % 8,
        0,
        "intermediate {} must be a multiple of 8 (GEMV 8-wide load)",
        c.inter
    );
    assert_eq!(
        c.hd_slide % 8,
        0,
        "head_dim {} must be a multiple of 8 (GEMV 8-wide load)",
        c.hd_slide
    );
    assert_eq!(
        c.hd_full % 8,
        0,
        "global_head_dim {} must be a multiple of 8 (GEMV 8-wide load)",
        c.hd_full
    );
    let qd_max = (c.heads / tp) * c.hd_slide.max(c.hd_full);
    // kv activation shards use the per-rank LOCAL kv-head count (shared-kv-head replication clamps
    // it to 1 when tp>kvh, so kvh/tp would under-size to 0 at tp=8 on full layers — §3a/§13.2).
    let kd_max =
        (kvh_local(c.kvh_slide, tp, 0) * c.hd_slide).max(kvh_local(c.kvh_full, tp, 0) * c.hd_full);
    let hd_max = c.hd_slide.max(c.hd_full);
    let inter_sh = c.inter / tp;
    // lm_head is REPLICATED under TP here, not vocab-sharded. ONE reason is left, and it is
    // specific to THIS emitter: Gemma TIES lm_head to embed_tokens, and the emitted lm_head Gemv
    // reads `emb` from offset 0 with no per-rank vocab offset, so a vocab shard would make every
    // rank argmax the SAME low-vocab slice (silently wrong). Replicating keeps the full-vocab
    // argmax correct on every rank (they agree), costs no extra memory (emb is already fully
    // resident for the embed lookup), and is one gemv/token.
    //
    // The OTHER reason this comment used to give — "XArgmaxFin (the cross-rank id-fold) is a stub"
    // — is no longer true: `d_xargmax_fin_mega` (runtime/amd/op_collective.h) is implemented and
    // GLM-5.2's untied head takes the column-parallel arm under `GLM_SHARD_HEAD=1` for a measured
    // -0.26 ms/token, bit-identical (perf-data/glm52-decode-emitter-abs.md §1). Sharding a TIED
    // head additionally needs the per-rank vocab offset on the `emb` read.
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
            b.tensor(
                &format!("kv.{l}.k"),
                db * (kvr * kvh_local * hd) as u64 * kv_elt,
            )
        } else {
            TENSOR_NONE
        });
        t.vc.push(if in_block {
            b.tensor(
                &format!("kv.{l}.v"),
                db * (kvr * kvh_local * hd) as u64 * kv_elt,
            )
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

/// KV-SPLIT HEURISTIC denominator: how many query rows the emitter *charges* to one flash work
/// item when sizing `nsplit`. NOT the kernel's tile height — see [`FLASH_Q_TILE_ROWS`], which is
/// half this. Left at 256 deliberately: `nsplit` here and the `Opart`/`mlpart` capacity in
/// `max_splits` are derived from the SAME number, so they cannot disagree with each other, and
/// changing it repartitions every prefill program (a measurement, not a correctness fix).
const Q_TILE_ROWS: u32 = 8 * 32;

/// The q-tile height `d_flash_prefill` ACTUALLY uses: `PLOW_WAVES * FA_BQ`, with `PLOW_WAVES = 4`.
///
/// `FlashPrefill` is wave-class 4 (`Builder::wave_class` in `crates/packet/src/devbuild.rs`), so
/// it executes in the object built with `-DPLOW_WG_WAVES=4` (`scripts/build_gfx950.sh`), and
/// `runtime/amd/op_attention.h:49` says it in as many words — *"query rows per wave; 4 waves =>
/// 128-row q-tile"*. `runtime/amd/op_attention.h:221` then tiles with `PLOW_WAVES * FA_BQ`.
///
/// This — not [`Q_TILE_ROWS`] — is the number [`flash_merge_map`] must use to decide which flash
/// work item wrote a given query row. Charging 256 there attributes every row in `[128, 256)` of a
/// tile to the wrong q-tile, which puts the wrong producer slices in the merge's [`Dep::Fine`]
/// wait set: the merge is then gated on a flash workgroup that did not write its `(o, m, l)`
/// partials and ungated on the one that did. That is a silent wrong token, not a crash.
///
/// It is INERT for every head count shipped today (8/16/32/64) because `heads * nsplit` divides
/// `nblk_f`, so the producer indices alias mod `nblk_f` and the union comes out complete anyway —
/// the map is byte-identical at 256 and 128 for those shapes. It is NOT inert for a head count
/// that does not divide 256: at `heads = 40, t = 256` half the producer edges go missing.
const FLASH_Q_TILE_ROWS: u32 = 4 * 32;

/// LDS the GEMM arena holds, in halves. Mirrors `GM_LDS_HALVES` in `op_gemm.h`:
/// `2*(GM_BM+GM_BN)*(GM_BK+8)` = `2*(256+256)*72`. A GEMV can stage its A-operand on-chip only
/// if `M*K` fits here, which [`DevOp::GemvGlu`] requires (it re-reads x per output column).
const GM_LDS_HALVES: u64 = 2 * (256 + 256) * (64 + 8);

/// `PLOW_GEMV_MAXM` from `runtime/amd/op_gemm.h` — the widest row bucket the GEMV path has a
/// compiled arm for. Mirrored, not read: it is a static assert over there
/// (`PLOW_SASSERT(PLOW_GEMV_MM <= PLOW_GEMV_MAXM, ...)`) and a build-script clamp here.
const GEMV_MAXM: u32 = 16;

/// `RN_REG` and `PLOW_THREADS` from `runtime/amd/amd_common.h`. Their product is the widest row
/// `d_rmsnorm` reduces on its register (`fits`) path, and therefore the widest row the fused-norm
/// GEMV ([`k3::fuse_norm_gemv`], `norm == 2`) may take: the fused arm is bit-exact only because it
/// walks the row with the SAME per-thread element map, which exists only on that path.
/// `GV_NORM_SCRATCH` from `op_gemm.h` — halves the fused-norm GEMV reserves at the TOP of the
/// arena for its cross-wave reduction, kept off the staged row (`plow_smem` is a union, so the
/// interpreter's `part` and `gm` are one buffer).
pub(crate) const GV_NORM_SCRATCH: u64 = 16;

pub(crate) const RN_REG: u32 = 16;
pub(crate) const PLOW_THREADS: u32 = 512;

/// The compiled GEMV row bucket (`PLOW_GEMV_MM`) the decode object will be built at.
///
/// Mirrors `scripts/build_gfx950.sh`: `next_pow2(PLOW_DECODE_BATCH)`, clamped to
/// [`GEMV_MAXM`], with an explicit `PLOW_GEMV_MM` override taking precedence. The override is
/// what makes an object NARROWER than the program it serves expressible at all, and that
/// combination is the whole subject of `plans/knob-contract.md` §6g-WALK.
fn gemv_row_bucket(t: u32) -> u32 {
    if let Some(v) = std::env::var("PLOW_GEMV_MM")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        return v.clamp(1, GEMV_MAXM);
    }
    let mut p = 1u32;
    while p < t.max(1) {
        p *= 2;
    }
    p.min(GEMV_MAXM)
}

/// Rows a fused decode GEMV stages in LDS at once — the quantity the fusion gate must bound.
///
/// # This is the §6g-WALK companion change, and without it the walk buys nothing
///
/// `gemv_qkv_rows` and `gemv_glu_rows` read `x` only through LDS — `op_gemm.h` says so
/// outright ("x is ALWAYS staged in LDS here: plowc emits this op only when M*K fits
/// GM_LDS_HALVES") — so the emitter must not choose the fused opcode unless the staged rows
/// fit. It gated on `t * hidden`, and at `t = 16, hidden = 5376` that is `86016 > 73728`, so
/// exactly 13 of 16 rows fit. That mis-gate is the third silent-corruption bug of the
/// campaign (§6g-BATCH: slots 13/14/15 fluent-but-WRONG), and the fix at the time was to
/// switch the fusion OFF at t=16 rather than to make it fit.
///
/// Switching it off is not free, and §6g-BATCH prices it: the B=16 device ceiling is
/// **142.4 tok/s against B=8's 202.3** — a 30% REGRESSION at twice the batch. Two things
/// cause it at once, `MM=16` spilling (16 scratch ops, 5536 B/lane) and the loss of BOTH
/// `fuse_qkv` and `glu_fused`; the instruction stream differs in COMPOSITION between t=8 and
/// t=16, not just in width.
///
/// `PLOW_GEMV_WALK` (`op_gemm.h`, default 0) moves the staging INSIDE the row loop, so the
/// bound becomes `min(MM, M) * K` — **independent of M**. At `MM = 8`, `8 * 5376 = 43008`
/// fits, and a t=16 program keeps both fusions. That is what this function expresses, and it
/// is the falsifiable half of the walk's case: at MM=8 serving t=16, B=16 should recover from
/// 142.4 toward 202.3. If it does not, the LDS/fusion explanation is wrong.
///
/// **Byte-identical when the walk is off, and byte-identical when it is on with no
/// `PLOW_GEMV_MM` override**, because then `gemv_row_bucket(t) >= t` and the `min` is `t`.
/// The gate only moves for the build that was built to move it.
fn gemv_staged_rows(t: u32) -> u32 {
    if std::env::var("PLOW_GEMV_WALK").ok().as_deref() == Some("1") {
        t.min(gemv_row_bucket(t))
    } else {
        t
    }
}

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
        window
            .next_power_of_two()
            .clamp(MAX_CHUNK_MIN, MAX_CHUNK_MAX)
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
        assert!(
            r.is_power_of_two(),
            "kv_ring size {r} (ctx {ctx}) must be a power of two"
        );
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

// `moe_down_fine_map` lived here: a GLU→Down fine dependency map that would have let a Down
// block wait only on the GLU blocks producing the slots it reads. It had no caller and had not
// had one for some time — the emitters use `Dep::Coarse` throughout, on the argument recorded
// in `devbuild.rs` (via `lean-plow/Plow/CounterGranularity.lean`'s `collapse`): where the work
// is UNIFORM across each stage's slices, the fine schedule's makespan is provably identical to
// the coarse one. Every MoE expert slice is identical, so the map bought nothing it could not
// already prove. Removed rather than left compiling as documentation of an unused idea.

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
/// `rows_per_item` is how many query rows one flash work item covers: [`FLASH_Q_TILE_ROWS`] in
/// prefill (flash tiles the q axis) and 1 in decode (there is one query row). It MUST be the
/// kernel's tile height, not the `nsplit` heuristic's [`Q_TILE_ROWS`] — see the former's docs for
/// what a mismatch costs.
///
/// D-SPLIT. `d_flash_merge` decomposes into `(b, h, d-chunk)` with
/// `dsplit = ceil(nblk_m / n_bh)`, item `w = (b*n_head + h)*dsplit + dp`. `dsplit` is DERIVED
/// from the workgroup count on both sides — here and in the kernel — precisely so the two cannot
/// drift: widening the merge's CU list is the only input. At `nblk_m <= n_bh` this is `dsplit==1`
/// and the map is identical to the pre-split one. The producer set of a merge item does not
/// depend on `dp` (every D-chunk of a `(b,h)` reads the same `nsplit` flash slices), so widening
/// adds no edges — it only spreads the same edges over more consumers.
fn flash_merge_map(
    n_bh: u32,
    nsplit: u32,
    rows_per_item: u32,
    n_head: u32,
    nblk_f: u32,
    nblk_m: u32,
) -> Vec<Vec<u32>> {
    let dsplit = nblk_m.div_ceil(n_bh.max(1)).max(1);
    (0..nblk_m)
        .map(|j| {
            let mut s: Vec<u32> = (0..n_bh * dsplit)
                .filter(|w| w % nblk_m == j) // the merge items THIS workgroup runs
                .flat_map(|w| {
                    let hb = w / dsplit; // the (b, h) this D-chunk belongs to
                    let (b, h) = (hb / n_head, hb % n_head);
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

/// How many D-chunks each `(row, head)` merge item is split into.
///
/// **This was the L1 decode lever and it is MEASURED DEAD. Default 1; do not change it.**
///
/// The premise was sound on paper: `flash_merge` is 32 workgroups on a 256-CU machine at
/// Gemma-31B decode (`n_bh = t*heads = 32`), holds the machine for 0.928 ms/token with a
/// measured-EMPTY ready queue behind it, and is embarrassingly parallel over D — so `dsplit=8`
/// takes it to 256 workgroups with no new reduction and no new gate, which an ideal-schedule
/// simulation on the real DAG priced at **−0.805 ms/token**, the largest single decode lever.
///
/// Measured (MI355X, Gemma-4-31B bf16, ctx 1024, interleaved, same code object in every arm):
/// **+0.555 ms/token at dsplit=8**, monotone in width (+0.243 at 2, +0.348 at 4), n=28 per arm
/// over three independent sets, every arm token-identical. Wrong sign, and not marginally.
/// `d_flash_merge` in `runtime/amd/op_attention.h` carries the numbers and the mechanism
/// (widening an op widens its COARSE consumer's gate: o_proj then waits on a max over 256
/// stragglers instead of 32).
///
/// The knob survives as the reproduction vehicle only. At 1 the emitted blob is byte-identical
/// to the pre-change emitter's, so nothing ships differently.
fn flash_merge_dsplit() -> u32 {
    std::env::var("PLOW_FLASH_MERGE_DSPLIT")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(1)
        .max(1)
}

#[cfg(test)]
mod flash_merge_map_tests {
    use super::*;

    /// The kernel's work walk, re-implemented here. If `d_flash_merge`'s decomposition and this
    /// map ever disagree the failure is a SILENT wrong token, so pin the contract in a test.
    fn kernel_items(n_bh: u32, nblk_m: u32, j: u32) -> Vec<u32> {
        let dsplit = nblk_m.div_ceil(n_bh.max(1)).max(1);
        (j..n_bh * dsplit).step_by(nblk_m as usize).collect()
    }

    /// Every merge item is run by EXACTLY ONE workgroup, at any width — including widths that are
    /// not a multiple of `n_bh` (there `n_work > nblk_m` and the walk wraps).
    #[test]
    fn every_d_chunk_is_covered_exactly_once() {
        for nblk_m in [1u32, 8, 32, 64, 200, 256] {
            let n_bh = 32;
            let dsplit = nblk_m.div_ceil(n_bh).max(1);
            let mut seen: Vec<u32> = (0..nblk_m)
                .flat_map(|j| kernel_items(n_bh, nblk_m, j))
                .collect();
            seen.sort_unstable();
            assert_eq!(
                seen,
                (0..n_bh * dsplit).collect::<Vec<_>>(),
                "nblk_m={nblk_m}"
            );
        }
    }

    /// `d_flash_prefill`'s OWN q-tile walk, re-implemented from `runtime/amd/op_attention.h:221`
    /// and `:228-230`: `q_tiles = ceil(n_q / (PLOW_WAVES*FA_BQ))`, item
    /// `w = (qt*n_head + h)*nsplit + sp`, run by workgroup `w % nblk_f`. The tile height is
    /// `PLOW_WAVES*FA_BQ` and `FlashPrefill` runs 4-wave, so it is 128 — `FA_BQ`'s own comment at
    /// `op_attention.h:49` spells that out.
    ///
    /// The `128` is written out here rather than taken from [`FLASH_Q_TILE_ROWS`] ON PURPOSE:
    /// this is the KERNEL's number, transcribed from the kernel, and a test that read the
    /// emitter's constant for both sides would be a tautology that passes at any value.
    fn kernel_flash_producers(row: u32, h: u32, nsplit: u32, n_head: u32, nblk_f: u32) -> Vec<u32> {
        const KERNEL_WAVES: u32 = 4; // scripts/build_gfx950.sh: the flash object is -DPLOW_WG_WAVES=4
        const KERNEL_FA_BQ: u32 = 32; // runtime/amd/op_attention.h:49
        let qt = row / (KERNEL_WAVES * KERNEL_FA_BQ);
        (0..nsplit)
            .map(|sp| ((qt * n_head + h) * nsplit + sp) % nblk_f)
            .collect()
    }

    /// EVERY flash slice that wrote a merge item's partials must be in that item's wait set.
    ///
    /// This is the flash -> merge `Dep::Fine` edge, and prefill programs KEEP their fine edges
    /// (`crates/packet/src/devbuild.rs`, the `PLOW_CHAIN_BYPASS` note: "the prefill programs carry
    /// Fine edges and are left untouched"). A missing edge is not a hang and not a fault: the
    /// merge reads `(o, m, l)` partials the flash has not written yet and folds garbage into
    /// `n.at`. Fluent, wrong, and invisible.
    ///
    /// The head counts below are the point. At 8/16/32/64 the producer indices alias mod `nblk_f`
    /// and a WRONG `rows_per_item` still yields a complete map, which is why this survived; at 40
    /// (Qwen3-14B, Qwen2.5-14B, Llama-2-13B) and 28 (Qwen2-57B) it does not.
    #[test]
    fn merge_waits_on_the_slice_that_actually_wrote_the_row() {
        let n_cu = 256u32;
        for heads in [8u32, 16, 20, 24, 28, 32, 40, 48, 64] {
            for t in [128u32, 192, 256, 384, 512, 768, 1024, 2048] {
                let ns = n_cu
                    .div_ceil((t.div_ceil(Q_TILE_ROWS) * heads).max(1))
                    .max(1);
                if ns <= 1 {
                    continue; // fused: flash normalizes in its epilogue, no FlashMerge op
                }
                let (n_bh, nblk_f) = (t * heads, n_cu);
                let nblk_m = (n_bh * flash_merge_dsplit()).min(n_cu).max(1);
                let map = flash_merge_map(n_bh, ns, FLASH_Q_TILE_ROWS, heads, nblk_f, nblk_m);
                let dsplit = nblk_m.div_ceil(n_bh.max(1)).max(1);
                for j in 0..nblk_m {
                    for w in kernel_items(n_bh, nblk_m, j) {
                        let hb = w / dsplit;
                        let (row, h) = (hb / heads, hb % heads);
                        for p in kernel_flash_producers(row, h, ns, heads, nblk_f) {
                            assert!(
                                map[j as usize].contains(&p),
                                "heads={heads} t={t} ns={ns}: merge wg {j} folds row {row} \
                                 head {h} but does not wait on flash slice {p} that wrote it"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The fix must be a NO-OP for every head count shipped today, so it needs no re-measurement:
    /// at 8/16/32/64 the emitted wait sets are byte-identical either way.
    #[test]
    fn shipped_head_counts_are_unaffected_by_the_tile_correction() {
        let n_cu = 256u32;
        for heads in [8u32, 16, 32, 64] {
            for t in [128u32, 256, 512, 1024, 2048] {
                let ns = n_cu
                    .div_ceil((t.div_ceil(Q_TILE_ROWS) * heads).max(1))
                    .max(1);
                if ns <= 1 {
                    continue;
                }
                let (n_bh, nblk_f) = (t * heads, n_cu);
                let nblk_m = (n_bh * flash_merge_dsplit()).min(n_cu).max(1);
                assert_eq!(
                    flash_merge_map(n_bh, ns, Q_TILE_ROWS, heads, nblk_f, nblk_m),
                    flash_merge_map(n_bh, ns, FLASH_Q_TILE_ROWS, heads, nblk_f, nblk_m),
                    "heads={heads} t={t}: the correction changed a shipped program"
                );
            }
        }
    }

    /// dsplit=1 must leave the map byte-identical: the widening is opt-in and the default path
    /// has to stay the shipped one.
    #[test]
    fn dsplit_one_is_the_old_map() {
        let (n_bh, ns, n_head, nblk_f) = (32, 8, 32, 256);
        let map = flash_merge_map(n_bh, ns, 1, n_head, nblk_f, n_bh);
        for (j, s) in map.iter().enumerate() {
            let (b, h) = (j as u32 / n_head, j as u32 % n_head);
            let mut want: Vec<u32> = (0..ns)
                .map(|sp| ((b * n_head + h) * ns + sp) % nblk_f)
                .collect();
            want.sort_unstable();
            want.dedup();
            assert_eq!(s, &want, "wg {j}");
        }
    }

    /// Widened: a merge workgroup must depend on exactly the flash slices of the `(b,h)` whose
    /// D-chunks it runs — no more (that would re-widen the gate), no fewer (that is a race).
    #[test]
    fn dsplit_eight_gates_on_its_own_bh_only() {
        let (n_bh, ns, n_head, nblk_f, nblk_m) = (32u32, 8u32, 32u32, 256u32, 256u32);
        let map = flash_merge_map(n_bh, ns, 1, n_head, nblk_f, nblk_m);
        assert_eq!(map.len(), nblk_m as usize);
        let dsplit = nblk_m / n_bh;
        for (j, s) in map.iter().enumerate() {
            let items = kernel_items(n_bh, nblk_m, j as u32);
            assert_eq!(items.len(), 1, "at 256 wgs each runs exactly one D-chunk");
            let hb = items[0] / dsplit;
            let (b, h) = (hb / n_head, hb % n_head);
            let mut want: Vec<u32> = (0..ns)
                .map(|sp| ((b * n_head + h) * ns + sp) % nblk_f)
                .collect();
            want.sort_unstable();
            want.dedup();
            assert_eq!(s, &want, "wg {j}");
            // 8 of 256 producers, per the doc comment above — not a dense 256-wide gate.
            assert_eq!(s.len(), ns as usize);
        }
    }
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
    emit_xreduce_gather(
        b,
        xgate,
        decode,
        xr_cus,
        &[dep],
        out,
        xr_elems,
        tp,
        slot,
        None,
    )
}

/// [`emit_xreduce`], plus an ALL-GATHER of a column-parallel partial folded into the same
/// packet: `out = sum_r reduced_r + concat_r gathered_r`.
///
/// `gather` is `(slot byte offset, per-rank column count, out row width)`. See
/// [`packet::dev::DevOp::XReduce`] for why the two collectives are one packet — the two
/// partials are ADDED, so the gather is one extra bf16 load per element on a rendezvous
/// that already happened, rather than its own packet (~5.3 us) and its own rendezvous.
///
/// ALWAYS THE ONE-SHOT when a gather is present, on both phases. The two-shot's
/// reduce-scatter owns a 1/N slice of the message and its all-gather phase reassembles it;
/// there is no place in that decomposition for a SECOND, differently-shaped gather, and
/// inventing one to save fabric on a prefill chunk is not worth a second reduction body.
/// The cost is prefill-only and bounded by the one-shot's N× fabric on one collective per
/// MoE layer.
#[allow(clippy::too_many_arguments)]
fn emit_xreduce_gather(
    b: &mut Builder,
    xgate: &mut u32,
    decode: bool,
    xr_cus: &[u32],
    deps: &[u32],
    out: u32,
    xr_elems: u32,
    tp: u32,
    slot: u32,
    gather: Option<(u32, u32, u32)>,
) -> u32 {
    // SIZE THE COLLECTIVE TO ITS ACTUAL WORK. `d_xreduce` gives each thread ONE element
    // (`base = slice*512 + threadIdx.x`, `step = nblk*512`), so a reduction of `xr_elems`
    // saturates at `ceil(xr_elems/512)` workgroups and every workgroup past that does
    // literally nothing — while still polling the packet's SYSTEM-scope arrival counter and
    // taking a SYSTEM-scope acquire (a full L1/L2 invalidate across all 8 XCDs).
    //
    // MEASURED, GLM-5.2 TP4 decode, ctx 1024, interleaved control: `xr_elems` = 6144 needs 12
    // workgroups and the GLM emitter was handing it 256. 244 of 256 workgroups did zero
    // arithmetic in each of the 156 collectives and burned 1.78 ms/CU/token doing it; a token
    // paid 156 x 256 = 39,936 system-scope L2 invalidates to reduce 6144 elements.
    // 28.964 -> 27.041 ms/token, -1.82 (-6.3%), token-identical over 24 generated ids, interleaved
    // controls at 28.964 / 28.639. A flat PLOW_XR_CUS=32 -- the dense emitter's old default --
    // gets only -1.54 on its own interleaved control, because 32 is itself 2.7x wider than this
    // reduction can use. Sizing beats a constant.
    // (perf-data/glm52-gate-stall-attribution.md)
    //
    // This is a pure NARROWING of whatever the caller already allowed, so it can only remove
    // idle participants: `PLOW_XR_CUS` still caps, prefill's t*hidden still asks for the whole
    // machine and still gets it, and the reduction is BIT-IDENTICAL either way — each element's
    // sum runs over the same N peer slots in the same order, only the element->workgroup
    // partition changes.
    //
    // `ceil(xr_elems/512)` is the saturation point of BOTH bodies: the one-shot grid-strides the
    // full `n` by `nblk*PLOW_THREADS`, and the two-shot's all-gather does the same (its
    // reduce-scatter phase saturates even earlier, at `n/nranks`), so this never over-narrows.
    let need = (xr_elems.div_ceil(512).max(1) as usize).min(xr_cus.len());
    let xr_cus = &xr_cus[..need];
    if decode || gather.is_some() {
        let gate = *xgate;
        *xgate += 1;
        let (gslot, gcols, row_w) = gather.unwrap_or((0, 0, 0));
        b.emit(DevOp::XReduce, xr_cus.to_vec(), deps, |d| {
            d.t[0] = out; // reduced [1,hidden] result (local)
            d.i[0] = xr_elems; // elements to reduce (decode: hidden)
            d.i[1] = tp; // n_gpu
            d.i[2] = slot; // partial slot byte offset (§7a)
            d.i[3] = gate; // xctr gate id (unique per collective)
            d.i[4] = gslot; // folded all-gather: its partial slot byte offset
            d.i[5] = gcols; // columns per rank (0 = no gather)
            d.i[6] = row_w; // `out`'s full row width, for the [T, row_w] case
        })
    } else {
        let gate_rs = *xgate;
        *xgate += 1;
        let gate_ag = *xgate;
        *xgate += 1;
        b.emit(DevOp::XReduceTwoShot, xr_cus.to_vec(), deps, |d| {
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
        let c_score = b.emit(score_op, (0..blocks).collect(), &[dep], |d| {
            d.t[0] = score;
            d.t[1] = resid;
            d.t[2] = proj;
            d.t[3] = scale;
            d.i[0] = hidden;
            d.i[1] = n_exp;
            d.i[2] = nb;
            d.f[0] = root;
            d.f[1] = eps;
        });
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
    // Target is AMD (gfx950). Only the prefill lm_head arm reads it — see `pf_gemv_head`.
    amd: bool,
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
                            //
                            // The default is 32, NOT n_cu. Measured on MI355X (Gemma-4 31B bf16, ctx 1024, TP4):
                            //   blocks    16     32     64    128    256(=n_cu)
                            //   us/coll  0.575  0.565  0.563  0.636  0.717
                            // and end-to-end the win is LARGER than the microbench, because the collateral is the
                            // L2 the surrounding GEMV wanted: 11.74 ms/token at 256 vs 10.93 at 32, a 6.9% token-level
                            // win. At 256 blocks a token pays 120 collectives x 256 workgroups = 30,720 system-scope
                            // L2 invalidates. H=5376 saturates on a handful of workgroups, so the extra 224 buy no
                            // bandwidth and cost the cache.
                            //
                            // Bit-identical to the old default: each element's sum still runs over the same N peer
                            // slots in the same order; only the element->workgroup partition changes.
    let xr_cus: Vec<u32> = {
        let k = std::env::var("PLOW_XR_CUS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(32)
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
    //
    // `rows` is the ONLY parallel axis these ops have, so at decode (t == 1) NormResidualNorm
    // runs on ONE workgroup 120 times per token, 0.71 ms with 255 CUs idle. A feature axis was
    // added to it and MEASURED, and it loses by a wide margin at every k -- the two extra
    // counter-gated packets it needs cost more than the whole op. The numbers, the mechanism,
    // and why knob-contract 7a's "a gate is <=0.64 us" does not apply to a SERIAL split are
    // recorded above d_norm_residual_norm in runtime/amd/op_norm.h. Read that before widening
    // this. At decode batch B the row axis already gives it B workgroups for free.
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
        // mma.sync.m16n8k32.
        //
        // The tile now comes from `pick_tile` ASKED FOR fp8, rather than from mapping the bf16
        // answer onto an fp8 twin afterwards. The old three-arm `match` was §4's bug shape in
        // miniature: its `_` arm sent every unrecognised opcode to `GemmFp8` (256x256), so the
        // two rungs added by the tile-inventory campaign would have silently collapsed to the
        // largest tile on exactly the fp8 shapes they were added for. Asking the selector for
        // the encoding also lets the answer DIFFER from bf16's, which is the point of making
        // precision an input.
        if !gemv_family && fp8 {
            let op = pick_tile(m, nn, k, n_cu, kernelcaps::QuantScheme::W8A8);
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
            pick_tile(m, nn, k, n_cu, kernelcaps::QuantScheme::None)
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
        // THE SAME LDS PRECONDITION `glu_fused` CHECKS. `gemv_qkv_rows` reads x
        // only through `ld_lds8` — it has no global-read arm — and `op_gemm.h`
        // says so: *"x is ALWAYS staged in LDS here: plowc emits this op only
        // when M*K fits GM_LDS_HALVES."* That precondition was stated and never
        // enforced for THIS op, only for `GemvGlu`.
        //
        // MEASURED, Gemma-4-31B (hidden 5376), PLOW_DECODE_BATCH=16: the arena
        // holds 73728 halves, `M*K` is 16*5376 = 86016, and row `m` lives at
        // `lds[m*K ..]` — so rows 0..12 fit and rows 13, 14, 15 run past the
        // end. Sequences 13/14/15 decoded fluent-but-WRONG streams (correct
        // first token from prefill, divergent from the second) while 0..12 were
        // token-identical to a batch-1 run. §4's bug shape: a documented
        // precondition with nothing checking it.
        let fuse_qkv = gemv_family
            && !keqv
            && !fp8
            // `gemv_staged_rows`, not `t`: with `PLOW_GEMV_WALK` the staging moves inside the
            // row loop and the bound stops depending on M. See §6g-WALK's companion change.
            && (gemv_staged_rows(t) as u64 * c.hidden as u64) <= GM_LDS_HALVES
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
        let ns =
            if gemv_family && full && ctx > 8192 && c.kvh_full >= 4 && c.kvh_slide != c.kvh_full {
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
            let nd: &[u32] = if block_mode && l == block.start {
                &[]
            } else {
                &[dep]
            };
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
        // The `8` is PLOW_WAVES and it is LOAD-BEARING: a headnorm work item is one HEAD, and a
        // head is exactly one wave's 64 lanes x E elements — the layout that keeps each RoPE pair
        // (i, i+hd/2) inside one lane. So `t*heads` waves is all the parallelism the op has, and
        // spreading them one-per-workgroup instead of eight only adds workgroups to the gate.
        // MEASURED (n=6 interleaved, Gemma-4-31B bf16, ctx 1024): 4 -> 32 workgroups is
        // 18.057 -> 18.460 ms/token, a REGRESSION. Do not widen this without first changing the
        // head->wave map, which cannot be done at hd=256 without breaking pair locality.
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
            // L1: fold a D-chunk axis into the merge's work id so it can occupy more than
            // `t*heads` (= 32 at Gemma-31B decode) of 256 CUs. `flash_merge_map` and
            // `d_flash_merge` both RE-DERIVE dsplit from this length, so this is the single
            // input. Default 1 => `(t*heads).min(n_cu)`, exactly as before.
            let mg_cus: Vec<u32> =
                (0..(t * heads * flash_merge_dsplit()).min(n_cu).max(1)).collect();
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
                if gemv_family { 1 } else { FLASH_Q_TILE_ROWS },
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
        // Same bound, same reason as `fuse_qkv` above: `gemv_glu_rows` also reads x only
        // through LDS, and with the walk on it stages `min(MM, M)` rows, not M.
        let glu_fused =
            gemv_family && (gemv_staged_rows(t) as u64 * c.hidden as u64) <= GM_LDS_HALVES;
        let gemm_glu = !gemv_family && glu_fusion_wins(t, inter_l, c.hidden, n_cu);
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
                let _glu_op = if fp8 {
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
                    b.emit(
                        DevOp::MoeExpertGluGemmaFp8,
                        glu_cus,
                        &[c_rt, c_xn2_local],
                        |d| {
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
                        },
                    )
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
                let tail_fuse =
                    std::env::var("PLOW_GEMMA_MOE_TAIL_FUSE").ok().as_deref() == Some("1");
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
                    let c_glu = b.emit(
                        DevOp::MoeGroupGluGemmaPfW8a8,
                        all.clone(),
                        &[c_align, c_xn2q],
                        |d| {
                            d.t[0] = n.moe_fug;
                            d.t[1] = n.xqh; // xn2 e4m3
                            d.t[2] = w.ewt; // fp8 expert weights
                            d.t[3] = n.moe_meta;
                            d.t[4] = n.moe_rowtok;
                            d.t[5] = n.ash; // per-token a_scale
                            d.t[6] = w.est; // per-channel weight scales
                            d.i[0] = c.moe_inter;
                            d.i[1] = c.hidden;
                            d.i[2] = c.n_exp;
                            d.i[5] = c.mlp_act;
                        },
                    );
                    // quant the gathered GLU output (total_pad rows, moe_inter width) for the down GEMM.
                    let c_fuq = b.emit(DevOp::QuantFp8, all.clone(), &[c_glu], |d| {
                        d.t[0] = n.moe_fuq;
                        d.t[1] = n.moe_fug;
                        d.t[2] = n.moe_fus;
                        d.i[0] = moe_total_pad;
                        d.i[1] = c.moe_inter;
                    });
                    b.emit(
                        DevOp::MoeGroupDownGemmaPfW8a8,
                        all.clone(),
                        &[c_fuq, c_align],
                        |d| {
                            d.t[0] = n.moe_part;
                            d.t[1] = n.moe_fuq; // fu e4m3
                            d.t[2] = w.ewt;
                            d.t[3] = n.moe_meta;
                            d.t[4] = n.moe_rowpart;
                            d.t[5] = n.moe_rowgate;
                            d.t[6] = w.est;
                            d.t[7] = n.moe_fus; // per-row fu scale
                            d.i[0] = c.hidden;
                            d.i[1] = c.moe_inter;
                            d.i[2] = c.n_exp;
                        },
                    )
                } else {
                    let c_glu = b.emit(
                        DevOp::MoeGroupGluGemmaPf,
                        all.clone(),
                        &[c_align, c_xn2],
                        |d| {
                            d.t[0] = n.moe_fug;
                            d.t[1] = n.moe_xn2;
                            d.t[2] = w.ewt;
                            d.t[3] = n.moe_meta;
                            d.t[4] = n.moe_rowtok;
                            d.i[0] = c.moe_inter;
                            d.i[1] = c.hidden;
                            d.i[2] = c.n_exp;
                            d.i[5] = c.mlp_act; // 0 GeGLU (Gemma)
                        },
                    );
                    // grouped down GEMM + gate-scale + scatter -> part[T,k,H].
                    b.emit(
                        DevOp::MoeGroupDownGemmaPf,
                        all.clone(),
                        &[c_glu, c_align],
                        |d| {
                            d.t[0] = n.moe_part;
                            d.t[1] = n.moe_fug;
                            d.t[2] = w.ewt;
                            d.t[3] = n.moe_meta;
                            d.t[4] = n.moe_rowpart;
                            d.t[5] = n.moe_rowgate;
                            d.i[0] = c.hidden;
                            d.i[1] = c.moe_inter;
                            d.i[2] = c.n_exp;
                        },
                    )
                };
                // T-row combine + sandwich: out[t] = RMSNorm(Σ_slot part[t][slot], g_pf2) + h1[t].
                let c_comb = b.emit(
                    DevOp::MoeCombineNormGemmaPf,
                    all.clone(),
                    &[c_dn, c_h1],
                    |d| {
                        d.t[0] = n.moe_comb;
                        d.t[1] = n.moe_part;
                        d.t[2] = n.moe_h1;
                        d.t[3] = w.g_pf2;
                        d.i[0] = c.hidden;
                        d.i[1] = c.top_k;
                        d.i[2] = t;
                        d.f[0] = c.eps;
                    },
                );
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
    // DEFAULT ON for gfx950. The argument is the row efficiency above and it does not depend on
    // dtype: at M=1 the narrowest tile still computes BM=64 rows to keep one, so 1.5% of the work
    // is used. The AMD prefill object carries `case PLOW_DOP_GEMV` unconditionally, so the arm is
    // simply there — unlike sm_120, where it is compiled out unless `-DPLOW_NV_PF_GEMV_HEAD=1` and
    // a packet using it would `__trap()`. Hence: on by default where the arm exists, opt-in where
    // it does not. `PLOW_PF_GEMV_HEAD` still forces it either way (=1 on, =0 off), so the sm_120
    // A/B and the AMD escape hatch are both preserved, and an empty `--arch` (the golden tests)
    // keeps the old emission byte for byte.
    let pf_gemv_head = !decode
        && match std::env::var("PLOW_PF_GEMV_HEAD").ok().as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => amd,
        };
    let lm_op = if gemv_family || pf_gemv_head {
        DevOp::Gemv
    } else {
        pick_tile(1, vocab_l, c.hidden, n_cu, kernelcaps::QuantScheme::None)
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
    // argmax and thus the SAME global token id, so no cross-rank XArgmaxFin fold is needed here —
    // c_fin already wrote the correct global id into in.ids on every rank. The fold itself exists
    // now (`d_xargmax_fin_mega`, and `mla.rs`'s `GLM_SHARD_HEAD` arm uses it); what blocks this
    // emitter is the TIED head's `emb` read, not the collective. See the declare() note.
    let _ = c_fin;
}

/// Blocks the argmax partial reduction is spread over. 64 x 512 threads covers a 262144-entry
/// vocab in one strided pass per thread.
const AMAX_BLOCKS: u32 = 64;

/// L2 — FINER DECODE-GEMV SLICES. `PLOW_GEMV_SPLIT=S` emits `S * n_cu` slices for the
/// machine-filling `Gemv` / `GemvGlu` / `GemvQkv` packets of the DECODE program instead of `n_cu`,
/// so a workgroup that finishes early claims another slice off the global queue instead of waiting
/// at the barrier. Decode only — see [`Builder::set_gemv_split`] for why prefill is excluded.
///
/// **DEFAULT 1, and it stays 1: S=2 is measured a LOSS.** The hypothesis was the straggler tail —
/// Σ(max−mean) over the ≥128-slice decode instructions is 1.80 ms/token, ±19% on identical work,
/// and that spread is random per-(CU, packet) rather than per-CU systematic, which is exactly the
/// shape finer dynamic slicing removes. Ideal-schedule simulation on the real 676-packet DAG with
/// the measured durations predicted 16.622 → 14.023 ms at S=2 against a 12.92 ms work bound.
///
/// It does not survive contact, and the reason is in the kernel, not the scheduler. `gemv_rows`
/// hands column `n` of its slice to wave `n % PLOW_WAVES`, so a slice costs `ceil(per/8)`
/// column-times with `per = ceil(N/nblk)` — halving `per` does NOT halve the slice, it rounds up
/// against 8 waves, and the two rounds then cost MORE than the one they replaced. Gemma-4-31B
/// o_proj (N=5376) goes 21 columns/slice → `ceil(21/8)=3` at S=1 versus 11 → `ceil(11/8)=2` ×2
/// rounds at S=2, and N=5376 is not even divisible by 512 (23 of the 512 slices get no work).
///
/// MEASURED, MI355X, Gemma-4-31B bf16 real weights, 1024-token prompt, 64 greedy decode steps,
/// interleaved arms under `perf-data/harness/gpulease -n 1`, contended (rc=76) runs discarded.
/// Objects `build-amd/l2-hsaco2`, blobs `build-amd/l2-s{1,2,4}`, harness `gemma4_chat.c`:
///
/// | S | slices/gemv | wg-packets/token | decode ms/token (median, n) | spread | Δ |
/// |---|---|---|---|---|---|
/// | **1** (ships) | 256 | 79,947 | **17.0** (n=8) | 17.0–17.3 | — |
/// | 2 | 512 | 139,083 | 19.8 (n=7) | 19.5–19.9 | **+2.8** |
/// | 4 | 1024 | 257,355 | 24.7 (n=7) | 24.6–25.2 | +7.7 |
///
/// Prefill is untouched (162 ms at every S) because the split is scoped to the decode builder.
///
/// Isolated-kernel confirmation (same `d_gemv_t`, grid = S·256, `build-amd/l2-hsaco2/gemmtest`):
/// o_proj 0.027 → 0.030 → 0.032 ms, down 0.046 → 0.048 → 0.052, gate/up 0.044 → 0.046 → 0.048.
/// Every shape is monotonically worse, so the loss is in the op and not in the packet protocol.
/// Charging the queue-claim cost back honestly: 59,136 extra workgroup-packets × 2.2 µs / 256 CUs
/// = +0.51 ms/token at S=2, which is under a fifth of the observed +2.8 ms. The rest is the
/// rounding above.
///
/// Tokens are BIT-IDENTICAL at S=1/2/4 (same md5 over 64 greedy tokens), as the output-stationary
/// argument in [`Builder::set_gemv_split`] requires, so this is a clean performance null and not a
/// correctness question. **Attacking the straggler tail needs the wave assignment inside
/// `gemv_rows` to become dynamic too; splitting packets alone cannot pay for the rounding.**
fn gemv_split() -> u32 {
    std::env::var("PLOW_GEMV_SPLIT")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(1)
}

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
    /// `PLOW_L2_PLACE` (née `PLOW_L2_PLACE`): the target's L2 partition geometry from
    /// `hwspec::GpuSpec::l2_partitioning`, plus its workgroup->domain map. `Some` groups the
    /// device blob's global-queue stream by L2 domain (via [`packet::devbuild::Builder`]'s
    /// `seg`-as-domain), so a physical-SM-aware interp pulls its domain's packets. `None` ⇒
    /// byte-identical. Dense-GQA path only. See `plans/l2-placement-generic.md`.
    pub l2_layout: Option<packet::devbuild::L2Layout>,
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

/// What the verification gate actually DID, recorded verbatim in `build.json`.
///
/// A SKIPPED GATE IS INDISTINGUISHABLE FROM A PASSING ONE unless the artifact
/// says which happened. That is this repo's signature failure — `tuning.tier`
/// reported `portable` both when the analytical model was chosen and when no
/// measurement had ever been taken, and the resulting GLM prefill numbers were
/// meaningless for a long time before anyone noticed. A warning line in a build
/// log is not a defence: the log is gone by the time someone asks "was this
/// blob verified?". The blob's own manifest has to answer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LeanReport {
    /// A Lean ordering certificate was obtained for EVERY program in the blob.
    pub verified: bool,
    /// The Lean lower-bound oracle ran and reported.
    pub oracle: bool,
    /// Why `verified`/`oracle` are false. `None` only when both are true.
    ///
    /// Never `Some` because verification FAILED — a rejection aborts emission,
    /// so no blob with a rejected program is ever written and no manifest for
    /// one can exist.
    pub reason: Option<String>,
}

impl LeanReport {
    /// The gate did not run, for `reason`.
    pub fn skipped(reason: impl Into<String>) -> LeanReport {
        LeanReport {
            verified: false,
            oracle: false,
            reason: Some(reason.into()),
        }
    }
}

/// Read-only verification hook for [`run_verified`], called with the finished
/// [`packet::devbuild::Model`] immediately before the blob is written
/// (dense-GQA path only for now). An `Err` ABORTS emission.
///
/// `Ok` carries a [`LeanReport`] rather than `()`: the hook is allowed to
/// decline to run (no verifier binary on this machine) and the caller must be
/// able to tell that apart from a clean pass. Reserve `Err` for a verifier that
/// looked at the program and REJECTED it.
pub type VerifyHook = Box<dyn Fn(&packet::devbuild::Model) -> Result<LeanReport, String>>;

/// A hook that runs nothing and says so. Lets a caller keep the "always supply
/// a hook" shape so the *reason* for skipping is authored where it is known,
/// instead of the manifest having to guess from a `None`.
pub fn skip_hook(reason: impl Into<String>) -> VerifyHook {
    let r = reason.into();
    Box::new(move |_| Ok(LeanReport::skipped(r.clone())))
}

/// Run the verification gate against the finished model, THE ONE WAY.
///
/// Every emit path calls this immediately before `std::fs::write` of the blob,
/// so "verified" means the same thing on all of them and a new emitter cannot
/// quietly acquire a different policy. `Err` is a rejection and is fatal; a
/// missing verifier is the hook's problem and arrives as `Ok(skipped)`.
pub(crate) fn apply_verify_gate(
    m: &packet::devbuild::Model,
    verify: Option<&VerifyHook>,
) -> LeanReport {
    match verify {
        Some(v) => match v(m) {
            Ok(r) => r,
            Err(e) => panic!("devblob verification rejected the emitted program: {e}"),
        },
        None => LeanReport::skipped("no verification hook supplied by the caller"),
    }
}

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
    /// Target is AMD. Set from `--arch`/`--gpu` at construction; reaches only `pf_gemv_head`.
    amd: bool,
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
        amd: bool,
    ) -> (Self, Vec<packet::devbuild::TensorDecl>, Vec<GenTensor>) {
        let mut tb = Builder::new(n_cu);
        let tn = declare(
            &mut tb,
            c,
            ctx,
            ns_pre,
            fp8,
            w8a8,
            fp8_kv,
            fp8_kv_full,
            dbatch,
            moe_pf,
            block.clone(),
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
            amd,
        };
        (e, tensors, gen)
    }
}

impl DevblobEmitter for DenseGqaEmitter<'_> {
    fn emit_prefill(&self, b: &mut Builder, t: u32) {
        let mut dummy = Vec::new();
        emit_phase(
            b,
            self.c,
            self.ls,
            &self.tn,
            t,
            self.ctx,
            Mode::Prefill,
            self.n_cu,
            &mut dummy,
            self.fp8,
            self.w8a8,
            self.fp8_kv,
            self.fp8_kv_full,
            self.block.clone(),
            self.block_mode,
            self.amd,
        );
    }
    fn emit_decode(&self, b: &mut Builder, dbatch: u32, dmode: Mode, kv_rows: &mut Vec<u32>) {
        // Decode passes w8a8=false, exactly as the historical call site did.
        emit_phase(
            b,
            self.c,
            self.ls,
            &self.tn,
            dbatch,
            self.ctx,
            dmode,
            self.n_cu,
            kv_rows,
            self.fp8,
            false,
            self.fp8_kv,
            self.fp8_kv_full,
            self.block.clone(),
            self.block_mode,
            self.amd,
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

    // BEFORE any emitter runs: the GEMV census needs the store's answer in hand by the time
    // the first `Builder::emit_dep` fires. Costs one store read on a `PLOW_TUNE_DUMP` run and
    // one on every other run too — the same read `pick_tile` already does, and a compile is a
    // process. Placed at the single entry point rather than per-emitter so no emit path can
    // be added that skips it.
    install_gfx950_gemv_cases();

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
    // PLOW_L2_PLACE is wired only on the dense-GQA path below (b/bd builders). The
    // GLM/Kimi/DeepSeek/Nemotron emitters have their own builders and never call
    // set_l2_placement, so the flag would silently no-op there — say so rather
    // than let a user believe placement is active. See plans/l2-placement-generic.md.
    //
    // AMD IS NO LONGER REFUSED HERE, and the reason the refusal existed is worth keeping.
    //
    // `StreamEnt::seg` carries the wave class the AMD host relaunches each segment at: 4 waves for
    // a `FlashPrefill` run, 8 for everything else. Placement REPURPOSES that field as an L2
    // domain, which is fine on sm_120 — that interpreter runs one cooperative launch at a fixed
    // block size and never reads a wave class — and catastrophic on a MULTI-SEGMENT gfx950
    // program, where losing the tags collapses the whole prefill program into segment 0. The host
    // then sees flash packets in that segment, dispatches the ENTIRE program on the 4-wave flash
    // object, and that object's body is `if (op == FLASH_PREFILL…)` with no switch: every GEMM,
    // every norm and the lm_head are silently dropped. Prefill "succeeds" in 8.7 ms instead of
    // 72.1 and `act.logits` is all zeros. This cost three agents a long time to find, because the
    // packet loads, runs, and differs from a correct one ONLY in this field.
    //
    // But the conflict is a property of the PROGRAM, not of the target. It exists exactly when a
    // program has more than one wave class. Decode has no `FlashPrefill` op at all, so every
    // `seg` is already 0 and there is nothing in the field to destroy — and decode is where the
    // locality lever is (§7b: 66% of packets and 45.5% of the token on ≤32 effective workgroups).
    // Refusing the whole target therefore gave up the case that matters to keep the case that
    // breaks. `Builder::finish` now applies the real test — it is the only place that knows
    // `cur_seg` — and skips placement per program, byte-identically, when the program is
    // segmented. Prefill on AMD is still not placed; it just is not the flag's decision to make
    // from the target name.
    //
    // The other half of the old reasoning — "the flag is explicitly NVIDIA-named, so using it
    // here buys nothing" — is why the flag was renamed. An L2 domain is a GPC on NVIDIA and an
    // XCD on AMD; `hwspec` describes both.
    //
    // Resolved once: three flags now depend on `amd`, and computing it per site is how the second
    // and third of them ended up ungated.
    let amd = target_is_amd(&arch, &gpu);
    warn_uniseg_on_amd(amd);
    if l2_layout.is_some()
        && matches!(
            model_type.as_str(),
            "glm_moe_dsa"
                | "kimi_k2"
                | "kimi"
                | "deepseek_v3"
                | "deepseek_v2"
                | "nemotron_h"
                | "nemotron3"
                | "nemotron"
        )
    {
        eprintln!(
            "  PLOW_L2_PLACE ignored: L2-domain placement is dense-GQA only, not wired for \
             model_type {model_type:?} (its emitter is a separate path)"
        );
    }
    // Kimi-K3 (`kimi_k3`): multimodal wrapper over a hybrid MLA+KDA `kimi_linear` text tower with
    // a latent mxfp4 MoE. Claimed HERE, before anything else, for two reasons. (1) It nests its
    // geometry under `text_config`, which `cfg_from` (crates/devgen/src/config.rs:70) treats as
    // "Gemma-4 multimodal" unconditionally — that is why an unmodified plowc died on
    // `config.rs:93` unwrapping Gemma's `layer_types`, an error naming the wrong field of the
    // wrong architecture. (2) Its MLA keys have the SAME spelling as DeepSeek's, so the `kimi`
    // arm below would parse it, mis-read the MoE (`n_routed_experts` is absent) and emit a blob
    // for a model that is 69/93 linear attention. It would ALSO have defaulted the absent
    // `rope_theta` to GLM's 8e6 — that half is now closed independently: `cfg_glm` reads the
    // theta as an `Option` and `require_mla_rope` refuses a NoPE checkpoint, so the `kimi` arm
    // is loud about this one too even without the claim above. `kimi_k3_emit` never returns: it
    // validates everything the front end can and then reports what is not implemented.
    if model_type == "kimi_k3" {
        // `K3_FULL=1` selects the real emit (`k3_emit_full`), mirroring GLM's
        // `GLM_FULL`. The DEFAULT stays the capability report, because the
        // host-side mxfp4 expert bind and the Mixtral `w1/w2/w3` name template
        // are still missing — a blob emitted today fails at LOAD with a missing
        // weight, which is loud and correct but is not what someone who has not
        // read the report is expecting.
        if std::env::var("K3_FULL").ok().as_deref() == Some("1") {
            mla::k3_emit_full(
                &dir,
                ctx,
                &out,
                n_cu,
                tp,
                rope_gen,
                verify.as_ref(),
                l2_layout,
            );
            return;
        }
        mla::kimi_k3_emit(&dir, ctx, tp, block_spec.as_deref());
    }
    if model_type == "glm_moe_dsa" {
        // GLM `--block` (M2, plans/block-asset-harness.md §5.3/§7): single-block
        // extraction on the separate GLM emitter. Absent => the unchanged glm_main
        // path (byte-identical).
        // THE HOOK USED TO STOP HERE. `verify` was moved into `emit_dense_gqa` and
        // these early returns dropped it, so `--lean-verify --emit devblob` on
        // GLM-5.2 — the model this repo actually ships — verified nothing and said
        // nothing. That is §4's recurring shape (an arm exists and nothing routes
        // to it) applied to a safety gate, and it is exactly why the manifest now
        // has to state whether the gate ran.
        match &block_spec {
            Some(spec) => glm_emit_block(
                &dir,
                ctx,
                &out,
                n_cu,
                tp,
                spec,
                rope_gen,
                &arch,
                verify.as_ref(),
            ),
            None => glm_main(&dir, ctx, &out, n_cu, tp, rope_gen, &arch, verify.as_ref()),
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
        let mla_arch = if model_type.starts_with("kimi") {
            MlaArch::Kimi
        } else {
            MlaArch::DeepSeek
        };
        match &block_spec {
            Some(spec) => kimi_emit_block(
                &dir, ctx, &out, n_cu, tp, spec, mla_arch, rope_gen, &arch, verify.as_ref(),
            ),
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
        verify,
    );
}

/// Is the emit target AMD? `--arch gfx*` or a `--gpu` the registry says is an AMD part.
///
/// Both signals, for the reason `check_fp8_a_scale_bound` documents: they can disagree, and the
/// assets that motivated that gate were emitted `--arch sm_120a` for an MI350X. An empty `--arch`
/// with an empty `--gpu` (the golden tests, the legacy CLI) answers false, so those emissions stay
/// byte for byte what they were.
fn target_is_amd(arch: &str, gpu: &str) -> bool {
    arch.starts_with("gfx")
        || hwspec::registry::lookup(gpu).is_some_and(|s| s.vendor == hwspec::Vendor::Amd)
}

/// The largest `top_k` the AMD MoE routers can select. Mirrors `PLOW_MOE_MAX_TOPK` in
/// `runtime/amd/op_moe.h`, and `moe_topk_matches_the_amd_kernel` PARSES that `#define` and fails
/// if the two drift — the same discipline `GFX950_DISPATCHED` applies to `interp.hip`.
///
/// This is the compile-time half of a two-part bound; `moe_bound_topk` in the kernel is the other.
/// Both routers select into a `[PLOW_MOE_MAX_TOPK]` array, and `d_moe_router_topk`'s rank pass
/// additionally writes `wl[rank]` into an LDS carve of exactly that many entries. Past the bound
/// the kernel cannot produce the routing the packet asked for, and the failure is not a crash: the
/// table slots above the bound are never written, every expert body loops to the packet's
/// (unbounded) `top_k` operand, and the renormalisation denominator covers only the slots that
/// were filled. Fluent output, wrong expert set, wrong gates.
///
/// So the emit refuses instead.
///
/// RAISED 8 -> 16 for Kimi-K3 (top-16 of 896), after measuring that the raise is free: the gfx950
/// decode object reports the identical budget at 8 and 16 (SGPR 106, VGPR 248, occupancy 2, spill
/// 80/0, LDS 147464), and the bound appears in no emitted instruction, so every existing packet is
/// byte-identical. `runtime/amd/op_moe.h` carries the measurement and the caveat that k > 8 has
/// not yet been executed on hardware.
pub(crate) const MOE_MAX_TOPK: u32 = 16;

/// Refuse to emit a MoE packet the AMD routers cannot execute.
///
/// Called from every `config.json` parse that produces a routed-expert count, so the refusal
/// happens once, at the earliest point that knows the model's name — not at the emit site, where
/// there are four of them and a fifth would be added without this check.
pub(crate) fn require_moe_topk(top_k: u32, model: &str) {
    assert!(
        top_k <= MOE_MAX_TOPK,
        "{model}: top_k = {top_k} exceeds PLOW_MOE_MAX_TOPK = {MOE_MAX_TOPK}, the width both AMD \
         MoE routers select into (runtime/amd/op_moe.h). Emitting anyway would route to the top \
         {MOE_MAX_TOPK} of {top_k} experts, renormalise the gates over that subset, and leave the \
         remaining {} table slots unwritten for the expert bodies to read as uninitialised \
         scratch — a model that is fluent and wrong, with nothing at runtime to say so. Raise \
         PLOW_MOE_MAX_TOPK and devgen::MOE_MAX_TOPK together (a drift test enforces the pair) \
         after re-checking the LDS carve at op_moe.h and the megakernel's register budget.",
        top_k - MOE_MAX_TOPK
    );
}

/// Refuse to emit an MLA packet whose positional encoding this compiler cannot determine.
///
/// Same discipline and same call site as [`require_moe_topk`]: run at the `config.json` parse,
/// once, where the model's name is still in hand — not at the emit, where there are four
/// `HeadNormRope` sites and a fifth would be added without the check.
///
/// # The three ways this went wrong silently
///
/// `cfg_glm` read the theta as `v["rope_theta"].as_f64().unwrap_or(8_000_000.0)`.
///
/// 1. **NoPE routed into the rope arm.** Kimi-K3 sets `mla_use_nope: true`; its modeling code has
///    `self.rotary_emb = None` and `assert self.use_nope`, and its config carries no `rope_theta`
///    anywhere. The default handed it GLM's 8e6 and rotated 64 dims the model uses as plain
///    content dims. This is the recurring bug shape with the polarity flipped — a *correct* arm
///    selected for a model that does not want it — and the output is fluent, not broken.
/// 2. **The default was load-bearing for the SHIPPING model.** GLM-5.2's `config.json` has no
///    top-level `rope_theta` either: transformers 5.x moved it to `rope_parameters.rope_theta`
///    (= 8000000). So the emitter was not reading GLM's theta at all — it matched only because
///    the literal in this tree happens to equal it. A retrain at a different theta reads exactly
///    the same and is wrong.
/// 3. **Scaling was ignored.** `rope_scaling` / a `rope_type` other than `"default"` (yarn,
///    linear, llama3) changes the frequencies, and `declare_glm` builds its tables with
///    `RopeScale::None`. An unhandled scheme is refused instead of being applied as `default`.
///
/// So: the theta must be FOUND, the NoPE flag must AGREE with whether one was found, and the
/// scaling scheme must be one the tables actually implement. Anything else is a refusal.
pub(crate) fn require_mla_rope(
    theta: Option<f64>,
    use_nope: bool,
    rope_type: Option<&str>,
    has_rope_scaling: bool,
    model: &str,
) {
    match (use_nope, theta) {
        (true, found) => panic!(
            "{model}: `mla_use_nope` is set — this MLA carries NO positional encoding, and plow's \
             MLA emit applies an interleaved partial RoPE unconditionally. Refusing rather than \
             rotating the qk_rope dims the model treats as content: the result would be plausible \
             logits from the wrong model, with nothing at runtime to say so. (config theta: {}.) \
             Implementing NoPE is not just deleting the two HeadNormRope ops — the k-side one is \
             also the only writer of the `kv.{{l}}.krot` cache row, AND the instruction the AMD \
             loader and glm52_decode.c both SCAN for to patch that row's position each step. \
             Dropping it leaves the rope half of every cached key uninitialised and quietly \
             removes the layer from the KV-row-writer list (see `GlmCfg::rope_theta`).",
            match found {
                Some(t) => format!("{t} — present AND `mla_use_nope`, which contradict"),
                None => "absent, consistent with NoPE".to_string(),
            }
        ),
        (false, None) => panic!(
            "{model}: no RoPE theta in config.json. Looked for `rope_theta` and \
             `rope_parameters.rope_theta` (transformers 5.x spelling) and found neither, and \
             `mla_use_nope` is not set, so this is not a NoPE model either. This used to default \
             to 8000000 — GLM's value — which meant the emitter silently substituted one model's \
             positional encoding for another's. Add the key, or set `mla_use_nope` if the model \
             genuinely has none."
        ),
        (false, Some(t)) => {
            assert!(
                t.is_finite() && t > 1.0,
                "{model}: rope_theta = {t} is not a usable base for cos(p / theta^(2i/d))"
            );
            // `declare_glm` materialises its tables with `RopeScale::None`. Any scheme that
            // rescales the frequencies would be dropped here and nowhere else.
            assert!(
                !has_rope_scaling && matches!(rope_type, None | Some("default")),
                "{model}: rope_type = {:?}, rope_scaling present = {has_rope_scaling}. The MLA \
                 tables are built with RopeScale::None, so a scaled scheme (yarn / linear / \
                 llama3) would be emitted as an UNSCALED RoPE at theta {t} — correct-looking \
                 tables, wrong long-context behaviour. Wire the scheme into \
                 `GenTensor::rope_pair` before accepting it.",
                rope_type.unwrap_or("default")
            );
        }
    }
}

/// The opcodes the gfx950 interpreter actually dispatches.
///
/// Kept as `PLOW_DOP_*` spellings so the drift test can compare them to `runtime/amd/interp.hip`
/// directly, with no name mapping in between to get wrong.
///
/// WHY THIS EXISTS. AMD's dispatch `default:` is `/* PLOW_DOP_NOP */` — it writes NOTHING. An
/// opcode with no arm therefore does not trap, it silently leaves the output buffer untouched, and
/// the failure surfaces as an accuracy bug somewhere downstream. sm_120's default is `__trap()`,
/// which is why the same class of mistake is loud there and silent here. Three separate instances
/// landed in one week (a bf16 `GemmSmall` compiled out of the fp8 prefill object; the flash object
/// chosen without following the KV axis; the KV axis dragging the weight axis), and a fourth was
/// found latent: `PLOW_FUSE_ARGMAX` emits `GEMV_ARGMAX`, which has no AMD arm at all, so a decode
/// would have argmaxed over an untouched buffer and returned token 0 forever.
///
/// Gating each flag as it is discovered is whack-a-mole. This is the general form: whatever the
/// flags did, the STREAM is checked against what the target can run.
const GFX950_DISPATCHED: &[&str] = &[
    "PLOW_DOP_ADD_NORM",
    "PLOW_DOP_ARGMAX",
    "PLOW_DOP_ARGMAX_FIN",
    "PLOW_DOP_ATTN_RES",
    "PLOW_DOP_ATTN_SELECT",
    "PLOW_DOP_DENSE_GLU_FP8_BLK",
    "PLOW_DOP_EMBED",
    "PLOW_DOP_FLASH_DECODE",
    "PLOW_DOP_FLASH_DECODE_FP8",
    "PLOW_DOP_FLASH_GATHER_DECODE",
    "PLOW_DOP_FLASH_GATHER_PREFILL",
    "PLOW_DOP_FLASH_MERGE",
    "PLOW_DOP_FLASH_MLA_DECODE",
    "PLOW_DOP_FLASH_MLA_DECODE_FP8",
    "PLOW_DOP_FLASH_MLA_PREFILL",
    "PLOW_DOP_FLASH_MLA_PREFILL_FP8",
    "PLOW_DOP_FLASH_PREFILL",
    "PLOW_DOP_FLASH_PREFILL_FP8",
    "PLOW_DOP_GEMM",
    "PLOW_DOP_GEMM_C5",
    "PLOW_DOP_GEMM_C5_FP8",
    "PLOW_DOP_GEMM_C5_MXFP4",
    "PLOW_DOP_GEMM_FP8",
    "PLOW_DOP_GEMM_FP8_BLK",
    "PLOW_DOP_GEMM_GLU",
    "PLOW_DOP_GEMM_GLU_FP8",
    "PLOW_DOP_GEMM_GLU_MXFP4",
    "PLOW_DOP_GEMM_MED",
    "PLOW_DOP_GEMM_MED_FP8",
    "PLOW_DOP_GEMM_MED_MXFP4",
    "PLOW_DOP_GEMM_MXFP4",
    "PLOW_DOP_GEMM_SMALL",
    "PLOW_DOP_GEMM_SMALL_FP8",
    "PLOW_DOP_GEMM_SMALL_MXFP4",
    "PLOW_DOP_GEMM_WIDE",
    "PLOW_DOP_GEMM_WIDE_FP8",
    "PLOW_DOP_GEMM_WIDE_MXFP4",
    "PLOW_DOP_GEMV",
    "PLOW_DOP_GEMV_FP8",
    "PLOW_DOP_GEMV_FP8_BLK",
    "PLOW_DOP_GEMV_GLU",
    "PLOW_DOP_GEMV_GLU_FP8",
    "PLOW_DOP_GEMV_GLU_MXFP4",
    "PLOW_DOP_GEMV_MXFP4",
    "PLOW_DOP_GEMV_QKV",
    "PLOW_DOP_GEMV_QKVG",
    "PLOW_DOP_GEMV_QKV_MXFP4",
    "PLOW_DOP_GLU",
    "PLOW_DOP_HEADNORM_ROPE",
    "PLOW_DOP_HEADNORM_ROPE_FP8",
    "PLOW_DOP_INDEX_SCORE",
    "PLOW_DOP_INDEX_SELECT",
    "PLOW_DOP_KDA_CONV",
    "PLOW_DOP_KDA_CONV3",
    "PLOW_DOP_KDA_GATE",
    "PLOW_DOP_KDA_GATED_NORM",
    "PLOW_DOP_KDA_STATE_STEP",
    "PLOW_DOP_KDA_STATE_STEP_G",
    "PLOW_DOP_LAYERNORM",
    "PLOW_DOP_MLA_MERGE_FOLD",
    "PLOW_DOP_MLA_OUT_GATE",
    "PLOW_DOP_MOE_ALIGN_PF",
    "PLOW_DOP_MOE_COMBINE",
    "PLOW_DOP_MOE_COMBINE_PF",
    "PLOW_DOP_MOE_EXPERT_DOWN",
    "PLOW_DOP_MOE_EXPERT_DOWN_FP8_BLK",
    "PLOW_DOP_MOE_EXPERT_GLU",
    "PLOW_DOP_MOE_EXPERT_GLU_FP8_BLK",
    "PLOW_DOP_MOE_GROUP_DOWN_FP8_BLK",
    "PLOW_DOP_MOE_GROUP_DOWN_PF",
    "PLOW_DOP_MOE_GROUP_GLU_FP8_BLK",
    "PLOW_DOP_MOE_GROUP_GLU_PF",
    "PLOW_DOP_MOE_ROUTER",
    "PLOW_DOP_MOE_ROUTER_TOPK",
    "PLOW_DOP_MOE_ROUTER_TOPK_PF",
    "PLOW_DOP_NORM_RESIDUAL",
    "PLOW_DOP_NORM_RESIDUAL_NORM",
    "PLOW_DOP_O_UV_FOLD",
    "PLOW_DOP_QUANT_FP8",
    "PLOW_DOP_RESIDUAL",
    "PLOW_DOP_RMSNORM",
    "PLOW_DOP_ROWRMS",
    "PLOW_DOP_SITU_GLU",
    "PLOW_DOP_SOFTCAP",
    "PLOW_DOP_XARGMAX_FIN",
    "PLOW_DOP_XFLASHMERGE",
    "PLOW_DOP_XREDUCE",
    "PLOW_DOP_XREDUCE2",
];

/// Refuse a packet carrying an opcode the gfx950 interpreter has no arm for.
///
/// Coarse ON PURPOSE: "is there a `case` anywhere in interp.hip", not "is there one under the
/// defines this packet's manifest asks for". The finer question needs the cmake table, which
/// `scripts/gfx950_objects.py --cover` parses — restating that table here is the drift this file
/// exists to prevent. The coarse check is what catches an opcode with NO implementation, which is
/// the failure that produces silent zeros.
///
/// ONE DIRECTION ONLY, and that was a hole. This asks *emitted opcode ⇒ arm exists*. It cannot ask
/// *arm exists ⇒ something emits it*, because it sees ONE packet built under ONE flag combination —
/// an arm nothing routes to looks exactly like an arm this particular packet did not need. That is
/// why `PLOW_MXFP4=1` on a dense model shipped a bf16 packet for as long as it did with every gate
/// in this file green. The reverse direction is therefore a TEST over the sources, not a runtime
/// check: see `gfx950_coverage_tests::every_dispatched_arm_has_an_emit_site` and
/// `precision_knob_table_matches_the_emitters`.
/// Refuse a group-limited-routing packet on a target whose interpreter routes flat.
///
/// GROUP-LIMITED ROUTING (DeepSeek `noaux_tc`, and Kimi-K3) partitions the experts into `n_group`
/// contiguous groups, keeps the top `topk_group` of them by summed top-2 biased score, and runs
/// the top-k only inside those. The rule is carried on the router instruction as `i[6] = n_group`
/// and `i[7] = topk_group`, and `moe_group_mask` (`runtime/amd/op_moe.h:271`) implements it.
///
/// ONLY THE AMD INTERPRETER HAS IT. `runtime/nvidia/` and `runtime/cpu/` read neither operand —
/// they route flat top-k over all experts. So the SAME packet computes a different model on a
/// different backend, and nothing on either side says so: there is no missing opcode for
/// `check_gfx950_opcode_coverage`'s cousin to catch, because the op exists everywhere and only
/// its operands are ignored. The divergence surfaces as an accuracy gap between an AMD and an
/// NVIDIA rank, or against the CPU golden reference used to validate the kernel — and it looks
/// like a numerics bug rather than a missing feature, which is the expensive way to find it.
///
/// Refused rather than ignored, by the rule `warn_uniseg_on_amd` states: ignoring a flag is
/// acceptable when the caller still gets the CORRECT packet, and here they would not.
///
/// Inert for every model shipping today — `n_group <= 1` (GLM-5.2, Gemma-4, Qwen) and
/// `topk_group >= n_group` are both the identity, and pre-existing blobs carry `i[6] = 0`.
fn check_group_routing_supported(m: &Model, amd: bool, arch: &str) {
    if amd {
        return;
    }
    let grouped = m.progs.iter().flat_map(|p| p.insts.iter()).find(|i| {
        (i.op == DevOp::MoeRouterTopk as u16 || i.op == DevOp::MoeRouterTopkPf as u16)
            && i.i[6] > 1
            && i.i[7] < i.i[6]
    });
    if let Some(i) = grouped {
        panic!(
            "group-limited MoE routing (n_group = {}, topk_group = {}) is not implemented for \
             {arch}. Missing capability: `moe_group_limited_routing` outside gfx950 — \
             `moe_group_mask` lives only in runtime/amd/op_moe.h, and the NVIDIA and CPU \
             interpreters ignore i[6]/i[7] and route FLAT top-{} over all experts. That is a \
             different expert set, so the same packet would compute a different model here than \
             it does on AMD, with no fault and no diagnostic on either side. Emit this model for \
             a gfx950 target, or implement the group mask in the target's router first.",
            i.i[6], i.i[7], i.i[2]
        );
    }
}

fn check_gfx950_opcode_coverage(m: &Model, amd: bool) {
    if !amd {
        return;
    }
    let mut missing: Vec<&'static str> = Vec::new();
    for p in &m.progs {
        for inst in &p.insts {
            let Some(op) = DevOp::ALL.iter().copied().find(|o| *o as u16 == inst.op) else {
                continue;
            };
            let c = op.c_name();
            if !GFX950_DISPATCHED.contains(&c) && !missing.contains(&c) {
                missing.push(c);
            }
        }
    }
    assert!(
        missing.is_empty(),
        "this packet carries {} opcode(s) the gfx950 interpreter has no arm for: {missing:?}. \
         Missing capability: `gfx950_opcode_arm`. AMD's dispatch default writes NOTHING rather than \
         trapping, so such a packet would RUN and be silently wrong — an untouched output buffer \
         read as a result. Either emit the arm on the AMD side or stop emitting the opcode (check \
         which env flag selected it; several are sm_120-conditioned and ungated).",
        missing.len()
    );
}

/// Warn when `PLOW_UNISEG` is set for an AMD target, where it is ignored.
///
/// THE THIRD sm_120-conditioned flag found with no arch gate, and the one that did the most damage:
/// the documented Gemma recipe passed it, so every asset built by following the documentation lost
/// AMD's wave-class split and produced zero logits in an 8.7 ms "prefill". See
/// [`packet::devbuild::Builder::deny_uniseg`] for the mechanism.
///
/// Ignored rather than refused, by the same rule as `PLOW_L2_PLACE`: ignoring it yields the CORRECT
/// packet, so the thing being dropped is not load-bearing for the result. Contrast w8a16-on-gfx950,
/// where ignoring the flag would have produced a WRONG packet and refusing was the only honest
/// option. The test is always "what does the caller get if I drop this?", not "how important does
/// the flag sound".
fn warn_uniseg_on_amd(amd: bool) {
    if amd && std::env::var("PLOW_UNISEG").ok().as_deref() == Some("1") {
        eprintln!(
            "  PLOW_UNISEG ignored: it collapses every op into ONE segment, which is spurious on \
             sm_120 but destroys the wave-class split an AMD host relaunches on — the whole prefill \
             program would be dispatched on the 4-wave flash object, which silently drops every op \
             that is not a flash. Emitting with wave-class segmentation instead."
        );
    }
}

/// Warn when `--arch` and `--gpu` name different vendors.
///
/// The manifest's whole job is to say what object a packet needs, and `arch` is what a backend
/// renders flags for — so an sm_120a manifest emitted for an MI350X describes a build for the wrong
/// TOOLCHAIN, not merely the wrong tuning. `build-amd/g31b-fp8` and `g31b-fp8kv` were both emitted
/// that way and then run on gfx950; the mismatch is how an sm_120-only w8a16 profile ended up in an
/// AMD build directory at all.
///
/// A warning rather than a refusal: emitting a manifest for a different target than the sizing GPU
/// is legitimate when cross-compiling, and `--n-cu` can be given explicitly. But it must never
/// happen SILENTLY, because the failure it produces — an object built with the wrong defines, or
/// with no arm for an opcode at all — surfaces as a fault at first launch with nothing pointing
/// back here.
fn warn_arch_gpu_vendor_mismatch(arch: &str, gpu: &str) {
    let Some(spec) = hwspec::registry::lookup(gpu) else {
        return;
    };
    let arch_amd = arch.starts_with("gfx");
    let arch_nv = arch.starts_with("sm_");
    let gpu_amd = spec.vendor == hwspec::Vendor::Amd;
    if (arch_amd && !gpu_amd) || (arch_nv && gpu_amd) {
        eprintln!(
            "  WARNING: --arch {arch} and --gpu {gpu} name different vendors. build.json will \
             describe an object for {arch}, but the packet is sized for {gpu} ({} CUs). If this is \
             not a deliberate cross-compile then one of the two is wrong — this exact mismatch is \
             how two w8a16 assets ended up in an AMD build directory and faulted on first launch.",
            spec.sm_count
        );
    }
}

/// Refuse a packet whose fp8 prefill GEMMs have no activation scale, when the target is gfx950.
///
/// `PLOW_FP8=1` alone emits **w8a16**: a bf16 activation in `t[1]` and `t[3]` (a_scale) left
/// `TENSOR_NONE`. That is a real profile — the sm_120 build has a w8a16 cubin for it. gfx950 does
/// not: `d_gemm_fp8` is **w8a8** unconditionally, it casts `t[1]` to `unsigned char*` and its
/// epilogue computes `acc * ascale[mm] * wscale[nn]` with NO null check
/// (`runtime/amd/op_gemm.h`). So a w8a16 packet on gfx950 dereferences a null pointer on the first
/// prefill GEMM — a GPU fault with no diagnostic, from a flag combination the README documented.
///
/// REFUSE rather than silently upgrade to w8a8. The two are not the same computation: w8a8
/// quantizes the ACTIVATIONS as well, so auto-substituting would change the numerics of a run under
/// a flag the caller set to mean something else, and any accuracy result from it would be
/// attributed to the wrong profile. Every substitution this emitter makes on its own —
/// `pick_tile`'s tile choice, say — is computation-preserving; this one would not be. The fix is
/// one flag and the message names it.
///
/// Derived from the EMITTED STREAM rather than from the flags, for the reason `manifest.rs` states
/// at length: an emitter flag says what was asked for, the stream says what the packet contains,
/// and only the second is what the object has to run. It also means the gate cannot drift if the
/// fp8 branching is ever restructured.
fn check_fp8_a_scale_bound(m: &Model, arch: &str, gpu: &str) {
    // WHICH TARGET, and why `arch` alone is not enough to ask.
    //
    // `--arch` records the ISA the manifest is written FOR; `--gpu` records the part the packet is
    // sized for. They can disagree, and when they do the arch string is the less trustworthy of the
    // two — the assets that motivated this gate are exactly that case: `build-amd/g31b-fp8kv` was
    // emitted `--arch sm_120a` (where w8a16 is a real profile) with `n_cu = 256` (an MI350X), and
    // then run on gfx950, where it faulted. An arch-only gate would have let it straight through.
    //
    // So: AMD if EITHER signal says AMD. A false positive costs an sm_120 user one flag and a clear
    // message; a false negative is the null dereference this exists to stop.
    let gpu_is_amd = hwspec::registry::lookup(gpu).is_some_and(|s| s.vendor == hwspec::Vendor::Amd);
    if !arch.starts_with("gfx") && !gpu_is_amd {
        return;
    }
    // DERIVED FROM `GFX950_RUNGS`, not restated. A hand-written list of the fp8 opcodes is the
    // one part of this gate that COULD drift, and it did: the tile-inventory campaign added the
    // 128x256 (`GemmWideFp8`) and 192x256 (`GemmC5Fp8`) rungs to `GFX950_RUNGS` — so `pick_tile`
    // began emitting them — while this closure still named only the original three. The gate went
    // quiet on exactly the two shapes the rungs were added for. `manifest.rs`'s `FP8_WEIGHT_OPS`
    // was updated in the same change and this was not, which is what made the omission invisible.
    //
    // `GFX950_RUNGS` is the table `pick_tile` selects from, so reading the fp8 column of it asks
    // the same question the emitter answers. A sixth rung is now covered by construction.
    let fp8_gemm = |op: u16| {
        GFX950_RUNGS
            .iter()
            .any(|(_, fp8, _, _, _, _)| *fp8 as u16 == op)
    };
    let bad = m
        .progs
        .iter()
        .flat_map(|p| p.insts.iter())
        .find(|i| fp8_gemm(i.op) && i.t[3] == TENSOR_NONE);
    if let Some(i) = bad {
        panic!(
            "fp8 prefill GEMM (op {}) has no activation scale (t[3] = TENSOR_NONE) and the target \
             is {arch}. Missing capability: `fp8_w8a16_prefill` on gfx950 — `d_gemm_fp8` there is \
             w8a8 unconditionally: it reads t[1] as e4m3 bytes and its epilogue dereferences \
             ascale[m] with no null check, so this packet would fault on its first prefill GEMM. \
             PLOW_FP8=1 alone emits w8a16, which only the sm_120 objects implement. Use \
             `PLOW_FP8=1 PLOW_W8A8=1` for fp8 on gfx950 (it emits QuantFp8 and binds t[3]), or \
             build for an sm_120 target. NOT auto-upgraded: w8a8 quantizes activations too, so it \
             is a different computation and would silently change what a run measures.",
            i.op
        );
    }
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
    l2_layout: Option<packet::devbuild::L2Layout>,
    gpu: String,
    arch: String,
    verify: Option<VerifyHook>,
) {
    // Empty --gpu ⇒ unknown target (0), not fnv("") — so the header stamp is 0
    // and unspecified-GPU blobs stay byte-stable (e.g. the golden test).
    let target_fp = if gpu.is_empty() {
        0
    } else {
        packet::devbuild::gpu_fingerprint(&gpu)
    };
    // Resolved ONCE, at the top, because several decisions below depend on it — the prefill bucket
    // ladder, the lm_head arm, `deny_uniseg`, and the opcode-coverage gate. Recomputing it per site
    // is how two of the three ungated sm_120 flags stayed ungated. Same predicate `run_verified`
    // uses; this function is also reached directly, so it does not inherit that one.
    let amd = target_is_amd(&arch, &gpu);
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
    // ===== THE FOUR PRECISION AXES =====================================================
    //
    // A packet's precision is FOUR independent choices — weight, activation, KV cache, experts —
    // and the flags did not say so. `PLOW_FP8` alone meant "fp8 weights", but the ACTIVATION axis
    // had no flag at all: it was decided by PHASE inside the weight axis (prefill w8a8, decode
    // w8a16), which is why a w8a16 packet could reach gfx950 with no way for the caller to say
    // otherwise, and why `check_fp8_a_scale_bound` has to exist. A distinction that cannot be
    // EXPRESSED cannot be checked at the point of request; it can only be caught afterwards.
    //
    // So the axes get names. `PLOW_FP8` stays an alias for the weight axis — every existing script
    // and every line of the README keeps working, and unset means unset:
    //
    //   weight      PLOW_W8A16=1 | PLOW_W8A8=1 | PLOW_W4A16=1   (alias: PLOW_FP8=1 -> w8a16)
    //   activation  implied by the weight axis (w8a8 quantizes activations; the others do not)
    //   kv          PLOW_KV_FP8=1                               (alias: PLOW_FP8_KV=1)
    //   experts     PLOW_MOE_ENC=bf16|fp8blk|mxfp4              (MLA family; see mla::MoeEnc)
    //
    // The activation axis is deliberately NOT a separate flag: w8a8 is one profile, not a free
    // cross-product with w8a16, and the kernels instantiate exactly those. Naming it separately
    // would invent combinations no object implements — the opposite of the problem being fixed.
    //
    // The refusal gate stays regardless of naming. Names make the distinction expressible at the
    // point of request; the gate is derived from the emitted STREAM, so it is the thing that cannot
    // drift when the flags are restructured again.
    let fp8 = std::env::var("PLOW_FP8").ok().as_deref() == Some("1")
        || std::env::var("PLOW_W8A16").ok().as_deref() == Some("1")
        || std::env::var("PLOW_W8A8").ok().as_deref() == Some("1");
    // T8 w8a8 (PLOW_W8A8=1, requires PLOW_FP8=1). PREFILL emits the true fp8 tensor-core path:
    // ONE per-row DevOp::QuantFp8 per activation site + GEMM_FP8/GEMM_GLU_FP8 re-pointed at the
    // fp8 activation (t1=xq) + a_scale (t3). The SAME opcodes serve T6 w8a16 (bf16 activation) —
    // the interp cubin selects the kernel by PLOW_NV_W8A8, so the w8a8 pkt MUST run against a
    // PLOW_NV_W8A8=1 prefill cubin (the T6 cubin would misread xq bytes as bf16). Weight side =
    // the same e4m3 twins + per-channel scales T6 declared. Unset => byte-identical emission.
    let w8a8 = std::env::var("PLOW_W8A8").ok().as_deref() == Some("1");
    assert!(
        !(w8a8 && std::env::var("PLOW_W8A16").ok().as_deref() == Some("1")),
        "PLOW_W8A8=1 and PLOW_W8A16=1 name two activation profiles on one weight axis; pick one"
    );
    assert!(
        !w8a8 || fp8,
        "PLOW_W8A8=1 requires PLOW_FP8=1 (the fp8 weight twins + scales)"
    );
    // WEIGHT AXIS, 4-bit (`PLOW_MXFP4=1`) — REFUSED here, not ignored.
    //
    // The flag is real and it works, on the OTHER family: `mla::mla_moe_enc_env` reads it and
    // returns `MoeEnc::Mxfp4`, and `scripts/build_gfx950.sh` builds and ships
    // `interp_{decode,prefill}_mxfp4[_gq].elf` when it is set. This emitter never read it. So
    // `PLOW_MXFP4=1` on Gemma/Qwen/Llama emitted a packet BYTE-IDENTICAL to the bf16 one, with
    // `build.json` reporting `mxfp4_weights: false` and not one `Gemv*Mxfp4` opcode in the stream —
    // next to a build directory full of objects named `mxfp4`. §4's recurring shape exactly: the
    // arm exists (`DevOp::GemvMxfp4`/`GemvGluMxfp4`/`GemmMxfp4`, 91/92/93, all three dispatched by
    // the gfx950 interpreter and all three in `GFX950_DISPATCHED`), it is correct, it is
    // register-gated — and on this path nothing routes to it.
    //
    // REFUSED rather than warned, by `warn_uniseg_on_amd`'s own test: "what does the caller get if
    // I drop this?" Dropping `PLOW_UNISEG` yields the CORRECT packet, so it is ignored with a
    // warning. Dropping `PLOW_MXFP4` yields a bf16 packet that the caller asked to be mxfp4, will
    // benchmark, and will report as mxfp4 — the precision substitution the apples-to-apples rule
    // exists to prevent, and the one failure mode this file's gates are all pointed at. That puts
    // it with w8a16-on-gfx950 (`check_fp8_a_scale_bound`), not with UNISEG. It also removes a
    // loudness asymmetry that was itself the defect: `PLOW_FP8=1` without `PLOW_W8A8` on gfx950
    // panics with a four-line explanation, while the 4-bit axis said nothing at all.
    //
    // Missing capability: `dense_mxfp4_weights`. NOT implemented rather than not wanted, and the
    // reason is measurement: mxfp4 projects only ~1.2x over w8a8 once the fixed per-packet overhead
    // is held constant, not the 2x its 4-bit weight implies, because that overhead does not shrink
    // with the weights. Building it would mean declaring e2m1 weight twins + one E8M0 scale per 32
    // K-elements per projection (bias 127 — byte 0 is 2^-127, not neutral), pointing decode at ops
    // 91/92 and prefill at 93, and adding the `mxfp4` object row to the manifest's `requires`.
    assert!(
        std::env::var("PLOW_MXFP4").ok().as_deref() != Some("1"),
        "PLOW_MXFP4=1 is not implementable on the dense-GQA family (Gemma / Qwen / Llama): this \
         emitter has no mxfp4 arm, so it would emit a packet byte-identical to bf16 while the \
         objects, the filename and the manifest all said mxfp4. Missing capability: \
         `dense_mxfp4_weights`. It IS implemented for the MLA/MoE family (GLM / Kimi / DeepSeek), \
         which is what `scripts/build_gfx950.sh`'s mxfp4 objects are for. NOT silently downgraded \
         to bf16: a bf16 asset labelled mxfp4 is the precision substitution that makes a \
         measurement unfalsifiable. Use PLOW_FP8=1 PLOW_W8A8=1 for the narrowest weight profile \
         this family has on gfx950, or unset PLOW_MXFP4 for bf16."
    );
    // FP8 KV-CACHE (PLOW_FP8_KV=1). Stores/reads K/V as e4m3 with a per-row f32 scale, halving the
    // decode KV stream (the HBM-bound part of flash-decode) and the KV footprint. Independent of the
    // fp8 WEIGHT path above so both can be A/B'd; the harness routes an fp8-KV pkt to the _fp8kv
    // interpreter objects (which carry the fp8 flash + HeadNormRopeFp8 arms).
    // KV axis. `PLOW_KV_FP8` is the axis-named spelling; `PLOW_FP8_KV` is the historical one and
    // stays an alias. This axis is INDEPENDENT of the weight axis — bf16 weights with an fp8 KV
    // cache is a real and useful profile (NVIDIA has always been able to build it), and the AMD
    // object matrix conflated the two until recently.
    let fp8_kv = std::env::var("PLOW_FP8_KV").ok().as_deref() == Some("1")
        || std::env::var("PLOW_KV_FP8").ok().as_deref() == Some("1");
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
    let shipped: Vec<u32> = [128u32, 512, 1024, 2048, 4096, 8192]
        .into_iter()
        .filter(|&x| x <= cap)
        .collect();
    // PLOW_PF_LADDER is sm_120-only, and its own comment above says why: the rungs are derived from
    // the 128x128 sm_120 tile, and "the AMD bodies tile differently, so the ladder is only derived
    // on the NVIDIA path; gate it on the flag rather than guessing the target's tile from here."
    // The target is now known here, so gate on the TARGET instead of hoping nobody sets the flag.
    // Unlike the other three this one only mis-TUNES the rungs — no opcode moves and nothing is
    // dropped — so it is ignored quietly on AMD rather than warned about at every emit.
    let ladder_wave = !amd && std::env::var("PLOW_PF_LADDER").ok().as_deref() == Some("wave");
    let buckets: Vec<u32> = if ladder_wave {
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
    let moe_pf =
        c.moe && (!fp8 || w8a8) && std::env::var("PLOW_MOE_PREFILL").ok().as_deref() != Some("0");

    // Phase 1 (plans/devgen-trait-refactor.md): the DenseGqaEmitter owns the dense
    // tensor declaration (declare) and the emit_phase call sites. Byte-identical —
    // `new` forwards to the same `declare`, `emit_*` to the same `emit_phase`.
    let (emitter, tensors, gen) = DenseGqaEmitter::new(
        &c,
        &ls,
        n_cu,
        ctx,
        fp8,
        w8a8,
        fp8_kv,
        fp8_kv_full,
        block.clone(),
        block_mode,
        ns_pre,
        dbatch,
        moe_pf,
        amd,
    );

    let mut progs = Vec::new();
    let mut tlist = Vec::new();
    for &t in &buckets {
        if c.moe && !moe_pf {
            break;
        } // MoE without prefill: decode-only blob
        let mut b = Builder::new(n_cu);
        b.adopt_tensors(tensors.clone());
        b.set_l2_placement(l2_layout); // PLOW_L2_PLACE: None ⇒ byte-identical
        if amd {
            b.deny_uniseg(); // PLOW_UNISEG collapses the wave-class split — see `warn_uniseg_amd`
        }
        emitter.emit_prefill(&mut b, t);
        progs.push(b.finish());
        tlist.push(t);
    }
    let mut bd = Builder::new(n_cu);
    bd.adopt_tensors(tensors.clone());
    bd.set_l2_placement(l2_layout); // PLOW_L2_PLACE: None ⇒ byte-identical
    if amd {
        bd.deny_uniseg();
    }
    bd.set_gemv_split(gemv_split()); // PLOW_GEMV_SPLIT: 1 (default) ⇒ byte-identical
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
    //
    // THE PANIC IS FOR A REJECTION ONLY. The hook takes `&Model`, so it cannot
    // change what gets serialized; and it is the hook's job — not this call
    // site's — to downgrade "no usable verifier here" into an `Ok` carrying a
    // skip reason. Anything that reaches this `Err` is the verifier saying the
    // program is wrong, i.e. a real bug caught, and must be loud.
    let lean = apply_verify_gate(&m, verify.as_ref());
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
        // The dense path declares every weight it reads; nothing reaches the device by a
        // name-pattern bind, a load-time fold or a host-side absorption, and nothing it
        // declares is synthesized before the bind, and no weight is conditionally covered.
        &[],
        &[],
        &[],
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
    // GATE: an fp8 GEMM whose a_scale operand is unbound is a NULL DEREFERENCE on gfx950. Checked
    // against the emitted stream and BEFORE the blob is written, so the bad packet never exists.
    check_fp8_a_scale_bound(&m, &arch, &gpu);
    check_gfx950_opcode_coverage(&m, amd);
    check_group_routing_supported(&m, amd, &arch);
    warn_arch_gpu_vendor_mismatch(&arch, &gpu);

    std::fs::write(&out, blob).unwrap();

    // BUILD MANIFEST (`build.json`, beside the .pkt). Derived from `m` — the exact
    // programs just serialized — so it cannot describe a packet other than the one
    // on disk. That is the whole point: the packet and the interpreter object were
    // two independent sources of truth, and every failure in the rtx-2x campaign
    // came out of the gap between them. See crates/devgen/src/manifest.rs.
    // Skipped when `arch` is empty (the legacy `gemma4` CLI), so that path's output
    // is unchanged.
    if !arch.is_empty() {
        let man = manifest::build(&m, &arch, &lean);
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
        // Same predicate the loaders bind on (`packet::names`), so the emitter's reported
        // "weights N GiB" is exactly the byte count the runtime will demand of the checkpoint.
        // Under the old prefix allowlist an untied `lm_head.weight` was reported as
        // activations here AND zeroed by the CUDA loader — the report agreed with the bug.
        .filter(|x| packet::names::is_checkpoint_weight(&x.name))
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

    fn router_program_b(split_plan: Option<(u32, DevOp)>, nrow: u32) -> packet::devbuild::Program {
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
mod gfx950_coverage_tests {
    //! The AMD opcode-coverage gate, and the drift test that keeps its list honest.
    use super::*;

    /// `MOE_MAX_TOPK` must equal `PLOW_MOE_MAX_TOPK` — PARSED out of `op_moe.h`, not restated.
    ///
    /// The two halves of this bound live in different languages and different repos-worth of
    /// build system, and only one of them refuses. If the Rust constant drifts HIGH the emit
    /// happily produces a packet the kernel truncates silently; if it drifts LOW the emit refuses
    /// a model that would have run. Neither shows up in any other test.
    #[test]
    fn moe_topk_matches_the_amd_kernel() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("runtime/amd/op_moe.h"));
        let Some(path) = root.filter(|p| p.exists()) else {
            eprintln!("op_moe.h not found — skipping (source checkout only)");
            return;
        };
        let src = std::fs::read_to_string(&path).unwrap();
        let def = src
            .lines()
            .find_map(|l| l.trim().strip_prefix("#define PLOW_MOE_MAX_TOPK "))
            .expect("op_moe.h has no `#define PLOW_MOE_MAX_TOPK`");
        let got: u32 = def
            .trim()
            .trim_end_matches('u')
            .parse()
            .unwrap_or_else(|e| panic!("cannot parse PLOW_MOE_MAX_TOPK {def:?}: {e}"));
        assert_eq!(
            got, MOE_MAX_TOPK,
            "devgen::MOE_MAX_TOPK ({MOE_MAX_TOPK}) disagrees with op_moe.h's \
             PLOW_MOE_MAX_TOPK ({got}). Raise them together: the kernel bound is what the routers \
             can select into, the Rust one is what refuses to emit past it."
        );
    }

    /// The refusal must name the model AND the limit, and must not fire at or below the bound.
    /// A gate that refuses everything is as useless as one that refuses nothing.
    #[test]
    fn moe_topk_refusal_is_a_threshold_and_names_the_model() {
        for k in 1..=MOE_MAX_TOPK {
            require_moe_topk(k, "in-bounds");
        }
        // Expressed against the constant, not a literal: this test must keep meaning the same
        // thing the next time the bound moves.
        let over = MOE_MAX_TOPK + 1;
        let err = std::panic::catch_unwind(|| require_moe_topk(over, "kimi_k3")).unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or("");
        assert!(
            msg.contains("kimi_k3"),
            "refusal must name the model: {msg}"
        );
        assert!(
            msg.contains("PLOW_MOE_MAX_TOPK"),
            "must name the limit: {msg}"
        );
        assert!(
            msg.contains("uninitialised"),
            "must say what goes wrong: {msg}"
        );
    }

    /// The list must equal what `interp.hip` actually dispatches — PARSED, not restated. A
    /// hand-maintained copy of a fact in another file is the drift `manifest.rs` was written to
    /// stop; this is the same discipline `packet`'s `dev_abi` test applies to `dev_isa.h`.
    ///
    /// Two dispatch FORMS, and missing the second would produce false refusals: most opcodes are
    /// `case PLOW_DOP_X:` in the switch, but the TP collectives are handled by `if (in->op == ...)`
    /// before it. A `case`-only parse undercounts them — which `docs/amd/model-op-coverage.md`
    /// warns about in its opening lines.
    #[test]
    fn dispatched_list_matches_the_amd_interpreter() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("runtime/amd/interp.hip"));
        let Some(path) = root.filter(|p| p.exists()) else {
            eprintln!("interp.hip not found — skipping (source checkout only)");
            return;
        };
        let src = std::fs::read_to_string(&path).unwrap();
        let mut found: Vec<String> = Vec::new();
        for line in src.lines() {
            let t = line.trim();
            // `case PLOW_DOP_X:` — the switch arms.
            if let Some(r) = t.strip_prefix("case PLOW_DOP_") {
                let name: String = r
                    .chars()
                    .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                    .collect();
                found.push(format!("PLOW_DOP_{name}"));
            }
            // `in->op == PLOW_DOP_X` — the collectives, dispatched ahead of the switch.
            let mut rest = t;
            while let Some(i) = rest.find("op == PLOW_DOP_") {
                let r = &rest[i + "op == PLOW_DOP_".len()..];
                let name: String = r
                    .chars()
                    .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                    .collect();
                found.push(format!("PLOW_DOP_{name}"));
                rest = &r[name.len()..];
            }
        }
        found.sort();
        found.dedup();
        let mut want: Vec<String> = GFX950_DISPATCHED.iter().map(|s| s.to_string()).collect();
        want.sort();
        let missing_here: Vec<&String> = found.iter().filter(|n| !want.contains(n)).collect();
        let stale: Vec<&String> = want.iter().filter(|n| !found.contains(n)).collect();
        assert!(
            missing_here.is_empty() && stale.is_empty(),
            "GFX950_DISPATCHED disagrees with interp.hip.\n  interp has, list lacks: {missing_here:?}\n  \
             list has, interp lacks: {stale:?}\nAn over-long list lets a packet through that will \
             silently write nothing; a short one refuses a packet that would have run."
        );
    }

    /// The gate refuses an opcode with no AMD arm. `GemvArgmax` is the real instance: it has no
    /// `case` in interp.hip, and `PLOW_FUSE_ARGMAX=1` emits it — a decode would have argmaxed over
    /// an untouched buffer and returned token 0 every step, with no fault anywhere.
    #[test]
    #[should_panic(expected = "gfx950_opcode_arm")]
    fn opcode_with_no_amd_arm_is_refused() {
        assert!(
            !GFX950_DISPATCHED.contains(&DevOp::GemvArgmax.c_name()),
            "if AMD ever gains a GEMV_ARGMAX arm this test should be re-pointed, not deleted"
        );
        let i = packet::dev::DevInst {
            op: DevOp::GemvArgmax as u16,
            blocks: 1,
            ..Default::default()
        };
        let p = packet::devbuild::Program {
            hier_base: 0,
            n_cu: 4,
            n_counter: 0,
            insts: vec![i],
            stream: vec![],
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            tensors: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_sms: 0,
            l2_domains: 0,
        };
        let m = Model {
            n_cu: 256,
            target: 0,
            tensors: vec![],
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        };
        check_gfx950_opcode_coverage(&m, true);
    }

    /// …and lets an ordinary packet through, on both targets.
    #[test]
    fn covered_opcodes_pass_and_nvidia_is_never_checked() {
        let p = packet::devbuild::Program {
            hier_base: 0,
            n_cu: 4,
            n_counter: 0,
            insts: vec![
                packet::dev::DevInst {
                    op: DevOp::Gemv as u16,
                    blocks: 1,
                    ..Default::default()
                },
                packet::dev::DevInst {
                    op: DevOp::XReduce as u16,
                    blocks: 1,
                    ..Default::default()
                },
            ],
            stream: vec![],
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            tensors: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_sms: 0,
            l2_domains: 0,
        };
        let m = Model {
            n_cu: 256,
            target: 0,
            tensors: vec![],
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        };
        check_gfx950_opcode_coverage(&m, true);
        // The Gemma-MoE family has no AMD arm at all; on an NVIDIA target that must not be checked.
        let p2 = packet::devbuild::Program {
            hier_base: 0,
            n_cu: 4,
            n_counter: 0,
            insts: vec![packet::dev::DevInst {
                op: DevOp::MoeRouterGemma as u16,
                blocks: 1,
                ..Default::default()
            }],
            stream: vec![],
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            tensors: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_sms: 0,
            l2_domains: 0,
        };
        let m2 = Model {
            n_cu: 170,
            target: 0,
            tensors: vec![],
            progs: vec![p2],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        };
        check_gfx950_opcode_coverage(&m2, false);
    }

    // ===== THE REVERSE DIRECTION ==================================================================
    //
    // Everything above asks "the packet carries opcode X — does gfx950 have an arm?". Neither of
    // these does, and that asymmetry is the single most-repeated bug in this tree (~10 instances):
    //
    //     an arm exists, is correct, is register-gated, and NOTHING ROUTES TO IT.
    //
    // The runtime gate is structurally unable to see it. It inspects one emitted packet, built
    // under one flag combination, so "no instruction selected this arm" and "this program did not
    // need this arm" are the same observation. The reverse question is about the SOURCE — is there
    // any reachable emit path at all — so it is asked here, against the source, exactly as
    // `dispatched_list_matches_the_amd_interpreter` asks its question against `interp.hip`.
    //
    // Two checks, because "unreachable" has two shapes, and the expensive one is the second:
    //   A. NO emit site anywhere. `every_dispatched_arm_has_an_emit_site`.
    //   B. an emit site exists on ONE emitter family, and the flag that selects it is not read by
    //      the others, so those silently emit the default precision. `PLOW_MXFP4` was this:
    //      `DevOp::GemvMxfp4` is emitted (by `mla.rs`), so check A is green, and a dense model with
    //      `PLOW_MXFP4=1` still produced a byte-identical-to-bf16 packet.
    //      `precision_knob_table_matches_the_emitters`.

    /// The emitter files. `manifest.rs` is EXCLUDED even though it names opcodes: it classifies a
    /// finished stream (`DevOp::GemvMxfp4 => s.mxfp4_proj = true`), so counting it as an emit site
    /// would let a reporter vouch for an arm no emitter reaches — the exact confusion this checks.
    const EMITTER_SRC: &[&str] = &[
        "lib.rs",
        "mla.rs",
        "block.rs",
        "ladder.rs",
        "kda.rs",
        "k3.rs",
    ];

    /// Arms gfx950 dispatches that NOTHING emits, each with why that is deliberate.
    ///
    /// An allowlist, not a suppression: the justification is required, and an arm that leaves this
    /// list without gaining an emit site fails the test. Adding a row is the moment to ask §4's
    /// question — "what selects this, and is that selector complete over precisions?"
    const GFX950_UNEMITTED: &[(&str, &str)] = &[
        ("PLOW_DOP_ATTN_SELECT",
         "DeepSeek DSA on-device top-k KV selection. The DSA path that ships emits IndexScore(58) \
          + IndexSelect(59) instead; ATTN_SELECT is the single-kernel alternative and is kept \
          because the two are being compared. Not a precision arm — no silent-wrong risk."),
        ("PLOW_DOP_O_UV_FOLD",
         "SUPERSEDED by MlaMergeFold(60), which fuses FlashMerge<512> + O_UV_FOLD into one packet \
          (mla.rs:1595 states the substitution). The arm stays for the unfused A/B."),
        ("PLOW_DOP_ROWRMS",
         "Precomputed row-RMS feeding GemmNorm's norm=1 mode (op_gemm.h:1086). Every emitter uses \
          the fused norm path, which needs no separate RMS packet."),
        ("PLOW_DOP_XFLASHMERGE",
         "CONTEXT-PARALLEL cross-rank LSE merge (plans/tp-design.md §8c). TP shards attention by \
          whole heads, so no rank holds a partial for another rank's head and there is nothing to \
          merge. Emitted only once CP exists; blocked on S4."),
    ];

    /// CHECK A — every arm gfx950 dispatches is either emitted by some emitter, or allowlisted with
    /// a reason.
    ///
    /// Deliberately coarse in the same way its forward twin is: "does any emitter file name this
    /// `DevOp`", not "is that site reachable under the flags this build sets". A reachability
    /// analysis would need the flag cross-product, and a wrong one would fail builds that work.
    /// Naming is the cheap 90%: it catches an opcode added to `dev_isa.h` + `interp.hip` +
    /// `GFX950_DISPATCHED` and then never wired, which is how five of the ~10 instances happened.
    #[test]
    fn every_dispatched_arm_has_an_emit_site() {
        let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut named: std::collections::BTreeSet<String> = Default::default();
        for f in EMITTER_SRC {
            let Ok(text) = std::fs::read_to_string(src_dir.join(f)) else {
                continue;
            };
            for line in text.lines() {
                // Comments are not emit sites. Without this, the paragraph explaining WHY an arm is
                // unwired would itself satisfy the check.
                let t = line.trim_start();
                if t.starts_with("//") {
                    continue;
                }
                let mut rest = line;
                while let Some(i) = rest.find("DevOp::") {
                    let r = &rest[i + "DevOp::".len()..];
                    let n: String = r.chars().take_while(char::is_ascii_alphanumeric).collect();
                    if !n.is_empty() {
                        named.insert(n.clone());
                    }
                    rest = &r[n.len()..];
                }
            }
        }
        assert!(
            named.len() > 20,
            "parsed only {} DevOp:: references from {EMITTER_SRC:?} — the parse broke, and a broken \
             parse here reports every arm as unemitted",
            named.len()
        );
        let allow: std::collections::BTreeMap<&str, &str> =
            GFX950_UNEMITTED.iter().copied().collect();
        let mut unemitted: Vec<&str> = Vec::new();
        let mut stale: Vec<&str> = Vec::new();
        for c in GFX950_DISPATCHED {
            let op = DevOp::ALL.iter().copied().find(|o| o.c_name() == *c);
            let emitted = op.is_some_and(|o| named.contains(&format!("{o:?}")));
            match (emitted, allow.contains_key(c)) {
                (false, false) => unemitted.push(c),
                (true, true) => stale.push(c),
                _ => {}
            }
        }
        assert!(
            unemitted.is_empty(),
            "gfx950 dispatches {} arm(s) NO emitter routes to: {unemitted:?}.\nThis is the recurring \
             shape: the arm exists, is correct, is register-gated, and nothing selects it — so it \
             ships in the object, is paid for in the register budget, and never runs. The runtime \
             coverage gate CANNOT see this (it only checks emitted opcode => arm exists).\nEither \
             wire an emit path, or add the opcode to GFX950_UNEMITTED with the reason it is \
             deliberately unrouted.",
            unemitted.len()
        );
        assert!(
            stale.is_empty(),
            "GFX950_UNEMITTED claims {stale:?} is deliberately unrouted, but an emitter now names \
             it. Drop the row — a stale allowlist entry re-hides the next real instance."
        );
    }

    /// How each emitter family treats a precision knob. The states are exhaustive on purpose:
    /// there is no "not applicable", because a knob a family neither honours nor refuses is
    /// SILENTLY IGNORED, which is the defect.
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Knob {
        /// Read, and it selects a different arm here.
        Wired,
        /// Read, and refused with a message — the family has no arm for it.
        Refused,
        /// Not read at all. **This is the bug state.** A row may only be `Ignored` with a written
        /// justification, and every justification here should be read as a debt.
        Ignored(&'static str),
    }

    /// The precision axes × the two emitter families, and the file each family lives in.
    ///
    /// WHY A TABLE AND NOT A GREP. The failure is not "a knob is unread"; it is "a knob is unread
    /// *by one family* while another family honours it, so the same env var means fp8 over here
    /// and nothing at all over there". Only a per-family statement can express that, and writing
    /// the statement down is what makes the asymmetry visible at review time instead of at
    /// benchmark time.
    ///
    /// Scope is the PRECISION knobs specifically, because those are the ones whose silent
    /// no-op yields an asset that runs, produces correct-looking output, and is wrong about what
    /// it measured — §0's apples-to-apples rule.
    ///
    /// This used to add "shape/scheduling knobs (`PLOW_GEMV_MM`, `PLOW_GQ_BATCH`, …) fail visibly
    /// or not at all". THAT WAS FALSE, and `PLOW_GEMV_MM` is the counterexample: it is a KERNEL
    /// knob, so the failure was not in either emitter family this table covers — no AMD build
    /// input defined it at all, every gfx950 decode object compiled at op_gemm.h's default of 1,
    /// and a B=4 asset with a correct B-wide program, 4× KV cache and `gv_mm_max: 4` in its
    /// build.json produced ONE non-zero logits row. Sequences 1..B-1 sampled token 0 forever.
    /// That is silent, and it survived because the two backends NAME the knob differently
    /// (`GV_MM_MAX` on NVIDIA, `PLOW_GEMV_MM` on AMD) so a grep for either one looked wired.
    ///
    /// The uncovered axis is therefore "devgen emits a program shaped by knob K, but the kernel
    /// build for some backend never receives K". This table cannot see it — both emitter families
    /// were correct here. Routed now in `scripts/build_gfx950.sh` and `runtime/CMakeLists.txt`.
    ///
    /// THE GUARD OVER KERNEL-BUILD INPUTS NOW EXISTS, for this knob. Routing it was necessary and
    /// not sufficient: routing fixes the objects someone remembers to rebuild, and says nothing
    /// about the pairing of a given packet with a given object directory. `PLOW_GEMV_MM` is now
    /// EMITTED INTO the object as `plow_gemv_mm_cap_<N>` (`runtime/amd/op_gemm.h`, named for the
    /// macro itself so it cannot disagree with what was compiled), and `check_gemv_capacity`
    /// (`crates/plowrt/src/exec/amd.rs`) refuses at load when the packet's widest GEMV asks for
    /// more rows than the object advertises — or when it advertises nothing at all. That is the
    /// shape any future entry on this axis should take: make the kernel build input OBSERVABLE in
    /// the object, then compare it against the packet where the two finally meet.
    const PRECISION_KNOBS: &[(&str, Knob, Knob)] = &[
        // knob            dense-GQA (lib.rs)                    MLA/MoE (mla.rs)
        ("PLOW_FP8", Knob::Wired, Knob::Wired),
        ("PLOW_W8A16", Knob::Wired, Knob::Wired),
        ("PLOW_W8A8", Knob::Wired, Knob::Refused),
        ("PLOW_MXFP4", Knob::Refused, Knob::Wired),
        // The K3 full-model path in mla.rs now emits the compressed-latent fp8 twins. Other MLA
        // entry points still need the same wiring, but the family no longer silently ignores the
        // knob universally; K3's structural tests pin the allocation and opcode swap.
        ("PLOW_FP8_KV", Knob::Wired, Knob::Wired),
        ("PLOW_KV_FP8", Knob::Wired, Knob::Wired),
    ];

    /// CHECK B — the table above is true of the sources.
    ///
    /// Catches bug shape B directly: a precision knob wired on one family and unread on another.
    /// Before `PLOW_MXFP4` was refused on the dense path, its dense column was `Ignored`, and the
    /// only way to make this test pass was to WRITE DOWN that a dense `PLOW_MXFP4=1` build emits
    /// bf16 — which nobody would have written down and left.
    ///
    /// The evidence is `env::var("KNOB")`, not a mention: a comment naming the flag is exactly what
    /// the dense path had, and it is worth nothing at runtime.
    #[test]
    fn precision_knob_table_matches_the_emitters() {
        let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let read = |f: &str| std::fs::read_to_string(src_dir.join(f)).unwrap_or_default();
        let dense = read("lib.rs");
        let mla = read("mla.rs");
        assert!(
            !dense.is_empty() && !mla.is_empty(),
            "emitter sources not readable"
        );
        for (knob, d, m) in PRECISION_KNOBS {
            for (family, file, src, state) in [
                ("dense-GQA", "lib.rs", &dense, d),
                ("MLA/MoE", "mla.rs", &mla, m),
            ] {
                let reads = src.contains(&format!("env::var(\"{knob}\")"));
                match state {
                    Knob::Wired | Knob::Refused => assert!(
                        reads,
                        "PRECISION_KNOBS says the {family} emitter handles {knob} as {state:?}, but \
                         {file} contains no `env::var(\"{knob}\")`. Either it never did and the \
                         table is wrong, or a refactor dropped the read — in which case {knob}=1 is \
                         now SILENTLY IGNORED on {family} and that build emits the default \
                         precision under a flag that named another one."
                    ),
                    Knob::Ignored(why) => {
                        assert!(
                            !why.is_empty(),
                            "{knob} is Ignored on {family} with no justification. An unread \
                             precision knob is a silently-wrong asset; say why in the table."
                        );
                        assert!(
                            !reads,
                            "PRECISION_KNOBS says {knob} is IGNORED on {family}, but {file} now \
                             reads it. Good — update the row to Wired or Refused so the next reader \
                             is not told a fixed hole is still open."
                        );
                    }
                }
            }
        }
    }

    /// The instance this whole section exists for, pinned as behaviour rather than as a table row:
    /// `PLOW_MXFP4=1` on a dense model must not hand back a bf16 packet.
    ///
    /// Emission reads process-global env, so the variable is restored on the way out whether or
    /// not the assert fires — leaving it set would change every blob a later test in this binary
    /// emits. (`tests/golden_blob.rs` runs in a separate process and takes its own `EMIT_LOCK`.)
    #[test]
    fn dense_mxfp4_is_refused_not_silently_bf16() {
        let dir = std::env::temp_dir().join("devgen_dense_mxfp4_refusal");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3","hidden_size":512,"intermediate_size":1024,
                "num_hidden_layers":2,"num_attention_heads":8,"head_dim":64,
                "num_key_value_heads":2,"rms_norm_eps":1e-6,"vocab_size":4096,
                "rope_theta":1000000.0,"rope_scaling":null,"tie_word_embeddings":true}"#,
        )
        .unwrap();
        let out = dir.join("model.pkt");
        let args = || EmitArgs {
            dir: dir.clone(),
            ctx: 256,
            out: out.to_str().unwrap().to_string(),
            n_cu: 256,
            tp: 1,
            block_spec: None,
            embed_cubin: None,
            embed_hsaco: None,
            rope_gen: true,
            l2_layout: None,
            gpu: "MI355X".into(),
            arch: "gfx950".into(),
        };
        std::env::set_var("PLOW_MXFP4", "1");
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(args())));
        std::env::remove_var("PLOW_MXFP4");
        let e = r.expect_err(
            "PLOW_MXFP4=1 emitted a dense packet instead of refusing. That packet is byte-identical \
             to the bf16 one and will be benchmarked as mxfp4.",
        );
        let msg = e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains("dense_mxfp4_weights"),
            "the refusal must name the missing capability so the message is actionable; got: {msg}"
        );
    }
}

/// `require_mla_rope`: the MLA positional-encoding contract, pinned.
///
/// Every case is stated as "this config JSON, therefore this outcome", with the expected theta
/// read back OUT of the same JSON rather than written as a literal — so if the value moves the
/// test moves with it instead of quietly asserting the old number.
#[cfg(test)]
mod mla_rope_tests {
    use serde_json::{json, Value};

    /// `cfg_glm`'s theta lookup and its refusal, extracted so both can be exercised without a
    /// checkpoint on disk. MUST stay identical to the three lines in `mla::cfg_glm`.
    fn resolve(v: &Value) -> Option<f64> {
        let rp = &v["rope_parameters"];
        let theta = v["rope_theta"]
            .as_f64()
            .or_else(|| rp["rope_theta"].as_f64());
        super::require_mla_rope(
            theta,
            v["mla_use_nope"].as_bool().unwrap_or(false),
            rp["rope_type"].as_str(),
            v["rope_scaling"].as_object().is_some(),
            v["model_type"].as_str().unwrap_or("<test>"),
        );
        theta
    }

    /// The SHIPPING model's spelling. GLM-5.2's `config.json` has NO top-level `rope_theta`; it
    /// carries `rope_parameters: {rope_theta, rope_type}` (transformers 5.x moved the key). The
    /// old `.unwrap_or(8_000_000.0)` therefore never read GLM's theta at all — it matched only
    /// because the literal in `mla.rs` happened to equal it.
    ///
    /// Asserted against the value IN the fixture, so a fixture edit cannot leave this passing
    /// while the parse reads nothing.
    #[test]
    fn the_theta_comes_from_rope_parameters_not_from_a_default() {
        let v = json!({
            "model_type": "glm_moe_dsa",
            "rope_parameters": { "rope_theta": 8_000_000.0, "rope_type": "default" },
        });
        assert_eq!(resolve(&v), v["rope_parameters"]["rope_theta"].as_f64());
        // A different theta under the same spelling must produce that theta, not GLM's. This is
        // the property the default destroyed: every model read as 8e6, and all of them looked
        // right as long as they were GLM.
        let other = json!({
            "model_type": "some_other_mla",
            "rope_parameters": { "rope_theta": 123_457.0, "rope_type": "default" },
        });
        assert_eq!(
            resolve(&other),
            other["rope_parameters"]["rope_theta"].as_f64()
        );
        assert_ne!(
            resolve(&other),
            resolve(&v),
            "two configs must not resolve to one theta"
        );
    }

    /// The flat spelling still works and takes precedence.
    #[test]
    fn the_top_level_spelling_is_still_read() {
        let v = json!({ "model_type": "deepseek_v3", "rope_theta": 10_000.0 });
        assert_eq!(resolve(&v), v["rope_theta"].as_f64());
    }

    /// Kimi-K3: `mla_use_nope: true`, no theta anywhere. VERIFIED against the checkpoint —
    /// `config.json`'s only `rope`-ish key is `text_config.qk_rope_head_dim`, and
    /// `modeling_kimi_linear.py` has `self.rotary_emb = None` / `assert self.use_nope`.
    #[test]
    #[should_panic(expected = "mla_use_nope")]
    fn a_nope_model_is_refused_not_given_glms_theta() {
        resolve(&json!({ "model_type": "kimi_k3", "mla_use_nope": true }));
    }

    /// The refusal names the consequence, not just the flag — a NoPE emit is not "delete the two
    /// HeadNormRope ops", because the k-side one is the only writer of the krot cache row.
    #[test]
    fn the_nope_refusal_names_the_krot_cache() {
        let msg = std::panic::catch_unwind(|| {
            resolve(&json!({ "model_type": "kimi_k3", "mla_use_nope": true }))
        })
        .unwrap_err();
        let msg = msg
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_else(|| msg.downcast_ref::<&str>().map(|s| s.to_string()).unwrap());
        assert!(
            msg.contains("krot"),
            "refusal must name the dangling cache write; got: {msg}"
        );
    }

    /// Contradiction: a theta AND `mla_use_nope`. One of the two is wrong and the compiler
    /// cannot tell which, so it refuses instead of picking.
    #[test]
    #[should_panic(expected = "contradict")]
    fn a_theta_alongside_use_nope_is_a_contradiction() {
        resolve(&json!({
            "model_type": "confused", "mla_use_nope": true, "rope_theta": 8_000_000.0,
        }));
    }

    /// No theta, no NoPE flag: the compiler does not know the model's positional encoding.
    /// This is the case the default silently answered with GLM's number.
    #[test]
    #[should_panic(expected = "no RoPE theta")]
    fn an_absent_theta_is_a_refusal_not_eight_million() {
        resolve(&json!({ "model_type": "mystery_mla" }));
    }

    /// `declare_glm` builds its tables with `RopeScale::None`, so a scaled scheme would be
    /// emitted as an UNSCALED RoPE at the base theta — right-looking tables, wrong long context.
    #[test]
    #[should_panic(expected = "rope_type")]
    fn a_scaled_rope_scheme_is_refused_rather_than_silently_unscaled() {
        resolve(&json!({
            "model_type": "yarned",
            "rope_parameters": { "rope_theta": 500_000.0, "rope_type": "yarn" },
        }));
    }

    /// The legacy `rope_scaling` object, same reason.
    #[test]
    #[should_panic(expected = "rope_scaling")]
    fn a_legacy_rope_scaling_object_is_refused() {
        resolve(&json!({
            "model_type": "scaled",
            "rope_theta": 500_000.0,
            "rope_scaling": { "type": "linear", "factor": 4.0 },
        }));
    }
}

#[cfg(test)]
mod fp8_key_tests {
    //! The `fp8/` key contract, pinned. Three parties have to agree on one string: this emitter
    //! declares the packet tensor name, `quantize_fp8.py` writes the safetensors key, and the
    //! runtime looks it up. They did not — the emitter and the quantizer wrote `fp8/<name>` while a
    //! loader stripped the prefix and looked up `<name>`, so a freshly generated fp8 checkpoint
    //! could not load. These assertions are the emitter's half of the contract, stated as code so
    //! the spelling cannot drift silently.
    use super::*;

    /// The canonical forms. `_scale` goes on the END, after the full weight name — `fp8/X.weight`
    /// pairs with `fp8/X.weight_scale`, NOT `fp8/X_scale.weight`.
    #[test]
    fn fp8_twin_names_are_the_declared_name_verbatim() {
        let w = format!("fp8/{}", "model.layers.3.self_attn.q_proj.weight");
        let s = format!("{w}_scale");
        assert_eq!(w, "fp8/model.layers.3.self_attn.q_proj.weight");
        assert_eq!(s, "fp8/model.layers.3.self_attn.q_proj.weight_scale");
        // The key is the packet name VERBATIM: no strip, no rewrite, nothing to apply twice.
        assert_eq!(
            w.strip_prefix("fp8/"),
            Some("model.layers.3.self_attn.q_proj.weight")
        );
        assert!(
            w.starts_with("fp8/"),
            "the prefix is part of the key, not a routing marker"
        );
    }

    /// An `fp8/` twin is checkpoint-bound weight bytes, and every reader agrees on that because
    /// there is now only one reader: `packet::names::is_checkpoint_weight`.
    ///
    /// This test used to spell the predicate out as
    /// `starts_with("model.") || starts_with("fp8/")` and named `manager.rs` / `exec/gpu.rs` as
    /// the two sites it mirrored. There were five sites, they disagreed, and the allowlist form
    /// silently zeroed an untied `lm_head.weight` on CUDA and would have zeroed the whole
    /// Kimi-K3 tower. Asserting against the shared predicate is the point — a re-spelt copy is
    /// exactly how the five diverged.
    #[test]
    fn fp8_twins_are_weight_bytes_under_the_shared_predicate() {
        use packet::names::is_checkpoint_weight as w;
        assert!(w("fp8/model.layers.0.mlp.down_proj.weight"));
        assert!(w("fp8/model.layers.0.mlp.down_proj.weight_scale"));
        assert!(w("model.layers.0.mlp.down_proj.weight"));
        assert!(
            w("lm_head.weight"),
            "untied head: declared at the top level, and a weight"
        );
        assert!(!w("act.x"), "activations are not weight bytes");
        assert!(!w("in.pos"));
    }

    /// The scale is per OUTPUT CHANNEL — one f32 per row of the `[out, in]` weight — and the
    /// dequant is a MULTIPLY. Both halves matter: `quantize_fp8.py` stores `amax/448` and the
    /// device epilogue computes `acc * a_scale[m] * w_scale[n]`, so a reciprocal on either side
    /// would be a silent 448²-ish error rather than a crash.
    #[test]
    fn fp8_scale_is_per_output_channel_and_multiplied() {
        let (out, inp) = (4096u64, 2560u64);
        assert_eq!(
            out * F32,
            16384,
            "scale vector is [out] f32, not [out,in] and not [in]"
        );
        // Round-trip the convention the quantizer documents, at f32 precision.
        let w: f32 = -0.37;
        let amax: f32 = 0.37;
        let scale = amax / 448.0;
        let q = (w / scale).round().clamp(-448.0, 448.0);
        assert!((q * scale - w).abs() < 1e-3, "dequant is w8 * scale");
        let _ = inp;
    }
}

#[cfg(test)]
mod fp8_profile_tests {
    //! The `PLOW_FP8=1` profile emits w8a16 — and gfx950 has no w8a16 prefill GEMM, only w8a8.
    //! `d_gemm_fp8` there reads t[1] as e4m3 bytes and dereferences `ascale[m]` with no null check,
    //! so a w8a16 packet faults on its first prefill GEMM with no diagnostic. These pin the gate
    //! that refuses it, and pin that the profile which DOES work is left alone.
    use super::*;
    use packet::dev::DevInst;
    use packet::devbuild::Program;

    fn prog(insts: Vec<DevInst>) -> Program {
        Program {
            hier_base: 0,
            n_cu: 4,
            n_counter: 0,
            insts,
            stream: vec![],
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            tensors: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_sms: 0,
            l2_domains: 0,
        }
    }

    /// `t[3]` is a_scale. `TENSOR_NONE` there is w8a16; a bound handle is w8a8.
    fn model(a_scale: u32) -> Model {
        let mut i = DevInst {
            op: DevOp::GemmFp8 as u16,
            blocks: 1,
            ..Default::default()
        };
        i.t[3] = a_scale;
        Model {
            n_cu: 256,
            target: 0,
            tensors: vec![],
            progs: vec![prog(vec![i])],
            kv_row_insts: vec![],
            prog_t: vec![128],
            gen: vec![],
        }
    }

    #[test]
    #[should_panic(expected = "fp8_w8a16_prefill")]
    fn w8a16_fp8_prefill_is_refused_on_gfx950() {
        check_fp8_a_scale_bound(&model(TENSOR_NONE), "gfx950", "");
    }

    /// w8a8 binds t[3], which is the profile that actually runs on gfx950.
    #[test]
    fn w8a8_fp8_prefill_passes_on_gfx950() {
        check_fp8_a_scale_bound(&model(7), "gfx950", "");
    }

    /// sm_120 HAS a w8a16 cubin, so the same packet is valid there and must not be refused —
    /// the gate is about one target's kernel, not about w8a16 being wrong.
    #[test]
    fn w8a16_fp8_prefill_is_fine_on_sm120() {
        check_fp8_a_scale_bound(&model(TENSOR_NONE), "sm_120a", "RTX5090");
    }

    /// The trap-asset case: `--arch sm_120a` (where w8a16 is legitimate) with an AMD `--gpu`. An
    /// arch-only gate would pass this, and `build-amd/g31b-fp8kv` is exactly it — emitted for
    /// sm_120a, sized for 256 CUs, run on gfx950, faulted. Either signal saying AMD is enough.
    #[test]
    #[should_panic(expected = "fp8_w8a16_prefill")]
    fn w8a16_is_refused_when_only_the_gpu_says_amd() {
        check_fp8_a_scale_bound(&model(TENSOR_NONE), "sm_120a", "MI350X");
    }

    /// …and the target predicate behind it agrees with both signals independently.
    #[test]
    fn target_is_amd_reads_either_signal() {
        assert!(target_is_amd("gfx950", ""));
        assert!(target_is_amd("", "MI350X"));
        assert!(
            target_is_amd("sm_120a", "MI350X"),
            "the gpu is enough on its own"
        );
        assert!(!target_is_amd("sm_120a", "RTX5090"));
        assert!(
            !target_is_amd("", ""),
            "no target => unchanged emission (golden tests)"
        );
    }

    /// EVERY fp8 rung is gated, not just the three the ladder started with.
    ///
    /// The regression this pins: the tile-inventory campaign grew `GFX950_RUNGS` from 3 rungs to
    /// 5, so `pick_tile` could return `GemmWideFp8` (128x256) and `GemmC5Fp8` (192x256) — and the
    /// gate's hand-written opcode list still named only `Gemm/GemmMed/GemmSmall`. A w8a16 packet
    /// whose shape resolved to either new rung compiled clean and null-dereferenced `ascale[m]` on
    /// device, which is precisely what this gate exists to stop.
    ///
    /// Written as a loop over the table rather than five literal cases on purpose: a sixth rung
    /// is covered the moment it is added, with no second place to remember to update. That is the
    /// same argument `GFX950_RUNGS`'s own doc comment makes for there being one table.
    #[test]
    fn every_fp8_rung_is_refused_when_a_scale_is_unbound() {
        for (_, fp8, _, bm, bn, _) in GFX950_RUNGS {
            let mut i = DevInst {
                op: fp8 as u16,
                blocks: 1,
                ..Default::default()
            };
            i.t[3] = TENSOR_NONE;
            let m = Model {
                n_cu: 256,
                target: 0,
                tensors: vec![],
                progs: vec![prog(vec![i])],
                kv_row_insts: vec![],
                prog_t: vec![128],
                gen: vec![],
            };
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check_fp8_a_scale_bound(&m, "gfx950", "")
            }));
            assert!(
                caught.is_err(),
                "the {bm}x{bn} fp8 rung ({fp8:?}) is emittable by pick_tile but not gated: a \
                 w8a16 packet on that shape would reach d_gemm_fp8's epilogue and dereference a \
                 null ascale on device"
            );
        }
    }

    /// A bf16 packet has no fp8 GEMM at all and must sail through on every target.
    #[test]
    fn bf16_packets_are_untouched() {
        let i = DevInst {
            op: DevOp::Gemm as u16,
            blocks: 1,
            ..Default::default()
        };
        let m = Model {
            n_cu: 256,
            target: 0,
            tensors: vec![],
            progs: vec![prog(vec![i])],
            kv_row_insts: vec![],
            prog_t: vec![128],
            gen: vec![],
        };
        check_fp8_a_scale_bound(&m, "gfx950", "");
    }
}

#[cfg(test)]
mod pick_tile_tests {
    //! The hwspec-driven picker is a STATIC, shape-agnostic choice — so it is testable
    //! offline, with no GPU. These lock in the tile chosen for every projection of the three
    //! supported architectures at the prefill chunk sizes that matter, proving the picker both
    //! fills the CUs on the underutilized shapes AND does not regress the ones that already
    //! saturate. `n_cu = 256` (MI350X).
    use super::{
        gemm_lds_bytes, glu_era_inventory, hwspec, pick_tile, select_gemm_over, DevOp, GFX950_TILES,
    };
    use costmodel::cost::{dma_cycles, macs_cycles};
    use costmodel::MmaDtype;
    use kernelcaps::QuantScheme;

    const N_CU: u32 = 256;
    fn pt(m: u32, n: u32, k: u32) -> DevOp {
        pick_tile(m, n, k, N_CU, QuantScheme::None)
    }
    /// The picker restricted to the three rungs that existed before the tile-inventory
    /// campaign — the set the legacy reference below ranks over.
    fn pt_legacy_rungs(m: u32, n: u32, k: u32, n_cu: u32) -> DevOp {
        select_gemm_over(glu_era_inventory(), m, n, k, n_cu, QuantScheme::None)
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

    const MS: [u32; 12] = [1, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
    const NS: [u32; 12] = [
        128, 512, 1024, 2048, 2560, 4096, 5376, 8192, 9728, 14336, 16384, 21504,
    ];
    const KS: [u32; 8] = [128, 512, 2560, 4096, 5376, 8192, 14336, 21504];
    const CUS: [u32; 5] = [1, 64, 128, 256, 304];

    /// Routing selection through the registry must not change a single answer **among the three
    /// rungs the old picker had**. Swept rather than sampled: tie-breaking was the real risk,
    /// since the old loop preferred the larger tile by table order while opcode order would put
    /// `GemmSmall` (14) ahead of `GemmMed` (15).
    ///
    /// Scoped to the legacy rungs deliberately. The campaign ADDED two tiles, so comparing the
    /// full picker against a three-tile reference would assert the new rungs are never chosen —
    /// the opposite of the intent. What must not drift is the *ranking rule*, and that is what
    /// this pins.
    #[test]
    fn the_original_rungs_still_rank_exactly_as_the_legacy_picker_did() {
        let mut checked = 0usize;
        for &m in &MS {
            for &n in &NS {
                for &k in &KS {
                    for &n_cu in &CUS {
                        let want = pick_tile_legacy(m, n, k, n_cu);
                        let got = pt_legacy_rungs(m, n, k, n_cu);
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
        assert_eq!(checked, MS.len() * NS.len() * KS.len() * CUS.len());
    }

    /// The added rungs must never be chosen where they cannot help, and the ranking must stay
    /// TOTAL — no shape may resolve by opcode number.
    ///
    /// The second half is the real content. `tile_cost`'s tie-break used three hand-written
    /// brackets over `BM*BN`, and 192x256 (49152) and 128x256 (32768) both landed in the same
    /// bracket as 256x256 (65536), so on any shape where their wall-clock costs tied — which is
    /// every shape small enough that all three take one round — the winner would have been
    /// decided by `DevOp` number. `GemmC5` is 95 and `Gemm` is 8, so the *old* tile would have
    /// won silently and the campaign would have measured nothing.
    #[test]
    fn every_shape_resolves_on_cost_rather_than_opcode_number() {
        use super::{gfx950_gemm_inventory, tile_cost};
        let spec = hwspec::registry::lookup("MI350X").unwrap();
        for &m in &MS {
            for &n in &NS {
                for &k in &KS {
                    for &n_cu in &CUS {
                        let chosen = pick_tile(m, n, k, n_cu, QuantScheme::None);
                        let costs: Vec<(DevOp, u64)> = gfx950_gemm_inventory()
                            .iter()
                            .filter(|s| s.quant == QuantScheme::None)
                            .map(|s| {
                                (
                                    s.id.0,
                                    tile_cost(spec, s, m as i64, n as i64, k as i64, n_cu),
                                )
                            })
                            .collect();
                        let best = costs.iter().map(|c| c.1).min().unwrap();
                        let ties: Vec<DevOp> =
                            costs.iter().filter(|c| c.1 == best).map(|c| c.0).collect();
                        assert_eq!(
                            ties.len(),
                            1,
                            "m={m} n={n} k={k} n_cu={n_cu}: {ties:?} tie at cost {best}, so the \
                             winner is decided by opcode number"
                        );
                        assert_eq!(chosen, ties[0], "m={m} n={n} k={k} n_cu={n_cu}");
                    }
                }
            }
        }
    }

    /// Precision changes the ANSWER, not just the opcode name.
    ///
    /// Two things are asserted, and the second is the one that was broken: every encoding must
    /// select from its OWN rungs (a bf16 opcode emitted for an mxfp4 op would read packed fp4
    /// bytes as bf16), and the fp8/fp4 answer must be free to differ from bf16's — the fp8 body
    /// moves half the operand bytes for the same MFMA count, so `max(compute, dma)` tips.
    #[test]
    fn each_encoding_selects_from_its_own_rungs() {
        for &m in &MS {
            for &n in &NS {
                for &k in &KS {
                    for (quant, ok) in [
                        (QuantScheme::None, &super::GFX950_RUNGS.map(|r| r.0)),
                        (QuantScheme::W8A8, &super::GFX950_RUNGS.map(|r| r.1)),
                        (QuantScheme::Mxfp4, &super::GFX950_RUNGS.map(|r| r.2)),
                    ] {
                        let got = pick_tile(m, n, k, N_CU, quant);
                        assert!(
                            ok.contains(&got),
                            "{quant:?} at {m}x{n}x{k} selected {got:?}, which is not one of its \
                             own rungs {ok:?}"
                        );
                    }
                }
            }
        }
    }

    /// The mxfp4 prefill GEMM is no longer pinned to 256x256 for every shape.
    ///
    /// This is the T3 regression guard. `mla.rs` used to emit `DevOp::GemmMxfp4` unconditionally,
    /// so Kimi's `kv_a_proj` (M=128, N=576) ran as THREE 256x256 tiles on 256 CUs — measured at
    /// ≈0.4% of peak, the worst number in the campaign.
    /// WHICH SHAPES THE CAMPAIGN CHANGED, and which it deliberately did not.
    ///
    /// `legacy` is the three-rung analytical picker that shipped; `new` is the five-rung one.
    /// Measurements do not apply here — the unit-test fixture inventory has build label
    /// `test-fixture`, so every record in `tuning/` is correctly stale against it — so this
    /// isolates what the RUNGS alone bought. The extra shapes measurement then corrects are
    /// pinned in `tests/tuned_tile_selection.rs`.
    ///
    /// Measured TF/s (whole GPU, 1660 sustained bf16 peak, `runtime/ubench/gemm_tile_sweep.c`):
    ///
    /// | shape                     | legacy   | new     | TF/s legacy -> new    |
    /// |---------------------------|----------|---------|-----------------------|
    /// | g31b q_proj      M=1024   | 256x256  | 128x256 |  521.3 ->  915.0 1.76x|
    /// | g31b kv global   M=4096   | 256x256  | 128x256 |  513.7 ->  921.7 1.79x|
    /// | llama-8B k/v     M=8192   | 256x256  | 128x256 |  475.4 ->  832.4 1.75x|
    /// | g31b o_proj      M=4096   | 256x256  | 192x256 |  792.1 -> 1194.3 1.51x|
    /// | qwen o_proj      M=4096   | 256x256  | 192x256 |  583.5 ->  926.5 1.59x|
    /// | g31b down_proj   M=2048   | 256x256  | 192x256 |  789.5 -> 1025.8 1.30x|
    ///
    /// And the six M=128 "utilisation disaster" shapes are UNCHANGED, on purpose: 64x128 was
    /// already selected and is already the fastest of all twelve tiles compiled into the sweep.
    /// Their deficit is CU fill (2-34 tiles on 256 CUs), which no tile can fix — see the report
    /// and `plans/` for why split-K is the lever there.
    #[test]
    fn the_new_rungs_change_the_fill_limited_shapes_and_leave_the_rest() {
        for (m, n, k, legacy, new, label) in [
            // Unchanged: already on the narrowest rung, and it is already optimal.
            (
                128u32,
                128u32,
                2816u32,
                DevOp::GemmSmall,
                DevOp::GemmSmall,
                "gemma26b router",
            ),
            (
                128,
                256,
                6144,
                DevOp::GemmSmall,
                DevOp::GemmSmall,
                "glm52 router",
            ),
            (
                128,
                576,
                6144,
                DevOp::GemmSmall,
                DevOp::GemmSmall,
                "glm52 kv_a_proj",
            ),
            (
                128,
                576,
                7168,
                DevOp::GemmSmall,
                DevOp::GemmSmall,
                "kimi kv_a_proj",
            ),
            (
                128,
                512,
                3840,
                DevOp::GemmSmall,
                DevOp::GemmSmall,
                "g12b k_proj global",
            ),
            (
                128,
                2112,
                2816,
                DevOp::GemmSmall,
                DevOp::GemmSmall,
                "g26b dense gate/up",
            ),
            (
                256,
                8192,
                5376,
                DevOp::GemmSmall,
                DevOp::GemmSmall,
                "g31b q M=256",
            ),
            (
                512,
                8192,
                5376,
                DevOp::GemmMed,
                DevOp::GemmMed,
                "g31b q M=512",
            ),
            // Changed: fill- or quantisation-limited at 256x256.
            (
                1024,
                8192,
                5376,
                DevOp::Gemm,
                DevOp::GemmWide,
                "g31b q M=1024",
            ),
            (
                4096,
                2048,
                5376,
                DevOp::Gemm,
                DevOp::GemmWide,
                "g31b kv global M=4096",
            ),
            (
                8192,
                1024,
                4096,
                DevOp::Gemm,
                DevOp::GemmWide,
                "llama-8B k/v M=8192",
            ),
            (
                4096,
                5376,
                8192,
                DevOp::Gemm,
                DevOp::GemmC5,
                "g31b o M=4096",
            ),
            (
                4096,
                2560,
                4096,
                DevOp::Gemm,
                DevOp::GemmC5,
                "qwen o M=4096",
            ),
            (
                2048,
                5376,
                21504,
                DevOp::Gemm,
                DevOp::GemmC5,
                "g31b down M=2048",
            ),
        ] {
            assert_eq!(pt_legacy_rungs(m, n, k, N_CU), legacy, "legacy: {label}");
            assert_eq!(pt(m, n, k), new, "new: {label}");
        }
    }

    #[test]
    fn mxfp4_prefill_is_tile_selected_not_pinned() {
        assert_eq!(
            pick_tile(128, 576, 7168, N_CU, QuantScheme::Mxfp4),
            DevOp::GemmSmallMxfp4,
            "Kimi kv_a_proj: the narrow-M rung, not the 256x256 default"
        );
        assert_eq!(
            pick_tile(128, 576, 6144, N_CU, QuantScheme::Mxfp4),
            DevOp::GemmSmallMxfp4,
            "GLM-5.2 kv_a_proj"
        );
        // ...and it still picks a large tile where a large tile is right, so this is selection
        // rather than a blanket swap in the other direction.
        assert_ne!(
            pick_tile(8192, 8192, 5376, N_CU, QuantScheme::Mxfp4),
            DevOp::GemmSmallMxfp4,
            "a saturating shape must not get the narrow tile"
        );
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
        // At M=8192 k/v make 32x4 = 128 tiles at 256x256 — HALF the machine. The old comment
        // here concluded "splitting to 128x128 would need 2 rounds for equal cost, so the
        // higher-intensity 256x256 stays", and that was true of the rungs available: both ways
        // of doubling the tile count halved BOTH dimensions or halved BN, and neither paid.
        //
        // 128x256 is the rung that was missing. It halves BM only, so the tile count doubles to
        // 64x4 = 256 — exactly full — while BN stays 256 and the A-operand reuse is untouched.
        // This is the shape class the campaign was for.
        assert_eq!(pt(8192, 1024, 4096), DevOp::GemmWide, "k/v at 8k");
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
        // down_proj is N=2560, which is 10 tile-columns at BN=256 — so at 256x256 it is
        // 16x10 = 160 tiles, 62.5% of the machine, and has been all along. 192x256 gives
        // 22x10 = 220 (86%). MEASURED on the sibling o_proj shape (4096x2560x4096, whole GPU,
        // runtime/ubench/gemm_tile_sweep.c): 256x256 587.7 TF/s vs 192x256 940.6 — **1.60x**,
        // the largest single-shape win in the campaign.
        assert_eq!(
            pt(4096, 2560, 9728),
            DevOp::GemmC5,
            "down_proj (fill: 62.5% -> 86%)"
        );
    }

    #[test]
    fn gemma31b_tiles() {
        // hidden 5376, inter 21504. The projections that genuinely saturate 256 CUs at 256x256
        // keep it — the campaign must not drag them onto a smaller tile.
        assert_eq!(
            pt(4096, 8192, 5376),
            DevOp::Gemm,
            "q sliding (32x32 = 1024 tiles)"
        );
        assert_eq!(pt(4096, 16384, 5376), DevOp::Gemm, "q global");
        assert_eq!(
            pt(4096, 4096, 5376),
            DevOp::Gemm,
            "kv sliding (N=4096, 16x16 = 256)"
        );
        assert_eq!(pt(4096, 21504, 5376), DevOp::Gemm, "gate/up");
        // o_proj and down_proj are both N=5376 = 21 tile-columns at BN=256, so 256x256 gives
        // 16x21 = 336 tiles = 2 rounds at 65.6% efficiency — the tile-count QUANTIZATION case
        // rather than the under-fill case. 192x256 gives 22x21 = 462 = 2 rounds at 90.2%.
        // MEASURED on this N at M=2048 (2048x5376x21504, whole GPU): 256x256 794.4 TF/s vs
        // 192x256 1033.4 — **1.30x**.
        assert_eq!(
            pt(4096, 5376, 8192),
            DevOp::GemmC5,
            "o sliding (quantization: 66% -> 90%)"
        );
        assert_eq!(
            pt(4096, 5376, 21504),
            DevOp::GemmC5,
            "down (same N, same quantization)"
        );
        // kv GLOBAL is N=2048 = 8 tile-columns at BN=256, so 16x8 = 128 tiles — HALF the
        // machine, and the previous version of this test asserted that as "no regression"
        // because there was no rung that could fix it. 128x256 makes it 32x8 = 256, exactly
        // full, at the same BN and so the same A-reuse.
        assert_eq!(
            pt(4096, 2048, 5376),
            DevOp::GemmWide,
            "kv global (fill: 50% -> 100%)"
        );
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
