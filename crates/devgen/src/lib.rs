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
pub mod emit_config;
pub mod k3;
pub mod kda;
use config::*;
mod ladder;
mod mla;
#[cfg(test)]
mod test_env;
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
// | `PLOW_L2_PLACE` | per-domain queue windows | formerly overwrote the wave-class tag on a MULTI-SEGMENT program → zero logits | domain now lives independently in flags; ordered kernel-family segments remain intact |
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
    emit_config::active()
        .fa_gf_full
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
    gemm_lds_bytes_buffered(bm, bn, bk, 2)
}

/// The same stage, with the buffer count named.
///
/// THIS MUST TRACK `GM_DBUF` IN `runtime/amd/op_gemm.h`, and for a while it did not. The kernel's
/// `GM_LDS_HALVES_T` used to hardcode a `2 *` double buffer and so did this; then CDNA3 got a
/// single-buffered stage, because on a 64 KiB part double-buffering IS the tile ceiling and no
/// tile wider than 64x128 fits with it. The compiler kept the old arithmetic, which does not
/// fail loudly -- it silently REJECTS the tile the kernel can actually run:
///
///   192x256x64  double-buffered 129,024 B -> over 65,536, filtered out
///               single-buffered  64,512 B -> fits, and measures 2.8x the 64x128 rung
///
/// So a stale `2` here does not produce a wrong answer, it produces a slow one, chosen
/// confidently. The buffer count is a property of the object being built, so it comes from the
/// ISA the emitter is targeting.
pub fn gemm_lds_bytes_buffered(bm: u64, bn: u64, bk: u64, buffers: u64) -> u64 {
    buffers * (bm + bn) * (bk + 8) * 2
}

/// Stage buffers the interpreter is built with for `isa`. CDNA3 single-buffers to buy the tile
/// back (see [`gemm_lds_bytes_buffered`]); everything else double-buffers so the operand fill
/// hides behind the matrix compute.
pub fn stage_buffers(isa: hwspec::IsaLevel) -> u64 {
    isa.geometry().map_or(2, |g| g.gemm_stage_buffers as u64)
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

    // Filter against the stage this TARGET is built with, not a fixed double buffer -- see
    // `gemm_lds_bytes_buffered`. Costing CDNA3 as double-buffered rejects every tile the
    // single-buffered object can hold.
    //
    // DERIVED FROM `spec`, NOT FROM THE AMBIENT `amd_target` THREAD-LOCAL, and that is the whole
    // point: this function already takes the part to cost against as an argument, so reading the
    // buffer count from somewhere else lets the two disagree. They did. `plowc tune select`
    // (crates/plowc/src/tune/mod.rs) looks the spec up from the fingerprint and never calls
    // `set_amd_target`, so `spec` was MI300X (64 KiB) while `active()` still answered the MI350X
    // default -- buffers 2 -- and every rung except 64x128 scored `u64::MAX`. `tune select` then
    // reported PLOW_DOP_GEMM_SMALL for EVERY gfx942 shape, including 8192x32768x2048 where the
    // measured ladder puts 64x128 at roughly HALF of 192x256 (247 vs 481 TF/s). The emitters were never wrong
    // (both `run_verified` and the direct entry call `set_amd_target` first), but the command
    // whose doc-comment promises "the SAME ranking the compiler uses" quietly disagreed with it.
    // Taking the ISA from `spec` makes the function self-consistent for every caller.
    let buffers = hwspec::IsaLevel::from_spec(spec).map_or(2, stage_buffers);
    if gemm_lds_bytes_buffered(bm, bn, bk, buffers) > spec.sm.shared_mem.0 {
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

/// Ambient emit target for [`pick_tile`]. Default `true` keeps gfx950 unit tests / tune reporting
/// byte-stable when no emit wrapper is installed; NVIDIA emit MUST call [`with_emit_target_amd`]
/// `(false, …)` or the gfx950 analytical inventory re-selects `GemmWide`/`GemmC5` and Hopper
/// prefills trap.
pub(crate) fn emit_is_amd() -> bool {
    EMIT_IS_AMD.with(|c| c.get())
}

thread_local! {
    static EMIT_IS_AMD: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Run `f` with [`pick_tile`] bound to an AMD (`true`) or NVIDIA (`false`) inventory.
pub(crate) fn with_emit_target_amd<R>(amd: bool, f: impl FnOnce() -> R) -> R {
    EMIT_IS_AMD.with(|c| {
        let prev = c.replace(amd);
        let out = f();
        c.set(prev);
        out
    })
}

/// RAII form of [`with_emit_target_amd`] for long emit functions that cannot be wrapped as a
/// single closure without a large extract.
struct EmitAmdGuard {
    prev: bool,
}

impl EmitAmdGuard {
    fn set(amd: bool) -> Self {
        let prev = EMIT_IS_AMD.with(|c| c.replace(amd));
        Self { prev }
    }
}

impl Drop for EmitAmdGuard {
    fn drop(&mut self) {
        EMIT_IS_AMD.with(|c| c.set(self.prev));
    }
}

fn pick_tile(m: u32, n: u32, k: u32, n_cu: u32, quant: kernelcaps::QuantScheme) -> DevOp {
    pick_tile_tiered(m, n, k, n_cu, quant).0
}

pub(crate) fn pick_gemm_emit_plan(
    m: u32,
    n: u32,
    k: u32,
    n_cu: u32,
    quant: kernelcaps::QuantScheme,
) -> (DevOp, u32, u32) {
    let c8 = tunedb::gemm_rung_emit_plan("128x384x64", quant);
    let c8 = c8.filter(|plan| {
        let blocks = plan.blocks(m, n);
        emit_is_amd()
            && amd_target::active().1 == hwspec::IsaLevel::Gfx950
            && quant == kernelcaps::QuantScheme::None
            && m.is_multiple_of(plan.bm)
            && n.is_multiple_of(plan.bn)
            && k.is_multiple_of(plan.bk)
            && blocks == n_cu
            && emit_config::active().gemm_wide_c8_for(m, n, k)
            && gfx950_gemm_measurements().variant_is_winner(
                m as i64,
                n as i64,
                k as i64,
                quant,
                plan.measurement_id,
            )
    });
    if let Some(plan) = c8 {
        // The tagged path queried the same qualified/current shape table as `pick_tile`, but it
        // bypasses `select_kernel` because the implementation shares an opcode with c2. Preserve
        // the one-lookup/one-decision accounting contract at this alternate decision boundary.
        tune_demand::record(m as i64, n as i64, k as i64, quant, true);
        tune_demand::note_decision(true);
        return (plan.op, plan.blocks(m, n), plan.packet_tag);
    }
    (gfx950_prefill_tile(m, n, k, n_cu, quant), n_cu, 0)
}

/// [`pick_tile`] plus WHAT DECIDED IT. One implementation, because two copies of this function
/// is exactly how the pair in `plowc/src/bin/` drifted apart.
fn pick_tile_tiered(
    m: u32,
    n: u32,
    k: u32,
    n_cu: u32,
    quant: kernelcaps::QuantScheme,
) -> (DevOp, kernelcaps::CalibrationTier) {
    // NVIDIA sm_90a / sm_120a objects dispatch Gemm / GemmMed / GemmSmall as ALIASES of ONE
    // body (one object-wide BM×BN×BK). GemmWide / GemmC5 exist only on gfx950. Selecting them
    // from the gfx950 analytical inventory while emitting for Hopper/Blackwell put
    // `PLOW_DOP_GEMM_WIDE` into GH200 packets; the NVIDIA interp has no case → `default:
    // __trap()` → `CUDA_ERROR_LAUNCH_FAILED` on first prefill.
    if !emit_is_amd() {
        return (
            nvidia_prefill_gemm_op(quant),
            kernelcaps::CalibrationTier::Portable,
        );
    }
    select_gemm_over(gfx950_gemm_inventory(), m, n, k, n_cu, quant)
}

/// Canonical NVIDIA prefill GEMM opcode for `quant`.
///
/// Tile geometry is fixed by the cubin’s `PGM_*` / `PGM90_*` macros; the three Gemm{,Med,Small}
/// opcodes share that body, so the emitted op only needs to be one the switch dispatches.
fn nvidia_prefill_gemm_op(quant: kernelcaps::QuantScheme) -> DevOp {
    match quant {
        kernelcaps::QuantScheme::None | kernelcaps::QuantScheme::W8A16 => DevOp::Gemm,
        // Prefill fp8 projections (w8a16 or w8a8 cubin) share GEMM_*_FP8 opcodes; body selected
        // by cubin `-D`, not by a fifth DevOp.
        kernelcaps::QuantScheme::W8A8 => DevOp::GemmFp8,
        kernelcaps::QuantScheme::Mxfp4 => DevOp::GemmMxfp4,
        other => panic!(
            "NVIDIA pick_tile: no prefill GEMM opcode for quant {other:?} — refuse rather than \
             emit a gfx950-only rung the sm_90a/sm_120a interpreter would trap on"
        ),
    }
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
    // Public AMD entry — always the gfx950 inventory regardless of ambient emit target.
    with_emit_target_amd(true, || pick_tile(m, n, k, n_cu, quant))
}

/// What DECIDED the tile for this shape: `SkuCalibrated` if a measurement did, `Portable` if
/// the analytical model did.
///
/// Distinct from [`gfx950_measured_rungs`] on purpose, and conflating the two is a bug this
/// tree has now shipped twice. A rung count is a LOOKUP; `select_kernel` uses measurements
/// "only if EVERY candidate has one" (`kernelcaps::select.rs:175`), so 4 measured rungs out of
/// 5 candidates is a fallback that every lookup still reports as a hit. Ask this when the
/// question is "was this build measured", and ask that when the question is "what does the
/// store hold".
pub fn gfx950_prefill_tile_tier(
    m: u32,
    n: u32,
    k: u32,
    n_cu: u32,
    quant: kernelcaps::QuantScheme,
) -> kernelcaps::CalibrationTier {
    // Asked DIRECTLY rather than by differencing `tune_demand::tally()`. The first version of
    // this function did the latter and was racy in exactly the way it was written to avoid:
    // the tally is process-global and sibling test threads bump it between the two reads.
    with_emit_target_amd(true, || pick_tile_tiered(m, n, k, n_cu, quant).1)
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
) -> (DevOp, kernelcaps::CalibrationTier) {
    let (spec, _isa) = amd_target::active();
    let hw = kernelcaps::HardwareFingerprint::from_spec(spec)
        .unwrap_or_else(|| panic!("no hardware fingerprint for {}", spec.name));
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

    // Provenance for build.json: what DECIDED this tile, not what a lookup found. See
    // `tune_demand::note_decision`.
    let tier = realization.rationale.tier();
    tune_demand::note_decision(tier == kernelcaps::CalibrationTier::SkuCalibrated);
    (realization.kernel.0, tier)
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
    // On NVIDIA the fused GemmGlu body is the only prefill GLU+GEMM tile; the gfx950
    // "is 256×256 the winner among GLU-capable rungs" question does not apply — and asking
    // it would refuse fusion whenever Wide/C5 ranked higher, then emit those as separate
    // GEMMs and trap. PLOW_NO_GLU_FUSE=1 opts out (occ-2 experiments: the fused body's
    // 128 f32 accumulators cannot live under a 128-register launch-bounds cap, two plain
    // 64-acc GEMMs can).
    if !emit_is_amd() {
        return !emit_config::active().no_glu_fuse;
    }
    select_gemm_over(
        glu_era_inventory(),
        m,
        n,
        k,
        n_cu,
        kernelcaps::QuantScheme::None,
    )
    .0 == DevOp::Gemm
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
    if !emit_is_amd() {
        return true;
    }
    select_gemm_over(
        glu_era_inventory_mxfp4(),
        m,
        2 * n,
        k,
        n_cu,
        kernelcaps::QuantScheme::Mxfp4,
    )
    .0 == DevOp::GemmMxfp4
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

    fn variant_is_winner(
        &self,
        m: i64,
        n: i64,
        k: i64,
        quant: kernelcaps::QuantScheme,
        measurement_id: u16,
    ) -> bool {
        let Some(costs) = self.by_case.get(&tunedb::gemm_op_case(m, n, k, quant)) else {
            return false;
        };
        let Some(candidate) = costs.get(&measurement_id) else {
            return false;
        };
        costs.values().all(|cost| candidate <= cost)
    }
}

/// Whether the qualified/current c8 measurement wins this exact BF16 shape.
pub fn gfx950_c8_is_measured_winner(m: i64, n: i64, k: i64) -> bool {
    gfx950_gemm_measurements().variant_is_winner(
        m,
        n,
        k,
        kernelcaps::QuantScheme::None,
        tunedb::GEMM_WIDE_C8_MEASUREMENT_ID,
    )
}

/// Qualified, non-stale GEMM measurements for the ACTIVE AMD target.
///
/// MEMOISED PER ISA, for exactly the reason `amd_gemm_inventory` is: a single global
/// `OnceLock` means whichever target resolved FIRST decides what every other target sees. That
/// is worse here than it is for the inventory, because the failure is silent in both
/// directions — a gfx942 compile that ran after a gfx950 one would read gfx950's cell contents
/// under gfx942's cell NAME, and a gfx950 compile that ran second would find gfx942's records
/// and call them stale. `amd_gemm_inventory` already carries this fix and this function, one
/// over, did not.
///
/// In `plowc` today one process compiles one target, so this was latent rather than live. It
/// is not latent in test binaries, which is what made a gfx942 tuning guard impossible to add
/// beside the gfx950 one.
fn gfx950_gemm_measurements() -> &'static GemmMeasurements {
    use std::sync::OnceLock;
    static CDNA3: OnceLock<GemmMeasurements> = OnceLock::new();
    static CDNA4: OnceLock<GemmMeasurements> = OnceLock::new();
    let slot = match amd_target::active().1 {
        hwspec::IsaLevel::Gfx942 => &CDNA3,
        _ => &CDNA4,
    };
    slot.get_or_init(|| {
        let mut by_case: std::collections::HashMap<String, std::collections::HashMap<u16, f64>> =
            Default::default();
        // EMPTY means "no store", which is how `plowc --no-tuning` reaches here. Unset means
        // "the default tree" — the two are deliberately different: a compile that never asked
        // about tuning should still get the calibrated answer, and one that explicitly asked
        // for the analytical model must get it.
        let root = match emit_config::active().tunedb_root() {
            None => return GemmMeasurements { by_case },
            Some(s) => s,
        };
        let store = tunedb::TuneStore::new(std::path::PathBuf::from(root));
        let source_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let Ok(build) = kernelcaps::dense_gemm_tuning_build(&source_root, amd_target::active().1)
        else {
            return GemmMeasurements { by_case };
        };
        // The sweep launches standalone GEMM kernels. Its key covers the
        // preprocessed dense family across all supported encodings, not
        // unrelated arms in the persistent interpreter.
        let want = tunedb::Digests {
            implementation: build.label(),
            interpreter: build.label(),
            toolchain: build.toolchain.clone(),
            oracle: tunedb::GEMM_ORACLE.to_string(),
        };
        let cell = amd_tuning_cell();
        let Ok(records) = store.load_kernels(&cell) else {
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
                cell,
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
    let root = match emit_config::active().tunedb_root() {
        None => {
            packet::devbuild::set_tuned_gemv_cases(cases);
            return;
        }
        Some(s) => s,
    };
    let want = tunedb::Digests {
        implementation: gfx950_gemm_inventory().build().label(),
        interpreter: gfx950_gemm_inventory().build().label(),
        toolchain: gfx950_gemm_inventory().build().toolchain.clone(),
        oracle: tunedb::GEMV_ORACLE.to_string(),
    };
    if let Ok(records) =
        tunedb::TuneStore::new(std::path::PathBuf::from(root)).load_kernels(&amd_tuning_cell())
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
/// THE ACTIVE AMD TARGET.
///
/// `select_gemm_over` used to open with `hwspec::registry::lookup("MI350X")`, so EVERY AMD tile
/// decision was costed against CDNA4 no matter what `--gpu` said: 160 KiB of LDS and double-rate
/// bf16 MFMA, on a part that has 64 KiB and half the MFMA rate. It did not fail, it chose badly
/// and silently -- the exact failure mode `kernelcaps` exists to prevent, one layer up.
///
/// Why ambient state and not a parameter: `pick_tile` is reached from dozens of call sites deep
/// inside the emitters, and threading a `&GpuSpec` through all of them would bury the fix in
/// mechanical churn. It is set once, at emit entry, from the `(arch, gpu)` the caller asked for,
/// and defaults to MI350X/Gfx950 so any path that never sets it behaves exactly as before.
///
/// THREAD-LOCAL, not a global. A global `RwLock` was tried first and is wrong for two reasons
/// that are really one: `cargo test` runs tests in parallel threads of ONE process, so an emit
/// test that set MI300X changed the tile another test was mid-way through asserting -- caught
/// immediately by `every_shape_resolves_on_cost_rather_than_opcode_number`, which costs against
/// its own `lookup("MI350X")` and started disagreeing with `pick_tile`. Emit itself is
/// single-threaded (devgen pulls in no rayon and spawns nothing), so a thread-local is exactly
/// as correct for production and isolates the tests for free.
mod amd_target {
    use hwspec::{GpuSpec, IsaLevel};
    use std::cell::Cell;

    thread_local! {
        static ACTIVE: Cell<Option<(&'static GpuSpec, IsaLevel)>> = const { Cell::new(None) };
    }

    /// Point the AMD emitters at a specific part. Unknown names leave the default in place and
    /// say so, rather than silently costing against the wrong hardware.
    pub fn set(gpu: &str) {
        set_for("", gpu)
    }

    /// The same, with `--arch` as the FALLBACK when `--gpu` cannot answer.
    ///
    /// The `--gpu`-only form had a hole that reads as a warning and behaves as a wrong answer:
    /// an emit with `--arch gfx942` and an absent or unrecognised `--gpu` fell through the
    /// `None` arm, which only `eprintln!`s, and left the MI350X/Gfx950 default active. The
    /// build then costed CDNA3 tiles against 160 KiB of LDS and double-rate MFMA and resolved
    /// its tuning cell to `amd/gfx950/mi350x` — the same silent wrong-arch costing that put a
    /// stale gfx950 cell in front of a gfx942 compile in the first place.
    ///
    /// `--arch` is the authority the caller already stated, so it decides. The representative
    /// part per arch is the one the repo's records are keyed to: gfx942 → MI300X, gfx950 →
    /// MI350X (which is also the pre-existing default, so a caller that names neither is
    /// unchanged). The warning is LOUD and names the part now in force, because "stays on the
    /// default" does not tell a reader which default that is.
    pub fn set_for(arch: &str, gpu: &str) {
        if let Some(spec) = hwspec::registry::lookup(gpu) {
            if let Some(isa) = IsaLevel::from_spec(spec) {
                if matches!(isa, IsaLevel::Gfx942 | IsaLevel::Gfx950) {
                    ACTIVE.with(|a| a.set(Some((spec, isa))));
                    return;
                }
            }
        }
        let why = if gpu.is_empty() {
            "--gpu was not given".to_string()
        } else if hwspec::registry::lookup(gpu).is_none() {
            format!("--gpu {gpu} is not in the hwspec registry")
        } else {
            format!("--gpu {gpu} is not a CDNA part")
        };
        // Fall back to the arch the caller DID state, rather than to a constant.
        let (sku, isa) = match arch {
            "gfx942" => ("MI300X", IsaLevel::Gfx942),
            "gfx950" => ("MI350X", IsaLevel::Gfx950),
            _ => {
                eprintln!(
                    "warning: {why} and --arch {arch:?} names no CDNA part; AMD tiles will be \
                     costed against MI350X (gfx950, 160 KiB LDS, double-rate MFMA) and the \
                     tuning cell resolved to {}. Pass --gpu.",
                    tunedb::GFX950_CELL
                );
                return;
            }
        };
        let spec = hwspec::registry::lookup(sku).expect("representative part in registry");
        ACTIVE.with(|a| a.set(Some((spec, isa))));
        eprintln!(
            "warning: {why}; costing AMD tiles against {sku} as the representative {} part. \
             Pass --gpu to name the actual SKU — the tuning cell follows it.",
            isa.arch_flag()
        );
    }

    /// The part AMD tiles are costed against. MI350X when nothing was set.
    pub fn active() -> (&'static GpuSpec, IsaLevel) {
        if let Some(t) = ACTIVE.with(|a| a.get()) {
            return t;
        }
        let spec = hwspec::registry::lookup("MI350X").expect("MI350X in registry");
        (spec, IsaLevel::Gfx950)
    }
}

/// Point the AMD emitters at `gpu` (see [`amd_target`]).
pub fn set_amd_target(gpu: &str) {
    amd_target::set(gpu);
}

/// Point the AMD emitters at `(arch, gpu)`, with `arch` as the fallback (see
/// [`amd_target::set_for`]). This is what the two emit entries call: they already know both.
pub fn set_amd_target_for(arch: &str, gpu: &str) {
    amd_target::set_for(arch, gpu);
}

#[cfg(not(test))]
fn gfx950_gemm_inventory() -> &'static kernelcaps::Inventory {
    amd_gemm_inventory(amd_target::active().1)
}

/// The tuning cell for the ACTIVE AMD target, e.g. `amd/gfx942/mi300x`.
///
/// A CONSTANT here (`tunedb::GFX950_CELL`) meant every target read MI350X's cell: a gfx942
/// compile probed a gfx942 build digest, compared it against gfx950's records, found all 3080
/// stale and fell back to the analytical model -- while reporting the mismatch as staleness
/// rather than as "you are reading another GPU's cell". The digest already follows
/// `amd_target::active()`; this makes the cell follow it too, so the two agree.
///
/// The RULE itself is `tunedb::amd_tuning_cell` and not written out here, because the campaign
/// binaries that WRITE records need the same answer this reader looks up — `tunedb-gemv` had it
/// hardcoded to gfx950 and would have published a gfx942 campaign into a cell this function
/// never opens.
fn amd_tuning_cell() -> String {
    tunedb::amd_tuning_cell(amd_target::active().0)
}

#[derive(Clone, Debug)]
struct AttentionDecisionReport {
    hardware: String,
    n_cu: u32,
    decode_rung: u32,
    kv_bucket: &'static str,
    shape: String,
    compiled_max_nsplit: u32,
    compiled_persistent: bool,
    selected_nsplit: u32,
    selected_algorithm: &'static str,
    selected_source: &'static str,
}

thread_local! {
    static ATTENTION_DECISIONS: std::cell::RefCell<Vec<AttentionDecisionReport>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

fn clear_attention_decisions() {
    ATTENTION_DECISIONS.with(|d| d.borrow_mut().clear());
}

fn attention_decisions() -> Vec<AttentionDecisionReport> {
    ATTENTION_DECISIONS.with(|d| d.borrow().clone())
}

/// Exact-cell attention selection. This is compile-time packet policy, not an
/// online tuner: only qualified records for this arch/CU/rung/KV bucket and
/// operator geometry can replace the fixed fallback.
fn select_amd_attention(
    n_cu: u32,
    decode_rung: u32,
    kv_len: u32,
    shape: String,
    max_nsplit: u32,
    fallback_nsplit: u32,
) -> tunedb::AttentionSelection {
    let fallback = || tunedb::AttentionSelection {
        algorithm: tunedb::AttentionAlgorithm::SplitReduce,
        nsplit: fallback_nsplit.clamp(1, max_nsplit.max(1)),
        source: tunedb::AttentionSource::FixedFallback,
    };
    let hardware = amd_tuning_cell();
    let records = emit_config::active()
        .tunedb_root()
        .and_then(|root| {
            tunedb::TuneStore::new(std::path::PathBuf::from(root))
                .load_attention(&hardware)
                .ok()
        })
        .unwrap_or_default();
    #[cfg(test)]
    let want = tunedb::Digests {
        implementation: "test-unprobed".into(),
        interpreter: "test-unprobed".into(),
        toolchain: "test-unprobed".into(),
        oracle: tunedb::ATTENTION_ORACLE.into(),
    };
    #[cfg(not(test))]
    let want = tunedb::Digests {
        implementation: gfx950_gemm_inventory().build().label(),
        interpreter: gfx950_gemm_inventory().build().label(),
        toolchain: gfx950_gemm_inventory().build().toolchain.clone(),
        oracle: tunedb::ATTENTION_ORACLE.into(),
    };
    let cell = tunedb::AttentionCell {
        hardware,
        n_cu,
        decode_rung,
        kv_bucket: tunedb::KvBucket::of(kv_len),
        shape,
    };
    let selected = if records.is_empty() {
        fallback()
    } else {
        tunedb::select_attention(
            &records,
            &cell,
            &want,
            tunedb::AttentionCapabilities {
                max_nsplit,
                // No distinct persistent-attention body is compiled in the AMD
                // interpreter today. A record requesting one stays ineligible.
                persistent: false,
            },
            fallback_nsplit,
        )
    };
    if selected.source == tunedb::AttentionSource::Qualified {
        eprintln!(
            "  attention tuned: {} -> {:?}/ns{}",
            cell.key(),
            selected.algorithm,
            selected.nsplit
        );
    }
    ATTENTION_DECISIONS.with(|d| {
        d.borrow_mut().push(AttentionDecisionReport {
            hardware: cell.hardware.clone(),
            n_cu,
            decode_rung,
            kv_bucket: cell.kv_bucket.label(),
            shape: cell.shape.clone(),
            compiled_max_nsplit: max_nsplit,
            compiled_persistent: false,
            selected_nsplit: selected.nsplit,
            selected_algorithm: match selected.algorithm {
                tunedb::AttentionAlgorithm::SplitReduce => "split_reduce",
                tunedb::AttentionAlgorithm::Persistent => "persistent",
            },
            selected_source: match selected.source {
                tunedb::AttentionSource::FixedFallback => "fixed_fallback",
                tunedb::AttentionSource::Qualified => "qualified",
            },
        });
    });
    selected
}

/// The prefill GEMM inventory for `isa`, probed once per ISA.
///
/// Memoised per level rather than once globally: the rungs carry their ISA and `runs_on` demands
/// an exact match, so one shared inventory means whichever target ran first decides what every
/// other target is allowed to select.
#[cfg(not(test))]
fn amd_gemm_inventory(isa: hwspec::IsaLevel) -> &'static kernelcaps::Inventory {
    use std::sync::OnceLock;
    static CDNA3: OnceLock<kernelcaps::Inventory> = OnceLock::new();
    static CDNA4: OnceLock<kernelcaps::Inventory> = OnceLock::new();
    let cell = match isa {
        hwspec::IsaLevel::Gfx942 => &CDNA3,
        _ => &CDNA4,
    };
    cell.get_or_init(|| {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        match kernelcaps::dense_gemm_inventory(&root, isa) {
            Ok(inv) => inv,
            Err(e) => {
                eprintln!(
                    "warning: cannot probe {} kernel inventory ({e}); \
                     using analytical fallback (known tile constants)",
                    isa.arch_flag()
                );
                amd_analytical_inventory(isa)
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

/// [`GFX950_RUNGS`] as `KernelSpec`s, tagged with the encoding each serves and the ISA they are
/// being offered for.
///
/// The `mma_dtype` for mxfp4 is bf16 and not fp4: the fp4 prefill GEMM is w4a16 and dequantizes
/// in the B-fetch, so the matrix instruction it issues is the ordinary bf16 MFMA. Mirrors
/// `kernelcaps::targets::GFX950_QUANT_OBJECTS`, which is what the real probe uses.
///
/// The tag is load-bearing, not decoration: `KernelSpec::runs_on` requires the kernel's ISA to
/// EQUAL the target fingerprint's, so a rung tagged Gfx950 is invisible to an MI300X target and
/// `select_kernel` fails with "no rung" rather than picking badly. That is the correct behaviour
/// for a kernel that genuinely does not exist on the part -- and the wrong one here, where the
/// same `op_gemm.h` builds for both.
fn amd_rung_specs(build_label: &str, isa: hwspec::IsaLevel) -> Vec<kernelcaps::KernelSpec> {
    use hwspec::MmaDtype;
    use kernelcaps::{KernelSpec, QuantScheme};
    let mut out = Vec::with_capacity(GFX950_RUNGS.len() * 3);
    for (bf16, fp8, mx, bm, bn, bk) in GFX950_RUNGS {
        for (op, quant, mma) in [
            (bf16, QuantScheme::None, MmaDtype::Bf16),
            (fp8, QuantScheme::W8A8, MmaDtype::Fp8),
            (mx, QuantScheme::Mxfp4, MmaDtype::Bf16),
        ] {
            let body = format!("{}:{}@{build_label}", isa.arch_flag(), op.c_name());
            out.push(KernelSpec::gemm_tile(op, isa, bm, bn, bk, &body).with_quant(quant, mma));
        }
    }
    out
}

/// Analytical fallback inventory for `isa`, used when the probe cannot run (no hipcc).
fn amd_analytical_inventory(isa: hwspec::IsaLevel) -> kernelcaps::Inventory {
    let build = kernelcaps::BuildId::new(
        isa,
        ["PLOW_BUCKET_DECODE=0".to_string()],
        "analytical-fallback",
        "analytical-fallback",
    );
    let specs = amd_rung_specs(&build.label(), isa);
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
    amd_gemm_inventory(amd_target::active().1)
}

#[cfg(test)]
fn amd_gemm_inventory(isa: hwspec::IsaLevel) -> &'static kernelcaps::Inventory {
    use std::sync::OnceLock;
    static CDNA3: OnceLock<kernelcaps::Inventory> = OnceLock::new();
    static CDNA4: OnceLock<kernelcaps::Inventory> = OnceLock::new();
    let cell = match isa {
        hwspec::IsaLevel::Gfx942 => &CDNA3,
        _ => &CDNA4,
    };
    cell.get_or_init(|| {
        let build = kernelcaps::BuildId::new(
            isa,
            ["PLOW_BUCKET_PREFILL=1".to_string()],
            "test-fixture",
            "test-fixture",
        );
        let specs = amd_rung_specs(&build.label(), isa);
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
    // NRN-fold residual ping-pong twin of `x` (Gemma fp8 decode on AMD only; TENSOR_NONE
    // otherwise so every other blob stays byte-identical). NRN1 writes its residual here and
    // the folded NRN2 inside the next layer's q/k/v GemvFp8s reads it back — the two buffers
    // alternate at every half-layer because the q/k/v trio runs CONCURRENTLY and an in-place
    // residual store would race its siblings' loads. See gemv_nrn_lds in op_gemm.h.
    xr: u32,
    // [MERGE-FOLD] per-(batch, head-group) done counters for the in-flash split merge (u32,
    // zeroed at load, SELF-CLEANING — the merging workgroup resets its slot before the packet's
    // own completion signal, and the next token's flash sits transitively behind that signal).
    // One shared tensor across layers for the same serial-chain reason. TENSOR_NONE unless the
    // fold is enabled, so every other blob stays byte-identical.
    mrgc: u32,
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
    // TP (n_gpu>1) peer-mapped partial slots: the row-parallel
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
    // Grouped-MoE PREFILL scratch. Declared only when moe && moe_pf;
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
    nrn_fold: bool,
    merge_fold: bool,
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
    // TP head split: each rank owns heads/N q-heads and kvh/N kv-heads,
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
        head8: if c.tied && emit_config::active().fp8_head {
            b.tensor(
                &format!("fp8/{}embed_tokens.weight", c.prefix),
                (c.vocab * c.hidden) as u64,
            )
        } else {
            TENSOR_NONE
        },
        head8s: if c.tied && emit_config::active().fp8_head {
            b.tensor(
                &format!("fp8/{}embed_tokens.weight_scale", c.prefix),
                c.vocab as u64 * F32,
            )
        } else {
            TENSOR_NONE
        },
        x: ac(b, "x", (rows * c.hidden) as u64 * BF16),
        xr: if nrn_fold {
            ac(b, "xr", (rows * c.hidden) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        // Sized b × heads u32s — an upper bound on b × n_grp (n_grp = heads/GF), so every
        // (b, head-group) slot the kernel indexes exists whatever GF the hd selects.
        mrgc: if merge_fold {
            ac(b, "mrgc", (dbatch.max(1) * c.heads) as u64 * 4)
        } else {
            TENSOR_NONE
        },
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
        // Per-layer head split with SHARED-KV-HEAD REPLICATION.
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

/// The GEMM arena the DECODE OBJECT will actually have, in halves, for the part being emitted for.
///
/// # This constant was the CDNA4 one on every part, and on gfx942 that is a silent-corruption bug
///
/// `gemv_qkv_rows` / `gemv_glu_rows` read `x` ONLY through LDS — `op_gemm.h` says so outright
/// ("x is ALWAYS staged in LDS here: plowc emits this op only when M*K fits GM_LDS_HALVES") — so
/// the emitter picking those fused opcodes is a PROMISE that `M*K` fits. On gfx942 it did not:
///
/// | part | decode tile | buffers | arena (halves) |
/// |---|---|--:|--:|
/// | gfx950 / sm_120 | 256x256x64 | 2 | 73,728 |
/// | gfx942, `scripts/build_gfx942.sh` default | 192x256x32 | 1 | 17,920 |
/// | gfx942, `PLOW_OCC4=1` (the SHIPPED decode profile) | 128x256x32 | 1 | **15,360** |
///
/// So an emitter believing 73,728 fuses up to `M*K = 73728`, and at Gemma-4-12B's `hidden = 3840`
/// that is every batch up to 19 — while the object it runs on holds four rows. Rows past the
/// arena are written past the end of `plow_smem`. Measured on this box: a `PLOW_DECODE_BATCH=16`
/// blob takes `HSA_STATUS_ERROR_EXCEPTION` at three concurrent requests on the occ4 objects, and
/// produces fluent-but-wrong text on the wider default tile. It is the same bug §6g-BATCH found
/// at `hidden = 5376` on gfx950 (slots 13/14/15 fluent-but-WRONG), with the gfx942 arena.
///
/// **gfx942 takes the OCC4 value, not the default tile's**, and deliberately: the object profile
/// is chosen when the OBJECTS are built (`PLOW_OCC4=1`, which is the shipped gfx942 decode
/// recipe), long after the blob is emitted, so the emitter cannot know which it will meet. The
/// smaller of the two is the only value that is right for both.
///
/// Every gfx950 and NVIDIA emit is UNCHANGED. So is every gfx942 emit at `PLOW_DECODE_BATCH=1`,
/// because `1 * hidden` fits both arenas for every model in the tree — which is why this went
/// unnoticed: batched decode had only ever been serve-gated on gfx950.
///
/// The table itself now lives in `hwspec::IsaLevel::geometry` — the fix above was a local
/// `match` here, which removes THIS bug and not its class. See [`arch_geometry`].
pub(crate) fn gm_lds_halves() -> u64 {
    arch_geometry().decode_arena_halves()
}

/// The device geometry of the part being emitted for. `hwspec` owns the table and
/// `crates/hwspec/tests/device_header_agreement.rs` holds it against `op_gemm.h` and the build
/// scripts, so this file no longer carries a second copy to go stale.
///
/// The AMD emitters only ever run with a CDNA target set ([`amd_target::active`] falls back to
/// MI350X), so the panic is unreachable rather than defensive.
pub(crate) fn arch_geometry() -> hwspec::ArchGeometry {
    let isa = amd_target::active().1;
    isa.geometry()
        .unwrap_or_else(|| panic!("no ArchGeometry for {}", isa.arch_flag()))
}

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
/// combination is the whole subject of the design notes.
fn gemv_row_bucket(t: u32) -> u32 {
    if let Some(v) = emit_config::active().gemv_mm {
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
    if emit_config::active().gemv_walk {
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
    let v = emit_config::active()
        .max_chunk
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

/// This rank's local KV-head count under TP with SHARED-KV-HEAD REPLICATION.
/// Two regimes, both keep every rank's q-heads mapped to a kv-head it owns:
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
    emit_config::active().flash_merge_dsplit.unwrap_or(1).max(1)
}

#[cfg(test)]
#[path = "lib_tests/flash_merge_map.rs"]
mod flash_merge_map_tests;

/// Emit the layer all-reduce for a row-parallel producer (o_proj/down), all-reduce #1/#2.
/// PREFILL uses the TWO-SHOT (reduce-scatter + all-gather): the [T,hidden] partial is
/// bandwidth-bound, so ~N/2× less fabric than the one-shot. DECODE
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

/// ONE BAND of a row-banded prefill TP seam: a two-shot all-reduce over `elems` elements of
/// the partial, at byte offset `slot` in the peer window and element offset `e0` into `out`.
///
/// This is what lets a collective take advantage of plow's counter design instead of
/// waiting on the whole tensor: the emitter splits the producer into K row-band packets
/// (the tiled GEMMs carry `a_row0`/`c_row0`, MoeCombinePf carries `t_row0` — all pure
/// pointer arithmetic) and each band's two-shot deps ONLY on its own band's producer, with
/// its OWN gate pair. Band 0's fabric transfer then runs while bands 1..K-1 still compute —
/// the TP data movement pipelines ahead of the compute instead of behind it. Disjoint rows,
/// same tiles, same per-element sum order: BIT-IDENTICAL to the unbanded emit.
///
/// Why not `Dep::Fine` on one unsplit producer: the tiled GEMM's workgroups claim tiles
/// round-robin across the whole M range, so every band would depend on nearly every
/// producer slice — the same collapse the M0 census measured. Splitting the PACKET is what
/// makes the edge genuinely band-structured.
#[allow(clippy::too_many_arguments)]
fn emit_xreduce_twoshot_band(
    b: &mut Builder,
    xgate: &mut u32,
    xr_cus: &[u32],
    deps: &[u32],
    out: u32,
    elems: u32,
    tp: u32,
    slot: u32,
    e0: u32,
    // Fused-residual operands `(resid, out2)` — see the kernel's fused-residual note: the
    // all-gather writes `out2 = bf16(resid + reduced)` and the trailing Residual packet is
    // not emitted at all. Bit-identical to d_residual at scale=1.
    res: Option<(u32, u32)>,
) -> u32 {
    // Same saturation rule as emit_xreduce_gather: never more workgroups than elements/512.
    let need = (elems.div_ceil(512).max(1) as usize).min(xr_cus.len());
    let xr_cus = &xr_cus[..need];
    let gate_rs = *xgate;
    *xgate += 1;
    let gate_ag = *xgate;
    *xgate += 1;
    b.emit(DevOp::XReduceTwoShot, xr_cus.to_vec(), deps, |d| {
        d.t[0] = out;
        if let Some((r, o2)) = res {
            d.t[1] = r;
            d.t[2] = o2;
        }
        d.i[0] = elems;
        d.i[1] = tp;
        d.i[2] = slot; // the REGION base (0 or slot_b) — never band-advanced: the loader
                       // infers the blob's slot_bytes as max(i2) (asset/devblob.rs), so a
                       // band offset here would inflate it and shift the second partial's
                       // binding. The dispatch derives the band's window offset from e0.
        d.i[3] = gate_rs;
        d.i[4] = gate_ag;
        d.i[5] = e0; // this band's element offset into `out` AND (×2 B) into the window
    })
}

/// [`emit_xreduce`], plus an ALL-GATHER of a column-parallel partial folded into the same
/// packet: `out = sum_r reduced_r + concat_r gathered_r`.
///
/// `gather` is `(slot byte offset, per-rank column count, out row width)`. See
/// [`packet::dev::DevOp::XReduce`] for why the two collectives are one packet — the two
/// partials are ADDED, so the gather is one extra bf16 load per element on a rendezvous
/// that already happened, rather than its own packet (~5.3 us) and its own rendezvous.
///
/// A large prefill gather may opt into the two-shot path. Its all-gather already visits every
/// final element, so it can add the owner-derived second partial there without another packet
/// or rendezvous. The shape gate requires a complete column partition (`row_w = tp*gcols`).
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
    let xr2_gather = !decode
        && emit_config::active().xr2_gather
        && gather.is_some_and(|(_, gcols, row_w)| row_w == tp.saturating_mul(gcols));
    if decode || (gather.is_some() && !xr2_gather) {
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
            if let Some((gslot, gcols, _)) = gather {
                d.i[6] = gslot; // folded all-gather partial slot
                d.i[7] = gcols; // columns per rank; row width = n_gpu*gcols
            }
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
/// kernels. See the design notes.
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
    !emit_config::active().gemma_moe_router_fused
}

/// `nrow` = decode batch B: the score work space is the (row, expert) PAIR space, so B rows
/// scale the useful CTA count (16 CTAs at B=1/E=128, capped at n_cu from B=12 up).
fn gemma_moe_router_split_plan(n_cu: u32, n_exp: u32, nrow: u32) -> Option<(u32, DevOp)> {
    if !gemma_moe_router_split_enabled() {
        return None;
    }
    let max_useful = (nrow * n_exp).div_ceil(8).max(1).min(n_cu.max(1));
    let blocks = emit_config::active()
        .gemma_moe_router_blocks
        .unwrap_or(max_useful)
        .clamp(1, max_useful);
    let op = if emit_config::active().gemma_moe_router_exact {
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
    // Shared GEN_TMAP_BF16 mint registry (see TmapMint). Inert unless PLOW_TMA_GEMM=1.
    tmaps: &std::cell::RefCell<TmapMint>,
) {
    // The two axes the old `decode` bool used to carry at once. Every former use site below is
    // now one or the other: `decode` for shape, `gemv_family` for kernel family. (Not `gemv` —
    // the `hn_dep` closure below already binds a `gemv: u32` parameter that would shadow it.)
    let decode = mode.decode_shape();
    let gemv_family = mode.gemv();
    // A decode LADDER makes every rung — the one-row rung included — address the KV cache per
    // sequence out of `pos[]`. See the two `i[6] = n_batch_kv` sites below for why.
    let seq_rows = emit_config::active().decode_ladder_on();
    // Fused w8a8 activation quant (PLOW_FUSE_QUANT=0 opts out): the two hidden-width
    let all = b.all();
    // TENSOR-PARALLEL local shards. For tp==1 these equal the full dims,
    // so the whole emit is byte-identical to the pre-TP path; for tp>1 (decode only) every head-,
    // intermediate- and vocab-dimensioned op runs 1/N wide, and o_proj/down get an XReduce.
    let tp = c.tp;
    let heads = c.heads / tp; // this rank's q-heads
    let inter_l = c.inter / tp; // this rank's gate/up/down intermediate lanes
    let vocab_l = c.vocab; // lm_head REPLICATED under TP (Phase 2); see declare() note above
    let mut xgate: u32 = 0; // xctr gate-id allocator for XReduce (unique per collective)
                            // XReduce runs on a REDUCED CU set (F-lever). The all-reduce is a
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
        let k = emit_config::active().xr_cus.unwrap_or(32).clamp(1, n_cu);
        (0..k).collect()
    };
    // TP prefill: the all-reduce partials are [T, hidden], not decode's
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
    // sm_90a TMA prefill GEMM (PLOW_TMA_GEMM=1 emit opt-in; pairs with the cubin's
    // PLOW_NV_TMA_GEMM): mint one GEN_TMAP_BF16 descriptor tensor per (target, rows, K)
    // and thread its handle through the GEMM's spare i6/i7 words (dev_isa.h GEMM doc).
    // `rows` is what the op touches — TMA zero-fills the box tail past globalDim, so a
    // shorter-rows map stays correct for a shorter chunk. Unset (default): packets are
    // byte-identical. Handles come from the SHARED [`TmapMint`] registry, NOT this
    // program's Builder: each program adopts a CLONE of the declared tensor list, so a
    // builder-local tensor_gen would die with the program (measured: n_tensor stayed at
    // the declare()-time count and every map decl vanished from the blob). run_verified
    // folds the registry into the Model after all programs are emitted.
    let tma_gemm = emit_config::active().tma_gemm;
    let tmap = |target: u32, rows: u32, k: u32| -> u32 {
        tmaps.borrow_mut().handle(target, rows, k, false)
    };
    let tmap8 = |target: u32, rows: u32, k: u32| -> u32 {
        tmaps.borrow_mut().handle(target, rows, k, true)
    };
    let tmap_kv = |kt: u32, vt: u32, ring: u32, hd: u32, nkv: u32| -> u32 {
        tmaps.borrow_mut().kv_pair(kt, vt, ring, hd, nkv)
    };

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
            return b.emit(DevOp::GemvFp8, gemv_wg_cap(cus), deps, |d| {
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
            // sm_90a TMA (see `tmap` above): w8a8 only — both operands are e4m3 tensors
            // the TMA e4m3 maps can describe. The w8a16 body keeps cp.async (its A is
            // bf16 and its weight is dequanted in-kernel; no TMA arm exists for it).
            let tm8 = (tma_gemm
                && w8a8
                && matches!(op, DevOp::GemmFp8 | DevOp::GemmMedFp8 | DevOp::GemmSmallFp8))
            .then(|| (tmap8(xq, m, k), tmap8(w8, nn, k)));
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
                if let Some((ma, mb)) = tm8 {
                    d.i[6] = ma;
                    d.i[7] = mb;
                }
            });
        }
        let fold = gemv_family && gamma != TENSOR_NONE;
        let op = if gemv_family {
            DevOp::Gemv
        } else {
            pick_tile(m, nn, k, n_cu, kernelcaps::QuantScheme::None)
        };
        // Only the three plain-tile rungs have the sm_90a TMA arm; Wide/C5 (gfx950) and
        // GLU/fp8 keep i6/i7 zero until their forks land.
        let tm = (tma_gemm && matches!(op, DevOp::Gemm | DevOp::GemmMed | DevOp::GemmSmall))
            .then(|| (tmap(a, m, k), tmap(w, nn, k)));
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
            if let Some((ma, mb)) = tm {
                d.i[6] = ma;
                d.i[7] = mb;
            }
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
    // T11 QUANT-INTO-NORM (PLOW_QNORM_FUSE=1, prefill w8a8 only): the two hidden-width
    // activation quants that directly follow an RmsNorm (pre-qkv, pre-gate/up) ride the norm
    // packet's registers instead of re-reading the normed row from HBM as a separate
    // QuantFp8 packet (t3=xq, t4=ascale on the RmsNorm — see d_rmsnorm). Deletes a packet +
    // gate + a full activation read per site per layer. PAIRING: needs a cubin whose RMSNORM
    // arm reads t3/t4 (this campaign's); an older cubin ignores them and the GEMMs read a
    // stale xq — same cubin/packet pairing contract as PLOW_W8A8 itself.
    // Two spellings, one fold: PLOW_QNORM_FUSE=1 is the opt-in either backend can take;
    // on AMD the fold is DEFAULT-ON (opt out with PLOW_FUSE_QUANT=0) — that default is the
    // measured Gemma <=4k TTFT win (PR #56) and every gfx942 object since carries the
    // RMSNORM t3/t4 arm.
    // IT PRODUCED WRONG OUTPUT ON gfx942, AND THE FAULT WAS NOT THIS NORM FOLD. Recorded because
    // the symptom pointed at the wrong half for a while. Gemma-4-12B fp8/w8a8, gfx942:
    //
    //   fold ON                "capital of France" -> ',1___....1.111111111111'
    //   PLOW_FUSE_QUANT=0      "capital of France" -> 'Paris'
    //
    // `qnorm_fuse` gates TWO folds, not one. The RMSNORM t3/t4 fold below is the one it is named
    // for and it was always fine. The other is the GLU-into-quant fold further down (QuantFp8
    // t3=gate t4=up i2=act), which DELETES the `Glu` packet — and the AMD dispatch ignored t3/t4,
    // so it quantized an `fu` nothing had written. NVIDIA honoured them; AMD never did.
    //
    // Why it looked arch-specific: the GLU fold is only taken when `gemm_glu` is false, i.e. when
    // the fused GLU prefill GEMM does NOT win the tile selection. On both AMD parts that is the
    // SMALLEST prefill bucket only (t=128 here) — so short prompts broke and the long-prompt TTFT
    // benchmarks that motivated PR #56 never touched the broken arm. gfx950 was not immune, it was
    // untested at that length.
    //
    // Fixed in `d_quant_fp8` (runtime/amd/op_gemm.h) with an unconditional
    // `plow_t11_gluquant_arm` marker + a `PLOW_T11_GLUQUANT` entry in the packet's `requires`, so
    // an object predating the fix is REFUSED at load instead of serving garbage.
    //
    // GLM/Kimi/DeepSeek/Nemotron ARE NOT AFFECTED, by construction: `qnorm_fuse` is local to
    // `emit_phase`, which only the dense-GQA family reaches (`mla.rs` and `k3.rs` contain zero
    // calls to it), and those families do not set `w8a8` at all — GLM's fp8 is block-scaled via
    // `glm_linear_fp8`. The blast radius is dense GQA (Gemma-4 / Llama / Qwen) at w8a8.
    let qnorm_fuse = w8a8
        && !gemv_family
        && (emit_config::active().qnorm_fuse || (amd && emit_config::active().fuse_quant));

    // Qwen/Llama PRE-NORM decode fuses each (residual add, RMSNorm) pair into ONE AddNorm packet
    // (see the AddNorm emits in the loop). Deletes 72 packets/token and, more importantly, 72
    // global gates off the critical path — decode here is fixed per-gate tax, not weight streaming.
    let fuse_norm = c.arch != Arch::Gemma4 && gemv_family;
    // Gemma SANDWICH decode fuses each (NormResidual, following RMSNorm) pair into ONE
    // NormResidualNorm packet (Experiment N1) — the narrow→narrow successor to AddNorm. Same two
    // sites as fuse_norm (post-attn→pre-ffn, and end-of-layer→next input norm), but the residual is
    // a post-normed sandwich add with a per-layer scale, not a plain sum. Deletes a gate + an HBM
    // round trip per fused pair. Decode only; prefill keeps the split (T rows parallelise the norm).
    // PLOW_PF_GFUSE=1 extends the sandwich fusion to PREFILL: the "T rows parallelise the
    // norm" rationale addressed serialization, but the GH200 per-op trace showed the real
    // cost is the extra HBM round trip + packet per pair, which holds at any T (norms ~9%
    // of a 4k chunk). Opt-in until token-gated on hardware.
    let gfuse = c.arch == Arch::Gemma4 && (gemv_family || emit_config::active().pf_gfuse);
    // NRN2 -> q/k/v FOLD (Experiment N2, the item "norm fusion capped at 0.43 ms by t[8]"):
    // delete the END-OF-LAYER NormResidualNorm packet by computing it inside the NEXT layer's
    // q/k/v GemvFp8 staging (op 30 i3; gemv_nrn_lds in op_gemm.h — bit-exact replication of
    // op 23's `fits` arm). Each of the trio computes the NRN redundantly; the q packet's slice 0
    // stores the residual, into the PING-PONG twin `n.xr` <-> `n.x` because the trio is
    // concurrent and an in-place store would race the siblings' loads. One fewer serial gate on
    // the 9-deep decode chain per layer. NRN1 cannot fold the same way: GemvGluFp8 has two free
    // t slots against the four operands. `n.xr` is TENSOR_NONE unless declare() enabled the
    // fold (AMD + fp8 + dense Gemma + env), and the shape bounds mirror the kernel's `fits`
    // preconditions, which the kernel re-checks and TRAPS on rather than staging garbage.
    let fuse_nrn = gfuse
        && fp8
        && amd
        && !block_mode
        && n.xr != TENSOR_NONE
        && (c.hidden & 7) == 0
        && c.hidden <= 16 * 512 /* RN_REG * PLOW_THREADS */
        && (gemv_staged_rows(t) as u64 * c.hidden as u64 + 16) <= gm_lds_halves();
    // (b tensor, gamma_b, gamma_n, layer_scale) of a skipped NRN2, consumed by the next
    // iteration's q/k/v emission. Crosses loop iterations by construction.
    let mut nrn_pending: Option<(u32, u32, u32, f32)> = None;

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
            && (gemv_staged_rows(t) as u64 * c.hidden as u64) <= gm_lds_halves()
            && !emit_config::active().no_fuse_qkv;
        // FUSED Q|K|V, per-channel fp8 (DevOp::GemvQkvFp8, op 115) — the arm the comment above
        // called "opcode 26 deferred", landed but OFF BY DEFAULT, because it MEASURES SLOWER.
        // Same output-axis merge and same LDS precondition as the bf16 form; the three f32
        // dequant-scale rows ride in i5/i6/i7 as TENSOR HANDLES, GemvQkvMxfp4's demotion.
        //
        // MEASURED (MI300X gfx942, Gemma-4-12B fp8 occ4, global queue, 48 steps, 3 interleaved
        // reps, ms/token): split3 12.080 vs fused 12.232 at ctx 4096 (+1.3%), 12.150 vs 12.273
        // at 8192 (+0.9%) — token-IDENTICAL on the serve gate, so this is schedule, not math.
        // Same verdict the T11 probe recorded for the bf16 form ("the split version measured
        // faster"), and the mechanism is the same: the traced q/k/v packets already start within
        // 0.3 us of each other on disjoint CU sets, so the two deleted gates were free — while
        // fusing coarsens the gemv->headnorm dependency from the fine per-head producer map to a
        // wait on the whole 304-WG packet. The op stays in the ISA (correct, golden-tested,
        // byte-exact to split3) because the trade flips where the fine map does not exist:
        // the static scheduler, and batched decode. Opt in with PLOW_FUSE_QKV_FP8=1.
        // AMD-ONLY: no sm_120 arm exists for op 115, and check_gfx950_opcode_coverage guards only
        // the AMD side, so the gate lives here.
        let fuse_qkv_fp8 = gemv_family
            && !keqv
            && fp8
            && amd
            && (gemv_staged_rows(t) as u64 * c.hidden as u64) <= gm_lds_halves()
            && emit_config::active().fuse_qkv_fp8;

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
        let mul: u32 = emit_config::active().ns_mul.unwrap_or(mul_default);
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
        // THE fp8_kv GATE IS DROPPED, and the measurement that forced it is on BF16 KV.
        //
        // The cap's own argument -- a sliding layer's flash span never exceeds `win` rows, so
        // splitting it below 64 rows/item is a FIXED per-token waste -- says nothing about the KV
        // DTYPE. The gate was conservatism ("flag-unset packets stay byte-identical"), and it left
        // the common Gemma-4 config (fp8 WEIGHTS, bf16 KV) on the uncapped path, where the
        // grid-alignment above rounds nsplit up to n_cu/gcd(n_grp,n_cu) = 304/8 = 38 on MI300X --
        // 27 rows of a 512-row FA_DEC_TILE.
        //
        // MEASURED on MI300X, Gemma-4-12B fp8 weights + BF16 KV, L2-placed blob + PLOW_GATE_HIER +
        // FA_DEC_ILV, `amd-bench --steps 48`, 3 reps, via the PLOW_NS_ABS override (ms/token):
        //
        //     ctx     ns8      ns16     ns38 (grid-aligned, was shipped)
        //     4096    12.433   12.132   12.428
        //     8192    12.730   12.325   12.490
        //
        // `win/64` at win=1024 is exactly 16 -- the cap was already computing the right number and
        // was simply switched off for this dtype. ns8 LOSES, reproducing the note above that
        // split-KV parallelism, not merge cost, is the ceiling.
        //
        // SCOPE: the bf16-KV extension is keyed to the ACTIVE gfx942 target (where the table
        // above was measured). gfx950 and sm_120 keep the fp8_kv-gated cap and stay
        // byte-identical (the dense goldens pin this); extend the key when the cap is
        // re-measured on those parts — the fp8-KV sweep above picks the same value, so the
        // extension is expected to hold.
        let cap_bf16_kv = amd && amd_target::active().1 == hwspec::IsaLevel::Gfx942;
        let ns = if gemv_family && !full && win > 0 && (fp8_kv || cap_bf16_kv) {
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
        let ns = emit_config::active()
            .ns_abs
            .filter(|_| gemv_family)
            .unwrap_or(ns);
        // Full-attention-only decode split override. Unlike PLOW_NS_ABS this does not also
        // over-split Gemma's many hd256 sliding layers. It is the controlled sweep knob for
        // full-layer GQA-fusion experiments on sm_120 (GF4/ns24 => 8 groups * 24 = 192 work
        // items on the 188-SM RTX PRO 6000). Default unset preserves every existing packet.
        let ns = emit_config::active()
            .ns_full_abs
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
                if qnorm_fuse {
                    d.t[3] = n.xqh; // fused w8a8 activation quant (T11): xq out
                    d.t[4] = n.ash; //   + per-row a_scale out
                }
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
        if fuse_qkv || fuse_qkv_fp8 {
            // ONE packet on all CUs: cols [0,qd) -> q, [qd,qd+kd) -> k, [qd+kd,qd+2kd) -> v.
            (nq, nk, nv) = (n_cu, n_cu, n_cu); // unused: fused headnorm deps are coarse
            let fused = if fuse_qkv_fp8 {
                b.emit(DevOp::GemvQkvFp8, all.clone(), &[c_n], |d| {
                    d.t[0] = n.qg;
                    d.t[1] = qkv_src;
                    d.t[2] = w.wq8;
                    d.t[3] = n.kg;
                    d.t[4] = w.wk8;
                    d.t[5] = n.vg;
                    d.t[6] = w.wv8;
                    d.i[0] = t;
                    d.i[1] = qd;
                    d.i[2] = c.hidden;
                    d.i[3] = kd;
                    d.i[4] = kd;
                    // The tenth, eleventh and twelfth pointers: the three per-channel f32
                    // dequant-scale rows, as handles in the integer slots op 22 leaves empty.
                    d.i[5] = w.sq;
                    d.i[6] = w.sk;
                    d.i[7] = w.sv;
                })
            } else {
                b.emit(DevOp::GemvQkv, all.clone(), &[c_n], |d| {
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
                })
            };
            (c_q, c_k, c_v, v_src) = (fused, fused, fused, n.vg);
            let _ = qkv_g; // norm is a shared packet here, never folded into the fused GEMV
        } else {
            let (cq, ck, cv) = if gemv_family {
                split3(gemv_wg_n(n_cu), qd, kd, if keqv { 0 } else { kd })
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
            // PLOW_QNORM_FUSE: the quant already rode the RmsNorm (t3/t4 above) — c_n IS it.
            let dq = if qnorm_fuse {
                c_n
            } else {
                quant(b, n.xqh, n.ash, qkv_src, c.hidden, c_n)
            };
            // A pending NRN fold from the previous layer's skipped NormResidualNorm: emit the
            // trio as GemvFp8 WITH the fold slots (op 30 i3; see exec_gemv_fp8/gemv_nrn_lds)
            // instead of through `proj`. Each packet computes the NRN redundantly from a=n.xr;
            // ONLY the q packet stores the residual to n.x (i3 bit 1) — its siblings would
            // otherwise race the store against their own loads of the same row.
            let nrn = nrn_pending.take();
            let fold_proj = |b: &mut Builder,
                             out: u32,
                             w8: u32,
                             sc: u32,
                             nn: u32,
                             cus: Vec<u32>,
                             store: bool,
                             f: &(u32, u32, u32, f32)|
             -> u32 {
                b.emit(DevOp::GemvFp8, cus, &[dq], |d| {
                    d.t[0] = out;
                    d.t[1] = n.xr; // a: the residual NRN1 wrote (no normed hn exists to read)
                    d.t[2] = w8;
                    d.t[3] = n.x; // resid_out: the ping-pong twin
                    d.t[4] = f.0; // b: the down/ffn output
                    d.t[5] = sc;
                    d.t[6] = f.1; // gamma_b (post-FFN norm)
                    d.t[7] = f.2; // gamma_n (next input norm)
                    d.i[0] = t;
                    d.i[1] = nn;
                    d.i[2] = c.hidden;
                    d.i[3] = if store { 3 } else { 1 };
                    d.f[0] = eps;
                    d.f[1] = f.3; // layer_scalar
                })
            };
            let cqc = if let Some(f) = &nrn {
                fold_proj(b, n.qg, w.wq8, w.sq, qd, cq, true, f)
            } else {
                proj(
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
                )
            };
            let ckc = if let Some(f) = &nrn {
                fold_proj(b, n.kg, w.wk8, w.sk, kd, ck, false, f)
            } else {
                proj(
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
                )
            };
            let (vsrc, cvc) = if keqv {
                (n.kg, ckc) // k_eq_v: V is the RAW k_proj output
            } else if let Some(f) = &nrn {
                (n.vg, fold_proj(b, n.vg, w.wv8, w.sv, kd, cv, false, f))
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
        // q/k/v HEADNORM ON DISJOINT CU SETS. The three are mutually INDEPENDENT -- each depends
        // only on its own projection, never on the others -- but all three were emitted on the
        // SAME `hn_cus`, so a workgroup had to finish q before it could start k. At decode
        // (t=1, heads=16) `nhn` is 2, so three 8.9 us packets ran back to back on 2 of 304 CUs.
        //
        // This is NOT the widening the note above refuses. Each op keeps its own `nhn`
        // workgroups and its own head->wave map; only their PLACEMENT moves apart, which is what
        // `Builder::split` exists for ("so independent ops overlap"). Guarded on 3*nhn fitting,
        // because at prefill `nhn` is already the whole machine and there is nothing to separate.
        // OFF BY DEFAULT, and NOW MEASURED: it is a NO-OP UNDER THE GLOBAL QUEUE.
        //
        // The premise -- "three 8.9 us packets ran back to back on 2 of 304 CUs" -- is a property
        // of the STATIC per-CU stream, where a workgroup executes its own fixed packet list and so
        // must finish q before it starts k. Under PLOW_GLOBAL_QUEUE the three packets are claimed
        // by whichever workgroups are free, so they spread on their own. TRACED on MI300X
        // (gfx942, 304 CU, Gemma-4-12B decode, PLOW_TRACE_RAW), layer 1, hn_split OFF:
        //
        //     inst 18 (q)  cus [65, 120]   starts 257.28 / 257.23 us
        //     inst 19 (k)  cus [41, 272]   starts 257.44 / 257.23
        //     inst 20 (v)  cus [40, 233]   starts 257.16 / 257.15
        //
        // SIX DISTINCT CUs, all three within 0.3 us of each other -- already fully concurrent, so
        // there is nothing for the split to recover. It may still be worth something on the static
        // path, which is where the original observation came from; it is dead weight on the
        // global-queue path that every AMD decode object ships today.
        // Enable with PLOW_HN_SPLIT=1 if you are on the static scheduler.
        let hn_split = 3 * nhn <= n_cu && emit_config::active().hn_split;
        let hn_set = |i: u32| -> Vec<u32> {
            if hn_split {
                (i * nhn..(i + 1) * nhn).collect()
            } else {
                hn_cus.clone()
            }
        };

        // gemv -> headnorm. headnorm workgroup j owns whole HEADS, and head h is the 256 (or
        // 512) consecutive output columns [h*hd, h*hd+hd) of the projection — so it needs only
        // the handful of gemv workgroups that produced those columns, not all 128.
        //
        // This is ONLY sparse under GV_BLOCKED (op_gemm.h). With the default wave-interleaved
        // column assignment a gemv workgroup's columns are scattered across all of N, and the
        // fan-in is 128 of 128 — measured, and the reason the first attempt at a fine chain
        // bought nothing.
        let hn_dep = |gemv: u32, nblk_g: u32, nheads: u32| -> Vec<Dep> {
            if !gemv_family || fuse_qkv || fuse_qkv_fp8 {
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

        // NRF: fold ALL THREE HeadNormRope packets into d_flash_decode's NRF template arm (see
        // op_attention.h). The freed integer slots carry the fold operands, so several fields
        // ship PACKED (the exec unpacks): i0 = n_batch | window<<8, i3 = kv_stride | nsplit<<20,
        // i1 bit16 = flag; gamma handles in i4/i5, cos/sin handles in j0/j1 (fj1.u/fj2.u), and
        // f0 becomes eps — legal only because Gemma's decode attn_scale is exactly 1.0.
        //
        // OFF BY DEFAULT: MEASURED A NULL, the same verdict as op 115 and for the same reason.
        // Gemma-4-12B fp8 occ4, 48 steps x 3 interleaved reps, ms/token, token-identical serve:
        //   coarse deps onto q/k/v:  11.36 -> 11.99 with an agent-scope fence per owner item
        //     (its buffer_inv is a FULL L1+L2 invalidate — see the kernel note), 11.36 -> 11.40
        //     with the workgroup release that is actually required;
        //   FINE per-head deps (this map): 11.33 -> 11.37 (+0.3%).
        // The deleted chain level was already almost fully OVERLAPPED: the hnr packets' fine
        // producer maps let them start before the slowest gemv workgroup, so their wall-clock
        // cost was ~the post-producer tail, not a 10 us gate. The fold trades three nearly-free
        // packets for equivalent staging work inside every flash item. It stays in the tree
        // (correct, token-identical, and the trade flips where fine deps do not exist — the
        // static scheduler, or a launch-per-op backend). Opt in with PLOW_FUSE_HNR=1.
        let fuse_hnr = gemv_family
            && amd
            && t == 1
            && !fp8_kv
            // The fine nrf_dep maps below are built per-projection (q/k/v as three GEMV
            // packets). A fused-QKV packet (bf16 default, or the op-115 opt-in) spans all
            // three projections, so the per-projection maps under-synchronize the flash
            // read — require the split3 emission the fold was designed against.
            && fp8
            && !fuse_qkv_fp8
            && c.arch == Arch::Gemma4
            && qk_skip == 0
            && v_skip == 0
            && c.attn_scale == 1.0
            && win < (1 << 24)
            && kvr < (1 << 20)
            && ns < (1 << 12)
            && heads < (1 << 16)
            && emit_config::active().fuse_hnr;
        let c_qn = if fuse_hnr {
            0 // no packet: the fold computes q's norm+rope in flash's staging
        } else {
            b.emit_dep(
                DevOp::HeadNormRope,
                hn_set(0),
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
            )
        };
        // fp8-KV: the k/v norm STORES the cache as e4m3 with a per-row scale (t6). q is unchanged
        // (it is not cached — flash reads it as bf16), so it stays plain HeadNormRope.
        let hn_op = if fp8_kv {
            DevOp::HeadNormRopeFp8
        } else {
            DevOp::HeadNormRope
        };
        let c_kn = if fuse_hnr {
            0
        } else {
            b.emit_dep(hn_op, hn_set(1), hn_dep(c_k, nk, kvh), |d| {
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
                //
                // UNDER A LADDER IT IS ARMED AT t == 1 TOO, and that is a correctness
                // requirement, not tidiness. The legacy arm takes its write row from `out_row0`,
                // a HOST-PATCHED immediate (`patch_kvrow`) that `decode_step_batched` never
                // writes — and a laddered blob is always a batched engine — so the one-row rung
                // would rewrite KV row 0 on every step. Armed, it reads `pos[0]`, which at one
                // row is the same address the legacy formula computes (`d_headnorm_rope`).
                if decode && (t > 1 || seq_rows) {
                    d.i[6] = t;
                }
            })
        };
        if decode && !fuse_hnr {
            kv_rows.push(c_kn);
        }
        // v_norm: WEIGHTLESS (gamma NONE) and NO RoPE (cos NONE).
        // On a full layer V comes from the RAW k_proj output, so its producer is c_k (nk wgs).
        let vn_dep = if keqv {
            hn_dep(c_v, nk, kvh)
        } else {
            hn_dep(c_v, nv, kvh)
        };
        let c_vn = if fuse_hnr {
            0
        } else {
            b.emit_dep(hn_op, hn_set(2), vn_dep, |d| {
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
                if decode && (t > 1 || seq_rows) {
                    d.i[6] = t;
                }
            })
        };
        if decode && !fuse_hnr {
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
        // [MERGE-FOLD] per-layer arm: rides the NRF packet (its spare i1/i6 bits carry the two
        // handles), so it exists only where the hnr fold does. When on, the FlashMerge packet
        // below is NOT emitted — the last-arriving split workgroup merges in d_flash_decode's
        // epilogue and the packet's own coarse completion signal covers the merge, so o_proj
        // just re-points its dep at the flash op with no threshold change.
        let fuse_merge = fuse_hnr && n.mrgc != TENSOR_NONE;
        let c_fa = if fuse_hnr {
            // NRF fold packet: flash depends on the three RAW projections directly (the hnr
            // level is gone). Operands per the exec's unpacking map; kv_rows gets nothing —
            // the fold reads the write position from the kv_len TENSOR (qpos = len-1), so
            // these layers leave the per-step host-patch surface entirely.
            // FINE deps onto the RAW projections, per flash work item's heads — the same
            // per-head producer map the deleted HeadNormRopes carried. With a coarse dep the
            // fold measured +0.3% (flash waited the SLOWEST gemv workgroup where hnr had
            // started early); the fine map restores that overlap. Item mapping mirrors the
            // kernel: w -> (sp = w % ns, hg = (w/ns) % n_grp, b), WG f runs items w ≡ f (mod nblk).
            let nrf_dep = {
                let nblk_f = all.len() as u32;
                let n_grp = heads / gf;
                let n_work = t * n_grp * ns;
                let gqa = heads / kvh;
                let mkc = |q: bool, nblk_g: u32| -> Vec<Vec<u32>> {
                    (0..nblk_f)
                        .map(|f| {
                            let mut s: Vec<u32> = (0..n_work)
                                .filter(|w| w % nblk_f == f)
                                .flat_map(|w| {
                                    let h0 = ((w / ns) % n_grp) * gf;
                                    if q {
                                        gemv_wgs_for_cols(
                                            heads * hd,
                                            nblk_g,
                                            h0 * hd,
                                            (h0 + gf) * hd,
                                        )
                                    } else {
                                        let hkv = h0 / gqa;
                                        gemv_wgs_for_cols(
                                            kvh * hd,
                                            nblk_g,
                                            hkv * hd,
                                            (hkv + 1) * hd,
                                        )
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
                        producer: c_q,
                        map: mkc(true, nq),
                    },
                    Dep::Fine {
                        producer: c_k,
                        map: mkc(false, nk),
                    },
                    Dep::Fine {
                        producer: c_v,
                        map: mkc(false, if keqv { nk } else { nv }),
                    },
                ]
            };
            b.emit_dep(DevOp::FlashDecode, all.clone(), nrf_dep, |d| {
                d.t[0] = n.opart;
                d.t[1] = n.mlpart;
                d.t[2] = n.qg; // RAW q projection
                d.t[3] = n.kc[l];
                d.t[4] = n.vc[l];
                d.t[5] = n.kvlen;
                d.t[6] = n.kg; // RAW k projection
                d.t[7] = v_src; // RAW v projection (= kg on k_eq_v layers)
                d.i[0] = t | (win << 8);
                d.i[1] = heads | (1 << 16); // bit16 = NRF flag (bit17 = skip_norm, 0 here)
                d.i[2] = kvh;
                d.i[3] = kvr | (ns << 20);
                d.i[4] = w.qn; // gamma_q handle
                d.i[5] = w.kn; // gamma_k handle
                d.i[6] = hd;
                d.i[7] = kvm;
                if fuse_merge {
                    // [MERGE-FOLD] handles in the layout's spare bits (every whole slot is
                    // taken): counter tensor in i1[18..32) — 14 bits, asserted — and the bf16
                    // attention output in i6[16..32). Zero = fold off on the exec side, and
                    // neither is ever tensor 0 (both are compiler-declared activations).
                    assert!(
                        n.mrgc != 0 && n.mrgc < (1 << 14),
                        "mrgc handle {} overflows i1[18..32)",
                        n.mrgc
                    );
                    assert!(
                        n.at != 0 && n.at < (1 << 16),
                        "at handle {} overflows i6[16..32)",
                        n.at
                    );
                    assert!(hd < (1 << 16), "hd {hd} collides with the packed at handle");
                    d.i[1] |= n.mrgc << 18;
                    d.i[6] |= n.at << 16;
                }
                d.f[0] = c.eps; // the scale slot, repurposed: attn_scale asserted 1.0
                d.j[0] = cs; // cos handle -> fj[1].u
                d.j[1] = sn; // sin handle -> fj[2].u
            })
        } else if gemv_family {
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
            // sm_90a TMA K/V staging for the hd256 wgmma arm: t[7] carries the
            // GEN_TMAP_KV_PAIR handle. Safe overload: t[7] is the fp8-KV V-scale slot, and
            // the TMA arm is bf16-KV-only (the cubin #errors on the combination), so on a
            // bf16-KV packet the slot was always TENSOR_NONE.
            // T33: hd512 (full-attn) too — the stager is HD-generic (NSUB sub-tiles), and
            // k_eq_v just encodes the same base at both pair slots.
            let fa_tm = (tma_gemm && !fp8_kv && (hd == 256 || hd == 512) && !gemv_family)
                .then(|| tmap_kv(n.kc[l], n.vc[l], kvr, hd, kvh));
            b.emit(fa_op, all.clone(), &[c_qn, c_kn, c_vn], |d| {
                d.t[0] = n.opart;
                d.t[1] = n.mlpart;
                d.t[2] = n.q;
                d.t[3] = n.kc[l];
                d.t[4] = n.vc[l];
                d.t[6] = n.kcs[l];
                d.t[7] = n.vcs[l]; // fp8-KV per-row scales (NONE in bf16 mode)
                if let Some(h) = fa_tm {
                    d.t[7] = h;
                }
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
        let attn_dep = if fused || fuse_merge {
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

        // o_proj is ROW-parallel: input = this rank's qd heads, output =
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
        let mut nrn1_fold = false;
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
        } else if gfuse && fuse_nrn {
            // NRN1 FOLD: the sandwich packet is computed inside the GemvGluFp8's staging instead
            // (fj[2].u; gemv_nrn_lds). `fuse_nrn` implies fp8 + the staged-LDS bound, so the GLU
            // below is guaranteed to be the GemvGluFp8 arm this fold lives in. One packet, so no
            // sibling race — its slice 0 stores the residual to n.xr, which the NRN2 fold in the
            // next layer's q/k/v reads back. The GLU's dep becomes o_proj directly: one fewer
            // serial gate on the decode chain.
            nrn1_fold = true;
            c_o
        } else if gfuse {
            // x = x + post_attn_norm(o); hn = pre_feedforward_norm(x) — Gemma sandwich in ONE packet.
            // Under the NRN fold the residual PING-PONGS: this packet writes its residual to
            // `n.xr`, which the folded NRN2 inside the next layer's q/k/v reads back and stores
            // to `n.x` — never in place, because the q/k/v trio is concurrent (see fuse_nrn).
            b.emit(DevOp::NormResidualNorm, rows.clone(), &[c_o], |d| {
                d.t[0] = n.hn;
                d.t[1] = if fuse_nrn { n.xr } else { n.x };
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
                if qnorm_fuse {
                    d.t[3] = n.xqh; // fused w8a8 activation quant (T11) — see the pre-qkv site
                    d.t[4] = n.ash;
                }
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
            gemv_family && (gemv_staged_rows(t) as u64 * c.hidden as u64) <= gm_lds_halves();
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
        // PLOW_QNORM_FUSE: the pre-FF RmsNorm carried the quant (t3/t4) — c_pf IS it. The
        // fuse_norm (decode) arm never fuses, so only the RmsNorm branch can set qnorm_fuse.
        let dmlp = if qnorm_fuse {
            c_pf
        } else {
            quant(b, n.xqh, n.ash, mlp_src, c.hidden, c_pf)
        };
        let c_gl = if glu_fused {
            // FP8 decode: gate|up fused GEMV+GLU on fp8 weights, each with its own dequant scale.
            if fp8 {
                b.emit(DevOp::GemvGluFp8, gemv_wg_cap(all.clone()), &[c_pf], |d| {
                    d.t[0] = n.fu;
                    d.t[1] = if nrn1_fold { n.x } else { mlp_src };
                    d.t[2] = w.wg8;
                    d.t[5] = w.wu8;
                    d.t[3] = w.sg;
                    d.t[4] = w.su;
                    d.i[0] = t;
                    d.i[1] = inter_l;
                    d.i[2] = c.hidden;
                    d.i[5] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU (Llama/Qwen)
                    if nrn1_fold {
                        // NRN1 fold operands (see exec_gemv_glu_fp8): resid_out/b/gammas as
                        // TENSOR HANDLES in the free integer slots; the flag rides fj[2].u (j1).
                        d.i[3] = n.xr; // resid_out: the ping-pong twin NRN2's fold reads
                        d.i[4] = n.og; // b: the attention output
                        d.i[6] = w.g_pa; // gamma_b (post-attention norm)
                        d.i[7] = w.g_pf; // gamma_n (pre-feedforward norm)
                        d.f[0] = c.eps;
                        d.f[1] = 1.0; // NRN1's layer scale is always 1
                        d.j[1] = 1;
                    }
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
            // sm_90a TMA GLU ring (see `tmap`): w8a8 only, 3 e4m3 maps in i6/i7/i3.
            let tmg8 = (tma_gemm && w8a8).then(|| {
                (
                    tmap8(n.xqh, t, c.hidden),
                    tmap8(w.wg8, inter_l, c.hidden),
                    tmap8(w.wu8, inter_l, c.hidden),
                )
            });
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
                if let Some((ma, mg, mu)) = tmg8 {
                    d.i[6] = ma;
                    d.i[7] = mg;
                    d.i[3] = mu;
                }
            })
        } else if gemm_glu {
            // sm_90a TMA GLU ring: bf16 maps in i6/i7/i3.
            let tmg = tma_gemm.then(|| {
                (
                    tmap(mlp_src, t, c.hidden),
                    tmap(w.wg, inter_l, c.hidden),
                    tmap(w.wu, inter_l, c.hidden),
                )
            });
            b.emit(DevOp::GemmGlu, all.clone(), &[c_pf], |d| {
                d.t[0] = n.fu;
                d.t[1] = mlp_src;
                d.t[2] = w.wg;
                d.t[5] = w.wu;
                d.i[0] = t;
                d.i[1] = inter_l;
                d.i[2] = c.hidden;
                d.i[5] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU (Llama/Qwen)
                if let Some((ma, mg, mu)) = tmg {
                    d.i[6] = ma;
                    d.i[7] = mg;
                    d.i[3] = mu;
                }
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
            if qnorm_fuse {
                // T11 GLU-INTO-QUANT: one row-owning packet computes fu = act(g)*u AND its
                // fp8 quant (QuantFp8 t3/t4/i2 — see d_quant_fp8), deleting the elementwise
                // Glu packet + gate + the inter-width fu re-read. bf16-rounded before quant,
                // so token-identical to the split form.
                b.emit(DevOp::QuantFp8, rows.clone(), &[c_g, c_u], |d| {
                    d.t[0] = n.xqi;
                    d.t[1] = n.fu;
                    d.t[2] = n.asi;
                    d.t[3] = n.gt;
                    d.t[4] = n.ut;
                    d.i[0] = t;
                    d.i[1] = inter_l;
                    d.i[2] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU
                })
            } else {
                b.emit(DevOp::Glu, elem(t * inter_l), &[c_g, c_u], |d| {
                    d.t[0] = n.fu;
                    d.t[1] = n.gt;
                    d.t[2] = n.ut;
                    d.i[0] = t * inter_l;
                    d.i[1] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU
                })
            }
        };
        // down_proj is ROW-parallel (input = inter_l lanes) → a PARTIAL H-vector. Under TP it
        // writes dg_tp and an XReduce sums the N peers into `dg` — all-reduce #2 of the layer,
        // at the second NormResidual boundary. proj() picks the fp8
        // (GemvFp8) arm on the decode fp8 path via the wd8/sd operands.
        // w8a8: quant the (inter-width) GLU output feeding down_proj.
        // qnorm_fuse (+ unfused GLU): c_gl IS the fused GLU+quant packet above.
        let dfu = if qnorm_fuse && !glu_fused && !gemm_glu {
            c_gl
        } else {
            quant(b, n.xqi, n.asi, n.fu, inter_l, c_gl)
        };
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
        // ===== Gemma-4 26B-A4B MoE branch (decode, B=1) =====
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
                let tail_fuse = emit_config::active().gemma_moe_tail_fuse;
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
                // ===== GROUPED-MoE PREFILL (T rows) =====
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
            if fuse_nrn && l + 1 < block.end {
                // NRN FOLD: skip the packet; the next layer's q/k/v GemvFp8s compute it in
                // their staging (they read a = n.xr and the q packet stores resid to n.x).
                // The chain dep becomes the down GEMV, which transitively covers NRN1 (the
                // n.xr writer) via GLU. LAST layer keeps the packet: its second norm is the
                // model's FINAL norm, consumed by the lm_head, not a GEMV. A k_eq_v next
                // layer folds into q+k only (no v proj exists).
                nrn_pending = Some((ffn_out, w.g_po, next_gin, ls[l]));
                c_d
            } else {
                b.emit(DevOp::NormResidualNorm, rows.clone(), &[c_d], |d| {
                    d.t[0] = n.hn;
                    d.t[1] = n.x;
                    d.t[2] = if fuse_nrn { n.xr } else { n.x };
                    d.t[3] = ffn_out;
                    d.t[4] = w.g_po;
                    d.t[5] = next_gin;
                    d.i[0] = t;
                    d.i[1] = c.hidden;
                    d.f[0] = c.eps;
                    d.f[1] = ls[l];
                })
            }
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
    // lm_head is COLUMN(vocab)-parallel: each rank produces its
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
        && match emit_config::active().pf_gemv_head.as_deref() {
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
    // sm_90a TMA (see `tmap` above): prefill lm_head reads A at a_row0=t-1, so its map
    // must span lm_row0 + lm_m rows; the weight map spans the vocab slice.
    let lm_tm = (tma_gemm && matches!(lm_op, DevOp::Gemm | DevOp::GemmMed | DevOp::GemmSmall))
        .then(|| {
            (
                tmap(head_src, lm_row0 + lm_m, c.hidden),
                tmap(head_w, vocab_l, c.hidden),
            )
        });
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
        if let Some((ma, mb)) = lm_tm {
            d.i[6] = ma;
            d.i[7] = mb;
        }
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
/// interleaved arms under `perf-data/tools/gpulease -n 1`, contended (rc=76) runs discarded.
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
    emit_config::active().gemv_split
}

/// E5 (rtx-19): PLOW_FUSE_ARGMAX fuses the greedy-argmax epilogue into the lm_head GEMV
/// (`DevOp::GemvArgmax`), replacing the `SoftCap` + `Argmax` packets. Default off → byte-identical.
fn fuse_argmax_on() -> bool {
    emit_config::active().fuse_argmax
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
// (real 256 experts, real [128,128] block-fp8 scales) — see the design notes
// "B4-CORE DONE". The offline glm_tests below assert byte-for-op equality with that
// reference, so the emitted layer inherits the harness's passing GPU result.
//
// MILESTONE-1 STAGING: the query/key RoPE is folded into
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
    /// byte-identical. Dense-GQA path only. See the design notes.
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
    /// Unified emit-time config. `Some` when driven by the new `plowc` CLI;
    /// `None` from the legacy `gemma4` `from_cli` path (env-var fallback).
    pub emit_cfg: Option<emit_config::EmitConfig>,
    /// Fusion candidates derived from the complete operator graph. Empty for
    /// legacy/direct callers; no model name participates in these decisions.
    pub whole_graph_fusions: WholeGraphFusionDecisions,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WholeGraphFusionDecisions {
    pub tp: u32,
    pub parallel_linear2: Vec<ParallelLinear2Decision>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParallelLinear2Decision {
    pub n0: u32,
    pub n1: u32,
    pub k: u32,
    pub instances: usize,
    /// Qualification is separate from structural eligibility. The compiler
    /// carries candidates immediately, but production emission stays unchanged
    /// until the exact full-token/performance gate promotes the shape.
    pub qualified: bool,
}

static WHOLE_GRAPH_FUSIONS: std::sync::atomic::AtomicPtr<WholeGraphFusionDecisions> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

fn install_whole_graph_fusions(decisions: WholeGraphFusionDecisions) {
    let ptr = Box::into_raw(Box::new(decisions));
    WHOLE_GRAPH_FUSIONS.store(ptr, std::sync::atomic::Ordering::Release);
}

pub(crate) fn whole_graph_parallel_linear2(n0: u32, n1_local: u32, k: u32) -> bool {
    let ptr = WHOLE_GRAPH_FUSIONS.load(std::sync::atomic::Ordering::Acquire);
    if ptr.is_null() {
        return false;
    }
    // SAFETY: install stores Box::into_raw and deliberately never frees it,
    // matching emit_config's process-wide per-compile snapshot.
    let decisions = unsafe { &*ptr };
    decisions.parallel_linear2.iter().any(|d| {
        d.qualified
            && d.n0 == n0
            && d.n1 == n1_local.saturating_mul(decisions.tp)
            && d.k == k
            && d.instances > 0
    })
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
/// The decode-object LDS arena in halves for the ACTIVE emit target — the bound
/// the LdsFitSound checkpoint (plow_verify checkpoint G) verifies every staged
/// GEMV instance against. Public so the plowc verify hook can hand the Lean
/// side the same number the emitter's fusion gate used.
pub fn decode_arena_halves() -> u64 {
    gm_lds_halves()
}

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
            emit_cfg: None,
            whole_graph_fusions: WholeGraphFusionDecisions::default(),
        }
    }
}

/// Uniform surface for lowering a checkpoint into a PLOWDEV program set, per model
/// family. Phase 0: only the dense-GQA family
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
/// "one DenseGqaEmitter" design decision in the design notes.
/// Mint registry for `GEN_TMAP_BF16` descriptor tensors (sm_90a TMA prefill GEMM,
/// `PLOW_TMA_GEMM=1`). Handles are GLOBAL — `base` (the declare()-time tensor count) plus
/// the mint index — because each program's Builder adopts a clone of the declared list
/// and drops mid-emission decls at finish. `run_verified` extends the Model's
/// tensors/gen from here after every program is emitted, so all programs agree on the
/// same handles and the blob carries each map exactly once.
#[derive(Default)]
struct TmapMint {
    base: u32,
    decls: Vec<(String, GenTensor)>,
    memo: std::collections::HashMap<(u32, u32, u32), u32>,
}

impl TmapMint {
    /// `e4m3`: the target is a `[rows][k]` e4m3 tensor (GEN_TMAP_E4M3, inner box 128);
    /// else bf16 (GEN_TMAP_BF16, inner box 64). A tensor has ONE dtype, so the memo key
    /// does not need the kind.
    fn handle(&mut self, target: u32, rows: u32, k: u32, e4m3: bool) -> u32 {
        if let Some(&h) = self.memo.get(&(target, rows, k)) {
            return h;
        }
        let h = self.base + self.decls.len() as u32;
        let mut g = if e4m3 {
            GenTensor::tmap_e4m3(target, rows, k, 128)
        } else {
            GenTensor::tmap_bf16(target, rows, k, 128)
        };
        g.tensor = h; // tensor_gen would have patched this; we are our own declarer
        self.decls.push((format!("tmap.{target}.{rows}x{k}"), g));
        self.memo.insert((target, rows, k), h);
        h
    }

    /// A `GEN_TMAP_KV_PAIR` (256 B: K map + V map) for the flash-prefill TMA stager.
    /// Memoised on the (K, V) tensor pair (`u32::MAX` marks the key as a pair — a rows
    /// value no real tensor reaches).
    fn kv_pair(&mut self, kt: u32, vt: u32, ring: u32, hd: u32, nkv: u32) -> u32 {
        if let Some(&h) = self.memo.get(&(kt, vt, u32::MAX)) {
            return h;
        }
        let h = self.base + self.decls.len() as u32;
        let mut g = GenTensor::tmap_kv_pair(kt, vt, ring, hd, nkv);
        g.tensor = h;
        self.decls.push((format!("tmap.kv.{kt}"), g));
        self.memo.insert((kt, vt, u32::MAX), h);
        h
    }
}

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
    /// See [`TmapMint`]. RefCell: `emit_prefill`/`emit_decode` take `&self`.
    tmaps: std::cell::RefCell<TmapMint>,
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
        // NRN2 -> q/k/v fold (op 30 i3, gemv_nrn_lds): Gemma dense fp8 decode, AMD arm only.
        // The env kill-switch mirrors PLOW_NO_FUSE_QKV; the op-115 opt-in disables it because
        // the fused QKV packet has no free slots to carry the fold.
        let nrn_fold = amd
            // gfx942 only: the op-30 fold arm (gemv_nrn_lds) is new in this branch's
            // op_gemm.h, and there is no marker-symbol check to refuse a pre-fold gfx950
            // object served against a folded blob (an old object reads t1=xr — the raw
            // un-normed residual — and decodes fluent garbage). gfx950 keeps main's
            // emission until the fold is measured there and a marker check lands.
            && amd_target::active().1 == hwspec::IsaLevel::Gfx942
            && fp8
            && c.arch == Arch::Gemma4
            && !c.moe
            && !emit_config::active().no_fuse_nrn
            && !emit_config::active().fuse_qkv_fp8;
        // [MERGE-FOLD] opt-in (PLOW_FUSE_MERGE=1) and rides the NRF packet's spare bits, so it
        // additionally requires the hnr fold's per-layer gates at emit time — this flag only
        // declares the counter tensor. Same arch scoping as nrn_fold (gfx942, measured there).
        let merge_fold = amd
            && amd_target::active().1 == hwspec::IsaLevel::Gfx942
            && fp8
            && c.arch == Arch::Gemma4
            && !c.moe
            && emit_config::active().fuse_hnr
            && emit_config::active().fuse_merge;
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
            nrn_fold,
            merge_fold,
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
            tmaps: std::cell::RefCell::new(TmapMint {
                base: tensors.len() as u32,
                ..Default::default()
            }),
        };
        (e, tensors, gen)
    }

    /// Drain the [`TmapMint`] registry into `(decls, gen)` for the Model tables.
    /// Call AFTER every program is emitted.
    fn take_tmaps(&self) -> (Vec<packet::devbuild::TensorDecl>, Vec<GenTensor>) {
        let mint = std::mem::take(&mut *self.tmaps.borrow_mut());
        let mut decls = Vec::with_capacity(mint.decls.len());
        let mut gens = Vec::with_capacity(mint.decls.len());
        for (name, g) in mint.decls {
            decls.push(packet::devbuild::TensorDecl {
                name,
                bytes: g.byte_len(),
                init: None,
            });
            gens.push(g);
        }
        (decls, gens)
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
            &self.tmaps,
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
            &self.tmaps,
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
        emit_cfg: _emit_cfg,
        whole_graph_fusions,
    } = args;

    // Resolve the unified emit config: either from the CLI (plowc path) or from env vars (legacy).
    // Installed process-wide so deeply nested emit functions can call emit_config::active().
    emit_config::install(_emit_cfg.unwrap_or_else(emit_config::EmitConfig::from_env));
    install_whole_graph_fusions(whole_graph_fusions);
    clear_attention_decisions();

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
    // than let a user believe placement is active. See the design notes.
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
    // Point the tile selector at the part actually being built for. Without this every AMD tile
    // is costed against MI350X -- 160 KiB of LDS and double-rate MFMA -- whatever --gpu said.
    // `--arch` is passed as the fallback: an unresolvable --gpu used to leave the gfx950 default
    // standing on a gfx942 build, warn once, and cost every tile wrong.
    if amd {
        set_amd_target_for(&arch, &gpu);
    }
    warn_uniseg_on_amd(amd);
    // Bind pick_tile for every emit path below (dense / GLM / Kimi / …). Nested
    // `EmitAmdGuard`s in emit_dense_gqa restore correctly on drop. NO target at all
    // (empty arch AND gpu — the golden tests' legacy mode) keeps the AMD inventory:
    // `target_is_amd("","")` is false, but "no target => unchanged emission" is the
    // invariant `target_is_amd_reads_either_signal` documents.
    let _emit_target = EmitAmdGuard::set(amd || (arch.is_empty() && gpu.is_empty()));
    if l2_layout.is_some()
        && matches!(
            model_type.as_str(),
            "kimi_k2"
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
        if emit_config::active().k3_full {
            mla::k3_emit_full(
                &dir,
                ctx,
                &out,
                n_cu,
                tp,
                rope_gen,
                &arch,
                verify.as_ref(),
                l2_layout,
            );
            return;
        }
        mla::kimi_k3_emit(&dir, ctx, tp, block_spec.as_deref());
    }
    if model_type == "glm_moe_dsa" {
        // GLM `--block` (M2): single-block
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
            None => glm_main(
                &dir,
                ctx,
                &out,
                n_cu,
                tp,
                rope_gen,
                &arch,
                l2_layout,
                verify.as_ref(),
            ),
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
                 milestone."
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
                 (or PLOW_BLOCK). Full-model Nemotron device emit is a later milestone."
            ),
        }
        return;
    }

    // Phase 2: the dense-GQA family is its own
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
    "PLOW_DOP_GEMV_QKV_FP8",
    "PLOW_DOP_GEMV_QKV_MXFP4",
    "PLOW_DOP_GLU",
    "PLOW_DOP_HEADNORM_ROPE",
    "PLOW_DOP_HEADNORM_ROPE_FP8",
    "PLOW_DOP_INDEX_SCORE",
    "PLOW_DOP_INDEX_SELECT",
    "PLOW_DOP_INDEX_SCORE_PF",
    "PLOW_DOP_INDEX_SELECT_PF",
    "PLOW_DOP_INDEX_UNION_PF",
    "PLOW_DOP_KDA_CONV",
    "PLOW_DOP_KDA_CONV3",
    "PLOW_DOP_KDA_CONV_STATE_STEP_G",
    "PLOW_DOP_KDA_CHUNK_CARRY",
    "PLOW_DOP_KDA_CHUNK_INTRA",
    "PLOW_DOP_KDA_CHUNK_PREPARE",
    "PLOW_DOP_KDA_CHUNK_WU",
    "PLOW_DOP_KDA_GATE",
    "PLOW_DOP_KDA_GATED_NORM",
    "PLOW_DOP_KDA_STATE_STEP",
    "PLOW_DOP_KDA_STATE_STEP_G",
    "PLOW_DOP_LAYERNORM",
    "PLOW_DOP_MLA_MERGE_FOLD",
    "PLOW_DOP_MLA_OUT_GATE",
    "PLOW_DOP_MOE_ALIGN_GEMMA_PF",
    "PLOW_DOP_MOE_ALIGN_PF",
    "PLOW_DOP_MOE_COMBINE",
    "PLOW_DOP_MOE_COMBINE_GEMMA",
    "PLOW_DOP_MOE_COMBINE_NORM_GEMMA",
    "PLOW_DOP_MOE_COMBINE_NORM_GEMMA_PF",
    "PLOW_DOP_MOE_COMBINE_PF",
    "PLOW_DOP_MOE_COMBINE_RESID_NORM_GEMMA",
    "PLOW_DOP_MOE_EXPERT_DOWN",
    "PLOW_DOP_MOE_EXPERT_DOWN_FP8_BLK",
    "PLOW_DOP_MOE_EXPERT_DOWN_GEMMA",
    "PLOW_DOP_MOE_EXPERT_DOWN_GEMMA_FP8",
    "PLOW_DOP_MOE_EXPERT_GLU",
    "PLOW_DOP_MOE_EXPERT_GLU_FP8_BLK",
    "PLOW_DOP_MOE_EXPERT_GLU_GEMMA",
    "PLOW_DOP_MOE_EXPERT_GLU_GEMMA_FP8",
    "PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA",
    "PLOW_DOP_MOE_GROUP_DOWN_FP8_BLK",
    "PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF",
    "PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF_W8A8",
    "PLOW_DOP_MOE_GROUP_DOWN_PF",
    "PLOW_DOP_MOE_GROUP_GLU_FP8_BLK",
    "PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF",
    "PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF_W8A8",
    "PLOW_DOP_MOE_GROUP_GLU_PF",
    "PLOW_DOP_MOE_ROUTER",
    "PLOW_DOP_MOE_ROUTER_GEMMA",
    "PLOW_DOP_MOE_ROUTER_GEMMA_PF",
    "PLOW_DOP_MOE_ROUTER_GEMMA_SCORE",
    "PLOW_DOP_MOE_ROUTER_GEMMA_SCORE_FAST",
    "PLOW_DOP_MOE_ROUTER_GEMMA_TOPK",
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
    "PLOW_DOP_XREDUCE_ADD_NORM",
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

/// gfx950-only dense-GEMM rungs. NVIDIA `plow_exec` has no `case` for these — `default: __trap()`.
const NVIDIA_UNIMPLEMENTED_GEMM: &[&str] = &[
    "PLOW_DOP_GEMM_WIDE",
    "PLOW_DOP_GEMM_WIDE_FP8",
    "PLOW_DOP_GEMM_WIDE_MXFP4",
    "PLOW_DOP_GEMM_C5",
    "PLOW_DOP_GEMM_C5_FP8",
    "PLOW_DOP_GEMM_C5_MXFP4",
];

/// Refuse a packet that carries gfx950-only GEMM tiles when the emit target is NVIDIA.
///
/// Sibling of [`check_gfx950_opcode_coverage`]: AMD fails silent, NVIDIA fails loud with
/// `CUDA_ERROR_LAUNCH_FAILED` on first prefill. Catch it at emit.
fn check_nvidia_opcode_coverage(m: &Model, amd: bool) {
    if amd {
        return;
    }
    let mut bad: Vec<&'static str> = Vec::new();
    for p in &m.progs {
        for inst in &p.insts {
            let Some(op) = DevOp::ALL.iter().copied().find(|o| *o as u16 == inst.op) else {
                continue;
            };
            let c = op.c_name();
            if NVIDIA_UNIMPLEMENTED_GEMM.contains(&c) && !bad.contains(&c) {
                bad.push(c);
            }
        }
    }
    assert!(
        bad.is_empty(),
        "this packet carries {} gfx950-only GEMM opcode(s) the NVIDIA interpreter has no arm for: \
         {bad:?}. Prefill would `__trap()` → CUDA_ERROR_LAUNCH_FAILED. `pick_tile` must run under \
         `with_emit_target_amd(false, …)` on sm_90a/sm_120a so only Gemm/GemmMed/GemmSmall (and \
         fp8/mxfp4 twins) are emitted.",
        bad.len()
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
    if amd && emit_config::active().uniseg {
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
    // Same no-target rule as run_verified's guard: legacy (golden) emission stays AMD.
    let _emit_target = EmitAmdGuard::set(amd || (arch.is_empty() && gpu.is_empty()));
    // Same reason as the other entry: the tile selector must cost against THIS part, and
    // `--arch` is the fallback when --gpu cannot answer.
    if amd {
        set_amd_target_for(&arch, &gpu);
    }
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
    let ecfg = emit_config::active();
    let fp8 = ecfg.any_fp8_weights();
    // T8 w8a8 (PLOW_W8A8=1, requires PLOW_FP8=1). PREFILL emits the true fp8 tensor-core path:
    // ONE per-row DevOp::QuantFp8 per activation site + GEMM_FP8/GEMM_GLU_FP8 re-pointed at the
    // fp8 activation (t1=xq) + a_scale (t3). The SAME opcodes serve T6 w8a16 (bf16 activation) —
    // the interp cubin selects the kernel by PLOW_NV_W8A8, so the w8a8 pkt MUST run against a
    // PLOW_NV_W8A8=1 prefill cubin (the T6 cubin would misread xq bytes as bf16). Weight side =
    // the same e4m3 twins + per-channel scales T6 declared. Unset => byte-identical emission.
    let w8a8 = ecfg.w8a8;
    assert!(
        !(w8a8 && ecfg.w8a16),
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
        !ecfg.mxfp4,
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
    let fp8_kv = ecfg.fp8_kv;
    // PLOW_FP8_KV_FULL=1: restrict the e4m3 cache to FULL-attention (hd512) layers — the shape
    // the beat-fp8-mma PIPE=1 fp8-mma prefill flash serves. Requires PLOW_FP8_KV=1.
    let fp8_kv_full = fp8_kv && ecfg.fp8_kv_full;
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
    // Tensor-parallel now emits SHARDED PREFILL buckets too: every prefill
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
    let ladder_wave = !amd && ecfg.pf_ladder.as_deref() == Some("wave");
    // T32 (PLOW_PF_LADDER_APPEND=4224[,..]): extra rungs. The chat template pushes a
    // 4096-token benchmark prompt to ~4110 rows, which chunks [4096, 128] — and the 128
    // tail is a SECOND full-model pass (~36 ms measured, weight restream + packet floors).
    // A 4096+128 rung swallows it in one chunk for 114 pad rows; the runtime chunk-cost
    // model picks it automatically (fewer launches at equal padding). Needs PLOW_MAX_CHUNK
    // >= the rung (the cap below filters otherwise).
    let append: Vec<u32> = ecfg
        .pf_ladder_append
        .as_deref()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<u32>().ok())
                .filter(|&x| x <= cap)
                .collect()
        })
        .unwrap_or_default();
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
    let buckets: Vec<u32> = {
        let mut b = buckets;
        for a in append {
            if !b.contains(&a) {
                b.push(a);
            }
        }
        b.sort_unstable();
        b
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
    // old value, byte-identical). See the design notes.
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
    // The rung ladder, ASCENDING. `[decode_batch]` when PLOW_DECODE_BATCH_LADDER is unset,
    // so `dbatch` below is the value it always was and the emit is byte-identical.
    let rungs: Vec<u32> = ecfg.decode_rungs();
    // Every per-slot resource is sized at the WIDEST rung, and that is the whole design:
    // slot `s`'s offset into the KV cache is `s * (kv_head*ring*hd)` — INVARIANT in B — so a
    // sequence keeps its slot while the program under it changes rung to rung.
    let dbatch: u32 = *rungs.last().expect("decode_rungs is non-empty");
    // 26B-A4B MoE decode is BATCHED (B in 1..=32): the router family, the flat expert GLU/down
    // and the combine all carry a batch row count and index [B][k] routing slots. See the
    // work-item ordering note in runtime/nvidia/op_moe.cuh for the weight-reuse design.
    // (The fp8 batch refusal is gone: the fp8 GEMV arms are batched as of the B=32 work.)
    assert!(
        !(c.moe && dbatch > 32),
        "MoE decode batch is capped at 32 (per-CTA inv[] scratch, PLOW_MOE_MAXB)"
    );

    // Grouped-MoE PREFILL: token-sorted grouped expert GEMM buckets.
    // Enabled by default for the 26B-A4B MoE bf16 path; PLOW_MOE_PREFILL=0 restores the decode-only
    // blob (byte-identical to the pre-prefill build — the buffer sizing and new tensors are gated on
    // this flag). beat26b: fp8 grouped MoE prefill is now implemented for the w8a8 path (ops 81/82),
    // so it is enabled under PLOW_W8A8; plain fp8 (w8a16 dequant) grouped prefill is still not
    // implemented and stays decode-only.
    let moe_pf = c.moe && (!fp8 || w8a8) && ecfg.moe_prefill.as_deref() != Some("0");

    // Phase 1: the DenseGqaEmitter owns the dense
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
        b.set_fuse_materialized_residual_inputs(ecfg.fuse_residual_input);
        b.adopt_tensors(tensors.clone());
        b.set_l2_placement(l2_layout); // PLOW_L2_PLACE: None ⇒ byte-identical
        b.set_lean_moe_stage2_segments(amd && emit_config::active().moe_stage2_lean);
        b.set_lean_moe_stage1_segments(amd && emit_config::active().moe_stage1_lean);
        b.set_lean_moe_combine_segments(amd && emit_config::active().moe_combine_lean);
        b.set_lean_kda_intra_segments(amd && emit_config::active().kda_intra_cached);
        b.set_kda_intra_wave_items_segments(
            amd && amd_target::active().1 == hwspec::IsaLevel::Gfx950
                && emit_config::active().kda_intra_wave_items,
        );
        b.set_lean_kda_key_factor_segments(
            amd && amd_target::active().1 == hwspec::IsaLevel::Gfx950
                && emit_config::active().kda_key_factor,
        );
        if amd {
            b.deny_uniseg(); // PLOW_UNISEG collapses the wave-class split — see `warn_uniseg_amd`
        }
        // T18 (PLOW_UNISEG_MAX_T=<t>): small buckets emit ONE segment so the serve side takes
        // the single-launch fat path — a ~50-token tail chunk pays ~480 segment launches
        // (~40 ms) for ~5 ms of work otherwise. Big buckets keep the segmented classes.
        if let Some(mx) = emit_config::active().uniseg_max_t {
            if t <= mx {
                b.force_uniseg();
            }
        }
        emitter.emit_prefill(&mut b, t);
        progs.push(b.finish());
        tlist.push(t);
    }
    let mut kv_rows = Vec::new();
    // `dbatch` is the SAME clamped(1,32) value used by declare() above — emission and
    // allocation must agree, so we reuse it here rather than re-reading the env (an unclamped
    // re-read would emit B>32 ops against buffers declare() sized for 32 → OOB writes).
    // DECODE-TILED (PLOW_DECODE_TILED=1): emit the decode bucket from
    // PREFILL kernels — tiled GEMM + FlashPrefill at one query row — instead of the GEMV family.
    // Targets long context, where GEMV does not scale with batch and FlashDecode caps at n_cu.
    // Unset emits a byte-identical program. **The sm_120 interpreter traps on every prefill
    // opcode** (interp_sm120.cu default arm), so this is AMD-only until those kernels exist; it
    // is a loud trap, not silent garbage. Correctness bar is a token stream IDENTICAL to the
    // Mode::Decode bucket at the same prompt — not "it ran".
    let dmode = if ecfg.decode_tiled {
        Mode::DecodeTiled
    } else {
        Mode::Decode
    };
    // THE DECODE BATCH LADDER (PLOW_DECODE_BATCH_LADDER=1,2,4,8,16). One decode program per
    // rung, ASCENDING, all sharing the tensor table `declare()` sized at the WIDEST rung. The
    // list is `[dbatch]` when the knob is unset, so this loop runs once and emits exactly the
    // program it always emitted — byte-identical, which `dense_blob_is_byte_identical` pins.
    //
    // WHY ASCENDING, i.e. why the WIDEST rung is last. `prog_t.last()` is the decode program by
    // convention (`Model::prog_t`, `manifest.rs`, `AmdEngine::load`'s `in.kvlen` cross-check),
    // and `in.kvlen` is sized at the widest rung because that is the slot count the KV cache
    // holds. Putting the narrowest rung last would make the blob refuse itself at load.
    //
    // The two ranges must not overlap or `decode_rung_lo` cannot separate them — see the assert.
    assert!(
        rungs.iter().all(|&r| buckets.iter().all(|&pb| pb > r)),
        "decode rungs {rungs:?} overlap the prefill bucket ladder {buckets:?}: the blob carries \
         no field distinguishing the two and `packet::devbuild::decode_rung_lo` separates them \
         by width. Emit with prefill buckets wider than the widest decode rung."
    );
    for (ri, &rb) in rungs.iter().enumerate() {
        let mut bd = Builder::new(n_cu);
        bd.set_fuse_materialized_residual_inputs(ecfg.fuse_residual_input);
        bd.adopt_tensors(tensors.clone());
        bd.set_l2_placement(l2_layout); // PLOW_L2_PLACE: None ⇒ byte-identical
        if amd {
            bd.deny_uniseg();
        }
        bd.set_gemv_split(gemv_split()); // PLOW_GEMV_SPLIT: 1 (default) ⇒ byte-identical
                                         // `kv_row_insts` indexes the LAST program (the one `AmdEngine::decode_prog` returns and
                                         // `patch_kvrow` writes into), so only the widest rung contributes. Under a ladder the
                                         // list is dead anyway: every rung carries `i[6] = n_batch_kv` and takes its KV write row
                                         // from `pos[]`, which is exactly why the one-row rung is correct without a host patch.
        let mut scratch = Vec::new();
        let sink = if ri + 1 == rungs.len() {
            &mut kv_rows
        } else {
            &mut scratch
        };
        emitter.emit_decode(&mut bd, rb, dmode, sink);
        progs.push(bd.finish());
        tlist.push(rb);
    }

    // Fold the GEN_TMAP_BF16 mint registry into the Model tables (see TmapMint: the
    // per-program Builders adopt clones, so the registry is the only durable record).
    // Empty (and byte-identical) unless PLOW_TMA_GEMM=1.
    let (mut tensors, mut gen) = (tensors, gen);
    {
        let (td, tg) = emitter.take_tmaps();
        assert!(
            td.is_empty() || tg[0].tensor == tensors.len() as u32,
            "tmap registry base drifted from the declared tensor count"
        );
        tensors.extend(td);
        gen.extend(tg);
    }

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
    // BLOCK MODE: embed the block.json descriptor
    // as SECT_METADATA — this also forces the to_blob_v6 path — and drop a
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
        Err(e) if ecfg.skip_coverage => {
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
    check_nvidia_opcode_coverage(&m, amd);
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
        let cpath = std::path::Path::new(&out).with_file_name("plow_config.h");
        manifest::write_config_header(&cpath, &man)
            .unwrap_or_else(|e| panic!("{}: compile config not written: {e}", cpath.display()));
        eprintln!("  compile config -> {}", cpath.display());
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
/// `PLOW_GEMV_WG` — cap the dispatch width of the dense-path DECODE GEMVs (rows-per-wave
/// probe; the GLM twin `PLOW_GLM_GEMV_WG` measured −1.4 ms at cap 152 on gfx942). Unset ⇒
/// byte-identical. Applied at the decode call sites only — prefill GEMM splitting is
/// tile-based and must not see it.
fn gemv_wg_env() -> Option<u32> {
    emit_config::active().gemv_wg.filter(|&c| c > 0)
}
fn gemv_wg_n(n: u32) -> u32 {
    gemv_wg_env().map_or(n, |c| n.min(c))
}
fn gemv_wg_cap(cus: Vec<u32>) -> Vec<u32> {
    match gemv_wg_env() {
        Some(c) if (c as usize) < cus.len() => cus[..c as usize].to_vec(),
        _ => cus,
    }
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
#[path = "lib_tests/gemma_router_emit.rs"]
mod gemma_router_emit_tests;

#[cfg(test)]
#[path = "lib_tests/mode.rs"]
mod mode_tests;

#[cfg(test)]
#[path = "lib_tests/gfx950_coverage.rs"]
mod gfx950_coverage_tests;

/// `require_mla_rope`: the MLA positional-encoding contract, pinned.
///
/// Every case is stated as "this config JSON, therefore this outcome", with the expected theta
/// read back OUT of the same JSON rather than written as a literal — so if the value moves the
/// test moves with it instead of quietly asserting the old number.
#[cfg(test)]
#[path = "lib_tests/mla_rope.rs"]
mod mla_rope_tests;

#[cfg(test)]
#[path = "lib_tests/fp8_key.rs"]
mod fp8_key_tests;

#[cfg(test)]
#[path = "lib_tests/fp8_profile.rs"]
mod fp8_profile_tests;

#[cfg(test)]
#[path = "lib_tests/pick_tile.rs"]
mod pick_tile_tests;

#[cfg(test)]
#[path = "lib_tests/chunk_default.rs"]
mod chunk_default_tests;
