/* interp_sm120.cu — the plow persistent on-device packet interpreter for sm_120a
 * (GB202 / RTX 5090). Grown from interp_sm120_poc.cu: same launch model, same counter
 * protocol, real PLOW_DOP_* op bodies in place of the PoC's two synthetic ones.
 *
 * WHAT IS LIVE: the 11 opcodes the Qwen3-4B DECODE program actually contains, as measured
 * (401 packets, prog 2, T=1) — EMBED, RMSNORM, HEADNORM_ROPE, GEMV_QKV, FLASH_DECODE,
 * FLASH_MERGE, GEMV, ADD_NORM, GEMV_GLU, ARGMAX, ARGMAX_FIN. RESIDUAL and GLU are also
 * wired because their kernels exist and are free to carry.
 *
 * WHAT IS NOT: prefill. The T=128 and T=512 buckets need GEMM_SMALL, GEMM_MED, GEMM_GLU and
 * FLASH_PREFILL, none of which has a validated sm_120 kernel yet. They hit the default arm,
 * which TRAPS rather than silently producing zeros — see the default case. Nothing here is
 * a stub that returns a wrong answer quietly.
 *
 * THE COUNTER PROTOCOL IS THE PoC'S, VERBATIM, and deliberately so: it is validated at
 * 0/200 failures across fan-out, fan-in (threshold 1020) and a 12-stage chain. It is not
 * redesigned here, only extended from 2 ops to 13.
 *
 * Build:
 *   nvcc -arch=sm_120a -I ../common -I . -c interp_sm120.cu
 * Occupancy/spill gate (must show 0 bytes spill and >= 1 block/SM):
 *   nvcc -arch=sm_120a -I ../common -I . -Xptxas -v -c interp_sm120.cu
 */
#include "dev_isa.h"

/* T10 occ-2 arena trim. The lean GEMM segment object must fit 2 blocks/SM under the 100 KiB
 * dynamic-smem cap. The default GEMM arena is 60 KiB (PGM_STAGES=3 plain / GLU_STAGES=2), so 2x =
 * 120 KiB > 100 KiB and ptxas caps occupancy at 1 (SMEM-bound, NOT register-bound — the register
 * gate at 128 already passes). Shrinking the pipeline depth to 2/1 drops the arena to 40 KiB
 * (2x = 80 KiB < 100 KiB) so ptxas reaches occ-2. Set BEFORE op_gemm.cuh, which now #ifndef-guards
 * these. Only the _pfgemm object; every other object keeps the tuned 3/2 depth. */
#if defined(PLOW_NV_SEG_GEMM) && PLOW_NV_SEG_GEMM
#if defined(PLOW_NV_SEG_GEMM_BN64) && PLOW_NV_SEG_GEMM_BN64
/* PX-3 (rtx-11): reach occ-2 by SHRINKING the N-tile to 64 instead of halving the pipeline.
 * BN=64 keeps the tuned 3-stage plain / 2-stage GLU ring; the arena is 3*(ABUF+BBUF) =
 * 3*(5120+2560) = 23040 bf16 = 45 KiB (plain) / 40 KiB (GLU), so 2 blocks/SM = 90 KiB < the
 * 100 KiB dynamic-smem cap. This is the pipeline T10 could NOT keep at BN=128 (60 KiB → occ-1). */
#define PGM_BN 64
#else
/* T10 default lean object: BN=128 but a HALVED pipeline (2/1) to squeeze the arena to 40 KiB. */
#define PGM_STAGES 2
#define PGM_GLU_STAGES 1
#endif
#endif

#include "op_attention.cuh" /* validated: d_flash_decode / d_flash_merge (harvested) */
#include "op_mla.cuh"        /* MLA (DeepSeek/GLM/Kimi) latent decode + fused merge-fold (P1) */
#include "op_dsa.cuh"        /* GLM DSA indexer: score (mma.sync) + top-k select (P3) */
#include "op_elementwise.cuh"
#include "op_gemm.cuh"
#include "op_norm.cuh"
#include "op_moe.cuh"        /* Gemma-4 26B-A4B bf16 MoE decode bodies (router/glu/down/combine) */
#include "op_mamba.cuh"      /* Nemotron-3 Mamba-2 SSD mixer (UNVERIFIED on GPU; nvcc-compile only) */

#include <cstdio>
#include <cuda_runtime.h>

/* ---- counter protocol ------------------------------------------------------------------
 * COPIED VERBATIM from interp_sm120_poc.cu. Validated at 0/200 failures across fan-out,
 * fan-in (threshold 1020) and a 12-stage chain. Not redesigned.
 *
 * Relaxed poll: `volatile` forces each read past L1 to L2 — the single coherence point on a
 * GB202 (one L2, unlike CDNA's per-XCD L2), so device scope suffices. */
/* ---- PTX SCOPED SYNC (PLOW_NV_PTXSYNC=1) ------------------------------------------------
 * The protocol above is AMD-shaped: a RELAXED poll, then a separate full device fence
 * (`__threadfence()` = `membar.gl`, i.e. seq_cst), then a plain `atomicAdd`. That is the only
 * vocabulary CDNA gives you. NVIDIA sm_70+ has a scoped memory model, so the ordering can ride
 * ON the access instead of standing next to it as a barrier:
 *
 *   acquire : `ld.acquire.gpu.u32`        replaces  volatile load + membar.gl
 *   release : `red.release.gpu.add.u32`   replaces  membar.gl + atomicAdd
 *
 * Three savings per packet, and the second is STRUCTURAL rather than a cheaper instruction:
 *   1. `membar.gl` is seq_cst and orders all prior traffic in both directions; an acquire load
 *      orders only what FOLLOWS it. Strictly less ordering for the same correctness.
 *   2. The acquire fence today must be run by thread 0 alone, which costs a `__syncthreads()`
 *      on EACH side just to scope it. Riding acquire on the polling threads' own load makes
 *      the leading barrier unnecessary — the trailing `__syncthreads()` already publishes the
 *      ordering to the rest of the block. That removes a whole block barrier per packet.
 *   3. `red` is a reduction with no return value; `atom` returns one. The bumps discard the
 *      result, so `red` avoids a pointless result-bus round trip.
 *
 * SEMANTICS PRESERVED, which matters because the gate comment says the acquire is load-bearing
 * and must not be removed. It is NOT removed — it MOVES ONTO THE LOAD. `.gpu` scope (not
 * `.cta`) is required: the producers are other blocks. Correctness bar is an identical token
 * stream, not a faster number. */
#ifndef PLOW_NV_PTXSYNC
#define PLOW_NV_PTXSYNC 1
#endif

#if PLOW_NV_PTXSYNC == 1
/* V1 — ordering rides ON the access. THIS IS THE MEASURED WIN, and now the DEFAULT (the
 * #ifndef above defaults to 1). It shipped opt-in in f42ca0f; re-measured on current HEAD it
 * still reproduces and still returns bit-identical tokens, so the default is flipped on:
 *
 *   RE-MEASUREMENT (Gemma-4-12B decode, ctx 3587, 112 timed steps after 16 warmup, median ms):
 *     ref (PTXSYNC=0)  18.4002 / 18.4025 / 18.3901     mean 18.398
 *     V1  (PTXSYNC=1)  18.2862 / 18.2870 / 18.2878     mean 18.287     -0.60%  (~8 sigma, sd 0.013)
 *   PLOW_IDS md5-identical across all six runs. Smaller than the original -0.95% (the kernel has
 *   grown since f42ca0f and ctx here is 3587 not 4096) but a clear, reproducible win.
 *
 *   ORIGINAL (f42ca0f):
 *   ctx    ref (2 runs)      V1 (2 runs)       delta
 *   4096   6.6454 / 6.6552   6.5823 / 6.5917   -0.95%
 *   8192   6.9079 / 6.9120   6.8474 / 6.8490   -0.89%
 *  16384   7.7002 (1 run)    7.6415 / 7.6360   -0.80%   [ref n=1, weaker]
 *
 * Interleaved and ORDER-REVERSED inside one gpulease session, per-run sd 0.005-0.016 ms, so
 * ~0.06 ms is 6-7 sigma. Token stream md5-identical to ref at every context and argmax AGREE.
 * 32768 is NOT measured.
 *
 * READ THE SASS CAREFULLY BEFORE "FIXING" THIS. Static counts say V1 should lose: membars go
 * 2 -> 31 because `red.release.gpu` does not fuse (ptxas emits MEMBAR.ALL.GPU + REDG) and
 * ctr_signal inlines ~31 times. That reasoning is WRONG and cost a measurement to disprove:
 * those 31 are 31 SITES, exactly one of which runs per packet. Static instruction count is not
 * dynamic instruction count. The win is real and comes from three places:
 *   - MEMBAR.ALL.GPU (acq_rel) replaces MEMBAR.SC.GPU (seq_cst) — a weaker, cheaper fence
 *   - one fewer BAR.SYNC per packet (36 -> 35): the acquire no longer needs a barrier to scope
 *     it to thread 0
 *   - REDG (no return) replaces ATOMG (returns a value nobody reads)
 * V2 below isolates which of the three matters: it takes the REDG and the scope fix but KEEPS
 * the seq_cst fences, and it measures as a wash (+0.16%). So the fence strength is the lever;
 * the atomic form and the load scope are not. */
__device__ __forceinline__ uint32_t ctr_poll(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void ctr_signal(uint32_t* p) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(p) : "memory");
}
#elif PLOW_NV_PTXSYNC == 2
/* V2 — keep the fence STRUCTURE exactly as the validated build has it (one acquire
 * `__threadfence()` after the gate, one release before the bumps) and change only the two
 * things the SASS says are actually wasteful:
 *
 *   1. POLL SCOPE. `volatile` compiles to `LDG.E.STRONG.SYS` — SYSTEM scope, which forces
 *      coherence against the host and any peer GPU. The producers are other blocks on THIS
 *      device, so `.gpu` is the correct scope and `.sys` is pure overhead on the spin loop
 *      that the gate sits in. (The existing comment reasons that device scope suffices on a
 *      one-L2 GB202 — it is right, but `volatile` does not express that; this does.)
 *   2. BUMP FORM. `atomicAdd` returns a value nobody uses, compiling to `ATOMG.E.ADD`. `red`
 *      is the no-return reduction: same effect, no result-bus round trip.
 *
 * Relaxed is correct for BOTH because the surrounding `__threadfence()`s are retained — the
 * ordering still comes from the fences, exactly as in the validated protocol. This is a
 * strictly narrower change than V1.
 *
 * MEASURED: 6.6576 / 6.6638 at ctx 4096 vs ref 6.6454 / 6.6552 — **+0.16%, a wash**. Kept as
 * the CONTROL that localises V1's win: V2 has V1's REDG and V1's narrowed load scope but not
 * its weaker fence, and it wins nothing. Therefore neither the atomic form nor the .sys->.gpu
 * poll scope is worth anything here — the seq_cst-to-acq_rel fence downgrade (and the barrier
 * it removes) is the entire effect. Do not ship V2; do not re-test it. */
__device__ __forceinline__ uint32_t ctr_poll(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void ctr_signal(uint32_t* p) {
    asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;" :: "l"(p) : "memory");
}
#elif PLOW_NV_PTXSYNC == 3
/* V3 — V1 with the acquire HOISTED OUT of the spin loop. V1's `ld.acquire.gpu` lowers to
 * LD.E.STRONG.GPU + CCTL.IVALL: a full L1 invalidate EVERY poll iteration, per polling
 * thread, trashing the SM's L1 while the other warps of the same block sit in the gate.
 * The acquire is only needed ONCE, after the final read that observed the threshold — so
 * spin RELAXED (plain LDG.E.STRONG.GPU, no CCTL) and issue one `fence.acquire.gpu` after
 * the whole gate loop exits (see the gate site), on the same observing threads; the
 * trailing __syncthreads() publishes the ordering to the block exactly as V1 argues for
 * its in-loop acquire. Signal side is identical to V1 (red.release.gpu).
 *
 * OPT-IN until A/B-measured against V1: the V1/V2 history above is explicit that static
 * reasoning about this protocol has been wrong before. Semantically this is V1 with the
 * acquire moved from every-iteration to loop-exit, which is where acquire semantics are
 * actually consumed. */
__device__ __forceinline__ uint32_t ctr_poll(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void ctr_signal(uint32_t* p) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(p) : "memory");
}
#else
__device__ __forceinline__ uint32_t ctr_poll(const uint32_t* p) {
    return *(const volatile uint32_t*)p;
}
#endif

/* ---- SINGLE-BLOCK TRACE (PLOW_NV_TRACE=1) ----------------------------------------------
 * Per-packet cycle attribution for ONE block, to answer where a packet's time actually goes:
 * gate (waiting on producers), body (the math), signal (fences + successor bumps).
 *
 * Only block 0 records, and only thread 0 writes — so the trace costs the other 169 blocks
 * nothing, and block 0 pays four clock64() reads per packet. It still perturbs: clock64() is
 * a serializing read, and the recording thread is the same thread 0 that does the signalling.
 * So READ THE SHAPE, NOT THE ABSOLUTE TOTAL — the ratio between gate/body/signal is the
 * finding; the sum will over-report against the untraced 6.6451 ms baseline.
 *
 * Compiled out entirely when PLOW_NV_TRACE=0 (the default): no buffer, no clock reads, no
 * register pressure, so the shipped interpreter is unchanged. */
#ifndef PLOW_NV_TRACE
#define PLOW_NV_TRACE 0
#endif
#if PLOW_NV_TRACE
/* 32493 stream entries over ~170 co-resident blocks is ~191 packets/block; 4096 is slack for
 * a badly skewed GQ claim distribution. Overflow stops recording rather than scribbling. */
#define PLOW_TRACE_MAX 4096
__device__ unsigned      g_tr_n;
__device__ unsigned      g_tr_op[PLOW_TRACE_MAX];
__device__ unsigned      g_tr_wait[PLOW_TRACE_MAX];
__device__ unsigned long long g_tr_gate[PLOW_TRACE_MAX];
__device__ unsigned long long g_tr_body[PLOW_TRACE_MAX];
__device__ unsigned long long g_tr_sig[PLOW_TRACE_MAX];
#endif
#if !PLOW_NV_PTXSYNC
/* Release before signal / acquire after the gate — one device-scope fence, exactly where
 * interp.hip puts its agent-scope buffer_wbl2 / buffer_inv. */
__device__ __forceinline__ void ctr_signal(uint32_t* p) { atomicAdd(p, 1u); }
#endif

/* 24-byte PlowStreamEnt fetched as 3x LDG.E.64 instead of the 6 scalar LDG.E the plain
 * struct copy compiles to (the struct is declared 4-aligned, so nvcc cannot widen it).
 * Every entry IS 8-aligned: the stream base is a cudaMalloc'd (256-aligned) allocation and
 * sizeof(PlowStreamEnt) == 24 is a multiple of 8. On the ~401-packets/token fixed path this
 * removes 3 issue slots + 3 L1 transactions per packet. */
__device__ __forceinline__ PlowStreamEnt ld_stream_ent(const PlowStreamEnt* p) {
    PLOW_SASSERT(sizeof(PlowStreamEnt) == 24, "stream entry must stay 24 B / 8-B multiple");
    PlowStreamEnt e;
    const uint2 d0 = ((const uint2*)p)[0];
    const uint2 d1 = ((const uint2*)p)[1];
    const uint2 d2 = ((const uint2*)p)[2];
    ((uint2*)&e)[0] = d0;
    ((uint2*)&e)[1] = d1;
    ((uint2*)&e)[2] = d2;
    return e;
}

/* ---- GQA fusion factor -----------------------------------------------------------------
 * PLOW_FA_GF(128) is hardcoded to 2 in runtime/common/dev_isa.h — a Gemma artifact. Qwen3-4B
 * is gqa = 32/8 = 4, and GF=2 there was MEASURED at 1.71x the cost of GF=4 (each K/V row
 * crosses HBM twice instead of once). So this build ships GF=4.
 *
 * ONE instantiation, not a runtime switch. The interpreter inlines every op body, so its
 * register allocation is the WORST CASE over all of them; instantiating GF=1/2/4 would pull
 * all three into that worst case for no benefit. A Gemma build overrides the macro.
 *
 * The correctness condition is gqa % GF == 0 (d_flash_decode indexes by head GROUP, which is
 * what makes GF < gqa correct). It cannot be a static_assert because gqa is a runtime packet
 * field, so it is checked at dispatch and trapped — never silently mis-attended. */
#ifndef PLOW_NV_FA_GF
#define PLOW_NV_FA_GF 4
#endif
#ifndef PLOW_NV_FA_GF_FULL
/* Gemma mixes hd256/GQA2 sliding attention with hd512 full attention in one
 * program.  Keep the sliding kernel at GF=2, but allow the full-attention
 * instantiation to be swept independently.  Defaulting to PLOW_NV_FA_GF keeps
 * every existing build byte-for-byte equivalent at the source-configuration
 * level; RTX experiments opt in with -DPLOW_NV_FA_GF_FULL=4 or 8. */
#define PLOW_NV_FA_GF_FULL PLOW_NV_FA_GF
#endif
#if PLOW_NV_FA_GF_FULL != 1 && PLOW_NV_FA_GF_FULL != 2 && PLOW_NV_FA_GF_FULL != 4 && \
    PLOW_NV_FA_GF_FULL != 8
#error "PLOW_NV_FA_GF_FULL must be one of {1,2,4,8}"
#endif
#define PLOW_NV_FA_HD 128 /* Qwen3 head_dim; the only instantiation the DEFAULT build carries */

/* ---- GEMMA build (PLOW_NV_GEMMA=1) ------------------------------------------------------
 * Gemma-4 decode uses head_dim 256 on its sliding layers and 512 on its full layers IN THE
 * SAME program. Sliding attention is fixed at GF=2 because its GQA is 2; full attention uses
 * PLOW_NV_FA_GF_FULL, which defaults to the same GF=2 but can be swept at GF=4/8 when the model's
 * local GQA permits it. A Gemma object drops the hd=128 arm and instantiates headnorm/flash/merge
 * at both 256 and 512, dispatching on the runtime head_dim field. It also carries the Gemma-only op bodies
 * NORM_RESIDUAL(16), NORM_RESIDUAL_NORM(23) and SOFTCAP(7). The DEFAULT (Qwen3) object is
 * byte-identical to before: every Gemma arm below is behind `#if PLOW_NV_GEMMA`.
 *
 * The megakernel inlines every arm, so its register/occupancy footprint is the WORST CASE over
 * all of them — normally the hd512/full-GF flash arm. Occupancy is re-checked at build (ptxas -v) and
 * pinned to 1 block/SM by __launch_bounds__(256,1); the cooperative launch refuses an oversized
 * grid, so a regression shows up as a launch error, not silent garbage. */
#ifndef PLOW_NV_GEMMA
#define PLOW_NV_GEMMA 0
#endif

/* ---- GEMMA PREFILL build (PLOW_NV_PREFILL=1, implies PLOW_NV_GEMMA=1) --------------------
 * The prefill bucket runs a different op family than decode: tiled mma.sync GEMM (q/k/v/o/down/
 * lm_head, plus the gate|up GEMM_GLU) and the multi-query FLASH_PREFILL (hd 256 sliding / 512 full),
 * with FLASH_MERGE folding the split-KV partials on the short buckets. Those arms are register- and
 * smem-hungry (hd=512 O accumulators, a 128x64 GEMM tile), so — per rtx-04 §3 / the rtx-06 register
 * warning — they build as a SEPARATE object from decode rather than stacking onto the decode
 * megakernel's 150-reg / hd512-flash-decode footprint. The DECODE-only arms (GEMV family,
 * FLASH_DECODE) compile OUT here; the prefill-only arms compile out of the decode object. Exported
 * symbols are suffixed `_pf` so both objects link into one harness. The default Qwen3 object and the
 * Gemma DECODE object are byte-identical to before (PLOW_NV_PREFILL=0). */
#ifndef PLOW_NV_PREFILL
#define PLOW_NV_PREFILL 0
#endif
#if PLOW_NV_PREFILL && !PLOW_NV_GEMMA
#error "PLOW_NV_PREFILL requires PLOW_NV_GEMMA (hd 256/512 dispatch lives behind the Gemma gate)"
#endif

/* ---- w8a8 fp8 prefill GEMM (PLOW_NV_W8A8=1, rtx-07 T7 L2) --------------------------------
 * The compute-bound fix: the GEMM_FP8 opcodes dispatch to the true w8a8 mma.sync.m16n8k32.e4m3
 * path (d_gemm_w8a8 / d_gemm_glu_w8a8, op_gemm.cuh) with BOTH operands e4m3 and a per-M-row
 * activation quant (QUANT_FP8 op). Default 0 keeps T6's w8a16 dequant path as the A/B control:
 * the SAME opcodes then read t1=A as bf16 and take only w_scale. Only the prefill object carries
 * these; QUANT_FP8 is needed by the w8a8 path (activation half) so it lives behind the flag too. */
#ifndef PLOW_NV_W8A8
#define PLOW_NV_W8A8 0
#endif

/* ---- fp8 (e4m3) KV cache (PLOW_FP8_KV=1, rtx-19 E3) --------------------------------------
 * Stores K/V in the cache as e4m3 (1 byte/elem, HALF the bf16 footprint) with a per-(token,kv_head)
 * f32 dequant scale; the decode flash reads the fp8 cache and dequants ×scale in the inner loop
 * (op_attention.cuh d_flash_decode<...,FP8KV=true>), the KV write quantizes at the norm's store
 * (op_norm.cuh d_headnorm_rope_fp8). Default 0 => the fp8 KV op-arms compile OUT entirely, so the
 * bf16 megakernel is BYTE-IDENTICAL. When 1, the fp8-KV packet's HEADNORM_ROPE_FP8 / FLASH_DECODE_FP8
 * opcodes are handled; q's norm stays plain HEADNORM_ROPE (q is not cached). */
#ifndef PLOW_FP8_KV
#define PLOW_FP8_KV 0
#endif

/* ---- SEGMENTED DISPATCH (PLOW_NV_SEGMENTS=1, T9c) ----------------------------------------
 * The AMD interpreter (interp.hip) relaunches once per wave-class SEGMENT with a distinct kernel
 * object per segment class (RUNSEG). The sm_120 interpreter has run every prefill program as ONE
 * cooperative launch over a single segment; that pins the whole program to one occupancy/register
 * profile. This flag ports the AMD model: the host relaunches once per segment with prog.cur_seg,
 * and the interp bounds its work to that segment's window (GQ: [gq_seg_ofs[cur_seg], ...+1); STATIC:
 * skips entries whose seg != cur_seg). Default 0 keeps the existing single-segment object BYTE-
 * IDENTICAL — cur_seg is never read, so the SASS is unchanged. Only the segmented prefill object
 * (built PLOW_NV_SEGMENTS=1) reads cur_seg and carries the distinct `_pfseg` symbol suffix. */
#ifndef PLOW_NV_SEGMENTS
#define PLOW_NV_SEGMENTS 0
#endif

/* ---- LEAN GEMM SEGMENT OBJECT (PLOW_NV_SEG_GEMM=1, T9c Stage 2) --------------------------
 * Design A: give the GEMM/tier-A segments their OWN kernel object, targeting occupancy 2. The
 * register/smem-hungry FLASH_PREFILL/FLASH_MERGE arms compile OUT (a GEMM segment never contains
 * them), the dynamic arena shrinks to the GEMM tile alone (≤50 KiB, so 2×<100 KiB), and
 * __launch_bounds__ caps registers at 128 so ptxas targets 2 blocks/SM. Flash segments keep the
 * occ-1 _pfseg object. Implies PLOW_NV_SEGMENTS (built with both). Distinct `_pfgemm` symbols. */
#ifndef PLOW_NV_SEG_GEMM
#define PLOW_NV_SEG_GEMM 0
#endif
#if PLOW_NV_SEG_GEMM && !PLOW_NV_SEGMENTS
#error "PLOW_NV_SEG_GEMM requires PLOW_NV_SEGMENTS=1 (it is a segmented object)"
#endif

/* ---- DECODE GF8 TWIN (PLOW_NV_GF8_TWIN=1, beat12b-ctx-switch) ----------------------------
 * A SECOND decode object built from the same source with PLOW_NV_FA_GF_FULL=8 (full-attn GQA
 * fusion 8: each of the 12B's single-global-KV-head bytes is read 2x instead of 8x). It is NOT a
 * prefill object — it carries the identical decode op family as the shipped GF2 decode object, so
 * the ONLY functional difference is the full-attn flash instantiation and its register/arena size
 * (234 vs 209 regs, 16448 vs 12352 B arena). Suffix `_gf8` so it co-links with the unsuffixed GF2
 * decode object; the host picks the launcher by kvlen (ctx-regime switching). Default 0 => the flag
 * is inert and the decode object is byte-identical. The macro only sets the SYMBOL suffix; the GF
 * value itself comes from -DPLOW_NV_FA_GF_FULL=8 on the CMake target (kept explicit so the object's
 * identity is legible at the build line). */
#ifndef PLOW_NV_GF8_TWIN
#define PLOW_NV_GF8_TWIN 0
#endif
#if PLOW_NV_GF8_TWIN && PLOW_NV_PREFILL
#error "PLOW_NV_GF8_TWIN is a DECODE twin (no prefill arms); do not combine with PLOW_NV_PREFILL"
#endif

/* ---- LEAN DECODE GEMV OBJECT (PLOW_NV_LEAN_DECODE=1, segmented-decode Step 1) ------------
 * The DECODE analog of PLOW_NV_SEG_GEMM: a decode object with the register-hungry attention
 * arms (d_flash_decode<512,GF>, the MLA/DSA latent-attn ops, flash-merge) compiled OUT, so it
 * carries ONLY the memory-bound GEMV family (+ norm/rope/embed/argmax). d_flash_decode<512>
 * owns the decode object's 208-reg / 1-block-SM ceiling; dropping it lets ptxas + a
 * PLOW_NV_FORCE_MINBLK cap reach 2-3 blocks/SM — the isolated-probe occupancy that lifts the
 * gemv from ~21% to ~46-58% of HBM. This is the high-occupancy GEMV segment object; the
 * flash-decode segment stays on the occ-1 object (analogous to the AMD 8-wave main + 4-wave
 * flash split, interp.hip:430). Default 0 keeps every existing decode object BYTE-IDENTICAL
 * (the arms are present); the trapped-out opcodes never appear in a GEMV-only segment. */
#ifndef PLOW_NV_LEAN_DECODE
#define PLOW_NV_LEAN_DECODE 0
#endif
#if PLOW_NV_LEAN_DECODE && PLOW_NV_PREFILL
#error "PLOW_NV_LEAN_DECODE is a DECODE object (no prefill arms); do not combine with PLOW_NV_PREFILL"
#endif

/* ---- DEAD-ARM GATING for model-specialized objects (h100-interp arm-ablation) -----------
 * MLA (DeepSeek/GLM/Kimi latent attention), MAMBA (Nemotron-3 SSD mixer) and DSA (GLM sparse
 * lightning-indexer) are op families a Gemma model NEVER emits. They follow the PLOW_NV_GEMMA /
 * PLOW_NV_W8A8 precedent: DEFAULT 1 keeps every existing object BYTE-IDENTICAL (the sm_120 cubins
 * and the op-test build never pass these flags). build_sm90a_cubin.sh passes =0 for the Gemma
 * serving objects it builds. A gated-out opcode falls through to `default: __trap()`, so a
 * mis-targeted packet fails loudly rather than silently emitting zeros.
 *
 * MEASURED PAYOFF (arm ablation, sm_90a): these do NOT relieve the PREFILL REG=255 ceiling — that
 * is owned by the LIVE Hopper wgmma GEMM arms (d_gemm_sm90 / d_gemm_w8a8_sm90 / d_gemm_glu_sm90),
 * each of which rounds to 255 in the megakernel regardless. What they buy is size and stack:
 *   DECODE: MLA is ~44% of the cubin (2.46 MB -> 1.42 MB) and owns the decode REG ceiling
 *           (208 -> 188, occupancy-neutral: both are 1 block/SM). DSA/MAMBA trim a little more.
 *   PREFILL: MAMBA owns the stack frame (1744 -> 688 B). MLA/DSA are not compiled into prefill
 *            (they sit behind `#if !PLOW_NV_PREFILL`), so their flags are no-ops there. */
#ifndef PLOW_NV_MLA
#define PLOW_NV_MLA 1
#endif
#ifndef PLOW_NV_MAMBA
#define PLOW_NV_MAMBA 1
#endif
#ifndef PLOW_NV_DSA
#define PLOW_NV_DSA 1
#endif

/* Two-level paste is intentional: interp_sm90a.cu aliases the public SM120
 * identifiers before including this shared implementation. A one-level
 * `n##_pf` suppresses expansion of that alias and silently exports an SM120
 * symbol from an SM90a cubin. */
#define PLOW_NV_CAT_I(a, b) a##b
#define PLOW_NV_CAT(a, b) PLOW_NV_CAT_I(a, b)
#if PLOW_NV_PREFILL && PLOW_NV_SEG_GEMM
#define PLOW_SYM(n) PLOW_NV_CAT(n, _pfgemm)
#elif PLOW_NV_PREFILL && PLOW_NV_SEGMENTS
#define PLOW_SYM(n) PLOW_NV_CAT(n, _pfseg)
#elif PLOW_NV_PREFILL
#define PLOW_SYM(n) PLOW_NV_CAT(n, _pf)
#elif PLOW_NV_GF8_TWIN
#define PLOW_SYM(n) PLOW_NV_CAT(n, _gf8)
#else
#define PLOW_SYM(n) n
#endif

/* occ target: 2 blocks/SM for the lean GEMM object, 1 for everything else.
 * PLOW_NV_FORCE_MINBLK overrides it for occupancy experiments — notably the
 * DECODE object (C2): decode is bandwidth-STARVED at 1 block/SM (12.5% occ,
 * ~21% of peak HBM), and its arms (gemv + flash-DECODE, no flash-PREFILL) may
 * fit 128 regs for 2 blocks/SM. Capping regs spills, so this is an A/B: the
 * occupancy gain must beat the spill traffic on a memory-bound kernel. */
#if defined(PLOW_NV_FORCE_MINBLK)
#define PLOW_NV_MINBLK PLOW_NV_FORCE_MINBLK
#elif PLOW_NV_SEG_GEMM
#define PLOW_NV_MINBLK 2
#else
#define PLOW_NV_MINBLK 1
#endif

/* ---- dynamic smem arena ----------------------------------------------------------------
 * A UNION across op bodies, not a sum: only one op body runs at a time in a block, and each
 * fully consumes its arena before the next instruction's gate. Sized by the largest claim,
 * which is flash-decode's Ssm+hmax+hsum+qsm+osm. The Gemma build covers both its fixed hd256/GF2
 * sliding arm and the independently selected hd512/full-GF arm. */
#if PLOW_NV_PREFILL
/* Prefill union: max over flash-prefill (256/512 tilings), the tiled GEMM, and FLASH_MERGE. The
 * hd=256 flash tile (BQ64,BKV32: Qs+KsT+Vs bf16 + Ss f32) dominates at 19840 floats (77.5 KiB)
 * — opt-in past 48 KiB. The KsT transpose + Ss score tile are the mma.sync QK^T additions (T1). */
#define PLOW_NV_PRE_A                                                                          \
    (FA_PRE_SMEM_FLOATS(256, 64, 32) > FA_PRE_SMEM_FLOATS(512, 32, 16)                         \
         ? FA_PRE_SMEM_FLOATS(256, 64, 32)                                                     \
         : FA_PRE_SMEM_FLOATS(512, 32, 16))
#define PLOW_NV_PRE_B ((PGM_ARENA_BF16 + 1) / 2)
#if PLOW_NV_SEG_GEMM
/* Lean GEMM object: no flash arm, so the arena is the GEMM tile alone (PLOW_NV_PRE_B). This is
 * the ≤50 KiB claim that lets 2 blocks/SM fit under the 100 KiB dynamic-smem cap. */
#define PLOW_NV_FA_ARENA PLOW_NV_PRE_B
#else
#define PLOW_NV_FA_ARENA (PLOW_NV_PRE_A > PLOW_NV_PRE_B ? PLOW_NV_PRE_A : PLOW_NV_PRE_B)
#endif
#elif PLOW_NV_GEMMA
#if PLOW_NV_LEAN_DECODE
/* Lean GEMV segment object: the flash arms are compiled out, so their arena claims drop to 0.
 * The arena collapses to the 2*WARPS-float floor below — no flash tile means no smem pressure,
 * so occupancy is register-bound (the point of this object). */
#define PLOW_NV_FULL_ARENA 0
#define PLOW_NV_SLIDE_ARENA 0
#else
#define PLOW_NV_FULL_ARENA FA_DEC_SMEM_FLOATS(512, PLOW_NV_FA_GF_FULL)
#define PLOW_NV_SLIDE_ARENA FA_DEC_SMEM_FLOATS(256, 2)
#endif
/* MLA latent decode (P1/P2). The GF ladder instantiates GF={2,4,8} unconditionally (GF=8 is the
 * emitter default for ctx>4096, 1.5-1.9x faster — P2 §7), so the arena must cover the GF=8 claim
 * (MLA_DEC_SMEM_FLOATS(512,64,8) = 6416 f = 25.1 KiB). Still dwarfed by the DSA score tile. */
/* Gated (PLOW_NV_MLA/PLOW_NV_DSA): when the arm is compiled out its arena claim drops to 0 so the
 * embedded plow_arena_bytes reflects the actual object (DSA's 33.6 KiB tile is the largest decode
 * claim, so gating DSA off shrinks the launch's dynamic-smem request). */
#if PLOW_NV_MLA
#define PLOW_NV_MLA_ARENA MLA_DEC_SMEM_FLOATS(512, 64, 8)
#else
#define PLOW_NV_MLA_ARENA 0
#endif
/* GLM DSA indexer SCORE (P3): the streamed-K + [head][pos] dump tile (DI=128, HI=32, TILE_N=64 =
 * 8608 f = 33.6 KiB) — the largest decode-side claim, dwarfing MLA's 4240 f. Select uses only the
 * global histogram/ctl tensors (no arena); LayerNorm uses the 2*WARPS `part` slot (covered below). */
#if PLOW_NV_DSA
#define PLOW_NV_DSA_ARENA DSA_SCORE_SMEM_FLOATS(128, 32, DSA_SCORE_TILE_N)
#else
#define PLOW_NV_DSA_ARENA 0
#endif
#define PLOW_NV_FA_ARENA0                                                                    \
    (PLOW_NV_FULL_ARENA > PLOW_NV_SLIDE_ARENA ? PLOW_NV_FULL_ARENA : PLOW_NV_SLIDE_ARENA)
#define PLOW_NV_FA_ARENA1                                                                    \
    (PLOW_NV_FA_ARENA0 > PLOW_NV_MLA_ARENA ? PLOW_NV_FA_ARENA0 : PLOW_NV_MLA_ARENA)
#define PLOW_NV_FA_ARENA                                                                     \
    (PLOW_NV_FA_ARENA1 > PLOW_NV_DSA_ARENA ? PLOW_NV_FA_ARENA1 : PLOW_NV_DSA_ARENA)
#else
#define PLOW_NV_FA_ARENA FA_DEC_SMEM_FLOATS(PLOW_NV_FA_HD, PLOW_NV_FA_GF)
#endif
#define PLOW_NV_ARENA_FLOATS                                                                   \
    (PLOW_NV_FA_ARENA > 2 * (int)PLOW_NV_WARPS ? PLOW_NV_FA_ARENA : 2 * (int)PLOW_NV_WARPS)
/* block_max_u64 needs PLOW_NV_WARPS u64 = 2*WARPS floats; block_sum needs WARPS floats. Both
 * fit inside the flash claim at any supported head dim, but the max above keeps that true if
 * flash is ever compiled out. */

/* The object's dynamic-smem contract, embedded IN THE CUBIN so a loader that only has the
 * module image (plowrt serve: cuModuleGetGlobal) launches with the arena this object was
 * compiled for. plow_sm120_smem() serves the statically linked harness instead; it does not
 * exist in a cubin. Without this, serve guessed the GF=2 default (12352 B) and a GF_FULL=4
 * flash-decode (16448 B) indexed past the arena — CUDA_ERROR_ILLEGAL_ADDRESS on the first
 * decode step. Cubin-only (scripts/build_sm120_cubin.sh sets PLOW_NV_EMBED_SMEM): nvcc's
 * host pass emits a registration reference but no host shadow for an extern "C" device
 * global, so linking it into the harness executable would fail. */
#ifndef PLOW_NV_EMBED_SMEM
#define PLOW_NV_EMBED_SMEM 0
#endif
#if PLOW_NV_EMBED_SMEM
extern "C" __device__ unsigned PLOW_SYM(plow_arena_bytes) = PLOW_NV_ARENA_FLOATS * sizeof(float);
/* Capability flag (cuModuleGetGlobal, like plow_arena_bytes): this object's
 * HeadNormRope derives the KV write row from pos[t] whenever n_batch_kv != 0
 * — so the engine may set i[6]=1 on a B=1 decode program's KV-write sites at
 * load and never patch i[3] again (immutable decode instruction stream, plan
 * plowrt-gpu-exec-critical-path stage 2). Absent on older cubins → the engine
 * keeps the legacy per-token patch. */
extern "C" __device__ unsigned PLOW_SYM(plow_dyn_kvrow) = 1;
#endif

/* ---- the op dispatch -------------------------------------------------------------------
 * Mirrors runtime/amd/interp.hip's switch. Operand slots are taken from the EMITTER
 * (crates/plowc/src/bin/gemma4.rs) and the AMD CONSUMER, never from crates/packet/src/dev.rs
 * — that file's doc comments are stale and wrong in three places relevant here (it omits
 * HeadNormRope i4=skip_norm and i5=interleave, omits FlashDecode i7=kv_mask, and omits
 * Gemv i4=a_row0).
 *
 * PLOW_TENSOR_NONE (0xFFFF) is the ONLY absent sentinel. Tensor handle 0 is a real
 * tensor — in.ids is handle 0 and is read by EMBED and written by ARGMAX_FIN.
 *
 * PLOW_T extracts a handle from tw_, the per-packet REGISTER copy of t[8] below —
 * one 16-byte vector load instead of ~150 per-site LDG.E.U16s, which measured
 * ~+0.4% decode TPOT (~3-4 cycles of dependent unpack per dispatch). t[] is at
 * offset 16 of a 64-byte record, so the access is 16-byte aligned (asserted in
 * dev_isa.h). k is always a literal, so tw_[k>>1] resolves to a register. */
#define PLOW_T(k) ((tw_[(k) >> 1] >> (((k) & 1) * 16)) & 0xFFFFu)
#define TEN(k) (PLOW_T(k) == PLOW_TENSOR_NONE ? nullptr : T[PLOW_T(k)])

__device__ __forceinline__ void plow_exec(const PlowDevInst* in, void* const* T, unsigned slice,
                                          unsigned nblk, float* arena) {
    const uint4 tv_ = *reinterpret_cast<const uint4*>(in->t);
    const unsigned tw_[4] = {tv_.x, tv_.y, tv_.z, tv_.w};
#if defined(PLOW_NV_ABLATE_LO) || defined(PLOW_NV_ABLATE_HI)
    /* MEASUREMENT ONLY (produces garbage logits, like PLOW_NV_SKELETON). Skip the BODY of any
     * opcode in the 128-bit mask while keeping every gate and signal intact, so the TPOT delta
     * is that op set's true wall-clock contribution AT THE SHIPPED GRID -- imbalance included,
     * which per-block trace attribution cannot give. Never ship. */
#ifndef PLOW_NV_ABLATE_LO
#define PLOW_NV_ABLATE_LO 0ull
#endif
#ifndef PLOW_NV_ABLATE_HI
#define PLOW_NV_ABLATE_HI 0ull
#endif
    {
        const unsigned op_ = (unsigned)in->op;
        const unsigned long long m_ = op_ < 64u ? (PLOW_NV_ABLATE_LO) : (PLOW_NV_ABLATE_HI);
        if ((m_ >> (op_ & 63u)) & 1ull) return;
    }
#endif
    switch (in->op) {
    /* ---- norms ---- */
    case PLOW_DOP_RMSNORM:
        d_rmsnorm((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                  (const __nv_bfloat16*)TEN(2), in->i[0], in->i[1], in->fj[0].f, in->i[2], slice,
                  nblk, arena);
        break;

    case PLOW_DOP_ADD_NORM:
        d_add_norm((__nv_bfloat16*)TEN(0), (__nv_bfloat16*)TEN(1), (const __nv_bfloat16*)TEN(2),
                   (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4), in->i[0], in->i[1],
                   in->fj[0].f, slice, nblk, arena);
        break;

#if PLOW_NV_GEMMA
    /* Gemma SANDWICH residual: out = (a + RMSNorm(b, gamma)) * scale. f1 folds layer_scalar.
     * Emitted only on the prefill (non-gfuse) Gemma path; carried here for completeness. */
    case PLOW_DOP_NORM_RESIDUAL:
        d_norm_residual((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                        (const __nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(3), in->i[0],
                        in->i[1], in->fj[0].f, in->fj[1].f, slice, nblk, arena);
        break;

    /* Gemma fused sandwich residual + next-sublayer norm, both Gemma residual sites in decode:
     * resid = (a + RMSNorm(b, gb)) * scale ; out = RMSNorm(resid, gn). */
    case PLOW_DOP_NORM_RESIDUAL_NORM:
        d_norm_residual_norm((__nv_bfloat16*)TEN(0), (__nv_bfloat16*)TEN(1),
                             (const __nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(3),
                             (const __nv_bfloat16*)TEN(4), (const __nv_bfloat16*)TEN(5), in->i[0],
                             in->i[1], in->fj[0].f, in->fj[1].f, slice, nblk, arena);
        break;

    /* Gemma final-logit softcap: out = cap * tanh(x / cap), cap = f0 = 30. */
    case PLOW_DOP_SOFTCAP:
        d_softcap((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1), in->i[0], in->fj[0].f, slice,
                  nblk);
        break;
#endif

#if PLOW_NV_PREFILL
    /* ---- PREFILL tiled GEMM (q/k/v/o/down/lm_head; one body, three tile opcodes) ----
     * t0=C t1=A t2=B  i0=M i1=N i2=K  i4=a_row0. No epilogue activation. */
    case PLOW_DOP_GEMM:
    case PLOW_DOP_GEMM_MED:
    case PLOW_DOP_GEMM_SMALL:
        d_gemm((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1), (const __nv_bfloat16*)TEN(2),
               in->i[0], in->i[1], in->i[2], in->i[4], slice, nblk, (__nv_bfloat16*)arena);
        break;

    /* Prefill gate|up GEMM with fused GLU epilogue. t0=fu t1=A t2=Wg t5=Wu  i0=M i1=N i2=K i5=act. */
    case PLOW_DOP_GEMM_GLU:
        d_gemm_glu((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                   (const __nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(5), in->i[0], in->i[1],
                   in->i[2], in->i[5], slice, nblk, (__nv_bfloat16*)arena);
        break;

#if PLOW_NV_W8A8
    /* fp8 w8a8 prefill GEMM (T7 L2). t0=C t1=A(e4m3) t2=W(e4m3) t3=a_scale(f32[M]) t4=w_scale(f32[N])
     * i0=M i1=N i2=K i4=a_row0. TRUE fp8 tensor cores (m16n8k32), two-scale dequant epilogue. */
    case PLOW_DOP_GEMM_FP8:
    case PLOW_DOP_GEMM_MED_FP8:
    case PLOW_DOP_GEMM_SMALL_FP8:
        d_gemm_w8a8((__nv_bfloat16*)TEN(0), (const uint8_t*)TEN(1), (const uint8_t*)TEN(2),
                    (const float*)TEN(3), (const float*)TEN(4), in->i[0], in->i[1], in->i[2],
                    in->i[4], slice, nblk, (__nv_bfloat16*)arena);
        break;

    /* fp8 w8a8 prefill GEMM+GLU (T7 L2). t0=fu t1=A(e4m3) t2=Wg(e4m3) t5=Wu(e4m3)
     * t3=a_scale t4=g_scale t6=u_scale  i0=M i1=N i2=K i5=act. */
    case PLOW_DOP_GEMM_GLU_FP8:
        d_gemm_glu_w8a8((__nv_bfloat16*)TEN(0), (const uint8_t*)TEN(1), (const uint8_t*)TEN(2),
                        (const uint8_t*)TEN(5), (const float*)TEN(3), (const float*)TEN(4),
                        (const float*)TEN(6), in->i[0], in->i[1], in->i[2], in->i[5], slice, nblk,
                        (__nv_bfloat16*)arena);
        break;

    /* Per-row fp8 activation quant (T7 L2). t0=xq(e4m3) t1=x(bf16) t2=a_scale(f32[M])  i0=M i1=K. */
    case PLOW_DOP_QUANT_FP8:
        d_quant_fp8((uint8_t*)TEN(0), (const __nv_bfloat16*)TEN(1), (float*)TEN(2), in->i[0],
                    in->i[1], slice, nblk);
        break;
#else
    /* fp8 w8a16 prefill GEMM (T6 L2). t0=C t1=A(bf16) t2=W(e4m3) t4=w_scale(f32[N])
     * i0=M i1=N i2=K i4=a_row0. dequant-to-bf16-in-smem, per-channel scale in the epilogue. */
    case PLOW_DOP_GEMM_FP8:
    case PLOW_DOP_GEMM_MED_FP8:
    case PLOW_DOP_GEMM_SMALL_FP8:
        d_gemm_fp8((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1), (const uint8_t*)TEN(2),
                   (const float*)TEN(4), in->i[0], in->i[1], in->i[2], in->i[4], slice, nblk,
                   (__nv_bfloat16*)arena);
        break;

    /* fp8 w8a16 prefill GEMM+GLU (T6 L2). t0=fu t1=A(bf16) t2=Wg(e4m3) t5=Wu(e4m3)
     * t4=g_scale t6=u_scale  i0=M i1=N i2=K i5=act. */
    case PLOW_DOP_GEMM_GLU_FP8:
        d_gemm_glu_fp8((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1), (const uint8_t*)TEN(2),
                       (const uint8_t*)TEN(5), (const float*)TEN(4), (const float*)TEN(6), in->i[0],
                       in->i[1], in->i[2], in->i[5], slice, nblk, (__nv_bfloat16*)arena);
        break;
#endif

    /* ---- Gemma-4 26B-A4B bf16 grouped-MoE PREFILL (plans/p9-26b-prefill-moe.md) ----
     * DEAD in a dense (12B/31B) GEMM segment; compiled OUT of the lean occ-2 object to relieve
     * register pressure toward 0 spill. A 26B MoE program would run its expert GEMV/GLU segments on
     * the occ-1 _pfseg object instead. Case gating only — op_moe.cuh (T9a) is untouched. */
#if !PLOW_NV_SEG_GEMM
    case PLOW_DOP_MOE_ROUTER_GEMMA_PF:
        d_moe_router_gemma_pf((unsigned char*)TEN(0), (const __nv_bfloat16*)TEN(1),
                              (const __nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(3),
                              (const __nv_bfloat16*)TEN(4), in->i[0], in->i[1], in->i[2],
                              in->i[3], in->fj[0].f, in->fj[1].f, slice, nblk, arena);
        break;

    case PLOW_DOP_MOE_ALIGN_GEMMA_PF:
        d_moe_align_gemma_pf((int*)TEN(0), (const unsigned char*)TEN(1), (unsigned*)TEN(2),
                             (unsigned*)TEN(3), (float*)TEN(4), in->i[0], in->i[1], in->i[2],
                             slice);
        break;

    case PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF:
        d_moe_group_glu_gemma_pf((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                                 (const unsigned long long*)TEN(2), (const int*)TEN(3),
                                 (const unsigned*)TEN(4), in->i[0], in->i[1], in->i[2], in->i[5],
                                 slice, nblk, (__nv_bfloat16*)arena);
        break;

    case PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF:
        d_moe_group_down_gemma_pf((float*)TEN(0), (const __nv_bfloat16*)TEN(1),
                                  (const unsigned long long*)TEN(2), (const int*)TEN(3),
                                  (const unsigned*)TEN(4), (const float*)TEN(5), in->i[0],
                                  in->i[1], in->i[2], slice, nblk, (__nv_bfloat16*)arena);
        break;

    case PLOW_DOP_MOE_COMBINE_NORM_GEMMA_PF:
        d_moe_combine_norm_gemma_pf((__nv_bfloat16*)TEN(0), (const float*)TEN(1),
                                    (const __nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(3),
                                    in->i[0], in->i[1], in->i[2], in->fj[0].f, slice, nblk, arena);
        break;

#if PLOW_NV_W8A8
    case PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF_W8A8: /* beat26b: fp8 w8a8 grouped gate/up + GeGLU */
        d_moe_group_glu_gemma_pf_w8a8((__nv_bfloat16*)TEN(0), (const uint8_t*)TEN(1),
                                      (const float*)TEN(5), (const unsigned long long*)TEN(2),
                                      (const unsigned long long*)TEN(6), (const int*)TEN(3),
                                      (const unsigned*)TEN(4), in->i[0], in->i[1], in->i[2],
                                      in->i[5], slice, nblk, (__nv_bfloat16*)arena);
        break;

    case PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF_W8A8: /* beat26b: fp8 w8a8 grouped down + scatter */
        d_moe_group_down_gemma_pf_w8a8((float*)TEN(0), (const uint8_t*)TEN(1), (const float*)TEN(7),
                                       (const unsigned long long*)TEN(2),
                                       (const unsigned long long*)TEN(6), (const int*)TEN(3),
                                       (const unsigned*)TEN(4), (const float*)TEN(5), in->i[0],
                                       in->i[1], in->i[2], slice, nblk, (__nv_bfloat16*)arena);
        break;
#endif
#endif /* !PLOW_NV_SEG_GEMM */

    /* Multi-query causal/sliding flash. t0=Opart t1=mlpart t2=Q t3=K t4=V t5=at(fused).
     * i0=seq_q i1=seq_kv i2=n_head i3=n_kv_head i4=q_pos0 i5=window i6=hd i7=nsplit.
     * j0=kv_stride(ring) j1=kv_mask  f0=scale. hd 256 (sliding) or 512 (full).
     * t6 (NONE on every legacy bf16 packet; host-patched in PX-1 batched-prefill mode) is the
     * packed chunk's request table — see d_flash_prefill_mux. */
#if !PLOW_NV_SEG_GEMM /* the lean GEMM object never runs flash — compile it OUT to hit occ-2 */
    case PLOW_DOP_FLASH_PREFILL:
        if (in->i[6] == 256)
            d_flash_prefill_mux<256, 64, 32>(
                (const int*)TEN(6), (float*)TEN(0), (float*)TEN(1), (const __nv_bfloat16*)TEN(2),
                (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4), (__nv_bfloat16*)TEN(5),
                in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->i[5], in->i[7], in->fj[1].u,
                in->fj[2].u, in->fj[0].f, slice, nblk, arena);
        else if (in->i[6] == 512)
            d_flash_prefill_mux<512, 32, 16>(
                (const int*)TEN(6), (float*)TEN(0), (float*)TEN(1), (const __nv_bfloat16*)TEN(2),
                (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4), (__nv_bfloat16*)TEN(5),
                in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->i[5], in->i[7], in->fj[1].u,
                in->fj[2].u, in->fj[0].f, slice, nblk, arena);
        else
            __trap();
        break;

#if PLOW_FP8_KV
    /* fp8-KV prefill READ: dequant the e4m3 cache (t3=K t4=V) ×per-row scale (t6=k_scale t7=v_scale)
     * at the smem stage, mma unchanged. Uses the PIPE=0 synchronous-staging arm (cp.async cannot
     * convert fp8 inline), so the fp8-KV prefill object MUST be built -DPLOW_NV_FA_PIPE=0. No batched
     * (PX-1) prefill for fp8 (t6/t7 are the scales, not the request table). */
    case PLOW_DOP_FLASH_PREFILL_FP8:
#if PLOW_NV_FA_PIPE
#if PLOW_NV_FA_FP8MMA && PLOW_FP8_KV
        /* beat-fp8-mma: the PIPE=1 fp8 prefill IS the px4 fp8-mma arm — hd512 FULL layers only
         * (mixed fp8-KV packets, PLOW_FP8_KV_FULL=1, never emit hd256 fp8 prefill; a legacy
         * all-layer fp8-KV packet on this object traps loudly rather than reading garbage). */
        if (in->i[6] == 512)
            d_flash_prefill_px4<512, 32, 16, true>(
                (float*)TEN(0), (float*)TEN(1), (const __nv_bfloat16*)TEN(2),
                (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4), (__nv_bfloat16*)TEN(5),
                in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->i[5], in->i[7], in->fj[1].u,
                in->fj[2].u, in->fj[0].f, slice, nblk, arena, (const float*)TEN(6),
                (const float*)TEN(7));
        else
            __trap();
#else
        __trap(); /* fp8 prefill needs PIPE=0 or the fp8-mma arm; neither is in this object */
#endif
#else
        if (in->i[6] == 256)
            d_flash_prefill<256, 64, 32, true>(
                (float*)TEN(0), (float*)TEN(1), (const __nv_bfloat16*)TEN(2),
                (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4), (__nv_bfloat16*)TEN(5),
                in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->i[5], in->i[7], in->fj[1].u,
                in->fj[2].u, in->fj[0].f, slice, nblk, arena, (const float*)TEN(6), (const float*)TEN(7));
        else if (in->i[6] == 512)
            d_flash_prefill<512, 32, 16, true>(
                (float*)TEN(0), (float*)TEN(1), (const __nv_bfloat16*)TEN(2),
                (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4), (__nv_bfloat16*)TEN(5),
                in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->i[5], in->i[7], in->fj[1].u,
                in->fj[2].u, in->fj[0].f, slice, nblk, arena, (const float*)TEN(6), (const float*)TEN(7));
        else
            __trap();
#endif
        break;
#endif /* PLOW_FP8_KV */
#endif /* !PLOW_NV_SEG_GEMM */
#endif

    /* i2 selects the head-dim template; i5==1 selects the INTERLEAVED (GPT-J) rotate.
     * Only hd=128 non-interleaved is instantiated in this build (Qwen3). Anything else
     * traps rather than falling through to a wrong head dim. */
    case PLOW_DOP_HEADNORM_ROPE:
#if PLOW_NV_GEMMA
        /* Gemma: head_dim is 256 (sliding) or 512 (full), both non-interleaved.
         * PREFILL object only: t6 (NONE on legacy bf16 packets; host-patched in PX-1 batched
         * mode on KV-write sites) is the per-row seq-slot map — see d_headnorm_rope. The decode
         * object never passes it, keeping its SASS byte-identical. */
#if PLOW_NV_PREFILL
#define PLOW_HNR_SLOT , (const int*)TEN(6)
#else
#define PLOW_HNR_SLOT
#endif
        /* GLM/Kimi MLA decoupled RoPE: hd=64 partial-rope slice is ALWAYS interleaved
         * (GPT-J), i[5] ignored — GLM is the only hd=64 user, matching runtime/amd/
         * interp.hip. GLM DSA indexer q_idx/k_idx: hd=128 interleaved (i[5]==1). Gemma
         * full/sliding attention: hd=256/512 non-interleaved (i[5]==0). */
        if (in->i[2] == 64)
            d_headnorm_rope<64, /*INTERLEAVE=*/true>(
                (__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                (const __nv_bfloat16*)TEN(2), (const float*)TEN(3), (const float*)TEN(4),
                (const int*)TEN(5), in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                in->i[4], slice, nblk, in->i[6] PLOW_HNR_SLOT);
        else if (in->i[2] == 128 && in->i[5] == 1)
            d_headnorm_rope<128, /*INTERLEAVE=*/true>(
                (__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                (const __nv_bfloat16*)TEN(2), (const float*)TEN(3), (const float*)TEN(4),
                (const int*)TEN(5), in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                in->i[4], slice, nblk, in->i[6] PLOW_HNR_SLOT);
        else if (in->i[2] == 256 && in->i[5] == 0)
            d_headnorm_rope<256>(
                (__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                (const __nv_bfloat16*)TEN(2), (const float*)TEN(3), (const float*)TEN(4),
                (const int*)TEN(5), in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                in->i[4], slice, nblk, in->i[6] PLOW_HNR_SLOT);
        else if (in->i[2] == 512 && in->i[5] == 0)
            d_headnorm_rope<512>(
                (__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                (const __nv_bfloat16*)TEN(2), (const float*)TEN(3), (const float*)TEN(4),
                (const int*)TEN(5), in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                in->i[4], slice, nblk, in->i[6] PLOW_HNR_SLOT);
        else
            __trap();
#undef PLOW_HNR_SLOT
#else
        if (in->i[2] == PLOW_NV_FA_HD && in->i[5] == 0)
            d_headnorm_rope<PLOW_NV_FA_HD>(
                (__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                (const __nv_bfloat16*)TEN(2), (const float*)TEN(3), (const float*)TEN(4),
                (const int*)TEN(5), in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                in->i[4], slice, nblk, in->i[6]);
        else
            __trap();
#endif
        break;

#if PLOW_FP8_KV
    /* fp8-KV write: k/v norm STORES the cache as e4m3 (t0=uint8) + per-row f32 scale (t6). Same
     * norm+RoPE math as HEADNORM_ROPE; q stays bf16 HEADNORM_ROPE above. pfslot unused (t6 is the
     * scale here, not the PX-1 slot map — fp8-KV does not compose with batched prefill). */
    case PLOW_DOP_HEADNORM_ROPE_FP8:
#if PLOW_NV_GEMMA
        if (in->i[5] != 0) { __trap(); break; }
        if (in->i[2] == 256)
            d_headnorm_rope_fp8<256>(
                (uint8_t*)TEN(0), (float*)TEN(6), (const __nv_bfloat16*)TEN(1),
                (const __nv_bfloat16*)TEN(2), (const float*)TEN(3), (const float*)TEN(4),
                (const int*)TEN(5), in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                in->i[4], slice, nblk, in->i[6]);
        else if (in->i[2] == 512)
            d_headnorm_rope_fp8<512>(
                (uint8_t*)TEN(0), (float*)TEN(6), (const __nv_bfloat16*)TEN(1),
                (const __nv_bfloat16*)TEN(2), (const float*)TEN(3), (const float*)TEN(4),
                (const int*)TEN(5), in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                in->i[4], slice, nblk, in->i[6]);
        else
            __trap();
#else
        if (in->i[2] == PLOW_NV_FA_HD && in->i[5] == 0)
            d_headnorm_rope_fp8<PLOW_NV_FA_HD>(
                (uint8_t*)TEN(0), (float*)TEN(6), (const __nv_bfloat16*)TEN(1),
                (const __nv_bfloat16*)TEN(2), (const float*)TEN(3), (const float*)TEN(4),
                (const int*)TEN(5), in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                in->i[4], slice, nblk, in->i[6]);
        else
            __trap();
#endif
        break;
#endif /* PLOW_FP8_KV */

    /* ---- pointwise ---- */
    case PLOW_DOP_EMBED:
        d_embed((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1), (const int*)TEN(2),
                in->i[0], in->i[1], in->fj[0].f, slice, nblk);
        break;

    case PLOW_DOP_RESIDUAL:
        d_residual((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                   (const __nv_bfloat16*)TEN(2), in->i[0], in->fj[0].f, slice, nblk);
        break;

    case PLOW_DOP_GLU:
        d_glu((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1), (const __nv_bfloat16*)TEN(2),
              in->i[0], in->i[1], slice, nblk);
        break;

#if !PLOW_NV_PREFILL
    /* ---- GEMV family (DECODE object only; the prefill object uses the tiled GEMM arms above) ----
     * i4=a_row0 is a row offset applied to x in units of K (undocumented in dev.rs; 0 on
     * every Qwen packet). i3=norm_flag selects the fused-norm GEMV, which this build does
     * NOT carry — trapped rather than silently skipping the norm. */
    case PLOW_DOP_GEMV:
        if (in->i[3] != 0) { __trap(); break; }
        if (in->i[2] <= PLOW_NV_ARENA_FLOATS * 2u)
            d_gemv((__nv_bfloat16*)TEN(0),
                   (const __nv_bfloat16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                   (const __nv_bfloat16*)TEN(2), in->i[0], in->i[1], in->i[2], slice, nblk,
                   (__nv_bfloat16*)arena);
        else
            d_gemv((__nv_bfloat16*)TEN(0),
                   (const __nv_bfloat16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                   (const __nv_bfloat16*)TEN(2), in->i[0], in->i[1], in->i[2], slice, nblk);
        break;

#if PLOW_NV_GEMMA
    /* E5 (rtx-19): fused lm_head GEMV + greedy-argmax epilogue (PLOW_FUSE_ARGMAX). M is always 1
     * (greedy decode); x is offset by a_row0 (i4) exactly as GEMV. t3=part(u64[nblk]); f0=cap. */
    case PLOW_DOP_GEMV_ARGMAX:
        d_gemv_argmax((__nv_bfloat16*)TEN(0),
                      (const __nv_bfloat16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                      (const __nv_bfloat16*)TEN(2), (unsigned long long*)TEN(3), in->i[1],
                      in->i[2], in->fj[0].f, slice, nblk, (__nv_bfloat16*)arena);
        break;
#endif

    /* Kernel arg order is (Nq, Nk, Nv, K): K lives in i2 but is passed LAST. */
    case PLOW_DOP_GEMV_QKV:
        d_gemv_qkv((__nv_bfloat16*)TEN(0), (__nv_bfloat16*)TEN(3), (__nv_bfloat16*)TEN(5),
                   (const __nv_bfloat16*)TEN(1), (const __nv_bfloat16*)TEN(2),
                   (const __nv_bfloat16*)TEN(4), (const __nv_bfloat16*)TEN(6), in->i[0], in->i[1],
                   in->i[3], in->i[4], in->i[2], slice, nblk, (__nv_bfloat16*)arena);
        break;

    case PLOW_DOP_GEMV_GLU:
        d_gemv_glu((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                   (const __nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(5), in->i[0], in->i[1],
                   in->i[2], in->i[5], slice, nblk, (__nv_bfloat16*)arena);
        break;

#if PLOW_NV_GEMMA
    /* ---- fp8 (w8a16) weight-only DECODE GEMV family (Gemma gate) ----
     * FFMA dequant-on-load, per-output-channel scale in the epilogue (plans/rtx-05 §T2).
     * Weight is e4m3 (uint8), so TEN(2)/TEN(5) are cast to uint8*; the scale(s) are f32.
     * GEMV_FP8   t0=C t1=x t2=W(fp8) t5=w_scale(f32[N])  i0=M i1=N i2=K i4=a_row0.
     * GEMV_GLU_FP8 t0=fu t1=x t2=Wg(fp8) t5=Wu(fp8) t3=g_scale t4=u_scale  i0=M i1=N i2=K i5=act. */
    case PLOW_DOP_GEMV_FP8:
        if (in->i[2] <= PLOW_NV_ARENA_FLOATS * 2u)
            d_gemv_fp8((__nv_bfloat16*)TEN(0),
                       (const __nv_bfloat16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                       (const uint8_t*)TEN(2), (const float*)TEN(5), in->i[0], in->i[1], in->i[2],
                       slice, nblk, (__nv_bfloat16*)arena);
        else
            d_gemv_fp8((__nv_bfloat16*)TEN(0),
                       (const __nv_bfloat16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                       (const uint8_t*)TEN(2), (const float*)TEN(5), in->i[0], in->i[1], in->i[2],
                       slice, nblk);
        break;

    case PLOW_DOP_GEMV_GLU_FP8:
        d_gemv_glu_fp8((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                       (const uint8_t*)TEN(2), (const uint8_t*)TEN(5), (const float*)TEN(3),
                       (const float*)TEN(4), in->i[0], in->i[1], in->i[2], in->i[5], slice, nblk,
                       (__nv_bfloat16*)arena);
        break;

    /* ---- block-fp8 (128x128 block-scaled) DENSE FFN, GLM/Kimi/DeepSeek (P1.5) ----
     * op_moe.cuh d_dense_glu_fp8_blk / d_gemv_fp8_blk (warp-per-output, block-scale dot).
     * DENSE_GLU_FP8_BLK t0=fu t1=x t2=Wg t5=Wu t3=Sg t4=Su  i0=N(inter) i1=K(hidden) i5=act.
     * GEMV_FP8_BLK (dense DOWN) t0=C t1=fu t2=W t5=scale  i0=M i1=N(hidden) i2=K(inter) i4=x_row. */
    case PLOW_DOP_DENSE_GLU_FP8_BLK:
        d_dense_glu_fp8_blk((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                            (const unsigned char*)TEN(2), (const unsigned char*)TEN(5),
                            (const float*)TEN(3), (const float*)TEN(4), in->i[0], in->i[1],
                            in->i[5], slice, nblk);
        break;
    case PLOW_DOP_GEMV_FP8_BLK:
        d_gemv_fp8_blk((__nv_bfloat16*)TEN(0),
                       (const __nv_bfloat16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                       (const unsigned char*)TEN(2), (const float*)TEN(5), in->i[1], in->i[2],
                       slice, nblk);
        break;
#endif

#if PLOW_NV_SZ
    /* ---- SplitZip (bf16 lossless) DECODE GEMV twins (p9-v2 C-1) ----
     * BIT-IDENTICAL bf16 outputs; the reconstruct hides in the weight load shadow. */
    case PLOW_DOP_GEMV_SZ:
        d_gemv_sz((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1), (const uint8_t*)TEN(2),
                  in->i[0], in->i[1], in->i[2], slice, nblk);
        break;
    case PLOW_DOP_GEMV_GLU_SZ:
        d_gemv_glu_sz((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1), (const uint8_t*)TEN(2),
                      (const uint8_t*)TEN(3), in->i[0], in->i[1], in->i[2], in->i[5], slice, nblk);
        break;
#endif

    /* ---- attention (validated, harvested op_attention.cuh) ---- */
#if PLOW_NV_LEAN_DECODE
    /* Lean GEMV segment object: the attention arms are compiled OUT (they own the register
     * ceiling). A GEMV-only segment never carries a FLASH_DECODE op; if one arrives it traps. */
    case PLOW_DOP_FLASH_DECODE:
        __trap();
        break;
#else
    case PLOW_DOP_FLASH_DECODE: {
        const unsigned gqa = in->i[1] / in->i[2];
#if PLOW_NV_GEMMA
        if (in->i[6] == 256) {
            if ((gqa % 2u) != 0) { __trap(); break; }
            d_flash_decode<256, 2>(
                (float*)TEN(0), (float*)TEN(1), (const __nv_bfloat16*)TEN(2),
                (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4), (const int*)TEN(5),
                in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->fj[0].f, in->i[5], in->i[7],
                slice, nblk, arena, in->fj[1].u);
        } else if (in->i[6] == 512) {
            if ((gqa % PLOW_NV_FA_GF_FULL) != 0) { __trap(); break; }
            d_flash_decode<512, PLOW_NV_FA_GF_FULL>(
                (float*)TEN(0), (float*)TEN(1), (const __nv_bfloat16*)TEN(2),
                (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4), (const int*)TEN(5),
                in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->fj[0].f, in->i[5], in->i[7],
                slice, nblk, arena, in->fj[1].u);
        } else {
            __trap();
        }
#else
        if ((gqa % PLOW_NV_FA_GF) != 0) { __trap(); break; }
        if (in->i[6] != PLOW_NV_FA_HD) { __trap(); break; }
        d_flash_decode<PLOW_NV_FA_HD, PLOW_NV_FA_GF>(
            (float*)TEN(0), (float*)TEN(1), (const __nv_bfloat16*)TEN(2),
            (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4), (const int*)TEN(5),
            in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->fj[0].f, in->i[5], in->i[7],
            slice, nblk, arena, in->fj[1].u);
#endif
        break;
    }
#endif /* PLOW_NV_LEAN_DECODE (FLASH_DECODE) */

#if PLOW_FP8_KV
    /* fp8-KV read: the e4m3 cache (t3=K t4=V, HALF the HBM bytes) dequanted ×per-row scale
     * (t6=k_scale t7=v_scale) in the flash inner loop; online-softmax/merge unchanged. */
    case PLOW_DOP_FLASH_DECODE_FP8: {
        const unsigned gqa = in->i[1] / in->i[2];
#if PLOW_NV_GEMMA
        if (in->i[6] == 256) {
            if ((gqa % 2u) != 0) { __trap(); break; }
            d_flash_decode<256, 2, true>(
                (float*)TEN(0), (float*)TEN(1), (const __nv_bfloat16*)TEN(2),
                (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4), (const int*)TEN(5),
                in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->fj[0].f, in->i[5], in->i[7],
                slice, nblk, arena, in->fj[1].u, (const float*)TEN(6), (const float*)TEN(7));
        } else if (in->i[6] == 512) {
            if ((gqa % PLOW_NV_FA_GF_FULL) != 0) { __trap(); break; }
            d_flash_decode<512, PLOW_NV_FA_GF_FULL, true>(
                (float*)TEN(0), (float*)TEN(1), (const __nv_bfloat16*)TEN(2),
                (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4), (const int*)TEN(5),
                in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->fj[0].f, in->i[5], in->i[7],
                slice, nblk, arena, in->fj[1].u, (const float*)TEN(6), (const float*)TEN(7));
        } else {
            __trap();
        }
#else
        if ((gqa % PLOW_NV_FA_GF) != 0) { __trap(); break; }
        if (in->i[6] != PLOW_NV_FA_HD) { __trap(); break; }
        d_flash_decode<PLOW_NV_FA_HD, PLOW_NV_FA_GF, true>(
            (float*)TEN(0), (float*)TEN(1), (const __nv_bfloat16*)TEN(2),
            (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4), (const int*)TEN(5),
            in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->fj[0].f, in->i[5], in->i[7],
            slice, nblk, arena, in->fj[1].u, (const float*)TEN(6), (const float*)TEN(7));
#endif
        break;
    }
#endif /* PLOW_FP8_KV */

#if PLOW_NV_GEMMA
    /* ---- MLA latent flash DECODE (DeepSeek/GLM/Kimi), P1. op_mla.cuh d_flash_mla_decode_sm120.
     * DK=512, DR=64 fixed (shared by all three models). GF is a COMPILE-TIME template arg baked
     * per-packet into i[7] by the emitter (glm_gf -> {2,4}); dispatch it with an explicit ladder.
     * FLASH_MLA_DECODE = dense (GATHER=false); FLASH_GATHER_DECODE = DSA sparse (GATHER=true, reads
     * the idx table t7 + top_k i6). Operand contract (op_mla.cuh:133 / emitter gemma4.rs:4196):
     *   t0=Opart(f32) t1=mlpart(f32) t2=Qabs(bf16) t3=Qrope(bf16) t4=Ckv(bf16) t5=Krope(bf16)
     *   t6=kv_len(i32) t7=idx(i32, GATHER only);  i0=n_batch i1=n_head i2=kv_stride i3=window
     *   i4=nsplit i5=kv_mask i6=top_k i7=GF ; f0=scale. Device param order: window, SCALE, nsplit,
     *   kv_mask (scale sits BETWEEN window and nsplit). MLA prefill (51) is unbuilt -> default trap. */
#if PLOW_NV_MLA
    case PLOW_DOP_FLASH_MLA_DECODE:
    case PLOW_DOP_FLASH_GATHER_DECODE: {
        const bool gather = (in->op == PLOW_DOP_FLASH_GATHER_DECODE);
        const unsigned gf = in->i[7] ? in->i[7] : 2u;
#define PLOW_MLA_DEC(GF, GATHER)                                                                \
    d_flash_mla_decode_sm120<512, 64, GF, GATHER>(                                              \
        (float*)TEN(0), (float*)TEN(1), (const __nv_bfloat16*)TEN(2),                           \
        (const __nv_bfloat16*)TEN(3), (const __nv_bfloat16*)TEN(4), (const __nv_bfloat16*)TEN(5), \
        (const int*)TEN(6), in->i[0], in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[4],      \
        in->i[5], slice, nblk, arena, (const int*)TEN(7), in->i[6])
        if (gather) {
            if (gf == 2u) PLOW_MLA_DEC(2, true);
            else if (gf == 4u) PLOW_MLA_DEC(4, true);
            else if (gf == 8u) PLOW_MLA_DEC(8, true);
            else __trap();
        } else {
            if (gf == 2u) PLOW_MLA_DEC(2, false);
            else if (gf == 4u) PLOW_MLA_DEC(4, false);
            else if (gf == 8u) PLOW_MLA_DEC(8, false);
            else __trap();
        }
#undef PLOW_MLA_DEC
        break;
    }

    /* ---- fused MLA merge + W_uv fold, P1. op_mla.cuh d_mla_merge_fold_sm120<DK=512, VT=256>.
     * Online-softmax-merges the nsplit latent partials, then folds olat @ W_uv -> o[h][V].
     *   t0=O(bf16) t1=Opart(f32) t2=mlpart(f32) t3=Wuv(bf16); i0=n_batch i1=n_head i2=V i4=nsplit. */
    case PLOW_DOP_MLA_MERGE_FOLD:
        d_mla_merge_fold_sm120<512, 256>((__nv_bfloat16*)TEN(0), (const float*)TEN(1),
                                         (const float*)TEN(2), (const __nv_bfloat16*)TEN(3),
                                         in->i[0], in->i[1], in->i[2], in->i[4], slice, nblk,
                                         arena);
        break;
#endif /* PLOW_NV_MLA */

#if PLOW_NV_DSA
    /* ---- GLM DSA lightning-indexer (P3, GLM only, ctx>65536). op_dsa.cuh. ----
     * V4 SCORE: score[t] = scale·Σ_h w[h]·ReLU(q_idx[h]·k_idx[t]) via mma.sync m16n8k16 (HI=32,
     *   DI=128 baked as template args; the emitter leaves i1/i3 unset). t0=Score(f32) t1=Qidx(bf16)
     *   t2=Kidx(bf16) t3=W(bf16) t4=kv_len(i32); i0=n_batch i2=kv_stride; f0=scale. */
    case PLOW_DOP_INDEX_SCORE:
        d_index_score_sm120<128, 32>((float*)TEN(0), (const __nv_bfloat16*)TEN(1),
                                     (const __nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(3),
                                     (const int*)TEN(4), in->i[0], in->i[2], in->fj[0].f, slice,
                                     nblk, arena);
        break;

    /* V5 SELECT: top-k radix threshold, ONE cooperative launch over the packet's `blocks` co-
     *   resident CUs (nblk here == in->blocks == 32; slice == 0..blocks-1); fenceless L2-atomic
     *   grid barrier. t0=idx(i32) t1=Score(f32) t2=gHist(u32) t3=gCtl(u32); i0=len i1=top_k. No
     *   n_sel tensor (gather reads top_k directly; ctx>65536 => len>top_k) -> nullptr. */
    case PLOW_DOP_INDEX_SELECT:
        d_index_select_sm120((int*)TEN(0), /*n_sel*/ nullptr, (const float*)TEN(1), in->i[0],
                             in->i[1], (unsigned*)TEN(2), (unsigned*)TEN(3), slice, nblk);
        break;

    /* V6 LAYERNORM+bias: indexer k_norm. t0=out t1=x t2=gamma t3=beta; i0=rows i1=feat i3=out_row0;
     *   f0=eps. op_norm.cuh d_layernorm_bias (block_sum reductions, one block per row). */
    case PLOW_DOP_LAYERNORM:
        d_layernorm_bias((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                         (const __nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(3), in->i[0],
                         in->i[1], in->i[3], in->fj[0].f, slice, nblk, arena);
        break;
#endif /* PLOW_NV_DSA */
#endif /* PLOW_NV_GEMMA */
#endif /* !PLOW_NV_PREFILL */

    case PLOW_DOP_FLASH_MERGE:
#if PLOW_NV_LEAN_DECODE
        __trap();
#elif PLOW_NV_GEMMA
        if (in->i[3] == 256)
            d_flash_merge<256>((__nv_bfloat16*)TEN(0), (const float*)TEN(1),
                               (const float*)TEN(2), in->i[0], in->i[1], in->i[2], slice, nblk);
        else if (in->i[3] == 512)
            d_flash_merge<512>((__nv_bfloat16*)TEN(0), (const float*)TEN(1),
                               (const float*)TEN(2), in->i[0], in->i[1], in->i[2], slice, nblk);
        else
            __trap();
#else
        if (in->i[3] != PLOW_NV_FA_HD) { __trap(); break; }
        d_flash_merge<PLOW_NV_FA_HD>((__nv_bfloat16*)TEN(0), (const float*)TEN(1),
                                     (const float*)TEN(2), in->i[0], in->i[1], in->i[2], slice,
                                     nblk);
#endif
        break;

#if PLOW_NV_GEMMA && !PLOW_NV_PREFILL
    /* ---- Gemma-4 26B-A4B bf16 sparse-MoE DECODE (plans/rtx-08-gemma4-moe-26b.md) ----
     * BATCH B>1 (PLOW_DECODE_BATCH): the decode MoE ops carry the batch row count in a
     * spare immediate. The compiler leaves it 0 at B=1 so the B=1 packet is byte-identical
     * to the pre-batch blob; 0 and 1 both mean "one row" here. */
#define PLOW_NROW(v) ((v) ? (v) : 1u)
    /* i[5] (router: i[3]) = BATCH B, 0/1 => single row and byte-identical to the pre-batch blob. */
    case PLOW_DOP_MOE_ROUTER_GEMMA:
        d_moe_router_gemma((unsigned char*)TEN(0), (const __nv_bfloat16*)TEN(1),
                           (const __nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(3),
                           (const __nv_bfloat16*)TEN(4), in->i[0], in->i[1], in->i[2],
                           in->fj[0].f, in->fj[1].f, slice, nblk, PLOW_NROW(in->i[3]), arena);
        break;

    case PLOW_DOP_MOE_ROUTER_GEMMA_SCORE:
        d_moe_router_gemma_score((float*)TEN(0), (const __nv_bfloat16*)TEN(1),
                                 (const __nv_bfloat16*)TEN(2),
                                 (const __nv_bfloat16*)TEN(3), in->i[0], in->i[1],
                                 in->fj[0].f, in->fj[1].f, slice, nblk, PLOW_NROW(in->i[2]));
        break;

    case PLOW_DOP_MOE_ROUTER_GEMMA_SCORE_FAST:
        d_moe_router_gemma_score_fast((float*)TEN(0), (const __nv_bfloat16*)TEN(1),
                                      (const __nv_bfloat16*)TEN(2),
                                      (const __nv_bfloat16*)TEN(3), in->i[0], in->i[1],
                                      in->fj[0].f, in->fj[1].f, slice, nblk, PLOW_NROW(in->i[2]),
                                      arena);
        break;

    case PLOW_DOP_MOE_ROUTER_GEMMA_TOPK:
        d_moe_router_gemma_topk((unsigned char*)TEN(0), (const float*)TEN(1),
                                (const __nv_bfloat16*)TEN(2), in->i[1], in->i[2],
                                slice, nblk, PLOW_NROW(in->i[3]), arena);
        break;

    case PLOW_DOP_MOE_EXPERT_GLU_GEMMA: {
        const unsigned soff = in->i[4];
        d_moe_expert_glu_gemma((__nv_bfloat16*)TEN(0) + (size_t)soff * in->i[1],
                               (const __nv_bfloat16*)TEN(1),
                               (const unsigned char*)TEN(2) + (size_t)soff * 8u,
                               (const unsigned long long*)TEN(3), in->i[0], in->i[1], in->i[2],
                               in->i[3], slice, nblk, PLOW_NROW(in->i[5]),
                               (__nv_bfloat16*)arena);
        break;
    }

    case PLOW_DOP_MOE_EXPERT_DOWN_GEMMA: {
        const unsigned soff = in->i[4];
        d_moe_expert_down_gemma((float*)TEN(0) + (size_t)soff * in->i[1],
                                (const __nv_bfloat16*)TEN(1) + (size_t)soff * in->i[2],
                                (const unsigned char*)TEN(2) + (size_t)soff * 8u,
                                (const unsigned long long*)TEN(3), in->i[0], in->i[1], in->i[2],
                                in->i[3], slice, nblk, PLOW_NROW(in->i[5]), arena);
        break;
    }

    case PLOW_DOP_MOE_EXPERT_GLU_GEMMA_FP8: {
        const unsigned soff = in->i[4];
        d_moe_expert_glu_gemma_fp8((__nv_bfloat16*)TEN(0) + (size_t)soff * in->i[1],
                                   (const __nv_bfloat16*)TEN(1),
                                   (const unsigned char*)TEN(2) + (size_t)soff * 8u,
                                   (const unsigned long long*)TEN(3),
                                   (const unsigned long long*)TEN(4), in->i[0], in->i[1],
                                   in->i[2], in->i[3], slice, nblk, PLOW_NROW(in->i[5]),
                                   (__nv_bfloat16*)arena);
        break;
    }

    case PLOW_DOP_MOE_EXPERT_DOWN_GEMMA_FP8: {
        const unsigned soff = in->i[4];
        d_moe_expert_down_gemma_fp8((float*)TEN(0) + (size_t)soff * in->i[1],
                                    (const __nv_bfloat16*)TEN(1) + (size_t)soff * in->i[2],
                                    (const unsigned char*)TEN(2) + (size_t)soff * 8u,
                                    (const unsigned long long*)TEN(3),
                                    (const unsigned long long*)TEN(4), in->i[0], in->i[1],
                                    in->i[2], in->i[3], slice, nblk, PLOW_NROW(in->i[5]));
        break;
    }

    case PLOW_DOP_MOE_COMBINE_GEMMA:
        d_moe_combine_gemma((__nv_bfloat16*)TEN(0), (const float*)TEN(1), in->i[0], in->i[1],
                            slice, nblk);
        break;

    case PLOW_DOP_MOE_COMBINE_NORM_GEMMA:
        d_moe_combine_norm_gemma((__nv_bfloat16*)TEN(0), (const float*)TEN(1),
                                  (const __nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(3),
                                  in->i[0], in->i[1], in->fj[0].f, slice, nblk,
                                  PLOW_NROW(in->i[2]), arena);
        break;

    /* Fused MoE layer tail: combine + post_ffn norm + sandwich residual + next input norm. */
    case PLOW_DOP_MOE_COMBINE_RESID_NORM_GEMMA:
        d_moe_combine_resid_norm_gemma((__nv_bfloat16*)TEN(0), (__nv_bfloat16*)TEN(1),
                                       (const float*)TEN(2), (const __nv_bfloat16*)TEN(3),
                                       (const __nv_bfloat16*)TEN(4), (const __nv_bfloat16*)TEN(5),
                                       (const __nv_bfloat16*)TEN(6), in->i[0], in->i[1],
                                       in->fj[0].f, in->fj[1].f, slice, arena);
        break;

    case PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA:
        d_moe_expert_glu_norm_gemma((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                                    (const __nv_bfloat16*)TEN(4),
                                    (const unsigned char*)TEN(2),
                                    (const unsigned long long*)TEN(3), in->i[0], in->i[1],
                                    in->i[2], in->i[3], in->fj[0].f, slice, nblk,
                                    PLOW_NROW(in->i[5]), arena);
        break;
#undef PLOW_NROW
#endif

    /* ---- greedy sampling ---- */
    case PLOW_DOP_ARGMAX:
        /* i1 = n_batch (0/1 => single sequence, byte-identical). */
        d_argmax((unsigned long long*)TEN(0), (const __nv_bfloat16*)TEN(1), in->i[0], in->i[1],
                 slice, nblk, (unsigned long long*)arena);
        break;

    case PLOW_DOP_ARGMAX_FIN:
        d_argmax_fin((int*)TEN(0), (const unsigned long long*)TEN(1), in->i[0], in->i[1], slice);
        break;

    /* Nemotron-3 Mamba-2 SSD mixer core (M4). UNVERIFIED on GPU — see op_mamba.cuh. Single-CU
     * correctness-first: reads/writes the carried conv_state (t6) + ssm_state (t7). Gated behind
     * PLOW_NV_MAMBA (default ON): a Gemma object never emits it, and it owns the prefill stack
     * frame (1744 -> 688 B when gated out). */
#if PLOW_NV_MAMBA
    case PLOW_DOP_MAMBA2_SCAN:
        d_mamba2_scan((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                      (const __nv_bfloat16*)TEN(2), (const __nv_bfloat16*)TEN(3),
                      (const __nv_bfloat16*)TEN(4), (const float*)TEN(5), (float*)TEN(6),
                      (float*)TEN(7), in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->i[5],
                      in->i[6], in->i[7], in->fj[0].f, slice);
        break;
#endif /* PLOW_NV_MAMBA */

    case PLOW_DOP_NOP:
        break;

    /* Any opcode this build does not implement — every prefill op, every fp8 op, every TP
     * op — lands here. It TRAPS. The alternative (break) would let a prefill program run to
     * completion producing zeros, which reads as a working interpreter emitting garbage
     * logits. A trap is a loud, locatable failure. */
    default:
        __trap();
        break;
    }
}
#undef TEN

/* ---- the persistent interpreter --------------------------------------------------------
 * One cooperative launch, grid == co-resident capacity. Structure and the ENTIRE counter
 * protocol are interp_sm120_poc.cu's, unchanged.
 *
 * THE ACQUIRE __threadfence() AT THE GATE IS LOAD-BEARING and must not be removed. The PoC's
 * own 2-op shape cannot detect its absence — its 4 MiB working set is evicted from L2 before
 * the consumer re-reads it, so the PoC passes with the fence deleted, by hardware accident.
 * A real program has pure-consumer blocks in a fan-out reading small, still-cached operands,
 * which is exactly the case the fence exists for. */
/* ---- SCHEDULER SELECT (M5 A/B) ----------------------------------------------------------
 * PLOW_NV_SCHED 0 = STATIC   : each block walks its own stream (the 6.8636 ms baseline).
 *               1 = GQ       : every block pulls the next entry from ONE atomic cursor over an
 *                              op-major (topological) permutation of the same stream. AMD
 *                              Experiment E1 (interp.hip:877-924) ported verbatim in structure.
 * PLOW_NV_SKELETON 1         : run the gate/signal skeleton with NO op bodies. Produces GARBAGE
 *                              logits by construction — a measurement-only build that isolates
 *                              the interpreter's own scheduling cost from the math. Never ship.
 *
 * The counter protocol below is byte-identical across all three: same relaxed poll, same
 * post-gate acquire __threadfence(), same release fence before the successor bumps. Only WHICH
 * block runs WHICH stream entry changes, so an A/B isolates scheduling. */
#ifndef PLOW_NV_SCHED
#define PLOW_NV_SCHED 1
#endif
#ifndef PLOW_NV_SKELETON
#define PLOW_NV_SKELETON 0
#endif
/* Backoff inside the counter-gate poll. 64 ns is the shipped value; 0 spins flat out. */
#ifndef PLOW_NV_GATE_SLEEP
#define PLOW_NV_GATE_SLEEP 64
#endif

__global__ __launch_bounds__(256, PLOW_NV_MINBLK) void PLOW_SYM(interp_sm120)(PlowProgram prog) {
    extern __shared__ float arena[];
#if PLOW_NV_SKELETON
#ifndef PLOW_NV_SKEL_PAD
#define PLOW_NV_SKEL_PAD 160
#endif
    /* MEASUREMENT ONLY. A body-less kernel is tiny, so the host's
     * cuOccupancyMaxActiveBlocksPerMultiprocessor reports 8 blocks/SM and the launch mismatches
     * the packet's n_cu=132. Pad static smem so occupancy is the real object's 1 block/SM and
     * the skeleton's gate/signal cost is measured at the shipped grid. */
    __shared__ char skel_occ_pad[PLOW_NV_SKEL_PAD * 1024];
    if (threadIdx.x > 10000u) ((volatile char*)skel_occ_pad)[0] = 1;
#endif
    const unsigned nblk_grid = gridDim.x;
#if PLOW_NV_SCHED == 1
    /* ONE shared cursor over the op-major gq_stream. gq_seg_ofs is a 2-word {0, n_stream}
     * window (this program is a SINGLE segment — measured: every entry seg==0). */
    __shared__ unsigned gq_claim;
#if PLOW_NV_PLACE_DISPATCH
    /* ===== EXPERIMENTAL / UNVALIDATED — physical-SM L2-domain dispatch =====
     * plans/devblob-locality-placement.md. Consumes a compiler PLOW_NV_PLACE blob,
     * whose gq_stream is grouped into P per-L2-domain windows (gq_seg_ofs). Each
     * block reads its PHYSICAL SM id and pulls ONLY its L2 domain's window via that
     * domain's cursor line, so a domain's packets run on the L2 slice that holds
     * their data. This is what makes the compiler's locality real (vs blockIdx,
     * which the HW scheduler maps arbitrarily to SMs).
     *
     * REQUIRES: -DPLOW_NV_L2_SMS=<SMs per L2 partition> (18 on H100, 32 on MI350).
     * CAVEAT: smid/PLOW_NV_L2_SMS assumes contiguous per-GPC smid numbering — NOT
     * guaranteed by CUDA. The robust form is a thread-block CLUSTER launch (sm_90+)
     * with cluster-rank as the domain. MUST be built and measured on a partitioned
     * GPU (H100/B200/MI300/MI350); default off => byte-identical SASS, never built. */
    unsigned plow_smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(plow_smid));
    const unsigned plow_dom = plow_smid / (PLOW_NV_L2_SMS);
    uint32_t* const cursor = PLOW_CTR(prog.gq_cursor, plow_dom);
    const unsigned gq_lo = prog.gq_seg_ofs[plow_dom];
    const unsigned gq_hi = prog.gq_seg_ofs[plow_dom + 1];
#elif PLOW_NV_SEGMENTS
    /* Segmented dispatch (AMD interp.hip:881-886): this launch owns ONE segment. Its cursor is a
     * per-segment line (own cache line via PLOW_CTR stride) so the host can enqueue every segment's
     * launch after a single zeroing; its window is the segment's contiguous [lo,hi) slice of the
     * op-major gq_stream. cur_seg==0 over a {0,n_stream} window is exactly the single-segment case. */
    uint32_t* const cursor = PLOW_CTR(prog.gq_cursor, prog.cur_seg);
    const unsigned gq_lo = prog.gq_seg_ofs[prog.cur_seg];
    const unsigned gq_hi = prog.gq_seg_ofs[prog.cur_seg + 1];
#else
    uint32_t* const cursor = PLOW_CTR(prog.gq_cursor, 0);
    const unsigned gq_lo = prog.gq_seg_ofs[0];
    const unsigned gq_hi = prog.gq_seg_ofs[1];
#endif
    const PlowStreamEnt* my = prog.gq_stream;
    for (;;) {
        /* REQUIRED barrier (AMD interp.hip:901-907): the whole block must retire the previous
         * packet — INCLUDING thread 0's successor signal — before any thread claims the next.
         * Without it the next claim races the tail of the previous packet through the shared
         * gq_claim broadcast and the grid deadlocks. */
        __syncthreads();
        if (threadIdx.x == 0) gq_claim = atomicAdd(cursor, 1u);
        __syncthreads();
        const unsigned ix = gq_lo + gq_claim;
        if (ix >= gq_hi) break;
        const PlowStreamEnt e = ld_stream_ent(my + ix);
        const PlowDevInst* in = prog.insts + e.inst;
#else
    const unsigned cu = blockIdx.x;
    const unsigned n = prog.stream_len[cu];
    const PlowStreamEnt* my = prog.stream + prog.stream_ofs[cu];

    for (unsigned ix = 0; ix < n; ix++) {
        const PlowStreamEnt e = ld_stream_ent(my + ix);
#if PLOW_NV_SEGMENTS
        /* Segmented dispatch (AMD interp.hip:929-932): run only this segment's entries. A skipped
         * entry costs a branch and nothing else — the launch that owns the segment executes it. */
        if (e.seg != prog.cur_seg) continue;
#endif
        const PlowDevInst* in = prog.insts + e.inst;
#endif

        /* GATES live on the STREAM ENTRY, always — the 64-byte PlowDevInst carries no
         * wait/succ metadata. Coarse entries point at the op's coarse lists (all slices
         * share them); a PLOW_SE_FINE entry carries per-slice lists so slice s blocks only
         * on the producer slices that feed it (dev_isa.h "PER-SLICE GATES"). SE_XCTR
         * (cross-GPU, system-scope counter region) is still unimplemented on this
         * single-GPU decode build and traps. */
        if (e.flags & PLOW_SE_XCTR) {
            if (threadIdx.x == 0) __trap();
        }
        const unsigned wait_len = e.wait_len;
        const unsigned wait_ofs = e.wait_ofs;
        const unsigned succ_len = e.succ_len;
        const unsigned succ_ofs = e.succ_ofs;

#if PLOW_NV_TRACE
        const bool tr = (blockIdx.x == 0 && threadIdx.x == 0 && g_tr_n < PLOW_TRACE_MAX);
        long long t_gate0 = 0, t_gate1 = 0, t_body1 = 0;
        if (tr) t_gate0 = clock64();
#endif
        /* Gate: one thread per counter, polled concurrently. */
        for (unsigned w = threadIdx.x; w < wait_len; w += blockDim.x) {
            const PlowWait pw = prog.waits[wait_ofs + w];
            while (ctr_poll(PLOW_CTR(prog.counters, pw.id)) < pw.threshold) {
#if PLOW_NV_GATE_SLEEP > 0
                __nanosleep(PLOW_NV_GATE_SLEEP);
#endif
            }
        }
#if PLOW_NV_PTXSYNC == 3
        /* The relaxed spin carries no ordering; ONE loop-exit acquire supplies it (V3). */
        if (wait_len) asm volatile("fence.acquire.gpu;" ::: "memory");
#endif
        __syncthreads(); /* every counter in the list is now satisfied */
#if PLOW_NV_TRACE
        if (tr) t_gate1 = clock64();
#endif

#if PLOW_NV_PTXSYNC == 1 || PLOW_NV_PTXSYNC == 3
        /* The acquire already rode in on ctr_poll's `ld.acquire.gpu` (V1) or on the loop-exit
         * fence above (V3), in the very threads that observed the counter reach threshold, and
         * the __syncthreads() above published it to the block. The separate fence AND the
         * barrier that scoped it to thread 0 are both redundant here — this is the structural
         * saving, not just a cheaper fence. */
#else
        /* ONE acquire for the whole block, only after the gate clears. */
        if (threadIdx.x == 0 && wait_len) __threadfence();
        __syncthreads();
#endif

        /* The op body slices by the PACKET's (slice, blocks), NOT by blockIdx/gridDim: inside
         * a megakernel the block IS the "CU" and its share is carried in the stream entry,
         * because the grid's blocks are not at one PC. `blocks` can be < gridDim (ARGMAX is
         * 64 blocks, FLASH_MERGE 32, the norms 1) and a block not in the packet's set simply
         * has no stream entry for it. */
#if PLOW_NV_SKELETON
        /* MEASUREMENT BUILD: no op body. The gate, the fences, the successor bumps and the
         * stream walk are all still here, so this times the interpreter's scheduling floor
         * on the REAL 401-packet / 32493-entry decode program. Output is garbage. */
        (void)nblk_grid;
#else
        plow_exec(in, prog.tensors, e.slice, in->blocks ? in->blocks : nblk_grid, arena);
#endif

        __syncthreads(); /* retire this block's stores before the release */
#if PLOW_NV_TRACE
        if (tr) t_body1 = clock64();
#endif

#if PLOW_NV_PTXSYNC != 1 && PLOW_NV_PTXSYNC != 3
        if (succ_len) {
            /* Publish thread 0's legacy release fence to the designated counter writers
             * before any of them performs its relaxed bump. */
            if (threadIdx.x == 0)
            __threadfence(); /* release: order stores ahead of the counter bump */
            __syncthreads();
        }
#endif
        /* One designated thread per successor counter, matching the concurrent wait side.
         * Fine/tile programs can carry many independent successors; walking them on thread 0
         * serializes their global-memory latency. PTXSYNC makes each bump a release. */
        for (unsigned s = threadIdx.x; s < succ_len; s += blockDim.x)
            ctr_signal(PLOW_CTR(prog.counters, prog.succs[succ_ofs + s]));
#if PLOW_NV_TRACE
        /* Include every designated counter writer in the signal duration. */
        __syncthreads();
        if (tr) {
            const unsigned k = g_tr_n;
            g_tr_op[k]   = in->op;
            g_tr_wait[k] = wait_len;
            g_tr_gate[k] = (unsigned long long)(t_gate1 - t_gate0);
            g_tr_body[k] = (unsigned long long)(t_body1 - t_gate1);
            g_tr_sig[k]  = (unsigned long long)(clock64() - t_body1);
            g_tr_n       = k + 1;
        }
#endif
    }
}

/* ---- host-side launch helper -----------------------------------------------------------
 * Grid comes from cudaOccupancyMaxActiveBlocksPerMultiprocessor x multiProcessorCount, cached
 * once. Co-residency is the CORRECTNESS condition, not a tuning knob: a block that is not
 * resident cannot bump the counter another resident block is spinning on, and the whole grid
 * deadlocks. Cooperative launch refuses an oversized grid at launch, which turns AMD's silent
 * hang into a launch-time error. */
extern "C" int PLOW_SYM(plow_sm120_grid)(int dev) {
    /* Per-device cache, not a single (cached, cached_dev) pair: the latter races under
     * concurrent calls for DIFFERENT devices (thread A publishes cached_dev=A then B
     * publishes cached_dev=B before A publishes its value, so A's caller reads B's grid).
     * A per-slot int keyed by dev removes the shared write set entirely — 0 means
     * "not computed", and every computed grid is bps*SMs > 0, so recompute is benign and
     * idempotent. Aligned int load/store is atomic on the host platforms, so no lock. */
    static int cached[16] = {0};
    if (dev >= 0 && dev < 16 && cached[dev]) return cached[dev];
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 0;
    /* Opt in to >48 KiB dynamic smem (the prefill flash tile is ~67.6 KiB). Idempotent; must be
     * set before the occupancy query and the launch will accept the arena. */
    if (PLOW_NV_ARENA_FLOATS * sizeof(float) > 48u * 1024u)
        cudaFuncSetAttribute((const void*)PLOW_SYM(interp_sm120),
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)(PLOW_NV_ARENA_FLOATS * sizeof(float)));
    int bps = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &bps, (const void*)PLOW_SYM(interp_sm120), 256,
            PLOW_NV_ARENA_FLOATS * sizeof(float)) != cudaSuccess)
        return 0;
#if PLOW_NV_SKELETON
    /* The skeleton has no op bodies and therefore a much smaller register footprint, which can
     * raise cudaOccupancyMaxActiveBlocks above the real kernel's. Pin it to 1 block/SM so the
     * skeleton runs the SAME grid (170) and the SAME stream partition as the build it is being
     * compared against — otherwise the comparison is against a different schedule entirely. */
    if (bps > 1) bps = 1;
#endif
    int grid = bps * p.multiProcessorCount;
    if (dev >= 0 && dev < 16) cached[dev] = grid;
    return grid;
}

extern "C" size_t PLOW_SYM(plow_sm120_smem)(void) { return PLOW_NV_ARENA_FLOATS * sizeof(float); }

/* Which scheduler this object was built with — printed by the harness so a log can never be
 * mis-attributed, and used to decide whether the gq_* kernarg fields must be populated. */
extern "C" int PLOW_SYM(plow_sm120_sched)(void) { return PLOW_NV_SCHED; }
extern "C" int PLOW_SYM(plow_sm120_skeleton)(void) { return PLOW_NV_SKELETON; }

extern "C" int PLOW_SYM(plow_sm120_launch)(PlowProgram* prog, int grid, cudaStream_t stream) {
    void* args[] = {prog};
#if PLOW_NV_TRACE
    /* Fire the block-0 dump on the launch index PLOW_NV_TRACE_SKIP (default 0 = first launch).
     * With a real prompt this lets the traced step land at a WARM context (e.g. skip past the
     * priming launches so flash-decode carries a realistic KV length), instead of priming-step-0.
     * g_tr_n is zeroed right before the target launch so the buffer holds exactly that one step's
     * block-0 packets rather than an accumulation across every prior launch. */
    static int launch_ix = -1;
    static long trace_skip = -2;
    if (trace_skip == -2) {
        const char* s = getenv("PLOW_NV_TRACE_SKIP");
        trace_skip = s ? atol(s) : 0;
    }
    launch_ix++;
    const bool trace_this = (launch_ix == (int)trace_skip);
    if (trace_this) {
        unsigned zero = 0;
        cudaMemcpyToSymbol(g_tr_n, &zero, sizeof(zero));
    }
#endif
    int rc = (int)cudaLaunchCooperativeKernel((void*)PLOW_SYM(interp_sm120), dim3(grid), dim3(256),
                                              args, PLOW_NV_ARENA_FLOATS * sizeof(float), stream);
#if PLOW_NV_TRACE
    static int dumped = 0;
    if (trace_this && !dumped) {
        cudaStreamSynchronize(stream);
        unsigned n = 0;
        cudaMemcpyFromSymbol(&n, g_tr_n, sizeof(n));
        if (n > PLOW_TRACE_MAX) n = PLOW_TRACE_MAX;
        if (n) {
            static unsigned      op[PLOW_TRACE_MAX], wl[PLOW_TRACE_MAX];
            static unsigned long long ga[PLOW_TRACE_MAX], bo[PLOW_TRACE_MAX], si[PLOW_TRACE_MAX];
            cudaMemcpyFromSymbol(op, g_tr_op,   n * sizeof(unsigned));
            cudaMemcpyFromSymbol(wl, g_tr_wait, n * sizeof(unsigned));
            cudaMemcpyFromSymbol(ga, g_tr_gate, n * sizeof(unsigned long long));
            cudaMemcpyFromSymbol(bo, g_tr_body, n * sizeof(unsigned long long));
            cudaMemcpyFromSymbol(si, g_tr_sig,  n * sizeof(unsigned long long));
            printf("PLOW_TRACE_N %u\n", n);
            for (unsigned i = 0; i < n; i++)
                printf("PLOW_TRACE %u op=%u wait=%u gate=%llu body=%llu sig=%llu\n",
                       i, op[i], wl[i], ga[i], bo[i], si[i]);
            fflush(stdout);
        }
        dumped = 1;
    }
#endif
    return rc;
}
