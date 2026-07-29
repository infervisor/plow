/* interp.hip — the persistent on-device interpreter (CDNA4 / gfx950).
 *
 * THIS IS THE ONLY KERNEL ON THE SERVING PATH. It is launched once per device,
 * with grid == CU count, and stays resident for the life of the model. Each
 * workgroup owns one CU and walks its own packet stream. Nothing is launched
 * per op.
 *
 * The file contains no math: it is a gate, a switch, a fence, and a signal. All
 * arithmetic lives in the op_*.h headers, which are shared verbatim with the
 * golden test wrappers (test_kernels.hip) — so the CPU-reference tests validate
 * exactly the code that runs here, not a second copy of it.
 *
 * See runtime/common/dev_isa.h for the execution model and the co-residency
 * safety condition (grid > CU count deadlocks).
 */
#include "dev_isa.h"

/* PLOW_K3 — compile the Kimi-K3 / KDA arms (op_k3.h, op_kda.h). DEFAULT 0.
 *
 * SEVEN arms hang off this: the four KDA ops (CONV, GATE, STATE_STEP, GATED_NORM) and the three
 * K3 block ops (ATTN_RES, SITU_GLU, MLA_OUT_GATE). They are an ENCODING/ARCHITECTURE AXIS exactly
 * like PLOW_FP8, PLOW_MXFP4, PLOW_MLA_PREFILL and PLOW_MOE_PREFILL, and they get an axis for the
 * same reason those do: this is a PERSISTENT MEGAKERNEL, so every arm compiled in shares one
 * instruction stream with every packet body, whether or not any packet dispatches it.
 *
 * THE MEASURED PRECEDENT, and why "the registers fit" was not enough to justify landing these
 * unconditionally. op_attention.h:1282 records the GF=8 flash arm: validated on REGISTERS ALONE
 * (+6 VGPR, occupancy 2 held, zero spill) and duly concluded "GF=8 fits". It does fit. It was
 * also a +32% DECODE REGRESSION on a byte-identical model.pkt — 313,024 -> 361,896 byte object,
 * 27.98 -> 37.01 ms — AND NOT BECAUSE THE ARM RAN. A packet with GF=4 pinned still measured
 * 36.12. It was the arm being PRESENT. That file states the general lesson outright: "for this
 * interpreter, 'the arm fits in the register budget' is necessary and NOT sufficient; the object
 * must be weighed too."
 *
 * These seven arms landed with no object-size or ITL measurement at all, in every object Gemma-4,
 * Qwen and GLM-5.2 load — none of which dispatch a single one of them. Gating restores the
 * property the other five axes already have: an object carries an arm because a model needs it.
 * `perf-data/kimi-k3-kernel-gap.md:791` independently reached the same conclusion from the
 * register side ("Build it as a 4th co-resident object, not an arm").
 *
 * TURNING THIS OFF IS NOT SILENT — see `plow_k3_arms_1` below. AMD's dispatch `default:` writes
 * NOTHING (it does not trap, unlike sm_120), so a K3 packet run against an object built without
 * this axis would compute nothing and surface as an accuracy bug. The marker symbol makes the
 * loader refuse that pairing before the object reaches a device. */
#ifndef PLOW_K3
#define PLOW_K3 0
#endif

#include "amd_common.h"
#include "op_attention.h"
#include "op_collective.h"
#include "op_elementwise.h"
#include "op_gemm.h"
#if PLOW_K3
#include "op_k3.h"
#include "op_kda.h"
#endif
#include "op_moe.h"
#include "op_norm.h"

/* The K3/KDA arms, made observable in the ELF the same way `PLOW_GEMV_WALK` is
 * (`runtime/amd/op_gemm.h:2557`) and for the same reason: the loader reads `.symtab` before the
 * object is on a device, so a name costs nothing where a value costs a device round-trip.
 *
 * DERIVED, never restated — the symbol exists iff the axis is on, in one preprocessor pass. An
 * object built with PLOW_K3=0 is byte-identical to one built before this axis existed, and every
 * object built before it carries no such symbol, so absence is unambiguous rather than "old".
 * `crates/plowrt/src/exec/amd.rs` refuses a packet carrying a K3/KDA opcode against an object
 * without it. */
#if PLOW_K3
extern "C" __device__ unsigned plow_k3_arms_1 = 1;
#endif

/* Cross-GPU collective spin deadline (s_memrealtime ticks). s_memrealtime is 1 GHz on
 * gfx950 (plans/tp-transport.md §1), so ~1e9 ticks ≈ 1 s: a fabric on which the
 * system-scope xctr atomic did NOT propagate bails with status 0xDEAD.... instead of
 * hanging the resident queue. Never reached on a healthy XGMI mesh. */
#define PLOW_XCTR_DEADLINE_TICKS 1000000000ull

/* One LDS arena, reused by whichever op the workgroup is currently running. Ops are
 * sequential within a workgroup, so a union is safe -- and it means the interpreter's
 * LDS footprint is set by its hungriest op, not the sum of them.
 *
 *   GEMM, double-buffered 256x256, BK=64 : 144 KiB  <- the max
 *   flash prefill, head_dim 512          :  57 KiB
 *   flash decode                         :   2 KiB
 *   block_sum scratch                    :  32 B
 * of the 160 KiB a gfx950 workgroup may have. */
/* ---------------------------------------------------------------------------
 * BUCKETS. plowc emits ONE INTERPRETER PER BUCKET, and the op set is part of the
 * bucket -- not because it is tidy, but because register allocation forces it.
 *
 * Every op is inlined into this one kernel, so the kernel's register count is the
 * worst case over all of them, and arch VGPRs and AccVGPRs are separate files
 * whose maxima SUM. A wave may use at most 512/waves-per-SIMD registers, so an
 * 8-wave dispatch (2 waves/SIMD) needs the whole kernel under 256. Measured:
 *
 *   op set                                     arch  agpr  total  waves/SIMD
 *   everything in one kernel                    148   128    276      1   <- rejected
 *   prefill: tier-A + GEMM + flash_prefill      128   128    256      2
 *   decode:  tier-A + GEMV + flash_dec/merge    120    32    152      2
 *
 * At 1 wave/SIMD the GEMM loses its ping-pong partner, which measures at ~7% through
 * the interpreter -- worth having, and the split costs nothing, but it is not the 2x an
 * earlier note claimed (that was the LDS alignment bug, not the wave count).
 *
 * GEMM_NORM is in NEITHER bucket, deliberately. It fuses an RMSNorm into the
 * A-operand load, and it is a second full instantiation of d_gemm_t -- on its own
 * that pushes the prefill kernel from 192 to 276 registers and costs it the 8-wave
 * dispatch. It is also a measured LOSS at every shape we actually run (62-113 TF/s
 * fused vs 194+ unfused): the A tile is re-fetched once per N-tile, so the fused
 * per-element work is multiplied by N/BN (168x at N=21504). It saves one 88 MB HBM
 * round-trip and pays 168x redundant bf16<->f32 conversion for it. The op is still
 * in op_gemm.h (correct, tested, and it does win at small N) -- it is just not
 * worth a bucket. */
#ifndef PLOW_BUCKET_DECODE
#define PLOW_BUCKET_DECODE 0
#endif
#define PLOW_BUCKET_PREFILL (!PLOW_BUCKET_DECODE)

/* ---------------------------------------------------------------------------
 * THE FOUR PRECISION AXES, AND WHICH FLAG IS WHICH.
 *
 * A packet's precision is not one choice, it is four independent ones, and the flag names on
 * this side collapse them. Written out so the next reader does not have to rediscover it:
 *
 *   axis            flag              what it adds                         prefill / decode
 *   weight (fp8)    PLOW_FP8          GEMM*_FP8, QUANT_FP8 (w8a8)          w8a8    / w8a16
 *                                     GEMV_FP8, GEMV_GLU_FP8 (w8a16)
 *   weight (fp4)    PLOW_MXFP4        GEMM_MXFP4 / GEMV*_MXFP4 (w4a16)     w4a16   / w4a16
 *   KV              PLOW_FP8_KV       *_FP8 flash + HEADNORM_ROPE_FP8      swap    / swap
 *   experts         PLOW_MOE_PF_A4W4  A4W4 grouped GEMM (ops 85/86)        A4W4    / —
 *                   + the runtime ENC field (i[3] prefill / i[6] decode)
 *
 * THREE THINGS THAT NAME IS WRONG ABOUT, all measured, none of them fixable here:
 *   1. `PLOW_FP8` is a WEIGHT-encoding axis, but its prefill arms are w8a8 (both operands e4m3,
 *      epilogue acc*a_scale[m]*w_scale[n]) and its decode arms are w8a16 (bf16 activation,
 *      dequant on load). One object legitimately carries both — see the arms below.
 *   2. Emit-time `PLOW_FP8=1` alone means w8a16 END TO END, which the AMD GEMM_FP8 arm cannot
 *      run (it indexes a_scale unconditionally and reads A as bytes). devgen refuses that
 *      combination on gfx950 and names the runnable one PLOW_W8A8=1. Correct, but it is a
 *      refusal standing in for a name: NVIDIA makes w8a16-vs-w8a8 a BUILD choice
 *      (-DPLOW_NV_W8A8) and AMD makes it a hard property, with the vendor prefix on only one.
 *   3. `PLOW_FP8` and `PLOW_FP8_KV` are different axes and the shared `FP8` says otherwise. A
 *      PLOW_FP8=1 packet keeps a BF16 KV and emits FlashPrefill; only PLOW_FP8_KV=1 emits
 *      FlashPrefillFp8. Select on the OPCODE, never on a precision flag — the class-4 segment
 *      classifier tested only the bf16 opcode and found 0 of 60 flash segments on an fp8-KV
 *      packet.
 *
 * WEIGHT AXES ARE ADDITIVE, THE KV AXIS IS A SWAP. That asymmetry is deliberate and measured:
 * a weight-quantized program provably MIXES precisions inside one bucket (the lm_head is a bf16
 * GEMM_SMALL in every fp8 prefill packet), while a packet's KV is uniformly one encoding, and
 * the 4-wave flash object is the one place the register budget is actually tight — adding the
 * bf16 flash arm to interp_flash_fp8kv moved its spill 228 -> 0, i.e. the allocator dropped the
 * intentional Q-hoist. Additive on the weight axes costs nothing (see the arms). */
#ifndef PLOW_FP8
#define PLOW_FP8 0
#endif

/* GLOBAL-QUEUE scheduling (Experiment E1). When 1, every workgroup pulls the next packet from ONE
 * shared atomic cursor over the op-major gq_stream instead of walking a static per-CU stream; the op
 * kernels/tiles/register configs are byte-identical, so this is a pure scheduling change. Built as
 * separate _gq objects and selected at launch by the host (PLOW_GLOBAL_QUEUE=1). Measured (batch 1):
 * consistent decode win on both models (dynamic claim smooths static per-CU imbalance), prefill cost
 * model-size-dependent (small on 4B). See plans/global-queue-e1-result.md.
 *
 * CORRECTION (measured 2026-07-27, MI355X, Gemma-4 31B bf16, real weights through gemma4_chat,
 * T=1024 single chunk): prefill GQ is NOT neutral on 31B, it is a WIN — 163 ms GQ vs 178 ms
 * static, i.e. -8.4%. The earlier 'neutral' reading predates the 31B end-to-end run.
 *
 * PLOW_GQ_BATCH must stay 1: claiming >1 packet per fetch-add starves parallelism (a K-wide batch is
 * drained by only n_cu/K workgroups), measured 6-8x slower at K=8. The knob exists only to document
 * that dead end; do not raise it. */
/* MoE PREFILL arms (ops 83-87). Own flag, like PLOW_MLA_PREFILL: only a Kimi/GLM/DeepSeek
 * prefill object needs the grouped-expert GEMM, and it is a second full MFMA body in a bucket
 * already at the 256/occ-2 cliff. Off => every existing object is byte-identical. */
#ifndef PLOW_MOE_PREFILL
#define PLOW_MOE_PREFILL 0
#endif
#ifndef PLOW_GLOBAL_QUEUE
#define PLOW_GLOBAL_QUEUE 0
#endif
#ifndef PLOW_GQ_BATCH
#define PLOW_GQ_BATCH 1
#endif

#define PLOW_GM_ARENA GM_LDS_HALVES_T(GM_BM, GM_BN, GM_BK)
#define PLOW_SMEM_HALVES                                                        \
    (PLOW_GM_ARENA > FA_LDS_HALVES(512) ? PLOW_GM_ARENA : FA_LDS_HALVES(512))

/* __align__(16) is LOAD-BEARING, and its absence cost the GEMM half its speed.
 *
 * The union's natural alignment is 4 (it contains a float), so without this the
 * compiler cannot prove `sm->gm` is 16-byte aligned -- and ds_read_b128 REQUIRES
 * 16-byte alignment. It silently falls back to pairs of ds_read2_b32, which doubles
 * the LDS instruction count and halves effective LDS bandwidth in the GEMM's hottest
 * loop. Measured on gate/up_proj: 24x ds_read_b128 + 16x ds_write_b128 (687 TF/s)
 * became 48x ds_read2_b32 + 32x ds_write2_b32 (371 TF/s).
 *
 * The same GEMM in a standalone kernel was fast, which is what made this look for a
 * long time like a register-allocation problem with the megakernel. It was not: the
 * standalone kernel passes a plain __shared__ bf16[] (16-byte aligned), the
 * interpreter passed a union member (4-byte aligned). One attribute. */
typedef union __align__(16) {
    float part[PLOW_WAVES];           /* block_sum scratch          */
    bf16  gm[PLOW_GM_ARENA];          /* GEMM A|B tiles, x2 buffers */
    bf16  fa[FA_LDS_HALVES(512)];     /* flash K|V|P tiles          */
    float fd[FA_DEC_LDS_FLOATS(512, PLOW_FA_GF(512))]; /* flash decode; the op owns this size */
    float fmla[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF_MAX)]; /* MLA latent decode (DK=512,DR=64) */
    unsigned long long am[PLOW_WAVES];  /* argmax block reduction   */
    bf16  raw[PLOW_SMEM_HALVES];
} plow_smem;

/* ---------------------------------------------------------------------------
 * THE PACKET LIVES IN SCALAR REGISTERS.
 *
 * A packet's fields and its tensor pointers are UNIFORM (every lane wants the same value)
 * and INVARIANT (nothing on device writes them). The compiler can prove the first and not
 * the second -- it cannot rule out that one of our tensor stores aliases the instruction
 * table -- so it refuses the scalar unit and emits, for EVERY field of EVERY packet, a
 * VECTOR load plus a v_readfirstlane to put the value back in an SGPR. The tensor operands
 * are a two-level chase (load the handle, then index the pointer table) and it serialises
 * them:
 *
 *     global_load_dword    v42, ... offset:32     <- in->t[4]
 *     s_waitcnt vmcnt(0)                          <- stall
 *     v_cmp_eq_u32  vcc, -1, v42                  <- TENSOR_NONE?
 *     global_load_dwordx2  v[2:3], v[2:3], off    <- tensors[t[4]]
 *     s_waitcnt vmcnt(0)                          <- stall
 *     global_load_dword    v42, ... offset:36     <- in->t[5], and so on
 *
 * One dependent round trip per operand per packet -- and `vmcnt(0)` is not specific to
 * these loads, it drains every in-flight WEIGHT load with it. The preamble was a
 * scheduling barrier planted in the middle of the memory stream it competes with.
 *
 * The fix is to say the thing the compiler cannot infer: this memory is constant. On
 * AMDGPU address space 4 IS the constant space, and a pointer into it is invariant by
 * construction -- so clang emits s_load, schedules the loads freely, and tracks them on
 * lgkmcnt, which does not interact with the vector memory stream at all. The host patches
 * `insts` BETWEEN launches, never during one, so the promise is true.
 *
 * NOTE that plow_exec must stay a real call (`static`, NOT `__forceinline__`). Force-inlining
 * it merges every op's live ranges into the caller, and PREFILL sits exactly at the register
 * cliff (128 arch + 128 agpr = the 256 a wave may use at 2 waves/SIMD). Inlined, the
 * allocator re-splits to 256 arch + 32 agpr, which does not fit, and the 8-wave dispatch is
 * then rejected outright with HSA_STATUS_ERROR_INVALID_ISA. */
#define PLOW_KONST __attribute__((address_space(4)))
typedef const PLOW_KONST PlowDevInst* CInst; /* the instruction stream   */
typedef void* const PLOW_KONST* CTens;       /* the tensor pointer table */
typedef const PLOW_KONST PlowStreamEnt* CStream; /* this CU's flattened stream */

__device__ __forceinline__ CInst as_konst(const PlowDevInst* p) {
    return (CInst)(const PLOW_KONST void*)p;
}
__device__ __forceinline__ CTens as_konst(void* const* p) {
    return (CTens)(const PLOW_KONST void*)p;
}
/* The STREAM is invariant for the same reason `insts` is: the host patches it BETWEEN
 * launches, never during one, and nothing on device writes it. Left as a plain global
 * pointer, each per-packet `e = my[pc]` becomes a vector global_load + v_readfirstlane +
 * a vmcnt(0) that sits at the TOP of the dependent chain (stream -> inst index -> the
 * instruction's own s_load). Marking it constant lets clang emit s_load on lgkmcnt and
 * feed the instruction fetch straight from a scalar register, with no vector round trip. */
__device__ __forceinline__ CStream as_konst(const PlowStreamEnt* p) {
    return (CStream)(const PLOW_KONST void*)p;
}

/* head_dim must be a compile-time constant for the attention ops: the O
 * accumulator is a register array of D/32 f32x16, and a runtime bound would
 * force it to scratch memory (a silent ~10x slowdown). Gemma 4 has exactly two
 * head dims, so the switch is closed. */
__device__ __forceinline__ void exec_flash_prefill(CInst in, CTens T, unsigned slice,
                                                   unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    const unsigned hd = in->i[6];
    /* THREE build shapes for the flash op, chosen at compile time:
     *
     *   PLOW_FLASH_HD128 — Llama/Qwen. Compile the D=128 arm ONLY (and its fused epilogue). D=128
     *       fits the 8-wave / 256-reg budget, so this flash runs INLINE on the 8-wave prefill
     *       interpreter (2 waves/SIMD hides the small-D tile's barrier+softmax latency) with NO
     *       segmented dispatch — the 256/512 arms would force the 512-reg / 1-wave flash object.
     *   PLOW_BUCKET_FLASH — Gemma. The 4-wave / 512-reg standalone flash object (D=128/256/512),
     *       reached by segmented dispatch; D=512's Q-hoist is what needs that budget.
     *   neither (the 8-wave Gemma prefill interpreter) — 256/512 arms compiled but never run
     *       (the seg filter skips them); present only so the switch is total. */
#if defined(PLOW_FLASH_HD128)
    if (hd == 128)
        d_flash_prefill<128>((float*)TEN(0), (float*)TEN(1), (bf16*)TEN(5), (const bf16*)TEN(2),
                             (const bf16*)TEN(3), (const bf16*)TEN(4), in->i[0], in->i[1],
                             in->i[2], in->i[3], in->i[4], in->i[5], in->fj[0].f, in->fj[1].u,
                             in->fj[2].u, in->i[7], slice, nblk, lds);
#else
#if defined(PLOW_BUCKET_FLASH)
    if (hd == 128)
        d_flash_prefill<128>((float*)TEN(0), (float*)TEN(1), (bf16*)TEN(5), (const bf16*)TEN(2),
                             (const bf16*)TEN(3), (const bf16*)TEN(4), in->i[0], in->i[1],
                             in->i[2], in->i[3], in->i[4], in->i[5], in->fj[0].f, in->fj[1].u,
                             in->fj[2].u, in->i[7], slice, nblk, lds);
    else
#endif
    if (hd == 256)
        d_flash_prefill<256>((float*)TEN(0), (float*)TEN(1), (bf16*)TEN(5), (const bf16*)TEN(2),
                             (const bf16*)TEN(3), (const bf16*)TEN(4), in->i[0], in->i[1],
                             in->i[2], in->i[3], in->i[4], in->i[5], in->fj[0].f, in->fj[1].u,
                             in->fj[2].u, in->i[7], slice, nblk, lds);
    else if (hd == 512)
        d_flash_prefill<512>((float*)TEN(0), (float*)TEN(1), (bf16*)TEN(5), (const bf16*)TEN(2),
                             (const bf16*)TEN(3), (const bf16*)TEN(4), in->i[0], in->i[1],
                             in->i[2], in->i[3], in->i[4], in->i[5], in->fj[0].f, in->fj[1].u,
                             in->fj[2].u, in->i[7], slice, nblk, lds);
#endif /* PLOW_FLASH_HD128 */
#undef TEN
}

#if PLOW_FP8_KV
/* FP8 KV twin of exec_flash_prefill: K/V are the e4m3 cache (t3/t4) with per-row scales (t6/t7);
 * d_flash_prefill<D,true> dequantizes them at the LDS stage so the MFMA is unchanged. */
__device__ __forceinline__ void exec_flash_prefill_fp8(CInst in, CTens T, unsigned slice,
                                                       unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    const unsigned hd = in->i[6];
#if defined(PLOW_FLASH_HD128)
    if (hd == 128)
        d_flash_prefill<128, true>((float*)TEN(0), (float*)TEN(1), (bf16*)TEN(5),
                                   (const bf16*)TEN(2), (const bf16*)TEN(3), (const bf16*)TEN(4),
                                   in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->i[5],
                                   in->fj[0].f, in->fj[1].u, in->fj[2].u, in->i[7], slice, nblk, lds,
                                   (const float*)TEN(6), (const float*)TEN(7));
#else
    /* Mirror of exec_flash_prefill's arm set: the class-4 object is the only one that RUNS
     * hd128 (segmented dispatch sends it there), so without this the fp8-KV flash object
     * would silently drop hd128 packets to the default no-op. */
#if defined(PLOW_BUCKET_FLASH)
    if (hd == 128)
        d_flash_prefill<128, true>((float*)TEN(0), (float*)TEN(1), (bf16*)TEN(5),
                                   (const bf16*)TEN(2), (const bf16*)TEN(3), (const bf16*)TEN(4),
                                   in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->i[5],
                                   in->fj[0].f, in->fj[1].u, in->fj[2].u, in->i[7], slice, nblk, lds,
                                   (const float*)TEN(6), (const float*)TEN(7));
    else
#endif
    if (hd == 256)
        d_flash_prefill<256, true>((float*)TEN(0), (float*)TEN(1), (bf16*)TEN(5),
                                   (const bf16*)TEN(2), (const bf16*)TEN(3), (const bf16*)TEN(4),
                                   in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->i[5],
                                   in->fj[0].f, in->fj[1].u, in->fj[2].u, in->i[7], slice, nblk, lds,
                                   (const float*)TEN(6), (const float*)TEN(7));
    else if (hd == 512)
        d_flash_prefill<512, true>((float*)TEN(0), (float*)TEN(1), (bf16*)TEN(5),
                                   (const bf16*)TEN(2), (const bf16*)TEN(3), (const bf16*)TEN(4),
                                   in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->i[5],
                                   in->fj[0].f, in->fj[1].u, in->fj[2].u, in->i[7], slice, nblk, lds,
                                   (const float*)TEN(6), (const float*)TEN(7));
#endif /* PLOW_FLASH_HD128 */
#undef TEN
}
#endif

__device__ __forceinline__ void exec_flash_decode(CInst in, CTens T,
                                                  unsigned slice, unsigned nblk, float* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    const unsigned hd = in->i[6];
    if (hd == 128)
        d_flash_decode<128, PLOW_FA_GF(128)>((float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2),
                            (const bf16*)TEN(3), (const bf16*)TEN(4), (const int*)TEN(5),
                            in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->fj[0].f,
                            in->i[5], in->i[7], slice, nblk, lds);
    else if (hd == 256)
        d_flash_decode<256, PLOW_FA_GF(256)>((float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2),
                            (const bf16*)TEN(3), (const bf16*)TEN(4), (const int*)TEN(5),
                            in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->fj[0].f,
                            in->i[5], in->i[7], slice, nblk, lds);
    else if (hd == 512)
        d_flash_decode<512, PLOW_FA_GF(512)>((float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2),
                            (const bf16*)TEN(3), (const bf16*)TEN(4), (const int*)TEN(5),
                            in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->fj[0].f,
                            in->i[5], in->i[7], slice, nblk, lds);
#undef TEN
}

#if PLOW_FP8_KV
/* FP8 KV twin: K/V are the e4m3 cache (t3/t4), t6/t7 the per-row f32 scales. Only the fp8
 * instantiation is compiled here (the fp8-KV decode object), so it does not add to the register
 * budget of the bf16 flash. */
__device__ __forceinline__ void exec_flash_decode_fp8(CInst in, CTens T, unsigned slice,
                                                      unsigned nblk, float* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    const unsigned hd = in->i[6];
    if (hd == 128)
        d_flash_decode<128, PLOW_FA_GF(128), true>(
            (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
            (const bf16*)TEN(4), (const int*)TEN(5), in->i[0], in->i[1], in->i[2], in->i[3],
            in->i[4], in->fj[0].f, in->i[5], in->i[7], slice, nblk, lds, (const float*)TEN(6),
            (const float*)TEN(7));
    else if (hd == 256)
        d_flash_decode<256, PLOW_FA_GF(256), true>(
            (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
            (const bf16*)TEN(4), (const int*)TEN(5), in->i[0], in->i[1], in->i[2], in->i[3],
            in->i[4], in->fj[0].f, in->i[5], in->i[7], slice, nblk, lds, (const float*)TEN(6),
            (const float*)TEN(7));
    else if (hd == 512)
        d_flash_decode<512, PLOW_FA_GF(512), true>(
            (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
            (const bf16*)TEN(4), (const int*)TEN(5), in->i[0], in->i[1], in->i[2], in->i[3],
            in->i[4], in->fj[0].f, in->i[5], in->i[7], slice, nblk, lds, (const float*)TEN(6),
            (const float*)TEN(7));
#undef TEN
}
#endif
/* Fused MLA merge+fold. VT PICKS THE WORKGROUP COUNT, so it must not be a constant.
 *
 * The kernel splits work over (batch, head, V-tile): `n_work = n_batch*n_head*ceil(V/VT)`. With
 * V=256, a hardcoded VT=256 gives vtiles=1 and n_work = n_head_local — which DEFEATS the entire
 * reason the V-tile axis exists (see d_mla_merge_fold's own OCCUPANCY note: "the standalone
 * merge/fold run on only n_batch*n_head workgroups, starving the 256-CU chip").
 *
 * That constant was swept at tp=1, where nh_l=64. At tp=4 nh_l is 16, so SIXTEEN of 256 workgroups
 * ran the whole op while the other 240 fell out of the loop in 4-7 us. Measured cost: 8.69 ms of a
 * 34.68 ms GLM-5.2 token at 0.6% of the bandwidth ceiling — the single largest line in
 * perf-data/glm52-decode-attribution.md. §4's recurring bug shape: a constant that is right for the
 * config it was tuned on and silently wrong for another.
 *
 * So scale vtiles to the workgroups the op was actually handed. tp=1 and prefill (where
 * n_batch*n_head is already >= nblk/8) keep VT=256, so the workgroup COUNT of those arms does not
 * move; their fold body still gets the rewritten thread map, which is a rounding change, not a
 * dispatch one. VT is picked together with that map — see d_mla_merge_fold's FOLD MAP note in
 * op_attention.h for the sweep that set VT=32. */
__device__ __forceinline__ void exec_mla_merge_fold(CInst in, CTens T, unsigned slice,
                                                    unsigned nblk, float* raw) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    const unsigned bh = in->i[0] * in->i[1]; /* n_batch * n_head_local */
    const unsigned V = in->i[2];             /* v_head_dim: GLM-5.2 256, Kimi-K3 128 */
    if (bh && bh * 8u <= nblk) {
        d_mla_merge_fold<512, PLOW_MLA_FOLD_VT>(
            (bf16*)TEN(0), (const float*)TEN(1), (const float*)TEN(2), (const bf16*)TEN(3),
            in->i[0], in->i[1], in->i[2], in->i[4], slice, nblk, raw);
    } else if (V < 256u && V >= 128u) {
        /* VT MUST NOT EXCEED V, or the fast body is unreachable and the op silently runs 7.7x
         * slower. The fold's fast path requires `v1 - v0 == VT` (op_attention.h), and with VT=256
         * at V=128 `vtiles = ceil(128/256) = 1` gives `v1 - v0 = 128 != 256` — so every workgroup
         * falls into the scalar `else`, the body the 2026-07-28 rewrite replaced for being 7.7x
         * slower. GLM-5.2's V=256 never hits it; Kimi-K3's `v_head_dim: 128` hits it at TP1
         * (bh = 96, 768 > nblk/8) and at batch >= 2 on TP4. Recorded in
         * perf-data/kimi-k3-kernel-gap.md §8b as item 11.
         *
         * THIS BRANCH does not touch GLM: only V in [128, 256) routes here, and GLM-5.2's V is
         * 256. V < 128 keeps the old behaviour (scalar body) rather than growing a third
         * instantiation for a shape no model in this tree has — if one appears, this is where it
         * announces itself, not where it silently degrades.
         *
         * WHAT THIS BRANCH DOES **NOT** SAY — corrected, because the earlier wording here claimed
         * "the GLM arm is untouched … GLM-5.2's fold is bit-identical" without qualification, and
         * a reader auditing "did GLM move?" would have stopped there and concluded no.
         *
         * GLM-5.2 does not reach the `V >= 256` arm at all on the shipping decode path: the FIRST
         * branch above wins. At TP4, n_batch=1 and nh_l=16, so bh=16 and `bh*8 = 128 <= nblk` (the
         * fold is dispatched at `mla_fold_cus` = n_batch*nh_l*ceil(vd/VT) = 128 workgroups — see
         * crates/devgen/src/mla.rs:2207). It therefore takes `<512, PLOW_MLA_FOLD_VT>` with VT=32,
         * which is the NEW map, not the literal <512,256> below.
         *
         * That is intended and is not a numerics regression — the map reassociates minimally
         * (pairwise, whose error bound is strictly BETTER than the sequential sum it replaces; see
         * op_attention.h:2435) and bit-identity is unreachable at the roofline. But it does move
         * GLM's greedy trajectory by the 1-ulp coin-flip that op_attention.h documents, so the
         * recorded 24-token reference (`--prompt 100,264,6722,315,9822,374`,
         * perf-data/glm52-linear-fp8-reeval.md:228) predates it and MUST be re-recorded before it
         * is used as a pass/fail control again. `-DPLOW_MLA_FOLD_MAP=0` restores the old fold for
         * an A/B without editing code. */
        d_mla_merge_fold<512, 128>((bf16*)TEN(0), (const float*)TEN(1), (const float*)TEN(2),
                                   (const bf16*)TEN(3), in->i[0], in->i[1], in->i[2], in->i[4],
                                   slice, nblk, raw);
    } else {
        d_mla_merge_fold<512, 256>((bf16*)TEN(0), (const float*)TEN(1), (const float*)TEN(2),
                                   (const bf16*)TEN(3), in->i[0], in->i[1], in->i[2], in->i[4],
                                   slice, nblk, raw);
    }
#undef TEN
}
/* MLA latent flash decode (DeepSeek). DK=512, DR=64, GF=8 — the validated D=512 decode budget.
 * The dense and GATHER (sparse top-k / DSA) instantiations share this wrapper; GATHER reads only
 * the top_k selected latent rows via the idx table (t7). Both emit latent-wide (O,m,l) partials
 * for FLASH_MERGE<512> + O_UV_FOLD. See d_flash_mla_decode in op_attention.h. [DEEPSEEK-MLA] */
__device__ __forceinline__ void exec_flash_mla_decode(CInst in, CTens T, unsigned slice,
                                                      unsigned nblk, float* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    // Per-PACKET head-fusion factor: the emitter bakes GF into i[7] from the pkt's max_ctx
    // (GF=2 for short-ctx pkts to keep the split/merge overhead low, 4 or 8 for long-ctx pkts
    // to cut latent re-reads). i[7]==0 => legacy default GLM_MLA_GF.
    //
    // THE GF=8 ARM EXISTS AS OF THIS COMMIT. Before it, `glm_gf` baked 8 into i[7] on EVERY
    // long-context GLM blob and the `else` here ran the GF=4 body — silently, with no warning
    // and no trap, so every ctx>4096 GLM-5.2 run measured GF=4 while the emitter, the emitter's
    // comments and `flash_mla_cus` all said 8. That is knob-contract §4's shape (an arm exists,
    // is register-gated, and nothing routes to it) inverted: the SELECTOR existed and the ARM
    // did not.
    //
    // MEASURED COST OF ADDING IT (ROCm 7.2.4 / gfx950, against HEAD 6153189, in ISOLATION from
    // any other in-flight megakernel change), `scripts/build_gfx950.sh`'s own cliff check:
    //   decode   248 -> 254 VGPR / 0 AGPR / occ 2 / spill 0 / LDS 147464   (+6 VGPR)
    //   prefill  256 -> 256 VGPR / 0 AGPR / occ 2 / spill 2 / LDS 147464   (unchanged)
    //   flash    512 -> 512 total / occ 1 / spill 228                       (unchanged)
    // amdgpu_waves_per_eu(2,2) still holds on every object, so the 8-wave dispatch is safe.
    // The GF=8 body is really emitted, not folded away: the decode code object grows 313024 ->
    // 361896 bytes (two instantiations, dense + GATHER).
    //
    // HEADROOM WARNING FOR THE NEXT PERSON TO WIDEN THIS KERNEL: 254 of 256 leaves 2 VGPRs.
    // The allocator does NOT then fail the launch — `amdgpu_waves_per_eu(2,2)` caps it at 256
    // and it SPILLS instead (measured: PLOW_GEMV_MM=8 lands at 256/spill 5 with and without
    // this arm, i.e. the two pressures are not additive, they saturate). So the next arm added
    // here buys scratch traffic, not an INVALID_ISA — check `spill`, not just `occ`.
    const unsigned gf = in->i[7] ? in->i[7] : (unsigned)GLM_MLA_GF;
    if (in->op == PLOW_DOP_FLASH_GATHER_DECODE) {
        if (gf == 2)
            d_flash_mla_decode<512, 64, 2, /*GATHER=*/true>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[1],
                in->i[2], in->i[3], in->fj[0].f, in->i[4], in->i[5], slice, nblk, lds,
                (const int*)TEN(7), in->i[6]);
#if PLOW_GLM_GF8_ARM
        else if (gf == 8)
            d_flash_mla_decode<512, 64, 8, /*GATHER=*/true>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[1],
                in->i[2], in->i[3], in->fj[0].f, in->i[4], in->i[5], slice, nblk, lds,
                (const int*)TEN(7), in->i[6]);
#endif
        else
            d_flash_mla_decode<512, 64, 4, /*GATHER=*/true>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[1],
                in->i[2], in->i[3], in->fj[0].f, in->i[4], in->i[5], slice, nblk, lds,
                (const int*)TEN(7), in->i[6]);
    } else {
        if (gf == 2)
            d_flash_mla_decode<512, 64, 2>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[1],
                in->i[2], in->i[3], in->fj[0].f, in->i[4], in->i[5], slice, nblk, lds);
#if PLOW_GLM_GF8_ARM
        else if (gf == 8)
            d_flash_mla_decode<512, 64, 8>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[1],
                in->i[2], in->i[3], in->fj[0].f, in->i[4], in->i[5], slice, nblk, lds);
#endif
        else
            d_flash_mla_decode<512, 64, 4>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[1],
                in->i[2], in->i[3], in->fj[0].f, in->i[4], in->i[5], slice, nblk, lds);
    }
#undef TEN
}

#if PLOW_MLA_PREFILL
/* MLA PREFILL + GATHERED MLA PREFILL (ops 51 / 55). [DEEPSEEK-MLA]
 *
 * Its own flag, like PLOW_FP8: the prefill bucket already sits AT the 256/occ-2 cliff on the
 * bf16 GEMM, and register allocation here is the worst case over every inlined arm. Only a
 * Kimi/DeepSeek/GLM prefill object needs these, and a Gemma prefill object must not pay for
 * them. Nothing about the existing objects changes when this is off.
 *
 * OPERANDS — identical to the decode twin except i4, which carries n_tok instead of nsplit.
 * That reuse is deliberate rather than a spare slot: nsplit MUST be 1 for prefill (an early
 * token's later splits are empty under a per-token causal bound, and an empty split emits
 * l=0 for d_flash_merge to divide by), so the slot that meant "how many ways the KV range is
 * split" now means "how many query tokens this packet carries" — in both cases the factor the
 * work decomposition multiplies by.
 *   t0=Opart t1=mlpart t2=Qabs t3=Qrope t4=Ckv t5=Krope t6=kv_len t7=idx(gather only)
 *   i0=n_batch i1=n_head i2=kv_stride i3=window i4=n_tok i5=kv_mask i6=top_k i7=gf
 *   f0=scale */
__device__ __forceinline__ void exec_flash_mla_prefill(CInst in, CTens T, unsigned slice,
                                                       unsigned nblk, float* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    const unsigned gf = in->i[7] ? in->i[7] : (unsigned)GLM_MLA_GF;
    if (in->op == PLOW_DOP_FLASH_GATHER_PREFILL) {
        if (gf == 2)
            d_flash_gather_prefill<512, 64, 2>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), (const int*)TEN(7),
                in->i[6], in->i[0], in->i[4], in->i[1], in->i[2], in->fj[0].f, in->i[5], slice,
                nblk, lds);
        else
            d_flash_gather_prefill<512, 64, 4>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), (const int*)TEN(7),
                in->i[6], in->i[0], in->i[4], in->i[1], in->i[2], in->fj[0].f, in->i[5], slice,
                nblk, lds);
    } else {
        if (gf == 2)
            d_flash_mla_prefill<512, 64, 2>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[4],
                in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice, nblk, lds);
        else
            d_flash_mla_prefill<512, 64, 4>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[4],
                in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice, nblk, lds);
    }
#undef TEN
}
#endif

/* These are __forceinline__, NOT noinline.
 *
 * They were noinline for a while, to stop the interpreter's loop state from being
 * live across each op body and inflating its register allocation. That was a
 * mistake: a noinline callee takes `bf16* lds` as a GENERIC pointer, and across a
 * real call boundary the compiler cannot prove it is addrspace(3) -- so LDS access
 * degrades to flat and the call frames spill. The register pressure it was papering
 * over had two real causes (a runtime bucket switch in d_gemv, and the d_gemm_cfg
 * tile switch), both of which are now compile-time. */
__device__ __forceinline__ void exec_gemm_small(CInst in, CTens T,
                                               unsigned slice, unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_small((bf16*)TEN(0), (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                 (const bf16*)TEN(2), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm_med(CInst in, CTens T,
                                              unsigned slice, unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_med((bf16*)TEN(0), (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
               (const bf16*)TEN(2), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm_wide(CInst in, CTens T,
                                               unsigned slice, unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_wide((bf16*)TEN(0), (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                (const bf16*)TEN(2), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm_c5(CInst in, CTens T,
                                             unsigned slice, unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_c5((bf16*)TEN(0), (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
              (const bf16*)TEN(2), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm(CInst in, CTens T,
                                                    unsigned slice, unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    /* i4 = a_row0: skip rows of A. Free — pointer arithmetic, not a kernel change. */
    d_gemm((bf16*)TEN(0), (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2], (const bf16*)TEN(2),
           in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm_norm(CInst in, CTens T,
                                                         unsigned slice, unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_norm((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), (const float*)TEN(3),
                (const bf16*)TEN(4), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemv(CInst in, CTens T,
                                                    unsigned slice, unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemv((bf16*)TEN(0), (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2], (const bf16*)TEN(2),
           (const float*)TEN(3), (const bf16*)TEN(4), in->i[0], in->i[1], in->i[2],
           (int)in->i[3], in->fj[0].f, slice, nblk, lds);
#undef TEN
}
#if PLOW_FP8
/* t0=C t1=x t2=W(fp8) t5=w_scale(f32[N])   i0=M i1=N i2=K i4=a_row0 */
__device__ __forceinline__ void exec_gemv_fp8(CInst in, CTens T, unsigned slice, unsigned nblk,
                                             bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemv_fp8((bf16*)TEN(0), (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
               (const unsigned char*)TEN(2), (const float*)TEN(5), in->i[0], in->i[1], in->i[2],
               slice, nblk, lds);
#undef TEN
}
/* (The block-fp8 decode GEMV wrapper lived here and was a duplicate of the inline body at
 * `case PLOW_DOP_GEMV_FP8_BLK`; the two were kept apart by complementary #if PLOW_FP8 guards.
 * Op 44 is a BLOCK-fp8 arm — a weight encoding independent of the per-channel fp8 this flag
 * names — so the case is now unconditional and there is one body, not two.) */
/* t0=fu t1=x t2=Wg(fp8) t5=Wu(fp8) t3=gate_scale t4=up_scale   i0=M i1=N i2=K i5=act */
__device__ __forceinline__ void exec_gemv_glu_fp8(CInst in, CTens T, unsigned slice, unsigned nblk,
                                                 bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemv_glu_fp8((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                   (const unsigned char*)TEN(5), (const float*)TEN(3), (const float*)TEN(4),
                   in->i[0], in->i[1], in->i[2], in->i[5], slice, nblk, lds);
#undef TEN
}

#if PLOW_BUCKET_PREFILL
/* w8a8 PREFILL arms. d_gemm_fp8* have been in op_gemm.h since the fp8 GEMM landed, but no
 * dispatch arm ever referenced them, so PLOW_DOP_GEMM_FP8 hit the default case and fp8
 * prefill was unreachable on AMD — the 2x-rate wide-K MFMA path was written and unused.
 *
 * These SWAP for the bf16 GEMM arms (see the dispatch), they do not add to them. Register
 * allocation in this kernel is the worst case over ALL inlined code, so carrying both the
 * bf16 and fp8 256x256 GEMMs in one object costs the 256/occ-2 cliff. Same discipline the
 * decode object already applies to its GEMV arms.
 *
 * t0=C(bf16) t1=A(fp8) t2=B(fp8) t3=a_scale(f32[M]) t4=w_scale(f32[N])  i0=M i1=N i2=K i4=a_row0 */
__device__ __forceinline__ void exec_gemm_fp8(CInst in, CTens T, unsigned slice, unsigned nblk,
                                              bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    /* i4 = a_row0 skips rows of A. A is 1 byte/elt here, so the stride is K bytes, not K*2 —
     * the bf16 arm's `+ i4*i2` on a bf16* is the same element count, a different byte count.
     *
     * THE ACTIVATION SCALE MUST BE SKIPPED BY THE SAME a_row0. It is a PER-ROW vector written by
     * QUANT_FP8 over the WHOLE activation, so row `mm` of the shifted A is scaled by
     * a_scale[a_row0 + mm]; the epilogue indexes a_scale[mm], so the base has to move with A.
     * Without this the rows get another row's scale — plausible-but-wrong values, not a fault.
     * LATENT in every packet emitted today (all 990 fp8 GEMMs in the Gemma-4 31B w8a8 packet
     * have a_row0 == 0) and armed the moment one does not: PLOW_FP8_HEAD=1 quantizes the lm_head,
     * which is precisely the op gemma4_chat.c patches with a_row0 = clen-1. The bf16 arm has no
     * such vector, which is why the asymmetry survived this long. */
    d_gemm_fp8((bf16*)TEN(0), (const unsigned char*)TEN(1) + (size_t)in->i[4] * in->i[2],
               (const unsigned char*)TEN(2), (const float*)TEN(3) + in->i[4],
               (const float*)TEN(4), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm_med_fp8(CInst in, CTens T, unsigned slice, unsigned nblk,
                                                  bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_med_fp8((bf16*)TEN(0), (const unsigned char*)TEN(1) + (size_t)in->i[4] * in->i[2],
                   (const unsigned char*)TEN(2), (const float*)TEN(3) + in->i[4],
                   (const float*)TEN(4), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm_small_fp8(CInst in, CTens T, unsigned slice,
                                                    unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_small_fp8((bf16*)TEN(0), (const unsigned char*)TEN(1) + (size_t)in->i[4] * in->i[2],
                     (const unsigned char*)TEN(2), (const float*)TEN(3) + in->i[4],
                     (const float*)TEN(4), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm_wide_fp8(CInst in, CTens T, unsigned slice,
                                                   unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_wide_fp8((bf16*)TEN(0), (const unsigned char*)TEN(1) + (size_t)in->i[4] * in->i[2],
                    (const unsigned char*)TEN(2), (const float*)TEN(3) + in->i[4],
                    (const float*)TEN(4), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm_c5_fp8(CInst in, CTens T, unsigned slice, unsigned nblk,
                                                 bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_c5_fp8((bf16*)TEN(0), (const unsigned char*)TEN(1) + (size_t)in->i[4] * in->i[2],
                  (const unsigned char*)TEN(2), (const float*)TEN(3) + in->i[4],
                  (const float*)TEN(4), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
#endif /* PLOW_BUCKET_PREFILL */
#endif

#if PLOW_MXFP4
/* MXFP4 decode arm. Its own flag, not PLOW_FP8: mxfp4 and fp8 are alternative weight encodings for
 * the same linear, so an object never wants both, and keeping the guards separate means the fp8
 * decode object's register budget is untouched by this existing at all.
 * t0=C t1=x t2=W(fp4) t3=S(e8m0)   i0=M i1=N i2=K */
__device__ __forceinline__ void exec_gemv_mxfp4(CInst in, CTens T, unsigned slice, unsigned nblk,
                                                bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemv_mxfp4((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                 (const unsigned char*)TEN(3), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}

/* MXFP4 decode fused gate|up GEMV+GLU. Mirrors exec_gemv_glu_fp8's slot layout, with E8M0 scale
 * byte tensors in place of the fp8 float scales.
 * t0=C t1=x t2=Wg(fp4) t5=Wu(fp4) t3=Sg(e8m0) t4=Su(e8m0)   i0=M i1=N i2=K i5=act */
__device__ __forceinline__ void exec_gemv_glu_mxfp4(CInst in, CTens T, unsigned slice,
                                                    unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemv_glu_mxfp4((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                     (const unsigned char*)TEN(5), (const unsigned char*)TEN(3),
                     (const unsigned char*)TEN(4), in->i[0], in->i[1], in->i[2], in->i[5], slice,
                     nblk, lds);
#undef TEN
}

#if PLOW_BUCKET_PREFILL
/* MXFP4 (w4a16) PREFILL GEMM arm. A bf16 activations, W packed fp4 + E8M0 scale rows.
 * t0=C(bf16) t1=A(bf16) t2=W(fp4) t3=wscale(e8m0)   i0=M i1=N i2=K */
__device__ __forceinline__ void exec_gemm_mxfp4(CInst in, CTens T, unsigned slice, unsigned nblk,
                                                bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_mxfp4((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                 (const unsigned char*)TEN(3), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
/* The non-default mxfp4 rungs. Identical body, different tile — the divergence that let
 * `d_gemm_mxfp4` keep 256x256 after the bf16 family gained a selector was exactly a
 * hand-copied body, so these are generated from one macro. */
#define PLOW_EXEC_GEMM_MXFP4(NAME, BODY)                                                    \
    __device__ __forceinline__ void NAME(CInst in, CTens T, unsigned slice, unsigned nblk,  \
                                         bf16* lds) {                                       \
        BODY((bf16*)(in->t[0] == PLOW_TENSOR_NONE ? nullptr : T[in->t[0]]),                 \
             (const bf16*)(in->t[1] == PLOW_TENSOR_NONE ? nullptr : T[in->t[1]]),           \
             (const unsigned char*)(in->t[2] == PLOW_TENSOR_NONE ? nullptr : T[in->t[2]]),  \
             (const unsigned char*)(in->t[3] == PLOW_TENSOR_NONE ? nullptr : T[in->t[3]]),  \
             in->i[0], in->i[1], in->i[2], slice, nblk, lds);                               \
    }
PLOW_EXEC_GEMM_MXFP4(exec_gemm_med_mxfp4, d_gemm_med_mxfp4)
PLOW_EXEC_GEMM_MXFP4(exec_gemm_small_mxfp4, d_gemm_small_mxfp4)
PLOW_EXEC_GEMM_MXFP4(exec_gemm_wide_mxfp4, d_gemm_wide_mxfp4)
PLOW_EXEC_GEMM_MXFP4(exec_gemm_c5_mxfp4, d_gemm_c5_mxfp4)
#endif /* PLOW_BUCKET_PREFILL */
#endif

#if PLOW_BUCKET_PREFILL
/* DENSE PREFILL BLOCK-FP8 GEMM arm (w8a16). A bf16, W e4m3 at 1 B/elt, plus DeepSeek/GLM's
 * [128,128] f32 weight_scale_inv grid — the same grid the decode GEMV_FP8_BLK (44) and the grouped
 * MoE prefill arms (85/86) read.
 *
 * NOT under `#if PLOW_FP8`, deliberately. That flag selects the w8a8 rung, and this build axis
 * SWAPS rather than adds (op_gemm.h's fp8 GEMM arms replace the bf16 ones), so gating here would
 * make the arm unavailable in exactly the object GLM's stacked blob loads — `interp_prefill_mla_moe`
 * is built with neither PLOW_FP8 nor PLOW_MXFP4, and it already runs block-fp8 through ops 85/86
 * because the ENCODING there is a runtime field, not a compile flag (knob-contract §3). Block-fp8 is
 * additive w8a16: a packet may legitimately mix a bf16 Gemm with a block-fp8 one, which is exactly
 * what a GLM_LINEAR_FP8 prefill bucket does (q/kv/router bf16, o_proj + shared expert block-fp8).
 *
 * PREFILL BUCKET ONLY, and that is load-bearing rather than tidy: this is a full MFMA body, and the
 * GF=8 arm measured a +32% DECODE regression purely by growing the decode object 15.6% inside the
 * persistent megakernel while holding occupancy 2 with zero spill. Decode's block-fp8 is ops 44/47
 * and has no use for a tiled GEMM.
 * t0=C(bf16) t1=A(bf16) t2=W(e4m3) t3=weight_scale_inv(f32)   i0=M i1=N i2=K */
__device__ __forceinline__ void exec_gemm_fp8_blk(CInst in, CTens T, unsigned slice, unsigned nblk,
                                                  bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_fp8_blk((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                   (const float*)TEN(3), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
#endif /* PLOW_BUCKET_PREFILL */

__device__ static void plow_exec(CInst in, unsigned slice, CTens T, plow_smem* sm) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    const unsigned nblk = in->blocks;
#if defined(PLOW_BUCKET_FLASH)
    /* Flash-ONLY object (segmented dispatch): this launch runs only the class-4 flash_prefill
     * segment; every other op is a class-8 segment on the 8-wave interpreter and is skipped by the
     * seg filter. Compiling ONLY flash frees it from the GEMM's 256 AGPRs, giving flash the full
     * register budget (for FA_DC / Q-hoist / a deeper pipeline). */
    /* SWAP under fp8-KV, exactly as the 8-wave prefill and decode objects do: an fp8-KV packet
     * emits FLASH_PREFILL_FP8 and never the bf16 twin, and carrying both would spend the
     * 512-register budget this object exists to give flash. Without this the fp8-KV path had no
     * class-4 object at all and its hd512 flash had to run on the 8-wave interpreter. */
#if PLOW_FP8_KV
    if (in->op == PLOW_DOP_FLASH_PREFILL_FP8) exec_flash_prefill_fp8(in, T, slice, nblk, sm->fa);
#else
    if (in->op == PLOW_DOP_FLASH_PREFILL) exec_flash_prefill(in, T, slice, nblk, sm->fa);
#endif
    (void)slice;
#else
    switch (in->op) {
        case PLOW_DOP_RMSNORM:
            /* i[2] = out_row0: the OUTPUT cache row (GLM decode-step latent write, patched per step);
             * 0 for every other RMSNORM (in-place), so all existing packets are bit-identical. */
            d_rmsnorm((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), in->i[0],
                      in->i[1], in->fj[0].f, in->i[2], slice, nblk, sm->part);
            break;
        case PLOW_DOP_ROWRMS:
            d_rowrms((float*)TEN(0), (const bf16*)TEN(1), in->i[0], in->i[1], in->fj[0].f, slice,
                     nblk, sm->part);
            break;
        /* DSA indexer k_norm: LayerNorm WITH bias + mean-subtract (the only non-RMS norm). [GLM52-DSA] */
        case PLOW_DOP_LAYERNORM:
            d_layernorm_bias((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                             (const bf16*)TEN(3), in->i[0], in->i[1], in->fj[0].f, in->i[3], slice,
                             nblk, sm->part);
            break;
        case PLOW_DOP_HEADNORM_ROPE:
            /* head_dim is a compile-time constant here for the same reason it is in the
             * attention ops: a runtime one makes `E = hd >> 6` runtime, the e-loops stop
             * unrolling, and every element load becomes a DEPENDENT round trip. */
            /* i[5]==1 selects the INTERLEAVED (GPT-J) HD=128 template — the DSA indexer q/k RoPE
             * (rope the first qk_rope=64 of the 128-d head, pass the rest via an identity-tail cos/sin
             * table). i[5]==0 keeps the existing non-interleaved HD=128 path (every other arch). [GLM52-DSA] */
            if (in->i[2] == 128 && in->i[5] == 1)
                d_headnorm_rope<128, /*INTERLEAVE=*/true>(
                                     (bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const float*)TEN(3), (const float*)TEN(4), (const int*)TEN(5),
                                     in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                                     in->i[4], slice, nblk, in->i[6]);
            else if (in->i[2] == 128)
                d_headnorm_rope<128>((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const float*)TEN(3), (const float*)TEN(4), (const int*)TEN(5),
                                     in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                                     in->i[4], slice, nblk, in->i[6]);
            else if (in->i[2] == 256)
                d_headnorm_rope<256>((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const float*)TEN(3), (const float*)TEN(4), (const int*)TEN(5),
                                     in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                                     in->i[4], slice, nblk, in->i[6]);
            else if (in->i[2] == 512)
                d_headnorm_rope<512>((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const float*)TEN(3), (const float*)TEN(4), (const int*)TEN(5),
                                     in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                                     in->i[4], slice, nblk, in->i[6]);
            /* HD=64 is GLM-5.2's partial-rope slice, and GLM applies INTERLEAVED (GPT-J) RoPE there
             * (dynamic per-step q_rope + shared k_rope for multi-token decode). No other arch uses
             * HD=64, so HD==64 selects the interleaved template; every 128/256/512 path stays
             * bit-identical (INTERLEAVE defaults false). See op_norm.h + rope_interleave_glm_test.c. */
            else if (in->i[2] == 64)
                d_headnorm_rope<64, /*INTERLEAVE=*/true>(
                                     (bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const float*)TEN(3), (const float*)TEN(4), (const int*)TEN(5),
                                     in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                                     in->i[4], slice, nblk, in->i[6]);
            break;
#if PLOW_FP8_KV
        /* FP8 KV write (k/v only; q still uses HEADNORM_ROPE above). t0=uint8 cache, t6=f32 scale. */
        case PLOW_DOP_HEADNORM_ROPE_FP8:
            if (in->i[2] == 128)
                d_headnorm_rope_fp8<128>((unsigned char*)TEN(0), (float*)TEN(6), (const bf16*)TEN(1),
                                         (const bf16*)TEN(2), (const float*)TEN(3),
                                         (const float*)TEN(4), (const int*)TEN(5), in->i[0],
                                         in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u, in->i[4],
                                         slice, nblk, in->i[6]);
            else if (in->i[2] == 256)
                d_headnorm_rope_fp8<256>((unsigned char*)TEN(0), (float*)TEN(6), (const bf16*)TEN(1),
                                         (const bf16*)TEN(2), (const float*)TEN(3),
                                         (const float*)TEN(4), (const int*)TEN(5), in->i[0],
                                         in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u, in->i[4],
                                         slice, nblk, in->i[6]);
            else if (in->i[2] == 512)
                d_headnorm_rope_fp8<512>((unsigned char*)TEN(0), (float*)TEN(6), (const bf16*)TEN(1),
                                         (const bf16*)TEN(2), (const float*)TEN(3),
                                         (const float*)TEN(4), (const int*)TEN(5), in->i[0],
                                         in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u, in->i[4],
                                         slice, nblk, in->i[6]);
            break;
#endif
        case PLOW_DOP_NORM_RESIDUAL:
            d_norm_residual((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                            (const bf16*)TEN(3), in->i[0], in->i[1], in->fj[0].f, in->fj[1].f, slice,
                            nblk, sm->part);
            break;
        case PLOW_DOP_RESIDUAL:
            d_residual((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), in->i[0],
                       in->fj[0].f, slice, nblk);
            break;
        case PLOW_DOP_ADD_NORM:
            d_add_norm((bf16*)TEN(0), (bf16*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                       (const bf16*)TEN(4), in->i[0], in->i[1], in->fj[0].f, slice, nblk, sm->part);
            break;
        case PLOW_DOP_NORM_RESIDUAL_NORM:
            d_norm_residual_norm((bf16*)TEN(0), (bf16*)TEN(1), (const bf16*)TEN(2),
                                 (const bf16*)TEN(3), (const bf16*)TEN(4), (const bf16*)TEN(5),
                                 in->i[0], in->i[1], in->fj[0].f, in->fj[1].f, slice, nblk, sm->part);
            break;
        case PLOW_DOP_GLU:
            d_glu((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), in->i[0], in->i[1],
                  slice, nblk);
            break;
        case PLOW_DOP_EMBED:
            d_embed((bf16*)TEN(0), (const bf16*)TEN(1), (const int*)TEN(2), in->i[0], in->i[1],
                    in->fj[0].f, slice, nblk);
            break;
        case PLOW_DOP_SOFTCAP:
            d_softcap((bf16*)TEN(0), (const bf16*)TEN(1), in->i[0], in->fj[0].f, slice, nblk);
            break;
        /* KDA — Kimi Delta Attention (69 of Kimi-K3's 93 layers). Four arms, present in BOTH
         * buckets because a KDA layer runs the same four ops at T=1 and T>1 (op 102's serial-T
         * loop is the reference `fused_recurrent` algorithm, exact at any T).
         *
         * These exist FROM DAY ONE because op 90 (MAMBA2_SCAN) has no arm here at all, so on
         * gfx950 it falls to the `default:` at the bottom of this switch — which is a silent NOP,
         * not a trap — and computes nothing. `check_gfx950_opcode_coverage` +
         * `dispatched_list_matches_the_amd_interpreter` exist to make that impossible to repeat,
         * and they read THIS FILE for the `case PLOW_DOP_*` labels.
         *
         * Behind PLOW_K3 (default 0) since the axis was added — see the macro's note at the top of
         * this file. `GFX950_DISPATCHED` still lists all seven, exactly as it lists the
         * `PLOW_MOE_PREFILL`-gated `MOE_GROUP_*_PF` ops: that list answers "does the interpreter
         * have an arm for this opcode", which is a property of the SOURCE. Whether THIS object was
         * built with the axis is a different question, answered at load time by `plow_k3_arms_1`.
         * The drift test reads `case PLOW_DOP_*` labels textually and is unaffected by the `#if`. */
#if PLOW_K3
        case PLOW_DOP_KDA_CONV:
            d_kda_conv((bf16*)TEN(0), (const bf16*)TEN(1), (const float*)TEN(2), (float*)TEN(3),
                       in->i[0], in->i[1], in->i[2], in->i[3], slice, nblk);
            break;
        case PLOW_DOP_KDA_GATE:
            d_kda_gate((float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                       (const float*)TEN(4), (const float*)TEN(5), in->i[0], in->i[1], in->i[2],
                       in->i[3], in->fj[0].f, slice, nblk);
            break;
        case PLOW_DOP_KDA_STATE_STEP:
            d_kda_state_step((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                             (const bf16*)TEN(3), (const float*)TEN(4), (const float*)TEN(5),
                             (float*)TEN(6), in->i[0], in->i[1], in->i[2], in->i[3], in->i[4],
                             in->fj[0].f, slice, nblk, (float*)sm->raw);
            break;
        case PLOW_DOP_KDA_GATED_NORM:
            d_kda_gated_norm((bf16*)TEN(0), (const bf16*)TEN(1), (const float*)TEN(2),
                             (const bf16*)TEN(3), in->i[0], in->i[1], in->i[2], in->fj[0].f, slice,
                             nblk);
            break;
        /* Kimi-K3 BLOCK structure (op_k3.h). Both arms are in BOTH buckets: a K3 layer runs the
         * same graph at T=1 and T>1, and AttnRes in particular replaces the residual add on the
         * PREFILL path too — an AttnRes that exists only in the decode object would make prefill
         * and decode compute different models, which is the failure the KDA gate's
         * prefill/decode-agreement check exists to catch. */
        case PLOW_DOP_ATTN_RES:
            d_attn_res((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                       (const float*)TEN(3), in->i[0], in->i[1], in->i[2], in->fj[0].f, slice, nblk,
                       (float*)sm->raw);
            break;
        case PLOW_DOP_SITU_GLU:
            d_situ_glu((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), in->i[0],
                       in->fj[0].f, in->fj[1].f, slice, nblk);
            break;
        case PLOW_DOP_MLA_OUT_GATE:
            d_mla_out_gate((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), in->i[0], slice,
                           nblk);
            break;
#endif /* PLOW_K3 */
        /* Greedy sampling on device. In BOTH buckets: prefill samples the first token, decode
         * every one after it, and the id lands in `in.ids` where the next EMBED reads it. */
        case PLOW_DOP_ARGMAX:
            /* i1 = n_batch (0/1 => single sequence, byte-identical). */
            d_argmax((unsigned long long*)TEN(0), (const bf16*)TEN(1), in->i[0], in->i[1], slice,
                     nblk, sm->am);
            break;
        case PLOW_DOP_ARGMAX_FIN:
            d_argmax_fin((int*)TEN(0), (const unsigned long long*)TEN(1), in->i[0], in->i[1],
                         slice);
            break;
#if PLOW_BUCKET_PREFILL
#if PLOW_FP8
        /* w8a8 PREFILL arms. ADDITIVE — they JOIN the bf16 GEMM arms below, they no longer
         * replace them, and that is a bug fix, not a tidy-up.
         *
         * THE SWAP WAS WRONG BECAUSE ITS PREMISE WAS WRONG. It read "an fp8 prefill program
         * emits no bf16 GEMM". It does: `pick_tile(1, vocab, hidden)` picks the lm_head's tile
         * from the BF16 table with no fp8 substitution (devgen emits `fp8_head` on DECODE only),
         * so every fp8 prefill bucket in build-amd/g31b-w8a8 carries exactly one bf16
         * GEMM_SMALL — M=1, N=262144, K=5376, a_row0=T-1 — and that is the lm_head. With the
         * swap in place `case PLOW_DOP_GEMM_SMALL` was not compiled into interp_prefill_fp8 at
         * all, so the lm_head fell to `default:`, wrote NOTHING, and the logits stayed zero.
         * Argmax over zeros returns index 0 => token id 0 for every prompt. Measured in the
         * shipped objects: interp_prefill.elf has 100 v_mfma_f32_32x32x16_bf16,
         * interp_prefill_fp8.elf has 24 — exactly the flash_prefill's, none of the GEMM's.
         *
         * AND THE COST IT WAS PAYING FOR IS GONE. The register argument was measured under
         * ROCm 7.0.2/clang-20, where these objects landed at 262/occ-1. Re-measured on 7.2.4
         * (2026-07-27), carrying BOTH the bf16 and fp8 256x256 GEMMs plus GEMM_GLU:
         * 256 VGPR / 0 AGPR / occ 2 / spill 20 — IDENTICAL to the fp8-only object, in both the
         * _fp8 and _fp8kv builds. The object grows 174 KB -> 202 KB and nothing else moves.
         *
         * This also makes the WEIGHT-ENCODING axis behave like the other two: PLOW_MXFP4 has
         * always been additive (w4a16 mixes with bf16 by construction) and now PLOW_FP8 is too.
         * The KV-encoding axis (PLOW_FP8_KV) stays a SWAP, deliberately — see its arms below.
         *
         * QUANT_FP8 produces the per-row-quantized activation these consume, so it is only ever
         * needed in the object that has them. */
        case PLOW_DOP_QUANT_FP8:
            /* t0=xq(fp8) t1=x(bf16) t2=a_scale(f32[M])   i0=M i1=K */
            d_quant_fp8((unsigned char*)TEN(0), (const bf16*)TEN(1), (float*)TEN(2),
                        in->i[0], in->i[1], slice, nblk);
            break;
        case PLOW_DOP_GEMM_FP8:
            exec_gemm_fp8(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMM_MED_FP8:
            exec_gemm_med_fp8(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMM_SMALL_FP8:
            exec_gemm_small_fp8(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMM_WIDE_FP8:
            exec_gemm_wide_fp8(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMM_C5_FP8:
            exec_gemm_c5_fp8(in, T, slice, nblk, sm->gm);
            break;
#if PLOW_WAVES == 8
        /* Twin of the bf16 GEMM_GLU arm below, and prefill-only for the same reason.
         * t0=fu t1=A(fp8) t2=Wg t5=Wu t3=a_scale t4=g_scale t6=u_scale  i0=M i1=N i2=K i5=act */
        case PLOW_DOP_GEMM_GLU_FP8:
            d_gemm_glu_fp8((bf16*)TEN(0), (const unsigned char*)TEN(1),
                           (const unsigned char*)TEN(2), (const unsigned char*)TEN(5),
                           (const float*)TEN(3), (const float*)TEN(4), (const float*)TEN(6),
                           in->i[0], in->i[1], in->i[2], in->i[5], slice, nblk, sm->gm);
            break;
#endif
#endif /* PLOW_FP8 — the fp8 arms ADD to the bf16 ones below, they do not replace them */
        /* THE M=1 GEMV ARM, IN THE PREFILL BUCKET. Not a copy-paste of the decode arm by
         * accident — the prefill lm_head is M=1, and running it through a 64x128 tiled GEMM
         * computes 64 rows to keep one (1.5% row efficiency). devgen now emits `Gemv` there on
         * gfx950 instead of `GemmSmall`, which is the right call and which NO PREFILL OBJECT
         * COULD DISPATCH: `case PLOW_DOP_GEMV` lived only in the decode branch, so op 10 in a
         * prefill program hit `default:`, wrote nothing, and the logits were zero — the exact
         * bug the additive fix above just closed, relocated from opcode 14 to opcode 10.
         *
         * NVIDIA has the same arm behind a build flag (PLOW_NV_PF_GEMV_HEAD) and can afford to,
         * because its `default:` is `__trap()` — a loud failure. This interpreter's `default:`
         * is a silent no-op, so the arm must be UNCONDITIONAL here: there is no build/emit
         * mismatch that a gfx950 packet would survive noisily.
         *
         * Measured: 256 VGPR / 0 AGPR / occ 2 / spill 20 with and without, in both the bf16 and
         * the fp8 prefill objects. d_gemv's MM<=PLOW_GEMV_MAXM instantiation is bandwidth-bound
         * scalar code; it adds no accumulator tile to the worst case the GEMM already sets. */
        case PLOW_DOP_GEMV:
            exec_gemv(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMM:
            /* ONE tile config, fixed at compile time — deliberately not a runtime
             * switch over the variants. The interpreter inlines every op, so its
             * register allocation is the WORST CASE over all inlined code. A
             * runtime cfg switch pulls all three GEMM instantiations into that
             * worst case (measured: 260 VGPR + 128 AGPR), which no longer fits 2
             * waves/SIMD and the dispatch is rejected outright.
             *
             * Bucket specialisation therefore has to be a compile-time split:
             * plowc emits an interpreter per bucket with the right tile baked in.
             * The variants live in test_kernels.hip, where they are benchmarked to
             * pick that tile. */
            exec_gemm(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMM_SMALL:
            exec_gemm_small(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMM_MED:
            exec_gemm_med(in, T, slice, nblk, sm->gm);
            break;
        /* The two rungs added by the tile-inventory campaign. Their accumulators (64 and 96
         * registers) are both SMALLER than the 256x256 arm above (128), and their LDS (108
         * and 126 KiB) smaller than its 144 KiB arena, so they do not move the worst case a
         * `case` in this switch sets. Measured, prefill object: 256 VGPR / 0 AGPR / occ 2 /
         * spill 2 -- byte-for-byte the budget before they were added. */
        case PLOW_DOP_GEMM_WIDE:
            exec_gemm_wide(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMM_C5:
            exec_gemm_c5(in, T, slice, nblk, sm->gm);
            break;
        /* DENSE block-fp8 (w8a16), the T-row twin of GEMV_FP8_BLK. Its accumulator is 32 registers
         * plus a 32-register PROMOTION set = 64, strictly under the 256x256 arm's 128, so it cannot
         * move the worst case this switch already sets; its LDS (2*(128+128)*64 halves = 64 KiB) is
         * likewise under the 144 KiB arena. Unconditional in the prefill bucket for the reason the
         * exec wrapper documents: the encoding is additive, not a precision SWAP. */
        case PLOW_DOP_GEMM_FP8_BLK:
            exec_gemm_fp8_blk(in, T, slice, nblk, sm->gm);
            break;
        /* PREFILL-ONLY, and it has to be: d_gemm_glu is a full 256x256 MFMA GEMM, so inlining
         * it into the DECODE interpreter drags 128 AGPRs in and drops that kernel from 2
         * waves/SIMD to 1 (measured: 159 + 128 = 287, over the 256 cliff). Register allocation
         * is the worst case over all inlined code -- the same trap the tile switch has. */
#if PLOW_WAVES == 8
        case PLOW_DOP_GEMM_GLU:
            d_gemm_glu((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                       (const bf16*)TEN(5), in->i[0], in->i[1], in->i[2], in->i[5], slice, nblk,
                       sm->gm);
            break;
#endif
#if PLOW_MXFP4
        /* MXFP4 (w4a16) PREFILL GEMM. This arm USED TO SIT IN THE DECODE BRANCH under a nested
         * `#if PLOW_BUCKET_PREFILL`, which is false by construction there — so it compiled into
         * NOTHING. Verified two ways before moving it: `d_gemm_mxfp4` appears in test_kernels.elf
         * (167 refs) and in ZERO of the seventeen interpreter objects, and a preprocessor sweep of
         * every define combination the build uses finds no `case PLOW_DOP_GEMM_MXFP4` surviving in
         * any of them. An emitter that reached for op 93 would have hit `default:` and silently
         * produced zeros — the arm was written, tested, and unreachable.
         *
         * ADDITIVE, not a swap, unlike the fp8 GEMM arms: mxfp4 here is w4a16, so the A operand is
         * still bf16 and a packet may legitimately mix a bf16 GEMM with an fp4-weight one. Measured
         * cost of adding it to the prefill bucket: 256 VGPR / 0 AGPR / occ 2 -> unchanged. */
        case PLOW_DOP_GEMM_MXFP4:
            exec_gemm_mxfp4(in, T, slice, nblk, sm->gm);
            break;
        /* ...and its four non-default tiles, so the fp4 family gets the SAME per-shape
         * selection the bf16 family has had since GEMM_MED/GEMM_SMALL landed. Without these
         * `pick_tile` had exactly one fp4 rung to choose from and every mxfp4 prefill GEMM
         * ran 256x256 — three tiles on 256 CUs at Kimi's N=576. */
        case PLOW_DOP_GEMM_MED_MXFP4:
            exec_gemm_med_mxfp4(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMM_SMALL_MXFP4:
            exec_gemm_small_mxfp4(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMM_WIDE_MXFP4:
            exec_gemm_wide_mxfp4(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMM_C5_MXFP4:
            exec_gemm_c5_mxfp4(in, T, slice, nblk, sm->gm);
            break;
#endif
#if PLOW_FP8_KV
        /* fp8-KV prefill object: only the fp8 flash is compiled, so the bf16 flash_prefill does
         * not weigh on the prefill register budget (SWAP, mirroring the fp8 GEMV arms). */
        case PLOW_DOP_FLASH_PREFILL_FP8:
            exec_flash_prefill_fp8(in, T, slice, nblk, sm->fa);
            break;
#else
        case PLOW_DOP_FLASH_PREFILL:
            exec_flash_prefill(in, T, slice, nblk, sm->fa);
            break;
#endif
#if PLOW_MOE_PREFILL
        /* MoE PREFILL (T>1) — the FFN half of an MLA-family prompt. Gated by its own flag, and
         * not folded into PLOW_MLA_PREFILL, for the reason every other axis here has its own
         * flag: this adds a second full MFMA GEMM body to a bucket that already sits at the
         * 256/occ-2 cliff, and a Gemma or Qwen prefill object must not pay for it. Measured
         * cost is reported in the build table. */
        case PLOW_DOP_MOE_ROUTER_TOPK_PF:
            d_moe_router_topk_pf((unsigned char*)TEN(0), (const bf16*)TEN(1),
                                 (in->i[3] & 4u) ? (const float*)TEN(3) : nullptr, in->i[1],
                                 in->i[2], in->i[3], in->fj[0].f, in->i[4], slice, nblk,
                                 (float*)sm->raw, in->i[6], in->i[7]);
            break;
        case PLOW_DOP_MOE_ALIGN_PF:
            d_moe_align_pf((int*)TEN(0), (const unsigned char*)TEN(1), (unsigned*)TEN(2),
                           (unsigned*)TEN(3), (float*)TEN(4), in->i[0], in->i[1], in->i[2], slice,
                           (unsigned*)sm->raw);
            break;
        /* i3 = weight encoding (0 bf16, 1 block-fp8, 2 MXFP4/A4W4) — a precision swap is a
         * field change, not a re-emit. t6=row_partidx, t7=row_gate on DOWN; under A4W4 the GLU
         * arm also takes t6 (so the fused bridge can skip pad rows) and t7 = the E8M0 scale
         * rows it WRITES, since the bridge is its epilogue rather than a separate op. */
        case PLOW_DOP_MOE_GROUP_GLU_PF:
            d_moe_group_glu_pf((bf16*)TEN(0), (const bf16*)TEN(1),
                               (const unsigned long long*)TEN(2),
                               (const unsigned long long*)TEN(3), (const int*)TEN(4),
                               (const unsigned*)TEN(5), in->i[0], in->i[1], in->i[2], in->i[3],
                               in->i[5], slice, nblk, sm->raw, nullptr, (const unsigned*)TEN(6),
                               (unsigned char*)TEN(7));
            break;
        case PLOW_DOP_MOE_GROUP_DOWN_PF:
            d_moe_group_down_pf((float*)TEN(0), (const bf16*)TEN(1),
                                (const unsigned long long*)TEN(2),
                                (const unsigned long long*)TEN(3), (const int*)TEN(4),
                                (const unsigned*)TEN(6), (const float*)TEN(7), in->i[0], in->i[1],
                                in->i[2], in->i[3], slice, nblk, sm->raw,
                                (const unsigned char*)TEN(5));
            break;
        case PLOW_DOP_MOE_COMBINE_PF:
            d_moe_combine_pf((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                             (const float*)TEN(3), in->i[0], in->i[1], in->i[2], slice, nblk);
            break;
#endif
#if PLOW_MLA_PREFILL
        /* MLA / gathered-MLA prefill — the arms whose absence meant Kimi K2.7, DeepSeek and
         * GLM 5.2 could decode on gfx950 but had nothing to run a real prompt through. */
        case PLOW_DOP_FLASH_MLA_PREFILL:
        case PLOW_DOP_FLASH_GATHER_PREFILL:
            exec_flash_mla_prefill(in, T, slice, nblk, sm->fmla);
            break;
        /* The latent epilogue has to be in the SAME object as the prefill that produces the
         * partials, or the packet has no way to finish a token. Decode carries its own copy. */
        case PLOW_DOP_MLA_MERGE_FOLD:
            exec_mla_merge_fold(in, T, slice, nblk, (float*)sm->raw);
            break;
        case PLOW_DOP_O_UV_FOLD:
            d_o_uv_fold<512>((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), in->i[0],
                             in->i[1], in->i[2], slice, nblk, sm->raw);
            break;
#endif
        /* split-KV prefill ALWAYS needs the merge, so it lives in this bucket too */
        case PLOW_DOP_FLASH_MERGE:
            if (in->i[3] == 128)
                d_flash_merge<128>((bf16*)TEN(0), (const float*)TEN(1), (const float*)TEN(2),
                                   in->i[0], in->i[1], in->i[2], slice, nblk);
            else if (in->i[3] == 256)
                d_flash_merge<256>((bf16*)TEN(0), (const float*)TEN(1), (const float*)TEN(2),
                                   in->i[0], in->i[1], in->i[2], slice, nblk);
            else if (in->i[3] == 512)
                d_flash_merge<512>((bf16*)TEN(0), (const float*)TEN(1), (const float*)TEN(2),
                                   in->i[0], in->i[1], in->i[2], slice, nblk);
            break;
#else
        case PLOW_DOP_GEMV:
            exec_gemv(in, T, slice, nblk, sm->gm);
            break;
#if PLOW_FP8
        /* The fp8 decode arms. w8a16, NOT w8a8: the activation stays bf16 and the weight is
         * dequantized on load, which is the opposite of what PLOW_FP8 selects in the PREFILL
         * bucket above (w8a8, both operands e4m3). One flag, two activation encodings, split by
         * phase — deliberate (a batch-1 GEMV is weight-bandwidth-bound, so quantizing the
         * activation buys nothing and costs accuracy) but NOT visible in the flag's name.
         * ADDITIVE, like the prefill arms: an fp8 decode program still emits a bf16 GEMV for the
         * lm_head (PLOW_FP8_HEAD off), and a bf16-weight/fp8-KV program emits GEMV_GLU and
         * GEMV_QKV — which this object used to drop. Measured: 248/0/occ 2/spill 0 with the bf16
         * arms back, IDENTICAL to without them. */
        case PLOW_DOP_GEMV_FP8:
            exec_gemv_fp8(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMV_GLU_FP8:
            exec_gemv_glu_fp8(in, T, slice, nblk, sm->gm);
            break;
#endif
#if PLOW_MXFP4
        case PLOW_DOP_GEMV_MXFP4:
            exec_gemv_mxfp4(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMV_GLU_MXFP4:
            exec_gemv_glu_mxfp4(in, T, slice, nblk, sm->gm);
            break;
#endif
        case PLOW_DOP_GEMV_GLU:
            d_gemv_glu((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                       (const bf16*)TEN(5), in->i[0], in->i[1], in->i[2], in->i[5], slice, nblk,
                       sm->gm);
            break;
        case PLOW_DOP_GEMV_QKV:
            d_gemv_qkv((bf16*)TEN(0), (bf16*)TEN(3), (bf16*)TEN(5), (const bf16*)TEN(1),
                       (const bf16*)TEN(2), (const bf16*)TEN(4), (const bf16*)TEN(6), in->i[0],
                       in->i[1], in->i[3], in->i[4], in->i[2], slice, nblk, sm->gm);
            break;
#if PLOW_FP8_KV
        /* fp8-KV decode object: SWAP in the fp8 flash so the bf16 flash_decode is not compiled and
         * the register budget stays flat against the 256/occ-2 cliff. */
        case PLOW_DOP_FLASH_DECODE_FP8:
            exec_flash_decode_fp8(in, T, slice, nblk, sm->fd);
            break;
#else
        case PLOW_DOP_FLASH_DECODE:
            exec_flash_decode(in, T, slice, nblk, sm->fd);
            break;
#endif
        /* ===== DeepSeek MLA (Multi-head Latent Attention) — the flash READ path.  [DEEPSEEK-MLA]
         * The latent flash (dense + gathered) emits latent-wide (O,m,l) folded by FLASH_MERGE<512>;
         * O_UV_FOLD folds the merged latent down to v_head_dim, the per-query epilogue. */
        case PLOW_DOP_FLASH_MLA_DECODE:
        case PLOW_DOP_FLASH_GATHER_DECODE:
            exec_flash_mla_decode(in, T, slice, nblk, sm->fmla);
            break;
        case PLOW_DOP_O_UV_FOLD:
            d_o_uv_fold<512>((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), in->i[0],
                             in->i[1], in->i[2], slice, nblk, sm->raw);
            break;
        /* FUSED MLA merge+fold (replaces FLASH_MERGE<512> + O_UV_FOLD): merges the nsplit latent
         * partials in LDS + folds W_uv straight to v_head. VT is chosen per dispatch — see
         * exec_mla_merge_fold. olds + the fold's cross-wave scratch reuse the raw arena
         * (DK + PLOW_WAVES*VT floats, well inside PLOW_SMEM_HALVES). */
        case PLOW_DOP_MLA_MERGE_FOLD:
            exec_mla_merge_fold(in, T, slice, nblk, (float*)sm->raw);
            break;
        /* On-device top-k KV selection -> idx table (the DSA selector, sparse-attn §3.2). Feeds
         * FLASH_GATHER_DECODE. Packed-key rank in LDS (reuse the arena as u64 score keys). */
        case PLOW_DOP_ATTN_SELECT:
            d_attn_select((int*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), (const int*)TEN(3),
                          in->i[0], in->i[1], in->i[2], in->i[3], in->fj[0].f, slice, nblk,
                          (unsigned long long*)sm->raw);
            break;
        /* DSA lightning-indexer SCORE -> f32 score[ctx] in HBM (consumed by INDEX_SELECT). WIDE-K MFMA
         * variant (perf floor 1: ~94->33us/full-layer @128k, HBM roofline). Streams the key matrix
         * contiguously through LDS and runs the [HI x DI].[DI x ctx] score-dot as 8 wide-K 32x32x16 bf16
         * MFMA k-steps; the fast variant it replaces was HBM-bandwidth-starved (DI-strided key gather).
         * Grid-strided over `slice` (emitted all-CU), NOT over blockIdx.x: under the global-queue
         * scheduler — the DECODE DEFAULT — the workgroup running a stream entry is not the entry's
         * logical index, so a blockIdx.x stride left slabs of `Score` UNWRITTEN. Carves three
         * scratch tiles from the raw arena: qlds (HI*QSTRIDE bf16) | ktile
         * (TILE_N*KSTRIDE bf16) | wlds (HI f32) = 78464 B, well under the raw arena (147464 B) so
         * occupancy is unchanged (relmax 0.0000 vs the _fast/scalar path).
         * index_head_dim=128, index_n_heads=32 are GLM-fixed (template).
         * t0=Score t1=Qidx t2=Kidx t3=W t4=kv_len; i0=n_batch i2=kv_stride; f0=scale. [GLM52-DSA] */
        case PLOW_DOP_INDEX_SCORE: {
            constexpr int DI_ = 128, HI_ = 32;
            constexpr int QSTRIDE = DI_ + FA_PAD, KSTRIDE = DI_ + FA_PAD, TILE_N = PLOW_WAVES * 32;
            bf16* qlds = (bf16*)sm->raw;
            bf16* ktile = qlds + HI_ * QSTRIDE;
            float* wlds = (float*)(ktile + TILE_N * KSTRIDE);
            d_index_score_mfma<DI_, HI_>((float*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                         (const bf16*)TEN(3), (const int*)TEN(4), in->i[0], in->i[2],
                                         in->fj[0].f, slice, nblk, qlds, ktile, wlds);
            break;
        }
        /* DSA top-k SELECT: ONE cooperative launch (this op MUST be emitted on a CU set whose `nblk`
         * workgroups are all co-resident) — 7-pass radix over the packed key with an L2-atomic grid
         * barrier. The kernel is handed `slice`, not blockIdx.x: it partitions the score array by
         * that index, and under the global-queue scheduler (the decode default) the claiming
         * workgroup is arbitrary. `PLOW_GQ_BATCH` MUST STAY 1 for this op to be safe — a workgroup
         * that claimed two select entries would block in the first one's grid barrier waiting for an
         * arrival only it could make. At batch 1 a workgroup blocked in the barrier cannot claim
         * again, so the `nblk` entries are guaranteed to land on `nblk` DISTINCT workgroups.
         * lh[256]+red[3] carve the raw arena.
         * t0=idx t1=Score t2=gHist t3=gCtl t4=kv_len; i0=len_max i1=top_k. [GLM52-DSA]
         *
         * t4 (kv_len) IS REQUIRED, and a null there is the bug this arm was built with: `i[0]` is the
         * packet's MAX ctx while INDEX_SCORE only writes `Score[pos < kv_len]`, so without the live
         * length the radix ranked tens of thousands of never-written words and handed the gather
         * positions past the end of the cache. The kernel falls back to `len_max` when t4 is
         * PLOW_TENSOR_NONE only so an old blob keeps its old (wrong) behaviour instead of faulting;
         * every blob this tree emits binds it. */
        case PLOW_DOP_INDEX_SELECT: {
            unsigned* lh = (unsigned*)sm->raw;
            d_index_select_coop((int*)TEN(0), (const float*)TEN(1), in->i[0], in->i[1],
                                (unsigned*)TEN(2), (unsigned*)TEN(3), slice, nblk, lh, lh + 256,
                                (const int*)TEN(4));
            break;
        }
        /* ===== MoE data-dependent counter-gate (plans/moe-ep-kernels.md §2-§3). Decode-path
         * (bs=1, K-slot dispatch). The body branches on the routing table; the interp still
         * signals the successor unconditionally, so the counter DAG is unchanged. The router's
         * n_exp<=256 score scratch reuses the LDS arena (sm->raw, the largest union member). */
        case PLOW_DOP_MOE_ROUTER:
            /* flags bit 2 (0x4) => t3 is the e_score_correction_bias (f32[n_exp]); else no bias. */
            d_moe_router((unsigned char*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                         (in->i[3] & 4u) ? (const float*)TEN(3) : nullptr, in->i[0], in->i[1],
                         in->i[2], in->i[3], in->fj[0].f, slice, nblk, (float*)sm->raw);
            break;
        case PLOW_DOP_MOE_ROUTER_TOPK: /* t1=logit(bf16) t3=bias; i1=n_exp i2=k i3=flags f0=route_scale */
            /* i6=n_group i7=topk_group (DeepSeek-V3 / Kimi K2 group-limited routing; 0 or 1
             * means flat top-k, which is every GLM/Qwen/Mixtral packet and is bit-identical
             * to before). */
            d_moe_router_topk((unsigned char*)TEN(0), (const bf16*)TEN(1),
                              (in->i[3] & 4u) ? (const float*)TEN(3) : nullptr, in->i[1], in->i[2],
                              in->i[3], in->fj[0].f, slice, nblk, (float*)sm->raw, in->i[6],
                              in->i[7]);
            break;
        case PLOW_DOP_MOE_EXPERT_GLU:
            d_moe_expert_glu((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                             (const unsigned long long*)TEN(3), in->i[0], in->i[1], in->i[2],
                             in->i[3], in->i[5], slice, nblk);
            break;
        case PLOW_DOP_MOE_EXPERT_DOWN:
            d_moe_expert_down((float*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                              (const unsigned long long*)TEN(3), in->i[0], in->i[1], in->i[2],
                              in->i[3], slice, nblk);
            break;
        /* f0/f1 carry Kimi-K3's `situ` betas and are read ONLY when i5 == PLOW_MOE_ACT_SITU. They
         * were free on every GLU-family op, so this consumes no `i` slot and every existing
         * emitter (which leaves them zero) is byte-identical. */
        case PLOW_DOP_MOE_EXPERT_GLU_FP8_BLK: /* t3=wtab t4=stab; i0=slot i1=I_moe i2=H i3=n_exp i5=act i6=ENC f0/f1=situ betas */
            d_moe_expert_glu_fp8_blk((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                                     (const unsigned long long*)TEN(3),
                                     (const unsigned long long*)TEN(4), in->i[0], in->i[1], in->i[2],
                                     in->i[3], in->i[5], slice, nblk, in->i[6], in->fj[0].f,
                                     in->fj[1].f);
            break;
        case PLOW_DOP_MOE_EXPERT_DOWN_FP8_BLK: /* t3=wtab t4=stab; i0=slot i1=H i2=I_moe i3=n_exp i6=ENC */
            d_moe_expert_down_fp8_blk((float*)TEN(0), (const bf16*)TEN(1),
                                      (const unsigned char*)TEN(2), (const unsigned long long*)TEN(3),
                                      (const unsigned long long*)TEN(4), in->i[0], in->i[1], in->i[2],
                                      in->i[3], slice, nblk, in->i[6]);
            break;
        case PLOW_DOP_MOE_GROUP_GLU_FP8_BLK: /* grouped: t3=wtab t4=stab; i0=k i1=I_moe i2=H i3=n_exp i5=act i6=ENC */
            d_moe_group_glu_fp8_blk((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                                    (const unsigned long long*)TEN(3),
                                    (const unsigned long long*)TEN(4), in->i[0], in->i[1], in->i[2],
                                    in->i[3], in->i[5], slice, nblk, in->i[6], in->fj[0].f,
                                    in->fj[1].f);
            break;
        case PLOW_DOP_MOE_GROUP_DOWN_FP8_BLK: /* grouped: t3=wtab t4=stab; i0=k i1=H i2=I_moe i3=n_exp i6=ENC */
            d_moe_group_down_fp8_blk((float*)TEN(0), (const bf16*)TEN(1),
                                     (const unsigned char*)TEN(2), (const unsigned long long*)TEN(3),
                                     (const unsigned long long*)TEN(4), in->i[0], in->i[1], in->i[2],
                                     in->i[3], slice, nblk, in->i[6]);
            break;
        case PLOW_DOP_MOE_COMBINE:
            d_moe_combine((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                          (const float*)TEN(3), in->i[0], in->i[1], slice, nblk);
            break;
        /* GLM dense layers (0-2): block-fp8 SwiGLU gate/up (op 47) + block-fp8 dense DOWN (op 44).
         * These live in the UNCONDITIONAL MoE region (like the expert ops 45/46) so the GLM decode
         * object dispatches them without the PLOW_FP8 object. There USED to be a second, identical
         * `case PLOW_DOP_GEMV_FP8_BLK` under `#if PLOW_FP8` a hundred lines up, the two kept
         * disjoint by `#if !PLOW_FP8` here — a duplicate that only survived because the two guards
         * partitioned the builds. Now that PLOW_FP8 is additive the partition is gone and so is the
         * duplicate: op 44 is a BLOCK-fp8 weight arm, a third weight encoding independent of the
         * per-channel fp8 the PLOW_FP8 axis names, so it belongs here unconditionally. */
        case PLOW_DOP_DENSE_GLU_FP8_BLK: /* t2=Wg t5=Wu t3=Sg t4=Su; i0=N(inter) i1=K(hidden) i5=act */
            d_dense_glu_fp8_blk((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                                (const unsigned char*)TEN(5), (const float*)TEN(3),
                                (const float*)TEN(4), in->i[0], in->i[1], in->i[5], slice, nblk);
            break;
        case PLOW_DOP_GEMV_FP8_BLK: /* dense DOWN: t2=W(fp8) t5=scale; i0=M i1=N i2=K i4=x_row */
            d_gemv_fp8_blk((bf16*)TEN(0), (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                           (const unsigned char*)TEN(2), (const float*)TEN(5), in->i[0], in->i[1],
                           in->i[2], slice, nblk, sm->gm);
            break;
        case PLOW_DOP_FLASH_MERGE:
            if (in->i[3] == 128)
                d_flash_merge<128>((bf16*)TEN(0), (const float*)TEN(1), (const float*)TEN(2),
                                   in->i[0], in->i[1], in->i[2], slice, nblk);
            else if (in->i[3] == 256)
                d_flash_merge<256>((bf16*)TEN(0), (const float*)TEN(1), (const float*)TEN(2),
                                   in->i[0], in->i[1], in->i[2], slice, nblk);
            else if (in->i[3] == 512)
                d_flash_merge<512>((bf16*)TEN(0), (const float*)TEN(1), (const float*)TEN(2),
                                   in->i[0], in->i[1], in->i[2], slice, nblk);
            break;
#endif /* bucket */
        default: /* PLOW_DOP_NOP */
            break;
    }
#endif /* PLOW_BUCKET_FLASH */
#undef TEN
}

/* Agent-scope counter access. The producer may sit on a different XCD with its own L2, so a
 * workgroup-scope load could be served from a stale line.
 *
 * The POLL is RELAXED, and that distinction is worth ~2x on decode.
 *
 * An agent-scope ACQUIRE load emits `buffer_inv` — a full L1/L2 invalidate. Putting one
 * inside the spin loop (which is where it naturally lands if you just write
 * `while (acquire_load(c) < n)`) invalidates the caches on EVERY poll iteration: s_sleep(1)
 * is ~64 cycles, so a gate that waits even briefly issues thousands of cache flushes. It
 * costs time in the spin AND destroys the cache for the work that follows.
 *
 * So: poll relaxed, and take exactly ONE acquire fence once the gate clears. That is all the
 * memory model needs — the fence orders everything the producer released before its counter
 * bump, and it happens once per gate instead of once per poll.
 *
 * The AGENT scope on the acquire is LOAD-BEARING; do NOT narrow it to workgroup. A workgroup
 * acquire drops the `buffer_inv`, which measured 1.6% faster (prefill) / 3.8% (decode) — and it
 * is a data race that a resident-line stress harness fires at a 100% stale-read rate, on BOTH
 * cross-XCD AND same-XCD producer/consumer pairs (see /tmp/rtb/xcd_race.hip, xcd_same.hip). It is
 * unsafe even same-XCD because L1 is per-CU, not per-XCD: two CUs sharing an XCD's L2 still have
 * separate L1s, and a workgroup-scope acquire orders nothing across CUs, so it never invalidates
 * the other CU's stale L1 line. The full model happens to pass only because the 57 GiB/token
 * weight stream evicts stale activation lines before re-read — a hardware accident, not safety.
 * There is no HIP scope between workgroup (no cross-CU) and agent (full L1+L2 inv), so agent is
 * the cheapest sound choice. Tried as an opt-in PLOW_CTR_FENCE macro; reverted once proven a
 * race. Do not re-add. */
/* ---------------------------------------------------------------------------------------
 * THE NEXT BIG LEVER LIVES HERE, AND IT IS MEASURED. [gate-sentinel, not started]
 *
 * MEASURED (MI355X, Gemma-4 31B bf16, real weights, PLOW_TRACE on one decode step): the token
 * is 12.45 ms of op-body work and 3.48 ms of GATE STALL — sync is 21% of every token, at 1134
 * packets/token. Work splits gemv 39% / gemv_glu 38% / gemv_qkv 15% / flash_decode 6.8%, so
 * 92% of the body is the weight-bandwidth-bound GEMV family and the gate is the only large
 * non-GEMV term left.
 *
 * The relaxed-poll + ONE-acquire-fence shape below already banks most of what a naive counter
 * costs (a comparable MI300X monokernel that kept the acquire inside the poll measures grid
 * sync at ~35% of its token; plow is at 21%). What remains is structural: this protocol still
 * pays a `buffer_inv` (full L1+L2 invalidate) per gated packet and a `buffer_wbl2` per signal,
 * 1134 times a token, because a COUNTER is a separate address from the DATA it guards and the
 * consumer therefore has to invalidate everything to see the data.
 *
 * THE ALTERNATIVE, and why it is worth the disruption: publish a SENTINEL INTO THE DATA. The
 * producer initialises its output to a reserved value (NaN) and the consumer polls the output
 * locations themselves; observing a non-sentinel value IS the dependency edge, so there is no
 * counter, no separate acquire, and no full-cache invalidate — only the handoff lines need
 * device-scope semantics. The same MI300X engine measures 0.80-0.93 us for that against
 * 7.59-7.88 us for the counter form on an identical 256-CU / 4-byte-per-CU publish.
 *
 * WHAT MAKES IT NON-TRIVIAL HERE, so nobody starts it thinking it is a small change:
 *   - the sentinel must be a value the math CANNOT produce. NaN works for activations; it does
 *     NOT work for the f32 (O,m,l) flash partials, where -inf/0 are legitimate.
 *   - plow's gates are per-SLICE and fine-grained (PLOW_SE_FINE), so a consumer would poll a
 *     RANGE of producer outputs, not one word — the poll cost scales with the edge, unlike a
 *     counter.
 *   - it interacts with the global queue: a GQ workgroup claims whatever packet is next, so it
 *     cannot know statically which output range to watch.
 * A hybrid (sentinel for the hot coarse activation edges, counters for the rest) is probably
 * the shape, and the trace above already says which edges are worth it.
 * ------------------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------------------
 * CEILING KNOBS. BOTH ARE DELIBERATELY UNSAFE AND MUST NEVER BE SHIPPED.
 *
 * Any sentinel-publish / monotonic-flag protocol replaces the counter with a flag that lives
 * IN the data, so it removes exactly two things and nothing else: the agent-scope acquire's
 * `buffer_inv` on the wait side, and the release RMW's `buffer_wbl2` on the signal side. It
 * does NOT remove the genuine dependency wait -- a consumer still cannot start before its
 * producer has finished, whatever it polls.
 *
 * So the honest question is not "how big is the gate" but "how much of the gate is CACHE
 * MAINTENANCE rather than waiting", and these two knobs answer it by deleting the maintenance
 * and leaving everything else identical. Whatever they measure is the CEILING on the protocol
 * rewrite; the real thing must be slower, because it still has to publish and observe a flag.
 *
 * PLOW_GATE_NOINV   drops the buffer_inv (workgroup-scope acquire orders nothing across CUs).
 * PLOW_GATE_RELAXSIG drops the buffer_wbl2 (a relaxed RMW publishes no prior store).
 *
 * Each is a REAL DATA RACE, not a conservative approximation -- runtime/amd/interp.hip's
 * ctr_poll comment records a resident-line stress harness firing at a 100% stale-read rate
 * with the workgroup-scope acquire, on both cross-XCD and same-XCD producer/consumer pairs.
 * The full model can still produce correct tokens under them, which is precisely the trap:
 * the 57 GiB/token weight stream evicts stale activation lines before re-read. That is a
 * hardware accident and it is why these are measurement instruments, not options. */
#ifndef PLOW_GATE_NOINV
#define PLOW_GATE_NOINV 0
#endif
#ifndef PLOW_GATE_RELAXSIG
#define PLOW_GATE_RELAXSIG 0
#endif

__device__ __forceinline__ uint32_t ctr_poll(const uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void ctr_acquire(void) {
#if PLOW_GATE_NOINV
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"); /* UNSAFE: no buffer_inv */
#else
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#endif
}
/* A RELEASE atomic RMW, and NOT an explicit release fence followed by relaxed RMWs.
 *
 * The fence form looks like the natural mirror of the acquire fix on the wait side ("one
 * invalidate, not one per poll"), and it is a MEASURED LOSS: coarse decode 16.8 -> 17.4
 * ms/token. `__builtin_amdgcn_fence(RELEASE, "agent")` is heavier than the release the atomic
 * already carries — the atomic emits the buffer_wbl2 that release requires and nothing else,
 * while the standalone fence adds an invalidate on top of it.
 *
 * So the asymmetry is real and is not an oversight: RELAX the polls and take ONE acquire fence
 * (an acquire fence is cheaper than an acquire load, which emits buffer_inv on EVERY poll), but
 * on the signal side let each atomic carry its own release. Do not "fix" this. */
__device__ __forceinline__ void ctr_signal(uint32_t* p) {
#if PLOW_GATE_RELAXSIG
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); /* UNSAFE: no wbl2 */
#else
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#endif
}

/* amdgpu_waves_per_eu(2,2) is REQUIRED, not an optimisation hint.
 *
 * __launch_bounds__ alone is not enough. Every op in this kernel fits in 256
 * registers on its own (arch=256, agpr=0), but when they are inlined together the
 * allocator helpfully grabs 128 AGPRs as spill relief -- and since arch and acc are
 * separate files whose maxima SUM, the kernel lands at 256+128 = 384 registers.
 * A wave may use at most 512/waves_per_simd, so 384 means 1 wave/SIMD, and a
 * 512-thread (8-wave) dispatch is then rejected with HSA_STATUS_ERROR_INVALID_ISA
 * -- a spectacularly unhelpful way for the runtime to say "too many registers".
 *
 * This attribute makes 2 waves/SIMD a hard constraint, so the allocator caps
 * itself at 256 total and spills instead of reaching for AGPRs. */
/* Distinct symbol per bucket: the runtime loads BOTH code objects into one HSA executable,
 * and two objects cannot export the same name. */
#if PLOW_BUCKET_DECODE
#define PLOW_INTERP_BASE plow_interp_dec_gfx950
#elif defined(PLOW_BUCKET_FLASH)
/* Segmented-dispatch flash object: prefill op set, but built at 4 waves / FA_DC=256 so the flash
 * segments get the 512-register budget. Distinct symbol; one HSA executable holds all three. */
#define PLOW_INTERP_BASE plow_interp_flash_gfx950
#else
#define PLOW_INTERP_BASE plow_interp_gfx950
#endif
/* Global-queue objects export a distinct _gq symbol so the static and GQ kernels co-load. */
#if PLOW_GLOBAL_QUEUE
#define PLOW_GQ_CAT2(a, b) a##b
#define PLOW_GQ_CAT(a, b) PLOW_GQ_CAT2(a, b)
#define PLOW_INTERP_NAME PLOW_GQ_CAT(PLOW_INTERP_BASE, _gq)
#else
#define PLOW_INTERP_NAME PLOW_INTERP_BASE
#endif

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
__attribute__((amdgpu_waves_per_eu(PLOW_WAVES / 4, PLOW_WAVES / 4))) void PLOW_INTERP_NAME(PlowProgram prog) {
    __shared__ plow_smem sm;

    const unsigned cu = blockIdx.x;
#if PLOW_GLOBAL_QUEUE
    /* One shared cursor over the op-major gq_stream, bounded to this segment's window. base==0
     * because gq_stream is already the global array (no per-CU offset); the trace slot is base+ix. */
    __shared__ unsigned gq_claim;
    /* Per-segment cursor line (its own cache line via PLOW_CTR stride). RUNSEG enqueues all segment
     * launches without a host wait and zeroes state ONCE before the loop, so each segment needs its
     * OWN cursor rather than a shared one reset between launches — same lifecycle as the counters. */
#ifdef PLOW_L2_PLACE_DISPATCH
    /* L2-DOMAIN DISPATCH (PLOW_L2_PLACE). The window this workgroup serves is NOT a host
     * kernarg — it is the L2 domain this workgroup is PHYSICALLY running on. All 8 domains
     * therefore drain concurrently inside ONE launch, each pulling only its own domain's packets,
     * so a consumer reads its producer out of the same XCD's 4 MiB L2 instead of crossing the
     * fabric. `gq_seg_ofs` has one window per domain (the compiler sorted `gq_stream` by domain).
     *
     * READ FROM HW_REG_XCC_ID, NOT COMPUTED FROM blockIdx.x, and the difference matters.
     * `blockIdx.x % n` is what the COMPILER uses to decide which slice lands in which window —
     * it has only the logical index. The DEVICE has the truth, and using it makes the locality
     * property hold BY CONSTRUCTION: window d is then executed only by XCD d's workgroups,
     * whatever the dispatcher does. Deriving it from blockIdx.x instead would make the whole
     * feature silently depend on the round-robin map continuing to hold.
     *
     * What the measurement is still needed for is BALANCE: `runtime/tests/xcd_map_gfx950_test.hip`
     * shows the dispatcher gives each of the 8 XCDs exactly 32 of the 256 workgroups (100.0%
     * round-robin, reproducible), so no domain is left with nobody to drain it. Since the
     * compiler's `n % 8` is uniform too, the windows are equal-sized and the two agree.
     *
     * The modulo guards a partitioned (CPX/NPS) mode where the reported XCC id could exceed the
     * domain count the packet was built for: correctness first, locality degrades gracefully.
     * l2_domains == 0 means the program is not placed — fall back to the host's `cur_seg`. */
    const unsigned plow_xcc =
        __builtin_amdgcn_s_getreg((20 /*HW_REG_XCC_ID*/) | (0 << 6) | ((32 - 1) << 11)) & 0xf;
    const unsigned my_seg = prog.l2_domains ? (plow_xcc % prog.l2_domains) : prog.cur_seg;
#else
    const unsigned my_seg = prog.cur_seg;
#endif
    uint32_t* const cursor = PLOW_CTR(prog.gq_cursor, my_seg);
    const unsigned lo = prog.gq_seg_ofs[my_seg];
    const unsigned hi = prog.gq_seg_ofs[my_seg + 1];
    const CStream my = as_konst(prog.gq_stream);
    const unsigned base = 0;
#else
    const unsigned n = prog.stream_len[cu];
    const CStream my = as_konst(prog.stream) + prog.stream_ofs[cu];

    const unsigned base = prog.stream_ofs[cu];
    /* PER-(CU, SEGMENT) WINDOW. Each CU's stream holds its segments in CONTIGUOUS ascending
     * runs — plowc's `seg_of[]` is a run-length encoding that only increments, and each CU's
     * entries are pushed in op order — so a [lo, hi) window is exactly as valid here as
     * `gq_seg_ofs` is on the op-major stream. With it, a segment launch STARTS at its own
     * entry instead of re-scanning from entry 0 and skipping everything that is not its own.
     *
     * The scalar `srow` is uniform across the workgroup and across the launch, so the guard
     * below is an SGPR compare the compiler hoists out of the packet body. NULL keeps the
     * original scan-and-filter loop, so a host that does not build the table is unaffected. */
    const uint32_t* const srow =
        prog.seg_ofs ? prog.seg_ofs + (size_t)cu * (size_t)(prog.n_seg + 1u) : nullptr;
    const unsigned lo = srow ? srow[prog.cur_seg] : 0u;
    const unsigned hi = srow ? srow[prog.cur_seg + 1u] : n;
#endif

    /* Everything the interpreter reads and never writes, in the constant address space. */
    const CInst insts = as_konst(prog.insts);
    const CTens tens = as_konst(prog.tensors);

#if PLOW_GLOBAL_QUEUE
    for (;;) {
        /* REQUIRED barrier: the whole workgroup must finish the previous packet — including its
         * designated successor signals — before any thread claims the next. The static loop needs no such barrier
         * (its PC is a uniform counter), but the global queue broadcasts the claim through shared
         * `gq_claim` (thread 0 writes, others read), and without this top barrier the next claim races
         * the tail of the previous packet and the kernel deadlocks. Verified: its absence hangs, its
         * presence gives bit-exact-vs-static output deterministically. One intra-workgroup barrier/packet. */
        __syncthreads();
        /* Thread 0 claims PLOW_GQ_BATCH packets with one fetch-add and broadcasts the base via LDS;
         * the whole workgroup then runs each claimed packet together, exactly as the static loop
         * runs one packet across all 512 threads. A per-lane atomicAdd would be a correctness bug.
         * Monotonic claiming over op-major (topological) order means the minimum in-flight index
         * always has all its producers retired — deadlock-free with no grid barrier.
         *
         * THE ARGUMENT SURVIVES PLOW_L2_PLACE_DISPATCH, where 8 domain queues drain at once
         * instead of one, and it is worth spelling out because "8 independent queues" sounds like
         * it should deadlock. Take the globally minimum UNRETIRED packet P across all 8 windows.
         * Every producer of P has a smaller op index, so it is retired. P is therefore runnable.
         * If P is unclaimed its domain's cursor hands it out; if it is claimed, its owner runs
         * its batch in index order and everything earlier in that batch is smaller-index, hence
         * retired — so the owner reaches P. Either way something progresses. This needs the
         * per-domain windows to stay in op-major order, which is why the compiler sorts
         * `gq_stream` by domain with a STABLE sort. */
        if (threadIdx.x == 0)
            gq_claim = __hip_atomic_fetch_add(cursor, (unsigned)PLOW_GQ_BATCH,
                                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __syncthreads();
        const unsigned c0 = lo + gq_claim;
        if (c0 >= hi) break;
        for (unsigned kk = 0; kk < (unsigned)PLOW_GQ_BATCH && c0 + kk < hi; kk++) {
            const unsigned ix = c0 + kk;
            const PlowStreamEnt e = my[ix];
            /* No seg check: the [lo,hi) window already bounds this launch's segment. */
            const CInst in = insts + e.inst;
#else
    for (unsigned ix = lo; ix < hi; ix++) {
        {
            const PlowStreamEnt e = my[ix];
            /* Segmented dispatch. With a window, [lo, hi) ALREADY bounds this launch's segment
             * and this compare is dead — same contract as the global-queue path above. Without
             * one (`seg_ofs == NULL`) the window is the whole stream and the filter is what
             * selects the segment: a skipped entry costs a branch and nothing else — no gate,
             * no exec, no signal — and it is executed by the segment launch that owns it.
             * Unsegmented programs set every seg==0 and launch with cur_seg==0. */
            if (!srow && e.seg != prog.cur_seg) continue;
            const CInst in = insts + e.inst;
#endif

        const uint64_t t_arrive = prog.trace ? __builtin_amdgcn_s_memrealtime() : 0;

        /* This packet's gates live on THIS STREAM ENTRY, always — the 64-byte PlowDevInst
         * carries no wait/succ metadata. A coarse entry points at the op's coarse lists
         * (every workgroup running the op blocks on the same counters); a PLOW_SE_FINE
         * entry carries per-slice lists so this slice blocks only on the producer slices
         * that actually feed it. See PLOW_SE_FINE in dev_isa.h for why the fine form
         * cannot deadlock. */
        const unsigned wait_len = e.wait_len;
        const unsigned wait_ofs = e.wait_ofs;
        const unsigned succ_len = e.succ_len;
        const unsigned succ_ofs = e.succ_ofs;

        /* CROSS-GPU gate (plans/tp-design.md §6a). When set, this entry's wait/succ
         * counters live in the SYSTEM-scope, peer-mapped `xctr` region, not the
         * agent-scope local `counters` — the fine per-tile-shard collective path
         * (§6c). The coarse self-contained XREDUCE (§6b, below) leaves this clear and
         * does its own cross-sync inside the op body. The xctr region sits at the same
         * byte offset in every rank's peer_scratch, so a peer's counter base is
         * peer_scratch[r] + xctr_off; the owner polls its LOCAL xctr. */
        const unsigned xctr = e.flags & PLOW_SE_XCTR;
        const size_t xctr_off = xctr
            ? (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank]
            : 0;

        /* Gate. ONE THREAD PER COUNTER, polled CONCURRENTLY — not one lane walking the list.
         *
         * A serial walk costs wait_len x (cross-XCD latency), and that is not a small constant:
         * each counter was last written by a producer workgroup on some other XCD, so the poll
         * misses L2 and crosses the fabric, ~0.5 us. With a coarse program wait_len is 1-3 and
         * nobody notices. Per-slice gates make it 32 (a headnorm workgroup owns 8 heads and each
         * head is produced by 4 gemv workgroups), and the serial walk then costs ~16 us of pure
         * dependent latency: MEASURED, headnorm_rope went 4.6 -> 21.9 us/inst and decode
         * 16.9 -> 19.8 ms/token. Polling the list concurrently costs ONE fabric round trip.
         *
         * This does not contradict the old "one lane spins so we don't hammer the fabric with
         * 512 loads" rule -- that rule is about 512 threads polling the SAME counter. Here each
         * thread polls a DIFFERENT one, which is parallelism, not contention. Threads beyond
         * wait_len do not poll at all. */
        for (unsigned w = threadIdx.x; w < wait_len; w += PLOW_THREADS) {
            const PlowWait pw = prog.waits[wait_ofs + w];
            /* The sleep constant does not matter. Swept 0 / 1 / 2 / 8, tight-then-backoff
             * and always-tight, on the real model: flat inside noise. Gate latency is NOT made
             * of poll granularity -- the "stall" in the trace is genuine dependency waiting.
             * Do not spend anything tuning this. */
            if (xctr) {
                /* This rank OWNS the counter in its LOCAL xctr; peers wrote it with a
                 * system-scope release. Poll relaxed at system scope (§6a). */
                while (xctr_poll(PLOW_CTR(prog.xctr, pw.id)) < pw.threshold)
                    __builtin_amdgcn_s_sleep(8);
            } else {
                while (ctr_poll(PLOW_CTR(prog.counters, pw.id)) < pw.threshold)
                    __builtin_amdgcn_s_sleep(8);
            }
        }
        __syncthreads(); /* every counter in the list is now satisfied */

        /* ONE acquire for the whole workgroup, and only after the gate clears.
         *
         * An agent-scope acquire emits buffer_inv, a full L1/L2 invalidate. Putting one inside
         * the poll loop (which is where it lands if you just write `while (acquire_load(c) < n)`)
         * flushes the caches on EVERY iteration -- it costs time in the spin AND destroys the
         * cache for the work that follows. That single change was worth ~2x on decode (§8). */
        if (threadIdx.x == 0 && wait_len) { if (xctr) xctr_acquire(); else ctr_acquire(); }
        __syncthreads(); /* the invalidate is visible to every thread before it reads */

        const uint64_t t_ready = prog.trace ? __builtin_amdgcn_s_memrealtime() : 0;

#if !defined(PLOW_BUCKET_FLASH)  /* collectives in the DECODE and 8-wave PREFILL interpreters —
         * TP PREFILL needs XReduce in the prefill interpreter too, not just decode
         * (plans/tp-prefill.md). Excluded from the 4-wave flash object (class-4 flash segments
         * never contain an XReduce, and it keeps that object's tight register budget untouched).
         * Bodies use only the PlowProgram peer fields, present in every bucket. */
        /* CROSS-GPU collectives run here, not in plow_exec, because their bodies need the
         * PlowProgram peer fields (peer_scratch/rank/n_gpu/xctr). */
        if (in->op == PLOW_DOP_XREDUCE) {
            /* COARSE self-contained one-shot all-reduce (plans/tp-design.md §6b, §8a).
             * The local agent gate above already ensured THIS rank's producing GEMV wrote
             * its partial into peer_scratch[rank]; d_xreduce_mega then cross-signals, waits
             * N arrivals (system xctr), and sums the N peer slots. Validated bit-exact +
             * ~0.29 us/collective by tp_allreduce_bench.
             *   t0=out i0=H i1=n_gpu i2=slot_bytes i3=xctr gate id */
            const size_t xoff =
                (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank];
            d_xreduce_mega((bf16*)tens[in->t[0]], prog.peer_scratch, prog.n_gpu, prog.rank,
                           in->i[0], in->i[2], xoff, in->i[3], PLOW_XCTR_DEADLINE_TICKS,
                           nullptr, e.slice, in->blocks);
        } else if (in->op == PLOW_DOP_XREDUCE2) {
            /* TWO-SHOT all-reduce for the LARGE prefill [T,hidden] partial (§4, bandwidth-
             * optimal): reduce-scatter this rank's owned 1/N slice from every peer, then
             * all-gather the N reduced slices. Two internal xctr rendezvous (gate_rs = i3,
             * gate_ag = i4). Same peer-scratch + xctr_off wiring as the one-shot.
             *   t0=out i0=n(t*hidden) i1=n_gpu i2=slot_bytes i3=gate_rs i4=gate_ag */
            const size_t xoff =
                (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank];
            d_xreduce_twoshot_mega((bf16*)tens[in->t[0]], prog.peer_scratch, prog.n_gpu, prog.rank,
                                   in->i[0], in->i[2], xoff, in->i[3], in->i[4],
                                   PLOW_XCTR_DEADLINE_TICKS, nullptr, e.slice, in->blocks);
        } else if (in->op == PLOW_DOP_XARGMAX_FIN) {
            /* Cross-rank fold for a VOCAB-COLUMN-PARALLEL lm_head (plans/tp-design.md §8d).
             * Subsumes ARGMAX_FIN: it does the local nparts fold too, then rebases the index
             * into global vocab space and takes the cross-rank max of the packed key.
             *   t0=ids t1=amax.part  i0=nparts i1=n_batch i2=vocab_l
             *   i3=xctr arrival-gate id  i4=xctr value-slot id (a DIFFERENT id) */
            const size_t xoff =
                (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank];
            d_xargmax_fin_mega((int*)tens[in->t[0]], (const unsigned long long*)tens[in->t[1]],
                               in->i[0], in->i[1], in->i[2], prog.peer_scratch, prog.n_gpu,
                               prog.rank, xoff, in->i[3], in->i[4], PLOW_XCTR_DEADLINE_TICKS,
                               nullptr, e.slice);
        } else if (in->op == PLOW_DOP_XFLASHMERGE) {
            /* STUB (plans/tp-design.md §8c): dispatch present so the switch is total and the
             * compiler's structural emit runs; the CP flash LSE-merge body lands with tp-host.
             * No-op keeps single-GPU decode bit-identical. */
        } else
#endif
        plow_exec(in, e.slice, tens, &sm);

        __syncthreads();

        /* Publish this workgroup's writes, then signal.
         *
         * There is deliberately NO __threadfence() here. ctr_signal is an atomic RMW with
         * __ATOMIC_RELEASE at AGENT scope, and the compiler already emits the buffer_wbl2
         * that release requires (verified in the ISA: dropping the explicit fence takes the
         * kernel from 2 wbl2 + 3 inv to 1 wbl2 + 2 inv -- the writeback survives). The fence
         * was a redundant SECOND full L2 writeback-and-invalidate, on every packet, on all 8
         * XCDs. Decode is 1134 packets per token, so that redundancy was pure tax.
         *
         * Nor is a single explicit release fence + relaxed bumps an improvement when a packet
         * has several successors: TRIED, and it is worse (coarse 16.8 -> 17.4 ms/token). The
         * standalone fence is heavier than the release the atomic already carries.
         *
         * The ordering that makes this sound: __syncthreads() above retires every thread's
         * vector stores (s_waitcnt vmcnt(0)), and the release atomic then orders them ahead
         * of the counter bump that the consumer acquires on. */
        /* One designated thread per successor counter, matching the concurrent wait side.
         * Fine/tile programs can carry many independent successors; walking them on thread 0
         * serializes their cross-XCD latency for no ordering benefit. Every designated writer
         * performs the same release RMW as before. */
        for (unsigned s = threadIdx.x; s < succ_len; s += PLOW_THREADS) {
            const uint32_t sid = prog.succs[succ_ofs + s];
            if (xctr) {
                /* Cross-GPU "partial ready" bump: system-scope release into every peer's
                 * xctr region (the fine per-tile publish, §6c). Each peer's region is at the
                 * same xctr_off inside its peer_scratch; the owner polls its local copy. */
                for (uint32_t r = 0; r < prog.n_gpu; r++) {
                    uint32_t* base = (uint32_t*)((char*)prog.peer_scratch[r] + xctr_off);
                    xctr_signal(PLOW_CTR(base, sid));
                }
            } else {
                ctr_signal(PLOW_CTR(prog.counters, sid));
            }
        }

        if (prog.trace) {
            /* Include every designated counter writer in the trace end timestamp. */
            __syncthreads();
            if (threadIdx.x == 0) {
                /* One store per (workgroup, packet). The slot is unique, so there is
                 * no atomic and no ring -- the trace cannot drop or reorder records. */
                PlowTraceRec r;
                r.cu = cu;
                r.pc = ix;
                r.inst = e.inst;
                r.op = in->op;
                r.slice = (uint16_t)e.slice;
                r.t_arrive = t_arrive;
                r.t_ready = t_ready;
                r.t_end = __builtin_amdgcn_s_memrealtime();
                prog.trace[base + ix] = r;
            }
        }
        }
    }
}
