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

#include "amd_common.h"
#include "op_attention.h"
#include "op_collective.h"
#include "op_elementwise.h"
#include "op_gemm.h"
#include "op_moe.h"
#include "op_norm.h"

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

/* FP8 DECODE object. When 1 (a SEPARATE interp_decode_fp8 object; see build_gfx950_qwen.sh), the
 * decode switch carries the fp8 w8a16 GEMV arms (GEMV_FP8 / GEMV_GLU_FP8) INSTEAD of the bf16
 * GEMV_GLU / GEMV_QKV arms — the fp8 decode program never emits the latter (no QKV fusion in fp8,
 * gate|up go through GEMV_GLU_FP8). Swapping rather than adding keeps the register footprint flat
 * against the 256-VGPR / occ-2 cliff. The plain bf16 GEMV arm (10) stays, for the bf16 lm_head. */
#ifndef PLOW_FP8
#define PLOW_FP8 0
#endif

/* GLOBAL-QUEUE scheduling (Experiment E1). When 1, every workgroup pulls the next packet from ONE
 * shared atomic cursor over the op-major gq_stream instead of walking a static per-CU stream; the op
 * kernels/tiles/register configs are byte-identical, so this is a pure scheduling change. Built as
 * separate _gq objects and selected at launch by the host (PLOW_GLOBAL_QUEUE=1). Measured (batch 1):
 * consistent decode win on both models (dynamic claim smooths static per-CU imbalance), prefill cost
 * model-size-dependent (small on 4B, neutral on 31B). See plans/global-queue-e1-result.md.
 *
 * PLOW_GQ_BATCH must stay 1: claiming >1 packet per fetch-add starves parallelism (a K-wide batch is
 * drained by only n_cu/K workgroups), measured 6-8x slower at K=8. The knob exists only to document
 * that dead end; do not raise it. */
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
    float fmla[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF)]; /* MLA latent decode (DK=512,DR=64,GLM_MLA_GF) */
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
/* MLA latent flash decode (DeepSeek). DK=512, DR=64, GF=8 — the validated D=512 decode budget.
 * The dense and GATHER (sparse top-k / DSA) instantiations share this wrapper; GATHER reads only
 * the top_k selected latent rows via the idx table (t7). Both emit latent-wide (O,m,l) partials
 * for FLASH_MERGE<512> + O_UV_FOLD. See d_flash_mla_decode in op_attention.h. [DEEPSEEK-MLA] */
__device__ __forceinline__ void exec_flash_mla_decode(CInst in, CTens T, unsigned slice,
                                                      unsigned nblk, float* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    // Per-PACKET head-fusion factor: the emitter bakes GF into i[7] from the pkt's max_ctx
    // (GF=2 for short-ctx pkts to keep the split/merge overhead low, GF=4 for long-ctx pkts
    // to cut latent re-reads). i[7]==0 => legacy default GLM_MLA_GF. Both variants are
    // instantiated; the megakernel register/LDS footprint is the GF=4 max (occ unchanged).
    const unsigned gf = in->i[7] ? in->i[7] : (unsigned)GLM_MLA_GF;
    if (in->op == PLOW_DOP_FLASH_GATHER_DECODE) {
        if (gf == 2)
            d_flash_mla_decode<512, 64, 2, /*GATHER=*/true>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[1],
                in->i[2], in->i[3], in->fj[0].f, in->i[4], in->i[5], slice, nblk, lds,
                (const int*)TEN(7), in->i[6]);
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
/* Block-fp8 decode GEMV — t5 is the [ceil(N/128)][ceil(K/128)] f32 scale grid. */
__device__ __forceinline__ void exec_gemv_fp8_blk(CInst in, CTens T, unsigned slice, unsigned nblk,
                                                 bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemv_fp8_blk((bf16*)TEN(0), (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                   (const unsigned char*)TEN(2), (const float*)TEN(5), in->i[0], in->i[1], in->i[2],
                   slice, nblk, lds);
#undef TEN
}
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
     * the bf16 arm's `+ i4*i2` on a bf16* is the same element count, a different byte count. */
    d_gemm_fp8((bf16*)TEN(0), (const unsigned char*)TEN(1) + (size_t)in->i[4] * in->i[2],
               (const unsigned char*)TEN(2), (const float*)TEN(3), (const float*)TEN(4),
               in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm_med_fp8(CInst in, CTens T, unsigned slice, unsigned nblk,
                                                  bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_med_fp8((bf16*)TEN(0), (const unsigned char*)TEN(1) + (size_t)in->i[4] * in->i[2],
                   (const unsigned char*)TEN(2), (const float*)TEN(3), (const float*)TEN(4),
                   in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm_small_fp8(CInst in, CTens T, unsigned slice,
                                                    unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_small_fp8((bf16*)TEN(0), (const unsigned char*)TEN(1) + (size_t)in->i[4] * in->i[2],
                     (const unsigned char*)TEN(2), (const float*)TEN(3), (const float*)TEN(4),
                     in->i[0], in->i[1], in->i[2], slice, nblk, lds);
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
#endif /* PLOW_BUCKET_PREFILL */
#endif

__device__ static void plow_exec(CInst in, unsigned slice, CTens T, plow_smem* sm) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    const unsigned nblk = in->blocks;
#if defined(PLOW_BUCKET_FLASH)
    /* Flash-ONLY object (segmented dispatch): this launch runs only the class-4 flash_prefill
     * segment; every other op is a class-8 segment on the 8-wave interpreter and is skipped by the
     * seg filter. Compiling ONLY flash frees it from the GEMM's 256 AGPRs, giving flash the full
     * register budget (for FA_DC / Q-hoist / a deeper pipeline). */
    if (in->op == PLOW_DOP_FLASH_PREFILL) exec_flash_prefill(in, T, slice, nblk, sm->fa);
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
                                     in->i[4], slice, nblk);
            else if (in->i[2] == 128)
                d_headnorm_rope<128>((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const float*)TEN(3), (const float*)TEN(4), (const int*)TEN(5),
                                     in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                                     in->i[4], slice, nblk);
            else if (in->i[2] == 256)
                d_headnorm_rope<256>((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const float*)TEN(3), (const float*)TEN(4), (const int*)TEN(5),
                                     in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                                     in->i[4], slice, nblk);
            else if (in->i[2] == 512)
                d_headnorm_rope<512>((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const float*)TEN(3), (const float*)TEN(4), (const int*)TEN(5),
                                     in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                                     in->i[4], slice, nblk);
            /* HD=64 is GLM-5.2's partial-rope slice, and GLM applies INTERLEAVED (GPT-J) RoPE there
             * (dynamic per-step q_rope + shared k_rope for multi-token decode). No other arch uses
             * HD=64, so HD==64 selects the interleaved template; every 128/256/512 path stays
             * bit-identical (INTERLEAVE defaults false). See op_norm.h + rope_interleave_glm_test.c. */
            else if (in->i[2] == 64)
                d_headnorm_rope<64, /*INTERLEAVE=*/true>(
                                     (bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const float*)TEN(3), (const float*)TEN(4), (const int*)TEN(5),
                                     in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                                     in->i[4], slice, nblk);
            break;
#if PLOW_FP8_KV
        /* FP8 KV write (k/v only; q still uses HEADNORM_ROPE above). t0=uint8 cache, t6=f32 scale. */
        case PLOW_DOP_HEADNORM_ROPE_FP8:
            if (in->i[2] == 128)
                d_headnorm_rope_fp8<128>((unsigned char*)TEN(0), (float*)TEN(6), (const bf16*)TEN(1),
                                         (const bf16*)TEN(2), (const float*)TEN(3),
                                         (const float*)TEN(4), (const int*)TEN(5), in->i[0],
                                         in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u, in->i[4],
                                         slice, nblk);
            else if (in->i[2] == 256)
                d_headnorm_rope_fp8<256>((unsigned char*)TEN(0), (float*)TEN(6), (const bf16*)TEN(1),
                                         (const bf16*)TEN(2), (const float*)TEN(3),
                                         (const float*)TEN(4), (const int*)TEN(5), in->i[0],
                                         in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u, in->i[4],
                                         slice, nblk);
            else if (in->i[2] == 512)
                d_headnorm_rope_fp8<512>((unsigned char*)TEN(0), (float*)TEN(6), (const bf16*)TEN(1),
                                         (const bf16*)TEN(2), (const float*)TEN(3),
                                         (const float*)TEN(4), (const int*)TEN(5), in->i[0],
                                         in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u, in->i[4],
                                         slice, nblk);
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
        /* Greedy sampling on device. In BOTH buckets: prefill samples the first token, decode
         * every one after it, and the id lands in `in.ids` where the next EMBED reads it. */
        case PLOW_DOP_ARGMAX:
            d_argmax((unsigned long long*)TEN(0), (const bf16*)TEN(1), in->i[0], slice, nblk,
                     sm->am);
            break;
        case PLOW_DOP_ARGMAX_FIN:
            d_argmax_fin((int*)TEN(0), (const unsigned long long*)TEN(1), in->i[0], slice);
            break;
#if PLOW_BUCKET_PREFILL
#if PLOW_FP8
        /* w8a8 PREFILL object. The fp8 arms SWAP for the bf16 GEMM/GEMM_GLU arms rather than
         * joining them: an fp8 prefill program emits no bf16 GEMM, and carrying both 256x256
         * GEMMs would set this kernel's worst-case register allocation over the 256/occ-2
         * cliff. QUANT_FP8 produces the per-row-quantized activation these consume, so it is
         * only ever needed in the object that has them. */
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
#else
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
#endif /* PLOW_FP8 (w8a8 prefill arms swap for the bf16 GEMM arms above) */
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
            d_mla_merge_fold<512, 256>((bf16*)TEN(0), (const float*)TEN(1), (const float*)TEN(2),
                                       (const bf16*)TEN(3), in->i[0], in->i[1], in->i[2], in->i[4],
                                       slice, nblk, (float*)sm->raw);
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
        /* The fp8 decode object carries the fp8 GEMV arms in place of the bf16 GEMV_GLU / GEMV_QKV
         * arms (the fp8 program emits neither of those — gate|up go through GEMV_GLU_FP8, and there
         * is no fp8 QKV fusion). Swapping keeps the register budget flat against the 256/occ-2 cliff. */
        case PLOW_DOP_GEMV_FP8:
            exec_gemv_fp8(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMV_GLU_FP8:
            exec_gemv_glu_fp8(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMV_FP8_BLK:
            exec_gemv_fp8_blk(in, T, slice, nblk, sm->gm);
            break;
#endif
#if PLOW_MXFP4
        case PLOW_DOP_GEMV_MXFP4:
            exec_gemv_mxfp4(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMV_GLU_MXFP4:
            exec_gemv_glu_mxfp4(in, T, slice, nblk, sm->gm);
            break;
#if PLOW_BUCKET_PREFILL
        case PLOW_DOP_GEMM_MXFP4:
            exec_gemm_mxfp4(in, T, slice, nblk, sm->gm);
            break;
#endif
#endif
#if !PLOW_FP8
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
#endif
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
         * partials in LDS + folds W_uv straight to v_head. VT=256 (whole-V per WG, no re-merge —
         * the measured best). olds reuses the raw arena (float[512] fits PLOW_SMEM_HALVES). */
        case PLOW_DOP_MLA_MERGE_FOLD:
            d_mla_merge_fold<512, 256>((bf16*)TEN(0), (const float*)TEN(1), (const float*)TEN(2),
                                       (const bf16*)TEN(3), in->i[0], in->i[1], in->i[2], in->i[4],
                                       slice, nblk, (float*)sm->raw);
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
         * Grid-strided over blockIdx.x (co-resident, emitted all-CU) — it DROPS the `slice` arg the fast
         * kernel used. Carves three scratch tiles from the raw arena: qlds (HI*QSTRIDE bf16) | ktile
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
                                         in->fj[0].f, nblk, qlds, ktile, wlds);
            break;
        }
        /* DSA top-k SELECT: ONE cooperative launch (this op MUST be emitted with blocks==n_cu so all
         * nblk WGs are co-resident) — 7-pass radix over the packed key with an L2-atomic grid barrier.
         * lh[256]+red[2] carve the raw arena. t0=idx t1=Score t2=gHist t3=gCtl; i0=len i1=top_k. [GLM52-DSA] */
        case PLOW_DOP_INDEX_SELECT: {
            unsigned* lh = (unsigned*)sm->raw;
            d_index_select_coop((int*)TEN(0), (const float*)TEN(1), in->i[0], in->i[1],
                                (unsigned*)TEN(2), (unsigned*)TEN(3), nblk, lh, lh + 256);
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
            d_moe_router_topk((unsigned char*)TEN(0), (const bf16*)TEN(1),
                              (in->i[3] & 4u) ? (const float*)TEN(3) : nullptr, in->i[1], in->i[2],
                              in->i[3], in->fj[0].f, slice, nblk, (float*)sm->raw);
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
        case PLOW_DOP_MOE_EXPERT_GLU_FP8_BLK: /* t3=wtab t4=stab; i0=slot i1=I_moe i2=H i3=n_exp i5=act */
            d_moe_expert_glu_fp8_blk((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                                     (const unsigned long long*)TEN(3),
                                     (const unsigned long long*)TEN(4), in->i[0], in->i[1], in->i[2],
                                     in->i[3], in->i[5], slice, nblk);
            break;
        case PLOW_DOP_MOE_EXPERT_DOWN_FP8_BLK: /* t3=wtab t4=stab; i0=slot i1=H i2=I_moe i3=n_exp */
            d_moe_expert_down_fp8_blk((float*)TEN(0), (const bf16*)TEN(1),
                                      (const unsigned char*)TEN(2), (const unsigned long long*)TEN(3),
                                      (const unsigned long long*)TEN(4), in->i[0], in->i[1], in->i[2],
                                      in->i[3], slice, nblk);
            break;
        case PLOW_DOP_MOE_GROUP_GLU_FP8_BLK: /* grouped: t3=wtab t4=stab; i0=k i1=I_moe i2=H i3=n_exp i5=act */
            d_moe_group_glu_fp8_blk((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                                    (const unsigned long long*)TEN(3),
                                    (const unsigned long long*)TEN(4), in->i[0], in->i[1], in->i[2],
                                    in->i[3], in->i[5], slice, nblk);
            break;
        case PLOW_DOP_MOE_GROUP_DOWN_FP8_BLK: /* grouped: t3=wtab t4=stab; i0=k i1=H i2=I_moe i3=n_exp */
            d_moe_group_down_fp8_blk((float*)TEN(0), (const bf16*)TEN(1),
                                     (const unsigned char*)TEN(2), (const unsigned long long*)TEN(3),
                                     (const unsigned long long*)TEN(4), in->i[0], in->i[1], in->i[2],
                                     in->i[3], slice, nblk);
            break;
        case PLOW_DOP_MOE_COMBINE:
            d_moe_combine((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                          (const float*)TEN(3), in->i[0], in->i[1], slice, nblk);
            break;
        /* GLM dense layers (0-2): block-fp8 SwiGLU gate/up (op 47) + block-fp8 dense DOWN (op 44).
         * These live in the UNCONDITIONAL MoE region (like the expert ops 45/46) so the GLM decode
         * object dispatches them without the PLOW_FP8 object (which drops the bf16 GEMV_GLU the GLM
         * shared expert needs). op 44 already has a case under #if PLOW_FP8 above, so guard this twin. */
        case PLOW_DOP_DENSE_GLU_FP8_BLK: /* t2=Wg t5=Wu t3=Sg t4=Su; i0=N(inter) i1=K(hidden) i5=act */
            d_dense_glu_fp8_blk((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                                (const unsigned char*)TEN(5), (const float*)TEN(3),
                                (const float*)TEN(4), in->i[0], in->i[1], in->i[5], slice, nblk);
            break;
#if !PLOW_FP8
        case PLOW_DOP_GEMV_FP8_BLK: /* dense DOWN: t2=W(fp8) t5=scale; i0=M i1=N i2=K i4=x_row */
            d_gemv_fp8_blk((bf16*)TEN(0), (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                           (const unsigned char*)TEN(2), (const float*)TEN(5), in->i[0], in->i[1],
                           in->i[2], slice, nblk, sm->gm);
            break;
#endif
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
__device__ __forceinline__ uint32_t ctr_poll(const uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void ctr_acquire(void) {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
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
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
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
    uint32_t* const cursor = PLOW_CTR(prog.gq_cursor, prog.cur_seg);
    const unsigned lo = prog.gq_seg_ofs[prog.cur_seg];
    const unsigned hi = prog.gq_seg_ofs[prog.cur_seg + 1];
    const CStream my = as_konst(prog.gq_stream);
    const unsigned base = 0;
#else
    const unsigned n = prog.stream_len[cu];
    const CStream my = as_konst(prog.stream) + prog.stream_ofs[cu];

    const unsigned base = prog.stream_ofs[cu];
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
         * always has all its producers retired — deadlock-free with no grid barrier. */
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
    for (unsigned ix = 0; ix < n; ix++) {
        {
            const PlowStreamEnt e = my[ix];
            /* Segmented dispatch: run only this segment's entries. A skipped entry costs a branch and
             * nothing else — no gate, no exec, no signal — and it is executed by the segment launch
             * that owns it. Unsegmented programs set every seg==0 and launch with cur_seg==0. */
            if (e.seg != prog.cur_seg) continue;
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
        } else if (in->op == PLOW_DOP_XFLASHMERGE || in->op == PLOW_DOP_XARGMAX_FIN) {
            /* STUBS (plans/tp-design.md §8c/§8d): dispatch present so the switch is total and
             * the compiler's structural emit runs; bodies land in Phase-2 with tp-host (the CP
             * flash LSE-merge and the sharded lm_head id-fold need the N-rank harness to
             * validate end-to-end). No-op keeps single-GPU decode bit-identical. */
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
