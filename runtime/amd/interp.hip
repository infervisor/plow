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

#ifdef PLOW_CONFIG
#include PLOW_CONFIG
#endif
#ifndef PLOW_PACKET_HASH
#define PLOW_PACKET_HASH 0ull
#endif
#ifndef PLOW_HAS_KDA_CONV
#define PLOW_HAS_KDA_CONV 1
#endif
#ifndef PLOW_HAS_KDA_GATE
#define PLOW_HAS_KDA_GATE 1
#endif
#ifndef PLOW_HAS_KDA_STATE_STEP
#define PLOW_HAS_KDA_STATE_STEP 1
#endif
#ifndef PLOW_HAS_KDA_GATED_NORM
#define PLOW_HAS_KDA_GATED_NORM 1
#endif
#ifndef PLOW_HAS_ATTN_RES
#define PLOW_HAS_ATTN_RES 1
#endif
#ifndef PLOW_HAS_SITU_GLU
#define PLOW_HAS_SITU_GLU 1
#endif
#ifndef PLOW_HAS_MLA_OUT_GATE
#define PLOW_HAS_MLA_OUT_GATE 1
#endif
#ifndef PLOW_HAS_KDA_CONV3
#define PLOW_HAS_KDA_CONV3 1
#endif
#ifndef PLOW_HAS_KDA_STATE_STEP_G
#define PLOW_HAS_KDA_STATE_STEP_G 1
#endif
#ifndef PLOW_HAS_KDA_CONV_STATE_STEP_G
#define PLOW_HAS_KDA_CONV_STATE_STEP_G 1
#endif
#ifndef PLOW_HAS_KDA_CHUNK_PREPARE
#define PLOW_HAS_KDA_CHUNK_PREPARE 1
#endif
#ifndef PLOW_HAS_KDA_CHUNK_INTRA
#define PLOW_HAS_KDA_CHUNK_INTRA 1
#endif
#ifndef PLOW_HAS_KDA_CHUNK_WU
#define PLOW_HAS_KDA_CHUNK_WU 1
#endif
#ifndef PLOW_HAS_KDA_CHUNK_CARRY
#define PLOW_HAS_KDA_CHUNK_CARRY 1
#endif
#ifndef PLOW_MATERIALIZED_RESIDUAL_INPUT
#define PLOW_MATERIALIZED_RESIDUAL_INPUT 0
#endif

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
 * `perf-data/archive/k3/kimi-k3-kernel-gap.md:791` independently reached the same conclusion from the
 * register side ("Build it as a 4th co-resident object, not an arm").
 *
 * TURNING THIS OFF IS NOT SILENT — see `plow_k3_arms_1` below. AMD's dispatch `default:` writes
 * NOTHING (it does not trap, unlike sm_120), so a K3 packet run against an object built without
 * this axis would compute nothing and surface as an accuracy bug. The marker symbol makes the
 * loader refuse that pairing before the object reaches a device. */
#ifndef PLOW_K3
#define PLOW_K3 0
#endif
#ifndef PLOW_KDA_CHUNK
#define PLOW_KDA_CHUNK 0
#endif
#if PLOW_KDA_CHUNK && !PLOW_K3
#error "PLOW_KDA_CHUNK requires the generic KDA bodies compiled by PLOW_K3"
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

#if MPF_BM == 64
extern "C" __device__ unsigned plow_moe_align_bm64_1 = 1;
#endif

/* This proves only that the object was compiled against the packed-prefill PlowProgram tail.
 * Descriptor-consuming math arms are a separate capability and are not enabled by this marker. */
#if PLOW_PACKET_HASH != 0ull
extern "C" __device__ unsigned plow_packet_hash_lo =
    (unsigned)(PLOW_PACKET_HASH & 0xFFFFFFFFull);
extern "C" __device__ unsigned plow_packet_hash_hi =
    (unsigned)((PLOW_PACKET_HASH >> 32) & 0xFFFFFFFFull);
#endif

extern "C" __device__ unsigned plow_packed_prefill_abi_1 = 1;
#if PLOW_PACKED_PREFILL_MLA_NORM_CONSUMERS && PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS && \
    (PLOW_MLA_PREFILL || PLOW_MLA_PF_V2_ARM)
extern "C" __device__ unsigned plow_packed_prefill_mla_consumers_1 = 1;
#endif
#if PLOW_PACKED_PREFILL_KDA_CONSUMERS && PLOW_K3
extern "C" __device__ unsigned plow_packed_prefill_kda_consumers_1 = 1;
#endif
#if defined(PLOW_BUCKET_PACKED_MLA_NORM)
#if !PLOW_PACKED_PREFILL_MLA_NORM_CONSUMERS
#error "packed MLA-norm bucket requires PLOW_PACKED_PREFILL_MLA_NORM_CONSUMERS"
#endif
extern "C" __device__ unsigned plow_packed_prefill_mla_norm_segments_1 = 1;
#endif
#if defined(PLOW_BUCKET_PACKED_KDA)
#if !PLOW_PACKED_PREFILL_KDA_CONSUMERS || !PLOW_K3
#error "packed KDA bucket requires PLOW_PACKED_PREFILL_KDA_CONSUMERS and PLOW_K3"
#endif
extern "C" __device__ unsigned plow_packed_prefill_kda_serial_segments_1 = 1;
#if PLOW_KDA_CHUNK
extern "C" __device__ unsigned plow_packed_prefill_kda_chunk_segments_1 = 1;
#endif
#endif
#if defined(PLOW_BUCKET_FLASH) && PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS && PLOW_MLA_PF_V2_ARM
extern "C" __device__ unsigned plow_packed_prefill_mla_flash_segments_1 = 1;
#endif

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
#if PLOW_HAS_KDA_CONV
extern "C" __device__ unsigned plow_opcode_kda_conv_1 = 1;
#endif
#if PLOW_HAS_KDA_GATE
extern "C" __device__ unsigned plow_opcode_kda_gate_1 = 1;
#endif
#if PLOW_HAS_KDA_STATE_STEP
extern "C" __device__ unsigned plow_opcode_kda_state_step_1 = 1;
#endif
#if PLOW_HAS_KDA_GATED_NORM
extern "C" __device__ unsigned plow_opcode_kda_gated_norm_1 = 1;
#endif
#if PLOW_HAS_ATTN_RES
extern "C" __device__ unsigned plow_opcode_attn_res_1 = 1;
#if PLOW_MATERIALIZED_RESIDUAL_INPUT
extern "C" __device__ unsigned plow_materialized_residual_input_1 = 1;
#endif
#endif
#if PLOW_HAS_SITU_GLU
extern "C" __device__ unsigned plow_opcode_situ_glu_1 = 1;
#endif
#if PLOW_HAS_MLA_OUT_GATE
extern "C" __device__ unsigned plow_opcode_mla_out_gate_1 = 1;
#endif
#if PLOW_HAS_KDA_CONV3
extern "C" __device__ unsigned plow_opcode_kda_conv3_1 = 1;
#endif
#if PLOW_HAS_KDA_STATE_STEP_G
extern "C" __device__ unsigned plow_opcode_kda_state_step_g_1 = 1;
#endif
#if PLOW_HAS_KDA_CONV_STATE_STEP_G
extern "C" __device__ unsigned plow_opcode_kda_conv_state_step_g_1 = 1;
#endif
#if PLOW_HAS_KDA_CHUNK_PREPARE
extern "C" __device__ unsigned plow_opcode_kda_chunk_prepare_1 = 1;
#endif
#if PLOW_HAS_KDA_CHUNK_INTRA
extern "C" __device__ unsigned plow_opcode_kda_chunk_intra_1 = 1;
#endif
#if PLOW_HAS_KDA_CHUNK_WU
extern "C" __device__ unsigned plow_opcode_kda_chunk_wu_1 = 1;
#endif
#if PLOW_HAS_KDA_CHUNK_CARRY
extern "C" __device__ unsigned plow_opcode_kda_chunk_carry_1 = 1;
#endif
#endif
#if PLOW_K3 && PLOW_KDA_CHUNK
extern "C" __device__ unsigned plow_kda_chunk_bt64_arm_1 = 1;
extern "C" __device__ unsigned plow_kda_chunk_qpre_arm_1 = 1;
#endif
#if PLOW_K3 && PLOW_KDA_CONV_STEP_DB
extern "C" __device__ unsigned plow_kda_conv_step_db_arm = 1;
#endif

/* Compact TP audit. `j[1] == 0` keeps old packets byte-compatible; plowrt patches
 * TP collective instructions to status_id+1 while loading them. */
__device__ __forceinline__ uint32_t* plow_xaudit_status(const PlowProgram& prog,
                                                        const PlowDevInst* in) {
    return in->fj[2].u ? PLOW_CTR(prog.xctr, in->fj[2].u - 1u) : nullptr;
}

typedef struct {
    const PlowDevInst* insts;
    uint32_t n_inst;
    uint32_t _pad;
    const uint32_t* xctr;
    uint32_t n_xctr;
    uint32_t n_gpu;
    uint32_t* status;
} PlowXAuditArgs;

typedef struct {
    uint64_t base;
    uint64_t slot_stride;
    uint32_t words;
    uint32_t _pad;
} PlowStateClearRange;

typedef struct {
    const PlowStateClearRange* ranges;
    uint32_t n_ranges;
    uint32_t slot;
} PlowStateClearArgs;

typedef struct {
    const uint32_t* ids;
    uint32_t* ring;
    uint32_t step;
    uint32_t quantum;
    uint32_t batch;
} PlowTokenCaptureArgs;

#if !defined(PLOW_BUCKET_FLASH)
/* PLOW_DOP_XREDUCE_ADD_NORM (116): fused TP all-reduce + residual + RMSNorm, one
 * workgroup. Same rendezvous as d_xreduce_mega; then d_add_norm's math with the `b`
 * load replaced by the peer reduction, ROUNDED TO BF16 FIRST so the fold is
 * bit-identical to the [XReduce -> AddNorm] pair it replaces (dev_isa.h op 116).
 * Marker symbol so a future loader check can refuse op-116 blobs on old objects. */
extern "C" __device__ unsigned plow_xrn_1 = 1;
#if PLOW_XR_AGG_ON
extern "C" __device__ unsigned plow_xr_agg_1 = 1;
#endif

__device__ __forceinline__ void d_xreduce_add_norm_mega(
    bf16* out, bf16* resid, const bf16* a, const bf16* gamma,
    const void* const* peer_scratch, uint32_t nranks, uint32_t rank, uint32_t feat,
    uint32_t slot_bytes, size_t xctr_byte_off, uint32_t gate_id, uint64_t deadline_ticks,
    uint32_t* status, float eps) {
    __shared__ float xrn_part[PLOW_WAVES];
    __shared__ int xrn_bailed;
    if (threadIdx.x == 0) xrn_bailed = 0;
    __syncthreads();
    /* Single-workgroup packet: this WG both announces and waits (d_xreduce_mega's
     * slice==0 role). */
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < nranks; r++) {
            uint32_t* base = (uint32_t*)((char*)peer_scratch[r] + xctr_byte_off);
            xctr_signal(PLOW_CTR(base, gate_id));
        }
        uint32_t* lg = PLOW_CTR((uint32_t*)((char*)peer_scratch[rank] + xctr_byte_off), gate_id);
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        while (xctr_poll(lg) < nranks) {
            if (__builtin_amdgcn_s_memrealtime() - t0 > deadline_ticks) {
                if (status) *status = 0xDEAD0000u | rank;
                xrn_bailed = 1;
                break;
            }
            __builtin_amdgcn_s_sleep(2);
        }
        if (!xrn_bailed) xctr_acquire();
    }
    __syncthreads();
    if (xrn_bailed) return;

    /* d_add_norm's `fits` element->thread map, verbatim — element i = (tid + c*512)*8 + j,
     * c-major, j inner — so the per-thread f32 summation ORDER (and therefore every rounded
     * bit of `inv`) matches the AddNorm this fold replaces. XRN_VEC=2 covers feat <= 8192
     * (enforced at emit, with feat % 8 == 0); GLM h = 6144. Peer loads are 16 B vectors,
     * nranks * XRN_VEC independent issues per thread, so the reduce phase has the MLP the
     * scalar form lacked. */
    constexpr unsigned XRN_VEC = 2;
    const auto* ag = as_glob(a);
    const auto* gg = as_glob(gamma);
    auto* og = as_glob(out);
    auto* rg = as_glob(resid);
    bf16v8 av[XRN_VEC], w[XRN_VEC];
    float accv[XRN_VEC][8];
#pragma unroll
    for (unsigned c = 0; c < XRN_VEC; c++) {
        const unsigned i = (threadIdx.x + c * PLOW_THREADS) * 8;
        av[c] = bf16v8_zero();
        w[c] = bf16v8_zero();
        if (i < feat) {
            av[c] = ld_glob8(ag + i);
            if (gamma) w[c] = ld_glob8(gg + i);
        }
#pragma unroll
        for (int j = 0; j < 8; j++) accv[c][j] = 0.0f;
    }
    for (uint32_t rr = 0; rr < nranks; rr++) {
        const bf16* p = (const bf16*)((const char*)peer_scratch[rr] + slot_bytes);
#pragma unroll
        for (unsigned c = 0; c < XRN_VEC; c++) {
            const unsigned i = (threadIdx.x + c * PLOW_THREADS) * 8;
            if (i < feat) {
                const bf16v8 pv = ld_glob8(as_glob(p) + i);
#pragma unroll
                for (int j = 0; j < 8; j++) accv[c][j] += bf2f(pv[j]);
            }
        }
    }
    float r[XRN_VEC * 8];
    float ss = 0.0f;
#pragma unroll
    for (unsigned c = 0; c < XRN_VEC; c++)
#pragma unroll
        for (int j = 0; j < 8; j++) {
            /* bf2f(f2bf(acc)): the round the standalone XREDUCE's store would have applied. */
            const float f = bf2f(av[c][j]) + bf2f(f2bf(accv[c][j]));
            r[c * 8 + j] = f;
            ss += f * f;
        }
    const float inv = rsqrtf(block_sum(ss, xrn_part) / (float)feat + eps);
#pragma unroll
    for (unsigned c = 0; c < XRN_VEC; c++) {
        const unsigned i = (threadIdx.x + c * PLOW_THREADS) * 8;
        if (i < feat) {
            bf16v8 ro, no;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float g = gamma ? bf2f(w[c][j]) : 1.0f;
                const float f = r[c * 8 + j];
                ro[j] = f2bf(f);
                no[j] = f2bf(f * inv * g);
            }
            st_glob8(rg + i, ro);
            st_glob8(og + i, no);
        }
    }
}

#endif

/* The KV-ENCODING axis, made observable the same way and for a STRICTER reason.  [MLA-FP8-KV]
 *
 * PLOW_FP8_KV is a SWAP, not an addition (`op_attention.h`'s axis table says so, and the
 * FLASH_DECODE / FLASH_MLA_DECODE arms below are `#if`/`#else`). So this axis is silent in BOTH
 * directions, where PLOW_K3 is silent in only one:
 *   - fp8 packet, bf16 object  -> no FLASH_*_FP8 arm      -> silent NOP
 *   - bf16 packet, fp8 object  -> no bf16 FLASH_* arm     -> silent NOP
 * MEASURED, not hypothesised: running the K3 MLA gate's bf16 packet against the fp8 object
 * reports "all packets executed on every slice: YES" and scores rel 1.000e+00 at the attention
 * output, i.e. a completely untouched Opart read as a result.
 *
 * The manifest pairs blob and object by SELECTING the right one (`fp8_kv -> PLOW_FP8_KV=1` in
 * `crates/devgen/src/manifest.rs`); this is the half that REFUSES a wrong pair that got selected
 * anyway. `crates/plowrt/src/exec/amd.rs` `check_kv_encoding` reads it out of `.symtab`. */
#if PLOW_FP8_KV
extern "C" __device__ unsigned plow_fp8_kv_1 = 1;
#endif

/* The Gemma-4 MoE arms, made observable the same way and for the same reason. [GEMMA4-MOE-AMD]
 *
 * These are ADDITIVE axes (a Gemma packet needs the arm; nothing needs the arm ABSENT), so one
 * direction of refusal is enough: a packet carrying opcodes 61-77 / 81-82 against an object
 * built without them would fall to the dispatch `default:`, which on AMD WRITES NOTHING. That
 * is indistinguishable from an op that ran and produced zeros — the exact failure mode
 * `plow_k3_arms_1` exists to make impossible. The symbol exists iff the axis is on, in one
 * preprocessor pass, so absence is unambiguous rather than "old". */
#if PLOW_MOE_GEMMA
extern "C" __device__ unsigned plow_moe_gemma_arms_1 = 1;
#endif
#if PLOW_MOE_GEMMA_PF
extern "C" __device__ unsigned plow_moe_gemma_pf_arms_1 = 1;
#endif
/* Same contract, for the V2 MLA-prefill arm in the FLASH object (d_flash_mla_prefill_v2).
 * The host routes FlashMlaPrefill segments to the flash object only when PLOW_MLA_PF_V2=1;
 * if the loaded flash object predates this arm, the dispatch below would skip those packets
 * silently — the marker lets exec/amd.rs refuse instead. */
#if PLOW_MLA_PF_V2_ARM
extern "C" __device__ unsigned plow_mla_pf_v2_arm_1 = 1;
#if PLOW_FP8_KV
extern "C" __device__ unsigned plow_mla_pf_v2_fp8_arm_1 = 1;
#endif

#if defined(PLOW_BUCKET_MLA_V2_RAW)
#if !PLOW_MLA_PF_V2_ARM || !PLOW_MLA_PF_SV || PLOW_FP8_KV || \
    PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS
#error "raw MLA V2 object requires bf16 V2+SV and cannot carry packed-prefill consumers"
#endif
/* Host routing requires this exact marker. `plow_mla_pf_v2_arm_1` alone is also
 * present in the packed and general flash objects and is therefore insufficient. */
extern "C" __device__ unsigned plow_mla_pf_v2_sv_raw_1 = 1;
#endif
#endif
/* Same contract as the arms markers above, for L2-DOMAIN DISPATCH. A blob compiled with
 * PLOW_L2_PLACE carries per-XCD queue windows, so an object built WITHOUT this axis would
 * mis-dispatch it silently. The runtime used to guard that with an env var the operator ASSERTED
 * (PLOW_L2_PLACE_DISPATCH=1); this symbol lets it CHECK instead. Exists iff the axis is on. */
#if PLOW_L2_PLACE_DISPATCH
extern "C" __device__ unsigned plow_l2_place_dispatch_1 = 1;
#endif

/* Cross-GPU collective spin deadline (s_memrealtime ticks). s_memrealtime is 1 GHz on
 * gfx950, so ~1e9 ticks ≈ 1 s: a fabric on which the
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
 *                                     dense GQA (38/39) AND MLA latent (109/110)
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
 * intentional Q-hoist. Additive on the weight axes costs nothing (see the arms).
 *
 * AND A SWAP IS SILENT IN BOTH DIRECTIONS, which is why `plow_fp8_kv_1` exists (above). An
 * additive axis only has to refuse "packet needs the arm, object lacks it"; a swap ALSO has to
 * refuse the reverse, because each object is missing an arm the other has. Measured on the K3
 * MLA gate: a bf16 packet on the fp8 object reports every packet executed and scores rel
 * 1.000e+00 at the attention output. */
#ifndef PLOW_FP8
#define PLOW_FP8 0
#endif

/* GLOBAL-QUEUE scheduling (Experiment E1). When 1, every workgroup pulls the next packet from ONE
 * shared atomic cursor over the op-major gq_stream instead of walking a static per-CU stream; the op
 * kernels/tiles/register configs are byte-identical, so this is a pure scheduling change. Built as
 * separate _gq objects and selected at launch by the host (PLOW_GLOBAL_QUEUE=1). Measured (batch 1):
 * consistent decode win on both models (dynamic claim smooths static per-CU imbalance), prefill cost
 * model-size-dependent (small on 4B). See the design notes.
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
#ifndef PLOW_GATE_HIER
#define PLOW_GATE_HIER 0
#endif
#if PLOW_GATE_HIER &&                                                                    \
    (!defined(PLOW_BUCKET_DECODE) || !PLOW_BUCKET_DECODE || !PLOW_GLOBAL_QUEUE ||        \
     !defined(PLOW_L2_PLACE_DISPATCH) || !PLOW_L2_PLACE_DISPATCH)
#error "PLOW_GATE_HIER requires a decode global-queue object with L2-domain dispatch"
#endif
#if PLOW_GATE_HIER
extern "C" __device__ unsigned plow_gate_hier_1 = 1;
#endif
#ifndef PLOW_GQ_BATCH
#define PLOW_GQ_BATCH 1
#endif
/* Hoist the instruction fetch above the gate poll (see the probe at the poll). Default OFF. */
#ifndef PLOW_INST_PF
#define PLOW_INST_PF 0
#endif

/* PLOW_DSA_IDX_ROW — the row-resident DSA prefill indexer (op 117) + the fewer-passes exit on
 * op 118. DEFAULT ON: both are bit-identical to the shipped arms and strictly cheaper, and the
 * whole sparse-prefill path they serve is itself default-OFF, so no shipped program moves.
 * Build with -DPLOW_DSA_IDX_ROW=0 to restore the shipped arms for an A/B.
 * IDXPF_TILE_N_DEF is the LDS slab width; 64 keys keeps three workgroups resident per CU. */
#ifndef PLOW_DSA_IDX_ROW
#define PLOW_DSA_IDX_ROW 1
#endif
#ifndef IDXPF_TILE_N_DEF
#define IDXPF_TILE_N_DEF 64u
#endif
#if PLOW_DSA_IDX_ROW
/* Marker so "which arm is in this object?" is answerable by a symbol scan rather than by
 * trusting a build flag. This carries NO `requires` entry (see the dispatch site: the arm is
 * bit-identical and changes no packet operand) — it exists because an entire agent-round of this
 * campaign was spent measuring an ablation in an object the run never opened. */
extern "C" __device__ unsigned plow_dsa_idx_row_arm = 1;
#endif

/* PLOW_TRACE_PHASE -- MEASUREMENT INSTRUMENT. Splits the packet into its FIVE protocol phases in
 * ONE traced run, so the per-packet cost can be DECOMPOSED instead of attributed by hand.
 *
 * The shipped record already carries three timestamps, and they bracket the wrong things for this
 * question: `t_arrive` is taken AFTER the claim (so the claim is invisible) and `t_end - t_ready`
 * lumps the op body together with the exit rendezvous and the successor release. This knob adds
 * two more probes and packs their deltas into `pc`, which the census reducers already discard
 * ("pc is the per-workgroup stream slot" -- scripts/glm52_decode_census.py).
 *
 *     T0  top of the iteration, BEFORE the claim barrier            -> r.t_arrive
 *     T1  after the gate poll and its __syncthreads(), before the acquire
 *     T2  after the acquire/XCD rendezvous and its __syncthreads()  -> r.t_ready
 *     T3  after the op body and its __syncthreads()
 *     T4  after the successor release                               -> r.t_end
 *
 *     r.pc = min(T1-T0, 0xffff) | (min(T3-T2, 0xffff) << 16)     [10 ns ticks, 655 us range]
 *
 * so a reducer recovers all four phases from one record:
 *     claim + gate wait = pc & 0xffff
 *     acquire + rendezvous = (t_ready - t_arrive) - (pc & 0xffff)
 *     op body = pc >> 16
 *     publish + release = (t_end - t_ready) - (pc >> 16)
 *
 * COSTS TWO EXTRA `s_memrealtime` PER (workgroup, packet), and they are inside `if (prog.trace)`
 * exactly like the three that already exist -- so an untraced run is unaffected, and a traced run
 * is perturbed by the same instrument in every arm. With the knob OFF the preprocessed source is
 * byte-identical to before it existed. Default OFF; never ship it on. */
#ifndef PLOW_TRACE_PHASE
#define PLOW_TRACE_PHASE 0
#endif

/* PLOW_DEC_ARENA_HALVES decouples the DECODE arena from the GEMM tile.
 *
 * The arena is normally the prefill GEMM stage, and that is what pins the decode object's LDS:
 * the smallest tile the fused-GLU SN==2 assert allows at 8 waves is 128x256x32 = 30,720 B, just
 * over the 32,768 B that would fit two workgroups per CU and well over the 16,384 B that would
 * fit four. But the decode object DEAD-STRIPS the whole GEMM family -- interp_decode_gq.elf
 * contains ZERO d_gemm* symbols -- so the tile only survives as a TEMPLATE INSTANTIATION whose
 * static_asserts fire. Sizing the arena separately keeps those asserts satisfied (the tile
 * constants are untouched) while giving the GEMV the LDS it actually needs.
 *
 * The staged-x arm is checked PER OP (`M*K <= GM_LDS_HALVES`), so a smaller arena does not break
 * correctness anywhere -- a projection whose activation no longer fits simply reads x from
 * global, which is the same fallback the unstaged arm has always been.
 *
 * NOT SAFE FOR BATCHED DECODE. At PLOW_DECODE_BATCH > 16 the decode program dispatches real
 * GEMM ops (M exceeds PLOW_GEMV_MAXM), and those would address an arena sized for the GEMV.
 * Guarded on PLOW_GEMV_MM below; a packet that violates it needs the same marker-symbol refusal
 * `check_moe_gemma_arms` uses. */
#if defined(PLOW_BUCKET_MLA_V2_RAW)
/* The raw MLA object dispatches no GEMM. Keeping the production GEMM tile in its union
 * wastes 89 KiB of LDS and obscures the object's actual V2 resource contract. */
#define PLOW_GM_ARENA 1
#elif PLOW_BUCKET_DECODE && defined(PLOW_DEC_ARENA_HALVES)
#if PLOW_GEMV_MM > 16
#error "PLOW_DEC_ARENA_HALVES is unsafe at PLOW_GEMV_MM > 16: batched decode dispatches GEMM"
#endif
#if PLOW_DEC_ARENA_HALVES < GM_LDS_HALVES
/* The staged-x arm and the NRN-fold scratch (GM_LDS_HALVES - GV_NORM_SCRATCH) both measure
 * against the TILE constant, not this arena — an arena below it passes every per-op check and
 * overflows the union. The knob's win is dropping the (BM+BN) GEMM stage, not going below the
 * GEMV budget. */
#error "PLOW_DEC_ARENA_HALVES below GM_LDS_HALVES: gemv staging checks would write past the arena"
#endif
#define PLOW_GM_ARENA (PLOW_DEC_ARENA_HALVES)
#else
#define PLOW_GM_ARENA GM_LDS_HALVES_T(GM_BM, GM_BN, GM_BK)
#endif
#if PLOW_BUCKET_DECODE
/* The decode bucket never runs flash PREFILL (its arm is not compiled), so its arena is the
 * GEMM/GEMV staging tile alone. See the `fa` member below. */
#define PLOW_SMEM_H0 PLOW_GM_ARENA
#else
#define PLOW_SMEM_H0                                                            \
    (PLOW_GM_ARENA > FA_LDS_HALVES(512) ? PLOW_GM_ARENA : FA_LDS_HALVES(512))
#endif
/* `raw` must cover the LARGEST member, not just the two biggest-on-gfx950 ones. On a 64 KiB
 * part the GEMM arena shrinks below the MLA-prefill tiles and the old two-way max left `raw`
 * smaller than `fmpf` -- which the static_assert below caught, but as a build failure rather
 * than the size it should have been. No effect on gfx950, where the GEMM arena still wins. */
#if PLOW_MLA_PREFILL
#define PLOW_MLA_PF_HALVES (FA_MLA_PF_LDS_FLOATS(512, 64) * 2)
#define PLOW_SMEM_HALVES                                                        \
    (PLOW_SMEM_H0 > PLOW_MLA_PF_HALVES ? PLOW_SMEM_H0 : PLOW_MLA_PF_HALVES)
#else
#define PLOW_SMEM_HALVES PLOW_SMEM_H0
#endif

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
/* Arm marker for the dense causal KV-split of the V2 MLA prefill (packet i6 on op 51,
 * manifest `requires` PLOW_MLA_PF_NS). UNCONDITIONAL in every object of this build: the split
 * is honored by the V2 flash dispatch, and the host refuses ns packets unless V2 routing is
 * live, so a non-flash object carrying the marker can never execute one. An OLDER build has
 * no marker and is refused at load — its V2 arm would write nsplit=1 partials while the
 * merge reads ns of them. */
extern "C" __device__ unsigned plow_mla_pf_ns_arm = 1;

/* Arm marker for the T11 GLU-into-quant fold (op QUANT_FP8, t3=gate t4=up i2=act). UNCONDITIONAL
 * in every object of this build. The emitter deletes the `Glu` packet when it folds, so an object
 * whose QUANT_FP8 arm ignores t3/t4 quantizes an `fu` that was never produced — a wrong FFN
 * output, a wrong KV cache, and fluent wrong tokens with nothing to see at runtime. Every gfx942
 * object before this marker is exactly that object; the packet's `requires` makes plowrt refuse
 * the pairing instead of serving garbage. */
extern "C" __device__ unsigned plow_t11_gluquant_arm = 1;

typedef union __align__(16) {
    float part[PLOW_WAVES];           /* block_sum scratch          */
    bf16  gm[PLOW_GM_ARENA];          /* GEMM A|B tiles, x2 buffers */
#if !PLOW_BUCKET_DECODE
    /* FLASH-PREFILL TILES, AND THEY ARE DEAD IN THE DECODE BUCKET. Verified on the shipped
     * object: interp_decode_gq.elf contains ZERO d_flash_prefill symbols and three
     * d_flash_decode ones -- the decode bucket already compiles the prefill arm out. Leaving the
     * MEMBER in still sized the union at 58,368 B, which is what pinned the decode object to one
     * workgroup per CU (2 waves/SIMD) no matter what else shrank. */
    bf16  fa[FA_LDS_HALVES(512)];     /* flash K|V|P tiles          */
#endif
    float fd[FA_DEC_LDS_FLOATS(512, PLOW_FA_GF(512))]; /* flash decode; the op owns this size */
/* PROBE GATE. `fmla` is the MLA latent-decode arena (GLM/DeepSeek/Kimi) and it sets the decode
 * union at 42,064 B once the flash-prefill `fa` is gated out -- above the 32,768 B that would
 * fit TWO workgroups per CU. A Gemma decode object never runs an MLA op. Gating it is NOT safe
 * to ship as-is: plowrt has no MLA arm-check for decode the way it has `check_moe_gemma_arms`,
 * so a GLM packet handed a Gemma-built object would silently skip its MLA ops. This exists to
 * MEASURE whether crossing the occupancy threshold pays at all, before that check is built. */
#ifndef PLOW_NO_MLA_DEC
#define PLOW_NO_MLA_DEC 0
#endif
/* The op body is compiled either way (the switch is total); under the probe gate it is aimed at
 * `raw`, which is at least as large. It never runs -- a Gemma decode packet emits no MLA op. */
#if PLOW_NO_MLA_DEC
#define PLOW_FMLA(sm) ((float*)(sm)->raw)
#else
#define PLOW_FMLA(sm) ((sm)->fmla)
#endif
#if !PLOW_NO_MLA_DEC
    float fmla[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF_MAX)]; /* MLA latent decode (DK=512,DR=64) */
#endif
#if PLOW_MLA_PREFILL
    /* MLA latent PREFILL tiles: Q|K staged plus the P transpose. Bigger than `fmla` (the
     * query tile is hoisted here) and still under the GEMM arena that sets the union, so
     * it costs no LDS. Its own member because the op owns the layout, not the decode's. */
    float fmpf[FA_MLA_PF_LDS_FLOATS(512, 64)];
#endif
    unsigned long long am[PLOW_WAVES];  /* argmax block reduction   */
    bf16  raw[PLOW_SMEM_HALVES];
} plow_smem;

#if PLOW_MLA_PREFILL
/* `raw` is what the launcher sizes the dynamic arena from, so a member that outgrows it
 * would read past the allocation rather than fail to compile. It fits today (128,512 B
 * against the GEMM's 147,456) and this is what keeps that true if either side moves. */
static_assert(sizeof(((plow_smem*)nullptr)->fmpf) <= sizeof(((plow_smem*)nullptr)->raw),
              "MLA prefill LDS arena exceeds the interpreter's shared allocation");
#endif

#if PLOW_MOE_GEMMA_PF
/* Same guard, for the grouped Gemma expert GEMM: it stages its tiles into `raw` rather than
 * declaring its own __shared__, so an arena smaller than the tile would read past the
 * allocation instead of failing to compile. GMPF_DBUF already drops to ONE buffer on a 64 KiB
 * part (op_moe.h); this is what keeps that true if either side moves. */
static_assert(GMPF_LDS_HALVES * sizeof(bf16) <= sizeof(((plow_smem*)nullptr)->raw),
              "Gemma grouped-MoE prefill LDS tile exceeds the interpreter's shared allocation");

/* The row-resident indexer stages its key slab in `raw`. Sizing it by hand is exactly how the
 * MoE-PF prefill shipped a live LDS overrun (81,920 B addressed against a 64,512 B union, every
 * gfx942 GLM prefill running on corrupted math while the serve gate passed), so the bound is
 * asserted here rather than reasoned about at the call site. */
#if PLOW_DSA_IDX_ROW
static_assert(IDXPF_TILE_N_DEF * (128 + FA_PAD) * sizeof(bf16)
                  <= sizeof(((plow_smem*)nullptr)->raw),
              "IDXPF_TILE_N_DEF key slab does not fit the interpreter LDS arena — lower it");
#endif
#endif

#if PLOW_NO_MLA_DEC
/* THE SAME GUARD FOR THE MLA LATENT DECODE ARENA UNDER THE PROBE GATE, and it was the one
 * member of the union that did not have one.
 *
 * With PLOW_NO_MLA_DEC=1 the `fmla` member is compiled OUT and `PLOW_FMLA(sm)` hands the op
 * `sm->raw` instead. Nothing then relates the arena the kernel lays out to the arena it was
 * given: `d_flash_mla_decode` receives a bare `float*` and writes GF*(FA_DEC_TILE + DK/2 +
 * DR/2) + 2*PLOW_WAVES + FA_DEC_NG(DK)*DK floats into it, whatever `raw` happens to be.
 *
 * On gfx942 under PLOW_OCC4 that arena is the 128x256x32 GEMM tile at GM_DBUF=1, i.e.
 * (128+256)*(32+8)*2 = 30,720 B, and the MLA decode layout at DK=512/DR=64/PLOW_WAVES=8 is
 *
 *     GF=4   800*4 + 4112 =  7,312 floats = 29,248 B   fits, 1,472 B to spare
 *     GF=8   800*8 + 4112 = 10,512 floats = 42,048 B   ELEVEN KILOBYTES PAST THE END
 *
 * so `PLOW_OCC4=1 PLOW_GLM_GF8=1` (which is -DPLOW_NO_MLA_DEC=1 -DPLOW_GLM_GF8_ARM=1, see
 * scripts/build_gfx942.sh) writes 11,328 B past a 30,720 B arena — past the 64 KiB physical
 * LDS as well. Nothing in the tree refused that composition; it is not in the canonical recipe,
 * which is the only reason it has never bitten. Same class as the MoE-PF LDS OOB above, which
 * silently corrupted every gfx942 MoE prefill for a whole campaign before 849fc33.
 *
 * REFUSED RATHER THAN MADE SAFE. Making it safe means growing `raw` to 42,048 B, and the whole
 * point of PLOW_OCC4 is an LDS budget under 32,768 B so two workgroups fit per CU (measured
 * -19.1% on fp8). A 42 KB arena puts the object back at one workgroup per CU, i.e. it buys
 * correctness by deleting the win. A GF=8 A/B under OCC4 has to move the arena deliberately
 * (PLOW_DEC_ARENA_HALVES) and say so, not inherit it.
 *
 * The bound is the LARGEST GF `exec_flash_mla_decode` instantiates, not GLM_MLA_GF_MAX: the
 * GF=8 arm only exists under PLOW_GLM_GF8_ARM, and asserting against 8 unconditionally would
 * refuse the shipped OCC4 object, which really does fit. */
static_assert(FA_MLA_DEC_LDS_FLOATS(512, 64, (PLOW_GLM_GF8_ARM ? 8 : 4)) * sizeof(float) <=
                  sizeof(((plow_smem*)nullptr)->raw),
              "PLOW_NO_MLA_DEC aims the MLA latent-decode arena at `raw` and the largest "
              "instantiated GF does not fit it (PLOW_OCC4 + PLOW_GLM_GF8 is 11 KB over). "
              "Widen the arena (PLOW_DEC_ARENA_HALVES) or drop one of the two flags.");
#endif

#if PLOW_MOE_PREFILL
/* Same guard for the GLM/Kimi/DeepSeek grouped expert GEMM (ops 85/86). This assert was
 * MISSING, and its absence shipped a gfx942 object whose double-buffered MPF stage addressed
 * 81,920 B against a 64,512 B arena — 16 KiB of it past the 64 KiB physical LDS, where a
 * hardware probe shows accesses do not round-trip. MPF_DBUF now drops to one buffer on a
 * 64 KiB part exactly as GMPF_DBUF does; this pin is what keeps that true. */
static_assert(MPF_LDS_HALVES * sizeof(bf16) <= sizeof(((plow_smem*)nullptr)->raw),
              "grouped-MoE prefill LDS tiles exceed the interpreter's shared allocation");
#if PLOW_MOE_PF_A4W4
/* Same pin for the mxfp4 grouped arm (both the CDNA4 A4W4 plan and the CDNA3 simulated one --
 * MPF4_LDS_BYTES is arch-conditional). It had no assert because only 147 KiB arenas ever built
 * it; the CDNA3 arm runs against a 64 KiB part, which is exactly where the MoE-PF class of
 * silent LDS overrun lived. */
static_assert(MPF4_LDS_BYTES <= sizeof(((plow_smem*)nullptr)->raw),
              "mxfp4 grouped-MoE prefill LDS tiles exceed the interpreter's shared allocation");
#endif
#endif

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
#define IEN(k) (in->i[k] == PLOW_TENSOR_NONE ? nullptr : T[in->i[k]])
    /* i1 bit 16 is the NRF fold (HeadNormRope computed here — see d_flash_decode's NRF arm).
     * The freed integer slots carry the fold's operands, so several fields arrive PACKED:
     *   t2=qg(raw) t6=kg(raw) t7=vg(raw, =kg on k_eq_v) · i0=n_batch|window<<8 ·
     *   i3=kv_stride|nsplit<<20 · i4=gamma_q i5=gamma_k (handles) · fj0=eps (scale is
     *   emitter-asserted 1.0) · fj1.u=cos fj2.u=sin (handles) · i1 bit17=skip_norm.
     * Absence TRAPS — a fold packet missing an operand would attend over garbage. */
    if (in->i[1] & 0x10000u) {
        const unsigned hd = in->i[6] & 0xFFFFu;
        const bf16* kg = (const bf16*)TEN(6);
        const bf16* vg = (const bf16*)TEN(7);
        const bf16* gq = (const bf16*)IEN(4);
        const bf16* gk = (const bf16*)IEN(5);
        const float* cs = (const float*)(in->fj[1].u == PLOW_TENSOR_NONE ? nullptr : T[in->fj[1].u]);
        const float* sn = (const float*)(in->fj[2].u == PLOW_TENSOR_NONE ? nullptr : T[in->fj[2].u]);
        if (!TEN(2) || !kg || !vg || !cs || !sn) __builtin_trap();
        const unsigned n_batch = in->i[0] & 0xFFu, window = in->i[0] >> 8;
        const unsigned n_head = in->i[1] & 0xFFFFu, skip = (in->i[1] >> 17) & 1u;
        const unsigned kv_stride = in->i[3] & 0xFFFFFu, nsplit = in->i[3] >> 20;
        const float eps = in->fj[0].f;
        /* [MERGE-FOLD] operands, packed into the bits the NRF layout leaves free (every whole
         * slot is taken): i1[18..32) = the done-counter tensor handle (14 bits — the emitter
         * asserts the handle fits), i6[16..32) = the bf16 attention-output handle. 0 = no fold
         * (both are compiler-owned tensors, never handle 0). See d_flash_decode [MERGE-FOLD]. */
        const unsigned mrg_h = in->i[1] >> 18, at_h = in->i[6] >> 16;
        unsigned* mrgc = mrg_h ? (unsigned*)T[mrg_h] : nullptr;
        bf16* o_fin = at_h ? (bf16*)T[at_h] : nullptr;
        if (mrg_h && (!mrgc || !o_fin)) __builtin_trap();
#define PLOW_FD_NRF(DD)                                                                            \
    d_flash_decode<DD, PLOW_FA_GF(DD), false, true>(                                               \
        (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),                  \
        (const bf16*)TEN(4), (const int*)TEN(5), n_batch, n_head, in->i[2], kv_stride, window,     \
        1.0f, nsplit, in->i[7], slice, nblk, lds, nullptr, nullptr, kg, vg, gq, gk, cs, sn, eps,   \
        skip, mrgc, o_fin)
        if (hd == 128) PLOW_FD_NRF(128);
        else if (hd == 256) PLOW_FD_NRF(256);
        else if (hd == 512) PLOW_FD_NRF(512);
#undef PLOW_FD_NRF
#undef IEN
        return;
    }
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
#if PLOW_MLA_FOLD_TB > 1
static_assert(PLOW_MLA_FOLD_TB_LDS * sizeof(float) <= sizeof(((plow_smem*)nullptr)->raw),
              "PLOW_MLA_FOLD_TB arena over `raw`");
#endif
__device__ __forceinline__ void exec_mla_merge_fold(CInst in, CTens T, unsigned slice,
                                                    unsigned nblk, float* raw) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    const unsigned bh = in->i[0] * in->i[1]; /* n_batch * n_head_local */
    const unsigned V = in->i[2];             /* v_head_dim: GLM-5.2 256, Kimi-K3 128 */
#if PLOW_MLA_FOLD_TB > 1
    /* TOKEN-BLOCKED PREFILL ARM (-DPLOW_MLA_FOLD_TB=<G>; op_attention.h d_mla_merge_fold_tb).
     *
     * The V>=256 arm below gives one workgroup one (token, head) row and streams the whole
     * 256 KiB W_uv[h] panel for it, so at GLM-5.2 TP8 T=8192 the packet pulls 16.8 GB out of L2
     * per layer to do 8.6 GMAC. This arm gives it TB rows of ONE head instead and divides that
     * stream by TB. Measured standalone at that shape (runtime/bench/amd/glm52_kbench_fold_pf):
     * 1626 -> 616 us/packet at TB=8, output BIT-IDENTICAL.
     *
     * EVERY guard here is a correctness precondition of the TB body, not a heuristic:
     *   !(bh*8 <= nblk)   leave the VT=32 decode map exactly where it is — a different VT is a
     *                     different fold map and therefore different arithmetic.
     *   V >= 256, V%VT==0 the TB body carries only the fast map; a ragged last V-tile has no
     *                     scalar arm to fall into.
     *   V % VEC == 0      the vector load's own precondition, as in the scalar body.
     *   n_batch % TB == 0 whole token groups; the body has no tail.
     *   nsplit < 8        the scalar body's MS=8 block is then unreachable in BOTH, so both run
     *                     the select-then-add tail and agree bit for bit. See the TB merge note.
     *   n_work >= nblk    do not trade chip fill for locality on a small bucket: at t=128 the
     *                     TB=8 grouping would leave 128 of 304 workgroups with no work.
     * The arena is TB*DK + RB*PLOW_WAVES*VT floats — asserted against `raw` at the call site. */
    {
        constexpr unsigned TB = PLOW_MLA_FOLD_TB, VT = 256;
        if (!(bh && bh * 8u <= nblk) && V >= VT && (V % VT) == 0u &&
            (V % (unsigned)PLOW_MLA_FOLD_VEC) == 0u && in->i[0] && in->i[0] % TB == 0u &&
            in->i[4] < 8u && (in->i[0] / TB) * in->i[1] * (V / VT) >= nblk) {
            d_mla_merge_fold_tb<512, (int)VT, (int)TB>(
                (bf16*)TEN(0), (const float*)TEN(1), (const float*)TEN(2), (const bf16*)TEN(3),
                in->i[0], in->i[1], in->i[2], in->i[4], slice, nblk, raw);
            return;
        }
    }
#endif
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
         * perf-data/archive/k3/kimi-k3-kernel-gap.md §8b as item 11.
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
/* Marker: this object's `d_flash_mla_decode` understands the q-rope fold (op 50 t7 = cos, i6 =
 * sin handle, t3 = the RAW q_rope projection). UNCONDITIONAL, because the fold is a runtime
 * branch and not a build axis — presence of the symbol is exactly "built at or after the fold".
 * plowrt's decode-object check refuses a folded blob when it is missing: a pre-fold object stages
 * t3 verbatim, so the flash would score an UNROPED query and the model would answer fluently and
 * wrongly (no trap, no NaN, nothing to see at runtime). */
extern "C" __device__ unsigned plow_glm_fuse_rope_arm = 1;

/* Q-NORM FOLD ARM MARKER (build axis PLOW_GLM_FUSE_QNORM; op_gemm.h carries the note).
 *
 * CONDITIONAL, unlike the rope marker above, because this fold IS a build axis: the branch is
 * compiled only under the define, so an object without it does not merely predate the fold, it
 * cannot execute it. And the failure it guards is the same silent class -- a pre-fold object
 * handed a folded packet ignores `t[7]`, stages the RAW q_a projection and runs the GEMV over an
 * UNNORMED row. No trap, no NaN, a fluent wrong answer. */
#if PLOW_GLM_FUSE_QNORM
extern "C" __device__ unsigned plow_glm_fuse_qnorm_arm = 1;
#endif

/* MLA latent flash decode (DeepSeek). DK=512, DR=64, GF=8 — the validated D=512 decode budget.
 * The dense and GATHER (sparse top-k / DSA) instantiations share this wrapper; GATHER reads only
 * the top_k selected latent rows via the idx table (t7). Both emit latent-wide (O,m,l) partials
 * for FLASH_MERGE<512> + O_UV_FOLD. See d_flash_mla_decode in op_attention.h. [DEEPSEEK-MLA] */
__device__ __forceinline__ void exec_flash_mla_decode(CInst in, CTens T, unsigned slice,
                                                      unsigned nblk, float* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
#define IEN(k) (in->i[k] == PLOW_TENSOR_NONE ? nullptr : T[in->i[k]])
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
        /* [HNR-FOLD] Q-ROPE FOLD, dense bf16 arm only (emit knob PLOW_GLM_FUSE_ROPE).
         *
         * t7 = cos table, i6 = sin table AS A DEMOTED HANDLE (the rule `DevOp::GemvQkvg`
         * established and `GemvQkvMxfp4` reuses for its three E8M0 scale rows: an i-slot may
         * carry a tensor handle when the t-slots are full, resolved through `IEN`). Both are
         * read-only generated tables, which is the class the demotion rule allows.
         *
         * PRESENCE OF t7 IS THE DISCRIMINATOR and it is unambiguous *on this opcode*: the
         * GATHER twin (op 54) and the FP8 twin (op 109) are DIFFERENT OPCODES that dispatch
         * elsewhere, and on op 50 t7 has never been anything but PLOW_TENSOR_NONE. i6 cannot
         * be the discriminator instead — 0 is a legal tensor handle. The emitter writes both
         * slots together or neither; a fold packet that reached here with only t7 would rope
         * against tensor 0, which is why the trap below refuses it.
         *
         * NOT wired into the GATHER/FP8 arms and never can be as slotted: both spend t7. The
         * emitter asserts the same exclusion so a blob cannot express the combination. */
        const float* qrc = (const float*)TEN(7);
        const float* qrs = qrc ? (const float*)IEN(6) : nullptr;
        if (qrc && !qrs) __builtin_trap();
        if (gf == 2)
            d_flash_mla_decode<512, 64, 2>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[1],
                in->i[2], in->i[3], in->fj[0].f, in->i[4], in->i[5], slice, nblk, lds, nullptr, 0,
                1, nullptr, 0, qrc, qrs);
#if PLOW_GLM_GF8_ARM
        else if (gf == 8)
            d_flash_mla_decode<512, 64, 8>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[1],
                in->i[2], in->i[3], in->fj[0].f, in->i[4], in->i[5], slice, nblk, lds, nullptr, 0,
                1, nullptr, 0, qrc, qrs);
#endif
        else
            d_flash_mla_decode<512, 64, 4>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[1],
                in->i[2], in->i[3], in->fj[0].f, in->i[4], in->i[5], slice, nblk, lds, nullptr, 0,
                1, nullptr, 0, qrc, qrs);
    }
#undef IEN
#undef TEN
}

#if PLOW_FP8_KV
/* FP8 (e4m3) LATENT MLA decode (op 109). [MLA-FP8-KV]
 *
 * The MLA twin of `exec_flash_decode_fp8`, and compiled ONLY into the fp8-KV object — the KV
 * encoding is a SWAP axis, not an additive one (see the dispatch note below).
 *
 * DENSE ONLY, and that is a constraint rather than an omission: the gathered twin needs `t7` for
 * its `idx` table and `t7` is where the scale array lives, so a gathered fp8 latent has nowhere to
 * put the dequant. `PLOW_DOP_FLASH_GATHER_DECODE` therefore keeps its bf16 arm in BOTH objects
 * (see the dispatch), so a DSA packet cannot land on a missing arm and silently no-op.
 *
 *   t0=Opart t1=mlpart t2=Qabs t3=Qrope t4=Ckv(fp8) t5=Krope t6=kv_len t7=kv_scale
 *   i0=n_batch i1=n_head i2=kv_stride i3=window i4=nsplit i5=kv_mask i6=krot_fp8 i7=gf  f0=scale */
__device__ __forceinline__ void exec_flash_mla_decode_fp8(CInst in, CTens T, unsigned slice,
                                                          unsigned nblk, float* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    /* The scale array is not optional: without it every row dequants through a null pointer.
     * Trap rather than read address 0 — this object has no silent-NOP budget left. */
    if (TEN(7) == nullptr) __builtin_trap();
    const unsigned gf = in->i[7] ? in->i[7] : (unsigned)GLM_MLA_GF;
    if (gf == 2)
        d_flash_mla_decode<512, 64, 2, /*GATHER=*/false, /*FP8=*/true>(
            (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
            (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[1],
            in->i[2], in->i[3], in->fj[0].f, in->i[4], in->i[5], slice, nblk, lds, nullptr, 0, 1,
            (const float*)TEN(7), in->i[6]);
    else
        d_flash_mla_decode<512, 64, 4, /*GATHER=*/false, /*FP8=*/true>(
            (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
            (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[1],
            in->i[2], in->i[3], in->fj[0].f, in->i[4], in->i[5], slice, nblk, lds, nullptr, 0, 1,
            (const float*)TEN(7), in->i[6]);
#undef TEN
}
#endif

#if PLOW_MLA_PREFILL
#if PLOW_FP8_KV
/* FP8 (e4m3) LATENT MLA PREFILL (op 110). Operands as the decode twin, `i4 = n_tok`. */
__device__ __forceinline__ void exec_flash_mla_prefill_fp8(CInst in, CTens T, unsigned slice,
                                                           unsigned nblk, float* lds,
                                                           const PlowProgram* prog) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    if (TEN(7) == nullptr) __builtin_trap();
#if PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS
    if (plow_packed_prefill_enabled(prog)) {
        const unsigned gf = in->i[7] ? in->i[7] : (unsigned)GLM_MLA_GF;
        for (unsigned si = 0; si < prog->n_prefill_spans; ++si) {
            const PlowPrefillSpan* s = plow_packed_prefill_span(prog, si);
            float* op = (float*)TEN(0) + (size_t)s->row0 * in->i[1] * 512u;
            float* ml = (float*)TEN(1) + (size_t)s->row0 * in->i[1] * 2u;
            const bf16* qa = (const bf16*)TEN(2) + (size_t)s->row0 * in->i[1] * 512u;
            const bf16* qr = (const bf16*)TEN(3) + (size_t)s->row0 * in->i[1] * 64u;
            const unsigned char* ck = (const unsigned char*)TEN(4) +
                                      (size_t)s->slot * in->i[2] * 512u;
            const bf16* kr = (const bf16*)TEN(5) + (size_t)s->slot * in->i[2] * 64u;
            const float* ks = (const float*)TEN(7) + (size_t)s->slot * in->i[2];
#if PLOW_MLA_PF_MFMA
            if (mla_pf_tiled_fills(1u, s->n_rows, in->i[1], nblk)) {
                d_flash_mla_prefill_mfma<512, 64, /*FP8=*/true>(
                    op, ml, qa, qr, (const bf16*)ck, kr, (const int*)TEN(6) + s->slot, 1u,
                    s->n_rows, in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice,
                    nblk, lds, ks, in->i[6]);
                continue;
            }
#endif
            if (gf == 2)
                d_flash_mla_prefill_fp8<512, 64, 2>(
                    op, ml, qa, qr, (const bf16*)ck, kr, (const int*)TEN(6) + s->slot, 1u,
                    s->n_rows, in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice,
                    nblk, lds, ks, in->i[6]);
            else
                d_flash_mla_prefill_fp8<512, 64, 4>(
                    op, ml, qa, qr, (const bf16*)ck, kr, (const int*)TEN(6) + s->slot, 1u,
                    s->n_rows, in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice,
                    nblk, lds, ks, in->i[6]);
        }
        return;
    }
#endif
#if PLOW_MLA_PF_MFMA
    /* `gf` (i[7]) is a DECODE knob — how many query heads share one re-stream of the latent.
     * The tiled kernel's reuse axis is the QUERY TILE, so it has no group factor to read and
     * ignores the immediate; the scalar fallback below still honours it. */
    if (mla_pf_tiled_fills(in->i[0], in->i[4], in->i[1], nblk)) {
        d_flash_mla_prefill_mfma<512, 64, /*FP8=*/true>(
            (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
            (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[4],
            in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice, nblk, lds,
            (const float*)TEN(7), in->i[6]);
        return;
    }
#endif
    const unsigned gf = in->i[7] ? in->i[7] : (unsigned)GLM_MLA_GF;
    if (gf == 2)
        d_flash_mla_prefill_fp8<512, 64, 2>(
            (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
            (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[4],
            in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice, nblk, lds,
            (const float*)TEN(7), in->i[6]);
    else
        d_flash_mla_prefill_fp8<512, 64, 4>(
            (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
            (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0], in->i[4],
            in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice, nblk, lds,
            (const float*)TEN(7), in->i[6]);
#undef TEN
}
#endif
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
                                                       unsigned nblk, float* lds,
                                                       const PlowProgram* prog) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    const unsigned gf = in->i[7] ? in->i[7] : (unsigned)GLM_MLA_GF;
#if PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS
    if (plow_packed_prefill_enabled(prog)) {
        if (in->op == PLOW_DOP_FLASH_GATHER_PREFILL || in->t[7] != PLOW_TENSOR_NONE)
            __builtin_trap();
        for (unsigned si = 0; si < prog->n_prefill_spans; ++si) {
            const PlowPrefillSpan* s = plow_packed_prefill_span(prog, si);
            float* op = (float*)TEN(0) + (size_t)s->row0 * in->i[1] * 512u;
            float* ml = (float*)TEN(1) + (size_t)s->row0 * in->i[1] * 2u;
            const bf16* qa = (const bf16*)TEN(2) + (size_t)s->row0 * in->i[1] * 512u;
            const bf16* qr = (const bf16*)TEN(3) + (size_t)s->row0 * in->i[1] * 64u;
            const bf16* ck = (const bf16*)TEN(4) + (size_t)s->slot * in->i[2] * 512u;
            const bf16* kr = (const bf16*)TEN(5) + (size_t)s->slot * in->i[2] * 64u;
#if PLOW_MLA_PF_MFMA
            if (mla_pf_tiled_fills(1u, s->n_rows, in->i[1], nblk)) {
                d_flash_mla_prefill_mfma<512, 64, /*FP8=*/false>(
                    op, ml, qa, qr, ck, kr, (const int*)TEN(6) + s->slot, 1u, s->n_rows,
                    in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice, nblk, lds,
                    nullptr, 0u);
                continue;
            }
#endif
            if (gf == 2)
                d_flash_mla_prefill<512, 64, 2>(
                    op, ml, qa, qr, ck, kr, (const int*)TEN(6) + s->slot, 1u, s->n_rows,
                    in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice, nblk, lds);
            else
                d_flash_mla_prefill<512, 64, 4>(
                    op, ml, qa, qr, ck, kr, (const int*)TEN(6) + s->slot, 1u, s->n_rows,
                    in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice, nblk, lds);
        }
        return;
    }
#endif
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
#if PLOW_MLA_PF_MFMA
        /* Dense only. The gathered arm above keeps the scalar body: its top_k set is per
         * QUERY, so a tile of query rows shares no KV range to stage. */
        if (mla_pf_tiled_fills(in->i[0], in->i[4], in->i[1], nblk)) {
            d_flash_mla_prefill_mfma<512, 64, /*FP8=*/false>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0],
                in->i[4], in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice, nblk,
                lds, nullptr, 0);
            return;
        }
#endif
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
    d_gemm_small((bf16*)TEN(0) + (size_t)in->i[5] * in->i[1],
           (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                 (const bf16*)TEN(2), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm_med(CInst in, CTens T,
                                              unsigned slice, unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_med((bf16*)TEN(0) + (size_t)in->i[5] * in->i[1],
           (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
               (const bf16*)TEN(2), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm_wide(CInst in, CTens T,
                                               unsigned slice, unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_wide((bf16*)TEN(0) + (size_t)in->i[5] * in->i[1],
           (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                (const bf16*)TEN(2), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm_c5(CInst in, CTens T,
                                             unsigned slice, unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_c5((bf16*)TEN(0) + (size_t)in->i[5] * in->i[1],
           (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
              (const bf16*)TEN(2), in->i[0], in->i[1], in->i[2], slice, nblk, lds);
#undef TEN
}
__device__ __forceinline__ void exec_gemm(CInst in, CTens T,
                                                    unsigned slice, unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    /* i4 = a_row0: skip rows of A; i5 = c_row0: land the output at a row offset. Both free —
     * pointer arithmetic, not a kernel change. Together they let an emitter split one GEMM
     * into row BANDS (same tiles, same sums, disjoint rows => bit-identical), which is what
     * the banded TP seams use to start a band's all-reduce while later bands still compute. */
    d_gemm((bf16*)TEN(0) + (size_t)in->i[5] * in->i[1],
           (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2], (const bf16*)TEN(2),
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
    /* No `part` here on purpose: `plow_smem` is a UNION, so `sm->part` and `sm->gm` are one
     * buffer. The fused-norm arm (norm == 2) takes its reduction scratch from the TOP of the
     * arena itself — handing it `sm->part` would write through the row it is reducing. */
    d_gemv((bf16*)TEN(0), (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2], (const bf16*)TEN(2),
           (const float*)TEN(3), (const bf16*)TEN(4), in->i[0], in->i[1], in->i[2],
           (int)in->i[3], in->fj[0].f, slice, nblk, lds);
#undef TEN
}
#if PLOW_FP8
/* t0=C t1=x t2=W(fp8) t5=w_scale(f32[N])   i0=M i1=N i2=K i4=a_row0
 *
 * i3 != 0 is the NRN FOLD (op 23 computed into this GEMV's staging — see gemv_nrn_lds):
 *   t1=a(resid in) t3=resid_out t4=b t6=gamma_b t7=gamma_n · f0=eps f1=scale ·
 *   i3 bit0 = fold, bit1 = this packet stores the residual (ONE of the q/k/v trio).
 * Absence TRAPS: `default:` writes nothing, so a fold packet missing an operand would stage an
 * un-normed row and decode fluent garbage — the exact failure the serve gate exists to catch. */
__device__ __forceinline__ void exec_gemv_fp8(CInst in, CTens T, unsigned slice, unsigned nblk,
                                             bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    if (in->i[3] != 0u) {
        if (!TEN(1) || !TEN(3) || !TEN(4) || !TEN(6) || !TEN(7)) __builtin_trap();
        d_gemv_fp8_nrn((bf16*)TEN(0), (bf16*)TEN(3), (const bf16*)TEN(1), (const bf16*)TEN(4),
                       (const bf16*)TEN(6), (const bf16*)TEN(7), (const unsigned char*)TEN(2),
                       (const float*)TEN(5), in->i[0], in->i[1], in->i[2], in->fj[0].f,
                       in->fj[1].f, in->i[3] & 2u, slice, nblk, lds);
        return;
    }
    d_gemv_fp8((bf16*)TEN(0), (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
               (const unsigned char*)TEN(2), (const float*)TEN(5), in->i[0], in->i[1], in->i[2],
               slice, nblk, lds);
#undef TEN
}
/* (The block-fp8 decode GEMV wrapper lived here and was a duplicate of the inline body at
 * `case PLOW_DOP_GEMV_FP8_BLK`; the two were kept apart by complementary #if PLOW_FP8 guards.
 * Op 44 is a BLOCK-fp8 arm — a weight encoding independent of the per-channel fp8 this flag
 * names — so the case is now unconditional and there is one body, not two.) */
/* FUSED q|k|v fp8 DECODE GEMV — op 22's fp8 twin, and op 30's own body with Nk=Nv=0.
 *
 * The three f32 dequant-scale rows are TENSOR HANDLES in i5/i6/i7 (ten pointers do not fit t[8];
 * dev_isa.h PLOW_DOP_GEMV_QKV_FP8 states why the demoted operands are scales and not outputs).
 *
 * NO SILENT FALLBACK, exactly as exec_gemv_qkv_mxfp4: this dispatch's `default:` is a NOP, so a
 * malformed packet quietly running a narrower sweep would leave an output exactly as it found it
 * — finite, fluent and wrong. Nq/Nk and their operands are REQUIRED; Nv==0 is the legal
 * two-stream form and then t5/t6/i7 must ALL be absent. Anything else traps. */
__device__ __forceinline__ void exec_gemv_qkv_fp8(CInst in, CTens T, unsigned slice, unsigned nblk,
                                                  bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
#define IEN(k) (in->i[k] == PLOW_TENSOR_NONE ? nullptr : T[in->i[k]])
    const float* Sq = (const float*)IEN(5);
    const float* Sk = (const float*)IEN(6);
    const float* Sv = (const float*)IEN(7);
    const bool q_ok = in->i[1] != 0u && TEN(0) && TEN(2) && Sq;
    const bool k_ok = in->i[3] != 0u && TEN(3) && TEN(4) && Sk;
    const bool v_on = TEN(5) || TEN(6) || Sv || in->i[4] != 0u;
    const bool v_ok = v_on ? (in->i[4] != 0u && TEN(5) && TEN(6) && Sv) : true;
    if (!q_ok || !k_ok || !v_ok) __builtin_trap();
    d_gemv_qkv_fp8((bf16*)TEN(0), (bf16*)TEN(3), (bf16*)TEN(5), (const bf16*)TEN(1),
                   (const unsigned char*)TEN(2), (const unsigned char*)TEN(4),
                   (const unsigned char*)TEN(6), Sq, Sk, Sv, in->i[0], in->i[1], in->i[3],
                   in->i[4], in->i[2], slice, nblk, lds);
#undef IEN
#undef TEN
}
/* t0=fu t1=x t2=Wg(fp8) t5=Wu(fp8) t3=gate_scale t4=up_scale   i0=M i1=N i2=K i5=act
 *
 * fj[2].u != 0 is the NRN1 FOLD (the post-attention NormResidualNorm computed into this GEMV's
 * staging): t1 becomes `a` (the residual), and i3/i4/i6/i7 carry the resid_out/b/gamma_b/gamma_n
 * TENSOR HANDLES (op 114's demotion); f0=eps f1=layer_scale. Absence TRAPS — a fold packet
 * missing an operand would stage an un-normed row and decode fluent garbage. */
__device__ __forceinline__ void exec_gemv_glu_fp8(CInst in, CTens T, unsigned slice, unsigned nblk,
                                                 bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
#define IEN(k) (in->i[k] == PLOW_TENSOR_NONE ? nullptr : T[in->i[k]])
    const unsigned nrn = in->fj[2].u;
    if (nrn != 0u) {
        if (!TEN(1) || !IEN(3) || !IEN(4) || !IEN(6) || !IEN(7)) __builtin_trap();
        d_gemv_glu_fp8_nrn((bf16*)TEN(0), (bf16*)IEN(3), (const bf16*)TEN(1), (const bf16*)IEN(4),
                           (const bf16*)IEN(6), (const bf16*)IEN(7), (const unsigned char*)TEN(2),
                           (const unsigned char*)TEN(5), (const float*)TEN(3),
                           (const float*)TEN(4), in->i[0], in->i[1], in->i[2], in->i[5],
                           in->fj[0].f, in->fj[1].f, slice, nblk, lds);
    } else {
        d_gemv_glu_fp8((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                       (const unsigned char*)TEN(5), (const float*)TEN(3), (const float*)TEN(4),
                       in->i[0], in->i[1], in->i[2], in->i[5], slice, nblk, lds);
    }
#undef IEN
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

/* MXFP4 fused q|k|v DECODE GEMV — op 22's fp4 twin, and op 91's own body with Nk=Nv=0.
 *
 * The three E8M0 scale rows are TENSOR HANDLES in i5/i6/i7, because ten pointers do not fit t[8];
 * dev_isa.h states why the demoted operands are scale rows and not outputs.
 *
 * NO SILENT FALLBACK. This interpreter's dispatch `default:` is a NOP, so the one thing that must
 * not happen is a malformed packet quietly running a narrower sweep and leaving an output exactly
 * as it found it. Nq/Nk and their operands are REQUIRED; Nv==0 is the legal two-stream form and
 * then t5/t6/i7 must ALL be absent. Anything else traps. */
__device__ __forceinline__ void exec_gemv_qkv_mxfp4(CInst in, CTens T, unsigned slice,
                                                    unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
#define IEN(k) (in->i[k] == PLOW_TENSOR_NONE ? nullptr : T[in->i[k]])
    const unsigned char* Sq = (const unsigned char*)IEN(5);
    const unsigned char* Sk = (const unsigned char*)IEN(6);
    const unsigned char* Sv = (const unsigned char*)IEN(7);
    const bool q_ok = in->i[1] != 0u && TEN(0) && TEN(2) && Sq;
    const bool k_ok = in->i[3] != 0u && TEN(3) && TEN(4) && Sk;
    const bool v_on = TEN(5) || TEN(6) || Sv || in->i[4] != 0u;
    const bool v_ok = v_on ? (in->i[4] != 0u && TEN(5) && TEN(6) && Sv) : true;
    if (!q_ok || !k_ok || !v_ok) __builtin_trap();
    d_gemv_qkv_mxfp4((bf16*)TEN(0), (bf16*)TEN(3), (bf16*)TEN(5), (const bf16*)TEN(1),
                     (const unsigned char*)TEN(2), (const unsigned char*)TEN(4),
                     (const unsigned char*)TEN(6), Sq, Sk, Sv, in->i[0], in->i[1], in->i[3],
                     in->i[4], in->i[2], slice, nblk, lds);
#undef IEN
#undef TEN
}

#if PLOW_BUCKET_PREFILL
/* MXFP4 (w4a16) PREFILL fused gate|up GEMM+GLU. Mirrors exec_gemm_mxfp4's operands with a second
 * fp4 weight and its own E8M0 scale rows, in op 92's slot layout.
 * t0=fu t1=A(bf16) t2=Wg(fp4) t5=Wu(fp4) t3=Sg(e8m0) t4=Su(e8m0)   i0=M i1=N i2=K i5=act */
#if PLOW_WAVES == 8
__device__ __forceinline__ void exec_gemm_glu_mxfp4(CInst in, CTens T, unsigned slice,
                                                    unsigned nblk, bf16* lds) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    d_gemm_glu_mxfp4((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                     (const unsigned char*)TEN(5), (const unsigned char*)TEN(3),
                     (const unsigned char*)TEN(4), in->i[0], in->i[1], in->i[2], in->i[5], slice,
                     nblk, lds);
#undef TEN
}
#endif
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

/* RESTORE THE SCALAR PROMISE ACROSS THE CALL BOUNDARY.
 *
 * `PLOW_KONST` (address space 4) buys s_load only while the pointer VALUE is in SGPRs.
 * plow_exec is a real call (see the note above -- it must stay one), and the AMDGPU ABI
 * passes non-`inreg` arguments in VGPRs. So `in` arrives in v[0:1] and every field read
 * reverts to exactly the vector chase the address-space annotation was added to remove:
 *
 *     v_mov_b32_e32     v150, v0                 <- `in` lands in VGPRs
 *     global_load_dword v167, v[150:151], off    <- in->op
 *     s_waitcnt vmcnt(0)                         <- full vector drain
 *     v_cmp_lt_i16_e32  vcc, 44, v167            <- opcode dispatch on the VALU
 *
 * Measured in interp_decode_fp8kv_k3_gq.elf: plow_exec contains 0 `s_load` against 1418
 * vector loads, 485 of them based on the instruction pointer, and 283 of those are followed
 * within six instructions by `s_waitcnt vmcnt(0)` -- a drain that is NOT specific to the
 * packet fields and takes every in-flight WEIGHT load down with it.
 *
 * readfirstlane says the thing the ABI threw away: these pointers are wave-uniform. They
 * genuinely are -- `in` is `&insts[pc]` with pc uniform per workgroup, and `T` is the
 * tensor table base. Same shape as `buf_rsrc_fp8_u` in amd_common.h, which does this for
 * buffer descriptors. Costs 4 VALU at entry; the loads then issue as s_load on lgkmcnt,
 * which does not interact with the vector memory stream at all. */
template <class P>
__device__ __forceinline__ P plow_scalar_ptr(P p) {
    unsigned long long u = (unsigned long long)(size_t)p;
    unsigned lo = __builtin_amdgcn_readfirstlane((unsigned)u);
    unsigned hi = __builtin_amdgcn_readfirstlane((unsigned)(u >> 32));
    return (P)(size_t)(((unsigned long long)hi << 32) | lo);
}

#if PLOW_MLA_PF_V2_ARM
/* Marker: this object's V2 arm understands the ofold epilogue (packet i[6] on op 51 —
 * normalized bf16 partials for the W_ofold fused o-GEMM). Inside the arm guard so its
 * presence IS the arm's presence; plowrt's flash-object check refuses an ofold blob when
 * it is missing (an old flash object would write unnormalized f32 partials that the fused
 * GEMM reads as bf16 garbage). */
extern "C" __device__ unsigned plow_glm_ofold_arm = 1;

#if PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS
template <bool FP8>
__device__ __forceinline__ void exec_flash_mla_prefill_v2_packed(
    CInst in, CTens T, unsigned slice, unsigned nblk, bf16* lds, const PlowProgram* prog) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
    if (!plow_packed_prefill_enabled(prog)) __builtin_trap();
    if ((!FP8 && in->t[7] != PLOW_TENSOR_NONE) || (FP8 && TEN(7) == nullptr)) __builtin_trap();
    for (unsigned si = 0; si < prog->n_prefill_spans; ++si) {
        const PlowPrefillSpan* s = plow_packed_prefill_span(prog, si);
        float* op = (float*)TEN(0) + (size_t)s->row0 * in->i[1] * 512u;
        float* ml = (float*)TEN(1) + (size_t)s->row0 * in->i[1] * 2u;
        const bf16* qa = (const bf16*)TEN(2) + (size_t)s->row0 * in->i[1] * 512u;
        const bf16* qr = (const bf16*)TEN(3) + (size_t)s->row0 * in->i[1] * 64u;
        const char* ck = (const char*)TEN(4) +
                         (size_t)s->slot * in->i[2] * 512u * (FP8 ? 1u : 2u);
        const bf16* kr = (const bf16*)TEN(5) + (size_t)s->slot * in->i[2] * 64u;
        const float* ks = FP8 ? (const float*)TEN(7) + (size_t)s->slot * in->i[2] : nullptr;
        d_flash_mla_prefill_v2<512, 64, /*GATHER=*/false, FP8>(
            op, ml, qa, qr, (const bf16*)ck, kr, (const int*)TEN(6) + s->slot, 1u,
            s->n_rows, in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice, nblk,
            lds, nullptr, 0u, FP8 ? 1u : ((in->i[6] & 0xffu) ? (in->i[6] & 0xffu) : 1u),
            FP8 ? false : ((in->i[6] >> 8) & 1u) != 0u, ks, FP8 ? in->i[6] : 0u);
    }
#undef TEN
}
#endif
#endif

__device__ static void plow_exec(CInst in_v, unsigned slice, CTens T_v, plow_smem* sm,
                                 const PlowProgram* prog_meta) {
    CInst in = plow_scalar_ptr(in_v);
    CTens T = plow_scalar_ptr(T_v);
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
#if PLOW_PACKED_PREFILL_MLA_NORM_CONSUMERS
#define PLOW_PACKED_NORM_ARGS , prog_meta, in->i[7]
#else
#define PLOW_PACKED_NORM_ARGS
#endif
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
#if !defined(PLOW_BUCKET_MLA_V2_RAW)
#if PLOW_FP8_KV
    if (in->op == PLOW_DOP_FLASH_PREFILL_FP8) exec_flash_prefill_fp8(in, T, slice, nblk, sm->fa);
#else
    if (in->op == PLOW_DOP_FLASH_PREFILL) exec_flash_prefill(in, T, slice, nblk, sm->fa);
#endif
#endif
#if PLOW_MLA_PF_V2_ARM
    /* V2 MLA prefill: the full-column-wave layout this 4-wave/512-reg object exists to
     * afford (op_attention.h d_flash_mla_prefill_v2).
     *   t0=Opart t1=mlpart t2=Qabs t3=Qrope t4=Ckv t5=Krope t6=kv_len
     *   i0=n_batch i1=n_head i2=kv_stride i3=window i4=n_tok i5=kv_mask f0=scale */
    static_assert(PLOW_WAVES == 4, "the V2 MLA prefill arm is shaped for the 4-wave object");
    static_assert(FA_MLA_PF2_LDS_BYTES(512, 64) <= (int)sizeof(((plow_smem*)nullptr)->fa),
                  "V2 MLA prefill LDS exceeds the flash arena");
#if !defined(PLOW_BUCKET_MLA_V2_RAW)
    if (in->op == PLOW_DOP_FLASH_MLA_PREFILL_FP8) {
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
#if PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS
        if (plow_packed_prefill_enabled(prog_meta))
            exec_flash_mla_prefill_v2_packed</*FP8=*/true>(in, T, slice, nblk, sm->fa,
                                                           prog_meta);
        else
#endif
            d_flash_mla_prefill_v2<512, 64, /*GATHER=*/false, /*FP8=*/true>(
            (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
            (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0],
            in->i[4], in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice, nblk,
            sm->fa, nullptr, 0u, 1u, false, (const float*)TEN(7), in->i[6]);
#undef TEN
    }
#endif
#if !PLOW_FP8_KV
#if defined(PLOW_BUCKET_MLA_V2_RAW)
    if (in->op == PLOW_DOP_FLASH_MLA_PREFILL) {
#else
    else if (in->op == PLOW_DOP_FLASH_MLA_PREFILL) {
#endif
#define TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
#if PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS
        if (plow_packed_prefill_enabled(prog_meta)) {
            exec_flash_mla_prefill_v2_packed</*FP8=*/false>(in, T, slice, nblk, sm->fa,
                                                            prog_meta);
        } else {
#endif
        /* t7 = the DSA per-64-query-tile union table (op 119) selects the GATHER arm:
         * same full-column-wave body, but the KV walk covers the tile's union entries and
         * per-query exactness comes from the staged membership masks. i6 = cap.
         *
         * SEPARATELY BUILD-GATED (PLOW_DSA_PF_ARM), because the interpreter inlines every
         * arm and register allocation is the WORST CASE over all of them: instantiating the
         * gathered body raised this object's spill 98 -> 287 even for blobs that never emit
         * a union table. Off by default, so the shipped flash object is unchanged; a sparse
         * blob loaded against an object without the arm keeps t7 unread and runs DENSE. */
#if PLOW_DSA_PF_ARM
        if (in->t[7] != PLOW_TENSOR_NONE)
            d_flash_mla_prefill_v2<512, 64, /*GATHER=*/true>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0],
                in->i[4], in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice, nblk,
                sm->fa, (const unsigned char*)TEN(7), in->i[6]);
        else
#endif
            /* DENSE arm i6 bitfield: low 8 bits = causal KV-split ns (0/1 = unsplit), bit 8
             * = the W_ofold fused epilogue (requires ns==1 — emit-enforced; the fold consumes
             * the un-split l). 0 on every pre-feature blob. The sparse GATHER arm above reuses
             * i6 whole as `cap`, disambiguated by t7's presence. */
            d_flash_mla_prefill_v2<512, 64>(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const int*)TEN(6), in->i[0],
                in->i[4], in->i[1], in->i[2], in->i[3], in->fj[0].f, in->i[5], slice, nblk,
                sm->fa, nullptr, 0u, (in->i[6] & 0xffu) ? (in->i[6] & 0xffu) : 1u,
                ((in->i[6] >> 8) & 1u) != 0u);
#if PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS
        }
#endif
#undef TEN
    }
#endif
#endif
    (void)slice;
#elif defined(PLOW_BUCKET_PACKED_MLA_NORM)
    if (!plow_packed_prefill_enabled(prog_meta)) __builtin_trap();
    switch (in->op) {
        case PLOW_DOP_RMSNORM:
            d_rmsnorm((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), in->i[0],
                      in->i[1], in->fj[0].f, in->i[2], slice, nblk, sm->part,
                      (unsigned char*)TEN(3), (float*)TEN(4), prog_meta, in->i[7]);
            break;
        case PLOW_DOP_HEADNORM_ROPE:
#define PLOW_PACKED_HEADNORM(HD, INTERLEAVE)                                                \
    d_headnorm_rope<HD, INTERLEAVE>(                                                        \
        (bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), (const float*)TEN(3),      \
        (const float*)TEN(4), (const int*)TEN(5), in->i[0], in->i[1], in->fj[0].f,         \
        in->i[3], in->fj[1].u, in->fj[2].u, in->i[4], slice, nblk, in->i[6], prog_meta,     \
        in->i[7])
            if (in->i[2] == 64)
                PLOW_PACKED_HEADNORM(64, true);
            else if (in->i[2] == 128 && in->i[5] == 1)
                PLOW_PACKED_HEADNORM(128, true);
            else if (in->i[2] == 128)
                PLOW_PACKED_HEADNORM(128, false);
            else if (in->i[2] == 256)
                PLOW_PACKED_HEADNORM(256, false);
            else if (in->i[2] == 512)
                PLOW_PACKED_HEADNORM(512, false);
            else
                __builtin_trap();
#undef PLOW_PACKED_HEADNORM
            break;
#if PLOW_FP8_KV
        case PLOW_DOP_HEADNORM_ROPE_FP8:
#define PLOW_PACKED_HEADNORM_FP8(HD)                                                        \
    d_headnorm_rope_fp8<HD>(                                                                \
        (unsigned char*)TEN(0), (float*)TEN(6), (const bf16*)TEN(1),                        \
        (const bf16*)TEN(2), (const float*)TEN(3), (const float*)TEN(4),                    \
        (const int*)TEN(5), in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u,        \
        in->fj[2].u, in->i[4], slice, nblk, in->i[6], prog_meta, in->i[7])
            if (in->i[2] == 64)
                PLOW_PACKED_HEADNORM_FP8(64);
            else if (in->i[2] == 128)
                PLOW_PACKED_HEADNORM_FP8(128);
            else if (in->i[2] == 256)
                PLOW_PACKED_HEADNORM_FP8(256);
            else if (in->i[2] == 512)
                PLOW_PACKED_HEADNORM_FP8(512);
            else
                __builtin_trap();
#undef PLOW_PACKED_HEADNORM_FP8
            break;
#endif
        default:
            __builtin_trap();
    }
#elif defined(PLOW_BUCKET_PACKED_KDA)
    if (!plow_packed_prefill_enabled(prog_meta)) __builtin_trap();
    switch (in->op) {
#if PLOW_HAS_KDA_STATE_STEP
        case PLOW_DOP_KDA_STATE_STEP:
            d_kda_state_step_packed</*GATED=*/false>(
                (bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                (const bf16*)TEN(3), (const float*)TEN(4), (const float*)TEN(5), nullptr,
                nullptr, nullptr, nullptr, 0u, 0.0f, (float*)TEN(6), in->i[1], in->i[2],
                in->i[3], in->i[4], in->fj[0].f, slice, nblk, (float*)sm->raw, prog_meta);
            break;
#endif /* PLOW_HAS_KDA_STATE_STEP */
#if PLOW_HAS_KDA_CONV3
        case PLOW_DOP_KDA_CONV3: {
            const float* wv = (in->i[4] == PLOW_TENSOR_NONE) ? nullptr : (const float*)T[in->i[4]];
            float* cs[3];
#pragma unroll
            for (int s = 0; s < 3; ++s)
                cs[s] = (in->i[5 + s] == PLOW_TENSOR_NONE) ? nullptr : (float*)T[in->i[5 + s]];
            if (wv == nullptr || cs[0] == nullptr || cs[1] == nullptr || cs[2] == nullptr ||
                TEN(6) == nullptr || TEN(7) == nullptr || in->i[1] == 0u)
                __builtin_trap();
            d_kda_conv3_packed(
                (bf16*)TEN(0), (bf16*)TEN(1), (bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const float*)TEN(6),
                (const float*)TEN(7), wv, cs[0], cs[1], cs[2], in->i[1], in->i[2],
                in->i[3], slice, nblk, prog_meta);
        } break;
#endif /* PLOW_HAS_KDA_CONV3 */
#if PLOW_HAS_KDA_STATE_STEP_G
        case PLOW_DOP_KDA_STATE_STEP_G: {
            const float* dtb = (in->i[5] == PLOW_TENSOR_NONE) ? nullptr : (const float*)T[in->i[5]];
            if (dtb == nullptr || TEN(7) == nullptr || TEN(4) == nullptr || TEN(5) == nullptr)
                __builtin_trap();
            d_kda_state_step_packed</*GATED=*/true>(
                (bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                (const bf16*)TEN(3), nullptr, nullptr, (const bf16*)TEN(4),
                (const bf16*)TEN(5), (const float*)TEN(7), dtb, in->i[6], in->fj[1].f,
                (float*)TEN(6), in->i[1], in->i[2], in->i[3], in->i[4], in->fj[0].f,
                slice, nblk, (float*)sm->raw, prog_meta);
        } break;
#endif /* PLOW_HAS_KDA_STATE_STEP_G */
        default:
            __builtin_trap();
    }
#else
    switch (in->op) {
        case PLOW_DOP_RMSNORM:
            /* i[2] = out_row0: the OUTPUT cache row (GLM decode-step latent write, patched per step);
             * 0 for every other RMSNORM (in-place), so all existing packets are bit-identical.
             * t[3]/t[4] = OPTIONAL fused w8a8 activation-quant outputs (xq bytes + per-row
             * a_scale) — bit-identical to the QUANT_FP8 packet they replace; absent (NONE) on
             * every pre-existing packet. */
            d_rmsnorm((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), in->i[0],
                      in->i[1], in->fj[0].f, in->i[2], slice, nblk, sm->part,
                      (unsigned char*)TEN(3), (float*)TEN(4) PLOW_PACKED_NORM_ARGS);
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
                                     in->i[4], slice, nblk, in->i[6] PLOW_PACKED_NORM_ARGS);
            else if (in->i[2] == 128)
                d_headnorm_rope<128>((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const float*)TEN(3), (const float*)TEN(4), (const int*)TEN(5),
                                     in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                                     in->i[4], slice, nblk, in->i[6] PLOW_PACKED_NORM_ARGS);
            else if (in->i[2] == 256)
                d_headnorm_rope<256>((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const float*)TEN(3), (const float*)TEN(4), (const int*)TEN(5),
                                     in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                                     in->i[4], slice, nblk, in->i[6] PLOW_PACKED_NORM_ARGS);
            else if (in->i[2] == 512)
                d_headnorm_rope<512>((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const float*)TEN(3), (const float*)TEN(4), (const int*)TEN(5),
                                     in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                                     in->i[4], slice, nblk, in->i[6] PLOW_PACKED_NORM_ARGS);
            /* HD=64 is GLM-5.2's partial-rope slice, and GLM applies INTERLEAVED (GPT-J) RoPE there
             * (dynamic per-step q_rope + shared k_rope for multi-token decode). No other arch uses
             * HD=64, so HD==64 selects the interleaved template; every 128/256/512 path stays
             * bit-identical (INTERLEAVE defaults false). See op_norm.h + rope_interleave_glm_test.c. */
            else if (in->i[2] == 64)
                d_headnorm_rope<64, /*INTERLEAVE=*/true>(
                                     (bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const float*)TEN(3), (const float*)TEN(4), (const int*)TEN(5),
                                     in->i[0], in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u,
                                     in->i[4], slice, nblk, in->i[6] PLOW_PACKED_NORM_ARGS);
            break;
#if PLOW_FP8_KV
        /* FP8 KV write (k/v only; q still uses HEADNORM_ROPE above). t0=uint8 cache, t6=f32 scale. */
        case PLOW_DOP_HEADNORM_ROPE_FP8:
            /* HD=64 is the MLA `krot` row (qk_rope_head_dim), the ONLY width below 128 any KV
             * cache uses. It exists so the ckv+krot fp8 variant is MEASURABLE; the shipped MLA
             * fp8 form leaves krot bf16 and never reaches here. [MLA-FP8-KV] */
            if (in->i[2] == 64)
                d_headnorm_rope_fp8<64>((unsigned char*)TEN(0), (float*)TEN(6), (const bf16*)TEN(1),
                                        (const bf16*)TEN(2), (const float*)TEN(3),
                                        (const float*)TEN(4), (const int*)TEN(5), in->i[0],
                                        in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u, in->i[4],
                                        slice, nblk, in->i[6] PLOW_PACKED_NORM_ARGS);
            else if (in->i[2] == 128)
                d_headnorm_rope_fp8<128>((unsigned char*)TEN(0), (float*)TEN(6), (const bf16*)TEN(1),
                                         (const bf16*)TEN(2), (const float*)TEN(3),
                                         (const float*)TEN(4), (const int*)TEN(5), in->i[0],
                                         in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u, in->i[4],
                                         slice, nblk, in->i[6] PLOW_PACKED_NORM_ARGS);
            else if (in->i[2] == 256)
                d_headnorm_rope_fp8<256>((unsigned char*)TEN(0), (float*)TEN(6), (const bf16*)TEN(1),
                                         (const bf16*)TEN(2), (const float*)TEN(3),
                                         (const float*)TEN(4), (const int*)TEN(5), in->i[0],
                                         in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u, in->i[4],
                                         slice, nblk, in->i[6] PLOW_PACKED_NORM_ARGS);
            else if (in->i[2] == 512)
                d_headnorm_rope_fp8<512>((unsigned char*)TEN(0), (float*)TEN(6), (const bf16*)TEN(1),
                                         (const bf16*)TEN(2), (const float*)TEN(3),
                                         (const float*)TEN(4), (const int*)TEN(5), in->i[0],
                                         in->i[1], in->fj[0].f, in->i[3], in->fj[1].u, in->fj[2].u, in->i[4],
                                         slice, nblk, in->i[6] PLOW_PACKED_NORM_ARGS);
            break;
#endif
        case PLOW_DOP_NORM_RESIDUAL:
            d_norm_residual((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                            (const bf16*)TEN(3), in->i[0], in->i[1], in->fj[0].f, in->fj[1].f, slice,
                            nblk, sm->part);
            break;
        /* t3 = the OUTER residual of a fused chained pair: `out = (t3 + bf16(t1 + t2)) * f0`.
         * PLOW_TENSOR_NONE is the plain two-input add, byte for byte. See d_residual's note. */
        case PLOW_DOP_RESIDUAL:
            d_residual((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), in->i[0],
                       in->fj[0].f, slice, nblk, (const bf16*)TEN(3));
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
#if PLOW_KDA_CHUNK
#if PLOW_HAS_KDA_CHUNK_PREPARE
        case PLOW_DOP_KDA_CHUNK_PREPARE:
            if (in->i[0] == 0u || in->i[1] == 0u ||
                (in->i[2] != 64u && in->i[2] != 128u))
                __builtin_trap();
#if PLOW_PACKED_PREFILL_KDA_CONSUMERS
            if (plow_packed_prefill_enabled(prog_meta)) {
                d_kda_chunk_prepare_packed_bt64(
                    (bf16*)TEN(0), (bf16*)TEN(1), (float*)TEN(2), (float*)TEN(3),
                    (const bf16*)TEN(4), (const bf16*)TEN(5), (const float*)TEN(6),
                    (const float*)TEN(7), in->i[1], in->i[2], in->i[3], in->fj[0].f,
                    slice, nblk, prog_meta);
                break;
            }
#endif
            d_kda_chunk_prepare_bt64(
                (bf16*)TEN(0), (bf16*)TEN(1), (float*)TEN(2), (float*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const float*)TEN(6),
                (const float*)TEN(7), nullptr, (in->i[0] + 63u) / 64u, in->i[0], in->i[1],
                in->i[2], in->i[3], in->fj[0].f, slice, nblk);
            break;
#endif /* PLOW_HAS_KDA_CHUNK_PREPARE */
#if PLOW_HAS_KDA_CHUNK_INTRA
        case PLOW_DOP_KDA_CHUNK_INTRA:
            if (in->i[0] == 0u || in->i[1] == 0u ||
                (in->i[2] != 64u && in->i[2] != 128u))
                __builtin_trap();
#if PLOW_PACKED_PREFILL_KDA_CONSUMERS
            if (plow_packed_prefill_enabled(prog_meta)) {
                d_kda_chunk_intra_packed_bt64(
                    (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2),
                    (const bf16*)TEN(3), (const float*)TEN(4), (const float*)TEN(5),
                    in->i[1], in->i[2], in->fj[0].f, slice, nblk, (float*)sm->raw,
                    prog_meta);
                break;
            }
#endif
            d_kda_chunk_intra_bt64(
                (float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const float*)TEN(4), (const float*)TEN(5), nullptr, (in->i[0] + 63u) / 64u,
                in->i[0], in->i[1], in->i[2], in->fj[0].f, slice, nblk, (float*)sm->raw);
            break;
#endif /* PLOW_HAS_KDA_CHUNK_INTRA */
#if PLOW_HAS_KDA_CHUNK_WU
        case PLOW_DOP_KDA_CHUNK_WU:
            if (in->i[0] == 0u || in->i[1] == 0u ||
                (in->i[2] != 64u && in->i[2] != 128u) || in->i[3] != in->i[2])
                __builtin_trap();
#if PLOW_PACKED_PREFILL_KDA_CONSUMERS
            if (plow_packed_prefill_enabled(prog_meta)) {
                d_kda_chunk_wu_packed_bt64(
                    (bf16*)TEN(0), (bf16*)TEN(1),
                    in->i[4] ? (bf16*)TEN(7) : nullptr, (const float*)TEN(2),
                    (const bf16*)TEN(3), (const bf16*)TEN(4), (const float*)TEN(5),
                    (const float*)TEN(6), in->i[1], in->i[2], in->i[3], in->fj[0].f,
                    in->i[4] != 0u, slice, nblk, prog_meta);
                break;
            }
#endif
            d_kda_chunk_wu_bt64(
                (bf16*)TEN(0), (bf16*)TEN(1), in->i[4] ? (bf16*)TEN(7) : nullptr,
                (const float*)TEN(2), (const bf16*)TEN(3), (const bf16*)TEN(4),
                (const float*)TEN(5), (const float*)TEN(6), nullptr,
                (in->i[0] + 63u) / 64u, in->i[0], in->i[1], in->i[2], in->i[3],
                in->fj[0].f, in->i[4] != 0u, slice, nblk);
            break;
#endif /* PLOW_HAS_KDA_CHUNK_WU */
#if PLOW_HAS_KDA_CHUNK_CARRY
        case PLOW_DOP_KDA_CHUNK_CARRY: {
            if (in->i[0] == 0u || in->i[1] == 0u ||
                (in->i[2] != 64u && in->i[2] != 128u) || in->i[3] != in->i[2])
                __builtin_trap();
            float* kst = (float*)sm->raw;
            bf16* kv = (bf16*)(kst + 16u * in->i[2]);
            float* ko = (float*)(kv + 64u * 16u);
#if PLOW_PACKED_PREFILL_KDA_CONSUMERS
            if (plow_packed_prefill_enabled(prog_meta)) {
                d_kda_chunk_carry_packed_bt64(
                    (bf16*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2),
                    (const bf16*)TEN(3), (const bf16*)TEN(4), (const bf16*)TEN(5),
                    (const float*)TEN(6), (const float*)TEN(7), in->i[1], in->i[2],
                    in->i[3], in->fj[0].f, slice, nblk, kst, kv, ko, prog_meta,
                    in->i[4] != 0u);
                break;
            }
#endif
            d_kda_chunk_carry_bt64(
                (bf16*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const bf16*)TEN(4), (const bf16*)TEN(5), (const float*)TEN(6),
                (const float*)TEN(7), nullptr, (in->i[0] + 63u) / 64u, in->i[0], in->i[1],
                in->i[2], in->i[3], in->fj[0].f, slice, nblk, kst, kv, ko,
                in->i[4] != 0u);
            break;
        }
#endif /* PLOW_HAS_KDA_CHUNK_CARRY */
#endif
#if !PLOW_KDA_CONV_STEP_DB
#if PLOW_HAS_KDA_CONV
        case PLOW_DOP_KDA_CONV:
            d_kda_conv((bf16*)TEN(0), (const bf16*)TEN(1), (const float*)TEN(2), (float*)TEN(3),
                       in->i[0], in->i[1], in->i[2], in->i[3], slice, nblk,
                       /* see the CONV3 note below: no free operand to select the path yet */
                       (size_t)0);
            break;
#endif /* PLOW_HAS_KDA_CONV */
#if PLOW_HAS_KDA_GATE
        case PLOW_DOP_KDA_GATE:
            d_kda_gate((float*)TEN(0), (float*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                       (const float*)TEN(4), (const float*)TEN(5), in->i[0], in->i[1], in->i[2],
                       in->i[3], in->fj[0].f, slice, nblk);
            break;
#endif /* PLOW_HAS_KDA_GATE */
#if PLOW_HAS_KDA_STATE_STEP
        case PLOW_DOP_KDA_STATE_STEP:
#if PLOW_PACKED_PREFILL_KDA_CONSUMERS
            if (plow_packed_prefill_enabled(prog_meta))
                d_kda_state_step_packed</*GATED=*/false>(
                    (bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                    (const bf16*)TEN(3), (const float*)TEN(4), (const float*)TEN(5), nullptr,
                    nullptr, nullptr, nullptr, 0u, 0.0f, (float*)TEN(6), in->i[1], in->i[2],
                    in->i[3], in->i[4], in->fj[0].f, slice, nblk, (float*)sm->raw, prog_meta);
            else
#endif
                d_kda_state_step((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                             (const bf16*)TEN(3), (const float*)TEN(4), (const float*)TEN(5),
                             (float*)TEN(6), in->i[0], in->i[1], in->i[2], in->i[3], in->i[4],
                             in->fj[0].f, slice, nblk, (float*)sm->raw,
                             (in->i[4] & PLOW_KDA_F_SEQ_ROWS)
                                 ? (size_t)in->i[1] * in->i[2] * in->i[2]
                                 : (size_t)0,
                             /* op 102 predates the per-row mask and carries no slot for it. */
                             nullptr);
            break;
#endif /* PLOW_HAS_KDA_STATE_STEP */
#endif
#if PLOW_HAS_KDA_GATED_NORM
        case PLOW_DOP_KDA_GATED_NORM:
            d_kda_gated_norm((bf16*)TEN(0), (const bf16*)TEN(1), (const float*)TEN(2),
                             (const bf16*)TEN(3), in->i[0], in->i[1], in->i[2], in->fj[0].f, slice,
                             nblk);
            break;
#endif /* PLOW_HAS_KDA_GATED_NORM */
#if !PLOW_KDA_CONV_STEP_DB
        /* FUSED KDA — ops 88x3 -> 109 and 89+102 -> 110. Six packets per layer become three, 207
         * per token over 69 KDA layers, against a measured ~12 us per packet.
         *
         * NO SILENT FALLBACK, in the shape this dispatch makes possible: `default:` is a NOP, so
         * the failure to refuse is a packet that convolves two of three streams, or that runs the
         * recurrence against an uninitialised gate, and leaves the rest finite and fluent. Both
         * arms trap on any demoted operand being absent. */
#if PLOW_HAS_KDA_CONV3
        case PLOW_DOP_KDA_CONV3: {
            const float* wv = (in->i[4] == PLOW_TENSOR_NONE) ? nullptr : (const float*)T[in->i[4]];
            float* cs[3];
#pragma unroll
            for (int s = 0; s < 3; s++)
                cs[s] = (in->i[5 + s] == PLOW_TENSOR_NONE) ? nullptr : (float*)T[in->i[5 + s]];
            if (wv == nullptr || cs[0] == nullptr || cs[1] == nullptr || cs[2] == nullptr ||
                TEN(6) == nullptr || TEN(7) == nullptr || in->i[1] == 0u)
                __builtin_trap();
#if PLOW_PACKED_PREFILL_KDA_CONSUMERS
            if (plow_packed_prefill_enabled(prog_meta))
                d_kda_conv3_packed(
                    (bf16*)TEN(0), (bf16*)TEN(1), (bf16*)TEN(2), (const bf16*)TEN(3),
                    (const bf16*)TEN(4), (const bf16*)TEN(5), (const float*)TEN(6),
                    (const float*)TEN(7), wv, cs[0], cs[1], cs[2], in->i[1], in->i[2],
                    in->i[3], slice, nblk, prog_meta);
            else
#endif
                d_kda_conv3((bf16*)TEN(0), (bf16*)TEN(1), (bf16*)TEN(2), (const bf16*)TEN(3),
                        (const bf16*)TEN(4), (const bf16*)TEN(5), (const float*)TEN(6),
                        (const float*)TEN(7), wv, cs[0], cs[1], cs[2], in->i[0], in->i[1],
                        in->i[2], in->i[3], slice, nblk,
                        /* PER-ROW CONV-WINDOW STRIDE, carried in `fj[1].u` (= j[0]).
                         *
                         * NOT `i[4]`. All eight integer slots are operands on this op — the
                         * disassembly names them `T C W act w_v cs_q cs_k cs_v`, so `i[4]` is a
                         * TENSOR INDEX (26/27), not a flags word. Reading it as flags MEASURED as
                         * a GPU memory fault: 26 has bit 2 set, so the flag fired on every conv
                         * packet and the window strided off the end of the state. `i[4]` IS the
                         * flags word on KDA_STATE_STEP (`i4=1` = PLOW_KDA_F_QK_L2NORM), which is
                         * what made that one safe and this one not — the two ops do not share an
                         * operand map and assuming they do is the bug.
                         *
                         * `fj` is untouched by this op's emitter (it sets `i[0..7]` and no `f`/`j`),
                         * so `j[0]` is free and zero in every blob emitted so far — which is the
                         * shared window. The stride is carried DIRECTLY rather than as a flag, so
                         * the kernel does not have to re-derive `C*W`. */
                        (size_t)in->fj[1].u,
                        /* PER-ROW PARKED MASK in `j[1]` (= `fj[2]`), the last free slot on this
                         * op. Gated on `fj[1].u != 0` — the per-row stride — because that is
                         * exactly the seq-rows condition the emitter sets both under, so a blob
                         * emitted before the mask existed reads no mask. */
                        in->fj[1].u ? (const unsigned*)T[in->fj[2].u] : nullptr);
            break;
        }
#endif /* PLOW_HAS_KDA_CONV3 */
#if PLOW_HAS_KDA_STATE_STEP_G
        case PLOW_DOP_KDA_STATE_STEP_G: {
            const float* dtb = (in->i[5] == PLOW_TENSOR_NONE) ? nullptr : (const float*)T[in->i[5]];
            if (dtb == nullptr || TEN(7) == nullptr || TEN(4) == nullptr || TEN(5) == nullptr)
                __builtin_trap();
#if PLOW_PACKED_PREFILL_KDA_CONSUMERS
            if (plow_packed_prefill_enabled(prog_meta))
                d_kda_state_step_packed</*GATED=*/true>(
                    (bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                    (const bf16*)TEN(3), nullptr, nullptr, (const bf16*)TEN(4),
                    (const bf16*)TEN(5), (const float*)TEN(7), dtb, in->i[6], in->fj[1].f,
                    (float*)TEN(6), in->i[1], in->i[2], in->i[3], in->i[4], in->fj[0].f,
                    slice, nblk, (float*)sm->raw, prog_meta);
            else
#endif
                d_kda_state_step_g((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                               (const bf16*)TEN(3), (const bf16*)TEN(4), (const bf16*)TEN(5),
                               (const float*)TEN(7), dtb, in->i[6], in->fj[1].f, (float*)TEN(6),
                               in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], in->fj[0].f,
                               slice, nblk, (float*)sm->raw,
                               /* per-row state stride: one state per row iff the rows are
                                * independent sequences (batched decode), else 0. */
                               (in->i[4] & PLOW_KDA_F_SEQ_ROWS)
                                   ? (size_t)in->i[1] * in->i[2] * in->i[2]
                                   : (size_t)0,
                               /* PER-ROW PARKED MASK in `i[7]`, and read ONLY under the seq-rows
                                * flag. 0 is a valid tensor index, so the flag — not a sentinel —
                                * is what distinguishes "no mask" from "tensor 0"; that is also
                                * what keeps a non-batched blob byte-identical, since the emitter
                                * leaves i[7] at 0 there. */
                               (in->i[4] & PLOW_KDA_F_SEQ_ROWS)
                                   ? (const unsigned*)T[in->i[7]]
                                   : nullptr);
            break;
        }
#endif /* PLOW_HAS_KDA_STATE_STEP_G */
#endif
#if PLOW_KDA_CONV_STEP_DB
#if PLOW_HAS_KDA_CONV_STATE_STEP_G
        case PLOW_DOP_KDA_CONV_STATE_STEP_G: {
            const unsigned* desc = (const unsigned*)TEN(7);
            if (desc == nullptr || in->i[0] != 1) __builtin_trap();
#pragma unroll
            for (unsigned i = 0; i < 12; ++i)
                if (desc[i] == PLOW_TENSOR_NONE) __builtin_trap();
            if (in->i[4] & PLOW_KDA_F_SEQ_ROWS) {
                if (desc[12] == PLOW_TENSOR_NONE) __builtin_trap();
                if (((const unsigned*)T[desc[12]])[0]) break;
            }
            const unsigned bank = ((const unsigned*)T[desc[11]])[0] & 1u;
            const unsigned source = 3u + bank * 3u;
            const unsigned target = 3u + (bank ^ 1u) * 3u;
            d_kda_conv_state_step_g(
                (bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                (const bf16*)TEN(3), (const bf16*)TEN(4), (const bf16*)TEN(5),
                (float*)TEN(6), (const float*)T[desc[0]], (const float*)T[desc[1]],
                (const float*)T[desc[2]], (const float*)T[desc[source]],
                (const float*)T[desc[source + 1]], (const float*)T[desc[source + 2]],
                (float*)T[desc[target]], (float*)T[desc[target + 1]],
                (float*)T[desc[target + 2]], (const float*)T[desc[9]],
                (const float*)T[desc[10]], in->i[1], in->i[2], in->i[3], in->i[5],
                in->i[4], in->i[6], in->fj[0].f, in->fj[1].f, slice, nblk,
                (float*)sm->raw);
            break;
        }
#endif /* PLOW_HAS_KDA_CONV_STATE_STEP_G */
#endif
        /* Kimi-K3 BLOCK structure (op_k3.h). Both arms are in BOTH buckets: a K3 layer runs the
         * same graph at T=1 and T>1, and AttnRes in particular replaces the residual add on the
         * PREFILL path too — an AttnRes that exists only in the decode object would make prefill
         * and decode compute different models, which is the failure the KDA gate's
         * prefill/decode-agreement check exists to catch. */
        /* i2 = nb (rows LIVE at this layer), i4 = the ring's row CAPACITY. They coincide at T = 1
         * and only there; the arm poisons on i4 < i2 rather than striding a T-row ring by the
         * live count. */
        /* t5 = the FUSED post-norm's gamma, and PLOW_TENSOR_NONE means "leave the raw mix", which
         * is the pre-fusion arm byte for byte. See d_attn_res' note: every AttnRes in a K3 program
         * is read by exactly one RMSNORM over its own output, so the two narrow packets collapse. */
#if PLOW_HAS_ATTN_RES
        case PLOW_DOP_ATTN_RES:
#if PLOW_MATERIALIZED_RESIDUAL_INPUT
            if (in->t[6] != PLOW_TENSOR_NONE) {
                if (in->t[7] == PLOW_TENSOR_NONE) __builtin_trap();
                d_materialize_residual(
                    (bf16*)TEN(1), (const bf16*)TEN(6), (const bf16*)TEN(7),
                    in->i[5] == PLOW_TENSOR_NONE ? nullptr : (const bf16*)T[in->i[5]], in->i[0],
                    in->i[1], slice, nblk);
            }
#endif
            d_attn_res((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                       (const float*)TEN(3), in->i[0], in->i[1], in->i[2], in->i[4], in->fj[0].f,
                       slice, nblk, (float*)sm->raw, (const bf16*)TEN(4), in->i[3],
                       (const bf16*)TEN(5));
            break;
#endif /* PLOW_HAS_ATTN_RES */
#if PLOW_HAS_SITU_GLU
        case PLOW_DOP_SITU_GLU:
            d_situ_glu((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), in->i[0],
                       in->fj[0].f, in->fj[1].f, slice, nblk);
            break;
#endif /* PLOW_HAS_SITU_GLU */
#if PLOW_HAS_MLA_OUT_GATE
        case PLOW_DOP_MLA_OUT_GATE:
            d_mla_out_gate((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), in->i[0], slice,
                           nblk);
            break;
#endif /* PLOW_HAS_MLA_OUT_GATE */
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
            /* t0=xq(fp8) t1=x(bf16) t2=a_scale(f32[M])   i0=M i1=K
             * T11 GLU FOLD: t3=gate t4=up i2=act (NONE/0 on every legacy packet). With them set
             * `x` is an OUTPUT — this packet also computes fu = act(gate)*up, because the emitter
             * DROPPED the Glu packet that used to. Ignoring t3/t4 here (which this dispatch did)
             * quantizes an `fu` nothing wrote: fluent wrong tokens, no fault. See
             * `plow_t11_gluquant_arm` and op_gemm.h. */
            d_quant_fp8((unsigned char*)TEN(0), (bf16*)TEN(1), (float*)TEN(2),
                        in->i[0], in->i[1], slice, nblk, (const bf16*)TEN(3),
                        (const bf16*)TEN(4), in->i[2]);
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
#if PLOW_WAVES == 8
        /* The fp4 twin of the bf16 GEMM_GLU arm below, prefill-only for the same reason (a full
         * 256x256 MFMA GEMM has no business in the decode object) and PLOW_WAVES==8 for the same
         * reason (the epilogue's wave->column remap needs SN == 2). It is `d_gemm_mxfp4`'s tile and
         * accumulator with `d_gemm_glu`'s epilogue, so it adds no register class this bucket does
         * not already carry. */
        case PLOW_DOP_GEMM_GLU_MXFP4:
            exec_gemm_glu_mxfp4(in, T, slice, nblk, sm->gm);
            break;
#endif
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
#if PLOW_MLA_PREFILL
        /* MLA / gathered-MLA prefill — the arms whose absence meant Kimi K2.7, DeepSeek and
         * GLM 5.2 could decode on gfx950 but had nothing to run a real prompt through. */
#if PLOW_FP8_KV
        /* Same swap as the decode bucket: the DENSE MLA prefill goes e4m3, the gathered one keeps
         * its bf16 arm (its `t7` is `idx`, not a scale array). */
        case PLOW_DOP_FLASH_MLA_PREFILL_FP8:
            exec_flash_mla_prefill_fp8(in, T, slice, nblk, sm->fmpf, prog_meta);
            break;
        case PLOW_DOP_FLASH_GATHER_PREFILL:
            exec_flash_mla_prefill(in, T, slice, nblk, sm->fmpf, prog_meta);
            break;
#else
        case PLOW_DOP_FLASH_MLA_PREFILL:
        case PLOW_DOP_FLASH_GATHER_PREFILL:
            exec_flash_mla_prefill(in, T, slice, nblk, sm->fmpf, prog_meta);
            break;
#endif
        /* ===== DSA sparse-prefill selection chain (ops 117-119). [GLM52-DSA-PF]
         * Produces the per-query top-k + per-64-query-tile union table the GATHER arm of the
         * V2 flash walks. All three run on this 8-wave prefill object; the gathered flash
         * itself rides the FLASH_MLA_PREFILL packet (t7 = the union table) on the 4-wave
         * flash object — if THIS object sees such a packet (no V2 routing), the dense arm
         * above simply ignores t7 and attends the full causal range: degrade, not corrupt. */
        case PLOW_DOP_INDEX_SCORE_PF: {
            constexpr int DI_ = 128, HI_ = 32; /* same baked geometry as INDEX_SCORE */
#if PLOW_DSA_IDX_ROW
            /* Row-resident arm. NOT a packet change: identical operands, identical arithmetic,
             * BIT-IDENTICAL output (gated byte-for-byte in dsa_pf_indexer_bench) — only the work
             * decomposition differs, so an object built WITHOUT this runs the shipped arm on the
             * same packet and is correct, just slower. That is why this arm carries no
             * `requires` marker: the refuse-at-load machinery exists for arms that change what a
             * packet's operands MEAN, and this one changes nothing a packet carries. */
            d_index_score_pf_row<DI_, HI_, IDXPF_TILE_N_DEF>(
                (float*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(3),
                (const int*)TEN(4), in->i[0], in->i[2], in->fj[0].f, slice, nblk, (bf16*)sm->raw);
#else
            d_index_score_pf<DI_, HI_>((float*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                       (const bf16*)TEN(3), (const int*)TEN(4), in->i[0],
                                       in->i[2], in->fj[0].f, slice, nblk);
#endif
        } break;
        case PLOW_DOP_INDEX_SELECT_PF:
            /* The fewer-passes early-out is EXACT (it fires only when the index bits cannot
             * change the emitted set), so it rides the same default-on/kill-switch axis. */
            d_index_select_pf<PLOW_DSA_IDX_ROW != 0>(
                (int*)TEN(0), (const float*)TEN(1), (const int*)TEN(2), in->i[0],
                              in->i[1], in->i[2], slice, nblk, (unsigned*)sm->raw,
                              (unsigned*)sm->raw + 256);
            break;
        case PLOW_DOP_INDEX_UNION_PF:
            /* i4 = queries per union tile (0 = legacy 64; the B2 head-batched walk emits 8).
             * The umask scratch is SLICE-indexed inside the kernel, so its size is bound by
             * this packet's block count, not the tile count. */
            d_index_union_pf((unsigned char*)TEN(0), (unsigned long long*)TEN(1),
                             (const int*)TEN(2), (const int*)TEN(3), in->i[0], in->i[1],
                             in->i[2], in->i[3], in->i[4], slice, nblk, (unsigned*)sm->raw);
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
        /* FUSED Q|K|V, per-channel fp8 — op 22's twin, sharing op 30's body (that one is this with
         * Nk=Nv=0). The three f32 scale rows are handles in i5/i6/i7; the arm TRAPS on a malformed
         * packet rather than running a narrower sweep, because `default:` here writes nothing. */
        case PLOW_DOP_GEMV_QKV_FP8:
            exec_gemv_qkv_fp8(in, T, slice, nblk, sm->gm);
            break;
#endif
#if PLOW_MXFP4
        case PLOW_DOP_GEMV_MXFP4:
            exec_gemv_mxfp4(in, T, slice, nblk, sm->gm);
            break;
        case PLOW_DOP_GEMV_GLU_MXFP4:
            exec_gemv_glu_mxfp4(in, T, slice, nblk, sm->gm);
            break;
        /* FUSED Q|K|V, fp4 — op 22's twin, sharing op 91's body (that one is this with Nk=Nv=0).
         * The three E8M0 scale rows are handles in i5/i6/i7; the arm TRAPS on a malformed packet
         * rather than running a narrower sweep, because `default:` here writes nothing. */
        case PLOW_DOP_GEMV_QKV_MXFP4:
            exec_gemv_qkv_mxfp4(in, T, slice, nblk, sm->gm);
            break;
#endif
        case PLOW_DOP_GEMV_GLU:
            /* f[0]/f[1] carry the situ betas, exactly as PLOW_DOP_SITU_GLU does. They are 0 on
             * every non-K3 packet, and act != 2 ignores them. */
            d_gemv_glu((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                       (const bf16*)TEN(5), in->i[0], in->i[1], in->i[2], in->i[5], slice, nblk,
                       sm->gm, in->fj[0].f, in->fj[1].f);
            break;
        case PLOW_DOP_GEMV_QKV:
            /* t7 PRESENT = THE Q-NORM FOLD (emit knob PLOW_GLM_FUSE_QNORM): t1 then carries the
             * RAW q_a projection and this packet computes `q_a_layernorm` into its own LDS
             * staging, deleting the one-workgroup RmsNorm that used to sit between the two
             * GemvQkvs. f0 is eps. Op 22 has never read t7, so the discriminator is
             * unambiguous — op 108 GemvQkvg's t7 is g_out, a DIFFERENT opcode.
             *
             * Under an object built WITHOUT the arm this is a null pointer into a branch that
             * was not compiled, i.e. the pre-fold reading, which is why plowrt refuses the
             * combination on the marker symbol rather than letting it serve. */
            d_gemv_qkv((bf16*)TEN(0), (bf16*)TEN(3), (bf16*)TEN(5), (const bf16*)TEN(1),
                       (const bf16*)TEN(2), (const bf16*)TEN(4), (const bf16*)TEN(6), in->i[0],
                       in->i[1], in->i[3], in->i[4], in->i[2], slice, nblk, sm->gm,
                       (const bf16*)TEN(7), in->fj[0].f);
            break;
        /* FUSED Q|K|V|G — op 22 plus a fourth output stream (Kimi-K3 KDA q/k/v/g_proj).
         *
         * t7 = g_out, i5 = Ng, i6 = the W_g HANDLE (nine pointers do not fit t[8]; dev_isa.h
         * states why the demoted one is a weight). Shares op 22's body: see op_gemm.h.
         *
         * NO SILENT FALLBACK. This interpreter's dispatch `default:` is a NOP, so the one thing
         * that must not happen is a malformed 4-stream packet quietly running as a 3-stream sweep
         * and leaving g_out exactly as it found it — finite, fluent, wrong. A packet that names
         * this opcode without both halves of the fourth stream TRAPS. */
        case PLOW_DOP_GEMV_QKVG: {
            const bf16* Wg = (in->i[6] == PLOW_TENSOR_NONE) ? nullptr : (const bf16*)T[in->i[6]];
            if (in->i[5] == 0u || Wg == nullptr || TEN(7) == nullptr) __builtin_trap();
            d_gemv_qkvg((bf16*)TEN(0), (bf16*)TEN(3), (bf16*)TEN(5), (bf16*)TEN(7),
                        (const bf16*)TEN(1), (const bf16*)TEN(2), (const bf16*)TEN(4),
                        (const bf16*)TEN(6), Wg, in->i[0], in->i[1], in->i[3], in->i[4], in->i[5],
                        in->i[2], slice, nblk, sm->gm);
            break;
        }
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
#if PLOW_FP8_KV
        /* fp8-KV object: the DENSE latent flash is SWAPPED for its e4m3 twin, exactly as
         * FLASH_DECODE is above — the KV encoding axis is a swap, not an additive one.
         *
         * FLASH_GATHER_DECODE is NOT swapped and stays bf16 in this object. Its `t7` is the `idx`
         * table and `t7` is where the fp8 dequant scales live, so a gathered fp8 latent has
         * nowhere to put them; keeping the bf16 gather arm compiled means a DSA packet meets a
         * REAL arm instead of the silent-NOP `default:`. The cost is that the bf16 instantiation
         * is still in this object, so the swap buys no register headroom for MLA — it buys the
         * halved KV, which is the point. */
        case PLOW_DOP_FLASH_MLA_DECODE_FP8:
            exec_flash_mla_decode_fp8(in, T, slice, nblk, PLOW_FMLA(sm));
            break;
        case PLOW_DOP_FLASH_GATHER_DECODE:
            exec_flash_mla_decode(in, T, slice, nblk, PLOW_FMLA(sm));
            break;
#else
        case PLOW_DOP_FLASH_MLA_DECODE:
        case PLOW_DOP_FLASH_GATHER_DECODE:
            exec_flash_mla_decode(in, T, slice, nblk, PLOW_FMLA(sm));
            break;
#endif
        case PLOW_DOP_O_UV_FOLD:
            d_o_uv_fold<512>((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2), in->i[0],
                             in->i[1], in->i[2], slice, nblk, sm->raw);
            break;
        /* FUSED MLA merge+fold (replaces FLASH_MERGE<512> + O_UV_FOLD): merges the nsplit latent
         * partials in LDS + folds W_uv straight to v_head. VT is chosen per dispatch — see
         * exec_mla_merge_fold. olds + the fold's cross-wave scratch reuse the raw arena
         * (DK + PLOW_WAVES*VT floats, well inside PLOW_SMEM_HALVES; PLOW_MLA_FOLD_TB_LDS
         * floats when the token-blocked arm is built, which the assert above bounds). */
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
        /* ===== MoE data-dependent counter-gate. Decode-path
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
        case PLOW_DOP_MOE_EXPERT_GLU: /* f0/f1 = situ betas, as on the block-fp8 twin */
            d_moe_expert_glu((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                             (const unsigned long long*)TEN(3), in->i[0], in->i[1], in->i[2],
                             in->i[3], in->i[5], slice, nblk, in->fj[0].f, in->fj[1].f);
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
#if PLOW_MOE_GEMMA
        /* ===== Gemma-4 26B-A4B sparse-MoE DECODE (ops 61-72). [GEMMA4-MOE-AMD] =====
         *
         * THESE ARE NOT THE GENERIC MoE ARMS ABOVE, and the difference is the ROUTER: Gemma's
         * carries a weightless-RMS scalar, a per-channel `scale[H]`, a `root` exponent and a
         * PER-EXPERT gate scale (`pes`), against `d_moe_router`'s pre-normalised x and single
         * `route_scale`. Its expert table is TWO ULLs per expert with gate and up FUSED, not
         * three with them separate. Routing a Gemma packet through the generic arms compiles and
         * computes a different model — see the family header in op_moe.h.
         *
         * BATCH B > 1: the row count rides a spare immediate, 0 and 1 both meaning one row, so a
         * B=1 packet is byte-identical to a pre-batch one. Every op's per-row body is the B=1
         * body, so a batched row is bit-identical to its own single-row result. */
#define PLOW_NROW(v) ((v) ? (v) : 1u)
        /* t0=table t1=resid t2=proj t3=scale t4=pes; i0=H i1=n_exp i2=k i3=B; f0=root f1=eps */
        case PLOW_DOP_MOE_ROUTER_GEMMA:
            d_moe_router_gemma((unsigned char*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                               (const bf16*)TEN(3), (const bf16*)TEN(4), in->i[0], in->i[1],
                               in->i[2], in->fj[0].f, in->fj[1].f, slice, nblk,
                               PLOW_NROW(in->i[3]), (float*)sm->raw);
            break;
        /* t0=score(f32[B,E]) t1=resid t2=proj t3=scale; i0=H i1=n_exp i2=B; f0=root f1=eps */
        case PLOW_DOP_MOE_ROUTER_GEMMA_SCORE:
            d_moe_router_gemma_score((float*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                     (const bf16*)TEN(3), in->i[0], in->i[1], in->fj[0].f,
                                     in->fj[1].f, slice, nblk, PLOW_NROW(in->i[2]),
                                     (float*)sm->raw);
            break;
        case PLOW_DOP_MOE_ROUTER_GEMMA_SCORE_FAST:
            d_moe_router_gemma_score_fast((float*)TEN(0), (const bf16*)TEN(1),
                                          (const bf16*)TEN(2), (const bf16*)TEN(3), in->i[0],
                                          in->i[1], in->fj[0].f, in->fj[1].f, slice, nblk,
                                          PLOW_NROW(in->i[2]), (float*)sm->raw);
            break;
        /* t0=table t1=score t2=pes; i1=n_exp i2=k i3=B */
        case PLOW_DOP_MOE_ROUTER_GEMMA_TOPK:
            d_moe_router_gemma_topk((unsigned char*)TEN(0), (const float*)TEN(1),
                                    (const bf16*)TEN(2), in->i[1], in->i[2], slice, nblk,
                                    PLOW_NROW(in->i[3]), (float*)sm->raw);
            break;
        /* t0=fu t1=x t2=table t3=ewt; i0=k i1=I_moe i2=H i3=n_exp i4=slot_off i5=B.
         * i4 shifts fu/table by whole SLOTS — the split-expert emission gives each packet its
         * own slice of the k slots; it is 0 on the fused emission. */
        case PLOW_DOP_MOE_EXPERT_GLU_GEMMA:
            d_moe_expert_glu_gemma((bf16*)TEN(0) + (size_t)in->i[4] * in->i[1],
                                   (const bf16*)TEN(1),
                                   (const unsigned char*)TEN(2) + (size_t)in->i[4] * 8u,
                                   (const unsigned long long*)TEN(3), in->i[0], in->i[1],
                                   in->i[2], in->i[3], slice, nblk, PLOW_NROW(in->i[5]));
            break;
        /* t0=part t1=fu t2=table t3=ewt; i0=k i1=H i2=I_moe i3=n_exp i4=slot_off i5=B */
        case PLOW_DOP_MOE_EXPERT_DOWN_GEMMA:
            d_moe_expert_down_gemma((float*)TEN(0) + (size_t)in->i[4] * in->i[1],
                                    (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                                    (const unsigned char*)TEN(2) + (size_t)in->i[4] * 8u,
                                    (const unsigned long long*)TEN(3), in->i[0], in->i[1],
                                    in->i[2], in->i[3], slice, nblk, PLOW_NROW(in->i[5]));
            break;
        /* t0=moe t1=part; i0=H i1=k */
        case PLOW_DOP_MOE_COMBINE_GEMMA:
            d_moe_combine_gemma((bf16*)TEN(0), (const float*)TEN(1), in->i[0], in->i[1], slice,
                                nblk);
            break;
        /* t0=out t1=part t2=resid t3=gamma; i0=H i1=k i2=B; f0=eps */
        case PLOW_DOP_MOE_COMBINE_NORM_GEMMA:
            d_moe_combine_norm_gemma((bf16*)TEN(0), (const float*)TEN(1), (const bf16*)TEN(2),
                                     (const bf16*)TEN(3), in->i[0], in->i[1], in->fj[0].f, slice,
                                     nblk, PLOW_NROW(in->i[2]), (float*)sm->raw);
            break;
        /* t0=hn t1=x(in/out) t2=part t3=h1 t4=g_pf2 t5=g_po t6=gn; i0=H i1=k; f0=eps f1=ls */
        case PLOW_DOP_MOE_COMBINE_RESID_NORM_GEMMA:
            d_moe_combine_resid_norm_gemma((bf16*)TEN(0), (bf16*)TEN(1), (const float*)TEN(2),
                                           (const bf16*)TEN(3), (const bf16*)TEN(4),
                                           (const bf16*)TEN(5), (const bf16*)TEN(6), in->i[0],
                                           in->i[1], in->fj[0].f, in->fj[1].f, slice,
                                           (float*)sm->raw);
            break;
        /* t0=fu t1=resid t2=table t3=ewt t4=gamma; i0=k i1=I i2=H i3=n_exp i5=B; f0=eps */
        case PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA:
            d_moe_expert_glu_norm_gemma((bf16*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(4),
                                        (const unsigned char*)TEN(2),
                                        (const unsigned long long*)TEN(3), in->i[0], in->i[1],
                                        in->i[2], in->i[3], in->fj[0].f, slice, nblk,
                                        PLOW_NROW(in->i[5]), sm->raw);
            break;
#if PLOW_FP8
        /* Per-output-channel e4m3 experts. t4=est ({gate_up scales [2I], down scales [H]} per
         * expert). PLOW_FP8 because that is the WEIGHT-encoding axis these belong to; the bf16
         * arms above stay compiled in alongside them, additively, as everywhere else here. */
        case PLOW_DOP_MOE_EXPERT_GLU_GEMMA_FP8:
            d_moe_expert_glu_gemma_fp8((bf16*)TEN(0) + (size_t)in->i[4] * in->i[1],
                                       (const bf16*)TEN(1),
                                       (const unsigned char*)TEN(2) + (size_t)in->i[4] * 8u,
                                       (const unsigned long long*)TEN(3),
                                       (const unsigned long long*)TEN(4), in->i[0], in->i[1],
                                       in->i[2], in->i[3], slice, nblk, PLOW_NROW(in->i[5]));
            break;
        case PLOW_DOP_MOE_EXPERT_DOWN_GEMMA_FP8:
            d_moe_expert_down_gemma_fp8((float*)TEN(0) + (size_t)in->i[4] * in->i[1],
                                        (const bf16*)TEN(1) + (size_t)in->i[4] * in->i[2],
                                        (const unsigned char*)TEN(2) + (size_t)in->i[4] * 8u,
                                        (const unsigned long long*)TEN(3),
                                        (const unsigned long long*)TEN(4), in->i[0], in->i[1],
                                        in->i[2], in->i[3], slice, nblk, PLOW_NROW(in->i[5]));
            break;
#endif /* PLOW_FP8 */
#undef PLOW_NROW
#endif /* PLOW_MOE_GEMMA */
        /* GLM dense layers (0-2): block-fp8 SwiGLU gate/up (op 47) + block-fp8 dense DOWN (op 44).
         * These live in the UNCONDITIONAL MoE region (like the expert ops 45/46) so the GLM decode
         * object dispatches them without the PLOW_FP8 object. There USED to be a second, identical
         * `case PLOW_DOP_GEMV_FP8_BLK` under `#if PLOW_FP8` a hundred lines up, the two kept
         * disjoint by `#if !PLOW_FP8` here — a duplicate that only survived because the two guards
         * partitioned the builds. Now that PLOW_FP8 is additive the partition is gone and so is the
         * duplicate: op 44 is a BLOCK-fp8 weight arm, a third weight encoding independent of the
         * per-channel fp8 the PLOW_FP8 axis names, so it belongs here unconditionally. */
        case PLOW_DOP_DENSE_GLU_FP8_BLK: /* t2=Wg t5=Wu t3=Sg t4=Su; i0=N(inter) i1=K(hidden) i5=act
                                          * f0/f1 = situ betas (K3's dense FFN + shared experts) */
            d_dense_glu_fp8_blk((bf16*)TEN(0), (const bf16*)TEN(1), (const unsigned char*)TEN(2),
                                (const unsigned char*)TEN(5), (const float*)TEN(3),
                                (const float*)TEN(4), in->i[0], in->i[1], in->i[5], slice, nblk,
                                in->fj[0].f, in->fj[1].f);
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
        /* THE GROUPED EXPERT CHAIN IS AN FFN AXIS, NOT A BUCKET-PREFILL ONE — which is why it
         * sits AFTER the PLOW_BUCKET_PREFILL block rather than inside it.
         *
         * A BATCHED DECODE runs these five packets too: at PLOW_DECODE_BATCH > 1 the decode
         * program carries B independent sequences, and grouping by expert is what makes an
         * expert's weights cross HBM once for every row that chose it — the memory-bound win of
         * batching, and correct because MoE has no cross-row state. That object defines
         * PLOW_MOE_PREFILL but NOT PLOW_BUCKET_PREFILL (it has no bucket ladder), so nesting
         * these inside the bucket block left all 92 MoE layers falling through the dispatch
         * `default:` and writing NOTHING — silently, since this interpreter does not trap.
         *
         * Every object that had these before defines both symbols, so hoisting them here is a
         * no-op for all of them. */
#if PLOW_MOE_PREFILL
        /* MoE PREFILL (T>1) — the FFN half of an MLA-family prompt. Gated by its own flag, and
         * not folded into PLOW_MLA_PREFILL, for the reason every other axis here has its own
         * flag: this adds a second full MFMA GEMM body to a bucket that already sits at the
         * 256/occ-2 cliff, and a Gemma or Qwen prefill object must not pay for it. Measured
         * cost is reported in the build table. */
        case PLOW_DOP_MOE_ROUTER_TOPK_PF:
            /* t2/i0 = PLOW_MOE_PF_ATOMIC's [T, i0] f32 accumulator to zero before the MoE chain
             * runs (TENSOR_NONE / 0 on every other blob). See the PLOW_MOE_PF_ATOMIC header. */
            d_moe_router_topk_pf((unsigned char*)TEN(0), (const bf16*)TEN(1),
                                 (in->i[3] & 4u) ? (const float*)TEN(3) : nullptr, in->i[1],
                                 in->i[2], in->i[3], in->fj[0].f, in->i[4], slice, nblk,
                                 (float*)sm->raw, in->i[6], in->i[7]
#if PLOW_MOE_PF_ATOMIC
                                 ,
                                 (float*)TEN(2), in->i[0]
#endif
#if PLOW_MOE_PF_DET
                                 /* t2/i0 = PLOW_MOE_PF_DET's [T, i0] f64 fixed-point accumulator
                                  * to zero before the MoE chain runs. */
                                 ,
                                 (double*)TEN(2), in->i[0]
#endif
            );
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
                               in->i[5], slice, nblk, in->fj[0].f, in->fj[1].f, sm->raw, nullptr,
                               (const unsigned*)TEN(6), (unsigned char*)TEN(7),
                               /* i6 = weight-table layout: 1 = preshuffled B'[K/64][N][64]
                                * (PLOW_MOE_PF_SHUF second slab); 0 = row-major, every
                                * existing blob. */
                               in->i[6],
                               /* i7 = a8: t6/t7 are fp8 A rows + per-token f32 scales
                                * (PLOW_MOE_PF_A8). 0 = bf16 A, every existing blob. */
                               in->i[7]);
            break;
        case PLOW_DOP_MOE_GROUP_DOWN_PF:
            d_moe_group_down_pf((float*)TEN(0), (const bf16*)TEN(1),
                                (const unsigned long long*)TEN(2),
                                (const unsigned long long*)TEN(3), (const int*)TEN(4),
                                (const unsigned*)TEN(6), (const float*)TEN(7), in->i[0], in->i[1],
                                in->i[2], in->i[3], slice, nblk, sm->raw,
                                (const unsigned char*)TEN(5), in->i[6] /* layout, see GLU */,
                                /* i7 = part16: the part scatter is bf16 (PLOW_MOE_PF_PART16).
                                 * 0 = f32, every existing blob. */
                                in->i[7]
#if PLOW_MOE_PF_ATOMIC
                                /* i4 = PLOW_MOE_PF_ATOMIC: log2(k)+1, and t0 is the [T,H] f32
                                 * accumulator rather than part[T*k,H]. 0 = the shipped
                                 * scatter, every existing blob. */
                                ,
                                in->i[4]
#endif
#if PLOW_MOE_PF_DET
                                /* i5 = PLOW_MOE_PF_DET: log2(k)+1, and t0 is the [T,H] f64
                                 * FIXED-POINT accumulator. A SEPARATE field from the atomic
                                 * arm's i4 so a blob can never be read as the other arm. */
                                ,
                                in->i[5]
#endif
            );
            break;
        case PLOW_DOP_MOE_COMBINE_PF: {
            /* i3 = t_row0: this packet combines token rows [i3, i3+i2). 0 = the whole chunk
             * (every existing blob). Pure pointer arithmetic — the banded TP seam splits the
             * combine so each band's two-shot can start as soon as ITS rows are combined. */
            const size_t r0 = in->i[3];
            const bf16* res = (const bf16*)TEN(1);
            const bf16* sh = (const bf16*)TEN(2);
            /* i7 = part16 (PLOW_MOE_PF_PART16): part is bf16, so the banding offset advances
             * by 2-byte elements, not 4. Same element count either way. */
            const float* partp =
                in->i[7] ? (const float*)((const bf16*)TEN(3) + r0 * in->i[1] * in->i[0])
                         : ((const float*)TEN(3) + r0 * in->i[1] * in->i[0]);
#if PLOW_MOE_PF_DET
            /* i4 = det: `part` is f64, so the BAND offset advances by 8-byte elements. Same
             * element count; getting this wrong would band into the middle of a double. */
            if (in->i[4])
                partp = (const float*)((const double*)TEN(3) + r0 * in->i[1] * in->i[0]);
#endif
            d_moe_combine_pf((bf16*)TEN(0) + r0 * in->i[0],
                             res ? res + r0 * in->i[0] : nullptr,
                             sh ? sh + r0 * in->i[0] : nullptr,
                             partp, in->i[0], in->i[1],
                             in->i[2], slice, nblk, in->i[7]
#if PLOW_MOE_PF_DET
                             /* i4 = PLOW_MOE_PF_DET: `part` is the f64 fixed-point accumulator
                              * op 86 summed in place, read as ONE contiguous stream (i1 == 1). */
                             ,
                             in->i[4]
#endif
            );
        } break;
#endif
#if PLOW_MOE_GEMMA_PF
        /* ===== Gemma-4 26B-A4B grouped-MoE PREFILL (ops 73-77, 81/82). [GEMMA4-MOE-AMD] =====
         *
         * OUTSIDE the bucket block, for the reason the PLOW_MOE_PREFILL family above is: a
         * BATCHED DECODE runs the grouped chain too (grouping by expert is what makes an
         * expert's weights cross HBM once per row that chose it), and that object defines the
         * MoE axis without PLOW_BUCKET_PREFILL. Nesting these inside the bucket would leave
         * every MoE layer falling through `default:` and writing NOTHING — silently.
         *
         * Its OWN flag, not PLOW_MOE_PREFILL's: this is a second full MFMA body, Gemma's fused
         * gate_up table is a different indexing rule from the three-pointer one ops 85/86 use,
         * and a Kimi/GLM prefill object must not carry a Gemma GEMM it can never reach. */
        /* t0=table t1=resid([T,H]) t2=proj t3=scale t4=pes; i0=H i1=n_exp i2=k i3=T;
         * f0=root f1=eps. Block-per-token loop of the decode router — bit-identical per token. */
        case PLOW_DOP_MOE_ROUTER_GEMMA_PF:
            d_moe_router_gemma_pf((unsigned char*)TEN(0), (const bf16*)TEN(1), (const bf16*)TEN(2),
                                  (const bf16*)TEN(3), (const bf16*)TEN(4), in->i[0], in->i[1],
                                  in->i[2], in->i[3], in->fj[0].f, in->fj[1].f, slice, nblk,
                                  (float*)sm->raw);
            break;
        /* t0=meta t1=table t2=row_token t3=row_partidx t4=row_gate; i0=T i1=n_exp i2=k.
         *
         * THE SAME BODY as op 84, deliberately. Histogram, MPF_BM-padded prefix, meta layout and
         * (token, part-row, gate) scatter are identical between the two families — the Gemma
         * routing table has the same 8-byte {u32 id, f32 gate} slot layout as every other one in
         * this file. Sharing it is what keeps the pad convention, and therefore both grouped
         * GEMMs' tile arithmetic, provably consistent instead of consistent by inspection. */
        case PLOW_DOP_MOE_ALIGN_GEMMA_PF:
            d_moe_align_pf((int*)TEN(0), (const unsigned char*)TEN(1), (unsigned*)TEN(2),
                           (unsigned*)TEN(3), (float*)TEN(4), in->i[0], in->i[1], in->i[2], slice,
                           (unsigned*)sm->raw);
            break;
        /* t0=fu_g t1=xn2([T,H]) t2=ewt t3=meta t4=row_token; i0=I_moe(N) i1=H(K) i2=n_exp i5=act */
        case PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF:
            d_moe_group_glu_gemma_pf((bf16*)TEN(0), (const bf16*)TEN(1),
                                     (const unsigned long long*)TEN(2), (const int*)TEN(3),
                                     (const unsigned*)TEN(4), in->i[0], in->i[1], in->i[2],
                                     in->i[5], slice, nblk, sm->raw);
            break;
        /* t0=part t1=fu_g t2=ewt t3=meta t4=row_partidx t5=row_gate; i0=H(N) i1=I_moe(K) i2=n_exp */
        case PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF:
            d_moe_group_down_gemma_pf((float*)TEN(0), (const bf16*)TEN(1),
                                      (const unsigned long long*)TEN(2), (const int*)TEN(3),
                                      (const unsigned*)TEN(4), (const float*)TEN(5), in->i[0],
                                      in->i[1], in->i[2], slice, nblk, sm->raw);
            break;
        /* t0=out t1=part([T,k,H]) t2=h1 t3=gamma; i0=H i1=k i2=T; f0=eps */
        case PLOW_DOP_MOE_COMBINE_NORM_GEMMA_PF:
            d_moe_combine_norm_gemma_pf((bf16*)TEN(0), (const float*)TEN(1), (const bf16*)TEN(2),
                                        (const bf16*)TEN(3), in->i[0], in->i[1], in->i[2],
                                        in->fj[0].f, slice, nblk, (float*)sm->raw);
            break;
#if PLOW_FP8
        /* w8a8 twins. GLU t5=ascale(f32[T]) t6=est; DOWN t6=est t7=fscale(f32[padded rows]). */
        case PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF_W8A8:
            d_moe_group_glu_gemma_pf_w8a8((bf16*)TEN(0), (const unsigned char*)TEN(1),
                                          (const float*)TEN(5), (const unsigned long long*)TEN(2),
                                          (const unsigned long long*)TEN(6), (const int*)TEN(3),
                                          (const unsigned*)TEN(4), in->i[0], in->i[1], in->i[2],
                                          in->i[5], slice, nblk, sm->raw);
            break;
        case PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF_W8A8:
            d_moe_group_down_gemma_pf_w8a8((float*)TEN(0), (const unsigned char*)TEN(1),
                                           (const float*)TEN(7), (const unsigned long long*)TEN(2),
                                           (const unsigned long long*)TEN(6), (const int*)TEN(3),
                                           (const unsigned*)TEN(4), (const float*)TEN(5), in->i[0],
                                           in->i[1], in->i[2], slice, nblk, sm->raw);
            break;
#endif /* PLOW_FP8 */
#endif /* PLOW_MOE_GEMMA_PF */
        default: /* PLOW_DOP_NOP */
            break;
    }
#endif /* PLOW_BUCKET_FLASH */
#undef PLOW_PACKED_NORM_ARGS
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

/* ---------------------------------------------------------------------------------------
 * PLOW_GATE_HIER_CEIL — TWO-LEVEL CACHE MAINTENANCE, CEILING INSTRUMENT. NEVER SHIP.
 *
 * `runtime/bench/ctr_convergence.hip` measured where an empty b=256 packet's 13.2 us goes, and
 * it is not the counter and not the atomic: the release RMW is a posted `global_atomic_add` at
 * 0.07-0.14 us in EVERY arm. What costs is that all 256 workgroups issue `buffer_wbl2` and
 * `buffer_inv`, which are PER-L2, so each XCD's L2 performs the SAME writeback and the SAME
 * invalidate 32 times and they serialise. Re-measured under lease 2026-07-30:
 *
 *     BASELINE  1 counter, release signal + agent acquire      13.16 us
 *     FAN=32    32 counters                                    10.20
 *     FAN=256   one counter per workgroup, zero contention      8.95
 *     NOACQ     drop buffer_inv                                 9.16
 *     RELAXSIG  drop buffer_wbl2                                5.49
 *     HIER2     per-XCD leader does the maintenance              3.46   <- this
 *
 * HIER2 does the maintenance ONCE PER XCD instead of once per workgroup. The gfx950 dispatcher
 * is exactly round-robin over the 8 XCDs (`runtime/tests/xcd_map_gfx950_test.hip` measures
 * 100.0%), so blocks 0..7 are one per XCD and `blockIdx.x < 8` elects one leader each.
 *
 * THIS KNOB IS THE UNSAFE HALF — the saving with the ORDERING SIMPLY OMITTED, which is what
 * `LEADER_MAINT` is to `HIER2` in the microbench. A follower neither writes back nor
 * invalidates, and nothing makes it wait for the leader that does, so a consumer can read a
 * line the producer's XCD has not published. It exists to price the idea END TO END ON THE REAL
 * MODEL before the two XCD-local rendezvous that make it sound are built, because those need a
 * per-(packet, XCD) arrival count that the GLOBAL QUEUE does not have: a workgroup claims
 * whatever entry is next, so which slices of a packet land on which XCD is decided at run time.
 * Fleet (arXiv 2604.15379) resolves exactly this on MI350 by electing "the last worker per XCD"
 * dynamically, which is the shape the sound version should take.
 *
 * Expect the ceiling to be SMALLER than the microbench's 3.8x: that bench is an empty packet in
 * a tight chain with nothing to overlap, and the real per-packet gate is 2.255 us (traced), not
 * 13.16. If this knob does not move the token, the sound version cannot either. */
#ifndef PLOW_GATE_HIER_CEIL
#define PLOW_GATE_HIER_CEIL 0
#endif

/* ---------------------------------------------------------------------------------------
 * PLOW_GATE_HIER — the SOUND two-level cache maintenance. This is the shippable form of what
 * PLOW_GATE_HIER_CEIL prices, and it differs from it in exactly one way: the ordering is real.
 *
 * The ceiling knob elects a leader and lets followers skip the maintenance with nothing making
 * them wait for the leader that does it. This one adds the two XCD-local rendezvous that make
 * that safe, using the per-(packet, domain) slice count the emitter packs into PLOW_SE_NPER:
 *
 *   publish   follower: __syncthreads() retires its stores into THIS XCD's L2, then bumps
 *                       ldn[p][d] RELAXED — no writeback, the data is in the L2 the leader is
 *                       about to write back.
 *             leader:   the workgroup whose bump returns nper-1 issues ONE release RMW per
 *                       successor, adding nper rather than 1. Its buffer_wbl2 writes back the
 *                       whole L2 — which is exactly the followers' stores — and publishes. The
 *                       global counter still reaches `blocks[p]`, so no consumer's threshold
 *                       changes and the graph is untouched.
 *   observe   leader:   the workgroup whose bump of arr[c][d] returns 0 polls the global
 *                       counters, takes ONE agent acquire (buffer_inv sc1, L1+L2) for the whole
 *                       XCD, then bumps opn[c][d] to release its followers.
 *             follower: waits opn[c][d] and NEVER POLLS THE GLOBAL COUNTER AT ALL — less fabric
 *                       traffic, not more — then invalidates only its own L1.
 *
 * WHY FOLLOWERS STILL INVALIDATE, and why the microbench's version would have been unsound.
 * `ctr_convergence.hip`'s HIER2 arm reasons entirely about L2 ("this XCD's L2 has been fresh
 * ever since") and lets followers do nothing. But L1 IS PER-CU, not per-XCD — the same fact
 * `ctr_poll`'s comment already records as the reason a workgroup-scope acquire is a race even
 * same-XCD. The leader's `buffer_inv sc1` invalidates the XCD's L2 and the LEADER's L1; a
 * follower on another CU keeps its own stale lines. The microbench cannot see this because its
 * body is empty and it reads no data. So followers issue `buffer_inv` with NO SC bits, which on
 * CDNA4 leaves L2 alone (SC1 is what selects L2) and costs a per-CU L1 invalidate — cheap, and
 * not the per-L2 serialising op the whole exercise exists to remove.
 *
 * REQUIRES PLOW_L2_PLACE_DISPATCH and an L2-placed blob. Without per-domain windows `nper` is
 * meaningless (a workgroup claims whatever entry is next, so the count is a run-time property),
 * and the emitter leaves the field zero — which this code reads as "no hierarchy". */
#if PLOW_GATE_HIER && PLOW_GATE_HIER_CEIL
#error "PLOW_GATE_HIER (sound) and PLOW_GATE_HIER_CEIL (the unsafe ceiling) are exclusive"
#endif

/* The two knobs do not compose, and combined they are silently worse than either. The HIER_CEIL
 * follower hand-rolls its relaxed RMW rather than calling `ctr_signal`, so under SC1 it would
 * also skip the `s_waitcnt vmcnt(0)` that arm performs — 248 of 256 workgroups would bump their
 * counter without retiring their stores at all. `perf-data/archive/k3/k3-hier2-ceiling.md` §2 says to drop
 * SC1 anyway; this makes the combination a build error instead of a debugging session. */
#if PLOW_GATE_HIER_CEIL && PLOW_GATE_SC1
#error "PLOW_GATE_HIER_CEIL and PLOW_GATE_SC1 are mutually exclusive (see k3-hier2-ceiling.md §2)"
#endif

/* ELECT THE MAINTENANCE LEADER OFF HW_REG_XCC_ID, NOT blockIdx.x.
 *
 * This started as `blockIdx.x < 8u`, which is the round-robin map assumed rather than read, and
 * this same file argues against exactly that 130 lines below (`PLOW_L2_PLACE_DISPATCH`: "Deriving
 * it from blockIdx.x instead would make the whole feature silently depend on the round-robin map
 * continuing to hold"). `ctr_convergence.hip` reads the register for its `xcd` too and uses the
 * block index only for the ROLE.
 *
 * `blockIdx.x == xcc` is one candidate per XCD id and the candidate must actually BE on that XCD,
 * so the hardware confirms the map instead of the map being assumed. Under the measured
 * round-robin (`n % 8` is 100.0% of the true id on MI355X) block `x` is on XCD `x` and this elects
 * exactly 8 leaders, one per XCD — identical to the old test, but now verified. If the map ever
 * changes the count DEGRADES (an XCD may get no leader) rather than silently electing two on one
 * XCD, which is the safe direction for an instrument whose failure mode is already understating
 * cost. It also inherits the partitioned-mode guard: a CPX/NPS reported id above the domain count
 * simply matches no block.
 *
 * The sound version does not use this at all — it elects dynamically off the per-(packet, XCD)
 * arrival counter its rendezvous already needs, which is Fleet's "last worker per XCD".
 *
 * HOISTED TO KERNEL ENTRY, and that is not cosmetic. Read per call, `s_getreg` lands on the gate
 * AND on every successor signal — twice per packet, 324,940 workgroup-packets on a K3 token — and
 * it MEASURED 0.43 ms (l2hier 27.849 -> 28.271, ~1.3 ns/workgroup-packet, ~8 sd). The XCD a
 * workgroup runs on cannot change while it is resident, so it is read once. */
__device__ __forceinline__ bool ctr_maint_leader_init(unsigned xcc) {
#if PLOW_GATE_HIER_CEIL
    return blockIdx.x == xcc;
#else
    (void)xcc;
    return true;
#endif
}

__device__ __forceinline__ uint32_t ctr_poll(const uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

/* L1-ONLY invalidate, for a PLOW_GATE_HIER follower whose XCD leader has already done the L2.
 *
 * There is no HIP fence scope that emits this: workgroup-scope acquire emits NO cache
 * maintenance at all (verified on the emitted ISA), and agent-scope emits `buffer_inv sc1`,
 * which is the L1+L2 form and the very thing being amortised. On CDNA4 SC1 is what selects L2
 * for BUFFER_INV, so the bare opcode is the L1 half. Verified to assemble distinctly on gfx950:
 *
 *     buffer_inv       E0A40000     <- this, L1 only
 *     buffer_inv sc0   E0A44000
 *     buffer_inv sc1   E0A48000     <- what ctr_acquire() emits
 *
 * The "memory" clobber is load-bearing: it stops the compiler hoisting the consumer's loads
 * above the invalidate. */
__device__ __forceinline__ void ctr_acquire_l1(void) {
    asm volatile("buffer_inv" ::: "memory");
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
#if PLOW_GATE_SC1
    /* >>> THE PARAGRAPH THAT WAS HERE CLAIMED THIS IS "not a race BY CONSTRUCTION". IT IS A RACE,
     * >>> AND THE KNOB IS BOTH UNSOUND AND SLOWER. MEASURED. DO NOT SHIP IT.
     *
     * The design was: activation stores are `sc0 sc1` so they bypass the non-coherent XCD L2 and
     * are already device-visible, `vmcnt(0)` retires them, and the relaxed RMW then publishes the
     * count — leaving no dirty line for `buffer_wbl2` to find. Coverage was completed
     * (`scripts/sc1_coverage.sh`, zero unscoped stores in all 48 objects) and the model STILL
     * generates degenerate tokens; adding `PLOW_GATE_SC1_KEEPREL`, which changes nothing but
     * keeping the release, is token-identical. Both scopes fail and the arm that keeps the
     * release passes, so the defect is neither coverage nor store scope: `s_waitcnt vmcnt(0)`
     * retires a store when it leaves the CU, NOT when it is visible at the scope the consumer
     * reads from, so the relaxed bump can be observed before the data it publishes. Dropping the
     * release needs a publish primitive that waits for VISIBILITY; this file has none.
     *
     * It is not a speedup either: the numerically-correct variant measured +0.67% on K3 TP8, and
     * `PLOW_GATE_HIER_CEIL` — which keeps the release and amortises it per XCD instead — measured
     * -14.5%. The two do not compose (see the #error above). Full record and the token streams:
     * `runtime/amd/amd_common.h` and `perf-data/archive/k3/k3-hier2-ceiling.md` §2.
     *
     * The wait side is UNCHANGED — `ctr_acquire` still takes the agent acquire and its
     * `buffer_inv`. Dropping it additionally requires every activation LOAD to be `sc1`.
     *
     * THE LOAD SIDE IS REACHABLE, and an earlier note here said it was not. That note reasoned
     * from inline asm, where a 128-bit `sc1` load needs `s_waitcnt vmcnt(0)` inside the asm block
     * (the compiler cannot track an asm-defined VGPR) and would serialise the GEMV staging loop.
     * True of asm, false of the load: `__builtin_amdgcn_raw_buffer_load_b128` takes the cache
     * policy in its `aux` operand, and **bit 4 is `sc1`** —
     *
     *     aux=3  -> buffer_load_dwordx4 ... sc0 nt      (what `buf_ld8` emits today)
     *     aux=16 -> buffer_load_dwordx4 ... sc1
     *     aux=19 -> buffer_load_dwordx4 ... sc0 nt sc1
     *
     * verified by compiling each against gfx950. Full width, intrinsic, compiler-tracked
     * waitcnts, so it pipelines identically to the plain load. (An earlier sweep of aux 0..7
     * missed it and concluded `sc1` was unreachable this way: bit 0 is `sc0`, bit 1 is `nt`, bit 2
     * really is ignored — the bit is 4, outside that range.)
     *
     * So the remaining work for the wait side is mechanical, not blocked: give the activation
     * loads a descriptor and `aux |= 16`, then drop `buffer_inv` in `ctr_acquire`. Together the two
     * halves are the superadditive -5.04 ms/token the UNSAFE knobs below measure as a ceiling.
     *
     * The COVERAGE CAVEAT that was here — "the ragged scalar tails in op_elementwise.h /
     * op_k3.h / op_norm.h / op_moe.h still write activations with plain stores" — is CLOSED.
     * Every store site in every op is routed through `st_act`/`st_act1`/`st_glob8`, and
     * `scripts/sc1_coverage.sh` fails the build on an unscoped one. Closing it is what proved
     * the defect above is ordering rather than coverage. */
    __builtin_amdgcn_s_waitcnt(/*vmcnt(0)*/ 0x0f70);
#if PLOW_GATE_SC1_KEEPREL
    /* DIAGNOSTIC ARM. Scoped stores, but the counter bump KEEPS its release (and therefore its
     * `buffer_wbl2`). This separates the two halves of PLOW_GATE_SC1 that the plain arm changes
     * together: it isolates the COST of scoped activation stores from the BENEFIT of dropping
     * the writeback. If this arm is numerically correct while the plain arm is not, the defect is
     * the ORDERING (`vmcnt(0)` retires a store when it leaves the CU, not when it is visible at
     * the scope the consumer reads from), not the store scope. Not a shipping mode either way. */
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
#elif PLOW_GATE_RELAXSIG
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
/* The exported symbol carries the ISA NAME, because the loader builds it from the live HSA
 * agent name -- crates/plowrt/src/exec/amd.rs:446 formats "{base}_{arch}{sched}" and
 * device/hsa.rs:1222 supplies `arch` verbatim from HSA_AGENT_INFO_NAME. On an MI300X that is
 * literally "gfx942", so an object exporting plow_interp_dec_gfx950 would load and then fail
 * symbol resolution. The build passes -DPLOW_ARCH_SUFFIX=<arch>; the default keeps an
 * unqualified build byte-identical to what it was. */
#ifndef PLOW_ARCH_SUFFIX
#define PLOW_ARCH_SUFFIX gfx950
#endif
#define PLOW_GQ_CAT2(a, b) a##b
#define PLOW_GQ_CAT(a, b) PLOW_GQ_CAT2(a, b)
#if PLOW_BUCKET_DECODE
#define PLOW_INTERP_BASE PLOW_GQ_CAT(plow_interp_dec_, PLOW_ARCH_SUFFIX)
#elif defined(PLOW_BUCKET_PACKED_MLA_NORM)
#define PLOW_INTERP_BASE PLOW_GQ_CAT(plow_interp_packed_mla_norm_, PLOW_ARCH_SUFFIX)
#elif defined(PLOW_BUCKET_PACKED_KDA)
#define PLOW_INTERP_BASE PLOW_GQ_CAT(plow_interp_packed_kda_, PLOW_ARCH_SUFFIX)
#elif defined(PLOW_BUCKET_FLASH)
/* Segmented-dispatch flash object: prefill op set, but built at 4 waves / FA_DC=256 so the flash
 * segments get the 512-register budget. Distinct symbol; one HSA executable holds all three. */
#if PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS
#define PLOW_INTERP_BASE PLOW_GQ_CAT(plow_interp_packed_mla_flash_, PLOW_ARCH_SUFFIX)
#else
#define PLOW_INTERP_BASE PLOW_GQ_CAT(plow_interp_flash_, PLOW_ARCH_SUFFIX)
#endif
#else
#define PLOW_INTERP_BASE PLOW_GQ_CAT(plow_interp_, PLOW_ARCH_SUFFIX)
#endif
/* Global-queue objects export a distinct _gq symbol so the static and GQ kernels co-load. */
#if PLOW_GLOBAL_QUEUE
#define PLOW_INTERP_NAME PLOW_GQ_CAT(PLOW_INTERP_BASE, _gq)
#else
#define PLOW_INTERP_NAME PLOW_INTERP_BASE
#endif

/* PLOW_WPE overrides the requested waves/EU. The default (PLOW_WAVES/4 = 2 at 8 waves) is the
 * 256-register budget every note in this tree is written against. It is also what pins the DECODE
 * GEMV -- 75% of a token's wall time -- to 2 waves per SIMD, because the megakernel's register
 * allocation is the UNION OF EVERY OP and the GEMV pays for arms it never runs. With only two
 * waves on a SIMD there is nothing to cover an s_waitcnt when a wave has 1.6 output rows of work.
 * Raising this forces the allocator under 128 VGPRs; it will spill, and whether the extra
 * latency hiding outruns the spill is the measurement. */
#ifndef PLOW_WPE
#define PLOW_WPE (PLOW_WAVES / 4)
#endif
/* BOTH, and the launch_bounds is the one that binds. Setting only the attribute leaves
 * __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4) asking for 2 waves/EU, which PERMITS the
 * allocator 256 registers -- and it spends them: 253, measured, whatever the LDS allows.
 * Measured on the GEMV loop alone: the same code is 141 VGPR when occupancy 2 is permitted and
 * 51 when it is not. The registers are not needed, they are merely allowed. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WPE)
__attribute__((amdgpu_waves_per_eu(PLOW_WPE, PLOW_WPE))) void PLOW_INTERP_NAME(PlowProgram prog) {
    __shared__ plow_smem sm;

    const unsigned cu = blockIdx.x;
    /* The XCD this workgroup is PHYSICALLY on. Read, not derived from blockIdx -- the round-robin
     * map is measured (100.0% on both MI300X and MI355X) but reading the register means the
     * hierarchy cannot silently depend on it continuing to hold. Available on every path.
     *
     * Behind the #if because s_getreg is NOT dead-code-eliminated when the hierarchy is off:
     * leaving it unconditional put one extra instruction into every object, including gfx950's,
     * and cost the byte-identity that makes "this change did not touch CDNA4" checkable rather
     * than merely asserted. */
#if PLOW_GATE_HIER || PLOW_GATE_HIER_CEIL || defined(PLOW_L2_PLACE_DISPATCH)
    const unsigned plow_xcd =
        __builtin_amdgcn_s_getreg((20 /*HW_REG_XCC_ID*/) | (0 << 6) | ((32 - 1) << 11)) & 0xf;
#else
    const unsigned plow_xcd = 0u;
#endif
    const bool maint_leader = ctr_maint_leader_init(plow_xcd);
#if PLOW_GLOBAL_QUEUE
    /* One shared cursor over the op-major gq_stream, bounded to this segment's window. base==0
     * because gq_stream is already the global array (no per-CU offset); the trace slot is base+ix. */
    __shared__ unsigned gq_claim;
    /* Per-segment cursor line (its own cache line via PLOW_CTR stride). RUNSEG enqueues all segment
     * launches without a host wait and zeroes state ONCE before the loop, so each segment needs its
     * OWN cursor rather than a shared one reset between launches — same lifecycle as the counters. */
#ifdef PLOW_L2_PLACE_DISPATCH
    /* L2-DOMAIN DISPATCH (PLOW_L2_PLACE). Every ordered kernel-family segment has one window per
     * physical L2 domain. All 8 domains drain concurrently inside each launch,
     * so a consumer reads its producer out of the same XCD's 4 MiB L2 instead of crossing the
     * fabric. The compiler sorts `gq_stream` by (ordered segment, domain).
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
     * Do not modulo the hardware id: that would alias distinct physical L2s into one rendezvous
     * and invalidate the hierarchy's same-L2 proof. The gfx950 bring-up gate verifies all 256
     * cooperative workgroups report exactly ids 0..7 on every served GPU; object/target pairing
     * rejects this path on a different architecture. l2_domains == 0 means unplaced. */
    if (prog.l2_domains && plow_xcd >= prog.l2_domains) __builtin_trap();
    const unsigned my_seg = prog.l2_domains
        ? prog.cur_seg * prog.l2_domains + plow_xcd
        : prog.cur_seg;
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
    /* Static streams retain the ordered kernel-family segment too. Their fixed per-CU
     * assignment already preserves locality; no domain-indexed relaunch is needed. */
    const unsigned st_seg = prog.cur_seg;
    const unsigned lo = srow ? srow[st_seg] : 0u;
    const unsigned hi = srow ? srow[st_seg + 1u] : n;
#endif

    /* Everything the interpreter reads and never writes, in the constant address space. */
    const CInst insts = as_konst(prog.insts);
    const CTens tens = as_konst(prog.tensors);

#if PLOW_GLOBAL_QUEUE
    for (;;) {
#if PLOW_TRACE_PHASE
        /* T0: BEFORE the claim barrier, so the claim's dependent atomic round trip is inside the
         * measured window rather than in front of it. */
        const uint64_t t_p0 = prog.trace ? __builtin_amdgcn_s_memrealtime() : 0;
#endif
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
#if PLOW_TRACE_PHASE
            /* Same probe on the static path, so an object built with the knob compiles on both. */
            const uint64_t t_p0 = prog.trace ? __builtin_amdgcn_s_memrealtime() : 0;
#endif
            const PlowStreamEnt e = my[ix];
            /* Segmented dispatch. With a window, [lo, hi) ALREADY bounds this launch's segment
             * and this compare is dead — same contract as the global-queue path above. Without
             * one (`seg_ofs == NULL`) the window is the whole stream and the filter is what
             * selects the segment: a skipped entry costs a branch and nothing else — no gate,
             * no exec, no signal — and it is executed by the segment launch that owns it.
             * Unsegmented programs set every seg==0 and launch with cur_seg==0. */
            if (!srow && e.seg != st_seg) continue;
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

        /* CROSS-GPU gate. When set, this entry's wait/succ
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

#if PLOW_GATE_HIER
        /* Two-level maintenance state for THIS entry. `h_n` is how many slices of this packet
         * landed on this L2 domain (emit-time constant, see PLOW_SE_NPER); 0 or 1 means there is
         * nobody to rendezvous with and the ordinary path runs unchanged. Cross-GPU entries are
         * excluded: their counters are system-scope and per-XCD leadership says nothing about a
         * fabric peer. */
        const unsigned h_n = PLOW_SE_NPER(e.flags);
        /* Fine slices may carry different wait/successor lists. Even if a malformed or old
         * emitter leaves NPER set, one elected slice cannot poll or signal on their behalf. */
        const bool h_on = (prog.hier_base != 0u) && (h_n > 1u)
                          && !(e.flags & PLOW_SE_FINE) && !xctr;
        const size_t h_b = h_on ? (size_t)prog.hier_base
                                      + ((size_t)e.inst * prog.l2_domains
                                         + plow_xcd) * 3u
                                : 0u;
        __shared__ unsigned h_lead;
        if (h_on) {
            /* OBSERVE ELECTION: first arriver on this (packet, domain) takes the invalidate. */
            if (threadIdx.x == 0)
                h_lead = (__hip_atomic_fetch_add(PLOW_CTR(prog.counters, h_b + 1u), 1u,
                                                 __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == 0u);
            __syncthreads();
        }
#else
        const bool h_on = false;
#endif

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
        /* Under PLOW_GATE_HIER a FOLLOWER never polls the global counter at all — its XCD's
         * leader does, and releases it through the XCD-local `opn`. That is less fabric traffic,
         * not more: one cross-XCD poll per domain instead of one per workgroup. */
        /* PROBE (PLOW_INST_PF): pull the 64-byte PlowDevInst into SGPRs BEFORE the gate poll, so
         * its scalar-cache miss overlaps the counter wait instead of following it. The gate
         * metadata lives on the STREAM ENTRY (e.wait_*), so nothing here depends on the
         * instruction -- the dependency is only that every `in->` read sits after the poll in
         * program order. NOTE `insts` is address_space(4) (constant), i.e. s_load through the
         * scalar cache, which LLVM already hoists aggressively -- so this may well be a no-op.
         * Measured either way rather than assumed. MUST sit ABOVE the GATE_HIER leader guard:
         * placed between the guard and the poll loop it HIJACKS the `if`, and every follower
         * then polls the global counters unconditionally — the exact behavior the hierarchy
         * exists to remove. */
#if PLOW_INST_PF
        {
            const unsigned pf_ = (unsigned)in->op | ((unsigned)in->blocks << 16);
            const unsigned pf2_ = in->i[0] ^ in->i[4];
            asm volatile("" ::"s"(pf_), "s"(pf2_));
        }
#endif
#if PLOW_GATE_HIER
        if (!h_on || h_lead)
#endif
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
#if PLOW_TRACE_PHASE
        /* T1: the dependency is met; nothing below here is waiting for a producer. */
        const uint64_t t_p1 = prog.trace ? __builtin_amdgcn_s_memrealtime() : 0;
#endif

        /* ONE acquire for the whole workgroup, and only after the gate clears.
         *
         * An agent-scope acquire emits buffer_inv, a full L1/L2 invalidate. Putting one inside
         * the poll loop (which is where it lands if you just write `while (acquire_load(c) < n)`)
         * flushes the caches on EVERY iteration -- it costs time in the spin AND destroys the
         * cache for the work that follows. That single change was worth ~2x on decode (§8). */
        /* THE LEADER TEST APPLIES TO THE LOCAL ACQUIRE ONLY. `xctr_acquire` is the SYSTEM-scope
         * acquire for data arriving over XGMI (op_collective.h), and there is no per-XCD leader
         * story for the fabric — HIER2 is about per-L2 maintenance INSIDE one device, and
         * ctr_convergence.hip's HIER2 arm has no cross-GPU component at all. Gating it here was a
         * measurement bug, not a design: leaders are `blockIdx.x < 8` while K3's collectives are
         * b=14 after `xrfit`, so the expected number of leaders on a collective is 14·8/256 =
         * 0.44 — i.e. almost every cross-GPU acquire was being deleted, on a TP8 run, and the
         * ceiling it produced was overstated by that deletion. */
#if PLOW_GATE_HIER
        if (h_on) {
            if (threadIdx.x == 0) {
                if (h_lead) {
                    /* ONE buffer_inv sc1 for the whole XCD, then open its local flag. The
                     * acquire keeps the relaxed bump below the invalidate; followers poll that
                     * same L2 and put an L1 invalidate below the poll with a memory clobber.
                     * A release bump would add the partition writeback this rendezvous exists
                     * to avoid. */
                    if (wait_len) ctr_acquire();
                    __hip_atomic_fetch_add(PLOW_CTR(prog.counters, h_b + 2u), 1u,
                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                } else {
                    while (ctr_poll(PLOW_CTR(prog.counters, h_b + 2u)) < 1u)
                        __builtin_amdgcn_s_sleep(8);
                    /* L1 only — the leader already did this XCD's L2, but L1 is per-CU. */
                    ctr_acquire_l1();
                }
            }
        } else
#endif
        if (threadIdx.x == 0 && wait_len) {
            if (xctr) xctr_acquire();
            else if (maint_leader) ctr_acquire();
        }
        __syncthreads(); /* the invalidate is visible to every thread before it reads */

        const uint64_t t_ready = prog.trace ? __builtin_amdgcn_s_memrealtime() : 0;

#if !defined(PLOW_BUCKET_FLASH)  /* collectives in the DECODE and 8-wave PREFILL interpreters —
         * TP PREFILL needs XReduce in the prefill interpreter too, not just decode
         * Excluded from the 4-wave flash object (class-4 flash segments
         * never contain an XReduce, and it keeps that object's tight register budget untouched).
         * Bodies use only the PlowProgram peer fields, present in every bucket. */
        /* CROSS-GPU collectives run here, not in plow_exec, because their bodies need the
         * PlowProgram peer fields (peer_scratch/rank/n_gpu/xctr). */
        if (in->op == PLOW_DOP_XREDUCE) {
            /* COARSE self-contained one-shot all-reduce.
             * The local agent gate above already ensured THIS rank's producing GEMV wrote
             * its partial into peer_scratch[rank]; d_xreduce_mega then cross-signals, waits
             * N arrivals (system xctr), and sums the N peer slots. Validated bit-exact +
             * ~0.29 us/collective by tp_allreduce_bench.
             *   t0=out i0=H i1=n_gpu i2=slot_bytes i3=xctr gate id
             *   i4=gather slot_bytes i5=gather cols/rank (0 = off) i6=out row width
             * i4/i5/i6 fold an ALL-GATHER of a column-parallel partial into this reduce —
             * K3's `routed_expert_up_proj`, see d_xreduce. Zero on every other collective,
             * and the arm is then byte-identical to the pre-gather one. */
            const size_t xoff =
                (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank];
            uint32_t* const xaudit = plow_xaudit_status(prog, in);
            d_xreduce_mega((bf16*)tens[in->t[0]], prog.peer_scratch, prog.n_gpu, prog.rank,
                           in->i[0], in->i[2], xoff, in->i[3], PLOW_XCTR_DEADLINE_TICKS,
                           xaudit, e.slice, in->blocks, in->i[4], in->i[5], in->i[6]);
        } else if (in->op == PLOW_DOP_XREDUCE_ADD_NORM) {
            /* Fused all-reduce + residual + RMSNorm (op 116, one workgroup):
             *   t0=out t1=resid t2=a t3=gamma  i0=feat i1=n_gpu i2=slot i3=gate  f0=eps */
            const size_t xoff =
                (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank];
            uint32_t* const xaudit = plow_xaudit_status(prog, in);
            d_xreduce_add_norm_mega((bf16*)tens[in->t[0]], (bf16*)tens[in->t[1]],
                                    (const bf16*)tens[in->t[2]], (const bf16*)tens[in->t[3]],
                                    prog.peer_scratch, prog.n_gpu, prog.rank, in->i[0], in->i[2],
                                    xoff, in->i[3], PLOW_XCTR_DEADLINE_TICKS, xaudit, in->fj[0].f);
        } else if (in->op == PLOW_DOP_XREDUCE2) {
            /* TWO-SHOT all-reduce for the LARGE prefill [T,hidden] partial (§4, bandwidth-
             * optimal): reduce-scatter this rank's owned 1/N slice from every peer, then
             * all-gather the N reduced slices. Two internal xctr rendezvous (gate_rs = i3,
             * gate_ag = i4). Same peer-scratch + xctr_off wiring as the one-shot.
             *   t0=out i0=n(t*hidden) i1=n_gpu i2=slot_bytes i3=gate_rs i4=gate_ag
             *   i5=e0: element offset of this packet's band into `out` AND into the window
             *   (the dispatch adds e0*2 to the slot). 0 = the whole partial (every existing
             *   blob). i2 stays the REGION base (0 or slot_b) because the loader INFERS the
             *   window slot size as max(i2) (asset/devblob.rs) — a band offset folded into
             *   i2 would inflate that inference and shift where the second partial binds
             *   (measured: NaN everything). A banded TP seam emits K of these packets, each
             *   with its own gate pair; band 0's transfer overlaps band 1..K-1's producers. */
            const size_t xoff =
                (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank];
            uint32_t* const xaudit = plow_xaudit_status(prog, in);
            /* t1/t2 = fused-residual operands (absent on every pre-fold blob): resid in,
             * summed output out2 — see the kernel's fused-residual note. Both take the
             * band offset e0, like out. */
            const bf16* xr_res = in->t[1] == PLOW_TENSOR_NONE
                                     ? nullptr
                                     : (const bf16*)tens[in->t[1]] + in->i[5];
            bf16* xr_o2 =
                in->t[2] == PLOW_TENSOR_NONE ? nullptr : (bf16*)tens[in->t[2]] + in->i[5];
            d_xreduce_twoshot_mega((bf16*)tens[in->t[0]] + in->i[5], prog.peer_scratch,
                                   prog.n_gpu, prog.rank,
                                   in->i[0], in->i[2] + in->i[5] * 2u, xoff, in->i[3], in->i[4],
                                   PLOW_XCTR_DEADLINE_TICKS, xaudit, e.slice, in->blocks,
                                   xr_res, xr_o2);
        } else if (in->op == PLOW_DOP_XARGMAX_FIN) {
            /* Cross-rank fold for a VOCAB-COLUMN-PARALLEL lm_head.
             * Subsumes ARGMAX_FIN: it does the local nparts fold too, then rebases the index
             * into global vocab space and takes the cross-rank max of the packed key.
             *   t0=ids t1=amax.part  i0=nparts i1=n_batch i2=vocab_l
             *   i3=xctr arrival-gate id  i4=xctr value-slot id (a DIFFERENT id) */
            const size_t xoff =
                (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank];
            uint32_t* const xaudit = plow_xaudit_status(prog, in);
            d_xargmax_fin_mega((int*)tens[in->t[0]], (const unsigned long long*)tens[in->t[1]],
                               in->i[0], in->i[1], in->i[2], prog.peer_scratch, prog.n_gpu,
                               prog.rank, xoff, in->i[3], in->i[4], PLOW_XCTR_DEADLINE_TICKS,
                               xaudit, e.slice, sm.am);
        } else if (in->op == PLOW_DOP_XFLASHMERGE) {
            /* STUB: dispatch present so the switch is total and the
             * compiler's structural emit runs; the CP flash LSE-merge body lands with tp-host.
             * No-op keeps single-GPU decode bit-identical. */
        } else
#endif
#ifdef PLOW_ICACHE_INV_PROBE
        /* MEASUREMENT INSTRUMENT, never on in a shipped build (guarded by a macro that nothing
         * defines; with it undefined the preprocessed source — and therefore the tunedb build
         * digest — is byte-identical).
         *
         * Question it answers: this interpreter is a persistent megakernel whose .text is 338 KB
         * (`interp_decode_fp8kv_k3.elf`), and consecutive packets run DIFFERENT op bodies, so the
         * instruction stream jumps across a large binary. How much of the runtime is the L1
         * instruction cache missing on those jumps?
         *
         * `s_icache_inv` flushes the SQC instruction cache, so with this on EVERY packet's body is
         * a GUARANTEED cold instruction fetch from L2 — the worst case a zero-hit-rate I-cache
         * could produce. The delta (probe - baseline) is therefore an UPPER BOUND on everything
         * instruction prefetch, body reordering or footprint shrinking could ever recover. It is
         * an upper bound and not an estimate: the real run already hits in I-cache some of the
         * time, and this also charges the cost of the s_icache_inv itself.
         *
         * gfx950 has no instruction-PREFETCH instruction to test the other direction with:
         * `s_inst_prefetch` is GFX10/GFX11 only and `s_prefetch_inst` is GFX12 only (verified
         * against the ROCm 7.2.4 assembler, /opt/rocm/llvm/bin/llvm-mc -mcpu=gfx950). Invalidation
         * is the only I-cache lever the ISA gives this part, so bounding from above is the only
         * experiment available. */
        __asm__ __volatile__("s_icache_inv" ::: "memory");
#endif
        plow_exec(in, e.slice, tens, &sm, &prog);

        __syncthreads();
#if PLOW_TRACE_PHASE
        /* T3: the op body is retired and its stores are in this XCD's L2. */
        const uint64_t t_p3 = prog.trace ? __builtin_amdgcn_s_memrealtime() : 0;
#endif

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
#if PLOW_GATE_HIER
        /* PUBLISH RENDEZVOUS. The __syncthreads() above has already retired this workgroup's
         * stores into its XCD's L2 (s_waitcnt vmcnt(0)), so a RELAXED bump is enough to announce
         * them — the data is in the L2 the leader is about to write back. The workgroup whose
         * bump completes the domain issues ONE release RMW per successor, adding `h_n` instead of
         * 1 so the global counter still reaches `blocks[p]` and no consumer's threshold moves. */
        if (h_on) {
            if (threadIdx.x == 0) {
                const uint32_t was = __hip_atomic_fetch_add(PLOW_CTR(prog.counters, h_b), 1u,
                                                            __ATOMIC_RELAXED,
                                                            __HIP_MEMORY_SCOPE_AGENT);
                if (was == h_n - 1u)
                    for (unsigned s = 0; s < succ_len; s++)
                        __hip_atomic_fetch_add(PLOW_CTR(prog.counters, prog.succs[succ_ofs + s]),
                                               h_n, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }
        } else
#endif
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
            } else if (maint_leader) {
                ctr_signal(PLOW_CTR(prog.counters, sid));
            } else {
                /* PLOW_GATE_HIER_CEIL follower: the count still has to reach the threshold or
                 * every consumer deadlocks, but the writeback is the leader's job. Relaxed. */
                __hip_atomic_fetch_add(PLOW_CTR(prog.counters, sid), 1u, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
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
#if PLOW_TRACE_PHASE
                {
                    const uint64_t a = t_p1 - t_p0, b = t_p3 - t_ready;
                    r.pc = (uint32_t)(a > 0xffffull ? 0xffffull : a) |
                           ((uint32_t)(b > 0xffffull ? 0xffffull : b) << 16);
                }
#else
                r.pc = ix;
#endif
                r.inst = e.inst;
                r.op = in->op;
                r.slice = (uint16_t)e.slice;
#if PLOW_TRACE_PHASE
                r.t_arrive = t_p0;
#else
                r.t_arrive = t_arrive;
#endif
                r.t_ready = t_ready;
                r.t_end = __builtin_amdgcn_s_memrealtime();
                prog.trace[base + ix] = r;
            }
        }
        }
    }
}

/* Runs only after every rank's interpreter has drained. It preserves the old
 * exact audit, including zero for unused gates, while reducing the healthy host
 * path to one status-word read per rank. Defined after the interpreter so the
 * build scripts' first-kernel resource summary remains the serving kernel. */
extern "C" __global__ void plow_xctr_audit(PlowXAuditArgs a) {
    for (uint32_t gate = blockIdx.x * blockDim.x + threadIdx.x; gate < a.n_xctr;
         gate += gridDim.x * blockDim.x) {
        uint32_t want = 0;
        bool data = false;
        for (uint32_t j = 0; j < a.n_inst; j++) {
            const PlowDevInst& d = a.insts[j];
            if (d.op == PLOW_DOP_XREDUCE || d.op == PLOW_DOP_XREDUCE_ADD_NORM) {
                if (d.i[3] == gate) want = a.n_gpu;
            } else if (d.op == PLOW_DOP_XREDUCE2) {
                if (d.i[3] == gate) want = a.n_gpu;
                if (d.i[4] == gate) want = a.n_gpu * d.blocks;
            } else if (d.op == PLOW_DOP_XARGMAX_FIN) {
                if (d.i[3] == gate) want = a.n_gpu;
                const uint32_t n = d.i[1] ? d.i[1] : 1u;
                const uint32_t lines = (n - 1u) / PLOW_XAMAX_LINE + 1u;
                if (gate >= d.i[4] && gate - d.i[4] < lines) data = true;
            }
        }
        if (!data) {
            const uint32_t got = xctr_poll(PLOW_CTR(a.xctr, gate));
            if (got != want)
                atomicCAS(a.status, 0u, 0xA0000000u | (gate & 0x000FFFFFu));
        }
    }
}

/* Admission-time clear for slot-major recurrent state. One block owns one
 * load-time range; ranges are chunked so large KDA states cannot strand CUs. */
extern "C" __global__ void plow_state_clear(PlowStateClearArgs a) {
    const uint32_t r = blockIdx.x;
    if (r >= a.n_ranges) return;
    const PlowStateClearRange d = a.ranges[r];
    uint32_t* p = (uint32_t*)(d.base + (uint64_t)a.slot * d.slot_stride);
    for (uint32_t i = threadIdx.x; i < d.words; i += blockDim.x) p[i] = 0;
}

/* Capture device-sampled ids without a host readback between decode tokens.
 * The launch is queued immediately after rank 0's interpreter dispatch, whose
 * AQL barrier orders the argmax stores before these loads. */
extern "C" __global__ void plow_token_capture(PlowTokenCaptureArgs a) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < a.batch) a.ring[row * a.quantum + a.step] = a.ids[row];
}
