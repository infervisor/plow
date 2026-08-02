/* MXFP4 decode GEMV: the pre-fix DUAL-COLUMN body vs the R-split one, at M=1 — the instrument
 * behind the `gemv_mx_rows_r` change. Same file as `perf-data/glm52_gemvblk_bench.hip` in spirit
 * and in method: production body included verbatim from the real header, launch geometry mirrors
 * the persistent interpreter (blockDim = PLOW_THREADS, grid = the emitter's workgroup count), x is
 * staged ONCE outside the rep loop so the number is the packet BODY and not an LDS restage.
 *
 * BOTH BODIES LIVE IN ONE OBJECT so the A/B is INTERLEAVED inside a single run (contract 6b-STALE)
 * rather than two builds minutes apart — the fix is worth single-digit microseconds at the small
 * shapes and clock drift between two binaries would swamp it. It also lets the host compare the two
 * output buffers ELEMENTWISE, which is the bit-exactness claim checked on real silicon instead of
 * argued from the source.
 *
 * The `_dual` body below is the old `gemv_rows_mxfp4` copied verbatim (only the name changed). It is
 * a FROZEN COPY for measurement, not a second production path: nothing includes this file.
 *
 * Build + run (ROCm tooling must be OUTSIDE `nix develop` — knob-contract 0a):
 *   hipcc --offload-arch=gfx950 -O3 -w -DPLOW_BUCKET_DECODE=1 -std=c++17 --genco \
 *         perf-data/k3_gemvmx_bench.hip -o /tmp/gvmx.co -Iruntime/amd -Iruntime/common
 *   g++ -O2 -std=c++17 perf-data/k3_gemvmx_bench.cpp -o /tmp/gvmx \
 *         -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ -L/opt/rocm/lib -lamdhip64
 *   perf-data/harness/gpulease -n 1 gvmx sg render -c \
 *         'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; /tmp/gvmx /tmp/gvmx.co 256 21'
 * `unset HIP_VISIBLE_DEVICES` is load-bearing: gpulease exports BOTH it and ROCR to the same
 * absolute id and they COMPOSE, so hipInit reports "no ROCm-capable device" on a leased GPU. */
#include <hip/hip_runtime.h>

#include "amd_common.h"
#include "op_gemm.h"

typedef union __align__(16) { bf16 xstage[8192]; float part[PLOW_WAVES]; } gv_smem;

/* ---- FROZEN COPY of the pre-fix body: `n2 = has2 ? n + PLOW_WAVES : n`, store-guarded only ---- */
template <int MM, bool XLDS, int UN = 3>
__device__ __forceinline__ void gemv_rows_mxfp4_dual(bf16* __restrict__ Cq_, bf16* __restrict__ Ck_,
                                                bf16* __restrict__ Cv_,
                                                const bf16* __restrict__ x_,
                                                const unsigned char* __restrict__ Wq_,
                                                const unsigned char* __restrict__ Wk_,
                                                const unsigned char* __restrict__ Wv_,
                                                const unsigned char* __restrict__ Sq_,
                                                const unsigned char* __restrict__ Sk_,
                                                const unsigned char* __restrict__ Sv_, unsigned M,
                                                unsigned Nq, unsigned Nk, unsigned Nv, unsigned K,
                                                unsigned slice, unsigned nblk, const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned step = PLOW_WAVE * 32; /* one pass: 64 lanes x 32 fp4 = 2048 K */
    const unsigned nchunk = (K + step - 1) / step;
    const unsigned wbytes = K >> 1;   /* packed weight row stride */
    const unsigned nsb = (K + 31) >> 5; /* scale bytes per row     */
    const unsigned N = Nq + Nk + Nv;    /* the CONCATENATED column count */

    auto* const Cq = as_glob(Cq_);
    auto* const Ck = as_glob(Ck_);
    auto* const Cv = as_glob(Cv_);
    const auto* const x = as_glob(x_);
    const auto* const Wq = as_glob(Wq_);
    const auto* const Wk = as_glob(Wk_);
    const auto* const Wv = as_glob(Wv_);
    const auto* const Sq = as_glob(Sq_);
    const auto* const Sk = as_glob(Sk_);
    const auto* const Sv = as_glob(Sv_);

    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 {
        const size_t xo = (size_t)m * K + kk;
        if constexpr (XLDS)
            return ld_lds8(lds + xo);
        else
            return ld_glob8(x + xo);
    };

    /* TWO OUTPUT COLUMNS PER WAVE-STEP, for the reason the fp8 twin documents: fp4 doubles the
     * K-elements per load again, so nchunk halves again and the in-flight-load ceiling per column
     * gets even tighter. Two independent rows keep 2*min(UN,nchunk) loads outstanding.
     *
     * The pair (n, n2) may straddle a stream boundary. That is fine and needs no special case:
     * each of the two carries its own weight row, scale row, output base and output width. */
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
    for (unsigned n = gv_n0 + wave; n < gv_n1; n += PLOW_WAVES * 2) {
        const bool has2 = (n + PLOW_WAVES) < gv_n1;
        const unsigned n2 = has2 ? n + PLOW_WAVES : n;
        /* Which projection each concatenated column belongs to. Wave-uniform, therefore scalar. */
        auto Wr = Wq, Wr2 = Wq;
        auto Sc = Sq, Sc2 = Sq;
        auto Co = Cq, Co2 = Cq;
        unsigned col = n, col2 = n2, Nout = Nq, Nout2 = Nq;
        if (n >= Nq + Nk) { Wr = Wv; Sc = Sv; Co = Cv; col = n - Nq - Nk; Nout = Nv; }
        else if (n >= Nq) { Wr = Wk; Sc = Sk; Co = Ck; col = n - Nq;      Nout = Nk; }
        if (n2 >= Nq + Nk) { Wr2 = Wv; Sc2 = Sv; Co2 = Cv; col2 = n2 - Nq - Nk; Nout2 = Nv; }
        else if (n2 >= Nq) { Wr2 = Wk; Sc2 = Sk; Co2 = Ck; col2 = n2 - Nq;      Nout2 = Nk; }
        const __amdgpu_buffer_rsrc_t wr = buf_rsrc_fp8(Wr + (size_t)col * wbytes, wbytes);
        const __amdgpu_buffer_rsrc_t wr2 = buf_rsrc_fp8(Wr2 + (size_t)col2 * wbytes, wbytes);
        const auto* const sr = Sc + (size_t)col * nsb;
        const auto* const sr2 = Sc2 + (size_t)col2 * nsb;
        float acc[MM], acc2[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) { acc[m] = 0.0f; acc2[m] = 0.0f; }

        for (unsigned c = 0; c < nchunk; c += UN) {
            fp4v32 wv[UN], wv2[UN];
            /* Both columns' weight loads issued before either is consumed: 2*UN in flight. */
#pragma unroll
            for (int u = 0; u < UN; u++)
                wv[u] = __builtin_bit_cast(
                    fp4v32, buf_ld_fp8(wr, ((c + (unsigned)u) * step + lane * 32) >> 1));
#pragma unroll
            for (int u = 0; u < UN; u++)
                wv2[u] = __builtin_bit_cast(
                    fp4v32, buf_ld_fp8(wr2, ((c + (unsigned)u) * step + lane * 32) >> 1));
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * step + lane * 32;
                const unsigned kx = (k < K) ? k : 0u;
                const unsigned sb = k >> 5; /* this lane's block index; one per fragment */
                const float sc = (sb < nsb) ? e8m0_to_f32(sr[sb]) : 0.0f;
                const float sc2 = (sb < nsb) ? e8m0_to_f32(sr2[sb]) : 0.0f;
                bf16v8 w0, w1, w2, w3, y0, y1, y2, y3;
                fp4_to_bf16v8x4(wv[u], sc, w0, w1, w2, w3);
                fp4_to_bf16v8x4(wv2[u], sc2, y0, y1, y2, y3);
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    const bool live = ((unsigned)m < M);
                    const bf16v8 x0 = xv8(live ? m : 0, kx);
                    const bf16v8 x1 = xv8(live ? m : 0, kx + 8);
                    const bf16v8 x2 = xv8(live ? m : 0, kx + 16);
                    const bf16v8 x3 = xv8(live ? m : 0, kx + 24);
                    acc[m] += live ? dot8(w3, x3, dot8(w2, x2, dot8(w1, x1, dot8(w0, x0, 0.0f))))
                                   : 0.0f;
                    acc2[m] += live ? dot8(y3, x3, dot8(y2, x2, dot8(y1, x1, dot8(y0, x0, 0.0f))))
                                    : 0.0f;
                }
            }
        }
        /* No dequant here — unlike the fp8 twin's `* wscale[n]`, the MX scale is already folded
         * into every converted element by fp4_to_bf16v8x4. */
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float t = wave_sum(acc[m]);
            const float t2 = wave_sum(acc2[m]);
            if (lane == 0 && (unsigned)m < M) {
                Co[(size_t)m * Nout + col] = f2bf(t);
                if (has2) Co2[(size_t)m * Nout2 + col2] = f2bf(t2);
            }
        }
    }
}


/* Both arms take the SAME UN at the same K — `d_gemv_qkv_mxfp4`'s table, verbatim — because a UN
 * difference would change the chunk grouping and the two would no longer be comparable (nor
 * bit-exact). 1-stream form: Nk = Nv = 0, k/v pointers null, exactly as `d_gemv_mxfp4` calls it. */
#define MX_ARMS(FN, C, X, W, S, N, K, LDS)                                                          \
    do {                                                                                         \
        const unsigned nchunk_ = ((K) + PLOW_WAVE * 32 - 1) / (PLOW_WAVE * 32);                   \
        if (nchunk_ >= 8u)                                                                       \
            FN<PLOW_GEMV_MM, true, 4>(C, nullptr, nullptr, X, W, nullptr, nullptr, S, nullptr,  \
                                      nullptr, 1u, N, 0u, 0u, K, blockIdx.x, gridDim.x, LDS);     \
        else if (nchunk_ >= 6u)                                                                   \
            FN<PLOW_GEMV_MM, true, 3>(C, nullptr, nullptr, X, W, nullptr, nullptr, S, nullptr,  \
                                      nullptr, 1u, N, 0u, 0u, K, blockIdx.x, gridDim.x, LDS);     \
        else                                                                                      \
            FN<PLOW_GEMV_MM, true, 2>(C, nullptr, nullptr, X, W, nullptr, nullptr, S, nullptr,  \
                                      nullptr, 1u, N, 0u, 0u, K, blockIdx.x, gridDim.x, LDS);     \
    } while (0)

/* nrep walks a fresh [N,K] weight slab each pass so the number is a STREAM and not an L2 hit; the
 * scale slab advances with it (K/32 bytes per row against the weights' K/2). */
#define MX_KERNEL(NAME, FN)                                                                       \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void NAME(                              \
        bf16* C, const bf16* x, const unsigned char* W, const unsigned char* S, unsigned nrep,     \
        unsigned N, unsigned K) {                                                                 \
        __shared__ gv_smem sm;                                                                     \
        for (unsigned i = threadIdx.x; i < K; i += PLOW_THREADS) sm.xstage[i] = x[i];              \
        __syncthreads();                                                                           \
        for (unsigned e = 0; e < nrep; e++) {                                                      \
            const unsigned char* Wp = W + (size_t)e * N * (K / 2);                                 \
            const unsigned char* Sp = S + (size_t)e * N * (K / 32);                                \
            MX_ARMS(FN, C, x, Wp, Sp, N, K, sm.xstage);                                                \
        }                                                                                          \
    }
/* ONE BODY PER OBJECT when asked. Both arms in one object is the better A/B (one process, one
 * clock state) but it makes the object carry 1.5x the instructions of either arm alone, and the
 * mxfp4 chunk loop is huge enough (288 cvt_scalef32_pk_bf16_fp4 per R=2 UN=4 instantiation) that
 * the layout of one arm perturbs the other. `-DMX_ARM=0` emits only the pre-fix body and `=1` only
 * the shipping one, so the host can load TWO objects and still interleave them in one process —
 * each compiled the way production compiles it. Default: both, for the bit-exactness check. */
/* THIRD ARM: the NEW inner function driven by the OLD always-R=2 loop, so the odd tail is still
 * recomputed-and-discarded. It exists to attribute the 2.5% that `new` loses on the EVEN control
 * shapes, where `new` never executes its R=1 body at all and yet is slower than `old`. Two
 * candidate causes: (a) the R-parameterised inner function generates worse code for R=2 than the
 * hand-written dual body, or (b) merely INSTANTIATING the R=1 body costs the R=2 loop something
 * (code footprint / layout). This arm has (a) and not (b) — it calls the same
 * `gemv_mx_rows_r<...,2>` and never instantiates R=1. If it matches `old`, the cost is (b). */
template <int MM, bool XLDS, int UN = 3>
__device__ __forceinline__ void gemv_rows_mxfp4_r2only(
    bf16* __restrict__ Cq_, bf16* __restrict__ Ck_, bf16* __restrict__ Cv_,
    const bf16* __restrict__ x_, const unsigned char* __restrict__ Wq_,
    const unsigned char* __restrict__ Wk_, const unsigned char* __restrict__ Wv_,
    const unsigned char* __restrict__ Sq_, const unsigned char* __restrict__ Sk_,
    const unsigned char* __restrict__ Sv_, unsigned M, unsigned Nq, unsigned Nk, unsigned Nv,
    unsigned K, unsigned slice, unsigned nblk, const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wbytes = K >> 1;
    const unsigned nsb = (K + 31) >> 5;
    const unsigned N = Nq + Nk + Nv;
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
    /* `n + PLOW_WAVES < gv_n1`, not `n < gv_n1`: an unconditional R=2 would read column
     * `n + PLOW_WAVES` past the end of the matrix (the old body clamped it to `n` instead, which is
     * exactly the duplicate this change deletes) and that faults. So this arm simply DROPS an odd
     * tail column — it is bit-exact on the EVEN shapes it exists to attribute and deliberately
     * wrong on the odd ones, which is why the bitexact column reads DIFFERS there. */
    for (unsigned n = gv_n0 + wave; n + PLOW_WAVES < gv_n1; n += PLOW_WAVES * 2)
        gemv_mx_rows_r<MM, XLDS, UN, 2>(Cq_, Ck_, Cv_, x_, Wq_, Wk_, Wv_, Sq_, Sk_, Sv_, M, Nq, Nk,
                                        Nv, K, wbytes, nsb, n, lane, lds);
}

#ifndef MX_ARM
#define MX_ARM 2 /* 0=old,1=new,2=both,3=r2only,4=A/A control,5=XLDS=false analysis */
#endif
#if MX_ARM != 1 && MX_ARM != 3 && MX_ARM != 4 && MX_ARM != 5
MX_KERNEL(k_mx_old, gemv_rows_mxfp4_dual) /* pre-fix: dual column, odd tail recomputed + discarded */
#endif
#if MX_ARM == 4
/* A/A CONTROL: `k_mx_new` IS the old body. Paired against an MX_ARM=0 object this measures the
 * harness's own bias between the arm named `old` and the arm named `new` — two objects, identical
 * device code, so anything but 1.000x is instrument and not kernel. It exists because the R-split
 * read 0.97x on the EVEN control shapes where it executes the same work as the old body, and a
 * 3% claim needs to be separated from a 3% artifact before it can be attributed to anything. */
MX_KERNEL(k_mx_new, gemv_rows_mxfp4_dual)
#elif MX_ARM == 3
MX_KERNEL(k_mx_new, gemv_rows_mxfp4_r2only) /* attribution arm: new inner fn, old always-R=2 loop  */
#elif MX_ARM != 0 && MX_ARM != 5
MX_KERNEL(k_mx_new, gemv_rows_mxfp4)      /* shipping: R=2 body + R=1 tail                         */
#endif

/* ---- PER-CHANNEL FP8 (op 43) — the same defect, the same fix, its own A/B. FROZEN COPY of the
 * pre-fix body below, exactly as for mxfp4 above. `d_gemv_fp8` picks UN=5 at K=9728 and the
 * GV_UNROLL_FP8 default of 3 everywhere else; both arms take the same one at the same K. ---- */
template <int MM, bool XLDS, int UN = GV_UNROLL_FP8>
__device__ __forceinline__ void gemv_rows_fp8_dual(bf16* __restrict__ C_, const bf16* __restrict__ x_,
                                              const unsigned char* __restrict__ W_,
                                              const float* __restrict__ wscale_, unsigned M,
                                              unsigned N, unsigned K, unsigned slice, unsigned nblk,
                                              const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned step = PLOW_WAVE * 16; /* one pass: 64 lanes x 16 fp8 */
    const unsigned nchunk = (K + step - 1) / step;

    auto* const C = as_glob(C_);
    const auto* const x = as_glob(x_);
    const auto* const W = as_glob(W_);
    const auto* const wscale = as_glob(wscale_);

    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 {
        const size_t xo = (size_t)m * K + kk;
        if constexpr (XLDS)
            return ld_lds8(lds + xo);
        else
            return ld_glob8(x + xo);
    };

    /* TWO OUTPUT COLUMNS PER WAVE-STEP. The fp8 chunk is 1024 K-elements (b128 = 16 fp8/lane), so
     * nchunk = ceil(K/1024) is HALF the bf16 GEMV's — and for K=2560 (q/k/v/gate/up) that is only 3,
     * a hard ceiling of 3 loads in flight per column no matter how large UN is. That starves the
     * memory-level parallelism a persistent megakernel (no occupancy to hide latency) lives on. So
     * each wave carries TWO independent weight rows, doubling the outstanding loads to 2*min(UN,
     * nchunk) — the same reason the 2-stream fp8 GLU fares better than this single stream. The
     * workgroup's column OWNERSHIP is unchanged (still [slice*per, slice*per+per)), so the decode
     * fine-grained gemv->headnorm map is unaffected.
     *
     * NOT FIXED HERE, and it is a real defect: `n2 = has2 ? n + PLOW_WAVES : n` makes a wave with
     * an ODD column count recompute its last column TWICE and discard one copy. gemv_rows_fp8_blk
     * below now splits that tail into an R=1 instantiation and measured +30% at N=6144 on gfx950.
     * The same fix applies verbatim here and in gemv_rows_mxfp4 — left alone only because this arm
     * serves Qwen/Llama shapes that were not in that A/B. Measure before changing. */
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
    for (unsigned n = gv_n0 + wave; n < gv_n1; n += PLOW_WAVES * 2) {
        const bool has2 = (n + PLOW_WAVES) < gv_n1;
        const unsigned n2 = has2 ? n + PLOW_WAVES : n;
        /* num_records = K BYTES (fp8 is 1 byte): the overshoot past the row end returns zero and
         * issues no memory request — same free bounds check the bf16 GEMV relies on. */
        const __amdgpu_buffer_rsrc_t wr = buf_rsrc_fp8(W + (size_t)n * K, K);
        const __amdgpu_buffer_rsrc_t wr2 = buf_rsrc_fp8(W + (size_t)n2 * K, K);
        float acc[MM], acc2[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) { acc[m] = 0.0f; acc2[m] = 0.0f; }

        for (unsigned c = 0; c < nchunk; c += UN) {
            fp8v16 wv[UN], wv2[UN];
            /* Both columns' loads issued before either is touched: 2*UN in flight per lane. */
#pragma unroll
            for (int u = 0; u < UN; u++) wv[u] = buf_ld_fp8(wr, (c + (unsigned)u) * step + lane * 16);
#pragma unroll
            for (int u = 0; u < UN; u++) wv2[u] = buf_ld_fp8(wr2, (c + (unsigned)u) * step + lane * 16);
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * step + lane * 16;
                const unsigned kx = (k < K) ? k : 0u; /* keep a NaN out of 0*NaN; wv is 0 anyway */
                bf16v8 wlo, whi, wlo2, whi2;
                fp8_to_bf16v8(wv[u], wlo, whi);
                fp8_to_bf16v8(wv2[u], wlo2, whi2);
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    const bool live = ((unsigned)m < M);
                    const bf16v8 xlo = xv8(live ? m : 0, kx);
                    const bf16v8 xhi = xv8(live ? m : 0, kx + 8);
                    acc[m] += live ? dot8(whi, xhi, dot8(wlo, xlo, 0.0f)) : 0.0f;
                    acc2[m] += live ? dot8(whi2, xhi, dot8(wlo2, xlo, 0.0f)) : 0.0f;
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float t = wave_sum(acc[m]) * wscale[n]; /* per-row dequant, once, in the epilogue */
            const float t2 = wave_sum(acc2[m]) * wscale[n2];
            if (lane == 0 && (unsigned)m < M) {
                C[(size_t)m * N + n] = f2bf(t);
                if (has2) C[(size_t)m * N + n2] = f2bf(t2);
            }
        }
    }
}

#define FP8_ARMS(FN, C, X, W, S, N, K, LDS)                                                       \
    do {                                                                                          \
        if ((K) == 9728u)                                                                          \
            FN<PLOW_GEMV_MM, true, 5>(C, X, W, S, 1u, N, K, blockIdx.x, gridDim.x, LDS);            \
        else                                                                                       \
            FN<PLOW_GEMV_MM, true, GV_UNROLL_FP8>(C, X, W, S, 1u, N, K, blockIdx.x, gridDim.x, LDS);\
    } while (0)
#define FP8_KERNEL(NAME, FN)                                                                       \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void NAME(                                \
        bf16* C, const bf16* x, const unsigned char* W, const float* s, unsigned nrep, unsigned N,  \
        unsigned K) {                                                                              \
        __shared__ gv_smem sm;                                                                      \
        for (unsigned i = threadIdx.x; i < K; i += PLOW_THREADS) sm.xstage[i] = x[i];               \
        __syncthreads();                                                                            \
        for (unsigned e = 0; e < nrep; e++)                                                         \
            FP8_ARMS(FN, C, x, W + (size_t)e * N * K, s, N, K, sm.xstage);                           \
    }
#if MX_ARM != 1 && MX_ARM != 3 && MX_ARM != 4 && MX_ARM != 5
FP8_KERNEL(k_fp8_old, gemv_rows_fp8_dual)
#endif
#if MX_ARM == 4
FP8_KERNEL(k_fp8_new, gemv_rows_fp8_dual) /* A/A control, as for mxfp4 */
#elif MX_ARM != 0 && MX_ARM != 3 && MX_ARM != 5
FP8_KERNEL(k_fp8_new, gemv_rows_fp8)
#endif

/* ---- ANALYSIS ARM (task 6): x from GLOBAL instead of LDS, mxfp4 only. NOT a production change —
 * `XLDS` is chosen by `d_gemv_qkv_mxfp4`'s `lds_ok` test and stays that way. This arm exists only to
 * price the LDS bank-conflict claim, because the arithmetic says it may be the BINDING constraint
 * rather than the hidden one the claim itself assumes:
 *
 *   A lane's fp4 fragment is 32 K-elements, so its four ds_read_b128 of x are 64 BYTES apart
 *   (16 dwords). LDS is 32 banks x 4 B, so in each dword phase lane L touches bank (16L + p) mod 32
 *   -> only TWO distinct banks across 32 lanes, i.e. a 16-WAY conflict. The bf16 GEMV's fragment is
 *   8 K-elements, 16 B apart (4 dwords), giving (4L + p) mod 32 = 8 distinct banks and a 4-WAY
 *   conflict. So the ratio is 4x, matching the claim's 8:2, but both absolute figures in the claim
 *   are low by 2x. The fp8 GEMV sits between them: 16 B fragments 32 B apart, 8-way.
 *
 *   And fp4 needs MORE LDS traffic per weight byte, not just worse-patterned traffic: x is bf16
 *   whatever the weight encoding, so per chunk-pass a wave reads 1024 B of weights either way but
 *   4096 B of x at fp4 against 2048 B at fp8 -- 4:1 against 2:1. Twice the bytes through a pattern
 *   with twice the conflict degree is 4x the LDS time per weight byte, and the measured weight
 *   bandwidths differ by very nearly that much once the fix is in (mxfp4 3451 GB/s vs fp8 6595 at
 *   the same 32768x6144 shape). A kernel that is genuinely HBM-bound should not care what encoding
 *   its weights are in; this one does.
 *
 * If the conflict is binding, moving x to the global path (L1-resident: K=6144 halves is 12 KB)
 * should be worth real time at M=1. If it is hidden behind weight latency as the claim assumes,
 * this arm reads at or below parity. Either answer is a measurement rather than an argument. */
#if MX_ARM == 5
/* `k_mx_old` here is the SHIPPING kernel (XLDS=true), so the pair isolates the x path alone —
 * both arms already carry the odd-tail fix. They must still come out bit-exact: LDS and global are
 * two roads to the same bytes. */
MX_KERNEL(k_mx_old, gemv_rows_mxfp4)
#define MX_XLDS_ARMS(FN, C, X, W, S, N, K, LDS)                                                   \
    do {                                                                                          \
        const unsigned nchunk_ = ((K) + PLOW_WAVE * 32 - 1) / (PLOW_WAVE * 32);                    \
        if (nchunk_ >= 8u)                                                                        \
            FN<PLOW_GEMV_MM, false, 4>(C, nullptr, nullptr, X, W, nullptr, nullptr, S, nullptr,    \
                                       nullptr, 1u, N, 0u, 0u, K, blockIdx.x, gridDim.x, LDS);     \
        else if (nchunk_ >= 6u)                                                                    \
            FN<PLOW_GEMV_MM, false, 3>(C, nullptr, nullptr, X, W, nullptr, nullptr, S, nullptr,    \
                                       nullptr, 1u, N, 0u, 0u, K, blockIdx.x, gridDim.x, LDS);     \
        else                                                                                       \
            FN<PLOW_GEMV_MM, false, 2>(C, nullptr, nullptr, X, W, nullptr, nullptr, S, nullptr,    \
                                       nullptr, 1u, N, 0u, 0u, K, blockIdx.x, gridDim.x, LDS);     \
    } while (0)
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_mx_new(
    bf16* C, const bf16* x, const unsigned char* W, const unsigned char* S, unsigned nrep,
    unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    for (unsigned i = threadIdx.x; i < K; i += PLOW_THREADS) sm.xstage[i] = x[i];
    __syncthreads();
    for (unsigned e = 0; e < nrep; e++)
        MX_XLDS_ARMS(gemv_rows_mxfp4, C, x, W + (size_t)e * N * (K / 2),
                     S + (size_t)e * N * (K / 32), N, K, sm.xstage);
}
#endif
