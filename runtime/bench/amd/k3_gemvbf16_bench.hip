/* bf16 decode GEMV (`gemv_rows`, DevOp::Gemv) at the REAL Kimi-K3 TP8 decode shapes.
 *
 * WHY THIS EXISTS. `Gemv` is 9.56 ms of K3's 19.21 ms of decode kernel-body time (PLOW_K3_ABLATE),
 * and the per-shape arithmetic says the family moves 18.16 GB/rank/token, which at a 6.4 TB/s
 * per-rank streaming ceiling is 2.84 ms. Neither `runtime/ubench/gemv_row_sweep.c` (~31 us
 * per-dispatch floor against a ~3 us body) nor the interpreter itself can say where the difference
 * goes. So: production body included VERBATIM from the shipped header, launch geometry mirrors the
 * persistent interpreter (blockDim = PLOW_THREADS, grid = the emitter's own workgroup count `b`
 * off `plowrt disasm <blob>/model.pkt --program 1`), in-kernel rep loop walking a fresh weight slab
 * out of a 1.5 GB arena, hipEvent median-of-41, palindromic interleave.
 *
 * THE ARM SPLIT IS THE POINT:
 *   k_steady   x staged once, then `nrep` back-to-back bodies. Reps pipeline into each other, so
 *              this is the steady-state stream: what the shape would cost as one long kernel.
 *   k_packet   every rep restages x into LDS with `__syncthreads()` on both sides, i.e. exactly
 *              what `d_gemv_t` does per PACKET. The BASELINE every candidate is read against.
 *   k_packet_aa  byte-identical device code to k_packet under a second name — the A/A control.
 *              It reads 0.996-1.003 here, which is what licenses reading the small deltas.
 *   k_stageonly  the packet arm with the GEMV deleted: attributes the restage.
 *   k_wstage   16-byte activation staging (shipped as `stage_x_lds`).
 *   k_r2       R-split with R=4/UN=3 below nchunk 4 — the variant that LOSES to R=2, kept as the
 *              record of why (ragged tail: half the waves fall into the R=1 path).
 *   k_rA       R=2/UN=7 below nchunk 8 + wide staging. THE SHIPPED COMBINATION.
 *   k_rL       R=4 -> R=2 -> R=1 ladder. Best arm at the current gfx950 b224/b256 geometry.
 *   k_rF       Current-WG128 fit: R=7 at nchunk=2, the ladder at nchunk=3, R=2 otherwise.
 *   k_noldsx   x from global instead of LDS. A 1.2-1.6x LOSS at every shape including gv_per==1,
 *              which is the measured answer to "the staging cannot be worth it for one column".
 *
 * ALL ARMS IN ONE OBJECT, interleaved by the host in one process: the deltas are single-digit
 * microseconds and clock drift between two binaries would swamp them. The host also checks the
 * candidate ELEMENTWISE against the baseline on device — the R-split only moves which wave takes
 * which column, so anything but bit-identical is a bug.
 *
 * RESULT (per-token totals over the real instance counts):
 *     current-grid gfx950: packet 3.660   shipped rA 2.998   rL 2.956 ms
 *
 * Build + run (ROCm tooling must be OUTSIDE `nix develop`):
 *   hipcc --offload-arch=gfx950 -O3 -w -std=c++17 --genco \
 *     -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_K3=1 -DGV_UNROLL=14 -DPLOW_FP8_KV=1 \
 *     runtime/bench/amd/k3_gemvbf16_bench.hip -o /tmp/gvbf.co -Iruntime/amd -Iruntime/common
 *   g++ -O2 -std=c++17 runtime/bench/amd/k3_gemvbf16_bench.cpp -o /tmp/gvbf \
 *     -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ -L/opt/rocm/lib -lamdhip64
 *   flock /tmp/plow_gpu.lock sg render -c 'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; \
 *     export ROCR_VISIBLE_DEVICES=6; /tmp/gvbf /tmp/gvbf.co 41'
 * Pin ROCR_VISIBLE_DEVICES to an IDLE gpu: an unpinned run on a contended box read A/A 0.27-4.57.
 */
#include <hip/hip_runtime.h>

#include "amd_common.h"
#include "op_gemm.h"

/* 8192 halves = 16 KB: K3's widest decode GEMV K is 7168. Matches d_gemv_t's `M*K <=
 * GM_LDS_HALVES` staging exactly at M=1. */
typedef union __align__(16) { bf16 xstage[8192]; float part[PLOW_WAVES]; } gv_smem;

/* ---- CANDIDATE: R output columns per wave-step, bf16. The exact shape of `gemv_blk_rows_r`
 * (op 44) and of the mxfp4 R-split, transplanted to the plain bf16 row. Rationale: a wave keeps
 * min(UN, nchunk) 16-byte loads outstanding per column, and nchunk = ceil(K/512) is 2 at K=768,
 * 3 at K=1536 and 7 at K=3584 — so at K3's own shapes the unroll is capped by the ROW, not by UN,
 * and the wave runs `qmax` SERIAL latency rounds. R columns multiply the outstanding loads by R
 * and divide the round count by R. Column OWNERSHIP is unchanged ([gv_n0, gv_n1) per workgroup),
 * so PLOW_FINE's gemv->headnorm map is unaffected; only which wave takes which column moves. ---- */
template <int MM, bool XLDS, int UN, int R>
__device__ __forceinline__ void gemv_rows_r_bench(bf16* __restrict__ C_, const bf16* __restrict__ x_,
                                                  const bf16* __restrict__ W_, unsigned M,
                                                  unsigned N, unsigned K, unsigned n, unsigned lane,
                                                  const bf16* lds) {
    const unsigned step = PLOW_WAVE * 8;
    const unsigned nchunk = (K + step - 1) / step;
    auto* const C = as_glob(C_);
    const auto* const x = as_glob(x_);
    const auto* const W = as_glob(W_);
    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 {
        const size_t xo = (size_t)m * K + kk;
        if constexpr (XLDS)
            return ld_lds8(lds + xo);
        else
            return ld_glob8(x + xo);
    };
    __amdgpu_buffer_rsrc_t wr[R];
#pragma unroll
    for (int r = 0; r < R; r++) wr[r] = buf_rsrc(W + (size_t)(n + (unsigned)r * PLOW_WAVES) * K, K);
    float acc[R][MM];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < MM; m++) acc[r][m] = 0.0f;

    for (unsigned c = 0; c < nchunk; c += UN) {
        bf16v8 wv[R][UN];
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int u = 0; u < UN; u++)
                wv[r][u] = buf_ld8(wr[r], ((c + (unsigned)u) * step + lane * 8) * 2u);
#pragma unroll
        for (int u = 0; u < UN; u++) {
            const unsigned k = (c + (unsigned)u) * step + lane * 8;
            const unsigned kx = (k < K) ? k : 0u;
#pragma unroll
            for (int m = 0; m < MM; m++) {
                const bool live = ((unsigned)m < M);
                const bf16v8 xv = xv8(live ? m : 0, kx);
#pragma unroll
                for (int r = 0; r < R; r++) acc[r][m] += live ? dot8(wv[r][u], xv, 0.0f) : 0.0f;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MM; m++)
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float t = wave_sum(acc[r][m]);
            if (lane == 0 && (unsigned)m < M)
                C[(size_t)m * N + n + (unsigned)r * PLOW_WAVES] = f2bf(t);
        }
}

template <int MM, bool XLDS, int UN, int R>
__device__ __forceinline__ void gemv_rows_rsplit(bf16* __restrict__ C, const bf16* __restrict__ x,
                                                 const bf16* __restrict__ W, unsigned M, unsigned N,
                                                 unsigned K, unsigned slice, unsigned nblk,
                                                 const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
    unsigned n = gv_n0 + wave;
    /* FULL R-GROUPS ONLY, then the leftover columns one at a time (R=1). */
    for (; n + (unsigned)(R - 1) * PLOW_WAVES < gv_n1; n += PLOW_WAVES * (unsigned)R)
        gemv_rows_r_bench<MM, XLDS, UN, R>(C, x, W, M, N, K, n, lane, lds);
    for (; n < gv_n1; n += PLOW_WAVES)
        gemv_rows_r_bench<MM, XLDS, UN * R, 1>(C, x, W, M, N, K, n, lane, lds);
}

/* ---------------------------------------------------------------------------------------- */
/* nrep walks a fresh [N,K] bf16 slab so the number is a STREAM, not an L2 hit. */
#define STAGE(sm, x, K)                                                                            \
    for (unsigned i = threadIdx.x; i < (K); i += PLOW_THREADS) sm.xstage[i] = x[i]

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_steady(bf16* C, const bf16* x,
                                                                    const bf16* W, unsigned nrep,
                                                                    unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    STAGE(sm, x, K);
    __syncthreads();
    for (unsigned e = 0; e < nrep; e++)
        gemv_rows<PLOW_GEMV_MM, true, false, GV_UNROLL>(C, x, W + (size_t)e * N * K, nullptr,
                                                        nullptr, 1u, N, K, blockIdx.x, gridDim.x,
                                                        sm.xstage);
}

/* Same body, PACKET FRAMING: d_gemv_t's restage + __syncthreads() around every one. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_packet(bf16* C, const bf16* x,
                                                                    const bf16* W, unsigned nrep,
                                                                    unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    for (unsigned e = 0; e < nrep; e++) {
        STAGE(sm, x, K);
        __syncthreads();
        gemv_rows<PLOW_GEMV_MM, true, false, GV_UNROLL>(C, x, W + (size_t)e * N * K, nullptr,
                                                        nullptr, 1u, N, K, blockIdx.x, gridDim.x,
                                                        sm.xstage);
        __syncthreads();
    }
}

/* A/A CONTROL: byte-identical device code to k_packet under a second name. Anything but 1.000x
 * between these two is instrument, not kernel. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_packet_aa(bf16* C, const bf16* x,
                                                                       const bf16* W, unsigned nrep,
                                                                       unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    for (unsigned e = 0; e < nrep; e++) {
        STAGE(sm, x, K);
        __syncthreads();
        gemv_rows<PLOW_GEMV_MM, true, false, GV_UNROLL>(C, x, W + (size_t)e * N * K, nullptr,
                                                        nullptr, 1u, N, K, blockIdx.x, gridDim.x,
                                                        sm.xstage);
        __syncthreads();
    }
}

/* R-SPLIT candidate, packet framing. UN is chosen per nchunk so R*UN vectors stays at or below
 * the 14 the shipping arm already budgets (14 x bf16v8 = 56 VGPRs), i.e. register-neutral. */
#define RS_ARMS(C, X, W, N, K, LDS)                                                                \
    do {                                                                                           \
        const unsigned nch_ = ((K) + PLOW_WAVE * 8 - 1) / (PLOW_WAVE * 8);                         \
        if (nch_ >= 8u)                                                                            \
            gemv_rows<PLOW_GEMV_MM, true, false, GV_UNROLL>(C, X, W, nullptr, nullptr, 1u, N, K,   \
                                                            blockIdx.x, gridDim.x, LDS);           \
        else if (nch_ >= 4u)                                                                       \
            gemv_rows_rsplit<PLOW_GEMV_MM, true, 7, 2>(C, X, W, 1u, N, K, blockIdx.x, gridDim.x,   \
                                                       LDS);                                       \
        else                                                                                       \
            gemv_rows_rsplit<PLOW_GEMV_MM, true, 3, 4>(C, X, W, 1u, N, K, blockIdx.x, gridDim.x,   \
                                                       LDS);                                       \
    } while (0)

/* MINIMAL R-SPLIT: ONE new pair-body for every nch < 8, tail at R=1/UN=14 so the tail arm is the
 * same register footprint as the shipping one. Two new instantiations instead of four. */
#define RS_ARMS2(C, X, W, N, K, LDS)                                                               \
    do {                                                                                           \
        const unsigned nch_ = ((K) + PLOW_WAVE * 8 - 1) / (PLOW_WAVE * 8);                         \
        if (nch_ >= 8u)                                                                            \
            gemv_rows<PLOW_GEMV_MM, true, false, GV_UNROLL>(C, X, W, nullptr, nullptr, 1u, N, K,   \
                                                            blockIdx.x, gridDim.x, LDS);           \
        else                                                                                       \
            gemv_rows_rsplit<PLOW_GEMV_MM, true, 7, 2>(C, X, W, 1u, N, K, blockIdx.x, gridDim.x,   \
                                                       LDS);                                       \
    } while (0)

/* Harness-only short-K ladder. It is not promoted into the production interpreter because the
 * measured 1.4% body gain would invalidate the tuned kernel profile without closing decode. */
template <int MM, bool XLDS>
__device__ __forceinline__ void gemv_rows_lad(bf16* __restrict__ C, const bf16* __restrict__ x,
                                              const bf16* __restrict__ W, unsigned M, unsigned N,
                                              unsigned K, unsigned slice, unsigned nblk,
                                              const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
    unsigned n = gv_n0 + wave;
    for (; n + 3u * PLOW_WAVES < gv_n1; n += 4u * PLOW_WAVES)
        gemv_rows_r_bench<MM, XLDS, 3, 4>(C, x, W, M, N, K, n, lane, lds);
    for (; n + PLOW_WAVES < gv_n1; n += 2u * PLOW_WAVES)
        gemv_rows_r_bench<MM, XLDS, 7, 2>(C, x, W, M, N, K, n, lane, lds);
    for (; n < gv_n1; n += PLOW_WAVES)
        gemv_rows_r_bench<MM, XLDS, GV_UNROLL, 1>(C, x, W, M, N, K, n, lane, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_rL(bf16* C, const bf16* x,
                                                                const bf16* W, unsigned nrep,
                                                                unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    for (unsigned e = 0; e < nrep; e++) {
        const unsigned nv = K >> 3;
        for (unsigned i = threadIdx.x; i < nv; i += PLOW_THREADS)
            *(bf16v8*)(sm.xstage + i * 8) = ld_glob8(as_glob(x) + i * 8);
        for (unsigned i = (nv << 3) + threadIdx.x; i < K; i += PLOW_THREADS) sm.xstage[i] = x[i];
        __syncthreads();
        const bf16* Wp = W + (size_t)e * N * K;
        if (((K + PLOW_WAVE * 8 - 1) / (PLOW_WAVE * 8)) >= 8u)
            gemv_rows<PLOW_GEMV_MM, true, false, GV_UNROLL>(C, x, Wp, nullptr, nullptr, 1u, N, K,
                                                            blockIdx.x, gridDim.x, sm.xstage);
        else
            gemv_rows_lad<PLOW_GEMV_MM, true>(C, x, Wp, 1u, N, K, blockIdx.x, gridDim.x,
                                              sm.xstage);
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_rF(bf16* C, const bf16* x,
                                                                const bf16* W, unsigned nrep,
                                                                unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    for (unsigned e = 0; e < nrep; e++) {
        const unsigned nv = K >> 3;
        for (unsigned i = threadIdx.x; i < nv; i += PLOW_THREADS)
            *(bf16v8*)(sm.xstage + i * 8) = ld_glob8(as_glob(x) + i * 8);
        for (unsigned i = (nv << 3) + threadIdx.x; i < K; i += PLOW_THREADS) sm.xstage[i] = x[i];
        __syncthreads();
        const bf16* Wp = W + (size_t)e * N * K;
        const unsigned nch = (K + PLOW_WAVE * 8 - 1) / (PLOW_WAVE * 8);
        if (nch >= 8u)
            gemv_rows<PLOW_GEMV_MM, true, false, GV_UNROLL>(C, x, Wp, nullptr, nullptr, 1u, N, K,
                                                            blockIdx.x, gridDim.x, sm.xstage);
        else if (nch == 2u)
            gemv_rows_rsplit<PLOW_GEMV_MM, true, 2, 7>(C, x, Wp, 1u, N, K, blockIdx.x,
                                                       gridDim.x, sm.xstage);
        else if (nch == 3u)
            gemv_rows_lad<PLOW_GEMV_MM, true>(C, x, Wp, 1u, N, K, blockIdx.x, gridDim.x,
                                              sm.xstage);
        else
            gemv_rows_rsplit<PLOW_GEMV_MM, true, 7, 2>(C, x, Wp, 1u, N, K, blockIdx.x,
                                                       gridDim.x, sm.xstage);
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_rA(bf16* C, const bf16* x,
                                                                const bf16* W, unsigned nrep,
                                                                unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    for (unsigned e = 0; e < nrep; e++) {
        const unsigned nv = K >> 3;
        for (unsigned i = threadIdx.x; i < nv; i += PLOW_THREADS)
            *(bf16v8*)(sm.xstage + i * 8) = ld_glob8(as_glob(x) + i * 8);
        for (unsigned i = (nv << 3) + threadIdx.x; i < K; i += PLOW_THREADS) sm.xstage[i] = x[i];
        __syncthreads();
        RS_ARMS2(C, x, W + (size_t)e * N * K, N, K, sm.xstage);
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_r2(bf16* C, const bf16* x,
                                                                const bf16* W, unsigned nrep,
                                                                unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    for (unsigned e = 0; e < nrep; e++) {
        STAGE(sm, x, K);
        __syncthreads();
        RS_ARMS(C, x, W + (size_t)e * N * K, N, K, sm.xstage);
        __syncthreads();
    }
}

/* NO LDS STAGING: x straight from global (XLDS=false), the arm d_gemv_t only takes when x does not
 * FIT in LDS. At `gv_per == 1` the staging is a pure loss — all 8 waves copy K halves into LDS so
 * that ONE wave can read them ONCE — and K3 emits three such shapes (f_a, k_rope_d, b_proj, 162
 * packets/token). No new instantiation: this template arm is already compiled. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_noldsx(bf16* C, const bf16* x,
                                                                    const bf16* W, unsigned nrep,
                                                                    unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    (void)sm;
    for (unsigned e = 0; e < nrep; e++) {
        gemv_rows<PLOW_GEMV_MM, false, false, GV_UNROLL>(C, x, W + (size_t)e * N * K, nullptr,
                                                         nullptr, 1u, N, K, blockIdx.x, gridDim.x,
                                                         nullptr);
        __syncthreads();
    }
}

/* COMBINED: wide staging + the R-split + the gv_per==1 global-x bypass. The production candidate. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_comb(bf16* C, const bf16* x,
                                                                  const bf16* W, unsigned nrep,
                                                                  unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    for (unsigned e = 0; e < nrep; e++) {
        const unsigned nv = K >> 3;
        for (unsigned i = threadIdx.x; i < nv; i += PLOW_THREADS)
            *(bf16v8*)(sm.xstage + i * 8) = ld_glob8(as_glob(x) + i * 8);
        for (unsigned i = (nv << 3) + threadIdx.x; i < K; i += PLOW_THREADS) sm.xstage[i] = x[i];
        __syncthreads();
        RS_ARMS(C, x, W + (size_t)e * N * K, N, K, sm.xstage);
        __syncthreads();
    }
}

/* STAGE-ONLY: the packet arm with the GEMV removed. `packet - stageonly` attributes how much of a
 * packet is the activation restage rather than the weight stream. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_stageonly(bf16* C, const bf16* x,
                                                                       const bf16* W, unsigned nrep,
                                                                       unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    for (unsigned e = 0; e < nrep; e++) {
        STAGE(sm, x, K);
        __syncthreads();
        if (sm.xstage[threadIdx.x & 7] == (bf16)0x7fffu && threadIdx.x == 1023u) C[0] = 1;
        __syncthreads();
    }
}

/* VECTORISED x STAGING candidate: d_gemv_t stages x with 2-BYTE loads and 2-byte LDS writes
 * (`lds[i] = x[i]`, i strided by PLOW_THREADS), which is 14 global_load_ushort + 14 ds_write_b16
 * per thread at K=7168 where two dwordx4 pairs would do. K is a multiple of 8 for every K3
 * projection and both pointers are 16-B aligned, so the wide form is legal. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_wstage(bf16* C, const bf16* x,
                                                                    const bf16* W, unsigned nrep,
                                                                    unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    for (unsigned e = 0; e < nrep; e++) {
        const unsigned nv = K >> 3;
        for (unsigned i = threadIdx.x; i < nv; i += PLOW_THREADS)
            *(bf16v8*)(sm.xstage + i * 8) = ld_glob8(as_glob(x) + i * 8);
        for (unsigned i = (nv << 3) + threadIdx.x; i < K; i += PLOW_THREADS) sm.xstage[i] = x[i];
        __syncthreads();
        gemv_rows<PLOW_GEMV_MM, true, false, GV_UNROLL>(C, x, W + (size_t)e * N * K, nullptr,
                                                        nullptr, 1u, N, K, blockIdx.x, gridDim.x,
                                                        sm.xstage);
        __syncthreads();
    }
}
