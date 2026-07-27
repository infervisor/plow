/* px6_wavequant_bench.cu — PX-6: does tiled-GEMM wave quantization against a non-power-of-2
 * SM count cost measurable time, and what is the FFMA GEMV arm's relative per-SM rate?
 *
 * THE ARITHMETIC UNDER TEST.  d_gemm walks output tiles with a grid-stride loop keyed on the
 * packet's (slice, nblk) (op_gemm.cuh:893-900).  With T = ceil(M/BM)*ceil(N/BN) tiles on P
 * blocks the makespan is ceil(T/P) tile-times, so the tail wave leaves P*ceil(T/P) - T SM-slots
 * idle.  On this box P = 170 = 2*5*17, which divides almost nothing.
 *
 * THE SPLIT THEOREM (the thing this bench exists to falsify).  Give the bulk P_g blocks and the
 * residue P_v = P - P_g blocks running an arm at relative per-SM rate r.  The bulk must shed a
 * whole wave (ceil is integer), so T_g <= (W-1)P_g and T_v >= (W-1)P_v + s, s = T - (W-1)P.
 * Requiring the residue to finish under W tile-times:
 *
 *     split beats baseline  <=>  r > (W-1)/W + s/(W*P_v)
 *     full arm swap (P_g=0) <=>  r > T/(W*P) = u, the wave utilization
 *
 * r is NOT a property of the kernels: the mma tile computes BM rows whether they exist or not
 * (predication suppresses the store and the gmem read, never the mma -- op_gemm.cuh:929-957), so
 * r ~ r_raw * BM/min(M,BM).  PRE-REGISTERED PREDICTION: r_raw in [0.03, 0.35], every W>=2 cell
 * needs r >= 0.50, therefore the FFMA-residue split LOSES at every prefill-M cell and wins only
 * where the mma arm is row-starved (M << BM).
 *
 * WHAT px3 (gemm_occ_bench.cu) COULD NOT SEE, and this fixes:
 *   1. L2.  This card has 96 MiB L2 and zg0_bwcal measures 4090 GB/s warm at 32 MB vs 1696 GB/s
 *      at 2 GB.  q_proj B is 31.5 MB and o_proj B is 31.5 MB -- fully L2-resident across px3's
 *      30-iteration loop.  Here every weight is replicated past PX6_COLD_MB and cycled per
 *      iteration, so each timed launch reads DRAM.  Reported per row as l2_cold.
 *   2. M.  px3 measured only M=4096/8192, where u >= 0.94 and there is nothing to win.  The
 *      quantization lives at M <= 2048 (u = 0.18-0.71).
 *   3. The GEMV arm was never timed against the GEMM arm at the same shape, so r was unmeasured.
 *
 * SECTIONS
 *   [cliff]  N=21760 (=170*128 -> T=170, W=1, u=1.000) vs N=21888 (=171*128 -> T=171, W=2,
 *            u=0.503).  Two shapes 0.6% apart in size, predicted 2x apart in time.  The u=1.000
 *            cell is also the NULL CONTROL: if it shows idle, the harness is broken and nothing
 *            else here is interpretable.
 *   [stair]  time vs grid, 1..P.  Compute-bound quantization is a staircase flat between wave
 *            boundaries; a bandwidth-bound op is smooth.  This is the premise test.
 *   [shapes] real Gemma-4-12B prefill shapes x M sweep, at grid=P and at an oracle grid G*|T.
 *   [rho]    k_gemv vs k_gemm at identical shape/grid/cold protocol -> r.
 *
 * BUILD (plain env -- nix CPATH collides with the CUDA math headers):
 *   export PATH=/usr/local/cuda/bin:/usr/bin:/bin; unset CPATH LIBRARY_PATH LD_LIBRARY_PATH
 *   nvcc -arch=sm_120a -O3 -I runtime/common -I runtime/nvidia \
 *        perf-data/px6_wavequant_bench.cu -o /tmp/px6            # PGM_BN=128 default
 *   nvcc ... -DPGM_BN=64 ... -o /tmp/px6_bn64                    # E3 arm
 * RUN (always under the lease -- a contended run silently invalidates every number):
 *   perf-data/harness/gpulease px6 /tmp/px6
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

#include "sm120_common.cuh"
#include "op_gemm.cuh"

typedef __nv_bfloat16 bf16;
#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA FAIL %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e_)); exit(1);} } while(0)

/* Replication target: total weight bytes per shape must exceed L2 by enough that a cycled
 * iteration never hits a line the previous iteration left resident. 96 MiB L2 -> 700 MB. */
#ifndef PX6_COLD_MB
#define PX6_COLD_MB 700
#endif
static const int ITERS = 30, WARM = 8;

static uint32_t rng = 12345u;
static float frand() { rng = rng*1664525u + 1013904223u; return ((rng>>8)&0xffff)/65535.0f - 0.5f; }

static bf16* dev_rand(size_t n) {
    std::vector<bf16> hb(n);
    for (size_t i = 0; i < n; i++) hb[i] = __float2bfloat16(frand());
    bf16* d; CK(cudaMalloc(&d, n*sizeof(bf16)));
    CK(cudaMemcpy(d, hb.data(), n*sizeof(bf16), cudaMemcpyHostToDevice));
    return d;
}
/* Replicas 1..nrep-1 are device-to-device copies of replica 0: same bytes, different lines.
 * Generating fresh random data per replica costs seconds of host RNG and buys nothing -- the
 * point is cache residency, not content. */
static void dev_rand_reps(std::vector<bf16*>& out, size_t n, int nrep) {
    out.resize(nrep);
    out[0] = dev_rand(n);
    for (int i = 1; i < nrep; i++) {
        CK(cudaMalloc(&out[i], n*sizeof(bf16)));
        CK(cudaMemcpy(out[i], out[0], n*sizeof(bf16), cudaMemcpyDeviceToDevice));
    }
}
static void free_reps(std::vector<bf16*>& v) { for (auto p : v) if (p) cudaFree(p); v.clear(); }

__global__ void k_gemm(bf16* C, const bf16* A, const bf16* B,
                       unsigned m, unsigned n, unsigned k, unsigned a_row0) {
    extern __shared__ bf16 smg[]; d_gemm(C,A,B,m,n,k,a_row0,blockIdx.x,gridDim.x,smg);
}
__global__ void k_gemm_glu(bf16* C, const bf16* A, const bf16* Wg, const bf16* Wu,
                           unsigned m, unsigned n, unsigned k, unsigned act) {
    extern __shared__ bf16 smg[]; d_gemm_glu(C,A,Wg,Wu,m,n,k,act,blockIdx.x,gridDim.x,smg);
}
/* The residue arm. Plain d_gemv (no arena): the arena overload bails to this one at M>1
 * (op_gemm.cuh:326), and every cell here is M>1 except the lm_head row. */
__global__ void k_gemv(bf16* C, const bf16* x, const bf16* W,
                       unsigned m, unsigned n, unsigned k) {
    d_gemv(C,x,W,m,n,k,blockIdx.x,gridDim.x);
}

enum Arm { ARM_GEMM = 0, ARM_GLU = 1, ARM_GEMV = 2 };
static const char* arm_name(Arm a) { return a==ARM_GEMM?"gemm":(a==ARM_GLU?"glu":"gemv"); }

struct Bufs {
    std::vector<bf16*> Bg, Bu;   /* replicated weights, cycled per iteration */
    bf16 *A = nullptr, *C = nullptr;
    int nrep = 1;
};

static double time_arm(Arm arm, Bufs& b, unsigned m, unsigned n, unsigned k,
                       size_t smem, int grid) {
    for (int i = 0; i < WARM; i++) {
        int r = i % b.nrep;
        if (arm == ARM_GEMM)      k_gemm    <<<grid,256,smem>>>(b.C,b.A,b.Bg[r],m,n,k,0);
        else if (arm == ARM_GLU)  k_gemm_glu<<<grid,256,smem>>>(b.C,b.A,b.Bg[r],b.Bu[r],m,n,k,0);
        else                      k_gemv    <<<grid,256,0>>>   (b.C,b.A,b.Bg[r],m,n,k);
    }
    CK(cudaDeviceSynchronize());
    cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    CK(cudaEventRecord(s));
    for (int i = 0; i < ITERS; i++) {
        int r = i % b.nrep;
        if (arm == ARM_GEMM)      k_gemm    <<<grid,256,smem>>>(b.C,b.A,b.Bg[r],m,n,k,0);
        else if (arm == ARM_GLU)  k_gemm_glu<<<grid,256,smem>>>(b.C,b.A,b.Bg[r],b.Bu[r],m,n,k,0);
        else                      k_gemv    <<<grid,256,0>>>   (b.C,b.A,b.Bg[r],m,n,k);
    }
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms = 0; CK(cudaEventElapsedTime(&ms,s,e));
    cudaEventDestroy(s); cudaEventDestroy(e);
    CK(cudaGetLastError());
    return ms/ITERS;
}

static void alloc(Bufs& b, unsigned M, unsigned N, unsigned K, int glu) {
    size_t wn = (size_t)N*K;
    size_t per_iter = wn*sizeof(bf16)*(glu?2:1);
    b.nrep = (int)std::max<size_t>(2, ((size_t)PX6_COLD_MB<<20) / std::max<size_t>(per_iter,1));
    b.nrep = std::min(b.nrep, 24);
    dev_rand_reps(b.Bg, wn, b.nrep);
    if (glu) dev_rand_reps(b.Bu, wn, b.nrep);
    b.A = dev_rand((size_t)M*K);
    CK(cudaMalloc(&b.C, (size_t)M*N*sizeof(bf16)));
}
static void freeb(Bufs& b) {
    free_reps(b.Bg); free_reps(b.Bu);
    if (b.A) cudaFree(b.A); if (b.C) cudaFree(b.C);
    b.A = b.C = nullptr;
}

/* tiles / waves / utilization for a given grid, from the SAME formula d_gemm walks */
struct Quant { unsigned tm, tn, T; int W; double u; };
static Quant quant(unsigned M, unsigned N, int grid) {
    Quant q;
    q.tm = (M + PGM_BM - 1)/PGM_BM;
    q.tn = (N + PGM_BN - 1)/PGM_BN;
    q.T  = q.tm*q.tn;
    q.W  = (int)((q.T + grid - 1)/grid);
    q.u  = (double)q.T/((double)q.W*grid);
    return q;
}
/* largest divisor of T that is <= P: the grid where every block gets exactly T/G tiles and
 * quantization is zero by construction. */
static int oracle_grid(unsigned T, int P) {
    for (int g = std::min<int>(P,(int)T); g >= 1; g--) if (T % (unsigned)g == 0) return g;
    return 1;
}
static double tflops(Arm a, unsigned M, unsigned N, unsigned K, double ms) {
    double f = 2.0*M*N*K*(a==ARM_GLU?2.0:1.0);
    return f/(ms*1e-3)/1e12;
}
static double gbps(Arm a, unsigned M, unsigned N, unsigned K, double ms, unsigned passes) {
    double wb = (double)N*K*2.0*(a==ARM_GLU?2.0:1.0)*passes;
    double ab = (double)M*K*2.0, cb = (double)M*N*2.0;
    return (wb+ab+cb)/(ms*1e-3)/1e9;
}

struct Shape { const char* name; unsigned N, K; int glu; };

int main(int argc, char** argv) {
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,0));
    const int P = pr.multiProcessorCount;
    const size_t smem = (size_t)PGM_ARENA_BF16*sizeof(bf16);
    CK(cudaFuncSetAttribute(k_gemm,     cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    CK(cudaFuncSetAttribute(k_gemm_glu, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    int occ=0, occg=0, occv=0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_gemm,    256,smem));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occg,k_gemm_glu,256,smem));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occv,k_gemv,    256,0));
    cudaFuncAttributes fa, fg, fv;
    CK(cudaFuncGetAttributes(&fa,k_gemm)); CK(cudaFuncGetAttributes(&fg,k_gemm_glu));
    CK(cudaFuncGetAttributes(&fv,k_gemv));

    printf("# px6 wave-quantization bench\n");
    printf("# gpu=%s SMs=%d cc=%d.%d L2=%dMiB\n", pr.name, P, pr.major, pr.minor, pr.l2CacheSize>>20);
    printf("# PGM_BM=%d PGM_BN=%d PGM_BK=%d arena_bf16=%d smem=%.1fKiB GV_MM_MAX=%d\n",
           PGM_BM, PGM_BN, PGM_BK, PGM_ARENA_BF16, smem/1024.0, GV_MM_MAX);
    printf("# regs: gemm=%d glu=%d gemv=%d | occ: gemm=%d glu=%d gemv=%d\n",
           fa.numRegs, fg.numRegs, fv.numRegs, occ, occg, occv);
    printf("# ITERS=%d WARM=%d cold_target=%dMB\n", ITERS, WARM, PX6_COLD_MB);

    const char* only = (argc > 1) ? argv[1] : nullptr;
    auto want = [&](const char* s){ return !only || !strcmp(only,s); };

    /* ---------------- [cliff] null control + the quantization cliff ---------------- */
    if (want("cliff")) {
        printf("\n# [cliff] M=128 K=3840, grid=P. N chosen so T lands exactly on / one past P.\n");
        printf("cliff_N,tiles,waves,u,ms,tflops,ratio_vs_first\n");
        unsigned Ns[] = { (unsigned)(P*PGM_BN), (unsigned)((P+1)*PGM_BN) };
        double first = 0;
        for (unsigned N : Ns) {
            const unsigned M = 128, K = 3840;
            Bufs b; alloc(b,M,N,K,0);
            double ms = time_arm(ARM_GEMM,b,M,N,K,smem,P);
            Quant q = quant(M,N,P);
            if (!first) first = ms;
            printf("%u,%u,%d,%.4f,%.5f,%.1f,%.4f\n",
                   N,q.T,q.W,q.u,ms,tflops(ARM_GEMM,M,N,K,ms),ms/first);
            freeb(b);
        }
    }

    /* ---------------- [stair] time vs grid: staircase = compute-bound quantization -------- */
    if (want("stair")) {
        printf("\n# [stair] time vs grid. Compute-bound quantization -> flat between wave\n");
        printf("# boundaries, step at each. Bandwidth-bound -> smooth decline.\n");
        struct { const char* name; unsigned M,N,K; int glu; } sc[] = {
            {"down_M1024",  1024, 3840, 15360, 0},
            {"gateup_M1024",1024,15360,  3840, 1},
        };
        for (auto& s : sc) {
            Bufs b; alloc(b,s.M,s.N,s.K,s.glu);
            Quant qP = quant(s.M,s.N,P);
            printf("stair_shape=%s M=%u N=%u K=%u T=%u\n", s.name,s.M,s.N,s.K,qP.T);
            printf("grid,waves,u,ms,tflops\n");
            for (int g = 10; g <= P; g += 5) {
                double ms = time_arm(s.glu?ARM_GLU:ARM_GEMM,b,s.M,s.N,s.K,smem,g);
                Quant q = quant(s.M,s.N,g);
                printf("%d,%d,%.4f,%.5f,%.1f\n",g,q.W,q.u,ms,
                       tflops(s.glu?ARM_GLU:ARM_GEMM,s.M,s.N,s.K,ms));
            }
            freeb(b);
        }
    }

    /* ---------------- [shapes] real 12B shapes x M, grid=P vs oracle grid ---------------- */
    if (want("shapes")) {
        printf("\n# [shapes] Gemma-4-12B prefill. oracle grid G* divides T exactly (zero\n");
        printf("# quantization); tau = per-tile-per-SM time from the oracle run.\n");
        printf("# ideal_ms = tau*T/P (no quantization). idle_meas = 1 - ideal_ms/ms_P.\n");
        Shape shapes[] = {
            {"q_proj",     8192,  3840, 0},   /* hd512 global layers: worst tail in the model */
            {"o_proj",     3840,  8192, 0},
            {"down_proj",  3840, 15360, 0},
            {"gateup_glu",15360,  3840, 1},
            {"synth_N2176",2176,  3840, 0},   /* 17*128: shares a factor with P=170 */
        };
        unsigned Ms[] = {128, 512, 1024, 2048};
        printf("shape,M,N,K,glu,tiles,waves_P,u_P,oracle_grid,ms_P,ms_oracle,tau_us,ideal_ms,idle_meas,idle_pred,tflops_P,gbps_P\n");
        for (unsigned M : Ms) for (auto& sh : shapes) {
            Bufs b; alloc(b,M,sh.N,sh.K,sh.glu);
            Arm arm = sh.glu ? ARM_GLU : ARM_GEMM;
            Quant q = quant(M,sh.N,P);
            int G = oracle_grid(q.T,P);
            double msP = time_arm(arm,b,M,sh.N,sh.K,smem,P);
            double msO = time_arm(arm,b,M,sh.N,sh.K,smem,G);
            double tau = msO/((double)q.T/G);              /* ms per tile per SM */
            double ideal = tau*(double)q.T/P;
            double idle_meas = 1.0 - ideal/msP;
            double idle_pred = 1.0 - q.u;
            printf("%s,%u,%u,%u,%d,%u,%d,%.4f,%d,%.5f,%.5f,%.3f,%.5f,%.4f,%.4f,%.1f,%.1f\n",
                   sh.name,M,sh.N,sh.K,sh.glu,q.T,q.W,q.u,G,msP,msO,tau*1e3,ideal,
                   idle_meas,idle_pred,tflops(arm,M,sh.N,sh.K,msP),
                   gbps(arm,M,sh.N,sh.K,msP,1));
            freeb(b);
        }
    }

    /* ---------------- [rho] the decisive number for the split ---------------- */
    if (want("rho")) {
        printf("\n# [rho] FFMA GEMV vs mma GEMM, identical shape/grid/cold protocol.\n");
        printf("# r = tflops_gemv/tflops_gemm. gemv makes ceil(M/GV_MM_MAX) weight passes.\n");
        printf("# split needs r > (W-1)/W + s/(W*P_v); full arm swap needs r > u.\n");
        struct { const char* name; unsigned M,N,K; } rc[] = {
            {"down_M1024",   1024, 3840,15360},
            {"down_M128",     128, 3840,15360},
            {"o_M1024",      1024, 3840, 8192},
            {"q_M1024",      1024, 8192, 3840},
            {"lmhead_M1",       1,262144,3840},   /* the one row-starved site in plow today */
        };
        /* NORMALIZATION (got this wrong once -- the correction is the point of the column names).
         * gemm_ms = W*tau already CONTAINS the mma arm's quantization waste; gemv_ms = T*tau/(r*P)
         * does not. So the raw time ratio is r_norm = gemm_ms/gemv_ms = r_true/u, and
         *     r_true = r_norm * u.
         * The swap therefore wins iff r_true > u, i.e. iff r_norm > 1, i.e. iff gemv is simply
         * faster in wall time. Comparing the raw ratio against u double-counts u and reports a
         * win for cells that are 2.7x SLOWER. Verdict below is on wall time, which cannot lie. */
        printf("case,M,N,K,tiles,waves,u,gemm_ms,gemv_ms,gemv_passes,r_true,r_norm,split_thresh,swap_thresh,verdict\n");
        for (auto& c : rc) {
            Bufs b; alloc(b,c.M,c.N,c.K,0);
            double mg = time_arm(ARM_GEMM,b,c.M,c.N,c.K,smem,P);
            double mv = time_arm(ARM_GEMV,b,c.M,c.N,c.K,0,P);
            Quant q = quant(c.M,c.N,P);
            unsigned passes = (c.M + GV_MM_MAX - 1)/GV_MM_MAX;
            double r_norm = mg/mv;
            double r_true = r_norm*q.u;
            /* split threshold at the P_v that minimizes it (P_v -> P): (W-1)/W + s/(W*P) */
            unsigned s = q.T - (unsigned)(q.W-1)*P;
            double split_thresh = q.W>1 ? (double)(q.W-1)/q.W + (double)s/((double)q.W*P) : 1e9;
            printf("%s,%u,%u,%u,%u,%d,%.4f,%.5f,%.5f,%u,%.4f,%.4f,%.4f,%.4f,%s\n",
                   c.name,c.M,c.N,c.K,q.T,q.W,q.u,mg,mv,passes,r_true,r_norm,
                   split_thresh,q.u, mv < mg ? "SWAP-WINS" : "swap-loses");
            freeb(b);
        }
    }
    /* ---------------- [ladder] where do the prefill bucket rungs BELONG? ---------------- */
    if (want("ladder")) {
        printf("\n# [ladder] per-op cost vs M for every distinct (N,K,glu) in Gemma-4-12B.\n");
        printf("# Layer cost is sum_op ceil(tm*tn_op/P)*tau_op -- a STAIRCASE in tm, flat between\n");
        printf("# wave boundaries. Rows added inside a tread are FREE. The shipped ladder\n");
        printf("# [128,512,1024,2048,4096,8192] sits on powers of two, which is unrelated to where\n");
        printf("# the treads are. Weights are allocated ONCE per shape and M is swept, so the only\n");
        printf("# thing varying across a row is tm.\n");
        struct { const char* name; unsigned N,K; int glu; } ls[] = {
            {"qk_4096_3840",  4096, 3840, 0},   /* q sliding, k full */
            {"k_2048_3840",   2048, 3840, 0},   /* k sliding */
            {"o_3840_4096",   3840, 4096, 0},   /* o sliding */
            {"q_8192_3840",   8192, 3840, 0},   /* q full */
            {"o_3840_8192",   3840, 8192, 0},   /* o full */
            {"gu_15360_3840",15360, 3840, 1},   /* gate/up, both layer types */
            {"down_3840_15360",3840,15360,0},   /* down, both layer types */
        };
        const unsigned TM_MAX = 32;             /* t = 128 .. 4096 */
        const unsigned M_MAX  = TM_MAX*PGM_BM;
        printf("shape,tm,M,tiles,waves,ms\n");
        for (auto& s : ls) {
            /* allocate at M_MAX once; a larger A/C is harmless since both are indexed [m][*] */
            Bufs b;
            size_t wn = (size_t)s.N*s.K;
            size_t per_iter = wn*sizeof(bf16)*(s.glu?2:1);
            b.nrep = (int)std::max<size_t>(2, ((size_t)PX6_COLD_MB<<20)/std::max<size_t>(per_iter,1));
            b.nrep = std::min(b.nrep, 24);
            dev_rand_reps(b.Bg, wn, b.nrep);
            if (s.glu) dev_rand_reps(b.Bu, wn, b.nrep);
            b.A = dev_rand((size_t)M_MAX*s.K);
            CK(cudaMalloc(&b.C, (size_t)M_MAX*s.N*sizeof(bf16)));
            Arm arm = s.glu ? ARM_GLU : ARM_GEMM;
            for (unsigned tm = 1; tm <= TM_MAX; tm++) {
                unsigned M = tm*PGM_BM;
                double ms = time_arm(arm,b,M,s.N,s.K,smem,P);
                Quant q = quant(M,s.N,P);
                printf("%s,%u,%u,%u,%d,%.5f\n", s.name,tm,M,q.T,q.W,ms);
            }
            freeb(b);
        }
    }

    /* ---------------- [knee] how many SMs does each phase actually need? ---------------- */
    if (want("knee")) {
        printf("\n# [knee] Decode GEMV and prefill GEMM bottleneck on DIFFERENT resources, so the\n");
        printf("# split theorem that kills an intra-op two-arm split does NOT apply to running a\n");
        printf("# decode request and a prefill request concurrently on disjoint CU sets. This\n");
        printf("# measures the sizing parameter for that: the grid at which each phase saturates.\n");
        printf("# Decode saturates HBM well below 170 blocks; every block above its knee is FREE\n");
        printf("# for prefill at ~zero decode cost.\n");
        /* decode-shaped GEMV: M=1 over the two biggest decode weight reads */
        struct { const char* name; unsigned M,N,K; int arm; } kc[] = {
            {"decode_gemv_down",   1,  3840,15360, 2},
            {"decode_gemv_lmhead", 1,262144, 3840, 2},
            {"prefill_gemm_down",1024, 3840,15360, 0},
            {"prefill_glu_gateup",1024,15360,3840, 1},
        };
        printf("case,arm,M,N,K,grid,ms,gbps,tflops,pct_of_grid170\n");
        for (auto& c : kc) {
            Bufs b; alloc(b,c.M,c.N,c.K,c.arm==1);
            Arm arm = c.arm==2?ARM_GEMV:(c.arm==1?ARM_GLU:ARM_GEMM);
            double base = time_arm(arm,b,c.M,c.N,c.K,arm==ARM_GEMV?0:smem,P);
            for (int g : {8,16,24,32,48,64,80,96,112,128,144,160,170}) {
                if (g > P) continue;
                double ms = time_arm(arm,b,c.M,c.N,c.K,arm==ARM_GEMV?0:smem,g);
                printf("%s,%s,%u,%u,%u,%d,%.5f,%.1f,%.1f,%.4f\n",
                       c.name,arm==ARM_GEMV?"gemv":(arm==ARM_GLU?"glu":"gemm"),
                       c.M,c.N,c.K,g,ms,gbps(arm,c.M,c.N,c.K,ms,1),
                       tflops(arm,c.M,c.N,c.K,ms), base/ms);
            }
            freeb(b);
        }
    }

    /* ---------------- [headparity] the correctness gate for the lm_head arm swap ------------ */
    if (want("headparity")) {
        printf("\n# [headparity] PX-6 rec A swaps prefill lm_head from the tiled mma arm to the\n");
        printf("# M=1 GEMV arm. The gate is that BOTH arms reproduce an f32 CPU reference, and\n");
        printf("# each other, at the real lm_head contract: M=1, a_row0 = t-1 (the LAST prompt\n");
        printf("# row), K = hidden. a_row0 != 0 is the part a naive swap gets wrong -- the tiled\n");
        printf("# arm applies it internally (op_gemm.cuh:833-844), the GEMV arm needs the caller\n");
        printf("# to offset x, exactly as interp_sm120.cu does.\n");
        printf("case,M,N,K,a_row0,relL2_gemm_vs_ref,relL2_gemv_vs_ref,relL2_gemm_vs_gemv,verdict\n");
        struct { unsigned N,K,rows,a_row0; } pc[] = {
            {4096, 3840, 16, 0}, {4096, 3840, 16, 7}, {2176, 3840, 16, 15},
        };
        for (auto& c : pc) {
            /* x holds `rows` rows so a_row0 selects a genuinely different one */
            size_t an = (size_t)c.rows*c.K, wn = (size_t)c.N*c.K;
            std::vector<float> hA(an), hB(wn);
            for (size_t i=0;i<an;i++) hA[i]=frand();
            for (size_t i=0;i<wn;i++) hB[i]=frand();
            std::vector<bf16> bA(an), bB(wn);
            for (size_t i=0;i<an;i++) bA[i]=__float2bfloat16(hA[i]);
            for (size_t i=0;i<wn;i++) bB[i]=__float2bfloat16(hB[i]);
            /* f32 reference from the ROUNDED bf16 inputs -- the oracle must see what the GPU sees */
            std::vector<double> ref(c.N, 0.0);
            for (unsigned n=0;n<c.N;n++) {
                double s=0.0;
                for (unsigned k=0;k<c.K;k++)
                    s += (double)__bfloat162float(bA[(size_t)c.a_row0*c.K+k])
                       * (double)__bfloat162float(bB[(size_t)n*c.K+k]);
                ref[n]=s;
            }
            bf16 *dA,*dB,*dCg,*dCv;
            CK(cudaMalloc(&dA,an*sizeof(bf16))); CK(cudaMalloc(&dB,wn*sizeof(bf16)));
            CK(cudaMalloc(&dCg,(size_t)c.N*sizeof(bf16))); CK(cudaMalloc(&dCv,(size_t)c.N*sizeof(bf16)));
            CK(cudaMemcpy(dA,bA.data(),an*sizeof(bf16),cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dB,bB.data(),wn*sizeof(bf16),cudaMemcpyHostToDevice));
            /* poison, so an UNWRITTEN column cannot coincidentally pass */
            CK(cudaMemset(dCg,0x7F,(size_t)c.N*sizeof(bf16)));
            CK(cudaMemset(dCv,0x7F,(size_t)c.N*sizeof(bf16)));
            k_gemm<<<P,256,smem>>>(dCg,dA,dB,1u,c.N,c.K,c.a_row0);
            /* the interpreter offsets x by a_row0*K for the GEMV arm (interp_sm120.cu) */
            k_gemv<<<P,256,0>>>(dCv,dA+(size_t)c.a_row0*c.K,dB,1u,c.N,c.K);
            CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            std::vector<bf16> hg(c.N), hv(c.N);
            CK(cudaMemcpy(hg.data(),dCg,(size_t)c.N*sizeof(bf16),cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hv.data(),dCv,(size_t)c.N*sizeof(bf16),cudaMemcpyDeviceToHost));
            double ng=0,nv=0,ngv=0,den=0;
            for (unsigned n=0;n<c.N;n++) {
                double g=__bfloat162float(hg[n]), v=__bfloat162float(hv[n]), r=ref[n];
                ng+=(g-r)*(g-r); nv+=(v-r)*(v-r); ngv+=(g-v)*(g-v); den+=r*r;
            }
            double rg=sqrt(ng/den), rv=sqrt(nv/den), rgv=sqrt(ngv/den);
            printf("lmhead_like,1,%u,%u,%u,%.3e,%.3e,%.3e,%s\n", c.N,c.K,c.a_row0,rg,rv,rgv,
                   (rg<2e-2 && rv<2e-2 && rgv<2e-2) ? "PASS" : "*** FAIL ***");
            cudaFree(dA);cudaFree(dB);cudaFree(dCg);cudaFree(dCv);
        }
        /* NEGATIVE CONTROL: feed the GEMV arm the WRONG row. It must FAIL, or the a_row0
         * plumbing above is untested and the swap could silently read the wrong prompt row. */
        {
            const unsigned N=4096,K=3840,rows=16,good=7;
            size_t an=(size_t)rows*K, wn=(size_t)N*K;
            std::vector<bf16> bA(an), bB(wn);
            for (size_t i=0;i<an;i++) bA[i]=__float2bfloat16(frand());
            for (size_t i=0;i<wn;i++) bB[i]=__float2bfloat16(frand());
            bf16 *dA,*dB,*dCg,*dCv;
            CK(cudaMalloc(&dA,an*sizeof(bf16))); CK(cudaMalloc(&dB,wn*sizeof(bf16)));
            CK(cudaMalloc(&dCg,(size_t)N*sizeof(bf16))); CK(cudaMalloc(&dCv,(size_t)N*sizeof(bf16)));
            CK(cudaMemcpy(dA,bA.data(),an*sizeof(bf16),cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dB,bB.data(),wn*sizeof(bf16),cudaMemcpyHostToDevice));
            k_gemm<<<P,256,smem>>>(dCg,dA,dB,1u,N,K,good);
            k_gemv<<<P,256,0>>>(dCv,dA+(size_t)(good+1)*K,dB,1u,N,K);  /* WRONG row on purpose */
            CK(cudaDeviceSynchronize());
            std::vector<bf16> hg(N),hv(N);
            CK(cudaMemcpy(hg.data(),dCg,(size_t)N*sizeof(bf16),cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hv.data(),dCv,(size_t)N*sizeof(bf16),cudaMemcpyDeviceToHost));
            double d=0,den=0;
            for (unsigned n=0;n<N;n++){ double a=__bfloat162float(hg[n]),b=__bfloat162float(hv[n]);
                d+=(a-b)*(a-b); den+=a*a; }
            double r=sqrt(d/den);
            printf("negctrl_wrong_a_row0,1,%u,%u,%u,-,-,%.3e,%s\n", N,K,good+1,r,
                   r>1e-1 ? "PASS (differs, as required)" : "*** FAIL: control cannot detect ***");
            cudaFree(dA);cudaFree(dB);cudaFree(dCg);cudaFree(dCv);
        }
    }

    printf("\n# done\n");
    return 0;
}
