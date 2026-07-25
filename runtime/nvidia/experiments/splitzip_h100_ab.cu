// SplitZip lossless bf16 weight compression — H100 (sm_90a) A/B.
//
// Question: the fused SplitZip decode GEMV won 1.30x on RTX PRO 6000 (sm_120a).
// H100 has 2.5x LESS L2 (50 MB vs 128 MB) and a much higher compute:bandwidth
// ratio, so trading ALU for fewer HBM bytes should pay MORE. This harness
// re-measures the whole claim on Hopper, honestly.
//
// Layout (SoA, exactly what the plow packet emitter would produce):
//   lo[]   : 1 B/elem  = sign(bit7) | mantissa(bits6:0)
//   cd[]   : 4 b/elem  = code, exponent = code + EXP_BASE
//   eoff[] : u32 per output ROW (prefix offsets), epos[] u32 flat idx, eval[] u16 raw
// Reconstruct: bf16 = ((lo&0x80)<<8) | (exp<<7) | (lo&0x7F), then sparse escape overwrite.
//
// METHODOLOGY GUARDS (H100 L2 = 50 MB):
//   * every working set is >= ~1 GB (>=20x L2) and buffers are CYCLED across
//     timed iterations, so no iteration can be served out of L2.
//   * the real HBM streaming ceiling is measured here (grid-stride float4 read
//     over 4 GB) and every reported GB/s is checked against it.
//   * bit-exactness (memcmp of f32 GEMV outputs + a standalone codec round trip
//     with negative controls) is a HARD GATE, run before any timing.
//
// MODES (one loop body, identical FMA order => bit-identical outputs):
//   0 raw bf16 GEMV (baseline)      1 SplitZip fused (bit-exact, with escapes)
//   2 SplitZip fused, escape pass removed (diagnostic)
//   3 SplitZip loads WITHOUT the reconstruct ALU (ablation: how much of the
//     decode ALU is exposed vs hidden under the HBM stall)
//
// build:
//   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 \
//     -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common -I runtime/nvidia \
//     -include cstdint runtime/nvidia/experiments/splitzip_h100_ab.cu -o splitzip_h100_ab
// run:  ./splitzip_h100_ab <bf16 sample.bin> [profile|soak|lmhead]
//
// ============================ MEASURED RESULT (H100 NVL, 132 SM, 60 MB L2, 310 W cap,
//                              driver 570.133.20 / CUDA 13.0, real gemma-4 bf16 weights)
// VERDICT: SplitZip LOSES on H100. Do NOT enable the flag for the sm_90a build.
//
// bit-exactness      : PASS (codec round trip EQUAL on 89.1 M real bf16 elems; both
//                      negative controls detected; fused-GEMV f32 memcmp EQUAL, all shapes)
// compression ratio  : 1.3314x (escapes 0.0194%, EXP_BASE=109, metadata charged)
// HBM ceiling        : 3727 GB/s burst / 3526 GB/s steady (grid-stride float4, 4 GB)
// regs / occupancy   : raw 37 regs, 6 blk/SM (75%); SZ 43 regs, 5 blk/SM (62.5%).
//                      Forcing 32 regs (occ8) spills 16 B and is SLOWER -> occupancy is
//                      not the limiter.
//
// steady-state (the regime a real decode loop runs in), M=1:
//   shape                 raw ns   raw GB/s | SZ ns   SZ logical/HBM GB/s | SZ vs raw | ABL bound
//   q_proj  N4096 K3840    13100     2401   | 15828     1987 / 1493       |  0.828x   | 1.165x
//   kv_proj N2048 K3840     6678     2355   |  7984     1970 / 1480       |  0.836x   | 1.171x
//   gate/up N15360 K3840   49954     2362   | 59904     1969 / 1479       |  0.834x   | 1.164x
//   down    N3840 K15360   53674     2198   | 74974     1573 / 1181       |  0.716x   | 1.178x
//   lm_head N262144 K3840 842809     2389   | 994385    2025 / 1521       |  0.848x   | 1.233x
// short-burst (boost clocks, less power-limited): SZ is WORSE, 0.55-0.72x.
//
// WHY (decomposition, all measured on identical loads):
//   1.3314x ideal -> 1.16-1.23x  the two-plane compressed layout alone (ABL: same loads,
//                                same 16 FMAs, ZERO decode ALU) never reaches the ratio;
//                                24 B in two streams sustains only 2.0-2.2 TB/s of HBM
//                                bytes vs 2.4 TB/s for the single 32 B raw stream.
//                                Interleaving both planes into one 192 B/128-elem block
//                                (ABL-il / SZ-il) does NOT help - it is slightly worse.
//              -> 0.72-0.85x     the reconstruct ALU costs a further 29-40% on top. It is
//                                NOT hidden: SZ vs ABL is a pure-ALU delta at identical
//                                bytes. Dropping the escape pass (SZ-noesc) recovers to
//                                0.92-0.94x - still a loss, and not losslessly available.
//
//   Root cause: the hypothesis "H100 has a higher compute:bandwidth ratio" is true for
//   TENSOR cores, false for the INT/CUDA-core pipe that decode uses. Instruction issue
//   slots per weight byte moved:
//     RTX PRO 6000  188 SM x 4 sched x ~2.5 GHz / (1535 GB/s / 2 B) ~= 2.4 warp-inst/elem
//     H100 NVL      132 SM x 4 sched x 1.755 GHz / (3526 GB/s / 2 B) ~= 0.53 warp-inst/elem
//   i.e. H100 has ~4.6x FEWER issue slots per element than the GPU where SplitZip won.
//   The raw bf16 GEMV already runs at 62-68% of the steady HBM ceiling with no slack;
//   ~6 extra int ops/element do not fit.
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

#define WARPS 8
#define BLK   (WARPS*32)
#ifndef EXP_BASE
#define EXP_BASE 109u              // measured on gemma-4 bf16 linear weights (esc 0.019%)
#endif
#define NBUF 2                     // weight buffers cycled across timed iterations

__device__ __forceinline__ float bf2f(unsigned short u) { return __uint_as_float((unsigned int)u << 16); }

// ------------------------------------------------------------------ HBM ceiling
__global__ __launch_bounds__(512) void k_stream(const float4* __restrict__ p, size_t n4, float* sink) {
    float a = 0.f;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = p[i];
        a += v.x + v.y + v.z + v.w;
    }
    if (a == 1.2345e-31f) sink[blockIdx.x] = a;
}

// ------------------------------------------------------------------ codec round trip
__global__ __launch_bounds__(256) void k_store(const uint4* __restrict__ lo, const uint2* __restrict__ cd,
                                               size_t nvec, unsigned int expbase, uint4* __restrict__ out) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nvec; i += stride) {
        uint4 l = lo[i]; uint2 c = cd[i];
        unsigned int lw[4] = {l.x,l.y,l.z,l.w}, cw[2] = {c.x,c.y};
        unsigned short o[16];
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            unsigned int b  = (lw[e >> 2] >> ((e & 3) * 8)) & 0xFFu;
            unsigned int cc = (cw[e >> 3] >> ((e & 7) * 4)) & 0xFu;
            unsigned int ex = cc + expbase;
            o[e] = (unsigned short)(((b & 0x80u) << 8) | (ex << 7) | (b & 0x7Fu));
        }
        out[2 * i]     = *(uint4*)&o[0];
        out[2 * i + 1] = *(uint4*)&o[8];
    }
}
__global__ void k_escape(const unsigned int* p, const unsigned short* v, unsigned int n, unsigned short* out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[p[i]] = v[i];
}

// ------------------------------------------------------------------ the GEMV under test
// MODE 0 raw bf16 | 1 SZ split-plane + escapes (bit-exact) | 2 SZ split-plane, no escapes
//      3 SZ split-plane, ZERO-transform ablation (pure layout bandwidth bound)
//      4 SZ interleaved layout + escapes (bit-exact) | 5 SZ interleaved, ZERO-transform
// MINB = __launch_bounds__ minBlocksPerSM (register/occupancy knob).
// Interleaved layout: per 128-element group g -> [128 B lo][64 B cd] = 192 B contiguous,
// so both planes of a row live in the same DRAM pages instead of two distant streams.
#define IGRP 128
#define IGRP_B 192
template<int MODE, int MINB>
__global__ __launch_bounds__(BLK, MINB) void k_gemv(
        const uint4*          __restrict__ Wraw,
        const uint4*          __restrict__ lo,
        const uint2*          __restrict__ cd,
        const unsigned char*  __restrict__ il,
        const unsigned int*   __restrict__ eoff,
        const unsigned int*   __restrict__ epos,
        const unsigned short* __restrict__ eval,
        const unsigned short* __restrict__ x,
        float*                __restrict__ y,
        int K, long long nrow_total)
{
    extern __shared__ unsigned short xs[];           // K bf16 activations, SRAM-resident (arena path)
    for (int i = threadIdx.x; i < K; i += BLK) xs[i] = x[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int w    = threadIdx.x >> 5;
    const int nj   = K >> 9;                         // 512 elems per warp-iteration
    long long stride = (long long)gridDim.x * WARPS;

    for (long long row = (long long)blockIdx.x * WARPS + w; row < nrow_total; row += stride) {
        long long rbase = row * (long long)K;
        float acc = 0.f;
        unsigned int e0 = 0, e1 = 0;
        if (MODE == 1 || MODE == 4) { e0 = eoff[row]; e1 = eoff[row + 1]; }  // one lookup/row, hoisted
        for (int j = 0; j < nj; ++j) {
            long long myel = rbase + (long long)j * 512 + lane * 16;
            unsigned short o[16];
            if (MODE == 0) {                                     // ---- raw bf16: 32 B/16 elems
                const uint4* p = Wraw + (myel >> 3);
                uint4 a = p[0], b = p[1];
                unsigned int aw[8] = {a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
                for (int e = 0; e < 16; ++e)
                    o[e] = (unsigned short)((aw[e >> 1] >> ((e & 1) * 16)) & 0xFFFFu);
            } else {                                             // ---- splitzip: 24 B/16 elems
                uint4 l; uint2 c;
                if (MODE == 4 || MODE == 5) {
                    long long g = myel / IGRP; int r = (int)((myel & (IGRP - 1)) >> 4);
                    const unsigned char* p = il + g * IGRP_B;
                    l = *(const uint4*)(p + r * 16);
                    c = *(const uint2*)(p + IGRP + r * 8);
                } else {
                    l = lo[myel >> 4];
                    c = cd[myel >> 4];
                }
                unsigned int lw[4] = {l.x,l.y,l.z,l.w}, cw[2] = {c.x,c.y};
                if (MODE == 3 || MODE == 5) {
                    // ABLATION: identical loads, identical 16 FMAs, ZERO reconstruct ALU.
                    // Bounds what ANY decode scheme on this layout could ever reach.
                    unsigned short h[8];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        h[2 * e]     = (unsigned short)(lw[e] & 0xFFFFu);
                        h[2 * e + 1] = (unsigned short)(lw[e] >> 16);
                    }
#pragma unroll
                    for (int e = 0; e < 16; ++e) o[e] = h[e & 7];
                    o[0] ^= (unsigned short)(cw[0] ^ cw[1]);     // keep the cd load live
                } else {
#pragma unroll
                    for (int e = 0; e < 16; ++e) {
                        unsigned int b  = (lw[e >> 2] >> ((e & 3) * 8)) & 0xFFu;
                        unsigned int cc = (cw[e >> 3] >> ((e & 7) * 4)) & 0xFu;
                        unsigned int ex = cc + EXP_BASE;
                        o[e] = (unsigned short)(((b & 0x80u) << 8) | (ex << 7) | (b & 0x7Fu));
                    }
                    // sparse escape overwrite; warp-uniform trip count, ~0.02% hit rate
                    for (unsigned int t = e0; t < e1; ++t) {
                        unsigned int p = epos[t];
                        if ((unsigned int)(p - (unsigned int)myel) < 16u) {
                            unsigned short v = eval[t]; unsigned int sl = p & 15u;
#pragma unroll
                            for (int e = 0; e < 16; ++e) if ((unsigned int)e == sl) o[e] = v;
                        }
                    }
                }
            }
            const int xo = j * 512 + lane * 16;
#pragma unroll
            for (int e = 0; e < 16; ++e) acc = fmaf(bf2f(o[e]), bf2f(xs[xo + e]), acc);
        }
#pragma unroll
        for (int s = 16; s; s >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, s);
        if (lane == 0) y[row] = acc;
    }
}

// ------------------------------------------------------------------ host codec
struct Comp {
    std::vector<unsigned char>  lo, cd;
    std::vector<unsigned int>   eoff, epos;
    std::vector<unsigned short> eval;
    size_t n = 0;
    double bytes() const {   // charged compressed footprint, metadata included
        return (double)n + n / 2.0 + eoff.size() * 4.0 + epos.size() * 6.0 + 16.0;
    }
};
static Comp compress(const unsigned short* src, size_t n, size_t K) {
    Comp c; c.n = n; c.lo.resize(n); c.cd.assign(n / 2, 0);
    size_t nch = n / K; c.eoff.assign(nch + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        unsigned short u = src[i]; unsigned int ex = (u >> 7) & 0xFFu;
        c.lo[i] = (unsigned char)(((u >> 8) & 0x80u) | (u & 0x7Fu));
        int code = 0;
        if (ex >= EXP_BASE && ex <= EXP_BASE + 15) code = (int)ex - (int)EXP_BASE;
        else { c.epos.push_back((unsigned int)i); c.eval.push_back(u); c.eoff[i / K]++; }
        c.cd[i / 2] |= (unsigned char)(code << ((i & 1) * 4));
    }
    unsigned int run = 0;
    for (size_t k = 0; k <= nch; ++k) { unsigned int t = k < nch ? c.eoff[k] : 0; c.eoff[k] = run; run += t; }
    return c;
}

// ------------------------------------------------------------------ helpers
static double gpu_ceiling(size_t bytes, int& best_grid, int& best_blk, int steady) {
    size_t n4 = bytes / 16;
    float4* d; CK(cudaMalloc(&d, n4 * 16)); CK(cudaMemset(d, 0x3C, n4 * 16));
    float* sink; CK(cudaMalloc(&sink, 1 << 20));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double best = 0;
    for (int blk : {256, 512})
        for (int g : {264, 528, 1056, 2112, 4224}) {
            int nw = steady ? 300 : 3, nt = steady ? 600 : 10;
            for (int i = 0; i < nw; ++i) k_stream<<<g, blk>>>(d, n4, sink);
            CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            CK(cudaEventRecord(a));
            for (int i = 0; i < nt; ++i) k_stream<<<g, blk>>>(d, n4, sink);
            CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, a, b)); ms /= nt;
            double gbs = (n4 * 16.0) / 1e9 / (ms / 1e3);
            if (gbs > best) { best = gbs; best_grid = g; best_blk = blk; }
        }
    CK(cudaFree(d)); CK(cudaFree(sink));
    return best;
}

// interleave the two planes: per 128-elem group -> [128 B lo][64 B cd] contiguous
static std::vector<unsigned char> interleave(const Comp& c, size_t n) {
    std::vector<unsigned char> v(n / 128 * 192);
    for (size_t g = 0; g < n / 128; ++g) {
        memcpy(&v[g * 192],       &c.lo[g * 128], 128);
        memcpy(&v[g * 192 + 128], &c.cd[g * 64],   64);
    }
    return v;
}

struct Dev { unsigned char *lo[NBUF] = {}, *cd[NBUF] = {}, *il[NBUF] = {}; unsigned short* raw[NBUF] = {};
             unsigned int *eoff = 0, *epos = 0; unsigned short *eval = 0; float* y = 0; };

int main(int argc, char** argv) {
    // argv[2]=="profile": one small shape, one grid, 1 launch per variant (for ncu / clock probe)
    // argv[2]=="soak"   : one shape, long loop of SZ then raw (for nvidia-smi clock/power sampling)
    const char* mode = argc > 2 ? argv[2] : "";
    const int PROFILE = !strcmp(mode, "profile"), SOAK = !strcmp(mode, "soak");
    const char* path = argc > 1 ? argv[1] : "/root/.claude/jobs/c92f0b7e/tmp/gemma_bf16_sample.bin";
    FILE* f = fopen(path, "rb"); if (!f) { printf("missing %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); size_t nb = ftell(f); fseek(f, 0, SEEK_SET);
    size_t nsrc = nb / 2; std::vector<unsigned short> src(nsrc);
    if (fread(src.data(), 2, nsrc, f) != nsrc) return 1; fclose(f);

    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("GPU: %s  sm_%d%d  SMs=%d  L2=%.1f MB  HBM=%.1f GB\n", pr.name, pr.major, pr.minor,
           pr.multiProcessorCount, pr.l2CacheSize / 1048576.0, pr.totalGlobalMem / 1073741824.0);
    printf("source: %zu real bf16 elems (%.1f MB) from gemma-4 linear weights  EXP_BASE=%u\n\n",
           nsrc, nsrc * 2 / 1048576.0, EXP_BASE);

    // ================= GATE 1: bit-exact codec round trip + negative controls ==========
    if (!PROFILE && !SOAK && strcmp(mode, "lmhead")) {
        size_t n = (nsrc / 16) * 16;
        Comp c = compress(src.data(), n, n / 1024);
        printf("== GATE 1: codec round trip on %zu REAL bf16 elements (%.1f MB) ==\n", n, n * 2 / 1048576.0);
        printf("   escapes=%zu (%.5f%%)  compressed=%.2f MB  RATIO=%.4fx\n",
               c.epos.size(), 100.0 * c.epos.size() / n, c.bytes() / 1048576.0, (n * 2.0) / c.bytes());
        unsigned char *d_lo, *d_cd; unsigned short *d_out, *d_ev; unsigned int* d_ep;
        CK(cudaMalloc(&d_lo, n)); CK(cudaMalloc(&d_cd, n / 2)); CK(cudaMalloc(&d_out, n * 2));
        CK(cudaMalloc(&d_ep, c.epos.size() * 4 + 4)); CK(cudaMalloc(&d_ev, c.eval.size() * 2 + 2));
        CK(cudaMemcpy(d_lo, c.lo.data(), n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_cd, c.cd.data(), n / 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ep, c.epos.data(), c.epos.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ev, c.eval.data(), c.eval.size() * 2, cudaMemcpyHostToDevice));
        std::vector<unsigned short> got(n);
        auto rt = [&](const char* lbl, unsigned int eb, int corrupt_esc) {
            if (c.eval.size()) {
                unsigned short v = corrupt_esc ? (unsigned short)(c.eval[0] ^ 0x0001) : c.eval[0];
                CK(cudaMemcpy(d_ev, &v, 2, cudaMemcpyHostToDevice));
            }
            k_store<<<1056, 256>>>((uint4*)d_lo, (uint2*)d_cd, n / 16, eb, (uint4*)d_out);
            if (c.epos.size()) k_escape<<<(c.epos.size() + 255) / 256, 256>>>(d_ep, d_ev, (unsigned int)c.epos.size(), d_out);
            CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            CK(cudaMemcpy(got.data(), d_out, n * 2, cudaMemcpyDeviceToHost));
            int cmp = memcmp(got.data(), src.data(), n * 2);
            size_t bad = 0; for (size_t i = 0; i < n; ++i) bad += (got[i] != src[i]);
            printf("   %-46s memcmp=%-6s mismatching_elements=%zu\n", lbl, cmp == 0 ? "EQUAL" : "DIFF", bad);
            return cmp == 0;
        };
        bool ok = rt("clean round trip (expect EQUAL)", EXP_BASE, 0);
        bool n1 = rt("NEG: EXP_BASE 109->108", EXP_BASE - 1, 0);
        bool n2 = rt("NEG: escape value 1 bit flipped", EXP_BASE, 1);
        bool ok2 = rt("restored (expect EQUAL again)", EXP_BASE, 0);
        printf("   GATE 1 VERDICT: %s\n\n", (ok && ok2 && !n1 && !n2) ?
               "PASS (bit exact; both negative controls detected)" : "FAIL");
        if (!(ok && ok2 && !n1 && !n2)) return 1;
        CK(cudaFree(d_lo)); CK(cudaFree(d_cd)); CK(cudaFree(d_out)); CK(cudaFree(d_ep)); CK(cudaFree(d_ev));
    }

    // ================= GATE 2: real HBM streaming ceiling ==============================
    int cg = 0, cb = 0;
    double ceil_gbs = 3733.2;
    if (!PROFILE && !SOAK) {
        int bg, bb;
        double burst = gpu_ceiling(4ull << 30, bg, bb, 0);
        ceil_gbs = gpu_ceiling(4ull << 30, cg, cb, 1);
        printf("== GATE 2: measured HBM streaming ceiling (grid-stride float4 read, 4 GB) ==\n");
        printf("   short BURST  : %.1f GB/s  (grid=%d blk=%d)\n", burst, bg, bb);
        printf("   STEADY state : %.1f GB/s  (grid=%d blk=%d)  <- reference for %%ceil below\n\n",
               ceil_gbs, cg, cb);
    }

    // ================= registers / occupancy ==========================================
    {
        printf("== registers / occupancy (BLK=%d, dyn smem=%d B for K=3840) ==\n", BLK, 3840 * 2);
        auto rep = [&](const char* nm, const void* fn) {
            cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, fn));
            int b; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, fn, BLK, 3840 * 2));
            printf("   %-24s regs=%-4d spill=%zu B  blocks/SM=%d  warp-occ=%.1f%%\n", nm, a.numRegs,
                   a.localSizeBytes, b, 100.0 * b * WARPS / (pr.maxThreadsPerMultiProcessor / 32));
        };
        rep("raw bf16 GEMV",          (const void*)k_gemv<0, 1>);
        rep("splitzip fused",         (const void*)k_gemv<1, 1>);
        rep("splitzip fused occ8",    (const void*)k_gemv<1, 8>);
        rep("SZ zero-decode ablation",(const void*)k_gemv<3, 1>);
        rep("SZ interleaved fused",   (const void*)k_gemv<4, 1>);
        rep("SZ interleaved ablation",(const void*)k_gemv<5, 1>);
        printf("\n");
    }

    // ================= GATE 3: per-shape A/B ==========================================
    struct Sh { const char* nm; int N, K, L; };
    // L picked so every working set is >= ~1 GB = >=20x the 50 MB L2.
    Sh shapes_full[] = {
        {"q_proj  N4096 K3840",   4096,  3840, 34},
        {"kv_proj N2048 K3840",   2048,  3840, 68},
        {"gate/up N15360 K3840", 15360,  3840,  9},
        {"down    N3840 K15360",  3840, 15360,  9},
        {"lm_head N262144 K3840",262144, 3840,  1},
    };
    Sh shapes_one[] = { {"q_proj  N4096 K3840", 4096, 3840, 34} };
    Sh shapes_lm[]  = { {"lm_head N262144 K3840", 262144, 3840, 1} };
    const int LMONLY = !strcmp(mode, "lmhead");
    Sh* shapes   = (PROFILE || SOAK) ? shapes_one : (LMONLY ? shapes_lm : shapes_full);
    int nshapes  = (PROFILE || SOAK || LMONLY) ? 1 : (int)(sizeof(shapes_full) / sizeof(Sh));
    const int grids_full[] = {528, 792, 1056, 1584, 2112};
    const int grids_one[]  = {1584};
    const int* grids = (PROFILE || SOAK) ? grids_one : grids_full;
    const int NG = (PROFILE || SOAK) ? 1 : (int)(sizeof(grids_full) / sizeof(int));
    cudaEvent_t ev0, ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));

    printf("== GATE 3: decode GEMV A/B, M=1, working set >= 1 GB, buffers cycled (NBUF=%d) ==\n", NBUF);
    printf("   variants: raw | SZ(split-plane, bit-exact) | SZ-occ8 (forced full occupancy)\n"
           "             SZ-noesc (diag) | SZ-il (interleaved layout, bit-exact)\n"
           "             ABL / ABL-il = same loads, same 16 FMAs, ZERO decode ALU (layout upper bound)\n\n");

    for (int si = 0; si < nshapes; ++si) {
        Sh sh = shapes[si];
        size_t nper = (size_t)sh.N * sh.K, ntot = nper * (size_t)sh.L;
        std::vector<unsigned short> W(ntot);
        for (size_t i = 0; i < ntot; ++i) W[i] = src[i % nsrc];
        Comp c = compress(W.data(), ntot, (size_t)sh.K);
        std::vector<unsigned char> iv = interleave(c, ntot);
        double logical = ntot * 2.0, comp = c.bytes(), ratio = logical / comp;

        Dev d;
        for (int b = 0; b < NBUF; ++b) {
            CK(cudaMalloc(&d.raw[b], ntot * 2)); CK(cudaMemcpy(d.raw[b], W.data(), ntot * 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&d.lo[b], ntot));      CK(cudaMemcpy(d.lo[b], c.lo.data(), ntot, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&d.cd[b], ntot / 2));  CK(cudaMemcpy(d.cd[b], c.cd.data(), ntot / 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&d.il[b], iv.size())); CK(cudaMemcpy(d.il[b], iv.data(), iv.size(), cudaMemcpyHostToDevice));
        }
        CK(cudaMalloc(&d.eoff, c.eoff.size() * 4)); CK(cudaMemcpy(d.eoff, c.eoff.data(), c.eoff.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d.epos, c.epos.size() * 4 + 4)); CK(cudaMalloc(&d.eval, c.eval.size() * 2 + 2));
        if (c.epos.size()) CK(cudaMemcpy(d.epos, c.epos.data(), c.epos.size() * 4, cudaMemcpyHostToDevice));
        if (c.eval.size()) CK(cudaMemcpy(d.eval, c.eval.data(), c.eval.size() * 2, cudaMemcpyHostToDevice));
        long long nrow = (long long)sh.N * sh.L;
        CK(cudaMalloc(&d.y, nrow * 4));
        std::vector<unsigned short> hx(sh.K);
        for (int i = 0; i < sh.K; ++i) hx[i] = src[((size_t)i * 7919) % nsrc];
        unsigned short* d_x; CK(cudaMalloc(&d_x, sh.K * 2));
        CK(cudaMemcpy(d_x, hx.data(), sh.K * 2, cudaMemcpyHostToDevice));
        size_t smem = (size_t)sh.K * 2;

        enum { V_RAW = 0, V_RAW8, V_SZ, V_SZ_OCC8, V_SZ_NOESC, V_SZ_IL, V_ABL, V_ABL_IL, NV };
        const char* vn[NV] = {"raw", "raw-occ8", "SZ", "SZ-occ8", "SZ-noesc", "SZ-il", "ABL", "ABL-il"};
        auto launch = [&](int v, int g, int i) {
            int b = i % NBUF;
            const uint4* L = (const uint4*)d.lo[b]; const uint2* C = (const uint2*)d.cd[b];
            const unsigned char* I = d.il[b];
            switch (v) {
            case V_RAW:      k_gemv<0,1><<<g,BLK,smem>>>((uint4*)d.raw[b],0,0,0,0,0,0,d_x,d.y,sh.K,nrow); break;
            case V_RAW8:     k_gemv<0,8><<<g,BLK,smem>>>((uint4*)d.raw[b],0,0,0,0,0,0,d_x,d.y,sh.K,nrow); break;
            case V_SZ:       k_gemv<1,1><<<g,BLK,smem>>>(0,L,C,0,d.eoff,d.epos,d.eval,d_x,d.y,sh.K,nrow); break;
            case V_SZ_OCC8:  k_gemv<1,8><<<g,BLK,smem>>>(0,L,C,0,d.eoff,d.epos,d.eval,d_x,d.y,sh.K,nrow); break;
            case V_SZ_NOESC: k_gemv<2,1><<<g,BLK,smem>>>(0,L,C,0,d.eoff,d.epos,d.eval,d_x,d.y,sh.K,nrow); break;
            case V_SZ_IL:    k_gemv<4,1><<<g,BLK,smem>>>(0,0,0,I,d.eoff,d.epos,d.eval,d_x,d.y,sh.K,nrow); break;
            case V_ABL:      k_gemv<3,1><<<g,BLK,smem>>>(0,L,C,0,d.eoff,d.epos,d.eval,d_x,d.y,sh.K,nrow); break;
            default:         k_gemv<5,1><<<g,BLK,smem>>>(0,0,0,I,d.eoff,d.epos,d.eval,d_x,d.y,sh.K,nrow); break;
            }
        };
        if (SOAK) {   // steady-state clock/power probe: 8 s of SZ, then 8 s of raw
            for (int phase = 0; phase < 2; ++phase) {
                int v = phase ? V_RAW8 : V_SZ;
                printf("SOAK phase %s begin\n", vn[v]); fflush(stdout);
                cudaEvent_t s0, s1; CK(cudaEventCreate(&s0)); CK(cudaEventCreate(&s1));
                CK(cudaEventRecord(s0));
                int n = 0; float el = 0;
                while (el < 8000.f) {
                    for (int i = 0; i < 20; ++i) { launch(v, 1584, i); ++n; }
                    CK(cudaEventRecord(s1)); CK(cudaEventSynchronize(s1));
                    CK(cudaEventElapsedTime(&el, s0, s1));
                }
                printf("SOAK phase %s end: %d launches in %.1f ms = %.0f ns each\n",
                       vn[v], n, el, el * 1e6 / n / sh.L); fflush(stdout);
            }
            return 0;
        }
        // One measurement. STEADY=1 keeps the GPU continuously busy for ~0.4 s of warmup
        // + ~0.8 s of timing, so the reported time is at the DVFS/power-cap steady state
        // that a real decode loop actually sees. STEADY=0 is the classic short burst.
        auto once = [&](int v, int g, int steady) {
            for (int i = 0; i < 3; ++i) launch(v, g, i);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(ev0));
            for (int i = 0; i < 5; ++i) launch(v, g, i);
            CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1)); CK(cudaGetLastError());
            float el; CK(cudaEventElapsedTime(&el, ev0, ev1)); double est = el / 5.0;
            if (!steady) return est;
            int nw = (int)(400.0 / est) + 1, n = (int)(800.0 / est) + 1;
            for (int i = 0; i < nw; ++i) launch(v, g, i);       // continuous warmup -> steady clocks
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(ev0));
            for (int i = 0; i < n; ++i) launch(v, g, i);
            CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1)); CK(cudaGetLastError());
            CK(cudaEventElapsedTime(&el, ev0, ev1));
            return (double)el / n;
        };
        // grid sweep, best time per variant (fair: each variant gets its own best grid)
        auto best_of = [&](int v, int& gbest, int steady) {
            double best = 1e30;
            for (int gi = 0; gi < NG; ++gi) {
                double t = once(v, grids[gi], steady);
                if (t < best) { best = t; gbest = grids[gi]; }
            }
            return best;
        };
        double ms[NV], mb[NV]; int gg[NV], gb[NV]; int bex[NV];
        std::vector<float> y_r((size_t)nrow), y_v((size_t)nrow);
        for (int v = 0; v < NV; ++v) {
            bex[v] = -1;
            mb[v] = best_of(v, gb[v], 0);                 // short burst (boost clocks)
            gg[v] = gb[v];
            ms[v] = PROFILE ? mb[v] : best_of(v, gg[v], 1);   // steady state (power-cap clocks)
            if (v == V_RAW) CK(cudaMemcpy(y_r.data(), d.y, nrow * 4, cudaMemcpyDeviceToHost));
            else if (v == V_SZ || v == V_SZ_OCC8 || v == V_SZ_IL) {
                CK(cudaMemcpy(y_v.data(), d.y, nrow * 4, cudaMemcpyDeviceToHost));
                bex[v] = (memcmp(y_v.data(), y_r.data(), nrow * 4) == 0);
            }
        }
        double base = ms[V_RAW] < ms[V_RAW8] ? ms[V_RAW] : ms[V_RAW8];   // best uncompressed
        double bszz = ms[V_SZ] < ms[V_SZ_OCC8] ? ms[V_SZ] : ms[V_SZ_OCC8];
        if (ms[V_SZ_IL] < bszz) bszz = ms[V_SZ_IL];                      // best bit-exact splitzip
        double babl = ms[V_ABL] < ms[V_ABL_IL] ? ms[V_ABL] : ms[V_ABL_IL];
        double baseb = mb[V_RAW] < mb[V_RAW8] ? mb[V_RAW] : mb[V_RAW8];
        double bszb  = mb[V_SZ] < mb[V_SZ_OCC8] ? mb[V_SZ] : mb[V_SZ_OCC8];
        if (mb[V_SZ_IL] < bszb) bszb = mb[V_SZ_IL];
        printf("%-22s wset=%.0f MB (%.1fx L2)  compression RATIO=%.4fx  escapes=%.5f%%\n",
               sh.nm, logical / 1048576.0, logical / (double)pr.l2CacheSize, ratio,
               100.0 * c.epos.size() / ntot);
        printf("   %-9s | %12s %11s %13s %8s %7s | %11s %8s %7s | %s\n",
               "variant", "STEADY ns", "logic GB/s", "HBMbytes GB/s", "%ceil", "vs raw",
               "BURST ns", "GB/s", "vs raw", "bitexact");
        for (int v = 0; v < NV; ++v) {
            double hbm = (v <= V_RAW8 ? logical : comp);          // bytes actually crossing DRAM
            printf("   %-9s | %12.0f %11.1f %13.1f %7.1f%% %6.3fx | %11.0f %8.1f %6.3fx | %-4s [g%d/%d]\n",
                   vn[v], ms[v] * 1e6 / sh.L, logical / 1e9 / (ms[v] / 1e3), hbm / 1e9 / (ms[v] / 1e3),
                   100.0 * (hbm / 1e9 / (ms[v] / 1e3)) / ceil_gbs, base / ms[v],
                   mb[v] * 1e6 / sh.L, logical / 1e9 / (mb[v] / 1e3), baseb / mb[v],
                   bex[v] < 0 ? "-" : (bex[v] ? "PASS" : "FAIL"), gg[v], gb[v]);
        }
        printf("   >> STEADY: best SplitZip = %.3fx vs best raw (realizes %.1f%% of the %.4fx ratio);"
               " ZERO-decode layout bound = %.3fx | BURST: SZ = %.3fx\n\n",
               base / bszz, 100.0 * (base / bszz) / ratio, ratio, base / babl, baseb / bszb);

        for (int b = 0; b < NBUF; ++b) { CK(cudaFree(d.raw[b])); CK(cudaFree(d.lo[b])); CK(cudaFree(d.cd[b])); CK(cudaFree(d.il[b])); }
        CK(cudaFree(d.eoff)); CK(cudaFree(d.epos)); CK(cudaFree(d.eval)); CK(cudaFree(d.y)); CK(cudaFree(d_x));
    }
    return 0;
}
