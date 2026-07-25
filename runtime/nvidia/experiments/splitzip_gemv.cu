// Layer 4b step 3: SplitZip decode FUSED into the decode GEMV's HBM->SRAM path.
//
// The compressed form is the ONLY thing that crosses HBM. Decode happens
// per-tile on the way into registers/SRAM; the logical bf16 weights are never
// materialized in global memory (that is the paper's mode, refuted here at
// 849 GB/s by /workspace/splitzip/sz.cu).
//
// Layout (SoA, exactly what plow's packet compiler would emit):
//   lo[]   : 1 B / elem   = sign(bit7) | mantissa(bits6:0)
//   cd[]   : 4 b / elem   = code, exponent = code + 109   (measured global run 109..124)
//   eoff[] : u32 per 4096-elem escape chunk (prefix offsets into epos/eval)
//   epos[] : u32 flat element index      eval[] : u16 raw bf16
// Reconstruct: bf16 = ((lo&0x80)<<8) | (exp<<7) | (lo&0x7F), then sparse overwrite.
//
// MODE 0 = raw bf16 GEMV (reference). MODE 1 = fused SplitZip GEMV.
// Both share one loop body, identical FMA order => output must be BIT-IDENTICAL.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

#define WARPS 8
#define BLK   (WARPS*32)
#define ESC_CHUNK_LOG 12            // 4096 elements per escape chunk
#define ESC_CHUNK (1<<ESC_CHUNK_LOG)

__device__ __forceinline__ float bf2f(unsigned short u) {
    return __uint_as_float((unsigned int)u << 16);
}

// EXP_BASE is a compile-time constant in the packet (the code table degenerates
// to an affine map for Qwen3-4B: exps 109..124 contiguous, verified on 21 tensors)
#define EXP_BASE 109u

template<int MODE>
__global__ __launch_bounds__(BLK) void k_gemv(
        const uint4*         __restrict__ Wraw,
        const uint4*         __restrict__ lo,
        const uint2*         __restrict__ cd,
        const unsigned int*  __restrict__ eoff,
        const unsigned int*  __restrict__ epos,
        const unsigned short*__restrict__ eval,
        const unsigned short*__restrict__ x,
        float*               __restrict__ y,
        int K, long long nrow_total)
{
    extern __shared__ unsigned short xs[];          // K bf16 activations, SRAM-resident
    for (int i = threadIdx.x; i < K; i += BLK) xs[i] = x[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int w    = threadIdx.x >> 5;
    const int nj   = K >> 9;                        // 512 elems per warp-iteration
    long long stride = (long long)gridDim.x * WARPS;

    for (long long row = (long long)blockIdx.x * WARPS + w; row < nrow_total; row += stride) {
        long long rbase = row * (long long)K;
        float acc = 0.f;
        // ONE escape-list lookup per row, hoisted out of the inner loop; its
        // latency overlaps the first weight load. Escapes are chunked per ROW.
        unsigned int e0 = 0, e1 = 0;
        if (MODE == 1) { e0 = eoff[row]; e1 = eoff[row + 1]; }
        for (int j = 0; j < nj; ++j) {
            long long base = rbase + (long long)j * 512;
            long long myel = base + lane * 16;
            unsigned short o[16];
            if (MODE == 0) {
                const uint4* p = Wraw + (myel >> 3);
                uint4 a = p[0], b = p[1];
                unsigned int aw[8] = {a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
                for (int e = 0; e < 16; ++e)
                    o[e] = (unsigned short)((aw[e >> 1] >> ((e & 1) * 16)) & 0xFFFFu);
            } else {  // MODE 1 (bit-exact) or MODE 2 (no-escape diagnostic)
                uint4 l = lo[myel >> 4];
                uint2 c = cd[myel >> 4];
                unsigned int lw[4] = {l.x,l.y,l.z,l.w}, cw[2] = {c.x,c.y};
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    unsigned int b  = (lw[e >> 2] >> ((e & 3) * 8)) & 0xFFu;
                    unsigned int cc = (cw[e >> 3] >> ((e & 7) * 4)) & 0xFu;
                    unsigned int ex = cc + EXP_BASE;
                    o[e] = (unsigned short)(((b & 0x80u) << 8) | (ex << 7) | (b & 0x7Fu));
                }
                // sparse escape overwrite, warp-uniform loop bounds (no divergence
                // in the trip count); body taken ~0.6 times per 4096 elements.
                for (unsigned int t = e0; t < (MODE == 1 ? e1 : e0); ++t) {
                    unsigned int p = epos[t];
                    if ((unsigned int)(p - (unsigned int)myel) < 16u) {
                        unsigned short v = eval[t];
                        unsigned int sl = p & 15u;
#pragma unroll
                        for (int e = 0; e < 16; ++e) if ((unsigned int)e == sl) o[e] = v;
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

// ---------------- host ----------------
struct Comp {
    std::vector<unsigned char>  lo, cd;
    std::vector<unsigned int>   eoff, epos;
    std::vector<unsigned short> eval;
    size_t n;
    double bytes() const {                       // charged compressed footprint
        return (double)n + n / 2.0 + eoff.size() * 4.0 + epos.size() * 6.0 + 16.0;
    }
};

static Comp compress(const unsigned short* src, size_t n, size_t K) {
    Comp c; c.n = n;
    c.lo.resize(n); c.cd.assign(n / 2, 0);
    size_t nch = n / K;                          // one escape chunk per output ROW
    c.eoff.resize(nch + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        unsigned short u = src[i];
        unsigned int ex = (u >> 7) & 0xFFu;
        c.lo[i] = (unsigned char)(((u >> 8) & 0x80u) | (u & 0x7Fu));
        int code = 0;
        if (ex >= EXP_BASE && ex <= EXP_BASE + 15) code = (int)ex - (int)EXP_BASE;
        else { c.epos.push_back((unsigned int)i); c.eval.push_back(u); c.eoff[i / K]++; }
        c.cd[i / 2] |= (unsigned char)(code << ((i & 1) * 4));
    }
    unsigned int run = 0;                        // counts -> prefix offsets
    for (size_t k = 0; k <= nch; ++k) { unsigned int t = k < nch ? c.eoff[k] : 0; c.eoff[k] = run; run += t; }
    return c;
}

struct Dev {
    unsigned char *lo = 0, *cd = 0; unsigned int *eoff = 0, *epos = 0;
    unsigned short *eval = 0, *raw = 0; float* y = 0;
};

int main(int argc, char** argv) {
    int do_bitexact = argc > 1 && !strcmp(argv[1], "bitexact");

    // ---- real Qwen3-4B weights: layers.17.mlp.gate_proj [9728,2560] ----
    FILE* f = fopen("/workspace/splitzip/qwen_gate17.bin", "rb");
    if (!f) { printf("missing qwen_gate17.bin\n"); return 1; }
    fseek(f, 0, SEEK_END); size_t nb = ftell(f); fseek(f, 0, SEEK_SET);
    size_t nsrc = nb / 2;
    std::vector<unsigned short> src(nsrc);
    if (fread(src.data(), 2, nsrc, f) != nsrc) return 1;
    fclose(f);
    printf("source weights: %zu real bf16 elements (%.1f MB) from Qwen3-4B layers.17.mlp.gate_proj\n",
           nsrc, nsrc * 2 / 1048576.0);

    // Every 2-D decode-resident shape in Qwen3-4B. COPIES = how many tensors of
    // this shape the real model reads per decoded token (36 layers).
    struct Shape { const char* name; int N, K; int L; int copies; };
    Shape shapes[] = {
        {"q_proj  [  4096,2560]",   4096, 2560,  32, 36},
        {"k/v_proj[  1024,2560]",   1024, 2560, 128, 72},
        {"o_proj  [  2560,4096]",   2560, 4096,  32, 36},
        {"gate/up [  9728,2560]",   9728, 2560,  14, 72},
        {"down    [  2560,9728]",   2560, 9728,  14, 36},
        {"lm_head [151936,2560]", 151936, 2560,   1,  1},
    };
    const int nsh = 6;
    double tok_ms_raw = 0, tok_ms_sz = 0, tok_bytes_raw = 0, tok_bytes_sz = 0;

    cudaEvent_t ev0, ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    printf("\n%-22s %5s %10s %9s %8s %10s %11s %11s %9s\n", "shape", "cop", "workset MB",
           "ratio", "ms", "comp GB/s", "LOGICAL GB/s", "vs raw", "bitexact");

    double sum_log_gbs_w = 0, sum_w = 0;
    for (int s = 0; s < nsh; ++s) {
        Shape sh = shapes[s];
        size_t nper = (size_t)sh.N * sh.K;
        size_t ntot = nper * sh.L;
        // tile the real tensor to fill the working set (real exponent statistics)
        std::vector<unsigned short> W(ntot);
        for (size_t i = 0; i < ntot; ++i) W[i] = src[i % nsrc];
        Comp c = compress(W.data(), ntot, (size_t)sh.K);
        double logical = ntot * 2.0, comp = c.bytes();
        double ratio = logical / comp;

        Dev d;
        CK(cudaMalloc(&d.lo, ntot)); CK(cudaMalloc(&d.cd, ntot / 2));
        CK(cudaMalloc(&d.eoff, c.eoff.size() * 4));
        CK(cudaMalloc(&d.epos, c.epos.size() * 4 + 4));
        CK(cudaMalloc(&d.eval, c.eval.size() * 2 + 2));
        CK(cudaMalloc(&d.y, (size_t)sh.N * sh.L * 4 * 2));
        CK(cudaMemcpy(d.lo, c.lo.data(), ntot, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.cd, c.cd.data(), ntot / 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.eoff, c.eoff.data(), c.eoff.size() * 4, cudaMemcpyHostToDevice));
        if (c.epos.size()) CK(cudaMemcpy(d.epos, c.epos.data(), c.epos.size() * 4, cudaMemcpyHostToDevice));
        if (c.eval.size()) CK(cudaMemcpy(d.eval, c.eval.data(), c.eval.size() * 2, cudaMemcpyHostToDevice));

        // activation vector x (bf16), SRAM resident
        std::vector<unsigned short> hx(sh.K);
        for (int i = 0; i < sh.K; ++i) hx[i] = src[(i * 7919) % nsrc];
        unsigned short* d_x; CK(cudaMalloc(&d_x, sh.K * 2));
        CK(cudaMemcpy(d_x, hx.data(), sh.K * 2, cudaMemcpyHostToDevice));

        size_t smem = (size_t)sh.K * 2;
        long long nrow = (long long)sh.N * sh.L;
        int grid = 1020;

        auto timeit = [&](int mode) {
            for (int wm = 0; wm < 2; ++wm) {
                if (wm) CK(cudaEventRecord(ev0));
                int it = wm ? 10 : 2;
                for (int i = 0; i < it; ++i) {
                    if (mode == 0) k_gemv<0><<<grid, BLK, smem>>>((uint4*)d.raw, 0, 0, 0, 0, 0, d_x, d.y, sh.K, nrow);
                    else if (mode == 1) k_gemv<1><<<grid, BLK, smem>>>(0, (uint4*)d.lo, (uint2*)d.cd, d.eoff, d.epos, d.eval, d_x, d.y, sh.K, nrow);
                    else k_gemv<2><<<grid, BLK, smem>>>(0, (uint4*)d.lo, (uint2*)d.cd, d.eoff, d.epos, d.eval, d_x, d.y, sh.K, nrow);
                }
                if (wm) { CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1)); }
                CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            }
            float ms; CK(cudaEventElapsedTime(&ms, ev0, ev1)); return (double)ms / 10.0;
        };

        // ---- fused (compressed) ----
        double ms_c = timeit(1);
        std::vector<float> y_c((size_t)nrow);
        CK(cudaMemcpy(y_c.data(), d.y, nrow * 4, cudaMemcpyDeviceToHost));

        // ---- raw bf16 reference on the SAME working set ----
        CK(cudaMalloc(&d.raw, ntot * 2));
        CK(cudaMemcpy(d.raw, W.data(), ntot * 2, cudaMemcpyHostToDevice));
        double ms_r = timeit(0);
        std::vector<float> y_r((size_t)nrow);
        CK(cudaMemcpy(y_r.data(), d.y, nrow * 4, cudaMemcpyDeviceToHost));

        int be = memcmp(y_c.data(), y_r.data(), nrow * 4) == 0;
        size_t bad = 0; for (long long i = 0; i < nrow; ++i) bad += (y_c[i] != y_r[i]);

        printf("%-22s %5s %10.1f %9.4f %8.3f %10.1f %11.1f %11s %9s\n", sh.name, "raw",
               logical / 1048576.0, 1.0, ms_r, logical / 1e9 / (ms_r / 1e3),
               logical / 1e9 / (ms_r / 1e3), "1.000x", "-");
        printf("%-22s %5s %10.1f %9.4f %8.3f %10.1f %11.1f %10.3fx %9s (mismatch=%zu)\n",
               sh.name, "SZ", comp / 1048576.0, ratio, ms_c, comp / 1e9 / (ms_c / 1e3),
               logical / 1e9 / (ms_c / 1e3), ms_r / ms_c, be ? "PASS" : "FAIL", bad);

        double ms_d = timeit(2);   // diagnostic: fused decode, escape pass removed
        printf("%-22s %5s %10.1f %9.4f %8.3f %10.1f %11.1f %10.3fx %9s\n", sh.name, "SZ-ne",
               (comp - c.eoff.size() * 4.0 - c.epos.size() * 6.0) / 1048576.0, ratio, ms_d,
               (comp - c.eoff.size() * 4.0 - c.epos.size() * 6.0) / 1e9 / (ms_d / 1e3),
               logical / 1e9 / (ms_d / 1e3), ms_r / ms_d, "DIAG");

        // scale the measured time of this working set to the real model's copies
        double scale = (double)sh.copies / (double)sh.L;
        tok_ms_raw += ms_r * scale;  tok_ms_sz  += ms_c * scale;
        tok_bytes_raw += logical * scale; tok_bytes_sz += comp * scale;

        sum_log_gbs_w += (logical / 1e9 / (ms_c / 1e3)) * nper; sum_w += nper;

        if (do_bitexact && s == 0) {
            // ---- negative controls: prove the memcmp can fail ----
            printf("\n-- negative controls on %s --\n", sh.name);
            auto probe = [&](const char* label) {
                k_gemv<1><<<grid, BLK, smem>>>(0, (uint4*)d.lo, (uint2*)d.cd, d.eoff, d.epos, d.eval, d_x, d.y, sh.K, nrow);
                CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
                std::vector<float> yy((size_t)nrow);
                CK(cudaMemcpy(yy.data(), d.y, nrow * 4, cudaMemcpyDeviceToHost));
                size_t b = 0; for (long long i = 0; i < nrow; ++i) b += (yy[i] != y_r[i]);
                printf("  %-44s memcmp=%-5s mismatching_outputs=%zu\n", label,
                       memcmp(yy.data(), y_r.data(), nrow * 4) == 0 ? "EQUAL" : "DIFF", b);
            };
            probe("clean fused GEMV (expect EQUAL)");
            { unsigned char b = c.lo[0] ^ 0x01;                  // corrupt one payload byte
              CK(cudaMemcpy(d.lo, &b, 1, cudaMemcpyHostToDevice)); probe("NEG: 1 payload (sign+mantissa) bit flipped");
              CK(cudaMemcpy(d.lo, c.lo.data(), 1, cudaMemcpyHostToDevice)); }
            { unsigned char b = c.cd[0] ^ 0x01;                  // corrupt one 4-bit code
              CK(cudaMemcpy(d.cd, &b, 1, cudaMemcpyHostToDevice)); probe("NEG: 1 four-bit exponent code corrupted");
              CK(cudaMemcpy(d.cd, c.cd.data(), 1, cudaMemcpyHostToDevice)); }
            if (c.eval.size()) {                                 // corrupt an escape value
              unsigned short v = c.eval[0] ^ 0x4000;
              CK(cudaMemcpy(d.eval, &v, 2, cudaMemcpyHostToDevice)); probe("NEG: escape VALUE exponent bit flipped");
              CK(cudaMemcpy(d.eval, c.eval.data(), 2, cudaMemcpyHostToDevice)); }
            if (c.eoff.size() > 1) {                             // drop all escapes in chunk 0
              unsigned int z = c.eoff[1];
              CK(cudaMemcpy(d.eoff, &z, 4, cudaMemcpyHostToDevice)); probe("NEG: escape list for chunk 0 dropped");
              CK(cudaMemcpy(d.eoff, c.eoff.data(), 4, cudaMemcpyHostToDevice)); }
            probe("restored (expect EQUAL again)");
            printf("  escapes total=%zu (%.5f%%), eoff entries=%zu\n\n",
                   c.epos.size(), 100.0 * c.epos.size() / ntot, c.eoff.size());
        }

        CK(cudaFree(d.lo)); CK(cudaFree(d.cd)); CK(cudaFree(d.eoff)); CK(cudaFree(d.epos));
        CK(cudaFree(d.eval)); CK(cudaFree(d.raw)); CK(cudaFree(d.y)); CK(cudaFree(d_x));
    }
    printf("\nparam-weighted mean LOGICAL GB/s (fused) = %.1f\n", sum_log_gbs_w / sum_w);
    printf("\n==== Qwen3-4B batch-1 DECODE, one token, all 252 GEMM tensors + lm_head ====\n");
    printf("  raw bf16 : %8.4f GB resident   %7.3f ms/token\n", tok_bytes_raw / 1e9, tok_ms_raw);
    printf("  SplitZip : %8.4f GB resident   %7.3f ms/token   ratio %.4fx  SPEEDUP %.3fx\n",
           tok_bytes_sz / 1e9, tok_ms_sz, tok_bytes_raw / tok_bytes_sz, tok_ms_raw / tok_ms_sz);
    printf("  (reference floors: bf16 8.00 GB/1673 GB/s = 4.78 ms, fp8 2.39 ms)\n");
    return 0;
}
