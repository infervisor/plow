/* mla_sparse_pf_bench_gfx950.hip — GLM-5.3 TP8 DSA sparse MLA prefill per rank (8 heads).
 *   v2gather : d_flash_mla_prefill_v2<512,64,GATHER> (shipped arm, partials + merge not timed)
 *   sparse   : d_mla_sparse_pf (mla_sparse_pf.h), normalized bf16 latent out
 * Selection = a dumped iidx_pf ([T][2048] i32, PLOW_DUMP_ACT=act.iidx_pf) or synthetic. The union
 * table is built on the host in IndexUnionPf's layout (pack 8). sparse is checked against a double
 * reference over each sampled query's OWN selection (rel-L2).
 *   ./bench T [iidx.bin]
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "amd_common.h"
#ifdef __HIP_DEVICE_COMPILE__
#include "op_attention.h"
#include "mla_sparse_pf.h"
#else
namespace mla_sp { constexpr unsigned LDS_BYTES = 161792; }
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
constexpr unsigned H = 8, TOPK = 2048, QP = 8;
constexpr unsigned V2_LDS = 65536;

extern "C" __global__ __launch_bounds__(256) void k_v2(float* op, float* ml, const bf16* qa, const bf16* qr,
                                                       const bf16* ck, const bf16* kr, const int* kvl,
                                                       const unsigned char* uni, unsigned T, unsigned cap,
                                                       float scale) {
#ifdef __HIP_DEVICE_COMPILE__
    extern __shared__ bf16 lds[];
    d_flash_mla_prefill_v2<512, 64, true, false>(op, ml, qa, qr, ck, kr, kvl, 1u, T, H, T, 0u, scale,
                                                 0xffffffffu, blockIdx.x, gridDim.x, lds, uni, cap, 1u,
                                                 false, nullptr, 0u);
#endif
}

extern "C" __global__ __launch_bounds__(256) void k_sparse(bf16* o, const bf16* qa, const bf16* qr,
                                                           const bf16* ck, const bf16* kr,
                                                           const unsigned char* uni, unsigned T,
                                                           unsigned cap, float scale) {
#ifdef __HIP_DEVICE_COMPILE__
    extern __shared__ unsigned char ldsb[];
    d_mla_sparse_pf(o, qa, qr, ck, kr, uni, T, cap, scale, blockIdx.x, gridDim.x, ldsb);
#endif
}

static unsigned short f2b(float f) { unsigned u; memcpy(&u, &f, 4); return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float b2f(unsigned short b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    const unsigned T = argc > 1 ? atoi(argv[1]) : 8192;
    std::vector<int> sel((size_t)T * TOPK, -1);
    if (argc > 2) {
        FILE* f = fopen(argv[2], "rb");
        if (!f || fread(sel.data(), 4, sel.size(), f) != sel.size()) { printf("bad selection file\n"); return 1; }
        fclose(f);
    } else {
        std::mt19937 rng(7);
        for (unsigned i = 0; i < T; i++) {
            if (i < TOPK) { for (unsigned k = 0; k <= i; k++) sel[(size_t)i * TOPK + k] = k; continue; }
            std::vector<int> all(i + 1);
            for (unsigned k = 0; k <= i; k++) all[k] = k;
            std::shuffle(all.begin(), all.end(), rng);
            for (unsigned k = 0; k < TOPK; k++) sel[(size_t)i * TOPK + k] = all[k];
        }
    }
    // Union table exactly as d_index_union_pf lays it out (tile_p = 8).
    const unsigned n_qt = (T + QP - 1) / QP, cap = std::min(QP * TOPK, T);
    const size_t hdr = (n_qt * 4u + 255u) / 256u * 256u;
    std::vector<unsigned char> uni(hdr + (size_t)n_qt * cap * 12u, 0);
    size_t pairs = 0, upairs = 0;
    for (unsigned qt = 0; qt < n_qt; qt++) {
        std::vector<unsigned long long> m(T, 0);
        for (unsigned q = 0; q < QP && qt * QP + q < T; q++) {
            const unsigned qi = qt * QP + q;
            for (unsigned k = 0; k < TOPK; k++) {
                const int s = sel[(size_t)qi * TOPK + k];
                if (s >= 0 && (unsigned)s <= qi) m[s] |= 1ull << q;
            }
        }
        unsigned cnt = 0;
        int* up = (int*)(uni.data() + hdr + (size_t)qt * cap * 12u);
        unsigned* lo = (unsigned*)((unsigned char*)up + (size_t)cap * 4u);
        unsigned* hi = (unsigned*)((unsigned char*)up + (size_t)cap * 8u);
        for (unsigned s = 0; s < T; s++)
            if (m[s]) { up[cnt] = s; lo[cnt] = (unsigned)m[s]; hi[cnt] = (unsigned)(m[s] >> 32); cnt++; pairs += __builtin_popcountll(m[s]); }
        ((unsigned*)uni.data())[qt] = cnt;
        upairs += (size_t)cnt * QP;
    }
    printf("T=%u packs=%u selected pairs %.1fM union pairs %.1fM (%.2fx)\n", T, n_qt, pairs / 1e6, upairs / 1e6,
           (double)upairs / pairs);

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    std::vector<unsigned short> qa((size_t)T * H * 512), qr((size_t)T * H * 64), ck((size_t)T * 512), kr((size_t)T * 64);
    for (auto& v : qa) v = f2b(U(rng));
    for (auto& v : qr) v = f2b(U(rng));
    for (auto& v : ck) v = f2b(U(rng));
    for (auto& v : kr) v = f2b(U(rng));
    const float scale = 1.0f / 16.0f;

    bf16 *dqa, *dqr, *dck, *dkr, *dout; float *dop, *dml; unsigned char* duni; int* dkvl;
    CK(hipMalloc(&dqa, qa.size() * 2)); CK(hipMalloc(&dqr, qr.size() * 2));
    CK(hipMalloc(&dck, ck.size() * 2)); CK(hipMalloc(&dkr, kr.size() * 2));
    CK(hipMalloc(&dout, (size_t)T * H * 512 * 2));
    CK(hipMalloc(&dop, (size_t)T * H * 512 * 4)); CK(hipMalloc(&dml, (size_t)T * H * 2 * 4));
    CK(hipMalloc(&duni, uni.size())); CK(hipMalloc(&dkvl, 4));
    CK(hipMemcpy(dqa, qa.data(), qa.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dqr, qr.data(), qr.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dck, ck.data(), ck.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dkr, kr.data(), kr.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(duni, uni.data(), uni.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dkvl, &T, 4, hipMemcpyHostToDevice));
    CK(hipFuncSetAttribute((const void*)k_sparse, hipFuncAttributeMaxDynamicSharedMemorySize, mla_sp::LDS_BYTES));
    CK(hipFuncSetAttribute((const void*)k_v2, hipFuncAttributeMaxDynamicSharedMemorySize, V2_LDS));

    auto timed = [&](auto launch) {
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        std::vector<float> ts;
        for (int it = 0; it < 6; it++) {
            CK(hipEventRecord(e0)); launch(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it) ts.push_back(ms);
        }
        std::sort(ts.begin(), ts.end());
        return ts[ts.size() / 2];
    };
    const float tv = timed([&] { hipLaunchKernelGGL(k_v2, dim3(256), dim3(256), V2_LDS, 0, dop, dml, dqa, dqr, dck, dkr, dkvl, duni, T, cap, scale); });
    const float ts = timed([&] { hipLaunchKernelGGL(k_sparse, dim3(256), dim3(256), mla_sp::LDS_BYTES, 0, dout, dqa, dqr, dck, dkr, duni, T, cap, scale); });
    const double fl = (double)upairs * H * 1088 * 2;
    printf("v2gather %.3f ms   sparse %.3f ms (%.2fx)   sparse %.0f TF/s on union pairs\n", tv, ts, tv / ts, fl / ts / 1e9);

    std::vector<unsigned short> out((size_t)T * H * 512);
    CK(hipMemcpy(out.data(), dout, out.size() * 2, hipMemcpyDeviceToHost));
    double num = 0, den = 0;
    unsigned checked = 0;
    for (unsigned smp = 0; smp < 48; smp++) {
        const unsigned qi = (unsigned)(((unsigned long long)smp * 2654435761u) % T);
        for (unsigned h = 0; h < H; h++) {
            std::vector<double> sc;
            std::vector<int> keys;
            for (unsigned k = 0; k < TOPK; k++) {
                const int s = sel[(size_t)qi * TOPK + k];
                if (s < 0 || (unsigned)s > qi || std::find(keys.begin(), keys.end(), s) != keys.end()) continue;
                double d = 0;
                for (unsigned j = 0; j < 512; j++) d += (double)b2f(qa[((size_t)qi * H + h) * 512 + j]) * b2f(ck[(size_t)s * 512 + j]);
                for (unsigned j = 0; j < 64; j++) d += (double)b2f(qr[((size_t)qi * H + h) * 64 + j]) * b2f(kr[(size_t)s * 64 + j]);
                sc.push_back(d * scale);
                keys.push_back(s);
            }
            const double mx = *std::max_element(sc.begin(), sc.end());
            double l = 0;
            for (auto& x : sc) { x = exp(x - mx); l += x; }
            for (unsigned j = 0; j < 512; j++) {
                double r = 0;
                for (size_t k = 0; k < keys.size(); k++) r += sc[k] * b2f(ck[(size_t)keys[k] * 512 + j]);
                r /= l;
                const double got = b2f(out[((size_t)qi * H + h) * 512 + j]);
                num += (got - r) * (got - r);
                den += r * r;
            }
            checked++;
        }
    }
    const double rel = sqrt(num / den);
    printf("sparse vs reference: %u (query,head) rows rel-L2 %.3e %s\n", checked, rel, rel < 1e-2 ? "PASS" : "FAIL");
    return rel < 1e-2 ? 0 : 2;
}
