/* k3_moe_tp8_bench.hip — the K3 routed-expert DECODE bodies at the shape the emitter actually
 * emits, and the only instrument in this tree that runs them at all.
 *
 * WHY IT EXISTS. `amd-bench` binds routed-expert weights only under `--checkpoint`
 * (`exec/amd.rs`: `if let Some(c) = ckpt` -> `bind_packed_experts`), and the Kimi-K3 checkpoint in
 * this tree is CONFIG ONLY — 412 KB, no safetensors. So on every K3 number this campaign has
 * published, `expert_weight_table` is the zero-filled tail allocation, every routed slot returns
 * at the `wg_base == 0ull` test, and ops 48/49 execute NO GEMV. The `PLOW_K3_ABLATE=48,49` line in
 * `plans/k3-decode-perf.md` ("MoE group glu+down 0.62 ms") is therefore the cost of a body that is
 * not running, and so is `PLOW_MOE_GROUP_FLAT=1`'s recorded -0.76 ms loss.
 *
 * SHAPE. Read off the emitted packet, not assumed: `MoeGroupGluFp8Blk k=16 I_moe=384 H=3584
 * n_exp=896 enc=2`. Note `I_moe` is the PER-RANK expert intermediate (3072/8 at TP8) — the earlier
 * probe in this tree used 3072, the TP1 value, and at 3072 the DOWN lane cliff this bench found
 * does not exist.
 *
 * DISCIPLINE. 1.97 GB arena (896 experts) with the routing table ROTATED every rep, so the 35 MB
 * working set never sits in the 256 MB LLC; hipEvent over 112 reps; two passes, which agree to
 * 0.5%; and a peak-read REFERENCE kernel over the same arena that is the validity gate — a shared
 * GPU reads 4060 GB/s clean and 100-500 GB/s contended, and any pass whose reference is low is
 * discarded rather than reported.
 *
 *     hipcc --offload-arch=gfx950 -O3 -w -DPLOW_BUCKET_DECODE=1 -DPLOW_MXFP4=1 \
 *           perf-data/k3_moe_tp8_bench.hip -o /tmp/k3moe -Iruntime/amd -Iruntime/common
 */
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "amd_common.h"
#include "op_moe.h"

/* PRE-CHANGE group bodies == the per-slot kernels (untouched) under a slot loop. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void b_glu_old(bf16* fu, const bf16* x, const unsigned char* tab, const unsigned long long* wtab,
               const unsigned long long* stab, unsigned k, unsigned I, unsigned H, unsigned n_exp) {
    for (unsigned s = 0; s < k; s++)
        d_moe_expert_glu_fp8_blk(fu, x, tab, wtab, stab, s, I, H, n_exp, PLOW_MOE_ACT_SILU,
                                 blockIdx.x, gridDim.x, PLOW_MOE_ENC_MXFP4, 0.f, 0.f);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void b_down_old(float* part, const bf16* fu, const unsigned char* tab,
                const unsigned long long* wtab, const unsigned long long* stab, unsigned k,
                unsigned H, unsigned I, unsigned n_exp) {
    for (unsigned s = 0; s < k; s++)
        d_moe_expert_down_fp8_blk(part, fu, tab, wtab, stab, s, H, I, n_exp, blockIdx.x,
                                  gridDim.x, PLOW_MOE_ENC_MXFP4);
}
/* SHIPPED group bodies. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void b_glu_new(bf16* fu, const bf16* x, const unsigned char* tab, const unsigned long long* wtab,
               const unsigned long long* stab, unsigned k, unsigned I, unsigned H, unsigned n_exp) {
    d_moe_group_glu_fp8_blk(fu, x, tab, wtab, stab, k, I, H, n_exp, PLOW_MOE_ACT_SILU, blockIdx.x,
                            gridDim.x, PLOW_MOE_ENC_MXFP4, 0.f, 0.f);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void b_down_new(float* part, const bf16* fu, const unsigned char* tab,
                const unsigned long long* wtab, const unsigned long long* stab, unsigned k,
                unsigned H, unsigned I, unsigned n_exp) {
    d_moe_group_down_fp8_blk(part, fu, tab, wtab, stab, k, H, I, n_exp, blockIdx.x, gridDim.x,
                             PLOW_MOE_ENC_MXFP4);
}
/* Peak-read reference over the same arena — the validity gate for a shared GPU. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void b_stream(float* sink, const unsigned char* arena, unsigned long long nbytes) {
    const unsigned long long tid = (unsigned long long)blockIdx.x * PLOW_THREADS + threadIdx.x;
    const unsigned long long nth = (unsigned long long)gridDim.x * PLOW_THREADS;
    const PLOW_GLOB fp4v32* p = (const PLOW_GLOB fp4v32*)(const PLOW_GLOB void*)arena;
    float acc = 0.0f;
    for (unsigned long long i = tid; i < nbytes / 16; i += nth) {
        const fp4v32 v = p[i];
        acc += (float)(v[0] ^ v[1] ^ v[2] ^ v[3]);
    }
    if (acc == 1234.5f) sink[0] = acc;
}

#define HC(x) do{hipError_t e_=(x); if(e_){printf("hip %s @%d\n",hipGetErrorString(e_),__LINE__);exit(1);} }while(0)

int main(int argc, char** argv) {
    const unsigned H = argc > 3 ? (unsigned)atoi(argv[3]) : 3584;
    const unsigned IM = argc > 4 ? (unsigned)atoi(argv[4]) : 384;
    const unsigned K = 16, NE = 896, NROT = 56;
    const unsigned NBLK = argc > 1 ? (unsigned)atoi(argv[1]) : 256;
    const unsigned REPS = argc > 2 ? (unsigned)atoi(argv[2]) : 112;
    const size_t wsz = (size_t)IM * H / 2, ssz = (size_t)IM * H / 32, per = 3 * (wsz + ssz);
    unsigned char* arena;
    HC(hipMalloc(&arena, per * NE));
    HC(hipMemset(arena, 0x42, per * NE));
    std::vector<unsigned long long> wt(NE * 3), st(NE * 3);
    for (unsigned e = 0; e < NE; e++) {
        unsigned char* b = arena + per * e;
        for (int j = 0; j < 3; j++) {
            wt[e * 3 + j] = (unsigned long long)(size_t)(b + (size_t)j * wsz);
            st[e * 3 + j] = (unsigned long long)(size_t)(b + 3 * wsz + (size_t)j * ssz);
        }
    }
    std::vector<unsigned char> tab((size_t)NROT * K * 8);
    for (unsigned r = 0; r < NROT; r++)
        for (unsigned s = 0; s < K; s++) {
            *(unsigned*)&tab[((size_t)r * K + s) * 8] = (r * K + s) % NE;
            *(float*)&tab[((size_t)r * K + s) * 8 + 4] = 0.05f;
        }
    bf16 *dx, *dfu; float* dpart; unsigned char* dtab;
    unsigned long long *dwt, *dst;
    HC(hipMalloc(&dx, (size_t)H * 2));        HC(hipMemset(dx, 0x3c, (size_t)H * 2));
    HC(hipMalloc(&dfu, (size_t)K * IM * 2));  HC(hipMemset(dfu, 0x3c, (size_t)K * IM * 2));
    HC(hipMalloc(&dpart, (size_t)K * H * 4));
    HC(hipMalloc(&dtab, tab.size()));
    HC(hipMemcpy(dtab, tab.data(), tab.size(), hipMemcpyHostToDevice));
    HC(hipMalloc(&dwt, NE * 24)); HC(hipMemcpy(dwt, wt.data(), NE * 24, hipMemcpyHostToDevice));
    HC(hipMalloc(&dst, NE * 24)); HC(hipMemcpy(dst, st.data(), NE * 24, hipMemcpyHostToDevice));
    const double gb = (double)K * 2.0 * IM * H / 2.0 + (double)K * 2.0 * IM * H / 32.0;
    const double db = (double)K * H * IM / 2.0 + (double)K * H * IM / 32.0;
    hipEvent_t e0, e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    {
        const size_t sb = 512ull << 20;
        auto g2 = [&] { hipLaunchKernelGGL(b_stream, dim3(NBLK), dim3(PLOW_THREADS), 0, 0, (float*)dpart, arena, (unsigned long long)sb); };
        for (int i = 0; i < 3; i++) g2();
        HC(hipDeviceSynchronize()); HC(hipEventRecord(e0));
        for (int i = 0; i < 20; i++) g2();
        HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
        float ms; HC(hipEventElapsedTime(&ms, e0, e1));
        printf("  REFERENCE stream read: %.1f GB/s\n", (double)sb * 20 / (ms * 1e-3) / 1e9);
    }
    unsigned rep = 0;
    auto run = [&](const char* nm, int which, double bytes) {
        auto go = [&] {
            const unsigned char* t = dtab + (size_t)(rep++ % NROT) * K * 8;
            switch (which) {
            case 0: hipLaunchKernelGGL(b_glu_old,  dim3(NBLK), dim3(PLOW_THREADS), 0, 0, dfu, dx, t, dwt, dst, K, IM, H, NE); break;
            case 1: hipLaunchKernelGGL(b_glu_new,  dim3(NBLK), dim3(PLOW_THREADS), 0, 0, dfu, dx, t, dwt, dst, K, IM, H, NE); break;
            case 2: hipLaunchKernelGGL(b_down_old, dim3(NBLK), dim3(PLOW_THREADS), 0, 0, dpart, dfu, t, dwt, dst, K, H, IM, NE); break;
            default:hipLaunchKernelGGL(b_down_new, dim3(NBLK), dim3(PLOW_THREADS), 0, 0, dpart, dfu, t, dwt, dst, K, H, IM, NE); break;
            }
        };
        for (int i = 0; i < 5; i++) go();
        HC(hipDeviceSynchronize()); HC(hipEventRecord(e0));
        for (unsigned i = 0; i < REPS; i++) go();
        HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
        float ms; HC(hipEventElapsedTime(&ms, e0, e1));
        const double us = ms * 1000.0 / REPS;
        printf("  %-20s %8.2f us %7.0f GB/s   x92 layers = %6.3f ms/token\n", nm, us,
               bytes / (us * 1e-6) / 1e9, us * 92 / 1000.0);
    };
    printf("K3 TP8 decode experts: K(H)=%u I_moe=%u top-%u of %u  arena=%.2f GB nblk=%u\n", H, IM,
           K, NE, per * (double)NE / 1e9, NBLK);
    for (int pass = 0; pass < 2; pass++) {
        run("GLU  OLD slot-outer", 0, gb);
        run("GLU  NEW flat+x2", 1, gb);
        run("DOWN OLD slot-outer", 2, db);
        run("DOWN NEW flat+lg", 3, db);
        printf("  --\n");
    }
    {
        const size_t fub = (size_t)K * IM * 2, pab = (size_t)K * H * 4;
        std::vector<unsigned char> a(fub), b(fub), c(pab), d(pab);
        hipLaunchKernelGGL(b_glu_old, dim3(NBLK), dim3(PLOW_THREADS), 0, 0, dfu, dx, dtab, dwt, dst, K, IM, H, NE);
        HC(hipDeviceSynchronize()); HC(hipMemcpy(a.data(), dfu, fub, hipMemcpyDeviceToHost));
        HC(hipMemset(dfu, 0, fub));
        hipLaunchKernelGGL(b_glu_new, dim3(NBLK), dim3(PLOW_THREADS), 0, 0, dfu, dx, dtab, dwt, dst, K, IM, H, NE);
        HC(hipDeviceSynchronize()); HC(hipMemcpy(b.data(), dfu, fub, hipMemcpyDeviceToHost));
        printf("  BITEXACT glu  OLD vs NEW : %s\n", memcmp(a.data(), b.data(), fub) ? "DIFFER" : "identical");
        hipLaunchKernelGGL(b_down_old, dim3(NBLK), dim3(PLOW_THREADS), 0, 0, dpart, dfu, dtab, dwt, dst, K, H, IM, NE);
        HC(hipDeviceSynchronize()); HC(hipMemcpy(c.data(), dpart, pab, hipMemcpyDeviceToHost));
        HC(hipMemset(dpart, 0, pab));
        hipLaunchKernelGGL(b_down_new, dim3(NBLK), dim3(PLOW_THREADS), 0, 0, dpart, dfu, dtab, dwt, dst, K, H, IM, NE);
        HC(hipDeviceSynchronize()); HC(hipMemcpy(d.data(), dpart, pab, hipMemcpyDeviceToHost));
        printf("  BITEXACT down OLD vs NEW : %s\n", memcmp(c.data(), d.data(), pab) ? "DIFFER" : "identical");
    }
    return 0;
}
