/* PACKET-PROTOCOL UNIT COSTS on gfx942 (MI300X, 8 XCDs, 304 CU).
 *
 * Prices, in isolation, every primitive the interpreter's inter-packet window issues, so the
 * traced per-packet gate cost can be DECOMPOSED rather than attributed by hand:
 *
 *   inv_sc1   buffer_inv sc1     the agent-scope acquire (L1 + XCD L2 invalidate)
 *   inv_l1    buffer_inv         the PLOW_GATE_HIER follower's L1-only invalidate
 *   wbl2_sc1  buffer_wbl2 sc1    the release RMW's writeback
 *   atom_pb   returning global_atomic_add, PER-BLOCK address (zero contention)
 *   atom_1    returning global_atomic_add, ONE shared address (full contention)
 *   ld_sc1    dependent device-scope load (what ctr_poll lowers to)
 *   ld_plain  dependent workgroup-scope load
 *   barrier   s_barrier at the interpreter's 512-thread workgroup
 *   nop       the timing loop itself
 *
 * Every arm is the SAME loop shape (R reps, s_memrealtime around the whole loop, one
 * timestamp pair per block) so `nop` subtracts out. Swept over block counts because the
 * cache-maintenance ops are PER-L2 and SERIALISE across concurrent issuers -- that scaling
 * curve is the whole point.
 *
 * Build:
 *   /opt/rocm-7.2.4/bin/hipcc -O3 --offload-arch=gfx942 -o /tmp/pktproto pktproto.hip
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

#define STRIDE 32u              /* uint32 slots per counter == 128 B, matches PLOW_CTR_STRIDE */
#define THREADS 512             /* the interpreter's workgroup */

/* One timestamp pair per block, written by thread 0. */
struct Rec { unsigned long long t0, t1; };

enum { A_NOP = 0, A_INV_SC1, A_INV_L1, A_INV_SYS, A_WBL2, A_ATOM_PB, A_ATOM_1, A_LD_SC1,
       A_LD_PLAIN, A_BARRIER, A_ATOM_PB_SYS, A_NARM };

static const char* ANAME[A_NARM] = {
    "nop", "inv_sc1", "inv_l1", "inv_sc0sc1", "wbl2_sc1", "atom_pb", "atom_1", "ld_sc1",
    "ld_plain", "barrier", "atom_pb_sys" };

__global__ __launch_bounds__(THREADS) void k_arm(int arm, unsigned R, unsigned* ctr, Rec* rec,
                                                 unsigned* sink) {
    const unsigned b = blockIdx.x;
    unsigned* mine = ctr + (size_t)(b + 1u) * STRIDE;   /* private counter line */
    unsigned* one  = ctr;                               /* the shared one */
    unsigned acc = 0;
    __syncthreads();
    const unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
    for (unsigned i = 0; i < R; i++) {
        switch (arm) {
            case A_NOP:
                asm volatile("s_nop 0" ::: "memory");
                break;
            case A_INV_SC1:
                if (threadIdx.x == 0) asm volatile("buffer_inv sc1" ::: "memory");
                break;
            case A_INV_L1:
                if (threadIdx.x == 0) asm volatile("buffer_inv" ::: "memory");
                break;
            case A_INV_SYS:
                if (threadIdx.x == 0) asm volatile("buffer_inv sc0 sc1" ::: "memory");
                break;
            case A_WBL2:
                if (threadIdx.x == 0) asm volatile("buffer_wbl2 sc1" ::: "memory");
                break;
            case A_ATOM_PB:
                /* DEPENDENT: the returned value feeds the next iteration, so this measures
                 * the full round trip, exactly as the claim/leader-election do. */
                if (threadIdx.x == 0)
                    acc += __hip_atomic_fetch_add(mine, 1u, __ATOMIC_RELAXED,
                                                  __HIP_MEMORY_SCOPE_AGENT);
                break;
            case A_ATOM_PB_SYS:
                if (threadIdx.x == 0)
                    acc += __hip_atomic_fetch_add(mine, 1u, __ATOMIC_RELAXED,
                                                  __HIP_MEMORY_SCOPE_SYSTEM);
                break;
            case A_ATOM_1:
                if (threadIdx.x == 0)
                    acc += __hip_atomic_fetch_add(one, 1u, __ATOMIC_RELAXED,
                                                  __HIP_MEMORY_SCOPE_AGENT);
                break;
            case A_LD_SC1:
                if (threadIdx.x == 0)
                    acc += __hip_atomic_load(mine + (acc & 0u), __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_AGENT);
                break;
            case A_LD_PLAIN:
                if (threadIdx.x == 0)
                    acc += __hip_atomic_load(mine + (acc & 0u), __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_WORKGROUP);
                break;
            case A_BARRIER:
                __syncthreads();
                break;
        }
    }
    const unsigned long long t1 = __builtin_amdgcn_s_memrealtime();
    if (threadIdx.x == 0) {
        rec[b].t0 = t0;
        rec[b].t1 = t1;
        if (acc == 0xdeadbeefu) sink[b] = acc; /* keep acc live */
    }
}

int main(int argc, char** argv) {
    const unsigned R = (argc > 1) ? (unsigned)atoi(argv[1]) : 2000u;
    const int NBS[] = {1, 8, 16, 32, 64, 152, 304};
    const int NNB = sizeof(NBS) / sizeof(NBS[0]);
    const int MAXB = 304;

    unsigned* ctr; CHECK(hipMalloc(&ctr, (size_t)(MAXB + 2) * STRIDE * sizeof(unsigned)));
    CHECK(hipMemset(ctr, 0, (size_t)(MAXB + 2) * STRIDE * sizeof(unsigned)));
    Rec* rec;      CHECK(hipMalloc(&rec, (size_t)MAXB * sizeof(Rec)));
    unsigned* sink;CHECK(hipMalloc(&sink, (size_t)MAXB * sizeof(unsigned)));
    std::vector<Rec> h(MAXB);

    printf("R=%u reps, %d threads/block, s_memrealtime @100MHz (10 ns/tick)\n", R, THREADS);
    printf("%-12s", "arm");
    for (int j = 0; j < NNB; j++) printf("%9d", NBS[j]);
    printf("   <- blocks; cell = ns per op (max over blocks, nop subtracted)\n");

    double base[16];
    for (int a = 0; a < A_NARM; a++) {
        printf("%-12s", ANAME[a]);
        fflush(stdout);
        for (int j = 0; j < NNB; j++) {
            const int nb = NBS[j];
            hipLaunchKernelGGL(k_arm, dim3(nb), dim3(THREADS), 0, 0, a, 64u, ctr, rec, sink);
            CHECK(hipDeviceSynchronize());
            double best = 1e30;
            for (int rep = 0; rep < 3; rep++) {
                hipLaunchKernelGGL(k_arm, dim3(nb), dim3(THREADS), 0, 0, a, R, ctr, rec, sink);
                CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(h.data(), rec, (size_t)nb * sizeof(Rec), hipMemcpyDeviceToHost));
                unsigned long long mx = 0;
                for (int b = 0; b < nb; b++) mx = std::max(mx, h[b].t1 - h[b].t0);
                const double ns = (double)mx * 10.0 / (double)R;
                best = std::min(best, ns);
            }
            if (a == A_NOP) base[j] = best;
            printf("%9.1f", best - (a == A_NOP ? 0.0 : base[j]));
            fflush(stdout);
        }
        printf("\n");
    }
    return 0;
}
