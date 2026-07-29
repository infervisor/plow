/* xcd_map_gfx950_test.hip — WHICH XCD DOES WORKGROUP n LAND ON? Measured, not assumed.
 *
 * plow's L2-domain placement (`PLOW_L2_PLACE`, née `PLOW_NV_PLACE`) computes a packet's L2
 * domain in the COMPILER as `domain = cu / sms_per_partition`, where `cu` is the LOGICAL
 * workgroup index (`interp.hip`: `const unsigned cu = blockIdx.x;`). That is BLOCK
 * partitioning, and it is the NVIDIA GPC model: block 0..17 -> GPC 0 on an H100.
 *
 * If AMD's hardware dispatcher instead assigns workgroups to XCDs ROUND-ROBIN
 * (workgroup n -> XCD n % n_xcd, which is what MI300X is widely documented to do), then a
 * blob built with `cu / 32` would hand every one of domain 0's packets to workgroups
 * 0..31 — which the hardware has scattered across ALL EIGHT XCDs. The result is not a
 * wrong answer; it is the CORRECT answer with L2 locality actively destroyed, invisible to
 * every token-identity test there is. So the formula has to be measured before anything is
 * built on it.
 *
 * `__builtin_amdgcn_s_getreg(HW_REG_XCC_ID)` gives a wave its true XCC/XCD id on gfx94x/gfx95x.
 * HW_REG_HW_ID gives SE_ID / CU_ID within that XCD, which cross-checks the reading: 256
 * workgroups at occupancy 1 must land on 256 DISTINCT (xcc, se, cu) triples.
 *
 * Geometries probed, because the answer may depend on them:
 *   - the interpreter's real decode grid: 256 blocks x 512 threads (8 waves), occ 1
 *   - the same grid at occupancy 2 (512 blocks), because `Builder::finish` skips placement
 *     when `n_cu > P*sms` precisely for this case
 *   - small grids (8, 64) to expose the stride directly
 *
 * Self-contained: no packet, no interpreter, no fixture.
 *
 *   /opt/rocm/bin/hipcc --offload-arch=gfx950 -O2 -w \
 *       runtime/tests/xcd_map_gfx950_test.hip -o /tmp/xcd_map
 *   sg render -c '/tmp/xcd_map'
 *
 * Build OUTSIDE `nix develop` (nix's glibc shadows the system one and hipcc dies with
 * GLIBC_2.38 not found); see plans/knob-contract.md §0a.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* s_getreg immediate encoding: id | (offset << 6) | ((size - 1) << 11). Spelled numerically
 * rather than with the `hwreg(...)` assembler sugar so this compiles on any ROCm that has the
 * builtin at all. */
#define GETREG_IMM(id, ofs, size) ((id) | ((ofs) << 6) | (((size) - 1) << 11))
#define HWREG_XCC_ID 20 /* gfx940+: bits [3:0] = XCC id */
#define HWREG_HW_ID 4   /* gfx9: CU_ID [11:8], SH_ID [12], SE_ID [15:13] */

struct Rec {
    unsigned xcc;
    unsigned hwid;
};

/* One record per block, written by lane 0. `nwaves` sets the wave count so the launch matches
 * the interpreter's; `occ` picks waves-per-EU so we can force occupancy 1 vs 2 per CU. */
template <int THREADS, int OCC>
__global__ __launch_bounds__(THREADS, OCC) void probe(Rec* out) {
    if (threadIdx.x == 0) {
        out[blockIdx.x].xcc = __builtin_amdgcn_s_getreg(GETREG_IMM(HWREG_XCC_ID, 0, 32)) & 0xf;
        out[blockIdx.x].hwid = __builtin_amdgcn_s_getreg(GETREG_IMM(HWREG_HW_ID, 0, 32));
    }
}

#define CK(x)                                                                              \
    do {                                                                                   \
        hipError_t e_ = (x);                                                               \
        if (e_ != hipSuccess) {                                                            \
            printf("FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            return 1;                                                                      \
        }                                                                                  \
    } while (0)

static unsigned hw_cu(unsigned h) { return (h >> 8) & 0xf; }
static unsigned hw_sh(unsigned h) { return (h >> 12) & 0x1; }
static unsigned hw_se(unsigned h) { return (h >> 13) & 0x7; }

/* Score a candidate formula against the measured map. Returns the number of blocks whose
 * predicted domain equals the measured XCC id. */
static int score(const Rec* r, int n, int n_xcd, int per, int round_robin) {
    int hit = 0;
    for (int i = 0; i < n; i++) {
        unsigned pred = round_robin ? (unsigned)(i % n_xcd) : (unsigned)(i / per);
        if (pred < (unsigned)n_xcd && pred == r[i].xcc) hit++;
    }
    return hit;
}

static int run(const char* label, int blocks, int threads, int occ, Rec* d, Rec* h, int n_xcd) {
    memset(h, 0xff, sizeof(Rec) * blocks);
    CK(hipMemset(d, 0xff, sizeof(Rec) * blocks));
    if (threads == 512 && occ == 1)
        hipLaunchKernelGGL((probe<512, 1>), dim3(blocks), dim3(512), 0, 0, d);
    else if (threads == 512 && occ == 2)
        hipLaunchKernelGGL((probe<512, 2>), dim3(blocks), dim3(512), 0, 0, d);
    else if (threads == 256 && occ == 1)
        hipLaunchKernelGGL((probe<256, 1>), dim3(blocks), dim3(256), 0, 0, d);
    else
        hipLaunchKernelGGL((probe<256, 4>), dim3(blocks), dim3(256), 0, 0, d);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h, d, sizeof(Rec) * blocks, hipMemcpyDeviceToHost));

    int per = (blocks + n_xcd - 1) / n_xcd;
    int rr = score(h, blocks, n_xcd, per, 1);
    int blk = score(h, blocks, n_xcd, per, 0);

    printf("\n=== %s : %d blocks x %d thr, occ %d ===\n", label, blocks, threads, occ);
    printf("  blockIdx -> XCC (first 40): ");
    for (int i = 0; i < blocks && i < 40; i++) printf("%u", h[i].xcc);
    printf("\n");

    int cnt[16];
    memset(cnt, 0, sizeof cnt);
    for (int i = 0; i < blocks; i++)
        if (h[i].xcc < 16) cnt[h[i].xcc]++;
    printf("  blocks per XCC:            ");
    for (int i = 0; i < n_xcd; i++) printf("[%d]=%d ", i, cnt[i]);
    printf("\n");

    printf("  ROUND-ROBIN (n %% %d): %d/%d = %.1f%%\n", n_xcd, rr, blocks,
           100.0 * rr / blocks);
    printf("  BLOCK       (n / %d): %d/%d = %.1f%%\n", per, blk, blocks,
           100.0 * blk / blocks);

    /* Distinctness of the PHYSICAL CU triple (xcc, se, sh, cu) — not of the whole HW_ID, which
     * also carries WAVE_ID / SIMD_ID / TG_ID and would call two workgroups co-resident on one CU
     * "distinct". At occupancy 1 with blocks == CU count every workgroup should own a physical CU
     * outright; that is what makes an XCD's 32 blocks a real 32-CU / 4 MiB-L2 group rather than a
     * label. A collision at occ 1 means the HW_ID decode is wrong and no conclusion survives it. */
    int coll = 0, percu[8][64];
    memset(percu, 0, sizeof percu);
    for (int i = 0; i < blocks; i++) {
        unsigned key_i = (h[i].xcc << 16) | (hw_se(h[i].hwid) << 8) | (hw_sh(h[i].hwid) << 4) |
                         hw_cu(h[i].hwid);
        for (int j = i + 1; j < blocks; j++) {
            unsigned key_j = (h[j].xcc << 16) | (hw_se(h[j].hwid) << 8) |
                             (hw_sh(h[j].hwid) << 4) | hw_cu(h[j].hwid);
            if (key_i == key_j) {
                coll++;
                break;
            }
        }
        if (h[i].xcc < 8) {
            int slot = (hw_se(h[i].hwid) << 4) | (hw_sh(h[i].hwid) << 3) | (hw_cu(h[i].hwid) & 7);
            if (slot < 64) percu[h[i].xcc][slot]++;
        }
    }
    printf("  distinct phys (xcc,se,sh,cu): %d of %d blocks (%d collide)\n", blocks - coll, blocks,
           coll);
    printf("  distinct CUs touched per XCC: ");
    for (int x = 0; x < n_xcd; x++) {
        int u = 0;
        for (int s = 0; s < 64; s++)
            if (percu[x][s]) u++;
        printf("[%d]=%d ", x, u);
    }
    printf("\n");
    printf("  sample hwid decode: b0 xcc=%u se=%u sh=%u cu=%u | b1 xcc=%u se=%u sh=%u cu=%u | "
           "b8 xcc=%u se=%u sh=%u cu=%u\n",
           h[0].xcc, hw_se(h[0].hwid), hw_sh(h[0].hwid), hw_cu(h[0].hwid), h[1].xcc,
           hw_se(h[1].hwid), hw_sh(h[1].hwid), hw_cu(h[1].hwid), h[8].xcc, hw_se(h[8].hwid),
           hw_sh(h[8].hwid), hw_cu(h[8].hwid));
    return 0;
}

int main(int argc, char** argv) {
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, 0));
    printf("device: %s  CUs=%d  arch=%s\n", p.name, p.multiProcessorCount, p.gcnArchName);

    /* hwspec says MI350X/MI355X is 8 XCDs x 32 CUs. The probe does not take that on faith:
     * `blocks per XCC` below is the observed partition count. */
    const int n_xcd = 8;
    const int maxb = 512;
    Rec* d = nullptr;
    Rec* h = (Rec*)malloc(sizeof(Rec) * maxb);
    CK(hipMalloc(&d, sizeof(Rec) * maxb));

    /* The interpreter's real decode geometry first — that is the launch the answer is for. */
    if (run("interp decode grid", 256, 512, 1, d, h, n_xcd)) return 1;
    if (run("occupancy 2", 512, 512, 2, d, h, n_xcd)) return 1;
    if (run("small: 8 blocks", 8, 512, 1, d, h, n_xcd)) return 1;
    if (run("small: 64 blocks", 64, 512, 1, d, h, n_xcd)) return 1;
    if (run("256 thr / occ 4", 256, 256, 4, d, h, n_xcd)) return 1;

    /* Repeat the headline geometry: the dispatcher is stateful across launches on some
     * firmware, so a mapping that is not REPRODUCIBLE cannot be baked into a blob. */
    if (run("interp decode grid (repeat)", 256, 512, 1, d, h, n_xcd)) return 1;

    hipFree(d);
    free(h);
    printf("\ndone\n");
    return 0;
}
