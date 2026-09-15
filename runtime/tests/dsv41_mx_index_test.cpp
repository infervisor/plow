// WFP8MX scale INDEXING, checked on the host because it does not need a GPU to be settled.
//
// The MFMA accumulation in `d_gemm_t<WFP8MX>` is shared with the WFP8BLK sibling already proven
// on gfx942. What is NEW in the [32,32] arm is only WHERE it reads the scale byte from, and that
// is integer arithmetic -- so it belongs in a test that runs anywhere, not behind a GPU lease.
//
// Two distinct properties, and the second is the one that was actually broken:
//
//   WRONG-SCALE: `nsblk[j]` carries NO LANE COMPONENT --
//       nsblk[j] = (n0 + wn*(BN/WN) + j*MFMA_N) >> 5
//   is one scale row for a whole 32-column fragment. Correct only if every column the fragment
//   owns falls in the SAME 32-wide N block, i.e. the base is 32-aligned. It is: MFMA_N is 32 and
//   `static_assert(BN % (WN * MFMA_N) == 0)` forces BN/WN to be a multiple of 32. Checked here
//   rather than trusted, for every column of every fragment.
//
//   OOB-READ: at a 128-wide N block one BN=128 tile is exactly ONE block row, so the row index
//   can never exceed ceil(N/128) and no guard is needed -- which is why the sibling never wanted
//   one. At 32 a BN=128 tile spans FOUR block rows, so when N is not a multiple of BN the tile's
//   upper j-groups address rows past ceil(N/32). Those fragments own no live column and their
//   output is discarded at the guarded store, but the promotion READS the scale byte first.
//
//   CORRECTION. This test was written claiming DeepSeek-V4.1's `attn.wkv` is N = 576 = 512 latent
//   + 64 rope. It is not: the shard is [512, 5120], read straight from the safetensors header, and
//   576 was devgen's own wrong declaration at the time (`head_dim + qk_rope`, before head_dim was
//   established to already CONTAIN the rope). No V4.1 dense projection has an N that misses 128.
//   So the overread this pins is not reachable from this checkpoint today -- it stays because the
//   guard is a property of the [32,32] arm and not of one model, and because the N=130 and N=33
//   cases below exercise it directly. `op_gemm_common.h` clamps the row, exactly as the K axis was
//   already clamped, and this test models the clamp and pins both properties.
//
// Worth noting why the GPU test did not stand in for this: it already covers N=576, N=130 and
// N=33, so it exercised the overread every run. hipMalloc pads to page granularity, so 320 bytes
// past a buffer end would almost certainly not have faulted -- it would have PASSED with the bug
// latent. An out-of-bounds read is not reliably observable from the output of the kernel doing it.
//
// Build and run (no GPU, no ROCm):
//     c++ -O2 -std=c++17 -o /tmp/mxidx runtime/tests/dsv41_mx_index_test.cpp && /tmp/mxidx
#include <cstdio>
#include <cstdlib>
#include <vector>

// The compiled geometry: d_gemm_fp8_mx instantiates d_gemm_t<GM_MX_BM, GM_MX_BN, 32, GM_WM, GM_WN,...>
static const int BM = 128, BN = 128, BK = 32;
static const int MFMA_M = 32, MFMA_N = 32;

static int fails = 0, checks = 0;

// Exactly the kernel's arithmetic, transcribed from op_gemm_common.h.
static unsigned kernel_nsblk(unsigned n0, int wn, int j, int WN, unsigned N) {
    unsigned ns = (unsigned)(n0 + wn * (BN / WN) + j * MFMA_N) >> 5;
    const unsigned nbmax = ((N + 31u) >> 5) - 1u;   // the clamp added to op_gemm_common.h
    if (ns > nbmax) ns = nbmax;
    return ns;
}

static void mx_case(const char* label, unsigned N, unsigned K, int WN) {
    const int SN = BN / WN / MFMA_N;
    const unsigned KB = (K + 31u) >> 5;          // kernel's K-scale-block count
    const unsigned NT = (K + BK - 1u) / BK;      // k-tiles; KEXACT means K % 32 == 0
    unsigned wrong_scale = 0, oob = 0, clamped = 0;

    for (unsigned n0 = 0; n0 < N; n0 += BN) {
        for (int wn = 0; wn < WN; wn++) {
            for (int j = 0; j < SN; j++) {
                const unsigned base = n0 + (unsigned)(wn * (BN / WN) + j * MFMA_N);
                const unsigned ns = kernel_nsblk(n0, wn, j, WN, N);
                // THE ALIGNMENT CLAIM: the fragment base must be a multiple of 32, or the
                // lane-free nsblk is wrong for part of the fragment.
                if (base % 32u != 0u) { wrong_scale++; continue; }
                // Every one of the fragment's MFMA_N columns must agree with the lane-free row.
                for (int lane_n = 0; lane_n < MFMA_N; lane_n++) {
                    const unsigned nn = base + (unsigned)lane_n;
                    if (nn >= N) continue;                 // ragged N is guarded per element
                    // The clamp must NEVER change the row of a column that really exists.
                    if ((nn >> 5) != ns) { wrong_scale++; }
                    checks++;
                }
                for (unsigned kt = 0; kt < NT; kt++) {
                    unsigned kb = kt;
                    if (kb >= KB) { kb = KB - 1; clamped++; }
                    // The byte this fragment reads for k-tile kt...
                    const size_t got = (size_t)ns * KB + kb;
                    // ...must be the block CONTAINING element (base, kt*BK).
                    const unsigned kcol = kt * BK;
                    // Only fragments with a live column have a "correct" answer to compare to;
                    // a fully-out-of-range fragment just has to stay in bounds.
                    if (base < N) {
                        const size_t want = (size_t)(base >> 5) * KB + (kcol >> 5);
                        if (got != want) { wrong_scale++; }
                    }
                    if (got >= (size_t)((N + 31) / 32) * KB) { oob++; }  // in-bounds
                    checks++;
                }
            }
        }
    }
    printf("  %-24s N=%-5u K=%-5u NB=%-4u  %s", label, N, K, (N + 31) / 32,
           (wrong_scale || oob) ? "FAIL" : "ok");
    if (wrong_scale) printf("  WRONG-SCALE x%u", wrong_scale);
    if (oob) printf("  OOB-READ x%u", oob);
    if (clamped) printf("  (kb clamp x%u)", clamped);
    printf("\n");
    if (wrong_scale || oob) fails++;
}

int main() {
    printf("WFP8MX scale-index check (host, no GPU)\n");
    // The real DeepSeek-V4.1 projection shapes, hidden 5120.
    for (int WN : {2, 4}) {
        printf("-- wave grid WN=%d --\n", WN);
        mx_case("attn.wq_a  [1280,5120]", 1280, 5120, WN);
        mx_case("attn.wq_b  [4608,1280]", 4608, 1280, WN);
        mx_case("attn.wkv   [ 512,5120]", 512, 5120, WN);
        /* N=576 is NOT a V4.1 shape (see the CORRECTION above); kept purely as a ragged case,
         * because 576 is the smallest interesting N that misses 128 while exceeding one BN tile. */
        mx_case("ragged N=576            ", 576, 5120, WN);
        mx_case("attn.wo_a  [8192,4096]", 8192, 4096, WN);
        mx_case("attn.wo_b  [5120,8192]", 5120, 8192, WN);
        mx_case("shared_exp [2304,5120]", 2304, 5120, WN);
        mx_case("engram.wkv [2048,6144]", 2048, 6144, WN);
        mx_case("indexer.wq_b [4096,1280]", 4096, 1280, WN);
        // Ragged tails: N not a multiple of BN, and the smallest legal K.
        mx_case("ragged N=33", 33, 512, WN);
        mx_case("ragged N=129", 129, 1024, WN);
        mx_case("ragged N=160", 160, 32, WN);
        mx_case("min K=32", 512, 32, WN);
    }
    printf(fails ? "\nFAILED (%d cases)\n" : "\nALL PASS (%d cases bad)\n", fails);
    printf("%d index assertions checked\n", checks);
    return fails ? 1 : 0;
}
