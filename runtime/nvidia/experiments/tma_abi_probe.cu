// tma_abi_probe.cu — the TMA *ABI* questions plow must answer before wiring TMA into the
// persistent packet interpreter. Not a performance harness: every test here is a correctness
// or a plumbing question, and the only timings taken are HOST encode cost and device rebind
// latency (both reported as raw latencies, no TF/s claim, so the H100 DVFS harness rules in
// experiments/README.md do not apply).
//
// WHAT THE EXISTING FILES ALREADY SETTLED (not re-derived here):
//   tma_ws_gemm_bf16.cu   — a CUtensorMap in ordinary global memory reached by POINTER works;
//                           device-side tensormap.replace.tile.global_address + proxy fence works;
//                           sizeof(CUtensorMap)=128, alignof=128; encode ~44 ns/call (HOST ONLY).
//   tma_ws_moe_group.cu   — ONE rank-3 map {K,N,E} addresses all E experts; OOB expert zero-fills.
//   tma_ws_flash_prefill.cu — KV needs no per-step rebuild (globalDim[seq] = kv_stride).
//
// WHAT THIS FILE ADDS — each one is a thing plow does and CUTLASS does not:
//   T1  RUNTIME-INDEXED DESCRIPTOR TABLE. The descriptor address is computed on device from an
//       id that was READ OUT OF A PACKET in global memory: desc = tmaps + (id-1)*128. Nothing
//       about it is a compile-time constant or a kernel parameter. This is the plow ABI.
//   T2  ONE TENSOR, TWO TILE SHAPES, ONE TABLE. Descriptor identity is (tensor, box rows), so the
//       selector picking BM=128 for one bucket and BM=64 for another must mint two ids over the
//       same weight. Both must be live simultaneously and independently correct.
//   T3  HANDLE-DERIVED IDS. Ops whose i[] words are all taken (FLASH_PREFILL) cannot carry an id,
//       so the id comes from a `tmap_of[tensor_handle]` side table. Two chased loads, no packet
//       bits. Must be bit-identical to T1.
//   T4  TWO RESIDENT MODELS. Two independent tables alternating across launches, no fence, no
//       cross-talk — the plow_sm120_switch.cu model applied to descriptors.
//   T5  BASE-OFFSET SUB-VIEW == RANK-3 COORDINATE. A rank-2 map encoded at (base + e*stride) must
//       produce the same tile as coordinate e of a rank-3 map. This is what lets ONE spec record
//       cover both the Gemma fused-expert layout and a GLM per-expert layout.
//   T6  ALIGNMENT. Is a descriptor at a 64-B-but-not-128-B offset usable? Decides whether the
//       table may be packed into a shared arena or must be its own 128-B-aligned allocation.
//   T7  REBIND COST. tensormap.replace + fence.proxy.tensormap release/acquire, ns per rebind —
//       the price of the paged-KV / per-packet-rebinding escape hatch.
//   T8  BUILD COST AT PLOW'S REAL DESCRIPTOR BUDGET (295 / 530 / 1050 maps).
//
// BUILD (executables MUST use the -gencode form; -arch=sm_90a alone is rejected for wgmma-class
// features and -arch=native resolves to sm_90):
//   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common -I runtime/nvidia \
//     runtime/nvidia/experiments/tma_abi_probe.cu -o /tmp/tma_abi -lcuda
//   flock /tmp/plow_gpu.lock env LD_LIBRARY_PATH=/usr/local/cuda/compat /tmp/tma_abi

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

typedef __nv_bfloat16 bf16;

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));             \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
#define CKD(x)                                                                                     \
    do {                                                                                           \
        CUresult e_ = (x);                                                                         \
        if (e_ != CUDA_SUCCESS) {                                                                  \
            const char* s_ = "?";                                                                  \
            cuGetErrorString(e_, &s_);                                                             \
            printf("CU ERR %s:%d %s\n", __FILE__, __LINE__, s_);                                   \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static double now_s() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

// ---------------------------------------------------------------- geometry
// BK is PINNED at 128 BYTES by the 128-B swizzle rule, so for bf16 it is 64 elements.
// That is why K never enters the descriptor identity key: only (tensor, box rows) does.
#define BK 64
#define CHUNKS (BK / 8) // 16-byte chunks per swizzled row

// ---------------------------------------------------------------- device helpers
__device__ __forceinline__ uint32_t su32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(su32(b)), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(su32(b)),
                 "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
    asm volatile("{\n.reg .pred p;\nTW%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra TW%=;\n}\n" ::"r"(su32(b)),
                 "r"(parity)
                 : "memory");
}
// 2-D tile load. `map` is a GENERIC 64-bit address — this is the whole point: it may be
// computed at runtime from packet data, it does not have to be a kernel parameter.
__device__ __forceinline__ void tma2d(uint32_t dst, const void* map, int c0, int c1, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst),
                 "l"(map), "r"(c0), "r"(c1), "r"(bar)
                 : "memory");
}
__device__ __forceinline__ void tma3d(uint32_t dst, const void* map, int c0, int c1, int c2,
                                      uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(dst),
                 "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
                 : "memory");
}
__device__ __forceinline__ void tmap_acquire(const void* map) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n" ::"l"(map) : "memory");
}
__device__ __forceinline__ bf16* align1k(void* p) {
    unsigned off = su32(p) & 1023u;
    return (bf16*)((char*)p + ((1024u - off) & 1023u));
}
// Physical offset of logical element (row, k) inside a 128-B-swizzled [rows][BK] bf16 tile.
// Recorded in experiments/README.md; the HARDWARE applies this when the map declares
// CU_TENSOR_MAP_SWIZZLE_128B, so no store-side XOR is ever written by hand on the TMA path.
__device__ __forceinline__ int swz_off(int row, int k) {
    const int c = k >> 3, j = k & 7;
    return row * BK + ((c ^ (row & 7)) * 8) + j;
}

// ---------------------------------------------------------------- the packet mirror
// A stand-in for PlowDevInst: 64 bytes, ids live in i[6]/i[7] packed two u16 per word,
// 1-BASED so that an all-zero legacy packet means "no descriptor" and falls back.
struct Pkt {
    uint16_t op, blocks;
    uint32_t fj[3];
    uint16_t t[8];
    uint32_t i[8];
};
static_assert(sizeof(Pkt) == 64, "packet mirror must be 64 B like PlowDevInst");

#define TMA_ID_NONE 0u
// device: id (1-based) -> descriptor address inside the table
__device__ __forceinline__ const void* tmap(const uint8_t* tmaps, unsigned id) {
    return tmaps + (size_t)(id - 1) * 128;
}

// ---------------------------------------------------------------- probe kernels
//
// Every probe stages ONE [boxRows][BK] tile through TMA into smem and writes it back
// UN-SWIZZLED, so the host oracle is a plain strided gather with zero-fill past the edges.

// T1/T2: descriptor id read out of a packet in global memory, table indexed at runtime.
__global__ void k_pkt_indexed(const Pkt* __restrict__ pkt, const uint8_t* __restrict__ tmaps,
                              bf16* __restrict__ out, int c0, int c1, int boxRows, int which) {
    extern __shared__ char smem_raw[];
    uint64_t* bar = (uint64_t*)smem_raw;
    bf16* tile = align1k(smem_raw + 16);

    const unsigned word = pkt->i[6];
    const unsigned id = which == 0 ? (word & 0xFFFFu) : (word >> 16);
    if (id == TMA_ID_NONE) return; // legacy packet: caller runs the cp.async path
    const void* map = tmap(tmaps, id);

    if (threadIdx.x == 0) mbar_init(bar, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        mbar_expect(bar, boxRows * BK * (int)sizeof(bf16));
        tma2d(su32(tile), map, c0, c1, su32(bar));
    }
    __syncthreads();
    mbar_wait(bar, 0);
    for (int L = threadIdx.x; L < boxRows * BK; L += blockDim.x)
        out[L] = tile[swz_off(L / BK, L % BK)];
}

// T3: id derived from a TENSOR HANDLE through a side table — the path for ops with no free
// i[] word (FLASH_PREFILL). Two dependent loads (t[] -> tmap_of[] -> tmaps).
__global__ void k_handle_indexed(const Pkt* __restrict__ pkt, const uint8_t* __restrict__ tmaps,
                                 const uint16_t* __restrict__ tmap_of, bf16* __restrict__ out,
                                 int c0, int c1, int boxRows, int slot) {
    extern __shared__ char smem_raw[];
    uint64_t* bar = (uint64_t*)smem_raw;
    bf16* tile = align1k(smem_raw + 16);

    const unsigned h = pkt->t[slot];
    const unsigned id = (h == 0xFFFFu) ? TMA_ID_NONE : tmap_of[h];
    if (id == TMA_ID_NONE) return;
    const void* map = tmap(tmaps, id);

    if (threadIdx.x == 0) mbar_init(bar, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        mbar_expect(bar, boxRows * BK * (int)sizeof(bf16));
        tma2d(su32(tile), map, c0, c1, su32(bar));
    }
    __syncthreads();
    mbar_wait(bar, 0);
    for (int L = threadIdx.x; L < boxRows * BK; L += blockDim.x)
        out[L] = tile[swz_off(L / BK, L % BK)];
}

// T5: rank-3 map, expert as a coordinate.
__global__ void k_rank3(const uint8_t* __restrict__ tmaps, unsigned id, bf16* __restrict__ out,
                        int c0, int c1, int e, int boxRows) {
    extern __shared__ char smem_raw[];
    uint64_t* bar = (uint64_t*)smem_raw;
    bf16* tile = align1k(smem_raw + 16);
    const void* map = tmap(tmaps, id);
    if (threadIdx.x == 0) mbar_init(bar, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        mbar_expect(bar, boxRows * BK * (int)sizeof(bf16));
        tma3d(su32(tile), map, c0, c1, e, su32(bar));
    }
    __syncthreads();
    mbar_wait(bar, 0);
    for (int L = threadIdx.x; L < boxRows * BK; L += blockDim.x)
        out[L] = tile[swz_off(L / BK, L % BK)];
}

// T7a: rebind a live descriptor, device-side, then release in the tensormap proxy.
__global__ void k_rebind(uint8_t* tmaps, unsigned id, void* new_addr) {
    if (threadIdx.x == 0) {
        void* map = (void*)(tmaps + (size_t)(id - 1) * 128);
        asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;\n" ::"l"(map),
                     "l"(new_addr)
                     : "memory");
        asm volatile("fence.proxy.tensormap::generic.release.gpu;\n" ::: "memory");
    }
}
// T7b: cost of N rebind+fence pairs (the per-tile price of a paged-KV mainloop).
__global__ void k_rebind_loop(uint8_t* tmaps, unsigned id, void* a, void* b, int n) {
    if (threadIdx.x != 0) return;
    void* map = (void*)(tmaps + (size_t)(id - 1) * 128);
    for (int i = 0; i < n; i++) {
        void* p = (i & 1) ? b : a;
        asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;\n" ::"l"(map),
                     "l"(p)
                     : "memory");
        asm volatile("fence.proxy.tensormap::generic.release.gpu;\n" ::: "memory");
    }
}
// Reader that ACQUIRES in the proxy first — mandatory after any device-side replace.
__global__ void k_read_after_rebind(const uint8_t* __restrict__ tmaps, unsigned id,
                                    bf16* __restrict__ out, int c0, int c1, int boxRows) {
    extern __shared__ char smem_raw[];
    uint64_t* bar = (uint64_t*)smem_raw;
    bf16* tile = align1k(smem_raw + 16);
    const void* map = tmap(tmaps, id);
    if (threadIdx.x == 0) {
        tmap_acquire(map);
        mbar_init(bar, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        mbar_expect(bar, boxRows * BK * (int)sizeof(bf16));
        tma2d(su32(tile), map, c0, c1, su32(bar));
    }
    __syncthreads();
    mbar_wait(bar, 0);
    for (int L = threadIdx.x; L < boxRows * BK; L += blockDim.x)
        out[L] = tile[swz_off(L / BK, L % BK)];
}

// ---------------------------------------------------------------- host side
static uint32_t g_xs = 0x9E3779B9u;
static float frand() {
    g_xs ^= g_xs << 13;
    g_xs ^= g_xs >> 17;
    g_xs ^= g_xs << 5;
    return ((g_xs >> 8) * (1.0f / 8388608.0f)) - 1.0f;
}

static void encode2d(CUtensorMap* m, void* base, int rows, int K, int boxRows) {
    uint64_t gd[2] = {(uint64_t)K, (uint64_t)rows};
    uint64_t gs[1] = {(uint64_t)K * 2};
    uint32_t bd[2] = {(uint32_t)BK, (uint32_t)boxRows};
    uint32_t es[2] = {1, 1};
    memset(m, 0, sizeof(*m));
    CKD(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
static void encode3d(CUtensorMap* m, void* base, int E, int rows, int K, int boxRows) {
    uint64_t gd[3] = {(uint64_t)K, (uint64_t)rows, (uint64_t)E};
    uint64_t gs[2] = {(uint64_t)K * 2, (uint64_t)rows * (uint64_t)K * 2};
    uint32_t bd[3] = {(uint32_t)BK, (uint32_t)boxRows, 1};
    uint32_t es[3] = {1, 1, 1};
    memset(m, 0, sizeof(*m));
    CKD(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, base, gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// Oracle: the tile TMA should have delivered, with OOB zero-fill.
static void oracle_tile(std::vector<float>& want, const std::vector<float>& src, int rows, int K,
                        int c0, int c1, int boxRows) {
    want.assign((size_t)boxRows * BK, 0.f);
    for (int r = 0; r < boxRows; r++) {
        const int gr = c1 + r;
        if (gr < 0 || gr >= rows) continue;
        for (int k = 0; k < BK; k++) {
            const int gk = c0 + k;
            if (gk < 0 || gk >= K) continue;
            want[(size_t)r * BK + k] = src[(size_t)gr * K + gk];
        }
    }
}

static int g_fail = 0;
static void check(const char* what, const std::vector<bf16>& got, const std::vector<float>& want) {
    size_t bad = 0;
    for (size_t i = 0; i < want.size(); i++)
        if (__bfloat162float(got[i]) != want[i]) bad++;
    printf("  %-58s %s  (%zu/%zu mismatched)\n", what, bad ? "FAIL" : "PASS", bad, want.size());
    if (bad) g_fail++;
}

int main() {
    CKD(cuInit(0));
    CUdevice dev;
    CUcontext ctx;
    CKD(cuDeviceGet(&dev, 0));
    CKD(cuDevicePrimaryCtxRetain(&ctx, dev));
    CKD(cuCtxSetCurrent(ctx));
    char name[128];
    CKD(cuDeviceGetName(name, sizeof name, dev));
    printf("== tma_abi_probe on %s ==\n", name);
    printf("   sizeof(CUtensorMap)=%zu alignof=%zu\n", sizeof(CUtensorMap), alignof(CUtensorMap));

    // ---- a "model": two weight tensors, both [rows][K] bf16, K-contiguous ----
    const int rows = 512, K = 640; // K is NOT a multiple of BK -> exercises the K tail
    const int E = 4;               // experts for the rank-3 test
    std::vector<float> W0((size_t)rows * K), W1((size_t)rows * K);
    std::vector<float> WE((size_t)E * rows * K);
    for (auto& x : W0) x = frand();
    for (auto& x : W1) x = frand();
    for (auto& x : WE) x = frand();
    auto to_bf = [](const std::vector<float>& v) {
        std::vector<bf16> o(v.size());
        for (size_t i = 0; i < v.size(); i++) o[i] = __float2bfloat16(v[i]);
        return o;
    };
    // Round the reference through bf16 so the oracle compares bit-for-bit.
    auto round_bf = [](std::vector<float>& v) {
        for (auto& x : v) x = __bfloat162float(__float2bfloat16(x));
    };
    round_bf(W0);
    round_bf(W1);
    round_bf(WE);
    std::vector<bf16> hW0 = to_bf(W0), hW1 = to_bf(W1), hWE = to_bf(WE);

    bf16 *dW0, *dW1, *dWE, *dOut;
    CK(cudaMalloc(&dW0, hW0.size() * 2));
    CK(cudaMalloc(&dW1, hW1.size() * 2));
    CK(cudaMalloc(&dWE, hWE.size() * 2));
    CK(cudaMalloc(&dOut, 256 * BK * 2));
    CK(cudaMemcpy(dW0, hW0.data(), hW0.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW1, hW1.data(), hW1.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWE, hWE.data(), hWE.size() * 2, cudaMemcpyHostToDevice));

    // ---- MODEL A descriptor table, built HOST-side at "load" ----
    // id 1 : W0, box rows 128     (the selector's BM=128 bucket)
    // id 2 : W0, box rows  64     (SAME tensor, different tile shape -> a DIFFERENT descriptor)
    // id 3 : W1, box rows 128
    // id 4 : rank-3 over WE, expert as coordinate 2
    // id 5 : rank-2 over WE + expert 2's byte offset (the base_off form)
    const int NMAP_A = 5;
    std::vector<CUtensorMap> hmapA(NMAP_A);
    encode2d(&hmapA[0], dW0, rows, K, 128);
    encode2d(&hmapA[1], dW0, rows, K, 64);
    encode2d(&hmapA[2], dW1, rows, K, 128);
    encode3d(&hmapA[3], dWE, E, rows, K, 128);
    encode2d(&hmapA[4], (char*)dWE + (size_t)2 * rows * K * 2, rows, K, 128);

    uint8_t* dmapA;
    CK(cudaMalloc(&dmapA, NMAP_A * 128));
    CK(cudaMemcpy(dmapA, hmapA.data(), NMAP_A * 128, cudaMemcpyHostToDevice));

    // ---- MODEL B: a second, independent table over the same allocations ----
    const int NMAP_B = 2;
    std::vector<CUtensorMap> hmapB(NMAP_B);
    encode2d(&hmapB[0], dW1, rows, K, 128); // id 1 in B means something else than id 1 in A
    encode2d(&hmapB[1], dW0, rows, K, 128);
    uint8_t* dmapB;
    CK(cudaMalloc(&dmapB, NMAP_B * 128));
    CK(cudaMemcpy(dmapB, hmapB.data(), NMAP_B * 128, cudaMemcpyHostToDevice));

    // ---- a packet: i[6] = (idB<<16)|idA, ids 1-BASED so a zeroed legacy packet = "none" ----
    Pkt hpkt;
    memset(&hpkt, 0, sizeof hpkt);
    hpkt.op = 8; // PLOW_DOP_GEMM
    hpkt.i[0] = 512;
    hpkt.i[1] = rows;
    hpkt.i[2] = K;
    hpkt.i[6] = (3u << 16) | 1u; // A-side id 1 (W0/box128), B-side id 3 (W1/box128)
    hpkt.i[7] = 0;
    hpkt.t[0] = 0;
    hpkt.t[3] = 11; // a "KV" handle for the handle-derived path
    hpkt.t[4] = 12;
    Pkt* dpkt;
    CK(cudaMalloc(&dpkt, sizeof(Pkt)));
    CK(cudaMemcpy(dpkt, &hpkt, sizeof(Pkt), cudaMemcpyHostToDevice));

    // legacy packet: every i[] zero -> id 0 -> no descriptor
    Pkt hlegacy;
    memset(&hlegacy, 0, sizeof hlegacy);
    hlegacy.op = 8;
    Pkt* dlegacy;
    CK(cudaMalloc(&dlegacy, sizeof(Pkt)));
    CK(cudaMemcpy(dlegacy, &hlegacy, sizeof(Pkt), cudaMemcpyHostToDevice));

    // handle -> id side table (0 = no descriptor for this tensor)
    std::vector<uint16_t> htof(32, 0);
    htof[11] = 1; // handle 11 (K cache) -> descriptor 1
    htof[12] = 3; // handle 12 (V cache) -> descriptor 3
    uint16_t* dtof;
    CK(cudaMalloc(&dtof, htof.size() * 2));
    CK(cudaMemcpy(dtof, htof.data(), htof.size() * 2, cudaMemcpyHostToDevice));

    const size_t SMEM = 16 + 1024 + (size_t)128 * BK * 2;
    std::vector<bf16> got(256 * BK);
    std::vector<float> want;
    auto fetch = [&](int n) {
        got.assign(256 * BK, __float2bfloat16(0.f));
        CK(cudaMemcpy(got.data(), dOut, (size_t)n * BK * 2, cudaMemcpyDeviceToHost));
        got.resize((size_t)n * BK);
        got.resize(256 * BK, __float2bfloat16(0.f));
    };
    auto clear = [&] { CK(cudaMemset(dOut, 0, 256 * BK * 2)); };

    // ================= T1: runtime-indexed table, id out of a packet =================
    printf("\n-- T1  descriptor id READ FROM A PACKET, table indexed at runtime --\n");
    {
        const int c0 = 128, c1 = 256, br = 128;
        clear();
        k_pkt_indexed<<<1, 256, SMEM>>>(dpkt, dmapA, dOut, c0, c1, br, 0);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        fetch(br);
        got.resize((size_t)br * BK);
        oracle_tile(want, W0, rows, K, c0, c1, br);
        check("A-side id from i[6] low half -> W0 tile", got, want);

        clear();
        k_pkt_indexed<<<1, 256, SMEM>>>(dpkt, dmapA, dOut, c0, c1, br, 1);
        CK(cudaDeviceSynchronize());
        fetch(br);
        got.resize((size_t)br * BK);
        oracle_tile(want, W1, rows, K, c0, c1, br);
        check("B-side id from i[6] high half -> W1 tile", got, want);

        // ragged: rows past the end AND a K tail (K=640 is not a multiple of BK=64 * n)
        const int c0t = 576, c1t = 448; // 448+128 = 576 > 512 rows, 576+64 = 640 == K
        clear();
        k_pkt_indexed<<<1, 256, SMEM>>>(dpkt, dmapA, dOut, c0t, c1t, br, 0);
        CK(cudaDeviceSynchronize());
        fetch(br);
        got.resize((size_t)br * BK);
        oracle_tile(want, W0, rows, K, c0t, c1t, br);
        check("ragged rows + K edge zero-filled by the copy engine", got, want);

        // legacy packet -> id 0 -> kernel returns without touching `out`
        clear();
        k_pkt_indexed<<<1, 256, SMEM>>>(dlegacy, dmapA, dOut, c0, c1, br, 0);
        CK(cudaDeviceSynchronize());
        fetch(br);
        got.resize((size_t)br * BK);
        want.assign((size_t)br * BK, 0.f);
        check("legacy all-zero packet -> id 0 -> no TMA (fallback path)", got, want);
    }

    // ================= T2: two tile shapes over ONE tensor, one table =================
    printf("\n-- T2  same tensor, two box shapes, both live in one table --\n");
    {
        const int c0 = 64, c1 = 192;
        Pkt p2 = hpkt;
        p2.i[6] = (2u << 16) | 1u; // low = id 1 (box 128), high = id 2 (box 64)
        CK(cudaMemcpy(dpkt, &p2, sizeof(Pkt), cudaMemcpyHostToDevice));
        clear();
        k_pkt_indexed<<<1, 256, SMEM>>>(dpkt, dmapA, dOut, c0, c1, 128, 0);
        CK(cudaDeviceSynchronize());
        fetch(128);
        got.resize((size_t)128 * BK);
        oracle_tile(want, W0, rows, K, c0, c1, 128);
        check("id 1 -> W0 box rows 128", got, want);

        clear();
        k_pkt_indexed<<<1, 256, SMEM>>>(dpkt, dmapA, dOut, c0, c1, 64, 1);
        CK(cudaDeviceSynchronize());
        fetch(64);
        got.resize((size_t)64 * BK);
        oracle_tile(want, W0, rows, K, c0, c1, 64);
        check("id 2 -> W0 box rows  64 (same tensor, different id)", got, want);
        CK(cudaMemcpy(dpkt, &hpkt, sizeof(Pkt), cudaMemcpyHostToDevice));
    }

    // ================= T3: handle-derived ids (ops with no free i[] word) =================
    printf("\n-- T3  id derived from the TENSOR HANDLE via tmap_of[] --\n");
    {
        const int c0 = 128, c1 = 256, br = 128;
        clear();
        k_handle_indexed<<<1, 256, SMEM>>>(dpkt, dmapA, dtof, dOut, c0, c1, br, 3);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        fetch(br);
        got.resize((size_t)br * BK);
        oracle_tile(want, W0, rows, K, c0, c1, br);
        check("t[3]=11 -> tmap_of[11]=1 -> W0 tile (== T1 result)", got, want);

        clear();
        k_handle_indexed<<<1, 256, SMEM>>>(dpkt, dmapA, dtof, dOut, c0, c1, br, 4);
        CK(cudaDeviceSynchronize());
        fetch(br);
        got.resize((size_t)br * BK);
        oracle_tile(want, W1, rows, K, c0, c1, br);
        check("t[4]=12 -> tmap_of[12]=3 -> W1 tile", got, want);

        clear();
        k_handle_indexed<<<1, 256, SMEM>>>(dpkt, dmapA, dtof, dOut, c0, c1, br, 0);
        CK(cudaDeviceSynchronize());
        fetch(br);
        got.resize((size_t)br * BK);
        want.assign((size_t)br * BK, 0.f);
        check("t[0]=0 -> tmap_of[0]=0 -> no descriptor (fallback)", got, want);
    }

    // ================= T4: two resident models, alternating =================
    printf("\n-- T4  two resident models: independent tables, same ids, alternating --\n");
    {
        const int c0 = 0, c1 = 0, br = 128;
        Pkt p = hpkt;
        p.i[6] = 1u; // id 1 in BOTH tables, meaning different tensors
        CK(cudaMemcpy(dpkt, &p, sizeof(Pkt), cudaMemcpyHostToDevice));
        int okA = 1, okB = 1;
        for (int it = 0; it < 8; it++) {
            uint8_t* tbl = (it & 1) ? dmapB : dmapA;
            clear();
            k_pkt_indexed<<<1, 256, SMEM>>>(dpkt, tbl, dOut, c0, c1, br, 0);
            CK(cudaDeviceSynchronize());
            fetch(br);
            got.resize((size_t)br * BK);
            oracle_tile(want, (it & 1) ? W1 : W0, rows, K, c0, c1, br);
            size_t bad = 0;
            for (size_t i = 0; i < want.size(); i++)
                if (__bfloat162float(got[i]) != want[i]) bad++;
            if (bad) ((it & 1) ? okB : okA) = 0;
        }
        printf("  %-58s %s\n", "8 alternating launches, model A id 1 -> W0", okA ? "PASS" : "FAIL");
        printf("  %-58s %s\n", "8 alternating launches, model B id 1 -> W1", okB ? "PASS" : "FAIL");
        if (!okA || !okB) g_fail++;
        CK(cudaMemcpy(dpkt, &hpkt, sizeof(Pkt), cudaMemcpyHostToDevice));
    }

    // ================= T5: rank-3 expert coordinate vs base-offset sub-view =================
    printf("\n-- T5  rank-3 expert coordinate == rank-2 map at base + e*stride --\n");
    {
        const int c0 = 64, c1 = 128, br = 128, e = 2;
        clear();
        k_rank3<<<1, 256, SMEM>>>(dmapA, 4, dOut, c0, c1, e, br);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        fetch(br);
        got.resize((size_t)br * BK);
        std::vector<float> WEe(WE.begin() + (size_t)e * rows * K,
                               WE.begin() + (size_t)(e + 1) * rows * K);
        oracle_tile(want, WEe, rows, K, c0, c1, br);
        check("rank-3 map, expert as coordinate 2", got, want);

        clear();
        k_pkt_indexed<<<1, 256, SMEM>>>(dpkt, dmapA, dOut, c0, c1, br, 0); // warm, ignored
        CK(cudaDeviceSynchronize());
        Pkt p5 = hpkt;
        p5.i[6] = 5u; // id 5 = rank-2 map already based at expert 2
        CK(cudaMemcpy(dpkt, &p5, sizeof(Pkt), cudaMemcpyHostToDevice));
        clear();
        k_pkt_indexed<<<1, 256, SMEM>>>(dpkt, dmapA, dOut, c0, c1, br, 0);
        CK(cudaDeviceSynchronize());
        fetch(br);
        got.resize((size_t)br * BK);
        check("rank-2 map at base + 2*stride (the base_off spec form)", got, want);

        // out-of-range expert coordinate must zero-fill, not fault
        clear();
        k_rank3<<<1, 256, SMEM>>>(dmapA, 4, dOut, c0, c1, E + 3, br);
        cudaError_t e2 = cudaDeviceSynchronize();
        fetch(br);
        got.resize((size_t)br * BK);
        want.assign((size_t)br * BK, 0.f);
        printf("  (launch status after OOB expert coord: %s)\n", cudaGetErrorString(e2));
        check("out-of-range expert coordinate zero-fills, no fault", got, want);
        CK(cudaMemcpy(dpkt, &hpkt, sizeof(Pkt), cudaMemcpyHostToDevice));
    }

    // ================= T6: descriptor alignment inside the table =================
    printf("\n-- T6  is a descriptor usable at a 64-B-but-not-128-B offset? --\n");
    {
        uint8_t* dmis;
        CK(cudaMalloc(&dmis, 4 * 128));
        CK(cudaMemcpy(dmis + 64, &hmapA[0], 128, cudaMemcpyHostToDevice)); // deliberately +64
        Pkt p6 = hpkt;
        p6.i[6] = 1u;
        CK(cudaMemcpy(dpkt, &p6, sizeof(Pkt), cudaMemcpyHostToDevice));
        const int c0 = 128, c1 = 256, br = 128;
        clear();
        k_pkt_indexed<<<1, 256, SMEM>>>(dpkt, dmis + 64, dOut, c0, c1, br, 0);
        cudaError_t e3 = cudaDeviceSynchronize();
        printf("  launch status at 64-B alignment: %s\n", cudaGetErrorString(e3));
        if (e3 == cudaSuccess) {
            fetch(br);
            got.resize((size_t)br * BK);
            oracle_tile(want, W0, rows, K, c0, c1, br);
            size_t bad = 0;
            for (size_t i = 0; i < want.size(); i++)
                if (__bfloat162float(got[i]) != want[i]) bad++;
            printf("  %-58s %s (%zu bad)\n", "64-B-aligned descriptor read", bad ? "WRONG" : "OK",
                   bad);
        } else {
            CK(cudaGetLastError()); // clear
            CK(cudaDeviceSynchronize());
        }
        CK(cudaFree(dmis));
        CK(cudaMemcpy(dpkt, &hpkt, sizeof(Pkt), cudaMemcpyHostToDevice));
        printf("  (design rule regardless: allocate the table on its own, cudaMalloc is 256-B "
               "aligned, entries are 128 B, so every entry is 128-B aligned by construction)\n");
    }

    // ================= T7: device-side rebind cost =================
    printf("\n-- T7  device-side tensormap.replace + proxy fence --\n");
    {
        const int c0 = 128, c1 = 256, br = 128;
        // rebind descriptor 3 (currently W1) to point at W0, then read it
        k_rebind<<<1, 32>>>(dmapA, 3, dW0);
        CK(cudaDeviceSynchronize());
        clear();
        k_read_after_rebind<<<1, 256, SMEM>>>(dmapA, 3, dOut, c0, c1, br);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        fetch(br);
        got.resize((size_t)br * BK);
        oracle_tile(want, W0, rows, K, c0, c1, br);
        check("descriptor 3 retargeted W1 -> W0, read after acquire", got, want);
        // put it back
        k_rebind<<<1, 32>>>(dmapA, 3, dW1);
        CK(cudaDeviceSynchronize());
        clear();
        k_read_after_rebind<<<1, 256, SMEM>>>(dmapA, 3, dOut, c0, c1, br);
        CK(cudaDeviceSynchronize());
        fetch(br);
        got.resize((size_t)br * BK);
        oracle_tile(want, W1, rows, K, c0, c1, br);
        check("descriptor 3 restored -> W1", got, want);

        for (int n : {1000, 10000}) {
            k_rebind_loop<<<1, 32>>>(dmapA, 2, dW0, dW1, 64); // warm
            CK(cudaDeviceSynchronize());
            double t0 = now_s();
            k_rebind_loop<<<1, 32>>>(dmapA, 2, dW0, dW1, n);
            CK(cudaDeviceSynchronize());
            double dt = now_s() - t0;
            printf("  %6d rebind+release pairs: %8.3f us total, %6.1f ns each\n", n, dt * 1e6,
                   dt * 1e9 / n);
        }
        // restore id 2 (W0, box 64)
        CK(cudaMemcpy(dmapA + 128, &hmapA[1], 128, cudaMemcpyHostToDevice));
    }

    // ================= T8: host encode cost at plow's real budget =================
    printf("\n-- T8  host build cost at plow's descriptor budget --\n");
    {
        for (int n : {295, 530, 1050}) {
            std::vector<CUtensorMap> tmp(n);
            double t0 = now_s();
            for (int i = 0; i < n; i++) encode2d(&tmp[i], dW0, rows, K, (i & 1) ? 128 : 64);
            double dt = now_s() - t0;
            uint8_t* d;
            CK(cudaMalloc(&d, (size_t)n * 128));
            double t1 = now_s();
            CK(cudaMemcpy(d, tmp.data(), (size_t)n * 128, cudaMemcpyHostToDevice));
            double dt1 = now_s() - t1;
            printf("  %5d descriptors: encode %7.1f us (%5.1f ns each), upload %6.1f us, "
                   "%6.1f KiB\n",
                   n, dt * 1e6, dt * 1e9 / n, dt1 * 1e6, n * 128 / 1024.0);
            CK(cudaFree(d));
        }
    }

    printf("\n== %s ==\n", g_fail ? "SOME TESTS FAILED" : "ALL CORRECTNESS TESTS PASS");
    return g_fail ? 1 : 0;
}
