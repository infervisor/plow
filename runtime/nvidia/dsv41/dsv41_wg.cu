// Hopper (sm_90a) warpgroup-MMA GEMM for the prefill rungs: C[M][N] = A[M][K] . deq(W)[N][K]^T.
//
// Both quantized operands become bf16 exactly before the tensor cores see them: A is the
// fake-quantized activation (e4m3 * ue8m0, what act_quant's `fq` output holds), W is decoded in
// shared memory from fp8 e4m3 with [32 x 32] ue8m0 block scales (FMT 0, the dense layers) or
// packed fp4 e2m1 with per-row 32-wide ue8m0 scales (FMT 1, the routed experts). Every product is
// then exact in fp32 and the bf16 wgmma accumulates in fp32, so this is the reference's
// per-32-block sum of products in another summation order -- with no per-block promotion on the
// CUDA cores, which is what holds the fp8 mma.sync kernels near 10% of peak. (bench_wg.py at 16k
// rows: ~370 TFLOP/s dense, ~320 grouped fp4, ~1.8x the mma.sync kernels; with the loads and the
// decode removed the same pipeline runs ~800, so the producer side and the bf16 operands' shared
// memory traffic are what is left -- an fp8 wgmma with per-32 promotion is the next step.)
//
// CTA: 384 threads = warpgroup 0 (producer) + warpgroups 1, 2 (consumers, 64 rows each).
// Tile 128 x WG_BN, 64 K per stage, WG_ST stages. Per stage the producer issues cp.async for the A
// tile (straight into the 128B-swizzled layout wgmma reads) and the raw W tile (into a staging
// buffer), both tracked by a `ld` mbarrier (cp.async.mbarrier.arrive.noinc); WG_LOOK stages later it
// waits that barrier, decodes the raw W to bf16 into the swizzled B tile, fences the async proxy
// and arrives on `full`. Decode comes before the next issue in the loop: the slot that issue
// refills is released by the consumers only once they have the stage just decoded. Consumers wait `full`, issue 4 wgmma m64nBNk16 per stage and release the
// slot through `empty` once the next stage's wgmmas are in flight.
//
// Grouped mode (MoE): tiles != null, tile blockIdx.y = (expert e, row0) over the expert-sorted rows
// row0 .. min(row0 + 128, offs[e + 1]); A row p is rows[p] (the token), or p itself when a_by_row;
// W / scales offset by e * stride.
// Dense mode: row0 = blockIdx.y * 128, rows up to M. blockIdx.z = batch (strides *_bstride).
// K % 64 == 0; N % 128 == 0 is not required (rows past N decode as zero, stores are guarded).
#define WG_BM 128
#ifndef WG_BN
#define WG_BN 256  // 128 x 256 tiles: 16 KiB of A + 16 (fp8) / 8 (fp4) KiB of raw W per 2M MACs
#endif
#define WG_BK 64
#if WG_BN == 256
#define WG_ST 3
#define WG_LOOK 2  // <= WG_ST - 1: stage ks + LOOK reuses the slot consumers release after stage ks arrives
#else
#define WG_ST 5
#define WG_LOOK 4
#endif
#define WG_TILE_BYTES (WG_BM * WG_BK * 2)  // A tile, bf16, 16 KiB
#define WG_BTILE_BYTES (WG_BN * WG_BK * 2) // B tile, bf16
#define WG_RAW_LD(FMT) ((FMT) == 0 ? 80 : 48)  // raw W row: 64 B fp8 / 32 B fp4 + pad (conflict-free 16 B reads)
#define WG_RAW_BYTES (WG_BN * 80)
#define WG_SMEM (WG_ST * (WG_TILE_BYTES + WG_BTILE_BYTES + WG_RAW_BYTES) + 3 * WG_ST * 8 + WG_BM * 4 + 1024)

// smem_u32: dsv41_gemm.cu
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_cp_async_arrive(uint64_t* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n"
        "}\n" ::"r"(smem_u32(b)),
        "r"(parity)
        : "memory");
}
__device__ __forceinline__ void wg_cp_async16(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
// K-major tile of 8-row x 128 B swizzle atoms: row r, 16-byte chunk c at r * 128 + ((c ^ (r & 7)) * 16)
__device__ __forceinline__ uint32_t sw128(int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); }
__device__ __forceinline__ uint64_t wg_desc(uint32_t addr) {
    // start (16 B units), leading byte offset 1 (unused for swizzled K-major), stride byte offset
    // 1024 B (one 8-row atom), layout 128B swizzle
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ void wg_mma_bf16_m64n128k16(float* d, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(da), "l"(db), "r"(1));
}
__device__ __forceinline__ void wg_mma_bf16_m64n256k16(float* d, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, "
        "%128, %129, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(da), "l"(db), "r"(1));
}

__device__ __forceinline__ uint32_t f2_to_bf2(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *(uint32_t*)&v;
}
// e4m3 -> bf16 without the (quarter-rate) conversion units: an e4m3 byte s.eeee.mmm moved to the bf16
// bit positions s.0000eeee.mmm0000 is the bf16 encoding of its value * 2^-120 -- subnormals included,
// both formats keep the subnormal step where the exponent field is 0 -- so one bf16x2 multiply by
// 2^(120 + S) applies a power-of-two block scale 2^S exactly. `t` holds the two bytes at bits 0-7 and
// 16-23 (other bits ignored).
__device__ __forceinline__ uint32_t e4m3x2_bits_to_bf16x2(uint32_t t, __nv_bfloat162 scale2) {
    const uint32_t r = ((t << 4) & 0x07F007F0u) | ((t << 8) & 0x80008000u);
    const __nv_bfloat162 v = __hmul2(*(const __nv_bfloat162*)&r, scale2);
    return *(const uint32_t*)&v;
}
// 2^x as bf16x2 (both halves), x in [-126, 127]
__device__ __forceinline__ __nv_bfloat162 bf16x2_pow2(int x) {
    const uint32_t h = (uint32_t)(x + 127) << 7;
    const uint32_t r = h | (h << 16);
    return *(const __nv_bfloat162*)&r;
}

template <int FMT>
__device__ __forceinline__ void wg_gemm(void* __restrict__ C, const bf16* __restrict__ A, const uint8_t* __restrict__ W,
                                        const uint8_t* __restrict__ sw, const int* __restrict__ tiles, const int* __restrict__ meta,
                                        const int* __restrict__ offs, const int* __restrict__ rows, int a_by_row, int M, int N, int K,
                                        long long lda, long long ldc, int c_f32,
                                        long long w_estride, long long s_estride, long long a_bstride, long long c_bstride) {
    extern __shared__ __align__(1024) uint8_t wg_smem_raw[];
    uint8_t* smem = (uint8_t*)(((uintptr_t)wg_smem_raw + 1023) & ~(uintptr_t)1023);
    uint8_t* As = smem;                                   // [ST][128 x 128 B]
    uint8_t* Bs = As + WG_ST * WG_TILE_BYTES;             // [ST][BN x 128 B]
    uint8_t* Rs = Bs + WG_ST * WG_BTILE_BYTES;            // [ST][raw]
    uint64_t* bar_ld = (uint64_t*)(Rs + WG_ST * WG_RAW_BYTES);
    uint64_t* bar_full = bar_ld + WG_ST;
    uint64_t* bar_empty = bar_full + WG_ST;
    int* Arow = (int*)(bar_empty + WG_ST);  // [128] the tile's A row indices, -1 past the tile

    int row0, rend;
    const uint8_t* We = W;
    const uint8_t* Se = sw;
    if (tiles) {
        const int tile = blockIdx.y;
        if (tile >= meta[0]) return;
        const int e = tiles[tile * 2];
        row0 = tiles[tile * 2 + 1];
        rend = min(offs[e + 1], row0 + WG_BM);
        We += (long long)e * w_estride;
        Se += (long long)e * s_estride;
    } else {
        row0 = blockIdx.y * WG_BM;
        if (row0 >= M) return;
        rend = min(M, row0 + WG_BM);
        We += (long long)blockIdx.z * w_estride;
        Se += (long long)blockIdx.z * s_estride;
        A += (long long)blockIdx.z * a_bstride;
    }
    const int n0 = blockIdx.x * WG_BN;
    const int KS = K / WG_BK, KB = K >> 5;
    const int tid = threadIdx.x, wg = tid >> 7, lt = tid & 127;

    if (tid < WG_BM) {
        const int p = row0 + tid;
        Arow[tid] = p < rend ? (tiles && !a_by_row ? rows[p] : p) : -1;
    }
    if (tid == 0) {
        for (int s = 0; s < WG_ST; s++) {
            mbar_init(&bar_ld[s], 128);
            mbar_init(&bar_full[s], 128);
            mbar_init(&bar_empty[s], 256);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (wg == 0) {
        // ------------------------------------------------------------------ producer
        constexpr int NR = WG_BN / 128;  // this thread's W rows in the tile: lt + 128 * rr
        const uint8_t* wrow[NR];
        const uint16_t* srow[NR];  // the row's ue8m0 scales, two (one stage) per 16-bit load
#pragma unroll
        for (int rr = 0; rr < NR; rr++) {
            const int gn = min(n0 + lt + rr * 128, N - 1);
            wrow[rr] = FMT == 0 ? We + (long long)gn * K : We + (long long)gn * (K / 2);
            srow[rr] = (const uint16_t*)(FMT == 0 ? Se + (long long)(gn >> 5) * KB : Se + (long long)gn * KB);
        }
        // this thread's 8 A chunks sit in rows lt / 8 + 16 i, column chunk lt % 8, for every stage
        int arow[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int r = (lt >> 3) + i * 16;
            arow[i] = tiles ? Arow[r] : (row0 + r < rend ? row0 + r : -1);
        }
        // scales of the stages in flight, oldest first: loaded at issue, first used WG_LOOK stages
        // later at decode (a load in the decode itself stalled every stage on a memory round trip)
        uint32_t scp[WG_LOOK][NR];
        auto issue = [&](int ks) {
            const int slot = ks % WG_ST;
            if (ks >= WG_ST) mbar_wait(&bar_empty[slot], ((ks / WG_ST) - 1) & 1);
#pragma unroll
            for (int rr = 0; rr < NR; rr++) scp[WG_LOOK - 1][rr] = srow[rr][ks];
            // A: 128 rows x 8 chunks, 8 per thread
            const uint32_t a_base = smem_u32(As + slot * WG_TILE_BYTES);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int r = (lt >> 3) + i * 16, c = lt & 7;
                const int ar = arow[i];
                const bf16* src = A + (long long)max(ar, 0) * lda + ks * WG_BK + c * 8;
                wg_cp_async16(a_base + sw128(r, c), src, ar >= 0 ? 16 : 0);
            }
            // raw W: rows n, 64 B (fp8) or 32 B (fp4) of this stage
            const uint32_t r_base = smem_u32(Rs + slot * WG_RAW_BYTES);
#pragma unroll
            for (int rr = 0; rr < NR; rr++) {
                const int n = lt + rr * 128;
                const bool nv = n0 + n < N;
                if (FMT == 0) {
#pragma unroll
                    for (int i = 0; i < 4; i++) wg_cp_async16(r_base + n * WG_RAW_LD(0) + i * 16, wrow[rr] + ks * WG_BK + i * 16, nv ? 16 : 0);
                } else {
#pragma unroll
                    for (int i = 0; i < 2; i++) wg_cp_async16(r_base + n * WG_RAW_LD(1) + i * 16, wrow[rr] + ks * (WG_BK / 2) + i * 16, nv ? 16 : 0);
                }
            }
            mbar_cp_async_arrive(&bar_ld[slot]);
        };
        auto decode = [&](int ks) {
            const int slot = ks % WG_ST;
            mbar_wait(&bar_ld[slot], (ks / WG_ST) & 1);
            uint8_t* b_tile = Bs + slot * WG_BTILE_BYTES;
            const uint8_t* raw = Rs + slot * WG_RAW_BYTES;
#pragma unroll
            for (int kr = 0; kr < 2 * NR; kr++) {  // (row, 32-wide scale block) pairs of this stage
                const int kb2 = kr & 1, n = lt + (kr >> 1) * 128, gn = n0 + n;
                const bool nv = gn < N;
                const int kb = ks * 2 + kb2;
                // the block scale 2^(sb - 127) times the placement factor: 2^120 (fp8), 2^126 (fp4 via
                // the 2^-6 shift decode); a row past N decodes as zero. Weight scales here stay far below
                // 2^7 (fp8) / 2^1 (fp4), where the factor would leave the bf16 exponent range.
                const int sb = (int)((scp[0][kr >> 1] >> (8 * kb2)) & 0xffu);
                const __nv_bfloat162 sc2 = nv ? bf16x2_pow2(sb - 127 + (FMT == 0 ? 120 : 126)) : __floats2bfloat162_rn(0.f, 0.f);
                // this scale block's 32 K: 4 chunks of 8, each 4 bf16 pairs in K order
                uint32_t out[4][4];
                if (FMT == 0) {
                    const uint4 p0 = *(const uint4*)(raw + n * WG_RAW_LD(0) + kb2 * 32);
                    const uint4 p1 = *(const uint4*)(raw + n * WG_RAW_LD(0) + kb2 * 32 + 16);
                    const uint32_t w8[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
                    for (int q = 0; q < 8; q++) {  // word q: elements 4q .. 4q+3
                        out[q >> 1][(q & 1) * 2] = e4m3x2_bits_to_bf16x2(__byte_perm(w8[q], 0, 0x4140), sc2);
                        out[q >> 1][(q & 1) * 2 + 1] = e4m3x2_bits_to_bf16x2(__byte_perm(w8[q], 0, 0x4342), sc2);
                    }
                } else {
                    const uint4 p0 = *(const uint4*)(raw + n * WG_RAW_LD(1) + kb2 * 16);
                    const uint32_t w4[4] = {p0.x, p0.y, p0.z, p0.w};
#pragma unroll
                    for (int c4 = 0; c4 < 4; c4++) {
                        // element 2i in the low nibble of byte i: the shift decode gives the even
                        // elements (ev) and the odd ones (od) as e4m3 bytes; pair i = (ev.b_i, od.b_i)
                        const uint32_t ev = fp4x8_lo(w4[c4]), od = fp4x8_hi(w4[c4]);
                        out[c4][0] = e4m3x2_bits_to_bf16x2(__byte_perm(ev, od, 0x4440), sc2);
                        out[c4][1] = e4m3x2_bits_to_bf16x2(__byte_perm(ev, od, 0x5551), sc2);
                        out[c4][2] = e4m3x2_bits_to_bf16x2(__byte_perm(ev, od, 0x6662), sc2);
                        out[c4][3] = e4m3x2_bits_to_bf16x2(__byte_perm(ev, od, 0x7773), sc2);
                    }
                }
#pragma unroll
                for (int c4 = 0; c4 < 4; c4++)
                    *(uint4*)(b_tile + sw128(n, kb2 * 4 + c4)) = make_uint4(out[c4][0], out[c4][1], out[c4][2], out[c4][3]);
            }
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            mbar_arrive(&bar_full[slot]);
        };
        auto shift = [&]() {
#pragma unroll
            for (int i = 0; i + 1 < WG_LOOK; i++)
#pragma unroll
                for (int rr = 0; rr < NR; rr++) scp[i][rr] = scp[i + 1][rr];
        };
        // prologue: stages 0 .. LOOK-1 (scales land in the last pipeline entry, then shift down)
        for (int ks = 0; ks < WG_LOOK; ks++) {
            if (ks < KS) issue(ks);
            if (ks + 1 < WG_LOOK) shift();
        }
        for (int ks = 0; ks < KS; ks++) {
            decode(ks);
            shift();
            if (ks + WG_LOOK < KS) issue(ks + WG_LOOK);
        }
    } else {
        // ------------------------------------------------------------------ consumers
        const int cw = wg - 1;  // rows cw * 64 .. + 64
        float d[WG_BN / 2];
#pragma unroll
        for (int i = 0; i < WG_BN / 2; i++) d[i] = 0.f;
        for (int ks = 0; ks < KS; ks++) {
            const int slot = ks % WG_ST;
            mbar_wait(&bar_full[slot], (ks / WG_ST) & 1);
            const uint64_t da = wg_desc(smem_u32(As + slot * WG_TILE_BYTES + cw * 64 * 128));
            const uint64_t db = wg_desc(smem_u32(Bs + slot * WG_BTILE_BYTES));
            wg_fence();
#pragma unroll
            for (int k = 0; k < WG_BK / 16; k++) {
#if WG_BN == 256
                wg_mma_bf16_m64n256k16(d, da + 2 * k, db + 2 * k);
#else
                wg_mma_bf16_m64n128k16(d, da + 2 * k, db + 2 * k);
#endif
            }
            wg_commit();
            wg_wait<1>();
            if (ks > 0) mbar_arrive(&bar_empty[(ks - 1) % WG_ST]);
        }
        wg_wait<0>();
        // epilogue: warp w of the warpgroup holds rows 16 w + g (+ 8), columns j * 8 + t4 * 2 (+ 1)
        const int w = lt >> 5, lane = lt & 31, g = lane >> 2, t4 = lane & 3;
        const long long cz = tiles ? 0 : (long long)blockIdx.z * c_bstride;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int r = row0 + cw * 64 + w * 16 + g + h * 8;
            if (r >= rend) continue;
#pragma unroll
            for (int j = 0; j < WG_BN / 8; j++) {
                const int col = n0 + j * 8 + t4 * 2;
                if (col >= N) continue;
                const float v0 = d[j * 4 + h * 2], v1 = d[j * 4 + h * 2 + 1];
                if (c_f32) {
                    *(float2*)((float*)C + cz + (long long)r * ldc + col) = make_float2(v0, v1);
                } else {
                    *(__nv_bfloat162*)((bf16*)C + cz + (long long)r * ldc + col) = __floats2bfloat162_rn(v0, v1);
                }
            }
        }
    }
}

// Dense: C (bf16, or f32 when c_f32) [z][M][ldc] = A [z][M][lda] . deq(W8 [z][N][K], sw [z][N/32][K/32])^T.
// grid = (ceil(N / WG_BN), ceil(M / 128), batch), 384 threads, WG_SMEM bytes.
DSV_EXTERN void __launch_bounds__(384, 1)
    dsv_gemm_wg_fp8(void* __restrict__ C, const bf16* __restrict__ A, const uint8_t* __restrict__ W, const uint8_t* __restrict__ sw,
                    int M, int N, int K, long long lda, long long ldc, int c_f32, long long w_bstride, long long s_bstride,
                    long long a_bstride, long long c_bstride) {
    wg_gemm<0>(C, A, W, sw, nullptr, nullptr, nullptr, nullptr, 0, M, N, K, lda, ldc, c_f32, w_bstride, s_bstride, a_bstride, c_bstride);
}

// Grouped fp4 experts: C [p][N] bf16 = A [rows[p] or p][K] (bf16) . deq(W4[e])^T for the 128-row
// tiles of dsv_moe_offsets (bm = 128). grid = (ceil(N / WG_BN), max_tiles), 384 threads, WG_SMEM.
DSV_EXTERN void __launch_bounds__(384, 1)
    dsv_moe_gemm_wg_fp4(bf16* __restrict__ C, const bf16* __restrict__ A, const uint8_t* __restrict__ W, const uint8_t* __restrict__ sw,
                        const int* __restrict__ tiles, const int* __restrict__ meta, const int* __restrict__ offs,
                        const int* __restrict__ rows, int a_by_row, int N, int K, long long w_estride, long long s_estride) {
    wg_gemm<1>(C, A, W, sw, tiles, meta, offs, rows, a_by_row, 0, N, K, K, N, 0, w_estride, s_estride, 0, 0);
}
