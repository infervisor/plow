/* sm90_wgmma.cuh — shared Hopper (sm_90a) warpgroup-MMA primitives.
 *
 * These are the EXACT primitives validated against f32 CPU oracles on an H100 NVL
 * by the probes in runtime/nvidia/experiments/ (wgmma_bf16_probe.cu,
 * wgmma_fp8_probe.cu, wgmma_moe_group_probe.cu, wgmma_flash_prefill_probe.cu).
 * The PTX bodies below are lifted verbatim from those probes so the numerics are
 * the measured ones, not a re-derivation.
 *
 * Included ONLY when PLOW_NV_HOPPER is defined (interp_sm90a.cu). sm_120a has no
 * wgmma; it keeps the shared mma.sync bodies.
 *
 * ============================ THE RECIPE ============================
 * Layout: 128-BYTE SWIZZLE, K-major, trans_a = trans_b = 0.
 *   logical tile is row-major [rows][BK]; physical offset of the 16-byte chunk
 *   c (= k/8 for bf16, k/16 for e4m3) of logical row r is
 *       sm90_swz_off(r, c, BK) = r*BK + ((c ^ (r & 7)) * ELEMS_PER_CHUNK)
 *   One logical row must be exactly 128 B (BK=64 bf16, or BK=128 e4m3), which is
 *   one swizzle atom row.
 * Descriptor: LBO = 16 B, SBO = 1024 B, swizzle bits[63:62] = 1, matrix base
 *   offset 0.
 *   - LBO/SBO are encoded as "byte offset with the low 4 bits dropped"
 *     (i.e. >>4). Shifting them again is the classic way to get silent garbage.
 *   - A k16 (bf16) or k32 (fp8) substep advances the START ADDRESS ONLY by
 *     +32 B. LBO/SBO never change. (The no-swizzle layout advances by a full
 *     +256 B K-core stride instead — do not mix the two recipes.)
 * ALIGNMENT: the tile base MUST be 1024-byte aligned, or the hardware address
 *   swizzle disagrees with the store-side XOR and results are silently WRONG
 *   (it does not fault). extern __shared__ does not give you this — round up.
 *
 * SWIZZLE IS A PERFORMANCE REQUIREMENT, NOT A CORRECTNESS ONE. A swizzle-free
 * core-matrix layout passes an oracle at identical relL2 but runs ~2.2x slower
 * (84 vs 177 TF/s dense; 73 vs 162 TF/s MoE): its 1024 B row-core stride puts
 * every row-core on one bank, giving 8-way conflicts on BOTH the cp.async store
 * and the wgmma operand read.
 *
 * ACCUMULATOR -> (row, col), verified against the PTX ISA m64nN fragment layout
 * and identical for bf16 k16 and e4m3 k32:
 *     warp w (0..3 within the warpgroup) owns rows [16w, 16w+16)
 *     for n-block g:  reg r = 4g + 2*hi + lo
 *                     row   = 16w + lane/4 + 8*hi
 *                     col   =  8g + 2*(lane%4) + lo
 * This is the SAME per-8-column pattern as the existing m16n8k16 C fragment, so
 * an existing store/activation epilogue ports over unchanged.
 *
 * scale-d is passed as a PTX predicate: 0 => D = A*B (seeds the accumulators, so
 * no zeroing pass is needed on the first k-step), 1 => D = A*B + D.
 *
 * FP8 NOTE: Hopper has NO native fp8 mma.sync — mma.sync.m16n8k32.e4m3 lowers to
 * 12x F2FP + 2x HMMA (emulated via f16) on sm_90a, vs 1x QMMA on sm_120a. wgmma
 * is the ONLY route to the fp8 tensor core here. Also, fp8 wgmma does NOT
 * accumulate in true f32: error grows with K (3.9e-5 @K=32 -> 1.14e-3 @K=3840).
 * Promote the accumulator into an f32 shadow every 128 k-elements (DeepGEMM
 * two-level accumulation) to cut it ~10.9x for 8-16% throughput.
 */
#ifndef PLOW_SM90_WGMMA_CUH
#define PLOW_SM90_WGMMA_CUH

#if !defined(PLOW_NV_HOPPER)
#error "sm90_wgmma.cuh is Hopper-only; include it under PLOW_NV_HOPPER"
#endif

#include <cuda_bf16.h>
#include <cstdint>

/* ---- smem matrix descriptor -------------------------------------------------
 * bits[13:0] start>>4, [29:16] LBO>>4, [45:32] SBO>>4, [51:49] matrix base
 * offset, [63:62] swizzle mode (1 = 128 B). */
#define PLOW_SM90_LBO 16ull   /* bytes */
#define PLOW_SM90_SBO 1024ull /* bytes: 8 rows x 128 B swizzle atom */
#define PLOW_SM90_SWZ 1ull    /* 128-byte swizzle */

__device__ __forceinline__ uint64_t sm90_desc_enc(uint64_t x) { return (x & 0x3FFFFull) >> 4; }

__device__ __forceinline__ uint64_t sm90_make_desc(const void* ptr, uint64_t lbo, uint64_t sbo) {
    uint64_t a = (uint64_t)__cvta_generic_to_shared(ptr);
    uint64_t d = 0;
    d |= sm90_desc_enc(a);
    d |= sm90_desc_enc(lbo) << 16;
    d |= sm90_desc_enc(sbo) << 32;
    d |= PLOW_SM90_SWZ << 62; /* matrix base offset 0: tile is 1024 B aligned */
    return d;
}
/* The standard 128B-swizzle descriptor for a tile whose rows are exactly 128 B. */
__device__ __forceinline__ uint64_t sm90_desc(const void* ptr) {
    return sm90_make_desc(ptr, PLOW_SM90_LBO, PLOW_SM90_SBO);
}

/* 128B-swizzled element offset of 16-byte chunk `c` of logical row `row`.
 * EPC = elements per 16-byte chunk (8 for bf16, 16 for e4m3). BK = elements/row. */
template <int BK, int EPC>
__device__ __forceinline__ int sm90_swz_off(int row, int c) {
    return row * BK + ((c ^ (row & 7)) * EPC);
}

/* Round a dynamic-smem pointer up to the 1024 B alignment the swizzle requires.
 * generic->shared is affine, so adding N bytes to the generic pointer adds N to
 * the shared address; that is what makes this legal. */
__device__ __forceinline__ void* sm90_align1024(void* p) {
    uint64_t s = (uint64_t)__cvta_generic_to_shared(p);
    uint64_t pad = (1024ull - (s & 1023ull)) & 1023ull;
    return (void*)((char*)p + pad);
}

/* ---- staging + pipeline sync ---------------------------------------------- */
__device__ __forceinline__ void sm90_cp16(void* smem, const void* gmem, int src_bytes) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem),
                 "r"(src_bytes));
}
__device__ __forceinline__ void sm90_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void sm90_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void sm90_wg_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void sm90_wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N> __device__ __forceinline__ void sm90_wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

/* ---- the MMAs (verbatim from the oracle-validated probes) ------------------ */
/* m64n128k16 .f32.bf16.bf16, both operands from smem (SS form), 64 f32 acc/thread. */
__device__ __forceinline__ void wgmma_m64n128k16(float* d, uint64_t da, uint64_t db,
                                                 int scaleD) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,"
        "%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
        "%64, %65, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
          "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
          "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
          "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
          "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]),
          "+f"(d[63])
        : "l"(da), "l"(db), "r"(scaleD));
}

/* m64n128k32 .f32.e4m3.e4m3, both operands from smem (SS form), 64 f32 acc/thread.
 * fp8 is K-MAJOR ONLY (no transpose immediates), which matches plow's TN contract. */
__device__ __forceinline__ void wgmma_m64n128k32(float* d, uint64_t desc_a, uint64_t desc_b, int scale_d) {
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7,"
      " %8, %9, %10, %11, %12, %13, %14, %15,"
      " %16, %17, %18, %19, %20, %21, %22, %23,"
      " %24, %25, %26, %27, %28, %29, %30, %31,"
      " %32, %33, %34, %35, %36, %37, %38, %39,"
      " %40, %41, %42, %43, %44, %45, %46, %47,"
      " %48, %49, %50, %51, %52, %53, %54, %55,"
      " %56, %57, %58, %59, %60, %61, %62, %63"
      "}, %64, %65, p, 1, 1;\n"
      "}\n"
      :
        "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}


/* m64n256k16 .f32.bf16.bf16 (SS form), 128 f32 acc/thread — the bf16 twin of the
 * m64n256k32 e4m3 shape below (T20b): same 128 B swizzle staging (64 bf16 = 128 B rows),
 * same two-box B tile, k16 substeps. */
__device__ __forceinline__ void wgmma_m64n256k16(float* d, uint64_t desc_a, uint64_t desc_b, int scale_d) {
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %130, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{"
      " %0, %1, %2, %3, %4, %5, %6, %7,"
      " %8, %9, %10, %11, %12, %13, %14, %15,"
      " %16, %17, %18, %19, %20, %21, %22, %23,"
      " %24, %25, %26, %27, %28, %29, %30, %31,"
      " %32, %33, %34, %35, %36, %37, %38, %39,"
      " %40, %41, %42, %43, %44, %45, %46, %47,"
      " %48, %49, %50, %51, %52, %53, %54, %55,"
      " %56, %57, %58, %59, %60, %61, %62, %63,"
      " %64, %65, %66, %67, %68, %69, %70, %71,"
      " %72, %73, %74, %75, %76, %77, %78, %79,"
      " %80, %81, %82, %83, %84, %85, %86, %87,"
      " %88, %89, %90, %91, %92, %93, %94, %95,"
      " %96, %97, %98, %99, %100, %101, %102, %103,"
      " %104, %105, %106, %107, %108, %109, %110, %111,"
      " %112, %113, %114, %115, %116, %117, %118, %119,"
      " %120, %121, %122, %123, %124, %125, %126, %127"
      "}, %128, %129, p, 1, 1, 0, 0;\n"
      "}\n"
      :
        "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
        "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
        "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
        "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
        "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
        "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
        "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
        "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
        "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}

/* m64n256k32 .f32.e4m3.e4m3 (SS form), 128 f32 acc/thread — the full-rate fp8 N-extent
 * (T13). B tile = 256 rows x 128 B staged as two contiguous 128-row TMA boxes; the 128B
 * swizzle's 1024 B core-matrix stride continues across the boundary, so one descriptor
 * covers the whole 32 KiB tile. */
__device__ __forceinline__ void wgmma_m64n256k32(float* d, uint64_t desc_a, uint64_t desc_b, int scale_d) {
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %130, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3 "
      "{"
      " %0, %1, %2, %3, %4, %5, %6, %7,"
      " %8, %9, %10, %11, %12, %13, %14, %15,"
      " %16, %17, %18, %19, %20, %21, %22, %23,"
      " %24, %25, %26, %27, %28, %29, %30, %31,"
      " %32, %33, %34, %35, %36, %37, %38, %39,"
      " %40, %41, %42, %43, %44, %45, %46, %47,"
      " %48, %49, %50, %51, %52, %53, %54, %55,"
      " %56, %57, %58, %59, %60, %61, %62, %63,"
      " %64, %65, %66, %67, %68, %69, %70, %71,"
      " %72, %73, %74, %75, %76, %77, %78, %79,"
      " %80, %81, %82, %83, %84, %85, %86, %87,"
      " %88, %89, %90, %91, %92, %93, %94, %95,"
      " %96, %97, %98, %99, %100, %101, %102, %103,"
      " %104, %105, %106, %107, %108, %109, %110, %111,"
      " %112, %113, %114, %115, %116, %117, %118, %119,"
      " %120, %121, %122, %123, %124, %125, %126, %127"
      "}, %128, %129, p, 1, 1;\n"
      "}\n"
      :
        "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
        "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
        "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
        "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
        "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
        "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
        "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
        "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
        "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}

/* ---- K precondition guard for the 128B-swizzle stagers --------------------
 * The swizzled staging issues 16-byte cp.async chunks from `src + row*K + gk`.
 * That global address is 16-byte aligned only when K is a multiple of the
 * per-chunk element count (8 bf16 / 16 e4m3); with K not a multiple, odd rows
 * misalign and the cp.async is undefined behaviour. k==0 additionally leaves
 * the accumulators unseeded (the mainloop seeds via the first wgmma's
 * scale-d=0). Both are inputs the emitter never produces for this path (every
 * Gemma projection K is a multiple of 128), so this is a defensive trap in the
 * same spirit as the dispatch's `default: __trap()`: fail loud, never compute
 * on a layout the swizzle cannot represent. Uniform over the block (k is
 * uniform), so the early return is counter-protocol-safe. Returns true if the
 * caller should bail. */
__device__ __forceinline__ bool sm90_bad_k(unsigned k, unsigned chunk_elems) {
    if (k != 0u && (k % chunk_elems) == 0u) return false;
    if (threadIdx.x == 0) __trap();
    return true;
}

/* ---- TMA + mbarrier primitives (lifted verbatim from tma_ws_gemm_bf16.cu) --
 * Used only by the PLOW_NV_TMA_GEMM warp-specialized mainloop (op_gemm_sm90.cuh).
 * The tensor map is an OPAQUE 128-byte, 128-byte-aligned blob encoded on the HOST
 * (cuTensorMapEncodeTiled) and reached by generic pointer — typed void* here so the
 * cubin TU does not need cuda.h. CU_TENSOR_MAP_SWIZZLE_128B with boxDim[0]*2B = 128 B
 * writes EXACTLY the 128B-swizzled layout the descriptors above describe, so the wgmma
 * side is unchanged (the store-side XOR just moves into the copy engine). */
__device__ __forceinline__ uint32_t sm90_su32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void sm90_mbar_init(uint64_t* b, int cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(sm90_su32(b)), "r"(cnt)
                 : "memory");
}
/* Invalidate before the arena is reused as plain data by the next op (PTX requires it;
 * cheap: one instruction per barrier per op). */
__device__ __forceinline__ void sm90_mbar_inval(uint64_t* b) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(sm90_su32(b)) : "memory");
}
__device__ __forceinline__ void sm90_mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(sm90_su32(b)) : "memory");
}
/* arrive + declare the transaction byte count this phase must also collect */
__device__ __forceinline__ void sm90_mbar_expect(uint64_t* b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(sm90_su32(b)),
                 "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void sm90_mbar_wait(uint64_t* b, int parity) {
    asm volatile("{\n.reg .pred p;\nTW%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra TW%=;\n}\n" ::"r"(sm90_su32(b)), "r"(parity)
                 : "memory");
}
/* TMA 2-D tile load, single destination CTA. c0 = inner (K) coord, c1 = row coord. */
__device__ __forceinline__ void sm90_tma2d(uint32_t dst, const void* map, int c0, int c1,
                                           uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst),
                 "l"(map), "r"(c0), "r"(c1), "r"(bar)
                 : "memory");
}
/* TMA 3-D tile load: c0 = innermost (elem) coord, c1/c2 outer. Used by the flash-prefill
 * KV stager: map {hd, ring, n_kv_head}, box {64, BKV, 1}. */
__device__ __forceinline__ void sm90_tma3d(uint32_t dst, const void* map, int c0, int c1, int c2,
                                           uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(dst),
                 "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
                 : "memory");
}
/* The map lives in ordinary global memory; if a device-side tensormap.replace ever wrote
 * it, the TMA read needs an acquire in the tensormap proxy. Host-written maps are covered
 * by launch ordering, but the fence is one instruction per op — keep the probe's exact
 * validated configuration (k_tma_gmem) rather than reasoning it away. */
__device__ __forceinline__ void sm90_tmap_acquire(const void* map) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n" ::"l"(map) : "memory");
}

/* ---- accumulator -> (row, col) --------------------------------------------
 * See the header comment. `g` is the n-block (0..N/8-1), `hi`/`lo` in {0,1}. */
__device__ __forceinline__ int sm90_acc_reg(int g, int hi, int lo) { return 4 * g + 2 * hi + lo; }
__device__ __forceinline__ int sm90_acc_row(int warp_in_wg, int lane, int hi) {
    return 16 * warp_in_wg + (lane >> 2) + 8 * hi;
}
__device__ __forceinline__ int sm90_acc_col(int g, int lane, int lo) {
    return 8 * g + 2 * (lane & 3) + lo;
}

#endif /* PLOW_SM90_WGMMA_CUH */
