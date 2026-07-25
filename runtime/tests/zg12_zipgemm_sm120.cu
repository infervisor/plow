/* zg12_zipgemm_sm120.cu — ZG-1 (offline TCA packer + host/fragment oracles) + ZG-2 (fused
 * register-direct ZipGEMM kernel `k_zipgemm`).
 *
 * Skeleton = the ZG-0 tuned TC bf16 GEMM (split-K / BK64 / STAGES4, runtime/tests/
 * zg0_tc_stream_sm120.cu). The ONLY change vs ZG-0: the weight B is streamed COMPRESSED
 * (sz12 byte-plane split, 1.5 B/elem) in tensor-core FRAGMENT-LANE order, and each lane
 * reconstructs its own mma B-fragment register->register via V3 `recon_pair`. No bf16 smem
 * tile, no ldmatrix for B, one sync.
 *
 * The offline packer AND the kernel SHARE ONE lane derivation (`zg_lo_byte`/`zg_cd_byte`),
 * so the fragment order is correct-by-construction. Derivation taken from the validated plow
 * path (op_gemm.cuh pgm_load_bfrags: n=lane>>2, b0={k=2*tig,+1}, b1={k=2*tig+8,+9}), and
 * empirically checked against actual ldmatrix output (fragment oracle, gate 2).
 *
 * Gates:
 *   1. Host lossless byte-exact: pack->unpack == original bf16 bytes (every tensor).
 *   2. Fragment oracle: recon(baked TCA bytes) == ldmatrix(uncompressed tile), per lane.
 *   3. Device bit-exact: k_zipgemm(ksplit=1) output == ZG-0 k_tc(ksplit=1), byte-identical.
 * Perf: k_zipgemm GB/s (logical bf16 bytes / time) vs the ZG-0 tuned TC-bf16 baseline.
 *
 * Build (SYSTEM toolchain): nvcc -std=c++17 -O3 -arch=sm_120a -Xptxas -v
 *   -Iinclude -Iruntime/common -Iruntime/nvidia runtime/tests/zg12_zipgemm_sm120.cu -o /tmp/zg12
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <functional>
#include <cuda_runtime.h>
#include "sm120_common.cuh"
#include "op_gemm.cuh"

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static const unsigned ZG_EXP_BASE = 109u;  /* audited sz12 window [109,124] */

/* ================================================================== SHARED TCA LANE LAYOUT
 * The one derivation used by BOTH the offline packer and the kernel. For a [BN][BK] B-block:
 *   TensorCoreTile = 16(k) x 8(n); 32 lanes each own 4 elements.
 *   lane = (nt%8)*4 + tig ;  tig picks the k-pair, groupID=nt%8 picks the n-column.
 *   per-lane lo = 4 contiguous bytes [k=2tig, 2tig+1, 2tig+8, 2tig+9]  -> one coalesced u32
 *   per-lane cd = 2 contiguous bytes [c0 = codes(2tig,2tig+1), c1 = codes(2tig+8,2tig+9)]
 *   TensorCoreTile order within the block: tct = (kt/16)*(BN/8) + (nt/8).
 * lo block = BN*BK bytes, cd block = BN*BK/2 bytes (both 16B-aligned for our shapes). */
__host__ __device__ __forceinline__ int zg_lo_byte(int nt, int kt, int BN) {
    const int nfrag = nt >> 3, ncol = nt & 7;
    const int kstep = kt >> 4, ksub = kt & 15;
    const int half = ksub >> 3, r = ksub & 7, tig = r >> 1, s = r & 1;
    const int slot = half * 2 + s;
    const int lane = ncol * 4 + tig;
    const int tct = kstep * (BN >> 3) + nfrag;
    return tct * 128 + lane * 4 + slot;
}
/* returns byte index into the cd block; *nib = which nibble (0=low,1=high) */
__host__ __device__ __forceinline__ int zg_cd_byte(int nt, int kt, int BN, int* nib) {
    const int nfrag = nt >> 3, ncol = nt & 7;
    const int kstep = kt >> 4, ksub = kt & 15;
    const int half = ksub >> 3, r = ksub & 7, tig = r >> 1, s = r & 1;
    const int lane = ncol * 4 + tig;
    const int tct = kstep * (BN >> 3) + nfrag;
    *nib = s;
    return tct * 64 + lane * 2 + half;
}

/* V3 recon: 2 lo bytes (@b0,b1) + 1 code byte (2 nibbles) -> 2 packed bf16 (u32). Bit-exact. */
__device__ __forceinline__ unsigned recon_pair(unsigned L, unsigned cbyte, unsigned baseK) {
    const unsigned spread = __byte_perm(L, 0u, 0x4140u);           /* lo0@b0, lo1@b2 */
    const unsigned b16 = (spread & 0x007F007Fu) | ((spread << 8) & 0x80008000u);
    const unsigned e = (cbyte & 0xFu) | ((cbyte & 0xF0u) << 12);   /* c0@b0, c1@b2 */
    return b16 | ((e + baseK) << 7);
}

/* ================================================================== HOST PACKER + ORACLE */
struct Comp {
    std::vector<uint8_t>  lo, cd;        /* TCA fragment-lane order, per (n_block,k_block) */
    std::vector<unsigned> eoff, epos;    /* per-row escape prefix + absolute elem index */
    std::vector<uint16_t> eval;          /* raw bf16 bits of each escape */
    unsigned nesc = 0;
    int BN = 0, BK = 0;
};

/* pack [N][K] bf16 (u16 bits) into TCA planes for the (BN,BK) kernel geometry. */
static Comp zg_pack(const uint16_t* s, unsigned N, unsigned K, int BN, int BK) {
    Comp c; c.BN = BN; c.BK = BK;
    const size_t NK = (size_t)N * K;
    c.lo.assign(NK, 0); c.cd.assign(NK / 2, 0);
    c.eoff.assign(N + 1, 0);
    const int kb = (int)K / BK;
    for (unsigned n = 0; n < N; ++n) {
        for (unsigned k = 0; k < K; ++k) {
            const uint16_t u = s[(size_t)n * K + k];
            const unsigned ex = (u >> 7) & 0xFF;
            const uint8_t lob = (uint8_t)(((u >> 8) & 0x80) | (u & 0x7F));
            const bool esc = !(ex >= ZG_EXP_BASE && ex <= ZG_EXP_BASE + 15);
            const int code = esc ? 0 : (int)(ex - ZG_EXP_BASE);
            const int nblk = n / BN, kblk = k / BK, nt = n % BN, kt = k % BK;
            const size_t base_lo = ((size_t)nblk * kb + kblk) * (size_t)(BN * BK);
            const size_t base_cd = ((size_t)nblk * kb + kblk) * (size_t)(BN * BK / 2);
            c.lo[base_lo + zg_lo_byte(nt, kt, BN)] = lob;
            int nib; const int cbi = zg_cd_byte(nt, kt, BN, &nib);
            c.cd[base_cd + cbi] |= (uint8_t)(code << (nib * 4));
            if (esc) { c.epos.push_back((unsigned)((size_t)n * K + k)); c.eval.push_back(u);
                       c.eoff[n]++; c.nesc++; }
        }
    }
    unsigned run = 0;                    /* prefix-sum eoff */
    for (unsigned n = 0; n <= N; ++n) { unsigned t = (n < N) ? c.eoff[n] : 0; c.eoff[n] = run; run += t; }
    return c;
}

/* host V3 recon of one code+lo (mirror of recon_pair for a single element) */
static inline uint16_t zg_host_recon(uint8_t lob, int code) {
    const unsigned sign = (lob & 0x80u) << 8;
    const unsigned mant = lob & 0x7Fu;
    const unsigned ex = (unsigned)code + ZG_EXP_BASE;
    return (uint16_t)(sign | (ex << 7) | mant);
}

/* ORACLE: unpack TCA planes back to [N][K] bf16 and memcmp vs original. Returns #mismatches. */
static size_t zg_unpack_check(const Comp& c, const uint16_t* orig, unsigned N, unsigned K) {
    const int BN = c.BN, BK = c.BK, kb = (int)K / BK;
    std::vector<uint16_t> out((size_t)N * K);
    for (unsigned n = 0; n < N; ++n) {
        for (unsigned k = 0; k < K; ++k) {
            const int nblk = n / BN, kblk = k / BK, nt = n % BN, kt = k % BK;
            const size_t base_lo = ((size_t)nblk * kb + kblk) * (size_t)(BN * BK);
            const size_t base_cd = ((size_t)nblk * kb + kblk) * (size_t)(BN * BK / 2);
            const uint8_t lob = c.lo[base_lo + zg_lo_byte(nt, kt, BN)];
            int nib; const int cbi = zg_cd_byte(nt, kt, BN, &nib);
            const int code = (c.cd[base_cd + cbi] >> (nib * 4)) & 0xF;
            out[(size_t)n * K + k] = zg_host_recon(lob, code);
        }
    }
    for (size_t t = 0; t < c.epos.size(); ++t) out[c.epos[t]] = c.eval[t];   /* patch escapes */
    size_t bad = 0;
    for (size_t i = 0; i < (size_t)N * K; ++i) if (out[i] != orig[i]) bad++;
    return bad;
}

/* ================================================================== ZG-0 baseline kernels (verbatim) */
__global__ void k_stream_reduce(const __nv_bfloat16* __restrict__ B, size_t nelem8, float* sink) {
    float acc = 0.f; const uint4* p = (const uint4*)B;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nelem8;
         i += (size_t)gridDim.x * blockDim.x) { uint4 v = p[i]; acc += (float)(v.x ^ v.y ^ v.z ^ v.w); }
    __shared__ float red[32]; float w = acc;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) w += __shfl_down_sync(0xffffffffu, w, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = w;
    __syncthreads();
    if (threadIdx.x == 0) { float s = 0.f; for (int i = 0; i < (int)(blockDim.x >> 5); i++) s += red[i]; atomicAdd(sink, s); }
}

/* Tuned TC bf16 GEMM (the ZG-0 baseline). C[M,N]=A[M,K].B[N,K]^T, split-K into f32 Cf. */
template <int BM, int BN, int BK, int NW, int STAGES>
__global__ void __launch_bounds__(NW * 32) k_tc(float* __restrict__ Cf,
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
        unsigned M, unsigned N, unsigned K, int ksplit) {
    constexpr int MFRAG = BM / 16, WN = BN / NW, NFRAG = WN / 8;
    constexpr int AS = BK + 8, BKS = BK + 8, ABUF = BM * AS, BBUF = BN * BKS, KCH = BK / 8;
    extern __shared__ char sm[];
    __nv_bfloat16* As = (__nv_bfloat16*)sm; __nv_bfloat16* Bs = As + STAGES * ABUF;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tiles_n = (N + BN - 1) / BN, totksteps = (K + BK - 1) / BK;
    const int kper = (totksteps + ksplit - 1) / ksplit, njob = tiles_n * ksplit;
    for (int job = blockIdx.x; job < njob; job += gridDim.x) {
        const int nt = job / ksplit, ksp = job % ksplit, tn = nt * BN;
        const int ks0 = ksp * kper, ks1 = (ks0 + kper < totksteps) ? (ks0 + kper) : totksteps;
        if (ks0 >= ks1) continue;
        float acc[MFRAG][NFRAG][4];
#pragma unroll
        for (int i = 0; i < MFRAG; i++) for (int j = 0; j < NFRAG; j++) for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
        auto stage = [&](int ks, int buf) {
#pragma unroll
            for (int L = tid; L < BM * KCH; L += NW * 32) { const int row = L / KCH, kk8 = (L % KCH) * 8;
                const int mm = row, kk = ks * BK + kk8; const bool in = (mm < (int)M) && (kk + 8 <= (int)K);
                const __nv_bfloat16* g = in ? A + (size_t)mm * K + kk : A;
                pgm_cp_async_cg16(&As[buf * ABUF + row * AS + kk8], g, in ? 16 : 0); }
#pragma unroll
            for (int L = tid; L < BN * KCH; L += NW * 32) { const int row = L / KCH, kk8 = (L % KCH) * 8;
                const int nn = tn + row, kk = ks * BK + kk8; const bool in = (nn < (int)N) && (kk + 8 <= (int)K);
                const __nv_bfloat16* g = in ? B + (size_t)nn * K + kk : B;
                pgm_cp_async_cg16(&Bs[buf * BBUF + row * BKS + kk8], g, in ? 16 : 0); }
        };
        const int nks = ks1 - ks0;
#pragma unroll 1
        for (int s = 0; s < STAGES - 1; s++) { if (s < nks) stage(ks0 + s, s); pgm_cp_commit(); }
        for (int i = 0; i < nks; i++) {
            const int fetch = i + STAGES - 1; if (fetch < nks) stage(ks0 + fetch, fetch % STAGES);
            pgm_cp_commit(); pgm_cp_wait<STAGES - 1>(); __syncthreads();
            const int cb = i % STAGES; __nv_bfloat16* Ad = As + cb * ABUF; __nv_bfloat16* Bd = Bs + cb * BBUF;
#pragma unroll
            for (int kf = 0; kf < BK; kf += 16) {
                unsigned af[MFRAG][4];
#pragma unroll
                for (int mi = 0; mi < MFRAG; mi++) { const int arow = mi * 16 + (lane % 16), acol = kf + (lane / 16) * 8;
                    pgm_ldmatrix_x4(af[mi], &Ad[arow * AS + acol]); }
                unsigned bf[NFRAG][2];
#pragma unroll
                for (int nj = 0; nj < NFRAG; nj++) { const int nrow = warp * WN + nj * 8 + (lane & 7);
                    const int kcol = kf + ((lane >> 3) & 1) * 8; pgm_ldmatrix_x2(bf[nj], &Bd[nrow * BKS + kcol]); }
#pragma unroll
                for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < NFRAG; nj++) pgm_mma(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < NFRAG; nj++) { const int gr = lane / 4, gc = warp * WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) { const int rr = mi * 16 + gr + (e / 2) * 8, cc = tn + gc + (e % 2);
                    if (rr < (int)M && cc < (int)N) atomicAdd(&Cf[(size_t)rr * N + cc], acc[mi][nj][e]); } }
        __syncthreads();
    }
}

/* ================================================================== ZG-2 fused k_zipgemm */
/* Same skeleton as k_tc; B replaced by compressed TCA planes lo/cd + escapes, reconstructed
 * register->register into the mma B-fragment. */
template <int BM, int BN, int BK, int NW, int STAGES>
__global__ void __launch_bounds__(NW * 32) k_zipgemm(float* __restrict__ Cf,
        const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ lo,
        const uint8_t* __restrict__ cd, const unsigned* __restrict__ eoff,
        const unsigned* __restrict__ epos, const uint16_t* __restrict__ eval,
        unsigned M, unsigned N, unsigned K, int ksplit) {
    constexpr int MFRAG = BM / 16, WN = BN / NW, NFRAG = WN / 8;
    constexpr int AS = BK + 8, ABUF = BM * AS, KCH = BK / 8;
    constexpr int LOBLK = BN * BK, CDBLK = BN * BK / 2;   /* bytes per k-block */
    extern __shared__ char sm[];
    __nv_bfloat16* As = (__nv_bfloat16*)sm;
    uint8_t* los = (uint8_t*)(As + STAGES * ABUF);
    uint8_t* cds = los + STAGES * LOBLK;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int groupID = lane >> 2, tig = lane & 3;
    const unsigned baseK = ZG_EXP_BASE | (ZG_EXP_BASE << 16);
    const int tiles_n = (N + BN - 1) / BN, totksteps = (K + BK - 1) / BK, kb = (int)K / BK;
    const int kper = (totksteps + ksplit - 1) / ksplit, njob = tiles_n * ksplit;
    for (int job = blockIdx.x; job < njob; job += gridDim.x) {
        const int nt = job / ksplit, ksp = job % ksplit, tn = nt * BN, nblk = tn / BN;
        const int ks0 = ksp * kper, ks1 = (ks0 + kper < totksteps) ? (ks0 + kper) : totksteps;
        if (ks0 >= ks1) continue;
        float acc[MFRAG][NFRAG][4];
#pragma unroll
        for (int i = 0; i < MFRAG; i++) for (int j = 0; j < NFRAG; j++) for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
        /* per-nj n-row + escape range (independent of ks/kf) */
        int nnj[NFRAG], e0[NFRAG], e1[NFRAG];
#pragma unroll
        for (int nj = 0; nj < NFRAG; nj++) { const int n = tn + warp * WN + nj * 8 + groupID;
            nnj[nj] = n; if (n < (int)N) { e0[nj] = eoff[n]; e1[nj] = eoff[n + 1]; } else { e0[nj] = e1[nj] = 0; } }
        auto stage = [&](int ks, int buf) {
#pragma unroll
            for (int L = tid; L < BM * KCH; L += NW * 32) { const int row = L / KCH, kk8 = (L % KCH) * 8;
                const int mm = row, kk = ks * BK + kk8; const bool in = (mm < (int)M) && (kk + 8 <= (int)K);
                const __nv_bfloat16* g = in ? A + (size_t)mm * K + kk : A;
                pgm_cp_async_cg16(&As[buf * ABUF + row * AS + kk8], g, in ? 16 : 0); }
            const size_t blo = ((size_t)nblk * kb + ks) * (size_t)LOBLK;
            const size_t bcd = ((size_t)nblk * kb + ks) * (size_t)CDBLK;
            uint8_t* dlo = los + buf * LOBLK; uint8_t* dcd = cds + buf * CDBLK;
            for (int o = tid * 16; o < LOBLK; o += NW * 32 * 16) pgm_cp_async_cg16(dlo + o, lo + blo + o, 16);
            for (int o = tid * 16; o < CDBLK; o += NW * 32 * 16) pgm_cp_async_cg16(dcd + o, cd + bcd + o, 16);
        };
        const int nks = ks1 - ks0;
#pragma unroll 1
        for (int s = 0; s < STAGES - 1; s++) { if (s < nks) stage(ks0 + s, s); pgm_cp_commit(); }
        for (int i = 0; i < nks; i++) {
            const int fetch = i + STAGES - 1; if (fetch < nks) stage(ks0 + fetch, fetch % STAGES);
            pgm_cp_commit(); pgm_cp_wait<STAGES - 1>(); __syncthreads();
            const int cb = i % STAGES; const int ks = ks0 + i;
            __nv_bfloat16* Ad = As + cb * ABUF;
            uint8_t* lob = los + cb * LOBLK; uint8_t* cdb = cds + cb * CDBLK;
#pragma unroll
            for (int kf = 0; kf < BK; kf += 16) {
                unsigned af[MFRAG][4];
#pragma unroll
                for (int mi = 0; mi < MFRAG; mi++) { const int arow = mi * 16 + (lane % 16), acol = kf + (lane / 16) * 8;
                    pgm_ldmatrix_x4(af[mi], &Ad[arow * AS + acol]); }
                unsigned bf[NFRAG][2];
#pragma unroll
                for (int nj = 0; nj < NFRAG; nj++) {
                    const int tct = (kf >> 4) * (BN >> 3) + (warp * NFRAG + nj);
                    const unsigned word = *(const unsigned*)(lob + tct * 128 + lane * 4);
                    const unsigned cdw  = *(const unsigned short*)(cdb + tct * 64 + lane * 2);
                    bf[nj][0] = recon_pair(word & 0xFFFFu, cdw & 0xFFu, baseK);
                    bf[nj][1] = recon_pair(word >> 16, cdw >> 8, baseK);
                    if (e1[nj] != e0[nj]) {                          /* rare: escape patch */
                        const size_t base_el = (size_t)nnj[nj] * K + (size_t)ks * BK + kf + tig * 2;
                        for (int t = e0[nj]; t < e1[nj]; ++t) { long d = (long)epos[t] - (long)base_el;
                            unsigned ev = eval[t];
                            if (d == 0) bf[nj][0] = (bf[nj][0] & 0xFFFF0000u) | ev;
                            else if (d == 1) bf[nj][0] = (bf[nj][0] & 0xFFFFu) | (ev << 16);
                            else if (d == 8) bf[nj][1] = (bf[nj][1] & 0xFFFF0000u) | ev;
                            else if (d == 9) bf[nj][1] = (bf[nj][1] & 0xFFFFu) | (ev << 16); }
                    }
                }
#pragma unroll
                for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < NFRAG; nj++) pgm_mma(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < NFRAG; nj++) { const int gr = lane / 4, gc = warp * WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) { const int rr = mi * 16 + gr + (e / 2) * 8, cc = tn + gc + (e % 2);
                    if (rr < (int)M && cc < (int)N) atomicAdd(&Cf[(size_t)rr * N + cc], acc[mi][nj][e]); } }
        __syncthreads();
    }
}

/* ================================================================== fragment oracle kernel (gate 2)
 * For one B-block [BN][BK] (ks=0), write per (warp,nj,kf,lane,reg) the bf-fragment from BOTH
 * ldmatrix (uncompressed bf16 smem tile) and recon (TCA planes). Host compares. Layout of out:
 * index = ((((warp*NFRAG+nj)*(BK/16)+kf/16)*32+lane)*2+reg). */
template <int BN, int BK, int NW>
__global__ void __launch_bounds__(NW * 32) k_frag_oracle(unsigned* out_ld, unsigned* out_rc,
        const __nv_bfloat16* __restrict__ B, const uint8_t* __restrict__ lo,
        const uint8_t* __restrict__ cd, const unsigned* __restrict__ eoff,
        const unsigned* __restrict__ epos, const uint16_t* __restrict__ eval, unsigned N, unsigned K) {
    constexpr int WN = BN / NW, NFRAG = WN / 8, BKS = BK + 8;
    extern __shared__ char sm[];
    __nv_bfloat16* Bs = (__nv_bfloat16*)sm;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int groupID = lane >> 2, tig = lane & 3;
    const unsigned baseK = ZG_EXP_BASE | (ZG_EXP_BASE << 16);
    const int KCH = BK / 8;
    for (int L = tid; L < BN * KCH; L += NW * 32) { const int row = L / KCH, kk8 = (L % KCH) * 8;
        for (int j = 0; j < 8; j++) Bs[row * BKS + kk8 + j] = B[(size_t)row * K + kk8 + j]; }
    __syncthreads();
    for (int kf = 0; kf < BK; kf += 16) {
        for (int nj = 0; nj < NFRAG; nj++) {
            unsigned ld[2]; { const int nrow = warp * WN + nj * 8 + (lane & 7);
                const int kcol = kf + ((lane >> 3) & 1) * 8; pgm_ldmatrix_x2(ld, &Bs[nrow * BKS + kcol]); }
            const int tct = (kf >> 4) * (BN >> 3) + (warp * NFRAG + nj);
            const unsigned word = *(const unsigned*)(lo + tct * 128 + lane * 4);
            const unsigned cdw  = *(const unsigned short*)(cd + tct * 64 + lane * 2);
            unsigned rc[2]; rc[0] = recon_pair(word & 0xFFFFu, cdw & 0xFFu, baseK);
            rc[1] = recon_pair(word >> 16, cdw >> 8, baseK);
            const int n = warp * WN + nj * 8 + groupID;             /* escape patch (full kernel path) */
            const unsigned e0 = eoff[n], e1 = eoff[n + 1];
            if (e1 != e0) { const size_t base_el = (size_t)n * K + kf + tig * 2;
                for (unsigned t = e0; t < e1; ++t) { long d = (long)epos[t] - (long)base_el; unsigned ev = eval[t];
                    if (d == 0) rc[0] = (rc[0] & 0xFFFF0000u) | ev; else if (d == 1) rc[0] = (rc[0] & 0xFFFFu) | (ev << 16);
                    else if (d == 8) rc[1] = (rc[1] & 0xFFFF0000u) | ev; else if (d == 9) rc[1] = (rc[1] & 0xFFFFu) | (ev << 16); } }
            const int base = (((warp * NFRAG + nj) * (BK / 16) + kf / 16) * 32 + lane) * 2;
            out_ld[base + 0] = ld[0]; out_ld[base + 1] = ld[1];
            out_rc[base + 0] = rc[0]; out_rc[base + 1] = rc[1];
        }
    }
}

__global__ void k_f32_to_bf16(__nv_bfloat16* C, const float* Cf, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) C[i] = __float2bfloat16(Cf[i]);
}

/* ================================================================== host harness */
static int g_sm = 188;
struct Shape { const char* name; unsigned N, K; };

/* Gaussian bf16 weights: std chosen so exponents land in the sz12 window [109,124] with a
 * realistic ~0.02% escape tail (matches the c0 audit). Deterministic per seed. */
static void fill_weights(std::vector<uint16_t>& v, uint64_t seed, float sigma) {
    std::mt19937_64 rng(seed); std::normal_distribution<float> nd(0.f, sigma);
    for (auto& e : v) { float f = nd(rng); __nv_bfloat16 b = __float2bfloat16(f); memcpy(&e, &b, 2); }
}
static void fill_act(std::vector<__nv_bfloat16>& v, uint64_t seed) {
    uint64_t s = seed | 1;
    for (auto& e : v) { s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        e = __float2bfloat16(((int)(s & 0xffff) - 32768) / 32768.0f * 0.25f); }
}

static double time_kernel(std::function<void()> launch, int iters, void* flushbuf, size_t flushbytes, cudaStream_t st) {
    for (int i = 0; i < 3; i++) launch(); CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double best = 1e30;
    for (int it = 0; it < iters; it++) {
        CK(cudaMemsetAsync(flushbuf, it & 0xff, flushbytes, st));
        CK(cudaEventRecord(e0, st)); launch(); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); if (ms < best) best = ms;
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1)); return best;
}

template <int BM, int BN, int BK, int NW, int STAGES>
static size_t smem_tc() { return (size_t)STAGES * (BM * (BK + 8) + BN * (BK + 8)) * sizeof(__nv_bfloat16); }
template <int BM, int BN, int BK, int NW, int STAGES>
static size_t smem_zip() { return (size_t)STAGES * (BM * (BK + 8) * sizeof(__nv_bfloat16) + BN * BK + BN * BK / 2); }

template <int BM, int BN, int BK, int NW, int STAGES>
static double run_tc(float* Cf, __nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
                     unsigned M, unsigned N, unsigned K, int ksplit, int iters, void* fb, size_t fbn,
                     cudaStream_t st, double* out_ms) {
    size_t smem = smem_tc<BM,BN,BK,NW,STAGES>();
    static size_t last = 0; if (smem != last) { CK(cudaFuncSetAttribute(k_tc<BM,BN,BK,NW,STAGES>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); last = smem; }
    int tiles_n = (N + BN - 1) / BN, njob = tiles_n * ksplit;
    int grid = njob < g_sm * 4 ? njob : g_sm * 4; if (grid < 1) grid = 1;
    size_t cf_n = (size_t)M * N;
    auto launch = [&]() { CK(cudaMemsetAsync(Cf, 0, cf_n * sizeof(float), st));
        k_tc<BM,BN,BK,NW,STAGES><<<grid, NW*32, smem, st>>>(Cf, A, B, M, N, K, ksplit); };
    double ms = time_kernel(launch, iters, fb, fbn, st);
    CK(cudaMemsetAsync(Cf, 0, cf_n * sizeof(float), st));
    k_tc<BM,BN,BK,NW,STAGES><<<grid, NW*32, smem, st>>>(Cf, A, B, M, N, K, ksplit);
    k_f32_to_bf16<<<(cf_n + 255) / 256, 256, 0, st>>>(C, Cf, cf_n); CK(cudaDeviceSynchronize());
    if (out_ms) *out_ms = ms; return (double)N * K * 2.0 / (ms * 1e-3) / 1e9;
}

template <int BM, int BN, int BK, int NW, int STAGES>
static double run_zip(float* Cf, __nv_bfloat16* C, const __nv_bfloat16* A, const uint8_t* lo,
                      const uint8_t* cd, const unsigned* eoff, const unsigned* epos, const uint16_t* eval,
                      unsigned M, unsigned N, unsigned K, int ksplit, int iters, void* fb, size_t fbn,
                      cudaStream_t st, double* out_ms) {
    size_t smem = smem_zip<BM,BN,BK,NW,STAGES>();
    static size_t last = 0; if (smem != last) { CK(cudaFuncSetAttribute(k_zipgemm<BM,BN,BK,NW,STAGES>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); last = smem; }
    int tiles_n = (N + BN - 1) / BN, njob = tiles_n * ksplit;
    int grid = njob < g_sm * 4 ? njob : g_sm * 4; if (grid < 1) grid = 1;
    size_t cf_n = (size_t)M * N;
    auto launch = [&]() { CK(cudaMemsetAsync(Cf, 0, cf_n * sizeof(float), st));
        k_zipgemm<BM,BN,BK,NW,STAGES><<<grid, NW*32, smem, st>>>(Cf, A, lo, cd, eoff, epos, eval, M, N, K, ksplit); };
    double ms = time_kernel(launch, iters, fb, fbn, st);
    CK(cudaMemsetAsync(Cf, 0, cf_n * sizeof(float), st));
    k_zipgemm<BM,BN,BK,NW,STAGES><<<grid, NW*32, smem, st>>>(Cf, A, lo, cd, eoff, epos, eval, M, N, K, ksplit);
    k_f32_to_bf16<<<(cf_n + 255) / 256, 256, 0, st>>>(C, Cf, cf_n); CK(cudaDeviceSynchronize());
    if (out_ms) *out_ms = ms; return (double)N * K * 2.0 / (ms * 1e-3) / 1e9;
}

/* device buffers for one compressed tensor */
struct DevComp { uint8_t *lo, *cd; unsigned *eoff, *epos; uint16_t *eval; };
static DevComp upload_comp(const Comp& c) {
    DevComp d;
    CK(cudaMalloc(&d.lo, c.lo.size())); CK(cudaMemcpy(d.lo, c.lo.data(), c.lo.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d.cd, c.cd.size())); CK(cudaMemcpy(d.cd, c.cd.data(), c.cd.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d.eoff, c.eoff.size()*4)); CK(cudaMemcpy(d.eoff, c.eoff.data(), c.eoff.size()*4, cudaMemcpyHostToDevice));
    size_t ne = c.epos.size() ? c.epos.size() : 1;
    CK(cudaMalloc(&d.epos, ne*4)); CK(cudaMalloc(&d.eval, ne*2));
    if (c.epos.size()) { CK(cudaMemcpy(d.epos, c.epos.data(), c.epos.size()*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.eval, c.eval.data(), c.eval.size()*2, cudaMemcpyHostToDevice)); }
    return d;
}
static void free_comp(DevComp& d) { cudaFree(d.lo); cudaFree(d.cd); cudaFree(d.eoff); cudaFree(d.epos); cudaFree(d.eval); }

/* pick the ZG-0 config per M and run zip; returns GB/s, sets bit-mismatch vs k_tc(ksplit=1). */
static double zip_run(unsigned M, float* Cf, __nv_bfloat16* C, const __nv_bfloat16* A, const DevComp& d,
                      unsigned N, unsigned K, int ksplit, int it, void* fb, size_t fbn, cudaStream_t st, double* ms) {
    if (M <= 16) return run_zip<16,128,64,8,4>(Cf,C,A,d.lo,d.cd,d.eoff,d.epos,d.eval,M,N,K,ksplit,it,fb,fbn,st,ms);
    if (M <= 32) return run_zip<32,64,64,4,4>(Cf,C,A,d.lo,d.cd,d.eoff,d.epos,d.eval,M,N,K,ksplit,it,fb,fbn,st,ms);
    return run_zip<64,64,64,4,4>(Cf,C,A,d.lo,d.cd,d.eoff,d.epos,d.eval,M,N,K,ksplit,it,fb,fbn,st,ms);
}
static double tc_run(unsigned M, float* Cf, __nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
                     unsigned N, unsigned K, int ksplit, int it, void* fb, size_t fbn, cudaStream_t st, double* ms) {
    if (M <= 16) return run_tc<16,128,64,8,4>(Cf,C,A,B,M,N,K,ksplit,it,fb,fbn,st,ms);
    if (M <= 32) return run_tc<32,64,64,4,4>(Cf,C,A,B,M,N,K,ksplit,it,fb,fbn,st,ms);
    return run_tc<64,64,64,4,4>(Cf,C,A,B,M,N,K,ksplit,it,fb,fbn,st,ms);
}
static int cfg_bn(unsigned M){ return M<=16?128:64; }

/* ---- fragment oracle (gate 2) for the M<=16 config (BN128) ---- */
static size_t run_frag_oracle(const uint16_t* Bblk_bits, unsigned N, unsigned K, cudaStream_t st) {
    constexpr int BN=128,BK=64,NW=8, NFRAG=(BN/NW)/8;
    /* Bblk is one [BN][BK] block; pack it and run. */
    Comp c = zg_pack(Bblk_bits, BN, BK, BN, BK);
    __nv_bfloat16* dB; CK(cudaMalloc(&dB, (size_t)BN*BK*2)); CK(cudaMemcpy(dB, Bblk_bits, (size_t)BN*BK*2, cudaMemcpyHostToDevice));
    DevComp d = upload_comp(c);
    int nfrags = NW*NFRAG*(BK/16)*32*2;
    unsigned *dld,*drc; CK(cudaMalloc(&dld,nfrags*4)); CK(cudaMalloc(&drc,nfrags*4));
    size_t smem = (size_t)BN*(BK+8)*2;
    CK(cudaFuncSetAttribute(k_frag_oracle<BN,BK,NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    k_frag_oracle<BN,BK,NW><<<1, NW*32, smem, st>>>(dld, drc, dB, d.lo, d.cd, d.eoff, d.epos, d.eval, BN, BK);
    CK(cudaDeviceSynchronize());
    std::vector<unsigned> hld(nfrags), hrc(nfrags);
    CK(cudaMemcpy(hld.data(),dld,nfrags*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hrc.data(),drc,nfrags*4,cudaMemcpyDeviceToHost));
    size_t bad=0; for(int i=0;i<nfrags;i++) if(hld[i]!=hrc[i]) bad++;
    cudaFree(dB);free_comp(d);cudaFree(dld);cudaFree(drc);
    return bad;
}

int main(int argc, char** argv) {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0)); g_sm = prop.multiProcessorCount;
    printf("# GPU %s  SMs=%d  smem/SM=%zuKB\n", prop.name, g_sm, prop.sharedMemPerMultiprocessor/1024);
    const double WALL = 1535.0; const float SIGMA = 0.02f;
    std::vector<Shape> shapes = { {"qkv",8192,3840}, {"o_proj",3840,4096}, {"gate/up",15360,3840}, {"down",3840,15360} };
    std::vector<unsigned> Ms = {1,2,4,8,16,32,64};
    std::string mode = argc > 1 ? argv[1] : "full";

    cudaStream_t st; CK(cudaStreamCreate(&st));
    size_t flushbytes = 256ull*1024*1024; void* flushbuf; CK(cudaMalloc(&flushbuf, flushbytes));
    float* d_sink; CK(cudaMalloc(&d_sink,4)); int iters = 30;

    /* ============ GATE 1: host lossless byte-exact (every tensor, both BN configs) ============ */
    printf("\n## GATE 1 — host lossless byte-exact (pack->unpack == original bf16)\n");
    bool g1_ok = true; double worst_ratio = 1e9, best_ratio = 0; unsigned tot_esc = 0; size_t tot_el = 0;
    for (auto& s : shapes) {
        size_t wn = (size_t)s.N * s.K; std::vector<uint16_t> hW(wn); fill_weights(hW, 100 + s.N, SIGMA);
        for (int bn : {128, 64}) {
            Comp c = zg_pack(hW.data(), s.N, s.K, bn, 64);
            size_t bad = zg_unpack_check(c, hW.data(), s.N, s.K);
            double comp_bytes = wn + wn/2.0 + c.nesc*6.0; double ratio = (wn*2.0)/comp_bytes;
            printf("  %-8s BN%-3d : %s  escapes=%u (%.4f%%)  ratio=%.4fx  (%zu mismatch)\n",
                   s.name, bn, bad==0?"OK":"MISMATCH", c.nesc, 100.0*c.nesc/wn, ratio, bad);
            if (bad) g1_ok = false;
            if (bn==64){ tot_esc += c.nesc; tot_el += wn; if(ratio<worst_ratio)worst_ratio=ratio; if(ratio>best_ratio)best_ratio=ratio; }
        }
    }
    /* negative control: corrupt one lo byte -> must mismatch */
    {   Shape s = shapes[0]; size_t wn=(size_t)s.N*s.K; std::vector<uint16_t> hW(wn); fill_weights(hW,7,SIGMA);
        Comp c = zg_pack(hW.data(), s.N, s.K, 64, 64); c.lo[12345] ^= 0x40;
        size_t bad = zg_unpack_check(c, hW.data(), s.N, s.K);
        printf("  neg-control (1 lo byte corrupted): %s (%zu mismatch, expect >0)\n", bad>0?"OK":"FAIL", bad);
        if (bad==0) g1_ok=false;
    }
    printf("  GATE 1: %s\n", g1_ok?"PASS":"FAIL");
    if (!g1_ok) { printf("STOP: gate 1 failed\n"); return 1; }

    /* ============ GATE 2: fragment oracle (recon == ldmatrix), per lane ============ */
    printf("\n## GATE 2 — fragment oracle (recon(baked TCA) == ldmatrix(uncompressed)), BN128/BK64\n");
    bool g2_ok = true;
    for (int trial = 0; trial < 4; trial++) {
        std::vector<uint16_t> blk(128*64); fill_weights(blk, 555 + trial, SIGMA);
        size_t bad = run_frag_oracle(blk.data(), 128, 64, st);
        printf("  trial %d: %s (%zu lane-frag mismatch)\n", trial, bad==0?"OK":"MISROUTE", bad);
        if (bad) g2_ok = false;
    }
    printf("  GATE 2: %s\n", g2_ok?"PASS":"FAIL");
    if (!g2_ok) { printf("STOP: gate 2 failed (lane derivation wrong)\n"); return 1; }

    if (mode == "gates") { printf("\n# gates-only mode done\n"); return 0; }

    /* ============ ONE: single launch of zip + tc for ncu (args: shapeIdx M) ============ */
    if (mode == "one") {
        int si = argc>2?atoi(argv[2]):3; unsigned M = argc>3?atoi(argv[3]):8; Shape s=shapes[si];
        size_t wn=(size_t)s.N*s.K; std::vector<uint16_t> hW(wn); fill_weights(hW,100+s.N,SIGMA);
        __nv_bfloat16* dW; CK(cudaMalloc(&dW,wn*2)); CK(cudaMemcpy(dW,hW.data(),wn*2,cudaMemcpyHostToDevice));
        Comp c=zg_pack(hW.data(),s.N,s.K,cfg_bn(M),64); DevComp d=upload_comp(c);
        std::vector<__nv_bfloat16> hA((size_t)M*s.K); fill_act(hA,2+M);
        __nv_bfloat16 *dA; float* dCf; CK(cudaMalloc(&dA,(size_t)M*s.K*2)); CK(cudaMalloc(&dCf,(size_t)M*s.N*4));
        CK(cudaMemcpy(dA,hA.data(),(size_t)M*s.K*2,cudaMemcpyHostToDevice));
        CK(cudaMemset(dCf,0,(size_t)M*s.N*4));
        int bn=cfg_bn(M); int tn=(s.N+bn-1)/bn; int ksplit=(4*g_sm+tn-1)/tn; if(ksplit<1)ksplit=1; if(ksplit>16)ksplit=16;
        int njob=tn*ksplit; int grid=njob<g_sm*4?njob:g_sm*4;
        size_t szip=smem_zip<16,128,64,8,4>(), stc=smem_tc<16,128,64,8,4>();
        CK(cudaFuncSetAttribute(k_zipgemm<16,128,64,8,4>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)szip));
        CK(cudaFuncSetAttribute(k_tc<16,128,64,8,4>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)stc));
        k_zipgemm<16,128,64,8,4><<<grid,256,szip>>>(dCf,dA,d.lo,d.cd,d.eoff,d.epos,d.eval,M,s.N,s.K,ksplit);
        CK(cudaMemset(dCf,0,(size_t)M*s.N*4));
        k_tc<16,128,64,8,4><<<grid,256,stc>>>(dCf,dA,dW,M,s.N,s.K,ksplit);
        CK(cudaDeviceSynchronize()); printf("# one shape=%s M=%u done\n",s.name,M); return 0;
    }

    /* ============ DIAG: is k_zipgemm BW-bound or recon-bound? ============
     * Compare zip's ACTUAL DRAM throughput (logical GB/s * compressed/logical) against the
     * cold-read ceiling of the SAME compressed footprint (lo+cd). If zip_actual ~= comp_ceiling
     * the recon is hidden and the slowdown is the cold-ramp footprint effect (smaller footprint =
     * lower on the latency ramp). If zip_actual << comp_ceiling the recon is NOT hidden -> the
     * decompress is on the critical path (the genuine kill). */
    if (mode == "diag" || mode == "full") {
        printf("\n## DIAG — BW-bound vs recon-bound (cold, L2-flushed)\n");
        printf("%-8s %3s | %8s %9s | %9s %9s | %8s %8s | %s\n",
               "shape","M","ziplog GB","zipDRAM","bf16ceil","compceil","zip/comp","bf/bfc","regime");
        for (auto& s : shapes) {
            size_t wn=(size_t)s.N*s.K; std::vector<uint16_t> hW(wn); fill_weights(hW,100+s.N,SIGMA);
            __nv_bfloat16* dW; CK(cudaMalloc(&dW,wn*2)); CK(cudaMemcpy(dW,hW.data(),wn*2,cudaMemcpyHostToDevice));
            Comp c128=zg_pack(hW.data(),s.N,s.K,128,64); DevComp d128=upload_comp(c128);
            Comp c64 =zg_pack(hW.data(),s.N,s.K,64, 64); DevComp d64 =upload_comp(c64);
            double comp_frac = (double)(wn + wn/2) / (double)(wn*2);   /* 0.751 */
            /* bf16 cold ceiling */
            double bf16ceil; { size_t ne8=wn/8; int grid=g_sm*8; auto l=[&](){k_stream_reduce<<<grid,256,0,st>>>(dW,ne8,d_sink);};
                double ms=time_kernel(l,iters,flushbuf,flushbytes,st); bf16ceil=(double)wn*2/(ms*1e-3)/1e9; }
            /* compressed cold ceiling: stream lo (wn bytes) + cd (wn/2 bytes) */
            double compceil; { uint8_t* dcat; size_t nb=wn+wn/2; CK(cudaMalloc(&dcat,nb));
                CK(cudaMemcpy(dcat,d128.lo,wn,cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(dcat+wn,d128.cd,wn/2,cudaMemcpyDeviceToDevice));
                size_t ne8=nb/16; int grid=g_sm*8; auto l=[&](){k_stream_reduce<<<grid,256,0,st>>>((const __nv_bfloat16*)dcat,ne8,d_sink);};
                double ms=time_kernel(l,iters,flushbuf,flushbytes,st);
                /* report as LOGICAL-equivalent GB/s: actual comp bytes / time */
                compceil=(double)nb/(ms*1e-3)/1e9; cudaFree(dcat); }
            for (unsigned M : {8u,64u}) {
                std::vector<__nv_bfloat16> hA((size_t)M*s.K); fill_act(hA,2+M);
                __nv_bfloat16 *dA,*dCz; float* dCf; CK(cudaMalloc(&dA,(size_t)M*s.K*2));
                CK(cudaMalloc(&dCz,(size_t)M*s.N*2)); CK(cudaMalloc(&dCf,(size_t)M*s.N*4));
                CK(cudaMemcpy(dA,hA.data(),(size_t)M*s.K*2,cudaMemcpyHostToDevice));
                int bn=cfg_bn(M); int tn=(s.N+bn-1)/bn; int ksplit=(4*g_sm+tn-1)/tn; if(ksplit<1)ksplit=1; if(ksplit>16)ksplit=16;
                const DevComp& d=(M<=16)?d128:d64; double ms;
                double gz=zip_run(M,dCf,dCz,dA,d,s.N,s.K,ksplit,iters,flushbuf,flushbytes,st,&ms);
                double zipDRAM=gz*comp_frac;
                double zc=zipDRAM/compceil, bfc=bf16ceil/bf16ceil; (void)bfc;
                const char* regime = zc>0.90?"BW-bound(recon hidden)":zc>0.75?"partial":"RECON-BOUND";
                printf("%-8s %3u | %8.1f %9.1f | %9.1f %9.1f | %7.1f%% %7s | %s\n",
                       s.name,M,gz,zipDRAM,bf16ceil,compceil,100*zc,"",regime);
                cudaFree(dA);cudaFree(dCz);cudaFree(dCf);
            }
            free_comp(d128); free_comp(d64); cudaFree(dW);
        }
        if (mode=="diag") return 0;
    }

    /* ============ GATE 3 + PERF: device bit-exact + GB/s vs tuned TC-bf16 ============ */
    FILE* jf = fopen("perf-data/zg12-zipgemm.json","w");
    fprintf(jf, "{\n  \"gpu\": \"%s\", \"sm\": %d, \"wall_achievable_gbs\": %.0f, \"sigma\": %.3f,\n", prop.name, g_sm, WALL, SIGMA);
    fprintf(jf, "  \"note\": \"k_zipgemm (sz12 TCA fused) vs ZG-0 tuned TC-bf16; GB/s=logical bf16 bytes/time; bit_mismatch vs TC-bf16 at ksplit=1\",\n  \"rows\": [\n");
    bool firstrow = true, g3_ok = true;

    printf("\n## GATE 3 + PERF — k_zipgemm vs tuned TC-bf16 (per shape x M)\n");
    printf("%-8s %3s | %8s | %8s | %6s | %8s | %s\n","shape","M","ZIP GB","TCbf GB","speedup","%1535","bit-exact");
    for (auto& s : shapes) {
        size_t wn = (size_t)s.N * s.K; std::vector<uint16_t> hW(wn); fill_weights(hW, 100 + s.N, SIGMA);
        __nv_bfloat16* dW; CK(cudaMalloc(&dW, wn*2)); CK(cudaMemcpy(dW, hW.data(), wn*2, cudaMemcpyHostToDevice));
        /* compressed twin per BN config (128 for M<=16, 64 for M>16) */
        Comp c128 = zg_pack(hW.data(), s.N, s.K, 128, 64); DevComp d128 = upload_comp(c128);
        Comp c64  = zg_pack(hW.data(), s.N, s.K, 64,  64); DevComp d64  = upload_comp(c64);
        for (unsigned M : Ms) {
            std::vector<__nv_bfloat16> hA((size_t)M*s.K); fill_act(hA, 2+M);
            __nv_bfloat16 *dA,*dCz,*dCt; float* dCf;
            CK(cudaMalloc(&dA,(size_t)M*s.K*2)); CK(cudaMalloc(&dCz,(size_t)M*s.N*2));
            CK(cudaMalloc(&dCt,(size_t)M*s.N*2)); CK(cudaMalloc(&dCf,(size_t)M*s.N*4));
            CK(cudaMemcpy(dA,hA.data(),(size_t)M*s.K*2,cudaMemcpyHostToDevice));
            int bn = cfg_bn(M); int tiles_n=(s.N+bn-1)/bn;
            int ksplit=(4*g_sm+tiles_n-1)/tiles_n; if(ksplit<1)ksplit=1; if(ksplit>16)ksplit=16;
            const DevComp& d = (M<=16)?d128:d64;
            double msz, mst;
            double gz = zip_run(M,dCf,dCz,dA,d,s.N,s.K,ksplit,iters,flushbuf,flushbytes,st,&msz);
            double gt = tc_run(M,dCf,dCt,dA,dW,s.N,s.K,ksplit,iters,flushbuf,flushbytes,st,&mst);
            /* bit-exact: zip(ksplit=1) vs tc(ksplit=1) */
            std::vector<__nv_bfloat16> gz1((size_t)M*s.N), gt1((size_t)M*s.N);
            zip_run(M,dCf,dCz,dA,d,s.N,s.K,1,3,flushbuf,flushbytes,st,&msz);
            CK(cudaMemcpy(gz1.data(),dCz,(size_t)M*s.N*2,cudaMemcpyDeviceToHost));
            tc_run(M,dCf,dCt,dA,dW,s.N,s.K,1,3,flushbuf,flushbytes,st,&mst);
            CK(cudaMemcpy(gt1.data(),dCt,(size_t)M*s.N*2,cudaMemcpyDeviceToHost));
            long bit=0; for(size_t i=0;i<gz1.size();i++){ uint16_t a,b; memcpy(&a,&gz1[i],2); memcpy(&b,&gt1[i],2); if(a!=b)bit++; }
            if (bit) g3_ok=false;
            double spd = gz/gt;
            printf("%-8s %3u | %8.1f | %8.1f | %5.3fx | %5.1f%% | %s(%ld)\n",
                   s.name, M, gz, gt, spd, 100*gz/WALL, bit==0?"PASS":"FAIL", bit);
            fprintf(jf, "%s    {\"shape\":\"%s\",\"N\":%u,\"K\":%u,\"M\":%u,\"zip_gbs\":%.1f,\"tc_bf16_gbs\":%.1f,"
                "\"speedup\":%.4f,\"zip_pct1535\":%.1f,\"ksplit\":%d,\"bit_mismatch\":%ld}",
                firstrow?"":",\n", s.name, s.N, s.K, M, gz, gt, spd, 100*gz/WALL, ksplit, bit);
            firstrow=false;
            cudaFree(dA);cudaFree(dCz);cudaFree(dCt);cudaFree(dCf);
        }
        free_comp(d128); free_comp(d64); cudaFree(dW);
    }
    fprintf(jf, "\n  ],\n  \"gate1_lossless\": %s, \"gate2_fragment\": %s, \"gate3_bitexact\": %s,\n",
            g1_ok?"true":"false", g2_ok?"true":"false", g3_ok?"true":"false");
    fprintf(jf, "  \"escape_rate_pct\": %.4f, \"ratio_best\": %.4f, \"ratio_worst\": %.4f\n}\n",
            100.0*tot_esc/tot_el, best_ratio, worst_ratio);
    fclose(jf);
    printf("\n# GATE 3: %s  |  wrote perf-data/zg12-zipgemm.json\n", g3_ok?"PASS":"FAIL");
    return g3_ok?0:1;
}
