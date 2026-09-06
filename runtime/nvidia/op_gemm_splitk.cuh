/* ------------------------------------------------------------------ tuned TC GEMM (template) */
/* All NW warps share the same M rows and split BN columns; MFRAG = BM/16 m-fragments. Weight
 * B[n][k] staged in its natural [n][k] layout, read with ldmatrix.x2 non-.trans (proven in
 * op_gemm.cuh's probe). Split-K: a k-slice per block, accumulate into a global f32 buffer via
 * atomicAdd (ksplit=1 => single writer => identical to a direct store => bit-exact vs k_tc_bf16). */
template <int BM, int BN, int BK, int NW, int STAGES>
__device__ __forceinline__ void d_gemm_splitk(float* __restrict__ Cf, /* [M*N] f32, pre-zeroed */
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
        unsigned M, unsigned N, unsigned K, int ksplit, unsigned slice, unsigned nblk, char* sm) {
    constexpr int MFRAG = BM / 16;
    constexpr int WN    = BN / NW;         /* columns per warp */
    constexpr int NFRAG = WN / 8;
    constexpr int AS    = BK + 8;          /* A smem stride [m][k] */
    constexpr int BKS   = BK + 8;          /* B smem stride [n][k] */
    constexpr int ABUF  = BM * AS;
    constexpr int BBUF  = BN * BKS;
    constexpr int KCH   = BK / 8;          /* 16B lines per row */

    __nv_bfloat16* As = (__nv_bfloat16*)sm;
    __nv_bfloat16* Bs = As + STAGES * ABUF;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tiles_n = (N + BN - 1) / BN;
    const int totksteps = (K + BK - 1) / BK;
    /* split K into `ksplit` contiguous groups of k-steps */
    const int kper = (totksteps + ksplit - 1) / ksplit;
    const int njob = tiles_n * ksplit;

    for (int job = slice; job < njob; job += nblk) {
        const int nt = job / ksplit;          /* which N-tile */
        const int ksp = job % ksplit;         /* which K-slice */
        const int tn = nt * BN;
        const int ks0 = ksp * kper;
        const int ks1 = (ks0 + kper < totksteps) ? (ks0 + kper) : totksteps;
        if (ks0 >= ks1) continue;

        float acc[MFRAG][NFRAG][4];
#pragma unroll
        for (int i = 0; i < MFRAG; i++)
            for (int j = 0; j < NFRAG; j++)
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

        auto stage = [&](int ks, int buf) {
            /* A tile [BM][BK] (k contiguous). OOB rows/cols => src_bytes 0 => HW zero-fill, no HBM. */
#pragma unroll
            for (int L = tid; L < BM * KCH; L += NW * 32) {
                const int row = L / KCH, kk8 = (L % KCH) * 8;
                const int mm = row, kk = ks * BK + kk8;
                const bool in = (mm < (int)M) && (kk + 8 <= (int)K);
                const __nv_bfloat16* g = in ? A + (size_t)mm * K + kk : A;
                pgm_cp_async_cg16(&As[buf * ABUF + row * AS + kk8], g, in ? 16 : 0);
            }
            /* B tile [BN][BK] */
#pragma unroll
            for (int L = tid; L < BN * KCH; L += NW * 32) {
                const int row = L / KCH, kk8 = (L % KCH) * 8;
                const int nn = tn + row, kk = ks * BK + kk8;
                const bool in = (nn < (int)N) && (kk + 8 <= (int)K);
                const __nv_bfloat16* g = in ? B + (size_t)nn * K + kk : B;
                pgm_cp_async_cg16(&Bs[buf * BBUF + row * BKS + kk8], g, in ? 16 : 0);
            }
        };

        const int nks = ks1 - ks0;
#pragma unroll 1
        for (int s = 0; s < STAGES - 1; s++) {
            if (s < nks) stage(ks0 + s, s);
            pgm_cp_commit();
        }
        for (int i = 0; i < nks; i++) {
            const int fetch = i + STAGES - 1;
            if (fetch < nks) stage(ks0 + fetch, fetch % STAGES);
            pgm_cp_commit();
            pgm_cp_wait<STAGES - 1>();
            __syncthreads();
            const int cb = i % STAGES;
            __nv_bfloat16* Ad = As + cb * ABUF;
            __nv_bfloat16* Bd = Bs + cb * BBUF;
#pragma unroll
            for (int kf = 0; kf < BK; kf += 16) {
                unsigned af[MFRAG][4];
#pragma unroll
                for (int mi = 0; mi < MFRAG; mi++) {
                    const int arow = mi * 16 + (lane % 16);
                    const int acol = kf + (lane / 16) * 8;
                    pgm_ldmatrix_x4(af[mi], &Ad[arow * AS + acol]);
                }
                unsigned bf[NFRAG][2];
#pragma unroll
                for (int nj = 0; nj < NFRAG; nj++) {
                    const int nrow = warp * WN + nj * 8 + (lane & 7);
                    const int kcol = kf + ((lane >> 3) & 1) * 8;
                    pgm_ldmatrix_x2(bf[nj], &Bd[nrow * BKS + kcol]);
                }
#pragma unroll
                for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < NFRAG; nj++)
                        pgm_mma(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            __syncthreads();
        }

        /* epilogue: atomicAdd f32 partials into Cf[m*N+n] */
#pragma unroll
        for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < NFRAG; nj++) {
                const int gr = lane / 4;
                const int gc = warp * WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int rr = mi * 16 + gr + (e / 2) * 8;
                    const int cc = tn + gc + (e % 2);
                    if (rr < (int)M && cc < (int)N)
                        atomicAdd(&Cf[(size_t)rr * N + cc], acc[mi][nj][e]);
                }
            }
        __syncthreads();
    }
    pgm_cp_wait<0>();
    __syncthreads();
}

__device__ __forceinline__ void d_zero_f32(float* out, size_t n, unsigned slice, unsigned nblk) {
    for (size_t i = (size_t)slice * blockDim.x + threadIdx.x; i < n; i += (size_t)nblk * blockDim.x) out[i] = 0.f;
}
__device__ __forceinline__ void d_cast_f32_bf16(__nv_bfloat16* out, const float* in, size_t n, unsigned slice, unsigned nblk) {
    for (size_t i = (size_t)slice * blockDim.x + threadIdx.x; i < n; i += (size_t)nblk * blockDim.x) out[i] = __float2bfloat16(in[i]);
}
