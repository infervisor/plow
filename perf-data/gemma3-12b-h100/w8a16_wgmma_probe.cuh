
__device__ __forceinline__ void stage_w8a16(__nv_bfloat16* dst,const uint8_t* src,int tid,int rows,int row0,int kbase,int R,int K) {
  for(int L=tid;L<rows*PGM90_CH;L+=256) {
    int row=L/PGM90_CH,c=L%PGM90_CH,gr=row0+row,gk=kbase+c*8;
    uint2 in={0,0};
    if(gr<R && gk+8<=K) in=*(const uint2*)(src+(size_t)gr*K+gk);
    const uint16_t* w=(const uint16_t*)&in;
    __nv_bfloat16 vals[8];
#pragma unroll
    for(int j=0;j<4;++j) {
      __half2_raw h=__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)w[j],__NV_E4M3);
      float2 v=__half22float2(*reinterpret_cast<__half2*>(&h));
      vals[2*j]=__float2bfloat16(v.x);vals[2*j+1]=__float2bfloat16(v.y);
    }
    *(uint4*)(dst+sm90_swz_off<PGM90_BK,8>(row,c))=*(uint4*)vals;
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
static __device__ void d_gemm_w8a16_probe(__nv_bfloat16* __restrict__ C,
                                   const __nv_bfloat16* __restrict__ A,
                                   const uint8_t* __restrict__ B, unsigned m, unsigned n,
                                   unsigned k, unsigned a_row0, unsigned slice, unsigned nblk,
                                   __nv_bfloat16* arena,
                                   const float* __restrict__ scale) {
    if (sm90_bad_k(k, 8u)) return; /* swizzle staging needs K%8==0, k>0 (bf16 chunk = 8 elems) */
    __nv_bfloat16* base = (__nv_bfloat16*)sm90_align1024(arena);
    __nv_bfloat16* As = base;                                /* [STAGES][128][64] swizzled */
    __nv_bfloat16* Bs = base + PGM90_STAGES * PGM90_ABUF;    /* [STAGES][128][64] swizzled */
    const int tid = (int)threadIdx.x;
    const int wg = tid >> 7;              /* warpgroup: owns A rows [64*wg, 64*wg+64) */
    const int wiw = (tid >> 5) & 3;       /* warp within the warpgroup */
    const int lane = tid & 31;
    const __nv_bfloat16* Ab = A + (size_t)a_row0 * k;

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_BN - 1) / PGM90_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + PGM90_BK - 1) / PGM90_BK;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        int tmi, tni;
        sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
        const int tm = tmi * PGM90_BM;
        const int tn = tni * PGM90_BN;
        float acc[PGM90_NACC];  /* seeded by the first wgmma (scale-d = 0), no zeroing pass */

        sm90_cp_wait<0>();   /* drain the previous tile so the group count starts clean */
        __syncthreads();
#pragma unroll
        for (int s = 0; s < PGM90_STAGES - 1; s++) {
            if (s < ksteps) {
                pgm90_stage_bf16(As + s * PGM90_ABUF, Ab, tid, PGM90_BM, tm, s * PGM90_BK, (int)m,
                                 (int)k);
                stage_w8a16(Bs + s * PGM90_BBUF, B, tid, PGM90_BN, tn, s * PGM90_BK, (int)n,
                                 (int)k);
            }
            sm90_cp_commit();
        }

        for (int ks = 0; ks < ksteps; ks++) {
            const int cur = ks % PGM90_STAGES;
            sm90_cp_wait<PGM90_STAGES - 2>();
            __syncthreads();
            const __nv_bfloat16* Ac = As + cur * PGM90_ABUF + wg * PGM90_MSLAB * PGM90_BK;
            const __nv_bfloat16* Bc = Bs + cur * PGM90_BBUF;
            sm90_wg_fence();
#pragma unroll
            for (int sub = 0; sub < PGM90_KSUB; sub++)
                wgmma_m64n128k16(acc, sm90_desc(Ac + sub * 16), sm90_desc(Bc + sub * 16),
                                 (ks == 0 && sub == 0) ? 0 : 1);
            sm90_wg_commit();
            sm90_wg_wait<1>();   /* group ks-1 retired -> its buffer is free (this warpgroup) */
            __syncthreads();     /* ...and for the OTHER warpgroup too, before we refill it */
            const int nxt = ks + PGM90_STAGES - 1;
            if (nxt < ksteps) {
                const int nb = nxt % PGM90_STAGES;
                pgm90_stage_bf16(As + nb * PGM90_ABUF, Ab, tid, PGM90_BM, tm, nxt * PGM90_BK,
                                 (int)m, (int)k);
                stage_w8a16(Bs + nb * PGM90_BBUF, B, tid, PGM90_BN, tn, nxt * PGM90_BK,
                                 (int)n, (int)k);
            }
            sm90_cp_commit();
        }
        sm90_wg_wait<0>();

        const int r0 = tm + wg * PGM90_MSLAB + wiw * 16 + (lane >> 2);
        const int c0 = tn + 2 * (lane & 3);
#pragma unroll
        for (int g = 0; g < PGM90_BN / 8; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = r0 + 8 * hi;
                if (rr >= (int)m) continue;
#pragma unroll
                for (int lo = 0; lo < 2; lo++) {
                    const int cc = c0 + 8 * g + lo;
                    if (cc < (int)n) {
                        float v = acc[4 * g + 2 * hi + lo];
                        v *= scale[cc];
                        C[(size_t)rr * n + cc] = __float2bfloat16(v);
                    }
                }
            }
        __syncthreads();
    }
}
