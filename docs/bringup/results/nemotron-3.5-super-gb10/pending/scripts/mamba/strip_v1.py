#!/usr/bin/env python3
"""strip_v1.py <op_mamba_pf.cuh>: drop the dev-only MPF_OG_* switches at their production values
(RBMAP=1, NV=0 -> volatile mpf_mma, SPLITP=0, DIAG=0)."""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


rep('''/* Same instruction, schedulable by the compiler (no `volatile`). */
__device__ __forceinline__ void mpf_mma_s(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#ifndef MPF_OG_NV /* dev only */
#define MPF_OG_NV 0
#endif
#if MPF_OG_NV
#define MPF_OG_MMA mpf_mma_s
#else
#define MPF_OG_MMA mpf_mma
#endif
''', '')
rep('''#ifndef MPF_OG_RBMAP /* dev only */
#define MPF_OG_RBMAP 1
#endif
#ifndef MPF_OG_SPLITP /* dev only: P of every block first, then the P.x mmas */
#define MPF_OG_SPLITP 0
#endif
#ifndef MPF_OG_DIAG /* dev only: 1 = no P.x, 2 = no C.E (wrong numerics, timing split) */
#define MPF_OG_DIAG 0
#endif
    const unsigned rb = MPF_OG_RBMAP ? (warp < 4u ? warp : 11u - warp) : warp,''',
    '''    const unsigned rb = warp < 4u ? warp : 11u - warp,''')
rep('''                if (MPF_OG_DIAG == 2) break;
''', '')
i = s.index('''            unsigned afs[MPF_OG_SPLITP ? 8 : 1][4];''')
j = s.index('''            const float Dh = mamba_bf(Dp, h);''', i)
s = s[:i] + '''#pragma unroll
            for (int j = 0; j < 8; j++) {
                if ((unsigned)j > rb) continue;
                const unsigned s0 = j * 16u;
                unsigned af[4];
#pragma unroll
                for (int a = 0; a < 2; a++) {
                    const unsigned sa = s0 + a * 8u + 2u * tq, sb = sa + 1u;
                    const float csa = s_cum[sa], csb = s_cum[sb], dsa = s_dt[sa], dsb = s_dt[sb];
                    const float v0 = sa <= ta ? cb[j][a][0] * __expf(fminf(cta - csa, 0.f)) * dsa : 0.f;
                    const float v1 = sb <= ta ? cb[j][a][1] * __expf(fminf(cta - csb, 0.f)) * dsb : 0.f;
                    const float v2 = sa <= tb ? cb[j][a][2] * __expf(fminf(ctb - csa, 0.f)) * dsa : 0.f;
                    const float v3 = sb <= tb ? cb[j][a][3] * __expf(fminf(ctb - csb, 0.f)) * dsb : 0.f;
                    af[2 * a] = mpf_bf2(v0, v1);
                    af[2 * a + 1] = mpf_bf2(v2, v3);
                }
#pragma unroll
                for (int pp = 0; pp < 4; pp++) {
                    unsigned bf[4];
                    mpf_ldsm_t(bf, xs + (s0 + (lane & 15u)) * MPF_SP + pp * 16u + (lane >> 4) * 8u);
                    mpf_mma(acc[2 * pp], af, bf[0], bf[1]);
                    mpf_mma(acc[2 * pp + 1], af, bf[2], bf[3]);
                }
            }
''' + s[j:]
s = s.replace("MPF_OG_MMA(", "mpf_mma(")
left = [l for l in s.splitlines() if "MPF_OG_" in l and "MPF_OG_STAGE" not in l]
assert not left, left
open(p, "w").write(s)
print("stripped")
