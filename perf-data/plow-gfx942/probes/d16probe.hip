/* #37 probe: ds_read_u16_d16_hi for the MLA-PF2 PV transpose read.
 *
 * The shipped SV loop builds a bf16x8 V-fragment from 8 strided ds_read_u16 out of the
 * [kv][d] slab; LLVM then emits 4 v_perm_b32 per output tile to pack the half-pairs
 * (128 perms per KV tile -- glm52-flash-streamed-v.md sec 8 item 1). ds_read_u16_d16_hi
 * writes the HIGH half of a VGPR in place, deleting them.
 *
 * Arms (identical loads, identical MFMA order -- only the SOURCE IDIOM differs):
 *   0  shipped: per-element insert into bf16x8, SV(3) cross-tile double buffer
 *   1  bf16x2 pair build          -- does LLVM form d16_hi from a 2-elt bf16 vector?
 *   2  u16x2 pair build           -- ... from the integer vector type it natively targets?
 *   3  inline asm, drained per tile (lgkmcnt(0)) -- prices the perms alone
 *   4  inline asm, KEEPS the double buffer (lgkmcnt(8))
 *
 * Arm 4 is the candidate. The `lgkmcnt(8)` is safe because DS-to-DS returns IN ORDER on
 * gfx9: 8 strictly-newer DS ops outstanding forces every older one complete. Older
 * non-DS lgkm ops (s_load) can only inflate the count, i.e. make the wait LONGER.
 *
 * Ksm sits at a NONZERO offset in the shared pool so the generic->LDS address cast the
 * asm needs is exercised the way the megakernel exercises it (there `lds` is a pointer
 * into the plow_smem union, not a directly-addressed array).
 *
 * Geometry from op_attention.h d_flash_mla_prefill_v2<512,64>: BKV 32, KSTR 584 halves,
 * NT 32, SWZ 16, 4 waves of 64.
 *
 * Gate: every arm must be BIT-IDENTICAL to arm 0 on random LDS content.
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef unsigned short bf16;
typedef __bf16 bf16_t;
typedef bf16_t bf16x8 __attribute__((ext_vector_type(8)));
typedef bf16_t plow_bf16x4 __attribute__((ext_vector_type(4)));
typedef bf16_t bf16x2 __attribute__((ext_vector_type(2)));
typedef unsigned short u16x2 __attribute__((ext_vector_type(2)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ f32x4 mfma16(bf16x8 a, bf16x8 b, f32x4 acc) {
    const plow_bf16x4 a0 = {a[0], a[1], a[2], a[3]}, a1 = {a[4], a[5], a[6], a[7]};
    const plow_bf16x4 b0 = {b[0], b[1], b[2], b[3]}, b1 = {b[4], b[5], b[6], b[7]};
    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a0, b0, acc, 0, 0, 0);
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a1, b1, acc, 0, 0, 0);
}

/* Generic pointer -> LDS byte offset, the address a DS instruction consumes. */
__device__ __forceinline__ unsigned lds_addr(const void* p) {
    return (unsigned)(unsigned long long)(__attribute__((address_space(3))) const void*)p;
}

#define DK 512
#define DR 64
#define PAD 8
#define BKV 32
#define SWZ 16
#define KSTR (DK + DR + PAD) /* 584 halves */
#define NT (DK / 16)         /* 32 output tiles */
#define KSLAB (BKV * KSTR + 3 * SWZ)
#define PSTRIP (4 * 16 * BKV)
#define SKEW 96 /* deliberate nonzero base for Ksm inside the pool */

__device__ __forceinline__ unsigned krow(unsigned kv) {
    return kv * (unsigned)KSTR + SWZ * (kv >> 3);
}

/* The 8 j-rows of one V fragment are KSTR halves apart, so the DS offset field
 * (16-bit unsigned byte immediate) carries them: 14*KSTR*2 = 16352 B < 65536. */
#define D16_ISSUE(A)                                                                     \
    "ds_read_u16 %0, " A " offset:0\n\t"                                                 \
    "ds_read_u16_d16_hi %0, " A " offset:%c[o1]\n\t"                                     \
    "ds_read_u16 %1, " A " offset:%c[o2]\n\t"                                            \
    "ds_read_u16_d16_hi %1, " A " offset:%c[o3]\n\t"                                     \
    "ds_read_u16 %2, " A " offset:%c[o4]\n\t"                                            \
    "ds_read_u16_d16_hi %2, " A " offset:%c[o5]\n\t"                                     \
    "ds_read_u16 %3, " A " offset:%c[o6]\n\t"                                            \
    "ds_read_u16_d16_hi %3, " A " offset:%c[o7]\n\t"
#define D16_OFFS                                                                         \
    [o1] "i"(2 * KSTR), [o2] "i"(4 * KSTR), [o3] "i"(6 * KSTR), [o4] "i"(8 * KSTR),      \
        [o5] "i"(10 * KSTR), [o6] "i"(12 * KSTR), [o7] "i"(14 * KSTR)

template <int ARM>
__global__ __launch_bounds__(256) void probe(const bf16* __restrict__ src,
                                             float* __restrict__ out, unsigned ntile) {
    __shared__ bf16 pool[SKEW + KSLAB + PSTRIP];
    bf16* Ksm = pool + SKEW;
    const unsigned tid = threadIdx.x, lane = tid & 63u;
    const unsigned fr = lane & 15u, kg = lane >> 4;
    for (unsigned i = tid; i < KSLAB + PSTRIP; i += 256) Ksm[i] = src[i];
    __syncthreads();

    f32x4 oacc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) oacc[t] = (f32x4){0.f, 0.f, 0.f, 0.f};

    bf16x8 pf;
    __builtin_memcpy(&pf, &Ksm[KSLAB + fr * BKV + kg * 8], 16);

    for (unsigned it = 0; it < ntile; it++) {
        if (ARM == 0) {
            bf16x8 vf[2];
            auto vload = [&](int t, int slot) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    bf16 vv = Ksm[krow(kg * 8 + (unsigned)j) + (unsigned)t * 16 + fr];
                    bf16_t vb;
                    __builtin_memcpy(&vb, &vv, 2);
                    vf[slot][j] = vb;
                }
            };
            vload(0, 0);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                if (t + 1 < NT) vload(t + 1, (t + 1) & 1);
                oacc[t] = mfma16(pf, vf[t & 1], oacc[t]);
            }
        } else if (ARM == 1) {
            bf16x8 vf[2];
            auto vload = [&](int t, int slot) {
                bf16x2 p[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    bf16_t lo, hi;
                    bf16 a = Ksm[krow(kg * 8 + (unsigned)(2 * j)) + (unsigned)t * 16 + fr];
                    bf16 b = Ksm[krow(kg * 8 + (unsigned)(2 * j + 1)) + (unsigned)t * 16 + fr];
                    __builtin_memcpy(&lo, &a, 2);
                    __builtin_memcpy(&hi, &b, 2);
                    p[j] = (bf16x2){lo, hi};
                }
                __builtin_memcpy(&vf[slot], p, 16);
            };
            vload(0, 0);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                if (t + 1 < NT) vload(t + 1, (t + 1) & 1);
                oacc[t] = mfma16(pf, vf[t & 1], oacc[t]);
            }
        } else if (ARM == 2) {
            bf16x8 vf[2];
            auto vload = [&](int t, int slot) {
                u16x2 p[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    u16x2 q;
                    q[0] = Ksm[krow(kg * 8 + (unsigned)(2 * j)) + (unsigned)t * 16 + fr];
                    q[1] = Ksm[krow(kg * 8 + (unsigned)(2 * j + 1)) + (unsigned)t * 16 + fr];
                    p[j] = q;
                }
                __builtin_memcpy(&vf[slot], p, 16);
            };
            vload(0, 0);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                if (t + 1 < NT) vload(t + 1, (t + 1) & 1);
                oacc[t] = mfma16(pf, vf[t & 1], oacc[t]);
            }
        } else if (ARM == 3) {
            unsigned r[4];
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const unsigned a = lds_addr(&Ksm[krow(kg * 8) + (unsigned)t * 16 + fr]);
                asm volatile(D16_ISSUE("%[a]") "s_waitcnt lgkmcnt(0)"
                             : "=v"(r[0]), "=v"(r[1]), "=v"(r[2]), "=v"(r[3])
                             : [a] "v"(a), D16_OFFS);
                bf16x8 vf;
                __builtin_memcpy(&vf, r, 16);
                oacc[t] = mfma16(pf, vf, oacc[t]);
            }
        } else if (ARM == 4) {
            unsigned r[2][4]; /* ping-pong; t is compile-time so the index folds away */
            asm volatile(D16_ISSUE("%[a]") "s_waitcnt lgkmcnt(0)"
                         : "=v"(r[0][0]), "=v"(r[0][1]), "=v"(r[0][2]), "=v"(r[0][3])
                         : [a] "v"(lds_addr(&Ksm[krow(kg * 8) + fr])), D16_OFFS);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const int c = t & 1, n = (t + 1) & 1;
                if (t + 1 < NT) {
                    const unsigned a =
                        lds_addr(&Ksm[krow(kg * 8) + (unsigned)(t + 1) * 16 + fr]);
                    asm volatile(D16_ISSUE("%[a]") "s_waitcnt lgkmcnt(8)"
                                 : "=v"(r[n][0]), "=v"(r[n][1]), "=v"(r[n][2]), "=v"(r[n][3]),
                                   "+v"(r[c][0]), "+v"(r[c][1]), "+v"(r[c][2]), "+v"(r[c][3])
                                 : [a] "v"(a), D16_OFFS);
                }
                bf16x8 vf;
                __builtin_memcpy(&vf, r[c], 16);
                oacc[t] = mfma16(pf, vf, oacc[t]);
            }
        } else if (ARM == 5) {
            /* lows (4 regs), DRAIN, highs -- d16_hi merges with the register at ISSUE, so
             * the low load must have retired first (proved by ldsaddr.hip). */
            unsigned r[4];
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const unsigned a = lds_addr(&Ksm[krow(kg * 8) + (unsigned)t * 16 + fr]);
                asm volatile("ds_read_u16 %0, %[a] offset:0\n\t"
                             "ds_read_u16 %1, %[a] offset:%c[o2]\n\t"
                             "ds_read_u16 %2, %[a] offset:%c[o4]\n\t"
                             "ds_read_u16 %3, %[a] offset:%c[o6]\n\t"
                             "s_waitcnt lgkmcnt(0)\n\t"
                             "ds_read_u16_d16_hi %0, %[a] offset:%c[o1]\n\t"
                             "ds_read_u16_d16_hi %1, %[a] offset:%c[o3]\n\t"
                             "ds_read_u16_d16_hi %2, %[a] offset:%c[o5]\n\t"
                             "ds_read_u16_d16_hi %3, %[a] offset:%c[o7]\n\t"
                             "s_waitcnt lgkmcnt(0)"
                             : "=v"(r[0]), "=v"(r[1]), "=v"(r[2]), "=v"(r[3])
                             : [a] "v"(a), D16_OFFS);
                bf16x8 vf;
                __builtin_memcpy(&vf, r, 16);
                oacc[t] = mfma16(pf, vf, oacc[t]);
            }
        } else if (ARM == 6) {
            /* KEEP the perms, delete the DRAINS: all 8 reads issue, ONE wait, then pack. */
            unsigned q[8], r[4];
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const unsigned a = lds_addr(&Ksm[krow(kg * 8) + (unsigned)t * 16 + fr]);
                asm volatile("ds_read_u16 %0, %[a] offset:0\n\t"
                             "ds_read_u16 %1, %[a] offset:%c[o1]\n\t"
                             "ds_read_u16 %2, %[a] offset:%c[o2]\n\t"
                             "ds_read_u16 %3, %[a] offset:%c[o3]\n\t"
                             "ds_read_u16 %4, %[a] offset:%c[o4]\n\t"
                             "ds_read_u16 %5, %[a] offset:%c[o5]\n\t"
                             "ds_read_u16 %6, %[a] offset:%c[o6]\n\t"
                             "ds_read_u16 %7, %[a] offset:%c[o7]\n\t"
                             "s_waitcnt lgkmcnt(0)"
                             : "=v"(q[0]), "=v"(q[1]), "=v"(q[2]), "=v"(q[3]), "=v"(q[4]),
                               "=v"(q[5]), "=v"(q[6]), "=v"(q[7])
                             : [a] "v"(a), D16_OFFS);
#pragma unroll
                for (int k = 0; k < 4; k++) r[k] = q[2 * k] | (q[2 * k + 1] << 16);
                bf16x8 vf;
                __builtin_memcpy(&vf, r, 16);
                oacc[t] = mfma16(pf, vf, oacc[t]);
            }
        } else if (ARM == 7) {
            /* ARM 6 + the SV(3) cross-tile double buffer: issue tile t+1's 8 reads, then
             * wait lgkmcnt(8) to retire tile t's. DS-to-DS returns in order on gfx9, so 8
             * strictly-newer DS ops outstanding forces every older one complete. */
            unsigned q[2][8];
#define D16_R8(A)                                                                        \
    "ds_read_u16 %0, " A " offset:0\n\t"                                                \
    "ds_read_u16 %1, " A " offset:%c[o1]\n\t"                                           \
    "ds_read_u16 %2, " A " offset:%c[o2]\n\t"                                           \
    "ds_read_u16 %3, " A " offset:%c[o3]\n\t"                                           \
    "ds_read_u16 %4, " A " offset:%c[o4]\n\t"                                           \
    "ds_read_u16 %5, " A " offset:%c[o5]\n\t"                                           \
    "ds_read_u16 %6, " A " offset:%c[o6]\n\t"                                           \
    "ds_read_u16 %7, " A " offset:%c[o7]\n\t"
            asm volatile(D16_R8("%[a]") "s_waitcnt lgkmcnt(0)"
                         : "=v"(q[0][0]), "=v"(q[0][1]), "=v"(q[0][2]), "=v"(q[0][3]),
                           "=v"(q[0][4]), "=v"(q[0][5]), "=v"(q[0][6]), "=v"(q[0][7])
                         : [a] "v"(lds_addr(&Ksm[krow(kg * 8) + fr])), D16_OFFS);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const int c = t & 1, n = (t + 1) & 1;
                if (t + 1 < NT) {
                    const unsigned a =
                        lds_addr(&Ksm[krow(kg * 8) + (unsigned)(t + 1) * 16 + fr]);
                    asm volatile(D16_R8("%[a]") "s_waitcnt lgkmcnt(8)"
                                 : "=v"(q[n][0]), "=v"(q[n][1]), "=v"(q[n][2]), "=v"(q[n][3]),
                                   "=v"(q[n][4]), "=v"(q[n][5]), "=v"(q[n][6]), "=v"(q[n][7]),
                                   "+v"(q[c][0]), "+v"(q[c][1]), "+v"(q[c][2]), "+v"(q[c][3]),
                                   "+v"(q[c][4]), "+v"(q[c][5]), "+v"(q[c][6]), "+v"(q[c][7])
                                 : [a] "v"(a), D16_OFFS);
                }
                unsigned r[4];
#pragma unroll
                for (int k = 0; k < 4; k++) r[k] = q[c][2 * k] | (q[c][2 * k + 1] << 16);
                bf16x8 vf;
                __builtin_memcpy(&vf, r, 16);
                oacc[t] = mfma16(pf, vf, oacc[t]);
            }
#undef D16_R8
        }
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < NT; t++)
        out[tid * NT + t] = oacc[t][0] + oacc[t][1] + oacc[t][2] + oacc[t][3];
}

#define HC(x)                                                                            \
    do {                                                                                 \
        hipError_t e = (x);                                                              \
        if (e != hipSuccess) {                                                           \
            printf("FAIL %s: %s\n", #x, hipGetErrorString(e));                           \
            return 2;                                                                    \
        }                                                                                \
    } while (0)

int main() {
    const int NSRC = KSLAB + PSTRIP, NOUT = 256 * NT;
    bf16* hs = (bf16*)malloc(NSRC * 2);
    /* bf16 values in a benign exponent range so the f32 accumulation is well-conditioned
     * -- the gate is bit-equality between arms, and NaN/Inf would make it vacuous. */
    unsigned s = 12345;
    for (int i = 0; i < NSRC; i++) {
        s = s * 1664525u + 1013904223u;
        hs[i] = (bf16)(0x3a00 | ((s >> 16) & 0x1ff)) ^ (bf16)((s & 1) << 15);
    }
    bf16* ds_;
    float* dout;
    HC(hipMalloc(&ds_, NSRC * 2));
    HC(hipMalloc(&dout, NOUT * 4));
    HC(hipMemcpy(ds_, hs, NSRC * 2, hipMemcpyHostToDevice));
    float* ref = (float*)malloc(NOUT * 4);
    float* got = (float*)malloc(NOUT * 4);
    int bad = 0;
    for (int arm = 0; arm < 8; arm++) {
        HC(hipMemset(dout, 0, NOUT * 4));
        switch (arm) {
        case 0: probe<0><<<1, 256>>>(ds_, dout, 3); break;
        case 1: probe<1><<<1, 256>>>(ds_, dout, 3); break;
        case 2: probe<2><<<1, 256>>>(ds_, dout, 3); break;
        case 3: probe<3><<<1, 256>>>(ds_, dout, 3); break;
        case 4: probe<4><<<1, 256>>>(ds_, dout, 3); break;
        case 5: probe<5><<<1, 256>>>(ds_, dout, 3); break;
        case 6: probe<6><<<1, 256>>>(ds_, dout, 3); break;
        case 7: probe<7><<<1, 256>>>(ds_, dout, 3); break;
        }
        HC(hipDeviceSynchronize());
        HC(hipMemcpy(arm ? got : ref, dout, NOUT * 4, hipMemcpyDeviceToHost));
        if (arm) {
            int d = 0;
            for (int i = 0; i < NOUT; i++)
                if (((unsigned*)got)[i] != ((unsigned*)ref)[i]) d++;
            printf("arm %d vs arm 0: %s (%d/%d words differ)\n", arm, d ? "MISMATCH" : "BIT-IDENTICAL",
                   d, NOUT);
            bad += (d != 0);
        } else {
            printf("arm 0 reference: out[0]=%g out[7]=%g\n", ref[0], ref[7]);
        }
    }
    printf(bad ? "\nGATE FAIL\n" : "\nGATE PASS: every arm bit-identical to the shipped form\n");
    return bad ? 1 : 0;
}
