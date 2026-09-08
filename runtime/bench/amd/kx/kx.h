/* kx — a single-block kernel-experiment harness.
 *
 * WHY THIS EXISTS. Every kernel question on this branch used to cost a campaign: rebuild 45
 * objects, emit a blob, take a lease, start a server, run a serving benchmark, then a separate
 * greedy-identity run. 20-60 minutes for a one-line answer, and one whole result was a FALSE NULL
 * because the candidate was measured against a stale binary. The questions themselves are local:
 * "does this tile spill?", "is R=2 better than R=4 here?", "how many ds_bpermute does this
 * softmax issue?". This asks them against ONE WORKGROUP, in seconds.
 *
 * ONE FILE PER EXPERIMENT. An experiment is a single .hip compiled TWICE:
 *   hipcc --genco    -> the device arms          (this header's !KX_HOST side)
 *   g++  -DKX_HOST   -> the `kx_experiment()` descriptor the driver reads (KX_HOST side)
 * so the arms and the thing that describes them cannot drift apart. See kx/README.md.
 *
 * WHAT THE HARNESS GUARANTEES, so an experiment does not have to re-earn it:
 *   * A/A CONTROL. Every experiment names an arm that is byte-identical device code to the
 *     baseline under a second name. The driver REFUSES to print ratios when it reads outside
 *     tolerance -- an unpinned run on a contended box has read A/A 0.27..4.57 on this machine.
 *   * PALINDROMIC INTERLEAVE. Forward then reverse pass, min per arm, all arms in ONE process:
 *     a monotone drift (clock ramp, a neighbour arriving) then biases both directions equally.
 *   * A DEVICE-SIDE CORRECTNESS GATE THAT IS NOT OPTIONAL. Every non-baseline arm is compared
 *     elementwise on device against the experiment's reference (shipped) arm. A candidate that is
 *     faster and wrong is reported WRONG, never as a win -- BKV=40 once looked like a win and was
 *     producing garbage, and a PLOW_MLA_NS_LIVE candidate produced a clean 20/20-identical null
 *     because it was silently not firing.
 *   * TWO GEOMETRIES, both labelled. `1wg` measures the body on its own; `model` measures it at
 *     the emitter's own workgroup count. They DISAGREE -- a 9% standalone spread became 16% in
 *     the megakernel and the standalone winner was not the model winner -- so every row says
 *     which it is and no single-block win may be generalised without the other column.
 */
#ifndef KX_H
#define KX_H

/* ------------------------------------------------------------------ shared descriptor types */

enum {
    KX_BASE = 1u << 0,    /* the denominator every ratio is taken against */
    KX_AA = 1u << 1,      /* byte-identical device code to the baseline; the noise floor */
    KX_REF = 1u << 2,     /* the shipped body; correctness is measured against THIS */
    KX_EXACT = 1u << 3,   /* arithmetic-preserving by construction: demand bit-identical output */
    KX_NOCHECK = 1u << 4, /* attribution arm with no comparable output (e.g. stage-only) */
    KX_SLOWOK = 1u << 5   /* expected to lose; a regression here is the RESULT, not a failure */
};

enum { KX_DT_BF16 = 0, KX_DT_F32 = 1 };

typedef struct {
    const char* label;   /* variant name, e.g. "mm4" */
    const char* defines; /* extra -D flags for THIS variant's code object */
} kx_variant;

typedef struct {
    const char* sym;  /* extern "C" __global__ symbol name */
    int variant;      /* index into kx_exp::var */
    unsigned flags;   /* KX_BASE | KX_AA | ... */
    const char* note; /* one line: what this arm changes, and nothing else */
} kx_arm;

/* d0/d1/d2 are passed straight to the kernel; their meaning belongs to the experiment. For the
 * GEMV family they are N, K and a spare. `grid` is THE EMITTER'S OWN workgroup count for this
 * shape (`plowrt disasm <blob>/model.pkt --program N`), never a CU count -- the K3 router GEMV
 * runs 224 and b_proj runs 12, so it has to come from the packet. */
typedef struct {
    const char* name;
    unsigned d0, d1, d2;
    unsigned grid;      /* model geometry */
    unsigned inst;      /* instances per token, for a per-token total (0 -> omit) */
    double slab_bytes;  /* bytes the in-kernel rep loop walks per rep; 0 -> no arena walk */
    unsigned reps;      /* fixed rep count; 0 -> derived from slab_bytes and the arena */
    unsigned out_elems; /* elements the correctness gate compares */
} kx_shape;

typedef struct {
    const char* name;
    const char* question; /* the one-line question this experiment answers */
    const char* source;   /* path of the .hip, relative to the repo root */
    /* THE SHIPPED OBJECT'S OWN DEFINES for the arm being studied, verbatim from the recipe in
     * scripts/build_gfx942.sh. They are not decoration: they participate in the digest the tuning
     * store keys on (kernelcaps::BuildId::label), and they gate which arms of op_gemm.h /
     * op_attention.h instantiate at all. A bench built without them measures a different kernel. */
    const char* defines;
    const char* arch;     /* --offload-arch, e.g. "gfx942" */
    unsigned threads;     /* blockDim; must match the arms' __launch_bounds__ */
    int dtype;            /* KX_DT_* of the compared output */
    const kx_variant* var;
    int nvar;
    const kx_arm* arm;
    int narm;
    const kx_shape* shp;
    int nshp;
    double aa_lo, aa_hi; /* A/A tolerance; outside it the run REFUSES to report ratios */
    double tol_max;      /* max abs error tolerated for arms without KX_EXACT */
    const char* expect;  /* the known answer, so a reproduction is checkable at a glance */
} kx_exp;

#ifdef __cplusplus
extern "C"
#endif
    const kx_exp*
    kx_experiment(void);

/* --------------------------------------------------------------------------- device side */
#ifndef KX_HOST

#include <hip/hip_runtime.h>

#include "amd_common.h"

#ifndef KX_THREADS
#define KX_THREADS PLOW_THREADS
#endif

/* The canonical arm signature. Uniform across every experiment so the driver can launch any arm
 * without knowing what it does: three inputs, one output, a rep count, the partition width and
 * three dims.
 *
 * kx_nblk IS ALWAYS THE SHAPE'S MODEL WORKGROUP COUNT, never gridDim.x, and an arm must pass it
 * as the `nblk` its body partitions on. That is what makes the 1wg geometry mean "one workgroup
 * doing THE WORK ONE WORKGROUP DOES IN THE MODEL" rather than "one workgroup doing the entire
 * shape" -- the latter is 300x the work at these grids, measures a different question, and takes
 * minutes instead of seconds. `slice` stays blockIdx.x, so a 1wg launch computes slice 0 only. */
#define KX_SIG                                                                                     \
    void *kx_out, const void *kx_w, const void *kx_x, const void *kx_aux, unsigned kx_nrep,        \
        unsigned kx_nblk, unsigned kx_d0, unsigned kx_d1, unsigned kx_d2
#define KX_FWD kx_out, kx_w, kx_x, kx_aux, kx_nrep, kx_nblk, kx_d0, kx_d1, kx_d2

/* A BODY is __forceinline__ so instantiating it under two names gives BYTE-IDENTICAL device code.
 * That is the whole mechanism behind the A/A control: KX_ARM_OF(k_base, b) and
 * KX_ARM_OF(k_base_aa, b) differ in the symbol and in nothing else. */
#define KX_BODY(name) __device__ __forceinline__ void name(KX_SIG)
#define KX_ARM_OF(armname, bodyname)                                                               \
    extern "C" __global__ __launch_bounds__(KX_THREADS) void armname(KX_SIG) {                     \
        bodyname(KX_FWD);                                                                          \
    }
#define KX_ARM(name) extern "C" __global__ __launch_bounds__(KX_THREADS) void name(KX_SIG)

#define KX_UNUSED                                                                                  \
    do {                                                                                           \
        (void)kx_out;                                                                              \
        (void)kx_w;                                                                                \
        (void)kx_x;                                                                                \
        (void)kx_aux;                                                                              \
        (void)kx_nrep;                                                                             \
        (void)kx_nblk;                                                                             \
        (void)kx_d0;                                                                            \
        (void)kx_d1;                                                                               \
        (void)kx_d2;                                                                               \
    } while (0)

/* ---- the correctness gate, ON DEVICE. acc[0] max|a-b|, acc[1] sum (a-b)^2, acc[2] exact
 * matches, acc[3] non-finite outputs. Compiled into every experiment's code object, so the
 * comparison never leaves the card and never costs a device-to-host copy of the whole output. */
__device__ __forceinline__ void kx_acc(float* acc, float d, int exact, int bad) {
    atomicMax((int*)&acc[0], __float_as_int(d)); /* d >= 0, so the int order IS the float order */
    atomicAdd(&acc[1], d * d);
    atomicAdd(&acc[2], (float)exact);
    atomicAdd(&acc[3], (float)bad);
}

extern "C" __global__ void kx_cmp_bf16(const unsigned short* a, const unsigned short* b, unsigned n,
                                       float* acc) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float fa = __uint_as_float((unsigned)a[i] << 16);
        const float fb = __uint_as_float((unsigned)b[i] << 16);
        const float d = fabsf(fa - fb);
        kx_acc(acc, (isfinite(d) ? d : 3.4e38f), (int)(a[i] == b[i]), (int)(!isfinite(fa)));
    }
}

/* Arena fill, ON DEVICE. A 1.5 GiB hipMemcpy fill costs seconds of PCIe and this harness's whole
 * claim is "seconds"; the pattern is index-derived so it is byte-reproducible across runs.
 * Magnitudes sit near exponent 0x3B so bf16 dot products over K=7168 stay finite. */
extern "C" __global__ void kx_fill_u16(unsigned short* p, unsigned long long n, unsigned seed) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned st = (unsigned)(i * 2654435761u) ^ seed;
        st = st * 1664525u + 1013904223u;
        p[i] = (unsigned short)(0x3B00u | ((st >> 20) & 0x7Fu));
    }
}

extern "C" __global__ void kx_cmp_f32(const float* a, const float* b, unsigned n, float* acc) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float d = fabsf(a[i] - b[i]);
        kx_acc(acc, (isfinite(d) ? d : 3.4e38f),
               (int)(__float_as_uint(a[i]) == __float_as_uint(b[i])), (int)(!isfinite(a[i])));
    }
}

#endif /* !KX_HOST */
#endif /* KX_H */
