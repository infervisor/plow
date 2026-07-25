/* kernel.h — uniform kernel entry, execution context, and golden ref math.
 *
 * Every kernel (golden or performant, any backend) has the same C entry shape
 * `kernel_fn`, so the dispatch table is one flat array of function pointers.
 * A kernel casts `body` to its family's struct (PlowGemmBody*, ...) and reads
 * everything else it needs from `kctx`.
 */
#ifndef PLOW_KERNEL_H
#define PLOW_KERNEL_H

#include <stddef.h>
#include <stdint.h>
#include "packet.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Op subtype carried out-of-band (the wire body doesn't encode silu-vs-gelu or
 * rms-vs-layernorm). The real loader derives these from the schedule; the host
 * harness supplies them via PlowBinding. */
enum { PLOW_ACT_NONE = 0, PLOW_ACT_SILU, PLOW_ACT_GELU, PLOW_ACT_GELU_TANH,
       PLOW_ACT_RELU, PLOW_ACT_SIGMOID, PLOW_ACT_QUICK_GELU };
enum { PLOW_NORM_RMS = 0, PLOW_NORM_LAYER, PLOW_NORM_SOFTMAX };
enum { PLOW_EW_ADD = 0, PLOW_EW_SUB, PLOW_EW_MUL, PLOW_EW_DIV };

/* Operand binding for one instruction (host-harness convention). Compute ops
 * read inputs from these slot handles; data-movement ops use in0 as source. */
typedef struct PlowBinding {
    uint16_t in0, in1, in2; /* input slot handles, PLOW_SLOT_NONE if unused */
    uint8_t  detail;        /* act/norm/ew subtype (family-dependent) */
    uint8_t  _pad;
    float    scale;         /* eps / rope theta / scale factor, family-dependent */
    uint32_t bytes;         /* byte count for LAYOUT copies (body lacks it) */
} PlowBinding;

/* Execution context. Golden/CPU tier is f32; `slots`/`tensors` hold f32 buffers.
 * `stream` is a CUDA/HIP stream on GPU backends, NULL on CPU. */
typedef struct kctx {
    void**    slots;       /* SRAM-slot buffers, indexed by slot handle */
    uint32_t  n_slots;
    void**    tensors;     /* HBM tensors, indexed by tensor handle */
    uint32_t  n_tensors;
    uint32_t* counters;    /* dependency counters, indexed by counter id */
    uint32_t  n_counters;
    const PlowBinding* bind; /* binding for the current inst (NULL if none) */
    void*     stream;      /* backend stream handle; NULL on CPU */
} kctx;

/* The one entry shape every kernel implements. `body` points into the decoded
 * record (cast to the family struct); `ctx` carries slots/tensors/binding. */
typedef void (*kernel_fn)(const void* body, kctx* ctx);

/* --- Golden reference math (f32, correctness-first, used by CPU kernels and
 *     directly by tests as the oracle) ------------------------------------- */

/* c[m,n] = a[m,k] · b[n,k]^T (+ bias[n]), then act. bias may be NULL. */
void plow_gemm_ref(float* c, const float* a, const float* b, const float* bias,
                   uint32_t m, uint32_t n, uint32_t k, int act);

/* o[heads,sq,hd] = softmax(q·k^T / sqrt(hd) + mask) · v, per head. */
void plow_flash_ref(float* o, const float* q, const float* k, const float* v,
                    uint32_t sq, uint32_t skv, uint32_t hd, uint32_t heads, int causal);

/* Row reduce: rmsnorm/layernorm/softmax over `feat` per row. gamma may be NULL. */
void plow_row_reduce_ref(float* out, const float* x, const float* gamma,
                         uint32_t rows, uint32_t feat, int norm, float eps);

/* Row pointwise: act(a) when b==NULL, else elementwise a (op) b, over n elems. */
void plow_row_pointwise_ref(float* out, const float* a, const float* b,
                            uint32_t n, int act, int ew);

/* Byte copy (LAYOUT / DMA / RDMA golden model). */
void plow_copy_ref(void* dst, const void* src, uint32_t bytes);

/* --- Per-backend kernel registration -------------------------------------- */
struct dispatch_table; /* fwd decl from dispatch.h */
void plow_register_cpu(struct dispatch_table* dt);

#ifdef __cplusplus
}
#endif

#endif /* PLOW_KERNEL_H */
