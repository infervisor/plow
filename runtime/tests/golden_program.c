/* golden_program — exercises the CPU golden kernels two ways:
 *   1. direct ref-math checks against hand-computed expected values (the oracle),
 *   2. an end-to-end stream: hand-build a `.pkt` (same layout as
 *      Program::to_bytes), decode it, run it through the interpreter + dispatch
 *      table, and confirm the GEMM output matches the oracle.
 */
#include "packet.h"
#include "decode.h"
#include "dispatch.h"
#include "interp.h"
#include "kernel.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); g_fail = 1; } } while (0)
static int close_f(float a, float b) { return fabsf(a - b) < 1e-4f; }

/* --- 1. direct ref-math checks ------------------------------------------- */

static void check_gemm(void) {
    /* m=1,n=2,k=2; a=[1,2]; b row-major [n,k]=[[1,0],[0,1]] -> c=[1,2] */
    float a[2] = {1, 2};
    float b[4] = {1, 0, 0, 1};
    float c[2] = {0, 0};
    plow_gemm_ref(c, a, b, NULL, 1, 2, 2, PLOW_ACT_NONE);
    CHECK(close_f(c[0], 1) && close_f(c[1], 2), "gemm identity");

    float bias[2] = {10, 20};
    plow_gemm_ref(c, a, b, bias, 1, 2, 2, PLOW_ACT_RELU);
    CHECK(close_f(c[0], 11) && close_f(c[1], 22), "gemm bias+relu");
}

static void check_flash(void) {
    /* heads=1, sq=1, skv=2, hd=1, q=[0], k=[[0],[0]] -> uniform softmax -> mean(v) */
    float q[1] = {0};
    float k[2] = {0, 0};
    float v[2] = {2, 6};
    float o[1] = {0};
    plow_flash_ref(o, q, k, v, 1, 2, 1, 1, 0);
    CHECK(close_f(o[0], 4), "flash uniform -> mean(v)");
}

static void check_row(void) {
    float x[2] = {1, 1};
    float out[2] = {0, 0};
    plow_row_reduce_ref(out, x, NULL, 1, 2, PLOW_NORM_SOFTMAX, 0);
    CHECK(close_f(out[0], 0.5f) && close_f(out[1], 0.5f), "softmax uniform");

    float x2[2] = {3, 4};
    plow_row_reduce_ref(out, x2, NULL, 1, 2, PLOW_NORM_RMS, 0);
    /* rms = sqrt((9+16)/2)=sqrt(12.5)=3.5355; out=x/rms */
    CHECK(close_f(out[0], 3.0f / 3.535534f) && close_f(out[1], 4.0f / 3.535534f), "rmsnorm");

    float a[3] = {-1, 0, 2};
    float pw[3] = {0, 0, 0};
    plow_row_pointwise_ref(pw, a, NULL, 3, PLOW_ACT_RELU, 0);
    CHECK(close_f(pw[0], 0) && close_f(pw[1], 0) && close_f(pw[2], 2), "relu pointwise");

    float bb[3] = {1, 1, 1};
    plow_row_pointwise_ref(pw, a, bb, 3, 0, PLOW_EW_ADD);
    CHECK(close_f(pw[0], 0) && close_f(pw[1], 1) && close_f(pw[2], 3), "elementwise add");
}

/* --- 2. end-to-end stream builder ---------------------------------------- */

typedef struct { uint8_t* p; size_t len, cap; } Buf;
static void put(Buf* b, const void* src, size_t n) {
    memcpy(b->p + b->len, src, n);
    b->len += n;
}
static void put_u32(Buf* b, uint32_t v) { put(b, &v, 4); }
static void put_u16(Buf* b, uint16_t v) { put(b, &v, 2); }

static void check_end_to_end(void) {
    /* One GEMM inst writing slot 2 from slots 0 (A) and 1 (B). */
    const uint32_t M = 2, N = 2, K = 3;
    float A[6] = {1, 2, 3, 4, 5, 6};        /* [M,K] */
    float B[6] = {1, 0, 0, 0, 1, 0};        /* [N,K] row-major */
    float C[4] = {0};
    float expect[4];
    plow_gemm_ref(expect, A, B, NULL, M, N, K, PLOW_ACT_NONE);

    uint8_t storage[256];
    Buf b = { storage, 0, sizeof(storage) };
    /* stream header (20 B) */
    put_u32(&b, PLOW_MAGIC);
    put_u16(&b, PLOW_VERSION);
    put_u16(&b, 0);          /* bucket_id */
    put_u32(&b, 1);          /* n_insts */
    put_u32(&b, 0);          /* n_counters */
    put_u16(&b, 0);          /* plan_gen */
    put_u16(&b, 0);          /* flags */

    /* record: header + GemmBody (no wait/succ) */
    PlowHeader h = { PLOW_GEMM, PLOW_RES_SM, 0, 0, 0, 0 };
    put(&b, &h, sizeof(h));
    PlowGemmBody g = { 0, 0, M, N, K, (uint16_t)M, (uint16_t)N, (uint16_t)K, /*out*/2, PLOW_SLOT_NONE, 0 };
    put(&b, &g, sizeof(g));

    void* slots[3] = { A, B, C };
    kctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.slots = slots;
    ctx.n_slots = 3;
    ctx.counters = NULL;
    ctx.n_counters = 0;

    PlowBinding bind = { 0, 1, PLOW_SLOT_NONE, PLOW_ACT_NONE, 0, 0.0f, 0 };
    /* bindings indexed by inst.index; our single inst has index 0 */
    PlowBinding bindings[1] = { bind };

    dispatch_table dt;
    plow_register_cpu(&dt);

    int rc = plow_interp_run(b.p, b.len, &dt, &ctx, bindings, 1);
    CHECK(rc == 0, "interp_run returns 0");
    int eq = 1;
    for (int i = 0; i < 4; i++) eq &= close_f(C[i], expect[i]);
    CHECK(eq, "end-to-end GEMM matches oracle");
}

/* Build a single-instruction stream (no wait/succ, no counters) into `b`. */
static void build_1inst(Buf* b, uint16_t opcode, const void* body, size_t bsz) {
    put_u32(b, PLOW_MAGIC);
    put_u16(b, PLOW_VERSION);
    put_u16(b, 0);          /* bucket_id */
    put_u32(b, 1);          /* n_insts */
    put_u32(b, 0);          /* n_counters */
    put_u16(b, 0);          /* plan_gen */
    put_u16(b, 0);          /* flags */
    PlowHeader h = { opcode, PLOW_RES_SM, 0, 0, 0, 0, 0 };
    put(b, &h, sizeof(h));
    if (bsz) put(b, body, bsz);
}

/* Exercise FLASH/ROW dispatch end-to-end (these families are now registered on
 * every backend; here we confirm the CPU wiring matches the oracle). */
static void check_e2e_flash(void) {
    float q[1] = {0}, k[2] = {0, 0}, v[2] = {2, 6}, o[1] = {0};
    uint8_t storage[256];
    Buf b = { storage, 0, sizeof(storage) };
    PlowFlashBody f = { 0, 0, /*seq_q*/1, /*seq_kv*/2, /*head_dim*/1, 0, 0, /*heads*/1, /*out*/3, 0 };
    build_1inst(&b, PLOW_FLASH, &f, sizeof(f));

    void* slots[4] = { q, k, v, o };
    kctx ctx; memset(&ctx, 0, sizeof(ctx));
    ctx.slots = slots; ctx.n_slots = 4;
    PlowBinding bind = { 0, 1, 2, /*detail=non-causal*/0, 0, 0.0f, 0 };
    PlowBinding bindings[1] = { bind };
    dispatch_table dt; plow_register_cpu(&dt);
    int rc = plow_interp_run(b.p, b.len, &dt, &ctx, bindings, 1);
    CHECK(rc == 0 && close_f(o[0], 4), "end-to-end FLASH matches oracle");
}

static void check_e2e_row_reduce(void) {
    float x[2] = {1, 1}, o[2] = {0, 0};
    uint8_t storage[256];
    Buf b = { storage, 0, sizeof(storage) };
    PlowRowBody r = { .coord=0, .rows=1, .feat=2, .args={0,0,0,0},
                      .br=0, .out=1, .operands=1, ._pad={0,0,0} };
    build_1inst(&b, PLOW_ROW_REDUCE, &r, sizeof(r));

    void* slots[2] = { x, o };
    kctx ctx; memset(&ctx, 0, sizeof(ctx));
    ctx.slots = slots; ctx.n_slots = 2;
    PlowBinding bind = { 0, PLOW_SLOT_NONE, PLOW_SLOT_NONE, PLOW_NORM_SOFTMAX, 0, 0.0f, 0 };
    PlowBinding bindings[1] = { bind };
    dispatch_table dt; plow_register_cpu(&dt);
    int rc = plow_interp_run(b.p, b.len, &dt, &ctx, bindings, 1);
    CHECK(rc == 0 && close_f(o[0], 0.5f) && close_f(o[1], 0.5f), "end-to-end ROW_REDUCE softmax");
}

static void check_e2e_row_pointwise(void) {
    float a[3] = {-1, 0, 2}, o[3] = {0, 0, 0};
    uint8_t storage[256];
    Buf b = { storage, 0, sizeof(storage) };
    PlowRowBody r = { .coord=0, .rows=1, .feat=3, .args={0,0,0,0},
                      .br=0, .out=1, .operands=1, ._pad={0,0,0} };
    build_1inst(&b, PLOW_ROW_POINTWISE, &r, sizeof(r));

    void* slots[2] = { a, o };
    kctx ctx; memset(&ctx, 0, sizeof(ctx));
    ctx.slots = slots; ctx.n_slots = 2;
    PlowBinding bind = { 0, PLOW_SLOT_NONE, PLOW_SLOT_NONE, PLOW_ACT_RELU, 0, 0.0f, 0 };
    PlowBinding bindings[1] = { bind };
    dispatch_table dt; plow_register_cpu(&dt);
    int rc = plow_interp_run(b.p, b.len, &dt, &ctx, bindings, 1);
    CHECK(rc == 0 && close_f(o[0], 0) && close_f(o[1], 0) && close_f(o[2], 2),
          "end-to-end ROW_POINTWISE relu");
}

/* A control op (NOP) needs no binding and must succeed; a compute op (GEMM)
 * with no binding must fail loud with -4 (interp guard), not silently no-op. */
static void check_binding_guard(void) {
    uint8_t storage[256];
    dispatch_table dt; plow_register_cpu(&dt);

    Buf nop = { storage, 0, sizeof(storage) };
    build_1inst(&nop, PLOW_NOP, NULL, 0);
    kctx ctx; memset(&ctx, 0, sizeof(ctx));
    CHECK(plow_interp_run(nop.p, nop.len, &dt, &ctx, NULL, 0) == 0, "NOP runs without a binding");

    float dummy[4] = {0};
    uint8_t storage2[256];
    Buf gemm = { storage2, 0, sizeof(storage2) };
    PlowGemmBody g = { 0, 0, 1, 1, 1, 1, 1, 1, 0, PLOW_SLOT_NONE, 0 };
    build_1inst(&gemm, PLOW_GEMM, &g, sizeof(g));
    void* slots[1] = { dummy };
    kctx c2; memset(&c2, 0, sizeof(c2));
    c2.slots = slots; c2.n_slots = 1;
    CHECK(plow_interp_run(gemm.p, gemm.len, &dt, &c2, NULL, 0) == -4,
          "GEMM without a binding fails with -4");
}

/* LAYOUT: kind==1 transpose of a 2x3 -> 3x2, and kind==0 contiguous copy. */
static void check_e2e_layout(void) {
    dispatch_table dt;
    plow_register_cpu(&dt);

    /* transpose: in (2,3) row-major -> out (3,2). out[i,j] = in[j,i]. */
    float X[6] = {1, 2, 3, 4, 5, 6};
    float O[6] = {0};
    float expect[6] = {1, 4, 2, 5, 3, 6};
    uint8_t storage[256];
    Buf b = { storage, 0, sizeof(storage) };
    PlowLayoutBody L = {0};
    L.kind = 1; L.rank = 2; L.elem_size = sizeof(float); L.out = 1;
    L.shape[0] = 3; L.shape[1] = 2;       /* output extents */
    L.in_stride[0] = 1; L.in_stride[1] = 3; /* read in[j,i] */
    L.out_stride[0] = 2; L.out_stride[1] = 1;
    build_1inst(&b, PLOW_LAYOUT, &L, sizeof(L));
    void* slots[2] = { X, O };
    kctx ctx; memset(&ctx, 0, sizeof(ctx));
    ctx.slots = slots; ctx.n_slots = 2;
    PlowBinding bind = { 0, PLOW_SLOT_NONE, PLOW_SLOT_NONE, 0, 0, 0.0f, 0 };
    PlowBinding bindings[1] = { bind };
    int rc = plow_interp_run(b.p, b.len, &dt, &ctx, bindings, 1);
    int eq = (rc == 0);
    for (int i = 0; i < 6; i++) eq &= close_f(O[i], expect[i]);
    CHECK(eq, "end-to-end LAYOUT transpose 2x3->3x2");

    /* contiguous copy (kind==0): byte-for-byte. */
    float Y[4] = {7, 8, 9, 10};
    float C2[4] = {0};
    Buf b2 = { storage, 0, sizeof(storage) };
    PlowLayoutBody Lc = {0};
    Lc.kind = 0; Lc.rank = 1; Lc.elem_size = 1; Lc.out = 1;
    Lc.shape[0] = sizeof(Y);              /* total bytes */
    Lc.in_stride[0] = 1; Lc.out_stride[0] = 1;
    build_1inst(&b2, PLOW_LAYOUT, &Lc, sizeof(Lc));
    void* slots2[2] = { Y, C2 };
    kctx ctx2; memset(&ctx2, 0, sizeof(ctx2));
    ctx2.slots = slots2; ctx2.n_slots = 2;
    int rc2 = plow_interp_run(b2.p, b2.len, &dt, &ctx2, bindings, 1);
    int eq2 = (rc2 == 0);
    for (int i = 0; i < 4; i++) eq2 &= close_f(C2[i], Y[i]);
    CHECK(eq2, "end-to-end LAYOUT contiguous copy");
}

/* LAYOUT kind==2 binary concat: inner axis (interleave) and outer axis (adjacency). */
static void check_e2e_concat(void) {
    dispatch_table dt;
    plow_register_cpu(&dt);
    uint8_t storage[256];

    float A[4] = {1, 2, 3, 4}; /* (2,2) */
    float B[4] = {5, 6, 7, 8}; /* (2,2) */
    PlowBinding bind = { 0, 1, PLOW_SLOT_NONE, 0, 0, 0.0f, 0 }; /* in0=A, in1=B */
    PlowBinding bindings[1] = { bind };

    /* axis 1 (inner) -> (2,4) interleaved by row. */
    float O1[8] = {0};
    float exp1[8] = {1, 2, 5, 6, 3, 4, 7, 8};
    Buf b1 = { storage, 0, sizeof(storage) };
    PlowLayoutBody L1 = {0};
    L1.kind = 2; L1.rank = 2; L1.elem_size = sizeof(float); L1.out = 2;
    L1.shape[0] = 2; L1.shape[1] = 4;
    L1.out_stride[0] = 4; L1.out_stride[1] = 1;
    L1.in_base = 1;  /* axis */
    L1.out_base = 2; /* split = A's extent along axis 1 */
    build_1inst(&b1, PLOW_LAYOUT, &L1, sizeof(L1));
    void* s1[3] = { A, B, O1 };
    kctx c1; memset(&c1, 0, sizeof(c1)); c1.slots = s1; c1.n_slots = 3;
    int rc1 = plow_interp_run(b1.p, b1.len, &dt, &c1, bindings, 1);
    int eq1 = (rc1 == 0); for (int i = 0; i < 8; i++) eq1 &= close_f(O1[i], exp1[i]);
    CHECK(eq1, "end-to-end LAYOUT concat axis-1 (interleave)");

    /* axis 0 (outer) -> (4,2) adjacency. */
    float O0[8] = {0};
    float exp0[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    Buf b0 = { storage, 0, sizeof(storage) };
    PlowLayoutBody L0 = {0};
    L0.kind = 2; L0.rank = 2; L0.elem_size = sizeof(float); L0.out = 2;
    L0.shape[0] = 4; L0.shape[1] = 2;
    L0.out_stride[0] = 2; L0.out_stride[1] = 1;
    L0.in_base = 0;  /* axis */
    L0.out_base = 2; /* split = A's extent along axis 0 */
    build_1inst(&b0, PLOW_LAYOUT, &L0, sizeof(L0));
    void* s0[3] = { A, B, O0 };
    kctx c0; memset(&c0, 0, sizeof(c0)); c0.slots = s0; c0.n_slots = 3;
    int rc0 = plow_interp_run(b0.p, b0.len, &dt, &c0, bindings, 1);
    int eq0 = (rc0 == 0); for (int i = 0; i < 8; i++) eq0 &= close_f(O0[i], exp0[i]);
    CHECK(eq0, "end-to-end LAYOUT concat axis-0 (adjacency)");
}

int main(void) {
    check_gemm();
    check_flash();
    check_row();
    check_end_to_end();
    check_e2e_flash();
    check_e2e_row_reduce();
    check_e2e_row_pointwise();
    check_e2e_layout();
    check_e2e_concat();
    check_binding_guard();
    printf("golden_program: %s\n", g_fail ? "FAIL" : "ok");
    return g_fail;
}
