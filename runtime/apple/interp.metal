// interp.metal — the device-ISA interpreter for Apple GPUs (plans/apple-silicon-backend.md §4.2).
//
// One threadgroup per virtual CU (grid = n_cu, residency required), 1024 threads = 32
// simdgroups. Legacy vector/head kernels use their own subgroup packing. Each group walks its stream of
// (inst, slice) entries for the segment the host dispatched:
//
//     thread 0 spins until every wait-counter reaches its threshold
//     all threads execute the op, computing only the `slice`-th share of its work
//     threadgroup_barrier(mem_device)
//     thread 0 bumps every successor counter (device-scope relaxed atomics; probe (c))
//
// Structures mirror runtime/common/dev_isa.h / crates/packet/src/dev.rs byte for byte. Op
// semantics mirror runtime/cpu/dev/golden/*.c (the same slicing contract every interpreter honours);
// the golden file is the reference for every operand slot below.
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

struct Inst {            // DevInst64, 64 bytes
    ushort op, blocks;
    uint fj[3];
    ushort t[8];
    uint i[8];
};
struct Ent {             // StreamEnt, 24 bytes
    uint inst, slice, wait_ofs, succ_ofs;
    ushort wait_len, succ_len, flags, seg;
};
struct Wait { uint id, threshold; };
struct Params { uint seg, n_cu, spin_max, inst_lo, inst_hi; };  // entries with inst outside [lo, hi) are skipped

constant ushort TNONE = 0xFFFFu;
constant uint NT = 1024;     // threads per threadgroup: 32 simdgroups on one core
constant uint NSG = 32;      // simdgroups
constant uint WAVES = 8;     // G_WAVES: (token, head) items packed per workgroup by the emitter
constant float NEG_INF = -3.0e38f;

// ---- bf16 / e4m3 -------------------------------------------------------------------------------
inline float bf2f(ushort h) { return as_type<float>(uint(h) << 16); }
inline ushort f2bf(float f) {
    uint u = as_type<uint>(f);
    if ((u & 0x7F800000u) == 0x7F800000u) return ushort((u >> 16) | ((u & 0xFFFFu) ? 0x40u : 0u));
    uint lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return ushort(u >> 16);
}
inline float rbf(float f) { return bf2f(f2bf(f)); }
inline uchar f2e4m3(float x) {
    uint u = as_type<uint>(x), sign = (u >> 24) & 0x80u;
    u &= 0x7fffffffu;
    if (u > 0x7f800000u) return uchar(sign | 0x7fu);
    float v = as_type<float>(u);
    if (v >= 448.0f) return uchar(sign | 0x7eu);
    if (v < 0x1p-6f) return uchar(sign | uint(rint(v * 512.0f)));
    u += 0x7ffffu + ((u >> 20) & 1u);
    return uchar(sign | ((u - 0x3c000000u) >> 20));
}
// Decode before multiplication: delayed exponent rebias loses valid products to FP32 flush-to-zero.
inline float e4m3_exact(uint c) {
    uint mag = c & 0x7fu;
    float v = mag < 8u ? float(mag) * 0x1p-9f
        : (mag == 0x7fu ? NAN : as_type<float>((mag << 20) + 0x3c000000u));
    return (c & 0x80u) ? -v : v;
}
inline float e4m3(uint c) { return e4m3_exact(c); }
inline float4 e4m3x4(uchar4 c) {
    return float4(e4m3_exact(c.x), e4m3_exact(c.y), e4m3_exact(c.z), e4m3_exact(c.w));
}
// Every finite E4M3 value is exact in BF16; no rounding step is needed.
inline ushort e4m3bf(uint c) {
    uint mag = c & 0x7fu;
    uint bits = mag < 8u ? (as_type<uint>(float(mag) * 0x1p-9f) >> 16)
        : (mag == 0x7fu ? 0x7fc0u : (mag << 4) + 0x3c00u);
    return ushort(bits | ((c & 0x80u) << 8));
}
inline ushort4 e4m3bf4(uchar4 c) {
    return ushort4(e4m3bf(c.x), e4m3bf(c.y), e4m3bf(c.z), e4m3bf(c.w));
}
inline float4 bf4(ushort4 h) { return as_type<float4>(uint4(h) << 16); }

template <typename T>
inline device T* ten(device const ulong* tab, const thread Inst& in, uint k) {
    ushort h = in.t[k];
    return h == TNONE ? (device T*)0 : reinterpret_cast<device T*>(tab[h]);
}
inline void range(uint n, uint slice, uint nblk, thread uint& lo, thread uint& hi) {
    uint per = (n + nblk - 1) / nblk;
    uint a = slice * per, b = a + per;
    lo = a > n ? n : a;
    hi = b > n ? n : b;
}
// Threadgroup-wide sum of one float per thread (all threads must call).
inline float tg_sum(float v, threadgroup float* red, uint lid, uint sg, uint lane) {
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint k = 0; k < NSG; k++) s += red[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// ---- activations (golden.h) --------------------------------------------------------------------
// tanhs(x) = 1 - 2 / (1 + exp(2x)): exp overflow lands on +-1 exactly (the library tanh returns
// NaN from inf/inf for |x| > ~44 in safe-math mode; Gemma's GELU arguments reach 1e3).
inline float tanhs(float x) { return 1.0f - 2.0f / (1.0f + exp(2.0f * x)); }
inline float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float siluf(float x) { return x * sigmoidf(x); }
inline float gelu_tanhs(float x) {
    float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhs(c));
}
inline float round_bf16_f32(float x) {
    uint bits = as_type<uint>(x);
    return as_type<float>((bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u);
}
inline float gelu_erf_bf16(float x) {
    x = round_bf16_f32(x);
    float z = x * M_SQRT1_2_F, t = 1.0f / (1.0f + 0.3275911f * abs(z));
    float r = (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t -
                  0.284496736f) * t + 0.254829592f) * t) * exp(-z * z);
    return round_bf16_f32(0.5f * x * (z < 0.0f ? r : 2.0f - r));
}
inline float act_gate_only(float g, uint act) { return act >= 2u ? NAN : (act == 1u ? siluf(g) : gelu_tanhs(g)); }
inline float glu_pair(float g, float u, uint act, float f0, float f1) {
    if (act == 2u) {
        float gate = f0 * tanhs(g / f0) * sigmoidf(g);
        float up = f1 > 0.0f ? f1 * tanhs(u / f1) : u;
        return gate * up;
    }
    if (act == 3u) {
        if (!(f1 > 0.0f)) return NAN;
        float gg = min(g, f1), uu = clamp(u, -f1, f1);
        return gg * sigmoidf(f0 * gg) * (uu + 1.0f);
    }
    return act_gate_only(g, act) * u;
}
inline ulong amax_pack(ushort b, uint i) {
    uint key = (b & 0x8000u) ? uint(ushort(~b)) : uint(b | 0x8000u);
    return (ulong(key) << 32) | ulong(~i);
}

// ---- dots: one simdgroup per output, lanes over K ------------------------------------------------
// K may be any multiple of 4 (every model here has K % 8 == 0); the tail is scalar per lane.
// The 8-wide sweeps cover [0, K & ~7) for every lane, so the tail starts there — not at the
// last 256-chunk, which the sweeps already consumed when K % 256 >= 8 (GPT-OSS: K = 2880).
// Bytes in flight: each lane issues 4 independent 16 B loads (bf16) / 4 x 8 B (fp8) per step,
// 1024 K per step per simdgroup, so a threadgroup keeps ~16 KB of weight reads outstanding.
inline float dot_bf16(device const ushort* w, device const ushort* x, uint K, uint lane) {
    float acc = 0.0f;
    uint k = lane * 8u;
    for (; k + 776u <= K; k += 1024u) {
        ushort4 a0 = *(device const ushort4*)(w + k), a1 = *(device const ushort4*)(w + k + 4);
        ushort4 b0 = *(device const ushort4*)(w + k + 256), b1 = *(device const ushort4*)(w + k + 260);
        ushort4 c0 = *(device const ushort4*)(w + k + 512), c1 = *(device const ushort4*)(w + k + 516);
        ushort4 d0 = *(device const ushort4*)(w + k + 768), d1 = *(device const ushort4*)(w + k + 772);
        acc += dot(bf4(a0), bf4(*(device const ushort4*)(x + k))) + dot(bf4(a1), bf4(*(device const ushort4*)(x + k + 4)));
        acc += dot(bf4(b0), bf4(*(device const ushort4*)(x + k + 256))) + dot(bf4(b1), bf4(*(device const ushort4*)(x + k + 260)));
        acc += dot(bf4(c0), bf4(*(device const ushort4*)(x + k + 512))) + dot(bf4(c1), bf4(*(device const ushort4*)(x + k + 516)));
        acc += dot(bf4(d0), bf4(*(device const ushort4*)(x + k + 768))) + dot(bf4(d1), bf4(*(device const ushort4*)(x + k + 772)));
    }
    for (; k + 8u <= K; k += 256u) {
        float4 w0 = bf4(*(device const ushort4*)(w + k)), w1 = bf4(*(device const ushort4*)(w + k + 4));
        float4 x0 = bf4(*(device const ushort4*)(x + k)), x1 = bf4(*(device const ushort4*)(x + k + 4));
        acc += dot(w0, x0) + dot(w1, x1);
    }
    for (k = (K & ~7u) + lane; k < K; k += 32u) acc += bf2f(w[k]) * bf2f(x[k]);
    return simd_sum(acc);
}

#ifdef PLOW_GLU_PAIR
inline float2 dot_pair_bf16(device const ushort* g, device const ushort* u,
                          device const ushort* x, uint K, uint lane) {
    float2 acc = 0.0f;
    uint k = lane * 8u;
    for (; k + 776u <= K; k += 1024u) {
        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            uint p = k + j * 256u;
            float4 x0 = bf4(*(device const ushort4*)(x + p));
            float4 x1 = bf4(*(device const ushort4*)(x + p + 4));
            acc.x += dot(bf4(*(device const ushort4*)(g + p)), x0) + dot(bf4(*(device const ushort4*)(g + p + 4)), x1);
            acc.y += dot(bf4(*(device const ushort4*)(u + p)), x0) + dot(bf4(*(device const ushort4*)(u + p + 4)), x1);
        }
    }
    for (; k + 8u <= K; k += 256u) {
        float4 x0 = bf4(*(device const ushort4*)(x + k));
        float4 x1 = bf4(*(device const ushort4*)(x + k + 4));
        acc.x += dot(bf4(*(device const ushort4*)(g + k)), x0) + dot(bf4(*(device const ushort4*)(g + k + 4)), x1);
        acc.y += dot(bf4(*(device const ushort4*)(u + k)), x0) + dot(bf4(*(device const ushort4*)(u + k + 4)), x1);
    }
    for (k = (K & ~7u) + lane; k < K; k += 32u) {
        float xv = bf2f(x[k]);
        acc.x += bf2f(g[k]) * xv;
        acc.y += bf2f(u[k]) * xv;
    }
    return float2(simd_sum(acc.x), simd_sum(acc.y));
}
#endif

// Four weight rows per simdgroup: 4 x 2 x 16 B loads in flight per lane per step, one x load
// shared. Returns the four dots (every lane holds all four after the reductions).
inline float4 dot4_bf16(device const ushort* W, uint ldw, device const ushort* x, uint K, uint lane) {
    float4 acc = 0.0f;
    uint k = lane * 8u;
    for (; k + 8u <= K; k += 256u) {
        float4 x0 = bf4(*(device const ushort4*)(x + k)), x1 = bf4(*(device const ushort4*)(x + k + 4));
        device const ushort* w0 = W + k;
        device const ushort* w1 = w0 + ldw;
        device const ushort* w2 = w1 + ldw;
        device const ushort* w3 = w2 + ldw;
        ushort4 a0 = *(device const ushort4*)w0, a1 = *(device const ushort4*)(w0 + 4);
        ushort4 b0 = *(device const ushort4*)w1, b1 = *(device const ushort4*)(w1 + 4);
        ushort4 c0 = *(device const ushort4*)w2, c1 = *(device const ushort4*)(w2 + 4);
        ushort4 d0 = *(device const ushort4*)w3, d1 = *(device const ushort4*)(w3 + 4);
        acc.x += dot(bf4(a0), x0) + dot(bf4(a1), x1);
        acc.y += dot(bf4(b0), x0) + dot(bf4(b1), x1);
        acc.z += dot(bf4(c0), x0) + dot(bf4(c1), x1);
        acc.w += dot(bf4(d0), x0) + dot(bf4(d1), x1);
    }
    for (k = (K & ~7u) + lane; k < K; k += 32u) {
        float xv = bf2f(x[k]);
        acc.x += bf2f(W[k]) * xv;
        acc.y += bf2f(W[ldw + k]) * xv;
        acc.z += bf2f(W[2 * ldw + k]) * xv;
        acc.w += bf2f(W[3 * ldw + k]) * xv;
    }
    return float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
}
inline float4 dot4_fp8(device const uchar* W, uint ldw, device const ushort* x, uint K, uint lane) {
    float4 acc = 0.0f;
    uint k = lane * 8u;
    for (; k + 8u <= K; k += 256u) {
        float4 x0 = bf4(*(device const ushort4*)(x + k)), x1 = bf4(*(device const ushort4*)(x + k + 4));
        device const uchar* w0 = W + k;
        device const uchar* w1 = w0 + ldw;
        device const uchar* w2 = w1 + ldw;
        device const uchar* w3 = w2 + ldw;
        uchar4 a0 = *(device const uchar4*)w0, a1 = *(device const uchar4*)(w0 + 4);
        uchar4 b0 = *(device const uchar4*)w1, b1 = *(device const uchar4*)(w1 + 4);
        uchar4 c0 = *(device const uchar4*)w2, c1 = *(device const uchar4*)(w2 + 4);
        uchar4 d0 = *(device const uchar4*)w3, d1 = *(device const uchar4*)(w3 + 4);
        acc.x += dot(e4m3x4(a0), x0) + dot(e4m3x4(a1), x1);
        acc.y += dot(e4m3x4(b0), x0) + dot(e4m3x4(b1), x1);
        acc.z += dot(e4m3x4(c0), x0) + dot(e4m3x4(c1), x1);
        acc.w += dot(e4m3x4(d0), x0) + dot(e4m3x4(d1), x1);
    }
    for (k = (K & ~7u) + lane; k < K; k += 32u) {
        float xv = bf2f(x[k]);
        acc.x += e4m3(W[k]) * xv;
        acc.y += e4m3(W[ldw + k]) * xv;
        acc.z += e4m3(W[2 * ldw + k]) * xv;
        acc.w += e4m3(W[3 * ldw + k]) * xv;
    }
    return float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
}
inline float dot_fp8(device const uchar* w, device const ushort* x, uint K, uint lane) {
    float acc = 0.0f;
    uint k = lane * 8u;
    for (; k + 776u <= K; k += 1024u) {
        uchar4 a0 = *(device const uchar4*)(w + k), a1 = *(device const uchar4*)(w + k + 4);
        uchar4 b0 = *(device const uchar4*)(w + k + 256), b1 = *(device const uchar4*)(w + k + 260);
        uchar4 c0 = *(device const uchar4*)(w + k + 512), c1 = *(device const uchar4*)(w + k + 516);
        uchar4 d0 = *(device const uchar4*)(w + k + 768), d1 = *(device const uchar4*)(w + k + 772);
        acc += dot(e4m3x4(a0), bf4(*(device const ushort4*)(x + k))) + dot(e4m3x4(a1), bf4(*(device const ushort4*)(x + k + 4)));
        acc += dot(e4m3x4(b0), bf4(*(device const ushort4*)(x + k + 256))) + dot(e4m3x4(b1), bf4(*(device const ushort4*)(x + k + 260)));
        acc += dot(e4m3x4(c0), bf4(*(device const ushort4*)(x + k + 512))) + dot(e4m3x4(c1), bf4(*(device const ushort4*)(x + k + 516)));
        acc += dot(e4m3x4(d0), bf4(*(device const ushort4*)(x + k + 768))) + dot(e4m3x4(d1), bf4(*(device const ushort4*)(x + k + 772)));
    }
    for (; k + 8u <= K; k += 256u) {
        uchar4 a = *(device const uchar4*)(w + k), b = *(device const uchar4*)(w + k + 4);
        float4 x0 = bf4(*(device const ushort4*)(x + k)), x1 = bf4(*(device const ushort4*)(x + k + 4));
        acc += dot(e4m3x4(a), x0) + dot(e4m3x4(b), x1);
    }
    for (k = (K & ~7u) + lane; k < K; k += 32u) acc += e4m3(w[k]) * bf2f(x[k]);
    return simd_sum(acc);
}
// norm==2 / q-norm fold: dot(w, bf16(x * inv * gamma))
inline float dot_normed(device const ushort* w, device const ushort* x, device const ushort* gamma,
                        float inv, uint K, uint lane) {
    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32u) {
        float g = gamma ? bf2f(gamma[k]) : 1.0f;
        acc += bf2f(w[k]) * rbf(bf2f(x[k]) * inv * g);
    }
    return simd_sum(acc);
}
// norm==1: sum w*x*gamma in f32
inline float dot_gamma(device const ushort* w, device const ushort* x, device const ushort* gamma,
                       uint K, uint lane) {
    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32u) {
        float g = gamma ? bf2f(gamma[k]) : 1.0f;
        acc += bf2f(w[k]) * bf2f(x[k]) * g;
    }
    return simd_sum(acc);
}
inline float row_ss(device const ushort* x, uint K, uint lane) {
    float ss = 0.0f;
    for (uint k = lane; k < K; k += 32u) { float v = bf2f(x[k]); ss += v * v; }
    return simd_sum(ss);
}

// ---- GEMV family ---------------------------------------------------------------------------------
// t0=C t1=x t2=W t3=rms? t4=gamma? t7=bias?  i0=M i1=N i2=K i3=norm i4=a_row0  f0=eps
void op_gemv(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint sg, uint lane) {
    uint M = in.i[0], N = in.i[1], K = in.i[2], norm = in.i[3];
    device ushort* C = ten<ushort>(tab, in, 0);
    device const ushort* x = ten<ushort>(tab, in, 1) + in.i[4] * K;
    device const ushort* W = ten<ushort>(tab, in, 2);
    device const float* rms = ten<float>(tab, in, 3);
    device const ushort* gamma = ten<ushort>(tab, in, 4);
    device const ushort* bias = ten<ushort>(tab, in, 7);
    float eps = as_type<float>(in.fj[0]);
    uint n0, n1;
    range(N, slice, nblk, n0, n1);
    for (uint m = 0; m < M; m++) {
        device const ushort* xm = x + m * K;
        float inv = norm == 2u ? rsqrt(row_ss(xm, K, lane) / float(K) + eps) : 1.0f;
        uint n = n0 + sg * 4u;
        if (norm == 0u) {
            for (; n + 4u <= n1; n += NSG * 4u) {
                float4 a = dot4_bf16(W + n * K, K, xm, K, lane);
                if (bias) a += float4(bf2f(bias[n]), bf2f(bias[n + 1]), bf2f(bias[n + 2]), bf2f(bias[n + 3]));
                if (lane == 0) { C[m * N + n] = f2bf(a.x); C[m * N + n + 1] = f2bf(a.y); C[m * N + n + 2] = f2bf(a.z); C[m * N + n + 3] = f2bf(a.w); }
            }
            // rows the 4-wide sweep left: at most 3 per simdgroup cursor, taken singly
            for (; n < n1; n++) {
                float acc = dot_bf16(W + n * K, xm, K, lane);
                if (bias) acc += bf2f(bias[n]);
                if (lane == 0) C[m * N + n] = f2bf(acc);
            }
            continue;
        }
        for (n = n0 + sg; n < n1; n += NSG) {
            device const ushort* w = W + n * K;
            float acc;
            if (norm == 1u) acc = dot_gamma(w, xm, gamma, K, lane) * rms[m];
            else acc = dot_normed(w, xm, gamma, inv, K, lane);
            if (bias) acc += bf2f(bias[n]);
            if (lane == 0) C[m * N + n] = f2bf(acc);
        }
    }
}
// t0=fu t1=x t2=W_gate t5=W_up t6=bias_gate? t7=bias_up?  i0=M i1=N i2=K i5=act  f0/f1 act imm
void op_gemv_glu(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint sg, uint lane) {
    uint M = in.i[0], N = in.i[1], K = in.i[2], act = in.i[5];
    device ushort* C = ten<ushort>(tab, in, 0);
    device const ushort* x = ten<ushort>(tab, in, 1);
    device const ushort* Wg = ten<ushort>(tab, in, 2);
    device const ushort* Wu = ten<ushort>(tab, in, 5);
    device const ushort* bg = ten<ushort>(tab, in, 6);
    device const ushort* bu = ten<ushort>(tab, in, 7);
    float f0 = as_type<float>(in.fj[0]), f1 = as_type<float>(in.fj[1]);
    uint n0, n1;
    range(N, slice, nblk, n0, n1);
    for (uint n = n0 + sg; n < n1; n += NSG)
        for (uint m = 0; m < M; m++) {
#ifdef PLOW_GLU_PAIR
            float2 pair = dot_pair_bf16(Wg + n * K, Wu + n * K, x + m * K, K, lane);
            float g = pair.x, u = pair.y;
#else
            float g = dot_bf16(Wg + n * K, x + m * K, K, lane);
            float u = dot_bf16(Wu + n * K, x + m * K, K, lane);
#endif
            if (bg) g += bf2f(bg[n]);
            if (bu) u += bf2f(bu[n]);
            if (lane == 0) C[m * N + n] = f2bf(glu_pair(g, u, act, f0, f1));
        }
}
// t0=q t1=x t2=W_q t3=k t4=W_k t5=v t6=W_v t7=q-norm gamma?  i0=M i1=Nq i2=K i3=Nk i4=Nv
// i5/i6/i7 = bias handles (0 = absent)  f0=eps
void op_gemv_qkv(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint sg, uint lane) {
    uint M = in.i[0], Nq = in.i[1], K = in.i[2], Nk = in.i[3], Nv = in.i[4];
    device const ushort* x = ten<ushort>(tab, in, 1);
    device const ushort* gnorm = ten<ushort>(tab, in, 7);
    float eps = as_type<float>(in.fj[0]);
    device ushort* Cs[3] = {ten<ushort>(tab, in, 0), ten<ushort>(tab, in, 3), ten<ushort>(tab, in, 5)};
    device const ushort* Ws[3] = {ten<ushort>(tab, in, 2), ten<ushort>(tab, in, 4), ten<ushort>(tab, in, 6)};
    device const ushort* Bs[3];
    for (uint s = 0; s < 3; s++) {
        uint h = in.i[5 + s];
        Bs[s] = h ? reinterpret_cast<device const ushort*>(tab[h]) : (device const ushort*)0;
    }
    uint Ns[3] = {Nq, Nk, Nv};
    uint n0, n1;
    range(Nq + Nk + Nv, slice, nblk, n0, n1);
    for (uint m = 0; m < M; m++) {
        device const ushort* xm = x + m * K;
        float inv = gnorm ? rsqrt(row_ss(xm, K, lane) / float(K) + eps) : 1.0f;
#ifdef PLOW_QKV_DOT4
        if (!gnorm) {
            uint offset = 0;
            for (uint s = 0; s < 3; s++) {
                uint lo = max(n0, offset), hi = min(n1, offset + Ns[s]);
                uint n = lo + sg * 4u;
                for (; n + 4u <= hi; n += NSG * 4u) {
                    uint col = n - offset;
                    float4 a = dot4_bf16(Ws[s] + col * K, K, xm, K, lane);
                    if (Bs[s]) a += float4(bf2f(Bs[s][col]), bf2f(Bs[s][col + 1]), bf2f(Bs[s][col + 2]), bf2f(Bs[s][col + 3]));
                    if (lane == 0) {
                        device ushort* out = Cs[s] + m * Ns[s] + col;
                        out[0] = f2bf(a.x); out[1] = f2bf(a.y); out[2] = f2bf(a.z); out[3] = f2bf(a.w);
                    }
                }
                for (; n < hi; n++) {
                    uint col = n - offset;
                    float a = dot_bf16(Ws[s] + col * K, xm, K, lane);
                    if (Bs[s]) a += bf2f(Bs[s][col]);
                    if (lane == 0) Cs[s][m * Ns[s] + col] = f2bf(a);
                }
                offset += Ns[s];
            }
            continue;
        }
#endif
        for (uint n = n0 + sg; n < n1; n += NSG) {
            uint s = 0, col = n;
            while (col >= Ns[s]) { col -= Ns[s]; s++; }
            device const ushort* w = Ws[s] + col * K;
            float acc = gnorm ? dot_normed(w, xm, gnorm, inv, K, lane) : dot_bf16(w, xm, K, lane);
            if (Bs[s]) acc += bf2f(Bs[s][col]);
            if (lane == 0) Cs[s][m * Ns[s] + col] = f2bf(acc);
        }
    }
}
// t0=C t1=x t2=W(e4m3) t5=w_scale i0=M i1=N i2=K i4=a_row0 (i3 != 0 -> poison)
void op_gemv_fp8(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint sg, uint lane) {
    uint M = in.i[0], N = in.i[1], K = in.i[2];
    device ushort* C = ten<ushort>(tab, in, 0);
    device const ushort* x = ten<ushort>(tab, in, 1) + in.i[4] * K;
    device const uchar* W = ten<uchar>(tab, in, 2);
    device const float* ws = ten<float>(tab, in, 5);
    uint n0, n1;
    range(N, slice, nblk, n0, n1);
    bool poison = in.i[3] != 0u || !ws;
    for (uint m = 0; m < M; m++) {
        if (poison) {
            for (uint n = n0 + sg; n < n1; n += NSG) if (lane == 0) C[m * N + n] = ushort(0x7fc1);
            continue;
        }
        uint n = n0 + sg * 4u;
        for (; n + 4u <= n1; n += NSG * 4u) {
            float4 a = dot4_fp8(W + n * K, K, x + m * K, K, lane) * float4(ws[n], ws[n + 1], ws[n + 2], ws[n + 3]);
            if (lane == 0) { C[m * N + n] = f2bf(a.x); C[m * N + n + 1] = f2bf(a.y); C[m * N + n + 2] = f2bf(a.z); C[m * N + n + 3] = f2bf(a.w); }
        }
        for (; n < n1; n++) {
            float acc = dot_fp8(W + n * K, x + m * K, K, lane) * ws[n];
            if (lane == 0) C[m * N + n] = f2bf(acc);
        }
    }
}
// Per-channel scales are tensor handles in i5/i6/i7.
void op_gemv_qkv_fp8(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint sg, uint lane) {
    uint M = in.i[0], K = in.i[2];
    uint Ns[3] = {in.i[1], in.i[3], in.i[4]};
    device const ushort* x = ten<ushort>(tab, in, 1);
    device ushort* Cs[3] = {ten<ushort>(tab, in, 0), ten<ushort>(tab, in, 3), ten<ushort>(tab, in, 5)};
    device const uchar* Ws[3] = {ten<uchar>(tab, in, 2), ten<uchar>(tab, in, 4), ten<uchar>(tab, in, 6)};
    uint n0, n1;
    range(Ns[0] + Ns[1] + Ns[2], slice, nblk, n0, n1);
    uint offset = 0;
    for (uint s = 0; s < 3; s++) {
        uint lo = max(n0, offset), hi = min(n1, offset + Ns[s]);
        if (lo < hi) {
            uint h = in.i[5 + s];
            device const float* ws = h == TNONE ? (device const float*)0 : reinterpret_cast<device const float*>(tab[h]);
            for (uint m = 0; m < M; m++) {
                if (!ws) {
                    for (uint n = lo + sg; n < hi; n += NSG)
                        if (lane == 0) Cs[s][m * Ns[s] + n - offset] = ushort(0x7fc1);
                    continue;
                }
                uint n = lo + sg * 4u;
                for (; n + 4u <= hi; n += NSG * 4u) {
                    uint col = n - offset;
                    float4 a = dot4_fp8(Ws[s] + col * K, K, x + m * K, K, lane)
                        * float4(ws[col], ws[col + 1], ws[col + 2], ws[col + 3]);
                    if (lane == 0) {
                        device ushort* out = Cs[s] + m * Ns[s] + col;
                        out[0] = f2bf(a.x); out[1] = f2bf(a.y); out[2] = f2bf(a.z); out[3] = f2bf(a.w);
                    }
                }
                for (; n < hi; n++) {
                    uint col = n - offset;
                    float a = dot_fp8(Ws[s] + col * K, x + m * K, K, lane) * ws[col];
                    if (lane == 0) Cs[s][m * Ns[s] + col] = f2bf(a);
                }
            }
        }
        offset += Ns[s];
    }
}
// t0=fu t1=x t2=Wg t3=g_scale t4=u_scale t5=Wu i0=M i1=N i2=K i5=act
void op_gemv_glu_fp8(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint sg, uint lane) {
    uint M = in.i[0], N = in.i[1], K = in.i[2], act = in.i[5];
    device ushort* C = ten<ushort>(tab, in, 0);
    device const ushort* x = ten<ushort>(tab, in, 1);
    device const uchar* Wg = ten<uchar>(tab, in, 2);
    device const uchar* Wu = ten<uchar>(tab, in, 5);
    device const float* gs = ten<float>(tab, in, 3);
    device const float* us = ten<float>(tab, in, 4);
    uint n0, n1;
    range(N, slice, nblk, n0, n1);
    for (uint m = 0; m < M; m++) {
        uint n = n0 + sg * 4u;
        for (; n + 4u <= n1; n += NSG * 4u) {
            float4 g = dot4_fp8(Wg + n * K, K, x + m * K, K, lane) * float4(gs[n], gs[n + 1], gs[n + 2], gs[n + 3]);
            float4 u = dot4_fp8(Wu + n * K, K, x + m * K, K, lane) * float4(us[n], us[n + 1], us[n + 2], us[n + 3]);
            if (lane == 0) {
                C[m * N + n] = f2bf(act_gate_only(g.x, act) * u.x);
                C[m * N + n + 1] = f2bf(act_gate_only(g.y, act) * u.y);
                C[m * N + n + 2] = f2bf(act_gate_only(g.z, act) * u.z);
                C[m * N + n + 3] = f2bf(act_gate_only(g.w, act) * u.w);
            }
        }
        for (; n < n1; n++) {
            float g = dot_fp8(Wg + n * K, x + m * K, K, lane) * gs[n];
            float u = dot_fp8(Wu + n * K, x + m * K, K, lane) * us[n];
            if (lane == 0) C[m * N + n] = f2bf(act_gate_only(g, act) * u);
        }
    }
}

// ---- GEMM family (prefill) ------------------------------------------------------------------------
// Output tiles of (BM, BN), linear tile id = slice + k*nblk (golden gemm_tiles order). Inside a tile:
// 64x64 sub-tiles on `simdgroup_matrix<float, 8, 8>` — 8 simdgroups, each a 16x32 block (2x4
// accumulators). K is staged 16 at a time as f32 in threadgroup memory, DOUBLE-BUFFERED: every
// thread owns one float4 of A and one of B per chunk, issues the next chunk's device loads before
// the current chunk's matrix ops and stores them to the other buffer afterwards. One threadgroup
// per core (the persistent grid) has nothing else to hide load latency behind, so this overlap is
// where the prefill rate comes from. The epilogue dumps one 8x8 accumulator at a time through a
// per-simdgroup scratch, so no full output tile is needed in threadgroup memory. Weights are bf16
// or e4m3 (per-column scale in the epilogue); GLU shares the machinery on 64x32 sub-tiles.
struct GemmArgs {
    device const ushort* A;      // [M][K] bf16 (row m offset applied by caller)
    device const uchar* A8;
    device const float* ascale;
    device const ushort* B16;    // [N][K] bf16, or
    device const uchar* B8;      //          e4m3, or
    device const uchar* B4;      //          e2m1 packed (row stride K/2) with
    device const uchar* S8;      //          E8M0 scales (row stride ceil(K/32)), folded at staging
    device const uint* AQ4;
    device const ushort* AQScale;
    device const ushort* AQBias;
    device const float* wscale;  // [N] for B8
    device const ushort* bias;   // [N] bf16 or null (indexed n*rs+ro like the B rows)
    uint K;
    uint M, N;                   // bounds for zero fill
    // MoE extensions (plain GEMMs: arow/Cf/crow/rowscale null, rs 1, ro 0):
    device const uint* arow;     // A row m is A + arow[m]*K (EXPERT_UNUSED -> zeros)
    device float* Cf;            // f32 output instead of bf16 C
    device const uint* crow;     // C row m lands at crow[m] (EXPERT_UNUSED -> dropped)
    device const float* rowscale;// per-row output scale (applied after the bias)
    uint rs, ro;                 // B row of output n = n*rs + ro
};
inline void gemm_args_reset(thread GemmArgs& g) {
    g.A8 = (device const uchar*)0; g.ascale = (device const float*)0;
    g.B16 = (device const ushort*)0; g.B8 = (device const uchar*)0; g.B4 = (device const uchar*)0;
    g.S8 = (device const uchar*)0; g.wscale = (device const float*)0; g.bias = (device const ushort*)0;
    g.AQ4 = (device const uint*)0; g.AQScale = g.AQBias = (device const ushort*)0;
    g.arow = (device const uint*)0; g.Cf = (device float*)0; g.crow = (device const uint*)0;
    g.rowscale = (device const float*)0; g.rs = 1u; g.ro = 0u;
}
// One 8x8 accumulator through the simdgroup's scratch: lane l handles elements 2l, 2l+1.
inline void epilogue8x8(simdgroup_float8x8 acc, threadgroup float* scratch, uint lane, uint m_base, uint n_base,
                        uint m1, uint n1, thread const GemmArgs& g, device ushort* C, uint ldc) {
    simdgroup_store(acc, scratch, 8u);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane * 2u; e < lane * 2u + 2u; e++) {
        uint m = m_base + e / 8u, n = n_base + e % 8u;
        if (m < m1 && n < n1) {
            float v = scratch[e];
            if (g.ascale) v *= g.ascale[m];
            if (g.B8) v *= g.wscale[n];
            if (g.bias) v += bf2f(g.bias[n * g.rs + g.ro]);
            if (g.rowscale) v *= g.rowscale[m];
            uint cm = g.crow ? g.crow[m] : m;
            if (cm == 0xffffffffu) continue;
            if (g.Cf) g.Cf[cm * ldc + n] = v;
            else C[cm * ldc + n] = f2bf(v);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}
// ---- MXFP4 (OCP e2m1 + one E8M0 scale per 32 K) ---------------------------------------------------
// Row layout (dev_isa.h op 91): W[n] is K/2 bytes, low nibble = even k; S[n] is K/32 E8M0 bytes,
// scale = bitcast(s << 23). A 32-block is accumulated in f32 and scaled once (the scale is a
// power of two, so folding it into the bf16 operand — the GEMM staging path — is exact).
constant float E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                           -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
inline float e8m0f(uchar s) { return as_type<float>(uint(s) << 23); }
// Byte -> (low nibble, high nibble) e2m1 pair: one lookup decodes two weights.
// Four rows' 32-block dots against x[32], one block per lane per step (16 B of weights per row).
#ifdef PLOW_MX4_FOUR_ROWS
inline void dot4_mx4_quad(device const uchar* W, device const uchar* S, uint ldw, uint lds,
    device const ushort* x, uint K, uint lane, thread float4& a0, thread float4& a1, thread float4& a2, thread float4& a3) {
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f), acc3 = float4(0.0f);
    for (uint b = lane; b < K / 32u; b += 32u) {
        float4 blk0 = float4(0.0f), blk1 = float4(0.0f), blk2 = float4(0.0f), blk3 = float4(0.0f);
        for (uint j = 0; j < 4u; j++) {
            device const ushort* xb = x + b * 32u + 8u * j;
            float4 x0 = bf4(*(device const ushort4*)xb), x1 = bf4(*(device const ushort4*)(xb + 4u));
            float4 y0 = bf4(*(device const ushort4*)(xb + K)), y1 = bf4(*(device const ushort4*)(xb + K + 4u));
            float4 z0 = bf4(*(device const ushort4*)(xb + 2u * K)), z1 = bf4(*(device const ushort4*)(xb + 2u * K + 4u));
            float4 t0 = bf4(*(device const ushort4*)(xb + 3u * K)), t1 = bf4(*(device const ushort4*)(xb + 3u * K + 4u));
            for (uint r = 0; r < 4u; r++) {
                uchar4 c = *(device const uchar4*)(W + r * ldw + b * 16u + 4u * j);
                float4 w0 = float4(E2M1[c.x & 15u], E2M1[c.x >> 4], E2M1[c.y & 15u], E2M1[c.y >> 4]);
                float4 w1 = float4(E2M1[c.z & 15u], E2M1[c.z >> 4], E2M1[c.w & 15u], E2M1[c.w >> 4]);
                blk0[r] += w0.x * x0.x + w0.y * x0.y + w0.z * x0.z + w0.w * x0.w
                         + w1.x * x1.x + w1.y * x1.y + w1.z * x1.z + w1.w * x1.w;
                blk1[r] += w0.x * y0.x + w0.y * y0.y + w0.z * y0.z + w0.w * y0.w
                         + w1.x * y1.x + w1.y * y1.y + w1.z * y1.z + w1.w * y1.w;
                blk2[r] += w0.x * z0.x + w0.y * z0.y + w0.z * z0.z + w0.w * z0.w
                         + w1.x * z1.x + w1.y * z1.y + w1.z * z1.z + w1.w * z1.w;
                blk3[r] += w0.x * t0.x + w0.y * t0.y + w0.z * t0.z + w0.w * t0.w
                         + w1.x * t1.x + w1.y * t1.y + w1.z * t1.z + w1.w * t1.w;
            }
        }
        for (uint r = 0; r < 4u; r++) {
            float scale = e8m0f(S[r * lds + b]);
            acc0[r] += blk0[r] * scale;
            acc1[r] += blk1[r] * scale;
            acc2[r] += blk2[r] * scale;
            acc3[r] += blk3[r] * scale;
        }
    }
    a0 = float4(simd_sum(acc0.x), simd_sum(acc0.y), simd_sum(acc0.z), simd_sum(acc0.w));
    a1 = float4(simd_sum(acc1.x), simd_sum(acc1.y), simd_sum(acc1.z), simd_sum(acc1.w));
    a2 = float4(simd_sum(acc2.x), simd_sum(acc2.y), simd_sum(acc2.z), simd_sum(acc2.w));
    a3 = float4(simd_sum(acc3.x), simd_sum(acc3.y), simd_sum(acc3.z), simd_sum(acc3.w));
}
#endif
inline void dot4_mx4_pair(device const uchar* W, device const uchar* S, uint ldw, uint lds,
    device const ushort* x, uint K, uint lane, thread float4& a0, thread float4& a1) {
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
    for (uint b = lane; b < K / 32u; b += 32u) {
        float4 blk0 = float4(0.0f), blk1 = float4(0.0f);
        for (uint j = 0; j < 4u; j++) {
            device const ushort* xb = x + b * 32u + 8u * j;
            float4 x0 = bf4(*(device const ushort4*)xb), x1 = bf4(*(device const ushort4*)(xb + 4u));
            float4 y0 = bf4(*(device const ushort4*)(xb + K)), y1 = bf4(*(device const ushort4*)(xb + K + 4u));
            for (uint r = 0; r < 4u; r++) {
                uchar4 c = *(device const uchar4*)(W + r * ldw + b * 16u + 4u * j);
                float4 w0 = float4(E2M1[c.x & 15u], E2M1[c.x >> 4], E2M1[c.y & 15u], E2M1[c.y >> 4]);
                float4 w1 = float4(E2M1[c.z & 15u], E2M1[c.z >> 4], E2M1[c.w & 15u], E2M1[c.w >> 4]);
                blk0[r] += w0.x * x0.x + w0.y * x0.y + w0.z * x0.z + w0.w * x0.w
                         + w1.x * x1.x + w1.y * x1.y + w1.z * x1.z + w1.w * x1.w;
                blk1[r] += w0.x * y0.x + w0.y * y0.y + w0.z * y0.z + w0.w * y0.w
                         + w1.x * y1.x + w1.y * y1.y + w1.z * y1.z + w1.w * y1.w;
            }
        }
        for (uint r = 0; r < 4u; r++) {
            float scale = e8m0f(S[r * lds + b]);
            acc0[r] += blk0[r] * scale;
            acc1[r] += blk1[r] * scale;
        }
    }
    a0 = float4(simd_sum(acc0.x), simd_sum(acc0.y), simd_sum(acc0.z), simd_sum(acc0.w));
    a1 = float4(simd_sum(acc1.x), simd_sum(acc1.y), simd_sum(acc1.z), simd_sum(acc1.w));
}
inline float4 dot4_mx4(device const uchar* W, device const uchar* S, uint ldw, uint lds,
                       device const ushort* x, uint K, uint lane) {
    float4 acc = float4(0.0f);
    uint nb = K / 32u;
    for (uint b = lane; b < nb; b += 32u) {
        device const ushort* xb = x + b * 32u;
        float xv[32];
        for (uint j = 0; j < 8u; j++) {
            float4 v = bf4(*(device const ushort4*)(xb + 4u * j));
            xv[4u * j] = v.x; xv[4u * j + 1u] = v.y; xv[4u * j + 2u] = v.z; xv[4u * j + 3u] = v.w;
        }
        for (uint r = 0; r < 4u; r++) {
            device const uchar* wb = W + r * ldw + b * 16u;
            float blk = 0.0f;
            for (uint j = 0; j < 4u; j++) {
                uchar4 c = *(device const uchar4*)(wb + 4u * j);
                blk += E2M1[c.x & 15u] * xv[8u * j] + E2M1[c.x >> 4] * xv[8u * j + 1u]
                     + E2M1[c.y & 15u] * xv[8u * j + 2u] + E2M1[c.y >> 4] * xv[8u * j + 3u]
                     + E2M1[c.z & 15u] * xv[8u * j + 4u] + E2M1[c.z >> 4] * xv[8u * j + 5u]
                     + E2M1[c.w & 15u] * xv[8u * j + 6u] + E2M1[c.w >> 4] * xv[8u * j + 7u];
            }
            acc[r] += blk * e8m0f(S[r * lds + b]);
        }
    }
    // Partial last block (K % 32): lane 0, scalar.
    if (lane == 0 && (K & 31u)) {
        uint k0 = nb * 32u;
        for (uint r = 0; r < 4u; r++) {
            float blk = 0.0f;
            for (uint k = k0; k < K; k++) {
                uchar c = W[r * ldw + (k >> 1)];
                blk += E2M1[(k & 1u) ? (c >> 4) : (c & 15u)] * bf2f(x[k]);
            }
            acc[r] += blk * e8m0f(S[r * lds + nb]);
        }
    }
    return float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
}
#ifdef PLOW_MX4_FOUR_ROWS
inline float4 dot_mx4_quad(device const uchar* W, device const uchar* S, device const ushort* x, uint K, uint lane) {
    float acc = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    uint nb = K / 32u;
    for (uint b = lane; b < nb; b += 32u) {
        device const ushort* xb = x + b * 32u;
        device const uchar* wb = W + b * 16u;
        float blk = 0.0f, blk1 = 0.0f, blk2 = 0.0f, blk3 = 0.0f;
        for (uint j = 0; j < 4u; j++) {
            uchar4 c = *(device const uchar4*)(wb + 4u * j);
            float4 x0 = bf4(*(device const ushort4*)(xb + 8u * j)), x1 = bf4(*(device const ushort4*)(xb + 8u * j + 4u));
            float4 y0 = bf4(*(device const ushort4*)(xb + K + 8u * j)), y1 = bf4(*(device const ushort4*)(xb + K + 8u * j + 4u));
            float4 z0 = bf4(*(device const ushort4*)(xb + 2u * K + 8u * j)), z1 = bf4(*(device const ushort4*)(xb + 2u * K + 8u * j + 4u));
            float4 t0 = bf4(*(device const ushort4*)(xb + 3u * K + 8u * j)), t1 = bf4(*(device const ushort4*)(xb + 3u * K + 8u * j + 4u));
            blk += E2M1[c.x & 15u] * x0.x + E2M1[c.x >> 4] * x0.y + E2M1[c.y & 15u] * x0.z + E2M1[c.y >> 4] * x0.w
                 + E2M1[c.z & 15u] * x1.x + E2M1[c.z >> 4] * x1.y + E2M1[c.w & 15u] * x1.z + E2M1[c.w >> 4] * x1.w;
            blk1 += E2M1[c.x & 15u] * y0.x + E2M1[c.x >> 4] * y0.y + E2M1[c.y & 15u] * y0.z + E2M1[c.y >> 4] * y0.w
                 + E2M1[c.z & 15u] * y1.x + E2M1[c.z >> 4] * y1.y + E2M1[c.w & 15u] * y1.z + E2M1[c.w >> 4] * y1.w;
            blk2 += E2M1[c.x & 15u] * z0.x + E2M1[c.x >> 4] * z0.y + E2M1[c.y & 15u] * z0.z + E2M1[c.y >> 4] * z0.w
                 + E2M1[c.z & 15u] * z1.x + E2M1[c.z >> 4] * z1.y + E2M1[c.w & 15u] * z1.z + E2M1[c.w >> 4] * z1.w;
            blk3 += E2M1[c.x & 15u] * t0.x + E2M1[c.x >> 4] * t0.y + E2M1[c.y & 15u] * t0.z + E2M1[c.y >> 4] * t0.w
                 + E2M1[c.z & 15u] * t1.x + E2M1[c.z >> 4] * t1.y + E2M1[c.w & 15u] * t1.z + E2M1[c.w >> 4] * t1.w;
        }
        float scale = e8m0f(S[b]);
        acc += blk * scale;
        acc1 += blk1 * scale;
        acc2 += blk2 * scale;
        acc3 += blk3 * scale;
    }
    return float4(simd_sum(acc), simd_sum(acc1), simd_sum(acc2), simd_sum(acc3));
}
#endif
inline float2 dot_mx4_pair(device const uchar* W, device const uchar* S, device const ushort* x, uint K, uint lane) {
    float acc = 0.0f, acc1 = 0.0f;
    uint nb = K / 32u;
    for (uint b = lane; b < nb; b += 32u) {
        device const ushort* xb = x + b * 32u;
        device const uchar* wb = W + b * 16u;
        float blk = 0.0f, blk1 = 0.0f;
        for (uint j = 0; j < 4u; j++) {
            uchar4 c = *(device const uchar4*)(wb + 4u * j);
            float4 x0 = bf4(*(device const ushort4*)(xb + 8u * j)), x1 = bf4(*(device const ushort4*)(xb + 8u * j + 4u));
            float4 y0 = bf4(*(device const ushort4*)(xb + K + 8u * j)), y1 = bf4(*(device const ushort4*)(xb + K + 8u * j + 4u));
            blk += E2M1[c.x & 15u] * x0.x + E2M1[c.x >> 4] * x0.y + E2M1[c.y & 15u] * x0.z + E2M1[c.y >> 4] * x0.w
                 + E2M1[c.z & 15u] * x1.x + E2M1[c.z >> 4] * x1.y + E2M1[c.w & 15u] * x1.z + E2M1[c.w >> 4] * x1.w;
            blk1 += E2M1[c.x & 15u] * y0.x + E2M1[c.x >> 4] * y0.y + E2M1[c.y & 15u] * y0.z + E2M1[c.y >> 4] * y0.w
                 + E2M1[c.z & 15u] * y1.x + E2M1[c.z >> 4] * y1.y + E2M1[c.w & 15u] * y1.z + E2M1[c.w >> 4] * y1.w;
        }
        float scale = e8m0f(S[b]);
        acc += blk * scale;
        acc1 += blk1 * scale;
    }
    return float2(simd_sum(acc), simd_sum(acc1));
}
inline float dot_mx4(device const uchar* W, device const uchar* S, device const ushort* x, uint K, uint lane) {
    float acc = 0.0f;
    uint nb = K / 32u;
    for (uint b = lane; b < nb; b += 32u) {
        device const ushort* xb = x + b * 32u;
        device const uchar* wb = W + b * 16u;
        float blk = 0.0f;
        for (uint j = 0; j < 4u; j++) {
            uchar4 c = *(device const uchar4*)(wb + 4u * j);
            float4 x0 = bf4(*(device const ushort4*)(xb + 8u * j)), x1 = bf4(*(device const ushort4*)(xb + 8u * j + 4u));
            blk += E2M1[c.x & 15u] * x0.x + E2M1[c.x >> 4] * x0.y + E2M1[c.y & 15u] * x0.z + E2M1[c.y >> 4] * x0.w
                 + E2M1[c.z & 15u] * x1.x + E2M1[c.z >> 4] * x1.y + E2M1[c.w & 15u] * x1.z + E2M1[c.w >> 4] * x1.w;
        }
        acc += blk * e8m0f(S[b]);
    }
    if (lane == 0 && (K & 31u)) {
        float blk = 0.0f;
        for (uint k = nb * 32u; k < K; k++) {
            uchar c = W[k >> 1];
            blk += E2M1[(k & 1u) ? (c >> 4) : (c & 15u)] * bf2f(x[k]);
        }
        acc += blk * e8m0f(S[nb]);
    }
    return simd_sum(acc);
}
// t0=C t1=x t2=W(fp4) t3=S(e8m0)  i0=M i1=N i2=K i4=x_row0  (bias t7 is a CPU-tier operand)
void op_gemv_mxfp4(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint sg, uint lane) {
    uint M = in.i[0], N = in.i[1], K = in.i[2];
    device ushort* C = ten<ushort>(tab, in, 0);
    device const ushort* x = ten<ushort>(tab, in, 1) + in.i[4] * K;
    device const uchar* W = ten<uchar>(tab, in, 2);
    device const uchar* S = ten<uchar>(tab, in, 3);
    uint ldw = K / 2u, lds = (K + 31u) / 32u;
    uint n0, n1;
    range(N, slice, nblk, n0, n1);
#ifdef PLOW_MX4_FOUR_ROWS
    if (M == 4 && (K % 32u) == 0 && (n0 % 4u) == 0 && (n1 % 4u) == 0) {
        for (uint n = n0 + sg * 4u; n + 4u <= n1; n += NSG * 4u) {
            float4 a0, a1, a2, a3;
            dot4_mx4_quad(W + n * ldw, S + n * lds, ldw, lds, x, K, lane, a0, a1, a2, a3);
            if (lane == 0) {
                C[n] = f2bf(a0.x); C[n+1] = f2bf(a0.y); C[n+2] = f2bf(a0.z); C[n+3] = f2bf(a0.w);
                C[N+n] = f2bf(a1.x); C[N+n+1] = f2bf(a1.y); C[N+n+2] = f2bf(a1.z); C[N+n+3] = f2bf(a1.w);
                C[2u*N+n] = f2bf(a2.x); C[2u*N+n+1] = f2bf(a2.y); C[2u*N+n+2] = f2bf(a2.z); C[2u*N+n+3] = f2bf(a2.w);
                C[3u*N+n] = f2bf(a3.x); C[3u*N+n+1] = f2bf(a3.y); C[3u*N+n+2] = f2bf(a3.z); C[3u*N+n+3] = f2bf(a3.w);
            }
        }
        return;
    }
#endif
    if (M == 2 && (K % 32u) == 0 && (n0 % 4u) == 0 && (n1 % 4u) == 0) {
        for (uint n = n0 + sg * 4u; n + 4u <= n1; n += NSG * 4u) {
            float4 a0, a1;
            dot4_mx4_pair(W + n * ldw, S + n * lds, ldw, lds, x, K, lane, a0, a1);
            if (lane == 0) {
                C[n] = f2bf(a0.x); C[n+1] = f2bf(a0.y); C[n+2] = f2bf(a0.z); C[n+3] = f2bf(a0.w);
                C[N+n] = f2bf(a1.x); C[N+n+1] = f2bf(a1.y); C[N+n+2] = f2bf(a1.z); C[N+n+3] = f2bf(a1.w);
            }
        }
        return;
    }
    for (uint m = 0; m < M; m++) {
        uint n = n0 + sg * 4u;
        for (; n + 4u <= n1; n += NSG * 4u) {
            float4 a = dot4_mx4(W + n * ldw, S + n * lds, ldw, lds, x + m * K, K, lane);
            if (lane == 0) { C[m * N + n] = f2bf(a.x); C[m * N + n + 1] = f2bf(a.y); C[m * N + n + 2] = f2bf(a.z); C[m * N + n + 3] = f2bf(a.w); }
        }
        for (; n < n1; n++) {
            float acc = dot_mx4(W + n * ldw, S + n * lds, x + m * K, K, lane);
            if (lane == 0) C[m * N + n] = f2bf(acc);
        }
    }
}
// t0=C t1=x t2=Wg(fp4) t5=Wu(fp4) t3=Sg t4=Su  i0=M i1=N i2=K i5=act
void op_gemv_glu_mxfp4(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint sg, uint lane) {
    uint M = in.i[0], N = in.i[1], K = in.i[2], act = in.i[5];
    float f0 = as_type<float>(in.fj[0]), f1 = as_type<float>(in.fj[1]);
    device ushort* C = ten<ushort>(tab, in, 0);
    device const ushort* x = ten<ushort>(tab, in, 1);
    device const uchar* Wg = ten<uchar>(tab, in, 2);
    device const uchar* Wu = ten<uchar>(tab, in, 5);
    device const uchar* Sg = ten<uchar>(tab, in, 3);
    device const uchar* Su = ten<uchar>(tab, in, 4);
    uint ldw = K / 2u, lds = (K + 31u) / 32u;
    uint n0, n1;
    range(N, slice, nblk, n0, n1);
#ifdef PLOW_MX4_FOUR_ROWS
    if (M == 4 && (K % 32u) == 0) {
        for (uint n = n0 + sg; n < n1; n += NSG) {
            float4 g = dot_mx4_quad(Wg + n * ldw, Sg + n * lds, x, K, lane);
            float4 u = dot_mx4_quad(Wu + n * ldw, Su + n * lds, x, K, lane);
            if (lane == 0) {
                C[n] = f2bf(glu_pair(g.x, u.x, act, f0, f1));
                C[N+n] = f2bf(glu_pair(g.y, u.y, act, f0, f1));
                C[2u*N+n] = f2bf(glu_pair(g.z, u.z, act, f0, f1));
                C[3u*N+n] = f2bf(glu_pair(g.w, u.w, act, f0, f1));
            }
        }
        return;
    }
#endif
    if (M == 2 && (K % 32u) == 0) {
        for (uint n = n0 + sg; n < n1; n += NSG) {
            float2 g = dot_mx4_pair(Wg + n * ldw, Sg + n * lds, x, K, lane);
            float2 u = dot_mx4_pair(Wu + n * ldw, Su + n * lds, x, K, lane);
            if (lane == 0) {
                C[n] = f2bf(glu_pair(g.x, u.x, act, f0, f1));
                C[N+n] = f2bf(glu_pair(g.y, u.y, act, f0, f1));
            }
        }
        return;
    }
    for (uint n = n0 + sg; n < n1; n += NSG)
        for (uint m = 0; m < M; m++) {
            float g = dot_mx4(Wg + n * ldw, Sg + n * lds, x + m * K, K, lane);
            float u = dot_mx4(Wu + n * ldw, Su + n * lds, x + m * K, K, lane);
            if (lane == 0) C[m * N + n] = f2bf(glu_pair(g, u, act, f0, f1));
        }
}

// ---- v3 tiles: bf16 staging in 8x8 blocks, K chunks of 32, 128-wide sub-tiles ---------------------
// One threadgroup per core (the persistent grid) cannot hide latency by occupancy, so the kernel
// pipelines inside the threadgroup: the next chunk's device loads sit in registers while this
// chunk's matrix ops run, and inside a chunk each simdgroup fetches the fragments of step kb+1
// from threadgroup memory before multiplying step kb. Chunks are staged as bf16 (exact for e4m3
// weights) in 8x8 blocks — block ib = (row/8)*(GK/8) + k/8, element (row%8)*8 + k%8 — so
// simdgroup_load reads one block with ld = 8. Two (SM + 128) x 32 stages fill the 32 KB of
// threadgroup memory at SM = 128. Products are bf16 x bf16 into f32 accumulators.
constant uint GK = 32;
constant uint SN2 = 128;
constant uint TILE_FLOATS = 8192;   // the whole 32 KB; `red`/`keys` alias into it (see plow_interp)
inline void stage_put(threadgroup ushort* buf, uint row, uint q, ushort4 v) {
    uint ib = (row / 8u) * (GK / 8u) + (q / 2u);
    *(threadgroup ushort4*)(buf + 64u * ib + (row % 8u) * 8u + (q % 2u) * 4u) = v;
}
inline ushort4 f4bf(float4 v) { return ushort4(f2bf(v.x), f2bf(v.y), f2bf(v.z), f2bf(v.w)); }
inline ushort4 load_a16(device const ushort* A, uint M, uint K, uint m, uint k) {
    if (m >= M || k >= K) return ushort4(0);
    if ((K & 3u) == 0u) return *(device const ushort4*)(A + m * K + k);
    ushort4 v = ushort4(0);
    for (uint j = 0; j < 4u && k + j < K; j++) v[j] = A[m * K + k + j];
    return v;
}
// A row through the optional gather map.
inline ushort4 load_arow(thread const GemmArgs& g, uint m, uint k) {
    if (m >= g.M) return ushort4(0);
    uint r = g.arow ? g.arow[m] : m;
    if (r == 0xffffffffu) return ushort4(0);
    if (g.A8) {
        ushort4 v = ushort4(0);
        for (uint j = 0; j < 4u && k + j < g.K; j++)
            v[j] = f2bf(e4m3_exact(g.A8[r * g.K + k + j]));
        return v;
    }
    return load_a16(g.A, 0xffffffffu, g.K, r, k);
}
inline ushort4 load_b16(thread const GemmArgs& g, uint n, uint k) {
    const uint N = g.N, K = g.K;
    if (n >= N || k >= K) return ushort4(0);
    n = n * g.rs + g.ro;
    if (g.AQ4) {
        uint q = g.AQ4[ulong(n) * (K / 8u) + k / 8u] >> ((k % 8u) * 4u);
        ulong group = ulong(n) * (K / 64u) + k / 64u;
        float s = bf2f(g.AQScale[group]), b = bf2f(g.AQBias[group]);
        return f4bf(float4(q & 15u, (q >> 4) & 15u, (q >> 8) & 15u, (q >> 12) & 15u) * s + b);
    }
    if (g.B4) {
        // Two packed bytes hold the k-quad; the block scale is a power of two, so the product is
        // exact in bf16.
        float s = e8m0f(g.S8[n * ((K + 31u) / 32u) + k / 32u]);
        if ((K & 3u) == 0u) {
            uchar2 pk = *(device const uchar2*)(g.B4 + n * (K / 2u) + k / 2u);
            uint c = uint(pk.x) | (uint(pk.y) << 8);
            return f4bf(float4(E2M1[c & 15u], E2M1[(c >> 4) & 15u], E2M1[(c >> 8) & 15u], E2M1[(c >> 12) & 15u]) * s);
        }
        ushort4 v = ushort4(0);
        for (uint j = 0; j < 4u && k + j < K; j++) {
            uchar c = g.B4[n * (K / 2u) + ((k + j) >> 1)];
            v[j] = f2bf(E2M1[((k + j) & 1u) ? (c >> 4) : (c & 15u)] * s);
        }
        return v;
    }
    if (g.A8 && g.B8) {
        ushort4 v = ushort4(0);
        for (uint j = 0; j < 4u && k + j < K; j++) v[j] = f2bf(e4m3_exact(g.B8[n * K + k + j]));
        return v;
    }
    if ((K & 3u) == 0u) {
#ifdef PLOW_FP8_CAST_BASELINE
        return g.B8 ? f4bf(e4m3x4(*(device const uchar4*)(g.B8 + n * K + k))) : *(device const ushort4*)(g.B16 + n * K + k);
#else
        return g.B8 ? e4m3bf4(*(device const uchar4*)(g.B8 + n * K + k)) : *(device const ushort4*)(g.B16 + n * K + k);
#endif
    }
    ushort4 v = ushort4(0);
    for (uint j = 0; j < 4u && k + j < K; j++) v[j] = g.B8 ? f2bf(e4m3(g.B8[n * K + k + j])) : g.B16[n * K + k + j];
    return v;
}
// Fragments of step kb for a simdgroup at grid (sgr, sgc): RB A blocks, CB B blocks (transposed).
template <uint RB, uint CB>
inline void frag_load(threadgroup const bfloat* As, threadgroup const bfloat* Bs, uint sgr, uint sgc, uint kb,
                      thread simdgroup_bfloat8x8 (&a)[RB], thread simdgroup_bfloat8x8 (&b)[CB]) {
    for (uint i = 0; i < RB; i++) simdgroup_load(a[i], As + 64u * ((sgr * RB + i) * (GK / 8u) + kb), 8u);
    for (uint j = 0; j < CB; j++) simdgroup_load(b[j], Bs + 64u * ((sgc * CB + j) * (GK / 8u) + kb), 8u, ulong2(0, 0), true);
}
// One SM x 128 sub-tile at (mm, nn). Simdgroups form an RS x CS grid (RS*CS == 32), each owning
// RB x CB 8x8 accumulators. Thread lid stages A group (row lid/8, k-quad lid%8) and B group
// (same indexing over the 128 B rows) of every chunk: two ushort4 device loads per chunk.
template <uint SM, uint SN, uint RB, uint CB>
inline void gemm_sub(thread const GemmArgs& g, device ushort* C, uint ldc, uint mm, uint nn, uint m1, uint n1,
                     threadgroup ushort* stage, uint lid, uint sg, uint lane) {
    constexpr uint RS = SM / (8u * RB), CS = NSG / RS;
    constexpr uint A_ELTS = SM * GK, STAGE = (SM + SN) * GK;
    constexpr uint QK = GK / 4u;                   // k-quads per row
    const uint nchunk = (g.K + GK - 1u) / GK;
    const bool a_side = lid < SM * QK, b_side = lid < SN * QK;
    const uint ar = lid / QK, aq = lid % QK, br = lid / QK, bq = lid % QK;
    const uint sgr = sg / CS, sgc = sg % CS;
    simdgroup_float8x8 acc[RB][CB];
    for (uint i = 0; i < RB; i++) for (uint j = 0; j < CB; j++) acc[i][j] = simdgroup_float8x8(0.0f);
    ushort4 ra = a_side ? load_arow(g, mm + ar, aq * 4u) : ushort4(0);
    ushort4 rb = b_side ? load_b16(g, nn + br, bq * 4u) : ushort4(0);
    if (a_side) stage_put(stage, ar, aq, ra);
    if (b_side) stage_put(stage + A_ELTS, br, bq, rb);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = 0; c < nchunk; c++) {
        threadgroup ushort* cur = stage + (c & 1u) * STAGE;
        threadgroup ushort* nxt = stage + ((c + 1u) & 1u) * STAGE;
        bool more = c + 1u < nchunk;
        if (more) {
            uint k0 = (c + 1u) * GK;
            if (a_side) ra = load_arow(g, mm + ar, k0 + aq * 4u);
            if (b_side) rb = load_b16(g, nn + br, k0 + bq * 4u);
        }
        threadgroup const bfloat* As = (threadgroup const bfloat*)cur;
        threadgroup const bfloat* Bs = As + A_ELTS;
        simdgroup_bfloat8x8 a0[RB], b0[CB], a1[RB], b1[CB];
        frag_load<RB, CB>(As, Bs, sgr, sgc, 0u, a0, b0);
        for (uint kb = 0; kb < GK / 8u; kb += 2u) {
            frag_load<RB, CB>(As, Bs, sgr, sgc, kb + 1u, a1, b1);
            for (uint i = 0; i < RB; i++)
                for (uint j = 0; j < CB; j++) simdgroup_multiply_accumulate(acc[i][j], a0[i], b0[j], acc[i][j]);
            if (kb + 2u < GK / 8u) frag_load<RB, CB>(As, Bs, sgr, sgc, kb + 2u, a0, b0);
            for (uint i = 0; i < RB; i++)
                for (uint j = 0; j < CB; j++) simdgroup_multiply_accumulate(acc[i][j], a1[i], b1[j], acc[i][j]);
        }
        if (more) {
            if (a_side) stage_put(nxt, ar, aq, ra);
            if (b_side) stage_put(nxt + A_ELTS, br, bq, rb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* scratch = (threadgroup float*)stage + sg * 64u;
    for (uint i = 0; i < RB; i++)
        for (uint j = 0; j < CB; j++)
            epilogue8x8(acc[i][j], scratch, lane, mm + (sgr * RB + i) * 8u, nn + (sgc * CB + j) * 8u, m1, n1, g, C, ldc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Sub-tile rows follow the remaining M: 128 (16x32 blocks), 64 (8x32) or 32 (8x16); 64-wide
// tiles (BN = 64: the narrow projections, where 128-wide tiles would leave cores idle) halve
// the block columns.
inline void gemm_tile2(thread const GemmArgs& g, device ushort* C, uint ldc, uint m0, uint m1, uint n0, uint n1,
                       uint bn, threadgroup float* tile, uint lid, uint sg, uint lane) {
    threadgroup ushort* stage = (threadgroup ushort*)tile;
    for (uint mm = m0; mm < m1;) {
        uint rem = m1 - mm;
        uint sm = rem > 64u ? 128u : (rem > 32u ? 64u : 32u);
        if (bn >= SN2) {
            for (uint nn = n0; nn < n1; nn += SN2) {
                if (sm == 128u) gemm_sub<128, SN2, 2, 4>(g, C, ldc, mm, nn, m1, n1, stage, lid, sg, lane);
                else if (sm == 64u) gemm_sub<64, SN2, 1, 4>(g, C, ldc, mm, nn, m1, n1, stage, lid, sg, lane);
                else gemm_sub<32, SN2, 1, 2>(g, C, ldc, mm, nn, m1, n1, stage, lid, sg, lane);
            }
        } else {
            for (uint nn = n0; nn < n1; nn += 64u) {
                if (sm == 128u) gemm_sub<128, 64, 2, 2>(g, C, ldc, mm, nn, m1, n1, stage, lid, sg, lane);
                else if (sm == 64u) gemm_sub<64, 64, 1, 2>(g, C, ldc, mm, nn, m1, n1, stage, lid, sg, lane);
                else gemm_sub<32, 64, 1, 1>(g, C, ldc, mm, nn, m1, n1, stage, lid, sg, lane);
            }
        }
        mm += sm;
    }
}
// GLU sub-tile: SM x (64 gate | 64 up) with one A chunk; each simdgroup owns RB x CB accumulators
// for gate and for up. Thread lid stages A (row lid/8) and, for lid < 512, one gate and one up
// group (row lid/8 of the 64-row halves).
template <uint SM, uint RB, uint CB>
inline void gemm_sub_glu(thread const GemmArgs& gg, thread const GemmArgs& gu, device ushort* C, uint act,
                         float f0, float f1, bool fp8, uint mm, uint nn, uint m1, uint n1,
                         threadgroup ushort* stage, uint lid, uint sg, uint lane) {
    constexpr uint RS = SM / (8u * RB), CS = NSG / RS, HN = 64u;
    constexpr uint A_ELTS = SM * GK, H_ELTS = HN * GK, STAGE = (SM + 2u * HN) * GK;
    constexpr uint QK = GK / 4u;
    const uint K = gg.K, M = gg.M, N = gg.N;
    const uint nchunk = (K + GK - 1u) / GK;
    const bool a_side = lid < SM * QK, h_side = lid < HN * QK;
    const uint ar = lid / QK, aq = lid % QK;
    const uint sgr = sg / CS, sgc = sg % CS;
    simdgroup_float8x8 ag[RB][CB], au[RB][CB];
    for (uint i = 0; i < RB; i++) for (uint j = 0; j < CB; j++) { ag[i][j] = simdgroup_float8x8(0.0f); au[i][j] = simdgroup_float8x8(0.0f); }
    ushort4 ra = a_side ? load_arow(gg, mm + ar, aq * 4u) : ushort4(0);
    ushort4 rg = h_side ? load_b16(gg, nn + ar, aq * 4u) : ushort4(0);
    ushort4 ru = h_side ? load_b16(gu, nn + ar, aq * 4u) : ushort4(0);
    if (a_side) stage_put(stage, ar, aq, ra);
    if (h_side) { stage_put(stage + A_ELTS, ar, aq, rg); stage_put(stage + A_ELTS + H_ELTS, ar, aq, ru); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = 0; c < nchunk; c++) {
        threadgroup ushort* cur = stage + (c & 1u) * STAGE;
        threadgroup ushort* nxt = stage + ((c + 1u) & 1u) * STAGE;
        bool more = c + 1u < nchunk;
        if (more) {
            uint k0 = (c + 1u) * GK;
            if (a_side) ra = load_arow(gg, mm + ar, k0 + aq * 4u);
            if (h_side) {
                rg = load_b16(gg, nn + ar, k0 + aq * 4u);
                ru = load_b16(gu, nn + ar, k0 + aq * 4u);
            }
        }
        threadgroup const bfloat* As = (threadgroup const bfloat*)cur;
        threadgroup const bfloat* Bg = As + A_ELTS;
        threadgroup const bfloat* Bu = Bg + H_ELTS;
        simdgroup_bfloat8x8 a0[RB], g0[CB], u0[CB], a1[RB], g1[CB], u1[CB];
        frag_load<RB, CB>(As, Bg, sgr, sgc, 0u, a0, g0);
        frag_load<RB, CB>(As, Bu, sgr, sgc, 0u, a0, u0);
        for (uint kb = 0; kb < GK / 8u; kb += 2u) {
            frag_load<RB, CB>(As, Bg, sgr, sgc, kb + 1u, a1, g1);
            frag_load<RB, CB>(As, Bu, sgr, sgc, kb + 1u, a1, u1);
            for (uint i = 0; i < RB; i++)
                for (uint j = 0; j < CB; j++) {
                    simdgroup_multiply_accumulate(ag[i][j], a0[i], g0[j], ag[i][j]);
                    simdgroup_multiply_accumulate(au[i][j], a0[i], u0[j], au[i][j]);
                }
            if (kb + 2u < GK / 8u) {
                frag_load<RB, CB>(As, Bg, sgr, sgc, kb + 2u, a0, g0);
                frag_load<RB, CB>(As, Bu, sgr, sgc, kb + 2u, a0, u0);
            }
            for (uint i = 0; i < RB; i++)
                for (uint j = 0; j < CB; j++) {
                    simdgroup_multiply_accumulate(ag[i][j], a1[i], g1[j], ag[i][j]);
                    simdgroup_multiply_accumulate(au[i][j], a1[i], u1[j], au[i][j]);
                }
        }
        if (more) {
            if (a_side) stage_put(nxt, ar, aq, ra);
            if (h_side) { stage_put(nxt + A_ELTS, ar, aq, rg); stage_put(nxt + A_ELTS + H_ELTS, ar, aq, ru); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Epilogue: gate and up 8x8 blocks side by side in the simdgroup's 128-float scratch.
    threadgroup float* scratch = (threadgroup float*)stage + sg * 128u;
    for (uint i = 0; i < RB; i++)
        for (uint j = 0; j < CB; j++) {
            simdgroup_store(ag[i][j], scratch, 8u);
            simdgroup_store(au[i][j], scratch + 64u, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint mb = mm + (sgr * RB + i) * 8u, nb = nn + (sgc * CB + j) * 8u;
            for (uint e = lane * 2u; e < lane * 2u + 2u; e++) {
                uint m = mb + e / 8u, n = nb + e % 8u;
                if (m >= m1 || n >= n1) continue;
                float gv = scratch[e], uv = scratch[64u + e];
                if (gg.ascale) { gv *= gg.ascale[m]; uv *= gg.ascale[m]; }
                if (fp8) { gv *= gg.wscale[n]; uv *= gu.wscale[n]; }
                if (gg.bias) gv += bf2f(gg.bias[n * gg.rs + gg.ro]);
                if (gu.bias) uv += bf2f(gu.bias[n * gu.rs + gu.ro]);
                C[m * N + n] = f2bf(fp8 ? act_gate_only(gv, act) * uv : glu_pair(gv, uv, act, f0, f1));
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
inline void gemm_tile2_glu(thread const GemmArgs& gg, thread const GemmArgs& gu, device ushort* C, uint act,
                           float f0, float f1, bool fp8, uint m0, uint m1, uint n0, uint n1,
                           threadgroup float* tile, uint lid, uint sg, uint lane) {
    threadgroup ushort* stage = (threadgroup ushort*)tile;
    for (uint mm = m0; mm < m1;) {
        uint rem = m1 - mm;
        uint sm = rem > 64u ? 128u : (rem > 32u ? 64u : 32u);
        for (uint nn = n0; nn < n1; nn += 64u) {
            if (sm == 128u) gemm_sub_glu<128, 2, 2>(gg, gu, C, act, f0, f1, fp8, mm, nn, m1, n1, stage, lid, sg, lane);
            else if (sm == 64u) gemm_sub_glu<64, 1, 2>(gg, gu, C, act, f0, f1, fp8, mm, nn, m1, n1, stage, lid, sg, lane);
            else gemm_sub_glu<32, 1, 1>(gg, gu, C, act, f0, f1, fp8, mm, nn, m1, n1, stage, lid, sg, lane);
        }
        mm += sm;
    }
}
// t0=C t1=A t2=B t7=bias?  i0=M i1=N i2=K i4=a_row0 i5=c_row0   (bf16)
// t0=C t1=A t2=B(e4m3) t3=a_scale? t4=w_scale i0=M i1=N i2=K i4=a_row0 i5=c_row0
// enc: 0 = bf16 weights, 1 = e4m3 + per-channel f32 scale (t4), 2 = MXFP4 (t3 = E8M0 scales)
void op_gemm(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint BM, uint BN, uint enc,
             threadgroup float* tile, uint lid, uint sg, uint lane) {
    uint M = in.i[0], N = in.i[1], K = in.i[2];
    bool fp8 = enc == 1u, mx4 = enc == 2u;
    device ushort* C = ten<ushort>(tab, in, 0) + in.i[5] * N;
    GemmArgs g;
    gemm_args_reset(g);
    g.A = ten<ushort>(tab, in, 1) + in.i[4] * K;
    if (fp8 && ten<float>(tab, in, 3)) {
        g.A8 = ten<uchar>(tab, in, 1) + in.i[4] * K;
        g.ascale = ten<float>(tab, in, 3) + in.i[4];
    }
    g.B16 = enc == 0u ? ten<ushort>(tab, in, 2) : (device const ushort*)0;
    g.B8 = fp8 ? ten<uchar>(tab, in, 2) : (device const uchar*)0;
    g.B4 = mx4 ? ten<uchar>(tab, in, 2) : (device const uchar*)0;
    g.S8 = mx4 ? ten<uchar>(tab, in, 3) : (device const uchar*)0;
    g.wscale = fp8 ? ten<float>(tab, in, 4) : (device const float*)0;
    g.bias = enc == 0u ? ten<ushort>(tab, in, 7) : (device const ushort*)0;
    g.K = K; g.M = M; g.N = N;
    bool poison = (fp8 && !g.wscale) || (mx4 && !g.S8);
    // The small tile op narrows to 64 columns only when 128-wide tiles would leave executors
    // idle (a 64-wide sub-tile halves the matrix work per simdgroup, so it must buy parallelism).
    if (BM == 64u && BN == 64u) {
        uint tm64 = (M + 63u) / 64u, tn128 = (N + 127u) / 128u;
        if (tm64 * tn128 >= nblk) BN = 128u;
    }
    uint tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    for (uint lin = slice; lin < tm * tn; lin += nblk) {
        uint m0 = (lin / tn) * BM, n0 = (lin % tn) * BN;
        uint m1 = min(m0 + BM, M), n1 = min(n0 + BN, N);
        if (poison) {
            for (uint e = lid; e < (m1 - m0) * (n1 - n0); e += NT)
                C[(m0 + e / (n1 - n0)) * N + n0 + e % (n1 - n0)] = ushort(0x7fc1);
            continue;
        }
        gemm_tile2(g, C, N, m0, m1, n0, n1, BN, tile, lid, sg, lane);
    }
}
void op_affine_q4(const thread Inst& in, device const ulong* tab, uint slice, uint nblk,
                  bool gemm, threadgroup float* tile, uint lid, uint sg, uint lane) {
    uint M = in.i[0], N = in.i[1], K = in.i[2];
    if (!nblk || slice >= nblk || !M || !N) return;
    device ushort* C = ten<ushort>(tab, in, 0);
    if (!C) return;
    C += ulong(in.i[5]) * N;
    GemmArgs g;
    gemm_args_reset(g);
    g.A = ten<ushort>(tab, in, 1);
    g.AQ4 = ten<uint>(tab, in, 2);
    g.AQScale = ten<ushort>(tab, in, 3);
    g.AQBias = ten<ushort>(tab, in, 4);
    g.M = M; g.N = N; g.K = K;
    bool valid = K && !(K % 64u) && !in.i[3] && g.A && g.AQ4 && g.AQScale && g.AQBias;
    if (valid) g.A += ulong(in.i[4]) * K;
    if (gemm) {
        ulong tn = (ulong(N) + 63u) / 64u, tm = (ulong(M) + 63u) / 64u;
        for (ulong t = slice; t < tm * tn; t += nblk) {
            uint m0 = uint(t / tn) * 64u, n0 = uint(t % tn) * 64u;
            uint m1 = m0 + min(64u, M - m0), n1 = n0 + min(64u, N - n0);
            if (valid) gemm_tile2(g, C, N, m0, m1, n0, n1, 64u, tile, lid, sg, lane);
            else for (uint e = lid; e < (m1 - m0) * (n1 - n0); e += NT)
                C[ulong(m0 + e / (n1 - n0)) * N + n0 + e % (n1 - n0)] = ushort(0x7fc1);
        }
        return;
    }
    uint n0, n1;
    range(N, slice, nblk, n0, n1);
    for (uint m = 0; m < M; m++) for (uint n = n0 + sg * 4u; n < n1; n += NSG * 4u) {
        float acc[4] = {0, 0, 0, 0};
        uint rows = min(4u, n1 - n);
        if (valid) for (uint k = lane * 16u; k < K; k += 512u) {
            float xs[16], sum = 0;
            for (uint j = 0; j < 16u; j += 4u) {
                float4 x = bf4(*(device const ushort4*)(g.A + ulong(m) * K + k + j));
                sum += rbf(rbf(rbf(x.x + x.y) + x.z) + x.w);
                xs[j] = x.x; xs[j + 1] = x.y / 16.0f;
                xs[j + 2] = x.z / 256.0f; xs[j + 3] = x.w / 4096.0f;
            }
            for (uint r = 0; r < rows; r++) {
                device const uint* w = g.AQ4 + ulong(n + r) * (K / 8u) + k / 8u;
                float dot = 0;
                for (uint j = 0; j < 4u; j++) {
                    uint h = (w[j / 2u] >> ((j % 2u) * 16u)) & 65535u;
                    dot += float(h & 15u) * xs[j * 4u] + float(h & 240u) * xs[j * 4u + 1u]
                         + float(h & 3840u) * xs[j * 4u + 2u] + float(h & 61440u) * xs[j * 4u + 3u];
                }
                ulong p = ulong(n + r) * (K / 64u) + k / 64u;
                acc[r] += bf2f(g.AQScale[p]) * dot + bf2f(g.AQBias[p]) * sum;
            }
        }
        for (uint r = 0; r < rows; r++) {
            float v = simd_sum(acc[r]);
            if (lane == 0) C[ulong(m) * N + n + r] = valid ? f2bf(v) : ushort(0x7fc1);
        }
    }
}

// GLU: t0=fu t1=A t2=Wg t5=Wu t6=bias_g? t7=bias_u? i5=act (bf16) / t3=a_scale? t4=g_scale t6=u_scale (fp8)
// 64x32 sub-tiles: chunk buffer = A 64x16 | Bg 32x16 | Bu 32x16 (2048 floats), double-buffered;
// simdgroup sg owns the 8x8 block (rows sg/4*8, cols sg%4*8): one gate + one up accumulator.
void op_gemm_glu(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint enc,
                 threadgroup float* tile, uint lid, uint sg, uint lane) {
    uint M = in.i[0], N = in.i[1], K = in.i[2], act = in.i[5];
    bool fp8 = enc == 1u, mx4 = enc == 2u;
    float f0 = as_type<float>(in.fj[0]), f1 = as_type<float>(in.fj[1]);
    device ushort* C = ten<ushort>(tab, in, 0);
    GemmArgs gg, gu;
    gemm_args_reset(gg);
    gemm_args_reset(gu);
    gg.A = gu.A = ten<ushort>(tab, in, 1);
    gg.K = gu.K = K; gg.M = gu.M = M; gg.N = gu.N = N;
    gg.B16 = gu.B16 = (device const ushort*)0;
    gg.B8 = gu.B8 = (device const uchar*)0;
    gg.B4 = gu.B4 = (device const uchar*)0;
    gg.S8 = gu.S8 = (device const uchar*)0;
    gg.wscale = gu.wscale = (device const float*)0;
    gg.bias = gu.bias = (device const ushort*)0;
    if (fp8) {
        gg.B8 = ten<uchar>(tab, in, 2); gu.B8 = ten<uchar>(tab, in, 5);
        gg.wscale = ten<float>(tab, in, 4); gu.wscale = ten<float>(tab, in, 6);
        gg.ascale = gu.ascale = ten<float>(tab, in, 3);
        if (gg.ascale) gg.A8 = gu.A8 = ten<uchar>(tab, in, 1);
    } else if (mx4) {
        gg.B4 = ten<uchar>(tab, in, 2); gu.B4 = ten<uchar>(tab, in, 5);
        gg.S8 = ten<uchar>(tab, in, 3); gu.S8 = ten<uchar>(tab, in, 4);
    } else {
        gg.B16 = ten<ushort>(tab, in, 2); gu.B16 = ten<ushort>(tab, in, 5);
        gg.bias = ten<ushort>(tab, in, 6); gu.bias = ten<ushort>(tab, in, 7);
    }
    bool poison = (fp8 && (!gg.wscale || !gu.wscale)) || (mx4 && (!gg.S8 || !gu.S8));
    const uint BM = 256, BN = 128;
    uint tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    for (uint lin = slice; lin < tm * tn; lin += nblk) {
        uint m0 = (lin / tn) * BM, n0 = (lin % tn) * BN;
        uint m1 = min(m0 + BM, M), n1 = min(n0 + BN, N);
        if (poison) {
            for (uint e = lid; e < (m1 - m0) * (n1 - n0); e += NT)
                C[(m0 + e / (n1 - n0)) * N + n0 + e % (n1 - n0)] = ushort(0x7fc1);
            continue;
        }
        gemm_tile2_glu(gg, gu, C, act, f0, f1, fp8, m0, m1, n0, n1, tile, lid, sg, lane);
    }
}

inline float tg_max(float v, threadgroup float* red, uint lid, uint sg, uint lane) {
    v = simd_max(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint k = 0; k < NSG; k++) s = max(s, red[k]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}
inline void quant_row(device const ushort* x, device uchar* q, device float* scale, uint K,
                      threadgroup float* red, uint lid, uint sg, uint lane) {
    float amax = 0.0f;
    for (uint k = lid; k < K; k += NT) amax = max(amax, abs(bf2f(x[k])));
    float s = max(tg_max(amax, red, lid, sg, lane) * (1.0f / 448.0f), 1e-12f);
    if (lid == 0) *scale = s;
    for (uint k = lid; k < K; k += NT) q[k] = f2e4m3(bf2f(x[k]) * (1.0f / s));
}
void op_quant_fp8(const thread Inst& in, device const ulong* tab, uint slice, uint nblk,
                  threadgroup float* red, uint lid, uint sg, uint lane) {
    device uchar* q = ten<uchar>(tab, in, 0);
    device ushort* x = ten<ushort>(tab, in, 1);
    device float* scales = ten<float>(tab, in, 2);
    device const ushort* gate = ten<ushort>(tab, in, 3);
    device const ushort* up = ten<ushort>(tab, in, 4);
    uint K = in.i[1], lo, hi;
    range(in.i[0], slice, nblk, lo, hi);
    for (uint m = lo; m < hi; m++) {
        ulong row = ulong(m) * K;
        if (gate) {
            for (uint k = lid; k < K; k += NT)
                x[row + k] = f2bf(act_gate_only(bf2f(gate[row + k]), in.i[2]) * bf2f(up[row + k]));
            threadgroup_barrier(mem_flags::mem_device);
        }
        quant_row(x + row, q + row, scales + m, K, red, lid, sg, lane);
    }
}

// ---- norms (rows = slice axis; whole threadgroup per row) ------------------------------------------
// t0=out t1=x t2=gamma? t3=quant? t4=scale? i0=rows i1=feat i2=out_row0 f0=eps
void op_rmsnorm(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, threadgroup float* red,
                uint lid, uint sg, uint lane) {
    device ushort* out = ten<ushort>(tab, in, 0);
    device const ushort* x = ten<ushort>(tab, in, 1);
    device const ushort* gamma = ten<ushort>(tab, in, 2);
    bool quant = ten<uchar>(tab, in, 3) != 0;
    uint rows = in.i[0], feat = in.i[1], out_row0 = in.i[2];
    float eps = as_type<float>(in.fj[0]);
    for (uint row = slice; row < rows; row += nblk) {
        device const ushort* xr = x + row * feat;
        device ushort* o = out + (out_row0 + row) * feat;
        float ss = 0.0f;
        for (uint i = lid; i < feat; i += NT) { float v = bf2f(xr[i]); ss += v * v; }
        ss = tg_sum(ss, red, lid, sg, lane);
        float inv = rsqrt(ss / float(feat) + eps);
        for (uint i = lid; i < feat; i += NT) {
            float g = gamma ? bf2f(gamma[i]) : 1.0f;
            o[i] = f2bf(bf2f(xr[i]) * inv * g);
        }
        if (quant) {
            threadgroup_barrier(mem_flags::mem_device);
            quant_row(o, ten<uchar>(tab, in, 3) + row * feat, ten<float>(tab, in, 4) + row,
                      feat, red, lid, sg, lane);
        }
    }
}
// t0=out t1=a t2=b t3=gamma?  i0=rows i1=feat  f0=eps f1=scale : out = (a + RMSNorm(b, gamma)) * scale
void op_norm_residual(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, threadgroup float* red,
                      uint lid, uint sg, uint lane) {
    device ushort* out = ten<ushort>(tab, in, 0);
    device const ushort* a = ten<ushort>(tab, in, 1);
    device const ushort* b = ten<ushort>(tab, in, 2);
    device const ushort* gamma = ten<ushort>(tab, in, 3);
    uint rows = in.i[0], feat = in.i[1];
    float eps = as_type<float>(in.fj[0]), scale = as_type<float>(in.fj[1]);
    for (uint row = slice; row < rows; row += nblk) {
        uint base = row * feat;
        float ss = 0.0f;
        for (uint i = lid; i < feat; i += NT) { float v = bf2f(b[base + i]); ss += v * v; }
        ss = tg_sum(ss, red, lid, sg, lane);
        float inv = rsqrt(ss / float(feat) + eps);
        for (uint i = lid; i < feat; i += NT) {
            float g = gamma ? bf2f(gamma[i]) : 1.0f;
            out[base + i] = f2bf((bf2f(a[base + i]) + bf2f(b[base + i]) * inv * g) * scale);
        }
    }
}
// t0=out t1=resid t2=a t3=b t4=gamma?  i0=rows i1=feat  f0=eps : resid = a + b ; out = RMSNorm(resid)
void op_add_norm(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, threadgroup float* red,
                 uint lid, uint sg, uint lane) {
    device ushort* out = ten<ushort>(tab, in, 0);
    device ushort* resid = ten<ushort>(tab, in, 1);
    device const ushort* a = ten<ushort>(tab, in, 2);
    device const ushort* b = ten<ushort>(tab, in, 3);
    device const ushort* gamma = ten<ushort>(tab, in, 4);
    uint rows = in.i[0], feat = in.i[1];
    float eps = as_type<float>(in.fj[0]);
    for (uint row = slice; row < rows; row += nblk) {
        uint base = row * feat;
        float ss = 0.0f;
        for (uint i = lid; i < feat; i += NT) { float f = bf2f(a[base + i]) + bf2f(b[base + i]); ss += f * f; }
        ss = tg_sum(ss, red, lid, sg, lane);
        float inv = rsqrt(ss / float(feat) + eps);
        for (uint i = lid; i < feat; i += NT) {
            float g = gamma ? bf2f(gamma[i]) : 1.0f;
            float f = bf2f(a[base + i]) + bf2f(b[base + i]);
            resid[base + i] = f2bf(f);
            out[base + i] = f2bf(f * inv * g);
        }
    }
}
// t0=out t1=resid t2=a t3=b t4=gamma_b? t5=gamma_n?  i0=rows i1=feat  f0=eps f1=scale
void op_norm_residual_norm(const thread Inst& in, device const ulong* tab, uint slice, uint nblk,
                           threadgroup float* red, uint lid, uint sg, uint lane) {
    device ushort* out = ten<ushort>(tab, in, 0);
    device ushort* resid = ten<ushort>(tab, in, 1);
    device const ushort* a = ten<ushort>(tab, in, 2);
    device const ushort* b = ten<ushort>(tab, in, 3);
    device const ushort* gb = ten<ushort>(tab, in, 4);
    device const ushort* gn = ten<ushort>(tab, in, 5);
    uint rows = in.i[0], feat = in.i[1];
    float eps = as_type<float>(in.fj[0]), scale = as_type<float>(in.fj[1]);
    for (uint row = slice; row < rows; row += nblk) {
        uint base = row * feat;
        float ss = 0.0f;
        for (uint i = lid; i < feat; i += NT) { float v = bf2f(b[base + i]); ss += v * v; }
        ss = tg_sum(ss, red, lid, sg, lane);
        float invb = rsqrt(ss / float(feat) + eps);
        float ssr = 0.0f;
        for (uint i = lid; i < feat; i += NT) {
            float g = gb ? bf2f(gb[i]) : 1.0f;
            ushort rb = f2bf((bf2f(a[base + i]) + bf2f(b[base + i]) * invb * g) * scale);
            resid[base + i] = rb;
            float rf = bf2f(rb);
            ssr += rf * rf;
        }
        ssr = tg_sum(ssr, red, lid, sg, lane);
        float invr = rsqrt(ssr / float(feat) + eps);
        for (uint i = lid; i < feat; i += NT) {
            float g = gn ? bf2f(gn[i]) : 1.0f;
            out[base + i] = f2bf(bf2f(resid[base + i]) * invr * g);
        }
    }
}

// ---- HEADNORM_ROPE: (token, head) items packed 8 per workgroup, one simdgroup each ----------------
// t0=out t1=x t2=gamma? t3=cos? t4=sin? t5=pos(i32)?
// i0=ntok i1=nhead i2=hd i3=out_row0 i4=skip_norm i5=rope_form i6=n_batch_kv  f0=eps fj1=out_stride fj2=kv_mask
template <bool FP8>
void op_headnorm_rope(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint sg, uint lane) {
    device ushort* out = ten<ushort>(tab, in, 0);
    device const ushort* x = ten<ushort>(tab, in, 1);
    device const ushort* gamma = ten<ushort>(tab, in, 2);
    device const float* cosb = ten<float>(tab, in, 3);
    device const float* sinb = ten<float>(tab, in, 4);
    device const int* pos = ten<int>(tab, in, 5);
    uint ntok = in.i[0], nhead = in.i[1], hd = in.i[2], out_row0 = in.i[3];
    uint skip_norm = in.i[4], n_batch_kv = in.i[6];
    uint out_stride = in.fj[1], kv_mask = in.fj[2];
    float eps = as_type<float>(in.fj[0]);
    bool interleave = in.i[5] == 2u ? false : (hd == 64u) || (hd == 128u && in.i[5] == 1u);
    uint H2 = hd >> 1, total = ntok * nhead;
    if (hd > 512u || (hd & 63u)) return;   // lane mapping needs hd % 64 == 0 (every model here)
    uint per = hd / 32u;                    // elements per lane: i = lane + 32*j
    for (uint w0 = slice * WAVES; w0 < total; w0 += nblk * WAVES) {
        uint w = w0 + sg;
        if (sg >= WAVES || w >= total) continue;
        uint t = w / nhead, hh = w % nhead;
        uint position = pos ? uint(pos[t]) : out_row0 + t;
        device const ushort* xr = x + (t * nhead + hh) * hd;
        ulong obase = out_stride
            ? (n_batch_kv != 0u ? (ulong(t * nhead + hh) * out_stride + (position & kv_mask)) * hd
                                : (ulong(hh) * out_stride + ((out_row0 + t) & kv_mask)) * hd)
            : (ulong(out_row0 + t) * nhead + hh) * hd;
        float v[16];
        float ss = 0.0f;
        for (uint j = 0; j < per; j++) { v[j] = bf2f(xr[lane + 32u * j]); ss += v[j] * v[j]; }
        ss = simd_sum(ss);
        float inv = skip_norm ? 1.0f : rsqrt(ss / float(hd) + eps);
        for (uint j = 0; j < per; j++) v[j] = v[j] * inv * (gamma ? bf2f(gamma[lane + 32u * j]) : 1.0f);
        if (cosb) {
            ulong p = ulong(position) * H2;
            float r[16];
            if (!interleave) {
                uint hj = per / 2u;   // j < hj: first half, partner j + hj
                for (uint j = 0; j < hj; j++) {
                    uint i = lane + 32u * j;
                    float c = cosb[p + i], s = sinb[p + i];
                    r[j] = v[j] * c - v[j + hj] * s;
                    r[j + hj] = v[j + hj] * c + v[j] * s;
                }
            } else {
                for (uint j = 0; j < per; j++) {
                    uint i = lane + 32u * j;
                    float c = cosb[p + (i >> 1)], s = sinb[p + (i >> 1)];
                    float partner = simd_shuffle_xor(v[j], 1);
                    r[j] = (i & 1u) == 0u ? v[j] * c - partner * s : v[j] * c + partner * s;
                }
            }
            for (uint j = 0; j < per; j++) v[j] = r[j];
        }
        if (FP8) {
            float amax = 0.0f;
            for (uint j = 0; j < per; j++) amax = max(amax, abs(v[j]));
            amax = simd_max(amax);
            float s = amax * (1.0f / 448.0f), inv = amax > 0.0f ? 448.0f / amax : 0.0f;
            if (lane == 0) ten<float>(tab, in, 6)[obase / hd] = s;
            for (uint j = 0; j < per; j++)
                ten<uchar>(tab, in, 0)[obase + lane + 32u * j] = f2e4m3(v[j] * inv);
        } else {
            for (uint j = 0; j < per; j++) out[obase + lane + 32u * j] = f2bf(v[j]);
        }
    }
}

// ---- pointwise ---------------------------------------------------------------------------------------
void op_glu(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint lid) {
    device ushort* out = ten<ushort>(tab, in, 0);
    device const ushort* gate = ten<ushort>(tab, in, 1);
    device const ushort* up = ten<ushort>(tab, in, 2);
    float f0 = as_type<float>(in.fj[0]), f1 = as_type<float>(in.fj[1]);
    uint lo, hi;
    range(in.i[0], slice, nblk, lo, hi);
    for (uint i = lo + lid; i < hi; i += NT)
        out[i] = f2bf(glu_pair(bf2f(gate[i]), bf2f(up[i]), in.i[1], f0, f1));
}
void op_residual(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint lid) {
    device ushort* out = ten<ushort>(tab, in, 0);
    device const ushort* a = ten<ushort>(tab, in, 1);
    device const ushort* b = ten<ushort>(tab, in, 2);
    device const ushort* pre = ten<ushort>(tab, in, 3);
    float scale = as_type<float>(in.fj[0]);
    uint lo, hi;
    range(in.i[0], slice, nblk, lo, hi);
    for (uint i = lo + lid; i < hi; i += NT) {
        float s = bf2f(a[i]) + bf2f(b[i]);
        out[i] = pre ? f2bf((bf2f(pre[i]) + rbf(s)) * scale) : f2bf(s * scale);
    }
}
void op_softcap(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint lid) {
    device ushort* out = ten<ushort>(tab, in, 0);
    device const ushort* x = ten<ushort>(tab, in, 1);
    float cap = as_type<float>(in.fj[0]), inv = 1.0f / cap;
    uint lo, hi;
    range(in.i[0], slice, nblk, lo, hi);
    for (uint i = lo + lid; i < hi; i += NT) out[i] = f2bf(cap * tanhs(bf2f(x[i]) * inv));
}
void op_embed(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint lid) {
    device ushort* out = ten<ushort>(tab, in, 0);
    device const ushort* table = ten<ushort>(tab, in, 1);
    device const int* ids = ten<int>(tab, in, 2);
    uint ntok = in.i[0], hidden = in.i[1];
    float scale = as_type<float>(in.fj[0]);
    for (uint t = slice; t < ntok; t += nblk) {
        device const ushort* src = table + ulong(ids[t]) * hidden;
        device ushort* dst = out + t * hidden;
        for (uint i = lid; i < hidden; i += NT) dst[i] = f2bf(bf2f(src[i]) * scale);
    }
}
// t0=part(u64[n_batch][nblk]) t1=x  i0=n i1=n_batch
void op_argmax(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, threadgroup ulong* keys, uint lid) {
    device ulong* part = ten<ulong>(tab, in, 0);
    device const ushort* x = ten<ushort>(tab, in, 1);
    uint n = in.i[0], B = in.i[1] ? in.i[1] : 1u;
    uint lo, hi;
    range(n, slice, nblk, lo, hi);
    for (uint b = 0; b < B; b++) {
        device const ushort* xb = x + b * n;
        ulong best = 0;
        for (uint i = lo + lid; i < hi; i += NT) { ulong p = amax_pack(xb[i], i); best = p > best ? p : best; }
        keys[lid] = best;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = NT / 2; s > 0; s >>= 1) {
            if (lid < s && keys[lid + s] > keys[lid]) keys[lid] = keys[lid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0) part[b * nblk + slice] = keys[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
void op_argmax_fin(const thread Inst& in, device const ulong* tab, uint slice, uint lid) {
    if (slice != 0u || lid != 0u) return;
    device int* ids = ten<int>(tab, in, 0);
    device const ulong* part = ten<ulong>(tab, in, 1);
    uint nparts = in.i[0], B = in.i[1] ? in.i[1] : 1u;
    for (uint b = 0; b < B; b++) {
        ulong best = 0;
        for (uint i = 0; i < nparts; i++) { ulong p = part[b * nparts + i]; best = p > best ? p : best; }
        ids[b] = int(~uint(best & 0xFFFFFFFFul));
    }
}

// ---- flash family (golden/attention.c): one simdgroup per query row / head, lanes over D ------------
constant uint FA_BQ_TILE = 128, FA_BKV = 32, FA_GF = 2;

// Online-softmax accumulation of one query against KV rows [lo, hi) (validity by caller's predicate).
// q/acc are per-lane slices: element d = lane + 32*j. Returns (m, l) via refs.
inline float kv_value(device const ushort* row, device const float* scales, uint r, uint d) { return bf2f(row[d]); }
inline float kv_value(device const uchar* row, device const float* scales, uint r, uint d) { return e4m3_exact(row[d]) * scales[r]; }
inline ushort kv_raw(device const ushort* row, uint d) { return row[d]; }
inline ushort kv_raw(device const uchar* row, uint d) { return f2bf(e4m3_exact(row[d])); }

template <uint D, typename KV>
void flash_prefill_tile(const thread Inst& in, device const ulong* tab, uint slice, uint nblk,
                        threadgroup float* tile, uint sg, uint lane) {
    device float* Opart = ten<float>(tab, in, 0);
    device float* mlpart = ten<float>(tab, in, 1);
    device const ushort* Q = ten<ushort>(tab, in, 2);
    device const KV* K = ten<KV>(tab, in, 3);
    device const KV* V = ten<KV>(tab, in, 4);
    device ushort* O = ten<ushort>(tab, in, 5);
    device const float* KS = ten<float>(tab, in, 6);
    device const float* VS = ten<float>(tab, in, 7);
    uint nq = in.i[0], nkv = in.i[1], nh = in.i[2], nkh = in.i[3];
    uint pos = in.i[4], window = in.i[5], splits = max(in.i[7], 1u);
    uint stride = in.fj[1], mask = in.fj[2], gqa = nh / nkh;
    float scale = as_type<float>(in.fj[0]);
    uint lid = sg * 32u + lane;
    threadgroup float* scores = tile;
    threadgroup ushort* qs = (threadgroup ushort*)(tile + 1024);
    threadgroup ushort* kvs = (threadgroup ushort*)(tile + 2048);
    threadgroup float* ps = tile + 3072;
    threadgroup float* state = tile + 4096;
    threadgroup float* scratch = tile + 4192 + sg * 64u;
    uint work = ((nq + 31u) / 32u) * nh * splits;
    for (uint w = slice; w < work; w += nblk) {
        uint sp = w % splits, h = (w / splits) % nh, qb = w / (splits * nh) * 32u;
        uint isa_qb = qb / FA_BQ_TILE * FA_BQ_TILE;
        uint end = min(pos + isa_qb + FA_BQ_TILE, nkv);
        uint first = window && pos + isa_qb >= window ? pos + isa_qb - window + 1u : 0u;
        uint lo = first / FA_BKV * FA_BKV;
        uint tiles = end > lo ? (end - lo + FA_BKV - 1u) / FA_BKV : 0u;
        uint per = (tiles + splits - 1u) / splits;
        uint hi = min(lo + (sp + 1u) * per * FA_BKV, end);
        lo += sp * per * FA_BKV;
        // Preserve the ISA's split ownership, but skip tiles masked for every query here.
        uint valid_first = window && pos + qb >= window ? pos + qb - window + 1u : 0u;
        lo = max(lo, valid_first / FA_BKV * FA_BKV);
        hi = min(hi, pos + min(qb + 32u, nq));
        ulong base = ulong(h / gqa) * stride;
        simdgroup_float8x8 out[D / 64u];
        for (uint dc = 0; dc < D / 64u; dc++) out[dc] = simdgroup_float8x8(0.0f);
        float m = NEG_INF, l = 0.0f;
        if (lane == 0) { state[sg] = m; state[32u + sg] = l; }
        for (uint kb = lo; kb < hi; kb += 32u) {
            simdgroup_float8x8 score(0.0f);
            for (uint dc = 0; dc < D; dc += 64u) {
                for (uint e = lid; e < 32u * 64u; e += NT) {
                    uint r = e / 64u, d = dc + e % 64u;
                    qs[e] = qb + r < nq ? Q[(ulong(qb + r) * nh + h) * D + d] : ushort(0);
                    kvs[e] = kb + r < hi ? kv_raw(K + (base + ((kb + r) & mask)) * D, d) : ushort(0);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg < 16u) {
                    for (uint k = 0; k < 64u; k += 8u) {
                        simdgroup_bfloat8x8 a, b;
                        simdgroup_load(a, (threadgroup bfloat*)qs + (sg / 4u) * 8u * 64u + k, 64u);
                        simdgroup_load(b, (threadgroup bfloat*)kvs + (sg % 4u) * 8u * 64u + k, 64u, ulong2(0), true);
                        simdgroup_multiply_accumulate(score, a, b, score);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (sg < 16u) simdgroup_store(score, scores + (sg / 4u) * 8u * 32u + (sg % 4u) * 8u, 32u);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint qpos = pos + qb + sg, key = kb + lane;
            bool valid = qb + sg < nq && key < hi && key <= qpos && (!window || qpos - key < window);
            float s = scores[sg * 32u + lane] * scale;
            if (sizeof(KV) == 1u && key < hi) s *= KS[base + (key & mask)];
            s = valid ? s : NEG_INF;
            float next = max(m, simd_max(s));
            float corr = m == NEG_INF ? 0.0f : exp(m - next);
            // The golden contract rounds P at each key's running maximum. Prefix maxima
            // preserve that rounding before expressing the whole tile at its final maximum.
            float prefix = max(m, s);
            for (uint offset = 1u; offset < 32u; offset *= 2u) {
                float prev = simd_shuffle_up(prefix, offset);
                if (lane >= offset) prefix = max(prefix, prev);
            }
            float p = valid ? rbf(exp(s - prefix)) * exp(prefix - next) : 0.0f;
            l = l * corr + simd_sum(p);
            m = next;
            ps[sg * 32u + lane] = p;
            if (lane == 0) { state[sg] = m; state[32u + sg] = l; state[64u + sg] = corr; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint dc = 0; dc < D / 64u; dc++) {
                for (uint e = lid; e < 32u * 64u; e += NT) {
                    uint r = e / 64u, d = dc * 64u + e % 64u;
                    ulong row = base + ((kb + r) & mask);
                    (tile + 1024)[e] = kb + r < hi ? kv_value(V + row * D, VS, uint(row), d) : 0.0f;
                }
                uint qr = sg / 8u * 8u, col = sg % 8u * 8u;
                simdgroup_store(out[dc], scratch, 8u);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64u; e += 32u) scratch[e] *= state[64u + qr + e / 8u];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(out[dc], scratch, 8u);
                for (uint k = 0; k < 32u; k += 8u) {
                    simdgroup_float8x8 a, b;
                    simdgroup_load(a, ps + qr * 32u + k, 32u);
                    simdgroup_load(b, tile + 1024 + k * 64u + col, 64u);
                    simdgroup_multiply_accumulate(out[dc], a, b, out[dc]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint dc = 0; dc < D / 64u; dc++) {
            simdgroup_store(out[dc], scratch, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64u; e += 32u) {
                uint qr = sg / 8u * 8u + e / 8u, q = qb + qr;
                uint d = dc * 64u + sg % 8u * 8u + e % 8u;
                if (q < nq) {
                    if (splits == 1u && O) {
                        float den = state[32u + qr];
                        O[(ulong(q) * nh + h) * D + d] = f2bf(den > 0.0f ? scratch[e] / den : 0.0f);
                    } else Opart[((ulong(q) * nh + h) * splits + sp) * D + d] = scratch[e];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (!(splits == 1u && O) && qb + sg < nq && lane == 0) {
            ulong off = ((ulong(qb + sg) * nh + h) * splits + sp) * 2u;
            mlpart[off] = m; mlpart[off + 1u] = l;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
template <typename KV>
inline void attend_rows(device const ushort* q, device const KV* kbase, device const KV* vbase,
                        device const float* ks, device const float* vs,
                        uint D, uint kv_mask, float scale, uint lo, uint hi, uint qpos, uint window, bool bf16_p,
                        uint lane, thread float (&acc)[16], thread float& m, thread float& l) {
    uint per = D / 32u;
    float qv[16];
    for (uint j = 0; j < per; j++) qv[j] = bf2f(q[lane + 32u * j]);
    for (uint kv = lo; kv < hi; kv++) {
        if (!(kv <= qpos && (window == 0u || qpos - kv < window))) continue;
        uint r = kv & kv_mask;
        device const KV* kr = kbase + ulong(r) * D;
        device const KV* vr = vbase + ulong(r) * D;
        float s = 0.0f;
        for (uint j = 0; j < per; j++) s += qv[j] * kv_value(kr, ks, r, lane + 32u * j);
        s = simd_sum(s) * scale;
        float mnew = max(m, s);
        float corr = m == NEG_INF ? 0.0f : exp(m - mnew);
        float pe = exp(s - mnew);
        if (bf16_p) pe = rbf(pe);
        l = l * corr + pe;
        m = mnew;
        for (uint j = 0; j < per; j++) acc[j] = acc[j] * corr + pe * kv_value(vr, vs, r, lane + 32u * j);
    }
}
// t0=Opart t1=mlpart t2=Q t3=K t4=V t5=O_final?  i0=n_q i1=n_kv i2=n_head i3=n_kv_head i4=q_pos0 i5=window
// i6=hd i7=nsplit  f0=scale fj1=kv_stride fj2=kv_mask
template <typename KV>
void op_flash_prefill(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, threadgroup float* tile, uint sg, uint lane) {
    switch (in.i[6]) {
        case 64: flash_prefill_tile<64, KV>(in, tab, slice, nblk, tile, sg, lane); return;
        case 128: flash_prefill_tile<128, KV>(in, tab, slice, nblk, tile, sg, lane); return;
        case 256: flash_prefill_tile<256, KV>(in, tab, slice, nblk, tile, sg, lane); return;
        case 512: flash_prefill_tile<512, KV>(in, tab, slice, nblk, tile, sg, lane); return;
    }
    device float* Opart = ten<float>(tab, in, 0);
    device float* mlpart = ten<float>(tab, in, 1);
    device const ushort* Q = ten<ushort>(tab, in, 2);
    device const KV* K = ten<KV>(tab, in, 3);
    device const KV* V = ten<KV>(tab, in, 4);
    device const float* KS = ten<float>(tab, in, 6);
    device const float* VS = ten<float>(tab, in, 7);
    device ushort* O_final = ten<ushort>(tab, in, 5);
    uint n_q = in.i[0], n_kv = in.i[1], n_head = in.i[2], n_kv_head = in.i[3];
    uint q_pos0 = in.i[4], window = in.i[5], D = in.i[6];
    uint nsplit = in.i[7] ? in.i[7] : 1u;
    float scale = as_type<float>(in.fj[0]);
    uint kv_stride = in.fj[1], kv_mask = in.fj[2];
    if (D > 512u || (D & 31u)) return;
    uint gqa = n_head / n_kv_head;
    uint q_tiles = (n_q + NSG - 1) / NSG;
    uint n_work = q_tiles * n_head * nsplit;
    uint per = D / 32u;
    for (uint w = slice; w < n_work; w += nblk) {
        uint sp = w % nsplit, h = (w / nsplit) % n_head, qt = w / (nsplit * n_head);
        uint hkv = h / gqa;
        uint q_base = qt * NSG;
        // KV split ownership stays on the ISA's 128-query tile, even though scheduling uses
        // one query per simdgroup. Changing that boundary changes partial-softmax outputs.
        uint split_q_base = q_base / FA_BQ_TILE * FA_BQ_TILE;
        uint q_tile_last = q_pos0 + split_q_base + FA_BQ_TILE - 1;
        uint kv_end = min(q_tile_last + 1, n_kv);
        uint q_tile_first = q_pos0 + split_q_base;
        uint win_lo = (window && q_tile_first >= window) ? q_tile_first - window + 1 : 0;
        uint kv_lo = (win_lo / FA_BKV) * FA_BKV;
        uint tiles_kv = kv_end > kv_lo ? (kv_end - kv_lo + FA_BKV - 1) / FA_BKV : 0u;
        uint perp = (tiles_kv + nsplit - 1) / nsplit;
        uint my_lo = kv_lo + sp * perp * FA_BKV;
        uint my_hi = min(kv_lo + (sp + 1) * perp * FA_BKV, kv_end);
        device const KV* kbase = K + ulong(hkv) * kv_stride * D;
        device const KV* vbase = V + ulong(hkv) * kv_stride * D;
        device const float* ks = KS ? KS + ulong(hkv) * kv_stride : KS;
        device const float* vs = VS ? VS + ulong(hkv) * kv_stride : VS;
        for (uint qi = q_base + sg; qi < q_base + NSG && qi < n_q; qi += NSG) {
            device const ushort* q = Q + (ulong(qi) * n_head + h) * D;
            uint qg = q_pos0 + qi;
            float m = NEG_INF, l = 0.0f, acc[16];
            for (uint j = 0; j < 16; j++) acc[j] = 0.0f;
            // rows past n_kv are excluded by my_hi <= kv_end <= n_kv
            attend_rows(q, kbase, vbase, ks, vs, D, kv_mask, scale, my_lo, my_hi, qg, window, true, lane, acc, m, l);
            if (nsplit == 1u && O_final) {
                float inv = l > 0.0f ? 1.0f / l : 0.0f;
                device ushort* orow = O_final + (ulong(qi) * n_head + h) * D;
                for (uint j = 0; j < per; j++) orow[lane + 32u * j] = f2bf(acc[j] * inv);
                continue;
            }
            device float* op = Opart + ((ulong(qi) * n_head + h) * nsplit + sp) * D;
            for (uint j = 0; j < per; j++) op[lane + 32u * j] = acc[j];
            if (lane == 0) {
                device float* ml = mlpart + ((ulong(qi) * n_head + h) * nsplit + sp) * 2;
                ml[0] = m; ml[1] = l;
            }
        }
    }
}
// t0=Opart t1=mlpart t2=Q t3=K t4=V t5=kv_len(i32)  i0=n_batch i1=n_head i2=n_kv_head i3=kv_stride i4=window
// i5=nsplit i6=hd i7=kv_mask  f0=scale   (i1 bit 16 NRF fold -> poison)
template <typename KV>
void op_flash_decode(const thread Inst& in, device const ulong* tab, uint slice, uint nblk,
                     threadgroup float* tile, uint sg, uint lane) {
    device float* Opart = ten<float>(tab, in, 0);
    device float* mlpart = ten<float>(tab, in, 1);
    device const ushort* Q = ten<ushort>(tab, in, 2);
    device const KV* K = ten<KV>(tab, in, 3);
    device const KV* V = ten<KV>(tab, in, 4);
    device const float* KS = ten<float>(tab, in, 6);
    device const float* VS = ten<float>(tab, in, 7);
    device const int* kv_len = ten<int>(tab, in, 5);
    bool nrf = (in.i[1] & 0x10000u) != 0u;
    uint n_batch = nrf ? (in.i[0] & 0xFFu) : in.i[0];
    uint n_head = in.i[1] & 0xFFFFu, n_kv_head = in.i[2];
    uint kv_stride = nrf ? (in.i[3] & 0xFFFFFu) : in.i[3];
    uint window = nrf ? (in.i[0] >> 8) : in.i[4];
    uint nsplit = nrf ? (in.i[3] >> 20) : in.i[5];
    uint D = in.i[6] & 0xFFFFu, kv_mask = in.i[7];
    float scale = as_type<float>(in.fj[0]);
    if (D > 512u || (D & 31u) || nsplit == 0u) return;
    uint gqa = n_head / n_kv_head;
#ifdef PLOW_DECODE_HEADS
    uint gf = 1u;
#else
    uint gf = gqa % FA_GF == 0u ? FA_GF : 1u;
#endif
    uint n_grp = (n_head + gf - 1) / gf;
    uint n_work = n_batch * n_grp * nsplit;
    uint per = D / 32u;
    // The whole threadgroup takes one work item: every simdgroup attends a slice of the key
    // range with its own online-softmax state, the partials go through `tile` (m, l, acc[D]
    // per simdgroup) and simdgroup 0 merges them. A head's keys are otherwise a serial chain
    // (dot, simd_sum, exp per key) on one simdgroup, which is 0.18 ms per key of context
    // per token with 30 simdgroups idle.
    uint stride = D + 2u;
    uint nsg_a = min(NSG, TILE_FLOATS / stride);
    for (uint w = slice; w < n_work; w += nblk) {
        uint sp = w % nsplit, hg = (w / nsplit) % n_grp, b = w / (nsplit * n_grp);
        uint h0 = hg * gf, hkv = h0 / gqa;
        uint len = uint(kv_len[b]), qpos = len - 1;
        uint first = (window && len > window) ? len - window : 0u;
        uint span = len - first, perp = (span + nsplit - 1) / nsplit;
        uint lo = first + sp * perp, hi = min(lo + perp, len);
        ulong base = (ulong(b) * n_kv_head + hkv) * kv_stride;
        device const KV* kbase = K + base * D;
        device const KV* vbase = V + base * D;
        device const float* ks = KS ? KS + base : KS;
        device const float* vs = VS ? VS + base : VS;
        for (uint hh = 0; hh < gf; hh++) {
            uint h = h0 + hh;
            if (h >= n_head) break;
            device float* op = Opart + ((ulong(b) * n_head + h) * nsplit + sp) * D;
            device float* ml = mlpart + ((ulong(b) * n_head + h) * nsplit + sp) * 2;
            if (nrf) {
                if (sg == 0) {
                    for (uint j = 0; j < per; j++) op[lane + 32u * j] = NAN;
                    if (lane == 0) { ml[0] = NAN; ml[1] = NAN; }
                }
                continue;
            }
            device const ushort* q = Q + (ulong(b) * n_head + h) * D;
            if (sg < nsg_a) {
                uint n = hi > lo ? hi - lo : 0u;
                uint chunk = (n + nsg_a - 1u) / nsg_a;
                uint my_lo = lo + sg * chunk, my_hi = min(my_lo + chunk, hi);
                float m = NEG_INF, l = 0.0f, acc[16];
                for (uint j = 0; j < 16; j++) acc[j] = 0.0f;
                attend_rows(q, kbase, vbase, ks, vs, D, kv_mask, scale, my_lo, my_hi, qpos, window, false, lane, acc, m, l);
                threadgroup float* part = tile + sg * stride;
                if (lane == 0) { part[0] = m; part[1] = l; }
                for (uint j = 0; j < per; j++) part[2u + lane + 32u * j] = acc[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                float M = NEG_INF;
                for (uint i = 0; i < nsg_a; i++) M = max(M, tile[i * stride]);
                float L = 0.0f, o[16];
                for (uint j = 0; j < per; j++) o[j] = 0.0f;
                for (uint i = 0; i < nsg_a; i++) {
                    threadgroup const float* part = tile + i * stride;
                    float mi = part[0];
                    if (mi == NEG_INF) continue;
                    float e = exp(mi - M);
                    L += part[1] * e;
                    for (uint j = 0; j < per; j++) o[j] += part[2u + lane + 32u * j] * e;
                }
                for (uint j = 0; j < per; j++) op[lane + 32u * j] = o[j];
                if (lane == 0) { ml[0] = M; ml[1] = L; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
// t0=O t1=Opart t2=mlpart t3=sinks?  i0=n_batch i1=n_head i2=nsplit i3=hd
void op_flash_merge(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint lid) {
    device ushort* O = ten<ushort>(tab, in, 0);
    device const float* Opart = ten<float>(tab, in, 1);
    device const float* mlpart = ten<float>(tab, in, 2);
    device const ushort* sinks = ten<ushort>(tab, in, 3);
    uint n_batch = in.i[0], n_head = in.i[1], nsplit = in.i[2], D = in.i[3];
    uint n_bh = n_batch * n_head;
    if (n_bh == 0u) return;
    uint dsplit = (nblk + n_bh - 1) / n_bh;
    uint dchunk = (D + dsplit - 1) / dsplit;
    uint n_work = n_bh * dsplit;
    for (uint w = slice; w < n_work; w += nblk) {
        uint dp = w % dsplit, hb = w / dsplit;
        uint d0 = dp * dchunk, d1 = min(d0 + dchunk, D);
        device const float* ml = mlpart + ulong(hb) * nsplit * 2;
        float gm = NEG_INF;
        for (uint s = 0; s < nsplit; s++) gm = max(gm, ml[s * 2]);
        float sink = sinks ? bf2f(sinks[hb % n_head]) : NEG_INF;
        gm = max(gm, sink);
        float gl = sinks ? exp(sink - gm) : 0.0f;
        for (uint s = 0; s < nsplit; s++) if (ml[s * 2] != NEG_INF) gl += ml[s * 2 + 1] * exp(ml[s * 2] - gm);
        float inv = gl > 0.0f ? 1.0f / gl : 0.0f;
        device const float* obase = Opart + ulong(hb) * nsplit * D;
        for (uint d = d0 + lid; d < d1; d += NT) {
            float acc = 0.0f;
            for (uint s = 0; s < nsplit; s++) if (ml[s * 2] != NEG_INF) acc += obase[s * D + d] * exp(ml[s * 2] - gm);
            O[ulong(hb) * D + d] = f2bf(acc * inv);
        }
    }
}

// ---- dispatch ------------------------------------------------------------------------------------------
// Returns false for an opcode this interpreter does not implement (the host reports the fault).
// Op 155: Gemma-4 E-series per-layer input block, in place on x (dev_isa.h). One threadgroup per
// row: 32 simdgroups compute the P gate dots (lanes strided over H), the products land in `tile`,
// every thread then owns H/NT rows of the projection, and the two norms reduce through `red`.
// Projection values must not overlap the reduction scratch (E4B: 256 + 2560).
bool op_per_layer_input(const thread Inst& in, device const ulong* tab, uint slice, uint nblk,
                        threadgroup float* tile, threadgroup float* red, uint lid, uint sg, uint lane) {
    uint rows = in.i[0], H = in.i[1], P = in.i[2], col0 = in.i[3], stride = in.i[4];
    if (P > TILE_FLOATS - NSG || H > TILE_FLOATS - NSG - P || (P & 7u)) return false;
    device ushort* x = ten<ushort>(tab, in, 0);
    device const ushort* wg = ten<ushort>(tab, in, 1);
    device const ushort* wp = ten<ushort>(tab, in, 2);
    device const ushort* gamma = ten<ushort>(tab, in, 3);
    device const ushort* ple = ten<ushort>(tab, in, 4);
    device ushort* hn = ten<ushort>(tab, in, 5);
    device const ushort* gnext = ten<ushort>(tab, in, 6);
    float eps = as_type<float>(in.fj[0]), ls = as_type<float>(in.fj[1]);
    threadgroup float* a = tile;
    threadgroup float* y = tile + P;
    for (uint t = slice; t < rows; t += nblk) {
        device ushort* xr = x + ulong(t) * H;
        device const ushort* pr = ple + ulong(t) * stride + col0;
        // Gate: one simdgroup per output p, lanes over H (the GEMV family's dot).
        for (uint p = sg; p < P; p += NSG) {
            float s = dot_bf16(wg + ulong(p) * H, xr, H, lane); // simd-reduced inside
            if (lane == 0) {
                float g = rbf(gelu_tanhs(rbf(s)));
                a[p] = rbf(g * bf2f(pr[p]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Projection: one simdgroup per output h, lanes over P in 8-wide coalesced loads.
        float ss = 0.0f;
        for (uint h = sg; h < H; h += NSG) {
            device const ushort* w = wp + ulong(h) * P;
            float s = 0.0f;
            for (uint p = lane * 8u; p + 8u <= P; p += 256u) {
                float4 w0 = bf4(*(device const ushort4*)(w + p)), w1 = bf4(*(device const ushort4*)(w + p + 4));
                s += dot(w0, float4(a[p], a[p + 1], a[p + 2], a[p + 3]))
                   + dot(w1, float4(a[p + 4], a[p + 5], a[p + 6], a[p + 7]));
            }
            s = simd_sum(s);
            if (lane == 0) {
                s = rbf(s);
                y[h] = s;
                ss += s * s;
            }
        }
        ss = tg_sum(ss, red, lid, sg, lane);
        float inv = rsqrt(ss / float(H) + eps);
        float ss2 = 0.0f;
        for (uint h = lid; h < H; h += NT) {
            float g = gamma ? bf2f(gamma[h]) : 1.0f;
            float n = rbf(y[h] * inv * g);
            ushort v = f2bf(rbf(bf2f(xr[h]) + n) * ls);
            xr[h] = v;
            float vb = bf2f(v);
            ss2 += vb * vb;
        }
        ss2 = tg_sum(ss2, red, lid, sg, lane);
        if (hn) {
            float inv2 = rsqrt(ss2 / float(H) + eps);
            device ushort* o = hn + ulong(t) * H;
            for (uint h = lid; h < H; h += NT) {
                float g = gnext ? bf2f(gnext[h]) : 1.0f;
                o[h] = f2bf(bf2f(xr[h]) * inv2 * g);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return true;
}

// ---- GPT-OSS flat MXFP4 MoE: router/align/combine (83/84/87) and the expert GEMMs (150-153) ----
// Expert e of W(fp4 [E][N][K/2]) starts at W + e*N*(K/2), scales at S + e*N*(K/32), bias at b + e*N.
// Decode ops are GEMV-shaped (one slot = one row); the prefill twins run the v3 tiles over each
// expert's token-sorted segment (MPF_BM-row tiles, A gathered through row_token).
struct MoeRoute { uint eid; float gate; };
constant uint EXPERT_UNUSED = 0xffffffffu;
constant uint MPF_BM = 64u;
constant uint ROUTE_MAX_TOPK = 16u;
constant uint MOE_CW = 512u;   // output columns per prefill work item
// Segments of at most this many rows take the per-row dot4 sweep in the prefill twins. Measured
// (GPT-OSS-20B, 37 tokens = ~5 rows per expert): tiles 268 ms, per-row sweep 242 ms; a
// decode-once/8-rows variant and a depth-2 prefetch in the tiles were both slower. The tiled
// path costs ~1.4 us per 128x32 chunk whatever M is (fragment traffic through threadgroup
// memory), the sweep re-decodes the weights per row (~250 G weights/s): neither is near the
// weight-bandwidth floor at small M; the open item for MoE prefill.
constant uint MOE_SMALL_ROWS = 16u;

// t0=table([T*k]) t1=logit([T,n_exp] bf16) t3=bias?(f32)  i1=n_exp i2=k i3=flags i4=T i6=n_group  f0=route_scale
// flags: bit0 sigmoid (else softmax), bit1 renormalise the winners. Selection order = the golden's:
// by (score + bias) descending, lower expert id on ties; gates are the unbiased scores.
void op_moe_router_topk_pf(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint sg, uint lane) {
    uint n_exp = in.i[1], k = in.i[2], flags = in.i[3], nt = in.i[4] ? in.i[4] : 1u;
    float route_scale = as_type<float>(in.fj[0]);
    device MoeRoute* table = ten<MoeRoute>(tab, in, 0);
    device const ushort* logit = ten<ushort>(tab, in, 1);
    device const float* bias = ten<float>(tab, in, 3);
    bool sigm = (flags & 1u) != 0u, norm_topk = (flags & 2u) != 0u;
    uint ni = min((n_exp + 31u) / 32u, 16u);   // n_exp <= 512
    uint kk = min(k, ROUTE_MAX_TOPK);
    for (uint tok = slice + sg * nblk; tok < nt; tok += nblk * NSG) {
        device MoeRoute* tr = table + tok * k;
        if (in.i[6] > 1u) {   // group-limited routing is not ported: poison, as the golden does
            for (uint j = lane; j < k; j += 32u) { tr[j].eid = EXPERT_UNUSED; tr[j].gate = NAN; }
            continue;
        }
        for (uint j = ROUTE_MAX_TOPK + lane; j < k; j += 32u) { tr[j].eid = EXPERT_UNUSED; tr[j].gate = 0.0f; }
        device const ushort* lr = logit + tok * n_exp;
        float sc[16];
        float m = -INFINITY;
        for (uint i = 0; i < ni; i++) {
            uint e = lane + 32u * i;
            sc[i] = e < n_exp ? bf2f(lr[e]) : -INFINITY;
            m = max(m, sc[i]);
        }
        if (sigm) {
            for (uint i = 0; i < ni; i++) sc[i] = 1.0f / (1.0f + exp(-sc[i]));
        } else {
            m = simd_max(m);
            float s = 0.0f;
            for (uint i = 0; i < ni; i++) { sc[i] = exp(sc[i] - m); s += sc[i]; }
            s = simd_sum(s);
            for (uint i = 0; i < ni; i++) sc[i] /= s;
        }
        uint claimed = 0u;
        float gsum = 0.0f;
        uint my_e = EXPERT_UNUSED;     // winner j owned by lane j (k <= 16 < 32)
        float my_g = 0.0f;
        for (uint j = 0; j < kk; j++) {
            float best = -INFINITY;
            uint be = EXPERT_UNUSED;
            for (uint i = 0; i < ni; i++) {
                uint e = lane + 32u * i;
                if (e >= n_exp || (claimed & (1u << i))) continue;
                float v = sc[i] + (bias ? bias[e] : 0.0f);
                if (v > best) { best = v; be = e; }
            }
            float gm = simd_max(best);
            uint we = simd_min(best == gm ? be : EXPERT_UNUSED);
            uint wi = (we - lane) / 32u;               // meaningful on the owning lane only
            bool owner = (we & 31u) == lane;
            if (owner) claimed |= 1u << wi;
            float g = simd_shuffle(sc[min(wi, ni - 1u)], we & 31u);
            gsum += g;
            if (lane == j) { my_e = we; my_g = g; }
        }
        if (lane < kk) {
            float g = my_g;
            if (norm_topk && gsum != 0.0f) g /= gsum;
            tr[lane].eid = my_e;
            tr[lane].gate = g * route_scale;
        }
    }
}
// t0=meta(i32 [3*n_exp+1]) t1=table t2=row_token(u32) t3=row_partidx(u32) t4=row_gate(f32)  i0=T i1=n_exp i2=k
// One thread: histogram, MPF_BM-padded prefix, scatter in ascending slot order (T*k <= a few thousand).
void op_moe_align_pf(const thread Inst& in, device const ulong* tab, uint slice, uint lid) {
    if (slice != 0u || lid != 0u) return;
    device int* meta = ten<int>(tab, in, 0);
    device const MoeRoute* table = ten<MoeRoute>(tab, in, 1);
    device uint* row_token = ten<uint>(tab, in, 2);
    device uint* row_partidx = ten<uint>(tab, in, 3);
    device float* row_gate = ten<float>(tab, in, 4);
    uint nt = in.i[0], n_exp = in.i[1], k = in.i[2], nslot = nt * k;
    device int* rowoff = meta;
    device int* cnt = meta + n_exp;
    device int* tilep = meta + 2u * n_exp;
    for (uint e = 0; e < n_exp; e++) cnt[e] = 0;
    for (uint s = 0; s < nslot; s++) {
        uint eid = table ? table[s].eid : 0u;
        if (eid < n_exp) cnt[eid]++;
    }
    uint off = 0, tiles = 0;
    for (uint e = 0; e < n_exp; e++) {
        rowoff[e] = int(off);
        tilep[e] = int(tiles);
        uint t = (uint(cnt[e]) + MPF_BM - 1u) / MPF_BM;
        tiles += t;
        off += t * MPF_BM;
    }
    tilep[n_exp] = int(tiles);
    for (uint r = 0; r < off; r++) { row_token[r] = EXPERT_UNUSED; row_partidx[r] = EXPERT_UNUSED; row_gate[r] = 0.0f; }
    for (uint s = 0; s < nslot; s++) {
        uint eid = table ? table[s].eid : 0u;
        if (eid >= n_exp) continue;
        uint pos = uint(rowoff[eid]++);
        row_token[pos] = s / k;
        row_partidx[pos] = s;
        row_gate[pos] = table ? table[s].gate : 1.0f;
    }
    for (uint e = 0; e < n_exp; e++) rowoff[e] -= cnt[e];
}
// t0=out([T,H] bf16) t1=residual? t2=shared? t3=part([T*k,H] f32)  i0=H i1=k i2=T
void op_moe_combine_pf(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint lid) {
    uint H = in.i[0], k = in.i[1], nt = in.i[2] ? in.i[2] : 1u;
    device ushort* out = ten<ushort>(tab, in, 0);
    device const ushort* residual = ten<ushort>(tab, in, 1);
    device const ushort* shared = ten<ushort>(tab, in, 2);
    device const float* part = ten<float>(tab, in, 3);
    uint lo, hi;
    range(nt * H, slice, nblk, lo, hi);
    for (uint i = lo + lid; i < hi; i += NT) {
        uint tok = i / H, h = i - tok * H;
        float acc = residual ? bf2f(residual[i]) : 0.0f;
        if (shared) acc += bf2f(shared[i]);
        device const float* pt = part + tok * k * H;
        for (uint j = 0; j < k; j++) acc += pt[j * H + h];
        out[i] = f2bf(acc);
    }
}
// DECODE gate|up + act. t0=fu(bf16 [B*k][I]) t1=x(bf16 [B][K]) t2=table t3=W_gu(fp4 [E][2I][K/2])
// t4=S_gu t5=bias_gu?  i0=k i1=I i2=K i3=n_exp i4=layout(0 interleaved, 1 blocked) i5=act i6=B  f0/f1
// Each threadgroup owns a column range of every slot; simdgroups take output pairs (interleaved
// layout: rows 2n..2n+3 are one dot4).
void op_moe_glu_mx(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint sg, uint lane) {
    uint k = in.i[0], I = in.i[1], K = in.i[2], E = in.i[3], layout = in.i[4], act = in.i[5];
    uint B = in.i[6] ? in.i[6] : 1u;
    float f0 = as_type<float>(in.fj[0]), f1 = as_type<float>(in.fj[1]);
    device ushort* fu = ten<ushort>(tab, in, 0);
    device const ushort* x = ten<ushort>(tab, in, 1);
    device const MoeRoute* table = ten<MoeRoute>(tab, in, 2);
    device const uchar* W = ten<uchar>(tab, in, 3);
    device const uchar* S = ten<uchar>(tab, in, 4);
    device const ushort* bias = ten<ushort>(tab, in, 5);
    uint ldw = K / 2u, lds = (K + 31u) / 32u, N2 = 2u * I;
    uint n0, n1;
    range(I, slice, nblk, n0, n1);
    for (uint s = 0; s < B * k; s++) {
        uint eid = table[s].eid;
        if (eid >= E) continue;
        device const ushort* xr = x + (s / k) * K;
        device const uchar* We = W + ulong(eid) * N2 * ldw;
        device const uchar* Se = S + ulong(eid) * N2 * lds;
        device const ushort* be = bias ? bias + ulong(eid) * N2 : (device const ushort*)0;
        device ushort* fr = fu + s * I;
        if (layout == 0u) {
            uint n = n0 + sg * 2u;
            for (; n + 2u <= n1; n += NSG * 2u) {
                float4 a = dot4_mx4(We + (2u * n) * ldw, Se + (2u * n) * lds, ldw, lds, xr, K, lane);
                if (be) a += float4(bf2f(be[2u * n]), bf2f(be[2u * n + 1u]), bf2f(be[2u * n + 2u]), bf2f(be[2u * n + 3u]));
                if (lane == 0) { fr[n] = f2bf(glu_pair(a.x, a.y, act, f0, f1)); fr[n + 1u] = f2bf(glu_pair(a.z, a.w, act, f0, f1)); }
            }
            for (; n < n1; n++) {
                float g = dot_mx4(We + (2u * n) * ldw, Se + (2u * n) * lds, xr, K, lane);
                float u = dot_mx4(We + (2u * n + 1u) * ldw, Se + (2u * n + 1u) * lds, xr, K, lane);
                if (be) { g += bf2f(be[2u * n]); u += bf2f(be[2u * n + 1u]); }
                if (lane == 0) fr[n] = f2bf(glu_pair(g, u, act, f0, f1));
            }
        } else {
            for (uint n = n0 + sg; n < n1; n += NSG) {
                float g = dot_mx4(We + n * ldw, Se + n * lds, xr, K, lane);
                float u = dot_mx4(We + (I + n) * ldw, Se + (I + n) * lds, xr, K, lane);
                if (be) { g += bf2f(be[n]); u += bf2f(be[I + n]); }
                if (lane == 0) fr[n] = f2bf(glu_pair(g, u, act, f0, f1));
            }
        }
    }
}
// DECODE down + gate. t0=part(f32 [B*k][H]) t1=fu(bf16 [B*k][I]) t2=table t3=W_d(fp4 [E][H][I/2])
// t4=S_d t5=bias_d?  i0=k i1=H i2=I i3=n_exp i6=B.  part[s][h] = gate * (W_e[h] . fu[s] + b_e[h]).
void op_moe_down_mx(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint sg, uint lane) {
    uint k = in.i[0], H = in.i[1], I = in.i[2], E = in.i[3];
    uint B = in.i[6] ? in.i[6] : 1u;
    device float* part = ten<float>(tab, in, 0);
    device const ushort* fu = ten<ushort>(tab, in, 1);
    device const MoeRoute* table = ten<MoeRoute>(tab, in, 2);
    device const uchar* W = ten<uchar>(tab, in, 3);
    device const uchar* S = ten<uchar>(tab, in, 4);
    device const ushort* bias = ten<ushort>(tab, in, 5);
    uint ldw = I / 2u, lds = (I + 31u) / 32u;
    uint h0, h1;
    range(H, slice, nblk, h0, h1);
    for (uint s = 0; s < B * k; s++) {
        device float* pr = part + s * H;
        uint eid = table[s].eid;
        if (eid >= E) {
            for (uint h = h0 + sg * 32u + lane; h < h1; h += NSG * 32u) pr[h] = 0.0f;
            continue;
        }
        float gate = table[s].gate;
        device const ushort* fr = fu + s * I;
        device const uchar* We = W + ulong(eid) * H * ldw;
        device const uchar* Se = S + ulong(eid) * H * lds;
        device const ushort* be = bias ? bias + ulong(eid) * H : (device const ushort*)0;
        uint h = h0 + sg * 4u;
        for (; h + 4u <= h1; h += NSG * 4u) {
            float4 a = dot4_mx4(We + h * ldw, Se + h * lds, ldw, lds, fr, I, lane);
            if (be) a += float4(bf2f(be[h]), bf2f(be[h + 1u]), bf2f(be[h + 2u]), bf2f(be[h + 3u]));
            if (lane == 0) { pr[h] = gate * a.x; pr[h + 1u] = gate * a.y; pr[h + 2u] = gate * a.z; pr[h + 3u] = gate * a.w; }
        }
        for (; h < h1; h++) {
            float acc = dot_mx4(We + h * ldw, Se + h * lds, fr, I, lane);
            if (be) acc += bf2f(be[h]);
            if (lane == 0) pr[h] = gate * acc;
        }
    }
}
// Prefill work item lin -> (expert tile, column chunk) through meta's tile prefix.
inline bool moe_tile(device const int* meta, uint E, uint lin, uint nch, thread uint& e, thread uint& r0,
                     thread uint& m0, thread uint& m1, thread uint& c0) {
    uint tl = lin / nch;
    c0 = (lin % nch) * MOE_CW;
    device const int* tilep = meta + 2u * E;
    e = 0;
    for (uint x = 1; x < E; x++) if (uint(tilep[x]) <= tl) e = x;
    r0 = uint(meta[e]);
    uint cnt = uint(meta[E + e]);
    m0 = (tl - uint(tilep[e])) * MPF_BM;
    m1 = min(m0 + MPF_BM, cnt);
    return m0 < m1;
}
// PREFILL gate|up: t0=fu_g(bf16 [rows][I]) t1=xn2(bf16 [T][K]) t2=W_gu t3=S_gu t4=meta t5=row_token
// t6=bias_gu?  i0=I i1=K i2=n_exp i3=layout i5=act  f0/f1. Rows of expert e = [rowoff[e], +cnt[e]).
void op_moe_glu_mx_pf(const thread Inst& in, device const ulong* tab, uint slice, uint nblk,
                      threadgroup float* tile, uint lid, uint sg, uint lane) {
    uint I = in.i[0], K = in.i[1], E = in.i[2], layout = in.i[3], act = in.i[5];
    float f0 = as_type<float>(in.fj[0]), f1 = as_type<float>(in.fj[1]);
    device ushort* fu = ten<ushort>(tab, in, 0);
    device const ushort* x = ten<ushort>(tab, in, 1);
    device const uchar* W = ten<uchar>(tab, in, 2);
    device const uchar* S = ten<uchar>(tab, in, 3);
    device const int* meta = ten<int>(tab, in, 4);
    device const uint* row_token = ten<uint>(tab, in, 5);
    device const ushort* bias = ten<ushort>(tab, in, 6);
    uint ldw = K / 2u, lds = (K + 31u) / 32u, N2 = 2u * I;
    uint ntile = uint(meta[3u * E]), nch = (I + MOE_CW - 1u) / MOE_CW;
    for (uint lin = slice; lin < ntile * nch; lin += nblk) {
        uint e, r0, m0, m1, c0;
        if (!moe_tile(meta, E, lin, nch, e, r0, m0, m1, c0)) continue;
        if (m1 <= MOE_SMALL_ROWS) {
            device const uchar* We = W + ulong(e) * N2 * ldw;
            device const uchar* Se = S + ulong(e) * N2 * lds;
            device const ushort* be = bias ? bias + ulong(e) * N2 : (device const ushort*)0;
            uint c1 = min(c0 + MOE_CW, I);
            for (uint n = c0 + sg * 2u; n < c1; n += NSG * 2u) {
                bool pair = layout == 0u && n + 2u <= c1;
                for (uint r = 0; r < m1; r++) {
                    uint tok = row_token[r0 + r];
                    if (tok == EXPERT_UNUSED) continue;
                    device const ushort* xr = x + tok * K;
                    device ushort* fr = fu + (r0 + r) * I;
                    if (pair) {
                        float4 a = dot4_mx4(We + (2u * n) * ldw, Se + (2u * n) * lds, ldw, lds, xr, K, lane);
                        if (be) a += float4(bf2f(be[2u * n]), bf2f(be[2u * n + 1u]), bf2f(be[2u * n + 2u]), bf2f(be[2u * n + 3u]));
                        if (lane == 0) { fr[n] = f2bf(glu_pair(a.x, a.y, act, f0, f1)); fr[n + 1u] = f2bf(glu_pair(a.z, a.w, act, f0, f1)); }
                    } else {
                        for (uint q = n; q < min(n + 2u, c1); q++) {
                            uint rg = layout ? q : 2u * q, ru = layout ? I + q : 2u * q + 1u;
                            float g = dot_mx4(We + rg * ldw, Se + rg * lds, xr, K, lane);
                            float u = dot_mx4(We + ru * ldw, Se + ru * lds, xr, K, lane);
                            if (be) { g += bf2f(be[rg]); u += bf2f(be[ru]); }
                            if (lane == 0) fr[q] = f2bf(glu_pair(g, u, act, f0, f1));
                        }
                    }
                }
            }
            continue;
        }
        GemmArgs gg, gu;
        gemm_args_reset(gg);
        gg.A = x; gg.K = K; gg.M = m1; gg.N = I; gg.arow = row_token + r0;
        gg.B4 = W + ulong(e) * N2 * ldw; gg.S8 = S + ulong(e) * N2 * lds;
        gg.bias = bias ? bias + ulong(e) * N2 : (device const ushort*)0;
        gu = gg;
        if (layout == 0u) { gg.rs = 2u; gu.rs = 2u; gu.ro = 1u; }
        else { gu.B4 = gg.B4 + ulong(I) * ldw; gu.S8 = gg.S8 + ulong(I) * lds; if (gu.bias) gu.bias = gg.bias + I; }
        gemm_tile2_glu(gg, gu, fu + r0 * I, act, f0, f1, false, m0, m1, c0, min(c0 + MOE_CW, I), tile, lid, sg, lane);
    }
}
// PREFILL down: t0=part(f32 [T*k][H]) t1=fu_g t2=W_d t3=S_d t4=meta t5=bias_d? t6=row_partidx t7=row_gate
// i0=H i1=I i2=n_exp.  part[row_partidx[r]][h] = row_gate[r] * (W_e[h] . fu_g[r] + b_e[h]).
void op_moe_down_mx_pf(const thread Inst& in, device const ulong* tab, uint slice, uint nblk,
                       threadgroup float* tile, uint lid, uint sg, uint lane) {
    uint H = in.i[0], I = in.i[1], E = in.i[2];
    device float* part = ten<float>(tab, in, 0);
    device const ushort* fu = ten<ushort>(tab, in, 1);
    device const uchar* W = ten<uchar>(tab, in, 2);
    device const uchar* S = ten<uchar>(tab, in, 3);
    device const int* meta = ten<int>(tab, in, 4);
    device const ushort* bias = ten<ushort>(tab, in, 5);
    device const uint* row_partidx = ten<uint>(tab, in, 6);
    device const float* row_gate = ten<float>(tab, in, 7);
    uint ldw = I / 2u, lds = (I + 31u) / 32u;
    uint ntile = uint(meta[3u * E]), nch = (H + MOE_CW - 1u) / MOE_CW;
    for (uint lin = slice; lin < ntile * nch; lin += nblk) {
        uint e, r0, m0, m1, c0;
        if (!moe_tile(meta, E, lin, nch, e, r0, m0, m1, c0)) continue;
        if (m1 <= MOE_SMALL_ROWS) {
            device const uchar* We = W + ulong(e) * H * ldw;
            device const uchar* Se = S + ulong(e) * H * lds;
            device const ushort* be = bias ? bias + ulong(e) * H : (device const ushort*)0;
            uint c1 = min(c0 + MOE_CW, H);
            for (uint h = c0 + sg * 4u; h < c1; h += NSG * 4u) {
                bool quad = h + 4u <= c1;
                for (uint r = 0; r < m1; r++) {
                    uint pidx = row_partidx[r0 + r];
                    if (pidx == EXPERT_UNUSED) continue;
                    float gate = row_gate[r0 + r];
                    device const ushort* fr = fu + (r0 + r) * I;
                    device float* pr = part + pidx * H;
                    if (quad) {
                        float4 a = dot4_mx4(We + h * ldw, Se + h * lds, ldw, lds, fr, I, lane);
                        if (be) a += float4(bf2f(be[h]), bf2f(be[h + 1u]), bf2f(be[h + 2u]), bf2f(be[h + 3u]));
                        if (lane == 0) { pr[h] = gate * a.x; pr[h + 1u] = gate * a.y; pr[h + 2u] = gate * a.z; pr[h + 3u] = gate * a.w; }
                    } else {
                        for (uint q = h; q < c1; q++) {
                            float acc = dot_mx4(We + q * ldw, Se + q * lds, fr, I, lane);
                            if (be) acc += bf2f(be[q]);
                            if (lane == 0) pr[q] = gate * acc;
                        }
                    }
                }
            }
            continue;
        }
        GemmArgs g;
        gemm_args_reset(g);
        g.A = fu + r0 * I; g.K = I; g.M = m1; g.N = H;
        g.B4 = W + ulong(e) * H * ldw; g.S8 = S + ulong(e) * H * lds;
        g.bias = bias ? bias + ulong(e) * H : (device const ushort*)0;
        g.Cf = part; g.crow = row_partidx + r0; g.rowscale = row_gate + r0;
        gemm_tile2(g, (device ushort*)0, H, m0, m1, c0, min(c0 + MOE_CW, H), SN2, tile, lid, sg, lane);
    }
}

// ---- backend-neutral FP32 primitives ------------------------------------------------------------
void op_q8_gemm_f32(const thread Inst& in, device const ulong* tab, uint slice,
                    threadgroup float* tile, uint lid, uint sg, uint lane) {
    uint M = in.i[0], N = in.i[1], K = in.i[2], activation = in.i[3];
    device float* C = ten<float>(tab, in, 0);
    device const float* A = ten<float>(tab, in, 1) + ulong(in.i[4]) * K;
    device const uchar* W = ten<uchar>(tab, in, 2);
    device const float* bias = ten<float>(tab, in, 3);
    uint nt = (N + 63u) / 64u, row0 = (slice / nt) * 128u, col = (slice % nt) * 64u;
    uint cluster = sg / 8u, local_sg = sg % 8u, row = row0 + cluster * 32u;
    uint sr = (local_sg / 4u) * 16u, sc = (local_sg % 4u) * 16u;
    uint row_bytes = (K / 32u) * 34u;
    threadgroup float* at = tile + cluster * 1024u;
    threadgroup float* bt = tile + 4096u;
    simdgroup_float8x8 a0(0.0f), a1(0.0f), a2(0.0f), a3(0.0f);
    for (uint base = 0; base < K; base += 32u) {
        for (uint e = lid; e < 4096u; e += NT) {
            uint r = e / 32u, c = e % 32u;
            tile[e] = row0 + r < M ? A[ulong(row0 + r) * K + base + c] : 0.0f;
        }
        for (uint e = lid; e < 2048u; e += NT) {
            uint r = e / 32u, c = e % 32u;
            if (col + r < N) {
                device const uchar* block = W + ulong(col + r) * row_bytes + (base / 32u) * 34u;
                ushort hs = ushort(block[0]) | ushort(ushort(block[1]) << 8u);
                bt[e] = float(as_type<char>(block[2u + c])) * float(as_type<half>(hs));
            } else bt[e] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0; q < 32u; q += 8u) {
            simdgroup_float8x8 ar0, ar1, br0, br1;
            simdgroup_load(ar0, at + sr * 32u + q, 32);
            simdgroup_load(ar1, at + (sr + 8u) * 32u + q, 32);
            simdgroup_load(br0, bt + sc * 32u + q, 32, ulong2(0), true);
            simdgroup_load(br1, bt + (sc + 8u) * 32u + q, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(a0, ar0, br0, a0);
            simdgroup_multiply_accumulate(a1, ar0, br1, a1);
            simdgroup_multiply_accumulate(a2, ar1, br0, a2);
            simdgroup_multiply_accumulate(a3, ar1, br1, a3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint quad = lane / 4u, fr = (quad & 4u) + ((lane / 2u) % 4u);
    uint fc = (quad & 2u) * 2u + (lane % 2u) * 2u;
    for (uint e = 0; e < 2u; e++) {
        uint r = row + sr + fr, c = col + sc + fc + e;
        float values[4] = {a0.thread_elements()[e], a1.thread_elements()[e],
                           a2.thread_elements()[e], a3.thread_elements()[e]};
        uint rr[4] = {r, r, r + 8u, r + 8u};
        uint cc[4] = {c, c + 8u, c, c + 8u};
        for (uint z = 0; z < 4u; z++) if (rr[z] < M && cc[z] < N) {
            float v = values[z] + (bias ? bias[cc[z]] : 0.0f);
            C[ulong(rr[z]) * N + cc[z]] = activation == 1u ? siluf(v) : v;
        }
    }
}

void op_dense_gemm_f32(const thread Inst& in, device const ulong* tab, uint slice,
                       threadgroup float* tile, uint lid, uint sg, uint lane) {
    uint M = in.i[0], N = in.i[1], K = in.i[2], activation = in.i[3];
    uint weight_stride = in.i[5] ? in.i[5] : K, onehot = in.i[6];
    uint flags = in.i[7];
    device float* C = ten<float>(tab, in, 0);
    device const float* A = ten<float>(tab, in, 1) + ulong(in.i[4]) * K;
    device const float* W = ten<float>(tab, in, 2);
    device const ushort* Wbf = ten<ushort>(tab, in, 2);
    device const float* bias = ten<float>(tab, in, 3);
    uint nt = (N + 63u) / 64u, row0 = (slice / nt) * 128u, col = (slice % nt) * 64u;
    uint cluster = sg / 8u, local_sg = sg % 8u, row = row0 + cluster * 32u;
    uint sr = (local_sg / 4u) * 16u, sc = (local_sg % 4u) * 16u;
    threadgroup float* at = tile + cluster * 1024u;
    threadgroup float* bt = tile + 4096u;
    simdgroup_float8x8 a0(0.0f), a1(0.0f), a2(0.0f), a3(0.0f);
    for (uint base = 0; base < K; base += 32u) {
        for (uint e = lid; e < 4096u; e += NT) {
            uint r = e / 32u, c = e % 32u;
            tile[e] = row0 + r < M && base + c < K ? A[ulong(row0 + r) * K + base + c] : 0.0f;
        }
        for (uint e = lid; e < 2048u; e += NT) {
            uint r = e / 32u, c = e % 32u;
            ulong wi = ulong(col + r) * weight_stride + base + c;
            bt[e] = col + r < N && base + c < K
                ? (flags & 4u ? as_type<float>(uint(Wbf[wi]) << 16u) : W[wi]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0; q < 32u; q += 8u) {
            simdgroup_float8x8 ar0, ar1, br0, br1;
            simdgroup_load(ar0, at + sr * 32u + q, 32);
            simdgroup_load(ar1, at + (sr + 8u) * 32u + q, 32);
            simdgroup_load(br0, bt + sc * 32u + q, 32, ulong2(0), true);
            simdgroup_load(br1, bt + (sc + 8u) * 32u + q, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(a0, ar0, br0, a0);
            simdgroup_multiply_accumulate(a1, ar0, br1, a1);
            simdgroup_multiply_accumulate(a2, ar1, br0, a2);
            simdgroup_multiply_accumulate(a3, ar1, br1, a3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint quad = lane / 4u, fr = (quad & 4u) + ((lane / 2u) % 4u);
    uint fc = (quad & 2u) * 2u + (lane % 2u) * 2u;
    for (uint e = 0; e < 2u; e++) {
        uint r = row + sr + fr, c = col + sc + fc + e;
        float values[4] = {a0.thread_elements()[e], a1.thread_elements()[e],
                           a2.thread_elements()[e], a3.thread_elements()[e]};
        uint rr[4] = {r, r, r + 8u, r + 8u};
        uint cc[4] = {c, c + 8u, c, c + 8u};
        for (uint z = 0; z < 4u; z++) if (rr[z] < M && cc[z] < N) {
            float value = values[z] + (bias ? bias[cc[z]] : 0.0f);
            if (in.i[5] && onehot < weight_stride) {
                ulong wi = ulong(cc[z]) * weight_stride + onehot;
                value += flags & 4u ? as_type<float>(uint(Wbf[wi]) << 16u) : W[wi];
            }
            if (flags & 2u) value = gelu_erf_bf16(value);
            else if (flags & 1u) value = round_bf16_f32(value);
            C[ulong(rr[z]) * N + cc[z]] = activation == 1u ? max(value, 0.0f) : value;
        }
    }
}

void op_conv2d_f32(const thread Inst& in, device const ulong* tab, uint slice,
                       threadgroup float* tile, uint lid, uint sg, uint lane) {
    uint frames = in.i[0], width = in.i[1], input_channels = in.i[2];
    uint output_channels = in.i[3], filter_size = in.i[4], step = in.i[5];
    uint pad_before = in.i[6], pad_after = in.i[7], flags = in.fj[1];
    uint batches = in.fj[2] ? in.fj[2] : 1u;
    if (!filter_size || !step || frames + pad_before + pad_after < filter_size ||
        width + pad_before + pad_after < filter_size) return;
    uint output_frames = (frames + pad_before + pad_after - filter_size) / step + 1u;
    uint output_width = (width + pad_before + pad_after - filter_size) / step + 1u;
    bool depthwise = (flags & 1u) != 0u, relu = (flags & 2u) != 0u;
    uint output_layout = (flags >> 2u) & 3u, input_layout = (flags >> 4u) & 3u;
    bool weight_f32 = (flags & 64u) != 0u, gelu = (flags & 128u) != 0u;
    if (output_layout > 2u || input_layout > 2u) return;
    device float* out = ten<float>(tab, in, 0);
    device const float* x = ten<float>(tab, in, 1);
    device const uchar* weight_raw = ten<uchar>(tab, in, 2);
    device const half* weight_h = (device const half*)weight_raw;
    device const float* weight_f = (device const float*)weight_raw;
    device const float* bias = ten<float>(tab, in, 3);

    if (filter_size == 3u && (input_layout == 2u || (input_layout == 0u && input_channels == 1u)) &&
        output_layout == 2u && !depthwise) {
        uint spatial = output_frames * output_width, M = batches * spatial;
        uint N = output_channels, K = input_channels * 9u;
        uint ntiles = (N + 63u) / 64u, row0 = (slice / ntiles) * 128u;
        uint col = (slice % ntiles) * 64u;
        uint cluster = sg / 8u, local_sg = sg % 8u, row = row0 + cluster * 32u;
        uint simd_row = (local_sg / 4u) * 16u, simd_col = (local_sg % 4u) * 16u;
        threadgroup float* at = tile + cluster * 1024u;
        threadgroup float* bt = tile + 4096u;
        simdgroup_float8x8 acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);
        for (uint base = 0u; base < K; base += 32u) {
            for (uint e = lid; e < 4096u; e += NT) {
                uint r = e / 32u, c = e % 32u, flat_row = row0 + r, q = base + c;
                uint batch = flat_row / spatial, pos = flat_row % spatial;
                int input_frame = int((pos / output_width) * step + (q % 9u) / 3u) -
                                  int(pad_before);
                int input_x = int((pos % output_width) * step + q % 3u) - int(pad_before);
                tile[e] = flat_row < M && q < K && input_frame >= 0 &&
                                  input_frame < int(frames) && input_x >= 0 && input_x < int(width)
                    ? x[((ulong(batch) * input_channels + q / 9u) * frames +
                         uint(input_frame)) * width + uint(input_x)]
                    : 0.0f;
            }
            for (uint e = lid; e < 2048u; e += NT) {
                uint r = e / 32u, c = e % 32u;
                ulong wi = ulong(col + r) * K + base + c;
                bt[e] = col + r < N && base + c < K
                    ? (weight_f32 ? weight_f[wi] : float(weight_h[wi])) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint q = 0u; q < 32u; q += 8u) {
                simdgroup_float8x8 a0, a1, b0, b1;
                simdgroup_load(a0, at + simd_row * 32u + q, 32);
                simdgroup_load(a1, at + (simd_row + 8u) * 32u + q, 32);
                simdgroup_load(b0, bt + simd_col * 32u + q, 32, ulong2(0), true);
                simdgroup_load(b1, bt + (simd_col + 8u) * 32u + q, 32, ulong2(0), true);
                simdgroup_multiply_accumulate(acc0, a0, b0, acc0);
                simdgroup_multiply_accumulate(acc1, a0, b1, acc1);
                simdgroup_multiply_accumulate(acc2, a1, b0, acc2);
                simdgroup_multiply_accumulate(acc3, a1, b1, acc3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint quad = lane / 4u, fr = (quad & 4u) + ((lane / 2u) % 4u);
        uint fc = (quad & 2u) * 2u + (lane % 2u) * 2u;
        for (uint e = 0u; e < 2u; e++) {
            uint r = row + simd_row + fr, c = col + simd_col + fc + e;
            uint rr[4] = {r, r, r + 8u, r + 8u};
            uint cc[4] = {c, c + 8u, c, c + 8u};
            float values[4] = {acc0.thread_elements()[e], acc1.thread_elements()[e],
                               acc2.thread_elements()[e], acc3.thread_elements()[e]};
            for (uint z = 0u; z < 4u; z++) if (rr[z] < M && cc[z] < N) {
                float value = values[z] + bias[cc[z]];
                if (gelu) value = gelu_erf_bf16(value);
                if (relu) value = max(value, 0.0f);
                uint batch = rr[z] / spatial, pos = rr[z] % spatial;
                out[(ulong(batch) * output_channels + cc[z]) * spatial + pos] = value;
            }
        }
        return;
    }

    if (filter_size == 1u && step == 1u && pad_before == 0u && pad_after == 0u &&
        !depthwise && input_layout == 0u) {
        uint M = batches * output_frames * output_width, N = output_channels, K = input_channels;
        uint ntiles = (N + 63u) / 64u, row0 = (slice / ntiles) * 128u;
        uint col = (slice % ntiles) * 64u;
        uint cluster = sg / 8u, local_sg = sg % 8u, row = row0 + cluster * 32u;
        uint simd_row = (local_sg / 4u) * 16u, simd_col = (local_sg % 4u) * 16u;
        threadgroup float* at = tile + cluster * 1024u;
        threadgroup float* bt = tile + 4096u;
        simdgroup_float8x8 acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);
        for (uint base = 0u; base < K; base += 32u) {
            for (uint e = lid; e < 4096u; e += NT) {
                uint r = e / 32u, c = e % 32u;
                tile[e] = row0 + r < M && base + c < K
                    ? x[ulong(row0 + r) * K + base + c] : 0.0f;
            }
            for (uint e = lid; e < 2048u; e += NT) {
                uint r = e / 32u, c = e % 32u;
                ulong wi = ulong(col + r) * K + base + c;
                bt[e] = col + r < N && base + c < K
                    ? (weight_f32 ? weight_f[wi] : float(weight_h[wi])) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint q = 0u; q < 32u; q += 8u) {
                simdgroup_float8x8 a0, a1, b0, b1;
                simdgroup_load(a0, at + simd_row * 32u + q, 32);
                simdgroup_load(a1, at + (simd_row + 8u) * 32u + q, 32);
                simdgroup_load(b0, bt + simd_col * 32u + q, 32, ulong2(0), true);
                simdgroup_load(b1, bt + (simd_col + 8u) * 32u + q, 32, ulong2(0), true);
                simdgroup_multiply_accumulate(acc0, a0, b0, acc0);
                simdgroup_multiply_accumulate(acc1, a0, b1, acc1);
                simdgroup_multiply_accumulate(acc2, a1, b0, acc2);
                simdgroup_multiply_accumulate(acc3, a1, b1, acc3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint quad = lane / 4u, fr = (quad & 4u) + ((lane / 2u) % 4u);
        uint fc = (quad & 2u) * 2u + (lane % 2u) * 2u;
        for (uint e = 0u; e < 2u; e++) {
            uint rr[4] = {row + simd_row + fr, row + simd_row + fr,
                          row + simd_row + fr + 8u, row + simd_row + fr + 8u};
            uint cc[4] = {col + simd_col + fc + e, col + simd_col + fc + e + 8u,
                          col + simd_col + fc + e, col + simd_col + fc + e + 8u};
            float values[4] = {acc0.thread_elements()[e], acc1.thread_elements()[e],
                               acc2.thread_elements()[e], acc3.thread_elements()[e]};
            for (uint z = 0u; z < 4u; z++) if (rr[z] < M && cc[z] < N) {
                float value = values[z] + bias[cc[z]];
                if (gelu) value = gelu_erf_bf16(value);
                if (relu) value = max(value, 0.0f);
                uint batch = rr[z] / (output_frames * output_width);
                uint spatial = rr[z] % (output_frames * output_width);
                uint output_frame = spatial / output_width, output_x = spatial % output_width;
                ulong oi;
                if (output_layout == 0u)
                    oi = (ulong(rr[z]) * output_channels) + cc[z];
                else if (output_layout == 1u)
                    oi = ((ulong(batch) * output_frames + output_frame) * output_channels + cc[z]) *
                             output_width + output_x;
                else
                    oi = ((ulong(batch) * output_channels + cc[z]) * output_frames + output_frame) *
                             output_width + output_x;
                out[oi] = value;
            }
        }
        return;
    }

    ulong count = ulong(batches) * output_frames * output_width * output_channels;
    ulong index = ulong(slice) * NT + lid;
    if (index >= count) return;
    uint output_channel = uint(index % output_channels);
    uint output_x = uint((index / output_channels) % output_width);
    uint output_frame = uint((index / (ulong(output_channels) * output_width)) % output_frames);
    uint batch = uint(index / (ulong(output_channels) * output_width * output_frames));
    uint channel_start = depthwise ? output_channel : 0u;
    uint channel_end = depthwise ? output_channel + 1u : input_channels;
    uint stored_channels = depthwise ? 1u : input_channels;
    float sum = bias[output_channel];
    for (uint ky = 0u; ky < filter_size; ky++) {
        int input_frame = int(output_frame * step + ky) - int(pad_before);
        if (input_frame < 0 || input_frame >= int(frames)) continue;
        for (uint kx = 0u; kx < filter_size; kx++) {
            int input_x = int(output_x * step + kx) - int(pad_before);
            if (input_x < 0 || input_x >= int(width)) continue;
            for (uint input_channel = channel_start; input_channel < channel_end; input_channel++) {
                uint weight_channel = depthwise ? 0u : input_channel;
                ulong wi = (((ulong(output_channel) * stored_channels + weight_channel) * filter_size + ky)
                            * filter_size) + kx;
                ulong xi;
                if (input_layout == 0u)
                    xi = ((ulong(batch) * frames + uint(input_frame)) * width + uint(input_x)) *
                             input_channels + input_channel;
                else if (input_layout == 1u)
                    xi = ((ulong(batch) * frames + uint(input_frame)) * input_channels +
                          input_channel) * width + uint(input_x);
                else
                    xi = ((ulong(batch) * input_channels + input_channel) * frames +
                          uint(input_frame)) * width + uint(input_x);
                sum = fma(x[xi], weight_f32 ? weight_f[wi] : float(weight_h[wi]), sum);
            }
        }
    }
    if (gelu) sum = gelu_erf_bf16(sum);
    if (relu) sum = max(sum, 0.0f);
    ulong oi;
    if (output_layout == 0u)
        oi = ((ulong(batch) * output_frames + output_frame) * output_width + output_x) *
                 output_channels + output_channel;
    else if (output_layout == 1u)
        oi = ((ulong(batch) * output_frames + output_frame) * output_channels + output_channel) *
                 output_width + output_x;
    else
        oi = ((ulong(batch) * output_channels + output_channel) * output_frames + output_frame) *
                 output_width + output_x;
    out[oi] = sum;
}

void op_pack_ncfw_rows_f32(const thread Inst& in, device const ulong* tab, uint slice,
                           uint lid) {
    device float* out = ten<float>(tab, in, 0);
    device const float* x = ten<float>(tab, in, 1);
    uint rows = in.i[0], channels = in.i[1], frames = in.i[2], width = in.i[3], batches = in.i[4];
    ulong count = ulong(rows) * channels * frames;
    ulong index = ulong(slice) * NT + lid;
    if (!rows || !channels || !frames || !width || !batches ||
        ulong(rows) > ulong(batches) * width || channels > 0xffffffffu / frames || index >= count)
        return;
    uint row_width = channels * frames;
    uint row = uint(index / row_width), column = uint(index % row_width);
    uint batch = row / width, position = row % width;
    uint channel = column / frames, frame = column % frames;
    out[index] = x[((ulong(batch) * channels + channel) * frames + frame) * width + position];
}

void op_layernorm_f32(const thread Inst& in, device const ulong* tab, uint slice, uint nblk,
                      threadgroup float* red, uint lid, uint sg, uint lane) {
    device float* out = ten<float>(tab, in, 0);
    device const float* x = ten<float>(tab, in, 1);
    device const float* gamma = ten<float>(tab, in, 2);
    device const float* beta = ten<float>(tab, in, 3);
    uint rows = in.i[0], feat = in.i[1];
    if (in.i[2] & 2u) {
        for (uint row = slice * NSG + sg; row < rows; row += nblk * NSG) {
            ulong base = ulong(row) * feat;
            float mean = 0.0f;
            if (lane == 0u) {
                for (uint i = 0u; i < feat; i++) mean += x[base + i];
                mean /= float(feat);
            }
            mean = simd_broadcast(mean, 0u);
            float inv = 0.0f;
            if (lane == 0u) {
                float variance = 0.0f;
                for (uint i = 0u; i < feat; i++) {
                    float value = x[base + i] - mean;
                    variance += value * value;
                }
                inv = rsqrt(variance / float(feat) + as_type<float>(in.fj[0]));
            }
            inv = simd_broadcast(inv, 0u);
            for (uint i = lane; i < feat; i += 32u) {
                float value = (x[base + i] - mean) * inv * (gamma ? gamma[i] : 1.0f) +
                              (beta ? beta[i] : 0.0f);
                out[base + i] = in.i[2] & 1u ? round_bf16_f32(value) : value;
            }
        }
        return;
    }
    for (uint row = slice; row < rows; row += nblk) {
        ulong base = ulong(row) * feat;
        float sum = 0.0f;
        for (uint i = lid; i < feat; i += NT) sum += x[base + i];
        float mean = tg_sum(sum, red, lid, sg, lane) / float(feat);
        float ss = 0.0f;
        for (uint i = lid; i < feat; i += NT) { float v = x[base + i] - mean; ss = fma(v, v, ss); }
        float inv = rsqrt(tg_sum(ss, red, lid, sg, lane) / float(feat) + as_type<float>(in.fj[0]));
        for (uint i = lid; i < feat; i += NT) {
            float value = (x[base + i] - mean) * inv * (gamma ? gamma[i] : 1.0f) +
                          (beta ? beta[i] : 0.0f);
            out[base + i] = in.i[2] & 1u ? round_bf16_f32(value) : value;
        }
    }
}

void op_scaled_add_f32(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint lid) {
    device float* out = ten<float>(tab, in, 0);
    device const float* a = ten<float>(tab, in, 1);
    device const float* b = ten<float>(tab, in, 2);
    uint lo, hi; range(in.i[0], slice, nblk, lo, hi);
    float scale = as_type<float>(in.fj[0]);
    for (uint i = lo + lid; i < hi; i += NT) {
        float value = fma(scale, b[i], a[i]);
        out[i] = in.i[1] & 1u ? round_bf16_f32(value) : value;
    }
}

void op_glu_f32(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint lid) {
    device float* out = ten<float>(tab, in, 0);
    device const float* x = ten<float>(tab, in, 1);
    uint rows = in.i[0], width = in.i[1], lo, hi; range(rows * width, slice, nblk, lo, hi);
    for (uint i = lo + lid; i < hi; i += NT) {
        uint row = i / width, col = i % width;
        out[i] = x[ulong(row) * 2u * width + col] * sigmoidf(x[ulong(row) * 2u * width + width + col]);
    }
}

void op_silu_f32(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint lid) {
    device float* out = ten<float>(tab, in, 0);
    device const float* x = ten<float>(tab, in, 1);
    uint lo, hi; range(in.i[0], slice, nblk, lo, hi);
    for (uint i = lo + lid; i < hi; i += NT) out[i] = siluf(x[i]);
}

void op_embed_f16_f32(const thread Inst& in, device const ulong* tab,
                      uint slice, uint nblk, uint lid) {
    device float* out = ten<float>(tab, in, 0);
    device const half* table = ten<half>(tab, in, 1);
    device const uint* token = ten<uint>(tab, in, 2);
    uint vocab = in.i[0], width = in.i[1], lo, hi; range(width, slice, nblk, lo, hi);
    if (*token >= vocab) return;
    for (uint i = lo + lid; i < hi; i += NT) out[i] = float(table[ulong(*token) * width + i]);
}

void op_embed_overlay_bf16(const thread Inst& in, device const ulong* tab,
                           uint slice, uint nblk, uint lid) {
    device ushort* out = ten<ushort>(tab, in, 0);
    device const ushort* table = ten<ushort>(tab, in, 1);
    device const uint* tokens = ten<uint>(tab, in, 2);
    device const float* overlay = ten<float>(tab, in, 3);
    device const uint* overlay_index = ten<uint>(tab, in, 4);
    uint rows = in.i[0], width = in.i[1], vocab = in.i[2], overlay_rows = in.i[3];
    if (!rows || !width || !vocab || rows > UINT_MAX / width) return;
    uint lo, hi; range(rows * width, slice, nblk, lo, hi);
    for (uint index = lo + lid; index < hi; index += NT) {
        uint row = index / width, column = index % width, selected = overlay_index[row];
        if (selected == UINT_MAX) {
            if (tokens[row] >= vocab) return;
            out[index] = table[ulong(tokens[row]) * width + column];
        } else {
            if (selected >= overlay_rows) return;
            out[index] = f2bf(overlay[ulong(selected) * width + column]);
        }
    }
}

void op_lstm_cell_f32(const thread Inst& in, device const ulong* tab,
                      uint slice, uint nblk, uint lid) {
    device float* h = ten<float>(tab, in, 0);
    device float* c = ten<float>(tab, in, 1);
    device const float* gates = ten<float>(tab, in, 2);
    device const float* previous = ten<float>(tab, in, 3);
    uint width = in.i[0], lo, hi; range(width, slice, nblk, lo, hi);
    for (uint i = lo + lid; i < hi; i += NT) {
        float cell = fma(sigmoidf(gates[width + i]), previous[i],
                         sigmoidf(gates[i]) * tanh(gates[2u * width + i]));
        c[i] = cell;
        h[i] = sigmoidf(gates[3u * width + i]) * tanh(cell);
    }
}

void op_argmax_f32(const thread Inst& in, device const ulong* tab, uint slice,
                   uint nblk, threadgroup float* scratch, uint lid) {
    device uint* ids = ten<uint>(tab, in, 0);
    device const float* x = ten<float>(tab, in, 1);
    uint rows = in.i[0], width = in.i[1];
    for (uint row = slice; row < rows; row += nblk) {
        float best = -INFINITY; uint best_id = 0u;
        for (uint i = lid; i < width; i += NT) {
            float value = x[ulong(row) * width + i];
            if (value > best || (value == best && i < best_id)) { best = value; best_id = i; }
        }
        scratch[lid] = best; scratch[NT + lid] = as_type<float>(best_id);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = NT / 2u; stride > 0u; stride >>= 1u) {
            if (lid < stride) {
                float other = scratch[lid + stride];
                uint other_id = as_type<uint>(scratch[NT + lid + stride]);
                uint own_id = as_type<uint>(scratch[NT + lid]);
                if (other > scratch[lid] || (other == scratch[lid] && other_id < own_id)) {
                    scratch[lid] = other; scratch[NT + lid] = as_type<float>(other_id);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0u) ids[row] = as_type<uint>(scratch[NT]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

void op_relu_f32(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint lid) {
    device float* out = ten<float>(tab, in, 0);
    device const float* x = ten<float>(tab, in, 1);
    uint lo, hi; range(in.i[0], slice, nblk, lo, hi);
    for (uint i = lo + lid; i < hi; i += NT) out[i] = max(x[i], 0.0f);
}

void op_broadcast_add_f32(const thread Inst& in, device const ulong* tab,
                          uint slice, uint nblk, uint lid) {
    device float* out = ten<float>(tab, in, 0);
    device const float* matrix = ten<float>(tab, in, 1);
    device const float* vector = ten<float>(tab, in, 2);
    uint width = in.i[1], lo, hi; range(in.i[0] * width, slice, nblk, lo, hi);
    for (uint i = lo + lid; i < hi; i += NT) out[i] = matrix[i] + vector[i % width];
}

void op_causal_depthwise_f32(const thread Inst& in, device const ulong* tab,
                             uint slice, uint nblk, uint lid) {
    device float* out = ten<float>(tab, in, 0);
    device const float* x = ten<float>(tab, in, 1);
    device const half* weight = ten<half>(tab, in, 2);
    uint rows = in.i[0], channels = in.i[1], kernel_width = in.i[2], lo, hi;
    range(rows * channels, slice, nblk, lo, hi);
    for (uint index = lo + lid; index < hi; index += NT) {
        uint row = index / channels, channel = index % channels; float sum = 0.0f;
        for (uint tap = 0; tap < kernel_width; tap++) {
            int source = int(row) + int(tap) + 1 - int(kernel_width);
            if (source >= 0) sum = fma(x[ulong(source) * channels + channel],
                                       float(weight[ulong(channel) * kernel_width + tap]), sum);
        }
        out[index] = sum;
    }
}

void op_relative_attention_f32(const thread Inst& in, device const ulong* tab, uint slice,
                               threadgroup float* scores, uint lid, uint sg, uint lane) {
    device float* context = ten<float>(tab, in, 0);
    device const float* query = ten<float>(tab, in, 1);
    device const float* key = ten<float>(tab, in, 2);
    device const float* value = ten<float>(tab, in, 3);
    device const float* position = ten<float>(tab, in, 4);
    device const float* bu = ten<float>(tab, in, 5);
    device const float* bv = ten<float>(tab, in, 6);
    uint rows = in.i[0], width = in.i[1], heads = in.i[2], chunk = in.i[3], left = in.i[4];
    uint cluster = sg / 4u, local_sg = sg % 4u, local_lid = lid % 128u;
    uint item = slice * 8u + cluster, qr = item / heads, head = item % heads, hw = width / heads;
    bool active = item < rows * heads;
    uint first = 0, last = rows;
    if (active && left != 0xffffffffu) { uint qc = qr / chunk; first = (qc > left ? qc - left : 0u) * chunk; last = min(rows, (qc + 1u) * chunk); }
    uint count = last - first, qb = qr * width + head * hw, bb = head * hw;
    threadgroup float* item_scores = scores + cluster * 64u;
    if (count > 64u) active = false;
    for (uint local = local_sg; active && local < count; local += 4u) {
        uint kr = first + local, kb = kr * width + bb, pb = (rows - 1u + kr - qr) * width + bb;
        float sum = 0.0f;
        for (uint col = lane; col < hw; col += 32u)
            sum += key[kb + col] * (query[qb + col] + bu[bb + col]) +
                   position[pb + col] * (query[qb + col] + bv[bb + col]);
        float score = simd_sum(sum) * rsqrt(float(hw));
        if (lane == 0u) item_scores[local] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active && local_lid == 0u) {
        float maximum = -INFINITY; for (uint i = 0; i < count; i++) maximum = max(maximum, item_scores[i]);
        float sum = 0.0f; for (uint i = 0; i < count; i++) { item_scores[i] = exp(item_scores[i] - maximum); sum += item_scores[i]; }
        for (uint i = 0; i < count; i++) item_scores[i] /= sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint col = local_lid; active && col < hw; col += 128u) {
        float sum = 0.0f;
        for (uint local = 0; local < count; local++) sum = fma(item_scores[local], value[ulong(first + local) * width + bb + col], sum);
        context[ulong(qr) * width + bb + col] = sum;
    }
}

void op_grouped_attention_f32(const thread Inst& in, device const ulong* tab, uint slice,
                              threadgroup float* scores, uint lid, uint sg, uint lane) {
    device float* context = ten<float>(tab, in, 0);
    device const float* query = ten<float>(tab, in, 1);
    device const float* key = ten<float>(tab, in, 2);
    device const float* value = ten<float>(tab, in, 3);
    uint rows = in.i[0], width = in.i[1], head_width = in.i[2];
    uint group_rows = in.i[3], flags = in.i[4];
    uint valid_rows = in.t[4] == 0xffffu ? rows : *ten<uint>(tab, in, 4);
    bool geometry = head_width != 0u && width % head_width == 0u &&
                    group_rows != 0u && group_rows <= 256u &&
                    valid_rows != 0u && valid_rows <= rows;
    uint heads = geometry ? width / head_width : 0u;
    uint item = slice * NSG + sg, row = heads ? item / heads : 0u;
    uint head = heads ? item % heads : 0u;
    bool active = geometry && item < valid_rows * heads;
    uint first = active ? row / group_rows * group_rows : 0u;
    uint last = min(first + group_rows, valid_rows), count = last - first;
    ulong qb = ulong(row) * width + head * head_width;
    uint score_stride = geometry ? group_rows : 1u;
    threadgroup float* item_scores = scores + sg * score_stride;
    for (uint local = lane; active && local < count; local += 32u) {
        ulong kb = ulong(first + local) * width + head * head_width;
        float score = 0.0f;
        for (uint col = 0; col < head_width; col++)
            score += query[qb + col] * key[kb + col];
        if (flags & 1u) score = round_bf16_f32(score);
        item_scores[local] = score * rsqrt(float(head_width));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (active && lane == 0u) {
        float maximum = -INFINITY, denominator = 0.0f;
        for (uint local = 0; local < count; local++)
            maximum = max(maximum, item_scores[local]);
        for (uint local = 0; local < count; local++) {
            item_scores[local] = exp(item_scores[local] - maximum);
            denominator += item_scores[local];
        }
        for (uint local = 0; local < count; local++) {
            float probability = item_scores[local] / denominator;
            item_scores[local] = flags & 2u ? round_bf16_f32(probability) : probability;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint col = lane; active && col < head_width; col += 32u) {
        float sum = 0.0f;
        for (uint local = 0; local < count; local++) {
            ulong vb = ulong(first + local) * width + head * head_width;
            sum += item_scores[local] * value[vb + col];
        }
        context[qb + col] = flags & 4u ? round_bf16_f32(sum) : sum;
    }
}

bool exec_op(const thread Inst& in, device const ulong* tab, uint slice, uint nblk,
             threadgroup float* tile, threadgroup float* red, threadgroup ulong* keys,
             uint lid, uint sg, uint lane) {
    switch (in.op) {
        case 0: return true;
        case 156: op_affine_q4(in, tab, slice, nblk, false, tile, lid, sg, lane); return true;
        case 157: op_affine_q4(in, tab, slice, nblk, true, tile, lid, sg, lane); return true;
        case 158: op_q8_gemm_f32(in, tab, slice, tile, lid, sg, lane); return true;
        case 159: op_layernorm_f32(in, tab, slice, nblk, red, lid, sg, lane); return true;
        case 160: op_scaled_add_f32(in, tab, slice, nblk, lid); return true;
        case 161: op_glu_f32(in, tab, slice, nblk, lid); return true;
        case 162: op_causal_depthwise_f32(in, tab, slice, nblk, lid); return true;
        case 163: op_relative_attention_f32(in, tab, slice, tile, lid, sg, lane); return true;
        case 164: op_silu_f32(in, tab, slice, in.blocks, lid); return true;
        case 165: op_dense_gemm_f32(in, tab, slice, tile, lid, sg, lane); return true;
        case 166: op_embed_f16_f32(in, tab, slice, in.blocks, lid); return true;
        case 167: op_lstm_cell_f32(in, tab, slice, in.blocks, lid); return true;
        case 168: op_argmax_f32(in, tab, slice, in.blocks, tile, lid); return true;
        case 169: op_relu_f32(in, tab, slice, in.blocks, lid); return true;
        case 170: op_broadcast_add_f32(in, tab, slice, in.blocks, lid); return true;
        case 171: op_conv2d_f32(in, tab, slice, tile, lid, sg, lane); return true;
        case 172: op_pack_ncfw_rows_f32(in, tab, slice, lid); return true;
        case 173: op_grouped_attention_f32(in, tab, slice, tile, lid, sg, lane); return true;
        case 174: op_embed_overlay_bf16(in, tab, slice, in.blocks, lid); return true;
        case 1: op_rmsnorm(in, tab, slice, nblk, red, lid, sg, lane); return true;
        case 3: op_headnorm_rope<false>(in, tab, slice, nblk, sg, lane); return true;
        case 37: op_headnorm_rope<true>(in, tab, slice, nblk, sg, lane); return true;
        case 32: op_quant_fp8(in, tab, slice, nblk, red, lid, sg, lane); return true;
        case 4: op_residual(in, tab, slice, nblk, lid); return true;
        case 5: op_glu(in, tab, slice, nblk, lid); return true;
        case 6: op_embed(in, tab, slice, nblk, lid); return true;
        case 7: op_softcap(in, tab, slice, nblk, lid); return true;
        case 8: op_gemm(in, tab, slice, nblk, 256, 256, 0u, tile, lid, sg, lane); return true;
        case 14: op_gemm(in, tab, slice, nblk, 64, 64, 0u, tile, lid, sg, lane); return true;
        case 15: op_gemm(in, tab, slice, nblk, 128, 128, 0u, tile, lid, sg, lane); return true;
        case 94: op_gemm(in, tab, slice, nblk, 128, 256, 0u, tile, lid, sg, lane); return true;
        case 95: op_gemm(in, tab, slice, nblk, 192, 256, 0u, tile, lid, sg, lane); return true;
        case 33: op_gemm(in, tab, slice, nblk, 256, 256, 1u, tile, lid, sg, lane); return true;
        case 34: op_gemm(in, tab, slice, nblk, 128, 128, 1u, tile, lid, sg, lane); return true;
        case 35: op_gemm(in, tab, slice, nblk, 64, 64, 1u, tile, lid, sg, lane); return true;
        // MXFP4 prefill rungs: the 256-row and wide rungs take the 128x128 geometry here.
        case 93: op_gemm(in, tab, slice, nblk, 128, 128, 2u, tile, lid, sg, lane); return true;
        case 96: op_gemm(in, tab, slice, nblk, 128, 128, 2u, tile, lid, sg, lane); return true;
        case 97: op_gemm(in, tab, slice, nblk, 64, 64, 2u, tile, lid, sg, lane); return true;
        case 98: op_gemm(in, tab, slice, nblk, 128, 128, 2u, tile, lid, sg, lane); return true;
        case 113: op_gemm_glu(in, tab, slice, nblk, 2u, tile, lid, sg, lane); return true;
        case 91: op_gemv_mxfp4(in, tab, slice, nblk, sg, lane); return true;
        case 92: op_gemv_glu_mxfp4(in, tab, slice, nblk, sg, lane); return true;
        case 100: op_gemm(in, tab, slice, nblk, 128, 256, true, tile, lid, sg, lane); return true;
        case 101: op_gemm(in, tab, slice, nblk, 192, 256, true, tile, lid, sg, lane); return true;
        case 20: op_gemm_glu(in, tab, slice, nblk, 0u, tile, lid, sg, lane); return true;
        case 36: op_gemm_glu(in, tab, slice, nblk, 1u, tile, lid, sg, lane); return true;
        case 10: op_gemv(in, tab, slice, nblk, sg, lane); return true;
        case 19: op_gemv_glu(in, tab, slice, nblk, sg, lane); return true;
        case 22: op_gemv_qkv(in, tab, slice, nblk, sg, lane); return true;
        case 115: op_gemv_qkv_fp8(in, tab, slice, nblk, sg, lane); return true;
        case 30: op_gemv_fp8(in, tab, slice, nblk, sg, lane); return true;
        case 31: op_gemv_glu_fp8(in, tab, slice, nblk, sg, lane); return true;
        case 11: op_flash_prefill<ushort>(in, tab, slice, nblk, tile, sg, lane); return true;
        case 12: op_flash_decode<ushort>(in, tab, slice, nblk, tile, sg, lane); return true;
        case 39: op_flash_prefill<uchar>(in, tab, slice, nblk, tile, sg, lane); return true;
        case 38: op_flash_decode<uchar>(in, tab, slice, nblk, tile, sg, lane); return true;
        case 13: op_flash_merge(in, tab, slice, nblk, lid); return true;
        case 16: op_norm_residual(in, tab, slice, nblk, red, lid, sg, lane); return true;
        case 17: op_argmax(in, tab, slice, nblk, keys, lid); return true;
        case 18: op_argmax_fin(in, tab, slice, lid); return true;
        case 21: op_add_norm(in, tab, slice, nblk, red, lid, sg, lane); return true;
        case 23: op_norm_residual_norm(in, tab, slice, nblk, red, lid, sg, lane); return true;
        case 155: return op_per_layer_input(in, tab, slice, nblk, tile, red, lid, sg, lane);
        case 83: op_moe_router_topk_pf(in, tab, slice, nblk, sg, lane); return true;
        case 84: op_moe_align_pf(in, tab, slice, lid); return true;
        case 87: op_moe_combine_pf(in, tab, slice, nblk, lid); return true;
        case 150: op_moe_glu_mx(in, tab, slice, nblk, sg, lane); return true;
        case 151: op_moe_down_mx(in, tab, slice, nblk, sg, lane); return true;
        case 152: op_moe_glu_mx_pf(in, tab, slice, nblk, tile, lid, sg, lane); return true;
        case 153: op_moe_down_mx_pf(in, tab, slice, nblk, tile, lid, sg, lane); return true;
        default: return false;
    }
}

kernel void plow_interp(device const Inst* insts [[buffer(0)]],
                        device const Ent* stream [[buffer(1)]],
                        device const uint* stream_ofs [[buffer(2)]],
                        device const uint* stream_len [[buffer(3)]],
                        device const Wait* waits [[buffer(4)]],
                        device const uint* succs [[buffer(5)]],
                        device atomic_uint* ctr [[buffer(6)]],
                        device const ulong* tab [[buffer(7)]],
                        constant Params& P [[buffer(8)]],
                        device uint* fault [[buffer(9)]],
                        uint cu [[threadgroup_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    // The whole 32 KB. The norm reduction scratch and the argmax keys alias into it: no op uses
    // them together with the region they overlap (red = the last 32 floats, keys = the first 8 KB).
    threadgroup float tile[TILE_FLOATS];
    threadgroup float* red = tile + TILE_FLOATS - NSG;
    threadgroup ulong* keys = (threadgroup ulong*)tile;
    // The wait-gate flag also aliases into `tile` (one slot below `red`); nothing is live there
    // between ops.
    threadgroup uint* gate = (threadgroup uint*)(tile + TILE_FLOATS - NSG - 1u);
    if (cu >= P.n_cu) return;
    uint head = stream_ofs[cu], end = head + stream_len[cu];
    for (; head < end; head++) {
        Ent e = stream[head];
        if (uint(e.seg) != P.seg) continue;
        // Segment boundary for a host/ANE op (Event mode): the host dispatches [0, i) and [i+1, n)
        // around instruction i and bumps i's successors itself.
        if (e.inst < P.inst_lo || e.inst >= P.inst_hi) continue;
        if (lid == 0) {
            uint ok = 1u;
            for (uint w = 0; w < e.wait_len; w++) {
                Wait wt = waits[e.wait_ofs + w];
                uint it = 0;
                while (atomic_load_explicit(ctr + wt.id, memory_order_relaxed) < wt.threshold) {
                    if (++it > P.spin_max) { ok = 0u; break; }
                }
                if (!ok) break;
            }
            *gate = ok;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (*gate == 0u) {
            if (lid == 0) fault[0] = 0x80000000u | e.inst;
            return;
        }
        // Acquire: producers' plain stores are published by their release fence below; this
        // side must not serve a stale L1 line (probe (f): plain loads after a counter are stale
        // whenever this core touched the address earlier in the dispatch).
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        Inst in = insts[e.inst];
        if (!exec_op(in, tab, e.slice, in.blocks, tile, red, keys, lid, sg, lane)) {
            if (lid == 0) fault[0] = 0x40000000u | e.inst;
            return;
        }
        threadgroup_barrier(mem_flags::mem_device);
        // Release: without this device-scope fence the op's plain stores stay invisible to every
        // other threadgroup for the rest of the dispatch (probe (f)); the barrier alone is not enough.
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (lid == 0)
            for (uint s = 0; s < e.succ_len; s++) atomic_fetch_add_explicit(ctr + succs[e.succ_ofs + s], 1u, memory_order_relaxed);
    }
}

// Diagnostic: one instruction per dispatch, threadgroup = slice. Kernel boundaries make every
// producer's stores visible, so a run that is right here and wrong in `plow_interp` is a
// cross-threadgroup visibility problem, not an op bug.
struct SingleParams { uint inst; };
kernel void plow_single(device const Inst* insts [[buffer(0)]],
                        device const ulong* tab [[buffer(7)]],
                        constant SingleParams& S [[buffer(8)]],
                        device uint* fault [[buffer(9)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    // The whole 32 KB. The norm reduction scratch and the argmax keys alias into it: no op uses
    // them together with the region they overlap (red = the last 32 floats, keys = the first 8 KB).
    threadgroup float tile[TILE_FLOATS];
    threadgroup float* red = tile + TILE_FLOATS - NSG;
    threadgroup ulong* keys = (threadgroup ulong*)tile;
    Inst in = insts[S.inst];
    if (tg >= in.blocks) return;
    if (!exec_op(in, tab, tg, in.blocks, tile, red, keys, lid, sg, lane))
        if (lid == 0) fault[0] = 0x40000000u | S.inst;
}

kernel void plow_mx4_dedicated(device const Inst* insts [[buffer(0)]],
 device const ulong* tab [[buffer(7)]], constant SingleParams& S [[buffer(8)]],
 device uint* fault [[buffer(9)]], uint tg [[threadgroup_position_in_grid]],
 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    Inst in = insts[S.inst];
    if ((in.op != 91 && in.op != 92) || !in.i[0] || !in.i[1] || (in.i[1] % 8)
        || !in.i[2] || (in.i[2] % 32)) {
        if (tg == 0 && sg == 0 && lane == 0) fault[0] = 0x40000000u | S.inst;
        return;
    }
    uint blocks = uint(in.i[1]) / (in.op == 92 ? 2u : 8u);
    if (tg >= blocks) return;
    if (in.op == 92) op_gemv_glu_mxfp4(in, tab, tg, blocks, sg, lane);
    else op_gemv_mxfp4(in, tab, tg, blocks, sg, lane);
}

kernel void plow_mx4_prefill(device const Inst* insts [[buffer(0)]],
 device const ulong* tab [[buffer(7)]], constant SingleParams& S [[buffer(8)]],
 device uint* fault [[buffer(9)]], uint tg [[threadgroup_position_in_grid]],
 uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tile[TILE_FLOATS];
    Inst in = insts[S.inst];
    if (tg >= in.blocks) return;
    if (in.op == 93 || in.op == 96 || in.op == 98)
        op_gemm(in, tab, tg, in.blocks, 128, 128, 2u, tile, lid, sg, lane);
    else if (in.op == 97)
        op_gemm(in, tab, tg, in.blocks, 64, 64, 2u, tile, lid, sg, lane);
    else if (lid == 0) fault[0] = 0x40000000u | S.inst;
}
