// interp.metal — the device-ISA interpreter for Apple GPUs (plans/apple-silicon-backend.md §4.2).
//
// One threadgroup per virtual CU (grid = n_cu, all resident — probe (b)), 256 threads = 8
// simdgroups = the G_WAVES packing every kernel assumes. Each threadgroup walks its own stream of
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
// e4m3 -> f32 exactly, branch-free: the 7 magnitude bits land in the f32 exponent/mantissa
// fields (exponent still biased by 7, i.e. the value times 2^-120), the sign in bit 31; the
// caller multiplies by 2^120 (per element here, or once per accumulator in the dot loops —
// exact either way). e == 0 codes become f32 denormals, which equals the fp8 subnormal value
// when denormals are honoured and 0 when flushed (the CPU tiers make the same DAZ choice).
constant float E4M3_REBIAS = 0x1p120f;
inline float e4m3_raw(uint c) { return as_type<float>(((c & 0x7Fu) << 20) | ((c & 0x80u) << 24)); }
inline float e4m3(uint c) { return e4m3_raw(c) * E4M3_REBIAS; }
inline float4 e4m3x4_raw(uchar4 c) {
    uint4 u = uint4(c);
    return as_type<float4>(((u & 0x7Fu) << 20) | ((u & 0x80u) << 24));
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
    for (k = (K & ~255u) + lane; k < K; k += 32u) acc += bf2f(w[k]) * bf2f(x[k]);
    return simd_sum(acc);
}
inline float4 e4m3x4(uchar4 c) { return e4m3x4_raw(c) * E4M3_REBIAS; }

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
    for (k = (K & ~255u) + lane; k < K; k += 32u) {
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
        acc.x += dot(e4m3x4_raw(a0), x0) + dot(e4m3x4_raw(a1), x1);
        acc.y += dot(e4m3x4_raw(b0), x0) + dot(e4m3x4_raw(b1), x1);
        acc.z += dot(e4m3x4_raw(c0), x0) + dot(e4m3x4_raw(c1), x1);
        acc.w += dot(e4m3x4_raw(d0), x0) + dot(e4m3x4_raw(d1), x1);
    }
    for (k = (K & ~255u) + lane; k < K; k += 32u) {
        float xv = bf2f(x[k]);
        acc.x += e4m3_raw(W[k]) * xv;
        acc.y += e4m3_raw(W[ldw + k]) * xv;
        acc.z += e4m3_raw(W[2 * ldw + k]) * xv;
        acc.w += e4m3_raw(W[3 * ldw + k]) * xv;
    }
    return float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w)) * E4M3_REBIAS;
}
inline float dot_fp8(device const uchar* w, device const ushort* x, uint K, uint lane) {
    float acc = 0.0f;
    uint k = lane * 8u;
    for (; k + 776u <= K; k += 1024u) {
        uchar4 a0 = *(device const uchar4*)(w + k), a1 = *(device const uchar4*)(w + k + 4);
        uchar4 b0 = *(device const uchar4*)(w + k + 256), b1 = *(device const uchar4*)(w + k + 260);
        uchar4 c0 = *(device const uchar4*)(w + k + 512), c1 = *(device const uchar4*)(w + k + 516);
        uchar4 d0 = *(device const uchar4*)(w + k + 768), d1 = *(device const uchar4*)(w + k + 772);
        acc += dot(e4m3x4_raw(a0), bf4(*(device const ushort4*)(x + k))) + dot(e4m3x4_raw(a1), bf4(*(device const ushort4*)(x + k + 4)));
        acc += dot(e4m3x4_raw(b0), bf4(*(device const ushort4*)(x + k + 256))) + dot(e4m3x4_raw(b1), bf4(*(device const ushort4*)(x + k + 260)));
        acc += dot(e4m3x4_raw(c0), bf4(*(device const ushort4*)(x + k + 512))) + dot(e4m3x4_raw(c1), bf4(*(device const ushort4*)(x + k + 516)));
        acc += dot(e4m3x4_raw(d0), bf4(*(device const ushort4*)(x + k + 768))) + dot(e4m3x4_raw(d1), bf4(*(device const ushort4*)(x + k + 772)));
    }
    for (; k + 8u <= K; k += 256u) {
        uchar4 a = *(device const uchar4*)(w + k), b = *(device const uchar4*)(w + k + 4);
        float4 x0 = bf4(*(device const ushort4*)(x + k)), x1 = bf4(*(device const ushort4*)(x + k + 4));
        acc += dot(e4m3x4_raw(a), x0) + dot(e4m3x4_raw(b), x1);
    }
    for (k = (K & ~255u) + lane; k < K; k += 32u) acc += e4m3_raw(w[k]) * bf2f(x[k]);
    return simd_sum(acc) * E4M3_REBIAS;
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
            float g = dot_bf16(Wg + n * K, x + m * K, K, lane);
            float u = dot_bf16(Wu + n * K, x + m * K, K, lane);
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
    device const ushort* B16;    // [N][K] bf16, or
    device const uchar* B8;      //          e4m3, or
    device const uchar* B4;      //          e2m1 packed (row stride K/2) with
    device const uchar* S8;      //          E8M0 scales (row stride ceil(K/32)), folded at staging
    device const float* wscale;  // [N] for B8
    device const ushort* bias;   // [N] bf16 or null
    uint K;
    uint M, N;                   // bounds for zero fill
};
// One 8x8 accumulator through the simdgroup's scratch: lane l handles elements 2l, 2l+1.
inline void epilogue8x8(simdgroup_float8x8 acc, threadgroup float* scratch, uint lane, uint m_base, uint n_base,
                        uint m1, uint n1, thread const GemmArgs& g, device ushort* C, uint ldc) {
    simdgroup_store(acc, scratch, 8u);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane * 2u; e < lane * 2u + 2u; e++) {
        uint m = m_base + e / 8u, n = n_base + e % 8u;
        if (m < m1 && n < n1) {
            float v = scratch[e];
            if (g.B8) v *= g.wscale[n];
            if (g.bias) v += bf2f(g.bias[n]);
            C[m * ldc + n] = f2bf(v);
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
inline ushort4 load_b16(thread const GemmArgs& g, uint n, uint k) {
    const uint N = g.N, K = g.K;
    if (n >= N || k >= K) return ushort4(0);
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
    if ((K & 3u) == 0u)
        return g.B8 ? f4bf(e4m3x4(*(device const uchar4*)(g.B8 + n * K + k))) : *(device const ushort4*)(g.B16 + n * K + k);
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
    ushort4 ra = a_side ? load_a16(g.A, g.M, g.K, mm + ar, aq * 4u) : ushort4(0);
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
            if (a_side) ra = load_a16(g.A, g.M, g.K, mm + ar, k0 + aq * 4u);
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
    ushort4 ra = a_side ? load_a16(gg.A, M, K, mm + ar, aq * 4u) : ushort4(0);
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
            if (a_side) ra = load_a16(gg.A, M, K, mm + ar, k0 + aq * 4u);
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
                if (fp8) { gv *= gg.wscale[n]; uv *= gu.wscale[n]; }
                if (gg.bias) gv += bf2f(gg.bias[n]);
                if (gu.bias) uv += bf2f(gu.bias[n]);
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
// t0=C t1=A t2=B(e4m3) t3=a_scale? t4=w_scale i0=M i1=N i2=K i4=a_row0 i5=c_row0 (fp8; a_scale -> poison)
// enc: 0 = bf16 weights, 1 = e4m3 + per-channel f32 scale (t4), 2 = MXFP4 (t3 = E8M0 scales)
void op_gemm(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint BM, uint BN, uint enc,
             threadgroup float* tile, uint lid, uint sg, uint lane) {
    uint M = in.i[0], N = in.i[1], K = in.i[2];
    bool fp8 = enc == 1u, mx4 = enc == 2u;
    device ushort* C = ten<ushort>(tab, in, 0) + in.i[5] * N;
    GemmArgs g;
    g.A = ten<ushort>(tab, in, 1) + in.i[4] * K;
    g.B16 = enc == 0u ? ten<ushort>(tab, in, 2) : (device const ushort*)0;
    g.B8 = fp8 ? ten<uchar>(tab, in, 2) : (device const uchar*)0;
    g.B4 = mx4 ? ten<uchar>(tab, in, 2) : (device const uchar*)0;
    g.S8 = mx4 ? ten<uchar>(tab, in, 3) : (device const uchar*)0;
    g.wscale = fp8 ? ten<float>(tab, in, 4) : (device const float*)0;
    g.bias = enc == 0u ? ten<ushort>(tab, in, 7) : (device const ushort*)0;
    g.K = K; g.M = M; g.N = N;
    bool poison = (fp8 && (ten<float>(tab, in, 3) != 0 || !g.wscale)) || (mx4 && !g.S8);
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
    } else if (mx4) {
        gg.B4 = ten<uchar>(tab, in, 2); gu.B4 = ten<uchar>(tab, in, 5);
        gg.S8 = ten<uchar>(tab, in, 3); gu.S8 = ten<uchar>(tab, in, 4);
    } else {
        gg.B16 = ten<ushort>(tab, in, 2); gu.B16 = ten<ushort>(tab, in, 5);
        gg.bias = ten<ushort>(tab, in, 6); gu.bias = ten<ushort>(tab, in, 7);
    }
    bool poison = (fp8 && (ten<float>(tab, in, 3) != 0 || !gg.wscale || !gu.wscale)) || (mx4 && (!gg.S8 || !gu.S8));
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

// ---- norms (rows = slice axis; whole threadgroup per row) ------------------------------------------
// t0=out t1=x t2=gamma? (t3 -> poison)  i0=rows i1=feat i2=out_row0  f0=eps
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
        if (quant) { for (uint i = lid; i < feat; i += NT) o[i] = ushort(0x7fc1); continue; }
        float ss = 0.0f;
        for (uint i = lid; i < feat; i += NT) { float v = bf2f(xr[i]); ss += v * v; }
        ss = tg_sum(ss, red, lid, sg, lane);
        float inv = rsqrt(ss / float(feat) + eps);
        for (uint i = lid; i < feat; i += NT) {
            float g = gamma ? bf2f(gamma[i]) : 1.0f;
            o[i] = f2bf(bf2f(xr[i]) * inv * g);
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
        for (uint j = 0; j < per; j++) out[obase + lane + 32u * j] = f2bf(v[j]);
    }
}

// ---- pointwise ---------------------------------------------------------------------------------------
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
inline void attend_rows(device const ushort* q, device const ushort* kbase, device const ushort* vbase,
                        uint D, uint kv_mask, float scale, uint lo, uint hi, uint qpos, uint window, bool bf16_p,
                        uint lane, thread float (&acc)[16], thread float& m, thread float& l) {
    uint per = D / 32u;
    float qv[16];
    for (uint j = 0; j < per; j++) qv[j] = bf2f(q[lane + 32u * j]);
    for (uint kv = lo; kv < hi; kv++) {
        if (!(kv <= qpos && (window == 0u || qpos - kv < window))) continue;
        device const ushort* kr = kbase + ulong(kv & kv_mask) * D;
        device const ushort* vr = vbase + ulong(kv & kv_mask) * D;
        float s = 0.0f;
        for (uint j = 0; j < per; j++) s += qv[j] * bf2f(kr[lane + 32u * j]);
        s = simd_sum(s) * scale;
        float mnew = max(m, s);
        float corr = m == NEG_INF ? 0.0f : exp(m - mnew);
        float pe = exp(s - mnew);
        if (bf16_p) pe = rbf(pe);
        l = l * corr + pe;
        m = mnew;
        for (uint j = 0; j < per; j++) acc[j] = acc[j] * corr + pe * bf2f(vr[lane + 32u * j]);
    }
}
// t0=Opart t1=mlpart t2=Q t3=K t4=V t5=O_final?  i0=n_q i1=n_kv i2=n_head i3=n_kv_head i4=q_pos0 i5=window
// i6=hd i7=nsplit  f0=scale fj1=kv_stride fj2=kv_mask
void op_flash_prefill(const thread Inst& in, device const ulong* tab, uint slice, uint nblk, uint sg, uint lane) {
    device float* Opart = ten<float>(tab, in, 0);
    device float* mlpart = ten<float>(tab, in, 1);
    device const ushort* Q = ten<ushort>(tab, in, 2);
    device const ushort* K = ten<ushort>(tab, in, 3);
    device const ushort* V = ten<ushort>(tab, in, 4);
    device ushort* O_final = ten<ushort>(tab, in, 5);
    uint n_q = in.i[0], n_kv = in.i[1], n_head = in.i[2], n_kv_head = in.i[3];
    uint q_pos0 = in.i[4], window = in.i[5], D = in.i[6];
    uint nsplit = in.i[7] ? in.i[7] : 1u;
    float scale = as_type<float>(in.fj[0]);
    uint kv_stride = in.fj[1], kv_mask = in.fj[2];
    if (D > 512u || (D & 31u)) return;
    uint gqa = n_head / n_kv_head;
    uint q_tiles = (n_q + FA_BQ_TILE - 1) / FA_BQ_TILE;
    uint n_work = q_tiles * n_head * nsplit;
    uint per = D / 32u;
    for (uint w = slice; w < n_work; w += nblk) {
        uint sp = w % nsplit, h = (w / nsplit) % n_head, qt = w / (nsplit * n_head);
        uint hkv = h / gqa;
        uint q_base = qt * FA_BQ_TILE;
        uint q_tile_last = q_pos0 + q_base + FA_BQ_TILE - 1;
        uint kv_end = min(q_tile_last + 1, n_kv);
        uint q_tile_first = q_pos0 + q_base;
        uint win_lo = (window && q_tile_first >= window) ? q_tile_first - window + 1 : 0;
        uint kv_lo = (win_lo / FA_BKV) * FA_BKV;
        uint tiles_kv = kv_end > kv_lo ? (kv_end - kv_lo + FA_BKV - 1) / FA_BKV : 0u;
        uint perp = (tiles_kv + nsplit - 1) / nsplit;
        uint my_lo = kv_lo + sp * perp * FA_BKV;
        uint my_hi = min(kv_lo + (sp + 1) * perp * FA_BKV, kv_end);
        device const ushort* kbase = K + ulong(hkv) * kv_stride * D;
        device const ushort* vbase = V + ulong(hkv) * kv_stride * D;
        for (uint qi = q_base + sg; qi < q_base + FA_BQ_TILE && qi < n_q; qi += NSG) {
            device const ushort* q = Q + (ulong(qi) * n_head + h) * D;
            uint qg = q_pos0 + qi;
            float m = NEG_INF, l = 0.0f, acc[16];
            for (uint j = 0; j < 16; j++) acc[j] = 0.0f;
            // rows past n_kv are excluded by my_hi <= kv_end <= n_kv
            attend_rows(q, kbase, vbase, D, kv_mask, scale, my_lo, my_hi, qg, window, true, lane, acc, m, l);
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
void op_flash_decode(const thread Inst& in, device const ulong* tab, uint slice, uint nblk,
                     threadgroup float* tile, uint sg, uint lane) {
    device float* Opart = ten<float>(tab, in, 0);
    device float* mlpart = ten<float>(tab, in, 1);
    device const ushort* Q = ten<ushort>(tab, in, 2);
    device const ushort* K = ten<ushort>(tab, in, 3);
    device const ushort* V = ten<ushort>(tab, in, 4);
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
    uint gf = gqa % FA_GF == 0u ? FA_GF : 1u;
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
        device const ushort* kbase = K + (ulong(b) * n_kv_head + hkv) * kv_stride * D;
        device const ushort* vbase = V + (ulong(b) * n_kv_head + hkv) * kv_stride * D;
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
                attend_rows(q, kbase, vbase, D, kv_mask, scale, my_lo, my_hi, qpos, window, false, lane, acc, m, l);
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
// Op 154: Gemma-4 E-series per-layer input block, in place on x (dev_isa.h). One threadgroup per
// row: 32 simdgroups compute the P gate dots (lanes strided over H), the products land in `tile`,
// every thread then owns H/NT rows of the projection, and the two norms reduce through `red`.
// Needs P + H <= TILE_FLOATS (E4B: 256 + 2560).
void op_per_layer_input(const thread Inst& in, device const ulong* tab, uint slice, uint nblk,
                        threadgroup float* tile, threadgroup float* red, uint lid, uint sg, uint lane) {
    device ushort* x = ten<ushort>(tab, in, 0);
    device const ushort* wg = ten<ushort>(tab, in, 1);
    device const ushort* wp = ten<ushort>(tab, in, 2);
    device const ushort* gamma = ten<ushort>(tab, in, 3);
    device const ushort* ple = ten<ushort>(tab, in, 4);
    device ushort* hn = ten<ushort>(tab, in, 5);
    device const ushort* gnext = ten<ushort>(tab, in, 6);
    uint rows = in.i[0], H = in.i[1], P = in.i[2], col0 = in.i[3], stride = in.i[4];
    float eps = as_type<float>(in.fj[0]), ls = as_type<float>(in.fj[1]);
    threadgroup float* a = tile;
    threadgroup float* y = tile + P;
    for (uint t = slice; t < rows; t += nblk) {
        device ushort* xr = x + ulong(t) * H;
        if (P + H > TILE_FLOATS || (P & 7u)) {
            // Same failure as the golden: a poisoned row, never a silent skip.
            for (uint h = lid; h < H; h += NT) xr[h] = 0x7FC0;
            continue;
        }
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
}

bool exec_op(const thread Inst& in, device const ulong* tab, uint slice, uint nblk,
             threadgroup float* tile, threadgroup float* red, threadgroup ulong* keys,
             uint lid, uint sg, uint lane) {
    switch (in.op) {
        case 0: return true;
        case 1: op_rmsnorm(in, tab, slice, nblk, red, lid, sg, lane); return true;
        case 3: op_headnorm_rope(in, tab, slice, nblk, sg, lane); return true;
        case 4: op_residual(in, tab, slice, nblk, lid); return true;
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
        case 30: op_gemv_fp8(in, tab, slice, nblk, sg, lane); return true;
        case 31: op_gemv_glu_fp8(in, tab, slice, nblk, sg, lane); return true;
        case 11: op_flash_prefill(in, tab, slice, nblk, sg, lane); return true;
        case 12: op_flash_decode(in, tab, slice, nblk, tile, sg, lane); return true;
        case 13: op_flash_merge(in, tab, slice, nblk, lid); return true;
        case 16: op_norm_residual(in, tab, slice, nblk, red, lid, sg, lane); return true;
        case 17: op_argmax(in, tab, slice, nblk, keys, lid); return true;
        case 18: op_argmax_fin(in, tab, slice, lid); return true;
        case 21: op_add_norm(in, tab, slice, nblk, red, lid, sg, lane); return true;
        case 23: op_norm_residual_norm(in, tab, slice, nblk, red, lid, sg, lane); return true;
        case 154: op_per_layer_input(in, tab, slice, nblk, tile, red, lid, sg, lane); return true;
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
