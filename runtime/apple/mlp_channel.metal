// Shared channel-MLP kernels, appended to interp.metal only for explicit offload/probes.
// The loops stride by the dispatched threadgroup count, so the host picks the width.
kernel void mlp_gate(device const ushort* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
                     device const uchar* wu [[buffer(2)]], device const uchar* sg_w [[buffer(3)]],
                     device const uchar* su_w [[buffer(4)]], device ushort* u [[buffer(5)]],
                     constant uint4& p [[buffer(6)]], uint slice [[threadgroup_position_in_grid]],
                     uint nblk [[threadgroups_per_grid]],
                     uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tile[TILE_FLOATS];
    GemmArgs g, up;
    gemm_args_reset(g); gemm_args_reset(up);
    g.A = up.A = x; g.M = up.M = p.x; g.K = up.K = p.y; g.N = up.N = p.z;
    if (p.w == 1u) {
        g.B8 = wg; up.B8 = wu;
        g.wscale = (device const float*)sg_w; up.wscale = (device const float*)su_w;
    } else if (p.w == 2u) {
        g.B4 = wg; up.B4 = wu; g.S8 = sg_w; up.S8 = su_w;
    } else {
        g.B16 = (device const ushort*)wg; up.B16 = (device const ushort*)wu;
    }
    uint tn = (p.z + 127u) / 128u, tm = (p.x + 255u) / 256u;
    for (uint lin = slice; lin < tm * tn; lin += nblk) {
        uint m0 = lin / tn * 256u, n0 = lin % tn * 128u;
        gemm_tile2_glu(g, up, u, 1u, 0.0f, 0.0f, p.w == 1u,
            m0, min(m0 + 256u, p.x), n0, min(n0 + 128u, p.z), tile, lid, sg, lane);
    }
}

kernel void mlp_down(device const ushort* u [[buffer(0)]], device const uchar* wd [[buffer(1)]],
                     device const uchar* sw [[buffer(2)]], device float* y [[buffer(3)]],
                     constant uint4& p [[buffer(4)]], uint slice [[threadgroup_position_in_grid]],
                     uint nblk [[threadgroups_per_grid]],
                     uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tile[TILE_FLOATS];
    GemmArgs g; gemm_args_reset(g);
    bool bf16_out = (p.w & 4u) != 0;
    g.A = u; g.M = p.x; g.N = p.y; g.K = p.z;
    if (!bf16_out) g.Cf = y;
    if ((p.w & 3u) == 1u) { g.B8 = wd; g.wscale = (device const float*)sw; }
    else if ((p.w & 3u) == 2u) { g.B4 = wd; g.S8 = sw; }
    else g.B16 = (device const ushort*)wd;
    uint tn = (p.y + 127u) / 128u, tm = (p.x + 127u) / 128u;
    for (uint lin = slice; lin < tm * tn; lin += nblk) {
        uint m0 = lin / tn * 128u, n0 = lin % tn * 128u;
        gemm_tile2(g, bf16_out ? (device ushort*)y : (device ushort*)0, p.y, m0, min(m0 + 128u, p.x),
            n0, min(n0 + 128u, p.y), 128u, tile, lid, sg, lane);
    }
}

kernel void mlp_finish(device const float* g [[buffer(0)]], device const float* a [[buffer(1)]],
                       device const ushort* residual [[buffer(2)]], device ushort* out [[buffer(3)]],
                       constant uint3& p [[buffer(4)]], uint e [[thread_position_in_grid]]) {
    if (e < p.x) {
        float value = p.z ? bf2f(((device const ushort*)g)[e]) : g[e];
        out[e] = f2bf((value + (p.y ? a[e] : 0.0f)) + bf2f(residual[e]));
    }
}

kernel void mlp_consume(device const ushort* x [[buffer(0)]], device ushort* out [[buffer(1)]],
                        constant uint2& p [[buffer(2)]], uint slice [[threadgroup_position_in_grid]],
                        uint nblk [[threadgroups_per_grid]],
                        uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    for (uint row = slice; row < p.x; row += nblk) {
        float ss = 0.0f;
        for (uint k = lid; k < p.y; k += NT) { float v = bf2f(x[row * p.y + k]); ss += v * v; }
        // Probe-only consumer: eps is Llama's, not read from the packet.
        float inv = rsqrt(tg_sum(ss, red, lid, sg, lane) / float(p.y) + 1e-5f);
        for (uint k = lid; k < p.y; k += NT) out[row * p.y + k] = f2bf(bf2f(x[row * p.y + k]) * inv);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
