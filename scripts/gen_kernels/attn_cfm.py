"""TileLang 3xTF32 attention for the S3Gen CFM (cached-prompt `csynth`): AttentionF32 with a key
prefix, head width 64, no bias / causal.

qkv: [B][T][3*H*64] f32 (Q | K | V columns); prefix: [blocks][pre][2*H*64] f32 (K | V), item b's
prefix block pidx[b]; klen[b]: visible keys (prefix rows first, then the item's own rows).
out: [B][T][H*64] f32. Scores, products and P are split hi/lo tf32 (3 products each, lo*lo
dropped), as op_speech_f32.cuh sp_attention_tc64p.
One block per (BM-row query tile, head, item).
"""
import tilelang
import tilelang.language as T

import os
PASS_CONFIGS = {} if os.environ.get("TL_WS") else {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def build(bm, bn, threads, stages, H=8, pre=306):
    B, Tq, nblk = T.dynamic("B, Tq, nblk")
    hw = 64
    f32 = "float32"
    log2e = 1.4426950408889634
    tf = T.tfloat32 if os.environ.get("TL_TF") else f32
    G = T.wgmma_gemm if os.environ.get("TL_WG") else T.gemm

    def hi(x):
        u = T.reinterpret("uint32", x)
        return T.reinterpret(f32, (u + T.uint32(0x1000)) & T.uint32(0xFFFFE000))

    @T.prim_func
    def attn_cfm(
        qkv: T.Tensor((B, Tq, 3 * H * hw), f32),
        prefix: T.Tensor((nblk, pre, 2 * H * hw), f32),
        pidx: T.Tensor((B,), "int32"),
        klen: T.Tensor((B,), "int32"),
        out: T.Tensor((B, Tq, H * hw), f32),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(Tq, bm), H, B, threads=threads) as (mt, h, b):
            Qh = T.alloc_shared((bm, hw), tf)
            Ql = T.alloc_shared((bm, hw), tf)
            Kh = T.alloc_shared((bn, hw), tf)
            Kl = T.alloc_shared((bn, hw), tf)
            Vh = T.alloc_shared((hw, bn), tf)
            Vl = T.alloc_shared((hw, bn), tf)
            Ph = T.alloc_shared((bm, bn), tf)
            Pl = T.alloc_shared((bm, bn), tf)
            S = T.alloc_fragment((bm, bn), f32)
            O = T.alloc_fragment((bm, hw), f32)
            m = T.alloc_fragment((bm,), f32)
            m_prev = T.alloc_fragment((bm,), f32)
            corr = T.alloc_fragment((bm,), f32)
            lsum = T.alloc_fragment((bm,), f32)
            l = T.alloc_fragment((bm,), f32)
            n = klen[b]
            pb = pidx[b]
            sl = scale * log2e
            for i, j in T.Parallel(bm, hw):
                r = mt * bm + i
                x = T.if_then_else(r < Tq, qkv[b, T.min(r, Tq - 1), h * hw + j], T.float32(0))
                Qh[i, j] = hi(x)
                Ql[i, j] = x - hi(x)
            T.fill(O, 0)
            T.fill(l, 0)
            T.fill(m, -T.infinity(f32))
            for t in T.Pipelined(T.ceildiv(n, bn), num_stages=stages):
                for i, j in T.Parallel(bn, hw):
                    kj = t * bn + i
                    xk = T.if_then_else(
                        kj < pre, prefix[pb, T.min(kj, pre - 1), h * hw + j],
                        T.if_then_else(kj < n, qkv[b, T.max(kj - pre, 0), H * hw + h * hw + j], T.float32(0)))
                    Kh[i, j] = hi(xk)
                    Kl[i, j] = xk - hi(xk)
                    xv = T.if_then_else(
                        kj < pre, prefix[pb, T.min(kj, pre - 1), H * hw + h * hw + j],
                        T.if_then_else(kj < n, qkv[b, T.max(kj - pre, 0), 2 * H * hw + h * hw + j], T.float32(0)))
                    Vh[j, i] = hi(xv)
                    Vl[j, i] = xv - hi(xv)
                G(Ql, Kh, S, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
                G(Qh, Kl, S, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                G(Qh, Kh, S, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(bm, bn):
                    S[i, j] = T.if_then_else(t * bn + j < n, S[i, j] * sl, -T.infinity(f32))
                T.copy(m, m_prev)
                T.reduce_max(S, m, dim=1, clear=False)
                for i in T.Parallel(bm):
                    corr[i] = T.exp2(m_prev[i] - m[i])
                for i, j in T.Parallel(bm, bn):
                    S[i, j] = T.exp2(S[i, j] - m[i])
                T.reduce_sum(S, lsum, dim=1)
                for i in T.Parallel(bm):
                    l[i] = l[i] * corr[i] + lsum[i]
                for i, j in T.Parallel(bm, hw):
                    O[i, j] *= corr[i]
                for i, j in T.Parallel(bm, bn):
                    Ph[i, j] = hi(S[i, j])
                    Pl[i, j] = S[i, j] - hi(S[i, j])
                G(Pl, Vh, O, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                G(Ph, Vl, O, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                G(Ph, Vh, O, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(bm, hw):
                r = mt * bm + i
                if r < Tq:
                    out[b, r, h * hw + j] = O[i, j] / l[i]
    return attn_cfm


def compile(bm, bn, threads, stages, H=8, pre=306, target=None):
    return tilelang.compile(build(bm, bn, threads, stages, H, pre),
                            target=target or {"kind": "cuda", "arch": "sm_90a"},
                            pass_configs=PASS_CONFIGS)
