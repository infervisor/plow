"""TileLang GQA decode attention (one query row per sequence) over plow's KV ring layout.

K/V: [slots][kv_heads][kv_stride][hd] bf16; logical position p lives in row p & kv_mask. Q/O:
[batch][heads][hd] bf16. Query position = kv_len[b] - 1; window > 0 keeps the last `window`
positions. One block per (kv head, split, sequence) handles the kv head's `gqa` query heads.
nsplit > 1 writes f32 partials and the last split to arrive merges them (no second launch).
"""
import tilelang
import tilelang.language as T

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}


def build(hd, gqa, block_n, stages, threads=128, nsplit=1, hsplit=1):
    assert hsplit == 1 or nsplit == 1
    g = gqa // hsplit
    batch, slots, kv_stride, kvh = T.dynamic("batch, slots, kv_stride, kvh")
    heads = kvh * gqa
    mp = 16  # mma M tile; rows >= gqa are zero padding
    dt, acc = "bfloat16", "float"
    log2e = 1.44269504

    @T.prim_func
    def attn_decode(
        Q: T.Tensor((batch, heads, hd), dt),
        K: T.Tensor((slots, kvh, kv_stride, hd), dt),
        V: T.Tensor((slots, kvh, kv_stride, hd), dt),
        kv_len: T.Tensor((batch,), "int32"),
        O: T.Tensor((batch, heads, hd), dt),
        Opart: T.Tensor((batch, heads, nsplit, hd), acc),
        ML: T.Tensor((batch, heads, nsplit, 2), acc),
        ctr: T.Tensor((batch, kvh), "int32"),
        window: T.int32,
        kv_mask: T.int32,
        scale: T.float32,
    ):
        with T.Kernel(kvh * hsplit * nsplit, batch, threads=threads) as (bx, b):
            hk = bx // (nsplit * hsplit)
            sp = bx % nsplit
            h0 = hk * gqa + (bx // nsplit) % hsplit * g
            Q_s = T.alloc_shared((mp, hd), dt)
            K_s = T.alloc_shared((block_n, hd), dt)
            V_s = T.alloc_shared((block_n, hd), dt)
            S = T.alloc_fragment((mp, block_n), acc)
            P = T.alloc_shared((mp, block_n), dt)
            Oacc = T.alloc_fragment((mp, hd), acc)
            m = T.alloc_fragment((mp,), acc)
            m_prev = T.alloc_fragment((mp,), acc)
            corr = T.alloc_fragment((mp,), acc)
            l = T.alloc_fragment((mp,), acc)
            lsum = T.alloc_fragment((mp,), acc)

            n = kv_len[b]
            lo = T.if_then_else(window > 0 and n > window, n - window, 0)
            t0 = lo // block_n
            nt = T.ceildiv(n, block_n) - t0
            per = T.ceildiv(nt, nsplit)
            ta = t0 + sp * per
            tb = T.min(ta + per, t0 + nt)
            sl = scale * log2e

            T.clear(Q_s)
            T.copy(Q[b, h0:h0 + g, :], Q_s[0:g, :])
            T.fill(Oacc, 0)
            T.fill(l, 0)
            T.fill(m, -T.infinity(acc))
            for t in T.Pipelined(T.max(tb - ta, 0), num_stages=stages):
                ls = (ta + t) * block_n
                ph = ls & kv_mask
                T.copy(K[b, hk, ph:ph + block_n, :], K_s)
                T.gemm(Q_s, K_s, S, transpose_B=True, clear_accum=True,
                       policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(mp, block_n):
                    S[i, j] = T.if_then_else(ls + j >= lo and ls + j < n, S[i, j] * sl,
                                             -T.infinity(acc))
                T.copy(m, m_prev)
                T.reduce_max(S, m, dim=1, clear=False)
                for i in T.Parallel(mp):
                    corr[i] = T.exp2(m_prev[i] - m[i])
                for i, j in T.Parallel(mp, block_n):
                    S[i, j] = T.exp2(S[i, j] - m[i])
                T.reduce_sum(S, lsum, dim=1)
                for i in T.Parallel(mp):
                    l[i] = l[i] * corr[i] + lsum[i]
                for i, j in T.Parallel(mp, hd):
                    Oacc[i, j] *= corr[i]
                T.copy(S, P)
                T.copy(V[b, hk, ph:ph + block_n, :], V_s)
                T.gemm(P, V_s, Oacc, policy=T.GemmWarpPolicy.FullCol)
            if nsplit == 1:
                for i, j in T.Parallel(mp, hd):
                    if i < g:
                        O[b, h0 + i, j] = Oacc[i, j] / l[i]
            else:
                for i, j in T.Parallel(mp, hd):
                    if i < g:
                        Opart[b, h0 + i, sp, j] = Oacc[i, j]
                for i in T.Parallel(mp):
                    if i < g:
                        ML[b, h0 + i, sp, 0] = m[i]
                        ML[b, h0 + i, sp, 1] = l[i]
                tx = T.get_thread_binding()
                flag = T.alloc_shared((1,), "int32")
                T.call_extern("handle", "__threadfence")
                T.sync_threads()
                if tx == 0:
                    flag[0] = T.atomic_add(ctr[b, hk], 1, return_prev=True)
                T.sync_threads()
                if flag[0] == nsplit - 1:
                    T.call_extern("handle", "__threadfence")
                    ms = T.alloc_fragment((mp, nsplit), acc)
                    ws = T.alloc_fragment((mp, nsplit), acc)
                    mx = T.alloc_fragment((mp,), acc)
                    lt = T.alloc_fragment((mp,), acc)
                    W = T.alloc_shared((mp, nsplit), acc)
                    Of = T.alloc_fragment((mp, hd), acc)
                    for i, s in T.Parallel(mp, nsplit):
                        ms[i, s] = T.if_then_else(i < g, ML[b, h0 + i % g, s, 0], 0)
                    T.reduce_max(ms, mx, dim=1)
                    for i, s in T.Parallel(mp, nsplit):
                        ws[i, s] = T.if_then_else(
                            i < g, T.exp2(ms[i, s] - mx[i]) * ML[b, h0 + i % g, s, 1], 1)
                    T.reduce_sum(ws, lt, dim=1)
                    for i, s in T.Parallel(mp, nsplit):
                        W[i, s] = T.if_then_else(i < g, T.exp2(ms[i, s] - mx[i]) / lt[i], 0)
                    T.fill(Of, 0)
                    for s in T.serial(nsplit):
                        for i, j in T.Parallel(mp, hd):
                            if i < g:
                                Of[i, j] += W[i, s] * Opart[b, h0 + i, s, j]
                    for i, j in T.Parallel(mp, hd):
                        if i < g:
                            O[b, h0 + i, j] = Of[i, j]
                    if tx == 0:
                        ctr[b, hk] = 0
    return attn_decode


def compile(hd, gqa, block_n, stages, threads=128, nsplit=1, hsplit=1, target=None):
    return tilelang.compile(build(hd, gqa, block_n, stages, threads, nsplit, hsplit),
                            target=target or {"kind": "cuda", "arch": "sm_90a"},
                            pass_configs=PASS_CONFIGS)
