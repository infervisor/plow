"""Triton attention for the S3Gen CFM (cached-prompt `csynth`): the op of attn_cfm.py (operands
and layout as documented there), one program per (BM-row query tile, head, item), key blocks of
BN rows: the item's prefix block first, then its own rows. The f32 products run at
`input_precision` PREC: "tf32x3" is the hi/lo split of the plow kernel (three tensor-core
products), "tf32" one truncated product, "ieee" FP32 FMA.
"""
import triton
import triton.language as tl


@triton.jit
def attn_cfm(qkv, prefix, pidx, klen, out, scale, Tq,
             PRE: tl.constexpr, H: tl.constexpr, HW: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
    mt = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    n = tl.load(klen + b)
    pb = tl.load(pidx + b)
    sl = scale * 1.4426950408889634
    rows = mt * BM + tl.arange(0, BM)
    cols = tl.arange(0, HW)
    qmask = rows < Tq
    q = tl.load(qkv + (b * Tq + rows)[:, None] * (3 * H * HW) + h * HW + cols[None, :],
                mask=qmask[:, None], other=0.0)
    m = tl.full([BM], float("-inf"), tl.float32)
    l = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HW], tl.float32)
    # The first prefix block always holds a visible key (n >= 1), so m is finite before any
    # fully masked block and exp2(m - m) stays 1.
    npre = tl.minimum(n, PRE)
    for start in range(0, PRE, BN):
        kj = start + tl.arange(0, BN)
        kmask = kj < npre
        base = prefix + (pb * PRE + kj)[:, None] * (2 * H * HW) + h * HW + cols[None, :]
        k = tl.load(base, mask=kmask[:, None], other=0.0)
        v = tl.load(base + H * HW, mask=kmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=PREC) * sl
        s = tl.where(kmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(s - m_new[:, None])
        l = l * alpha + tl.sum(p, 1)
        acc = tl.dot(p, v, acc * alpha[:, None], input_precision=PREC)
        m = m_new
    nown = n - PRE
    for start in range(0, Tq, BN):
        kj = start + tl.arange(0, BN)
        kmask = kj < nown
        base = qkv + (b * Tq + kj)[:, None] * (3 * H * HW) + (H + h) * HW + cols[None, :]
        k = tl.load(base, mask=kmask[:, None], other=0.0)
        v = tl.load(base + H * HW, mask=kmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=PREC) * sl
        s = tl.where(kmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(s - m_new[:, None])
        l = l * alpha + tl.sum(p, 1)
        acc = tl.dot(p, v, acc * alpha[:, None], input_precision=PREC)
        m = m_new
    tl.store(out + (b * Tq + rows)[:, None] * (H * HW) + h * HW + cols[None, :],
             acc / l[:, None], mask=qmask[:, None])


def launcher(bm, bn, num_warps, num_stages, prec, H=8, HW=64, pre=306):
    """launch(qkv, prefix, pidx, klen, out, scale): compiles on the first call."""
    def launch(qkv, prefix, pidx, klen, out, scale):
        B, Tq, _ = qkv.shape
        attn_cfm[(triton.cdiv(Tq, bm), H, B)](qkv, prefix, pidx, klen, out, scale, Tq,
                                                PRE=pre, H=H, HW=HW, BM=bm, BN=bn, PREC=prec,
                                                num_warps=num_warps, num_stages=num_stages)
    return launch
