"""Short-query FP32 causal stream attention for gen_search; no precision conversion."""
import triton
import triton.language as tl


@triton.jit
def vector_attention(qkv, positions, lengths, out, T: tl.constexpr, L: tl.constexpr,
                     H: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
                     EDGE: tl.constexpr, NK: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    pos = tl.load(positions + b)
    n = tl.load(lengths + b) if EDGE else L + T
    j = tl.arange(0, NK)
    d = tl.arange(0, D)
    query = tl.load(qkv + ((b * (L + T) + L + r) * (3 * H) + h) * D + d)
    valid = (j >= L - tl.minimum(pos, L)) & (j < n) & (j <= L + r)
    valid = valid & (j > L + r - W)
    base = qkv + ((b * (L + T) + j[:, None]) * (3 * H) + H + h) * D + d[None, :]
    k = tl.load(base, mask=valid[:, None], other=0.)
    scores = tl.sum(k * query[None, :], 1) * (D ** -0.5)
    scores = tl.where(valid, scores, float("-inf"))
    maximum = tl.max(scores, 0)
    maximum = tl.where(maximum == float("-inf"), 0., maximum)
    p = tl.exp(scores - maximum)
    total = tl.sum(p, 0)
    v = tl.load(base + H * D, mask=valid[:, None], other=0.)
    output = tl.sum(v * p[:, None], 0) / tl.maximum(total, 1.e-30)
    tl.store(out + ((b * T + r) * H + h) * D + d, output)


def launcher(cfg, c, inp, out):
    return lambda: vector_attention[(c["T"], c["H"], c["B"])](
        inp["qkv"], inp["positions"], inp["lengths"], out,
        T=c["T"], L=c["L"], H=c["H"], D=c["D"], W=c["W"],
        EDGE=c.get("edge", False), NK=triton.next_power_of_2(c["L"] + c["T"]),
        num_warps=cfg["warps"], num_stages=1)
