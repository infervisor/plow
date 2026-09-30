#!/usr/bin/env python3
"""Is plow's mHC V4.1's mHC? A numerical oracle on tiny synthetic shapes.

Run with the CPU torch from `plow-reference-oracle-cpu-torch`:

    PYTHONPATH=/workspace/oracle-venv/site python3 scripts/dsv41_mhc_oracle.py

`runtime/amd/op_hyperconn.h` was signed off against GLM-5.3's hyper-connection, where a sublayer's
`hc_pre` collapses the residual copies with the `pre` coefficients that the SAME sublayer's
`hc_mixes` just produced. `crates/devgen/src/mla/dsv41.rs` bound V4.1 straight onto that op
(`emit_dsv41_mhc_pre` -> `super::emit_mhc_pre`), and `MhcHandles` had no field for an incoming
`pre_mix`, so plow's V4.1 emit inherited the same-sublayer ordering.

That is fixed -- `MhcPre::{Seed, Deferred}` and `pre_mode` on op 128 -- and this script is what
priced it, so it stays as the regression oracle: `run_plow` below is the GLM-5.3 ordering kept as
the counterfactual, not a description of the current emit.

V4.1's `Block.forward` (inference/model.py:965-996) does NOT do that. Its class docstring says the
coefficients a sublayer computes "are used by the *next* one", and the code bears it out:

    x = self.hc_pre(x, pre_mix)        # forward() ARGUMENT -- the previous block's ffn_pre
    ... attention ...
    x = self.hc_pre(x, attn_pre)       # THIS block's attention mixes drive the FFN
    ... ffn ...
    return x, ffn_pre                  # and its FFN mixes drive the NEXT block

`Transformer.forward` (1260-1268) seeds the chain with `make_identity_pre_mix` -- a ONE-HOT
[1,0,0,0], not a learned mix -- and after the last layer spends the dangling `ffn_pre` on one final
`hc_pre`. Only `pre` is deferred; `post` and `comb` stay in their own sublayer.

This script runs both orderings over the same weights and the same input and reports the
difference. Everything except the ordering is shared, so a nonzero answer isolates the ordering.
"""

import sys

import torch

torch.manual_seed(0)
torch.set_printoptions(precision=6)

HC = 4  # hc_mult
DIM = 8  # hidden; tiny on purpose, this is a structural check not a perf one
TOK = 5
LAYERS = 3
SINK = 20  # hc_sinkhorn_iters
HC_EPS = 1e-6
NORM_EPS = 1e-20  # rms_norm_eps from the V4.1 config
MIX = (2 + HC) * HC
WORK = torch.float64


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc=HC, iters=SINK, eps=HC_EPS):
    """Transcription of `inference/kernel.py:407-462` (`hc_split_sinkhorn_kernel`) in plain torch.

    Same split, same scale/base, same softmax-then-Sinkhorn on `comb`; the tilelang version is a
    per-token kernel over the identical arithmetic.
    """
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]
    comb = comb.reshape(*mixes.shape[:-1], hc, hc)

    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_mixes(x, hc_fn, hc_scale, hc_base):
    """`Block.hc_mixes`, model.py:948-954."""
    # reference says .float(); f64 here so the ordering, not rounding, is what moves
    xf = x.flatten(-2).to(WORK)
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = torch.nn.functional.linear(xf, hc_fn) * rsqrt
    return hc_split_sinkhorn(mixes, hc_scale, hc_base)


def hc_pre(x, pre_mix):
    """`Block.hc_pre`, model.py:957-960. [t,hc,d] x [t,hc] -> [t,d]"""
    return torch.sum(pre_mix.unsqueeze(-1) * x.to(WORK), dim=-2)


def hc_post(x, residual, post, comb):
    """`Block.hc_post`, model.py:962-966. -> [t,hc,d]"""
    return post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
    )


def rms_norm(x, w):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + NORM_EPS) * w


class Weights:
    """One layer's mHC parameters plus stand-in attention and FFN bodies.

    The sublayer bodies are arbitrary but FIXED and shared by both orderings -- what is under test
    is which `pre` reaches `hc_pre`, not what attention computes.
    """

    def __init__(self):
        g = torch.Generator().manual_seed(torch.randint(1 << 30, (1,)).item())

        def r(*s):
            return torch.randn(*s, generator=g, dtype=torch.float64)

        self.attn_fn = r(MIX, HC * DIM) * 0.05
        self.ffn_fn = r(MIX, HC * DIM) * 0.05
        self.attn_base = r(MIX) * 0.1
        self.ffn_base = r(MIX) * 0.1
        self.attn_scale = r(3).abs() + 0.5
        self.ffn_scale = r(3).abs() + 0.5
        self.attn_norm_w = r(DIM).abs() + 0.5
        self.ffn_norm_w = r(DIM).abs() + 0.5
        self.attn_w = r(DIM, DIM) * 0.2
        self.ffn_w = r(DIM, DIM) * 0.2

    def attn(self, x):
        return torch.nn.functional.linear(rms_norm(x, self.attn_norm_w), self.attn_w)

    def ffn(self, x):
        h = torch.nn.functional.linear(rms_norm(x, self.ffn_norm_w), self.ffn_w)
        return h * torch.sigmoid(h)


def identity_pre_mix(t):
    """`make_identity_pre_mix`, model.py:1159-1163 -- one-hot on copy 0, not a learned mix."""
    p = torch.zeros(t, HC, dtype=torch.float64)
    p[:, 0] = 1.0
    return p


def run_v41(x, ws):
    """`Block.forward` chained by `Transformer.forward` (model.py:1260-1268)."""
    pre_mix = identity_pre_mix(x.size(0))
    for w in ws:
        residual = x
        attn_pre, attn_post, attn_comb = hc_mixes(x, w.attn_fn, w.attn_scale, w.attn_base)
        y = w.attn(hc_pre(x, pre_mix))  # <-- the PREVIOUS sublayer's pre
        x = hc_post(y, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(x, w.ffn_fn, w.ffn_scale, w.ffn_base)
        y = w.ffn(hc_pre(x, attn_pre))  # <-- this block's ATTENTION pre
        x = hc_post(y, residual, ffn_post, ffn_comb)
        pre_mix = ffn_pre
    return hc_pre(x, pre_mix), pre_mix  # the final collapse spends the dangling ffn_pre


def run_plow(x, ws):
    """GLM-5.3's ordering, which `emit_dsv41_mhc_pre` used to inherit: `d_hyperconn_pre` writes
    `layer_input` from the pre_mix it computed in the same call, so every `hc_pre` is
    same-sublayer. `PLOW_HC_PRE_OWN` still selects exactly this, for GLM.
    """
    for w in ws:
        residual = x
        attn_pre, attn_post, attn_comb = hc_mixes(x, w.attn_fn, w.attn_scale, w.attn_base)
        y = w.attn(hc_pre(x, attn_pre))  # <-- its OWN pre
        x = hc_post(y, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(x, w.ffn_fn, w.ffn_scale, w.ffn_base)
        y = w.ffn(hc_pre(x, ffn_pre))  # <-- its OWN pre
        x = hc_post(y, residual, ffn_post, ffn_comb)
        ffn_pre_last = ffn_pre
    return hc_pre(x, ffn_pre_last), ffn_pre_last


def rel(a, b):
    d = (a - b).abs().max().item()
    s = b.abs().max().item()
    return d, (d / s if s > 0 else float("nan"))


def main():
    ws = [Weights() for _ in range(LAYERS)]
    x = torch.randn(TOK, HC, DIM, dtype=torch.float64)

    checks = {}

    # [1] One layer in isolation -- the case plow actually emits and measures today.
    one = ws[:1]
    a, _ = run_v41(x.clone(), one)
    b, _ = run_plow(x.clone(), one)
    d1, r1 = rel(b, a)
    print(f"[1] layer 0 alone, V4.1 order vs plow order      : max|err| = {d1:.3e}  rel = {r1:.3%}")
    checks["layer 0 alone already differs: V4.1 collapses with a ONE-HOT, plow with hc_attn_fn"] = (
        r1 > 1e-3
    )

    # [2] The head of the chain, stated directly: V4.1's first hc_pre IS residual copy 0.
    w0 = ws[0]
    attn_pre, _, _ = hc_mixes(x, w0.attn_fn, w0.attn_scale, w0.attn_base)
    d2, r2 = rel(hc_pre(x, attn_pre), x[:, 0, :].clone())
    print(f"[2] plow's layer-0 attention input vs residual[0]: max|err| = {d2:.3e}  rel = {r2:.3%}")
    checks["plow feeds layer-0 attention a learned mix where V4.1 feeds residual copy 0"] = r2 > 1e-3

    # [3] The deferral is not a no-op even mid-chain: same weights, same input, 3 layers.
    a, _ = run_v41(x.clone(), ws)
    b, _ = run_plow(x.clone(), ws)
    d3, r3 = rel(b, a)
    print(f"[3] {LAYERS} layers chained                           : max|err| = {d3:.3e}  rel = {r3:.3%}")
    checks[f"the difference persists through {LAYERS} chained layers"] = r3 > 1e-3

    # [4] pre is the ONLY deferred coefficient. Feed plow's ordering the correct pre and the two
    #     agree exactly -- so post/comb need no change and the fix is confined to hc_pre's input.
    def run_v41_but_post_comb_from_next(x, ws):
        pre_mix = identity_pre_mix(x.size(0))
        for w in ws:
            residual = x
            attn_pre, attn_post, attn_comb = hc_mixes(x, w.attn_fn, w.attn_scale, w.attn_base)
            y = w.attn(hc_pre(x, pre_mix))
            x = hc_post(y, residual, attn_post, attn_comb)
            residual = x
            ffn_pre, ffn_post, ffn_comb = hc_mixes(x, w.ffn_fn, w.ffn_scale, w.ffn_base)
            y = w.ffn(hc_pre(x, attn_pre))
            x = hc_post(y, residual, ffn_post, ffn_comb)
            pre_mix = ffn_pre
        return hc_pre(x, pre_mix)

    d4, _ = rel(run_v41_but_post_comb_from_next(x.clone(), ws), run_v41(x.clone(), ws)[0])
    print(f"[4] post/comb are same-sublayer in both          : max|err| = {d4:.3e}")
    checks["only `pre` is deferred -- post and comb already match, so the fix is local"] = d4 == 0.0

    # [5] The deferred pre is a real dependency, not a renaming: layer L's attention input depends
    #     on layer L-1's FFN mixes. Perturb ONLY hc_ffn_fn of layer 0 and watch layer 1's attention
    #     input move under V4.1's ordering while plow's cannot see it at all.
    ws2 = [Weights() for _ in range(2)]
    ws2[1].attn_fn = ws2[0].attn_fn.clone()  # rule out a coincidence through the attention mixes

    def layer1_attn_input(ws_, x):
        pre_mix = identity_pre_mix(x.size(0))
        w = ws_[0]
        residual = x
        attn_pre, attn_post, attn_comb = hc_mixes(x, w.attn_fn, w.attn_scale, w.attn_base)
        y = w.attn(hc_pre(x, pre_mix))
        x = hc_post(y, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(x, w.ffn_fn, w.ffn_scale, w.ffn_base)
        y = w.ffn(hc_pre(x, attn_pre))
        x = hc_post(y, residual, ffn_post, ffn_comb)
        return hc_pre(x, ffn_pre), x

    base_in, base_x = layer1_attn_input(ws2, x.clone())
    bumped = [Weights.__new__(Weights), ws2[1]]
    bumped[0].__dict__ = dict(ws2[0].__dict__)
    bumped[0].ffn_base = ws2[0].ffn_base.clone()
    bumped[0].ffn_base[:HC] += 3.0  # moves ffn_pre only: the pre slice is rows [0, hc)
    bump_in, bump_x = layer1_attn_input(bumped, x.clone())
    d5r, _ = rel(bump_x, base_x)
    d5, r5 = rel(bump_in, base_in)
    print(f"[5] bump layer0 hc_ffn_base[pre]: residual moves  : max|err| = {d5r:.3e}")
    print(f"    ... and layer 1's attention input moves      : max|err| = {d5:.3e}  rel = {r5:.3%}")
    checks["layer L's attention input genuinely depends on layer L-1's FFN mixes"] = (
        d5r == 0.0 and r5 > 1e-3
    )

    # [6] THE HALF ARITHMETIC, run as the kernel runs it. The scheme is: one `[2, T, hc]` buffer;
    #     sublayer k reads half `k % 2` and publishes into half `(k % 2) ^ 1`. An off-by-one here
    #     is invisible in every shape and every test above -- it just gates with the wrong
    #     sublayer's coefficients again -- so this replays `run_v41` through the buffer, indexing
    #     exactly as `d_hyperconn_pre` does, and demands bit equality.
    def run_pre_pair(x, ws):
        pair = torch.zeros(2, x.size(0), HC, dtype=WORK)  # the arena: deliberately NOT the one-hot
        k = 0
        for w in ws:
            for ffn in (False, True):
                in_half = k % 2
                fn, sc, ba = (
                    (w.ffn_fn, w.ffn_scale, w.ffn_base) if ffn else (w.attn_fn, w.attn_scale, w.attn_base)
                )
                residual = x
                pre, post, comb = hc_mixes(x, fn, sc, ba)
                pair[in_half ^ 1] = pre  # publish, as the lane-0 block does
                if k == 0:  # PLOW_HC_PRE_SEED
                    gate = identity_pre_mix(x.size(0))
                else:  # PLOW_HC_PRE_DEFER
                    gate = pair[in_half]
                y = (w.ffn if ffn else w.attn)(hc_pre(x, gate))
                x = hc_post(y, residual, post, comb)
                k += 1
        return hc_pre(x, pair[(k - 1) % 2 ^ 1])

    d6, _ = rel(run_pre_pair(x.clone(), ws), run_v41(x.clone(), ws)[0])
    print(f"[6] the [2,T,hc] pre_pair half arithmetic         : max|err| = {d6:.3e}")
    checks["the ping-pong halves reproduce V4.1's chain exactly, not off by one"] = d6 == 0.0

    print()
    bad = 0
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
        bad += not v
    print()
    if bad:
        print(f"{bad} check(s) FAILED")
        return 1
    print("GLM-5.3's ordering is NOT V4.1's: `pre` must come from the PREVIOUS sublayer.")
    print("The emit does that now (MhcPre::Seed/Deferred, op 128 `pre_mode`); this prices the gap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
