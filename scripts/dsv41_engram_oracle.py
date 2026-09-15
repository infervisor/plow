#!/usr/bin/env python3
"""Are ops 182/183 V4.1's Engram? A numerical oracle on tiny synthetic shapes.

    PYTHONPATH=/workspace/oracle-venv/site python3 scripts/dsv41_engram_oracle.py

Engram is next on the emit list -- layer 1 is the cheapest remaining layer because its
`compress_ratio` is 0, so Engram is the ONLY thing it is missing. Three times now a V4.1 subsystem
has turned out to be a rearrangement of a shipped one with identical tensor shapes and different
structure (the CSA2 read side, op 180's missing tap, the cross-sublayer mHC), each marked Done on
a reading. So the device side gets measured against the reference before anything is built on it.

`run_ref_*` are transcriptions of `inference/model.py` (`ParallelEngramEmbedding.forward` 312-325,
`Engram.forward` 350-365). `run_op_*` are transcriptions of `runtime/amd/op_engram.h`
(`d_engram_embed`, `d_engram_gate`). The two were written from the two sources separately; a shared
mistake would have to be made twice.
"""

import sys

import torch

torch.manual_seed(0)

HC = 4
DIM = 16  # args.dim; tiny, this is a structural check
HEAD_DIM = 32  # layout.head_dim (256 in the checkpoint)
BLK = 8  # fp8_block_size (32 in the checkpoint: embed.scale is [rows, 256/32] = [rows, 8])
NCOLS = 6  # (max_ngram_size - 1) * n_heads (24 in the checkpoint)
ROWS = 40  # table rows
TOK = 5
EPS = 1e-20  # args.norm_eps
CLAMP = 1e-6  # Engram.clamp_value
WORK = torch.float64


def fp8_e4m3(x):
    """Round-trip through e4m3 so both sides see the same stored values."""
    return x.to(torch.float8_e4m3fn).float()


def e8m0(x):
    """ue8m0: a power of two, stored as a biased exponent byte. Both sides see the same scales."""
    return torch.exp2(torch.round(torch.log2(x.clamp_min(1e-30))))


# ---------------------------------------------------------------- op 183 / the embed


def run_ref_embed(ids, table, scale, vocab_start, part_rows):
    """`ParallelEngramEmbedding.forward`, model.py:312-325."""
    mask = (ids < vocab_start) | (ids >= vocab_start + part_rows)
    local = (ids - vocab_start).masked_fill(mask, 0)
    values = table[local]
    scales = scale[local]
    values = values.to(WORK).unflatten(-1, (-1, BLK)) * scales.to(WORK).unsqueeze(-1)
    values = values.flatten(-2)
    return values.masked_fill(mask.unsqueeze(-1), 0)


def run_op_embed(ids, table, scale, vocab_start, part_rows):
    """`d_engram_embed`, op_engram.h. One flat `n_cols * head_dim` row per token, which is what the
    `wkv` GEMM consumes -- so this returns the flattened form and the reference is flattened to
    match. `spr = head_dim / blk`; an id outside the shard writes ZEROS."""
    spr = HEAD_DIM // BLK
    t_n, n_cols = ids.shape
    out = torch.zeros(t_n, n_cols * HEAD_DIM, dtype=WORK)
    flat_scale = scale.reshape(-1)
    flat_table = table.reshape(-1)
    for t in range(t_n):
        for i in range(n_cols * HEAD_DIM):
            col, d = i // HEAD_DIM, i % HEAD_DIM
            local = int(ids[t, col]) - vocab_start  # signed, compared as signed
            if local < 0 or local >= part_rows:
                out[t, i] = 0.0
                continue
            v = flat_table[local * HEAD_DIM + d]
            sc = flat_scale[local * spr + d // BLK]
            out[t, i] = v.to(WORK) * sc.to(WORK)
    return out


# ---------------------------------------------------------------- op 182 / the gate


def run_ref_gate(x, kv, qw, kw, tmask):
    """`Engram.forward`, model.py:350-365, from `kv` on (the projection is an ordinary GEMM)."""
    key, value = kv.split([HC * DIM, DIM], dim=-1)
    key = key.to(WORK).unflatten(-1, (HC, DIM))
    weight = qw.to(WORK) * kw.to(WORK)  # only ever used as a product
    h = x.to(WORK)
    rstd = torch.rsqrt(h.square().mean(-1) + EPS) * torch.rsqrt(key.square().mean(-1) + EPS)
    dot = (h * weight * key).sum(-1) * rstd * DIM**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(CLAMP).sqrt(), dot))
    if tmask is not None:
        gate = gate.masked_fill(~tmask.unsqueeze(-1), 0)
    return h + gate.unsqueeze(-1) * value.to(WORK).unsqueeze(-2), gate


def run_op_gate(x, kv, qw, kw, tmask):
    """`d_engram_gate`, op_engram.h: one workgroup per token, three reductions per hc copy over
    `dim`, in place on `x`. A masked token passes through UNTOUCHED -- gate 0, not value 0."""
    t_n = x.shape[0]
    out = x.to(WORK).clone()
    gates = torch.zeros(t_n, HC, dtype=WORK)
    inv_dim = 1.0 / DIM
    dim_rsqrt = DIM**-0.5
    for t in range(t_n):
        krow = kv[t]
        vrow = krow[HC * DIM :]  # the shared value, after the n keys
        masked = tmask is not None and not bool(tmask[t])
        for c in range(HC):
            hc_ = out[t, c]
            kc = krow[c * DIM : (c + 1) * DIM].to(WORK)
            qc = qw[c].to(WORK)
            wc = kw[c].to(WORK)
            ssh = (hc_ * hc_).sum()
            ssk = (kc * kc).sum()
            dot = (hc_ * (qc * wc) * kc).sum()
            if masked:
                gate = torch.zeros((), dtype=WORK)
            else:
                rstd = torch.rsqrt(ssh * inv_dim + EPS) * torch.rsqrt(ssk * inv_dim + EPS)
                dv = dot * rstd * dim_rsqrt
                mag = torch.sqrt(torch.maximum(dv.abs(), torch.tensor(CLAMP, dtype=WORK)))
                gate = torch.sigmoid(torch.copysign(mag, dv))
            gates[t, c] = gate
            if gate != 0.0:
                out[t, c] = hc_ + gate * vrow.to(WORK)
    return out, gates


def rel(a, b):
    d = (a - b).abs().max().item()
    s = b.abs().max().item()
    return d, (d / s if s > 0 else float("nan"))


def main():
    checks = {}

    # ---- op 183 ----------------------------------------------------------
    part_rows, vocab_start = ROWS, 7  # rank 1 of a sharded table
    table = fp8_e4m3(torch.randn(part_rows, HEAD_DIM) * 4).to(torch.float8_e4m3fn)
    scale = e8m0(torch.rand(part_rows, HEAD_DIM // BLK) * 3 + 0.1)
    ids = torch.randint(vocab_start, vocab_start + part_rows, (TOK, NCOLS))

    a = run_ref_embed(ids, table.float(), scale, vocab_start, part_rows).flatten(1)
    b = run_op_embed(ids, table.float(), scale, vocab_start, part_rows)
    d1, _ = rel(b, a)
    print(f"[1] op 183 vs ParallelEngramEmbedding, all in shard : max|err| = {d1:.3e}")
    checks["op 183's dequant and its flat [n_cols*head_dim] layout are the reference's"] = d1 == 0.0

    # Ids outside the shard. The kernel compares SIGNED, so an id below vocab_start must fall out
    # rather than wrapping to a huge in-range-looking index.
    ids2 = ids.clone()
    ids2[0, 0] = vocab_start - 3  # below
    ids2[1, 1] = vocab_start + part_rows + 5  # above
    a = run_ref_embed(ids2, table.float(), scale, vocab_start, part_rows).flatten(1)
    b = run_op_embed(ids2, table.float(), scale, vocab_start, part_rows)
    d2, _ = rel(b, a)
    z = b[0, :HEAD_DIM].abs().max().item() + b[1, HEAD_DIM : 2 * HEAD_DIM].abs().max().item()
    print(f"[2] ... with ids below AND above the shard         : max|err| = {d2:.3e}, out-of-shard = {z:.3e}")
    checks["an id on either side of the shard writes zeros, not a wrapped row"] = d2 == 0.0 and z == 0.0

    # ---- op 182 ----------------------------------------------------------
    x = torch.randn(TOK, HC, DIM)
    kv = torch.randn(TOK, (HC + 1) * DIM)
    qw = torch.randn(HC, DIM)
    kw = torch.randn(HC, DIM)

    a, ga = run_ref_gate(x, kv, qw, kw, None)
    b, gb = run_op_gate(x, kv, qw, kw, None)
    d3, _ = rel(b, a)
    print(f"[3] op 182 vs Engram.forward, no mask              : max|err| = {d3:.3e}")
    checks["op 182's gate and mix are Engram.forward's"] = d3 < 1e-12

    tmask = torch.tensor([True, False, True, True, False])
    a, ga = run_ref_gate(x, kv, qw, kw, tmask)
    b, gb = run_op_gate(x, kv, qw, kw, tmask)
    d4, _ = rel(b, a)
    untouched = (b[1] - x[1].to(WORK)).abs().max().item()
    print(f"[4] ... with a token_mask                          : max|err| = {d4:.3e}")
    print(f"    a masked token passes through UNTOUCHED        : max|err| = {untouched:.3e}")
    checks["a masked token is left alone -- gate 0, not value 0"] = d4 < 1e-12 and untouched == 0.0

    # ---- the two details the op's header says are not details ------------

    # THE SIGNED SQRT. The clamp sits INSIDE it, on the magnitude, so dot = 0 gives sqrt(1e-6) with
    # the sign of zero -- sigmoid(+0.001), not sigmoid(0) and not 0. Dropping the sqrt entirely
    # (a plain sigmoid(dot)) is a different gate everywhere, not just near zero.
    dots = torch.linspace(-8.0, 8.0, 3201, dtype=WORK)  # swept, so the peak is found not guessed
    signed = torch.sigmoid(torch.copysign(dots.abs().clamp_min(CLAMP).sqrt(), dots))
    plain = torch.sigmoid(dots)
    gap = (signed - plain).abs()
    d5 = gap.max().item()
    at = dots[gap.argmax()].item()
    at_zero = torch.sigmoid(
        torch.copysign(torch.tensor(0.0, dtype=WORK).abs().clamp_min(CLAMP).sqrt(),
                       torch.tensor(0.0, dtype=WORK))
    ).item()
    print(f"[5] signed-sqrt gate vs a plain sigmoid(dot)       : max|err| = {d5:.3e} at dot = {at:+.3f}")
    print(f"    the gate at dot = 0 (the clamp, sqrt'd)        : {at_zero:.9f}")
    # The two agree exactly at dot in {0 (up to the clamp), 1, -1} and nowhere else -- a gate that
    # is 10 points of probability different across the working range is a different model, so 0.05
    # is a floor on "materially", not a tuned threshold.
    checks["the signed sqrt is the gate's shape, not a guard that can be dropped"] = d5 > 0.05

    # PER-COPY NORMALIZATION. `rstd` is computed per (token, hc copy) over `dim`, NOT jointly over
    # the copies. Joint normalization compiles and runs and is a different model.
    h = x.to(WORK)
    key = kv[:, : HC * DIM].to(WORK).unflatten(-1, (HC, DIM))
    per = torch.rsqrt(h.square().mean(-1) + EPS) * torch.rsqrt(key.square().mean(-1) + EPS)
    joint = (
        torch.rsqrt(h.flatten(1).square().mean(-1) + EPS)
        * torch.rsqrt(key.flatten(1).square().mean(-1) + EPS)
    ).unsqueeze(-1).expand_as(per)
    d6, r6 = rel(joint, per)
    print(f"[6] joint normalization vs per-(token, hc copy)    : max|err| = {d6:.3e}  rel = {r6:.3%}")
    checks["normalizing jointly over the copies is a different model"] = r6 > 1e-3

    print()
    bad = 0
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
        bad += not v
    print()
    if bad:
        print(f"{bad} check(s) FAILED")
        return 1
    print("Ops 182 and 183 ARE V4.1's Engram. The gap for layers 1 and 14 is the EMIT, not the kernels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
