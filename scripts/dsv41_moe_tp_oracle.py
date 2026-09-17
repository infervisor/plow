#!/usr/bin/env python3
"""dsv41_moe_tp_oracle.py — what the routed combine's `shared` operand has to be.

`d_moe_combine_pf` computes

    out[t] = residual[t] + shared[t] + SUM_slot part[t*k + slot]

and the band all-reduce that FOLLOWS it sums `out` across all `tp` ranks. So the `shared`
it is handed decides whether the shared expert is counted once or `tp` times, and NOTHING
about the shapes distinguishes the two: both are [T, hidden] bf16, both are finite, and the
model keeps producing plausible activations either way.

This prices the difference. It needs no checkpoint -- it is a dataflow identity, not a
weight check.

    PYTHONPATH=/workspace/oracle-venv/site python3 scripts/dsv41_moe_tp_oracle.py
"""
import torch

torch.manual_seed(0)
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


T, H, I, E, K, TP = 8, 256, 512, 16, 6, 8

x = torch.randn(T, H)
# shared expert: H -> I -> H, the second matmul INPUT-parallel (row-parallel) under TP
w_sh_up = torch.randn(H, I) / H**0.5
w_sh_dn = torch.randn(I, H) / I**0.5
# routed experts, same structure
w_up = torch.randn(E, H, I) / H**0.5
w_dn = torch.randn(E, I, H) / I**0.5

gate = torch.softmax(torch.randn(T, E), dim=-1)
topv, topi = gate.topk(K, dim=-1)

# ---- the reference: shared + routed, summed once ------------------------------------------
shared_ref = torch.relu(x @ w_sh_up) @ w_sh_dn
routed_ref = torch.zeros(T, H)
for t in range(T):
    for s in range(K):
        e = topi[t, s].item()
        routed_ref[t] += topv[t, s] * (torch.relu(x[t] @ w_up[e]) @ w_dn[e])
y_ref = shared_ref + routed_ref

# ---- the TP8 decomposition: `down` is row-parallel, so every rank holds a PARTIAL ----------
sl = I // TP
shared_partial = []
routed_partial = []
for r in range(TP):
    lo, hi = r * sl, (r + 1) * sl
    shared_partial.append(torch.relu(x @ w_sh_up)[:, lo:hi] @ w_sh_dn[lo:hi])
    rp = torch.zeros(T, H)
    for t in range(T):
        for s in range(K):
            e = topi[t, s].item()
            rp[t] += topv[t, s] * (torch.relu(x[t] @ w_up[e])[lo:hi] @ w_dn[e][lo:hi])
    routed_partial.append(rp)

shared_full = sum(shared_partial)  # what a separate all-reduce produces on EVERY rank
check(
    "the row-parallel split reproduces the shared expert exactly",
    torch.allclose(shared_full, shared_ref, atol=1e-4),
    f"max|err| = {(shared_full - shared_ref).abs().max():.3e}",
)

# ---- the emit as it was: combine takes the ALREADY-REDUCED shared, then reduces again ------
out_old = [shared_full + routed_partial[r] for r in range(TP)]
y_old = sum(out_old)

# ---- the emit as it is now: combine takes the PARTIAL, one reduce total --------------------
out_new = [shared_partial[r] + routed_partial[r] for r in range(TP)]
y_new = sum(out_new)

check(
    "handing the combine the PARTIAL reproduces the reference",
    torch.allclose(y_new, y_ref, atol=1e-4),
    f"max|err| = {(y_new - y_ref).abs().max():.3e}",
)
check(
    "handing it the REDUCED buffer does not",
    not torch.allclose(y_old, y_ref, atol=1e-4),
    f"max|err| = {(y_old - y_ref).abs().max():.3e}",
)
check(
    "and the error is exactly (tp - 1) copies of the shared expert",
    torch.allclose(y_old - y_ref, (TP - 1) * shared_ref, atol=1e-4),
    f"max|err| = {((y_old - y_ref) - (TP - 1) * shared_ref).abs().max():.3e}",
)

rel = (y_old - y_ref).norm() / y_ref.norm()
check(
    "it is not a rounding detail",
    rel > 0.5,
    f"relative error {rel:.3f} of the layer's FFN output",
)

# ---- and the shapes agree throughout, which is why nothing caught it ----------------------
check(
    "every buffer is the same shape either way, and both are finite",
    y_old.shape == y_new.shape == y_ref.shape
    and torch.isfinite(y_old).all()
    and torch.isfinite(y_new).all(),
    f"{tuple(y_ref.shape)}",
)

print()
print(f"[1] shared expert, once   : |y_ref|   = {y_ref.norm():.4f}")
print(f"[2] shared expert, tp x   : |y_old|   = {y_old.norm():.4f}")
print(f"[3] relative error        : {rel:.4f}")
raise SystemExit(1 if FAIL else 0)
