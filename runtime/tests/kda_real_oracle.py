#!/usr/bin/env python3
# kda_real_oracle.py — REAL-WEIGHT reference for ONE Kimi-K3 KDA layer.            [K3-KDA-GATE]
#
# The analogue of glm52_real_oracle.py, for the mixer in 69 of K3's 93 layers. It loads ONE real
# KDA layer out of the 1.5 TB checkpoint (mmap, header-driven, never the whole file), computes
# every stage of the layer in fp32 against the REFERENCE implementation, and writes a fixture that
# runtime/tests/kda_block_gfx950_test.c diffs plow against stage by stage.
#
# ---------------------------------------------------------------------------------------------
# WHAT THE REFERENCE IS, AND WHY IT IS NOT transformers
#
# The reference is `fla`'s own pure-torch `naive_recurrent_kda` and `naive_kda_lowerbound_gate`,
# VENDORED VERBATIM below. They are the functions fla's test suite gates its own Triton kernels
# against, they need no triton and no GPU, and they run on CPU in fp32.
#
# The HF-shipped `modeling_kimi_linear.py` is NOT usable here. It declares `A_log` with
# `num_heads = 96` entries (`:520-521`) against the checkpoint's `[128]`, so it SIZE-MISMATCHES ON
# LOAD. The shipped reference is broken as-is; "it loads in transformers" is not available as a
# correctness signal for K3, and neither is any released vLLM (0.23.0 and main implement
# Kimi-Linear, with the low-rank output gate and no `gate_lower_bound` — the two knobs K3 changed).
#
# ---------------------------------------------------------------------------------------------
# STATE DTYPE: f32, settled from code.
#
# AMD's day-0 post says "FP32 KDA SSM states" in prose and uses 2 bytes per element in its own
# formula; the two halves of that document disagree. The reference does not:
# `fla/ops/kda/fused_recurrent.py` allocates the final state `dtype=torch.float32` in BOTH layout
# branches and accumulates in `tl.float32`; `naive_recurrent_kda` casts every input `.to(float)`.
# This fixture is f32 throughout for the state and the gate.
#
# ---------------------------------------------------------------------------------------------
# INPUT is a seeded synthetic hidden state; the WEIGHTS are real. We have no real layer-0
# activations without running the embedding, and the de-risk here is the recurrence + the gate +
# the V-first layout, not prompt-specific activations. Documented, not hidden — same posture as
# the GLM B4 oracle.
#
#   K3_DIR=<snapshot>   KDA_LAYER=<0-based layer>   KDA_T=<tokens>
#   python3 kda_real_oracle.py <out.bin>

import json
import os
import struct
import sys
import mmap

import numpy as np
import torch

MAGIC = 0x4B444131  # "KDA1"

SNAP = os.environ.get(
    "K3_DIR",
    "/home/lava/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
)
LAYER = int(os.environ.get("KDA_LAYER", "0"))
T = int(os.environ.get("KDA_T", "4"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "kda_fixture.bin"

torch.manual_seed(0xB4)
np.random.seed(0xB4)

# ---------------------------------------------------------------------------------------------
# safetensors, header-driven and mmap'd. Same skeleton as glm52_real_oracle.py.

_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
_MM = {}


def _mm(path):
    if path not in _MM:
        f = open(path, "rb")
        _MM[path] = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    return _MM[path]


def _index(snap):
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    idx, hdrs = {}, {}
    for name, shard in wm.items():
        if shard not in hdrs:
            p = os.path.join(snap, shard)
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdrs[shard] = (json.loads(fh.read(n)), 8 + n, p)
        idx[name] = shard
    return idx, hdrs


IDX, HDRS = _index(SNAP)


def load(name):
    h, base, path = HDRS[IDX[name]]
    m = h[name]
    lo, hi = m["data_offsets"]
    buf = _mm(path)[base + lo : base + hi]
    return torch.frombuffer(bytearray(buf), dtype=_DT[m["dtype"]]).view(*m["shape"])


# ---------------------------------------------------------------------------------------------
# Config.

with open(os.path.join(SNAP, "config.json")) as f:
    CFG = json.load(f)["text_config"]
LA = CFG["linear_attn_config"]
HID = CFG["hidden_size"]
H = LA["num_heads"]
D = LA["head_dim"]
P = H * D
W = LA["short_conv_kernel_size"]
LB = LA["gate_lower_bound"]
EPS = CFG["rms_norm_eps"]
SCALE = D ** -0.5

# The 1-BASED list, and the assertion that the layer we are about to load really is KDA. A modulus
# rule (`i % 4 == 3`) agrees with this list everywhere except 0-based layer 92 — the tail is
# `KKK MM`, not a clean 3:1 motif.
assert (LAYER + 1) in LA["kda_layers"], f"layer {LAYER} (0-based) is not KDA"

PFX = f"language_model.model.layers.{LAYER}."
AT = PFX + "self_attn."

print(f"K3 KDA layer {LAYER}: hidden={HID} H={H} D={D} W={W} lower_bound={LB} T={T}")

# ---------------------------------------------------------------------------------------------
# Real weights.

Wq = load(AT + "q_proj.weight").float()
Wk = load(AT + "k_proj.weight").float()
Wv = load(AT + "v_proj.weight").float()
Wg = load(AT + "g_proj.weight").float()
Wo = load(AT + "o_proj.weight").float()
Wfa = load(AT + "f_a_proj.weight").float()
Wfb = load(AT + "f_b_proj.weight").float()
Wb = load(AT + "b_proj.weight").float()
convw = [load(AT + f"{s}_conv1d.weight").float().view(P, W) for s in ("q", "k", "v")]
o_norm = load(AT + "o_norm.weight").float()
ln_w = load(PFX + "input_layernorm.weight").float()

# THE NARROW. The checkpoint ships A_log as `[128]` = head_dim, but it is a per-head `[96]`
# ZERO-PADDED to head_dim: exactly indices 0..95 are non-zero, verified in 69/69 KDA layers by
# scripts/kda_verify_ckpt.py. The kernel indexes `A_log + i_hv` with i_hv in [0,96), so the
# padding is never read. A loader that consumes all 128 as if per-head-dim silently computes the
# wrong decay for every token of every KDA layer.
A_raw = load(AT + "A_log").float()
assert A_raw.numel() == D, f"A_log is {A_raw.numel()}, expected head_dim={D}"
assert torch.all(A_raw[H:] == 0), "A_log[96:] is not zero — the padding assumption is wrong"
assert torch.all(A_raw[:H] != 0), "A_log[:96] has a zero — the narrow may be wrong"
A_log = A_raw[:H].contiguous()
dt_bias = load(AT + "dt_bias").float()
assert dt_bias.numel() == P and torch.all(dt_bias != 0), "dt_bias is padded or short"
print(f"  A_log narrowed [{D}] -> [{H}];  exp(A_log) in "
      f"[{A_log.exp().min():.3f}, {A_log.exp().max():.3f}];  dt_bias mean {dt_bias.mean():.3f}")

# ---------------------------------------------------------------------------------------------
# fla references, VENDORED VERBATIM. Do not "simplify" these — they are the oracle.
#   fla/ops/kda/gate.py::naive_kda_lowerbound_gate
#   fla/ops/kda/naive.py::naive_recurrent_kda


def naive_kda_lowerbound_gate(g, A_log, dt_bias=None, lower_bound=-5.0, output_dtype=torch.float32):
    Hh = g.shape[-2]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(g.shape[-2:])
    g = lower_bound * torch.sigmoid(A_log.view(Hh, 1).exp() * g)
    return g.to(output_dtype)


def naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False):
    dtype = v.dtype
    B, Tn, Hn, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // Hn
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)
    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S = S + initial_state
    o = torch.zeros_like(v)
    for i in range(0, Tn):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "b h k, b h v -> b h k v", b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2)
        )
        o[:, i] = torch.einsum("b h k, b h k v -> b h v", q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


# ---------------------------------------------------------------------------------------------
# Inputs. Seeded, and both carried states are NON-ZERO so the decay, the carry and the conv
# history are all actually exercised — a zero initial state hides a decay bug completely.

gen = torch.Generator().manual_seed(0xB4)
hidden = (0.3 * torch.randn(T, HID, generator=gen)).to(torch.bfloat16)
state_in_kv = 0.05 * torch.randn(H, D, D, generator=gen)  # MATH layout [h][k][v]
conv_state_in = [0.2 * torch.randn(P, W, generator=gen) for _ in range(3)]

# ---------------------------------------------------------------------------------------------
# Stage by stage, all fp32.

# P0 — pre-norm.
xf = hidden.float()
x = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * ln_w
x = x.to(torch.bfloat16).float()

# P1-P7 — the seven projections.
q_raw, k_raw, v_raw = x @ Wq.T, x @ Wk.T, x @ Wv.T
g_hat = x @ Wg.T                       # output gate, FULL RANK (use_full_rank_gate: true)
f_lat = x @ Wfa.T                      # forget gate stays LOW RANK 128 regardless of that flag
f_raw = f_lat @ Wfb.T
b_raw = x @ Wb.T                       # [T,H] — ONE SCALAR PER HEAD


def short_conv(xin, weight, cache):
    """Causal depthwise width-W conv + SiLU, with a rolling window.

    `cache` is [P,W] holding the last W inputs per channel with the CURRENT token at slot W-1
    ([fla] short_conv.py:232-235):  cache.roll(-1); cache[:,-1] = x; y = (cache*weight).sum(-1).
    Activation is applied AFTER the convolution.
    """
    c = cache.clone()
    out = torch.empty(T, P)
    for t in range(T):
        c = torch.roll(c, -1, dims=-1)
        c[:, W - 1] = xin[t]
        out[t] = (c * weight).sum(-1)
    return torch.nn.functional.silu(out), c


qc, cs_q = short_conv(q_raw, convw[0], conv_state_in[0])
kc, cs_k = short_conv(k_raw, convw[1], conv_state_in[1])
vc, cs_v = short_conv(v_raw, convw[2], conv_state_in[2])
# The device writes bf16, so round here too or the residual measures the rounding.
qc, kc, vc = (t.to(torch.bfloat16).float() for t in (qc, kc, vc))

# P9 — gate and beta.
gate = naive_kda_lowerbound_gate(f_raw.view(T, H, D), A_log, dt_bias, LB).view(T, P)
beta = torch.sigmoid(b_raw)
assert gate.max() < 0 and gate.min() >= LB, "bounded gate escaped [lower_bound, 0)"

# P10 — the recurrence. L2 norm is `use_qk_l2norm_in_kernel=True`, which `naive_recurrent_kda`
# does NOT do, so it is applied here exactly as the kernel does: eps INSIDE the sqrt, and q is
# scaled by D^-0.5 AFTER the norm (naive applies `scale` itself) while k is NOT scaled.
qh = qc.view(T, H, D)
kh = kc.view(T, H, D)
qn = qh / torch.sqrt(qh.pow(2).sum(-1, keepdim=True) + 1e-6)
kn = kh / torch.sqrt(kh.pow(2).sum(-1, keepdim=True) + 1e-6)
o_ref, S_kv = naive_recurrent_kda(
    qn[None], kn[None], vc.view(T, H, D)[None], gate.view(T, H, D)[None], beta[None],
    scale=SCALE, initial_state=state_in_kv[None], output_final_state=True,
)
o_ref = o_ref[0].reshape(T, P)
# STATE IS V-FIRST ON DEVICE. The reference carries [h][k][v]; K3 sets transpose_state_layout=True
# and stores [h][v][k]. V == K == 128, so a transposed state has EXACTLY the right norm and no
# magnitude check finds the error. This transpose is the whole point of the fixture.
S_vk = S_kv[0].transpose(1, 2).contiguous()

# P11 — gated output norm. Over D=128 INSIDE a head, with a [D] weight shared by all heads, and
# the sigmoid on the RAW g_proj output applied AFTER the norm.
oh = o_ref.view(T, H, D)
y = oh * torch.rsqrt(oh.pow(2).mean(-1, keepdim=True) + EPS) * o_norm
y = (y.view(T, P) * torch.sigmoid(g_hat)).to(torch.bfloat16).float()

# P12/P13.
attn = y @ Wo.T
block_out = xf + attn

print(f"  |x| {x.norm():.3f}  |qc| {qc.norm():.3f}  |o| {o_ref.norm():.3f}  "
      f"|y| {y.norm():.3f}  |out| {block_out.norm():.3f}")
print(f"  gate in [{gate.min():.4f}, {gate.max():.4f}]  decay exp(g) in "
      f"[{gate.exp().min():.5f}, {gate.exp().max():.5f}]  beta in "
      f"[{beta.min():.4f}, {beta.max():.4f}]")
print(f"  state |in| {state_in_kv.norm():.3f} -> |out| {S_vk.norm():.3f}")

# ---------------------------------------------------------------------------------------------
# PREFILL/DECODE AGREEMENT, checked HERE on the reference itself (spec section 5.2 / 7.8 item 5).
# Run T tokens in one call, then T-1 followed by 1, and require the states to match. This is the
# invariant a chunked-prefill scheduler depends on and the one most likely to catch a layout bug.
if T > 1:
    _, S_a = naive_recurrent_kda(
        qn[None, : T - 1], kn[None, : T - 1], vc.view(T, H, D)[None, : T - 1],
        gate.view(T, H, D)[None, : T - 1], beta[None, : T - 1],
        scale=SCALE, initial_state=state_in_kv[None], output_final_state=True)
    _, S_b = naive_recurrent_kda(
        qn[None, T - 1 :], kn[None, T - 1 :], vc.view(T, H, D)[None, T - 1 :],
        gate.view(T, H, D)[None, T - 1 :], beta[None, T - 1 :],
        scale=SCALE, initial_state=S_a, output_final_state=True)
    d = (S_b - S_kv).abs().max().item()
    print(f"  prefill/decode agreement on the REFERENCE: max|dS| = {d:.3e}")
    assert d < 1e-4, "the reference disagrees with itself — split T is not equivalent"

# ---------------------------------------------------------------------------------------------
# Fixture.


def w_bf(f, t):
    f.write(np.ascontiguousarray(t.to(torch.bfloat16).view(torch.uint16).numpy()).tobytes())


def w_f32(f, t):
    f.write(np.ascontiguousarray(t.float().numpy().astype(np.float32)).tobytes())


with open(OUT, "wb") as f:
    f.write(struct.pack("<8i", MAGIC, T, H, D, HID, W, 1, 16))  # gate_mode=1 (bounded), BV=16
    f.write(struct.pack("<3f", EPS, LB, SCALE))
    # inputs + real weights
    w_bf(f, hidden)
    w_bf(f, ln_w)
    for t in (Wq, Wk, Wv, Wg, Wo, Wfa, Wfb, Wb):
        w_bf(f, t)
    for t in convw:
        w_f32(f, t)
    w_f32(f, A_log)
    w_f32(f, dt_bias)
    w_f32(f, o_norm)
    for t in conv_state_in:
        w_f32(f, t)
    w_f32(f, state_in_kv.transpose(1, 2).contiguous())  # V-FIRST on the wire
    # per-stage references
    w_bf(f, x)
    w_bf(f, qc)
    w_bf(f, kc)
    w_bf(f, vc)
    w_f32(f, gate)
    w_f32(f, beta)
    w_bf(f, o_ref)
    w_f32(f, S_vk)
    w_bf(f, y)
    w_bf(f, block_out)
    for t in (cs_q, cs_k, cs_v):
        w_f32(f, t)
    sz = f.tell()

print(f"wrote {OUT}  ({sz / 1e6:.1f} MB)")
