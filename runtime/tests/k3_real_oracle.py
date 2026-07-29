#!/usr/bin/env python3
# k3_real_oracle.py — REAL-WEIGHT reference for ONE COMPLETE Kimi-K3 BLOCK.            [K3-BLOCK-GATE]
#
# The KDA gate (kda_real_oracle.py) de-risked the MIXER. This de-risks the BLOCK AROUND IT, which
# is where K3 stops looking like GLM:
#
#     a K3 layer is NOT `residual + attn` then `residual + mlp`.
#
# `attn_res_block_size: 12` routes every layer through `_forward_attn_residual`
# (modeling_kimi_linear.py:973). The plain residual ADD is replaced by AttnRes — a softmax mix over
# the running prefix sum and up to 8 snapshots of it — applied TWICE per layer, and the block's
# OUTPUT is the prefix sum, not `input + attn + mlp`. Wiring it the plain way gives the same shapes,
# the same dtypes and similar magnitudes: fluent, wrong output.
#
# LAYER 0, the first rung, traced from the reference line by line:
#
#     prefix_sum = hidden                              # the embedding output
#     # block_residual has 0 rows, so the FIRST AttnRes is SKIPPED (`if ... .shape[1] > 0`)
#     # layer_idx % 12 == 0, so:
#     block_residual = [hidden] ;  prefix_sum = None
#     h  = KDA(input_layernorm(hidden))
#     prefix_sum = h                                   # NOT hidden + h — prefix_sum was None
#     h2 = AttnRes(prefix_sum, block_residual, mlp_res_proj, mlp_res_norm)
#     h3 = mlp(post_attention_layernorm(h2))           # DENSE, situ-gated (first_k_dense_replace=1)
#     out = prefix_sum + h3                            # = KDA_out + MLP_out
#
# Note what is NOT in `out`: the embedding hidden state appears only THROUGH the AttnRes mix and
# through `block_residual`. `self_attention_res_norm`/`self_attention_res_proj` exist on layer 0 in
# the checkpoint and are NEVER READ there — which is why the coverage list for this layer must be
# derived from the reference's control flow, not from the tensor set.
#
# WHAT THE REFERENCE IS. For KDA: `fla`'s pure-torch `naive_recurrent_kda` and
# `naive_kda_lowerbound_gate`, vendored verbatim (see kda_real_oracle.py for why transformers and
# every released vLLM are unusable). For AttnRes and `situ`: `_apply_attn_res` and `SituAndMul`
# transcribed verbatim from `modeling_kimi_linear.py`, which is K3's own shipped code and is
# correct for those two even though its `A_log` declaration makes the module unloadable.
#
# INPUT is a seeded synthetic hidden state; the WEIGHTS are real. Same posture as the GLM B4 and
# KDA gates, stated rather than hidden.
#
#   K3_DIR=<snapshot>   K3_LAYER=<0-based layer>   K3_T=<tokens>
#   python3 k3_real_oracle.py <out.bin>

import json
import os
import struct
import sys
import mmap

import numpy as np
import torch

MAGIC = 0x4B334231  # "K3B1"

SNAP = os.environ.get(
    "K3_DIR",
    "/home/lava/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
)
LAYER = int(os.environ.get("K3_LAYER", "0"))
T = int(os.environ.get("K3_T", "4"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "k3_fixture.bin"

torch.manual_seed(0xB4)
np.random.seed(0xB4)

# ---------------------------------------------------------------------------------------------
# safetensors, header-driven and mmap'd. Never load 1.5 TB.

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
ARBS = CFG["attn_res_block_size"]
BETA = CFG["activation_situ_beta"]
LBETA = CFG["activation_situ_linear_beta"]
INTER = CFG["intermediate_size"]
FKDR = CFG["first_k_dense_replace"]

assert CFG["hidden_act"] == "situ", CFG["hidden_act"]
# The 1-BASED list. A modulus rule (`i % 4 == 3`) agrees everywhere except 0-based layer 92.
assert (LAYER + 1) in LA["kda_layers"], f"layer {LAYER} (0-based) is not KDA"
assert LAYER < FKDR, (
    f"layer {LAYER} is MoE (first_k_dense_replace={FKDR}); this oracle covers the DENSE rung. "
    "The LatentMoE rung is a separate fixture."
)
assert LAYER % ARBS == 0, "this oracle assumes the layer takes a block-residual snapshot"

PFX = f"language_model.model.layers.{LAYER}."
AT = PFX + "self_attn."

print(f"K3 block, layer {LAYER}: hidden={HID} H={H} D={D} W={W} lb={LB} T={T}")
print(f"  dense FFN inter={INTER}  situ beta={BETA} linear_beta={LBETA}  attn_res_block={ARBS}")

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
post_ln_w = load(PFX + "post_attention_layernorm.weight").float()

# THE FOLD. `score_weight = norm.weight * proj.weight.squeeze(0)` (_apply_attn_res:1084) is
# CONSTANT — both factors are parameters — so it collapses at prep time into one [HID] f32 and the
# device never sees either factor. `proj.weight` is [1, HID]: a Linear(hidden, 1), i.e. the
# "query" of this one-query attention, and squeezing the wrong axis would be a silent transpose of
# a vector onto itself at HID=7168 only if it were square, which it is not — so it fails loudly.
mlp_res_norm = load(PFX + "mlp_res_norm.weight").float()
mlp_res_proj = load(PFX + "mlp_res_proj.weight").float()
assert mlp_res_proj.shape == (1, HID), mlp_res_proj.shape
mlp_score_w = mlp_res_norm * mlp_res_proj.squeeze(0)

# Layer 0's self_attention_res_* pair EXISTS and is NEVER READ: the first AttnRes is skipped
# because block_residual is empty. Loaded only to assert it is there and to say so out loud.
_ = load(PFX + "self_attention_res_norm.weight")
_ = load(PFX + "self_attention_res_proj.weight")
print("  NOTE: self_attention_res_{norm,proj} exist on this layer and are UNUSED "
      "(block_residual is empty at layer 0, so the attn-side AttnRes is skipped)")

# Dense FFN. `mlp.*` is in quantization_config.ignore, so bf16, not mxfp4.
Wgate = load(PFX + "mlp.gate_proj.weight").float()
Wup = load(PFX + "mlp.up_proj.weight").float()
Wdown = load(PFX + "mlp.down_proj.weight").float()
assert Wgate.shape == (INTER, HID) and Wdown.shape == (HID, INTER)

A_raw = load(AT + "A_log").float()
assert A_raw.numel() == D, f"A_log is {A_raw.numel()}, expected head_dim={D}"
assert torch.all(A_raw[H:] == 0), "A_log[96:] is not zero — the padding assumption is wrong"
assert torch.all(A_raw[:H] != 0), "A_log[:96] has a zero — the narrow may be wrong"
A_log = A_raw[:H].contiguous()
dt_bias = load(AT + "dt_bias").float()
assert dt_bias.numel() == P and torch.all(dt_bias != 0)

# ---------------------------------------------------------------------------------------------
# fla references, VENDORED VERBATIM (fla/ops/kda/{gate,naive}.py). Do not "simplify".


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


# K3's own references, VENDORED VERBATIM from modeling_kimi_linear.py.
#   _apply_attn_res              :1075-1088
#   SituAndMul.forward           :75-85


def apply_attn_res(prefix_sum, block_residual, score_weight, eps):
    """prefix_sum [T,H]; block_residual [T,nb,H]; score_weight [H] (already folded)."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v_float).squeeze(1), probs.squeeze(1)


def situ(gate, up, beta, linear_beta):
    gate = gate.to(torch.float32)
    up = up.to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return situ_a * up


# ---------------------------------------------------------------------------------------------
# Inputs. Seeded; both carried KDA states are NON-ZERO so decay, carry and conv history are all
# exercised — a zero initial state hides a decay bug completely.

gen = torch.Generator().manual_seed(0xB4)
hidden = (0.3 * torch.randn(T, HID, generator=gen)).to(torch.bfloat16)
state_in_kv = 0.05 * torch.randn(H, D, D, generator=gen)  # MATH layout [h][k][v]
conv_state_in = [0.2 * torch.randn(P, W, generator=gen) for _ in range(3)]

# ---------------------------------------------------------------------------------------------
# The block, stage by stage, all fp32.

# S0 — pre-norm. Reads the RAW hidden: the attn-side AttnRes is skipped at layer 0.
xf = hidden.float()
x = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * ln_w
x = x.to(torch.bfloat16).float()

# S1..S7 — the seven KDA projections.
q_raw, k_raw, v_raw = x @ Wq.T, x @ Wk.T, x @ Wv.T
g_hat = x @ Wg.T
f_lat = x @ Wfa.T
f_raw = f_lat @ Wfb.T
b_raw = x @ Wb.T


def short_conv(xin, weight, cache):
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
qc, kc, vc = (t.to(torch.bfloat16).float() for t in (qc, kc, vc))

gate_g = naive_kda_lowerbound_gate(f_raw.view(T, H, D), A_log, dt_bias, LB).view(T, P)
beta = torch.sigmoid(b_raw)
assert gate_g.max() < 0 and gate_g.min() >= LB

qh, kh = qc.view(T, H, D), kc.view(T, H, D)
qn = qh / torch.sqrt(qh.pow(2).sum(-1, keepdim=True) + 1e-6)
kn = kh / torch.sqrt(kh.pow(2).sum(-1, keepdim=True) + 1e-6)
o_ref, S_kv = naive_recurrent_kda(
    qn[None], kn[None], vc.view(T, H, D)[None], gate_g.view(T, H, D)[None], beta[None],
    scale=SCALE, initial_state=state_in_kv[None], output_final_state=True,
)
o_ref = o_ref[0].reshape(T, P)
S_vk = S_kv[0].transpose(1, 2).contiguous()  # V-FIRST on the wire

oh = o_ref.view(T, H, D)
y = oh * torch.rsqrt(oh.pow(2).mean(-1, keepdim=True) + EPS) * o_norm
y = (y.view(T, P) * torch.sigmoid(g_hat)).to(torch.bfloat16).float()
attn = (y @ Wo.T).to(torch.bfloat16).float()

# S8 — THE BLOCK STRUCTURE. `prefix_sum` was set to None by the snapshot, so it becomes the bare
# attention output. This is the line a `residual + attn` wiring gets wrong, and nothing downstream
# would look odd.
prefix_sum = attn
block_residual = xf.unsqueeze(1)          # [T, 1, HID] — the snapshot taken before the norm
h2_f, probs = apply_attn_res(prefix_sum.to(torch.bfloat16), block_residual.to(torch.bfloat16),
                             mlp_score_w, EPS)
h2 = h2_f.to(torch.bfloat16).float()

# S9 — post-attention norm + the DENSE situ FFN.
h2v = h2
h3 = h2v * torch.rsqrt(h2v.pow(2).mean(-1, keepdim=True) + EPS) * post_ln_w
h3 = h3.to(torch.bfloat16).float()
ff_gate = (h3 @ Wgate.T).to(torch.bfloat16).float()
ff_up = (h3 @ Wup.T).to(torch.bfloat16).float()
ff_act = situ(ff_gate, ff_up, BETA, LBETA).to(torch.bfloat16).float()
ff_out = (ff_act @ Wdown.T).to(torch.bfloat16).float()

# S10 — the block output IS the prefix sum. Not `hidden + attn + mlp`.
block_out = prefix_sum + ff_out

print(f"  |x| {x.norm():.3f}  |attn| {attn.norm():.3f}  |h2| {h2.norm():.3f}  "
      f"|ffn| {ff_out.norm():.3f}  |out| {block_out.norm():.3f}")
print(f"  AttnRes probs (row0=snapshot, row1=prefix): {probs[0].tolist()}")
print(f"  situ: gate in [{ff_gate.min():.2f},{ff_gate.max():.2f}]  "
      f"up in [{ff_up.min():.2f},{ff_up.max():.2f}]  act in [{ff_act.min():.3f},{ff_act.max():.3f}]")

# A control the residual table cannot provide: how far AttnRes is from the plain residual add the
# rest of this tree would have emitted. If these were close, the whole op would be unfalsifiable
# by a numeric gate and only a code read would find a wrong wiring.
plain = (xf + attn) + ff_out
d_plain = (plain - block_out).norm() / block_out.norm()
print(f"  AttnRes vs the PLAIN `hidden+attn+mlp` wiring: rel {d_plain:.3e} "
      f"(must be LARGE — it is what the gate would NOT have caught)")
assert d_plain > 0.1, "AttnRes and the plain residual agree — this gate proves nothing"

if T > 1:
    _, S_a = naive_recurrent_kda(
        qn[None, : T - 1], kn[None, : T - 1], vc.view(T, H, D)[None, : T - 1],
        gate_g.view(T, H, D)[None, : T - 1], beta[None, : T - 1],
        scale=SCALE, initial_state=state_in_kv[None], output_final_state=True)
    _, S_b = naive_recurrent_kda(
        qn[None, T - 1 :], kn[None, T - 1 :], vc.view(T, H, D)[None, T - 1 :],
        gate_g.view(T, H, D)[None, T - 1 :], beta[None, T - 1 :],
        scale=SCALE, initial_state=S_a, output_final_state=True)
    d = (S_b - S_kv).abs().max().item()
    print(f"  prefill/decode agreement on the REFERENCE: max|dS| = {d:.3e}")
    assert d < 1e-4

# ---------------------------------------------------------------------------------------------
# Fixture.


def w_bf(f, t):
    f.write(np.ascontiguousarray(t.to(torch.bfloat16).view(torch.uint16).numpy()).tobytes())


def w_f32(f, t):
    f.write(np.ascontiguousarray(t.float().numpy().astype(np.float32)).tobytes())


with open(OUT, "wb") as f:
    f.write(struct.pack("<10i", MAGIC, T, H, D, HID, W, 1, 16, INTER, 1))  # gate_mode, BV, nb=1
    f.write(struct.pack("<5f", EPS, LB, SCALE, BETA, LBETA))
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
    w_f32(f, state_in_kv.transpose(1, 2).contiguous())
    w_f32(f, mlp_score_w)                # the FOLD — one [HID] f32, not two bf16 factors
    w_bf(f, post_ln_w)
    w_bf(f, Wgate)
    w_bf(f, Wup)
    w_bf(f, Wdown)
    # per-stage references
    w_bf(f, x)
    w_bf(f, qc)
    w_bf(f, kc)
    w_bf(f, vc)
    w_f32(f, gate_g)
    w_f32(f, beta)
    w_bf(f, o_ref)
    w_f32(f, S_vk)
    w_bf(f, y)
    w_bf(f, attn)
    w_bf(f, h2)
    w_bf(f, h3)
    w_bf(f, ff_gate)
    w_bf(f, ff_act)
    w_bf(f, ff_out)
    w_bf(f, block_out)
    for t in (cs_q, cs_k, cs_v):
        w_f32(f, t)
    sz = f.tell()

print(f"wrote {OUT}  ({sz / 1e6:.1f} MB)")
