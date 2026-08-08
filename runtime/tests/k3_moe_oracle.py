#!/usr/bin/env python3
# k3_moe_oracle.py — REAL-WEIGHT reference for a COMPLETE Kimi-K3 MoE BLOCK (layer 1).  [K3-MOE-GATE]
#
# Rung 2. Rung 1 (k3_real_oracle.py) covered layer 0: KDA + one AttnRes + a dense situ FFN. This
# adds the two things layer 0 does not have:
#
#   * BOTH AttnRes applications. Layer 1 does NOT take a block-residual snapshot (1 % 12 != 0), so
#     `prefix_sum` survives across the attention and the ATTN-SIDE AttnRes — skipped at layer 0
#     because block_residual was empty — is live. The two use DIFFERENT weights
#     (`self_attention_res_*` vs `mlp_res_*`); layer 0 never reads the first pair at all.
#   * STABLE LATENTMOE. The routed experts do NOT run at hidden width:
#
#         x(7168) --down--> xe(3584) --896 experts, top-16, K=3584, I=3072, MXFP4-->
#              --combine--> ylat(3584) --RMSNorm--> --up--> yh(7168)
#         out = yh + shared_experts(x)          # shared reads the PRE-DOWN hidden, at 7168
#
#     AMD's day-0 post confirms it independently: "Stable LatentMoE first projects the
#     7168-dimensional hidden state down to 3584 dimensions before running the expert computation".
#
# T = 1, and that is not a simplification — it is the shape of the op. plow's decode MoE path
# (`MoeRouterTopk` + per-slot `MoeExpertGluFp8Blk`) carries ONE token: the routing table is `[k]`,
# not `[T,k]`, and `d_moe_expert_glu_fp8_blk` reads a single `x` row. T>1 is the separate prefill
# grouped path (ops 83-87), which this gate does not touch.
#
# ONLY THE SELECTED EXPERTS ARE MATERIALIZED. 896 experts x 3 mxfp4 tensors is 15.7 GB; top-16 is
# 280 MB. The pointer table's entries for unselected experts are left NULL, which the kernel treats
# as "not my expert" and skips — so if plow's router picked a DIFFERENT expert from this reference,
# the partial comes back zero and the residual explodes. That is deliberate: it converts a routing
# divergence from silent into loud. The harness ALSO diffs the routing table itself, expert id by
# expert id, so a divergence is localized rather than merely detected.
#
#   K3_DIR=<snapshot>   K3_MOE_LAYER=<0-based, must be MoE + KDA + no snapshot>
#   python3 k3_moe_oracle.py <out.bin>

import json
import os
import struct
import sys
import mmap

import numpy as np
import torch

MAGIC = 0x4B334D31  # "K3M1"

SNAP = os.environ.get(
    "K3_DIR",
    "/home/lava/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
)
LAYER = int(os.environ.get("K3_MOE_LAYER", "1"))
T = 1
OUT = sys.argv[1] if len(sys.argv) > 1 else "k3_moe_fixture.bin"

torch.manual_seed(0xB4)
np.random.seed(0xB4)

_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "U8": torch.uint8}
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


def raw(name):
    """The tensor's bytes and metadata, without a dtype interpretation (mxfp4 is U8)."""
    h, b, path = HDRS[IDX[name]]
    m = h[name]
    lo, hi = m["data_offsets"]
    return _mm(path)[b + lo : b + hi], m


def load(name):
    buf, m = raw(name)
    return torch.frombuffer(bytearray(buf), dtype=_DT[m["dtype"]]).view(*m["shape"])


with open(os.path.join(SNAP, "config.json")) as f:
    CFG = json.load(f)["text_config"]
LA = CFG["linear_attn_config"]
HID = CFG["hidden_size"]
H, D = LA["num_heads"], LA["head_dim"]
P = H * D
W, LB = LA["short_conv_kernel_size"], LA["gate_lower_bound"]
EPS = CFG["rms_norm_eps"]
SCALE = D ** -0.5
ARBS = CFG["attn_res_block_size"]
BETA, LBETA = CFG["activation_situ_beta"], CFG["activation_situ_linear_beta"]
NEXP = CFG["num_experts"]
TOPK = CFG["num_experts_per_token"]
IMOE = CFG["moe_intermediate_size"]
HE = CFG["routed_expert_hidden_size"]
NSHARED = CFG["num_shared_experts"]
SHI = IMOE * NSHARED
RSCALE = CFG["routed_scaling_factor"]
FKDR = CFG["first_k_dense_replace"]

assert (LAYER + 1) in LA["kda_layers"], f"layer {LAYER} is not KDA"
assert LAYER >= FKDR, f"layer {LAYER} is dense, not MoE"
assert LAYER % ARBS != 0, (
    f"layer {LAYER} takes a block-residual snapshot, which resets prefix_sum to None and turns "
    "the attn-side AttnRes off — rung 1's shape, not rung 2's"
)
assert CFG["moe_router_activation_func"] == "sigmoid" and CFG["moe_renormalize"]
assert CFG["num_expert_group"] == 1 and CFG["topk_group"] == 1, (
    "group-limited routing would be live; plow implements it but K3 does not exercise it"
)
assert CFG["latent_moe_use_norm"] and HE != HID

PFX = f"language_model.model.layers.{LAYER}."
AT = PFX + "self_attn."
MOE = PFX + "block_sparse_moe."

print(f"K3 MoE block, layer {LAYER}: hidden={HID} latent={HE} experts={NEXP} top-{TOPK} "
      f"I_moe={IMOE} shared_inter={SHI}")

# ---------------------------------------------------------------------------------------------
# Weights.

Wq, Wk, Wv = (load(AT + n + "_proj.weight").float() for n in ("q", "k", "v"))
Wg = load(AT + "g_proj.weight").float()
Wo = load(AT + "o_proj.weight").float()
Wfa, Wfb = load(AT + "f_a_proj.weight").float(), load(AT + "f_b_proj.weight").float()
Wb = load(AT + "b_proj.weight").float()
convw = [load(AT + f"{s}_conv1d.weight").float().view(P, W) for s in ("q", "k", "v")]
o_norm = load(AT + "o_norm.weight").float()
ln_w = load(PFX + "input_layernorm.weight").float()
post_ln_w = load(PFX + "post_attention_layernorm.weight").float()

A_raw = load(AT + "A_log").float()
assert A_raw.numel() == D and torch.all(A_raw[H:] == 0) and torch.all(A_raw[:H] != 0)
A_log = A_raw[:H].contiguous()
dt_bias = load(AT + "dt_bias").float()

# BOTH AttnRes score folds. Layer 0 never read the first pair; layer 1 does, and they are
# DIFFERENT weights — using one for both is a silent wiring bug with the right shapes.
attn_score_w = (load(PFX + "self_attention_res_norm.weight").float()
                * load(PFX + "self_attention_res_proj.weight").float().squeeze(0))
mlp_score_w = (load(PFX + "mlp_res_norm.weight").float()
               * load(PFX + "mlp_res_proj.weight").float().squeeze(0))
assert not torch.allclose(attn_score_w, mlp_score_w), "the two AttnRes folds are identical?"

Wrouter = load(MOE + "gate.weight").float()               # [NEXP, HID]
rbias = load(MOE + "gate.e_score_correction_bias").float()  # [NEXP] f32
Wdown_l = load(MOE + "routed_expert_down_proj.weight").float()   # [HE, HID]
lat_norm = load(MOE + "routed_expert_norm.weight").float()       # [HE]
Wup_l = load(MOE + "routed_expert_up_proj.weight").float()       # [HID, HE]
Wshg = load(MOE + "shared_experts.gate_proj.weight").float()     # [SHI, HID]
Wshu = load(MOE + "shared_experts.up_proj.weight").float()
Wshd = load(MOE + "shared_experts.down_proj.weight").float()     # [HID, SHI]
assert Wdown_l.shape == (HE, HID) and Wup_l.shape == (HID, HE) and Wshg.shape == (SHI, HID)

# ---------------------------------------------------------------------------------------------
# MXFP4 dequant. Nibble order = element 2i in the LOW nibble, CONFIRMED ON HARDWARE by
# runtime/tests/k3_mxfp4_nibble_test.c against these exact bytes (1.6e-3 vs 1.4e0 for the swap).
# E8M0 bias 127.

E2M1 = np.concatenate([np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)] * 1)
E2M1 = np.concatenate([E2M1, -E2M1])


def deq_mxfp4(name):
    pb, pm = raw(name + ".weight_packed")
    sb, sm = raw(name + ".weight_scale")
    packed = np.frombuffer(pb, np.uint8).reshape(*pm["shape"])
    scale = np.frombuffer(sb, np.uint8).reshape(*sm["shape"])
    N, KH = packed.shape
    K = KH * 2
    assert scale.shape == (N, K // 32)
    w = np.empty((N, K), np.float32)
    w[:, 0::2] = E2M1[packed & 0x0F]
    w[:, 1::2] = E2M1[packed >> 4]
    w *= np.repeat(np.ldexp(np.ones((N, K // 32), np.float32), scale.astype(np.int32) - 127),
                   32, axis=1)
    return torch.from_numpy(w), bytes(pb), bytes(sb)


# ---------------------------------------------------------------------------------------------
# fla references, VENDORED VERBATIM, and K3's own, likewise. Same as k3_real_oracle.py.


def naive_kda_lowerbound_gate(g, A_log, dt_bias=None, lower_bound=-5.0, output_dtype=torch.float32):
    Hh = g.shape[-2]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(g.shape[-2:])
    return (lower_bound * torch.sigmoid(A_log.view(Hh, 1).exp() * g)).to(output_dtype)


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
        S = S + torch.einsum("b h k, b h v -> b h k v", b_i[..., None] * k_i,
                             v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum("b h k, b h k v -> b h v", q_i, S)
    return o.to(dtype), (S if output_final_state else None)


def apply_attn_res(prefix_sum, block_residual, score_weight, eps):
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v_float).squeeze(1), probs.squeeze(1)


def situ(gate, up, beta, linear_beta):
    gate, up = gate.to(torch.float32), up.to(torch.float32)
    a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return a * up


def bf(x):
    """Round through bf16 — the device stores bf16 between every packet."""
    return x.to(torch.bfloat16).float()


# ---------------------------------------------------------------------------------------------
# Inputs. `prefix_in` stands in for layer 0's output and `blkres` for its snapshot; both are
# seeded rather than produced by a real layer-0 run, which is the same "weights real, activations
# synthetic" posture as the GLM B4, KDA and rung-1 gates.

gen = torch.Generator().manual_seed(0xB4)
prefix_in = (0.3 * torch.randn(T, HID, generator=gen)).to(torch.bfloat16)
blkres = (0.35 * torch.randn(T, 1, HID, generator=gen)).to(torch.bfloat16)
NB = 1
state_in_kv = 0.05 * torch.randn(H, D, D, generator=gen)
conv_state_in = [0.2 * torch.randn(P, W, generator=gen) for _ in range(3)]

# ---------------------------------------------------------------------------------------------
# The block.

# A0 — the ATTN-SIDE AttnRes. Live here, absent at layer 0.
h_a_f, probs_a = apply_attn_res(prefix_in, blkres, attn_score_w, EPS)
h_a = bf(h_a_f)

xf = h_a
x = bf(xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * ln_w)

q_raw, k_raw, v_raw = x @ Wq.T, x @ Wk.T, x @ Wv.T
g_hat = x @ Wg.T
f_raw = (x @ Wfa.T) @ Wfb.T
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
qc, kc, vc = (bf(t) for t in (qc, kc, vc))

gate_g = naive_kda_lowerbound_gate(f_raw.view(T, H, D), A_log, dt_bias, LB).view(T, P)
kbeta = torch.sigmoid(b_raw)
qh, kh = qc.view(T, H, D), kc.view(T, H, D)
qn = qh / torch.sqrt(qh.pow(2).sum(-1, keepdim=True) + 1e-6)
kn = kh / torch.sqrt(kh.pow(2).sum(-1, keepdim=True) + 1e-6)
o_ref, S_kv = naive_recurrent_kda(qn[None], kn[None], vc.view(T, H, D)[None],
                                  gate_g.view(T, H, D)[None], kbeta[None], scale=SCALE,
                                  initial_state=state_in_kv[None], output_final_state=True)
o_ref = o_ref[0].reshape(T, P)
S_vk = S_kv[0].transpose(1, 2).contiguous()
oh = o_ref.view(T, H, D)
y = oh * torch.rsqrt(oh.pow(2).mean(-1, keepdim=True) + EPS) * o_norm
y = bf(y.view(T, P) * torch.sigmoid(g_hat))
attn = bf(y @ Wo.T)

# A1 — prefix_sum SURVIVES here (no snapshot at this layer), so it accumulates.
prefix = bf(prefix_in.float() + attn)

# A2 — the MLP-SIDE AttnRes, with the OTHER fold.
h2_f, probs_m = apply_attn_res(prefix, blkres, mlp_score_w, EPS)
h2 = bf(h2_f)
h3 = bf(h2 * torch.rsqrt(h2.pow(2).mean(-1, keepdim=True) + EPS) * post_ln_w)

# ---------------------------------------------------------------------------------------------
# THE ROUTER.
#
# HF computes the logits in fp32 (`F.linear(hidden.float(), weight.float())`); plow's router reads
# a BF16 logit vector written by an ordinary Gemv. That is a real difference and at 896 experts it
# can flip a marginal selection, so the reference selection is computed from the BF16-ROUNDED
# logits — matching what the device is given — and the fp32-vs-bf16 disagreement is reported
# SEPARATELY rather than folded into the residual. Blaming plow's router for a precision choice
# made two packets earlier is exactly the "compare against the right reference" trap.

logit_f32 = h3.float() @ Wrouter.T          # [T, NEXP]
logit_bf = bf(logit_f32)


def route(logit):
    scores = torch.sigmoid(logit)
    sel_key = scores + rbias.unsqueeze(0)
    idx = torch.topk(sel_key, k=TOPK, dim=-1, sorted=True)[1]
    w = scores.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20)   # moe_renormalize
    return idx, w * RSCALE


topk_idx, topk_w = route(logit_bf)
idx32, _ = route(logit_f32)
flip = int((torch.sort(topk_idx, -1)[0] != torch.sort(idx32, -1)[0]).sum())
print(f"  router: top-{TOPK} of {NEXP};  fp32-vs-bf16 logit selection differs in {flip} slot(s)")
print(f"  selected experts: {sorted(topk_idx[0].tolist())}")
print(f"  gates in [{topk_w.min():.5f}, {topk_w.max():.5f}], sum {topk_w.sum():.6f}")

# ---------------------------------------------------------------------------------------------
# STABLE LATENTMOE. Down to 3584, experts, norm, up to 7168 — and the shared expert reads the
# PRE-DOWN hidden.

xe = bf(h3 @ Wdown_l.T)                     # [T, HE]

sel = topk_idx[0].tolist()
gates = topk_w[0].tolist()
exp_bytes = []                              # (eid, w1p, w1s, w3p, w3s, w2p, w2s) in TABLE ORDER
fu_ref = torch.zeros(TOPK, IMOE)
part_ref = torch.zeros(TOPK, HE)
for j, e in enumerate(sel):
    base = MOE + f"experts.{e}."
    W1, w1p, w1s = deq_mxfp4(base + "w1")   # gate  [IMOE, HE]
    W3, w3p, w3s = deq_mxfp4(base + "w3")   # up    [IMOE, HE]
    W2, w2p, w2s = deq_mxfp4(base + "w2")   # down  [HE, IMOE]
    assert W1.shape == (IMOE, HE) and W2.shape == (HE, IMOE), (W1.shape, W2.shape)
    g_ = xe[0] @ W1.T
    u_ = xe[0] @ W3.T
    fu_ref[j] = bf(situ(g_, u_, BETA, LBETA))
    # The GATE MULTIPLY LANDS ON THE DOWN PARTIAL (op_moe.h `part_slot[h] = gate * y`), not on the
    # combine. Put it in the same place or the per-slot `part` diff is meaningless.
    part_ref[j] = gates[j] * (fu_ref[j] @ W2.T)
    exp_bytes.append((e, w1p, w1s, w3p, w3s, w2p, w2s))

ylat = bf(part_ref.sum(0)).unsqueeze(0)                      # MoeCombine, f32 accum -> bf16
yn = bf(ylat * torch.rsqrt(ylat.pow(2).mean(-1, keepdim=True) + EPS) * lat_norm)
yh = bf(yn @ Wup_l.T)                                        # [T, HID]

shg = bf(h3 @ Wshg.T)
shu = bf(h3 @ Wshu.T)
sha = bf(situ(shg, shu, BETA, LBETA))
shd = bf(sha @ Wshd.T)

moe_out = bf(yh + shd)
block_out = bf(prefix.float() + moe_out)

print(f"  |prefix_in| {prefix_in.float().norm():.3f}  |attn| {attn.norm():.3f}  "
      f"|prefix| {prefix.norm():.3f}  |h2| {h2.norm():.3f}  |h3| {h3.norm():.3f}")
print(f"  |xe| {xe.norm():.3f}  |ylat| {ylat.norm():.3f}  |yh| {yh.norm():.3f}  "
      f"|shared| {shd.norm():.3f}  |out| {block_out.norm():.3f}")
print(f"  AttnRes probs  attn-side {probs_a[0].tolist()}   mlp-side {probs_m[0].tolist()}")

# CONTROLS — and the first one is a correction worth stating, because it changes where this gate
# has to look.
#
# At layer 0 the block OUTPUT is `attn + mlp` and the plain wiring is `hidden + attn + mlp`, so a
# wrong wiring is a ~1.0 relative error at the output and rung 1's last row catches it. At layer 1
# there is NO snapshot, so `prefix = prefix_in + attn` and the block output is
# `prefix_in + attn + moe` — WHICH IS EXACTLY WHAT THE PLAIN WIRING PRODUCES. The two differ only
# through the MoE's INPUT, and the MoE contributes ~3% of the residual's norm here.
#
# Measured: 3.0e-3 at the block output. A block-output-only gate would NOT catch AttnRes being
# wired as a plain residual at this layer. That is the "compare against the right reference" trap
# in a new place, so the controls below are taken at the two points where the op ACTUALLY ACTS —
# the attention input and the MLP input — and the harness diffs `h_a` and `h2` as their own rows
# rather than trusting the output.
plain_out = bf(prefix_in.float() + attn + moe_out)
print(f"  [control] block out vs the PLAIN wiring: rel "
      f"{(plain_out - block_out).norm() / block_out.norm():.3e} — SMALL, and that is the point: "
      f"at a non-snapshot layer the block output cannot see this")
d_ha = (h_a - prefix_in.float()).norm() / prefix_in.float().norm()
d_h2 = (h2 - prefix.float()).norm() / prefix.float().norm()
print(f"  [control] attn-side AttnRes moves its input by rel {d_ha:.3e};  "
      f"mlp-side by rel {d_h2:.3e}  (both must be LARGE)")
assert d_ha > 0.1 and d_h2 > 0.1, "AttnRes is a no-op here — the gate would prove nothing"
d_shared = (yh - moe_out).norm() / moe_out.norm()
print(f"  [control] routed-only vs routed+shared: rel {d_shared:.3e} "
      f"(the shared expert is not negligible)")
assert d_shared > 0.05

# ---------------------------------------------------------------------------------------------
# Fixture.


def w_bf(f, t):
    f.write(np.ascontiguousarray(t.to(torch.bfloat16).view(torch.uint16).numpy()).tobytes())


def w_f32(f, t):
    f.write(np.ascontiguousarray(t.float().numpy().astype(np.float32)).tobytes())


with open(OUT, "wb") as f:
    f.write(struct.pack("<16i", MAGIC, T, H, D, HID, W, 1, 16, IMOE, HE, NEXP, TOPK, len(sel),
                        SHI, NB, 1 | 2 | 4))
    f.write(struct.pack("<6f", EPS, LB, SCALE, BETA, LBETA, RSCALE))
    w_bf(f, prefix_in)
    w_bf(f, blkres.view(T * NB, HID))
    w_f32(f, attn_score_w)
    w_f32(f, mlp_score_w)
    w_bf(f, ln_w)
    w_bf(f, post_ln_w)
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
    w_bf(f, Wrouter)
    w_f32(f, rbias)
    w_bf(f, Wdown_l)
    w_bf(f, lat_norm)
    w_bf(f, Wup_l)
    w_bf(f, Wshg)
    w_bf(f, Wshu)
    w_bf(f, Wshd)
    # experts, in TABLE order: slot j -> (gate=w1, up=w3, down=w2), packed then scale for each
    f.write(np.array([e for e, *_ in exp_bytes], np.uint32).tobytes())
    for _, w1p, w1s, w3p, w3s, w2p, w2s in exp_bytes:
        for blob in (w1p, w1s, w3p, w3s, w2p, w2s):
            f.write(blob)
    # references
    w_bf(f, h_a)
    w_bf(f, x)
    w_f32(f, S_vk)
    w_bf(f, attn)
    w_bf(f, prefix)
    w_bf(f, h2)
    w_bf(f, h3)
    w_bf(f, logit_bf)
    f.write(np.array(sel, np.uint32).tobytes())
    f.write(np.array(gates, np.float32).tobytes())
    w_bf(f, xe)
    w_bf(f, fu_ref)
    w_f32(f, part_ref)
    w_bf(f, ylat)
    w_bf(f, yn)
    w_bf(f, yh)
    w_bf(f, shd)
    w_bf(f, block_out)
    for t in (cs_q, cs_k, cs_v):
        w_f32(f, t)
    sz = f.tell()

print(f"wrote {OUT}  ({sz / 1e6:.1f} MB, {len(sel)} experts materialized of {NEXP})")
