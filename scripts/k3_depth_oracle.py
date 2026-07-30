#!/usr/bin/env python3
"""k3_depth_oracle.py — CPU reference for a DEPTH-TRUNCATED Kimi-K3 text tower.

Runs `KimiLinearModel.forward` truncated to N layers, on real weights, and writes the
full [vocab] logit row for the LAST prompt token, so it can be compared elementwise
against plow's `--dump-logits` output at the same `K3_NLAYERS=N`.

The module math is transcribed from `modeling_kimi_linear.py` (the text tower); the
KDA mixer reuses the vendored fla references already used by
`runtime/tests/k3_real_oracle.py`, and the mxfp4 dequant + latent-MoE order reuse
`runtime/tests/k3_moe_oracle.py`.

  K3_NLAYERS=<N> K3_PROMPT=1008,10484,318,15383,387 \
      python3 scripts/k3_depth_oracle.py <out.f32>

`<out.f32>` gets `[vocab]` float32.  `<out.f32>.h.npy` gets the per-layer hidden
states for the last token, `[N+2, hidden]` (embed, after each layer, final normed).
"""

import json
import mmap
import os
import struct
import sys

import numpy as np
import torch

SNAP = os.environ.get(
    "K3_DIR",
    "/home/lava/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
)
NL = int(os.environ.get("K3_NLAYERS", "1"))
PROMPT = [int(x) for x in os.environ.get("K3_PROMPT", "1008,10484,318,15383,387").split(",")]
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/k3ref_logits.f32"
BF = os.environ.get("K3_BF16", "1") == "1"   # round module outputs to bf16, as the ref does
# Deliberate MIS-wirings, one per suspected plow bug. Each is a hypothesis: if plow's logits
# match a variant to bf16 tolerance while disagreeing with the base, that variant IS the bug.
VAR = set(x for x in os.environ.get("K3_VAR", "").split(",") if x)
PARTS = []   # (name, [hidden]) components of the LAST token, for the basis fit

torch.set_grad_enabled(False)

_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "U8": torch.uint8}
_MM = {}


def _mmf(path):
    if path not in _MM:
        f = open(path, "rb")
        _MM[path] = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    return _MM[path]


def _index(snap):
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    hdrs, idx = {}, {}
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
    h, base, path = HDRS[IDX[name]]
    m = h[name]
    lo, hi = m["data_offsets"]
    return _mmf(path)[base + lo : base + hi], m


def load(name):
    buf, m = raw(name)
    return torch.frombuffer(bytearray(buf), dtype=_DT[m["dtype"]]).view(*m["shape"])


def loadf(name):
    return load(name).float()


with open(os.path.join(SNAP, "config.json")) as f:
    CFG = json.load(f)["text_config"]
LA = CFG["linear_attn_config"]
HID = CFG["hidden_size"]
H, D = LA["num_heads"], LA["head_dim"]
P = H * D
KW, LB = LA["short_conv_kernel_size"], LA["gate_lower_bound"]
EPS = CFG["rms_norm_eps"]
KSCALE = D ** -0.5
ARBS = CFG["attn_res_block_size"]
BETA, LBETA = CFG["activation_situ_beta"], CFG["activation_situ_linear_beta"]
NEXP, TOPK = CFG["num_experts"], CFG["num_experts_per_token"]
IMOE = CFG["moe_intermediate_size"]
HE = CFG["routed_expert_hidden_size"]
SHI = IMOE * CFG["num_shared_experts"]
RSCALE = CFG["routed_scaling_factor"]
FKDR = CFG["first_k_dense_replace"]
INTER = CFG["intermediate_size"]
VOCAB = CFG["vocab_size"]
# MLA
QL = CFG["q_lora_rank"]
KVL = CFG["kv_lora_rank"]
QKR = CFG["qk_rope_head_dim"]
QKN = CFG["qk_nope_head_dim"]
VHD = CFG["v_head_dim"]
NH = CFG["num_attention_heads"]
QHD = QKN + QKR
MSCALE = QHD ** -0.5
KDA1 = set(LA["kda_layers"])

PFX = "language_model.model."


def is_kda(l):
    return (l + 1) in KDA1


def bf(x):
    return x.to(torch.bfloat16).float() if BF else x


def rms(x, w, eps=EPS):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def situ(gate, up):
    gate, up = gate.float(), up.float()
    a = BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate)
    return a * (LBETA * torch.tanh(up / LBETA))


def apply_attn_res(prefix_sum, block_residual, score_w):
    """prefix_sum [T,H]; block_residual [T,nb,H]; score_w [H] folded."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1).float()
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)
    scores = (k * score_w).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v).squeeze(1), probs.squeeze(1)


# --------------------------------------------------------------------------- fla, vendored


def kda_gate(g, A_log, dt_bias):
    Hh = g.shape[-2]
    g = g.float() + dt_bias.view(g.shape[-2:])
    return LB * torch.sigmoid(A_log.view(Hh, 1).exp() * g)


def naive_recurrent_kda(q, k, v, g, beta, scale, initial_state=None):
    B, Tn, Hn, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    G = HV // Hn
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)
    S = q.new_zeros(B, HV, K, V)
    if initial_state is not None:
        S = S + initial_state
    o = torch.zeros_like(v)
    for i in range(Tn):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "b h k, b h v -> b h k v", b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2)
        )
        o[:, i] = torch.einsum("b h k, b h k v -> b h v", q_i, S)
    return o, S


E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
E2M1 = np.concatenate([E2M1, -E2M1])


def deq_mxfp4(name):
    pb, pm = raw(name + ".weight_packed")
    sb, sm = raw(name + ".weight_scale")
    packed = np.frombuffer(pb, np.uint8).reshape(*pm["shape"])
    scale = np.frombuffer(sb, np.uint8).reshape(*sm["shape"])
    N, KH = packed.shape
    K = KH * 2
    w = np.empty((N, K), np.float32)
    lo, hi = (1, 0) if "nibbleswap" in VAR else (0, 1)
    w[:, lo::2] = E2M1[packed & 0x0F]
    w[:, hi::2] = E2M1[packed >> 4]
    w *= np.repeat(
        np.ldexp(np.ones((N, K // 32), np.float32), scale.astype(np.int32) - 127), 32, axis=1
    )
    return torch.from_numpy(w)


def q_mxfp4(x):
    """OCP MX quantization of an ACTIVATION row to E2M1 with one E8M0 scale per 32."""
    v = x.reshape(-1, 32).numpy().astype(np.float32)
    amax = np.abs(v).max(-1, keepdims=True)
    exp = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-30))) - 2.0, 0.0)
    sc = np.exp2(exp)
    y = v / sc
    lv = E2M1[:8]
    idx = np.abs(y[..., None] - lv).argmin(-1)
    q = np.sign(y) * lv[idx]
    return torch.from_numpy((q * sc).reshape(x.shape))


# --------------------------------------------------------------------------- mixers


CARRY = {"S": None, "conv": None}


def kda_mixer(l, x):
    """x: [T, HID] pre-normed. Returns [T, HID]."""
    at = f"{PFX}layers.{l}.self_attn."
    T = x.shape[0]
    Wq, Wk, Wv = (loadf(at + n + "_proj.weight") for n in ("q", "k", "v"))
    Wg = loadf(at + "g_proj.weight")
    Wo = loadf(at + "o_proj.weight")
    Wfa, Wfb = loadf(at + "f_a_proj.weight"), loadf(at + "f_b_proj.weight")
    Wb = loadf(at + "b_proj.weight")
    convw = [loadf(at + f"{s}_conv1d.weight").view(P, KW) for s in ("q", "k", "v")]
    o_norm = loadf(at + "o_norm.weight")
    A_raw = loadf(at + "A_log")
    A_log = A_raw[:H].contiguous()
    dt_bias = loadf(at + "dt_bias")

    q_raw, k_raw, v_raw = x @ Wq.T, x @ Wk.T, x @ Wv.T
    g_hat = x @ Wg.T
    f_raw = (x @ Wfa.T) @ Wfb.T
    b_raw = x @ Wb.T

    def short_conv(xin, weight, cs):
        c = cs.clone() if cs is not None else torch.zeros(P, KW)
        out = torch.empty(T, P)
        for t in range(T):
            c = torch.roll(c, -1, dims=-1)
            c[:, KW - 1] = xin[t]
            out[t] = (c * weight).sum(-1)
        return torch.nn.functional.silu(out), c

    cin = (CARRY["conv"] or [None]*3) if "conv_carry" in VAR else [None, None, None]
    _o = [short_conv(r, w, c) for r, w, c in
          ((q_raw, convw[0], cin[0]), (k_raw, convw[1], cin[1]), (v_raw, convw[2], cin[2]))]
    CARRY["conv"] = [t[1] for t in _o]
    qc, kc, vc = (bf(t[0]) for t in _o)
    gg = kda_gate(f_raw.view(T, H, D), A_log, dt_bias)
    beta = torch.sigmoid(b_raw)
    qh, kh = qc.view(T, H, D), kc.view(T, H, D)
    qn = qh / torch.sqrt(qh.pow(2).sum(-1, keepdim=True) + 1e-6)
    kn = kh / torch.sqrt(kh.pow(2).sum(-1, keepdim=True) + 1e-6)
    s0 = CARRY["S"] if "kda_carry" in VAR else None
    o, Sf = naive_recurrent_kda(qn[None], kn[None], vc.view(T, H, D)[None], gg[None],
                               beta[None], KSCALE, initial_state=s0)
    CARRY["S"] = Sf
    oh = o[0]
    y = oh * torch.rsqrt(oh.pow(2).mean(-1, keepdim=True) + EPS) * o_norm
    sg = torch.sigmoid(g_hat)
    print(f"    L{l} gate sigmoid: mean={sg.mean():.4f} last-token mean={sg[-1].mean():.4f} "
          f"|y|/|y_nogate| would be {(y.view(T,P)[-1]*sg[-1]).norm()/y.view(T,P)[-1].norm():.4f}")
    if "nogate" in VAR and l > 0:
        sg = torch.ones_like(sg)
    if "gate_silu" in VAR and l > 0:
        sg = torch.nn.functional.silu(g_hat)
    y = bf(y.view(T, P) * sg)
    return bf(y @ Wo.T)


def mla_mixer(l, x):
    """x: [T, HID] pre-normed. Dense causal MLA, NoPE, output gate."""
    at = f"{PFX}layers.{l}.self_attn."
    T = x.shape[0]
    Wqa = loadf(at + "q_a_proj.weight")
    qan = loadf(at + "q_a_layernorm.weight")
    Wqb = loadf(at + "q_b_proj.weight")
    Wkva = loadf(at + "kv_a_proj_with_mqa.weight")
    kvan = loadf(at + "kv_a_layernorm.weight")
    Wkvb = loadf(at + "kv_b_proj.weight")
    Wo = loadf(at + "o_proj.weight")
    Wg = loadf(at + "g_proj.weight")

    q = ((rms(x @ Wqa.T, qan)) @ Wqb.T).view(T, NH, QHD)
    q_pass, q_rot = q[..., :QKN], q[..., QKN:]
    ckv = x @ Wkva.T
    k_pass_l, k_rot = ckv[:, :KVL], ckv[:, KVL:]
    kp = (rms(k_pass_l, kvan) @ Wkvb.T).view(T, NH, QKN + VHD)
    k_pass, v = kp[..., :QKN], kp[..., QKN:]
    k_rot_e = k_rot[:, None, :].expand(T, NH, QKR)
    qs = torch.cat((q_pass, q_rot), -1)            # [T, NH, 192]
    ks = torch.cat((k_pass, k_rot_e), -1)
    scores = torch.einsum("qhd,khd->hqk", qs, ks) * MSCALE
    mask = torch.full((T, T), float("-inf")).triu(1)
    scores = scores + mask
    probs = scores.softmax(-1)
    o = torch.einsum("hqk,khd->qhd", probs, v).reshape(T, NH * VHD)
    o = o * torch.sigmoid(x @ Wg.T)
    return bf(o @ Wo.T)


def latent_moe(l, x):
    """x: [T, HID] post-attention-normed. Returns [T, HID]."""
    mo = f"{PFX}layers.{l}.block_sparse_moe."
    T = x.shape[0]
    Wr = loadf(mo + "gate.weight")
    rb = loadf(mo + "gate.e_score_correction_bias")
    Wd = loadf(mo + "routed_expert_down_proj.weight")
    ln = loadf(mo + "routed_expert_norm.weight")
    Wu = loadf(mo + "routed_expert_up_proj.weight")

    logits = x.float() @ Wr.T
    scores = torch.sigmoid(logits)
    sel = scores + rb
    idx = torch.topk(sel, k=TOPK, dim=-1, sorted=False)[1]
    w = (sel if "biasw" in VAR else scores).gather(1, idx)
    if "norenorm" not in VAR:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    w = w * RSCALE

    xe = bf(x @ Wd.T)                                     # [T, HE]
    y = torch.zeros(T, HE)
    cache = {}
    for t in range(T):
        for s in range(TOPK):
            e = int(idx[t, s])
            if e not in cache:
                b = f"{mo}experts.{e}."
                g_n, u_n = ("w3", "w1") if "swapgu" in VAR else ("w1", "w3")
                cache[e] = (deq_mxfp4(b + g_n), deq_mxfp4(b + u_n), deq_mxfp4(b + "w2"))
            W1, W3, W2 = cache[e]
            xt = q_mxfp4(xe[t]) if "a4w4" in VAR else xe[t]   # A operand of gate/up
            a = bf(situ(bf(xt @ W1.T), bf(xt @ W3.T)))
            if "a4w4" in VAR or "qfu" in VAR:
                a = q_mxfp4(a)
            y[t] += float(w[t, s]) * bf(a @ W2.T)
    y = bf(y)
    if "nolatnorm" in VAR:
        y = bf(y @ Wu.T)
    elif "normafterup" in VAR:
        y = bf(rms(bf(y @ Wu.T), torch.ones(HID)))
    else:
        y = bf(bf(rms(y, ln)) @ Wu.T)

    sh = f"{mo}shared_experts."
    Wsg, Wsu, Wsd = (loadf(sh + n + "_proj.weight") for n in ("gate", "up", "down"))
    if "swapshgu" in VAR:
        Wsg, Wsu = Wsu, Wsg
    s = bf(situ(bf(x @ Wsg.T), bf(x @ Wsu.T)))
    sd = bf(s @ Wsd.T)
    PARTS.append(("routed", y[-1].clone()))
    PARTS.append(("shared", sd[-1].clone()))
    if "noshared" in VAR:
        return y
    if "nolatent" in VAR:
        return sd
    return y + sd


def dense_mlp(l, x):
    lp = f"{PFX}layers.{l}.mlp."
    Wg, Wu, Wd = (loadf(lp + n + "_proj.weight") for n in ("gate", "up", "down"))
    a = bf(situ(bf(x @ Wg.T), bf(x @ Wu.T)))
    return bf(a @ Wd.T)


# --------------------------------------------------------------------------- forward

ids = torch.tensor(PROMPT, dtype=torch.long)
T = len(PROMPT)
emb_h, emb_m = raw(PFX + "embed_tokens.weight")
emb_np = np.frombuffer(emb_h, np.uint16).reshape(*emb_m["shape"])
hidden = torch.from_numpy(
    (emb_np[ids.numpy()].astype(np.uint32) << 16).view(np.float32).copy()
)
print(f"N={NL} T={T} |embed|={hidden.norm():.4f}")

traces = [hidden[-1].clone()]
block_residual = torch.zeros(T, 0, HID)
prefix_sum = hidden

for l in range(NL):
    lp = f"{PFX}layers.{l}."
    sa_w = loadf(lp + "self_attention_res_norm.weight") * loadf(
        lp + "self_attention_res_proj.weight").squeeze(0)
    mlp_w = loadf(lp + "mlp_res_norm.weight") * loadf(
        lp + "mlp_res_proj.weight").squeeze(0)

    if "swapfolds" in VAR:
        sa_w, mlp_w = mlp_w, sa_w
    if "sa_w_norm" in VAR:
        sa_w = loadf(lp + "self_attention_res_norm.weight")
    if "sa_w_proj" in VAR:
        sa_w = loadf(lp + "self_attention_res_proj.weight").squeeze(0)
    if "sa_from_mlp" in VAR:
        sa_w = mlp_w
    if "sa_from_l0" in VAR:
        sa_w = loadf(f"{PFX}layers.0.self_attention_res_norm.weight") * loadf(
            f"{PFX}layers.0.self_attention_res_proj.weight").squeeze(0)
    hs = prefix_sum
    if block_residual.shape[1] > 0 and "nosares" not in VAR:
        hs, _ = apply_attn_res(bf(prefix_sum), bf(block_residual), sa_w)
        hs = bf(hs)
    if l % ARBS == 0:
        block_residual = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)
        prefix_sum = None

    hn = bf(rms(hs, loadf(lp + "input_layernorm.weight")))
    if l > 0 and os.environ.get("K3_HA_SCALE"):
        hn = bf(hn * float(os.environ["K3_HA_SCALE"]))
    a = kda_mixer(l, hn) if is_kda(l) else mla_mixer(l, hn)
    if "prefix_from_mix" in VAR and prefix_sum is not None:
        prefix_sum = hs + a
    elif "noprefixadd" in VAR:
        prefix_sum = a
    else:
        prefix_sum = a if prefix_sum is None else prefix_sum + a

    h2, _ = apply_attn_res(bf(prefix_sum), bf(block_residual), mlp_w)
    h3 = bf(rms(bf(h2), loadf(lp + "post_attention_layernorm.weight")))
    if "noffn" in VAR:
        f = torch.zeros(T, HID)
    else:
        f = latent_moe(l, h3) if l >= FKDR else dense_mlp(l, h3)
    prefix_sum = prefix_sum + f
    PARTS.append((f"attn{l}", a[-1].clone()))
    PARTS.append((f"ffn{l}", f[-1].clone()))
    traces.append(prefix_sum[-1].clone())
    print(f"  L{l:<3d} {'KDA' if is_kda(l) else 'MLA'} "
          f"{'MoE' if l >= FKDR else 'dense'}  |attn|={a.norm():.4f} "
          f"|ffn|={f.norm():.4f} |prefix|={prefix_sum.norm():.4f} nb={block_residual.shape[1]}")

ores = loadf(PFX + "output_attn_res_norm.weight") * loadf(
    PFX + "output_attn_res_proj.weight").squeeze(0)
h, probs = apply_attn_res(bf(prefix_sum), bf(block_residual), ores)
print(f"  output AttnRes probs (last token): {[round(v,4) for v in probs[-1].tolist()]}")
h = bf(rms(bf(h), loadf(PFX + "norm.weight")))
traces.append(h[-1].clone())

head_b, head_m = raw("language_model.lm_head.weight")
head = np.frombuffer(head_b, np.uint16).reshape(*head_m["shape"])
row = h[-1].numpy()
logits = np.empty(VOCAB, np.float32)
CH = 16384
for i in range(0, VOCAB, CH):
    w = (head[i : i + CH].astype(np.uint32) << 16).view(np.float32)
    logits[i : i + CH] = w @ row

logits.tofile(OUT)
np.save(OUT + ".h.npy", torch.stack(traces).numpy())
np.savez(OUT + ".parts.npz", embed=traces[0].numpy(),
         **{f"{i}_{n}": v.numpy() for i, (n, v) in enumerate(PARTS)})
top = np.argsort(-logits)[:10]
print(f"top10: {[(int(t), round(float(logits[t]), 3)) for t in top]}")
print(f"wrote {OUT}  ({VOCAB} f32)")
