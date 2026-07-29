#!/usr/bin/env python3
# k3_mla_oracle.py — REAL-WEIGHT reference for a COMPLETE Kimi-K3 **GATED MLA** BLOCK.  [K3-MLA-GATE]
#
# Rung 3, and the last of K3's three block types. Rung 1 (k3_real_oracle.py, layer 0) covered
# KDA + one AttnRes + a dense `situ` FFN; rung 2 (k3_moe_oracle.py, layer 1) covered both AttnRes
# applications + KDA + Stable LatentMoE. This is layer 3 — the same block SHAPE as rung 2 with the
# mixer swapped:
#
#     AttnRes -> **Gated MLA** -> AttnRes -> Stable LatentMoE
#
# 24 of the 93 layers are this. `linear_attn_config.full_attn_layers` is **1-BASED**
# (`configuration_kimi_k3.py::is_kda_layer` tests `(layer_idx + 1) in kda_layers`), so 0-based 3 is
# the first MLA layer — and the list's tail is `... 88, 92, 93`, i.e. 0-based 91 AND 92 are both
# MLA. An `i % 4 == 3` rule gets the LAST block of the model wrong; this file asserts membership
# against the list rather than deriving it.
#
# ======================================================================================
# THE FOUR THINGS THIS RUNG EXISTS TO PROVE, and where each one is falsifiable
# ======================================================================================
#
# 1. **MLA ABSORPTION, fed to a kernel for the first time.** `scripts/kimi_k3_prep.py` computes
#    `Wqa = einsum('hpl,hpk->hlk', k_nope, q_nope)` and `Wuv = value^T` and verified them
#    numerically (max abs err 1.0e-8 on q, 0.0 on v) — but nothing had ever consumed the output.
#    plow's MLA decode NEVER materializes q_pass/k_pass/value: it scores the absorbed query against
#    the 512-wide LATENT and folds W_uv on the way out. So this file computes the attention BOTH
#    ways — the model's own definition (materialize k_pass and value per cached row) and the
#    absorbed form the device runs — and reports their difference as its own row. The residual
#    table is scored against the MODEL'S definition, because that is what "correct" means; the
#    absorbed-vs-unabsorbed row says how much of any error is the absorption itself.
#    K3's dims are NOT GLM's: q_lora 1536 (GLM 2048), qk_nope 128 (192), v_head 128 (256).
#
# 2. **The MLA OUTPUT GATE** (`mla_use_output_gate: true`), gap #5, never touched.
#        g = self.g_proj(hidden_states).sigmoid();  attn_output = attn_output * g;  o_proj(...)
#    `g_proj` is [12288, 7168] and reads `hidden_states` — the post-`input_layernorm` MLA INPUT, NOT
#    the attention output. Feeding it the attention output has the right shape and the wrong model,
#    so the gated and ungated tensors are BOTH written and the harness diffs both.
#
# 3. **NoPE**, gap #6 — AND IT IS NOT A REMOVAL. `mla_use_nope: true`, `rotary_emb = None`, and
#    `forward` concatenates `q_rot`/`k_rot` back UNROTATED: the 64 "rope" dims are extra CONTENT
#    channels. But the k-side `HeadNormRope` is the ONLY writer of the `kv.{l}.krot` cache row AND
#    is the instruction `plowrt::exec::amd`'s kv-row-writer classifier and `glm52_decode.c:419`
#    both SCAN FOR to patch that row's position each step — delete it and the scan silently finds
#    fewer layers, with no count check. So the WRITE is kept and the ROTATION removed with an
#    **identity cos=1 / sin=0 table**, and the harness asserts the op is a BIT-EXACT copy rather
#    than merely a small residual. A table that were only nearly-identity would pass a 1.5e-2 check
#    while quietly rotating.
#
# 4. `d_mla_merge_fold`'s **V=128 dispatch** — a perf defect, not a correctness one, so it is not a
#    row here. It is measured compile-side (`scripts/k3_rung3_regcheck.sh`) and fixed in
#    `interp.hip`; this gate's job is to prove the fixed arm still computes the right numbers.
#
# ======================================================================================
# WHERE THIS GATE HAS TO LOOK — the rung-2 finding, unchanged and still load-bearing
# ======================================================================================
# Layer 3 takes no block-residual snapshot (3 % 12 != 0), so `prefix = prefix_in + attn` and the
# block output is `prefix_in + attn + moe` — EXACTLY what a plain-residual wiring produces. Rung 2
# measured the difference at 3.0e-3 at the block output against 8.1e-1 at the AttnRes outputs
# themselves. **A block-output-only gate does not see AttnRes at 85 of 93 layers.** The controls
# are therefore taken at the sub-layer INPUTS, and the same discipline is applied to the two new
# things here: the output gate gets a gated-vs-ungated control, and NoPE gets a bit-exactness
# assertion plus a "what a real RoPE would have done" number.
#
#   K3_DIR=<snapshot>  K3_MLA_LAYER=<0-based, must be MLA + MoE + no snapshot>  K3_MLA_CTX=<L>
#   python3 k3_mla_oracle.py <out.bin>

import json
import os
import struct
import sys
import mmap

import numpy as np
import torch

MAGIC = 0x4B334D41  # "K3MA"

SNAP = os.environ.get(
    "K3_DIR",
    "/home/lava/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
)
LAYER = int(os.environ.get("K3_MLA_LAYER", "3"))
L = int(os.environ.get("K3_MLA_CTX", "64"))       # KV rows; the current token is row L-1
T = 1
OUT = sys.argv[1] if len(sys.argv) > 1 else "k3_mla_fixture.bin"

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
    """Read the safetensors HEADERS only, via mmap. The checkpoint is ~1.5 TB across 96 shards."""
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
NH = CFG["num_attention_heads"]
QL = CFG["q_lora_rank"]
DK = CFG["kv_lora_rank"]
DR = CFG["qk_rope_head_dim"]
QN = CFG["qk_nope_head_dim"]
VD = CFG["v_head_dim"]
EPS = CFG["rms_norm_eps"]
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
SCALE = float(QN + DR) ** -0.5           # q_head_dim ** -0.5 = 192 ** -0.5

# The layer must be MLA, MoE, and NOT a snapshot layer — the same three preconditions rung 2 pinned,
# with "KDA" flipped to "MLA". Membership is tested against the CONFIG LIST, 1-based, never derived
# from a modulus: the tail of `full_attn_layers` is `88, 92, 93`, so 0-based 91 and 92 are BOTH MLA
# and any `i % 4` rule is wrong at the end of the model.
assert (LAYER + 1) in LA["full_attn_layers"], f"layer {LAYER} (1-based {LAYER+1}) is not MLA"
assert (LAYER + 1) not in LA["kda_layers"], "the two layer lists must partition 0..93"
assert LAYER >= FKDR, f"layer {LAYER} is dense, not MoE"
assert LAYER % ARBS != 0, (
    f"layer {LAYER} takes a block-residual snapshot, which resets prefix_sum to None and turns the "
    "attn-side AttnRes off — rung 1's shape, not this one's"
)
assert CFG["mla_use_nope"] is True, "this gate implements NoPE; the config must ask for it"
assert CFG["mla_use_output_gate"] is True, "this gate implements the output gate"
assert "rope_theta" not in CFG and "rope_scaling" not in CFG, (
    "a theta in the config would mean the MLA rotates after all"
)
assert CFG["moe_router_activation_func"] == "sigmoid" and CFG["moe_renormalize"]
assert CFG["num_expert_group"] == 1 and CFG["topk_group"] == 1
assert CFG["latent_moe_use_norm"] and HE != HID
assert L >= 8, "a KV cache with a couple of rows cannot exercise the softmax"

PFX = f"language_model.model.layers.{LAYER}."
AT = PFX + "self_attn."
MOE = PFX + "block_sparse_moe."

print(f"K3 GATED MLA block, layer {LAYER} (1-based {LAYER+1}): hidden={HID} heads={NH} "
      f"q_lora={QL} kv_lora={DK} qk={QN}+{DR} v_head={VD} ctx={L}")
print(f"  latent MoE: {NEXP} experts top-{TOPK} I_moe={IMOE} he={HE} shared_inter={SHI}")

# ---------------------------------------------------------------------------------------------
# Weights.

Wqa_down = load(AT + "q_a_proj.weight").float()               # [QL, HID]
gqa = load(AT + "q_a_layernorm.weight").float()               # [QL]
q_b = load(AT + "q_b_proj.weight").float()                    # [NH*(QN+DR), QL]
kv_a = load(AT + "kv_a_proj_with_mqa.weight").float()         # [DK+DR, HID]
gkva = load(AT + "kv_a_layernorm.weight").float()             # [DK]
kv_b = load(AT + "kv_b_proj.weight").float()                  # [NH*(QN+VD), DK]
Wg = load(AT + "g_proj.weight").float()                       # [NH*VD, HID]  THE OUTPUT GATE
Wo = load(AT + "o_proj.weight").float()                       # [HID, NH*VD]
ln_w = load(PFX + "input_layernorm.weight").float()
post_ln_w = load(PFX + "post_attention_layernorm.weight").float()

assert Wqa_down.shape == (QL, HID) and q_b.shape == (NH * (QN + DR), QL)
assert kv_a.shape == (DK + DR, HID) and kv_b.shape == (NH * (QN + VD), DK)
assert Wg.shape == (NH * VD, HID), (
    f"g_proj is {tuple(Wg.shape)}; mla_use_output_gate demands [n_head*v_head, hidden] = "
    f"[{NH*VD}, {HID}]"
)
assert Wo.shape == (HID, NH * VD)

attn_score_w = (load(PFX + "self_attention_res_norm.weight").float()
                * load(PFX + "self_attention_res_proj.weight").float().squeeze(0))
mlp_score_w = (load(PFX + "mlp_res_norm.weight").float()
               * load(PFX + "mlp_res_proj.weight").float().squeeze(0))
assert not torch.allclose(attn_score_w, mlp_score_w), "the two AttnRes folds are identical?"

Wrouter = load(MOE + "gate.weight").float()
rbias = load(MOE + "gate.e_score_correction_bias").float()
Wdown_l = load(MOE + "routed_expert_down_proj.weight").float()
lat_norm = load(MOE + "routed_expert_norm.weight").float()
Wup_l = load(MOE + "routed_expert_up_proj.weight").float()
Wshg = load(MOE + "shared_experts.gate_proj.weight").float()
Wshu = load(MOE + "shared_experts.up_proj.weight").float()
Wshd = load(MOE + "shared_experts.down_proj.weight").float()

# ---------------------------------------------------------------------------------------------
# THE ABSORPTION, exactly as scripts/kimi_k3_prep.py does it (which is exactly as glm52_prep.py
# does it). Verified there at max abs error 1.0e-8 on the q path and 0.0 on the v path; what is
# NEW here is that a kernel consumes the result.
#
#   q_absorbed[h,l] = sum_p k_nope[h,p,l] * q_nope[h,p]      so  q_absorbed . c_kv == q_pass . k_pass
#   o[h,v]          = sum_l olat[h,l] * Wuv[h,l,v]           so  the latent mix folds to v_head
#
# The split points come from the config, never from a hardcoded 192/256: K3's are 128/64 and 128,
# where GLM's are 192/64 and 256.

q_b_h = q_b.view(NH, QN + DR, QL)
q_b_nope, q_b_rope = q_b_h[:, :QN, :], q_b_h[:, QN:, :]
kv_b_h = kv_b.view(NH, QN + VD, DK)
k_nope_w, value_w = kv_b_h[:, :QN, :], kv_b_h[:, QN:, :]
Wqa = torch.einsum("hpl,hpk->hlk", k_nope_w, q_b_nope).contiguous()   # [NH, DK, QL]
Wuv = value_w.transpose(-1, -2).contiguous()                          # [NH, DK, VD]
W_ckv_down = kv_a[:DK]                                                # [DK, HID]
W_krot_down = kv_a[DK:DK + DR]                                        # [DR, HID]  raw, NEVER rotated
Wqr = q_b_rope.reshape(NH * DR, QL).contiguous()                      # raw, NEVER rotated


def bf(x):
    """Round through bf16 — the device stores bf16 between every packet."""
    return x.to(torch.bfloat16).float()


# The device reads the ABSORBED weights as bf16. Round them ONCE, here, and use the rounded copies
# for the absorbed reference — otherwise the absorbed-vs-unabsorbed row would be reporting the
# fixture's own quantization as if it were a device error.
Wqa_b = bf(Wqa)
Wuv_b = bf(Wuv)
Wqr_b = bf(Wqr)


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


def rms(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
E2M1 = np.concatenate([E2M1, -E2M1])


def deq_mxfp4(name):
    """Element 2i in the LOW nibble, E8M0 bias 127 — both CONFIRMED ON HARDWARE against these exact
    bytes (runtime/tests/k3_mxfp4_nibble_test.c: 1.6e-3 for this reading vs 1.4e0 for the swap)."""
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
# Inputs. `prefix_in` stands in for layer 2's output and `blkres` for layer 0's snapshot; the KV
# HISTORY is generated by pushing L-1 synthetic hidden states through the REAL kv_a projection and
# the REAL kv_a_layernorm, rather than by seeding the latent directly. That matters: the latent's
# magnitude is set by `kv_a_layernorm.weight`, and a cache seeded at the wrong scale makes the
# softmax either one-hot or uniform — in both cases the attention stops mixing and the gate stops
# testing anything. The control at the bottom asserts it did not.

gen = torch.Generator().manual_seed(0xB4)
prefix_in = (0.3 * torch.randn(T, HID, generator=gen)).to(torch.bfloat16)
blkres = (0.35 * torch.randn(T, 1, HID, generator=gen)).to(torch.bfloat16)
NB = 1

hist_h = (0.3 * torch.randn(L - 1, HID, generator=gen)).to(torch.bfloat16).float()
hist_kva = hist_h @ kv_a.T                                         # [L-1, DK+DR]
ckv_hist = bf(rms(hist_kva[:, :DK], gkva))                         # [L-1, DK]
krot_hist = bf(hist_kva[:, DK:])                                   # [L-1, DR]  UNROTATED

# ---------------------------------------------------------------------------------------------
# THE BLOCK.

# A0 — the ATTN-SIDE AttnRes.
h_a_f, probs_a = apply_attn_res(prefix_in, blkres, attn_score_w, EPS)
h_a = bf(h_a_f)

# A1 — input_layernorm. This is `hidden_states` as the reference's MLA forward sees it, and it is
# what BOTH the q/kv down-projections AND `g_proj` read.
x = bf(rms(h_a, ln_w))

# Q0/Q1 — q_a down, q_a_layernorm.
qlr = bf(x @ Wqa_down.T)                                           # [T, QL]
qlat = bf(rms(qlr, gqa))                                           # [T, QL]

# Q2 — the ABSORBED q_nope and the RAW q_rope. plow never materializes q_pass.
qa = bf(torch.einsum("k,hlk->hl", qlat[0], Wqa_b)).unsqueeze(0)     # [T, NH, DK]
qrr = bf(qlat @ Wqr_b.T).view(T, NH, DR)                            # [T, NH, DR]
# NoPE: `HeadNormRope` runs with an identity cos=1/sin=0 table, gamma absent and skip_norm set, so
# it is a BIT-EXACT copy. Not "approximately" — the harness asserts the device bytes are identical.
qr = qrr.clone()

# The MODEL's own q, for the unabsorbed reference.
q_pass = (qlat @ q_b_nope.reshape(NH * QN, QL).T).view(T, NH, QN)
q_rot = qrr                                                         # bf16-rounded, as the device has

# K0/K1 — the current token's KV row. Both go into the cache at position qpos.
kva_cur = bf(x @ kv_a.T)                                            # [T, DK+DR]
ckv_cur = bf(rms(kva_cur[:, :DK], gkva))                            # [T, DK]   -> kv.{l}.ckv[qpos]
krot_cur = bf(kva_cur[:, DK:])                                      # [T, DR]   -> kv.{l}.krot[qpos]

QPOS = L - 1
ckv_cache = torch.cat([ckv_hist, ckv_cur], dim=0)                   # [L, DK]
krot_cache = torch.cat([krot_hist, krot_cur], dim=0)                # [L, DR]
assert ckv_cache.shape == (L, DK) and krot_cache.shape == (L, DR)

# ------------------------------------------------------------------------------------------
# THE ATTENTION, computed TWICE.
#
# (a) THE MODEL'S DEFINITION — materialize k_pass and value per cached row from kv_b. This is
#     `KimiMLAAttention.forward` and it is what the residual table is scored against.
# (b) THE ABSORBED FORM — score the absorbed query against the raw latent, then fold W_uv. This is
#     what plow's FLASH_MLA_DECODE + MLA_MERGE_FOLD actually compute.
# Their difference is reported as its own row. It is NOT zero: `Wqa` is a sum over the 128 nope
# dims stored in bf16, so the absorption costs bf16-level precision on the score. That cost is the
# thing rung 3 is measuring for the first time.

k_pass = torch.einsum("td,hpd->thp", ckv_cache, k_nope_w)           # [L, NH, QN]
value = torch.einsum("td,hvd->thv", ckv_cache, value_w)             # [L, NH, VD]
sc_ref = (torch.einsum("hp,thp->ht", q_pass[0], k_pass)
          + torch.einsum("hd,td->ht", q_rot[0], krot_cache)) * SCALE       # [NH, L]
P_ref = torch.softmax(sc_ref, dim=-1)
oat_ref = torch.einsum("ht,thv->hv", P_ref, value)                   # [NH, VD]

sc_abs = (torch.einsum("hl,tl->ht", qa[0], ckv_cache)
          + torch.einsum("hd,td->ht", qr[0], krot_cache)) * SCALE
P_abs = torch.softmax(sc_abs, dim=-1)
olat = torch.einsum("ht,tl->hl", P_abs, ckv_cache)                   # [NH, DK] the latent output
oat_abs = torch.einsum("hl,hlv->hv", olat, Wuv_b)                    # [NH, VD]

oat = bf(oat_abs).reshape(T, NH * VD)          # what MLA_MERGE_FOLD writes: HEAD-MAJOR [nh][v]
d_absorb = (oat_abs - oat_ref).norm() / oat_ref.norm()

# ------------------------------------------------------------------------------------------
# G0 — THE OUTPUT GATE. `g_proj` reads `hidden_states` = the input_layernorm output `x`, NOT the
# attention output. It is sigmoid(g), not silu(g).

gl = bf(x @ Wg.T)                                                    # [T, NH*VD]
oatg = bf(oat * torch.sigmoid(gl))

# A2 — o_proj, on the GATED output.
attn = bf(oatg @ Wo.T)                                               # [T, HID]

# A3 — prefix_sum accumulates (no snapshot at this layer).
prefix = bf(prefix_in.float() + attn)

# A4/A5 — the MLP-SIDE AttnRes, with the OTHER fold, then post_attention_layernorm.
h2_f, probs_m = apply_attn_res(prefix, blkres, mlp_score_w, EPS)
h2 = bf(h2_f)
h3 = bf(rms(h2, post_ln_w))

# ---------------------------------------------------------------------------------------------
# STABLE LATENTMOE — identical to rung 2, re-run here because the block is the unit under test and
# because it is what consumes `h3`. Rows M0..M8 are the same rows rung 2 passed.

logit_f32 = h3.float() @ Wrouter.T
logit_bf = bf(logit_f32)


def route(logit):
    scores = torch.sigmoid(logit)
    sel_key = scores + rbias.unsqueeze(0)
    idx = torch.topk(sel_key, k=TOPK, dim=-1, sorted=True)[1]
    w = scores.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return idx, w * RSCALE


topk_idx, topk_w = route(logit_bf)
idx32, _ = route(logit_f32)
flip = int((torch.sort(topk_idx, -1)[0] != torch.sort(idx32, -1)[0]).sum())
print(f"  router: top-{TOPK} of {NEXP};  fp32-vs-bf16 logit selection differs in {flip} slot(s)")
print(f"  selected experts: {sorted(topk_idx[0].tolist())}")

# HOW RESOLVABLE IS THE RANKING? The selection key is `sigmoid(logit) + bias` and the device is
# given its OWN bf16 logits from an ordinary Gemv, so any pair of experts whose keys are closer
# together than that discrepancy can move them will rank in an arbitrary order. That is not a bug —
# `d_moe_combine` SUMS the slots, so the MoE output is order-invariant — but it does mean an
# ORDER-EXACT assertion is asserting torch's tie-break, so the margin is measured and printed here
# and the harness applies the corresponding bound. Rung 2's layer-1 input happened to have no near
# tie and matched order exactly; layer 3's has one, at slots 9/10.
_key = (torch.sigmoid(logit_bf) + rbias.unsqueeze(0))[0]
_sk = _key[topk_idx[0]]
_gaps = (_sk[:-1] - _sk[1:]).abs()
print(f"  router: selection-key gaps between adjacent ranks: min {float(_gaps.min()):.3e} "
      f"(at rank {int(_gaps.argmin())}), median {float(_gaps.median()):.3e} — a pair closer than "
      f"the bf16 logit discrepancy cannot be ordered by either party")

xe = bf(h3 @ Wdown_l.T)

sel = topk_idx[0].tolist()
gates = topk_w[0].tolist()
exp_bytes = []
fu_ref = torch.zeros(TOPK, IMOE)
part_ref = torch.zeros(TOPK, HE)
for j, e in enumerate(sel):
    base = MOE + f"experts.{e}."
    W1, w1p, w1s = deq_mxfp4(base + "w1")   # gate
    W3, w3p, w3s = deq_mxfp4(base + "w3")   # up
    W2, w2p, w2s = deq_mxfp4(base + "w2")   # down
    assert W1.shape == (IMOE, HE) and W2.shape == (HE, IMOE)
    fu_ref[j] = bf(situ(xe[0] @ W1.T, xe[0] @ W3.T, BETA, LBETA))
    part_ref[j] = gates[j] * (fu_ref[j] @ W2.T)     # the gate multiplies INSIDE the down kernel
    exp_bytes.append((e, w1p, w1s, w3p, w3s, w2p, w2s))

ylat = bf(part_ref.sum(0)).unsqueeze(0)
yn = bf(rms(ylat, lat_norm))
yh = bf(yn @ Wup_l.T)

shg = bf(h3 @ Wshg.T)
shu = bf(h3 @ Wshu.T)
sha = bf(situ(shg, shu, BETA, LBETA))
shd = bf(sha @ Wshd.T)

moe_out = bf(yh + shd)
block_out = bf(prefix.float() + moe_out)

print(f"  |prefix_in| {prefix_in.float().norm():.3f}  |x| {x.norm():.3f}  |qa| {qa.norm():.3f}  "
      f"|oat| {oat.norm():.3f}  |attn| {attn.norm():.3f}  |out| {block_out.norm():.3f}")
print(f"  AttnRes probs  attn-side {probs_a[0].tolist()}   mlp-side {probs_m[0].tolist()}")

# ---------------------------------------------------------------------------------------------
# CONTROLS. Each one exists because the row above it can be green while the thing it names is
# wrong, and each is printed by the harness next to the verdict.

# (1) The ABSORPTION. If it were wrong the device would still produce a finite, correctly-shaped
#     attention output — so the number that matters is how close the absorbed form is to the
#     model's own, and it has to be SMALL. A large value here means the einsum is wrong, not that
#     the kernel is.
print(f"  [absorption] absorbed vs the model's materialized k_pass/value: rel {d_absorb:.3e}")
assert d_absorb < 5e-3, (
    f"the absorbed attention disagrees with the model's own definition by {d_absorb:.3e} — that is "
    "the einsum or the split point, not a kernel"
)

# (2) THE ATTENTION MUST ACTUALLY MIX. A degenerate softmax (one-hot, or uniform over a cache of
#     identical rows) would make every downstream row pass while proving nothing about the scores.
#     The statistic is the MEAN over heads of each head's peak, not the max: with 96 heads a few
#     peaked ones are ordinary, and a max would make this control fire on a healthy input.
pk = P_ref.max(-1).values                                            # [NH]
ent = float(-(P_ref * (P_ref + 1e-30).log()).sum(-1).mean())
print(f"  [attention] per-head peak prob: mean {float(pk.mean()):.4f} max {float(pk.max()):.4f}, "
      f"{int((pk > 0.9).sum())} of {NH} heads above 0.9;  mean entropy {ent:.3f} "
      f"(uniform over {L} would be {np.log(L):.3f})")
assert float(pk.mean()) < 0.9, "the attention is one-hot — the KV history is not exercising the softmax"
assert ent > 0.3 * np.log(L), "the attention is nearly uniform — the scores are not discriminating"

# (3) THE OUTPUT GATE MUST MOVE THE OUTPUT. `sigmoid` is bounded in (0,1) and if `g_proj` happened
#     to be large and positive everywhere the gate would be ~1 and the row would be vacuous.
d_gate = (oatg - oat).norm() / oat.norm()
sig = torch.sigmoid(gl)
print(f"  [output gate] sigmoid(g) in [{float(sig.min()):.4f}, {float(sig.max()):.4f}], "
      f"mean {float(sig.mean()):.4f};  gated vs UNGATED: rel {d_gate:.3e} (must be LARGE)")
assert d_gate > 0.1, "the output gate is a no-op on this input — the gate row would prove nothing"

# (4) NoPE. Two numbers. The first is the claim: with an identity table the rotation is the
#     identity, BIT-EXACTLY, on both the q side and the k side. The second is what would have
#     happened if plow had applied GLM's rotation to dims the model treats as content — the error
#     that is invisible to every shape and symbol check, and which grows with position.
assert torch.equal(qr, qrr), "NoPE: q_rope is not a bit-exact copy in the reference itself"


def rope_interleave(v, pos, theta=8e6):
    """GPT-J interleaved RoPE at `pos` — what plow WOULD apply, and what K3 must not.

    `pos` broadcasts over the leading dims, so the cache is rotated ROW BY ROW at each row's own
    position. That is not a detail: rotating q and every k at the SAME angle is an orthogonal map
    and leaves every dot product EXACTLY unchanged, so a control written that way measures 1e-7 and
    concludes RoPE is harmless. It was written that way first. RoPE is a RELATIVE encoding; the
    error only appears when key `t` is rotated by `t` and the query by `qpos`."""
    d = v.shape[-1]
    j = torch.arange(d // 2, dtype=torch.float64)
    inv = 1.0 / theta ** (2.0 * j / d)
    a = torch.as_tensor(pos, dtype=torch.float64).unsqueeze(-1) * inv
    c, s = a.cos().float(), a.sin().float()
    ev, od = v[..., 0::2], v[..., 1::2]
    out = torch.empty_like(v)
    out[..., 0::2] = ev * c - od * s
    out[..., 1::2] = od * c + ev * s
    return out


k_roped = rope_interleave(krot_cache, torch.arange(L, dtype=torch.float64))   # row t at position t
q_roped = rope_interleave(q_rot[0], float(QPOS))
d_rope = (k_roped - krot_cache).norm() / krot_cache.norm()
sc_roped = (torch.einsum("hp,thp->ht", q_pass[0], k_pass)
            + torch.einsum("hd,td->ht", q_roped, k_roped)) * SCALE
oat_roped = torch.einsum("ht,thv->hv", torch.softmax(sc_roped, dim=-1), value)
d_rope_out = (oat_roped - oat_ref).norm() / oat_ref.norm()
print(f"  [NoPE] rotating each cached k_rot at ITS OWN position would move the cache by rel "
      f"{d_rope:.3e} and the attention output by rel {d_rope_out:.3e} — the error NoPE avoids, and "
      f"NOTHING downstream can see it (it is a valid-looking attention over the wrong scores)")
assert d_rope_out > 0.05, (
    "a rotation would barely change the output, so the NoPE rows would not be falsifiable — raise "
    "K3_MLA_CTX, or check that the control rotates q and k at DIFFERENT positions"
)

# (5) AttnRes, at the two points where it acts. Rung 2's finding: at a non-snapshot layer the BLOCK
#     OUTPUT cannot tell AttnRes from a plain residual.
plain_out = bf(prefix_in.float() + attn + moe_out)
d_plain = (plain_out - block_out).norm() / block_out.norm()
d_ha = (h_a - prefix_in.float()).norm() / prefix_in.float().norm()
d_h2 = (h2 - prefix.float()).norm() / prefix.float().norm()
print(f"  [control] block out vs the PLAIN wiring: rel {d_plain:.3e} — SMALL, and that is the "
      f"point: at a non-snapshot layer the block output cannot see this")
print(f"  [control] attn-side AttnRes moves its input by rel {d_ha:.3e};  mlp-side by rel {d_h2:.3e}"
      f"  (both must be LARGE)")
assert d_ha > 0.1 and d_h2 > 0.1, "AttnRes is a no-op here — the gate would prove nothing"

d_shared = (yh - moe_out).norm() / moe_out.norm()
assert d_shared > 0.05, "the shared expert is negligible here"

# ---------------------------------------------------------------------------------------------
# Fixture.


def w_bf(f, t):
    f.write(np.ascontiguousarray(t.to(torch.bfloat16).view(torch.uint16).numpy()).tobytes())


def w_f32(f, t):
    f.write(np.ascontiguousarray(t.float().numpy().astype(np.float32)).tobytes())


# The IDENTITY cos/sin tables, [L][DR/2] f32. Written by the oracle rather than built in C so that
# exactly one place decides what "no rotation" means, and so the harness's bit-exactness assertion
# is testing the DEVICE and not its own table constructor.
cos_id = torch.ones(L, DR // 2)
sin_id = torch.zeros(L, DR // 2)

with open(OUT, "wb") as f:
    f.write(struct.pack("<16i", MAGIC, T, NH, HID, QL, DK, DR, QN, VD, L, QPOS, IMOE, HE, NEXP,
                        TOPK, len(sel)))
    f.write(struct.pack("<8i", SHI, NB, 1 | 2 | 4, 4, 8, 0, 0, 0))   # gf=4, nsplit=8
    f.write(struct.pack("<6f", EPS, SCALE, BETA, LBETA, RSCALE, 0.0))
    # inputs
    w_bf(f, prefix_in)
    w_bf(f, blkres.view(T * NB, HID))
    w_f32(f, attn_score_w)
    w_f32(f, mlp_score_w)
    w_bf(f, ln_w)
    w_bf(f, post_ln_w)
    # MLA weights, in the DERIVED form kimi_k3_prep.py emits
    w_bf(f, Wqa_down)                       # [QL, HID]
    w_bf(f, gqa)                            # [QL]
    w_bf(f, Wqa.reshape(NH * DK, QL))       # absorbed q_nope
    w_bf(f, Wqr)                            # raw q_rope
    w_bf(f, W_ckv_down)                     # [DK, HID]
    w_bf(f, W_krot_down)                    # [DR, HID]
    w_bf(f, gkva)                           # [DK]
    w_bf(f, Wuv.reshape(NH * DK, VD))       # absorbed value
    w_bf(f, Wg)                             # [NH*VD, HID]  the output gate
    w_bf(f, Wo)                             # [HID, NH*VD]
    # KV cache HISTORY (rows 0..L-2); row L-1 is written by the device under test
    w_bf(f, ckv_hist)
    w_bf(f, krot_hist)
    w_f32(f, cos_id)
    w_f32(f, sin_id)
    # MoE weights
    w_bf(f, Wrouter)
    w_f32(f, rbias)
    w_bf(f, Wdown_l)
    w_bf(f, lat_norm)
    w_bf(f, Wup_l)
    w_bf(f, Wshg)
    w_bf(f, Wshu)
    w_bf(f, Wshd)
    f.write(np.array([e for e, *_ in exp_bytes], np.uint32).tobytes())
    for _, w1p, w1s, w3p, w3s, w2p, w2s in exp_bytes:
        for blob in (w1p, w1s, w3p, w3s, w2p, w2s):
            f.write(blob)
    # references
    w_bf(f, h_a)
    w_bf(f, x)
    w_bf(f, qlat)
    w_bf(f, qa.view(T, NH * DK))
    w_bf(f, qrr.view(T, NH * DR))           # == qr; the harness asserts the device matches BITWISE
    w_bf(f, ckv_cur)
    w_bf(f, krot_cur)
    w_bf(f, oat_ref.reshape(T, NH * VD))    # THE MODEL'S DEFINITION, not the absorbed form
    w_bf(f, gl)
    w_bf(f, oatg)
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
    sz = f.tell()

print(f"wrote {OUT}  ({sz / 1e6:.1f} MB, {len(sel)} experts materialized of {NEXP})")
