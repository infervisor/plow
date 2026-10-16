#!/usr/bin/env python3
# muse_block_oracle.py: a real-weight host oracle for a chain of Muse Spark decoder layers
# (`LAYERS` = l or l..r, r exclusive; decode, B1).
#
# It mirrors modeling_muse_spark.py (transformers) op by op in torch on the CPU. The layer weights
# come straight from the Quark MXFP4 checkpoint: bf16 tensors are used verbatim, and the routed
# experts are dequantized from fp4 e2m1 + e8m0/32 (W4A16, which is what the device runs). It writes
# the operands `plowrt amd-block --input-dir` consumes:
#   act.x.bin                 bf16 [H]                 the decode token's first-layer input (pos ctx-1)
#   kv.{L}.k.bin / kv.{L}.v.bin bf16 [kvh][ctx][hd]    head-major history, all kv heads (amd-block
#                                                      splits it per rank under --tp)
#   reference.bf16            act.xnext (last layer output)
#   <stage>.reference.bf16    the last layer's act.xmid / act.xn2 / act.ylat (+ act.attn at TP1)
#   reference.json            ctx, batch, tolerance, stages, router top-k, A4W4 delta
# The INPUT is a seeded synthetic hidden row (N(0,1), so input_layernorm sees unit RMS like the
# embed-normed stream). The WEIGHTS are real. Each layer's K/V history (rows 0..ctx-2) is its own
# projections of a per-layer seeded N(0,1) sequence; the decode row is the chained hidden. The
# histories need not be the previous layer's outputs: they are operands, and the gate is that the
# device computes the same chain from the same operands.
#
# --compare DUMP diffs an `amd-block --dump DUMP` run against the references. It prints max abs,
# max rel, rel-L2 and cosine per stage, and the router top-k agreement.
#
# Run with a torch Python. No GPU is needed, but expect a large RAM footprint.
import argparse, glob, json, os, struct, sys
import numpy as np
import torch

torch.set_grad_enabled(False)
BF16 = torch.bfloat16
P = "model.language_model.layers.{}."

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("layers")
ap.add_argument("out")
ap.add_argument("--ctx", type=int, default=4096)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--attn-f32", action="store_true",
                help="fused-attention numerics: qk_scale_factor folded into an f32 softmax scale, f32 "
                     "scores and probabilities, instead of eager HF's bf16 q*scale, bf16 score "
                     "matmul and bf16 probability cast")
ap.add_argument("--force-last", help="comma-separated expert ids: the last layer routes to these "
                "(score the continuous stages under a device near-tie routing flip)")
ap.add_argument("--tp", type=int, default=1, help="omit rank-local stages (act.attn) at TP > 1")
ap.add_argument("--compare", help="amd-block --dump directory to diff against OUT")
args = ap.parse_args()
lo_hi = [int(v) for v in args.layers.split("..")]
LAYERS = list(range(lo_hi[0], lo_hi[1] if len(lo_hi) > 1 else lo_hi[0] + 1))
CTX, OUT = args.ctx, args.out


def to_bytes(t):
    return np.ascontiguousarray(t.to(BF16).contiguous().view(torch.int16).numpy()).tobytes()


def from_bf16(path, shape=None):
    a = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    t = torch.from_numpy(a.view(np.float32).copy())
    return t if shape is None else t.reshape(shape)


def metrics(dev, ref):
    dev, ref = dev.double().flatten(), ref.double().flatten()
    err = (dev - ref).abs()
    return dict(
        max_abs=err.max().item(),
        max_rel=(err / ref.abs().clamp_min(1e-2 * ref.abs().max())).max().item(),
        rel_l2=(err.norm() / ref.norm()).item(),
        cosine=torch.nn.functional.cosine_similarity(dev, ref, dim=0).item(),
        ref_rms=ref.pow(2).mean().sqrt().item(),
    )


if args.compare:
    meta = json.load(open(os.path.join(OUT, "reference.json")))
    rows = {}
    for name in ["act.xnext"] + meta["stages"]:
        ref = from_bf16(os.path.join(OUT, "reference.bf16" if name == "act.xnext"
                                     else f"{name}.reference.bf16"))
        path = os.path.join(args.compare, f"rank0.{name}.bin")
        if not os.path.exists(path):
            continue
        dev = from_bf16(path)[: ref.numel()]
        rows[name] = metrics(dev, ref)
    tab = np.fromfile(os.path.join(args.compare, "rank0.act.tab.bin"), dtype=np.uint32)
    k = len(meta["router_topk"]["ids"])
    dev_ids = tab[: 2 * k : 2].tolist()
    dev_w = tab[1 : 2 * k : 2].view(np.float32).tolist()
    ref = dict(zip(meta["router_topk"]["ids"], meta["router_topk"]["weights"]))
    rows["router"] = dict(
        same_set=sorted(dev_ids) == sorted(ref),
        max_weight_err=max(abs(w - ref.get(i, 0.0)) for i, w in zip(dev_ids, dev_w)),
        device_ids=dev_ids,
    )
    rows["a4w4_vs_w4a16_xnext"] = meta["a4w4_vs_w4a16_xnext"]
    print(json.dumps(rows, indent=2))
    sys.exit(0)

cfg = json.load(open(os.path.join(args.model, "config.json")))
tc = cfg.get("text_config", cfg)
H, NH, NKV, HD = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
EPS, PEPS = tc["rms_norm_eps"], tc["post_norm_eps"]
E, K, M = tc["num_local_experts"], tc["num_experts_per_tok"], tc["moe_hidden_size"]
THETA = tc["rope_parameters"]["rope_theta"]

# Shard index from the headers (works on a partially downloaded checkpoint).
index = {}
for f in sorted(glob.glob(os.path.join(args.model, "*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        for name in json.loads(fh.read(n)):
            index[name] = f
from safetensors import safe_open

_handles = {}


def tensor(name):
    f = index.get(name)
    if f is None:
        sys.exit(f"missing {name} (checkpoint incomplete?)")
    if f not in _handles:
        _handles[f] = safe_open(f, framework="pt")
    return _handles[f].get_tensor(name)



FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant_mxfp4(packed, scale):
    # Quark/OCP packing: element 2i in the LOW nibble. Scale is e8m0, one byte per 32 along K.
    q = packed.to(torch.int64)
    vals = torch.stack([FP4[q & 15], FP4[q >> 4]], dim=-1).reshape(packed.shape[0], -1)
    s = torch.pow(2.0, scale.to(torch.float32) - 127.0)
    return (vals.reshape(vals.shape[0], -1, 32) * s[..., None]).reshape(vals.shape).to(BF16)


def quant_mxfp4_act(x):
    # Dynamic OCP MXFP4 activation quant (Quark `even` scale): e = floor(log2(amax)) - 2,
    # elements rounded half-even onto the e2m1 grid and clamped to +-6.
    g = x.float().reshape(-1, 32)
    amax = g.abs().amax(-1, keepdim=True)
    e = torch.where(amax > 0, torch.floor(torch.log2(amax)) - 2, torch.zeros_like(amax))
    y = (g / torch.pow(2.0, e)).clamp(-6, 6)
    grid = FP4[:8]
    dist = (y.abs()[..., None] - grid).abs()
    idx = dist.argmin(-1)  # the lower point on a tie; e2m1 mantissa parity = index parity
    tie = (dist == dist.min(-1, keepdim=True).values).sum(-1) > 1
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    q = grid[idx] * torch.sign(y)
    return (q * torch.pow(2.0, e)).reshape(x.shape).to(BF16)


def rms(x, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def norm_weightless(x, eps):        # MuseSparkRMSNorm(with_scale=False)
    xf = x.float()
    return (xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)).to(x.dtype)


def norm_centered(x, wt, eps):      # MuseSparkTextCenteredRMSNorm
    return (rms(x, eps) * (1.0 + wt).float()).to(x.dtype)


def norm_input_scaled(x, wt, eps):  # MuseSparkTextInputScaledRMSNorm
    y = (x.float() * (1.0 + wt).float()).to(x.dtype)
    return rms(y, eps).to(x.dtype)


def residual_gate(res, h, g):
    g = g.float() / tc["residual_gate_temperature"]
    beta = torch.sigmoid(g)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-g) * (1.0 + beta), min=1e-3))
    return (alpha * res + beta * h).to(res.dtype)


def lin(x, wt):
    return torch.nn.functional.linear(x, wt)


def rope(x, pos):                   # x [n, heads, hd] bf16, pos [n]
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float) / HD))
    fr = pos[:, None].float() * inv
    emb = torch.cat((fr, fr), -1)
    cos, sin = emb.cos().to(BF16)[:, None], emb.sin().to(BF16)[:, None]
    rot = torch.cat((-x[..., HD // 2:], x[..., : HD // 2]), -1)
    return x * cos + rot * sin


def layer_fwd(L, x, hist, force=None):
    """One decoder layer on the decode row `x` [1,H] with K/V history from `hist` [CTX-1,H]."""
    w = lambda n: tensor(P.format(L) + n)
    sliding = tc["layer_types"][L] == "sliding_attention"
    rope_on = tc["layer_rope_theta"][L] != 0
    window = tc["sliding_window"] if sliding else None
    xs = torch.cat([hist, x])
    pos = torch.arange(CTX)
    qpos = CTX - 1
    hn = norm_centered(xs, w("input_layernorm.weight"), EPS)
    k = norm_weightless(lin(hn, w("self_attn.k_proj.weight")).view(CTX, NKV, HD), EPS)
    v = lin(hn, w("self_attn.v_proj.weight")).view(CTX, NKV, HD)
    hq = hn[qpos : qpos + 1]
    q = norm_weightless(lin(hq, w("self_attn.q_proj.weight")).view(1, NH, HD), EPS)
    qk_scale = tc["qk_scale_factor"] * HD ** -0.5
    if not args.attn_f32:  # eager HF: q * qk_scale_factor rounds to bf16 before the score matmul
        q, qk_scale = q * tc["qk_scale_factor"], HD ** -0.5
    if rope_on:
        k = rope(k, pos)
        q = rope(q, pos[qpos : qpos + 1])
    lo = max(qpos - window + 1 if window else 0, 0)
    kk = k[lo : qpos + 1].transpose(0, 1).repeat_interleave(NH // NKV, 0)    # [NH, n, HD]
    vv = v[lo : qpos + 1].transpose(0, 1).repeat_interleave(NH // NKV, 0)
    qq = q.transpose(0, 1)                                                     # [NH, 1, HD]
    if args.attn_f32:
        sc = torch.matmul(qq.float(), kk.float().transpose(1, 2)) * qk_scale
    else:
        sc = torch.matmul(qq, kk.transpose(1, 2)) * qk_scale
    pr = torch.softmax(sc, dim=-1, dtype=torch.float32)
    if args.attn_f32:
        attn = torch.matmul(pr, vv.float()).to(BF16)
    else:
        attn = torch.matmul(pr.to(BF16), vv)
    attn = attn.transpose(0, 1).reshape(1, NH, HD)                            # [1, NH, HD]
    at = norm_weightless(attn, EPS).reshape(1, NH * HD)
    at = at * torch.sigmoid(lin(hq, w("self_attn.gate_proj.weight")))
    o_raw = lin(at, w("self_attn.o_proj.weight"))
    o = norm_weightless(o_raw, PEPS)
    xmid = residual_gate(x, o, w("post_attention_residual_gate.gate"))

    xn2 = norm_centered(xmid, w("pre_feedforward_layernorm.weight"), EPS)
    logits = lin(xn2.float(), w("mlp.gate.weight").float()) * tc["output_multiplier"]
    scores = torch.sigmoid(logits)
    bias = w("mlp.gate.e_score_correction_bias").float()
    top = torch.topk(scores + bias, k=K, dim=-1, sorted=False)[1][0]
    ranked = torch.sort((scores + bias)[0], descending=True).values
    margin = (ranked[K - 1] - ranked[K]).item()  # a device flip below ~1e-3 is a near tie
    if force is not None:
        top = torch.tensor(force)
    gates = scores[0, top]
    gates = gates / (gates.sum() + 1e-15) * tc["routed_scaling_factor"]
    xe = norm_centered(lin(xn2, w("mlp.pre_expert_proj.weight")), w("mlp.pre_expert_norm.weight"), EPS)
    w_pen = w("mlp.experts.post_expert_norm.weight")
    mx = (P.format(L) + "mlp.experts.0.gate_proj.weight_scale") in index

    def experts(a4w4):
        acc = torch.zeros(1, M, dtype=BF16)
        for j in torch.argsort(top).tolist():                 # HF loops experts in id order
            ep = f"mlp.experts.{top[j].item()}."
            wt = {n: dequant_mxfp4(w(ep + n + ".weight"), w(ep + n + ".weight_scale")) if mx
                  else w(ep + n + ".weight") for n in ("gate_proj", "up_proj", "down_proj")}
            xi = quant_mxfp4_act(xe) if a4w4 and mx else xe
            hcur = torch.nn.functional.silu(lin(xi, wt["gate_proj"])) * lin(xi, wt["up_proj"])
            hcur = quant_mxfp4_act(hcur) if a4w4 and mx else hcur
            y = norm_input_scaled(lin(hcur, wt["down_proj"]), w_pen, PEPS)
            acc = acc + (y * gates[j]).to(BF16)
        return acc

    def tail(ylat):
        y = lin(ylat, w("mlp.post_expert_proj.weight"))
        y = norm_input_scaled(y, w("post_feedforward_layernorm.weight"), PEPS)
        return residual_gate(xmid, y, w("post_feedforward_residual_gate.gate"))

    ylat = experts(False)
    xnext = tail(ylat)
    xnext_a4w4 = tail(experts(True))
    kind = ("sliding" if sliding else "full") + ("+rope" if rope_on else "+nope") + ("" if mx else "+bf16-experts")
    return dict(xnext=xnext, xnext_a4w4=xnext_a4w4, k=k, v=v, top=top, gates=gates, kind=kind,
                margin=margin,
                stages={"act.attn": attn.reshape(1, NH * HD), "act.at": at, "act.o": o_raw,
                        "act.on": o, "act.xmid": xmid, "act.xn2": xn2, "act.ylat": ylat})


torch.manual_seed(args.seed)
x0 = torch.randn(1, H).to(BF16)
os.makedirs(OUT, exist_ok=True)
x, kinds, margins = x0, [], []
for L in LAYERS:
    g = torch.Generator().manual_seed(args.seed + 1 + L)
    hist = torch.randn(CTX - 1, H, generator=g).to(BF16)
    force = [int(v) for v in args.force_last.split(",")] if args.force_last and L == LAYERS[-1] else None
    r = layer_fwd(L, x, hist, force)
    # head-major [kvh][ctx][hd]; row ctx-1 is rewritten by the device itself
    open(os.path.join(OUT, f"kv.{L}.k.bin"), "wb").write(to_bytes(r["k"].transpose(0, 1)))
    open(os.path.join(OUT, f"kv.{L}.v.bin"), "wb").write(to_bytes(r["v"].transpose(0, 1)))
    kinds.append(f"{L}:{r['kind']}")
    margins.append(r["margin"])
    print(f"layer {L} ({r['kind']}) ctx={CTX}: top{K}={r['top'].tolist()} margin={r['margin']:.2e} "
          f"xnext rms={r['xnext'].float().pow(2).mean().sqrt():.4f}", flush=True)
    x = r["xnext"]
open(os.path.join(OUT, "act.x.bin"), "wb").write(to_bytes(x0))
open(os.path.join(OUT, "reference.bf16"), "wb").write(to_bytes(x))
stages = {n: t for n, t in r["stages"].items() if args.tp == 1 or n not in ("act.attn", "act.at")}
for name, t in stages.items():
    open(os.path.join(OUT, f"{name}.reference.bf16"), "wb").write(to_bytes(t))
d = (r["xnext_a4w4"].double() - x.double()).flatten()
json.dump(dict(
    model=args.model, layers=LAYERS, ctx=CTX, batch=1, seed=args.seed, tp=args.tp, kinds=kinds,
    router_margins=margins, forced_last=args.force_last,
    reference="torch CPU mirror of modeling_muse_spark.py, W4A16 experts"
              + (", f32 attention scores/probabilities" if args.attn_f32 else ""),
    tolerance_rel_l2=0.03, stages=list(stages),
    router_topk=dict(ids=r["top"].tolist(), weights=r["gates"].tolist()),
    a4w4_vs_w4a16_xnext=dict(rel_l2=(d.norm() / x.double().norm()).item(),
                             max_abs=d.abs().max().item()),
), open(os.path.join(OUT, "reference.json"), "w"), indent=2)
print(f"layers {LAYERS} -> {OUT}")
