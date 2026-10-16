#!/usr/bin/env python3
# muse_block_oracle.py: a real-weight host oracle for a chain of Muse Spark decoder layers
# (`LAYERS` = l or l..r, r exclusive; decode B1, or a `--tokens T` prefill chunk).
#
# It mirrors modeling_muse_spark.py (transformers) op by op in torch on the CPU. The layer weights
# come straight from the Quark MXFP4 checkpoint: bf16 tensors are used verbatim, and the routed
# experts are dequantized from fp4 e2m1 + e8m0/32 (W4A16, which is what the device runs). It writes
# the operands `plowrt amd-block --input-dir` consumes:
#   act.x.bin                 bf16 [T][H]              the first-layer input rows (pos ctx-1 at decode,
#                                                      prefix..prefix+T-1 in prefill)
#   kv.{L}.k.bin / kv.{L}.v.bin bf16 [kvh][ctx][hd]    head-major history, all kv heads (amd-block
#                                                      splits it per rank under --tp); rows past the
#                                                      history are the reference's own (the device
#                                                      rewrites them)
#   reference.bf16            act.xnext (last layer output); reference_a4w4.bf16 the A4W4 experts' chain
#   <stage>.reference.bf16    the last layer's act.xmid / act.xn2 / act.ylat (+ act.attn at TP1)
#   reference.json            ctx, batch, tolerance, stages, dump (extra tensors `--dump` writes),
#                             router top-k, A4W4 delta
# The INPUT is a seeded synthetic hidden row (N(0,1), so input_layernorm sees unit RMS like the
# embed-normed stream). The WEIGHTS are real. Each layer's K/V history (rows 0..ctx-2) is its own
# projections of a per-layer seeded N(0,1) sequence; the decode row is the chained hidden. The
# histories need not be the previous layer's outputs: they are operands, and the gate is that the
# device computes the same chain from the same operands.
#
# --compare DUMP diffs an `amd-block --dump DUMP` run against the references. It prints max abs,
# max rel, rel-L2 and cosine per stage, and the router top-k agreement (per row in prefill).
#
# --tokens T (prefill): T rows at positions prefix..prefix+T-1 over a `--prefix`-row history,
# causal and windowed on sliding layers, experts routed per row.
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
ap.add_argument("--tokens", type=int, default=0, help="prefill rows (0 = the B1 decode row)")
ap.add_argument("--prefix", type=int, default=0, help="prefill history rows before the chunk")
ap.add_argument("--from-dump", help="amd-block --dump directory: run the experts on the device's "
                "act.xe and routing (isolates the grouped-MoE kernels from upstream rounding)")
ap.add_argument("--mx-rule", choices=("quark", "device", "aiter"), default="quark",
                help="A4W4 activation quant scale: Quark 'even' (the checkpoint's declared rule; "
                     "amax rounded up a binade at mantissa >= 1.75, then 2^(floor(log2)-2)), the old "
                     "device rule (smallest 2^e with amax/2^e <= 6), or AITER's RCEIL "
                     "(ceil_pow2(amax * fl(1/6))). Elements: half-even, saturating at 6 (device: ties "
                     "away from zero)")
ap.add_argument("--gain-f32", action="store_true",
                help="centered / input-scaled norm gain 1 + float(w) (vLLM, the device) instead of "
                     "HF's float(1 + w), whose bf16 add rounds the gain")
ap.add_argument("--rope-fused", action="store_true",
                help="RoPE as one f32 multiply-add per element (the device) instead of HF's bf16 ops")
args = ap.parse_args()
lo_hi = [int(v) for v in args.layers.split("..")]
LAYERS = list(range(lo_hi[0], lo_hi[1] if len(lo_hi) > 1 else lo_hi[0] + 1))
CTX, OUT = args.ctx, args.out
T = max(args.tokens, 1)
C0 = args.prefix if args.tokens else CTX - 1
assert C0 + T <= CTX, f"prefix {C0} + tokens {T} exceeds ctx {CTX}"


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


def ref_hd(meta):
    c = json.load(open(os.path.join(meta["model"], "config.json")))
    return c.get("text_config", c)["head_dim"]


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
        a4 = os.path.join(OUT, "reference_a4w4.bf16" if name == "act.xnext" else f"{name}.a4w4.bf16")
        if os.path.exists(a4):
            rows[f"{name}_vs_a4w4"] = metrics(dev, from_bf16(a4))
    for f in sorted(glob.glob(os.path.join(OUT, "kv.*.reference.bf16"))):
        name = os.path.basename(f)[: -len(".reference.bf16")]
        path = os.path.join(args.compare, f"rank0.{name}.bin")
        if os.path.exists(path):  # the chunk's rows of the rank's [kvh][ctx][hd] cache
            hd, c0, t = ref_hd(meta), meta["prefix"], meta["tokens"]
            ref = from_bf16(f).reshape(-1, t, hd)
            dev = from_bf16(path).reshape(-1, meta["ctx"], hd)[:, c0 : c0 + t]
            h = min(dev.shape[0], ref.shape[0])  # slot 0's heads; a TP rank's share
            rows[name] = metrics(dev[:h], ref[:h])
    tab = np.fromfile(os.path.join(args.compare, "rank0.act.tab.bin"), dtype=np.uint32)
    topk = meta["router_topk"]
    k = len(topk["ids"][0])
    same, werr, flips = 0, 0.0, []
    for r, (ids, ws) in enumerate(zip(topk["ids"], topk["weights"])):
        dev_ids = tab[2 * k * r : 2 * k * (r + 1) : 2].tolist()
        dev_w = tab[2 * k * r + 1 : 2 * k * (r + 1) : 2].view(np.float32).tolist()
        ref = dict(zip(ids, ws))
        if sorted(dev_ids) == sorted(ref):
            same += 1
            werr = max(werr, max(abs(w - ref[i]) for i, w in zip(dev_ids, dev_w)))
        else:
            flips.append(r)
    rows["router"] = dict(same_set_rows=same, rows=len(topk["ids"]), max_weight_err=werr,
                          flipped_rows=flips[:16])
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
    # Dynamic OCP MXFP4 activation quant, per 32 along the last axis. Quark `even`
    # (quark/torch/quantization/utils.py even_round + mxfp4/fake_kernels.cuh): bits(amax) + 2^21
    # truncated to a power of two (0 -> 2^-126), e = clamp(floor(log2) - 2, -127, 127); elements
    # x / 2^e rounded half-even onto e2m1, saturating at 6.
    g = x.float().reshape(-1, 32)
    amax = g.abs().amax(-1, keepdim=True)
    if args.mx_rule == "device":  # amd_common.h e8m0_for_amax / quant_fp4 (PLOW_MX_ACT_EVEN=0)
        e = torch.where(amax > 0, torch.frexp(amax / 6.0).exponent.float(), torch.zeros_like(amax))
        e = e.clamp(-126, 127)
    elif args.mx_rule == "aiter":  # mx_quant_utils.h RoundUp
        u = (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32)).view(torch.int32)
        be = (u >> 23) & 0xFF
        e = (be + ((be < 0xFF) & ((u & 0x7FFFFF) != 0)).int() - 127).float()
    else:
        r = ((amax.view(torch.int32).long() + (1 << 21)) & 0xFF800000).int().view(torch.float32)
        r = torch.where(amax == 0, torch.full_like(r, 2.0 ** -126), r)
        e = (torch.floor(torch.log2(r)) - 2).clamp(-127, 127)
    s = torch.pow(2.0, e.double())
    y = (g.double() / s).clamp(-6, 6)
    grid = FP4[:8].double()
    dist = (y.abs()[..., None] - grid).abs()
    idx = dist.argmin(-1)  # the lower point on a tie; e2m1 mantissa parity = index parity
    tie = (dist == dist.min(-1, keepdim=True).values).sum(-1) > 1
    idx = torch.where(tie & ((idx % 2 == 1) | (args.mx_rule == "device")), idx + 1, idx)
    q = torch.where(torch.signbit(g), -grid[idx], grid[idx])
    return (q * s).float().reshape(x.shape).to(BF16)


def rms(x, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def norm_weightless(x, eps):        # MuseSparkRMSNorm(with_scale=False)
    xf = x.float()
    return (xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)).to(x.dtype)


def gain(wt):
    return 1.0 + wt.float() if args.gain_f32 else (1.0 + wt).float()


def norm_centered(x, wt, eps):      # MuseSparkTextCenteredRMSNorm
    return (rms(x, eps) * gain(wt)).to(x.dtype)


def norm_input_scaled(x, wt, eps):  # MuseSparkTextInputScaledRMSNorm
    y = (x.float() * gain(wt)).to(x.dtype)
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
    if args.rope_fused:             # the device: f32 multiply-add, one bf16 round
        return (x.float() * cos.float() + rot.float() * sin.float()).to(BF16)
    return x * cos + rot * sin


def layer_fwd(L, x, hist, force=None):
    """One decoder layer on the rows `x` [T,H] at positions C0.. with K/V history `hist` [C0,H]."""
    w = lambda n: tensor(P.format(L) + n)
    sliding = tc["layer_types"][L] == "sliding_attention"
    rope_on = tc["layer_rope_theta"][L] != 0
    window = tc["sliding_window"] if sliding else None
    xs = torch.cat([hist, x])
    n = C0 + T
    pos = torch.arange(n)
    hn = norm_centered(xs, w("input_layernorm.weight"), EPS)
    k = norm_weightless(lin(hn, w("self_attn.k_proj.weight")).view(n, NKV, HD), EPS)
    v = lin(hn, w("self_attn.v_proj.weight")).view(n, NKV, HD)
    hq = hn[C0:]
    q = norm_weightless(lin(hq, w("self_attn.q_proj.weight")).view(T, NH, HD), EPS)
    qk_scale = tc["qk_scale_factor"] * HD ** -0.5
    if not args.attn_f32:  # eager HF: q * qk_scale_factor rounds to bf16 before the score matmul
        q, qk_scale = q * tc["qk_scale_factor"], HD ** -0.5
    if rope_on:
        k = rope(k, pos)
        q = rope(q, pos[C0:])
    qp = pos[C0:, None]
    keep = pos[None, :] <= qp                                                  # [T, n]
    if window:
        keep &= pos[None, :] > qp - window
    attn = torch.empty(NH, T, HD, dtype=BF16)
    G = NH // NKV
    for h0 in range(0, NH, 16):  # head blocks bound the [heads, T, n] score buffer
        hs = range(h0, min(h0 + 16, NH))
        kk = torch.stack([k[:, h // G] for h in hs])                             # [h, n, HD]
        vv = torch.stack([v[:, h // G] for h in hs])
        qq = q[:, h0 : hs.stop].transpose(0, 1)                                  # [h, T, HD]
        if args.attn_f32:
            sc = torch.matmul(qq.float(), kk.float().transpose(1, 2)) * qk_scale
        else:
            sc = torch.matmul(qq, kk.transpose(1, 2)) * qk_scale
        sc = sc.float().masked_fill(~keep, float("-inf"))
        pr = torch.softmax(sc, dim=-1, dtype=torch.float32)
        if args.attn_f32:
            attn[h0 : hs.stop] = torch.matmul(pr, vv.float()).to(BF16)
        else:
            attn[h0 : hs.stop] = torch.matmul(pr.to(BF16), vv)
    attn = attn.transpose(0, 1)                                                 # [T, NH, HD]
    at = norm_weightless(attn, EPS).reshape(T, NH * HD)
    at = at * torch.sigmoid(lin(hq, w("self_attn.gate_proj.weight")))
    o_raw = lin(at, w("self_attn.o_proj.weight"))
    o = norm_weightless(o_raw, PEPS)
    xmid = residual_gate(x, o, w("post_attention_residual_gate.gate"))

    xn2 = norm_centered(xmid, w("pre_feedforward_layernorm.weight"), EPS)
    logits = lin(xn2.float(), w("mlp.gate.weight").float()) * tc["output_multiplier"]
    scores = torch.sigmoid(logits)
    bias = w("mlp.gate.e_score_correction_bias").float()
    top = torch.topk(scores + bias, k=K, dim=-1, sorted=False)[1]               # [T, K]
    ranked = torch.sort(scores + bias, dim=-1, descending=True).values
    margin = (ranked[:, K - 1] - ranked[:, K]).min().item()  # a device flip below ~1e-3 is a near tie
    if force is not None:
        top = torch.tensor([force])
    gates = torch.gather(scores, 1, top)
    gates = gates / (gates.sum(-1, keepdim=True) + 1e-15) * tc["routed_scaling_factor"]
    xe = norm_centered(lin(xn2, w("mlp.pre_expert_proj.weight")), w("mlp.pre_expert_norm.weight"), EPS)
    if args.from_dump:
        xe = from_bf16(os.path.join(args.from_dump, "rank0.act.xe.bin"))[: T * M].reshape(T, M).to(BF16)
        tab = np.fromfile(os.path.join(args.from_dump, "rank0.act.tab.bin"), dtype=np.uint32)[: 2 * T * K]
        top = torch.from_numpy(tab[0::2].astype(np.int64)).reshape(T, K)
        gates = torch.from_numpy(tab[1::2].view(np.float32).copy()).reshape(T, K)
    w_pen = w("mlp.experts.post_expert_norm.weight")
    mx = (P.format(L) + "mlp.experts.0.gate_proj.weight_scale") in index

    def experts():
        accs = {a4w4: torch.zeros(T, M, dtype=BF16) for a4w4 in (False, True)}
        for e in sorted(set(top.flatten().tolist())):         # HF loops experts in id order
            rows, slots = (top == e).nonzero(as_tuple=True)
            ep = f"mlp.experts.{e}."
            wt = {n: dequant_mxfp4(w(ep + n + ".weight"), w(ep + n + ".weight_scale")) if mx
                  else w(ep + n + ".weight") for n in ("gate_proj", "up_proj", "down_proj")}
            for a4w4, acc in accs.items():
                xi = quant_mxfp4_act(xe[rows]) if a4w4 and mx else xe[rows]
                hcur = torch.nn.functional.silu(lin(xi, wt["gate_proj"])) * lin(xi, wt["up_proj"])
                hcur = quant_mxfp4_act(hcur) if a4w4 and mx else hcur
                y = norm_input_scaled(lin(hcur, wt["down_proj"]), w_pen, PEPS)
                acc[rows] = acc[rows] + (y * gates[rows, slots][:, None]).to(BF16)
        return accs[False], accs[True]

    def tail(ylat):
        y = lin(ylat, w("mlp.post_expert_proj.weight"))
        y = norm_input_scaled(y, w("post_feedforward_layernorm.weight"), PEPS)
        return residual_gate(xmid, y, w("post_feedforward_residual_gate.gate"))

    ylat, ylat_a4w4 = experts()
    xnext = tail(ylat)
    xnext_a4w4 = tail(ylat_a4w4)
    kind = ("sliding" if sliding else "full") + ("+rope" if rope_on else "+nope") + ("" if mx else "+bf16-experts")
    return dict(xnext=xnext, xnext_a4w4=xnext_a4w4, ylat_a4w4=ylat_a4w4, k=k, v=v, top=top,
                gates=gates, kind=kind,
                margin=margin,
                stages={"act.attn": attn.reshape(T, NH * HD), "act.at": at, "act.o": o_raw,
                        "act.on": o, "act.xmid": xmid, "act.xn2": xn2, "act.ylat": ylat})


torch.manual_seed(args.seed)
x0 = torch.randn(T, H).to(BF16)
os.makedirs(OUT, exist_ok=True)
x, kinds, margins = x0, [], []
for L in LAYERS:
    g = torch.Generator().manual_seed(args.seed + 1 + L)
    hist = torch.randn(CTX - 1, H, generator=g).to(BF16)[:C0]
    force = [int(v) for v in args.force_last.split(",")] if args.force_last and L == LAYERS[-1] else None
    r = layer_fwd(L, x, hist, force)
    # head-major [kvh][ctx][hd]; the device writes rows C0.. itself. A prefill operand leaves them
    # zero and gets the chunk's rows as kv.{L}.{k,v}.reference.bf16 ([kvh][T][hd]).
    for n, t in (("k", r["k"]), ("v", r["v"])):
        full = torch.zeros(CTX, NKV, HD, dtype=BF16)
        full[: C0 if args.tokens else C0 + T] = t[: C0 if args.tokens else C0 + T]
        open(os.path.join(OUT, f"kv.{L}.{n}.bin"), "wb").write(to_bytes(full.transpose(0, 1)))
        if args.tokens:
            open(os.path.join(OUT, f"kv.{L}.{n}.reference.bf16"), "wb").write(
                to_bytes(t[C0:].transpose(0, 1)))
    kinds.append(f"{L}:{r['kind']}")
    margins.append(r["margin"])
    print(f"layer {L} ({r['kind']}) ctx={CTX} rows {C0}..{C0 + T}: top{K}[0]={r['top'][0].tolist()} "
          f"margin={r['margin']:.2e} xnext rms={r['xnext'].float().pow(2).mean().sqrt():.4f}", flush=True)
    x = r["xnext"]
open(os.path.join(OUT, "act.x.bin"), "wb").write(to_bytes(x0))
open(os.path.join(OUT, "reference.bf16"), "wb").write(to_bytes(x))
if len(LAYERS) == 1:
    open(os.path.join(OUT, "reference_a4w4.bf16"), "wb").write(to_bytes(r["xnext_a4w4"]))
    open(os.path.join(OUT, "act.ylat.a4w4.bf16"), "wb").write(to_bytes(r["ylat_a4w4"]))
stages = {n: t for n, t in r["stages"].items() if args.tp == 1 or n not in ("act.attn", "act.at")}
for name, t in stages.items():
    open(os.path.join(OUT, f"{name}.reference.bf16"), "wb").write(to_bytes(t))
d = (r["xnext_a4w4"].double() - x.double()).flatten()
json.dump(dict(
    model=args.model, layers=LAYERS, ctx=CTX, batch=1, tokens=args.tokens, prefix=C0, seed=args.seed,
    tp=args.tp, kinds=kinds,
    router_margins=margins, forced_last=args.force_last,
    reference="torch CPU mirror of modeling_muse_spark.py, W4A16 experts"
              + (", f32 attention scores/probabilities" if args.attn_f32 else ""),
    tolerance_rel_l2=0.03, stages=list(stages),
    dump=["act.hn", "act.qg", "act.kg", "act.vg", "act.ogate", "act.q", "act.xe"],
    router_topk=dict(ids=r["top"].tolist(), weights=r["gates"].tolist()),
    a4w4_vs_w4a16_xnext=dict(rel_l2=(d.norm() / x.double().norm()).item(),
                             max_abs=d.abs().max().item()),
), open(os.path.join(OUT, "reference.json"), "w"), indent=2)
print(f"layers {LAYERS} -> {OUT}")
