"""ref_layer.py <hf_dir> <layer> <ctx> <outdir> [chunk] [offset] [tp]: FP32 reference for one Gemma-4 text decoder layer at one decode step.

Runs the HF model in FP32 on <ctx> tokens of real text (repo docs), then one decode token, stopping after <layer>.
Dumps to <outdir>: the layer's BF16 weights, its inputs at the decode step (hidden state, per-layer input, RoPE
cos/sin, the layer's KV cache for positions 0..ctx-1 incl. the new token), and FP32 intermediates at every op
boundary. `layer_ref` re-implements the layer standalone; it must match HF's captured intermediates, and its
BF16 mode (BF16 weights, activations rounded to BF16 at every GEMV input, BF16 KV, FP32 accumulation) gives the
error bar a BF16 kernel is held to. Files: raw little-endian, listed in manifest.txt as `name dtype d0 d1 ...`.

Models without per-layer input and KV sharing (12B / 26B-A4B / 31B) load layers 0..<layer> only. tp = S > 1 writes S
tensor-parallel socket slices `<outdir>.tp<S>r<r>` instead of <outdir>: whole q heads with their KV heads (a KV head
shared by several ranks is replicated), a block of FFN rows, o / down split by K. Their `ref.o` / `ref.down` are the
rank's partial sums; `o_rest` / `down_rest` hold the other ranks' share, which the stage adds where the cross-socket
all-reduce lands."""
import glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

hf, L, ctx, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
off = int(sys.argv[6]) if len(sys.argv) > 6 else 0  # tokens of text skipped: distinct sequences for batch rows
TP = int(sys.argv[7]) if len(sys.argv) > 7 else 1
torch.set_grad_enabled(False)

from transformers import AutoConfig, AutoTokenizer, AutoModelForImageTextToText

tok = AutoTokenizer.from_pretrained(hf)
mcfg = AutoConfig.from_pretrained(hf)
tc = mcfg.text_config
if not tc.hidden_size_per_layer_input and not getattr(tc, "num_kv_shared_layers", 0):
    tc.num_hidden_layers, tc.layer_types = L + 1, tc.layer_types[: L + 1]
model = AutoModelForImageTextToText.from_pretrained(hf, config=mcfg, dtype=torch.float32)
lm = model.model.language_model
cfg = lm.config
layer = lm.layers[L]
attn = layer.self_attn
PLE = cfg.hidden_size_per_layer_input
MOE = bool(getattr(layer, "enable_moe_block", False))

repo = os.path.dirname(os.path.abspath(__file__)) + "/../../../.."
text = ""
for f in sorted(glob.glob(repo + "/docs/**/*.md", recursive=True)):
    text += open(f, errors="ignore").read() + "\n\n"
    if len(text) > (ctx + off) * 6:
        break
ids = tok(text, return_tensors="pt").input_ids
ids = torch.cat((ids[:, :1], ids[:, 1 + off : off + ctx + 1]), 1)  # BOS kept
assert ids.shape[1] == ctx + 1, ids.shape

cap = {}


class Stop(Exception):
    pass


def hook(name, kind="out"):
    def f(mod, args, kwargs, output):
        if kind == "in":
            cap[name] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
        else:
            o = output[0] if isinstance(output, tuple) else output
            cap[name] = o.detach().clone()
    return f


def pre_layer(mod, args, kwargs):
    cap["x_in"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
    if PLE:
        cap["per_layer_input"] = (args[1] if len(args) > 1 else kwargs["per_layer_input"]).detach().clone()
    cos, sin = kwargs["position_embeddings"]
    cap["cos"], cap["sin"] = cos.detach().clone(), sin.detach().clone()


def post_layer(mod, args, kwargs, output):
    cap["out"] = output.detach().clone()
    raise Stop


def router_hook(mod, args, kwargs, output):
    cap["router_w"], cap["router_idx"] = output[1].detach().clone(), output[2].detach().clone()


hs = []
mods = [("xn", layer.input_layernorm), ("q", attn.q_proj), ("qn", attn.q_norm), ("k", attn.k_proj),
        ("kn", attn.k_norm), ("v", attn.v_proj), ("o", attn.o_proj), ("pa", layer.post_attention_layernorm),
        ("xn2", layer.pre_feedforward_layernorm), ("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj),
        ("down", layer.mlp.down_proj), ("pf", layer.post_feedforward_layernorm)]
if PLE:
    mods += [("pg", layer.per_layer_input_gate), ("pp", layer.per_layer_projection),
             ("pn", layer.post_per_layer_input_norm)]
if MOE:
    mods += [("xn3", layer.pre_feedforward_layernorm_2), ("moe", layer.experts)]
    hs.append(layer.router.register_forward_hook(router_hook, with_kwargs=True))
for name, mod in mods:
    if mod is not None:
        hs.append(mod.register_forward_hook(hook(name), with_kwargs=True))
hs.append(attn.o_proj.register_forward_hook(hook("attn", "in"), with_kwargs=True))
hs.append(layer.register_forward_pre_hook(pre_layer, with_kwargs=True))
hs.append(layer.register_forward_hook(post_layer, with_kwargs=True))

from transformers import DynamicCache

cache = DynamicCache(config=cfg)
# chunked prefill ([chunk] tokens, default 8192) keeps the attention masks and scores small at 64K-128K context
chunk = int(sys.argv[5]) if len(sys.argv) > 5 else 8192
for c0 in range(0, ctx, chunk):
    try:
        lm(input_ids=ids[:, c0:min(ctx, c0 + chunk)], past_key_values=cache, use_cache=True)
    except Stop:
        pass
cap.clear()
# The layer's cache before the decode step (sliding layers keep window - 1 rows); the decode token attends to these
# plus its own new row.
kc, vc = cache.layers[L].keys[0].float().clone(), cache.layers[L].values[0].float().clone()  # [kvh, T, hd]
try:
    lm(input_ids=ids[:, ctx:], past_key_values=cache, use_cache=True)
except Stop:
    pass

hd = attn.head_dim
nh, kvh = cfg.num_attention_heads, kc.shape[0]
window = attn.sliding_window
if window and kc.shape[1] > window - 1:
    kc, vc = kc[:, -(window - 1):], vc[:, -(window - 1):]
eps = cfg.rms_norm_eps
W = {k: v.detach().float() for k, v in layer.state_dict().items()}


def rms(x, w=None):
    x = x.float()
    y = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return y * w if w is not None else y


def rot(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., h:], x[..., :h]), -1) * sin


def layer_ref(bf16):
    """Returns the boundary dict. bf16=True: BF16 weights (exact), GEMV inputs rounded to BF16, BF16 KV cache."""
    r = (lambda t: t.to(torch.bfloat16).float()) if bf16 else (lambda t: t)
    lin = lambda x, w: r(x) @ W[w].T
    b = {}
    x = cap["x_in"].float().reshape(-1)
    cos, sin = cap["cos"].float().reshape(-1), cap["sin"].float().reshape(-1)
    b["xn"] = rms(x, W["input_layernorm.weight"])
    b["q"] = lin(b["xn"], "self_attn.q_proj.weight")
    q = rot(rms(b["q"].view(nh, hd), W["self_attn.q_norm.weight"]), cos, sin)
    b["qn"] = q.reshape(-1)
    b["k"] = lin(b["xn"], "self_attn.k_proj.weight")
    # attention_k_eq_v (full layers of 12B / 26B / 31B): V is the raw K projection, normed without scale
    b["v"] = lin(b["xn"], "self_attn.v_proj.weight") if "self_attn.v_proj.weight" in W else b["k"]
    kn = rot(rms(b["k"].view(kvh, hd), W["self_attn.k_norm.weight"]), cos, sin)
    vn = rms(b["v"].view(kvh, hd))
    b["kn"] = kn.reshape(-1)
    b["vn"] = vn.reshape(-1)
    K = torch.cat((r(kc), r(kn)[:, None]), 1)  # prior rows + the new token's row as the kernel writes it
    V = torch.cat((r(vc), r(vn)[:, None]), 1)
    g = nh // kvh
    o = torch.empty(nh, hd)
    for h in range(nh):
        s = (K[h // g] @ q[h]) * 1.0
        p = torch.softmax(s, -1)
        o[h] = p @ V[h // g]
    b["attn"] = o.reshape(-1)
    b["o"] = lin(b["attn"], "self_attn.o_proj.weight")
    h1 = x + rms(b["o"], W["post_attention_layernorm.weight"])
    b["h1"] = h1
    b["xn2"] = rms(h1, W["pre_feedforward_layernorm.weight"])
    b["gate"] = lin(b["xn2"], "mlp.gate_proj.weight")
    b["up"] = lin(b["xn2"], "mlp.up_proj.weight")
    b["act"] = F.gelu(b["gate"], approximate="tanh") * b["up"]
    b["down"] = lin(b["act"], "mlp.down_proj.weight")
    ffn = b["down"]
    if MOE:
        # router on h1 (BF16 inputs in BF16 mode); expert choice pinned to HF's top-k so near-ties cannot flip it
        rn = rms(h1) * W["router.scale"] * cfg.hidden_size ** -0.5
        b["rs"] = lin(rn, "router.proj.weight")
        p = torch.softmax(b["rs"], -1)
        idx = cap["router_idx"].reshape(-1)
        wk = p[idx] / p[idx].sum() * W["router.per_expert_scale"][idx]
        b["rw"] = wk
        b["xn3"] = rms(h1, W["pre_feedforward_layernorm_2.weight"])
        moe = torch.zeros(cfg.hidden_size)
        I2 = W["experts.down_proj"].shape[2]
        for j, e in enumerate(idx.tolist()):
            gu = r(b["xn3"]) @ W["experts.gate_up_proj"][e].T
            a = F.gelu(gu[:I2], approximate="tanh") * gu[I2:]
            b[f"eact{j}"] = a
            moe += (r(a) @ W["experts.down_proj"][e].T) * wk[j]
        b["moe"] = moe
        ffn = rms(b["down"], W["post_feedforward_layernorm_1.weight"]) + rms(moe, W["post_feedforward_layernorm_2.weight"])
    h2 = h1 + rms(ffn, W["post_feedforward_layernorm.weight"])
    b["h2"] = h2
    if not PLE:
        b["out"] = h2 * W["layer_scalar"]
        return b
    b["pg"] = lin(h2, "per_layer_input_gate.weight")
    b["pact"] = F.gelu(b["pg"], approximate="tanh") * cap["per_layer_input"].float().reshape(-1)
    b["pp"] = lin(b["pact"], "per_layer_projection.weight")
    b["out"] = (h2 + rms(b["pp"], W["post_per_layer_input_norm.weight"])) * W["layer_scalar"]
    return b


def err(a, ref):
    a, ref = a.reshape(-1).double(), ref.reshape(-1).double()
    d = a - ref
    return dict(max_abs=d.abs().max().item(), rms=d.pow(2).mean().sqrt().item(),
                rel_rms=(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item(),
                cos=F.cosine_similarity(a, ref, 0).item())


f32 = layer_ref(False)
hf_names = {"xn": "xn", "q": "q", "k": "k", "attn": "attn", "o": "o", "xn2": "xn2", "gate": "gate", "up": "up",
            "down": "down", "out": "out"}
if attn.v_proj is not None:
    hf_names["v"] = "v"
if PLE:
    hf_names.update(pg="pg", pp="pp")
if MOE:
    hf_names.update(xn3="xn3", moe="moe")
check = {k: err(f32[k], cap[v]) for k, v in hf_names.items()}
if MOE:
    check["rw"] = err(f32["rw"], cap["router_w"])
bad = {k: e for k, e in check.items() if e["rel_rms"] > 1e-5}
bf = layer_ref(True)
I = int(W["mlp.gate_proj.weight"].shape[0])


def full_slice():
    return dict(q=(0, nh * hd), kv=(0, kvh * hd), f=(0, I), heads=nh, kvh=kvh, kv0=0, kv1=kvh)


def rank_slice(r):
    """Rank r of TP: q heads [h0, h1), their KV heads [k0, k1), FFN rows [f0, f1)."""
    nr, Ir, g = nh // TP, I // TP, nh // kvh
    assert nh % TP == 0 and I % TP == 0, (nh, I, TP)
    h0, h1 = r * nr, (r + 1) * nr
    k0, k1 = h0 // g, (h1 - 1) // g + 1
    assert nr == (k1 - k0) * g or (k1 - k0 == 1 and g % nr == 0), (nr, k0, k1, g)  # whole groups or one shared head
    return dict(q=(h0 * hd, h1 * hd), kv=(k0 * hd, k1 * hd), f=(r * Ir, (r + 1) * Ir), heads=nr, kvh=k1 - k0, kv0=k0,
                kv1=k1)


def slice_weights(s):
    q0, q1 = s["q"]; v0, v1 = s["kv"]; f0, f1 = s["f"]
    ws = {}
    for k, v in layer.state_dict().items():
        if not ((k.endswith("weight") and v.dim() >= 1) or k.startswith(("experts.", "router."))):
            continue
        if k.endswith(("q_proj.weight",)):
            v = v[q0:q1]
        elif k.endswith(("k_proj.weight", "v_proj.weight")):
            v = v[v0:v1]
        elif k.endswith("o_proj.weight"):
            v = v[:, q0:q1]
        elif k.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
            v = v[f0:f1]
        elif k.endswith("mlp.down_proj.weight"):
            v = v[:, f0:f1]
        ws[k] = v
    return ws


def slice_refs(b, s, bf16):
    """Boundaries of one rank: sliced rows, o / down as the rank's partial sums, the rest full."""
    q0, q1 = s["q"]; v0, v1 = s["kv"]; f0, f1 = s["f"]
    rd = (lambda t: t.to(torch.bfloat16).float()) if bf16 else (lambda t: t)
    o = dict(b)
    for k in ("q", "qn", "attn"):
        o[k] = b[k][q0:q1]
    for k in ("k", "v", "kn", "vn"):
        o[k] = b[k][v0:v1]
    for k in ("gate", "up", "act"):
        o[k] = b[k][f0:f1]
    if (q0, q1) != (0, nh * hd):
        o["o"] = rd(b["attn"][q0:q1]) @ W["self_attn.o_proj.weight"][:, q0:q1].T
        o["down"] = rd(b["act"][f0:f1]) @ W["mlp.down_proj.weight"][:, f0:f1].T
    return o


def write(d, s, rank):
    os.makedirs(d, exist_ok=True)
    man = []

    def dump(name, t, dt):
        a = t.detach().contiguous()
        raw = a.to(torch.bfloat16).view(torch.int16).numpy() if dt == "bf16" else a.float().numpy()
        raw.tofile(f"{d}/{name}.{dt}")
        man.append(f"{name} {dt} " + " ".join(str(x) for x in a.shape))

    for k, v in slice_weights(s).items():
        dump("w." + k, v, "bf16")
    dump("layer_scalar", W["layer_scalar"].reshape(1), "f32")
    dump("x_in", cap["x_in"].reshape(-1), "f32")
    if PLE:
        dump("per_layer_input", cap["per_layer_input"].reshape(-1), "f32")
    if MOE:
        dump("router_idx", cap["router_idx"].reshape(-1).float(), "f32")
    dump("cos", cap["cos"].reshape(-1), "f32")
    dump("sin", cap["sin"].reshape(-1), "f32")
    dump("kcache", kc[s["kv0"]:s["kv1"]], "bf16")  # rows before the decode token; the stage appends its own row
    dump("vcache", vc[s["kv0"]:s["kv1"]], "bf16")
    f, g = slice_refs(f32, s, False), slice_refs(bf, s, True)
    for k, v in f.items():
        dump("ref." + k, v, "f32")
    if rank is not None:
        dump("o_rest", f32["o"] - f["o"], "f32")
        dump("down_rest", f32["down"] - f["down"], "f32")
    open(f"{d}/manifest.txt", "w").write("\n".join(man) + "\n")
    meta = dict(hf=hf, layer=L, ctx=ctx, offset=off, layer_type=cfg.layer_types[L], hidden=cfg.hidden_size,
                heads=s["heads"], kv_heads=s["kvh"], head_dim=hd, window=window, cache_len=int(kc.shape[1]), eps=eps,
                inter=s["f"][1] - s["f"][0], ple=PLE, moe=MOE, kv_eq=attn.v_proj is None, tp=TP,
                rank=rank if rank is not None else 0, full_heads=nh, full_kv_heads=kvh, full_inter=I,
                token=int(ids[0, ctx]), torch=torch.__version__, hf_check=check, hf_check_fail=bad,
                bf16_ref_err={k: err(g[k], f[k]) for k in f})
    json.dump(meta, open(f"{d}/meta.json", "w"), indent=1)
    return meta


if TP == 1:
    metas = [write(out, full_slice(), None)]
else:
    assert not MOE, "tp: dense layers only"
    metas = [write(f"{out}.tp{TP}r{r}", rank_slice(r), r) for r in range(TP)]
m = metas[0]
print(json.dumps(dict(layer=L, ctx=ctx, type=m["layer_type"], cache_len=m["cache_len"], tp=TP, hf_check_fail=list(bad),
                      out_rel_rms_bf16=m["bf16_ref_err"]["out"]["rel_rms"], out_cos_bf16=m["bf16_ref_err"]["out"]["cos"])))
sys.exit(1 if bad else 0)
