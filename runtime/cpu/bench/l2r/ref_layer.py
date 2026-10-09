"""ref_layer.py <hf_dir> <layer> <ctx> <outdir> [chunk]: FP32 reference for one Gemma-4 text decoder layer at one decode step.

Runs the HF model in FP32 on <ctx> tokens of real text (repo docs), then one decode token, stopping after <layer>.
Dumps to <outdir>: the layer's BF16 weights, its inputs at the decode step (hidden state, per-layer input, RoPE
cos/sin, the layer's KV cache for positions 0..ctx-1 incl. the new token), and FP32 intermediates at every op
boundary. `layer_ref` re-implements the layer standalone; it must match HF's captured intermediates, and its
BF16 mode (BF16 weights, activations rounded to BF16 at every GEMV input, BF16 KV, FP32 accumulation) gives the
error bar a BF16 kernel is held to. Files: raw little-endian, listed in manifest.txt as `name dtype d0 d1 ...`."""
import glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

hf, L, ctx, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(out, exist_ok=True)
torch.set_grad_enabled(False)

from transformers import AutoTokenizer, AutoModelForImageTextToText

tok = AutoTokenizer.from_pretrained(hf)
model = AutoModelForImageTextToText.from_pretrained(hf, dtype=torch.float32)
lm = model.model.language_model
cfg = lm.config
layer = lm.layers[L]
attn = layer.self_attn

repo = os.path.dirname(os.path.abspath(__file__)) + "/../../../.."
text = ""
for f in sorted(glob.glob(repo + "/docs/**/*.md", recursive=True)):
    text += open(f, errors="ignore").read() + "\n\n"
    if len(text) > ctx * 6:
        break
ids = tok(text, return_tensors="pt").input_ids[:, : ctx + 1]
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
    cap["per_layer_input"] = (args[1] if len(args) > 1 else kwargs["per_layer_input"]).detach().clone()
    cos, sin = kwargs["position_embeddings"]
    cap["cos"], cap["sin"] = cos.detach().clone(), sin.detach().clone()


def post_layer(mod, args, kwargs, output):
    cap["out"] = output.detach().clone()
    raise Stop


hs = []
for name, mod in (("xn", layer.input_layernorm), ("q", attn.q_proj), ("qn", attn.q_norm), ("k", attn.k_proj),
                  ("kn", attn.k_norm), ("v", attn.v_proj), ("o", attn.o_proj), ("pa", layer.post_attention_layernorm),
                  ("xn2", layer.pre_feedforward_layernorm), ("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj),
                  ("down", layer.mlp.down_proj), ("pf", layer.post_feedforward_layernorm),
                  ("pg", layer.per_layer_input_gate), ("pp", layer.per_layer_projection),
                  ("pn", layer.post_per_layer_input_norm)):
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
    b["v"] = lin(b["xn"], "self_attn.v_proj.weight")
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
    h2 = h1 + rms(b["down"], W["post_feedforward_layernorm.weight"])
    b["h2"] = h2
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
hf_names = {"xn": "xn", "q": "q", "k": "k", "v": "v", "attn": "attn", "o": "o", "xn2": "xn2", "gate": "gate",
            "up": "up", "down": "down", "pg": "pg", "pp": "pp", "out": "out"}
check = {k: err(f32[k], cap[v]) for k, v in hf_names.items()}
bad = {k: e for k, e in check.items() if e["rel_rms"] > 1e-5}
bf = layer_ref(True)
bf_err = {k: err(bf[k], f32[k]) for k in f32}

man = []


def dump(name, t, dt):
    a = t.detach().contiguous()
    if dt == "bf16":
        raw = a.to(torch.bfloat16).view(torch.int16).numpy()
    else:
        raw = a.float().numpy()
    raw.tofile(f"{out}/{name}.{dt}")
    man.append(f"{name} {dt} " + " ".join(str(s) for s in a.shape))


for k, v in layer.state_dict().items():
    if k.endswith("weight") and v.dim() >= 1:
        dump("w." + k, v, "bf16")
dump("layer_scalar", W["layer_scalar"].reshape(1), "f32")
dump("x_in", cap["x_in"].reshape(-1), "f32")
dump("per_layer_input", cap["per_layer_input"].reshape(-1), "f32")
dump("cos", cap["cos"].reshape(-1), "f32")
dump("sin", cap["sin"].reshape(-1), "f32")
dump("kcache", kc, "bf16")  # rows before the decode token; the stage appends its own row
dump("vcache", vc, "bf16")
for k, v in f32.items():
    dump("ref." + k, v, "f32")
open(f"{out}/manifest.txt", "w").write("\n".join(man) + "\n")
meta = dict(hf=hf, layer=L, ctx=ctx, layer_type=cfg.layer_types[L], hidden=cfg.hidden_size, heads=nh, kv_heads=kvh,
            head_dim=hd, window=window, cache_len=int(kc.shape[1]), eps=eps, inter=int(W["mlp.gate_proj.weight"].shape[0]),
            ple=cfg.hidden_size_per_layer_input, token=int(ids[0, ctx]), torch=torch.__version__,
            hf_check=check, hf_check_fail=bad, bf16_ref_err=bf_err)
json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
print(json.dumps(dict(layer=L, ctx=ctx, type=meta["layer_type"], cache_len=meta["cache_len"], hf_check_fail=list(bad),
                      out_rel_rms_bf16=bf_err["out"]["rel_rms"], out_cos_bf16=bf_err["out"]["cos"])))
sys.exit(1 if bad else 0)
