"""Layer-level parity: the sm_90a kernel schedule (pyengine) against the checkpoint's reference model.py
Blocks, with real weights, over a prefill and several decode steps.

  perf-data/tools/gpulease -n 1 dsv41-layers <venv-python> scripts/dsv41_nv/test_layers.py <cubin> [layers] [T] [steps]
  layers: comma list forming one contiguous stack, e.g. 0,1,2,3 or 20,21
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pyengine as PE  # noqa: E402

SNAP = os.environ.get(
    "DSV41_CKPT",
    "/root/dsv41/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277",
)
REF = os.path.join(SNAP, "inference")
sys.path.insert(0, REF)

cubin = sys.argv[1]
layers = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2,3").split(",")]
T = int(sys.argv[3]) if len(sys.argv) > 3 else 300
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 6
MAXLEN = 2048
TEACHER = os.environ.get("DSV41_TEACHER", "1") == "1"
SUB = os.environ.get("DSV41_SUB", "0") == "1"
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False

ck = PE.Ckpt(SNAP)
cfg = ck.cfg


# ------------------------------------------------------------------ shared host-side Engram gather
class EngramTable:
    """Row gather straight from the mmapped shard, dequantized as ParallelEngramEmbedding does."""

    def __init__(self, L):
        self.w = self._mm(f"layers.{L}.engram.embed.weight")
        self.s = self._mm(f"layers.{L}.engram.embed.scale")

    def _mm(self, name):
        f = ck.index[name]
        hdr, base = ck._hdr(f)
        h = hdr[name]
        s, _ = h["data_offsets"]
        return np.memmap(os.path.join(SNAP, f), dtype=np.uint8, mode="r", offset=base + s, shape=tuple(h["shape"]))

    def lookup(self, ids):  # ids [..., n_cols] int64 -> bf16 [..., n_cols, 256] (ParallelEngramEmbedding's shape)
        flat = ids.reshape(-1).cpu().numpy()
        v = torch.from_numpy(np.ascontiguousarray(self.w[flat])).view(torch.float8_e4m3fn).float()
        s = torch.from_numpy(np.ascontiguousarray(self.s[flat])).view(torch.float8_e8m0fnu).float()
        out = (v.unflatten(-1, (-1, 32)) * s.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
        return out.reshape(*ids.shape, -1).to(dev)


tables = {L: EngramTable(L) for L in layers if L in cfg["engram_layer_ids"]}

# ------------------------------------------------------------------ reference model
import model as M  # noqa: E402

M.world_size, M.rank = 1, 0
M.default_dtype = torch.float8_e4m3fn


class HostEngramEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, dim):
        super().__init__()
        self.L = None

    def forward(self, indices):
        return tables[self.L].lookup(indices)


M.ParallelEngramEmbedding = HostEngramEmbedding
rargs = json.load(open(os.path.join(REF, "config.json")))
args = M.ModelArgs(**rargs)
args.max_batch_size, args.max_seq_len = 1, MAXLEN
torch.set_default_dtype(torch.bfloat16)
layout = M.EngramLayout.from_args(args)
from safetensors import safe_open  # noqa: E402


def ref_state(L):
    sd = {}
    p = f"layers.{L}."
    names = [n for n in ck.index if n.startswith(p) and ".engram.embed." not in n]
    by_file = {}
    for n in names:
        by_file.setdefault(ck.index[n], []).append(n)
    for f, ns in by_file.items():
        with safe_open(os.path.join(SNAP, f), framework="pt", device="cpu") as fh:
            for n in ns:
                sd[n[len(p):]] = fh.get_tensor(n)
    w, s = sd.pop("attn.wo_a.weight"), sd.pop("attn.wo_a.scale")
    sd["attn.wo_a.weight"] = (w.unflatten(0, (-1, 32)).unflatten(-1, (-1, 32)).float() * s[:, None, :, None].float()).flatten(2, 3).flatten(0, 1).bfloat16()
    for k in list(sd):
        if ".experts." in k and "shared" not in k and k.endswith(".weight"):
            sd[k] = sd[k].view(torch.float4_e2m1fn_x2)
    return sd


ref_blocks = {}
with torch.device(dev):
    for L in layers:
        b = M.Block(L, args, layout)
        if b.engram is not None:
            b.engram.embed.L = L
        missing, unexpected = b.load_state_dict(ref_state(L), strict=False)
        missing = [m for m in missing if "engram.embed" not in m]
        assert not unexpected and not missing, (L, missing[:5], unexpected[:5])
        ref_blocks[L] = b
print("reference blocks built", flush=True)
torch.set_default_device(dev)  # as generate.py does after loading: the reference builds index tensors on it

hash_state = None
if tables:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAP)
    with torch.device(dev):
        hash_state = M.NgramHashState(args, layout, tok)

# ------------------------------------------------------------------ ours
eng = PE.Engine(cubin, cfg, MAXLEN + 64)
ours = {L: PE.Layer(ck, L, dev) for L in layers}
caches = {L: PE.SeqCache(ours[L], MAXLEN, dev) for L in layers}
print("engine layers loaded", flush=True)

embed = ck.get("embed.weight").to(dev)
torch.manual_seed(1)
ids = torch.randint(0, cfg["vocab_size"], (1, T + STEPS), device=dev)


def compare(tag, a, b):
    a, b = a.float(), b.float()
    rel = ((a - b).norm() / b.norm()).item()
    mx = (a - b).abs().max().item()
    print(f"  {tag:34s} rel={rel:.3e} maxabs={mx:.3e}", flush=True)
    return rel


worst = 0.0
rx = embed[ids[:, :T]].unsqueeze(2).repeat(1, 1, 4, 1)
rpre = M.make_identity_pre_mix(rx, 4)
ox = rx[0].clone()
opre = rpre[0].clone()
start = 0
for step in range(STEPS + 1):
    n = T if step == 0 else 1
    sp = 0 if step == 0 else T + step - 1
    if step > 0:
        rx = embed[ids[:, sp:sp + 1]].unsqueeze(2).repeat(1, 1, 4, 1)
        rpre = M.make_identity_pre_mix(rx, 4)
        ox, opre = rx[0].clone(), rpre[0].clone()
    hashes = hash_state(ids[:, sp:sp + n], sp) if hash_state is not None else None
    sh = PE.Shared()
    print(f"step {step} start_pos={sp} T={n}", flush=True)
    with torch.inference_mode():
        for L in layers:
            b = ref_blocks[L]
            if TEACHER:  # both sides start this layer from the reference's input
                ox, opre = rx[0].clone(), rpre[0].clone()
            if b.engram is not None:
                hid = hashes[:, :, b.engram.layer_hash_index, :]
                emb = tables[L].lookup(hid[0]).flatten(-2).contiguous()
                rx = b.engram(rx, hid)
                ox = ox.contiguous()
                eng.engram(ours[L], ox, emb)
                compare(f"L{L} engram", ox, rx[0])
                if TEACHER:
                    ox = rx[0].clone()
            if SUB:
                # Block.forward, unrolled so its intermediates can be compared
                res = rx
                a_pre, a_post, a_comb = b.hc_mixes(rx, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
                r_hn = b.attn_norm(b.hc_pre(rx, rpre))
                r_ao = b.attn(r_hn, sp)
                r_x2 = b.hc_post(r_ao, res, a_post, a_comb)
                f_pre, f_post, f_comb = b.hc_mixes(r_x2, b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base)
                r_hn2 = b.ffn_norm(b.hc_pre(r_x2, a_pre))
                r_y = b.ffn(r_hn2, None)
                rx, rpre = b.hc_post(r_y, r_x2, f_post, f_comb), f_pre
            else:
                rx, rpre = b(rx, sp, rpre, None)
            ox, opre = eng.layer(ours[L], ox.contiguous(), opre.contiguous(), caches[L], sh, sp, None, do_engram=False)
            if ours[L].index_source and sh.topk is not None and M.shared_attn.topk_idxs is not None:
                a, r_ = sh.topk.long(), M.shared_attn.topk_idxs[0].long()
                ov = []
                for i in range(a.shape[0]):
                    sa = set(a[i][a[i] >= 0].tolist())
                    sb = set(r_[i][r_[i] >= 0].tolist())
                    if sb:
                        ov.append(len(sa & sb) / len(sb))
                if ov:
                    print(f"  L{L} topk overlap mean={sum(ov)/len(ov):.4f} min={min(ov):.4f} rows={len(ov)}", flush=True)
            if SUB:
                d = eng.dbg
                compare(f"L{L}  attn in (hn)", d["hn"], r_hn[0])
                compare(f"L{L}  attn out", d["ao"], r_ao[0])
                compare(f"L{L}  ffn in (hn2)", d["hn2"], r_hn2[0])
                compare(f"L{L}  ffn out", d["y"], r_y[0])
                # MoE alone, from the reference's own input
                y_tf = eng.moe(ours[L], r_hn2[0].contiguous())
                compare(f"L{L}  ffn out (same input)", y_tf, r_y[0])
                rw, ri = b.ffn.gate(r_hn2.view(-1, r_hn2.shape[-1]))
                oi, ow = eng.dbg_route
                same = (oi.long().sort(-1).values == ri.sort(-1).values).all(-1).float().mean().item()
                print(f"  L{L}  routing: tokens with identical expert set {same:.4f}", flush=True)
                if same == 1.0:
                    ws = (ow.gather(1, oi.long().argsort(-1)) - rw.float().gather(1, ri.argsort(-1))).abs().max().item()
                    print(f"  L{L}  routing weights maxabs diff {ws:.3e}", flush=True)
            worst = max(worst, compare(f"L{L} x", ox, rx[0]))
            compare(f"L{L} pre_mix", opre, rpre[0])
torch.cuda.synchronize()
print(f"worst x rel = {worst:.3e}")
print("RESULT", "PASS" if worst < 2e-2 else "FAIL")
