#!/usr/bin/env python3
"""Sweep fp8 fraction f with per-tile selection: SORTED (worst-relL2 tiles -> bf16)
vs RANDOM (same f, arbitrary tiles). Token-match gate on 3 prompts."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/models/Qwen3-4B"; BN, BK = 128, 64; NEW = 40
PROMPTS = [
 "The capital of France is Paris. Explain in detail why the Roman Empire fell, step by step:",
 "Write a Python function that merges two sorted linked lists and explain its complexity.\n\ndef merge(",
 "Q: A train leaves Boston at 3:15pm travelling 62 mph. Another leaves New York at 4:05pm travelling 78 mph toward it. The cities are 215 miles apart. When do they meet? Reason carefully.\nA:",
]
TH = {0.25: 0.026250, 0.50: 0.026448, 0.75: 0.026644, 0.90: 0.026823}

def fp8_rt(W):
    s = (W.abs().amax(1, keepdim=True) / 448.).clamp_min(1e-30)
    return (W / s).clamp_(-448., 448.).to(torch.float8_e4m3fn).float() * s

def mixed(W, f, mode, g):
    if f >= 1.0: return fp8_rt(W)
    if f <= 0.0: return W
    q = fp8_rt(W); N, K = W.shape; nb, kb = N // BN, K // BK
    if mode == "sorted":
        w = W.view(nb, BN, kb, BK).permute(0, 2, 1, 3); d = q.view(nb, BN, kb, BK).permute(0, 2, 1, 3)
        r = (d - w).flatten(2).pow(2).sum(-1).sqrt() / w.flatten(2).pow(2).sum(-1).sqrt().clamp_min(1e-30)
        use8 = r <= TH[f]
    else:
        use8 = torch.rand((nb, kb), generator=g) < f
    m = use8.view(nb, kb, 1, 1).expand(nb, kb, BN, BK).permute(0, 2, 1, 3).reshape(N, K)
    return torch.where(m, q, W)   # W is already bf16-valued (checkpoint is bf16)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    tgts = [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and m.weight.shape[0] % BN == 0 and m.weight.shape[1] % BK == 0]
    tgts += [("emb", model.model.embed_tokens)]
    orig = {n: m.weight.data.clone() for n, m in tgts}
    ids = [tok(p, return_tensors="pt").input_ids for p in PROMPTS]
    def gen(k):
        with torch.no_grad():
            return model.generate(ids[k], max_new_tokens=NEW, do_sample=False,
                                  pad_token_id=tok.eos_token_id)[0, ids[k].shape[1]:].tolist()
    def run(f, mode):
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for n, m in tgts: m.weight.data = mixed(orig[n], f, mode, g)
        return [gen(k) for k in range(3)]
    refs = run(0.0, "sorted")
    def match(a, b):
        k = 0
        while k < NEW and a[k] == b[k]: k += 1
        return k
    print("f      mode     GB    ms/tok@1673   token-match p1/p2/p3   gate(>=20 all)", flush=True)
    print("0.00   bf16    8.00     4.78         40/40/40               PASS (reference)", flush=True)
    for f in [1.0, 0.90, 0.75, 0.50, 0.25]:
        for mode in (["sorted", "random"] if f < 1.0 else ["-"]):
            o = run(f, mode)
            m = [match(o[k], refs[k]) for k in range(3)]
            print("%.2f   %-7s %.2f   %6.2f         %2d/%2d/%2d               %s"
                  % (f, mode, 8 - 4 * f, (8 - 4 * f) / 1.673, m[0], m[1], m[2],
                     "PASS" if min(m) >= 20 else "FAIL"), flush=True)

main()
