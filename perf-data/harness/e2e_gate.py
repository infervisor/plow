#!/usr/bin/env python3
"""End-to-end greedy token-match gate for mixed bf16/fp8 per-tile residency.

All configs run with fp32 arithmetic; only the *weight storage format* varies,
so the experiment isolates format error from arithmetic precision.
  f = fraction of GEMM tiles stored fp8 (best-relL2 tiles first; worst 1-f stay bf16)
"""
import sys, torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/models/Qwen3-4B"
BN, BK = 128, 64
NEW = 40
PROMPT = "The capital of France is Paris. Explain in detail why the Roman Empire fell, step by step:"

# thresholds from tiles.pt distribution
TH = {0.0: -1.0, 0.25: 0.026250, 0.50: 0.026448, 0.75: 0.026644, 0.90: 0.026823, 1.0: 1e9}

def fp8_rt(W):
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    s = amax / 448.0
    return (W / s).clamp_(-448., 448.).to(torch.float8_e4m3fn).float() * s

def apply(W32, thresh, perturb=0.0):
    """W32: fp32 [N,K]. Returns fp32 with per-tile format applied. thresh<0 => all bf16."""
    bf = W32.to(torch.bfloat16).float()
    if thresh < 0:
        return bf
    q = fp8_rt(W32)
    if thresh > 1e8:
        out = q
    else:
        N, K = W32.shape
        nb, kb = N // BN, K // BK
        w = W32.view(nb, BN, kb, BK).permute(0, 2, 1, 3)
        d = q.view(nb, BN, kb, BK).permute(0, 2, 1, 3)
        num = (d - w).flatten(2).pow(2).sum(-1).sqrt()
        den = w.flatten(2).pow(2).sum(-1).sqrt().clamp_min(1e-30)
        use8 = (num / den) <= thresh                       # [nb,kb]
        m = use8.view(nb, kb, 1, 1).expand(nb, kb, BN, BK).permute(0, 2, 1, 3).reshape(N, K)
        out = torch.where(m, q, bf)
    if perturb:
        out = out * (1.0 + perturb)
    return out

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    print("loading fp32...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    tgts = [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and m.weight.shape[0] % BN == 0 and m.weight.shape[1] % BK == 0]
    tgts += [("model.embed_tokens", model.model.embed_tokens)]
    orig = {n: m.weight.data.clone() for n, m in tgts}
    print(f"{len(tgts)} tiled tensors, "
          f"{sum(v.numel() for v in orig.values())/1e9:.3f}B params covered", flush=True)

    ids = tok(PROMPT, return_tensors="pt").input_ids
    ref = None
    configs = [("f=0.00 all-bf16 (REF)", -1.0, 0.0), ("f=1.00 all-fp8", 1e9, 0.0),
               ("f=0.90", TH[0.90], 0.0), ("f=0.75", TH[0.75], 0.0),
               ("f=0.50", TH[0.50], 0.0), ("f=0.25", TH[0.25], 0.0),
               ("NEGCTRL all-bf16 +1e-3 rel perturb", -1.0, 1e-3),
               ("NEGCTRL all-bf16 +1e-4 rel perturb", -1.0, 1e-4)]
    for label, th, pt in configs:
        t0 = time.time()
        with torch.no_grad():
            for n, m in tgts:
                m.weight.data = apply(orig[n], th, pt)
            out = model.generate(ids, max_new_tokens=NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = out[0, ids.shape[1]:].tolist()
        if ref is None:
            ref = gen
            print(f"\n[{label}] {time.time()-t0:.0f}s\n  {tok.decode(gen)!r}", flush=True)
        else:
            k = 0
            while k < NEW and gen[k] == ref[k]:
                k += 1
            print(f"\n[{label}] {time.time()-t0:.0f}s  match={k}/{NEW} consecutive greedy tokens"
                  f"  {'PASS(>=20)' if k>=20 else 'FAIL'}\n  {tok.decode(gen)!r}", flush=True)

if __name__ == "__main__":
    main()
