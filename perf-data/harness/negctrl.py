#!/usr/bin/env python3
"""Valid negative control + per-layer activation error + multi-prompt token gate.

Negative control: i.i.d. multiplicative Gaussian noise on every GEMM weight at a
*controlled relative-L2 energy*, so it is directly comparable to the fp8 round-trip
(relL2 = 0.0264). Uniform rescaling is NOT a valid control: RMSNorm makes the network
invariant to it.
"""
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/models/Qwen3-4B"
BN, BK = 128, 64
NEW = 40
PROMPTS = [
 "The capital of France is Paris. Explain in detail why the Roman Empire fell, step by step:",
 "Write a Python function that merges two sorted linked lists and explain its complexity.\n\ndef merge(",
 "Q: A train leaves Boston at 3:15pm travelling 62 mph. Another leaves New York at 4:05pm travelling 78 mph toward it. The cities are 215 miles apart. When do they meet? Reason carefully.\nA:",
]

def fp8_rt(W):
    s = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / 448.0
    return (W / s).clamp_(-448., 448.).to(torch.float8_e4m3fn).float() * s

def noisy(W, rel, g):
    n = torch.randn(W.shape, generator=g, dtype=torch.float32)
    n *= rel * W.norm() / n.norm().clamp_min(1e-30)
    return (W + n).to(torch.bfloat16).float()

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    tgts = [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and m.weight.shape[0] % BN == 0 and m.weight.shape[1] % BK == 0]
    tgts += [("emb", model.model.embed_tokens)]
    orig = {n: m.weight.data.clone() for n, m in tgts}
    def setw(fn):
        with torch.no_grad():
            for n, m in tgts:
                m.weight.data = fn(orig[n])

    ids = [tok(p, return_tensors="pt").input_ids for p in PROMPTS]

    # ---- per-layer activation error, fp8 vs fp32 reference ----
    setw(lambda W: W)
    with torch.no_grad():
        h32 = model(ids[2], output_hidden_states=True).hidden_states
    setw(lambda W: W.to(torch.bfloat16).float())
    with torch.no_grad():
        hbf = model(ids[2], output_hidden_states=True).hidden_states
    setw(fp8_rt)
    with torch.no_grad():
        o8 = model(ids[2], output_hidden_states=True)
    h8 = o8.hidden_states
    print("per-layer hidden-state relative L2 vs fp32 reference")
    print("layer   bf16        fp8")
    for i in list(range(0, 37, 4)) + [36]:
        r = lambda a: ((a[i] - h32[i]).norm() / h32[i].norm()).item()
        print("  %2d   %.3e   %.3e" % (i, r(hbf), r(h8)))
    print()

    # ---- token gate, 3 prompts, fp32 / bf16 / fp8 ----
    def gen(k):
        with torch.no_grad():
            return model.generate(ids[k], max_new_tokens=NEW, do_sample=False,
                                  pad_token_id=tok.eos_token_id)[0, ids[k].shape[1]:].tolist()
    def match(a, b):
        k = 0
        while k < NEW and a[k] == b[k]: k += 1
        return k

    refs = {}
    setw(lambda W: W.to(torch.bfloat16).float())
    for k in range(3): refs[k] = gen(k)
    print("=== token match vs all-bf16 reference (40 greedy tokens, 3 prompts) ===")
    setw(lambda W: W)
    print("fp32 weights      :", [match(gen(k), refs[k]) for k in range(3)])
    setw(fp8_rt)
    r8 = [match(gen(k), refs[k]) for k in range(3)]
    print("fp8 e4m3 per-chan :", r8)

    # ---- NEGATIVE CONTROL: matched-energy and larger random noise ----
    print("\n=== NEGATIVE CONTROL: i.i.d. Gaussian weight noise at relL2 = X ===")
    for rel in [0.0264, 0.05, 0.10, 0.20]:
        g = torch.Generator().manual_seed(1234)
        setw(lambda W: noisy(W, rel, g))
        m = [match(gen(k), refs[k]) for k in range(3)]
        print("relL2=%.4f      : %s  %s" % (rel, m, "DETECTED (gate fails)" if min(m) < 20 else "not detected"))

if __name__ == "__main__":
    main()
