"""CORRECTED FP8 decode accounting for Gemma-4, and why 26B FP8 decode still loses.

Supersedes fp8_roof.py / fp8_roof2.py. Those charged the 26B MoE experts at bf16 in the FP8
packet, on the strength of `shapes.moe_enc == []` / `precision.expert_enc == "none"`. That was
WRONG: manifest.rs populates moe_enc only from MoeGroupGluPf/MoeGroupDownPf, MoeAiterFp8Pf and the
*Fp8Blk ops -- never the Gemma opcode family. Proof it carries no signal: the BF16 packet also
reports expert_enc "none". The FP8 packet's union has MoeExpertGluGemmaFp8 / MoeExpertDownGemmaFp8
(decode) and MoeGroupGluGemmaPfW8a8 / MoeGroupDownGemmaPfW8a8 (prefill) and ZERO bf16 expert arms,
so the experts ARE quantised.

What actually stays bf16 in both FP8 packets is lm_head: the audited op at N=vocab is Gemm/Gemv,
with no Fp8 sibling (task #93).

So the real question is not bytes. Decode is bandwidth-bound in both precisions, and this script
shows fp8 wins the BYTE axis even so -- which means the measured ~5x FP8 decode regression on the
26B cannot be a byte effect. It is a TRAVERSAL/EFFICIENCY effect: lib.rs puts the grouped
(expert-union) decode route inside the bf16-only arm, and runtime/nvidia/op_moe.cuh's
d_moe_dec_group_glu_gemma calls the bf16 grouped prefill body, so the fp8 path forgoes the grouped
tensor-core GEMM and walks each expert per slot-dot. That is task #62, and this quantifies why it
is the lever rather than more quantisation.
"""
import importlib.util
import json
import sys

REPO = "/home/lava/plow/.claude/worktrees/gemma4-26b-beat-vllm"
spec = importlib.util.spec_from_file_location("rf", f"{REPO}/scripts/campaign/roofline.py")
rf = importlib.util.module_from_spec(spec)
sys.modules["rf"] = rf
spec.loader.exec_module(rf)

h = None
for nm in dir(rf):
    o = getattr(rf, nm)
    if isinstance(o, dict):
        for key in o:
            if "h100" in str(key).lower() and "sxm" in str(key).lower():
                h = o[key]
bw = h.bandwidth_for_bound_gbps * 1e9
G = 2 ** 30
print(f"H100 SXM5 {h.bandwidth_for_bound_gbps} GB/s; ridge bf16 "
      f"{h.bf16_tflops_dense*1e12/bw:.0f} / fp8 {h.fp8_tflops_dense*1e12/bw:.0f} FLOP/B")
print()

CK = "/opt/dlami/nvme/hf-cache/hub/gemma-4-26b-a4b-it"
m = rf.spec_from_hf_config(CK, "bf16")
t = json.loads(open(f"{CK}/config.json").read())
t = t.get("text_config", t)
lm = int(t["vocab_size"]) * int(t["hidden_size"])
experts = m.moe_expert_params
per_expert = experts / m.moe_top_k          # params in ONE expert
other = m.active_params - experts - lm

print(f"26B-A4B: active {m.active_params/1e9:.3f}G = dense-linear {other/1e9:.3f}G"
      f" + lm_head {lm/1e9:.3f}G + experts(top{m.moe_top_k} of {m.moe_experts})"
      f" {experts/1e9:.3f}G ({per_expert/1e9:.3f}G each)")
print()


def union(B):
    """Distinct experts touched by B rows at top_k: n*(1-(1-k/n)^B)."""
    if B <= 1:
        return float(m.moe_top_k)
    miss = 1.0 - m.moe_top_k / m.moe_experts
    return m.moe_experts * (1.0 - miss ** B)


print("Expert-weight bytes per decode step. GROUPED streams each touched expert ONCE;")
print("PER-SLOT streams one expert per slot-dot, i.e. B*top_k times.")
print(f"{'B':>3} {'union':>7} {'B*k':>5} | {'bf16 grouped':>14} {'fp8 grouped':>13} "
      f"{'fp8 per-slot':>14} | {'fp8 per-slot vs bf16 grouped':>30}")
for B in (1, 4, 16, 32):
    u, slots = union(B), B * m.moe_top_k
    bf_g = u * per_expert * 2
    f8_g = u * per_expert * 1
    f8_s = slots * per_expert * 1
    print(f"{B:>3} {u:>7.1f} {slots:>5} | {bf_g/G:>8.2f} GiB {f8_g/G:>8.2f} GiB "
          f"{f8_s/G:>9.2f} GiB | {f8_s/bf_g:>13.2f}x bytes")
print()

print("Whole-step bytes (dense-linear + lm_head + experts), lm_head bf16 in BOTH (task #93):")
print(f"{'B':>3} | {'bf16 (grouped)':>22} {'fp8 as shipped (per-slot)':>28} "
      f"{'fp8 + grouped (#62)':>24} {'fp8+grouped+lm_head (#93)':>27}")
for B in (1, 4, 16, 32):
    u, slots = union(B), B * m.moe_top_k
    def step(dense_b, lm_b, exp_streams, exp_b):
        return other * dense_b + lm * lm_b + exp_streams * per_expert * exp_b
    a = step(2, 2, u, 2)            # bf16, grouped
    b = step(1, 2, slots, 1)        # fp8 as shipped: per-slot expert walk
    c = step(1, 2, u, 1)            # fp8 + grouped route (#62)
    d = step(1, 1, u, 1)            # + fp8 lm_head (#93)
    def f(v):
        return f"{v/G:6.2f}GiB/{v/bw*1e3:6.2f}ms"
    print(f"{B:>3} | {f(a):>22} {f(b):>28} {f(c):>24} {f(d):>27}")
    print(f"    | {'':>22} {100*(1-b/a):>26.1f}% {100*(1-c/a):>22.1f}% "
          f"{100*(1-d/a):>25.1f}%")
print()
print("READING:")
print(" 1. The per-slot walk costs B*top_k expert streams, growing LINEARLY in B, while grouped")
print("    saturates as the union approaches all 128 experts. They cross at B ~ 28: below it fp8")
print("    wins on bytes even per-slot (+40.3 / +41.8 / +24.1% at B=1/4/16), at B=32 it LOSES")
print("    (-9.3%), because 256 slot-streams exceed 2x the 111.8-expert union.")
print(" 2. So bytes explain a modest B=32 regression, NOT the measured ~5x. The dominant term is")
print("    efficiency: the per-slot dot walk gives up the grouped tensor-core GEMM.")
print(" 3. #62 (grouped route under fp8) restores a uniform ~48% byte saving at every B -- at B=32")
print("    that is a 2.1x byte swing, -9.3% -> +48.3%. It is THE lever.")
print(" 4. #92 is a NON-TASK: the experts are already fp8.")
print(" 5. #93 (fp8 lm_head) adds +9.7 points at B=1 and +1.7 at B=32, on top of #62.")
