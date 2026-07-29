#!/usr/bin/env python3
"""(a) audit: classify GLM-5.2 decode weight bytes into
   fp8-on-disk (merely dequantised by the prep)  vs  genuinely derived  vs  natively bf16.

Reads ONLY safetensors headers (no tensor data), from the SOURCE checkpoint.
Prints per-rank-per-token active weight bytes at TP=4.
"""
import os, sys, json, struct, glob

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    glob.glob('/home/lava/.cache/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/*/')[0]

def headers(d):
    idx = {}
    for fn in sorted(os.listdir(d)):
        if not (fn.startswith('model-') and fn.endswith('.safetensors')):
            continue
        p = os.path.join(d, fn)
        with open(p, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            h = json.loads(f.read(n))
        for k, v in h.items():
            if k != '__metadata__':
                idx[k] = (v['dtype'], v['shape'])
    return idx

idx = headers(SRC)
cfg = json.load(open(os.path.join(SRC, 'config.json')))
H   = cfg['hidden_size']; NH = cfg['num_attention_heads']; DK = cfg['kv_lora_rank']
DR  = cfg['qk_rope_head_dim']; QN = cfg['qk_nope_head_dim']; VD = cfg['v_head_dim']
QL  = cfg['q_lora_rank']; IMOE = cfg['moe_intermediate_size']; DI = cfg['intermediate_size']
E   = cfg['n_routed_experts']; TK = cfg['num_experts_per_tok']; V = cfg['vocab_size']
NL  = cfg['num_hidden_layers']; FKD = cfg['first_k_dense_replace']
TP  = 4
nh_l, imoe_l, di_l = NH // TP, IMOE // TP, DI // TP
NL = 78                    # decoder layers actually emitted (78 = MTP head, skipped)
NSPARSE = NL - FKD

print(f"cfg H={H} NH={NH} DK={DK} DR={DR} QN={QN} VD={VD} QL={QL} IMOE={IMOE} DI={DI} "
      f"E={E} topk={TK} V={V} layers={NL} dense={FKD} | TP={TP} nh_l={nh_l} imoe_l={imoe_l} di_l={di_l}")
for n in ('lm_head.weight', 'model.embed_tokens.weight', 'model.layers.3.mlp.gate.weight',
          'model.layers.3.self_attn.q_a_proj.weight', 'model.layers.3.self_attn.o_proj.weight',
          'model.layers.3.self_attn.q_b_proj.weight', 'model.layers.3.self_attn.kv_b_proj.weight',
          'model.layers.3.self_attn.kv_a_proj_with_mqa.weight',
          'model.layers.3.mlp.shared_experts.gate_proj.weight'):
    print(f"  on disk: {n:60s} {idx.get(n)}")

MB = 1 << 20
# rows: (label, per-layer bytes at bf16 today, class, n_layers)
#   class A = fp8 on disk, dequantised for convenience -> convertible with NO numeric change
#   class B = derived product -> would need RE-quantisation (numerics change)
#   class C = natively bf16 in the checkpoint -> no fp8 source at all
rows = [
    ("q_a_proj            [QL,H]",            QL * H * 2,              'A', NL),
    ("derived.kv_a_latent [DK,H]",            DK * H * 2,              'A', NL),
    ("derived.k_rope      [DR,H]",            DR * H * 2,              'A', NL),
    ("derived.q_rope      [nh_l*DR,QL]",      nh_l * DR * QL * 2,      'A', NL),
    ("o_proj              [H,nh_l*VD]",       H * nh_l * VD * 2,       'A', NL),
    ("shared gate+up      2x[imoe_l,H]",      2 * imoe_l * H * 2,      'A', NSPARSE),
    ("shared down         [H,imoe_l]",        H * imoe_l * 2,          'A', NSPARSE),
    ("derived.q_absorb    [nh_l*DK,QL]",      nh_l * DK * QL * 2,      'B', NL),
    ("derived.v_absorb    [nh_l*DK,VD]",      nh_l * DK * VD * 2,      'B', NL),
    ("mlp.gate (router)   [E,H]",             E * H * 2,               'C', NSPARSE),
]
lm_dt = idx['lm_head.weight'][0]
rows.append((f"lm_head             [V,H] (disk {lm_dt})", V * H * 2, 'A' if lm_dt == 'F8_E4M3' else 'C', 1))

# already-fp8 streams (unchanged by (a)) for the total
fp8_rows = [
    ("routed experts topk*(2*[imoe_l,H]+[H,imoe_l]) fp8", TK * (3 * imoe_l * H + 0), NSPARSE),
    ("dense FFN (2*[di_l,H]+[H,di_l]) fp8",               3 * di_l * H,              FKD),
]

tot_bf16 = {'A': 0, 'B': 0, 'C': 0}
print(f"\n{'tensor':45s} {'MB/layer':>10s} {'x layers':>9s} {'MB/token':>10s}  class")
for lab, per, cls, n in rows:
    tot_bf16[cls] += per * n
    print(f"{lab:45s} {per/MB:10.2f} {n:9d} {per*n/MB:10.1f}  {cls}")
fp8_tot = 0
for lab, per, n in fp8_rows:
    fp8_tot += per * n
    print(f"{lab:45s} {per/MB:10.2f} {n:9d} {per*n/MB:10.1f}  fp8")

BW = 6200e9
print(f"\n--- per rank per token, TP4 ---")
print(f"bf16 class A (fp8 on disk, dequantised for convenience) : {tot_bf16['A']/MB:9.1f} MB")
print(f"bf16 class B (DERIVED products, no native fp8 form)     : {tot_bf16['B']/MB:9.1f} MB")
print(f"bf16 class C (natively bf16 in the checkpoint)          : {tot_bf16['C']/MB:9.1f} MB")
print(f"already block-fp8                                       : {fp8_tot/MB:9.1f} MB")
tot = sum(tot_bf16.values()) + fp8_tot
print(f"TOTAL active weight stream                              : {tot/MB:9.1f} MB = {tot/1e9:.2f} GB")
print(f"\nfloor at 6200 GB/s: today {tot/BW*1e3:.3f} ms")
for k in 'AB':
    print(f"  convert class {k}: -{tot_bf16[k]/2/MB:8.1f} MB -> -{tot_bf16[k]/2/BW*1e3:.3f} ms of floor")
print(f"  A+B together     : -{(tot_bf16['A']+tot_bf16['B'])/2/MB:8.1f} MB -> "
      f"-{(tot_bf16['A']+tot_bf16['B'])/2/BW*1e3:.3f} ms of floor "
      f"(floor becomes {(tot-(tot_bf16['A']+tot_bf16['B'])/2)/BW*1e3:.3f} ms)")
