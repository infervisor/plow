#!/usr/bin/env python3
"""Decode roofline for DeepSeek-V4-Flash on ONE MI325X, from the real checkpoint.

Counts the bytes a single decode step must actually read: every attention and
hyper-connection weight, the compressor/indexer weights on the layers that have
them, the shared expert, `num_experts_per_tok` of the 256 routed experts, the
lm_head, and the KV cache at the target context. Weight bytes come from the
safetensors headers, so fp8/fp4/bf16 are counted as they are stored, not assumed.
"""
import json, os, re, struct, collections

DIR = '/home/lava/models/DeepSeek-V4-Flash-0731'
BW = 4164e9          # measured, registry
TARGET_TOKS = 150.0
CTX = 16384

cfg = json.load(open(os.path.join(DIR, 'config.json')))
L = cfg['num_hidden_layers']
ratios = cfg['compress_ratios']
topk = cfg['num_experts_per_tok']
n_exp = cfg['n_routed_experts']
hd = cfg['head_dim']
win = cfg['sliding_window']
idx_hd = cfg['index_head_dim']

wm = json.load(open(os.path.join(DIR, 'model.safetensors.index.json')))['weight_map']

# ---- byte size of every tensor, from the shard headers -----------------------
by_file = collections.defaultdict(list)
for name, f in wm.items():
    by_file[f].append(name)

size = {}
for f, names in by_file.items():
    with open(os.path.join(DIR, f), 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for nm in names:
        a, b = hdr[nm]['data_offsets']
        size[nm] = b - a

def tot(pred):
    return sum(v for k, v in size.items() if pred(k))

# mtp.* is the DSpark draft network — not part of a main-tower decode step.
main = lambda k: not k.startswith('mtp.')

# One representative routed expert, times topk, per layer.
exp_re = re.compile(r'^layers\.(\d+)\.ffn\.experts\.(\d+)\.')
per_layer_one_expert = collections.defaultdict(int)
for k, v in size.items():
    m = exp_re.match(k)
    if m and m.group(2) == '0':
        per_layer_one_expert[int(m.group(1))] += v

routed = sum(per_layer_one_expert[l] * topk for l in range(L))
shared = tot(lambda k: main(k) and '.ffn.shared_experts.' in k)
gate = tot(lambda k: main(k) and '.ffn.gate.' in k and 'tid2eid' not in k)
attn = tot(lambda k: main(k) and '.attn.' in k
           and '.compressor.' not in k and '.indexer.' not in k)
comp = tot(lambda k: main(k) and '.attn.compressor.' in k)
indexer = tot(lambda k: main(k) and '.attn.indexer.' in k)
hc = tot(lambda k: main(k) and ('hc_attn' in k or 'hc_ffn' in k or 'hc_head' in k))
norms = tot(lambda k: main(k) and (k.endswith('attn_norm.weight') or k.endswith('ffn_norm.weight')
                                   or k == 'norm.weight'))
head = size.get('head.weight', 0)

# ---- KV cache at CTX --------------------------------------------------------
kv = 0
for l in range(L):
    r = ratios[l] if l < len(ratios) else 0
    kv += win * hd * 2                              # sliding window ring
    if r:
        kv += (CTX // r) * hd * 2                   # compressed history
    if r == 4:
        kv += (CTX // r) * idx_hd * 2               # indexer's own compressed KV

rows = [
    ('attention projections', attn),
    ('routed experts (top-%d of %d)' % (topk, n_exp), routed),
    ('shared expert', shared),
    ('lm_head', head),
    ('KV compressors', comp),
    ('sparse indexer', indexer),
    ('hyper-connections', hc),
    ('router gates', gate),
    ('norms', norms),
    ('KV cache @ %dk' % (CTX // 1024), kv),
]
total = sum(v for _, v in rows)

print('DeepSeek-V4-Flash decode step, TP1, ctx %d' % CTX)
print('bandwidth 4164 GB/s (measured on MI325X)\n')
print('  %-34s %10s  %6s  %8s' % ('component', 'MB/token', 'share', 'ms @roof'))
for nm, v in sorted(rows, key=lambda x: -x[1]):
    print('  %-34s %10.1f  %5.1f%%  %8.3f' % (nm, v/1e6, 100*v/total, v/BW*1e3))
print('  %-34s %10.1f  %5.1f%%  %8.3f' % ('TOTAL', total/1e6, 100.0, total/BW*1e3))

roof_ms = total / BW * 1e3
print('\nroofline           %.2f ms/token  =  %.0f tok/s' % (roof_ms, 1000.0/roof_ms))
print('target             %.2f ms/token  =  %.0f tok/s' % (1000.0/TARGET_TOKS, TARGET_TOKS))
print('=> target needs %.0f%% of peak HBM bandwidth sustained end to end'
      % (100.0 * roof_ms / (1000.0/TARGET_TOKS)))

# Weights resident on one GPU?
allw = tot(lambda k: True)
print('\ncheckpoint on disk %.1f GB; MI325X has 256 GB, so TP1 is resident with %.0f GB spare'
      % (allw/1e9, 256 - allw/1e9 - kv/1e9))
