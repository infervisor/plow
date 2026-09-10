import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_num_threads(8)
p = Path(sys.argv[1])
index = json.loads((p / 'checkpoint/model.safetensors.index.json').read_text())['weight_map']
prefix = 'language_model.model.layers.0.'
weights = {}
for name, shard in index.items():
    if name.startswith(prefix):
        with safe_open(p / 'checkpoint' / shard, framework='pt') as f:
            weights[name.removeprefix(prefix)] = f.get_tensor(name)

def norm(x, name):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
            * (1 + weights[name].float())).to(torch.bfloat16)

def linear(x, name):
    return F.linear(x, weights[name])

x = torch.from_numpy(np.load(p / 'block-input.npy')).to(torch.bfloat16)
t = x.shape[0]
a = norm(x, 'input_layernorm.weight')
q = norm(linear(a, 'self_attn.q_proj.weight').reshape(t, 16, 256), 'self_attn.q_norm.weight')
k = norm(linear(a, 'self_attn.k_proj.weight').reshape(t, 8, 256), 'self_attn.k_norm.weight')
v = linear(a, 'self_attn.v_proj.weight').reshape(t, 8, 256)
freq = torch.arange(t).float()[:, None] * (10000.0 ** (-torch.arange(0, 256, 2).float() / 256))[None, :]
cos = torch.cat([freq.cos(), freq.cos()], -1).to(torch.bfloat16)[:, None, :]
sin = torch.cat([freq.sin(), freq.sin()], -1).to(torch.bfloat16)[:, None, :]
def rotate(z):
    return z * cos + torch.cat([-z[..., 128:], z[..., :128]], -1) * sin
q, k = rotate(q), rotate(k)
k, v = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
scores = (q.transpose(0, 1) @ k.permute(1, 2, 0)) * (1 / 16)
mask = torch.ones(t, t, dtype=torch.bool).triu(1)
scores.masked_fill_(mask, float('-inf'))
probs = scores.float().softmax(-1).to(torch.bfloat16)
attn = (probs @ v.transpose(0, 1)).transpose(0, 1).reshape(t, 4096)
x = x + norm(linear(attn, 'self_attn.o_proj.weight'), 'post_attention_layernorm.weight')
a = norm(x, 'pre_feedforward_layernorm.weight')
gate = F.gelu(linear(a, 'mlp.gate_proj.weight'), approximate='tanh')
up = linear(a, 'mlp.up_proj.weight')
x = x + norm(linear(gate * up, 'mlp.down_proj.weight'), 'post_feedforward_layernorm.weight')
np.save(p / 'block-output-reference.npy', x.float().numpy())
for candidate in sorted(p.glob('block-output-*.npy')):
    if candidate.name == 'block-output-reference.npy':
        continue
    y = torch.from_numpy(np.load(candidate)).float()
    delta = y - x.float()
    print(json.dumps({'candidate': candidate.name, 'relative_l2': (delta.norm() / x.float().norm()).item(),
                      'max_abs': delta.abs().max().item(), 'finite': bool(y.isfinite().all())}))
