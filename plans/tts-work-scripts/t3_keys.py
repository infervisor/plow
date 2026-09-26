import glob, collections
from safetensors import safe_open
d = glob.glob("/root/tts-work/hf/hub/models--ResembleAI--chatterbox/snapshots/*")[0]
with safe_open(f"{d}/t3_cfg.safetensors", "pt") as f:
    keys = list(f.keys())
    groups = collections.Counter(k.split(".layers.")[0] if ".layers." in k else k for k in keys)
    for k, n in sorted(groups.items()):
        shape = tuple(f.get_slice(k).get_shape()) if n == 1 else f"x{n}"
        print(k, shape)
    print([k for k in keys if "layers.0." in k])
import torch
c = torch.load(f"{d}/conds.pt", map_location="cpu", weights_only=True)
for part in c:
    for k, v in c[part].items():
        print(part, k, getattr(v, "shape", v), getattr(v, "dtype", ""))
