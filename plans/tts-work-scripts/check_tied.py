import json, glob, os
from safetensors import safe_open
d = glob.glob("/root/tts-work/hf/hub/models--maya-research--Veena/snapshots/*")[0]
idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
def get(n):
    with safe_open(f"{d}/{idx[n]}", "pt") as f:
        return f.get_tensor(n)
e, h = get("model.embed_tokens.weight"), get("lm_head.weight")
print(e.shape, h.shape, e.dtype, "equal:", bool((e == h).all()), "maxdiff:", float((e.float() - h.float()).abs().max()))
print("n tensors", len(idx))
