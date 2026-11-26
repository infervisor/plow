"""shapes.py <ckpt dir> <name regex>: dtype/shape of matching tensors (one per distinct suffix)."""
import json, re, struct, sys, os
d, pat = sys.argv[1], re.compile(sys.argv[2])
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
hdr = {}
seen = set()
for name, f in sorted(idx.items()):
    if not pat.search(name): continue
    suf = re.sub(r"layers\.\d+\.", "layers.N.", name)
    if suf in seen: continue
    seen.add(suf)
    if f not in hdr:
        with open(os.path.join(d, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr[f] = json.loads(fh.read(n))
    h = hdr[f][name]
    print(suf, h["dtype"], h["shape"])
