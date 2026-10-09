import json, struct, glob, re, collections
import sys
d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/models/google/gemma-4-E4B-it"
shapes = {}
for f in glob.glob(d + "/*.safetensors"):
    with open(f, "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]
        hdr = json.loads(h.read(n))
    for k, v in hdr.items():
        if k != "__metadata__":
            shapes[k] = (v["dtype"], v["shape"])
per = collections.defaultdict(dict)
for k, (dt, sh) in shapes.items():
    m = re.search(r"language_model\.layers\.(\d+)\.(.*)", k)
    if m:
        per[int(m.group(1))][m.group(2)] = (dt, sh)
def nbytes(dt, sh):
    n = 1
    for x in sh: n *= x
    return n * {"BF16": 2, "F32": 4, "F16": 2}[dt]
for L in (0, 5):
    print(f"layer {L}:")
    tot = 0
    for k, (dt, sh) in sorted(per[L].items()):
        b = nbytes(dt, sh); tot += b
        print(f"  {k:55s} {dt} {sh} {b/2**20:.2f} MiB")
    print(f"  total {tot/2**20:.1f} MiB = {tot/1e6:.1f} MB")
tl = [sum(nbytes(*v) for v in per[L].values()) for L in sorted(per)]
print("layers", len(tl), "min/max MiB", min(tl)/2**20, max(tl)/2**20, "sum GiB", sum(tl)/2**30)
other = sum(nbytes(*v) for k, v in shapes.items() if "language_model.layers." not in k)
print("non-layer GiB", other/2**30)
for k, (dt, sh) in shapes.items():
    if "language_model.layers." not in k and nbytes(dt, sh) > 2**26: print(" ", k, dt, sh, nbytes(dt, sh)/2**30, "GiB")
