#!/usr/bin/env python3
"""Re-key a compressed-tensors FP8 (per-channel weight, dynamic per-token activation) Gemma-4
checkpoint to plow's `fp8/` contract. Nothing is requantized; the inverse of gemma4_fp8_vllm_ckpt.py.

  X.weight        F8_E4M3        -> fp8/X.weight        (bytes verbatim)
  X.weight_scale  BF16 [N, 1]    -> fp8/X.weight_scale  F32 [N] (exact widening)
  everything else                -> unchanged

Usage: gemma4_fp8_hub_rekey.py <hub-fp8-dir> <out-dir>
"""
import json, os, struct, sys

src_dir, out_dir = sys.argv[1], sys.argv[2]
src = os.path.join(src_dir, "model.safetensors")
os.makedirs(out_dir, exist_ok=True)

with open(src, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
base = 8 + n
hdr.pop("__metadata__", None)
fp8 = {k for k, v in hdr.items() if v["dtype"] == "F8_E4M3"}
scales = {k + "_scale" for k in fp8}
assert all(s in hdr and hdr[s]["dtype"] == "BF16" for s in scales)


def bf16_to_f32(b):
    out = bytearray(len(b) * 2)
    out[2::4] = b[0::2]
    out[3::4] = b[1::2]
    return bytes(out)


plan = []  # (out_name, dtype, shape, src_off, src_len, convert)
for k in sorted(hdr, key=lambda k: hdr[k]["data_offsets"][0]):
    v = hdr[k]
    lo, hi = v["data_offsets"]
    if k in fp8:
        plan.append(("fp8/" + k, "F8_E4M3", v["shape"], lo, hi - lo, False))
    elif k in scales:
        assert v["shape"][1] == 1
        plan.append(("fp8/" + k, "F32", [v["shape"][0]], lo, hi - lo, True))
    else:
        plan.append((k, v["dtype"], v["shape"], lo, hi - lo, False))

out_hdr, off = {}, 0
for name, dt, shape, _, ln, conv in plan:
    size = ln * 2 if conv else ln
    out_hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + size]}
    off += size
out_hdr["__metadata__"] = {"format": "pt", "source": "compressed-tensors fp8 rekey"}
blob = json.dumps(out_hdr, separators=(",", ":")).encode()
blob += b" " * ((-len(blob)) % 8)

dst = os.path.join(out_dir, "model.safetensors")
with open(src, "rb") as fi, open(dst + ".tmp", "wb") as fo:
    fo.write(struct.pack("<Q", len(blob)))
    fo.write(blob)
    for name, dt, shape, lo, ln, conv in plan:
        fi.seek(base + lo)
        if conv:
            fo.write(bf16_to_f32(fi.read(ln)))
        else:
            left = ln
            while left:
                b = fi.read(min(left, 256 << 20))
                fo.write(b)
                left -= len(b)
os.rename(dst + ".tmp", dst)

idx = {"metadata": {"total_size": off}, "weight_map": {k: "model.safetensors" for k in out_hdr if k != "__metadata__"}}
json.dump(idx, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=1)
for f in os.listdir(src_dir):
    if f.endswith((".json", ".jinja")) and f not in ("model.safetensors.index.json",):
        p = os.path.join(out_dir, f)
        if not os.path.exists(p):
            os.symlink(os.path.join(src_dir, f), p)
print(f"{len(fp8)} fp8 weights, {len(scales)} scales, {len(plan)} tensors, {off/2**30:.2f} GiB")
