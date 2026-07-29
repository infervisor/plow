#!/usr/bin/env python3
"""glm52_fetch_qb.py — fetch the two MLA tensors weight-prep ABSORBS AWAY.  [GLM52-PF-GATE]

    python3 scripts/glm52_fetch_qb.py [out-dir] [layer ...]      # default: /home/lava/models/glm52_hf_extra 0 3

`scripts/glm52_prep.py` replaces `self_attn.q_b_proj` / `self_attn.kv_b_proj` with their absorbed
folds — `derived.q_absorb = k_nope_wᵀ·q_b_nope` and `derived.v_absorb = value_wᵀ`. A product is not
invertible, so a plow-prepped checkpoint cannot reconstruct HF's `GlmMoeDsaAttention`, which is the
TRUSTED half of `runtime/tests/glm52_real_oracle.py`. Without these two the oracle would have to be
rewritten in the absorbed basis — i.e. in plow's own algebra, which is not an oracle.

The originals are ~48 MB per layer against a 715 GB checkpoint, so this does NOT download shards:
it reads `model.safetensors.index.json`, then HTTP-Range-GETs each shard's safetensors header and
only the byte ranges of the four tensors, and writes them into one small local safetensors file.

VERIFY IT IS THE SAME CHECKPOINT before trusting it — folding the fetched q_b/kv_b must reproduce
the prepped `derived.*`. Measured on GLM-5.2, layers 0 and 3: `q_rope` and `v_absorb` are BIT-
IDENTICAL to the prepped bf16, and `q_absorb` agrees to 3e-5 relative (f32 einsum summation order).
"""
import json, os, struct, sys, urllib.request

REPO = os.environ.get("GLM_HF_REPO", "zai-org/GLM-5.2-FP8")
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
TENSORS = ("q_b_proj", "kv_b_proj")


def rng(url, a, b):
    r = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b-1}"})
    return urllib.request.urlopen(r, timeout=180).read()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/lava/models/glm52_hf_extra"
    layers = [int(x) for x in sys.argv[2:]] or [0, 3]
    os.makedirs(out, exist_ok=True)
    with urllib.request.urlopen(BASE + "model.safetensors.index.json", timeout=180) as f:
        wm = json.load(f)["weight_map"]
    want = [f"model.layers.{l}.self_attn.{t}.weight{s}"
            for l in layers for t in TENSORS for s in ("", "_scale_inv")]
    missing = [w for w in want if w not in wm]
    if missing:
        raise SystemExit(f"not in {REPO}'s index: {missing}")

    hdrs = {}
    for s in sorted({wm[w] for w in want}):
        url = BASE + s
        n = struct.unpack("<Q", rng(url, 0, 8))[0]
        hdrs[s] = (url, json.loads(rng(url, 8, 8 + n)), 8 + n)
        print(f"{s}: header {n} B")

    blob, hdr, off = [], {}, 0
    for w in want:
        url, h, b0 = hdrs[wm[w]]
        m = h[w]
        a, b = m["data_offsets"]
        d = rng(url, b0 + a, b0 + b)
        assert len(d) == b - a, w
        hdr[w] = {"dtype": m["dtype"], "shape": m["shape"], "data_offsets": [off, off + len(d)]}
        off += len(d)
        blob.append(d)
        print(f"  {w} {m['dtype']} {m['shape']} {len(d)} B")

    hj = json.dumps(hdr).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)      # safetensors requires an 8-aligned data section
    p = os.path.join(out, "model-00001-of-00001.safetensors")
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for d in blob:
            f.write(d)
    print(f"wrote {p} ({os.path.getsize(p)/1e6:.1f} MB)  -> pass as GLM_EXTRA_DIRS={out}")


if __name__ == "__main__":
    main()
