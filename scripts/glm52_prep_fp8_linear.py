#!/usr/bin/env python3
"""(a) prep: publish the block-fp8 form of the GLM-5.2 projections the existing prep DEQUANTISES.

`scripts/glm52_prep.py` writes `o_proj` and the shared expert as bf16 (`p_bf16(dequant_blockfp8(...))`).
Both are block-fp8 WHOLE TENSORS on disk, so their fp8 bytes and their [128,128] `weight_scale_inv`
grids can be republished VERBATIM — no dequant, no requant, no numeric change of any kind.

This writes them ADDITIVELY into a new weight dir that SYMLINKS the existing 79 prepped shards, so
nothing already on disk is touched and no other agent's run is disturbed. The fp8 weight takes a
`.weight_fp8` name (the bf16 `.weight` still lives in the base shard and must not collide); the scale
grid keeps its checkpoint name `.weight_scale_inv`, which does not exist in the base shards.

Both names keep the projection substring the harness's `glm_col`/`glm_row` predicates match on, so
the TP slicing of the weight AND of its scale grid is already correct with no host change.

  python3 prep_fp8_linear.py --src <GLM-5.2-FP8 snapshot> --base <GLM-5.2-plow> --out <dir>
"""
import argparse, json, mmap, os, struct, sys, time

DT_ELT = {"F8_E4M3": 1, "F32": 4, "BF16": 2, "F16": 2}


def index_shards(model_dir):
    idx = {}
    for fn in sorted(os.listdir(model_dir)):
        if not (fn.startswith("model-") and fn.endswith(".safetensors")):
            continue
        path = os.path.join(model_dir, fn)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            idx[k] = (path, base + v["data_offsets"][0], base + v["data_offsets"][1],
                      v["dtype"], v["shape"])
    return idx


_MM = {}
def mm(path):
    if path not in _MM:
        f = open(path, "rb")
        _MM[path] = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    return _MM[path]


def write_shard(path, entries):
    """entries: [(out_name, (src_path, a, b, dtype, shape))]. Streams bytes straight through."""
    hdr, off = {}, 0
    for name, (_p, a, b, dt, shape) in entries:
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + (b - a)]}
        off += b - a
    blob = json.dumps(hdr, separators=(",", ":")).encode()
    blob += b" " * ((-((8 + len(blob)) % 8)) % 8)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _name, (p, a, b, _dt, _shape) in entries:
            src = mm(p)
            o = a
            while o < b:
                n = min(1 << 24, b - o)
                f.write(src[o:o + n])
                o += n
    os.replace(tmp, path)
    return off


def shard_ok(path, entries):
    try:
        sz = os.path.getsize(path)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
    except Exception:
        return False
    if set(hdr) != {e[0] for e in entries}:
        return False
    end = max(v["data_offsets"][1] for v in hdr.values())
    return sz == 8 + n + end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="zai-org/GLM-5.2-FP8 checkpoint (block-fp8 on disk)")
    ap.add_argument("--base", default="/home/lava/models/GLM-5.2-plow", help="existing prepped dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=78)
    ap.add_argument("--first-k-dense", type=int, default=3)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    # symlink the base dir's shards + config so the harness sees ONE complete weight dir
    for fn in sorted(os.listdir(a.base)):
        dst = os.path.join(a.out, fn)
        if os.path.lexists(dst):
            continue
        if fn.endswith(".safetensors"):
            os.symlink(os.path.join(a.base, fn), dst)
        elif fn in ("config.json", "model.safetensors.index.json"):
            os.symlink(os.path.join(a.base, fn), dst)

    src = index_shards(a.src)
    t0, tot = time.time(), 0
    for L in range(a.layers):
        p = f"model.layers.{L}."
        want = [(p + "self_attn.o_proj.weight", p + "self_attn.o_proj.weight_fp8")]
        if L >= a.first_k_dense:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                want.append((p + f"mlp.shared_experts.{proj}.weight",
                             p + f"mlp.shared_experts.{proj}.weight_fp8"))
        entries = []
        for sname, oname in want:
            rec = src[sname]
            assert rec[3] == "F8_E4M3", (sname, rec[3])
            entries.append((oname, rec))
            srec = src[sname + "_scale_inv"]
            assert srec[3] == "F32", (sname, srec[3])
            entries.append((oname.replace("_fp8", "_scale_inv"), srec))
        path = os.path.join(a.out, f"model-idx-{L:05d}-of-idx.safetensors")
        if shard_ok(path, entries):
            tot += os.path.getsize(path)
            continue
        tot += write_shard(path, entries)
        print(f"[fp8lin] layer {L:2d}: {len(entries)} tensors, {tot/1e9:.2f} GB cum, "
              f"{time.time()-t0:.0f}s", flush=True)
    print(f"[fp8lin] DONE {tot/1e9:.2f} GB in {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
