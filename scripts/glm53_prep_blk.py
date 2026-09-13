#!/usr/bin/env python3
"""Publish the checkpoint block-FP8 form of the four GLM-5.3 prefill projections the block-scale
FP8 route (`PLOW_GLM_GEMM_BLK`) reads, as an overlay on the served weight dir.

Everything here is a VERBATIM byte copy of `GLM-5.3-FP8`: no dequant, no requant.
  q_a_proj.weight_fp8                  <- q_a_proj.weight                       [2048,6144]
  o_proj.weight_fp8                    <- o_proj.weight                         [6144,16384]
  derived.kv_a_latent.weight_fp8       <- kv_a_proj_with_mqa.weight rows 0..511 [512,6144]
  derived.kv_a_latent.weight_scale_inv <- its scale grid rows 0..3              [4,48]
  indexer.wq_b.weight_fp8              <- indexer.wq_b.weight                   [4096,2048]
The kv_a slice is exact because 512 = 4 x 128: rows 0..511 are a contiguous prefix of the weight
and block-rows 0..3 a contiguous prefix of its [5,48] grid. The other grids
(`q_a_proj`/`o_proj`/`indexer.wq_b` `.weight_scale_inv`) are already visible through the base
dir's raw-shard links and `zz-derived` sidecars under their checkpoint names, so they are not
duplicated here. `q_absorb`, `q_rope` and `v_absorb` are prep products with no grid and stay BF16.

The output dir symlinks every file of `--base` and adds `zz3-blk-*` sidecars, so the production
dir is not touched and the production packet's checkpoint is unchanged.

  python3 scripts/glm53_prep_blk.py --src /workspace/models/GLM-5.3-FP8 \
      --base /workspace/models/GLM-5.3-plow-lite --out /workspace/models/GLM-5.3-plow-blk
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import glm52_prep_fp8_linear as L  # noqa: E402

KV_LATENT = 512
BLOCK = 128


def rows_prefix(rec, rows):
    path, a, b, dtype, shape = rec
    row_bytes = (b - a) // shape[0]
    assert row_bytes * shape[0] == b - a and rows <= shape[0], (rec, rows)
    return (path, a, a + rows * row_bytes, dtype, [rows] + list(shape[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=78)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    for fn in sorted(os.listdir(a.base)):
        dst = os.path.join(a.out, fn)
        if not os.path.lexists(dst) and (fn.endswith(".safetensors") or fn.endswith(".json")):
            os.symlink(os.path.join(a.base, fn), dst)

    src = L.index_shards(a.src)
    t0, tot = time.time(), 0
    for layer in range(a.layers):
        p = f"model.layers.{layer}.self_attn."
        entries = []
        for name in ("q_a_proj", "o_proj"):
            rec = src[p + name + ".weight"]
            assert rec[3] == "F8_E4M3", (name, rec[3])
            entries.append((p + name + ".weight_fp8", rec))
        kv = src[p + "kv_a_proj_with_mqa.weight"]
        kvs = src[p + "kv_a_proj_with_mqa.weight_scale_inv"]
        assert kv[3] == "F8_E4M3" and kvs[3] == "F32", (kv[3], kvs[3])
        assert kvs[4][0] == -(-kv[4][0] // BLOCK), (kv[4], kvs[4])
        entries.append((p + "derived.kv_a_latent.weight_fp8", rows_prefix(kv, KV_LATENT)))
        entries.append((p + "derived.kv_a_latent.weight_scale_inv",
                        rows_prefix(kvs, KV_LATENT // BLOCK)))
        wq_b = src.get(p + "indexer.wq_b.weight")
        if wq_b is not None:
            assert wq_b[3] == "F8_E4M3", wq_b[3]
            entries.append((p + "indexer.wq_b.weight_fp8", wq_b))
        path = os.path.join(a.out, f"zz3-blk-{layer:05d}.safetensors")
        if L.shard_ok(path, entries):
            tot += os.path.getsize(path)
            continue
        tot += L.write_shard(path, entries)
        print(f"[blk] layer {layer:2d}: {len(entries)} tensors, {tot / 1e9:.2f} GB cum, "
              f"{time.time() - t0:.0f}s", flush=True)
    print(f"[blk] DONE {tot / 1e9:.2f} GB in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
