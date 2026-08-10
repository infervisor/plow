#!/usr/bin/env python3
# glm52_prep_lite.py — space-efficient GLM-5.2 weight prep for boxes that cannot hold a second
# copy of the ~700 GB routed experts (glm52_prep_full.py writes them VERBATIM into new shards;
# on an 8.7 TB box that already holds the raw checkpoint there is no room for the copy).
#
# The trick is plowrt's checkpoint loader contract (asset/checkpoint.rs): it scans
# `*.safetensors` in SORTED order and builds its name index with plain HashMap inserts, so a
# tensor present in TWO shards resolves to the LATER file. This prep therefore:
#
#   1. SYMLINKS every raw shard (model-*.safetensors) into the out dir — the routed experts
#      (block-fp8 .weight + .weight_scale_inv) resolve from them byte-verbatim, exactly the
#      bytes glm52_prep_full.py would have copied;
#   2. writes ONLY the derived/dequantised tensors (bf16 MLA projections, shared experts,
#      norms, indexer, globals) into `zz-derived-*.safetensors` shards, whose names sort AFTER
#      `model-*` so their entries SHADOW the same-named fp8 originals.
#
# Output is ~40 GB of new data instead of ~750. The per-layer content is produced by the SAME
# `glm52_prep.prep_layer` the full prep and the ms1 gate use — only expert-named entries are
# filtered out of the writer, everything else is byte-identical to the full prep.
#
# Usage:
#   python3 scripts/glm52_prep_lite.py --model /workspace/models/GLM-5.2-FP8 \
#       --out /workspace/models/GLM-5.2-plow-lite [--layers 0-77]
#
# Resumable: a complete zz-shard is skipped on re-run (same integrity check as the full prep).
import os, sys, re, json, struct, argparse, time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import glm52_prep as P
from glm52_prep_full import _shard_ok, _globals_builder, _parse_layers, N_LAYERS

EXPERT_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.")


class FilterWriter(P.STWriter):
    """STWriter that drops routed-expert entries — they resolve verbatim from the symlinked
    raw shards, so writing them would only duplicate ~9 GB per layer."""

    def add(self, name, dtype_str, shape, nbytes, producer):
        if EXPERT_RE.match(name):
            return
        super().add(name, dtype_str, shape, nbytes, producer)


def _write_filtered(build, path):
    w = FilterWriter()
    build(w)
    tmp = f"{path}.{os.getpid()}.tmp"
    w.flush(tmp)
    os.replace(tmp, path)
    return len(w.entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=None, help="e.g. 0-5 for a smoke slice; default all 78")
    args = ap.parse_args()
    layers = _parse_layers(args.layers) if args.layers else list(range(N_LAYERS))

    os.makedirs(args.out, exist_ok=True)
    # 1. symlink raw shards (idempotent) + config for provenance.
    linked = 0
    for fn in sorted(os.listdir(args.model)):
        if fn.endswith(".safetensors") or fn in ("config.json", "tokenizer.json"):
            dst = os.path.join(args.out, fn)
            if not os.path.exists(dst):
                os.symlink(os.path.join(args.model, fn), dst)
                linked += 1
    print(f"[lite] symlinked {linked} raw files into {args.out}")

    idx = P._index_shards(args.model)
    cfg = P.load_cfg(args.model)

    # 2. derived shards, one per layer, shadowing the fp8 originals by sort order.
    for L in layers:
        path = os.path.join(args.out, f"zz-derived-{L:05d}.safetensors")
        ok, _ = _shard_ok(path)
        if ok:
            print(f"[lite] layer {L}: shard complete, skipped")
            continue
        t0 = time.time()
        n = _write_filtered(lambda w: P.prep_layer(idx, cfg, w, 512, L, 511), path)
        print(f"[lite] layer {L}: {n} derived tensors in {time.time()-t0:.1f}s")

    # 3. globals (embed / final norm / lm_head).
    gpath = os.path.join(args.out, "zz-globals.safetensors")
    ok, _ = _shard_ok(gpath)
    if not ok:
        t0 = time.time()
        n = _write_filtered(_globals_builder(idx, cfg), gpath)
        print(f"[lite] globals: {n} tensors in {time.time()-t0:.1f}s")
    print("[lite] done")


if __name__ == "__main__":
    main()
