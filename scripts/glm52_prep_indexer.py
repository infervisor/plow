#!/usr/bin/env python3
# glm52_prep_indexer.py — INCREMENTAL DSA-indexer prep.                              [GLM52-DSA G1]
#
# Adds ONLY the 7 lightning-indexer tensors per 'full' layer to an EXISTING plow-ready dir
# (/home/lava/models/GLM-5.2-FP8-plow) WITHOUT recopying the ~715 GB base weights. The base layer
# shards are already prepped and unchanged; the indexer tensors are tiny (~9 MB/layer, ~190 MB total).
#
# It writes one small shard per full layer (model-idx-{L:05d}-of-idx.safetensors) into the dir and
# MERGES their tensors into model.safetensors.index.json (append-only: existing weight_map entries are
# preserved, indexer entries added). The single-dir loader then binds indexer tensors by name exactly
# like every other tensor — no driver change, no side dir.
#
# Usage:
#   nix develop -c python3 scripts/glm52_prep_indexer.py \
#       --model /home/lava/models/GLM-5.2-FP8 --out /home/lava/models/GLM-5.2-FP8-plow
#   nix develop -c python3 scripts/glm52_prep_indexer.py --out <dir> --verify-only
import os, sys, json, struct, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import glm52_prep as P
from transformers import AutoConfig

IDX_SHARD = "model-idx-{:05d}-of-idx.safetensors"
IX_SUFFIXES = ("wq_b.weight", "wq_b.weight_scale_inv", "wk.weight", "wk.weight_scale_inv",
               "k_norm.weight", "k_norm.bias", "weights_proj.weight")


def full_layers(cfg):
    return [l for l, t in enumerate(cfg.indexer_types) if t == "full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/lava/models/GLM-5.2-FP8")
    ap.add_argument("--out", default="/home/lava/models/GLM-5.2-FP8-plow")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model)
    fulls = full_layers(cfg)
    idxjson = os.path.join(args.out, "model.safetensors.index.json")
    with open(idxjson) as f:
        index = json.load(f)
    wmap = index["weight_map"]

    # names the emitter (declare_glm) binds per full layer.
    want = {f"model.layers.{l}.self_attn.indexer.{suf}" for l in fulls for suf in IX_SUFFIXES}

    if args.verify_only:
        missing = {n for n in want if n not in wmap}
        print(f"[idx-verify] full layers {len(fulls)}, indexer tensors expected {len(want)}, "
              f"present {len(want) - len(missing)}, missing {len(missing)}")
        if missing:
            print("  MISSING (sample):", sorted(missing)[:6])
        sys.exit(0 if not missing else 1)

    srcidx = P._index_shards(args.model)
    added = 0
    for l in fulls:
        shard = IDX_SHARD.format(l)
        path = os.path.join(args.out, shard)
        # skip if this layer's indexer tensors are already mapped to a present shard.
        names = [f"model.layers.{l}.self_attn.indexer.{suf}" for suf in IX_SUFFIXES]
        if all(n in wmap for n in names) and os.path.exists(os.path.join(args.out, wmap[names[0]])):
            print(f"[idx] layer {l:2d}: present — skip")
            continue
        w = P.STWriter()
        P.add_indexer_tensors(srcidx, cfg, w, l)
        tmp = f"{path}.{os.getpid()}.tmp"
        w.flush(tmp)
        os.replace(tmp, path)
        for name, _dt, _shape, nb, _prod in w.entries:
            wmap[name] = shard
            index["metadata"]["total_size"] = index["metadata"].get("total_size", 0) + nb
        added += 1
        print(f"[idx] layer {l:2d}: wrote {shard} ({os.path.getsize(path)/1e6:.1f} MB, {len(w.entries)} tensors)")

    # publish the merged index ATOMICALLY (append-only: base entries untouched; the plow dir is shared,
    # so tmp+replace avoids a torn read by a concurrent loader).
    tmpj = f"{idxjson}.{os.getpid()}.tmp"
    with open(tmpj, "w") as f:
        json.dump(index, f, indent=1)
    os.replace(tmpj, idxjson)
    print(f"[idx] DONE — {added} indexer shards added, {len(want)} indexer tensors mapped -> {args.out}")
    missing = {n for n in want if n not in wmap}
    print(f"[idx] verify: {'ALL PRESENT' if not missing else f'*** {len(missing)} MISSING ***'}")


if __name__ == "__main__":
    main()
