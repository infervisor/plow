#!/usr/bin/env python3
# dsv4_tensor_inventory.py — regenerate docs/amd/deepseek-v4-flash-tensor-inventory.json.
#
# Reads the safetensors HEADERS of a DeepSeek-V4-Flash checkpoint (never the payload; the
# 48 shards are 156 GB) and emits a lossless grouped inventory: one entry per tensor-name
# TEMPLATE with the explicit instance indices, per-variant shape, dtype, byte totals and
# the shard span each group lands in. `model.safetensors.index.json` carries the flat
# name -> shard map; this file carries the shapes and dtypes it does not.
#
# It also emits the tensor-parallel per-rank budget the capacity plan quotes, so the
# doc's arithmetic is re-derivable rather than transcribed.
#
# Usage:
#   python3 scripts/dsv4_tensor_inventory.py \
#       --model /workspace/models/DeepSeek-V4-Flash-0731 \
#       --out docs/amd/deepseek-v4-flash-tensor-inventory.json
import argparse, collections, glob, json, os, re, struct

GiB = 1 << 30

# Sharding policy per tensor, keyed on the full name. See the capacity plan's §4 for why
# each one is what it is; the load-bearing entries are the REPLICATED KV latent (one KV
# head cannot be split) and the o-LoRA, whose block-diagonal groups are indivisible.
SHARD_POLICY = [
    (r"^(layers\.\d+|mtp\.\d+)\.attn\.wq_b\.(weight|scale)$", "q_head"),
    (r"^(layers\.\d+|mtp\.\d+)\.attn\.attn_sink$", "q_head"),
    (r"^(layers\.\d+|mtp\.\d+)\.attn\.wo_(a|b)\.(weight|scale)$", "o_group"),
    (r"^layers\.\d+\.attn\.indexer\.wq_b\.(weight|scale)$", "index_head"),
    (r"^layers\.\d+\.attn\.indexer\.weights_proj\.weight$", "index_head"),
    (r"^(layers\.\d+|mtp\.\d+)\.ffn\.experts\.\d+\.", "expert"),
    (r"^(layers\.\d+|mtp\.\d+)\.ffn\.shared_experts\.", "moe_inter"),
    (r"^(embed|head)\.weight$", "vocab"),
    (r"^mtp\.\d+\.markov_head\.", "vocab"),
    (r"^mtp\.\d+\.main_proj\.(weight|scale)$", "moe_inter"),
]

ROLE_RULES = [
    (r"^embed\.weight$", "embed"),
    (r"^head\.weight$", "lm_head"),
    (r"^norm\.weight$", "final_norm"),
    (r"^hc_head_", "mhc_head"),
    (r"\.ffn\.experts\.\d+\.", "moe_routed"),
    (r"\.ffn\.shared_experts\.", "moe_shared"),
    (r"\.ffn\.gate\.", "moe_router"),
    (r"\.attn\.indexer\.", "indexer"),
    (r"\.attn\.compressor\.", "csa_compressor"),
    (r"\.attn\.(wq_|q_norm\.)", "attn_q"),
    (r"\.attn\.(wkv|kv_norm)", "attn_kv"),
    (r"\.attn\.wo_", "attn_o"),
    (r"\.attn\.attn_sink$", "attn_sink"),
    (r"\.hc_", "mhc"),
    (r"\.(attn_norm|ffn_norm|main_norm|norm)\.weight$", "norm"),
    (r"\.(markov_head|confidence_head)\.", "dspark_head"),
    (r"\.main_proj\.", "dspark_input"),
]


def policy(name):
    for pat, p in SHARD_POLICY:
        if re.match(pat, name):
            return p
    return "replicated"


def role(name):
    for pat, r in ROLE_RULES:
        if re.search(pat, name):
            return r
    return "other"


def template(name):
    t = re.sub(r"^layers\.\d+\.", "layers.{L}.", name)
    t = re.sub(r"^mtp\.\d+\.", "mtp.{B}.", t)
    return re.sub(r"\.experts\.\d+\.", ".experts.{E}.", t)


def instance(name):
    m = re.match(r"^(layers|mtp)\.(\d+)\.", name)
    return int(m.group(2)) if m else None


def read_headers(model):
    out = {}
    for path in sorted(glob.glob(os.path.join(model, "*.safetensors"))):
        shard = os.path.basename(path)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        for k, v in head.items():
            if k == "__metadata__":
                continue
            off = v["data_offsets"]
            out[k] = {
                "shape": v["shape"],
                "dtype": v["dtype"],
                "shard": shard,
                "bytes": off[1] - off[0],
            }
    return out


def build(model, tensors, cfg):
    groups = collections.defaultdict(
        lambda: {"instances": [], "variants": collections.defaultdict(int), "shards": set(),
                 "bytes": 0, "count": 0}
    )
    for name, v in tensors.items():
        t = template(name)
        g = groups[t]
        g["count"] += 1
        g["bytes"] += v["bytes"]
        g["shards"].add(v["shard"])
        g["variants"][(tuple(v["shape"]), v["dtype"])] += 1
        i = instance(name)
        if i is not None and i not in g["instances"]:
            g["instances"].append(i)

    ratios = cfg["compress_ratios"]
    out_groups = []
    for t in sorted(groups):
        g = groups[t]
        inst = sorted(g["instances"])
        concrete = t.replace("{L}", "0").replace("{B}", "0").replace("{E}", "0")
        variants = []
        for (shape, dtype), n in sorted(g["variants"].items(), key=lambda x: -x[1]):
            variants.append({"shape": list(shape), "dtype": dtype, "tensors": n})
        shards = sorted(g["shards"])
        out_groups.append({
            "template": t,
            "role": role(concrete),
            "shard_policy": policy(concrete),
            "tensors": g["count"],
            "bytes": g["bytes"],
            "variants": variants,
            "instances": inst,
            "compress_ratios_of_instances": (
                sorted({ratios[i] for i in inst}) if t.startswith("layers.") and inst else None
            ),
            "shard_files": [shards[0], shards[-1]] if shards else [],
        })

    by_dtype = collections.defaultdict(lambda: [0, 0])
    by_role = collections.defaultdict(lambda: [0, 0])
    by_policy = collections.defaultdict(int)
    total = 0
    for name, v in tensors.items():
        by_dtype[v["dtype"]][0] += v["bytes"]
        by_dtype[v["dtype"]][1] += 1
        r = role(name)
        by_role[r][0] += v["bytes"]
        by_role[r][1] += 1
        by_policy[policy(name)] += v["bytes"]
        total += v["bytes"]

    replicated = by_policy.pop("replicated", 0)
    sharded = sum(by_policy.values())
    tp = {}
    for degree in (1, 2, 4, 8):
        tp[f"tp{degree}"] = {
            "replicated_bytes": replicated,
            "sharded_bytes_per_rank": sharded // degree,
            "weights_bytes_per_rank": replicated + sharded // degree,
            "weights_gib_per_rank": round((replicated + sharded / degree) / GiB, 3),
        }

    return {
        "model": os.path.basename(os.path.abspath(model)),
        "tensors": len(tensors),
        "shards": len(sorted({v["shard"] for v in tensors.values()})),
        "total_bytes": total,
        "total_gib": round(total / GiB, 3),
        "by_dtype": {
            d: {"tensors": n, "bytes": b, "gib": round(b / GiB, 3)}
            for d, (b, n) in sorted(by_dtype.items(), key=lambda x: -x[1][0])
        },
        "by_role": {
            r: {"tensors": n, "bytes": b, "gib": round(b / GiB, 3)}
            for r, (b, n) in sorted(by_role.items(), key=lambda x: -x[1][0])
        },
        "by_shard_policy": {
            "replicated": {"bytes": replicated, "gib": round(replicated / GiB, 3)},
            **{
                k: {"bytes": b, "gib": round(b / GiB, 3)}
                for k, b in sorted(by_policy.items(), key=lambda x: -x[1])
            },
        },
        "tensor_parallel_weight_budget": tp,
        "groups": out_groups,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/DeepSeek-V4-Flash-0731")
    ap.add_argument("--out", default="docs/amd/deepseek-v4-flash-tensor-inventory.json")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    tensors = read_headers(args.model)
    doc = build(args.model, tensors, cfg)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"{doc['tensors']} tensors, {doc['total_gib']} GiB -> {args.out}")


if __name__ == "__main__":
    main()
