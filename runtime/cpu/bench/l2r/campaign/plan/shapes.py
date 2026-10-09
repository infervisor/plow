"""shapes.py: per-layer weight bytes and structure of the big Gemma 4 checkpoints (layer 0 sliding, first full layer)."""
import json, struct, os, collections
for m in ("12B", "26B-A4B", "31B"):
    d = f"/tmp/models/google/gemma-4-{m}-it"
    cfg = json.load(open(f"{d}/config.json"))
    t = cfg.get("text_config", cfg)
    keys = ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "num_global_key_value_heads", "head_dim", "global_head_dim", "sliding_window", "attention_k_eq_v",
            "num_experts", "top_k_experts", "moe_intermediate_size", "enable_moe_block", "hidden_size_per_layer_input",
            "vocab_size", "tie_word_embeddings"]
    print(f"== {m}", {k: t.get(k) for k in keys if k in t})
    lt = t.get("layer_types", [])
    full = [i for i, x in enumerate(lt) if x == "full_attention"]
    print("   full layers", full[:4], "... count", len(full))
    shapes = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".safetensors"):
            continue
        with open(f"{d}/{f}", "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                shapes[k] = (v["shape"], v["dtype"])
    tot = 0
    for L in (0, full[0] if full else 1):
        pre = f"model.language_model.layers.{L}."
        by = collections.OrderedDict()
        for k, (s, dt) in shapes.items():
            if k.startswith(pre):
                n = 1
                for x in s:
                    n *= x
                by[k[len(pre):]] = (s, n * 2)
        b = sum(v[1] for v in by.values())
        print(f"   layer {L} ({lt[L] if lt else '?'}): {b / 2**20:.1f} MiB = {b / 90 / 2**20:.2f} MiB/core over 90")
        for k, (s, nb) in by.items():
            if nb > 2**20:
                print(f"      {k:50s} {s} {nb / 2**20:.1f} MiB")
    other = sum(int(__import__('math').prod(s)) * 2 for k, (s, dt) in shapes.items() if ".layers." not in k)
    print(f"   non-layer {other / 2**30:.2f} GiB; total {sum(int(__import__('math').prod(s)) * 2 for s, _ in shapes.values()) / 2**30:.1f} GiB")
