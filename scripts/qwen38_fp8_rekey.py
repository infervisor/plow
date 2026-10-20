#!/usr/bin/env python3
"""Quantize and re-key a BF16 Qwen3.8 / Qwen3.5 checkpoint to Plow's FP8 contract.

For projection matrices:
  W [N, K] (BF16) -> fp8/W (F8_E4M3, [N, K]) + fp8/W_scale (F32, [N])
  scale[n] = max(abs(W[n, :])) / 448.0
All other tensors (embeddings, norms, convs, lm_head) are preserved verbatim.
Output is a complete safetensors checkpoint (sharded matching the input structure)
with index and symlinked configuration files.
"""

import json
import os
import sys
import torch
import safetensors.torch
from safetensors import safe_open

E4M3_MAX = 448.0

QUANT_PROJS = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.out_proj.weight",
}


def is_quant_target(name: str) -> bool:
    return any(name.endswith(proj) for proj in QUANT_PROJS)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <src_bf16_dir> <out_fp8_dir>")
        sys.exit(1)

    src_dir = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2])
    os.makedirs(out_dir, exist_ok=True)

    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]
        weight_map = None

    new_weight_map = {}
    total_bytes = 0

    print(f"Rekeying BF16 -> FP8 from {src_dir} to {out_dir}")
    print(f"Found {len(shards)} shards to process.")

    for i, shard_file in enumerate(shards):
        shard_src = os.path.join(src_dir, shard_file)
        shard_dst = os.path.join(out_dir, shard_file)
        print(f"[{i+1}/{len(shards)}] Processing {shard_file}...")

        out_tensors = {}
        with safe_open(shard_src, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if is_quant_target(k) and t.ndim == 2:
                    N, K = t.shape
                    t_f32 = t.float()
                    amax = t_f32.abs().amax(dim=1)
                    scale = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax)).to(torch.float32)
                    q = (t_f32 / scale.unsqueeze(1)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

                    w_name = f"fp8/{k}"
                    s_name = f"fp8/{k}_scale"
                    out_tensors[w_name] = q
                    out_tensors[s_name] = scale
                    new_weight_map[w_name] = shard_file
                    new_weight_map[s_name] = shard_file
                    total_bytes += q.numel() + scale.numel() * 4
                else:
                    out_tensors[k] = t
                    new_weight_map[k] = shard_file
                    total_bytes += t.numel() * t.element_size()

        safetensors.torch.save_file(out_tensors, shard_dst)

    # Write model.safetensors.index.json
    new_idx = {
        "metadata": {"total_size": total_bytes},
        "weight_map": new_weight_map,
    }
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_idx, f, indent=2)

    # Symlink auxiliary files
    for fname in os.listdir(src_dir):
        if fname.endswith((".json", ".jinja", ".txt", ".model")) and not fname.endswith(".safetensors.index.json"):
            dst_link = os.path.join(out_dir, fname)
            if not os.path.exists(dst_link):
                os.symlink(os.path.join(src_dir, fname), dst_link)

    print(f"Done! Rekeyed FP8 checkpoint ready at {out_dir} (total {total_bytes / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
