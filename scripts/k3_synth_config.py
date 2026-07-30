#!/usr/bin/env python3
"""Write a config.json with Kimi-K3's REAL geometry, for deriving tuning demand.

`plowc tune gemm --shapes auto` derives its shape list by running a real emit, so it needs a
checkpoint DIRECTORY to point `--hf-dir` at. There is no K3 checkpoint on this box, and the
shapes are a function of the CONFIG alone -- the emitter reads geometry, not tensors -- so this
writes the config and nothing else. It is not a checkpoint and cannot be served; it exists so a
campaign measures K3's shapes instead of a hand-guessed approximation of them.

Geometry, and where each number is load-bearing for the GEMM shapes:

  hidden 7168                     the K of every projection off the residual stream
  q_lora 1536 / kv_lora 512       the MLA low-rank waists; these do NOT shard under TP
  routed_expert_hidden_size 3584  the LATENT width the routed experts run at
  moe_intermediate 3072           per-expert FFN width; shards by TP
  intermediate 18432              the dense (first_k_dense_replace) FFN width
  vocab 163840                    the lm_head N
  896 experts, top-16             routing, not a GEMM dimension
  93 layers = 24 MLA + 69 KDA

`full_attn_layers`/`kda_layers` are **1-BASED** and must PARTITION 1..93 -- `hf_config.rs`
checks both and refuses a gap or an overlap. The MLA stride is every 4th layer, which is what
the real checkpoint shows (0-based 3, 7, 11, ...); 93 is not a multiple of 4, so the last layer
joins the MLA set to reach the documented 24/69 split. WHICH layers are MLA does not move the
shape list at all -- `tune_demand::distinct` dedupes, so layer placement only changes how often
a shape is asked -- but the partition has to be exact or the config is rejected.

    usage: k3_synth_config.py [outdir]      (default /tmp/k3-synth)
"""
import json
import os
import sys

L = 93
FULL = list(range(4, 93, 4)) + [93]  # 1-BASED, 24 MLA layers
KDA = [i for i in range(1, L + 1) if i not in set(FULL)]
assert len(FULL) == 24 and len(KDA) == 69
assert set(FULL) | set(KDA) == set(range(1, L + 1)) and not (set(FULL) & set(KDA))

CONFIG = {
    "model_type": "kimi_k3",
    "text_config": {
        "model_type": "kimi_linear",
        "hidden_size": 7168,
        "num_attention_heads": 64,
        "num_hidden_layers": L,
        "vocab_size": 163840,
        "intermediate_size": 18432,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_shared_experts": 2,
        "moe_intermediate_size": 3072,
        "routed_expert_hidden_size": 3584,
        "first_k_dense_replace": 1,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-05,
        # NO `rope_theta`. K3's MLA is NoPE (`mla_use_nope: true`) and the KDA layers carry
        # position; `k3.rs` records that text_config carries no rope_theta, and `mla.rs` reads
        # its absence as a FACT ("this model has no RoPE") rather than defaulting it.
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "attn_res_block_size": 12,
        "latent_moe_use_norm": True,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "routed_scaling_factor": 2.5,
        "num_expert_group": 1,
        "topk_group": 1,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "torch_dtype": "bfloat16",
        # `parse_weight_dtype` reads this from text_config for kimi_k3; the nested
        # `config_groups` is where `k3_cfg_from` reads the group size and bit width.
        "quantization_config": {
            "quant_method": "mxfp4",
            "format": "mxfp4-pack-quantized",
            "config_groups": {"group_0": {"weights": {"group_size": 32, "num_bits": 4}}},
        },
        "linear_attn_config": {
            "full_attn_layers": FULL,
            "kda_layers": KDA,
            # KDA geometry as `devgen::k3`'s own fixture records it (96 heads x 128, conv 4).
            "num_heads": 96,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": False,
            "gate_lower_bound": -5.0,
        },
    },
}

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/k3-synth"
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2)
print(f"wrote {out}/config.json  ({L} layers: {len(FULL)} MLA / {len(KDA)} KDA)")
