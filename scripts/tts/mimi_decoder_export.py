#!/usr/bin/env python3
"""Export a Kyutai Mimi DECODER (HF `MimiModel`, e.g. kyutai/mimi for Maya 2 Native) for plowc
(crates/devgen/src/fish_codec.rs, model_type `mimi_dac`).

Writes OUT_DIR/model.safetensors (fp32, torch layouts, folds in fp64) and OUT_DIR/config.json:
  vq.T{k}                   [codebook][hidden] codebook k's centroids (embed_sum / clamp(usage, 1e-5))
                            through its group's output_proj (no bias): decode = sum_k T_k[code_k]
                            (k = 0 semantic, 1.. acoustic)
  pre_up.w                  [hidden][hidden][4] the depthwise (groups = hidden) causal ConvTranspose
                            k4 s2 expanded block-diagonal (no bias)
  post.layers.{l}.ln1.{w,b} / .ln2.{w,b}  LayerNorm; .wqkv [3 * hidden][hidden]
  post.layers.{l}.wo_s / .fc2_s           LayerScale-folded o_proj / fc2; .fc1 (GELU erf between)
  dec.in.{w,b}              causal conv k7 hidden -> channels
  dec.blk{i}.up.{w,b}       ELU, causal ConvTranspose k = 2r, stride r, d -> d / 2
  dec.blk{i}.ru0.c1.{w,b}   ELU, causal conv k3 d/2 -> d/4;  .c2.{w,b}  ELU, conv k1 d/4 -> d/2, + x
  dec.out.{w,b}             ELU, causal conv k3 -> 1

  HF_HOME=... python scripts/tts/mimi_decoder_export.py OUT_DIR REPO REVISION
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    out, repo, rev = sys.argv[1:]
    os.makedirs(out, exist_ok=True)
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    src = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", rev)
    sd = load_file(f"{src}/model.safetensors")
    c = json.load(open(f"{src}/config.json"))
    T = {}

    def put(name, t):
        T[name] = t.detach().double().float().contiguous()

    assert c["use_causal_conv"] and not c["use_conv_shortcut"] and c["num_residual_layers"] == 1
    assert c["audio_channels"] == 1 and c["num_key_value_heads"] == c["num_attention_heads"]
    assert c["upsample_groups"] == c["hidden_size"] and c["trim_right_ratio"] == 1.0
    hidden, sem = c["hidden_size"], c["num_semantic_quantizers"]
    q = "quantizer.{}_residual_vector_quantizer"
    k = 0
    for g in ("semantic", "acoustic"):
        proj = sd[f"{q.format(g)}.output_proj.weight"][:, :, 0].double()  # [hidden][cb_dim]
        i = 0
        while f"{q.format(g)}.layers.{i}.codebook.embed_sum" in sd:
            cb = f"{q.format(g)}.layers.{i}.codebook"
            e = sd[f"{cb}.embed_sum"].double() / sd[f"{cb}.cluster_usage"].double().clamp(min=1e-5)[:, None]
            put(f"vq.T{k}", e @ proj.T)
            k += 1
            i += 1
    codebooks = k

    w = sd["upsample.conv.weight"].double()  # [hidden][1][4] depthwise ConvTranspose
    dense = torch.zeros(hidden, hidden, w.shape[-1], dtype=torch.float64)
    dense[torch.arange(hidden), torch.arange(hidden)] = w[:, 0]
    put("pre_up.w", dense)

    for l in range(c["num_hidden_layers"]):
        p, o = f"decoder_transformer.layers.{l}", f"post.layers.{l}"
        a = f"{p}.self_attn"
        put(f"{o}.ln1.w", sd[f"{p}.input_layernorm.weight"])
        put(f"{o}.ln1.b", sd[f"{p}.input_layernorm.bias"])
        put(f"{o}.ln2.w", sd[f"{p}.post_attention_layernorm.weight"])
        put(f"{o}.ln2.b", sd[f"{p}.post_attention_layernorm.bias"])
        put(f"{o}.wqkv", torch.cat([sd[f"{a}.{s}_proj.weight"] for s in "qkv"]))
        put(f"{o}.wo_s", sd[f"{p}.self_attn_layer_scale.scale"].double()[:, None] * sd[f"{a}.o_proj.weight"].double())
        put(f"{o}.fc1", sd[f"{p}.mlp.fc1.weight"])
        put(f"{o}.fc2_s", sd[f"{p}.mlp_layer_scale.scale"].double()[:, None] * sd[f"{p}.mlp.fc2.weight"].double())

    ratios = c["upsampling_ratios"]
    put("dec.in.w", sd["decoder.layers.0.conv.weight"])
    put("dec.in.b", sd["decoder.layers.0.conv.bias"])
    blocks, li, d = [], 2, sd["decoder.layers.0.conv.weight"].shape[0]
    for i, r in enumerate(ratios):
        put(f"dec.blk{i}.up.w", sd[f"decoder.layers.{li}.conv.weight"])
        put(f"dec.blk{i}.up.b", sd[f"decoder.layers.{li}.conv.bias"])
        put(f"dec.blk{i}.ru0.c1.w", sd[f"decoder.layers.{li + 1}.block.1.conv.weight"])
        put(f"dec.blk{i}.ru0.c1.b", sd[f"decoder.layers.{li + 1}.block.1.conv.bias"])
        put(f"dec.blk{i}.ru0.c2.w", sd[f"decoder.layers.{li + 1}.block.3.conv.weight"])
        put(f"dec.blk{i}.ru0.c2.b", sd[f"decoder.layers.{li + 1}.block.3.conv.bias"])
        blocks.append(dict(cin=d, cout=d // 2, kernel=2 * r, stride=r, dilations=[1]))
        d //= 2
        li += 3
    put("dec.out.w", sd[f"decoder.layers.{li}.conv.weight"])
    put("dec.out.b", sd[f"decoder.layers.{li}.conv.bias"])

    save_file(T, os.path.join(out, "model.safetensors"))
    up = 2
    for r in ratios:
        up *= r
    config = dict(
        model_type="mimi_dac", sampling_rate=c["sampling_rate"], frame_samples=up,
        codebook_sizes=[c["codebook_size"]] * codebooks,
        transformer=dict(layers=c["num_hidden_layers"], dim=hidden, heads=c["num_attention_heads"],
                         head_dim=c["head_dim"], intermediate=c["intermediate_size"], eps=c["norm_eps"],
                         window=c["sliding_window"], rope_theta=c["rope_theta"]),
        pre_upsample=dict(stride=2, kernel=4), upsample=[],
        decoder=dict(channels=sd["decoder.layers.0.conv.weight"].shape[0], blocks=blocks,
                     residual_kernel=c["residual_kernel_size"], out_kernel=c["last_kernel_size"]),
    )
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    print(f"wrote {out}: {len(T)} tensors, {codebooks} codebooks")


if __name__ == "__main__":
    main()
