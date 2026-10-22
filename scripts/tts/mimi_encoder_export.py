#!/usr/bin/env python3
"""Export Breeze TTS 2's reference-audio encoder (Qwen3-TTS-Tokenizer-12Hz `encoder.*` = HF Mimi) for
plowc (crates/devgen/src/mimi_encoder.rs).

Writes OUT_DIR/model.safetensors (fp32, torch layouts, folds in fp64) and OUT_DIR/config.json,
model_type mimi_encoder:
  enc.c0.{w,b}                 causal conv k7 1 -> filters
  enc.s{i}.r1.{w,b} / .r2.{w,b}  residual unit: ELU, conv k3 d -> d/2, ELU, conv k1 d/2 -> d, + x
  enc.s{i}.down.{w,b}          ELU, causal conv k = 2r, stride r, d -> 2d
  enc.out.{w,b}                ELU, causal conv k3 -> hidden
  xf.{l}.ln1.{w,b} / .ln2.{w,b}  LayerNorm; xf.{l}.wqkv [3 * hidden][hidden]
  xf.{l}.wo_s / .fc2_s         LayerScale-folded o_proj / fc2; xf.{l}.fc1 (GELU erf between)
  down.w                       causal conv k4 stride 2, replicate pad, no bias
  {sem,ac}.proj                [codebook_dim][hidden] input projection of each RVQ
  {sem,ac}.E{k} / .nb{k}       centroids embed_sum / clamp(cluster_usage, 1e-5) and -|e|^2 / 2: the
                               nearest centroid is the argmax of x . e - |e|^2 / 2

  HF_HOME=... python scripts/tts/mimi_encoder_export.py OUT_DIR REVISION
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

REPO = "models--BreezeBlue--Breeze-TTS-2"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    src = os.path.join(hub, REPO, "snapshots", sys.argv[2], "audio_tokenizer")
    sd = {k[len("encoder."):]: v for k, v in load_file(f"{src}/model.safetensors").items() if k.startswith("encoder.")}
    top = json.load(open(f"{src}/config.json"))
    ec = top["encoder_config"]
    T = {}

    def put(name, t):
        T[name] = t.detach().double().float().contiguous()

    assert ec["use_causal_conv"] and not ec["use_conv_shortcut"] and ec["num_residual_layers"] == 1
    assert ec["audio_channels"] == 1 and ec["num_key_value_heads"] == ec["num_attention_heads"]
    ratios = list(reversed(ec["upsampling_ratios"]))
    put("enc.c0.w", sd["encoder.layers.0.conv.weight"])
    put("enc.c0.b", sd["encoder.layers.0.conv.bias"])
    li = 1
    for i, _ in enumerate(ratios):
        put(f"enc.s{i}.r1.w", sd[f"encoder.layers.{li}.block.1.conv.weight"])
        put(f"enc.s{i}.r1.b", sd[f"encoder.layers.{li}.block.1.conv.bias"])
        put(f"enc.s{i}.r2.w", sd[f"encoder.layers.{li}.block.3.conv.weight"])
        put(f"enc.s{i}.r2.b", sd[f"encoder.layers.{li}.block.3.conv.bias"])
        put(f"enc.s{i}.down.w", sd[f"encoder.layers.{li + 2}.conv.weight"])
        put(f"enc.s{i}.down.b", sd[f"encoder.layers.{li + 2}.conv.bias"])
        li += 3
    put("enc.out.w", sd[f"encoder.layers.{li + 1}.conv.weight"])
    put("enc.out.b", sd[f"encoder.layers.{li + 1}.conv.bias"])

    for l in range(ec["num_hidden_layers"]):
        p, o = f"encoder_transformer.layers.{l}", f"xf.{l}"
        a = f"{p}.self_attn"
        put(f"{o}.ln1.w", sd[f"{p}.input_layernorm.weight"])
        put(f"{o}.ln1.b", sd[f"{p}.input_layernorm.bias"])
        put(f"{o}.ln2.w", sd[f"{p}.post_attention_layernorm.weight"])
        put(f"{o}.ln2.b", sd[f"{p}.post_attention_layernorm.bias"])
        put(f"{o}.wqkv", torch.cat([sd[f"{a}.{s}_proj.weight"] for s in "qkv"]))
        put(f"{o}.wo_s", sd[f"{p}.self_attn_layer_scale.scale"].double()[:, None] * sd[f"{a}.o_proj.weight"].double())
        put(f"{o}.fc1", sd[f"{p}.mlp.fc1.weight"])
        put(f"{o}.fc2_s", sd[f"{p}.mlp_layer_scale.scale"].double()[:, None] * sd[f"{p}.mlp.fc2.weight"].double())
    put("down.w", sd["downsample.conv.weight"])

    keep = top["encoder_valid_num_quantizers"]
    sem_n = ec["num_semantic_quantizers"]
    groups = [("sem", "semantic_residual_vector_quantizer", sem_n), ("ac", "acoustic_residual_vector_quantizer", keep - sem_n)]
    for g, q, n in groups:
        put(f"{g}.proj", sd[f"quantizer.{q}.input_proj.weight"][:, :, 0])
        for k in range(n):
            cb = f"quantizer.{q}.layers.{k}.codebook"
            e = sd[f"{cb}.embed_sum"].double() / sd[f"{cb}.cluster_usage"].double().clamp(min=1e-5)[:, None]
            put(f"{g}.E{k}", e)
            put(f"{g}.nb{k}", -0.5 * (e * e).sum(1))

    save_file(T, os.path.join(out, "model.safetensors"))
    config = dict(
        model_type="mimi_encoder", sampling_rate=ec["sampling_rate"],
        frame_samples=top["encode_downsample_rate"], filters=ec["num_filters"], ratios=ratios,
        kernel=ec["kernel_size"], last_kernel=ec["last_kernel_size"],
        residual_kernel=ec["residual_kernel_size"], compress=ec["compress"], hidden=ec["hidden_size"],
        layers=ec["num_hidden_layers"], heads=ec["num_attention_heads"], head_dim=ec["head_dim"],
        intermediate=ec["intermediate_size"], eps=ec["norm_eps"], rope_theta=ec["rope_theta"],
        downsample=dict(kernel=4, stride=2), codebook_dim=ec["codebook_dim"], codebook_size=ec["codebook_size"],
        semantic=sem_n, acoustic=keep - sem_n,
    )
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    print(f"wrote {out}: {len(T)} tensors")


if __name__ == "__main__":
    main()
