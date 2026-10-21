#!/usr/bin/env python3
"""Export Breeze TTS 2's codec decoder (Qwen3-TTS-Tokenizer-12Hz, audio_tokenizer/) for plowc.

Writes OUT_DIR/model.safetensors (fp32, torch layouts, folds in fp64) and OUT_DIR/config.json in
the fish_dac export's naming (crates/devgen/src/fish_codec.rs), model_type qwen3_tts_12hz:
  vq.T{k}            [2048][512]   (embedding_sum / clamp(cluster_usage, 1e-5)) @ output_proj^T of
                                   rvq_first (k = 0) or rvq_rest (k >= 1): the decoder input is
                                   sum_k vq.T{k}[code_k]
  pre.{w,b}          causal conv k3 512->1024
  post.in.{w,b} / post.out.{w,b}   the transformer's input (1024->512) / output (512->1024) Linear
  post.layers.N.*    window-72 causal transformer (q/k/v 512->1024, o 1024->512, half-split RoPE);
                     wo_s / w2_s = LayerScale-folded o_proj / down_proj
  up{i}.convt.{w,b}  CausalTransConv1d(k=2, s=2); up{i}.cnx.* ConvNeXt (gamma folded into 2.w_s / 2.b_s)
  dec.in.{w,b}       causal conv k7 1024->1536
  dec.blk{i}.snake / .up.{w,b} / .ru{j}.{a1, c1.w, c1.b, a2, c2.w, c2.b}; dec.out.snake / dec.out.{w,b}
                     every snake is SnakeBeta as [exp(alpha) | 1 / (exp(beta) + 1e-9)]; then clamp

  HF_HOME=... python scripts/tts/breeze_codec_export.py OUT_DIR REVISION
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
    sd = {k[len("decoder."):]: v for k, v in load_file(f"{src}/model.safetensors").items() if k.startswith("decoder.")}
    dc = json.load(open(f"{src}/config.json"))["decoder_config"]
    T = {}

    def put(name, t):
        T[name] = t.detach().double().float().contiguous()

    def sb(p):
        return torch.cat([sd[f"{p}.alpha"].double().exp(), 1.0 / (sd[f"{p}.beta"].double().exp() + 1e-9)])

    n_q = dc["num_quantizers"]
    for k in range(n_q):
        g, layer = ("rvq_first", 0) if k == 0 else ("rvq_rest", k - 1)
        cb = f"quantizer.{g}.vq.layers.{layer}._codebook"
        emb = sd[f"{cb}.embedding_sum"].double() / sd[f"{cb}.cluster_usage"].double().clamp(min=1e-5)[:, None]
        put(f"vq.T{k}", emb @ sd[f"quantizer.{g}.output_proj.weight"].double()[:, :, 0].t())

    put("pre.w", sd["pre_conv.conv.weight"])
    put("pre.b", sd["pre_conv.conv.bias"])
    pt = "pre_transformer"
    put("post.in.w", sd[f"{pt}.input_proj.weight"])
    put("post.in.b", sd[f"{pt}.input_proj.bias"])
    put("post.out.w", sd[f"{pt}.output_proj.weight"])
    put("post.out.b", sd[f"{pt}.output_proj.bias"])
    put("post.norm", sd[f"{pt}.norm.weight"])
    for n in range(dc["num_hidden_layers"]):
        p, o = f"{pt}.layers.{n}", f"post.layers.{n}"
        a = f"{p}.self_attn"
        put(f"{o}.wqkv", torch.cat([sd[f"{a}.{s}_proj.weight"] for s in "qkv"]))
        put(f"{o}.attn_norm", sd[f"{p}.input_layernorm.weight"])
        put(f"{o}.ffn_norm", sd[f"{p}.post_attention_layernorm.weight"])
        put(f"{o}.w1", sd[f"{p}.mlp.gate_proj.weight"])
        put(f"{o}.w3", sd[f"{p}.mlp.up_proj.weight"])
        put(f"{o}.wo_s", sd[f"{p}.self_attn_layer_scale.scale"].double()[:, None] * sd[f"{a}.o_proj.weight"].double())
        put(f"{o}.w2_s", sd[f"{p}.mlp_layer_scale.scale"].double()[:, None] * sd[f"{p}.mlp.down_proj.weight"].double())

    ups = len(dc["upsampling_ratios"])
    for u in range(ups):
        p, o = f"upsample.{u}", f"up{u}"
        put(f"{o}.convt.w", sd[f"{p}.0.conv.weight"])
        put(f"{o}.convt.b", sd[f"{p}.0.conv.bias"])
        put(f"{o}.cnx.dw.w", sd[f"{p}.1.dwconv.conv.weight"])
        put(f"{o}.cnx.dw.b", sd[f"{p}.1.dwconv.conv.bias"])
        put(f"{o}.cnx.ln.w", sd[f"{p}.1.norm.weight"])
        put(f"{o}.cnx.ln.b", sd[f"{p}.1.norm.bias"])
        put(f"{o}.cnx.1.w", sd[f"{p}.1.pwconv1.weight"])
        put(f"{o}.cnx.1.b", sd[f"{p}.1.pwconv1.bias"])
        gamma = sd[f"{p}.1.gamma"].double()
        put(f"{o}.cnx.2.w_s", gamma[:, None] * sd[f"{p}.1.pwconv2.weight"].double())
        put(f"{o}.cnx.2.b_s", gamma * sd[f"{p}.1.pwconv2.bias"].double())

    put("dec.in.w", sd["decoder.0.conv.weight"])
    put("dec.in.b", sd["decoder.0.conv.bias"])
    blocks = []
    rates = dc["upsample_rates"]
    for i, r in enumerate(rates):
        p, o = f"decoder.{1 + i}.block", f"dec.blk{i}"
        w = sd[f"{p}.1.conv.weight"]
        cin, cout, k = w.shape
        assert k == 2 * r, (k, r)
        blocks.append(dict(cin=cin, cout=cout, kernel=k, stride=r, dilations=[1, 3, 9]))
        put(f"{o}.snake", sb(f"{p}.0"))
        put(f"{o}.up.w", w)
        put(f"{o}.up.b", sd[f"{p}.1.conv.bias"])
        for j in range(3):
            q = f"{p}.{2 + j}"
            put(f"{o}.ru{j}.a1", sb(f"{q}.act1"))
            put(f"{o}.ru{j}.c1.w", sd[f"{q}.conv1.conv.weight"])
            put(f"{o}.ru{j}.c1.b", sd[f"{q}.conv1.conv.bias"])
            put(f"{o}.ru{j}.a2", sb(f"{q}.act2"))
            put(f"{o}.ru{j}.c2.w", sd[f"{q}.conv2.conv.weight"])
            put(f"{o}.ru{j}.c2.b", sd[f"{q}.conv2.conv.bias"])
    n = len(rates)
    put("dec.out.snake", sb(f"decoder.{1 + n}"))
    put("dec.out.w", sd[f"decoder.{2 + n}.conv.weight"])
    put("dec.out.b", sd[f"decoder.{2 + n}.conv.bias"])

    save_file(T, os.path.join(out, "model.safetensors"))
    hidden, heads, hd = dc["hidden_size"], dc["num_attention_heads"], dc["head_dim"]
    assert dc["num_key_value_heads"] == heads
    config = dict(
        model_type="qwen3_tts_12hz", sampling_rate=24000,
        frame_samples=2 ** ups * int(torch.tensor(rates).prod()), frame_codes=n_q,
        codebook_sizes=[dc["codebook_size"]] * n_q, vq_dim=dc["codebook_dim"], latent_dim=dc["latent_dim"],
        pre_kernel=3,
        transformer=dict(layers=dc["num_hidden_layers"], dim=hidden, heads=heads, head_dim=hd,
                         intermediate=dc["intermediate_size"], eps=dc["rms_norm_eps"],
                         window=dc["sliding_window"], rope_theta=dc["rope_theta"]),
        upsample=[dict(stride=s, kernel=s, convnext_kernel=7, ln_eps=1e-6) for s in dc["upsampling_ratios"]],
        decoder=dict(in_kernel=7, in_channels=dc["latent_dim"], channels=dc["decoder_dim"], blocks=blocks,
                     out_kernel=7),
    )
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    print(f"wrote {out}: {len(T)} tensors")


if __name__ == "__main__":
    main()
