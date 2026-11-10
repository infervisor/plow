#!/usr/bin/env python3
"""Export the speechbrain ECAPA-TDNN speaker embedding (spkrec-ecapa-voxceleb) for plowrt's host port.

    python scripts/tts/ecapa_export.py OUT_DIR [--ckpt embedding_model.ckpt]

Writes OUT_DIR/model.safetensors (all F32) and OUT_DIR/config.json, nothing else. Consumer:
crates/plowrt/src/tts/ecapa.rs. Layout (C = 1024, M = 3072, A = 128, E = 192):

    python scripts/tts/ecapa_export.py OUT_DIR --qwen-checkpoint HF_SNAPSHOT

The Qwen variant exports an `ecapa_tdnn` checkpoint with explicit frontend, precision and
normalization settings. It requires a separate packet lowering; the Maya host port cannot load it.
Convolution results round before bias addition, matching the CUDA reference's BF16 boundary.

  Conv weights are [Cout, K, Cin] (torch's [Cout, Cin, K] with the tap axis moved before Cin); 1x1 convs are
  [Cout, 1, Cin]. Every TDNN block is conv -> ReLU -> eval BatchNorm, so BN cannot fold into the conv; it is
  stored as `<p>.bn.scale` = gamma / sqrt(var + 1e-5) and `<p>.bn.shift` = beta - mean * scale.

  blocks.0.conv.{weight [C,5,80], bias}, blocks.0.bn.{scale, shift}
  blocks.{1,2,3}.tdnn1.{conv.*, bn.*}               1x1 C -> C
  blocks.{1,2,3}.res2net.{0..6}.{conv.*, bn.*}       [C/8, 3, C/8], dilation = block index + 1
  blocks.{1,2,3}.tdnn2.{conv.*, bn.*}               1x1 C -> C
  blocks.{1,2,3}.se.conv1.{weight [A,1,C], bias}, blocks.{1,2,3}.se.conv2.{weight [C,1,A], bias}
  mfa.{conv.*, bn.*}                                1x1 3C -> M
  asp.tdnn.conv.weight_{x,mean,std} [A,1,M]         the 3M-wide input [x, mean, std] split by column block
  asp.tdnn.conv.bias, asp.tdnn.bn.{scale, shift}
  asp.conv.{weight [M,1,A], bias}
  fc.{weight [E,1,2M], bias}                        asp_bn (eval, no activation before fc) folded in
"""
import argparse
import glob
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BN_EPS = 1e-5
CHANNELS = [1024, 1024, 1024, 1024, 3072]
KERNELS = [5, 3, 3, 3, 1]
DILATIONS = [1, 2, 3, 4, 1]


def default_ckpt():
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    hits = sorted(glob.glob(os.path.join(
        hub, "models--speechbrain--spkrec-ecapa-voxceleb/snapshots/*/embedding_model.ckpt"
    ))) or sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/*/embedding_model.ckpt"
    )))
    return hits[0] if hits else None


def bn_affine(sd, p):
    scale = sd[f"{p}.weight"] / torch.sqrt(sd[f"{p}.running_var"] + BN_EPS)
    return scale, sd[f"{p}.bias"] - sd[f"{p}.running_mean"] * scale


def conv_w(w):
    return w.permute(0, 2, 1)


def qwen_config(config):
    if config.get("tts_model_type") != "base":
        raise ValueError("speaker export requires a Base checkpoint")
    source = config["speaker_encoder_config"]
    channels = source.get("enc_channels", [512, 512, 512, 512, 1536])
    kernels = source.get("enc_kernel_sizes", [5, 3, 3, 3, 1])
    dilations = source.get("enc_dilations", [1, 2, 3, 4, 1])
    scale = source.get("enc_res2net_scale", 8)
    se = source.get("enc_se_channels", 128)
    attention = source.get("enc_attention_channels", 128)
    dim = source.get("enc_dim", 1024)
    positive = channels + kernels + dilations + [scale, se, attention, dim]
    if (len(channels) < 3 or len(channels) != len(kernels) or len(channels) != len(dilations)
            or any(type(x) is not int or x <= 0 for x in positive)
            or scale < 2 or any(k % 2 == 0 for k in kernels)
            or any(c != channels[0] or c % scale for c in channels[:-1])
            or sum(channels[1:-1]) != channels[-1]
            or source.get("mel_dim", 128) != 128 or source.get("sample_rate", 24000) != 24000
            or dim != config["talker_config"]["hidden_size"]):
        raise ValueError("unsupported speaker encoder geometry")
    return dict(
        model_type="ecapa_tdnn", version=1, channels=channels, kernel_sizes=kernels,
        dilations=dilations, res2net_scale=scale, se_channels=se, attention_channels=attention,
        embedding_dim=dim, batch_norm=False, l2_normalize=False, statistics_eps=1e-12,
        compute_dtype="bfloat16", conv_bias_round="before_add", weight_layout="out_tap_in",
        frontend=dict(compute_dtype="float32", sample_rate=24000, n_mels=128,
                      n_fft=1024, hop=256, win_length=1024,
                      pad=384, pad_mode="reflect", center=False, magnitude_eps=1e-9,
                      mel_norm="slaney", mel_htk=False, fmin=0, fmax=12000,
                      log_floor=1e-5, sentence_normalize=False))


def qwen_tensors(weights, config):
    out, used = {}, set()

    def conv(src, dst, cin, cout, kernel):
        for suffix, shape in (("weight", (cout, cin, kernel)), ("bias", (cout,))):
            key = f"{src}.{suffix}"
            value = weights[key]
            if tuple(value.shape) != shape or value.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError(f"{key}: expected floating tensor {shape}")
            value = value.to(torch.bfloat16).float()
            if not torch.isfinite(value).all():
                raise ValueError(f"{key}: nonfinite BF16 speaker weight")
            used.add(key)
            out[f"{dst}.{suffix}"] = (conv_w(value) if suffix == "weight" else value).contiguous()

    channels, kernels = config["channels"], config["kernel_sizes"]
    scale, se, attention = (config[k] for k in ("res2net_scale", "se_channels", "attention_channels"))
    conv("blocks.0.conv", "blocks.0.conv", 128, channels[0], kernels[0])
    for b in range(1, len(channels) - 1):
        c = channels[b]
        conv(f"blocks.{b}.tdnn1.conv", f"blocks.{b}.tdnn1.conv", c, c, 1)
        for i in range(scale - 1):
            conv(f"blocks.{b}.res2net_block.blocks.{i}.conv", f"blocks.{b}.res2net.{i}.conv",
                 c // scale, c // scale, kernels[b])
        conv(f"blocks.{b}.tdnn2.conv", f"blocks.{b}.tdnn2.conv", c, c, 1)
        conv(f"blocks.{b}.se_block.conv1", f"blocks.{b}.se.conv1", c, se, 1)
        conv(f"blocks.{b}.se_block.conv2", f"blocks.{b}.se.conv2", se, c, 1)
    m = channels[-1]
    conv("mfa.conv", "mfa.conv", m, m, kernels[-1])
    conv("asp.tdnn.conv", "asp.tdnn.conv", 3 * m, attention, 1)
    conv("asp.conv", "asp.conv", attention, m, 1)
    conv("fc", "fc", 2 * m, config["embedding_dim"], 1)
    if set(weights) != used:
        raise ValueError(f"unexpected speaker tensors: {sorted(set(weights) - used)}")
    return out


def export_qwen(checkpoint, out_dir):
    import librosa

    src = Path(checkpoint)
    cfg = qwen_config(json.loads((src / "config.json").read_text()))
    prefix = "speaker_encoder."
    with safe_open(src / "model.safetensors", framework="pt", device="cpu") as reader:
        weights = {key[len(prefix):]: reader.get_tensor(key)
                   for key in reader.keys() if key.startswith(prefix)}
    out = qwen_tensors(weights, cfg)
    out["frontend.mel"] = torch.from_numpy(librosa.filters.mel(
        sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000, htk=False, norm="slaney"))
    out["frontend.window"] = torch.hann_window(1024)
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    save_file(out, str(od / "model.safetensors"))
    (od / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"wrote {len(out)} speaker/frontend tensors -> {od}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--ckpt")
    source.add_argument("--qwen-checkpoint")
    a = ap.parse_args()
    if a.qwen_checkpoint:
        export_qwen(a.qwen_checkpoint, a.out_dir)
        return
    a.ckpt = a.ckpt or default_ckpt()
    sd = {k: v.double() for k, v in torch.load(a.ckpt, map_location="cpu", weights_only=True).items()
          if not k.endswith("num_batches_tracked")}
    out = {}

    def tdnn(src, dst):
        out[f"{dst}.conv.weight"] = conv_w(sd[f"{src}.conv.conv.weight"])
        out[f"{dst}.conv.bias"] = sd[f"{src}.conv.conv.bias"]
        out[f"{dst}.bn.scale"], out[f"{dst}.bn.shift"] = bn_affine(sd, f"{src}.norm.norm")

    def conv(src, dst):
        out[f"{dst}.weight"] = conv_w(sd[f"{src}.conv.weight"])
        out[f"{dst}.bias"] = sd[f"{src}.conv.bias"]

    tdnn("blocks.0", "blocks.0")
    for b in (1, 2, 3):
        tdnn(f"blocks.{b}.tdnn1", f"blocks.{b}.tdnn1")
        for i in range(7):
            tdnn(f"blocks.{b}.res2net_block.blocks.{i}", f"blocks.{b}.res2net.{i}")
        tdnn(f"blocks.{b}.tdnn2", f"blocks.{b}.tdnn2")
        conv(f"blocks.{b}.se_block.conv1", f"blocks.{b}.se.conv1")
        conv(f"blocks.{b}.se_block.conv2", f"blocks.{b}.se.conv2")
    tdnn("mfa", "mfa")
    m = CHANNELS[-1]
    w = conv_w(sd["asp.tdnn.conv.conv.weight"])
    for i, n in enumerate(("x", "mean", "std")):
        out[f"asp.tdnn.conv.weight_{n}"] = w[:, :, i * m:(i + 1) * m]
    out["asp.tdnn.conv.bias"] = sd["asp.tdnn.conv.conv.bias"]
    out["asp.tdnn.bn.scale"], out["asp.tdnn.bn.shift"] = bn_affine(sd, "asp.tdnn.norm.norm")
    conv("asp.conv", "asp.conv")
    s, t = bn_affine(sd, "asp_bn.norm")
    fw = sd["fc.conv.weight"][:, :, 0]
    out["fc.weight"] = (fw * s[None, :])[:, None, :]
    out["fc.bias"] = sd["fc.conv.bias"] + fw @ t

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    save_file({k: v.float().contiguous() for k, v in out.items()}, str(od / "model.safetensors"))
    cfg = {"arch": "speechbrain.ECAPA_TDNN", "source": str(a.ckpt), "sample_rate_in": 24000, "sample_rate": 16000,
           "n_mels": 80, "n_fft": 400, "hop": 160, "channels": CHANNELS, "kernel_sizes": KERNELS,
           "dilations": DILATIONS, "res2net_scale": 8, "se_channels": 128, "attention_channels": 128,
           "lin_neurons": 192, "bn_eps": BN_EPS}
    (od / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"wrote {len(out)} tensors -> {od}")


if __name__ == "__main__":
    main()
