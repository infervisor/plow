#!/usr/bin/env python3
"""Breeze TTS 2 reference-audio encoder (Qwen3-TTS-Tokenizer-12Hz encoder = HF Mimi), fp32 CPU.

Runs the official `Qwen3TTSTokenizer.encode` on each wav and saves, per wav, in OUT/<stem>/:
  wav24k.npy  [n] f32       mono 24 kHz input the encoder saw (after the wrapper's resample)
  seanet.npy  [T25][512]    SEANet output (25 Hz rows)
  xf.npy      [T25][512]    encoder_transformer output
  down.npy    [T12][512]    downsample output (12.5 Hz rows)
  codes.npy   [F][16] i32   codes the model returns (F = ceil(n / 1920))

  venv-breeze/bin/python scripts/tts/mimi_encoder_ref.py OUT REVISION a.wav [b.wav ...] \
      [--golden newtts/breeze/golden/clone/input_values.npy]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

REPO = "models--BreezeBlue--Breeze-TTS-2"


def score_diagnosis(values, exported, semantic, quantizers, reference, actual):
    variants = {}
    for mode in ("exported_norm", "f32_norm", "cdist_exported", "augmented_squared", "augmented_sqrt"):
        predicted = []
        for group, count in (("sem", semantic), ("ac", quantizers - semantic)):
            x = torch.nn.functional.conv1d(torch.from_numpy(values).T.contiguous()[None],
                                          exported[group + ".proj"][:, :, None])[0].T.contiguous()
            for k in range(count):
                e = exported[f"{group}.E{k}"]
                if mode == "exported_norm":
                    index = (x @ e.T + exported[f"{group}.nb{k}"]).argmax(dim=1)
                elif mode == "f32_norm":
                    index = (x @ e.T - 0.5 * e.square().sum(dim=1)).argmax(dim=1)
                elif mode == "cdist_exported":
                    index = torch.cdist(x[None], e[None], p=2)[0].argmin(dim=1)
                else:
                    a = torch.cat((-2 * x, x.square().sum(dim=1, keepdim=True), torch.ones(x.shape[0], 1)), dim=1)
                    b = torch.cat((e, torch.ones(e.shape[0], 1), e.square().sum(dim=1, keepdim=True)), dim=1)
                    distance = (a @ b.T).clamp_min(0)
                    if mode == "augmented_sqrt":
                        distance = distance.sqrt()
                    index = distance.argmin(dim=1)
                predicted.append(index)
                x = x - e[index]
        predicted = torch.stack(predicted, dim=1).numpy().astype(np.int32)
        variants[mode] = {"vs_reference": int(np.count_nonzero(predicted != reference)),
                          "vs_native": int(np.count_nonzero(predicted != actual)),
                          "difference_positions": np.argwhere(predicted != reference).tolist()}
    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("revision")
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--golden", help="codes [1][F][16] the official clone run encoded from wavs[0]")
    ap.add_argument("--encoder", type=Path, help="wrapped tokenizer or unwrapped Mimi snapshot; inputs must already match its sample rate")
    ap.add_argument("--native-result", type=Path, help="replay captured native downsample outputs through the official CPU quantizer")
    ap.add_argument("--exported-encoder", type=Path, help="also diagnose exported linear-score quantization; requires --native-result")
    args = ap.parse_args()
    if args.native_result and not args.encoder:
        ap.error("--native-result requires --encoder")
    if args.exported_encoder and not args.native_result:
        ap.error("--exported-encoder requires --native-result")
    native = json.loads(args.native_result.read_text()) if args.native_result else None
    native_bytes = Path(native["output_data"]).read_bytes() if native else None
    import soundfile as sf

    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    if args.encoder:
        from safetensors.torch import load_file
        from transformers import MimiConfig, MimiModel

        torch.set_num_threads(4)
        top = json.loads((args.encoder / "config.json").read_text())
        wrapped = "encoder_config" in top
        cfg = MimiConfig(**(top["encoder_config"] if wrapped else top))
        cfg._attn_implementation = "eager"
        enc = MimiModel(cfg)
        weights = load_file(args.encoder / "model.safetensors")
        if wrapped:
            enc.upsample = enc.decoder_transformer = enc.decoder = None
            weights = {k[len("encoder."):]: v for k, v in weights.items() if k.startswith("encoder.")}
        enc.load_state_dict(weights, strict=True)
        enc.eval()
        exported = load_file(args.exported_encoder / "model.safetensors") if args.exported_encoder else None
        quantizers = top["encoder_valid_num_quantizers"] if wrapped else cfg.num_quantizers
        frame_samples = top["encode_downsample_rate"] if wrapped else round(cfg.sampling_rate / cfg.frame_rate)
        manifest = {"scope": "official Mimi encoder CPU correctness reference; no benchmark",
                    "sample_rate": cfg.sampling_rate, "frame_samples": frame_samples, "frame_codes": quantizers,
                    "source": str(args.encoder), "torch": torch.__version__, "attention": "eager",
                    "source_sha256": {name: hashlib.file_digest((args.encoder / name).open("rb"), "sha256").hexdigest()
                                      for name in ("config.json", "model.safetensors")}, "cases": []}
    else:
        from qwen_tts import Qwen3TTSTokenizer

        tok = Qwen3TTSTokenizer.from_pretrained(
            os.path.join(hub, REPO, "snapshots", args.revision, "audio_tokenizer"), device_map="cpu")
        enc = tok.model.encoder
    seen = {}

    def keep(name):
        def hook(_m, _i, o):
            o = o[0] if isinstance(o, tuple) else o
            o = getattr(o, "last_hidden_state", o)
            seen[name] = o.detach().float().numpy()
        return hook

    enc.encoder.register_forward_hook(keep("seanet"))
    enc.encoder_transformer.register_forward_hook(keep("xf"))
    enc.downsample.register_forward_hook(keep("down"))
    for i, path in enumerate(args.wavs):
        wav, sr = sf.read(path, always_2d=True, dtype="float32")
        wav = np.mean(wav, axis=1)
        if args.encoder:
            if sr != cfg.sampling_rate:
                raise ValueError(f"input sample rate {sr} != encoder {cfg.sampling_rate}")
            wav24 = wav
            with torch.inference_mode():
                result = enc.encode(torch.from_numpy(wav24)[None, None], num_quantizers=quantizers,
                                    use_streaming=False, return_dict=True)
            codes = result.audio_codes[0].T.numpy().astype(np.int32)
            assert codes.shape == ((len(wav24) + frame_samples - 1) // frame_samples, quantizers)
        else:
            wav24 = tok._normalize_audio_inputs(wav, sr=sr)[0]
            codes = tok.encode(wav, sr=sr)["audio_codes"][0].numpy().astype(np.int32)
        d = os.path.join(args.out, os.path.splitext(os.path.basename(path))[0])
        os.makedirs(d, exist_ok=True)
        np.save(f"{d}/wav24k.npy", wav24)
        # seanet / down are [1][C][T]; xf is [1][T][C].
        np.save(f"{d}/seanet.npy", seen["seanet"][0].T)
        np.save(f"{d}/xf.npy", seen["xf"][0])
        np.save(f"{d}/down.npy", seen["down"][0].T)
        np.save(f"{d}/codes.npy", codes)
        if args.encoder:
            pcm_path, codes_path = Path(d) / "pcm.f32", Path(d) / "codes.u32"
            wav24.astype("<f4").tofile(pcm_path)
            codes.astype("<u4").tofile(codes_path)
            cell = {"case": Path(path).stem, "samples": len(wav24), "frames": codes.shape[0],
                    "pcm": str(pcm_path.resolve()), "codes": str(codes_path.resolve())}
            for name in ("seanet", "xf", "down"):
                values = seen[name][0] if name == "xf" else seen[name][0].T
                stage_path = Path(d) / (name + ".f32")
                values.astype("<f4").tofile(stage_path)
                cell[name] = {"path": str(stage_path.resolve()), "rows": values.shape[0], "width": values.shape[1]}
            manifest["cases"].append(cell)
            if native:
                for check in native["checks"]:
                    if check["case"] != Path(path).stem:
                        continue
                    stage = next(s for s in check["stages"] if s["stage"] == "down")
                    rows, width = stage["rows"], stage["width"]
                    values = np.frombuffer(native_bytes, dtype="<f4", count=rows * width,
                                           offset=stage["output_offset_bytes"]).copy().reshape(rows, width)
                    with torch.inference_mode():
                        replay = enc.quantizer.encode(torch.from_numpy(values).T.contiguous()[None],
                                                      num_quantizers=quantizers)[:, 0].T.numpy().astype(np.int32)
                    actual = np.frombuffer(native_bytes, dtype="<u4", count=rows * quantizers,
                                           offset=check["codes_offset_bytes"]).reshape(rows, quantizers)
                    assert replay.shape == codes.shape == actual.shape
                    item = {"case": check["case"], "capacity": check["capacity"],
                            "cpu_replay_vs_reference": int(np.count_nonzero(replay != codes)),
                            "cpu_replay_vs_native": int(np.count_nonzero(replay != actual)),
                            "native_difference_positions": np.argwhere(actual != codes).tolist()}
                    if exported:
                        variants = score_diagnosis(values, exported, cfg.num_semantic_quantizers, quantizers, codes, actual)
                        linear = variants.pop("exported_norm")
                        item["diagnostic_linear_vs_reference"] = linear["vs_reference"]
                        item["diagnostic_linear_vs_native"] = linear["vs_native"]
                        item["diagnostic_linear_difference_positions"] = linear["difference_positions"]
                        item["diagnostic_score_variants"] = variants
                    manifest.setdefault("native_replay_checks", []).append(item)
                    print(json.dumps(item), flush=True)
        msg = f"{path}: {len(wav24)} samples -> codes {codes.shape}"
        if i == 0 and args.golden:
            g = np.load(args.golden)[0].astype(np.int32)
            same = (g == codes).all(axis=1) if g.shape == codes.shape else None
            msg += f"; golden {g.shape}: " + (
                f"{int(same.sum())}/{len(same)} frames identical" if same is not None else "shape differs")
        print(msg, flush=True)
    if args.encoder:
        (Path(args.out) / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
