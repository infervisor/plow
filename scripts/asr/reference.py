"""Export reference fixtures for the native Qwen3-ASR bring-up."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel
from transformers import AutoProcessor
from safetensors import safe_open
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=["frontend", "encoder", "model"], default="frontend")
    parser.add_argument("--packed-windows", action="store_true",
                        help="Apply the encoder's packed-window mask omitted by upstream eager attention")
    parser.add_argument("--trace", action="store_true", help="Export encoder component outputs")
    parser.add_argument("--decoder-attention", choices=["eager", "sdpa"], default="eager")
    args = parser.parse_args()
    torch.set_num_threads(8)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audio, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    assert rate == 16000
    audio = audio.mean(axis=1)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    text = processor.apply_chat_template([
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": args.audio}]},
    ], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], audio=[audio], return_tensors="pt", padding=True)
    features = inputs["input_features"].numpy()
    features.astype("<f4").tofile(out / "mel.f32")
    (out / "input.json").write_text(json.dumps({
        "samples": len(audio), "shape": list(features.shape),
        "packed_windows": args.packed_windows,
        "decoder_attention": args.decoder_attention,
        "feature_lengths": inputs["feature_attention_mask"].sum(-1).tolist(),
        "prompt_ids": inputs["input_ids"].tolist(), "prompt": text,
        "packages": {p: importlib.metadata.version(p) for p in ["qwen-asr", "transformers", "torch"]},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(args.checkpoint).glob("*.json")},
    }, indent=2))
    if args.stage == "frontend":
        return
    if args.stage == "encoder":
        config = Qwen3ASRConfig.from_pretrained(args.checkpoint, local_files_only=True)
        config.thinker_config.audio_config._attn_implementation = "eager"
        encoder = Qwen3ASRAudioEncoder(config.thinker_config.audio_config).to(torch.bfloat16).eval()
        weights = {}
        for shard in Path(args.checkpoint).glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                for name in tensors.keys():
                    if name.startswith("thinker.audio_tower."):
                        weights[name.removeprefix("thinker.audio_tower.")] = tensors.get_tensor(name)
        encoder.load_state_dict(weights, strict=True)
        del weights
        if args.packed_windows:
            packed_windows(encoder)
        if args.trace:
            trace_encoder(encoder, out)
        features = inputs["input_features"].to(torch.bfloat16)
        lengths = inputs["feature_attention_mask"].sum(-1)
        with torch.inference_mode():
            embeddings = encoder(features.permute(0, 2, 1)[inputs["feature_attention_mask"].bool()].T,
                                 feature_lens=lengths).last_hidden_state
            embeddings.float().numpy().astype("<f4").tofile(out / "audio.f32")
        return
    model = Qwen3ASRModel.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager", max_new_tokens=1024, local_files_only=True,
    )
    if args.packed_windows:
        packed_windows(model.model.thinker.audio_tower)
    model.model.thinker.model.config._attn_implementation = args.decoder_attention
    if args.trace:
        trace_encoder(model.model.thinker.audio_tower, out)
    inputs = inputs.to(model.model.device).to(model.model.dtype)
    with torch.inference_mode():
        embeddings = model.model.thinker.get_audio_features(
            inputs["input_features"], inputs["feature_attention_mask"])
        embeddings.float().numpy().astype("<f4").tofile(out / "audio.f32")
        if args.stage == "model":
            thinker = model.model.thinker
            text_embeddings = thinker.get_input_embeddings()(inputs["input_ids"])
            mask = thinker.get_placeholder_mask(inputs["input_ids"], inputs_embeds=text_embeddings)
            spliced = text_embeddings.masked_scatter(mask, embeddings)
            spliced.float().numpy().astype("<f4").tofile(out / "spliced.f32")
            first = thinker(inputs_embeds=spliced, attention_mask=inputs["attention_mask"], use_cache=False)
            first.logits[0, -1].float().numpy().astype("<f4").tofile(out / "logits.f32")
            generated = model.model.generate(**inputs, max_new_tokens=1024)
            ids = generated.sequences[0, inputs["input_ids"].shape[1]:].tolist()
            (out / "tokens.json").write_text(json.dumps({"ids": ids,
                "raw_text": processor.tokenizer.decode(ids, skip_special_tokens=True),
                "head_tied": thinker.lm_head.weight.data_ptr() == thinker.get_input_embeddings().weight.data_ptr()}))
            result = model.transcribe(audio=(audio, rate))
            (out / "transcript.json").write_text(json.dumps([
                {"language": r.language, "text": r.text} for r in result
            ], ensure_ascii=False, indent=2))

def packed_windows(encoder):
    def mask(module, args, kwargs):
        kwargs["attention_mask"] = encoder._prepare_attention_mask(args[0], args[1])
        return args, kwargs
    for layer in encoder.layers:
        layer.register_forward_pre_hook(mask, with_kwargs=True)


def trace_encoder(encoder, out):
    for name, module in encoder.named_modules():
        if name in ["conv2d1", "conv2d2", "conv2d3", "conv_out", "ln_post", "proj1"] or name.startswith("layers.") and name.count(".") == 1:
            def save(module, args, value, name=name):
                if isinstance(value, tuple):
                    value = value[0]
                if name.startswith("conv2d") or name == "proj1":
                    value = torch.nn.functional.gelu(value)
                value.detach().float().numpy().astype("<f4").tofile(out / f"{name}.f32")
            module.register_forward_hook(save)


if __name__ == "__main__":
    main()
