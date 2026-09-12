#!/usr/bin/env python3
"""Trace decoder activations on saved reference embeddings without audio execution."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    wrapper = Qwen3ASRModel.from_pretrained(args.checkpoint, dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager", max_new_tokens=1024, local_files_only=True)
    thinker = wrapper.model.thinker
    hidden = thinker.config.text_config.hidden_size
    embeddings = np.fromfile(args.reference / "spliced.f32", dtype="<f4").reshape(1, -1, hidden)
    value = torch.from_numpy(embeddings).to(torch.bfloat16)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {}

    def save(name, tensor):
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if not isinstance(tensor, torch.Tensor):
            return
        tensor.detach().float().numpy().astype("<f4").tofile(args.out / f"{name}.f32")
        metadata[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}

    hooks = []
    for name, module in thinker.model.named_modules():
        if name == "norm" or name.startswith("layers.0.") or name.startswith("layers.") and name.count(".") == 1:
            def capture(module, inputs, output, name=name):
                if inputs:
                    save(name + ".input", inputs[0])
                save(name + ".output", output)
            hooks.append(module.register_forward_hook(capture))
    with torch.inference_mode():
        output = thinker(inputs_embeds=value, attention_mask=torch.ones(value.shape[:2], dtype=torch.long), use_cache=False)
        save("logits", output.logits[:, -1, :])
    for hook in hooks:
        hook.remove()
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"tensors": len(metadata), "out": str(args.out)}))


if __name__ == "__main__":
    main()
