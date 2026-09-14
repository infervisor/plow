#!/usr/bin/env python3
"""Measure MLX decoder numerical drift using the same reference embeddings."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_qwen3_asr import load_model
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("references", type=Path, nargs="+")
    args = parser.parse_args()
    model, config = load_model(args.checkpoint, dtype=mx.bfloat16)
    hidden = config.text_config.hidden_size
    for reference in args.references:
        embeddings = np.fromfile(reference / "spliced.f32", dtype="<f4").reshape(1, -1, hidden)
        length = embeddings.shape[1]
        positions = mx.broadcast_to(mx.arange(length)[None, None, :], (1, 3, length))
        state = model.model(inputs_embeds=mx.array(embeddings, dtype=mx.bfloat16), position_ids=positions)
        logits = model.lm_head(state[:, -1:, :]).astype(mx.float32)
        mx.eval(logits)
        actual = np.array(logits).reshape(-1).astype(np.float64)
        expected = np.fromfile(reference / "logits.f32", dtype="<f4").astype(np.float64)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError("invalid logits")
        delta = actual - expected
        print(json.dumps({"reference": str(reference), "dtype": "bfloat16",
                          "max_abs": float(np.max(np.abs(delta))),
                          "relative_l2": float(np.linalg.norm(delta)/np.linalg.norm(expected)),
                          "reference_argmax": int(expected.argmax()), "mlx_argmax": int(actual.argmax())}), flush=True)


if __name__ == "__main__":
    main()
