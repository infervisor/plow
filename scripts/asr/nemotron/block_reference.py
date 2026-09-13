#!/usr/bin/env python3
"""Independent NumPy reference for one Nemotron FastConformer GGUF block."""

import argparse
from pathlib import Path

import gguf
import numpy as np


def q8(tensor):
    data = np.asarray(tensor.data, dtype=np.uint8)
    rows = data.reshape(data.shape[0], -1, 34)
    scales = rows[:, :, :2].copy().view("<f2").reshape(rows.shape[0], -1).astype(np.float32)
    quant = rows[:, :, 2:].view(np.int8).astype(np.float32)
    return (quant * scales[:, :, None]).reshape(data.shape[0], -1)


def linear(x, tensor):
    return np.einsum(
        "mk,nk->mn", x.astype(np.float64), q8(tensor).astype(np.float64), optimize=False
    ).astype(np.float32)


def layer_norm(x, weight, bias):
    mean = x.astype(np.float64).mean(axis=1, keepdims=True)
    variance = ((x.astype(np.float64) - mean) ** 2).mean(axis=1, keepdims=True)
    normalized = ((x.astype(np.float64) - mean) / np.sqrt(variance + 1e-5)).astype(np.float32)
    return (normalized * weight.data + bias.data).astype(np.float32)


def sigmoid(x):
    positive = x >= 0
    out = np.empty_like(x)
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    out[~positive] = exponential / (1 + exponential)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("layer", type=int)
    parser.add_argument("frames", type=int)
    parser.add_argument("rust_output", type=Path)
    args = parser.parse_args()

    reader = gguf.GGUFReader(args.model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    prefix = f"encoder.layers.{args.layer}"
    width = 1024
    heads = 8
    head_width = width // heads
    residual = np.array(
        [((index * 17) % 251 - 125) / 127 for index in range(args.frames * width)],
        dtype=np.float32,
    ).reshape(args.frames, width)

    def norm(name, x):
        return layer_norm(
            x,
            tensors[f"{prefix}.{name}.weight"],
            tensors[f"{prefix}.{name}.bias"],
        )

    def feed_forward(number, x):
        hidden = linear(norm(f"norm_feed_forward{number}", x), tensors[f"{prefix}.feed_forward{number}.linear1.weight"])
        hidden *= sigmoid(hidden)
        return x + np.float32(0.5) * linear(hidden, tensors[f"{prefix}.feed_forward{number}.linear2.weight"])

    residual = feed_forward(1, residual)
    normalized = norm("norm_self_att", residual)
    attention_prefix = f"{prefix}.self_attn"
    query = linear(normalized, tensors[f"{attention_prefix}.linear_q.weight"]).reshape(args.frames, heads, head_width)
    key = linear(normalized, tensors[f"{attention_prefix}.linear_k.weight"]).reshape(args.frames, heads, head_width)
    value = linear(normalized, tensors[f"{attention_prefix}.linear_v.weight"]).reshape(args.frames, heads, head_width)
    relative_count = args.frames * 2 - 1
    position_input = tensors["encoder.pos_enc.pe"].data[5000 - args.frames:5000 + args.frames - 1]
    position = linear(position_input, tensors[f"{attention_prefix}.linear_pos.weight"]).reshape(relative_count, heads, head_width)
    bias_u = tensors[f"{attention_prefix}.pos_bias_u"].data
    bias_v = tensors[f"{attention_prefix}.pos_bias_v"].data
    scores = np.full((heads, args.frames, args.frames), -np.inf, dtype=np.float32)
    for head in range(heads):
        for query_frame in range(args.frames):
            query_chunk = query_frame // 4
            for key_frame in range(args.frames):
                key_chunk = key_frame // 4
                if key_chunk > query_chunk or query_chunk - key_chunk > 14:
                    continue
                content = np.dot(
                    key[key_frame, head].astype(np.float64),
                    (query[query_frame, head] + bias_u[head]).astype(np.float64),
                )
                relative_index = args.frames - 1 + key_frame - query_frame
                relative = np.dot(
                    position[relative_index, head].astype(np.float64),
                    (query[query_frame, head] + bias_v[head]).astype(np.float64),
                )
                scores[head, query_frame, key_frame] = np.float32(
                    (content + relative) / np.sqrt(head_width)
                )
    scores -= np.max(scores, axis=2, keepdims=True)
    probability = np.exp(scores)
    probability /= np.sum(probability, axis=2, keepdims=True)
    context = np.einsum("hqk,khd->qhd", probability, value, dtype=np.float64).astype(np.float32)
    residual += linear(context.reshape(args.frames, width), tensors[f"{attention_prefix}.linear_out.weight"])

    normalized = norm("norm_conv", residual)
    pointwise = linear(normalized, tensors[f"{prefix}.conv.pointwise_conv1.weight"])
    gated = pointwise[:, :width] * sigmoid(pointwise[:, width:])
    kernel = tensors[f"{prefix}.conv.depthwise_conv.weight"].data[:, 0, :].astype(np.float32)
    depthwise = np.zeros_like(gated)
    for frame in range(args.frames):
        for tap in range(kernel.shape[1]):
            source = frame + tap + 1 - kernel.shape[1]
            if source >= 0:
                depthwise[frame] += gated[source] * kernel[:, tap]
    activated = layer_norm(
        depthwise,
        tensors[f"{prefix}.conv.batch_norm.weight"],
        tensors[f"{prefix}.conv.batch_norm.bias"],
    )
    activated *= sigmoid(activated)
    residual += linear(activated, tensors[f"{prefix}.conv.pointwise_conv2.weight"])
    residual = feed_forward(2, residual)
    expected = norm("norm_out", residual)

    actual = np.fromfile(args.rust_output, dtype="<f4").reshape(args.frames, width)
    difference = np.abs(actual - expected)
    print(f"max_abs={difference.max():.9g} mean_abs={difference.mean():.9g}")
    if not np.allclose(actual, expected, rtol=2e-5, atol=2e-4):
        raise SystemExit("Rust block differs from independent NumPy reference")


if __name__ == "__main__":
    main()
