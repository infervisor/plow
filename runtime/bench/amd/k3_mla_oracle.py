#!/usr/bin/env python3
"""Independent FP32 oracle for k3_mla_decode_exact merged latent dumps."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

HEADS, DK, DR, STRIDE, CONTEXT = 12, 512, 64, 32768, 8192
SCALE = 0.07216878364870322


def filled_bf16(shape: tuple[int, ...], seed: int, offset: int = 0) -> torch.Tensor:
    count = int(np.prod(shape))
    index = np.arange(offset, offset + count, dtype=np.uint64)
    x = (index * np.uint64(1664525) + np.uint64(seed * 1013904223)).astype(np.uint32)
    bits = (np.uint32(0x3C00) + ((x >> np.uint32(28)) << np.uint32(3))) | (
        (x >> np.uint32(16)) & np.uint32(0x8000)
    )
    signed = bits.astype(np.uint16).view(np.int16).copy()
    return torch.from_numpy(signed).view(torch.bfloat16).reshape(shape)


def reference(batch: int) -> torch.Tensor:
    qabs = filled_bf16((batch, HEADS, DK), 1).float()
    qrope = filled_bf16((batch, HEADS, DR), 2).float()
    output = torch.empty((batch, HEADS, DK), dtype=torch.float32)
    for b in range(batch):
        ckv = filled_bf16((CONTEXT, DK), 3, b * STRIDE * DK).float()
        krope = filled_bf16((CONTEXT, DR), 4, b * STRIDE * DR).float()
        score = (qabs[b] @ ckv.T + qrope[b] @ krope.T) * SCALE
        output[b] = torch.softmax(score, dim=-1) @ ckv
    return output


def load_dump(path: Path, batch: int) -> tuple[torch.Tensor, str]:
    raw = path.read_bytes()
    expected = batch * HEADS * DK * 2
    if len(raw) != expected:
        raise RuntimeError(f"{path}: got {len(raw)} bytes, expected {expected}")
    signed = np.frombuffer(raw, dtype=np.uint16).view(np.int16).copy()
    value = torch.from_numpy(signed).view(torch.bfloat16).float().reshape(batch, HEADS, DK)
    return value, hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--rel-l2", type=float, default=5e-3)
    parser.add_argument("--max-abs", type=float, default=1.6e-5)
    args = parser.parse_args()

    failed = False
    for batch in (1, 8):
        ref = reference(batch)
        ref_hash = hashlib.sha256(ref.numpy().tobytes()).hexdigest()
        denom = torch.linalg.vector_norm(ref).item()
        outputs = {}
        cases = [
            (f"ns{nsplit}", Path(f"{args.prefix}.b{batch}.ns{nsplit}.olat.bf16"))
            for nsplit in (32, 64)
        ]
        cases.extend(
            (f"gf6_ns{nsplit}", Path(f"{args.prefix}.b{batch}.gf6.ns{nsplit}.olat.bf16"))
            for nsplit in (32, 64)
            if Path(f"{args.prefix}.b{batch}.gf6.ns{nsplit}.olat.bf16").exists()
        )
        external = Path(f"{args.prefix}.b{batch}.external_gf6.ns32.olat.bf16")
        if external.exists():
            cases.append(("external_gf6_ns32", external))
        for label, path in cases:
            got, dump_hash = load_dump(path, batch)
            outputs[label] = got
            delta = got - ref
            rel_l2 = torch.linalg.vector_norm(delta).item() / max(denom, 1e-30)
            max_abs = delta.abs().max().item()
            finite = bool(torch.isfinite(got).all())
            passed = finite and rel_l2 <= args.rel_l2 and max_abs <= args.max_abs
            failed |= not passed
            print(
                f"ORACLE,B{batch},case={label},rel_l2={rel_l2:.9g},max_abs={max_abs:.9g},"
                f"finite={str(finite).lower()},pass={str(passed).lower()},"
                f"dump_sha256={dump_hash},ref_f32_sha256={ref_hash}"
            )
        cross = outputs["ns32"] - outputs["ns64"]
        cross_rel = torch.linalg.vector_norm(cross).item() / max(
            torch.linalg.vector_norm(outputs["ns64"]).item(), 1e-30
        )
        print(
            f"CROSS,B{batch},ns32_vs_ns64_rel_l2={cross_rel:.9g},"
            f"max_abs={cross.abs().max().item():.9g},"
            f"unequal={int(torch.count_nonzero(cross).item())}"
        )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
