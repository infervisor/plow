#!/usr/bin/env python3
import argparse
import ctypes
import statistics
from pathlib import Path

import torch


TOKENS = 8192
HIDDEN = 3584
INTER = 384
EXPERTS = 896
TOPK = 16
BLOCK_M = 64
GRID = 512
SITU = 2
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
UNUSED = 0xFFFFFFFF


class Module:
    def __init__(self, path, symbol, threads, shared_bytes):
        self.lib = ctypes.CDLL("libamdhip64.so")
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        self.threads = threads
        self.shared_bytes = shared_bytes
        self.call("hipModuleLoad", ctypes.byref(self.module), str(path).encode())
        self.call("hipModuleGetFunction", ctypes.byref(self.function), self.module, symbol.encode())

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with hipError_t {status}")

    def launch(self, tensors):
        holders = [ctypes.c_void_p(t.data_ptr()) for t in tensors]
        holders += [
            ctypes.c_uint32(INTER), ctypes.c_uint32(HIDDEN), ctypes.c_uint32(EXPERTS),
            ctypes.c_uint32(SITU), ctypes.c_float(SITU_BETA),
            ctypes.c_float(SITU_LINEAR_BETA), ctypes.c_uint32(0), ctypes.c_uint32(0),
        ]
        params = (ctypes.c_void_p * len(holders))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in holders)
        )
        self.call(
            "hipModuleLaunchKernel", self.function,
            ctypes.c_uint(GRID), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_uint(self.threads), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_uint(self.shared_bytes),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), params, ctypes.c_void_p(),
        )


def xorshift(state):
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def production_metadata():
    buckets = [[] for _ in range(EXPERTS)]
    state = 930100
    for partidx in range(TOKENS * TOPK):
        state = xorshift(state)
        expert = state % EXPERTS
        state = xorshift(state)
        if state & 3 == 0:
            state = xorshift(state)
            expert = state % (EXPERTS // 8)
        buckets[expert].append(partidx)

    rowoff, counts, tilep = [], [], [0]
    row_token, row_partidx = [], []
    for bucket in buckets:
        rowoff.append(tilep[-1] * BLOCK_M)
        counts.append(len(bucket))
        tiles = (len(bucket) + BLOCK_M - 1) // BLOCK_M
        tilep.append(tilep[-1] + tiles)
        row_token.extend(partidx // TOPK for partidx in bucket)
        row_partidx.extend(bucket)
        pad = tiles * BLOCK_M - len(bucket)
        row_token.extend([UNUSED] * pad)
        row_partidx.extend([UNUSED] * pad)
    meta = torch.tensor(rowoff + counts + tilep, dtype=torch.int32, device="cuda")
    return (
        meta,
        torch.tensor(row_token, dtype=torch.uint32, device="cuda"),
        torch.tensor(row_partidx, dtype=torch.uint32, device="cuda"),
        len(row_token),
    )


def inputs():
    torch.manual_seed(930100)
    meta, row_token, row_partidx, rows = production_metadata()
    activation = torch.empty((TOKENS, HIDDEN), dtype=torch.bfloat16, device="cuda")
    activation.uniform_(-1.0, 1.0)
    weight = torch.randint(
        0, 256, (EXPERTS, 2, INTER, HIDDEN // 2), dtype=torch.uint8, device="cuda"
    )
    weight_scale = torch.randint(
        124, 131, (EXPERTS, 2, INTER, HIDDEN // 32), dtype=torch.uint8, device="cuda"
    )

    weight_stride = weight.stride(0) * weight.element_size()
    branch_weight_stride = weight.stride(1) * weight.element_size()
    scale_stride = weight_scale.stride(0) * weight_scale.element_size()
    branch_scale_stride = weight_scale.stride(1) * weight_scale.element_size()
    weight_table = [0] * (EXPERTS * 3)
    scale_table = [0] * (EXPERTS * 3)
    for expert in range(EXPERTS):
        for branch in range(2):
            weight_table[expert * 3 + branch] = (
                weight.data_ptr() + expert * weight_stride + branch * branch_weight_stride
            )
            scale_table[expert * 3 + branch] = (
                weight_scale.data_ptr() + expert * scale_stride + branch * branch_scale_stride
            )
    return (
        activation,
        torch.tensor(weight_table, dtype=torch.int64, device="cuda"),
        torch.tensor(scale_table, dtype=torch.int64, device="cuda"),
        meta, row_token, row_partidx, rows, weight, weight_scale,
    )


def launch_args(out, out_scale, data):
    activation, weight_table, scale_table, meta, row_token, row_partidx = data[:6]
    return [out, activation, weight_table, scale_table, meta, row_token, row_partidx, out_scale]


def compare(shipping, candidate, data, rows):
    shipping_out = torch.full((rows, INTER // 2), 0xA5, dtype=torch.uint8, device="cuda")
    candidate_out = shipping_out.clone()
    shipping_scale = torch.full((rows, INTER // 32), 0xA5, dtype=torch.uint8, device="cuda")
    candidate_scale = shipping_scale.clone()
    shipping.launch(launch_args(shipping_out, shipping_scale, data))
    candidate.launch(launch_args(candidate_out, candidate_scale, data))
    torch.cuda.synchronize()
    payload_bad = int((shipping_out != candidate_out).sum().item())
    scale_bad = int((shipping_scale != candidate_scale).sum().item())
    print(f"oracle payload_bytes={shipping_out.numel()} bad={payload_bad} "
          f"scale_bytes={shipping_scale.numel()} bad={scale_bad}")
    if payload_bad or scale_bad:
        raise SystemExit(2)
    return shipping_out, shipping_scale, candidate_out, candidate_scale


def timing(shipping, candidate, data, outputs):
    shipping_out, shipping_scale, candidate_out, candidate_scale = outputs
    flush = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    pairs = {
        "shipping": (shipping, launch_args(shipping_out, shipping_scale, data)),
        "candidate": (candidate, launch_args(candidate_out, candidate_scale, data)),
    }
    for module, args in pairs.values():
        flush.add_(1)
        module.launch(args)
    torch.cuda.synchronize()
    samples = {name: [] for name in pairs}

    def measured(name):
        module, args = pairs[name]
        flush.add_(1)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        module.launch(args)
        end.record()
        end.synchronize()
        samples[name].append(start.elapsed_time(end))

    for sample in range(31):
        order = ("shipping", "candidate") if sample % 2 == 0 else ("candidate", "shipping")
        for name in order:
            measured(name)
    for name, values in samples.items():
        print(f"{name} n=31 median_ms={statistics.median(values):.6f} "
              f"min_ms={min(values):.6f} max_ms={max(values):.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shipping", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--run", action="store_true", help="authorize GPU allocation and launches")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("dry by default; pass --run only after the static review")
    shipping = Module(args.shipping, "plow_moe1_mxfp4_bk256_gfx950", 512, 119808)
    candidate = Module(
        args.candidate, "plow_moe1_mxfp4_bm64_bn128_bk256_xcd8_wgm4_gfx950", 256, 52224
    )
    data = inputs()
    rows = data[6]
    outputs = compare(shipping, candidate, data, rows)
    timing(shipping, candidate, data, outputs)


if __name__ == "__main__":
    main()
