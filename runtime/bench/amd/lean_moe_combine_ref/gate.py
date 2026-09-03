#!/usr/bin/env python3
import argparse
import ctypes
import json
from pathlib import Path

import torch


class Module:
    def __init__(self, path, symbols):
        self.lib = ctypes.CDLL("libamdhip64.so")
        self.module = ctypes.c_void_p()
        self._call("hipModuleLoad", ctypes.byref(self.module), str(path).encode())
        self.functions = {}
        for symbol in symbols:
            function = ctypes.c_void_p()
            self._call("hipModuleGetFunction", ctypes.byref(function), self.module, symbol.encode())
            self.functions[symbol] = function

    def _call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with hipError_t {status}")

    def launch(self, symbol, grid, out, residual, shared, part, hidden, topk, tokens):
        values = [
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(0 if residual is None else residual.data_ptr()),
            ctypes.c_void_p(0 if shared is None else shared.data_ptr()),
            ctypes.c_void_p(part.data_ptr()),
            ctypes.c_uint32(hidden), ctypes.c_uint32(topk), ctypes.c_uint32(tokens),
        ]
        params = (ctypes.c_void_p * len(values))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values)
        )
        self._call(
            "hipModuleLaunchKernel", self.functions[symbol],
            ctypes.c_uint(grid), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_uint(256), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_uint(0), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            params, ctypes.c_void_p(),
        )


def median_ms(run, warmups=5, samples=31):
    for _ in range(warmups):
        run()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record(); run(); end.record(); end.synchronize()
        values.append(start.elapsed_time(end))
    return sorted(values)[len(values) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("object", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=3584)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    candidate = manifest["object"]["symbol"]
    control = manifest["control_object"]["symbol"]
    candidate_module = Module(args.object, [candidate])
    control_module = Module(args.object.parent / manifest["control_object"]["file"], [control])

    tokens, hidden = args.tokens, args.hidden
    topk = manifest["contract"]["topk"]
    torch.manual_seed(1701)
    primary = None
    for shape_tokens, shape_hidden in dict.fromkeys(
        ((tokens, hidden), (257, 2816), (128, 4096))
    ):
        part = torch.empty(
            (shape_tokens, topk, shape_hidden), dtype=torch.float32, device="cuda"
        ).normal_(0, 0.25)
        residual = torch.empty(
            (shape_tokens, shape_hidden), dtype=torch.bfloat16, device="cuda"
        ).normal_(0, 0.5)
        shared = torch.empty_like(residual).normal_(0, 0.5)
        reference = torch.empty_like(residual)
        output = torch.empty_like(residual)

        control_module.launch(
            control, 256, reference, residual, shared, part,
            shape_hidden, topk, shape_tokens,
        )
        candidate_module.launch(
            candidate, min(shape_tokens, 512), output, residual, shared, part,
            shape_hidden, topk, shape_tokens,
        )
        torch.cuda.synchronize()
        mismatches = int((reference.view(torch.int16) != output.view(torch.int16)).sum())
        print(
            f"oracle T={shape_tokens} H={shape_hidden} "
            f"values={output.numel()} mismatches={mismatches}"
        )
        if mismatches:
            raise SystemExit(2)
        if (shape_tokens, shape_hidden) == (tokens, hidden):
            primary = (part, reference, output)

    part, reference, output = primary

    control_ms = median_ms(
        lambda: control_module.launch(control, 256, reference, None, None, part, hidden, topk, tokens)
    )
    print(f"control grid=256 median_ms={control_ms:.6f}")
    for grid in (256, 512, 1024, 2048, tokens):
        if grid > tokens:
            continue
        elapsed = median_ms(
            lambda grid=grid: candidate_module.launch(
                candidate, grid, output, None, None, part, hidden, topk, tokens
            )
        )
        print(f"candidate grid={grid} median_ms={elapsed:.6f}")


if __name__ == "__main__":
    main()
