#!/usr/bin/env python3
import ctypes
import struct
import sys

import torch

T, H, TOPK, E = 8192, 3584, 16, 896
BEGIN, END = 0, E // 8


def call(lib, name, *args):
    rc = getattr(lib, name)(*args)
    if rc:
        raise RuntimeError(f"{name}: hipError_t {rc}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: combine_gate.py combine.elf")
    lib = ctypes.CDLL("libamdhip64.so")
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    call(lib, "hipModuleLoad", ctypes.byref(module), sys.argv[1].encode())
    call(lib, "hipModuleGetFunction", ctypes.byref(function), module,
         b"plow_moe_ep_combine_gfx950")

    state = 9301
    experts = []
    for _ in range(T * TOPK):
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        state &= 0xffffffff
        experts.append(state % E)
    gate_bits = struct.unpack("I", struct.pack("f", 0.25))[0]
    routes = torch.tensor([e | (gate_bits << 32) for e in experts],
                          dtype=torch.int64, device="cuda")
    part = torch.empty((T * TOPK, H), dtype=torch.float32, device="cuda")
    selected = torch.tensor([p for p, e in enumerate(experts) if BEGIN <= e < END],
                            dtype=torch.int64, device="cuda")
    part.index_fill_(0, selected, 0.25)
    out = torch.empty((T, H), dtype=torch.bfloat16, device="cuda")
    holders = [ctypes.c_void_p(x.data_ptr()) for x in (out, part, routes)]
    holders += [ctypes.c_uint32(x) for x in (T, H, TOPK, BEGIN, END)]
    params = (ctypes.c_void_p * len(holders))(*[
        ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in holders
    ])

    def launch():
        call(lib, "hipModuleLaunchKernel", function,
             ctypes.c_uint(T * ((H + 255) // 256)), ctypes.c_uint(1), ctypes.c_uint(1),
             ctypes.c_uint(256), ctypes.c_uint(1), ctypes.c_uint(1),
             ctypes.c_uint(0), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
             params, ctypes.c_void_p())

    launch(); torch.cuda.synchronize()
    counts = torch.tensor([
        sum(BEGIN <= experts[t * TOPK + s] < END for s in range(TOPK))
        for t in range(T)
    ], dtype=torch.float32)
    got = out[:, 0].float().cpu()
    if not torch.equal(got, (counts * 0.25).to(torch.bfloat16).float()):
        raise SystemExit("fixed-slot combine oracle mismatch")
    if not torch.equal(out[:, 0], out[:, -1]):
        raise SystemExit("combine columns disagree")

    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    samples = []
    for _ in range(31):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record(); launch(); end.record(); end.synchronize()
        samples.append(begin.elapsed_time(end))
    samples.sort()
    print(f"exact T={T} H={H} topk={TOPK} owned={END-BEGIN} selected={selected.numel()} "
          f"median_ms={samples[len(samples)//2]:.6f} errors=0")


if __name__ == "__main__":
    main()
