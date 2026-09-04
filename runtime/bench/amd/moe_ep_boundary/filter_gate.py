#!/usr/bin/env python3
import ctypes
import struct
import sys

import torch

T, TOPK, E, RANKS, BM = 8192, 16, 896, 8, 64
BEGIN, END = 0, E // RANKS
NPART = 64
CAPACITY = T * TOPK + E * (BM - 1)


def call(lib, name, *args):
    rc = getattr(lib, name)(*args)
    if rc:
        raise RuntimeError(f"{name}: hipError_t {rc}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: filter_gate.py filter.elf")
    lib = ctypes.CDLL("libamdhip64.so")
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    call(lib, "hipModuleLoad", ctypes.byref(module), sys.argv[1].encode())
    call(lib, "hipModuleGetFunction", ctypes.byref(function), module,
         b"plow_moe_ep_filter_align_gfx950")

    state = 9301
    experts = []
    for _ in range(T * TOPK):
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        state &= 0xffffffff
        experts.append(state % E)
    gate_bits = struct.unpack("I", struct.pack("f", 0.25))[0]
    packed = torch.tensor([e | (gate_bits << 32) for e in experts],
                          dtype=torch.int64, device="cuda")
    meta = torch.empty(3 * E + 1, dtype=torch.int32, device="cuda")
    partial = torch.empty(NPART * E, dtype=torch.int32, device="cuda")
    row_token = torch.empty(CAPACITY, dtype=torch.int32, device="cuda")
    row_partidx = torch.empty(CAPACITY, dtype=torch.int32, device="cuda")
    row_gate = torch.empty(CAPACITY, dtype=torch.float32, device="cuda")

    holders = [ctypes.c_void_p(x.data_ptr()) for x in
               (packed, meta, partial, row_token, row_partidx, row_gate)]
    holders += [ctypes.c_uint32(x) for x in
                (T, TOPK, E, BEGIN, END, CAPACITY)]

    def launch():
        for phase, blocks in ((1, NPART), (2, 1), (3, NPART), (4, NPART)):
            tail = [ctypes.c_uint32(phase), ctypes.c_uint32(NPART)]
            params = (ctypes.c_void_p * (len(holders) + len(tail)))(*[
                ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in holders + tail
            ])
            call(lib, "hipModuleLaunchKernel", function,
                 ctypes.c_uint(blocks), ctypes.c_uint(1), ctypes.c_uint(1),
                 ctypes.c_uint(256), ctypes.c_uint(1), ctypes.c_uint(1),
                 ctypes.c_uint(0), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                 params, ctypes.c_void_p())

    launch(); torch.cuda.synchronize()
    mh = meta.cpu()
    rows = int(mh[3 * E]) * BM
    expected = [[] for _ in range(E)]
    for p, e in enumerate(experts):
        if BEGIN <= e < END:
            expected[e].append(p)
    expected_idx = []
    for bucket in expected:
        expected_idx.extend(bucket + [-1] * ((-len(bucket)) % BM))
    got = row_partidx[:rows].cpu().tolist()
    if got != expected_idx:
        raise SystemExit("stable filter/sort oracle mismatch")
    if any(int(mh[E + e]) != len(bucket)
           for e, bucket in enumerate(expected)):
        raise SystemExit("count oracle mismatch")

    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    samples = []
    for _ in range(31):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record(); launch(); end.record(); end.synchronize()
        samples.append(begin.elapsed_time(end))
    samples.sort()
    print(f"exact T={T} topk={TOPK} E={E} owned={END-BEGIN} rows={rows} "
          f"median_ms={samples[len(samples)//2]:.6f} errors=0")


if __name__ == "__main__":
    main()
