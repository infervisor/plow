#!/usr/bin/env python3
"""Qualify query-row partitioning and time a conservative TP8 peer-copy gather."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capture-rows", type=int, default=8192)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--protocol", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.device_count() == 8
    assert all("gfx942" in torch.cuda.get_device_properties(r).gcnArchName for r in range(8))
    assert all(torch.cuda.can_device_access_peer(r, s)
               for r in range(8) for s in range(8) if r != s)
    lib = ctypes.CDLL(str(args.library.resolve()))
    call = lib.plow_index
    call.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint] * 5 + [ctypes.c_void_p]
    call.restype = ctypes.c_int
    peer_gather = lib.plow_gather
    peer_gather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                           ctypes.c_uint, ctypes.c_void_p]
    peer_gather.restype = ctypes.c_int
    if args.protocol:
        protocol = lib.plow_index_protocol
        protocol.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_uint] * 4 + [ctypes.c_uint64, ctypes.c_void_p]
        protocol.restype = ctypes.c_int
    assert lib.plow_enable_peers() == 0

    def read(name, dtype):
        return torch.frombuffer(bytearray((args.capture / name).read_bytes()), dtype=dtype)

    q = read("q.bin", torch.bfloat16).reshape(-1, 32, 128)
    k = read("k.bin", torch.bfloat16).reshape(-1, 128)
    w = read("w.bin", torch.bfloat16).reshape(-1, 32)
    captured_idx = read("idx.bin", torch.int32).reshape(-1, 2048)
    captured_len = int(read("len.bin", torch.int32)[0])
    assert 1 <= args.capture_rows <= min(q.shape[0], w.shape[0], captured_idx.shape[0], captured_len)
    stride = k.shape[0]
    inputs = [(q.to(r), k.to(r), w.to(r)) for r in range(8)]

    def sync():
        for r in range(8):
            torch.cuda.synchronize(r)

    results = []
    short_rows = min(129, args.capture_rows)
    cases = sorted({(1, 1), (short_rows, short_rows), (short_rows, captured_len),
                    (min(4464, args.capture_rows), captured_len),
                    (args.capture_rows, captured_len)})
    for rows, length in cases:
        band = (rows + 7) // 8
        padded = band * 8
        scores, indices, lengths = [], [], []
        for r in range(8):
            scores.append(torch.full((rows, stride), -float("inf"), device=r))
            indices.append(torch.full((padded, 2048), -1, dtype=torch.int32, device=r))
            lengths.append(torch.tensor([length], dtype=torch.int32, device=r))
        peers = [torch.tensor([t.data_ptr() for t in indices], dtype=torch.uint64, device=r)
                 for r in range(8)]
        if args.protocol:
            counter_offset = rows * 2048 * 4
            staging = [torch.zeros(counter_offset + 4096 * 128, dtype=torch.uint8, device=r)
                       for r in range(8)]
            counter_peers = [torch.tensor([t.data_ptr() for t in staging], dtype=torch.uint64, device=r)
                             for r in range(8)]
            status = [torch.zeros(1, dtype=torch.int32, device=r) for r in range(8)]
            next_gate = 0

        def launch(r, ranged):
            with torch.cuda.device(r):
                begin, end = (r * band, min((r + 1) * band, rows)) if ranged else (0, rows)
                tensors = [indices[r], scores[r], *inputs[r], lengths[r]]
                rc = call(*(t.data_ptr() for t in tensors), rows, stride, begin, end,
                          int(ranged), torch.cuda.current_stream().cuda_stream)
                assert rc == 0, rc

        def compute(ranged):
            for r in range(8):
                launch(r, ranged)
            sync()

        def gather():
            for dst in range(8):
                with torch.cuda.device(dst):
                    rc = peer_gather(indices[dst].data_ptr(), peers[dst].data_ptr(), band,
                                     dst, torch.cuda.current_stream().cuda_stream)
                    assert rc == 0, rc
            sync()

        def device_protocol(wait=True):
            nonlocal next_gate
            assert next_gate + 2 < 4096
            for r in range(8):
                with torch.cuda.device(r):
                    tensors = [indices[r], staging[r], scores[r], *inputs[r], lengths[r],
                               counter_peers[r], status[r]]
                    rc = protocol(*(t.data_ptr() for t in tensors), rows, stride, r,
                                  next_gate, counter_offset, torch.cuda.current_stream().cuda_stream)
                    assert rc == 0, rc
            next_gate += 3
            if wait:
                sync()

        compute(False)
        reference_score = scores[0].clone()
        reference_idx = indices[0][:rows].sort(dim=1).values
        capture_match = None
        if rows == args.capture_rows and length == captured_len:
            capture_match = torch.equal(reference_idx.cpu(), captured_idx[:rows].sort(dim=1).values)
            if not capture_match:
                got = reference_idx.cpu()
                expected = captured_idx[:rows].sort(dim=1).values
                overlap = {r: int(torch.isin(got[r], expected[r]).sum())
                           for r in sorted({0, min(127, rows - 1), rows // 2, rows - 1})}
                raise AssertionError(f"model capture mismatch: sorted entries "
                                     f"{int((got != expected).sum())}; row overlaps {overlap}")
        for r in range(8):
            scores[r].fill_(-float("inf"))
            indices[r].fill_(-1)
        compute(True)
        for r in range(8):
            lo, hi = min(r * band, rows), min((r + 1) * band, rows)
            assert torch.equal(scores[r][lo:hi].to(0), reference_score[lo:hi]), (rows, r)
            assert bool(torch.isneginf(scores[r][:lo]).all()), (rows, r, "prefix overwritten")
            assert bool(torch.isneginf(scores[r][hi:]).all()), (rows, r, "suffix overwritten")
            assert bool((indices[r][:lo] == -1).all()), (rows, r, "index prefix overwritten")
            assert bool((indices[r][hi:] == -1).all()), (rows, r, "index suffix overwritten")
        gather()
        for r in range(8):
            assert torch.equal(indices[r][:rows].sort(dim=1).values.to(0), reference_idx), (rows, r)
            assert torch.equal(indices[r].to(0), indices[0]), (rows, r, "gather differs by rank")
            assert bool((indices[r][rows:] == -1).all()), (rows, r, "padding overwritten")
        if args.protocol:
            for r in range(8):
                scores[r].fill_(-float("inf"))
                indices[r].fill_(-1)
            device_protocol()
            for r in range(8):
                lo, hi = min(r * band, rows), min((r + 1) * band, rows)
                assert torch.equal(scores[r][lo:hi].to(0), reference_score[lo:hi]), (rows, r)
                assert torch.equal(indices[r][:rows].sort(dim=1).values.to(0), reference_idx), (rows, r)
            history = [[torch.empty_like(indices[r]) for r in range(8)] for _ in range(16)]
            sync()
            for step in range(16):
                device_protocol(wait=False)
                for r in range(8):
                    with torch.cuda.device(r):
                        history[step][r].copy_(indices[r], non_blocking=True)
                        staging[r][:counter_offset].fill_(0xA5)
            sync()
            for step in range(16):
                for r in range(8):
                    got = history[step][r]
                    assert torch.equal(got[:rows].sort(dim=1).values.to(0), reference_idx), (step, rows, r)
                    assert torch.equal(got.to(0), history[step][0]), (step, rows, r, "protocol gather")
                    assert bool((got[rows:] == -1).all()), (step, rows, r, "protocol padding")
            del history
        del reference_score, reference_idx

        def measure(fn):
            for _ in range(3):
                fn()
            times = []
            for _ in range(args.reps):
                start = time.perf_counter()
                fn()
                times.append((time.perf_counter() - start) * 1000)
            return {"median_ms": statistics.median(times), "samples_ms": times}

        def partitioned():
            compute(True)
            gather()

        record = {"rows": rows, "kv_len": length, "stride": stride,
                  "score_bit_exact": True, "topk_set_exact_all_ranks": True,
                  "gather_bit_exact_all_ranks": True,
                  "model_capture_topk_set_exact": capture_match,
                  "replicated": measure(lambda: compute(False)),
                  "partitioned_compute": measure(lambda: compute(True)),
                  "gather": measure(gather), "partitioned_total": measure(partitioned)}
        if args.protocol:
            record["device_protocol"] = measure(device_protocol)
            for r in range(8):
                assert int(status[r][0]) == 0, (rows, r, "protocol timeout")
                counters = staging[r][counter_offset:].view(torch.int32)[::32]
                assert bool((counters[:next_gate] == 8).all()), (rows, r, "gate arrivals")
                assert bool((counters[next_gate:] == 0).all()), (rows, r, "unused gates")
            record["protocol_reuse_iterations"] = 16
            record["audited_gates"] = next_gate
            record["protocol_status_zero_all_ranks"] = True
            del staging, counter_peers, status
        results.append(record)
        print(json.dumps(record), flush=True)
        del scores, indices, lengths, peers
        torch.cuda.empty_cache()

    metadata = {"torch": torch.__version__, "hip": torch.version.hip,
                "device": torch.cuda.get_device_name(0), "ranks": 8,
                "capture_rows": args.capture_rows,
                "protocol_gates_per_layer": 3 if args.protocol else None,
                "scope": ("Captured layer38 index score+select; native device protocol with three gates"
                          if args.protocol else "Captured layer38 index score+select; host barriers and peer copies")
                         + "; no model serving claim",
                "capture_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(args.capture.glob("*.bin"))},
                "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
                "results": results}
    args.out.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
