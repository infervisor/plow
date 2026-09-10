import argparse
import ctypes
import hashlib
import json
import re
import statistics
import struct
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object", type=Path, required=True)
    p.add_argument("--selected", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--export", type=Path)
    p.add_argument("--mapping", choices=["linear", "xcc"], default="linear")
    args = p.parse_args()
    torch.cuda.set_device(0)
    torch.cuda.init()
    torch.backends.cuda.preferred_blas_library("hipblaslt")
    torch.backends.cuda.matmul.allow_tf32 = False
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    hip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    module = ctypes.c_void_p()
    assert hip.hipModuleLoad(ctypes.byref(module), str(args.object).encode()) == 0
    selected = json.loads(args.selected.read_text())

    def read(name, shape):
        raw = bytearray((args.capture / name).read_bytes())
        return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(shape).cuda()

    results = []
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
    for name, n, k, af, bf in [("q_a", 2048, 6144, "x.bin", "wqa.bin"),
                               ("q_absorb", 4096, 2048, "qlat.bin", "wabs.bin"),
                               ("kv_latent", 512, 6144, "x.bin", "wkv.bin")]:
        a = read(af, (8192, k))
        b = read(bf, (n, k))
        for rows in [1, 129, 4464, 8192]:
            # Both large-row library choices are checked at all live row counts.
            for tuned_rows in [4464, 8192]:
                config = next(d for d in selected if int(d["N"]) == tuned_rows and int(d["M"]) == n and int(d["K"]) == k)
                symbol = config["kernel_name"]
                mt0, mt1, _ = map(int, re.search(r"MT(\d+)x(\d+)x(\d+)", symbol).groups())
                wg = re.search(r"_WG(\d+)_(\d+)_(\d+)$", symbol)
                threads = int(wg[1]) * int(wg[2]) * int(wg[3])
                kernel = ctypes.c_void_p()
                assert hip.hipModuleGetFunction(ctypes.byref(kernel), module, symbol.encode()) == 0
                out = torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16)
                reference = a[:rows] @ b.T
                grid = ((n + mt0 - 1) // mt0) * ((rows + mt1 - 1) // mt1)
                raw = bytearray(160)
                mapping = re.search(r"_WGM(\d+)_WGMXCC(\d+)_WGMXCCGn1$", config["solution_Name"])
                assert mapping
                info1 = 1 if args.mapping == "linear" else (int(mapping[1]) & 0xFFFF) | (int(mapping[2]) << 16)
                # Inline single-GEMM args, GSU=1, no stagger; XCC group zero uses the whole grid.
                struct.pack_into("<8I", raw, 0, 1, 1, info1, grid, n, rows, 1, k)
                struct.pack_into("<4Q", raw, 32, out.data_ptr(), out.data_ptr(), b.data_ptr(), a.data_ptr())
                struct.pack_into("<8I", raw, 64, n, n*rows, n, n*rows, k, k*n, k, k*rows)
                struct.pack_into("<2f", raw, 96, 1.0, 0.0)
                struct.pack_into("<Q", raw, 140, out.data_ptr())
                buffer = ctypes.create_string_buffer(bytes(raw))
                size = ctypes.c_size_t(160)
                extra = (ctypes.c_void_p * 5)(1, ctypes.addressof(buffer), 2, ctypes.addressof(size), 3)

                def launch():
                    rc = hip.hipModuleLaunchKernel(kernel, grid, 1, 1, threads, 1, 1, 0,
                            torch.cuda.current_stream().cuda_stream, None, extra)
                    assert rc == 0, rc

                launch()
                torch.cuda.synchronize()
                assert bool(torch.isfinite(out).all()), (name, rows, tuned_rows, "nonfinite")
                relative = float(torch.linalg.vector_norm(out.float()-reference.float()) / torch.linalg.vector_norm(reference.float()))
                assert relative < 0.004, (name, rows, tuned_rows, relative)
                if args.export and tuned_rows == (4464 if rows <= 4464 else 8192):
                    (args.export / f"{name}-{rows}.bin").write_bytes(out.view(torch.uint8).cpu().numpy().tobytes())
                samples = []
                for _ in range(3): launch()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(20): launch()
                for _ in range(10):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); graph.replay(); end.record(); end.synchronize()
                    samples.append(start.elapsed_time(end)/20)
                result = dict(name=name, rows=rows, n=n, k=k, tuned_rows=tuned_rows,
                              solution_index=int(config["solution_index"]), kernel_name=symbol,
                              threads=threads, grid=grid, info0=1, info1=info1, kernarg_bytes=160,
                              relative_l2_vs_library=relative, bit_identical=torch.equal(out, reference),
                              median_ms=statistics.median(samples), samples_ms=samples)
                print(json.dumps(result), flush=True)
                results.append(result)
    record = dict(torch=torch.__version__, hip=torch.version.hip, device=torch.cuda.get_device_name(0),
                  object_sha256=hashlib.sha256(args.object.read_bytes()).hexdigest(),
                  scope="Direct HIP module launch of hipBLASLt assembly, no library matmul in timed path; inline ABI; not HSA serving qualification",
                  mapping=args.mapping, results=results)
    args.out.write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()
