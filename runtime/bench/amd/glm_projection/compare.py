import argparse
import ctypes
import hashlib
import json
import statistics
from pathlib import Path

import torch


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    assert "gfx942" in torch.cuda.get_device_properties(0).gcnArchName
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = ctypes.CDLL(str(args.library.resolve()))
    fn = lib.plow_projection
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint] * 4 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int

    def read(name, shape):
        raw = bytearray((args.capture / name).read_bytes())
        value = torch.frombuffer(raw, dtype=torch.bfloat16)
        assert value.numel() == shape[0] * shape[1], (name, value.numel(), shape)
        return value.reshape(shape).cuda()

    def measure(call):
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                call()
        samples = []
        for _ in range(10):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 20)
        return {"median_ms": statistics.median(samples), "samples_ms": samples}

    shapes = [
        ("q_a", 2048, 6144, 1, ("x.bin", "wqa.bin", "qlr.bin")),
        ("kv_latent", 512, 6144, 2, ("x.bin", "wkv.bin", "ckvraw.bin")),
        ("q_absorb", 4096, 2048, 0, ("qlat.bin", "wabs.bin", None)),
        ("k_rope", 64, 6144, 3, None),
        ("q_rope", 512, 2048, 2, None),
        ("index_k", 128, 6144, 3, None),
        ("index_weight", 32, 6144, 3, None),
        ("router", 256, 6144, 3, None),
        ("o_proj", 6144, 2048, 0, None),
        ("shared_down", 6144, 256, 0, None),
    ]
    results = []
    for rows in [8192, 4464]:
        for name, n, k, kind, files in shapes:
            if files:
                a = read(files[0], (8192, k))[:rows]
                b = read(files[1], (n, k))
                captured = read(files[2], (8192, n))[:rows] if files[2] else None
            else:
                a = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5
                captured = None
            c = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
            d = torch.empty_like(c)

            def plow():
                rc = fn(c.data_ptr(), a.data_ptr(), b.data_ptr(), rows, n, k, kind,
                        torch.cuda.current_stream().cuda_stream)
                assert rc == 0, rc

            def blas():
                torch.mm(a, b.T, out=d)

            plow()
            torch.cuda.synchronize()
            sample = torch.tensor([0, rows // 2, rows - 1], device="cuda")
            oracle = a[sample].float() @ b.float().T

            def error(out):
                actual = out[sample].float()
                assert bool(torch.isfinite(out).all())
                rel = float(torch.linalg.vector_norm(actual - oracle) / torch.linalg.vector_norm(oracle))
                assert rel < 0.01, (name, rows, rel)
                return rel

            result = {"name": name, "rows": rows, "n": n, "k": k, "plow_kind": kind,
                      "inputs": "captured_layer38_full_chunk_prefix" if files else "seed0_random",
                      "plow_oracle_relative_l2": error(c), "plow": measure(plow)}
            if captured is not None:
                result["capture_bit_identical"] = torch.equal(c, captured)
                result["capture_mismatched_elements"] = int((c != captured).sum())
                result["capture_relative_l2"] = float(torch.linalg.vector_norm(c.float() - captured.float()) / torch.linalg.vector_norm(captured.float()))
            for backend in ["hipblas", "hipblaslt"]:
                torch.backends.cuda.preferred_blas_library(backend)
                blas()
                result[backend] = measure(blas)
                result[backend]["oracle_relative_l2"] = error(d)
            results.append(result)
            print(json.dumps(result), flush=True)
    record = {"torch": torch.__version__, "hip": torch.version.hip,
              "device": torch.cuda.get_device_name(0), "tp_shape": 8, "rank": 0,
              "scope": "Isolated BF16 projection bodies vs PyTorch BLAS preferences; GPU graph timings; excludes interpreter scheduling, collectives and serving",
              "capture_sha256": {p.name: sha(p) for p in sorted(args.capture.glob("*.bin"))},
              "library_sha256": sha(args.library), "results": results}
    args.out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
