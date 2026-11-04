#!/usr/bin/env python3
"""Build <out>/gemm_fp8_block_sm90.cubin, the native block-scaled FP8 decode GEMM object
(runtime/nvidia/gemm_fp8_block_sm90.cu; segment_roles NATIVE_DECODE_FP8_BLOCK). Packet-independent,
so it is built before the emit, which hashes it into segment_roles.json (PLOW_EMIT_DECODE_FP8_BLOCK).

Recipe use: a `[[prep]]` step with `args = ["{out}/assets"]`. Toolchain: PLOW_NVCC (default
/usr/local/cuda/bin/nvcc), PLOW_NVCC_PATH, NVCC_PREPEND_FLAGS / NVCC_APPEND_FLAGS as for build_sm90a_cubin.sh.
"""
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
NAME = "gemm_fp8_block_sm90.cubin"
SYMBOL = "plow_gemm_fp8_block_reduce"


def command(nvcc, out):
    nv = REPO / "runtime/nvidia"
    return [nvcc, "-std=c++17", "-arch=sm_90a", "-O3", "-cubin", "-I", str(REPO / "runtime/common"),
            "-I", str(nv), "-o", str(out), str(nv / "gemm_fp8_block_sm90.cu")]


def build(assets):
    nvcc = os.environ.get("PLOW_NVCC", "/usr/local/cuda/bin/nvcc")
    env = {"PATH": os.environ.get("PLOW_NVCC_PATH") or f"{os.path.dirname(nvcc)}:/usr/bin:/bin"}
    for k in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    assets.mkdir(parents=True, exist_ok=True)
    out = assets / NAME
    subprocess.run(command(nvcc, out), env=env, check=True)
    symbols = subprocess.run([os.path.join(os.path.dirname(nvcc), "cuobjdump"), "-symbols", str(out)],
                             env=env, check=True, capture_output=True, text=True).stdout
    if SYMBOL not in symbols:
        sys.exit(f"FATAL: {SYMBOL} missing in {out}")
    print(f"built {out} ({out.stat().st_size} B)")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: build_sm90a_fp8_block.py <assets dir>")
    build(pathlib.Path(sys.argv[1]))
