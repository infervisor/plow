#!/usr/bin/env python3
"""Self-test of build_sm90a_fp8_block.py: the compile line, and a real build when nvcc is present."""
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_sm90a_fp8_block as b  # noqa: E402

cmd = b.command("/x/nvcc", pathlib.Path("/o/f.cubin"))
assert cmd[:6] == ["/x/nvcc", "-std=c++17", "-arch=sm_90a", "-O3", "-cubin", "-I"], cmd
assert cmd[-1].endswith("runtime/nvidia/gemm_fp8_block_sm90.cu") and "/o/f.cubin" in cmd
assert subprocess.run([sys.executable, str(HERE / "build_sm90a_fp8_block.py")], capture_output=True).returncode != 0
if os.path.exists(os.environ.get("PLOW_NVCC", "/usr/local/cuda/bin/nvcc")):
    with tempfile.TemporaryDirectory() as d:
        out = b.build(pathlib.Path(d) / "assets")
        assert out.name == b.NAME and out.stat().st_size > 0
print("ok")
