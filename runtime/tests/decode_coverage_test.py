#!/usr/bin/env python3
"""CPU-only rejection gates for whole-object decode capability exports."""
import argparse
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
cmd = ["env", "-i", "PATH=/usr/local/cuda/bin:/usr/bin:/bin", args.nvcc,
       "-E", "-arch=sm_90a", "-I", str(root / "runtime/common"),
       "-I", str(root / "runtime/nvidia"), "-DPLOW_NV_GEMMA=1",
       "-DPLOW_NV_EMBED_SMEM=1", "-DPLOW_NV_FA_GF=2",
       "-DPLOW_NV_MLA=0", "-DPLOW_NV_MAMBA=0", "-DPLOW_NV_DSA=0",
       str(root / "runtime/nvidia/interp_sm90a.cu")]
cases=[('broad',[],1),('lean',['PLOW_NV_LEAN_DECODE=1'],0),('segmented',['PLOW_NV_SEGMENTS=1'],0),('skeleton',['PLOW_NV_SKELETON=1'],0),('trace',['PLOW_NV_TRACE=1'],0),('static',['PLOW_NV_SCHED=0'],0),('pruned-heads',['PLOW_NV_GEMMA=0'],0),('gemv512',[],0),('ablation',['PLOW_NV_ABLATE_LO=0'],0),('prefill',['PLOW_NV_PREFILL=1'],0),('fa-only',['PLOW_NV_PREFILL=1','PLOW_NV_SEGMENTS=1','PLOW_NV_FA_ONLY=1'],0),('gemm-only',['PLOW_NV_PREFILL=1','PLOW_NV_SEGMENTS=1','PLOW_NV_SEG_GEMM=1','PLOW_NV_GEMM_ONLY=1'],0)]
results = []
for name, defines, expected in cases:
    keys = {d.split("=")[0] for d in defines}
    call = [x for x in cmd if not (x.startswith("-D") and x[2:].split("=")[0] in keys)]
    call[-1:-1] = ["-D" + d for d in defines]
    if name == "gemv512":
        call[-1] = str(root / "runtime/nvidia/interp_sm90a_gemv512.cu")
    result = subprocess.run(call, text=True, capture_output=True)
    values = re.findall(r"unsigned\s+plow_decode_bf16_abi(?:_\w+)?\s*=\s*([01])\s*;", result.stdout)
    passed = result.returncode == 0 and bool(values) and all(int(v) == expected for v in values)
    results.append({"case": name, "expected": expected, "values": values,
                    "passed": passed, "rc": result.returncode, "stderr": result.stderr})
print(json.dumps(results, indent=2))
assert all(r["passed"] for r in results)
