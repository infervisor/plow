#!/usr/bin/env bash
# Probe the vLLM CPU build for a native int8 MoE kernel and the capability gate.
# libnuma lives in /usr/lib, gcc runtime in vllm-cpu/gcc-lib — both needed.
export LD_LIBRARY_PATH=/home/lava/vllm-cpu/gcc-lib/lib:/usr/lib/x86_64-linux-gnu
cd /home/lava/vllm-cpu
./venv/bin/python - <<'PY' 2>&1 | grep -v "^INFO\|^WARNING"
import torch, vllm._C  # noqa: F401
for op in ("cpu_fused_moe", "cpu_fused_moe_int8"):
    print(f"{op:20s}:", hasattr(torch.ops._C, op))
from vllm.platforms import current_platform as P
print("platform            :", P.device_name)
print("device_capability   :", P.get_device_capability())
print("supported_quant     :", P.supported_quantization)
PY
