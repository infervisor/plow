#!/usr/bin/env python3
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
SWEEP = (ROOT / "runtime/bench/gemm/gemv_row_sweep.c").read_text()
KERNELS = (ROOT / "runtime/amd/test_kernels.hip").read_text()


def values(name: str) -> list[int]:
    match = re.search(rf"static const unsigned {name}\[\] = \{{([^}}]+)\}};", SWEEP)
    assert match, f"{name} is missing"
    return [int(value) for value in match.group(1).split(",")]


assert values("DEMAND_MMS") == [1, 2, 4, 8, 16, 32, 64, 128]
assert values("BUCKET_MMS") == [1, 2, 4, 8, 16]
assert "if (MM > M) continue;" not in SWEEP
assert "for (int im = 0; im < NDEMAND; im++)" in SWEEP
assert "for (int ib = 0; ib < NBUCKET; ib++)" in SWEEP
assert '"gemv_qkvg_m"' in SWEEP
assert 'getenv("PLOW_GEMV_ARMS")' in SWEEP
assert "if (!enabled[arm]) continue;" in SWEEP
assert "if (mx_ok && want_mx)" in SWEEP
assert "want_bf16 ? plow_hsa_alloc" in SWEEP
assert "dWq2 = plow_hsa_alloc" in SWEEP
assert "if (enabled[5])" in SWEEP

for bucket in values("BUCKET_MMS"):
    assert f"GEMV_QKVG_WALK_VARIANT(gemv_qkvg_m{bucket}, {bucket})" in KERNELS

assert "else if (K == 7168)" in KERNELS
print("gemv row sweep static test: ok")
