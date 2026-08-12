# Kimi-K3 B1 router TopK block sweep

Date: 2026-08-11. Hardware: one MI325X (`gfx942`). Toolchain: repository Nix
ROCm 7.14.0. Scope: exact one-workgroup router tail, without model weights.

## Experiment

The control rescans each lane's expert keys on every one of the 16 block-wide
maximum reductions. The candidate sorts and caches each lane's keys once. Both
run the production `d_moe_router_topk` at the K3 B1 shape: 896 experts,
top-16, sigmoid, correction bias, renormalization, and one 512-thread block.
Synthetic BF16 logits rotate across 64 rows. Twenty-one samples are interleaved.

Both objects use 51 VGPR, occupancy 4, and zero spills. Candidate and control
produce byte-identical routing tables across all 64 rows; an independent host
oracle also reports zero id differences and zero maximum gate difference.

| arm | median us/layer | p10 | p90 | projected ms/token (92 layers) |
|---|---:|---:|---:|---:|
| rescanned control | 17.290 | 17.225 | 17.333 | 1.591 |
| cached local keys | 15.409 | 15.369 | 15.457 | 1.418 |

The candidate saves only **1.881 us/layer**, or **0.173 ms/token**. This is a
valid single-block optimization, but its entire projected gain is too small to
justify rebuilding or serving the full TP8 model. Do not promote it to B1.
The router score → TopK packet boundary cannot recover multiple milliseconds
without also changing the score projection itself.

## Reproduction

```bash
nix develop --command cmake -S runtime -B /tmp/plow-k3-router-topk-sweep \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build /tmp/plow-k3-router-topk-sweep \
  --target k3_router_topk_sweep -j2
nix develop --command perf-data/tools/gpulease -n 1 k3-router-topk-sweep \
  /tmp/plow-k3-router-topk-sweep/bench/k3_router_topk_sweep \
  /tmp/plow-k3-router-topk-sweep/bench/k3_router_topk_control_gfx942.elf \
  /tmp/plow-k3-router-topk-sweep/bench/k3_router_topk_local_gfx942.elf
```

Raw output is `/tmp/k3-router-topk-block-sweep.txt` (SHA256
`9c07e2258b229e7f92a699ac96e208a76d89bab4c6c9b1be9c8d762f749c15b9`).
Control/local object SHA256 values are `87392715c87f...` and `fe931e6e9df9...`.
