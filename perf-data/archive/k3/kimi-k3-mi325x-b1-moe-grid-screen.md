# Kimi-K3 B1 grouped-MoE block/grid screen

Date: 2026-08-11. Hardware: one MI325X (`gfx942`). Toolchain: repository Nix ROCm 7.14.0.

## Question

Can a literal one-block sweep replace a full-model TP8 experiment when screening K3 grouped-MoE
changes?

The model-free harness executes the exact B1 routed-expert shapes: top-16 of 896 experts,
`H=3584`, per-rank `I=384`, native packed MXFP4 weights, and the production GLU and DOWN device
bodies. It rotates through a 1.83 GiB arena spanning all experts, so the selected 35 MiB phase
cannot turn into a cache-resident microbenchmark. Every grid is checked byte-for-byte against the
304-block output.

## Grid result

| Blocks | GLU | DOWN | Chain | Chain x 92 layers |
|---:|---:|---:|---:|---:|
| 1 | 3642.90 us | 2475.93 us | 6120.67 us | 563.101 ms |
| 12 | 309.47 us | 211.99 us | 521.01 us | 47.933 ms |
| 32 | 119.07 us | 81.93 us | 200.62 us | 18.457 ms |
| 64 | 61.78 us | 42.86 us | 104.23 us | 9.589 ms |
| 96 | 42.58 us | 31.35 us | 73.48 us | 6.760 ms |
| 128 | 32.91 us | 23.03 us | 55.56 us | 5.112 ms |
| 152 | 32.40 us | 20.19 us | 52.18 us | 4.801 ms |
| 192 | 23.20 us | 17.27 us | 40.13 us | 3.692 ms |
| 256 | 18.36 us | 14.36 us | 32.26 us | 2.968 ms |
| 304 | 18.42 us | 11.50 us | 29.48 us | 2.713 ms |

All grids produce zero differing GLU and DOWN bytes. The independent 512 MiB streaming control
measures 4445 GB/s. Raw CSV SHA256:
`1ad63b5346a93038e201b85cdc4a0bd5499e64b62d7f935586dc4ceb986dacec`.

A literal one-block result is therefore not promotion evidence. It underfills the 304-CU device
and overstates the production-grid chain by 207x. A single-op full-grid sweep is useful: it removes
model load and TP8 iteration cost while retaining the relevant ownership, concurrency, traffic,
and cold-weight behavior.

## Interpreter integration screen

The matched current packet has 2343 packets and a 50.474 ms observed dependency-spine span.
The grouped GLU and DOWN spine charges are 4.499 and 3.973 ms, versus 2.713 ms for their isolated
full-grid chain. This exposes 5.758 ms of integration/queue/resource headroom, but it is not all
device-call overhead.

A default-off build axis force-inlined the two grouped bodies into the interpreter without changing
VGPR, LDS, private memory, packet bytes, or arithmetic. It removed the two callable bodies but grew
`plow_exec` instructions by 3.7%. The matched TP8 trace moved only:

| Metric | Control | Force-inline | Delta |
|---|---:|---:|---:|
| Full dependency-spine span | 50.474 ms | 50.142 ms | -0.332 ms |
| Grouped GLU + DOWN charge | 8.471 ms | 8.132 ms | -0.339 ms |
| Two-step wall | 52.476 ms/token | 52.146 ms/token | -0.330 ms/token |

Both arms generate `[13,646]`, all eight ranks agree, and the compact TP audit is clean. A more
aggressive top-level direct dispatch increased interpreter spills and scratch instructions, so it
was rejected at the static gate and removed before GPU execution.

## Reproduction

```bash
nix develop --command cmake -S runtime -B /tmp/plow-k3-moe-grid-build \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build /tmp/plow-k3-moe-grid-build \
  --target k3_moe_grid_sweep -j2
nix develop --command bash -lc \
  'perf-data/tools/gpulease -n 1 k3-moe-grid \
   /tmp/plow-k3-moe-grid-build/bench/k3_moe_grid_sweep \
   /tmp/plow-k3-moe-grid-build/bench/k3_moe_grid_sweep_gfx942.co \
   > /tmp/k3-moe-grid-sweep.csv 2> /tmp/k3-moe-grid-sweep.log'

nix develop --command python3 scripts/k3_trace_spine.py \
  /tmp/k3-moe-inline-control-trace.bin /tmp/k3-current-disasm.json --top 50

nix develop --command env PLOW_DECODE_BATCH=1 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 PLOW_K3_DECODE_MXFP4_PROJ=0 \
  PLOW_K3_MOE_GROUP_FORCEINLINE=1 JOBS=2 \
  scripts/build_gfx942.sh /tmp/k3-moe-inline-candidate
```

The HSACO bundle SHA256 is
`20affba7beac667081c58c0ad2225cd2f6212b4ef45ae1cd9190d518049256b4`.

## Decision

Keep the exact single-op/full-grid harness as the first screening gate. Do not use literal
one-block timing to choose a production grid, and do not promote grouped-body force-inlining: its
0.34 ms gain is far below the 5 ms structural-experiment bar. The next isolated target is the KDA
Conv3/state sequence, where the current observed spine charges about 13.2 ms and a sound fusion
requires double-buffered recurrent convolution state.
