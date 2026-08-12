# Kimi-K3 B1 MoeCombine → XReduce screen (MI325X TP8)

## Decision

Reject the fused boundary. The exact TP8 full-grid candidate is bit-exact, but its median is
`1.536 us/layer` versus `1.367 us/layer` for the current two-packet protocol model. Across 92 MoE
layers this projects a `0.016 ms/token` regression. No packet, opcode, interpreter, or runtime
integration is justified.

## Question and protocol

The adopted B1 trace charges about `2.193 ms` to 92 `MoeCombine` packets and `1.502 ms` to their
latent-width `XReduce` successors. The screen tests whether removing that packet boundary exposes
material latency.

Both arms use the production K3 shape: TP8, latent width 3584, top-k 16, seven 512-thread
workgroups. Both execute the same arithmetic and rounding:

1. Accumulate 16 f32 expert partials in fixed slot order.
2. Store the local combined value as BF16 into peer-visible memory.
3. Sum the eight ranks' BF16 values in rank order and store BF16 output.

The control recreates the interpreter boundary: seven device-local release arrivals, an agent
acquire, one rank-wide system release, then the peer reduction. The candidate publishes each
completed combine slice through the proven `PLOW_XR_AGG` protocol: seven system-release arrivals
on word 1, one closer/acquire, and one `+7` release to every peer. The remote target is `8*7`.
Distinct scratch and gate lines per iteration prevent generation aliasing.

## Result

ROCm 7.14 Nix toolchain, gfx942, one exclusive eight-GPU lease, 2,048 iterations per sample, 12
alternating samples:

| arm | median us/layer | projected 92-layer contribution |
|---|---:|---:|
| current two-packet protocol model | 1.367385 | 0.1258 ms/token |
| fused per-slice publication | 1.536313 | 0.1413 ms/token |
| fused minus control | +0.168928 | +0.015541 ms/token |

Every element on every rank matched the independent CPU oracle and the control snapshot byte for
byte. Both kernels compile at 16 VGPR, 72 SGPR, 4 B LDS, zero private memory, and zero spills.
The cold first control sample is intentionally excluded by the median; a separate 32-iteration
ABBA run also rejects the candidate (`1.389 us` control, `1.563 us` fused).

Raw output: `/tmp/k3-cxr-final.txt`.

## Reproduction

```bash
nix develop --command cmake -S runtime -B /tmp/plow-k3-cxr-build \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build /tmp/plow-k3-cxr-build \
  --target tp_moe_combine_xreduce_bench -j2

nix develop --command env GPU_LEASE_NGPU=8 TP_ITERS=2048 TP_SAMPLES=12 \
  perf-data/tools/gpulease -n 8 k3-cxr \
  /tmp/plow-k3-cxr-build/bench/tp_moe_combine_xreduce_bench \
  /tmp/plow-k3-cxr-build/bench/tp_moe_combine_xreduce_gfx942.elf
```

The packet trace's 3.695 ms combined envelope is therefore not removable combine/XReduce seam
overhead. It includes scheduling and dependency time outside the isolated bodies. Continue with a
different multi-operation boundary; do not add this fusion.
