# Kimi-K3 B1 QKVG-to-Conv producer fusion screen (rejected)

Date: 2026-08-11. Hardware: one MI325X (`gfx942`). Toolchain: repository Nix ROCm 7.14.0.

## Question

Can KDA Q/K/V producers execute the shipping depthwise Conv3 body in the same workgroup and
remove the packet boundary without a device-wide barrier? The candidate assigns Q/K/V rows to the
same 304 workgroups used by `d_kda_conv3`, preserves the production K-order dot product and BF16
materialization, then calls the unchanged Conv3 body after a workgroup barrier. G remains an
independent output of the fused QKVG kernel.

The harness rotates eight exact TP8 shapes (`N=1536`, `K=7168`, `W=4`) through about 704 MiB of
weights, compares every raw Q/K/V/G byte, every mixed output byte, and every f32 Conv3 state byte,
and alternates the two-launch control with the one-launch candidate for 12 measured samples.

## Result

Two independent runs bracket zero:

| run | control, 8 layers | fused, 8 layers | projected delta over 69 layers |
|---|---:|---:|---:|
| initial | 0.184078 ms | 0.185798 ms | +0.014831 ms regression |
| recorded | 0.186577 ms | 0.185618 ms | 0.008276 ms improvement |

All runs have `raw_diff=0`, `mix_diff=0`, and `state_diff=0`. The candidate is 168 VGPR, 50 SGPR,
64,512 B LDS, zero private memory, and zero spills. The QKVG and Conv3 controls are respectively
132/41 and 20/50 VGPR/SGPR with zero spills.

## Decision

STOP. The boundary is bit-exactly removable, but the result is noise-level and at most about
0.01 ms/token, not the required multi-millisecond gain. Do not add a fused opcode or run a TP8
full-model campaign for this axis. Single-block/full-grid screening prevented an expensive model
run while exercising the production ownership, arithmetic, and cold-weight shape.

Raw recorded result: `/tmp/k3-qkvg-conv-final.jsonl`.

## Reproduction

```bash
nix develop --command cmake -S runtime -B /tmp/plow-k3-qkvg-conv \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build /tmp/plow-k3-qkvg-conv \
  --target k3_qkvg_conv_sweep -j2
nix develop --command perf-data/tools/gpulease -n 1 k3-qkvg-conv \
  /tmp/plow-k3-qkvg-conv/bench/k3_qkvg_conv_sweep \
  /tmp/plow-k3-qkvg-conv/bench/k3_qkvg_conv_sweep_gfx942.co
```
