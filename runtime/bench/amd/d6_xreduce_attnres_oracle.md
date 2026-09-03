# D6-A XR14→AttnRes carrier oracle

This isolated TP8 harness compares the existing device-side XR→AttnRes packet
protocol with an eight-XCD gang carrier. It does not route production graphs or
add an interpreter arm.

The control stays in one persistent WG14 launch: production
`d_xreduce_oneshot`, a 14-WG completion counter, then production
`d_materialize_residual` and `d_attn_res` on block 0. The candidate uses the
same reduction, elects one leader per physical XCD, and distributes the exact
AttnRes thread order over those eight leaders. This prices the device-side
packet boundary and ordering, not a separate host/AQL launch.

## Reproduce

From the repository root at `3890af8`:

```sh
nix develop -c bash -lc '
hipcc --offload-arch=gfx950 -O3 -std=c++17 -x hip --genco \
  -Rpass-analysis=kernel-resource-usage \
  -DPLOW_K3=1 -DPLOW_BUCKET_DECODE=1 \
  -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=1 \
  -DPLOW_L2_PLACE_DISPATCH=1 -DPLOW_GATE_HIER=1 \
  -DPLOW_MATERIALIZED_RESIDUAL_INPUT=1 \
  runtime/bench/amd/d6_xreduce_attnres_oracle.hip \
  -Iruntime/amd -Iruntime/common -o /tmp/d6.co
p=$(readlink -f "$(command -v hipcc)")
b=$(dirname "$(dirname "$p")")/lib/llvm/bin
"$b/clang-offload-bundler" -unbundle -type=o -input=/tmp/d6.co \
  -targets=hipv4-amdgcn-amd-amdhsa--gfx950 -output=/tmp/d6.elf
'

/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/gcc -O2 -std=gnu11 \
  -Wall -Wextra -Werror -o /tmp/d6_oracle \
  runtime/bench/amd/d6_xreduce_attnres_oracle.c runtime/amd/hsa_backend.c \
  -Iruntime/amd -I/opt/rocm/include -L/opt/rocm/lib \
  -lhsa-runtime64 -lm

D6_ELF=/tmp/d6.elf D6_ITERS=200 \
  perf-data/tools/gpulease -n 8 d6-a-xr-attnres-tp8 \
  /tmp/d6_oracle 0 1 2 3 4 5 6 7
```

The preflight uses the candidate's WG512, grid14, wave and 147,460-byte LDS
envelope. It reads the hardware XCD ID from thread 0 and requires histogram
`{2,2,2,2,2,2,1,1}` without assuming block ordering. A mismatch fails before
the carrier launch, and every device wait has a finite deadline.

## Resource gate

ROCm 7.14.0, gfx950:

| kernel | SGPR | VGPR | LDS bytes | occupancy | wave | private | SGPR/VGPR spills |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d6_control` | 106 | 114 | 147,456 | 2 | 64 | 0 | 0/0 |
| `d6_fused` | 102 | 100 | 147,460 | 2 | 64 | 0 | 0/0 |
| `d6_xcd_map` | 12 | 3 | 147,460 | 2 | 64 | 0 | 0/0 |

All three kernels have maximum workgroup size 512. The LDS envelope admits one
workgroup per CU; the 14-workgroup launch is intentional.

## Exact and timing result

The 108-case sweep passed byte-for-byte on all eight ranks: direct/t6/t7,
`nb=0..8`, ring push off/on, and gamma/raw output. It checked the mixed result,
rank-ordered BF16 reduction scratch against a CPU oracle, materialized prefix,
ring contents, peer gates, phase counters, local rendezvous counters, and the
successor counter.

Timing uses the maximum device cycles across TP8. Site weights are the exact
eligible graph census: direct=8, t6=92, t7=86; the unrelated initial `nb=0`
AttnRes is excluded.

| contract | nb | sites | control us | carrier us | delta us |
|---|---:|---:|---:|---:|---:|
| direct | 1 | 1 | 2.144 | 2.728 | +0.584 |
| direct | 2 | 1 | 2.215 | 2.818 | +0.603 |
| direct | 3 | 1 | 2.284 | 2.894 | +0.610 |
| direct | 4 | 1 | 2.368 | 2.987 | +0.619 |
| direct | 5 | 1 | 2.443 | 3.061 | +0.618 |
| direct | 6 | 1 | 2.526 | 3.161 | +0.635 |
| direct | 7 | 1 | 2.597 | 3.238 | +0.641 |
| direct | 8 | 1 | 2.685 | 3.324 | +0.639 |
| t6 | 1 | 11 | 2.406 | 3.383 | +0.978 |
| t6 | 2 | 12 | 2.479 | 3.460 | +0.981 |
| t6 | 3 | 12 | 2.545 | 3.541 | +0.996 |
| t6 | 4 | 12 | 2.623 | 3.632 | +1.009 |
| t6 | 5 | 12 | 2.708 | 3.718 | +1.010 |
| t6 | 6 | 12 | 2.796 | 3.801 | +1.005 |
| t6 | 7 | 12 | 2.873 | 3.890 | +1.017 |
| t6 | 8 | 9 | 2.963 | 3.978 | +1.015 |
| t7 | 1 | 12 | 2.407 | 3.376 | +0.969 |
| t7 | 2 | 11 | 2.482 | 3.464 | +0.982 |
| t7 | 3 | 11 | 2.546 | 3.542 | +0.996 |
| t7 | 4 | 11 | 2.621 | 3.628 | +1.007 |
| t7 | 5 | 11 | 2.709 | 3.716 | +1.007 |
| t7 | 6 | 11 | 2.797 | 3.802 | +1.005 |
| t7 | 7 | 11 | 2.873 | 3.889 | +1.016 |
| t7 | 8 | 8 | 2.962 | 3.979 | +1.017 |

Weighted total: control=0.494 ms, carrier=0.677 ms, delta=+0.183 ms across
186 sites. D6-A is rejected: the exact, zero-spill carrier is slower before any
interpreter integration cost. Keep it out of production graph/runtime routes.
