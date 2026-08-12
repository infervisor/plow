# Kimi-K3 MI325X B1 GEMV grid sweep

Date: 2026-08-11. Base: `9ae29f2749279af2cbd433c80b7b5f8079185a32`.
Toolchain: repository Nix shell, ROCm 7.14.0, gfx942, one MI325X.

## Result

The current-shape standalone sweep is useful as a rejection screen, but its
per-kernel optimum is not a packet optimum. Every winning grid was byte-exact.

| Shape | Current | Isolated best | Current us | Best us | Speedup |
|---|---:|---:|---:|---:|---:|
| o_proj 7168x1536 | 128 | 299 | 15.047 | 8.086 | 1.861x |
| router 896x7168 | 128 | 224 | 3.848 | 3.256 | 1.182x |
| routed_up 896x3584 | 128 | 302 | 3.435 | 2.429 | 1.414x |
| shared_down 7168x768 | 128 | 299 | 14.735 | 7.189 | 2.050x |
| latent_down 3584x7168 | 128 | 304 | 11.057 | 9.822 | 1.126x |
| kda_f_b 1536x128 | 128 | 256 | 3.510 | 2.575 | 1.363x |
| mla_q_a_g 1536x7168 | 128 | 293 | 5.415 | 4.580 | 1.183x |
| mla_q_rope 768x1536 | 128 | 299 | 2.834 | 2.151 | 1.318x |
| mla_q_absorb 6144x1536 | 128 | 293 | 13.089 | 7.581 | 1.727x |
| output_attn_res 4224x7168 | 128 | 176 | 12.994 | 11.649 | 1.115x |
| lm_head 163840x7168 | 304 | 128 | 483.794 | 450.874 | 1.073x |
| output_attn_proj 7168x4224 | 128 | 293 | 16.867 | 12.512 | 1.348x |

The instance-weighted isolated body projection was 6.296 -> 4.396 ms, a
1.432x speedup and 1.900 ms projected saving. Shapes not listed kept their
current grid.

The matched TP8 served result reversed the isolated result:

| Arm | TPOT ms | TTFT ms | Output tok/s |
|---|---:|---:|---:|
| current WG128 | 53.520 | 391.713 | 18.456 |
| all isolated winners | 62.922 | 392.923 | 15.731 |
| only o_proj grid299 | 54.932 | 392.651 | 17.988 |

Both candidates produced the exact control text and passed compact TP counter
auditing. The full profile regressed TPOT 17.57%; the serial-only o_proj arm
regressed 2.64%. More workgroups improve an isolated cold-weight body but add
interpreter/GQ/counter convergence work and disrupt useful packet overlap.
Keep the adopted WG128 packet policy.

## Reproduction

```sh
nix develop --command cmake -S runtime -B build-amd/k3-gemv-grid \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build build-amd/k3-gemv-grid \
  --target k3_gemv_grid_sweep -j2
nix develop --command perf-data/tools/gpulease -n1 k3-b1-gemv-grid \
  env PLOW_K3_GEMV_GRID_JSONL=/tmp/k3-b1-gemv-grid.jsonl \
  build-amd/k3-gemv-grid/bench/k3_gemv_grid_sweep \
  build-amd/k3-gemv-grid/bench/k3_gemv_grid_sweep_gfx942.co 15
```

The sweep uses the 16 live B1 BF16 GEMV shapes, their packet instance counts,
the production MM1/UN14/LG kernel, 1.5 GiB of rotated weights per timing arm,
forward/reverse grid order, and 15-sample medians. It tests grids
12,24,32,48,64,76,96,112,128,152,176,200,224,256,293,299,302,304.

Raw sweep JSONL SHA256:
`ec954711364c3f86d74d94da2927cef2dcde5f67d0febf54ab52c62850b076b1`.
The full-profile packet SHA256 was
`2bc0ac1845f0ea511dbc390b6240d3435c73b6ddd46a59df5e33c7172d985844`;
the o_proj-only packet was
`fd40552b74e8abf8286aecc7f708268755a85ecc72f162ca8ecb4354f73d5ff6`.
