# Kimi-K3 B1 dense W8A16 full-grid screen (rejected)

Date: 2026-08-11. Hardware: one MI325X (`gfx942`). Toolchain: repository Nix ROCm 7.14.0.

## Question

Can offline E4M3 weight quantization halve enough of K3's dense decode traffic to save at least
5 ms/token before paying the cost of a new checkpoint, packet op selection, and TP8 serving gate?

The model-free harness uses the production BF16, block-scaled E4M3, and per-output-row E4M3 GEMV
bodies. It executes all 16 current K3 B1 shapes at their emitted full-card grids, advances through
3 GiB BF16 and 1.5 GiB FP8 weight arenas, and weights each result by its packet count. This is a
full-grid single-op screen, not a one-block cache benchmark.

## Result

| Projected dense GEMV body | Time | Delta from BF16 |
|---|---:|---:|
| BF16 control | 6.329 ms | -- |
| block FP8 `[128,128]` | 5.805 ms | -0.524 ms |
| per-output-row FP8 | 6.945 ms | +0.616 ms |

The block arm wins large shapes such as `o_proj`, `shared_down`, `q_absorb`, and `lm_head`, but
loses every important narrow-output K=7168 projection. Per-row scaling avoids K-block promotion
but its production body is slower overall. Four independent runs projected block savings of
0.505--0.524 ms; the result is well below the 5 ms promotion bar. The final weighted BF16 A/A
control is 0.9976.

The synthetic quantization oracle uses `N=257,K=7168`, real BF16 inputs, and weights quantized from
the same random source. Relative L2 error against the BF16-weight reference is 3.04% for block FP8
and 2.98% for per-row FP8; cosine is 0.99954 and 0.99956. This only gates the harness. A real-weight
accuracy campaign would still be required if the performance screen won.

Kernel resources from the gfx942 ELF:

| Kernel | VGPR | SGPR | LDS | Private |
|---|---:|---:|---:|---:|
| BF16 | 152 | 52 | 16,384 B | 0 B |
| block FP8 | 256 | 71 | 16,384 B | 24 B |
| per-row FP8 | 113 | 59 | 16,384 B | 0 B |

The block arm reaches the VGPR cliff before being composed into the already resource-bound
interpreter. The per-row arm has safe resources but loses weighted time.

## Reproduction

```bash
nix develop --command cmake -S runtime -B /tmp/plow-k3-w8-screen \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build /tmp/plow-k3-w8-screen \
  --target k3_gemv_w8a16_sweep -j2
nix develop --command env \
  PLOW_K3_W8A16_JSONL=/tmp/k3-gemv-w8a16-screen-final2.jsonl \
  perf-data/tools/gpulease -n 1 k3-w8a16-screen \
  /tmp/plow-k3-w8-screen/bench/k3_gemv_w8a16_sweep \
  /tmp/plow-k3-w8-screen/bench/k3_gemv_w8a16_sweep_gfx942.co 9
```

## Decision

STOP. Do not create K3 FP8 dense-weight sidecars or add packet/runtime selection for this axis.
Even granting the entire isolated block-FP8 projection, current B1 TPOT would move by only about
0.5 ms against the roughly 33.7 ms gap to 20 ms/token. The next experiment must change live graph
overlap or a multi-millisecond kernel family rather than weight format alone.
