# MI355X TP8 AITER/Plow collective parity — 2026-09-04

## Contract

- vLLM image: `vllm/vllm-openai-rocm:v0.28.0`, digest `e0a3b2bd...`, AITER 0.1.19.
- AITER source: `/tmp/aiter-v0.1.19`, commit `31350226161346314b3d8882c8085bd31dce6a34`.
- Eight MI355X ranks in physical order 0 through 7, BF16, 10 warmups and 25 samples.
- AITER time is the maximum GPU-event time across ranks. `registered` isolates the graph-style
  kernel path; `eager` includes its device-to-device copy into the IPC pool.
- Plow uses the production grid: 7/14 workgroups for B1 one-shot and 256 workgroups for prefill
  RS-U2 plus staggered AG. Three repetitions report the median rank-0 device time.
- Every GPU run held `/tmp/gpulease` for all eight devices. `xreduce-aiter-full`,
  `xreduce-plow-full`, `xreduce-empty-parity`, and `xreduce-aiter-benign` all exited zero with no
  foreign process.

The model-generic AITER harness is `scripts/bench_aiter_custom_ar.py`. The existing Plow harness
now accepts `TP_NWG`, so decode can use the emitted grid instead of its old fixed 64-workgroup
probe grid.

## Pinned dispatch and resources

TP8 AITER uses one stage below 80 KiB and two-stage RS+AG otherwise. Both use 512 threads and
16-byte packs. The one-stage grids are 7 and 14 blocks for 3,584 and 7,168 BF16 values. Large
messages use 80 blocks. The two-stage kernel assigns one output partition to each rank, rotates
the peer list by local rank, reduces one peer per wave through LDS, publishes the local reduced
partition, then gathers all peer partitions.

vLLM's custom-AR window ends at 64 MiB. Its two actual 4,096-token chunks therefore use custom AR
at 28 and 56 MiB. The 8,192x7,168 112 MiB row below directly probes the kernel with a larger pool;
the served vLLM path falls through to its other backend at that size.

| object | wave | VGPR | SGPR | LDS | private | spills |
|---|---:|---:|---:|---:|---:|---:|
| AITER TP8 BF16 one-stage | 64 | 42 | 30 | 16,384 B | 80 B | 0 |
| AITER TP8 BF16 two-stage | 64 | 48 | 77 | 8,192 B | 0 | 0 |
| Plow focused two-shot U2 | 64 | 24 | 68 | 4 B | 0 | 0 |
| Plow focused folded-gather U2 | 64 | 28 | 74 | 4 B | 0 | 0 |

## Result

Times are milliseconds per collective. Lower is better.

| rows x width | bytes/rank | AITER registered | AITER eager | Plow | AITER / Plow |
|---|---:|---:|---:|---:|---:|
| 1 x 3,584 | 7 KiB | 0.014680 | 0.016347 | 0.000964 | 15.23x |
| 1 x 7,168 | 14 KiB | 0.020533 | 0.023392 | 0.000981 | 20.93x |
| 4,096 x 3,584 | 28 MiB | 0.145302 | 0.141334 | 0.020913 | 6.95x |
| 4,096 x 7,168 | 56 MiB | 0.256183 | 0.271300 | 0.035292 | 7.26x |
| 8,192 x 3,584 | 56 MiB | 0.256763 | 0.271489 | 0.035308 | 7.27x |
| 8,192 x 7,168 | 112 MiB | 0.500250 | 0.541644 | 0.063450 | 7.88x |

The 16-byte AITER registered cell is 11.503 us. Its fixed launch plus two peer barriers therefore
accounts for 78% of the 7 KiB decode cell and 56% of the 14 KiB cell. Plow's corresponding inline
cell is 0.868 us. AITER eager copy-in adds 2.7 us at decode and 14.7 to 41.4 us at the stable
large-message points. The anomalous 28 MiB eager inversion is only 4.0 us and is not used as a
mechanism.

Both implementations pass the benign rank-value oracle. AITER one-stage also passes the strict
rank-order oracle. AITER two-stage does not: with rank values `[2^24, 1, -2^24, 0, ...]`, exactly
25% of every large output differs from the rank-0-through-rank-7 FP32 sum on every rank. This is
the expected consequence of its rank-rotated peer order. Plow passes the strict oracle at every
shape.

## Decision

Do not port the AITER topology or packet schedule. It is 6.9x to 20.9x slower in the isolated
apples-to-apples cells and its large-message arithmetic contract is weaker. No generic AITER
mechanism clears the 10% family or 15 ms projected-TTFT gate.

The remaining in-network 70.708 ms RS and 106.892 ms AG are not an AITER parity gap. The focused
Plow object sums to only about 17.1 ms over the model's 92 half, 94 full, and 92 folded-gather
calls, while the primary prefill interpreter carries 256 VGPR, 1,348 B private storage, and both
VGPR and SGPR spills. The next design remains a graph-selected, spill-free XReduce phase object:
retain scalar U2 and strict order, then test the per-workgroup ready-word handoff already specified
in `mi355x-xreduce-wave-rs-design-20260904.md`. Its gate must include the known +464 segment
launches (~3.03 ms), ready-word reset bytes, all-rank exactness, and an in-network phase trace.

Raw artifacts: `/tmp/xreduce-parity-results/`. Plow object SHA-256:
`1da28a5df78ad37f790ff96008bfd04aa5b8667f9569a0268bc6381767c22d7c`.
AITER gfx950 bundle SHA-256:
`cea5f6c89bf2346db7b1131c7160cfc1ec5b65b3cd77f64cba1cdbadb6ae0b6c`.
