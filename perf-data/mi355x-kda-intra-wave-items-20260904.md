# MI355X KDA Intra wave-item gate

Date: 2026-09-04. Shape: `T=8192, H=12, D=V=128, BT=64`, one gfx950 GPU.

## Result

The candidate assigns independent `(chunk, head)` items to wave64 waves while leaving each
item's ten BF16 MFMA block pairs and strict FP32 `i,j` forward substitution in production order.
It does not use the previously rejected reassociated parallel solve.

| body | samples (ms) | median | speedup |
|---|---|---:|---:|
| production | 1.743255, 1.741776, 1.739895 | 1.741776 ms | 1.000x |
| wave-item | 0.570285, 0.571365, 0.567406 | 0.570285 ms | 3.054x |

The isolated saving is 1.171491 ms/layer, or 80.833 ms across 69 identical KDA layers before
segment-launch overhead. The older singleton-Intra network gate attributed about 7.6 ms total
overhead to 69 standalone boundaries, so this result leaves a material network opportunity.

## Exact oracle

The two Intra outputs and the complete downstream q-precompute/carry boundary are bitwise equal:

| tensor | mismatches / elements |
|---|---:|
| Aqk FP32 | 0 / 6,291,456 |
| Ainv FP32 | 0 / 6,291,456 |
| q BF16 after q-precompute | 0 / 12,582,912 |
| W BF16 | 0 / 12,582,912 |
| U BF16 | 0 / 12,582,912 |
| output BF16 | 0 / 12,582,912 |
| final state FP32 | 0 / 196,608 |

At this shape there are 1,536 independent items and 256 workgroups, so waves 0 through 5 each
own one item. Each wave uses a disjoint 64x64 FP32 matrix. Wave-local barriers replace workgroup
barriers; waves with no item may exit without blocking an active wave.

## Resource gate

| kernel | VGPR | SGPR | occupancy | LDS | scratch | VGPR/SGPR spills |
|---|---:|---:|---:|---:|---:|---:|
| production | 67 | 78 | 7 | 16,384 B | 0 | 0 / 0 |
| wave-item | 79 | 68 | 2 | 131,072 B | 0 | 0 / 0 |

The candidate meets its limits: wave64/WG512, VGPR at most 96, occupancy at least 2, LDS at
most 128 KiB, zero scratch, and zero register spills. It belongs in a separate singleton object,
not the already-spilling primary prefill interpreter.

## Production gate

Keep the route default-off. Reuse the existing singleton KDA-Intra segment machinery with a new
exact capability marker. Select only on gfx950, wave64, BT64, D128, the raw non-packed ABI, and
enough independent items for wave assignment; otherwise use the production body. Before promotion:

1. Verify the emitted singleton object has the same resource envelope and packet pairing.
2. Pass exact TP8 BF16 `8192 -> 256` output IDs and checksum on carried state.
3. Pass three order-alternated `8192 -> 1` folds with trace attribution.

No model name participates in selection.

## Artifacts

- Raw gate: `/tmp/plow-kda-intra-wave-items-final.log`, SHA-256
  `ed9d74f3534ce8622aa2032d71ae91fcc2087c1b75d0751b93e201a773aa8a19`, lease return code 0.
- Executable: SHA-256 `7f952829c0caa01ba81ceb84d9182eb72504af9f04c3b8862d890d8f68691fb4`.
- Resource log: SHA-256 `c9d0dcf9e2bd257a5e617be2985d2c70f66200bbe4204caa4951455e90273486`.
