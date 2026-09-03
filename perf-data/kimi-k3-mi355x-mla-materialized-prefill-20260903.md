# gfx950 materialized MLA prefill gate — 2026-09-03

## Decision

The generic `D_QK=192`, `D_V=128` materialized path clears the isolated kernel and
full attention-path cost gates. Keep it default-off until an exact TP8 full-network
fold passes. Continuation chunks and packed prefill remain fail-closed because the
current KV projection materializes only one exact initial bucket.

## Configuration

- Device: one uncontended MI355X through `perf-data/tools/gpulease -n 1`
- Harness: `runtime/bench/amd/mla_materialized_prefill/run.sh`
- Samples: nine, median reported, cache flushed before each sample
- Production object: dependency-free AITER Opus subset at upstream revision
  `10b192f5b5bda90f2af33ceae7a6c2f416bfc674`, MIT license
- Capability selection: gfx950, bf16, wave64, `D_QK=192`, `D_V=128`, ABI 1;
  no model-name predicate

## Correctness

The Plow object flattens upstream's `(q-block, head, batch)` grid into the 1D grid
accepted by plowrt. A separately built unmodified 3D-grid oracle produced byte-identical
bf16 output for all 12 local Q/KV heads:

| T | flat vs 3D mismatches | head coverage |
|---:|---:|---:|
| 1024 | 0 | 12/12 |
| 8192 | 0 | 12/12 |
| 1025 (ragged grid) | 0 | 12/12 |

Against the recorded absorbed-form numerical oracle on exact buckets:

| T | max abs | RMSE | limit |
|---:|---:|---:|---:|
| 1024 | 0.0038681 | 0.00277082 | 0.02 / 0.003 |
| 8192 | 3.05176e-5 | 3.20288e-7 | 0.02 / 0.003 |

The standalone pack object's K and V outputs are bit-exact over every element.

## Timing

Both totals include their query projection GEMMs. The candidate additionally includes
the KV projection, K/V packing, and standalone Opus attention. The control includes
absorbed attention and the output fold.

| T | absorbed total | materialized total | speedup | Opus only |
|---:|---:|---:|---:|---:|
| 1024 | 2818.656 us | 1650.449 us | 1.708x | 33.881 us |
| 8192 | 12836.146 us | 3976.153 us | 3.228x | 349.763 us |

This is an isolated nine-sample screen, not a publication baseline. Promotion still
requires three order-alternated exact TP8 8192→1 folds and an exact 8192→256 gate.

## Resources

| object/kernel | wave/WG | VGPR/SGPR/AGPR | occupancy | LDS | private/spill |
|---|---|---|---:|---:|---:|
| materialized Opus | 64/512 | 254/88/0 | 2 | 149760 B | 0/0 |
| K/V pack | 64/256 | 24/48/0 | 8 | 0 | 0/0 |

The oracle's four projection wrappers inherit the current `d_gemm` resource shape:
VGPR 256, SGPR 84, occupancy 1, LDS 147456 B, and 2092 B/lane scratch. This affects
both control query projections and candidate projections. It does not erase the measured
candidate gain, but it remains a separate GEMM optimization lever; it is not packed into
either standalone raw object.
