# gfx950 materialized MLA prefill gate — 2026-09-03

## Decision

The generic `D_QK=192`, `D_V=128` materialized path clears the isolated kernel and
full attention-path cost gates, and reduces median TP8 8192-token TTFT by 8.63%.
Keep it default-off: the 8192→256 continuation gate diverges after the first output
token. Continuation chunks and packed prefill remain fail-closed because the current
KV projection materializes only one exact initial bucket.

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

## TP8 full-network qualification

Both BF16 arms were emitted from commit `f5e3ec7` with 7650/7650 GEMM tile lookups
selected from measured TuneDB records. Every program passed Lean ordering and LDS
certification. The only emitter knob delta was
`PLOW_MLA_MATERIALIZED_PREFILL=1`; packed prefill remained off. Separate object sets
carried matching packet hashes (`0x48a4ccb34189de4a` control,
`0xd5519e6021ff8cc2` candidate).

| fold | control TTFT | materialized TTFT | delta |
|---:|---:|---:|---:|
| 1 | 1506.236 ms | 1377.952 ms | -8.52% |
| 2 | 1508.941 ms | 1379.950 ms | -8.55% |
| 3 | 1509.457 ms | 1378.661 ms | -8.67% |
| median | 1508.941 ms | 1378.661 ms | -8.63% |

All three 8192→1 folds produced token 6896 and checksum
`fnv1a64:7d749e3b002fafa7`. The independent 8192→256 gate measured TTFT
1508.407 vs 1378.715 ms, end-to-end latency 12861.235 vs 12773.381 ms, and TPOT
44.521 vs 44.685 ms. It did
not pass exact continuation parity: token zero matched, token one was 444 vs 633,
and 255/256 positions differed. The checksums were
`fnv1a64:6bdfaa7b84ee4e7e` vs `fnv1a64:a1ff6fcc6ce97b50`.
Therefore the performance result is accepted as an experiment, but promotion is
rejected pending a continuation-quality gate appropriate for numerically equivalent
attention formulations. Raw JSON, logs, traces, and SHA-256 inventory are under
`/tmp/k3-mla-b9c8a9b/results`.

### Continuation diagnosis

Offline disassembly rules out a different decode graph. Both packets have 2165 decode
instructions, 54125 counters, and 307454 stream entries. Opcode order, block counts,
named tensor operands, scalar operands, and all cache-referencing instruction classes
are identical. The raw packet differs in 396 integer fields named `W_g`, `w_v`,
`cs_q`, `cs_k`, `cs_v`, or `dt_bias`; all are tensor handles shifted by the additional
prefill-only tensors, and every shifted pair resolves to the same tensor name.

The 8192 prefill cache writers are also identical: 24 `RmsNorm` writes to `kv.*.ckv`
and 24 `HeadNormRope` writes to `kv.*.krot`, with byte-identical named operands and
scalars after removing handle numbers. The candidate's 24 additional GEMMs read
`kv.*.ckv` to materialize K/V but write transient `act.pf.kv_materialized`; they do not
alter cache layout or cache contents. The divergence is therefore numerical drift in
the alternative prefill attention formulation, amplified by autoregressive decode,
not a continuation-program or cache-routing difference.

Rollback/current status: the emitter knob remains default-off. Unset
`PLOW_MLA_MATERIALIZED_PREFILL` (or set it to `0`) to retain the exact absorbed path.
Continuation chunks and packed-cache routing continue to fail closed.

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
