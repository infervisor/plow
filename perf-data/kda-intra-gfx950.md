# gfx950 KDA full-BT64 intra optimization

Date: 2026-09-03. Shape: `T=8192 D=128 BT=64`, bf16 Q/K and f32 Aqk/Ainv. Runs use
`GPU_LEASE_DIR=/tmp/plow-gpulease-shared perf-data/tools/gpulease` and the production body plus
an isolated candidate wrapper from `runtime/tests/kda_step_cdna3_test.hip`.

## Initial TP8-local result (rejected at network correctness)

K3 TP8 executes `H=12` per rank. Three no-foreign-PID folds at grid 256:

| fold | production intra ms | candidate intra ms |
|---:|---:|---:|
| 1 | 1.8775 | 0.4136 |
| 2 | 1.8800 | 0.4134 |
| 3 | 1.8768 | 0.4133 |
| median | **1.8775** | **0.4134** |

The candidate is 4.54x faster and clears the 0.9 ms/layer isolated target. At the combined trace's
69 KDA layers this is a 101.0 ms body-time opportunity before interpreter integration effects.

The candidate caches the ten gated lower-triangular block-pair operands in LDS. More importantly,
it distributes each row of the f32 64x64 forward substitution over all eight wave64 waves. The
original solve used wave 0 only. Four waves measured 0.6042 ms in the schedule screen; eight waves
is selected.

## Numerical and resource gates

- Aqk is bit-identical to the production body on the direct full-BT64 oracle.
- Candidate Aqk max abs vs f64 = `2.9e-9..3.4e-9`, gate `5e-5`.
- Candidate Ainv max abs vs f64 = `2.6e-8..4.3e-8`, gate `3e-4`. Reassociation in the parallel
  solve means Ainv is not bit-identical to the serial f32 solve.
- Composed chunk path vs serial recurrence at T8192: output relL2 `5.8287e-3`, state relL2
  `4.8086e-3`, gates `1e-2` / `5e-3`.
- The isolated benchmark wrapper compiled wave64 with 74 VGPR, 79 SGPR, 114,688 B LDS, zero
  private bytes, and zero VGPR/SGPR spills. These are not the shipping object's resources.

## Global-H96 comparator and rejected probes

`H=96` is the unsharded/global shape; it is not the per-rank TP8 trace shape. The no-foreign-PID
shared-lease fold measured production 15.0141 ms vs candidate 3.5061 ms. This does not clear
0.9 ms on one GPU.

- LDS caching without the parallel solve: 1.6941 ms at H12. Operand reuse alone is insufficient.
- Four waves: 0.6042 ms at H12, slower than eight waves.
- BF16-hi/lo block-MFMA inverse: 0.3761 ms at H12 and Ainv max abs `1.34e-6`, but 10 SGPR spills;
  rejected by the zero-spill gate.
- A 12-block base cache plus re-anchoring factors cut LDS to 72,704 B with zero spills, but the
  extra bf16 rounding failed the direct oracle (Aqk `1.145e-4`, Ainv `8.466e-4`); rejected.

## Segment integration result

The cached body was integrated behind `PLOW_KDA_INTRA_CACHED=1` as a pure raw segment and a
separately compiled gfx950 object. The runtime selects it only for a singleton BT64/D128 packet
with the exact raw ABI; a missing, rejected, mixed, packed, or unsupported object falls back to
the ordinary interpreter. The object is never compiled into the mega-interpreter.

On current BF16 defaults (qpre and materialized-residual fusion enabled, packed segmentation
disabled), three order-balanced TP8 `8192 -> 1` `plowrt bench` folds measured:

| fold | interpreter ms | parallel cached ms | delta ms |
|---:|---:|---:|---:|
| 1 | 1781.735 | 1688.381 | -93.354 |
| 2 | 1780.307 | 1686.330 | -93.977 |
| 3 | 1780.309 | 1687.627 | -92.682 |

The paired median was -93.354 ms (-5.2%). All `8192 -> 1` arms returned token 6896 and checksum
`fnv1a64:7d749e3b002fafa7` with zero failures. This is not promotable evidence: the required
`8192 -> 256` carried-state gate failed at output index 1 (control 444, candidate 633), with
252/256 token mismatches and checksums `fnv1a64:6bdfaa7b84ee4e7e` vs
`fnv1a64:11646861c8bd56b6`.

The fault was the eight-wave parallel forward substitution. It reduced products in wave-major
order instead of the production loop's strict `j=1..63` subtraction order. Small one-layer Ainv
differences compounded across 69 recurrent layers. Restoring the production wave-0 substitution
order makes both Aqk and Ainv bit-exact in the bounded direct `IT=145, IH=2, D=128` oracle. Its
`0/18560` counts cover the 18,560 stored triangle slots compared by that oracle, not every element
of the T8192 timing shape. With operand caching retained, the separate T8192/H12 timing is
1.6980 ms vs 1.8790 ms. Metadata from the exact repaired shipping object is wave64, 68 VGPR,
73 SGPR, 114,688 B LDS, zero private bytes, and zero VGPR/SGPR spills.

## Decision

Reject the faster parallel-solve variant. Keep only the bit-exact serial-order cached segment as
an opt-in experiment; do not enable it by default until it passes the full `8192 -> 256` token
oracle and fresh order-balanced network folds. No model-name heuristic is used: the ABI/resource
gate is gfx950, BT64, D128, wave64, zero private/spills, and 114,688 B static LDS.
