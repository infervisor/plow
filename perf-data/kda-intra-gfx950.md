# gfx950 KDA full-BT64 intra optimization

Date: 2026-09-03. Shape: `T=8192 D=128 BT=64`, bf16 Q/K and f32 Aqk/Ainv. Runs use
`GPU_LEASE_DIR=/tmp/plow-gpulease-shared perf-data/tools/gpulease` and the production body plus
an isolated candidate wrapper from `runtime/tests/kda_step_cdna3_test.hip`.

## Accepted TP8-local result

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
- `k_intra_cached`: wave64, 74 VGPR, 79 SGPR, 114,688 B LDS, zero private bytes, zero VGPR/SGPR
  spills. The benchmark script fails closed on wave/private/spill and VGPR/LDS ceilings.

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

## Decision

Keep this as an isolated generic kernel/harness candidate. Do not route the interpreter yet. The
next gate is a separately compiled lean KDA-intra object or an opt-in interpreter arm, followed by
the persistent-packet oracle and full-network trace. No model-name heuristic is needed: the ABI
shape gate is BT64, D128, wave64, and adequate gfx950 LDS.
