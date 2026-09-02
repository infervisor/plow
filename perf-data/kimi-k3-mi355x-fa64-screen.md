# Kimi K3 MI355X `f_a_proj` workgroup screen

Status: **screened candidate; not promoted**.

## Change

The packet-only candidate applies the generic exact-shape table
`PLOW_GEMV_WG_TUNING=128x7168=64`. It changes 69 KDA `f_a_proj` GEMV
instructions from 128 to 64 blocks and saves 4,416 workgroups per decode token.
No other instruction field changes. The interpreter objects are identical
between arms.

## Cell

- GPU: 8 × AMD MI355X (`gfx950`, 256 CUs/rank), TP8.
- Input/output: random greedy 8192 → 64, C1, two measured requests after one warmup.
- Runtime: production `plowrt bench`, global queue, B1 low-rung object, compact TP audit,
  device state clear, identical row-parallel prefill object.
- The reverse-order pair held an exclusive eight-GPU `gpulease` and returned
  rc=0. The first pair is supporting screen evidence only because it predates
  the lease wrapper.
- Correct object: `build-amd/k3-gfx950-hsaco-b128-kda-rowpar/hsaco`. The older
  similarly named object is not part of this comparison.

## Results

| Pair | Arm | TTFT mean (ms) | TPOT mean (ms) | Output checksum |
|---|---|---:|---:|---|
| control → candidate | control | 3600.280 | 45.1423 | `fnv1a64:060423f6dbc80987` |
| control → candidate | candidate | 3600.883 | 44.6879 | `fnv1a64:060423f6dbc80987` |
| candidate → control | candidate | 3598.783 | 44.6707 | `fnv1a64:060423f6dbc80987` |
| candidate → control | control | 3598.607 | 45.1513 | `fnv1a64:060423f6dbc80987` |

Mean TPOT: 45.1468 → 44.6793 ms = **1.04% faster**. TTFT is unchanged within
noise. The reverse-order repeat reproduces the direction and output identity.

## Promotion gate

Do not make this the default from this screen. Promotion requires a newly
emitted control with `build.json`, at least five interleaved endpoint rounds,
the pinned 8192 → 1024 cell, and the full correctness gate. The current
candidate manifest reports Lean ordering and oracle checks as passed, but all
7,650 GEMM tile lookups remain analytical because the MI355X tuning cell has no
qualified measurements.
