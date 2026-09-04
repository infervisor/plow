# gfx950 KDA intra wave-item fold — 2026-09-04

## Decision

Promote the model-neutral exact-shape route by default on gfx950. A packet marks only pure,
ungated singleton `KdaChunkIntra` segments with `BT=64,D=128`; unsupported graphs keep the
ordinary interpreter. `PLOW_KDA_INTRA_WAVE_ITEMS=0` is the emission rollback. A marked packet
requires its packet-hash-paired specialist and the current runtime fails closed if it is absent,
stale, resource-invalid, or marker-invalid.

## Isolated body gate

The candidate assigns independent `(chunk, head)` items to wave64 waves while retaining each
item's ten BF16 MFMA block pairs and strict FP32 `i,j` forward substitution order. It does not use
the rejected reassociated parallel solve.

| body | samples (ms) | median | speedup |
|---|---|---:|---:|
| production | 1.743255, 1.741776, 1.739895 | 1.741776 | 1.000x |
| wave-item | 0.570285, 0.571365, 0.567406 | 0.570285 | 3.054x |

The isolated saving is 1.171491 ms/layer, projecting to 80.833 ms across 69 identical layers.
The complete downstream boundary is bitwise equal: zero mismatches in Aqk FP32 (6,291,456),
Ainv FP32 (6,291,456), q/W/U/output BF16 (12,582,912 each), and final state FP32 (196,608).
At this shape 1,536 independent items map six per workgroup, one per wave.

## Artifact gate

- Source: `aab03bb`; ROCm toolchain label: `rocm-7.14.0-nix`; target: MI355X/gfx950 TP8.
- Exact workload: BF16 KV, `T=8192`, one request, 256 generated tokens, concurrency 1.
- Both emissions used the measured TuneDB: 7,650/7,650 lookups measured.
- Control packet SHA256: `4d411aa532da53f38126b9d3547d08af0d3185eb3f8d5b85ddfef31c5dcd7e08`.
- Candidate packet SHA256: `f1bf783dac96791b7116ffb549862c8206ba33351310c7c113504916611e8921`.
- Candidate structure: 69 marked singleton segments, 17,664 marked entries (`69*256`), exact
  shape `[8192,12,128]`, zero gated, impure, or non-KDA entries. Control has zero markers and is
  byte-identical to the qualified C8 control packet.
- Specialist SHA256: `af59ce16918b56b7882394aed891c00c0ec49149a0b2ce86cde662e04e248d45`.
- Specialist resources: wave64, WG512, VGPR79, SGPR68, occupancy 2, LDS 131,072 B, private 0,
  zero VGPR/SGPR spills. All six ABI/capability markers and packet hash stamps are present.

Control and candidate differ only in structural segment marking and the required paired object.
C8, XReduce wave, packed-prefill consumers, and materialized MLA were off; decode MLA inventory
pruning and the current decode defaults were on.

## Full-network alternating folds

Each arm loaded its matching assets and HSACO directory under one exclusive eight-GPU lease per
fold group. Order was candidate/control, control/candidate, candidate/control. Every arm completed
and returned the same 256 token IDs (SHA256
`3b1345553d40748ce2baf58be3a0c20419d8662548dc3d4afa1d6ef04673a1ea`).

| fold | control TTFT ms | candidate TTFT ms | delta ms | TPOT delta ms | E2E delta ms |
|---|---:|---:|---:|---:|---:|
| 1 | 1500.393 | 1416.149 | -84.244 | +0.0257 | -77.702 |
| 2 | 1500.156 | 1415.907 | -84.249 | +0.0877 | -61.877 |
| 3 | 1500.480 | 1416.498 | -83.982 | +0.0413 | -73.455 |
| mean | 1500.343 | 1416.185 | **-84.158** | **+0.0516** | **-71.011** |

TTFT delta range is `[-84.249,-83.982]` ms with a 95% half-width of 0.379 ms. The TPOT delta
95% interval is `[-0.0287,+0.1318]` ms: no statistically material regression, and the route is
prefill-only. Every E2E fold wins. Raw fold summary SHA256:
`6cae8f246b056f0d8413923d3b42d82273a2b1ace4302464a73beaf223571ab3`.
