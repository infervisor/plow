# Fixed-order MoE combine experiment — gfx950

Date: 2026-09-03. Status: generic runtime integration is available behind the
packet opt-in; promotion remains pending. Hardware: one uncontended MI355X
under `gpulease`.

## Current-model attribution

The final P0 BF16-KV 8192-token artifact contains 92 structurally eligible
stage-1 packets and 92 structurally eligible stage-2 packets. Runtime segment
drains provide an independent route check:

| Boundary | Segments | Drain range | Approximate total |
|---|---:|---:|---:|
| lean MXFP4 stage 1 | 92 | 2.100–2.469 ms | 221–224 ms from the qualified folds |
| lean MXFP4 stage 2 | 92 | 0.683–0.803 ms | about 72 ms |

The critical-rank packet trace attributes 146.655 ms to 92
`MoeCombinePf` packets, or 1.594 ms/layer. Those packets still execute in the
primary prefill interpreter, whose object has 256 VGPR, 8 VGPR spills, and
1,348 bytes of private memory per thread. The combine reads
`8192*16*3584*4 = 1,879,048,192` bytes of f32 part data per layer before its
small bf16 output write. At the repository's measured 6.2 TB/s ceiling, the
part read alone is about 0.303 ms/layer; the current body sustains about
1.18 TB/s.

## Candidate

The standalone kernel assigns complete token rows to workgroups and sums f32
parts in the exact interpreter order: residual, shared, then slots 0 through
15. It retains the materialized part tensor. This removes the flat-grid
division/modulo from the hot loop and permits a resident-grid sweep without
changing arithmetic.

The initial build gate reports wave64, workgroup 256, 37 VGPR, 49 SGPR, no LDS,
zero private memory, and zero VGPR/SGPR spills. Disassembly has no divide or
reciprocal sequence. It issues the 16 part loads in two batches and retains 16
sequential f32 additions. Repeated builds have identical executable `.text`
SHA-256 `20852a9f9e4a47779c97668c9877d7df0d0c83666987b3fb5ff744593faa4da8`;
the outer ELF digest varies because the compiler embeds the output path.

Removing the launch boundary later is compatible with fixed-order semantics if
a resident two-phase Down+Combine kernel keeps the f32 part buffer and places a
device-wide phase barrier between scatter and combine. Removing the part buffer
is not compatible with the interpreter association under the current
expert-sorted work assignment: atomic arrival changes addition order, while
token-major computation discards expert weight reuse.

## Isolated GPU gate

The T8192 oracle compared all 29,360,128 bf16 outputs with the
interpreter-order control while both optional residual and shared inputs were
present: zero mismatches.

Median of 31 HIP-event samples after five warmups, with the production-null
residual/shared contract:

| Kernel | Grid | Time |
|---|---:|---:|
| interpreter-order flat control | 256 | 2.874306 ms |
| token-row candidate | 256 | 0.386084 ms |
| token-row candidate | 512 | **0.327283 ms** |
| token-row candidate | 1024 | 0.333483 ms |
| token-row candidate | 2048 | 0.334523 ms |
| token-row candidate | 8192 | 0.330883 ms |

Grid 512, two workgroups per CU, is selected. Its absolute body time projects
to 30.110 ms over 92 layers. Relative to the current in-network 146.655 ms
Combine category, the available gain is about 116.5 ms. The standalone control
is intentionally not used for that projection because it lacks the production
interpreter's queue context; the candidate's absolute time and the current
full-network attribution are the relevant endpoints.

After splitting the candidate and control into separate shipping/test objects,
the oracle was repeated at three shapes. All outputs remained bit exact:
T8192/H3584 = 0 mismatches over 29,360,128 values, T257/H2816 = 0 over 723,712,
and T128/H4096 = 0 over 524,288. The repeated T8192 medians were 2.874664 ms for
the control and 0.393243/0.332603/0.333643/0.333323/0.330522 ms for candidate
grids 256/512/1024/2048/8192. The 0.6% reversal between grids 512 and 8192 is
noise-scale across the two folds, so the bounded `min(T,512)` resident policy
remains the opt-in runtime choice pending the three-fold promotion gate.

Promotion still requires isolated segment/runtime routing followed by a TP8
BF16 8192→1 exact network gate. The packet should isolate only a structurally
compatible `MoeCombinePf`: arbitrary nonzero `H` and `T`, `k=16`, f32
materialized part, and no part16/deterministic accumulator. The ordinary
interpreter remains the fallback.
