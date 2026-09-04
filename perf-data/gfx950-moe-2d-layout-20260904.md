# Single-resident 2D routed-expert layout on gfx950

## Decision

Prototype **EP2 x intra-expert-TP4**. It is the best all-metric Pareto point of the
single-resident factorizations tested. Keep production promotion fail-closed until the compiler
rewrites decode and prefill to the same table and an eight-rank compound tolerance gate passes.

The layout is graph-derived. For world size `W`, choose `EP * TPexpert = W`. Rank
`r` owns expert group `r / TPexpert` and intermediate slice `r % TPexpert`. Replicated routed-FFN
input and route tables need no dispatch. The existing all-rank reduction is sufficient: it sums
the `TPexpert` down-projection partials inside every expert group and the disjoint expert groups at
the same time. No subgroup collective or new final combine is required.

## Exact resident bytes

For 92 layers, `E=896`, `H=3584`, full `I=3072`, MXFP4 payload plus E8M0 scales:

| layout | owned experts | local I | primary B/rank | shuffled-down B/rank | total GiB/rank | additional vs canonical |
|---|---:|---:|---:|---:|---:|---:|
| TP8 | 896 | 384 | 180,807,008,256 | 61,450,747,904 | 225.620 | 0 |
| EP2 x TP4 | 448 | 768 | 180,807,008,256 | 60,269,002,752 | 224.520 | 0 |
| EP4 x TP2 | 224 | 1536 | 180,807,008,256 | 60,269,002,752 | 224.520 | 0 |
| EP8 x TP1 | 112 | 3072 | 180,807,008,256 | 60,269,002,752 | 224.520 | 0 |

The 1.100 GiB TP8 difference is only pad256x8 waste in its shuffled E8M0 scale view. A canonical
2D table replaces the current TP table; it does not add the 224.520 GiB EP companion. Gate/up and
down cannot be address views across factorizations, so phase remapping would move 7/8 of the
168.390 GiB primary set per transition and is rejected.

`packet::moe_ep::Moe2dLayout` is the checked ownership/slice and byte-accounting contract.
`layout_tradeoff.py` computes the exact multivariate-hypergeometric route-tail distribution, with
no Monte Carlo or model-name predicate.

## Decode route tail and kernel result

Top-16 experts are sampled without replacement from 896 balanced experts. `max routes` is the
critical expert group; every rank in that group performs the same routes over its intra-expert
slice.

| layout | max routes mean | P50 | P95 | GLU+down P50 us/layer | P95 us/layer | P50 x92 delta vs TP8 |
|---|---:|---:|---:|---:|---:|---:|
| TP8 | 16.000 | 16 | 16 | 26.17 | 26.17 | baseline |
| EP2 x TP4 | 9.557 | 9 | 12 | 30.47 | 38.21 | +0.396 ms/token |
| EP4 x TP2 | 6.118 | 6 | 8 | 32.69 | 38.99 | +0.600 ms/token |
| EP8 x TP1 | 4.176 | 4 | 6 | not promoted | not promoted | — |

The exact-shape probe rotates a 1.97 GB resident expert arena, leaves remote pointer-table entries
null, and runs the shipping grouped MXFP4 decode bodies. Reference streaming was 4.07-4.10 TB/s.
Old vs current GLU and down outputs are byte-identical after null slots are initialized equally.
EP2 dominates EP4 for decode and limits the expected tail to 1.195x the ideal equal-work body.

## Prefill isolated projection

All cells use `T=8192`, `H=3584`, global top-k 16 and identical aggregate expert work. Stage-1
payload and E8M0 output were exact. Stage-2 checked 114,688 values with zero error. Every compiled
object is wave64 with zero private bytes and zero VGPR/SGPR spills.

| layout | filter ms | reusable-A4 stage 1 ms | stage 2 ms | fixed-slot combine ms | boundary ms | x92 saving vs TP8 |
|---|---:|---:|---:|---:|---:|---:|
| TP8 | 0.211121 | 1.728292 | 0.846248 | 0.424331 | 3.209992 | baseline |
| EP2 x TP4 | 0.176400 | 1.441531 | 0.572225 | 0.328482 | 2.518638 | 63.605 ms |
| EP4 x TP2 | 0.130720 | 1.213570 | 0.494664 | 0.221521 | 2.060475 | 105.756 ms |
| EP8 x TP1 | 0.085081 | 1.056448 | 0.479004 | 0.154281 | 1.774814 | 132.036 ms |

EP2 gives up 42.151 ms of projected TTFT saving vs EP4 but avoids another 0.204 ms/token median
decode regression. Over 1024 output tokens that decode difference is about 209 ms, so EP2 is the
better end-to-end factorization.

## Production gate

The prototype does not change a runtime default. Promotion requires:

1. Emit one canonical table name for both decode and prefill; encode `EP` and `TPexpert` in the
   packet rather than inferring them from a model.
2. Loader owns `E/EP` experts and applies the ordinary row/column shard with rank
   `r % TPexpert`; reject any declared local-I mismatch before allocation.
3. Compile EP2 stage-2 as its own phase object (`I=768`): 120 VGPR, 40 SGPR, wave64, no spills.
4. Run the matched TP8 compound boundary and teacher-forced/logit tolerances. The global reduction
   is algebraically sufficient but re-associates f32 partials, so bit identity is not claimed.
5. Require decode GLU+down to recover the measured +0.396 ms/token before enabling by default.

Phase remap, remote peer reads, and dual tables are rejected. Their minimum transferred or
resident bytes exceed the measured prefill saving.
