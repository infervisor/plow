# MI355X deterministic grouped-MoE reduction tree screen (2026-09-04)

## Decision

Reject the isolated `DOWN → COMBINE` tree for runtime promotion. The best exact-shape object saves
0.445 ms/token over 92 sites, below the predeclared 0.5 ms/token gate. It is deterministic,
spill-free, and numerically benign, but no TP8/network run is justified after the isolated miss.

The contract is model-independent: top-k is a power of two, every output element has one ordered
f32 partial per routed slot, the compiler assigns consecutive leaves and a balanced binary root,
and a 64-row tile's leaves execute on one physical XCD. The arithmetic order depends only on the
compiler layout, never on arrival order. Unsupported topology, dimensions, block placement, or
wave size trap in the probe and would fail closed in a production capability route.

## Layout and synchronization

The winning top-16 layout has eight leaves:

```text
leaf[q,h] = slot[2q,h] + slot[2q+1,h]
out[h] = (((leaf0+leaf1)+(leaf2+leaf3)) +
          ((leaf4+leaf5)+(leaf6+leaf7)))
```

One WG512 owns two consecutive slots and 64 adjacent output rows. Block placement is transposed so
all eight leaves of a row tile land on one hardware-reported XCD while each expert still reads a
contiguous 64-row weight stripe. A workgroup publishes once after its stores retire. The last of
eight arrivals invalidates locally and evaluates the fixed tree. There is no global wait, and
kernel completion is the only global publication.

This changes the rejected row-ready protocol from 16 arrivals × 448 row octets = 7,168 serialized
local atomics to 8 arrivals × 56 row tiles = 448. Leaf width four uses only 224 arrivals, but loses
too much producer parallelism.

## Isolated hardware gate

Shape: top-16, H=3584, I=384, E=896, MXFP4 weights, BF16 activation/output. Fifty-six expert
rotations stream a 1.83 GiB arena. Each result is the median of 12 complete rotation sets. The
control is the fastest ordinary grid768 DOWN followed by the seven-WG fixed-order combine.

| slots/leaf | leaves | WG | complete | control | projected gain/token |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 512 | 10.897 us | 14.099 us | 0.295 ms |
| **2** | **8** | **512** | **9.290 us** | **14.148 us** | **0.447 ms** |
| 2 | 8 | 256 | 9.640 us | 14.108 us | 0.411 ms |
| 4 | 4 | 512 | 12.107 us | 14.130 us | 0.186 ms |
| 8 | 2 | 512 | 21.445 us | 14.080 us | -0.678 ms |

The winning object is wave64, 64 VGPR, 38 SGPR, occupancy 8 waves/SIMD, zero private memory, and
zero VGPR/SGPR spills. Thirty-two identical reruns have zero output-bit differences.

The gate is effectively above this seam's measured ceiling. Ordinary DOWN alone is 8.82 us, so
even a free combine would save only about `(14.10 - 8.82) × 92 = 0.486 ms/token`. A credible
follow-up must remove another dependency edge or speed up the producer; lowering the threshold to
promote a 0.445 ms isolated projection would not close the 7.72 ms/token vLLM gap.

## Numerical and teacher-forced contract

The nonuniform oracle uses neutral E8M0 scales, deterministic nonuniform BF16 activations, and
positive normalized irregular route weights. Against the existing sequential slot order, the
balanced f32 result differs in 3,584/3,584 rows with relL2 `2.18011927e-7` and maximum absolute
error `1.1920929e-7` (one f32 ULP). All differences disappear at the required BF16 output boundary:
0/3,584 BF16 words differ. GPU tree output matches the independent CPU tree in every word.

If a wider phase passes its performance gate, quality must be evaluated in this order:

1. Repeat determinism, finite output, and the independent tree oracle at the real boundary.
2. Candidate vs fixed-order Plow on identical teacher-forced histories: no severe argmax flip,
   top-1 agreement at least 99.5%, and no row outside the established Plow repeat floor.
3. Candidate vs pinned vLLM on the same history hashes must not worsen full-row/head64 relL2,
   top-64 overlap, or stable-reference argmax agreement relative to fixed-order Plow.
4. Long-generation token agreement and task-quality parity before default-on promotion.

The pinned-vLLM captures cannot presently be used as a direct acceptance oracle for this isolated
change: fixed-order Plow itself is already 4.14–5.66× outside the conservative vLLM repeat floor.
Candidate teacher-forced capture was therefore not run after the performance gate failed. Doing so
would neither isolate this reduction nor authorize a runtime path.

## Next architecture experiment

Carry the compiler-defined leaves through the next semantic reduction instead of materializing a
standalone BF16 combine. The first screen should fuse the balanced root with the owning-rank
one-shot `XReduce` epilogue, retaining expert/slot weight locality and reconciling globally only at
the collective. Require at least two removed packet edges, ≥0.75 ms/token isolated projection,
the numerical ladder above, and a zero-spill phase object. If that cannot retain XCD ownership,
the larger redesign is persistent expert/slot workers across GLU, DOWN, tree root, and collective,
with compiler-certified resource-compatible phase objects rather than the mega interpreter.

Artifacts: `/tmp/k3_moe_tree_final.out`, `/tmp/k3_moe_tree_final.compile.log`, and
`runtime/bench/amd/k3_moe_grid_sweep.{hip,cpp}`. Final object SHA-256:
`8035d82bce90a2be1ce6153c491c658008cf26aceca50dabae1656c3861985d6`.
