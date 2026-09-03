# Kimi-K3 decode inventory pruning (static gate)

Scope: exact BF16 Kimi-K3 decode inventory, compiled for gfx950. Qualified
paired assets and objects are under `/tmp/k3-inventory-qualified-65fca83`.

## Result

| object | bytes | VGPR | VGPR spill | SGPR / spill | private | wave / WG | LDS |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 412,208 | 256 | 2 | 108 / 84 | 624 B | 64 / 512 | 147,512 B |
| inventory-pruned | 186,096 | 248 | 0 | 106 / 77 | 440 B | 64 / 512 | 147,512 B |

The opt-in object preserves the one-interpreter/one-segment launch shape. The
default-off control's `.text` SHA256 is byte-identical before and after the
change: `b9dd07e09a8a34e2880937049c90256e24397570b7905566ca0314e65406a937`.

The retained-arm static builds locate the remaining register floor:

| retained arm alone | VGPR | private |
|---|---:|---:|
| FlashMlaDecode | 255 | 0 B |
| MlaMergeFold | 232 | 0 B |
| Gemv | 215 | 0 B |
| AttnRes | 134 | 0 B |
| GemvGlu | 127 | 0 B |
| MoeGroupDownFp8Blk | 113 | 0 B |
| GemvQkvg | 110 | 0 B |

Conclusion: inventory pruning deletes the existing spills but cannot improve
occupancy beyond two waves. `FlashMlaDecode` alone is already at 255 VGPR, so
the next device-side lever is reducing that retained body's live range/register
footprint (including its `n_split` path), not further dispatch-arm deletion or
adding a segment/AQL boundary.

The generated arm inventory now includes the HeadNormRope head dimension, and
the existing packet-pairing hash therefore identifies opcode variants as well
as opcodes for newly emitted assets. The loader's existing hash-stamp check
fails closed on partial, absent-manifest, or mismatched specialized objects.
The fresh control/candidate packet is byte-identical, carries pairing hash
`0x866caa2fa6a1d6a5`, and records `lean.verified=true` and `lean.oracle=true`.
It has the qualified topology: 3,394 programs and 624 segments at the 8,192
decode bucket. An earlier 949-segment/5,557-program packet was emitted with
packed-prefill segmentation and is diagnostic only.

## Exact TP8 8,192 → 256 gate

Three trace-enabled, order-balanced folds reproduced exact token IDs (768/768,
canonical JSON SHA256
`3b1345553d40748ce2baf58be3a0c20419d8662548dc3d4afa1d6ef04673a1ea`):

| metric, candidate − control | fold 1 | fold 2 | fold 3 | mean | sample SD |
|---|---:|---:|---:|---:|---:|
| TTFT ms | -1.703 | +3.656 | +5.006 | +2.320 | 3.549 |
| TPOT ms | -6.261 | -6.222 | -6.295 | -6.259 | 0.037 |
| E2E ms | -1,598.321 | -1,582.897 | -1,600.245 | -1,593.821 | 9.509 |

A separate no-trace candidate→control pair is the production-like result:

| arm | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|
| inventory-pruned | 1,602.255 | 39.430 | 11,656.810 |
| control | 1,602.697 | 44.355 | 12,913.124 |
| delta | -0.442 | **-4.925** | **-1,256.314** |

The no-trace run disproves raw tracing as the cause of the roughly 103 ms
absolute TTFT shift from the older qualified gate. It also shows that tracing
overstates this feature's TPOT gain by 1.334 ms. The paired quiet result remains
large and exact; absolute TTFT provenance is unresolved and must not be
attributed to packet topology or pruning without further evidence.
