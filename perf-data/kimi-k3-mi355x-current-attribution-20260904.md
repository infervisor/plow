# MI355X TP8 current-default attribution — 2026-09-04

## Result

This is the qualified BF16-KV `T=8192` graph after promotion of the generic gfx950
KDA-Intra wave-item specialist. C8, Nv0, materialized MLA, and XReduce-wave experiments are off;
decode MLA inventory pruning, raw decode MLA segments, and per-XCD hierarchy are at their current
defaults. Both runs used the measured TuneDB (7,650/7,650 measured lookups), packet SHA256
`f1bf783dac96791b7116ffb549862c8206ba33351310c7c113504916611e8921`, and paired specialist
SHA256 `af59ce16918b56b7882394aed891c00c0ec49149a0b2ce86cde662e04e248d45`.

| run | endpoint | device critical envelope | untraced host/other |
|---|---:|---:|---:|
| prefill, 8192→1 | TTFT 1413.848 ms | 1400.078 ms | 13.770 ms |
| decode, 8192→256 | TPOT 30.0319 ms | 28.0914 ms | 1.9405 ms/token |

The exact 8192→256 run completed all 256 tokens with checksum
`fnv1a64:6bdfaa7b84ee4e7e`; its first token matches the 8192→1 run, whose checksum is
`fnv1a64:7d749e3b002fafa7`. TTFT/E2E for the traced full run were 1414.483/9072.620 ms.

## Prefill critical envelope

The primary interpreter accounts for 1,021.926 ms. The 378.151 ms residual is time between its
traced packets, principally separately launched lean MoE and KDA-Intra segment objects; it must
not be assigned to the adjacent interpreter opcode without a segment timestamp.

| category | critical ms | share of 1400.078 ms |
|---|---:|---:|
| external segment/residual | 378.151 | 27.01% |
| dense GEMM/GEMV | 242.928 | 17.35% |
| traced KDA excluding external Intra | 237.604 | 16.97% |
| TP reductions (`XReduce2`) | 210.216 | 15.01% |
| MLA attention/merge/gate | 145.245 | 10.37% |
| routing/norm/other interpreter ops | 97.241 | 6.95% |
| AttnRes | 88.692 | 6.34% |

Largest traced op families are `XReduce2` 210.216 ms, `KdaChunkCarry` 150.043 ms,
`GemmWide` 146.187 ms, `FlashMlaPrefill` 108.653 ms, `AttnRes` 88.692 ms, and `GemmC5`
83.066 ms. The wave-item specialist removes `KdaChunkIntra` from the primary trace; the prior
122 ms interpreter charge is therefore absent rather than silently zero-cost.

## Decode critical envelope

The reporter accounts for 27.8444 ms of the 28.0914 ms device span; its residual is 0.2470 ms.

| category | critical ms | share of accounted envelope |
|---|---:|---:|
| dense GEMV families | 11.337 | 40.72% |
| MoE experts/router/combine | 5.394 | 19.37% |
| TP reductions | 4.594 | 16.50% |
| AttnRes | 3.288 | 11.81% |
| KDA | 1.656 | 5.95% |
| MLA | 1.546 | 5.55% |
| other | 0.030 | 0.11% |

The single largest decoded family is 468 `b=256` GEMVs at 5.905 ms, followed by AttnRes
3.288 ms and the two XReduce widths at 4.594 ms combined. Interpreter protocol gate is
2.527 ms total; the remaining endpoint/device difference is larger than the raw-trace residual.

## Priority

1. `XReduce2`: 210.216 ms prefill and 4.594 ms decode reductions are the largest fully
   attributed common family. Gate a generic wave/workgroup specialist against exact IDs and the
   full-network envelope, not only isolated bandwidth.
2. `KdaChunkCarry`: 150.043 ms across 69 calls (2.174 ms/call) is the next largest exact traced
   kernel. Preserve FP32 recurrence/order and prove state plus 256-token equality before routing.

Before optimizing the 378.151 ms external bucket, add per-segment device timestamps so lean MoE,
KDA-Intra body, launch gaps, and overlap are separately measured. Its size alone is not evidence
for which external object to change.

## Artifacts

- Prefill trace/report SHA256: `2fc6a13ad814685ffb9a4de61d2cb075e5f6da7eb8a2885eec62aadacbba5e86` /
  `1419d0fdec2dc2979caa5bd2a38cc57b65214a1ad427bfa57e715e99a8e29875`.
- Decode trace/report SHA256: `d25a8f85881d284656d0ea20bb783b1862dabc1361d04954f915d27405fc7790` /
  `c37dbb71bacdb42ecb0012f248f641b013b1a239c1c3c97a8e004e84a7247e96`.
- Prefill/decode JSON SHA256: `4154c0b1772de3ae1c7a324763e9b6c4f2f0f64187f11c3b70a6452d9983a497` /
  `69f88765fca9464ae578bc02e114cbc356308918fca901f0697457d272c02dab`.
- Exclusive TP8 lease `kda-current-trace`: rc=0, held 197 s, no overlapping lease.
