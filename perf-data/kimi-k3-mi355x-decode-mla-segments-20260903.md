# Kimi K3 MI355X decode MLA segments

Date: 2026-09-03

## Change

On gfx950 AMD emission, an exact adjacent `FlashMlaDecode` +
`MlaMergeFold` pair is placed in one specialist interpreter segment. Selection
is opcode/shape based; runtime contains no model-name check. The 24 MLA blocks
therefore add 24 specialist launches while GEMVs stay grouped in 25 ordinary
segments. `PLOW_SEG_DECODE_MLA=0` is the emission rollback.

Runtime routes only a pure two-op pair. Mixed ordinary segments remain on the
ordinary interpreter, while a pure half-pair fails closed. Specialist objects
must carry the packet pairing hash.

## Static resources

Exact paired TP8 packets used the same K3 checkpoint/config, B1 decode,
ctx8192, 2,165 instructions, and 307,454 stream entries. Lean ordering and
`LdsFitSound` passed for both, and the oracle reported the same 3,723.3 us HBM
lower bound.

| object | ELF B | `.text` B | VGPR | SGPR | VGPR spill | SGPR spill | private B | LDS B | occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control main | 186096 | 174528 | 248 | 106 | 0 | 77 | 440 | 147512 | 2 |
| candidate main | 146912 | 135616 | 248 | 106 | 0 | 84 | 216 | 147512 | 2 |
| candidate MLA pair | 67392 | 57536 | 256 | 106 | 6 | 76 | 28 | 147512 | 2 |

The candidate has 49 launches per rank per decode step vs one for control.
At the measured 1.458 us ordered AQL dispatch floor, the extra 48 launches
predict 69.984 us per step. Static resources alone do not predict the measured
result: the specialist reaches 256 VGPR and spills six VGPRs.

## Exact BF16-KV TP8 qualification

Both arms were emitted under Nix with
`PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix`. Manifests report 7,650/7,650 tile
selections by measurement and digest `gfx950-76ef5b9982d04cbd`. Three
order-balanced, exclusive `gpulease -n 8`, no-trace 8192->256 runs used BF16
KV, one request, no warmup, and the same checkpoint.

| fold / order | arm | TTFT ms | TPOT ms | E2E ms |
|---|---|---:|---:|---:|
| 1, candidate -> control | candidate | 1603.061376 | 29.973490 | 9246.301446 |
| 1, candidate -> control | control | 1604.207034 | 38.104324 | 11320.809874 |
| 2, control -> candidate | control | 1604.936771 | 38.067789 | 11312.223037 |
| 2, control -> candidate | candidate | 1602.935691 | 29.993442 | 9251.263503 |
| 3, candidate -> control | candidate | 1605.182613 | 29.968896 | 9247.251322 |
| 3, candidate -> control | control | 1604.296380 | 38.052743 | 11307.746023 |

Paired candidate-minus-control TPOT deltas were -8.130834, -8.074347, and
-8.083847 ms: mean -8.096343 ms (-21.26%), sample SD 0.030246 ms. Mean paired
TTFT delta was -0.753502 ms and mean E2E delta was -2,065.320888 ms.

All six arms completed 256/256 tokens with identical arrays. The newline token
SHA-256 was
`1398465e8212d27e43a6d52e95163ae34912b72255d6d14b82d7eacdcf4d718e`
and runtime checksum was `fnv1a64:6bdfaa7b84ee4e7e`.

Qualification hashes:

- candidate/control packet: `a8f3c6ee795e1e775691fe278c6a2a964766863353c84bf40a67798748f259f6` /
  `f1406330acad952d9f1145620a63e2f860d99d809ca5c0f4d6f5bfde74b25ef7`
- candidate/control main object: `bc61a93203a5788db3e1d75fc3bc2b07488a3cbc80dd8716f2e96884532c1bb6` /
  `78f021b0711ebe9d6f8c3895db43bea51b69503af7e55ce87be9830ca4dd6b95`
- candidate MLA object: `e18258db20a706b6810cd988fde5080b6a635ab2dc7a1c535985a887a89f12e9`
- runtime: `bb12a0520f142a03ef62a102ad3d38b7b6fd85c66001fe81fa8ee609df0f5243`

Raw evidence: `/tmp/k3-decode-segment-split-gate`.

## Production gates

The manifest emits `PLOW_PACKET_HAS_DECODE_MLA_SEGMENTS`. CMake and
`scripts/build_gfx950.sh` build both specialist scheduler variants only when
the packet requires them. Both paths retain inventory pruning, marker,
packet-stamp, resource-cliff, and cleanup checks. The packet builder defaults
off, preserving legacy no-config behavior; plowc defaults on and restricts
activation to AMD gfx950.
