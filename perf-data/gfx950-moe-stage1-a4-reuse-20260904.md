# gfx950 reusable-A4 MoE stage-1 gate — 2026-09-04

## Contract and storage

- Generic route: MXFP4/E8M0 stage-1, K divisible by 128, and at least two 256-column shipping
  N tiles. No model-name predicate.
- Existing row-major expert weight/scale pointer tables are read directly. An abandoned
  preshuffled design required 1,336 MB/rank of additional gate/up companions and was not kept.
- Scratch is one max-live allocation shared by sequential prefill segments:
  `row_capacity * (hidden / 2 + hidden / 32)` bytes. The emitted T8192 prefill tensor is exactly
  979,456 B = 244,864 rows, so runtime logs exactly 466,221,056 B/rank (444.62 MiB). Its compiler
  bound is `(topk * T + experts * 127) * 1,904` at this geometry: 30,464 B per additional input
  token plus a 216,659,968 B expert-padding constant. The isolated distribution used 151,232
  actual padded rows, exactly 287,945,728 B (274.61 MiB).
- Decode does not route through these objects. Rollback is `PLOW_MOE_STAGE1_A4_REUSE=0`.

## Exact boundary accounting

Geometry: T8192, H3584, I384, E896, top-k16, 151,232 BM64-padded sorted rows.

| Item | Shipping | Reusable A4 |
|---|---:|---:|
| BF16 activation reads | 2,818,572,288 B | 939,524,096 B |
| A4 payload + scale writes | 0 | 287,945,728 B |
| A4 payload + scale reads | 0 | 863,837,184 B |
| activation-side total | 2,818,572,288 B | 2,091,307,008 B |
| launches | 1 | 2 |

The candidate removes 727,265,280 B of activation-side traffic. Expert-weight traffic is
unchanged because both arms read the same row-major tables.

## Isolated result

One exclusive MI355X lease, 31 order-alternated samples, and a 256 MiB compute cache flush before
each timed boundary:

| Arm | Mean | Median |
|---|---:|---:|
| shipping | 2.122398 ms | 2.119457 ms |
| quant/sort + reusable-A4 GEMM | 1.728507 ms | 1.726974 ms |

Whole-boundary gain = 0.393891 ms, or 18.55%. The oracle found zero differing bytes over
29,036,544 output payload bytes and 1,814,784 output scale bytes.

Static object gate: wave64; WG256/four waves; VGPR=182; SGPR=44; occupancy=2; private=0;
VGPR spills=0; SGPR spills=0. The quant/sort kernel is also private/spill-free.

## TP8 endpoint

The publication gate used packet SHA-256
`f1bf783dac96791b7116ffb549862c8206ba33351310c7c113504916611e8921` and the clean combined-
default object stack containing `plow_xr_rs_u2` and `plow_mla_pf_tr16_arm_1`. A full directory
hash diff proved that only `moe_stage1_mxfp4_gfx950.elf` differed. Two order-reversed knob-only
8192-to-1 pairs used the same binary, packet, objects, prompt, and environment:

| Order | Current default | + reusable A4 | Delta |
|---|---:|---:|---:|
| control, candidate | 1,367.597 ms | 1,279.219 ms | -88.379 ms |
| candidate, control | 1,369.737 ms | 1,282.105 ms | -87.632 ms |
| mean | 1,368.667 ms | 1,280.662 ms | -88.006 ms (-6.430%) |

Every arm completed with no failure and emitted the identical token id 6896.

Two earlier order-reversed diagnostic pairs also reproduced an 88.105 ms average delta, but
their interpreter objects predated the combined RS-U2/TR16 defaults. They are not used for the
promotion claim.

An 8192-to-256 reusable-A4 cell on the same current-default stack completed with
TTFT=1,282.044 ms and TPOT=30.036 ms. Its full
256-token stream has SHA-256 `3b1345553d40748ce2baf58be3a0c20419d8662548dc3d4afa1d6ef04673a1ea`,
identical to the rollback control. The route exists only in the prefill-program branch; decode
objects and routes are unchanged.
