# Kimi-K3 MI355X combined BF16 prefill gate

Date: 2026-09-03. Hardware: TP8 MI355X. Shape: exact random 8192-token
prompt, one greedy output token, C1, no warmup, BF16 KV cache.

## Contract correction

The earlier standalone P1 and P3 full-network cells used packets whose manifests
record `precision.kv_enc=fp8`. They remain useful FP8-KV lever screens, but are
not comparable to the live vLLM BF16-KV baseline. This gate regenerates both
arms with `precision.kv_enc=bf16` and is the first coherent BF16 network
measurement of P1 + P2 + P3 together.

## Arms

Both arms use the same checkpoint, runtime binary, measured TuneDB profile
(7650/7650 lookups), GQ scheduler, exact compact TP audit, and
`--amd-kda-family-route=true`. The control uses its packet-matched ordinary KDA
family object; the candidate uses its packet-matched q-precompute family object.
This avoids attributing mega-interpreter spill isolation to q-precompute.

| arm | emit flags | packet structure |
|---|---|---|
| control | `PLOW_FP8_KV=0 PLOW_MOE_STAGE2_LEAN=0 PLOW_XR2_GATHER=0 PLOW_KDA_CHUNK_QPRE=0` | 49 ordered segments; 92 one-shot folded gathers |
| candidate | `PLOW_FP8_KV=0 PLOW_MOE_STAGE2_LEAN=1 PLOW_XR2_GATHER=1 PLOW_KDA_CHUNK_QPRE=1` | 233 ordered segments; 92 deterministic lean Down segments; 278 two-shot reductions |

Both runtime arms also set `PLOW_MLA_PF_V2=false`,
`PLOW_PACKED_PREFILL_ROUTE=false`, and leave P4/P5 off. The intended A/B
difference is the combined P1/P2/P3 packet and its exact-capability objects,
not a model-specific runtime heuristic.

## Three order-alternated folds

| fold / order | control | P1+P2+P3 | delta | exact parity |
|---|---:|---:|---:|---|
| 1, candidate -> control | 3055.381 ms | 2063.032 ms | -992.349 ms (-32.479%) | pass |
| 2, control -> candidate | 3053.885 ms | 2064.388 ms | -989.497 ms (-32.401%) | pass |
| 3, candidate -> control | 3056.855 ms | 2065.506 ms | -991.349 ms (-32.430%) | pass |
| mean | 3055.374 ms | 2064.309 ms | -991.065 ms (-32.435%) | pass |

All six processes completed 1/1 requests with zero failures, no trace overflow,
all-rank prefill completion, and every-dispatch counter audit. Every arm used
the identical prompt token array (SHA-256
`dd51e931308683300f372862a682d56a332e0e3e8d71cd70ef06c34739362dcd`)
and produced token 6896 with checksum `fnv1a64:7d749e3b002fafa7`.

The first candidate is the separately leased clean smoke at
`/tmp/k3-combined-bf16-smoke-candidate.{json,log,trace}`. It is fold 1's
candidate arm, not an extra sample.

## Trace attribution

The table is the mean of all three raw traces per arm. Times are critical-
envelope body time, not a sum of overlapping HIP events.

| category | control | P1+P2+P3 | delta |
|---|---:|---:|---:|
| `MoeGroupDownPf` in mega interpreter | 700.552 ms | 0 ms traced | -700.552 ms traced |
| trace residual, including 92 standalone Down segments | 3.316 ms | 84.241 ms | +80.925 ms |
| `XReduce` + `XReduce2` | 449.566 ms | 217.618 ms | -231.947 ms |
| `KdaChunkCarry` | 283.402 ms | 156.445 ms | -126.957 ms |
| `KdaChunkWu` | 40.575 ms | 43.220 ms | +2.645 ms |
| `KdaChunkIntra` | 124.514 ms | 122.155 ms | -2.358 ms |
| full trace span | 2879.373 ms | 1888.242 ms | -991.131 ms |

The standalone P1 object does not write `PlowTraceRec`, so its 92 packets
disappear from the opcode table and mostly reappear in the candidate's 84.241
ms trace residual. Conservatively pairing that residual increase with the
removed interpreter body gives about 619.6 ms of P1 network gain. P2 removes
about 231.9 ms from the collective family, and P3's carry/WU pair removes about
124.3 ms. These are attribution estimates inside the combined schedule, not
independent promotion folds.

The trace span agrees with the paired endpoint delta to 0.066 ms. Endpoint
minus trace span is about 176 ms in both arms, so the comparison has no new
host-side imbalance.

## Objects and artifacts

- Deterministic stage 2: wave64, 98 VGPR, 40 SGPR, zero private/spill,
  occupancy 4.
- KDA family: wave64, 248 VGPR, zero VGPR spill; control 444 B/thread private,
  candidate 440 B/thread private. This still fails the zero-private promotion
  target.
- Weight slab: 29207 MiB/rank control vs 29211 MiB/rank candidate before the
  separately allocated shuffled stage-2 companion. The companion cost remains
  637 MiB/rank as recorded in the P1 report.

| artifact | control | candidate |
|---|---|---|
| packet SHA-256 | `449c8c99397efda3093cb801d741ed2c99984b3cd75551c10afb27a2c560e6b7` | `29e1dbedb2bf344d69aad41569af60d4f5e37d8a7ad908fa52ab3c4125ad57ce` |
| build manifest SHA-256 | `89ac14b4d8b1b48f980df1862bb52ed5d32f7cc7a1e3353293019d5484d2ebc6` | `59005e518fa0d0998c49093d0cfebf8209503e73a1df07f27a35809649e8f9a0` |
| pairing hash | `0x5232e4c34e6ad4f0` | `0x00c8f138072a14b4` |
| KDA-family GQ object SHA-256 | `704f4d2048d98e1f04b42bc9ee79f857ec89a5add3bca64a4289f96af3f95342` | `3c68b6513e1180b4f984add84af558356d80d574ba81cc7ccc2370485663c87b` |
| deterministic stage-2 object SHA-256 | `865e9656aeecb789dc78ef68d8d3763742424e198a5f7f0a428cc1367f34cd0a` | `82c3bb065157bea82a2d7a278d4430bff45e34cf61dbfb92ba0b9117dddab886` |

Runtime SHA-256 is
`4cd4c2aa45c100d227ef77a8fdd88a7d5906b762c0bcd67a61b9792f21339a81`.
Raw JSON/log/trace/report files are
`/tmp/k3-combined-bf16-{control,candidate}-{1,2,3}.*`, with fold-1 candidate
using the smoke path named above.

## Decision

The combined BF16 candidate is stable and exact, but it is not a vLLM win:
2064.309 ms mean TTFT vs the current vLLM median 568.35 ms = 3.63x slower,
leaving about 1.50 s. Keep P1 and P3 default-off. This combined gate cannot
replace isolated three-fold promotion evidence for either lever, and the KDA
family still carries private memory. P2 remains independently promoted from
its own exact three-fold gate.
