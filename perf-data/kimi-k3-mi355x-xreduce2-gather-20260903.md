# MI355X TP8 folded-gather XReduce gate — 2026-09-03

## Attribution

`/tmp/k3-best-fp8-stage2-trace.bin`, exact T=8192:

| opcode | packets | geometry | gate | body |
|---|---:|---|---:|---:|
| XReduce | 92 | 256 WG, 58,720,256 bf16 | 0.328 ms | 319.208 ms |
| XReduceTwoShot | 186 | 256 WG, 29,360,128 or 58,720,256 bf16 | 0.418 ms | 104.229 ms |

The gate is 0.18% of the combined 424.18 ms. The remaining one-shot packets are the
folded-gather form: reduce one full `[T,H]` partial, then add the owner rank's compact
column shard. The one-shot reads every rank's full reduction partial on every rank.

Local vLLM/AITER sources use two-shot reduce-scatter/all-gather for large messages:
256-thread tiles, 16-byte atoms, and up to 1,216 workgroups. The applicable idea is the
topology, not their optional lossy codecs. Plow's existing two-shot preserves rank-order
f32 accumulation and the bf16 boundary.

## Change

The default route emits a two-shot packet when the gather is a complete column
partition (`row_w = n_gpu*gcols`). Phase 1 is the existing reduce-scatter. Phase 2 adds
the owner-derived compact gather value while assembling each reduced slice. It adds no
packet, removes no rendezvous, and preserves the existing reduction-to-bf16 rounding
before the gather add. Decode and unsupported shapes remain one-shot.
`PLOW_XR2_GATHER=0` restores the one-shot rollback.

At K3 TP8/T=8192 the emit changes 92 XReduce packets to XReduceTwoShot: 278 two-shot,
92 carrying `i6=gslot,i7=gcols`, and zero one-shot packets. Both rendezvous gates remain
unique.

## Focused TP8 gate

Command shape: `TP_ROWS=8192 TP_ITERS=1 TP_GATHER=1`, eight MI355X ranks, 256x512
threads. The oracle checks all 58,720,256 outputs on all eight ranks against the exact
two-packet bf16 boundary.

| arm | us/collective | parity | timeout |
|---|---:|---|---|
| one-shot folded gather | 330.068 | PASS, all ranks | no |
| two-shot folded gather | 94.424 | PASS, all ranks | no |

Result: 3.50x faster, -235.644 us (-71.4%). The production `PLOW_XR_AGG=1` protocol
was enabled in both arms. The focused gfx950 object is wave64, private segment 0,
SGPR/VGPR spills 0, and 26 VGPR for the two-shot gather wrapper.

## Full-network promotion gate

Same `vllm bench serve` client and `/v1/completions` contract: random exact 8192 input
tokens, C1, one output token, one request per independent server process. The coherence
probe passed before both measurements; both cells completed 1/1 requests and generated
exactly 1/1 requested tokens.

| fold/order | one-shot control | folded gather | delta | parity |
|---|---:|---:|---:|---|
| 1, candidate→control | 2980.28 ms | 2739.65 ms | -240.63 ms (-8.07%) | coherence and exact output count |
| 2, control→candidate | 3070.337776 ms | 2719.590423 ms | -350.747353 ms (-11.42%) | exact token/checksum |
| 3, candidate→control | 2949.428766 ms | 2718.925641 ms | -230.503125 ms (-7.82%) | exact token/checksum |

Median paired delta is -240.63 ms (-8.07%). All six processes completed 1/1 requests
without a timeout. Folds 2 and 3 used the identical random prompt (token-array SHA-256
`dd51e931308683300f372862a682d56a332e0e3e8d71cd70ef06c34739362dcd`)
and produced token 9618 with checksum `fnv1a64:499ccc4012ebcff0` in both arms.

Static disassembly confirms route selection at T8192: the candidate has 278
`XReduceTwoShot` packets, including 92 with folded-gather metadata, and zero one-shot
`XReduce`; the control has 186 two-shot packets, 92 one-shot folded gathers, and zero
two-shot packets carrying gather metadata.

Candidate packet SHA-256 is
`daffdf09cda7c983da70c6a48022543b2206dbaa6d9b8263b64ba0064690f789`;
control packet SHA-256 is
`7891d912f5e0efdafe77f20a8b241cc2a146d585fba9660818c371b756c6bea3`.
Their stamped matching prefill objects are respectively
`68346c9af293436d3f0475f54f81c3ab166c9d64d3f2b1c5e67e5628a93ecc9e`
and `f56730fd332b64643be3d43a6f038de482f19d3a58968237eb171ca4527cb38f`.
The runtime correctly rejects swapping these packet/object pairs because their
capability stamps differ.

Fold 1 logs are `/tmp/plowrt_bench_xr2g_{one,control}_8134.log`. Fold 2 and 3 raw
bench evidence is `/tmp/xr2-fold{2,3}-{candidate,control}.{json,log}`. Candidate asset:
`/tmp/plow-k3-xr2g-16k.xxjPEc`; control asset:
`/tmp/plow-k3-fp8kv-49seg-16k.dUY4dr`.

Decision: promote the generic structural route to the emit default. The shape predicate,
decode exclusion, two rendezvous gates, rank-order f32 accumulation, and bf16 boundary
remain unchanged. `PLOW_XR2_GATHER=0` is the explicit rollback.
