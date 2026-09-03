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

`PLOW_XR2_GATHER=1` emits a two-shot packet when the gather is a complete column
partition (`row_w = n_gpu*gcols`). Phase 1 is the existing reduce-scatter. Phase 2 adds
the owner-derived compact gather value while assembling each reduced slice. It adds no
packet, removes no rendezvous, and preserves the existing reduction-to-bf16 rounding
before the gather add. Decode and unsupported shapes remain one-shot. Default is off.

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

## Full-network gate

Same `vllm bench serve` client and `/v1/completions` contract: random exact 8192 input
tokens, C1, one output token, one request per independent server process. The coherence
probe passed before both measurements; both cells completed 1/1 requests and generated
exactly 1/1 requested tokens.

| arm | TTFT |
|---|---:|
| control | 2980.28 ms |
| `PLOW_XR2_GATHER=1` | 2739.65 ms |

Result: -240.63 ms (-8.07%). Logs:
`/tmp/plowrt_bench_xr2g_control_8134.log` and
`/tmp/plowrt_bench_xr2g_one_8134.log`. Candidate asset:
`/tmp/plow-k3-xr2g-16k.xxjPEc`.

This first full-network fold validates the expected direction and magnitude but is not
the publication gate. Keep the emit default off until at least three order-alternated
full-network folds pass coherence, exact output-count, counter audit, and token parity.
