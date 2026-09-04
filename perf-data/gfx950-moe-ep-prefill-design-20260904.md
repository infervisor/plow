# Generic routed-MoE EP-prefill boundary on gfx950

## Decision

The checkpoint/runtime memory layout does **not** block expert parallelism. The packed-expert
loader already switches from TP to EP when a graph packet declares the checkpoint's full expert
intermediate width: it packs a contiguous balanced expert range and leaves remote entries null.
The shared ownership helper now covers non-divisible expert counts as well as the 896/8 case.
The missing work is graph lowering and full-width specialist kernels.

Transport must follow graph tensor placement:

- replicated `[T,H]` input: filter the common route table locally, compute owned experts, combine
  a local f32 `[T,H]` partial, and reuse the existing reduction; no activation all-to-all;
- token-sharded input: compact one activation per `(token,destination rank)`, carry its ordered
  route entries, return one locally-combined row per `(token,source rank)`, then restore the graph's
  declared placement.

This is geometry-derived (`E`, top-k, world size, tensor placement), with no model-name predicate.

## Current exact boundary

The exact all-rank segment diagnostic at `T=8192`, top-k 16, `E=896`, `H=3584`, TP8
`I_local=384` measured:

| phase | 92-layer critical time | mean/layer |
|---|---:|---:|
| MXFP4 stage 1 | 213.543 ms | 2.321 ms |
| MXFP4 stage 2 | 70.918 ms | 0.771 ms |
| fixed-slot combine | 39.038 ms | 0.424 ms |
| **boundary** | **323.499 ms** | **3.516 ms** |

One logical layer has 131,072 routed rows. Its minimum per-rank traffic is:

| item | TP8 bytes/rank/layer | EP8 bytes/rank/layer |
|---|---:|---:|
| gate+up MXFP4 payload + E8M0 | 1.310 GB | 1.310 GB |
| down MXFP4 payload + E8M0 | 0.655 GB | 0.655 GB |
| stage-1 MXFP4 intermediate | 100.7 MB logical | 100.7 MB logical |
| stage-2 f32 route partials | 1.879 GB | 234.9 MB |
| BF16 `[T,H]` reduction message | 58.72 MB | 58.72 MB |

EP changes GEMM geometry, not FLOPs or weight bytes: stage 1 changes from all 896 experts at
`N=384` to 112 experts at `N=3072`; stage 2 changes from `K=384` to `K=3072`. It also removes
seven eighths of the route-partial stream. The existing two-shot reduction moves
`2*(7/8)*58.72 = 102.76 MB/rank/layer`; it remains required under both decompositions and must not
be claimed as an EP saving. The latest matched collective trace measured 195.638 ms across all
278 prefill collectives, but did not separately tag the 92 routed-MoE reductions.

## Peer all-to-all prototype

`plow_hsa_copy_p2p_batch` submits all independent peer SDMA copies before waiting. The isolated
8-GPU probe used 56 directed 6,473,415-byte windows, matching the expected BF16 activation window
after destination deduplication. Under the stable `/tmp/gpulease` all-eight-GPU lease:

```text
ranks=8 copies=56 chunk=6473415 bytes_per_rank=45313905
median_ms=0.147113 send_GBps_per_rank=308.021 errors=0
```

Lease `moe-ep-p2p-exact` returned rc=0 after 7 s; the post-run audit found no foreign process. For
uniform independent top-k, the expected number of remote destination ranks per token is
`7*(1-(7/8)^16)=6.1735`. Dispatch + return + 16-byte `(token,slot,expert,gate)` metadata is
90.86 MB/rank/layer, or 0.2950 ms at the measured rate and 27.14 ms over 92 layers.

The acceptance target is `max(15%,35 ms)`: 323.499 ms must fall to at most 288.499 ms. A
token-sharded implementation therefore has 261.36 ms left for route/sort + both expert stages +
combine after transport, requiring at least 23.10 ms (8.12%) off the current 284.461 ms expert
bodies.
bodies. The replicated-input path pays no new fabric charge; reducing f32 route-partial traffic
alone has a 34.16 ms proportional ceiling, just short of the gate without a measured GEMM gain.

## Implemented ABI and remaining integration

`packet::moe_ep` now provides:

- balanced contiguous ownership for non-divisible `E/ranks` geometries;
- a stable 64-byte boundary descriptor and 40-byte peer-window descriptor;
- exact token-sharded window planning that deduplicates activations per destination while retaining
  ordered route entries;
- a cost gate that charges measured fabric time before exposing the compute/sort budget.

The smallest production integration is:

1. Graph lowering detects a routed-MoE boundary whose input placement is replicated and whose
   packet declares full `I`; it must not key on a model/config name.
2. Emit owned-expert align metadata (`E_local=ceil/floor(E/world)`), full-I stage-1/stage-2 phase
   objects, and a local `[T,H]` f32 accumulator. Keep decode on TP.
3. Compile the phase objects at wave64 and reject any private segment or VGPR/SGPR spill. Include
   route/sort, MXFP4 activation quantization, both GEMMs, local combine, and the existing reduction
   in the timed gate.
4. Require route ids, gates, copied MXFP4 payloads, and E8M0 scales to be exact. Full-I changes
   reduction association, so compare output with finite checks, relative RMS `<=5e-3`, max absolute
   error `<=6.25e-2`, then run the existing teacher-forced/logit and long-generation gates. Do not
   label reassociated output bit-exact.

No graph integration is promoted yet: the transport primitive passes, but a full-I wave64,
zero-spill specialist has not been built and the required whole-boundary gain has not been
measured. The next experiment is an isolated owned-112-expert full-I pair using the existing packed
loader, compared against eight TP partials with identical route tables and cache flushing.
