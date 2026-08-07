# VMM lazy-commit weight slab — GH200, gemma-4 12B bf16 (2026-08-06)

Follow-up to `coldstart-plow-vs-vllm-gh200.md` §4b, which recorded that the
weight slab's single `cuMemAlloc` pays the driver's ~13 GiB/s page-commit rate
upfront (1.74 s at 12B, 4.5 s at 31B) and that "killing the term needs the
commit avoided rather than batched: VMM with lazy mapping". This is that
patch, measured.

Setup: `plowrt serve`, `PLOW_LOAD_PROFILE=1`, assets `gh200-base/12b-bf16`
(22.18 GiB weights + 0.75 GiB KV, slab 23.22 GiB, B=1 ctx 8192), same binary
built from the branch with only the slab change toggled via `PLOW_WEIGHT_VMM`.

## Warm page cache (restart / S1 model switch)

| | flat cuMemAlloc slab | VMM lazy-commit slab |
|---|---|---|
| slab bringup | 1716–1744 ms | **0.1 ms** (reserve + mapper spawn) |
| upload blocked on commit | — (serial, upfront) | **0.1 ms** (watermark waits) |
| slab tail commit join | — | 0.0 ms |
| upload_all wall | 3550 ms | 1827 ms |
| **total `GpuEngine::load`** | **3670–3696 ms** | **1986–2001 ms (−46 %)** |
| effective upload rate | 6.2 GiB/s | 12.0 GiB/s |

The commit still happens at the same ~13 GiB/s — a background `wslab-map`
thread `cuMemCreate`+`cuMemMap`+`cuMemSetAccess`es 256 MiB chunks front-to-back
while the upload writes behind it. The upload's only exposure is the watermark
wait, measured at 0.1 ms total across 356 chunks.

## Cold page cache (echo 3 > drop_caches)

| | flat slab | VMM slab |
|---|---|---|
| Disk→RAM wall | 4581 ms @ 4.84 GiB/s | 4633 ms @ 4.79 GiB/s |
| total load | 4724 ms | 4792 ms |

**A wash, as expected.** Cold, the load is I/O-bound at ~4.8 GiB/s: the
baseline's alloc stall was already hidden behind the prefetcher's populate
(spawned before the alloc), so removing it moves nothing — the same seconds
are just spent waiting on the disk instead of the driver. The win is every
load whose I/O beats ~13 GiB/s: warm restarts, S1 model switches, page-cache
hits.

## Correctness / lifecycle

- Greedy 48-token completion byte-identical between flat and VMM builds.
- `PLOW_WEIGHT_VMM=0` fallback reproduces the flat numbers exactly.
- `gpu_lifecycle::load_serve_unload_reload_cycle` passes: VRAM returns to
  baseline to the MiB across two load/unload cycles (VmmSlab Drop unmaps,
  releases, address-frees; no leak, no double-free).
- Planner footprint identical: used 23.238 GiB, overhead 19 MiB, both builds.

## Round 2: direct-from-mmap upload + physical-handle pool (2026-08-07)

The staging memcpy was the next wall (1688 ms at 13.2 GiB/s, one thread).
`examples/h2d_bench.rs` on this box:

| path | rate |
|---|---|
| staged 2×64 MiB pinned double-buffer | 13.2 GiB/s |
| **direct `cuMemcpyHtoDAsync` from the warm mmap** | **332 GiB/s** |
| parallel mmap→heap memcpy, 16 threads | 118 GiB/s |
| VMM commit, 1–8 threads | 13.4 GiB/s flat (does NOT parallelize) |
| commit phase split (8 GiB) | create 595 ms, map 0.2 ms, set_access 2 ms |

Grace-Hopper ATS (device attribute 100) lets the DMA engines read pageable
file-backed memory at link speed — so the loader now skips staging entirely
on coherent parts (`Backend::coherent_host_dma`, `PLOW_UPLOAD_DIRECT=0`
reverts; staged pipe remains the path everywhere else and for generated
tensors). That exposed `cuMemCreate` as the entire remaining cost — serial,
thread-invariant, ~13 GiB/s — which only reuse can kill: `PLOW_SLAB_KEEP=1`
parks a dropped slab's physical chunks in the backend pool and the next load
re-maps them (map+set_access ≈ free).

Measured, gemma-4 12B bf16:

| | warm | cold | reload |
|---|---|---|---|
| flat slab + staged (pre-branch) | 3.69 s | 4.72 s | 3.69 s |
| VMM slab, staged copy | 1.99 s | 4.79 s | 1.94 s |
| + direct copy (defaults now) | **1.88 s** | **4.69 s** | 1.94 s |
| + `PLOW_SLAB_KEEP=1` | — | — | **0.38 s** |

Reload correctness under reuse (pooled chunks are NOT zeroed):
`examples/reload_bench.rs` asserts the reused-chunk rounds greedy-decode
byte-identically to the fresh-commit round — they do, because every slab byte
is uploaded, memset, or KV (write-before-read). `PLOW_SLAB_KEEP` stays
**opt-in**: pooled chunks hold VRAM between loads and the serve planner's
free-VRAM arithmetic does not credit them yet.

## What the load is now (warm, 1879 ms, defaults)

`cuMemCreate` commit 1523 ms (the `slab commit wait`) ≈ the whole wall; blob
parse 128 ms; iter/lookup ~130 ms. The copy is free (73 ms DMA overlay). The
remaining levers: planner-aware slab retention by default (turns every warm
load into the 0.38 s shape), or a driver that commits faster.

## Context vs vLLM

vLLM 0.26.0 on this box: 97–103 s to serving at 12B (§1 of the coldstart doc).
plowrt with this patch: ~2.0 s warm / ~4.8 s cold to engine-loaded.

## MI300X addendum (2026-08-07, measured on the gfx942 branch)

The opt-in AMD arm (`PLOW_WEIGHT_VMM=1`) was measured on MI300X, Gemma-4-12B
fp8 (14.6 GiB slab), warm page cache, two reps each:

  flat slab (default)   engine load 3.61 / 3.59 s   named-tensors 2.20 s
  VMM slab  (opt-in)    engine load 4.12 / 4.23 s   named-tensors 2.65 / 2.73 s

+15% — the GH200 win came from taking cuMemAlloc's page-commit stall off the
load path, and HSA's flat allocation has no such stall to hide, so the chunked
mapping is pure overhead here. The opt-in default for AMD is therefore correct
and stays; do not flip it without new hardware evidence. `coherent_host_dma`
correctly answers false on discrete CDNA parts, so the direct-from-mmap upload
is inert on this box.
