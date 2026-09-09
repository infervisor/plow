# Token-batch readiness on main, 2026-09-08

Base: `8d98edcba5cf1650cbce89536482b6522bbb218c`, with the local readiness changes.
Device: NVIDIA H100 80GB HBM3. No AMD device was used for this review.

Token-batch selection defaults on, with `--token-batch=false` / `PLOW_TOKEN_BATCH=0`
rollback. This is capability-gated selection, not production qualification of every backend.

| Check | Result |
|---|---|
| CUDA executor | Missing. Startup reports unavailable and ordinary execution continues. |
| AMD eligibility | gfx942, multiple slots, no tensor parallelism or prefix-cache mode; explicit fusion wins. |
| AMD program/object | Existing BF16 dense opcode, object-marker, unsplit-attention and fused-epilogue gates retained. |
| Failed AMD kernel lookup | Module now owned by the cleanup guard before lookup. |
| Startup observability | `armed` and `ready` are separate; `fires=false` until a successful device dispatch. |
| Post-dispatch validation | Invalid token IDs or failed frontier commit are device errors; no ordinary-path retry. |
| Host verification | CUDA + HSA library suite: 594 passed, 16 ignored, zero failures. |
| Shared contract and CPU integration | 18 asset-contract tests, 5 C/Rust resolver checks and 4 compact-tail tests passed. |
| H100 default/fallback smoke | FP8 server starts with `token_batch=true`, explicitly reports CUDA executor unavailable, and generates through ordinary execution. |
| AMD device correctness/performance | Not tested on this host. |
| H100 BF16/FP8 token-batch correctness/performance | Cannot be qualified until its executor exists. |
| Prefix-cache integration | Excluded by the current AMD serving route; remains outstanding. |

Verification command:

```sh
nix develop -c cargo test -p plowrt --features cuda,hsa,hub --lib
nix develop -c cargo test -p plow-asset token_batch
nix develop -c cargo test -p plowrt --features cpu --test token_batch_resolver --test token_batch_tail
```

The shared descriptor ABI and CUDA RowGather arm do not establish a CUDA token-batch
execution path. Both CUDA program constructors still pass a null token-batch descriptor.
The current AMD adapter uses prefix-free spans through the legacy program tail; it is not
the complete shared-descriptor/compact-terminal route described in the architecture design.

The broader Gemma 4 31B IT campaign is incomplete. BF16 and FP8 assets compiled for H100;
FP8 weights use the same per-channel bytes as the vLLM export, but Plow decode activations
remain BF16. A fresh vLLM BF16 screen completed all 30 input/concurrency cells (1K–16K,
concurrency 1/4/8/16/32/64), three measured waves each, 32 output tokens and 95% requested
shared prefix. Hits reused 960 tokens at 1K through 15552 at 16K; 21 of the 375 measured
16K requests reported zero cached tokens, so the workload did include cache misses.
CPU builds overlapped this screen; final comparisons require idle-host repetitions.
There is no Plow-vs-vLLM win claim from these measurements.

The CUDA cache lifecycle review also found finished requests retaining radix references until
slot reuse. They now release those references after the tick, preserving the writable mappings
needed by inactive decode rows. Snapshot bytes count toward the cache budget. A matched FP8
rerun (same six chat prompts followed by the same six cached workload cells) reduced retained
cache from 5760 to 3680 MiB against a 4096 MiB cap. At 16K input, cached tokens increased from
2048 to 14336. Concurrency-4 median TTFT was 63.842 s before and 9.738 s after, with one measured
wave per case: regression evidence, not a statistically qualified performance result.
All six natural-text completions and all 15 measured completions were unchanged.
A separate 16K natural-text cold/warm check reused 16384 tokens on the warm request and
matched the original cold completion exactly (64 generated tokens in both requests).
That validated runtime reused zero tokens at 1K.

The current progress commit adds publication at 32-token boundaries. Whole VMM blocks
remain shared; partial full-KV blocks and sliding windows are copied from boundary snapshots.
Snapshots remain pinned through restoration, and short prefixes use second-chance eviction.
Five new host tests cover partial matching, private allocation, eviction, snapshot lifetime,
and invalid publication. Initial H100 testing found a warm-request CUDA memory fault:
default-stream D2D copies could still be running when their VMM backing was remapped.
Snapshot copies now run on the engine stream and finish before remapping or snapshot release.
The real H100 backend test checks ordered kernel → D2D → D2H execution and passes.
CUDA documents that device-to-device transfers do not synchronize the host.
[CUDA API synchronization behavior](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html).

The corrected FP8 runtime passes six natural-text cold/warm pairs at 1K/4K/16K, all with
64 generated tokens and exact agreement with the preceding runtime's cold completions.
The cached workload screen also preserves all 15 measured completions and prompt hashes.

| Input tokens | Cached tokens, concurrency 1 and 4 | TTFT at concurrency 1 | TTFT at concurrency 4 |
|---|---:|---:|---:|
| 1024 | 960 | 352 ms | 873 ms |
| 4096 | 3872 | 993 ms | 2446 ms |
| 16384 | 15552 | 2065 ms | 5132 ms |

These are single-wave screens, following six cold/warm quality pairs. The preceding
retirement screen followed six cold quality requests, so cache histories differ.
Retained cache after the current screen is 3917.5 MiB against the 4096 MiB cap.
The same FP8 server passes isolated-versus-concurrent ragged request parity, exact output
limits, slot reuse, disconnects during prefill and decode, context rejection, and recovery.
Retained cache remains under the cap after these checks.
The 1K concurrency-1 TTFT still exceeds the earlier vLLM BF16 screen's 33.1 ms.
At this stage, the full concurrency matrix and production qualification were pending,
and prefix caching remained opt-in. Later qualification is recorded below.

BF16 startup initially failed because the planner counted all 10 GiB of virtual full-KV
as resident. Planning now uses the same validated prefix layout as runtime bringup and
counts initial mapped blocks plus the block pool. Planned tensor memory changed from
80.45 to 71.57 GiB; the successful H100 load measured 71.13 GiB.

BF16 passes all six cold/warm natural-text pairs and eight concurrent natural-text requests,
including a four-request batch mixing 1K and 16K inputs. Cached and concurrent completions
match isolated cold output exactly. Four of six cold completions match vLLM BF16 exactly;
the remaining two first diverge at re-tokenized positions 43 and 22. This small corpus is
a regression check, not comprehensive model-quality qualification.

| BF16 input tokens | Cached tokens, concurrency 1 and 4 | TTFT at concurrency 1 | TTFT at concurrency 4 |
|---|---:|---:|---:|
| 1024 | 960 | 445 ms | 1053 ms |
| 4096 | 3872 | 1156 ms | 2891 ms |
| 16384 | 15552 | 2341 ms | 5842 ms |

BF16 also retains 3917.5 MiB after its six-cell screen. Its decode ladder selects among
1/2/4 rows; FP8 used width 4 in these earlier screens. Both screens use 32 output tokens,
one warmup wave and one measured wave per cell. Neither establishes a performance win.
BF16 API checks also pass ragged request parity, exact limits, slot reuse, prefill/decode
disconnects, context rejection and recovery; retained cache remains within the cap afterward.

Extending the workload to 2K exposed eviction of the primed 1920-token prefix at concurrency
4 and 16. Snapshots attached to radix nodes were protected for the entire KV lease, even
after restoration, forcing eviction of the hot short-prefix snapshot. Snapshots now become
evictable after restoration finishes; radix references still protect the shared KV blocks.
That second-chance policy considered all unpinned snapshots when no radix leaf was reclaimable.
The 2K replay reuses 1920 tokens in all 85 measured requests through concurrency 64.
The 8K replay likewise reuses 7776 tokens in all 85 measured requests. Retained cache
after both workloads is 3327.5 MiB against the 4096 MiB cap. At concurrency 64,
2K/8K median TTFT is 25.54/50.79 s and throughput is 39.10/20.03 output tokens/s.
The physical decode capacity is still 4. In the single-wave concurrency-4 screen, TTFT
changed from 9.92 s with cache misses to 1.07 s with hits. This is regression evidence.

FP8 narrow dispatch now admits the channel-scaled `GemvFp8` and `GemvGluFp8` projections
while retaining the existing KV geometry and cross-rung shape checks. The actual Gemma
packet qualifies at widths 1/2/4. On H100, 50 full-logit snapshots match widest-only execution
bit for bit across prefill, sparse-slot decoding and slot reuse. All 12 natural-text cold/warm
requests and all 15 measured cached preflight completions match the preceding FP8 runtime.

| FP8 input tokens | Concurrency-1 TPOT, width 4 | TPOT, narrow dispatch | Cached tokens |
|---|---:|---:|---:|
| 1024 | 35.61 ms | 20.93 ms | 960 |
| 4096 | 35.94 ms | 21.21 ms | 3872 |
| 16384 | 37.00 ms | 22.29 ms | 15552 |

Concurrency-4 performance is essentially unchanged because it still uses width 4.
These are single-wave regression screens. Retained cache is 3935 MiB after the new screen.
The new FP8 runtime also passes ragged request parity, output limits, slot reuse,
prefill/decode disconnects, context rejection and recovery. Raw results and executable
hashes are in `plow-fp8-cache-rung-*` and `plow-fp8-cache-rung-provenance.json`.
Default multi-step decoding delivers text in bursts of four chunks; chunk-gap tails and
cached TTFT remain performance targets. These measurements alone establish neither a
full-matrix win nor production readiness.

Prefill profiling attributes most warm TTFT to the GEMM-class segments: 282.5 ms at
1K and 1680.6 ms at 16K. The ordinary Hopper W8A8 segmented GEMM object capped registers
at 128, with a 1784-byte stack frame. Building it with `PLOW_NV_SEG_OCC1=1` permits 255
registers and reduces the stack to 488 bytes (928 spill-store / 1068 spill-load bytes
reported by ptxas). It runs 132 resident blocks instead of 264. Spills remain.

CMake now selects this register budget for ordinary Hopper Gemma W8A8 segmented prefill.
Other architecture/precision and packed-prefill build axes retain their existing defaults.
`PLOW_EXTRA_DEFINES=-DPLOW_NV_SEG_OCC1=0` restores the prior budget. Existing assets must
be rebuilt to use the change.

The focused object passes 50 bit-exact full-logit snapshots against the old widest-only
baseline, including the smallest prefill bucket, sparse slots and reuse. All 12 natural
cold/warm requests and all 15 measured preflight completions also remain exact.

| FP8 input tokens | Prior C1 TTFT | New C1 TTFT | Prior C4 TTFT | New C4 TTFT |
|---|---:|---:|---:|---:|
| 1024 | 353 ms | 150 ms | 859 ms | 310 ms |
| 4096 | 986 ms | 305 ms | 2443 ms | 721 ms |
| 16384 | 2079 ms | 719 ms | 5153 ms | 1767 ms |

These single-wave screens follow identical natural quality requests. The new screen also
uses `--multistep 1`; C1 TPOT remains about 21.0–22.4 ms, with regular chunk delivery.
Separate event-instrumented single-step controls isolate the prefill effect: sending all
segments to the existing high-register object changes C1 TTFT 344/984/2075 ms to
141/301/721 ms, preserving all three measured completions. This diagnostic is not a new
serving default. The focused object also passes API ragged parity, limits, slot reuse,
cancellation, context rejection and recovery.

Artifacts: `plow-fp8-prefill-{profile,fat-profile}*`, `build-fp8-occ1.{json,log}`,
`fp8-occ1-gpu-logits.log`, `plow-fp8-occ1-*`, and `occ1-default-build.log`.

Fresh vLLM 0.28.0 FP8 screening completed 90 waves / 1875 measured requests on the idle
H100, without overlapping builds. It uses the same per-channel FP8 weight bytes, BF16 KV,
95% primed shared input, 32 output tokens, one warmup and three measured waves per cell.
vLLM uses FP8 decode activations; Plow uses BF16 decode activations. vLLM permits 64 active
sequences and 2048 batched prefill tokens. Results below are medians across the three waves.

| Input tokens | vLLM C1 TTFT | vLLM C1 TPOT | vLLM C64 TTFT | vLLM C64 output tokens/s |
|---|---:|---:|---:|---:|
| 1024 | 24.0 ms | 15.16 ms | 323 ms | 1568 |
| 2048 | 26.9 ms | 15.17 ms | 434 ms | 1216 |
| 4096 | 40.8 ms | 15.18 ms | 827 ms | 873 |
| 8192 | 67.7 ms | 15.25 ms | 1485 ms | 544 |
| 16384 | 134.6 ms | 15.35 ms | 3119 ms | 267 |

One request at 16K/C64 reported zero cached tokens; its latency remains included.
The subsequent Plow full-grid run completed all 90 waves / 1875 measured requests,
with all prompt hashes matched to vLLM and zero cache misses. It uses the default rebuilt
Hopper W8A8 prefill object, explicit prefix caching with a 4096 MiB cap and `--multistep 1`.
No CPU builds overlapped either FP8 full-grid run. All six fresh natural-text completions
also match Plow's preceding runtime exactly. Cache retained after the grid: 3935 MiB.

| Input tokens | C1 TTFT, Plow / vLLM | C64 TTFT, Plow / vLLM | C64 output tokens/s, Plow / vLLM |
|---|---:|---:|---:|
| 1024 | 139.1 / 24.0 ms | 12.85 / 0.38 s | 76.29 / 1567.88 |
| 2048 | 137.9 / 26.9 ms | 13.23 / 0.45 s | 74.35 / 1215.56 |
| 4096 | 295.0 / 40.8 ms | 18.35 / 0.78 s | 54.28 / 873.03 |
| 8192 | 341.6 / 67.7 ms | 20.14 / 1.49 s | 49.56 / 543.73 |
| 16384 | 720.9 / 134.6 ms | 33.04 / 3.16 s | 30.65 / 267.17 |

This comparison pools request latency percentiles across three measured waves; throughput
is the median wave throughput. The preceding vLLM-only table uses medians of wave summaries.
[All 30 cells, including latency and chunk-gap tails](gemma31-h100-fp8-cached-comparison.csv)
are available as CSV. Plow loses throughput in every cell. At 16K/C64 it has lower pooled
TPOT (111.86 vs 133.28 ms), but much longer queueing and end-to-end latency. The physical
Plow capacity is four slots; vLLM permits 64 active sequences. No all-metrics win is established.
Raw paired results and provenance: `plow-fp8-occ1-full.jsonl`,
`matched-fp8-cached-comparison.json`, and `matched-fp8-cached-provenance.json`.
The vLLM-only results, all 30 summary cells and provenance remain in
`vllm-fp8-cached-screen.jsonl`, `vllm-fp8-cached-screen-summary.json` and
`vllm-fp8-screen-provenance.json`.

None of the six matching natural prompts produces identical 64-token text between vLLM
FP8 and Plow FP8. Median first divergence after re-tokenization is 13.5/15/8 at 1K/4K/16K.
Cache reuse and narrow dispatch preserve Plow's own output exactly. Different activation
precision is a known difference, not a proven explanation for every divergence. This
small prose corpus does not establish equivalent model quality. Comparison artifact:
`plow-vllm-fp8-quality-comparison.json`.

Raw artifacts: `/opt/dlami/nvme/tmp/gemma31-glm53-h100-20260908/`.
Host test log: `/tmp/plow-token-batch-readiness-tests-final.log`.
Final lifecycle test log: `/tmp/plow-prefix-retire-tests.log`.
Current partial-prefix host test log: `/tmp/plow-subblock-library-tests.log`.
Stream-ordering verification: `/tmp/plow-subblock-stream-library-tests.log` and
`/tmp/plow-subblock-stream-cuda-test.log`.
Current GPU artifacts: `plow-fp8-stream-*`; source and executable hashes are in
`plow-fp8-stream-provenance.json` under the raw-artifact directory.
BF16 artifacts: `plow-bf16-prefix-plan-*`, with hashes in
`plow-bf16-prefix-plan-provenance.json`. Planner host checks:
`/tmp/plow-prefix-plan-library-tests.log` (582 passed, 14 ignored).
Latest host and FP8 device checks: `/tmp/plow-cache-rung-library-tests.log` (584 passed,
14 ignored) and `/tmp/plow-fp8-rung-gpu.log`. Queue artifacts: `plow-bf16-cache-rung-*`.
CPU integration log: `/tmp/plow-token-batch-cpu-integration-tests.log`.
The compact-tail fixture needed the new optional worker-pinning argument before it could run.

## Automatic prefix-cache qualification

Prefix reuse now defaults on for validated Hopper (CC 9.0) hybrid BF16-KV packets with
HD 256 sliding attention, HD 512 full attention and window 1024. Both Gemma BF16 and FP8
weights qualify. The default retained-cache budget is 4096 MiB, including boundary snapshots;
it is a soft cap while entries are pinned. Explicit `--vmm-prefix=false` disables reuse,
while explicit true retains the broader supported-layout policy. Automatic selection
excludes TP, recurrent state, mixed/prepared decode, and explicit packed-prefill/live-KV
modes. Packed-prefill metadata alone yields to prefix reuse; `--pf-batch=true` retains the
packed route. Combined CUDA packed-prefill/prefix execution remains outstanding.

The planner and runtime use the same eligibility decision. Cold-prefill benchmarking checks
the actual loaded cache state. Host coverage includes selection, geometry rejection and
explicit overrides: 586 CUDA/HSA library tests pass, 14 ignored; 9 CPU integration tests pass.

The frozen `bin/plowrt-prefix-auto` starts both batch-4 models without cache flags and logs
`requested=None selected=true`. Each precision passes six natural cold/warm pairs at 1K/4K/16K:
all 12 completions match the preceding cold baseline, including exact warm/cold agreement.
Four distinct 16K cold prompts then complete concurrently, reproduce exactly on isolated
replay, and are followed by successful short-request recovery. Output counts and finish
reasons are checked. FP8 also preserves all 9 pressure completions from the explicit-cache
baseline. Retained cache after pressure is 3310 MiB FP8 and 2510 MiB BF16, below 4096 MiB.
BF16 GPU memory was observed at 78880 MiB during pressure. Both precisions pass API ragged
parity, exact output limits, slot reuse, disconnects during prefill/decode, context rejection
and recovery. An explicit-false FP8 restart generates identical repeated completions with
zero cached tokens. These are correctness checks; CPU builds overlapped and their timings
are not performance evidence.

Raw results are `plow-{bf16,fp8}-auto-*`, `plow-fp8-auto-off-*`, and
`prefix-auto-qualification.json` under the campaign directory. This qualifies the tested
H100 batch-4 configuration and default policy; it does not qualify larger batch assets,
other GPU families, or the missing CUDA token-batch executor. The matched performance grid
above still loses throughput to vLLM in every cell.

## Larger batch assets and snapshot-copy experiment

H100 assets with 1024-token maximum prefill chunks now compile at BF16 batch 8 and FP8 batch 16,
with independently emitted widest-only reference packets. Full-logit GPU comparisons are
bit-exact across 74 BF16 and 118 FP8 snapshots, including sparse slots and reset. VMM prefix
allocation is enabled for these tests because fully resident virtual KV would exceed 80 GB.
These gates used the experimental copy runtime described below; serving pressure, quality
and performance at the larger capacities remain pending. Provenance and object hashes are
in `larger-batch-c1024-qualification.json`.

A separate experiment grouped snapshot heads with `cuMemcpy2DAsync_v2`. CUDA rejected the
large partial full-KV VMM pitch: natural completions remained exact but short-prefix cache
publication failed. The corrected experiment kept full-KV copies per head and grouped only
sliding windows. It passes the H100 short/full/wrapped pitched-copy test, all 12 natural FP8
cold/warm completions with exact warm cache counts, all 15 matched cached screen requests,
four concurrent 16K cold requests with exact replay and short recovery, and all three API
lifecycle checks. The corrected host suite also passes 586 tests, with 14 ignored.

| Input | Concurrency | Per-head TTFT | Pitched sliding TTFT | Per-head tok/s | Pitched sliding tok/s |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 150.1ms | 154.0ms | 39.92 | 39.75 |
| 1024 | 4 | 310.1ms | 290.0ms | 73.37 | 75.36 |
| 4096 | 1 | 304.7ms | 288.2ms | 33.15 | 33.75 |
| 4096 | 4 | 720.6ms | 695.6ms | 52.62 | 53.77 |
| 16384 | 1 | 718.5ms | 711.3ms | 22.63 | 22.75 |
| 16384 | 4 | 1766.9ms | 1751.5ms | 30.19 | 30.45 |

These are single-wave screens, with identical prompts, completions and cached-token counts.
The small changes and the 1K/C1 regression do not justify promotion without stronger timing
and broader layout qualification. The branch retains the tested per-head copy implementation.
Experimental binaries, source patches and results remain under `prefix-2d*` and
`plow-fp8-2d*` in the campaign directory for follow-up.

## Reclaiming retired prefix KV

The FP8 batch-16 pressure run exposed OOM after earlier requests had finished. Retirement
released cache holds but retained each slot's long KV mappings and saved position. Only
12 of 16 distinct cold 16K requests completed; two decode streams and two prefills failed.
The CUDA context remained usable, and a subsequent short request completed.

Retirement now publishes the valid output prefix, resets the position, and releases the
slot's mappings. Existing decode backstops remap idle row zero before launch. Background
pre-mapping hints carry a slot generation checked under the allocation lock, so an old hint
cannot recreate a retired or reused window. Allocation retries also preserve successfully
mapped heads of a partially completed block instead of trying to map them again.

The same FP8 batch-16 workload now completes all 16 cold requests, 16 exact isolated replays,
and short recovery. All 12 previously successful cold completions remain exact. Retained
cache ends at 3310 MiB against the 4096 MiB cap. All 12 natural cold/warm completions, all
15 matched preflight completions and their cache-hit counts are unchanged. API ragged parity,
limits, slot reuse, prefill/decode disconnects, context rejection and recovery pass.

Host tests: 588 passed, 14 ignored. The full-logit GPU gate now retires lower slots only in
the candidate while the highest slot continues decoding against a non-retired reference:
122 FP8 batch-16 and 78 BF16 batch-8 snapshots are bit-exact. BF16 batch-8 API serving and
pressure qualification were pending at that stage; the results below complete them. Raw failure and corrected results are
`plow-fp8-b16-c1024-*`, `plow-fp8-b16-reclaim-*`, and `prefix-reclaim-qualification.json`.

## Prefix eviction under batch-16 bursts

The batch-16 timing run stopped after 47 complete waves and 821 requests because the
shared prefix was repeatedly evicted. It recorded 728 cache misses, starting at 1K/C8.
All completed prompt hashes match the earlier batch-4 grid. These incomplete results are
preserved as `plow-fp8-b16-full.*` and `plow-fp8-b16-failed-summary.json`; they do not form
a full performance comparison.

When no radix leaf is reclaimable, eviction now protects the most recently reused
unpinned snapshot and selects other snapshots by LRU. This keeps one shared prefix
through bursts of unique tails while allowing a new workload to replace older entries.
Active radix leases may temporarily retain that snapshot above the soft budget; releasing
leases trims the cache again, and OOM reclamation can still remove the snapshot. Snapshot
pins and radix leases retain their existing lifetime rules. A first candidate
that protected multiple historical reused snapshots failed natural 16K warm reuse and
was rejected; its binary, patch and results remain under `prefix-scan*` and
`plow-fp8-b16-scan-*` in the campaign directory.

The expanded unique-tail regression fails with the prior second-chance policy. The
revised policy passes that test, new-prefix admission, a new long prompt competing with
its output tail, active-lease budget pressure, OOM eviction of the protected snapshot,
and the full 592-test host suite. Before the active-lease exemption, H100 retained all
192 measured burst prefixes through 8K, but the first 16K wave lost all 16 prefixes.
It completed with cold fallback; debugger attachment was attempted during diagnosis,
so that wave is excluded from performance claims.
The final policy retains all 240 measured batch-16 prefixes across 1K/2K/4K/8K/16K,
with three waves at each length. All 208 prompts also completed by the previous candidate
have identical hashes and output text, including the 16K cold-to-cached comparison.
All 12 natural outputs and six warm cache counts pass. Sixteen distinct cold 16K requests,
16 exact isolated replays, short recovery, and all three API lifecycle checks pass.
Retained cache returns to 3310 MiB under the 4096 MiB budget. Raw results and hashes are
`plow-fp8-b16-scan3-*` and `prefix-scan3-qualification.json`.

These bursts use a subset seed sequence and are not prompt-matched to the full vLLM grid;
they support cache regression claims only. No CPU build overlaps their timing. A separate
packed-prefill prototype was built during functional pressure checks, whose durations are
not used for performance claims. The prototype is not included in this change.

BF16 batch 8 starts at 71.48 GiB with automatic prefix reuse. Its 12 natural cold/warm
outputs match the prior batch-4 baseline and all six warm cache counts pass. All eight
distinct cold 16K requests, eight exact isolated replays, short recovery, and three API
lifecycle checks pass. Retained cache returns to 4030 MiB. Results are
`plow-bf16-b8-scan3-*`, with hashes in the `bf16_batch8` section of
`prefix-scan3-qualification.json`. This qualifies serving and memory behavior at batch 8;
the fresh matched BF16 performance grid remains outstanding.

## Packed-prefix prototype: numerical diagnosis and failed pressure test

The initial explicit BF16 batch-8 packed-prefix prototype failed qualification.
It packs up to eight requests per launch. All
39 preflight prompts and expected cache counts match ordinary execution, but one output
differs. Natural cold/warm outputs agree within the prototype; only 8 of 12 match the
ordinary baseline. Cold 16K pressure completes two requests before six decode streams
fail with nonfatal allocation OOM. A short recovery request succeeds; lifecycle checks
were not reached. These results reject promotion.

Single-wave screening shows 1K/concurrency-8 throughput rising from 42.0 to 66.3 output
tokens/s. At 16K/concurrency-8, throughput rises from 11.84 to 13.91, but median TTFT
worsens from 10.39 to 15.60 seconds. This is neither a full comparison nor an all-metrics
improvement. Raw results remain under `plow-bf16-b8-packed-prefix2-*` and
`bf16-b8-packed2-preflight-diagnostic.json` in the campaign directory.

The `prefix_logits` example isolates the numerical difference using two natural prompts
(1055 and 16415 tokens), fresh engines, 64 output steps, and all 262144 vocabulary logits.
Candidate modes consume the ordinary run's selected tokens at every subsequent step:

| Comparison | Result across 128 full-vocabulary snapshots |
|---|---|
| Full-prompt prefill vs isolated prefill ending before the final token, then decode | Three argmax differences; maximum absolute logit difference 0.84375. |
| Isolated split prefill vs packed prefill, both using decode for the final token | Bit-exact. |
| Full-prompt prefill vs packed prefill followed by an ordinary GEMM terminal row | Bit-exact. |

For these prompts, the final-token GEMM/GEMV choice explains the observed divergence;
packing adds no further differences. This single-request diagnostic does not qualify
multi-request numerical behavior or memory admission. A terminal GEMM row is a verified
correction path, but its serving integration and performance remain untested. The harness,
source patch, build logs, output arrays and SHA256 records are preserved in
`prefix-logits-*`, `prefix-logits/`, and `packed-prefix2-numerics-qualification.json`.

Run the harness with `nix develop -c cargo run -p plowrt --example prefix_logits
--features cuda,hsa,hub -- <assets> <mode> <cases.json> <new-output-directory> [reference]`.
Cases contain `messages`, `prompt_tokens`, and `steps`. Run `ordinary` first, then pass
its output directory to `split`, `packed`, `packed-tail`, or `packed-complete`. Use prefix reuse and disable
multistep; packed modes also require compatible packed-prefix runtime support and assets.
The rejected prototype's patch is preserved externally as `packed-prefix2-source.patch`.

## Explicit packed-prefix KV admission

The explicit CUDA combination `--pf-batch=true --vmm-prefix=true` now requires compatible
packed-prefill metadata, direct-KV segmented programs, and a valid prefix layout. Before
packing, each request attaches its cached prefix and reserves physical full-KV backing
for its prompt, output budget, and prefetch/padding margin. Idle rows needed by wider
decode rungs are also backed. Allocation OOM before launch rolls back that request and
queues it while admitted requests finish. Waiters retry only after retirement releases
an admission; existing waiters get priority over fresh arrivals. Cancellation retires
both admitted and waiting requests.

On H100, the batch-8 cold 16K workload now completes all eight requests, all eight exact
isolated replays, and short recovery. The preceding candidate completed only two requests
before six decode streams failed OOM. All 12 natural outputs match that preceding
candidate and all six warm cache counts pass. API ragged parity, limits, reuse,
prefill/decode cancellation, context rejection and recovery pass. An additional API test
observes KV admission waiting under eight requests with 16000-token output budgets,
cancels all streams, and recovers with an exact one-token-prompt completion.

The new device test fills HBM with full-context reservations: three requests admit and
five wait. Sixteen retry rounds create no new KV blocks. Retirement admits an older
waiter before a new arrival. Cancelling all slots still permits decode in the highest
slot, exercising every idle-row mapping. The device test passes in addition to the
592-test host suite; it remains ignored in ordinary host runs.

At this admission milestone, the experimental combination still used decode for the
last prompt token, with the numerical differences documented above. The compact terminal
correction is described below. Matched performance qualification, FP8 packed assets,
and the CUDA unified token-batch executor remain outstanding. Admission validation is
not a vLLM win claim.
Functional pressure durations include a CPU diagnostic build and are not performance
evidence. Raw proof and hashes are `plow-bf16-b8-packed-admission2-*`,
`packed-admission2-gpu-test.log`, and `packed-admission2-qualification.json`.

## Compact BF16 packed-prefix terminal, 2026-09-09

Compatible explicit packed-prefix execution now includes the final prompt row in the
packed transformer body. Completed requests gather their residual rows into a compact
buffer and run the ordinary RmsNorm → GEMM head → SoftCap → Argmax chain. Sampling reads
compact logits rows while decode continues on each request's physical slot. Intermediate
chunks produce no output; completion stops the prefill pass before another launch can
overwrite logits. Upload buffers survive until the engine stream drains, including errors.

Load-time checks require matching ordinary BF16 tails across prefill buckets, valid buffer
capacities, non-overlapping residual storage, and a prefill object with RowGather support.
Unsupported tails retain the legacy final-token decode route. The compact implementation
currently runs after the body's unused original tail, so it still pays for an extra head.
It does not implement the shared-descriptor CUDA token-batch executor or enable FP8 packing.

The single-request probe compares 128 complete vocabulary snapshots with ordinary prefill:
1055- and 16415-token prompts, 64 teacher-forced steps each. Every BF16 logit is bit-exact.
A second GPU test completes ragged final chunks in reversed order on physical slots 3 and
7, checks both compact output rows, then teacher-forces decode through step 64. All 128
snapshots are bit-exact against the same ordinary references. The admission device test
still passes with the extra terminal buffers; the host suite passes 594 tests, 16 ignored.

All 12 natural cold/warm completions now match ordinary BF16, including the four completions
that differed under the previous final-token decode route. All six warm cache counts and
the three API lifecycle checks pass. Eight cold 16K requests, eight exact isolated replays,
and recovery pass. Eight streams are cancelled after observing KV admission waiting;
one-token-prompt recovery remains exact. Sixteen isolated/concurrent sampling pairs also
match, covering penalties, logit bias, nonzero temperature, reversed submission order,
and one-token prompts.

The combination remains experimental and off by default; automatic ordinary prefix
caching remains enabled on its qualified layouts.

The matched nine-cell screen uses 1K/4K/16K input, concurrency 1/4/8, 32 output tokens,
and 95% requested shared prefix. All 39 prompt hashes, texts and cached-token counts
match ordinary BF16. No CPU builds or other GPU jobs overlapped these timings.

| Input / concurrency | Ordinary → compact TTFT, ms | Ordinary → compact median TPOT, ms | Ordinary → compact output tok/s |
|---|---:|---:|---:|
| 1K / 8 | 1914 → 1478 | 124.66 → 76.55 | 42.00 → 66.44 |
| 4K / 8 | 5247 → 4568 | 211.46 → 78.77 | 21.09 → 36.50 |
| 16K / 4 | 5746 → 6081 | 168.75 → 128.94 | 11.52 → 12.50 |
| 16K / 8 | 10393 → 15682 | 349.65 → 87.46 | 11.84 → 13.91 |

These are single-wave screens, not production performance qualification. Long-context
TTFT still regresses; this does not meet the all-metrics target or establish a vLLM win.
The frozen runtime is `bin/plowrt-packed-terminal`, SHA256
`f9c46bd79de758bb5010cf2bf358f297d2e5fab063cc8cdb02dee4337f3005c6`.
Raw proof, scripts and hashes are preserved under the campaign directory in
`packed-terminal-qualification.json`, `bf16-b8-packed-terminal-preflight-comparison.json`,
`plow-bf16-b8-packed-terminal-*`, and `packed-terminal-final-*-gpu.log`.

## Packed BF16 GEMM register budget

The H100 packed BF16 GEMM kernel was capped at 128 registers and used 1128 bytes of
stack per thread. `PLOW_NV_SEG_OCC1=1` permits 255 registers and reduces the stack
to 112 bytes. CMake now selects it for Hopper Gemma packed GEMM without W8A8 or
FP8 KV. Ordinary BF16, packed W8A8, Blackwell and other model families keep their
previous settings; ordinary Hopper Gemma W8A8 already uses OCC1. The existing
`PLOW_EXTRA_DEFINES=-DPLOW_NV_SEG_OCC1=0` override restores the cap.

Only the packed GEMM cubin changes. Its 128 sparse/reversed-slot full-logit snapshots
are bit-exact against ordinary prefill. All 12 natural outputs and all 39 matched
cached-screen outputs, prompt hashes and cache counts remain exact. The default
CMake build produces a byte-identical cubin to the device-tested candidate; six
configuration checks cover selection, rollback and excluded scopes.

| Input / concurrency | Previous → OCC1 TTFT, ms | Previous → OCC1 output tok/s |
|---|---:|---:|
| 1K / 1 | 426 → 154 | 24.33 → 30.67 |
| 4K / 8 | 4568 → 1336 | 36.50 → 67.68 |
| 16K / 1 | 2295 → 703 | 9.91 → 19.56 |
| 16K / 8 | 15682 → 5173 | 13.91 → 32.41 |

All nine cells improve TTFT and throughput. This is one measured wave per cell,
with no build or other GPU workload overlap. Decode is unchanged: 16K/C8 median
TPOT is 87.46 → 87.61 ms. Full data is in
`gemma31-h100-bf16-packed-occ1-comparison.csv`. The combined packed-prefix route
remains opt-in, and a full BF16 comparison against vLLM remains outstanding.

Eight cold 16K requests, eight exact isolated replays, short recovery, all three API
lifecycle checks, 16 sampling pairs, and cancellation of eight streams after observed
KV admission waiting pass. Pressure durations include a CMake build and are excluded
from performance claims. The candidate/default cubin SHA256 is
`62a313cd4fce717db1bb13e26f85774fe59c40052700b9675e566b37d0033005`.
Raw data, scripts, build scope checks and hashes are recorded in
`bf16-packed-occ1-qualification.json` under the campaign directory.

## Full BF16 cached workload

The packed OCC1 runtime completes the full 1K/2K/4K/8K/16K grid at requested
concurrency 1/4/8/16/32/64: three measured waves per cell, 32 generated tokens,
and 95% requested shared prefix. All 1875 measured requests reuse the expected
32-token-aligned prefix; none report a cache miss. All six natural completions
match the preceding qualified BF16 runtime. Retained cache after the run is
3935 MiB against the 4096 MiB budget. Physical capacity is eight slots; higher
requested concurrency queues over those slots.

| Input / concurrency | Request TTFT p50, ms | Request TPOT p50, ms | Median wave output tok/s |
|---|---:|---:|---:|
| 1K / 1 | 153 | 28.66 | 30.70 |
| 1K / 64 | 11510 | 76.69 | 83.12 |
| 16K / 1 | 699 | 30.02 | 19.60 |
| 16K / 64 | 33329 | 87.56 | 31.94 |

No CPU builds or other GPU workloads overlapped the run. The raw waves, quality
capture, metrics, frozen server log and hashes are recorded in
`plow-bf16-packed-occ1-full-qualification.json` and `plow-bf16-packed-occ1-full.*`
under the campaign directory.

## Clean matched BF16 comparison

The fresh vLLM 0.28.0 run also completes all 90 waves and 1875 requests with no
overlapping CPU builds or other GPU workloads. All prompt hashes and output lengths
match Plow. vLLM reports 16 cache misses; Plow reports none. Completion text matches
for 1687/1875 requests and four of the six natural prompts. Text agreement is a
numerical diagnostic, not a task-quality score.

Plow loses all 30 cells on median TTFT, median end-to-end latency, median TPOT and
median-wave throughput. The prefix-cache improvements do not meet the performance goal.

| Input / concurrency | Plow / vLLM TTFT p50, ms | Plow / vLLM TPOT p50, ms | Plow / vLLM output tok/s |
|---|---:|---:|---:|
| 1K / 1 | 153 / 34 | 28.66 / 23.58 | 30.70 / 41.82 |
| 1K / 64 | 11510 / 350 | 76.69 / 42.45 | 83.12 / 1202.64 |
| 16K / 1 | 699 / 146 | 30.02 / 23.79 | 19.60 / 36.13 |
| 16K / 64 | 33329 / 7112 | 87.56 / 49.63 | 31.94 / 124.07 |

Both engines use BF16 weights, activations and KV. Plow has eight physical slots;
vLLM permits up to 64 sequences under its memory scheduler. Both receive concurrency
through 64. Prefill limits are 1024 tokens for Plow and 2048 for vLLM. Plow uses a
4096 MiB retained-prefix budget; vLLM uses 90% GPU memory utilization. These are the
tested serving configurations. The CSV includes p50/p95/p99 distributions and
per-cell cache-miss counts; text-chunk delivery gaps do not guarantee per-token ITL.

Full results are in `gemma31-h100-bf16-cached-clean-comparison.csv`. Raw proof and
hashes are `bf16-cached-clean-comparison.json`,
`vllm-bf16-cached-clean-qualification.json`, and `vllm-bf16-cached-clean.*` under
the campaign directory. FP8 packing and a faster batched decode path remain work
toward the requested result; the full objective is not achieved.

## FP8 packed contract qualification

The LIVE KV audit now accepts direct FP8 GEMV, GEMM and activation-quantization
operands. Tensor handles still undergo range and cache-alias checks. Indirect
FP8 tensor-map handles and folded GLU operands remain rejected. FP8 packed
metadata requires explicit compiler selection; ordinary attention planning is
preserved. Recompiling the full model produces byte-identical default and explicit
packets against their respective references.

On the H100, explicit packed prefill plus VMM prefix caching passes with physical
batch 16, a 1024-token chunk, FP8 weights and BF16 KV:

- 87 asset tests, 594 runtime tests and four compiler capability tests pass.
  Sixteen runtime GPU tests remain ignored in the host suite; the selected GPU
  checks below ran separately.
- Packed-complete and sparse-slot execution each match 128 ordinary full-vocabulary
  logit snapshots bit for bit. Sparse slots 15 and 7 advance to the widest decode rung.
- Admission retries preserve waiter priority without allocation while blocked;
  retirement, cancellation and recovery pass on the GPU.
- All 12 natural completions match the prior FP8 path, with expected warm cache counts.
  All 16 concurrent cold 16K requests finish and match their isolated replays.
- Three API lifecycle checks, 32 isolated/concurrent sampling pairs and 16 cancelled
  streams followed by exact one-token-prompt recovery pass.

This qualification uses a manually rebuilt packed GEMM with OCC1; that FP8 CMake
default is not promoted. Performance measurement follows separately. Combined
packed-prefix execution remains opt-in, and this work does not implement the
shared CUDA token-batch executor. Repository-wide formatting checks still report
existing differences; unrelated formatting is unchanged.

`fp8-packed-functional-qualification.json` records source, binary, asset and raw
proof hashes in the campaign directory. The runtime SHA256 is
`3b769811adc02fd90273c29c546eef29253d16fec50b9b19c4fbc18cd8da23e3`.

The six-cell FP8 screen preserves all 15 prompt hashes, completion texts and cached
token counts against the prior ordinary batch-16 reclaim runtime. Throughput rises
in all six cells, but 16K/C4 median TTFT regresses from 1732 ms to 2642 ms. At that
cell, median TPOT improves from 82.73 ms to 49.47 ms and throughput from 29.00 to
30.62 tokens/s. This is one measured wave per cell after warmup, covering
1K/4K/16K at concurrency 1/4; it does not qualify a default change. The candidate
also includes subsequent prefix-cache fixes, so this is a serving comparison,
not an isolated kernel experiment. No builds or other GPU workloads overlap timing.
Results are in `gemma31-h100-fp8-packed-preflight.csv`; raw proof is
`fp8-packed-preflight-comparison.json` and `plow-fp8-packed-occ1-preflight.*` in
the campaign directory.

## Full matched FP8 packed comparison

The packed batch-16 candidate completes all 90 waves and 1875 requests across
1K/2K/4K/8K/16K and concurrency 1/4/8/16/32/64. All requests reuse the expected
32-token-aligned prefix; all prompt hashes, output lengths and completion texts
match the preceding ordinary batch-4 Plow run. The six natural completions also
match the qualified FP8 path. Retained prefix memory after the run is 3455 MiB
against the 4096 MiB budget.

Against the matched vLLM 0.28.0 FP8 reference, all 1875 prompt hashes and output
lengths agree; 1475 completion texts agree. vLLM has one cache miss, Plow none.
Plow loses all 30 cells on median TTFT, median end-to-end latency and median-wave
throughput. Median TPOT is lower in one cell, 16K/C64, without a throughput win.

| Input / concurrency | Plow / vLLM TTFT p50, ms | Plow / vLLM TPOT p50, ms | Plow / vLLM output tok/s |
|---|---:|---:|---:|
| 1K / 1 | 141 / 24 | 21.08 / 15.16 | 40.24 / 64.75 |
| 1K / 64 | 8303 / 384 | 109.12 / 29.80 | 109.64 / 1567.88 |
| 16K / 1 | 670 / 135 | 22.43 / 15.35 | 23.39 / 52.20 |
| 16K / 64 | 41650 / 3165 | 130.97 / 133.28 | 27.96 / 267.17 |

Both engines use the same per-channel FP8 weights and BF16 KV. Plow uses FP8
prefill activations and BF16 decode activations; vLLM uses FP8 activations for
both. Plow has 16 physical slots and queues higher concurrency; vLLM permits
up to 64 sequences under its memory scheduler. The prefill limits remain 1024
tokens for Plow and 2048 for vLLM. No builds or other GPU workloads overlap
the timed run. These results do not qualify combined packing as the default.

`gemma31-h100-fp8-packed-cached-comparison.csv` contains all cells and request
tail distributions. Raw waves, quality captures, metrics, frozen server logs and
verified hashes are in `plow-fp8-packed-occ1-full-qualification.json`,
`fp8-packed-cached-comparison.json` and `fp8-packed-vs-prior-b4-quality.json`
under the campaign directory. The compute performance goal remains unmet.

## Experimental Gemma BF16 cuBLASLt decode

The opcode traces identify matrix-vector bodies as the main sampled decode cost:
about 80% of block-0 cycles for BF16 batch 8 and 75% for FP8 batch 16. These are
bounded, instrumented single-block samples, not whole-GPU performance counters.
Nsight Compute did not produce a usable report.

Gemma's existing `--emit-decode-cublaslt=true` compiler option supports BF16
decode widths, for example `--emit-decode-batch-ladder=1,2,4,8`.
It separates Q/K/V and gate/up projections and isolates body GEMV instructions
for the existing cuBLASLt runtime integration. Segmentation preserves instruction
operands and dependencies; the LM head keeps its native GEMV. Compilation without
the option produces a byte-identical BF16 packet.

In a fresh direct-engine diagnostic with a 1K synthetic prompt, 16 warmup steps
and 64 measured steps, the batch-8 median falls from 76.51 ms to 36.24 ms (2.11×).
With one active slot, the fixed batch-8 candidate regresses from 28.61 ms to
31.02 ms. These are decode-step measurements, not serving throughput.

Functional checks pass: 12 matching cold/warm natural completions with expected
cache reuse, eight concurrent cold 16K requests and exact isolated replays,
three API lifecycle checks, 32 sampling pairs, and 16 cancelled streams followed
by exact recovery. Compiler tests cover isolation, unchanged dependencies,
unsupported combinations, and the existing Qwen cuBLASLt behavior.

Numerics change when replacing fused GEMV bodies with separate BF16 projections
and cuBLASLt. Initial prefill logits are exact; teacher-forced decode top choices
agree on 126/128 frames. The minimum full-logit cosine similarity is 0.99970,
but the maximum absolute difference reaches 1.50. Three of six natural responses
match the prior Plow path, and two match vLLM; text agreement is not an accuracy
score. Quality assessment and adaptive decode widths remain necessary before
default promotion. This option is experimental and does not meet the full goal.

Raw results and hashes are in `gemma-cublaslt-experimental-qualification.json`,
`gemma-cublaslt-logits-comparison.json`, and
`gemma-bf16-cublaslt-step-comparison.json` under the campaign directory.


The fixed-width serving screen completes nine cells: 1K/4K/16K inputs at
concurrency 1/4/8, 95% shared prefix, 32 output tokens, one measured wave after
warmup. All 39 prompt hashes, output lengths and cache counts match the native
packed BF16 control; 38 completion texts agree. Throughput improves in all six
C4/C8 cells but regresses in all three C1 cells. At 1K/C8 it rises from 84.25 to
142.78 output tokens/s; at 1K/C1 it falls from 30.67 to 28.77. TTFT is mixed:
1K/C4 regresses from 305 to 514 ms. No builds or other GPU jobs overlap timing.

All cells are in `gemma31-h100-bf16-cublaslt-preflight.csv`; raw comparison is
`gemma-bf16-cublaslt-serving-comparison.json` in the campaign directory. These
single-wave results support investigating adaptive widths, not default promotion.

## Adaptive cuBLASLt decode widths

The compiler emits projection roles for every requested width. The CUDA loader
checks role coverage, matching dependencies, instruction equivalence and KV
addressing before selecting a narrower graph. Each graph owns its route plans
and instruction tables; the plans share one 256 MiB workspace on the ordered
engine stream. Counter storage includes a separate cursor for every segment.

Narrower plans reuse the widest projection's algorithm after a cuBLASLt geometry
and workspace compatibility check. Independently tuning each width changed
reduction order and failed the sparse-slot numerical check, reaching a 6.5
absolute logit difference. Reusing algorithms restores exact full-logit equality
in the 78-frame transition screen. The test covers widths 1/2/4/8, reversed and
sparse slot lists, repeated transitions, slot reset, prompt continuation, and
retirement of lower slots while the highest slot continues decoding.

The transition test reuses the same loaded plans to isolate width selection from
load-time autotuning. A separate 128-frame natural-prompt capture at 1K and 16K
also matches the earlier fixed-width cuBLASLt reference exactly. The full runtime
host suite passes 596 tests, with 17 device/asset-dependent tests ignored; the
78-frame GPU test is run explicitly. All seven CUDA objects compile successfully.

These results do not establish equality with native GEMV or vLLM;
the existing cuBLASLt numerical differences and model-quality caveat still apply.
Adaptive serving checks pass: all 12 cold/warm natural completions match the
fixed-width cuBLASLt reference, eight concurrent cold 16K requests match their
isolated replays, and recovery, three API checks, 32 sampling pairs and 16
cancelled streams complete successfully.
The option remains experimental and is not enabled by default.

Raw results and artifact hashes are in
`cublaslt-ladder-experimental-qualification.json`, `cublaslt-ladder-exact-gpu.log`
and `cublaslt-ladder-shared-natural-comparison.json` in the campaign directory.

A fresh direct-engine comparison at 1K context, with 16 warmup steps and 64
measured steps, removes the fixed-width single-request regression. At one active
slot, median step time is 27.59 ms vs native 28.62 ms. At eight active slots it
is 36.22 ms vs 76.55 ms. Both paths use the same newly built benchmark binary;
no builds or other GPU jobs overlap timing. These are decode-step diagnostics,
not serving throughput or a vLLM comparison. Raw measurements are in
`gemma-bf16-cublaslt-adaptive-step-comparison.json` and
`gemma-bf16-{native,adaptive}-shared-step-b{1,8}.log` in the campaign directory.

The matched nine-cell serving screen completes all 39 requests with identical
prompt hashes, cache counts and output lengths. All 39 completion texts match
fixed-width cuBLASLt; 38 match native BF16. Adaptive throughput exceeds native
BF16 in all nine cells, including all three single-request cases. At 1K/C1,
throughput rises from 30.67 to 31.90 output tokens/s; at 1K/C8, from 84.25 to
142.41. Median TPOT improves in every cell.

TTFT remains mixed: four cells regress vs native, including 1K/C4 from 305 to
441 ms and 4K/C8 from 1336 to 1407 ms. Adaptive throughput also regresses vs the
fixed-width candidate at 4K/C4, from 72.91 to 65.17 tokens/s. This is one measured
wave per cell after warmup, with no overlapping builds or GPU jobs; it does not
establish a vLLM win or qualify default promotion. Full results are in
`gemma31-h100-bf16-cublaslt-adaptive-preflight.csv`, with raw comparison in
`gemma-bf16-cublaslt-adaptive-serving-comparison.json` in the campaign directory.


## cuBLASLt decode launch elision, 2026-09-09

The runtime removes the interpreter launch after each isolated cuBLASLt projection.
For the Gemma packet this removes 410 launches from its 651-segment decode chain.
The loader validates ordered coarse dependencies, then clears only waits on
library-produced counters in the uploaded tables. Each consumer launch follows
the complete library operation on the same CUDA stream. Ordinary counter waits
remain intact; captured and direct launch paths use the same rule.

The final host suite passes 597 tests with 17 ignored. A controlled GPU check
compares the original launch sequence against elided adaptive graphs using the
same cuBLASLt plans: all 78 full-vocabulary frames are bit-exact across widths
1/2/4/8, sparse/reversed slots and lifecycle transitions. That GPU binary preceded
a final loader-only guard requiring dependency thresholds to equal producer block
counts; the final host suite includes that rejection case.

A fresh, isolated 1K direct-step screen used the same packet, 16 warmups and 64
measured steps per case:

| Active slots | Original median ms | Elided median ms | Reduction |
|---:|---:|---:|---:|
| 1 | 27.585 | 25.807 | 6.4% |
| 8 | 36.222 | 34.349 | 5.2% |

These are decode-step measurements, not serving throughput or vLLM wins. Each
runtime loaded and autotuned independently. A separate natural-prompt comparison
against a previous process load was exact for all 64 frames at 16K, but at 1K
matched top-1 in 62/64 frames (max absolute difference 1.515625, minimum cosine
0.999710). Cross-load algorithm selection is a possible cause; it is not established
by that comparison. The same-plan controlled test above isolates the launch change.

Raw evidence in the campaign directory: `cublaslt-elide-final-host-tests.log`,
`cublaslt-elide-controlled-gpu.log`, `cublaslt-elide-natural-comparison.json`, and
`gemma-bf16-cublaslt-elision-step-comparison.json`.

## Native tensor-core projection screens, 2026-09-09

Two standalone experiments are now included under `runtime/nvidia/experiments/`.
They do not change serving dispatch or defaults.

- `gemma31_decode_tc.cu`: transposed BF16 MMA tiles improve all 24 dense shape/batch
  cells against production GEMV and the existing split-K tensor-core control.
  cuBLASLt remains faster in all 24 cells. At B8 the candidate takes 15.81–88.93 us
  across eight shapes, versus cuBLASLt 14.30–81.38 us.
- `gemma_fp8_w8a16_probe.cu`: exact E4M3-to-BF16 conversion followed by BF16 MMA
  preserves activation precision. Selected candidates improve nine synthetic
  projection cells by 1.32–3.29x against the best measured production GEMV grid.
  Maximum selected-candidate relative L2 difference is 6.42e-5; maximum absolute
  difference is 0.015625.

Both pass sampled FP64 reference checks and Compute Sanitizer memcheck with zero
errors. Their adjacent Markdown files document controls, reproduction and results.
Standalone grids and register pressure differ from the persistent interpreter.
Full-model arithmetic, fused GLU, sparse rows and serving require qualification
before either candidate can become a production route.


### Default-cache serving qualification after launch elision

The final `b828a24` runtime passes a serving check with `PLOW_VMM_PREFIX`,
`PLOW_TOKEN_BATCH` and `PLOW_PF_BATCH` omitted. Startup selects VMM prefix caching
automatically; token-batch selection remains enabled but explicitly reports the
missing CUDA executor and uses ordinary execution. Packed prefill is not selected
by this default-cache configuration.

All 12 cold/warm natural completions at 1K/4K/16K match each other and the previous
adaptive runtime. Warm requests have the required cache counts, including 16384
cached tokens for the 16K cases. Eight concurrent cold 16K requests complete; all
eight isolated replays match, followed by successful short-request recovery and
three API checks. This is functional evidence, not a new performance campaign.

The runner stopped its owned server after completion.
`cublaslt-elide-experimental-qualification.json` records source, artifact hashes,
controlled logits, direct-step timing and the default-serving results. Raw
serving artifacts use `plow-bf16-cublaslt-elide-default-*` in the campaign directory.


## Skip the discarded packed-prefill terminal

When a validated compact terminal is available, packed prefill now replaces its
unused final normalization, vocabulary projection, soft cap and two argmax
instructions with no-ops. Their counters and dependency signals remain intact.
The compact terminal still gathers the completing requests' hidden rows and
performs the original normalization and vocabulary math. Switching a bucket
back to ordinary prefill restores the five opcodes and uploads the full changed
instruction range.

The runtime host suite passes 597 tests with 17 ignored. BF16 and FP8 GPU checks
each match all 128 ordinary-reference full-vocabulary snapshots exactly, including
sparse highest slots and reversed completion order. Additional packed→ordinary→
packed checks on the same bucket preserve full logits. Raw logs are
`packed-tail-skip-{bf16,fp8}-gpu.log` and `packed-tail-skip-host-tests.log` in the
campaign directory. This change is within the existing packed-prefix path;
request-latency results follow below.


## Opt-in FP8 tensor cores inside the decode interpreter

`PLOW_NV_FP8_DECODE_MMA=1` now selects a native BF16 MMA implementation with
exact E4M3 weight conversion on Hopper. GEMV uses it for M≥8; fused GLU for
M≥16. Both require K divisible by 256 and the existing 256-thread block.
The instruction's output-slice ownership and activation precision are preserved,
with no new scratch allocation, launch or inter-block reduction. The flag
defaults to zero; rebuilding without it produces the identical baseline cubin.

Final paired synthetic 1K decode-step results, same frozen helper for both
assets, 16 warmups and 64 measured steps:

| Active slots | Baseline mean ms | Candidate mean ms | Reduction |
|---:|---:|---:|---:|
| 1 | 20.983 | 20.954 | 0.1% |
| 8 | 92.263 | 85.761 | 7.0% |
| 16 | 109.139 | 107.570 | 1.4% |

The helper predates the latest packed-serving changes. These are decode-step
measurements, with no CPU builds or other GPU jobs overlapping, not serving
throughput or vLLM wins. Preserving the interpreter's slice geometry loses
much of the standalone prototype's speedup. An initial M=8 tensor-core GLU
regressed, so that shape retains the existing FFMA implementation.

The final probe passes 90 numerical, output-guard and slice-ownership checks;
Compute Sanitizer reports zero errors. The complete interpreter uses 189
registers versus 188, with zero stack/local memory and unchanged shared memory.

Model-level checks cover 1K/16K natural prompts at physical slots 7 and 15,
forcing decode widths 8 and 16. All 256 frames are finite, all four initial
prefill frames are bit-exact, and greedy top-1 agrees in every frame. Logits
are not bit-exact: maximum absolute difference 1.078125, minimum cosine
0.9994801, maximum relative L2 difference 0.0347104. Broad model-quality and
serving qualification remain open; the route is not promoted to a default.

Reproduction and controls are documented in
`runtime/nvidia/experiments/gemma_fp8_persistent_probe.md`. Campaign proofs are
`gemma-fp8-persistent2-qualification.json`, `gemma-fp8-persistent2-step-results.json`
and `fp8-mma2-model-logits-comparison.json`.


### Matched cached-request latency after removing the discarded terminal

The old and new runtimes used identical qualified packed assets for each
precision. This screen used one-token completions on natural 1055/16415-token
prompts, concurrency 1/8, two warmup waves and 12 measured waves per cell.
All 864 measured requests reported the expected cached-token count (1024 or
16384); every measured text matched between runtimes. No CPU builds or other
GPU jobs overlapped the timing. All four owned servers stopped afterward.

Median HTTP request latency, milliseconds:

| Precision | Prompt | Concurrency | Control | Skip discarded tail | Reduction |
|---|---:|---:|---:|---:|---:|
| BF16 | 1055 | 1 | 137.72 | 136.02 | 1.2% |
| BF16 | 1055 | 8 | 592.84 | 565.97 | 4.5% |
| BF16 | 16415 | 1 | 203.94 | 203.93 | <0.1% |
| BF16 | 16415 | 8 | 1201.17 | 1217.40 | -1.4% |
| FP8 | 1055 | 1 | 131.92 | 122.60 | 7.1% |
| FP8 | 1055 | 8 | 607.88 | 593.13 | 2.4% |
| FP8 | 16415 | 1 | 230.14 | 193.54 | 15.9% |
| FP8 | 16415 | 8 | 1186.21 | 1181.72 | 0.4% |

Results are mixed: the largest gain is FP8 at 16K/C1, while BF16 at 16K/C8
regresses. These nearly fully cached, one-token latency measurements do not
establish longer-output throughput or a vLLM win. The CSV is
`gemma31-h100-packed-tail-latency.csv`; raw requests, source and artifact hashes
are recorded in campaign `packed-tail-skip-qualification.json`.


### Prefix snapshot copies and deferred row-zero allocation

Prefix admission now waits until cache lookup before allocating private KV.
Previously `begin_slot` mapped row zero, and `try_attach` immediately unmapped
that allocation on a hit. Execution still maps every row it writes, including
inactive decode rows; packed admission retains its capacity reservation.
Live-only KV keeps its existing initialization.

Sliding-window snapshots use depth-one `cuMemcpy3DAsync_v2` transfers with
one pitched row per attention head. The snapshot layout and stream-drain
contract are unchanged. Gemma's 50 sliding layers now need 100 driver calls
per nonwrapped snapshot, instead of 1600; a wrapped snapshot needs at most
200 instead of 3200. Partial full-KV and FP8 scale copies retain their existing
implementation. The 3D API avoids the allocation-pitch restriction documented
for `cuMemcpy2DAsync`.

An isolated 800 MiB snapshot benchmark alternated 64 measured repetitions per
variant. Median restore time fell from 4.81 to 1.27 ms without wrap and from
8.44 to 1.55 ms with wrap. Copy-only HTTP results were mixed for BF16 and
improved FP8 by 2–7%; that experiment is preserved separately in campaign
`pitched-prefix-qualification.json`.

The combined change was compared with `plowrt-packed-tail-skip`, using the
same qualified native packed assets, natural 1055/16415-token prompts,
concurrency 1/8, one-token completions, two warmup waves and 12 measured waves.
All 864 measured requests reported the expected 1024/16384 cached tokens;
all measured text matched. No CPU builds or other GPU jobs overlapped timing.
All four servers stopped afterward.

Median HTTP request latency, milliseconds:

| Precision | Prompt | Concurrency | Previous runtime | Combined change | Reduction |
|---|---:|---:|---:|---:|---:|
| BF16 | 1055 | 1 | 133.20 | 123.20 | 7.5% |
| BF16 | 1055 | 8 | 563.77 | 488.64 | 13.3% |
| BF16 | 16415 | 1 | 205.55 | 195.66 | 4.8% |
| BF16 | 16415 | 8 | 1202.30 | 1143.99 | 4.8% |
| FP8 | 1055 | 1 | 124.93 | 112.84 | 9.7% |
| FP8 | 1055 | 8 | 551.14 | 461.88 | 16.2% |
| FP8 | 16415 | 1 | 192.43 | 184.66 | 4.0% |
| FP8 | 16415 | 8 | 1191.76 | 1125.93 | 5.5% |

This is a cached one-token latency screen, not a fresh full vLLM comparison
or evidence of longer-output throughput. The full 1K–16K/concurrency-64 goal
remains unmet. Results are in `gemma31-h100-prefix-admission-latency.csv`.

Verification: 597 host tests passed, 18 tests ignored by the host
suite. Explicit H100 tests pass 27 pitched-copy geometries with sentinel
padding and byte-exact round trips. Both BF16 and FP8 pass 272 full-logit
comparisons each, including cached reuse after retirement into swapped slots
and a snapshot at boundary 2112 crossing the ring wrap. The cache test also
asserts that a fresh prefix slot has no mapped rows before lookup. The initial
backend test exposed a pageable-upload/default-stream setup race; its fixture
now synchronizes uploads before the nonblocking copy stream starts.

Default-serving checks omitted prefix, token-batch, packed-prefill and multi-step
enable flags. Both precisions selected prefix reuse automatically and enabled
multi-step quantum 8; CUDA unified batching still reported its explicit
fallback. All 24 cold/warm natural completions at 1K/4K/16K matched the previous
runtime, with every expected warm cache count.

Packed pressure checks completed eight concurrent cold 16K requests for BF16
and sixteen for FP8, followed by exact isolated replays and short recovery
requests. Admission exercised its memory-pressure wait path. Both precisions
passed the three API lifecycle checks (ragged prompts/output limits/slot reuse,
cancellation/recovery, and context rejection/recovery). All owned servers stopped.
Campaign `prefix-lazy-qualification.json` records source and 46 artifact hashes.

The checkpoint request for BF16 weights with FP8 KV is a separate configuration:
the results above use BF16 KV for both weight precisions.

### BF16 weights / FP8 KV runnable checkpoint (2026-09-09)

Saved the complete local checkpoint at
`/opt/dlami/nvme/plow-checkpoints/gemma4-31b-it-h100/bf16-fp8kv-20260909`.
It contains frozen plowrt source `25fb3d7`, regular Hugging Face weight files
from revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, tokenizer/chat template,
all emitted cubins and programs, bundled ELF libraries, launch/workload scripts,
and a SHA-256 manifest. Decode rungs **1/2/4/8/16** and prefill buckets
**128/512/1024** are retained and usable. Context limit is 32768; physical slots
are 16. See [build and run instructions](../docs/runtime/gemma4-h100-checkpoint.md).

FP8-KV ladder validation now checks cache data, per-row scale storage and
reader/writer addressing. Verification passed:

- 600 host tests; 19 hardware/environment tests ignored in the host run.
- 122 full-vocabulary comparisons against an independently emitted widest-only
  FP8-KV build, including sparse/reversed slots, every rung, reset and retirement.
- 240 cached full-vocabulary comparisons against 48 resident-prefix reference
  frames, with identical suffix buckets, at 127/129/513/1057/2113/16417 tokens.
- Six real-text cold/warm pairs at 1K/4K/16K: identical text and expected hits.
- Sixteen concurrent cold 16K requests, sixteen exact isolated replays, recovery,
  and the three API lifecycle checks.
- Packaged launch from `/` with ambient Plow/library settings cleared: no Nix
  libraries mapped; 22 sample requests passed, including 16 concurrent warm
  replays. Test servers stopped afterward.

The launcher selects FP8-KV cubins explicitly and enables prefix reuse. This
configuration uses ordinary single-segment prefill and disables multi-step;
segmented/packed FP8-KV prefill is unavailable. CUDA still uses the serving
fallback executor. Cold/warm bucket changes can alter logits: maxima 0.84375
at 129 tokens and 1.25 at 513 tokens were reproduced without a cache attach.
Cache restoration itself was bit-exact against the matching suffix computation.
All six natural completions differed from the BF16-KV reference; text agreement
is not a quality score. Broader quality and sustained-load qualification remain
open. Serving checks overlapped checkpoint copying/hashing and are functional
evidence, not performance measurements. No vLLM win or off-instance backup is
claimed. The checkpoint's `evidence/qualification.json` records the detailed proof.

### Experimental native FP8 WGMMA decode (2026-09-09)

The default-off `PLOW_NV_FP8_DECODE_WGMMA=1` path quantizes BF16 activations
inside each CTA and uses Hopper E4M3 tensor cores for M8/M16 projections.
It uses FP8 weights and BF16 KV; the saved BF16-weight/FP8-KV checkpoint is
unchanged. [Implementation and evidence](../runtime/nvidia/experiments/gemma_fp8_w8a8_persistent.md).

Matched isolated full-model decode steps at 1024 context, 16 warmups and 64
measurements: B1 **21.012→21.177 ms** (+0.79%), B8 **92.319→43.465 ms**
(2.124× faster), B16 **109.235→60.527 ms** (1.805× faster). No CPU builds or
other GPU jobs overlapped these measurements. These are direct steps, not HTTP
throughput or vLLM wins.

All 384 M1/M2/M4 full-vocabulary fallback frames were bit-exact. M8 and M16
matched each other on 128 frames; each matched the native top prediction on
125/128. Maximum logit difference was 4.3125, worst cosine 0.9954132, and
worst relative L2 0.1604071. Activation precision changes with the rung, so
model-quality qualification remains open. The production-dispatch probe passed
171 checks plus 12 fallback checks; bounded memcheck reported zero errors.
Default-off cubin bytes remain identical. Full proof is in campaign
`fp8-w8a8-model-qualification.json`; this option remains off by default.


### Shared scheduling controls and unified adapter qualification (2026-09-09)

Prefill batching, interleave/chunk caps, no-chunk/no-interleave and deferred
decode controls now belong to shared runtime configuration. Both AMD and NVIDIA
read those fields; CLI names, environment variables and default values are
preserved. The common token planner no longer allocates an ordering vector on
every step. Eighteen planner tests, 638 runtime host tests and 17 configuration,
mux, co-serving and preemption integration tests passed. All-target checks passed
with CUDA+HSA together and each GPU backend separately. No AMD hardware run was
performed on this H100 instance.

The NVIDIA execution adapter now consumes the common token plan, preserving
logical output ownership and checking slot generations before planning and
commit. It borrows input slices, commits host frontiers only after the body and
compact output complete, appends prefix history once, and publishes snapshots
only for completed prefill spans. It reuses existing packed-prefill bodies and
supports intermediate chunks with no sampled output.

H100 BF16-weight/B8 and FP8-weight/B16 assets, both with BF16 KV, each passed
32 full-vocabulary comparisons against isolated execution on the same resident
prefix with the same compact output tail. Coverage includes a cached 16K suffix
combined with 1K decode, reversed sparse slots, frontier/history checks,
intermediate zero-output chunks, final output without replay, stale generations
and duplicate refusal. Campaign logs: `unified-compact-control-{bf16,fp8}-gpu.log`.

The ordinary and compact output tails use different RMSNorm reductions. Before
matching the reference tail, the BF16 test failed at step 7 with maximum logit
difference 0.0625; repeating the ordinary reference reproduced it exactly. Using
the compact tail on isolated requests reproduced the difference without batching:
2/32 BF16 and 1/32 FP8 frames differ from the ordinary tail, maximum 0.125.
This establishes batching parity for the selected computation, not equivalence
between output algorithms or model-quality qualification.

At this adapter-only milestone it was not yet wired into the serving mux and
accepted greedy selection only (superseded by the serving milestone below). Unified serving default promotion, stochastic sampling, SLO-aware
admission and packed FP8-KV assets remain pending. The immutable checkpoint is
unchanged. These functional checks are not throughput measurements.

Main `38e3316` is merged. Merge checks still expose unchanged-main failures:
the tokenizer API special-token expectation (also blocks `nix build`), the
kernelcaps RowGather guard/callee parser test, the unregistered legacy AMD
benchmark script, and repository-wide formatting. The branch remains a draft
for merge preparation until the relevant readiness work and checks are resolved.


### Default unified serving on qualified packed assets (2026-09-09)

Qualified Hopper packed-prefill assets now select prefix reuse and unified
serving with `PLOW_VMM_PREFIX`, `PLOW_PF_BATCH` and `PLOW_TOKEN_BATCH` unset.
The mux combines existing decode inputs with admitted prompt spans, reserves
compiled row capacity for decode, and uses the shared fair prefill planner.
It delivers compact outputs by logical slot and consumes each decode feed once.
A failed combined launch retires all participating slots. Pure decode retains
the ordinary adaptive ladder. Unsupported configurations retain ordinary
execution; startup reports the capability actually loaded.

Both BF16-weight/B8 and FP8-weight/B16 H100 assets use BF16 KV. Each passed
12-request functional serving screens covering warm 1K/16K cache hits, overlapping
prefill/decode, stochastic sampling, exact output limits, cancellation and context
rejection/recovery. Debug logs show 19 BF16 and 25 FP8 committed mixed batches.
A larger screen completed 77 requests per precision, including 64 queued requests
and recovery, with 87/72 mixed batches. All 32 copies of each pressure prompt
produced identical four-token text. These are 64 queued requests served through
physical B8/B16, not 64 resident sequences.

The first BF16 burst hit the default 32-request ingress bound and returned 429;
that failed run is preserved. The successful 64-request screens explicitly use
`--max-queued-requests 256`. All serving screens use `--slo-ms 600000` and diagnostic
logging, so they do not prove deadline compliance or production throughput.
Long-prefix cache hits under pressure were 24/32 BF16 versus 4/32 FP8; bounded
cache retention remains a performance concern.

`PLOW_TOKEN_BATCH=0` passed the same 12-request BF16 functional screen, retained
prefix hits and produced zero mixed commits. The API's existing host sampling
path handles stochastic and penalty-adjusted compact outputs. Restoring the
device-sampling fast path for mixed stochastic rows remains pending.

Host validation: 638 passed, 20 ignored; 17 configuration/mux/co-serving/preemption
integration tests passed. All-target checks passed for CUDA+HSA together and each
backend separately. With selector overrides unset, the BF16 GPU test passed 32
full-vocabulary matching-computation frames plus intermediate/final/stale/duplicate
checks. Campaign `unified-serving-default-qualification.json` records the frozen
binary hash and the five final HTTP campaigns (190 successful requests).

This enables actual default dispatch on the qualified H100 assets. AMD prefix
integration, fresh compiler defaults, packed FP8-KV assets, ordinary-decode quality
comparison, cache retention, sampling performance and broader production
qualification remain open. The sealed BF16-weight/FP8-KV checkpoint is unchanged.
