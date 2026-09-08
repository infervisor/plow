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
| Host verification | CUDA + HSA library suite: 586 passed, 14 ignored, zero failures. |
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
Second-chance eviction considers all unpinned snapshots when no radix leaf can be reclaimed.
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
