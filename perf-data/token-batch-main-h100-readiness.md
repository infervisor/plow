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
| Host verification | CUDA + HSA library suite: 584 passed, 14 ignored, zero failures. |
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
The full concurrency matrix and production qualification remain pending. Prefix caching
remains opt-in.

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
cached TTFT remain performance targets. No full-matrix win or production-ready default
prefix selection is established.

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
