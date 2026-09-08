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
| Host verification | CUDA + HSA library suite: 582 passed, 14 ignored, zero failures. |
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
and invalid publication. This path has not yet been validated on H100; the measurements
above describe the preceding runtime. Prefix caching remains opt-in pending qualification.

Raw artifacts: `/opt/dlami/nvme/tmp/gemma31-glm53-h100-20260908/`.
Host test log: `/tmp/plow-token-batch-readiness-tests-final.log`.
Final lifecycle test log: `/tmp/plow-prefix-retire-tests.log`.
Current partial-prefix host test log: `/tmp/plow-subblock-library-tests.log`.
CPU integration log: `/tmp/plow-token-batch-cpu-integration-tests.log`.
The compact-tail fixture needed the new optional worker-pinning argument before it could run.
