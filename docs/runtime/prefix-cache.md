# Prefix caching and unified batching

Both selectors default on for AMD and NVIDIA. Runtime selection also requires
compatible assets; enabling a selector does not qualify an unsupported execution
path. Check startup selection messages and actual token-batch dispatch logs.

| Control | Default | Effect |
|---|---|---|
| `--prefix-cache` / `PLOW_PREFIX_CACHE` | `true` | Permit prefix reuse on either backend. |
| `--token-batch` / `PLOW_TOKEN_BATCH` | `true` | Permit unified prefill/decode dispatch. |
| `--vmm-cache-memory-utilization` / `PLOW_VMM_CACHE_MEMORY_UTILIZATION` | `0.05` | Prefix block/snapshot budget per engine as a fraction of device memory, in the unit of vLLM's `--gpu-memory-utilization` (4 GiB on an 80 GiB H100); zero allows OOM-driven eviction only. |
| `--vmm-cache-mib` / `PLOW_VMM_CACHE_MIB` | unset | Explicit budget in MiB, overriding the fraction; zero allows OOM-driven eviction only. |

Use `--prefix-cache=false` or `PLOW_PREFIX_CACHE=0` to disable reuse. On NVIDIA,
this also overrides `PLOW_VMM_PREFIX=1`. `PLOW_VMM_PREFIX=0` remains a
backend-specific opt-out. Disable unified dispatch separately with
`--token-batch=false` or `PLOW_TOKEN_BATCH=0`. Explicit fusion takes precedence
over unified batching.

NVIDIA uses the shared VMM prefix cache on compatible assets. AMD currently
reuses a prefix within the same physical sequence slot, retaining flat KV and
snapshotting recurrent state, sliding-window rows and any FP8 KV scales. Its
snapshots are bounded and evictable; an allocation OOM skips caching. Prefix
publication keeps a final input row available for sampling. Cancellation of a
miss invalidates its old prefix before new KV writes.

AMD prefix reuse is unavailable with its current lazy VMM allocation path,
unknown KV layouts, or a snapshot larger than the configured budget. Explicit
fusion retains its existing execution path. Tensor-parallel prefix snapshots
require all ranks; AMD unified batching still requires its single-GPU packet
and code-object capabilities. All compiled decode and prefill rungs remain
available to ordinary execution.

The GLM-5.3 MI300X TP8 screen selects prefix caching with the default settings.
Unified batching is requested by default but declines with
`token batching does not support tensor parallelism`; that run uses ordinary
execution. Enabling the selector does not remove this implementation limit.

Fresh single-GPU Hopper builds emit packed-request metadata by default for
BF16 weights and BF16-activation FP8 weights (`--fp8` or `--w8a16`) with BF16 KV.
`--emit-packed-prefill=false` disables that emission. Activation-FP8 packing
(`--w8a8`) remains explicit opt-in; packed FP8-KV execution is not qualified.
AMD retains its own packed-program contract and does not receive NVIDIA request
metadata, including when `--emit-packed-prefill=true` is selected.

Host tests and emitted AMD BF16/FP8-KV packet checks cover the snapshot layout,
window wrap, configuration and scheduling invariants. H100 tests cover actual
BF16/FP8-weight dispatch. AMD device correctness/performance and sustained
production qualification remain pending; build compatibility alone is not that
evidence. The H100 readiness report lives with the campaign's raw `perf-data`, which is
kept out of source control.
