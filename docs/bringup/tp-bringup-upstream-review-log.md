# tp-bringup-mi300x → worktree-tp-merge: upstream review log

Running tracker for every upstream batch merged into `worktree-tp-merge`. One row per potential
issue; status moves `open → fixed <commit>` or `open → wontfix (reason)`. Companion to
`tp-bringup-mi300x-merge-review.md` (the initial full review).

Severity: **blocker** (must fix before main), **should-fix** (correctness/robustness, land soon),
**perf** (unmeasured or suspected regression), **note** (hygiene/docs).

## Open

| # | Merged | Commit | Sev | Where | Issue | Status |
|---|---|---|---|---|---|---|
| 1 | 2026-09-11 | 33a5b7bf, 3ca64e93 | perf | config.rs:82-102 | `token_batch`, `prefix_cache` default-on for all backends with no isolated A/B on either GPU; one recorded negative (H100 packed 16k/c4 TTFT +52%) | open — A/B owed (H100 later, MI300X available) |
| 2 | 2026-09-11 | c1d34e53 | perf | exec/amd.rs `SharedPrefix::new` bring-up | Cross-request AMD MLA prefix sharing is a **new default-on GLM/MI300X route** (`PLOW_AMD_SHARED_PREFIX` auto). Qualification on record: 18/18 retrieval at C20. No throughput/TTFT A/B vs the slot-local route | open — A/B on this host |
| 3 | 2026-09-11 | c1d34e53 | note | exec/amd/shared_prefix.rs `copy_snapshot` | `Vec::new()` for copy pairs per attach/publish (per request, not per step) | note — acceptable, watch if it ever moves onto the step path |
| 4 | 2026-09-11 | cc46cad6 | perf | op_attention_sm90.cuh GQA2 pair | Opt-in only (`PLOW_NV_FA_GQA2_PAIR`, `PLOW_PF_SEG_FA256_GQA2` default off); kernel correctness/perf unmeasured here | open — H100 |
| 5 | 2026-09-11 | 0e52a559 | perf | interp_sm90a_pfgemm_w8a16_m1.cu | Opt-in W8A16 M1 prefill role; unmeasured here | open — H100 |
| 6 | 2026-09-11 | (branch) | should-fix | mux.rs `retire_slot(slot, tick_fault.is_none())` | Disconnected clients' partial KV is published into the prefix cache (cache pollution, not corruption) | open — needs a per-tick disconnect record |
| 8 | 2026-09-11 | 55ce86d7 | perf | obs/serving.rs:344 `RequestMetrics::token` | Per produced token per slot: one `Instant::now()` + 21-bin histogram observe + 2 relaxed atomics (~50-100 ns). A few µs per tick at C64 on the tick thread; `model_metrics()` (RwLock write + alloc) is per dispatcher spawn, not per request | note — measure at high concurrency on H100; consider one `Instant::now()` per tick shared by slots |
| 7 | 2026-09-11 | (main) | note | devgen/tests/tuned_tile_selection.rs | 5 tests red on main and branch: tunedb records stale vs kernel-source digest; gfx950 cell cannot be refreshed here | open — re-run gfx942 campaign; gfx950 needs MI350X |

## Fixed on this branch

| # | Commit(s) | Where | Issue | Fixed in |
|---|---|---|---|---|
| F1 | 84ce7840 (main) | .github/workflows/build.yml:105 | `--lib dist::` plain scalar → workflow did not parse, zero CI jobs | b1626c5c |
| F2 | 97e1a7f3 | devgen/mla.rs | raw `std::env::var` knobs bypassing EmitConfig (`no_raw_env_reads` red) | f6a06e69 |
| F3 | 836337f7 | kernelcaps/sweep.rs test | W8A16 static_assert made PGM90_BN/BK `Asserted` | f6a06e69 |
| F4 | f5f15dd2 | devgen/mla.rs:3866 | CUDA batched DSA `IndexSelect` has no row offset; emit now refuses unless gfx942 | cb420a08 |
| F5 | 704a0518 | memory/vmm.rs `publish_at` | pool lock held across snapshot alloc + D2D + sync | cb420a08 |
| F6 | baebe32d | serve/mux.rs unified path | `decode_progress` never reported; `expect()` on vanished slots | cb420a08 |
| F7 | (branch) | gpu/prefix.rs | snapshot layout check was `debug_assert` only | cb420a08 |
| F8 | (branch) | gpu.rs vs amd.rs slab carve | CUDA carved `bytes`, AMD `bytes.max(1)`: zero-byte tensor aliasing | cb420a08 (`memory::slab_carve`) |
| F9 | 676f98b4 | build_gfx942.sh / kimi-k3 recipe | unset `PLOW_DECODE_TIERS` auto-built rungs that explicit `[[objects.lowrung]]` then overwrote | cb420a08 |
| F10 | (branch) | config.rs | 4 GiB fixed prefix-cache cap regardless of device | 48e71585 (`--vmm-cache-memory-utilization`) |
| F11 | c1d34e53 | exec/amd.rs bring-up | still called removed `prefix_cache_mib()` after merge | 34376b3c |

## Batches merged

| Merged at | Upstream range | Merge commit | Conflicts | Checks |
|---|---|---|---|---|
| 2026-09-11 | 0e52a559 (d5f320df, 0e52a559) | 2f7ed6d3 | none | check ws + cuda,hsa; devgen/plow-asset tests |
| 2026-09-11 | cc46cad6 (c1d34e53, b4bf5cab, cc46cad6) | 34376b3c | exec/amd.rs ×2 (slab sizing, `prefix_cache_capable`) | check ws + cuda,hsa; cpu suite; 739 hsa/cuda lib tests; plow-asset |
| 2026-09-11 | 34c58c80 | bca27b8d | none | new py test 9/9; devgen check + dense_cublaslt tests |
| 2026-09-11 | 55ce86d7 (64b15dc0, 55ce86d7) | 701fc7f0 | none | check cuda,hsa; `metrics` test; 96 obs/serve lib tests; campaign tests 11/11. Reviewed: model-scoped Prometheus metrics (per-model `Arc<Metrics>`, unloaded models fold into the process totals), ladder-campaign refuses incomparable workloads / concurrent latency regressions |
