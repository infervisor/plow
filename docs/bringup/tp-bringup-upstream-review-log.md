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
| 9 | 2026-09-11 | cc46cad6 | should-fix | runtime/tests/packed_flash_sm90_correct.cu:109-111,171 | Under `PLOW_NV_FA_WGITEM` the test sizes smem with `FA_SM90_WGI_FLOATS` (208,896 B) even for the paired body; production claims `FA_SM90_GQA2_PAIR_FLOATS` (141,312 B, `interp_sm120.cu:749`). Hand count says the pair fits (139,264 + align), so latent, not live | open — test should launch with the pair arena when `PLOW_NV_FA_GQA2_PAIR` |
| 10 | 2026-09-11 | 34c58c80 | should-fix | scripts/gemma4_ladder_campaign.py:537-556 `gate_kernels` | `minimum_weighted_speedup` (1.01) applied per (phase,rung,topology,family,kv_length) group → a GEMM-only candidate fails on every attention group unless the gate is lowered to ≤1.0 | open — gate on campaign total or per touched family |
| 11 | 2026-09-11 | 34c58c80 | should-fix | scripts/gemma4_ladder_campaign.py:482-534 | `reduce_serving`/`run_serving`/`campaign_plan`/`validate_spec`/`main` untested (kernel side is); vLLM arm mandatory, `arch=="sm90a"` + 16384 ctx hard-wired, missing baseline cell → bare `KeyError` | open — fixture test for serving records; optional vLLM arm |
| 12 | 2026-09-11 | 0e52a559 | note | interp_sm90a_pfgemm_w8a16_m1.cu:86 vs interp_sm120.cu:1286 | Role body is `gemv_rows_fp8<1>` (FFMA f32 warp-sum) where the interpreter arm is `d_gemm_fp8` (dequant-to-bf16 mma): same math, different rounding → bitwise drift on M1 prefill vs interpreter | note — acceptable if blessed on hardware; H100 |
| 13 | 2026-09-11 | 0e52a559 | note | exec/gpu.rs:4062-4208 | Role 9 has no runtime opt-out: role in packet + object present → mandatory; missing object/symbol/marker fails engine start (consistent with MXFP4/GEMV512 roles) | note — document in flags-reference |
| 14 | 2026-09-11 | (pre-existing) | note | packet/devbuild.rs:3015 `decode_rung_lo` | A packet with no decode rungs reports its last prefill rung as decode; both prefill roles then skip it (`prog_t=[1]` alone → W8A16 role never tagged). No panic | note |
| 15 | 2026-09-11 | c1d34e53 | perf | exec/amd/shared_prefix.rs:329 + vmm.rs pre-mapper | `enable_block_pool` per cache group → 3 pools/rank, 6 threads/rank (48 at TP8); `advise()` is never called so the pre-mapper idles and every block boundary is mapped synchronously in `prefill_prepare` (~69 µs × tracks per crossed block, ~10 ms for 61 tracks × 3 groups) | open — call `advise(slot, frontier)` after each chunk, or accept |
| 16 | 2026-09-11 | c1d34e53 | note | shared_prefix.rs:486,508-513 | Publish boundary is `rows/32*32` but the pool block is the VMM granule; `publish_completed_chunk` fires mid-prompt only when `frontier % block_rows == 0` for every group, so with a 2 MiB granule (krot block_rows 16384) intermediate publishes almost never happen and a completed prompt pays one large synchronous snapshot per group (tail up to `block_rows-1` rows × tracks) | open — measure REC_GRANULE on MI300X; consider a sub-granule `block_hint` |
| 17 | 2026-09-11 | c1d34e53 | note | shared_prefix.rs tests | Eviction under cap is untested (every `SharedPrefix` test passes `cache_cap = 0`); publish-failure path and the unlocked `publish_at` under concurrency untested; only the `#[ignore]`d `compiled_glm_cache_layout` checks a real packet | open — add a cap-bounded eviction test |
| 18 | 2026-09-11 | c1d34e53 | note | shared_prefix.rs:61,83 vs packet/dev.rs:92-110 | `HeadNormRope`/`RmsNorm` `i[7]` is checked against `context` but undocumented for op 3 in dev.rs | open — document i7 |
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
| F12 | c1d34e53 | serve/engine.rs `prefill_chunked_at_most` | cache-side `publish_shared_prefix` failure (snapshot OOM after eviction, radix collision, rejected boundary) aborted a prefill whose KV was already correct; CUDA warned and continued | this commit (warn unless `is_fatal()`) |
| F13 | c1d34e53 | exec/amd_tp.rs `shared_prefix_enabled` | group predicate was `any` while `attach_shared_prefixes` requires `all` → a per-rank VMM mismatch would fail every admission | this commit |

## Artefact policy (applied on every merge)

Raw measurement files pushed upstream are removed here before the branch goes to main:
`runtime/bench/**/*.{json,jsonl,csv}`, `perf-data/**/*.csv`, per-probe write-ups
(`runtime/nvidia/experiments/*.md`, `RESULTS.md`, oracle notes) and the per-campaign
`runtime/bench/amd/*/README.md`. Kept: inputs the code reads
(`glm_fold_tail/gemm-selected.json` ← `amd_mla_fold.rs`; `tuning/**/*.jsonl` ← tunedb;
`runtime/ubench/mfma_shape_results/summary.csv` ← `collect_occ1.sh`; `*.example.json`
templates), `scripts/*.json` build contracts, `runtime/amd/glm_*_gfx942.json` pinned kernel
specs, `docs/schemas`. `.gitignore` now blocks re-adding them locally; tracked files arriving
through a merge still need the sweep (gitignore does not apply to tracked paths).

| Sweep | Removed | Kept (code-read) |
|---|---|---|
| 2026-09-11 (initial) | 23 `perf-data/*.csv` (H100 campaign tables), 48 `runtime/bench/amd/*/mi300x-*.json` + `decode-selected/lean-compiled-audit/validation.json`, 6 `runtime/nvidia/experiments/*.md`, `d6_xreduce_attnres_oracle.md`, `lean_moe_combine_ref/RESULTS.md` | `gemm-selected.json`, `vllm_capture.example.json`, `summary.csv` |

| 2026-09-11 (amd md) | 26 `runtime/bench/amd/*/README.md` campaign write-ups (user decision) | `runtime/bench/README.md` (tooling index), `docs/amd/*.md` (referenced by recipes / code comments) |

## H100 punch list (no NVIDIA GPU on the review host — everything here is unrun)

### Correctness first — run before any number is trusted
| # | Item | Where | Done when |
|---|---|---|---|
| H1 | Build and run the four sm90 harnesses that CI never ran: `gemma3_norm_sm90`, `packed_flash_sm90_correct`, `packed_flash_fp8_sm90_correct`, `packed_kv_padding_sm90` (CMake targets under `PLOW_CUDA`, guarded) | runtime/tests/*.cu | all pass on H100 |
| H2 | Fix the paired-GQA2 test arena: under `PLOW_NV_FA_GQA2_PAIR` the test must launch with `FA_SM90_GQA2_PAIR_FLOATS` (141,312 B), not `FA_SM90_WGI_FLOATS` (208,896 B) — today it passes while production claims a smaller arena | runtime/tests/packed_flash_sm90_correct.cu:109-111,171 | test launches with the production arena and passes |
| H3 | Bless the W8A16 M1 prefill role numerically: `gemv_rows_fp8<1>` (FFMA, f32 warp-sum) vs the interpreter's `d_gemm_fp8` (dequant-to-bf16 mma) — same math, different rounding | interp_sm90a_pfgemm_w8a16_m1.cu:86 vs interp_sm120.cu:1286 | rel-L2 vs interpreter recorded; greedy agreement on a real prompt set |
| H4 | Run this branch's CUDA host-path changes with a driver: `publish_at` alloc+fill outside the pool lock, the release-mode snapshot-layout check with `begin_seq` rollback, the unified-path `decode_progress`, the `filter_map` request builder, the shared `slab_carve` (zero-byte tensors now get distinct addresses) | memory/vmm.rs, exec/gpu/prefix.rs, serve/mux.rs, exec/gpu.rs | `cargo test -p plowrt --features cuda --lib`, `tests/gpu_consume_prompt.rs` with `PLOW_GPU_TEST=1`, the `#[ignore]`d tests in exec/gpu/{decode_rung_tests,token_batch}.rs, plus a c8 serve with prefix hits/misses |
| H5 | Batched DSA decode on CUDA: `d_index_select_sm120` ignores the per-row offset `i[3]`; the emitter now refuses `rows>1 && dsa` off gfx942. If GLM batched decode on H100 is wanted, add the offset to the kernel and lift the refusal | interp_sm120.cu:2112, devgen/mla.rs | kernel takes `i[3]`; refusal replaced by an nvcc requirement |
| H6 | `retire_slot(slot, tick_fault.is_none())` publishes disconnected clients' partial KV into the 4 GiB prefix cache | serve/mux.rs | per-tick disconnect record; skip publish for closed clients |
| H7 | Metrics cost on the tick thread: `RequestMetrics::token` is one `Instant::now()` + histogram + 2 atomics per produced token | obs/serving.rs:344 | measured at C64; one `Instant::now()` per tick if it shows |

### Default-flip A/Bs owed (one variable each, Gemma-4 BF16 and FP8, c1/c8, 1k and 16k, cache-miss traffic)
| # | Variable | Why |
|---|---|---|
| H8 | `--token-batch=false` vs default | replaces the decode step with `token_batch_step` on sm_90a packets with packed metadata; only "functionally validated" |
| H9 | `--prefix-cache=false` / `PLOW_VMM_PREFIX=0` vs default | measured wins are admission-side only; the recorded negative is packed 16k/c4 TTFT 1732→2642 ms (+52%) on FP8 — re-measure |
| H10 | `--vmm-cache-memory-utilization` 0.05 vs `--vmm-cache-mib 4096` | identical on 80 GiB; on H200 (141 GiB) the fraction gives 7 GiB — check hit rate and VRAM headroom |
| H11 | hd256 FA object main-vs-branch cubin (0f626502 stage-state bitfields) | expected neutral/positive; unmeasured |
| H12 | Decode objects lose T17 warp-per-row norm at rows≥32 | only reachable with native-TC B32; measure once |

### Opt-ins to measure (all default off today)
| # | Knob / object | Recorded so far |
|---|---|---|
| H13 | W8A16 M1 prefill role (0e52a559) | none |
| H14 | Paired GQA2 hd256 prefill attention (`PLOW_NV_FA_GQA2_PAIR`, `PLOW_PF_SEG_FA256_GQA2`) | none |
| H15 | hd512 prefill WGMMA (`PLOW_BUILD_PFATTN_WGMMA=1`) | none; default mma.sync arm is 2.1× off FA3 |
| H16 | Decode GEMV native TC (`--emit-decode-native-tc`) and adaptive cuBLASLt | 35.4 vs 32.7 µs B8; cuBLASLt c4 TPOT 54→35 ms but c1 −6% |
| H17 | FP8 decode WGMMA (`PLOW_NV_FP8_DECODE_WGMMA`) | 2.1× step B8 but logits cos 0.9954 — not qualified |

### The perf gaps that are not knobs (kernel work, ranked by gap × share)
1. Decode attention hd512: 5.5–7.9× vs vLLM attention lib — CUDA-core softmax body, no TMA, 208 regs (`op_attention.cuh:588`).
2. Prefill attention hd256: 4–5× — 2 stages, both warpgroups recompute S, no producer/consumer.
3. Decode GEMV bf16 B8/B16: 1.6–4.5× vs cuBLASLt — FFMA ladder, `GV_MM_MAX=8`.
4. Megakernel slot cap: TPOT flat 76.7 ms c8→c64 vs vLLM 24.9→42.4.
5. Prefill GEMM: 175/219 TF/s uniform vs ~450 warp-specialized (ws384 already served for Gemma-4).

### Housekeeping on the H100 box
- `tuned_tile_selection`: the `nvidia/sm_90a/h100-nvl` tunedb cell may be stale against the kernel-source digest as well; re-run the decode campaign + `plowc tune ingest` if `plowc tune status` says so.
- Run the self-hosted CI job's `cargo test --workspace` there (the hosted job only checks the CUDA features).

## Batches merged

| Merged at | Upstream range | Merge commit | Conflicts | Checks |
|---|---|---|---|---|
| 2026-09-11 | 0e52a559 (d5f320df, 0e52a559) | 2f7ed6d3 | none | check ws + cuda,hsa; devgen/plow-asset tests |
| 2026-09-11 | cc46cad6 (c1d34e53, b4bf5cab, cc46cad6) | 34376b3c | exec/amd.rs ×2 (slab sizing, `prefix_cache_capable`) | check ws + cuda,hsa; cpu suite; 739 hsa/cuda lib tests; plow-asset |
| 2026-09-11 | 34c58c80 | bca27b8d | none | new py test 9/9; devgen check + dense_cublaslt tests |
| 2026-09-11 | 55ce86d7 (64b15dc0, 55ce86d7) | 701fc7f0 | none | check cuda,hsa; `metrics` test; 96 obs/serve lib tests; campaign tests 11/11. Reviewed: model-scoped Prometheus metrics (per-model `Arc<Metrics>`, unloaded models fold into the process totals), ladder-campaign refuses incomparable workloads / concurrent latency regressions |
