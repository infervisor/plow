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
| 19 | 2026-09-11 | d60bdeb2 | perf | exec/amd.rs load (`need_mla_small`), exec/amd/segment.rs `small_mla_segments`, build_gfx942.sh | **New default route on gfx942**: pure MLA prefill segments of rungs <2048 now dispatch to a separate 8-wave `interp_mla_small` object whenever `PLOW_MLA_PF_V2` is on (the production default) — not gated by FP8 KV. Unmeasured vs the shared prefill object; and a missing object is a hard load error (`"small MLA segments require …"`), so every existing gfx942 object dir (frozen serving sets included) must be rebuilt | open — A/B GLM TP8 prefill at 128/512/2048 rows; note the rebuild requirement in the recipe docs |
| 20 | 2026-09-11 | d60bdeb2 | note | scripts/obj_baseline_gfx942.json | Three new objects (`interp_mla_small`, `interp_mla_small_fp8kv`, `interp_mla_split_fp8kv`) have no contract row, so `asm_audit.py --contract` skips their VGPR/LDS/occupancy budgets (only universal rules apply) | open — `--bless` them on the next build |
| 21 | 2026-09-11 | d60bdeb2 | perf | mla.rs `glm_small_pf_split_cap` + `interp_mla_split_fp8kv` | FP8-KV-only small-rung split (fj[2] = capacity ≤32) with matching `MlaMergeFold`; opt-in via `PLOW_GLM_FP8_KV`; unmeasured | open — measure with the FP8 KV recipe |
| 22 | 2026-09-11 | 5c26d90f | perf | exec/amd_sparse_mla.rs (`pack_fp8_single`), runtime/amd/mla_sparse_adapter.hip | Single-pass GLM sparse prefill pack: engages automatically for FP8 rows ≥512 whenever the adapter object was built with `--single-pass` (exports `plow_mla_sparse_single_abi_1`). Opt-in only through the object build, but once built it flips the route with no measurement recorded in the commit; kernarg ABI pinned to 104/360 B | open — measure vs the two-split path; the ignored `sparse_mla_single_pass_hsa` test needs a lease |
| 23 | 2026-09-11 | 86e76151 | **blocker if NVIDIA GLM DSA is served** | runtime/nvidia/op_dsa.cuh (`d_index_select_sm120` clamps `len = min(kv_len[row], len_max)`, writes `ib[0..len)`) vs runtime/nvidia/op_mla.cuh:181,189,211 (gather reads a fixed `top_k` span) | A request with `kv_len ≤ top_k` (2048) makes the gather consume `top_k − kv_len` stale indices → wrong attention or out-of-range latent-cache row. AMD clamps the gather span (`op_attention.h:2352`); the new interp comment claims the gather derives the live count but it does not | open — clamp the gather span to `min(top_k, kv_len[b])` or sentinel-fill `ib[len..top_k_max)`; **H100 punch list H0** |
| 24 | 2026-09-11 | 86e76151 | perf | exec/gpu.rs `inferred_segment_policy` + asset/devblob.rs `seg_classes_with` | Segment classing is now inferred from the packet on the default H100 SegPf path: any TMA-mapped pure-GEMM segment sets `pure_mode=1`, which routes every light-op segment (norm/rope/quant) to the fat flash object instead of the lean occ-2 GEMM object. Correct either way; per-chunk time changes, unmeasured | open — measure on the gemma4 ladder before merge; or infer only when the lean object exports a GEMM_ONLY marker |
| 25 | 2026-09-11 | 86e76151 | note | exec/gpu.rs `prefill_segment_kernel` vs roles-only path | MXFP4 MoE prefill role launched at `role.grid` (4× grid) on the segmented path, `self.grid` on the roles-only path | open — confirm intent, use `role.grid` on both |
| 26 | 2026-09-11 | 86e76151 | note | runtime/CMakeLists.txt:176-192 | the four sm_90a harnesses are in the default `all` target for every `PLOW_CUDA` configure (a `static_assert` in one breaks the runtime build on sm120 boxes) | open — `EXCLUDE_FROM_ALL` |
| 27 | 2026-09-11 | 86e76151 | note | exec/gpu/cublaslt.rs:381 | every failing cuBLASLt prefill-segment condition now reports the same "invalid packet-declared projection segments" | open — keep phase-specific messages |
| 28 | 2026-09-11 | 5b263108 | note | devgen/mla.rs (`d.i = [.., 64, 2, ..]`), exec/amd_moe_aiter.rs `Mode::SortedBf16` | emitter default flip: the GLM sorted A8 MoE prefill output is now rounded to BF16 before `MoeCombinePf` (mode 2) instead of carried as FP32 (mode 0). The cross-expert accumulation was already BF16 in-kernel, so the expected delta is one extra rounding of the routed sum, but it is an unmeasured numerics change on the default path. gfx942-only op, no CUDA surface | open — MI300X A/B of GLM prefill logits mode 0 vs 2 before merge to main |
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
| F14 | d60bdeb2 | devgen/mla.rs:265-266 | `glm_small_pf_split_cap` read `PLOW_MLA_PF_V2`/`PLOW_UNISEG` raw → `no_raw_env_reads` red again; `uniseg` now via `EmitConfig`, `PLOW_MLA_PF_V2` on the audit's ALLOWED list (it mirrors devbuild's raw read) | this commit |
| F15 | d60bdeb2 (exposed) | devgen/mla/glm_tests.rs | 15 of 40 GLM tests failed in parallel runs, 0 single-threaded: 24 tests read the global `EmitConfig` without `env_guard()` while the new test's `EnvScope` resnapshots it. Guard added to every such test (two that call a guarded helper are left alone: non-reentrant mutex) | this commit |
| F16 | (branch) | serve/mux.rs | disconnected clients' partial KV was published into the prefix cache | upstream 86e76151: per-tick `disconnected` record → `retire_slot(slot, tick_ok && !disconnected)` |
| F17 | f5f15dd2 / cb420a08 | devgen/mla.rs, runtime/nvidia/op_dsa.cuh | CUDA batched DSA had no row offset; this branch refused it at emit | upstream 86e76151 adds `batch_row`/`kv_len` to `d_index_select_sm120` behind `PLOW_DSA_DECODE_BATCH` with an nvcc requirement + header define; the emit-time refusal is lifted here (this merge). See #23 for the gather-side gap it exposes |
| F18 | 86e76151 (exposed) | devgen/mla/kimi_k3.rs | 1 of 26 kimi tests flaked in the full devgen run (unguarded global-config readers vs a new `EnvScope` test) | this merge: `env_guard()` on the 18 unguarded kimi tests; devgen 423/423 ×2 |
| F19 | 86e76151 | exec/gpu.rs GQA2 object auto-load | the HD256/GQA2 object was auto-loaded whenever it sat in the object dir and then refused any non-packed prefill packet at engine load (a file the operator never asked for killed serving) | this merge: auto-load only for packed BF16 KV metadata (warn otherwise; the policy clears itself); an explicit `PLOW_PF_SEG_FA256_GQA2` still refuses |

## Knob organization (2026-09-11)

Inventory: 138 runtime knobs (`RuntimeConfig` 33 shared / NVIDIA 34 / AMD 48 / Apple 14 / CPU 9)
and 162 emit knobs; every field has a code reader (devgen enforces it; the runtime's three
"unread" ones are consumed inside config.rs), so bloat is duplication and stale opt-ins, not
dead code. Production defaults are unchanged throughout.

Done:
- `packet::devbuild` read 21 `PLOW_*` variables straight from the environment (the reason the
  manifest carries `UNRECORDED_ENV`). They are now one `SegKnobs` struct with a single
  `from_env`, installed by `devgen::emit_config::install` from the same snapshot; nothing
  installed = read the environment, as before. `mla.rs` reads the builder's knob instead of
  the variable. Remaining raw reads outside the config structs are deliberate and documented:
  `PLOW_GLM_GF` (dual read for mid-process A/B repin), `PLOW_BLOCK`/`PLOW_ROOT`/`PLOW_UNISEG`
  in plowc (plowc-owned), tool env in kernelcaps/lean_verify/plowc tune (`PLOW_SOURCE_ROOT`,
  `PLOW_TOOLCHAIN_LABEL`, `PLOW_VERIFY_*`, `ROCM_PATH`…).
- `PLOW_VMM_BLOCK_MIB` and `PLOW_WEIGHT_VMM` were each declared twice (NVIDIA and AMD twins
  with `id=` disambiguation). One shared knob each: `--vmm-block-mib` (default 2) and
  `--weight-vmm` as `Option<bool>` — unset keeps the vendor default (CUDA on, AMD off).
- `PLOW_STATIC` / `PLOW_STATIC_DECODE` / `PLOW_STATIC_PREFILL` → one `--amd-static
  <both|decode|prefill>` (`PLOW_STATIC=1` still means both). Set by no script or recipe.

Next step (not done here): promote the non-diagnostic `SegKnobs` fields to `EmitConfig` fields
so `build.json` records them and `UNRECORDED_ENV` shrinks to the four diagnostics
(`PLOW_TUNE_DUMP`, `PLOW_TR_QUIET`, `PLOW_SEG_DUMP`, `PLOW_PLACE_REPORT`).

Retirement candidates (default-off, no script/recipe sets them; each deletes a code path, so
they are the author's call):
| Knob | Why | Cost of removal |
|---|---|---|
| `PLOW_CPU_L2_PLACE` | measured 1.5× slower, never faster (config.rs doc) | `cu_domains`/`cu_work`/`node_plan` + `WorkerPool::spawn` place plumbing + tests (~250 lines) |
| `PLOW_GLM_PLACE_PF` | +2.2% tput, P99 TPOT +10.5% (review) | emitter placement arm |
| `PLOW_GLM_FOLD_LT` | +0.49%, P99 +3.6% (review) | `amd_mla_fold.rs` (564 lines), pinned `glm_fold_lt_gfx942.json`, objects |
| `PLOW_GLM_XR_RES`, `GLM_FUSE_XRN` | unmeasured on GLM-5.3 TP8; +1.8–5% only on TP4 GLM-5.2 blobs | collective fold arms |
| `PLOW_MLA_PF_PSWZ` | −0.7% vs controls (review) | P-tile swizzle build arm |

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
| H0 | **DSA gather span vs `kv_len` clamp** (86e76151): `d_index_select_sm120` writes only `ib[0..min(kv_len,len_max))`; the MLA gather (`op_mla.cuh:181,189,211`) still reads `top_k` indices. Any GLM DSA request with `kv_len ≤ 2048` reads stale indices | runtime/nvidia/op_dsa.cuh, op_mla.cuh | gather span clamped to `min(top_k, kv_len[b])` (as AMD does) or select sentinel-fills; a GLM DSA packet with a 1k prompt decodes correctly on H100 |
| H1 | Build and run the four sm90 harnesses that CI never ran: `gemma3_norm_sm90`, `packed_flash_sm90_correct`, `packed_flash_fp8_sm90_correct`, `packed_kv_padding_sm90` (CMake targets under `PLOW_CUDA`, guarded) | runtime/tests/*.cu | all pass on H100 |
| H2 | Fix the paired-GQA2 test arena: under `PLOW_NV_FA_GQA2_PAIR` the test must launch with `FA_SM90_GQA2_PAIR_FLOATS` (141,312 B), not `FA_SM90_WGI_FLOATS` (208,896 B) — today it passes while production claims a smaller arena | runtime/tests/packed_flash_sm90_correct.cu:109-111,171 | test launches with the production arena and passes |
| H3 | Bless the W8A16 M1 prefill role numerically (upstream cf66dba6 added the test: `cargo test -p plowrt --features cuda --test gpu_consume_prompt w8a16_m1_role_matches_interpreter_on_real_prompts -- --ignored` with `PLOW_GPU_TEST=1` and the `TEST_W8A16_M1_*` paths): `gemv_rows_fp8<1>` (FFMA, f32 warp-sum) vs the interpreter's `d_gemm_fp8` (dequant-to-bf16 mma) — same math, different rounding | interp_sm90a_pfgemm_w8a16_m1.cu:86 vs interp_sm120.cu:1286 | rel-L2 vs interpreter recorded; greedy agreement on a real prompt set |
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
| 2026-09-11 | cf66dba6 | (this merge) | none | check cuda,hsa. Reviewed: one `#[ignore]`d H100 test (`w8a16_m1_role_matches_interpreter_on_real_prompts`, env-driven: `TEST_W8A16_M1_{BASELINE,ASSETS,LOGITS_OUT}`, `/tmp`-only output) — this is the hardware check for punch-list H3 |
| 2026-09-11 | 872ea411 (86e76151, 872ea411) | (this merge) | exec/gpu/prefix.rs, memory/vmm.rs (upstream re-implemented this branch's layout-check rollback and unlocked publish — took theirs), serve/mux.rs (CUDA arm moved; took theirs and re-applied the three branch fixes), `glm_prefill_collective/README.md` (stays deleted) | check ws + cuda,hsa; cpu suite; 754 hsa/cuda lib tests; devgen 423/423 ×2; plow-asset/packet; campaign tests 14/14. Reviewed by agent: see #23–#27, F16–F18 |
| 2026-09-11 | 5c26d90f | (this merge) | `mla_sparse_aiter/README.md` (deleted here, modified upstream → stays deleted) | check cuda,hsa; exec::amd_sparse_mla 4/4 (2 GPU-gated). Reviewed: opt-in single-pass sparse pack (object-gated, auto when present), NaN-guard test around the 512-byte workspace prefix |
| 2026-09-11 | d60bdeb2 | (this merge) | devgen/mla/glm_tests.rs (fmt vs new tests), `glm_ladder/README.md` (deleted here, modified upstream → stays deleted) | check ws + cuda,hsa; devgen lib 420/420 (GLM module 40/40 ×2 parallel); plowrt hsa/cuda exec::amd + CPU suite. Reviewed: GLM append attention specialised per query/KV ladder (small-rung object + FP8-KV split object), MoE AITER rows 1..8192, ladder boundary asserted via `decode_rung_lo`; runtime `exec/amd/mla_prefill.rs` validates split sites (pure dense QH8 FP8 segment, bounded partials, matching merge) |
| 2026-09-11 | 5b263108 | (this merge) | none | check ws + cuda,hsa; cpu suite 396/396; exec::amd_moe_aiter 8/8 (7 GPU-gated); devgen GLM module 40/40; packet 125/125. Reviewed: `MoeAiterFp8Pf.i6` becomes an output *mode* (0 sorted→FP32, 1 flat decode, 2 sorted→direct BF16); emitter defaults the GLM sorted A8 prefill route to mode 2; host refuses mode 2 unless the following `MoeCombinePf` bands (i7=1, H=6144) contiguously cover all T rows and the output aliases no input; output workspace 4→2 B/elt. See #28 |
| 2026-09-11 | 55ce86d7 (64b15dc0, 55ce86d7) | 701fc7f0 | none | check cuda,hsa; `metrics` test; 96 obs/serve lib tests; campaign tests 11/11. Reviewed: model-scoped Prometheus metrics (per-model `Arc<Metrics>`, unloaded models fold into the process totals), ladder-campaign refuses incomparable workloads / concurrent latency regressions |
