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
| 22 | 2026-09-11 | 5c26d90f | perf | exec/amd_sparse_mla.rs (`pack_fp8_single`), runtime/amd/mla_sparse_adapter.hip | Single-pass GLM sparse prefill pack: engages automatically for FP8 rows ≥512 whenever the adapter object was built with `--single-pass` (exports `plow_mla_sparse_single_abi_1`). Opt-in only through the object build, but once built it flips the route with no measurement recorded in the commit; kernarg ABI pinned to 104/360 B | **measured and shipped as the served default.** In flow (frozen packet, one binary, last steady 8192 chunk at prior 65536, rank 0, per-launch drains): route 2746 -> 2466 us/layer (-10.2%): pack 178 -> 182, attention 2428 -> 2285 (ns 2 -> 1), reduce 140 -> 0 (launch removed). That is -22 ms/chunk, about -19 s on the 100-prompt run. The standalone -24% overstated it, because the second split costs far less in flow. Bench, same pair: 49.08 -> 49.91 out tok/s (+1.7%, which is what -22 ms/chunk predicts), retrieval 18/18. The serving-set recipe (`scripts/build_glm53_gfx942_serving_objects.sh`) now builds the adapter with `--single-pass`; the frozen serving-safe set takes it when it is regenerated |
| 23 | 2026-09-11 | 86e76151 | **blocker if NVIDIA GLM DSA is served** | runtime/nvidia/op_dsa.cuh (`d_index_select_sm120` clamps `len = min(kv_len[row], len_max)`, writes `ib[0..len)`) vs runtime/nvidia/op_mla.cuh:181,189,211 (gather reads a fixed `top_k` span) | A request with `kv_len ≤ top_k` (2048) makes the gather consume `top_k − kv_len` stale indices → wrong attention or out-of-range latent-cache row. AMD clamps the gather span (`op_attention.h:2352`); the new interp comment claims the gather derives the live count but it does not | open — clamp the gather span to `min(top_k, kv_len[b])` or sentinel-fill `ib[len..top_k_max)`; **H100 punch list H0** |
| 24 | 2026-09-11 | 86e76151 | perf | exec/gpu.rs `inferred_segment_policy` + asset/devblob.rs `seg_classes_with` | Segment classing is now inferred from the packet on the default H100 SegPf path: any TMA-mapped pure-GEMM segment sets `pure_mode=1`, which routes every light-op segment (norm/rope/quant) to the fat flash object instead of the lean occ-2 GEMM object. Correct either way; per-chunk time changes, unmeasured | open — measure on the gemma4 ladder before merge; or infer only when the lean object exports a GEMM_ONLY marker |
| 25 | 2026-09-11 | 86e76151 | note | exec/gpu.rs `prefill_segment_kernel` vs roles-only path | MXFP4 MoE prefill role launched at `role.grid` (4× grid) on the segmented path, `self.grid` on the roles-only path | open — confirm intent, use `role.grid` on both |
| 26 | 2026-09-11 | 86e76151 | note | runtime/CMakeLists.txt:176-192 | the four sm_90a harnesses are in the default `all` target for every `PLOW_CUDA` configure (a `static_assert` in one breaks the runtime build on sm120 boxes) | open — `EXCLUDE_FROM_ALL` |
| 27 | 2026-09-11 | 86e76151 | note | exec/gpu/cublaslt.rs:381 | every failing cuBLASLt prefill-segment condition now reports the same "invalid packet-declared projection segments" | open — keep phase-specific messages |
| 28 | 2026-09-11 | 5b263108 | note | devgen/mla.rs (`d.i = [.., 64, 2, ..]`), exec/amd_moe_aiter.rs `Mode::SortedBf16` | emitter default flip: the GLM sorted A8 MoE prefill output is now rounded to BF16 before `MoeCombinePf` (mode 2) instead of carried as FP32 (mode 0). The cross-expert accumulation was already BF16 in-kernel, so the expected delta is one extra rounding of the routed sum, but it is an unmeasured numerics change on the default path. gfx942-only op, no CUDA surface | open — MI300X A/B of GLM prefill logits mode 0 vs 2 before merge to main |
| 29 | 2026-09-11 | c58a9b11 | note | devgen/mla.rs `glm_prefill_projection_op` | measured shape table baked into the emitter: single-row GLM prefill projections take `DevOp::Gemv` only for `tp == 8 && hidden == 6144 && n_cu == 304` (MI300X), BF16, and seven literal `(n, k)` pairs. gfx950 (256 CU), TP4 and MXFP4 keep `pick_tile`. Not a bug; same family as #7 — these belong in the tunedb / `kernelcaps` sweep so gfx950 and TP4 get the same measurement instead of a code edit. Runtime side checked: `required_gemv_m` reads M off the instructions and prefill objects already carry M=1 GEMV (lm_head), so no object-bucket refusal. ABI: Gemv `i3=norm`/`t3..t4` left zero/`TENSOR_NONE` by the shared emit closure = plain GEMM | open — author: tunedb entry instead of literal table; MI300X perf only, no CUDA surface |
| 30 | 2026-09-11 | 454a2564 | perf-risk (H100) | plow-asset/segment_roles.rs `CUBLASLT_PREFILL_ROWS`, exec/gpu/cublaslt.rs:386, devgen/dense_cublaslt.rs:181 | cuBLASLt prefill qualification for the Gemma-4 BF16 projections (3840×15360, 3840×8192) narrows from M ∈ 1..=128 to exactly M ∈ {128, 256, 512}. Net: M=1,2,4,8,16,32,64 prefill rungs (short prompts, prefix-hit tails) drop from cuBLASLt back to native GEMM; M=256/512 newly gain cuBLASLt. 86e76151 called the 1..128 cells "measured"; this commit calls 128/256/512 "measured" and carries no numbers for the dropped cells (raw results are local by policy). Emitter and runtime share the one predicate `cublaslt_prefill_bf16`, so they cannot disagree. sm90a only; sm120/AMD untouched. The new `#[ignore]` full-logit gate (`gpu_prefill_roles_match_control_logits`, env `TEST_PREFILL_RUNG_*`, `/tmp`-only output) is the correctness half; the perf half is unrun | open — H18: A/B M ∈ {1..64} native vs cuBLASLt and confirm M256/512 win on H100 before merge to main |
| 31 | 2026-09-11 | (this branch) | note | `perf-data/tools/gpulease` | under `nix develop` the wrapper finds no `amd-smi`, detects ONE GPU, and an "exclusive" lease holds only gpu0 (`gpus=[0]` in the lease log). Every lease taken from inside the flake shell this session had that signature, including the C20 baseline (`glm53-c20-baseline-head`); the bench itself was not overlapped (agents were told to wait for the release), but the lease gave no protection. Fix used: `GPU_LEASE_NGPU=8` in every wrapper | open — make the wrapper find `amd-smi`/`rocm-smi` from the flake or default the count from `/sys/class/kfd` |
| 32 | 2026-09-11 | (this branch) | note | `plowrt op-audit` | the audit classifies statically: a token-batch body's `HeadNormRope`/`HeadNormRopeFp8`/`FlashMlaPrefillFp8` still read as class C although the body resolves them through the `PlowTokenBatch` descriptor (`PLOW_PACKED_PREFILL_BAND`). `Capabilities::amd_dense_gqa` is still the only constructor; a GLM-MLA constructor naming the converted ops is owed so `refuse_program` can gate bodies at load instead of the loader's ad-hoc checks | open |
| 33 | 2026-09-11 | (this branch) | note | `scripts/build_gfx942.sh` vs `build.json` | emit→object contract audit (report `emit-object-contract.md`): gfx942 objects are compiled from hand-maintained `-D` axes, no `PLOW_HSACO_CONFIG`, so the packet's `requires`/`objects`/pairing hash never reach hipcc; loader has no check for `PLOW_FP8` weight arms, GM_BM/GM_BN/GM_DBUF tile geometry, the pairing stamp (every gfx942 object is unstamped), or inventory pruning. Recipe drift: the frozen packet contradicts `recipes/infervisor/glm-5.3/*.toml` on the decode ladder, DSA and rope fusion | open — task: consume `plow_config.h` in the gfx942 build, add markers + refusals |
| 34 | 2026-09-11 | 7c2cd12f | note (H100 policy) | exec/gpu.rs `live_rings_for_context` | CUDA live-ring KV allocation is now auto-enabled for any packet with `max_ctx >= 131072` even when `--nv-vmm-live-rings` is unset (logged "live ring allocation enabled for long-context KV"). A default memory-policy flip at 128K: on-demand ring commits replace the static allocation. Correctness is covered by the new mock test (`gemma_128k_live_kv_reserves_logical_windows_and_commits_on_demand`); the serving cost (commit latency on first touch, TTFT at 128K) is unmeasured | open — H19: H100 A/B of live rings vs static at 128K on a Gemma-4 packet before merge to main |
| 35 | 2026-09-11 | 184d2337 | note (H100 policy) | exec/gpu.rs `live_rings_for_capacity`, memory/vmm.rs `VmmRings` | (a) live rings now also auto-enable when the packet's decode `batch >= 64` (second default flip on top of #34); (b) ring units are ref-counted and released on slot release (`release_slot`), so KV backing is reclaimed per request; (c) the eager `ensure_rows(b, 1)` for every slot at load is REMOVED — the mux contract says an unfed row still writes KV at its own pos, so an unfed slot whose row 0 was never mapped would touch an unmapped page unless the decode path maps it first. Covered by the new `vmm_ring_tests` and `gpu_consume_prompt` test on the mock; not exercised on hardware | open — H20: H100 run with a partially occupied batch (unfed slots) on a live-VMM packet; plus the #34 A/B |
| 36 | 2026-09-11 | (planner merge 842b6171; pre-existing) | perf (prefix hits, both GPUs) | `serve/engine.rs` `packable_prefill_step` (AMD); `config.rs` `pf_batch_cuda` + `exec/gpu/prefix.rs` `select_vmm_prefix_layout` (CUDA) | Hit restoration does NOT bypass packing on either backend: AMD `prepare_prefill_cursor` plans only the suffix from `resume` (snap_after=None on a hit); the CUDA packed arm derives `kv_row0 = frontier` after `admit_packed_slot`. What defeats coalescing is (a) AMD: a span is packable only if it is not the cursor's final chunk (the prefill head samples one token per launch), so a hit whose suffix fits one rung is always isolated at rung 128/512/2048; (b) CUDA: `pf_batch_cuda` defaults OFF, so hits run through `gpu_prefill_advance` → isolated `prefill_chunk`, and turning packing on (`PLOW_PF_BATCH=1`) disables the prefix layout's auto-selection unless `PLOW_NV_VMM_PREFIX=1` is also explicit. The AMD mixed-step (Gemma) arm coalesces finals with last-token replay. Fix for (a) is the token-batch body (per-slot sampling rows; bodies exist at exactly 128/512/2048) once the multi-member collapse is closed; (b) is a default/gating change (H21) | measured (control arm, frozen packet, 2026-09-11): a ~31k-token prefix warmed into 4 slots, then 4 concurrent requests sharing it with ~100-token suffixes. Every request resumed from the cache at row 31776 and answered correctly (4/4). But each suffix ran ALONE in the 128 bucket, one after another, at ~1.7 s apiece: **cursor 1.24 s** (slot prepare with the prefix still held by the cache, the case the slot-recycling agent flagged as unmeasured) plus 0.45 s of dense chunk at 31k prior. The host cursor, not the GPU, is the larger cost of a prefix hit. The bodies arm could not load: the loader counted token-batch bodies toward the plain packed-prefill pair (fixed in 77920f72, pending merge), so co-packing is still unmeasured. |
| 37 | 2026-09-11 | (infrastructure) | process | `perf-data/tools/gpulease` | The lease has **no fairness**: acquisition is a non-blocking scan retried until `GPU_LEASE_TIMEOUT`, so with several 8-GPU campaigns queued it is a thundering herd and the oldest waiter can starve arbitrarily. Observed today with seven campaigns queued: jobs waiting 54 and 67 minutes while a job queued 30 minutes later acquired. Total wall time is unaffected (the campaigns are serialised either way) but *which* result arrives first is random, so do not design a plan whose next step depends on a specific campaign finishing first, and set `GPU_LEASE_TIMEOUT` generously (14400) on every queued job. A FIFO ticket file would fix it | mitigated for this session by #38; gpulease itself unchanged |
| 41 | 2026-09-11 | branch `shared-expert-fold` | perf | devgen mla.rs `glm_shared_fold`, exec/amd_moe_aiter.rs, exec/amd.rs `bind_packed_experts`, runtime/amd/{op_moe.h,moe_aiter_adapter.hip} | New opt-in `PLOW_GLM_MOE_SHARED_FOLD` (default off; knob-off packet byte-identical): GLM shared expert folded into the native AITER MoE call as expert 256, 257/top-9 on the pinned objects (AITER gfx942 GLM-5 table is indexed 257/9). ABA on 8×MI300X, merged runtime (job 1789170528): −12.9 ms per 8192 prefill chunk (resolved against a 17.6 ms control gap), −0.42% C20 out tok/s inside a +2.46% control drift (no resolvable change), ≈ −11 s of the 1511 s reference, retrieval 18/18 on all five arms; the merged 120-byte adapter ABI and both older adapters (104 legacy, 112 swizzle) load. About half of the earlier −25 ms/chunk went to the merged tile64 + swizzle fmoe. Shared expert moves bf16 → A8: FFN rel-L2 vs f64 1.7% → 4.8% (≈2.8×); decomposition shows fold == routed + shared_a8 to ≤3.4e-3 and flat column blocks, so a precision cost, not a defect. Needs packet-stamped objects incl. `interp_mla_{small,split}_fp8kv_gq` | open — merges opt-in only (default OFF) once job 1789170528 (merged runtime + adapter-ABI compat) passes; any default flip is the user's call, gated on a broader accuracy eval (GSM8K plus long retrieval at C20) |
| 38 | 2026-09-11 | (infrastructure) | process | `/root/.claude/jobs/c08d1232/tmp/gpuq/` | **FIFO GPU queue** replacing per-campaign lease races: one runner (`runner.py`) holds all 8 GPUs under a single `gpulease -n 8` hold and runs spool jobs back to back in submission order; agents submit with `submit.sh <label> <ngpu> <cmd...>` from the job's working directory. Jobs run under `nix develop --command` from that directory and **no environment is captured** (the auto-mode classifier rightly refused copying other processes' environments to disk), so campaigns set their own knobs or pass `env VAR=value`. Two lessons it encodes: a whole-box `gpulease` hold does NOT export `ROCR_VISIBLE_DEVICES` (it only pins partial leases), so unset means every card; and every job preflights its paths before submitting. | in use |
| 39 | 2026-09-11 | d60bdeb2 (gap surfaced today) | correctness (load) | `/tmp/tp-glm53-pswz/serving-safe` | Packets emitted by current HEAD `plowc` carry GLM small-MLA / FP8-KV split segments (they set `PLOW_MLA_PREFILL_FP8_SPLIT=1`, which makes the runtime hard-load `interp_mla_split_fp8kv_gq.elf`, plus `interp_mla_small_fp8kv_gq.elf` for small segments; the production packet from the older plowc needed neither; confirmed independently by three agents), and the frozen serving-safe object set predates their objects, so ANY freshly emitted packet run against it dies at load with `small MLA segments require .../interp_mla_split_fp8kv_gq.elf` (the per-rung campaign lost a hold to it). The frozen packet itself is unaffected. Per object dir: build the `interp_mla_small*` / `interp_mla_split*` rows with `PLOW_ROWS_ONLY=interp_mla_s PLOW_HSACO_CONFIG=<packet>/plow_config.h`, and list `interp_mla_split_fp8kv_gq.elf` as required in preflights. Related: `scripts/freeze_serving_set.sh` and `scripts/pack_objset.py` copy only `*.elf`, so they silently drop every pinned vendor `.co` (the 64-row MoE tile included). Also note the frozen set is no longer byte-for-byte what was frozen: the MoE agent linked `fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_psx_64x256.co` into it (merge a9951294). Load behaviour there is unchanged, because both new MoE defaults are auto-on only when the adapter carries its marker (`plow_moe_aiter_tile64_abi_1`, `plow_moe_aiter_swizzle_abi_1`) and serving-safe's adapter predates both. Durable fix: regenerate the serving set from HEAD (the per-rung agent has a script for it).  **Stamping:** build these rows UNSTAMPED (no `PLOW_HSACO_CONFIG`) whenever one object set serves several packets — a stamped object is refused against any other packet, and packets differ in hash by knob (e.g. a no-GEMM_LT emit is `0xe8cc92348e494a59` against `0xfb318c7e3408f87b`); the rest of serving-safe is unstamped. Stamp only a set used by exactly one packet (as `hsaco-tb2` is, for `assets-body2`). **Durable fix landed** (merge of 95ecee21): `scripts/build_glm53_gfx942_serving_objects.sh OUT ASSETS VENDOR` regenerates the whole set (any existing serving set is a valid VENDOR dir); `freeze_serving_set.sh` now carries `*.co`; `pack_objset.py` records each `.co` as a vendor object, so the objset id changes when one goes missing. | fix landed; serving-safe itself not yet regenerated |
| 40 | 2026-09-11 | (process) | correctness (measurement) | `/app/plow/.claude/worktrees/tp-merge` | **A Codex agent works inside the shared review worktree** (parent process `codex`, cwd tp-merge). It created branches `gemma4-12b-mi300x-vllm` and `...-rungs` there, switching the checkout mid-merge twice; leaves uncommitted token-batch edits (`token_batch_rows(sample_rows, decode_rows, prefill_rows)`) in `serve/{mux,engine}.rs` and `exec/amd/mixed_step.rs`; and **rebuilt `target/release/plowrt` at 19:32 from that uncommitted work**. Never measure `tp-merge/target/*` binaries; the pinned clean binary built from the committed head is `/root/.claude/jobs/c08d1232/tmp/gpuq/bin/plowrt` (sha 2b51868e). Merges here now check the branch before pushing, verify from a clean export of the committed head, and commit only the index. | open, needs the user to relocate that agent |
| 41 | 2026-09-11 | (process) | correctness (measurement) | `/tmp/tp-glm53-pswz/serving-safe` | **The "frozen" serving set is a symlink farm into mutable build dirs.** Every `interp_decode*.elf` is a symlink (the GLM TP8 decode object `interp_decode_fp8kv_gq.elf` into `/tmp/tp-glm53-rungs/objects/`, `interp_decode_gq.elf` into `/tmp/tp-glm53-select-local/decode-objects/`), and the low-rung decode tiers `lowrung1/2/4/8` next to it are DIRECTORY symlinks the runtime loads when `PLOW_HSACO_LOWRUNG` is unset. Those targets were rebuilt in place at **18:43:12** while the set's own mtime stayed at 18:05, so nothing looked different from outside (found by the per-rung agent). The rebuilt content is byte-identical to a HEAD build and keeps `plow_dsa_decode_batch_arm`, but **it is STAMPED for the production packet (0xfb318c7e3408f87b), not unstamped** as first reported. It is harmless only for packets with that hash. The loader correctly refuses it for any other packet (`specialised AMD object ... interp_decode_fp8kv_gq.elf stamps 0xfb318c7e..., asset requires 0x78432b97...`). That refusal cost the collapse requalification its slot, because freezing hsaco-tb2 by dereferencing pulled the stamped object in for the bodies packet; its decode objects were rebuilt for that packet and the job requeued. A manifest proves the bytes did not change, not that they match the packet, so a frozen set used with a non-production packet must have its specialised objects built for that packet. Separately, but a rebuild during a hold changes objects between one job's arms. The lease log shows exactly one hold straddled it, `glue-fusion-main` (18:34:44 → after 19:05); its object dirs were assembled dereferenced, and that is being confirmed. **Cross-campaign comparisons spanning 18:43 compared different decode objects.** Rule: before a hold, freeze every object dir a job loads with `cp -rL` (including the tier subdirectories), write a manifest over every depth (`find . -type f \( -name '*.elf' -o -name '*.co' \)` → `MANIFEST.sha256`), require `find <dir> -type l` to be empty, and have the job run `sha256sum -c` before any server starts. Frozen here: `hsaco-tb2` (83 objects) and `serving-safe-frozen` (69). | open |
| 42 | 2026-09-11 | (tooling, pre-existing) | correctness (metadata) | `scripts/plow_dist.py` `packet_hash_from_symbols`, used by `scripts/pack_objset.py` | The object-set tooling cannot see an AMD packet stamp. It parses the hash out of a symbol NAME (`plow_packet_hash_lo_<hex>`), but a gfx942 object carries the stamp as the 32-bit CONTENTS of two data symbols: `llvm-nm` on `hsaco-tb2/interp_decode_fp8kv_gq.elf` shows `D plow_packet_hash_lo` / `D plow_packet_hash_hi`, whose symbol values are addresses (0x305A58 / 0x305A5C), not the hash. The runtime reads the contents (`elf_symbol_u32(image, "plow_packet_hash_lo")` in `exec/amd/object.rs`) and refuses mismatches at load (tracker #41 shows it doing so), so serving is unaffected. But pack_objset records every AMD stamped object as unstamped, and an object set's identity does not reflect which packet its specialised objects pair with. The same trap misled a symbol-name grep during today's audit. The name form looks wrong on CUDA too (unverified): its roles use names like `plow_packet_hash_lo_fp8m1`, read as a value with `cubin::global_u32`, so the suffix is a role, not a hex hash, and `int("fp8m1", 16)` would raise. Fix: read each symbol's 32-bit contents from its containing section, as both loaders do. Reported by the per-rung agent; the AMD half is confirmed here. | open |
| 43 | 2026-09-11 | (tooling) | false positive | `scripts/asm_audit.py` | The object audit fails an FP8-KV decode object for a missing `d_flash_merge` head_dim-64 arm even when the packet has `PLOW_HAS_FLASH_MERGE=0` and no decode rung dispatches it. It surfaced only because a second `build_gfx942.sh` run audited the first run's objects in the same output dir, so it is easy to misread as a broken build. The audit should key the arm requirement on the packet's `PLOW_HAS_FLASH_MERGE`, as the loader does. Reported by the shared-expert agent; not fixed. | open |
| 44 | 2026-09-11 | (pre-existing) | correctness (pairing) | `crates/devgen/src/manifest.rs` `pairing_hash` | **The packet stamp does not cover the recipe's object requirements.** `pairing_hash` feeds only the opcode union, the `objects` list and `tuning`. The re-baseline re-emit of the production packet from 1e793cd9 (same recipe, `--replay-knobs` from the frozen build.json) produced a DIFFERENT model.pkt (sha 76a0018f vs b7566b8f) whose header newly carries `PLOW_PACKET_REQUIRES_DSA_DECODE_BATCH 1` and `PLOW_DSA_DECODE_BATCH=1` in `PLOW_PACKET_OBJECT_REQUIRES`, plus the EXT decode GEMM route as a production default, yet it kept the SAME `PLOW_PACKET_HASH` 0xfb318c7e3408f87b. Nothing in plowrt or runtime/ reads the `PLOW_PACKET_REQUIRES_*` macros back; they are compile-time `#ifndef` defaults only. So an object set compiled for the frozen recipe pairs with the new packet by stamp and loads without complaint, and any arm the new stream needs that the old objects compiled out falls to the interpreter's `default:` (the class of the token-batch collapse). Fix direction: feed `backends.<arch>.requires` (or the header's REQUIRES/OBJECT_REQUIRES block) into `pairing_hash`. **Fixed in d9e4b743**: every backend's `requires` now feeds the hash (`recommends` do not); plowrt reads the stored `pairing.hash` and never recomputes it, so already-built packet/object pairs are unaffected and only packets emitted from then on get a new stamp. The re-baseline A/B was unaffected (each arm paired a packet with objects built for it). |
| 45 | 2026-09-11 | (process) | infrastructure (disk) | box-wide: `ulimit -c` unlimited, `/usr/share/apport/apport` absent | **A GPU fault writes its full coredump into the job's working directory.** The kernel core pattern pipes to apport, which is not installed, so the ROCm runtime falls back to a file-based dump (`gpucore.<pid>.gpu`) in the cwd, and the core limit is unlimited. One decode memory fault in an 8-rank amd-bench (`fp8-capture`, 22:15) wrote **92 GB** into an agent worktree; older ones hold ~4-6 GB each (`agent-a783b53c6acd9b96f` 4 files, `tp-bringup-harness` 3 files, ~30 GB together). The root overlay had 341 GB free, so a few faults would stall every build and the GPU queue. The switch that turns the GPU dump off is `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` (present in ROCr 7.14's `libhsa-runtime64.so`, next to `HSA_COREDUMP_PATTERN`); every GPU campaign script should export it. The FIFO queue runner's soft `RLIMIT_CORE` was also set to 0 (`prlimit`), which stops CPU core files, but it is NOT shown to stop the GPU dump: ROCr imports `getrlimit`, and nothing here shows it gates the GPU dump on `RLIMIT_CORE`. The queue's submit script now prefixes every new job with the switch. A `gpucore.*` file must never be staged; never `git add -A` in a GPU worktree. The existing dumps are left for their owners to delete. |
| 46 | 2026-09-12 | (process) | throughput of experiments | the shared FIFO GPU queue | **Experiments promote through a ladder instead of starting as full serving runs.** Over 2026-09-11/12 the queue ran 26 jobs: 340 min of 8-GPU time and 4 min of 1-GPU time, but most jobs waited 37-186 min to start; 1-GPU kernel checks needing 0.1-3 min waited 39-141 min behind full campaigns, and four jobs failed in under 3 min on setup errors (file mode, wrong object) only after long waits. The decisive numbers of the day all came from kernel- or tick-level measurements (gather -24.5, glue -25, FP8 o_proj -24.4 ms per chunk), which the 20-40-prompt serving benches could not resolve inside their +/-1-2 % spread. Ladder: tier 1 CPU only (tests, byte-identical packets, object audits); tier 2 one GPU, kernel harness on captured activations; tier 3 8 GPUs, a truncated packet (`plowc --layers N`, the first N layers, a hidden truncation instrument, "never a served packet"; it keeps TP8 and the token-batch body route, whereas `--layers single:L` on GLM is a one-GPU bring-up path that asserts tp == 1 and emits no bodies) or one 8192 chunk plus decode ticks under `PLOW_TICK_LOG`, and a load smoke test; tier 4 full serving A/B plus retrieval, only to flip a default, 100 prompts only for re-baselines. Promote only when the lower tier shows an effect above its own noise. **Kernel work is proven on a single block** (user, 2026-09-12): tier 3 defaults to a truncated TP8 packet (`--layers N`, with N just past the first layer containing every op being changed; GLM's first layers are dense), and per-layer savings are projected to the full model, with the fixed per-tick costs that do not scale with layers stated separately. A truncated packet gets its own stamp: the stamp covers `tuning`, whose `tile_lookups` scale with layer count (2109 tile lookups for the full packet, 95 at `--layers 4`; hashes 0xa91a3e9d10deeba9 vs 0xb4b8f14a275b2db1), so it needs an object set built for it (about 10 min of CPU with the regeneration script). Compare `PLOW_PACKET_HASH` on a CPU-only emit before queueing. Hashing the distinct tile choices rather than the lookups would let truncated packets pair with the full set; not done. The full model runs only for serving and scheduler effects, retrieval, and default flips. Effects that exist only in serving (host work per tick, slot recycling, CPU socket placement, cross-process decode drift, prefix cache) and retrieval quality still need tier 4, as confirmation. The queue's submit script now sorts tier 2/3 jobs (`GPUQ_QUICK=1`, and every 1-GPU job) ahead of full campaigns. |
| 47 | 2026-09-12 | (pre-existing) | correctness (crash) | `plowrt amd-bench --prompt`, greedy decode on GLM TP8 | **amd-bench's greedy-decode path faults on GLM-5.3 TP8, after a correct prefill.** Reproduced twice: `fp8-capture` on the production packet with the frozen production objects (prefill 8192 tokens, "all 8 ranks agree", then `Memory access fault by GPU node-8 ... Reason: Unknown` at `greedy decode:`), and `sp-seams-t3-p1` on the t8 truncated CONTROL packet (seams off; prefill 16,384 tokens in 186.4 ms, ranks agree, the same fault at the first decode step). `plowrt serve` decodes correctly on the same packets, so the fault is in amd-bench's decode driver, not the kernels. Until it is fixed, decode measurements go through `plowrt serve` (TICK lines) and amd-bench is prefill-only. The first fault wrote a 92 GB GPU coredump (#45); the second wrote none, because the queue now sets `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` on every job. Not fixed here. |
| 48 | 2026-09-12 | 3a236fc3, 093aaf8c (branch `collective-bw`) | perf (opt-in) | `runtime/amd/op_collective.h` `PLOW_XR_SCHED_NWG` / `_NWG_RS`, `scripts/build_gfx942.sh` `PLOW_XR_SCHED` | **Prefill collective: the strict-rank-order 16-byte schedule on 24 data workgroups, its reduce-scatter on 8** (`PLOW_XR_SCHED=aiter`, objects only, default off). The split cap (093aaf8c) takes the isolated 8192x6144 two-shot from 795 (one cap of 24) to 728 µs, −27 % against the shipped 1002 µs; the in-model numbers below are for the single cap of 24. Every prefill two-shot / op 25 / op 26 packet is still emitted on 304 CUs; workgroups past the cap take only their `gate_ag` arrival and leave, so the packet and the host audit are unchanged. Bit-identical: knob-off prefill objects are byte-identical to HEAD's, and every microbench arm matches the host strict-order oracle with one checksum across arms and ranks. Microbench, 8x MI300X, 8192x6144 bf16: 970 → 794 µs (−18 %, 181 → 222 GB/s effective); 2048 / 512 / 128 rows −29 / −32 / −32 %. In model (4-layer TP8 packet, last 8192 chunk at prior 65536, ctl / K24 / ctl2): 1.08 → 0.99 ms per collective; projected −12 to −14 ms per 78-layer 8192 chunk, about −11 to −13 s on the 100-prompt run. Decode rungs emit only the one-shot `XReduce` and are not touched. Default flip needs a served A/B and the retrieval screen. | open (opt-in) |
| 49 | 2026-09-12 | (pre-existing) | tooling (pairing) | `crates/devgen/src/manifest.rs` `pairing_hash`; `build.json` `tuning.tile_lookups` | **A truncated `plowc --layers N` packet does not pair with the full object set.** `tuning.tile_lookups` counts per-layer tile lookups (2109 at full depth, 95 at `--layers 4`) and feeds `pairing_hash`, so the stamp changes (0xa91a3e9d10deeba9 vs 0xb4b8f14a275b2db1 on the d9fe7690 recipe) and plowrt refuses the full set's stamped objects. A tier-3 truncated probe therefore needs an object set built for its own packet (`build_glm53_gfx942_serving_objects.sh`, about 10 min of CPU). Fix direction: keep the count out of the hash. | open |
| 50 | 2026-09-12 | (pre-existing) | correctness (measurement) | GLM-5.3 TP8 prefill on gfx942 (probably the AITER MoE's bf16 atomic output) | **Prefill logits are not reproducible across processes, so byte-comparing logits cannot gate a change.** Same binary, packet (4-layer TP8) and object set, two processes (`collbw-probe2` ctl vs ctl2, 73,728-row prompt → 65,000 at decode): rank 0's `act.logits` shard after prefill differs in 70 % of elements (max \|d\| 0.047, mean 0.0065; argmax agrees); after five rung-20 decode steps argmax agreement is 45–55 % for every arm, controls included. Treatment arms differ from control by exactly as much as a second control does. A bit-identical kernel change must therefore be gated by a strict-order oracle on the kernel itself (e.g. `tp_allreduce_prefill_bench TP_RANDOM=1`) plus object/`.text` identity, and a numerics-changing one by statistics against the control-vs-control spread, not by `cmp`. | open |
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

## TP token batch (2026-09-11, this branch — `plans/unified-token-batch.md` "AMD TP8 lowering decision")

| Piece | Where | State |
|---|---|---|
| Physical cover `SpanCover::SlotBand { band }` (row t == KV slot t in the band; completing prompts sample from their slot's row; spans follow), `plan_requests_into_cover`, `validate_slot_band` (backend-neutral) | `plow-asset/mixed_step.rs` + tests | landed, CPU tests |
| Shared staging under a cover, sampled-slot delivery | `exec/mixed_step_staging.rs` | landed |
| Packet tag `TOKEN_BATCH_PROG` (bit 30), `Builder::set_token_batch_band`, band segment class 27, `DevProg/Program.token_batch_body`, manifest/audit labels | `packet/devbuild.rs`, `asset/devblob.rs`, `plow-asset/program.rs`, `devgen/manifest.rs` | landed |
| Emitter: `PLOW_TOKEN_BATCH_TP=1` → one body per prefill bucket wider than the band (packed-segment topology + batched decode attention over the band behind the prefill fold + band-sampling tail with a tiled-GEMM head); sparse buckets skipped; `emit_glm_dsa_select_rows` shared | `devgen/mla.rs`, `emit_config.rs`, `glm_tests.rs` (`token_batch_body_band_rides_the_prefill_program`) | landed; flag off ⇒ byte-identical (44 GLM tests) |
| Engine: `token_batch_bodies`, `token_batch_body_prepare` (spans/parked binding, VMM growth, per-slot kvlen shared by band and spans, ids/pos from the plan, live-row shrink, no lm a_row0 patch), descriptor upload (`PlowTokenBatch` header + input_ids/positions/active/sample_rows_idx), kernarg descriptor pointer only while a body is staged | `exec/amd.rs` | landed |
| Packed route: family 27 → decode object; body MLA family segments → `_tb` twins; `check_dsa_select_local` accepts the band form; body variant of the packed-MLA compatibility predicate allows AITER MoE / hipBLASLt / native fold | `exec/amd.rs`, `exec/amd/segment.rs`, `exec/amd/object.rs` | landed |
| Group step: prepare + re-arm every rank, one `xctr` reset, per-segment enqueue-all/drain-all, audit, band ids from rank 0, all-rank agreement over sampled rows on cadence | `exec/amd_tp.rs::token_batch_body_step` | landed |
| Serving: `TokenBatchTp` (staging + agreed bodies), `token_batch_rows/step` `Ranks::Tp` arms, `route=unified-token-batch/slot-band armed=… fires=…` | `serve/engine.rs` | landed |
| Device: `PLOW_PACKED_PREFILL_BAND` axis in `runtime/amd/packed_prefill.h` (band rows resolve to slot=row, position from the descriptor; parked band rows INACTIVE — the one new hazard, a mid-prefill slot's band row clobbering its span's frontier row, is closed here), KV-write address from span OR band in `op_norm.h`, marker `plow_packed_prefill_band_1` | `runtime/amd/*` | landed; 8 `_tb` objects compiled (`PLOW_TOKEN_BATCH_TP_OBJECTS=1`), contract PASS, norm_tb 101 VGPR / 0 spill |
| Body packet | `/root/.claude/jobs/c08d1232/tmp/tb-qual/assets-body` (13 programs: 4 prefill, 3 bodies 128/512/2048, 6 decode) | emitted from HEAD plowc |
| Sparse 8192 rung span-aware (`PLOW_PACKED_SPARSE_PF`): ABI-2 TP indexer adapter takes a `PlowKvSpan` table (`dsa_tp_adapter.hip`, `dev_isa.h`), AITER sparse flash runs one chain per span (`amd_sparse_mla::enqueue_spans`), no `IndexUnionPf` in packed programs (flash names `iidx_pf`), predicates admit the chain only with both routes present (`packed_sparse_refusal`), `packed_span_admissible` (kv_row0 ≥ 2047) gates packs and body members, native routes dispatch under an active packed binding (previously skipped → interpreter) | `devgen/mla.rs`, `emit_config.rs`, `exec/amd.rs`, `amd_index_tp.rs`, `amd_sparse_mla.rs`, `amd/object.rs`, `serve/engine.rs`, `scripts/build_dsa_tp.sh` output | landed on `spanaware-sparse-rung`; emit test `sparse_rung_joins_the_packed_passes_only_under_packed_sparse_pf`; device: retrieval 18/18 at C20 with the 8192 body firing (decode rows riding), A/B 48.85 vs control 50.17 tok/s (−2.6 %; TPOT 222 vs 239 ms, TTFT +5.2 %), 0 packs / 18 body fires; identity screen inconclusive (control diverges too) with degenerate body outputs → `PLOW_PACKED_SPARSE_PF` stays off; `docs/amd/tp-bringup-mi300x.md` §22, report `span-aware-sparse-rung.md` |
| GPU qualification | `tb-qual/serve-qual.sh` (identity C8 packed vs isolated, 18-case retrieval C20), `bench-arm.sh` (user's vllm bench command) | in progress — first load refused at `check_dsa_select_local` (fixed), rerun pending |

Baseline for the target workload, HEAD runtime + frozen TP8 packet + serving-safe objects, user's command (100 × 70k/700/.14, C20): **47.80 output tok/s**, 4762 total tok/s, mean/median TTFT 34.6 s / 16.0 s, mean/median TPOT 362 / 385 ms, median/p99 ITL 102 / 3361 ms, 1489 s. Target 150 tok/s = 3.1×. Fresh C20/65k decode attribution (GEMM agent): 104.9 ms/step; traced bodies XReduce 17.2, FlashMlaDecodeFp8 13.7, GEMV 11.9, MergeFold 8.7, IndexSelect 4.6, IndexScore 3.7 ms. Attention agent: attention ≈ 29% of workload wall; the 150 tok/s target is in the prefill linear term (GEMM/MoE).

Merged agent work (opt-in, shipped defaults byte-identical): AITER match for GLM dims (report `aiter-glm-kernels.md`): `PLOW_MOE_AITER_TILE64` — AITER's own GLM-5 gfx942 tuning table picks the 64-row persistent fmoe tile from 2048 tokens where plow dispatched the 32-row tile at every row count; new hash-pinned object, adapter `block_m`, rows ≥ 1024 use it; route-inclusive warm medians 1024 −9.7%, 2048 −8.5%, 4464 −27%, 8192 −38% (3480 → 2174 µs), numerics at the A8 floor (4.0–4.3% rel-L2 vs FP64 for both tiles). `PLOW_GLM_MLA_DEC_AITER` — the pinned QH8 MLA decode object as a `DecodeSegmentRoute::SparseMlaDecode` (pack → AITER attention → relayout into `MlaMergeFold`), rel-L2 1.4e-3 vs FP64, 58/92/153 µs at rows 1/8/20 vs ~176 µs/layer interpreter tail — gain unproven, stays opt-in. Nothing else in AITER matches these dims on gfx942 without changing the precision contract. Also  `PLOW_GLM_GEMM_LT_DECODE_EXT` (decode rung 8 + five narrow shapes to the pinned Tensile kernels; −7.5 ms/step standalone at rows 20, −9 ms at rung 8); DSA select load batching (rows 20: 268 → 146 µs at 65k) and sparse decode split policy (rows 20 ns16 → ns4: 169 → 108 µs) — the latter two need a re-emit / decode object rebuild and the 8-GPU A/B before any default changes. Reports: `/root/.claude/jobs/c08d1232/tmp/reports/{gemm-gemv-ladder,attention-ladder,emit-object-contract}.md`.


## Continuous batching by default (2026-09-11, `worktree-agent-a182a4b057a0ad61d`)

| Piece | Where | State |
|---|---|---|
| Emit: packed siblings beside native AITER MoE / hipBLASLt / resident MoE; index-TP assert on the builder topology; sparse buckets skipped; `glm_small_pf_split_cap` per builder so ordinary programs stay byte-identical; GLM gfx942 emits siblings by default (`production_default`) | `devgen/mla.rs`, `devgen/lib.rs`, `packet/devbuild.rs`, `glm_tests.rs` | landed, unit-pinned (two-emit comparison) |
| Runtime: `packed_mla_compatible` mirrors the body predicate; native AITER MoE / hipBLASLt / MLA-fold routes accept `packed_prefill_only`; route follows the packet, missing family object = load error by name; `_fp8kv` packed objects in `build_gfx942.sh` | `exec/amd.rs`, `exec/amd_{moe_aiter,gemm_lt,mla_fold}.rs`, `scripts/build_gfx942.sh` | landed; sibling packet loads with every rung armed, retrieval 18/18 at C20, bench 49.83 tok/s (= frozen packet, no pack forms at default planning so the siblings never executed) |
| Scheduler: backend-neutral step planner (`plan(backend, tick, decodes, candidates)`), `ServeEngine::step_backend` per backend, the `SeqEngine` arm and the CUDA pass lower its plan; `PLOW_PF_INTERLEAVE` unset = widest rung on AMD, `PLOW_PF_BATCH` unset = on (oldest-first), `PLOW_PACKED_PREFILL_ROUTE` unset = follows the packet | `sched/step.rs`, `serve/{mux,engine,step_lowering_tests}.rs`, `config.rs`, `exec/gpu.rs` | landed; 12 planner + 3 lowering tests; CUDA lowering unmeasured (no device here) |
| A/B (20 × 70k/700/.14, C20, frozen packet, same runtime/objects) | `docs/amd/tp-bringup-mi300x.md` §21 | old defaults **16.81** out tok/s, TTFT med 352.9 s, TPOT med 646.8 ms → new defaults **49.68** (+196 %), TTFT med 100.3 s, TPOT med 234.9 ms; packing fired in neither arm (DSA 8192 rung is class C, nothing packable at default planning) |
| Sibling packet (GLM gfx942 `production_default` emits packed siblings) on device, same defaults | `cb-qual/campaign-{qual,bench}-packed-v2.log` | retrieval 18/18, identity clean; **49.83** out tok/s vs 49.68 frozen (parity), TTFT med 101.2 s, TPOT med 233.6 ms, 0 packs fired — the emit flip is safe and neutral until the sparse 8192 rung is span-aware (task #15) |
| **User's 100-prompt command, new defaults, frozen packet** | `tb-qual/campaign-default-100.log` | **47.09** out tok/s, 1511 s, TTFT med 15.97 s, TPOT med 395.8 ms, ITL med 108 ms, 100/100 — parity with the 47.80 reference recipe (`PLOW_PF_INTERLEAVE=0`); the +196 % at 20 prompts was against the old shipped default that the recipe already overrode. Scheduling is at its structural limit here; the 3.2× is in the 8192-tick kernels (task #14 → #13) |
| `PLOW_MOE_AITER_TILE64=1` vs control, same binary, 20 prompts, frozen packet | `tb-qual/campaign-ab-tile64.log`, `campaign-tile64-b.log` | control **49.79**, tile64 **50.57** (+1.6 %), TTFT med 100.9 → 94.1 s, TPOT med 237.6 → 248.5 ms, ITL med 101.5 → 107.1 — object loaded on 8 ranks; adapter rebuilt (`scripts/build_moe_aiter.sh` with the 64-row object). Retrieval screen on this arm still owed before a default flip |
| Merge into this branch | 6b9b25ec (2d72ba7d) + merge of 842b6171 (planner, native routes accept packed siblings) + d6496f9e (`tests/config_env.rs` Option fields) | pushed; check ws, plowrt cuda,hsa lib 787/787, devgen GLM+packed_siblings 45/45 |

Where the target workload's time goes (100 × 70k/700/.14, C20; 7.07 M input tokens): ≈ 963 prefill launches of the 8192 sparse rung at ~1.3 s ≈ 1250 s, plus ≈ 2540 decode-only ticks at rung 20 (108.6 ms) ≈ 275 s — consistent with the 47.8 tok/s baseline. 150 tok/s = 467 s total ⇒ the 8192 tick must reach ≤ ~0.45 s even with free decode. The kernel reports attribute only ~0.72 s of that tick (sparse attention 78 × 3.7 ms = 0.29, AITER MoE 75 × 3.48 = 0.26, GEMM 0.17), so ~0.5–0.7 s per tick is unattributed → task #14 (rocprofv3 attribution on 8 ranks). Cross-request packing and decode-band fusion are worth ~5–10 % here and both need the sparse 8192 rung span-aware (class C today) → task #15.

Report: `/root/.claude/jobs/c08d1232/tmp/reports/continuous-batching.md`.

## Where the 8192 tick goes (2026-09-11, profiler agent, `prefill-tick-attribution.md`)

Production serve, frozen packet, new default planner, C20 × 66k; `rocprofv3` hangs plow's inline collectives, so GPU attribution is the interpreter packet trace + per-segment drains. Instrument merged: `PLOW_TICK_LOG=1` (7e19dbb0; `TICK` / `PFCHUNK` / `PFSEG` lines, off by default).

| tick kind | total | of which GPU drain | notes |
|---|---:|---:|---|
| steady 8192 sparse chunk + rung-20 decode | 1.13 s | 0.93–0.97 s | host around the drain 26 + 5.6 + 3.8 ms; decode after a chunk 163 ms (vs 107 alone: `decode_prepare` maps the next VMM block per layer, 38–41 ms) |
| first chunk of a request | 1.37 s | 0.99 s | + `begin_slot` 219 ms on a fresh slot, **1.9 s** on a slot that held a 66k prompt (VMM `release_window` unmaps ~400 blocks serially) |
| ragged tail 464 rows → dense 512 bucket at 65k prior | **2.04 s** | **1.89 s** | twice a full sparse chunk — the small buckets are dense attention over every prior key |
| decode-only, rung 20, 65k | 107 ms | 95 ms | flat in live rows: a lone straggler pays rung 20 |

Inside the 969 ms chunk drain: native sparse MLA 203 ms (2.60 ms/layer), AITER MoE 164 (2.19), indexer 86, hipBLASLt 71, interpreter segments 448 — of which `XReduceTwoShot` ×2/layer **174 ms** (~160 GB/s xGMI), o_proj 49, `MlaMergeFold` 46, router+top-k 42, align 51, residual 38, shared GLU 35, combine 35, norms 29. Reconciles the 100-prompt run to +1.4 %.

Ranked levers: slot-recycle unmap off the engine thread (−1.5..1.7 s/request ≈ −9 % wall); tails to the sparse arm (−1.0 s/request; `PLOW_AMD_TAIL_SPARSE_CTX`, 5ee91ece, A/B queued); two-shot collective 174 → ~90 ms (RCCL-class bandwidth or fused reduce+norm; glue-ops agent); post-prefill block mapping (−40 ms/prefill tick); MoE tile64 (−60 ms/chunk: **measured 49.79 → 50.57 tok/s, +1.6 %, TTFT med 100.9 → 94.1 s** on 20 prompts, same binary, `tb-qual/campaign-tile64-b.log`; adapter must carry `plow_moe_aiter_tile64_abi_1`); GEMM choice (−28); `prefill_prepare` (−15..20); async prefix publish (−10); decode enqueue+rearm (−9 ms/decode tick). Contradicts the kernel reports: sparse MLA in-flow 2.60 ms/layer not 3.7, MoE 2.19 not 3.48, tail cost 3–6× the attention report's estimate, collective data movement 18 % of the chunk not 1.3 %.

`PLOW_AMD_TAIL_SPARSE_CTX=16384` measured (5ee91ece, `tb-qual/campaign-ab-tail.log`, one lease, same binary):

* **It does what it claims.** Identity screen, a 15-row tail at 61,152 prior rows: the chunk goes from the dense 128 bucket at **724 ms** to the sparse 8192 bucket at **152 ms** (PFCHUNK `chunk=`), 4.7x; a 20-row tail at 15,360 prior stays dense (below the floor) and is unchanged, which is the gate working.
* **It is a no-op on the target workload** (wrong: see the 2026-09-12 correction at the end of this list). At `--random-input-len 70000 --random-range-ratio 0.14` every prompt is 60.2k–79.8k tokens, so after the 8192-row chunks every tail is 2049..8192 rows and ALREADY lands in the sparse bucket. 20-prompt bench: control **48.79** vs arm **47.25** out tok/s — the arm changed no chunk in that run, so the 3.2 % is campaign drift (same-binary controls across today: 49.68, 49.79, 48.79), not the flag. The flag's A/B needs short-suffix traffic (multi-turn / prefix-cache hits), i.e. the #36 / H21 workload.
* Stays **off by default**: for a tail at long prior context the sparse arm selects top-2048 keys where the dense bucket attended over every prior key. That is the same approximation the model's own DSA path uses for every wide chunk, but it is a numerics change for those rows, so a retrieval screen on this arm is owed before any default flip.
* The attribution's "tails cost 1.9 s" lever therefore lands on the 464-row-class tail (dense 512 bucket at 65k prior, 1.89 s) that appears when a request's length is just past a multiple of 8192 — real but rarer than the report's 21-of-21 sample suggested, because that sample used one fixed 66,000-token prompt.
* **Correction (2026-09-12): it is not a no-op here, and it is now the default (`16384`; `0` is the rollback).** The no-op bullet assumed every tail exceeds 2048 rows. A tail is the prompt length mod 8192, uniform over 0..8191. The bench's seed-0 sampler (`default_rng(0).integers(60200, 79801)`) reproduces both runs' input totals to the token, and gives 807 full chunks and 100 tails for the 100-prompt run. **19 of those tails are ≤ 2048 rows**: 16 land in the dense 2048 bucket, 2 in 512, 1 in 128. Each costs 1.4–1.9 s at 57–74k prior. The 20-prompt A/B above carried only 2 of them, too few to show.
  - Tier 3 (job `0-1789184638-tail-sparse-probe`, serving set d9fe7690): at 65,536 prior, 271/698/1464-row tails go from 439/1429/1489 ms to 185/211/275 ms. Full chunks are unchanged (793.6 vs 791.6 ms).
  - Tier 4 (job `0-1789188399-tail-sparse-ab`: 100 × 70k/700/.14, C20, serving set d9fe7690, ctrl / 16384 / ctrl2, one binary): 64.95 / **66.56** / 65.17 out tok/s, +2.3 % against the control mean with the two controls 0.34 % apart. Duration 1095.4 / 1068.9 / 1091.7 s (−24.6 s). Tick log over the bench only: each control ran exactly the 19 predicted dense tails (buckets 128/512/2048) in 25.7 s; the treatment ran all 19 in the sparse 8192 bucket in 4.3 s (−21.4 s). Prefill-tick wall −22.8 s; decode-only ticks unchanged (median 91.3 / 90.9 / 91.1 ms).
  - Retrieval on the treatment arm: **18/18 base plus 21/21 short-tail cells**, every tail cell served at its exact length; the control arm's tail cells were 21/21 as well. That is the 18 base cases plus the 21 short-tail cases (`quality.py --suite all`, 9c4a7c32). Those cases put the question inside a 271/698/1464-row final chunk at 65,536 prior, so they exercise exactly the rows this changes.

Body arm (5ee91ece, host-side, needs the collapse fix + a bodies packet to fire): cursors seeded for waiting slots when the route is armed, whole next chunks oldest-first, chunks no body holds skipped instead of blocking the pack, no cutting (a sparse 8192 step sliced into a dense body costs several times the launch it displaces). Unit tests `amd_token_batch_pack_*`, `dense_tail_moves_to_the_sparse_bucket_*`.

## Root cause of the token-batch body collapse (2026-09-11)

The symptom chased all day: with the token-batch body route armed, a body step returned token 0 for
every band row, and the identity screen showed degenerate repetitions the control never produced.
It was first blamed on multi-member steps, then narrowed to the 512-row body bucket, and neither
was the cause.

**The cause.** A token-batch body carries the packed-segment topology, which includes segments whose
only implementation is a NATIVE route — `GemmLtPf` (hipBLASLt) and `MoeAiterFp8Pf` (AITER MoE). The
dispatch gate withheld every native route while a packed binding was active, so those segments fell
through to the primary interpreter, whose dispatch `default:` **writes nothing and does not trap**.
The residual stream therefore went NaN from the first MoE layer, every KV row the body wrote was
NaN, and every band row sampled token 0. Nothing in the run reported an error, which is why the
route looked armed and firing while producing garbage.

**Two fixes, both merged or pending merge.**
1. `8306ac54` (span-aware agent) makes native routes dispatch under an active packed binding — the
   actual defect. Every dense body launch measured before this commit ran through the broken gate,
   so every body measurement taken today predates the fix.
2. `4f5bbec5` (debug agent) adds `token_batch_body_native_routes`: at load, a body program whose
   stream carries `GemmLtPf` or `MoeAiterFp8Pf` in a segment without the matching route is refused
   BY NAME, so the silent-no-op path can never be reached again. Plus a per-step `TBSTEP` tick log.

**The lesson worth keeping**: the interpreter's dispatch `default:` is silent. Any opcode that has
only a native arm must be refused at load when its route is absent, not discovered by reading NaNs
out of a KV cache. The same class of hole exists wherever a route can be withheld by a binding — the
packed and band bindings are the two that exist today.

## Plan of record: 100 output tok/s at C20 / 64K (re-baselined 2026-09-11, user's decision)

The 150 tok/s target required halving both the sparse attention (203 ms/chunk) and the MoE
(164 ms/chunk) kernels, and the research established that neither has a faster implementation for
gfx942 anywhere — both would be written from scratch (5–8 weeks each). The goal is therefore
re-baselined to **100 out tok/s**, which the measured levers do reach, and the remaining gap to
150 is recorded as kernel-authoring work rather than integration work.

**The budget.** Today's 100-prompt run (7.02 M in / 71.1 k out) is 1511 s = 47.09 tok/s, and it
decomposes as: 857 prefill-carrying ticks × (973 ms chunk + 163 ms decode) = 974 s, plus ~2600
decode-only ticks × 107 ms = 278 s, plus ~250 s of ragged tails and slot recycling. 100 tok/s is
**711 s**. Every figure below is seconds removed from the 1511 s run; ranges are the honest spread,
and the cumulative column assumes the phase lands at the low end of its range.

| Phase | Work | Saves | Cumulative | tok/s | Confidence |
|---|---|---:|---:|---:|---|
| 1. Host, no numerics change | KV map-ahead (#18, map-ahead proven to fire: 792/1416 maps moved into the chunk window, `prefill_prepare` 54.5 → 25.2 ms); slot-recycle unmap off the engine thread (#17, 1.5–1.9 s per recycled slot × ~80); prepare/publish/audit shadowed behind the drain (#23) | 210–280 s | 1231 s | **58** | high — all three measured |
| 2. Kernels already built, flags already there | 64-row MoE tile default (measured +1.6 % e2e); decode GEMM routes at rungs 8/16/20 and the narrow shapes; DSA select-load batching (268 → 146 µs at rung 20, needs a re-emit); decode split policy ns16 → ns4 (169 → 108 µs, needs a decode object rebuild) | 60–90 s | 1171 s | **61** | medium — each needs its 8-GPU A/B and a retrieval screen |
| 3. Collectives (#21, #26) | Raise the two-shot from 157 GB/s toward MORI-class ~300; then sequence-parallel seams: reduce-scatter → shard-local residual/norm/router/q_a/kv_a → all-gather only the projection outputs | 90–130 s | 1081 s | **66** | medium-high — 174 ms/chunk is measured, the bandwidth headroom is published |
| 4. Dense GEMM + MoE glue (#24, #25) | FP8 block-scale prefill GEMMs via CK/AITER (vendor table has our exact M=8192 shapes at 838–873 TF/s); fold the shared expert into the fused MoE call; expert-sliced MoE inside TP8 (no dispatch needed — every rank already holds all rows after the attention all-reduce) | 130–175 s | 951 s | **75** | medium — W8A8 is the checkpoint's reference numerics, so a quality screen gates it |
| 5. Token batch + indexer | Token-batch body carrying the decode rows at the 8192 rung, which removes the separate decode dispatch after every chunk (163 → ~20 ms incremental); fused FP8 indexer score + AITER's shape-exact asm top-k (−55..−65 ms/chunk) | 120–160 s | 831 s | **86** | medium — the body needs #15 (span-aware, landed opt-in) and the collapse fix |
| 6. Glue-op fusion | Merge fold into the attention epilogue or o_proj prologue; router → top-k → align as one kernel; combine + residual + norm | 35–55 s | 796 s | **90** | medium — e-graph rules exist, the win is measured per-op |
| 7. Remainder | Whatever of the above lands at the top of its range, plus the ragged-tail arm on short-suffix traffic | 60–85 s | 711 s | **100** | this is the slack the ranges above already contain |

**Phase 1 measured, and its composition is not what the estimate assumed.** KV map-ahead
(#18) is done and measured: at the tick level it moves 792 of 1416 driver mappings into the chunk's
shadow, `dec_vmm` 35.2 → 0.01 ms, decode-after-a-chunk 129.8 → 94.4 ms median (at the common
dec_rows 4–7, 136.9 → 95.5), and the prefill-carrying tick 1127.2 → 1081.7 ms. Total driver maps
over the run are IDENTICAL in both arms (276,480) — the work moved into the GPU's shadow rather than
disappearing, which is exactly the design. On the 100-prompt run that is ≈ 45 ms × 857 ≈ **39 s**.

The 20-prompt bench cannot resolve it: same binary, control 49.86 vs arm 49.86 out tok/s, duration
276.66 vs 276.65 s. Expected effect there is ~6 s of 276 (2 %), and today's same-binary controls have
drifted 48.8–50.6 tok/s, so a 2 % effect is below the noise floor of one run. What the bench DOES
show is the latency half: median TPOT 253.5 → 243.3 ms (−4.0 %), mean ITL 238.4 → 234.3 (−1.7 %).
**Treat tick-level instrumentation, not the 20-prompt bench, as the measurement instrument for
anything worth less than ~5 % here.**

Consequence for the phase-1 estimate (210–280 s): map-ahead is ~39 s of it, not the ~40–50 s
assumed *per lever*; the bulk must come from slot recycling (~1.5–1.9 s × ~80 recycles ≈ 120–150 s),
which a 20-prompt bench structurally CANNOT show because 20 requests over 20 slots recycle nothing —
that arm must run at 40+ prompts. Revised phase 1: **≈ 180 s**, dominated by slot recycling.

**Slot recycling measured (#17, merged 50a23464, default on — `PLOW_VMM_DEFERRED_RECLAIM`).**
Recycling a slot no longer unmaps the previous occupant's window on the engine thread. 40-prompt
bench (the smallest shape that recycles at all: 40 requests over 20 slots = 20 recycles), same
binary, one campaign:

| | control | deferred reclaim |
|---|---:|---:|
| out tok/s | 48.72 | **52.84 (+8.5 %)** |
| duration | 582.2 s | **536.8 s (−45.4 s)** |
| median TPOT | 349.0 ms | **310.6 ms (−11 %)** |
| P99 ITL | 3226 ms | **1178 ms (−63 %)** |
| median TTFT | 24.6 s | 23.6 s |

Retrieval 18/18 on the changed arm. Error-line counts are identical in both arms (the known
`fault_ms="0"` false positives in the loader's phase lines).

The P99 ITL is the signature: the 3.2 s stalls were a recycle blocking every live decode on the
engine thread, and they are gone. ≈ 2.3 s saved per recycle; the 100-prompt reference run has
~80 recycles, which suggests ~150–180 s there — to be confirmed on a 100-prompt run rather than
extrapolated. This makes slot recycling, not map-ahead, the bulk of phase 1, as the revised
estimate predicted.

**Phase 2 per-rung outcomes (per-rung agent, one job, one pinned binary, frozen objects):** `PLOW_GLM_GEMM_LT_DECODE_EXT` 50.68 -> 51.65 out tok/s (+1.9 %), decode tick 96.8 -> 90.3 ms, retrieval 18/18, **flipped on** as a production default (433cff05, byte identity shown). `PLOW_GLM_MLA_DEC_AITER` stays **off**: the emit isolates FlashMlaDecode without its MlaMergeFold partner, so it does not load on a HEAD packet, and its ceiling is ~1.8 ms/step anyway. `PLOW_GLM_GEMM_LT` is already on by default and was not re-priced (it would need a full packet-matched object build, for a number that changes no decision). **New finding:** the frozen production packet still runs the old ns16 decode split. The rows-aware split merged in ed09509c measures 106.3 -> 96.8 ms per decode tick (about 9.5 ms, twice its standalone estimate), so **re-emitting the production packet from HEAD is worth about 30 s on its own**, and it would carry every emit default merged today.

**Sparse MLA kernel rewrites, attributed in flow (sparse agent, `glm53-sparse-fast2-r2` + `glm53-gather-verify`, one binary, fresh objects):** per 8192-row chunk at prior 65536, rank 0, single-pass vs fast2 in the same lease: the **peer-group DSA TP gather** 2387.7 -> 1219.7 us/layer, **-24.5 ms/chunk** (merged with this entry; it matches the ~1.2 ms/layer estimate for 59 MB crossing one xGMI link at a time); the **FP8 pack rewrite** 181.9 -> 135.1 us, -3.7 ms/chunk; the **128-key score slab** 1808 -> 1740 us, -1.4 ms/chunk (keeps only ~40% of its standalone gain in flow). Together -29.6 ms/chunk; the sparse attention share of a chunk is ctrl 300 -> single-pass 280 -> fast2 251 ms, so the <=150 ms target is not met. The served pair measured 49.91 -> 49.73 tok/s (-0.4%), inside the +/-1% control spread, so one pair does not resolve ~3% of a chunk; use the in-flow split as the attribution. Retrieval 18/18; both FP8 pack entry points bit-identical; the gather is byte-exact vs the serial loop on grids 304/7/305 and from the real object with status 0. Remaining levers of that size: FP8 keys in the attention kernel (up to -70 ms, needs the a8w8 ABI) and a proper W_uv fold (-31 ms, `PLOW_GLM_FOLD_LT` + re-emit).

**Re-baseline, 100 prompts (2026-09-12, job `rebase-100-r2`, one binary for both arms built from the verified clean export of 594737d9):** the user's benchmark command (`vllm bench serve`, random, `--num-prompts 100 --random-input-len 70000 --random-output-len 700 --random-range-ratio 0.14 --max-concurrency 20`). **ctrl** = the frozen production packet (`/tmp/tp-glm53-prefix-keys/assets`) + the frozen real-file serving set: **56.18 out tok/s**, 1266.4 s, median TTFT 11.95 s, median ITL 106.6 ms, P99 ITL 1113 ms, 100/100. **head** = the production packet re-emitted from 1e793cd9 with `--replay-knobs` (same recipe; today's defaults: rows-aware DSA decode batch, EXT decode GEMM) + its regenerated object set, with the DSA TP adapter from c936e7fa (peer-group gather) and the 32 prefill/flash objects from fbdf47e3 (glue defaults): **65.10 out tok/s (+15.9 %)**, 1092.9 s (-173 s), median TTFT 10.51 s, median ITL 91.8 ms (-14 %), P99 ITL 1004 ms, 100/100, retrieval **18/18**. The earlier 100-prompt parity number (47.09) used an older binary with the frozen set, so the runtime merged since then accounts for most of 47.09 -> 56.18, and the packet plus objects at HEAD for 56.18 -> 65.10. **The frozen serving set is now ~16 % behind HEAD; regenerate it (and the production packet) from HEAD between campaigns, together with the #44 stamp fix.** Against the plan of record (100 out tok/s), the remaining gap is ~35 tok/s.

**Token-batch body, requalification verdict (collapse agent, jobs `tb-fix-all-r3` and `tb-fix-stat`):** the token-0 collapse is fixed (8306ac54 / 842b6171; the load-time refusal of an unrouted native step keeps it from going silent again), and the packed-family loader fix is merged (0c6bb7d3). On r3 the route arms and fires, retrieval is 18/18 at C20, the P1 [7] "wrong entry" was the ordinary route's miss (no cross-slot mixing), and the body writes the same MLA latent KV (layer 0 bit-identical, layer 77 within the ordinary route's own variation). **Still open: answers routed through the body degenerate.** Under the rule fixed in the report's §4.4 before the data (3 processes per route), every body process looped on 6-9 of 12 prompts and every ordinary process on 2-4; it happens with a single request too, mostly on short prompts; correct answers 31/36 vs 36/36. The prefill cache and the first token match, so the fault is in the decode after a body step. Probe (A) is approved: a diagnostic, off-by-default dump of the cache at prompt completion and of each decode step's inputs, DSA top-k selection and scores, one ordinary and one body process. Its first target is the DSA indexer key cache and scales, which the MLA latent comparison did not cover. **The token-batch body stays opt-in until that probe explains the degeneration.** Separately, this recipe does not reproduce its own tokens across server processes (1/8 in a second fresh control server), so identity screens on this stack must be statistical, not token-by-token.

**Token-batch body degeneration: root cause and fix (probe A, `tb-fix-probeA`; fix 004d9bdf):** the loops were a host bookkeeping bug, not numerics. `AmdServe::token_batch_step` (serve/engine.rs, from 58699e9f) asked `finishes()` whether `pos_stage[slot] + take == prompt.len()` AFTER `commit_after_device_success` had already advanced `pos_stage` to the prompt end, so a prompt that ended inside a body step never retired: the slot never went live and `slot_decode_position` returned the prompt length for every decode step. Each step overwrote one KV row and attended over the prompt plus only the newest token, so the model never saw what it had generated ("Cuskar travelled from Cuskar"); the first token was right because step 0's position was right. Probe A showed the rest was clean: every decode step selected the full DSA set (390/390 steps per process, no non-finite scores), and the caches matched the ordinary route within its own variation at every layer, indexer keys included. The single-rank path shared the loop and the fault. Fix: decide the finishers from the pre-step frontiers and retire exactly that set (`token_batch_finishing` / `token_batch_host_step`), with a CPU test that fails on the old ordering. The body stays opt-in until the quick-lane full-model check (`tb-fix-fixcheck`: position chaining, loop count within the ordinary 2-4/12, retrieval 18/18) passes.

**Re-baseline rerun with the engine thread pinned (job `rebase-100-pin`, 2026-09-12, HEAD arm only, binary 1cc0c868 with `--amd-engine-affinity auto`):** the same re-emitted packet and object set as `rebase-100-r2`'s head arm: **64.28 out tok/s**, 1106.9 s, median TTFT 13.73 s, median ITL 91.76 ms, P99 ITL 1009.7 ms, 100/100, retrieval **18/18**. The unpinned head arm had 91.8 ms median ITL too, so that server had drawn the fast socket by chance; the 1.3 % throughput gap (65.10 vs 64.28) is run-to-run variation. **HEAD serves about 64-65 out tok/s on the 100-prompt benchmark**, with no socket luck either way. The control arm was dropped (the frozen set is being replaced by the HEAD-regenerated serving set, task #34).

**New production serving set, regenerated from HEAD (task #34, 2026-09-12):** from the verified clean export of d9fe7690, with the production packet re-emitted by replaying the frozen recipe's knobs (today's defaults apply) and carrying the d9e4b743 stamp, `PLOW_PACKET_HASH 0xa91a3e9d10deeba9` (the frozen packet's was 0xfb318c7e3408f87b). Its object set came from the regeneration script alone: 57 interpreter and adapter objects (glue defaults, the peer-group DSA gather, the merged 120-byte MoE adapter, the single-pass sparse MLA adapter), 4 pinned vendor objects, 5 low-rung tiers, 0 symlinks, a 111-file manifest. Validation job `serving-head-validate` (tier 3): loads on the pinned d9fe7690 binary with the token-batch object and the 64-row MoE kernel, retrieval **18/18**. Its throughput is the pinned re-baseline's (same packet content, older stamp): about 64-65 out tok/s. Durable copy: `/workspace/plow-serving/glm53-tp8-d9fe7690` (packet, objects, binaries, manifest). **It replaces the frozen symlink-farm set (#41) as the control for every later A/B.**

**Roadmap and vLLM-parity audits (2026-09-12; reports throughput-3x-roadmap.md and vllm-parity.md, both read-only analyses of tonight's logs and code):** the 100-prompt HEAD run (1093 s) is ~78 % prefill-carrying ticks (the 8192 chunks alone ~68 %), ~22 % decode-only ticks, and ~0 idle (GPU busy ~99 %; serial host ~12 s, so host shadowing is used up). One 8192 chunk, ~855 ms mean: sparse attention ~269 ms (MLA 189 + indexer 80), collectives ~160, dense GEMMs ~150, MoE ~102 plus glue ~47, MlaMergeFold ~47, norms and other glue ~50; about 13 % of BF16 peak. A decode tick is latency-bound (~11 % of HBM bandwidth). **3x (~195 tok/s) is not reachable without speculation;** the no-speculation ceiling is ~120 tok/s from structural and integration work and ~140 with new kernels, consistent with the plan of record. Serving parity: continuous batching and chunked prefill are default; decode rows riding inside the prefill pass (vLLM's chunked-prefill mixing) is opt-in via the token-batch body and today rides only final chunks, because middle chunks are planned whole at 8192. Ranked near-term levers (estimates, each vs HEAD): the token-batch body carrying decode rows at 8192, -54 to -63 s; sequence-parallel seams plus an FP8 all-gather of the MoE input, -77 to -103 s; a faster two-shot all-reduce, -36 to -65 s; FP8 block-scale GEMMs v2, -29 s (tier 3 measured, tier-4 A/B queued); short final chunks on the sparse bucket, about -24 s. **Correction:** the earlier verdict that `PLOW_AMD_TAIL_SPARSE_CTX` does nothing on this workload is wrong; an exact replay of the benchmark's prompt lengths gives 19 of 100 tails at <= 2048 rows, each 1.4-1.9 s in a dense bucket (tier-3 probe pending). Also corrected: EP8 MoE weight traffic stays ~4.8 GB (rows per expert stay 256), not 38.6 GB.

**Token-batch body requalified after the retirement fix (job `tb-fix-fixcheck`, fixed binary, the assets-body2 packet and hsaco-tb2 objects):** (1) the body process's decode positions now chain step by step, with no position or length difference from the ordinary process, and every step still selects the full DSA set; (2) on the 12-prompt loop set the body loops on 2/12 against 4/12 for the ordinary process run alongside it (before the fix 6-9/12), both 12/12 correct; (3) retrieval 18/18 with bodies armed. The body stays opt-in: default-on needs the planner change that lets middle chunks carry the decode band at the 8192 rung, a tier-3 pricing probe, and a tier-4 served A/B. flags-reference's mixed-batching note is corrected (it said not implemented).

**Sequencing rules.** Phases 1 and 2 are independent of everything else and change no numerics
(phase 2 changes kernels, so each flip carries a retrieval screen). Phase 3 and phase 4 both touch
the same per-layer seam and must not be measured concurrently on one packet. Phase 5's body work is
blocked on the token-0 collapse; the native-route-under-packed-binding fix (`8306ac54`) is the
current lead and re-probing it is the gate. Nothing in phases 3–6 should start a default flip
without an A/B on the same binary plus the 18-case retrieval screen.

**What is explicitly NOT in the plan**: DP attention + EP8 (wrong shape for C20/70k — only ~2.4
requests are mid-prefill on average, so one GPU would run a chunk's 64 heads while the others pad);
two chunks per tick (tick conservation caps it at ~24 s); hipGraph (plow already enqueues raw AQL);
MXFP4 on gfx942 (dequant-only); MTP speculative decoding (real, ~9.9 GB of shipped weights, but
3–5 weeks and lowest priority by the user's instruction).

**The 100 → 150 gap**, for the record: native-FP8 sparse gather (−100..−125 ms/chunk, 5–8 weeks) and
a from-scratch MoE kernel below AITER's tuned ceiling. Both are kernel-authoring projects.

## MoE side of the 8192 chunk (2026-09-11, MoE agent, `moe-prefill-kernel.md`)

Shipped: **the 64-row AITER tile is the default** (`PLOW_MOE_AITER_TILE64` unset = on when the pinned `psx_64x256.co` and a tile64-marked adapter are in the object dir; the serving set needs `scripts/build_moe_aiter.sh OUT 32x256.co flat.co psx_64x256.co` re-run once — the 64-row object is already linked into `serving-safe`, the adapter is not). Retrieval 18/18 (C20, agree-every=1), same-binary A/B 49.79 → 50.57 tok/s (+1.6 %); 2.19 → ~1.4 ms per MoE layer in the chunk (−60 ms/chunk).

Where the MoE time is (single MI300X, `moe_aiter_xcd_swizzle_matches_and_splits`, 8192 rows random top-8): prepare 121 µs + fmoe 2628 (32x256) / **1241 (64x256)**. The kernel is bound by **weight re-streaming**, not MFMA: every tile streams its expert's 4.7 MB, 256 rows per expert per rank → 8 (32-row) or 4 (64-row) re-reads = 9.7 / 4.8 GB per layer against a 1.2 GB weights-once floor; FLOP floor 0.24–0.48 ms. Disassembly: `psx_64x256` remaps workgroup w → tile (w % 8)·38 + w/8 so an expert's tiles share one XCD's L2; `ps_32x256` does not, and its 8 blocks per expert land on 8 XCDs. Opt-in `PLOW_MOE_AITER_XCD` reproduces that order host-side for the 32-row object (pure block permutation, host mirror + GPU validator): fmoe 2628 → 1500 µs at 8192 rows, 851 → 547 at 2048. 8-GPU A/B (both arms `TILE64=0`, same binary, 20 prompts) + retrieval: ctrl32 **49.89** → xcd32 **50.54** tok/s (+1.3 %), TTFT med 100.8 → 97.3 s, ITL med 100.2 → 106.6 ms, P99 ITL 1348 → 1330 ms; retrieval on xcd32 **18/18** (`tmp/moe/campaign-ab-xcd.log`, `quality-xcd32.json`). The swizzled 32-row route lands where the 64-row route did in the coordinator's A/B (50.57), as the single-GPU split predicts (1.63 vs 1.36 ms/layer is inside the run-to-run band). Default flipped to auto-on (adapter-gated) on the strength of this pair.

No faster AITER object exists for 8192/6144/256/256/top-8 on gfx942: AITER's own tuner picks the 1-stage `vs_ps_64x256` asm (1314 µs kernel-only) over CK 2-stage and everything else from 2048 tokens up; the blockscale family has no tile wider than 64 rows. EP over the 8 ranks (32 experts at I = 2048 each) is a loss for this kernel family: same memory, collective bytes within ±20 % (input already replicated by the attention all-reduce), but 8× the weight re-streaming (38.6 GB/layer at a 64-row tile) plus expert-popularity imbalance — design note in the report, not implemented. What ≤ 90 ms/chunk (1.2 ms/layer; today 1.36) still needs: a plow-owned grouped GEMM with a ≥ 256-row expert tile (weights read once: 2.4 GB → 0.46 ms floor) or fewer BF16 output atomics (805 MB/layer) — kernel work, ranked in the report.

## Phase 2 (per-rung routes already built): what paid (2026-09-11, `rung-flips`)

One GPU-queue job, one pinned plowrt, one frozen object set (79 real files + sha256 manifest), the
20-prompt bench, the 18-case retrieval screen and `PLOW_TICK_LOG` on every arm, each arm a HEAD
re-emit differing from its own control by one flag. Report: `rung-flips.md`.

| route | out tok/s vs control 50.68 | decode tick | retrieval | decision |
|---|---:|---:|---|---|
| `PLOW_GLM_GEMM_LT_DECODE_EXT` | **51.65 (+1.9 %)** | 96.8 → 90.3 ms | 18/18 | **ON** for GLM gfx942 TP8 (joins the qualified recipe) |
| DSA decode split ns16 → ns8/ns4 (ed09509c, merged) | ns16 arm 49.19 (**−2.9 %**) | 106.3 vs 96.8 ms | control 18/18 | already default in a HEAD emit; **the frozen production packet predates it — re-emit it** |
| `PLOW_GLM_MLA_DEC_AITER` | does not load | — | — | off: emit isolates `FlashMlaDecodeFp8` alone, runtime needs the flash+merge pair in one segment; ceiling ≈ 1.8 ms/step |
| `PLOW_GLM_GEMM_LT` | not re-priced | — | recipe 18/18 | stays on (8b125bb3); an off arm needs a packet-matched object build and informs no decision |
| `PLOW_MOE_AITER_TILE64` | — | — | — | handed to the MoE agent |

Projected on the 1511 s / 100-prompt run: EXT ≈ 20 s, the split policy ≈ 30 s once production
re-emits its packet (tick model; the run-level ratios say 28 s / 46 s). The decode tick is flat in
live rows (≈ 97 ms from 2 to 20 rows), so every decode route is really a rung-16/20 route on this
workload; EXT's rung-8 half is idle here.

Found on the way (tracker #39, #41): a HEAD-emitted GLM packet does not load against the frozen
`serving-safe` set (small/split MLA objects missing, d60bdeb2); `freeze_serving_set.sh` /
`pack_objset.py` dropped every pinned `.co` (95ecee21 fixes both and adds
`scripts/build_glm53_gfx942_serving_objects.sh`); and `serving-safe`'s decode objects are
symlinks into directories other agents rebuild in place, now packet-stamped `0xfb31…`, so any
campaign must `cp -rL` its set before its hold.

## What the research says is actually available (2026-09-11, two research agents)

Reports: `research-kernels-parallelism.md`, `research-serving-techniques.md` (both under `/root/.claude/jobs/c08d1232/tmp/reports/`, provenance-tagged measured / fetched / estimated).

Findings that CHANGE the plan:

* **Two chunks per tick is worth ~24 s of 1511 (1.6 %), not the ~70 s assumed.** Tick conservation: the sum of live decoders' remaining tokens fixes the tick count at ~3548 however the prefill rows are arranged. Rearranging chunks only doubles ITL during prefill ticks. Deprioritized.
* **Host shadowing is the highest-confidence prefill lever, ~110 s (7 %).** ~100 ms of host work per prefill-carrying tick runs with the GPU idle (VMM map 40, prepare 26, prefix publish ~20, enqueue/rearm 9, audit 5.6) plus 9.3 ms per decode-only tick. Pre-map from `prefill_prepare`, double-buffer the patched program, append decode packets behind the chunk on the barrier-ordered queue. This is what vLLM's async scheduling and SGLang's overlap scheduler do. Open items #17/#18 are the first two pieces.
* **No faster sparse-MLA kernel exists for gfx942.** FlashMLA sparse is SM90/SM100; AITER's DSA prefill is gfx950-only; its `mla_v4` sparse objects are gfx1250-only; SGLang/vLLM on MI300X use TileLang or the same AITER QH8 decode-kernel trick plow already uses (vLLM pads 8 heads to 16). The gather is Infinity-Cache-bound and today's BF16 pack runs at ~45 % of last-level-cache peak; a native-FP8 gather would be ~1.2 ms/layer vs 2.60 (−100..−125 ms/chunk) but must be written from scratch (5–8 weeks).
* **The MoE kernel is already at AITER's own tuned ceiling** for this shard (their GLM-5 gfx942 table: 1314 µs at 8192 tokens with the 64-row tile plow pins, 530 TF/s = 20 % of FP8 peak). New lever instead: fold the shared expert into the fused MoE call as an extra expert, as AITER and vLLM do, removing ~54 ms of 105 TF/s interpreter GEMMs (−35..−40 ms/chunk).
* **FP8 block-scale GEMMs are available and sized**: AITER's gfx942 CK table has our exact shapes at M=8192 (839/838/873 TF/s), −70..−95 ms/chunk plus −3..−6 ms/decode step. hipBLASLt cannot do block scales on gfx942 — use CK or AITER's pre-shuffled asm. The checkpoint is `activation_scheme: dynamic`, so W8A8 is the reference serving numerics.
* **Collectives**: two-shot moves 176 MB/rank in 1.12 ms (157 GB/s); RCCL does ~165 GB/s at this size, the xGMI floor is 0.53 ms, and MORI-class dispatch reaches ~300 GB/s (−72 s). Sequence-parallel seams (reduce-scatter → shard-local residual/norm/router/q_a/kv_a → all-gather of projection outputs) are worth ~150 ms/chunk on top. Skip quantized all-reduce: vLLM disables it at TP8 BF16.
* **DP attention + EP8 is the wrong shape for this workload.** It fits memory (~111 GB weights + ~15 GB KV per GPU) and would lift the concurrency cap, but only ~2.4 requests are mid-prefill on average, so one GPU would run a chunk's 64 heads for ~1.8–2 s while the others pad. The useful half is **expert-sliced MoE inside TP8** (no dispatch needed — every rank already holds all rows after the attention all-reduce): per-rank MoE fixed costs shrink ~8× and the GEMMs get N=2048, worth 30–60 s.
* **hipGraph buys nothing here** (plow already enqueues raw AQL at ~0.6 µs/packet; the vLLM/SGLang graph wins on MI300X are Python-overhead wins). **MXFP4 on gfx942 is dequant-only** (no FP4 matrix core) — no prefill gain.
* **Constraint worth recording**: MLA latent KV is replicated on every TP rank (~50 KB/token/rank FP8 ⇒ ~70 GB/rank at C20 × 70k). That, not compute, is the concurrency ceiling under TP8.
* **MTP is real but last**: layer 78 ships complete in the checkpoint (shards 136–138: `eh_proj`, `enorm`, `hnorm`, full MLA + indexer, 256-expert MoE, sharing `lm_head`), ~9.9 GB FP8, skipped by the emitter today. DeepSeek-V3 reports 85–90 % acceptance and 1.8× at low batch; estimated −125..−180 s here with low confidence at batch 20, 3–5 weeks.

**Honest arithmetic**: host shadowing + a 300 GB/s collective + tile64 + the glue and GEMM levers + the slot/tail fixes land the run near 700–980 s, i.e. **71–105 out tok/s**. The 150 tok/s target additionally requires the sparse attention (203 ms) and MoE (164 ms) roughly halved, which means writing two kernels that do not exist for this architecture today.

## KV map-ahead shipped (2026-09-11, `kv-map-ahead.md`)

Lever 4 of the 8192-tick attribution, implemented and on by default. `AmdEngine::prefill_map_ahead`
makes the decode's own `ensure_rows(end + 1)` call on every rank between a chunk's segment-major
enqueue and its drain, where the host is ~900 ms ahead of the GPU; the decode that follows the chunk
then maps nothing. `PLOW_KV_MAP_AHEAD=0` is the rollback. `PLOW_TICK_LOG=1` gained `dec_vmm` /
`dec_maps` per `TICK` and `prepare_maps` / `map_ahead` / `map_ahead_maps` per `PFSEG` — the numbers
are in the phase-1 paragraph above and in the report.

Why it needs no budget change: map-ahead maps exactly the block the same tick's decode would map, so
the peak resident set per slot is unchanged (`ensure_rows` still clamps at `geo.max_ctx`), and the
run-level identity holds — **276,480 driver mappings in both arms**. At the cap it refuses rather
than exceeds: a failed map-ahead only warns, leaves the frontier where the chunk put it, and the
decode's `vmm_ensure` retries as the backstop (unit test
`refused_map_ahead_holds_the_budget_and_the_decode_finishes_it`, plus boundary / mid-block / max_ctx
cases in `memory::vmm`).

Scope and open items: only the segment-major TP prefill path maps ahead — single-GPU
`AmdEngine::prefill_chunk`, the `PLOW_PREFILL_SEG_TIMING` diagnostic path and `prefill_packed_chunk`
(per-span frontiers) keep the old behaviour. The chunk still pays its own 26 ms / 624-map
`prefill_prepare` boundary cost, which could move into the PREVIOUS chunk's shadow the same way.
Decode-only ticks read +4.6 ms in the changed arm's tick phase (n=46) although the change does no
work there; the bench's median ITL moved +0.4 %, so this reads as arm-order drift rather than a cost.

## Glue ops between GEMM / attention / MoE (2026-09-11, branch `glue-fusion`)

Inventory + fusion table + measurements: `/root/.claude/jobs/c08d1232/tmp/reports/glue-ops-fusion.md`.

| Piece | Where | State |
|---|---|---|
| `residual3-rmsnorm-fuse` → `FusedResidual3Norm`: the MoE block boundary `RmsNorm(add(x, add(routed, shared)))` as one fused target (combine + residual + norm); Lean constructor/`expand`/theorem/`soundRules`; `fuse_glm` asserts it | `crates/rewrite/{egl,extract,bridge}`, `lean-plow/Plow/Rewrite.lean` | landed (9992d1a8); `cargo test -p rewrite`, `lake build Plow` green. Analysis-only: `rewrite` is still not on the emit path, and the TP collective seams (`PLOW_GLM_XR_RES`) have no graph form |
| `PLOW_GLM_DECODE_GLUE_CUS` (emit, default off): FP8 latent KV writer 1 WG → one wave per row; router top-k 304 → `rows` WGs; MoE combine 304 → `elem_cus` | `devgen/mla.rs`, `emit_config.rs` | landed (4deb1528); bit-identical. Measured at rung 20: KV writer 2.68 → 1.52 ms/step, router top-k 2.88 → 2.12, **step 93.56 → 93.05 (−0.5%)** — real but small; the decode step is gated on the collective and the native segments |
| `PLOW_XR_DEC_CUS=N` (emit, default unset): cap the decode one-shot `XReduce` workgroups (240 at rows 20) | `devgen/lib.rs` `emit_xreduce_gather` | landed; bit-identical; **measured a LOSS at N=60**: the collective's body is unchanged (19.40 → 19.63 ms/step — it is fabric-bound, not arrival-bound) and the step costs +6.7 ms. Kept as the record |
| `PLOW_GLM_XR_RES=1` (pre-existing, off in the frozen packet): Residual folded into the two-shot all-gather | — | **measured a LOSS**: all 156 `Residual` packets go (−11.5 ms) but `XReduceTwoShot` goes 161.8 → 181.2 (+19.4) — the AG's staggered pure copy becomes a read-modify-write. Chunk 937.91 → 942.55. A seam epilogue belongs on the reduce-scatter side (sequence-parallel), not the all-gather |
| Opt-in kernel arms, header default = shipped body: `-DPLOW_COMBINE_VEC=1` (8-wide k==1 combine), `-DPLOW_RN_ROWS=R` (multi-row RMSNorm loads), `-DPLOW_RESID_U=U` (unrolled residual); `PLOW_HSACO_EXTRA_DEFINES` hook in `build_gfx942.sh` (refuses packet-paired axes) recorded in `build_defines.json` | `runtime/amd/op_{moe,norm,elementwise}.h`, `scripts/build_gfx942.sh` | landed; objects byte-identical when the axes are unset. **Numerics-changing unless proven otherwise** (the screens show different tokens than the control objects; see the serving row). **Measured on 8 GPUs** (packet trace of a steady 8192 sparse chunk at prior 65k): `MoeCombinePf` **31.74 → 6.68 ms/chunk** (0.74 → 3.5 TB/s), `RmsNorm` 22.20 → 20.44, `Residual` 12.25 → 11.53, **chunk 963.65 → 937.91 ms (−2.7%)** |
| `PLOW_MLA_FOLD_TB_FLASH=1`: the token-blocked `MlaMergeFold` arm, default-on for `interp_prefill_*` since 2026-08-10, was never compiled into the object that runs the op on this recipe (segment 11 of the 8192 program is `object_class: flash`) | `scripts/build_gfx942.sh` | landed (d0da0781) default off, **measured NULL** (fd4217f5): fold 46.59 ms/chunk either way — at nsplit=1 the fold is VALU-bound (29 of ~163 TFLOP/s) and the 2 MiB W_uv panel is L2-resident, so dividing the stream buys nothing. Knob kept as the record |
| Not done, priced in the report §7: collective+norm epilogue (AG must stay flat 2 B), sequence-parallel GLM seams (K3's `PLOW_SEQ_PAR_SEAMS` shape, ~−70 ms/chunk est.), MFMA `MlaMergeFold` (~−37 ms/chunk est.), wave-per-token router top-k (~−18 ms), tagged one-shot for GLM decode (tag slot 20 KiB < 122,880 elems) | | |

Serving (20 × 70k/700/.14, C20, same binary and packet, objects the only variable): the candidate passes the 18-case retrieval screen 18/18 and the C8 identity screen (every answer correct; the post-answer near-tie divergence between rung 1 and rung 8 the unmodified packet also shows). The first bench round is **confounded, not a result**: 46.95 vs 49.82 tok/s, but the candidate server ran the retrieval screen (nine 68.8k-token prompts) just before its bench, and the TTFT gap appears exactly at requests 10–18 — the ones that landed on slots a 68.8k prompt had held, each paying `begin_slot`'s synchronous window unmap (1.9 s per recycle on this pre-`slot-recycle-lazy-unmap` binary). TPOT median, which the confound does not reach, is 249.6 vs 252.3 ms. **Clean interleaved A/B** (fresh server per arm, bench only, ctrl/cand/ctrl/cand, same binary/packet/request set, manifest-verified objects): ctrl 50.72 / 49.63, cand 50.85 / 51.45 out tok/s — **mean 50.17 → 51.15 (+1.94%)**, the candidate ahead in both rounds (+0.26%, +3.67%) and in every pairing (min cand > max ctrl), median TTFT −1.5% / −2.2%. The control's own round-to-round spread is 2.17%, so the size is "about +1–2%", the direction is not in doubt. With retrieval 18/18 the rule's two halves are formally met, **but the call is: keep `PLOW_COMBINE_VEC` opt-in for now.** The served evidence is positive, not decisive — the paired deltas (+0.13, +1.82 tok/s) straddle the control's 2.2% self-spread, and "every candidate run above every control run" has a 1-in-6 chance with no effect at n = 2 per arm. The trace is the stronger evidence (−25.7 ms/chunk on the op, reproduced to the digit across two builds), so a real +1–2% is likely; two more interleaved rounds would make the served number decisive (not queued — ask first). A flip must also wait for the standalone harness (`glue-fusion-bitid`) to show that the object-level token differences are rounding-level and not a tail-path bug — its ragged cases (1000 rows, 6,144,001 elements, T=1000) cover the paths the arms add. The serve-ab.sh "server error lines: 16" in every arm is benign: all 16 hits are the `mfault_ms` field of the LOAD PHASES info lines (the pattern matches "fault"), identical in count and content across all four arms. **Open:** the control-objects screens contradict "bit-identical" — each server is deterministic at C1 round-to-round, but the candidate's C1 outputs differ from the control's on 4/8 prompts at the same positions in both rounds, and 13/18 retrieval texts differ from the first token (all 18 pass in both arms). A second process per arm (`1789161787-glue-fusion-screen-twice`) separates an arm effect from cross-process nondeterminism (a candidate for which is the native AITER MoE route, whose prepare step zero-fills the output the `fmoe` kernel then accumulates into); the arms are numerics-changing unless proven otherwise. The flip does not need bit-identity: the rule for a numerics-changing default is a positive same-binary A/B plus retrieval 18/18 (the candidate has the 18/18), so the clean bench decides it. The combine in this packet runs at `k = 1` (the native MoE delivers one bf16 partial per token), so a reordered top-8 expert sum is not the mechanism; a standalone harness comparing the three device bodies byte-for-byte is built to settle it.

**Cross-process screens** (`glue-fusion-screen-twice`, a second fresh server per arm, same binary, packet and manifest-verified objects): the control's tokens change across server restarts — a second control server reproduces only 1 of 8 of the first control server's C1 (isolated) identity outputs, the candidate reproduces itself on 3 of 8, and cross-arm pairs land in the same 0–4 of 8 range. Every process is internally deterministic (each isolated count is identical in round 0 and round 1). So the earlier cand-vs-ctrl token differences are not evidence about the glue arms, and more generally the token-level identity screen cannot show bit-identity across restarts for any object or packet A/B on this recipe. The source of the cross-process variation is not measured (a candidate: the native AITER MoE route's accumulate-into-zeroed-output). The standalone kernel harness (`glue-fusion-bitid`) is still queued and is the direct test of the arms. (`identity rc=1` in every screen run — cand, ctrl, ctrl2, cand2 — is `tb_identity.py`'s exit when any round has an isolated-vs-packed divergence; it is the same greedy near-tie between decode rung 1 and rung 8 in every arm, not a failure.)

**Standalone kernel harness** (`glue-fusion-bitid`): the three arms are **bit-identical**. The device bodies were compiled from the production headers with each production object's exact `-D` set (8-wave prefill; 4-wave flash with `PLOW_WAVE_RED_DPP`), with and without `PLOW_COMBINE_VEC=1 PLOW_RN_ROWS=2 PLOW_RESID_U=4`, run with 304 blocks on identical inputs, and compared byte for byte: 606,699,522 bytes per build, identical in both geometries across all 12 cases, including the odd-sized ones (RMSNorm 1000×6144, residual 6,144,001 elements and scale 0.5, combine T=1000, the k=8 path the vector arm must not take, no residual, no shared). No edge-path bug. The claim covers the same device functions in a separate translation unit, not the interpreter megakernel object. With the cross-process result above, the served token differences are engine nondeterminism, not the arms. **Recommendation: flip `PLOW_COMBINE_VEC` (and the two smaller arms, as measured together) on for gfx942 objects.** The numerics rule no longer binds, and the evidence all points one way: −25.1 ms/chunk on the op in the trace, reproduced in two builds; chunk −2.7%; served +1.9% with the candidate ahead in both rounds; retrieval 18/18 in both arms; resource-identical objects. The flip is a `build_gfx942.sh` default plus a re-bless of `obj_baseline_gfx942.json` (today's baseline is recorded without the axes); no packet changes.

**Flipped on (coordinator approval, per the standing turn-on-what-qualifies instruction):** `scripts/build_gfx942.sh` now defaults `PLOW_COMBINE_VEC=1 PLOW_RN_ROWS=2 PLOW_RESID_U=4` for the gfx942 prefill and flash objects (rollback: each `=0`). Evidence: bit-identical in the standalone harness (both geometries, all 12 cases, ragged included); MoeCombinePf 31.7 → 6.7 ms and the 8192 chunk −2.7% in the trace; served +1.9% (candidate ahead in both interleaved rounds); retrieval 18/18 both arms. Decode, mixed, token-batch, packed and small/split MLA objects keep the shipped bodies (end-to-end effect not measured; the decode megakernel is at its register limit). `obj_baseline_gfx942.json` re-blessed from a build with the new defaults; per-object register/LDS/spill before and after are in the flip commit's message. Objects must be rebuilt to pick the defaults up; packets are unchanged.

Incident: the first object-build script wrote rebuilt objects THROUGH `serving-safe`'s symlinks (18:07:57–18:12:29 UTC); all four files restored and verified against preserved `.co` bundles — report §3b.

## Host shadowing of the prefill tick: phase 1's last piece (2026-09-11, branch `host-shadow`, `host-shadowing.md`)

Re-measured on this head (map-ahead and slot recycling on): a steady 8192 chunk tick carries
**~40 ms** of serial host work, not ~100: `prefill_prepare` 25.9 ms (24.98 of it the 624 driver
maps of the chunk's second 4096-row block), audit 6.0, prefix publish 19.5 ms mean on 5 of 9
chunks. One FIFO job, one pinned binary, tick log on 20 × 66000 × 96 out:

| piece | flag | tick-level effect | invariant (code comment + test) |
|---|---|---|---|
| A: map the next chunk's rows while this one drains | `PLOW_KV_MAP_NEXT_CHUNK` | prepare 25.9 → 0.9 ms, prefill-carrying tick 1078.7 → 1050.8 ms; driver maps 276,480 in both arms | maps exactly what the next `prefill_prepare` would, one chunk early (`map_ahead_of_the_next_chunk_moves_its_maps_without_adding_any`) |
| B: prefix publish into the next GPU drain window | `PLOW_AMD_PUBLISH_DEFER` | 19.5 ms per publishing chunk → 0.06; the 2.16 s of flushes run under decodes that are no slower (86.8 vs 87.4 ms) | snapshot rows < frontier are not written before the flush; `begin_slot` / attach / publish / release flush first (`deferred_publish_*`) |
| C: prefill counter audit through large BAR | `PLOW_TP_PREFILL_AUDIT_DIRECT` | 5.35 → 3.50 ms per chunk | same gates, same expectations, every chunk (a stale read can only fail loudly) |
| D: decode enqueue + re-arm | — | nothing to take: rank 0 still has ~1230 of the tick's dispatches in flight after the enqueue and ~1198 after the re-arm, so both run under the GPU; the serial decode host remainder is ~1.7 ms of 105 | not built |

Why the prefill audit stays every chunk and on the critical path: the next dispatch's `zero_xctr`
erases its evidence, and a timed-out collective corrupts KV every later row reads (and the prefix
cache would publish it). The device compact audit cannot be reused for prefill: `plow_xctr_audit`
has no `IndexTpPf` / `XReduceScatter` / `XAllGather` case and would fail every GLM chunk.

Decode cost per piece (the bench's median ITL rose 100.9 → 106.5 ms, so checked per arm):
decode-only tick median ctrl 105.1 / ctrl2 105.5 vs A 98.9, B 99.1, C 99.3, ABC 99.3 ms; decode
after a chunk 92.5 / 92.6 vs 85.8, 86.9, 85.6, 86.5; dispatches in flight after the decode enqueue
1232 vs 1239. No piece raises decode time. B's deferred publish lands only in the decode of a tick
that already carries a ~1 s chunk, and those decodes are no slower (86.8 ms with a flush, n = 96,
vs 87.4 without, n = 76); decode-only ticks never carry one. The bench ITL gap is the per-process
decode mode below: the ctrl configuration decoded slow in both tick arms but fast in its bench
process, the ABC configuration fast in its tick arm but slow in its bench process. No TTFT/ITL
trade-off, so all three flipped.

Projected on the 100-prompt run: A ≈ 19 s, B ≈ 10 s, C ≈ 2 s, D 0 — ≈ 31 s (−2 %). 40-prompt bench,
same binary, ctrl vs A+B+C: 55.10 → 55.69 out tok/s (+1.1 %), duration 514.9 → 509.4 s, mean TTFT 56.0 → 53.3 s; the two bench processes drew opposite decode modes (median ITL 100.9 vs 106.5 ms, see below), so this is a no-regression check, not the measurement. Retrieval on A+B+C: 18/18. Defaults: all three on (`=0` is each rollback); D not built. **Measurement note:** decode ticks split ~6 ms (~6 %) between server processes independently of every flag (both tick-arm controls 105 ms, all flagged arms 99 ms, yet the flagless bench process drew the fast mode); host enqueue, the SDMA re-arm and the GPU's per-dispatch retirement are all slower in the slow processes — suspected engine-thread NUMA placement against rank 0's busy-polled completion signal, under test (`PLOW_HSA_DRAIN_BLOCKED`, diagnostic).

## The ~6 % per-process decode split is the engine thread's socket; pinned by default (2026-09-12, `host-shadowing.md` §5)

Host-shadowing's A/B found GLM-5.3 TP8 decode ticks differing by ~6 ms between server processes
with no flag, config or load-time difference behind it (the ctrl-vs-ABC bench pair drew opposite
modes, median ITL 100.9 vs 106.5 ms). Job 2 (`host-shadow-j2`, one pinned binary, 20 × 16384 × 96
out at C20, `taskset -a` on all 412 server threads after READY, the engine thread
`plow-eng-glm-5.` sampled every 5 s; rank 0's GPU is on NUMA node 3, socket 0):

| arm | engine thread on | decode-only tick | decode after a chunk | DSTEP enqueue / re-arm / drain (µs/token) |
|---|---|---:|---:|---|
| free1 (unpinned) | node 0, socket 0 | 96.2 ms | 83.1 ms | 4290 / 2398 / 87 597 |
| near (pinned) | node 3, socket 0 | 97.5 ms | 83.9 ms | 4306 / 2433 / 88 846 |
| Cfree (unpinned, a job-1 flag) | node 3, socket 0 | 98.6 ms | 84.3 ms | |
| near2 (pinned, repeat) | node 3, socket 0 | 98.8 ms | 83.9 ms | |
| far (pinned) | node 7, socket 1 | **103.4 ms** | **90.2 ms** | **6566 / 3781 / 91 115** |
| free2 (unpinned) | node 7, socket 1 | **103.7 ms** | **90.3 ms** | |
| blocked (unpinned, blocked drain wait) | node 4, socket 1 | 102.8 ms | 89.9 ms | |
| blockedfar (pinned, blocked drain wait) | node 7, socket 1 | 103.5 ms | 90.7 ms | |

Every arm follows its socket, pinned or not, flag or not. On the far socket the host's AQL/kernarg
writes (+2.3 ms) and SDMA re-arm (+1.4 ms) are slower and the GPU retires the tick's ~1240
dispatches ~3 ms slower; single-shot host operations (audit, `zero_xctr`, reads) do not move. A
blocked drain wait does not help (103.5 vs 103.4 ms), so the busy poll is not the mechanism: it is
the socket of the thread that feeds every dispatch.

**Fix, default on:** `--amd-engine-affinity` / `PLOW_AMD_ENGINE_AFFINITY` = `auto` (default) | `off` |
a CPU list. `auto` pins the engine thread, once at its first tick, to every online CPU of the socket
that holds rank 0's GPU, derived from `HSA_AMD_AGENT_INFO_DOMAIN`/`_BDFID` → the device's sysfs
`numa_node` → the node's CPUs' `physical_package_id` (`exec::engine_affinity`, fake-sysfs test).
Only the engine thread is pinned, after load, so model load keeps every CPU. Qualifies without a
served A/B: the pinned arms are causal and the pin changes no numerics. Worth ~6 ms on every decode
tick that would have landed on socket 1 (half of all processes): ≈ 15 s of decode-only and ≈ 5 s
of post-chunk decodes on the 100-prompt run for such a process, and it removes a ±3 % per-process
term from every same-binary bench pair on this box. `PLOW_HSA_DRAIN_BLOCKED` (diagnostic, off)
stays for the record. Confirmation job `host-shadow-j3` (auto from an unpinned start; auto and off
with the whole process started on socket 1) follows.

## Decode tick: native decode GEMM overlap and grouping, measured (2026-09-12, branch `decode-latency`)

The rung-20 decode tick is 1431 AQL packets per rank (832 native `GemmLtPf`, 75 AITER MoE, 524
interpreter segments), each with the barrier bit and agent-scope fences. A `PLOW_TRACE_RAW`
timeline of one full-model step splits it into 64.8 ms of interpreter bodies and 27.4 ms of
intervals holding native work; a lone native GEMM between two interpreter segments costs ~31 µs
for ~14 µs of work, so each dependent transition costs ~8 µs. Two opt-in knobs (code on
`decode-latency`), both bit-identical by construction:

* `PLOW_AMD_DECODE_GEMM_OVERLAP` (runtime): a native decode GEMM that directly follows native
  GEMMs it is independent of (byte ranges, planned at load) is dispatched without the barrier bit.
  **Measured loss, keep off.** 7-layer TP8 packet, 14 runs: median +1.5 ms/step (+14 %). The
  PLOW_TRACE_RAW intervals show why: the native GEMM runs do not get shorter (a second GEMM still
  adds ~15 µs, so nothing overlaps usefully), and on the grouped packet every native interval grows
  40–50 µs (single GEMM 45 → 93 µs, three GEMMs 74 → 112 µs). A follow-up that also drops the
  followers' acquire fence was not run: it weakens a memory fence and needs the user's approval.
* `PLOW_GLM_DECODE_GEMM_GROUP` (emit): same-input native GEMMs emitted adjacent (shared gate/up
  after the router GEMM, so top-k and Glu share a segment; indexer k/weights/q beside their
  siblings). Same instructions and dependencies; pairing hash and objects unchanged; knob-off packet
  byte-identical. Rung-20 chain 1431 → 1293 packets (−138 interpreter segments at 78 layers).
  **Unproven.** One traced step: 16 fewer interpreter segments at 7 layers, −0.56 ms GPU span, a
  class-weighted 78-layer projection of −4.4 ms/tick; but over 14 runs the median is +0.25 ms/step,
  because each process lands on one of three levels (~9.9 / ~10.9 / ~12.5 ms at 7 layers) and that
  per-process spread swamps the effect.

Found on the way:
* `rocprofv3 --kernel-trace` deadlocks on TP8 decode: its queue interposition stalls the ranks'
  in-kernel rendezvous ("Async signal handler still waiting on signal"). Use `PLOW_TRACE_RAW`
  (`PLOW_TRACE_ALLRANKS=1` for every rank) for per-segment decode timing.
* No within-process decode drift at 7 layers: a C20 serve run held 11.26 ms median (p10 11.03,
  p90 11.52) over 2722 decode-only ticks, and every GPU held mclk 1300 / fclk 1800 under load, so
  DPM clocks are not what moves the full-model tick between ~95 and ~108 ms; that needs full scale.
* At the one-shot XReduce, ranks 4–7 arrive 107–180 µs after ranks 0–3 in 12–14 of every 14
  collectives, in every traced process; ranks 0–3 spend that time waiting inside the op.
* `amd-bench --prompt` (non-batched TP) decode numbers taken before b11475c0 ran the widest decode
  rung (rung 20) with stale rows, not a single-sequence decode, and are invalid.
* A `--layers N` packet never pairs with the full object set: the pairing hash folds in
  `tuning.tile_lookups`, which scales with layer count.

Report: `/root/.claude/jobs/c08d1232/tmp/reports/decode-latency.md`.

## Kernarg rings and host staging on the GPU's socket (2026-09-12, branch `hsa-numa-kernarg`)

`HsaBackend::new` took the first CPU agent's pools for every GPU, so all eight kernarg rings and
pinned staging buffers lived on NUMA node 0, and ranks 4–7 (socket 1) read every dispatch's
kernargs across the socket link. At the one-shot decode XReduce they arrived 107–180 µs late in
every traced process; the collective agent's split put the extra ~100 µs per window in the
non-interpreter gap (native kernels + launch), not in interpreter work or the collective.

`PLOW_AMD_NUMA_HOST_POOLS` (3385e6e4) takes each rank's fine-grained and kernarg pools from its
GPU's nearest CPU agent and names that agent in host-memory copies. No fine-grained allocation is
shared across ranks (peer buffers are VRAM). Every load logs one `HSA host placement` line per rank.

Tier 3 (7-layer TP8, 6 processes per arm interleaved, all-rank `PLOW_TRACE_RAW`): **12.576 →
9.665 ms/step median (−23 %)**, spread 1.8 → 0.23 ms (the per-process slow/bimodal levels are
gone), ranks 0–3 time inside XReduce 196 → 39 µs, mean window between collectives 682 → 525 µs.
Projected ≈ −25 ms per 78-layer rung-20 tick (92 → ~67 ms), ≈ +9 % on the 100-prompt run. Off by
default; flip pending the combined tier 4.

Open: a ~65 µs per-window socket asymmetry remains and flips sign per process; best hypothesis is
the unpinned `amd-bench` host thread (GPUs on the other socket snoop the writer's cache on every
dispatch read), testable with `taskset` per socket; the fix would be device-memory or
write-combined kernargs. Placement anomalies (a few rings on a sibling node of the right socket)
are page-cache pressure: nodes 1, 2, 3, 5, 6 had ≤ 1 GB free, so the kernel fell back from the
preferred node. `nearest_cpu_node` in the log is a KFD topology id, not a Linux NUMA id.

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

## The qualified GLM recipe is the default (2026-09-11, `glm-production-defaults`)

The packet serving GLM-5.3 on 8x MI300X was emitted by naming eight `glm_*` flags, every one of
which was `default_value_t = false`. So the qualified configuration lived as an incantation in a
shell script, and forgetting one emitted a different, slower packet that still loaded and still
served — a silent regression with no failing gate.

`PLOW_GLM_FP8_KV`, `PLOW_GLM_MOE_AITER`, `PLOW_GLM_MOE_RESIDENT`, `PLOW_GLM_INDEX_TP`,
`PLOW_GLM_SELECT_LOCAL`, `PLOW_GLM_DECODE_NORM_ROWS`, `PLOW_GLM_GEMM_LT` and
`PLOW_GLM_GEMM_LT_DECODE` are now `Option<bool>` (the `emit_packed_prefill` pattern) with a
resolved accessor each, and `apply_production_defaults` sets the unset ones for
`capabilities.glm && gfx942 && tp == 8 && n_cu == 304 && !mxfp4`. That predicate is the emit
sites' own: four of these knobs assert gfx942 / TP8 / 304 CU, so a wider default would turn a
working TP1 or MI300A GLM emit into a panic. A new `glm` capability rather than reusing
`packed_prefill_siblings`, which names an emitter capability and not a measured recipe.

Precedence: explicit flag > env > `--replay-knobs` > production default > plain default. The
replay half needed no change — `apply_replay_knobs` `set_var`s before clap parses, so a replayed
value (including `false`) arrives as `ValueSource::EnvVariable` and outranks the default.
`build.json` records the winner per knob and omits production defaults from `replay`.

Byte identity, real emits, GLM-5.3-plow-lite TP8 gfx942 (`plowc` built at `worktree-tp-merge`
44925e02 vs this branch; `PLOW_UNISEG=0 PLOW_GLM_DSA_PF_SPAN=3 PLOW_MLA_PF_V2=1
PLOW_MLA_PF_AITER=1` in every arm):

| arm | `model.pkt` |
|---|---|
| full explicit flag list, base `plowc` | `a6621d32` |
| full explicit flag list, this branch | `a6621d32` |
| **no `glm_*` flags at all**, this branch | `a6621d32` |
| all eight `=false` | `943f8f7d` (104.0 MB vs 73.2 MB) |
| replay of the all-`false` `build.json`, no flags | `943f8f7d` |
| replay of the unflagged `build.json` | `a6621d32` |
| GLM gfx942 **TP4** (not qualified), base vs branch | `51f800a3` both |
| GLM **sm_90a** TP8, base vs branch | `809f04e7` both |
| Gemma-4-12B gfx942 TP1, base vs branch | `bad5030e` both |

Tests: `emit_capabilities.rs` gains the target matrix and two precedence tests reading the knob
record; `glm_tests.rs` gains a three-arm emit comparison (explicit / unflagged / rolled back)
plus an off-plus-one probe showing each of the eight still moves the packet. That probe is from
OFF rather than from ON because `native_moe = glm_moe_aiter || glm_moe_resident`, so dropping
AITER out of the full recipe changes nothing.

NOT defaulted, and why: `PLOW_GLM_MOE_FLAT_DECODE` and `PLOW_GLM_MLA_DEC_AITER` (serving
qualification pending; the latter is measured per layer only), `PLOW_GLM_FOLD_LT` (+0.49% with
P99 +3.6%), `PLOW_GLM_PLACE_PF` (+2.2% with P99 TPOT +10.5% — a retirement candidate above),
`PLOW_GLM_FUSE_ROPE` (unmeasured on this recipe), `PLOW_TOKEN_BATCH_TP` and
`PLOW_PACKED_SPARSE_PF` (both unqualified on device; see #22 and the token-batch section).
`PLOW_GLM_DSA_PF` is also left alone — sparse prefill is its own qualification — and the emits
above show it makes no difference to this packet either way.

Two things this does not fix, both pre-existing: the emit still panics rather than falling back
if a GLM checkpoint at this target is NOT block-fp8 (the native MoE arms assert `Fp8Blk`, which
`apply_production_defaults` cannot see — it reads the checkpoint's `quantization_config`), and
`--glm-ofold` now collides with the defaulted `PLOW_GLM_FP8_KV` (the existing assert names both
and says to unset one). Both are loud failures with a named rollback.

## FP8 block-scale prefill projections: eligibility and asm screen (2026-09-11, `fp8-prefill-gemms.md`)

Branch `fp8-prefill-gemms`. **Weights:** `q_a`, `kv_a_latent` (a block-row-aligned slice of
`kv_a_proj_with_mqa`), indexer `wq_b` and `o_proj` are checkpoint block-FP8 tensors with intact
`[128,128]` grids, so no requantization is needed. `q_absorb` (an einsum of `kv_b` and `q_b`),
`q_rope` (a slice not aligned to 128-row blocks) and `v_absorb` are prep-derived and stay BF16. The
shared expert is also eligible, but it is scoped to the MoE fold (expert 257).

**Code:** the new `exec/amd_gemm_blk.rs` loads AITER's gfx942
`fp8gemm_bf16_blockscale_BpreShuffle_*x128.co`, hash-pinned. Like the AITER MoE objects, the
descriptor's `KERNARG_SIZE` is zero; it is normalised after hashing. The resource ABI and pinned
shapes are refused by name. It is runtime-only: no emit half, no flag, no serving path. Nothing
changes numerics yet, so there is no retrieval screen or serving A/B.

**Screen:** one MI300X at M=8192. Time / throughput per projection: q_a 263 us / 783 TF/s, kv_a 92
us / 563, o_proj 281 us / 733, wq_b 195 us / 706. That is 84-94 % of AITER's gfx942 **CK** table
(838-873 TF/s). The kernel against its own operands is at 3.3e-3 rel-L2; the W8A8 floor against
FP64-of-BF16 is 3.6-3.8e-2 (the MoE route's floor is 4.0-4.4e-2). Standalone, the route is worth -40
ms/chunk, against -46 on CK; in flow expect less, because o_proj has to leave the post-attention
interpreter segment. Next: the emit half on these asm objects. CK is worth about 6 ms/chunk more,
which does not justify a new build dependency.

## FP8 block-scale prefill route served (2026-09-12, `fp8-prefill-gemms.md` §7)

Branch `fp8-prefill-gemms`, on `594737d9`: route `bc8849a5`, flags row `85d40176`, then this
entry. `PLOW_GLM_GEMM_BLK` is opt-in and default off, and the packet is byte-identical when off.
It runs `q_a_proj`, `kv_a_latent`, indexer `wq_b` and `o_proj` as W8A8 on AITER's gfx942
pre-shuffled assembly. The weights are the checkpoint's own FP8 bytes (overlay:
`scripts/glm53_prep_blk.py`), and the activation quant runs inside the GEMM's native segment.
`q_absorb` stays BF16, and the shared expert is left to the MoE fold.

**Numerics.** On captured layer-77 activations, per-row rel-L2 against the BF16 path is 1.8e-2
(q_a), 2.0e-2 (kv_a), 1.9e-2 (o_proj) and 2.4e-2 (wq_b), with no row above 3.3e-2. Retrieval is
18/18 in both treatment arms.

**Served A/B, not decisive.** 20 prompts, C20, two controls, the bench's main run only: the same 161
steady 8192-row chunks per arm. Out tok/s: 55.00 and 56.08 for the controls; 55.64 with all four
projections (+0.2 % vs the control mean); 54.71 without o_proj (-1.5 %). Both treatments sit inside
the controls' 2 % spread. Per-chunk drain: -24.4 ms (-3.0 %) with all four, 0.0 without o_proj; the
controls agree to 0.6 ms. **o_proj is the whole gain**, matching its standalone prediction, so
splitting its segment did not cost it; q_a/kv_a/wq_b net zero at the current quant cost. **Not a
default**, but a qualified opt-in: ~-21 s per 100-prompt run. It holds ~2.4 GB/rank of extra HBM,
~1.0 GB of it for o_proj.

**Where the saving went.** It reaches the wall clock: prefill-tick wall falls 2.6-3.1 s (-1.5 %),
and against the control whose decode matches, the whole run is -2.8 s. It is hidden by decode-side
run-to-run drift. The second control's decode ticks ran 5 % faster (95.6 against 102.3 ms median)
with the same binary and packet, a 3.8 s swing, larger than the whole saving. Decode-tick variance
is now the dominant noise in these A/Bs; control it before trying to resolve 1-2 % prefill effects
end to end. Next here: a vectorized activation quant (14.8 to ~3.5 ms/chunk; built, awaiting GPU
validation), which should take the route to about -35 ms/chunk.

**Found on the way.** In `amd-bench --prompt`, greedy decode with the production packet and
objects hits a GPU memory fault after a clean prefill, and it wrote a 97.8 GB coredump into the
worktree. Scripts now set `ulimit -c 0` and `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`; ROCr writes
its own GPU coredump and does not honour the core rlimit.

## FP8 block-scale prefill route, second served A/B (2026-09-12, `fp8-prefill-gemms.md` §7e)

Second tier-4 A/B (`1789182456-fp8-blk-ab2`): ctrl1 -> route with the v2 quant -> ctrl2, retrieval
on the route arm. The binary was built from `d9fe7690`, which includes the socket pin `1cc0c868`
and the v2 quant `ea202992`. The packets and object sets are the frozen pairs from the first A/B.

**Result: a resolved no-gain.** Out tok/s: 57.05 / **56.60** / 56.50. The control spread is now 0.97
% (it was 2.0 % before the socket pin), and the route is -0.31 % against the control mean. Retrieval
18/18. Prefill saving: steady-chunk drain -24.7 ms (-3.2 %; the controls agree to 0.5 ms),
prefill-tick wall -4.1 s (-2.4 %), median TTFT -1.4 %. Decode took it back: the route arm's
decode-only ticks cost +4.6 s, and median ITL was 107.1 ms against 97.8 / 102.3.

**The ITL excess is not the route.** The decode program is identical (1,189 segments per tick,
in-flight enqueue 1,237 vs 1,239). KV geometry is identical (the same VMM pools, `n_kvrow=177`,
`kv_buffers=255`), and so are the KV block maps (2,040 per arm, in both A/Bs). The extra FP8
weights only enlarge the weight slab, 14,387 -> 16,712 MiB/rank. The same route packet decoded at
control speed in the first A/B (102.5 vs 102.3 ms). Within this run, the route arm's decode tick
swung 13 % over time (107-108, then 94.7, then ~104 ms), while the controls still differ by 3.8 %.
That is residual decode-state drift the socket pin reduced but did not remove.

**Call: stays opt-in.** A flip would need a resolved served gain, which this A/B does not show, and
the broader accuracy gate (GSM8K plus long retrieval at C20) for ~2e-2 per-projection error.

## FP8 keys for the sparse MLA prefill: measured negative (2026-09-12, `sparse-mla-fp8-keys.md`)

Code on branch `sparse-mla-fp8-keys` (7fbd385a + 104a1ec9), opt-in `PLOW_MLA_PF_FP8_KEYS`, **not merged**.
It routes the sparse MLA prefill through AITER's gfx942 QH8 a8w8 object (`mla_a8w8_qh8_qseqlen1_gqaratio8_v1.co`)
with a per-launch requantisation: an amax pass; a pack to e4m3fnuz with one Q scale and one K scale; the
attention; then a widen ×Sk into the FP32 buffer the fold reads.

* **Why the earlier attempt returned NaNs** (read off the object's disassembly): gfx942 FP8 is e4m3fnuz, so
  plow's OCP bytes read as half their value and OCP −0 (0x80) is fnuz NaN; at one KV split the kernel writes
  BF16 (`s_cmp_eq_u32 kv_split, 1 → R_write_out_bf16`), not FP32; and it never applies DSK to its output.
* **Shape measured:** the steady 8192-row chunk at prior 65,536 (ctx 73,728), real top-2048 selections,
  layers 0–3 captured from a 76,000-token real-text request on the truncated TP8 packet (`plowc --layers 4`),
  one GPU (job `0-1789186943-fp8keys-tier2`).
* **Per-launch split** (drained, medians over 4 layers × 11 launches): BF16 single-pass
  130.9 + 2101.6 = **2232.5 µs/layer**; FP8 keys 70.1 (amax) + 83.7 (pack) + 1918.8 (attention) + 81.8 (widen)
  = **2154.4 µs/layer**. Attention −8.7%, route −78 µs/layer: **−6.1 ms per 8192 chunk (×78)**. The ceiling,
  with free requantisation, is the attention alone at **−14.3 ms**.
* **Numerics**, attention-output rel-L2 against an f64 reference over the exact dequantised cache (every 16th
  row × 8 heads): FP8 keys 2.01e-2 / 3.61e-2 / 3.22e-2 / 7.56e-2 for layers 0–3 (worst single head-row 0.046 /
  0.15 / 0.18 / 0.52), against 1.1e-3 / 6.8e-4 / 5.7e-4 / 1.5e-3 for the BF16 route. The device numbers match
  an offline model of the same rounding (2.0e-2 / 3.8e-2 / 3.4e-2 / 7.1e-2), so this is the kernel's floor: Q
  and K each lose ~2.5% to e4m3's 3-bit mantissa, and the one K scale must also cover rope keys 60–215× the
  latent. Rebalancing the rope between Q and K (exact for the scores) did not help in the model.
* **Verdict:** no-go. −6 ms per chunk does not buy a 20–50× worse attention error, so no truncated-packet or
  served run was spent on it.
* **Why v1 misses the 2× the halved gather bytes suggested:** not established. Both objects run one 256-thread
  workgroup per CU (64 KB LDS each); the v1 a8w8 kernel is an older design than the v3 BF16 one. A plausible
  reading is that at that occupancy the per-query 2048-row gather is latency-bound rather than
  Infinity-Cache-bandwidth-bound, but no counter data was taken.

## Decode rows riding the 8192 body: measured negative (2026-09-12, job `0-1789187400-tb-fix-price2`)

Code on branch `tb-body-chunk-cap` (2d8c62cf + 160c5396), **not merged**. With bodies armed, a middle
chunk on the 8192 bucket is capped at the 8192 body's span (8172 rows: 8192 minus the 20-row band) so it
can ride the body with the tick's decode rows. The cap applies only where the planner's launch count is
kept. On the seed-0 100-prompt lengths it binds in 3 prompts, and 694 of 707 body-admissible middle
chunks ride.

* **Shape measured:** truncated TP8 span-aware body packets (`PLOW_LAYERS=4` and `=8`, bodies
  128/512/2048/8192, each on its own stamped objects). Load: 19 decoders plus five ~28.6k-token prompts,
  with `PLOW_TICK_LOG=1`. Control `PLOW_TOKEN_BATCH=0` against body `=1`, 10 matched middle-chunk ticks
  per point. Per layer = (N=8 − N=4) / 4; fixed = the intercept; 78 layers = N=8 + 70 × per layer.

| Median tick | 4 layers | 8 layers | per layer | fixed | 78 layers |
|---|---|---|---|---|---|
| Isolated 8192 chunk | 50.72 ms | 96.03 ms | 11.33 ms | 5.42 ms | 889 ms |
| Separate decode pass (19 rows) | 6.55 ms | 11.60 ms | 1.26 ms | 1.49 ms | 100 ms |
| Control: chunk + decode pass | 57.24 ms | 107.64 ms | 12.60 ms | 6.84 ms | 990 ms |
| Body: 8172 rows + decode rows | 57.19 ms | 108.31 ms | 12.78 ms | 6.06 ms | 1003 ms |

* **Result:** at 78 layers the body saves **−14.0 ms per middle chunk**, i.e. it is slower (bootstrap
  90% [−61.7, +15.9] ms). Over the run's 707 ridable middle chunks that is −9.9 s [−43.6, +11.2] s.
  Final chunks, which already ride without the cap, save +6.7 ms [−2.5, +14.5] per chunk.
* **Why the roadmap's −54 to −63 s does not appear:** it assumed the separate decode pass (~100 ms per
  8192 tick at 78 layers) disappears when the decode rows ride. It does not. Inside the body the band
  still runs the decode attention chain in every layer, after the prefill fold. That chain costs
  1.46 ms/layer above the isolated chunk, against 1.26 ms/layer for the whole separate pass. Only about
  0.8 ms of fixed per-tick cost is saved.
* **What could make it pay:** overlap, not planning (estimate only, not built). The band chain would have
  to run concurrently with the prefill's fabric-bound two-shot collectives, about 1.1 ms per seam and
  two seams per layer, with half the CUs idle at `PLOW_XR_CUS=152`. The ceiling is the decode pass itself:
  about 100 ms per middle chunk (about 71 s over 707). If only the attention seam can be used, it is
  about 72 ms (about 51 s).
* **Known limitation (code read, not run; fix before bodies are reconsidered for default):** the body
  path ignores `PLOW_AMD_TAIL_SPARSE_CTX`.
  - `token_batch_prefill_rows` takes a member's rows from `step.clen` (`token_batch_cursor_rows`), and the
    body width from that row count alone (`rows_for`, `body_for_capacity`), never from `step.prog`.
  - `packed_span_admissible` refuses only sparse programs, only below `SPAN_MIN_PRIOR`, so a dense body
    is admissible at any depth.
  - So a deep-context short final chunk that `retarget_dense_tail` planned onto the sparse 8192 program
    (e.g. 1,500 rows at 65K prior) rides the dense 2048 body. That is the dense-at-depth path the
    retarget exists to avoid: 2.4–6.8× per the serving-8k agent's read. The retarget's own note
    measures a 464-row dense tail at 65K prior at 1.9 s, twice a full sparse 8192 chunk.
* **Verdict:** no-go. The branch stays unmerged and the token-batch body stays opt-in.

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
| H18 | cuBLASLt prefill row set narrowed to {128,256,512} (454a2564) | unrun: A/B the two Gemma-4 projections at M ∈ {1,2,4,8,16,32,64} native vs cuBLASLt (`scripts/gemma4_h100_kernel_tuner.py` + `bf16_decode_exact_sm90_bench`), confirm M256/512 gain, then `scripts/gemma4_prefill_role_full_logits.sh` for the logit gate. See #30 |
| H19 | CUDA live rings auto-on at max_ctx ≥ 128K (7c2cd12f) | unrun: TTFT/TPOT + peak memory, live rings vs `--nv-vmm-live-rings=false`, 128K Gemma-4 packet; see #34 |
| H20 | Demand-driven live VMM KV (184d2337): eager row-0 map removed, ring units ref-counted/released | unrun: batch ≥ 64 or 128K packet, concurrency below batch (unfed slots), watch for faults on unmapped KV; memory reclaim after release; see #35 |
| H21 | Prefix-hit suffixes on H100 run as isolated small-M `prefill_chunk` launches by default (#36): `pf_batch_cuda` off, and `PLOW_PF_BATCH=1` switches the VMM prefix layout auto-select off (`select_vmm_prefix_layout`) | unrun: multi-turn traffic (shared 8k–32k prefix, 100–500-token suffixes, c8/c16) A/B of default vs `PLOW_PF_BATCH=1 PLOW_NV_VMM_PREFIX=1` vs `PLOW_TOKEN_BATCH=1`; count coalesced launches vs isolated; then decide whether CUDA follows AMD's `pf_batch` default and whether the prefix-layout gate should stop keying on `pf_batch` |

### The perf gaps that are not knobs (kernel work, ranked by gap × share)
1. Decode attention hd512: 5.5–7.9× vs vLLM attention lib — CUDA-core softmax body, no TMA, 208 regs (`op_attention.cuh:588`).
2. Prefill attention hd256: 4–5× — 2 stages, both warpgroups recompute S, no producer/consumer.
3. Decode GEMV bf16 B8/B16: 1.6–4.5× vs cuBLASLt — FFMA ladder, `GV_MM_MAX=8`.
4. Megakernel slot cap: TPOT flat 76.7 ms c8→c64 vs vLLM 24.9→42.4.
5. Prefill GEMM: 175/219 TF/s uniform vs ~450 warp-specialized (ws384 already served for Gemma-4).

### Housekeeping on the H100 box
- `tuned_tile_selection`: the `nvidia/sm_90a/h100-nvl` tunedb cell may be stale against the kernel-source digest as well; re-run the decode campaign + `plowc tune ingest` if `plowc tune status` says so.
- Run the self-hosted CI job's `cargo test --workspace` there (the hosted job only checks the CUDA features).

### Slot recycling off the engine thread (2026-09-11, branch `slot-recycle-lazy-unmap`)

Measured on GLM-5.3 TP8 (`PLOW_TICK_LOG=1` + `begin_seq` debug lines): recycling a slot whose
previous occupant held a 66k-token prompt cost 1.9 s on the engine thread (2.1 s under the C20
workload) — `VmmKv::begin_seq` → `release_window` issuing one `hsa_amd_vmem_unmap` per mapped block,
1905 per rank across the packet's three cache groups, ~15 k per recycle, at 138 µs each (probe:
map 5.5 µs, set_access 9.4, create 18, release 10.6, unmap 138; a range unmap over 8 granules costs
8 × 133 µs; 2/8 threads scale 0.96×/0.90× — ROCr serializes). Even a "fresh" slot paid 182 ms
unmapping the warm-up's one column per track.

Change (`memory/vmm.rs`, opt-in via `enable_deferred_reclaim`, wired by the AMD shared-prefix groups
and `PLOW_VMM_KV`; `PLOW_VMM_DEFERRED_RECLAIM=0` restores the synchronous path; CUDA untouched):
`begin_seq` keeps a private row-0 block in place (no driver call), unmaps a cache-shared row-0 block
inline (the idle decode row parks at `pos = 0` and prefill writes row 0 at once — the floor with a
frozen packet), and marks the rest of the window `Slot::Stale` for the pool's existing background
thread, which unmaps lowest-column-first under the pool lock (one block per hold); `ensure_rows`
reuses a private stale column in place or clears a shared one inline if it gets there first;
`try_attach`/`Drop` drop stale slots with the window; zero-ref handle releases go to the thread
too. A process-wide reclaim gate + engine-priority flag keeps the reclaimers from barging on ROCr's
memory lock (v1 without it: engine maps 5 µs → 1 ms, `prefill_prepare` p90 26 → 891 ms). Budget
unchanged (stale blocks reach the pool/driver as soon as the thread runs; pool and cache caps as
before). Five mock-backend tests; shared-prefix budget test syncs the thread.

Result: recycled-slot first-chunk `cursor` 2151 → 1 ms (private blocks, the user's workload), fresh
slot 214 → 1 ms; cache-held column 0 still pays the swap. 40 × 70k C20 bench, same packet:
**48.72 → 52.84 out tok/s (+8.5 %), P99 ITL 3226 → 1178 ms, median TPOT 349 → 311 ms**; retrieval
screen 18/18. Report: `/root/.claude/jobs/c08d1232/tmp/reports/slot-recycling.md`.

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
| 2026-09-11 | local `glm53-mi300x` 3fe8ec8d (main checkout; 0 uncommitted tracked edits verified via index scan) | 515d8260 | none | check ws; devgen `gemv_staged_rows`/`emit_capabilities` 6/6; CPU suite 396/396 + 15. Reviewed: doc-comment + test correction only — the `PLOW_GEMV_WALK` B=16 prediction is falsified on gfx942 (decode arena is 15,360 halves, not gfx950's 73,728; B=16/MM=16 163.3 tok/s vs B=8 130.7 on Gemma-4-31B); `scripts/walk_b16_ab.sh`, new `docs/amd/gemma4-31b-mi300x.md` section. No emitter default change |
| 2026-09-11 | ac11af31 | (this merge) | none | check ws; plowrt cuda,hsa lib incl. shared_prefix + memory::vmm tests. Reviewed: AMD VMM pools (legacy KV pool and the shared-prefix cache groups) switch from eager block PRECREATION at load (`enable_block_pool`) to on-demand creation with recycling inside the same budget (`enable_block_recycling`); the test now asserts zero blocks created at load and reuse == created. Effect on MI300X: less memory held at load, first-touch block creation moves into the first prefills (a small TTFT cost on cold requests; the mock test does not price it). Same policy the CUDA side adopted in 184d2337 |
| 2026-09-11 | ba638609 | (this merge) | none | no Rust change (CMake + sm90 harness + build script). Reviewed: removes the DUPLICATE guarded sm90 harness target block this branch had added in its CI fix pass (upstream's own block, note #26, remains — verified by grep); `packed_flash_sm90_correct.cu` gains `--requests N` (multi-request packed attention rows with per-request KV slots and tensor maps); `build_sm90a_gemma4_segments.sh` builds the hd256/bkv64 attention object by default (`PLOW_BUILD_PFATTN_KV64` 0 → 1; role-gated at load, see 7c2cd12f) |
| 2026-09-11 | 184d2337 | (this merge) | none | check ws; plowrt cuda,hsa lib incl. memory::vmm / exec::gpu::prefix. Reviewed: CUDA live-VMM KV becomes demand driven (ref-counted ring units, release on slot release, `enable_block_pool` → `enable_block_recycling`), auto-enable widened to `batch >= 64` (#35/H20); per-segment-site wall-time logging in the CUDA prefill trace. No AMD surface; no raw artefacts |
| 2026-09-11 | 7c2cd12f | (this merge) | none | check ws; plowrt cuda,hsa lib; plow-asset/kernelcaps/tunedb; devgen attention_prefill_role. Reviewed: segment role 10 `PREFILL_ATTENTION_HD256_BKV64` (dedicated sm90a Gemma sliding-attention object, exact sha256 + `AttentionCapability` geometry + module globals checked at load, refused otherwise); `toolchain_label(isa)` public; VMM mock gains granularity/reserved accounting + a 128K live-KV test; `live_rings_for_context` default flip → #34/H19. No AMD surface; no raw artefacts |
| 2026-09-11 | 454a2564 | (this merge) | none | check ws + cuda,hsa; plow-asset segment_roles 7/7; devgen dense_cublaslt 6/6; plowrt exec::gpu::{cublaslt,decode_rung} + `no_raw_env_reads` 24/24 (9 GPU-gated); campaign py tests 36/36 (new tuner test). No raw results / campaign markdown in the batch (bench `.cu`, experiments `.cu`, scripts kept as tooling). Reviewed: cuBLASLt prefill row set → {128,256,512} (see #30/H18); `op_attention_sm90.cuh` gains `PLOW_NV_FA_WGITEM_ONE` (default 0, `#error`-guarded, production loop identical); `packed_flash_sm90_correct.cu` campaign flags refuse non-defaults unless `PLOW_TEST_FA_ROWS` is compiled in; new `#[ignore]` Gemma-4 prefill-role full-logit gate |
| 2026-09-11 | c58a9b11 | (this merge) | none | check ws; devgen GLM module 41/41 (new `single_row_prefill_gemv_preserves_bf16_projection_layout` is env-guarded); `no_raw_env_reads` ok (no new env reads). Reviewed: single-row (prefix-hit tail) GLM TP8 prefill projections switch GemmSmall→Gemv on gfx942 for measured shapes; see #29 |
| 2026-09-11 | 5b263108 | (this merge) | none | check ws + cuda,hsa; cpu suite 396/396; exec::amd_moe_aiter 8/8 (7 GPU-gated); devgen GLM module 40/40; packet 125/125. Reviewed: `MoeAiterFp8Pf.i6` becomes an output *mode* (0 sorted→FP32, 1 flat decode, 2 sorted→direct BF16); emitter defaults the GLM sorted A8 prefill route to mode 2; host refuses mode 2 unless the following `MoeCombinePf` bands (i7=1, H=6144) contiguously cover all T rows and the output aliases no input; output workspace 4→2 B/elt. See #28 |
| 2026-09-11 | 55ce86d7 (64b15dc0, 55ce86d7) | 701fc7f0 | none | check cuda,hsa; `metrics` test; 96 obs/serve lib tests; campaign tests 11/11. Reviewed: model-scoped Prometheus metrics (per-model `Arc<Metrics>`, unloaded models fold into the process totals), ladder-campaign refuses incomparable workloads / concurrent latency regressions |
| 2026-09-11 | agent branch `worktree-agent-a182a4b057a0ad61d` 2d72ba7d, 842b6171 (continuous batching by default) | 6b9b25ec, (second merge), d6496f9e | `packet/devbuild.rs` getter doc-comment, `docs/flags-reference.md` (kept both agents' rows); `tests/config_env.rs` needed the Option-typed `pf_interleave`/`pf_batch` | check ws all-targets; plowrt cuda,hsa lib 787/787; devgen GLM + packed_siblings 45/45. Reviewed: see "Continuous batching by default" above; default flips: AMD `PLOW_PF_INTERLEAVE` unset = widest rung, `PLOW_PF_BATCH` unset = on (oldest-first), route follows the packet, GLM gfx942 emits packed siblings (`production_default`). CUDA defaults unchanged |
| 2026-09-11 | pushed directly to `origin/worktree-tp-merge` by Codex: 44694e0c, 7dc66e21 | (this merge) | none | check plowrt cuda,hsa --tests. Reviewed: `serve/manager.rs` load plan subtracts the live-ring tensors' virtual bytes when `live_rings_for_capacity` says the rings are demand-backed (previously the plan charged them as resident, over-reserving KV at load; `exec/gpu.rs` fn made `pub(crate)`); `packed_flash_sm90_correct.cu` gains `--json`, `--mapped-only`, a per-request kv_length ramp (`kv_length_min..kv_length`, topology single/packed_homogeneous/packed_ragged) and drives the mux with the real row count instead of `capacity`; `scripts/gemma4_attention_h100_sweep.sh` uses them. CUDA-only, no AMD surface, no raw artefacts. H100: the harness change is the tool for H21/#36 (ragged packed rows) |
