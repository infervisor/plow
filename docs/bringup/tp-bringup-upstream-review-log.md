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
| 36 | 2026-09-11 | (planner merge 842b6171; pre-existing) | perf (prefix hits, both GPUs) | `serve/engine.rs` `packable_prefill_step` (AMD); `config.rs` `pf_batch_cuda` + `exec/gpu/prefix.rs` `select_vmm_prefix_layout` (CUDA) | Hit restoration does NOT bypass packing on either backend: AMD `prepare_prefill_cursor` plans only the suffix from `resume` (snap_after=None on a hit); the CUDA packed arm derives `kv_row0 = frontier` after `admit_packed_slot`. What defeats coalescing is (a) AMD: a span is packable only if it is not the cursor's final chunk (the prefill head samples one token per launch), so a hit whose suffix fits one rung is always isolated at rung 128/512/2048; (b) CUDA: `pf_batch_cuda` defaults OFF, so hits run through `gpu_prefill_advance` → isolated `prefill_chunk`, and turning packing on (`PLOW_PF_BATCH=1`) disables the prefix layout's auto-selection unless `PLOW_NV_VMM_PREFIX=1` is also explicit. The AMD mixed-step (Gemma) arm coalesces finals with last-token replay. Fix for (a) is the token-batch body (per-slot sampling rows; bodies exist at exactly 128/512/2048) once the multi-member collapse is closed; (b) is a default/gating change (H21) | open |
| 37 | 2026-09-11 | (infrastructure) | process | `perf-data/tools/gpulease` | The lease has **no fairness**: acquisition is a non-blocking scan retried until `GPU_LEASE_TIMEOUT`, so with several 8-GPU campaigns queued it is a thundering herd and the oldest waiter can starve arbitrarily. Observed today with seven campaigns queued: jobs waiting 54 and 67 minutes while a job queued 30 minutes later acquired. Total wall time is unaffected (the campaigns are serialised either way) but *which* result arrives first is random, so do not design a plan whose next step depends on a specific campaign finishing first, and set `GPU_LEASE_TIMEOUT` generously (14400) on every queued job. A FIFO ticket file would fix it | mitigated for this session by #38; gpulease itself unchanged |
| 38 | 2026-09-11 | (infrastructure) | process | `/root/.claude/jobs/c08d1232/tmp/gpuq/` | **FIFO GPU queue** replacing per-campaign lease races: one runner (`runner.py`) holds all 8 GPUs under a single `gpulease -n 8` hold and runs spool jobs back to back in submission order; agents submit with `submit.sh <label> <ngpu> <cmd...>` from the job's working directory. Jobs run under `nix develop --command` from that directory and **no environment is captured** (the auto-mode classifier rightly refused copying other processes' environments to disk), so campaigns set their own knobs or pass `env VAR=value`. Two lessons it encodes: a whole-box `gpulease` hold does NOT export `ROCR_VISIBLE_DEVICES` (it only pins partial leases), so unset means every card; and every job preflights its paths before submitting. | in use |
| 39 | 2026-09-11 | d60bdeb2 (gap surfaced today) | correctness (load) | `/tmp/tp-glm53-pswz/serving-safe` | Packets emitted by current HEAD `plowc` carry GLM small-MLA / FP8-KV split segments, and the frozen serving-safe object set predates their objects, so ANY freshly emitted packet run against it dies at load with `small MLA segments require .../interp_mla_split_fp8kv_gq.elf` (the per-rung campaign lost a hold to it). The frozen packet itself is unaffected. Per object dir: build the `interp_mla_small*` / `interp_mla_split*` rows with `PLOW_ROWS_ONLY=interp_mla_s PLOW_HSACO_CONFIG=<packet>/plow_config.h`, and list `interp_mla_split_fp8kv_gq.elf` as required in preflights. Related: `scripts/freeze_serving_set.sh` and `scripts/pack_objset.py` copy only `*.elf`, so they silently drop every pinned vendor `.co` (the 64-row MoE tile included). Also note the frozen set is no longer byte-for-byte what was frozen: the MoE agent linked `fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_psx_64x256.co` into it (merge a9951294). Load behaviour there is unchanged, because both new MoE defaults are auto-on only when the adapter carries its marker (`plow_moe_aiter_tile64_abi_1`, `plow_moe_aiter_swizzle_abi_1`) and serving-safe's adapter predates both. Durable fix: regenerate the serving set from HEAD (the per-rung agent has a script for it).  **Stamping:** build these rows UNSTAMPED (no `PLOW_HSACO_CONFIG`) whenever one object set serves several packets — a stamped object is refused against any other packet, and packets differ in hash by knob (e.g. a no-GEMM_LT emit is `0xe8cc92348e494a59` against `0xfb318c7e3408f87b`); the rest of serving-safe is unstamped. Stamp only a set used by exactly one packet (as `hsaco-tb2` is, for `assets-body2`). **Durable fix landed** (merge of 95ecee21): `scripts/build_glm53_gfx942_serving_objects.sh OUT ASSETS VENDOR` regenerates the whole set (any existing serving set is a valid VENDOR dir); `freeze_serving_set.sh` now carries `*.co`; `pack_objset.py` records each `.co` as a vendor object, so the objset id changes when one goes missing. | fix landed; serving-safe itself not yet regenerated |
| 40 | 2026-09-11 | (process) | correctness (measurement) | `/app/plow/.claude/worktrees/tp-merge` | **A Codex agent works inside the shared review worktree** (parent process `codex`, cwd tp-merge). It created branches `gemma4-12b-mi300x-vllm` and `...-rungs` there, switching the checkout mid-merge twice; leaves uncommitted token-batch edits (`token_batch_rows(sample_rows, decode_rows, prefill_rows)`) in `serve/{mux,engine}.rs` and `exec/amd/mixed_step.rs`; and **rebuilt `target/release/plowrt` at 19:32 from that uncommitted work**. Never measure `tp-merge/target/*` binaries; the pinned clean binary built from the committed head is `/root/.claude/jobs/c08d1232/tmp/gpuq/bin/plowrt` (sha 2b51868e). Merges here now check the branch before pushing, verify from a clean export of the committed head, and commit only the index. | open, needs the user to relocate that agent |
| 41 | 2026-09-11 | (process) | correctness (measurement) | `/tmp/tp-glm53-pswz/serving-safe` | **The "frozen" serving set is a symlink farm into mutable build dirs.** Every `interp_decode*.elf` is a symlink (the GLM TP8 decode object `interp_decode_fp8kv_gq.elf` into `/tmp/tp-glm53-rungs/objects/`, `interp_decode_gq.elf` into `/tmp/tp-glm53-select-local/decode-objects/`), and the low-rung decode tiers `lowrung1/2/4/8` next to it are DIRECTORY symlinks the runtime loads when `PLOW_HSACO_LOWRUNG` is unset. Those targets were rebuilt in place at **18:43:12** while the set's own mtime stayed at 18:05, so nothing looked different from outside (found by the per-rung agent). The rebuilt content is byte-identical to a HEAD build and keeps `plow_dsa_decode_batch_arm`, but **it is STAMPED for the production packet (0xfb318c7e3408f87b), not unstamped** as first reported. It is harmless only for packets with that hash. The loader correctly refuses it for any other packet (`specialised AMD object ... interp_decode_fp8kv_gq.elf stamps 0xfb318c7e..., asset requires 0x78432b97...`). That refusal cost the collapse requalification its slot, because freezing hsaco-tb2 by dereferencing pulled the stamped object in for the bodies packet; its decode objects were rebuilt for that packet and the job requeued. A manifest proves the bytes did not change, not that they match the packet, so a frozen set used with a non-production packet must have its specialised objects built for that packet. Separately, but a rebuild during a hold changes objects between one job's arms. The lease log shows exactly one hold straddled it, `glue-fusion-main` (18:34:44 → after 19:05); its object dirs were assembled dereferenced, and that is being confirmed. **Cross-campaign comparisons spanning 18:43 compared different decode objects.** Rule: before a hold, freeze every object dir a job loads with `cp -rL` (including the tier subdirectories), write a manifest over every depth (`find . -type f \( -name '*.elf' -o -name '*.co' \)` → `MANIFEST.sha256`), require `find <dir> -type l` to be empty, and have the job run `sha256sum -c` before any server starts. Frozen here: `hsaco-tb2` (83 objects) and `serving-safe-frozen` (69). | open |
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
* **It is a no-op on the target workload.** At `--random-input-len 70000 --random-range-ratio 0.14` every prompt is 60.2k–79.8k tokens, so after the 8192-row chunks every tail is 2049..8192 rows and ALREADY lands in the sparse bucket. 20-prompt bench: control **48.79** vs arm **47.25** out tok/s — the arm changed no chunk in that run, so the 3.2 % is campaign drift (same-binary controls across today: 49.68, 49.79, 48.79), not the flag. The flag's A/B needs short-suffix traffic (multi-turn / prefix-cache hits), i.e. the #36 / H21 workload.
* Stays **off by default**: for a tail at long prior context the sparse arm selects top-2048 keys where the dense bucket attended over every prior key. That is the same approximation the model's own DSA path uses for every wide chunk, but it is a numerics change for those rows, so a retrieval screen on this arm is owed before any default flip.
* The attribution's "tails cost 1.9 s" lever therefore lands on the 464-row-class tail (dense 512 bucket at 65k prior, 1.89 s) that appears when a request's length is just past a multiple of 8192 — real but rarer than the report's 21-of-21 sample suggested, because that sample used one fixed 66,000-token prompt.

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
