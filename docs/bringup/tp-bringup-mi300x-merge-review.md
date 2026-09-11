# tp-bringup-mi300x → main: merge review and fix plan

Reviewed 2026-09-11 on `worktree-tp-merge` = `origin/tp-bringup-mi300x` (edfd8387, 207 commits)
⊕ `origin/main` (84ce7840, PR #28 plow-dist). Textual merge clean (042702de). Host: 8× MI300X;
no NVIDIA GPU, so every H100 statement is static review plus recorded measurements.

## Verdict

Mergeable after Phase 1. Nothing on the default kernel path regresses either vendor; the risk is
three runtime defaults flipped for all backends without an isolated A/B on either GPU.

| Gate | State |
|---|---|
| `.github/workflows/build.yml` | **main's workflow does not parse** (`--lib dist::` plain scalar, line 105; run 34434255859 started zero jobs). Quoted on this branch. |
| `cargo fmt --all --check` (nix rustfmt 1.95) | branch-only drift formatted: `serve/mux.rs` + 17 files (f6a06e69, b314a719). 122 hunks remain vs main's 74 — all in `exec/amd.rs`, `exec/gpu.rs`, `serve/engine.rs`, `devgen/lib.rs`, `manifest.rs`, which drift on both sides and cannot be split by author. CI's `rust-toolchain@stable` rustfmt may differ from nix's. |
| `cargo check --workspace --all-targets`, `-p plowrt --features cuda,hsa` | ok |
| `cargo test -p plowrt --features cpu` | `/tokenize` default test fixed (pre-existing on main) |
| `cargo test --workspace` | devgen `no_raw_env_reads` fixed (branch); kernelcaps `hopper_wgmma_gemm_knobs` fixed (branch); devgen `tuned_tile_selection` ×5 fails **on main too** — tunedb records stale against the kernel-source digest |

## 1. Default flips (the merge risk)

| Knob | main → branch | Platform | Commit | Evidence |
|---|---|---|---|---|
| `PLOW_TOKEN_BATCH` (config.rs:88) | false → **true** | all | 33a5b7bf | none in-commit; docs: "functionally validated" |
| `PLOW_PREFIX_CACHE` (config.rs:92) | nv-only false → **true shared**; AMD selects it (serve/engine.rs:922) | all | 3ca64e93 | none in-commit; every AMD screen ran with it on, never A/B'd |
| `PLOW_VMM_CACHE_MIB` (config.rs:99) | 0 → **4096** | all | 3ca64e93 | none; MI300X B20 sits at ~183/192 GiB |
| `PLOW_VMM_PREFIX` (config.rs:461) | false → auto (sm_90 TP1 BF16) | CUDA | c584339e | admission −4.8..−16.2%; packed 16k/c4 TTFT **+52%** (perf-data/gemma31-h100-fp8-packed-preflight.csv) |
| packed_prefill_default (devgen/lib.rs:6964) | sm_90a → + gfx942 non-FP8 | HSA | 4f830ec6 | none |
| `PLOW_DECODE_TIERS` (build_gfx942.sh:1528) | opt-in → unset = auto 1/2/4/8 | HSA build | 676f98b4 | +24.9% conc-1, +0.07% C20 |

Everything else new on the branch is opt-in and default-off (emit_config.rs:579-611, config.rs:985).

## 2. Kernel findings

### 2.1 H100 default path — no regression found

| Change | Guard | Default | Risk |
|---|---|---|---|
| Decode objects lose T17 warp-per-row norm (op_norm.cuh:71-75; interp_sm120.cu:181 defines `PLOW_NV_PREFILL 0` before the include) | compile-time | on | unreachable at B≤16; B32 native-TC only → measure |
| Attention TMA stage state → bitfields (op_attention_sm90.cuh:612,716,750; 0f626502) | none | on where mapkv | removes local-mem arrays; unmeasured, one A/B |
| GEMM `d_gemm_sm90_impl<BIAS, Weight>` template (op_gemm_sm90.cuh:280-380) | `if constexpr` | bf16 = old sequence | none |
| W8A16/FP8 decode routes, packed WGMMA attention, hd512 role objects | `PLOW_NV_*` build knobs | 0 | none |

### 2.2 H100 gaps (plow native vs reference), ranked

1. Decode attention hd512: 5.5–7.9× vs vLLM attn lib (perf-data/attention-library-h100-20260905.csv). CUDA-core softmax body (op_attention.cuh:588), no TMA, 208 regs. No fix on branch.
2. Prefill attention hd256: 4–5× — WGMMA body with 2 stages, both warpgroups recompute S, cp.async Q, no producer/consumer (op_attention_sm90.cuh:43-66).
3. Prefill attention hd512: 2.1× — default `d_flash_prefill<512,32,16>` mma.sync; WGMMA arm behind `PLOW_BUILD_PFATTN_WGMMA` (unmeasured).
4. Decode GEMV bf16 B8/B16: 1.6–4.5× vs cuBLASLt — FFMA ladder, `GV_MM_MAX=8` (op_gemm.cuh:48-76). Branch fix: native transposed TC object (35.4 vs 32.7 µs) and cuBLASLt route (c4 TPOT 54.1→34.8 ms, c1 −6%), both opt-in.
5. Prefill GEMM: 175/219 TF/s uniform vs ~450 warp-specialized; ws384 objects served for Gemma-4 already.
6. Megakernel slot cap: TPOT flat 76.7 ms c8→c64 vs vLLM 24.9→42.4.

### 2.3 MI300X — where the 5.3× gap to H200 sits

Best branch run 51.9 tok/s (all natives on) vs default 28–32 vs H200 273.7. 70k prefill ablation:
MoE k-loop 21.6%, dense GEMM k-loop 18.1%, collective sync 1.3%, **~59% dispatch structure**
(norms, gather/scatter, epilogues, packet boundaries) — docs/amd/tp-bringup-mi300x.md:1580-1700.

| # | Component | Gap | Root cause | Branch fix | Default? |
|---|---|---|---|---|---|
| 1 | MoE packing | 4.83 vs AITER 1.97 ms @8192 | W8A16 dequant MFMA, BM64, single-WG align (op_moe.h:5758,2260) | `glm_moe_resident` +13.4% | no — A8 3.8% rel-L2 |
| 2 | Decode GEMV | 6–11× vs Lt small-M | 24 VALU/16 B (no v_dot2c bf16), MM16+WALK 2 passes (op_gemm.h:2780,4770) | `glm_gemm_lt_decode` +5.7% | no |
| 3 | DSA select | 325→185 µs; +10.6% | one radix select per row (interp.hip:4349) | `glm_select_local` | **exact, ready** |
| 4 | Sparse MLA prefill | −30.6% vs AITER tail | union walk ∝ union, occ-1 VGPR K/V, LDS conflicts (op_attention.h:4122) | `PLOW_MLA_PF_AITER` −6.6% prefill | no |
| 5 | Indexer replicated on ranks | 19.0→4.05 ms; +8.2% | all T rows on every rank | `glm_index_tp` | **exact, ready** |
| 6 | Batched norms | +2.7%, P99 TPOT −11% | 1 WG for 20 rows (op_norm.h:789) | `glm_decode_norm_rows` | **bit-identical, ready** |
| 7 | Dense projections | 1.3–2× vs hipBLASLt | GM_DBUF 1 on CDNA3 (op_gemm.h:209), 192×256 tile | `glm_gemm_lt` +2.1% | no |
| 8 | MoE decode walk | 119 vs ~432 GB/s | one load in flight (op_moe.h:633) | via resident only | — |
| 9 | Flash staging | FA_LDS_DMA +21.7% single-stream | K/V through VGPRs (op_attention.h:219) | build opt-in | unmeasured w/ FP8KV |
| 10 | Interpreter | ~1134 packets/token, 75% ≤32 CUs | per-op barrier + release RMW (interp.hip:5827) | folds off | — |

Measured nulls — do not re-run: tile campaign, FP8 KV as speed lever (+49.6% TPOT), wider ladder,
`PLOW_XR_MLP`, PSWZ, selective-pack MoE decode (−4.3%), `glm_fold_lt`, `glm_place_pf`.

## 3. Correctness (no blockers)

| Sev | Finding | Where |
|---|---|---|
| should-fix | CUDA batched DSA `IndexSelect` has no row offset: gate relaxed to `index_kpool>1` (mla.rs:3866), `PLOW_DSA_DECODE_BATCH=1` pushed only in `backend_amd` (manifest.rs:1155,1289), sm120 dispatch ignores `i[3]` (interp_sm120.cu:2112) → rows read row 0 scores | f5f15dd2 |
| should-fix | `VmmKv::publish_at` holds `inner` across alloc + D2D + `stream_synchronize` (vmm.rs:1285; fill = gpu/prefix.rs:489) | 704a0518 |
| should-fix | Unified path never sets `decode_progress` (mux.rs:1612-1628; rung controller reads :951) | baebe32d |
| should-fix | `PLOW_DECODE_TIERS` unset⇒auto collides with kimi-k3 recipe's explicit `[[objects.lowrung]]` (same paths, different defines, last writer wins) | 676f98b4 |
| note | `expect()` in tick loop (mux.rs:3387,3400); retire_slot publishes disconnected clients' KV (mux.rs:1984); vmm.rs unwraps :1023,:1502; `evict_one` O(S²); dev_blob.h lacks `ROPE_SCALE_YARN_DS 3u` | |

Checked OK: TP all-rank agreement, FP8 scale snapshot/restore, ring/attach bounds, FairSplit
reclaim, no new per-step allocations, no new lock across device calls in serve/.

## 4. Reuse across CUDA / HSA / CPU

Clean already: `sched/*` (zero `cfg`), `plow_asset::mixed_step`, `mixed_step_staging.rs`,
`memory/vmm.rs` (`VmmOps`, dyn only at bringup), `runtime/common/{mixed_step,dev_isa}.h`.
All runtime knob reads go through the static `RuntimeConfig`; no `env::var` on any step path.

| Duplication | Files | Difference | Zero-cost fix |
|---|---|---|---|
| Serve tick body ×2 | mux.rs:1350-2005 (CUDA) vs 2006-3108 (`SeqEngine`) | CUDA first-token yield vs AMD tick cap; both use shared `sched` | `impl SeqEngine for GpuEngine`; CUDA extras as trait methods |
| Token-batch planner ×2 | gpu/token_batch.rs + plow-asset/token_batch.rs vs amd/mixed_step.rs + plow-asset/mixed_step.rs | generation- vs frontier-checked requests; AMD validates plan, CUDA doesn't | one request/staging type in plow_asset; keep device descriptors |
| Slab carve | gpu.rs:3136 `bytes` vs amd.rs:7994 `bytes.max(1)` | zero-byte tensor aliases next base on CUDA; tests assert opposite | one carve fn in memory/mod.rs |
| Prefix cache ×2 | vmm.rs + gpu/prefix.rs vs engine.rs:786 + amd.rs:12145 + amd/prefix.rs | 32-row block hash cross-request vs slot-local LCP `MIN_PREFIX=128`; CUDA trusts config.json, AMD derives from packet; AMD VMM GQA-only and exclusive with snapshots | policy in memory/prefix.rs; packet-derived regions on both; then `shared_prefix.rs` (planned, absent) |
| Row-resolver header ×2 | runtime/common/token_batch.h vs runtime/amd/token_batch.h | names only | delete AMD copy |
| Rung covering ×3, ELF walker ×2, TP forwarding | sched/rungs.rs:108, gpu/decode_rung.rs:24, amd.rs:12590; cubin.rs vs amd/object.rs:1188; engine.rs `Ranks` | trivial | shared fns |
| Mixed-step source ×2 | exec/mixed_packet.rs (packet) vs exec/mixed_program.rs (runtime synth, GQA only) | two truths | emit the section for gfx942 in plowc |

Enabler: `EngineDevice` is implemented only by `HsaBackend`; `CudaBackend` lacks 9 of 32 methods.

## 5. Hygiene

- 13 raw JSON traces, 6.9 MB / 240k lines under runtime/bench/amd, read by nothing (moe_aiter resident-trace 3.6 MB, resident-prefill 1.8 MB).
- 3fdd322b deleted 13 pre-existing CPU summaries main still has (perf-data/SUMMARY.md, CPU-BACKEND-SUMMARY.md, perf-data/cpu-*/**) → dangling refs in docs/flags-reference.md:671,813, docs/runtime/cpu.md, docs/runtime/prefix-cache.md:49, docs/bringup/07-perf-campaign.md, gpu.rs:4534, build_sm90a_cubin.sh:9.
- Feature-gated test files (exec/amd/tests.rs 4620 lines, exec/gpu/*_tests.rs) compile under `cargo check --features cuda,hsa` but no CI job runs them; 6 runtime/tests/*.{hip,cu} unregistered; `/home/lava` paths at exec/amd/tests.rs:3685,4373,4422 early-return silently; 28 new `#[ignore]`.
- 27 new knobs undocumented; `PLOW_PREFIX_CACHE` row stale, `PLOW_TOKEN_BATCH` absent.
- Other agent's uncommitted shared-prefix WIP lives in /tmp/tp-glm53-publish (vmm.rs, amd.rs, engine.rs, untracked exec/amd/shared_prefix.rs) — not on this branch.

## 6. Fix plan

### Phase 0 — done (f6a06e69, b314a719, b1626c5c)
devgen env reads → EmitConfig; kernelcaps classifier test; `/tokenize` test; `mux.rs` + 17 files
fmt; `build.yml` parses again.

### Fix pass 2026-09-11 — done on this branch (see the commit after b1626c5c)
Decision: the three default flips (`token_batch`, `prefix_cache`, `vmm_cache_mib`) **stay on**.
plans/unified-token-batch.md records default enablement as the production intent; the owed
artefact is the isolated A/B (MI300X here, H100 when a host is available), not a revert.

| Finding | Fix |
|---|---|
| CUDA batched DSA `IndexSelect` without row offset | emit refuses `rows>1 && dsa` unless `emit_is_amd()` (mla.rs) |
| `PLOW_DECODE_TIERS` unset ⇒ auto vs kimi-k3 explicit lowrung | recipe sets `PLOW_DECODE_TIERS = ""`; `check_recipe.py` reports the collision (+ self-test); recipe doc re-rendered |
| `publish_at` lock across alloc + D2D + sync | alloc/fill outside `inner`; re-lock, dedupe a racing publish, register (vmm.rs) |
| Unified path leaves `decode_progress = None` | progress recorded from the feeds the token batch consumed (mux.rs) |
| `expect()` in the unified request builder | `filter_map`: a vanished slot is skipped, re-fed next tick |
| `debug_assert` on snapshot layout | release check: `begin_seq` rollback + `Rejected` (gpu/prefix.rs) |
| Slab carve `.max(1)` mismatch CUDA vs AMD | shared `memory::slab_carve`; both loaders and both slab tests use it |
| `dev_blob.h` missing `YARN_DS` | `PLOW_ROPE_SCALE_YARN_DS 3u` |
| `/home/lava` fixtures | `PLOW_TEST_ABI144_DECODE_ELF`, `PLOW_TEST_K3_SNAPSHOTS`, `PLOW_TEST_GLM52_PREFILL_ELF` |
| exec/{amd,gpu} tests never run in CI | self-hosted step `cargo test -p plowrt --features cuda,hsa --lib exec::` (327 pass / 30 ignored here) |
| 4 sm90 `.cu` harnesses unregistered | CMake targets under `PLOW_CUDA`, guarded; all four compile with the flake's nvcc (`-gencode arch=compute_90a,code=sm_90a`) |
| 2 raw traces (5.4 MB) | removed + gitignored; README cites them as local artefacts |
| 12 dangling `perf-data` links | reworded (reports are kept out of source control per d5f320df) |
| 27 undocumented knobs | rows added for the emit knobs, `PLOW_TOKEN_BATCH`, `PLOW_PREFIX_CACHE` (corrected), `PLOW_MLA_PF_AITER`, `PLOW_DECODE_TIERS`, `PLOW_MAX_REQUEST_CHUNK`, `PLOW_EMIT_DECODE_NATIVE_TC` |

Deferred (needs a GPU host or is a Phase 3/4 item): the A/Bs; tunedb gfx942 re-campaign;
`retire_slot` publish-on-disconnect policy (needs a per-tick disconnect record); the two `.hip`
harnesses stay script-driven (see runtime/bench/amd/dsa_decode/README.md); Phase 4 reuse beyond
the slab carve.

### Phase 1 — before merging to main
1. **Default flips: measure or hold.** Preferred: A/B on both vendors with one variable each.
   - MI300X (this host, GLM-5.3 TP8, frozen set /tmp/tp-glm53-prefix-keys + /tmp/tp-glm53-pswz/serving-safe): `PLOW_PREFIX_CACHE=0/1`, `PLOW_VMM_CACHE_MIB=0/4096`, C1 and C20, 70k/700 and 5k/700; report tok/s, TTFT, TPOT p50/p99.
   - H100 (needs a host): Gemma-4 BF16, `--token-batch=false --prefix-cache=false` vs defaults, c1/c8, 1k and 16k, cache-miss traffic.
   - If either A/B cannot run before the merge: set `token_batch`, `prefix_cache` = false and `vmm_cache_mib` = 0 in config.rs for the merge and re-flip per vendor with the numbers. The auto-selection machinery stays.
2. **CUDA batched DSA**: at mla.rs:3866 require `crate::emit_is_amd()` for `rows>1 && dsa`, or push an nvcc requirement that the sm120 object cannot satisfy until `d_index_select_sm120` takes `i[3]`. Add the row-offset test to devgen for the nvcc backend.
3. **`PLOW_DECODE_TIERS`**: add `PLOW_DECODE_TIERS = ""` to recipes/infervisor/kimi-k3/*.toml `[objects.env]` (or restore opt-in in build_gfx942.sh); extend scripts/check_recipe.py to refuse a recipe that both leaves TIERS unset and declares `[[objects.lowrung]]`.
4. **Restore** the 13 CPU summaries from origin/main (`git checkout origin/main -- perf-data/SUMMARY.md perf-data/CPU-BACKEND-SUMMARY.md perf-data/cpu-*`), or fix the 12 dangling references.
5. **Docs**: flags-reference rows for the 27 knobs; correct `PLOW_PREFIX_CACHE` / add `PLOW_TOKEN_BATCH`.
6. **Bench JSON**: drop the 13 raw traces (keep `*-selected.json`, `compare.py` outputs and READMEs); fix record.py:20 hard-coded /tmp path. If a trace must stay, gzip it or move to perf-data storage referenced by SHA.

### Phase 2 — right after merge (low risk)
- vmm.rs: allocate + copy outside `inner`; take the lock only to publish the snapshot handle.
- mux.rs: set `decode_progress` on the unified path; `expect` → `continue` with a warn; skip `vmm_publish` for `retire_slot` on client disconnect.
- dev_blob.h: add `ROPE_SCALE_YARN_DS 3u`; gpu/prefix.rs:520 `debug_assert` → refuse-at-attach.
- Tests: register runtime/tests/*.{hip,cu} in CMake behind `PLOW_ROCM`/`PLOW_CUDA`; add `cargo test -p plowrt --features cuda,hsa --lib exec::` to the self-hosted job (these are host-side tests); replace `/home/lava` paths with `PLOW_TEST_*` env or delete.
- tunedb: re-run scripts/rebench_tune_gemm_gfx942.sh against the merged tree + `plowc tune ingest` so `tuned_tile_selection` is green for gfx942 (~1% gain; gfx950 needs an MI350X).

### Phase 3 — performance, MI300X (gain / risk order)
1. Default-on after one repeat C20 screen each: `glm_select_local` (+10.6%), `glm_index_tp` (+8.2%, keep ≥2048 gate), `glm_decode_norm_rows` (+2.7%, P99 −11%). Exact numerics already tested.
2. `glm_gemm_lt` + `glm_gemm_lt_decode` (+2.1% / +5.7%, ≤0.17% rel-L2, 5 Tensile objects pinned by SHA) — default after a second run confirms.
3. `glm_moe_resident` (+13.4%, largest single win) and `PLOW_MLA_PF_AITER` (prefill −6.6%): decide the numerics contract (A8 3.5–3.9% rel-L2 vs DET FP64) with a broader eval than 18 needles (GSM8K + long-retrieval at C20) before any default.
4. Follow-ups with measured primitive gains: `FA_LDS_DMA` under sparse+FP8KV; hipBLASLt for the 7 remaining prefill shapes; a real small-M kernel for the 199 residual decode GEMVs; `XR_RES`/`FUSE_XRN` on GLM-5.3 TP8; packet coalescing in the interpreter (the 59%).

### Phase 3' — performance, H100
1. A/B the default flips (Phase 1.1) and the hd256 FA cubin main-vs-branch (0f626502).
2. Qualify the decode GEMV TC route (`--emit-decode-native-tc`) and adaptive cuBLASLt selection (c4 win, c1 loss).
3. Measure `PLOW_BUILD_PFATTN_WGMMA=1` for hd512 prefill; then the two unaddressed gaps: decode attention hd512 (TC + TMA body) and hd256 prefill producer/consumer pipeline.

### Phase 4 — reuse (zero-cost, dependency order)
1. `impl EngineDevice for CudaBackend` (~100 lines) — unblocks generic monomorphized engine code.
2. Shared slab carve in memory/mod.rs; settle `.max(1)` one way; make both slab tests assert it.
3. `covering()` in sched/rungs.rs; ELF symtab iterator in plow-asset; delete runtime/amd/token_batch.h.
4. One token-batch request + staging type in plow_asset (host planner only; device descriptors stay per object).
5. `impl SeqEngine for GpuEngine` → one mux tick body. Gate: greedy-token equivalence on both vendors and step time within noise.
6. Prefix policy (`prefix_plan`, boundary rounding, snapshot regions from the packet) in memory/prefix.rs; then the planned `exec/amd/shared_prefix.rs` with `VmmKv::new_tensors` for GLM ckv/krot/kidx.
7. Emit the mixed-step section for gfx942 in plowc; retire exec/mixed_program.rs.

Every Phase 4 step is judged by CLAUDE.md's rule: no new allocation, copy, lock or indirection on
the step path; measure when uncertain.
