# Emit-knob audit (`crates/devgen/src/emit_config.rs`), 2026-09-04

Every `EmitConfig` field on `codex/amd-agent-harness` (1ad90aa) classified against
`perf-data/kimi-k3-mi355x-campaign-summary-20260904.md`, the plan's closed list and
`docs/flags-reference.md`. Readers were located with `grep -rn '\.<field>' crates`; env
mentions with `grep -rn <ENV> scripts runtime crates docs`.

| | before | after |
|---|---:|---:|
| `EmitConfig` fields | 123 | 114 |
| hidden (`hide = true`) | 4 | 9 |
| removed | | 7 |
| derived / merged | | 5 fields -> 3 (`mla_ns`, `layers`, `gemm_wide_c8`) |
| newly hidden | | 4 |
| kept | | 103 |

Evidence that the defaults did not move (same `plowc` source apart from this change, same
TuneDB, `PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix`):

| packet | before | after |
|---|---|---|
| Kimi-K3 TP8 gfx950 `--max-ctx 16384 --n-cu 256` `model.pkt` | `31808e16...` | `31808e16...` (byte-identical, `build.json` identical) |
| gemma-4-12B-it gfx950 `--max-ctx 4096` `model.pkt` | `d0ed748f...` | `d0ed748f...` |

Checks: `cargo build --release -p plowc`, `cargo test -p devgen -p packet -p plowrt --features hsa`
(green apart from `tuned_tile_selection::gfx942_*_measurements_reach_the_compiler`, which fail
identically on the base commit: the gfx942 TuneDB cell is stale against the probed build
digest), `cargo fmt --all --check`.

GLM-5.3-Flash and Llama-4-Scout checkpoints on this host are not emittable by this `plowc`
(`unsupported architecture: glm5_next_text` / `llama4_text`); the GLM emitter is covered by
`devgen`'s synthetic `glm_tests` instead.

## Rules applied

* **Generic mechanism** -> kept; doc comment says what it does and, for a promoted default,
  that `=0` is the rollback.
* **Model-specific value** -> derived from the emit (ladder cap, TuneDB records) or merged into
  one family-independent override; the old env names are gone (`PLOW_K3_NS`, `PLOW_GLM_NS`,
  `PLOW_K3_LAYERS`, `PLOW_GLM_LAYERS`, `PLOW_GEMM_WIDE_C8_SHAPE`) and the scripts/docs that set
  them were updated.
* **Rejected / closed / superseded** -> knob and its emit-side code path removed when nothing
  else reaches the path. Kernel arms, CMake options, loader refusal tables and runtime tests that
  exercise the same arms directly were deliberately left alone (they are runtime, not emit,
  surface).
* **Diagnostic** -> `hide = true`; env and flag still work.

## Table

| field | env | default | action | class | now | reason |
|---|---|---|---|---|---|---|
| `fp8` | `PLOW_FP8` | false | kept | generic precision | `PLOW_FP8` | weight/activation/KV encoding; applies to every family |
| `w8a8` | `PLOW_W8A8` | false | kept | generic precision | `PLOW_W8A8` | weight/activation/KV encoding; applies to every family |
| `w8a16` | `PLOW_W8A16` | false | kept | generic precision | `PLOW_W8A16` | weight/activation/KV encoding; applies to every family |
| `mxfp4` | `PLOW_MXFP4` | false | kept | generic precision | `PLOW_MXFP4` | weight/activation/KV encoding; applies to every family |
| `fp8_kv` | `PLOW_FP8_KV` | false | kept | generic precision | `PLOW_FP8_KV` | weight/activation/KV encoding; applies to every family |
| `fp8_kv_full` | `PLOW_FP8_KV_FULL` | false | kept | generic precision | `PLOW_FP8_KV_FULL` | weight/activation/KV encoding; applies to every family |
| `fp8_head` | `PLOW_FP8_HEAD` | false | kept | generic precision | `PLOW_FP8_HEAD` | weight/activation/KV encoding; applies to every family |
| `uniseg` | `PLOW_UNISEG` | false | kept | generic scheduling | `PLOW_UNISEG` | sm_120 prefill interpreter requirement (AMD refuses it) |
| `decode_mla_segments` | `PLOW_SEG_DECODE_MLA` | true | kept | rollback of promoted default | `PLOW_SEG_DECODE_MLA` | MLA specialist segments promoted 09-03 (TPOT -8.1 ms) |
| `decode_grouped_moe_segments` | `PLOW_SEG_DECODE_GROUPED_MOE` | unset | kept | override of a promoted rule | `PLOW_SEG_DECODE_GROUPED_MOE` | MoE decode route rule promoted 09-04; unset = TuneDB measurement decides |
| `decode_batch` | `PLOW_DECODE_BATCH` | 1 | kept | generic scheduling | `PLOW_DECODE_BATCH` | program geometry; model-independent |
| `decode_ladder` | `PLOW_DECODE_BATCH_LADDER` | unset | kept | generic scheduling | `PLOW_DECODE_BATCH_LADDER` | program geometry; model-independent |
| `max_chunk` | `PLOW_MAX_CHUNK` | unset | kept | generic scheduling | `PLOW_MAX_CHUNK` | program geometry; model-independent |
| `gemv_split` | `PLOW_GEMV_SPLIT` | 1 | kept | generic scheduling | `PLOW_GEMV_SPLIT` | program geometry; model-independent |
| `decode_tiled` | `PLOW_DECODE_TILED` | false | kept | generic scheduling | `PLOW_DECODE_TILED` | program geometry; model-independent |
| `fuse_argmax` | `PLOW_FUSE_ARGMAX` | false | kept | generic fusion A/B | `PLOW_FUSE_ARGMAX` | NVIDIA-era cross-model fusion controls, unset = byte-identical |
| `no_fuse_qkv` | `PLOW_NO_FUSE_QKV` | false | kept | generic fusion A/B | `PLOW_NO_FUSE_QKV` | NVIDIA-era cross-model fusion controls, unset = byte-identical |
| `fuse_qkv_fp8` | `PLOW_FUSE_QKV_FP8` | false | kept | generic fusion A/B | `PLOW_FUSE_QKV_FP8` | NVIDIA-era cross-model fusion controls, unset = byte-identical |
| `no_fuse_nrn` | `PLOW_NO_FUSE_NRN` | false | kept | generic fusion A/B | `PLOW_NO_FUSE_NRN` | NVIDIA-era cross-model fusion controls, unset = byte-identical |
| `fuse_hnr` | `PLOW_FUSE_HNR` | false | kept | generic fusion A/B | `PLOW_FUSE_HNR` | NVIDIA-era cross-model fusion controls, unset = byte-identical |
| `fuse_merge` | `PLOW_FUSE_MERGE` | false | kept | generic fusion A/B | `PLOW_FUSE_MERGE` | NVIDIA-era cross-model fusion controls, unset = byte-identical |
| `hn_split` | `PLOW_HN_SPLIT` | false | kept | generic fusion A/B | `PLOW_HN_SPLIT` | NVIDIA-era cross-model fusion controls, unset = byte-identical |
| `fa_gf_full` | `PLOW_FA_GF_FULL` | unset | kept | generic attention/ladder geometry | `PLOW_FA_GF_FULL` | flash-decode split and prefill-ladder controls shared by every family |
| `flash_merge_dsplit` | `PLOW_FLASH_MERGE_DSPLIT` | unset | hidden | diagnostic | `PLOW_FLASH_MERGE_DSPLIT` (hidden) | measured no effect; already hidden |
| `ns_mul` | `PLOW_NS_MUL` | unset | kept | generic attention/ladder geometry | `PLOW_NS_MUL` | flash-decode split and prefill-ladder controls shared by every family |
| `ns_abs` | `PLOW_NS_ABS` | unset | kept | generic attention/ladder geometry | `PLOW_NS_ABS` | flash-decode split and prefill-ladder controls shared by every family |
| `ns_full_abs` | `PLOW_NS_FULL_ABS` | unset | kept | generic attention/ladder geometry | `PLOW_NS_FULL_ABS` | flash-decode split and prefill-ladder controls shared by every family |
| `pf_ladder` | `PLOW_PF_LADDER` | unset | kept | generic attention/ladder geometry | `PLOW_PF_LADDER` | flash-decode split and prefill-ladder controls shared by every family |
| `pf_ladder_append` | `PLOW_PF_LADDER_APPEND` | unset | kept | generic attention/ladder geometry | `PLOW_PF_LADDER_APPEND` | flash-decode split and prefill-ladder controls shared by every family |
| `pf_gemv_head` | `PLOW_PF_GEMV_HEAD` | unset | kept | generic attention/ladder geometry | `PLOW_PF_GEMV_HEAD` | flash-decode split and prefill-ladder controls shared by every family |
| `xr_cus` | `PLOW_XR_CUS` | unset | kept | generic collective geometry | `PLOW_XR_CUS` | XReduce participant cap, honoured by every TP emitter |
| `xr2_gather` | `PLOW_XR2_GATHER` | true | kept | rollback of promoted default | `PLOW_XR2_GATHER` | folded-gather two-shot promoted 09-03 (TTFT -240 ms) |
| `no_xreduce` | `PLOW_NO_XREDUCE` | false | hidden | diagnostic | `PLOW_NO_XREDUCE` (hidden) | numerically wrong by design; already hidden |
| `moe_prefill` | `PLOW_MOE_PREFILL` | unset | kept | Gemma-MoE mechanism A/B | `PLOW_MOE_PREFILL` | router/tail mechanism switches for the Gemma MoE family; not values derivable from the config |
| `gemma_moe_router_fused` | `PLOW_GEMMA_MOE_ROUTER_FUSED` | false | kept | Gemma-MoE mechanism A/B | `PLOW_GEMMA_MOE_ROUTER_FUSED` | router/tail mechanism switches for the Gemma MoE family; not values derivable from the config |
| `gemma_moe_router_blocks` | `PLOW_GEMMA_MOE_ROUTER_BLOCKS` | unset | kept | Gemma-MoE mechanism A/B | `PLOW_GEMMA_MOE_ROUTER_BLOCKS` | router/tail mechanism switches for the Gemma MoE family; not values derivable from the config |
| `gemma_moe_router_exact` | `PLOW_GEMMA_MOE_ROUTER_EXACT` | false | kept | Gemma-MoE mechanism A/B | `PLOW_GEMMA_MOE_ROUTER_EXACT` | router/tail mechanism switches for the Gemma MoE family; not values derivable from the config |
| `gemma_moe_tail_fuse` | `PLOW_GEMMA_MOE_TAIL_FUSE` | false | kept | Gemma-MoE mechanism A/B | `PLOW_GEMMA_MOE_TAIL_FUSE` | router/tail mechanism switches for the Gemma MoE family; not values derivable from the config |
| `k3_full` | `K3_FULL` | true | hidden | diagnostic | `K3_FULL` (hidden) | K3_FULL=0 prints the legacy capability report instead of emitting |
| `k3_fuse_a` | `PLOW_K3_FUSE_A` | false | kept | opt-in candidate | `PLOW_K3_FUSE_A` | decode-only A-projection fusion; never network-gated on MI355X |
| `k3_ns` | `PLOW_K3_NS` | unset | derived | merged into `mla_ns` (`PLOW_MLA_NS`) | - | same MLA flash-decode nsplit pin as PLOW_GLM_NS; one generic override |
| `k3_layers` | `PLOW_K3_LAYERS` | all | derived | merged into `layers` (`PLOW_LAYERS`, hidden) | - | layer truncation is family-independent; one knob, one parser |
| `k3_prefill` | `K3_PREFILL` | unset | kept | K3 prefill ladder control | `K3_PREFILL` | decode-only / rung-list emits used by scripts/k3_block_sweep.sh and k3_tp_equivalence.sh; unification with PLOW_MLA_PREFILL/PLOW_MOE_PREFILL deferred (different grammars) |
| `glm_dsa` | `PLOW_GLM_DSA` | unset | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_DSA` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_gf` | `PLOW_GLM_GF` | unset | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_GF` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_ns` | `PLOW_GLM_NS` | unset | derived | merged into `mla_ns` (`PLOW_MLA_NS`) | - | same MLA flash-decode nsplit pin as PLOW_K3_NS; one generic override |
| `glm_shard_head` | `GLM_SHARD_HEAD` | false | kept | GLM / gfx942 campaign mechanism | `GLM_SHARD_HEAD` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_moe_coresident` | `GLM_MOE_CORESIDENT` | unset | kept | GLM / gfx942 campaign mechanism | `GLM_MOE_CORESIDENT` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_shared_cus` | `GLM_SHARED_CUS` | unset | kept | GLM / gfx942 campaign mechanism | `GLM_SHARED_CUS` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_spine_cus` | `GLM_SPINE_CUS` | unset | kept | GLM / gfx942 campaign mechanism | `GLM_SPINE_CUS` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_linear_fp8` | `GLM_LINEAR_FP8` | false | kept | GLM / gfx942 campaign mechanism | `GLM_LINEAR_FP8` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_shared_glu_split` | `GLM_SHARED_GLU_SPLIT` | false | kept | GLM / gfx942 campaign mechanism | `GLM_SHARED_GLU_SPLIT` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_layers` | `PLOW_GLM_LAYERS` | all | derived | merged into `layers` (`PLOW_LAYERS`, hidden) | - | layer truncation is family-independent; legacy GLM_FULL/GLM_NLAYERS/GLM_LAYER synthesis kept in from_env |
| `mla_prefill` | `PLOW_MLA_PREFILL` | unset | kept | GLM / gfx942 campaign mechanism | `PLOW_MLA_PREFILL` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_ep` | `GLM_EP` | false | kept | GLM / gfx942 campaign mechanism | `GLM_EP` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_group` | `GLM_GROUP` | false | kept | GLM / gfx942 campaign mechanism | `GLM_GROUP` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_fuse_b1` | `PLOW_GLM_FUSE_B1` | false | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_FUSE_B1` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_fuse_seam` | `PLOW_GLM_FUSE_SEAM` | false | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_FUSE_SEAM` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_fuse_rope` | `PLOW_GLM_FUSE_ROPE` | false | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_FUSE_ROPE` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_fuse_qnorm` | `PLOW_GLM_FUSE_QNORM` | false | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_FUSE_QNORM` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_router_off_shared` | `GLM_ROUTER_OFF_SHARED` | false | kept | GLM / gfx942 campaign mechanism | `GLM_ROUTER_OFF_SHARED` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_router_old` | `GLM_ROUTER_OLD` | false | kept | GLM / gfx942 campaign mechanism | `GLM_ROUTER_OLD` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `k3_fuse_ngemv` | `PLOW_K3_FUSE_NGEMV` | unset | kept | rollback of promoted default | `PLOW_K3_FUSE_NGEMV` | norm->GEMV fold default on, `0`/`lat`/`q` bisect |
| `k3_kda_conv_step_db` | `PLOW_K3_KDA_CONV_STEP_DB` | false | kept | opt-in candidate | `PLOW_K3_KDA_CONV_STEP_DB` | B1 experiment paired with a `PLOW_K3_KDA_CONV_STEP_DB=1` object (CMake option) |
| `kda_decode_fused` | `PLOW_KDA_DECODE_FUSED` | false | kept | opt-in candidate | `PLOW_KDA_DECODE_FUSED` | benchmark-only fused KDA decode block (+0.49 % gate); geometry-gated |
| `mla_materialized_prefill` | `PLOW_MLA_MATERIALIZED_PREFILL` | false | kept | opt-in candidate under gate | `PLOW_MLA_MATERIALIZED_PREFILL` | rejected default-off (continuation diverged) but the next TTFT lever; `mla_materialized_gate.sh` drives it |
| `kda_chunk` | `PLOW_KDA_CHUNK` | unset | kept | rollback of promoted default | `PLOW_KDA_CHUNK` | chunk-KDA prefill scan / qpre promoted 09-03 |
| `kda_chunk_qpre` | `PLOW_KDA_CHUNK_QPRE` | true | kept | rollback of promoted default | `PLOW_KDA_CHUNK_QPRE` | chunk-KDA prefill scan / qpre promoted 09-03 |
| `kda_intra_cached` | `PLOW_KDA_INTRA_CACHED` | false | removed | rejected / superseded | - | cached intra object superseded by wave-items (promoted 09-04); the packet marking it set is the same bit wave-items sets, so the knob added nothing. Object build switch `PLOW_KDA_INTRA_CACHED` in scripts/build_gfx950.sh and the runtime route are untouched |
| `kda_intra_wave_items` | `PLOW_KDA_INTRA_WAVE_ITEMS` | true | kept | rollback of promoted default | `PLOW_KDA_INTRA_WAVE_ITEMS` | promoted 09-04 (TTFT -84 / -112 ms) |
| `kda_carry_regstate` | `PLOW_KDA_CARRY_REGSTATE` | true | kept | rollback of promoted default | `PLOW_KDA_CARRY_REGSTATE` | promoted 09-04 (TTFT -84 / -112 ms) |
| `kda_key_factor` | `PLOW_KDA_KEY_FACTOR` | true | kept | rollback of promoted default | `PLOW_KDA_KEY_FACTOR` | emit marking default on; the object pair is OFF at build (`PLOW_HSACO_KDA_KEY_FACTOR`) after the 09-04 screen, doc now says so |
| `kda_wu_lean` | `PLOW_KDA_WU_LEAN` | false | kept | opt-in candidate under gate | `PLOW_KDA_WU_LEAN` | lean Wu + key-fed carry, TP8 gate pending (prefill-kda-chain-status.md) |
| `kda_carry_keyfeed` | `PLOW_KDA_CARRY_KEYFEED` | false | kept | opt-in candidate under gate | `PLOW_KDA_CARRY_KEYFEED` | lean Wu + key-fed carry, TP8 gate pending (prefill-kda-chain-status.md) |
| `k3_up_nogather` | `PLOW_K3_UP_NOGATHER` | false | removed | diagnostic, purpose served | - | bisection instruments that found the bf16 round in d_xreduce's gather arm; the sharded up-projection has been correct since. Code paths (no-gather / replicated weight) removed with the knobs |
| `k3_up_gather_only` | `PLOW_K3_UP_GATHER_ONLY` | false | removed | diagnostic, purpose served | - | bisection instruments that found the bf16 round in d_xreduce's gather arm; the sharded up-projection has been correct since. Code paths (no-gather / replicated weight) removed with the knobs |
| `k3_shard_head` | `PLOW_K3_SHARD_HEAD` | false | kept | rejected, kept for a script | `PLOW_K3_SHARD_HEAD` | lm_head sharding rejected 09-03 (TTFT +8 ms) and closed in the plan, but scripts/k3_tp_equivalence.sh sweeps it; doc marks it rejected |
| `k3_seq_rows` | `PLOW_K3_SEQ_ROWS` | false | hidden | diagnostic | `PLOW_K3_SEQ_ROWS` (hidden) | B=1 per-sequence carrier bisect |
| `gemv_mm` | `PLOW_GEMV_MM` | unset | kept | generic GEMV geometry | `PLOW_GEMV_MM` | AMD decode row-batch bucket and walk loop |
| `gemv_walk` | `PLOW_GEMV_WALK` | false | kept | generic GEMV geometry | `PLOW_GEMV_WALK` | AMD decode row-batch bucket and walk loop |
| `fuse_residual_input` | `PLOW_FUSE_RESIDUAL_INPUT` | true | kept | rollback of promoted default | `PLOW_FUSE_RESIDUAL_INPUT` | materialized residual fusion (all models) / AttnRes+norm fold |
| `k3_fuse_arnorm` | `PLOW_K3_FUSE_ARNORM` | true | kept | rollback of promoted default | `PLOW_K3_FUSE_ARNORM` | materialized residual fusion (all models) / AttnRes+norm fold |
| `qnorm_fuse` | `PLOW_QNORM_FUSE` | false | kept | generic fusion / GEMV geometry | `PLOW_QNORM_FUSE` | gfx942-campaign knobs on the shared emitter |
| `fuse_quant` | `PLOW_FUSE_QUANT` | true | kept | generic fusion / GEMV geometry | `PLOW_FUSE_QUANT` | gfx942-campaign knobs on the shared emitter |
| `gemv_wg` | `PLOW_GEMV_WG` | unset | kept | generic fusion / GEMV geometry | `PLOW_GEMV_WG` | gfx942-campaign knobs on the shared emitter |
| `gemv_wg_tuning` | `PLOW_GEMV_WG_TUNING` | unset | kept | generic A/B override | `PLOW_GEMV_WG_TUNING` | already shape-keyed and model-independent; there is no TuneDB record kind for GEMV workgroup width to derive from (`scripts/gemv_wg_plan.py` produces the string), the one pending row (`128x7168=64`) is one pair |
| `glm_dsa_pf` | `PLOW_GLM_DSA_PF` | false | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_DSA_PF` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_fp8_kv` | `PLOW_GLM_FP8_KV` | false | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_FP8_KV` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_gemv_wg` | `PLOW_GLM_GEMV_WG` | unset | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_GEMV_WG` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_ofold` | `PLOW_GLM_OFOLD` | false | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_OFOLD` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_pf_ns` | `PLOW_GLM_PF_NS` | unset | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_PF_NS` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_pf_wide` | `PLOW_GLM_PF_WIDE` | true | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_PF_WIDE` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_place_pf` | `PLOW_GLM_PLACE_PF` | false | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_PLACE_PF` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_xr_band` | `PLOW_GLM_XR_BAND` | unset | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_XR_BAND` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_xr_band_cus` | `PLOW_GLM_XR_BAND_CUS` | unset | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_XR_BAND_CUS` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `attnres_decode_mwg` | `PLOW_ATTNRES_DECODE_MWG` | unset | kept | opt-in candidate | `PLOW_ATTNRES_DECODE_MWG` | banded decode AttnRes arm paired with `runtime/bench/amd/attnres_decode_mwg`; never network-gated |
| `glm_xr_band_seam` | `PLOW_GLM_XR_BAND_SEAM` | unset | hidden | diagnostic | `PLOW_GLM_XR_BAND_SEAM` (hidden) | divergence-bisect instrument |
| `glm_xr_res` | `PLOW_GLM_XR_RES` | false | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_XR_RES` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `glm_fuse_xrn` | `GLM_FUSE_XRN` | false | kept | GLM / gfx942 campaign mechanism | `GLM_FUSE_XRN` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `moe_pf_a8` | `PLOW_MOE_PF_A8` | false | removed | rejected | - | gfx942 activation-arm screen: near-null (-0.3..0.5 %), fails the ship gate (top-1 flips). Emit wiring (xn2q/xn2s operands, GLU i[7]) removed; kernel arm and loader refusal untouched |
| `xr_combine_fold` | `PLOW_XR_COMBINE_FOLD` | true | kept | rollback of promoted default | `PLOW_XR_COMBINE_FOLD` | promoted 09-03/09-04, doc comments now say what they do and that `=0` is the rollback |
| `kda_fb_fold` | `PLOW_KDA_FB_FOLD` | false | kept | opt-in candidate | `PLOW_KDA_FB_FOLD` | L3 f_b GEMV fold into KdaStateStepG, landed upstream during this audit (b740ae8); needs a PLOW_KDA_FB_FOLD=1 object, default off |
| `moe_stage2_lean` | `PLOW_MOE_STAGE2_LEAN` | true | kept | rollback of promoted default | `PLOW_MOE_STAGE2_LEAN` | promoted 09-03/09-04, doc comments now say what they do and that `=0` is the rollback |
| `moe_stage1_lean` | `PLOW_MOE_STAGE1_LEAN` | true | kept | rollback of promoted default | `PLOW_MOE_STAGE1_LEAN` | promoted 09-03/09-04, doc comments now say what they do and that `=0` is the rollback |
| `moe_combine_lean` | `PLOW_MOE_COMBINE_LEAN` | true | kept | rollback of promoted default | `PLOW_MOE_COMBINE_LEAN` | promoted 09-03/09-04, doc comments now say what they do and that `=0` is the rollback |
| `attnres_f32mix` | `PLOW_ATTNRES_F32MIX` | true | kept | rollback of promoted default | `PLOW_ATTNRES_F32MIX` | promoted 09-03/09-04, doc comments now say what they do and that `=0` is the rollback |
| `moe_align_par` | `PLOW_MOE_ALIGN_PAR` | true | kept | rollback of promoted default | `PLOW_MOE_ALIGN_PAR` | promoted 09-03/09-04, doc comments now say what they do and that `=0` is the rollback |
| `seq_par_seams` | `PLOW_SEQ_PAR_SEAMS` | true | kept | rollback of promoted default | `PLOW_SEQ_PAR_SEAMS` | promoted 09-03/09-04, doc comments now say what they do and that `=0` is the rollback |
| `moe_prefill_ep` | `PLOW_MOE_PREFILL_EP` | false | kept | opt-in, experiment input | `PLOW_MOE_PREFILL_EP` | EP prefill boundary is design-only (K3 blocked by memory) but the emitted EP asset is the input of `runtime/bench/amd/moe_ep_boundary` and the pending 2D-layout prototype |
| `moe_pf_atomic` | `PLOW_MOE_PF_ATOMIC` | false | removed | rejected / superseded | - | atomic prefill accumulate rejected on K3 (TTFT +29 ms, three checksums) and closed; superseded on GLM by the deterministic twin. `MoePfFuse::Atomic` emit arm removed; kernel arm, CMake option, runtime tests and loader refusal untouched |
| `moe_pf_det` | `PLOW_MOE_PF_DET` | false | kept | opt-in candidate | `PLOW_MOE_PF_DET` | gate-passed on GLM-5.2/gfx942 and set by the gfx942 recipe; default stays off for the reason in its doc comment |
| `moe_pf_part16` | `PLOW_MOE_PF_PART16` | false | removed | rejected | - | bf16 part scatter measured a wash and flipped top-1 (glm52-moepf-activation-arms.md). Emit wiring (part width, i[7]) removed; kernel arm and loader refusal untouched |
| `moe_pf_shuf` | `PLOW_MOE_PF_SHUF` | false | removed | closed | - | preshuffled expert companions measured null ("preshuffled companions" in the closed list). `expert_weight_table_pf` declaration and i[6] removed; loader slab code untouched |
| `no_glu_fuse` | `PLOW_NO_GLU_FUSE` | false | kept | generic pre-campaign knobs | `PLOW_NO_GLU_FUSE` | NVIDIA/Gemma-era controls, unset = byte-identical |
| `tma_gemm` | `PLOW_TMA_GEMM` | false | kept | generic pre-campaign knobs | `PLOW_TMA_GEMM` | NVIDIA/Gemma-era controls, unset = byte-identical |
| `pf_gfuse` | `PLOW_PF_GFUSE` | false | kept | generic pre-campaign knobs | `PLOW_PF_GFUSE` | NVIDIA/Gemma-era controls, unset = byte-identical |
| `uniseg_max_t` | `PLOW_UNISEG_MAX_T` | unset | kept | generic pre-campaign knobs | `PLOW_UNISEG_MAX_T` | NVIDIA/Gemma-era controls, unset = byte-identical |
| `glm_wgfit` | `PLOW_GLM_WGFIT` | true | kept | GLM / gfx942 campaign mechanism | `PLOW_GLM_WGFIT` | mechanism switch on the shared MLA+MoE emitter; not covered by the MI355X campaign and not measurable here (no GPU), env names kept for the gfx942 scripts |
| `tunedb` | `PLOW_TUNEDB` | unset | kept | generic tuning config | `PLOW_TUNEDB` | TuneDB root |
| `tune_dump` | `PLOW_TUNE_DUMP` | false | hidden | diagnostic | `PLOW_TUNE_DUMP` (hidden) | TUNEDUMP census printer |
| `gemm_wide_c8_shape` | `PLOW_GEMM_WIDE_C8_SHAPE` | 8192x1536x7168 | derived | derived; replaced by `gemm_wide_c8` rollback (`PLOW_GEMM_WIDE_C8`) | - | the shape was K3's q-projection at the widest chunk. Now derived: ladder-cap chunk rows x exact-shape TuneDB c8 winner x full grid. A pure TuneDB rule also tags `2048x6144x1536` (isolated -14.5 %, never network-gated) and changes the packet, so the ladder-cap row restriction is the gate boundary |
| `skip_coverage` | `PLOW_SKIP_COVERAGE` | false | hidden | diagnostic | `PLOW_SKIP_COVERAGE` (hidden) | already hidden |
| `k3_ablate` | `PLOW_K3_ABLATE` | unset | hidden | diagnostic | `PLOW_K3_ABLATE` (hidden) | already hidden |

## Deliberately left alone

* `PLOW_FUSE_XR_ATTNRES`, `PLOW_XR_WAVE_RS`, `PLOW_PHASE_OBJECTS`, `PLOW_MOE_DECODE_STANDALONE`:
  raw `std::env::var` reads in `packet::devbuild`, not `EmitConfig` fields. The first two are
  rejected (+91.7 ms / +3.6 ms) but `k3::tests::full_graph_xreduce_attnres_fusion_*` and the
  builder's `xreduce_wave_rs` tests still exercise the segmentation; phase objects are a runtime
  flag (`--amd-phase-objects`) with the AQL replay design pending. Out of this audit's scope.
* `PLOW_GV_DYNCLAIM`, `PLOW_XR_SLICES`: no reader anywhere in `crates/`, `runtime/` or
  `scripts/` on this branch — already gone.
* `scripts/build_gfx942.sh` / `runtime/CMakeLists.txt` `PLOW_MOE_PF_ATOMIC` build axis and
  `runtime/tests/moe_prefill_a4w4_*` which compile the atomic kernel arm: kernel side, still
  self-consistent, no emitter produces the packet any more.
* `scripts/build_gfx950.sh` `PLOW_KDA_INTRA_CACHED=1` (object build) and the plowrt
  `kda_intra_cached` route: a wave-items-marked packet still routes to the cached object when
  it is the one loaded.
* `crates/packet/src/names.rs` `expert_weight_table_pf` and the plowrt preshuffled-slab loader:
  packet-driven, inert without a declaring blob.
* `K3_PREFILL` / `PLOW_MLA_PREFILL` / `PLOW_MOE_PREFILL`: three prefill-ladder grammars on three
  families; merging them changes script contracts for no packet benefit.
* `k3_shard_head` and `glm_shard_head` are the same mechanism under two names; K3's is rejected
  and a script depends on it, GLM's measured a win on gfx942. Left as two knobs.
