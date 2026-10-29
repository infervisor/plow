# 25 — DSpark block drafter (pending support)

**Status: paused (2026-10-22).** MTP (the Gemma-4 assistant drafter) is the shipped speculative
decoder. DSpark phases D1 (target-side context) and D2 (draft program) were implemented and gated on
branch `g12-dspark` (`e3a556c5`) and merged on 2026-10-26 (`172dd6f7b`), off unless emitted with
`PLOW_SPEC_KIND=gemma4_dspark`; D3 (serving) is not built: `spec_serve` refuses block drafters. This page
records what exists, what was measured and the plan to finish it.

## What DSpark is

DeepSeek's `dspark_gemma4_12b_block7` drafts a block of 7 tokens in one non-causal forward of a
5-layer drafter. That forward reads the target's hidden states at layers 5/17/29/41/46 through the
drafter's own KV cache, and a Markov logit correction is applied between draft positions. One
verify then checks `[pending, d1..d7]` on the target, using the same span-mode verify, accept,
commit and rollback as MTP.

It plugs into the generic spec contract (`devgen/src/mtp.rs` emit, `exec/gpu/spec.rs` verify,
`spec_draft.rs` draft programs, `orch/speculative.rs` policy) as `drafter.kind = device_block`,
spec family `gemma4_dspark`, selected with `PLOW_EMIT_SPEC=1 PLOW_SPEC_KIND=gemma4_dspark`.

## Done (branch `g12-dspark`, merged `172dd6f7b`)

| Phase | Commit | Content |
|---|---|---|
| D1 | 6606a5a3, 6f77a95a, e3a556c5 | Taps after layers 5/17/29/41/46 in every target program (prefill buckets, packed prefill, decode rungs 1–128, span verify). Context tail (fc per tap, hidden_norm, k_proj) writes drafter KV `kv.48..52.*` through clones of the target's KV writers. `spec.context.json` layout digest. Rekey (`scripts/gemma4_dspark_rekey.py`) and FP8 twins (`scripts/gemma4_dspark_fp8.py`). plowrt refuses live-KV for these packets (VMM prefix layout only). |
| D2 | a9001981 | Block draft program: 7-row non-causal forward over the 5 drafter layers, span attention at kv_len = c+7, lm_head + softcap, unfused Markov loop, rows {1,2,4,8}. Block drafters in the plow-asset spec-draft manifest. `step_bench --spec dspark` and block `--draft-bench`. `spec_serve` refuses block drafters. |

## Measured (Gemma-4 12B FP8, H100)

**D1 gates** (DSpark-on vs DSpark-off packet):
- Target token digests are identical in 21/21 cells (B 1–128 × ctx 1K/4K/16K).
- Decode tail cost is ≤ 1% in 19/21 cells; c1K B32 is +1.36% and c16K B16 is +1.15%. The bandwidth floor is ≈ 0.65%.
- Prefill TTFT is +1.1–1.75% (gate ≤ +2%).
- W8A8 tail weights cost 0.6–0.7 pt of acceptance vs the reference tail (kept; limit 1 pt).
- Poison and boundary sweeps (4090/4094/8187/8191, prefix hit then decode) pass.

**D2 agreement:** teacher-forced draft tokens match a CPU torch reference on 98.2% of steps (275/280). All misses are near-ties (top1–top2 margin ≤ 0.25); the reference against itself agrees 278/280.

**Draft cost D(B)** at k = 7, in ms:

| ctx | B1 | B2 | B4 | B8 |
|---|---|---|---|---|
| 2048 | 2.78 | 3.11 | 3.98 | 45.7 |
| 6144 | 3.41 | 4.36 | 6.51 | 50.7 |
| 16000 | 4.45 | 7.27 | 12.46 | 62.6 |

**End-to-end gain** (rung model at the gpu04 slot mix):

| Drafter | Gain |
|---|---|
| DSpark, measured D(B) | 1.08x |
| DSpark, projected with the D3 fixes below | ~1.25x |
| MTP | 1.42–1.51x |

DSpark's own KV cache also costs ≈ 55% of KV capacity, which shrinks the prefix cache. DSpark would
win only at batch 1 in latency-bound deployments, and only with the kernel fixes.

## Plan to finish (when resumed)

Each step lists its gate; stop at any failing gate.

1. **Draft cost (prerequisite for D3).** Gate: measured D(B) ≤ 0.5x of today at B1–B4 and gain12 ≥ MTP + 0.1 on the target deployment's slot mix.
   - **Block attention reads KV once:** run flash decode with all 7 block rows in one GQA group (R × heads) instead of re-reading the slot's KV per row group.
   - **Rows ≥ 32 on tensor cores:** a TC64 / split-K GEMV for 56+ rows (B ≥ 8), so B8 stops falling off the TC path (`PLOW_NV_FP8_DECODE_TC64`).
   - **Explain the ~1.8 ms above the weight-bandwidth floor:** trace the draft object (`PLOW_NV_TRACE`, `STEP_BENCH_DRAFT_TRACE`).
   - **Fuse the Markov loop:** a fused `MarkovArgmax` op.
2. **D1 tail cost.** Gate: tail ≤ 1% in all 21 cells over two sessions. Attribute c16K B16 (+1.15%) and confirm c1K B32 (+1.36%) is session-to-session variation.
3. **D3 serve path.** Shared with MTP's P3 tick integration. Gate: FP32 gate at c1/c16; replay A/B at rate 1.8 vs MTP control, including cached% and prefix capacity.
   - Block-drafter draft step in `spec_serve` and a flat-cost policy for block drafters.
   - KV budget and prefix-capacity accounting for the drafter KV layers.
   - Drafter KV contents after a prefix-cache hit (not yet checked).
   - Spans over 32 rows: today there is no B ≥ 8 spec on 12B.
4. **Recipe.** `[spec] kind = "gemma4_dspark"` in the 12B recipe, `emit = true`, `serve = false` until step 3 passes.
5. **Other targets.** 26B-A4B has only third-party DSpark checkpoints; revisit when an official one exists.

## Resume triggers

- A DSpark checkpoint whose acceptance length beats MTP by ≥ 1 token at the deployment's
  temperature, or
- A batch-1, latency-bound deployment where step 1's draft-cost gate passes.

## Where things are

- Code: in the tree since `172dd6f7b` (branch `g12-dspark`, `6606a5a3..e3a556c5`).
- Checkpoint: `deepseek-ai/dspark_gemma4_12b_block7`, revision `2fa72e76`.
