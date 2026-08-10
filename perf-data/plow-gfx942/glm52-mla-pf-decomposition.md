# GLM-5.2 MLA prefill flash: measured decomposition + the split-softmax fix (SMX)

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **CDNA3-SPECIFIC** — ablation-derived term split at gfx942 rates; SMX is gated ON for CDNA3 specifically.

2026-08-07, branch flash-mla-pf-rate (base a373889). Instrument: PLOW_MLA_PF_ABL probe
objects (op_attention.h; WRONG OUTPUT, build-axis-gated like PLOW_XR_NOWAIT) on the real
model — amd-bench --tp 8, T=8192 prompt, PLOW_TRACE_RAW, per-layer FlashMlaPrefill span
via trace_block.py. Wall-clock single-shots swing ±20% with DVFS/thermal ordering (the
2.2→1.58 GHz swing the trace header documents), so TRACED SPANS are the instrument and
final numbers are interleaved.

## Decomposition (flash span per MoE layer @8k; base 18.34 ms)

| arm | span | term priced | share |
|---|---|---|---|
| no-softmax (ABL=1) | 12.77 | **online-softmax reductions/exp/corr: 5.57 ms** | 30% |
| no-QK-MFMA (ABL=2) | 13.21 | QK^T + rope MFMAs (8×-redundant): 5.13 ms | 28% |
| stage-once (ABL=3) | 16.46 | K/rope global→LDS staging: 1.88 ms | 10% |
| no-PV-restage (ABL=4) | 17.81 | KSPLIT PV re-stage: 0.53 ms | 3% |
| (residual) | — | barriers + PV MFMA + P-store + loop: ~5.2 ms | 28% |
| QK1 (clean re-test) | 19.34 | **+1.0 ms — CONFIRMED NEGATIVE on correct objects** | — |

The softmax costs as much as the entire score matrix math: every one of the 8 waves
redoes the identical per-row max/exp/sum chains (8× redundancy). QK1's serialize-on-wave-0
answer loses because seven waves idle behind a barrier while one does 8× work.

## The fix: PLOW_MLA_PF_SMX (default ON for CDNA3, kill switch =0)

Split, don't serialize: each column-group wave OWNS 32/WPM rows of the M-tile
(`(row*WPM)/32 == cgrp` — half-wave-uniform under mfma_acc_m, so the owning half-wave's
shfl partners all stay active), computes reductions/exp/P-store for those rows only,
publishes corr[row] through the Csm strip, and EVERY wave applies the correction after
the existing P barrier (the same after-barrier position QK1 used — identical values,
identical multiply order, no new barrier). (m,l) epilogue moves to the row's owner.

**BIT-IDENTICAL: logits byte-compare vs hsaco_glm8 PASS.** Measured (interleaved,
2 rounds):

| metric | glm8 | SMX | Δ |
|---|---|---|---|
| flash span/layer @8k | 18.32 / 18.36 ms | **15.50 / 15.48** | −15.5% (reproduces to 0.02) |
| layer span @8k | 34.4 ms | 31.8 | −7.7% |
| prefill wall 8k | 2680 / 2766 ms | **2486 / 2467** | −8..−11% |
| prefill wall 16k | 8844 / 8621 ms | 7970 / 6567 | −8..−10% (16k r2 thermally lucky) |

## What remains in the flash body (per layer @8k), with prices attached

QK MFMA 5.13 ms (the WPM=8 structure makes every wave compute the full S — a genuine
restructure, not a knob), residual barriers/PV/loop ~5.2, staging 1.88 (register-prefetch
the next chunk behind the MFMA run — the d_moe_group_pf_t pattern), PV re-stage 0.53,
softmax remainder ~2.7. Probe axes stay in the tree for the next round.
