# Gemma-4-26B-A4B grouped-MoE PREFILL — plow sm_120 (bf16) — campaign P9

Token-sorted grouped expert GEMM prefill for the 26B-A4B MoE block on one RTX PRO 6000
Blackwell (sm_120, 188 SMs). First correctness-first implementation; measured 2026-07-20.
Branch `worktree-agent-a5787ca0ad7745959`.

## What was built

Five device ops (ids 73–77), the `_pf` (`PLOW_NV_PREFILL`) interpreter object, and the
`gemma4.rs` emitter branch for the MoE FFN at T>1:

| id | op | body |
|----|----|------|
| 73 | `MOE_ROUTER_GEMMA_PF` | T-token router (block-per-token loop of the exact decode router; bit-identical per token) |
| 74 | `MOE_ALIGN_GEMMA_PF` | single-block histogram → padded prefix (128-tile) → scatter (token,part-row,gate) into expert-contiguous gathered rows |
| 75 | `MOE_GROUP_GLU_GEMMA_PF` | grouped gate_up GEMM + GeGLU, gathered A via `row_token`, fused-expert B; reuses the op_gemm.cuh tiled-GEMM (cp.async, m16n8k16, 128×128) |
| 76 | `MOE_GROUP_DOWN_GEMMA_PF` | grouped down GEMM, ×gate, scatter to `part[token*k+slot]`; pad rows skipped |
| 77 | `MOE_COMBINE_NORM_GEMMA_PF` | T-row combine + sandwich (`post_ffn_norm_2(Σ_slot part)+h1`) |

Design: total routed rows `T*k` sorted by expert (atomic scatter is order-safe — per-row
GEMM math is independent and the combine order is fixed by `(token,slot)`); a flat
(expert, m-tile)×n-tile work list read from a device `meta` table drives the persistent
grouped GEMM; the gather is fused into the A-tile cp.async load.

## Correctness gates (all PASS)

- **Oracle** (`sm120_interp_op_test: ok`): full chain vs f32 CPU golden at real geometry
  (H2816/E128/k8/I704) — router ids, lowest-id tie, align invariants (counts/unique/pad/
  gate-copy), combine relL2 4.3e-4…5.5e-4 (bf16-ulp maxabs) at T=16/129/512 + a 16-expert case.
- **Byte-identical decode**: decode-only packet md5-identical HEAD vs this branch
  (`PLOW_MOE_PREFILL=0`), also under `PLOW_UNISEG=1`.
- **Prefill == decode-consume parity (exact)**:
  - 512-tok exact-bucket prompt: **32/32 EXACT**.
  - 300-tok padded to bucket 512 (212 pad rows): **24/24 EXACT** — pad-row path proven clean.
  - 2204-tok prompt (padded to 4096): first gen token EXACT (8957 == control).
  - ≥1024-tok synthetic "random dataset" (arithmetic-sequence) prompts diverge after the
    near-tie boundary. This is inherent bf16 prefill-vs-decode drift, **reproduced identically
    by the proven dense 12B prefill** (512: 32/32 EXACT; 1024: 1/32) — not a MoE bug.

## TTFT — bucket ladder [128..8192], steady-state (run1≈run2 <0.1%)

| ctx | plow TTFT (ms) | tok/s | vLLM TTFT (ms) | plow/vLLM |
|-----|---------------:|------:|---------------:|----------:|
| 1k  | 112.9 | 9070  | 75  | 1.51× |
| 4k  | 323.4 | 12663 | 169 | 1.91× |
| 16k | 1403  | 11676 | 799 | 1.76× |

vLLM = tuned FlashInfer CUTLASS unquantized MoE (550 s warmup autotune); plow = first
correctness-first grouped kernels, **untuned**. ~1.5–1.9× behind — the same untuned-vs-tuned
band as the 26B decode gap (1.5–1.7×). Headroom: persistent (expert,tile) scheduling,
larger N-tiles, activation-stationary reuse, and fp8 grouped MoE (vLLM's fp8 MoE here is
untuned Triton — the softest target).

## Honest gaps

- Not yet beating vLLM TTFT; kernels are correctness-first (single design, no tile sweep).
- fp8 grouped MoE prefill not implemented (bf16 only; `PLOW_MOE_PREFILL` decode-only fallback
  stays fp8-safe).
- Parity on natural-language prompts not run (no tokenizer at hand); the 512/300-tok exact
  matches + dense-12B calibration establish correctness on confident prompts.
