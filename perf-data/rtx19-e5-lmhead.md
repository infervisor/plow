# rtx-19 E5 — lm_head optimization (decode fixed-cost intercept)

**Branch** `e5-lmhead` · **GPU** RTX PRO 6000 Blackwell Server (188 SM) · **date** 2026-07-22

Goal: shrink the fixed per-token decode intercept by optimizing the lm_head — the
262 144-vocab × hidden GEMV, the biggest single fixed-cost decode op. Two levers were
evaluated: (1) fuse the greedy argmax into the lm_head epilogue; (2) fp8-weight lm_head.

## Recon (file:line)

- **lm_head emit**: `crates/plowc/src/bin/gemma4.rs:2912-3005`. Gemma ties embed & lm_head
  (`tied=true`), so the head GEMV reads `n.emb` (embed_tokens). Op = `DevOp::Gemv` (M=1 decode,
  bf16) / `DevOp::GemvFp8` when a fp8 head twin is bound. Writes the full `[vocab]` bf16 logits
  to `n.logits`.
- **logit consumption**: `SoftCap` (cap 30, in-place read+write over 262 k) → `Argmax`
  (64-block, packed-u64 partials → `n.amax`) → `ArgmaxFin` (fold → `in.ids`). Kernels:
  `runtime/nvidia/op_elementwise.cuh:66/118/138`; packed key `sm120_common.cuh:75`
  (`[63:32]=order-preserving bf16, [31:0]=~index`, ties → lowest index). GEMV kernels
  `op_gemm.cuh:206` (bf16) / `:1665` (fp8); dispatch `interp_sm120.cu:604/637/833`.
- **fp8 status**: the `PLOW_FP8_HEAD` flag already exists (`gemma4.rs:724-736` declares
  `head8`/`head8s` from `fp8/…embed_tokens.weight`), BUT **the shipped fp8 twins (12B + 31B)
  contain no fp8 embed/lm_head** (verified: 656 / 820 tensors, zero `embed`/`lm_head`) → the flag
  had no weight to bind (`MISSING FP8 WEIGHT`). The fp8 head twin was manufactured here by
  quantizing the bf16 embed → e4m3, per-output-row scale = amax/448 (matches the layer-twin
  recipe: layer `q_proj` row max-abs verified = 448.0, scale `F32[N]`).

## Trace attribution (ctx 1k, 12B bf16, block-0 sample, `PLOW_NV_TRACE` decode object)

| op | body cycles | share of decode |
|----|------------:|----------------:|
| **lm_head Gemv** | **3 168 136** | **~7.9 %** (biggest single op; ~8× a per-layer GemvGlu ~385 k) |
| Argmax (64-blk) | 9 603 | ~0.02 % |
| SoftCap (262 k elementwise) | ~10 000 | ~0.03 % |
| Σ block-0 body | ~39.9 M ↔ 18.8 ms | 100 % |

The lm_head GEMV is **weight-bandwidth-bound** (12B: 262 144×3840×2 B = **2.01 GB** read/token;
31B: ×5376 = **2.82 GB**). The logit round-trip (write 0.5 MB + softcap 1 MB rw + argmax 0.5 MB
read ≈ 2 MB) is **~0.1 % of the weight read** → **fusing the argmax cannot move the intercept**;
**halving the weight read (fp8) can** (~1.5 ms fixed on 12B, ~2.5 ms on 31B).

## Built (behind flags; default byte-identical)

1. **`PLOW_FUSE_ARGMAX`** (default off) — new `DevOp::GemvArgmax` (=80) + `d_gemv_argmax`
   kernel (`op_gemm.cuh`). Each block folds its owned vocab slice into one packed-u64 argmax
   partial **in the lm_head GEMV epilogue**, reproducing `SoftCap`→`Argmax` **bit-for-bit**
   (stores softcapped logits too), then `ArgmaxFin` folds `n_cu` partials. Drops the `SoftCap`
   and `Argmax` packets (**542 → 540** decode packets). **ptxas: 0 spill bytes.** Greedy B=1
   bf16-head path only.
2. **fp8 lm_head** — manufactured e4m3 head twins, bound via `PLOW_FP8_DIR`. (Exercises the
   pre-existing `PLOW_FP8_HEAD` emit path.)

## Gate — token identity (`PLOW_FUSE_ARGMAX` on vs off)

| model | ctx | token stream | step-0 logits | argmax check |
|-------|----:|-------------|---------------|-------------|
| 12B | 1k / 4k / 16k | **IDENTICAL** | **BYTE-IDENTICAL** | AGREE |
| 31B (fp8 model, bf16 head) | 1k | **IDENTICAL** | **BYTE-IDENTICAL** | AGREE |

## 12B decode TPOT (mean ms/token) — RTX PRO 6000 Blackwell, 188 SM

| ctx | OFF (classic bf16 head) | FUSE (bf16 fused argmax) | Δ FUSE | FP8HEAD (fp8 lm_head) | Δ FP8HEAD |
|----:|------------------------:|-------------------------:|-------:|----------------------:|----------:|
| 1k  | 18.784 | 18.778 | −0.006 (noise) | **18.141** | **−0.643 (−3.4 %)** |
| 4k  | 18.841 | 18.840 | −0.001 (noise) | **18.205** | **−0.636 (−3.4 %)** |
| 16k | 19.248 | 19.249 | +0.001 (noise) | **18.616** | **−0.632 (−3.3 %)** |

- **fused argmax**: TPOT delta is noise, exactly as the trace predicted (the round-trip is
  ~0.1 % of the lm_head weight read). It is a correct, byte-identical cleanup that removes 2
  packets + a ~1.5 MB round-trip; it does **not** shrink the intercept.
- **fp8 lm_head**: a **fixed ~0.64 ms** shrink at every ctx (the intercept is ctx-independent),
  so the **relative** win is largest at short ctx. Token stream matched bf16 at 1k/4k; **one**
  token flipped at 16k over 32 gen steps (e4m3 precision — reported as its own row, vLLM keeps
  lm_head bf16).

## 31B (fp8 model) decode TPOT — headline

_(decode-only prompt path: the 31B prefill object hits an illegal access at 79.75 KB smem in the
standalone harness — the bf16-head baseline fails identically, so it is orthogonal to E5.)_

| ctx | FP8 (bf16 head) | FP8HEAD (fp8 lm_head) | Δ FP8HEAD | FUSE (bf16 fused) |
|----:|----------------:|----------------------:|----------:|------------------:|
| 1k  | 26.336 | **25.406** | **−0.930 (−3.5 %)** | 26.315 (noise) |
| 4k  | 26.483 | **25.569** | **−0.914 (−3.5 %)** | — |

- 31B fp8-head tokens **byte-identical** to the bf16 head at ctx 1k **and 4k** (head twin
  rel-err 0.003 → no flips); argmax device==host AGREE.
- Absolute shrink is larger than 12B (−0.93 vs −0.64 ms) because the 31B lm_head is bigger
  (2.82 GB → 1.41 GB per token vs 12B's 2.01 → 1.01 GB).
- **31B FUSE gate PASSES**: token stream + step-0 logits **byte-identical** vs the FP8 bf16-head
  baseline; TPOT 26.315 vs 26.336 ms (noise). argmax AGREE.

## Verdict

- **fp8 lm_head extends the short-ctx decode lead**: −0.64 ms fixed on 12B (−3.4 % TPOT at
  ctx 1k) and **−0.93 ms on 31B (−3.5 %)**, the biggest single fixed-cost op halved. Because the
  intercept is ctx-independent the relative win is largest at short ctx — exactly the regime where
  plow already leads vLLM-fp8. It costs +1 GB VRAM (the fp8 head twin resident alongside the bf16
  embed) and, at 3 % e4m3 error on 12B, a rare token flip (none observed on 31B, rel-err 0.3 %) —
  reported as its own row since vLLM keeps lm_head bf16.
- **fused argmax is a wash for the intercept** (proven by trace + measurement): the lm_head is
  weight-bandwidth-bound; the logit round-trip it removes is ~0.1 %. Kept behind a flag,
  byte-identical, 0 spills — a clean no-cost cleanup, not an intercept lever.
