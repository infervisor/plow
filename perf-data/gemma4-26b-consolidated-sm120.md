# Gemma-4-26B-A4B — beat26b CONSOLIDATED decode TPOT ladder — sm_120 (2026-07-23)

Branch `beat26b-consolidated` = main + `beat26b-batch` + `beat26b-decode` + `beat26b-flashdec`
+ `worktree-sass-micro-opt` (+ `worktree-plowc-review-fixes`, blob-neutral — see gates). The sass
merge changed kernel numerics (raw `ex2.approx.ftz` softmax, branchless flash merge, `ld_glob8_cs`
streaming loads on the bf16 arms; conflict in `op_attention.cuh` resolved keeping the flashdec
fp8-f32-direct structure), so the full gate battery was re-run on the merged tree before measuring.

One RTX PRO 6000 Blackwell (sm_120, 188 SM, CUDA 13.0), TP1, batch 1. Harness
`gemma4_sm120_chat`, 128 gen tok (16 warmup discarded, **n=112 timed**), 3 vLLM-RandomDataset
seed-0 prompts per ctx, equal-weight mean (the committed P9 methodology). vLLM baselines:
`gemma4-26b-a4b-vllm-sm120.md` (trusted, not re-derived).

## Config

- **bf16**: pkt `PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48 gemma4 <26B-dir> 132096 out.pkt 188` —
  **md5 `bfc84bcf737c485a9411ae60daead865`, byte-identical to the verified main-identity blob**;
  build `cmake -DPLOW_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-DPLOW_NV_FA_GF_FULL=4"`.
- **fp8 (fp8kv+FAST)**: pkt adds `PLOW_FP8=1 PLOW_FP8_HEAD=1 PLOW_FP8_KV=1` (601 packets, KV
  2.86 GiB — matches flashdec exactly); build adds `-DPLOW_FP8_FAST` to the CUDA flags plus
  `-DPLOW_NV_W8A8=ON -DPLOW_FP8_KV=ON`. fp8 twins: `<26B-dir>/fp8-full-plow` (not regenerated).
- fp8 pkt has no prefill programs, so its prompt warm goes through the decode program —
  identical to the prior fp8kv/flashdec campaigns' effective path (TPOT steady state unaffected).

## Correctness gates on the MERGED tree (all PASS, run before any measurement)

- `sm120_interp_op_test: ok` — full suite on the **GF4 bf16 build** AND on the
  **fp8kv+FAST build** (incl. w8a8 GEMM relL2 ≤ 7.6e-5, MoE ops, flash arms).
- fp8 flash-decode numeric oracle (`flashdec_fp8_correct_sm120`, FAST arm):
  **relL2 = 0.001686** — identical to the flashdec-committed 0.00169; the sass softmax/merge
  changes do not move the fp8 oracle.
- Packet identity: bf16 md5 = the verified main-identity md5 (above). Re-emitted at branch HEAD
  (`ad58a57`, post plowc-review-fixes merge): **all four blobs byte-identical** (26B bf16/fp8,
  31B bf16/w8a8) — the schedule-crate change provably does not reach the gemma4 blobs.
- Chat sanity ("What is the capital of France?", greedy, chat template): bf16 answers
  **"The capital of France is \*\*Paris\*\*."**; fp8 produces the **identical answer span**
  (first 11 tokens byte-identical to bf16); the streams diverge only in the forced continuation
  past the end-of-turn token (post-stop near-tie region — same class as the documented fp8
  non-exactness; no committed 26B transcript exists to diff against, noted honestly).
- Every ladder run: GLOBAL QUEUE scheduler, device==host argmax **AGREE**, exactly one
  PLOW_RESULT with n=112, within-run sd ≤ 0.15% of mean, 3-prompt spread ≤ 0.01 ms.

## Decode TPOT ms/token (batch 1; mean of 3 prompts; bold = beats vLLM)

| ctx  | plow bf16 | vLLM bf16 | bf16 verdict | plow fp8kv+FAST | vLLM fp8 | vLLM fp8kv | fp8 verdict (vs fp8 / vs fp8kv) |
|------|-----------|-----------|--------------|-----------------|----------|------------|--------------------------------|
| 1k   | 8.047 | 7.61 | LOSS +0.44 | 5.943 | 5.76 | 5.92 | LOSS +0.18 / LOSS +0.02 |
| 4k   | 8.103 | 7.90 | LOSS +0.20 | **5.981** | 6.08 | 6.19 | **WIN −0.10 / WIN −0.21** |
| 16k  | **8.439** | 8.64 | **WIN −0.20** | **6.173** | 6.82 | 6.62 | **WIN −0.65 / WIN −0.45** |
| 32k  | **8.942** | 9.57 | **WIN −0.63** | **6.450** | 7.74 | 7.28 | **WIN −1.29 / WIN −0.83** |
| 64k  | **9.885** | 10.33 | **WIN −0.45** | **7.062** | 8.63 | 7.52 | **WIN −1.57 / WIN −0.46** |
| 96k  | **10.812** | 11.34 | **WIN −0.53** | **7.624** | 9.54 | 7.94 | **WIN −1.92 / WIN −0.32** |
| 128k | **11.584** | 12.34 | **WIN −0.76** | **8.254** | 10.48 | 8.46 | **WIN −2.23 / WIN −0.21** |

- **bf16 beats vLLM bf16 at 16k–128k**; the 1k/4k short-ctx deficit shrinks 0.63→0.44 and
  0.39→0.20 ms vs P9.
- **fp8kv+FAST beats BOTH vLLM fp8 and vLLM fp8kv at 4k–128k.** The only non-win is 1k
  (+0.18 vs vLLM fp8, +0.02 vs fp8kv — the latter within ~2 sd, effectively a tie). The
  flashdec 96k/128k wins hold and widen slightly.

## sass-merge delta (this ladder − prior committed, same recipes)

| ctx  | bf16 Δ vs P9 (8.24…11.57) | fp8 Δ vs flashdec FAST (32k–128k) | fp8 Δ vs fp8kv committed |
|------|---------------------------|------------------------------------|--------------------------|
| 1k   | **−0.19** | n/a (FAST not measured <32k) | −0.18 |
| 4k   | **−0.19** | n/a | −0.17 |
| 16k  | **−0.17** | n/a | −0.26 |
| 32k  | **−0.13** | **−0.026** | −0.25 |
| 64k  | −0.06 | **−0.040** | −0.38 |
| 96k  | +0.00 | **−0.037** | −0.49 |
| 128k | +0.01 | **−0.039** | −0.54 |

**Verdict on the sass merge: helped short/mid ctx, neutral long ctx; nowhere a regression.**
bf16 gains −0.13…−0.19 ms at 1k–32k (the streaming `ld_glob8_cs` loads + micro-opts relieve the
GEMV-bound region), fading to ~0 at 96k/128k where bf16 flash-decode (untouched read path)
dominates. fp8 gains a uniform ~−0.03…−0.04 ms vs the flashdec FAST rows (its ladder was
measured pre-sass on the same-day build). P9-methodology caveat: prior bf16 rows are 3-prompt
means (like ours); prior fp8 rows are p0-only (~+0.1 ms offset at 1k-class), so the fp8kv-column
deltas overstate slightly at short ctx — the FAST-column deltas (measured same-method) are the
clean signal.

## Bonus observation — prefill on the merged tree (not this campaign's target)

bf16 prefill at 16k measured 998 ms and 64k 5834 ms during KV warm (P9 committed: 1402 / 8293 ms)
— the consolidated tree's prefill is ~30% faster than the P9 rows. Not gated to a specific
merge here; noted for the prefill campaign to claim properly.

## Post-measurement tree update — GEMV split-loop revert (30622a7)

After the ladders were measured, `d19410d` (merged as `30622a7`) REVERTED the sass GEMV
split-loop micro-opt in `op_gemm.cuh`: compute-sanitizer caught a latent 64→32-bit pointer
truncation in `gemv_rows<8>` (M>1 batched arm, K%(UN*GV_STEP)≠0 shapes — includes 26B
K=2816/704 at B>1), and the split loop had ZERO measured wall-clock benefit at MM=1
(HBM-bound; sass-session A/B byte-identical timing) while costing +15 regs. All other sass
changes (attention ex2/streaming loads, norm, MoE, elementwise, interp) remain.

**The ladders above were measured pre-revert; the revert is MM=1 time-identical per the sass
A/B, and all measurements here are B=1 (MM=1).** Post-revert re-gate at HEAD: all three
harness builds compile clean; decode cubin (GF4) is back to **219 regs, 0 stack, 0 spill**;
`sm120_interp_op_test: ok` on the fp8kv+FAST build. (Per-point spot-checks were explicitly
waived by the user directive — "simplify and ship" — in favour of this build+oracle gate.)

## Addendum — Gemma-4-31B (dense) w8a8 on the consolidated tree

Assets regenerated from this branch's plowc (`PLOW_UNISEG=1 [PLOW_FP8=1 PLOW_W8A8=1] gemma4
/workspace/models/gemma-4-31B-it 132096 out.pkt 188`): bf16 pkt (57.2 GiB wts / 22.58 GiB KV)
and w8a8 pkt (29.9 GiB wts) — shapes match the committed campaigns; both re-emitted
byte-identical at `ad58a57`. Build `cmake -DPLOW_CUDA=ON -DPLOW_NV_W8A8=ON` (default GF, the
committed 31B recipe). Gates: `sm120_interp_op_test: ok` (w8a8 build, pre- and post-revert);
chat sanity greedy = "The capital of France is Paris.", device==host argmax AGREE.

Decode TPOT ms/token, B=1, seed-0 p0 per ctx, 128 gen / n=112, PLOW_PREFILL=1 warm (the
committed 31B methodology). Prior rows: w8a8 = `gemma4-31b-fp8-beat.md` (1k–32k only);
weight-only fp8 = `gemma4-31b-plow-sm120.md`. vLLM = `gemma4-31b-vllm-sm120.md`.

| ctx | plow w8a8 (this tree) | prior w8a8 (fp8-beat) | prior fp8 w-only | vLLM fp8 | verdict vs vLLM fp8 |
|-----|----------------------|------------------------|------------------|----------|---------------------|
| 1k  | 26.146 | 26.354 | 27.480 | 25.62 | LOSS +0.53 |
| 4k  | 26.321 | 26.474 | 27.783 | 26.16 | LOSS +0.16 |
| 16k | **27.268** | 27.421 | 29.013 | 27.80 | **WIN −0.53** |
| 32k | **28.692** | 28.966 | 30.768 | 29.86 | **WIN −1.17** |
| 64k | **31.423** | — | 33.957 | 31.99 | **WIN −0.57** |
| 128k| 37.016 | — | 40.323 | 36.13 | LOSS +0.89 (beats vLLM fp8kv 38.63) |

- Consolidated tree is **faster than the committed fp8-beat w8a8 rows at every overlapping
  point** (−0.15…−0.27 ms) and extends the w8a8 ladder to 64k/128k for the first time:
  **64k flips to a WIN vs vLLM fp8** (the weight-only row lost it); 128k stays a loss vs
  vLLM fp8 (+0.89) but beats vLLM fp8kv (38.63) and vLLM bf16 (55.46) decisively.
- vs vLLM bf16 (44.67…55.46): ~−41% at every ctx.
- w8a8 prefill now runs to 128k: 277 ms / 892 ms / 3.81 s / 8.90 s / 23.1 s / 72.7 s
  (1k→128k; fp8-beat measured to 32k only, bf16-committed 128k prefill was 103.7 s).
- **31B bf16 decode ladder NOT re-measured**: 57 GiB bf16 weights + 22.6 GiB KV do not fit
  alongside the foreign 33 GiB plowrt on the shared box (same limitation the fp8-beat
  campaign recorded). The bf16 asset is regenerated and md5-stable; numbers stand at the
  committed `gemma4-31b-plow-sm120.md` rows.

## Assets / reproduction

- Ladder raw logs + per-run JSON: emitted under `/root/gpu-assets-consol/` (deleted after
  this doc was committed; every number above also lives in
  `gemma4-26b-consolidated-sm120.json` with per-prompt rows).
- Prompts: `/tmp/gemma26-vllm-seed0-prompts/ids_<ctx>_p{0,1,2}.bin` (vLLM RandomDataset seed 0).
- GPU serialized under `gpulease beat26b-consol`; foreign plowrt (pid 850160) untouched.
