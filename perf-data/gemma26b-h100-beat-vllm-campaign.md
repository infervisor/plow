# Gemma-4-26B-A4B on H100 NVL — "beat vLLM" campaign (2026-07-24)

Goal: beat vLLM (bf16+fp8) on prefill / decode / concurrent decode, per ctx.
Status: **diagnosis complete + one measured tuning win (occ2 +6.2%); plow still
loses short-ctx B=1 decode, prefill, and concurrent — the only documented win
is long-ctx decode.** Levers identified and ranked below.

## Baselines (this card, trusted)

| metric | plowrt (latest-main) | vLLM 0.25.1 | plow/vLLM |
|---|---|---|---|
| decode TPOT bf16 @ctx1024 | 9.340 ms | 4.833 ms | **1.93× slower** |
| decode TPOT fp8 @ctx1024 | 7.395 ms | 4.417 ms | 1.67× slower |
| prefill tok/s @ctx1024 | 8,127 | ~18,600 | 2.3× slower |
| prefill peak (ctx4096) | 10,449 | — | — |
| concurrent aggregate | B=4: 243.9 tok/s | peak 1,850 (bf16) | far behind |

## C1 diagnosis (measured, uncontended window)

- **Decode is occupancy-limited, not op-immature and not extra bytes.** The
  megakernel runs `__launch_bounds__(256,1)` → 1 block/SM → **12.5% occupancy**.
  Roofline: 7.63 GB/token ÷ 9.356 ms = **816 GB/s = 20.9% of peak** vs vLLM ~40%;
  40/20.9 = 1.92× = the exact gap. Triply corroborated (source annotation
  `interp_sm120.cu:411`, grid=132, roofline).
- **Op-class trace:** body 68% | inter-op counter-gate WAIT **29%** | signal 3%.
  GEMV-family = ~84% of body; FlashDecode ~16%. The 29% gate stall is a second
  occupancy symptom (too few warps to overlap op tails).
- **Fair comparison:** decode gathers only top-8 experts (~7.6 GB/token), same
  active-param regime as vLLM. All-128 is physically impossible (12.9 ms floor).
- **fp8 has NO prefill program** (capability gap). Concurrency is B=1 (1.00×) vs
  vLLM 17.42× — structural in the emitter.
- **plow WINS long-ctx decode** (`decode-only-sweep.md`): plow decode is ~flat in
  ctx (weight-bound) while vLLM balloons (+~12 ms/tok, attention-bound). 31B
  crossovers TP1≈119k, TP4≈42k. This is the winnable axis.

## Tuning results (measured)

- **occ2 (2 blocks/SM): +6.2%.** `PLOW_NV_FORCE_MINBLK=2` → 177→128 regs (spills
  to 1664 B) → grid 264 = 2 blocks/SM. Decode TPOT **9.340 → 8.763 ms** @ctx1024,
  correctness-neutral. Attacks the 29% gate stall (more warps overlap op tails);
  the body BW gain is limited (c1r: occupancy "inferior" for body). To SERVE it,
  the prefill object must share the grid (n_cu=264) — currently decode-only.
- **B=4 batched decode:** 16.40 ms/4-tok = **243.9 tok/s aggregate** (61/user).
  Weight-sharing works (~7 ms fixed shared, ~2.35 ms/extra token) but the slow
  base makes aggregate far below vLLM's 1,850. Batched decode inherits the
  occupancy limit.

## Tiling / split-K analysis (per the "check split-k" review)

- Decode GEMV is **N-split** (column ownership blocked, `per=N/nblk`), **K read
  whole** per warp. NOT K-split.
- **K-split explicitly rejected for QKV** (`op_gemm.cuh:1700`: breaks the
  gemv→headnorm column map).
- **K-split is UNEXPLORED for non-headnorm projections** (down/o/gate/up/lm_head)
  — would add concurrent HBM streams (more MLP → higher achieved BW, directly
  attacking the 21%-of-peak body), at the cost of a cross-block partial-sum
  reduce. **A genuine untried lever.**
- Prefill GEMM: BM×BN×BK tiles + cp.async double-buffer; `weight_tiling` pins
  (BN,BK) via the tune system.

## Ranked levers (to close/beat)

| # | lever | axis | evidence | status |
|---|---|---|---|---|
| 1 | **cp.async row-staged gemv** | decode body | c1r +25–30%, saturates HBM, occ-neutral | scoped, probe had 3 defects — needs correct impl |
| 2 | **occ2 (2 blocks/SM)** | decode gate-stall | **measured +6.2%** | done; needs grid-matched prefill to serve |
| 3 | **K-split non-headnorm gemv** | decode body BW | untried; physics-sound | new (this review) |
| 4 | **long-ctx decode** | decode @≥~40k | documented plow win | measure 26B crossover |
| 5 | wgmma batched decode (B≥16) | concurrent | c1t: TC wins 4× at B≥16 | after decode-body fix |
| 6 | fp8 prefill program + fp8 TC decode | prefill/decode | absent capabilities | larger work |

## Honest verdict

On 26B short-ctx / B=1, plow loses ~1.9× and the occupancy-limited megakernel is
a structural handicap; occ2 (+6%) + cp.async (+25–30%) + K-split would narrow but
likely not beat vLLM's 4.83 ms there. plow's **winnable "beat vLLM" ground is
long-ctx decode** (flat plow vs ballooning vLLM) and, if the decode body is fixed,
concurrent decode. Prefill (~2×, vLLM tensor-core) is the hardest.

## UPDATE — long-ctx decode REFUTED (the last winnable axis)

Measured plow vs vLLM decode at ctx 8k→128k (bf16, TP1, B=1):

| ctx | plow ms | vLLM ms | ratio |
|---|---|---|---|
| 8192 | 9.531 | 5.034 | 1.89 |
| 16384 | 9.661 | 4.888 | 1.98 |
| 32768 | 9.930 | 5.032 | 1.97 |
| 65536 | 10.439 | 5.317 | 1.96 |
| 131072 | 11.467 | 5.917 | 1.94 |

**No crossover. plow loses every ctx and the gap WIDENS** (plow slope 1.58e-5 vs
vLLM 0.72e-5 ms/tok). The 31B/MI350X long-ctx win does NOT transfer: it needed
vLLM's attention to balloon (TRITON_ATTN, +12 ms/tok). On H100/26B vLLM uses
**FA4 + cudagraphs**, and Gemma's **5:1 sliding-window** keeps KV at 5.66 GiB even
at 128k, so vLLM ITL barely moves (+0.88 ms over the whole range).

## FINAL VERDICT — beating vLLM on 26B/H100 is NOT achievable with current plow

plow loses on **every measured axis** — short-ctx decode (1.9×), long-ctx decode
(1.9×, widening), prefill (2.3×), concurrent (B=4 244 vs 1850 tok/s). One root
cause: the occupancy-limited megakernel (12.5% occ, 20.9% of peak HBM) vs a very
strong H100 vLLM (FA4, cudagraphs, sliding-window KV).

The gap is ~90%. The available levers do not close it:
- occ2 (2 blocks/SM): **+6.2% measured**.
- cp.async row-staged gemv: +25–30% proven (unimplemented) → ~6.5 ms, still 1.35× behind.
- K-split non-headnorm gemv: unquantified, best-case tens of %.
Even stacked (~35%), they cannot close 90%. Short-ctx B=1 is structurally the
megakernel's worst case, and vLLM does not weaken at long ctx here.

**What would be required to win:** a fundamentally different decode path (not the
one-block/SM megakernel) — e.g. per-op high-occupancy kernels with cudagraph-style
capture (i.e. converging toward vLLM's design), or targeting a regime this
hardware/model does not exercise (much larger MoE where active-param sparsity is a
bigger lever, or bandwidth-starved GPUs where vLLM's kernels underperform). On
26B/H100 specifically, vLLM is the stronger engine and the goal as framed is not
reachable by tuning.
