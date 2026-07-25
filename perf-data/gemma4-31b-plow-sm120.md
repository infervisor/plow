# Gemma-4-31B (dense) — plow vs vLLM on sm_120 (RTX PRO 6000 Blackwell)

**Campaigns T7-31b-decode / T7-31b-prefill-t5 / T7-31b-fp8.** Re-measures the dense
**Gemma-4-31B** on current-HEAD kernels (rtx `fd3a790`, ≥`3be7888`), lands one kept
decode win (flash-decode `nsplit` 12→16), re-establishes prefill on the T5+T6
kernels, and runs the full fp8 decode ladder now that T6's slim fp8 packet
(commit `3be7888`) elides the dead bf16 prefill weights. Single sequence, batch 1.

Supersedes the P2-31b baseline (measured PRE-T5 at `7fe44b8`); that data is kept in
`git log` and the campaign deltas below are stated against it.

## Setup

- **Model:** google/gemma-4-31B-it, rev `b9ea41a2`, `/workspace/models/gemma-4-31B-it`.
  60 layers (10 full / 50 sliding), hidden 5376, inter 21504, 32 Q heads,
  kv 16 slide / **4 full**, head_dim 256 slide / 512 full, sliding_window 1024,
  final_logit_softcapping 30, tied embeddings, vocab 262144.
- **plow build:** interp kernels **byte-identical to committed HEAD** (`fd3a790` rtx).
  Decode object `-DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2` (174 regs, 0 spill, occ-1);
  prefill object adds `-DPLOW_NV_PREFILL=1` (T5 cp.async KV-stream flash-prefill +
  T4 mma P·V + T6 fp8 GEMM). Harness `runtime/tests/gemma4_sm120_chat.cu`.
  Global-queue scheduler (default). **No kernel change this campaign** (see H1 below).
- **The one emitter change:** `crates/plowc/src/bin/gemma4.rs` — the decode `nsplit`
  for the 31B long-ctx shape is floored to 16 (was 12). Gated to the 31B signature
  (`kvh_full >= 4 && kvh_slide != kvh_full`, i.e. mixed sliding/full attention with
  4-KV full layers), so 12B (`kvh_full=1`), Qwen/Llama (`kvh_full==kvh_slide`) and
  short-ctx pkts (`<=8192`) are byte-identical. Verified: a default `gemma4` emit now
  produces a **byte-identical packet** to the old `PLOW_NS_ABS=16`.
- **Packet recipe:** `PLOW_UNISEG=1 gemma4 <dir> 132096 out.pkt 188` (bf16);
  fp8 adds `PLOW_FP8=1`. The latent prefill-argmax OOB the P2 note flagged is already
  fixed in HEAD (`nb_argmax` keyed off `decode && t>1`), so 31B prefill no longer
  faults; `PLOW_UNISEG=1` is still required.
- **vLLM baseline:** `gemma4-31b-vllm-sm120.json` (B1, vLLM 0.25.1, TRITON_ATTN
  forced, cudagraphs ON, `gpu_memory_utilization` 0.9, output_len 128).
- **Methodology:** prompt of length = ctx primed via PREFILL, then 128 decode tokens;
  discard 16 warmup, report over 112 timed steps. sd ≤ 0.06 % of mean at every point.

## The three hypotheses (A/B'd; winners kept)

**H1 — decode GEMV unroll: REFUTED on sm_120.** The AMD gemv-autotune win (commit
`46dccd6`: deeper per-shape unroll, +15 % on the 31B o_proj GEMV) does **not** transfer.
A/B at 1k+16k, decode object rebuilt at `GV_UNROLL 8→16` (+ GLU 4→8):

| variant | 1k | 16k | regs |
|---|---|---|---|
| UN=8 (baseline) | **47.591** | **49.581** | 174 |
| UN=16 | 48.030 (+0.9 %) | 50.013 (+0.9 %) | 204 |
| UN=16 / GLU=8 | 48.037 | 50.039 | 240 |

Root cause: the sm_120 decode megakernel is **already occ-1** (174 regs = 1 block /
8 warps per SM, register-bound; smem is 16 B). It allocates for the UNION of every
inlined decode arm, so deepening the GEMV unroll raised the WHOLE-kernel register count
(174→204→240) with no occupancy to gain back — and slightly starved flash-decode / the
norms on the critical path. AMD's win came from occ-2→deeper-unroll hiding latency; here
there is no such headroom. **Kept UN=8 → no kernel change**, interp byte-identical to HEAD.

**H2 — flash-decode `nsplit`: WON, shipped.** The 188-SM CU-fill formula
(`n_cu*mul/heads = 188*2/32`) gives `nsplit=12` for the 32-head 31B — a 1.0× flash fill
(`n_grp*ns = 16*12 = 192 ≈ 188`). MEASURED: `ns16` (1.36× oversubscribe) beats `ns12` at
**every** ctx — the fine full-layer KV splits (10 layers × kv4 × hd512) hide the long-ctx
read latency and the 32-workgroup merge still absorbs the extra partials. `ns24` over-splits.

| ctx | ns12 (old) | **ns16 (new)** | ns24 |
|------:|-----------:|-----------:|-----:|
|   1024 | 47.576 | **47.488** (−0.2 %) | — |
|   4096 | 47.947 | **47.789** (−0.3 %) | — |
|  16384 | 49.577 | **48.982** (−1.2 %) | — |
|  32768 | 51.837 | **50.727** (−2.1 %) | 51.156 |
|  65536 | 55.930 | **53.946** (−3.5 %) | — |
| 131072 | 64.109 | **60.374** (−5.8 %) | 61.595 |

fp8 (weight-only → identical KV/flash bytes) gains **more** long-ctx: 64k −5.6 %, 128k −8.6 %.

**H3 — fixed step-cost intercept: MEASURED.** Least-squares `tpot = a + b·ctx`:

| | intercept a (ms) | slope b (ms / 1k-ctx) |
|---|---:|---:|
| plow ns12 | 47.51 | 127.2 |
| **plow ns16** | 47.40 | **99.2** |
| vLLM bf16 | 45.38 | 81.1 |

`ns16` cuts the **slope** 127→99 (−22 %, closing most of the gap to vLLM's 81) — the flash
long-ctx scaling was the fixable part. It does **not** move the **intercept**: plow's ~47.4 ms
fixed per-token cost vs vLLM's 45.4 ms is a **+2.0 ms (≈4.5 %)** floor — the 60-layer
global-queue dispatch/gate overhead. Per the task this is measured and reported, not
redesigned; it is the residual reason 31B decode still trails vLLM at short ctx.

## Decode TPOT — bf16 (ms/token, lower is better) — SHIPPED ns16

| ctx | plow bf16 | sd% | tok/s | vLLM bf16 | **plow vs vLLM** | (was, P2/ns12) |
|------:|----------:|----:|------:|----------:|-----------------:|---------------:|
|   1024 | 47.488 | 0.05 | 21.06 | 44.67 | **+6.3 %** | +6.5 % |
|   4096 | 47.789 | 0.04 | 20.93 | 45.20 | **+5.7 %** | +6.0 % |
|  16384 | 48.982 | 0.06 | 20.42 | 46.93 | **+4.4 %** | +5.6 % |
|  32768 | 50.727 | 0.05 | 19.71 | 49.14 | **+3.2 %** | +5.4 % |
|  65536 | 53.946 | 0.06 | 18.54 | 51.22 | **+5.3 %** | +9.1 % |
| 131072 | 60.374 | 0.06 | 16.56 | 55.46 | **+8.9 %** | +15.5 % |

**31B bf16 decode still LOSES to vLLM at every ctx**, but ns16 roughly halves the long-ctx
gap (128k +15.5 %→+8.9 %, 64k +9.1 %→+5.3 %). It does **not** win like the 12B does; the
residual is the +2.0 ms fixed intercept (H3), not the KV slope (near-closed).

## Prefill (prefill_ms, lower is better) — bf16 — T5+T6 kernels

| ctx | plow prefill_ms | tok/s | vLLM TTFT_ms | **plow vs vLLM** | (was, P2) |
|------:|----------------:|------:|-------------:|-----------------:|----------:|
|   1024 |     473.4 | 2163.0 |    222.4 | 2.13× slower | 2.28× |
|   4096 |    1543.3 | 2654.1 |    705.7 | 2.19× slower | 2.53× |
|  16384 |    6169.8 | 2655.5 |   3450.7 | 1.79× slower | 2.64× |
|  32768 |   14126.9 | 2319.6 |  10730.2 | 1.32× slower | 3.23× |
|  65536 |   35412.6 | 1850.6 |  31095.9 | 1.14× slower | 4.21× |
| 131072 |  103726.0 | 1263.6 |  99564.7 | **1.04× slower** | 4.96× |

**Prefill is the big current-HEAD win.** T5's cp.async KV-stream flash-prefill collapses
the O(ctx²) tail: 128k went 493 s → 104 s (**4.7× faster**), from 4.96× behind vLLM to
**1.04× — near parity**, and the whole ladder is now 1.04–2.19× (was 2.3–5.0×). *(vs vLLM
served TTFT, which includes the 1st decode token; the task's specified baseline.)*
fp8 prefill (w8a16 GEMM) measured ≈ bf16 prefill (463 ms→104 s across the ladder) —
T6's fp8 GEMM is memory-neutral on prefill (as its commit noted); the fp8 win is decode-only.

## Decode TPOT — fp8 weight-only (ms/token) — SHIPPED ns16

| ctx | plow fp8 | tok/s | vLLM fp8 | **plow vs vLLM** | fp8 vs plow bf16 |
|------:|---------:|------:|---------:|-----------------:|-----------------:|
|   1024 | 27.480 | 36.39 | 25.62 | +7.3 % | 1.73× faster |
|   4096 | 27.783 | 35.99 | 26.16 | +6.2 % | 1.72× faster |
|  16384 | 29.013 | 34.47 | 27.80 | +4.4 % | 1.69× faster |
|  32768 | 30.768 | 32.50 | 29.86 | +3.0 % | 1.65× faster |
|  65536 | 33.957 | 29.45 | 31.99 | +6.1 % | 1.59× faster |
| 131072 | 40.323 | 24.80 | 36.13 | +11.6 % | 1.50× faster |

**The full fp8 decode ladder now runs 1k→128k** (was OOM-blocked above ~4.6k). T6's slim
fp8 packet elides the 57 GiB bf16 prefill projections — weights **29.9 GiB** (per-row e4m3),
so 29.9 + 22.6 KV + 2.7 act = **55.2 GiB at 128k**, fits with ~36 GiB headroom. fp8 gives
the expected **1.5–1.7× decode speedup over bf16** and is coherent, but like bf16 stays a few
% behind vLLM fp8 (KV is still bf16, so the flash floor is shared with the bf16 path).

## Correctness / parity gates

- **bf16 ns16 ≡ ns12 greedy tokens** — byte-identical generated sequences at 32k AND 128k.
  The shipped ns16 change is greedy-token-equivalent to the P2 default (ns12), which PASSED
  the HF-verified set (p2 48/48, p3 31/31 to EOS incl. the >1024-window-crossing prompt,
  p1 reconverging bf16 near-tie). Parity is **inherited, PASS**. (nsplit is an exact
  online-softmax reduction reorder; only the flash accumulation order moves, and greedy
  argmax is unchanged here.)
- **device == host argmax AGREE** at every measured ctx (1k…128k), bf16 and fp8, ns16.
- **fp8 vs bf16**: identical generated tokens at 16k/32k/64k/128k; 4k differs at 3/129
  near-ties — matches the P2 fp8 agreement pattern (fp8 agrees with bf16 except at bf16
  near-ties). **PASS.**
- **Oracle**: interp kernels byte-identical to HEAD (no kernel change), so the sm_120
  interp-op oracle is unaffected — no shape-specific kernel path was added.

## Memory at 128k (measured previously; unchanged)

bf16: weights 57.18 + KV 22.58 + act 2.69 = **82.45 GiB**; fits 96 GB, batch-1 only.
fp8: 29.9 + 22.58 + 2.69 = **55.2 GiB**.

## Headline

Current-HEAD 31B is **much closer to vLLM than P2**: **prefill went from 2.3–5.0× behind to
1.04–2.19× (near parity at 128k)** on the T5+T6 kernels, and the shipped `nsplit 12→16`
emitter tune roughly **halves the long-ctx decode gap** (128k +15.5 %→+8.9 %) while being
greedy-token-identical and 12B-safe. **fp8 decode now runs the full ladder** (slim packet,
1.5–1.7× over bf16). But **31B decode still does not WIN like the 12B**: it trails vLLM
bf16 by +3.2–8.9 %, and the A/B's honest verdict is that the residual is a **fixed ~2.0 ms
60-layer dispatch intercept** (H3) plus a shared bf16-KV flash floor — not the GEMV unroll
(H1, refuted) or the KV split (H2, now near-closed). Closing the last gap is a
scheduler/dispatch-overhead problem, out of this campaign's kernel/emitter scope.
