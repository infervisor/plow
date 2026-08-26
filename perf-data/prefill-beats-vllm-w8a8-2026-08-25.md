# plow beats vLLM on prefill — one fixed setting, w8a8, Gemma-4-12B / RTX 5090 (sm_120a)

Follow-on to `perf-data/prefill-single-setting-win-2026-08-25.md`, same session/box. That report
closed the generic-input gap from 31% to ~19% at input-len 8192 (bf16, single-chunk bucket +
`PGM_BN=192`) via two zero/low numerics-risk levers. This report adds a third, higher-risk lever
— **true w8a8 (fp8 weights AND fp8 activations, native `mma.m16n8k32.e4m3`)** — and **plow now
beats vLLM at this one fixed setting.**

**Setting**: input-len 8192, concurrency 1, `--random-output-len 8` (TTFT-only), same
`vllm bench serve --backend openai-chat` protocol as every number in this repo's prior reports.
Chosen per explicit user direction: plow assets are precompiled per-shape, so pick ONE operating
point and specialize for it rather than chasing a generic win.

## Headline

| config | TTFT (ms), mean of N runs | vs vLLM |
|---|---|---|
| vLLM 0.27.0, bf16 (re-verified fresh, same session) | 1220.9 (3 runs, 1211.8-1233.3) | — |
| plow, original chunked bf16 baseline | 1744.3 | 0.70x (31% behind) |
| plow, single-chunk bucket + `PGM_BN=192` (bf16) | ~1437 | 0.85x (19% behind) |
| **plow, w8a8 (this report)** | **~959 (10 runs, 943.6-969.4)** | **1.27x — plow 27% FASTER** |

Zero failed requests at every point, every engine, every run.

## What changed from the bf16-best config

Same asset shape as before (single 8192-token bucket via `PLOW_MAX_CHUNK=8192`, eliminating the
per-chunk weight-restream cost `prefill-single-setting-win-2026-08-25.md` already identified as
the dominant fixable cost) — but weights AND activations are now fp8 (e4m3), dispatched through
the **native** fp8 tensor-core path (`mma.m16n8k32.e4m3`), not the w8a16
dequant-to-bf16-then-bf16-mma path tried and rejected earlier this session (see below).

- **Emit**: `PLOW_UNISEG=1 PLOW_MAX_CHUNK=8192 PLOW_W8A8=1 plowc --hf-dir ... --fp8
  --weight-dtype fp8 --out assets/gemma4-12b-prefill-w8a8-mc8192`. `PLOW_W8A8` is an existing,
  documented emitter flag (`crates/devgen/src/emit_config.rs:56-67`, mutually exclusive with
  `--w8a16`) selecting the "fp8 weights + fp8 activations" profile — distinct from `--fp8
  --weight-dtype fp8` alone, which is the w8a16 (fp8-weight, bf16-activation) profile.
- **Cubin**: `scripts/build_sm120_cubin.sh <out> -DPLOW_NV_W8A8=ON` — an existing, documented
  CMake option (`runtime/CMakeLists.txt:20`, off by default) that compiles in the true w8a8
  dispatch arms (`d_gemm_w8a8`/`d_gemm_glu_w8a8` in `op_gemm.cuh`). Register cost on the real
  production prefill object: `REG:240 → 242` (negligible), `STACK`/`SHARED` unchanged.
- Layered on top of the single-chunk bucket fix; **not** combined with the bf16-only `PGM_BN=192`
  tile change (orthogonal, not re-tested combined — see Open items).

## Correctness discipline — the honest numerics picture

This is a real precision change (fp8 weights AND fp8 activations, not a bit-exact refactor like
the earlier `PGM_BN_GLU` split), so it does not get the "exact match" bar the bf16-only levers
did. Gated as follows:

- `grep -aq libcuda.so.1` on `plowrt`, no sibling GPU process — pass, every run.
- Greedy "Paris" and the bicycle-balance paragraph — **exact text match** to the bf16 baseline on
  both smoke prompts (surprising given the precision change, but consistent across w8a16 and
  w8a8 alike; these are easy, low-entropy completions and should not be over-read as proof of
  general fidelity).
- **GSM8K, first N=50 then the full N=200 (8-shot CoT, greedy)**, same question set across
  configs, reusing this repo's own method (`scripts/bench_gsm8k.sh`'s prompt format,
  `final-number` extraction, and exact-match scoring) via a standalone harness pointed at the
  already-running server (the shipped script starts its own server through `nix develop`,
  incompatible with this session's manual cubin/asset wiring):

  | config | GSM8K acc (N=50) | GSM8K acc (N=200) |
  |---|---|---|
  | bf16 (single-chunk + BN=192) | 98.0% (49/50) | **96.5% (193/200)** |
  | w8a16 (fp8 weight, bf16 activation) | 96.0% (48/50) | not re-run (already rejected on speed) |
  | **w8a8 (fp8 weight + fp8 activation)** | 96.0% (48/50) | **96.0% (192/200)** |

  **The N=50 read was misleading — the N=200 gate (this repo's own documented minimum before
  trusting an accuracy number) shows only a 0.5-point gap** (96.5% bf16 vs 96.0% w8a8), well
  inside one standard error (≈1.4pp at this N and accuracy level) — i.e., **not distinguishable
  from noise.** bf16's N=50 subset (98.0%) was itself the outlier, not w8a8's number; w8a8 was
  stable at 96.0% across both sample sizes. This is the headline accuracy result: **w8a8's cost
  on GSM8K, at this sample size, is not measurably different from bf16.** Still recommend the
  full N=1319 split and a benchmark beyond GSM8K before a real shipping decision — fp8 is known
  to be benign on arithmetic CoT specifically and can degrade differently elsewhere — but the
  single biggest worry from the first pass of this report is resolved.

## Why w8a16 was rejected but w8a8 wins (the mechanism)

Tried first, in-session, and correctly rejected before reaching for w8a8:

| config | TTFT (ms) | GSM8K (N=50) |
|---|---|---|
| bf16 baseline (this setting) | ~1437 | 98.0% |
| **w8a16** (fp8 weight, bf16 activation — dequant to bf16 in smem, same bf16 `mma.sync` loop) | **~1542 (SLOWER, +7-8%)** | 96.0% |
| **w8a8** (fp8 weight + fp8 activation — native `mma.m16n8k32.e4m3`) | **~959 (FASTER, -33%)** | 96.0% |

w8a16 only halves the HBM bytes for the weight read; it still runs the identical bf16 mma
instruction, and pays an extra fp8→bf16 dequant step in the smem stage for that saving. At this
single-chunk size the dequant overhead outweighs the halved read — a clean, reproducible
regression, paid for nothing (same 96.0% accuracy as w8a8, for a slowdown instead of a speedup).
w8a8 is different in kind, not degree: PX-9 (`perf-data/px9-gemm-body.md`, isolated microbench,
different box) measured `mma.m16n8k32.e4m3` at 516.9 TFLOP/s vs `mma.m16n8k16.bf16`'s 259.2
TFLOP/s on this same GPU generation — a genuine 2x compute-throughput instruction, not a
bandwidth trick. That's the lever that actually pays off here.

## What this report does and does not claim

- **Does claim**: at input-len 8192, concurrency 1, TTFT, on this box, this session — **plow
  (w8a8) beats vLLM 0.27.0 (bf16) by ~27%**, reproduced over 10 runs against a freshly
  re-verified vLLM baseline, zero failed requests either side.
- **Does not claim**: a win at any other context length or concurrency (not retested this pass —
  the whole point of this exercise, per user direction, was ONE fixed setting, not a generalized
  claim). Does not claim this is an apples-to-apples "best vLLM can do" comparison — vLLM was run
  bf16 with no quantization flags, matching every prior report's protocol; an fp8-quantized vLLM
  comparison point was not attempted (would need a different vLLM launch, not in scope for this
  pass). The N=200 GSM8K gate (run this session, see above) found w8a8's accuracy
  indistinguishable from bf16's at this sample size — a stronger result than the N=50 first read
  suggested — but N=200 is still short of the full 1319-question split, so "production-ready" is
  still not claimed outright.

## Open items

1. Run the full N=1319 GSM8K split for a tighter confidence interval than N=200 gives, and a
   benchmark beyond GSM8K (fp8 can be selectively benign on arithmetic CoT) before a real
   shipping decision.
2. Try `PGM_BN=192` combined with w8a8 (untested combination — the bf16-only tile change and the
   w8a8 precision change were validated independently, not stacked).
3. Try w8a8 at the other two previously-tested context lengths (2048, 16000) to see whether the
   win generalizes or is specific to 8192's tile/wave alignment.
4. Compare against an fp8-quantized vLLM baseline for a same-precision apples-to-apples number,
   not just plow-fp8-vs-vLLM-bf16.
5. Re-run the isolated GEMM oracle test used earlier this session (bit-exact bf16 check) — extend
   it with a w8a8 relL2 tolerance check against an f64 CPU reference, mirroring PX-9's own gates,
   rather than relying on GSM8K alone for numerics confidence.
