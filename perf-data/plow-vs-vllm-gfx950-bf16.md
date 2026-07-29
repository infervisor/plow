# plow vs vLLM — bf16 vs bf16, MI355X (gfx950), single GPU, concurrency 1

> ## ⚠ PROVISIONAL — mixed instruments. Do not quote the plow column as a like-for-like result.
>
> The two columns were produced by **different instruments**, and the difference favours plow:
>
> | | instrument | includes |
> |---|---|---|
> | vLLM | `vllm bench serve` → HTTP endpoint | server scheduling loop, HTTP, detokenisation |
> | plow | `plowrt amd-bench` | **device-side decode loop only** — no server, no HTTP, no detok |
>
> vLLM's per-token server overhead at concurrency 1 is plausibly ~0.5–1.5 ms, which is the **same
> order as the 0.81 ms margin** in the 12B 64k crossover below. So that crossover is *not* a safe
> claim yet, and neither are the ratios in either table.
>
> Per `plans/knob-contract.md` §0-BENCH, every headline number must now come from **`vllm bench
> serve` pointed at the plowrt OpenAI endpoint** — same client binary, same harness, same metric
> definitions for both engines. That path does not exist yet: `crates/plowrt/src/serve/mod.rs:252`
> gates the engine map on `#[cfg(feature = "cuda")]`, so there is no AMD serve today. Building it
> is in flight.
>
> **What survives regardless of instrument**, because it is a plow-vs-plow comparison across two
> models measured the same way: the fixed-overhead analysis at the bottom of this file (plow's
> overhead is a ~7 ms constant tracking layer count, not weight bytes). That is the finding worth
> keeping from this document; the head-to-head ratios are placeholders until re-measured.


Apples-to-apples: **both columns bf16**, same box, same context ladder, decode TPOT in ms/token.
Lower is better. Measured 2026-07-27.

- **vLLM**: `rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0` (vLLM
  `0.23.1.dev1`) — the newest CDNA tag on Docker Hub as of this date. Served endpoint,
  `vllm bench serve`, HIP graphs on. Data: `perf-data/vllm-rocm/*_ctxsweep_c1.csv`.
- **plow**: `plowrt amd-bench`, real weights bound, real schedule, global-queue scheduler,
  64 decode steps per point, under `perf-data/harness/gpulease -n 1` (rc=0, uncontended).
- Both checkpoints carry no `quantization_config` and BF16 tensors, so both columns really are
  bf16 — see `perf-data/vllm-rocm/PRECISION-LABELS.md` for why that check matters here.

## Gemma-4-12B-it — plow WINS at long context

| ctx | vLLM bf16 | plow bf16 | ratio |
|---|--:|--:|--:|
| 1,024 | **6.80** | 10.758 | 1.58× slower |
| 4,096 | **7.57** | 10.513 | 1.39× |
| 8,192 | **8.62** | 10.601 | 1.23× |
| 16,384 | **9.21** | 10.830 | 1.18× |
| 32,768 | **11.18** | 11.241 | 1.01× (parity) |
| 65,536 | 12.70 | **11.89** | **0.94× — plow wins by 6.3%** |

**Degradation 1k → 64k: plow 1.10×, vLLM 1.87×.**

The 64k crossover was repeated because it is the headline claim: **11.882 / 11.925 / 11.886 /
11.923 / 11.891 ms** — 5 runs (one at 64 steps, four at 96), **sd 0.02 ms**, all under an
uncontended lease. The 6.3% margin is ~40× the run-to-run spread, so it is not noise.

## Gemma-4-31B-it

| ctx | vLLM bf16 | plow bf16 | ratio |
|---|--:|--:|--:|
| 1,024 | **13.51** | 17.35 | 1.28× slower |
| 8,192 | **15.57** | 17.9 | 1.15× |
| 32,768 | **19.00** | 19.3 | 1.02× |
| 65,536 | **20.79** | 21.1 | 1.01× (parity) |

**Degradation 1k → 64k: plow 1.22×, vLLM 1.54×.**

## What the two models jointly show: plow's overhead is a CONSTANT, not a rate

Weight-stream floor at the measured 6200 GB/s (`runtime/amd/op_gemm.h`), ctx 1024:

| model | weights | floor | vLLM | vLLM over floor | plow | **plow over floor** |
|---|--:|--:|--:|--:|--:|--:|
| Gemma-4-12B | 22.2 GiB | 3.84 ms | 6.80 | +2.96 ms | 10.758 | **+6.92 ms** |
| Gemma-4-31B | 57.2 GiB | 9.90 ms | 13.51 | +3.61 ms | 17.35 | **+7.45 ms** |

The model nearly triples and **plow's overhead moves 6.92 → 7.45 ms** (+8%). It tracks *layer
count* (48 → 60, +25%), not weight bytes. vLLM's overhead is ~3–3.6 ms on the same ladder.

This is the whole story, and it is consistent with the trace-level diagnosis in
`plans/knob-contract.md §6a`: at full 256-workgroup occupancy plow runs at **96% of the memory
ceiling**, and the deficit is time when the machine is *starved* — 300 of 676 decode packets
(`norm_residual_norm` on **1** CU of 256, `headnorm_rope` on 4, `flash_merge` on 32) with a
measured ready-queue of **zero** behind them, plus a 1.80 ms straggler tail.

Two consequences:

1. **plow wins where the constant is amortised** — long context, where attention grows and the
   fixed per-layer cost stops dominating. That is exactly where the 12B result crosses over, and
   why plow's degradation curve is roughly half vLLM's on both models.
2. **plow loses on small models and short contexts**, and it will lose *harder* at lower precision:
   the ~7 ms is precision-invariant, so halving the weight stream halves only the floor. Measured
   on 31B: fp8 buys plow just 16–19% where the floor says ~45% was available.

So the ranked work is width, not precision — see `plans/knob-contract.md §6c`.

## Reproduce

```
# vLLM
scripts/bench_vllm_rocm.sh google/gemma-4-12B-it 1     # PHASES=ctxsweep for the c1 ladder

# plow
./target/release/plowc --hf-dir <ckpt> --emit devblob --arch gfx950 --gpu mi355x \
    --n-cu 256 --max-ctx 131072 --out build-amd/g12b-bf16
perf-data/harness/gpulease -n 1 bench sg render -c \
  'plowrt amd-bench --blob build-amd/g12b-bf16/model.pkt --hsaco build-amd/hsaco \
      --checkpoint <ckpt> --steps 64 --ctx 65536'
```

`sg render -c` is required: this account is in the `render` group but the login session's
supplementary groups omit it, so HSA init otherwise fails with 4104.

## Caveats, stated plainly

- One run per plow point (64 steps each, medians internally). The 12B 64k crossover is a 6.4%
  gap; it deserves a repeat before being leaned on hard.
- `amd-bench` decodes from a given ctx without prefilling it, so the KV attended over is not the
  KV a real conversation would have. Timing is representative; **the token ids are not**. The
  correctness oracle is `runtime/tests/gemma4_chat.c` (needs `plowc --no-rope-gen`).
- vLLM TTFT/prefill is not compared here. plow's prefill gap is larger than its decode gap.
