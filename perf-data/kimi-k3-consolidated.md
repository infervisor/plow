# Kimi-K3 performance record — consolidated

This is the authoritative index for K3 performance work in this repository. Detailed reports
remain as dated evidence links; this file owns the current baseline, decisions, and open work.

## Scope and measurement contract

- Current target: TP8, 8×MI355X/gfx950, native MXFP4 weights, BF16 KV.
- Current performance authority: `plowrt bench` for Plow and vLLM 0.28
  `bench serve` for the reference server.
- A valid result requires completed requests, exact output length, empty errors, all-rank token/counter
  agreement, and uncontended `gpulease`. Standalone kernels are screening evidence, not throughput.
- Current records: `kimi-k3-vllm-mi355x-baseline.md` and
  `kimi-k3-plowrt-mi355x-baseline.md`. The production Plow C1 record uses the
  same vLLM 0.28 client, raw-completions endpoint, tokenizer, 8192→1024 shape,
  and request contract. `kimi-k3-plowrt-mi355x-smoke-20260902.md` remains only
  a correctness and scheduler-coverage record.

## Current MI355X C1 baseline

| Engine | TTFT p50 | TPOT p50 | ITL p99 | Output throughput |
|---|---:|---:|---:|---:|
| Plow | 1271.86 ms | 28.53 ms | 28.60 ms | 33.63 tok/s |
| vLLM 0.28 | 567.74 ms | 20.81 ms | 20.97 ms | 46.86 tok/s |

Plow is 2.24× slower at median TTFT and 1.37× slower at median TPOT. Its
output throughput is 28.2% lower. These are fresh three-fold, order-alternated
served deficits on the same vLLM 0.28 client and BF16-KV contract. The active
priorities remain prefill KDA/MoE/MLA architecture and decode GEMV, MoE, and
boundary amortization; endpoint or benchmark-driver changes cannot close them.

## Historical MI325X B1 baseline

| Effective context | TPOT | Decode tok/s |
|---:|---:|---:|
| short/C1 | 53.4 ms | 18.7 |
| 8K | 56.33 ms | 17.8 |
| 16K | 57.81 ms | 17.3 |
| 32K | 61.34 ms | 16.3 |
| 64K | 68.37 ms | 14.6 |
| 128K | 82.58 ms | 12.1 |

The 50 tok/s requirement means <=20 ms/token. It is not met at any validated context. The short
trace is dominated by ordinary BF16 GEMV (~22 ms), while the long-context slope is MLA KV scanning
(about 0.2196 us per context token). Host work is roughly 3.1 ms/token and is not the main limit.

## Adopted production changes

- KDA Conv3 suffix-row parallelization: 14.2% lower 8192-token TTFT with an
  exact full-model output checksum; decode and strided-row paths are unchanged.
- Narrow KDA norm-to-GEMV fusion: ~1.3–1.45 ms/token saved, full-logit exact.
- KDA gated-norm workgroup fit: served B1 48.232 ms/token in the strongest exact gate.
- KDA Conv/state double-bank arm: explicit opt-in; 50.140 ms/token, GSM8K 196/200, not default.
- B8 two-shot collective aggregation: ~3% aggregate throughput improvement.
- FP8 MLA prefill V2: large TTFT improvement at long contexts; decode TPOT unchanged.
- Fixed K3 `PLOW_K3_NS=64`: best validated long-context split; GF4/ns96 loses short context.

## Rejected or closed experiments

W8A16, dense MXFP4, GEMV grid/R-split, KDU unroll, early shared-expert ordering, generic row
banding, combine-XReduce fusion, grouped A8W4, W2 cache touch, grouped weight NT, dead-arm-only
interpreter specialization, control-queue resident interpreter, MM8/WALK B16, and GF12 MLA all
failed their full-model, numerical, resource, or performance gates. Do not repeat them without a
new implementation hypothesis.

## Open experiments

1. New short-context GEMV/body mechanism: the only credible path to remove multiple milliseconds
   from the fixed B1 body; it must change load/reduction or a producer/consumer boundary.
2. Runtime live-`kv_len` MLA split selection: preserve packet partial sizing, merge order, counters,
   and state while choosing split count from the actual context.
3. K3 TP8 `XReduceTwoShotGather`: standalone f64/bit-exact owner-index oracle, then 8K/32K/128K
   serving A/B. Generic row-banding is closed.
4. KDA recurrent-state residency for prefill: static and numerical gates pass; TP8 timing remains.
5. New FP8 MLA architecture: GF12 is rejected in its current form (54,848 B LDS, 113 spills,
   failed oracle); only a resource-safe rewrite should reopen it.
6. DSpark/speculative decoding: experimental only; corrected causal frontier, MLA merge map,
   recurrent-state journal, and full-model token/state gates are still required.

## Detailed evidence index

### Reproducibility, architecture, and runtime

- [`kimi-k3-README.md`](kimi-k3-README.md) — build, flags, serving, measurement rules.
- [`archive/k3/k3-batched-decode-design.md`](archive/k3/k3-batched-decode-design.md) — batch/state contract.
- [`archive/k3/k3-decode-counter-graph.md`](archive/k3/k3-decode-counter-graph.md) — packet/counter graph.
- [`archive/k3/k3-prefix-cache-design.md`](archive/k3/k3-prefix-cache-design.md) — recurrent-state prefix cache.
- [`archive/k3/k3-throughput-architecture-review.md`](archive/k3/k3-throughput-architecture-review.md) — runtime review.
- [`k3-tp-peer-slots.md`](k3-tp-peer-slots.md) — TP slot-D proposal.
- [`archive/k3/k3-narrow-gate-fusion.md`](archive/k3/k3-narrow-gate-fusion.md) — adopted fusion rationale.
- [`archive/k3/k3-prefill-attribution.md`](archive/k3/k3-prefill-attribution.md) — prefill roofline attribution.
- [`archive/k3/k3-speculative-decoding.md`](archive/k3/k3-speculative-decoding.md) — prior speculation decision.

### MI325X B1 and kernel screens

See `kimi-k3-mi325x-b1-*.md` for the complete B1 screen set: 128K serving, ablation census,
collectives, dense quantization, GEMV cohorts/grids, KDA state/conv/norm, interpreter, MLA KDU,
MoE grid, router blocks, shared scheduling, and W8A16. The latest MLA split records are
[`archive/k3/kimi-k3-mi325x-mla-nsplit.md`](archive/k3/kimi-k3-mi325x-mla-nsplit.md) and
[`archive/k3/kimi-k3-mi325x-mla-nsplit-fine-sweep.md`](archive/k3/kimi-k3-mi325x-mla-nsplit-fine-sweep.md).

### Batched and long-context serving

- `archive/k3/kimi-k3-mi325x-b8-serve.md`, `archive/k3/kimi-k3-mi325x-b8-long-context.md`
- `archive/k3/kimi-k3-mi325x-b16-mm8-walk.md`, `archive/k3/kimi-k3-mi325x-b32-serve.md`
- `archive/k3/kimi-k3-mi325x-ladder-130tps.md`, `archive/k3/kimi-k3-mi325x-lowrung-b1.md`
- `archive/k3/kimi-k3-mi325x-fp8-mla-v2.md`, `archive/k3/kimi-k3-mi325x-state-clear.md`
- `archive/k3/kimi-k3-mi325x-prefill-experiments.md`, `archive/k3/kimi-k3-mi325x-prefill-interleave.md`

### Staged MI325X evidence

`archive/k3/kimi-k3-mi325x-stage4.md` through `archive/k3/kimi-k3-mi325x-stage7.md` remain historical stage records;
they are not independent current baselines. `archive/k3/kimi-k3-mi325x-kernel-audit.md`,
`archive/k3/kimi-k3-mi325x-kernel-capacity.md`, `archive/k3/kimi-k3-mi325x-cache-policy.md`, and
`archive/k3/kimi-k3-mi325x-decode-xr-agg.md` contain the corresponding static/resource audits.

### DSpark

- `kimi-k3-mi325x-dspark-mla.md`
- `kimi-k3-mi325x-dspark-block.md`
- `kimi-k3-mi325x-dspark-verifier.md`

These are explicitly experimental and excluded from the production merge.

### Historical design/reference records

`archive/k3/k3-gsm8k.md`, `archive/k3/k3-hier2-ceiling.md`, `archive/k3/k3-serving-speed.md`, `archive/k3/k3-75tps-program.md`,
`archive/k3/kimi-k3-kernel-gap.md`, `archive/k3/kimi-k3-atom-reference.md`, and `archive/k3/coldstart-amd-k3-tp8.md` are retained
only when their historical context is needed; their numbers must not be compared with the current
MI355X campaign.

## Merge policy

Production merge excludes DSpark source and verifier edits. Use the clean non-DSpark cutoff
`9d179ee9` plus reviewed merge-blocker cleanup; do not merge the current DSpark working tree.
