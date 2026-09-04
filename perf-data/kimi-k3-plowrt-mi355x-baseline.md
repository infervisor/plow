# Kimi-K3 Plow baseline on MI355X

Measured 2026-09-04 on one 8×AMD Instinct MI355X node. The current C1 result is
the predeclared median-of-folds reading from the three alternating exact cells in
campaign `k3-showdown-c1-fe871e6-final`. All older C1 publication and candidate
rows are superseded.

## Contract

- TP8, native MXFP4 weights, BF16 KV, MTP/speculation and prefix caching off.
- Raw `/v1/completions`, random exact 8192-token inputs and exact 1024-token
  outputs, greedy sampling, `--ignore-eos`, infinite request rate, C1.
- Each fold used one discarded warmup and ten measured requests. All 30
  measured requests completed, with exactly 81,920 input and 10,240 output
  tokens per fold.
- Current final-default packet/runtime: global-queue prefill and decode, TP
  prefill segment-major dispatch, MLA PF v2, strict-order KDA intra specialist,
  RS-U2, reusable sorted-A4 stage-1 scratch, and reachable phase objects.
  Materialized MLA, grouped-MoE opt-in, and packed prefill were off.
- Production timing used `--amd-tp-no-audit`; the separate exact 8192→256 gate
  matched all 256 output IDs and TP agreement. Every measured fold generated
  exactly 10,240 tokens.
- Exact packet verification: Lean D 7/7, Lean G 7/7, oracle run; 7,650/7,650
  TuneDB selections were measured. Packet/object pairing hash
  `0x1df8ef184df9a71c`.
- Artifact-set digest:
  `9333f7c51c29e22e3c031af78268cfc54763b0aded7eb3f50b7d7b46f064e79a`.

## Three-fold result

| metric | Plow | vLLM 0.28 | gap |
|---|---:|---:|---:|
| duration | 304.52 s | 218.53 s | Plow 1.39× longer |
| output throughput | 33.63 tok/s | 46.86 tok/s | vLLM 1.39× higher |
| total token throughput | 302.64 tok/s | 421.72 tok/s | vLLM 1.39× higher |
| median TTFT | 1271.86 ms | 567.74 ms | Plow 2.24× longer |
| P90 / P99 TTFT | 1275.35 / 1276.59 ms | 570.08 / 570.58 ms | Plow 2.24× / 2.24× longer |
| median TPOT | 28.53 ms | 20.81 ms | Plow 1.37× longer |
| median / P90 / P99 ITL | 28.53 / 28.56 / 28.60 ms | 20.81 / 20.88 / 20.97 ms | Plow 1.37× / 1.37× / 1.36× longer |
| median E2E | 30.451 s | 21.852 s | Plow 1.39× longer |

This is endpoint performance, not an isolated-kernel measurement.

## Exact fold provenance

| fold | mean / median / P99 TTFT (ms) | mean / median / P99 TPOT (ms) | output tok/s | mean / median / P99 E2E (ms) |
|---|---:|---:|---:|---:|
| showdown-1 | 1274.23 / 1274.82 / 1280.82 | 28.54 / 28.54 / 28.55 | 33.60 | 30473.49 / 30472.93 / 30485.21 |
| showdown-2 | 1269.72 / 1269.81 / 1271.90 | 28.53 / 28.53 / 28.53 | 33.63 | 30452.11 / 30451.08 / 30459.02 |
| showdown-3 | 1272.03 / 1271.86 / 1276.59 | 28.51 / 28.51 / 28.52 | 33.64 | 30437.22 / 30435.27 / 30448.70 |

The JSON preserves every `cells.tsv` field, source log basename, artifact
digest, and config/tokenizer hashes. Headline values are medians of the three
fold statistics; P90 and duration values come
from the three raw client logs.

## Superseded result

The 2026-09-03 three-fold publication row (3762.81 ms median TTFT, 63.17 ms
median TPOT, 14.97 output tok/s) is superseded by the final-default campaign.
The stale short-sample `c1-current` candidate file was removed rather than
being relabeled as publication evidence.

Raw data: `perf-data/kimi-k3-plowrt-mi355x-c1.json`.
Comparator: `perf-data/kimi-k3-vllm-mi355x-c1.json`.

### FP8-KV ceiling (lossy, not apples-to-apples)

The clean 49-segment precision control isolates KV storage: BF16 measured
2931.36 ms median TTFT / 55.40 ms mean TPOT / 17.18 output tok/s; FP8 measured
2966.55 ms / 49.61 ms / 19.1 tok/s. FP8 therefore improves decode about 10.5%
without improving TTFT.

Combining FP8 KV with the opt-in shuffled MXFP4 stage-2 object produces the
current ceiling: 2204.33 ms median TTFT, 49.40 ms mean TPOT, 49.41/49.71 ms
median/P99 ITL, and 19.41 output tok/s. Relative to the FP8 49-segment control,
stage 2 improves TTFT 25.7% while leaving decode effectively unchanged. It
still trails vLLM BF16 by 3.88x TTFT, 2.37x TPOT, and 2.41x output throughput.

FP8 KV is deliberately not promoted here: it is lossy, greedy tokens are known
to diverge from the BF16 path, and the pinned vLLM comparator uses BF16 KV.
Exact ceiling provenance is in
`perf-data/kimi-k3-plowrt-mi355x-c1-fp8kv-ceiling.json`.

## Post-baseline gfx950 KDA-scan promotion gate

The model-independent BT64/BC16 KDA prefill scan passed its first complete
production-engine 8192-token gate after this baseline was recorded. Serial and
scan packets used the same TP8 checkpoint, BF16 KV, deterministic seed
`20260903`, one request, and 32 exact greedy output tokens. The parity report
confirmed identical 8192-token prompts and identical output IDs
(`[9618, 13]` repeated 16 times; checksum
`fnv1a64:5c73abff345f2d25`).

| path | TTFT | TPOT | E2E |
|---|---:|---:|---:|
| serial recurrence | 3714.31 ms | 67.91 ms | 5819.52 ms |
| BT64/BC16 scan | 2960.95 ms | 67.63 ms | 5057.63 ms |

The scan reduces TTFT by **20.28%** (1.254x) and E2E by 13.09%; decode is
neutral as expected. Packet checksums were `fnv1a64:a0a363433acc2fee`
(serial) and `fnv1a64:c577f59d5c2c1133` (scan). Checkpoint layout checksum was
`fnv1a64-layout:bf5f9b877998972d` for both.

A subsequent 8192-to-1024 C1 run completed one warmup and all three measured
requests with zero failures and complete all-rank counter audits:

| metric | scan candidate | pinned vLLM 0.28 | remaining gap |
|---|---:|---:|---:|
| median TTFT | 2952.02 ms | 568.35 ms | Plow 5.19x longer |
| median TPOT | 67.43 ms | 20.86 ms | Plow 3.23x longer |
| output throughput | 14.22 tok/s | 46.73 tok/s | vLLM 3.29x higher |

The candidate generated 3,072/3,072 output tokens. Its short sample is a
promotion result, not a replacement for the 30-request three-fold baseline;
repeat that release cell after the remaining MoE and segment-object work.
