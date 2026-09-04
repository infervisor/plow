# Kimi-K3 Plow baseline on MI355X

Measured 2026-09-04 on one 8×AMD Instinct MI355X node. The current C1 result is
the predeclared median-of-folds reading from the two alternating exact cells in
campaign `k3-showdown-c1-stack4-20260904` (the full 2026-09-04 stack: prefill
seams and decode promotions on top of the earlier stacks). All older C1
publication and candidate rows are superseded.

## Contract

- TP8, native MXFP4 weights, BF16 KV, MTP/speculation and prefix caching off.
- Raw `/v1/completions`, random exact 8192-token inputs and exact 1024-token
  outputs, greedy sampling, `--ignore-eos`, infinite request rate, C1.
- Each fold used one discarded warmup and ten measured requests. All 20
  measured requests completed, with exactly 81,920 input and 10,240 output
  tokens per fold. The harness ran three alternating rounds; it was stopped
  after round 2 by request, so two Plow and two vLLM folds are recorded.
- Current final-default packet/runtime, all default-on with no flags set:
  sequence-parallel TP seams, register-resident KDA carry (interpreter fallback
  below 512 rows), f32-mix AttnRes, align-parallel MoE, GemmWide c8 tile at
  `8192x1536x7168`, ASAP global-queue window order, standalone grouped-MoE
  decode route (measured TuneDB rule), tagged one-shot decode XReduce,
  split-tile decode MLA merge-fold, wave-parallel router top-k select, GLU GEMV
  K=7168 UN=7 rung, MLA PF v2, strict-order KDA intra specialist, RS-U2, sorted-A4
  stage-1 scratch, HSA queue 4096 with exact AQL chain reservation. Materialized
  MLA and packed prefill were off.
- Production timing used `--amd-tp-no-audit`. Exactness: every promoted route
  except the f32-mix AttnRes reproduced the pre-stack 8192→256 output IDs
  bit-for-bit in its own gate (`fnv1a64:b7682a38c151ac99` for the exact arm,
  `fnv1a64:71a28c1449921c95` with the f32-mix seam), including an audited run of
  the seams packet; the f32-mix AttnRes is a deliberate numerics change gated on
  GSM8K (n=200: 122 correct vs 124 for the BF16-seam control).
- Exact packet verification: Lean ordering certificate for every program,
  oracle run; 7,650/7,650 TuneDB selections were measured. Packet/object
  pairing hash `0x6892b68e52f0e447`, packet `c896deb6…`, 62 paired objects,
  source head `4886f7e` (runtime/perf head `3ee9ac5`).
- Artifact-set digest:
  `d3cffd6131a493bce320dea12f520625f55d2dcb9cecb918894731b95537f25b`.

## Two-fold result

| metric | Plow | vLLM 0.28 (same campaign) | gap |
|---|---:|---:|---:|
| duration | 265.65 s | 219.23 s | Plow 1.21× longer |
| output throughput | 38.55 tok/s | 46.71 tok/s | vLLM 1.21× higher |
| total token throughput | 346.93 tok/s | 420.39 tok/s | vLLM 1.21× higher |
| median TTFT | 985.22 ms | 567.30 ms | Plow 1.74× longer |
| P90 / P99 TTFT | 988.69 / 991.82 ms | 571.18 / 573.37 ms | Plow 1.73× / 1.73× longer |
| median TPOT | 25.00 ms | 20.88 ms | Plow 1.20× longer |
| median / P90 / P99 ITL | 25.00 / 25.05 / 25.12 ms | 20.88 / 20.95 / 21.06 ms | Plow 1.20× / 1.20× / 1.19× longer |
| median E2E | 26.564 s | 21.925 s | Plow 1.21× longer |

This is endpoint performance, not an isolated-kernel measurement. Against the
morning publication (1271.86 ms / 28.53 ms / 33.63 tok/s) the day's stack moves
TTFT −287 ms (-22.5%), TPOT −3.53 ms/token
(-12.4%), and output throughput +14.6%; the vLLM cells
reproduced the published vLLM baseline (567.74 / 20.81 / 46.86) within 2 ms and
that JSON is unchanged. Plow still trails vLLM on every metric; the remaining
gap is 418 ms TTFT and 4.12 ms/token.

## Exact fold provenance

| fold | mean / median / P99 TTFT (ms) | mean / median / P99 TPOT (ms) | output tok/s | mean / median / P99 E2E (ms) |
|---|---:|---:|---:|---:|
| showdown-1 | 982.37 / 981.92 / 984.55 | 25.01 / 25.00 / 25.01 | 38.55 | 26563.40 / 26562.52 / 26570.00 |
| showdown-2 | 987.59 / 988.53 / 999.08 | 25.00 / 25.00 / 25.02 | 38.55 | 26565.87 / 26565.97 / 26580.55 |

The JSON preserves every `cells.tsv` field, source log basename, artifact
digest, and config/tokenizer hashes. Headline values, including P90 and
duration, are medians of the two fold statistics. The served model id is
`kimi-k3` (the packet slug); the harness accepts it via `MODEL_ID`.

Known like-for-like caveat: every campaign today logs "KDA key-factor segments
have no paired objects — using interpreter fallback" at load; the key-factor
family has no lean object yet in any bundle.

## Superseded results

The 2026-09-04 `k3-showdown-c1-stack2-20260904` row (1113.26 ms median TTFT,
25.25 ms median TPOT, 38.01 output tok/s), the `k3-showdown-c1-stack-20260904d`
row (1284.84 / 27.54 / 34.76), the `k3-showdown-c1-fe871e6-final` row
(1271.86 / 28.53 / 33.63), and the 2026-09-03 three-fold row (3762.81 / 63.17 /
14.97) are superseded by this campaign.

Post-publication engine gate (not yet a served cell): the segment-relative ASAP
window order (default on, commit "packet: segment-relative ASAP window order by
default") measured 955.2 / 956.2 ms TTFT and 24.23 / 24.31 ms TPOT against
962.4 / 25.06 on this packet's 8192→256 gate, exact. The next served cell should
read ~980 ms TTFT and ~24.3 ms TPOT.

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
