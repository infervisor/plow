# Kimi-K3 Plow baseline on MI355X

Measured 2026-09-04 on one 8×AMD Instinct MI355X node. The current C1 result is
the reading from the single alternating exact round in
campaign `k3-showdown-c1-stack5-20260904` (the full 2026-09-04 stack with the
segment-relative window order and the combine-into-publish fold). All older C1
publication and candidate rows are superseded.

## Contract

- TP8, native MXFP4 weights, BF16 KV, MTP/speculation and prefix caching off.
- Raw `/v1/completions`, random exact 8192-token inputs and exact 1024-token
  outputs, greedy sampling, `--ignore-eos`, infinite request rate, C1.
- The fold used one discarded warmup and ten measured requests. All 10
  measured requests completed, with exactly 81,920 input and 10,240 output
  tokens. The harness ran alternating whole-server rounds; it was stopped after
  round 1 by request, so one Plow and one vLLM fold are recorded.
- Current final-default packet/runtime, all default-on with no flags set:
  sequence-parallel TP seams, register-resident KDA carry (interpreter fallback
  below 512 rows), f32-mix AttnRes, align-parallel MoE, GemmWide c8 tile at
  `8192x1536x7168`, ASAP global-queue window order, standalone grouped-MoE
  decode route (measured TuneDB rule), tagged one-shot decode XReduce,
  split-tile decode MLA merge-fold, wave-parallel router top-k select,
  segment-relative ASAP window order, MoE combine folded into the tagged
  publish, GLU GEMV
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
  pairing hash `0x49a0fe9bd3262c04`, packet `31808e16…`, 62 paired objects,
  source head `7e900b7` (runtime/perf head `e26daf3`).
- Artifact-set digest:
  `b30c286bed61d37f08f80739b09af15ee44cac5416f828a8af05541960b63a13`.

## Single-round result

| metric | Plow | vLLM 0.28 (same campaign) | gap |
|---|---:|---:|---:|
| duration | 257.33 s | 218.65 s | Plow 1.18× longer |
| output throughput | 39.79 tok/s | 46.83 tok/s | vLLM 1.18× higher |
| total token throughput | 358.14 tok/s | 421.50 tok/s | vLLM 1.18× higher |
| median TTFT | 978.55 ms | 569.36 ms | Plow 1.72× longer |
| P90 / P99 TTFT | 980.63 / 987.46 ms | 574.05 / 575.30 ms | Plow 1.71× / 1.72× longer |
| median TPOT | 24.20 ms | 20.82 ms | Plow 1.16× longer |
| median / P90 / P99 ITL | 24.19 / 24.25 / 24.35 ms | 20.81 / 20.91 / 21.02 ms | Plow 1.16× / 1.16× / 1.16× longer |
| median E2E | 25.732 s | 21.864 s | Plow 1.18× longer |

This is endpoint performance, not an isolated-kernel measurement. Against the
morning publication (1271.86 ms / 28.53 ms / 33.63 tok/s) the day's stack moves
TTFT −293 ms (-23.1%), TPOT −4.33 ms/token
(-15.2%), and output throughput +18.3%; the vLLM cells
reproduced the published vLLM baseline (567.74 / 20.81 / 46.86) within 2 ms and
that JSON is unchanged. Plow still trails vLLM on every metric; the remaining
gap is 409 ms TTFT and 3.38 ms/token.

## Exact fold provenance

| fold | mean / median / P99 TTFT (ms) | mean / median / P99 TPOT (ms) | output tok/s | mean / median / P99 E2E (ms) |
|---|---:|---:|---:|---:|
| showdown-1 | 977.98 / 978.55 / 987.46 | 24.20 / 24.20 / 24.21 | 39.79 | 25732.41 / 25731.74 / 25752.75 |

The JSON preserves every `cells.tsv` field, source log basename, artifact
digest, and config/tokenizer hashes. Headline values are the single
fold's statistics. The served model id is
`kimi-k3` (the packet slug); the harness accepts it via `MODEL_ID`.

Known like-for-like caveat: every campaign today logs "KDA key-factor segments
have no paired objects — using interpreter fallback" at load; the key-factor
family has no lean object yet in any bundle.

## Superseded results

The 2026-09-04 `k3-showdown-c1-stack4-20260904` row (985.22 ms median TTFT,
25.00 ms median TPOT, 38.55 output tok/s), the `k3-showdown-c1-stack2-20260904`
row (1113.26 / 25.25 / 38.01), the `k3-showdown-c1-stack-20260904d`
row (1284.84 / 27.54 / 34.76), the `k3-showdown-c1-fe871e6-final` row
(1271.86 / 28.53 / 33.63), and the 2026-09-03 three-fold row (3762.81 / 63.17 /
14.97) are superseded by this campaign.

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
