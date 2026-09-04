# Kimi-K3 Plow baseline on MI355X

Measured 2026-09-04 on one 8×AMD Instinct MI355X node. The current C1 result is
the predeclared median-of-folds reading from the two alternating exact cells in
campaign `k3-showdown-c1-stack2-20260904` (the stack-2 campaign: prefill and
decode promotions on top of the decode stack). All older C1 publication and
candidate rows are superseded.

## Contract

- TP8, native MXFP4 weights, BF16 KV, MTP/speculation and prefix caching off.
- Raw `/v1/completions`, random exact 8192-token inputs and exact 1024-token
  outputs, greedy sampling, `--ignore-eos`, infinite request rate, C1.
- Each fold used one discarded warmup and ten measured requests. All 20
  measured requests completed, with exactly 81,920 input and 10,240 output
  tokens per fold. Two rounds were run (the third was cut by request).
- Current final-default packet/runtime: global-queue prefill and decode with
  ASAP window order, TP prefill segment-major dispatch, MLA PF v2, strict-order
  KDA intra specialist, RS-U2, reusable sorted-A4 stage-1 scratch, reachable
  phase objects, the standalone grouped-MoE decode route selected by the
  measured TuneDB rule, the GLU GEMV K=7168 UN=7 rung, HSA queue depth 4096
  with exact per-segment AQL chain reservation, and the four routes promoted
  on 2026-09-04: register-resident KDA carry (`PLOW_KDA_CARRY_REGSTATE`),
  f32-mix AttnRes with vLLM's separate output-norm epsilon
  (`PLOW_ATTNRES_F32MIX`), tagged one-shot decode XReduce (`PLOW_XR_TAGGED`),
  and the split-tile decode MLA merge-fold (`PLOW_MLA_FOLD_DVT=8`).
  Materialized MLA and packed prefill were off.
- Production timing used `--amd-tp-no-audit`. Exactness: the stack with the
  f32-mix AttnRes opted out reproduces the pre-stack 8192→256 output IDs
  bit-for-bit (`fnv1a64:b7682a38c151ac99`, two folds, TTFT 1144.2 ms / TPOT
  25.25 ms); the f32-mix AttnRes changes numerics by design (C3 contract,
  seam relL2 2.8e-3 → 2e-7 against a CPU port of vLLM's residual/norm) and was
  gated on GSM8K (n=200: 122 correct vs 124 for the BF16-seam control). Every
  measured fold generated exactly 10,240 tokens.
- Exact packet verification: Lean ordering certificate for every program,
  oracle run; 7,650/7,650 TuneDB selections were measured. Packet/object
  pairing hash `0x9c1fcd45eac7022c`, packet `cd4e349f…`, 62 paired objects,
  source head `dd8ff8ed` (runtime/perf head `5e87b84`).
- Artifact-set digest:
  `f8864b37257348c37e6a9f1db17b7f16d17205ea7735ca37d7fec05812b60904`.

## Two-fold result

| metric | Plow | vLLM 0.28 (same campaign) | gap |
|---|---:|---:|---:|
| duration | 269.38 s | 219.25 s | Plow 1.23× longer |
| output throughput | 38.01 tok/s | 46.70 tok/s | vLLM 1.23× higher |
| total token throughput | 342.12 tok/s | 420.35 tok/s | vLLM 1.23× higher |
| median TTFT | 1113.26 ms | 566.36 ms | Plow 1.97× longer |
| P90 / P99 TTFT | 1115.68 / 1119.70 ms | 567.61 / 568.32 ms | Plow 1.97× / 1.97× longer |
| median TPOT | 25.25 ms | 20.88 ms | Plow 1.21× longer |
| median / P90 / P99 ITL | 25.23 / 25.31 / 25.38 ms | 20.88 / 20.96 / 21.06 ms | Plow 1.21× / 1.21× / 1.21× longer |
| median E2E | 26.936 s | 21.925 s | Plow 1.23× longer |

This is endpoint performance, not an isolated-kernel measurement. Against the
previous Plow publication (1284.84 ms / 27.54 ms / 34.76 tok/s) the stack-2
campaign moves TTFT −171.6 ms (−13.4%), TPOT −2.29 ms/token (−8.3%), and
output throughput +9.3%; the vLLM cells reproduced the published vLLM baseline
(567.74 / 20.81 / 46.86) within 2 ms, and that JSON is unchanged. Plow still
trails vLLM on every metric; the remaining gap is 547 ms TTFT and 4.4 ms/token.

## Exact fold provenance

| fold | mean / median / P99 TTFT (ms) | mean / median / P99 TPOT (ms) | output tok/s | mean / median / P99 E2E (ms) |
|---|---:|---:|---:|---:|
| showdown-1 | 1114.44 / 1114.53 / 1124.18 | 25.23 / 25.23 / 25.26 | 38.03 | 26924.14 / 26921.88 / 26947.17 |
| showdown-2 | 1112.13 / 1111.98 / 1115.23 | 25.26 / 25.26 / 25.27 | 37.99 | 26950.98 / 26950.73 / 26958.86 |

The JSON preserves every `cells.tsv` field, source log basename, artifact
digest, and config/tokenizer hashes. Headline values, including P90 and
duration, are medians of the two fold statistics. The served model id is
`kimi-k3` (the packet slug); the harness accepts it via `MODEL_ID`.

Known like-for-like caveat: this campaign and the superseded ones log
"KDA key-factor segments have no paired objects — using interpreter fallback"
at load; the key-factor family has no lean object yet in any bundle.

## Superseded results

The 2026-09-04 `k3-showdown-c1-stack-20260904d` row (1284.84 ms median TTFT,
27.54 ms median TPOT, 34.76 output tok/s; decode stack only), the
`k3-showdown-c1-fe871e6-final` row (1271.86 / 28.53 / 33.63), and the
2026-09-03 three-fold row (3762.81 / 63.17 / 14.97) are superseded by this
campaign.

Post-publication engine gate (not yet a served cell): promoting `PLOW_MOE_ALIGN_PAR`
and the GemmWide c8 tile at `8192x1536x7168` on this packet measured 8192→1 TTFT
1072.3 / 1072.0 / 1072.0 ms vs 1095.1 / 1095.0 / 1094.7 ms (−22.9 ms, three
alternating folds), TPOT 25.28 vs 25.30 ms, with identical 1- and 256-token
output checksums; both are default-on from commit "amd: promote align-parallel
MoE and the c8 tile shape". The next served cell should read ~1090 ms TTFT.

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
