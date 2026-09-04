# Kimi-K3 Plow baseline on MI355X

Measured 2026-09-04 on one 8×AMD Instinct MI355X node. The current C1 result is
the predeclared median-of-folds reading from the three alternating exact cells in
campaign `k3-showdown-c1-stack-20260904d` (the decode-stack campaign). All older
C1 publication and candidate rows are superseded.

## Contract

- TP8, native MXFP4 weights, BF16 KV, MTP/speculation and prefix caching off.
- Raw `/v1/completions`, random exact 8192-token inputs and exact 1024-token
  outputs, greedy sampling, `--ignore-eos`, infinite request rate, C1.
- Each fold used one discarded warmup and ten measured requests. All 30
  measured requests completed, with exactly 81,920 input and 10,240 output
  tokens per fold.
- Current final-default packet/runtime: global-queue prefill and decode with
  ASAP window order, TP prefill segment-major dispatch, MLA PF v2, strict-order
  KDA intra specialist, RS-U2, reusable sorted-A4 stage-1 scratch, reachable
  phase objects, the standalone grouped-MoE decode route selected by the
  measured TuneDB rule (`moe_decode_measurement.jsonl`), the GLU GEMV K=7168
  UN=7 rung, and HSA queue depth 4096 with exact per-segment AQL chain
  reservation. Materialized MLA, f32-mix AttnRes, KDA carry register-state, and
  packed prefill were off.
- Production timing used `--amd-tp-no-audit`; the separate exact 8192→256
  gates of every promoted item matched all 256 output IDs against the previous
  default (`fnv1a64:337f0f290d5ae157`) with TP agreement. Every measured fold
  generated exactly 10,240 tokens.
- Exact packet verification: Lean ordering certificate for every program,
  oracle run; 7,650/7,650 TuneDB selections were measured. Packet/object
  pairing hash `0x0a3297821329ae02`, packet `c2040cf8…`, 119 paired objects,
  source head `4258e3d` (runtime/perf head `0f66fda`).
- Artifact-set digest:
  `78c43bae089c4c9ddc1871276ab3f33ccd4e51c97e255c1d7a4b31163a5c473c`.

## Three-fold result

| metric | Plow | vLLM 0.28 (same campaign) | gap |
|---|---:|---:|---:|
| duration | 294.59 s | 219.40 s | Plow 1.34× longer |
| output throughput | 34.76 tok/s | 46.67 tok/s | vLLM 1.34× higher |
| total token throughput | 312.84 tok/s | 420.06 tok/s | vLLM 1.34× higher |
| median TTFT | 1284.84 ms | 566.02 ms | Plow 2.27× longer |
| P90 / P99 TTFT | 1295.47 / 1299.29 ms | 568.29 / 569.30 ms | Plow 2.28× / 2.28× longer |
| median TPOT | 27.54 ms | 20.85 ms | Plow 1.32× longer |
| median / P90 / P99 ITL | 27.54 / 27.57 / 27.61 ms | 20.85 / 20.94 / 21.03 ms | Plow 1.32× / 1.32× / 1.31× longer |
| median E2E | 29.459 s | 21.901 s | Plow 1.35× longer |

This is endpoint performance, not an isolated-kernel measurement. Against the
previous Plow publication (1271.86 ms / 28.53 ms / 33.63 tok/s) the stack moves
decode −0.99 ms/token (−3.5%) and output throughput +3.4%; TTFT is within
+13 ms (+1.0%), consistent with the −6 ms ASAP gate result plus fold noise (the
promoted items were decode-side). The vLLM cells reproduced the published vLLM
baseline (567.74 / 20.81 / 46.86) within 2 ms; that JSON is unchanged.

## Exact fold provenance

| fold | mean / median / P99 TTFT (ms) | mean / median / P99 TPOT (ms) | output tok/s | mean / median / P99 E2E (ms) |
|---|---:|---:|---:|---:|
| showdown-1 | 1286.16 / 1284.84 / 1302.77 | 27.43 / 27.43 / 27.44 | 34.89 | 29347.03 / 29348.82 / 29369.73 |
| showdown-2 | 1284.25 / 1282.30 / 1299.29 | 27.55 / 27.55 / 27.57 | 34.74 | 29471.83 / 29470.69 / 29488.47 |
| showdown-3 | 1288.28 / 1290.02 / 1298.87 | 27.54 / 27.54 / 27.54 | 34.76 | 29458.43 / 29458.88 / 29472.53 |

The JSON preserves every `cells.tsv` field, source log basename, artifact
digest, and config/tokenizer hashes. Headline values, including P90 and
duration, are medians of the three fold statistics. The served model id is
`kimi-k3` (the packet slug); the harness accepts it via `MODEL_ID`.

Known like-for-like caveat: both this campaign and the superseded one log
"KDA key-factor segments have no paired objects — using interpreter fallback"
at load; the key-factor family has no lean object yet in either bundle.

## Superseded result

The 2026-09-04 `k3-showdown-c1-fe871e6-final` publication row (1271.86 ms
median TTFT, 28.53 ms median TPOT, 33.63 output tok/s) and the 2026-09-03
three-fold row (3762.81 ms, 63.17 ms, 14.97 tok/s) are superseded by this
campaign.

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
