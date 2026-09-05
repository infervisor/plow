# H100 native prefill progress, 2026-09-05

Source: `48ab142` on `perf/h100-native`. Existing kernel comparator and
`block_run` harness, B1, two runs with reversed candidate/control order.
Block measurements use synthetic hidden inputs and are not whole-model
quality or serving comparisons. The vLLM performance goal remains unmet.

## FP8 interpreter isolation

The packet-selected FP8 GEMM object reduces GDN block prefill time by
6.9–10.5% across contexts 1024/4096/8192/32768 relative to the broad object.
Attention block gains are 1.0–2.9%. Decode timing is approximately unchanged.
All 18 previously recorded fresh-bucket block checks were bit exact.

Raw per-run medians: `qwen-fp8-packet-role-blocks-h100-20260905.csv`.

## Producer coordinate cursor

`PGM90_FP8_TMA_ISSUE_CURSOR` remains default off. It avoids repeated producer
coordinate divisions without changing MMA math, pipeline depth or barriers.
All 120 kernel cases passed strict bit equality against the comparator,
including quantized activation bytes/scales, finite outputs and canaries.
All 12 paired fresh-bucket block checks passed bit equality at M1024/4096/8192.

For QKV N10240/K5120, kernel time fell 10.1–14.9% in both runs. Every measured
shape improved; the native kernel still trails the CUTLASS reference.

| Block | Context | Control prefill ms | Cursor prefill ms | Reduction |
|---|---:|---:|---:|---:|
| GDN layer 0 | 1024 | 3.087 | 2.900 | 6.1% |
| GDN layer 0 | 4096 | 10.741 | 10.019 | 6.7% |
| GDN layer 0 | 8192 | 20.833 | 19.328 | 7.2% |
| GDN layer 0 | 32768 | 83.216 | 77.176 | 7.3% |
| Attention layer 3 | 1024 | 5.512 | 5.330 | 3.3% |
| Attention layer 3 | 4096 | 34.670 | 33.912 | 2.2% |
| Attention layer 3 | 8192 | 107.341 | 106.131 | 1.1% |
| Attention layer 3 | 32768 | 1351.810 | 1346.454 | 0.4% |

Values are medians of the two per-run medians. Raw timings and evidence:
`fp8-producer-cursor-h100-20260905.csv`,
`fp8-producer-cursor-blocks-h100-20260905.csv`, and
`native-next-measurements-20260905-evidence.json`.

## Remaining gates

Long-context attention dominates the attention block. A packet-selected
HD256 attention object is the next controlled experiment: the broad
interpreter spills score accumulators and waits inside the score MMA loop;
the unchanged attention-only body avoids those spills and waits. GPU
correctness and timing are required before selecting it for serving.

The live-KV GPU gate stopped during flat-engine loading because the test
setup omitted the segmented prefill directory. It did not test live-KV
parity or memory savings. Corrected fixture qualification remains required.

Gemma12 vLLM 0.28 repeat 2 completed all 15 cells, 480/480 requests and exact
input/output lengths. Gemma31's separate BF16 MBNT2048 capacity probe is
running; its original MBNT8192 startup failure remains recorded.
