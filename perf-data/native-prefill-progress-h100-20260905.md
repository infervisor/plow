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

## Packet-selected attention role

`7e7728f` adds a default-off compiler-selected HD256 attention role. Runtime
selection follows packet metadata. The broad interpreter spills score
accumulators and waits inside the score MMA loop; the unchanged
attention-only body avoids those spills and waits. No tiling, split count,
precision or optional operand changes were combined with this experiment.

All nine coarse/isolated/role fresh-bucket checks passed bit equality.
The newly emitted default packet is byte-identical to the prior packet.

| Context | Broad prefill ms | Isolated broad ms | Attention role ms | Speedup vs broad |
|---:|---:|---:|---:|---:|
| 1024 | 5.486 | 5.552 | 4.255 | 1.29x |
| 4096 | 34.591 | 34.574 | 17.296 | 2.00x |
| 8192 | 107.017 | 107.212 | 39.157 | 2.73x |
| 32768 | 1349.088 | 1350.756 | 245.028 | 5.51x |

Values are medians of two per-run medians, with reversed ordering in run 2.
Decode is approximately unchanged. These runs retain the control FP8 GEMM
object; the producer cursor experiment is not combined here. Raw results:
`qwen-attention-role-block-h100-20260905.csv` and its evidence JSON.
Whole-model quality and matched serving comparisons remain required.

## KV correctness and capacity

The first live-KV gate stopped at loading because the fixture omitted the
segmented prefill directory. The corrected fixture passed full-logit bit
equality between live and flat allocation at 8192/32768/8193/8192 tokens,
including four decode steps and reset/repeat checks. Live allocation used
864,026,624 fewer bytes at load (824 MiB), with prefix reuse disabled.

The B16 flat packet passed two interleaved active slots, distinct 8K/16K
prompts, resets and decode continuation against isolated full logits.
This does not qualify B16 live allocation or serving throughput. Evidence:
`gemma12-live-kv-and-slot-parity-h100-20260905.json`.

## vLLM references

Gemma12 vLLM 0.28 repeat 2 completed all 15 cells, 480/480 requests and exact
input/output lengths. Gemma31's separate BF16 MBNT2048 capacity probe completed 32/32 exact-length
requests at 32K/C1: TTFT 4577.47 ms, TPOT 24.42 ms, output 30.02 tokens/s. Its
original MBNT8192 startup failure remains recorded. See the separate
Gemma31 MBNT2048 CSV and evidence JSON.
