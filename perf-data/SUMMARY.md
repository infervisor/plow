# Performance campaign summary

This is the consolidated performance record retained for the Plow repository as of
2026-09-06. Detailed Markdown reports and JSON snapshots were removed before the
`perf/h100-native` merge freeze. The CSV files listed below retain the measurements
needed to reproduce the main comparisons.

The complete pre-consolidation artifact set remains available in Git at commit
`40da0e6`. Historical source comments that name an old report refer to that snapshot.

## Scope and claim status

- Target: one 80 GB H100 SXM, CUDA `sm_90a`, TP1.
- Models brought up: Gemma 4 12B, Gemma 4 31B, and Qwen3.8 27B.
- All three models served BF16 and Plow W8A16 smoke workloads.
- Plow does not yet beat vLLM across serving metrics. No kernel or serving default is
  promoted by this checkpoint.
- Serving promotion requires numerical/reset gates followed by the 15-cell grid at
  input 1K/4K/8K/16K/32K and concurrency 1/4/16, using 32 measured and 16 warmup
  requests in alternating engine order.

The current runtime uses packet metadata for optional segment routes. Compiler
selection emits the role; PlowRT validates the packet contract and executes it. Model
names are not runtime tuning controls.

## Serving status

Gemma 4 31B Plow measurements are provisional screens with four requests at C1. The
vLLM values use the matched BF16 serving configuration.

| Input | Plow TTFT ms | vLLM TTFT ms | Plow TPOT ms | vLLM TPOT ms | Plow output tok/s | vLLM output tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 186.046 | 102.687 | 28.689 | 23.632 | 34.487 | 42.040 |
| 4K | 876.161 | 425.030 | 28.959 | 23.732 | 32.665 | 40.790 |
| 8K | 2026.307 | 906.979 | 29.310 | 23.837 | 30.111 | 39.120 |
| 32K | 13899.355 | 4565.956 | 31.469 | 24.417 | 17.078 | 30.041 |

No Plow Gemma 31B C4/C16 promotion cell, repeated reverse-order grid, or complete p99
serving comparison exists yet. The current result therefore establishes functionality
and the remaining gap, not parity.

Gemma 4 12B packet-rung decode improved the old widest-rung path by 3.62–3.85x across
1K–32K C1 screens, reaching 12.65–13.73 ms TPOT. It still trails matched vLLM and does
not establish a concurrency win. Early real-world C1/C4/C16 results also left Plow
slower in every completed cell.

The repeated vLLM BF16 reference covers all three models at 128/1K/4K and C1/C4/C16.
At 1K, mean TPOT across two repeats was:

| Model | C1 ms | C4 ms | C16 ms |
|---|---:|---:|---:|
| Gemma 4 12B | 10.619 | 11.177 | 13.749 |
| Gemma 4 31B | 23.691 | 25.364 | 30.212 |
| Qwen3.8 27B | 20.043 | 21.106 | 24.278 |

## Kernel findings

- Sustained HBM ceiling on the SXM system is 3.154 TB/s. The best one-CTA-per-SM
  synthetic read reached 2.752 TB/s. Old measurements from a power-capped H100 NVL
  are not used to choose occupancy.
- Native attention remains the main gap. Across the retained comparator, Gemma 31B
  full prefill is 1.98–2.47x the library latency, sliding prefill 4.30–5.11x, full
  decode 1.91–5.59x, and sliding decode 1.82–3.16x. The GQA-reuse full-decode
  candidate narrows its measured Gemma 31B gap to 1.38–3.48x but still wins no
  library comparison cell.
- The HD512 WG64 prefill prototype was 2.98–3.45x faster than the deployed standalone
  body but remained 1.61–3.48x behind FA4. Its integrated full-model logit gate failed
  and the object spilled, so it is not promotable.
- The four-stage E3 split-K skinny GEMM is the strongest batching candidate. Median
  speedup over the scalar kernel is 1.47x/1.97x/3.49x/5.89x at M=4/8/16/32. Its
  median latency is 1.07x/1.08x/1.12x/1.25x cuBLASLt respectively. Packet integration,
  fused GLU/QKV coverage, scratch lifetime, and serving qualification remain open.
- The FP8 producer coordinate cursor is bit-exact and improves measured projection
  kernels by roughly 10–15%, with a 6–7% GDN block-prefill improvement. It remains a
  packet/compiler choice pending full-model gates.
- Cross-packet prefetch passed its numerical checks but improved warm decode by only
  0.78–0.96%. This is too small to prioritize ahead of attention and batched GEMM.

## Next measured work

1. Qualify tensor-core GQA-packed decode attention at B1/B4/B16 and 1K–32K.
2. Integrate E3 projection routes block-by-block for Gemma 31B, including q/k/down/o,
   and require complete block numerical and memory gates before serving runs.
3. Fix HD512 and sliding HD256 prefill attention using real packet bucket geometry.
4. Run the full Plow C1/C4/C16 serving grid only after block-level gains pass.
5. Evaluate cross-request prefill batching and mixed prefill/decode scheduling after
   kernel parity, including straggler and p99 ITL measurements.

## Method changes adopted

- Existing harness selection is mandatory before launching or writing a new
  runner. Missing functionality should extend the closest reusable harness.
- Broad tuning runs at single-block or minimal mixed-block scope. Standalone
  probes prune; only 2–3 block finalists reach whole-model steps; serving is
  reserved for promotion.
- Experiment priority uses measured block op share times candidate improvement.
  Small local gains in a kernel still several times behind its library peer do
  not outrank algorithmic attention and batched-GEMM work.
- Block finalist grids cover B=1/4/16, T=1K/4K/8K/16K/32K, and boundary-adjacent
  points. Context/rung choices belong in compiler-emitted packet metadata.
- Paired runs reverse order, evict L2 symmetrically, log SM clock/power, and bind
  packet/object digests plus register/spill evidence to every result.
- Raw JSON/JSONL and logs stay on campaign storage. The repository retains this
  summary and only decision-critical CSV measurements.

## Retained raw CSV files

| File | Purpose |
|---|---|
| `attention-library-h100-20260905.csv` | Native attention vs FA3/FA4 matrix |
| `attention-gqa-reuse-full-h100-20260905.csv` | Full-decode GQA reuse comparison |
| `e0-hbm-ceiling-sxm-h100-20260905.csv` | SXM HBM occupancy ceiling |
| `e3-grid132-m4-h100-20260906.csv` | E3 M4 fixed-grid measurements |
| `e3-grid132-m8-h100-20260906.csv` | E3 M8 fixed-grid measurements |
| `e3-grid132-m16-h100-20260906.csv` | E3 M16 fixed-grid measurements |
| `e3-zg0-timings-h100-20260906.csv` | E3 scalar, split-K, and cuBLASLt timings |
| `fp8-producer-cursor-h100-20260905.csv` | FP8 projection cursor A/B |
| `fp8-producer-cursor-blocks-h100-20260905.csv` | FP8 block-prefill cursor A/B |
| `gemma12-packet-rung-serving-h100-20260905.csv` | Gemma 12B packet-rung C1 screen |
| `gemma31-packet-rung-serving-h100-20260905.csv` | Gemma 31B C1 screen |
| `vllm-028-h100-fp8-baseline.csv` | vLLM matched FP8 reference |
| `vllm-028-h100-gemma31-bf16-mnbt2048-32k-c1.csv` | Gemma 31B vLLM 32K C1 reference |
| `vllm-028-h100-three-model-baseline-repeats.csv` | Repeated BF16 reference matrix |

CSV measurements are evidence, not configuration. Benchmark tools remain under
`perf-data/tools/`; new campaign output should be written outside the repository and
only consolidated results needed for review should be committed.
