# Gemma-4 results summary (H100 unless noted)

Numbers are carried from the cited records, not re-derived. A final plow-vs-baseline claim needs
a strict `campaign.py report` ([agent-tools.md](../../agent-tools.md#final-performance-report-strict));
rows marked "grid" come from `llm_grid.sh` or earlier same-client grids and are not final.

| Model | Precision | Recipe | plowrt | Gate | vs baseline | Qualified |
|---|---|---|---|---|---|---|
| E4B | BF16 | [`recipes/infervisor/gemma-4-e4b/sm90a-h100-tp1.toml`](../../../../recipes/infervisor/gemma-4-e4b/sm90a-h100-tp1.toml) | main `fc0271e8` (repro) | `llm_logit_parity` PASS | grid only | no strict report |
| 12B | FP8 W8A8, FP8 per-token-head KV | [`recipes/infervisor/gemma-4-12b/sm90a-h100-tp1.toml`](../../../../recipes/infervisor/gemma-4-12b/sm90a-h100-tp1.toml) | `ee57f7b7` | `llm_fp32_ref` PASS | strict, 10 cells | **yes**, vs vLLM 0.28 FP8 KV |
| 12B, 26B-A4B | BF16 | `scripts/campaign/recipes/gemma4-{12b,26b-a4b}.h100.bf16-*.toml` | | | ledgers only | no |
| 31B | BF16 / FP8 | frozen release, no recipe | `3ca64e9` / `25fb3d7` | functional | none on H100 | no |

## Gemma-4 E4B (voice-agent LLM)

- Recipe: `recipes/infervisor/gemma-4-e4b/sm90a-h100-tp1.toml` (google/gemma-4-E4B-it `ee0ef602`).
  Build and gate steps: [h100-speech-recipes.md](../../../runtime/h100-speech-recipes.md).
  Mechanisms: [gemma4-e4b-h100.md](../../../runtime/gemma4-e4b-h100.md).
- Latest reproduction: main `fc0271e8`, packet `62408e49`, 2026-10-03, one H100 SXM, 2 repeats.
  Gate `llm_logit_parity` PASS: top1 0.9897, KL mean 7.4e-4, KL max 0.014 (gate top1 >= 0.98).
- 2026-10-09 per-rung routes (fd0fe175): cuBLASLt decode from 2 rows and generated prefill
  attention on the 1024+ rungs; gate top1 0.9819; voice bench c1 TTFT 21 -> 14.6 ms, c32 out
  tok/s +32%: [native-kernels-20261009.md](native-kernels-20261009.md) (grid only).
- Baseline: vLLM 0.28, same client, `llm_grid.sh` ISL 128 / OSL 512, greedy, per-cell seeds.
  No strict `campaign.py report` exists for E4B.

| Metric (grid) | plow recorded (round 2) | plow main `fc0271e8` | vLLM 0.28 |
|---|---:|---:|---:|
| out tok/s c64 | 9,169 | 9,029 [9,009-9,049] | 8,916 |
| out tok/s c128 | 14,662 | 14,471 [14,430-14,512] | 14,088 |
| TTFT p50 c64 | 51 ms | 75.3 ms [67.8-82.8] | 181 ms |
| TTFT p50 c128 | 80 ms | 91.0 ms [76.6-105.4] | 338 ms |
| c1 TPOT / tok/s | 5.69 ms / 175 | 175 tok/s | 5.70 ms / 174 |
| d64x1024 / d128x1024 tok/s | 6,139 / 8,078 | 5,548 / 6,999 | 5,866 / 7,664 |

- The recorded 51 ms TTFT was one favourable repeat (raw repeats 51-130 ms); pre-squash
  `dd1be445` measures 70.7 / 94.3 ms. At c64 main is about 10 ms higher, all in the second repeat,
  and the excess goes away with `PLOW_VMM_CACHE_MEMORY_UTILIZATION=0.05` (prefix-pool growth under
  the uncapped default). Not a confirmed regression.
- Open bug found in that bisect: E4B on main with prefix reuse disabled (`PLOW_VMM_PREFIX=0` or
  `PLOW_PREFIX_CACHE=0`) faults at c64 with `CUDA_ERROR_ILLEGAL_ADDRESS` in the decode pipeline
  (3 of 3 runs).
- Evidence: `/opt/dlami/nvme/lava-tts/repro-main/` (`res/`, `gates/gemma-4-e4b/`,
  `report/gates-score.log`, `report/summ.py`); regression bisect `/opt/dlami/nvme/lava-tts/bisect/`. Round-2 grid:
  [gemma4-e4b-h100.md](../../../runtime/gemma4-e4b-h100.md#round-2-sliding-attention-two-blocks-per-sm-ple-projection-on-cublaslt-served-config).

## Gemma-4 12B FP8

Campaign comparison with the qualified wins, strict tables and FP32 gate:
[gemma12b-fp8-20260930/comparison.md](../gemma12b-fp8-20260930/comparison.md). Every matched row:
[comparison.csv](../gemma12b-fp8-20260930/comparison.csv).

- Qualified: plowrt `ee57f7b7`, packet `5fbe627c02af` (gemma12b-next `6a21c8c3`), recipe
  `recipes/infervisor/gemma-4-12b/sm90a-h100-tp1.toml` (production). Baseline vLLM 0.28.0 TRITON_ATTN with matched `fp8_per_token_head` KV,
  prefix caching on both, one H100, 2 repeats.
- Gate: `llm_fp32_ref` PASS for that packet, KL mean 0.1048 vs vLLM 0.1277.
- Total-throughput ratio (strict): 4K/128 1.38x / 1.37x (c32 / c128); 15K/128 1.79x / 1.75x;
  agentic16k 1.40x / 1.31x / 4.02x (c32 / c64 / c128). Open-loop production mix at
  0.628 / 0.771 / 0.987 sessions/s: 1.08x / 1.18x / 1.50x total throughput, goodput
  1.08x / 1.57x / 4.57x (open-loop P99 latencies FLAGGED, direction only). TTFT/TPOT P99 ratios
  are in the comparison.
- Not a win against vLLM's fastest config everywhere: vs vLLM BF16 KV, the matched BF16-KV plow
  packet `af6ee1c20678` (gate PASS, KL 0.0821) is 0.83x / 0.79x at 4K, 0.86x / 0.86x at 15K and
  0.68x / 2.01x / 1.39x on agentic16k (c32 / c64 / c128).
- Evidence: `/opt/dlami/nvme/lava-tts/final3/` (plow arms, gates, reports), vLLM arms in
  `/opt/dlami/nvme/lava-tts/final2/` (closed loop) and `/opt/dlami/nvme/lava-tts/prodbench/`
  (open loop), `/opt/dlami/nvme/lava-tts/fp32gate/` (gate calibration).

Reproduction on main `fc0271e8` (2026-10-03; packet `31a1f44b` built clean from main through
`$S/gate12b.toml`, prefix cache on, 0 prefix hits): `llm_fp32_ref` PASS (KL mean 0.105 vs 0.128,
KL p99 2.96 vs 3.19, top1_decisive 0.974 vs 0.982, cont_frac 0.651 vs 0.574, needle 1.0). Out
tok/s 405.2 / 444.2 at 4K c32 / c128 and 108.3 / 108.9 at 15K, against the recorded 381.3 / 385.4 /
94.6 / 95.7 (`genrungs/m/serve3/cand2`, packet `bd968263`). The prompt count differs (3x conc vs
2x / 1x), so these are not like-for-like. Evidence: `/opt/dlami/nvme/lava-tts/repro-main/`.

### Reproduce

`scripts/campaign/repro_gemma12b_h100.sh` rebuilds the whole strict comparison from a checkout:
both packets (`build`), the FP32 reference (`ref`, skipped when `REF` exists), both gates
(`gate fp8|bf16`), every arm (`bench <fp8|bf16|vllm|vllm-bf16> <st4k|st15k|agentic|prod>`) and the
reports plus one combined markdown (`report`). It pins both vLLM flag sets; one GPU step per queue
lease (`scripts/bench/gpuq.py submit`). Inputs are env (`OUT`, `GEMMA12B_CHECKPOINT`, `HF`, `PYREF`,
`VLLM_PY`, `REF`/`REF_VLLM`, `CORPUS`, `OBJECT_ENV`, `RT_ENV`, `VLLM_ENV`; see the header). The
qualified gate used `ref.json` sha256 `b2988c51`, prompts `10302403`, vLLM peer capture `74a7a572`
(`/opt/dlami/nvme/lava-tts/fp32gate/`); passing those files as `REF`/`REF_VLLM` reproduces it. Its
`report` step re-renders the 2026-10-05 tables byte-for-byte from the campaign's result dirs.
Expect the usual run-to-run spread on re-measured arms (cells marked `*` most).

### Experimental history (not qualified)

Manifests, JSON, HTML, logs and kernel audits from before 2026-10-01 are archived at
`/opt/dlami/nvme/tmp/gemma12b-main-20260930/repo-evidence-archive-327413c3` (manifest with
SHA256s). Per-rung route notes:
[kernel-generators.md](../../../runtime/kernel-generators.md#gemma-4-12b-fp8-per-rung-routes-h100-fp8-kv-16k-128-slots).

- 2026-10-03 final compare (rows `final-compare-20261003-*`): no qualified win. The FP8-KV arm
  was 0.50x / 0.45x at 4K and 0.37x / 0.36x at 15K. The BF16-KV arm (NOT MATCHED on KV precision)
  was 1.58x / 1.53x at 4K and 2.43x / 2.50x at 15K.
- Earlier unmatched or diagnostic arms, c128, output tok/s plow vs vLLM: 128/128 16K/128-slot
  FP8-KV sampled 4,228 vs 6,552 (64.5%); 128/128 1K BF16-KV sampled 5,814 vs 7,526 (77.2%);
  4096/128 16K/64-slot BF16-KV greedy 665 vs 893 (74.4%); 15000/128 same packet 196 vs 244
  (80.0%). Exact-match decode top-1 vs vLLM was 37/64; that gate was retired for `llm_fp32_ref`.
- Internal A/Bs, four-arm, matching token digests against the plow control:
  - FATLITE packed prefill: 8.1% lower instrumented prefill time, 5.7% higher long-serving
    throughput.
  - Dedicated cached GLU+quant: 2.5-3.4% lower block time across six rungs.
  - Lightweight FP8 attention route: 8.6% lower decode time.
  - B128 FP8-KV light attention on cached norm/quant: 4.80% lower decode step at ctx 128, 4.82% at 4K.
  - B128 cuBLASLt output head: 13.38% lower decode step at ctx 128, 5.40% at 4K (max KL 2.17e-5).
  - Head-Lt plus FP8 light attention: a further 4.96% / 3.21%.
  - FP8 attention with 16-byte loads: a further 2.76% / 6.71%.
  - Direct FP8 value conversion with 16-byte K loads: a further 1.65% / 4.49% / 3.24% (ctx 128 / 4K / 15K).
  - Segmented FP8 HD256 FlashDecode, 1,024-block grid: 10.7% / 9.6% lower decode step, 7.0%
    higher sampled short-serving throughput; 95 registers, 11,328 B shared memory, no spills.
- Packed FP8 HD256 prefill (`PLOW_NV_FA_PIPE=1`, `PLOW_NV_FP8_PACKED_VARLEN=1`), wrapper timings
  at 16 requests x 2,048 rows: 6.5731 -> 0.2683 ms (KV 256), 10.4859 -> 0.3621 ms (4K),
  10.7184 -> 0.3724 ms (15K). Dedicated cubin with flat scheduling: 3.52x / 3.86x / 3.86x faster.
  The layer-0 block A/B **failed** parity at the 1,152 -> 2,048-row bucket switch (first
  difference in the cuBLASLt FP8 `down_proj` output, 0.413% relative L2). Nsight Compute: 12.5%
  occupancy, 255 registers, 132,160 B shared memory, DRAM 5.8-6.1% of peak; not at a measured
  roofline. Evidence: `packed-bench-build/production-flags` and
  `head-lt-build/grid-trial/block-fa-probe` in campaign scratch.
- 128 slots at 16K, BF16 KV with live rings (`sm90a-h100-tp1-fp8-16k-c128-rq1k-live.toml`):
  loads and serves (71-78 live at 4K, 64 at 15K) but is prefill-bound and no faster than the
  64-slot packet (4K c128 557 vs 616 out tok/s; 15K c128 189 vs 195). FP8-KV live rings
  (`sm90a-h100-tp1-fp8kv-16k-c128-rq1k-live.toml`) fit 128 x 16K but could not serve at 4K
  (packed prefill attention ~55 ms per riding decode row). Raw cells:
  `/opt/dlami/nvme/lava-tts/cap128/serve`.
- The superseded 1K trial recipes (`-lt` and `-lt-head`) equal
  `sm90a-h100-tp1-fp8-1k-c128-lt-head-tc64.toml` with `PLOW_FP8_DECODE_TC64=0` (and
  `PLOW_EMIT_DECODE_CUBLASLT_HEAD` unset for `-lt`). The 4K/128-slot BF16-KV trial failed
  allocation on an 80 GB H100 and was dropped.

## Gemma-4 12B and 26B-A4B BF16

No strict report and no qualified win is recorded. Trial recipes:
`scripts/campaign/recipes/gemma4-12b.h100.bf16-*.toml` and `gemma4-26b-a4b.h100.*.toml`.
`campaign.py ledger` rows: `perf-data/campaign/gemma4-12b.h100.bf16*.csv` and
`gemma4-26b-a4b.h100.bf16-{ctx16k,c32-16k}.csv`. They are compared against the vLLM 0.28
reference CSVs `perf-data/campaign/*.reference-vllm028-{bf16,bf16-high-concurrency,fp8}.csv`
that the recipes' `[reference]` sections name. Packet recreate records:
[perf-data/packets](../../../../perf-data/packets/README.md). Campaign notes:
`plans/gemma4-dense-realtime-tracker.md`, `plans/gemma4-26b-a4b-h100-tracker.md`.

## Gemma-4 31B

- H100: frozen release `/opt/dlami/nvme/plow-releases/gemma4-31b-h100-20260909`, functionally
  validated profiles only ([gemma4-h100-release.md](../../../runtime/gemma4-h100-release.md),
  [gemma4-h100-checkpoint.md](../../../runtime/gemma4-h100-checkpoint.md)). No serving
  comparison is recorded.
- MI300X: qualified 2026-09-07 (TP1, BF16, 8K context) on numerics and serving checks; the stock
  `vllm bench serve` baseline favors vLLM in every throughput, TTFT and TPOT cell
  ([gemma4-31b-mi300x.md](../../../amd/gemma4-31b-mi300x.md)).
