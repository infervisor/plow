# Gemma-4 E4B per-rung routes: cuBLASLt decode from 2 rows, generated prefill attention (H100)

Campaign `native-kernels`, base `fd0fe175` (gemma12b-next), local H100 SXM through `gpuq`.
Raw logs, builds and scripts: `/opt/dlami/nvme/lava-tts/nativek/` (`res/`, `n-*/`, `jobs/`).

## Change (recipe only; no plowrt or kernel source change)

`recipes/infervisor/gemma-4-e4b/sm90a-h100-tp1.toml`:

* `PLOW_EMIT_DECODE_CUBLASLT_MIN_ROWS` 48 -> 2. Decode rungs 2..32 now run the routed program
  (cuBLASLt projections + light launches) that 48..128 already ran; B=1 keeps the interpreter.
* `PLOW_BUILD_GEN_KERNELS` / `PLOW_EMIT_GEN_KERNELS = attn_pf_hd256_sliding,attn_pf_hd512`
  plus the two `gen_sm90a_*` role files: the generated-kernel catalog's prefill attention on
  the 1024+ row rungs (both entries' `min_rows`).

## Per-rung decode (step_bench ctx 1024, 64 steps, `--spread`, `PLOW_LT_RUNG_ALGOS=true`, ms, 2 passes)

| B | base (MIN_ROWS 48) | candidate (MIN_ROWS 2 + gen) | change |
|---|---|---|---|
| 1 | 5.716 / 5.718 | 5.716 / 5.713 | 0 |
| 2 | 5.749 / 5.746 | 5.619 / 5.616 | -2.3% |
| 4 | 5.865 / 5.857 | 5.686 / 5.682 | -3.0% |
| 8 | 6.169 / 6.167 | 5.912 / 5.953 | -3.8% |
| 16 | 6.765 / 6.770 | 6.253 / 6.235 | -7.7% |
| 32 | 8.622 / 8.616 | 6.790 / 6.775 | -21.3% |
| 48 | 7.038 / 7.140 | 7.140 / 7.130 | 0 (same route) |
| 64 | 7.396 / 7.442 | 7.400 / 7.393 | 0 (same route) |
| 128 | 9.513 / 9.481 | 9.481 / 9.480 | 0 (same route) |

The native B=32 rung was slower than the routed B=48 rung (8.62 vs 7.14 ms). The earlier
rejection of routing from 32 rows (B=32 9.11 -> 9.68) predates the light launches.
Roofline (weights 9.22 GB + KV at ctx 1024, 3.35 TB/s): B=32 3.44 ms -> 51% (was 40%), B=16
3.12 ms -> 50% (46%), B=1 2.83 ms -> 49% (unchanged).

## Prefill (step_bench `--packed-prefill`, 8 reps, wall ms of reps 3..8)

| shape | base | generated roles | change |
|---|---|---|---|
| 1 x 1000 rows | 19.1-19.4 | 17.9-18.6 | -6% |
| 1 x 2000 | 32.3-33.0 | 31.6-32.1 | -3% |
| 4 x 1000 (packed) | 61.6-62.4 | 59.9-60.8 | -3% |
| 1 x 500 (rung 512, unchanged route) | 12.7-12.8 | 12.7-12.8 | 0 |

Two passes at fd0fe175 (an earlier pair at fcb2250c gave 19.3 -> 18.3, 32.7 -> 31.7,
62.2 -> 60.3). First tokens are equal in every shape. Standalone (kernel-generators.md): hd256
sliding 1000 rows 37.7 -> 26.7 us, hd512 94.1 -> 48.1 us; 35 + 7 sites per 1000-row prefill.

## Gates and serving

Gate `llm_logit_parity` (campaign.py gate run.sh through gpuq, plowrt fd0fe175):

| arm | top1 | KL mean / max | note |
|---|---|---|---|
| base (recipe at fd0fe175) | 0.9897 (383/387) | 7.37e-4 / 1.44e-2 | |
| candidate (MIN_ROWS 2 + gen) | **0.9819** (326/332) PASS | 7.77e-4 / 1.54e-2 | cmpl0 ends at 18 tokens (the known load-to-load state, rung 64); chat4 (1929 rows, gen roles) greedy identical to HF 39/39 vs 11/48 base |
| MIN_ROWS 8 + gen | 0.9819 (326/332) | 7.77e-4 / 1.54e-2 | identical to the candidate (c1 gate: B=1 + prefill) |

Served, `scripts/llm/gemma_voice_bench.sh plow` (same client, random ISL 1000 / OSL 128,
`--temperature 0`, ABAB, 2 passes; values pass1 / pass2):

| cell | base out tok/s | candidate out tok/s | base TTFT p50 ms | candidate TTFT p50 ms | base TPOT p50 | candidate TPOT p50 |
|---|---|---|---|---|---|---|
| c1 | 169.4 / 169.4 | 170.8 / 170.8 | 21.0 / 20.8 | **14.8 / 14.5** | 5.70 | 5.70 |
| c8 | 1117 / 1117 | 1233 / 1213 (+9.5%) | 65.3 / 64.5 | 43.5 / 43.1 | 6.63 | 6.10 / 6.21 |
| c16 | 1854 / 1849 | 2117 / 2169 (+16%) | 67.3 / 68.4 | 45.7 / 45.1 | 8.03 | 7.14 / 6.96 |
| c32 | 2568 / 2569 | 3401 / 3403 (+32%) | 69.3 / 69.1 | 47.4 / 47.2 | 11.51 | 8.73 / 8.72 |
| c64 | 4104 / 4114 | 4974 / 4992 (+21%) | 107.3 / 105.7 | 75.6 / 75.2 | 14.30 | 11.74 / 11.78 |

Grid only (no vLLM arm in this run, no strict `campaign.py report`): not a qualified
plow-vs-baseline claim. Raw: `res/serve1/`, gates `gate/n-*/`, rungs `res/rungs2/`, prefill `res/pf2/`.

## Tried, not kept

* hd256 light attention K/V loads through L2 without the evict-first hint (the partner head
  group re-reads the rows): B=48/64/96/128 unchanged (7.14/7.44/8.98/9.49 ms vs 7.14/7.43/8.99/9.50).
* hd256 light attention at head group 4 (KV read once): 128-register cap, 320 B spills
  (536 B loads); with `PLOW_NV_FA_RG_U=2` still 156 B spills. Not run.
* Customer H100 baseline (kit3 bundle, before the host was withdrawn): B=1/8/32/48/64/96/128
  5.58/6.02/8.35/6.83/7.13/8.79/9.17 ms; at B=128 attention is 3.4 ms of the 9.2 ms step
  (sliding 59 us/layer, full 203 us/layer), launch gaps 0.57 ms over 489 launches.
