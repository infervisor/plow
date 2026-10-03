# H100 production recipes: reproduction on main fc0271e8

- Main commit: `fc0271e81bbfd4239b60e9c010cbb80804521ac7`. Build records have `dirty=true`, but the tracked diff is empty. The only untracked path is `lean-plow/.lake`.
- `plowrt` sha256 = `be62ab42a8a02081d4f52e75ef5f9f0a7220f1aef4cb1fc6a8b7e228c81fdc47`
- `plowc` sha256 = `dc83457f7186193a6dbac12ab1d9fae96520e4934e05b277557358676c49e1f3`, the same at build start and end.
- Toolchain: CUDA 12.9 nvcc with `-ccbin g++-14`, no nix (`PLOW_CAMPAIGN_NO_NIX=1`). Driver 595.91.07.
- Box: 1× NVIDIA H100 80GB HBM3 (SXM). One `gpulease` lease ran from 2026-10-03 22:01 to 22:57 UTC. Every cell exited with rc=0.
- Scratch: `S=/opt/dlami/nvme/lava-tts/repro-main`. Results are in `$S/res`, gate captures in `$S/gates/<model>`, and score output in `$S/report/gates-score.log`.

## Packets (from `$S/<model>/build-record.json`, matched against `gates/*/packet.sha256` and grid `provenance.json`)

| model | packet | aux |
|---|---|---|
| gemma-4-e4b | 62408e49 | |
| qwen3-asr | 561172a6 | encoder 0eb16493 |
| veena | c45cf845 | codec c7026b0f |
| chatterbox | bfc34919 | s3gen 64ef9339 |
| chatterbox-mtl | 62abc913 | s3gen 9720eec7 |
| gemma-4-12b | 31a1f44b | |

## Gates (`campaign.py gate ... --score-only`, all PASS)

| model | gate | result | metrics | recorded (h100-speech-recipes.md:45-51) |
|---|---|---|---|---|
| qwen3-asr | asr_wer | PASS | WER 3.913%, 0 errors | 3.913% |
| gemma-4-e4b | llm_logit_parity | PASS | top1 0.9897, KL mean 7.4e-4, KL max 0.014 | top1 0.985, KL 8e-4 |
| veena | tts_cer | PASS | CER median 0.005, n=80 | 0.005 |
| chatterbox | tts_cer, s3gen_rel_l2 | PASS | CER 0 (n=32); mel rel-L2 1.23e-5 | 0; 1.2e-5 |
| chatterbox-mtl | tts_cer, s3gen_rel_l2 | PASS | CER 0 (n=32), worst language fr 0.148; mel rel-L2 1.23e-5 | 0 (fr 0.148); 1.2e-5 |
| gemma-4-12b | llm_fp32_ref (`$S/gate12b.toml`) | PASS | KL mean 0.105 vs vLLM 0.128; KL p99 2.96 vs 3.19; top1_decisive 0.974 vs 0.982; cont_frac 0.651 vs 0.574; needle 1.0 vs 1.0 | 0.109/0.128, 2.50/3.19, 0.984/0.982, 0.650/0.574 (kernel-generators.md:380-382) |

`packet.sha256` was written at gate dry-run or build time, before the lease. No file was created during scoring. The 12B top1_decisive (0.974) is below both the recorded 0.984 and vLLM's 0.982, and KL p99 is higher (2.96 vs 2.50). It still passes the gate's relative criteria.

## Performance (mean of 2 repeats, [min-max])

Recorded sources:
- Speech and E4B: `docs/runtime/h100-speech-recipes.md:68-74`.
- 12B: `docs/runtime/kernel-generators.md:392-395`. The raw data is in `/opt/dlami/nvme/lava-tts/genrungs/m/serve3/cand2`.

| model | metric | recorded (source) | main fc0271e8 | delta |
|---|---|---|---|---|
| E4B | out tok/s c64 | 9,169 (:70) | 9,029 [9,009-9,049] | -1.5% |
| E4B | out tok/s c128 | 14,662 (:70) | 14,471 [14,430-14,512] | -1.3% |
| E4B | TTFT p50 c64 | 51 ms (:71) | 75.3 ms [67.8-82.8] | **+48%** |
| E4B | TTFT p50 c128 | 80 ms (:71) | 91.0 ms [76.6-105.4] | **+14%** |
| Veena | out tok/s c64 | ~13,780 (:68) | 13,824 [13,819-13,830] | +0.3% |
| Veena | out tok/s c128 | ~20,590 (:68) | 20,653 [20,621-20,685] | +0.3% |
| Veena | stream aps c64 | 76.2 (:69) | 77.4 [77.4-77.4] | +1.6% |
| Veena | TTFA p50 c64 | 138 ms (:69) | 136.6 ms | -1.0% |
| Qwen3-ASR | c1 latency p50 | 55.2 ms (:72) | 54.5 ms [54.5-54.6] | -1.3% |
| Qwen3-ASR | RTFx c64 | 728 (:72) | 645 [639-651] | **-11.4%** |
| Qwen3-ASR | RTFx c128 | 758 (:72) | 588 [575-602] | **-22.4%** |
| Chatterbox MTL | stream steady aps c200 (0 failed) | 71.7 (:73) | 78.8 [78.7-78.9]* (plain aps 77.6) | +9.9%* |
| Chatterbox | stream aps c64 | 58.4 (:74) | 61.9 [61.8-62.0] | +6.0% |
| 12B 4K c32 | out tok/s | 381.3 (:392) | 405.2 [404.1-406.3]† | +6.3%† |
| 12B 4K c128 | out tok/s | 385.4 (:393) | 444.2 [444.1-444.2]† | +15.3%† |
| 12B 15K c32 | out tok/s | 94.6 (:394) | 108.3 [108.2-108.4]† | +14.5%† |
| 12B 15K c128 | out tok/s | 95.7 (:395) | 108.9 [108.9-108.9]† | +13.8%† |
| 12B (4K c32 / 4K c128 / 15K c32 / 15K c128) | TPOT p50 ms | 73.4 / 160.2 / 156.6 / 172.9 | 71.2 / 137.5 / 152.9 / 154.2 | -3% / -14% / -2% / -11% |
| 12B (same order) | TTFT p50 s | 0.87 / 21.7 / 19.8 / 82.6 | 0.75 / 18.6 / 17.6 / 129.2† | -14% / -14% / -11% / +56%† |

\* The steady-aps figure comes from an approximation (see caveats). Plain aps, which reads lower, already exceeds the recorded 71.7 by 8.2%.
† The workload differs from the recorded run: n = 3×conc here vs 2×conc (4K) and 1×conc (15K) recorded. The 12B rows are therefore not like-for-like.

Other cells, measured but with no recorded value:
- E4B: c1 175 tok/s; d64x1024 5,548 tok/s; d128x1024 6,999 tok/s.
- Veena stream aps: c1 3.4 (TTFA 56 ms), c8 21.3, c32 52.5, c128 100.3 (TTFA 220 ms).
- ASR: c16 RTFx 523; WER 3.913% in every arm.
- Chatterbox: c1 stream 8.9 aps (TTFA 132 ms); c8 31.0.
- MTL: c1 8.7 aps (TTFA 133 ms); c16 46.4; c200 TTFA p50 6.2 s.
- Failed requests: 0 in every cell.

## Regressions > 5%

- **Qwen3-ASR RTFx c64 -11.4%, c128 -22.4%.**
  - Same client, same 73-clip / 481 s manifest, same RTFx definition (`served_bench.py` is unchanged since af3fda87).
  - Both repeats fall well below the recorded values.
  - c1 latency and WER are unchanged.
  - The c64/c128 runs are short (wall 0.75-0.80 s), so sensitivity to noise is high. Even so, the gap is about 8× the observed spread.
  - Candidate cause: the 29963a91 scheduling rewrite (turn-aware scheduling, `PLOW_OBJECTIVE=auto`, changes to `asr/serving.rs`).
- **E4B TTFT p50 c64 +48%, c128 +14%.**
  - Same `llm_grid` workload as recorded: n = 3×conc, ISL 128 / OSL 512, greedy.
  - Throughput is only -1.3% to -1.5%, inside the doc's ±1.5% band.
  - Both repeats sit above the recorded c64 value.
  - Possible contributors: oldest-first admission and the scheduling objective from 29963a91. A second contributor: main's `llm_grid` now runs an nvidia-smi 100 ms memory sampler and a ~50 ms `/metrics` poller on the plow side.
- **12B TTFT p50 15K c128 +56%.** This is not a confirmed regression. The recorded run sent n=128, a single wave; this run sent n=384, three waves, so queueing adds TTFT. TPOT and throughput both improved.
- No other metric regressed by more than 5%.

## Workload and definition mismatches

- **steady_aps.** Main's `scripts/tts/tts_bench.py` has no `steady_aps`.
  - The job used a scratch `$S/src/scripts/tts/tts_bench_steady.py`: main's `tts_bench.py` plus a per-request start offset and a steady-aps estimate.
  - The estimate counts audio delivered inside the middle 60% of the wall time, with each request's audio spread uniformly over [start+TTFA, end].
  - The recorded 71.7 used an unknown, original definition.
  - Plain `audio_s_per_s` is unchanged from main.
- **12B.**
  - Prompt count: recorded n was 64/256 (4K) and 32/128 (15K); here it was 96/384 at both lengths.
  - Prefix cache: recorded with the prefix cache off; here `PLOW_PREFIX_CACHE=1 PLOW_VMM_PREFIX=1`. Prefix hits were 0 in `metrics.tsv` (unique per-cell seeds), so this is no confound.
  - Packet: 31a1f44b, built clean from fc0271e8. Recorded: cand2 bd968263, built from dirty a9262902 (`genrungs/pk/cand2/build-record.json`).
- **Veena / Chatterbox / MTL stream cells.** The job served `plowrt serve` directly with the recipe env and the same `tts_bench` arms, not through `plow_speech_probe.sh`.
- **E4B / Veena tok/s and ASR.** Workload matches: `llm_grid` np = 3×conc (same formula at af3fda87), and the same `served_bench.py` with the same manifest.

## Commands

```
# gates (score-only; run from worktree agent-aa773a95e50ca2d3c after `. $S/env.sh`, PLOW_CAMPAIGN_NO_NIX=1)
python3 scripts/campaign/campaign.py gate recipes/infervisor/<m>/sm90a-h100-tp1.toml --assets $S/<m>/assets --out $S/gates/<m> --score-only
python3 scripts/campaign/campaign.py gate $S/gate12b.toml --assets $S/gemma-4-12b/assets --out $S/gates/gemma-4-12b --score-only
# benches: $S/job.sh (gate run.sh -> llm_grid.sh plow REPS=2 -> plowrt serve + tts_bench_steady.py / served_bench.py, 2 reps)
# summaries
python3 $S/report/summ.py        # means/spread over repeats (grid_summ.py reads only the first rep per cell)
python3 $S/report/rec12.py       # recorded 12B raw bench.json workload check
```

## Caveats

- The speech interpreter `ptxas` took about 7 h in this build, vs 2h13 on 834d0b6c (caller-reported; not re-measured here).
- The 12B packet differs from the recorded one (above). Its gate passes, but top1_decisive fell to 0.974 (recorded 0.984).
- MTL c200 steady aps is an approximation (above). The no-regression conclusion holds on plain aps.
- Host CPU load during the lease was not recorded. Builds had finished before it (`build2.status` ALL_DONE), and the box has 16 cores with load ~1.2 afterwards. Main's `llm_grid` adds its own nvidia-smi and `/metrics` pollers during cells.
- There were only 2 repeats per cell. The doc states single-run variance of ±1.5%.
