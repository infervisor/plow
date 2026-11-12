# Speech serving concurrency on one H100: plowrt vs reference servers (screen)

These are screens, not qualified results: other builds ran on the box during every lease (no
CPU-quiet lock), each cell ran once or twice, and there is no strict `campaign.py report` for speech.
No row is a qualified plow-vs-baseline win or loss; the ratios are large enough to rank the work.

- Box: one H100 80GB HBM3, 2026-11-11/12. Same client for every server
  (`scripts/tts/tts_bench.py`, `scripts/asr/nvidia/served_bench.py`), same texts / clips, same
  seeded open-loop arrival schedules (`--rate`, Poisson, 60 s TTS, 45-60 s ASR).
- plowrt: shipping binary 0.1.3+4df7a70 (sha256 `1d76b0d02fc4`), released packets, each recipe's
  `[serve]` env, default `auto` objective.
- References: Maya2 provider engine `maya3` 0.1.0 (224 seats, capacity 1536, busy fraction 0.6);
  vLLM-Omni 0.32.0 for Qwen3-TTS (its one-GPU high-concurrency deploy profile, private MPS);
  vLLM 0.32.0 for Qwen3-ASR (vLLM-Omni has no Qwen3-ASR pipeline).
- TTS SLO: first audio within 500 ms and no playback underrun over 100 ms. ASR SLO: answer in 1 s.

## TTS

| Model | Cell | plowrt | Reference |
|---|---|---:|---:|
| Maya2 | c32 / c128 audio s/s | 63 / 105 (111 stalls at c128) | 114 / 231 (0 stalls) |
| Maya2 | c1 first audio p50 | 21 ms | 16 ms |
| Maya2 | highest open-loop rate within SLO | 20 req/s (108 audio s/s) | 40 req/s at 100 %, 60 req/s at 95 % (275 audio s/s) |
| Maya2 | 40 req/s | 2 % within SLO, 4526 stalls | 100 %, 0 stalls |
| Qwen3-TTS | c32 / c128 / c512 audio s/s | 102 / 119 / 123 | 174 / 356 / 371 |
| Qwen3-TTS | highest open-loop rate within SLO | 20 req/s (106 audio s/s) | none: every stream has one 100-260 ms underrun after its first 80 ms chunk |
| Qwen3-TTS | 40 req/s | 4 % within SLO, first audio ~10 s, 603 of 2388 refused | all complete, first audio p99 41 ms, one underrun per stream |
| Orpheus | c32 / c128 / c512 audio s/s | 61 / 139 / 152 | not measured |

## ASR (Qwen3-ASR 1.7B)

| Cell | plowrt | plowrt, encoder hold 50 ms | vLLM |
|---|---:|---:|---:|
| c1 latency p50 | 53.8 ms | 54.1 ms | 67.0 ms |
| c16 / c64 / c256 RTFx | 597 / 850 / 1169 | 594 / 1036 / 1155 | 821 / 1673 / 1778 |
| 100 req/s p50 / p99 | 283 / 1041 ms | 222 / 822 ms | 119 / 513 ms |
| 150 req/s within SLO | 0 % | 99 % | 99.9 % |
| 300 req/s within SLO | 0 % | 0 % | 98 % |
| saturation audio s/s | ~985 | ~1277 | 1974 at 300 req/s |
| WER | 0.0391 | 0.0391 | 0.037 |

## Where plowrt's time goes at saturation

- Qwen3-TTS: isolated talker step 2.6-3.5 ms at 64 rows; in-serve frame 26.8 ms at 64 rows and
  ~0.36 ms per row to 128 rows (code predictor). Stateful codec step: ~4.4 ms per launch +
  ~0.12 ms per row-frame (4.6 ms at 1x1, 133 ms at 64x17). A 256-row ladder gave no throughput.
- Maya2: over a saturated 94 s cell, depth 47 s, codec 38 s, talker 15 s; a 224-row frame is
  ~48 ms of depth. Codec step ~3.2 ms + ~0.06 ms per row-frame.
- Qwen3-ASR with the hold: encoder packs of 64-128 chunks take ~63 % of the device (~0.40 ms per
  chunk); without it ~2100 single-clip launches per 40 s.
- Rejected here: codec hold / deadline order knobs and a longer codec burst gather (5-9 % more
  Qwen3-TTS audio s/s, but 2-6x more streams with small underruns and a Maya2 regression).

## Evidence

`/opt/dlami/nvme/lava-tts/sconc`: `res/{base1,base2,peer1,steps1,w256b,codec1,asr2,asr3,asrhold,asrhold2,gather1,gather2}`,
lease scripts beside them, reference installs and launch notes in `peers/PEERS.md`.
