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

## Block vs roofline (Qwen3-ASR 1.7B encoder, 128-chunk packed launch)

Ceilings measured on this box: HBM read 3093 GB/s, BF16 781 TFLOP/s. Released packet, per-op times
from `packet_bench --sweep`, floors from `speech_roofline.py`. Only this model has its table so far;
the TTS depth and codec stages are owed theirs before any further change to them.

| Block (rung 128 chunks = 1664 rows) | Block time | Roofline floor | % of floor | Delta vs current |
|---|---:|---:|---:|---:|
| conv front end (3 convs + pack + first layer head) | 20.6 ms | 1.04 ms | 5.1 % | 10.4 ms with the sm_90a conv paths |
| conv2 480->480, 64x50 | 11.9 ms | 0.54 ms | 4.5 % | |
| conv1 1->480, 128x100 (3xTF32) | 4.2 ms | 0.25 ms | 5.9 % | |
| conv3 480->480, 32x25 | 3.4 ms | 0.14 ms | 4.1 % | |
| transformer layer (x22) | 1.19 ms | 0.11 ms | 9.1 % | |
| fc1 + fc2 per layer | 0.53 ms | 0.036 ms | 6.9 % | |
| q/k/v + attention per layer (one launch) | 0.19 ms | 0.014 ms | 7 % | |
| residual add + LayerNorm, 4 passes per layer | 0.10 ms | 0.022 ms | 22 % | |

Fusion list: residual add into the preceding GEMM epilogue (2 passes per layer, ~49 us, bit
preserving after the GEMM's own round); residual add + LayerNorm in one row-owning pass (~20-30 us
against ~48, serial statistics chains kept); q/k/v as one N = 3072 launch only with per-projection
reference chunking kept. LayerNorm into the dense prologue does not fuse: the reference-split dense
path rejects a LayerNorm prologue.

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

"hold" = `PLOW_ASR_ENCODE_HOLD_MS=50` (the branch default). "hold + encoder" = that binary with the
packet re-emitted from the branch (direct conv and reference-split dense paths on sm_90a; only the
speech object differs from the released packet), mean of two interleaved runs.

| Cell | plowrt shipping | hold | hold + encoder | vLLM |
|---|---:|---:|---:|---:|
| c1 latency p50 | 53.8 ms | 54.1 ms | 53.4 ms | 67.0 ms |
| c64 / c256 RTFx | 850 / 1169 | 1041 / 1152 | 1190 / 1305 | 1673 / 1778 |
| 150 req/s within SLO, p99 | 0 % | 99 %, 993 ms | 100 %, 747 ms | 99.9 %, 637 ms |
| 200 req/s within SLO | 0 % | 0 % | 98 % | 99.7 % |
| 300 req/s within SLO | 0 % | 0 % | not run (0 % at 250) | 98 % |
| saturation audio s/s | ~985 | ~1270 | ~1425-1525 | 1974 at 300 req/s |
| WER, gate cell (73 clips, c1 / c16) | 0.03913 | 0.03913 | 0.03826 | 0.037 (292 clips) |

Qwen3-ASR 0.6B, hold vs hold + encoder (one run): c64 RTFx 1207 -> 1409, c256 1318 -> 1545,
200 req/s within SLO 0 % -> 99.7 %, gate WER c1 0.04435 -> 0.04174, c16 0.04261 = 0.04261.

Packing: the encoder is bit-exact under packing with every route. `speech_f32_op_test
--conv-packing-only` (each front-end conv, a clip alone vs inside batches of 2 / 16 / 128, at 132 and
44 blocks) and `asr_encoder_pack_check` (73 recordings, each alone vs in packed launches of 4 and 16)
report 0 differing elements for the released packet, both conv-path re-emits, the dense-only re-emit
and the released 0.6B packet. Served transcripts still depend on concurrency in a few clips (292-clip
set, c1 vs c64: released 1.7B 0-1, re-emit 7-12, released 0.6B 8), so that dependence enters after
the encoder, in batched prefill / decode, and is present in the released packets. Open: localize it
in the decoder.

## Where plowrt's time goes at saturation

- Qwen3-TTS: isolated talker step 2.6-3.5 ms at 64 rows; in-serve frame 26.8 ms at 64 rows and
  ~0.36 ms per row to 128 rows (code predictor). Stateful codec step: ~4.4 ms per launch +
  ~0.12 ms per row-frame (4.6 ms at 1x1, 133 ms at 64x17). A 256-row ladder gave no throughput.
- Maya2: over a saturated 94 s cell, depth 47 s, codec 38 s, talker 15 s; a 224-row frame is
  ~48 ms of depth. Codec step ~3.2 ms + ~0.06 ms per row-frame.
- Qwen3-ASR with the hold: encoder packs of 64-128 chunks take ~63 % of the device (~0.40 ms per
  chunk); without it ~2100 single-clip launches per 40 s. Isolated packed.128 launch: 42.2 ms, of
  which the three front-end convolutions 19.6 ms; 33.1 ms with the sm_90a paths (conv 9.8 ms). The
  22 transformer programs stay ~1 ms each (dense ~48 TFLOP/s effective).
- Rejected here: codec hold / deadline order knobs and a longer codec burst gather (5-9 % more
  Qwen3-TTS audio s/s, but 2-6x more streams with small underruns and a Maya2 regression).

## Evidence

Campaign scratch `sconc`: `res/{base1,base2,peer1,steps1,w256b,codec1,asr2,asr3,asrhold,asrhold2,gather1,gather2,encbench,enc2,gate3,final1,roof}`,
lease scripts beside them, reference installs and launch notes in `peers/PEERS.md`.
