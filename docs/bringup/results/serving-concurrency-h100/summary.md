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

### Encoder dense: continuous bf16 accumulation (128-chunk packed launch, 1.7B)

Reference servers run the encoder in bf16. The encoder's dense layers already multiply bf16 operands
exactly; `PLOW_ASR_DENSE_CONTINUOUS=1` drops the split-K sum (and the per-row reference that kept
packed launches equal to single ones) for one k16 accumulation chain per output, which a 128 x 128
warpgroup tile computes at any batch. Packed vs single stays bit-identical (0 of 73 recordings differ
at pack 4 and 16, both models).

| Dense op, 1664 rows | split-K reference (64-row tiles) | continuous, 128 x 128 warpgroup | cuBLASLt bf16 operands |
|---|---:|---:|---:|
| 1024 x 1024 projection | 57 us | 30 us | 18 us |
| 4096 x 1024 + erf-GELU (fc1) | 168 us | 132 us | 32 us (no GELU) |
| 1024 x 4096 (fc2) | 224 us | 94 us | 30 us |
| 1024 x 7680 (conv out) | 395 us | 162 us | 57 us |

| Encoder, same tree | reference split | continuous |
|---|---:|---:|
| packed.128 (1.7B / 0.6B) | 31.2 / 23.0 ms | 20.4 / 16.5 ms |
| packed.32 (1.7B / 0.6B) | 10.7 / 7.9 ms | 8.1 / 6.4 ms |
| forward.800, one utterance (1.7B / 0.6B) | 4.64 / 3.57 ms | 5.09 / 3.81 ms |
| conv front end program (1.7B) | 9.9 ms | 9.4 ms |
| one transformer-layer program (1.7B) | 0.90-1.16 ms | 0.47-0.59 ms |

Served, two interleaved runs each (1.7B): gate WER 0.03826 at c1 and c16 on both; c64 RTFx
1218 / 1220 -> 1343 / 1331; c256 1302 / 1285 -> 1498 / 1463; 250 req/s within the 1 s SLO 0 % -> 84 % /
98 %; 300 req/s 1538 / 1545 -> 1681 / 1815 audio s/s; 0 faults. 0.6B (one run): c64 1427 -> 1515, c256
1532 -> 1623, 300 req/s 1582 -> 1925 audio s/s, but gate WER c1 0.04174 -> 0.04348 and c16 0.04261 ->
0.04522 (limit 0.045): the 0.6B recipe stays on the reference split until its reference-server WER
bounds are measured. The single-utterance program loses 7-10 % (no split-K over idle blocks); c1 RTFx
109.3 -> 108.3.

The conv front end is now 46 % of the packed encoder and the next kernel target.

### Conv front end (128-chunk packed launch, 1.7B)

conv2 (480 -> 480, 3x3 stride 2) at 5.0 ms splits into input loads 1.6 ms, weight conversion 1.9 ms,
MMAs 0.9 ms, the rest 1.1 ms (variants with each part skipped); conv3 1.6 ms the same way. Both are
bound by L2 traffic (about 3 TB/s), not by the tensor cores.

`PLOW_ASR_CONV_BF16=1` stores the conv2 / conv3 weights as bf16 in the kernel's channel-block order,
so each tile copies them instead of converting FP32 rows: the same values and sums (op outputs
bit-identical to the FP32-stored weights; encoder pack parity 0 of 73). conv2 5.0 -> 4.2 ms, conv3
1.6 -> 1.5 ms; packed.128 encoder 20.4 -> 19.2 ms, forward.800 5.09 -> 4.94 ms. Served, two
interleaved runs: WER unchanged, c64 RTFx 1355 / 1342 -> 1387 / 1373, c256 1473 / 1494 -> 1542 /
1522, 250 req/s within SLO 32 % / 0 % -> 95 % / 99 %, 0 faults.

### Where a served request's GPU time goes (1.7B, 300 req/s, continuous dense)

GPU-alone cost per 6.7 s request: encoder 1.13 ms; decoder prefill 1.07 ms (a 512-row packed launch
takes 5.15 ms and carries about 4.8 requests: 3.4 ms fixed + 3.6 us per row); decode 0.76 ms (one
3.2-3.7 ms step per tick for ~105 live rows, ~23 ticks per request). 2.96 ms serialized against
3.7 ms observed. Decode rows did not ride the prefill launch on this packet at that point (its
audio overlay excluded the unified launch), so every tick swept the decoder weights twice. With the encoder
running beside it a prefill launch stretches from 6.4 ms (p50) to 32-45 ms (p90-p99) and a decode
step from 3.7 to 13.5 ms (p90).

Larger rungs alone do not help: prefill rungs to 1024 rows with 256 decode slots gave c256 RTFx
1562 / 1612 against 1542 / 1522, 250 req/s within SLO 15 % / 79 % against 95 % / 99 %, and p99 near
5 s past saturation (2.9 s before). Rejected.

### Decoder: decode rows ride the packed prefill launch (1.7B)

The audio-overlay packet now takes the unified launch through the staged step (the overlay rows and
their launch-row index are written per launch; decode rows index the token table). A launch at
saturation holds about 85 decode rows and 430 prefill rows in the 512-row rung and takes 6.6 ms,
against 5.2 + 3.3 ms for the two separate launches. One binary, control = the split-attention
riders off (route not loaded), two interleaved runs, then two more runs of the unified arm:

| Cell | separate | unified |
|---|---:|---:|
| c64 RTFx | 1384 / 1400 | 1418 / 1456 / 1456 / 1443 |
| c256 RTFx | 1506 / 1532 | 1604 / 1572 / 1586 / 1559 |
| 250 req/s within 1 s | 98.7 % / 98.6 % | 99.1 % / 93.6 % / 96.8 % / 99.5 % |
| saturation, audio s/s | 1880 / 1837 | 1937 / 1974 / 1961 / 1968 |

Gate WER 0.03826 at c1 and c16 in every run, 0 faults. With the larger-rung packet under the unified
launch c256 reaches 1663 / 1713 and one run holds 300 req/s at 95 % within 1 s, but the other run
collapses at 250 req/s (24 % within SLO, p99 5.1 s): not adopted until that is explained.

Per-op table of the decoder (registry ceilings, not calibrated on this box):

| Program | measured | floor | reading |
|---|---:|---:|---|
| prefill, 512 rows | 5.5 ms | 2.1 ms | 10 launches per layer at 12-50 us each: attention 49 us (floor ~2), q / k+v / o projections 13-14 us each (floor 4.4), gate+up 42 (26), down 26 (13), head norm + RoPE 21, residual + norm 2 x 15, GLU 12 |
| prefill, 96 rows | 3.7 ms | 1.2 ms | the same segments at nearly the same cost: launch-latency bound |
| decode, 1 row | 2.0 ms | 1.0 ms | GEMVs at 65-110 % of their floors; 0.57 ms is the instruction walk (285 instructions) |
| decode, 16 rows | 2.3 ms | 1.1 ms | as above; walk 0.59 ms |
| decode, 128 rows | 3.2-3.4 ms | 1.9 ms | library-routed rung, not swept per op |

The k|v head-norm fusion does not apply to this layer shape (q and k are normed, v is not).

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
