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

### Qwen3-TTS: depth stage vs rows, and what serving is bound by

Wide decode ladder (rungs to 256 rows, live KV). `fish_gen --slots N` isolated frame step, ms
(backbone / depth), and the depth stage per instruction at 128 rows (`--depth-sweep`, 961
instructions per frame):

| rows | backbone | depth | + attention fold | + tensor-core head |
|---:|---:|---:|---:|---:|
| 16 | 2.04 | 5.16 | 4.58 | 4.38 |
| 64 | 2.51 | 10.65 | 9.32 | 8.02 |
| 128 | 2.77 | 18.77 | 16.47 | 12.66 |
| 256 | 3.58 | 33.25 | 29.86 | 22.03 |

| op (count per frame) | us each at 128 rows |
|---|---:|
| Gemv (176) / GemvGlu (80) / GemvQkv (80) | 25.7 / 24.4 / 19.8 |
| GemvF32Out, codebook heads (16) | 288.7 |
| FlashDecode (80) / FlashMerge (80) | 18.5 / 7.3 |
| AddNorm (160) / HeadNormRope (240) / SampleSmallF32 (16) | 3.3 / 2.5 / 18.2 |
| instruction walk (cap 0) | 2520 per frame |

- Attention fold (`PLOW_DEPTH_ATTN_FOLD=1`, 961 -> 641 instructions) and the heads on the decode
  walk's tensor cores (`-DPLOW_NV_HEAD_MMA=1`: fp32 accumulate, the same bf16 rounding of the
  logits): depth -33 % at 128 and 256 rows. Sampled codes of 48 frames x 16 codebooks are equal to
  the dot-product heads at 6 of 7 widths (one flip at 16 rows).
- Projections: a block pulls ~20 GB/s of operands, so a projection has to use every block. The
  row-block walk does and re-reads the activations per block (132 x 256 KB at 128 rows = its
  24 us); a weight-tile-owner variant on wgmma is bit-identical (wgmma k16 and mma.sync m16n8k16
  agree when each MMA gets the same 16 elements in the same accumulation order) but uses 8-16
  blocks and is slower below 256 rows; the stream-K arm with its byte floor lowered is a wash
  (15.8 vs 16.5 ms at 128 rows): its park / wait / reduce costs what the K split saves. Open: a
  K split with a cheaper reduce.

Served, same lease per row group (branch binary, live KV, closed loop c32 / c128 and open loop
30 / 40 req/s; audio s/s, first audio p50 at 30 req/s):

| packet | c32 | c128 | 30 req/s | 40 req/s |
|---|---:|---:|---|---|
| wide ladder | 88 | 106 | 123 (123 failed), 4.6 s | - |
| + attention fold | 99 | 106 | 142, 2.1 s | - |
| + tensor-core head | 104 | 125 | 150, 0.56 s | 146 (362 failed) |
| + codec wgmma knobs | 106 | 142 | 153, 0.11 s (p99 0.51 s) | 170 (79 failed) |

Recipe gate on the last row (Whisper CER, c1 / c32 / c128, 208 clips): median 0.

The codec is the bound from c64 up. Its launches run serialized with the talker tick; a stream
steps every 17 frames on its own phase, so a tick carries one or two launches of ~5 streams x 17
frames. Device ms per audio frame: talker 0.19 at 64 rows, 0.12 at 128 (fold + head), 0.10 at 256;
codec 0.21 at the served launch sizes, 0.12 at 32 x 17. The reference's 371 audio s/s is 0.2 ms per
frame all-in.

Codec step per instruction (`packet_bench --sweep`, released packet): 1 x 1 4.5 ms (registry
roofline 0.16), 64 x 1 13.6 (2.6), 64 x 17 133 (33). At 1 x 1 the cost is weight traffic through
the 8-24 blocks an op's column tiles give it (f32 weights): `dec.in` 1024 -> 1536 k7 565 us,
768-channel k7 115-143 us x 3, the transformer's pointwise convs 41 x ~25 us. At 64 x 17 the
96-channel block is 50 of 133 ms (ConvT 192 -> 96 11.0, k7 96 -> 96 3 x 7.5, output conv 4.3).

Single-block screen of the k7 residual-unit conv (fused SnakeBeta output, f32 in / out), us per
launch on 132 blocks; "taps" is the wgmma taps tile the codec knobs select (split-bf16, three
products per k16):

| channels | streams x frames | wide tile | taps tile | staged-once bf16 |
|---:|---|---:|---:|---:|
| 96 | 64 x 1 | 446 | 340 | 89 |
| 96 | 8 x 17 | 908 | 694 | 173 |
| 192 | 8 x 17 | 616 | 615 | 273 |
| 384 | 8 x 17 | 553 | 554 | 271 |
| 768 | 8 x 17 | 525 | 526 | 227 |
| 768 | 1 x 1 | 129 | 130 | 51 |

Staged-once bf16: the tile's input rows are converted to bf16 and staged once (one 128-byte-swizzled
plane per 64 channels); every tap is the same staged rows at a row offset (the swizzle is a
function of the absolute shared address, so a plain descriptor at the offset row is correct: 5.7e-7
against a bf16-operand oracle; the descriptor's base-offset field must stay 0); one product per
k16; the 96-channel weight (132 KiB as bf16) stays resident in shared memory, wider ones stream
their bf16 half one k-tile at a time. Against the f32 reference the relative error is 2.2e-3
(operand rounding). Not in the interpreter yet.

### Codec in the interpreter: bf16 taps arm, stream fusion, per-packet object

Codec step, one persistent interpreter launch, us (`packet_bench`, median of 20, 2 passes):

| packet / object | instructions | b1 x f1 | b8 x f17 | b64 x f17 |
|---|---:|---:|---:|---:|
| f32 baseline (split-bf16 taps tile) | 244 | 3930 | 17327 | 113459 |
| bf16 taps arm v1 (188 KiB arena, B resident) | 244 | 3926 | 15140 | 95794 |
| fused stream convs, v2 arm (130 KiB arena, B streamed) | 218 | 4118 | 15763 | 99806 |
| + compiled variants of this packet only | 218 | 4108 | 15495 | 97924 |
| + one A staging per row tile | 218 | 4109 | 14601 | 91815 |
| + branch-free tail k-tile (no wgmma serialization) | 218 | 3866 | 13869 | 87290 |
| + 16 staging loads in flight per thread (was 4) | 218 | 3890 | 13548 | 83950 |

- Stream fusion: the residual unit's StreamTail gather and input SnakeBeta move into the conv's
  staging; the state update is one StreamTail store from (state, new rows). StreamTail + Unary per
  b8 x f17 step 3171 -> 1163 us. Fused output and state equal the unfused chain bit for bit.
- The 188 KiB arena made the bf16 convs fastest but moved the carve-out and slowed every other
  speech arm; the arm stays inside the 130 KiB arena.
- Per-packet object: the speech object compiles only the conv arms (`PLOW_SPEECH_CONV_ARMS`) and
  bf16 bodies (`PLOW_SPEECH_CONV_BF16_VARIANTS`) the packet's instructions reach, on top of the
  op and epilogue masks; the loader refuses a packet that needs an arm the object lacks. Dropping
  unused out-of-line arms cut STACK 7368 -> 5656 and the cubin 10 -> 6.8 MB with no change in step
  time (REG stays 255 at one block per SM); compiling only the used inlined bf16 bodies gained 1.7 %.
- The flat plan's last k-tile used to break out of the k16 loop between wgmmas. ptxas then
  serialized every wgmma in the speech object (C7520, a function-wide verdict), so ConvT and the
  k1 convs lost time too. The tail groups now read zero B past K: -5 %, CER gate PASS.
- A bulk-copy B ring (mbarrier full/empty, thread 0 producing, up to 6 stages) was 8 % slower
  than the per-thread `cp.async` double buffer and is not kept. B transport is not the limit.
  Probes on the 96-channel k7 conv (op test `--bench-conv-bf16`, 8 x 32646 rows, 286 us)
  attribute ~7 us to B copies, ~95 us to A staging, ~113 us to MMA (~300 TFLOP/s, one k-tile
  in flight) and ~43 us to the store. All four run back to back within a unit.
- Ping-pong warpgroups (each its own 64-row units, A, B and named barrier; the same bits) cut the
  standalone 96 / 192-channel convs 21 / 29 %. Inside the interpreter the step was 3 % slower. The
  codec's convs stage the input SnakeBeta, which the 64-row halo repeats ~30 % more often, and the
  unchanged 384 / 768-channel convs lost 10-18 % to register allocation in the one shared function.
  A second inlined body of the arm put the function over its registers, and ptxas serialized every
  wgmma (C7512). Standalone timing of this arm does not predict its speed inside the interpreter.
- With the input SnakeBeta in staging (`--bench-conv-bf16` "pre"), the standalone convs are
  96: 264-285 us, 192: 354-364, 384: 326-342, 768: 374-399; ping-pong 214-236 / 254-276 for 96 /
  192. In the packet the same convs take 371-482 us. The interpreter's shared code shape costs
  them 30-40 %, which points to running them as their own code object.
- Split screen (`packet_exec.rs` `speech_conv_node_split_matches_packet`, gpuq only): the 14 bf16
  convs of a codec program run on their own kernel entry (`sp_wg_conv1d` alone, 256 threads,
  same 130 KiB claim and plan) between interpreter spans. All tensors are bit-identical to the
  packet's run. Codec step us:

  | program | packet | split, conv as interpreter span | conv entry, same body | conv entry, ping-pong |
  |---|---:|---:|---:|---:|
  | b1 x f1 | 3922 | 4182 | 3994 | 3988 |
  | b8 x f17 | 13598 | 14014 | 12739 | 12349 |
  | b64 x f17 | 83996 | 84447 | 75026 | 72295 |

  The screen's spans are serial chains (no cross-instruction overlap), so the split column is an
  upper bound on the cost of splitting. A packet-declared conv object with the program's own
  dependency counters is the production form.
- Routing every wgmma conv arm (81 instructions: the bf16 k7 convs, the k1 split convs, ConvT) to
  the conv entry (ping-pong body), again bit-identical: b1 x f1 3927 -> 4165 us (split only 5018),
  b8 x f17 13589 -> 11636 (-14 %), b64 x f17 83960 -> 65510 (-22 %). Small programs lose to the
  extra launches, so routing has to be chosen per program.
- Biggest instructions left at b8 x f17: the 96 -> 1 output conv (577 us, staged direct arm), the
  k7 convs (410-545 us each, a ~70 us roofline at 96 channels), ConvT (420-430 us).

### Codec convs on a conv object (production form of the split screen)

- Devgen (`PLOW_EMIT_CODEC_CONV_OBJECT` = minimum million MACs) isolates each large wgmma conv
  (flags bit 17: the bf16 k7 convs, the k1 split convs, ConvT) into its own raw segment. Stream
  order replaces its counter edges. plowc then builds `interp_sm90a_speechconv.cubin`, whose kernel
  entry `speech_conv_node` runs that conv over its blocks with the ping-pong bf16 body.
- plowrt launches the entry for every counter-free single-conv segment and the speech interpreter
  for the rest. Without the object those segments run on the interpreter with the same bits.
- First form: the conv object was the speech interpreter compiled a second time with only the
  conv arms. Bits matched, but it was 50-70 % slower. Inside the interpreter function the ping-pong
  body spilled (10300 LDL / 6574 STL). Without ping-pong it equalled the speech object (14109 vs
  14188 us), so a separate register budget for the same body gains nothing. The kernel entry
  holds the body at 254 registers.
- Every tensor is bit-identical to the speech-object-only run of the same packet on programs
  0-18, 27, 30 (`packet_exec.rs` `speech_conv_object_matches_speech_object`). The tts_cer gate
  passes (median 0, n = 80).
- Codec step us (`packet_bench`, median of 20): the unsegmented packet vs the conv object
  (threshold 1000 M MACs, ConvT counted per input row). Isolation alone, on the interpreter, costs
  1-4 %. Small programs isolate nothing:

  | step | f1 before | f1 after | f4 before | f4 after | f17 before | f17 after |
  |---|---:|---:|---:|---:|---:|---:|
  | b1 | 3887 | 3889 | 4116 | 4107 | 4703 | 4608 |
  | b4 | 4213 | 4208 | 4824 | 4727 | 8554 | 7485 |
  | b16 | 5427 | 5304 | 8908 | 7861 | 23404 | 18916 |
  | b64 | 11563 | 10348 | 24062 | 19899 | 84030 | 65445 (-22 %) |

  b8 x f17 13560 -> 11286 (-17 %), b32 x f17 43640 -> 34450 (-21 %).
- Served A/B (m9 vs conv object, same binary, audio s/s): c32 101.7 -> 126.0, c64 137.6 -> 146.1,
  c128 150.6 -> 165.9, c256 175.2 -> 176.4. Median latency r20 1.18 -> 0.92 s, r30 1.84 -> 1.56 s.
  No faults.
- With the codec cheaper, the 256-row talker ladder pays off. Rungs to 64 vs to 256, conv object
  on both:

  | ladder | c32 | c128 | c256 | c512 | 40 req/s |
  |---|---:|---:|---:|---:|---:|
  | to 64 rows | 126.3 | 150.4 | 181.0 | 176.7 (699 of 1024 rejected) | 193.3 |
  | to 256 rows | 118.5 | 162.3 | 201.6 | 221.7 (0 rejected) | 198.3 |

- Where a c512 cell's 24.4 s of ticks go (256-row ladder):
  - Codec: 15.1 s over 1272 launches, ~57 frames each. That is ~0.21 ms per frame, against 0.06 for
    a full 64 x 17 launch.
  - Talker: 192-row frames take 39 ms served, against ~26 ms isolated at 256 rows.
  - Interpreter walk: a codec step with no instruction executed (`--sweep` cap 0) costs 1.40 ms, about
    6.4 us per instruction. On a 1 x 17 step the convs add 2.10 ms and the other 141 instructions 1.09 ms.
- Codec scheduling knobs at that point (audio s/s, c128 / c256 / c512):
  - Default: 170 / 190 / 210.
  - `PLOW_TTS_STREAM_DEADLINE=0`: 174 / 179 / 227.
  - `PLOW_TTS_CODEC_HOLD_MS=60`: 76 / 77 / 169.
  - Holding steps to gather larger launches stalls the talker more than it saves; neither knob is the
    lever. Open:
    - fuller codec launches without holding the talker (codec beside the talker tick, not serial with it);
    - the served talker frame vs isolated (39 vs ~26 ms).

### Where c512 throughput stops (admission, codec padding, rung controller)

- KV admission: each Qwen3-TTS slot commits its full 2048-row KV (224 MiB), and the codec's
  b128 / b256 x f17 step capacities sized a 29 GiB arena. Capping codec steps at 1088 rows
  (`PLOW_EMIT_FISH_CODEC_MAX_ROWS=1088`: those two capacities ran 2 and 0 times at c512) cuts the
  codec to 8.9 GiB and lifts the KV budget 36.7 -> 56.9 GiB (163 -> 260 sequences). Throughput did
  not move: memory was not the limit.
- Realtime admission (`PLOW_TTS_REALTIME=0` A/B): c512 198.7 aps either way; the best open-loop
  cell with it off is 257 aps with stalls. The cap is GPU time per frame (~0.35 ms: codec ~0.18,
  talker ~0.17 at 80-100 rows per tick).
- Codec steps pay their full padded shape: b16 x f17 is 18.9 ms with every stream carrying 1, 2,
  4, 8 or 17 new frames, b64 x f17 65.4 ms (`packet_bench`, `PB_STEP_FRAMES`). Served launches
  mixed the 2, 2, 4, 8, 16, 17 step ramp and padded every job to the longest.
- `PLOW_TTS_STREAM_STEP_FRAMES` caps a stream's later steps (runtime, opt-in). Equal short steps
  pad less: codec ms per frame at c512 0.231 (packet cap 17) -> 0.189 (4) -> 0.179 (2). Gaps drop
  sharply (c512 544 -> 98, 30 req/s first audio p90 344 -> 109 ms); peak throughput stays ~200.
- The decode rung controller narrowed 256 -> 128 with 200+ requests queued, and at 40-60 req/s
  to 16-48 rows. Its throughput seat compares rungs by tick service time, and the codec's work
  inside a tick lags the rows that caused it, so narrow rungs read cheap. With a coupled stage
  the seat now follows demand only (step cap 2, 4 vs 4 runs, audio s/s):

  | binary | c128 | c256 | c512 | 40 req/s | 50 req/s | 60 req/s |
  |---|---:|---:|---:|---:|---:|---:|
  | throughput seat | 170.5 | 174.2 | 212.4 | 197.6 | 226.9 (20 refused, p99 first audio 6.3 s) | 229.3 |
  | demand seat with a coupled codec | 176.5 | 177.7 | 223.3 | 197.3 | 234.5 (0 refused, 3.6 s) | 234.5 |

  Rung switches 359 -> 46 over the run. No faults.
- Open: GPU time per frame. The codec pays for padded frames (ragged step rows) and a ~3.9 ms
  floor per launch; the talker runs 80-100 rows per tick at ~0.17 ms per frame.

### Codec attention on a slot-indexed KV ring (Qwen3-TTS codec)

The Qwen3-TTS codec's step programs now take the KV ring the Mimi codec already had
(`PLOW_CODEC_RING_KV=1` at emit). A step of up to 4 frames attends to each layer's ring in place.
A longer step gathers the ring first. Every step then stores its new rotated K | V rows into the
ring, so the per-layer carry gather / store pairs are gone.

- Accuracy (`codec_stream_check`, 160 frames, 16 streams sharing launches): stream-vs-whole rel-L2
  and max |d| equal the carry packet's at chunks 2, 17 and random. Batch invariance is
  bit-identical at fixed chunks. At random chunks both packets differ by ~3e-3.
- Step time (`packet_bench`): b64 x f2 13.26 -> 11.31 ms (-15 %), b32 x f4 11.78 -> 10.88 (-8 %),
  b16 x f17 18.89 -> 18.70.
- Served, step cap 2, same binary (audio s/s): c256 201.7 -> 208.7, c512 216.1 -> 233.4 (191 -> 150
  refused), 50 req/s 233.2 -> 239.5 (median latency 5.7 -> 4.2 s), 60 req/s 235.7 -> 245.1 (365 ->
  176 refused). No faults.
- Recipe tts_cer gate: PASS, median CER 0 over 80 clips.
- Where the time goes at 40 req/s (199 audio s/s, 50 s): codec launches 21.4 s, ticks 41.7 s with
  the codec interleaved, so the talker is ~20 s (~11 ms per tick at ~100 rows, isolated speed).
  2620 codec launches average ~47 frames; the ~3.9 ms per-launch floor is about half the codec's
  time.

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
