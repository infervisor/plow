# TTS bring-up tracker: Veena (maya-research/Veena) + Chatterbox (ResembleAI/chatterbox)

Started 2026-09-26. Box: 4x H200 SXM (sm_90a). User asked for "H100"; the part present is H200,
so every number here is an H200 number and does not transfer to H100.
All GPU runs go through `perf-data/tools/gpulease -n 1 <label> ...`.

## Target parameter block (docs/bringup/target.md)

| param | value | source |
|---|---|---|
| $VENDOR | nvidia | nvidia-smi |
| $ISA | sm_90a | IsaLevel for compute 9.0 |
| $GPU | "H200 SXM" (alias h200) | `plowc --list-gpus` |
| $NCU | default (spec sm_count, 132) | |
| $NGPU / $PARALLEL | 1 / tp | both models fit one GPU |
| $MAXCTX | 2048 (Veena: card says 2048 ctx; prompt <=~250 + <=700 audio tokens). T3: 4096 | |
| $TOOLCHAIN | nvcc 12.9 (nix cuda-merged-12.9), driver 590.48 / CUDA 13.1 | |
| $BUILD | `plowc --emit devblob+cubin` (CMake `sm120_cubins`, PLOW_SM90A_CUBIN=ON) | |
| $FEATURES | `plowrt --features cuda` | |
| $BW_BOUND | TODO measure; datasheet 4.8 TB/s is NOT a measurement | |
| $COMPUTE_CEIL | TODO measure | |
| $RESULTS | /root/tts-work/results | |

Environment notes: `nix develop` is unusable here (stale `/homeless-shelter` from another job's
unsandboxed build makes nix refuse local builds; the default shell also wants ROCm TheRock).
`/root/tts-work/cuda-env.sh` replicates the shell's CUDA half from realised store paths.

## Baselines (same H200, gpulease, greedy, bf16)

Veena, 8 fixed EN/HI prompts (scripts/tts/veena_ref.py):

| engine | conc | audio s / s | median RTF | median LM s |
|---|---|---|---|---|
| HF transformers 5.2 eager | 1 | 0.83 | 1.197 | 6.91 |
| vLLM 0.30 (CUDA graphs, FA) | 1 | 4.38 | 0.233 | 1.26 |
| vLLM | 8 | 23.5 | 0.322 | 1.90 |
| vLLM | 32 | 85.0 | 0.363 | 2.08 |

Whisper-v3-turbo round trip: median CER 0.026 (n=16, HF+vLLM c1).

Chatterbox stock (chatterbox-tts 0.1.7, torch 2.6), 8 EN prompts: median RTF 0.419,
T3 1.26 s (12.7 ms / speech token, CFG batch 2, HF eager loop), S3Gen 0.36 s. CER median 0.000.

## Measured denominators

$BW_BOUND = 4.29 TB/s (4 GiB d2d copy, read+write); read-only reduce 3.78 TB/s.

## Veena rung board (step ms, H200, greedy, raw engine step unless noted)

| rung | vLLM 0.30 | plow base | current (v4) | lever notes |
|---|---|---|---|---|
| 1 | 2.73 | 3.05 | 2.95 (served ITL) | MMA_B1 off -4.1% (landed); skeleton 0.38 ms/step (313 packets x ~1.2 us) |
| 16 | 2.90 | 4.31 | 4.03 served / 3.89 raw | GF=3 flash 39->13 us/layer (landed); kv-hnr fuse -0.8% (opt-in) |
| 32 | 3.18 | n/a (ladder 16) | pending v5 | GV_MM_MAX 64 overflows 48 KB static smem |

Killed/parked: cuBLASLt decode (+30% c1), half-warp WPR hd128 (+4% c16).
Prefill (TTFT p50 @60/512 tok): 56.7/105.8 -> 5.8/9.2 ms (cuBLASLt prefill incl. unfused gate/up).

## End-to-end speech (same client, streaming, H200)

| conc | vLLM+SNAC audio s/s / RTF / TTFA p50,p90 | plow v4 |
|---|---|---|
| 1 | 4.02 / 0.248 / 72,76 ms | 3.83 / 0.261 / 92,93 ms |
| 8 | 14.12 / 0.472 / 162,2298 ms | 16.88 / 0.422 / 206,317 ms |
| 32 | 34.57 / 0.852 / 2404,2613 ms | 29.27 / 0.885 / 2602,3108 ms (16 slots) |

ASR gate v4: median CER 0.008 (n=40); baseline 0.026.

## Chatterbox

T3 on plow: logits rel-L2 0.006-0.020 vs fp32 (top-1 4/4); 1.69 ms/token at 1 request (stock 12.7),
8 concurrent 50.9 audio s/s. S3Gen native: mel rel-L2 2-5e-4, 30-37 ms/utterance (stock ~285).

End to end on plowrt serve (/v1/audio/speech, full wav, H200, after the asr-nvidia merge):
| conc | audio s/s | RTF | stock chatterbox |
|---|---|---|---|
| 1 | 17.42 | 0.055 | RTF 0.419 (~2.4 audio s/s, single request only) |
| 8 | 49.94 | 0.152 | n/a (no batching) |
ASR gate: median CER 0.000 (n=40), same as stock.
Note: after merging worktree-asr-nvidia, rebuild lean-plow (plow_verify) before emitting.

## Findings

- Veena `tie_word_embeddings: true` but the checkpoint also has `lm_head.weight`, differing from
  `embed_tokens` by up to 0.0105. vLLM ties (uses embed); transformers 5 does not. plow ties.

## H100 SXM (eai-gpu-03-h100, 2026-09-26) — second box, same branch

Toolchain: no nix; /home/lava/tts-work/env.sh (CUDA 12.9 from runfile components, gcc-14 host,
rustup 1.95, elan). Assets /opt/dlami/nvme/lava-tts/assets, results .../results. gpulease for all GPU.

Ceilings (measured): bf16 GEMM 803 TF, TF32 389 TF, fp32 47.6 TF; HBM read 3.15 TB/s, copy 3.05 TB/s.

Baselines: vLLM Qwen3-ASR (73 LS clips, 481 s) WER 3.913%, p50 76.8 ms, p90 131 ms, seq RTFx 76, batch RTFx 1266.
Stock Chatterbox RTF 0.608 (T3 19.2 ms/tok, S3Gen 485 ms).

End to end (same tts_bench client):
| arm | plow | vLLM+SNAC |
|---|---|---|
| Veena c1 stream aps / TTFA p50 | 3.24 / 108 ms | 3.15 / 91 ms |
| Veena c8 stream | 14.96 / 235 (p90 356) | 13.89 / 209 (p90 2102) |
| Veena c32 stream | 26.8 / 2883 (16 slots) | 39.1 / 1186 |
| Chatterbox c1 full RTF | 0.065 (15.5 aps) | stock 0.608 |
| Chatterbox c1 stream TTFA | 105 ms (after T3 yield fix; was 300) | n/a |
| Chatterbox c8 full / stream aps | 41.0 / 39.1 | n/a |
ASR gate CER: plow Veena 0.008, vLLM Veena 0.005, plow Chatterbox stream 0.000.
Qwen3-ASR on plowrt CUDA (encoder.pkt on CudaPacketRuntime + speech object): WER 3.913% (= vLLM),
p50 139 ms, p90 221 ms, seq RTFx 42.8. WS partials ("partials": true) every 1 s of audio.

User rule (2026-09-26): all model support via plowc packets; plowrt runs packets only (CPU threads ok
for host tasks); no model-specific ops/segments in plowrt. Open: SNAC + S3Gen are bespoke .so;
t3.rs/chatterbox.rs/asr qwen host code is model-specific.

## Roofline (H100, 2026-09-26) — measured per op

Ceilings: HBM 3.15 TB/s (read), bf16 TC 803 TF, TF32 389 TF, FP32 47.6 TF.
Method: LM decode by instruction-cap deltas (step_bench, PLOW_DEBUG_MAX_INST); packets
(codec/encoder) by packet_bench cap sweeps or ncu per program launch.

LM decode, B=1, ctx 512 (plow step / vLLM GPU kernel time per step):
| model | step | floor | % | notes |
|---|---|---|---|---|
| Qwen3-ASR decoder 1.7B | 2.30 ms (vLLM 2.12) | 1.11 ms | 48% | skeleton 0.40 ms; layer 59 us vs 32 floor; lm_head 0.203 ms = 97% (vLLM 0.203) |
| Veena 3B | 3.56 ms | 2.04 ms | 57% | skeleton 0.41; layer 100 us vs 64 (GLU 68%, down 62%, qkv 67%, o 55%); lm_head 0.310 = 99% |
| Chatterbox T3 520M (B=2 CFG) | 1.94 ms | 0.33 ms | 17% | latency-bound: skeleton 0.48; ~44 us/layer vs 11 floor |
Per-layer GEMVs: plow ~46 us vs vLLM nvjet ~44 us (Qwen). Attention decode plow 7 us vs FA3 11.3 us.
Tried, no gain: PLOW_GEMV_PREFETCH (claim-ahead L2), PLOW_NV_GATE_SLEEP 0/16.
Veena rung 32 (PLOW_DECODE_BATCH_LADDER=1..32, GV_MM_MAX=32 object): B1 3.60 / B16 4.81 / B32 6.09 ms
(5256 tok/s); needed the dynamic-smem opt-in fix. Serving c32 26.8 -> 33.9 aps (native SNAC).

Qwen3-ASR encoder (390 rows, ncu): 63 ms GPU; DenseGemmF32 79% at 1.6 TF (floor ~0.3 ms bf16 TC,
5.3 ms FP32); vLLM does prefill incl. encoder in 14.4 ms. -> tensor-core GEMM path (in progress).

SNAC codec packet (parity 3-4e-6 vs native): b1.f8 4.29 ms (ConvT 55%, pointwise 31%);
b32.f8 31.8 ms (pointwise 37%, ConvT 23%, snake 18%, binary 11%) -> kernel work in progress.
