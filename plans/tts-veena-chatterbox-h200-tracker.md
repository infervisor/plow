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
8 concurrent 50.9 audio s/s. S3Gen native stage: agent in progress.

## Findings

- Veena `tie_word_embeddings: true` but the checkpoint also has `lm_head.weight`, differing from
  `embed_tokens` by up to 0.0105. vLLM ties (uses embed); transformers 5 does not. plow ties.
