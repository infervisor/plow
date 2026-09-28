# Attention kernels on H100: roofline

Every flash kernel plow serves on sm_90a, at the shapes it serves, against its roofline and
against FlashAttention-3 (vLLM 0.28 `vllm_flash_attn`, `fa_version=3`) and FlashInfer 0.6.16 at
identical shapes.

* Roofline = max(FLOP / 989 TF/s, bytes / 3.35 TB/s). Decode bytes = distinct K+V bytes read
  (sliding layers: the 512-row window). Prefill bytes = Q + K + V + O.
* Standalone timing: a CUDA graph of back-to-back calls over rotating, distinct K/V buffers
  (> 160 MB in total, so every call reads cold KV like consecutive layers), time / calls. plow's
  kernels run the shipped device bodies in a 132 x 256 persistent grid, as the interpreter does.
  Tools: `runtime/nvidia/experiments/fa_decode_bench.cu` (decode), `fa_v3_bench.cu` (prefill),
  `/home/lava/tts-work/attn/ref_attn.py` (FA3 / FlashInfer, same method).
* Geometries. E4B: 8 q / 2 kv heads, hd256 sliding (window 512) on 35 layers, hd512 full on 7;
  GQA 4 (one work item reads a KV row once for 4 heads). Veena (Llama-3.2-3B): hd128 24/8. Qwen3-ASR
  decoder: hd128 16/8. Chatterbox T3: hd64 16/16.

## Decode (one query row per sequence)

Baseline = HEAD 5eea79c4 bodies with the packet's own nsplit and a separate FlashMerge. The
shipped state after the attn_1 patch is under [After](#after-attn_1).

| layer | B | ctx | KV MB | floor us | plow us | % | FA3 us | FlashInfer us |
|---|---|---|---|---|---|---|---|---|
| hd256 sliding | 1 | any | 1.05 | 0.31 | 8.6 | 4 | 11.8 | 10.6 |
| hd256 sliding | 8 | any | 8.4 | 2.5 | 18.1 | 14 | 14.9 | 13.3 |
| hd256 sliding | 32 | any | 33.6 | 10.0 | 31.1 | 32 | 23.9 | 25.9 |
| hd256 sliding | 64 | any | 67.1 | 20.0 | 35.5 | 56 | 39.4 | 38.0 |
| hd256 sliding | 128 | any | 134 | 40.1 | 66.5 | 60 | 68.2 | 62.1 |
| hd512 full | 1 | 1024 | 4.2 | 1.25 | 15.2 | 8 | - | 12.7 |
| hd512 full | 1 | 8192 | 33.6 | 10.0 | 25.9 | 39 | - | 24.8 |
| hd512 full | 8 | 1024 | 33.6 | 10.0 | 28.6 | 35 | - | 22.0 |
| hd512 full | 8 | 8192 | 268 | 80.1 | 169.9 | 47 | - | 97.5 |
| hd512 full | 32 | 1024 | 134 | 40.1 | 90.9 | 44 | - | 53.9 |
| hd512 full | 32 | 8192 | 1074 | 321 | 640 | 50 | - | 351 |
| hd512 full | 64 | 1024 | 268 | 80.1 | 103.4 | 78 | - | 98.1 |
| hd512 full | 128 | 1024 | 537 | 160 | 202 | 79 | - | 177 |
| hd512 full | 128 | 8192 | 4295 | 1282 | 1474 | 87 | - | 1356 |
| hd128 Veena (fold) | 1 | 1024 | 4.2 | 1.25 | 9.2 | 14 | 13.0 | 10.0 |
| hd128 Veena (fold) | 32 | 1024 | 134 | 40.1 | 64.5 | 62 | 55.6 | 52.2 |
| hd128 Veena (fold) | 128 | 1024 | 537 | 160 | 200 | 80 | 187 | 181 |
| hd128 Qwen3-ASR (fold) | 32 | 1024 | 134 | 40.1 | 58.4 | 69 | 55.3 | 52.2 |
| hd128 Qwen3-ASR (fold) | 128 | 1024 | 537 | 160 | 186 | 86 | 187 | 181 |
| hd64 T3 | 32 | 1024 | 134 | 40.1 | 54.9 | 73 | 58.6 | 56.0 |
| hd64 T3 | 128 | 1024 | 537 | 160 | 191 | 84 | 195 | 188 |

FA3 has no hd512 kernel; vLLM serves E4B's full layers on its Triton backend.

## Prefill

| layer | rows (requests) | GFLOP | roofline us | plow v3 us | % | FA3 us | FlashInfer (fa3 / fa2) us |
|---|---|---|---|---|---|---|---|
| hd256 sliding | 1000 (1) | 4.2 | 4.2 | 37.7 | 11 | 33.2 | 23.3 / 39.6 |
| hd256 sliding | 2000 (1) | 7.3 | 7.4 | 40.4 | 18 | 27.3 | 25.7 / 37.2 |
| hd256 sliding | 1000 (4 x 250) | 1.0 | 3.1 | 43.3 | 7 | 18.5 | 14.3 / 30.4 |
| hd512 full | 1000 (1) | 8.2 | 8.3 | 95.5 | 9 | - | - / 106.7 |
| hd512 full | 2000 (1) | 32.8 | 33.2 | 191 | 17 | - | - / 310 |
| hd512 full | 1 row, kv 1024 | 0.0 | 1.3 | 90.0 | 1 | - | - |
| hd512 full | 1 row, kv 4096 | 0.0 | 5.0 | 342 | 1 | - | - |
| hd256 sliding | 1 row, kv 1024 | 0.0 | 0.3 | 26.0 | 1 | - | - |

(FlashInfer's fa3 backend does not build hd512 here.)

## After (attn_1)

Shipped:

* E-series hd256/512 decode folds the merge into the last split (`PLOW_NV_FA_FOLD_WIDE`, default
  on): no FlashMerge op. The fold tail (`fa_fold_tail`, hd >= 256) computes the split weights once
  and streams float4 columns with 8 split loads in flight. It is `__noinline__`: inlined, it cost
  the row-group body 9 registers and 27% at B=64 (hd512 103 -> 132 us).
* nsplit = one wave, capped at 16 (512-row window) / 33 (full layer) instead of 8.
* v3 prefill packs GQA heads of short requests (qlen x gqa <= 64) into one tile
  (`PLOW_BUILD_FA_V3_PACK=1`, object env): token-batch riders.
* hd128 (Veena, Qwen3-ASR) is bit-identical to HEAD (greedy fnv equal at B=1..128).

Standalone decode, shipped E4B body. The nsplit is the new rule. Rows marked * ran while a 2.7 GB
foreign process held the GPU.

| layer | B | ctx | floor us | before us | after us | % floor | FlashInfer us |
|---|---|---|---|---|---|---|---|
| hd256 sliding | 1 | any | 0.31 | 8.6 | 8.0 | 4 | 10.6 |
| hd256 sliding | 8 | any | 2.5 | 18.1 | 10.7 | 23 | 13.3 |
| hd256 sliding | 32 | any | 10.0 | 31.1 | 21.1 | 48 | 25.9 |
| hd256 sliding | 64 | any | 20.0 | 35.5 | 31.6 | 63 | 38.0 |
| hd256 sliding | 128 | any | 40.1 | 66.5 | 60.0 | 67 | 62.1 |
| hd512 full | 1 | 1024 | 1.25 | 15.2 | 13.7 | 9 | 12.7 |
| hd512 full | 1 | 8192 | 10.0 | 25.9 | 31.6* | 32 | 24.8 |
| hd512 full | 8 | 1024 | 10.0 | 28.6 | 20.7 | 48 | 22.0 |
| hd512 full | 8 | 8192 | 80.1 | 169.9 | 101.3 | 79 | 97.5 |
| hd512 full | 32 | 1024 | 40.1 | 90.9 | 55.2 | 73 | 53.9 |
| hd512 full | 32 | 8192 | 321 | 640 | 376* | 85 | 351 |
| hd512 full | 64 | 1024 | 80.1 | 103.4 | 98.5 | 81 | 98.1 |
| hd512 full | 128 | 1024 | 160 | 202 | 193 | 83 | 177 |
| hd512 full | 128 | 8192 | 1282 | 1474 | 1461* | 88 | 1356 |

hd512 B=1 ctx 8192 regresses because the 33-split cap trades long context for short context. At ns
33 vs 66: ctx 1024 13.9 vs 16.5 us, ctx 4096 21.3 vs 20.5 us, ctx 8192 31.6 vs 26.5 us.

E4B in-model decode step, ms (`step_bench`, 48 steps):

| ctx | B | HEAD | attn_1 | delta |
|---|---|---|---|---|
| 1024 | 1 | 5.876 | 5.735 | -2.4% |
| 1024 | 8 | 6.398 | 6.139 | -4.0% |
| 1024 | 32 | 9.092 | 8.706 | -4.2% |
| 1024 | 64 | 12.335 | 11.518 | -6.6% |
| 1024 | 128 | 16.474 | 15.892 | -3.5% |
| 4096 | 1 | 5.919 | 5.785 | -2.3% |
| 4096 | 8 | 6.840 | 6.391 | -6.6% |
| 4096 | 32 | 10.824 | 9.713 | -10.3% |
| 4096 | 64 | 13.821 | 13.507 | -2.3% |
| 4096 | 128 | 20.418 | 19.857 | -2.7% |

E4B served, `gemma_voice_bench.sh` (ISL 1000 / OSL 128):

| conc | HEAD out tok/s | attn_1 out tok/s | HEAD TPOT p50 ms | attn_1 TPOT p50 ms |
|---|---|---|---|---|
| 1 | 160.9 | 165.1 | 6.03 | 5.89 |
| 64 | 3296 | 3459 | 17.03 | 16.16 |
| 128 | 3688 | 4218 | 29.13 | 24.79 |

Parity, `gemma_logit_parity.py`:
* HEAD: top1 0.9879, KL 7.68e-4.
* attn_1: top1 0.9841, KL 6.87e-4. Greedy fnv is identical to the parity build.

The KV-shared tail still loads with decode_rungs 9.

Riders: v3 packed prefill, 1-row requests, standalone.

| layer | requests x kv | before us | after us |
|---|---|---|---|
| hd256 | 64 x 1024 | 107.6 | 46.6 |
| hd512 | 64 x 1024 | 360 | 115 |
| hd512 | 64 x 4096 | 1360 | 372 |
| hd512 | 1-8 x 1024 | 90 | 90-94 |

With 8 or fewer riders there are fewer items than SMs. That case needs split-KV, which is not done.

### Tried, not shipped (kept opt-in, default off)

* `PLOW_NV_FA_RGM` (`PLOW_NV_FA_MMA_HD128` / manifest `fa_rgm`) is an mma.sync row-group decode
  body. It stages K/V through per-warp cp.async.bulk rings, computes S with m16n8k16 and O^T with
  m16n8k8, and applies to hd128 and hd256.
  * Standalone it reaches FlashInfer:

    | layer | B | RGM us | row-group body us | FlashInfer us |
    |---|---|---|---|---|
    | hd128 GF3 | 64 | 96.9 | 114 | 96.5 |
    | hd128 GF3 | 128 | 184.8 | 224 | 181 |
    | hd256 | 64 | 27.8 | 31.6 | — |

  * In-model it does not pay off:
    * Veena B=64 flash per layer is 113 -> 109 us (cap sweep).
    * Veena step times move by B:

      | B | step ms change |
      |---|---|
      | 128 | -1% |
      | 64 | +0.3% |
      | 8-32 | +2-4% |

    * E4B, with RGM on hd256 and RGT on hd512, regressed at B=64 (+4%).
* `PLOW_NV_FA_RGT` stages the scalar row-group body through the same per-warp bulk rings. hd512
  B=64 ctx 1024 standalone is 95 us, vs 98.5 us for the shipped body.
* Veena B=64 flash in-model is 113 us/layer × 28 = 3.2 ms. That is 71% of the 3.35 TB/s floor
  (80 us) and 81% of the ~2.9 TB/s practical peak.
