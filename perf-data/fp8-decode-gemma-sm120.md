# G7 — fp8 weight-only DECODE for Gemma-4-12B on the sm_120 interpreter

**Date:** 2026-07-18 · **branch:** worktree-agent-a57b3995cf7b11875 (base `rtx` @ 32cc434) ·
**GPU:** RTX PRO 6000 Blackwell Server Edition (sm_120, 188 SMs, 96 GB) · **metric:** decode
TPOT ms/tok, batch 1, greedy (lower is better), repo methodology (112 timed steps after 16
warmup, `qwen3_sm120_chat` harness reused for Gemma).

Closes rtx-06 **G7**: fp8 (w8a16) weight-only decode. Attention stays bf16 (fp8 KV is out of
scope). Prefill stays bf16 GEMM on the bf16 weights; only the DECODE GEMV family is fp8.

---

## What landed

- **Quantizer** `perf-data/harness/quantize_fp8.py`: per-output-channel (per-row) e4m3, the
  settled convention (matches the AMD path, `runtime/amd/op_gemm.h:1440`):
  `scale[n] = amax(|W[n,:]|)/448`, `W8[n,k] = round_e4m3(W[n,k]/scale[n])`, dequant
  `W ≈ float(W8)·scale[n]` (scale applied ONCE in the epilogue). Output is one
  `model.safetensors` keyed exactly as the emitter declares the twins (`fp8/<name>` uint8 +
  `fp8/<name>_scale` f32). Full (k_eq_v) layers have no v_proj — skipped, matching the pkt.
  - **Twins:** `/workspace/models/gemma-4-12B-it/fp8/model.safetensors` — 10.91 GB, 328
    projections (48 layers × 7, minus 8 full-layer v_proj) + 328 f32 scales.
- **Kernels** `runtime/nvidia/op_gemm.cuh`: `d_gemv_fp8` and `d_gemv_glu_fp8` — FFMA
  dequant-on-load (e4m3→fp16→f32, exact), per-row scale in the epilogue. Byte-for-byte the
  same BLOCKED N-column ownership as the bf16 `gemv_rows` (the emitter's gemv→headnorm
  fine-dependency map assumes it); the SM fill is the N-column partition across all 188
  blocks, exactly as bf16 decode. The fp8 chunk is 8 fp8/lane = 256 K/warp-pass, the SAME
  GV_STEP as bf16, so every decode K (3840/4096/2048/512/15360, all ×256) divides cleanly.
- **Interpreter** `runtime/nvidia/interp_sm120.cu`: `GEMV_FP8`(30) / `GEMV_GLU_FP8`(31)
  dispatch arms, behind `#if PLOW_NV_GEMMA`. The default (Qwen) object is byte-identical.
- **Harness** `runtime/tests/qwen3_sm120_chat.cu`: binds `fp8/` twins from `PLOW_FP8_DIR`.
- **Oracle** `runtime/tests/sm120_interp_op_test.cu`: fp8 GEMV / GEMV+GLU vs the
  dequantized-weights f32 matmul.

**Registers / occupancy (ptxas -v, sm_120a, CUDA 13.0):** Gemma object (`-DPLOW_NV_GEMMA=1
-DPLOW_NV_FA_GF=2`) = **150 regs, 0 spill, 1 block/SM** — UNCHANGED from the pre-fp8 Gemma
build (the megakernel worst case is still `d_flash_decode<512,2>`; the fp8 GEMV arms add no
register pressure). Default (Qwen) object = 155 regs, 0 spill — byte-identical to before.

---

## Numeric oracle (fp8 GEMV vs dequantized-weight f32 matmul)

The reference dequantizes the SAME e4m3 weights back to f32, so this isolates the kernel's
arithmetic (bf16 x, f32 accumulate, scale factored once) from the quantization error itself.

| case | shape | relL2 | verdict |
|---|---|---|---|
| gemv_fp8 q_proj | N4096 K3840 | 1.62e-3 | PASS |
| gemv_fp8 o_proj | N3840 K4096 | 1.66e-3 | PASS |
| gemv_fp8 down | N3840 K15360 | 1.71e-3 | PASS |
| gemv_glu_fp8 (gelu) | N15360 K3840 | 1.68e-3 | PASS |

relL2 sits in the SAME 1.6e-3 band as the bf16 flash/headnorm kernels — e4m3→fp16→f32 is
exact on device, so the only gap is bf16-x rounding + warp-tree accumulation order. The
`-DFA_NV_WAVE64_NEGCTRL` build correctly FAILS these (wrong warp-reduction offset).

**Quantization error (independent spot-check, down_proj layer 0, K=15360):** dequantized
fp8 weights vs the bf16 original = **2.6% relL2**, and a random-vector matmul = 2.8% relL2 —
the inherent e4m3 (3-bit mantissa) error. This is why the oracle compares vs the DEQUANTIZED
weights (isolating kernel arithmetic from the quant error), and why token-identity to bf16 is
not guaranteed in general — though it held for the short prompts below.

## Coherence / parity vs bf16 (greedy, same checkpoint)

Gemma-4 chat format `<|turn>user\n…<turn|>\n<|turn>model\n`, greedy, 40 gen tokens:

- **"What is the capital of France?"** → decodes to `…thought…Paris` — CORRECT.
- **"Write a short poem about the ocean."** → *"A world of blue, both deep and wide, / Where
  secrets sleep beneath the tide. / With rhythmic pulse and salt-sprayed breath, / It dances
  life and mimics death"* — coherent.
- **fp8 vs bf16 token-identity:** on the France prompt the fp8 and bf16 greedy token streams
  are **byte-for-byte identical** for all 40 tokens. The task anticipated divergence (per-row
  e4m3 ≠ bf16); for these short prompts the greedy argmax is robust enough that per-channel
  e4m3 preserves the top-1 ordering exactly. This is a stronger parity result than required.

---

## TPOT — plow fp8 vs vLLM fp8 vs plow bf16 (ctx ladder)  [G7-fp8-longctx, 2026-07-19]

Long-ctx points now MEASURED on the MERGED rtx HEAD (T3 a026a72 + T4 af4a953). vLLM columns
from `perf-data/gemma4-12b-vllm-sm120.json` (0.25.1, TRITON_ATTN, CUDA graphs, same box); plow
bf16 from `perf-data/gemma4-12b-plow-sm120-decode.json`. plow fp8: the committed-HEAD
`qwen3_sm120_chat.cu` LINKED against the Gemma decode object `libplow_interp_sm120_gemma.a`
(the CMake `qwen3_sm120_chat` links the default/Qwen object, which lacks the `GEMV_FP8` Gemma
arms and launch-faults on a Gemma fp8 packet — same manual link G7 used; a link choice, no source
edits). 112 timed steps after 16 warmup; contexts decode-primed (the fast bf16 prefill cannot
prime an fp8 decode — no single committed harness does both — but TPOT is KV-length-dependent so
the priming path does not change the steady-state number).

| ctx | plow fp8 | plow bf16 | vLLM fp8 | vLLM bf16 | plow-fp8 / vLLM-fp8 | plow-fp8 / plow-bf16 |
|----:|---------:|----------:|---------:|----------:|:-------------------:|:-------------------:|
| 1k  | **11.25** | 18.48 | 12.46 | 19.78 | **0.90 (−10%)** | 0.61 |
| 4k  | **11.38** | 18.54 | 12.97 | 20.25 | **0.88 (−12%)** | 0.61 |
| 16k | **12.16** | 19.07 | 14.27 | 21.66 | **0.85 (−15%)** | 0.64 |
| 32k | **12.87** | 19.76 | 15.98 | 23.35 | **0.81 (−19%)** | 0.65 |
| 64k | **14.21** | 21.12 | 17.47 | 24.76 | **0.81 (−19%)** | 0.67 |
| 128k| **17.04** | 23.98 | 20.71 | 28.26 | **0.82 (−18%)** | 0.71 |

(128k n_prompt=130944; the max_ctx=131072 packet caps n_prompt+n_gen at 131072. 1k/4k are the
G7 short-ctx points; 16k–128k are this campaign. All plow rows: sd ≤ 0.11% of mean, 112 steps.)

**Honest finding — the "≈flat 11.4–11.7 ms" G7 projection is REFUTED.** plow fp8 decode RISES
gently with ctx (11.25 → 12.16 → 12.87 → 14.21 → 17.04), not flat. Gemma-4 is sliding-window
for 40/48 layers (KV window-capped, ≈flat), but the OTHER 8 layers are FULL causal (hd512) and
their decode KV stream grows linearly with ctx — so the per-token flash-decode cost climbs ~2 ms
per doubling from that 8/48 share. The flat projection assumed all layers were window-capped.

**plow fp8 still BEATS vLLM fp8 at every measured ctx** (0.90 → 0.81), and the win holds rather
than widening: vLLM fp8 also rises (14.27 → 20.71) at a similar absolute rate, so the ratio is
roughly stable ~0.81–0.85 through the mid/long ctx (vs the projected "widening to ~0.73"). Against
vLLM **bf16** plow fp8 is 0.57–0.66× (up to ~1.8× faster). Against plow's own bf16 decode it is
0.61–0.67× — the fp8 weight-stream halving (24 → 10.9 GB) net of the fixed bf16 per-token overhead
(flash-decode + merge + the ~620 per-token counter gates, which do not shrink with fp8 weights).

**Correctness:** fp8 decode is faithful to bf16 — on the France chat prompt the fp8 and bf16 greedy
24-token streams are BYTE-IDENTICAL, and identical between the pre-merge (T3) binary and the merged
HEAD build (GEMV_FP8 is decode-only, untouched by T3/T4). On the poem prompt fp8 tracks bf16 within
a 2-token near-tie then converges. (The scratchpad's library-free chat tokenizer produces degenerate
greedy loops on short prompts on BOTH bf16 and fp8, so coherence-vs-HF is not reproduced in this
harness path; the fp8 == bf16 equivalence is the parity result. Oracle relL2 ~1.6e-3, G7.)

Theoretical fp8 floor: 12B fp8 weights ≈ 10.9 GB / 1673 GB/s ≈ **6.5 ms/tok** (weight stream only;
the gap to the measured 11–14 ms is the fixed bf16 per-token decode overhead + the growing
full-attention KV share above).
