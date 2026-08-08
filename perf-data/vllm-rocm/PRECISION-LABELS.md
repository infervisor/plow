# The `_bf16_` in these filenames is NOT a measured precision. Read this before quoting them.

`scripts/bench_vllm_rocm.sh:64` builds the filename from `QTAG`, which defaults to the *env var*
`QUANT` (default `bf16`) — it is **not** derived from what the server actually loaded. And
`scripts/bench_vllm_all.sh:36-37` runs the two large models with **`--dtype auto`**, not
`--dtype bfloat16`. `--dtype` sets the compute dtype anyway; the WEIGHT precision comes from the
checkpoint's own `quantization_config`.

Verified against the checkpoints on disk, 2026-07-27:

| CSV name says | model | `quantization_config` | tensor dtypes on disk | **what actually ran** |
|---|---|---|---|---|
| `bf16_tp1` | gemma-4-12B-it | none | BF16 | **bf16** ✅ label correct |
| `bf16_tp1` | gemma-4-26B-A4B-it | none | BF16 | **bf16** ✅ |
| `bf16_tp1` | gemma-4-31B-it | none | BF16 | **bf16** ✅ |
| `bf16_tp4` | **GLM-5.2-FP8** | `quant_method: fp8`, `fmt: e4m3`, `weight_block_size: [128,128]`, `activation_scheme: dynamic` | **F8_E4M3 426 + F32 426** (scales) | **block-fp8 w8a8-dynamic** ❌ **NOT bf16** |
| `bf16_tp4` | **Kimi-K2.7-Code** | none in config.json | **I32 2304 + BF16 1165** | **int4-packed experts + bf16 attention** ❌ **NOT bf16** |

## Consequences

1. **The three Gemma rows are sound** and are the only clean bf16-vs-bf16 baselines here. Every
   apples-to-apples claim in this campaign rests on those.
2. **GLM-5.2's baseline is an fp8 run.** An earlier note in the design notes claimed the
   opposite — that "GLM-5.2-FP8 is a model name, not a run precision, so that row is bf16." That
   was wrong in the other direction: the name *and* the run are both fp8. Compare plow's GLM
   numbers against it as **fp8 vs fp8**, never as fp8 vs bf16.
3. **Kimi's baseline is an int4 run**, so it is not comparable to a plow bf16 or fp8 Kimi number,
   and it is *also* not comparable to AMD's published Kimi figure, which is **MXFP4** — a
   different 4-bit encoding (OCP microscaling, E8M0 block scales) from this checkpoint's packed
   int4. Matching AMD's number requires AMD's MXFP4 checkpoint, not this one.
4. **A vLLM fp8 baseline for Gemma-4-31B IS possible — via a pre-quantized checkpoint.**
   *On-the-fly* `--quantization fp8` fails: `hipblasLtMatmul` returns
   `HIPBLAS_STATUS_INTERNAL_ERROR` on the down_proj shape (m=5376, n=320, k=21504) during graph
   capture, reproduced under default, `VLLM_ROCM_USE_AITER=1`, and `+AITER_LINEAR+TRITON_GEMM`,
   and also with capture sizes capped at 256.

   But **`RedHatAI/gemma-4-31B-it-FP8-block` serves cleanly** — healthy in 165 s, answers "Paris"
   correctly. It is `compressed-tensors` / `float-quantized`, and vLLM runs it through the
   **W8A8 Block FP8** path — the *same* block-fp8 format plow uses, so it is a genuinely
   format-matched fp8-vs-fp8 pair rather than an approximate one.

   **Disclose this handicap whenever quoting it.** vLLM logs, for every projection shape:

   > `Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file
   > not found at .../N=43008,K=5376,device_name=AMD_Instinct_MI355X,dtype=fp8_w8a8,block_shape=[128,128].json`

   vLLM ships **no tuned block-fp8 kernel config for MI355X** and falls back to defaults. So a plow
   win against this baseline is a win against an *untuned* opponent, and must say so. If a
   like-for-like fp8 number matters, also run `RedHatAI/gemma-4-31B-it-FP8-dynamic` (per-tensor,
   more likely to hit tuned kernels) and report both — the block variant for format-match, the
   dynamic variant for vLLM's best foot forward.

## Fix for future runs

Have the harness read the precision back from the *server* (`/v1/models`, or the engine's startup
log line `Selected ...LinearKernel for ...`) and name the file from that, rather than from an env
var the caller may never have set.

Image, for the record: `rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0`
(vLLM `0.23.1.dev1+g9ddef7117`), which as of 2026-07-27 **is** the newest CDNA tag on Docker Hub —
the `latest` tag is older (2025-12-12).
