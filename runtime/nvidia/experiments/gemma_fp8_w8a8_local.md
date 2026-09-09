# CTA-local FP8 decode experiment

`gemma_fp8_w8a8_local.cuh` extends the [separate-quantization screen](gemma_fp8_w8a8_decode.md). It computes each token's scale once per instruction CTA and quantizes BF16 activation panels directly into shared memory. Weights remain E4M3 with per-channel scales. Accumulation promotes to FP32 after each K128 panel. This changes activation precision relative to W8A16.

The kernel preserves blocked output ownership, including combined Q/K/V's 66/33/33 slices in one 132-CTA launch. It requires no global activation buffer or separate quantization launch. Quantization repeats across instruction CTAs and GLU output tiles. Ordinary shared stores use an explicit async-proxy fence before WGMMA consumption.

The harness's mode 3 is this candidate. It must produce identical BF16 output bits to mode 2 (separately quantized, promoted FP8), in addition to existing ownership, guards, finiteness, and sampled quantized-input FP64 oracle checks. Both modes have intentional numerical differences from the BF16-activation control.

Build on the H100 host:

```sh
nix develop -c env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  /usr/local/cuda/bin/nvcc -gencode=arch=compute_90a,code=sm_90a \
  -O3 -std=c++17 -Xptxas=-v -I runtime/common -I runtime/nvidia \
  runtime/nvidia/experiments/gemma_fp8_w8a8_decode.cu \
  -o /opt/dlami/nvme/tmp/gemma31-glm53-h100-20260908/bin/gemma_fp8_w8a8_local
```

The initial build failed because nvcc rejected a variable-template expression immediately adjoining the kernel-launch closing brackets. A local `constexpr` fixes that syntax. Both logs are retained as `gemma-fp8-w8a8-local-build.log` and `gemma-fp8-w8a8-local-build2.log` in the campaign directory.

Compilation succeeds with 52–64 registers for GEMV, 79/93 for B16/B8 GLU, and no spills or stack. Dynamic shared memory is 35/37 KiB for B8/B16 GEMV and 67/69 KiB for GLU, plus 64 bytes of scales. These are standalone resources; subsequent interpreter measurements are recorded in the [opt-in integration notes](gemma_fp8_w8a8_persistent.md).

## H100 screen, 2026-09-09

All 117 correctness checks passed, including all 39 CTA-local checks. Every owned output bit from CTA-local quantization matched the separate-quantization promoted reference. The checks cover exact integer inputs, M7/8/15/16, N tails, partial K272, empty slices, output guards, and combined Q/K/V in the full 132-CTA grid. Large-shape inputs and weights are synthetic. FP8 activation quantization still differs from BF16 input: relative L2 is 2.55–2.59% for GEMV and 3.60–3.68% for GLU; the worst sampled quantized-input FP64 oracle relative L2 is 0.0011028903.

Median microseconds, four warmups and 15 timed repetitions, with a 256 MiB cache flush before each measurement. Both FP8 columns include activation quantization. The GPU and CPU build resources were reserved for this screen.

| Batch | Shape | Slices | W8A16 | Separate quantization | CTA-local |
|---:|---|---:|---:|---:|---:|
| 8 | Q 8192×5376 | 66 | 247.072 | 52.096 | 45.216 |
| 8 | K/V 4096×5376 | 33 | 252.224 | 45.024 | 48.704 |
| 8 | Combined Q/K/V | 66+33+33 | 248.160 | 61.216 | 53.792 |
| 8 | O 5376×8192 | 132 | 180.864 | 63.808 | 97.376 |
| 8 | GLU 21504×5376 | 132 | 370.816 | 131.840 | 139.552 |
| 8 | Down 5376×21504 | 132 | 492.096 | 159.168 | 238.816 |
| 16 | Q 8192×5376 | 66 | 264.448 | 63.904 | 80.576 |
| 16 | K/V 4096×5376 | 33 | 267.296 | 56.896 | 78.656 |
| 16 | Combined Q/K/V | 66+33+33 | 262.240 | 73.120 | 82.752 |
| 16 | O 5376×8192 | 132 | 191.616 | 83.360 | 107.200 |
| 16 | GLU 21504×5376 | 132 | 431.168 | 148.000 | 186.176 |
| 16 | Down 5376×21504 | 132 | 526.944 | 211.840 | 263.904 |

CTA-local is 1.79–5.46× faster than this W8A16 control. Separate quantization is faster in 10/12 cells. CTA-local removes a launch and global scratch, but repeats reduction and quantization work; the B8 Q and combined-QKV cells are its only wins over separate quantization in this screen. These are standalone kernel results, not model or serving speedups.

Compute Sanitizer memcheck exited zero with `ERROR SUMMARY: 0 errors`, covering all exact/tail cases and the large B16 GLU case in every mode. Combined QKV passed the ordinary correctness screen but was not included in this bounded sanitizer run. Sanitizer timings are discarded.

Run the screen and bounded sanitizer check:

```sh
nix develop -c env \
  LD_LIBRARY_PATH=/opt/dlami/nvme/tmp/gemma31-glm53-h100-20260908/cuda-driver-lib:/usr/local/cuda/lib64 \
  /opt/dlami/nvme/tmp/gemma31-glm53-h100-20260908/bin/gemma_fp8_w8a8_local
nix develop -c env \
  LD_LIBRARY_PATH=/opt/dlami/nvme/tmp/gemma31-glm53-h100-20260908/cuda-driver-lib:/usr/local/cuda/lib64 \
  /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 86 \
  /opt/dlami/nvme/tmp/gemma31-glm53-h100-20260908/bin/gemma_fp8_w8a8_local \
  16 21504 5376 132 1
```

Evidence in the campaign directory: `gemma-fp8-w8a8-local-screen.{log,json}`, `gemma-fp8-w8a8-local-memcheck.log`, and `gemma-fp8-w8a8-local-proof.json`. The original separate-quantization binary, source snapshot, and proof remain unchanged.

## Integration decision

CTA-local remains the smaller opt-in interpreter experiment: reuse the current instruction ownership and arena export, with no new tensor lifetimes, dependencies, or launches. Its wins over the W8A16 control justify measuring actual model execution before a broader compiler change. Start with a shape-gated Hopper B8/B16 route, preserve B1/B4 fallbacks, and measure the enlarged arena/register effect on those fallbacks. Full-logit and model-quality gates remain necessary because activation precision changes.

Separate quantization is the stronger performance direction for most tested shapes, but needs a real shared quantized tensor and scale lifetime in the decode instruction graph. Reusing one quantized activation across Q/K/V avoids duplicating work. A separate launch cannot be inserted inside the persistent device body; use explicit existing `QuantFp8` graph dependencies and matching FP8 consumers if pursuing this route. This is a larger compiler/runtime experiment and is not implemented here.

The subsequent [interpreter integration](gemma_fp8_w8a8_persistent.md) is explicitly experimental and default off.
