# Opt-in FP8 WGMMA decode

`PLOW_NV_FP8_DECODE_WGMMA=1` enables CTA-local activation quantization and promoted E4M3 WGMMA in the arena-aware FP8 GEMV and GELU-tanh GLU decode paths. It is default off and restricted to Hopper decode, exactly M8/M16, positive K divisible by 16, and 256-thread CTAs. M1/M2/M4, other batch sizes, unsupported K, and other activations retain their existing paths.

The [CTA-local screen](gemma_fp8_w8a8_local.md) supplies the kernel design and standalone evidence. The production helper preserves each instruction's blocked output slice and uses the existing arena. No tensor allocation, instruction, dependency, counter protocol, or launch is added. Activation scales use the existing vLLM-style per-token convention. Activation precision changes from BF16 to E4M3; FP32 promotion occurs after every K128 panel, and GLU retains FP32 gate/up values until the final BF16 store.

The interpreter exports an arena maximum of 70,720 bytes through the existing `plow_arena_bytes` contract. Its original GEMV staging threshold is unchanged, preserving fallback dispatch. Supported multi-row FP8 GEMV explicitly reaches the arena overload even when K exceeds that old staging threshold. Shared memory is reused only after WGMMA completion and the CTA barrier. A compile-time assertion rejects incompatible role flags that would reduce the arena below the required size, including `PLOW_NV_GEMV512_ROLE`.

## Build and isolation

Campaign directory: `/opt/dlami/nvme/tmp/gemma31-glm53-h100-20260908`.

Candidate assets: `fp8-b16-c1024-persistent-wgmma-h100`. Only `interp_sm90a.cubin` is replaced; all other assets reference `fp8-b16-c1024-packed-occ1-h100`. The runnable BF16-weight/FP8-KV checkpoint is unchanged.

Exact build argument arrays are retained in `fp8-w8a8-local-decode-build-command.json`. The control is rebuilt before and after source changes with the canonical asset's same compiler and options. The candidate adds `-DPLOW_NV_FP8_DECODE_WGMMA=1` and diagnostic `-Xptxas=-v`.

| Cubin | SHA256 |
|---|---|
| Canonical, pre-edit control, post-edit default-off control | `85e9f293f60d4d1228844fb52377441412fbe27bc27de8eccb21bf59d03a95c4` |
| Opt-in candidate | `95b2e405a202f328fa457e777eb4fef8609361443be6eb38bdb2708c8be0d32b` |

Default-off cubin bytes are identical. Adding the arena assertion and rebuilding also preserves both candidate and default-off hashes. The candidate interpreter uses 215 registers versus 188, with zero stack/spills/local memory and unchanged 1,040-byte static shared memory. The extra dynamic arena and register allocation apply to the complete decode kernel, including fallback batches; the B1 measurement below captures their combined effect in one model/context cell.

Build the production-dispatch probe:

```sh
nix develop -c env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  /usr/local/cuda/bin/nvcc -gencode=arch=compute_90a,code=sm_90a \
  -O3 -std=c++17 -Xptxas=-v -DPLOW_FP8_W8A8_PERSISTENT_PROBE=1 \
  -I runtime/common -I runtime/nvidia \
  runtime/nvidia/experiments/gemma_fp8_w8a8_decode.cu \
  -o /opt/dlami/nvme/tmp/gemma31-glm53-h100-20260908/bin/gemma_fp8_w8a8_persistent
```

Probe mode 3 calls the actual arena-aware production dispatcher. Other modes retain the standalone W8A16 and separately quantized FP8 controls. Additional cases exercise M1/M2/M4, unsupported K136, and SILU GLU fallbacks.

## Production-dispatch probe

All 171 ordinary checks passed, including 57 mode-3 checks bit-identical to the promoted FP8 reference. Twelve random-input fallback checks also passed bitwise comparison to the existing W8A16 dispatcher. Memcheck exited zero with `ERROR SUMMARY: 0 errors`; coverage includes every exact/tail/fallback case and large B16 GLU in all modes. Combined QKV passed ordinary numerical/guard checks but was outside the bounded memcheck run.

Median microseconds with four warmups, 15 repetitions, a 256 MiB cache flush, and reserved CPU/GPU resources. FP8 columns include quantization. These are standalone production-dispatch timings, not complete model steps.

| Batch | Shape | W8A16 | Separate quantization | Production CTA-local |
|---:|---|---:|---:|---:|
| 8 | Q 8192×5376, 66 slices | 246.624 | 52.608 | 45.760 |
| 8 | K/V 4096×5376, 33 slices | 252.288 | 45.440 | 47.520 |
| 8 | Combined Q/K/V, 66+33+33 | 248.768 | 61.952 | 53.760 |
| 8 | O 5376×8192, 132 slices | 181.856 | 64.096 | 99.648 |
| 8 | GLU 21504×5376, 132 slices | 380.768 | 132.160 | 143.616 |
| 8 | Down 5376×21504, 132 slices | 490.944 | 159.584 | 243.712 |
| 16 | Q 8192×5376, 66 slices | 263.200 | 64.000 | 79.264 |
| 16 | K/V 4096×5376, 33 slices | 268.544 | 56.928 | 76.448 |
| 16 | Combined Q/K/V, 66+33+33 | 261.888 | 73.504 | 82.400 |
| 16 | O 5376×8192, 132 slices | 190.720 | 83.456 | 105.888 |
| 16 | GLU 21504×5376, 132 slices | 431.552 | 147.680 | 190.528 |
| 16 | Down 5376×21504, 132 slices | 525.600 | 211.040 | 258.624 |

Evidence: `gemma-fp8-w8a8-persistent-screen.{log,json}`, `gemma-fp8-w8a8-persistent-memcheck.log`, and `fp8-w8a8-local-integration-build-proof.json`. Sanitizer timings are discarded. Actual model-step timing and full-logit/model-quality qualification are separate gates. No production-default promotion is supported by these probe results.

## Full-model logits

The frozen runtime probe tested natural 1K/16K prompts with 64 teacher-forced frames per case and a 262,144-token vocabulary. M1/M2/M4 fallback captures matched the existing native decoder bit for bit across all 384 frames. All prefill frames remained unchanged.

M8 and M16 produced identical logits to each other across their 128 frames. Against the BF16-activation native decoder, each width matched top-1 in 125/128 frames. All values were finite, but the largest absolute difference was 4.3125, worst cosine 0.9954132, and worst relative L2 0.1604071. Mean reference-to-candidate KL was 0.0022932 for the 1K case and 0.004254 for the 16K case. These changes require broader model-quality evaluation; bitwise agreement with the separately quantized probe does not establish model-quality equivalence to BF16 activations.

Evidence: `fp8-w8a8-persistent-candidate-model-logits-comparison.json`, with candidate cubin hash above and runtime-probe hash `cb569adf9d86015e8ec1afa8823d8a5a924ef9c841ea8ee03bc49667a2dfc360`.

## Actual model-step timing

Six isolated cells used the same frozen `step-bench-profile`, context 1024, 16 warmup steps, and 64 measured steps. No GPU work or CPU builds overlapped. The baseline is the original native FP8-weight/BF16-activation decode cubin, rather than the earlier opt-in W8A16 MMA candidate.

| Batch | Native median step, ms | WGMMA median step, ms | Result |
|---:|---:|---:|---|
| 1 | 21.012 | 21.177 | 0.79% slower |
| 8 | 92.319 | 43.465 | 2.124× faster |
| 16 | 109.235 | 60.527 | 1.805× faster |

This is a measured improvement in model decode execution at B8/B16, with the numerical changes above and a small B1 regression in this screen. It does not establish serving latency, throughput, broad model quality, or a vLLM win. The route remains experimental and default off.

Evidence: `fp8-w8a8-step-comparison.json`, `fp8-w8a8-step-{native,wgmma}-b{1,8,16}.log`, and the final `fp8-w8a8-local-integration-proof.json`. Root timing session 50824 completed all six cells successfully.
