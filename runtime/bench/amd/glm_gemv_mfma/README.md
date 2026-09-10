# GLM-5.3 batched BF16 GEMV on gfx942

This experiment qualifies the existing `gemv_rows_mfma4` body on actual GLM
operands and exposes a build-time K limit for its use. It does not change the
default VALU path or add speculative decoding.

The capture contains rank-0 weights and activation inputs for all ten ordinary
BF16 projection sites in layer 77. Eight repeated-token prompts use lengths
512 (seven slots) and 32768 (one slot), followed by two decode steps. Primitive
inputs come from the second step. Full-model serving is a separate gate.

## Primitive comparison

`kernels.hip` includes the production bodies. `compare.py` tests VALU, unrestricted
MFMA, and the production `d_gemv_t` dispatcher with `GV_MFMA4_MAXK=2048`.

- 80 numerical cases: ten sites, compiled widths 2/4/8, live rows 1/width−1/width.
- All live outputs compared with CPU FP64 matmul; relative L2 must be below 0.01,
  matching the existing GLM projection benchmark's limit.
- NaNs in inactive activation rows; inactive output rows and 256-byte guards
  on each side must remain untouched.
- MFMA row values must be bit-identical across the tested compiled/live widths
  and under row permutation. This does not assert identity between production
  B1 VALU and batched MFMA.
- Selective production output must match MFMA bitwise at K≤2048 and VALU above it.
- Timing streams approximately 3 GiB of fresh weight slabs within each launch,
  with forward/reverse arm ordering. Repeated small GLM weights otherwise fit
  in L2. Primitive timing does not include interpreter scheduling or register
  pressure from other opcode arms.

Build and run inside `nix develop`, using the existing Torch/vLLM environment
and a GPU lease:

```sh
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -shared -fPIC \
  -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=8 -DPLOW_GEMV_LG=1 \
  -DGV_MFMA4=1 -DGV_MFMA4_MAXK=2048 -Iruntime/amd -Iruntime/common \
  runtime/bench/amd/glm_gemv_mfma/kernels.hip -o /tmp/libglm-gemv.so
python runtime/bench/amd/glm_gemv_mfma/compare.py \
  --library /tmp/libglm-gemv.so --capture /path/to/capture --out /tmp/result.json
```

All 80 cases pass. Worst unrestricted MFMA relative L2 against FP64 is 0.001873.
At B8, wide K=2048/256 projections improve by approximately 1.6–1.8× in the
primitive, while narrow K=6144 sites regress by approximately 36%. The selective
dispatcher retains VALU for those regressions.

## Interpreter objects

```sh
PLOW_DECODE_BATCH=8 PLOW_DECODE_TIERS=1,2,4 PLOW_ROWS_ONLY==interp_decode \
PLOW_GEMV_MFMA4=1 PLOW_GEMV_MFMA4_MAXK=2048 \
bash scripts/build_gfx942.sh /path/to/decode-objects
```

`PLOW_GEMV_MFMA4` remains off by default. When enabled, the new
`PLOW_GEMV_MFMA4_MAXK` limits it to K≤the supplied value; unset or zero preserves
the existing unrestricted MFMA probe. Both staged and global-activation paths
use the limit. External-RMS mode (`norm=1`) keeps its existing VALU body.

Use a B8 root for this B8 deployment. The default MFMA column tile at B16
exceeds its accumulator bound, and compilation correctly refuses it. The
experiment does not weaken that bound.

The geometry contract reads the new `plow_geom_GV_MFMA4_MAXK` marker and verifies
the supplied define. B1 executable sections are identical between control and
candidate; full ELF files differ because marker values differ. Default-off
executable sections are unchanged at all four rungs. Candidate B2/B4/B8 objects
contain 88/256/192 MFMA4 instructions respectively; control contains none.

Count the instruction family when inspecting objects: the installed LLVM prints
`v_mfma_f32_4x4x4_16b_bf16` for the source builtin
`__builtin_amdgcn_mfma_f32_4x4x4bf16_1k`. A search only for the builtin-style
spelling is insufficient. The earlier qualified GLM objects were rechecked with
both the family search and their `GV_MFMA4=0` markers; the path is absent there.

## Serving screen

The paired TP8 run uses identical packets, frozen runtime and prefill images,
changing only the BF16 decode images. Each arm passes 18/18 retrieval cases and
20/20 serving requests at 70k input / 700 output, range ratio 0.14 and concurrency
20. Both generate 13,795 tokens from 1,414,538 input tokens, with identical
per-request lengths. Four retrieval continuations differ in text while retaining
the expected answer.

| Metric | VALU control | Selective MFMA | Change |
|---|---:|---:|---:|
| Output tokens/s | 31.068 | 31.633 | +1.82% |
| Duration, s | 444.031 | 436.095 | −1.79% |
| Mean TPOT, ms | 194.541 | 191.647 | −1.49% |
| Median TPOT, ms | 200.428 | 186.106 | −7.15% |
| Mean TTFT, ms | 164465.009 | 166310.898 | +1.12% |
| P99 TPOT, ms | 262.118 | 277.347 | +5.81% |

This is one screening pair without a repeatability estimate. The throughput is
below the prior 33.4 tokens/s screen, and TTFT/P99 TPOT regress against its own
control. Keep MFMA opt-in. The 20-request workload is not the supplied 100-request
H200 benchmark; the H200 target remains unmet. Retrieval checks do not establish
broad model quality or routing identity.

[Primitive evidence](mi300x-primitive.json) and [serving evidence](mi300x-serving.json)
pin sources, capture inputs, executable sections, all 65 loaded images per arm,
build recipes, quality checks and measurements. Default-off executable sections
remain unchanged at B1/B2/B4/B8.

The follow-up ragged-fold build found that publishing the new cap marker in
MFMA-disabled prefill objects conflicts with their baseline geometry profile.
Current builds emit the marker only when MFMA is enabled; disabled dispatch
remains unchanged. The original records above describe the objects actually
measured at that revision.
