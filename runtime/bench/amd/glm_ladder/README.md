# GLM-5.3 interpreter and segment ladder coverage

MI300X builds now produce narrow decode interpreter variants by default, in
both `build_gfx942.sh` and CMake. The shell recursion previously built only
`interp_decode{,_gq}.elf`, leaving FP8-KV packets without matching tiers.
CMake previously built only the widest decode objects. The serve wrapper now
uses the runtime's precision-aware discovery, including its explicit empty
override, instead of constructing a BF16-only tier list.

Existing specialized kernel bodies are reused. No runtime width switch or
additional dispatch is added to the token loop. A full build adds widths
1/2/4/8 below the primary object's compiled width, for every enabled decode
precision and scheduler variant. This increases build time and disk usage.
For controlled single-object builds, use `PLOW_DECODE_TIERS=''` in the shell
or `-DPLOW_HSACO_DECODE_TIERS=OFF` in CMake. `PLOW_HSACO_LOWRUNG=''` disables
runtime tier selection explicitly.

## Initial GLM TP8 ladder

The [coverage record](mi300x-coverage.json) inventories all ten emitted programs,
projection shapes, native segment counts, object hashes, and compiled markers.
These are the paths in the qualified FP8-KV packet:

| Phase / rung | Projection implementation | Attention implementation |
|---|---|---|
| Decode 1 | MM1 interpreter, fused QKV/GLU | Sparse FP8 MLA decode |
| Decode 2 | MM2 interpreter, fused QKV/GLU | Sparse FP8 MLA decode |
| Decode 4 | MM4 interpreter, fused QKV | Sparse FP8 MLA decode |
| Decode 8 | MM8 interpreter; 832 GEMV instructions | Sparse FP8 MLA decode |
| Decode 16 | 633 native GEMM segments; 199 MM16 GEMVs | Sparse FP8 MLA decode |
| Decode 20 | 633 native GEMM segments; 199 GEMVs with the MM16 row walk | Sparse FP8 MLA decode |
| Prefill 128 | Small tiled GEMM and fused GLU interpreter | Eight-wave MLA prefill interpreter |
| Prefill 512 | Small/medium tiled GEMM and fused GLU interpreter | Eight-wave MLA prefill interpreter |
| Prefill 2048 | 234 native GEMM segments plus tiled interpreter | Four-wave MLA V2; conditional native sparse MLA |
| Prefill 8192 | 255 native GEMM segments plus tiled interpreter | Four-wave MLA V2; conditional native sparse MLA |

All rungs also contain 75 native routed-MoE instructions. Small remaining
projections, merge/fold, normalization and dense MLP operations retain their
interpreter implementations. Native sparse MLA requires a rung of at least
2048, a selected union and at least 2047 prior tokens; initial chunks use
the rung's attention interpreter. A native
symbol in the asset directory does not establish that this condition holds.

## Verification

The [result record](mi300x-results.json) contains:

- Five CMake configurations: B1, B4, B20, explicit tier disable, and MM8
  override. Every generated tier preserves its primary object's precision,
  scheduler, packet configuration and other flags, changing only GEMV width.
- The shared `packet_ladder_audit` checks every instruction's work slices and
  dependencies in all ten programs. Its AMD path now verifies per-XCD
  rendezvous counts independently: these counts are derived for the global
  queue and are absent from the static stream. Three audit tests pass,
  including incorrect-count and changed-dependency rejection.
- Ten freshly compiled FP8-KV interpreter objects: static/GQ pairs at
  MM1/2/4/8/16, with resource and symbol gates.
- 60 GPU projection cases using captured GLM layer-77 tensors and a full CPU
  FP64 oracle. Maximum relative L2 error 0.001685; guards and three poisoned
  output reuses pass. Cases cover all six decode rungs, including the B20 walk.
- TP8 serving: automatic tiers discovered on all eight ranks; all six decode
  rungs observed in execution; 51 requests at concurrency 1/2/4/8/16/20 pass
  a counting smoke check. The separate long-context retrieval screen passes
  18/18 at concurrency 20. Runtime, packet and all serving objects are hashed.

Projection timings exclude interpreter overhead, TP communication, attention,
MoE and the serving scheduler. The sum of six cold projection medians at
MM1/2/4/8 is 0.445/0.540/0.566/0.763 times the same shapes on MM16. These are
one ordered standalone sweep, not model throughput gains or tuned optima.
The earlier [100-request tier comparison](../mla_fp8_kv/mi300x-decode-tiers.json)
was neutral at concurrency 20. The H200 serving target remains unmet.

The merged manifest suite passes 45 tests serially. The initial parallel run
failed `the_lean_block_does_not_change_the_pairing_hash`; that source is
unchanged here. The frozen serving manifest predates the new all-op inventory,
so the shared audit checks the packet directly without a manifest comparison.

Run the build check through Nix:

```sh
nix develop -c python3 runtime/bench/amd/glm_ladder/check_build.py --out /tmp/build-check.json
```

Check an emitted GLM FP8-KV ladder and its serving overlay:

```sh
nix develop -c env -u LD_LIBRARY_PATH python3 runtime/bench/amd/glm_ladder/check.py \
  --packet-json /path/to/disasm.json --objects /path/to/hsaco \
  --require-specialized --out /tmp/coverage.json
```

`--disable-tiers --require-specialized` must reject a multi-rung packet on
the widest object. The checker audits available routes and compiled markers;
loader pairing and GPU execution remain separate checks. NVIDIA builds and
gfx950 defaults are outside this MI300X change.

## Full append ladder and context-dependent attention

The compiler accepts prefill rungs
`1,2,4,8,16,20,32,64,128,256,512,1024,2048,4096,8192` alongside decode
`1,2,4,8,16,20`. The phase boundary distinguishes overlapping widths.
Pass these prefill widths explicitly with `--mla-prefill full:...` and `--seq`.
Add `--require-full-ladder` to `check.py` to require all 21 programs.

MLA segments are isolated from GEMM at every prefill width. On gfx942 TP8,
ordinary dense FP8 MLA with eight local heads, latent dimension 512 and rope
dimension 64 uses a dedicated four-wave split object below 2048 rows.
The runtime selects a power-of-two split count using both the actual query
rows and live KV length, bounded by compiled partial storage and CU occupancy.
It patches attention and merge together. Other small dense MLA uses a separate
eight-wave object; large-rung and native sparse attention retain their existing
routes. Packed metadata cannot enter the small dense split route.

Both build systems emit static and global-queue small BF16, small FP8 and
split FP8 objects. The loader checks family, KV precision, wave count, packet
pairing and kernarg ABI. Split packets require their matching object and reject
an incompatible runtime setting instead of falling back to unsplit arithmetic.
GEMM bodies remain shared across compatible rungs; the complete ladder does
not establish that every shape has its fastest kernel.

Two additional qualification tools cover this path:

- `replay_attention.cpp` compares all outputs against the broad interpreter and
  checks selected query/head rows against independent FP64 CPU attention,
  including causal masking, FP8 row scales and all 512 output columns. Output
  guards cover every GPU result. It excludes device merge/fold, TP and serving.
- `serve_attention.py URL OUTPUT` checks known-answer retrieval for every
  append rung at 1024, 16384 and 65536 cached tokens, with exact first-request
  cache accounting and a repeated request. `--mode exact` retains a separate
  bitwise greedy-repeat diagnostic. Existing native BF16 atomic MoE reductions
  can change greedy output; repeat equality alone does not isolate an attention
  or prefix-cache defect.

The full ladder passed 45 append/context cells, 10 shared-prefix checks and
18 concurrent retrieval checks on eight MI300X ranks with per-token rank
agreement enabled. Attention replay passed 736 comparisons over 92 shapes
(query rows 1–1024, KV lengths 32–81920). Maximum absolute difference was
0.008972 against the broad GPU interpreter and 0.005092 against the sampled
independent FP64 CPU reference. The reference self-check covers E4M3FN
encoding, causal masking and per-row value scales.
These are correctness results, not a serving speedup or H200 parity claim.

A matched 20-request run at 70K input / 700 output, range ratio 0.14 and
concurrency 20 produced 49.45 output tokens/s for the old packet and 49.77
for the full ladder (+0.64%). Both completed 20 requests without failures,
with identical input/output length arrays and the same runtime and objects.
Mean TPOT changed from 235.64 to 233.84 ms; P99 ITL from 1361.07 to
1344.16 ms. One ordered pair does not demonstrate a repeatable serving win.
It is not the 100-request H200 reference, whose target is 273.67 tokens/s.

Build the replay with the Nix host C++ toolchain; the program loads existing
GPU objects rather than compiling device kernels:

```sh
nix develop -c bash -c 'c++ -std=c++17 -O3 -D__HIP_PLATFORM_AMD__ \
  -I "$ROCM_PATH/include" -I runtime/common \
  runtime/bench/amd/glm_ladder/replay_attention.cpp \
  -L "$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 \
  -o /tmp/glm-attention-replay'
nix develop -c /tmp/glm-attention-replay --check-reference
```

The GPU invocation takes four object paths in order: broad FP8 prefill,
small FP8 prefill, four-wave flash, and four-wave split prefill. Use the
corresponding global-queue objects. The CPU reference decodes the KV cache
as OCP E4M3FN; the native MoE object's FNUZ representation does not apply.
