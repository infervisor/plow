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

## Actual GLM TP8 ladder

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
