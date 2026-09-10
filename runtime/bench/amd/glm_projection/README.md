# GLM-5.3 TP8 projection comparison

The native-indexer 70k prefill profile puts 4.689 of 9.061 seconds in ordinary
interpreter segments. Those segments combine projections, normalization,
shared experts and TP collectives; this is not a GEMM-only attribution.
MoE takes 2.076 s, sparse MLA 1.585 s, indexer 0.488 s and flash segments 0.223 s.

[The profile record](mi300x-profile.json) includes per-chunk attribution and
layer-38 instruction mapping. `PLOW_PREFILL_SEG_TIMING=1` drains every segment
on all ranks, changing the normal execution boundary. Its timings are diagnostic,
not a new serving measurement. The baseline runtime/packet are the qualified
native TP indexer with native AITER MoE/MLA, B8 and BF16 KV.

## Projection boundary

`kernels.hip` directly instantiates the existing `d_gemm_t` bodies for the four
emitted BF16 tile families. The 304-workgroup, 512-thread launch preserves the
per-rank TP8 matrix shapes. It excludes interpreter scheduling, normalization,
collectives and the shared GLU epilogue. No production kernel is changed.

`compare.py` compares those bodies against PyTorch's `hipblas` and `hipblaslt`
preferences. A preference can fall back; it does not by itself prove which
library kernel executed. See the [PyTorch backend contract](https://docs.pytorch.org/docs/stable/backends.html#torch.backends.cuda.preferred_blas_library).
The Q-A, KV-latent and absorbed-Q inputs/weights are actual layer-38 captures.
Other shapes use seeded BF16 random tensors. The 4464-row cases use the prefix
of the full-chunk capture, not a separate tail capture. A sampled FP32 oracle
checks each output; available captured outputs also check the standalone body
against the production interpreter.

Timing uses ten GPU-event samples of twenty captured graph calls after warmup.
Input loading and oracle calculations are outside timing. The GPU lease reserves
all eight cards; this isolated comparison executes on rank 0.

## Reproduction

Inside `nix develop`, with ROCm PyTorch available:

```sh
bench=runtime/bench/amd/glm_projection
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -fPIC \
  -Iruntime/amd -Iruntime/common -c "$bench/kernels.hip" -o /tmp/projection.o
c++ -shared /tmp/projection.o -L"$ROCM_PATH/lib" \
  -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 -o /tmp/projection.so
perf-data/tools/gpulease -n 8 glm-projection python "$bench/compare.py" \
  --library /tmp/projection.so --capture /path/to/capture \
  --out /tmp/projection.json
```

Capture at `8192@57344:170` for the packet identified in the profile record.
Verify the segment after changing the packet. Use `PLOW_PF_CAPTURE` with these
seven `tensor=path` bindings; the capture interface permits at most eight
tensors totaling 256 MiB:

| Tensor | File |
|---|---|
| `act.xn` | `x.bin` |
| `act.qlr` | `qlr.bin` |
| `act.qlat` | `qlat.bin` |
| `act.ckvraw` | `ckvraw.bin` |
| `model.layers.38.self_attn.q_a_proj.weight` | `wqa.bin` |
| `model.layers.38.self_attn.derived.kv_a_latent.weight` | `wkv.bin` |
| `model.layers.38.self_attn.derived.q_absorb.weight` | `wabs.bin` |

Run `plowrt bench --assets /path/to/assets --prefill-sweep --prefill-lengths
70000 --prefill-reps 1 --prefill-warmups 0` with `PLOW_MLA_PF_V2=1`,
`PLOW_MLA_PF_AITER=1`, `PLOW_PF_CHUNK=8192`, `PLOW_PF_INTERLEAVE=0` and the
qualified `PLOW_HSACO` directory. Capture files must not already exist.

## Measured results

| Projection | Rows | Plow body, µs | hipBLASLt preference, µs |
|---|---:|---:|---:|
| q_a | 8192 | 495.00 | 318.76 |
| kv_latent | 8192 | 152.64 | 104.63 |
| q_absorb | 8192 | 363.89 | 227.90 |
| k_rope | 8192 | 59.93 | 30.22 |
| q_rope | 8192 | 58.67 | 41.69 |
| index_k | 8192 | 68.07 | 41.54 |
| index_weight | 8192 | 54.57 | 27.78 |
| router | 8192 | 84.98 | 62.77 |
| o_proj | 8192 | 490.71 | 356.64 |
| shared_down | 8192 | 105.95 | 76.49 |
| q_a | 4464 | 265.26 | 195.28 |
| kv_latent | 4464 | 133.54 | 70.03 |
| q_absorb | 4464 | 239.13 | 146.22 |
| k_rope | 4464 | 55.24 | 22.04 |
| q_rope | 4464 | 49.76 | 26.60 |
| index_k | 4464 | 61.59 | 27.70 |
| index_weight | 4464 | 53.68 | 17.35 |
| router | 4464 | 67.42 | 42.25 |
| o_proj | 4464 | 258.14 | 196.16 |
| shared_down | 4464 | 54.99 | 43.15 |

All 20 cases pass the sampled FP32 oracle. Plow's Q-A and KV-latent outputs
match the model captures bit-for-bit at both row counts. These are standalone
body timings; their gains do not establish a serving improvement.
See [raw comparison samples](mi300x-compare.json).

A separate `HIPBLASLT_LOG_MASK=160` trace identifies the library's solution
indices and kernel names for every tested shape. The
[selected kernels](mi300x-selected.json) come from one compressed gfx942
BF16 Tensile object. The [AMD logging documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.2.3/logging-heuristics.html)
describes these diagnostics. Logging was disabled during timing.

## Direct assembly and HSA qualification

Four unique assembly kernels cover the large-row Q-A, absorbed-Q and KV-latent
choices. Their notes declare a 160-byte inline argument block, 256-thread
workgroups and zero private scratch. The kernel descriptors report zero
argument bytes, so the HSA harness supplies the inspected size explicitly.
The [validation record](validation.json) pins the compressed and unbundled
objects, resource metadata, source hashes and logs.

`direct.py` launches these kernels directly through HIP's module API. It uses
GSU=1 and no stagger, with optional XCC workgroup remapping. It checks both
large-row kernel choices at 1, 129, 4464 and 8192 rows: 24 cases per mapping.
All pass against library outputs; 16/24 are bit-identical. The largest all-output
relative L2 difference is below 0.000047. Keep the numerical distinction explicit.
The selected mapping reduces Q-A at 8192 rows from 358.38 to 315.29 µs;
the library call measured 318.76 µs. See [linear](mi300x-direct-linear.json)
and [XCC](mi300x-direct-xcc.json) raw records.

`hsa.c` uses plow's C HSA backend without HIP or hipBLASLt. Twelve cases compare
complete outputs to the direct HIP exports. All match exactly on three repeats,
with 512-byte output guards preserved. This found and fixed the C loader's
128-byte symbol-name truncation; long Tensile names now resolve completely.
[HSA timings](mi300x-hsa.json) include host enqueue and drain, unlike the GPU-event
measurements above. The C loader fix is outside per-dispatch execution.

Reproduce the native boundary inside `nix develop`:

```sh
co=/opt/rocm/core-7.14/lib/hipblaslt/library/gfx942/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co
"$PLOW_BUNDLER" --unbundle --type=o \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx942 \
  --input="$co" --output=/tmp/lt-bf16.elf
bench=runtime/bench/amd/glm_projection
perf-data/tools/gpulease -n 8 glm-lt-direct python "$bench/direct.py" \
  --object /tmp/lt-bf16.elf --selected "$bench/mi300x-selected.json" \
  --capture /path/to/capture --mapping xcc --export /tmp/lt-expected \
  --out /tmp/lt-direct.json
cc -O2 -Iruntime/amd -I"$ROCM_PATH/include" \
  "$bench/hsa.c" runtime/amd/hsa_backend.c -L"$ROCM_PATH/lib" \
  -Wl,-rpath,"$ROCM_PATH/lib" -lhsa-runtime64 -o /tmp/lt-hsa
perf-data/tools/gpulease -n 8 glm-lt-hsa python "$bench/hsa.py" \
  --driver /tmp/lt-hsa --object /tmp/lt-bf16.elf --direct /tmp/lt-direct.json \
  --capture /path/to/capture --expected /tmp/lt-expected --out /tmp/lt-hsa.json
```

No serving adapter or new serving gain is claimed here. Next are an isolated,
opt-in HSA route, full-model retrieval checks and an adjacent C20 comparison.
The latest qualified serving result remains 32.478 output tokens/s; the H200
100-request target remains unmet.
