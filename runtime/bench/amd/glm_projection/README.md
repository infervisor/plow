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

## Native HSA serving

`--glm-gemm-lt=true` / `PLOW_GLM_GEMM_LT=1` enables an isolated `GemmLtPf`
route for unpacked TP8 GLM prefill buckets of 2048–8192 rows on gfx942.
It replaces BF16 Q-A (N2048/K6144), KV-latent (N512/K6144), and absorbed-Q /
indexer-Q (N4096/K2048) projections. Smaller buckets and other shapes retain
the existing implementation. Kernel selection uses the actual live row count:
the qualified 4464-row choice through 4464 rows, then the 8192-row choice.

Build the pinned object into the existing qualified MLA/MoE/indexer object
directory and emit inside `nix develop`:

```sh
co=/opt/rocm/core-7.14/lib/hipblaslt/library/gfx942/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co
bash scripts/build_glm_lt.sh /path/to/objects "$co"
PLOW_UNISEG=0 PLOW_GLM_DSA_PF_SPAN=3 PLOW_MLA_PF_V2=1 PLOW_MLA_PF_AITER=1 \
plowc --hf-dir /path/to/GLM-5.3-plow-lite --gpu MI300X --num-gpus 8 \
  --batch 1,4,8 --seq 512,2048,8192 --max-ctx 81920 --arch gfx942 \
  --emit devblob --replay-knobs /path/to/qualified/build.json \
  --glm-gemm-lt=true --emit-packed-prefill=false --out /path/to/native-assets
PLOW_HSACO=/path/to/objects PLOW_MLA_PF_V2=1 PLOW_MLA_PF_AITER=1 \
PLOW_PF_CHUNK=8192 PLOW_PF_INTERLEAVE=0 \
plowrt serve --assets /path/to/native-assets --port 8080
```

The build helper verifies both compressed and unbundled hashes. The runtime
verifies the full object before filling the four pinned descriptors' missing
argument-size fields. It checks kernel resources, operand capacities and
segment isolation. Each projection uses one ordered HSA launch, stack arguments
and no additional activation workspace. Serving requires neither HIP nor a
hipBLASLt library call. The existing native MLA, MoE and indexer options remain
part of the measured configuration.

The emitted 8192-row program has 255 native projections: 78 each of Q-A,
KV-latent and absorbed-Q, plus 21 indexer-Q projections. Lean verifies all
eight programs, and option-off emission is byte-identical to the preceding
native-indexer packet. Packet tests (120), manifest tests (47), route/ABI tests
(2), CUDA+HSA compilation and the release build pass. The config suite passes
23/24; its existing `no_raw_env_reads` failure names `PLOW_GLM_DSA_PF_SPAN`
and `PLOW_GLM_DSA_PF_DEXACT` reads already present before this change.

Full-model retrieval passes 18/18 through 68,802 actual prompt tokens; 12/18
continuations match the preceding native-indexer packet exactly. This limited
check does not establish general quality equivalence. Keep the option opt-in.

## Paired serving screen

Both arms use the same frozen runtime and object directory, native MLA/MoE/
indexer, TP8 B8 and BF16 KV. Only `--glm-gemm-lt` differs. The vLLM client
uses random 70k/700 lengths, range ratio 0.14, seed 0, concurrency 20 and its
sampling defaults. No speculative decoding is added.

| Metric | Projection route off | Projection route on | Change |
|---|---:|---:|---:|
| Output tokens/s | 32.726 | 33.400 | +2.1% |
| Mean TTFT, s | 160.345 | 155.044 | -3.3% |
| Median TTFT, s | 173.793 | 165.733 | -4.6% |
| P99 TTFT, s | 338.879 | 329.360 | -2.8% |
| Mean TPOT, ms | 188.317 | 186.924 | -0.7% |
| Median TPOT, ms | 195.904 | 198.048 | +1.1% |
| P99 TPOT, ms | 245.095 | 235.226 | -4.0% |
| Mean ITL, ms | 188.220 | 186.170 | -1.1% |
| Median ITL, ms | 117.299 | 116.366 | -0.8% |
| P99 ITL, ms | 1161.941 | 1132.449 | -2.5% |

Both complete 20/20 with zero failures and identical per-request token lengths:
1,414,538 input and 13,795 output tokens. Native runs first. This is a modest
**2.1% throughput gain in one screen per arm**, not a repeated estimate.
Median TPOT regresses 1.1%; the other reported latency metrics improve.
The supplied H200 result uses 100 requests and includes speculative decoding.
Its non-speculative contribution cannot be inferred from aggregate metrics;
H200 parity remains unmet.

After timing, loader ownership validation was changed from repeated stream
scans to a single scan per program. Dispatch and arithmetic are unchanged;
the final loader passes its tests and another 18/18 retrieval run (14/18
continuations match the indexer baseline, 15/18 the earlier native run).
Atomic selection ordering does not promise repeated text identity. The
[serving record](mi300x-serving.json) distinguishes both runtime hashes and
includes metrics, quality cells, resolved knobs, object/source hashes and logs.

## Ordinary decode GEMV workgroup tuning

`PLOW_GEMV_WG_TUNING` now reaches GLM's plain BF16 MLA decode projections and
batched MoE router/shared gate-up projections. Unset preserves the previous
packet bytes. Untargeted shapes and MXFP4/FP8-block operators retain their
existing workgroup selection. This does not change the DSA indexer or the
single-row MoE path.

The first MI300X experiment uses:

```sh
--gemv-wg-tuning '64x6144=304,256x6144=304'
```

The existing blocked-GEMV helper applies that cap, then removes empty trailing
groups. For these two shapes, `ceil(N/304) = 1`; narrowing to 64 or 256 groups
keeps every surviving logical slice's output columns and dot-product order.
This is an emitter change using the existing GPU objects, with no assembly or
runtime changes. Smaller caps are separate experiments and are not qualified
by the ownership argument for this setting.

The paired packets have identical instruction operands. B2 removes 3,600
stream entries; B4/B8 each remove 29,520 (B8: 400,819 → 371,299). Prefill and B1
are unchanged. All 31 GLM emitter tests pass, including a new TP8 B1/2/4/8
ownership check. Both emitted blobs pass Lean ordering checks for all eight
programs. These checks establish structural properties, not a serving speedup.

One exclusive TP8 control/candidate pair, C20 with 20 random requests at
70k/700 and range ratio 0.14, produced:

| Metric | Control | Trim | Change |
|---|---:|---:|---:|
| Output tokens/s | 31.534 | 31.987 | +1.4% |
| Duration, s | 437.463 | 431.264 | -1.4% |
| Mean TTFT, ms | 166526.542 | 162174.073 | -2.6% |
| Median TTFT, ms | 180470.308 | 170197.152 | -5.7% |
| P99 TTFT, ms | 349253.168 | 349280.813 | +0.01% |
| Mean TPOT, ms | 193.818 | 193.301 | -0.3% |
| Median TPOT, ms | 197.927 | 200.715 | +1.4% |
| P99 ITL, ms | 1133.218 | 1130.932 | -0.2% |

Both complete 20/20 with zero failures and identical per-request lengths:
1,414,538 input and 13,795 output tokens. Both pass 18/18 retrieval cases;
12/18 continuations match exactly. This uses the same frozen merged runtime
and 75 GPU images for both arms. The control's normal SIGTERM shutdown caused
the first wrapper to stop; the candidate ran after correcting that wrapper,
under a fresh exclusive eight-GPU lease. Neither benchmark was interrupted.

The small throughput difference and mixed latency results do not establish a
repeatable improvement. The knob stays opt-in. This screen is also below the
earlier runtime's 33.400 tokens/s screen; it does not replace that result or
establish parity with the supplied 100-request H200 result. The
[record](mi300x-gemv-width.json) includes raw metric summaries, quality outputs,
source/object hashes, recipes and the disabled MFMA4 path inspection.
