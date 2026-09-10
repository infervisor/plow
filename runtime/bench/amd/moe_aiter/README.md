# GLM TP8 grouped MoE: plow and AITER

This compares the emitted per-rank shape: H=6144, expert I=256, E=256, top-k=8.
It calls plow's actual grouped align/GLU/down/combine bodies and AITER's fused
MoE entry point on synthetic and captured inputs. The native adapter also has
an opt-in serving route, described below.

## Flat A16 decode candidate

The gfx942 flat A16 object performs activation quantization, routing and output
initialization internally. The local high-level AITER dispatcher still rejects
flat kernels on gfx942; the isolated harness calls the low-level entry point
with the pinned gfx942 object. `flat-active` adds one kernel that packs only
routed experts and unpacks the raw routing table. It does not sort tokens.

Layer-77, rank-0 decode activations and TP8 checkpoint weights give these
packing-inclusive results. Smaller batches use prefixes of the batch-8 capture.

| Rows | Plow warm, µs | Flat + pack warm, µs | Plow cold, µs | Flat + pack cold, µs |
|---:|---:|---:|---:|---:|
| 2 | 225.86 | 113.21 | 254.54 | 156.03 |
| 4 | 228.78 | 114.65 | 259.91 | 164.65 |
| 8 | 243.44 | 126.95 | 278.86 | 199.54 |

Warm savings are 48–50%; cache-flushed savings are 28–39%. The resident-weight
flat kernel alone takes 52.16 µs at batch 8. All twelve cells, including a
batch-1 grouped-adapter experiment, passed the complete FP32 oracle screening.
Flat relative L2 error is about 3%, versus 0.23% for Plow. This does not establish
model-quality equivalence or a serving speedup.

Each flat arm checks twelve poisoned output reuses and a 512-byte guard after
the eight-byte coordination region. Packing checks exact active weights and
routing, untouched inactive experts, and changed routing/layer pointer tables.
See [samples and provenance](mi300x-flat.json).

Build `flat_pack.hip` using the same compiler flags and includes as `kernels.hip`
below, then link it as a separate shared library. Build the Plow control with
`-DPLOW_MOE_BENCH_DECODE=1`. Run `captured.py` with `--arm flat-resident` or
`--arm flat-active`, `--flat-library /path/to/flat-pack.so`, the pinned `--object`,
`--rows 8 --oracle-all-rows --cache-flush-mib 512`, and the existing capture,
checkpoint and library arguments. The runner verifies both the supplied and
actually loaded AITER object against the recorded SHA-256.

Router top-k, shared expert, residual, TP communication and interpreter
scheduling remain outside timing.

`--glm-moe-flat-decode` / `PLOW_GLM_MOE_FLAT_DECODE=1` enables an experimental
runtime route for gfx942 TP8 decode rungs 2/4/8 at this geometry. It defaults
off. The route preserves ordered segments and XCD placement, skips alignment,
and launches active-expert packing followed by the flat assembly kernel. The
existing combine reads its BF16 output directly. Other decode rungs retain
their existing kernels. Both enabled and disabled full-model BF16-KV batch-8
packets now pass 18/18 retrieval cases at concurrency 8. Broader quality and
serving gains remain unqualified; the isolated timings above used the earlier
benchmark adapter.

Build the adapter with `scripts/build_moe_aiter.sh OBJECT_DIR
AITER_MOE_CODE_OBJECT AITER_FLAT_CODE_OBJECT`. Both assembly objects are pinned
by SHA-256. Under a gfx942 GPU lease, set `PLOW_TEST_AITER_DIR=OBJECT_DIR` and run
`cargo test -p plowrt --lib --features hsa moe_aiter_flat_hsa_dispatch -- --ignored`
to check raw HSA dispatch, changed routes, poisoned output reuse and buffer guards.
Both flat and sorted HSA dispatch tests pass. The shared packing implementation
also passes the captured batch-8 FP32 screen (3.08% relative L2), exact packing
and routing checks, and twelve poisoned output reuses with guards. The
`adapter_qualification` entry in the record contains these results and hashes.
Both full-model TP8 packets pass Lean ordering verification for all eight
programs. With flat decode disabled, the packet is byte-identical to the
existing baseline.

### Wider batches and routing stress

The [wide-batch record](mi300x-flat-wide.json) includes complete FP32 oracle
checks at rows 16/20, using the shared production packing kernel. Captured
routing shows packing-inclusive warm gains, but distinct expert IDs across
all token/top-k pairs remove that advantage:

| Rows / active experts | Plow warm, µs | Flat + pack warm, µs | Resident flat warm, µs | Plow cold, µs | Flat + pack cold, µs |
|---:|---:|---:|---:|---:|---:|
| 16 / 128 | 458.36 | 614.08 | 175.22 | 488.37 | 653.29 |
| 20 / 160 | 709.38 | 706.90 | 242.00 | 735.98 | 755.97 |

Resident timing excludes packing. These results identify repeated weight
packing as a bottleneck and do not justify extending the runtime route to
rows 16/20. Flat relative L2 error is 4.05–4.33% for these spread-routing cells;
all flat cells pass twelve poisoned reuses and output guards.

Reproduce with `--rows 16` or `--rows 20`, `--routing spread`,
`--oracle-all-rows --cache-flush-mib 512`, and the corresponding wide capture.
Spread routing retains captured activations and gates but replaces expert IDs;
it is a stress case, not a measured serving routing distribution. Omit
`--verify-part`, since the captured partials belong to the original routing.
The default `--routing capture` preserves the original benchmark behavior.

Inside `nix develop`, with ROCm PyTorch and `amd-aiter==0.1.19` available:

```sh
bench=runtime/bench/amd/moe_aiter
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -fPIC \
  -Iruntime/amd -Iruntime/common -c "$bench/kernels.hip" -o /tmp/moe-compare.o
c++ -shared /tmp/moe-compare.o -L"$ROCM_PATH/lib" \
  -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 -o /tmp/moe-compare.so
for arm in plow aiter ck; do
  perf-data/tools/gpulease -n 1 moe-compare env AITER_FLYDSL_FORCE=0 \
    python "$bench/compare.py" --arm "$arm" --library /tmp/moe-compare.so \
    --out "/tmp/moe-$arm.json" --rows 128 2048 8192
done
```

The measured environment is MI300X gfx942, ROCm 7.14 compiler/runtime, Torch
`2.12.0+git6bbd260` (HIP build `7.2.53211`). The local Python command was
`/app/plow/build-gemma31/vllm-python` with
`VLLM_ROCM_LIB=/opt/rocm/core-7.14/lib`.

## Matched inputs and timing boundary

- Seed 31; BF16 normal activations; exactly representable FP8 integer weights
  in [-8,8], with independently varying scales per 128x128 block. Routing has
  eight unique experts per token and normalized positive weights.
- AITER receives FNUZ FP8; plow receives numerically converted OCP FP8. Weight
  values and block scales match. This is not a reinterpretation of FP8 bytes.
- Plow uses eight waves, epilogue hoisting and deterministic FP64 accumulation,
  matching the measured production object's flags. Its adapter includes the
  accumulator clear, four align phases, GLU, down and combine.
- AITER includes activation quantization, sorting, expert computation and
  reduction. Weight conversion/preshuffling and router top-k are outside timing
  in both arms. Shared expert, residual, TP communication and packet protocol
  are excluded in both arms.
- GPU graph/event timing uses seven samples after warmup and reports the
  median. Allocations, JIT and oracle evaluation are outside timing.
- The independent FP32 oracle dequantizes each selected expert with every block
  scale, applies SiLU and routing weights, and checks first/middle/last tokens.
  All outputs must be finite. Plow's screening threshold is 1% relative L2;
  the activation-quantized AITER arms use 10%. These thresholds are explicitly
  **not** a model-quality acceptance gate.

## Measured results

The installed heuristic chose one-stage assembly at all three row counts.
`--arm ck` disables that candidate before the first dispatch lookup to measure
the two-stage CK path. This changes only the benchmark process.

| Query rows | Plow ms | AITER assembly ms | AITER CK ms |
|---:|---:|---:|---:|
| 128 | 0.910 | 0.317 | 0.485 |
| 2048 | 1.574 | 0.771 | 0.847 |
| 8192 | 4.732 | 2.640 | 2.439 |

At 8192 rows, CK is **1.94x faster** than the plow adapter. Assembly wins at
the smaller row counts. Sampled relative L2 error is 0.23–0.24% for plow and
4.06–4.37% for AITER. The latter includes FP8 activation quantization absent
from plow's W8A16 path; no full-model quality conclusion follows.

The measured assembly object is
`fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co`; the CK implementation
loaded `module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so`.
Recorded environment, library hashes and all nine cells are in
[mi300x-results.json](mi300x-results.json). No third-party binary is vendored.

Adapting these kernels requires activation block quantization, scale/weight
layout conversion, native launch boundaries and actual-model quality checks.
The result supports that work; it does not establish that rewriting plow's
existing inner loop in assembly would yield the same gain.


## Actual GLM capture and reusable weight packing

`captured.py` reads the checkpoint's TP8 expert slices and actual layer-40
activations/routing from a 70k-token prefill. Two captures cover the full
8192-token chunk at c0=57344 and the final 4464 live rows at c0=65536.
The baseline GLU is **bit-identical across all 16,777,216 / 9,142,272 captured
values**, after matching the runtime's OCP negative-zero canonicalization.
This checks the capture, gate/up order, expert routing and TP weight slices.

OCP-to-FNUZ conversion must preserve values near the format limit. Reinterpret
canonicalized OCP bytes as FNUZ and double the weight block scales. The runner
checks exact value equality across every weight. Casting OCP values directly
to FNUZ would overflow values above FNUZ's range. The checkpoint slice contains
4523 negative-zero bytes; matching the runtime's upload scrub is necessary for
plow's optimized CDNA3 weight staging too.

| Captured live rows | Plow, ms | AITER assembly, ms | AITER CK, ms | Assembly + reusable packing, ms |
|---:|---:|---:|---:|---:|
| 8192 | 4.833 | 1.971 | 2.371 | 2.647 |
| 4464 | 2.825 | 1.245 | 1.485 | 1.918 |

The full-chunk path is **45.2% faster including packing**, or 1.83x the baseline
throughput at this kernel boundary. Sampled FP32-oracle relative L2 is
**0.237% plow vs 3.80–3.85% AITER** for the full chunk and **0.237% vs
3.55–3.56%** for the tail. These are three sampled output rows, not model-level
quality or serving measurements. The standalone baseline reproduces the model's
GLU exactly, but its downstream combine excludes shared expert and TP work.
Across every output element, AITER differs from plow by 3.54–3.60% relative L2.
Resident and repacked assembly runs differ by 0.20–0.28% despite identical
packed weights. Do not assume the fused output preserves plow's deterministic
FP64 combine contract; that needs an explicit integration decision and test.

Keeping a shuffled copy of all 75 routed layers would duplicate about 84.4 GiB/rank
for these expert weights, before scale storage. Instead, `plow_moe_pack`
accepts plow's per-expert weight/scale pointer tables and packs one layer into
**1,208,254,464 reusable bytes** (about 1.125 GiB). Weight shuffling and doubled
scales are included on every dispatch in the `aiter-repack` arm. Input
quantization, sorting, expert computation and reduction are included too.

The first pack loop took 1.285 ms. Assigning eight workgroups per expert and
hoisting the expert pointers outside each copy loop reduced this to about
0.674 ms. Both versions match every packed weight and scale byte against
AITER's shuffle. The final test also reverses the expert pointer tables and
restores them, checking exact packed bytes after each reuse. No additional
assembly was needed for this packing loop. The attention-independent MoE
assembly object remains the one named above.

The [capture result record](mi300x-captured-results.json) includes artifact,
checkpoint-slice and source hashes. All measurements held eight GPU leases and
used one GPU for the isolated MoE kernels. Captures used TP8 BF16 KV, native
AITER sparse attention and the existing B1 packet; the capture request uses
seed 0. They do not use the vLLM serving benchmark's random-token generator.

### Reproduce the capture

Use the B1 native sparse-MLA bundle from the
[MLA capture instructions](../mla_sparse_aiter/README.md), whose layer-40
post-attention/MoE work completes in segment 82. Set `CAPTURE` to a fresh
existing directory; snapshots use create-new semantics. Inside `nix develop`:

```sh
# C0=57344 for all 8192 rows; C0=65536 for the 4464-row tail.
export PLOW_PF_CAPTURE="8192@$C0:82:act.xn2=$CAPTURE/x.bin,act.tab=$CAPTURE/tab.bin,act.moe_fug=$CAPTURE/fu.bin,act.moe_meta=$CAPTURE/meta.bin,act.moe_rowtok=$CAPTURE/rt.bin,act.moe_rowpart=$CAPTURE/rp.bin,act.moe_rowgate=$CAPTURE/rg.bin,in.kvlen=$CAPTURE/len.bin"
PLOW_HSACO="$OBJECT_DIR" PLOW_MLA_PF_AITER=1 PLOW_MLA_PF_V2=1 \
PLOW_PF_CHUNK=8192 PLOW_PF_INTERLEAVE=0 \
  perf-data/tools/gpulease -n 8 moe-capture target/release/plowrt bench \
    --assets "$B1_ASSETS" --prefill-sweep --prefill-lengths 70000 \
    --prefill-reps 1 --prefill-warmups 0
unset PLOW_PF_CAPTURE
```

Build the library with the command at the top of this document. Then run each
arm with ROCm PyTorch/AITER, using `ROWS=8192` or `ROWS=4464` for its capture:

```sh
for arm in plow aiter ck aiter-repack; do
  perf-data/tools/gpulease -n 8 moe-captured env AITER_FLYDSL_FORCE=0 \
    python runtime/bench/amd/moe_aiter/captured.py --arm "$arm" \
      --library /tmp/moe-compare.so --capture "$CAPTURE" \
      --checkpoint /workspace/models/GLM-5.3-plow-lite --layer 40 --rank 0 \
      --rows "$ROWS" --out "/tmp/moe-captured-$arm.json"
done
```

## Native gfx942 serving adapter

`--glm-moe-aiter=true` (`PLOW_GLM_MOE_AITER=1`) emits `MoeAiterFp8Pf` (156)
for routed prefill MoE. The instruction explicitly specifies dynamic A8
activation quantization and BF16 routed accumulation, stored as FP32 for
plow's existing shared-expert/TP combine. The previous FP64 path remains the
default and still handles decode. This option requires TP8, H6144/I256/E256,
top-k 8, buckets 128..8192 and `--emit-packed-prefill=false`.

Each isolated segment enqueues four ordered HSA kernels: weight packing,
input quantization/routing conversion/output clear, pinned AITER assembly,
and BF16-to-FP32 output storage. There is no HIP or Python dependency in the
serving process. One **1,361,486,080-byte workspace per rank** serves all
75 routed layers at the 8192-row capacity. The three dense layers use the
existing path; the earlier 78-layer duplication estimate included those
dense layers and overstated the avoided weight storage.

The loader verifies the assembly SHA-256, gfx942/TP8 geometry, adapter ABI,
64-row alignment marker, tensor capacities and isolated segments without
interpreter counter obligations. It refuses decode and packed uses. The
pinned assembly descriptor omits its kernarg size; after checking its hash,
the loader fills the 448-byte size at the known descriptor offset for ROCr.
The assembly instructions remain unchanged.

| Captured live rows | Plow, ms | Native adapter, ms | Reduction |
|---:|---:|---:|---:|
| 8192 | 4.833 | 2.975 | 38.4% |
| 4464 | 2.825 | 2.186 | 22.6% |

These HIP-harness measurements include per-dispatch weight packing, plow alignment,
quantization, assembly and output storage. Input FP8 bytes, scales and routed
entries match AITER exactly. Sampled FP32-oracle relative L2 is 3.85% / 3.55%.
The direct HSA test also exercises ragged rows 1/129, varying expert scales,
unequal routing weights and workspace reuse.

A deterministic experiment (`--arm aiter-slots`) quantizes once, duplicates
FP8 inputs for eight top-k-1 calls' rows, and sums their BF16 expert outputs
in FP64. It passed exact reduction and three identical repeats, but took
**4.194 ms** at 8192 rows versus 4.541 ms when duplicating BF16 before
quantization. Its extra per-slot storage and smaller speedup made the explicit
fused BF16 option the better serving candidate. This experiment does not
establish determinism for other inputs or hardware.

Build the adapter beside the already qualified interpreter/MLA objects:

```sh
# Run inside nix develop. Supply the object from amd-aiter 0.1.19.
bash scripts/build_moe_aiter.sh "$OBJECT_DIR" "$AITER_MOE_OBJECT"
```

The accepted assembly object is named above and has SHA-256
`65b4c0a0b290dd83039047c18e0bb86f4253790e926dce324ddb6b45a7b28650`.
No third-party binary is vendored. Its launch contract follows
[AITER's native wrapper](https://github.com/ROCm/aiter/blob/main/csrc/py_itfs_cu/asm_fmoe.cu).

Replay the qualified BF16-KV B8 compiler configuration, preserving the
unrecorded sparse-attention knobs as well:

```sh
PLOW_UNISEG=0 PLOW_GLM_DSA_PF_SPAN=3 PLOW_MLA_PF_V2=1 PLOW_MLA_PF_AITER=1 \
  target/release/plowc --replay-knobs "$BASELINE_ASSETS/build.json" \
    --hf-dir /workspace/models/GLM-5.3-plow-lite --emit devblob \
    --gpu MI300X --arch gfx942 --max-ctx 81920 --n-cu 0 --num-gpus 8 \
    --glm-moe-aiter=true --emit-packed-prefill=false --out "$ASSETS/model.pkt"
PLOW_HSACO="$OBJECT_DIR" PLOW_MLA_PF_V2=1 PLOW_MLA_PF_AITER=1 \
PLOW_PF_CHUNK=8192 PLOW_PF_INTERLEAVE=0 \
  perf-data/tools/gpulease -n 8 moe-native target/release/plowrt serve \
    --assets "$ASSETS" --port 18965
```

Use the existing checkpoint/tokenizer/weights metadata alongside the emitted
packet. Packet/operand ABI tests, native route tests, CUDA+HSA compilation and
Lean ordering certificates pass. With the option off, the compiler reproduces
the previous baseline packet byte-for-byte. TP8 B8 retrieval passes **18/18**
through 68.8k prompt tokens; **17/18** outputs match the baseline text exactly.
This is a limited retrieval gate, not a broad accuracy evaluation.

### Matched serving screen

An adjacent baseline uses the same final runtime and interpreter/attention
objects, TP8 B8, BF16 KV, chunk 8192 and no prefill interleaving. Both runs
complete **20/20, zero failures**, with identical per-request input/output
lengths: 1,414,538 input and 13,795 output tokens. The client uses the user's
70k/700, range-ratio 0.14, C20, seed-0 random workload, reduced to 20 requests
for screening. No speculative decoding is enabled.

| Metric | Baseline | Native A8 MoE | Change |
|---|---:|---:|---:|
| Duration, s | 488.906 | 460.454 | −5.8% |
| Output tokens/s | 28.216 | 29.960 | +6.2% |
| Mean TTFT, s | 196.985 | 178.419 | −9.4% |
| P99 TTFT, s | 411.138 | 380.260 | −7.5% |
| Mean TPOT, ms | 216.916 | 205.565 | −5.2% |
| P99 TPOT, ms | 282.553 | 266.923 | −5.5% |
| Median ITL, ms | 116.833 | 119.005 | +1.9% |
| P99 ITL, ms | 1686.281 | 1471.807 | −12.7% |

The earlier B8 baseline was 28.191 tokens/s; the adjacent baseline confirms
it within 0.1%. This is one screen per arm. The median-ITL regression and
limited accuracy coverage keep the new numerical path opt-in. The H200
100-request target of 273.67 tokens/s remains unmet.

[The measured record](mi300x-native-results.json) includes both serving
summaries, per-request lengths, retrieval outputs, kernel measurements,
runtime/object/packet hashes, configuration and verification results. The
baseline contains unused packed-prefill companions; the native option omits
them. Sparse MLA prevents their use in this serving workload.

## GLM batch-8 decode screen (2026-09-10)

[mi300x-decode.json](mi300x-decode.json) records two rank-zero, layer-77 decode
captures from TP8. The emitted B8 packet uses the same grouped operator family
as prefill: `MoeAlignPf`, `MoeGroupGluPf`, `MoeGroupDownPf`, `MoeCombinePf`.
The ordinary single-row decode kernels are a different path and are not covered.
The benchmark build matches the decode tile: BM64/BK64/DBUF1, eight waves,
`PLOW_MOE_PF_DET=1`, `PLOW_MOE_PF_EPI=0`. Both arms use the emitted
single-workgroup phase-0 alignment (`PLOW_MOE_BENCH_DECODE=1`). The baseline
clears its fixed-point accumulator separately; production router top-k normally
performs that clear. Initial four-phase alignment screens were superseded.

| Capture | Active experts | Plow cold (µs) | AITER active-pack cold (µs) | Reduction |
|---|---:|---:|---:|---:|
| Step 0 | 18 | 280.70 | 220.73 | 21.4% |
| Step 1 | 21 | 284.86 | 241.12 | 15.4% |

At step 1, packing all 256 experts costs 823.95 µs including execution.
The resident-weight lower bound is 121.82 µs, but retaining shuffled copies of
all 75 MoE layers requires another 84.4 GiB per rank. Selective packing retains
the existing 1,208,254,464-byte packed-weight/scale allocation and only writes
experts whose aligned routing count is nonzero. It does not need a second copy
of every layer. This kernel is currently in the benchmark only.

Cold timings are medians of nine graph replays, each preceded by a 512 MiB
cache flush outside the timed region. Warm timings and cold samples are in the
record. Both arms include alignment and routed expert combination. Native arms
also include activation quantization and the final FP32 store; `native-active`
includes selective packing on every replay. Shared experts, residual, router
top-k, interpreter overhead, and TP communication are excluded from both arms.

Both captures match all 16,384 baseline GLU values and all 49,152 fixed-point
FP64 accumulator values exactly. All eight output rows are checked against an
FP32 checkpoint-weight oracle. Plow relative L2 error is 0.23%; AITER is
3.07–3.11%. Selected packed bytes/scales match the full-pack reference; poisoned
inactive experts remain untouched, including after reversing and restoring the
weight/scale pointer tables. Activation quantization and aligned routing are
also checked against the installed AITER implementation.

These are diagnostic prompts with distinct repeated token IDs per slot, not the
random serving workload. The primitive result does not establish model quality
or a serving throughput gain. Production native MoE remains prefill-only.
Decode integration must add ordered TP segments, validate counter boundaries,
and pass model-level quality and matched serving runs before enabling it.

Reproduce under an exclusive eight-GPU lease using the qualified runtime,
packet, object overlay, and checkpoint paths (hashes are in the record):

```sh
python3 runtime/bench/amd/moe_aiter/capture_decode.py \
  --runtime "$PLOW_RUNTIME" --blob "$PLOW_PACKET" --objects "$PLOW_OBJECTS" \
  --checkpoint "$PLOW_LITE_CHECKPOINT" --out "$PLOW_CAPTURE"
```

Build the benchmark library in `nix develop`:

```sh
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -fPIC \
  -DPLOW_MOE_PF_EPI=0 -DMPF_DBUF=1 -DMPF_BK=64 -DPLOW_MOE_BENCH_DECODE=1 \
  -Iruntime/amd -Iruntime/common -c runtime/bench/amd/moe_aiter/kernels.hip \
  -o /tmp/moe-decode.o
c++ -shared /tmp/moe-decode.o -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" \
  -lamdhip64 -o /tmp/moe-decode.so
```

For each capture and arm, run the following with the AITER/Torch Python
environment under an exclusive single-GPU lease. Add `--verify-part` for
`--arm plow`. The other arms are `native`, `native-active`, and
`native-resident`.

```sh
python runtime/bench/amd/moe_aiter/captured.py --arm plow \
  --library /tmp/moe-decode.so --object "$PLOW_AITER_OBJECT" \
  --capture "$PLOW_CAPTURE/step1" --checkpoint "$PLOW_FP8_CHECKPOINT" \
  --layer 77 --rows 8 --oracle-all-rows --cache-flush-mib 512 \
  --verify-part --out /tmp/moe-decode-plow.json
```

The capture helper reproduces the executed capture command; the original
capture-script hash is retained separately. Upstream dispatch reference:
[ROCm AITER fused MoE](https://github.com/ROCm/aiter/blob/main/aiter/fused_moe.py).

### Decode integration rejected after serving measurement

The [full-model record](mi300x-decode-serving.json) tests selective packing in
TP8 decode rungs 2/4/8. The experimental emitter isolated 75 native MoE
segments per rung and selected FP32 combination of the native BF16 output.
Prefill programs and decode rung 1 were unchanged; disabling the experiment
reproduced the qualified packet byte-for-byte.

Both adjacent runs used the same frozen runtime and 75 GPU images, with the
candidate first. Each completed 20/20 requests, zero failures, and identical
per-request lengths: 1,414,538 input / 13,795 output tokens. The workload was
70k/700, ratio 0.14, C20, seed 0, without speculation.

| Metric | Existing decode | Native selective packing | Change |
|---|---:|---:|---:|
| Output tokens/s | 31.771 | 30.396 | −4.33% |
| Duration, s | 434.206 | 453.837 | +4.52% |
| Mean TTFT, s | 164.013 | 166.369 | +1.44% |
| Mean TPOT, ms | 195.462 | 201.313 | +2.99% |
| P99 TPOT, ms | 229.911 | 266.115 | +15.75% |
| Median ITL, ms | 118.009 | 123.376 | +4.55% |

Both passed 18/18 retrieval cases; 16 texts matched exactly. The experimental
path also passed 173 AMD host tests, 33 emitter tests, a direct HSA adapter
test, and a 64-step TP8 smoke with all ranks agreeing. These checks establish
limited correctness coverage, not a performance benefit or broad accuracy.

The production integration was removed. Selective packing remains a benchmark
experiment, and native serving MoE remains prefill-only. The patch is preserved
locally under the record's artifact root, with source, runtime, packet, and
object hashes. The primitive speedup above did not survive full-model serving;
the added dispatch boundaries and changed schedule need separate attribution.
A subsequent `rocprofv3` diagnostic stalled in its first prefill with repeated
completion-signal waits and was stopped; it supplies no usable timings.

This is one paired 20-request screen, without a repeatability estimate. It does
not reproduce the supplied 100-request H200 run or establish its 273.67 output
tokens/s target. Prefix caching and unified batching selectors remain on by
default; unified batching still falls back for TP8.

## Resident expert weights (gfx942 TP8)

`--glm-moe-resident=true` / `PLOW_GLM_MOE_RESIDENT=1` packs primary expert
weights once during loading. It implies native sorted prefill and flat decode
at batches 1, 2, 4, 8, 16 and 20. It requires block-FP8 GLM geometry
H6144/I256/E256/top8, without EP or packed prefill. The option defaults to false.

The existing weight allocation holds native 16x32 tiles: gate/up interleaved
by expert, followed by down. Scales are doubled for FNUZ, as in GPU packing.
The loader rejects incompatible consumers or companion tables across all
programs. Opcode 156's `i7=1` declares this layout; the adapter requires
`plow_moe_resident_fp8_abi_1`. Build it with `scripts/build_moe_aiter.sh` and the
same pinned sorted and flat objects used above.

Sorted prefill skips weight packing (three launches instead of four). Flat
decode copies routing in one workgroup before the native kernel. Primary
expert storage remains 84.40 GiB per rank. Reusable MoE workspace falls from
1,361,486,080 to 153,231,616 bytes per rank, saving 1,208,254,464 bytes.
The native segments retain their ordered per-XCD packet queues.

[mi300x-resident-serving.json](mi300x-resident-serving.json) records a matched
20-request, 70k/700, range-ratio 0.14, C20, seed-0 comparison without speculation.
Both arms use the same runtime and refreshed FP8 prefill image, 633 native
decode GEMMs, and local selection; decode tiers and native fold are disabled.
The control uses staged native prefill and interpreter MoE decode.

| Metric | Control | Resident | Change |
| --- | ---: | ---: | ---: |
| Output throughput (tok/s) | 43.387 | 49.198 | +13.39% |
| Mean TTFT (ms) | 110874.87 | 102836.43 | -7.25% |
| Mean TPOT (ms) | 270.67 | 241.21 | -10.89% |
| P99 TPOT (ms) | 406.76 | 382.78 | -5.90% |
| Median ITL (ms) | 133.44 | 109.99 | -17.57% |

Both arms complete 20/20 requests without failures, with identical input and
output length arrays, and pass 18/18 retrieval checks. Twelve short tasks
produce the expected first answer in both arms; generated continuations often
repeat or add text, so this is not a strict instruction-following test. Only
5/20 random-serving texts match exactly. These checks do not establish broad
quality equivalence. One paired run supplies no repeatability estimate and
does not meet the 100-request H200 target.

Validation: 183 AMD runtime, 37 GLM emitter and 120 packet tests pass. Five HSA
checks cover legacy/resident sorted and flat dispatch, poisoned output reuse
with guards across all decode rungs, and exact CPU/GPU weight and scale packing
for all 256 experts. Both full 78-layer packets pass Lean ordering and LDS
checks for all ten programs. Disabling the option reproduces the previous
packet byte-for-byte. The record contains source, binary, packet, object and
result hashes plus reproduction scripts.

### Interpreter GEMV width after resident MoE

[mi300x-resident-gemv-width.json](mi300x-resident-gemv-width.json) tests compiled
GEMV row widths 16, 8 and 4 with the existing row walk enabled. All three images
use the resident packet's operation inventory; only the main decode image
differs. This is a short-input screen: 40 random requests at 2048/256 tokens,
range ratio 0.14, concurrency 20. It is not the H200 workload.

| GEMV width | Private scratch (bytes) | VGPR spills | Output tok/s | Mean TPOT (ms) | P99 TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 6384 | 8 | 132.053 | 126.61 | 142.14 |
| 8 | 3152 | 7 | 132.666 | 126.63 | 142.78 |
| 4 | 1824 | 2 | 126.070 | 133.74 | 148.83 |

All retain 256 VGPRs, 108 SGPRs, 87 SGPR spills and 64,568 bytes LDS. Pruning
alone leaves these resource limits unchanged. Width 8 changes throughput by
only +0.46% with slightly worse TPOT; width 4 loses 4.53% throughput. Lower
private scratch does not establish a speedup: narrower tiles require more
weight passes. Keep width 16 pending stronger evidence.

Each arm completes 40/40 requests with zero failures, identical input/output
length arrays, and 12/12 expected first answers. The loader selects the intended
GQ decode image on all eight ranks. One run per width, in order 16/8/4, supplies
no repeatability estimate or broad quality qualification. No production
defaults change.
