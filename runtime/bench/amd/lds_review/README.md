# GEMM and attention LDS / pipeline review

Reviewed 2026-09-10 for GLM-5.3 on MI300X, with AMD and NVIDIA source coverage.
This is a static review, not a new GPU bank-conflict profile or performance
qualification. The six compiled objects are from the frozen resident serving
overlay. Source hashes, geometry markers, selected native symbols, instruction
counts and all ten packet opcode inventories are in
[mi300x-static.json](mi300x-static.json).

## Findings

### 1. MLA V2 has remaining QK and probability-tile read conflicts

In `runtime/amd/op_attention.h:4084`, the V2 QK reader uses
`Ksm[krow(nt*16+fr) + kt*32 + kg*8]`, with `fr=lane%16`, `kg=lane/16`.
The 584-half row stride plus the existing 16-half row-block shift predicts
two distinct dwords per bank in a gfx942 `ds_read_b128` phase. The existing
V-stage swizzle fixes the PV transpose pattern; it does not clear this QK
pattern. All 36 QK fragments were checked for both swizzle settings.

The probability read at `runtime/amd/op_attention.h:4321`,
`Pw[fr*32 + kg*8]`, also predicts two-way reads. A candidate logical-to-physical
mapping `row*32 + (column ^ ((row&3)*8))` reduces the modeled read conflicts to
one without increasing LDS. Its per-row permutation is bijective. Updating
every probability producer and validating scalar writes is still required;
this candidate is not implemented or GPU-qualified.

The frozen FP8 flash object contains both dense and gathered V2 functions.
Each has 37 static `ds_read_b128` and 256 `ds_read_u16` instructions. Those
counts corroborate the access widths, not dynamic execution frequencies or
cycle savings. Native sparse MLA bypasses the interpreted gathered body on
eligible segments; a symbol being present does not prove it ran.

### 2. FP8 QK is missing the BF16 fragment-prefetch path

The `PLOW_MLA_PF_SV` branch at `runtime/amd/op_attention.h:4090` rotates two
K fragments only for `!FP8`. FP8 still loads one fragment immediately before
its consumer, separately accumulating latent and rope scores. PV fragment
prefetch is shared. Extend QK prefetch only while preserving the separate
scale/accumulation contract and after checking register cost.

The serving flash entry has 512 VGPRs, 58,376 bytes LDS and 1,552 bytes private
segment per work-item. Dense V2 contains 201 static scratch loads and 201
stores; these include outlined-function save/restore traffic and must not be
reported as per-tile spills. Deeper staging can lose by increasing this cost.
An isolated compiled body and the actual interpreter asset both need checking.

### 3. The BK=32 GEMM swizzle does not inherit BK=64's result

`GM_XORSWZ` in `runtime/amd/op_gemm.h:541` and `MPF_XORSWZ` in
`runtime/amd/op_moe.h:1868` use `column ^ ((row & (BK/8-1))*8)`.
For the 32x32 BF16 MFMA lane map, BK=32 predicts two-way b128 reads while
stores are conflict-free. BK=64 and BK=128 pass both modeled access patterns.
The gfx942 A4W4 variant's BK=32 option uses the equivalent byte-address map
at `runtime/amd/op_moe.h:4282`; its default BK=64 has the wider permutation.

A BK=32-only candidate uses `column ^ (((row>>1)&3)*8)`: the row-stride bit
supplies one bank bit and the permutation supplies the other two. Read/write
b128 phase checks and row bijectivity pass. This remains an unimplemented
address model. **The frozen GLM main decode object has GM_BK=64 and MPF_BK=64**;
fixing BK=32 would not improve that currently measured path.

### 4. Source defaults and GPU names do not establish active coverage

The shell recipe enables `PLOW_MLA_PF_SV` by default at
`scripts/build_gfx942.sh:900`; CMake also defaults it on at
`runtime/CMakeLists.txt:850`, despite older opt-in comments in the header.
Do not propose enabling it as a new optimization. The frozen packet uses 633
native GEMM packets and 199 Gemv packets at B20, plus 75 resident native MoE
packets. Changes to the generic GEMM body do not affect the native GEMMs.

The current remaining Gemv shapes and dense-FFN interpreter paths should be
ranked using the existing resident trace before importing more kernels.
Native GEMM/MoE/MLA are included in the compiled inventory, but their bank
addresses were not symbolically proven here. Their prior numerical/serving
qualification is not evidence that every native LDS access is conflict-free.

## Family coverage and producer/consumer patterns

| Family | Existing mechanism | Review outcome / next boundary |
|---|---|---|
| AMD BF16 GEMM, fused norm/GLU, weight-only FP8/MXFP4 (`op_gemm.h`) | Compact XOR LDS; register global prefetch; local fragment prefetch; one LDS buffer on gfx942 | BK=64/128 modeled reads/writes pass. BK=32 finding above. Block-FP8 scale promotion adds a second accumulator set. |
| AMD native FP8 GEMM (`d_gemm_fp8_t`) | 32-byte fragment swizzle; architecture-specific MFMA lowering | b128 rules cannot certify b256 accesses. `GM_PLR8` is already disabled after recorded register/spill regressions. |
| AMD GEMV variants | Staged activations, vector weight reads, register unrolling; optional MFMA and split-K | Shared-X reads include broadcasts. Bank swizzling is not automatically useful; the B20 trace and existing MM16/8/4 results are stronger prioritization evidence. |
| AMD grouped/dense MoE, Gemma/MXFP4 variants (`op_moe.h`) | XOR tiles; single/double LDS buffers selected by size; optional register and gather pipelines | BK=32 caveat; B-weight prefetch alone previously failed to improve serving. Resident AITER already removes repeated weight packing on the active path. |
| AMD dense flash prefill | Padded K; row-major V; register next-tile prefetch | Checked 32x32-MFMA K strides are conflict-free. Transposing V changes producer coalescing and has prior negative results; narrow LDS loads alone are not sufficient evidence. |
| AMD MLA V1/MFMA decode and DSA score | Padded MFMA32 K/Q tiles | Representative 72/136/520/584-half strides pass b128 reads. This is not a proof for every scalar/atomic access. |
| AMD MLA V2 BF16/FP8, dense/gathered | Register Q; staged K/V; wave-private P; online softmax | QK/P read conflicts and asymmetric FP8 prefetch above. Existing V swizzle and deferred softmax are already implemented. |
| AMD scalar GQA/MLA decode and merge/fold | Grouped heads, split KV, wave reductions, streamed KV and staged output | Different lane maps from MFMA prefill. Prior GF/unroll/fold results and byte traffic matter more than transferring a prefill tile recipe. |
| AMD native hipBLASLt / AITER MoE / sparse MLA | Selected assembly; persistent weight layout; adapter pack/reduce stages | Six object hashes and ISA inventories recorded. Selected hipBLASLt symbols only, not every unused function in its 91 MiB container. Need per-symbol address tracing/counters for bank certification. |
| AMD gfx950 materialized Opus attention | Vendored AITER subset and separate pack kernel | Architecture/shape-specific, not the gfx942 GLM 512+64 MLA path. Do not transfer its transpose instructions or bank rules to MI300X. |
| NVIDIA generic GEMM and split-K | `cp.async`, padded A/B, `ldmatrix`, multistage rings | AMD b128 phases do not apply. Preserve async wait/reuse boundaries and include split-K clear/reduction cost. |
| NVIDIA SM90 GEMM | 128-byte swizzles, TMA/WGMMA; uniform and dedicated-producer variants | Role splitting already exists in eligible segment objects. Fixed 256-thread interpreter geometry and dedicated 384-thread variants have different tradeoffs. |
| NVIDIA flash prefill/decode and SM90 flash | Async KV staging; SM90 swizzled K/V; separately laid out P | Check producer stores and MMA consumers together, including alignment and buffer reuse. Existing V double-buffer experiments include negative results. |
| NVIDIA MLA decode/fold (`op_mla.cuh`) | Scalar streamed KV, grouped heads, shared reductions | Source reviewed; no CUDA binary or GPU bank-conflict validation in this run. |

Coverage is at implementation-family level. CPU GEMM has no GPU LDS, and
non-attention recurrent KDA/GDN kernels are outside this review. Golden kernels,
experimental files, and unused library symbols are not certified by this audit.

Concurrent commit `61796f7c` was merged and reviewed before publication. It
adds opt-in per-slot packed-attention TMA descriptor resolution. Its separate
H100 audit recorded four oracle cases and clean memcheck, but Nsight Compute returned
`ERR_NVGPUCTRPERM`; no NVIDIA bank-conflict counts were obtained. HD512/BKV16
keeps its copy fallback because the descriptor box has 32 rows. This evidence
does not qualify GLM on CUDA or prove a serving improvement.

## CK techniques to adapt

Use CK's approach of checking the **producer and consumer lane distributions
separately**, then choosing a common physical layout. Our first candidates are
the probability tile and the conditional BK=32 layout above. Preserve vector
alignment and validate each instruction's phase groups; a single global
`address % 32` histogram is insufficient. See AMD's
[LDS bank-conflict guide](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html).

CK compute-v4 explicitly requires double shared-memory buffering and sizes
the allocation accordingly. Its intrawave scheduler interleaves LDS reads,
writes, global reads and MFMA; it is not a requirement to dedicate producer
waves. Copy the dependency structure only when the tile fits MI300X's LDS and
register budget. See the
[CK v4 implementation](https://raw.githubusercontent.com/ROCm/composable_kernel/develop/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp).

Dedicated producer waves consume a statically allocated register budget on
CDNA3/4. Alternating memory/compute roles can retain more useful compute than
reserving permanent producers; this is also the motivation documented in
[HipKittens](https://hazyresearch.stanford.edu/static/posts/2025-11-09-hk/hipkittens.pdf).
This is an architecture-specific tradeoff, not a ban on producer patterns.

For attention, CK represents the QK-to-PV layout handoff and K/V load/store
distributions explicitly. Apply that discipline to both the probability
producer and PV consumer rather than optimizing either in isolation. See
[CK-Tile FlashAttention](https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html).

CK also offers preshuffled weight GEMM to bypass a weight LDS round trip.
Plow already has a persistent shuffled resident-MoE path. Extending that idea
to another projection needs evidence covering layout conversion, memory use,
all weight consumers and serving, as in the existing resident qualification.
See [CK GEMM examples](https://github.com/ROCm/composable_kernel/blob/develop/example/ck_tile/03_gemm/README.md).

## Verification and implementation order

`check.py` ran 98 gfx942 address-model cases, known conflict/broadcast controls,
and candidate bijectivity checks. It read six frozen objects, their resource
notes and static ISA, and all ten programs. No new GPU bank counters were
collected and no speedup is claimed for these candidates.

```sh
nix develop -c env -u LD_LIBRARY_PATH python3 runtime/bench/amd/lds_review/check.py \
  --assets /path/to/frozen/overlay --packet-json /path/to/disassembly.json \
  --out /tmp/lds-review.json
```

First isolate the P-tile permutation, preserving every arithmetic operation.
Next evaluate FP8 QK fragment prefetch separately. Treat QK's shared K/V layout
as a joint producer/QK/PV problem; fixing one access must not regress another.
BK=32 is a separate geometry-specific task. For each: oracle/guards/ragged
tails, compiled address and register checks, actual interpreter comparison,
then matched repeated serving. Use ASM only if the qualified dataflow still
lowers poorly; instruction scheduling cannot remove a true dependency or a
bad address map.
