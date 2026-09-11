# Sparse MLA: plow / AITER comparison on MI300X

This isolates the GLM TP8 attention geometry: 8 query heads, 512 latent + 64
rope dimensions, BF16 Q/KV, top-k 2048, context 81920. It compares plow's
8-query union walk against AITER's per-query assembly kernel. It is an adapter
prototype and oracle. An opt-in native runtime route is qualified below.

`kernels.hip` calls the actual `d_index_union_pf` and
`d_flash_mla_prefill_v2<512,64,true>` bodies, with LDS DMA enabled. Union launches 512 threads, matching the
interpreter; V2 flash launches its fixed 256 threads. Its packing
kernel converts plow's separate latent/rope buffers to AITER's contiguous
576-wide rows using aligned 16-byte loads/stores. The Python runner uses
`mla_decode_fwd` with two KV splits and treats each query as one batch entry
sharing the same KV cache.

## Reproduce

From the repository root, inside `nix develop`, with ROCm PyTorch and AITER
available in the selected Python environment:

```sh
bench=runtime/bench/amd/mla_sparse_aiter
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -fPIC \
  -Iruntime/amd -Iruntime/common -c "$bench/kernels.hip" -o /tmp/mla-compare.o
c++ -shared /tmp/mla-compare.o -L"$ROCM_PATH/lib" \
  -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 -o /tmp/mla-compare.so
perf-data/tools/gpulease -n 1 mla-compare python "$bench/compare.py" \
  --library /tmp/mla-compare.so --out /tmp/mla-compare.json --rows 1 8 129 8192
```

Recorded environment: MI300X gfx942, ROCm 7.14 HIP compiler/runtime libraries,
Torch `2.12.0+git6bbd260` (reports HIP build `7.2.53211`), `amd-aiter==0.1.19`.
The local qualification used `/app/plow/build-gemma31/vllm-python` with
`VLLM_ROCM_LIB=/opt/rocm/core-7.14/lib` as the Python command.

The installed AITER object is
`aiter_meta/hsa/gfx942/mla/mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co`, SHA256
`cd8fa62e18abada15beeeedd49357bcc1e9e2eee7353d038533f30cac93c3607`.
The measured wheel is distinct from the current upstream source revision used
for the review. No AITER binary or source is vendored here.

## Measurement boundaries

- Seed 17, identical random BF16 Q/KV in both arms. Every query selects unique,
  causal past positions. Cases share all, half, or none of the top-k positions
  within each full 8-query pack. Query counts include a ragged pack.
- Both outputs must be finite everywhere and pass an independent FP32
  QK/softmax/PV oracle at the first, middle and last query, relative L2 < 0.02.
  Packing must be bit-exact. This samples numerical accuracy; it is not a
  full-model quality test. The attention scale is `1/sqrt(576)` in both arms.
- Timings are GPU events around graph replay, median of seven samples after
  warmup. Repeated data is cache-warm. Python/JIT preparation is outside timing.
- `plow_ms` includes normalization to FP32; `union_ms` is separate because
  shared-indexer layers can reuse it. `aiter_ms` includes the assembly kernel
  and split reduction. `adapter_ms` also packs Q and the **entire** KV cache and
  converts the BF16 result to FP32. It is measured together, not summed.
- Index scoring/selection, output projection/fold, TP collectives and scheduling
  are excluded in both arms. Serving needs to preserve the merge/fold contract
  and resolve packing-buffer lifetime and graph/packet dispatch.

All 12 cells passed: rows 1/8/129/8192 across the three overlap patterns.
At 8192 query rows, times in milliseconds:

| Selection overlap | Mean union size | Union | Plow attention + normalize | AITER + reduce | Full adapter |
|---|---:|---:|---:|---:|---:|
| All shared | 2048 | 1.112 | 2.110 | 2.185 | 2.328 |
| Half shared | 9216 | 1.132 | 8.662 | 2.766 | 2.893 |
| Distinct | 16384 | 1.137 | 15.496 | 2.486 | 2.706 |

The distinct-selection adapter is 5.73x faster than plow attention alone here;
the shared-selection adapter is slower when the union is already available.
Vectorizing packing reduced its separate time from approximately 0.480 ms to
0.115–0.117 ms without changing any bits. Packing is included in the final
adapter column. Plow sampled relative L2 errors were below 0.00112, AITER below
0.00196. See [mi300x-results.json](mi300x-results.json) for all cells. Timing
depends on index order and cache reuse as well as union size: an earlier
different disjoint-index pattern measured AITER at 4.06 ms vs plow at 14.83 ms
for 8192 rows. Do not extrapolate these synthetic ratios to serving.

The installed wheel's automatic single-split path produced NaNs at one row and
a GPU memory fault at 8192 rows. The cause is not established. Two splits passed
all qualification cells; the benchmark deliberately fixes that value. Do not
enable the automatic path based on these results.

## Assembly review and integration direction

The AITER object uses 16x16 BF16 MFMA, vector buffer/LDS loads, explicit waits
and barriers. ELF metadata reports 256 threads, 64 KiB LDS and no private spill
segment. Plow already uses MFMA intrinsics and direct global-to-LDS loads in
the tested path. The measured difference does not establish that replacing
intrinsics with inline assembly alone would help: work decomposition, selected
KV traffic and instruction scheduling differ together.

## Native ABI and actual model capture

`native.hip` launches the pinned assembly object directly and reduces its two
FP32 splits. Its 320-byte argument layout is object-specific; the runner refuses
a different SHA256. The native reduction writes a normalized FP32 partial and
`(m,l)=(0,1)` for each head, matching the existing one-split merge/fold contract.
Torch still owns benchmark buffers and event timing; the native adapter's GPU
path is two packing kernels, assembly attention and native FP32 reduction.

```sh
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -fPIC \
  -c "$bench/native.hip" -o /tmp/mla-native.o
c++ -shared /tmp/mla-native.o -L"$ROCM_PATH/lib" \
  -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 -o /tmp/mla-native.so
perf-data/tools/gpulease -n 1 mla-native python "$bench/compare.py" \
  --library /tmp/mla-compare.so --native-library /tmp/mla-native.so \
  --out /tmp/mla-native.json --rows 1 129 8192
```

All nine [native synthetic cells](mi300x-native-results.json) passed the sampled FP32 oracle and finite-output
check, including a ragged pack. The normalizer contract is checked everywhere.

The actual-model capture uses GLM layer 77 after 70,000 deterministic random
prompt tokens, with the last 4,464 query rows in the 8192-row bucket, TP8, B1, context capacity 81,920,
and all 78 sparse layers enabled. The shared layer uses layer 74's selected
indices. Its mean 8-query union is **3,931.8**, much smaller than the synthetic
distinct-selection case. This is a model-tensor capture, not the vLLM workload.
The model attention scale is **0.0625**, supplied explicitly.

The B1 bundle was emitted with `--emit-decode-batch-ladder 1
--emit-packed-prefill=false`, replaying the all-layer sparse TP8 `build.json`.
Its layer-77 attention is segment 155. To capture from this exact bundle, create
`/tmp/mla-capture` and run the existing prefill sweep with:

```sh
export PLOW_PF_CAPTURE='8192@65536:155:act.iidx_pf=/tmp/mla-capture/idx.bin,act.qa=/tmp/mla-capture/qa.bin,act.qr=/tmp/mla-capture/qr.bin,in.kvlen=/tmp/mla-capture/len.bin,kv.77.ckv=/tmp/mla-capture/ck.bin,kv.77.krot=/tmp/mla-capture/kr.bin'
# Add to the TP8 plowrt bench invocation for the B1 bundle:
# --prefill-sweep --prefill-lengths 70000 --prefill-reps 1 --prefill-warmups 0
perf-data/tools/gpulease -n 1 mla-capture python "$bench/compare.py" \
  --library /tmp/mla-compare.so --native-library /tmp/mla-native.so \
  --capture /tmp/mla-capture --rows 4464 --scale 0.0625 \
  --out /tmp/mla-model.json
```

The runner validates unique causal indices, hashes all six capture files, and
reads only the live query prefix. Raw tensors total 226 MiB and are not committed.
The final repeated measurement is recorded in
[mi300x-model-results.json](mi300x-model-results.json):

| Path | ms | Sampled relative L2 vs FP32 |
|---|---:|---:|
| Plow attention + normalization | 1.984 | 0.000387 |
| AITER Python adapter, including packing | 1.420 | 0.001703 |
| Native FP32 adapter, including packing | 1.377 | 0.000368 |

Native adapter latency is **30.6% lower** for this captured layer/chunk. Union
construction is separately 0.412 ms; packing the entire KV cache is included in
both adapter times. The reducer uses aligned float4 accesses to share its softmax
weights across four columns. This isolated measurement does not establish a serving improvement.

This replaces the earlier T2048 capture: that bucket uses dense attention and
retained indices from the preceding chunk. Its Q/K and selected indices were
not a current sparse-attention pair. The optional `@C0` selector now captures
the intended chunk explicitly; the replacement uses current layer-74 indices.

## Native plow runtime route

`PLOW_MLA_PF_AITER=1` opts into the native HSA route. Emit a new asset with that
variable and `PLOW_MLA_PF_V2=1`; the emitter removes cross-segment counter edges
at ordered launch boundaries. The loader rejects sparse segments that still
have counter obligations, mix other instructions, or have unsupported geometry.
The original interpreter route remains available with the runtime flag off.

Build the adapter beside the existing interpreter objects, using the qualified
AITER code object from the installed wheel:

```sh
bash scripts/build_mla_sparse_aiter.sh "$OBJECT_DIR" \
  "$AITER_META/hsa/gfx942/mla/mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co"
# Add PLOW_MLA_PF_AITER=1 to the normal TP8 plowc emit environment.
PLOW_MLA_PF_AITER=1 PLOW_HSACO="$OBJECT_DIR" \
  target/release/plowrt serve --assets "$NEW_ASSET_DIR" --port 8080
```

The production adapter is `runtime/amd/mla_sparse_adapter.hip`. Each eligible
segment launches packing/CSR initialization, assembly attention, and FP32
reduction through HSA. A 438,961,152-byte workspace per rank is reused at the
8192-row / 81920-context ceiling. The hot path allocates no buffers. Packing
uses the live KV prefix, including rebased slots, so unused VMM capacity is not
read. Early chunks whose first query has fewer than 2048 causal keys retain
plow's interpreter kernel. Ragged rows update the native launch extent.
Packed-prefill dispatch also retains its existing route.

The exact AITER object leaves `KERNARG_SIZE` unspecified in its descriptor while
metadata declares 320 bytes. HIP accepts this; ROCr reports zero to plow's
bounded argument allocator. After checking the original hash, the loader sets
that descriptor field to 320 in its in-memory image. Instructions and the file
on disk are unchanged. See the
[LLVM kernel descriptor contract](https://llvm.org/docs/AMDGPUUsage.html#kernel-descriptor).

The ignored `sparse_mla_hsa_dispatch` test exercises native HSA launch, two KV
slots, rows 1/129, and the merge normalizer contract. Run it inside a GPU lease
with `PLOW_TEST_AITER_DIR` pointing to the built object directory. Ordinary
unit tests check ragged/early chunk transitions, capacity limits, counter
obligations in static/global streams, mixed segments and unsupported shapes.

## Full-model runtime qualification

Paired TP8 measurements use the same newly emitted B8 packet, all 78 sparse
prefill layers, BF16 KV, context 81920, chunk 8192, no prefill interleaving,
and TP audit retained. Each arm discards one warmup and measures three repeats.

| Prompt tokens | Interpreter mean ms | Native mean ms |
|---|---:|---:|
| 8192 | 1250.450 | 1248.894 |
| 8321 | 1494.294 | 1494.483 |
| 70000 | 13254.466 | 12379.396 |

The 70k prefill mean is **6.60% lower**, saving 875 ms. The two short cases use
the interpreter fallback and produce identical output checksums between arms.
The long-case checksum changes. A separate instrumented 12,288-token run
confirms 78 native segments in the ragged second chunk; its barrier-instrumented
timings are excluded from the table.

Both arms pass **18/18 concurrent fact-retrieval cases**: actual prompt lengths
5433–5438 and 68797–68802, depths 0.1/0.5/0.9, three facts, concurrency 4,
temperature 0, and 24 output tokens. Text is identical in 13/18 paired cases;
the five different continuations still retrieve the expected fact. This is a
limited quality screen, not a general model-quality evaluation. Results and
object/packet hashes are in [mi300x-runtime-results.json](mi300x-runtime-results.json).

Inside one eight-GPU lease, run both arms with the same assets and objects:

```sh
export PLOW_MLA_PF_V2=1 PLOW_PF_CHUNK=8192 PLOW_PF_INTERLEAVE=0
export PLOW_HSACO="$OBJECT_DIR"
for arm in 0 1; do
  PLOW_MLA_PF_AITER=$arm target/release/plowrt bench --assets "$NEW_ASSET_DIR" \
    --prefill-sweep --prefill-lengths 8192,8321,70000 \
    --prefill-reps 3 --prefill-warmups 1 --engine-diagnostics > "sweep-$arm.log" 2>&1
done
# For each arm's separately started server:
python "$bench/quality.py" http://127.0.0.1:8080 "$arm" "quality-$arm.json"
```

These results measure prefill and retrieval. They do not establish improved
concurrency-20 serving throughput or parity with the supplied H200 reference.
Decode remains dense; no speculative decoding is added.

Primary sources:

- [AITER MLA dispatch, pinned source](https://github.com/ROCm/aiter/blob/10f8874dc2cd69c07ed84b5f125c27d12baccb10/aiter/mla.py)
- [vLLM sparse MLA backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py)
- [AMD matrix-core instruction and format reference](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)

## FP32 single-split prefill

The pinned QH8 v3 object can write normalized FP32 directly when KV splits
is one and `out_16_nosplit` remains zero. This differs from the automatic
BF16-output wrapper path described above. The FP8 adapter exposes this through
`plow_mla_sparse_single_abi_1`: native routes with at least 512 actual query
rows initialize the merge normalizer while packing, then omit the separate
partial reduction. Build with `scripts/build_mla_sparse_aiter.sh OBJECT_DIR
AITER_QH8_CODE_OBJECT --single-pass` to expose that marker. The default build,
smaller queries, BF16 KV and older adapters retain two splits. Native-route
eligibility and compiler segment boundaries are unchanged.

`replay_splits.cpp` compares one, two and four splits on the pinned object.
It covers 72 cases: rows 1/8/128/512/2048/8192, contexts 16384/81920, and
shared or row-shifted causal key sets (`distinct` in the CSV; sets can overlap
after wrapping). Three poisoned reuses and output guards pass;
an independent FP64 attention reference samples the first/middle/last query
and heads 0/7 across all 512 output columns. Maximum absolute error was
0.00009774 and relative L2 0.001520. Attention-only event medians exclude
packing, reduction, TP and serving: one split reduced time by 5.96–6.68% at
8192 rows but increased it by 27–75% below 512 rows.

Build this host-only HIP API probe inside `nix develop`:

```sh
c++ -std=c++17 -O3 -D__HIP_PLATFORM_AMD__ -I"$ROCM_PATH/include" \
  runtime/bench/amd/mla_sparse_aiter/replay_splits.cpp \
  -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 \
  -o /tmp/mla-splits
GPU_LEASE_NGPU=8 perf-data/tools/gpulease -n 8 mla-splits \
  /tmp/mla-splits "$AITER_QH8_CODE_OBJECT"
```

Runtime qualification covers rows 1/129/511/512/513, two rebased KV slots,
BF16 and both FP8 dispatch paths, including output/normalizer guards. TP8
serving with agreement checked every token passes six prefix append cases
(512/2048/8192 at contexts 16384/65536), ten shared-prefix retrievals and
18 concurrent fact-retrieval cases. Raw artifacts are under
`/tmp/tp-glm53-sparse-splits`.

A candidate-then-control serving pair used 20 random requests at 70K input,
700 output, range ratio 0.14 and concurrency 20. Both completed 20/20 with
identical input/output length arrays. The runtime, full ladder packet and 77
objects were identical; only the adapter's capability marker differed. A
10-second process monitor observed no compilation from API readiness through
either timed arm. An earlier control overlapped compilation and is excluded.

| Metric | Two splits | Single pass | Change |
|---|---:|---:|---:|
| Output tokens/s | 49.65 | 50.41 | +1.52% |
| Mean TTFT, ms | 103456 | 100742 | -2.62% |
| Mean TPOT, ms | 230.45 | 234.19 | +1.62% |
| Median ITL, ms | 101.57 | 107.44 | +5.78% |
| P99 ITL, ms | 1356.41 | 1349.17 | -0.53% |

Keep this path opt-in: throughput and TTFT improve in this pair, while mean
TPOT and median ITL worsen. This is one 20-request pair, not a repeatability
study or the full 100-request H200 comparison. The supplied 273.67 output
tokens/s target remains unmet.
