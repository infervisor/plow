# Sparse MLA: plow / AITER comparison on MI300X

This isolates the GLM TP8 attention geometry: 8 query heads, 512 latent + 64
rope dimensions, BF16 Q/KV, top-k 2048, context 81920. It compares plow's
8-query union walk against AITER's per-query assembly kernel. It is an adapter
prototype and oracle, not a serving backend.

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

The actual-model capture uses GLM layer 77 after 67,584 deterministic random
prompt tokens, with the last 2,048 query rows, TP8, B1, context capacity 81,920,
and all 78 sparse layers enabled. The shared layer uses layer 74's selected
indices. Its mean 8-query union is **3,902.8**, much smaller than the synthetic
distinct-selection case. This is a model-tensor capture, not the vLLM workload.
The model attention scale is **0.0625**, supplied explicitly.

The B1 bundle was emitted with `--emit-decode-batch-ladder 1
--emit-packed-prefill=false`, replaying the all-layer sparse TP8 `build.json`.
Its layer-77 attention is segment 155. To capture from this exact bundle, create
`/tmp/mla-capture` and run the existing prefill sweep with:

```sh
export PLOW_PF_CAPTURE='2048:155:act.iidx_pf=/tmp/mla-capture/idx.bin,act.qa=/tmp/mla-capture/qa.bin,act.qr=/tmp/mla-capture/qr.bin,in.kvlen=/tmp/mla-capture/len.bin,kv.77.ckv=/tmp/mla-capture/ck.bin,kv.77.krot=/tmp/mla-capture/kr.bin'
# Add to the TP8 plowrt bench invocation for the B1 bundle:
# --prefill-sweep --prefill-lengths 67584 --prefill-reps 1 --prefill-warmups 0
perf-data/tools/gpulease -n 1 mla-capture python "$bench/compare.py" \
  --library /tmp/mla-compare.so --native-library /tmp/mla-native.so \
  --capture /tmp/mla-capture --rows 2048 --scale 0.0625 \
  --out /tmp/mla-model.json
```

The runner validates unique causal indices, hashes all six capture files, and
reads only the live query prefix. Raw tensors total 226 MiB and are not committed.
The final repeated measurement is recorded in
[mi300x-model-results.json](mi300x-model-results.json):

| Path | ms | Sampled relative L2 vs FP32 |
|---|---:|---:|
| Plow attention + normalization | 0.920 | 0.000166 |
| AITER Python adapter, including packing | 0.662 | 0.001774 |
| Native FP32 adapter, including packing | 0.652 | 0.000553 |

Native adapter latency is **29.1% lower** for this captured layer/chunk. Union
construction is separately 0.200 ms; packing the entire KV cache is included in
both adapter times. The reducer uses aligned float4 accesses to share its softmax weights across
four columns; the scalar reducer measured 0.670 ms including the adapter.
Neither measurement establishes a serving improvement or full-model quality.

Production integration still needs HSA dispatch, workspace lifetime, early
chunks with fewer than 2048 causal keys, KV-slot rebasing and ragged rows. A raw
assembly replacement must also preserve or explicitly remove cross-segment
counter edges; HSA queue ordering alone does not publish interpreter counters.
Retain the union path where selection reuse wins.

Primary sources:

- [AITER MLA dispatch, pinned source](https://github.com/ROCm/aiter/blob/10f8874dc2cd69c07ed84b5f125c27d12baccb10/aiter/mla.py)
- [vLLM sparse MLA backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py)
- [AMD matrix-core instruction and format reference](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
