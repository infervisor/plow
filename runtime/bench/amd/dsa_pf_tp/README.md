# GLM prefill indexer across TP8 query rows

The serving packet currently repeats all 32-head index scores and top-k
selection on every rank. This prototype partitions query rows, preserves all
32 heads per MFMA tile, and gathers only the selected int32 indices. At 8192
rows, each rank contributes 8 MiB to a 64 MiB result. Query/key projections,
union construction and attention are outside this benchmark.

`op_attention.h` accepts an optional half-open row range for the row-resident
score and radix-select bodies. The causal base remains `kv_len - global_rows`.
Empty ranks still launch; the gather uses equally sized, padded row bands so
ragged chunks never split a row between ranks. Defaults retain all-row work.

## Numerical gate

The inputs come from layer 38 of a TP8 BF16-KV GLM-5.3 70k-token prefill:
8192 live rows at chunk base 57344 and 4464 live rows at base 65536. Both
standalone all-row runs must reproduce the captured model's top-k sets.
Partitioned causal scores must be bit-identical to the standalone all-row
scores; selected sets must match on every rank. The gathered buffers must
also be byte-identical across ranks. Unowned scores, indices and padded rows
are checked for unintended writes. Selection order is inherently arbitrary
because the existing radix selector uses atomic append.

The emitted scale is `0x3c7fffff`, not the rounded real-number simplification
`1/64`. Assembly inspection also found that inlining the score body let LLVM
replace some contracted FMAs with packed multiplies and separate adds. The
GLM head reduction now uses explicit `__builtin_fmaf`, matching the shipped
interpreter's fused rounding. This restored exact captured top-k sets. The
64-head reduction retains its previous expression.

## Reproduction

Inside `nix develop`, with gfx942 ROCm PyTorch available:

```sh
bench=runtime/bench/amd/dsa_pf_tp
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -fPIC \
  -Iruntime/amd -Iruntime/common -c "$bench/kernels.hip" -o /tmp/index-tp.o
c++ -shared /tmp/index-tp.o -L"$ROCM_PATH/lib" \
  -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 -o /tmp/index-tp.so
perf-data/tools/gpulease -n 8 index-tp python "$bench/captured.py" \
  --library /tmp/index-tp.so --capture /path/to/capture \
  --capture-rows 8192 --out /tmp/index-tp.json
```

Capture `act.qidx_pf=q.bin`, `kv.38.kidx=k.bin`, `act.widx_pf=w.bin`,
`act.iidx_pf=idx.bin`, and `in.kvlen=len.bin` using `PLOW_PF_CAPTURE`.
The qualified B1 packet has layer-38 attention in segment 77; verify the
segment for a different packet. Use `8192@57344:77:...` for the full chunk
and `8192@65536:77:...` for its tail, with `--capture-rows 4464` for the latter.
Capture paths must not already exist. Run `plowrt bench --prefill-sweep
--prefill-lengths 70000 --prefill-reps 1 --prefill-warmups 0` with chunk size
8192 and native MLA enabled. No speculative decoding is involved.

The local Python wrapper was `/app/plow/build-gemma31/vllm-python` with
`VLLM_ROCM_LIB=/opt/rocm/core-7.14/lib`. The JSON records include Torch/HIP
versions, capture/library hashes and all timing samples. The benchmark
requires peer access between all eight MI300X GPUs.

## Scope and next gate

| Live rows | Context | Replicated ms | Partition + gather ms |
|---:|---:|---:|---:|
| 1 | 1 | 0.185 | 0.328 |
| 129 | 129 | 0.183 | 0.327 |
| 129 | 65536 | 0.550 | 0.482 |
| 8192 | 65536 | 19.000 | 4.051 |
| 4464 | 70000 | 12.094 | 2.359 |

See [full-chunk records](mi300x-full.json) and [tail records](mi300x-tail.json).
The [build validation](validation.json) records unchanged default-interpreter
resource metadata and identical native GPU code after limiting explicit FMA
to GLM's 32 heads. The default score function's instruction scheduling changes;
its machine code is not claimed to be byte-identical.

Timings are wall-clock medians over 15 repetitions after three warmups.
The replicated arm runs score+select on all eight GPUs. The partitioned arm
includes host submission, an all-device completion barrier, a direct GPU
peer-copy gather and final completion. The direct gather preserves raw int32
bits and does not invoke a numerical BF16 collective. Tiny/early chunks lose
time to this extra work and should retain the existing replicated route.

This is an isolated prototype, not a serving speedup. The serving route still
uses replicated index work. Integration needs device-side ready and completion
rendezvous, audited gate allocation, and safe scratch reuse before later TP
collectives. The normal segment-major path queues later segments without a
host drain at every boundary, so the benchmark's host barrier cannot simply
be omitted. Follow with retrieval quality and an adjacent C20 serving A/B.
