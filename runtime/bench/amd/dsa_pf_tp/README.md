# GLM prefill indexer across TP8 query rows

The default serving packet repeats all 32-head index scores and top-k
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

## Isolated measurements

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

These timings are isolated indexer measurements. The native serving path below
has separate protocol and model checks.

## Native HSA serving

`--glm-index-tp=true` / `PLOW_GLM_INDEX_TP=1` emits an isolated `IndexTpPf`
instruction for buckets of 2048–8192 rows. It requires TP8, 304 CUs, GLM
H6144/HI32/DI128/top2048 and `--emit-packed-prefill=false`. Smaller buckets
keep the existing route. A ragged native bucket uses its actual live row count.
The runtime restricts the route to gfx942 and host-mapped peer status.

Add the adapter to an existing qualified object directory inside `nix develop`:

```sh
bash scripts/build_dsa_tp.sh /path/to/objects
plowc --hf-dir /path/to/GLM-5.3 --gpu MI300X --num-gpus 8 \
  --batch 1,4,8 --seq 512,2048,8192 --max-ctx 81920 --arch gfx942 \
  --replay-knobs /path/to/qualified/build.json --glm-index-tp=true \
  --emit-packed-prefill=false --out /path/to/native-assets
```

The qualified configuration retains `PLOW_GLM_DSA_PF=1`,
`PLOW_MLA_PREFILL=full:128,512,2048,8192`, sparse MLA and the preceding MoE
optimization. Serving uses `PLOW_HSACO=/path/to/objects`,
`PLOW_MLA_PF_V2=1`, `PLOW_MLA_PF_AITER=1`, `PLOW_PF_CHUNK=8192` and
`PLOW_PF_INTERLEAVE=0`. HIP and Python are only needed for building/benchmarking;
serving launches the four kernels through HSA.

The protocol uses three consecutive system-scope arrival gates:

1. Selection waits until all ranks have finished earlier work before writing
   the reused `act.dg_tp` scratch slot.
2. Gather waits until all selected bands have been written before reading peers.
3. A separate completion kernel waits until every gather has finished before
   later collectives can overwrite the slot.

The gather staggers source peers by destination rank, as plow's all-gather
already does. It copies only live int32 indices and allocates no additional
activation workspace. The loader checks the helper ABI, operand capacities,
scratch binding, gate ranges/collisions and segment isolation. Each chunk checks
the device's timeout status even if the general counter audit is disabled.

`captured.py --protocol` checks exact scores and selected sets, then queues 16
iterations without intermediate host barriers and poisons scratch after each
completion. It verifies every saved output, all three gates per iteration,
unused counter lines and timeout status. The final staggered protocol measures
**4.021 ms** for the full capture and **2.433 ms** for the actual tail.

Model retrieval passes **18/18**, with **15/18** continuations text-identical
to the prior MoE-only packet. This is limited retrieval coverage. Atomic
selection ordering and downstream floating-point accumulation do not promise
identical model continuations. Keep the route opt-in.

## Paired serving screen

Both arms use the same frozen runtime and GPU object directory, native AITER
MoE/MLA, TP8 B8 and BF16 KV. Only `--glm-index-tp` differs. The vLLM serving
client uses random 70k/700 lengths, range ratio 0.14, seed 0 and concurrency 20.
There is no speculative decoding in either plow arm.

| Metric | Indexer off | Indexer on | Change |
|---|---:|---:|---:|
| Output tokens/s | 30.010 | 32.478 | +8.2% |
| Mean TTFT, s | 175.661 | 160.254 | −8.8% |
| P99 TTFT, s | 375.041 | 339.029 | −9.6% |
| Mean TPOT, ms | 203.923 | 188.560 | −7.5% |
| Median ITL, ms | 117.248 | 117.134 | −0.1% |
| P99 ITL, ms | 1475.313 | 1163.319 | −21.1% |

Both complete 20/20 with zero failures and identical per-request token lengths:
1,414,538 input and 13,795 output tokens. Native ran first. This is one screen
per arm; median ITL is effectively unchanged. The supplied H200 benchmark uses
100 requests and reports 273.67 output tokens/s, so parity remains unmet.

The [serving record](mi300x-serving.json) includes complete summary metrics,
per-request lengths/TTFT, limited retrieval results, configuration and hashes
of the runtime, packets, GPU objects and validation logs. The device protocol
has separate [full](mi300x-protocol-full.json) and
[tail](mi300x-protocol-tail.json) records.
