# GLM-5.3 ragged prefill folding

The qualified TP8 packet folds MLA attention through `MlaMergeFold`: eight local
heads, a 512-wide latent and 256 value columns. The existing token-blocked kernel
handles eight tokens per work item, sharing each loaded weight across them. Its
old dispatcher required the entire live token count to be divisible by eight.
An 8,191-row tail therefore ran the scalar fold for every row.

The dispatcher now sends complete groups to the unchanged blocked kernel and
uses the unchanged scalar kernel for at most seven trailing tokens. Output,
partial and softmax-statistic pointers advance by whole token/head rows; weights
keep their original head indexing. The new route is restricted to gfx942, TB=8, non-K3 objects and V=256, and
retains the split-count, vector-width, decode-map and workgroup-fill guards.
V=128 eligibility is unchanged. Disabling `PLOW_MLA_FOLD_TB` still disables the
blocked path. No packet, tensor allocation or collective protocol changes.

The first implementation put scalar tail handling inside the blocked kernel.
Its interpreter build added 64 scratch instructions and failed the resource
budget. Moving the tail to the dispatcher preserves the original kernel body
and passes the GLM resource contract without raising its limits. The broader
build then found two extra scratch instructions in several K3 objects; excluding
those objects from the new path restores their budgets.

## Evidence and reproduction

`kernels.hip` calls the production blocked/scalar bodies. Build two shared
libraries from the before/after headers with `FOLD_TAIL=0/1`, respectively:

```sh
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -shared -fPIC \
  -DFOLD_TAIL=1 -Iruntime/amd -Iruntime/common \
  runtime/bench/amd/glm_fold_tail/kernels.hip -o /tmp/fold-candidate.so
```

Run through `nix develop` and the GPU lease. `compare.py --root /path/to/capture
--out /tmp/result.json` expects `fold-control.so`, `fold-candidate.so` and the
four captured `.prefill.bin` files. The `.so` files require the same HIP runtime
as the Torch environment.

The actual-model capture uses rank 0, layer 77 and a 65,535-token repeated-token
prompt. Its final chunk has 8,191 live rows. `PLOW_DUMP_ACT` captures `act.opart`,
`act.mlpart`, `act.oat` and `model.layers.77.self_attn.derived.v_absorb.weight`.
Partial storage is FP32; outputs and weights are BF16. No model weights are
included in this directory.

All 68 cases pass with bit-identical control/candidate outputs. Cases cover
1–8,192 rows, one- and seven-row tails around block boundaries, the 304-workgroup fill
threshold and split counts 1/2/7. The single-split cases use actual inputs and
also match every captured output bit. Multi-split cases use seeded inputs,
actual weights and dead splits. Every case checks output guards and untouched
inactive rows. These gates cover GLM's V=256 shape, not every model family.

| Live rows, one split | Control, µs | Candidate, µs |
|---|---:|---:|
| 2047 | 362.90 | 136.26 |
| 2048 | 128.81 | 127.62 |
| 4463 | 773.76 | 275.80 |
| 4464 | 271.61 | 267.90 |
| 8191 | 1408.74 | 478.18 |

These primitive timings exclude interpreter resource pressure and scheduling.
The small weight panel is reused naturally within one fold; this tests its
repeated L2 reads, not cold-weight HBM bandwidth. Arm order is reversed within
each measurement. See [raw samples and input hashes](mi300x-primitive.json).

## Profile boundary

A fresh 70k prefill profile of the qualified native-Lt packet attributes 3.880 s
to ordinary interpreter segments, 2.047 s to MoE, 1.658 s to sparse MLA, 0.711 s
to Lt projections, 0.489 s to the indexer and 0.208 s to flash segments. This
instrument drains all ranks after each segment and changes normal scheduling.

A separate rank-0 packet trace ends with an 8,191-row chunk at position 57,344.
Its interpreter body aggregates include 158.07 ms in two-shot all-reduce and
129.93 ms in MLA merge/fold. Body spans can overlap; they are not additive
speedup predictions. Native kernels do not populate this trace. Records older
than the final chunk's Embed were excluded: native sparse dispatch leaves
23,712 old FlashMlaPrefill records from the first dense chunk in the buffer.
Counting those records would incorrectly report 7.18 seconds for the last
chunk; its actual trace window is 1.003 seconds.

## Build contracts

Both comparison arms build `interp_prefill_mla_moe{,_gq}.elf` with the same Nix
ROCm 7.14 toolchain and default axes. Both pass the existing geometry/resource
contracts. The final broader build passes all 28 prefill objects at unchanged
budgets. Its eligibility guard produces GLM ELF files byte-identical to the
measured candidate. The serving comparison uses these matched rebuilds; their executable
sections differ from the older qualified objects, so byte identity with those
historical objects is not claimed.

The build also found a marker regression from the MFMA experiment: default-off
prefill objects exposed `GV_MFMA4_MAXK=0`, which the unchanged baseline geometry
profile rejected. The marker is now emitted only when MFMA is enabled. Its
value remains checked for enabled builds. This changes no MFMA dispatch rule.

## Paired serving screen

Both arms complete 20/20 requests at 70k input / 700 output, range ratio 0.14
and concurrency 20. Each processes 1,414,538 input and 13,795 output tokens,
with identical per-request lengths. Both pass 18/18 retrieval cases; 15/18
continuations are text-identical.

| Metric | Control | Ragged fold | Change |
|---|---:|---:|---:|
| Output tokens/s | 31.427 | 31.591 | +0.52% |
| Duration, s | 438.959 | 436.670 | -0.52% |
| Mean TTFT, ms | 165586.199 | 165202.369 | -0.23% |
| Mean TPOT, ms | 194.602 | 192.739 | -0.96% |
| Median TPOT, ms | 201.689 | 199.179 | -1.24% |
| P99 TPOT, ms | 260.706 | 257.070 | -1.39% |

P99 TTFT regresses 0.85% and P99 ITL 0.13%. This is one pair without a
repeatability estimate. The small serving gain is consistent with accelerating
one tail per prompt, not every full chunk. The result remains below the prior
33.4 tokens/s screen and far below the H200 reference; this 20-request screen
does not establish parity with its 100-request workload.

[Serving and profile evidence](mi300x-serving.json) pins the sources, runtime,
packet, complete image set, recipes, quality outputs and measurements. Both
runs retain default prefix caching; unified batching still declines TP support.


## FP32 matrix pipeline candidate

The [matrix-fold record](mi300x-gemm.json) compares the current ragged TB8 fold
with a three-kernel FP32 pipeline: normalize/transposition, per-head GEMM, and
BF16 output conversion. It also records two slower fused MFMA alternatives.
This is an isolated candidate; the full-model runtime still uses the TB8 fold.

| Rows, splits=1 | TB8 cold, µs | Direct FP32 pipeline cold, µs |
|---|---:|---:|
| 2048 | 144.01 | 97.79 |
| 4463 | 292.00 | 196.45 |
| 8191 | 504.11 | 343.03 |

Timings include all three pipeline kernels. Cold runs flush 512 MiB before
one GPU-graph replay. The 2 MiB BF16 weight tensor is converted to 4 MiB FP32
once outside timing; native integration would also need FP32 scratch.
Small rows 1/7/9 lose against the existing fold and are not deployment candidates.

All 72 cells pass a complete CPU FP64 merge/product oracle (relative L2 < .003),
output guards, three poisoned direct reuses and three poisoned graph replays.
Rows cover 1/7/9/128/512/2048/4463/4464/8191; split counts 2/7 additionally cover
128/2048/8191 with dead splits and fully masked rows. Direct assembly outputs
match the library FP32 products exactly in all 12 direct-pipeline cases. The
BF16 outputs are not bit-identical to the current fold because the reductions
are reassociated; full-model quality and serving performance remain unqualified.

Five selected hipBLASLt kernels already exist in the installed gfx942 FP32
object. The library logs supply their grid, mapping and inline argument bytes.
The unbundled kernels declare 144-byte arguments, zero private bytes and zero
spills. Six separate tests through Plow's C HSA backend match complete FP32
matrix outputs on three poisoned reuses, preserving 512-byte guards. This checks
the matrix kernel boundary only; normalization/conversion and a Rust native route
still need HSA integration. The record embeds the executed HSA sources/recipe.

Build and run inside `nix develop`, using a ROCm PyTorch Python environment:

```sh
bench=runtime/bench/amd/glm_fold_tail
root=/tmp/fold-gemm
capture=/path/to/captured-fold
mkdir -p "$root"
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -fPIC \
  -Iruntime/amd -Iruntime/common -c "$bench/gemm_kernels.hip" -o "$root/helpers.o"
c++ -shared "$root/helpers.o" -L"$ROCM_PATH/lib" \
  -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 -o "$root/helpers.so"
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -fPIC -DFOLD_TAIL=1 \
  -Iruntime/amd -Iruntime/common -c "$bench/kernels.hip" -o "$root/fold.o"
c++ -shared "$root/fold.o" -L"$ROCM_PATH/lib" \
  -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 -o "$capture/fold-candidate.so"
co=/opt/rocm/core-7.14/lib/hipblaslt/library/gfx942/TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx942.co
perf-data/tools/gpulease -n 1 fold-gemm python "$bench/gemm.py" \
  --root "$root" --capture "$capture" --out "$root/results.json" \
  --selected "$bench/gemm-selected.json" --object "$co"
```

The capture contains `weight`, `opart`, `mlpart`, and `oat` with suffix
`.prefill.bin`; it has 8191 live rows. The runner pins the object SHA before
loading assembly. An earlier graph attempt used a cached HIP stream and was
excluded; the final runner obtains the current stream inside every call and
checks that poisoned outputs are restored by graph replay itself.
