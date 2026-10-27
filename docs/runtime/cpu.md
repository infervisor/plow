# CPU execution

Build the CPU runtime without GPU backends:

```sh
nix develop --command cargo build -p plowrt --release --no-default-features --features cpu
```

Serve a compiled device-blob bundle containing `model.pkt`, its manifest, and a
real tokenizer. Supply the matching checkpoint directory:

```sh
target/release/plowrt serve --assets /path/to/bundle \
  --rt-checkpoint /path/to/checkpoint --cpu-isa avx512 --cpu-numa auto
```

`--executors` controls the reference interpreter. Use `--cpu-threads` for the
persistent CPU kernel workers. Zero selects the model-dependent width — physical
cores for MoE, logical CPUs for dense decode — then caps it at the packet's `--n-cu`,
because a worker past that owns no work in any program and only costs its spin. An
explicit count overrides both. Each loaded model owns a pool; budget threads across concurrently
served models. A CPU-only build does not probe or link CUDA/HSA drivers.

## Compiling a bundle

Compile for a CPU target: `--gpu xeon6975p` (alias `xeon6`; `plowc --list-gpus`) with
`--arch amx` (AMX-BF16 + AVX-512, the default for that part) or `--arch avx512`. The
interpreter-object step of the GPU quickstart is skipped entirely — a CPU bundle is
`model.pkt`, its manifest, and a tokenizer, with no cubin or hsaco.

```sh
CKPT=/path/to/gemma-4-12B-it
target/release/plowc --hf-dir "$CKPT" \
  --gpu xeon6975p --arch amx --n-cu 96 --max-ctx 2048 \
  --batch 1,4 --seq 128,512 --out /path/to/bundle
```

The CPU engine interprets the NVIDIA-format packet, so a CPU target takes every
`sm_120a` emit decision (single-segment packet, NVIDIA fusions) and the packet is
byte-identical to an `--gpu rtx6000pro --arch sm_120a` one except the header's
target hash (bytes 28-31). The one CPU-specific default: the NVIDIA E-series decode
merge-fold (`PLOW_NV_FA_FOLD_WIDE`) is off, since the CPU flash-decode kernels need
FLASH_MERGE. `--arch` labels the packet; the runtime tier is still chosen by
`--cpu-isa`. An AMD gfx942/gfx950 packet carries AMD-specific fusion the CPU kernels
do not implement and is rejected at load, typically as a KV-row site past the decode
program's instruction count. Bundles in `perf-data/cpu-*` were built as
`--gpu rtx6000pro`, which still loads.

`--n-cu` is the packet's count of *virtual* executors, not a thread count. Kernels
take "the `slice`-th of `nblk` shares", and the worker pool maps whatever thread
count it has onto those executors: a worker owns several when threads are fewer,
and tail workers own none when threads are more. **One bundle therefore serves any
core count**, and `--cpu-threads` need not match `--n-cu`. Fewer threads than
executors is the normal and cheap direction; more is not, since the surplus workers
own nothing, so the automatic width caps itself at `--n-cu`. Compile once at a width that divides the
largest machine you intend to serve; the emitter caps `--n-cu` at 256, because the
per-domain slice count is a nine-bit field. Leaving it at `0` takes the `--gpu`
spec's SM/CU count, which is why an explicit value is usually the better choice
for CPU.

Emit-time choices the CPU loader constrains:

| choice | why |
| --- | --- |
| `--qnorm-fuse` (default off) | leave it off for W8A8; CPU loading rejects that RMSNorm fusion |
| `--max-ctx` | sizes the KV cache, which is host RAM here — the 2048 above is a bench setting, not a serving one |
| `--batch` / `--seq` | the compiled prefill buckets and decode rungs; a bundle with no grouped-prefill program prefills one token at a time |
| MXFP4 | `--gpu` must be a target that permits it at emit; the twin is supplied at run time with `--cpu-mxfp4-dir` |

### Pipeline stage plan

`plowc --hf-dir <ckpt> stage-plan` splits a checkpoint into weight-stationary pipeline stages: one socket per
stage, every weight L2-resident, tokens streamed through the stages. It packs the BF16 weights in execution order
into stages of `--cores` x `--l2-weight-kib` (`auto` = the lockable L2 per core that `/dev/pseudo_lock` reports).
The units are the per-layer-input projection, then per layer attention, FFN and the per-layer-input block, then
the tied LM head. FFN and head units split by output rows, in multiples of `--min-rows-per-core` x `--cores`.
An FFN piece is a slice of the intermediate dimension, so the next piece adds its partial down sum. Attention is
never split. The result can be whole layers, several layers or part of a layer per stage. A KV-shared layer
lists the stage that needs a replica of its source layer's cache.

Every stage gets a predicted time: resident bytes per core / `--gemv-gbps`, plus exchanges x `--exchange-us`,
plus KV bytes at `--batch` x `--ctx` / `--kv-gbps`, plus (`--batch` - 1) x the activation bytes every core
gathers per row / `--act-gbps`. The defaults are the single-socket stage measurements
(`docs/bringup/results/xeon6-l2r-bf16-20261008/`). Output is `stage_plan.json` (units, stages, summary) plus one
line per stage.

`--split tp` (default) handles layers larger than one socket and MoE layers. `--split pipe` packs only, and fails on
MoE.

* **Dense layer larger than one socket**: a tensor-parallel group of the smallest socket count S that fits.
  * Each socket holds whole q heads with their KV heads; a KV head is replicated when S exceeds the KV heads.
  * Each socket also holds an FFN row block.
  * o / down are split by K, so each layer needs 2 cross-socket all-reduces of `--allreduce-us` each.
* **MoE layer**: a head stage (attention, dense MLP, router) followed by an experts stage.
  * The experts stage takes ceil(experts / experts per socket) sockets, each expert striped over all cores.
  * Its predicted bytes are the experts active at `--batch` with top-k routing.
* **LM head**: vocabulary-parallel, 1 all-reduce.

`runtime/cpu/bench/l2r/cluster.py` maps measured single-socket stage runs onto these plans.

```sh
plowc --hf-dir "$CKPT" stage-plan --cores 90 --batch 16 --ctx 16384 --out stage_plan.json
```

## Flags

Every CPU runtime knob is a `--cpu-*` flag with a `PLOW_CPU_*` environment twin;
the CLI wins over the environment. `plowrt serve --help` prints them under the
"CPU runtime" heading.

| flag | env | default | effect |
| --- | --- | --- | --- |
| `--cpu-threads N` | `PLOW_CPU_THREADS` | `0` | Persistent workers. `0` selects the model-dependent width (physical cores for MoE, logical CPUs for dense decode), capped at the packet's `--n-cu`. May be set above `--n-cu`, but the surplus workers own nothing and cost their spin — measured **8.3x** on decode at 240 idle workers. |
| `--cpu-numa MODE` | `PLOW_CPU_NUMA` | `auto` | `auto` interleaves large tensors across the allowed nodes (best effort); `off` keeps the OS policy, including an external `numactl`; a list such as `0,1` requires successful placement and rejects unavailable nodes. |
| `--cpu-isa TIER` | `PLOW_CPU_ISA` | `auto` | Kernel tier ceiling: `scalar`, `avx512`, `amx`. For A/B, and for hosts without AMX. Never activates above what cpuid and OS state permit. |
| `--cpu-huge-pages=B` | `PLOW_CPU_HUGE_PAGES` | unset | Override transparent-huge-page *advice*: by default ordinary pages for interleaved tensors, huge-page advice for single-node or OS placement. Changes advice, not the system THP setting. |
| `--cpu-spin-us N` | `PLOW_CPU_SPIN_US` | `2000` | Spin budget (µs) before a blocked worker yields and parks. Decode packets are 100–500 µs apart; parking on every gap measured **+17% TPOT** at 50 µs versus 1000. |
| `--cpu-combine G` | `PLOW_CPU_COMBINE` | `0` | Combining-tree completion counters. A counter bumped by at least `2G` static executors gets one sub-counter per group of `G` executors on the same node; the group's last arrival adds the group's share to the real counter, so consumers and thresholds are unchanged. Off, every slice of a full-width op bumps one shared line and the last bump measured 2 µs median, up to 16 µs. `G=16` measured TPOT p50 −7 to −10% at c1 (E2B, E4B; ISL 1000-15900) on the Xeon 6975P-C, outputs identical (`docs/bringup/results/xeon6-l2r-bf16-20261008/p5_serving_report.md`). |
| `--cpu-prefill-chunk N` | `PLOW_CPU_PF_CHUNK` | `0` | Largest prefill chunk (rows) one tick may run while other slots decode; `0` = whole prompt. Measured **negative** at concurrency ≥ 4, so it stays off. |
| `--cpu-pack-prefill=B` | `PLOW_CPU_PACK_PREFILL` | `true` | Prefill the prompts waiting in one serve tick together, in one launch, bit-identical to prefilling each alone (see *Packed prefill*). |
| `--cpu-mxfp4-dir DIR` | `PLOW_MXFP4_DIR` | unset | Directory holding the MXFP4 weight twin (`mxfp4/<name>` plus `_scale` rows, from `perf-data/tools/quantize_mxfp4.py`). |
| `--fp8-dir DIR` | `PLOW_FP8_DIR` | unset | The fp8 weight twin. Runtime-wide rather than CPU-specific, but this is how a CPU bundle gets W8A16/W8A8 weights. |
| `--cpu-global-queue=B` | `PLOW_CPU_GQ` | `false` | Take the blob's op-major global work queue, windowed per segment and locality domain, instead of static per-cu streams. **Measured ~2x slower** on the EPYC 9654; kept for A/B where the static partition is a poor fit. |
| `--cpu-l2-place=B` | `PLOW_CPU_L2_PLACE` | `false` | Place executors by the packet's L2 locality domains instead of `cu % nodes`. **Measured 1.5x slower** and never faster (placement report in the `perf-data/cpu-numa-placement` campaign, kept out of source control). Inert on a blob carrying no domains, and the balance guard declines a losing plan even when this is on. |

The last two default to off because they were measured worse, not because they are
unfinished. Both are safe to flip for an A/B on a different host; neither changes
what is computed.

`--executors` (not CPU-specific) sizes the reference interpreter. Each loaded model
owns its own worker pool, so budget threads across concurrently served models.

## Packed prefill

Prefill on the CPU is weight-bandwidth bound per launch, so prompts that wait in the same
serve tick share one launch: their rows are concatenated (`M` = the sum, at most the widest
bucket whose attention merges split partials) and the weights stream once. Only three ops
depend on which request a row belongs to, and the engine hands them a pack descriptor
(`PlowCpuCtx::pack`) for that run only: the KV-writing `HEADNORM_ROPE` writes row `r` into
slot `row_slot[r]`, `FLASH_PREFILL` walks its work per span (that span's rows, positions, KV
length and slot), and the head samples each finished prompt's last row (a second, head-only
run at `M = S` when several finish). Everything else is row-independent; `pack_route`
refuses a packet with any op it does not know to be, and logs why.

Prefill is invariant to the pack, the bucket and the chunking. GEMMs are per-row invariant in
`M`; attention splits the KV at absolute positions (`split_rows` = max(2048, max_ctx / 16) keys)
instead of a per-bucket count, so a row's partial over a split depends only on its own position,
and the splits a wider q tile adds are empty for it (exact zeros in `FLASH_MERGE`). Partials go
to an engine-owned scratch at per-row offsets (`row_off`). Every prefill chunk, including one
prefilled alone, runs this way on a bucket that carries a `FLASH_MERGE`, so a prompt's logits and
KV are the same bits whether it is prefilled whole, in chunks, after a reused prefix (session
resume, cross-slot prefix share) or packed with others. `plowc` keeps a merge in every bucket
with `PLOW_DENSE_PF_NS_MIN=2` (the 12B/26B-A4B/31B recipes set it; their 2048 bucket otherwise
writes attention directly and neither packs nor gets the invariance).

`cpu_pack_check` is the gate: packed, alone in a pack, on the widest bucket, and split into
chunks across packs must all equal the whole prompt prefilled alone, bit for bit (logits, first
token, greedy decode). Measured on Xeon 6975P-C, seven chat/natural prompts (3503 rows, two packs
of <= 2048), all exact: E2B 1.66x, E4B 1.61x, 12B 1.53x faster than one prefill per prompt.
`cpu_share_check` holds a reused prefix (prompt rows, decoded rows, a chain of both) to the same
bits as a fresh prefill; only prefill-written rows are reused.

The 2048-key floor on `split_rows` is for accuracy, not speed. Gemma-4-12B BF16 is sensitive to
how attention partitions a long prompt's keys: with 256-key splits its FP32-reference gate KL
p99 was 0.79 (vLLM 0.47), with every layer's attention as close to an f64 reference as the
unsplit kernel's; at 2048 (one split per q tile up to 2K context, the unsplit sums) it is 0.33.

## ISA coverage

The x86-64 scalar code stays at the baseline ISA. Runtime detection checks OS
register state and AVX-512 F/BW/VL/BF16/VNNI before using the vector tier. AMX also
requires TILE/BF16/INT8 and OS permission. Neither vector tier enables AVX-512
FP16. This supports AMD EPYC 9654 (Zen 4), which provides BF16/VNNI but not FP16
or AMX. A modern C compiler supporting these target flags is required.

The AVX-512 tier includes BF16, FP8, and MXFP4 GEMV/GEMM, fused GLU and BF16 norm,
FP32-output split-K GEMM, attention, attention-residual mixing, normalization,
RoPE, pointwise operations, and the existing Gemma/GPT-OSS MoE kernels. FP8 GEMM
accepts BF16 or scaled FP8 activations and decodes all e4m3fn codes, including
subnormals and NaNs. MXFP4 uses the existing packed even-K row layout and E8M0
block scales. Packet tile/slice ownership is preserved.

The AMX tier runs `FLASH_PREFILL` on TMUL (`amx/attention_amx.c`): same units, absolute
splits and partials as the AVX-512 kernel, with QK^T and PV as 2x2-blocked TDPBF16PS over a
staged Q tile and per-64-key VNNI K^T / V blocks. Each row's scores and output keep a fixed
summation order, so packed and chunked prefill stay exact (`cpu_pack_check`). Accuracy against
an f64 reference matches the AVX-512 kernel; one thread runs 3.5-6x faster (hd 512 causal 2048:
63 vs 384 ms), and a Gemma-4-E4B 2048-row prefill drops from 485 to 424 ms.

`QUANT_FP8` supports per-row BF16-to-e4m3fn activation quantization with FP32
scales, round-to-nearest ties-to-even, and finite saturation. Both its ordinary
path and its fused gate/up activation use AVX-512; the fused form computes the
activation, the bf16 store, and the row maximum in one pass, measuring 9.7-13.4x
its scalar reference on one EPYC 9654 core. Compile W8A8
without `--qnorm-fuse`: CPU loading rejects that unsupported RMSNorm fusion.
W8A16 uses FP8 weights and BF16 activations; neither mode requires native FP8
arithmetic instructions. Gemma FP8 expert GLU/down kernels have scalar and
AVX-512 implementations, with exact e4m3fn weight decoding and per-expert row scales.
A decode-only bundle uses its single-row decode program to prefill one token at
a time. This supports the current Gemma-26B W8A16 bundle, whose compiler emits no
grouped-prefill program; long-prompt latency is consequently higher.

NOP, zero fill, expert alignment, and the exact Gemma router-score opcode retain
their scalar implementations. The fast router-score opcode has vector coverage.
Unsupported opcodes are rejected during model loading. This is not support for
every GPU model or GPU-specific packet fusion. Use trusted compiler-produced
assets; tensor-handle validation is not a complete sandbox for hostile packets.

Vector reductions can differ from sequential f32 accumulation. Kernel tests use
numerical tolerances and independent references where available; bit-identical
generated text across ISA tiers is not promised. The vector gelu evaluates
`0.5*x*(1 + tanh(c))` in its equivalent `x * sigmoid(2c)` form, which the scalar
tier does not: below roughly `x = -4` the `1 + tanh` form cancels down to noise
and `tanhf` then saturates the scalar tier's output to zero, so the two tiers
disagree in that tail, on values whose magnitude is far under one fp8 code of the
row scale. Existing optional INT8/INT16 MoE
paths and approximate FP8 decode GEMVs retain their earlier accuracy tradeoffs.

## NUMA policy

Topology is restricted to the calling thread's allowed CPUs, including cpusets
and `taskset`. A permitted SMT sibling remains usable when its lower-numbered
sibling is excluded. Worker placement alternates nodes at each core position,
then places SMT siblings. Worker metadata uses the same placement list.

### Executor placement

The packet fixes how work is *divided* — `n_cu` virtual executors, each kernel
computing the `slice`-th of `nblk` shares — but not where those executors run.
A pool of any width covers them: a worker owns several cus when there are fewer
threads than cus, and tail workers own none when there are more. One blob
therefore serves any core count, and `n_cu` never has to match it. Compile once
at a width that divides the largest core count you intend to serve; the emitter
caps `n_cu` at 256, because the per-domain slice count is a nine-bit field.

Executors are placed on node `cu % nodes`. Placing them by the packet's L2
locality domains instead is implemented behind `--cpu-l2-place` and is **off**,
because it measured 1.5x slower and never faster (placement report in the
`perf-data/cpu-numa-placement` campaign, kept out of source control).

The reason is worth keeping, because the idea is superficially attractive and the
obvious rescue does not work. An L2 domain says which slices share a *GPU* cache.
It does not say which weights they touch, and CPU model tensors are `mbind`
interleaved across every node regardless — a weight read is ~1/8 local wherever
the reading thread sits, so grouping by domain creates no memory locality here.

What it does cost is concurrency. A contiguous domain map confines a k-slice op to
`ceil(k / sms_per_partition)` nodes, and ops are not all full width: in one
Gemma-4-31B prefill program `GEMM` is 291 instructions averaging 89.6 of 144
slices, so those run on 5 of 8 nodes while the round-robin spreads them over all
8. Summed per node the two look balanced — 0.14% apart on that blob — while every
barrier still waits on a narrower machine, and placement measured 1.24x slower
anyway. Balancing the domains does not rescue it, because the narrowing is in the
map's contiguity, not in how domains are assigned to nodes.

The mechanism is kept, tested, and A/B-able because it costs nothing when off, not
because a win is expected: balanced domains were measured and still lost. What has
not been tried is the AMD round-robin domain map, which spreads each op across
nodes rather than narrowing it, and so is the shape most likely to come out
neutral. When placement is on, `node_plan`
still declines any plan that would leave a node busier than the round-robin
would in ANY ONE PROGRAM, which is what rejects the case above. The per-program
test is the load-bearing part: programs are alternatives — a prefill bucket or
the decode program per dispatch — so a plan has to be safe for each separately,
and testing their summed work instead would let one program's ruin hide behind
another that leans the other way. It also declines when the blob
carries no domains, when the legacy layout encoded the domain in `seg`, on a
single node or domain, when placed programs disagree, and when the domains do not
divide evenly over the nodes. Under the AMD round-robin map the plan reduces to
`cu % nodes` exactly whenever the domain count equals the node count, so on an
8-node host an 8-XCD blob would see no difference either way.

| Setting | Workers | Large model tensors |
| --- | --- | --- |
| `auto` | All allowed CPU nodes | Bind on one node, interleave across multiple nodes; report failure and retain OS policy |
| `off` | All allowed CPU nodes | Inherit OS memory policy, including an external `numactl` policy |
| `0` or `0,1` | Requested nodes | Require successful binding/interleaving; reject unavailable nodes |

Tensors of at least 256 KiB use fresh, 2 MiB-aligned anonymous mappings. Apply
`mbind` before initializing or copying tensor data. Multi-node interleave uses
ordinary pages by default; single-node binding and OS placement advise transparent
huge pages. Override with `--cpu-huge-pages=true|false` or
`PLOW_CPU_HUGE_PAGES=true|false`. This changes allocation advice, not the system's
THP setting. Fresh mappings avoid stale placement from allocator reuse and release
the VMA policy on drop. Smaller allocations retain the ordinary heap path. Per-worker
scratch is first touched after pinning to its worker CPU. No memory-policy syscall
or tensor allocation is added to a kernel invocation.

This is a shared-memory engine with interleaved tensors, not NUMA tensor parallelism.
It does not replicate weights or KV by socket, nor use remote-node work stealing
outside the global queue's own stealing. Those require model-level measurements and
a matching compiler/runtime ownership design. The packet carries no host topology and
the compiler emits none: a blob is never built for a particular node count, and
executor placement is decided at load. Interleave requests balanced page placement;
allocation fallback can still concentrate physical pages on fewer nodes. It does not
guarantee local access or a multi-socket inference speedup, and following the packet's
locality domains measured slower rather than faster.

## Verification on EPYC 9654

The local checks passed: Rust CPU runtime tests, all 10 C test targets, sliced/tail
GEMMs, FP8 activation/weight codes, restricted affinity using CPUs 192 and 216,
and live NUMA placement. ASan/UBSan passed the vector and new GEMM suites. The
AMX-specific test self-skips on this AMD host. The live test checks the requested
node mask, full residency, single-node binding, and huge-page advice. Interleaved
allocations can fall back under memory pressure; equal residency on every node is
not a guaranteed property of the policy.

One single-thread, warm `M=64,N=512,K=1024` microbenchmark measured:

| Weights | Scalar | AVX-512 |
| --- | ---: | ---: |
| BF16 | 26.42 ms | 1.32 ms |
| FP8 | 184.50 ms | 1.51 ms |
| MXFP4 | 33.36 ms | 1.95 ms |

These compare local kernels against their scalar implementations, not whole-model
throughput or vLLM. They are not a deployment SLA.

Run the checks with:

```sh
nix develop --command cargo test -p plowrt --no-default-features --features cpu
nix develop --command cmake -S runtime -B /tmp/plow-cpu -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build /tmp/plow-cpu --parallel 8
nix develop --command ctest --test-dir /tmp/plow-cpu -R cpu_dev --output-on-failure
nix develop --command cargo test -p plowrt --features cpu --lib live_numa_policy -- --ignored --nocapture
```

The full Gemma-4-26B BF16 network passed HTTP generation, streaming, four-client
concurrency, 932-token multi-chunk prefill, disconnect recovery, and the direct
slot lifecycle test on this host. Four longer responses produced 512 tokens in
16.225 seconds with 96 AVX-512 workers. The `perf-data/cpu-gemma26b` run report (kept out
of source control) has the summarized results, commands, and the observed imbalance in
physical NUMA placement.

The full 48-layer Gemma-4-12B BF16 network also passed the HTTP checks with both
96 and 192 physical cores. The all-core run verified 24 pinned workers per NUMA
node and measured 3.33 output tokens/s serially or 11.34 tokens/s at concurrency
four. The `perf-data/cpu-gemma` 12B report (kept out of source control) has first-token
latency, prefill measurements, and the limits of this comparison.

The `perf-data/cpu-release` experiment report (kept out of source control)
adds BF16/W8A16/W8A8 answer comparisons, near-capacity context checks, a 256-request
26B FP8 soak, and repeated NUMA measurements at fixed worker count. At 24 workers,
distributed placement reduced 12B decode latency by 2.3–2.6x versus node 0. The
report includes 96/192-core comparisons and the FP8 prefill/decode tradeoff.

These are bounded local checks. Broader quality evaluation, long-duration load,
and other CPU hosts remain production-qualification work. Repository-wide
formatting and a parallel compiler-test failure also prevent a clean merge gate.
The original gitignored CPU plans were absent; their outstanding items cannot be
declared closed from this checkout.
