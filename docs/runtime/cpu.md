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
persistent CPU kernel workers. Zero selects the existing model-dependent width:
physical cores for MoE; logical CPUs for dense decode. An explicit count overrides
that choice. Each loaded model owns a pool; budget threads across concurrently
served models. A CPU-only build does not probe or link CUDA/HSA drivers.

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

Executors are then assigned to nodes from the packet's L2 locality domains when
the blob carries them (`PLOW_L2_PLACE`), so cus that feed each other share a
node. Domains are read back from the per-entry domain bits, which makes the
recovery independent of the emitter's workgroup-to-domain map. The domain is a
relative hint, never a node id: the mapping onto real nodes happens at load,
where the host topology is known, which is what keeps one blob portable across
hosts with different node counts. Each worker's first global-queue claim follows
the same assignment instead of its bare node index.

Placement falls back to the previous `cu % nodes` round-robin whenever the blob
expresses no usable locality: no `PLOW_L2_PLACE`, the legacy layout that encoded
the domain in `seg`, a single node or domain, placed programs that disagree, or
domains that do not divide evenly over the nodes. The last case is deliberate —
spreading over every node is the larger measured effect, so an uneven split is
not traded for locality. An unplaced program alongside a placed one is
indifferent to the choice and simply follows it.

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
a matching compiler/runtime ownership design. Executor placement follows the packet's
locality domains, but the packet still carries no host topology and the compiler emits
none: a blob is never built for a particular node count. Interleave requests balanced
page placement; allocation fallback can still concentrate physical pages on fewer
nodes. Domain-following placement is a locality hint acted on at load, not a guarantee
of local access or of a multi-socket inference speedup; it is not yet measured against
the round-robin on a placed blob.

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
16.225 seconds with 96 AVX-512 workers. See the
[run report](../../perf-data/cpu-gemma26b/epyc9654-avx512/README.md) for summarized results,
commands, and the observed imbalance in physical NUMA placement.

The full 48-layer Gemma-4-12B BF16 network also passed the HTTP checks with both
96 and 192 physical cores. The all-core run verified 24 pinned workers per NUMA
node and measured 3.33 output tokens/s serially or 11.34 tokens/s at concurrency
four. See the [12B report](../../perf-data/cpu-gemma/epyc9654-avx512/README.md)
for first-token latency, prefill measurements, and the limits of this comparison.

The [release experiment report](../../perf-data/cpu-release/epyc9654-avx512/README.md)
adds BF16/W8A16/W8A8 answer comparisons, near-capacity context checks, a 256-request
26B FP8 soak, and repeated NUMA measurements at fixed worker count. At 24 workers,
distributed placement reduced 12B decode latency by 2.3–2.6x versus node 0. The
report includes 96/192-core comparisons and the FP8 prefill/decode tradeoff.

These are bounded local checks. Broader quality evaluation, long-duration load,
and other CPU hosts remain production-qualification work. Repository-wide
formatting and a parallel compiler-test failure also prevent a clean merge gate.
The original gitignored CPU plans were absent; their outstanding items cannot be
declared closed from this checkout.
