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

NOP, zero fill, expert alignment, and the exact Gemma router-score opcode retain
their scalar implementations. The fast router-score opcode has vector coverage.
Unsupported opcodes are rejected during model loading. This is not support for
every GPU model or GPU-specific packet fusion. Use trusted compiler-produced
assets; tensor-handle validation is not a complete sandbox for hostile packets.

Vector reductions can differ from sequential f32 accumulation. Kernel tests use
numerical tolerances and independent references where available; bit-identical
generated text across ISA tiers is not promised. Existing optional INT8/INT16 MoE
paths and approximate FP8 decode GEMVs retain their earlier accuracy tradeoffs.

## NUMA policy

Topology is restricted to the calling thread's allowed CPUs, including cpusets
and `taskset`. A permitted SMT sibling remains usable when its lower-numbered
sibling is excluded. Worker placement alternates nodes at each core position,
then places SMT siblings. Worker metadata uses the same placement list.

| Setting | Workers | Large model tensors |
| --- | --- | --- |
| `auto` | All allowed CPU nodes | Bind on one node, interleave across multiple nodes; report failure and retain OS policy |
| `off` | All allowed CPU nodes | Inherit OS memory policy, including an external `numactl` policy |
| `0` or `0,1` | Requested nodes | Require successful binding/interleaving; reject unavailable nodes |

Tensors of at least 256 KiB use fresh, 2 MiB-aligned anonymous mappings. Apply
`mbind` before initializing or copying tensor data, then advise transparent huge
pages. Fresh mappings avoid stale placement from allocator reuse and release the
VMA policy on drop. Smaller allocations retain the ordinary heap path. Per-worker
scratch is first touched after pinning to its worker CPU. No memory-policy syscall
or tensor allocation is added to a kernel invocation.

This is a shared-memory engine with interleaved tensors, not NUMA tensor parallelism.
It does not replicate weights or KV by socket, change packet ownership, or use
remote-node work stealing. Those require model-level measurements and a matching
compiler/runtime ownership design. Interleave requests balanced page placement;
allocation fallback can still concentrate physical pages on fewer nodes. It does
not guarantee local access or a multi-socket inference speedup.

## Verification on EPYC 9654

The local checks passed: Rust CPU runtime tests, all 10 C test targets, sliced/tail
GEMMs, FP8 activation/weight codes, restricted affinity using CPUs 192 and 216,
and live NUMA placement. ASan/UBSan passed the vector and new GEMM suites. The
AMX-specific test self-skips on this AMD host. The live check observed 1,024 pages on each of eight
nodes for a 32 MiB mapping, and all pages on node 0 for a bound mapping.

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
[run report](../../perf-data/cpu-gemma26b/epyc9654-avx512/README.md) for raw results,
commands, and the observed imbalance in physical NUMA placement.

The full 48-layer Gemma-4-12B BF16 network also passed the HTTP checks with both
96 and 192 physical cores. The all-core run verified 24 pinned workers per NUMA
node and measured 3.33 output tokens/s serially or 11.34 tokens/s at concurrency
four. See the [12B report](../../perf-data/cpu-gemma/epyc9654-avx512/README.md)
for first-token latency, prefill measurements, and the limits of this comparison.

Production certification still requires full-network scalar/quantized quality
comparisons, extended context/soak tests, and NUMA scaling measurements. The
original gitignored CPU plans were absent; their outstanding items cannot be
declared closed from this checkout.
