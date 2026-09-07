# CPU release experiments — EPYC 9654

2026-09-07, runtime/kernel source `2c38afe4`, `cpu-production-avx512`.
Two EPYC 9654 sockets: 192 physical cores, 384 logical CPUs, eight NUMA nodes.
AVX-512 BF16/VNNI; no AMX or AVX-512 FP16. Real Gemma-4 12B and 26B-A4B
weights, all text layers, context capacity 2048, decode batch ladder 1/2/4.
The 96-worker profiles use physical cores across all eight nodes. A 192-worker
profile uses all physical cores; neither profile uses SMT siblings.

## Changes validated

- BF16-to-e4m3fn activation quantization: scalar and AVX-512, per-row FP32 scales,
  ties-to-even rounding, finite saturation, signed zero, and fused gate/up fallback.
- Gemma FP8 expert GLU/down projections: scalar and AVX-512, exact weight decode,
  per-expert scales, batched rows, tails, and missing-expert handling.
- Decode-only bundles prefill through the single-row decode program. The current
  26B W8A16 compiler emits no grouped-prefill program, so its prefill is sequential.
- Multi-node interleave uses ordinary pages by default. Single-node binding and
  OS placement retain THP advice; `--cpu-huge-pages=true|false` overrides it.
  Unsupported `--qnorm-fuse` bundles fail at loading instead of ignoring FP8 outputs.

## Page-size experiment

96 workers, batch one, direct CPU engine, warm complete prefill and 16 decode
steps per prompt. Repeated prompt text is recomputed; there is no prefix-cache hit.

| Page advice | 32-token prefill | Decode mean | 512-token prefill | Decode mean |
| --- | ---: | ---: | ---: | ---: |
| THP | 807.75 ms | 190.154 ms | 5972.26 ms | 183.203 ms |
| Ordinary | 767.74 ms | 99.233 ms | 5813.51 ms | 107.138 ms |

A second pair with 32 decode steps measured 210.6 ms with THP versus 147.7 ms
with ordinary pages. Generated text matched in both pairs. These are local paired
observations, not a universal speedup or a comparison against the earlier HTTP
benchmark, which includes different scheduling overhead and output lengths.

The separate 2 GiB allocation probe shows the tradeoff: ordinary pages took
1.06–1.15 seconds to fault versus 0.17 seconds with THP, but placed exactly
65,536 pages on every node. THP spilled some of node 4's allocation onto node 5.
Full-model ordinary-page placement still falls back unevenly under this node's
memory pressure. The default improves allocation granularity; it does not promise
balanced residency or local memory access for every worker.

## Width, NUMA, and FP8 measurements

Batch one, 32 timed decode steps after each warm complete prefill. All profiles
below explicitly use ordinary pages. This campaign's experiments were serialized
on a shared node; unrelated system activity and allocation fallback were not
controlled. Raw rows and commands are in `performance.json`; actual pinned CPU
IDs, physical-core counts, and page placement are in `placement.json`.

| Profile | 32-token prefill | Decode mean | 512-token prefill | Decode mean |
| --- | ---: | ---: | ---: | ---: |
| 12b-bf16-24-node0 | 1210.26 ms | 253.929 ms | 12867.75 ms | 251.609 ms |
| 12b-bf16-24-all | 778.51 ms | 96.077 ms | 10226.26 ms | 95.015 ms |
| 12b-bf16-96 | 594.00 ms | 86.015 ms | 5732.76 ms | 87.163 ms |
| 12b-bf16-192 | 1001.26 ms | 90.812 ms | 5198.00 ms | 88.421 ms |
| 12b-fp8-96 | 1014.23 ms | 76.476 ms | 5573.25 ms | 73.429 ms |
| 12b-w8a8-96 | 793.49 ms | 49.820 ms | 8509.91 ms | 52.994 ms |
| 26b-bf16-96 | 304.74 ms | 37.608 ms | 2079.75 ms | 37.992 ms |
| 26b-fp8-96 | 800.25 ms | 27.304 ms | 13396.00 ms | 25.922 ms |
| 12b-bf16-24-all-repeat | 777.00 ms | 105.195 ms | 10300.00 ms | 108.937 ms |
| 12b-bf16-24-node0-repeat | 1265.51 ms | 252.944 ms | 12825.74 ms | 254.389 ms |

`24-node0` binds 24 cores and model memory to node 0; `24-all` uses three cores
on each node with interleaved memory. The reverse-order repeat confirms the
multi-node benefit at fixed worker count. Full-model pages remain uneven even
when the requested interleave policy and worker placement are correct.

At 96 workers, 12B W8A16 is a modest decode improvement with similar long-prompt
prefill to BF16. W8A8 improves decode further but costs more prefill time. Moving
BF16 to all 192 physical cores improves 512-token prefill by about 9%, increases
32-token prefill latency, and does not improve decode in this run. Thread count
should follow the workload; doubling workers is not a general speedup.

26B W8A16 improves decode, but its sequential prefill takes 13.40 seconds at
512 tokens versus BF16's 2.08 seconds. Grouped FP8 prefill remains an optimization
opportunity. Quantized continuations can differ, including selected MoE experts;
these are equal input prompts, not a bit-identical output trace. The benchmark's
model-size-based bandwidth estimate is not valid for sparsely activated MoE;
only measured latency is reported here.

## Correctness and scope

The checked-in quality probe asks eight deterministic facts/arithmetic/mapping
questions, retrieves codes across the 128/512/1024 prefill boundaries and near
the 2048-token limit, rejects an oversized prompt, and submits fresh concurrent
requests with unique identifiers. These are operational smoke checks, not a
perplexity benchmark, broad model-quality certification, or a long-duration soak.

12B BF16, W8A16, and W8A8 each pass 44 answers (12 initial/context cases plus
32 soak requests). All three match the same eight short answers. Their scalar
BF16/W8A16 counterparts also match all eight texts and token-ID sequences.
12B W8A16 and W8A8 each pass the separate eight-case HTTP lifecycle probe.
26B BF16 passes 44 answers; W8A16 passes 268 (12 initial/context cases plus
256 soak requests) and all eight HTTP lifecycle checks. Both scalar paths pass
the eight short cases and match BF16 AVX-512 texts/token IDs. See
`quality-comparison.json`. The 26B W8A16 process RSS grew from 26.39 GiB after
loading to 27.14 GiB after the full probes; cold-to-warm growth is recorded in
`quality-memory.json`, without claiming a long-duration leak test.

FP8 twins use the existing per-output-channel quantizer with CPU-only PyTorch
2.10.0. Norms, routers, embeddings, and the output head retain BF16 storage.
See `fp8-checkpoints.json` for artifact sizes and SHA-256 hashes. This is Plow's
per-row FP8 recipe, not a claim of vLLM W8A8 arithmetic parity. Existing dense
FP8 decode GEMVs still approximate FP8 subnormals; the new GEMMs and expert
kernels decode them exactly. Optional INT8/INT16/MXFP4 full-model quality is not
certified by these runs.

## Reproduce

Build with `nix develop`. Rust release uses opt-level 3, fat LTO, one codegen unit;
CPU C kernels use the existing `-O2` build and explicit per-tier ISA flags.
The compiler's GPU-named device-blob metadata is shared; no GPU kernel executes.
No CPU-specific autotuning database is available, so compilation uses `--no-tuning`.

```sh
nix develop --command cargo build -p plowrt --release --no-default-features --features cpu --bin plowrt --example cpu_bench
nix develop --command target/release/plowc \
  --hf-dir /workspace/models/gemma-4-12B-it-cpu \
  --emit devblob --arch sm_120a --gpu rtx6000pro --n-cu 96 \
  --max-ctx 2048 --emit-max-chunk 512 --emit-decode-batch-ladder 1,2,4 \
  --no-tuning --w8a16 --out build-cpu-12b-fp8/assets
nix develop --command target/release/plowrt serve \
  --assets build-cpu-12b-fp8/assets \
  --rt-checkpoint /workspace/models/gemma-4-12B-it-cpu \
  --fp8-dir /workspace/models/gemma-4-12B-it-fp8-cpu \
  --cpu-threads 96 --cpu-isa avx512 --cpu-numa auto --port 18680
# Separate terminal:
nix develop --command python3 perf-data/probes/cpu_quality_gate.py \
  --output quality.json --soak-requests 32
nix develop --command python3 perf-data/probes/cpu_http_e2e.py --output http.json
```

For W8A8, replace `--w8a16` with `--w8a8`; use a separate bundle directory.
For BF16, omit the weight-mode flag and `--fp8-dir`. The 26B model uses
`gemma-4-26B-A4B-it-cpu` and its matching FP8 twin. Use 256 soak requests for
the 26B W8A16 check. `--short-only` selects the eight-case scalar comparison.

Direct performance measurements use:

```sh
nix develop --command env PLOW_CPU_HUGE_PAGES=false \
  target/release/examples/cpu_bench build-cpu-12b/assets/model.pkt \
  /workspace/models/gemma-4-12B-it-cpu --threads 96 --numa auto --isa avx512 \
  --prompt-lens 32,512 --decode 32 --json perf.json
```

For FP8, set `PLOW_FP8_DIR` and use the corresponding bundle. For the NUMA
comparison, use 24 workers with `--numa 0` versus `--numa auto`; both explicitly
use ordinary pages. Reverse the order for the repeat. The 192-worker run uses
the separately compiled 192-lane BF16 bundle. The complete prompt is recomputed
after warm-up; load time, HTTP scheduling, and prefix reuse are excluded.

## Merge checks

`verification.json` records the checks and limitations. CPU Rust tests: 252 pass,
7 ignored. All ten C targets pass; the AMX hardware portion self-skips. The new
quantization/expert tests also pass ASan/UBSan. Live NUMA policy/residency/advice
checks pass, and a real qnorm-fused bundle is rejected before inference.

Workspace compilation and doctests pass. Full parallel compiler tests retain an
unresolved Kimi K3 test failure in unchanged code; the test passes individually,
and untouched main's full compiler suite passed. Global rustfmt fails on 18
unchanged files; every modified Rust file passes. These prevent claiming a clean
merge gate. Lean verification was not available during model compilation.

The original gitignored CPU plans were absent. This report closes the experiments
it actually measures, not those unavailable plans. Further production qualification
needs broader quality/long-duration load coverage and additional host testing.
