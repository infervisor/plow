# Gemma-4-26B CPU end-to-end run

Date: 2026-09-07. Worktree: `cpu-production-avx512`, based on
`origin/cpu-vllm-26b-baseline` at `dbd9c74c`, with this branch's CPU audit changes.

## Configuration

- Two AMD EPYC 9654 sockets, 192 physical cores, 384 logical CPUs, eight NUMA nodes.
- Google `gemma-4-26B-A4B-it`, full 30-layer BF16 model; both downloaded
  safetensors shards verified against their Hugging Face LFS SHA-256 hashes.
- Checkpoint: `/workspace/models/gemma-4-26B-A4B-it-cpu`.
- Bundle: `/app/plow/.worktrees/cpu-production-avx512/build-cpu-e2e/assets`.
- CPU-only runtime, AVX-512, 96 workers, NUMA auto, static packet schedule.
- 2,048-token context, prefill buckets 128/512, decode ladder 1/2/4.
- Runtime reports 47.00 GiB of weights. No CUDA, HSA, or HIP library was mapped.
- The Lean verifier was unavailable; the compiler recorded `lean.verified=false`.
  This run provides runtime evidence, not a formal ordering certificate.

## Results

| Check | Result |
| --- | --- |
| Health and model discovery | Passed |
| Single request | `Paris`, 0.434 s |
| Streaming, usage, terminal SSE event | `Berlin`, 0.428 s; first content 0.359 s |
| Four simultaneous short requests | Rome, Tokyo, Madrid, Ottawa; 0.428–1.853 s |
| Multi-chunk prefill | 932 prompt tokens → `Paris`, 4.094 s |
| Disconnect active stream, then generate again | `4`, 0.486 s |
| Direct slot lifecycle regression | Passed in 16.14 s, including model load |
| Four simultaneous longer responses | 512 output tokens in 16.225 s = 31.56 tokens/s aggregate |

The longer requests each reached their 128-token cap and correctly returned
`finish_reason=length`. First-content latency was 0.384–1.838 s. Mean time after
the first token was 113.3–121.9 ms per output token, calculated from client wall
time and reported token usage. This includes serving overhead and admission
interference. All requests reported zero cached prompt tokens. These are local
smoke/performance observations, not a quality benchmark, vLLM comparison, or SLA.

The direct test changes rungs, releases a middle slot, refills that slot, and
continues with only slot 2 live. It checks the answer `Rome`. It deliberately
continues stepping beyond EOS to exercise the engine lifecycle; the HTTP tests
separately check normal stopping. Its old Gemma-3 markers and omitted output
tokens were corrected before running it against Gemma-4.

## NUMA observation

All 96 kernel workers had distinct, single-CPU affinity masks. The process had
379 anonymous VMAs with `interleave:0-7`, but physical placement was uneven:
most anonymous pages were on nodes 3 and 7. The host had substantial file cache
and little free memory per node. Allocation fallback or huge-page availability
may contribute; the cause was not isolated. An accepted interleave policy is
not proof of balanced physical placement. Do not claim NUMA scaling from this
run. No system memory policy or cache settings were changed.

Raw JSON artifacts were removed; results and NUMA observations are summarized
above. Retained logs: [slot test](live-slots.log), [compiler output](compile.log).

## Reproduce

Run from the worktree root. Compiler architecture metadata uses the existing
shared device-blob format; `--emit devblob` does not compile GPU kernels.

```sh
nix develop --command cargo build -p plowc --release --bin plowc
nix develop --command cargo build -p plowrt --release --no-default-features --features cpu
nix develop --command target/release/plowc \
  --hf-dir /workspace/models/gemma-4-26B-A4B-it-cpu \
  --emit devblob --arch sm_120a --gpu rtx6000pro --n-cu 96 \
  --max-ctx 2048 --emit-max-chunk 512 --emit-decode-batch-ladder 1,2,4 \
  --out build-cpu-e2e/assets --no-tuning
nix develop --command target/release/plowrt serve \
  --assets build-cpu-e2e/assets \
  --rt-checkpoint /workspace/models/gemma-4-26B-A4B-it-cpu \
  --cpu-isa avx512 --cpu-threads 96 --cpu-numa auto --port 18680
```

In another terminal:

```sh
nix develop --command python3 perf-data/probes/cpu_http_e2e.py \
  --output /tmp/cpu-http-e2e.json
nix develop --command python3 perf-data/probes/cpu_http_decode.py \
  --output /tmp/cpu-http-decode.json
```

Stop the server before running the direct test to avoid competing worker pools:

```sh
nix develop --command env \
  PLOW_LADDER_BLOB=/app/plow/.worktrees/cpu-production-avx512/build-cpu-e2e/assets/model.pkt \
  PLOW_CKPT=/workspace/models/gemma-4-26B-A4B-it-cpu PLOW_THREADS=96 \
  cargo test -p plowrt --release --no-default-features --features cpu \
  --test cpu_serve_live -- --ignored --nocapture
```

Still open: scalar/quantized full-network quality parity, maximum-context and
extended soak tests, single-node versus multi-node scaling, and the original
missing gitignored CPU plans. This run closes the real-weight BF16 serving smoke
gate, not complete production certification.
