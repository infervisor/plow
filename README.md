# plow — Packet Language for On-device Warps

A product of **Infervisor**.

plow is an LLM inference stack built around one idea: compile the whole model
into a **packet stream** and run it with a **persistent on-device interpreter**
— one cooperative kernel launch that stays resident, with warp-granularity op
bodies and zero per-op dispatch. The compiler (`plowc`) lowers a checkpoint
into packets; the runtime (`plowrt`) loads them and serves an OpenAI-compatible
API; an egglog rewrite stage (`crates/rewrite`) fuses the operator graph, with
Lean checks (`lean-plow/`, `crates/lean_verify`) on the non-obvious rewrites.

Supported today: Gemma-4 (12B / 31B dense, 26B-A4B MoE), Qwen3, Llama-3.1 —
bf16 and fp8 (weight-only e4m3) — on NVIDIA consumer Blackwell (sm_120, e.g.
RTX 5090 / RTX PRO 6000 Blackwell) and AMD CDNA (gfx942/gfx950).

## Requirements

- Rust via **nix**: `nix develop` in the repo root provides cargo/rustc,
  cmake, gcc. (First run: install nix, enable flakes.)
- NVIDIA path: driver + **CUDA ≥ 12.9** (13.0 tested); `nvcc` on PATH.
  sm_120 GPU for the shipped interpreter objects.
- A HuggingFace checkpoint directory (safetensors + config.json + tokenizer).

## Build

```bash
nix develop                               # toolchain shell
cargo build --workspace --release         # compiler + runtime (CPU paths)
cargo test  --workspace                   # full test suite

# GPU server binary (CUDA driver backend, loaded via dlopen — no -lcuda):
cargo build --release -p plowrt --features cuda,hf-tokenizer

# Interpreter cubins (decode + prefill objects, sm_120a):
scripts/build_sm120_cubin.sh <out-dir>
```

## CUDA Backend Architecture

### No link-time GPU dependency — `dlopen` at runtime

The `plowrt` binary is compiled **without** `-lcuda`. The CUDA driver library
(`libcuda.so.1`) is loaded at runtime through `dlopen` (via `libloading`), and
every needed `cu*` entry point is resolved by symbol name at startup. This
means:

* **The binary is portable** — it runs on machines with no NVIDIA driver at
  all (falls back to the CPU reference backend), and the same binary serves
  GPU-accelerated inference wherever `libcuda.so.1` is reachable.
* **No CUDA toolkit is required at compile time** — only standard Rust +
  system C compiler. The toolkit (`nvcc`) is needed only to build the
  interpreter `.cubin` objects (prebuilt binaries ship with assets).
* **NVIDIA + AMD coexist** — `libcuda`'s `cu*` symbols and `libamdhip64`'s
  `hip*` symbols share no names, so a heterogeneous build can `dlopen` both
  in one process.

### Driver probe order

At startup, `plowrt` tries these paths in order (first successful `dlopen` wins):

1. `$PLOW_LIBCUDA` environment variable (explicit override)
2. `libcuda.so.1` (system ld.so resolution)
3. `libcuda.so` (dev symlink)
4. `/usr/lib/x86_64-linux-gnu/libcuda.so.1` (Debian/Ubuntu)
5. `/usr/local/nvidia/lib64/libcuda.so.1` (NVIDIA container runtime)
6. `/usr/lib64/libcuda.so.1` (RHEL/Fedora)

If all probes fail, the runtime logs a prominent warning and falls back to the
CPU reference backend. Set `RUST_LOG=trace` to see each probe attempt.

### Building with CUDA support

```bash
nix develop                               # toolchain shell

# Full GPU server binary (CUDA driver backend + real tokenizer + safetensors):
cargo build --release -p plowrt --features cuda,hf-tokenizer

# The cargo build needs NO nvcc and NO CUDA toolkit — only the Rust compiler.
# The interpreter cubins (prebuilt .cubin objects loaded at runtime) DO need nvcc:
scripts/build_sm120_cubin.sh <out-dir>    # sm_120a decode + prefill objects
```

### Runtime requirements (deploy)

Only the **NVIDIA kernel-mode driver** needs to be installed (provides
`libcuda.so.1`). No CUDA toolkit, no `nvcc`, no cuDNN — the interpreter cubins
are prebuilt artifacts shipped alongside the model assets.

### CPU fallback

When no GPU driver is found, `plowrt` falls back to the CPU reference backend.
The same compiled packet programs are interpreted identically — correctness is
preserved — but performance is orders of magnitude slower (useful only for
testing and development). The fallback is logged with a prominent warning
banner that is impossible to miss in production.

## Compile a model (plowc)

```bash
# Decode+prefill PLOWDEV device blob for a Gemma-4 checkpoint:
#   --hf-dir   checkpoint directory
#   --max-ctx  max context tokens (e.g. 131072)
#   --n-cu     executor (SM) count of the target GPU
#   --out      a .pkt path (bare blob) or a directory (full servable bundle:
#              model.pkt + weights.json)
PLOW_UNISEG=1 cargo run --release -p plowc -- \
    --hf-dir /path/to/gemma-4-12B-it --emit devblob --max-ctx 131072 --n-cu 188 \
    --out model.pkt
```

> The standalone `gemma4`/`tinygemma` binaries that predated `--emit devblob`
> are deprecated; build them with `--features legacy-gemma-bins` if still needed.

**Emission model.** `plowc` emits **op-level packets** — one record per operator
(GEMM/flash/norm), each referencing whole weight *tensors* by handle. On-device
block tiling (`bm/bn/bk` in the op record) is a runtime parameter the persistent
kernel loops over on-chip; there is **no per-weight-tile packet**. Programs are
**chunk-level**: prefill emits one program per sequence-chunk *bucket* (the
`[128…8192]`-row ladder, `t == chunk size`), decode one per batch size. Only the
physical weight byte *layout* is chosen offline by the emitter — the hook the fp8
twins use today (and where a tensor-core-aware tile layout would live).

Useful knobs:
- `PLOW_UNISEG=1` — single-segment programs (required for the prefill buckets
  on the sm_120 interpreter).
- `PLOW_DECODE_BATCH=B` — emit a batched decode program (B ∈ 1..8) for
  multi-user serving (WS-GEMV shares weight reads across streams). `B=1` blobs
  are byte-identical to unset.
- `PLOW_FP8_HEAD=1` — emit an e4m3 tied embed/lm_head (rtx-19 E5). −3.4/−3.5%
  decode TPOT (the lm_head is the biggest fixed-cost decode op; the win is
  ctx-independent so it's largest at short ctx). Requires the fp8 twin to
  **include** the embed/lm_head tensor (the stock twins do not — regenerate).
- `PLOW_FUSE_ARGMAX=1` — fold greedy argmax into the lm_head GEMV epilogue
  (byte-identical, ~0 perf — the logit round-trip is ~0.1%; a correctness-neutral
  cleanup, kept as a flag).

### Enabling fp8 (a compile/emit-time decision, off by default)

fp8 is **not** a runtime toggle — it is baked when the model is compiled and the
interp cubins are built. The default build is the accuracy-safe **bf16** path;
fp8 is opt-in because it is *lossy* (see the drift notes in the flag tables).
Full fp8 "beat-vLLM" profile:

```bash
# 1. weight twins (per-row e4m3 + f32 scales under fp8/ next to the checkpoint)
python perf-data/harness/quantize_fp8.py <hf-dir>          # + regenerate to include the head for E5
# 2. emit the fp8 blob (fp8 twins auto-detected; add the E5 head + argmax fusion)
PLOW_UNISEG=1 PLOW_FP8_HEAD=1 PLOW_FUSE_ARGMAX=1 \
    cargo run --release -p plowc -- --hf-dir <hf-dir> --emit devblob --max-ctx <ctx> --n-cu 188 --out model.pkt
# 3. build the interp cubins with the fp8 kernel arms
scripts/build_sm120_cubin.sh <out> -DPLOW_NV_W8A8=ON -DPLOW_FP8_KV=ON
```

Measured (gemma-4-31B, single-user): fp8 **decode beats vLLM-fp8** (−41% vs vLLM
bf16, parity-to-−3% vs vLLM fp8); fp8 **prefill** beats vLLM-bf16 at 32k and
closes the short-ctx gap to ~1.1–1.3× (still trails vLLM-fp8, which uses
cudagraphs + FA-class flash). `PLOW_FP8_KV` doubles the 31B concurrency ceiling.

## Run the server (plowrt)

Assemble an assets directory:

```
assets/
  model.pkt                  # from plowc
  interp_sm120.cubin         # decode object   (build_sm120_cubin.sh)
  interp_sm120_pf.cubin      # prefill object  (optional but strongly advised)
  tokenizer.json             # from the checkpoint
  checkpoint -> /path/to/hf-checkpoint-dir   # symlink (weights are mmap'd)
  weights.json               # minimal manifest ({"buckets": []})
```

Then:

```bash
plowrt serve --assets assets/ --port 8080
# OpenAI-compatible:
curl localhost:8080/v1/chat/completions -d '{
  "model": "<manifest network name>",
  "messages": [{"role":"user","content":"What is the capital of France?"}]
}'
```

Streaming (`"stream": true`), `/v1/models`, `/healthz`, `/metrics`, and
`/trace` are supported. Multiple `--assets` dirs register multiple models.

## Feature flags & knobs

Perf features from the rtx-11/12/13 campaigns are gated so the **default build is
a fixed, validated configuration**; every flag is an A/B control with a
correctness gate (token-identity or a bit-exact vs-f32 oracle). Unset = shipped
default. Grouped by layer; measured results live in `perf-data/`.

**Kernel build flags** (`nvcc -D…` / CMake `option`; the shipped cubin uses each
at its default):

| flag | default | effect |
|---|---|---|
| `PLOW_NV_FA_PX4` | **1 (on)** | PX-4 restructured hd512 full-layer flash (register softmax + 8-warp QK). −24% flash op, −16% end-to-end 128k prefill. |
| `PLOW_NV_FA_PIPE` | **1 (on)** | cp.async KV-stream flash pipeline (T5); −81% flash at 128k. |
| `PLOW_NV_FA_TMA` | 0 | TMA (`cp.async.bulk`) KV staging arm — ~2× *slower* on sm_120 (single-CTA 1-D bulk); kept as the A/B control. |
| `-DPLOW_NV_W8A8=ON` | OFF | PX-2 native w8a8 fp8 mainloop (BK64 + `Swizzle<3,4,3>` + `mma.sync.m16n8k32`). −48% GEMM vs bf16, −30…34% end-to-end prefill; on 31B it flips the vs-vLLM-bf16 prefill deficit to a win at 32k. Needs the fp8 weight twins. |
| `-DPLOW_FP8_KV=ON` | OFF | rtx-19 E3 — e4m3 KV cache (halves KV bytes). Lifts the 31B multi-user batch cap (bf16-KV B=4 → fp8-KV B=7–8), beats vLLM fp8kv at ≥16k ctx; ~−1.2% decode TPOT at 32k. **Lossy** (e4m3 KV, ~3–6% logit relL2, greedy diverges after ~21 tokens) — that's why it is opt-in. |
| `PLOW_NV_SEG_GEMM_BN64` | off | PX-3 BN=64 occ-2 prefill seg-GEMM object (A/B vs the BN=128 occ-1 default; small GLU-only win). |
| `PLOW_NV_SZ`, `PLOW_NV_ZG` | off | **Experimental**, lossless bf16 weight decompression (SplitZip decode / fused ZipGEMM). Bit-exact but measured non-viable for multi-user decode on sm_120 (recon competes with an already bandwidth-saturated weight stream); retained as bit-exact A/B references. |
| `PLOW_NV_TRACE_DECODE`, `PLOW_NV_TRACE_PF`, `PLOW_NV_SKELETON_DECODE` | off | per-op cycle trace / dispatch-floor diagnostics. |

**Serving / runtime knobs** (`plowrt` env):

| var | default | effect |
|---|---|---|
| `PLOW_PF_BATCH=1` | off | PX-1 cross-request batched prefill — packs waiting requests' prefill chunks into one launch (shared weight reads); block-diagonal varlen flash keeps per-request attention isolated. +27% saturated multi-user throughput. Off = serialized prefill (byte-identical default). |
| `PLOW_PF_INTERLEAVE=N` | 2048 | batched-prefill interleave quantum (rows admitted per tick under decode load). |
| `PLOW_PF_CHUNK=C` | 0 (off) | **Experimental** per-request prefill chunk-row cap: clamps each request's per-launch rows so more requests co-pack (R ≈ quantum/C). Measured ~10% regression at B=8 (finer chunks add launches + attention KV re-reads); off = uncapped (byte-identical). |
| `PLOW_PF_CHUNK_COST=R` | 512 | cost of ONE prefill launch, in padded-row equivalents — the currency the bucket pick minimizes (`rows + R × launches`). A launch re-streams every layer's weights, so it is not free: measured `ttft_ms = 0.112·rows + 60.1·chunks` on sm_120 / 12B (±2.8%), i.e. **60 ms ≈ 537 rows**. Charging it stops a just-under-rung tail cascading down the ladder (8190 rows ran `[4096,2048,1024,512,128×4]` = 8 launches, +25% TTFT, vs the 2-row-padded `[8192]`). `0` = the old pure-minimum-padding objective. |
| `PLOW_PF_COVER=1` | off | restore the covering-bucket prefill policy (exact-parity A/B vs the cost-aware default). |
| `PLOW_PF_PACKLOG=1` | off | per-launch pack diagnostics (requests/rows/bucket + prefill/decode wall split). |
| `PLOW_VMM_PREFIX=1` | off | VMM-backed KV prefix cache — a new request attaches the blocks a previous one already built instead of re-prefilling them. Only the FULL-attention layers are VMM-backed (sliding layers stay on `cudaMalloc`), and reuse is block-granular, so `cached_tokens` floors to a block multiple. Measured on 12B: warm TTFT 3.6× (4k) → **23.8× (128k)**, and warm TTFT is near-flat in context; cold costs ~0.2–1.5% above 8k. Incompatible with `PLOW_PF_BATCH=1` (which is then ignored, with a warning). |
| `PLOW_VMM_BLOCK_MIB=M` | 2 | VMM sharing block size. The driver granularity (2 MiB ≈ 4096 tokens at hd256 bf16) is the finest match unit, which is what makes shared system prompts and multi-turn histories actually hit. Raise (e.g. 64) for 128k-dedup work. |
| `PLOW_VMM_CACHE_MIB=M` | 0 | cap on retained (unreferenced) VMM blocks; `0` = no cache. |
| `PLOW_NV_SCHED=1` | **on** | global-queue interpreter scheduler; the static per-block-stream path is the build-time A/B control. |

Loader/asset overrides: `PLOW_NV_CUBIN[_PF]`, `PLOW_NV_KERNEL[_PF]`,
`PLOW_NV_SMEM_PF`, `PLOW_CHECKPOINT`, `PLOW_LIBCUDA`.

## Standalone harnesses (kernel-level, no server)

`runtime/tests/` carries the HF-parity-gated chat/bench harnesses the perf
campaigns use, e.g. `gemma4_sm120_chat.cu` (chunked prefill → decode;
`PLOW_PREFILL=1`, `PLOW_FP8_DIR=<fp8-twin dir>`). Build them with nvcc
`-arch=sm_120a` or via `runtime/CMakeLists.txt` (`PLOW_CUDA=ON`). The numeric
oracle is `runtime/tests/sm120_interp_op_test.cu` — every interpreter op body
vs an f32 CPU reference, with a negative-control build that must fail.

## Benchmarks

Measured results live in `perf-data/` (one JSON + MD per campaign;
`consolidate_perf.py` regenerates the flat index). Multi-user sweeps use
HuggingFace `inference-benchmarker` via `perf-data/bench_ib.sh` against both
plow and vLLM with identical profiles.
