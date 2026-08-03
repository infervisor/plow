# plow — Packet Language for On-device Warps

A product of **Infervisor**.

plow is an LLM inference stack built around one idea: compile the whole model
into a **packet stream** and run it with a **persistent on-device interpreter**
— one cooperative kernel launch that stays resident, with warp-granularity op
bodies and zero per-op dispatch. The compiler (`plowc`) lowers a checkpoint into
packets; the runtime (`plowrt`) loads them and serves an OpenAI-compatible API;
Lean checks (`lean-plow/`, `crates/lean_verify`) cover the non-obvious rewrites
and the counter protocol.

**Supported today:** Gemma-4 (12B / 31B dense, 26B-A4B MoE), Qwen3, Llama-3.1 —
bf16 and fp8 (weight-only e4m3) — on NVIDIA consumer Blackwell (sm_120, e.g.
RTX 5090 / RTX PRO 6000 Blackwell) and AMD CDNA (gfx942/gfx950).

**Architecture:** see [`docs/arch/`](docs/arch/00-overview.md) — the overview
indexes the compiler pipeline, tile graph, scheduler, packet ABI, counter
system, runtime, cost model, formal verification, and multi-GPU. The full
build-system rationale (one binary for CPU+GPU, `dlopen`, static linking, TLS)
is in [`docs/BUILD.md`](docs/BUILD.md). **Every emit/build/runtime flag** is
catalogued in [`docs/flags-reference.md`](docs/flags-reference.md); this README
keeps only what you need to build, compile a model, and serve — plus the handful
of traps that break an asset *silently*.

> **On `crates/rewrite` (egglog):** it runs, but it is **advisory — no rewrite it
> finds reaches a GPU.** Both `plowc` paths discard their fused graph, and
> `crates/devgen` (the emitter every shipped asset comes out of) has no
> dependency on it. The fusions actually in a packet (`GemvQkv`, `GemvGlu`,
> `NormResidualNorm`) are hand-written in `devgen`. This is not a bug to fix on
> the way to a win: deleting 100 packets/token is worth ≤0.064 ms on gfx950 and
> the un-fused arm ran *faster*. See `docs/arch/01-compiler-pipeline.md`.

## Requirements

- Rust + Lean via **nix**: `nix develop` in the repo root provides cargo/rustc,
  cmake, gcc, and elan. (First run: install nix, enable flakes.)
- **NVIDIA path:** driver + **CUDA ≥ 12.9** (13.0 tested), `nvcc` on PATH,
  sm_120 GPU for the shipped interpreter objects.
- **AMD path:** a local **ROCm ≥ 7.2.4** install (`/opt/rocm`) for `hipcc` and
  `clang-offload-bundler` — *not* from nix (the code objects are built by the
  system ROCm toolchain). **7.0.2 is too old**: clang-20 lands the fp8/MLA
  prefill objects at 262 regs / occ 1 and the build's register-cliff gate
  rejects them; 7.2.4 (clang-22) lands them at 256 / occ-2.
- A HuggingFace checkpoint directory (safetensors + config.json + tokenizer).

## Build

```bash
nix develop                               # toolchain shell
cargo build --workspace --release         # compiler + runtime (CPU paths)
cargo test  --workspace                   # full test suite

# GPU server binary. Both vendor backends are compiled in but neither is LINKED
# — cuda dlopens libcuda.so.1, hsa dlopens libhsa-runtime64.so — so ONE binary
# serves NVIDIA, AMD, and CPU-fallback hosts. The AMD feature is `hsa` (direct
# ROCr, no HIP), not `rocm`:
cargo build --release -p plowrt --features cuda,hsa,hub

# Lean verifier (universal rewrite lemmas + the plow_verify CLI):
(cd lean-plow && lake build)

# Interpreter objects — NVIDIA cubins (sm_120a) and AMD code objects (gfx950):
scripts/build_sm120_cubin.sh <out-dir>
cmake -S runtime -B build-amd -DPLOW_GFX950_HSACO=ON && cmake --build build-amd -j
```

The `plowrt` binary has **no link-time GPU dependency** — the driver is loaded at
runtime via `dlopen`, so the same binary runs GPU-accelerated where a driver is
present and falls back to the CPU reference backend (correct, orders of magnitude
slower, logged with a loud banner) where none is. Building needs only Rust + a C
compiler; `nvcc`/`hipcc` are needed *only* to build the interpreter objects,
which ship prebuilt alongside assets. Driver probe order, the CPU-fallback
contract, static-linking limits, and the (not-yet-wired) kernel plugin ABI are
all in [`docs/BUILD.md`](docs/BUILD.md).

> Binaries built in the nix shell link the nix glibc loader. On a host where the
> nix store is only visible inside nix (e.g. `nix-portable`), run them through
> `nix develop --command ...` — outside it the kernel reports the missing ELF
> interpreter as a bare `No such file or directory` on a file that plainly exists.

### AMD interpreter objects

cmake drives the compile; **`hipcc` comes from the local ROCm**, not nix. One
configure emits every object, so a packet cannot reach for a variant the build
quietly skipped:

```bash
nix develop                               # for cmake — NOT for hipcc
cmake -S runtime -B build-amd -DPLOW_GFX950_HSACO=ON -DPLOW_HSACO_ARCH=gfx950
cmake --build build-amd -j
```

17 code objects land in `build-amd/hsaco/` (`interp_{prefill,decode,flash}.elf`
+ `_gq` twins, the fp8 / fp8-KV / MX-FP4 / MLA-prefill variants, and
`test_kernels.elf`). They keep the `.elf` suffix because `gemma4_chat.c` opens
them by literal filename; they are hsaco code objects. Variant axes default
**ON** — `-DPLOW_HSACO_{GQ,FP8,FP8KV,MXFP4,MLA}=OFF` to drop one. Two gates run
*inside* the build so both are build errors, not runtime ones: a **register
cliff** check (an 8-wave interpreter over 256 VGPR+AGPR drops to 1 wave and fails
to launch with `HSA_STATUS_ERROR_INVALID_ISA`; the 4-wave flash object is allowed
512/occ-1) and a **kernel-symbol** check (entry points resolved by name).
`scripts/build_gfx950.sh` additionally builds the host `chat` harness and runs
the `asm_audit.py` instruction-selection pass. Only the amdgpu kernel driver +
ROCr (`libhsa-runtime64.so`) need be present to *deploy*.

## Compile a model (plowc)

```bash
# Decode+prefill device blob for a Gemma-4 checkpoint.
#   --hf-dir   checkpoint directory      --max-ctx  max context tokens
#   --n-cu     executor (SM) count       --out      .pkt path, or a dir for a
#                                                    full servable bundle
# NVIDIA (sm_120):
PLOW_UNISEG=1 cargo run --release -p plowc -- \
    --hf-dir /path/to/gemma-4-12B-it --emit devblob --max-ctx 131072 --n-cu 188 \
    --out model.pkt

# AMD (gfx950): NO PLOW_UNISEG, and target the GPU explicitly so --arch and
# --gpu agree. Verify build.json shows 121 segments per prefill bucket.
cargo run --release -p plowc -- \
    --hf-dir /path/to/gemma-4-31B-it --emit devblob --arch gfx950 --gpu mi355x \
    --max-ctx 131072 --out model.pkt
```

> ⚠️ **`PLOW_UNISEG=1` is NVIDIA-only. Never pass it when targeting gfx950** — it
> collapses every op into one segment (right on sm_120, which runs one
> cooperative launch) but on AMD it sends the **entire prefill program to the
> 4-wave flash object**, which silently drops every GEMM/norm/lm_head. Prefill
> "completes" fast and the logits are all zero. A correct Gemma-4 31B emit has
> **121 segments per prefill bucket** (`2·layers + 1`) — check `build.json`.

`plowc` emits one **instruction** per operator, but the unit of *work* is finer:
each program carries a per-CU stream of `StreamEnt{inst, slice, …}` tasks (one
per `(op, slice)`), so a devblob is an **SM-level task graph**, not a kernel
graph. Programs are chunk-level (prefill: one per sequence-chunk bucket; decode:
one per batch size). The full emission model, the per-CU-stream vs global-queue
scheduler note, and every emit knob (`PLOW_DECODE_BATCH`, `PLOW_MAX_CHUNK`,
`PLOW_PF_LADDER`, `PLOW_L2_PLACE`, `PLOW_FINE_FORCE`, …) are in
[`docs/flags-reference.md`](docs/flags-reference.md).

> The standalone `gemma4`/`tinygemma` binaries that predated `--emit devblob` are
> deprecated; build them with `--features legacy-gemma-bins` if still needed.

### fp8 (compile/emit-time, off by default)

fp8 is **not** a runtime toggle — it is baked when the model is compiled and the
interp objects are built. The default is the accuracy-safe **bf16** path; fp8 is
opt-in because it is *lossy*. Precision flags are named by **axis** — `PLOW_W8A16`
(fp8 weights / bf16 acts), `PLOW_W8A8` (fp8 weights + acts), `PLOW_MXFP4`,
`PLOW_KV_FP8` (fp8 KV cache); the axes compose and an axis a family can't realize
is *refused*, not silently downgraded. The spelling differs by model family and
GPU — see the axis/family table in
[`docs/flags-reference.md`](docs/flags-reference.md).

```bash
# 1. weight twins (per-row e4m3 + f32 scales under fp8/ next to the checkpoint)
python perf-data/harness/quantize_fp8.py <hf-dir>
# 2. emit the fp8 blob (twins auto-detected). NVIDIA:
PLOW_UNISEG=1 PLOW_FP8_HEAD=1 PLOW_FUSE_ARGMAX=1 \
    cargo run --release -p plowc -- --hf-dir <hf-dir> --emit devblob --max-ctx <ctx> --n-cu 188 --out model.pkt
# AMD (gfx950) — no PLOW_UNISEG, axis-named precision flags:
PLOW_W8A8=1 PLOW_FP8_HEAD=1 PLOW_FUSE_ARGMAX=1 \
    cargo run --release -p plowc -- --hf-dir <hf-dir> --emit devblob --arch gfx950 --gpu mi355x --max-ctx <ctx> --out model.pkt
# 3. build the interp objects WITH the matching fp8 kernel arms:
scripts/build_sm120_cubin.sh <out> -DPLOW_NV_W8A8=ON -DPLOW_FP8_KV=ON
```

> ⚠️ **The emit flag and its `-D` build flag must agree.** An fp8 packet served
> against an interpreter built *without* the matching arm hits `default: __trap()`
> and every launch dies with `CUDA_ERROR_LAUNCH_FAILED` — it looks like a driver
> problem and is not. On gfx950, plain `PLOW_FP8=1` is *refused* at emit (it means
> w8a16, but the gfx950 GEMM arm is w8a8); use `PLOW_W8A8=1`. Keep `--arch` and
> `--gpu` in agreement.

Measured (gemma-4-31B, single-user): fp8 **decode beats vLLM-fp8** (−41% vs vLLM
bf16); fp8 **prefill** beats vLLM-bf16 at 32k. `PLOW_KV_FP8` doubles the 31B
concurrency ceiling — but it is lossy and *degrades with context* (validate
retrieval at *your* context length, not a short one). Full numbers and the KV
caveats: [`docs/flags-reference.md`](docs/flags-reference.md).

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

```bash
plowrt serve --assets assets/ --port 8080
curl localhost:8080/v1/chat/completions -d '{
  "model": "<manifest network name>",
  "messages": [{"role":"user","content":"What is the capital of France?"}]
}'
```

Streaming (`"stream": true`), `/v1/models`, `/healthz`, `/metrics`, and `/trace`
are supported. Multiple `--assets` dirs register multiple models.

**Serving knobs** (`plowrt` env) tune prefill batching, chunking, and the VMM KV
prefix cache — the useful defaults work out of the box. Chunked prefill is on by
default; `PLOW_PF_BATCH=1` packs waiting requests' prefill into one launch
(+27% saturated throughput); `PLOW_VMM_PREFIX=1` reuses a prior request's KV
prefix (warm TTFT up to 23.8× at 128k). One thing plow does **not** do: mixed
prefill⊕decode in a single launch (a tick that does both runs two launches). The
full table, the three easily-conflated batching modes, and every loader override
are in [`docs/flags-reference.md`](docs/flags-reference.md).

## Feature flags & configurations

Perf features from the rtx-11/12/13 campaigns are gated so the **default build is
a fixed, validated bf16 configuration**; every flag is an A/B control with a
correctness gate, and **unset = shipped default**. Almost nobody needs the
individual knobs — pick one of the four configurations below and use its flags.
The **emit** column is `plowc` env; the **build** column is `nvcc -D…` / CMake.

| you want | emit | build | notes |
|---|---|---|---|
| **Default** — validated bf16 | `PLOW_UNISEG=1` | *(none)* | The shipped configuration. |
| **fp8 weights** — faster prefill GEMM | `PLOW_UNISEG=1 PLOW_W8A8=1` | `-DPLOW_NV_W8A8=1` | −48% GEMM, −30…34% prefill. Needs fp8 twins. **Both flags or neither** — mismatch = `__trap()`. |
| **Long context, multi-user** — fp8-KV | `PLOW_UNISEG=1 PLOW_W8A8=1 PLOW_FP8_KV_FULL=1` | `-DPLOW_NV_W8A8=1 -DPLOW_FP8_KV=ON -DPLOW_FP8_KV_FASTPF=ON` | Halves KV bytes (B=8 at 127k fits 32 GB). `FASTPF=ON` keeps prefill on the fast PIPE=1 arm (−21% at 67k). ⚠️ Lossy, degrades with context — validate retrieval at *your* ctx. |
| **Legacy all-layer fp8 KV** | `… PLOW_FP8_KV=1` (no `_FULL`) | `-DPLOW_FP8_KV=ON` (FASTPF off) | e4m3 every layer; `FASTPF` must stay OFF (traps under PIPE=1). Prefer the mixed row above. |

The **~200 individual build/emit/runtime knobs** — object-selection arms,
precision, attention (`FA_PX4`/`FA_PIPE` carry the long-context wins), GEMM/GEMV,
MoE, scheduling, per-model-family fusion arms, measurement-only ablations, and
the runtime serving knobs — are documented, with measured deltas and the reason
each default sits where it does, in
[`docs/flags-reference.md`](docs/flags-reference.md).

## Standalone harnesses & benchmarks

`runtime/tests/` carries the HF-parity-gated chat/bench harnesses the perf
campaigns use (e.g. `gemma4_sm120_chat.cu`). The numeric oracle is
`runtime/tests/sm120_interp_op_test.cu` — every interpreter op body vs an f32 CPU
reference, with a negative-control build that must fail. Build with nvcc
`-arch=sm_120a` or via `runtime/CMakeLists.txt` (`PLOW_CUDA=ON`).

Measured results live in `perf-data/` (one JSON + MD per campaign). Multi-user
sweeps use HuggingFace `inference-benchmarker` via `perf-data/bench_ib.sh`
against both plow and vLLM with identical profiles.
</content>
