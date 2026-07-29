# plow — Packet Language for On-device Warps

A product of **Infervisor**.

plow is an LLM inference stack built around one idea: compile the whole model
into a **packet stream** and run it with a **persistent on-device interpreter**
— one cooperative kernel launch that stays resident, with warp-granularity op
bodies and zero per-op dispatch. The compiler (`plowc`) lowers a checkpoint
into packets; the runtime (`plowrt`) loads them and serves an OpenAI-compatible
API; Lean checks (`lean-plow/`, `crates/lean_verify`) cover the non-obvious
rewrites and the counter protocol.

> **On `crates/rewrite` (egglog).** It runs, but it is **advisory: no rewrite it
> finds reaches a GPU.** Both `plowc` paths discard their fused graph, and
> `crates/devgen` — the emitter every shipped asset comes out of — has no
> dependency on `rewrite` at all. On Gemma-4-31B the pass reports
> `fusions_found=662` and applies **0** of them. The fusions that are actually in
> a packet (`GemvQkv`, `GemvGlu`, `NormResidualNorm`) are hand-written in
> `devgen`. This is not a regression to fix on the way to a win: measured on
> gfx950, deleting 100 packets/token is worth ≤0.064 ms and the un-fused arm ran
> *faster*. See `docs/arch/01-compiler-pipeline.md`.

Supported today: Gemma-4 (12B / 31B dense, 26B-A4B MoE), Qwen3, Llama-3.1 —
bf16 and fp8 (weight-only e4m3) — on NVIDIA consumer Blackwell (sm_120, e.g.
RTX 5090 / RTX PRO 6000 Blackwell) and AMD CDNA (gfx942/gfx950).

## Requirements

- Rust + Lean via **nix**: `nix develop` in the repo root provides cargo/rustc,
  cmake, gcc, and elan. (First run: install nix, enable flakes.)
- NVIDIA path: driver + **CUDA ≥ 12.9** (13.0 tested); `nvcc` on PATH.
  sm_120 GPU for the shipped interpreter objects.
- AMD path: a **local ROCm ≥ 7.2.4** install (`/opt/rocm`), for `hipcc` and
  `clang-offload-bundler`. Not from nix — the code objects are built by the
  system ROCm toolchain. **7.0.2 is too old**: its clang-20 lands the fp8 and
  MLA prefill objects at 262 regs / occupancy 1 and the build's register-cliff
  gate rejects them. 7.2.4 (clang-22) lands them at 256 / occ-2.
- A HuggingFace checkpoint directory (safetensors + config.json + tokenizer).

## Build

```bash
nix develop                               # toolchain shell
cargo build --workspace --release         # compiler + runtime (CPU paths)
cargo test  --workspace                   # full test suite

# GPU server binary. Both vendor backends are compiled in but neither is LINKED
# — `cuda` dlopens libcuda.so.1, `hsa` dlopens libhsa-runtime64.so — so one
# binary serves NVIDIA, AMD, and CPU-fallback hosts. The AMD feature is named
# `hsa` (direct ROCr, no HIP), not `rocm`:
cargo build --release -p plowrt --features cuda,hsa,hub

# Lean verifier (universal rewrite lemmas + the plow_verify CLI that
# crates/lean_verify drives). elan installs the lean-toolchain pin on first use:
(cd lean-plow && lake build)

# Interpreter objects — NVIDIA cubins (sm_120a) and AMD code objects (gfx950):
scripts/build_sm120_cubin.sh <out-dir>
cmake -S runtime -B build-amd -DPLOW_GFX950_HSACO=ON && cmake --build build-amd -j
```

Binaries built in the nix shell link the nix glibc loader. On a host where the
nix store is only visible inside nix (e.g. `nix-portable`), run them through
`nix develop --command ...` — outside it the kernel reports the missing ELF
interpreter as a bare `No such file or directory` on a file that plainly exists.

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
# The interpreter cubins (prebuilt .cubin objects loaded at runtime) DO need nvcc.
# The build itself lives in runtime/CMakeLists.txt (-DPLOW_SM120_CUBIN=ON); this
# script is a thin wrapper over it, so there is ONE definition of every object's
# define set rather than two that can drift apart:
scripts/build_sm120_cubin.sh <out-dir>    # sm_120a decode + prefill objects
```

`plowc` writes a `build.json` beside `model.pkt` describing what the packet needs of
the object that runs it — opcodes and shapes derived from the emitted instruction
stream, plus the rule-derived tuning constants (`GV_MM_MAX`, `PLOW_NV_FA_GF_FULL`).
`plowc --emit devblob+cubin` builds the matching object from it. That form — and only
that form — needs a CUDA toolkit; `--emit devblob` still needs none.

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

## AMD (ROCm) Backend

Same shape as the CUDA path: `plowrt` is built with no link-time GPU dependency
(`--features hsa` `dlopen`s `libhsa-runtime64.so` — direct ROCr, not HIP), and
the persistent interpreter ships as prebuilt device code objects the runtime
loads at startup.

### Building the interpreter code objects

cmake drives the compile; **`hipcc` comes from the local ROCm**, not from nix.
One configure emits every object, so a packet cannot reach for a variant the
build quietly skipped:

```bash
nix develop                               # for cmake — NOT for hipcc

cmake -S runtime -B build-amd -DPLOW_GFX950_HSACO=ON -DPLOW_HSACO_ARCH=gfx950
cmake --build build-amd -j
```

17 code objects land in `build-amd/hsaco/` — `interp_{prefill,decode,flash}.elf`,
each with its global-queue `_gq` twin, plus the fp8 (w8a16 decode / w8a8 prefill),
fp8-KV, MX-FP4, and MLA-prefill variants, and `test_kernels.elf` (the golden
`__device__` wrappers, rebuilt with the interpreter because they share the same
`op_*.h` bodies). They keep the `.elf` suffix because `gemma4_chat.c` opens them
by literal filename; they are hsaco code objects.

Variant axes, all default **ON** — `-DPLOW_HSACO_{GQ,FP8,FP8KV,MXFP4,MLA}=OFF` to
drop one. Toolchain overrides: `PLOW_HSACO_HIPCC`, `PLOW_HSACO_BUNDLER`,
`PLOW_HSACO_DIR`, `PLOW_HSACO_EXTRA_DEFINES`.

Two gates run inside the build, so both failures are build errors rather than
runtime ones:

* **Register cliff** — an 8-wave interpreter over 256 total regs (VGPR+AGPR)
  drops to 1 wave/SIMD and past the budget fails to launch at all with
  `HSA_STATUS_ERROR_INVALID_ISA`. The 4-wave flash object is allowed 512/occ-1,
  and *its* spill is intentional (the Q-hoist spills Q to scratch, cheaper than
  the L2 re-read it replaces).
* **Kernel symbol** — the entry points are resolved by name, so an object that
  compiled fine but whose symbol moved is caught here.

`scripts/build_gfx950.sh` remains the way to get the two things not ported to
cmake: the host `chat` harness (a gcc link, not a kernel artifact) and the
`scripts/asm_audit.py` instruction-selection pass, which wants the whole object
set at once and asserts on the disassembly — the check that catches a kernel
which launches, is correct, and is silently slow because the backend picked a
narrow MFMA.

### Runtime requirements (deploy)

Only the **amdgpu kernel driver + ROCr** (`libhsa-runtime64.so`) need to be
present. No HIP, no hipcc — the code objects are prebuilt artifacts shipped
alongside the model assets.

## Compile a model (plowc)

```bash
# Decode+prefill PLOWDEV device blob for a Gemma-4 checkpoint:
#   --hf-dir   checkpoint directory
#   --max-ctx  max context tokens (e.g. 131072)
#   --n-cu     executor (SM) count of the target GPU
#   --out      a .pkt path (bare blob) or a directory (full servable bundle:
#              model.pkt + weights.json)
# NVIDIA (sm_120). On gfx950 DROP PLOW_UNISEG — see the warning below the block.
PLOW_UNISEG=1 cargo run --release -p plowc -- \
    --hf-dir /path/to/gemma-4-12B-it --emit devblob --max-ctx 131072 --n-cu 188 \
    --out model.pkt

# AMD (gfx950): no PLOW_UNISEG, and target the GPU explicitly so --arch and
# --gpu agree. Verify build.json shows 121 segments per prefill bucket.
cargo run --release -p plowc -- \
    --hf-dir /path/to/gemma-4-31B-it --emit devblob --arch gfx950 --gpu mi355x \
    --max-ctx 131072 --out model.pkt
```

> The standalone `gemma4`/`tinygemma` binaries that predated `--emit devblob`
> are deprecated; build them with `--features legacy-gemma-bins` if still needed.

**Emission model.** `plowc` emits one **instruction** per operator
(GEMM/flash/norm), each referencing whole weight *tensors* by handle — but the
unit of *work* on the device is finer. Each program carries a **per-CU stream of
`StreamEnt{inst, slice, …}` records**: one task per `(op, slice)` with its own
dependency edges, indexed `prog.stream + prog.stream_ofs[blockIdx.x]`. `slice`
IS the tile — `op_gemm.cuh` strides `for tile = slice; tile < ntiles; tile +=
nblk` over its output tiles. One Gemma-4-12B layer at T=1024 emits **2,890
tasks**; a prefill chunk emits ~138,720 (PX-19, measured). So the devblob is an
**SM-level task graph**, not a kernel graph.

Two caveats. On-device block tiling (`bm/bn/bk`) is still a runtime parameter —
there is no per-weight-*tile* packet, only a per-output-*slice* task. And the
per-CU streams are **not what the shipped cubin uses**: `PLOW_NV_SCHED` defaults
to 1, the global-queue scheduler, where every block claims work off one atomic
cursor — so `stream_ofs` is the static-schedule A/B control, not the live path.
(An earlier revision of this paragraph said there is "no per-tile packet" flatly;
that was wrong and sent one investigation down a dead end.) Programs are
**chunk-level**: prefill emits one program per sequence-chunk *bucket* (the
`[128…8192]`-row ladder, `t == chunk size`), decode one per batch size. Only the
physical weight byte *layout* is chosen offline by the emitter — the hook the fp8
twins use today (and where a tensor-core-aware tile layout would live).

Useful knobs:
- ⚠️ **`PLOW_UNISEG=1` is NVIDIA-only. Do not pass it when targeting gfx950** —
  every recipe on this page that sets it produces a *broken* AMD asset, and the
  breakage is silent. It collapses every op into one segment, which is right on
  sm_120 (that interpreter runs one cooperative launch and never reads a wave
  class) and destroys AMD's wave-class split. With one segment, segment 0
  contains the flash packets, the class-4 test matches, and the **entire prefill
  program is dispatched on `interp_flash`** — whose body is
  `if (op == FLASH_PREFILL…)` with no switch, so every GEMM, norm and lm_head is
  silently dropped. Prefill "completes" in 8.7 ms instead of 72.1 and the logits
  are all zero. A correct Gemma-4 31B emit has **121 segments per prefill
  bucket** (`2·layers + 1`); check `build.json` if in doubt.
- `PLOW_UNISEG=1` — single-segment programs (required for the prefill buckets
  on the sm_120 interpreter).
- `PLOW_DECODE_BATCH=B` — emit a batched decode program (B ∈ 1..8) for
  multi-user serving (WS-GEMV shares weight reads across streams). `B=1` blobs
  are byte-identical to unset.
- `PLOW_MAX_CHUNK=N` — largest prefill chunk for this compile (power of two,
  ≤ 8192). **This caps the bucket ladder**, so it also sets the ceiling for the
  runtime `PLOW_PF_INTERLEAVE` — raising that knob above this value is a no-op.
  Default is *window-derived*: `next_pow2(window)` clamped to [128, 8192], so a
  Gemma-4 asset (window 1024) emits a **1024**-row max chunk, not 8192. The
  tradeoff is prefill launches against sliding KV: the ring is sized
  `next_pow2(window + chunk - 1)`, so at chunk 8192 Gemma-4 rings 16384 rows =
  5.0 GiB/seq, while chunk 1024 rings 2048 = 0.625 GiB/seq — **8×** the KV for
  fewer launches. Bigger chunks are modestly faster per prefill once px4 is on
  (65k prefill, B=2: 2048 → 12.39 s, 4096 → 11.57 s, 8192 → 11.31 s, i.e. **9%**),
  so this is a KV-capacity vs prefill-latency dial: worth raising at B≤2 on a
  large-VRAM part, not at B=8 on 32 GB. Check what your asset actually emitted
  before attributing a prefill cost to chunking.
- `PLOW_PF_LADDER=wave` — derive the prefill bucket rungs from the target's **SM
  count** instead of the default power-of-two ladder (PX-6,
  `perf-data/px6-sm-quantization.md`). Prefill GEMM cost is a *staircase* in
  `tm = ceil(t/128)`: flat between wave boundaries, so rows added inside a tread
  are free and one row past a tread top costs a whole extra wave of every op that
  stepped — measured, `N = 170·128` runs 1 wave in 0.18362 ms and `N = 171·128`
  runs 2 in 0.30368 ms, i.e. **0.6% more work for 65% more time**. The shipped
  `[128, 512, 1024, 2048, 4096, 8192]` rungs are powers of two, which is unrelated
  to where the treads are; on the Gemma-4-12B op mix at `n_cu=170` they give up
  **9.6%** of prefill GEMM time on average over L = 128…4096 (worst cells +41.9%
  at 640 rows, which must be served as 128+512). The tread-top rungs the model
  picks — 1408, 2176, 640, 1792, none a power of two — take the mean loss to
  **1.4%**. Same rung *count*, so blob size and compile time are unchanged; only
  the positions move. Unset ⇒ byte-identical.
  **The ladder is a function of `n_cu` and is NOT portable**: 170 = 2·5·17 and
  188 = 2²·47 put the treads in completely different places, which is the whole
  reason it is derived rather than hardcoded. Emitting for the wrong SM count is
  worse than the power-of-two default.
  **Scope:** this optimises *covering* loss — the padding waste when a prompt
  length falls between rungs. It is worth nothing at long context, where a prompt
  is served as many repetitions of the *max* rung and the interior rungs are never
  used (measured on a 127k prompt: 31.00 s → 30.94 s). Use it for short/medium
  prefill. NVIDIA-only (the rungs assume the sm_120 `PGM_BM/PGM_BN = 128` tile).
- `PLOW_FP8_HEAD=1` — emit an e4m3 tied embed/lm_head (rtx-19 E5). −3.4/−3.5%
  decode TPOT (the lm_head is the biggest fixed-cost decode op; the win is
  ctx-independent so it's largest at short ctx). Requires the fp8 twin to
  **include** the embed/lm_head tensor (the stock twins do not — regenerate).
- `PLOW_FUSE_ARGMAX=1` — fold greedy argmax into the lm_head GEMV epilogue
  (byte-identical, ~0 perf — the logit round-trip is ~0.1%; a correctness-neutral
  cleanup, kept as a flag).
- `PLOW_FINE_FORCE=1` — keep per-slice (**fine**) counter gates instead of the
  default whole-op (**coarse**) ones. The emitter declares a fine edge wherever a
  consumer slice reads only part of a producer (headnorm→flash, flash→merge, MoE
  down→GLU); by default `select_granularity` collapses every *homogeneous* region
  back to coarse, because `lean-plow/Plow/CounterGranularity.lean:collapse` proves
  fine buys nothing when per-slice work is uniform — and it isn't free (an extra
  counter per producer slice, an extra atomic per producer, a wider wait list).
  This lever keeps the fine edge iff it is genuinely *sparse* (some consumer slice
  waits on strictly fewer than all producer slices) so it isolates the recoverable
  straggler gates without paying the 256×256-atomic all-to-all cost. It exists to
  **measure** the real-hardware straggler delta the uniform cost model can't see:
  on dense Gemma it was a wash-to-loss (16.9 → 17.2 ms/token), which is why coarse
  is the default. Lean-safe (a fine list only lowers a threshold / narrows a wait
  set), and **unset = byte-identical** coarse. There is no all-to-all "everything
  fine" mode — see `plans/fine-counter-deadlock-fix.md`.
- `PLOW_L2_PLACE=1` — **L2-domain packet grouping** (compiler half of physical-SM
  locality). *Was `PLOW_NV_PLACE`, still accepted as a deprecated alias: an L2
  domain is a GPC on NVIDIA and an XCD on AMD, and `hwspec` describes both, so the
  NVIDIA-specific name was wrong about its own scope.* Groups the device blob's
  global-queue stream into P per-L2-domain windows, so a full op's slices spread
  evenly across all domains and slice `s` stays in one domain across ops (consumer
  reads producer from the same L2 slice). It does NOT touch `cus` (so it can't
  regress `Builder::split` disjointness) and prints a static allocation report
  (`l2 placement: … map … packets/domain […] skew …%`). **Unset = byte-identical**;
  no-op on unpartitioned GPUs (e.g. consumer Blackwell).

  **The workgroup→domain map is vendor-specific and is MEASURED, not assumed.**
  `interp`'s `cu` is `blockIdx.x`, a *logical* index. NVIDIA fills a GPC with
  consecutive blocks (`n / sms_per_partition`); AMD's dispatcher assigns
  workgroups to XCDs **round-robin** (`n % partition_count`) — measured at
  **100.0%** against `HW_REG_XCC_ID` on MI355X over six geometries
  (`runtime/tests/xcd_map_gfx950_test.hip`), where the block formula scores 12.5%.
  Using the wrong one still emits correct tokens; it just destroys the locality it
  claims to create, invisibly. `L2Map` in `packet::devbuild` carries this as data.

  The locality is realized by the runtime half (`-DPLOW_L2_PLACE_DISPATCH`, or the
  `PLOW_L2_PLACE=1` arm of `scripts/build_gfx950.sh`), where each workgroup takes
  its window from the domain it is **physically** running on and all domains drain
  concurrently in one launch. Costs nothing: 248 VGPR / occ 2 / 0 spill, identical
  to the plain GQ decode object (`scripts/l2_regcheck.sh`).

  Guards: placement is **skipped** (byte-identical, with a note) when the block map
  would run off the end (`n_cu > partition_count·sms` — occupancy>1 or a
  grid≠sm_count mismatch; round-robin needs no such guard and is measured to hold
  at occupancy 2), and on any program with **more than one wave class on a target
  that relaunches per segment** — there `seg` already carries the class the host
  dispatches on, and overwriting it sends the whole prefill to the 4-wave flash
  object and returns zero logits. So on gfx950 the **decode** program is placed and
  prefill is not. A placement blob carries a header flag (`PLOW_BLOB_F_L2DOM`, plus
  SMs/partition + domain count in `reserved`) that a runtime **without**
  `PLOW_L2_PLACE_DISPATCH` refuses at load. See `plans/l2-placement-generic.md`.

  **Measured on Gemma-4-31B decode (MI355X, 3 interleaved folds, 64 steps): no
  effect** — 16.57 vs 16.54 ms/token, with fold deltas of +2.7%, −3.1%, −0.2%.
  Expected, and the arithmetic says so: decode streams **61.4 GB of weights per
  token** with zero reuse, against ~7 MB of activations, so even perfect L2 capture
  addresses ~0.01% of the traffic. The mechanism works (8 windows, 3.0% skew,
  token-identical); the *lever* is not on this workload. See the plan for where it
  is.

### Enabling fp8 (a compile/emit-time decision, off by default)

fp8 is **not** a runtime toggle — it is baked when the model is compiled and the
interp cubins are built. The default build is the accuracy-safe **bf16** path;
fp8 is opt-in because it is *lossy* (see the drift notes in the flag tables).
Full fp8 "beat-vLLM" profile:

```bash
# 1. weight twins (per-row e4m3 + f32 scales under fp8/ next to the checkpoint)
python perf-data/harness/quantize_fp8.py <hf-dir>          # + regenerate to include the head for E5
# 2. emit the fp8 blob (fp8 twins auto-detected; add the E5 head + argmax fusion)
# NVIDIA (sm_120):
PLOW_UNISEG=1 PLOW_FP8_HEAD=1 PLOW_FUSE_ARGMAX=1 \
    cargo run --release -p plowc -- --hf-dir <hf-dir> --emit devblob --max-ctx <ctx> --n-cu 188 --out model.pkt
# AMD (gfx950) — no PLOW_UNISEG, axis-named precision flags:
PLOW_W8A8=1 PLOW_FP8_HEAD=1 PLOW_FUSE_ARGMAX=1 \
    cargo run --release -p plowc -- --hf-dir <hf-dir> --emit devblob --arch gfx950 --gpu mi355x --max-ctx <ctx> --out model.pkt
# 3. build the interp cubins with the fp8 kernel arms
scripts/build_sm120_cubin.sh <out> -DPLOW_NV_W8A8=ON -DPLOW_FP8_KV=ON
```

**On AMD (gfx950) the profile above is `PLOW_W8A8=1` (older spelling:
`PLOW_FP8=1 PLOW_W8A8=1`), plus `PLOW_KV_FP8=1` for an fp8 KV cache — plain
`PLOW_FP8=1` is refused at emit, deliberately.**

The precision flags are named by **axis**: `PLOW_W8A16` / `PLOW_W8A8` /
`PLOW_W4A16` for weights+activations, `PLOW_KV_FP8` for the KV cache, and
`PLOW_MOE_ENC=` for the routed experts. `PLOW_FP8` and `PLOW_FP8_KV` remain as
aliases. Setting two weight flags is refused. The axes are independent and
compose: bf16 weights with an fp8 KV is a legal combination and has its own
object — it did not, once, because the KV axis silently expanded to the weight
axis too.

**The spelling differs by model family, because the kernels do.** An axis name
promises an encoding, so where a family cannot realize one it refuses rather
than resolving to the nearest thing it has:

| family | fp8 weights, bf16 acts | fp8 weights + fp8 acts | fp8 KV |
|---|---|---|---|
| dense (Gemma / Llama / Qwen) | `PLOW_W8A16=1` — **sm_120 only** | `PLOW_W8A8=1` — **the gfx950 recipe** | `PLOW_KV_FP8=1` |
| MLA+MoE (Kimi / GLM / DeepSeek) | `PLOW_W8A16=1` — works on gfx950 | **not implementable, refuses** | `PLOW_KV_FP8=1` |

The asymmetry is real, not an oversight. "fp8 weights, bf16 activations" is a
per-*channel* fp8 GEMM on the dense path, which gfx950 has no arm for, and a
*block*-fp8 GEMV on the MLA path, which it does. Same axis value, different
kernels, different coverage. And the MLA family's expert ops (45/46/48/49) are
w8a16 in **every** instantiation — the block-fp8 arms take fp8 weights and leave
the activation bf16 — so `PLOW_W8A8` there would hand back w8a16 under a flag
that named w8a8 precisely. It refuses instead, naming `moe_w8a8` and pointing at
`PLOW_W8A16` (fp8 weights) or `PLOW_MXFP4` (A4W4, the one path in that family
that narrows the activation too).

For a checkpoint that is already quantized — GLM-5.2-FP8, say — **no precision
flag is needed at all**: the encoding is read from `quantization_config`. Note
the two traps there, both live in that config: the key is **`dtype`, not
`torch_dtype`** (HF renamed it), and `dtype` reads `"bfloat16"` on an e4m3
checkpoint anyway, because it describes the *compute* dtype while the storage
dtype lives in `quantization_config`. The checkpoint wins over the flags — it is
a fact, they are a request — and a contradiction is refused rather than silently
granted or silently ignored.

There is deliberately **no separate activation flag**. w8a8 is one *profile*,
not a free cross-product with w8a16 — the kernels instantiate exactly those
two — so a `PLOW_A8` would name combinations no object implements, which is the
failure this naming exists to prevent. The activation encoding is *derived* and
reported in the manifest as `precision.act_enc`, read off **`QuantFp8`'s
presence** (the op that *is* the activation quant) rather than inferred from the
weight flag. That inference is precisely what once let a w8a16 packet reach a
w8a8-only object.

**Keep `--arch` and `--gpu` in agreement** unless you are deliberately
cross-compiling; a mismatch now warns. Two assets in this tree were emitted
`--arch sm_120a` with `--n-cu 256` (an MI350X count) and then run on gfx950 —
an arch/GPU disagreement is trusted toward the **GPU**, because `--arch` records
intent while `--gpu` records what the packet was actually sized for. `PLOW_FP8=1` alone emits
**w8a16**: it leaves the activation scale `t[3]` unbound because activations
stay bf16. But the gfx950 `GEMM_FP8` arm is **w8a8** — its epilogue computes
`acc * a_scale[m] * w_scale[n]` with no null check, and reads the A operand as
packed bytes. Handed a w8a16 packet it would fault on the null scale, and
misread A even if the scale were bound.

plowc therefore refuses, naming the fix, rather than upgrading w8a16 → w8a8 on
your behalf. Every other substitution the emitter makes is
computation-preserving (picking a 128×128 tile over 256×256 computes the same
matmul); this one is not, because w8a8 quantizes the *activations* too. Silently
switching would change a run's numerics under a flag you set to mean something
else, and any accuracy number would then be filed against the wrong profile. The
check keys off the emitted stream — any fp8 GEMM with an unbound `a_scale` —
not off the flags, so it cannot drift if the fp8 branching is restructured.
sm_120 has a real w8a16 cubin and is unaffected.

The fp8 weight twins are keyed **verbatim, including the `fp8/` prefix** — the
key is the packet tensor name exactly, with no stripping on either side. The
emitter declares `fp8/<name>`, `quantize_fp8.py` writes `fp8/<name>`, and the
loader looks up `fp8/<name>`. Stripping is a transformation, and a
transformation applied in one place but not another is exactly how this broke
once already (a freshly generated twin file could not be loaded at all).
`checkpoint.rs`'s coverage gate does map `fp8/<name>` → `<name>`, but only to
ask "is the bf16 weight covered?" — that is a coverage question, not the key
rule, and copying that line into a loader is what caused the mismatch.

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

### Interpreter compilation knobs (`nvcc -D…`)

Pass them through the build script, which forwards `PLOW_EXTRA_DEFINES` verbatim
to every object it compiles:

```bash
PLOW_EXTRA_DEFINES="-DPLOW_NV_W8A8=1 -DPGM_BN=64" scripts/build_sm120_cubin.sh <out.cubin>
PLOW_ROOT=$(pwd) …            # build a WORKTREE's sources; defaults to /root/plow
```

#### Start here: the four configurations anyone actually builds

Almost nobody needs to reason about the individual knobs. Pick a row, use the
flags in it, and skip to the tables only if you are doing kernel work. The
**build** column is `nvcc -D…` / CMake; the **emit** column is `plowc` env.

| you want | emit | build | notes |
|---|---|---|---|
| **Default** — validated, bf16 | `PLOW_UNISEG=1` | *(none)* | The shipped configuration. Everything below is a deviation from it. |
| **fp8 weights** — faster prefill GEMM | `PLOW_UNISEG=1 PLOW_W8A8=1` | `-DPLOW_NV_W8A8=1` | −48% GEMM, −30…34% prefill. Needs the fp8 weight twins. **Both flags or neither** — mismatch = `__trap()`. |
| **Long context, multi-user** — the fp8-KV path | `PLOW_UNISEG=1 PLOW_W8A8=1 PLOW_FP8_KV_FULL=1` | `-DPLOW_NV_W8A8=1 -DPLOW_FP8_KV=ON -DPLOW_FP8_KV_FASTPF=ON` | Halves KV bytes (B=8 at 127k fits 32 GB). `FASTPF=ON` is what keeps prefill on the fast PIPE=1 arm — **−21% prefill at 67k** vs leaving it off. ⚠️ **Lossy, and it degrades with context**: at 7.8k every arm (bf16 and fp8) retrieves a needle; at 66.9k **only bf16 does** — the fp8-KV loss sits somewhere between those two and is a property of the e4m3 cache, not of any flash arm (§9, `gemma4-12b-longctx-5090.md`). Greedy also diverges ~21 tokens. Validate retrieval at *your* context length, not a short one. |
| **Legacy all-layer fp8 KV** | `… PLOW_FP8_KV=1` (no `_FULL`) | `-DPLOW_FP8_KV=ON` (**FASTPF off**) | e4m3 on every layer. `FASTPF` must stay OFF — the hd256 fp8 prefill op traps under PIPE=1. Slower prefill; prefer the mixed row above. |

Measurement-only builds (garbage logits by construction — never serve them) are
in their own table at the end of this section.

**Defaults deliberately NOT flipped, and why.** These are measured wins that stay
off, so the reasoning is recorded rather than rediscovered:

| flag | measured | why still off |
|---|---|---|
| `PLOW_FP8_KV_FASTPF` | **−21%** prefill at 67k, byte-identical token stream | Legality depends on the *packet*, not the build: it is only valid for MIXED fp8-KV packets (`PLOW_FP8_KV_FULL=1`), and all-layer packets trap under PIPE=1. A build cannot know which packet it will load, so this cannot be defaulted safely — CMake now emits a `WARNING` when you are on the slow path instead. **If you emit mixed packets, turn it on.** |
| `PLOW_NV_PF_GEMV_HEAD` | −39% on prefill's `lm_head` | Traps on M≠1. Prefill emits `lm_head` at M=1 today, so it is *probably* safe to default, but that has not been validated across every model family — and the failure is a hard launch failure, not a slow path. Worth ~0.5% at 127k, more on short prompts. |
| `PLOW_NV_FA_FP8PV` | 1.40× on the hd512 flash op | Changes numerics, and is only **+1.5%** end-to-end once `FASTPF` is on. Gate status in its row below. |

Already defaulted **on** because they are bit-exact wins: `PGM_W8A8_LDS64` and
`PGM_SW8_V2` (px9, +2.2% weighted on the w8a8 GEMM).

**Why there are so many.** The interpreter is one persistent megakernel that
inlines every op arm, so its **register and shared-memory footprint is the WORST
CASE over everything compiled in**, and smem is the *union* over all ops in the
object. A knob therefore usually does one of two things: compile an arm *out* to
buy back registers/occupancy for everything else, or A/B a body against the
shipped one. Nearly all default to the shipped, validated configuration, and most
are byte-identical when unset.

Two traps worth knowing before you touch any of this:

- **An emit flag and its `-D` must agree.** Emitting `PLOW_W8A8=1` packets against
  a cubin built *without* `-DPLOW_NV_W8A8=1` hits `default: __trap()` and every
  launch dies with `CUDA_ERROR_LAUNCH_FAILED`. The failure looks like a driver
  problem and is not.
- **Decode and prefill are separate objects.** `-DPLOW_NV_PREFILL=1` builds the
  prefill object (tiled GEMM + flash-prefill, ~236 regs, 77.5 KiB smem) rather
  than stacking those arms onto the decode megakernel's budget. A flag marked
  *prefill only* below is a no-op in the decode object, and vice versa.

**Object selection / model-family arms.** These decide what is compiled in at
all. The family arms buy **cubin size, smem and stack frame** — not occupancy.
On sm_120a the decode object is 241 regs and gating all three out leaves it at
229, still 1 block/SM (occ 2 at 256 threads needs ≤128 regs/thread), and the
register ceiling is not monotone in what you delete: dropping DSA costs 9 regs
on decode, dropping MAMBA costs 2 on prefill. Numbers below are `ptxas -v` on
the megakernel symbol, CUDA 13.0, and are **per-arch — they do not transfer**.

| flag | default | effect |
|---|---|---|
| `PLOW_NV_PREFILL` | 0 | build the PREFILL object (`interp_sm120_pf`) instead of decode. |
| `PLOW_NV_GEMMA` | 0 | Gemma arms (hd512 full-attn, GF flash). Off ⇒ byte-identical to a non-Gemma build. |
| `PLOW_NV_HOPPER` | off | sm_90a wgmma GEMM + Hopper attention arms instead of the sm_120 ones. |
| `PLOW_NV_MLA` | **1** | MLA (DeepSeek/Kimi/GLM). `0` trims the decode cubin — **sm_90a** 2,457,864 → 1,407,496 B (−43%), regs 208 → 188; **sm_120a** 2,804,312 → 1,873,376 B (−33%), regs 241 → 224. Occupancy-neutral on both (1 blk/SM either way). Not compiled into prefill: on sm_120a the prefill cubin is bit-identical with it on or off. |
| `PLOW_NV_MAMBA` | **1** | Mamba/Nemotron arms. Owns the *prefill* stack frame: **sm_90a** 1744 → 672 B, **sm_120a** 1024 → 0 B. Costs 2 regs on the sm_120a prefill object to turn off (238 → 240). |
| `PLOW_NV_DSA` | **1** | DeepSeek sparse-attention arms. Owns the *decode* smem (2192 → 1168 B on both arches). Costs 9 regs on the sm_120a decode object to turn off (241 → 250). Not compiled into prefill. |
| `PLOW_NV_GF8_TWIN` | 0 | co-linkable GF=8 full-attn decode twin (234 vs 209 regs, 16448 vs 12352 B arena); host picks per model. |
| `PLOW_NV_SEG_GEMM` | 0 | lean GEMM-segment object targeting occupancy 2, separate from the register/smem-hungry flash object. |
| `PLOW_NV_SEG_GEMM_BN64` | off | that object at BN=64. **PX-7: ~1.05× end-to-end**, not the ~2× the occupancy argument implies — `d_gemm_glu_w8a8` (gate\|up, ~2/3 of prefill GEMM FLOPs) is *register*-limited to occ=1 and does not move. |

**Precision.**

| flag | default | effect |
|---|---|---|
| `PLOW_NV_W8A8` | 0 | PX-2 native w8a8 fp8 mainloop (BK64 + `Swizzle<3,4,3>` + `mma.sync.m16n8k32`). −48% GEMM vs bf16, −30…34% end-to-end prefill; on 31B it flips the vs-vLLM-bf16 prefill deficit to a **win** at 32k. Needs the fp8 weight twins **and** the matching emit flag. |
| `PLOW_FP8_KV` | 0 | e4m3 KV cache, half the KV bytes; per-(token, kv_head) f32 dequant scale. **Lossy** — ~3–6% logit relL2, greedy diverges after ~21 tokens, which is why it is opt-in. Lifts the 31B multi-user batch cap (bf16-KV B=4 → fp8-KV B=7–8), beats vLLM fp8kv at ≥16k ctx, ~−1.2% decode TPOT at 32k. **Read `PLOW_FP8_KV_FASTPF` before using this** — on its own it silently costs you the PIPE=1 prefill pipeline. |
| `PLOW_FP8_KV_FASTPF` | **OFF** | ⚠️ **Without this, enabling fp8 KV makes prefill slower than not using fp8 KV at all** — measured 1670.3 ms vs bf16's 1315.6 ms at 7.8k, because the PIPE=0 arm's inline dequant costs more than the halved KV bytes save. **Turning on `PLOW_FP8_KV` alone forces the prefill objects to `PLOW_NV_FA_PIPE=0`** — `FLASH_PREFILL_FP8` dequants at the smem stage (cp.async cannot convert fp8 inline), so it exists only on the synchronous-staging build and *traps* under PIPE=1. You therefore lose the cp.async flash pipeline (the −81%-at-128k feature) for prefill, silently, just by enabling fp8 KV. `FASTPF=ON` keeps prefill on PIPE=1 and routes `FLASH_PREFILL_FP8` to the px4 fp8-mma arm. **px8 measured the flip: 13843.3 → 10947.3 ms, −21% total prefill at 67k** — one flag, on an arm already in the tree. hd512-only, so pair it with mixed-KV packets (`PLOW_FP8_KV_FULL=1`). Decode is PIPE-agnostic and unaffected; bf16-KV runs never take this path. |
| `PLOW_NV_FA_FP8MMA` | derived | feed the RAW e4m3 K tile to the mma — no dequant pass, no dequant barrier; Q quantised once per q-tile. Requires `PLOW_FP8_KV`. |
| `PLOW_NV_FP8_RB` | **1** | fp8 GEMV row-blocking. |
| `PLOW_FP8_FAST` | off | faster/looser fp8 conversion path. |

**Attention.** `FA_PX4` and `FA_PIPE` are the two that carry the shipped
long-context performance; the rest are mostly A/B controls, several of which
measured *negative* and are kept only so nobody re-runs them.

| flag | default | effect |
|---|---|---|
| `PLOW_NV_FA_PX4` | **1** | restructured hd512 full-layer flash (register softmax + 8-warp QK splitting the HD-512 contraction across two warp groups). −24% flash op, −16% end-to-end 128k prefill. |
| `PLOW_NV_FA_PIPE` | **1** | cp.async KV-stream pipeline. Bit-identical logits; −16%@4k → **−81%@128k** prefill (the win grows with context). **Forced to 0 for the prefill objects when `PLOW_FP8_KV` is on unless `PLOW_FP8_KV_FASTPF=ON`** — see that row; this is the single easiest way to lose most of plow's long-context prefill performance without noticing. |
| `PLOW_NV_FA_FP8PV` | 0 | **px8/px12 — the largest single prefill lever in the campaign.** e4m3 P·V via 8-bit `ldmatrix.trans`, absorbing the fragment permutation into the smem staging order. **1.18× on a 127k prefill end-to-end** (32.39 → 27.59 s), 1.40× on the flash op alone. **Unreachable without `PLOW_FP8_KV_FASTPF=ON`** — `op_attention.cuh` `#error`s without `PIPE=1`, and `scripts/build_sm120_cubin.sh` hardcodes `PIPE=0`, so no stock build could select it. Also needs `-DPLOW_FP8_KV=1`. sm_120a-only opcode. ⚠️ **NOT parity-preserving: greedy diverges at completion token 28.** Both continuations are fluent and the first 28 tokens are bit-identical, but that is one sample, not a gate — and this model's fp8 KV already loses a 67k needle (§9a). **Run a retrieval test before shipping it.** Default off for that reason, not for lack of speed. |
| `PLOW_NV_FA_GF` | 4 | head-group fold factor for flash-decode. Correctness needs `gqa % GF == 0`, checked at dispatch (gqa is a runtime field, so it cannot be a `static_assert`). Register allocation is the worst case over instantiations, so do not add more. |
| `PLOW_NV_FA_GF_FULL` | 4 (via `build_sm120_cubin.sh`; `CMakeLists.txt` says 2) | **CONTESTED — do not change without an end-to-end measurement.** PX-11 measured `=8` at **1.52× on the flash-decode OP** (bit-exact at fixed nsplit, 0 net registers) and recommended it. PX-15 then measured whole **decode steps** on the 12B and found `=4` wins **all 8 cells at ctx ≥ 8k**, both batches (**−29% at 130k, B=1**); `=8` never wins. PX-15 eliminated packet and occupancy differences by measurement (packets sha-identical across GF; smem/registers/occupancy identical), leaving **grid fill**: `n_grp = 16/GF`, so `=8` leaves 2 work-item groups for 170 SMs. **Trust the end-to-end result** — substituting an op-level ratio for a step-level one is exactly what `tuning/README-decode-tuner.md` §2 forbids, and it has produced three wrong rankings in this campaign. Correctness needs `gqa % GF == 0`, checked at dispatch. |
| `PLOW_FP8_LD16`, `PLOW_FP8_FAST` | unset | **UNVALIDATED END-TO-END — treat the headline number as op-level only.** PX-11 measured **1.61× on the flash-decode op** together with `GF_FULL=8`, bit-exact (65536/65536 identical) and register-neutral (241→241, 0 spills). Two later results undercut that framing: `GF_FULL=8` is itself **CONTRADICTED** end-to-end (see its row — `=4` wins all 8 cells at ctx ≥ 8k), and PX-15 could not measure these flags at all — **every fp8-KV block asset dies in prefill with `CUDA_ERROR_LAUNCH_FAILED`**, all 8 arm×batch combinations, and the documented fix (`-DPLOW_NV_FA_PIPE=0`, "cp.async cannot convert fp8 inline") does **not** help. That is a live bug with a reproducer, not a configuration error: the packet is a genuine fp8-KV packet (KV halved 0.25 → 0.13 GiB) and the object is built `-DPLOW_FP8_KV=1`. **Nothing in the tree sets either flag**, and nothing should until the crash is chased down and a step-level measurement exists. |
| `PLOW_NV_FA_TMA` | 0 | TMA (`cp.async.bulk`) KV staging — ~2× *slower* on sm_120 (single-CTA 1-D bulk). A/B control. |
| `PLOW_NV_FA_KUN` | **1** | K-stream pre-issue depth for flash-decode; 1 = original consume-immediately loop. |
| `PLOW_NV_FA_WPR` | 0 | warp-per-row score phase (vs one-row-at-a-time). |
| `PLOW_NV_FA_WPR_RB` | 1 | rows a warp carries concurrently in that phase — at nsplit=32 one-at-a-time leaves ~1 load in flight. |
| `PLOW_NV_FA_QGLOB` | 0 | read Q from global instead of staging to smem (WPR path); staging costs a `__syncthreads` on every work item. ⚠️ **`FA_WPR=1` + `FA_QGLOB=1` silently CORRUPTS the fp8/SZ arms** (PX-11): the Q-staging guard is a *preprocessor* condition that strips staging from every instantiation, but the global-read replacement sits inside an `if constexpr` that excludes fp8/SZ, so those arms read a never-written `qsm` — measured divergence 5.0e-01 on a 2.9e-03 tensor. Non-default combination; nothing shipped is affected; **not fixed**. |
| `PLOW_NV_FA_REDBOUND` | 0 | bound the softmax reductions to the tile's LIVE rows (entries past `rmax_t` are NEG_INF anyway). |
| `PLOW_NV_FA_VDBUF` | 0 | V double-buffer. **MEASURED NEGATIVE**: a wash at 32k/64k, **+2.2% slower at 128k** — the context it was built for. The 128k full-attn flash is HBM-bound. |
| `PLOW_NV_FA_CORRSKIP` | 0 | fp8mma only — skip the softmax rescale when every lane's `corr` is exactly 1.0. Bitwise identical. |
| `PLOW_NV_KVBOUNDS` | 0 | per-batch KV bounds checking. |

**GEMM / GEMV.**

| flag | default | effect |
|---|---|---|
| `PGM_BN` | 128 | GEMM N-tile. `64` shrinks the plain arena to 45 KiB so the occ-2 segment object fits. |
| `PGM_STAGES` | 3 | GEMM cp.async pipeline depth. **px9 measured 3→6 = 0%** — the mainloop is not latency-bound. |
| `PGM_GLU_STAGES` | 2 | same for the fused GLU arm (kept shallower to fit the 100 KiB dynamic-smem cap). |
| `PGM_W8A8_LDS64` | **1** | **px9** — read the fp8 fragment as one `uint2`. The 4-byte read touched only even words (≤16 of 32 banks, a conflict no XOR can remove). **+6.5% on plain w8a8, 0% on GLU, +2.2% weighted.** Bit-exact. |
| `PGM_SW8_V2` | **1** | **px9** — `Swizzle<2,4,2>` matched to the ACTUAL 64-byte fp8 row; the shipped `Swizzle<3,4,3>` assumed a 128-byte row and never saw row bit 0, so adjacent rows got the same permutation. +0.5% on top of LDS64 (−2.5% alone). |
| `PGM_SW8_OFF` | unset | A/B control: make `pgm_sw8` the identity, i.e. no fp8 smem swizzle at all. **px9 measured the swizzle earns its keep** — removing it (with `uint2` reads) is +3.1% cycles/QMMA, so this is a diagnostic, not a tuning win. |
| `PLOW_NV_GEMV_RB` | 0 | MoE GEMV row-blocking — gives a warp `GV_MOE_RB` rows, multiplying loads-in-flight at constant occupancy. Off keeps every sm_120 object byte-identical; **the sm_90a build sets it to 1**. |
| `PLOW_NV_PF_GEMV_HEAD` | 0 | run prefill's `lm_head` on the M=1 GEMV arm instead of a BM=128 tile (0.78% row efficiency). **1.991 → 1.213 ms, −39%**, 98% of the 1695.6 GB/s ceiling. Prefill only; traps on M≠1. ~0.5% at 127k — a short-prompt lever. |
| `GV_MM_MAX` | **8** | Widest `gemv_*_rows<MM>` rung instantiated for batched decode: one weight row is loaded once and dotted against `MM` activation rows, so batch `B` costs `ceil(B/GV_MM_MAX)` weight passes, not `B`. **Match it to the batch you actually serve — mismatched, it is expensive in both directions.** Wider rungs spill, and because this is one megakernel the spill tax lands on *every* arm including B=1 (8 → 212 regs / 0 spill; 16 → 255 regs / 72 B spill; 32 → 1162 B spill). Measured (12B bf16, aggregate tok/s): `=8` gives 355 at B=8 but only 387 at B=16; `=16` gives 294 at B=8 and 520 at B=16. So 16 costs 1.1% at B=1 and **17% at B=8** to buy 34% at B=16. **PX-10 measured the same mismatch end-to-end on this card: an asset built `=16` and served at B=8 loses 19.4% at 131k and 33.8% at 1k.** Pin `=16` only if you pin B≥16. |
| `PLOW_NV_GEMV_LS` | 0 | GEMV row-blocking. Wins in isolation (qkv 1.43×, lm_head 1.42×) but **loses in the megakernel**, and not via registers — a 177-reg variant measured slower than a 229-reg one. Compiled out, intact. |
| `PLOW_NV_GEMV_NOSTAGE` | 0 | skip GEMV smem staging. |
| `PLOW_NV_GEMV_STAGE_MINROWS` | 16 | row threshold below which staging is skipped. |
| `PLOW_NV_RB_QKV`, `PLOW_NV_RB_LMHEAD`, `PLOW_NV_RB_GEMV` | 0 | per-op row-blocking A/B controls. |
| `PLOW_NV_SZ`, `PLOW_NV_ZG` | 0 | **experimental** lossless bf16 weight decompression. Bit-exact, measured non-viable (recon competes with an already saturated weight stream); kept as A/B references. |

**MoE.**

| flag | default | effect |
|---|---|---|
| `PLOW_MOE_XN_BF16` | 0 | bf16 expert-N staging buffer. |
| `PLOW_MOE_XN_MAX` | 2816 | expert-N staging cap. |
| `PLOW_MOE_DOWN_SG` | 4 | subgroup count for the expert `down` arm. |
| `PLOW_MOE_DOWN_LANESPLIT`, `PLOW_MOE_DOWN_STAGE_FU` | 0 | `down` lane-split / staged fixups. |
| `PLOW_MOE_ROUTER_WIDE` | 0 | wide router arm. |
| `PLOW_MOE_COMBINE_ALLBLK` | 0 | all-block combine. |

**Scheduling, sync, occupancy.**

| flag | default | effect |
|---|---|---|
| `PLOW_NV_SCHED` | **1** | global-queue scheduler; `0` = static per-block streams (build-time A/B). The counter protocol is byte-identical across both — only WHICH block runs WHICH stream entry changes, so an A/B isolates scheduling. |
| `PLOW_NV_SEGMENTS` | 0 | host relaunches once per segment (the AMD model) instead of one cooperative launch, so each segment gets its own occupancy/register profile. |
| `PLOW_NV_PTXSYNC` | **1** | inline-PTX counter gate (`red` instead of a result-bus round trip). The load-bearing acquire is *moved*, not removed. |
| `PLOW_NV_GATE_SLEEP` | 64 | backoff (ns) inside the counter-gate poll; `0` spins flat out. |
| `PLOW_NV_LEAN_DECODE` | 0 | drop arms owning the decode object's 208-reg / 1-blk-SM ceiling so ptxas + `PLOW_NV_FORCE_MINBLK` can reach 2–3 blk/SM. |
| `PLOW_NV_FORCE_MINBLK` | off | force a `__launch_bounds__` min-blocks-per-SM. |
| `PLOW_NV_THREADS`, `PLOW_THREADS` | 256 | block size. Raising it is the precondition for BQ=64 flash tiling (px8 found BQ=64 is *register*-infeasible at 256). |
| `PLOW_NV_EMBED_SMEM` | 0 | embed the object's smem requirement so `serve` reads it instead of guessing the GF=2 default (a GF_FULL=4 flash-decode then indexed past the arena). |
| `PLOW_L2_PLACE_DISPATCH` | off | L2 placement dispatch (alias: `PLOW_NV_PLACE_DISPATCH`). Vendor-neutral — GPC on NVIDIA, XCD on AMD. |

**Measurement-only — never ship a build with these.** All produce wrong logits
by construction.

| flag | default | effect |
|---|---|---|
| `PLOW_NV_SKELETON` | 0 | run gates + signals with no op bodies: the interpreter's dispatch floor. Garbage logits. |
| `PLOW_NV_SKEL_PAD` | 160 | padding for that skeleton. |
| `PLOW_NV_ABLATE_LO`, `PLOW_NV_ABLATE_HI` | 0 | 128-bit opcode mask — skip those ops' BODIES, keep every gate/signal, so the TPOT delta is that op set's true wall-clock contribution. Garbage logits. |
| `PLOW_NV_FA_FP8ABL` | 0 | flash fp8 ablation bitmask. **Never set on a shipped build.** |
| `PLOW_NV_TRACE` | 0 | per-op `gate`/`body`/`signal` cycle trace. **Read the SHAPE, not the absolute total** — the recording thread is the same thread 0 that signals, so the sum over-reports. |
| `PLOW_NV_TRACE_DECODE`, `PLOW_NV_TRACE_PF`, `PLOW_NV_SKELETON_DECODE` | off | the same, scoped per object. |

Harness-only (not the served cubins): `PLOW_SM120_SMS` (188) and
`PLOW_SMP_THREADS` (256).

**Serving / runtime knobs** (`plowrt` env):

| var | default | effect |
|---|---|---|
| `PLOW_PF_BATCH=1` | off | PX-1 cross-request batched prefill — packs waiting requests' prefill chunks into one launch (shared weight reads); block-diagonal varlen flash keeps per-request attention isolated. +27% saturated multi-user throughput. Off = serialized prefill (byte-identical default). This is prefill⊕prefill: it does **not** put decode rows in the prefill launch (see the note below the table). **Silently inert on fp8-KV packets** — `FlashPrefillFp8` uses `t6`/`t7` for the k/v dequant scales, so there is no handle left for the request table and the kernel has no fp8 mux arm; the engine logs `PLOW_PF_BATCH=1 ignored: fp8-KV packets have no batched prefill arm` and serializes. Also ignored under `PLOW_VMM_PREFIX=1`. |
| `PLOW_PF_INTERLEAVE=N` | 2048 | **Chunked prefill quantum — the default path, not a `PLOW_PF_BATCH` knob.** Once any slot is decoding, a tick admits at most `N` prefill rows, then runs the decode launch; a mid-decode arrival therefore stalls live streams by one chunk instead of one whole prompt. With NO decoder live the chain runs to completion (fastest cold TTFT — the pre-interleave behavior). `0` = uncapped. **It can only clamp BELOW the emitted ladder** — the largest prefill bucket is fixed at emit time by `PLOW_MAX_CHUNK` (below), so raising this above that is a no-op. Measured on a `window=1024` asset (emitted max chunk 1024): `2048` vs `8192` are identical to within noise (376.00 s vs 376.09 s duration, P99 ITL 3117.8 vs 3118.6 ms). Check the emitted ladder before tuning this. |
| `PLOW_PF_CHUNK=C` | 0 (off) | **Experimental** per-request prefill chunk-row cap: clamps each request's per-launch rows so more requests co-pack (R ≈ quantum/C). Measured ~10% regression at B=8 (finer chunks add launches + attention KV re-reads); off = uncapped (byte-identical). |
| `PLOW_PF_CHUNK_COST=R` | 512 | cost of ONE prefill launch, in padded-row equivalents — the currency the bucket pick minimizes (`rows + R × launches`). A launch re-streams every layer's weights, so it is not free: measured `ttft_ms = 0.112·rows + 60.1·chunks` on sm_120 / 12B (±2.8%), i.e. **60 ms ≈ 537 rows**. Charging it stops a just-under-rung tail cascading down the ladder (8190 rows ran `[4096,2048,1024,512,128×4]` = 8 launches, +25% TTFT, vs the 2-row-padded `[8192]`). `0` = the old pure-minimum-padding objective. |
| `PLOW_PF_COVER=1` | off | restore the covering-bucket prefill policy (exact-parity A/B vs the cost-aware default). |
| `PLOW_PF_DEFER_DECODE=1` | off | **Throughput mode — trades streaming latency for aggregate tok/s.** A tick with ANY slot still mid-prefill runs its prefill chain to completion and skips the decode launch entirely, so every later decode tick runs at the full batch. A decode launch costs `a + b·B`, and `a` (the ~12 GiB weight re-read plus launch turnaround) is paid whatever `B` is; interleaving spends `a` on ~992 ticks at an average `B` well below the engine batch, deferring spends it on the minimum number of full-batch ticks. Measured on the 8×127k cell: **29.91 → 31.05 out tok/s (+7.1%, −18.73 s)** — PX-17, and it beats that campaign's *projection* for full prefill⊕decode fusion, which is why fusion was not built. ⚠️ **No token leaves the server until every prompt is resident**, so TTFT and streaming behaviour degrade badly under mixed load. Correct for batch/offline throughput; wrong as an interactive serving default — which is why it is off. |
| `PLOW_PF_PACKLOG=1` | off | per-launch pack diagnostics (requests/rows/bucket + prefill/decode wall split). |
| `PLOW_VMM_PREFIX=1` | off | VMM-backed KV prefix cache — a new request attaches the blocks a previous one already built instead of re-prefilling them. Only the FULL-attention layers are VMM-backed (sliding layers stay on `cudaMalloc`), and reuse is block-granular, so `cached_tokens` floors to a block multiple. Measured on 12B: warm TTFT 3.6× (4k) → **23.8× (128k)**, and warm TTFT is near-flat in context; cold costs ~0.2–1.5% above 8k. Incompatible with `PLOW_PF_BATCH=1` (which is then ignored, with a warning). |
| `PLOW_VMM_BLOCK_MIB=M` | 2 | VMM sharing block size. The driver granularity (2 MiB ≈ 4096 tokens at hd256 bf16) is the finest match unit, which is what makes shared system prompts and multi-turn histories actually hit. Raise (e.g. 64) for 128k-dedup work. |
| `PLOW_VMM_CACHE_MIB=M` | 0 | cap on retained (unreferenced) VMM blocks; `0` = no cache. |
| `PLOW_NV_SCHED=1` | **on** | global-queue interpreter scheduler; the static per-block-stream path is the build-time A/B control. |

**Prefill/decode batching — what plow does and does not fuse.** Three things are
easily conflated:

1. **Chunked prefill** — shipped, on by default, no flag (`PLOW_PF_INTERLEAVE`).
   A long prompt is admitted a chunk at a time so live decode streams are not
   stalled for a whole prompt.
2. **Cross-request prefill packing** — `PLOW_PF_BATCH=1`, off by default. Several
   *waiting requests'* prefill chunks share one launch, so the GEMMs see one
   `M = Σ len` and the weights are read once across requests.
3. **Mixed batching (prefill ⊕ decode in one launch)** — **not implemented**, and
   not hidden behind a flag. A tick that does both runs two launches: the prefill
   chunk, then a separate decode launch, each re-reading the full weight set
   (~12 GiB fp8 on the 12B asset, ~9 ms at 1.3 TB/s). vLLM's chunked prefill
   carries the decode rows in the same forward pass and pays that read once.

The gap in (3) is bounded by one weight read per tick, so it scales with how
small the chunk is: ~12% of a tick at 2k prompts, but only **~0.6% at 127k**,
where the prefill chunk dominates the tick. It is a short-context / high-QPS
lever, not a long-context one.

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

Measured results live in `perf-data/` (one JSON + MD per campaign). Multi-user
sweeps use HuggingFace `inference-benchmarker` via `perf-data/bench_ib.sh`
against both plow and vLLM with identical profiles.
