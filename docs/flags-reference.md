# plow — full flag & knob reference

Exhaustive reference for every emit-time (`plowc` env), build-time (`nvcc`/CMake
`-D…`), and runtime (`plowrt` env) flag. The README keeps only the load-bearing
traps and the four configurations anyone actually builds; everything here is for
kernel/perf work. Measured results live in `perf-data/` (one JSON + MD per
campaign).

Perf features from the rtx-11/12/13 campaigns are gated so the **default build is
a fixed, validated configuration**; every flag is an A/B control with a
correctness gate (token-identity or a bit-exact vs-f32 oracle). **Unset = shipped
default**, and most flags are byte-identical when unset.

---

## Unified CLI args (new)

Both `plowc` and `plowrt` now support structured **clap CLI args with
environment-variable fallback**. Every knob formerly set via `PLOW_*` env can now
be passed as a `--long-flag`, and the old env var still works transparently
(clap's `env` attribute reads it). **CLI takes precedence over env** — with the
exception below.

> **Env wins for eight knobs.** `PLOW_MULTISTEP`, `PLOW_VMM_PREFIX`,
> `PLOW_VMM_BLOCK_MIB`, `PLOW_DRAIN_TIMEOUT_MS`, `PLOW_SLAB_KEEP`,
> `PLOW_KV_POOL_MIB`, `PLOW_DEV_SAMPLE` and `PLOW_NV_CUBIN_SAMPLE` are read
> env-first, because tests and benches flip them mid-process — after the
> config snapshot is cached, which a CLI-first read could not observe. For
> these eight a systemd envfile beats the command line.

```bash
# These are equivalent:
PLOW_PF_INTERLEAVE=4096 plowrt serve …
plowrt serve --pf-interleave 4096 …

# CLI overrides env:
PLOW_PF_INTERLEAVE=2048 plowrt serve --pf-interleave 4096 …   # uses 4096
```

### Config structs

| binary | struct | file | scope |
|--------|--------|------|-------|
| `plowc` | `EmitConfig` | `crates/devgen/src/emit_config.rs` | Emit-time knobs (60+ fields) |
| `plowrt` | `RuntimeConfig` | `crates/plowrt/src/config.rs` | Shared runtime knobs |
| `plowrt` | `NvidiaRuntimeConfig` | (nested in RuntimeConfig) | NVIDIA/sm_120 serving |
| `plowrt` | `AmdRuntimeConfig` | (nested in RuntimeConfig) | AMD/gfx950 serving |

### Hot-path access pattern

Both structs use a **process-global `OnceLock`** initialized once after CLI
parse, so deep call sites pay a single atomic load (identical cost to the old
`env_flag!` macro):

```rust
// devgen (emit time)
emit_config::active().k3_full

// plowrt (runtime)
RuntimeConfig::get().nv.pf_interleave_rows()
RuntimeConfig::get().pf_packlog
```

Use `get()`, never `global()`: `global()` panics when the config was never
installed, which is the normal state for every library embedder — GPU tests,
examples and benches build engines directly and never run `main()`'s CLI
parse. `get()` falls back to an env-only parse there.

### Discovering all flags

```bash
plowc --help          # shows all emit-time flags with env fallback
plowrt serve --help   # shows all runtime flags with env fallback
```

### Legacy compatibility

All `PLOW_*` environment variables continue to work. The macros `env_flag!` and
`env_usize!` remain in `plowrt/src/lib.rs` for any code that has not yet been
migrated — they read once and cache in a `OnceLock`, identical semantics.

**Why there are so many.** The interpreter is one persistent megakernel that
inlines every op arm, so its **register and shared-memory footprint is the WORST
CASE over everything compiled in**, and smem is the *union* over all ops in the
object. A knob therefore usually does one of two things: compile an arm *out* to
buy back registers/occupancy for everything else, or A/B a body against the
shipped one.

---

## `plowc` emit-time knobs

- ⚠️ **`PLOW_UNISEG=1` is NVIDIA-only. Do not pass it when targeting gfx950** —
  every AMD recipe that sets it produces a *broken* AMD asset, and the breakage is
  silent. It collapses every op into one segment, which is right on sm_120 (that
  interpreter runs one cooperative launch and never reads a wave class) and
  destroys AMD's wave-class split. With one segment, segment 0 contains the flash
  packets, the class-4 test matches, and the **entire prefill program is dispatched
  on `interp_flash`** — whose body is `if (op == FLASH_PREFILL…)` with no switch, so
  every GEMM, norm and lm_head is silently dropped. Prefill "completes" in 8.7 ms
  instead of 72.1 and the logits are all zero. A correct Gemma-4 31B emit has
  **121 segments per prefill bucket** (`2·layers + 1`); check `build.json` if in
  doubt.
- `PLOW_UNISEG=1` — single-segment programs (required for the prefill buckets on
  the sm_120 interpreter).
- `PLOW_DECODE_BATCH=B` — emit a batched decode program for multi-user serving
  (WS-GEMV shares weight reads across streams). Kimi-K3 supports `B` in 1..32;
  `B>16` requires `PLOW_GEMV_WALK=1`. Other model families may impose a lower
  ceiling. `B=1` blobs are byte-identical to unset.
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
  fine" mode — see the design notes.
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
  `PLOW_L2_PLACE_DISPATCH` refuses at load.

  **Measured on Gemma-4-31B decode (MI355X, 3 interleaved folds, 64 steps): no
  effect** — 16.57 vs 16.54 ms/token, with fold deltas of +2.7%, −3.1%, −0.2%.
  Expected, and the arithmetic says so: decode streams **61.4 GB of weights per
  token** with zero reuse, against ~7 MB of activations, so even perfect L2 capture
  addresses ~0.01% of the traffic. The mechanism works (8 windows, 3.0% skew,
  token-identical); the *lever* is not on this workload. See the plan for where it
  is.

More emit knobs (all **byte-identical when unset** unless noted):

- `PLOW_GEMV_SPLIT=S` — emit `S·n_cu` decode slices for Gemv/GemvGlu/GemvQkv
  packets (finer CU work-stealing). Ships at `1`; `S>1` measured a loss,
  bit-identical.
- `PLOW_GEMV_MM=M` / `PLOW_GEMV_WALK=1` — **AMD** compile-time decode row-batch
  bucket and its wide-arm walk loop. `PLOW_GEMV_MM` is derived from
  `next_pow2(PLOW_DECODE_BATCH)` clamped to 16 and baked into the ELF symbol
  `plow_gemv_mm_cap_<MM>` (the plowrt loader validates it). `PLOW_GEMV_WALK`
  toggles the outer `ceil(M/MM)` loop so one MM object serves any M; **serving
  batch > MM without WALK is refused.** Kimi-K3 requires WALK for decode batches
  17..32 because the largest compiled AMD bucket is 16. NVIDIA analogue is the
  build-time `GV_MM_MAX`.
- `PLOW_PF_GEMV_HEAD` — force prefill `lm_head` onto the M=1 GEMV arm vs a tiled
  BM=128 tile (on by default for AMD; `=1`/`=0` to force). Traps on M≠1.
- `PLOW_NO_FUSE_QKV=1` — revert the fused QKV projection to the historical
  split-3 path (packet-reduction A/B).
- `PLOW_DECODE_TILED=1` — **AMD only**; emit prefill (tiled) opcodes into the
  decode bucket.
- `PLOW_SEG_PER_OP=1` — one SEGMENT per op (host-side AQL chaining instead of
  batched). `PLOW_SEG_CLASS_SLICE=1` re-slices GEMM segments so both occ-2
  blocks/SM get work.
- `PLOW_TUNEDB=<dir>` — tune-DB root the compiler resolves shapes against
  (default `tuning/`). `PLOW_TUNE_DUMP=1` prints a `TUNEDUMP…` census line per
  resolved GEMV shape; `PLOW_GEMM_JSONL=<path>` appends raw per-tile GEMM
  samples (tuning-harness diagnostics).
- **Known-wrong escape hatches** (produce garbage tokens by design — never
  serve): `PLOW_CHAIN_BYPASS=<op[,op…]>` splices opcodes out of the chain,
  `PLOW_SKIP_COVERAGE=1` emits a model that fails coverage checks,
  `PLOW_K3_ABLATE=<op…>` / `PLOW_K3_UP_NOGATHER` / `PLOW_K3_UP_GATHER_ONLY` /
  `PLOW_K3_SEQ_ROWS` are K3 bisection instruments.

### Model-family emit arms (K3 / GLM / Gemma-MoE / KDA)

Advanced per-model fusion/sharding knobs, read by `plowc`/`devgen` at emit
(they change the emitted packet). Most default **on** and their `=0` restores an
un-fused, byte-identical control; the experimental ones are **off** because they
are not parity-preserving.

| flag | default | effect |
|---|---|---|
| `PLOW_K3` | off | **BUILD `-D`** + emit gate: compiles the seven Kimi-K3/KDA interpreter arms (ops 99–106) and marks a K3 blob's `req`, so a K3 packet is refused against an object built without it. |
| `PLOW_K3_FUSE_ARNORM` | **on** | fuse each `AttnRes` mix with the following RMSNorm (bit-exact, 186 sites). |
| `PLOW_K3_FUSE_BRESID` | **on** | fold the LatentMoE-tail residual into the block-output residual (bit-exact). |
| `PLOW_K3_FUSE_SHGLU` | **on** | collapse shared-expert `gate/up/SituGlu` into one `GemvGlu` (decode-only, LDS-bounded). |
| `PLOW_K3_FUSE_A` | off | fuse the MLA `q/kv/k_rope/gate` A-projection into one `GemvQkvg` (decode-only). |
| `PLOW_K3_SHARD_UP` | **on** (tp>1) | column-parallel `routed_expert_up_proj`; `=0` = replicated A/B. |
| `PLOW_K3_WGFIT` | **on** | narrow workgroup/CU counts to what a packet uses (bit-identical). |
| `PLOW_K3_NS` | measured | pin FlashMLA-decode `nsplit` for K3 MLA layers. |
| `PLOW_K3_FUSE_NGEMV` | **on** | fold the two B1 `norm→GEMV` sites; `0` restores the unfused control, `lat`/`q` isolate one site. Full TP8 BF16-logit gate is byte-exact. |
| `PLOW_K3_KDA_CONV_STEP_DB` | off | B1-only experiment: emit `KdaConvStateStepG` and allocate ping-pong KDA convolution-window banks. The selected decode object must also be built with `PLOW_KDA_CONV_STEP_DB=1`; packet/object mismatch is refused. |
| `PLOW_K3_SHARD_HEAD` | **off** | vocab-column-parallel `lm_head`; measured on real K3 gfx950 TP8 with token-identical output, 33.160 to 32.781 ms/token (-1.14%). |
| `PLOW_GLM_FUSE_A` / `PLOW_GLM_FUSE_G` | **on** | fuse the GLM A-projection / gate GEMVs (byte-identical); `=0` splits. |
| `PLOW_GLM_DSA` | **on** (ctx>65536) | arm the DSA sparse-indexer gather above the measured ctx crossover; `=0` forces dense. |
| `PLOW_GLM_GF` | ctx-adaptive (2/4) | pin the MLA head-fusion factor; `=8` needs a `-DPLOW_GLM_GF8_ARM=1` object. |
| `PLOW_GLM_NS` / `PLOW_GLM_WGFIT` | computed / **on** | pin MLA flash-decode `nsplit` / narrow sizing rules (bit-identical). |
| `PLOW_GLM_FUSE_B1` | **off** ⚠️ | fuse `Residual+RmsNorm` into `AddNorm`; fp reorder can flip an early argmax — ship only behind an HF-coherence gate. |
| `PLOW_GLM_GF8_ARM` | **0** (BUILD `-D`) | compile the GF=8 flash-decode arm; merely compiling it in is a measured **+32% decode regression**. |
| `PLOW_GEMMA_MOE_ROUTER_FUSED=1` | off | disable the split router, serializing the score GEMV on one CTA (escape hatch). |
| `PLOW_GEMMA_MOE_ROUTER_BLOCKS=N` | max useful | CTA count for the split router score GEMV. |
| `PLOW_GEMMA_MOE_ROUTER_EXACT=1` | off | exact `MoeRouterGemmaScore` op instead of the default `…ScoreFast`. |
| `PLOW_GEMMA_MOE_TAIL_FUSE=1` | **off** ⚠️ | fuse the MoE-combine residual/norm tail (op72); **B=1 only, reorders summation** (not byte-identical). |
| `PLOW_KDA_FUSE` | **on** | collapse the KDA mixer chain 6→3 packets; `=0` emits the six-packet control. |
| `PLOW_KDA_FUSE_QKVG` | **on** | fuse KDA `q/k/v/g` projections into one 4-stream `GemvQkvg` (decode-only). |
| `PLOW_NS_MUL` | 1 / 2 (ctx>8k) | scale the CU-fill target for the flash-decode `nsplit` calc. |
| `PLOW_NS_ABS` / `PLOW_NS_FULL_ABS` | unset | pin `nsplit` absolutely / for full-attention layers only; both baked into the tuned-decode packet record (`tunedb`). |

---

## fp8 precision — full detail

fp8 is **not** a runtime toggle — it is baked when the model is compiled and the
interp cubins are built. The default build is the accuracy-safe **bf16** path;
fp8 is opt-in because it is *lossy*.

The precision flags are named by **axis**: `PLOW_W8A16` / `PLOW_W8A8` for
weights+activations, `PLOW_MXFP4` for the A4W4 (fp4+E8M0 microscale) path, and
`PLOW_KV_FP8` for the KV cache. `PLOW_FP8` and `PLOW_FP8_KV` remain as aliases.
Setting two weight flags is refused. The axes are independent and compose: bf16
weights with an fp8 KV is a legal combination and has its own object.

> **Not real flags** (comment-only axis labels, do not set them): `PLOW_W4A16`
> and `PLOW_MOE_ENC=`. The w4a16 / A4W4 encoding is realized by `PLOW_MXFP4=1`;
> the routed-expert encoding is *derived* from `PLOW_MXFP4`/`PLOW_FP8` (the
> `MoeEnc` enum in `crates/*/mla.rs`), not selected by a `PLOW_MOE_ENC` env var.

**The spelling differs by model family, because the kernels do.** An axis name
promises an encoding, so where a family cannot realize one it refuses rather than
resolving to the nearest thing it has:

| family | fp8 weights, bf16 acts | fp8 weights + fp8 acts | fp8 KV |
|---|---|---|---|
| dense (Gemma / Llama / Qwen) | `PLOW_W8A16=1` — **sm_120 only** | `PLOW_W8A8=1` — **the gfx950 recipe** | `PLOW_KV_FP8=1` |
| MLA+MoE (Kimi / GLM / DeepSeek) | `PLOW_W8A16=1` — works on gfx950 | **not implementable, refuses** | `PLOW_KV_FP8=1` |

The asymmetry is real, not an oversight. "fp8 weights, bf16 activations" is a
per-*channel* fp8 GEMM on the dense path, which gfx950 has no arm for, and a
*block*-fp8 GEMV on the MLA path, which it does. Same axis value, different
kernels, different coverage. The MLA family's expert ops (45/46/48/49) are w8a16
in **every** instantiation, so `PLOW_W8A8` there refuses and points at
`PLOW_W8A16` (fp8 weights) or `PLOW_MXFP4` (A4W4).

**On AMD (gfx950)** the fp8 profile is `PLOW_W8A8=1` (older spelling:
`PLOW_FP8=1 PLOW_W8A8=1`), plus `PLOW_KV_FP8=1` for an fp8 KV cache — plain
`PLOW_FP8=1` is refused at emit, deliberately: `PLOW_FP8=1` alone emits **w8a16**
(activation scale `t[3]` unbound), but the gfx950 `GEMM_FP8` arm is **w8a8** and
would fault on the null scale and misread A. plowc refuses, naming the fix, rather
than upgrading w8a16 → w8a8 on your behalf — every other emitter substitution is
computation-preserving, but this one quantizes the *activations* too and would
change a run's numerics under a flag set to mean something else. sm_120 has a real
w8a16 cubin and is unaffected.

**There is deliberately no separate activation flag.** w8a8 is one *profile*, not
a free cross-product with w8a16 — the kernels instantiate exactly those two. The
activation encoding is *derived* and reported in the manifest as
`precision.act_enc`, read off **`QuantFp8`'s presence** rather than inferred from
the weight flag (that inference is what once let a w8a16 packet reach a w8a8-only
object).

For a checkpoint that is already quantized — GLM-5.2-FP8, say — **no precision
flag is needed at all**: the encoding is read from `quantization_config`. Two
traps there: the key is **`dtype`, not `torch_dtype`** (HF renamed it), and
`dtype` reads `"bfloat16"` on an e4m3 checkpoint anyway (it describes the *compute*
dtype; storage dtype lives in `quantization_config`). The checkpoint wins over the
flags — it is a fact, they are a request — and a contradiction is refused.

The fp8 weight twins are keyed **verbatim, including the `fp8/` prefix** — the
emitter declares `fp8/<name>`, `quantize_fp8.py` writes `fp8/<name>`, and the
loader looks up `fp8/<name>`, with no stripping on either side.

**Keep `--arch` and `--gpu` in agreement** unless deliberately cross-compiling; a
mismatch now warns and is trusted toward the **GPU** (`--arch` records intent,
`--gpu` records what the packet was sized for).

Measured (gemma-4-31B, single-user): fp8 **decode beats vLLM-fp8** (−41% vs vLLM
bf16, parity-to-−3% vs vLLM fp8); fp8 **prefill** beats vLLM-bf16 at 32k and
closes the short-ctx gap to ~1.1–1.3× (still trails vLLM-fp8, which uses
cudagraphs + FA-class flash). `PLOW_FP8_KV` doubles the 31B concurrency ceiling.

---

## Interpreter compilation knobs (`nvcc -D…`)

Pass them through the build script, which forwards `PLOW_EXTRA_DEFINES` verbatim
to every object it compiles:

```bash
PLOW_EXTRA_DEFINES="-DPLOW_NV_W8A8=1 -DPGM_BN=64" scripts/build_sm120_cubin.sh <out.cubin>
PLOW_ROOT=$(pwd) …            # build a WORKTREE's sources; defaults to /root/plow
```

### The four configurations anyone actually builds

Pick a row, use the flags in it, and skip to the tables only if you are doing
kernel work. The **build** column is `nvcc -D…` / CMake; the **emit** column is
`plowc` env.

| you want | emit | build | notes |
|---|---|---|---|
| **Default** — validated, bf16 | `PLOW_UNISEG=1` | *(none)* | The shipped configuration. Everything below is a deviation from it. |
| **fp8 weights** — faster prefill GEMM | `PLOW_UNISEG=1 PLOW_W8A8=1` | `-DPLOW_NV_W8A8=1` | −48% GEMM, −30…34% prefill. Needs the fp8 weight twins. **Both flags or neither** — mismatch = `__trap()`. |
| **Long context, multi-user** — the fp8-KV path | `PLOW_UNISEG=1 PLOW_W8A8=1 PLOW_FP8_KV_FULL=1` | `-DPLOW_NV_W8A8=1 -DPLOW_FP8_KV=ON -DPLOW_FP8_KV_FASTPF=ON` | Halves KV bytes (B=8 at 127k fits 32 GB). `FASTPF=ON` is what keeps prefill on the fast PIPE=1 arm — **−21% prefill at 67k** vs leaving it off. ⚠️ **Lossy, and it degrades with context**: at 7.8k every arm retrieves a needle; at 66.9k **only bf16 does**. Greedy also diverges ~21 tokens. Validate retrieval at *your* context length. |
| **Legacy all-layer fp8 KV** | `… PLOW_FP8_KV=1` (no `_FULL`) | `-DPLOW_FP8_KV=ON` (**FASTPF off**) | e4m3 on every layer. `FASTPF` must stay OFF — the hd256 fp8 prefill op traps under PIPE=1. Slower prefill; prefer the mixed row above. |

### Two build/emit traps

- **An emit flag and its `-D` must agree.** Emitting `PLOW_W8A8=1` packets against
  a cubin built *without* `-DPLOW_NV_W8A8=1` hits `default: __trap()` and every
  launch dies with `CUDA_ERROR_LAUNCH_FAILED`. The failure looks like a driver
  problem and is not.
- **Decode and prefill are separate objects.** `-DPLOW_NV_PREFILL=1` builds the
  prefill object rather than stacking those arms onto the decode megakernel's
  budget. A flag marked *prefill only* below is a no-op in the decode object, and
  vice versa.

### Defaults deliberately NOT flipped, and why

| flag | measured | why still off |
|---|---|---|
| `PLOW_FP8_KV_FASTPF` | **−21%** prefill at 67k, byte-identical token stream | Legality depends on the *packet*, not the build: valid only for MIXED fp8-KV packets (`PLOW_FP8_KV_FULL=1`); all-layer packets trap under PIPE=1. A build cannot know which packet it will load. CMake emits a `WARNING` on the slow path. **If you emit mixed packets, turn it on.** |
| `PLOW_NV_PF_GEMV_HEAD` | −39% on prefill's `lm_head` | Traps on M≠1. Prefill emits `lm_head` at M=1 today, so *probably* safe to default, but not validated across every model family — and the failure is a hard launch failure. Worth ~0.5% at 127k. |
| `PLOW_NV_FA_FP8PV` | 1.40× on the hd512 flash op | Changes numerics, and only **+1.5%** end-to-end once `FASTPF` is on. |

Already defaulted **on** because they are bit-exact wins: `PGM_W8A8_LDS64` and
`PGM_SW8_V2` (px9, +2.2% weighted on the w8a8 GEMM).

### Object selection / model-family arms

These decide what is compiled in at all. The family arms buy **cubin size, smem
and stack frame** — not occupancy. Numbers below are `ptxas -v` on the megakernel
symbol, CUDA 13.0, and are **per-arch — they do not transfer**.

| flag | default | effect |
|---|---|---|
| `PLOW_NV_PREFILL` | 0 | build the PREFILL object (`interp_sm120_pf`) instead of decode. |
| `PLOW_NV_GEMMA` | 0 | Gemma arms (hd512 full-attn, GF flash). Off ⇒ byte-identical to a non-Gemma build. |
| `PLOW_NV_HOPPER` | off | sm_90a wgmma GEMM + Hopper attention arms instead of the sm_120 ones. |
| `PLOW_NV_MLA` | **1** | MLA (DeepSeek/Kimi/GLM). `0` trims the decode cubin — sm_90a −43% / regs 208→188; sm_120a −33% / regs 241→224. Occupancy-neutral. Not compiled into prefill. |
| `PLOW_NV_MAMBA` | **1** | Mamba/Nemotron arms. Owns the *prefill* stack frame (sm_120a 1024→0 B). Costs 2 regs on the sm_120a prefill object to turn off. |
| `PLOW_NV_DSA` | **1** | DeepSeek sparse-attention arms. Owns the *decode* smem (2192→1168 B). Costs 9 regs on the sm_120a decode object to turn off. Not compiled into prefill. |
| `PLOW_NV_GF8_TWIN` | 0 | co-linkable GF=8 full-attn decode twin (234 vs 209 regs); host picks per model. |
| `PLOW_NV_SEG_GEMM` | 0 | lean GEMM-segment object targeting occupancy 2, separate from the register/smem-hungry flash object. |
| `PLOW_NV_SEG_GEMM_BN64` | off | that object at BN=64. PX-7: ~1.05× end-to-end, not the ~2× the occupancy argument implies. |
| `PLOW_MLA_PREFILL` | off | compile the MLA chunked-prefill ops (51/55); also read at emit for the tune census. |
| `PLOW_MOE_PREFILL` | off | grouped MoE prefill ops (83–87). Also an emit gate: on for MoE bf16 by default, `=0` opts the bf16 MoE-prefill path out. |
| `PLOW_MOE_PF_A4W4` | off | A4W4 grouped-expert GEMM body (ops 85/86, MXFP4 on both operands); set for K3 rows in CMake, required by the devgen manifest. |
| `PLOW_K3_DECODE_GROUPED` | off | Build-only K3 override for a B1 object that must serve grouped ladder packets. Adds the A4W4 expert body and capability marker without changing `PLOW_DECODE_BATCH`; required by the K3 MI325X rung-1 recipe. |
| `PLOW_MOE_ROUTER_SELECT` | =`PLOW_K3` | `1` = k parallel block-max router passes (K3's 896-expert / top-16); `0` = single all-pairs rank pass. |
| `PLOW_BUCKET_DECODE` / `PLOW_BUCKET_PREFILL` | decode=1 selects | which interp bucket the object serves (`PLOW_BUCKET_PREFILL` is derived as `!DECODE`); emitted into manifest `req` strings. |
| `PLOW_BUCKET_FLASH` | off | compile the standalone 4-wave flash-decode object. |
| `PLOW_GLOBAL_QUEUE` | 0 (build); on at runtime | build the `_gq` global-queue objects; also a runtime env selecting GQ vs static (see serving knobs). |
| `PLOW_GQ_BATCH` | 1 | packets claimed per fetch-add in the global queue. **Must stay 1.** |
| `PLOW_PACKET_HASH` | emitted | the manifest emits `#define PLOW_PACKET_HASH 0x…`; the loader refuses a packet whose hash does not match the object. |

### Precision

| flag | default | effect |
|---|---|---|
| `PLOW_NV_W8A8` | 0 | PX-2 native w8a8 fp8 mainloop (BK64 + `Swizzle<3,4,3>` + `mma.sync.m16n8k32`). −48% GEMM vs bf16, −30…34% end-to-end prefill. Needs the fp8 weight twins **and** the matching emit flag. |
| `PLOW_FP8_KV` | 0 | e4m3 KV cache, half the KV bytes; per-(token, kv_head) f32 dequant scale. **Lossy** — ~3–6% logit relL2, greedy diverges after ~21 tokens. Lifts the 31B batch cap (B=4 → 7–8). **Read `PLOW_FP8_KV_FASTPF` before using this** — on its own it silently costs the PIPE=1 prefill pipeline. |
| `PLOW_FP8_KV_FASTPF` | **OFF** | ⚠️ **Without this, enabling fp8 KV makes prefill slower than not using it** (1670.3 ms vs bf16's 1315.6 ms at 7.8k). Turning on `PLOW_FP8_KV` alone forces prefill to `PLOW_NV_FA_PIPE=0` (fp8 dequants at the smem stage; cp.async cannot convert fp8 inline; traps under PIPE=1), losing the cp.async flash pipeline for prefill. `FASTPF=ON` keeps prefill on PIPE=1: **13843.3 → 10947.3 ms, −21% at 67k**. hd512-only; pair with mixed-KV packets (`PLOW_FP8_KV_FULL=1`). Decode is unaffected. |
| `PLOW_NV_FA_FP8MMA` | derived | feed the RAW e4m3 K tile to the mma — no dequant pass. Requires `PLOW_FP8_KV`. |
| `PLOW_NV_FP8_RB` | **1** | fp8 GEMV row-blocking. |
| `PLOW_FP8_FAST` | off | faster/looser fp8 conversion path. |

### Attention

`FA_PX4` and `FA_PIPE` carry the shipped long-context performance; the rest are
mostly A/B controls, several of which measured *negative* and are kept only so
nobody re-runs them.

| flag | default | effect |
|---|---|---|
| `PLOW_NV_FA_PX4` | **1** | restructured hd512 full-layer flash (register softmax + 8-warp QK). −24% flash op, −16% end-to-end 128k prefill. |
| `PLOW_NV_FA_PIPE` | **1** | cp.async KV-stream pipeline. Bit-identical logits; −16%@4k → **−81%@128k** prefill. **Forced to 0 for the prefill objects when `PLOW_FP8_KV` is on unless `PLOW_FP8_KV_FASTPF=ON`** — the single easiest way to lose most of plow's long-context prefill performance without noticing. |
| `PLOW_NV_FA_FP8PV` | 0 | **px8/px12 — the largest single prefill lever in the campaign.** e4m3 P·V via 8-bit `ldmatrix.trans`. **1.18× on a 127k prefill end-to-end** (32.39 → 27.59 s), 1.40× on the flash op. **Unreachable without `PLOW_FP8_KV_FASTPF=ON`** (`op_attention.cuh` `#error`s without PIPE=1). Also needs `-DPLOW_FP8_KV=1`. sm_120a-only. ⚠️ **NOT parity-preserving: greedy diverges at completion token 28.** Run a retrieval test before shipping. |
| `PLOW_NV_FA_GF` | 4 | head-group fold factor for flash-decode. Correctness needs `gqa % GF == 0` (checked at dispatch). Register allocation is the worst case over instantiations, so do not add more. |
| `PLOW_NV_FA_GF_FULL` | 4 (via `build_sm120_cubin.sh`; `CMakeLists.txt` says 2) | **CONTESTED — do not change without an end-to-end measurement.** PX-11 measured `=8` at 1.52× on the flash-decode OP and recommended it; PX-15 then found `=4` wins **all 8 cells at ctx ≥ 8k** (−29% at 130k, B=1) via grid fill (`n_grp = 16/GF`, so `=8` leaves 2 groups for 170 SMs). **Trust the end-to-end result.** |
| `PLOW_FP8_LD16`, `PLOW_FP8_FAST` | unset | **UNVALIDATED END-TO-END.** PX-11: 1.61× on the flash-decode op with `GF_FULL=8`, bit-exact, register-neutral — but `GF_FULL=8` is itself contradicted end-to-end, and PX-15 could not measure these flags (**every fp8-KV block asset dies in prefill with `CUDA_ERROR_LAUNCH_FAILED`**, a live bug with a reproducer). Nothing in the tree sets either flag; nothing should until the crash is chased down. |
| `PLOW_NV_FA_TMA` | 0 | TMA (`cp.async.bulk`) KV staging — ~2× *slower* on sm_120. A/B control. |
| `PLOW_NV_FA_KUN` | **1** | K-stream pre-issue depth for flash-decode; 1 = original consume-immediately loop. |
| `PLOW_NV_FA_WPR` | 0 | warp-per-row score phase (vs one-row-at-a-time). |
| `PLOW_NV_FA_WPR_RB` | 1 | rows a warp carries concurrently in that phase. |
| `PLOW_NV_FA_QGLOB` | 0 | read Q from global instead of staging to smem (WPR path). ⚠️ **`FA_WPR=1` + `FA_QGLOB=1` silently CORRUPTS the fp8/SZ arms** (PX-11): those arms read a never-written `qsm`. Non-default; **not fixed**. |
| `PLOW_NV_FA_REDBOUND` | 0 | bound the softmax reductions to the tile's LIVE rows. |
| `PLOW_NV_FA_VDBUF` | 0 | V double-buffer. **MEASURED NEGATIVE**: wash at 32k/64k, **+2.2% slower at 128k** — the 128k full-attn flash is HBM-bound. |
| `PLOW_NV_FA_CORRSKIP` | 0 | fp8mma only — skip the softmax rescale when every lane's `corr` is exactly 1.0. Bitwise identical. |
| `PLOW_NV_KVBOUNDS` | 0 | per-batch KV bounds checking. |
| `PLOW_NV_FA256_BKV` / `PLOW_NV_FA512_BKV` | 32 / 16 | KV rows staged per flash-**prefill** tile, per head dim. The trade is smem footprint vs staging granularity. **64 / 32 are the sm_90a measured optima** and are what `build_sm90a_cubin.sh` is driven with; they are not validated as sm_120a defaults. |
| `PLOW_FLASH_HD128` | off | enable the fused-write path in inline flash for D=128 (Llama/Qwen), avoiding register spill on the 8-wave interpreter. |
| `PLOW_FA_GF_FULL` | 2 | **AMD** flash-decode GQA fusion factor on full-attention layers (paired env+define; the NVIDIA analogue is `PLOW_NV_FA_GF_FULL`, which it must agree with or kernel/packet disagree). |

### Counter-gate, collective, MLA & geometry (`-D`)

Mostly AMD (gfx950) codegen/geometry arms. The gate and collective families are
dominated by **measurement instruments** — several are numerically wrong or admit
a data race by design and must never be in a shipped build; those are listed at
the end, not tabled.

| flag | default | effect |
|---|---|---|
| `PLOW_GATE_HIER` | 0 | gfx950 two-level counter-gate rendezvous. The CMake option is default-off and applies only to decode global-queue objects; it requires `PLOW_HSACO_GQ=ON` and `PLOW_L2_PLACE_DISPATCH=ON`. Passing it through global `PLOW_HSACO_EXTRA_DEFINES` is rejected. The gfx942 shell build's existing default is unchanged. |
| `PLOW_GATE_SC1` | 0 | device-scope (not system-scope) activation stores so the release fence can be elided; the counter-gate carries the ordering. |
| `PLOW_MLA_FOLD_MAP` / `_UN` / `_VEC` / `_VT` | 0 | fold the MLA up-projection map / output un-projection / V-cache load / V^T transpose into the adjacent kernel to save a launch + round-trip. |
| `PLOW_MLA_PF_MFMA` | 0 | MLA prefill uses MFMA matrix-core instructions for QK/PV instead of the vector-FMA fallback. |
| `PLOW_MLA_PF_WPM` | numeric | MLA-prefill waves-per-M-tile, clamped by `min(PLOW_WAVES, PLOW_MLA_PF_WPM)`. |
| `PLOW_XR_CUS` | 32 | **emit** — cap XReduce participant CUs (clamped 1..n_cu); a TP8 NUMA lever cutting L2 invalidates from idle WGs. |
| `PLOW_NO_XREDUCE` | unset | **emit** — disable all XReduce all-reduce collectives (diagnostic; numerically wrong). |
| `PLOW_WG_WAVES` (alias `PLOW_WAVES`) | 8 | waves per AMD workgroup (8×64 = 512 threads); feeds reduction/tiling geometry. |
| `PLOW_ATTNRES_MAXB` | 16 | compile-time hard cap on K3 attention-residual batch (`nb`); over it traps. |
| `PLOW_ATTNRES_RG` | 9 | rows per sweep of the K3 hidden axis (tuned so one sweep covers `nb_max=8`). |

**Measurement/diagnostic instruments — never ship** (numerically wrong or admit a
data race; used only to price a protocol cost): `PLOW_GATE_HIER_CEIL`,
`PLOW_GATE_NOINV`, `PLOW_GATE_RELAXSIG`, `PLOW_GATE_SC1_KEEPREL`, `PLOW_XR_ACQ_N`,
`PLOW_XR_NOSIG`, `PLOW_XR_NOWAIT`, `PLOW_XR_NOWAIT_RS`, `PLOW_XR_SHUFFLE`,
`PLOW_ACT_NT`, `PLOW_ACT_SCOPE_AGENT`, `PLOW_ICACHE_INV_PROBE`,
`PLOW_FLASH_MERGE_DSPLIT` (measured dead), `PLOW_NSTAGE` (`experiments/` only).

### GEMM / GEMV

| flag | default | effect |
|---|---|---|
| `PGM_BM` | 128 | GEMM M-tile height (`BM/WARPS_M` must be ×16). Overridable but no shipped alternate object. |
| `PGM_BN` | 128 | GEMM N-tile. `64` shrinks the plain arena to 45 KiB so the occ-2 segment object fits (driven to 64 by `PLOW_NV_SEG_GEMM_BN64`). |
| `PGM_BK` | 32 | GEMM K-tile depth (bf16 elts/step). Fixed, not overridable. |
| `PGM_STAGES` | 3 | GEMM cp.async pipeline depth. **px9 measured 3→6 = 0%** — the mainloop is not latency-bound. |
| `GV_UNROLL` | 8 (NV) / 11 (AMD); 14 for K3 | dense bf16 GEMV inner-K unroll = 128-bit weight vectors prefetched before consumed (memory-level parallelism). Swept by `tunedb`, per-arch in CMake. Bit-exact. |
| `GV_UNROLL_GLU` | 4 (NV) / 6 (AMD) | same unroll for the bf16 GLU/SwiGLU-fused GEMV. Also swept. |
| `GV_UN16` / `GV_UN_GLU16` | 4 / 2 | inner-K unroll for the **MM=16** decode rung specifically (`GV_UNROLL` covers the base rungs). Needs `GV_MM_MAX=16` to be reachable at all. Worth ~2%; `=8` measured best on sm_120a. |
| `GV_UN32` / `GV_UN_GLU32` | 2 / 1 | same for the MM=32 rung. |
| `GV_UNROLL_FP8` / `GV_UNROLL_GLU_FP8` | = the bf16 twins | rung unroll on the fp8 GEMV arms. The optimum is **precision-dependent** — sweep at the precision you ship, do not inherit the bf16 winner. |
| `GV_MOE_RB` / `GV_MOE_RB_DN` / `GV_MOE_UN` | 2 / 2 / 2 | MoE GEMV output-channels-per-warp (main / `down` arm) and inner unroll. Shape-dependent; the source notes the previous optimum stopped being one once neighbouring arms moved. |
| `PGM_GLU_STAGES` | 2 | same for the fused GLU arm (kept shallower to fit the 100 KiB dynamic-smem cap). |
| `PGM_W8A8_LDS64` | **1** | **px9** — read the fp8 fragment as one `uint2`. **+6.5% on plain w8a8, 0% on GLU, +2.2% weighted.** Bit-exact. |
| `PGM_SW8_V2` | **1** | **px9** — `Swizzle<2,4,2>` matched to the ACTUAL 64-byte fp8 row. +0.5% on top of LDS64. |
| `PGM_SW8_OFF` | unset | A/B control: make `pgm_sw8` the identity. px9: removing the swizzle is +3.1% cycles/QMMA — a diagnostic, not a win. |
| `PLOW_NV_GEMV_RB` | 0 | MoE GEMV row-blocking. Off keeps every sm_120 object byte-identical; **the sm_90a build sets it to 1**. |
| `PLOW_NV_PF_GEMV_HEAD` | 0 | run prefill's `lm_head` on the M=1 GEMV arm. **1.991 → 1.213 ms, −39%.** Prefill only; traps on M≠1. |
| `GV_MM_MAX` | **8** | Widest `gemv_*_rows<MM>` rung for batched decode: batch `B` costs `ceil(B/GV_MM_MAX)` weight passes. **Match it to the batch you actually serve — mismatched, expensive both directions.** `=16` costs 1.1% at B=1 and 17% at B=8 to buy 34% at B=16. PX-10: an asset built `=16` and served at B=8 loses 19.4% at 131k. Pin `=16` only if you pin B≥16. |
| `PLOW_NV_GEMV_LS` | 0 | GEMV row-blocking. Wins in isolation (qkv 1.43×) but **loses in the megakernel**. Compiled out, intact. |
| `PLOW_NV_GEMV_NOSTAGE` | 0 | skip GEMV smem staging. |
| `PLOW_NV_GEMV_STAGE_MINROWS` | 16 | row threshold below which staging is skipped. |
| `PLOW_NV_RB_QKV`, `PLOW_NV_RB_LMHEAD`, `PLOW_NV_RB_GEMV` | 0 | per-op row-blocking A/B controls. |
| `PLOW_NV_SZ` | 0 | **experimental** lossless bf16 weight decompression (SplitZip GEMV twins). Bit-exact, measured non-viable; kept as an A/B reference. (`PLOW_NV_ZG` does not exist.) |

### MoE

| flag | default | effect |
|---|---|---|
| `PLOW_MOE_XN_BF16` | 0 | bf16 expert-N staging buffer. |
| `PLOW_MOE_XN_MAX` | 2816 | expert-N staging cap. |
| `PLOW_MOE_DOWN_SG` | 4 | subgroup count for the expert `down` arm. |
| `PLOW_MOE_DOWN_LANESPLIT`, `PLOW_MOE_DOWN_STAGE_FU` | 0 | `down` lane-split / staged fixups. |
| `PLOW_MOE_ROUTER_WIDE` | 0 | wide router arm. |
| `PLOW_MOE_COMBINE_ALLBLK` | 0 | all-block combine. |

### Scheduling, sync, occupancy

| flag | default | effect |
|---|---|---|
| `PLOW_NV_SCHED` | **1** | global-queue scheduler; `0` = static per-block streams (build-time A/B). Counter protocol byte-identical across both. |
| `PLOW_NV_SEGMENTS` | 0 | host relaunches once per segment (the AMD model) instead of one cooperative launch. |
| `PLOW_NV_PTXSYNC` | **1** | inline-PTX counter gate (`red` instead of a result-bus round trip). |
| `PLOW_NV_GATE_SLEEP` | 64 | backoff (ns) inside the counter-gate poll; `0` spins flat out. |
| `PLOW_NV_LEAN_DECODE` | 0 | drop arms owning the decode object's 208-reg / 1-blk-SM ceiling so ptxas + `PLOW_NV_FORCE_MINBLK` can reach 2–3 blk/SM. |
| `PLOW_NV_FORCE_MINBLK` | off | force a `__launch_bounds__` min-blocks-per-SM. |
| `PLOW_NV_THREADS` | 256 | NVIDIA block size (`op_attention.cuh`). Raising it is the precondition for BQ=64 flash tiling. Distinct from the AMD `PLOW_THREADS` (512 = 8 waves × 64); not a rename. |
| `PLOW_NV_EMBED_SMEM` | 0 | embed the object's smem requirement so `serve` reads it instead of guessing the GF=2 default. |
| `PLOW_L2_PLACE_DISPATCH` | off | L2 placement dispatch (alias: `PLOW_NV_PLACE_DISPATCH`). Vendor-neutral — GPC on NVIDIA, XCD on AMD. |

### Measurement-only — never ship a build with these

All produce wrong logits by construction.

| flag | default | effect |
|---|---|---|
| `PLOW_NV_SKELETON` | 0 | run gates + signals with no op bodies: the interpreter's dispatch floor. Garbage logits. |
| `PLOW_NV_SKEL_PAD` | 160 | padding for that skeleton. |
| `PLOW_NV_ABLATE_LO`, `PLOW_NV_ABLATE_HI` | 0 | 128-bit opcode mask — skip those ops' BODIES, keep every gate/signal. Garbage logits. |
| `PLOW_NV_FA_FP8ABL` | 0 | flash fp8 ablation bitmask. **Never set on a shipped build.** |
| `PLOW_NV_TRACE` | 0 | per-op `gate`/`body`/`signal` cycle trace. **Read the SHAPE, not the absolute total.** |

Harness-only (not the served cubins): `PLOW_SM120_SMS` (188) and
`PLOW_SMP_THREADS` (256).

### sm_90a object selection (`PLOW_BUILD_*`) — **sm_90a only**

`scripts/build_sm90a_cubin.sh` compiles a **five-object** prefill stack rather
than one megakernel, and these envs — read by the build script, not passed as
raw `-D` — decide which objects it emits and which arms each carries. They have
no effect on any other `--arch`; the sm_120 script builds one decode + one
prefill object and ignores them.

The reason the stack exists is the register-allocation coupling that the section
header above describes: a heavyweight wgmma body loses probe-grade allocation
when compiled into the wide-armed interpreter TU, so on sm_90a the tuning axis
is **which object gets built**, not which tile a macro selects.

| build env | object it shapes | meaning |
|---|---|---|
| `PLOW_BUILD_SEG` | `_pfseg` + `_pfgemm` | build the segmented pair at all. Packets must be emitted **without** `PLOW_UNISEG`. |
| `PLOW_BUILD_FATLITE` | `_pfseg` | the fat object arm-stripped of flash → 128 regs, occupancy 2. |
| `PLOW_BUILD_GEMM_WS384` | `_pfgemm` | 384-thread producer/consumer GEMM; carries **both** precisions' n256 bodies, so one lean object serves bf16 and fp8. |
| `PLOW_BUILD_FA512` + `PLOW_BUILD_FA_WG` + `PLOW_BUILD_FA_HD256` | `_pffa` | the dedicated flash object: wgmma arms, hd512 and hd256. `--pf-seg-fa512 all` **requires** `FA_HD256=1` (the loader refuses the mismatch). |
| `PLOW_BUILD_TMA_GEMM` / `PLOW_BUILD_W8A8` | all | TMA GEMM bodies / fp8 w8a8 arms. Drop `W8A8` for bf16-only cubins. |

The canonical build measured in the GH200 campaign:

```bash
PLOW_EXTRA_DEFINES="-DPLOW_NV_FA256_BKV=64 -DPLOW_NV_FA512_BKV=32" \
PLOW_BUILD_TMA_GEMM=1 PLOW_BUILD_W8A8=1 PLOW_BUILD_SEG=1 \
PLOW_BUILD_FATLITE=1 PLOW_BUILD_GEMM_WS384=1 \
PLOW_BUILD_FA512=1 PLOW_BUILD_FA_WG=1 PLOW_BUILD_FA_HD256=1 \
scripts/build_sm90a_cubin.sh <out-dir>/interp_sm90a.cubin
```

Its serve-side and emit-side counterparts are in
[Segmented prefill (sm_90a / GH200)](#segmented-prefill-sm_90a--gh200) below;
the emit knob and the serve knob **must pair** (`PLOW_SEG_PURE_GEMM` ↔
`--pf-seg-pure`, `PLOW_SEG_FA512` ↔ `--pf-seg-fa512`) — the classing decides
which object a packet lands on and a mismatched object `__trap()`s by design.

Ablation-only `PLOW_BUILD_*` switches, default off, leave them off unless
reproducing a specific finding: `FA_ROPE`, `FA_WGITEM`, `FP8KV`, `GEMM_OCC1`,
`GEMM_ONLY`, `GEMM_UNI256`, `GEMV_HEAD`, `SEG_NOGLU`, `SEG_WS`, `SEG_WS_ENTRY`,
`WS_BN256`.

Two sm_90a-only `-D` tuning knobs on the GEMM side:

| flag | default | effect |
|---|---|---|
| `PGM90_TILE_BAND` | 16 | band rasterization width for the sm_90a GEMM — how many M-tiles share a B-tile in L2 before the walk advances. |
| `PGM90_UNI256_NS` | 4 | TMA ring depth of the n256 body. bf16 256-byte k-stages at `NS=2` measured **−44 ms** (ring starvation); do not lower it casually. |

The runtime companion of an sm_90a decode build is `PLOW_NS_FULL_ABS`, and its
value is a **cliff, not a slope** — see the header of `build_sm90a_cubin.sh`,
which derives it as `n_cu / gcd(n_grp, n_cu)` and records what the neighbouring
values cost.

---

## Serving / runtime knobs (`plowrt` env)

| var | default | effect |
|---|---|---|
| `PLOW_PF_BATCH=1` | off | Cross-request prefill policy. CUDA packs waiting chunks into one launch (+27% saturated multi-user throughput; inert on fp8-KV and ignored under `PLOW_VMM_PREFIX=1`). AMD round-robins serialized chunks across pending slots; it does not co-pack because packet rows currently carry one sequence's KV and recurrent-state view. |
| `PLOW_PF_INTERLEAVE=N` | 2048 | **CUDA + AMD TP chunked-prefill quantum — the default path, not a `PLOW_PF_BATCH` knob.** Once any slot is decoding, a tick admits at most `N` prefill rows, then runs decode. AMD selects an existing packet rung at or below `N` and still handles one request per tick. `0` = uncapped. It can only clamp below the emitted ladder. |
| `PLOW_PF_CHUNK=C` | 0 (off) | **Experimental, CUDA + AMD TP serving** per-request prefill chunk-row cap. AMD selects only compiled packet rungs at or below `C` (for example, K3 8192 with `C=4096` plans 4096+4096); it does not pack chunks across requests. The ~10% B=8 regression was measured on CUDA; AMD remains unmeasured. Off preserves the existing plan. |
| `PLOW_PF_CHUNK_COST=R` | 512 | cost of ONE prefill launch in padded-row equivalents (`rows + R × launches`). A launch re-streams every layer's weights: measured `ttft_ms = 0.112·rows + 60.1·chunks`, i.e. **60 ms ≈ 537 rows**. `0` = pure-minimum-padding. |
| `PLOW_PF_COVER=1` | off | restore the covering-bucket prefill policy (exact-parity A/B vs the cost-aware default). |
| `PLOW_PF_DEFER_DECODE=1` | off | **CUDA + AMD TP throughput mode — trades streaming latency for aggregate tok/s.** While pending prefill remains, skips decode so later decode ticks run at full batch. A completed prefill may emit its first token, but no request advances decode until admitted prefill drains. The 8×127k **+7.1% out tok/s** result was measured on CUDA; AMD is unmeasured. Wrong as an interactive default. |
| `PLOW_PF_PACKLOG=1` | off | per-launch pack diagnostics. |
| `PLOW_PF_NO_CHUNK=1` | off | restore whole-prompt-per-tick (disable chunked prefill). |
| `PLOW_PF_NO_INTERLEAVE=1` | off | restore a prefill-only tick (disable prefill/decode interleave). |
| `PLOW_VMM_PREFIX=1` | off | VMM-backed KV prefix cache — a new request attaches blocks a previous one built. Only FULL-attention layers are VMM-backed; reuse is block-granular. 12B: warm TTFT 3.6× (4k) → **23.8× (128k)**; cold ~0.2–1.5% above 8k. Incompatible with `PLOW_PF_BATCH=1`. |
| `PLOW_VMM_BLOCK_MIB=M` | 2 | VMM sharing block size. 2 MiB ≈ 4096 tokens at hd256 bf16. Raise (e.g. 64) for 128k-dedup work. |
| `PLOW_VMM_CACHE_MIB=M` | 0 | cap on retained (unreferenced) VMM blocks; `0` = no cache. |
| `PLOW_VMM_KV=1` | off | **AMD** — VMM-backed KV on ROCr (`hsa_amd_vmem_*`); warns and falls back if the platform can't support it. |
| `PLOW_PREFIX_CACHE=1` | off | enable the TP-only prefix cache. |
| `PLOW_NV_SCHED=1` | **on** | global-queue interpreter scheduler; the static per-block-stream path is the build-time A/B. |
| `PLOW_GLOBAL_QUEUE=0` | on | force the static per-block-stream scheduler (AMD runtime read; build-time A/B otherwise). |
| `PLOW_STATIC` / `PLOW_STATIC_DECODE` / `PLOW_STATIC_PREFILL` | off | force the static scheduler for both phases / decode only / prefill only. |
| `PLOW_SEG_WINDOW` | on | AMD segment enqueue/drain windowing (A/B; `=0` off). |
| `PLOW_MULTISTEP=K` | 8 (K∈[2,64]) | bounded device multi-step decode (K steps/launch); needs the dynamic-kvrow decode cubin + sampler. `0`/`1` opts out. |
| `PLOW_LAUNCH_ROWS=N` | `LAUNCH_ROWS` | override the prefill pad/launch-rows tradeoff. |
| `PLOW_PREFETCH=N` | 256 | checkpoint prefetch depth in tensors. `PLOW_PREFETCH_THREADS=N` (16) sets prefetch threads/rank; `0` disables prefetch. |
| `PLOW_WEIGHT_SLAB` | on | single-allocation weight slab; `=0` turns it off (both backends). |
| `PLOW_UPLOAD_SLOTS=N` | 4 | AMD upload-ring pipeline depth; `1` = pre-pipeline one-slab shape. |
| `PLOW_SHARE_CKPT` | on | shared (vs per-rank) checkpoint mapping across TP ranks; `=0` restores per-rank. |
| `PLOW_VRAM_BUDGET_MIB=M` | unset | cap the ModelManager VRAM budget (MiB). |
| `PLOW_WEIGHT_VMM` | CUDA on, AMD off | VMM (reserve+map) weight slab; `=0` falls back to one flat allocation (`=1` opts AMD in). |
| `PLOW_SLAB_KEEP` | multi-model on | park evicted models' 256 MiB slab chunks in a per-device pool for the next load; `=0` releases them (`=1` forces on for single-model). |
| `PLOW_KV_POOL_MIB=N` | 512 | per-engine KV physical-block reuse pool cap (MiB); `0` disables pooling. |
| `PLOW_DRAIN_TIMEOUT_MS=N` | unset (unbounded) | S1 switch drain deadline; past it the victim's live generations are preempted (`Preempted` finish, queued jobs 429). `0` preempts immediately. |
| `PLOW_PRELOAD` | on | speculative next-model preload after an S1 switch; `=0` disables. |
| `PLOW_TP_AGREE_EVERY=N` | 1 | TP cross-rank agreement interval. `PLOW_TP_NO_AUDIT=1` disables the redundant-rank audit (timing runs); `PLOW_TP_SERIAL_LOAD=1` restores one-at-a-time per-rank load. |
| `PLOW_LOAD_PROFILE=1` | off | split upload wall time into alloc / stage+DMA profiling. |
| `PLOW_STEP_TIME=1`, `PLOW_TTFT_LOG=1` | off | per-decode-step host-op timing / TTFT breakdown logging (diagnostics). |
| `PLOW_HSACO_LOWRUNG=dir:max[,dir:max…]` | unset | AMD decode-object tiers. The runtime selects the narrowest tier whose `max` covers the occupied decode rung, pairing-checks each tier at that width, and falls back to the primary HSACO inventory above it. A single legacy `dir` uses `PLOW_LOWRUNG_MAX` (default 2). |
| `PLOW_STATE_CLEAR_DEVICE=1` | off | AMD admission experiment: clear slot-major recurrent state with one device kernel per rank instead of host-staged SDMA fills. Requires rebuilt decode objects carrying `plow_state_clear`. |
| `PLOW_SEG_PACKED_PREFILL=1` | off | AMD emit-time experiment: split descriptor-consuming MLA norm/cache, MLA flash, and serial-KDA ops into pure topological segments. No model-name predicates. Unset preserves packet bytes. |
| `PLOW_PACKED_PREFILL_ROUTE=1` | off | Load the optional lean packed-family HSACO objects and permit exact-family routing after metadata is staged. Missing/wrong markers and mixed segments refuse; it does not enable mux co-packing by itself. |

### Segmented prefill (sm_90a / GH200)

The five-object prefill stack from the GH200 campaign
(`perf-data/gemma12b-gh200-prefill-campaign.md`). The classing knobs are
**serve-side mirrors of emit-side knobs and must match the blob** — `--pf-seg-pure`
pairs with `plowc`'s `PLOW_SEG_PURE_GEMM`, `--pf-seg-fa512` with `PLOW_SEG_FA512`.
A mismatch is a wrong-object launch, i.e. a device trap, not a slowdown.

| flag (env) | default | effect |
|---|---|---|
| `--pf-seg-dir` (`PLOW_PF_SEG_DIR`) | unset | dir holding `interp_sm90a_pfseg/_pfgemm[/_pffa].cubin`. Unset = single-object prefill. Packets must be emitted **without** `PLOW_UNISEG`. |
| `--pf-seg-pure` (`PLOW_PF_SEG_PURE`) | unset | segment classing: `1` = every plain tiled GEMM is GEMM-class, `fp8` = only TMA-mapped fp8 GEMMs (the ws-entry object's sole arm). |
| `--pf-seg-fa512` (`PLOW_PF_SEG_FA512`) | unset | hd512 flash on the dedicated `_pffa` object: `1` = hd512 only, `all` = both head dims — `all` **requires** an object built `PLOW_BUILD_FA_HD256=1`; the loader refuses the mismatch rather than trapping. |
| `--pf-seg-graph` (`PLOW_PF_SEG_GRAPH`) | off | submit each chunk's whole segment chain as ONE CUDA graph (T35). |
| `--pf-seg-eqsmem` (`PLOW_PF_SEG_EQSMEM`) | off | launch every object with the same dynamic-smem request (avoids per-launch carveout reconfig). |
| `--pf-seg-v2` (`PLOW_PF_SEG_V2`) | unset | classing v2 (`1`) / q8 variant (`q8`). |
| `--pf-seg-time`, `--pf-seg-fatonly`, `--pf-seg-noncoop` | off | diagnostics: per-class event timing / every segment on the fat object / plain (non-cooperative) launches. |

The canonical serving configuration measured in the campaign:

```bash
plowrt serve --assets <dir> \
  --pf-seg-dir <cubins> --pf-seg-pure fp8 --pf-seg-fa512 all --pf-seg-graph
```

Loader/asset overrides: `PLOW_NV_CUBIN[_PF]`, `PLOW_NV_KERNEL[_PF]`, `PLOW_NV_SMEM`
/ `PLOW_NV_SMEM_PF` (override decode/prefill dynamic-smem arena bytes), `PLOW_HSACO`
(AMD `.hsaco` dir), `PLOW_CHECKPOINT`, `PLOW_LIBCUDA`.

### What plow does and does not fuse

Three things are easily conflated:

1. **Chunked prefill** — shipped, on by default, no flag (`PLOW_PF_INTERLEAVE`). A
   long prompt is admitted a chunk at a time so live decode streams are not
   stalled for a whole prompt.
2. **Cross-request prefill packing** — `PLOW_PF_BATCH=1`, off by default. Several
   *waiting requests'* prefill chunks share one launch.
3. **Mixed batching (prefill ⊕ decode in one launch)** — **not implemented**, and
   not hidden behind a flag. A tick that does both runs two launches, each
   re-reading the full weight set (~12 GiB fp8 on the 12B asset, ~9 ms). vLLM's
   chunked prefill carries the decode rows in the same forward pass.

The gap in (3) is bounded by one weight read per tick: ~12% of a tick at 2k
prompts, but only **~0.6% at 127k**. A short-context / high-QPS lever, not a
long-context one.
</content>
</invoke>
